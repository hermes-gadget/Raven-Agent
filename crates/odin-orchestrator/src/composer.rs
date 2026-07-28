//! Composer — the user-facing orchestrator agent.
//!
//! The Composer is the central intelligence of Raven Agent. It:
//! - Receives user intent (one or more requests in one message)
//! - Decomposes into a task graph
//! - Detects independent vs. dependent workstreams
//! - Spawns scoped sub-agents with appropriate tools/files/permissions
//! - Manages file lock acquisition and queueing
//! - Tracks progress and lifecycle of all sub-agents
//! - Handles interruptions (pause, cancel, redirect, reprioritize)
//! - Merges parallel results into one coherent response
//!
//! Default behavior: multi-agent orchestration. One user message → auto-split.

use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use odin_core::traits::Provider;
use odin_core::types::{CompletionOptions, Message, MessageContent, Role};

use crate::file_lock::FileLockManager;
use crate::lifecycle::{AgentLifecycle, AgentPhase};
use crate::merge::{MergeResolver, MergeStrategy, SubAgentResult};
use crate::progress::{ProgressTracker, WorkstreamStatus};
use crate::sub_agent::{SubAgent, SubAgentConfig, SubAgentConfigBuilder};
use crate::task_graph::{TaskGraph, TaskGraphStatus, TaskNode, TaskNodeStatus};

/// Result from the composer — the merged output of all sub-agents.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComposerResult {
    /// The original user goal.
    pub goal: String,
    /// Whether the overall goal succeeded.
    pub success: bool,
    /// Human-readable summary.
    pub summary: String,
    /// Per-agent results.
    pub agent_results: Vec<SubAgentResult>,
    /// Files modified.
    pub modified_files: Vec<String>,
    /// Any merge conflicts requiring user input.
    pub conflicts: Vec<crate::merge::FileConflict>,
    /// Total duration in milliseconds.
    pub duration_ms: u64,
    /// Number of sub-agents spawned.
    pub agent_count: usize,
}

/// Configuration for the Composer.
#[derive(Debug, Clone)]
pub struct ComposerConfig {
    /// Maximum sub-agents to run in parallel.
    pub max_parallel: usize,
    /// Default max iterations per sub-agent.
    pub default_max_iterations: u32,
    /// Whether to auto-merge results without user review.
    pub auto_merge: bool,
    /// Merge strategy for parallel results.
    pub merge_strategy: MergeStrategy,
    /// Workspace root for file operations.
    pub workspace_root: String,
    /// Whether to persist orchestration state to SQLite.
    pub persist_state: bool,
    /// Max retries for failed sub-agents (0 = no retry).
    pub max_retries: u32,
}

impl Default for ComposerConfig {
    fn default() -> Self {
        Self {
            max_parallel: 10,
            default_max_iterations: 50,
            auto_merge: true,
            merge_strategy: MergeStrategy::Auto,
            workspace_root: ".".to_string(),
            persist_state: true,
            max_retries: 1,
        }
    }
}

/// The Composer — main orchestration engine.
pub struct Composer {
    /// Configuration.
    config: ComposerConfig,
    /// File lock manager for concurrent access.
    file_locks: Arc<FileLockManager>,
    /// Merge resolver for combining results.
    merge_resolver: MergeResolver,
    /// Progress tracker for status reporting.
    progress: ProgressTracker,
    /// Active task graphs.
    graphs: HashMap<String, TaskGraph>,
    /// Active sub-agents: agent_id → (SubAgent, AgentLifecycle)
    agents: HashMap<Uuid, (SubAgent, AgentLifecycle)>,
    /// Workstreams: label → Vec<agent_id>
    workstreams: HashMap<String, Vec<Uuid>>,
}

impl Default for Composer {
    fn default() -> Self {
        Self::new(ComposerConfig::default())
    }
}

impl Composer {
    /// Create a new Composer.
    pub fn new(config: ComposerConfig) -> Self {
        Self {
            config,
            file_locks: Arc::new(FileLockManager::new()),
            merge_resolver: MergeResolver::new(),
            progress: ProgressTracker::new(),
            graphs: HashMap::new(),
            agents: HashMap::new(),
            workstreams: HashMap::new(),
        }
    }

    /// Create a Composer with a custom FileLockManager.
    pub fn with_file_locks(mut self, locks: Arc<FileLockManager>) -> Self {
        self.file_locks = locks;
        self
    }

    // ── Intent Intake & Decomposition ───────────────────────────

    /// Process a user message. This is the main entry point.
    ///
    /// The composer analyzes the message, detects if it contains multiple
    /// independent requests, creates a task graph, and schedules work.
    pub fn intake(&mut self, goal: &str) -> &TaskGraph {
        let graph = self.decompose(goal);
        let root = graph.root_goal.clone();
        self.graphs.insert(root.clone(), graph);
        self.graphs.get(&root).unwrap()
    }

    /// Decompose a goal into a task graph.
    ///
    /// This synchronous path uses heuristic splitting. Call
    /// [`decompose_with_llm`](Self::decompose_with_llm) for provider-guided
    /// decomposition with heuristic fallback.
    fn decompose(&self, goal: &str) -> TaskGraph {
        let mut graph = TaskGraph::new(goal.to_string());

        // Heuristic: split on "and", ";", or numbered items
        let sub_goals = self.split_goal(goal);

        if sub_goals.len() <= 1 {
            // Single task — create one node
            graph.add_node(TaskNode {
                id: Uuid::new_v4(),
                label: "main".to_string(),
                goal: goal.to_string(),
                read_files: vec![],
                write_files: vec![],
                required_capabilities: default_capabilities_for_goal(goal),
                priority: 0,
                status: TaskNodeStatus::Pending,
                result: None,
                agent_id: None,
            });
        } else {
            // Multiple sub-goals — they're independent (no dependencies)
            for (i, sg) in sub_goals.iter().enumerate() {
                let label = format!("task-{}", i + 1);
                graph.add_node(TaskNode {
                    id: Uuid::new_v4(),
                    label,
                    goal: sg.clone(),
                    read_files: vec![],
                    write_files: vec![],
                    required_capabilities: default_capabilities_for_goal(sg),
                    priority: i as u32,
                    status: TaskNodeStatus::Pending,
                    result: None,
                    agent_id: None,
                });
            }
        }

        graph.status = TaskGraphStatus::Running;
        graph
    }

    /// Heuristic goal splitting.
    fn split_goal(&self, goal: &str) -> Vec<String> {
        // Split on comma or semicolon first
        let parts: Vec<&str> = goal.split_inclusive(&[',', ';'][..]).collect();

        if parts.len() > 1 {
            parts
                .iter()
                .map(|s| s.trim().trim_matches(&[',', ';'][..]).trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        } else {
            // Try splitting on " and " as a fallback
            let parts: Vec<&str> = goal.split(" and ").collect();
            if parts.len() > 1 {
                parts.iter().map(|s| s.trim().to_string()).collect()
            } else {
                vec![goal.to_string()]
            }
        }
    }

    // ── LLM-Based Decomposition ─────────────────────────────────────

    /// Decompose a goal using an LLM planning model.
    ///
    /// Sends a structured prompt to the provider asking it to break down
    /// the goal into sub-tasks with dependencies, likely files, required
    /// capabilities, risk assessment, and execution order. If the LLM is
    /// unavailable or fails, falls back to heuristic decomposition.
    pub async fn decompose_with_llm(
        &self,
        goal: &str,
        provider: &dyn Provider,
        model: &str,
    ) -> TaskGraph {
        // Build a structured decomposition prompt
        let system_prompt = "You are a task-decomposition planner for an AI agent system.\n\
            Break down the user's goal into independent sub-tasks.\n\
            Respond ONLY with valid JSON — no markdown, no explanation.\n\
            The JSON must follow this schema:\n\
            {\n\
              \"tasks\": [\n\
                {\n\
                  \"id\": \"task-1\",\n\
                  \"goal\": \"Description of what this task does\",\n\
                  \"reads\": [\"file/path\"],\n\
                  \"writes\": [\"file/path\"],\n\
                  \"capabilities\": [\"read\", \"write\", \"shell\"],\n\
                  \"depends_on\": [],\n\
                  \"risk\": \"safe|dangerous\",\n\
                  \"priority\": 0\n\
                }\n\
              ]\n\
            }\n\
            Rules:\n\
            - Split ONLY if there are genuinely independent sub-tasks.\n\
            - Tasks in the same workstream share files; different workstreams don't.\n\
            - Give each task a unique id like \"task-1\", \"task-2\".\n\
            - Use `depends_on` to list task ids this task must wait for.\n\
            - `risk`: \"safe\" for read-only, \"dangerous\" for writes or shell.\n\
            - `priority`: 0=first, higher numbers run later.\n\
            - For a simple single request, return ONE task.";

        let user_message = format!("Decompose this goal: {}", goal);

        let messages = vec![
            Message {
                role: Role::System,
                content: MessageContent::Text {
                    content: system_prompt.to_string(),
                },
                name: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::Text {
                    content: user_message,
                },
                name: None,
                tool_call_id: None,
            },
        ];

        let options = CompletionOptions {
            temperature: Some(0.2),
            max_tokens: Some(1024),
            ..Default::default()
        };

        match provider.chat(model, &messages, &[], &options).await {
            Ok(response) => match self.parse_llm_decomposition(&response.message, goal) {
                Ok(graph) => {
                    tracing::info!(
                        "[COMPOSER] LLM decomposition: {} tasks from {}",
                        graph.nodes.len(),
                        provider.name()
                    );
                    graph
                }
                Err(e) => {
                    tracing::warn!(
                        "[COMPOSER] LLM decomposition parse failed: {}, falling back to heuristic",
                        e
                    );
                    self.decompose(goal)
                }
            },
            Err(e) => {
                tracing::warn!(
                    "[COMPOSER] LLM decomposition failed: {}, falling back to heuristic",
                    e
                );
                self.decompose(goal)
            }
        }
    }

    /// Parse the LLM's JSON decomposition response into a TaskGraph.
    fn parse_llm_decomposition(&self, message: &Message, goal: &str) -> Result<TaskGraph, String> {
        let text = match &message.content {
            MessageContent::Text { content } => content.clone(),
            _ => return Err("Expected text response".into()),
        };

        // Try to extract JSON from the response (may be wrapped in markdown)
        let json_str = if let Some(start) = text.find("```json") {
            let inner = &text[start + 7..];
            if let Some(end) = inner.find("```") {
                &inner[..end]
            } else {
                &text[start + 7..]
            }
        } else if let Some(start) = text.find('{') {
            &text[start..]
        } else {
            return Err("No JSON found in response".into());
        };

        let parsed: serde_json::Value = serde_json::from_str(json_str.trim())
            .map_err(|e| format!("JSON parse error: {}", e))?;

        let tasks_arr = parsed["tasks"].as_array().ok_or("Missing 'tasks' array")?;

        if tasks_arr.is_empty() {
            return Err("Empty tasks array".into());
        }

        let mut graph = TaskGraph::new(goal.to_string());
        let mut id_map: HashMap<String, Uuid> = HashMap::new();

        // First pass: create all nodes
        for task in tasks_arr {
            let task_id = task["id"].as_str().unwrap_or("unknown").to_string();
            let task_goal = task["goal"]
                .as_str()
                .unwrap_or("No description")
                .to_string();
            let reads: Vec<String> = task["reads"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let writes: Vec<String> = task["writes"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let capabilities: Vec<String> = task["capabilities"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let priority = task["priority"].as_u64().unwrap_or(0) as u32;

            let node_id = Uuid::new_v4();
            id_map.insert(task_id.clone(), node_id);

            graph.add_node(TaskNode {
                id: node_id,
                label: task_id,
                goal: task_goal,
                read_files: reads,
                write_files: writes,
                required_capabilities: capabilities,
                priority,
                status: TaskNodeStatus::Pending,
                result: None,
                agent_id: None,
            });
        }

        // Second pass: add dependencies
        for task in tasks_arr {
            let task_id = task["id"].as_str().unwrap_or("unknown");
            let deps_arr = task["depends_on"].as_array();
            let node_id = id_map.get(task_id).copied();

            if let (Some(deps), Some(node_id)) = (deps_arr, node_id) {
                for dep_id in deps {
                    let dep_id_str = dep_id.as_str().unwrap_or("");
                    if let Some(&dep_node_id) = id_map.get(dep_id_str) {
                        graph.add_edge(dep_node_id, node_id);
                    }
                }
            }
        }

        graph.status = TaskGraphStatus::Running;
        Ok(graph)
    }

    // ── Sub-Agent Management ────────────────────────────────────

    /// Create a sub-agent config for a task graph node.
    ///
    /// Tools are scoped from `required_capabilities` (or a default agent toolkit
    /// when capabilities are empty) so sub-agents do not receive the full registry.
    pub fn create_sub_agent(&self, node: &TaskNode) -> SubAgentConfig {
        let capabilities = if node.required_capabilities.is_empty() {
            default_capabilities_for_goal(&node.goal)
        } else {
            node.required_capabilities.clone()
        };
        let allowed_tools = capabilities_to_tools(&capabilities);
        SubAgentConfigBuilder::new(&node.label, &node.goal)
            .read_files(node.read_files.clone())
            .write_files(node.write_files.clone())
            .allowed_tools(allowed_tools)
            .capabilities(capabilities)
            .max_iterations(self.config.default_max_iterations)
            .priority(node.priority)
            .task_node(node.id)
            .build()
    }

    /// Register a sub-agent and its lifecycle.
    pub fn register_agent(&mut self, config: SubAgentConfig) -> Uuid {
        let agent = SubAgent::new(config.clone());
        let agent_id = agent.id;
        let lifecycle = AgentLifecycle::new(agent_id);

        self.agents.insert(agent_id, (agent, lifecycle));

        // Track by workstream
        let ws_label = config.name.clone();
        self.workstreams.entry(ws_label).or_default().push(agent_id);

        if let Some(node_id) = config.task_node_id {
            for graph in self.graphs.values_mut() {
                if let Some(node) = graph.nodes.get_mut(&node_id) {
                    node.agent_id = Some(agent_id);
                    break;
                }
            }
        }

        agent_id
    }

    /// Get a sub-agent by ID.
    pub fn get_agent(&self, id: &Uuid) -> Option<&(SubAgent, AgentLifecycle)> {
        self.agents.get(id)
    }

    /// Get a mutable reference to a sub-agent and lifecycle.
    pub fn get_agent_mut(&mut self, id: &Uuid) -> Option<&mut (SubAgent, AgentLifecycle)> {
        self.agents.get_mut(id)
    }

    /// Start a sub-agent (transition from Queued to Running).
    pub fn start_agent(&mut self, id: Uuid) -> Result<(), String> {
        let file_locks = self.file_locks.clone();
        let agent_id = id.to_string();
        let held_before: std::collections::HashSet<String> = file_locks
            .snapshot()
            .held_locks
            .into_iter()
            .filter(|lock| lock.agent_id == agent_id)
            .map(|lock| lock.path)
            .collect();
        let rollback_new_locks = |lifecycle: &mut AgentLifecycle| {
            for path in lifecycle.held_locks.clone() {
                if !held_before.contains(&path) {
                    file_locks.release(&path, id);
                    lifecycle.lock_released(&path);
                }
            }
        };

        let task_node_id = {
            let (agent, lifecycle) = self
                .agents
                .get_mut(&id)
                .ok_or_else(|| format!("Agent {} not found", id))?;

            // Acquire file locks if needed. If any acquisition fails, roll
            // back only locks added by this attempt and preserve FIFO grants.
            if agent.needs_write_locks() {
                for file in &agent.config.write_files {
                    match file_locks.acquire_write(file, id) {
                        Ok(()) => {
                            lifecycle.lock_acquired(file);
                        }
                        Err(msg) => {
                            rollback_new_locks(lifecycle);
                            lifecycle.wait_for_lock(file);
                            return Err(msg);
                        }
                    }
                }
            }

            // A write lock already grants read access for the same path.
            for file in &agent.config.read_files {
                if agent.config.write_files.contains(file) {
                    continue;
                }
                match file_locks.acquire_read(file, id) {
                    Ok(()) => lifecycle.lock_acquired(file),
                    Err(msg) => {
                        rollback_new_locks(lifecycle);
                        lifecycle.wait_for_lock(file);
                        return Err(msg);
                    }
                }
            }

            agent.phase = AgentPhase::Running;
            lifecycle.start();
            agent.config.task_node_id
        };
        self.update_task_node(task_node_id, TaskNodeStatus::Running, None);
        Ok(())
    }

    /// Complete a sub-agent.
    pub fn complete_agent(&mut self, id: Uuid, result: SubAgentResult) {
        let task_node_id = if let Some((agent, lifecycle)) = self.agents.get_mut(&id) {
            agent.result = Some(result.summary.clone());
            if result.success {
                agent.phase = AgentPhase::Done;
                lifecycle.complete();
            } else {
                let error = result
                    .error
                    .clone()
                    .unwrap_or_else(|| result.summary.clone());
                agent.phase = AgentPhase::Failed;
                agent.error = Some(error.clone());
                lifecycle.fail(error);
            }

            // Release all file locks
            self.file_locks.release_all(id);
            agent.config.task_node_id
        } else {
            None
        };
        self.update_task_node(
            task_node_id,
            if result.success {
                TaskNodeStatus::Done
            } else {
                TaskNodeStatus::Failed
            },
            Some(result.summary),
        );
    }

    /// Fail a sub-agent.
    pub fn fail_agent(&mut self, id: Uuid, error: impl Into<String>) {
        let error = error.into();
        let task_node_id = if let Some((agent, lifecycle)) = self.agents.get_mut(&id) {
            agent.phase = AgentPhase::Failed;
            agent.error = Some(error.clone());
            lifecycle.fail(error);

            // Release all file locks
            self.file_locks.release_all(id);
            agent.config.task_node_id
        } else {
            None
        };
        self.update_task_node(task_node_id, TaskNodeStatus::Failed, None);
    }

    /// Cancel a sub-agent.
    pub fn cancel_agent(&mut self, id: Uuid, reason: impl Into<String>) {
        let task_node_id = if let Some((agent, lifecycle)) = self.agents.get_mut(&id) {
            agent.phase = AgentPhase::Cancelled;
            lifecycle.cancel(reason);

            // Release all file locks
            self.file_locks.release_all(id);
            agent.config.task_node_id
        } else {
            None
        };
        self.update_task_node(task_node_id, TaskNodeStatus::Cancelled, None);
    }

    fn update_task_node(
        &mut self,
        node_id: Option<Uuid>,
        status: TaskNodeStatus,
        result: Option<String>,
    ) {
        let Some(node_id) = node_id else {
            return;
        };
        for graph in self.graphs.values_mut() {
            let Some(node) = graph.nodes.get_mut(&node_id) else {
                continue;
            };
            node.status = status;
            if result.is_some() {
                node.result = result;
            }

            let all_terminal = graph.nodes.values().all(|node| {
                matches!(
                    node.status,
                    TaskNodeStatus::Done | TaskNodeStatus::Failed | TaskNodeStatus::Cancelled
                )
            });
            if all_terminal {
                graph.status = if graph
                    .nodes
                    .values()
                    .all(|node| node.status == TaskNodeStatus::Done)
                {
                    TaskGraphStatus::Complete
                } else if graph
                    .nodes
                    .values()
                    .any(|node| node.status == TaskNodeStatus::Failed)
                {
                    TaskGraphStatus::Failed
                } else {
                    TaskGraphStatus::Cancelled
                };
            }
            break;
        }
    }

    /// Pause all running agents (mark as blocked, release file locks).
    pub fn pause_all(&mut self) {
        let ids: Vec<Uuid> = self
            .agents
            .iter()
            .filter(|(_, (a, _))| a.phase == AgentPhase::Running)
            .map(|(id, _)| *id)
            .collect();

        for id in &ids {
            // Release file locks for this agent so others can proceed
            self.file_locks.release_all(*id);
        }

        for (agent, lifecycle) in self.agents.values_mut() {
            if agent.phase == AgentPhase::Running {
                agent.phase = AgentPhase::Blocked;
                lifecycle.block("Paused by user");
            }
        }
    }

    /// Resume all blocked agents (re-acquire file locks if needed).
    pub fn resume_all(&mut self) -> Result<usize, String> {
        let mut resumed = 0;
        let ids: Vec<Uuid> = self
            .agents
            .iter()
            .filter(|(_, (a, _))| a.phase == AgentPhase::Blocked)
            .map(|(id, _)| *id)
            .collect();

        for id in ids {
            // Re-acquire file locks if needed, then start
            let needs_locks = {
                if let Some((agent, _)) = self.agents.get(&id) {
                    agent.needs_write_locks()
                } else {
                    false
                }
            };

            if needs_locks {
                // Try to acquire write locks
                let write_files: Vec<String> = self
                    .agents
                    .get(&id)
                    .map(|(a, _)| a.config.write_files.clone())
                    .unwrap_or_default();

                let mut all_acquired = true;
                for file in &write_files {
                    if self.file_locks.acquire_write(file, id).is_err() {
                        all_acquired = false;
                        break;
                    }
                }
                if !all_acquired {
                    continue; // could not re-acquire locks, skip
                }
            }

            // Also acquire read locks
            if let Some((agent, _)) = self.agents.get(&id) {
                for file in &agent.config.read_files {
                    let _ = self.file_locks.acquire_read(file, id);
                }
            }

            // Transition to running
            if let Some((agent, lifecycle)) = self.agents.get_mut(&id) {
                agent.phase = AgentPhase::Running;
                lifecycle.start();
            }
            resumed += 1;
        }
        Ok(resumed)
    }

    // ── Workstream Detection ──────────────────────────────────────

    /// Detect independent workstreams from a task graph.
    /// Returns groups of node IDs that can run in parallel.
    pub fn detect_workstreams(&self, graph: &TaskGraph) -> Vec<Vec<Uuid>> {
        graph.independent_groups()
    }

    /// Check if two sub-agents have overlapping file access.
    pub fn has_file_overlap(&self, a_id: Uuid, b_id: Uuid) -> bool {
        let a = self.agents.get(&a_id);
        let b = self.agents.get(&b_id);
        match (a, b) {
            (Some((a_agent, _)), Some((b_agent, _))) => !a_agent.file_overlap(b_agent).is_empty(),
            _ => false,
        }
    }

    // ── Merge & Results ───────────────────────────────────────────

    /// Collect results from completed agents and merge.
    pub fn collect_results(&self) -> Vec<SubAgentResult> {
        self.agents
            .values()
            .filter(|(a, _)| a.phase == AgentPhase::Done || a.phase == AgentPhase::Failed)
            .map(|(agent, lifecycle)| SubAgentResult {
                agent_id: agent.id,
                name: agent.config.name.clone(),
                summary: agent.result.clone().unwrap_or_else(|| "No summary".into()),
                output: None,
                modified_files: agent.config.write_files.clone(),
                success: agent.phase == AgentPhase::Done,
                error: agent.error.clone(),
                duration_ms: lifecycle
                    .active_duration()
                    .map(|d| d.num_milliseconds() as u64)
                    .unwrap_or(0),
            })
            .collect()
    }

    /// Merge all collected results into a final response.
    pub fn merge_results(&self, results: Vec<SubAgentResult>) -> ComposerResult {
        let merged = self
            .merge_resolver
            .merge(results, self.config.merge_strategy);

        ComposerResult {
            goal: self
                .graphs
                .values()
                .next()
                .map(|g| g.root_goal.clone())
                .unwrap_or_default(),
            success: merged.success,
            summary: merged.summary,
            agent_results: merged.results,
            modified_files: merged.all_modified_files,
            conflicts: merged.conflicts,
            duration_ms: 0, // set externally
            agent_count: self.agents.len(),
        }
    }

    // ── Progress & Status ─────────────────────────────────────────

    /// Update the progress tracker from current state.
    pub fn update_progress(&mut self) {
        for (ws_label, agent_ids) in &self.workstreams {
            let total = agent_ids.len();
            let completed = agent_ids
                .iter()
                .filter(|id| {
                    self.agents
                        .get(id)
                        .map(|(a, _)| a.phase == AgentPhase::Done)
                        .unwrap_or(false)
                })
                .count();
            let running = agent_ids
                .iter()
                .filter(|id| {
                    self.agents
                        .get(id)
                        .map(|(a, _)| a.phase.is_active())
                        .unwrap_or(false)
                })
                .count();
            let failed = agent_ids
                .iter()
                .filter(|id| {
                    self.agents
                        .get(id)
                        .map(|(a, _)| a.phase == AgentPhase::Failed)
                        .unwrap_or(false)
                })
                .count();
            let queued = agent_ids
                .iter()
                .filter(|id| {
                    self.agents
                        .get(id)
                        .map(|(a, _)| a.phase == AgentPhase::Queued)
                        .unwrap_or(false)
                })
                .count();

            self.progress.update_workstream(WorkstreamStatus {
                id: Uuid::new_v4(),
                label: ws_label.clone(),
                total_agents: total,
                completed,
                running,
                failed,
                queued,
                started_at: chrono::Utc::now(),
                estimated_completion: None,
                status_message: if failed > 0 {
                    format!("{} failed", failed)
                } else {
                    format!("{}/{} complete", completed, total)
                },
            });
        }
    }

    /// Get a progress summary string.
    pub fn progress_summary(&self) -> String {
        self.progress.format_summary()
    }

    /// Get a compact one-line status.
    pub fn compact_status(&self) -> String {
        self.progress.format_compact()
    }

    /// Check if all work is complete.
    pub fn is_all_done(&self) -> bool {
        self.agents.values().all(|(a, _)| a.phase.is_terminal())
    }

    /// Check if any agent has failed.
    pub fn has_failures(&self) -> bool {
        self.agents
            .values()
            .any(|(a, _)| a.phase == AgentPhase::Failed)
    }

    // ── Interruption Handling ─────────────────────────────────────

    /// Cancel all agents.
    pub fn cancel_all(&mut self, reason: &str) {
        let ids: Vec<Uuid> = self.agents.keys().copied().collect();
        for id in ids {
            self.cancel_agent(id, reason);
        }
    }

    /// Reprioritize an agent (set its priority).
    pub fn reprioritize(&mut self, id: Uuid, new_priority: u32) -> Result<(), String> {
        if let Some((agent, _)) = self.agents.get_mut(&id) {
            agent.config.priority = new_priority;
            Ok(())
        } else {
            Err(format!("Agent {} not found", id))
        }
    }

    // ── File Lock Access ──────────────────────────────────────────

    /// Get a reference to the file lock manager.
    pub fn file_locks(&self) -> &Arc<FileLockManager> {
        &self.file_locks
    }

    /// Get file lock summary.
    pub fn lock_summary(&self) -> crate::file_lock::FileLockSummary {
        self.file_locks.summary()
    }

    // ── Graph Access ──────────────────────────────────────────────

    /// Get the task graph for a root goal.
    pub fn get_graph(&self, root_goal: &str) -> Option<&TaskGraph> {
        self.graphs.get(root_goal)
    }

    /// Set the persisted run status without changing active agent lifecycles.
    ///
    /// This is used for drain-to-safe-point pauses: no new agents start while
    /// already-running agents retain their locks until they finish.
    pub fn set_graph_status(&mut self, root_goal: &str, status: TaskGraphStatus) {
        if let Some(graph) = self.graphs.get_mut(root_goal) {
            graph.status = status;
        }
    }

    /// Update a node status in the task graph.
    pub fn update_node_status(&mut self, root_goal: &str, node_id: Uuid, status: TaskNodeStatus) {
        if let Some(graph) = self.graphs.get_mut(root_goal) {
            graph.update_node_status(node_id, status);
        }
    }
}

/// Default capability tags for heuristic (non-LLM) task nodes.
pub fn default_capabilities_for_goal(goal: &str) -> Vec<String> {
    let lower = goal.to_ascii_lowercase();
    let mut words: std::collections::HashSet<String> = lower
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .filter(|word| !word.is_empty())
        .map(str::to_string)
        .collect();
    // Normalize common English inflections without falling back to unsafe
    // substring matching (for example, `prefix` must not imply `fix`).
    for word in words.clone() {
        for suffix in ["ing", "ed", "es", "s"] {
            if let Some(stem) = word.strip_suffix(suffix).filter(|stem| stem.len() >= 3) {
                words.insert(stem.to_string());
                if matches!(suffix, "ing" | "ed") {
                    words.insert(format!("{stem}e"));
                    let mut reversed = stem.chars().rev();
                    if let (Some(last), Some(previous)) = (reversed.next(), reversed.next())
                        && last == previous
                    {
                        let shortened = &stem[..stem.len() - last.len_utf8()];
                        words.insert(shortened.to_string());
                    }
                }
            }
        }
    }
    let has_any = |candidates: &[&str]| candidates.iter().any(|word| words.contains(*word));

    let mut caps = vec!["read".to_string(), "filesystem".to_string()];
    if has_any(&[
        "write",
        "edit",
        "fix",
        "create",
        "update",
        "refactor",
        "implement",
        "add",
        "remove",
        "delete",
        "rename",
        "replace",
        "change",
        "modify",
        "move",
        "changes",
        "edits",
        "updates",
        "fixes",
        "improve",
    ]) {
        caps.push("write".to_string());
    }
    if has_any(&[
        "shell", "command", "commands", "run", "test", "tests", "build", "builds",
    ]) {
        caps.push("shell".to_string());
    }
    if has_any(&["git", "commit", "commits", "branch", "branches"]) {
        caps.push("git".to_string());
    }
    if has_any(&[
        "http", "https", "web", "fetch", "url", "urls", "search", "searches",
    ]) {
        caps.push("web".to_string());
    }
    if has_any(&["github", "issue", "issues"]) || lower.contains("pull request") {
        caps.push("github".to_string());
    }
    caps.sort();
    caps.dedup();
    caps
}

/// Map capability tags to concrete built-in tool names for sub-agent scoping.
pub fn capabilities_to_tools(capabilities: &[String]) -> Vec<String> {
    if capabilities.is_empty() {
        return default_agent_tools();
    }

    let mut tools = Vec::new();
    for cap in capabilities {
        match cap.to_ascii_lowercase().as_str() {
            "read" | "filesystem" => {
                tools.extend(
                    [
                        "file_read",
                        "file_list",
                        "file_exists",
                        "text_search",
                        "json_extract",
                        "json_validate",
                    ]
                    .map(str::to_string),
                );
            }
            "write" => {
                tools.extend(["file_write", "file_delete"].map(str::to_string));
            }
            "shell" => {
                tools.extend(["shell", "process_list", "network_ping"].map(str::to_string));
            }
            "git" => tools.push("git".into()),
            "web" => {
                tools.extend(["web_fetch", "web_search", "http_request"].map(str::to_string));
            }
            "github" => {
                tools.extend(
                    [
                        "github_issue_create",
                        "github_issue_search",
                        "github_pr_create",
                        "github_pr_status",
                        "github_actions_status",
                    ]
                    .map(str::to_string),
                );
            }
            "safe" => tools.extend(default_agent_tools()),
            // Capability strings may be model-generated. Unknown values must
            // never be interpreted as concrete tool names.
            _ => {}
        }
    }
    tools.sort();
    tools.dedup();
    if tools.is_empty() {
        default_agent_tools()
    } else {
        tools
    }
}

/// Default tool allow-list for a general-purpose sub-agent (not the full registry).
pub fn default_agent_tools() -> Vec<String> {
    [
        "file_read",
        "file_list",
        "file_exists",
        "text_search",
        "json_extract",
        "json_validate",
        "time_now",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_single_goal() {
        let composer = Composer::default();
        let goal = "Fix the CLI bug";
        let graph = composer.decompose(goal);

        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.root_goal, goal);
        let node = graph.nodes.values().next().unwrap();
        assert!(
            !node.required_capabilities.is_empty(),
            "heuristic nodes must declare capabilities for tool scoping"
        );
    }

    #[test]
    fn test_decompose_multiple_goals() {
        let composer = Composer::default();
        let goal = "Fix the CLI bug, improve docs, add tests";
        let graph = composer.decompose(goal);

        assert_eq!(graph.nodes.len(), 3);
        assert!(graph.nodes.values().any(|n| n.goal.contains("Fix")));
        assert!(graph.nodes.values().any(|n| n.goal.contains("docs")));
        assert!(graph.nodes.values().any(|n| n.goal.contains("tests")));
    }

    #[test]
    fn test_create_sub_agent_scopes_tools_from_capabilities() {
        let composer = Composer::default();
        let node = TaskNode {
            id: Uuid::new_v4(),
            label: "docs".into(),
            goal: "read the docs".into(),
            read_files: vec!["README.md".into()],
            write_files: vec![],
            required_capabilities: vec!["read".into(), "filesystem".into()],
            priority: 0,
            status: TaskNodeStatus::Pending,
            result: None,
            agent_id: None,
        };
        let config = composer.create_sub_agent(&node);
        assert!(config.allowed_tools.contains(&"file_read".to_string()));
        assert!(config.allowed_tools.contains(&"file_list".to_string()));
        assert!(!config.allowed_tools.contains(&"shell".to_string()));
        assert!(
            !config
                .allowed_tools
                .contains(&"github_pr_create".to_string())
        );
    }

    #[test]
    fn test_capabilities_to_tools_default_and_mapping() {
        let defaults = capabilities_to_tools(&[]);
        assert!(defaults.contains(&"file_read".to_string()));
        for privileged in [
            "file_write",
            "file_delete",
            "shell",
            "git",
            "web_fetch",
            "web_search",
            "env_var",
        ] {
            assert!(
                !defaults.contains(&privileged.to_string()),
                "default scope must not grant {privileged}"
            );
        }

        let generic = default_capabilities_for_goal("summarize the README");
        let generic_tools = capabilities_to_tools(&generic);
        assert!(!generic_tools.contains(&"shell".to_string()));
        assert!(!generic_tools.contains(&"file_write".to_string()));
        assert!(!generic_tools.contains(&"git".to_string()));

        for mutation in [
            "add a test",
            "remove dead code",
            "delete a file",
            "rename a symbol",
            "replace the parser",
            "modify configuration",
            "improve docs",
            "fixing the parser",
            "implementing auth",
        ] {
            assert!(
                default_capabilities_for_goal(mutation).contains(&"write".to_string()),
                "mutation goal should receive write capability: {mutation}"
            );
        }
        assert!(
            default_capabilities_for_goal("create regression tests").contains(&"shell".to_string()),
            "plural test goals should receive shell capability"
        );
        for git_goal in ["committing the current work", "committed the result"] {
            assert!(
                default_capabilities_for_goal(git_goal).contains(&"git".to_string()),
                "git inflection should receive git capability: {git_goal}"
            );
        }

        let substring_traps = default_capabilities_for_goal("summarize latest legitimate prefix");
        assert!(!substring_traps.contains(&"shell".to_string()));
        assert!(!substring_traps.contains(&"write".to_string()));
        assert!(!substring_traps.contains(&"git".to_string()));

        let unknown = capabilities_to_tools(&["rust".into(), "testing".into()]);
        assert_eq!(unknown, defaults, "unknown capabilities must fail closed");

        let shell = capabilities_to_tools(&["shell".into()]);
        assert!(shell.contains(&"shell".to_string()));
        assert!(shell.contains(&"process_list".to_string()));
    }

    #[test]
    fn test_split_goal_semicolons() {
        let composer = Composer::default();
        let parts = composer.split_goal("task a; task b; task c");
        assert_eq!(parts.len(), 3);
    }

    #[test]
    fn test_split_goal_and() {
        let composer = Composer::default();
        let parts = composer.split_goal("task a and task b and task c");
        assert_eq!(parts.len(), 3);
    }

    #[test]
    fn test_split_goal_single() {
        let composer = Composer::default();
        let parts = composer.split_goal("just one task");
        assert_eq!(parts.len(), 1);
    }

    #[test]
    fn test_agent_registration_and_lifecycle() {
        let mut composer = Composer::default();
        let config = SubAgentConfigBuilder::new("test", "do work").build();
        let id = composer.register_agent(config);

        assert!(composer.get_agent(&id).is_some());
        assert_eq!(composer.get_agent(&id).unwrap().1.phase, AgentPhase::Queued);

        composer.start_agent(id).unwrap();
        assert_eq!(
            composer.get_agent(&id).unwrap().1.phase,
            AgentPhase::Running
        );
    }

    #[test]
    fn test_agent_completion() {
        let mut composer = Composer::default();
        let config = SubAgentConfigBuilder::new("test", "do work").build();
        let id = composer.register_agent(config);
        composer.start_agent(id).unwrap();

        let result = SubAgentResult {
            agent_id: id,
            name: "test".into(),
            summary: "done!".into(),
            output: None,
            modified_files: vec![],
            success: true,
            error: None,
            duration_ms: 100,
        };
        composer.complete_agent(id, result);
        assert_eq!(composer.get_agent(&id).unwrap().1.phase, AgentPhase::Done);
    }

    #[test]
    fn test_unsuccessful_completion_marks_agent_failed() {
        let mut composer = Composer::default();
        let config = SubAgentConfigBuilder::new("test", "do work").build();
        let id = composer.register_agent(config);
        composer.start_agent(id).unwrap();

        composer.complete_agent(
            id,
            SubAgentResult {
                agent_id: id,
                name: "test".into(),
                summary: "verification failed".into(),
                output: None,
                modified_files: vec![],
                success: false,
                error: Some("tests failed".into()),
                duration_ms: 100,
            },
        );

        let (agent, lifecycle) = composer.get_agent(&id).unwrap();
        assert_eq!(agent.phase, AgentPhase::Failed);
        assert_eq!(lifecycle.phase, AgentPhase::Failed);
        assert_eq!(agent.error.as_deref(), Some("tests failed"));
    }

    #[test]
    fn test_agent_failure() {
        let mut composer = Composer::default();
        let config = SubAgentConfigBuilder::new("test", "do work").build();
        let id = composer.register_agent(config);
        composer.start_agent(id).unwrap();
        composer.fail_agent(id, "something broke");

        let (agent, lifecycle) = composer.get_agent(&id).unwrap();
        assert_eq!(lifecycle.phase, AgentPhase::Failed);
        assert_eq!(agent.error.as_deref(), Some("something broke"));
    }

    #[test]
    fn test_cancel_all() {
        let mut composer = Composer::default();
        let id1 = composer.register_agent(SubAgentConfigBuilder::new("a", "a").build());
        let id2 = composer.register_agent(SubAgentConfigBuilder::new("b", "b").build());
        composer.start_agent(id1).unwrap();
        composer.start_agent(id2).unwrap();

        composer.cancel_all("user interrupted");
        assert!(composer.is_all_done());
    }

    #[test]
    fn test_pause_and_resume() {
        let mut composer = Composer::default();
        let id = composer.register_agent(SubAgentConfigBuilder::new("a", "a").build());
        composer.start_agent(id).unwrap();

        composer.pause_all();
        assert_eq!(
            composer.get_agent(&id).unwrap().0.phase,
            AgentPhase::Blocked
        );

        let resumed = composer.resume_all().unwrap();
        assert_eq!(resumed, 1);
        assert_eq!(
            composer.get_agent(&id).unwrap().0.phase,
            AgentPhase::Running
        );
    }

    #[test]
    fn test_file_lock_acquire_on_start() {
        let mut composer = Composer::default();
        let config = SubAgentConfigBuilder::new("writer", "write stuff")
            .write_files(vec!["main.rs".into()])
            .build();
        let id = composer.register_agent(config);

        composer.start_agent(id).unwrap();
        assert!(composer.file_locks.has_write_lock("main.rs"));

        composer.complete_agent(
            id,
            SubAgentResult {
                agent_id: id,
                name: "writer".into(),
                summary: "done".into(),
                output: None,
                modified_files: vec!["main.rs".into()],
                success: true,
                error: None,
                duration_ms: 0,
            },
        );
        assert!(!composer.file_locks.has_write_lock("main.rs"));
    }

    #[test]
    fn test_write_lock_queueing() {
        let mut composer = Composer::default();

        // First writer gets the lock
        let config1 = SubAgentConfigBuilder::new("w1", "write 1")
            .write_files(vec!["shared.rs".into()])
            .build();
        let id1 = composer.register_agent(config1);
        composer.start_agent(id1).unwrap();

        // Second writer should queue
        let config2 = SubAgentConfigBuilder::new("w2", "write 2")
            .write_files(vec!["shared.rs".into()])
            .build();
        let id2 = composer.register_agent(config2);
        let result = composer.start_agent(id2);
        assert!(result.is_err()); // queued

        // Release first → second should get lock
        composer.complete_agent(
            id1,
            SubAgentResult {
                agent_id: id1,
                name: "w1".into(),
                summary: "done".into(),
                output: None,
                modified_files: vec!["shared.rs".into()],
                success: true,
                error: None,
                duration_ms: 0,
            },
        );

        assert!(composer.file_locks.has_write_lock("shared.rs"));
        composer.start_agent(id2).unwrap();
        assert_eq!(
            composer.get_agent(&id2).unwrap().0.phase,
            AgentPhase::Running
        );
    }

    #[test]
    fn test_reader_waits_for_active_writer() {
        let mut composer = Composer::default();
        let writer = SubAgentConfigBuilder::new("writer", "write")
            .write_files(vec!["shared.rs".into()])
            .build();
        let writer_id = composer.register_agent(writer);
        composer.start_agent(writer_id).unwrap();

        let reader = SubAgentConfigBuilder::new("reader", "read")
            .read_files(vec!["shared.rs".into()])
            .build();
        let reader_id = composer.register_agent(reader);

        assert!(composer.start_agent(reader_id).is_err());
        assert_eq!(
            composer.get_agent(&reader_id).unwrap().1.phase,
            AgentPhase::WaitingForLock
        );

        composer.complete_agent(
            writer_id,
            SubAgentResult {
                agent_id: writer_id,
                name: "writer".into(),
                summary: "done".into(),
                output: None,
                modified_files: vec!["shared.rs".into()],
                success: true,
                error: None,
                duration_ms: 1,
            },
        );

        composer.start_agent(reader_id).unwrap();
        assert_eq!(
            composer.get_agent(&reader_id).unwrap().0.phase,
            AgentPhase::Running
        );
    }

    #[test]
    fn test_failed_start_rolls_back_newly_acquired_locks() {
        let mut composer = Composer::default();
        let blocker = SubAgentConfigBuilder::new("blocker", "write")
            .write_files(vec!["blocked.rs".into()])
            .build();
        let blocker_id = composer.register_agent(blocker);
        composer.start_agent(blocker_id).unwrap();

        let waiter = SubAgentConfigBuilder::new("waiter", "read")
            .read_files(vec!["free.rs".into(), "blocked.rs".into()])
            .build();
        let waiter_id = composer.register_agent(waiter);

        assert!(composer.start_agent(waiter_id).is_err());
        assert!(!composer.file_locks.is_locked("free.rs"));
        assert!(
            composer
                .get_agent(&waiter_id)
                .unwrap()
                .1
                .held_locks
                .is_empty()
        );
    }

    #[test]
    fn test_failed_start_preserves_preexisting_fifo_grant() {
        let mut composer = Composer::default();
        let first = composer.register_agent(
            SubAgentConfigBuilder::new("first", "write")
                .write_files(vec!["a.rs".into()])
                .build(),
        );
        let blocker = composer.register_agent(
            SubAgentConfigBuilder::new("blocker", "write")
                .write_files(vec!["b.rs".into()])
                .build(),
        );
        composer.start_agent(first).unwrap();
        composer.start_agent(blocker).unwrap();

        let waiter = composer.register_agent(
            SubAgentConfigBuilder::new("waiter", "write")
                .write_files(vec!["a.rs".into(), "b.rs".into()])
                .build(),
        );
        assert!(composer.start_agent(waiter).is_err());

        composer.complete_agent(
            first,
            SubAgentResult {
                agent_id: first,
                name: "first".into(),
                summary: "done".into(),
                output: None,
                modified_files: vec!["a.rs".into()],
                success: true,
                error: None,
                duration_ms: 1,
            },
        );

        assert!(composer.start_agent(waiter).is_err());
        assert_eq!(composer.file_locks.lock_holders("a.rs"), vec![waiter]);

        composer.complete_agent(
            blocker,
            SubAgentResult {
                agent_id: blocker,
                name: "blocker".into(),
                summary: "done".into(),
                output: None,
                modified_files: vec!["b.rs".into()],
                success: true,
                error: None,
                duration_ms: 1,
            },
        );
        composer.start_agent(waiter).unwrap();
        assert_eq!(
            composer.get_agent(&waiter).unwrap().0.phase,
            AgentPhase::Running
        );
    }

    #[test]
    fn test_collect_and_merge_results() {
        let mut composer = Composer::default();
        let id1 = composer.register_agent(
            SubAgentConfigBuilder::new("a", "task a")
                .write_files(vec!["a.txt".into()])
                .build(),
        );
        let id2 = composer.register_agent(
            SubAgentConfigBuilder::new("b", "task b")
                .write_files(vec!["b.txt".into()])
                .build(),
        );

        composer.start_agent(id1).unwrap();
        composer.start_agent(id2).unwrap();

        composer.complete_agent(
            id1,
            SubAgentResult {
                agent_id: id1,
                name: "a".into(),
                summary: "Fixed A".into(),
                output: None,
                modified_files: vec!["a.txt".into()],
                success: true,
                error: None,
                duration_ms: 100,
            },
        );
        composer.complete_agent(
            id2,
            SubAgentResult {
                agent_id: id2,
                name: "b".into(),
                summary: "Fixed B".into(),
                output: None,
                modified_files: vec!["b.txt".into()],
                success: true,
                error: None,
                duration_ms: 200,
            },
        );

        let results = composer.collect_results();
        assert_eq!(results.len(), 2);

        let merged = composer.merge_results(results);
        assert!(merged.success);
        assert_eq!(merged.modified_files.len(), 2);
    }

    #[test]
    fn test_reprioritize() {
        let mut composer = Composer::default();
        let config = SubAgentConfigBuilder::new("test", "do work")
            .priority(10)
            .build();
        let id = composer.register_agent(config);

        composer.reprioritize(id, 1).unwrap();
        let (agent, _) = composer.get_agent(&id).unwrap();
        assert_eq!(agent.config.priority, 1);
    }

    #[test]
    fn test_is_all_done() {
        let mut composer = Composer::default();
        assert!(composer.is_all_done()); // no agents = done

        let id = composer.register_agent(SubAgentConfigBuilder::new("a", "a").build());
        assert!(!composer.is_all_done()); // queued

        composer.start_agent(id).unwrap();
        assert!(!composer.is_all_done()); // running

        composer.complete_agent(
            id,
            SubAgentResult {
                agent_id: id,
                name: "a".into(),
                summary: "done".into(),
                output: None,
                modified_files: vec![],
                success: true,
                error: None,
                duration_ms: 0,
            },
        );
        assert!(composer.is_all_done());
    }

    #[test]
    fn test_progress_update() {
        let mut composer = Composer::default();
        let id = composer.register_agent(SubAgentConfigBuilder::new("ws1", "task").build());
        composer.start_agent(id).unwrap();
        composer.complete_agent(
            id,
            SubAgentResult {
                agent_id: id,
                name: "ws1".into(),
                summary: "done".into(),
                output: None,
                modified_files: vec![],
                success: true,
                error: None,
                duration_ms: 0,
            },
        );

        composer.update_progress();
        let summary = composer.progress_summary();
        assert!(summary.contains("ws1"));
    }
}
