//! Runtime — orchestrates multiple agents, manages sessions, and spawns sub-agents.

use dashmap::DashMap;
use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::MemoryStore;
use odin_core::types::{AgentId, AgentTask, SessionId, TaskResult};
use std::sync::Arc;

use crate::agent::Agent;
use crate::session::Session;

/// The core runtime that orchestrates agents and sessions.
///
/// The Runtime is the top-level coordinator. It:
/// - Manages a pool of named agents
/// - Tracks multiple sessions with their message history
/// - Provides sub-agent spawning for parallel task execution
/// - Task submission and result collection
/// - Optional persistent memory via MemoryStore
pub struct Runtime {
    /// Active sessions, keyed by SessionId.
    sessions: Arc<DashMap<SessionId, Session>>,

    /// Registered agents, keyed by AgentId.
    agents: Arc<DashMap<AgentId, Agent>>,

    /// Sub-agents spawned for parallel execution.
    sub_agents: Arc<DashMap<AgentId, Agent>>,

    /// Default max iterations for tasks.
    default_max_iterations: u32,

    /// Optional persistent memory store.
    memory: Option<Arc<dyn MemoryStore>>,
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

impl Runtime {
    /// Create a new empty runtime.
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(DashMap::new()),
            agents: Arc::new(DashMap::new()),
            sub_agents: Arc::new(DashMap::new()),
            default_max_iterations: 100,
            memory: None,
        }
    }

    /// Attach a persistent memory store.
    pub fn with_memory(mut self, memory: Arc<dyn MemoryStore>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Set the default max iterations.
    pub fn with_default_max_iterations(mut self, max: u32) -> Self {
        self.default_max_iterations = max;
        self
    }

    // ── Session Management ──────────────────────────────────────────

    /// Create a new session.
    pub fn create_session(&self) -> Session {
        let session = Session::new();
        self.sessions.insert(session.id, session.clone());
        tracing::info!("[RUNTIME] Created session {}", session.id);
        session
    }

    /// Create a new session with a label.
    pub fn create_session_with_label(&self, label: &str) -> Session {
        let session = Session::with_label(label);
        self.sessions.insert(session.id, session.clone());
        tracing::info!("[RUNTIME] Created session '{}' ({})", label, session.id);
        session
    }

    /// Get a session by its ID.
    pub fn get_session(&self, id: &SessionId) -> Option<Session> {
        self.sessions.get(id).map(|s| s.clone())
    }

    /// Delete a session.
    pub fn delete_session(&self, id: &SessionId) -> Option<Session> {
        let session = self.sessions.remove(id).map(|(_k, v)| v);
        if session.is_some() {
            tracing::info!("[RUNTIME] Deleted session {}", id);
        }
        session
    }

    /// List all active sessions.
    pub fn list_sessions(&self) -> Vec<Session> {
        self.sessions.iter().map(|s| s.clone()).collect()
    }

    /// Get the number of active sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    // ── Agent Management ────────────────────────────────────────────

    /// Register an agent with the runtime.
    pub fn register_agent(&self, agent: Agent) {
        let id = agent.id;
        tracing::info!("[RUNTIME] Registered agent '{}' ({})", agent.name, id);
        self.agents.insert(id, agent);
    }

    /// Get an agent by ID.
    pub fn get_agent(&self, id: &AgentId) -> Option<Agent> {
        self.agents.get(id).map(|a| a.value().clone())
    }

    /// Find agents by name (returns all matching).
    pub fn find_agents_by_name(&self, name: &str) -> Vec<Agent> {
        self.agents
            .iter()
            .filter(|a| a.name == name)
            .map(|a| a.value().clone())
            .collect()
    }

    /// Remove an agent.
    pub fn remove_agent(&self, id: &AgentId) -> Option<Agent> {
        let agent = self.agents.remove(id).map(|(_k, v)| v);
        if agent.is_some() {
            tracing::info!("[RUNTIME] Removed agent {}", id);
        }
        agent
    }

    /// List all registered agents.
    pub fn list_agents(&self) -> Vec<Agent> {
        self.agents.iter().map(|a| a.value().clone()).collect()
    }

    /// Resolve an agent by UUID or exact name, rejecting ambiguous defaults.
    pub fn resolve_agent(&self, selector: Option<&str>) -> OdinResult<Agent> {
        if let Some(selector) = selector {
            let selector = selector.trim();
            if selector.is_empty() {
                return Err(OdinError::Config("agent selector cannot be empty".into()));
            }
            if let Ok(agent_id) = uuid::Uuid::parse_str(selector) {
                return self.get_agent(&agent_id).ok_or_else(|| {
                    OdinError::Config(format!("configured agent '{selector}' is not registered"))
                });
            }
            let matches = self.find_agents_by_name(selector);
            return match matches.as_slice() {
                [agent] => Ok(agent.clone()),
                [] => Err(OdinError::Config(format!(
                    "configured agent '{selector}' is not registered"
                ))),
                _ => Err(OdinError::Config(format!(
                    "agent selector '{selector}' matches multiple agents"
                ))),
            };
        }

        let agents = self.list_agents();
        match agents.as_slice() {
            [agent] => Ok(agent.clone()),
            [] => Err(OdinError::Config("runtime has no registered agent".into())),
            _ => Err(OdinError::Config(
                "runtime has multiple agents; an explicit agent selector is required".into(),
            )),
        }
    }

    /// Get the number of registered agents.
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    // ── Task Execution ──────────────────────────────────────────────

    /// Submit a task to a specific agent.
    pub async fn submit_task(
        &self,
        agent_id: &AgentId,
        task: &AgentTask,
        session_id: Option<SessionId>,
    ) -> OdinResult<TaskResult> {
        let agent = self
            .agents
            .get(agent_id)
            .map(|entry| entry.value().clone())
            .ok_or_else(|| {
                odin_core::error::OdinError::Internal(format!("Agent {agent_id} not found"))
            })?;

        tracing::info!(task_id = %task.id, agent_id = %agent.id, "Submitting task to agent");

        let result = agent.execute_task(task).await?;

        // If a session is provided, record the task result
        if let Some(sid) = session_id
            && let Some(mut session) = self.sessions.get_mut(&sid)
        {
            session.add_message(odin_core::types::Message::system(format!(
                "Task result: {}",
                result.summary
            )));
        }

        Ok(result)
    }

    // ── Sub-Agent Spawning ──────────────────────────────────────────

    /// Spawn a sub-agent to execute a task in the background.
    ///
    /// Returns the sub-agent's ID. The caller can poll for completion
    /// using `get_sub_agent_result`.
    pub fn spawn_sub_agent(&self, agent: Agent) -> AgentId {
        let id = agent.id;
        self.sub_agents.insert(id, agent);
        tracing::info!(
            "[RUNTIME] Spawned sub-agent {} ({})",
            id,
            self.sub_agents.len()
        );
        id
    }

    /// Get a sub-agent by ID.
    pub fn get_sub_agent(&self, id: &AgentId) -> Option<Agent> {
        self.sub_agents.get(id).map(|a| a.value().clone())
    }

    /// Remove a completed sub-agent.
    pub fn remove_sub_agent(&self, id: &AgentId) -> Option<Agent> {
        let agent = self.sub_agents.remove(id).map(|(_k, v)| v);
        if agent.is_some() {
            tracing::info!("[RUNTIME] Removed sub-agent {}", id);
        }
        agent
    }

    /// Get the number of active sub-agents.
    pub fn sub_agent_count(&self) -> usize {
        self.sub_agents.len()
    }

    /// List all sub-agents.
    pub fn list_sub_agents(&self) -> Vec<Agent> {
        self.sub_agents.iter().map(|a| a.value().clone()).collect()
    }

    // ── Utility ─────────────────────────────────────────────────────

    /// Create a basic agent task from a goal string.
    pub fn create_task(goal: impl Into<String>) -> AgentTask {
        AgentTask {
            id: uuid::Uuid::new_v4(),
            goal: goal.into(),
            context: None,
            sub_tasks: vec![],
            success_criteria: vec![],
            max_iterations: 100,
            created_at: chrono::Utc::now(),
        }
    }

    /// Get a summary of the runtime state.
    pub fn summary(&self) -> RuntimeSummary {
        RuntimeSummary {
            sessions: self.session_count(),
            agents: self.agent_count(),
            sub_agents: self.sub_agent_count(),
        }
    }
}

/// Summary of the runtime state.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RuntimeSummary {
    pub sessions: usize,
    pub agents: usize,
    pub sub_agents: usize,
}

// Clone implementation for Runtime (Arc-based, so cheap)
impl Clone for Runtime {
    fn clone(&self) -> Self {
        Self {
            sessions: self.sessions.clone(),
            agents: self.agents.clone(),
            sub_agents: self.sub_agents.clone(),
            default_max_iterations: self.default_max_iterations,
            memory: self.memory.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use async_trait::async_trait;
    use odin_core::traits::LoopEngine;
    use odin_core::types::*;
    use std::sync::Arc;

    struct MockEngine;

    #[async_trait]
    impl LoopEngine for MockEngine {
        async fn execute_task(&self, task: &AgentTask) -> OdinResult<TaskResult> {
            Ok(TaskResult {
                task_id: task.id,
                success: true,
                summary: "Done".into(),
                iterations: 1,
                tool_calls: 0,
                duration_ms: 0,
                sub_tasks: vec![],
                confidence: 1.0,
                error: None,
            })
        }

        async fn execute_phase(
            &self,
            _phase: LoopPhase,
            _state: &mut odin_core::traits::LoopState,
        ) -> OdinResult<odin_core::traits::PhaseResult> {
            Err(odin_core::error::OdinError::Other(
                "mock engine does not expose phases".into(),
            ))
        }

        fn state_summary(&self) -> StateSummary {
            StateSummary {
                goal: String::new(),
                current_phase: LoopPhase::Plan,
                completed_steps: vec![],
                pending_steps: vec![],
                last_action: None,
                last_result: None,
                errors: vec![],
                confidence: 1.0,
                token_usage: TokenUsage::default(),
            }
        }

        fn confidence(&self) -> ConfidenceScore {
            ConfidenceScore::new(1.0)
        }
    }

    struct MockProvider;

    #[async_trait]
    impl odin_core::traits::Provider for MockProvider {
        fn name(&self) -> &str {
            "mock"
        }
        async fn list_models(&self) -> OdinResult<Vec<ModelInfo>> {
            Ok(vec![])
        }
        async fn chat(
            &self,
            _model: &str,
            _messages: &[Message],
            _tools: &[ToolSchema],
            _options: &CompletionOptions,
        ) -> OdinResult<ChatResponse> {
            Err(odin_core::error::OdinError::Other(
                "mock provider chat is not used".into(),
            ))
        }
        async fn chat_stream(
            &self,
            _model: &str,
            _messages: &[Message],
            _tools: &[ToolSchema],
            _options: &CompletionOptions,
        ) -> OdinResult<Box<dyn odin_core::traits::ChatStream>> {
            Err(odin_core::error::OdinError::Other(
                "mock provider does not stream".into(),
            ))
        }
        async fn health_check(&self) -> OdinResult<bool> {
            Ok(true)
        }
    }

    fn make_agent(name: &str) -> Agent {
        Agent::new(name, Arc::new(MockEngine), Arc::new(MockProvider), vec![])
    }

    #[test]
    fn test_runtime_session_management() {
        let rt = Runtime::new();
        assert_eq!(rt.session_count(), 0);

        let session = rt.create_session();
        assert_eq!(rt.session_count(), 1);

        let fetched = rt.get_session(&session.id);
        assert!(fetched.is_some());
        assert_eq!(fetched.unwrap().id, session.id);

        let deleted = rt.delete_session(&session.id);
        assert!(deleted.is_some());
        assert_eq!(rt.session_count(), 0);
    }

    #[test]
    fn test_runtime_session_with_label() {
        let rt = Runtime::new();
        let session = rt.create_session_with_label("test-session");
        assert_eq!(session.label, Some("test-session".into()));
    }

    #[test]
    fn test_runtime_agent_management() {
        let rt = Runtime::new();
        assert_eq!(rt.agent_count(), 0);

        let agent = make_agent("worker-1");
        let id = agent.id;
        rt.register_agent(agent);
        assert_eq!(rt.agent_count(), 1);

        let fetched = rt.get_agent(&id);
        assert!(fetched.is_some());
        assert_eq!(fetched.unwrap().name, "worker-1");

        let removed = rt.remove_agent(&id);
        assert!(removed.is_some());
        assert_eq!(rt.agent_count(), 0);
    }

    #[test]
    fn test_runtime_find_agents_by_name() {
        let rt = Runtime::new();
        rt.register_agent(make_agent("builder"));
        rt.register_agent(make_agent("builder"));

        let builders = rt.find_agents_by_name("builder");
        assert_eq!(builders.len(), 2);
    }

    #[test]
    fn agent_resolution_rejects_ambiguous_defaults_and_names() {
        let rt = Runtime::new();
        rt.register_agent(make_agent("builder"));
        assert_eq!(rt.resolve_agent(None).unwrap().name, "builder");
        rt.register_agent(make_agent("reviewer"));
        assert!(rt.resolve_agent(None).is_err());
        assert_eq!(rt.resolve_agent(Some("reviewer")).unwrap().name, "reviewer");
        rt.register_agent(make_agent("reviewer"));
        assert!(rt.resolve_agent(Some("reviewer")).is_err());
    }

    #[test]
    fn test_runtime_sub_agents() {
        let rt = Runtime::new();
        let agent = make_agent("sub-worker");
        let id = rt.spawn_sub_agent(agent);

        assert_eq!(rt.sub_agent_count(), 1);
        assert!(rt.get_sub_agent(&id).is_some());

        let removed = rt.remove_sub_agent(&id);
        assert!(removed.is_some());
        assert_eq!(rt.sub_agent_count(), 0);
    }

    #[tokio::test]
    async fn test_runtime_submit_task() {
        let rt = Runtime::new();
        let agent = make_agent("executor");
        let id = agent.id;
        rt.register_agent(agent);

        let task = Runtime::create_task("Test task");
        let result = rt.submit_task(&id, &task, None).await.unwrap();

        assert!(result.success);
        assert_eq!(result.summary, "Done");
    }

    #[test]
    fn test_runtime_summary() {
        let rt = Runtime::new();
        rt.register_agent(make_agent("a"));
        rt.register_agent(make_agent("b"));
        rt.create_session();

        let s = rt.summary();
        assert_eq!(s.agents, 2);
        assert_eq!(s.sessions, 1);
    }
}
