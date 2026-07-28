//! Background orchestration runner used by the TUI.
//!
//! The CLI `raven run` path is still the non-interactive execution surface.
//! This module gives the interactive UI its own real controller instead of
//! persisting a static plan and asking the user to leave the TUI.

use std::collections::HashSet;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;
use futures::FutureExt;
use odin_core::config::OdinConfig;
use odin_core::traits::{AuditLogger, ChatStream, LoopEngine as _, Provider};
use odin_core::types::{
    AgentId, AgentTask, AuditEntry, AuditEventType, AuditResult, ChatResponse, CompletionOptions,
    Message, ModelInfo, SessionId, ToolSchema,
};
use odin_orchestrator::Composer;
use odin_orchestrator::composer::ComposerConfig;
use odin_orchestrator::lifecycle::AgentPhase;
use odin_orchestrator::merge::{MergeStrategy, SubAgentResult};
use odin_orchestrator::persistence::{OrchestrationStore, SqliteOrchestrationStore};
use odin_orchestrator::task_graph::{TaskGraph, TaskGraphStatus, TaskNodeStatus};
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use uuid::Uuid;

#[derive(Debug)]
pub struct RunHandle {
    pub run_id: String,
    pub command_tx: mpsc::UnboundedSender<RunnerCommand>,
    pub event_rx: mpsc::UnboundedReceiver<RunnerEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunnerCommand {
    Pause,
    Resume,
    Cancel,
    Redirect(String),
    Reprioritise {
        agent_id_prefix: String,
        priority: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunnerEvent {
    RunStage {
        run_id: String,
        stage: AgentStage,
        detail: String,
        elapsed_ms: u64,
    },
    RunStarted {
        run_id: String,
        goal: String,
        task_count: usize,
    },
    AgentQueued {
        agent_id: String,
        label: String,
        reason: String,
    },
    AgentStarted {
        agent_id: String,
        label: String,
        task: String,
    },
    AgentStage {
        agent_id: String,
        label: String,
        stage: AgentStage,
        detail: String,
        elapsed_ms: u64,
    },
    AgentFinished {
        agent_id: String,
        label: String,
        success: bool,
        summary: String,
    },
    RunPaused {
        run_id: String,
    },
    RunResumed {
        run_id: String,
    },
    RunRedirected {
        run_id: String,
        message: String,
    },
    RunCancelled {
        run_id: String,
    },
    RunFinished {
        run_id: String,
        success: bool,
        summary: String,
    },
    Reprioritised {
        agent_id: String,
        priority: u32,
    },
    Error {
        message: String,
    },
    FatalError {
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentStage {
    Planning,
    Decomposing,
    SpawningAgents,
    WaitingForModel,
    RunningTool,
    WaitingForLock,
    ApprovalNeeded,
    Retrying,
    Failed,
    Done,
    Paused,
    Cancelled,
}

impl AgentStage {
    pub fn label(self) -> &'static str {
        match self {
            AgentStage::Planning => "planning",
            AgentStage::Decomposing => "decomposing",
            AgentStage::SpawningAgents => "spawning_agents",
            AgentStage::WaitingForModel => "waiting_for_model",
            AgentStage::RunningTool => "running_tool",
            AgentStage::WaitingForLock => "waiting_for_lock",
            AgentStage::ApprovalNeeded => "approval_needed",
            AgentStage::Retrying => "retrying",
            AgentStage::Failed => "failed",
            AgentStage::Done => "done",
            AgentStage::Paused => "paused",
            AgentStage::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone)]
struct ExecAgent {
    agent_id: Uuid,
    node_id: Uuid,
    label: String,
    goal: String,
    read_files: Vec<String>,
    write_files: Vec<String>,
    allowed_tools: Vec<String>,
    priority: u32,
}

#[derive(Debug)]
struct AgentExecution {
    agent_id: Uuid,
    label: String,
    write_files: Vec<String>,
    result: odin_core::error::OdinResult<odin_core::types::TaskResult>,
    elapsed: std::time::Duration,
}

#[derive(Clone)]
struct ExecutionResources {
    provider: Option<Arc<dyn Provider>>,
    policy_engine: Option<Arc<odin_permissions::PolicyEngine>>,
    tool_registry: Option<Arc<odin_tools::ToolRegistry>>,
    audit_logger: Option<Arc<dyn AuditLogger>>,
    reliability_tracker: Option<Arc<odin_tools::ReliabilityTracker>>,
    memory: Option<Arc<odin_memory::SqliteMemoryStore>>,
    memory_max_entries: usize,
    model_name: String,
    run_id: String,
}

impl ExecutionResources {
    async fn from_environment() -> Result<Self> {
        let config = load_config(None)?;

        let provider_name = &config.models.default_provider;
        let provider_cfg = config
            .models
            .providers
            .get(provider_name)
            .cloned()
            .unwrap_or_else(default_provider_config);
        let provider =
            odin_providers::create_provider_chain(&provider_cfg, &config.models.providers)
                .or_else(|_| odin_providers::create_provider(&provider_cfg))?;

        let policy_engine = Arc::new(odin_permissions::PolicyEngine::new(
            config.safety.permissions.clone(),
            &config.safety.dangerous_commands,
            config.tools.path_boundary.clone(),
            config.safety.max_rate_per_minute,
            config.safety.require_approval,
        ));

        let sandbox = Arc::new(odin_tools::Sandbox::new(config.tools.path_boundary.clone()));
        let enabled_tools = config.tools.effective_enabled_tools();
        let tool_registry = Arc::new(
            odin_tools::builtin_registry(sandbox, Some(&enabled_tools))
                .map_err(|error| anyhow::anyhow!("failed to build tool registry: {error}"))?,
        );
        load_mcp_tools(&tool_registry, &config).await;

        let audit_logger: Arc<dyn AuditLogger> = Arc::new(build_audit_logger(&config));
        let reliability_path = configured_data_dir(&config).join("reliability.db");
        let reliability_tracker = Arc::new(odin_tools::ReliabilityTracker::persistent(
            &reliability_path,
            odin_tools::ReliabilityConfig::default(),
        )?);
        let model_name = config
            .models
            .default_model
            .clone()
            .or_else(|| provider_cfg.default_model.clone())
            .unwrap_or_default();
        let memory = if config.memory.enabled {
            let path = config.memory.db_path.as_ref().map_or_else(
                || configured_data_dir(&config).join("memory.db"),
                |path| {
                    std::path::PathBuf::from(
                        shellexpand::tilde(&path.to_string_lossy()).to_string(),
                    )
                },
            );
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            Some(Arc::new(odin_memory::SqliteMemoryStore::new(
                &path.to_string_lossy(),
            )?))
        } else {
            None
        };

        Ok(Self {
            provider: Some(provider),
            policy_engine: Some(policy_engine),
            tool_registry: Some(tool_registry),
            audit_logger: Some(audit_logger),
            reliability_tracker: Some(reliability_tracker),
            memory,
            memory_max_entries: config.memory.max_entries.max(1),
            model_name,
            run_id: String::new(),
        })
    }
}

/// Prepare and spawn a real orchestration run, persisting its initial graph
/// before returning so the UI has an authoritative run ID immediately.
pub async fn spawn_run(db_path: PathBuf, goal: String, max_iterations: u32) -> Result<RunHandle> {
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut composer = Composer::new(ComposerConfig {
        max_parallel: 10,
        default_max_iterations: max_iterations,
        auto_merge: true,
        merge_strategy: MergeStrategy::Concatenate,
        workspace_root: ".".to_string(),
        persist_state: true,
        max_retries: 0,
    });
    composer.intake(&goal);

    let prepare_start = Instant::now();
    let graph = composer
        .get_graph(&goal)
        .ok_or_else(|| anyhow::anyhow!("decomposition produced no task graph"))?
        .clone();
    let run_id = graph.id.to_string();
    let exec_agents = register_agents(&mut composer, &goal, &graph, max_iterations);

    let store = SqliteOrchestrationStore::new(db_path.clone()).await?;
    store.initialize().await?;
    persist_graph(&store, &composer, &goal, Some(TaskGraphStatus::Running)).await?;
    persist_agent_lifecycles(
        &store,
        &composer,
        exec_agents.iter().map(|agent| agent.agent_id),
    )
    .await?;
    persist_locks(&store, &composer).await?;
    let (command_tx, command_rx) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();

    let _ = event_tx.send(RunnerEvent::RunStage {
        run_id: run_id.clone(),
        stage: AgentStage::Decomposing,
        detail: format!("decomposed goal into {} task(s)", exec_agents.len()),
        elapsed_ms: prepare_start.elapsed().as_millis() as u64,
    });

    let run_id_for_task = run_id.clone();
    tokio::spawn(async move {
        let _ = event_tx.send(RunnerEvent::RunStage {
            run_id: run_id_for_task.clone(),
            stage: AgentStage::Planning,
            detail: "loading provider, tools, policy, MCP, and audit resources".into(),
            elapsed_ms: 0,
        });
        let resources = match ExecutionResources::from_environment().await {
            Ok(mut resources) => {
                resources.run_id = run_id_for_task.clone();
                resources
            }
            Err(error) => {
                let _ = event_tx.send(RunnerEvent::FatalError {
                    message: odin_permissions::SecretRedactor::full().redact(&format!(
                        "failed to initialize execution resources: {error}"
                    )),
                });
                return;
            }
        };
        if let Err(error) = run_loop(
            db_path,
            goal,
            run_id_for_task,
            max_iterations,
            composer,
            exec_agents,
            resources,
            command_rx,
            event_tx.clone(),
        )
        .await
        {
            let _ = event_tx.send(RunnerEvent::FatalError {
                message: odin_permissions::SecretRedactor::full().redact(&error.to_string()),
            });
        }
    });

    Ok(RunHandle {
        run_id,
        command_tx,
        event_rx,
    })
}

fn register_agents(
    composer: &mut Composer,
    goal_key: &str,
    graph: &TaskGraph,
    max_iterations: u32,
) -> Vec<ExecAgent> {
    let mut nodes: Vec<_> = graph.nodes.values().cloned().collect();
    nodes.sort_by_key(|node| (node.priority, node.label.clone()));

    nodes
        .into_iter()
        .map(|node| {
            let mut agent_config = composer.create_sub_agent(&node);
            agent_config.max_iterations = max_iterations;
            let allowed_tools = agent_config.allowed_tools.clone();
            let agent_id = composer.register_agent(agent_config);
            composer.update_node_status(goal_key, node.id, TaskNodeStatus::Pending);
            ExecAgent {
                agent_id,
                node_id: node.id,
                label: node.label,
                goal: node.goal,
                read_files: node.read_files,
                write_files: node.write_files,
                allowed_tools,
                priority: node.priority,
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
async fn run_loop(
    db_path: PathBuf,
    goal_key: String,
    run_id: String,
    max_iterations: u32,
    mut composer: Composer,
    mut exec_agents: Vec<ExecAgent>,
    resources: ExecutionResources,
    mut command_rx: mpsc::UnboundedReceiver<RunnerCommand>,
    event_tx: mpsc::UnboundedSender<RunnerEvent>,
) -> Result<()> {
    let store = SqliteOrchestrationStore::new(db_path).await?;
    store.initialize().await?;

    let task_count = exec_agents.len();
    let run_start = Instant::now();
    let _ = event_tx.send(RunnerEvent::RunStarted {
        run_id: run_id.clone(),
        goal: goal_key.clone(),
        task_count,
    });
    let _ = event_tx.send(RunnerEvent::RunStage {
        run_id: run_id.clone(),
        stage: AgentStage::SpawningAgents,
        detail: format!("spawning up to {task_count} ready agent(s)"),
        elapsed_ms: run_start.elapsed().as_millis() as u64,
    });

    let mut join_set: JoinSet<AgentExecution> = JoinSet::new();
    let mut spawned = HashSet::<Uuid>::new();
    let mut terminal = HashSet::<Uuid>::new();
    let mut paused = false;
    let mut cancelled = false;

    loop {
        // Live cross-process controls from CLI/WebSocket.
        if let Ok(commands) = store.claim_pending_controls(&run_id).await {
            for command in commands {
                match command.kind {
                    odin_orchestrator::RunControlKind::Pause => {
                        paused = true;
                        composer.set_graph_status(&goal_key, TaskGraphStatus::Paused);
                        let _ = event_tx.send(RunnerEvent::RunStage {
                            run_id: run_id.clone(),
                            stage: AgentStage::Paused,
                            detail: format!("live pause from {}", command.source),
                            elapsed_ms: run_start.elapsed().as_millis() as u64,
                        });
                    }
                    odin_orchestrator::RunControlKind::Resume => {
                        paused = false;
                        composer.set_graph_status(&goal_key, TaskGraphStatus::Running);
                        let _ = event_tx.send(RunnerEvent::RunStage {
                            run_id: run_id.clone(),
                            stage: AgentStage::SpawningAgents,
                            detail: format!("live resume from {}", command.source),
                            elapsed_ms: run_start.elapsed().as_millis() as u64,
                        });
                    }
                    odin_orchestrator::RunControlKind::Cancel => {
                        cancelled = true;
                        composer.set_graph_status(&goal_key, TaskGraphStatus::Cancelled);
                        let _ = event_tx.send(RunnerEvent::RunStage {
                            run_id: run_id.clone(),
                            stage: AgentStage::Cancelled,
                            detail: format!("live cancel from {}", command.source),
                            elapsed_ms: run_start.elapsed().as_millis() as u64,
                        });
                    }
                }
                store.mark_control_applied(command.id).await?;
            }
        }

        while let Ok(command) = command_rx.try_recv() {
            handle_command(
                command,
                &store,
                &mut composer,
                &goal_key,
                &run_id,
                &mut exec_agents,
                &mut paused,
                &mut cancelled,
                &mut join_set,
                &event_tx,
            )
            .await?;
        }

        if cancelled {
            composer.cancel_all("Cancelled from TUI");
            persist_all(
                &store,
                &composer,
                &goal_key,
                Some(TaskGraphStatus::Cancelled),
                exec_agents.iter().map(|agent| agent.agent_id),
            )
            .await?;
            let _ = event_tx.send(RunnerEvent::RunCancelled {
                run_id: run_id.clone(),
            });
            return Ok(());
        }

        if !paused {
            start_ready_agents(
                &store,
                &mut composer,
                &goal_key,
                &exec_agents,
                max_iterations,
                &resources,
                &mut spawned,
                &mut join_set,
                &event_tx,
            )
            .await?;
        }

        if terminal.len() == exec_agents.len() && join_set.is_empty() {
            break;
        }

        tokio::select! {
            Some(command) = command_rx.recv() => {
                handle_command(
                    command,
                    &store,
                    &mut composer,
                    &goal_key,
                    &run_id,
                    &mut exec_agents,
                    &mut paused,
                    &mut cancelled,
                    &mut join_set,
                    &event_tx,
                ).await?;
            }
            Some(joined) = join_set.join_next(), if !join_set.is_empty() => {
                match joined {
                    Ok(execution) => {
                        finish_agent_execution(
                            &store,
                            &mut composer,
                            &goal_key,
                            &execution,
                            &mut terminal,
                            &event_tx,
                        ).await?;
                    }
                    Err(error) => {
                        // The agent body catches panics itself. Any JoinError here
                        // means the outer wrapper failed, so abort and terminalize
                        // every in-flight agent before scheduling more work.
                        let message = format!("sub-agent join failed: {error}");
                        let _ = event_tx.send(RunnerEvent::Error {
                            message: message.clone(),
                        });
                        join_set.abort_all();
                        while join_set.join_next().await.is_some() {}

                        let stuck: Vec<_> = exec_agents
                            .iter()
                            .filter(|agent| {
                                spawned.contains(&agent.agent_id)
                                    && !terminal.contains(&agent.agent_id)
                            })
                            .cloned()
                            .collect();
                        for agent in stuck {
                            let execution = AgentExecution {
                                agent_id: agent.agent_id,
                                label: agent.label.clone(),
                                write_files: agent.write_files.clone(),
                                result: Err(odin_core::error::OdinError::Internal(
                                    message.clone(),
                                )),
                                elapsed: Duration::ZERO,
                            };
                            finish_agent_execution(
                                &store,
                                &mut composer,
                                &goal_key,
                                &execution,
                                &mut terminal,
                                &event_tx,
                            )
                            .await?;
                        }
                    }
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {}
        }
    }

    let results = composer.collect_results();
    let merged = composer.merge_results(results);
    let status = if merged.success {
        TaskGraphStatus::Complete
    } else {
        TaskGraphStatus::Failed
    };
    persist_all(
        &store,
        &composer,
        &goal_key,
        Some(status),
        exec_agents.iter().map(|agent| agent.agent_id),
    )
    .await?;

    let _ = event_tx.send(RunnerEvent::RunFinished {
        run_id,
        success: merged.success,
        summary: merged.summary,
    });

    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn handle_command(
    command: RunnerCommand,
    store: &SqliteOrchestrationStore,
    composer: &mut Composer,
    goal_key: &str,
    run_id: &str,
    exec_agents: &mut [ExecAgent],
    paused: &mut bool,
    cancelled: &mut bool,
    join_set: &mut JoinSet<AgentExecution>,
    event_tx: &mpsc::UnboundedSender<RunnerEvent>,
) -> Result<()> {
    match command {
        RunnerCommand::Pause => {
            *paused = true;
            composer.set_graph_status(goal_key, TaskGraphStatus::Paused);
            persist_all(
                store,
                composer,
                goal_key,
                Some(TaskGraphStatus::Paused),
                exec_agents.iter().map(|agent| agent.agent_id),
            )
            .await?;
            let _ = event_tx.send(RunnerEvent::RunPaused {
                run_id: run_id.to_string(),
            });
            for agent in exec_agents.iter() {
                let _ = event_tx.send(RunnerEvent::AgentStage {
                    agent_id: agent.agent_id.to_string(),
                    label: agent.label.clone(),
                    stage: AgentStage::Paused,
                    detail: "pause requested from TUI".into(),
                    elapsed_ms: 0,
                });
            }
        }
        RunnerCommand::Resume => {
            *paused = false;
            composer.set_graph_status(goal_key, TaskGraphStatus::Running);
            persist_all(
                store,
                composer,
                goal_key,
                Some(TaskGraphStatus::Running),
                exec_agents.iter().map(|agent| agent.agent_id),
            )
            .await?;
            let _ = event_tx.send(RunnerEvent::RunResumed {
                run_id: run_id.to_string(),
            });
            for agent in exec_agents.iter() {
                let _ = event_tx.send(RunnerEvent::AgentStage {
                    agent_id: agent.agent_id.to_string(),
                    label: agent.label.clone(),
                    stage: AgentStage::Retrying,
                    detail: "resume requested from TUI".into(),
                    elapsed_ms: 0,
                });
            }
        }
        RunnerCommand::Cancel => {
            *cancelled = true;
            join_set.abort_all();
            for agent in exec_agents.iter() {
                let _ = event_tx.send(RunnerEvent::AgentStage {
                    agent_id: agent.agent_id.to_string(),
                    label: agent.label.clone(),
                    stage: AgentStage::Cancelled,
                    detail: "cancel requested from TUI; aborting active work".into(),
                    elapsed_ms: 0,
                });
            }
        }
        RunnerCommand::Redirect(message) => {
            apply_redirect(composer, goal_key, exec_agents, &message);
            persist_graph(store, composer, goal_key, Some(TaskGraphStatus::Running)).await?;
            let _ = event_tx.send(RunnerEvent::RunRedirected {
                run_id: run_id.to_string(),
                message,
            });
        }
        RunnerCommand::Reprioritise {
            agent_id_prefix,
            priority,
        } => {
            let matches: Vec<Uuid> = exec_agents
                .iter()
                .filter(|agent| agent.agent_id.to_string().starts_with(&agent_id_prefix))
                .map(|agent| agent.agent_id)
                .collect();
            for agent_id in matches {
                if composer.reprioritize(agent_id, priority).is_ok() {
                    if let Some(agent) = exec_agents
                        .iter_mut()
                        .find(|agent| agent.agent_id == agent_id)
                    {
                        agent.priority = priority;
                    }
                    let _ = event_tx.send(RunnerEvent::Reprioritised {
                        agent_id: agent_id.to_string(),
                        priority,
                    });
                }
            }
            persist_graph(store, composer, goal_key, Some(TaskGraphStatus::Running)).await?;
        }
    }
    Ok(())
}

fn apply_redirect(
    composer: &mut Composer,
    goal_key: &str,
    exec_agents: &mut [ExecAgent],
    message: &str,
) {
    let note = format!("\n\nUser steering update: {message}");
    let pending_ids: HashSet<Uuid> = composer
        .get_graph(goal_key)
        .map(|graph| {
            graph
                .nodes
                .values()
                .filter(|node| {
                    matches!(
                        node.status,
                        TaskNodeStatus::Pending | TaskNodeStatus::Blocked
                    )
                })
                .map(|node| node.id)
                .collect()
        })
        .unwrap_or_default();

    for agent in exec_agents
        .iter_mut()
        .filter(|agent| pending_ids.contains(&agent.node_id))
    {
        agent.goal.push_str(&note);
    }
}

#[allow(clippy::too_many_arguments)]
async fn start_ready_agents(
    store: &SqliteOrchestrationStore,
    composer: &mut Composer,
    goal_key: &str,
    exec_agents: &[ExecAgent],
    max_iterations: u32,
    resources: &ExecutionResources,
    spawned: &mut HashSet<Uuid>,
    join_set: &mut JoinSet<AgentExecution>,
    event_tx: &mpsc::UnboundedSender<RunnerEvent>,
) -> Result<()> {
    let mut ordered: Vec<_> = exec_agents
        .iter()
        .filter(|agent| !spawned.contains(&agent.agent_id))
        .collect();
    ordered.sort_by_key(|agent| (agent.priority, agent.label.clone()));

    for agent in ordered {
        let (phase, lifecycle_before) = composer
            .get_agent(&agent.agent_id)
            .map(|(_, lifecycle)| {
                (
                    lifecycle.phase,
                    (lifecycle.phase_changed_at, lifecycle.held_locks.clone()),
                )
            })
            .unwrap_or((AgentPhase::Cancelled, (chrono::Utc::now(), Vec::new())));

        match phase {
            AgentPhase::Queued | AgentPhase::WaitingForLock => {
                match composer.start_agent(agent.agent_id) {
                    Ok(()) => {
                        spawn_agent_execution(
                            agent.clone(),
                            max_iterations,
                            resources.clone(),
                            spawned,
                            join_set,
                            event_tx.clone(),
                        );
                        if let Some((_, lifecycle)) = composer.get_agent(&agent.agent_id) {
                            store.save_agent_lifecycle(lifecycle).await?;
                        }
                        persist_graph(store, composer, goal_key, Some(TaskGraphStatus::Running))
                            .await?;
                        persist_locks(store, composer).await?;
                        let _ = event_tx.send(RunnerEvent::AgentStarted {
                            agent_id: agent.agent_id.to_string(),
                            label: agent.label.clone(),
                            task: agent.goal.clone(),
                        });
                    }
                    Err(reason) => {
                        // Retry frequently, but persist only genuine state changes
                        // and announce only the initial transition into waiting.
                        let lifecycle_changed = composer
                            .get_agent(&agent.agent_id)
                            .map(|(_, lifecycle)| {
                                (lifecycle.phase_changed_at, lifecycle.held_locks.clone())
                                    != lifecycle_before
                            })
                            .unwrap_or(false);
                        if phase != AgentPhase::WaitingForLock || lifecycle_changed {
                            if let Some((_, lifecycle)) = composer.get_agent(&agent.agent_id) {
                                store.save_agent_lifecycle(lifecycle).await?;
                            }
                            persist_graph(
                                store,
                                composer,
                                goal_key,
                                Some(TaskGraphStatus::Running),
                            )
                            .await?;
                            persist_locks(store, composer).await?;
                        }
                        if phase != AgentPhase::WaitingForLock {
                            let _ = event_tx.send(RunnerEvent::AgentQueued {
                                agent_id: agent.agent_id.to_string(),
                                label: agent.label.clone(),
                                reason,
                            });
                            let _ = event_tx.send(RunnerEvent::AgentStage {
                                agent_id: agent.agent_id.to_string(),
                                label: agent.label.clone(),
                                stage: AgentStage::WaitingForLock,
                                detail: "waiting for file lock or dependency".into(),
                                elapsed_ms: 0,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Ok(())
}

async fn catch_agent_panic<F, T>(future: F) -> odin_core::error::OdinResult<T>
where
    F: Future<Output = odin_core::error::OdinResult<T>>,
{
    match AssertUnwindSafe(future).catch_unwind().await {
        Ok(result) => result,
        Err(_) => Err(odin_core::error::OdinError::Internal(
            "sub-agent task panicked".into(),
        )),
    }
}

fn spawn_agent_execution(
    agent: ExecAgent,
    max_iterations: u32,
    resources: ExecutionResources,
    spawned: &mut HashSet<Uuid>,
    join_set: &mut JoinSet<AgentExecution>,
    event_tx: mpsc::UnboundedSender<RunnerEvent>,
) {
    spawned.insert(agent.agent_id);
    join_set.spawn(async move {
        let start = std::time::Instant::now();
        let agent_id = agent.agent_id;
        let label = agent.label.clone();
        let write_files = agent.write_files.clone();

        // Catch panics in the same future so aborting the outer JoinSet task
        // still drops provider/tool work immediately.
        let result = catch_agent_panic(async move {
            let _ = event_tx.send(RunnerEvent::AgentStage {
                agent_id: agent.agent_id.to_string(),
                label: agent.label.clone(),
                stage: AgentStage::Planning,
                detail: "agent loop starting".into(),
                elapsed_ms: 0,
            });
            let context = {
                let files_context = format!(
                    "Read files: {}\nWrite files: {}",
                    if agent.read_files.is_empty() {
                        "-".to_string()
                    } else {
                        agent.read_files.join(", ")
                    },
                    if agent.write_files.is_empty() {
                        "-".to_string()
                    } else {
                        agent.write_files.join(", ")
                    }
                );
                let mut parts = vec![files_context];
                if let Some(memory) = resources.memory.as_ref()
                    && let Ok(Some(memory_context)) = odin_memory::retrieve_task_context(
                        memory.as_ref(),
                        &agent.goal,
                        resources.memory_max_entries.min(8),
                        odin_memory::DEFAULT_CONTEXT_CHARS,
                    )
                    .await
                {
                    parts.push(memory_context);
                }
                Some(parts.join("\n\n"))
            };
            let task_id = Uuid::new_v4();
            let task = AgentTask {
                id: task_id,
                goal: agent.goal.clone(),
                context,
                sub_tasks: vec![],
                success_criteria: vec![],
                max_iterations,
                created_at: chrono::Utc::now(),
            };
            let mut engine = odin_loop::LoopEngine::new().with_max_iterations(max_iterations);
            if let Some(provider) = resources.provider.clone() {
                engine = engine.with_provider(Arc::new(ProgressProvider::new(
                    provider,
                    agent.agent_id,
                    agent.label.clone(),
                    event_tx.clone(),
                )));
            }
            if !resources.model_name.is_empty() {
                engine = engine.with_model_name(resources.model_name.clone());
            }
            if let Some(policy_engine) = resources.policy_engine.clone() {
                engine = engine.with_policy_engine(policy_engine);
            }
            if let Some(registry) = resources.tool_registry.clone() {
                let scoped = registry.scoped(&agent.allowed_tools).map_err(|error| {
                    odin_core::error::OdinError::Validation(format!(
                        "invalid tool scope for agent '{}': {error}",
                        agent.label
                    ))
                })?;
                engine = engine.with_tool_registry(Arc::new(scoped));
            }
            if let Some(audit_logger) = resources.audit_logger.clone() {
                engine = engine.with_audit_logger(Arc::new(ProgressAuditLogger::new(
                    audit_logger,
                    agent.agent_id,
                    agent.label.clone(),
                    event_tx.clone(),
                )));
            }
            if let Some(reliability_tracker) = resources.reliability_tracker.clone() {
                engine = engine.with_reliability_tracker(reliability_tracker);
            }
            let result = engine.execute_task(&task).await;
            if let (Some(memory), Ok(task_result)) = (resources.memory.as_ref(), result.as_ref()) {
                let redacted =
                    odin_permissions::SecretRedactor::full().redact(&task_result.summary);
                let _ = odin_memory::store_task_outcome(
                    memory.as_ref(),
                    redacted,
                    &resources.run_id,
                    &task_id.to_string(),
                    &agent.agent_id.to_string(),
                    if task_result.success { 0.7 } else { 0.3 },
                )
                .await;
            }
            let elapsed_ms = start.elapsed().as_millis() as u64;
            let stage = if result.as_ref().map(|r| r.success).unwrap_or(false) {
                AgentStage::Done
            } else {
                AgentStage::Failed
            };
            let detail = match &result {
                Ok(result) => result.summary.clone(),
                Err(error) => error.to_string(),
            };
            let _ = event_tx.send(RunnerEvent::AgentStage {
                agent_id: agent.agent_id.to_string(),
                label: agent.label.clone(),
                stage,
                detail: odin_permissions::SecretRedactor::full().redact(&detail),
                elapsed_ms,
            });
            result
        })
        .await;
        AgentExecution {
            agent_id,
            label,
            write_files,
            result,
            elapsed: start.elapsed(),
        }
    });
}

struct ProgressProvider {
    inner: Arc<dyn Provider>,
    agent_id: Uuid,
    label: String,
    event_tx: mpsc::UnboundedSender<RunnerEvent>,
}

impl ProgressProvider {
    fn new(
        inner: Arc<dyn Provider>,
        agent_id: Uuid,
        label: String,
        event_tx: mpsc::UnboundedSender<RunnerEvent>,
    ) -> Self {
        Self {
            inner,
            agent_id,
            label,
            event_tx,
        }
    }

    fn send_stage(&self, stage: AgentStage, detail: impl Into<String>, elapsed: Duration) {
        let _ = self.event_tx.send(RunnerEvent::AgentStage {
            agent_id: self.agent_id.to_string(),
            label: self.label.clone(),
            stage,
            detail: odin_permissions::SecretRedactor::full().redact(&detail.into()),
            elapsed_ms: elapsed.as_millis() as u64,
        });
    }
}

#[async_trait]
impl Provider for ProgressProvider {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn list_models(&self) -> odin_core::error::OdinResult<Vec<ModelInfo>> {
        self.inner.list_models().await
    }

    async fn chat(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolSchema],
        options: &CompletionOptions,
    ) -> odin_core::error::OdinResult<ChatResponse> {
        let call_kind = classify_model_call(messages, tools);
        let started = Instant::now();
        self.send_stage(
            AgentStage::WaitingForModel,
            format!("model call started: {call_kind}"),
            Duration::ZERO,
        );

        let heartbeat = tokio::time::sleep(Duration::from_secs(1));
        tokio::pin!(heartbeat);
        let call = self.inner.chat(model, messages, tools, options);
        tokio::pin!(call);

        loop {
            tokio::select! {
                result = &mut call => {
                    match &result {
                        Ok(response) if response.message.tool_calls().is_empty() => {
                            self.send_stage(
                                AgentStage::WaitingForModel,
                                format!("model response received for {call_kind}"),
                                started.elapsed(),
                            );
                        }
                        Ok(response) => {
                            let names = response.message.tool_calls()
                                .iter()
                                .map(|call| call.function.name.as_str())
                                .collect::<Vec<_>>()
                                .join(", ");
                            self.send_stage(
                                AgentStage::RunningTool,
                                format!("model requested tool call(s): {names}"),
                                started.elapsed(),
                            );
                        }
                        Err(error) => {
                            self.send_stage(
                                AgentStage::Failed,
                                format!("model call failed during {call_kind}: {error}"),
                                started.elapsed(),
                            );
                        }
                    }
                    return result;
                }
                _ = &mut heartbeat => {
                    let elapsed = started.elapsed();
                    let seconds = elapsed.as_secs();
                    let detail = if seconds >= 10 {
                        format!("waiting for model... {seconds}s elapsed ({call_kind})")
                    } else {
                        format!("model call in progress... {seconds}s elapsed ({call_kind})")
                    };
                    self.send_stage(AgentStage::WaitingForModel, detail, elapsed);
                    heartbeat.as_mut().reset(tokio::time::Instant::now() + Duration::from_secs(1));
                }
            }
        }
    }

    async fn chat_stream(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolSchema],
        options: &CompletionOptions,
    ) -> odin_core::error::OdinResult<Box<dyn ChatStream>> {
        self.inner
            .chat_stream(model, messages, tools, options)
            .await
    }

    async fn health_check(&self) -> odin_core::error::OdinResult<bool> {
        self.inner.health_check().await
    }
}

struct ProgressAuditLogger {
    inner: Arc<dyn AuditLogger>,
    agent_id: Uuid,
    label: String,
    event_tx: mpsc::UnboundedSender<RunnerEvent>,
}

impl ProgressAuditLogger {
    fn new(
        inner: Arc<dyn AuditLogger>,
        agent_id: Uuid,
        label: String,
        event_tx: mpsc::UnboundedSender<RunnerEvent>,
    ) -> Self {
        Self {
            inner,
            agent_id,
            label,
            event_tx,
        }
    }

    fn send_tool_stage(&self, entry: &AuditEntry) {
        if entry.event_type != AuditEventType::ToolCall {
            return;
        }

        let (stage, detail) = match entry.result {
            AuditResult::Success => (
                AgentStage::RunningTool,
                format!("tool '{}' completed", entry.action),
            ),
            AuditResult::Failure | AuditResult::Denied => (
                AgentStage::Failed,
                format!("tool '{}' failed: {}", entry.action, entry.details),
            ),
            AuditResult::Pending => (
                AgentStage::ApprovalNeeded,
                format!("tool '{}' awaiting approval", entry.action),
            ),
        };

        let _ = self.event_tx.send(RunnerEvent::AgentStage {
            agent_id: self.agent_id.to_string(),
            label: self.label.clone(),
            stage,
            detail: odin_permissions::SecretRedactor::full().redact(&detail),
            elapsed_ms: entry
                .details
                .get("duration_ms")
                .and_then(|value| value.as_u64())
                .unwrap_or(0),
        });
    }
}

#[async_trait]
impl AuditLogger for ProgressAuditLogger {
    async fn log(&self, entry: AuditEntry) -> odin_core::error::OdinResult<()> {
        self.send_tool_stage(&entry);
        self.inner.log(entry).await
    }

    async fn query(
        &self,
        agent_id: Option<AgentId>,
        session_id: Option<SessionId>,
        event_type: Option<AuditEventType>,
        limit: usize,
    ) -> odin_core::error::OdinResult<Vec<AuditEntry>> {
        self.inner
            .query(agent_id, session_id, event_type, limit)
            .await
    }

    async fn recent(&self, limit: usize) -> odin_core::error::OdinResult<Vec<AuditEntry>> {
        self.inner.recent(limit).await
    }
}

fn classify_model_call(messages: &[Message], tools: &[ToolSchema]) -> &'static str {
    let last = messages
        .iter()
        .rev()
        .filter_map(Message::text)
        .find(|text| !text.trim().is_empty())
        .unwrap_or_default()
        .to_ascii_lowercase();

    if last.contains("break this goal into sub-tasks") {
        "planning/decomposition"
    } else if last.contains("decide what action to take") {
        if tools.is_empty() {
            "action planning"
        } else {
            "action/tool selection"
        }
    } else if last.contains("critique") || last.contains("self-evaluat") {
        "critique"
    } else if last.contains("revise") {
        "retry/revision"
    } else if last.contains("verify") {
        "verification"
    } else {
        "model request"
    }
}

async fn finish_agent_execution(
    store: &SqliteOrchestrationStore,
    composer: &mut Composer,
    goal_key: &str,
    execution: &AgentExecution,
    terminal: &mut HashSet<Uuid>,
    event_tx: &mpsc::UnboundedSender<RunnerEvent>,
) -> Result<()> {
    match &execution.result {
        Ok(task_result) => {
            composer.complete_agent(
                execution.agent_id,
                SubAgentResult {
                    agent_id: execution.agent_id,
                    name: execution.label.clone(),
                    summary: task_result.summary.clone(),
                    output: Some(task_result.summary.clone()),
                    modified_files: execution.write_files.clone(),
                    success: task_result.success,
                    error: task_result.error.clone(),
                    duration_ms: execution.elapsed.as_millis() as u64,
                },
            );
            let _ = event_tx.send(RunnerEvent::AgentFinished {
                agent_id: execution.agent_id.to_string(),
                label: execution.label.clone(),
                success: task_result.success,
                summary: task_result.summary.clone(),
            });
        }
        Err(error) => {
            let error = odin_permissions::SecretRedactor::full().redact(&error.to_string());
            composer.fail_agent(execution.agent_id, error.clone());
            let _ = event_tx.send(RunnerEvent::AgentFinished {
                agent_id: execution.agent_id.to_string(),
                label: execution.label.clone(),
                success: false,
                summary: error,
            });
        }
    }
    terminal.insert(execution.agent_id);
    if let Some((_, lifecycle)) = composer.get_agent(&execution.agent_id) {
        store.save_agent_lifecycle(lifecycle).await?;
    }
    persist_graph(store, composer, goal_key, None).await?;
    persist_locks(store, composer).await?;
    Ok(())
}

async fn persist_all(
    store: &SqliteOrchestrationStore,
    composer: &Composer,
    goal_key: &str,
    status_override: Option<TaskGraphStatus>,
    agent_ids: impl Iterator<Item = Uuid>,
) -> Result<()> {
    persist_graph(store, composer, goal_key, status_override).await?;
    persist_agent_lifecycles(store, composer, agent_ids).await?;
    persist_locks(store, composer).await?;
    Ok(())
}

async fn persist_graph(
    store: &SqliteOrchestrationStore,
    composer: &Composer,
    goal_key: &str,
    status_override: Option<TaskGraphStatus>,
) -> Result<()> {
    if let Some(graph) = composer.get_graph(goal_key) {
        let mut graph = graph.clone();
        if let Some(status) = status_override {
            graph.status = status;
        } else if !matches!(
            graph.status,
            TaskGraphStatus::Complete | TaskGraphStatus::Failed | TaskGraphStatus::Cancelled
        ) {
            graph.status = TaskGraphStatus::Running;
        }
        store.save_task_graph(&graph).await?;
    }
    Ok(())
}

async fn persist_agent_lifecycles(
    store: &SqliteOrchestrationStore,
    composer: &Composer,
    agent_ids: impl Iterator<Item = Uuid>,
) -> Result<()> {
    for agent_id in agent_ids {
        if let Some((_, lifecycle)) = composer.get_agent(&agent_id) {
            store.save_agent_lifecycle(lifecycle).await?;
        }
    }
    Ok(())
}

async fn persist_locks(store: &SqliteOrchestrationStore, composer: &Composer) -> Result<()> {
    let snapshot = serde_json::to_string(&composer.file_locks().snapshot())?;
    store.save_lock_snapshot(&snapshot).await?;
    Ok(())
}

fn default_provider_config() -> odin_core::config::ProviderConfig {
    odin_core::config::ProviderConfig {
        provider_type: "openai_compat".into(),
        base_url: Some("http://localhost:11434/v1".into()),
        api_key: None,
        api_key_env: None,
        default_model: None,
        headers: Default::default(),
        timeout_secs: 120,
        max_retries: 3,
        fallback_chain: None,
        health_check_interval_secs: 0,
        circuit_breaker_threshold: 0,
    }
}

fn load_config(path: Option<&std::path::Path>) -> Result<OdinConfig> {
    match path {
        Some(path) if path.exists() => OdinConfig::load(path).map_err(|error| {
            anyhow::anyhow!("failed to load Raven config '{}': {error}", path.display())
        }),
        Some(path) => anyhow::bail!(
            "config file '{}' does not exist; create it with 'raven config' or pass an existing path",
            path.display()
        ),
        None => {
            if let Some(path) = std::env::var_os("RAVEN_CONFIG")
                .or_else(|| std::env::var_os("ODIN_CONFIG"))
                .map(PathBuf::from)
            {
                return load_config(Some(&path));
            }

            for path in default_config_paths() {
                if path.exists() {
                    return load_config(Some(&path));
                }
            }
            Ok(OdinConfig::default())
        }
    }
}

fn default_config_paths() -> Vec<PathBuf> {
    [
        "~/.config/raven/config.yaml",
        "~/.raven-agent/config.yaml",
        "~/.odin/config.yaml",
        "raven.yaml",
        "raven.yml",
        "odin.yaml",
        "odin.yml",
    ]
    .into_iter()
    .map(expand_path)
    .collect()
}

fn expand_path(path: &str) -> PathBuf {
    PathBuf::from(shellexpand::tilde(path).to_string())
}

fn expand_config_path(path: &std::path::Path) -> PathBuf {
    expand_path(&path.to_string_lossy())
}

fn configured_data_dir(config: &OdinConfig) -> PathBuf {
    config.general.data_dir.as_ref().map_or_else(
        || expand_path("~/.raven-agent"),
        |path| expand_config_path(path),
    )
}

fn configured_audit_path(config: &OdinConfig) -> PathBuf {
    config.audit.log_path.as_ref().map_or_else(
        || configured_data_dir(config).join("audit.jsonl"),
        |path| expand_config_path(path),
    )
}

fn build_audit_logger(config: &OdinConfig) -> odin_audit::AuditLoggerImpl {
    odin_audit::AuditLoggerImpl::new(odin_audit::AuditLoggerConfig {
        enabled: config.audit.enabled,
        file_path: config
            .audit
            .enabled
            .then_some(configured_audit_path(config)),
        db_path: None,
        json_format: config.audit.json_format,
        buffer_size: 100,
        mask_secrets: true,
    })
}

async fn load_mcp_tools(registry: &odin_tools::ToolRegistry, config: &OdinConfig) {
    use odin_core::traits::Tool;
    use odin_mcp::client::McpClient;
    use odin_mcp::tool_adapter::McpToolAdapter;
    use odin_mcp::transport::StdioTransport;
    use tokio::sync::Mutex;

    for server_cfg in &config.tools.mcp_servers {
        if !server_cfg.enabled {
            continue;
        }
        if server_cfg.transport_type != "stdio" {
            tracing::warn!(
                "[TUI/MCP] Skipping server '{}': unsupported transport '{}'",
                server_cfg.name,
                server_cfg.transport_type
            );
            continue;
        }

        let transport: Arc<Mutex<dyn odin_mcp::transport::McpTransport>> = Arc::new(Mutex::new(
            StdioTransport::new(&server_cfg.command, server_cfg.args.clone())
                .with_env(server_cfg.env.clone()),
        ));

        let mut client = McpClient::new(transport);
        let shared_client = match client.connect().await {
            Ok(()) => Arc::new(Mutex::new(client)),
            Err(error) => {
                tracing::warn!(
                    "[TUI/MCP] Failed to connect to server '{}': {}",
                    server_cfg.name,
                    error
                );
                continue;
            }
        };

        let tools = {
            let client = shared_client.lock().await;
            client.list_tools().await
        };

        let Ok(tools) = tools else {
            tracing::warn!("[TUI/MCP] Failed to list tools from '{}'", server_cfg.name);
            continue;
        };

        for tool_def in tools {
            let adapter = McpToolAdapter::new_with_tags(
                tool_def,
                shared_client.clone(),
                server_cfg.tags.clone(),
            )
            .with_approval(server_cfg.requires_approval)
            .with_safety(server_cfg.safe);
            let name = adapter.name().to_string();
            if config.tools.disabled.contains(&name) {
                continue;
            }
            if let Err(error) = registry.register(Box::new(adapter)) {
                tracing::warn!("[TUI/MCP] Failed to register tool '{}': {}", name, error);
            }
        }
    }
}

#[cfg(test)]
mod runner_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct DropFlag(Arc<AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn test_agent_panic_is_converted_to_internal_error() {
        let result: odin_core::error::OdinResult<()> = catch_agent_panic(async {
            panic!("agent panic");
            #[allow(unreachable_code)]
            Ok(())
        })
        .await;

        assert!(matches!(
            result,
            Err(odin_core::error::OdinError::Internal(message))
                if message.contains("panicked")
        ));
    }

    #[tokio::test]
    async fn test_aborting_agent_wrapper_drops_in_flight_work() {
        let dropped = Arc::new(AtomicBool::new(false));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let dropped_in_task = dropped.clone();

        let handle = tokio::spawn(catch_agent_panic(async move {
            let _drop_flag = DropFlag(dropped_in_task);
            let _ = started_tx.send(());
            std::future::pending::<odin_core::error::OdinResult<()>>().await
        }));

        started_rx.await.unwrap();
        handle.abort();
        assert!(handle.await.unwrap_err().is_cancelled());
        assert!(dropped.load(Ordering::SeqCst));
    }
}
