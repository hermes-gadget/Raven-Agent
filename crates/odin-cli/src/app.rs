//! Raven Agent CLI — Multi-agent AI orchestration platform.
//!
//! Usage:
//!   raven                      — Open the interactive terminal UI
//!   raven run <task>           — Execute with orchestration by default
//!   raven run --direct <task>  — Execute with one agent
//!   raven serve [--addr]       — Start the HTTP API server
//!   raven --help               — Show the complete command list

use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use odin_core::config::OdinConfig;
use odin_core::traits::{AuditLogger, LoopEngine, Tool};
use odin_core::types::AgentTask;
use odin_orchestrator::Composer;
use odin_orchestrator::persistence::OrchestrationStore;
use odin_runtime::{
    CompositionResources, EngineBuildOptions, ExecutionSurface, ProductionComposition, Runtime,
};
use tracing_subscriber::EnvFilter;

// Scheduler types
use odin_scheduler::{JobId, Scheduler, SchedulerJobConfig, SqliteSchedulerStore};

type AgentTaskOutput = (
    uuid::Uuid,
    String,
    odin_core::error::OdinResult<odin_core::types::TaskResult>,
    std::time::Duration,
);

/// Tracks orchestration tasks by agent and yields them in completion order.
/// Dropping this set aborts every remaining task, preventing detached work on errors.
struct AgentTaskSet {
    tasks: tokio::task::JoinSet<AgentTaskOutput>,
    agents_by_task: std::collections::HashMap<tokio::task::Id, uuid::Uuid>,
}

/// Owns the terminal approval responder for the lifetime of a CLI run.
struct CliApprovalResponder {
    gate: Arc<odin_permissions::ApprovalGate>,
    task: tokio::task::JoinHandle<()>,
}

impl CliApprovalResponder {
    fn spawn(gate: Arc<odin_permissions::ApprovalGate>) -> Self {
        let mut requests = gate.subscribe();
        let responder_gate = gate.clone();
        let task = tokio::spawn(async move {
            loop {
                let request = match requests.recv().await {
                    Ok(request) => request,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        responder_gate.disconnect();
                        break;
                    }
                };

                let current = responder_gate.get_request(&request.id).await;
                if !current.is_some_and(|current| {
                    current.status == odin_permissions::ApprovalStatus::Pending
                }) {
                    continue;
                }

                let prompt_request = request.clone();
                let decision = tokio::task::spawn_blocking(move || {
                    use std::io::Write;

                    println!();
                    println!("Approval required [{}]", prompt_request.id);
                    println!("Action: {}", prompt_request.action);
                    println!("Arguments: {}", prompt_request.details);
                    print!("Approve this exact call? [y/N] ");
                    if std::io::stdout().flush().is_err() {
                        return None;
                    }
                    let mut input = String::new();
                    match std::io::stdin().read_line(&mut input) {
                        Ok(0) | Err(_) => None,
                        Ok(_) => Some(matches!(
                            input.trim().to_ascii_lowercase().as_str(),
                            "y" | "yes"
                        )),
                    }
                })
                .await
                .unwrap_or(Some(false));

                match decision {
                    Some(true) => {
                        let _ = responder_gate
                            .approve(&request.id, &request.argument_fingerprint)
                            .await;
                    }
                    Some(false) => {
                        let _ = responder_gate.deny(&request.id).await;
                    }
                    None => {
                        responder_gate.disconnect();
                        break;
                    }
                }
            }
        });
        Self { gate, task }
    }
}

impl Drop for CliApprovalResponder {
    fn drop(&mut self) {
        self.gate.disconnect();
        self.task.abort();
    }
}

impl AgentTaskSet {
    fn new() -> Self {
        Self {
            tasks: tokio::task::JoinSet::new(),
            agents_by_task: std::collections::HashMap::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    fn spawn<F>(&mut self, agent_id: uuid::Uuid, future: F)
    where
        F: Future<Output = AgentTaskOutput> + Send + 'static,
    {
        let handle = self.tasks.spawn(future);
        self.agents_by_task.insert(handle.id(), agent_id);
    }

    async fn join_next(
        &mut self,
    ) -> Option<(uuid::Uuid, Result<AgentTaskOutput, tokio::task::JoinError>)> {
        let joined = self.tasks.join_next_with_id().await?;
        match joined {
            Ok((task_id, output)) => {
                let agent_id = self
                    .agents_by_task
                    .remove(&task_id)
                    .expect("completed orchestration task must be tracked");
                debug_assert_eq!(agent_id, output.0);
                Some((agent_id, Ok(output)))
            }
            Err(error) => {
                let agent_id = self
                    .agents_by_task
                    .remove(&error.id())
                    .expect("failed orchestration task must be tracked");
                Some((agent_id, Err(error)))
            }
        }
    }

    async fn abort_all_and_drain(&mut self) {
        self.tasks.abort_all();
        while self.join_next().await.is_some() {}
        debug_assert!(self.agents_by_task.is_empty());
    }
}

async fn cleanup_failed_orchestration(
    agent_handles: &mut AgentTaskSet,
    composer: &mut Composer,
    agent_ids: &[uuid::Uuid],
    reason: &str,
) {
    agent_handles.abort_all_and_drain().await;
    for agent_id in agent_ids {
        let is_terminal = composer
            .get_agent(agent_id)
            .map(|(_, lifecycle)| lifecycle.phase.is_terminal())
            .unwrap_or(true);
        if !is_terminal {
            composer.fail_agent(*agent_id, reason);
        }
    }
}

async fn recover_failed_orchestration(
    agent_handles: &mut AgentTaskSet,
    composer: &mut Composer,
    store: &odin_orchestrator::persistence::SqliteOrchestrationStore,
    root_goal: &str,
    agent_ids: &[uuid::Uuid],
    error: anyhow::Error,
) -> anyhow::Error {
    let primary_error = error.to_string();
    let failure_reason = format!("Orchestration supervisor failed: {primary_error}");
    cleanup_failed_orchestration(agent_handles, composer, agent_ids, &failure_reason).await;

    let mut cleanup_errors = Vec::new();
    for agent_id in agent_ids {
        if let Err(cleanup_error) =
            persist_orchestration_state(store, composer, root_goal, *agent_id).await
        {
            cleanup_errors.push(cleanup_error.to_string());
        }
    }

    if cleanup_errors.is_empty() {
        anyhow::anyhow!(primary_error)
    } else {
        anyhow::anyhow!(
            "{primary_error}; cleanup persistence also failed: {}",
            cleanup_errors.join("; ")
        )
    }
}

async fn persist_orchestration_state(
    store: &odin_orchestrator::persistence::SqliteOrchestrationStore,
    composer: &Composer,
    root_goal: &str,
    agent_id: uuid::Uuid,
) -> anyhow::Result<()> {
    let graph_root_id = composer
        .get_graph(root_goal)
        .map(|graph| graph.id.to_string())
        .ok_or_else(|| anyhow::anyhow!("orchestration graph '{root_goal}' not found"))?;
    if let Some((_, lifecycle)) = composer.get_agent(&agent_id) {
        store
            .save_agent_lifecycle(&graph_root_id, lifecycle)
            .await?;
    }
    if let Some(graph) = composer.get_graph(root_goal) {
        store.save_task_graph(graph).await?;
    }
    let lock_snapshot = serde_json::to_string(&composer.file_locks().snapshot())?;
    store
        .save_lock_snapshot(&graph_root_id, &lock_snapshot)
        .await?;
    Ok(())
}

// ── CLI Definition ───────────────────────────────────────────────────

/// Raven Agent — multi-agent AI orchestration platform.
#[derive(Parser, Debug)]
#[command(
    name = "raven",
    version,
    about = "Raven Agent — multi-agent AI orchestration platform",
    long_about = "Raven Agent is a multi-agent AI orchestration platform.\n\
                  It decomposes user goals, spawns hidden sub-agents with\n\
                  scoped tools/files/permissions, manages file locks, and\n\
                  merges parallel results into one coherent response.\n\
                  Default behavior: multi-agent orchestration.",
    author = "Raven Agent Contributors"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Execute a task through the agent loop (orchestrated by default).
    Run {
        /// The task goal to execute.
        task: String,

        /// Optional config file path.
        #[arg(short, long, global = true, env = "RAVEN_CONFIG")]
        config: Option<PathBuf>,

        /// Max iterations for the task.
        #[arg(short = 'n', long, default_value_t = 100)]
        max_iterations: u32,

        /// Use direct single-agent execution (skip orchestration).
        #[arg(long)]
        direct: bool,
    },

    /// Orchestrate a goal with hidden sub-agents (default mode).
    Orchestrate {
        #[command(subcommand)]
        action: OrchestrateAction,
    },

    /// Start the HTTP API server.
    Serve {
        /// Address to listen on.
        #[arg(short, long, env = "RAVEN_HTTP_ADDR")]
        addr: Option<String>,

        /// Optional config file path.
        #[arg(short, long, global = true, env = "RAVEN_CONFIG")]
        config: Option<PathBuf>,
    },

    /// Start the interactive terminal UI (chat panel + orchestration side panel).
    Ui {
        /// Optional orchestration database path.
        #[arg(long, env = "RAVEN_ORCH_DB")]
        db_path: Option<PathBuf>,
    },

    /// Show or edit configuration.
    Config {
        /// Path to the config file.
        #[arg(default_value = "~/.config/raven/config.yaml", env = "RAVEN_CONFIG")]
        path: PathBuf,

        /// Edit the config in $EDITOR.
        #[arg(short, long)]
        edit: bool,
    },

    /// Show version information.
    Version,

    /// Manage scheduled cron jobs.
    Schedule {
        #[command(subcommand)]
        action: ScheduleAction,
        /// Optional config file path.
        #[arg(short, long, global = true, env = "RAVEN_CONFIG")]
        config: Option<PathBuf>,
    },

    /// List configured providers with health status.
    Providers {
        #[command(subcommand)]
        action: ProvidersAction,
        #[arg(short, long, global = true, env = "RAVEN_CONFIG")]
        config: Option<PathBuf>,
    },

    /// Run small/local/cheap model evaluations.
    Eval {
        #[command(subcommand)]
        action: EvalAction,
    },

    /// Manage and list skills.
    Skills {
        #[command(subcommand)]
        action: SkillsAction,
    },

    /// Manage and list tasks.
    Tasks {
        #[command(subcommand)]
        action: TasksAction,
    },

    /// List and inspect sessions.
    Sessions {
        #[command(subcommand)]
        action: SessionsAction,
    },

    /// List registered tools.
    Tools {
        #[command(subcommand)]
        action: ToolsAction,
    },

    /// Audit log operations.
    Audit {
        #[command(subcommand)]
        action: AuditAction,
    },

    /// Show runtime status summary.
    Status,
}

#[derive(Subcommand, Debug)]
enum ScheduleAction {
    /// Continuously host due jobs until SIGINT/SIGTERM.
    Host,
    /// Show scheduler configuration, job counts, and the latest outcome.
    Status,
    /// Show recent durable execution outcomes.
    History {
        /// Maximum number of outcomes to show.
        #[arg(short, long, default_value_t = 20)]
        limit: usize,
    },
    /// Add a new scheduled job.
    Add {
        /// Human-readable name for this job.
        name: String,
        /// Cron expression (e.g., "0 */6 * * *").
        schedule: String,
        /// Task goal to execute when the job fires.
        task: String,
        /// Runtime agent UUID or exact name (required when the runtime has multiple agents).
        #[arg(long)]
        agent: Option<String>,
    },
    /// List all scheduled jobs.
    List,
    /// Remove a job by ID.
    Remove {
        /// Job ID (UUID) to remove.
        job_id: String,
    },
    /// Enable a disabled job by ID.
    Enable {
        /// Job ID (UUID) to enable.
        job_id: String,
    },
    /// Disable an enabled job by ID.
    Disable {
        /// Job ID (UUID) to disable.
        job_id: String,
    },
}

#[derive(Subcommand, Debug)]
enum OrchestrateAction {
    /// Submit a new task for orchestration.
    Submit {
        /// The goal to orchestrate.
        goal: String,

        /// Optional config file path.
        #[arg(short, long, global = true, env = "RAVEN_CONFIG")]
        config: Option<PathBuf>,

        /// Max iterations per sub-agent.
        #[arg(short = 'n', long, default_value_t = 100)]
        max_iterations: u32,
    },
    /// Show status of the current orchestration run.
    Status,
    /// Inspect a specific task by ID.
    Inspect {
        /// Task ID.
        id: String,
    },
    /// Cancel a running task.
    Cancel {
        /// Task ID.
        id: String,
    },
    /// Mark all unfinished stored run records as paused (does not signal another process).
    Pause,
    /// Mark paused stored run records as running (does not restart execution).
    Resume,
    /// List all active sub-agents.
    Agents,
    /// List declared file access and the latest persisted lock snapshot.
    Locks,
    /// Summarize stored graphs and agent lifecycle states.
    Queue,
    /// Discover and restore unfinished runs from the DB.
    Restore {
        /// Optional run ID to restore (lists all if omitted).
        id: Option<String>,
    },
}

#[derive(Subcommand, Debug)]
enum ProvidersAction {
    /// List all configured providers.
    List,
}

#[derive(Subcommand, Debug)]
enum EvalAction {
    /// Run the deterministic mocked small-model suite.
    Mocked {
        /// Built-in model profile ID.
        #[arg(long, default_value = "ollama-qwen2.5-coder-7b")]
        profile: String,
        /// Output format: table or json.
        #[arg(short, long, default_value = "table")]
        format: String,
        /// Optional output file.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// List built-in small-model profiles.
    Profiles {
        /// Output format: table or json.
        #[arg(short, long, default_value = "table")]
        format: String,
    },
    /// Check optional live eval readiness for a provider/model.
    Live {
        /// Provider name, e.g. ollama, openai_compat, deepseek.
        #[arg(long)]
        provider: String,
        /// Model name to evaluate.
        #[arg(long)]
        model: String,
        /// OpenAI-compatible base URL for local providers.
        #[arg(long)]
        base_url: Option<String>,
        /// Environment variable that contains the provider API key.
        #[arg(long)]
        api_key_env: Option<String>,
    },
}

#[derive(Subcommand, Debug)]
enum SkillsAction {
    /// List available skills.
    List {
        #[arg(short, long)]
        dir: Option<String>,
    },
    /// Show required and recommended tools for a skill.
    Tools {
        /// Skill name.
        name: String,
    },
}

#[derive(Subcommand, Debug)]
enum TasksAction {
    /// List recent tasks.
    List {
        #[arg(short, long, default_value_t = 20)]
        limit: usize,
        #[arg(short, long)]
        status: Option<String>,
    },
    /// Inspect a task by ID.
    Inspect { id: String },
}

#[derive(Subcommand, Debug)]
enum SessionsAction {
    /// List active sessions.
    List,
    /// Inspect a session by ID.
    Inspect { id: String },
}

/// Manage and inspect registered tools via the `raven tools` subcommand.
#[derive(Subcommand, Debug)]
enum ToolsAction {
    /// List all registered tools.
    List {
        /// Filter by capability tag (can be specified multiple times).
        #[arg(short, long)]
        tag: Vec<String>,
    },
    /// Inspect a tool by name.
    Inspect { name: String },
    /// Validate the tool registry.
    Validate,
    /// Test a specific tool.
    Test {
        name: String,
        #[arg(short, long)]
        args: Option<String>,
        /// Run in dry-run mode — validate args but return mock result.
        #[arg(long)]
        dry_run: bool,
        /// Explicitly approve execution of a tool marked dangerous.
        #[arg(long)]
        approve: bool,
    },
    /// Run a comprehensive doctor check on all registered tools.
    Doctor,
    /// Show the tool catalog grouped by category.
    Catalog {
        /// Output format: table (default), json, yaml.
        #[arg(short, long, default_value = "table")]
        format: String,
        /// Filter by category (e.g., "filesystem", "shell", "web").
        #[arg(short, long)]
        category: Option<String>,
        /// Filter by tag (e.g., "safe", "dangerous", "read").
        #[arg(short, long)]
        tag: Option<String>,
    },
    /// Show reliability scores for all tools.
    Reliability,
}

#[derive(Subcommand, Debug)]
enum AuditAction {
    /// Replay audit entries for a task ID.
    Replay { task_id: String },
}

// ── Entrypoint ───────────────────────────────────────────────────────

pub async fn run(default_to_ui: bool) -> anyhow::Result<()> {
    let cli = Cli::parse();
    let launching_tui = matches!(&cli.command, Some(Commands::Ui { .. }))
        || (cli.command.is_none() && default_to_ui);
    init_tracing(launching_tui)?;

    match cli.command {
        Some(Commands::Run {
            task,
            config,
            max_iterations,
            direct,
        }) => cmd_run(task, config, max_iterations, direct).await,
        Some(Commands::Orchestrate { action }) => cmd_orchestrate(action).await,
        Some(Commands::Serve { addr, config }) => cmd_serve(addr, config).await,
        Some(Commands::Ui { db_path }) => cmd_ui(db_path).await,
        Some(Commands::Config { path, edit }) => cmd_config(path, edit),
        Some(Commands::Schedule { action, config }) => cmd_schedule(action, config).await,
        Some(Commands::Providers { action, config }) => cmd_providers(action, config).await,
        Some(Commands::Eval { action }) => cmd_eval(action).await,
        Some(Commands::Skills { action }) => cmd_skills(action).await,
        Some(Commands::Tasks { action }) => cmd_tasks(action).await,
        Some(Commands::Sessions { action }) => cmd_sessions(action).await,
        Some(Commands::Tools { action }) => cmd_tools(action).await,
        Some(Commands::Audit { action }) => cmd_audit(action).await,
        Some(Commands::Version) => cmd_version(),
        Some(Commands::Status) => cmd_status(),
        None if default_to_ui => {
            // Default: open the TUI
            tracing::info!("[raven] No command, launching interactive TUI");
            cmd_ui(None).await
        }
        None => {
            // Keep the legacy alias non-interactive when no command is provided.
            eprintln!("Usage: odin <COMMAND>");
            eprintln!("Try 'odin --help' for more information.");
            eprintln!("Or run 'raven' to launch the interactive TUI.");
            std::process::exit(1);
        }
    }
}

fn init_tracing(launching_tui: bool) -> anyhow::Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    if launching_tui {
        let log_path = tui_log_path();
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut options = std::fs::OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(&log_path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&log_path, std::fs::Permissions::from_mode(0o600))?;
        }
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .compact()
            .with_writer(move || {
                file.try_clone()
                    .expect("failed to clone Raven TUI tracing log file")
            })
            .try_init();
    } else {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .compact()
            .with_writer(std::io::stderr)
            .try_init();
    }
    Ok(())
}

fn tui_log_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home).join(".raven-agent/tui.log")
}

// ── Command Implementations ──────────────────────────────────────────

/// `raven run <task>` — Execute a task (orchestrated by default, --direct for one agent).
async fn cmd_run(
    task: String,
    config_path: Option<PathBuf>,
    max_iterations: u32,
    direct: bool,
) -> anyhow::Result<()> {
    tracing::info!(direct, "Starting Raven Agent task");

    // Warn about any unfinished orchestration runs from previous sessions
    warn_about_unfinished_runs().await;

    // Load configuration
    let config = load_config(config_path.as_deref())?;
    tracing::debug!("[CLI] Config loaded");
    let composition = ProductionComposition::from_config(&config)?;
    tracing::info!(
        provider = %composition.resolved().provider_name,
        model = %composition.resolved().model_name,
        fallbacks = ?composition.resolved().fallback_providers,
        "[CLI] Validated production composition"
    );

    let audit_logger = Arc::new(build_audit_logger(&config)?);
    let approval_gate = Arc::new(odin_permissions::ApprovalGate::default());
    let policy_engine = Arc::new(
        odin_permissions::PolicyEngine::new(
            config.safety.permissions.clone(),
            &config.safety.dangerous_commands,
            config.tools.path_boundary.clone(),
            config.safety.max_rate_per_minute,
            config.safety.require_approval,
        )
        .with_approval_gate(approval_gate.clone())
        .with_audit_logger(audit_logger.clone()),
    );
    let _approval_responder = CliApprovalResponder::spawn(approval_gate);
    tracing::info!("[CLI] Policy engine initialized");

    // ── Shared resources (used by both orchestrated and direct paths) ──
    let sandbox = Arc::new(odin_tools::Sandbox::new(config.tools.path_boundary.clone()));
    let enabled_tools = config.tools.effective_enabled_tools();
    let tool_registry = Arc::new(build_tool_registry_with(
        sandbox.clone(),
        Some(&enabled_tools),
    ));

    // Load MCP tools from configured servers
    load_mcp_tools(&tool_registry, &config).await;

    let memory = Arc::new(build_memory_store(&config)?);
    let reliability_tracker = build_reliability_tracker(&config)?;
    let composition = composition.with_resources(CompositionResources {
        policy_engine: Some(policy_engine.clone()),
        tool_registry: Some(tool_registry.clone()),
        audit_logger: Some(audit_logger.clone()),
        reliability_tracker: Some(reliability_tracker),
    });
    tracing::info!("[CLI] Memory store and audit logger initialized");

    // Branch: orchestrated (default) or direct single-agent
    if !direct {
        return run_orchestrated(
            &task,
            max_iterations,
            composition,
            tool_registry,
            &config,
            memory,
        )
        .await;
    }

    // ── Direct single-agent mode (legacy) ──────────────────────────
    let agent = composition.build_agent(
        ExecutionSurface::Cli,
        "default-agent",
        EngineBuildOptions {
            max_iterations: Some(max_iterations),
            ..Default::default()
        },
    );
    let agent_id = agent.id;

    // Register agent in runtime with memory store
    let runtime = Runtime::new()
        .with_memory(memory)
        .with_default_max_iterations(max_iterations);
    runtime.register_agent(agent);
    let session = runtime.create_session_with_label("cli-run-direct");

    // Create the task
    let agent_task = AgentTask {
        id: uuid::Uuid::new_v4(),
        goal: task.clone(),
        context: None,
        sub_tasks: vec![],
        success_criteria: vec![],
        max_iterations,
        created_at: chrono::Utc::now(),
    };

    // Log task start
    let start_entry = odin_core::types::AuditEntry {
        id: uuid::Uuid::new_v4(),
        timestamp: chrono::Utc::now(),
        agent_id,
        session_id: session.id,
        event_type: odin_core::types::AuditEventType::SessionStart,
        action: "cli_run".to_string(),
        details: serde_json::json!({
            "task_id": agent_task.id,
            "session_id": session.id,
            "task": task,
            "max_iterations": max_iterations,
        }),
        result: odin_core::types::AuditResult::Success,
    };
    audit_logger.log(start_entry).await?;

    // Submit the task
    tracing::info!(task_id = %agent_task.id, agent_id = %agent_id, "Submitting CLI task");
    let start = std::time::Instant::now();
    let result = runtime
        .submit_task(&agent_id, &agent_task, Some(session.id))
        .await?;
    let elapsed = start.elapsed();

    // Log task end
    let end_entry = odin_core::types::AuditEntry {
        id: uuid::Uuid::new_v4(),
        timestamp: chrono::Utc::now(),
        agent_id,
        session_id: session.id,
        event_type: odin_core::types::AuditEventType::SessionEnd,
        action: "cli_run_complete".to_string(),
        details: serde_json::json!({
            "task_id": result.task_id,
            "session_id": session.id,
            "success": result.success,
            "iterations": result.iterations,
            "duration_ms": elapsed.as_millis(),
            "tool_calls": result.tool_calls,
            "confidence": result.confidence,
        }),
        result: if result.success {
            odin_core::types::AuditResult::Success
        } else {
            odin_core::types::AuditResult::Failure
        },
    };
    audit_logger.log(end_entry).await?;

    // Print the result
    println!();
    println!("╔══════════════════════════════════════════╗");
    println!("║          Task Result                     ║");
    println!("╠══════════════════════════════════════════╣");
    println!(
        "║  Goal:    {:32} ║",
        task.chars().take(32).collect::<String>()
    );
    println!(
        "║  Status:  {:32} ║",
        if result.success {
            "✓ COMPLETED"
        } else {
            "✗ STOPPED"
        }
    );
    println!(
        "║  Duration: {:31} ║",
        format!("{}.{:03}s", elapsed.as_secs(), elapsed.subsec_millis())
    );
    println!("║  Iterations: {:26} ║", result.iterations);
    println!("║  Tool calls: {:25} ║", result.tool_calls);
    println!(
        "║  Confidence: {:25}% ║",
        (result.confidence * 100.0) as u32
    );
    println!("╚══════════════════════════════════════════╝");
    println!();
    println!("Summary: {}", result.summary);

    if !result.sub_tasks.is_empty() {
        println!();
        println!("Sub-tasks:");
        for st in &result.sub_tasks {
            let icon = match st.status {
                odin_core::types::SubTaskStatus::Completed => "✓",
                odin_core::types::SubTaskStatus::Failed => "✗",
                odin_core::types::SubTaskStatus::InProgress => "◷",
                odin_core::types::SubTaskStatus::Pending => "○",
                odin_core::types::SubTaskStatus::Skipped => "−",
            };
            println!("  {} {} — {}", icon, st.id, st.description);
            if let Some(ref res) = st.result {
                println!("        Result: {}", res);
            }
        }
    }

    if let Some(ref err) = result.error {
        println!();
        println!("Error: {}", err);
    }

    Ok(())
}

/// Orchestrated execution — decompose goal, spawn parallel sub-agents, merge results.
fn subagent_retry_is_idempotent(allowed_tools: &[String]) -> bool {
    const READ_ONLY_TOOLS: &[&str] = &[
        "disk_usage",
        "env_var",
        "file_exists",
        "file_list",
        "file_read",
        "github_actions_status",
        "github_issue_search",
        "github_pr_status",
        "json_extract",
        "json_validate",
        "network_ping",
        "process_list",
        "system_info",
        "text_search",
        "time_now",
        "web_fetch",
        "web_search",
    ];
    allowed_tools
        .iter()
        .all(|tool| READ_ONLY_TOOLS.contains(&tool.as_str()))
}

#[allow(clippy::too_many_arguments)]
async fn run_orchestrated(
    goal: &str,
    max_iterations: u32,
    composition: ProductionComposition,
    tool_registry: Arc<odin_tools::ToolRegistry>,
    config: &OdinConfig,
    memory: Arc<odin_memory::SqliteMemoryStore>,
) -> anyhow::Result<()> {
    use odin_orchestrator::RunControlKind;
    use odin_orchestrator::composer::ComposerConfig;
    use odin_orchestrator::merge::{MergeStrategy, SubAgentResult};
    use odin_orchestrator::persistence::OrchestrationStore;

    let mut composer = Composer::new(ComposerConfig {
        max_parallel: 10,
        default_max_iterations: max_iterations,
        auto_merge: true,
        merge_strategy: MergeStrategy::Concatenate,
        workspace_root: ".".to_string(),
        persist_state: true,
        max_retries: 1,
    });

    // Decompose the goal into a task graph
    composer.intake(goal);
    let (run_id, node_count) = {
        let graph = composer.get_graph(goal).unwrap();
        (graph.id, graph.nodes.len())
    };
    let store = odin_orchestrator::persistence::SqliteOrchestrationStore::new(dirs_state_path(
        "orchestration.db",
    ))
    .await
    .map_err(|error| anyhow::anyhow!("Failed to open orchestration state: {error}"))?;
    store.initialize().await?;
    store
        .save_task_graph(composer.get_graph(goal).unwrap())
        .await?;
    let memory_enabled = config.memory.enabled;
    let memory_max_entries = config.memory.max_entries.max(1);
    let run_id_str = run_id.to_string();

    println!("╔══════════════════════════════════════════╗");
    println!("║     Raven Agent — Orchestrated Run      ║");
    println!("╠══════════════════════════════════════════╣");
    println!("║  Run ID: {:<32} ║", run_id);
    println!(
        "║  Goal:  {:<32} ║",
        goal.chars().take(32).collect::<String>()
    );
    println!("║  Tasks: {:<32} ║", format!("{} sub-task(s)", node_count));
    println!("╚══════════════════════════════════════════╝");
    println!();

    // For each workstream group, spawn sub-agents.
    // Collect node data first to avoid holding an immutable borrow of composer.
    let root_goal = goal.to_string();
    let graph = composer
        .get_graph(&root_goal)
        .ok_or_else(|| anyhow::anyhow!("decomposition produced no task graph for goal"))?
        .clone();
    let groups = composer.detect_workstreams(&graph);

    // Extract all node data before mutating composer
    type NodeTaskList = Vec<(uuid::Uuid, String, String, Vec<String>, Vec<String>, u32)>;
    #[allow(clippy::type_complexity)]
    let mut node_tasks: Vec<(usize, NodeTaskList)> = Vec::new();
    for (group_idx, group) in groups.iter().enumerate() {
        let mut tasks = Vec::new();
        for &node_id in group {
            let node = &graph.nodes[&node_id];
            tasks.push((
                node.id,
                node.label.clone(),
                node.goal.clone(),
                node.read_files.clone(),
                node.write_files.clone(),
                node.priority,
            ));
        }
        node_tasks.push((group_idx, tasks));
    }

    // Register all agents first; only spawn execution when locks are granted.
    #[derive(Clone)]
    struct PendingAgent {
        agent_id: uuid::Uuid,
        label: String,
        task_goal: String,
        allowed_tools: Vec<String>,
        priority: u32,
    }

    let mut pending: Vec<PendingAgent> = Vec::new();
    for (_group_idx, tasks) in &node_tasks {
        for (node_id, label, task_goal, _read_files, _write_files, priority) in tasks {
            let node = graph
                .nodes
                .get(node_id)
                .ok_or_else(|| anyhow::anyhow!("missing task node {node_id}"))?;
            let mut agent_config = composer.create_sub_agent(node);
            agent_config.max_iterations = max_iterations;
            agent_config.priority = *priority;
            let allowed_tools = agent_config.allowed_tools.clone();
            let agent_id = composer.register_agent(agent_config);
            pending.push(PendingAgent {
                agent_id,
                label: label.clone(),
                task_goal: task_goal.clone(),
                allowed_tools,
                priority: *priority,
            });
        }
    }

    let mut agent_handles = AgentTaskSet::new();
    let mut spawned = std::collections::HashSet::<uuid::Uuid>::new();
    let mut terminal = std::collections::HashSet::<uuid::Uuid>::new();
    let max_retries = config.agent.max_retries.min(3);

    println!(
        "🔄 Dispatching up to {} agent(s) with file-lock awareness...",
        pending.len()
    );

    let all_agent_ids: Vec<uuid::Uuid> = pending.iter().map(|agent| agent.agent_id).collect();
    let mut paused = false;
    let mut cancelled = false;
    let execution_result: anyhow::Result<()> = async {
        while terminal.len() < pending.len() {
            // Apply live cross-process control commands (CLI/WS/second process).
            let controls = store
                .claim_pending_controls(&run_id_str)
                .await
                .map_err(|error| anyhow::anyhow!("Failed to claim control commands: {error}"))?;
            for command in controls {
                match command.kind {
                    RunControlKind::Pause => {
                        paused = true;
                        composer.set_graph_status(
                            &root_goal,
                            odin_orchestrator::task_graph::TaskGraphStatus::Paused,
                        );
                        println!("⏸️  Live pause received from {}", command.source);
                    }
                    RunControlKind::Resume => {
                        paused = false;
                        composer.set_graph_status(
                            &root_goal,
                            odin_orchestrator::task_graph::TaskGraphStatus::Running,
                        );
                        println!("▶️  Live resume received from {}", command.source);
                    }
                    RunControlKind::Cancel => {
                        cancelled = true;
                        composer.set_graph_status(
                            &root_goal,
                            odin_orchestrator::task_graph::TaskGraphStatus::Cancelled,
                        );
                        println!("🛑 Live cancel received from {}", command.source);
                    }
                }
                if let Some(graph) = composer.get_graph(&root_goal) {
                    store.save_task_graph(graph).await?;
                }
                store.mark_control_applied(command.id).await?;
            }
            if cancelled {
                agent_handles.abort_all_and_drain().await;
                composer.cancel_all("cancelled by live control command");
                for agent_id in &all_agent_ids {
                    let _ =
                        persist_orchestration_state(&store, &composer, &root_goal, *agent_id).await;
                }
                break;
            }
            // Start any queued agents that can acquire locks now.
            let mut ordered: Vec<_> = if paused {
                Vec::new()
            } else {
                pending
                    .iter()
                    .filter(|agent| {
                        !spawned.contains(&agent.agent_id) && !terminal.contains(&agent.agent_id)
                    })
                    .cloned()
                    .collect()
            };
            ordered.sort_by_key(|agent| (agent.priority, agent.label.clone()));

            for agent in ordered {
                let scoped_registry = match tool_registry.scoped(&agent.allowed_tools) {
                    Ok(registry) => Arc::new(registry),
                    Err(error) => {
                        composer.fail_agent(
                            agent.agent_id,
                            format!("Invalid tool scope for agent '{}': {error}", agent.label),
                        );
                        terminal.insert(agent.agent_id);
                        persist_orchestration_state(&store, &composer, &root_goal, agent.agent_id)
                            .await?;
                        continue;
                    }
                };

                match composer.start_agent(agent.agent_id) {
                    Ok(()) => {
                        tracing::info!("[ORCH] Agent '{}' started", agent.label);
                        spawned.insert(agent.agent_id);
                        if let Some((_, lifecycle)) = composer.get_agent(&agent.agent_id) {
                            store.save_agent_lifecycle(&run_id_str, lifecycle).await?;
                        }
                        let graph = composer.get_graph(&root_goal).ok_or_else(|| {
                            anyhow::anyhow!("task graph missing for root goal during spawn")
                        })?;
                        store.save_task_graph(graph).await?;
                        let lock_snapshot =
                            serde_json::to_string(&composer.file_locks().snapshot())?;
                        store
                            .save_lock_snapshot(&run_id_str, &lock_snapshot)
                            .await?;

                        let composition = composition.clone();
                        let memory = memory.clone();
                        let task_goal = agent.task_goal.clone();
                        let label_for_result = agent.label.clone();
                        let agent_id = agent.agent_id;
                        let run_id_for_mem = run_id_str.clone();
                        let retry_is_idempotent =
                            subagent_retry_is_idempotent(&agent.allowed_tools);

                        agent_handles.spawn(agent.agent_id, async move {
                            let mut final_result = Err(odin_core::error::OdinError::Internal(
                                "Agent did not execute".into(),
                            ));
                            let mut total_elapsed = std::time::Duration::ZERO;

                            let retry_limit = if retry_is_idempotent { max_retries } else { 0 };
                            for attempt in 0..=retry_limit {
                                let engine = composition.build_engine(
                                    ExecutionSurface::Orchestration,
                                    EngineBuildOptions {
                                        max_iterations: Some(max_iterations),
                                        principal_id: Some(agent_id),
                                        tool_registry: Some(scoped_registry.clone()),
                                        ..Default::default()
                                    },
                                );

                                let task_id = uuid::Uuid::new_v4();
                                let context = if memory_enabled {
                                    match odin_memory::retrieve_task_context(
                                        memory.as_ref(),
                                        &task_goal,
                                        memory_max_entries.min(8),
                                        odin_memory::DEFAULT_CONTEXT_CHARS,
                                    )
                                    .await
                                    {
                                        Ok(value) => value,
                                        Err(error) => {
                                            tracing::warn!(
                                                "[ORCH] memory retrieve failed for '{}': {error}",
                                                label_for_result
                                            );
                                            None
                                        }
                                    }
                                } else {
                                    None
                                };
                                let task = AgentTask {
                                    id: task_id,
                                    goal: task_goal.clone(),
                                    context,
                                    sub_tasks: vec![],
                                    success_criteria: vec![],
                                    max_iterations,
                                    created_at: chrono::Utc::now(),
                                };

                                let start = std::time::Instant::now();
                                let result = engine.execute_task(&task).await;
                                let elapsed = start.elapsed();
                                total_elapsed += elapsed;

                                let is_success =
                                    result.as_ref().map(|r| r.success).unwrap_or(false);

                                if memory_enabled && let Ok(task_result) = result.as_ref() {
                                    let redacted = odin_permissions::SecretRedactor::full()
                                        .redact(&task_result.summary);
                                    if let Err(error) = odin_memory::store_task_outcome(
                                        memory.as_ref(),
                                        redacted,
                                        &run_id_for_mem,
                                        &task_id.to_string(),
                                        &agent_id.to_string(),
                                        if task_result.success { 0.7 } else { 0.3 },
                                    )
                                    .await
                                    {
                                        tracing::warn!(
                                            "[ORCH] memory store failed for '{}': {error}",
                                            label_for_result
                                        );
                                    }
                                }

                                let retryable_failure = result
                                    .as_ref()
                                    .err()
                                    .is_some_and(odin_core::error::OdinError::is_retryable);
                                if is_success || attempt == retry_limit || !retryable_failure {
                                    final_result = result;
                                    break;
                                }

                                let err_msg = result
                                    .as_ref()
                                    .err()
                                    .map(|e| e.to_string())
                                    .or_else(|| result.as_ref().ok().and_then(|r| r.error.clone()))
                                    .unwrap_or_else(|| "unknown error".to_string());
                                tracing::warn!(
                                    "[ORCH] Agent '{}' attempt {}/{} failed: {}",
                                    label_for_result,
                                    attempt + 1,
                                    retry_limit + 1,
                                    err_msg
                                );
                            }

                            (agent_id, label_for_result, final_result, total_elapsed)
                        });
                    }
                    Err(msg) => {
                        tracing::info!("[ORCH] Agent '{}' queued: {}", agent.label, msg);
                        if let Some((_, lifecycle)) = composer.get_agent(&agent.agent_id) {
                            store.save_agent_lifecycle(&run_id_str, lifecycle).await?;
                        }
                        if let Some(graph) = composer.get_graph(&root_goal) {
                            store.save_task_graph(graph).await?;
                        }
                        let lock_snapshot =
                            serde_json::to_string(&composer.file_locks().snapshot())?;
                        store
                            .save_lock_snapshot(&run_id_str, &lock_snapshot)
                            .await?;
                    }
                }
            }

            if agent_handles.is_empty() {
                if paused {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    continue;
                }
                // No in-flight work remains, but some agents never started — fail them.
                let stuck: Vec<uuid::Uuid> = pending
                    .iter()
                    .filter(|agent| {
                        !spawned.contains(&agent.agent_id) && !terminal.contains(&agent.agent_id)
                    })
                    .map(|agent| agent.agent_id)
                    .collect();
                for agent_id in stuck {
                    composer.fail_agent(
                        agent_id,
                        "Could not acquire file locks and no running agents hold them",
                    );
                    terminal.insert(agent_id);
                    persist_orchestration_state(&store, &composer, &root_goal, agent_id).await?;
                }
                break;
            }

            // Process whichever agent finishes first so its locks are released promptly.
            let joined = tokio::select! {
                joined = agent_handles.join_next() => joined,
                _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => continue,
            };
            let (spawned_agent_id, completion) =
                joined.expect("non-empty orchestration task set must yield a completion");
            match completion {
                Ok((agent_id, label, result, elapsed)) => {
                    match result {
                        Ok(task_result) => {
                            println!(
                                "  {} {} — {} ({}ms, {} iters)",
                                if task_result.success { "✅" } else { "⚠️" },
                                label,
                                if task_result.success {
                                    "success"
                                } else {
                                    "failed"
                                },
                                elapsed.as_millis(),
                                task_result.iterations,
                            );
                            composer.complete_agent(
                                agent_id,
                                SubAgentResult {
                                    agent_id,
                                    name: label,
                                    summary: task_result.summary.clone(),
                                    output: Some(task_result.summary),
                                    modified_files: vec![],
                                    success: task_result.success,
                                    error: task_result.error,
                                    duration_ms: elapsed.as_millis() as u64,
                                },
                            );
                        }
                        Err(error) => {
                            println!("  ❌ {} — error: {}", label, error);
                            composer.fail_agent(agent_id, error.to_string());
                        }
                    }
                    terminal.insert(agent_id);
                    persist_orchestration_state(&store, &composer, &root_goal, agent_id).await?;
                }
                Err(error) => {
                    tracing::error!("[ORCH] Sub-agent task failed: {}", error);
                    composer
                        .fail_agent(spawned_agent_id, format!("Sub-agent task failed: {error}"));
                    terminal.insert(spawned_agent_id);
                    persist_orchestration_state(&store, &composer, &root_goal, spawned_agent_id)
                        .await?;
                }
            }
        }
        Ok(())
    }
    .await;

    if let Err(error) = execution_result {
        return Err(recover_failed_orchestration(
            &mut agent_handles,
            &mut composer,
            &store,
            &root_goal,
            &all_agent_ids,
            error,
        )
        .await);
    }

    let mut total_success = 0;
    let mut total_fail = 0;
    for agent in &pending {
        if let Some((sub, _)) = composer.get_agent(&agent.agent_id) {
            match sub.phase {
                odin_orchestrator::lifecycle::AgentPhase::Done => total_success += 1,
                _ => total_fail += 1,
            }
        } else {
            total_fail += 1;
        }
    }

    // Merge and print final result
    let results = composer.collect_results();
    let merged = composer.merge_results(results);
    let lock_snapshot = serde_json::to_string(&composer.file_locks().snapshot())?;
    store
        .save_lock_snapshot(&run_id_str, &lock_snapshot)
        .await?;

    println!();
    println!("╔══════════════════════════════════════════╗");
    println!("║          Orchestrated Result            ║");
    println!("╠══════════════════════════════════════════╣");
    println!(
        "║  Status:  {:32} ║",
        if total_fail == 0 {
            "✓ ALL COMPLETED"
        } else {
            "✗ PARTIAL FAILURE"
        }
    );
    println!(
        "║  Success: {}/{} agents passed           ║",
        total_success,
        total_success + total_fail
    );
    println!("╚══════════════════════════════════════════╝");
    println!();
    println!("{}", merged.summary);

    if !merged.conflicts.is_empty() {
        println!();
        println!("⚠️  Merge conflicts in files:");
        for c in &merged.conflicts {
            println!("  - {} (agents: {})", c.file, c.agents.join(", "));
        }
    }

    Ok(())
}

/// Get a path in the Raven Agent state directory.
fn dirs_state_path(filename: &str) -> std::path::PathBuf {
    let base = dirs_state_dir();
    std::fs::create_dir_all(&base).ok();
    base.join(filename)
}

/// Get the Raven Agent state directory.
fn dirs_state_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    std::path::PathBuf::from(home).join(".raven-agent")
}

/// Check for unfinished orchestration runs and warn the user.
#[allow(clippy::collapsible_if)]
async fn warn_about_unfinished_runs() {
    use odin_orchestrator::persistence::{OrchestrationStore, SqliteOrchestrationStore};

    let db_path = dirs_state_path("orchestration.db");
    if !db_path.exists() {
        return;
    }

    if let Ok(store) = SqliteOrchestrationStore::new(&db_path).await {
        let _ = store.initialize().await;
        if let Ok(unfinished) = store.find_unfinished_graphs().await {
            if !unfinished.is_empty() {
                println!();
                println!("╔══════════════════════════════════════════╗");
                println!("║  ⚠️  Unfinished Orchestration Runs      ║");
                println!("╠══════════════════════════════════════════╣");
                for g in &unfinished {
                    let icon = if g.status == "running" {
                        "🟢"
                    } else {
                        "⏸️"
                    };
                    println!(
                        "║  {} '{}' ({} nodes) [{}]",
                        icon,
                        g.root_goal.chars().take(30).collect::<String>(),
                        g.node_count,
                        g.status
                    );
                }
                println!("╠══════════════════════════════════════════╣");
                println!("║ Restore: raven orchestrate restore      ║");
                println!("╚══════════════════════════════════════════╝");
                println!();
            }
        }
    }
}

/// Load MCP tools from configured MCP servers into a tool registry.
async fn load_mcp_tools(registry: &odin_tools::ToolRegistry, config: &OdinConfig) {
    use odin_mcp::client::McpClient;
    use odin_mcp::tool_adapter::McpToolAdapter;
    use odin_mcp::transport::StdioTransport;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    for server_cfg in &config.tools.mcp_servers {
        if !server_cfg.enabled {
            continue;
        }
        if server_cfg.transport_type != "stdio" {
            tracing::warn!(
                "[MCP] Skipping server '{}': unsupported transport '{}'",
                server_cfg.name,
                server_cfg.transport_type
            );
            continue;
        }

        tracing::info!("[MCP] Loading tools from server '{}'", server_cfg.name);

        let transport = Arc::new(Mutex::new(
            StdioTransport::new(&server_cfg.command, server_cfg.args.clone())
                .with_env(server_cfg.env.clone()),
        ));

        if let Err(error) = transport.lock().await.connect().await {
            tracing::warn!(
                "[MCP] Failed to spawn server '{}': {}",
                server_cfg.name,
                error
            );
            continue;
        }

        let transport: Arc<Mutex<dyn odin_mcp::transport::McpTransport>> = transport;
        let mut client = McpClient::new(transport.clone());
        let shared_client = match client.connect().await {
            Ok(()) => {
                tracing::debug!("[MCP] Connected to server '{}'", server_cfg.name);
                Arc::new(Mutex::new(client))
            }
            Err(e) => {
                tracing::warn!(
                    "[MCP] Failed to connect to server '{}': {}",
                    server_cfg.name,
                    e
                );
                continue;
            }
        };

        let list_result = {
            let c = shared_client.lock().await;
            c.list_tools().await
        };
        match list_result {
            Ok(tools) => {
                let count = tools.len();
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
                        tracing::info!(tool = %name, "Skipping disabled MCP tool");
                        continue;
                    }
                    if let Err(e) = registry.register(Box::new(adapter)) {
                        tracing::warn!("[MCP] Failed to register tool '{}': {}", name, e);
                    }
                }
                tracing::info!(
                    "[MCP] Loaded {} tool(s) from server '{}'",
                    count,
                    server_cfg.name
                );
            }
            Err(e) => {
                tracing::warn!(
                    "[MCP] Failed to list tools from server '{}': {}",
                    server_cfg.name,
                    e
                );
            }
        }
    }
}

/// `raven orchestrate` — Multi-agent orchestration state commands.
async fn cmd_orchestrate(action: OrchestrateAction) -> anyhow::Result<()> {
    match action {
        OrchestrateAction::Submit {
            goal,
            config: _config,
            max_iterations: _max_iterations,
        } => {
            tracing::info!("Creating stored orchestration plan");

            // Initialize the orchestration store
            let store = odin_orchestrator::persistence::SqliteOrchestrationStore::new(
                dirs_state_path("orchestration.db"),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Failed to initialize orchestration store: {e}"))?;
            store
                .initialize()
                .await
                .map_err(|e| anyhow::anyhow!("Store init: {e}"))?;

            let mut composer = Composer::default();
            composer.intake(&goal);

            let (run_id, node_count) = {
                let graph = composer.get_graph(&goal).unwrap();
                (graph.id, graph.nodes.len())
            };

            // Persist the task graph
            if let Some(graph) = composer.get_graph(&goal) {
                let mut graph = graph.clone();
                graph.status = odin_orchestrator::task_graph::TaskGraphStatus::Building;
                store.save_task_graph(&graph).await.map_err(|error| {
                    anyhow::anyhow!("Failed to persist orchestration plan: {error}")
                })?;
            }

            println!("╔══════════════════════════════════════════╗");
            println!("║     Raven Agent — Orchestration         ║");
            println!("╠══════════════════════════════════════════╣");
            println!("║  Run ID: {:<32} ║", run_id.to_string());
            println!(
                "║  Goal:  {:<32} ║",
                goal.chars().take(32).collect::<String>()
            );
            println!("║  Tasks: {:<32} ║", format!("{} sub-task(s)", node_count));
            println!("╚══════════════════════════════════════════╝");
            println!();
            println!("Use 'raven orchestrate status' to inspect stored state.");
            println!("Use 'raven orchestrate inspect {}' for details.", run_id);

            if node_count <= 1 {
                println!("   Single task — no parallelization needed.");
                println!("   Goal: {}", goal);
            } else {
                println!("   Decomposed into {} parallel sub-tasks:", node_count);
                if let Some(graph) = composer.get_graph(&goal) {
                    for (i, node) in graph.nodes.values().enumerate() {
                        let mut info = format!("     {}. {}", i + 1, node.label);
                        if !node.write_files.is_empty() {
                            info.push_str(&format!(" [writes: {}]", node.write_files.join(", ")));
                        }
                        println!("{}", info);
                    }
                }
            }

            // Show what workstreams were detected
            if let Some(graph) = composer.get_graph(&goal) {
                let workstreams = composer.detect_workstreams(graph);
                println!();
                println!("   Workstreams: {} parallel group(s)", workstreams.len());
                for (i, ws) in workstreams.iter().enumerate() {
                    println!("     Group {}: {} agent(s)", i + 1, ws.len());
                }
            }
            println!();
        }
        OrchestrateAction::Status => {
            let store = odin_orchestrator::persistence::SqliteOrchestrationStore::new(
                dirs_state_path("orchestration.db"),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Store: {e}"))?;
            store
                .initialize()
                .await
                .map_err(|e| anyhow::anyhow!("Store init: {e}"))?;

            let graphs = store.list_task_graphs().await.unwrap_or_default();
            let lifecycles = store.list_agent_lifecycles().await.unwrap_or_default();

            println!("📊 Orchestration Status");
            println!("   Stored task graphs: {}", graphs.len());
            for g in &graphs {
                println!(
                    "     - {} '{}' ({} nodes, {})",
                    g.run_id, g.root_goal, g.node_count, g.status
                );
            }
            println!("   Stored agent lifecycles: {}", lifecycles.len());
            for lc in &lifecycles {
                println!("     - {} ({})", lc.agent_id, lc.phase);
            }
            if graphs.is_empty() && lifecycles.is_empty() {
                println!(
                    "   No stored orchestration state. Use 'raven orchestrate submit' to create a plan."
                );
            }
        }
        OrchestrateAction::Inspect { id } => {
            let store = odin_orchestrator::persistence::SqliteOrchestrationStore::new(
                dirs_state_path("orchestration.db"),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Store: {e}"))?;
            store
                .initialize()
                .await
                .map_err(|e| anyhow::anyhow!("Store init: {e}"))?;

            // New records use a run UUID; old records also accept their goal key.
            if let Ok(graph) = store.load_task_graph(&id).await {
                println!("🔍 Task Graph: {}", graph.root_goal);
                println!("   Status: {:?}", graph.status);
                println!("   Nodes: {}", graph.nodes.len());
                for (i, node) in graph.nodes.values().enumerate() {
                    println!("     {}. {} [{:?}]", i + 1, node.label, node.status);
                    println!("        Goal: {}", node.goal);
                    if !node.write_files.is_empty() {
                        println!("        Writes: {}", node.write_files.join(", "));
                    }
                }
                return Ok(());
            }

            // Try to load as an agent lifecycle
            if let Ok(run_id) = uuid::Uuid::parse_str(&id)
                && let Ok(lc) = store.load_agent_lifecycle(run_id).await
            {
                println!("🔍 Agent Lifecycle: {}", lc.agent_id);
                println!("   Phase: {:?}", lc.phase);
                println!("   Created: {}", lc.created_at);
                if let Some(finished) = lc.finished_at {
                    println!("   Finished: {}", finished);
                }
                if let Some(err) = &lc.error {
                    println!("   Error: {}", err);
                }
                println!("   History: {} transition(s)", lc.history.len());
                return Ok(());
            }

            println!(
                "🔍 Not found: '{}' — no task graph or agent lifecycle with that ID.",
                id
            );
            println!("   Use 'raven orchestrate status' to list stored items.");
        }
        OrchestrateAction::Cancel { id } => {
            let store = odin_orchestrator::persistence::SqliteOrchestrationStore::new(
                dirs_state_path("orchestration.db"),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Store: {e}"))?;
            store
                .initialize()
                .await
                .map_err(|e| anyhow::anyhow!("Store init: {e}"))?;

            match store.update_graph_status(&id, "cancelled").await {
                Ok(()) => {
                    let command = odin_orchestrator::RunControlCommand::new(
                        &id,
                        odin_orchestrator::RunControlKind::Cancel,
                        "cli:orchestrate-cancel",
                        Some("cancelled via raven orchestrate cancel".into()),
                    );
                    if let Err(error) = store.enqueue_control(&command).await {
                        println!("⚠️  Status updated but live control enqueue failed: {error}");
                    } else {
                        println!("🛑 Task graph '{}' cancelled (live control enqueued).", id);
                    }
                    // Also cancel any associated agent lifecycles
                    let graph_root_id = store
                        .list_task_graphs()
                        .await
                        .unwrap_or_default()
                        .into_iter()
                        .find(|graph| graph.run_id == id || graph.root_goal == id)
                        .map(|graph| graph.run_id);
                    let lifecycles = store.list_agent_lifecycles().await.unwrap_or_default();
                    for lc in &lifecycles {
                        if graph_root_id.as_deref() == lc.graph_root_id.as_deref()
                            && lc.phase != "done"
                            && lc.phase != "failed"
                            && lc.phase != "cancelled"
                        {
                            let _ = store
                                .update_lifecycle_phase(&lc.agent_id, "cancelled")
                                .await;
                        }
                    }
                }
                Err(_) => {
                    // Try as agent lifecycle ID
                    match store.update_lifecycle_phase(&id, "cancelled").await {
                        Ok(()) => println!("🛑 Agent lifecycle '{}' cancelled.", id),
                        Err(_) => {
                            println!(
                                "❌ Not found: '{}' — no task graph or agent with that ID.",
                                id
                            );
                            println!("   Use 'raven orchestrate status' to list stored items.");
                        }
                    }
                }
            }
        }
        OrchestrateAction::Pause => {
            let store = odin_orchestrator::persistence::SqliteOrchestrationStore::new(
                dirs_state_path("orchestration.db"),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Store: {e}"))?;
            store
                .initialize()
                .await
                .map_err(|e| anyhow::anyhow!("Store init: {e}"))?;

            let graphs = store.list_task_graphs().await.unwrap_or_default();
            let mut paused = 0;
            for g in &graphs {
                if g.status != "complete" && g.status != "failed" && g.status != "cancelled" {
                    let _ = store.update_graph_status(&g.run_id, "paused").await;
                    let command = odin_orchestrator::RunControlCommand::new(
                        &g.run_id,
                        odin_orchestrator::RunControlKind::Pause,
                        "cli:orchestrate-pause",
                        None,
                    );
                    let _ = store.enqueue_control(&command).await;
                    paused += 1;
                }
            }
            println!(
                "Marked {} stored task graph(s) as paused and enqueued live pause control.",
                paused
            );
        }
        OrchestrateAction::Resume => {
            let store = odin_orchestrator::persistence::SqliteOrchestrationStore::new(
                dirs_state_path("orchestration.db"),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Store: {e}"))?;
            store
                .initialize()
                .await
                .map_err(|e| anyhow::anyhow!("Store init: {e}"))?;

            let graphs = store.list_task_graphs().await.unwrap_or_default();
            let mut resumed = 0;
            for g in &graphs {
                if g.status == "paused" {
                    let _ = store.update_graph_status(&g.run_id, "running").await;
                    let command = odin_orchestrator::RunControlCommand::new(
                        &g.run_id,
                        odin_orchestrator::RunControlKind::Resume,
                        "cli:orchestrate-resume",
                        None,
                    );
                    let _ = store.enqueue_control(&command).await;
                    resumed += 1;
                }
            }
            println!(
                "Marked {} stored task graph(s) as running and enqueued live resume control.",
                resumed
            );
        }
        OrchestrateAction::Agents => {
            let store = odin_orchestrator::persistence::SqliteOrchestrationStore::new(
                dirs_state_path("orchestration.db"),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Store: {e}"))?;
            store
                .initialize()
                .await
                .map_err(|e| anyhow::anyhow!("Store init: {e}"))?;

            let lifecycles = store.list_agent_lifecycles().await.unwrap_or_default();
            if lifecycles.is_empty() {
                println!("🤖 No stored agent lifecycles.");
                println!("   Use 'raven orchestrate submit' to create a plan.");
                return Ok(());
            }

            println!("🤖 Stored Agent Lifecycles:");
            println!();
            for lc in &lifecycles {
                let icon = match lc.phase.as_str() {
                    "running" => "🟢",
                    "queued" => "🔵",
                    "blocked" => "🟡",
                    "waiting_for_lock" => "🟠",
                    "done" => "✅",
                    "failed" => "❌",
                    "cancelled" => "🛑",
                    "paused" => "⏸️",
                    _ => "⚪",
                };
                println!("  {} {} [{}]", icon, lc.agent_id, lc.phase);
                println!("    Created: {}", lc.created_at);
                if let Some(finished) = lc.finished_at {
                    println!("    Finished: {}", finished);
                }
            }
        }
        OrchestrateAction::Locks => {
            let store = odin_orchestrator::persistence::SqliteOrchestrationStore::new(
                dirs_state_path("orchestration.db"),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Store: {e}"))?;
            store
                .initialize()
                .await
                .map_err(|e| anyhow::anyhow!("Store init: {e}"))?;

            let graphs = store.list_task_graphs().await.unwrap_or_default();
            if graphs.is_empty() {
                println!("🔒 No stored task graphs — no lock state available.");
                println!("   Use 'raven orchestrate submit' to create a plan.");
                return Ok(());
            }

            println!("🔒 Declared File Access (from stored task graphs):");
            println!();
            let mut total_write_files = 0usize;
            for g in &graphs {
                if g.status == "running" || g.status == "paused" {
                    // Load the full graph to see file requirements
                    if let Ok(graph) = store.load_task_graph(&g.run_id).await {
                        for node in graph.nodes.values() {
                            if !node.write_files.is_empty() {
                                total_write_files += node.write_files.len();
                                println!(
                                    "  ✍️  {} -> writes: {}",
                                    node.label,
                                    node.write_files.join(", ")
                                );
                            } else if !node.read_files.is_empty() {
                                println!(
                                    "  📖 {} -> reads: {}",
                                    node.label,
                                    node.read_files.join(", ")
                                );
                            }
                        }
                    }
                }
            }
            println!();
            println!("  Total declared write targets: {}", total_write_files);
        }
        OrchestrateAction::Queue => {
            let store = odin_orchestrator::persistence::SqliteOrchestrationStore::new(
                dirs_state_path("orchestration.db"),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Store: {e}"))?;
            store
                .initialize()
                .await
                .map_err(|e| anyhow::anyhow!("Store init: {e}"))?;

            let graphs = store.list_task_graphs().await.unwrap_or_default();
            let lifecycles = store.list_agent_lifecycles().await.unwrap_or_default();

            if graphs.is_empty() && lifecycles.is_empty() {
                println!("📝 Stored Orchestration Summary — no stored state.");
                println!("   Use 'raven orchestrate submit' to create a plan.");
                return Ok(());
            }

            println!("📝 Stored Orchestration Summary:");
            println!();
            println!("  Task graphs: {} stored", graphs.len());
            for g in &graphs {
                let status_icon = match g.status.as_str() {
                    "running" => "🟢",
                    "paused" => "⏸️",
                    "cancelled" => "🛑",
                    "complete" => "✅",
                    "failed" => "❌",
                    _ => "⚪",
                };
                println!(
                    "    {} {} '{}' — {} nodes [{}]",
                    status_icon, g.run_id, g.root_goal, g.node_count, g.status
                );
            }

            println!();
            println!("  Agent lifecycles: {} stored", lifecycles.len());
            let mut phase_counts: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            for lc in &lifecycles {
                *phase_counts.entry(lc.phase.clone()).or_default() += 1;
            }
            for (phase, count) in &phase_counts {
                println!("    {}: {} agent(s)", phase, count);
            }
        }
        OrchestrateAction::Restore { id } => {
            let store = odin_orchestrator::persistence::SqliteOrchestrationStore::new(
                dirs_state_path("orchestration.db"),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Store: {e}"))?;
            store
                .initialize()
                .await
                .map_err(|e| anyhow::anyhow!("Store init: {e}"))?;

            if let Some(ref run_id) = id {
                // Restore a specific run
                match store.load_task_graph(run_id).await {
                    Ok(graph) => {
                        let status = format!("{:?}", graph.status);
                        println!("♻️  Restored task graph: {}", graph.root_goal);
                        println!("   Status: {}", status);
                        println!("   Nodes: {}", graph.nodes.len());
                        for (i, node) in graph.nodes.values().enumerate() {
                            println!("     {}. {} [{:?}]", i + 1, node.label, node.status);
                        }
                        // Reset to running if currently paused
                        if graph.status == odin_orchestrator::task_graph::TaskGraphStatus::Running
                            || graph.status
                                == odin_orchestrator::task_graph::TaskGraphStatus::Building
                        {
                            // Already active
                        } else if matches!(
                            graph.status,
                            odin_orchestrator::task_graph::TaskGraphStatus::Cancelled
                                | odin_orchestrator::task_graph::TaskGraphStatus::Failed
                                | odin_orchestrator::task_graph::TaskGraphStatus::Complete
                        ) {
                            println!(
                                "   Run is finalized. Use 'raven orchestrate submit' for a new plan."
                            );
                        }
                        println!();
                        println!("   To execute, use: raven run \"{}\"", graph.root_goal);
                    }
                    Err(_) => {
                        println!("❌ Run '{}' not found in DB.", run_id);
                        println!(
                            "   Use 'raven orchestrate restore' (no args) to list unfinished runs."
                        );
                    }
                }
            } else {
                // List all unfinished runs
                match store.find_unfinished_graphs().await {
                    Ok(unfinished) => {
                        if unfinished.is_empty() {
                            println!("♻️  No unfinished runs found.");
                            println!(
                                "   All stored task graphs are complete, failed, or cancelled."
                            );
                        } else {
                            println!("♻️  Unfinished Runs ({} found):", unfinished.len());
                            println!();
                            for g in &unfinished {
                                let icon = if g.status == "running" {
                                    "🟢"
                                } else {
                                    "⏸️"
                                };
                                println!(
                                    "  {} {} '{}' — {} nodes [{}]",
                                    icon, g.run_id, g.root_goal, g.node_count, g.status
                                );
                                println!("    Updated: {}", g.updated_at);
                                println!("    Restore: raven orchestrate restore {}", g.run_id);
                                println!();
                            }
                        }
                    }
                    Err(e) => {
                        println!("⚠️  Could not query DB: {}", e);
                    }
                }
            }
        }
    }
    Ok(())
}

/// `raven serve` — Start the HTTP API server with a task handler.
async fn cmd_serve(addr: Option<String>, config_path: Option<PathBuf>) -> anyhow::Result<()> {
    // Warn about any unfinished orchestration runs from previous sessions
    warn_about_unfinished_runs().await;

    let config = load_config(config_path.as_deref())?;
    let addr = addr.unwrap_or_else(|| config.gateway.http_addr.clone());
    let management_addr = config.gateway.management_addr.clone();
    let (operator_token, generated_operator_token) = if let Some(token) = config
        .gateway
        .control_token
        .clone()
        .filter(|token| !token.trim().is_empty())
    {
        (token, false)
    } else if let Some(env_name) = config.gateway.control_token_env.as_deref() {
        let token = std::env::var(env_name).map_err(|_| {
            anyhow::anyhow!(
                "gateway.control_token_env names '{env_name}', but that environment variable is not set"
            )
        })?;
        if token.trim().is_empty() {
            anyhow::bail!(
                "gateway.control_token_env names '{env_name}', but that environment variable is empty"
            );
        }
        (token, false)
    } else {
        (
            format!(
                "{}{}",
                uuid::Uuid::new_v4().simple(),
                uuid::Uuid::new_v4().simple()
            ),
            true,
        )
    };
    let public_auth_token = if let Some(token) = config
        .gateway
        .public_auth_token
        .clone()
        .filter(|token| !token.trim().is_empty())
    {
        Some(token)
    } else if let Some(env_name) = config.gateway.public_auth_token_env.as_deref() {
        let token = std::env::var(env_name).map_err(|_| {
            anyhow::anyhow!(
                "gateway.public_auth_token_env names '{env_name}', but that environment variable is not set"
            )
        })?;
        if token.trim().is_empty() {
            anyhow::bail!(
                "gateway.public_auth_token_env names '{env_name}', but that environment variable is empty"
            );
        }
        Some(token)
    } else {
        None
    };
    tracing::info!("[CLI] Starting HTTP server on {addr}");
    let composition = ProductionComposition::from_config(&config)?;
    let sandbox = Arc::new(odin_tools::Sandbox::new(config.tools.path_boundary.clone()));
    let enabled_tools = config.tools.effective_enabled_tools();

    let audit_logger = Arc::new(build_audit_logger(&config)?);
    let approval_gate = Arc::new(odin_permissions::ApprovalGate::default());
    let policy_engine = Arc::new(
        odin_permissions::PolicyEngine::new(
            config.safety.permissions.clone(),
            &config.safety.dangerous_commands,
            config.tools.path_boundary.clone(),
            config.safety.max_rate_per_minute,
            config.safety.require_approval,
        )
        .with_approval_gate(approval_gate.clone())
        .with_audit_logger(audit_logger.clone()),
    );
    tracing::info!("[CLI/serve] Policy engine initialized");

    let tool_registry = Arc::new(build_tool_registry_with(sandbox, Some(&enabled_tools)));

    // Load MCP tools from configured servers
    load_mcp_tools(&tool_registry, &config).await;

    // Wire persistent memory store
    let memory = Arc::new(build_memory_store(&config)?);
    tracing::info!("[CLI/serve] Memory store initialized");

    // Reuse the policy engine's logger so the audit sink has one writer.
    let reliability_tracker = build_reliability_tracker(&config)?;
    let composition = composition.with_resources(CompositionResources {
        policy_engine: Some(policy_engine.clone()),
        tool_registry: Some(tool_registry.clone()),
        audit_logger: Some(audit_logger.clone()),
        reliability_tracker: Some(reliability_tracker),
    });
    tracing::info!("[CLI/serve] Audit logger initialized");

    // Discord shares the configured provider, tools, memory, policy, and audit
    // surfaces with HTTP, but has its own long-lived runtime agent.
    let discord_gateway = if config.gateway.discord_enabled {
        let token = config.gateway.discord_token.clone().or_else(|| {
            config
                .gateway
                .discord_token_env
                .as_deref()
                .and_then(|name| std::env::var(name).ok())
        });
        if token.is_none() {
            anyhow::bail!(
                "Discord is enabled but no token is configured. Set gateway.discord_token or gateway.discord_token_env."
            );
        }

        let agent = composition.build_agent(
            ExecutionSurface::Discord,
            "discord-agent",
            EngineBuildOptions::default(),
        );
        let runtime = Arc::new(Runtime::new().with_memory(memory.clone()));
        runtime.register_agent(agent);

        let gateway = odin_gateway::DiscordGateway::new(
            odin_gateway::DiscordConfig {
                enabled: true,
                token,
                admin_role: config.gateway.discord_admin_role.clone(),
                admin_user_ids: config.gateway.discord_admin_user_ids.clone(),
                guild_id: config.gateway.discord_guild_id,
                allow_dms: config.gateway.discord_allow_dms,
                command_prefix: None,
                orchestration_db_path: Some(dirs_state_path("orchestration.db")),
                agent_selector: Some("discord-agent".into()),
            },
            runtime,
        );
        gateway
            .start()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to start Discord gateway: {e}"))?;
        Some(gateway)
    } else {
        None
    };

    // Public HTTP submissions share one process-lifetime security principal so
    // repeated requests cannot reset per-principal tool rate limits.
    let serve_principal_id = uuid::Uuid::new_v4();

    // Build the task handler closure
    let handler: odin_gateway::TaskHandlerFn = {
        let memory = memory;
        let audit_logger = audit_logger.clone();
        let composition = composition.clone();
        Arc::new(move |req: odin_gateway::ChatRequest| {
            let composition = composition.clone();
            let memory = memory.clone();
            let audit_logger = audit_logger.clone();
            Box::pin(async move {
                let start = std::time::Instant::now();
                let max_iterations = req
                    .max_iterations
                    .unwrap_or(composition.default_max_iterations());

                let agent = composition.build_agent(
                    ExecutionSurface::Http,
                    "serve-agent",
                    EngineBuildOptions {
                        principal_id: Some(serve_principal_id),
                        max_iterations: Some(max_iterations),
                        ..Default::default()
                    },
                );
                let agent_id = agent.id;

                let runtime = Runtime::new().with_memory(memory.clone());
                runtime.register_agent(agent);

                // Parse session_id if provided
                let session_id = req
                    .session_id
                    .clone()
                    .and_then(|sid| uuid::Uuid::parse_str(&sid).ok())
                    .unwrap_or_else(uuid::Uuid::new_v4);

                let runtime_task = AgentTask {
                    id: uuid::Uuid::new_v4(),
                    goal: req.task.clone(),
                    context: req.context.clone(),
                    sub_tasks: vec![],
                    success_criteria: vec![],
                    max_iterations,
                    created_at: chrono::Utc::now(),
                };

                // Log task start
                let start_entry = odin_core::types::AuditEntry {
                    id: uuid::Uuid::new_v4(),
                    timestamp: chrono::Utc::now(),
                    agent_id,
                    session_id,
                    event_type: odin_core::types::AuditEventType::SessionStart,
                    action: "serve_run".to_string(),
                    details: serde_json::json!({
                        "task_id": runtime_task.id,
                        "session_id": session_id,
                        "task": req.task,
                        "max_iterations": max_iterations,
                    }),
                    result: odin_core::types::AuditResult::Success,
                };
                audit_logger.log(start_entry).await?;

                let result = runtime
                    .submit_task(&agent_id, &runtime_task, Some(session_id))
                    .await?;

                let elapsed = start.elapsed().as_millis() as u64;

                // Log task end
                let end_entry = odin_core::types::AuditEntry {
                    id: uuid::Uuid::new_v4(),
                    timestamp: chrono::Utc::now(),
                    agent_id,
                    session_id,
                    event_type: odin_core::types::AuditEventType::SessionEnd,
                    action: "serve_run_complete".to_string(),
                    details: serde_json::json!({
                        "task_id": result.task_id,
                        "session_id": session_id,
                        "success": result.success,
                        "iterations": result.iterations,
                        "duration_ms": elapsed,
                        "tool_calls": result.tool_calls,
                        "confidence": result.confidence,
                    }),
                    result: if result.success {
                        odin_core::types::AuditResult::Success
                    } else {
                        odin_core::types::AuditResult::Failure
                    },
                };
                audit_logger.log(end_entry).await?;

                Ok(odin_gateway::ChatResponse {
                    success: result.success,
                    summary: result.summary,
                    iterations: result.iterations,
                    tool_calls: result.tool_calls,
                    duration_ms: elapsed.max(result.duration_ms),
                    confidence: result.confidence,
                    error: result.error,
                })
            })
        })
    };

    println!("╔══════════════════════════════════════════╗");
    println!("║     Raven Agent — API Gateway           ║");
    println!("╠══════════════════════════════════════════╣");
    println!("║  HTTP API: http://{addr:<15}  ║", addr = addr);
    println!("║  Health:   http://{addr:<15}/health  ║", addr = addr);
    println!("║  Chat:     POST http://{addr:<15}/chat  ║", addr = addr);
    println!("║  WebSocket: ws://{addr:<15}/ws    ║", addr = addr);
    println!("╚══════════════════════════════════════════╝");
    println!("Operator API: http://{management_addr}");
    if generated_operator_token {
        println!("Generated operator token (shown once): {operator_token}");
    }
    println!();

    let orch_store = Arc::new(
        odin_orchestrator::persistence::SqliteOrchestrationStore::new(dirs_state_path(
            "orchestration.db",
        ))
        .await
        .map_err(|error| anyhow::anyhow!("Failed to open orchestration store: {error}"))?,
    );
    orch_store
        .initialize()
        .await
        .map_err(|error| anyhow::anyhow!("Failed to initialize orchestration store: {error}"))?;
    let ws_manager = Arc::new(
        odin_gateway::ws::WsConnectionManager::new(256)
            .with_control(orch_store, Some(operator_token.clone())),
    );
    let server_result = odin_gateway::run_http_server_with_management_auth_and_budgets(
        &addr,
        &management_addr,
        operator_token,
        public_auth_token,
        config.gateway.allow_insecure_non_loopback,
        config.effective_resource_budgets(),
        Some(handler),
        Some(ws_manager),
        Some(tool_registry),
        Some(approval_gate),
    )
    .await;

    if let Some(gateway) = discord_gateway
        && let Err(error) = gateway.stop().await
    {
        tracing::warn!("[CLI/serve] Failed to stop Discord gateway: {error}");
    }

    if let Err(error) = audit_logger.flush().await {
        tracing::error!(%error, "Failed to flush audit sink during shutdown");
        if server_result.is_ok() {
            return Err(anyhow::anyhow!("Failed to flush audit sink: {error}"));
        }
    }

    server_result.map_err(|e| anyhow::anyhow!("Server error: {e}"))?;

    Ok(())
}

/// `raven config` — Show or edit configuration.
fn cmd_config(path: PathBuf, edit: bool) -> anyhow::Result<()> {
    let expanded_path = shellexpand::tilde(&path.to_string_lossy()).to_string();
    let config_path = PathBuf::from(&expanded_path);

    if edit {
        // Open the config in the user's editor
        let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vim".into());
        let status = std::process::Command::new(&editor)
            .arg(&config_path)
            .status()
            .map_err(|e| anyhow::anyhow!("Failed to open editor '{}': {e}", editor))?;

        if !status.success() {
            anyhow::bail!("Editor exited with non-zero status");
        }

        println!("Config saved to: {}", config_path.display());
    } else {
        // Show config
        if config_path.exists() {
            let contents = std::fs::read_to_string(&config_path).map_err(|e| {
                anyhow::anyhow!("Failed to read config {}: {e}", config_path.display())
            })?;
            println!("Configuration file: {}", config_path.display());
            println!("---");
            println!(
                "{}",
                odin_permissions::SecretRedactor::full().redact(&contents)
            );
        } else {
            println!("No config file found at {}", config_path.display());
            println!("Creating with defaults...");

            let config = OdinConfig::default();
            if let Some(parent) = config_path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            config
                .save(&config_path)
                .map_err(|e| anyhow::anyhow!("Failed to save config: {e}"))?;

            println!("Created default config at {}", config_path.display());
        }
    }

    Ok(())
}

/// `raven version` — Show version information.
fn cmd_version() -> anyhow::Result<()> {
    println!("Raven Agent {}", env!("CARGO_PKG_VERSION"));
    println!("Repository: https://github.com/hermes-gadget/raven-ai-harness");
    println!("License:    {}", env!("CARGO_PKG_LICENSE"));
    println!();
    println!("Built with:");
    println!("  Rust: {}", rustc_version());
    println!("  Profile: {}", build_profile());
    println!("  Target:  {}", std::env::consts::ARCH);

    Ok(())
}

/// `raven ui` — Start the interactive terminal UI.
///
/// Launches a ratatui-based TUI with a chat panel and side panel
/// showing live orchestration state from the persistent SQLite store.
async fn cmd_ui(db_path: Option<PathBuf>) -> anyhow::Result<()> {
    tracing::info!("[CLI] Starting interactive TUI...");
    odin_tui::start(db_path).await?;
    Ok(())
}

/// `raven schedule` — Manage scheduled cron jobs.
async fn cmd_schedule(action: ScheduleAction, config_path: Option<PathBuf>) -> anyhow::Result<()> {
    let config = load_config(config_path.as_deref())?;
    let db_path = config.scheduler.db_path.as_ref().map_or_else(
        || {
            config.general.data_dir.clone().map_or_else(
                || PathBuf::from(shellexpand::tilde("~/.raven-agent/scheduler.db").to_string()),
                |dir| dir.join("scheduler.db"),
            )
        },
        |path| PathBuf::from(shellexpand::tilde(path).to_string()),
    );
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            anyhow::anyhow!(
                "Failed to create scheduler data directory '{}': {error}",
                parent.display()
            )
        })?;
    }
    let db_path_str = db_path.to_string_lossy();
    let store = Arc::new(SqliteSchedulerStore::new(&db_path_str).map_err(|error| {
        anyhow::anyhow!(
            "Failed to open scheduler database '{}': {error}",
            db_path.display()
        )
    })?);
    let scheduler = Scheduler::new(config.scheduler.clone()).with_store(store);
    scheduler.load_persisted().await?;

    match action {
        ScheduleAction::Host => {
            if !config.scheduler.enabled {
                println!(
                    "Scheduler host is disabled. Set scheduler.enabled: true in the Raven config."
                );
                return Ok(());
            }

            let composition = ProductionComposition::from_config(&config)?;
            let policy_engine = Arc::new(odin_permissions::PolicyEngine::new(
                config.safety.permissions.clone(),
                &config.safety.dangerous_commands,
                config.tools.path_boundary.clone(),
                config.safety.max_rate_per_minute,
                config.safety.require_approval,
            ));
            let sandbox = Arc::new(odin_tools::Sandbox::new(config.tools.path_boundary.clone()));
            let enabled_tools = config.tools.effective_enabled_tools();
            let tool_registry = Arc::new(build_tool_registry_with(sandbox, Some(&enabled_tools)));
            load_mcp_tools(&tool_registry, &config).await;
            let audit_logger = Arc::new(build_audit_logger(&config)?);
            let composition = composition.with_resources(CompositionResources {
                policy_engine: Some(policy_engine),
                tool_registry: Some(tool_registry),
                audit_logger: Some(audit_logger.clone()),
                reliability_tracker: Some(build_reliability_tracker(&config)?),
            });
            let agent = composition.build_agent(
                ExecutionSurface::Scheduler,
                "scheduler-host",
                EngineBuildOptions::default(),
            );
            let runtime = Arc::new(
                Runtime::new()
                    .with_memory(Arc::new(build_memory_store(&config)?))
                    .with_default_max_iterations(config.agent.max_iterations),
            );
            runtime.register_agent(agent);

            let store = Arc::new(SqliteSchedulerStore::new(&db_path_str)?);
            let host = Scheduler::new(config.scheduler.clone())
                .with_store(store)
                .with_runtime(runtime)
                .with_audit_logger(audit_logger);
            host.start().await?;
            println!(
                "Scheduler host running: {} job(s), tick={}s, concurrency={}.",
                host.job_count().await,
                config.scheduler.check_interval_secs,
                config.scheduler.max_concurrent
            );
            println!("Press Ctrl-C to stop; in-flight jobs will be allowed to finish.");
            wait_for_shutdown_signal().await?;
            println!("Shutdown requested; draining scheduler jobs...");
            host.stop().await?;
            println!("Scheduler host stopped.");
            return Ok(());
        }
        ScheduleAction::Status => {
            let jobs = scheduler.list_jobs().await;
            let enabled_jobs = jobs.iter().filter(|job| job.enabled).count();
            let latest = scheduler.recent_runs(1).await?.into_iter().next();
            println!("Scheduler health:");
            println!(
                "  Host mode: {}",
                if config.scheduler.enabled {
                    "enabled"
                } else {
                    "disabled"
                }
            );
            println!("  Database: {}", db_path.display());
            println!(
                "  Jobs: {} total, {} enabled, {} disabled",
                jobs.len(),
                enabled_jobs,
                jobs.len() - enabled_jobs
            );
            println!(
                "  Tick/concurrency: {}s / {}",
                config.scheduler.check_interval_secs, config.scheduler.max_concurrent
            );
            if let Some(run) = latest {
                println!(
                    "  Latest outcome: {:?} — {} at {}{}",
                    run.status,
                    run.job_name,
                    run.started_at.to_rfc3339(),
                    run.error
                        .as_deref()
                        .map(|error| format!(" ({error})"))
                        .unwrap_or_default()
                );
            } else {
                println!("  Latest outcome: none");
            }
        }
        ScheduleAction::History { limit } => {
            let runs = scheduler.recent_runs(limit).await?;
            if runs.is_empty() {
                println!("No scheduler execution history.");
            }
            for run in runs {
                println!(
                    "{:?} {} job={} task={} started={}{}",
                    run.status,
                    run.job_name,
                    run.job_id,
                    run.task_id,
                    run.started_at.to_rfc3339(),
                    run.error
                        .as_deref()
                        .map(|error| format!(" error={error}"))
                        .unwrap_or_default()
                );
            }
        }
        ScheduleAction::Add {
            name,
            schedule,
            task,
            agent,
        } => {
            // Parse the schedule to validate it before adding
            if let Err(e) = odin_scheduler::Schedule::parse(&schedule) {
                anyhow::bail!("Invalid cron expression '{}': {}", schedule, e);
            }

            let mut job_config = SchedulerJobConfig::new(task);
            if let Some(agent) = agent {
                job_config = job_config.with_agent_selector(agent);
            }
            let job_id = scheduler
                .add_job_with_config(&name, &schedule, job_config)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to add job: {}", e))?;
            println!("Job '{}' added with ID: {}", name, job_id);
        }
        ScheduleAction::List => {
            let jobs = scheduler.list_jobs().await;
            if jobs.is_empty() {
                println!("No scheduled jobs.");
            } else {
                println!("Scheduled jobs:");
                println!();
                for job in &jobs {
                    let status = if job.enabled { "enabled" } else { "disabled" };
                    println!(
                        "  {} — {} ({}) [{}]",
                        job.id, job.name, job.schedule, status
                    );
                    println!(
                        "         Last run: {}",
                        job.last_run.map_or("never".into(), |t| t.to_rfc3339())
                    );
                    println!(
                        "         Next run: {}",
                        job.next_run.map_or("none".into(), |t| t.to_rfc3339())
                    );
                    println!("         Runs: {}", job.run_count);
                }
            }
        }
        ScheduleAction::Remove { job_id } => {
            let id: JobId = job_id
                .parse()
                .map_err(|e| anyhow::anyhow!("Invalid job ID '{}': {}", job_id, e))?;
            match scheduler.remove_job(id).await {
                Ok(true) => println!("Job '{}' removed.", job_id),
                Ok(false) => println!("Job '{}' not found.", job_id),
                Err(e) => anyhow::bail!("Failed to remove job: {}", e),
            }
        }
        ScheduleAction::Enable { job_id } => {
            let id: JobId = job_id
                .parse()
                .map_err(|e| anyhow::anyhow!("Invalid job ID '{}': {}", job_id, e))?;
            match scheduler.set_job_enabled(id, true).await {
                Ok(true) => println!("Job '{}' enabled.", job_id),
                Ok(false) => println!("Job '{}' not found.", job_id),
                Err(e) => anyhow::bail!("Failed to enable job: {}", e),
            }
        }
        ScheduleAction::Disable { job_id } => {
            let id: JobId = job_id
                .parse()
                .map_err(|e| anyhow::anyhow!("Invalid job ID '{}': {}", job_id, e))?;
            match scheduler.set_job_enabled(id, false).await {
                Ok(true) => println!("Job '{}' disabled.", job_id),
                Ok(false) => println!("Job '{}' not found.", job_id),
                Err(e) => anyhow::bail!("Failed to disable job: {}", e),
            }
        }
    }

    Ok(())
}

#[cfg(unix)]
async fn wait_for_shutdown_signal() -> anyhow::Result<()> {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    tokio::select! {
        result = tokio::signal::ctrl_c() => result?,
        _ = terminate.recv() => {}
    }
    Ok(())
}

#[cfg(not(unix))]
async fn wait_for_shutdown_signal() -> anyhow::Result<()> {
    tokio::signal::ctrl_c().await?;
    Ok(())
}

/// `raven providers list` — List configured providers.
async fn cmd_providers(
    action: ProvidersAction,
    config_path: Option<PathBuf>,
) -> anyhow::Result<()> {
    match action {
        ProvidersAction::List => {
            let config = load_config(config_path.as_deref())?;
            let providers = &config.models.providers;
            let default = &config.models.default_provider;

            if providers.is_empty() {
                println!("No providers configured.");
                return Ok(());
            }

            println!("Configured providers:");
            println!();
            for (name, cfg) in providers {
                let is_default = if name == default { " (default)" } else { "" };
                println!(
                    "  {name}{is_default}:",
                    name = name,
                    is_default = is_default
                );
                println!("    Type:      {}", cfg.provider_type);
                if let Some(ref url) = cfg.base_url {
                    println!("    Base URL:  {}", url);
                }
                if let Some(ref model) = cfg.default_model {
                    println!("    Model:     {}", model);
                }
                println!();
            }
        }
    }

    Ok(())
}

/// `raven eval ...` — Run small/local/cheap model evaluations.
async fn cmd_eval(action: EvalAction) -> anyhow::Result<()> {
    match action {
        EvalAction::Mocked {
            profile,
            format,
            output,
        } => {
            let profile = odin_loop::SmallModelProfile::by_id(&profile).ok_or_else(|| {
                anyhow::anyhow!(
                    "Unknown profile '{}'. Run `raven eval profiles` to list built-ins.",
                    profile
                )
            })?;
            let report = odin_eval::run_mocked_eval(profile).await?;
            let rendered = render_eval_output(&format, &report)?;

            if let Some(path) = output {
                std::fs::write(&path, rendered)?;
                println!("Wrote eval report to {}", path.display());
            } else {
                println!("{rendered}");
            }
        }
        EvalAction::Profiles { format } => {
            let profiles = odin_loop::SmallModelProfile::built_ins();
            let rendered = match format.as_str() {
                "table" | "markdown" => odin_eval::render_profiles_table(&profiles),
                "json" => serde_json::to_string_pretty(&profiles)?,
                other => anyhow::bail!("Unsupported eval profile format '{other}'"),
            };
            println!("{rendered}");
        }
        EvalAction::Live {
            provider,
            model,
            base_url,
            api_key_env,
        } => {
            let config = odin_eval::LiveEvalConfig {
                provider,
                model,
                base_url,
                api_key_env,
            };
            let readiness = odin_eval::check_live_eval_readiness(&config);
            if readiness.ready {
                println!("Live eval readiness: ready — {}", readiness.reason);
                println!(
                    "Live provider execution is opt-in; use this readiness gate before running provider-backed eval jobs."
                );
            } else {
                anyhow::bail!("Live eval is not configured: {}", readiness.reason);
            }
        }
    }

    Ok(())
}

fn render_eval_output(format: &str, report: &odin_eval::EvalReport) -> anyhow::Result<String> {
    match format {
        "table" | "markdown" => Ok(odin_eval::render_report_table(report)),
        "json" => Ok(serde_json::to_string_pretty(report)?),
        other => anyhow::bail!("Unsupported eval report format '{other}'"),
    }
}

/// `raven skills list` — List available skills.
async fn cmd_skills(action: SkillsAction) -> anyhow::Result<()> {
    match action {
        SkillsAction::List { dir } => {
            let skills_dir = dir
                .map(PathBuf::from)
                .or_else(|| Some(default_skills_dir()))
                .unwrap();

            if !skills_dir.exists() {
                println!("Skills directory not found: {}", skills_dir.display());
                println!("Create it with: mkdir -p {}", skills_dir.display());
                return Ok(());
            }

            let registry = odin_skills::SkillRegistry::load_from_dir(&skills_dir)
                .map_err(|e| anyhow::anyhow!("Failed to load skills: {e}"))?;
            let skills = registry.all();
            let enabled = registry.enabled();

            if skills.is_empty() {
                println!("No skills found in {}", skills_dir.display());
                return Ok(());
            }

            // Build available tool names from the tool registry for validation
            let tool_registry = build_tool_registry();
            let available: Vec<String> = tool_registry
                .all_tools()
                .iter()
                .map(|t| t.name().to_string())
                .collect();
            let validations = registry.validate_tools(&available);

            println!("Skills ({}/{} enabled):", enabled.len(), skills.len());
            println!();
            for skill in &skills {
                let status = if skill.enabled { "✓" } else { " " };
                let req_count = skill.required_tools.len();
                let rec_count = skill.recommended_tools.len();
                let tool_info = if req_count > 0 || rec_count > 0 {
                    format!(" [{} req, {} rec]", req_count, rec_count)
                } else {
                    String::new()
                };
                println!(
                    "  {status} {name:<30} {desc}{tool_info}",
                    name = skill.name,
                    desc = skill.description
                );
            }

            // Show validation warnings
            if !validations.is_empty() {
                println!();
                println!("⚠  Tool availability warnings:");
                for v in &validations {
                    if v.has_errors {
                        println!(
                            "  ✗ {} — missing required: {}",
                            v.skill_name,
                            v.missing_required.join(", ")
                        );
                    } else if !v.missing_required.is_empty() {
                        println!(
                            "  ⚠ {} — missing required: {}",
                            v.skill_name,
                            v.missing_required.join(", ")
                        );
                    }
                    if !v.missing_recommended.is_empty() {
                        println!(
                            "  ℹ {} — missing recommended: {}",
                            v.skill_name,
                            v.missing_recommended.join(", ")
                        );
                    }
                }
            }

            println!();
            println!("Skills directory: {}", skills_dir.display());
        }
        SkillsAction::Tools { name } => {
            let skills_dir = default_skills_dir();

            let registry = if skills_dir.exists() {
                odin_skills::SkillRegistry::load_from_dir(&skills_dir)
                    .map_err(|e| anyhow::anyhow!("Failed to load skills: {e}"))?
            } else {
                println!("Skills directory not found: {}", skills_dir.display());
                return Ok(());
            };

            match registry.tools_for_skill(&name) {
                Some(tools) => {
                    println!("Skill: {}", tools.skill_name);
                    println!();
                    if tools.required.is_empty() && tools.recommended.is_empty() {
                        println!("  No tool dependencies declared.");
                    } else {
                        if !tools.required.is_empty() {
                            println!("  Required tools:");
                            for t in &tools.required {
                                println!("    - {t}");
                            }
                        }
                        if !tools.recommended.is_empty() {
                            println!("  Recommended tools:");
                            for t in &tools.recommended {
                                println!("    - {t}");
                            }
                        }
                    }

                    // Cross-check with tool registry
                    let tool_registry = build_tool_registry();
                    let available: Vec<String> = tool_registry
                        .all_tools()
                        .iter()
                        .map(|t| t.name().to_string())
                        .collect();

                    let missing_req: Vec<_> = tools
                        .required
                        .iter()
                        .filter(|t| !available.contains(t))
                        .collect();
                    let missing_rec: Vec<_> = tools
                        .recommended
                        .iter()
                        .filter(|t| !available.contains(t))
                        .collect();

                    if !missing_req.is_empty() {
                        println!();
                        println!("  ⚠  Missing required tools (skill may not work):");
                        for t in &missing_req {
                            println!("    - {t}");
                        }
                    }
                    if !missing_rec.is_empty() {
                        println!();
                        println!("  ℹ  Unavailable recommended tools:");
                        for t in &missing_rec {
                            println!("    - {t}");
                        }
                    }
                    if missing_req.is_empty() && missing_rec.is_empty() {
                        println!();
                        println!("  ✓ All tool dependencies satisfied.");
                    }
                }
                None => {
                    println!("Skill '{name}' not found.");
                    println!();
                    println!("Available skills:");
                    for skill in registry.all() {
                        println!("  - {}", skill.name);
                    }
                }
            }
        }
    }

    Ok(())
}

/// `raven tasks` — List and inspect tasks.
async fn cmd_tasks(action: TasksAction) -> anyhow::Result<()> {
    match action {
        TasksAction::List { limit, status } => {
            let config = load_config(None)?;
            let audit_path = configured_audit_path(&config);
            let path = audit_path.as_path();

            if !path.exists() {
                println!("No audit log found at {}", audit_path.display());
                return Ok(());
            }

            let contents = read_redacted_audit(path)?;

            let mut entries = Vec::new();
            for line in contents.lines().rev() {
                if entries.len() >= limit {
                    break;
                }
                if let Ok(entry) = serde_json::from_str::<odin_core::types::AuditEntry>(line) {
                    // Filter by status if provided
                    if let Some(ref s) = status {
                        let status_match = match s.as_str() {
                            "success" => entry.result == odin_core::types::AuditResult::Success,
                            "failure" | "fail" => {
                                entry.result == odin_core::types::AuditResult::Failure
                            }
                            _ => true,
                        };
                        if !status_match {
                            continue;
                        }
                    }
                    entries.push(entry);
                }
            }

            if entries.is_empty() {
                println!("No tasks found in audit log.");
                return Ok(());
            }

            println!("Recent tasks (last {}):", entries.len());
            println!();
            for entry in &entries {
                let icon = match entry.result {
                    odin_core::types::AuditResult::Success => "✓",
                    odin_core::types::AuditResult::Failure => "✗",
                    odin_core::types::AuditResult::Denied => "−",
                    odin_core::types::AuditResult::Pending => "○",
                };
                println!(
                    "  {icon} {id} [{event}] {action} — {ts}",
                    id = entry.id,
                    event = entry.event_type,
                    action = entry.action,
                    ts = entry.timestamp.to_rfc3339()
                );
            }
        }
        TasksAction::Inspect { id } => {
            let config = load_config(None)?;
            let audit_path = configured_audit_path(&config);
            let path = audit_path.as_path();

            if !path.exists() {
                println!("No audit log found at {}", audit_path.display());
                return Ok(());
            }

            let contents = read_redacted_audit(path)?;

            let target_id = uuid::Uuid::parse_str(&id)
                .map_err(|e| anyhow::anyhow!("Invalid task ID '{id}': {e}"))?;

            for line in contents.lines() {
                if let Ok(entry) = serde_json::from_str::<odin_core::types::AuditEntry>(line)
                    && entry.id == target_id
                {
                    println!("Task: {}", entry.id);
                    println!("  Event:    {}", entry.event_type);
                    println!("  Action:   {}", entry.action);
                    println!("  Agent:    {}", entry.agent_id);
                    println!("  Session:  {}", entry.session_id);
                    println!("  Time:     {}", entry.timestamp.to_rfc3339());
                    println!("  Result:   {:?}", entry.result);
                    println!(
                        "  Details:  {}",
                        serde_json::to_string_pretty(&entry.details)?
                    );
                    return Ok(());
                }
            }

            println!("Task '{id}' not found in audit log.");
        }
    }

    Ok(())
}

/// `raven sessions` — List and inspect sessions.
async fn cmd_sessions(action: SessionsAction) -> anyhow::Result<()> {
    match action {
        SessionsAction::List => {
            let config = load_config(None)?;
            let audit_path = configured_audit_path(&config);
            let path = audit_path.as_path();

            if !path.exists() {
                println!("No audit log found at {}", audit_path.display());
                return Ok(());
            }

            let contents = read_redacted_audit(path)?;

            let mut sessions: std::collections::BTreeMap<
                uuid::Uuid,
                Vec<odin_core::types::AuditEntry>,
            > = std::collections::BTreeMap::new();

            for line in contents.lines() {
                if let Ok(entry) = serde_json::from_str::<odin_core::types::AuditEntry>(line) {
                    sessions.entry(entry.session_id).or_default().push(entry);
                }
            }

            if sessions.is_empty() {
                println!("No sessions found.");
                return Ok(());
            }

            println!("Sessions ({}):", sessions.len());
            println!();
            for (session_id, entries) in &sessions {
                let first_ts = entries.first().map(|e| e.timestamp.to_rfc3339());
                let last_ts = entries.last().map(|e| e.timestamp.to_rfc3339());
                println!("  Session: {session_id}");
                println!("    Entries: {}", entries.len());
                if let Some(ref ts) = first_ts {
                    println!("    First:   {ts}");
                }
                if let Some(ref ts) = last_ts {
                    println!("    Last:    {ts}");
                }
                let has_error = entries
                    .iter()
                    .any(|e| e.result == odin_core::types::AuditResult::Failure);
                if has_error {
                    println!("    Status:  ⚠ has failures");
                } else {
                    println!("    Status:  ✓ all ok");
                }
                println!();
            }
        }
        SessionsAction::Inspect { id } => {
            let config = load_config(None)?;
            let audit_path = configured_audit_path(&config);
            let path = audit_path.as_path();

            if !path.exists() {
                println!("No audit log found at {}", audit_path.display());
                return Ok(());
            }

            let contents = read_redacted_audit(path)?;

            let session_id = uuid::Uuid::parse_str(&id)
                .map_err(|e| anyhow::anyhow!("Invalid session ID '{id}': {e}"))?;

            let mut entries = Vec::new();
            for line in contents.lines() {
                if let Ok(entry) = serde_json::from_str::<odin_core::types::AuditEntry>(line)
                    && entry.session_id == session_id
                {
                    entries.push(entry);
                }
            }

            if entries.is_empty() {
                println!("Session '{id}' not found.");
                return Ok(());
            }

            println!("Session: {id}");
            println!("  Entries: {}", entries.len());
            println!();
            for entry in &entries {
                let icon = match entry.result {
                    odin_core::types::AuditResult::Success => "✓",
                    odin_core::types::AuditResult::Failure => "✗",
                    odin_core::types::AuditResult::Denied => "−",
                    odin_core::types::AuditResult::Pending => "○",
                };
                println!(
                    "  {icon} {action:<30} [{event:<15}] {ts}",
                    action = entry.action,
                    event = format!("{:?}", entry.event_type),
                    ts = entry.timestamp.to_rfc3339()
                );
            }
        }
    }

    Ok(())
}

/// `raven tools` — Manage and inspect registered tools.
async fn cmd_tools(action: ToolsAction) -> anyhow::Result<()> {
    match action {
        ToolsAction::List { tag } => {
            let registry = build_tool_registry();
            let all_tools = registry.all_tools();

            // Filter by tags if specified
            let tools: Vec<_> = if tag.is_empty() {
                all_tools
            } else {
                all_tools
                    .into_iter()
                    .filter(|t| {
                        let tt = t.capability_tags();
                        tag.iter()
                            .all(|filter_tag| tt.iter().any(|tool_tag| tool_tag == filter_tag))
                    })
                    .collect()
            };

            if tools.is_empty() {
                println!("No tools registered.");
                if !tag.is_empty() {
                    println!("(Filtered by tags: {:?})", tag);
                }
                return Ok(());
            }

            let filter_note = if tag.is_empty() {
                String::new()
            } else {
                format!(" [filtered by tags: {:?}]", tag)
            };

            println!("╔══════════════════════════════════════════╗");
            println!(
                "║         Registered Tools ({:>2}){:>12} ║",
                tools.len(),
                filter_note
            );
            println!("╠══════════════════════════════════════════╣");

            let mut sorted = tools.clone();
            sorted.sort_by(|a, b| a.name().cmp(b.name()));
            for tool in &sorted {
                let dangerous = if tool.is_dangerous() { " ⚠" } else { "  " };
                let name = tool.name();
                let truncated = if name.len() > 28 {
                    format!("{}…", name.chars().take(27).collect::<String>())
                } else {
                    name.to_string()
                };
                println!("║ {dangerous} {truncated:<30} ║",);
            }
            println!("╚══════════════════════════════════════════╝");
        }
        ToolsAction::Inspect { name } => {
            let registry = build_tool_registry();
            match registry.get(&name) {
                Some(tool) => {
                    let schema = tool.schema();
                    println!("Tool: {}", tool.name());
                    println!("  Description:  {}", tool.description());
                    println!("  Dangerous:    {}", tool.is_dangerous());
                    println!("  Safe:         {}", tool.is_safe());
                    println!("  Requires approval: {}", tool.requires_approval());
                    println!("  Capabilities: {:?}", tool.capability_tags());
                    println!();
                    println!("  Schema:");
                    println!(
                        "    {}",
                        serde_json::to_string_pretty(&schema).unwrap_or_else(|_| "{}".into())
                    );
                }
                None => {
                    println!("Tool '{name}' not found in registry.");
                    println!();
                    println!("Available tools:");
                    let registry = build_tool_registry();
                    let tools = registry.all_tools();
                    let mut names: Vec<String> =
                        tools.iter().map(|t| t.name().to_string()).collect();
                    names.sort();
                    for n in &names {
                        println!("  - {n}");
                    }
                }
            }
        }
        ToolsAction::Validate => {
            let registry = build_tool_registry();
            let tools = registry.all_tools();
            let mut valid = true;

            println!("Validating tool registry ({} tools)...", tools.len());
            println!();

            for tool in &tools {
                let name = tool.name();
                let mut issues = Vec::new();

                if name.is_empty() {
                    issues.push("empty name");
                }
                if tool.description().is_empty() {
                    issues.push("no description");
                }

                if issues.is_empty() {
                    println!("  ✓ {name}");
                } else {
                    println!("  ✗ {name} — {}", issues.join(", "));
                    valid = false;
                }
            }

            if valid {
                println!();
                println!("All tools valid.");
            } else {
                println!();
                println!("Validation failed — some tools have issues.");
                std::process::exit(1);
            }
        }
        ToolsAction::Test {
            name,
            args,
            dry_run,
            approve,
        } => {
            let registry = build_tool_registry();
            let config = load_config(None)?;
            let reliability_tracker = if dry_run {
                None
            } else {
                Some(build_reliability_tracker(&config)?)
            };
            match registry.get(&name) {
                Some(tool) => {
                    let json_args: serde_json::Value = match args {
                        Some(ref inline) => match serde_json::from_str(inline) {
                            Ok(value) => value,
                            Err(error) => {
                                if let Some(tracker) = &reliability_tracker {
                                    tracker
                                        .record_outcome_async(
                                            &name,
                                            odin_tools::ReliabilityOutcome::ValidationFailure,
                                            0,
                                        )
                                        .await;
                                }
                                return Err(anyhow::anyhow!("Invalid JSON args: {error}"));
                            }
                        },
                        None => serde_json::json!({}),
                    };

                    if dry_run {
                        // Wrap in DryRunTool for safe execution
                        let dry_tool = odin_tools::DryRunTool::new(tool);
                        println!("[DRY RUN] Testing tool: {}", dry_tool.name());
                        let context = odin_core::traits::ToolContext {
                            agent_id: uuid::Uuid::new_v4(),
                            session_id: uuid::Uuid::new_v4(),
                            working_dir: std::env::current_dir().unwrap_or_default(),
                            env: std::collections::HashMap::new(),
                            resource_budgets: config.effective_resource_budgets(),
                        };
                        let start = std::time::Instant::now();
                        match dry_tool.execute(json_args, &context).await {
                            Ok(result) => {
                                let elapsed = start.elapsed();
                                println!(
                                    "  Status:   {}",
                                    if result.success {
                                        "✓ PASS"
                                    } else {
                                        "✗ FAIL"
                                    }
                                );
                                println!("  Duration: {:.3}s", elapsed.as_secs_f64());
                                let redactor = odin_permissions::SecretRedactor::full();
                                println!("  Output:   {}", redactor.redact(&result.output));
                                if let Some(ref err) = result.error {
                                    println!("  Error:    {}", redactor.redact(err));
                                }
                            }
                            Err(e) => {
                                println!(
                                    "  Error: {}",
                                    odin_permissions::SecretRedactor::full().redact(&e.to_string())
                                );
                                std::process::exit(1);
                            }
                        }
                    } else {
                        if (tool.requires_approval() || tool.is_dangerous()) && !approve {
                            if let Some(tracker) = &reliability_tracker {
                                tracker
                                    .record_outcome_async(
                                        tool.name(),
                                        odin_tools::ReliabilityOutcome::PolicyDenial,
                                        0,
                                    )
                                    .await;
                            }
                            anyhow::bail!(
                                "Tool '{}' requires approval. Review the arguments, then rerun with --approve; use --dry-run to validate without execution.",
                                tool.name()
                            );
                        }
                        let redactor = odin_permissions::SecretRedactor::full();
                        // Real execution
                        println!("Testing tool: {}", tool.name());
                        println!("  Description: {}", tool.description());
                        println!(
                            "  Args:        {}",
                            redactor.redact(&serde_json::to_string(&json_args)?)
                        );
                        println!();
                        let context = odin_core::traits::ToolContext {
                            agent_id: uuid::Uuid::new_v4(),
                            session_id: uuid::Uuid::new_v4(),
                            working_dir: std::env::current_dir().unwrap_or_default(),
                            env: std::collections::HashMap::new(),
                            resource_budgets: config.effective_resource_budgets(),
                        };
                        let start = std::time::Instant::now();
                        match tool.execute(json_args, &context).await {
                            Ok(result) => {
                                let elapsed = start.elapsed();
                                if let Some(tracker) = &reliability_tracker {
                                    tracker
                                        .record_outcome_async(
                                            tool.name(),
                                            if result.success {
                                                odin_tools::ReliabilityOutcome::Success
                                            } else {
                                                odin_tools::ReliabilityOutcome::ToolFailure
                                            },
                                            elapsed.as_millis() as u64,
                                        )
                                        .await;
                                }
                                println!(
                                    "  Status:   {}",
                                    if result.success {
                                        "✓ PASS"
                                    } else {
                                        "✗ FAIL"
                                    }
                                );
                                println!("  Duration: {:.3}s", elapsed.as_secs_f64());
                                println!("  Output:   {}", redactor.redact(&result.output));
                                if let Some(ref err) = result.error {
                                    println!("  Error:    {}", redactor.redact(err));
                                }
                            }
                            Err(e) => {
                                if let Some(tracker) = &reliability_tracker {
                                    tracker
                                        .record_outcome_async(
                                            tool.name(),
                                            odin_tools::classify_tool_error(&e),
                                            start.elapsed().as_millis() as u64,
                                        )
                                        .await;
                                }
                                println!("  Error: {}", redactor.redact(&e.to_string()));
                                std::process::exit(1);
                            }
                        }
                    }
                }
                None => {
                    println!("Tool '{name}' not found.");
                    std::process::exit(1);
                }
            }
        }
        ToolsAction::Doctor => {
            let registry = build_tool_registry();
            let report = odin_tools::ToolDoctor::check(&registry);
            println!("Tool Doctor: {} tools checked", report.summary.total_tools);
            println!(
                "  Passed: {}, Failed: {}, Warnings: {}",
                report.summary.passed, report.summary.failed, report.summary.warnings
            );
            for tc in &report.tool_checks {
                let icon = if tc
                    .checks
                    .iter()
                    .any(|c| c.status == odin_tools::DoctorCheckStatus::Fail)
                {
                    "✗"
                } else {
                    "✓"
                };
                println!("  {icon} {}", tc.tool_name);
            }
            for ec in &report.ecosystem_checks {
                println!("  Ecosystem: {} — {:?}", ec.name, ec.status);
            }
            if report.summary.failed > 0 {
                std::process::exit(1);
            }
        }
        ToolsAction::Catalog {
            format,
            category,
            tag,
        } => {
            let registry = build_tool_registry();
            let catalog = odin_tools::ToolCatalog::from_registry(&registry);
            match format.as_str() {
                "json" => println!("{}", serde_json::to_string_pretty(&catalog)?),
                "yaml" => println!("{}", serde_yaml::to_string(&catalog)?),
                _ => {
                    // Table output showing categories and tools
                    let filtered_categories: Vec<String> = if let Some(ref cat) = category {
                        if catalog.by_category.contains_key(cat) {
                            vec![cat.clone()]
                        } else {
                            vec![]
                        }
                    } else {
                        catalog.categories().iter().map(|c| c.to_string()).collect()
                    };

                    for cat_name in &filtered_categories {
                        if let Some(group) = catalog.by_category(cat_name) {
                            let tools: Vec<&odin_tools::CatalogEntry> = if let Some(ref t) = tag {
                                group
                                    .tools
                                    .iter()
                                    .filter(|e| e.tags.iter().any(|gt| gt == t))
                                    .collect()
                            } else {
                                group.tools.iter().collect()
                            };

                            if tools.is_empty() {
                                continue;
                            }

                            println!("\n── {} ── {}", cat_name, group.description);
                            for entry in &tools {
                                print_tool_entry(entry);
                            }
                        }
                    }

                    println!(
                        "\n{} tools in {} categories",
                        catalog.total,
                        catalog.categories().len()
                    );
                }
            }
        }
        ToolsAction::Reliability => {
            let config = load_config(None)?;
            let tracker = build_reliability_tracker(&config)?;
            let reliability = tracker.all();
            if reliability.is_empty() {
                println!("No persisted tool reliability samples are available.");
                return Ok(());
            }

            println!(
                "{:<24} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>9}",
                "Tool", "Score", "Calls", "Success", "Policy", "Valid.", "Trans.", "Tool", "Avg ms"
            );
            for item in reliability {
                println!(
                    "{:<24} {:>6.0}% {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>9.1}",
                    item.tool_name,
                    item.score * 100.0,
                    item.total_calls,
                    item.success_count,
                    item.policy_denial_count,
                    item.validation_failure_count,
                    item.transport_failure_count,
                    item.tool_failure_count,
                    item.avg_duration_ms,
                );
                if let Some(timestamp) = item.latest_timestamp {
                    println!("{:>33} latest {}", "", timestamp.to_rfc3339());
                }
            }
        }
    }
    Ok(())
}

/// `raven audit replay <id>` — Replay audit entries for a task.
async fn cmd_audit(action: AuditAction) -> anyhow::Result<()> {
    match action {
        AuditAction::Replay { task_id } => {
            let config = load_config(None)?;
            let audit_path = configured_audit_path(&config);
            let path = audit_path.as_path();

            if !path.exists() {
                println!("No audit log found at {}", audit_path.display());
                return Ok(());
            }

            let contents = read_redacted_audit(path)?;

            let target_id = uuid::Uuid::parse_str(&task_id)
                .map_err(|e| anyhow::anyhow!("Invalid task ID '{task_id}': {e}"))?;

            let mut entries = Vec::new();
            for line in contents.lines() {
                if let Ok(entry) = serde_json::from_str::<odin_core::types::AuditEntry>(line)
                    && (entry.id == target_id || entry.session_id == target_id)
                {
                    entries.push(entry);
                }
            }

            if entries.is_empty() {
                println!("No audit entries found for task '{task_id}'.");
                return Ok(());
            }

            println!("Audit replay for task '{task_id}':");
            println!();

            for entry in &entries {
                let icon = match entry.result {
                    odin_core::types::AuditResult::Success => "✓",
                    odin_core::types::AuditResult::Failure => "✗",
                    odin_core::types::AuditResult::Denied => "−",
                    odin_core::types::AuditResult::Pending => "○",
                };
                println!(
                    "  {icon} [{ts}] {event:<15} | {action}",
                    ts = entry.timestamp.to_rfc3339(),
                    event = format!("{:?}", entry.event_type),
                    action = entry.action,
                );
                println!(
                    "       agent={agent} session={session} id={id} result={result:?}",
                    agent = entry.agent_id,
                    session = entry.session_id,
                    id = entry.id,
                    result = entry.result,
                );
                if !entry.details.is_null() {
                    println!(
                        "       details: {}",
                        serde_json::to_string(&entry.details).unwrap_or_default()
                    );
                }
                println!();
            }

            println!("{} entries replayed.", entries.len());
        }
    }

    Ok(())
}

/// `raven status` — Show runtime status summary.
fn cmd_status() -> anyhow::Result<()> {
    let config = load_config(None)?;

    println!("╔══════════════════════════════════════════╗");
    println!("║       Raven Agent Status                 ║");
    println!("╠══════════════════════════════════════════╣");
    println!("║  Version:    {:30} ║", env!("CARGO_PKG_VERSION"));
    println!("║  Instance:   {:30} ║", config.general.instance_name);
    println!("║  Provider:   {:30} ║", config.models.default_provider);
    println!(
        "║  Debug:      {:30} ║",
        if cfg!(debug_assertions) { "yes" } else { "no" }
    );
    println!("║  Log Level:  {:30} ║", config.general.log_level);
    println!("║  Providers:  {:30} ║", config.models.providers.len());
    println!("║  Skills Dir: {:30} ║", config.agent.skills_dir);
    println!("╚══════════════════════════════════════════╝");

    Ok(())
}

/// Build and register all available tools.
fn build_tool_registry() -> odin_tools::ToolRegistry {
    build_tool_registry_with(Arc::new(odin_tools::Sandbox::default()), None)
}

/// Build the shared built-in registry used by execution and inspection paths.
fn build_tool_registry_with(
    sandbox: Arc<odin_tools::Sandbox>,
    enabled: Option<&[String]>,
) -> odin_tools::ToolRegistry {
    odin_tools::builtin_registry(sandbox, enabled)
        .expect("the static built-in registry must not contain duplicate names")
}

/// Print a single catalog entry with its metadata.
fn print_tool_entry(tool: &odin_tools::CatalogEntry) {
    let icon = if tool.is_dangerous { "⚠" } else { " " };
    println!(
        "  {icon} {name:<30} {desc}",
        name = tool.name,
        desc = tool.description
    );
    println!("     Tags: {}", tool.tags.join(", "));
    if tool.requires_approval {
        println!("     [requires approval]");
    }
    println!();
}

// ── Helpers ──────────────────────────────────────────────────────────

fn configured_data_dir(config: &OdinConfig) -> PathBuf {
    config.general.data_dir.as_ref().map_or_else(
        || PathBuf::from(shellexpand::tilde("~/.raven-agent").to_string()),
        |path| PathBuf::from(shellexpand::tilde(&path.to_string_lossy()).to_string()),
    )
}

fn configured_reliability_path(config: &OdinConfig) -> PathBuf {
    configured_data_dir(config).join("reliability.db")
}

fn build_reliability_tracker(
    config: &OdinConfig,
) -> anyhow::Result<Arc<odin_tools::ReliabilityTracker>> {
    let path = configured_reliability_path(config);
    odin_tools::ReliabilityTracker::persistent(&path, odin_tools::ReliabilityConfig::default())
        .map(Arc::new)
        .map_err(|error| {
            anyhow::anyhow!(
                "failed to open reliability store '{}': {error}",
                path.display()
            )
        })
}

fn build_memory_store(config: &OdinConfig) -> anyhow::Result<odin_memory::SqliteMemoryStore> {
    if !config.memory.enabled {
        return odin_memory::SqliteMemoryStore::in_memory()
            .map_err(|error| anyhow::anyhow!("Failed to initialize in-memory store: {error}"));
    }

    let path = config.memory.db_path.as_ref().map_or_else(
        || configured_data_dir(config).join("memory.db"),
        |path| PathBuf::from(shellexpand::tilde(&path.to_string_lossy()).to_string()),
    );
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            anyhow::anyhow!(
                "Failed to create memory directory '{}': {error}",
                parent.display()
            )
        })?;
    }
    odin_memory::SqliteMemoryStore::new(&path.to_string_lossy())
        .map(|store| store.with_retention_limit(config.memory.max_entries))
        .map_err(|error| {
            anyhow::anyhow!(
                "Failed to open memory database '{}': {error}",
                path.display()
            )
        })
}

fn build_audit_logger(config: &OdinConfig) -> anyhow::Result<odin_audit::AuditLoggerImpl> {
    let file_path = configured_audit_path(config);
    odin_audit::AuditLoggerImpl::try_new(odin_audit::AuditLoggerConfig {
        enabled: config.audit.enabled,
        file_path: config.audit.enabled.then_some(file_path),
        json_format: true,
        buffer_size: 1,
        history_size: 1_000,
        mask_secrets: true,
    })
    .map_err(|error| anyhow::anyhow!("Failed to open audit sink: {error}"))
}

fn configured_audit_path(config: &OdinConfig) -> PathBuf {
    config.audit.log_path.as_ref().map_or_else(
        || configured_data_dir(config).join("audit.jsonl"),
        |path| PathBuf::from(shellexpand::tilde(&path.to_string_lossy()).to_string()),
    )
}

fn read_redacted_audit(path: &std::path::Path) -> anyhow::Result<String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|error| anyhow::anyhow!("Failed to read audit log: {error}"))?;
    Ok(odin_permissions::SecretRedactor::full().redact(&contents))
}

/// Load configuration from an optional path.
fn load_config(path: Option<&std::path::Path>) -> anyhow::Result<OdinConfig> {
    let config = match path {
        Some(p) if p.exists() => OdinConfig::load(p).map_err(|e| {
            anyhow::anyhow!("Failed to load Raven Agent config '{}': {e}", p.display())
        }),
        Some(p) => anyhow::bail!(
            "Config file '{}' does not exist. Create it with 'raven config' or pass an existing path.",
            p.display()
        ),
        None => {
            // RAVEN_CONFIG is canonical. ODIN_CONFIG and old filenames remain
            // read-only compatibility fallbacks.
            let env_path = std::env::var_os("RAVEN_CONFIG")
                .or_else(|| std::env::var_os("ODIN_CONFIG"))
                .map(PathBuf::from);
            if let Some(path) = env_path {
                return load_config(Some(&path));
            }

            let default_paths = [
                shellexpand::tilde("~/.config/raven/config.yaml").to_string(),
                shellexpand::tilde("~/.raven-agent/config.yaml").to_string(),
                shellexpand::tilde("~/.odin/config.yaml").to_string(),
                "raven.yaml".to_string(),
                "raven.yml".to_string(),
                "odin.yaml".to_string(),
                "odin.yml".to_string(),
            ];

            if let Some(path) = default_paths
                .iter()
                .map(std::path::Path::new)
                .find(|path| path.exists())
            {
                OdinConfig::load(path).map_err(|e| anyhow::anyhow!("Config load error: {e}"))
            } else {
                Ok(OdinConfig::default())
            }
        }
    }?;
    config
        .validate()
        .map_err(|error| anyhow::anyhow!("Invalid Raven Agent configuration: {error}"))?;
    Ok(config)
}

/// Resolve the canonical skills directory with a legacy read fallback.
fn default_skills_dir() -> PathBuf {
    let canonical = PathBuf::from(shellexpand::tilde("~/.config/raven/skills").to_string());
    let legacy = PathBuf::from(shellexpand::tilde("~/.odin/skills").to_string());
    if !canonical.exists() && legacy.exists() {
        legacy
    } else {
        canonical
    }
}

/// Get the Rust compiler version.
fn rustc_version() -> String {
    option_env!("CARGO_PKG_RUST_VERSION")
        .unwrap_or("stable")
        .to_string()
}

/// Get the build profile name.
fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::AgentTaskSet;
    use super::Cli;
    use super::Commands;
    use super::cleanup_failed_orchestration;
    use super::load_config;
    use super::recover_failed_orchestration;
    use super::subagent_retry_is_idempotent;
    use clap::Parser;

    #[test]
    fn orchestration_retry_policy_only_allows_read_only_tools() {
        assert!(subagent_retry_is_idempotent(&[
            "file_read".into(),
            "web_search".into()
        ]));
        assert!(!subagent_retry_is_idempotent(&["shell".into()]));
        assert!(!subagent_retry_is_idempotent(&["file_write".into()]));
    }

    #[test]
    fn invalid_loaded_config_is_rejected_before_startup() {
        let path = std::env::temp_dir().join(format!(
            "raven-config-validation-{}.yaml",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&path, "budgets:\n  max_tool_calls_per_turn: 0\n").unwrap();

        let result = load_config(Some(&path));
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_agent_task_set_yields_first_completed_agent() {
        let mut tasks = AgentTaskSet::new();
        let slow_id = uuid::Uuid::new_v4();
        let fast_id = uuid::Uuid::new_v4();

        tasks.spawn(slow_id, async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            (
                slow_id,
                "slow".to_string(),
                Err(odin_core::error::OdinError::Internal("slow".into())),
                std::time::Duration::from_millis(50),
            )
        });
        tasks.spawn(fast_id, async move {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            (
                fast_id,
                "fast".to_string(),
                Err(odin_core::error::OdinError::Internal("fast".into())),
                std::time::Duration::from_millis(1),
            )
        });

        let (completed_id, result) = tasks.join_next().await.unwrap();
        assert_eq!(completed_id, fast_id);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_agent_task_set_abort_all_drops_inflight_future() {
        struct DropFlag(std::sync::Arc<std::sync::atomic::AtomicBool>);

        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }

        let mut tasks = AgentTaskSet::new();
        let agent_id = uuid::Uuid::new_v4();
        let dropped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let drop_flag = dropped.clone();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();

        tasks.spawn(agent_id, async move {
            let _drop_flag = DropFlag(drop_flag);
            let _ = started_tx.send(());
            std::future::pending::<()>().await;
            unreachable!("aborted task must never complete normally")
        });
        started_rx.await.unwrap();

        tasks.abort_all_and_drain().await;

        assert!(tasks.is_empty());
        assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_cleanup_failed_orchestration_terminalizes_agents_and_clears_locks() {
        let mut composer = odin_orchestrator::Composer::default();
        let first_id = composer.register_agent(odin_orchestrator::sub_agent::SubAgentConfig {
            name: "first".into(),
            goal: "write first".into(),
            read_files: vec![],
            write_files: vec!["shared.rs".into()],
            allowed_tools: vec![],
            required_capabilities: vec![],
            max_iterations: 1,
            priority: 0,
            task_node_id: None,
            injected_context: None,
        });
        let second_id = composer.register_agent(odin_orchestrator::sub_agent::SubAgentConfig {
            name: "second".into(),
            goal: "write second".into(),
            read_files: vec![],
            write_files: vec!["shared.rs".into()],
            allowed_tools: vec![],
            required_capabilities: vec![],
            max_iterations: 1,
            priority: 0,
            task_node_id: None,
            injected_context: None,
        });
        composer.start_agent(first_id).unwrap();
        assert!(composer.start_agent(second_id).is_err());

        let before = composer.file_locks().snapshot();
        assert!(!before.held_locks.is_empty());
        assert!(!before.write_queues.is_empty());

        let mut tasks = AgentTaskSet::new();
        tasks.spawn(first_id, async move {
            std::future::pending::<()>().await;
            unreachable!("cleanup must abort the running agent")
        });

        cleanup_failed_orchestration(
            &mut tasks,
            &mut composer,
            &[first_id, second_id],
            "persistence failure",
        )
        .await;

        assert!(tasks.is_empty());
        assert_eq!(
            composer.get_agent(&first_id).unwrap().1.phase,
            odin_orchestrator::lifecycle::AgentPhase::Failed
        );
        assert_eq!(
            composer.get_agent(&second_id).unwrap().1.phase,
            odin_orchestrator::lifecycle::AgentPhase::Failed
        );
        let after = composer.file_locks().snapshot();
        assert!(after.held_locks.is_empty());
        assert!(after.write_queues.is_empty());
    }

    #[tokio::test]
    async fn test_recover_failed_orchestration_persists_terminal_state() {
        use odin_orchestrator::persistence::OrchestrationStore;

        let root_goal = "write persisted cleanup";
        let mut composer = odin_orchestrator::Composer::default();
        let node_id = {
            let graph = composer.intake(root_goal);
            assert_eq!(graph.nodes.len(), 1);
            *graph.nodes.keys().next().unwrap()
        };
        let agent_id = composer.register_agent(odin_orchestrator::sub_agent::SubAgentConfig {
            name: "persisted".into(),
            goal: root_goal.into(),
            read_files: vec![],
            write_files: vec!["persisted.rs".into()],
            allowed_tools: vec![],
            required_capabilities: vec![],
            max_iterations: 1,
            priority: 0,
            task_node_id: Some(node_id),
            injected_context: None,
        });
        composer.start_agent(agent_id).unwrap();

        let store = odin_orchestrator::persistence::SqliteOrchestrationStore::new_in_memory()
            .await
            .unwrap();
        store.initialize().await.unwrap();
        store
            .save_task_graph(composer.get_graph(root_goal).unwrap())
            .await
            .unwrap();
        let graph_root_id = composer.get_graph(root_goal).unwrap().id.to_string();
        let (_, initial_lifecycle) = composer.get_agent(&agent_id).unwrap();
        store
            .save_agent_lifecycle(&graph_root_id, initial_lifecycle)
            .await
            .unwrap();
        store
            .save_lock_snapshot(
                &graph_root_id,
                &serde_json::to_string(&composer.file_locks().snapshot()).unwrap(),
            )
            .await
            .unwrap();

        let mut tasks = AgentTaskSet::new();
        tasks.spawn(agent_id, async move {
            std::future::pending::<()>().await;
            unreachable!("recovery must abort the running agent")
        });

        let recovered = recover_failed_orchestration(
            &mut tasks,
            &mut composer,
            &store,
            root_goal,
            &[agent_id],
            anyhow::anyhow!("persistence failure"),
        )
        .await;

        assert!(recovered.to_string().contains("persistence failure"));
        let lifecycle = store.load_agent_lifecycle(agent_id).await.unwrap();
        assert_eq!(
            lifecycle.phase,
            odin_orchestrator::lifecycle::AgentPhase::Failed
        );
        let graph = store.load_task_graph(root_goal).await.unwrap();
        assert_eq!(
            graph.status,
            odin_orchestrator::task_graph::TaskGraphStatus::Failed
        );
        let snapshot: odin_orchestrator::file_lock::LockSnapshot = serde_json::from_str(
            &store
                .load_lock_snapshot(&graph_root_id)
                .await
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        assert!(snapshot.held_locks.is_empty());
        assert!(snapshot.write_queues.is_empty());
    }

    #[test]
    fn test_cli_parses_run_command() {
        let _cli = Cli::parse_from(["raven", "run", "write a test"]);
    }

    #[test]
    fn test_cli_parses_serve_command() {
        let _cli = Cli::parse_from(["raven", "serve"]);
    }

    #[test]
    fn test_cli_parses_serve_with_addr() {
        let _cli = Cli::parse_from(["raven", "serve", "--addr", "0.0.0.0:8080"]);
    }

    #[test]
    fn test_cli_parses_config() {
        let _cli = Cli::parse_from(["raven", "config", "/tmp/test.yaml"]);
    }

    #[test]
    fn test_cli_parses_version() {
        let _cli = Cli::parse_from(["raven", "version"]);
    }

    #[test]
    fn test_verify_cli() {
        let _cmd = <Cli as clap::CommandFactory>::command();
    }

    #[test]
    fn test_cli_parses_schedule_add() {
        let cli = Cli::parse_from([
            "raven",
            "schedule",
            "add",
            "my-job",
            "0 */6 * * *",
            "run tests",
        ]);
        assert!(matches!(cli.command, Some(Commands::Schedule { .. })));
    }

    #[test]
    fn test_cli_parses_schedule_list() {
        let cli = Cli::parse_from(["raven", "schedule", "list"]);
        assert!(matches!(cli.command, Some(Commands::Schedule { .. })));
    }

    #[test]
    fn test_cli_parses_schedule_host_and_health_surfaces() {
        for args in [
            vec!["raven", "schedule", "host"],
            vec!["raven", "schedule", "status"],
            vec!["raven", "schedule", "history", "--limit", "5"],
        ] {
            let cli = Cli::parse_from(args);
            assert!(matches!(cli.command, Some(Commands::Schedule { .. })));
        }
    }

    #[test]
    fn test_cli_parses_schedule_remove() {
        let cli = Cli::parse_from([
            "raven",
            "schedule",
            "remove",
            "550e8400-e29b-41d4-a716-446655440000",
        ]);
        assert!(matches!(cli.command, Some(Commands::Schedule { .. })));
    }

    #[test]
    fn test_cli_parses_schedule_enable() {
        let cli = Cli::parse_from([
            "raven",
            "schedule",
            "enable",
            "550e8400-e29b-41d4-a716-446655440000",
        ]);
        assert!(matches!(cli.command, Some(Commands::Schedule { .. })));
    }

    #[test]
    fn test_cli_parses_schedule_disable() {
        let cli = Cli::parse_from([
            "raven",
            "schedule",
            "disable",
            "550e8400-e29b-41d4-a716-446655440000",
        ]);
        assert!(matches!(cli.command, Some(Commands::Schedule { .. })));
    }

    #[test]
    fn test_cli_parses_providers_list() {
        let cli = Cli::parse_from(["raven", "providers", "list"]);
        assert!(matches!(cli.command, Some(Commands::Providers { .. })));
    }

    #[test]
    fn test_cli_parses_eval_mocked() {
        let cli = Cli::parse_from(["raven", "eval", "mocked", "--format", "json"]);
        assert!(matches!(cli.command, Some(Commands::Eval { .. })));
    }

    #[test]
    fn test_cli_parses_eval_profiles() {
        let cli = Cli::parse_from(["raven", "eval", "profiles"]);
        assert!(matches!(cli.command, Some(Commands::Eval { .. })));
    }

    #[test]
    fn test_cli_parses_eval_live() {
        let cli = Cli::parse_from([
            "raven",
            "eval",
            "live",
            "--provider",
            "ollama",
            "--model",
            "qwen2.5-coder:7b",
            "--base-url",
            "http://localhost:11434/v1",
        ]);
        assert!(matches!(cli.command, Some(Commands::Eval { .. })));
    }

    #[test]
    fn test_cli_parses_skills_list() {
        let cli = Cli::parse_from(["raven", "skills", "list"]);
        assert!(matches!(cli.command, Some(Commands::Skills { .. })));
    }

    #[test]
    fn test_cli_parses_tasks_list() {
        let cli = Cli::parse_from(["raven", "tasks", "list"]);
        assert!(matches!(cli.command, Some(Commands::Tasks { .. })));
    }

    #[test]
    fn test_cli_parses_tasks_inspect() {
        let cli = Cli::parse_from([
            "raven",
            "tasks",
            "inspect",
            "550e8400-e29b-41d4-a716-446655440000",
        ]);
        assert!(matches!(cli.command, Some(Commands::Tasks { .. })));
    }

    #[test]
    fn test_cli_parses_sessions_list() {
        let cli = Cli::parse_from(["raven", "sessions", "list"]);
        assert!(matches!(cli.command, Some(Commands::Sessions { .. })));
    }

    #[test]
    fn test_cli_parses_sessions_inspect() {
        let cli = Cli::parse_from([
            "raven",
            "sessions",
            "inspect",
            "550e8400-e29b-41d4-a716-446655440000",
        ]);
        assert!(matches!(cli.command, Some(Commands::Sessions { .. })));
    }

    #[test]
    fn test_cli_parses_tools_list() {
        let cli = Cli::parse_from(["raven", "tools", "list"]);
        assert!(matches!(cli.command, Some(Commands::Tools { .. })));
    }

    #[test]
    fn test_cli_parses_tools_inspect() {
        let cli = Cli::parse_from(["raven", "tools", "inspect", "file_read"]);
        assert!(matches!(cli.command, Some(Commands::Tools { .. })));
    }

    #[test]
    fn test_cli_parses_tools_validate() {
        let cli = Cli::parse_from(["raven", "tools", "validate"]);
        assert!(matches!(cli.command, Some(Commands::Tools { .. })));
    }

    #[test]
    fn test_cli_parses_tools_test() {
        let cli = Cli::parse_from(["raven", "tools", "test", "file_read"]);
        assert!(matches!(cli.command, Some(Commands::Tools { .. })));
    }

    #[test]
    fn test_cli_parses_tools_doctor() {
        let cli = Cli::parse_from(["raven", "tools", "doctor"]);
        assert!(matches!(cli.command, Some(Commands::Tools { .. })));
    }

    #[test]
    fn test_cli_parses_tools_catalog() {
        let cli = Cli::parse_from(["raven", "tools", "catalog"]);
        assert!(matches!(cli.command, Some(Commands::Tools { .. })));
    }

    #[test]
    fn test_cli_parses_audit_replay() {
        let cli = Cli::parse_from([
            "raven",
            "audit",
            "replay",
            "550e8400-e29b-41d4-a716-446655440000",
        ]);
        assert!(matches!(cli.command, Some(Commands::Audit { .. })));
    }

    #[test]
    fn test_cli_parses_status() {
        let cli = Cli::parse_from(["raven", "status"]);
        assert!(matches!(cli.command, Some(Commands::Status)));
    }
}
