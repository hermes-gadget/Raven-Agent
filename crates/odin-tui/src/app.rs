//! Application state for Raven TUI.
//! Clean unified state model for the redesigned layout.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::events::Action;
use crate::runner::{self, AgentStage, RunnerCommand, RunnerEvent};

// ── Panels (numbered tabs) ───────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Panel {
    Chat = 1,
    Agents = 2,
    TaskGraph = 3,
    Locks = 4,
    Tools = 5,
    Logs = 6,
    History = 7,
    Conflicts = 8,
}

impl Panel {
    pub const ALL: &[Panel] = &[
        Panel::Chat,
        Panel::Agents,
        Panel::TaskGraph,
        Panel::Locks,
        Panel::Tools,
        Panel::Logs,
        Panel::History,
        Panel::Conflicts,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            Panel::Chat => "Chat",
            Panel::Agents => "Agents",
            Panel::TaskGraph => "Task Graph",
            Panel::Locks => "Files/Locks",
            Panel::Tools => "Tools",
            Panel::Logs => "Logs/Audit",
            Panel::History => "History",
            Panel::Conflicts => "Conflicts",
        }
    }

    pub fn number(&self) -> usize {
        *self as usize
    }

    pub fn next(&self) -> Panel {
        let idx = Panel::ALL.iter().position(|p| p == self).unwrap_or(0);
        Panel::ALL[(idx + 1) % Panel::ALL.len()]
    }

    pub fn prev(&self) -> Panel {
        let idx = Panel::ALL.iter().position(|p| p == self).unwrap_or(0);
        let len = Panel::ALL.len();
        Panel::ALL[(idx + len - 1) % len]
    }
}

// ── Messages ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    pub timestamp: Instant,
    pub run_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageRole {
    User,
    Agent,
    System,
}

// ── Orchestration Snapshot ───────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct OrchSnapshot {
    pub active_runs: Vec<RunInfo>,
    pub agents: Vec<AgentInfo>,
    pub locks: Vec<LockInfo>,
    pub write_queue: Vec<QueueInfo>,
    pub conflicts: Vec<ConflictInfo>,
    pub tool_calls: Vec<ToolCallInfo>,
    pub task_graph_nodes: Vec<GraphNode>,
    pub task_graph_edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone)]
pub struct RunInfo {
    pub run_id: String,
    pub goal: String,
    pub status: String,
    pub task_count: usize,
    pub done_count: usize,
    pub failed_count: usize,
}

#[derive(Debug, Clone)]
pub struct AgentInfo {
    pub agent_id: String,
    pub label: String,
    pub goal: String,
    pub phase: String,
    pub current_task: String,
    pub current_tool_call: Option<String>,
    pub read_files: Vec<String>,
    pub write_files: Vec<String>,
    pub held_locks: Vec<String>,
    pub queued_write_locks: Vec<String>,
    pub retry_count: u32,
    pub tool_calls: u64,
    pub progress_pct: u32,
    pub locked_file: Option<String>,
    pub last_update: String,
    pub last_error: Option<String>,
    pub stage: String,
    pub heartbeat: String,
    pub elapsed_secs: u64,
    pub last_event_age_secs: Option<u64>,
    pub blocked_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LockInfo {
    pub file_path: String,
    pub lock_type: String,
    pub holder: String,
    pub queued_waiters: usize,
}

#[derive(Debug, Clone)]
pub struct QueueInfo {
    pub file_path: String,
    pub position: usize,
    pub requester: String,
}

#[derive(Debug, Clone)]
pub struct ConflictInfo {
    pub file_path: String,
    pub agent_a: String,
    pub agent_b: String,
}

#[derive(Debug, Clone)]
pub struct ToolCallInfo {
    pub tool_name: String,
    pub agent_id: String,
    pub duration_ms: u64,
    pub success: bool,
    pub summary: String,
}

#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub goal: String,
    pub status: String,
    pub agent_id: Option<String>,
    pub children: Vec<String>,
    pub depth: usize,
}

#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub from: String,
    pub to: String,
    pub label: String,
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: Instant,
    pub level: LogLevel,
    pub source: String,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn name(&self) -> &'static str {
        match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolDisplay {
    pub name: String,
    pub description: String,
    pub category: String,
    pub tags: Vec<String>,
    pub is_dangerous: bool,
    pub reliability: Option<odin_tools::ToolReliability>,
}

#[derive(Debug, Clone)]
pub struct SkillDisplay {
    pub name: String,
    pub description: String,
    pub required_tools: Vec<String>,
    pub recommended_tools: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ProviderDisplay {
    pub name: String,
    pub provider_type: String,
    pub model: String,
    pub healthy: bool,
    pub fallback_order: usize,
}

#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    pub title: String,
    pub details: String,
    pub action: PendingDangerAction,
}

#[derive(Debug, Clone)]
pub enum PendingDangerAction {
    CancelActiveRun,
}

#[derive(Debug, Clone)]
pub struct LiveAgentStatus {
    pub agent_id: String,
    pub label: String,
    pub stage: String,
    pub detail: String,
    pub current_call: Option<String>,
    pub elapsed_ms: u64,
    pub first_seen_at: Instant,
    pub last_event_at: Instant,
    pub model_wait_notice_sent: bool,
}

// ── App ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunMode {
    Idle,
    Running,
    Paused,
}

pub struct App {
    pub db_path: PathBuf,
    pub messages: Vec<ChatMessage>,
    pub input: String,
    pub cursor: usize,
    pub focused_panel: Panel,
    pub side_scroll: usize,
    pub chat_scroll: usize,
    pub show_help: bool,
    pub should_quit: bool,
    pub orch: OrchSnapshot,
    pub running_runs: Vec<String>,
    pub mode: RunMode,
    pub last_tick: Instant,
    pub log_entries: Vec<LogEntry>,
    pub tool_displays: Vec<ToolDisplay>,
    pub skill_displays: Vec<SkillDisplay>,
    pub provider_displays: Vec<ProviderDisplay>,
    pub search_query: String,
    pub is_searching: bool,
    pub provider_name: String,
    pub model_name: String,
    pub elapsed: std::time::Duration,
    pub tool_call_count: u64,
    pub error_count: u64,
    pub last_action: String,
    pub runner_tx: Option<tokio::sync::mpsc::UnboundedSender<RunnerCommand>>,
    pub runner_rx: Option<tokio::sync::mpsc::UnboundedReceiver<RunnerEvent>>,
    pub active_run_id: Option<String>,
    pub pending_approval: Option<ApprovalRequest>,
    pub max_iterations: u32,
    pub last_runner_event_at: Option<Instant>,
    pub last_runner_stage: String,
    pub stale_warning_emitted: bool,
    pub offline_run_notice_emitted: bool,
    pub live_agents: HashMap<String, LiveAgentStatus>,
}

impl App {
    pub async fn new(db_path: PathBuf) -> Result<Self> {
        let mut app = Self {
            db_path,
            messages: Vec::new(),
            input: String::new(),
            cursor: 0,
            focused_panel: Panel::Chat,
            side_scroll: 0,
            chat_scroll: 0,
            show_help: false,
            should_quit: false,
            orch: OrchSnapshot::default(),
            running_runs: Vec::new(),
            mode: RunMode::Idle,
            last_tick: Instant::now(),
            log_entries: Vec::new(),
            tool_displays: Vec::new(),
            skill_displays: Vec::new(),
            provider_displays: Vec::new(),
            search_query: String::new(),
            is_searching: false,
            provider_name: String::new(),
            model_name: String::new(),
            elapsed: std::time::Duration::ZERO,
            tool_call_count: 0,
            error_count: 0,
            last_action: String::new(),
            runner_tx: None,
            runner_rx: None,
            active_run_id: None,
            pending_approval: None,
            max_iterations: 100,
            last_runner_event_at: None,
            last_runner_stage: String::new(),
            stale_warning_emitted: false,
            offline_run_notice_emitted: false,
            live_agents: HashMap::new(),
        };

        let registry = odin_tools::builtin_registry(
            std::sync::Arc::new(odin_tools::Sandbox::default()),
            None,
        )?;
        let display_config = load_display_config().ok();
        let reliability_tracker = display_config.as_ref().and_then(|config| {
            let path = configured_data_dir(config).join("reliability.db");
            odin_tools::ReliabilityTracker::persistent(
                path,
                odin_tools::ReliabilityConfig::default(),
            )
            .ok()
        });
        let catalog = odin_tools::ToolCatalog::from_registry(&registry);
        let mut tools: Vec<ToolDisplay> = catalog
            .by_name
            .into_values()
            .map(|tool| {
                let reliability = reliability_tracker.as_ref().and_then(|tracker| {
                    let info = tracker.get(&tool.name);
                    (info.total_calls > 0).then_some(info)
                });
                ToolDisplay {
                    name: tool.name,
                    description: tool.description,
                    category: tool.category,
                    tags: tool.tags,
                    is_dangerous: tool.is_dangerous,
                    reliability,
                }
            })
            .collect();
        if let Some(tracker) = &reliability_tracker {
            for reliability in tracker.all() {
                if tools.iter().any(|tool| tool.name == reliability.tool_name) {
                    continue;
                }
                tools.push(ToolDisplay {
                    name: reliability.tool_name.clone(),
                    description: "Observed external tool".into(),
                    category: "observed".into(),
                    tags: vec!["observed".into()],
                    is_dangerous: false,
                    reliability: Some(reliability),
                });
            }
        }
        tools.sort_by(|a, b| a.name.cmp(&b.name));
        app.tool_displays = tools;

        if let Some(config) = display_config {
            app.provider_name = config.models.default_provider.clone();
            app.model_name = config
                .models
                .default_model
                .clone()
                .or_else(|| {
                    config
                        .models
                        .providers
                        .get(&config.models.default_provider)
                        .and_then(|p| p.default_model.clone())
                })
                .unwrap_or_default();
        }

        app.refresh_orchestration().await.ok();
        app.add_message(
            MessageRole::System,
            "Raven Agent - type a goal to start an orchestration run, or press ? for help.",
        );
        Ok(app)
    }

    pub fn add_message(&mut self, role: MessageRole, content: impl Into<String>) {
        self.messages.push(ChatMessage {
            role,
            content: content.into(),
            timestamp: Instant::now(),
            run_id: None,
        });
    }

    pub fn add_log(&mut self, level: LogLevel, source: &str, message: impl Into<String>) {
        let message = odin_permissions::SecretRedactor::full().redact(&message.into());
        self.log_entries.push(LogEntry {
            timestamp: Instant::now(),
            level,
            source: source.to_string(),
            message,
        });
        while self.log_entries.len() > 500 {
            self.log_entries.remove(0);
        }
    }

    pub fn dispatch(&mut self, action: Action) {
        match action {
            Action::Submit => {
                // Submission performs async persistence in `submit_goal`.
            }
            Action::NextPanel => {
                self.focused_panel = self.focused_panel.next();
                self.side_scroll = 0;
            }
            Action::PrevPanel => {
                self.focused_panel = self.focused_panel.prev();
                self.side_scroll = 0;
            }
            Action::FocusPanel(panel) => {
                self.focused_panel = panel;
                self.side_scroll = 0;
            }
            Action::ScrollUp => {
                if self.focused_panel == Panel::Chat {
                    self.chat_scroll = self.chat_scroll.saturating_sub(1);
                } else {
                    self.side_scroll = self.side_scroll.saturating_sub(1);
                }
            }
            Action::ScrollDown => {
                if self.focused_panel == Panel::Chat {
                    self.chat_scroll = self.chat_scroll.saturating_add(1);
                } else {
                    self.side_scroll = self.side_scroll.saturating_add(1);
                }
            }
            Action::ToggleHelp => {
                self.show_help = !self.show_help;
            }
            Action::ToggleSearch => {
                self.is_searching = !self.is_searching;
                if !self.is_searching {
                    self.search_query.clear();
                }
            }
            Action::CancelSearch => {
                if self.is_searching {
                    self.is_searching = false;
                    self.search_query.clear();
                } else if self.show_help {
                    self.show_help = false;
                } else {
                    self.should_quit = true;
                }
            }
            Action::Quit => {
                self.should_quit = true;
            }
            Action::Tick => {
                if self.mode == RunMode::Running {
                    self.elapsed += Duration::from_millis(500);
                }
                self.last_tick = Instant::now();
                self.check_stale_runner_event();
                self.apply_live_agent_overlays();
            }
            Action::InsertChar(c) => {
                if self.is_searching {
                    if self.search_query.len() < 256 {
                        self.search_query.push(c);
                    }
                } else if self.input.len() < 4096 {
                    self.input.insert(self.cursor, c);
                    self.cursor += c.len_utf8();
                }
            }
            Action::InsertNewline => {
                if !self.is_searching && self.input.len() < 4096 {
                    self.input.insert(self.cursor, '\n');
                    self.cursor += 1;
                }
            }
            Action::DeletePrev => {
                if self.is_searching {
                    self.search_query.pop();
                } else if self.cursor > 0 {
                    let previous = self.input[..self.cursor]
                        .char_indices()
                        .next_back()
                        .map_or(0, |(index, _)| index);
                    self.input.drain(previous..self.cursor);
                    self.cursor = previous;
                }
            }
            Action::DeleteNext => {
                if !self.is_searching && self.cursor < self.input.len() {
                    let len = self.input[self.cursor..]
                        .chars()
                        .next()
                        .map_or(0, char::len_utf8);
                    self.input.drain(self.cursor..self.cursor + len);
                }
            }
            Action::MoveCursorLeft => {
                if !self.is_searching {
                    self.cursor = self.input[..self.cursor]
                        .char_indices()
                        .next_back()
                        .map_or(0, |(index, _)| index);
                }
            }
            Action::MoveCursorRight => {
                if !self.is_searching && self.cursor < self.input.len() {
                    self.cursor += self.input[self.cursor..]
                        .chars()
                        .next()
                        .map_or(0, char::len_utf8);
                }
            }
            Action::MoveCursorHome => {
                self.cursor = 0;
            }
            Action::MoveCursorEnd => {
                self.cursor = self.input.len();
            }
            Action::RefreshOrch => {}
        }
    }

    pub async fn handle_action(&mut self, action: Action) -> Result<()> {
        match action {
            Action::Submit => self.submit_goal().await,
            Action::InsertChar('y' | 'Y') if self.pending_approval.is_some() => {
                self.approve_pending().await
            }
            Action::InsertChar('n' | 'N') if self.pending_approval.is_some() => {
                self.deny_pending();
                Ok(())
            }
            other => {
                self.dispatch(other);
                Ok(())
            }
        }
    }

    /// Submit the current chat input.
    ///
    /// If no run is active, this starts a real background orchestration run.
    /// If a run is active, the message is sent as steering input to that run
    /// instead of creating a disconnected plan.
    pub async fn submit_goal(&mut self) -> Result<()> {
        let goal = std::mem::take(&mut self.input);
        self.cursor = 0;
        if goal.trim().is_empty() {
            return Ok(());
        }
        let goal = goal.trim().to_string();
        self.add_message(MessageRole::User, &goal);

        if self.handle_control_input(&goal).await? {
            return Ok(());
        }

        if self.has_active_run() {
            self.send_runner_command(RunnerCommand::Redirect(goal.clone()))?;
            self.add_message(
                MessageRole::Agent,
                "Steered the active run with your latest message.",
            );
            self.last_action = "redirected active run".into();
            return Ok(());
        }

        self.mode = RunMode::Running;
        self.elapsed = Duration::ZERO;
        self.last_runner_event_at = Some(Instant::now());
        self.last_runner_stage = "creating run".into();
        self.stale_warning_emitted = false;
        self.live_agents.clear();
        self.last_action = "creating run".into();
        self.add_message(
            MessageRole::Agent,
            "Creating run, decomposing the goal, and loading execution resources...",
        );

        let handle =
            runner::spawn_run(self.db_path.clone(), goal.clone(), self.max_iterations).await?;
        let run_id = handle.run_id.clone();
        self.active_run_id = Some(run_id.clone());
        self.running_runs = vec![run_id.clone()];
        self.runner_tx = Some(handle.command_tx);
        self.runner_rx = Some(handle.event_rx);
        self.last_action = format!("started run {}", short_id(&run_id));
        self.add_message(
            MessageRole::Agent,
            format!(
                "Started run {}. Use /pause, /resume, /cancel, /redirect, or /prio.",
                run_id
            ),
        );
        self.add_log(
            LogLevel::Info,
            "orchestration",
            format!("Started run {run_id}: {goal}"),
        );
        self.drain_runner_events();
        self.refresh_orchestration().await?;
        Ok(())
    }

    async fn handle_control_input(&mut self, input: &str) -> Result<bool> {
        let trimmed = input.trim();
        let Some(command) = trimmed.strip_prefix('/') else {
            return Ok(false);
        };

        let mut parts = command.splitn(2, char::is_whitespace);
        let name = parts.next().unwrap_or_default().to_ascii_lowercase();
        let rest = parts.next().unwrap_or_default().trim();

        match name.as_str() {
            "pause" => {
                self.send_runner_command(RunnerCommand::Pause)?;
                self.mode = RunMode::Paused;
                self.add_message(
                    MessageRole::System,
                    "Pause requested: no new agents will start. In-flight model/tool calls may still finish.",
                );
                self.last_action = "pause requested".into();
            }
            "resume" => {
                self.send_runner_command(RunnerCommand::Resume)?;
                self.mode = RunMode::Running;
                self.add_message(MessageRole::System, "Resume requested for active run.");
                self.last_action = "resume requested".into();
            }
            "cancel" => {
                let run = self
                    .active_run_id
                    .as_deref()
                    .map(short_id)
                    .unwrap_or_else(|| "-".into());
                self.pending_approval = Some(ApprovalRequest {
                    title: "Cancel active run?".into(),
                    details: format!(
                        "Approve cancelling run {run}. Press y to approve or n to deny."
                    ),
                    action: PendingDangerAction::CancelActiveRun,
                });
                self.add_message(
                    MessageRole::System,
                    "Cancel requires approval. Press y to approve or n to deny.",
                );
                self.last_action = "cancel approval pending".into();
            }
            "redirect" => {
                if rest.is_empty() {
                    self.add_message(MessageRole::System, "Usage: /redirect <new instruction>");
                } else {
                    self.send_runner_command(RunnerCommand::Redirect(rest.to_string()))?;
                    self.add_message(MessageRole::Agent, "Redirected active run.");
                    self.last_action = "redirect requested".into();
                }
            }
            "prio" | "priority" | "reprioritise" | "reprioritize" => {
                let args: Vec<&str> = rest.split_whitespace().collect();
                if args.len() != 2 {
                    self.add_message(
                        MessageRole::System,
                        "Usage: /prio <agent-id-prefix> <priority>",
                    );
                } else if let Ok(priority) = args[1].parse::<u32>() {
                    self.send_runner_command(RunnerCommand::Reprioritise {
                        agent_id_prefix: args[0].to_string(),
                        priority,
                    })?;
                    self.add_message(
                        MessageRole::Agent,
                        "Reprioritised matching active agent(s).",
                    );
                    self.last_action = "reprioritise requested".into();
                } else {
                    self.add_message(MessageRole::System, "Priority must be a number.");
                }
            }
            "help" => {
                self.show_help = true;
            }
            _ => {
                self.add_message(
                    MessageRole::System,
                    "Unknown command. Use /pause, /resume, /cancel, /redirect, /prio, or /help.",
                );
            }
        }
        self.refresh_orchestration().await.ok();
        Ok(true)
    }

    fn has_active_run(&self) -> bool {
        self.runner_tx.is_some() && matches!(self.mode, RunMode::Running | RunMode::Paused)
    }

    fn send_runner_command(&mut self, command: RunnerCommand) -> Result<()> {
        let Some(tx) = &self.runner_tx else {
            self.add_message(MessageRole::System, "No active run to control.");
            return Ok(());
        };
        tx.send(command)
            .map_err(|_| anyhow::anyhow!("active run controller is no longer available"))?;
        Ok(())
    }

    pub async fn approve_pending(&mut self) -> Result<()> {
        let Some(request) = self.pending_approval.take() else {
            return Ok(());
        };
        match request.action {
            PendingDangerAction::CancelActiveRun => {
                self.send_runner_command(RunnerCommand::Cancel)?;
                self.add_message(MessageRole::System, "Cancel approved.");
                self.last_action = "cancel approved".into();
            }
        }
        Ok(())
    }

    pub fn deny_pending(&mut self) {
        self.pending_approval = None;
        self.add_message(MessageRole::System, "Request denied.");
        self.last_action = "approval denied".into();
    }

    pub fn drain_runner_events(&mut self) {
        let Some(mut rx) = self.runner_rx.take() else {
            return;
        };
        while let Ok(event) = rx.try_recv() {
            let terminal = matches!(
                &event,
                RunnerEvent::RunFinished { .. }
                    | RunnerEvent::RunCancelled { .. }
                    | RunnerEvent::FatalError { .. }
            );
            self.apply_runner_event(event);
            if terminal {
                return;
            }
        }
        self.runner_rx = Some(rx);
    }

    pub(crate) fn apply_runner_event(&mut self, event: RunnerEvent) {
        self.note_runner_event(&event);
        match event {
            RunnerEvent::RunStage {
                run_id,
                stage,
                detail,
                elapsed_ms,
            } => {
                let stage_label = stage.label();
                self.add_log(
                    LogLevel::Info,
                    "orchestration",
                    format!("Run {} {}: {}", short_id(&run_id), stage_label, detail),
                );
                self.add_message(
                    MessageRole::Agent,
                    format!("{}: {}", stage_label.replace('_', " "), detail),
                );
                self.last_action = format!("{} {}ms", stage_label, elapsed_ms);
            }
            RunnerEvent::RunStarted {
                run_id,
                goal,
                task_count,
            } => {
                self.active_run_id = Some(run_id.clone());
                self.running_runs = vec![run_id.clone()];
                self.mode = RunMode::Running;
                self.add_log(
                    LogLevel::Info,
                    "orchestration",
                    format!("Run {run_id} started with {task_count} task(s)"),
                );
                self.add_message(
                    MessageRole::Agent,
                    format!("Running {task_count} task(s) for: {goal}"),
                );
            }
            RunnerEvent::AgentQueued {
                agent_id,
                label,
                reason,
            } => {
                self.update_live_agent_status(
                    &agent_id,
                    &label,
                    AgentStage::WaitingForLock,
                    &reason,
                    0,
                );
                self.add_log(
                    LogLevel::Info,
                    "locks",
                    format!("{} queued ({}): {}", label, short_id(&agent_id), reason),
                );
                self.add_message(
                    MessageRole::Agent,
                    format!("{} is waiting: {}", label, reason),
                );
            }
            RunnerEvent::AgentStarted {
                agent_id,
                label,
                task,
            } => {
                self.update_live_agent_status(
                    &agent_id,
                    &label,
                    AgentStage::Planning,
                    "agent started; entering loop",
                    0,
                );
                self.add_log(
                    LogLevel::Info,
                    "agent",
                    format!("{} ({}) started: {}", label, short_id(&agent_id), task),
                );
                self.add_message(
                    MessageRole::Agent,
                    format!("{} started: {}", label, truncate_for_message(&task, 96)),
                );
            }
            RunnerEvent::AgentStage {
                agent_id,
                label,
                stage,
                detail,
                elapsed_ms,
            } => {
                let should_announce_wait =
                    self.update_live_agent_status(&agent_id, &label, stage, &detail, elapsed_ms);
                let level = match stage {
                    AgentStage::Failed => LogLevel::Error,
                    AgentStage::WaitingForLock | AgentStage::ApprovalNeeded => LogLevel::Warn,
                    _ => LogLevel::Info,
                };
                self.add_log(
                    level,
                    "agent",
                    format!(
                        "{} ({}) {}: {}",
                        label,
                        short_id(&agent_id),
                        stage.label(),
                        detail
                    ),
                );
                match stage {
                    AgentStage::WaitingForModel if should_announce_wait => {
                        self.add_message(MessageRole::Agent, format!("{}: {}", label, detail));
                    }
                    AgentStage::Failed => {
                        self.error_count += 1;
                        self.add_message(
                            MessageRole::System,
                            format!("{} failed: {}", label, detail),
                        );
                    }
                    AgentStage::Done => {
                        self.add_message(
                            MessageRole::Agent,
                            format!("{} done: {}", label, truncate_for_message(&detail, 120)),
                        );
                    }
                    AgentStage::WaitingForLock | AgentStage::ApprovalNeeded => {
                        self.add_message(
                            MessageRole::System,
                            format!("{} blocked: {}", label, detail),
                        );
                    }
                    _ => {}
                }
            }
            RunnerEvent::AgentFinished {
                agent_id,
                label,
                success,
                summary,
            } => {
                self.add_log(
                    if success {
                        LogLevel::Info
                    } else {
                        LogLevel::Warn
                    },
                    "agent",
                    format!("{} ({}) finished: {}", label, short_id(&agent_id), summary),
                );
                self.update_live_agent_status(
                    &agent_id,
                    &label,
                    if success {
                        AgentStage::Done
                    } else {
                        AgentStage::Failed
                    },
                    &summary,
                    0,
                );
            }
            RunnerEvent::RunPaused { run_id } => {
                self.mode = RunMode::Paused;
                self.add_message(
                    MessageRole::System,
                    format!("Run {} paused.", short_id(&run_id)),
                );
            }
            RunnerEvent::RunResumed { run_id } => {
                self.mode = RunMode::Running;
                self.add_message(
                    MessageRole::System,
                    format!("Run {} resumed.", short_id(&run_id)),
                );
            }
            RunnerEvent::RunRedirected { run_id, message } => {
                self.add_log(
                    LogLevel::Info,
                    "orchestration",
                    format!("Run {} redirected: {}", short_id(&run_id), message),
                );
            }
            RunnerEvent::RunCancelled { run_id } => {
                self.mode = RunMode::Idle;
                self.running_runs.retain(|id| id != &run_id);
                self.runner_tx = None;
                self.runner_rx = None;
                self.active_run_id = None;
                self.add_message(
                    MessageRole::System,
                    format!("Run {} cancelled.", short_id(&run_id)),
                );
            }
            RunnerEvent::RunFinished {
                run_id,
                success,
                summary,
            } => {
                self.mode = RunMode::Idle;
                self.runner_tx = None;
                self.runner_rx = None;
                self.active_run_id = None;
                self.running_runs.retain(|id| id != &run_id);
                self.add_message(
                    if success {
                        MessageRole::Agent
                    } else {
                        MessageRole::System
                    },
                    format!("Final summary for {}:\n{}", short_id(&run_id), summary),
                );
            }
            RunnerEvent::Reprioritised { agent_id, priority } => {
                self.add_log(
                    LogLevel::Info,
                    "orchestration",
                    format!("{} priority -> {}", short_id(&agent_id), priority),
                );
            }
            RunnerEvent::Error { message } => {
                self.error_count += 1;
                self.add_log(LogLevel::Error, "runner", &message);
                self.add_message(MessageRole::System, format!("Runner error: {message}"));
            }
            RunnerEvent::FatalError { message } => {
                let run_id = self
                    .active_run_id
                    .clone()
                    .unwrap_or_else(|| "unknown".into());
                self.on_run_failed(&run_id, &message);
                self.runner_rx = None;
                self.active_run_id = None;
            }
        }
        self.apply_live_agent_overlays();
    }

    pub fn on_run_failed(&mut self, run_id: &str, error: &str) {
        self.running_runs.retain(|r| r != run_id);
        self.runner_tx = None;
        self.error_count += 1;
        let error = odin_permissions::SecretRedactor::full().redact(error);
        self.add_log(LogLevel::Error, "orchestration", &error);
        self.add_message(MessageRole::System, format!("Failed: {error}"));
        if self.running_runs.is_empty() {
            self.mode = RunMode::Idle;
        }
    }

    pub async fn refresh_orchestration(&mut self) -> Result<()> {
        use odin_orchestrator::persistence::{OrchestrationStore, SqliteOrchestrationStore};
        use uuid::Uuid;
        let store = match SqliteOrchestrationStore::new(&self.db_path).await {
            Ok(s) => s,
            Err(_) => return Ok(()),
        };
        store.initialize().await?;

        let mut agent_node_info: std::collections::HashMap<
            String,
            (String, String, Vec<String>, Vec<String>),
        > = std::collections::HashMap::new();

        if let Ok(graphs) = store.list_task_graphs().await {
            self.orch.active_runs.clear();
            for summary in &graphs {
                let graph = store.load_task_graph(&summary.run_id).await.ok();
                let done_count = graph.as_ref().map_or(0, |graph| {
                    graph
                        .nodes
                        .values()
                        .filter(|node| {
                            node.status == odin_orchestrator::task_graph::TaskNodeStatus::Done
                        })
                        .count()
                });
                let failed_count = graph.as_ref().map_or(0, |graph| {
                    graph
                        .nodes
                        .values()
                        .filter(|node| {
                            node.status == odin_orchestrator::task_graph::TaskNodeStatus::Failed
                        })
                        .count()
                });
                self.orch.active_runs.push(RunInfo {
                    run_id: summary.run_id.clone(),
                    goal: summary.root_goal.clone(),
                    status: summary.status.clone(),
                    task_count: summary.node_count as usize,
                    done_count,
                    failed_count,
                });
            }
            let persisted_unfinished_runs: Vec<String> = self
                .orch
                .active_runs
                .iter()
                .filter(|run| matches!(run.status.as_str(), "running" | "paused" | "building"))
                .map(|run| run.run_id.clone())
                .collect();
            let has_live_runner = self.runner_tx.is_some();

            if has_live_runner {
                self.running_runs = self.active_run_id.iter().cloned().collect::<Vec<String>>();
                if !matches!(self.mode, RunMode::Paused) {
                    self.mode = RunMode::Running;
                }
            } else {
                self.running_runs.clear();
                self.active_run_id = None;
                self.mode = RunMode::Idle;
                if !persisted_unfinished_runs.is_empty() && !self.offline_run_notice_emitted {
                    self.offline_run_notice_emitted = true;
                    self.add_log(
                        LogLevel::Warn,
                        "orchestration",
                        format!(
                            "{} persisted unfinished run(s) found without a live TUI runner",
                            persisted_unfinished_runs.len()
                        ),
                    );
                    self.add_message(
                        MessageRole::System,
                        format!(
                            "Found {} persisted unfinished run(s), but no live runner is attached. They are shown for inspection only; submit a goal to start a live run.",
                            persisted_unfinished_runs.len()
                        ),
                    );
                }
            }

            if let Some(run) = self
                .active_run_id
                .as_ref()
                .and_then(|active| {
                    self.orch
                        .active_runs
                        .iter()
                        .find(|run| &run.run_id == active)
                })
                .or_else(|| self.orch.active_runs.first())
                && let Ok(graph) = store.load_task_graph(&run.run_id).await
            {
                self.orch.task_graph_nodes.clear();
                self.orch.task_graph_edges.clear();
                self.orch.conflicts.clear();
                for (nid, node) in &graph.nodes {
                    self.orch.task_graph_nodes.push(GraphNode {
                        id: nid.to_string(),
                        label: node.label.clone(),
                        goal: node.goal.clone(),
                        status: format!("{:?}", node.status).to_lowercase(),
                        agent_id: node.agent_id.map(|id| id.to_string()),
                        children: Vec::new(),
                        depth: 0,
                    });
                    if let Some(agent_id) = node.agent_id {
                        agent_node_info.insert(
                            agent_id.to_string(),
                            (
                                node.label.clone(),
                                node.goal.clone(),
                                node.read_files.clone(),
                                node.write_files.clone(),
                            ),
                        );
                    }
                }
                for edge in &graph.edges {
                    self.orch.task_graph_edges.push(GraphEdge {
                        from: edge.from.to_string(),
                        to: edge.to.to_string(),
                        label: edge.label.clone().unwrap_or_default(),
                    });
                }
                let nodes: Vec<_> = graph.nodes.values().collect();
                for (index, left) in nodes.iter().enumerate() {
                    for right in nodes.iter().skip(index + 1) {
                        for path in left
                            .write_files
                            .iter()
                            .filter(|path| right.write_files.contains(path))
                        {
                            self.orch.conflicts.push(ConflictInfo {
                                file_path: path.clone(),
                                agent_a: left.label.clone(),
                                agent_b: right.label.clone(),
                            });
                        }
                    }
                }
                compute_depths(&mut self.orch.task_graph_nodes, &self.orch.task_graph_edges);
            }
        }

        let mut held_locks_by_agent: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        let mut queued_locks_by_agent: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        let lock_snapshot_id = self
            .active_run_id
            .clone()
            .or_else(|| self.orch.active_runs.first().map(|run| run.run_id.clone()));
        let lock_snapshot = match lock_snapshot_id {
            Some(run_id) => store.load_lock_snapshot(&run_id).await,
            None => store.load_latest_lock_snapshot().await,
        };
        if let Ok(Some(snapshot)) = lock_snapshot
            && let Ok(snapshot) =
                serde_json::from_str::<odin_orchestrator::file_lock::LockSnapshot>(&snapshot)
        {
            self.orch.locks = snapshot
                .held_locks
                .iter()
                .map(|lock| {
                    let queued_waiters = snapshot
                        .write_queues
                        .iter()
                        .find(|queue| queue.path == lock.path)
                        .map_or(0, |queue| queue.queued_agents.len());
                    held_locks_by_agent
                        .entry(lock.agent_id.clone())
                        .or_default()
                        .push(lock.path.clone());
                    LockInfo {
                        file_path: lock.path.clone(),
                        lock_type: lock.mode.clone(),
                        holder: lock.agent_id.clone(),
                        queued_waiters,
                    }
                })
                .collect();
            self.orch.write_queue = snapshot
                .write_queues
                .iter()
                .flat_map(|queue| {
                    queue
                        .queued_agents
                        .iter()
                        .enumerate()
                        .map(|(index, requester)| {
                            queued_locks_by_agent
                                .entry(requester.clone())
                                .or_default()
                                .push(queue.path.clone());
                            QueueInfo {
                                file_path: queue.path.clone(),
                                position: index + 1,
                                requester: requester.clone(),
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
        }

        if let Ok(lifecycles) = store.list_agent_lifecycles().await {
            let active_agent_ids: std::collections::HashSet<String> =
                agent_node_info.keys().cloned().collect();
            let mut agents = Vec::new();
            for summary in lifecycles
                .iter()
                .filter(|lifecycle| active_agent_ids.contains(lifecycle.agent_id.as_str()))
            {
                let lifecycle = if let Ok(id) = Uuid::parse_str(&summary.agent_id) {
                    store.load_agent_lifecycle(id).await.ok()
                } else {
                    None
                };
                let (label, goal, read_files, write_files) = agent_node_info
                    .get(&summary.agent_id)
                    .cloned()
                    .unwrap_or_else(|| {
                        (
                            summary.agent_id.chars().take(8).collect(),
                            String::new(),
                            Vec::new(),
                            Vec::new(),
                        )
                    });
                let phase = lifecycle
                    .as_ref()
                    .map_or_else(|| summary.phase.clone(), |lc| lc.phase.label().to_string());
                let held_locks = held_locks_by_agent
                    .get(&summary.agent_id)
                    .cloned()
                    .unwrap_or_default();
                let queued_write_locks = queued_locks_by_agent
                    .get(&summary.agent_id)
                    .cloned()
                    .unwrap_or_default();
                let last_error = lifecycle.as_ref().and_then(|lc| lc.error.clone());
                let last_update = lifecycle
                    .as_ref()
                    .and_then(|lc| lc.current_reason().map(ToOwned::to_owned))
                    .unwrap_or_else(|| format!("phase: {phase}"));
                let retry_count = lifecycle.as_ref().map_or(0, |lc| lc.retry_count);
                let current_tool_call = (phase == "running").then_some("agent_loop".to_string());
                let locked_file = held_locks.first().cloned();
                let elapsed_secs = lifecycle
                    .as_ref()
                    .map(|lc| lc.elapsed().num_seconds().max(0) as u64)
                    .unwrap_or(0);
                agents.push(AgentInfo {
                    agent_id: summary.agent_id.clone(),
                    label,
                    goal: goal.clone(),
                    phase: phase.clone(),
                    current_task: goal,
                    current_tool_call,
                    read_files,
                    write_files,
                    held_locks,
                    queued_write_locks,
                    retry_count,
                    tool_calls: 0,
                    progress_pct: progress_for_phase(&phase),
                    locked_file,
                    last_update,
                    last_error,
                    stage: phase.clone(),
                    heartbeat: "-".into(),
                    elapsed_secs,
                    last_event_age_secs: None,
                    blocked_reason: None,
                });
            }
            agents.sort_by(|a, b| a.label.cmp(&b.label));
            self.orch.agents = agents;
        }

        self.apply_live_agent_overlays();

        Ok(())
    }

    fn note_runner_event(&mut self, event: &RunnerEvent) {
        self.last_runner_event_at = Some(Instant::now());
        self.stale_warning_emitted = false;
        self.last_runner_stage = match event {
            RunnerEvent::RunStage { stage, detail, .. }
            | RunnerEvent::AgentStage { stage, detail, .. } => {
                format!("{}: {}", stage.label(), detail)
            }
            RunnerEvent::RunStarted { .. } => "run started".into(),
            RunnerEvent::AgentQueued { reason, .. } => format!("agent queued: {reason}"),
            RunnerEvent::AgentStarted { label, .. } => format!("agent {label} started"),
            RunnerEvent::AgentFinished { label, success, .. } => {
                format!("agent {label} finished success={success}")
            }
            RunnerEvent::RunPaused { .. } => "run paused".into(),
            RunnerEvent::RunResumed { .. } => "run resumed".into(),
            RunnerEvent::RunRedirected { .. } => "run redirected".into(),
            RunnerEvent::RunCancelled { .. } => "run cancelled".into(),
            RunnerEvent::RunFinished { success, .. } => format!("run finished success={success}"),
            RunnerEvent::Reprioritised { agent_id, priority } => {
                format!("agent {} priority -> {}", short_id(agent_id), priority)
            }
            RunnerEvent::Error { message } => format!("error: {message}"),
            RunnerEvent::FatalError { message } => format!("fatal_error: {message}"),
        };
        tracing::info!(
            target: "odin_tui::event_trace",
            stage = %self.last_runner_stage,
            event = ?event,
            "TUI received runner event"
        );
    }

    fn update_live_agent_status(
        &mut self,
        agent_id: &str,
        label: &str,
        stage: AgentStage,
        detail: &str,
        elapsed_ms: u64,
    ) -> bool {
        let now = Instant::now();
        let entry = self
            .live_agents
            .entry(agent_id.to_string())
            .or_insert_with(|| LiveAgentStatus {
                agent_id: agent_id.to_string(),
                label: label.to_string(),
                stage: stage.label().into(),
                detail: detail.to_string(),
                current_call: None,
                elapsed_ms,
                first_seen_at: now,
                last_event_at: now,
                model_wait_notice_sent: false,
            });
        entry.label = label.to_string();
        entry.stage = stage.label().into();
        entry.detail = detail.to_string();
        entry.elapsed_ms = elapsed_ms.max(entry.first_seen_at.elapsed().as_millis() as u64);
        entry.last_event_at = now;
        entry.current_call = match stage {
            AgentStage::WaitingForModel => Some("model".into()),
            AgentStage::RunningTool => Some(detail.to_string()),
            AgentStage::WaitingForLock => Some("file_lock".into()),
            AgentStage::ApprovalNeeded => Some("approval".into()),
            AgentStage::Planning | AgentStage::Decomposing | AgentStage::SpawningAgents => {
                Some(stage.label().into())
            }
            _ => None,
        };
        let should_announce_wait = stage == AgentStage::WaitingForModel
            && elapsed_ms >= 10_000
            && !entry.model_wait_notice_sent;
        if should_announce_wait {
            entry.model_wait_notice_sent = true;
        }
        should_announce_wait
    }

    fn apply_live_agent_overlays(&mut self) {
        let mut seen = HashSet::new();
        let spinner = spinner_frame(self.elapsed);

        for agent in &mut self.orch.agents {
            if let Some(live) = self.live_agents.get(&agent.agent_id) {
                seen.insert(agent.agent_id.clone());
                agent.stage = live.stage.clone();
                agent.phase = live.stage.clone();
                agent.current_tool_call = live.current_call.clone();
                agent.last_update = live.detail.clone();
                agent.heartbeat = spinner.to_string();
                agent.elapsed_secs = (live.elapsed_ms / 1000).max(agent.elapsed_secs);
                agent.last_event_age_secs = Some(live.last_event_at.elapsed().as_secs());
                agent.blocked_reason = matches!(
                    live.stage.as_str(),
                    "waiting_for_lock" | "approval_needed" | "failed"
                )
                .then(|| live.detail.clone());
                agent.progress_pct = progress_for_phase(&live.stage).max(agent.progress_pct);
                if live.stage == "failed" {
                    agent.last_error = Some(live.detail.clone());
                }
            }
        }

        for live in self.live_agents.values() {
            if seen.contains(&live.agent_id) {
                continue;
            }
            self.orch.agents.push(AgentInfo {
                agent_id: live.agent_id.clone(),
                label: live.label.clone(),
                goal: String::new(),
                phase: live.stage.clone(),
                current_task: live.detail.clone(),
                current_tool_call: live.current_call.clone(),
                read_files: Vec::new(),
                write_files: Vec::new(),
                held_locks: Vec::new(),
                queued_write_locks: Vec::new(),
                retry_count: 0,
                tool_calls: 0,
                progress_pct: progress_for_phase(&live.stage),
                locked_file: None,
                last_update: live.detail.clone(),
                last_error: (live.stage == "failed").then(|| live.detail.clone()),
                stage: live.stage.clone(),
                heartbeat: spinner.to_string(),
                elapsed_secs: live.elapsed_ms / 1000,
                last_event_age_secs: Some(live.last_event_at.elapsed().as_secs()),
                blocked_reason: matches!(
                    live.stage.as_str(),
                    "waiting_for_lock" | "approval_needed" | "failed"
                )
                .then(|| live.detail.clone()),
            });
        }
        self.orch.agents.sort_by(|a, b| a.label.cmp(&b.label));
    }

    fn check_stale_runner_event(&mut self) {
        if !matches!(self.mode, RunMode::Running | RunMode::Paused) || self.stale_warning_emitted {
            return;
        }
        let Some(last_event_at) = self.last_runner_event_at else {
            return;
        };
        if last_event_at.elapsed() >= Duration::from_secs(15) {
            self.stale_warning_emitted = true;
            let stage = if self.last_runner_stage.is_empty() {
                "unknown".to_string()
            } else {
                self.last_runner_stage.clone()
            };
            let message = format!(
                "No runner event for 15s. Last known stage: {stage}. The run may be blocked or waiting on external I/O."
            );
            self.add_log(LogLevel::Warn, "orchestration", &message);
            self.add_message(MessageRole::System, message);
            self.last_action = "no event for 15s".into();
        }
    }

    pub fn set_tools(&mut self, tools: Vec<ToolDisplay>) {
        self.tool_displays = tools;
    }
    pub fn set_skills(&mut self, skills: Vec<SkillDisplay>) {
        self.skill_displays = skills;
    }
    pub fn set_providers(&mut self, providers: Vec<ProviderDisplay>) {
        self.provider_displays = providers;
    }
}

fn compute_depths(nodes: &mut [GraphNode], edges: &[GraphEdge]) {
    let has_incoming: std::collections::HashSet<String> =
        edges.iter().map(|e| e.to.clone()).collect();
    let mut roots: Vec<String> = nodes
        .iter()
        .filter(|n| !has_incoming.contains(&n.id))
        .map(|n| n.id.clone())
        .collect();
    let mut visited = std::collections::HashSet::new();
    let mut depth = 0;
    while !roots.is_empty() {
        let mut next = Vec::new();
        for rid in &roots {
            if visited.contains(rid) {
                continue;
            }
            visited.insert(rid.clone());
            for node in nodes.iter_mut() {
                if &node.id == rid {
                    node.depth = depth;
                    break;
                }
            }
            for edge in edges.iter() {
                if &edge.from == rid {
                    next.push(edge.to.clone());
                }
            }
        }
        roots = next;
        depth += 1;
    }
}

fn progress_for_phase(phase: &str) -> u32 {
    match phase {
        "queued" => 0,
        "decomposing" => 5,
        "spawning_agents" => 10,
        "waiting_for_lock" => 10,
        "approval_needed" => 10,
        "blocked" => 20,
        "planning" => 25,
        "waiting_for_model" => 45,
        "running_tool" => 60,
        "retrying" => 65,
        "running" => 50,
        "reviewing" => 90,
        "done" | "failed" | "cancelled" => 100,
        _ => 0,
    }
}

fn short_id(id: &str) -> String {
    id.chars().take(8).collect()
}

fn load_display_config() -> Result<odin_core::config::OdinConfig> {
    if let Some(path) = std::env::var_os("RAVEN_CONFIG")
        .or_else(|| std::env::var_os("ODIN_CONFIG"))
        .map(PathBuf::from)
        && path.exists()
    {
        return Ok(odin_core::config::OdinConfig::load(&path)?);
    }
    for candidate in [
        "~/.config/raven/config.yaml",
        "~/.raven-agent/config.yaml",
        "~/.odin/config.yaml",
        "raven.yaml",
        "raven.yml",
        "odin.yaml",
        "odin.yml",
    ] {
        let path = PathBuf::from(shellexpand::tilde(candidate).to_string());
        if path.exists() {
            return Ok(odin_core::config::OdinConfig::load(&path)?);
        }
    }
    Ok(odin_core::config::OdinConfig::default())
}

fn configured_data_dir(config: &odin_core::config::OdinConfig) -> PathBuf {
    config.general.data_dir.as_ref().map_or_else(
        || PathBuf::from(shellexpand::tilde("~/.raven-agent").to_string()),
        |path| PathBuf::from(shellexpand::tilde(&path.to_string_lossy()).to_string()),
    )
}

fn spinner_frame(elapsed: Duration) -> &'static str {
    match (elapsed.as_millis() / 250) % 4 {
        0 => "|",
        1 => "/",
        2 => "-",
        _ => "\\",
    }
}

fn truncate_for_message(value: &str, width: usize) -> String {
    let mut chars = value.chars();
    let mut out: String = chars.by_ref().take(width).collect();
    if chars.next().is_some() && width > 3 {
        out = value.chars().take(width.saturating_sub(3)).collect();
        out.push_str("...");
    }
    out
}
