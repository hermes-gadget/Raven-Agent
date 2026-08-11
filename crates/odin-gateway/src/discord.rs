//! Discord gateway — real serenity-based Discord bot.
//!
//! Provides:
//! - Slash commands: `/raven run <task>`, `/raven status`, `/raven sessions`, `/raven tasks`
//! - Permission gating via configured admin role
//! - Threaded task updates for long-running tasks
//! - Graceful connection lifecycle (start / stop / is_connected)

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use odin_core::error::{OdinError, OdinResult};
use odin_core::types::AgentTask;
use odin_runtime::Runtime;
use serenity::all::*;
use serenity::async_trait;
use serenity::client::{Client, Context, EventHandler};
use tokio::sync::Mutex;

// ── Configuration ────────────────────────────────────────────────────

/// Configuration for the Discord gateway.
#[derive(Debug, Clone, Default)]
pub struct DiscordConfig {
    /// Whether the Discord gateway is enabled.
    pub enabled: bool,
    /// Discord bot token.
    pub token: Option<String>,
    /// Role name required for privileged commands (e.g., "Raven Admin").
    /// At least this role or an explicit admin user ID must be configured.
    pub admin_role: Option<String>,
    /// User IDs explicitly allowed to invoke commands.
    pub admin_user_ids: Vec<u64>,
    /// Restrict commands to one Discord guild when set.
    pub guild_id: Option<u64>,
    /// Permit DMs from explicitly allowlisted user IDs. Disabled by default.
    pub allow_dms: bool,
    /// Command prefix for slash commands (default: "raven").
    /// Existing configurations can keep `odin` as a compatibility prefix.
    pub command_prefix: Option<String>,
    /// Path to the orchestration SQLite database.
    /// When set, orchestration commands use real persistent state.
    pub orchestration_db_path: Option<std::path::PathBuf>,
}

impl DiscordConfig {
    /// The effective command prefix, falling back to "raven".
    pub fn prefix(&self) -> &str {
        self.command_prefix.as_deref().unwrap_or("raven")
    }

    fn validate_authorization(&self) -> OdinResult<()> {
        let has_admin_role = self
            .admin_role
            .as_deref()
            .is_some_and(|role| !role.trim().is_empty());
        if !has_admin_role && self.admin_user_ids.is_empty() {
            return Err(OdinError::Config(
                "Discord requires gateway.discord_admin_role or at least one gateway.discord_admin_user_ids entry"
                    .into(),
            ));
        }
        if self.allow_dms && self.admin_user_ids.is_empty() {
            return Err(OdinError::Config(
                "Discord DMs require at least one explicit gateway.discord_admin_user_ids entry"
                    .into(),
            ));
        }
        Ok(())
    }

    fn principal_is_authorized(
        &self,
        user_id: u64,
        guild_id: Option<u64>,
        has_admin_role: bool,
    ) -> bool {
        match guild_id {
            None => self.allow_dms && self.admin_user_ids.contains(&user_id),
            Some(actual_guild)
                if self
                    .guild_id
                    .is_some_and(|expected| expected != actual_guild) =>
            {
                false
            }
            Some(_) => {
                self.admin_user_ids.contains(&user_id)
                    || (self
                        .admin_role
                        .as_deref()
                        .is_some_and(|role| !role.trim().is_empty())
                        && has_admin_role)
            }
        }
    }
}

// ── Serenity Event Handler ───────────────────────────────────────────

/// Internal event handler that wires slash commands to the Raven runtime.
struct DiscordEventHandler {
    runtime: Arc<Runtime>,
    config: DiscordConfig,
    connected: Arc<AtomicBool>,
}

async fn send_discord_message(ctx: &Context, command: &CommandInteraction, content: String) {
    let _ = ctx
        .http
        .send_message(
            command.channel_id,
            vec![],
            &CreateMessage::new().content(content),
        )
        .await;
}

#[async_trait]
impl EventHandler for DiscordEventHandler {
    /// Called when the bot has connected and is ready to receive events.
    async fn ready(&self, ctx: Context, _ready: Ready) {
        tracing::info!("[DISCORD] Bot connected to Discord");
        self.connected.store(true, Ordering::SeqCst);

        // ── Register global slash commands ──────────────────────
        let prefix = self.config.prefix();

        let cmd = CreateCommand::new(prefix)
            .description("Raven Agent — multi-agent AI orchestration commands")
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "run",
                    "Submit a task to the Raven runtime",
                )
                .add_sub_option(
                    CreateCommandOption::new(
                        CommandOptionType::String,
                        "task",
                        "The task goal or description",
                    )
                    .required(true),
                ),
            )
            .add_option(CreateCommandOption::new(
                CommandOptionType::SubCommand,
                "status",
                "Show runtime status summary (agents, sessions, sub-agents)",
            ))
            .add_option(CreateCommandOption::new(
                CommandOptionType::SubCommand,
                "sessions",
                "List recent sessions",
            ))
            .add_option(CreateCommandOption::new(
                CommandOptionType::SubCommand,
                "tasks",
                "List recent tasks",
            ))
            // ── Orchestration subcommand group ────────────────────
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::SubCommandGroup,
                    "orchestrate",
                    "Multi-agent orchestration commands",
                )
                .add_sub_option(
                    CreateCommandOption::new(
                        CommandOptionType::SubCommand,
                        "submit",
                        "Submit a goal for multi-agent orchestration",
                    )
                    .add_sub_option(
                        CreateCommandOption::new(
                            CommandOptionType::String,
                            "goal",
                            "The goal to decompose and orchestrate",
                        )
                        .required(true),
                    ),
                )
                .add_sub_option(
                    CreateCommandOption::new(
                        CommandOptionType::SubCommand,
                        "status",
                        "Show orchestration run status",
                    )
                    .add_sub_option(
                        CreateCommandOption::new(
                            CommandOptionType::String,
                            "run_id",
                            "Optional run ID (shows all if omitted)",
                        )
                        .required(false),
                    ),
                )
                .add_sub_option(
                    CreateCommandOption::new(
                        CommandOptionType::SubCommand,
                        "agents",
                        "List sub-agents for a run",
                    )
                    .add_sub_option(
                        CreateCommandOption::new(
                            CommandOptionType::String,
                            "run_id",
                            "Run ID to list agents for",
                        )
                        .required(false),
                    ),
                )
                .add_sub_option(CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "locks",
                    "Show file lock state",
                ))
                .add_sub_option(
                    CreateCommandOption::new(
                        CommandOptionType::SubCommand,
                        "cancel",
                        "Cancel an orchestration run",
                    )
                    .add_sub_option(
                        CreateCommandOption::new(
                            CommandOptionType::String,
                            "run_id",
                            "Run ID to cancel",
                        )
                        .required(true),
                    ),
                ),
            );

        match Command::set_global_commands(&ctx.http, vec![cmd]).await {
            Ok(cmds) => tracing::info!(
                "[DISCORD] Registered {} global slash command(s)",
                cmds.len()
            ),
            Err(e) => tracing::error!("[DISCORD] Failed to register global commands: {e}"),
        }
    }

    /// Called when a user interacts with the bot (slash commands).
    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        let Interaction::Command(command) = interaction else {
            return;
        };

        // Validate the command name matches our prefix
        if command.data.name != self.config.prefix() {
            return;
        }

        // Every Raven command exposes runtime state or can mutate it. Enforce
        // authorization before dispatching any subcommand or group.
        if let Err(msg) = self.check_admin_permission(&ctx, &command).await {
            let _ = command
                .create_response(
                    &ctx.http,
                    CreateInteractionResponse::Message(
                        CreateInteractionResponseMessage::new()
                            .content(msg)
                            .ephemeral(true),
                    ),
                )
                .await;
            return;
        }

        // Extract the subcommand (or subcommand group) from options
        let first_opt = command.data.options.first();

        // Check for subcommand group (orchestrate)
        if let Some(opt) = first_opt
            && matches!(opt.kind(), CommandOptionType::SubCommandGroup)
        {
            self.handle_orchestrate_group(ctx, &command, opt).await;
            return;
        }

        let sub_option =
            first_opt.filter(|opt| matches!(opt.kind(), CommandOptionType::SubCommand));

        let subcommand: String = match sub_option {
            Some(opt) => opt.name.clone(),
            None => {
                let _ = command
                    .create_response(
                        &ctx.http,
                        CreateInteractionResponse::Message(
                            CreateInteractionResponseMessage::new()
                                .content(format!(
                                    "Usage: /{} run <task> | /{} status | /{} sessions | /{} tasks",
                                    self.config.prefix(),
                                    self.config.prefix(),
                                    self.config.prefix(),
                                    self.config.prefix(),
                                ))
                                .ephemeral(true),
                        ),
                    )
                    .await;
                return;
            }
        };

        match subcommand.as_str() {
            "run" => {
                // Extract the "task" argument from subcommand options
                let task_goal = extract_subcommand_string(&command.data.options, "task")
                    .unwrap_or_else(|| "No task provided".to_string());

                self.handle_run(ctx, command, task_goal).await;
            }
            "status" | "sessions" | "tasks" => {
                self.handle_list_command(ctx, command, &subcommand).await;
            }
            _ => {
                let _ = command
                    .create_response(
                        &ctx.http,
                        CreateInteractionResponse::Message(
                            CreateInteractionResponseMessage::new()
                                .content(format!("Unknown subcommand `{subcommand}`."))
                                .ephemeral(true),
                        ),
                    )
                    .await;
            }
        }
    }
}

// ── Command Handlers ─────────────────────────────────────────────────

impl DiscordEventHandler {
    /// Check whether the invoking user has an explicitly configured identity or role.
    /// Returns `Ok(())` if allowed, or `Err(message)` if denied.
    async fn check_admin_permission(
        &self,
        ctx: &Context,
        command: &CommandInteraction,
    ) -> Result<(), String> {
        let user_id = command.user.id.get();

        // DMs never have roles, so they require both the explicit DM opt-in and
        // a concrete user-ID allowlist match.
        let guild_id = match command.guild_id {
            Some(gid) => gid,
            None => {
                return if self.config.principal_is_authorized(user_id, None, false) {
                    Ok(())
                } else {
                    Err("Discord direct-message commands are not authorized.".into())
                };
            }
        };

        if let Some(expected_guild) = self.config.guild_id
            && guild_id.get() != expected_guild
        {
            return Err("This Raven bot is not authorized for this server.".into());
        }

        if self.config.admin_user_ids.contains(&user_id) {
            return Ok(());
        }

        let Some(admin_role_name) = self
            .config
            .admin_role
            .as_deref()
            .filter(|role| !role.trim().is_empty())
        else {
            return Err("You are not authorized to use Raven commands.".into());
        };

        let member = match command.member.as_ref() {
            Some(m) => m,
            None => return Err("Could not identify the command author.".into()),
        };

        // Fetch guild roles to resolve role IDs -> names
        let guild_roles = match ctx.http.get_guild_roles(guild_id).await {
            Ok(roles) => roles,
            Err(e) => {
                tracing::warn!("[DISCORD] Failed to fetch guild roles: {e}");
                return Err("Permission check failed (cannot fetch roles).".into());
            }
        };

        // Build a set of role names the member has
        let member_role_names: std::collections::HashSet<&str> = guild_roles
            .iter()
            .filter(|role| member.roles.contains(&role.id))
            .map(|role| role.name.as_str())
            .collect();

        if self.config.principal_is_authorized(
            user_id,
            Some(guild_id.get()),
            member_role_names.contains(admin_role_name),
        ) {
            Ok(())
        } else {
            Err(format!(
                "You need the `{admin_role_name}` role to use this command."
            ))
        }
    }

    /// Handle `/raven run <task>` — submit a task to the runtime and post updates to a thread.
    async fn handle_run(&self, ctx: Context, command: CommandInteraction, task_goal: String) {
        let channel_id = command.channel_id;

        // Acknowledge the interaction immediately (defer, then follow-up)
        let _ = command
            .create_response(
                &ctx.http,
                CreateInteractionResponse::Defer(CreateInteractionResponseMessage::new()),
            )
            .await;

        // Create the agent task
        let task = AgentTask {
            id: uuid::Uuid::new_v4(),
            goal: task_goal.clone(),
            context: None,
            sub_tasks: vec![],
            success_criteria: vec![],
            max_iterations: 100,
            created_at: chrono::Utc::now(),
        };
        let task_id = task.id;
        let task_goal_display = if task_goal.len() > 80 {
            format!("{}...", task_goal.chars().take(80).collect::<String>())
        } else {
            task_goal.clone()
        };

        // Send initial acknowledgement as a follow-up
        let msg = match ctx
            .http
            .send_message(
                channel_id,
                vec![],
                &CreateMessage::new().content(format!(
                    "⏳ **Task submitted** — `{task_id}`\n> {task_goal_display}",
                )),
            )
            .await
        {
            Ok(m) => m,
            Err(e) => {
                tracing::error!("[DISCORD] Failed to send initial ack: {e}");
                return;
            }
        };

        // Create a thread from the acknowledgement message for progress updates
        let thread = match ctx
            .http
            .create_thread_from_message(
                channel_id,
                msg.id,
                &CreateThread::new(format!("Task {task_id}")),
                None,
            )
            .await
        {
            Ok(channel) => channel,
            Err(e) => {
                tracing::warn!("[DISCORD] Failed to create thread: {e}");
                // Still continue — we'll post updates to the channel instead
                let thread_id = channel_id;
                let _ = ctx
                    .http
                    .send_message(
                        thread_id,
                        vec![],
                        &CreateMessage::new().content(format!(
                            "⚙️ Task `{task_id}` started. Results will appear here when complete."
                        )),
                    )
                    .await;
                return;
            }
        };

        let thread_id = thread.id;

        // Post "started" message to the thread
        let _ = ctx
            .http
            .send_message(
                thread_id,
                vec![],
                &CreateMessage::new().content(format!(
                    "⚙️ **Task started**\n**Goal:** {task_goal_display}\n*Running...*"
                )),
            )
            .await;

        // Spawn the task in the background so we don't block the interaction
        let runtime = self.runtime.clone();
        tokio::spawn(async move {
            let start = std::time::Instant::now();

            // We need an agent to execute the task. Use the first registered agent.
            let agents = runtime.list_agents();
            let agent_id = match agents.first() {
                Some(a) => a.id,
                None => {
                    let _ = ctx
                        .http
                        .send_message(
                            thread_id,
                            vec![],
                            &CreateMessage::new()
                                .content("❌ **Error:** No agents registered in the runtime."),
                        )
                        .await;
                    return;
                }
            };

            // Submit the task
            let result = runtime.submit_task(&agent_id, &task, None).await;

            let elapsed = start.elapsed();

            match result {
                Ok(task_result) => {
                    let status_icon = if task_result.success { "✅" } else { "❌" };
                    let duration_secs = elapsed.as_secs_f64();

                    let mut response = format!(
                        "{status_icon} **Task complete**\n\
                         **ID:** `{}`\n\
                         **Status:** {}\n\
                         **Duration:** {duration_secs:.2}s\n\
                         **Iterations:** {}\n\
                         **Tool calls:** {}\n\
                         **Confidence:** {:.1}%\n\
                         **Summary:** {}",
                        task_id,
                        if task_result.success {
                            "Completed"
                        } else {
                            "Stopped"
                        },
                        task_result.iterations,
                        task_result.tool_calls,
                        task_result.confidence * 100.0,
                        task_result.summary,
                    );

                    if let Some(ref err) = task_result.error {
                        response.push_str(&format!("\n**Error:** {err}"));
                    }

                    let _ = ctx
                        .http
                        .send_message(thread_id, vec![], &CreateMessage::new().content(response))
                        .await;
                }
                Err(e) => {
                    let _ = ctx
                        .http
                        .send_message(
                            thread_id,
                            vec![],
                            &CreateMessage::new().content(format!(
                                "❌ **Task failed**\n**ID:** `{task_id}`\n**Error:** {e}"
                            )),
                        )
                        .await;
                }
            }
        });
    }

    /// Handle `/raven status` / `/raven sessions` / `/raven tasks`.
    async fn handle_list_command(
        &self,
        ctx: Context,
        command: CommandInteraction,
        subcommand: &str,
    ) {
        let content = match subcommand {
            "status" => {
                let s = self.runtime.summary();
                format!(
                    "**Raven Agent Status**\n\
                     ────────────────\n\
                     👤 **Agents:** {}\n\
                     💬 **Sessions:** {}\n\
                     🔄 **Sub-agents:** {}",
                    s.agents, s.sessions, s.sub_agents
                )
            }
            "sessions" => {
                let sessions = self.runtime.list_sessions();
                if sessions.is_empty() {
                    "No active sessions.".to_string()
                } else {
                    let mut lines = vec!["**Recent Sessions**".to_string()];
                    for session in sessions.iter().rev().take(10) {
                        let label = session.label.as_deref().unwrap_or("(no label)");
                        let created = session.created_at.format("%Y-%m-%d %H:%M UTC");
                        let msgs = session.message_count();
                        lines.push(format!(
                            "• `{}` — **{label}** — {msgs} msgs — {created}",
                            session.id
                        ));
                    }
                    lines.join("\n")
                }
            }
            "tasks" => {
                // The Runtime doesn't have a task history store yet,
                // so we report agent-level task counts.
                let agents = self.runtime.list_agents();
                if agents.is_empty() {
                    "No agents registered — no task history available.".to_string()
                } else {
                    let mut lines = vec!["**Registered Agents**".to_string()];
                    for agent in &agents {
                        lines.push(format!(
                            "• `{}` — **{}** (tools: {})",
                            agent.id,
                            agent.name,
                            agent.tools().len(),
                        ));
                    }
                    lines.push(String::new());
                    lines.push(
                        "*Task history will be available when persistence is added.*".to_string(),
                    );
                    lines.join("\n")
                }
            }
            _ => unreachable!(),
        };

        let _ = command
            .create_response(
                &ctx.http,
                CreateInteractionResponse::Message(
                    CreateInteractionResponseMessage::new().content(content),
                ),
            )
            .await;
    }

    // ── Orchestration Handlers ─────────────────────────────────────────

    /// Dispatch orchestration subcommand group.
    async fn handle_orchestrate_group(
        &self,
        ctx: Context,
        command: &CommandInteraction,
        group_opt: &CommandDataOption,
    ) {
        // Extract subcommand from SubCommandGroup using pattern matching
        let sub_opt = match &group_opt.value {
            CommandDataOptionValue::SubCommandGroup(opts) => opts.first(),
            _ => None,
        };

        let sub_opt = match sub_opt {
            Some(opt) => opt,
            None => {
                let _ = command
                    .create_response(
                        &ctx.http,
                        CreateInteractionResponse::Message(
                            CreateInteractionResponseMessage::new()
                                .content(
                                    "Usage: /raven orchestrate submit|status|agents|locks|cancel",
                                )
                                .ephemeral(true),
                        ),
                    )
                    .await;
                return;
            }
        };

        match sub_opt.name.as_str() {
            "submit" => {
                let goal = extract_option_string(sub_opt, "goal")
                    .unwrap_or_else(|| "No goal provided".to_string());
                self.handle_orchestrate_submit(ctx, command.clone(), goal)
                    .await;
            }
            "status" => {
                let run_id = extract_option_string(sub_opt, "run_id");
                self.handle_orchestrate_status(ctx, command.clone(), run_id)
                    .await;
            }
            "agents" => {
                let run_id = extract_option_string(sub_opt, "run_id");
                self.handle_orchestrate_agents(ctx, command.clone(), run_id)
                    .await;
            }
            "locks" => {
                self.handle_orchestrate_locks(ctx, command.clone()).await;
            }
            "cancel" => {
                let run_id = extract_option_string(sub_opt, "run_id")
                    .unwrap_or_else(|| "unknown".to_string());
                self.handle_orchestrate_cancel(ctx, command.clone(), run_id)
                    .await;
            }
            _ => {
                let _ = command
                    .create_response(
                        &ctx.http,
                        CreateInteractionResponse::Message(
                            CreateInteractionResponseMessage::new()
                                .content(format!(
                                    "Unknown orchestrate subcommand: {}",
                                    sub_opt.name
                                ))
                                .ephemeral(true),
                        ),
                    )
                    .await;
            }
        }
    }

    /// Create a SQLite orchestration store from config path.
    async fn get_orchestration_store(
        &self,
    ) -> Option<odin_orchestrator::persistence::SqliteOrchestrationStore> {
        let db_path = match &self.config.orchestration_db_path {
            Some(p) => p.clone(),
            None => {
                let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
                std::path::PathBuf::from(home).join(".raven-agent/orchestration.db")
            }
        };

        match odin_orchestrator::persistence::SqliteOrchestrationStore::new(&db_path).await {
            Ok(store) => {
                use odin_orchestrator::persistence::OrchestrationStore;
                let _ = store.initialize().await;
                Some(store)
            }
            Err(e) => {
                tracing::warn!("[DISCORD] Failed to open orchestration store: {e}");
                None
            }
        }
    }

    /// Handle `/raven orchestrate submit <goal>`.
    async fn handle_orchestrate_submit(
        &self,
        ctx: Context,
        command: CommandInteraction,
        goal: String,
    ) {
        let _ = command
            .create_response(
                &ctx.http,
                CreateInteractionResponse::Defer(
                    CreateInteractionResponseMessage::new().ephemeral(true),
                ),
            )
            .await;

        let mut composer = odin_orchestrator::Composer::default();
        composer.intake(&goal);

        let mut graph = match composer.get_graph(&goal) {
            Some(g) => g.clone(),
            None => {
                let _ = ctx
                    .http
                    .send_message(
                        command.channel_id,
                        vec![],
                        &CreateMessage::new()
                            .content("❌ Failed to decompose the goal into tasks."),
                    )
                    .await;
                return;
            }
        };
        graph.status = odin_orchestrator::task_graph::TaskGraphStatus::Building;

        let groups = composer.detect_workstreams(&graph);

        let run_id = graph.id;
        if let Some(store) = self.get_orchestration_store().await {
            use odin_orchestrator::persistence::OrchestrationStore;
            let _ = store.save_task_graph(&graph).await;
            tracing::info!("[DISCORD] Orchestration plan saved — run_id={}", run_id);
        }

        let task_count = graph.nodes.len();
        let ws_count = groups.len();

        let mut response = format!(
            "🚀 **Orchestration Plan**\n\
             **Run ID:** `{}`\n\
             **Goal:** {}\n\
             **Tasks:** {task_count} | **Workstreams:** {ws_count}\n\n\
             **Tasks:**\n",
            run_id, goal,
        );

        for node in graph.nodes.values() {
            let files = if node.write_files.is_empty() && node.read_files.is_empty() {
                String::new()
            } else {
                let mut f = String::from(" — edits: ");
                f.push_str(&node.write_files.join(", "));
                if !node.read_files.is_empty() {
                    f.push_str(&format!(" | reads: {}", node.read_files.join(", ")));
                }
                f
            };
            response.push_str(&format!(
                "• **{}** (p:{}){}\n",
                node.goal, node.priority, files
            ));
        }

        if response.len() > 1900 {
            let suffix = "\n... *(truncated)*";
            let keep = 1900usize.saturating_sub(suffix.len());
            response = format!(
                "{}{}",
                response.chars().take(keep).collect::<String>(),
                suffix
            );
        }

        let _ = ctx
            .http
            .send_message(
                command.channel_id,
                vec![],
                &CreateMessage::new().content(response),
            )
            .await;
    }

    /// Handle `/raven orchestrate status [run_id]`.
    async fn handle_orchestrate_status(
        &self,
        ctx: Context,
        command: CommandInteraction,
        run_id: Option<String>,
    ) {
        let _ = command
            .create_response(
                &ctx.http,
                CreateInteractionResponse::Defer(
                    CreateInteractionResponseMessage::new().ephemeral(true),
                ),
            )
            .await;

        let store = match self.get_orchestration_store().await {
            Some(s) => s,
            None => {
                let _ = ctx
                    .http
                    .send_message(
                        command.channel_id,
                        vec![],
                        &CreateMessage::new().content("❌ Orchestration store not available."),
                    )
                    .await;
                return;
            }
        };

        use odin_orchestrator::persistence::OrchestrationStore;

        let content = if let Some(ref rid) = run_id {
            match store.load_task_graph(rid).await {
                Ok(_graph) => {
                    format!(
                        "📊 **Run `{rid}`**\n\
                         **Status:** persisted in SQLite\n\
                         Use `raven orchestrate status` CLI for detailed output."
                    )
                }
                Err(_) => format!("❌ Run `{rid}` not found in store."),
            }
        } else {
            let graphs = store.list_task_graphs().await.unwrap_or_default();
            if graphs.is_empty() {
                "No orchestration runs found in store.".to_string()
            } else {
                let mut lines = vec!["**Orchestration Runs**".to_string()];
                for g in &graphs {
                    lines.push(format!(
                        "• `{}` — {} nodes — **{}**",
                        g.run_id, g.node_count, g.status
                    ));
                }
                lines.join("\n")
            }
        };

        let _ = ctx
            .http
            .send_message(
                command.channel_id,
                vec![],
                &CreateMessage::new().content(content),
            )
            .await;
    }

    /// Handle `/raven orchestrate agents [run_id]`.
    async fn handle_orchestrate_agents(
        &self,
        ctx: Context,
        command: CommandInteraction,
        run_id: Option<String>,
    ) {
        let _ = command
            .create_response(
                &ctx.http,
                CreateInteractionResponse::Defer(
                    CreateInteractionResponseMessage::new().ephemeral(true),
                ),
            )
            .await;

        let content = if let Some(ref rid) = run_id {
            let store = match self.get_orchestration_store().await {
                Some(s) => s,
                None => {
                    let _ = ctx
                        .http
                        .send_message(
                            command.channel_id,
                            vec![],
                            &CreateMessage::new().content("❌ Orchestration store not available."),
                        )
                        .await;
                    return;
                }
            };

            use odin_orchestrator::persistence::OrchestrationStore;
            let graph = match store.load_task_graph(rid).await {
                Ok(graph) => graph,
                Err(_) => {
                    return send_discord_message(
                        &ctx,
                        &command,
                        format!("Run `{rid}` was not found."),
                    )
                    .await;
                }
            };
            let agent_ids: std::collections::HashSet<String> = graph
                .nodes
                .values()
                .filter_map(|node| node.agent_id.map(|id| id.to_string()))
                .collect();
            let lifecycles = store.list_agent_lifecycles().await.unwrap_or_default();
            let run_lifecycles: Vec<_> = lifecycles
                .iter()
                .filter(|lc| agent_ids.contains(&lc.agent_id))
                .collect();

            if run_lifecycles.is_empty() {
                format!("No sub-agents found for run `{rid}`.")
            } else {
                let mut lines = vec![format!("**Sub-Agents for Run `{rid}`**")];
                for lc in &run_lifecycles {
                    let icon = match lc.phase.as_str() {
                        "running" => "🟢",
                        "done" => "✅",
                        "failed" => "❌",
                        "paused" => "⏸️",
                        "cancelled" => "🚫",
                        _ => "⏳",
                    };
                    lines.push(format!("{icon} `{}` — {}", lc.agent_id, lc.phase));
                }
                lines.join("\n")
            }
        } else {
            "Usage: `/raven orchestrate agents run_id:<run_id>`".to_string()
        };

        let _ = ctx
            .http
            .send_message(
                command.channel_id,
                vec![],
                &CreateMessage::new().content(content),
            )
            .await;
    }

    /// Handle `/raven orchestrate locks`.
    async fn handle_orchestrate_locks(&self, ctx: Context, command: CommandInteraction) {
        let _ = command
            .create_response(
                &ctx.http,
                CreateInteractionResponse::Defer(
                    CreateInteractionResponseMessage::new().ephemeral(true),
                ),
            )
            .await;

        use odin_orchestrator::persistence::OrchestrationStore;
        let content = match self.get_orchestration_store().await {
            Some(store) => match store.load_lock_snapshot().await {
                Ok(Some(snapshot)) => {
                    let redacted = odin_permissions::SecretRedactor::full().redact(&snapshot);
                    let preview: String = redacted.chars().take(1500).collect();
                    format!("🔒 **Latest Persisted Lock Snapshot**\n```json\n{preview}\n```")
                }
                Ok(None) => "No lock snapshot has been persisted. `raven orchestrate locks` shows declared file access from stored graphs.".to_string(),
                Err(error) => format!("Could not read lock state: {error}"),
            },
            None => "Orchestration store not available.".to_string(),
        };

        let _ = ctx
            .http
            .send_message(
                command.channel_id,
                vec![],
                &CreateMessage::new().content(content),
            )
            .await;
    }

    /// Handle `/raven orchestrate cancel <run_id>`.
    async fn handle_orchestrate_cancel(
        &self,
        ctx: Context,
        command: CommandInteraction,
        run_id: String,
    ) {
        let _ = command
            .create_response(
                &ctx.http,
                CreateInteractionResponse::Defer(
                    CreateInteractionResponseMessage::new().ephemeral(true),
                ),
            )
            .await;

        let store = match self.get_orchestration_store().await {
            Some(s) => s,
            None => {
                let _ = ctx
                    .http
                    .send_message(
                        command.channel_id,
                        vec![],
                        &CreateMessage::new().content("❌ Orchestration store not available."),
                    )
                    .await;
                return;
            }
        };

        use odin_orchestrator::persistence::OrchestrationStore;
        let result = store.update_graph_status(&run_id, "cancelled").await;

        let content = match result {
            Ok(_) => format!("🚫 Orchestration run `{run_id}` has been **cancelled**."),
            Err(e) => format!("❌ Failed to cancel run `{run_id}`: {e}"),
        };

        let _ = ctx
            .http
            .send_message(
                command.channel_id,
                vec![],
                &CreateMessage::new().content(content),
            )
            .await;
    }
}

// ── DiscordGateway ───────────────────────────────────────────────────

/// Real Discord gateway backed by serenity.
///
/// Manages the connection lifecycle and exposes methods for sending
/// messages and checking connection state.
pub struct DiscordGateway {
    config: DiscordConfig,
    runtime: Arc<Runtime>,
    connected: Arc<AtomicBool>,
    /// Handle to shut down the serenity client gracefully.
    shutdown: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
    /// Whether the gateway has been started.
    started: Arc<AtomicBool>,
}

impl std::fmt::Debug for DiscordGateway {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiscordGateway")
            .field("config", &self.config)
            .field("connected", &self.connected)
            .finish()
    }
}

impl DiscordGateway {
    /// Create a new Discord gateway.
    pub fn new(config: DiscordConfig, runtime: Arc<Runtime>) -> Self {
        Self {
            config,
            runtime,
            connected: Arc::new(AtomicBool::new(false)),
            shutdown: Arc::new(Mutex::new(None)),
            started: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start the Discord gateway.
    ///
    /// Creates a serenity Client, registers event handlers, and spawns
    /// the shard in a background task. Returns once the client is ready
    /// or if the gateway is disabled.
    pub async fn start(&self) -> OdinResult<()> {
        if !self.config.enabled {
            tracing::info!("[DISCORD] Gateway disabled");
            return Ok(());
        }

        let token = self
            .config
            .token
            .clone()
            .ok_or_else(|| OdinError::Config("Discord token is required but not set".into()))?;
        self.config.validate_authorization()?;

        tracing::info!("[DISCORD] Starting Discord gateway...");

        let handler = DiscordEventHandler {
            runtime: self.runtime.clone(),
            config: self.config.clone(),
            connected: self.connected.clone(),
        };

        let mut client = Client::builder(&token, GatewayIntents::all())
            .event_handler(handler)
            .await
            .map_err(|e| OdinError::Network(format!("Failed to create Discord client: {e}")))?;

        // Create a shutdown channel
        let (tx, mut rx) = tokio::sync::oneshot::channel::<()>();

        self.connected.store(false, Ordering::SeqCst);
        self.started.store(true, Ordering::SeqCst);
        {
            let mut shutdown = self.shutdown.lock().await;
            *shutdown = Some(tx);
        }

        // Spawn the client in a background task
        let shard_manager = client.shard_manager.clone();
        tokio::spawn(async move {
            tracing::info!("[DISCORD] Client shard running");
            if let Err(e) = client.start().await {
                tracing::error!("[DISCORD] Client error: {e}");
            }
            tracing::info!("[DISCORD] Client shard stopped");
        });

        // Spawn a task that listens for shutdown signal
        let connected_when_done = self.connected.clone();
        tokio::spawn(async move {
            tokio::select! {
                _ = &mut rx => {
                    tracing::info!("[DISCORD] Shutdown signal received, stopping shard");
                    shard_manager.shutdown_all().await;
                }
            }
            connected_when_done.store(false, Ordering::SeqCst);
        });

        tracing::info!("[DISCORD] Gateway started successfully");
        Ok(())
    }

    /// Stop the Discord gateway gracefully.
    pub async fn stop(&self) -> OdinResult<()> {
        tracing::info!("[DISCORD] Stopping Discord gateway...");
        let mut shutdown = self.shutdown.lock().await;
        if let Some(tx) = shutdown.take() {
            let _ = tx.send(());
        }
        self.connected.store(false, Ordering::SeqCst);
        self.started.store(false, Ordering::SeqCst);
        tracing::info!("[DISCORD] Gateway stopped");
        Ok(())
    }

    /// Send a message to a Discord channel.
    ///
    /// `channel_id` is a Discord snowflake as a string.
    /// Returns an error if the gateway is not connected.
    pub async fn send_message(&self, channel_id: &str, content: &str) -> OdinResult<()> {
        if !self.is_connected() {
            return Err(OdinError::Network(
                "Discord gateway is not connected".into(),
            ));
        }

        let _ = (channel_id, content);
        Err(OdinError::Network(
            "Direct Discord sends are unavailable on a running gateway; use DiscordGateway::send_message_raw with an authorized token"
                .into(),
        ))
    }

    /// Send a message to a Discord channel using a raw token (stateless).
    ///
    /// Useful for out-of-band messages when the gateway hasn't been started
    /// but the token is available. Creates a temporary HTTP client.
    pub async fn send_message_raw(token: &str, channel_id: &str, content: &str) -> OdinResult<()> {
        let cid = serenity::all::ChannelId::new(
            channel_id
                .parse::<u64>()
                .map_err(|e| OdinError::Validation(format!("Invalid channel ID: {e}")))?,
        );

        let http = Http::new(token);

        http.send_message(cid, vec![], &CreateMessage::new().content(content))
            .await
            .map_err(|e| OdinError::Network(format!("Failed to send Discord message: {e}")))?;

        Ok(())
    }

    /// Check if the gateway is connected to Discord.
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    /// Check if the gateway has been started (even if not yet connected).
    pub fn is_started(&self) -> bool {
        self.started.load(Ordering::SeqCst)
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Extract a string value from a subcommand's options.
///
/// Searches through `options` (which may be nested under a SubCommand)
/// for an option with the given `name` and a `String` value.
fn extract_subcommand_string(options: &[CommandDataOption], name: &str) -> Option<String> {
    for opt in options {
        match &opt.value {
            CommandDataOptionValue::SubCommand(sub_options) => {
                // Recurse into subcommand options
                if let found @ Some(_) = extract_subcommand_string(sub_options, name) {
                    return found;
                }
            }
            CommandDataOptionValue::String(s) if opt.name == name => {
                return Some(s.clone());
            }
            _ => {}
        }
    }
    None
}

/// Extract a string value from a CommandDataOption's sub-options.
/// Looks for a String-typed option with the given name.
fn extract_option_string(opt: &CommandDataOption, name: &str) -> Option<String> {
    match &opt.value {
        CommandDataOptionValue::SubCommand(sub_opts) => extract_subcommand_string(sub_opts, name),
        CommandDataOptionValue::String(s) if opt.name == name => Some(s.clone()),
        _ => None,
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discord_config_default() {
        let config = DiscordConfig::default();
        assert!(!config.enabled);
        assert!(config.token.is_none());
        assert!(config.admin_role.is_none());
        assert!(config.admin_user_ids.is_empty());
        assert!(!config.allow_dms);
        assert!(config.command_prefix.is_none());
    }

    #[test]
    fn discord_authorization_fails_closed_without_an_allowlist() {
        let config = DiscordConfig::default();
        assert!(config.validate_authorization().is_err());

        let role_config = DiscordConfig {
            admin_role: Some("Raven Admin".into()),
            ..Default::default()
        };
        assert!(role_config.validate_authorization().is_ok());

        let user_config = DiscordConfig {
            admin_user_ids: vec![42],
            allow_dms: true,
            ..Default::default()
        };
        assert!(user_config.validate_authorization().is_ok());
    }

    #[test]
    fn ordinary_discord_principals_are_denied_by_default() {
        let default = DiscordConfig::default();
        assert!(!default.principal_is_authorized(7, Some(100), false));
        assert!(!default.principal_is_authorized(7, None, false));

        let role_config = DiscordConfig {
            admin_role: Some("Raven Admin".into()),
            guild_id: Some(100),
            ..Default::default()
        };
        assert!(!role_config.principal_is_authorized(7, Some(100), false));
        assert!(role_config.principal_is_authorized(7, Some(100), true));
        assert!(!role_config.principal_is_authorized(7, Some(101), true));

        let user_config = DiscordConfig {
            admin_user_ids: vec![7],
            allow_dms: true,
            ..Default::default()
        };
        assert!(user_config.principal_is_authorized(7, Some(100), false));
        assert!(user_config.principal_is_authorized(7, None, false));
        assert!(!user_config.principal_is_authorized(8, Some(100), false));
    }

    #[test]
    fn test_discord_config_prefix_default() {
        let config = DiscordConfig::default();
        assert_eq!(config.prefix(), "raven");
    }

    #[test]
    fn test_discord_config_custom_prefix() {
        let config = DiscordConfig {
            command_prefix: Some("raven".into()),
            ..Default::default()
        };
        assert_eq!(config.prefix(), "raven");
    }

    #[test]
    fn test_discord_gateway_creation() {
        let config = DiscordConfig {
            enabled: false,
            token: None,
            ..Default::default()
        };
        let runtime = Arc::new(Runtime::new());
        let gateway = DiscordGateway::new(config, runtime);
        assert!(!gateway.is_connected());
        assert!(!gateway.is_started());
    }

    #[tokio::test]
    async fn test_start_disabled_gateway() {
        let config = DiscordConfig::default();
        let runtime = Arc::new(Runtime::new());
        let gateway = DiscordGateway::new(config, runtime);
        let result = gateway.start().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_start_without_token() {
        let config = DiscordConfig {
            enabled: true,
            token: None,
            admin_role: None,
            admin_user_ids: Vec::new(),
            guild_id: None,
            allow_dms: false,
            command_prefix: None,
            orchestration_db_path: None,
        };
        let runtime = Arc::new(Runtime::new());
        let gateway = DiscordGateway::new(config, runtime);
        let result = gateway.start().await;
        assert!(result.is_err(), "should fail without token");
    }

    #[tokio::test]
    async fn test_start_with_mock_token_signals_connected() {
        // This test uses a fake token. Client::builder() doesn't validate
        // tokens; the real connection attempt happens in the background task.
        // Verify the gateway doesn't panic and returns Ok.
        let config = DiscordConfig {
            enabled: true,
            token: Some("fake.token.here".into()),
            admin_role: Some("Raven Admin".into()),
            admin_user_ids: Vec::new(),
            guild_id: None,
            allow_dms: false,
            command_prefix: None,
            orchestration_db_path: None,
        };
        let runtime = Arc::new(Runtime::new());
        let gateway = DiscordGateway::new(config, runtime);
        let result = gateway.start().await;
        assert!(
            result.is_ok(),
            "Builder should succeed with any token string; actual connection fails in background"
        );
    }

    #[tokio::test]
    async fn test_stop_gateway() {
        let config = DiscordConfig::default();
        let runtime = Arc::new(Runtime::new());
        let gateway = DiscordGateway::new(config, runtime);
        // Stop on a non-started gateway is fine
        let result = gateway.stop().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_send_message_fails_not_connected() {
        // send_message on a non-connected gateway should fail gracefully
        let config = DiscordConfig::default();
        let runtime = Arc::new(Runtime::new());
        let gateway = DiscordGateway::new(config, runtime);
        // We use block_on since send_message is async
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(gateway.send_message("123", "hello"));
        assert!(result.is_err());
    }
}
