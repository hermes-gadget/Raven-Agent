//! Individual loop phase implementations.
//!
//! Each phase is a self-contained function that transforms loop state, using
//! a configured provider when present and deterministic heuristics otherwise.

use async_trait::async_trait;
use odin_core::error::OdinResult;
use odin_core::traits::{
    AuditLogger, LoopState, PermissionEngine, PhaseRecord, PhaseResult, Provider, ToolContext,
};
use odin_core::types::*;
use odin_permissions::SecretRedactor;
use std::sync::Arc;

use crate::confidence::ConfidenceScorer;
use crate::decomposer::{DecomposedPlan, GoalDecomposer};
use crate::small_model::{
    SmallModelProfile, parse_plan_response, repair_tool_argument_value, repair_tool_arguments_once,
    verify_evidence,
};
use crate::summarizer::StateSummarizer;

// ── Phase Traits ────────────────────────────────────────────────────

/// A single phase of the agent loop.
#[async_trait]
pub trait Phase: Send + Sync {
    /// Execute this phase.
    async fn execute(
        &self,
        state: &mut LoopState,
        context: &PhaseContext,
    ) -> OdinResult<PhaseResult>;
}

/// Shared context passed to all phases.
pub struct PhaseContext {
    /// Stable principal for authorization, approvals, auditing, and rate limits.
    pub principal_id: AgentId,
    pub confidence_scorer: ConfidenceScorer,
    pub decomposer: GoalDecomposer,
    pub summarizer: StateSummarizer,
    pub plan: Option<DecomposedPlan>,
    /// Model name to pass to the provider (e.g., "deepseek-v4-pro")
    pub model_name: String,
    /// Optional cheaper/dedicated model for planning.
    pub planning_model_name: Option<String>,
    /// Optional model for critique, revision, and verification.
    pub critique_model_name: Option<String>,
    /// Optional LLM provider for real model calls
    pub provider: Option<Arc<dyn Provider>>,
    /// Optional stronger provider for escalation
    pub escalation_provider: Option<Arc<dyn Provider>>,
    /// Optional stronger model selected when escalation occurs.
    pub escalation_model_name: Option<String>,
    /// Optional tool registry for dispatching real tool calls
    pub tool_registry: Option<Arc<odin_tools::ToolRegistry>>,
    /// Optional policy engine for permission checking on tool calls
    pub policy_engine: Option<Arc<odin_permissions::PolicyEngine>>,
    /// Optional skill registry for loading and using markdown skills
    pub skill_registry: Option<Arc<odin_skills::SkillRegistry>>,
    /// Optional audit logger for recording tool calls and events
    pub audit_logger: Option<Arc<dyn AuditLogger>>,
    /// Optional persistent tracker for real tool attempts.
    pub reliability_tracker: Option<Arc<odin_tools::ReliabilityTracker>>,
    /// Optional small/local model profile for bounded prompts and retries
    pub model_profile: Option<SmallModelProfile>,
}

impl PhaseContext {
    fn planning_model(&self) -> &str {
        self.planning_model_name
            .as_deref()
            .unwrap_or(&self.model_name)
    }

    fn critique_model(&self) -> &str {
        self.critique_model_name
            .as_deref()
            .unwrap_or(&self.model_name)
    }
}

// ── Plan Phase ──────────────────────────────────────────────────────

pub struct PlanPhase {
    pub decomposer: Arc<GoalDecomposer>,
}

impl PlanPhase {
    pub fn new(decomposer: GoalDecomposer) -> Self {
        Self {
            decomposer: Arc::new(decomposer),
        }
    }
}

#[async_trait]
impl Phase for PlanPhase {
    async fn execute(
        &self,
        state: &mut LoopState,
        context: &PhaseContext,
    ) -> OdinResult<PhaseResult> {
        tracing::info!(task_id = %state.task.id, "Planning task");

        // Inject available skills into the system prompt if a registry is loaded
        if let Some(ref registry) = context.skill_registry {
            let skills = registry.enabled();
            if !skills.is_empty() {
                let mut skills_prompt = String::from(
                    "## Available Skills\n\nYou have access to the following reusable workflows (skills). When appropriate, use them:\n\n",
                );
                for skill in &skills {
                    let tools = if skill.required_tools.is_empty() {
                        "none".to_string()
                    } else {
                        skill.required_tools.join(", ")
                    };
                    skills_prompt.push_str(&format!(
                        "- **{}**: {}. Required tools: {}.\n",
                        skill.name, skill.description, tools,
                    ));
                }
                skills_prompt.push_str(
                    "\nTo use a skill, include [USE_SKILL: skill-name] in your response.\n",
                );

                // Append to the first system message
                if let Some(first) = state.messages.first_mut()
                    && first.role == Role::System
                    && let MessageContent::Text { content } = &mut first.content
                {
                    content.push_str("\n\n");
                    content.push_str(&skills_prompt);
                }
            }
        }

        // Try LLM-based decomposition if a provider is available
        let plan = if let Some(ref provider) = context.provider {
            let prompt = context.model_profile.as_ref().map_or_else(
                || {
                    "Break this goal into sub-tasks. Prefer JSON with \
                     {\"sub_tasks\":[{\"id\":\"task_1\",\"description\":\"short action\"}]}; \
                     if you cannot, list each sub-task as '- '. Keep descriptions concise and actionable."
                        .to_string()
                },
                SmallModelProfile::plan_prompt,
            );
            let mut msgs = state.messages.clone();
            msgs.push(Message::user(prompt));
            match provider
                .chat(
                    context.planning_model(),
                    &msgs,
                    &[],
                    &CompletionOptions::default(),
                )
                .await
            {
                Ok(response) => {
                    let text = response.message.text().unwrap_or("").to_string();
                    tracing::info!("[PLAN] LLM decomposition response received");
                    if let Some(plan) =
                        parse_plan_response(&state.task.goal, &text, self.decomposer.max_sub_tasks)
                    {
                        plan
                    } else {
                        tracing::warn!(
                            "[PLAN] LLM returned no parseable sub-tasks, falling back to heuristic"
                        );
                        self.decomposer.decompose_heuristic(&state.task.goal)
                    }
                }
                Err(_error) => {
                    tracing::warn!("Planning model call failed; using heuristic decomposition");
                    self.decomposer.decompose_heuristic(&state.task.goal)
                }
            }
        } else {
            tracing::info!("[PLAN] No provider available, using heuristic decomposition");
            self.decomposer.decompose_heuristic(&state.task.goal)
        };
        let task_count = plan.sub_tasks.len();

        // Update the task's sub-tasks
        state.task.sub_tasks = plan.sub_tasks.clone();

        // Add a plan message to the conversation
        let plan_text = format!(
            "I've decomposed the goal into {} sub-tasks:\n{}",
            task_count,
            plan.sub_tasks
                .iter()
                .map(|st| format!("- [{}] {}", st.id, st.description))
                .collect::<Vec<_>>()
                .join("\n")
        );
        state.messages.push(Message::assistant(plan_text.clone()));

        state.current_phase = LoopPhase::Plan;

        let record = PhaseRecord {
            phase: LoopPhase::Plan,
            input: Some(state.task.goal.clone()),
            output: Some(plan_text),
            confidence: Some(ConfidenceScore::new(0.9)),
            duration_ms: 0,
            error: None,
        };
        state.history.push(record);

        Ok(PhaseResult {
            phase: LoopPhase::Plan,
            decision: LoopDecision::Continue,
            output: Some(format!("Decomposed into {} sub-tasks", task_count)),
            confidence: ConfidenceScore::new(0.9),
            tool_results: vec![],
        })
    }
}

// ── Act Phase ───────────────────────────────────────────────────────

pub struct ActPhase;

#[async_trait]
impl Phase for ActPhase {
    async fn execute(
        &self,
        state: &mut LoopState,
        context: &PhaseContext,
    ) -> OdinResult<PhaseResult> {
        tracing::info!("[ACT] Executing action for iteration {}", state.iteration);

        state.current_phase = LoopPhase::Act;

        // If we have a real provider, call the LLM to decide what to do
        let (action_desc, tool_results) = if let Some(ref provider) = context.provider {
            let pending_desc = context
                .plan
                .as_ref()
                .and_then(|p| {
                    p.sub_tasks
                        .iter()
                        .find(|st| st.status == SubTaskStatus::Pending)
                })
                .map(|st| st.description.as_str())
                .unwrap_or("the current goal");

            let prompt = context.model_profile.as_ref().map_or_else(
                || {
                    format!(
                        "You are working on: {}\nGoal: {}\nDecide what action to take next. If you need a tool, specify which one. Otherwise, provide the result.",
                        pending_desc, state.task.goal
                    )
                },
                |profile| profile.action_prompt(pending_desc, &state.task.goal),
            );

            let mut msgs = state.messages.clone();
            msgs.push(Message::user(prompt));

            let options = CompletionOptions {
                temperature: Some(0.3),
                max_tokens: Some(1024),
                ..Default::default()
            };

            let tool_schemas: Vec<ToolSchema> = context
                .tool_registry
                .as_ref()
                .map(|r| r.list_schemas())
                .unwrap_or_default();

            match provider
                .chat(&context.model_name, &msgs, &tool_schemas, &options)
                .await
            {
                Ok(response) => {
                    let text = response.message.text().unwrap_or("").to_string();
                    let calls = response.message.tool_calls().to_vec();

                    if !calls.is_empty() {
                        let desc = format!("Calling tool: {}", calls[0].function.name);
                        state.messages.push(response.message);

                        // Actually dispatch tool calls via the tool registry
                        let tool_results: Vec<ToolResult> = if let Some(ref registry) =
                            context.tool_registry
                        {
                            let mut results = Vec::new();
                            for tc in calls {
                                let attempt_start = std::time::Instant::now();
                                let tool_name = tc.function.name.clone();
                                let requested_dry_run =
                                    !should_record_reliability(&tc.function.arguments, &[]);
                                let tool = match registry.get(&tool_name) {
                                    Some(t) => t,
                                    None => {
                                        if !requested_dry_run
                                            && let Some(tracker) = &context.reliability_tracker
                                        {
                                            tracker.record_outcome(
                                                &tool_name,
                                                odin_tools::ReliabilityOutcome::ValidationFailure,
                                                attempt_start.elapsed().as_millis() as u64,
                                            );
                                        }
                                        let tr = ToolResult {
                                            call_id: tc.id.clone(),
                                            tool_name: tool_name.clone(),
                                            success: false,
                                            output: String::new(),
                                            error: Some(format!(
                                                "Tool '{}' not found in registry",
                                                tool_name
                                            )),
                                            duration_ms: 0,
                                            timestamp: chrono::Utc::now(),
                                        };
                                        results.push(tr);
                                        continue;
                                    }
                                };
                                let is_dry_run = !should_record_reliability(
                                    &tc.function.arguments,
                                    tool.capability_tags(),
                                );

                                let schema = tool.schema();
                                let args: serde_json::Value = match serde_json::from_str(
                                    &tc.function.arguments,
                                ) {
                                    Ok(value) => match tool.validate_args(&value) {
                                        Ok(()) => value,
                                        Err(validation_error) => {
                                            match repair_tool_argument_value(
                                                &tool_name, value, &schema,
                                            ) {
                                                Some(repair)
                                                    if tool.validate_args(&repair.args).is_ok() =>
                                                {
                                                    tracing::info!(
                                                        tool = %tool_name,
                                                        reason = %repair.reason,
                                                        "[ACT] Repaired parsed tool arguments"
                                                    );
                                                    repair.args
                                                }
                                                _ => {
                                                    if !is_dry_run
                                                        && let Some(tracker) =
                                                            &context.reliability_tracker
                                                    {
                                                        tracker.record_outcome(
                                                                &tool_name,
                                                                odin_tools::ReliabilityOutcome::ValidationFailure,
                                                                attempt_start.elapsed().as_millis()
                                                                    as u64,
                                                            );
                                                    }
                                                    let tr = tool_arg_error_result(
                                                        &tc,
                                                        &tool_name,
                                                        format!(
                                                            "Invalid tool args: {}",
                                                            validation_error
                                                        ),
                                                    );
                                                    results.push(tr);
                                                    continue;
                                                }
                                            }
                                        }
                                    },
                                    Err(parse_error) => {
                                        match repair_tool_arguments_once(
                                            &tool_name,
                                            &tc.function.arguments,
                                            &schema,
                                        ) {
                                            Some(repair)
                                                if tool.validate_args(&repair.args).is_ok() =>
                                            {
                                                tracing::info!(
                                                    tool = %tool_name,
                                                    reason = %repair.reason,
                                                    "[ACT] Repaired malformed tool arguments"
                                                );
                                                repair.args
                                            }
                                            _ => {
                                                if !is_dry_run
                                                    && let Some(tracker) =
                                                        &context.reliability_tracker
                                                {
                                                    tracker.record_outcome(
                                                            &tool_name,
                                                            odin_tools::ReliabilityOutcome::ValidationFailure,
                                                            attempt_start.elapsed().as_millis()
                                                                as u64,
                                                        );
                                                }
                                                let tr = tool_arg_error_result(
                                                    &tc,
                                                    &tool_name,
                                                    format!("Invalid tool args: {}", parse_error),
                                                );
                                                results.push(tr);
                                                continue;
                                            }
                                        }
                                    }
                                };

                                let tool_context = ToolContext {
                                    agent_id: context.principal_id,
                                    session_id: state.task.id,
                                    working_dir: std::env::current_dir().unwrap_or_default(),
                                    env: std::collections::HashMap::new(),
                                };

                                // Enforce rate limits, explicit policy rules, approval gates,
                                // dangerous-command checks, and filesystem boundaries before
                                // calling any tool implementation.
                                let policy_error = if let Some(ref policy) = context.policy_engine {
                                    let mut error = None;
                                    let mut approval_granted = false;

                                    match policy
                                        .check_rate_limit(tool_context.agent_id, &tool_name)
                                        .await
                                    {
                                        Ok(true) => {}
                                        Ok(false) => {
                                            error = Some(format!(
                                                "Rate limit exceeded for tool '{tool_name}'"
                                            ));
                                        }
                                        Err(e) => {
                                            error = Some(format!("Rate-limit check failed: {e}"))
                                        }
                                    }

                                    let action = if error.is_none() {
                                        match policy
                                            .check_tool(tool_context.agent_id, &tool_name, &args)
                                            .await
                                        {
                                            Ok(action) => Some(action),
                                            Err(e) => {
                                                error =
                                                    Some(format!("Permission check failed: {e}"));
                                                None
                                            }
                                        }
                                    } else {
                                        None
                                    };

                                    if let Some(PermissionAction::Deny) = action {
                                        error = Some(format!(
                                            "Tool '{tool_name}' is denied by the permission policy"
                                        ));
                                    }

                                    let needs_approval =
                                        matches!(action, Some(PermissionAction::AskUser))
                                            || (matches!(action, Some(PermissionAction::Allow))
                                                && tool.requires_approval_for(&args)
                                                && policy.requires_approval());
                                    if error.is_none() && needs_approval {
                                        match policy
                                            .request_approval(
                                                tool_context.agent_id,
                                                &tool_name,
                                                &args.to_string(),
                                            )
                                            .await
                                        {
                                            Ok(true) => approval_granted = true,
                                            Ok(false) => {
                                                error = Some(format!(
                                                    "Approval required for tool '{tool_name}', but the action was not approved"
                                                ));
                                            }
                                            Err(e) => {
                                                error =
                                                    Some(format!("Approval request failed: {e}"));
                                            }
                                        }
                                    }

                                    if error.is_none()
                                        && tool_name == "shell"
                                        && !approval_granted
                                        && let Some(command) =
                                            args.get("command").and_then(|value| value.as_str())
                                        && matches!(
                                            policy
                                                .check_command(tool_context.agent_id, command)
                                                .await,
                                            Ok(PermissionAction::AskUser | PermissionAction::Deny)
                                        )
                                    {
                                        error = Some(
                                            "Dangerous shell command requires explicit approval"
                                                .to_string(),
                                        );
                                    }

                                    if error.is_none()
                                        && matches!(tool_name.as_str(), "file_read" | "file_write")
                                        && let Some(path) =
                                            args.get("path").and_then(|value| value.as_str())
                                        && let Err(e) = policy.check_path_boundary(
                                            std::path::Path::new(path),
                                            tool_name == "file_write",
                                        )
                                    {
                                        error = Some(e.to_string());
                                    }

                                    error
                                } else if tool.requires_approval_for(&args) {
                                    Some(format!(
                                        "Tool '{tool_name}' requires approval, but no permission engine is configured"
                                    ))
                                } else {
                                    None
                                };

                                if let Some(error) = policy_error {
                                    if !is_dry_run
                                        && let Some(tracker) = &context.reliability_tracker
                                    {
                                        tracker.record_outcome(
                                            &tool_name,
                                            odin_tools::ReliabilityOutcome::PolicyDenial,
                                            attempt_start.elapsed().as_millis() as u64,
                                        );
                                    }
                                    let tr = ToolResult {
                                        call_id: tc.id.clone(),
                                        tool_name: tool_name.clone(),
                                        success: false,
                                        output: String::new(),
                                        error: Some(error),
                                        duration_ms: 0,
                                        timestamp: chrono::Utc::now(),
                                    };
                                    results.push(tr);
                                    continue;
                                }

                                let start = std::time::Instant::now();
                                // Capture a redacted input summary before moving args into
                                // execute. Redact the complete value before truncating so a
                                // credential cannot be split across the summary boundary.
                                let redacted_args =
                                    SecretRedactor::full().redact(&args.to_string());
                                let input_summary: String =
                                    redacted_args.chars().take(200).collect();
                                let (mut tr, outcome) =
                                    match tool.execute(args, &tool_context).await {
                                        Ok(tr) => {
                                            let outcome = if tr.success {
                                                odin_tools::ReliabilityOutcome::Success
                                            } else {
                                                odin_tools::ReliabilityOutcome::ToolFailure
                                            };
                                            (tr, outcome)
                                        }
                                        Err(e) => {
                                            let outcome = odin_tools::classify_tool_error(&e);
                                            (
                                                ToolResult {
                                                    call_id: tc.id.clone(),
                                                    tool_name: tool_name.clone(),
                                                    success: false,
                                                    output: String::new(),
                                                    error: Some(e.to_string()),
                                                    duration_ms: start.elapsed().as_millis() as u64,
                                                    timestamp: chrono::Utc::now(),
                                                },
                                                outcome,
                                            )
                                        }
                                    };
                                if !is_dry_run && let Some(tracker) = &context.reliability_tracker {
                                    tracker.record_outcome(
                                        &tool_name,
                                        outcome,
                                        start.elapsed().as_millis() as u64,
                                    );
                                }

                                // Apply secret redaction before audit logging
                                let redactor = SecretRedactor::new();
                                tr.output = redactor.redact(&tr.output);
                                tr.error = tr.error.as_ref().map(|e| redactor.redact(e));

                                // Audit log the tool call
                                if let Some(ref audit_logger) = context.audit_logger {
                                    let details = serde_json::json!({
                                        "input_summary": input_summary,
                                        "result": if tr.success { "success" } else { "failure" },
                                        "duration_ms": tr.duration_ms,
                                        "permission_decision": "allowed",
                                    });
                                    let audit_entry = AuditEntry {
                                        id: uuid::Uuid::new_v4(),
                                        timestamp: chrono::Utc::now(),
                                        agent_id: uuid::Uuid::default(),
                                        session_id: uuid::Uuid::default(),
                                        event_type: AuditEventType::ToolCall,
                                        action: tool_name.clone(),
                                        details,
                                        result: if tr.success {
                                            AuditResult::Success
                                        } else {
                                            AuditResult::Failure
                                        },
                                    };
                                    let _ = audit_logger.log(audit_entry).await;
                                }

                                results.push(tr);
                            }
                            results
                        } else {
                            // A model requested tools that this engine cannot dispatch.
                            calls
                                .iter()
                                .map(|tc| ToolResult {
                                    call_id: tc.id.clone(),
                                    tool_name: tc.function.name.clone(),
                                    success: false,
                                    output: String::new(),
                                    error: Some(format!(
                                        "Tool '{}' cannot run because no tool registry is configured",
                                        tc.function.name
                                    )),
                                    duration_ms: 0,
                                    timestamp: chrono::Utc::now(),
                                })
                                .collect()
                        };

                        for tr in &tool_results {
                            state.tool_results.push(tr.clone());
                            state.messages.push(Message::tool_result(
                                &tr.call_id,
                                serde_json::to_string(tr).unwrap_or_default(),
                            ));
                        }

                        (desc, tool_results)
                    } else {
                        state.messages.push(Message::assistant(text.clone()));
                        (text, vec![])
                    }
                }
                Err(e) => {
                    let err_msg = format!("LLM call failed: {}", e);
                    state.messages.push(Message::assistant(err_msg.clone()));
                    (err_msg, vec![])
                }
            }
        } else {
            // Offline heuristic mode — no provider attached.
            let pending = context.plan.as_ref().and_then(|p| {
                p.sub_tasks
                    .iter()
                    .find(|st| st.status == SubTaskStatus::Pending)
            });

            let desc = match pending {
                Some(task) => format!("Working on: {}", task.description),
                None => "Executing next action".to_string(),
            };

            state.messages.push(Message::assistant(desc.clone()));
            (desc, vec![])
        };

        let record = PhaseRecord {
            phase: LoopPhase::Act,
            input: Some(format!("Iteration {}", state.iteration)),
            output: Some(action_desc.clone()),
            confidence: None,
            duration_ms: 0,
            error: None,
        };
        state.history.push(record);

        Ok(PhaseResult {
            phase: LoopPhase::Act,
            decision: LoopDecision::Continue,
            output: Some(action_desc),
            confidence: ConfidenceScore::new(0.7),
            tool_results,
        })
    }
}

// ── Inspect Phase ────────────────────────────────────────────────────

pub struct InspectPhase {
    pub summarizer: Arc<StateSummarizer>,
}

impl InspectPhase {
    pub fn new(summarizer: StateSummarizer) -> Self {
        Self {
            summarizer: Arc::new(summarizer),
        }
    }
}

#[async_trait]
impl Phase for InspectPhase {
    async fn execute(
        &self,
        state: &mut LoopState,
        context: &PhaseContext,
    ) -> OdinResult<PhaseResult> {
        tracing::info!("[INSPECT] Inspecting results of action");

        state.current_phase = LoopPhase::Inspect;

        // Check context window
        let context_limit = context
            .model_profile
            .as_ref()
            .map(|profile| profile.context_tokens)
            .unwrap_or(32_768);
        let needs_compression = self
            .summarizer
            .needs_compression(&state.messages, context_limit);

        if needs_compression {
            tracing::info!("[INSPECT] Context window nearing limit, compressing...");
            state.messages = self.summarizer.compress(&state.messages, 3);
        }

        // Validate last tool result if any
        let last_tool = state.tool_results.last();
        let inspection = match last_tool {
            Some(tr) if !tr.success => {
                format!(
                    "Tool '{}' failed: {}",
                    tr.tool_name,
                    tr.error.as_deref().unwrap_or("unknown error")
                )
            }
            Some(tr) => {
                format!("Tool '{}' succeeded in {}ms", tr.tool_name, tr.duration_ms)
            }
            None => "No tool results to inspect".to_string(),
        };

        let record = PhaseRecord {
            phase: LoopPhase::Inspect,
            input: Some("Inspect results".into()),
            output: Some(inspection.clone()),
            confidence: None,
            duration_ms: 0,
            error: None,
        };
        state.history.push(record);

        Ok(PhaseResult {
            phase: LoopPhase::Inspect,
            decision: LoopDecision::Continue,
            output: Some(inspection),
            confidence: ConfidenceScore::new(0.8),
            tool_results: vec![],
        })
    }
}

// ── Critique Phase ───────────────────────────────────────────────────

pub struct CritiquePhase {
    pub scorer: Arc<ConfidenceScorer>,
}

impl CritiquePhase {
    pub fn new(scorer: ConfidenceScorer) -> Self {
        Self {
            scorer: Arc::new(scorer),
        }
    }
}

#[async_trait]
impl Phase for CritiquePhase {
    async fn execute(
        &self,
        state: &mut LoopState,
        context: &PhaseContext,
    ) -> OdinResult<PhaseResult> {
        tracing::info!("[CRITIQUE] Self-evaluating action");

        state.current_phase = LoopPhase::Critique;

        // Score the last action — try LLM critique first if a provider is available
        let confidence = if let Some(ref provider) = context.provider {
            let last_action = state
                .messages
                .last()
                .and_then(|m| m.text())
                .unwrap_or("unknown action");
            let last_result = state
                .tool_results
                .last()
                .map(|tr| {
                    format!(
                        "Tool: {}, success: {}, output: {}, error: {}",
                        tr.tool_name,
                        tr.success,
                        tr.output.len(),
                        tr.error.as_deref().unwrap_or("none")
                    )
                })
                .unwrap_or_else(|| "No tool result".to_string());

            let prompt = format!(
                "Evaluate the last action. Was it successful? What could be improved? \
                 Score your confidence 0.0-1.0.\n\
                 \nLast action: {}\nLast result: {}\nGoal: {}\n\n\
                 Respond with your analysis and then a confidence score.",
                last_action, last_result, state.task.goal
            );

            let mut msgs = state.messages.clone();
            msgs.push(Message::user(prompt));
            match provider
                .chat(
                    context.critique_model(),
                    &msgs,
                    &[],
                    &CompletionOptions::default(),
                )
                .await
            {
                Ok(response) => {
                    let text = response.message.text().unwrap_or("").to_string();
                    tracing::info!("[CRITIQUE] LLM critique received");

                    // Parse confidence from LLM response
                    let parsed = parse_confidence_from_text(&text);
                    match parsed {
                        Some(score) => ConfidenceScore::new(score),
                        None => {
                            tracing::warn!(
                                "[CRITIQUE] Could not parse confidence from LLM, using heuristics"
                            );
                            fallback_critique_confidence(state, &self.scorer)
                        }
                    }
                }
                Err(_error) => {
                    tracing::warn!("Critique model call failed; using heuristic scoring");
                    fallback_critique_confidence(state, &self.scorer)
                }
            }
        } else {
            tracing::info!("[CRITIQUE] No provider available, using heuristic scoring");
            fallback_critique_confidence(state, &self.scorer)
        };

        let retry_limit = context
            .model_profile
            .as_ref()
            .map(|profile| profile.retry_limit)
            .unwrap_or(2);

        let decision = if confidence.is_high() {
            LoopDecision::Continue
        } else if confidence.is_low() && state.retry_count >= retry_limit {
            LoopDecision::Escalate
        } else if confidence.is_low() {
            LoopDecision::Retry
        } else {
            LoopDecision::Continue
        };

        let critique = format!(
            "Confidence: {:.0}% → Decision: {:?}",
            confidence.value() * 100.0,
            decision
        );

        // Update the last history record's confidence
        if let Some(last_record) = state.history.last_mut() {
            last_record.confidence = Some(confidence);
        }

        let record = PhaseRecord {
            phase: LoopPhase::Critique,
            input: Some("Score last action".into()),
            output: Some(critique.clone()),
            confidence: Some(confidence),
            duration_ms: 0,
            error: None,
        };
        state.history.push(record);

        Ok(PhaseResult {
            phase: LoopPhase::Critique,
            decision,
            output: Some(critique),
            confidence,
            tool_results: vec![],
        })
    }
}

// ── Revise Phase ─────────────────────────────────────────────────────

pub struct RevisePhase;

#[async_trait]
impl Phase for RevisePhase {
    async fn execute(
        &self,
        state: &mut LoopState,
        context: &PhaseContext,
    ) -> OdinResult<PhaseResult> {
        tracing::info!(
            "[REVISE] Revising approach (retry count: {})",
            state.retry_count
        );

        state.current_phase = LoopPhase::Revise;
        state.retry_count += 1;

        // Try LLM-based revision if a provider is available
        let strategy = if let Some(ref provider) = context.provider {
            let last_error = state
                .tool_results
                .last()
                .and_then(|tr| tr.error.as_deref())
                .unwrap_or("unknown");
            let last_msg = state.messages.last().and_then(|m| m.text()).unwrap_or("");

            let prompt = format!(
                "The last attempt was not fully successful. Suggest a revised approach.\n\n\
                 Goal: {}\n\
                 Last message: {}\n\
                 Last error: {}\n\
                 Attempt number: {}\n\n\
                 Suggest what to do differently. Be specific and actionable.",
                state.task.goal, last_msg, last_error, state.retry_count
            );

            let mut msgs = state.messages.clone();
            msgs.push(Message::user(prompt));
            match provider
                .chat(
                    context.critique_model(),
                    &msgs,
                    &[],
                    &CompletionOptions::default(),
                )
                .await
            {
                Ok(response) => {
                    let text = response.message.text().unwrap_or("").to_string();
                    tracing::info!("[REVISE] LLM revision suggestion received");
                    text
                }
                Err(_error) => {
                    tracing::warn!("Revision model call failed; using heuristic strategy");
                    fallback_revise_strategy(state.retry_count)
                }
            }
        } else {
            tracing::info!("[REVISE] No provider available, using heuristic strategy");
            fallback_revise_strategy(state.retry_count)
        };

        state.messages.push(Message::system(format!(
            "[REVISE] {} (attempt {})",
            strategy, state.retry_count
        )));

        let retry_limit = context
            .model_profile
            .as_ref()
            .map(|profile| profile.retry_limit)
            .unwrap_or(3);

        let decision = if state.retry_count >= retry_limit {
            LoopDecision::Escalate
        } else {
            LoopDecision::Retry
        };

        let record = PhaseRecord {
            phase: LoopPhase::Revise,
            input: Some(format!("Retry attempt {}", state.retry_count)),
            output: Some(strategy.to_string()),
            confidence: None,
            duration_ms: 0,
            error: None,
        };
        state.history.push(record);

        Ok(PhaseResult {
            phase: LoopPhase::Revise,
            decision,
            output: Some(strategy.to_string()),
            confidence: ConfidenceScore::new(0.5),
            tool_results: vec![],
        })
    }
}

// ── Verify Phase ─────────────────────────────────────────────────────

pub struct VerifyPhase {
    pub scorer: Arc<ConfidenceScorer>,
}

impl VerifyPhase {
    pub fn new(scorer: ConfidenceScorer) -> Self {
        Self {
            scorer: Arc::new(scorer),
        }
    }
}

#[async_trait]
impl Phase for VerifyPhase {
    async fn execute(
        &self,
        state: &mut LoopState,
        context: &PhaseContext,
    ) -> OdinResult<PhaseResult> {
        tracing::info!("[VERIFY] Verifying results");

        state.current_phase = LoopPhase::Verify;

        // Check if the current sub-task succeeded
        let current_task = context.plan.as_ref().and_then(|p| {
            p.sub_tasks
                .iter()
                .find(|st| st.status == SubTaskStatus::Pending)
        });

        // Try LLM-based verification if a provider is available
        let (mut verification, mut confidence) = if let Some(ref provider) = context.provider {
            let criteria_summary = if state.task.success_criteria.is_empty() {
                "No specific success criteria defined.".to_string()
            } else {
                format!(
                    "Success criteria:\n{}",
                    state
                        .task
                        .success_criteria
                        .iter()
                        .map(|c| format!("- {}", c))
                        .collect::<Vec<_>>()
                        .join("\n")
                )
            };

            let task_desc = match current_task {
                Some(task) => format!("Current sub-task: {}", task.description),
                None => "All sub-tasks appear complete".to_string(),
            };

            let prompt = format!(
                "Has the goal been achieved? Check against these success criteria and the conversation history.\n\n\
                     Goal: {}\n\
                     {}\n\
                     {}\n\n\
                     Respond with your analysis and a clear conclusion: 'VERIFIED' or 'NOT VERIFIED'.\n\
                     Also provide a confidence score 0.0-1.0.",
                state.task.goal, criteria_summary, task_desc
            );

            let mut msgs = state.messages.clone();
            msgs.push(Message::user(prompt));
            match provider
                .chat(
                    context.critique_model(),
                    &msgs,
                    &[],
                    &CompletionOptions::default(),
                )
                .await
            {
                Ok(response) => {
                    let text = response.message.text().unwrap_or("").to_string();
                    tracing::info!("[VERIFY] LLM verification received");

                    // Determine verification status from LLM response
                    let lower = text.to_lowercase();
                    let not_verified =
                        lower.contains("not verified") || lower.contains("not_verified");
                    let verified = lower.contains("verified") && !not_verified;

                    // A negative verification conclusion must not be promoted by
                    // an independently reported numeric score.
                    let conf = if not_verified {
                        ConfidenceScore::new(0.0)
                    } else {
                        parse_confidence_from_text(&text)
                            .map(ConfidenceScore::new)
                            .unwrap_or_else(|| {
                                if verified {
                                    ConfidenceScore::new(0.85)
                                } else {
                                    ConfidenceScore::new(0.5)
                                }
                            })
                    };

                    (text, conf)
                }
                Err(_error) => {
                    tracing::warn!("Verification model call failed; using heuristic verification");
                    // Fall through to heuristic
                    let verification = match current_task {
                        Some(task) => {
                            format!("Verifying completion of: {}", task.description)
                        }
                        None => "All sub-tasks appear complete".to_string(),
                    };
                    let all_criteria_met = state.task.success_criteria.is_empty()
                        || state.task.success_criteria.iter().all(|c| {
                            state
                                .messages
                                .iter()
                                .any(|m| m.text().map(|t| t.contains(c.as_str())).unwrap_or(false))
                        });
                    let conf = if all_criteria_met {
                        ConfidenceScore::new(0.9)
                    } else {
                        ConfidenceScore::new(0.6)
                    };
                    (verification, conf)
                }
            }
        } else {
            tracing::info!("[VERIFY] No provider available, using heuristic verification");
            let verification = match current_task {
                Some(task) => {
                    format!("Verifying completion of: {}", task.description)
                }
                None => "All sub-tasks appear complete".to_string(),
            };
            let all_criteria_met = state.task.success_criteria.is_empty()
                || state.task.success_criteria.iter().all(|c| {
                    state
                        .messages
                        .iter()
                        .any(|m| m.text().map(|t| t.contains(c.as_str())).unwrap_or(false))
                });
            let conf = if all_criteria_met {
                ConfidenceScore::new(0.9)
            } else {
                ConfidenceScore::new(0.6)
            };
            (verification, conf)
        };

        let evidence = verify_evidence(state, &state.task.success_criteria);
        if !evidence.verified {
            confidence = ConfidenceScore::new(confidence.value().min(evidence.confidence));
            verification.push_str(&format!(
                "\nEvidence missing: {}",
                evidence.missing.join(", ")
            ));
        } else if !evidence.evidence.is_empty() {
            confidence = ConfidenceScore::new(confidence.value().min(evidence.confidence));
            verification.push_str(&format!(
                "\nEvidence checked: {}",
                evidence.evidence.join(", ")
            ));
        }

        let record = PhaseRecord {
            phase: LoopPhase::Verify,
            input: Some("Verify results".into()),
            output: Some(verification.clone()),
            confidence: Some(confidence),
            duration_ms: 0,
            error: None,
        };
        state.history.push(record);

        Ok(PhaseResult {
            phase: LoopPhase::Verify,
            decision: LoopDecision::Continue,
            output: Some(verification),
            confidence,
            tool_results: vec![],
        })
    }
}

// ── Decide Phase ─────────────────────────────────────────────────────

pub struct DecidePhase;

#[async_trait]
impl Phase for DecidePhase {
    async fn execute(
        &self,
        state: &mut LoopState,
        context: &PhaseContext,
    ) -> OdinResult<PhaseResult> {
        tracing::info!("[DECIDE] Deciding whether to continue or stop");

        state.current_phase = LoopPhase::Decide;

        // Check if we've hit max iterations
        if state.iteration >= state.task.max_iterations {
            return Ok(PhaseResult {
                phase: LoopPhase::Decide,
                decision: LoopDecision::Stop,
                output: Some("Max iterations reached".into()),
                confidence: ConfidenceScore::new(0.5),
                tool_results: vec![],
            });
        }

        // Check if all sub-tasks are complete
        let all_done = context
            .plan
            .as_ref()
            .map(|p| {
                p.sub_tasks.iter().all(|st| {
                    st.status == SubTaskStatus::Completed
                        || st.status == SubTaskStatus::Skipped
                        || st.status == SubTaskStatus::Failed
                })
            })
            .unwrap_or(false);

        if all_done {
            return Ok(PhaseResult {
                phase: LoopPhase::Decide,
                decision: LoopDecision::Stop,
                output: Some("All sub-tasks complete".into()),
                confidence: ConfidenceScore::new(0.95),
                tool_results: vec![],
            });
        }

        // Check confidence from last critique
        let last_confidence = state
            .history
            .iter()
            .rev()
            .find(|r| r.confidence.is_some())
            .and_then(|r| r.confidence);

        let retry_limit = context
            .model_profile
            .as_ref()
            .map(|profile| profile.retry_limit)
            .unwrap_or(2);

        let decision = match last_confidence {
            Some(c) if c.is_low() && state.retry_count >= retry_limit => LoopDecision::Escalate,
            Some(c) if c.is_low() => LoopDecision::Retry,
            _ => LoopDecision::Continue,
        };

        let record = PhaseRecord {
            phase: LoopPhase::Decide,
            input: Some(format!(
                "Iteration {} / {}",
                state.iteration, state.task.max_iterations
            )),
            output: Some(format!("Decision: {:?}", decision)),
            confidence: last_confidence,
            duration_ms: 0,
            error: None,
        };
        state.history.push(record);

        Ok(PhaseResult {
            phase: LoopPhase::Decide,
            decision,
            output: Some(format!("{:?}", decision)),
            confidence: last_confidence.unwrap_or(ConfidenceScore::new(0.7)),
            tool_results: vec![],
        })
    }
}

// ── Helper Functions ──────────────────────────────────────────────────

fn tool_arg_error_result(tc: &ToolCall, tool_name: &str, error: String) -> ToolResult {
    ToolResult {
        call_id: tc.id.clone(),
        tool_name: tool_name.to_string(),
        success: false,
        output: String::new(),
        error: Some(error),
        duration_ms: 0,
        timestamp: chrono::Utc::now(),
    }
}

fn should_record_reliability(arguments: &str, capability_tags: &[&str]) -> bool {
    if capability_tags.contains(&"dry-run") {
        return false;
    }
    !serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|value| value.get("dry_run").and_then(|value| value.as_bool()))
        .unwrap_or(false)
}

/// Parse a confidence score (0.0–1.0) from LLM response text.
///
/// Looks for patterns like:
/// - "Confidence: 0.85"
/// - "Score: 92%"
/// - "confidence: 0.75"
/// - "0.9" near end of text
/// - "XX%" anywhere in text
fn parse_confidence_from_text(text: &str) -> Option<f64> {
    let lower = text.to_lowercase();

    // Pattern 1: Explicit "confidence: 0.X" or "score: 0.X"
    for prefix in &["confidence:", "score:", "confidence score:"] {
        if let Some(idx) = lower.find(prefix) {
            let rest = &lower[idx + prefix.len()..];
            // Try to find a decimal number after the colon
            if let Some(num_start) = rest.find(|c: char| c.is_ascii_digit() || c == '.') {
                let num_str: String = rest[num_start..]
                    .chars()
                    .take_while(|c| c.is_ascii_digit() || *c == '.')
                    .collect();
                if let Ok(val) = num_str.parse::<f64>()
                    && (0.0..=1.0).contains(&val)
                {
                    return Some(val);
                }
            }
        }
    }

    // Pattern 2: "XX%" anywhere in the text
    for (i, _) in lower.match_indices('%') {
        let start = (i.saturating_sub(4)..i)
            .rev()
            .find(|offset| lower.is_char_boundary(*offset))
            .unwrap_or(0);
        let before = &lower[start..i];
        if let Some(num_str) = trailing_ascii_number(before)
            && let Ok(val) = num_str.parse::<f64>()
            && (0.0..=100.0).contains(&val)
        {
            return Some(val / 100.0);
        }
    }

    // Pattern 3: Look for a decimal 0.X pattern near the end of text (last 200 chars)
    let tail = if text.chars().count() > 200 {
        text.chars()
            .rev()
            .take(200)
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>()
    } else {
        text.to_string()
    };
    for num_str in tail.split_whitespace() {
        // Strip trailing punctuation
        let cleaned = num_str.trim_end_matches(|c: char| c.is_ascii_punctuation());
        if let Ok(val) = cleaned.parse::<f64>()
            && (0.0..=1.0).contains(&val)
            && val > 0.0
        {
            return Some(val);
        }
    }

    None
}

fn trailing_ascii_number(value: &str) -> Option<&str> {
    let value = value.trim_end();
    let mut start = value.len();
    for (offset, character) in value.char_indices().rev() {
        if character.is_ascii_digit() || character == '.' {
            start = offset;
        } else {
            break;
        }
    }
    (start < value.len()).then(|| &value[start..])
}

/// Fallback heuristic confidence scoring for the CRITIQUE phase.
fn fallback_critique_confidence(state: &LoopState, scorer: &ConfidenceScorer) -> ConfidenceScore {
    if let Some(last_tool) = state.tool_results.last() {
        scorer.score_tool_result(
            last_tool.success,
            !last_tool.output.is_empty(),
            last_tool.error.is_some(),
            last_tool.duration_ms,
        )
    } else if let Some(last_msg) = state.messages.last() {
        scorer.score_text_response(last_msg.text().unwrap_or(""), Some(&state.task.goal))
    } else {
        ConfidenceScore::new(0.5)
    }
}

/// Fallback heuristic revision strategy string.
fn fallback_revise_strategy(retry_count: u32) -> String {
    match retry_count {
        1 => "Retrying with same parameters".to_string(),
        2 => "Retrying with adjusted parameters".to_string(),
        _ => "Escalating to stronger model".to_string(),
    }
}

#[cfg(test)]
mod reliability_tests {
    use super::should_record_reliability;

    #[test]
    fn dry_run_requests_are_excluded_from_reliability() {
        assert!(!should_record_reliability(r#"{"dry_run":true}"#, &[]));
        assert!(!should_record_reliability("{}", &["dry-run", "safe"]));
        assert!(should_record_reliability(r#"{"dry_run":false}"#, &[]));
        assert!(should_record_reliability("{}", &[]));
    }
}
