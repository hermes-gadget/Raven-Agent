//! Small/local model support primitives.
//!
//! This module keeps small-model adaptation explicit and testable:
//! model profiles, bounded prompts, structured plan parsing, one-shot
//! tool-argument repair, context distillation, failure taxonomy, and
//! evidence-based verification helpers.

use crate::decomposer::{DecomposedPlan, Dependency};
use odin_core::traits::LoopState;
use odin_core::types::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Maximum tool-use complexity a profile should attempt before escalating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolComplexity {
    None,
    SingleSafeTool,
    MultipleSafeTools,
    RepoEditTools,
    ParallelSubAgents,
}

/// Prompt style that works best for a model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptStyle {
    MinimalSchema,
    ExplicitExamples,
    StepwiseBullets,
}

/// A concrete profile for a small/local/cheap model family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SmallModelProfile {
    pub id: String,
    pub provider_hint: String,
    pub model_family: String,
    pub context_tokens: u32,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub max_tool_complexity: ToolComplexity,
    pub prompt_style: PromptStyle,
    pub retry_limit: u32,
    pub escalation_rules: Vec<String>,
}

impl SmallModelProfile {
    /// Conservative default for OpenAI-compatible Ollama/local models.
    pub fn ollama_qwen_coder_7b() -> Self {
        Self {
            id: "ollama-qwen2.5-coder-7b".into(),
            provider_hint: "ollama/openai-compatible".into(),
            model_family: "Qwen Coder".into(),
            context_tokens: 32_768,
            strengths: vec![
                "repo edits".into(),
                "short code generation".into(),
                "structured JSON with examples".into(),
            ],
            weaknesses: vec![
                "long multi-file context".into(),
                "ambiguous tool schemas".into(),
                "deep debugging without decomposition".into(),
            ],
            max_tool_complexity: ToolComplexity::RepoEditTools,
            prompt_style: PromptStyle::MinimalSchema,
            retry_limit: 2,
            escalation_rules: vec![
                "escalate when the same tool argument error repeats".into(),
                "use decomposition for multi-file or debugging tasks".into(),
                "distill context when prompt exceeds 70% of the context window".into(),
            ],
        }
    }

    /// Cheap DeepSeek-compatible profile.
    pub fn deepseek_cheap() -> Self {
        Self {
            id: "deepseek-small-cheap".into(),
            provider_hint: "deepseek/openai-compatible".into(),
            model_family: "DeepSeek".into(),
            context_tokens: 64_000,
            strengths: vec![
                "code reasoning".into(),
                "debugging with evidence".into(),
                "low-cost verifier passes".into(),
            ],
            weaknesses: vec![
                "over-confident final answers".into(),
                "tool overuse on simple tasks".into(),
                "verbose plans unless bounded".into(),
            ],
            max_tool_complexity: ToolComplexity::RepoEditTools,
            prompt_style: PromptStyle::StepwiseBullets,
            retry_limit: 2,
            escalation_rules: vec![
                "require evidence before VERIFIED".into(),
                "escalate verifier after a failed test or missing evidence".into(),
                "keep simple single-file tasks direct".into(),
            ],
        }
    }

    /// Llama 8B-class local profile.
    pub fn llama_8b() -> Self {
        Self {
            id: "ollama-llama3.1-8b".into(),
            provider_hint: "ollama/openai-compatible".into(),
            model_family: "Llama".into(),
            context_tokens: 8_192,
            strengths: vec!["summaries".into(), "simple docs edits".into()],
            weaknesses: vec![
                "strict tool JSON".into(),
                "long context".into(),
                "multi-step repo changes".into(),
            ],
            max_tool_complexity: ToolComplexity::SingleSafeTool,
            prompt_style: PromptStyle::ExplicitExamples,
            retry_limit: 1,
            escalation_rules: vec![
                "escalate for multi-file edits".into(),
                "escalate for shell or git tools".into(),
                "distill context aggressively".into(),
            ],
        }
    }

    /// Qwen 14B-class local profile.
    pub fn qwen_coder_14b() -> Self {
        Self {
            id: "ollama-qwen2.5-coder-14b".into(),
            provider_hint: "ollama/openai-compatible".into(),
            model_family: "Qwen Coder".into(),
            context_tokens: 32_768,
            strengths: vec![
                "multi-file code edits".into(),
                "schema following".into(),
                "tool-use planning".into(),
            ],
            weaknesses: vec!["large refactors without verifier evidence".into()],
            max_tool_complexity: ToolComplexity::RepoEditTools,
            prompt_style: PromptStyle::MinimalSchema,
            retry_limit: 2,
            escalation_rules: vec![
                "use verifier for test-backed claims".into(),
                "escalate after timeout or repeated missing context".into(),
            ],
        }
    }

    /// Built-in profiles Raven ships with.
    pub fn built_ins() -> Vec<Self> {
        vec![
            Self::ollama_qwen_coder_7b(),
            Self::deepseek_cheap(),
            Self::llama_8b(),
            Self::qwen_coder_14b(),
        ]
    }

    /// Resolve a built-in profile by ID.
    pub fn by_id(id: &str) -> Option<Self> {
        Self::built_ins()
            .into_iter()
            .find(|profile| profile.id == id)
    }

    /// Resolve a built-in profile from a configured model name when possible.
    pub fn for_model(model: &str) -> Option<Self> {
        let model = model.to_ascii_lowercase();
        if model.contains("deepseek") {
            Some(Self::deepseek_cheap())
        } else if model.contains("qwen") && model.contains("14b") {
            Some(Self::qwen_coder_14b())
        } else if model.contains("qwen") && (model.contains("7b") || model.contains("coder")) {
            Some(Self::ollama_qwen_coder_7b())
        } else if model.contains("llama") && model.contains("8b") {
            Some(Self::llama_8b())
        } else {
            None
        }
    }

    /// Compact system instruction for small models.
    pub fn system_instruction(&self) -> String {
        format!(
            "You are Raven running with profile {}. Be concise. Use strict JSON when requested. \
             Use tools only when needed. If evidence is missing, say what to check next.",
            self.id
        )
    }

    /// Strict planning prompt that is easier for small models than free-form prose.
    pub fn plan_prompt(&self) -> String {
        "Return only JSON using this schema: \
         {\"sub_tasks\":[{\"id\":\"task_1\",\"description\":\"short action\"}],\
         \"dependencies\":[{\"from\":\"task_1\",\"to\":\"task_2\",\"reason\":\"why\"}]}. \
         Use 2-5 concise sub_tasks. No markdown."
            .into()
    }

    /// Bounded action prompt with tool-choice hints.
    pub fn action_prompt(&self, pending: &str, goal: &str) -> String {
        format!(
            "Task: {pending}\nGoal: {goal}\nRespond with one short result, or one valid tool call. \
             Tool arguments must be a JSON object. Do not call tools speculatively."
        )
    }
}

/// Coarse task complexity used by the adaptive policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskComplexity {
    Simple,
    ToolUse,
    MultiFile,
    Debugging,
    LongContext,
    Unknown,
}

/// Execution mode selected for a task/profile/evidence combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    CheapDirect,
    Looped,
    Decompose,
    StrongerVerifier,
    EscalateModel,
}

/// Failure taxonomy used by evals, retry logic, and reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    ModelConfusion,
    BadToolArgs,
    MissingContext,
    PermissionDenied,
    Timeout,
    HallucinatedFile,
    HallucinatedTool,
    ProviderError,
    VerificationGap,
}

/// Adaptive policy for keeping easy work cheap while escalating hard cases.
#[derive(Debug, Clone, Default)]
pub struct AdaptiveExecutionPolicy;

impl AdaptiveExecutionPolicy {
    pub fn choose(
        &self,
        profile: &SmallModelProfile,
        complexity: TaskComplexity,
        failures: &[FailureKind],
        confidence: f64,
    ) -> ExecutionMode {
        if failures
            .iter()
            .any(|kind| matches!(kind, FailureKind::PermissionDenied | FailureKind::Timeout))
        {
            return ExecutionMode::EscalateModel;
        }

        if failures
            .iter()
            .any(|kind| matches!(kind, FailureKind::VerificationGap))
        {
            return ExecutionMode::StrongerVerifier;
        }

        let bad_tool_repeats = failures
            .iter()
            .filter(|kind| {
                matches!(
                    kind,
                    FailureKind::BadToolArgs | FailureKind::HallucinatedTool
                )
            })
            .count();
        if bad_tool_repeats > profile.retry_limit as usize {
            return ExecutionMode::EscalateModel;
        }

        match (complexity, profile.max_tool_complexity) {
            (TaskComplexity::Simple, _) if confidence >= 0.75 => ExecutionMode::CheapDirect,
            (TaskComplexity::LongContext, _) => ExecutionMode::Decompose,
            (TaskComplexity::MultiFile, ToolComplexity::SingleSafeTool | ToolComplexity::None) => {
                ExecutionMode::EscalateModel
            }
            (TaskComplexity::Debugging | TaskComplexity::MultiFile, _) => ExecutionMode::Decompose,
            (TaskComplexity::ToolUse, ToolComplexity::None) => ExecutionMode::EscalateModel,
            _ if confidence < 0.45 => ExecutionMode::Decompose,
            _ => ExecutionMode::Looped,
        }
    }
}

/// Distilled state for small context windows.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistilledContext {
    pub facts: Vec<String>,
    pub decisions: Vec<String>,
    pub files_changed: Vec<String>,
    pub errors: Vec<String>,
    pub next_action: Option<String>,
}

/// Convert loop state into a bounded fact/decision/error summary.
pub fn distill_context(state: &LoopState) -> DistilledContext {
    let mut distilled = DistilledContext {
        facts: vec![
            format!("goal: {}", state.task.goal),
            format!("phase: {}", state.current_phase),
            format!("iteration: {}", state.iteration),
        ],
        ..Default::default()
    };

    for sub_task in &state.task.sub_tasks {
        distilled.facts.push(format!(
            "sub_task {} {}: {}",
            sub_task.id, sub_task.status, sub_task.description
        ));
    }

    for record in &state.history {
        if matches!(
            record.phase,
            LoopPhase::Plan | LoopPhase::Decide | LoopPhase::Verify
        ) && let Some(output) = &record.output
        {
            distilled.decisions.push(output.clone());
        }
        if let Some(error) = &record.error {
            distilled.errors.push(error.clone());
        }
    }

    for result in &state.tool_results {
        if !result.success
            && let Some(error) = &result.error
        {
            distilled
                .errors
                .push(format!("{}: {}", result.tool_name, error));
        }

        if matches!(
            result.tool_name.as_str(),
            "file_write" | "file_delete" | "git"
        ) && !distilled.files_changed.contains(&result.tool_name)
        {
            distilled.files_changed.push(result.tool_name.clone());
        }
    }

    distilled.next_action = state
        .task
        .sub_tasks
        .iter()
        .find(|sub_task| sub_task.status == SubTaskStatus::Pending)
        .map(|sub_task| sub_task.description.clone());

    distilled
}

/// Evidence-based verification output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceCheck {
    pub verified: bool,
    pub confidence: f64,
    pub evidence: Vec<String>,
    pub missing: Vec<String>,
}

/// Verify against concrete tool/message evidence instead of self-confidence only.
pub fn verify_evidence(state: &LoopState, criteria: &[String]) -> EvidenceCheck {
    let mut evidence = Vec::new();
    let mut missing = Vec::new();

    let successful_tools: Vec<&ToolResult> = state
        .tool_results
        .iter()
        .filter(|result| result.success)
        .collect();
    if !successful_tools.is_empty() {
        evidence.push(format!(
            "{} successful tool result(s)",
            successful_tools.len()
        ));
    }

    let substantive_actions = state
        .history
        .iter()
        .filter(|record| record.phase == LoopPhase::Act)
        .filter_map(|record| record.output.as_deref())
        .filter(|output| is_substantive_action_output(output))
        .count();
    if substantive_actions > 0 {
        evidence.push(format!("{substantive_actions} substantive ACT result(s)"));
    }

    let all_text = state
        .messages
        .iter()
        .filter_map(|message| message.text())
        .collect::<Vec<_>>()
        .join("\n")
        .to_lowercase();

    for criterion in criteria {
        if all_text.contains(&criterion.to_lowercase()) {
            evidence.push(format!("criterion mentioned: {}", criterion));
        } else {
            missing.push(criterion.clone());
        }
    }

    if criteria.is_empty()
        && (successful_tools.is_empty()
            && substantive_actions == 0
            && !state
                .task
                .sub_tasks
                .iter()
                .any(|sub_task| sub_task.status == SubTaskStatus::Completed))
    {
        missing.push("no concrete tool result or completed sub-task".into());
    }

    let verified = missing.is_empty();
    let confidence = if verified {
        if evidence.is_empty() { 0.7 } else { 0.9 }
    } else if evidence.is_empty() {
        0.35
    } else {
        0.55
    };

    EvidenceCheck {
        verified,
        confidence,
        evidence,
        missing,
    }
}

/// Whether ACT produced a result rather than a planning/status/error placeholder.
pub(crate) fn is_substantive_action_output(output: &str) -> bool {
    let output = output.trim();
    if output.is_empty() {
        return false;
    }
    let lower = output.to_ascii_lowercase();
    !lower.starts_with("working on:")
        && lower != "executing next action"
        && !lower.starts_with("calling tool:")
        && !lower.starts_with("llm call failed:")
}

/// Classify a failed tool result into the small-model failure taxonomy.
pub fn classify_tool_failure(result: &ToolResult) -> Option<FailureKind> {
    if result.success {
        return None;
    }

    let text = result.error.as_deref().unwrap_or_default().to_lowercase();

    if text.contains("invalid tool args")
        || text.contains("missing required")
        || text.contains("arguments must")
    {
        Some(FailureKind::BadToolArgs)
    } else if text.contains("not found in registry") || text.contains("cannot run") {
        Some(FailureKind::HallucinatedTool)
    } else if text.contains("permission")
        || text.contains("denied")
        || text.contains("approval")
        || text.contains("path boundary")
    {
        Some(FailureKind::PermissionDenied)
    } else if text.contains("timeout") || text.contains("timed out") {
        Some(FailureKind::Timeout)
    } else if text.contains("no such file") || text.contains("not found") {
        Some(FailureKind::HallucinatedFile)
    } else {
        Some(FailureKind::ModelConfusion)
    }
}

/// Parse either strict JSON plans or a bullet fallback into a validated plan.
pub fn parse_plan_response(goal: &str, text: &str, max_sub_tasks: usize) -> Option<DecomposedPlan> {
    parse_json_plan(goal, text, max_sub_tasks)
        .or_else(|| parse_bullet_plan(goal, text, max_sub_tasks))
}

fn parse_json_plan(goal: &str, text: &str, max_sub_tasks: usize) -> Option<DecomposedPlan> {
    let candidate = extract_json_object(text)?;
    let value: Value = serde_json::from_str(&candidate).ok()?;
    let object = value.as_object()?;
    let tasks_value = object.get("sub_tasks").or_else(|| object.get("tasks"))?;
    let tasks = tasks_value.as_array()?;

    let mut sub_tasks = Vec::new();
    for (index, task) in tasks.iter().take(max_sub_tasks).enumerate() {
        let (id, description) = match task {
            Value::String(description) => (format!("task_{}", index + 1), description.clone()),
            Value::Object(task_object) => {
                let description = task_object
                    .get("description")
                    .or_else(|| task_object.get("task"))
                    .or_else(|| task_object.get("title"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                let id = task_object
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("task_{}", index + 1));
                (id, description)
            }
            _ => continue,
        };

        if description.is_empty() {
            continue;
        }

        sub_tasks.push(SubTask {
            id,
            description,
            status: SubTaskStatus::Pending,
            result: None,
        });
    }

    if sub_tasks.is_empty() {
        return None;
    }

    let dependencies = object
        .get("dependencies")
        .and_then(Value::as_array)
        .map(|dependencies| {
            dependencies
                .iter()
                .filter_map(|dependency| {
                    let dependency = dependency.as_object()?;
                    Some(Dependency {
                        from: dependency.get("from")?.as_str()?.to_string(),
                        to: dependency.get("to")?.as_str()?.to_string(),
                        reason: dependency
                            .get("reason")
                            .and_then(Value::as_str)
                            .unwrap_or("Sequential order")
                            .to_string(),
                    })
                })
                .collect::<Vec<_>>()
        })
        .filter(|dependencies| !dependencies.is_empty())
        .unwrap_or_else(|| sequential_dependencies(&sub_tasks));

    Some(DecomposedPlan {
        goal: goal.to_string(),
        sub_tasks,
        dependencies,
    })
}

fn parse_bullet_plan(goal: &str, text: &str, max_sub_tasks: usize) -> Option<DecomposedPlan> {
    let sub_tasks: Vec<SubTask> = text
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            trimmed
                .strip_prefix("- ")
                .or_else(|| trimmed.strip_prefix("* "))
                .or_else(|| trimmed.strip_prefix("• "))
                .map(str::trim)
        })
        .filter(|description| !description.is_empty())
        .take(max_sub_tasks)
        .enumerate()
        .map(|(index, description)| SubTask {
            id: format!("task_{}", index + 1),
            description: description.to_string(),
            status: SubTaskStatus::Pending,
            result: None,
        })
        .collect();

    if sub_tasks.is_empty() {
        return None;
    }

    Some(DecomposedPlan {
        goal: goal.to_string(),
        dependencies: sequential_dependencies(&sub_tasks),
        sub_tasks,
    })
}

fn sequential_dependencies(sub_tasks: &[SubTask]) -> Vec<Dependency> {
    sub_tasks
        .windows(2)
        .map(|pair| Dependency {
            from: pair[0].id.clone(),
            to: pair[1].id.clone(),
            reason: "Sequential order".into(),
        })
        .collect()
}

fn extract_json_object(text: &str) -> Option<String> {
    if let Some(start) = text.find("```") {
        let after_start = &text[start + 3..];
        let after_lang = after_start
            .strip_prefix("json")
            .unwrap_or(after_start)
            .trim_start_matches(['\n', '\r']);
        if let Some(end) = after_lang.find("```") {
            let fenced = after_lang[..end].trim();
            if fenced.starts_with('{') && fenced.ends_with('}') {
                return Some(fenced.to_string());
            }
        }
    }

    let start = text.find('{')?;
    let end = text.rfind('}')?;
    (end > start).then(|| text[start..=end].to_string())
}

/// Result of one-shot tool argument repair.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolArgRepair {
    pub args: Value,
    pub reason: String,
}

/// Repair malformed tool-call JSON once, using the tool schema as a bound.
pub fn repair_tool_arguments_once(
    tool_name: &str,
    raw: &str,
    schema: &ToolSchema,
) -> Option<ToolArgRepair> {
    let mut candidates = Vec::new();

    if let Some(json) = extract_json_object(raw) {
        candidates.push(json);
    }
    candidates.push(raw.trim().to_string());
    candidates.push(
        raw.trim()
            .trim_matches('`')
            .replace('\'', "\"")
            .replace(",}", "}")
            .replace(",]", "]"),
    );

    for candidate in candidates {
        if candidate.is_empty() {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<Value>(&candidate)
            && let Some(repair) = repair_tool_argument_value(tool_name, value, schema)
        {
            return Some(repair);
        }
    }

    let required = required_fields(schema);
    if required.len() == 1 {
        let mut object = Map::new();
        object.insert(required[0].clone(), Value::String(raw.trim().to_string()));
        return Some(ToolArgRepair {
            args: Value::Object(object),
            reason: format!("wrapped raw value as required field '{}'", required[0]),
        });
    }

    None
}

/// Repair parsed arguments that failed schema validation.
pub fn repair_tool_argument_value(
    tool_name: &str,
    args: Value,
    schema: &ToolSchema,
) -> Option<ToolArgRepair> {
    let required = required_fields(schema);
    if args.is_object() && required.iter().all(|field| args.get(field).is_some()) {
        return Some(ToolArgRepair {
            args,
            reason: "arguments already satisfy required fields".into(),
        });
    }

    if !args.is_object() {
        if required.len() == 1 {
            let mut object = Map::new();
            object.insert(required[0].clone(), args);
            return Some(ToolArgRepair {
                args: Value::Object(object),
                reason: format!("wrapped scalar as required field '{}'", required[0]),
            });
        }
        return None;
    }

    let mut object = args.as_object().cloned().unwrap_or_default();
    apply_aliases(tool_name, &mut object);

    if required.iter().all(|field| object.get(field).is_some()) {
        return Some(ToolArgRepair {
            args: Value::Object(object),
            reason: "mapped known argument aliases".into(),
        });
    }

    None
}

fn required_fields(schema: &ToolSchema) -> Vec<String> {
    schema
        .function
        .parameters
        .get("required")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn apply_aliases(tool_name: &str, object: &mut Map<String, Value>) {
    let aliases = [
        ("file", "path"),
        ("filename", "path"),
        ("filepath", "path"),
        ("text", "content"),
        ("body", "content"),
        ("cmd", "command"),
        ("q", "query"),
    ];

    for (alias, canonical) in aliases {
        if !object.contains_key(canonical)
            && let Some(value) = object.get(alias).cloned()
        {
            object.insert(canonical.to_string(), value);
        }
    }

    if tool_name == "file_write"
        && !object.contains_key("content")
        && let Some(value) = object.get("data").cloned()
    {
        object.insert("content".into(), value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema(name: &str, required: &[&str]) -> ToolSchema {
        ToolSchema {
            schema_type: "function".into(),
            function: FunctionSchema {
                name: name.into(),
                description: "test".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": required,
                }),
            },
        }
    }

    #[test]
    fn parses_strict_json_plan() {
        let plan = parse_plan_response(
            "fix bug",
            r#"{"sub_tasks":[{"id":"a","description":"reproduce"},{"id":"b","description":"fix"}]}"#,
            5,
        )
        .expect("plan");

        assert_eq!(plan.sub_tasks.len(), 2);
        assert_eq!(plan.sub_tasks[0].id, "a");
        assert_eq!(plan.dependencies.len(), 1);
    }

    #[test]
    fn parses_bullet_plan_fallback() {
        let plan = parse_plan_response("docs", "- read docs\n- update docs", 5).expect("plan");

        assert_eq!(plan.sub_tasks.len(), 2);
        assert_eq!(plan.sub_tasks[1].description, "update docs");
    }

    #[test]
    fn repairs_single_quoted_tool_json() {
        let repaired = repair_tool_arguments_once(
            "shell",
            "{'cmd':'cargo test',}",
            &schema("shell", &["command"]),
        )
        .expect("repair");

        assert_eq!(repaired.args["command"], "cargo test");
    }

    #[test]
    fn wraps_scalar_for_single_required_field() {
        let repaired = repair_tool_arguments_once(
            "web_search",
            "rust async",
            &schema("web_search", &["query"]),
        )
        .expect("repair");

        assert_eq!(repaired.args["query"], "rust async");
    }

    #[test]
    fn classifies_bad_tool_args() {
        let result = ToolResult {
            call_id: "1".into(),
            tool_name: "shell".into(),
            success: false,
            output: String::new(),
            error: Some("Invalid tool args: expected object".into()),
            duration_ms: 0,
            timestamp: chrono::Utc::now(),
        };

        assert_eq!(
            classify_tool_failure(&result),
            Some(FailureKind::BadToolArgs)
        );
    }
}
