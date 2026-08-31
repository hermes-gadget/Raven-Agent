//! Core traits for Raven Agent.
//!
//! These traits define the contracts between crates, enabling
//! loose coupling and testability through mocking.

use crate::error::OdinResult;
use crate::types::*;
use async_trait::async_trait;
use std::collections::HashMap;

// ── Provider Trait ─────────────────────────────────────────────────

/// A model provider that can send chat completions.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Unique name for this provider (e.g., "openai", "anthropic", "ollama").
    fn name(&self) -> &str;

    /// List available models.
    async fn list_models(&self) -> OdinResult<Vec<ModelInfo>>;

    /// Send a chat completion request.
    async fn chat(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolSchema],
        options: &CompletionOptions,
    ) -> OdinResult<ChatResponse>;

    /// Stream a chat completion.
    async fn chat_stream(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolSchema],
        options: &CompletionOptions,
    ) -> OdinResult<Box<dyn ChatStream>>;

    /// Health check — is the provider reachable?
    async fn health_check(&self) -> OdinResult<bool>;
}

/// A streaming chat completion.
#[async_trait]
pub trait ChatStream: Send + Unpin {
    /// Get the next chunk from the stream.
    async fn next(&mut self) -> OdinResult<Option<ChatResponse>>;
}

// ── Tool Trait ──────────────────────────────────────────────────────

/// A tool that an agent can invoke.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Unique tool name.
    fn name(&self) -> &str;

    /// Human-readable description for the model.
    fn description(&self) -> &str;

    /// JSON Schema for the tool's parameters.
    fn schema(&self) -> ToolSchema;

    /// Execute the tool with the given arguments (JSON value).
    async fn execute(
        &self,
        args: serde_json::Value,
        context: &ToolContext,
    ) -> OdinResult<ToolResult>;

    /// Whether this tool requires user approval.
    fn requires_approval(&self) -> bool {
        false
    }

    /// Whether this specific invocation requires approval. Tools whose risk
    /// depends on their arguments can override this while retaining a
    /// conservative static classification for catalogs and diagnostics.
    fn requires_approval_for(&self, _args: &serde_json::Value) -> bool {
        self.requires_approval()
    }

    /// Whether this tool is safe to run without sandboxing.
    fn is_safe(&self) -> bool {
        true
    }

    /// Capability tags for this tool (e.g., ["filesystem", "read", "safe"]).
    ///
    /// Tags are returned as owned strings so adapters can expose dynamic
    /// metadata without leaking process-lifetime allocations to manufacture a
    /// `&'static str` slice.
    fn capability_tags(&self) -> Vec<String> {
        Vec::new()
    }

    /// Quick check for dangerous tools.
    fn is_dangerous(&self) -> bool {
        false
    }

    /// Validate arguments against the tool's JSON Schema.
    ///
    /// The default implementation applies the shared JSON Schema validator so
    /// every execution surface enforces the same type, range, enum, nested,
    /// and additional-property constraints.
    fn validate_args(&self, args: &serde_json::Value) -> OdinResult<()> {
        let schema = self.schema();
        validate_json_schema(args, &schema.function.parameters).map_err(|error| {
            crate::error::OdinError::Validation(format!("tool '{}': {error}", schema.function.name))
        })
    }
}

/// Validate a JSON value against the subset of JSON Schema used by tool
/// contracts. The validator is deliberately centralized so callers cannot
/// bypass type, range, enum, nested-object, or additional-property checks by
/// choosing a different production surface.
pub fn validate_json_schema(
    value: &serde_json::Value,
    schema: &serde_json::Value,
) -> OdinResult<()> {
    validate_json_schema_at(value, schema, "$")
}

fn validate_json_schema_at(
    value: &serde_json::Value,
    schema: &serde_json::Value,
    path: &str,
) -> OdinResult<()> {
    if schema == &serde_json::Value::Bool(true) {
        return Ok(());
    }
    if schema == &serde_json::Value::Bool(false) {
        return Err(crate::error::OdinError::Validation(format!(
            "{path} is rejected by the schema"
        )));
    }
    let Some(schema) = schema.as_object() else {
        return Err(crate::error::OdinError::Validation(format!(
            "{path} has an invalid schema"
        )));
    };

    if let Some(expected) = schema.get("type") {
        let matches_type = expected.as_str().map_or_else(
            || {
                expected
                    .as_array()
                    .is_some_and(|types| types.iter().any(|kind| matches_json_type(value, kind)))
            },
            |kind| matches_json_type(value, &serde_json::Value::String(kind.to_string())),
        );
        if !matches_type {
            return Err(crate::error::OdinError::Validation(format!(
                "{path} has the wrong JSON type"
            )));
        }
    }

    if let Some(constant) = schema.get("const")
        && value != constant
    {
        return Err(crate::error::OdinError::Validation(format!(
            "{path} does not match const"
        )));
    }
    if let Some(values) = schema.get("enum").and_then(serde_json::Value::as_array)
        && !values.iter().any(|candidate| candidate == value)
    {
        return Err(crate::error::OdinError::Validation(format!(
            "{path} is not one of the allowed enum values"
        )));
    }

    for (keyword, requirement) in [("allOf", 1usize), ("anyOf", 1usize), ("oneOf", 1usize)] {
        if let Some(branches) = schema.get(keyword).and_then(serde_json::Value::as_array) {
            let matches = branches
                .iter()
                .filter(|branch| validate_json_schema_at(value, branch, path).is_ok())
                .count();
            let valid = match keyword {
                "allOf" => matches == branches.len(),
                "anyOf" => matches >= requirement,
                "oneOf" => matches == requirement,
                _ => false,
            };
            if !valid {
                return Err(crate::error::OdinError::Validation(format!(
                    "{path} does not satisfy {keyword}"
                )));
            }
        }
    }
    if let Some(not) = schema.get("not")
        && validate_json_schema_at(value, not, path).is_ok()
    {
        return Err(crate::error::OdinError::Validation(format!(
            "{path} must not match the schema"
        )));
    }

    if let Some(object) = value.as_object() {
        if let Some(minimum) = schema
            .get("minProperties")
            .and_then(serde_json::Value::as_u64)
            && object.len() < minimum as usize
        {
            return Err(crate::error::OdinError::Validation(format!(
                "{path} has too few properties"
            )));
        }
        if let Some(maximum) = schema
            .get("maxProperties")
            .and_then(serde_json::Value::as_u64)
            && object.len() > maximum as usize
        {
            return Err(crate::error::OdinError::Validation(format!(
                "{path} has too many properties"
            )));
        }
        if let Some(required) = schema.get("required").and_then(serde_json::Value::as_array) {
            for field in required.iter().filter_map(serde_json::Value::as_str) {
                if !object.contains_key(field) {
                    return Err(crate::error::OdinError::Validation(format!(
                        "{path}.{field} is required"
                    )));
                }
            }
        }
        let properties = schema
            .get("properties")
            .and_then(serde_json::Value::as_object);
        let additional_schema = schema.get("additionalProperties");
        for (name, child) in object {
            if let Some(property_schema) = properties.and_then(|properties| properties.get(name)) {
                validate_json_schema_at(child, property_schema, &format!("{path}.{name}"))?;
            } else {
                match additional_schema {
                    Some(serde_json::Value::Bool(false)) => {
                        return Err(crate::error::OdinError::Validation(format!(
                            "{path}.{name} is not an allowed property"
                        )));
                    }
                    Some(serde_json::Value::Bool(true)) | None => {}
                    Some(property_schema) => {
                        validate_json_schema_at(child, property_schema, &format!("{path}.{name}"))?;
                    }
                }
            }
        }
    }

    if let Some(array) = value.as_array() {
        if let Some(minimum) = schema.get("minItems").and_then(serde_json::Value::as_u64)
            && array.len() < minimum as usize
        {
            return Err(crate::error::OdinError::Validation(format!(
                "{path} has too few items"
            )));
        }
        if let Some(maximum) = schema.get("maxItems").and_then(serde_json::Value::as_u64)
            && array.len() > maximum as usize
        {
            return Err(crate::error::OdinError::Validation(format!(
                "{path} has too many items"
            )));
        }
        if schema
            .get("uniqueItems")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        {
            for (index, item) in array.iter().enumerate() {
                if array.iter().skip(index + 1).any(|other| other == item) {
                    return Err(crate::error::OdinError::Validation(format!(
                        "{path} contains duplicate items"
                    )));
                }
            }
        }
        if let Some(item_schema) = schema.get("items") {
            for (index, item) in array.iter().enumerate() {
                validate_json_schema_at(item, item_schema, &format!("{path}[{index}]"))?;
            }
        }
    }

    if let Some(string) = value.as_str() {
        if let Some(minimum) = schema.get("minLength").and_then(serde_json::Value::as_u64)
            && string.chars().count() < minimum as usize
        {
            return Err(crate::error::OdinError::Validation(format!(
                "{path} is shorter than minLength"
            )));
        }
        if let Some(maximum) = schema.get("maxLength").and_then(serde_json::Value::as_u64)
            && string.chars().count() > maximum as usize
        {
            return Err(crate::error::OdinError::Validation(format!(
                "{path} is longer than maxLength"
            )));
        }
        if let Some(pattern) = schema.get("pattern").and_then(serde_json::Value::as_str) {
            let regex = regex::Regex::new(pattern).map_err(|error| {
                crate::error::OdinError::Validation(format!(
                    "{path} uses an invalid schema pattern: {error}"
                ))
            })?;
            if !regex.is_match(string) {
                return Err(crate::error::OdinError::Validation(format!(
                    "{path} does not match the required pattern"
                )));
            }
        }
    }

    if let Some(number) = value.as_f64() {
        if let Some(minimum) = schema.get("minimum").and_then(serde_json::Value::as_f64)
            && number < minimum
        {
            return Err(crate::error::OdinError::Validation(format!(
                "{path} is below minimum"
            )));
        }
        if let Some(maximum) = schema.get("maximum").and_then(serde_json::Value::as_f64)
            && number > maximum
        {
            return Err(crate::error::OdinError::Validation(format!(
                "{path} is above maximum"
            )));
        }
        if schema
            .get("exclusiveMinimum")
            .and_then(serde_json::Value::as_f64)
            .is_some_and(|minimum| number <= minimum)
            || schema
                .get("exclusiveMaximum")
                .and_then(serde_json::Value::as_f64)
                .is_some_and(|maximum| number >= maximum)
        {
            return Err(crate::error::OdinError::Validation(format!(
                "{path} violates an exclusive numeric bound"
            )));
        }
    }

    Ok(())
}

fn matches_json_type(value: &serde_json::Value, expected: &serde_json::Value) -> bool {
    match expected.as_str() {
        Some("null") => value.is_null(),
        Some("boolean") => value.is_boolean(),
        Some("object") => value.is_object(),
        Some("array") => value.is_array(),
        Some("string") => value.is_string(),
        Some("integer") => value.as_i64().is_some() || value.as_u64().is_some(),
        Some("number") => value.is_number(),
        _ => false,
    }
}

/// Context passed to tools during execution.
#[derive(Debug, Clone)]
pub struct ToolContext {
    pub agent_id: AgentId,
    pub session_id: SessionId,
    pub working_dir: std::path::PathBuf,
    pub env: HashMap<String, String>,
    /// Central limits available to direct tool implementations as well as the
    /// loop-level enforcement wrapper.
    pub resource_budgets: crate::config::ResourceBudgetConfig,
}

// ── Memory Store Trait ──────────────────────────────────────────────

/// Persistent memory storage.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Store a memory entry.
    async fn store(&self, entry: MemoryEntry) -> OdinResult<()>;

    /// Retrieve a memory entry by ID.
    async fn get(&self, id: &str) -> OdinResult<Option<MemoryEntry>>;

    /// Search memory entries by semantic similarity.
    async fn search(&self, query: &str, limit: usize) -> OdinResult<Vec<MemoryEntry>>;

    /// List entries by category.
    async fn list_by_category(
        &self,
        category: MemoryCategory,
        limit: usize,
    ) -> OdinResult<Vec<MemoryEntry>>;

    /// Delete a memory entry.
    async fn delete(&self, id: &str) -> OdinResult<()>;

    /// Get the total number of entries.
    async fn count(&self) -> OdinResult<usize>;
}

// ── Skill Trait ─────────────────────────────────────────────────────

/// A skill — a reusable workflow or procedure.
#[async_trait]
pub trait Skill: Send + Sync {
    /// Unique skill name.
    fn name(&self) -> &str;

    /// Human-readable description.
    fn description(&self) -> &str;

    /// Load the skill content (markdown instructions).
    async fn load(&self) -> OdinResult<String>;

    /// List any required tools for this skill.
    fn required_tools(&self) -> Vec<String> {
        vec![]
    }

    /// List recommended (optional) tools for this skill.
    fn recommended_tools(&self) -> Vec<String> {
        vec![]
    }

    /// Whether this skill is enabled.
    fn enabled(&self) -> bool {
        true
    }
}

// ── Audit Logger Trait ──────────────────────────────────────────────

/// Audit trail logger.
#[async_trait]
pub trait AuditLogger: Send + Sync {
    /// Log an audit entry.
    async fn log(&self, entry: AuditEntry) -> OdinResult<()>;

    /// Query audit entries.
    async fn query(
        &self,
        agent_id: Option<AgentId>,
        session_id: Option<SessionId>,
        event_type: Option<AuditEventType>,
        limit: usize,
    ) -> OdinResult<Vec<AuditEntry>>;

    /// Get recent entries.
    async fn recent(&self, limit: usize) -> OdinResult<Vec<AuditEntry>>;
}

// ── Permission Engine Trait ─────────────────────────────────────────

/// Safety permission engine.
#[async_trait]
pub trait PermissionEngine: Send + Sync {
    /// Check if a tool call is allowed.
    async fn check_tool(
        &self,
        agent_id: AgentId,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> OdinResult<PermissionAction>;

    /// Check if a shell command is allowed.
    async fn check_command(&self, agent_id: AgentId, command: &str)
    -> OdinResult<PermissionAction>;

    /// Check rate limits.
    async fn check_rate_limit(&self, agent_id: AgentId, tool_name: &str) -> OdinResult<bool>;

    /// Request user approval for an action (returns true if approved).
    async fn request_approval(
        &self,
        agent_id: AgentId,
        action: &str,
        details: &str,
    ) -> OdinResult<bool>;
}

// ── Loop Engine Trait ───────────────────────────────────────────────

/// The agent loop engine — the core innovation.
#[async_trait]
pub trait LoopEngine: Send + Sync {
    /// Execute a task through the full plan→act→inspect→critique→revise→verify loop.
    async fn execute_task(&self, task: &AgentTask) -> OdinResult<TaskResult>;

    /// Execute a single phase of the loop (for fine-grained control).
    async fn execute_phase(
        &self,
        phase: LoopPhase,
        state: &mut LoopState,
    ) -> OdinResult<PhaseResult>;

    /// Get the current state summary.
    fn state_summary(&self) -> StateSummary;

    /// Get the confidence score for the last action.
    fn confidence(&self) -> ConfidenceScore;
}

/// Mutable state carried through the loop phases.
#[derive(Debug, Clone)]
pub struct LoopState {
    pub task: AgentTask,
    pub messages: Vec<Message>,
    pub tool_results: Vec<ToolResult>,
    pub current_phase: LoopPhase,
    pub iteration: u32,
    pub retry_count: u32,
    pub history: Vec<PhaseRecord>,
}

/// Record of a single phase execution.
#[derive(Debug, Clone)]
pub struct PhaseRecord {
    pub phase: LoopPhase,
    pub input: Option<String>,
    pub output: Option<String>,
    pub confidence: Option<ConfidenceScore>,
    pub duration_ms: u64,
    pub error: Option<String>,
}

/// Result of executing a single phase.
#[derive(Debug, Clone)]
pub struct PhaseResult {
    pub phase: LoopPhase,
    pub decision: LoopDecision,
    pub output: Option<String>,
    pub confidence: ConfidenceScore,
    pub tool_results: Vec<ToolResult>,
}

#[cfg(test)]
mod tests {
    use super::validate_json_schema;

    #[test]
    fn shared_schema_validator_enforces_nested_types_ranges_and_properties() {
        let schema = serde_json::json!({
            "type": "object",
            "required": ["name", "settings"],
            "additionalProperties": false,
            "properties": {
                "name": {"type": "string", "minLength": 2, "pattern": "^[a-z]+$"},
                "settings": {
                    "type": "object",
                    "required": ["retries"],
                    "additionalProperties": false,
                    "properties": {
                        "retries": {"type": "integer", "minimum": 0, "maximum": 3}
                    }
                },
                "tags": {
                    "type": "array",
                    "maxItems": 2,
                    "uniqueItems": true,
                    "items": {"type": "string"}
                }
            }
        });

        assert!(
            validate_json_schema(
                &serde_json::json!({
                    "name": "raven",
                    "settings": {"retries": 2},
                    "tags": ["one", "two"]
                }),
                &schema
            )
            .is_ok()
        );
        assert!(
            validate_json_schema(
                &serde_json::json!({"name": "Raven", "settings": {"retries": 2}}),
                &schema
            )
            .is_err()
        );
        assert!(
            validate_json_schema(
                &serde_json::json!({"name": "raven", "settings": {"retries": 4}}),
                &schema
            )
            .is_err()
        );
        assert!(
            validate_json_schema(
                &serde_json::json!({
                    "name": "raven",
                    "settings": {"retries": 2},
                    "unexpected": true
                }),
                &schema
            )
            .is_err()
        );
        assert!(
            validate_json_schema(
                &serde_json::json!({
                    "name": "raven",
                    "settings": {"retries": 2},
                    "tags": ["same", "same"]
                }),
                &schema
            )
            .is_err()
        );
    }

    #[test]
    fn shared_schema_validator_supports_boolean_and_combinator_schemas() {
        assert!(
            validate_json_schema(&serde_json::json!("anything"), &serde_json::json!(true)).is_ok()
        );
        assert!(
            validate_json_schema(&serde_json::json!("anything"), &serde_json::json!(false))
                .is_err()
        );

        let schema = serde_json::json!({
            "anyOf": [
                {"type": "string", "const": "safe"},
                {"type": "integer", "minimum": 10}
            ]
        });
        assert!(validate_json_schema(&serde_json::json!("safe"), &schema).is_ok());
        assert!(validate_json_schema(&serde_json::json!(12), &schema).is_ok());
        assert!(validate_json_schema(&serde_json::json!(3), &schema).is_err());
    }

    #[test]
    fn shared_schema_validator_applies_schemas_to_additional_properties() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"known": {"type": "string"}},
            "additionalProperties": {"type": "integer", "minimum": 0}
        });

        assert!(
            validate_json_schema(&serde_json::json!({"known": "ok", "count": 2}), &schema).is_ok()
        );
        assert!(
            validate_json_schema(&serde_json::json!({"known": "ok", "count": -1}), &schema)
                .is_err()
        );
        assert!(
            validate_json_schema(&serde_json::json!({"known": "ok", "count": "two"}), &schema)
                .is_err()
        );
    }
}
