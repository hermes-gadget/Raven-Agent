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
    fn capability_tags(&self) -> &[&str] {
        &[]
    }

    /// Quick check for dangerous tools.
    fn is_dangerous(&self) -> bool {
        false
    }

    /// Validate arguments against the tool's JSON Schema.
    ///
    /// Default implementation checks that required fields from the schema
    /// are present in the arguments.
    fn validate_args(&self, args: &serde_json::Value) -> OdinResult<()> {
        let schema = self.schema();
        let params = &schema.function.parameters;

        // Must be an object
        if !args.is_object() {
            return Err(crate::error::OdinError::Validation(
                "Arguments must be a JSON object".into(),
            ));
        }

        // Check required fields
        if let Some(required) = params.get("required").and_then(|v| v.as_array()) {
            for field in required {
                if let Some(field_name) = field.as_str()
                    && args.get(field_name).is_none()
                {
                    return Err(crate::error::OdinError::Validation(format!(
                        "Missing required field '{}' for tool '{}'",
                        field_name, schema.function.name
                    )));
                }
            }
        }

        Ok(())
    }
}

/// Context passed to tools during execution.
#[derive(Debug, Clone)]
pub struct ToolContext {
    pub agent_id: AgentId,
    pub session_id: SessionId,
    pub working_dir: std::path::PathBuf,
    pub env: HashMap<String, String>,
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
