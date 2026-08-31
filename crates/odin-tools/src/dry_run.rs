//! Dry-run tool wrapper — intercepts tool execution for safe testing.
//!
//! [`DryRunTool`] wraps any [`Tool`] and replaces its `execute()` method
//! with a mock that validates arguments and returns a synthetic result
//! without performing any real side effects. Useful for:
//! - CI validation of dangerous tools
//! - Agent tool-selection testing
//! - Schema/argument validation without real execution

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use serde::{Deserialize, Serialize};

use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::{Tool, ToolContext};
use odin_core::types::{ToolResult, ToolSchema};

/// Configuration for a dry-run tool's mock behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DryRunConfig {
    /// If true, validate args against schema before returning mock result.
    #[serde(default = "default_true")]
    pub validate_args: bool,
    /// Mock output string to return on successful execution.
    #[serde(default = "default_mock_output")]
    pub mock_output: String,
    /// Mock duration in milliseconds.
    #[serde(default = "default_mock_duration")]
    pub mock_duration_ms: u64,
    /// If true, simulate a successful result. If false, simulate failure.
    #[serde(default = "default_true")]
    pub mock_success: bool,
    /// Mock error message (only used when mock_success is false).
    #[serde(default)]
    pub mock_error: Option<String>,
}

fn default_true() -> bool {
    true
}

fn default_mock_output() -> String {
    "[DRY RUN] Tool execution was intercepted by DryRunTool — no side effects performed.".into()
}

fn default_mock_duration() -> u64 {
    0
}

impl Default for DryRunConfig {
    fn default() -> Self {
        Self {
            validate_args: true,
            mock_output: default_mock_output(),
            mock_duration_ms: default_mock_duration(),
            mock_success: true,
            mock_error: None,
        }
    }
}

/// A tool wrapper that intercepts execution for safe testing.
///
/// Wraps an inner tool and replaces `execute()` with a mock that:
/// 1. Optionally validates arguments against the inner tool's schema
/// 2. Returns a synthetic [`ToolResult`] with a dry-run marker
///
/// All other trait methods (`name()`, `description()`, etc.) are
/// forwarded to the inner tool, so the dry-run wrapper is transparent
/// to tool selection and catalog.
pub struct DryRunTool {
    inner: Arc<dyn Tool>,
    config: DryRunConfig,
}

impl DryRunTool {
    /// Create a new dry-run wrapper around an existing tool.
    pub fn new(inner: Arc<dyn Tool>) -> Self {
        Self {
            inner,
            config: DryRunConfig::default(),
        }
    }

    /// Create with custom dry-run configuration.
    pub fn with_config(inner: Arc<dyn Tool>, config: DryRunConfig) -> Self {
        Self { inner, config }
    }

    /// Get the inner tool (without the wrapper).
    pub fn inner_tool(&self) -> &Arc<dyn Tool> {
        &self.inner
    }
}

#[async_trait]
impl Tool for DryRunTool {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn schema(&self) -> ToolSchema {
        self.inner.schema()
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _context: &ToolContext,
    ) -> OdinResult<ToolResult> {
        // Validate args if configured
        if self.config.validate_args {
            self.inner.validate_args(&args).map_err(|e| {
                OdinError::Validation(format!(
                    "[DRY RUN] Argument validation failed for tool '{}': {e}",
                    self.inner.name()
                ))
            })?;
        }

        let call_id = uuid::Uuid::new_v4().to_string();
        let tool_name = self.inner.name().to_string();

        if self.config.mock_success {
            Ok(ToolResult {
                call_id,
                tool_name,
                success: true,
                output: self.config.mock_output.clone(),
                error: None,
                duration_ms: self.config.mock_duration_ms,
                timestamp: Utc::now(),
            })
        } else {
            Ok(ToolResult {
                call_id,
                tool_name,
                success: false,
                output: String::new(),
                error: self
                    .config
                    .mock_error
                    .clone()
                    .or_else(|| Some("[DRY RUN] Simulated failure".into())),
                duration_ms: self.config.mock_duration_ms,
                timestamp: Utc::now(),
            })
        }
    }

    fn requires_approval(&self) -> bool {
        // Dry-run tools never need real approval — they're safe
        false
    }

    fn is_safe(&self) -> bool {
        // Dry-run tools are always safe
        true
    }

    fn capability_tags(&self) -> Vec<String> {
        vec!["dry-run".into(), "safe".into()]
    }

    fn is_dangerous(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::shell::Shell;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_dry_run_shell_safe_command() {
        let shell = Arc::new(Shell::new());
        let dry_run = DryRunTool::new(shell.clone());
        let ctx = ToolContext {
            agent_id: uuid::Uuid::new_v4(),
            session_id: uuid::Uuid::new_v4(),
            working_dir: std::path::PathBuf::from("/tmp"),
            env: HashMap::new(),
            resource_budgets: Default::default(),
        };

        let result = dry_run
            .execute(serde_json::json!({"command": "echo hello"}), &ctx)
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("DRY RUN"));
        assert_eq!(result.duration_ms, 0);

        // DryRunTool should be safe regardless of inner tool
        assert!(dry_run.is_safe());
        assert!(!dry_run.is_dangerous());
        assert!(!dry_run.requires_approval());
    }

    #[tokio::test]
    async fn test_dry_run_shell_dangerous_command_validated() {
        let shell = Arc::new(Shell::new());
        let dry_run = DryRunTool::new(shell.clone());
        let ctx = ToolContext {
            agent_id: uuid::Uuid::new_v4(),
            session_id: uuid::Uuid::new_v4(),
            working_dir: std::path::PathBuf::from("/tmp"),
            env: HashMap::new(),
            resource_budgets: Default::default(),
        };

        // A dangerous command (rm -rf) — should still work in dry-run
        // because dry-run only validates schema, not the command itself
        let result = dry_run
            .execute(serde_json::json!({"command": "rm -rf /tmp/test"}), &ctx)
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("DRY RUN"));
    }

    #[tokio::test]
    async fn test_dry_run_validation_failure() {
        let shell = Arc::new(Shell::new());
        let dry_run = DryRunTool::new(shell.clone());
        let ctx = ToolContext {
            agent_id: uuid::Uuid::new_v4(),
            session_id: uuid::Uuid::new_v4(),
            working_dir: std::path::PathBuf::from("/tmp"),
            env: HashMap::new(),
            resource_budgets: Default::default(),
        };

        // Missing required 'command' field
        let result = dry_run
            .execute(serde_json::json!({"wrong_field": "value"}), &ctx)
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("DRY RUN") || err.contains("Missing required field"),
            "Expected validation error, got: {err}"
        );
    }

    #[tokio::test]
    async fn test_dry_run_custom_config() {
        let shell = Arc::new(Shell::new());
        let config = DryRunConfig {
            validate_args: false,
            mock_output: "custom mock output".into(),
            mock_duration_ms: 42,
            mock_success: true,
            mock_error: None,
        };
        let dry_run = DryRunTool::with_config(shell.clone(), config);
        let ctx = ToolContext {
            agent_id: uuid::Uuid::new_v4(),
            session_id: uuid::Uuid::new_v4(),
            working_dir: std::path::PathBuf::from("/tmp"),
            env: HashMap::new(),
            resource_budgets: Default::default(),
        };

        let result = dry_run.execute(serde_json::json!({}), &ctx).await.unwrap();

        assert!(result.success);
        assert_eq!(result.output, "custom mock output");
        assert_eq!(result.duration_ms, 42);
    }

    #[tokio::test]
    async fn test_dry_run_simulated_failure() {
        let shell = Arc::new(Shell::new());
        let config = DryRunConfig {
            mock_success: false,
            mock_error: Some("Simulated network error".into()),
            ..Default::default()
        };
        let dry_run = DryRunTool::with_config(shell.clone(), config);
        let ctx = ToolContext {
            agent_id: uuid::Uuid::new_v4(),
            session_id: uuid::Uuid::new_v4(),
            working_dir: std::path::PathBuf::from("/tmp"),
            env: HashMap::new(),
            resource_budgets: Default::default(),
        };

        let result = dry_run
            .execute(serde_json::json!({"command": "echo test"}), &ctx)
            .await
            .unwrap();

        assert!(!result.success);
        assert_eq!(result.error.as_deref(), Some("Simulated network error"));
    }

    #[tokio::test]
    async fn test_dry_run_forwards_trait_methods() {
        let shell = Arc::new(Shell::new());
        let dry_run = DryRunTool::new(shell.clone());

        // Name/description/schema should match inner tool
        assert_eq!(dry_run.name(), shell.name());
        assert_eq!(dry_run.description(), shell.description());

        let dry_schema = dry_run.schema();
        let inner_schema = shell.schema();
        assert_eq!(dry_schema.function.name, inner_schema.function.name);
        assert_eq!(
            dry_schema.function.description,
            inner_schema.function.description
        );
    }

    #[test]
    fn test_dry_run_config_defaults() {
        let config = DryRunConfig::default();
        assert!(config.validate_args);
        assert!(config.mock_success);
        assert!(config.mock_output.contains("DRY RUN"));
        assert_eq!(config.mock_duration_ms, 0);
        assert!(config.mock_error.is_none());
    }

    #[test]
    fn test_dry_run_config_serde() {
        let json = r#"{
            "validate_args": false,
            "mock_output": "test output",
            "mock_duration_ms": 100,
            "mock_success": false,
            "mock_error": "test error"
        }"#;

        let config: DryRunConfig = serde_json::from_str(json).unwrap();
        assert!(!config.validate_args);
        assert_eq!(config.mock_output, "test output");
        assert_eq!(config.mock_duration_ms, 100);
        assert!(!config.mock_success);
        assert_eq!(config.mock_error.unwrap(), "test error");
    }
}
