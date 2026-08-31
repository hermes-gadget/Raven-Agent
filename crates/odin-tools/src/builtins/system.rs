//! System diagnostic tools — system info and disk usage.
//!
//! These tools gather OS-level and filesystem information using standard
//! command-line utilities (`uname`, `free`, `df`). They are safe, read-only
//! diagnostic tools with no side effects.

use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::Utc;
use serde::Deserialize;
use tracing::instrument;

use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::{Tool, ToolContext};
use odin_core::types::{FunctionSchema, ToolResult, ToolSchema};

use crate::process::{self, ProcessError};

fn map_process_error(tool_name: &str, error: ProcessError) -> OdinError {
    match error {
        ProcessError::Timeout => OdinError::Timeout(format!(
            "{tool_name} subprocess exceeded its resource budget"
        )),
        ProcessError::Io(error) => OdinError::Tool {
            tool: tool_name.to_string(),
            message: format!("Failed to execute subprocess: {error}"),
            source: Some(Box::new(error)),
        },
    }
}

// ── system_info ─────────────────────────────────────────────────────

/// Tool that returns basic OS system information.
///
/// Runs `uname -a` for kernel/OS info and `free -h` for memory usage.
/// Returns a combined text report. Requires no arguments.
pub struct SystemInfo {
    name: String,
    description: String,
}

impl SystemInfo {
    /// Create a new `SystemInfo` tool.
    pub fn new() -> Self {
        Self {
            name: "system_info".into(),
            description: "Get operating system information: kernel version, hostname, CPU architecture, and memory usage. Safe read-only diagnostic tool.".into(),
        }
    }

    fn make_schema(name: &str) -> ToolSchema {
        ToolSchema {
            schema_type: "function".into(),
            function: FunctionSchema {
                name: name.into(),
                description: "Get OS information (uname + memory).".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            },
        }
    }
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for SystemInfo {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn schema(&self) -> ToolSchema {
        Self::make_schema(&self.name)
    }

    fn is_safe(&self) -> bool {
        true
    }

    fn capability_tags(&self) -> Vec<String> {
        vec!["diagnostic".into(), "read".into(), "safe".into()]
    }

    #[instrument(skip(self, context), fields(tool = self.name))]
    async fn execute(
        &self,
        _args: serde_json::Value,
        context: &ToolContext,
    ) -> OdinResult<ToolResult> {
        let start = Instant::now();
        let timeout = Duration::from_secs(context.resource_budgets.max_tool_timeout_secs.max(1));
        let max_output_bytes = context.resource_budgets.max_tool_output_bytes.max(1);

        let mut uname = tokio::process::Command::new("uname");
        uname.arg("-a");
        let uname_output = process::run_command(uname, timeout, max_output_bytes)
            .await
            .map_err(|error| map_process_error(self.name(), error))?;

        let mut free = tokio::process::Command::new("free");
        free.arg("-h");
        let free_output = process::run_command(free, timeout, max_output_bytes)
            .await
            .map_err(|error| map_process_error(self.name(), error))?;

        let sysname = String::from_utf8_lossy(&uname_output.stdout)
            .trim()
            .to_string();
        let memory = String::from_utf8_lossy(&free_output.stdout)
            .trim()
            .to_string();

        let mut output = format!(
            "── System Information ──\n\
             Kernel/OS: {sysname}\n\n\
             ── Memory ──\n\
             {memory}"
        );
        if uname_output.stdout_truncated || free_output.stdout_truncated {
            output.push_str("\n\n[TRUNCATED: output exceeded the configured per-stream limit]");
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(ToolResult {
            call_id: String::new(),
            tool_name: self.name.clone(),
            success: uname_output.status.success() && free_output.status.success(),
            output,
            error: if uname_output.status.success() && free_output.status.success() {
                None
            } else {
                Some("one or more system diagnostic commands failed".into())
            },
            duration_ms,
            timestamp: Utc::now(),
        })
    }
}

// ── disk_usage ──────────────────────────────────────────────────────

/// Arguments for `disk_usage`.
#[derive(Debug, Deserialize)]
struct DiskUsageArgs {
    path: Option<String>,
}

/// Tool that reports disk usage via `df -h`.
///
/// Optionally accepts a path argument; without it defaults to showing
/// all mounted filesystems (`df -h`).
pub struct DiskUsage {
    name: String,
    description: String,
}

impl DiskUsage {
    /// Create a new `DiskUsage` tool.
    pub fn new() -> Self {
        Self {
            name: "disk_usage".into(),
            description: "Show disk usage information. Runs `df -h` optionally scoped to a path. Safe read-only diagnostic tool.".into(),
        }
    }

    fn make_schema(name: &str) -> ToolSchema {
        ToolSchema {
            schema_type: "function".into(),
            function: FunctionSchema {
                name: name.into(),
                description: "Show disk usage (df -h).".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Optional path to check disk usage for (default: all filesystems)"
                        }
                    },
                    "required": []
                }),
            },
        }
    }
}

impl Default for DiskUsage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for DiskUsage {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn schema(&self) -> ToolSchema {
        Self::make_schema(&self.name)
    }

    fn is_safe(&self) -> bool {
        true
    }

    fn capability_tags(&self) -> Vec<String> {
        vec!["diagnostic".into(), "read".into(), "safe".into()]
    }

    #[instrument(skip(self, context), fields(tool = self.name))]
    async fn execute(
        &self,
        args: serde_json::Value,
        context: &ToolContext,
    ) -> OdinResult<ToolResult> {
        let start = Instant::now();

        let parsed: DiskUsageArgs = serde_json::from_value(args).map_err(|e| OdinError::Tool {
            tool: self.name.clone(),
            message: format!("Invalid arguments: {e}"),
            source: Some(Box::new(e)),
        })?;

        let mut cmd = tokio::process::Command::new("df");
        cmd.arg("-h");

        if let Some(ref path) = parsed.path {
            cmd.arg(path);
        }

        let output = process::run_command(
            cmd,
            Duration::from_secs(context.resource_budgets.max_tool_timeout_secs.max(1)),
            context.resource_budgets.max_tool_output_bytes.max(1),
        )
        .await
        .map_err(|error| map_process_error(self.name(), error))?;

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();

        let duration_ms = start.elapsed().as_millis() as u64;
        let success = output.status.success();

        let mut out = stdout;
        let error = if !success {
            Some(if stderr.is_empty() {
                format!("df exited with code {:?}", output.status.code())
            } else {
                stderr.clone()
            })
        } else {
            None
        };

        // Append stderr to output when there's useful info
        if !stderr.is_empty() {
            out.push_str("\n--- stderr ---\n");
            out.push_str(&stderr);
        }
        if output.stdout_truncated || output.stderr_truncated {
            out.push_str("\n\n[TRUNCATED: output exceeded the configured per-stream limit]");
        }

        Ok(ToolResult {
            call_id: String::new(),
            tool_name: self.name.clone(),
            success,
            output: out,
            error,
            duration_ms,
            timestamp: Utc::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn test_context() -> ToolContext {
        ToolContext {
            agent_id: Default::default(),
            session_id: Default::default(),
            working_dir: PathBuf::from("/tmp"),
            env: HashMap::new(),
            resource_budgets: Default::default(),
        }
    }

    #[tokio::test]
    async fn test_system_info_basic() {
        let tool = SystemInfo::new();
        let result = tool
            .execute(serde_json::json!({}), &test_context())
            .await
            .unwrap();
        assert!(result.success);
        // Should contain at least "Linux" or similar kernel info
        assert!(!result.output.is_empty());
        assert!(result.output.contains("System Information"));
        assert!(result.output.contains("Memory"));
    }

    #[tokio::test]
    async fn test_system_info_empty_args() {
        let tool = SystemInfo::new();
        let result = tool
            .execute(serde_json::Value::Null, &test_context())
            .await
            .unwrap();
        assert!(result.success);
        assert!(!result.output.is_empty());
    }

    #[tokio::test]
    async fn test_disk_usage_basic() {
        let tool = DiskUsage::new();
        let result = tool
            .execute(serde_json::json!({}), &test_context())
            .await
            .unwrap();
        assert!(result.success);
        // Should show at least a filesystem header or data
        assert!(!result.output.is_empty());
    }

    #[tokio::test]
    async fn test_disk_usage_with_path() {
        let tool = DiskUsage::new();
        let result = tool
            .execute(serde_json::json!({"path": "/"}), &test_context())
            .await
            .unwrap();
        assert!(result.success);
        assert!(!result.output.is_empty());
    }

    #[tokio::test]
    async fn test_disk_usage_invalid_path() {
        let tool = DiskUsage::new();
        let result = tool
            .execute(
                serde_json::json!({"path": "/nonexistent_path_xyz_123"}),
                &test_context(),
            )
            .await
            .unwrap();
        // df will return non-zero exit code for invalid paths
        assert!(!result.success || result.output.contains("nonexistent"));
    }
}
