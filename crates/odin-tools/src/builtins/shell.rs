//! Shell command execution tool with dangerous-command detection.

use std::time::Instant;
use std::{path::PathBuf, sync::Arc};

use async_trait::async_trait;
use chrono::Utc;
use regex::Regex;
use serde::Deserialize;
use tokio::process::Command;
use tracing::instrument;

use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::{Tool, ToolContext};
use odin_core::types::{FunctionSchema, ToolResult, ToolSchema};

use crate::Sandbox;
use crate::process::{self, DEFAULT_MAX_OUTPUT_BYTES, ProcessError};

/// Arguments for the `shell` tool.
#[derive(Debug, Deserialize)]
struct ShellArgs {
    command: String,
    #[serde(default)]
    workdir: Option<String>,
    #[serde(default = "default_timeout")]
    timeout_secs: u64,
    /// If true, validate the command without executing it.
    #[serde(default)]
    dry_run: bool,
    /// Maximum output bytes retained per stream (default: 1MB). Zero uses the default.
    #[serde(default = "default_max_output_bytes")]
    max_output_bytes: usize,
}

fn default_timeout() -> u64 {
    60
}

fn default_max_output_bytes() -> usize {
    DEFAULT_MAX_OUTPUT_BYTES
}

/// Tool that executes shell commands.
///
/// The shell tool runs commands via `/bin/sh -c`. It checks the command
/// against a list of dangerous patterns and marks itself as requiring
/// approval if a match is found.
pub struct Shell {
    name: String,
    description: String,
    dangerous_patterns: Vec<Regex>,
    sandbox: Arc<Sandbox>,
}

impl Shell {
    /// Create a new `Shell` tool with default dangerous-command patterns.
    pub fn new() -> Self {
        let patterns = [
            r"rm\s+-rf",
            r"rm\s+-r\s+/",
            r"git\s+reset\s+--hard",
            r"git\s+push\s+--force",
            r"sudo\s+",
            r"chmod\s+777",
            r">\s*/dev/",
            r"mkfs\.",
            r"dd\s+if=",
            r":\(\)\s*\{",
            r">\s*/dev/sda",
            r"mv\s+/\s+",
            r"shutdown\s",
            r"reboot\s",
            r"init\s+0",
            r"init\s+6",
            r"poweroff",
        ];

        let re_patterns: Vec<Regex> = patterns
            .iter()
            .map(|p| Regex::new(p).expect("Invalid dangerous command regex"))
            .collect();

        Self {
            name: "shell".into(),
            description: "Execute a shell command and return its stdout and stderr. Use for running terminal commands, scripts, or any system interaction.".into(),
            dangerous_patterns: re_patterns,
            sandbox: Arc::new(Sandbox::default()),
        }
    }

    /// Create a shell tool constrained by the supplied filesystem boundary.
    pub fn with_sandbox(sandbox: Arc<Sandbox>) -> Self {
        Self {
            sandbox,
            ..Self::new()
        }
    }

    /// Create a `Shell` with custom dangerous-command patterns.
    pub fn with_patterns(patterns: Vec<Regex>) -> Self {
        Self {
            name: "shell".into(),
            description: "Execute a shell command and return its stdout and stderr.".into(),
            dangerous_patterns: patterns,
            sandbox: Arc::new(Sandbox::default()),
        }
    }

    /// Check whether a command matches any dangerous pattern.
    pub fn is_dangerous(&self, command: &str) -> bool {
        self.dangerous_patterns
            .iter()
            .any(|re| re.is_match(command))
    }

    /// Construct the JSON schema for this tool.
    fn make_schema(name: &str) -> ToolSchema {
        ToolSchema {
            schema_type: "function".into(),
            function: FunctionSchema {
                name: name.into(),
                description: "Execute a shell command.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute (passed to /bin/sh -c)"
                        },
                        "workdir": {
                            "type": "string",
                            "description": "Working directory for the command (optional, defaults to agent working dir)"
                        },
                        "timeout_secs": {
                            "type": "integer",
                            "description": "Timeout in seconds (optional, defaults to 60)",
                            "default": 60,
                            "minimum": 1,
                            "maximum": 300
                        },
                        "max_output_bytes": {
                            "type": "integer",
                            "description": "Maximum output bytes retained per stream",
                            "minimum": 1,
                            "maximum": 16777216
                        }
                    },
                    "required": ["command"]
                }),
            },
        }
    }
}

impl Default for Shell {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for Shell {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn schema(&self) -> ToolSchema {
        Self::make_schema(&self.name)
    }

    fn requires_approval(&self) -> bool {
        true // shell commands always require approval for safety
    }

    fn is_safe(&self) -> bool {
        false // shell is not inherently safe
    }

    fn capability_tags(&self) -> Vec<String> {
        vec!["shell".into(), "system".into(), "dangerous".into()]
    }

    fn is_dangerous(&self) -> bool {
        true
    }

    #[instrument(skip(self, context), fields(tool = self.name))]
    async fn execute(
        &self,
        args: serde_json::Value,
        context: &ToolContext,
    ) -> OdinResult<ToolResult> {
        let start = Instant::now();

        let parsed: ShellArgs = serde_json::from_value(args).map_err(|e| OdinError::Tool {
            tool: self.name.clone(),
            message: format!("Invalid arguments: {e}"),
            source: Some(Box::new(e)),
        })?;

        let command_str = &parsed.command;

        // Check for dangerous commands
        if self.is_dangerous(command_str) {
            return Ok(ToolResult {
                call_id: String::new(),
                tool_name: self.name.clone(),
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Command matches dangerous pattern and was blocked: {command_str}"
                )),
                duration_ms: 0,
                timestamp: Utc::now(),
            });
        }

        let requested_workdir = parsed
            .workdir
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(|| context.working_dir.clone());
        let requested_workdir = if requested_workdir.is_absolute() {
            requested_workdir
        } else {
            context.working_dir.join(requested_workdir)
        };
        let workdir = self.sandbox.check_write(&requested_workdir)?;

        // Dry-run mode: validate without executing
        if parsed.dry_run {
            return Ok(ToolResult {
                call_id: String::new(),
                tool_name: self.name.clone(),
                success: true,
                output: format!(
                    "[DRY RUN] Command would be executed (not dangerous, {}s timeout): {}",
                    parsed.timeout_secs, command_str
                ),
                error: None,
                duration_ms: 0,
                timestamp: Utc::now(),
            });
        }

        // Build the command
        let mut cmd = Command::new("/bin/sh");
        cmd.args(["-c", command_str]);

        // Set working directory if provided
        cmd.current_dir(workdir);

        // Set timeout
        let timeout_secs = parsed
            .timeout_secs
            .max(1)
            .min(context.resource_budgets.max_tool_timeout_secs.max(1));
        let timeout = std::time::Duration::from_secs(timeout_secs);

        let requested_max_bytes = if parsed.max_output_bytes == 0 {
            DEFAULT_MAX_OUTPUT_BYTES
        } else {
            parsed.max_output_bytes
        };
        let max_bytes =
            requested_max_bytes.min(context.resource_budgets.max_tool_output_bytes.max(1));
        let output = process::run_command(cmd, timeout, max_bytes)
            .await
            .map_err(|error| match error {
                ProcessError::Timeout => OdinError::Timeout(format!(
                    "Shell command timed out after {}s: {command_str}",
                    timeout_secs
                )),
                ProcessError::Io(error) => OdinError::Tool {
                    tool: self.name.clone(),
                    message: format!("Failed to execute command: {error}"),
                    source: Some(Box::new(error)),
                },
            })?;

        let duration_ms = start.elapsed().as_millis() as u64;

        // Combine stdout and stderr
        let mut result_output = String::new();
        if !output.stdout.is_empty() {
            result_output.push_str(&String::from_utf8_lossy(&output.stdout));
        }
        if !output.stderr.is_empty() {
            if !result_output.is_empty() {
                result_output.push('\n');
            }
            result_output.push_str("STDERR:\n");
            result_output.push_str(&String::from_utf8_lossy(&output.stderr));
        }

        if output.stdout_truncated || output.stderr_truncated {
            result_output.push_str(&format!(
                "\n\n[TRUNCATED: output exceeded {max_bytes} bytes per stream]"
            ));
        }

        let success = output.status.success();
        let error = if success {
            None
        } else {
            Some(format!("Exit code: {}", output.status.code().unwrap_or(-1)))
        };

        Ok(ToolResult {
            call_id: String::new(),
            tool_name: self.name.clone(),
            success,
            output: result_output,
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

    fn test_context() -> ToolContext {
        ToolContext {
            agent_id: Default::default(),
            session_id: Default::default(),
            working_dir: std::env::current_dir().unwrap(),
            env: HashMap::new(),
            resource_budgets: Default::default(),
        }
    }

    #[tokio::test]
    async fn test_shell_echo() {
        let shell = Shell::new();
        let args = serde_json::json!({
            "command": "echo 'hello raven'",
            "timeout_secs": 10
        });
        let result = shell.execute(args, &test_context()).await.unwrap();
        assert!(result.success, "STDERR: {:?}", result.error);
        assert!(
            result.output.contains("hello raven"),
            "output: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn test_shell_pwd() {
        let shell = Shell::new();
        let expected = std::env::current_dir().unwrap();
        let args = serde_json::json!({
            "command": "pwd",
            "workdir": expected,
            "timeout_secs": 10
        });
        let result = shell.execute(args, &test_context()).await.unwrap();
        assert!(result.success);
        assert!(
            result
                .output
                .contains(&std::env::current_dir().unwrap().display().to_string()),
            "output: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn test_shell_rejects_workdir_outside_sandbox() {
        let allowed = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let shell = Shell::with_sandbox(Arc::new(Sandbox::new(odin_core::types::PathBoundary {
            allowed_read: vec![allowed.path().display().to_string()],
            allowed_write: vec![allowed.path().display().to_string()],
            denied: vec![],
        })));
        let args = serde_json::json!({
            "command": "pwd",
            "workdir": outside.path(),
            "timeout_secs": 10
        });
        assert!(shell.execute(args, &test_context()).await.is_err());
    }

    #[tokio::test]
    async fn test_shell_dangerous_blocked() {
        let shell = Shell::new();
        let args = serde_json::json!({
            "command": "rm -rf /important",
            "timeout_secs": 5
        });
        let result = shell.execute(args, &test_context()).await.unwrap();
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap_or("").contains("dangerous"));
    }

    #[tokio::test]
    async fn test_is_dangerous() {
        let shell = Shell::new();
        assert!(shell.is_dangerous("rm -rf /"));
        assert!(shell.is_dangerous("sudo rm -rf /"));
        assert!(shell.is_dangerous("git push --force origin main"));
        assert!(!shell.is_dangerous("ls -la"));
        assert!(!shell.is_dangerous("echo hello"));
        assert!(!shell.is_dangerous("cat /etc/hostname"));
    }

    #[tokio::test]
    async fn test_shell_timeout() {
        let shell = Shell::new();
        let args = serde_json::json!({
            "command": "sleep 30",
            "timeout_secs": 1
        });
        let result = shell.execute(args, &test_context()).await;
        assert!(result.is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_shell_timeout_kills_delayed_side_effect() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("should-not-exist");
        let command = format!("sleep 3; touch '{}'", marker.display());
        let shell = Shell::new();
        let args = serde_json::json!({
            "command": command,
            "timeout_secs": 1
        });

        assert!(shell.execute(args, &test_context()).await.is_err());
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert!(!marker.exists());
    }

    #[tokio::test]
    async fn test_shell_invalid_command() {
        let shell = Shell::new();
        let args = serde_json::json!({
            "command": "nonexistent_command_xyz123",
            "timeout_secs": 5
        });
        let result = shell.execute(args, &test_context()).await.unwrap();
        assert!(!result.success);
    }

    #[tokio::test]
    async fn test_shell_dry_run_safe_command() {
        let shell = Shell::new();
        let args = serde_json::json!({
            "command": "echo 'dry run test'",
            "dry_run": true
        });
        let result = shell.execute(args, &test_context()).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("[DRY RUN]"));
        assert!(result.output.contains("echo 'dry run test'"));
    }

    #[tokio::test]
    async fn test_shell_dry_run_dangerous_blocked() {
        let shell = Shell::new();
        let args = serde_json::json!({
            "command": "rm -rf /important",
            "dry_run": true
        });
        let result = shell.execute(args, &test_context()).await.unwrap();
        // Dangerous commands are blocked even in dry-run mode
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap_or("").contains("dangerous"));
    }

    #[tokio::test]
    async fn test_shell_output_truncation() {
        let shell = Shell::new();
        // Generate ~500 bytes then cap at 100
        let args = serde_json::json!({
            "command": "python3 -c \"print('x' * 500)\"",
            "timeout_secs": 10,
            "max_output_bytes": 100
        });
        let result = shell.execute(args, &test_context()).await.unwrap();
        assert!(result.success, "STDERR: {:?}", result.error);
        assert!(result.output.contains("[TRUNCATED"));
        assert!(result.output.len() < 200); // should be 100 + truncation message
    }

    #[tokio::test]
    async fn test_shell_no_truncation_when_under_limit() {
        let shell = Shell::new();
        let args = serde_json::json!({
            "command": "echo 'short output'",
            "timeout_secs": 10,
            "max_output_bytes": 1024
        });
        let result = shell.execute(args, &test_context()).await.unwrap();
        assert!(result.success, "STDERR: {:?}", result.error);
        assert!(!result.output.contains("[TRUNCATED"));
    }
}
