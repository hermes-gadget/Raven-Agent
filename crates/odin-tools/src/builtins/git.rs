//! Git tool — wraps git commands for repository management.

use std::time::Instant;
use std::{path::PathBuf, sync::Arc};

use async_trait::async_trait;
use chrono::Utc;
use serde::Deserialize;
use tokio::process::Command;
use tracing::instrument;

use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::{Tool, ToolContext};
use odin_core::types::{FunctionSchema, ToolResult, ToolSchema};

use crate::Sandbox;
use crate::process::{self, ProcessError};

/// Arguments for the `git` tool.
#[derive(Debug, Deserialize)]
struct GitArgs {
    /// Git subcommand and arguments (e.g., "status", "log --oneline -5").
    command: String,
    /// Path to the git repository.
    #[serde(default)]
    repo_path: Option<String>,
    /// Timeout in seconds (optional, default: 120).
    #[serde(default = "default_timeout")]
    timeout_secs: u64,
}

fn default_timeout() -> u64 {
    120
}

/// Tool that executes git commands in a repository.
///
/// Wraps the `git` CLI and runs commands in the specified repository
/// directory. Supports any git subcommand.
pub struct Git {
    name: String,
    description: String,
    sandbox: Arc<Sandbox>,
}

impl Git {
    /// Create a new `Git` tool.
    pub fn new() -> Self {
        Self {
            name: "git".into(),
            description: "Execute git commands in a repository. Use for cloning, committing, pushing, pulling, and other git operations.".into(),
            sandbox: Arc::new(Sandbox::default()),
        }
    }

    /// Create a Git tool constrained by the supplied filesystem boundary.
    pub fn with_sandbox(sandbox: Arc<Sandbox>) -> Self {
        Self {
            sandbox,
            ..Self::new()
        }
    }

    /// Construct the JSON schema.
    fn make_schema(name: &str) -> ToolSchema {
        ToolSchema {
            schema_type: "function".into(),
            function: FunctionSchema {
                name: name.into(),
                description: "Execute a git command in a repository.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Git command and arguments (e.g., 'status', 'log --oneline -5', 'add .')"
                        },
                        "repo_path": {
                            "type": "string",
                            "description": "Path to the git repository (optional, defaults to agent working directory)"
                        },
                        "timeout_secs": {
                            "type": "integer",
                            "description": "Timeout in seconds (optional, defaults to 120)",
                            "default": 120,
                            "minimum": 1,
                            "maximum": 300
                        }
                    },
                    "required": ["command"]
                }),
            },
        }
    }

    /// Build a list of arguments for the git command.
    fn build_args(command_str: &str) -> Vec<String> {
        let mut args = vec!["-c".to_string(), "color.ui=false".to_string()];

        // Split the command string into individual arguments, respecting quotes
        let parts = shlex_split(command_str);
        args.extend(parts);

        args
    }
}

impl Default for Git {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for Git {
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
        // Git push/push --force require approval
        true
    }

    fn is_safe(&self) -> bool {
        false
    }

    fn capability_tags(&self) -> Vec<String> {
        vec!["version-control".into(), "git".into(), "dangerous".into()]
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

        let parsed: GitArgs = serde_json::from_value(args).map_err(|e| OdinError::Tool {
            tool: self.name.clone(),
            message: format!("Invalid arguments: {e}"),
            source: Some(Box::new(e)),
        })?;

        let command_str = &parsed.command;

        let requested_repo = parsed
            .repo_path
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(|| context.working_dir.clone());
        let requested_repo = if requested_repo.is_absolute() {
            requested_repo
        } else {
            context.working_dir.join(requested_repo)
        };
        let repo_path = self.sandbox.check_write(&requested_repo)?;

        // Build git command
        let git_args = Self::build_args(command_str);
        let subcommand = git_args.first().cloned().unwrap_or_default();

        let mut cmd = Command::new("git");
        cmd.args(&git_args);

        // Set repository path
        cmd.current_dir(repo_path);

        // Set timeout
        let timeout_secs = parsed
            .timeout_secs
            .max(1)
            .min(context.resource_budgets.max_tool_timeout_secs.max(1));
        let timeout = std::time::Duration::from_secs(timeout_secs);
        let max_output_bytes = context.resource_budgets.max_tool_output_bytes.max(1);

        let output = process::run_command(cmd, timeout, max_output_bytes)
            .await
            .map_err(|error| match error {
                ProcessError::Timeout => OdinError::Timeout(format!(
                    "Git command timed out after {}s: git {}",
                    timeout_secs, command_str
                )),
                ProcessError::Io(error) => OdinError::Tool {
                    tool: self.name.clone(),
                    message: format!("Failed to execute git command: {error}"),
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
                "\n\n[TRUNCATED: output exceeded {max_output_bytes} bytes per stream]"
            ));
        }

        let success = output.status.success();
        let error = if success {
            None
        } else {
            let exit_code = output.status.code().unwrap_or(-1);
            let stderr = String::from_utf8_lossy(&output.stderr);
            Some(format!(
                "Git {} failed (exit {exit_code}): {stderr}",
                subcommand
            ))
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

/// Simple shell-style argument splitter that respects single and double quotes.
///
/// This is a minimal implementation used to split a git command string into
/// individual arguments while preserving quoted segments.
fn shlex_split(input: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;

    for ch in input.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }

        match ch {
            '\\' if in_double => {
                escaped = true;
            }
            '\\' if !in_single => {
                // Outside quotes, backslash escapes next char
                escaped = true;
            }
            '\'' if !in_double => {
                in_single = !in_single;
            }
            '"' if !in_single => {
                in_double = !in_double;
            }
            ' ' | '\t' if !in_single && !in_double => {
                if !current.is_empty() {
                    args.push(current.clone());
                    current.clear();
                }
            }
            _ => {
                current.push(ch);
            }
        }
    }

    if !current.is_empty() {
        args.push(current);
    }

    args
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
    async fn test_git_version() {
        let git = Git::new();
        let args = serde_json::json!({
            "command": "version",
            "timeout_secs": 10
        });
        let result = git.execute(args, &test_context()).await.unwrap();
        assert!(result.success, "STDERR: {:?}", result.error);
        assert!(
            result.output.contains("git version"),
            "output: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn test_git_invalid_repo() {
        let git = Git::new();
        let args = serde_json::json!({
            "command": "status",
            "repo_path": "/nonexistent/path",
            "timeout_secs": 5
        });
        let result = git.execute(args, &test_context()).await;
        // Should either error on spawn or get a non-zero exit from git
        assert!(result.is_ok() || result.is_err());
        if let Ok(res) = result {
            assert!(!res.success);
        }
    }

    #[tokio::test]
    async fn test_git_init_and_status() {
        // Create a temp directory, init a git repo, and check status
        let dir = tempfile::tempdir().unwrap();
        let repo_path = dir.path();

        // Initialize git repo
        let git = Git::with_sandbox(Arc::new(Sandbox::new(odin_core::types::PathBoundary {
            allowed_read: vec![repo_path.display().to_string()],
            allowed_write: vec![repo_path.display().to_string()],
            denied: vec![],
        })));
        let init_args = serde_json::json!({
            "command": "init",
            "repo_path": repo_path.to_string_lossy(),
            "timeout_secs": 10
        });
        let result = git.execute(init_args, &test_context()).await.unwrap();
        assert!(result.success, "git init failed: {:?}", result.error);

        // Check status
        let status_args = serde_json::json!({
            "command": "status",
            "repo_path": repo_path.to_string_lossy(),
            "timeout_secs": 10
        });
        let result = git.execute(status_args, &test_context()).await.unwrap();
        assert!(result.success, "git status failed: {:?}", result.error);
    }

    #[test]
    fn test_shlex_split() {
        assert_eq!(shlex_split("status"), vec!["status"]);
        assert_eq!(
            shlex_split("log --oneline -5"),
            vec!["log", "--oneline", "-5"]
        );
        assert_eq!(
            shlex_split("commit -m 'my message'"),
            vec!["commit", "-m", "my message"]
        );
        assert_eq!(
            shlex_split("add \"file name with spaces.txt\""),
            vec!["add", "file name with spaces.txt"]
        );
    }

    #[test]
    fn test_build_args() {
        let args = Git::build_args("status");
        assert_eq!(args[0], "-c");
        assert_eq!(args[1], "color.ui=false");
        assert_eq!(args[2], "status");

        let args = Git::build_args("log --oneline -5");
        assert_eq!(args[2], "log");
        assert_eq!(args[3], "--oneline");
        assert_eq!(args[4], "-5");
    }
}
