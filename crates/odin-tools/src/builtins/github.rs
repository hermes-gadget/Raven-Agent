//! GitHub tools — wraps `gh` CLI for repository management.
//!
//! All tools shell out to the GitHub CLI (`gh`) which must be installed
//! and authenticated. Each tool runs with a configurable timeout.
//! Read-only tools are safe; write tools require approval.

use std::time::Instant;

use async_trait::async_trait;
use chrono::Utc;
use serde::Deserialize;
use tokio::process::Command;
use tracing::instrument;

use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::{Tool, ToolContext};
use odin_core::types::{FunctionSchema, ToolResult, ToolSchema};

use crate::process::{self, DEFAULT_MAX_OUTPUT_BYTES, ProcessError};

// ── Shared Helper ───────────────────────────────────────────────────

/// Run a `gh` subcommand and return a [`ToolResult`].
async fn run_gh(
    subcommand_args: &[String],
    timeout_secs: u64,
    tool_name: &str,
) -> OdinResult<ToolResult> {
    let start = Instant::now();

    let mut cmd = Command::new("gh");
    cmd.args(subcommand_args);

    let timeout = std::time::Duration::from_secs(timeout_secs.max(1));

    let output = process::run_command(cmd, timeout, DEFAULT_MAX_OUTPUT_BYTES)
        .await
        .map_err(|error| match error {
            ProcessError::Timeout => {
                let joined = subcommand_args.join(" ");
                OdinError::Timeout(format!(
                    "gh command timed out after {timeout_secs}s: gh {joined}"
                ))
            }
            ProcessError::Io(error) => OdinError::Tool {
                tool: tool_name.to_string(),
                message: format!("Failed to execute gh command: {error}"),
                source: Some(Box::new(error)),
            },
        })?;

    let duration_ms = start.elapsed().as_millis() as u64;

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
            "\n\n[TRUNCATED: output exceeded {DEFAULT_MAX_OUTPUT_BYTES} bytes per stream]"
        ));
    }

    let success = output.status.success();
    let error = if success {
        None
    } else {
        let exit_code = output.status.code().unwrap_or(-1);
        let stderr = String::from_utf8_lossy(&output.stderr);
        Some(format!("gh command failed (exit {exit_code}): {stderr}"))
    };

    Ok(ToolResult {
        call_id: String::new(),
        tool_name: tool_name.to_string(),
        success,
        output: result_output,
        error,
        duration_ms,
        timestamp: Utc::now(),
    })
}

/// Build `gh issue create` arguments from parsed input.
fn build_issue_create_args(
    repo: &str,
    title: &str,
    body: Option<&str>,
    labels: Option<&str>,
) -> Vec<String> {
    let mut args = vec![
        "issue".into(),
        "create".into(),
        "-R".into(),
        repo.into(),
        "--title".into(),
        title.into(),
    ];
    if let Some(b) = body {
        args.push("--body".into());
        args.push(b.into());
    }
    if let Some(l) = labels {
        args.push("--label".into());
        args.push(l.into());
    }
    args
}

/// Build `gh issue list` arguments.
fn build_issue_search_args(
    repo: &str,
    query: Option<&str>,
    state: Option<&str>,
    limit: u32,
) -> Vec<String> {
    let mut args = vec![
        "issue".into(),
        "list".into(),
        "-R".into(),
        repo.into(),
        "-L".into(),
        limit.to_string(),
    ];
    if let Some(q) = query {
        args.push("--search".into());
        args.push(q.into());
    }
    if let Some(s) = state {
        let normalized = match s.to_lowercase().as_str() {
            "open" | "closed" => s.to_lowercase(),
            _ => "open".to_string(),
        };
        args.push("-s".into());
        args.push(normalized);
    }
    args
}

/// Build `gh pr create` arguments.
fn build_pr_create_args(
    repo: &str,
    title: &str,
    body: Option<&str>,
    base: &str,
    head: &str,
) -> Vec<String> {
    let mut args = vec![
        "pr".into(),
        "create".into(),
        "-R".into(),
        repo.into(),
        "--title".into(),
        title.into(),
        "-B".into(),
        base.into(),
        "-H".into(),
        head.into(),
    ];
    if let Some(b) = body {
        args.push("--body".into());
        args.push(b.into());
    }
    args
}

/// Build `gh pr view` or `gh pr list` arguments.
fn build_pr_status_args(repo: &str, pr_number: Option<u32>) -> Vec<String> {
    match pr_number {
        Some(num) => vec![
            "pr".into(),
            "view".into(),
            num.to_string(),
            "-R".into(),
            repo.into(),
            "--json".into(),
            "title,state,url,headRefName,baseRefName,author".into(),
        ],
        None => vec![
            "pr".into(),
            "list".into(),
            "-R".into(),
            repo.into(),
            "-L".into(),
            "10".into(),
        ],
    }
}

/// Build `gh run list` arguments.
fn build_actions_status_args(repo: &str, workflow: Option<&str>) -> Vec<String> {
    let mut args = vec![
        "run".into(),
        "list".into(),
        "-R".into(),
        repo.into(),
        "-L".into(),
        "5".into(),
    ];
    if let Some(w) = workflow {
        args.push("--workflow".into());
        args.push(w.into());
    }
    args
}

// ── github_issue_create ─────────────────────────────────────────────

/// Arguments for the `github_issue_create` tool.
#[derive(Debug, Deserialize)]
struct IssueCreateArgs {
    /// Repository in `owner/repo` format.
    repo: String,
    /// Issue title.
    title: String,
    /// Issue body / description (optional).
    #[serde(default)]
    body: Option<String>,
    /// Comma-separated label names (optional).
    #[serde(default)]
    labels: Option<String>,
}

/// Tool that creates a GitHub issue in a repository.
pub struct GithubIssueCreate {
    name: String,
    description: String,
}

impl GithubIssueCreate {
    pub fn new() -> Self {
        Self {
            name: "github_issue_create".into(),
            description: "Create a GitHub issue in a repository. Requires 'gh' CLI to be installed and authenticated.".into(),
        }
    }

    fn make_schema(name: &str) -> ToolSchema {
        ToolSchema {
            schema_type: "function".into(),
            function: FunctionSchema {
                name: name.into(),
                description: "Create a GitHub issue in the specified repository.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "Repository in owner/repo format (e.g. 'octocat/Hello-World')"
                        },
                        "title": {
                            "type": "string",
                            "description": "Issue title"
                        },
                        "body": {
                            "type": "string",
                            "description": "Issue body / description (optional)"
                        },
                        "labels": {
                            "type": "string",
                            "description": "Comma-separated label names (optional, e.g. 'bug,urgent')"
                        }
                    },
                    "required": ["repo", "title"]
                }),
            },
        }
    }
}

impl Default for GithubIssueCreate {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for GithubIssueCreate {
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
        true
    }

    fn is_safe(&self) -> bool {
        false
    }

    fn capability_tags(&self) -> &[&str] {
        &["github", "issue", "write", "dangerous"]
    }

    fn is_dangerous(&self) -> bool {
        true
    }

    #[instrument(skip(self, _context), fields(tool = self.name))]
    async fn execute(
        &self,
        args: serde_json::Value,
        _context: &ToolContext,
    ) -> OdinResult<ToolResult> {
        let parsed: IssueCreateArgs =
            serde_json::from_value(args).map_err(|e| OdinError::Tool {
                tool: self.name.clone(),
                message: format!("Invalid arguments: {e}"),
                source: Some(Box::new(e)),
            })?;

        let gh_args = build_issue_create_args(
            &parsed.repo,
            &parsed.title,
            parsed.body.as_deref(),
            parsed.labels.as_deref(),
        );

        run_gh(&gh_args, 60, &self.name).await
    }
}

// ── github_issue_search ──────────────────────────────────────────────

/// Arguments for the `github_issue_search` tool.
#[derive(Debug, Deserialize)]
struct IssueSearchArgs {
    /// Repository in `owner/repo` format.
    repo: String,
    /// Search query (optional — matches title, body, comments).
    #[serde(default)]
    query: Option<String>,
    /// Issue state filter: "open" or "closed" (default: open).
    #[serde(default)]
    state: Option<String>,
    /// Max results (default: 10).
    #[serde(default = "default_limit")]
    limit: u32,
}

fn default_limit() -> u32 {
    10
}

/// Tool that searches GitHub issues in a repository.
pub struct GithubIssueSearch {
    name: String,
    description: String,
}

impl GithubIssueSearch {
    pub fn new() -> Self {
        Self {
            name: "github_issue_search".into(),
            description:
                "Search GitHub issues in a repository. Supports state filtering and text search."
                    .into(),
        }
    }

    fn make_schema(name: &str) -> ToolSchema {
        ToolSchema {
            schema_type: "function".into(),
            function: FunctionSchema {
                name: name.into(),
                description: "Search GitHub issues in a repository with optional filters.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "Repository in owner/repo format (e.g. 'octocat/Hello-World')"
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query (optional — matches against title, body, and comments)"
                        },
                        "state": {
                            "type": "string",
                            "enum": ["open", "closed"],
                            "description": "Issue state filter (default: open)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results to return (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["repo"]
                }),
            },
        }
    }
}

impl Default for GithubIssueSearch {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for GithubIssueSearch {
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
        false
    }

    fn is_safe(&self) -> bool {
        true
    }

    fn capability_tags(&self) -> &[&str] {
        &["github", "issue", "read", "safe"]
    }

    fn is_dangerous(&self) -> bool {
        false
    }

    #[instrument(skip(self, _context), fields(tool = self.name))]
    async fn execute(
        &self,
        args: serde_json::Value,
        _context: &ToolContext,
    ) -> OdinResult<ToolResult> {
        let parsed: IssueSearchArgs =
            serde_json::from_value(args).map_err(|e| OdinError::Tool {
                tool: self.name.clone(),
                message: format!("Invalid arguments: {e}"),
                source: Some(Box::new(e)),
            })?;

        let gh_args = build_issue_search_args(
            &parsed.repo,
            parsed.query.as_deref(),
            parsed.state.as_deref(),
            parsed.limit,
        );

        run_gh(&gh_args, 30, &self.name).await
    }
}

// ── github_pr_create ─────────────────────────────────────────────────

/// Arguments for the `github_pr_create` tool.
#[derive(Debug, Deserialize)]
struct PrCreateArgs {
    /// Repository in `owner/repo` format.
    repo: String,
    /// PR title.
    title: String,
    /// PR body / description (optional).
    #[serde(default)]
    body: Option<String>,
    /// Base branch to merge into (default: "main").
    #[serde(default = "default_base")]
    base: String,
    /// Head branch to merge from (required).
    head: String,
}

fn default_base() -> String {
    "main".into()
}

/// Tool that creates a GitHub pull request.
pub struct GithubPrCreate {
    name: String,
    description: String,
}

impl GithubPrCreate {
    pub fn new() -> Self {
        Self {
            name: "github_pr_create".into(),
            description:
                "Create a GitHub pull request. Requires 'gh' CLI to be installed and authenticated."
                    .into(),
        }
    }

    fn make_schema(name: &str) -> ToolSchema {
        ToolSchema {
            schema_type: "function".into(),
            function: FunctionSchema {
                name: name.into(),
                description: "Create a pull request on a GitHub repository.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "Repository in owner/repo format (e.g. 'octocat/Hello-World')"
                        },
                        "title": {
                            "type": "string",
                            "description": "Pull request title"
                        },
                        "body": {
                            "type": "string",
                            "description": "Pull request body / description (optional)"
                        },
                        "base": {
                            "type": "string",
                            "description": "Base branch to merge into (default: 'main')",
                            "default": "main"
                        },
                        "head": {
                            "type": "string",
                            "description": "Head branch to merge from (required)"
                        }
                    },
                    "required": ["repo", "title", "head"]
                }),
            },
        }
    }
}

impl Default for GithubPrCreate {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for GithubPrCreate {
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
        true
    }

    fn is_safe(&self) -> bool {
        false
    }

    fn capability_tags(&self) -> &[&str] {
        &["github", "pr", "write", "dangerous"]
    }

    fn is_dangerous(&self) -> bool {
        true
    }

    #[instrument(skip(self, _context), fields(tool = self.name))]
    async fn execute(
        &self,
        args: serde_json::Value,
        _context: &ToolContext,
    ) -> OdinResult<ToolResult> {
        let parsed: PrCreateArgs = serde_json::from_value(args).map_err(|e| OdinError::Tool {
            tool: self.name.clone(),
            message: format!("Invalid arguments: {e}"),
            source: Some(Box::new(e)),
        })?;

        let gh_args = build_pr_create_args(
            &parsed.repo,
            &parsed.title,
            parsed.body.as_deref(),
            &parsed.base,
            &parsed.head,
        );

        run_gh(&gh_args, 60, &self.name).await
    }
}

// ── github_pr_status ─────────────────────────────────────────────────

/// Arguments for the `github_pr_status` tool.
#[derive(Debug, Deserialize)]
struct PrStatusArgs {
    /// Repository in `owner/repo` format.
    repo: String,
    /// PR number (optional — if omitted, lists all open PRs).
    #[serde(default)]
    pr_number: Option<u32>,
}

/// Tool that views or lists GitHub pull request status.
pub struct GithubPrStatus {
    name: String,
    description: String,
}

impl GithubPrStatus {
    pub fn new() -> Self {
        Self {
            name: "github_pr_status".into(),
            description: "View a specific pull request or list all open PRs in a repository."
                .into(),
        }
    }

    fn make_schema(name: &str) -> ToolSchema {
        ToolSchema {
            schema_type: "function".into(),
            function: FunctionSchema {
                name: name.into(),
                description: "View details of a specific PR or list open PRs for a repository."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "Repository in owner/repo format (e.g. 'octocat/Hello-World')"
                        },
                        "pr_number": {
                            "type": "integer",
                            "description": "PR number to view (optional — if omitted, lists all open PRs)"
                        }
                    },
                    "required": ["repo"]
                }),
            },
        }
    }
}

impl Default for GithubPrStatus {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for GithubPrStatus {
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
        false
    }

    fn is_safe(&self) -> bool {
        true
    }

    fn capability_tags(&self) -> &[&str] {
        &["github", "pr", "read", "safe"]
    }

    fn is_dangerous(&self) -> bool {
        false
    }

    #[instrument(skip(self, _context), fields(tool = self.name))]
    async fn execute(
        &self,
        args: serde_json::Value,
        _context: &ToolContext,
    ) -> OdinResult<ToolResult> {
        let parsed: PrStatusArgs = serde_json::from_value(args).map_err(|e| OdinError::Tool {
            tool: self.name.clone(),
            message: format!("Invalid arguments: {e}"),
            source: Some(Box::new(e)),
        })?;

        let gh_args = build_pr_status_args(&parsed.repo, parsed.pr_number);

        run_gh(&gh_args, 30, &self.name).await
    }
}

// ── github_actions_status ────────────────────────────────────────────

/// Arguments for the `github_actions_status` tool.
#[derive(Debug, Deserialize)]
struct ActionsStatusArgs {
    /// Repository in `owner/repo` format.
    repo: String,
    /// Workflow filename or name (optional — if omitted, shows latest across all workflows).
    #[serde(default)]
    workflow: Option<String>,
}

/// Tool that checks GitHub Actions CI status.
pub struct GithubActionsStatus {
    name: String,
    description: String,
}

impl GithubActionsStatus {
    pub fn new() -> Self {
        Self {
            name: "github_actions_status".into(),
            description: "Check the status of GitHub Actions workflow runs for a repository."
                .into(),
        }
    }

    fn make_schema(name: &str) -> ToolSchema {
        ToolSchema {
            schema_type: "function".into(),
            function: FunctionSchema {
                name: name.into(),
                description: "View recent GitHub Actions workflow runs for a repository.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "Repository in owner/repo format (e.g. 'octocat/Hello-World')"
                        },
                        "workflow": {
                            "type": "string",
                            "description": "Workflow filename (e.g. 'ci.yml') or workflow name (optional — if omitted, shows latest runs across all workflows)"
                        }
                    },
                    "required": ["repo"]
                }),
            },
        }
    }
}

impl Default for GithubActionsStatus {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for GithubActionsStatus {
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
        false
    }

    fn is_safe(&self) -> bool {
        true
    }

    fn capability_tags(&self) -> &[&str] {
        &["github", "ci", "read", "safe"]
    }

    fn is_dangerous(&self) -> bool {
        false
    }

    #[instrument(skip(self, _context), fields(tool = self.name))]
    async fn execute(
        &self,
        args: serde_json::Value,
        _context: &ToolContext,
    ) -> OdinResult<ToolResult> {
        let parsed: ActionsStatusArgs =
            serde_json::from_value(args).map_err(|e| OdinError::Tool {
                tool: self.name.clone(),
                message: format!("Invalid arguments: {e}"),
                source: Some(Box::new(e)),
            })?;

        let gh_args = build_actions_status_args(&parsed.repo, parsed.workflow.as_deref());

        run_gh(&gh_args, 30, &self.name).await
    }
}

// ── Tests ────────────────────────────────────────────────────────────

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
        }
    }

    // ── Schema & Tag Tests ─────────────────────────────────────────

    #[test]
    fn test_github_issue_create_schema() {
        let tool = GithubIssueCreate::new();
        assert_eq!(tool.name(), "github_issue_create");
        let schema = tool.schema();
        let params = &schema.function.parameters;
        assert!(
            params
                .get("properties")
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("repo")
        );
        assert!(
            params
                .get("properties")
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("title")
        );
        assert!(
            params
                .get("required")
                .unwrap()
                .as_array()
                .unwrap()
                .contains(&"repo".to_string().into())
        );
        assert!(
            params
                .get("required")
                .unwrap()
                .as_array()
                .unwrap()
                .contains(&"title".to_string().into())
        );
    }

    #[test]
    fn test_github_issue_create_tags() {
        let tool = GithubIssueCreate::new();
        assert!(tool.is_dangerous());
        assert!(tool.requires_approval());
        assert!(!tool.is_safe());
        assert_eq!(
            tool.capability_tags(),
            &["github", "issue", "write", "dangerous"]
        );
    }

    #[test]
    fn test_github_issue_search_schema() {
        let tool = GithubIssueSearch::new();
        assert_eq!(tool.name(), "github_issue_search");
        let schema = tool.schema();
        let params = &schema.function.parameters;
        assert!(
            params
                .get("properties")
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("repo")
        );
        assert!(
            params
                .get("properties")
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("query")
        );
        assert!(
            params
                .get("required")
                .unwrap()
                .as_array()
                .unwrap()
                .contains(&"repo".to_string().into())
        );
    }

    #[test]
    fn test_github_issue_search_tags() {
        let tool = GithubIssueSearch::new();
        assert!(!tool.is_dangerous());
        assert!(!tool.requires_approval());
        assert!(tool.is_safe());
        assert_eq!(tool.capability_tags(), &["github", "issue", "read", "safe"]);
    }

    #[test]
    fn test_github_pr_create_schema() {
        let tool = GithubPrCreate::new();
        assert_eq!(tool.name(), "github_pr_create");
        let schema = tool.schema();
        let params = &schema.function.parameters;
        assert!(
            params
                .get("properties")
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("repo")
        );
        assert!(
            params
                .get("properties")
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("title")
        );
        assert!(
            params
                .get("properties")
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("head")
        );
        assert!(
            params
                .get("required")
                .unwrap()
                .as_array()
                .unwrap()
                .contains(&"head".to_string().into())
        );
    }

    #[test]
    fn test_github_pr_create_tags() {
        let tool = GithubPrCreate::new();
        assert!(tool.is_dangerous());
        assert!(tool.requires_approval());
        assert!(!tool.is_safe());
        assert_eq!(
            tool.capability_tags(),
            &["github", "pr", "write", "dangerous"]
        );
    }

    #[test]
    fn test_github_pr_status_schema() {
        let tool = GithubPrStatus::new();
        assert_eq!(tool.name(), "github_pr_status");
        let schema = tool.schema();
        let params = &schema.function.parameters;
        assert!(
            params
                .get("properties")
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("repo")
        );
        assert!(
            params
                .get("properties")
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("pr_number")
        );
        assert!(
            params
                .get("required")
                .unwrap()
                .as_array()
                .unwrap()
                .contains(&"repo".to_string().into())
        );
    }

    #[test]
    fn test_github_pr_status_tags() {
        let tool = GithubPrStatus::new();
        assert!(!tool.is_dangerous());
        assert!(!tool.requires_approval());
        assert!(tool.is_safe());
        assert_eq!(tool.capability_tags(), &["github", "pr", "read", "safe"]);
    }

    #[test]
    fn test_github_actions_status_schema() {
        let tool = GithubActionsStatus::new();
        assert_eq!(tool.name(), "github_actions_status");
        let schema = tool.schema();
        let params = &schema.function.parameters;
        assert!(
            params
                .get("properties")
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("repo")
        );
        assert!(
            params
                .get("properties")
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("workflow")
        );
        assert!(
            params
                .get("required")
                .unwrap()
                .as_array()
                .unwrap()
                .contains(&"repo".to_string().into())
        );
    }

    #[test]
    fn test_github_actions_status_tags() {
        let tool = GithubActionsStatus::new();
        assert!(!tool.is_dangerous());
        assert!(!tool.requires_approval());
        assert!(tool.is_safe());
        assert_eq!(tool.capability_tags(), &["github", "ci", "read", "safe"]);
    }

    // ── Command Construction Tests ──────────────────────────────────

    #[test]
    fn test_build_issue_create_args_minimal() {
        let args = build_issue_create_args("owner/repo", "Test Issue", None, None);
        assert_eq!(args[0], "issue");
        assert_eq!(args[1], "create");
        assert_eq!(args[3], "owner/repo");
        assert_eq!(args[5], "Test Issue");
        assert_eq!(args.len(), 6);
    }

    #[test]
    fn test_build_issue_create_args_with_body_and_labels() {
        let args = build_issue_create_args(
            "owner/repo",
            "Test Issue",
            Some("Description here"),
            Some("bug,urgent"),
        );
        assert_eq!(args[6], "--body");
        assert_eq!(args[7], "Description here");
        assert_eq!(args[8], "--label");
        assert_eq!(args[9], "bug,urgent");
    }

    #[test]
    fn test_build_issue_search_args_defaults() {
        let args = build_issue_search_args("owner/repo", None, None, 10);
        assert_eq!(args[0], "issue");
        assert_eq!(args[1], "list");
        assert_eq!(args[3], "owner/repo");
        assert_eq!(args[5], "10");
        assert_eq!(args.len(), 6);
    }

    #[test]
    fn test_build_issue_search_args_with_all() {
        let args = build_issue_search_args("o/r", Some("bug"), Some("closed"), 25);
        assert_eq!(args[6], "--search");
        assert_eq!(args[7], "bug");
        assert_eq!(args[8], "-s");
        assert_eq!(args[9], "closed");
    }

    #[test]
    fn test_build_pr_create_args_minimal() {
        let args = build_pr_create_args("owner/repo", "My PR", None, "main", "feature-branch");
        assert_eq!(args[0], "pr");
        assert_eq!(args[1], "create");
        assert_eq!(args[3], "owner/repo");
        assert_eq!(args[5], "My PR");
        assert_eq!(args[7], "main");
        assert_eq!(args[9], "feature-branch");
        assert_eq!(args.len(), 10);
    }

    #[test]
    fn test_build_pr_create_args_with_body() {
        let args = build_pr_create_args("o/r", "My PR", Some("Desc"), "develop", "fix");
        assert_eq!(args[10], "--body");
        assert_eq!(args[11], "Desc");
    }

    #[test]
    fn test_build_pr_status_args_with_number() {
        let args = build_pr_status_args("owner/repo", Some(42));
        assert_eq!(args[0], "pr");
        assert_eq!(args[1], "view");
        assert_eq!(args[2], "42");
        assert_eq!(args[4], "owner/repo");
        assert!(args.contains(&"--json".into()));
    }

    #[test]
    fn test_build_pr_status_args_without_number() {
        let args = build_pr_status_args("owner/repo", None);
        assert_eq!(args[0], "pr");
        assert_eq!(args[1], "list");
        assert_eq!(args[3], "owner/repo");
        assert_eq!(args[5], "10");
    }

    #[test]
    fn test_build_actions_status_args_minimal() {
        let args = build_actions_status_args("owner/repo", None);
        assert_eq!(args[0], "run");
        assert_eq!(args[1], "list");
        assert_eq!(args[3], "owner/repo");
        assert_eq!(args[5], "5");
        assert_eq!(args.len(), 6);
    }

    #[test]
    fn test_build_actions_status_args_with_workflow() {
        let args = build_actions_status_args("owner/repo", Some("ci.yml"));
        assert_eq!(args[6], "--workflow");
        assert_eq!(args[7], "ci.yml");
    }

    // ── Integration Tests (skip if gh not available) ────────────────

    #[tokio::test]
    async fn test_gh_cli_available() {
        let output = Command::new("gh").arg("--version").output().await;
        match output {
            Ok(out) if out.status.success() => {
                let ver = String::from_utf8_lossy(&out.stdout);
                assert!(
                    ver.contains("gh") || ver.contains("GitHub"),
                    "output: {ver}"
                );
            }
            _ => {
                eprintln!("gh CLI not installed — skipping integration tests");
                return;
            }
        }
    }

    #[tokio::test]
    async fn test_issue_create_invalid_repo() {
        let tool = GithubIssueCreate::new();
        let args = serde_json::json!({
            "repo": "nonexistent-user-12345/nonexistent-repo-67890",
            "title": "Test issue — please ignore if seen"
        });
        let result = tool.execute(args, &test_context()).await;
        // Should either succeed or fail gracefully — gh will report auth/not-found
        assert!(result.is_ok());
        if let Ok(res) = result {
            // If gh is not installed, we might get a failure
            if !res.success {
                assert!(
                    res.error.as_ref().unwrap_or(&String::new()).contains("gh")
                        || res
                            .error
                            .as_ref()
                            .unwrap_or(&String::new())
                            .contains("not found")
                        || res
                            .error
                            .as_ref()
                            .unwrap_or(&String::new())
                            .contains("Authentication")
                );
            }
        }
    }
}
