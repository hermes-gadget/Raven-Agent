//! Utility built-in tools.

use async_trait::async_trait;
use chrono::Utc;
use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::{Tool, ToolContext};
use odin_core::types::ToolResult;
use rand::Rng;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::process::Command;

use crate::Sandbox;
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

fn command_result(tool_name: &str, output: process::BoundedOutput, start: Instant) -> ToolResult {
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let mut combined = stdout;
    if !stderr.is_empty() {
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str("STDERR:\n");
        combined.push_str(&stderr);
    }
    if output.stdout_truncated || output.stderr_truncated {
        combined.push_str(&format!(
            "\n\n[TRUNCATED: output exceeded the configured per-stream limit]"
        ));
    }

    ToolResult {
        call_id: String::new(),
        tool_name: tool_name.to_string(),
        success: output.status.success(),
        output: combined.trim().to_string(),
        error: (!output.status.success()).then(|| {
            if stderr.is_empty() {
                format!("subprocess exited with code {:?}", output.status.code())
            } else {
                stderr
            }
        }),
        duration_ms: start.elapsed().as_millis() as u64,
        timestamp: Utc::now(),
    }
}

// ── file_list ─────────────────────────────────────────────────────────

pub struct FileList {
    sandbox: Arc<Sandbox>,
}

impl FileList {
    pub fn new(sandbox: Arc<Sandbox>) -> Self {
        Self { sandbox }
    }
}

#[derive(Deserialize)]
struct FileListArgs {
    path: Option<String>,
    pattern: Option<String>,
}

#[async_trait]
impl Tool for FileList {
    fn name(&self) -> &str {
        "file_list"
    }
    fn description(&self) -> &str {
        "List files and directories at a given path, optionally filtered by a glob pattern."
    }
    fn schema(&self) -> odin_core::types::ToolSchema {
        odin_core::types::ToolSchema {
            schema_type: "function".into(),
            function: odin_core::types::FunctionSchema {
                name: "file_list".into(),
                description: self.description().into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path (default: current directory)"},
                        "pattern": {"type": "string", "description": "Optional glob pattern to filter (e.g., '*.rs')"}
                    }
                }),
            },
        }
    }
    fn is_dangerous(&self) -> bool {
        false
    }
    fn is_safe(&self) -> bool {
        true
    }
    fn requires_approval(&self) -> bool {
        false
    }
    fn capability_tags(&self) -> Vec<String> {
        vec!["filesystem".into(), "read".into(), "safe".into()]
    }

    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> OdinResult<ToolResult> {
        let start = Instant::now();
        let args: FileListArgs = serde_json::from_value(args)
            .map_err(|e| OdinError::tool("file_list", format!("args: {e}")))?;

        let requested = args
            .path
            .as_deref()
            .map(Path::new)
            .map(|path| {
                if path.is_absolute() {
                    path.to_path_buf()
                } else {
                    ctx.working_dir.join(path)
                }
            })
            .unwrap_or_else(|| ctx.working_dir.clone());
        let dir = self.sandbox.check_read(&requested)?;
        let timeout = Duration::from_secs(ctx.resource_budgets.max_tool_timeout_secs.max(1));
        let max_output_bytes = ctx.resource_budgets.max_tool_output_bytes.max(1);
        let mut cmd = Command::new("ls");
        cmd.arg("-1A"); // one per line, include dotfiles
        if let Some(ref pat) = args.pattern {
            // Filter by pattern — ls doesn't do globs natively, so we fall back
            // to using the pattern as a shell glob via find
            let mut find = Command::new("find");
            find.arg(&dir)
                .arg("-maxdepth")
                .arg("1")
                .arg("-name")
                .arg(pat);
            let output = process::run_command(find, timeout, max_output_bytes)
                .await
                .map_err(|error| map_process_error(self.name(), error))?;
            return Ok(command_result(self.name(), output, start));
        } else {
            cmd.current_dir(&dir);
        }

        let output = process::run_command(cmd, timeout, max_output_bytes)
            .await
            .map_err(|error| map_process_error(self.name(), error))?;
        Ok(command_result(self.name(), output, start))
    }
}

// ── file_delete ────────────────────────────────────────────────────────

pub struct FileDelete {
    sandbox: Arc<Sandbox>,
}

impl FileDelete {
    pub fn new(sandbox: Arc<Sandbox>) -> Self {
        Self { sandbox }
    }
}

#[derive(Deserialize)]
struct FileDeleteArgs {
    path: String,
}

#[async_trait]
impl Tool for FileDelete {
    fn name(&self) -> &str {
        "file_delete"
    }
    fn description(&self) -> &str {
        "Delete a file or empty directory at the specified path. Use with caution."
    }
    fn schema(&self) -> odin_core::types::ToolSchema {
        odin_core::types::ToolSchema {
            schema_type: "function".into(),
            function: odin_core::types::FunctionSchema {
                name: "file_delete".into(),
                description: self.description().into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "required": ["path"],
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file or empty directory to delete"}
                    }
                }),
            },
        }
    }
    fn is_dangerous(&self) -> bool {
        true
    }
    fn is_safe(&self) -> bool {
        false
    }
    fn requires_approval(&self) -> bool {
        true
    }
    fn capability_tags(&self) -> Vec<String> {
        vec!["filesystem".into(), "write".into(), "dangerous".into()]
    }

    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> OdinResult<ToolResult> {
        let start = Instant::now();
        let args: FileDeleteArgs = serde_json::from_value(args)
            .map_err(|e| OdinError::tool("file_delete", format!("args: {e}")))?;

        let requested = PathBuf::from(&args.path);
        let requested = if requested.is_absolute() {
            requested
        } else {
            ctx.working_dir.join(requested)
        };
        let path = self.sandbox.check_write(&requested)?;
        if !path.exists() {
            return Ok(ToolResult {
                call_id: String::new(),
                tool_name: self.name().to_string(),
                success: false,
                output: format!("Path does not exist: {}", args.path),
                error: Some("ENOENT".into()),
                duration_ms: start.elapsed().as_millis() as u64,
                timestamp: Utc::now(),
            });
        }

        let result = if path.is_dir() {
            std::fs::remove_dir(&path)
        } else {
            std::fs::remove_file(&path)
        };

        match result {
            Ok(()) => Ok(ToolResult {
                call_id: String::new(),
                tool_name: self.name().to_string(),
                success: true,
                output: format!("Deleted: {}", args.path),
                error: None,
                duration_ms: start.elapsed().as_millis() as u64,
                timestamp: Utc::now(),
            }),
            Err(e) => Ok(ToolResult {
                call_id: String::new(),
                tool_name: self.name().to_string(),
                success: false,
                output: format!("Failed to delete {}: {e}", args.path),
                error: Some(e.to_string()),
                duration_ms: start.elapsed().as_millis() as u64,
                timestamp: Utc::now(),
            }),
        }
    }
}

// ── file_exists ────────────────────────────────────────────────────────

pub struct FileExists {
    sandbox: Arc<Sandbox>,
}

impl FileExists {
    pub fn new(sandbox: Arc<Sandbox>) -> Self {
        Self { sandbox }
    }
}

#[derive(Deserialize)]
struct FileExistsArgs {
    path: String,
}

#[async_trait]
impl Tool for FileExists {
    fn name(&self) -> &str {
        "file_exists"
    }
    fn description(&self) -> &str {
        "Check whether a file or directory exists at the given path."
    }
    fn schema(&self) -> odin_core::types::ToolSchema {
        odin_core::types::ToolSchema {
            schema_type: "function".into(),
            function: odin_core::types::FunctionSchema {
                name: "file_exists".into(),
                description: self.description().into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "required": ["path"],
                    "properties": {
                        "path": {"type": "string", "description": "Path to check"}
                    }
                }),
            },
        }
    }
    fn is_dangerous(&self) -> bool {
        false
    }
    fn is_safe(&self) -> bool {
        true
    }
    fn requires_approval(&self) -> bool {
        false
    }
    fn capability_tags(&self) -> Vec<String> {
        vec!["filesystem".into(), "read".into(), "safe".into()]
    }

    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> OdinResult<ToolResult> {
        let start = Instant::now();
        let args: FileExistsArgs = serde_json::from_value(args)
            .map_err(|e| OdinError::tool("file_exists", format!("args: {e}")))?;

        let requested = PathBuf::from(&args.path);
        let requested = if requested.is_absolute() {
            requested
        } else {
            ctx.working_dir.join(requested)
        };
        let path = self.sandbox.check_read(&requested)?;
        let exists = path.exists();
        let kind = if exists {
            if path.is_dir() { "directory" } else { "file" }
        } else {
            "none"
        };

        Ok(ToolResult {
            call_id: String::new(),
            tool_name: self.name().to_string(),
            success: true,
            output: serde_json::json!({
                "exists": exists,
                "path": args.path,
                "type": kind,
            })
            .to_string(),
            error: None,
            duration_ms: start.elapsed().as_millis() as u64,
            timestamp: Utc::now(),
        })
    }
}

// ── env_var ────────────────────────────────────────────────────────────

pub struct EnvVar;

#[derive(Deserialize)]
struct EnvVarArgs {
    name: String,
}

#[async_trait]
impl Tool for EnvVar {
    fn name(&self) -> &str {
        "env_var"
    }
    fn description(&self) -> &str {
        "Read the value of an environment variable."
    }
    fn schema(&self) -> odin_core::types::ToolSchema {
        odin_core::types::ToolSchema {
            schema_type: "function".into(),
            function: odin_core::types::FunctionSchema {
                name: "env_var".into(),
                description: self.description().into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string", "description": "Environment variable name"}
                    }
                }),
            },
        }
    }
    fn is_dangerous(&self) -> bool {
        true
    }
    fn is_safe(&self) -> bool {
        false
    }
    fn requires_approval(&self) -> bool {
        true
    }
    fn capability_tags(&self) -> Vec<String> {
        vec![
            "system".into(),
            "environment".into(),
            "sensitive".into(),
            "dangerous".into(),
        ]
    }

    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> OdinResult<ToolResult> {
        let start = Instant::now();
        let args: EnvVarArgs = serde_json::from_value(args)
            .map_err(|e| OdinError::tool("env_var", format!("args: {e}")))?;

        match ctx.env.get(&args.name) {
            Some(val) => Ok(ToolResult {
                call_id: String::new(),
                tool_name: self.name().to_string(),
                success: true,
                output: val.clone(),
                error: None,
                duration_ms: start.elapsed().as_millis() as u64,
                timestamp: Utc::now(),
            }),
            None => Ok(ToolResult {
                call_id: String::new(),
                tool_name: self.name().to_string(),
                success: true,
                output: String::new(),
                error: Some(format!("ENV_NOT_ALLOWED: {}", args.name)),
                duration_ms: start.elapsed().as_millis() as u64,
                timestamp: Utc::now(),
            }),
        }
    }
}

// ── time_now ───────────────────────────────────────────────────────────

pub struct TimeNow;

#[async_trait]
impl Tool for TimeNow {
    fn name(&self) -> &str {
        "time_now"
    }
    fn description(&self) -> &str {
        "Get the current date and time in ISO 8601 (RFC 3339) format."
    }
    fn schema(&self) -> odin_core::types::ToolSchema {
        odin_core::types::ToolSchema {
            schema_type: "function".into(),
            function: odin_core::types::FunctionSchema {
                name: "time_now".into(),
                description: self.description().into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            },
        }
    }
    fn is_dangerous(&self) -> bool {
        false
    }
    fn is_safe(&self) -> bool {
        true
    }
    fn requires_approval(&self) -> bool {
        false
    }
    fn capability_tags(&self) -> Vec<String> {
        vec!["system".into(), "read".into(), "safe".into()]
    }

    async fn execute(
        &self,
        _args: serde_json::Value,
        _ctx: &ToolContext,
    ) -> OdinResult<ToolResult> {
        let start = Instant::now();
        let now = Utc::now();
        Ok(ToolResult {
            call_id: String::new(),
            tool_name: self.name().to_string(),
            success: true,
            output: serde_json::json!({
                "iso8601": now.to_rfc3339(),
                "unix_secs": now.timestamp(),
                "unix_millis": now.timestamp_millis(),
            })
            .to_string(),
            error: None,
            duration_ms: start.elapsed().as_millis() as u64,
            timestamp: now,
        })
    }
}

// ── random_number ─────────────────────────────────────────────────────

pub struct RandomNumber;

#[derive(Deserialize)]
struct RandomNumberArgs {
    #[serde(default = "default_min")]
    min: i64,
    #[serde(default = "default_max")]
    max: i64,
}

fn default_min() -> i64 {
    0
}
fn default_max() -> i64 {
    100
}

#[async_trait]
impl Tool for RandomNumber {
    fn name(&self) -> &str {
        "random_number"
    }
    fn description(&self) -> &str {
        "Generate a random integer between min and max (inclusive)."
    }
    fn schema(&self) -> odin_core::types::ToolSchema {
        odin_core::types::ToolSchema {
            schema_type: "function".into(),
            function: odin_core::types::FunctionSchema {
                name: "random_number".into(),
                description: self.description().into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "min": {"type": "integer", "description": "Minimum value (default: 0)"},
                        "max": {"type": "integer", "description": "Maximum value (default: 100)"}
                    }
                }),
            },
        }
    }
    fn is_dangerous(&self) -> bool {
        false
    }
    fn is_safe(&self) -> bool {
        true
    }
    fn requires_approval(&self) -> bool {
        false
    }
    fn capability_tags(&self) -> Vec<String> {
        vec!["utility".into(), "safe".into()]
    }

    async fn execute(&self, args: serde_json::Value, _ctx: &ToolContext) -> OdinResult<ToolResult> {
        let start = Instant::now();
        let args: RandomNumberArgs = serde_json::from_value(args)
            .map_err(|e| OdinError::tool("random_number", format!("args: {e}")))?;

        if args.min > args.max {
            return Ok(ToolResult {
                call_id: String::new(),
                tool_name: self.name().to_string(),
                success: false,
                output: String::new(),
                error: Some("min must be <= max".into()),
                duration_ms: start.elapsed().as_millis() as u64,
                timestamp: Utc::now(),
            });
        }

        let mut rng = rand::thread_rng();
        let value = rng.gen_range(args.min..=args.max);

        Ok(ToolResult {
            call_id: String::new(),
            tool_name: self.name().to_string(),
            success: true,
            output: serde_json::json!({"value": value, "min": args.min, "max": args.max})
                .to_string(),
            error: None,
            duration_ms: start.elapsed().as_millis() as u64,
            timestamp: Utc::now(),
        })
    }
}

// ── json_validate ─────────────────────────────────────────────────────

pub struct JsonValidate;

#[derive(Deserialize)]
struct JsonValidateArgs {
    input: String,
}

#[async_trait]
impl Tool for JsonValidate {
    fn name(&self) -> &str {
        "json_validate"
    }
    fn description(&self) -> &str {
        "Validate whether a string is valid JSON. Returns parse errors if invalid."
    }
    fn schema(&self) -> odin_core::types::ToolSchema {
        odin_core::types::ToolSchema {
            schema_type: "function".into(),
            function: odin_core::types::FunctionSchema {
                name: "json_validate".into(),
                description: self.description().into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "required": ["input"],
                    "properties": {
                        "input": {"type": "string", "description": "JSON string to validate"}
                    }
                }),
            },
        }
    }
    fn is_dangerous(&self) -> bool {
        false
    }
    fn is_safe(&self) -> bool {
        true
    }
    fn requires_approval(&self) -> bool {
        false
    }
    fn capability_tags(&self) -> Vec<String> {
        vec!["data".into(), "validation".into(), "safe".into()]
    }

    async fn execute(&self, args: serde_json::Value, _ctx: &ToolContext) -> OdinResult<ToolResult> {
        let start = Instant::now();
        let args: JsonValidateArgs = serde_json::from_value(args)
            .map_err(|e| OdinError::tool("json_validate", format!("args: {e}")))?;

        match serde_json::from_str::<serde_json::Value>(&args.input) {
            Ok(v) => Ok(ToolResult {
                call_id: String::new(),
                tool_name: self.name().to_string(),
                success: true,
                output: serde_json::json!({
                    "valid": true,
                    "type": if v.is_object() { "object" }
                            else if v.is_array() { "array" }
                            else if v.is_string() { "string" }
                            else if v.is_number() { "number" }
                            else if v.is_boolean() { "boolean" }
                            else if v.is_null() { "null" }
                            else { "unknown" },
                })
                .to_string(),
                error: None,
                duration_ms: start.elapsed().as_millis() as u64,
                timestamp: Utc::now(),
            }),
            Err(e) => Ok(ToolResult {
                call_id: String::new(),
                tool_name: self.name().to_string(),
                success: true,
                output: serde_json::json!({
                    "valid": false,
                    "error": e.to_string(),
                    "line": e.line(),
                    "column": e.column(),
                })
                .to_string(),
                error: None,
                duration_ms: start.elapsed().as_millis() as u64,
                timestamp: Utc::now(),
            }),
        }
    }
}

// ── text_search ────────────────────────────────────────────────────────

pub struct TextSearch;

#[derive(Deserialize)]
struct TextSearchArgs {
    text: String,
    pattern: String,
    #[serde(default)]
    case_insensitive: bool,
}

#[async_trait]
impl Tool for TextSearch {
    fn name(&self) -> &str {
        "text_search"
    }
    fn description(&self) -> &str {
        "Search for a regex pattern in text. Returns matching lines with line numbers."
    }
    fn schema(&self) -> odin_core::types::ToolSchema {
        odin_core::types::ToolSchema {
            schema_type: "function".into(),
            function: odin_core::types::FunctionSchema {
                name: "text_search".into(),
                description: self.description().into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "required": ["text", "pattern"],
                    "properties": {
                        "text": {"type": "string", "description": "Text to search in"},
                        "pattern": {"type": "string", "description": "Regex pattern to match"},
                        "case_insensitive": {"type": "boolean", "description": "Case-insensitive search (default: false)"}
                    }
                }),
            },
        }
    }
    fn is_dangerous(&self) -> bool {
        false
    }
    fn is_safe(&self) -> bool {
        true
    }
    fn requires_approval(&self) -> bool {
        false
    }
    fn capability_tags(&self) -> Vec<String> {
        vec!["data".into(), "search".into(), "safe".into()]
    }

    async fn execute(&self, args: serde_json::Value, _ctx: &ToolContext) -> OdinResult<ToolResult> {
        let start = Instant::now();
        let args: TextSearchArgs = serde_json::from_value(args)
            .map_err(|e| OdinError::tool("text_search", format!("args: {e}")))?;

        let re = if args.case_insensitive {
            regex::RegexBuilder::new(&args.pattern)
                .case_insensitive(true)
                .build()
        } else {
            regex::Regex::new(&args.pattern)
        };

        let re = match re {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolResult {
                    call_id: String::new(),
                    tool_name: self.name().to_string(),
                    success: false,
                    output: String::new(),
                    error: Some(format!("Invalid regex: {e}")),
                    duration_ms: start.elapsed().as_millis() as u64,
                    timestamp: Utc::now(),
                });
            }
        };

        let matches: Vec<serde_json::Value> = args
            .text
            .lines()
            .enumerate()
            .filter_map(|(i, line)| {
                if re.is_match(line) {
                    Some(serde_json::json!({
                        "line": i + 1,
                        "content": line,
                    }))
                } else {
                    None
                }
            })
            .collect();

        Ok(ToolResult {
            call_id: String::new(),
            tool_name: self.name().to_string(),
            success: true,
            output: serde_json::json!({
                "count": matches.len(),
                "matches": matches,
            })
            .to_string(),
            error: None,
            duration_ms: start.elapsed().as_millis() as u64,
            timestamp: Utc::now(),
        })
    }
}

// ── process_list ──────────────────────────────────────────────────────

pub struct ProcessList;

#[async_trait]
impl Tool for ProcessList {
    fn name(&self) -> &str {
        "process_list"
    }
    fn description(&self) -> &str {
        "List running processes (ps aux). Read-only, no arguments."
    }
    fn schema(&self) -> odin_core::types::ToolSchema {
        odin_core::types::ToolSchema {
            schema_type: "function".into(),
            function: odin_core::types::FunctionSchema {
                name: "process_list".into(),
                description: self.description().into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            },
        }
    }
    fn is_dangerous(&self) -> bool {
        false
    }
    fn is_safe(&self) -> bool {
        true
    }
    fn requires_approval(&self) -> bool {
        true
    }
    fn capability_tags(&self) -> Vec<String> {
        vec!["system".into(), "read".into(), "safe".into()]
    }

    async fn execute(&self, _args: serde_json::Value, ctx: &ToolContext) -> OdinResult<ToolResult> {
        let start = Instant::now();
        let output = process::run_command(
            {
                let mut command = Command::new("ps");
                command.arg("aux");
                command
            },
            Duration::from_secs(ctx.resource_budgets.max_tool_timeout_secs.max(1)),
            ctx.resource_budgets.max_tool_output_bytes.max(1),
        )
        .await
        .map_err(|error| map_process_error(self.name(), error))?;

        Ok(command_result(self.name(), output, start))
    }
}

// ── network_ping ──────────────────────────────────────────────────────

pub struct NetworkPing;

#[derive(Deserialize)]
struct NetworkPingArgs {
    host: String,
    #[serde(default = "default_count")]
    count: u32,
}

fn default_count() -> u32 {
    1
}

#[async_trait]
impl Tool for NetworkPing {
    fn name(&self) -> &str {
        "network_ping"
    }
    fn description(&self) -> &str {
        "Ping a host to check connectivity. Uses system ping command."
    }
    fn schema(&self) -> odin_core::types::ToolSchema {
        odin_core::types::ToolSchema {
            schema_type: "function".into(),
            function: odin_core::types::FunctionSchema {
                name: "network_ping".into(),
                description: self.description().into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "required": ["host"],
                    "properties": {
                        "host": {"type": "string", "description": "Hostname or IP address to ping"},
                        "count": {"type": "integer", "minimum": 1, "maximum": 32, "description": "Number of ping packets (default: 1)"}
                    }
                }),
            },
        }
    }
    fn is_dangerous(&self) -> bool {
        false
    }
    fn is_safe(&self) -> bool {
        true
    }
    fn requires_approval(&self) -> bool {
        false
    }
    fn capability_tags(&self) -> Vec<String> {
        vec!["network".into(), "diagnostic".into(), "safe".into()]
    }

    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> OdinResult<ToolResult> {
        let start = Instant::now();
        let args: NetworkPingArgs = serde_json::from_value(args)
            .map_err(|e| OdinError::tool("network_ping", format!("args: {e}")))?;

        let max_count = ctx.resource_budgets.max_network_ping_count.max(1);
        if args.count == 0 || args.count > max_count {
            return Err(OdinError::Validation(format!(
                "network_ping count must be between 1 and {max_count}"
            )));
        }

        let mut command = Command::new("ping");
        command
            .arg("-c")
            .arg(args.count.to_string())
            .arg("-W")
            .arg("5") // 5 second timeout
            .arg(&args.host);
        let output = process::run_command(
            command,
            Duration::from_secs(ctx.resource_budgets.max_tool_timeout_secs.max(1)),
            ctx.resource_budgets.max_tool_output_bytes.max(1),
        )
        .await
        .map_err(|error| map_process_error(self.name(), error))?;

        Ok(command_result(self.name(), output, start))
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use odin_core::traits::ToolContext;

    fn test_ctx() -> ToolContext {
        ToolContext {
            agent_id: uuid::Uuid::new_v4(),
            session_id: uuid::Uuid::new_v4(),
            working_dir: std::env::current_dir().unwrap_or_default(),
            env: std::collections::HashMap::from([(
                "RAVEN_SAFE_TEST_VALUE".to_string(),
                "allowlisted-value".to_string(),
            )]),
            resource_budgets: Default::default(),
        }
    }

    fn test_sandbox() -> Arc<Sandbox> {
        Arc::new(Sandbox::default())
    }

    #[tokio::test]
    async fn test_file_list_current_dir() {
        let tool = FileList::new(test_sandbox());
        let result = tool
            .execute(serde_json::json!({}), &test_ctx())
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("Cargo.toml") || result.output.contains("src"));
    }

    #[tokio::test]
    async fn test_file_exists_true() {
        let tool = FileExists::new(test_sandbox());
        let result = tool
            .execute(serde_json::json!({"path": "Cargo.toml"}), &test_ctx())
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("\"exists\":true"));
    }

    #[tokio::test]
    async fn test_file_exists_false() {
        let tool = FileExists::new(test_sandbox());
        let missing = std::env::current_dir()
            .unwrap()
            .join("nonexistent-path-xyzzy");
        let result = tool
            .execute(serde_json::json!({"path": missing}), &test_ctx())
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("\"exists\":false"));
    }

    #[tokio::test]
    async fn test_env_var_allowlisted_value() {
        let tool = EnvVar;
        let result = tool
            .execute(
                serde_json::json!({"name": "RAVEN_SAFE_TEST_VALUE"}),
                &test_ctx(),
            )
            .await
            .unwrap();
        assert!(result.success);
        assert_eq!(result.output, "allowlisted-value");
    }

    #[tokio::test]
    async fn test_env_var_missing() {
        let tool = EnvVar;
        let result = tool
            .execute(
                serde_json::json!({"name": "THIS_VAR_DOES_NOT_EXIST_XYZ"}),
                &test_ctx(),
            )
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.error.unwrap().contains("ENV_NOT_ALLOWED"));
    }

    #[tokio::test]
    async fn test_env_var_does_not_fall_back_to_process_environment() {
        let tool = EnvVar;
        let mut context = test_ctx();
        context.env.clear();
        let result = tool
            .execute(serde_json::json!({"name": "HOME"}), &context)
            .await
            .unwrap();
        assert!(result.output.is_empty());
        assert!(result.error.unwrap().contains("ENV_NOT_ALLOWED"));
        assert!(tool.requires_approval());
        assert!(!tool.is_safe());
    }

    #[tokio::test]
    async fn filesystem_utilities_reject_paths_outside_the_shared_sandbox() {
        let allowed = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let outside_file = outside.path().join("secret.txt");
        std::fs::write(&outside_file, "synthetic secret").unwrap();
        let sandbox = Arc::new(Sandbox::new(odin_core::types::PathBoundary {
            allowed_read: vec![allowed.path().display().to_string()],
            allowed_write: vec![allowed.path().display().to_string()],
            denied: vec![],
        }));
        let context = test_ctx();

        assert!(
            FileList::new(sandbox.clone())
                .execute(serde_json::json!({"path": outside.path()}), &context)
                .await
                .is_err()
        );
        assert!(
            FileExists::new(sandbox.clone())
                .execute(serde_json::json!({"path": outside_file}), &context)
                .await
                .is_err()
        );
        assert!(
            FileDelete::new(sandbox)
                .execute(serde_json::json!({"path": outside_file}), &context)
                .await
                .is_err()
        );
        assert!(outside_file.exists());
    }

    #[tokio::test]
    async fn test_time_now() {
        let tool = TimeNow;
        let result = tool
            .execute(serde_json::json!({}), &test_ctx())
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("iso8601"));
        assert!(result.output.contains("unix_secs"));
    }

    #[tokio::test]
    async fn test_random_number() {
        let tool = RandomNumber;
        let result = tool
            .execute(serde_json::json!({"min": 1, "max": 10}), &test_ctx())
            .await
            .unwrap();
        assert!(result.success);
        let v: serde_json::Value = serde_json::from_str(&result.output).unwrap();
        let val = v["value"].as_i64().unwrap();
        assert!((1..=10).contains(&val));
    }

    #[tokio::test]
    async fn test_json_validate_valid() {
        let tool = JsonValidate;
        let result = tool
            .execute(
                serde_json::json!({"input": r#"{"key": "value"}"#}),
                &test_ctx(),
            )
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("\"valid\":true"));
    }

    #[tokio::test]
    async fn test_json_validate_invalid() {
        let tool = JsonValidate;
        let result = tool
            .execute(serde_json::json!({"input": "{bad json}"}), &test_ctx())
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("\"valid\":false"));
    }

    #[tokio::test]
    async fn test_text_search() {
        let tool = TextSearch;
        let result = tool
            .execute(
                serde_json::json!({
                    "text": "hello world\nfoo bar\nhello again\nbaz",
                    "pattern": "hello"
                }),
                &test_ctx(),
            )
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("\"count\":2"));
    }

    #[tokio::test]
    async fn test_process_list() {
        let tool = ProcessList;
        let result = tool
            .execute(serde_json::json!({}), &test_ctx())
            .await
            .unwrap();
        assert!(result.success);
        assert!(!result.output.is_empty());
    }

    #[test]
    fn process_list_requires_approval() {
        assert!(ProcessList.requires_approval());
        assert!(ProcessList.is_safe());
    }
}
