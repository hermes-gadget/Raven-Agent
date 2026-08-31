//! File read/write tools with sandbox boundary enforcement.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use chrono::Utc;
use serde::Deserialize;
use tracing::instrument;

use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::{Tool, ToolContext};
use odin_core::types::{FunctionSchema, ToolResult, ToolSchema};

use crate::sandbox::Sandbox;

/// Arguments for `file_read`.
#[derive(Debug, Deserialize)]
struct FileReadArgs {
    path: String,
}

/// Tool that reads the contents of a file.
///
/// Enforces sandbox boundaries — the path must be within the allowed read
/// directories and not in the denied list.
pub struct FileRead {
    name: String,
    description: String,
    sandbox: Arc<Sandbox>,
}

impl FileRead {
    /// Create a new `FileRead` tool with the given sandbox.
    pub fn new(sandbox: Arc<Sandbox>) -> Self {
        Self {
            name: "file_read".into(),
            description: "Read the contents of a file at the given path. Returns the file contents as a string.".into(),
            sandbox,
        }
    }

    /// Construct the JSON schema for this tool.
    fn make_schema(name: &str) -> ToolSchema {
        ToolSchema {
            schema_type: "function".into(),
            function: FunctionSchema {
                name: name.into(),
                description: "Read the contents of a file at the given path.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative path to the file to read"
                        }
                    },
                    "required": ["path"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for FileRead {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn schema(&self) -> ToolSchema {
        Self::make_schema(&self.name)
    }

    fn capability_tags(&self) -> Vec<String> {
        vec!["filesystem".into(), "read".into(), "safe".into()]
    }

    #[instrument(skip(self, _context), fields(tool = self.name))]
    async fn execute(
        &self,
        args: serde_json::Value,
        _context: &ToolContext,
    ) -> OdinResult<ToolResult> {
        let start = Instant::now();

        let parsed: FileReadArgs = serde_json::from_value(args).map_err(|e| OdinError::Tool {
            tool: self.name.clone(),
            message: format!("Invalid arguments: {e}"),
            source: Some(Box::new(e)),
        })?;

        let path = Path::new(&parsed.path);
        let canonical = self.sandbox.check_read(path)?;

        let content = tokio::fs::read_to_string(&canonical)
            .await
            .map_err(OdinError::Io)?;

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(ToolResult {
            call_id: String::new(),
            tool_name: self.name.clone(),
            success: true,
            output: content,
            error: None,
            duration_ms,
            timestamp: Utc::now(),
        })
    }
}

/// Arguments for `file_write`.
#[derive(Debug, Deserialize)]
struct FileWriteArgs {
    path: String,
    content: String,
}

/// Tool that writes content to a file.
///
/// Enforces sandbox boundaries — the path must be within the allowed write
/// directories and not in the denied list.
pub struct FileWrite {
    name: String,
    description: String,
    sandbox: Arc<Sandbox>,
}

impl FileWrite {
    /// Create a new `FileWrite` tool with the given sandbox.
    pub fn new(sandbox: Arc<Sandbox>) -> Self {
        Self {
            name: "file_write".into(),
            description: "Write content to a file at the given path. Creates parent directories if they don't exist.".into(),
            sandbox,
        }
    }

    /// Construct the JSON schema for this tool.
    fn make_schema(name: &str) -> ToolSchema {
        ToolSchema {
            schema_type: "function".into(),
            function: FunctionSchema {
                name: name.into(),
                description: "Write content to a file at the given path.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative path to the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
                        }
                    },
                    "required": ["path", "content"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for FileWrite {
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
        true // file writes always require approval for safety
    }

    fn is_safe(&self) -> bool {
        false // file writes are not inherently safe
    }

    fn is_dangerous(&self) -> bool {
        true
    }

    fn capability_tags(&self) -> Vec<String> {
        vec!["filesystem".into(), "write".into(), "dangerous".into()]
    }

    #[instrument(skip(self, _context), fields(tool = self.name))]
    async fn execute(
        &self,
        args: serde_json::Value,
        _context: &ToolContext,
    ) -> OdinResult<ToolResult> {
        let start = Instant::now();

        let parsed: FileWriteArgs = serde_json::from_value(args).map_err(|e| OdinError::Tool {
            tool: self.name.clone(),
            message: format!("Invalid arguments: {e}"),
            source: Some(Box::new(e)),
        })?;

        let path = Path::new(&parsed.path);
        let canonical = self.sandbox.check_write(path)?;

        // Create parent directories if needed
        if let Some(parent) = canonical.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(OdinError::Io)?;
        }

        tokio::fs::write(&canonical, &parsed.content)
            .await
            .map_err(OdinError::Io)?;

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(ToolResult {
            call_id: String::new(),
            tool_name: self.name.clone(),
            success: true,
            output: format!(
                "Successfully wrote {} bytes to {}",
                parsed.content.len(),
                canonical.display()
            ),
            error: None,
            duration_ms,
            timestamp: Utc::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_sandbox() -> Arc<Sandbox> {
        let boundary = odin_core::types::PathBoundary {
            allowed_read: vec!["/tmp".into()],
            allowed_write: vec!["/tmp".into()],
            denied: vec![],
        };
        Arc::new(Sandbox::new(boundary))
    }

    fn test_context() -> ToolContext {
        ToolContext {
            agent_id: Default::default(),
            session_id: Default::default(),
            working_dir: PathBuf::from("/tmp"),
            env: std::collections::HashMap::new(),
            resource_budgets: Default::default(),
        }
    }

    #[tokio::test]
    async fn test_file_read_write_roundtrip() {
        let sandbox = test_sandbox();
        let read = FileRead::new(sandbox.clone());
        let write = FileWrite::new(sandbox.clone());

        let test_path = PathBuf::from("/tmp/odin_test_file.txt");

        // Write
        let write_args = serde_json::json!({
            "path": test_path.to_string_lossy(),
            "content": "Hello, Raven!"
        });
        let write_result = write.execute(write_args, &test_context()).await.unwrap();
        assert!(write_result.success);
        assert!(write_result.output.contains("bytes"));

        // Read back
        let read_args = serde_json::json!({
            "path": test_path.to_string_lossy()
        });
        let read_result = read.execute(read_args, &test_context()).await.unwrap();
        assert!(read_result.success);
        assert_eq!(read_result.output, "Hello, Raven!");

        // Cleanup
        std::fs::remove_file(&test_path).ok();
    }

    #[tokio::test]
    async fn test_file_read_nonexistent() {
        let sandbox = test_sandbox();
        let read = FileRead::new(sandbox);
        let args = serde_json::json!({
            "path": "/tmp/does_not_exist_xyz.txt"
        });
        let result = read.execute(args, &test_context()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_file_write_denied() {
        let sandbox = Arc::new(Sandbox::new(odin_core::types::PathBoundary {
            allowed_read: vec!["/tmp".into()],
            allowed_write: vec!["/tmp".into()],
            denied: vec!["/etc".into()],
        }));
        let write = FileWrite::new(sandbox);
        let args = serde_json::json!({
            "path": "/etc/odin_forbidden.txt",
            "content": "should not work"
        });
        let result = write.execute(args, &test_context()).await;
        assert!(result.is_err());
    }
}
