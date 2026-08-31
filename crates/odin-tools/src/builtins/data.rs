//! Data manipulation tools — JSON extraction, transformation utilities.
//!
//! Provides tools for working with structured data. Currently supports
//! dot-path extraction from JSON values.

use std::time::Instant;

use async_trait::async_trait;
use chrono::Utc;
use serde::Deserialize;
use tracing::instrument;

use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::{Tool, ToolContext};
use odin_core::types::{FunctionSchema, ToolResult, ToolSchema};

/// Arguments for `json_extract`.
#[derive(Debug, Deserialize)]
struct JsonExtractArgs {
    /// The JSON string to extract from.
    input: String,
    /// Dot-path query, e.g. "users.0.name" or "data.items.1.title".
    query: String,
}

/// Tool that extracts values from a JSON string using a simple dot-path query.
///
/// Supports array indices (e.g., `users.0.name`) and nested object access
/// (e.g., `data.metadata.author`).
pub struct JsonExtract {
    name: String,
    description: String,
}

impl JsonExtract {
    /// Create a new `JsonExtract` tool.
    pub fn new() -> Self {
        Self {
            name: "json_extract".into(),
            description: "Extract a value from a JSON string using a dot-path query (e.g. 'users.0.name'). Supports nested objects and array indices. Safe read-only data transformation tool.".into(),
        }
    }

    fn make_schema(name: &str) -> ToolSchema {
        ToolSchema {
            schema_type: "function".into(),
            function: FunctionSchema {
                name: name.into(),
                description: "Extract a value from JSON using dot-path query.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "The JSON string to extract from"
                        },
                        "query": {
                            "type": "string",
                            "description": "Dot-path query like 'users.0.name' or 'data.metadata.author'"
                        }
                    },
                    "required": ["input", "query"]
                }),
            },
        }
    }
}

impl Default for JsonExtract {
    fn default() -> Self {
        Self::new()
    }
}

/// Resolve a dot-path against a parsed JSON value.
///
/// Supports:
/// - `key.subkey` — nested object access
/// - `key.0.field` — array index access
/// - `key.0` — array element access (returns the whole element)
fn resolve_path(mut value: &serde_json::Value, parts: &[&str]) -> OdinResult<serde_json::Value> {
    for (i, part) in parts.iter().enumerate() {
        // Try array index first
        if let Ok(idx) = part.parse::<usize>() {
            match value {
                serde_json::Value::Array(arr) => {
                    value = arr.get(idx).ok_or_else(|| OdinError::Tool {
                        tool: "json_extract".into(),
                        message: format!(
                            "Array index {idx} out of bounds (length {}) at path segment {i}",
                            arr.len()
                        ),
                        source: None,
                    })?;
                }
                _ => {
                    return Err(OdinError::Tool {
                        tool: "json_extract".into(),
                        message: format!(
                            "Cannot index into non-array value with '{part}' at path segment {i}"
                        ),
                        source: None,
                    });
                }
            }
        } else {
            // Object key access
            match value {
                serde_json::Value::Object(map) => {
                    value = map.get(*part).ok_or_else(|| OdinError::Tool {
                        tool: "json_extract".into(),
                        message: format!("Key '{part}' not found at path segment {i}"),
                        source: None,
                    })?;
                }
                _ => {
                    return Err(OdinError::Tool {
                        tool: "json_extract".into(),
                        message: format!(
                            "Cannot access key '{part}' on non-object value at path segment {i}"
                        ),
                        source: None,
                    });
                }
            }
        }
    }
    Ok(value.clone())
}

#[async_trait]
impl Tool for JsonExtract {
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
        vec!["data".into(), "transform".into(), "safe".into()]
    }

    #[instrument(skip(self, _context), fields(tool = self.name))]
    async fn execute(
        &self,
        args: serde_json::Value,
        _context: &ToolContext,
    ) -> OdinResult<ToolResult> {
        let start = Instant::now();

        let parsed: JsonExtractArgs =
            serde_json::from_value(args).map_err(|e| OdinError::Tool {
                tool: self.name.clone(),
                message: format!("Invalid arguments: {e}"),
                source: Some(Box::new(e)),
            })?;

        // Parse the input JSON
        let json_value: serde_json::Value =
            serde_json::from_str(&parsed.input).map_err(|e| OdinError::Tool {
                tool: self.name.clone(),
                message: format!("Invalid JSON input: {e}"),
                source: Some(Box::new(e)),
            })?;

        // Split query into segments
        let parts: Vec<&str> = parsed.query.split('.').collect();
        let result = resolve_path(&json_value, &parts)?;

        // Serialize the result back
        let output = serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string());

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(ToolResult {
            call_id: String::new(),
            tool_name: self.name.clone(),
            success: true,
            output,
            error: None,
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
    async fn test_json_extract_nested_object() {
        let tool = JsonExtract::new();
        let result = tool
            .execute(
                serde_json::json!({
                    "input": r#"{"users":[{"name":"Alice"},{"name":"Bob"}]}"#,
                    "query": "users.1.name"
                }),
                &test_context(),
            )
            .await
            .unwrap();
        assert!(result.success);
        assert_eq!(result.output.trim(), "\"Bob\"");
    }

    #[tokio::test]
    async fn test_json_extract_missing_key() {
        let tool = JsonExtract::new();
        let result = tool
            .execute(
                serde_json::json!({
                    "input": r#"{"a":1}"#,
                    "query": "b"
                }),
                &test_context(),
            )
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_json_extract_invalid_json() {
        let tool = JsonExtract::new();
        let result = tool
            .execute(
                serde_json::json!({
                    "input": "not valid json",
                    "query": "key"
                }),
                &test_context(),
            )
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid JSON"));
    }

    #[tokio::test]
    async fn test_json_extract_array_out_of_bounds() {
        let tool = JsonExtract::new();
        let result = tool
            .execute(
                serde_json::json!({
                    "input": r#"[1,2,3]"#,
                    "query": "5"
                }),
                &test_context(),
            )
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of bounds"));
    }

    #[tokio::test]
    async fn test_json_extract_top_level_array() {
        let tool = JsonExtract::new();
        let result = tool
            .execute(
                serde_json::json!({
                    "input": r#"["a","b","c"]"#,
                    "query": "1"
                }),
                &test_context(),
            )
            .await
            .unwrap();
        assert!(result.success);
        assert_eq!(result.output.trim(), "\"b\"");
    }
}
