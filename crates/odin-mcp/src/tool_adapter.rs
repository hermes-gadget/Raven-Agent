//! Adapter that wraps MCP tools as [`odin_core::traits::Tool`] implementations.
//!
//! [`McpToolAdapter`] bridges between MCP tool definitions discovered from
//! a server and Raven Agent's tool trait system, allowing MCP tools to be
//! registered in a [`ToolRegistry`] alongside built-in tools.

use crate::client::McpClient;
use crate::error::McpError;
use crate::types::McpToolDef;
use async_trait::async_trait;
use chrono::Utc;
use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::{Tool, ToolContext};
use odin_core::types::{FunctionSchema, ToolResult, ToolSchema};
use std::sync::Arc;
use tokio::sync::Mutex;

/// An adapter that wraps an MCP server tool as a Raven Agent [`Tool`].
///
/// When executed, it forwards the call to the MCP server via the shared
/// [`McpClient`] and converts the response into a [`ToolResult`].
pub struct McpToolAdapter {
    /// The definition from the MCP server's tools/list response.
    def: McpToolDef,
    /// Shared reference to the client managing the MCP connection.
    client: Arc<Mutex<McpClient>>,
    /// Capability tags for this tool.
    tags: Vec<String>,
    /// Whether this tool requires user approval.
    requires_approval: bool,
    /// Whether this tool is safe.
    safe: bool,
}

impl McpToolAdapter {
    /// Create a new adapter wrapping an MCP tool definition.
    ///
    /// The `client` is shared across all tools from the same MCP server.
    /// External tools are unsafe and approval-gated by default because MCP
    /// discovery does not include a portable safety classification.
    pub fn new(def: McpToolDef, client: Arc<Mutex<McpClient>>) -> Self {
        Self {
            tags: vec!["mcp".into(), "external".into(), "dangerous".into()],
            requires_approval: true,
            safe: false,
            def,
            client,
        }
    }

    /// Create a new adapter with custom capability tags.
    pub fn new_with_tags(
        def: McpToolDef,
        client: Arc<Mutex<McpClient>>,
        tags: Vec<String>,
    ) -> Self {
        Self {
            tags,
            requires_approval: true,
            safe: false,
            def,
            client,
        }
    }

    /// Mark this tool as requiring user approval.
    pub fn with_approval(mut self, requires: bool) -> Self {
        self.requires_approval = requires;
        self
    }

    /// Mark this tool as safe or unsafe.
    pub fn with_safety(mut self, safe: bool) -> Self {
        self.safe = safe;
        self.tags.retain(|tag| tag != "safe" && tag != "dangerous");
        self.tags.push(if safe {
            "safe".into()
        } else {
            "dangerous".into()
        });
        self
    }

    /// Convert an MCP `inputSchema` (JSON Schema) into the Raven Agent
    /// [`ToolSchema`] format expected by models.
    ///
    /// This simply maps the tool's name and description with the raw
    /// `inputSchema` as the parameters object.
    fn convert_schema(def: &McpToolDef) -> ToolSchema {
        // Use the inputSchema directly as the parameters, or fall back
        // to an empty object schema.
        let parameters =
            if def.input_schema.is_null() || def.input_schema == serde_json::Value::Null {
                serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                })
            } else {
                def.input_schema.clone()
            };

        ToolSchema {
            schema_type: "function".into(),
            function: FunctionSchema {
                name: def.name.clone(),
                description: def.description.clone(),
                parameters,
            },
        }
    }

    /// Get the name of the underlying MCP tool.
    pub fn tool_name(&self) -> &str {
        &self.def.name
    }

    /// Get the description of the underlying MCP tool.
    pub fn tool_description(&self) -> &str {
        &self.def.description
    }
}

#[async_trait]
impl Tool for McpToolAdapter {
    fn name(&self) -> &str {
        &self.def.name
    }

    fn description(&self) -> &str {
        &self.def.description
    }

    fn schema(&self) -> ToolSchema {
        Self::convert_schema(&self.def)
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _context: &ToolContext,
    ) -> OdinResult<ToolResult> {
        let start = std::time::Instant::now();

        let client = self.client.lock().await;
        let mcp_result = client
            .call_tool(&self.def.name, args.clone())
            .await
            .map_err(|e| map_mcp_error(&self.def.name, e))?;

        let duration_ms = start.elapsed().as_millis() as u64;

        // Flatten the MCP content items into a single output string
        let output = crate::client::flatten_result(&mcp_result);

        Ok(ToolResult {
            call_id: format!("mcp-{}", self.def.name),
            tool_name: self.def.name.clone(),
            success: !mcp_result.is_error,
            output,
            error: if mcp_result.is_error {
                Some("MCP tool returned an error".into())
            } else {
                None
            },
            duration_ms,
            timestamp: Utc::now(),
        })
    }

    fn requires_approval(&self) -> bool {
        self.requires_approval
    }

    fn is_safe(&self) -> bool {
        self.safe
    }

    fn capability_tags(&self) -> Vec<String> {
        self.tags.clone()
    }

    fn is_dangerous(&self) -> bool {
        !self.safe
    }

    fn validate_args(&self, args: &serde_json::Value) -> OdinResult<()> {
        let schema = self.schema();
        // MCP permits a null params value for no-argument tools. Validate it
        // as an empty object only when the advertised schema has no required
        // fields; schemas with required fields must still reject null.
        let value = if args.is_null()
            && schema
                .function
                .parameters
                .get("type")
                .and_then(serde_json::Value::as_str)
                == Some("object")
            && schema
                .function
                .parameters
                .get("required")
                .and_then(serde_json::Value::as_array)
                .is_none_or(Vec::is_empty)
        {
            serde_json::json!({})
        } else {
            args.clone()
        };
        odin_core::traits::validate_json_schema(&value, &schema.function.parameters).map_err(
            |error| OdinError::Validation(format!("MCP tool '{}': {error}", self.def.name)),
        )
    }
}

/// Map MCP errors to the shared internal error type.
fn map_mcp_error(tool_name: &str, err: McpError) -> OdinError {
    match err {
        McpError::Transport(msg) | McpError::Connection(msg) => OdinError::Network(msg),
        McpError::Serialization(e) => OdinError::Validation(format!("Serialization error: {e}")),
        McpError::Protocol { code, message } => {
            OdinError::tool(tool_name, format!("Protocol error ({code}): {message}"))
        }
        McpError::Timeout(msg) => OdinError::Timeout(msg),
        McpError::Tool { tool, message } => OdinError::tool(tool, message),
        McpError::InvalidResponse(msg) => {
            OdinError::tool(tool_name, format!("Invalid response: {msg}"))
        }
        McpError::AlreadyInitialized => OdinError::tool(tool_name, "Server already initialized"),
        McpError::NotInitialized => {
            OdinError::tool(tool_name, "Server not initialized".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MockTransport;
    use std::collections::HashMap;

    #[test]
    fn test_schema_conversion() {
        let def = McpToolDef {
            name: "test_tool".into(),
            description: "A test MCP tool".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to process"
                    }
                },
                "required": ["message"]
            }),
        };

        let schema = McpToolAdapter::convert_schema(&def);
        assert_eq!(schema.schema_type, "function");
        assert_eq!(schema.function.name, "test_tool");
        assert_eq!(schema.function.description, "A test MCP tool");
        assert_eq!(
            schema.function.parameters.get("required"),
            Some(&serde_json::json!(["message"]))
        );
    }

    #[test]
    fn test_schema_conversion_with_empty_input_schema() {
        let def = McpToolDef {
            name: "no_schema".into(),
            description: "Tool without input schema".into(),
            input_schema: serde_json::Value::Null,
        };

        let schema = McpToolAdapter::convert_schema(&def);
        assert_eq!(schema.function.name, "no_schema");
        assert_eq!(
            schema.function.parameters,
            serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            })
        );
    }

    #[test]
    fn test_schema_conversion_preserves_full_schema() {
        let def = McpToolDef {
            name: "complex".into(),
            description: "Complex tool".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "count": { "type": "integer", "minimum": 0 },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" }
                    }
                },
                "required": ["name"],
                "additionalProperties": false
            }),
        };

        let schema = McpToolAdapter::convert_schema(&def);
        assert_eq!(schema.function.name, "complex");

        // Verify the full structure is preserved
        let params = &schema.function.parameters;
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["name"]["type"].as_str() == Some("string"));
        assert!(params["properties"]["count"]["type"].as_str() == Some("integer"));
        assert!(params["properties"]["tags"]["type"].as_str() == Some("array"));
        assert_eq!(params["required"].as_array().unwrap()[0], "name");
        assert_eq!(params["additionalProperties"], false);
    }

    #[test]
    fn test_adapter_name_and_description() {
        let def = McpToolDef {
            name: "my_tool".into(),
            description: "My description".into(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        };

        // For name/desc tests we create an adapter without a client
        // (the execute method needs one, but name/desc don't)
        let adapter = McpToolAdapter::new_with_tags(
            def,
            Arc::new(Mutex::new(McpClient::new(Arc::new(Mutex::new(
                MockTransport::new(std::collections::HashMap::new()),
            ))))),
            vec!["mcp".into(), "custom".into()],
        );

        assert_eq!(adapter.name(), "my_tool");
        assert_eq!(adapter.description(), "My description");
    }

    #[test]
    fn test_adapter_validation() {
        let def = McpToolDef {
            name: "validated".into(),
            description: "Needs object args".into(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        };

        let adapter = McpToolAdapter::new_with_tags(
            def,
            Arc::new(Mutex::new(McpClient::new(Arc::new(Mutex::new(
                MockTransport::new(std::collections::HashMap::new()),
            ))))),
            vec![],
        );

        // Object args should be valid
        assert!(
            adapter
                .validate_args(&serde_json::json!({"key": "value"}))
                .is_ok()
        );

        // Null args should be valid (some MCP tools accept no args)
        assert!(adapter.validate_args(&serde_json::Value::Null).is_ok());

        // Non-object args should fail
        assert!(
            adapter
                .validate_args(&serde_json::json!("string_arg"))
                .is_err()
        );
        assert!(adapter.validate_args(&serde_json::json!(42)).is_err());
        assert!(
            adapter
                .validate_args(&serde_json::json!([1, 2, 3]))
                .is_err()
        );
    }

    #[test]
    fn test_adapter_validation_enforces_mcp_schema() {
        let def = McpToolDef {
            name: "strict".into(),
            description: "Strict arguments".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "required": ["count"],
                "additionalProperties": false,
                "properties": {"count": {"type": "integer", "minimum": 1, "maximum": 3}}
            }),
        };
        let adapter = McpToolAdapter::new(
            def,
            Arc::new(Mutex::new(McpClient::new(Arc::new(Mutex::new(
                MockTransport::new(HashMap::new()),
            ))))),
        );

        assert!(
            adapter
                .validate_args(&serde_json::json!({"count": 2}))
                .is_ok()
        );
        assert!(
            adapter
                .validate_args(&serde_json::json!({"count": 4}))
                .is_err()
        );
        assert!(
            adapter
                .validate_args(&serde_json::json!({"count": 2, "extra": true}))
                .is_err()
        );
        assert!(adapter.validate_args(&serde_json::Value::Null).is_err());
    }

    #[test]
    fn test_adapter_default_tags() {
        let def = McpToolDef {
            name: "tagged".into(),
            description: "Has tags".into(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        };

        let adapter = McpToolAdapter::new(
            def,
            Arc::new(Mutex::new(McpClient::new(Arc::new(Mutex::new(
                MockTransport::new(std::collections::HashMap::new()),
            ))))),
        );

        // Unknown external tools fail closed by default.
        assert!(!adapter.is_safe());
        assert!(adapter.is_dangerous());
        assert!(adapter.requires_approval());
        assert!(
            adapter
                .capability_tags()
                .iter()
                .any(|tag| tag == "dangerous")
        );
    }

    #[test]
    fn test_adapter_custom_safety() {
        let def = McpToolDef {
            name: "dangerous".into(),
            description: "Risky tool".into(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        };

        let adapter = McpToolAdapter::new(
            def,
            Arc::new(Mutex::new(McpClient::new(Arc::new(Mutex::new(
                MockTransport::new(std::collections::HashMap::new()),
            ))))),
        )
        .with_approval(true)
        .with_safety(false);

        assert!(!adapter.is_safe());
        assert!(adapter.is_dangerous());
        assert!(adapter.requires_approval());
    }

    #[test]
    fn test_mcp_failure_classes_are_preserved() {
        assert!(matches!(
            map_mcp_error("tool", McpError::Transport("closed".into())),
            OdinError::Network(_)
        ));
        assert!(matches!(
            map_mcp_error("tool", McpError::Timeout("late".into())),
            OdinError::Timeout(_)
        ));
        assert!(matches!(
            map_mcp_error(
                "tool",
                McpError::Tool {
                    tool: "tool".into(),
                    message: "failed".into(),
                }
            ),
            OdinError::Tool { .. }
        ));
    }
}
