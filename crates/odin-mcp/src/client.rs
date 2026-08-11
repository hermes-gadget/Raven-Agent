//! MCP client — connects to MCP servers, discovers tools, and calls them.
//!
//! The [`McpClient`] wraps a [`McpTransport`] and handles the MCP protocol
//! lifecycle: initialize, tools/list, tools/call, and shutdown.

use crate::error::{McpError, McpResult};
use crate::transport::McpTransport;
use crate::types::{
    JsonRpcResponse, McpContentItem, McpInitializeResult, McpToolCallParams, McpToolDef,
    McpToolListResult, McpToolResult,
};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

/// A client for communicating with an MCP (Model Context Protocol) server.
///
/// Provides methods to initialize a session, discover tools, and invoke them.
pub struct McpClient {
    /// The underlying transport (e.g., stdio or HTTP/SSE).
    transport: Arc<Mutex<dyn McpTransport>>,
    /// Whether the MCP initialize handshake has completed.
    initialized: bool,
    /// Server name (from initialize response).
    server_name: Option<String>,
    /// Server version (from initialize response).
    server_version: Option<String>,
}

impl McpClient {
    /// Create a new MCP client wrapping the given transport.
    ///
    /// The client is not connected until [`connect`](#method.connect) is called.
    pub fn new(transport: Arc<Mutex<dyn McpTransport>>) -> Self {
        Self {
            transport,
            initialized: false,
            server_name: None,
            server_version: None,
        }
    }

    /// Connect to the MCP server by performing the initialize handshake.
    ///
    /// Sends an `initialize` request with the client's protocol version and
    /// capabilities, then waits for the server's response.
    pub async fn connect(&mut self) -> McpResult<()> {
        if self.initialized {
            return Err(McpError::AlreadyInitialized);
        }

        let params = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "raven-agent",
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        let response = self.send_request_inner("initialize", Some(params)).await?;

        let result: McpInitializeResult =
            serde_json::from_value(response.result.ok_or_else(|| {
                McpError::InvalidResponse("Missing result in initialize response".into())
            })?)
            .map_err(|e| McpError::InvalidResponse(format!("Invalid initialize result: {e}")))?;

        self.server_name = Some(result.server_info.name);
        self.server_version = Some(result.server_info.version);
        self.initialized = true;

        // Send initialized notification (no response expected).
        let _ = self
            .send_notification_inner("notifications/initialized", None)
            .await;

        Ok(())
    }

    /// List all tools available from the MCP server.
    ///
    /// Calls the `tools/list` method and returns the tool definitions.
    /// The client must be initialized before calling this.
    pub async fn list_tools(&self) -> McpResult<Vec<McpToolDef>> {
        self.ensure_initialized()?;

        let response = self.send_request_inner("tools/list", None).await?;

        let result: McpToolListResult =
            serde_json::from_value(response.result.ok_or_else(|| {
                McpError::InvalidResponse("Missing result in tools/list response".into())
            })?)
            .map_err(|e| McpError::InvalidResponse(format!("Invalid tools/list result: {e}")))?;

        Ok(result.tools)
    }

    /// Call a tool exposed by the MCP server.
    ///
    /// The `name` must match one of the tools returned by [`list_tools`](#method.list_tools).
    /// `args` is a JSON object with the tool's input parameters.
    pub async fn call_tool(&self, name: &str, args: Value) -> McpResult<McpToolResult> {
        self.ensure_initialized()?;

        let params = McpToolCallParams::new(name, args);
        let response = self
            .send_request_inner("tools/call", Some(serde_json::to_value(params)?))
            .await?;

        let result: McpToolResult = serde_json::from_value(response.result.ok_or_else(|| {
            McpError::InvalidResponse("Missing result in tools/call response".into())
        })?)
        .map_err(|e| McpError::InvalidResponse(format!("Invalid tools/call result: {e}")))?;

        Ok(result)
    }

    /// Close the connection to the MCP server.
    pub async fn close(&mut self) -> McpResult<()> {
        let mut transport = self.transport.lock().await;
        transport.close().await?;
        self.initialized = false;
        Ok(())
    }

    /// Get the server name, if initialized.
    pub fn server_name(&self) -> Option<&str> {
        self.server_name.as_deref()
    }

    /// Get the server version, if initialized.
    pub fn server_version(&self) -> Option<&str> {
        self.server_version.as_deref()
    }

    /// Whether the client has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    // ── Internal Helpers ────────────────────────────────────────────

    fn ensure_initialized(&self) -> McpResult<()> {
        if !self.initialized {
            return Err(McpError::NotInitialized);
        }
        Ok(())
    }

    async fn send_request_inner(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> McpResult<JsonRpcResponse> {
        let transport = self.transport.lock().await;
        let response = transport.send_request(method, params).await?;

        // Check for JSON-RPC error in the response
        if let Some(err) = response.error {
            return Err(McpError::Protocol {
                code: err.code,
                message: err.message,
            });
        }

        Ok(response)
    }

    async fn send_notification_inner(&self, method: &str, params: Option<Value>) -> McpResult<()> {
        let transport = self.transport.lock().await;
        transport.send_notification(method, params).await
    }
}

/// Helper to flatten MCP tool results into a single string.
///
/// Concatenates all text content items from the result.
pub fn flatten_result(result: &McpToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|item| match item {
            McpContentItem::Text { text } => Some(text.clone()),
            McpContentItem::Resource { text, .. } => text.clone(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::{MockTransport, StdioTransport};
    use crate::types::{JsonRpcError, JsonRpcResponse};

    fn make_response(id: u64, result: Option<Value>) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result,
            error: None,
        }
    }

    fn make_error_response(id: u64, code: i64, message: &str) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }

    #[tokio::test]
    async fn test_connect_success() {
        let mut responses = std::collections::HashMap::new();
        responses.insert(
            "initialize".into(),
            Ok(make_response(
                1,
                Some(serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": { "listChanged": false }
                    },
                    "serverInfo": {
                        "name": "test-server",
                        "version": "1.0.0"
                    }
                })),
            )),
        );
        responses.insert(
            "notifications/initialized".into(),
            Ok(make_response(2, None)),
        );

        let mock = Arc::new(Mutex::new(MockTransport::new(responses)));
        // Connect the mock
        mock.lock().await.set_connected(true);

        let mut client = McpClient::new(mock.clone());
        client.connect().await.unwrap();

        assert!(client.is_initialized());
        assert_eq!(client.server_name(), Some("test-server"));
        assert_eq!(client.server_version(), Some("1.0.0"));

        // Cleanup
        client.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_list_tools() {
        let mut responses = std::collections::HashMap::new();
        responses.insert(
            "initialize".into(),
            Ok(make_response(
                1,
                Some(serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": { "name": "ts", "version": "1" }
                })),
            )),
        );
        responses.insert(
            "notifications/initialized".into(),
            Ok(make_response(2, None)),
        );
        responses.insert(
            "tools/list".into(),
            Ok(make_response(
                3,
                Some(serde_json::json!({
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo back input",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "message": {
                                        "type": "string",
                                        "description": "Message to echo"
                                    }
                                },
                                "required": ["message"]
                            }
                        }
                    ]
                })),
            )),
        );

        let mock = Arc::new(Mutex::new(MockTransport::new(responses)));
        mock.lock().await.set_connected(true);

        let mut client = McpClient::new(mock);
        client.connect().await.unwrap();

        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");
        assert_eq!(tools[0].description, "Echo back input");
    }

    #[tokio::test]
    async fn test_call_tool() {
        let mut responses = std::collections::HashMap::new();
        responses.insert(
            "initialize".into(),
            Ok(make_response(
                1,
                Some(serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": { "name": "ts", "version": "1" }
                })),
            )),
        );
        responses.insert(
            "notifications/initialized".into(),
            Ok(make_response(2, None)),
        );
        responses.insert(
            "tools/call".into(),
            Ok(make_response(
                3,
                Some(serde_json::json!({
                    "content": [
                        { "type": "text", "text": "hello back" }
                    ],
                    "isError": false
                })),
            )),
        );

        let mock = Arc::new(Mutex::new(MockTransport::new(responses)));
        mock.lock().await.set_connected(true);

        let mut client = McpClient::new(mock);
        client.connect().await.unwrap();

        let result = client
            .call_tool("echo", serde_json::json!({"message": "hello"}))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert_eq!(result.content.len(), 1);
    }

    #[tokio::test]
    async fn test_not_initialized_error() {
        let responses = std::collections::HashMap::new();
        let mock = Arc::new(Mutex::new(MockTransport::new(responses)));

        let client = McpClient::new(mock);
        let err = client.list_tools().await.unwrap_err();
        assert!(matches!(err, McpError::NotInitialized));
    }

    #[tokio::test]
    async fn test_connect_twice_fails() {
        let mut responses = std::collections::HashMap::new();
        responses.insert(
            "initialize".into(),
            Ok(make_response(
                1,
                Some(serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": { "name": "ts", "version": "1" }
                })),
            )),
        );
        responses.insert(
            "notifications/initialized".into(),
            Ok(make_response(2, None)),
        );

        let mock = Arc::new(Mutex::new(MockTransport::new(responses)));
        mock.lock().await.set_connected(true);

        let mut client = McpClient::new(mock);
        client.connect().await.unwrap();
        let err = client.connect().await.unwrap_err();
        assert!(matches!(err, McpError::AlreadyInitialized));

        client.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_list_tools_handles_protocol_error() {
        let mut responses = std::collections::HashMap::new();
        responses.insert(
            "initialize".into(),
            Ok(make_response(
                1,
                Some(serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": { "name": "ts", "version": "1" }
                })),
            )),
        );
        responses.insert(
            "notifications/initialized".into(),
            Ok(make_response(2, None)),
        );
        responses.insert(
            "tools/list".into(),
            Ok(make_error_response(3, -32601, "Method not found")),
        );

        let mock = Arc::new(Mutex::new(MockTransport::new(responses)));
        mock.lock().await.set_connected(true);

        let mut client = McpClient::new(mock);
        client.connect().await.unwrap();

        let err = client.list_tools().await.unwrap_err();
        match err {
            McpError::Protocol { code, message } => {
                assert_eq!(code, -32601);
                assert!(message.contains("Method not found"));
            }
            _ => panic!("Expected Protocol error, got: {err}"),
        }

        client.close().await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_stdio_transport_handles_notifications_batches_and_stderr() {
        let server = r#"
while IFS= read -r line; do
  case "$line" in
    *notifications/initialized*)
      :
      ;;
    *initialize*)
      i=0
      while [ "$i" -lt 10000 ]; do
        printf 'stderr-noise\n' >&2
        i=$((i + 1))
      done
      printf '%s\n' '{"jsonrpc":"2.0","method":"server/notification"}'
      printf '%s\n' '{"jsonrpc":"2.0","id":99,"result":{"ignored":true}}'
      printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"mock","version":"1"}}}'
      ;;
    *tools/list*)
      printf '%s\n' '{"jsonrpc":"2.0","method":"notifications/tools/list_changed"}'
      printf '%s\n' '{"jsonrpc":"2.0","id":98,"result":{"tools":[]}}'
      printf '%s\n' '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"echo","description":"Echo","inputSchema":{"type":"object"}}]}}'
      ;;
  esac
done
"#;
        let transport = Arc::new(Mutex::new(StdioTransport::new(
            "sh",
            vec!["-c".into(), server.into()],
        )));
        transport.lock().await.connect().await.unwrap();

        let mut client = McpClient::new(transport);
        client.connect().await.unwrap();
        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools[0].name, "echo");
        client.close().await.unwrap();
    }
}
