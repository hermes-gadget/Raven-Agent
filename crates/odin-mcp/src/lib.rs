//! odin-mcp — Model Context Protocol (MCP) client infrastructure.
//!
//! This crate provides the foundation for loading external tools from MCP
//! servers into Raven Agent. It includes:
//!
//! - **Transport**: `StdioTransport` for spawning MCP server subprocesses
//!   and communicating via stdin/stdout with JSON-RPC 2.0.
//! - **Client**: `McpClient` for the MCP protocol lifecycle (initialize,
//!   tools/list, tools/call, shutdown).
//! - **Tool Adapter**: `McpToolAdapter` that wraps MCP tool definitions
//!   as [`odin_core::traits::Tool`] implementations, enabling MCP tools
//!   to be registered in a [`odin_tools::tool::ToolRegistry`].
//! - **Config**: Types for configuring MCP server connections.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use odin_mcp::client::McpClient;
//! use odin_mcp::transport::StdioTransport;
//! use std::sync::Arc;
//! use tokio::sync::Mutex;
//!
//! async fn example() {
//!     let transport = Arc::new(Mutex::new(
//!         StdioTransport::new("npx", vec![
//!             "@modelcontextprotocol/server-filesystem".into(),
//!             "/tmp".into(),
//!         ])
//!     ));
//!
//!     // Spawn the configured child before performing the MCP handshake.
//!     transport.lock().await.connect().await.unwrap();
//!
//!     // Connect and initialize
//!     let mut client = McpClient::new(transport);
//!     client.connect().await.unwrap();
//!
//!     // Discover tools
//!     let tools = client.list_tools().await.unwrap();
//!     for tool in &tools {
//!         println!("MCP tool: {} - {}", tool.name, tool.description);
//!     }
//!
//!     client.close().await.unwrap();
//! }
//! ```

pub mod client;
pub mod error;
pub mod tool_adapter;
pub mod transport;
pub mod types;

/// MCP configuration types for Raven Agent configuration.
pub mod config {
    use serde::{Deserialize, Serialize};

    /// Configuration for a single MCP server connection.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct McpServerConfig {
        /// Human-readable name for this MCP server.
        pub name: String,

        /// The command to execute (e.g., "npx", "python", "node").
        pub command: String,

        /// Arguments to pass to the command.
        #[serde(default)]
        pub args: Vec<String>,

        /// Transport type: "stdio" or "http".
        #[serde(default = "default_transport_type")]
        pub transport_type: String,

        /// URL for HTTP transport (required when transport_type is "http").
        #[serde(skip_serializing_if = "Option::is_none")]
        pub url: Option<String>,

        /// Environment variables to set for the server process.
        #[serde(default)]
        pub env: std::collections::HashMap<String, String>,

        /// Whether this server is enabled.
        #[serde(default = "default_enabled")]
        pub enabled: bool,

        /// Capability tags to assign to all tools from this server.
        #[serde(default = "default_tags")]
        pub tags: Vec<String>,
    }

    fn default_transport_type() -> String {
        "stdio".into()
    }

    fn default_enabled() -> bool {
        true
    }

    fn default_tags() -> Vec<String> {
        vec!["mcp".into(), "external".into(), "safe".into()]
    }

    impl McpServerConfig {
        /// Create a new stdio-based MCP server config.
        pub fn new_stdio(
            name: impl Into<String>,
            command: impl Into<String>,
            args: Vec<String>,
        ) -> Self {
            Self {
                name: name.into(),
                command: command.into(),
                args,
                transport_type: "stdio".into(),
                url: None,
                env: std::collections::HashMap::new(),
                enabled: true,
                tags: default_tags(),
            }
        }

        /// Create a new HTTP-based MCP server config.
        pub fn new_http(name: impl Into<String>, url: impl Into<String>) -> Self {
            Self {
                name: name.into(),
                command: String::new(),
                args: vec![],
                transport_type: "http".into(),
                url: Some(url.into()),
                env: std::collections::HashMap::new(),
                enabled: true,
                tags: default_tags(),
            }
        }
    }
}
