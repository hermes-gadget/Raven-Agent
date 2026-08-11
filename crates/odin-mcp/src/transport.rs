//! Transport layer for MCP communication.
//!
//! The stdio implementation keeps one asynchronous buffered reader for the
//! lifetime of the child, drains stderr continuously, and correlates response
//! IDs so notifications and out-of-order messages cannot be lost.

use crate::error::{McpError, McpResult};
use crate::types::{JsonRpcRequest, JsonRpcResponse};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

const RESPONSE_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_RESPONSE_LINE_BYTES: usize = 1024 * 1024;

/// Transport abstraction for MCP communication.
#[async_trait]
pub trait McpTransport: Send + Sync {
    /// Send a JSON-RPC request and await the correlated response.
    async fn send_request(&self, method: &str, params: Option<Value>)
    -> McpResult<JsonRpcResponse>;

    /// Send a JSON-RPC notification. Notifications never receive a response.
    async fn send_notification(&self, method: &str, params: Option<Value>) -> McpResult<()>;

    /// Close the transport.
    async fn close(&mut self) -> McpResult<()>;
}

/// STDIO transport for MCP servers launched as child processes.
pub struct StdioTransport {
    child: Mutex<Option<Child>>,
    stdin: Mutex<Option<ChildStdin>>,
    stdout: Mutex<Option<BufReader<ChildStdout>>>,
    stderr_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
    request_id: AtomicU64,
    io_lock: Mutex<()>,
    command: String,
    args: Vec<String>,
    env: HashMap<String, String>,
}

impl StdioTransport {
    /// Create a new StdioTransport configuration.
    ///
    /// The process is not spawned until connect is called.
    pub fn new(command: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            child: Mutex::new(None),
            stdin: Mutex::new(None),
            stdout: Mutex::new(None),
            stderr_task: Mutex::new(None),
            request_id: AtomicU64::new(1),
            io_lock: Mutex::new(()),
            command: command.into(),
            args,
            env: HashMap::new(),
        }
    }

    /// Add configured environment variables to the MCP child process.
    pub fn with_env(mut self, env: HashMap<String, String>) -> Self {
        self.env = env;
        self
    }

    /// Spawn the child process and open communication channels.
    pub async fn connect(&self) -> McpResult<()> {
        let mut child = Command::new(&self.command)
            .args(&self.args)
            .envs(&self.env)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| {
                McpError::Connection(format!(
                    "Failed to spawn MCP server '{}': {error}",
                    self.command
                ))
            })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Connection("Failed to capture child stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Connection("Failed to capture child stdout".into()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| McpError::Connection("Failed to capture child stderr".into()))?;

        let stderr_task = tokio::spawn(async move {
            drain_stderr(stderr).await;
        });

        *self.child.lock().await = Some(child);
        *self.stdin.lock().await = Some(stdin);
        *self.stdout.lock().await = Some(BufReader::new(stdout));
        *self.stderr_task.lock().await = Some(stderr_task);
        Ok(())
    }

    async fn write_message(&self, message: Value) -> McpResult<()> {
        let encoded = serde_json::to_vec(&message).map_err(McpError::Serialization)?;
        let mut stdin = self.stdin.lock().await;
        let stdin = stdin
            .as_mut()
            .ok_or_else(|| McpError::Connection("Not connected".into()))?;
        stdin.write_all(&encoded).await.map_err(|error| {
            McpError::Transport(format!("Failed to write to child stdin: {error}"))
        })?;
        stdin.write_all(b"\n").await.map_err(|error| {
            McpError::Transport(format!("Failed to write newline to child stdin: {error}"))
        })?;
        stdin.flush().await.map_err(|error| {
            McpError::Transport(format!("Failed to flush child stdin: {error}"))
        })?;
        Ok(())
    }

    async fn read_response(&self, expected_id: u64) -> McpResult<JsonRpcResponse> {
        let mut stdout = self.stdout.lock().await;
        let reader = stdout
            .as_mut()
            .ok_or_else(|| McpError::Connection("Not connected".into()))?;

        loop {
            let line = read_bounded_line(reader).await?;
            let Some(line) = line else {
                return Err(McpError::Transport("MCP server closed connection".into()));
            };
            let value: Value = serde_json::from_slice(&line).map_err(|error| {
                McpError::InvalidResponse(format!("Failed to parse JSON-RPC response: {error}"))
            })?;

            // Notifications have no id. Other request responses may be
            // interleaved, so continue until the response for this request.
            if value.get("id").and_then(Value::as_u64) != Some(expected_id) {
                continue;
            }

            return serde_json::from_value(value).map_err(|error| {
                McpError::InvalidResponse(format!("Invalid JSON-RPC response: {error}"))
            });
        }
    }
}

async fn read_bounded_line<R: AsyncBufRead + Unpin>(reader: &mut R) -> McpResult<Option<Vec<u8>>> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf().await.map_err(|error| {
            McpError::Transport(format!("Failed to read child stdout: {error}"))
        })?;
        if available.is_empty() {
            return if line.is_empty() {
                Ok(None)
            } else {
                Err(McpError::InvalidResponse(
                    "MCP server closed in the middle of a response".into(),
                ))
            };
        }

        if let Some(newline) = available.iter().position(|byte| *byte == b'\n') {
            let count = newline + 1;
            if line.len() + count > MAX_RESPONSE_LINE_BYTES {
                reader.consume(count);
                return Err(McpError::InvalidResponse(format!(
                    "MCP response exceeds {} bytes",
                    MAX_RESPONSE_LINE_BYTES
                )));
            }
            line.extend_from_slice(&available[..newline]);
            reader.consume(count);
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            return Ok(Some(line));
        }

        if line.len() + available.len() > MAX_RESPONSE_LINE_BYTES {
            let count = available.len();
            reader.consume(count);
            return Err(McpError::InvalidResponse(format!(
                "MCP response exceeds {} bytes",
                MAX_RESPONSE_LINE_BYTES
            )));
        }
        line.extend_from_slice(available);
        let count = available.len();
        reader.consume(count);
    }
}

async fn drain_stderr<R: AsyncRead + Unpin>(mut stderr: R) {
    let mut buffer = [0_u8; 8192];
    loop {
        match stderr.read(&mut buffer).await {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
    }
}

#[async_trait]
impl McpTransport for StdioTransport {
    async fn send_request(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> McpResult<JsonRpcResponse> {
        let _io_guard = self.io_lock.lock().await;
        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        let request = JsonRpcRequest::new(id, method, params);
        let message = serde_json::to_value(request).map_err(McpError::Serialization)?;

        tokio::time::timeout(RESPONSE_TIMEOUT, async {
            self.write_message(message).await?;
            self.read_response(id).await
        })
        .await
        .map_err(|_| {
            McpError::Timeout(format!(
                "MCP request '{method}' timed out after {} seconds",
                RESPONSE_TIMEOUT.as_secs()
            ))
        })?
    }

    async fn send_notification(&self, method: &str, params: Option<Value>) -> McpResult<()> {
        let _io_guard = self.io_lock.lock().await;
        let mut message = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
        });
        if let Some(params) = params {
            message["params"] = params;
        }
        self.write_message(message).await
    }

    async fn close(&mut self) -> McpResult<()> {
        *self.stdin.lock().await = None;
        *self.stdout.lock().await = None;
        if let Some(task) = self.stderr_task.lock().await.take() {
            task.abort();
        }
        if let Some(mut child) = self.child.lock().await.take() {
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
        Ok(())
    }
}

impl Drop for StdioTransport {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.try_lock()
            && let Some(mut child) = child.take()
        {
            let _ = child.start_kill();
        }
        if let Ok(mut task) = self.stderr_task.try_lock()
            && let Some(task) = task.take()
        {
            task.abort();
        }
    }
}

/// A mock transport for unit testing McpClient without a real subprocess.
pub struct MockTransport {
    responses: HashMap<String, McpResult<JsonRpcResponse>>,
    sent_requests: std::sync::Mutex<Vec<(String, Option<Value>)>>,
    connected: std::sync::atomic::AtomicBool,
}

impl MockTransport {
    /// Create a new mock transport with the given method-to-response map.
    pub fn new(responses: HashMap<String, McpResult<JsonRpcResponse>>) -> Self {
        Self {
            responses,
            sent_requests: std::sync::Mutex::new(Vec::new()),
            connected: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Mark the transport as connected.
    pub fn set_connected(&self, connected: bool) {
        self.connected
            .store(connected, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get the list of requests and notifications sent through this transport.
    pub fn sent_requests(&self) -> Vec<(String, Option<Value>)> {
        self.sent_requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl McpTransport for MockTransport {
    async fn send_request(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> McpResult<JsonRpcResponse> {
        self.sent_requests
            .lock()
            .unwrap()
            .push((method.to_string(), params));

        if let Some(response) = self.responses.get(method) {
            match response {
                Ok(response) => Ok(response.clone()),
                Err(error) => Err(match error {
                    McpError::Protocol { code, message } => McpError::Protocol {
                        code: *code,
                        message: message.clone(),
                    },
                    _ => McpError::Transport(error.to_string()),
                }),
            }
        } else {
            Err(McpError::Protocol {
                code: -32601,
                message: format!("Method not found: {method}"),
            })
        }
    }

    async fn send_notification(&self, method: &str, params: Option<Value>) -> McpResult<()> {
        self.sent_requests
            .lock()
            .unwrap()
            .push((method.to_string(), params));
        Ok(())
    }

    async fn close(&mut self) -> McpResult<()> {
        self.connected
            .store(false, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}
