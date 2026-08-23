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
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{Mutex, oneshot};

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
    stdout_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
    stderr_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
    pending: std::sync::Arc<Mutex<HashMap<u64, oneshot::Sender<McpResult<JsonRpcResponse>>>>>,
    request_id: AtomicU64,
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
            stdout_task: Mutex::new(None),
            stderr_task: Mutex::new(None),
            pending: std::sync::Arc::new(Mutex::new(HashMap::new())),
            request_id: AtomicU64::new(1),
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
        let pending = self.pending.clone();
        let stdout_task = tokio::spawn(async move {
            dispatch_responses(BufReader::new(stdout), pending).await;
        });

        *self.child.lock().await = Some(child);
        *self.stdin.lock().await = Some(stdin);
        *self.stdout_task.lock().await = Some(stdout_task);
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
}

async fn dispatch_responses<R: AsyncBufRead + Unpin>(
    mut reader: R,
    pending: std::sync::Arc<Mutex<HashMap<u64, oneshot::Sender<McpResult<JsonRpcResponse>>>>>,
) {
    loop {
        let line = match read_bounded_line(&mut reader).await {
            Ok(Some(line)) => line,
            Ok(None) => {
                fail_pending(&pending, "MCP server closed connection").await;
                break;
            }
            Err(error) => {
                fail_pending(&pending, &error.to_string()).await;
                continue;
            }
        };
        let value: Value = match serde_json::from_slice(&line) {
            Ok(value) => value,
            Err(error) => {
                fail_pending(
                    &pending,
                    &format!("Failed to parse JSON-RPC response: {error}"),
                )
                .await;
                continue;
            }
        };
        let Some(response_id) = value.get("id").and_then(Value::as_u64) else {
            continue;
        };
        let response = match serde_json::from_value(value) {
            Ok(response) => response,
            Err(error) => {
                if let Some(sender) = pending.lock().await.remove(&response_id) {
                    let _ = sender.send(Err(McpError::InvalidResponse(format!(
                        "Invalid JSON-RPC response: {error}"
                    ))));
                }
                continue;
            }
        };
        if let Some(sender) = pending.lock().await.remove(&response_id) {
            let _ = sender.send(Ok(response));
        } else {
            tracing::warn!(
                response_id,
                "Received MCP response without a pending request"
            );
        }
    }
}

async fn fail_pending(
    pending: &Mutex<HashMap<u64, oneshot::Sender<McpResult<JsonRpcResponse>>>>,
    message: &str,
) {
    for (_, sender) in pending.lock().await.drain() {
        let _ = sender.send(Err(McpError::Transport(message.to_string())));
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
            drain_through_newline(reader).await?;
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

async fn drain_through_newline<R: AsyncBufRead + Unpin>(reader: &mut R) -> McpResult<()> {
    loop {
        let available = reader.fill_buf().await.map_err(|error| {
            McpError::Transport(format!("Failed to drain child stdout: {error}"))
        })?;
        if available.is_empty() {
            return Ok(());
        }
        let count = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |newline| newline + 1);
        let drained_line = available.get(count - 1) == Some(&b'\n');
        reader.consume(count);
        if drained_line {
            return Ok(());
        }
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
        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        let request = JsonRpcRequest::new(id, method, params);
        let message = serde_json::to_value(request).map_err(McpError::Serialization)?;
        let (response_tx, response_rx) = oneshot::channel();
        self.pending.lock().await.insert(id, response_tx);
        if let Err(error) = self.write_message(message).await {
            self.pending.lock().await.remove(&id);
            return Err(error);
        }

        match tokio::time::timeout(RESPONSE_TIMEOUT, response_rx).await {
            Ok(Ok(response)) => response,
            Ok(Err(_)) => Err(McpError::Connection(
                "MCP response dispatcher stopped".into(),
            )),
            Err(_) => {
                self.pending.lock().await.remove(&id);
                Err(McpError::Timeout(format!(
                    "MCP request '{method}' timed out after {} seconds",
                    RESPONSE_TIMEOUT.as_secs()
                )))
            }
        }
    }

    async fn send_notification(&self, method: &str, params: Option<Value>) -> McpResult<()> {
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
        if let Some(task) = self.stdout_task.lock().await.take() {
            task.abort();
        }
        fail_pending(&self.pending, "MCP transport closed").await;
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
        if let Ok(mut task) = self.stdout_task.try_lock()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn oversized_frame_is_drained_before_the_next_response() {
        let (reader, mut writer) = tokio::io::duplex(8 * 1024);
        let writer_task = tokio::spawn(async move {
            writer
                .write_all(&vec![b'x'; MAX_RESPONSE_LINE_BYTES + 1])
                .await
                .unwrap();
            writer
                .write_all(b"\n{\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{}}\n")
                .await
                .unwrap();
        });
        let mut reader = BufReader::new(reader);

        assert!(matches!(
            read_bounded_line(&mut reader).await,
            Err(McpError::InvalidResponse(message)) if message.contains("exceeds")
        ));
        let next = read_bounded_line(&mut reader).await.unwrap().unwrap();
        let response: JsonRpcResponse = serde_json::from_slice(&next).unwrap();
        assert_eq!(response.id, 7);
        writer_task.await.unwrap();
    }

    #[tokio::test]
    async fn dispatcher_routes_interleaved_responses_by_id() {
        let (reader, mut writer) = tokio::io::duplex(4096);
        let pending = std::sync::Arc::new(Mutex::new(HashMap::new()));
        let (first_tx, first_rx) = oneshot::channel();
        let (second_tx, second_rx) = oneshot::channel();
        pending.lock().await.insert(1, first_tx);
        pending.lock().await.insert(2, second_tx);
        let dispatcher = tokio::spawn(dispatch_responses(BufReader::new(reader), pending));

        writer
            .write_all(
                b"{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"value\":2}}\n\
                  {\"jsonrpc\":\"2.0\",\"method\":\"progress\"}\n\
                  {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"value\":1}}\n",
            )
            .await
            .unwrap();

        assert_eq!(
            second_rx.await.unwrap().unwrap().result.unwrap()["value"],
            2
        );
        assert_eq!(first_rx.await.unwrap().unwrap().result.unwrap()["value"], 1);
        drop(writer);
        dispatcher.await.unwrap();
    }
}
