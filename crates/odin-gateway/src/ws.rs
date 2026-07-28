//! WebSocket gateway — real-time bidirectional communication.
//!
//! Provides:
//! - Persistent WebSocket connections for streaming responses
//! - JSON message protocol for task submission and results
//! - Ping/pong keepalive
//! - Connection pooling with broadcast support
//!
//! ## Protocol
//!
//! ### Client → Server
//! ```json
//! {"type": "task_submit", "payload": {"goal": "...", "max_iterations": 100}, "correlation_id": "..."}
//! {"type": "task_cancel", "payload": {"task_id": "..."}, "correlation_id": "..."}
//! {"type": "status_query", "correlation_id": "..."}
//! {"type": "ping"}
//! ```
//!
//! ### Server → Client
//! ```json
//! {"type": "task_started", "payload": {"task_id": "...", "goal": "..."}, "correlation_id": "..."}
//! {"type": "task_progress", "payload": {"task_id": "...", "iteration": 3, "confidence": 0.85, "phase": "ACT"}, "correlation_id": "..."}
//! {"type": "task_complete", "payload": {"task_id": "...", "success": true, "summary": "...", "iterations": 4, "confidence": 0.9}, "correlation_id": "..."}
//! {"type": "task_error", "payload": {"task_id": "...", "error": "..."}, "correlation_id": "..."}
//! {"type": "status", "payload": {"agents": 2, "sessions": 1, "uptime_secs": 3600}}
//! {"type": "pong"}
//! ```

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::IntoResponse;
use futures::{SinkExt, StreamExt};
use odin_core::error::OdinResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use tracing;

// ── Configuration ─────────────────────────────────────────────────────

/// Configuration for the WebSocket gateway.
#[derive(Debug, Clone)]
pub struct WsConfig {
    /// Whether the WebSocket gateway is enabled.
    pub enabled: bool,

    /// Listen address (used for logging; actual binding is via Axum on the HTTP port).
    pub addr: String,

    /// Max connections.
    pub max_connections: usize,

    /// Ping interval in seconds.
    pub ping_interval_secs: u64,

    /// Max message size in bytes.
    pub max_message_size: usize,
}

impl Default for WsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            addr: "127.0.0.1:9178".into(),
            max_connections: 100,
            ping_interval_secs: 30,
            max_message_size: 65536, // 64KB
        }
    }
}

// ── Message Types ─────────────────────────────────────────────────────

/// A WebSocket message in the Raven Agent protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsMessage {
    /// Message type.
    #[serde(rename = "type")]
    pub msg_type: String,

    /// Payload as a JSON value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,

    /// Optional correlation ID for request-response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
}

impl WsMessage {
    /// Create a new task-started message.
    pub fn task_started(task_id: &str, goal: &str, correlation_id: Option<String>) -> Self {
        Self {
            msg_type: "task_started".into(),
            payload: Some(serde_json::json!({
                "task_id": task_id,
                "goal": goal,
            })),
            correlation_id,
        }
    }

    /// Create a progress message.
    pub fn task_progress(task_id: &str, iteration: u32, confidence: f64, phase: &str) -> Self {
        Self {
            msg_type: "task_progress".into(),
            payload: Some(serde_json::json!({
                "task_id": task_id,
                "iteration": iteration,
                "confidence": confidence,
                "phase": phase,
            })),
            correlation_id: None,
        }
    }

    /// Create a completion message.
    pub fn task_complete(
        task_id: &str,
        success: bool,
        summary: &str,
        iterations: u32,
        confidence: f64,
        correlation_id: Option<String>,
    ) -> Self {
        Self {
            msg_type: "task_complete".into(),
            payload: Some(serde_json::json!({
                "task_id": task_id,
                "success": success,
                "summary": summary,
                "iterations": iterations,
                "confidence": confidence,
            })),
            correlation_id,
        }
    }

    /// Create an error message.
    pub fn task_error(task_id: &str, error: &str, correlation_id: Option<String>) -> Self {
        Self {
            msg_type: "task_error".into(),
            payload: Some(serde_json::json!({
                "task_id": task_id,
                "error": error,
            })),
            correlation_id,
        }
    }

    /// Create a status update message.
    pub fn status(
        agents: usize,
        sessions: usize,
        connected_clients: usize,
        uptime_secs: u64,
    ) -> Self {
        Self {
            msg_type: "status".into(),
            payload: Some(serde_json::json!({
                "agents": agents,
                "sessions": sessions,
                "connected_clients": connected_clients,
                "uptime_secs": uptime_secs,
            })),
            correlation_id: None,
        }
    }

    /// Create a pong response.
    pub fn pong() -> Self {
        Self {
            msg_type: "pong".into(),
            payload: None,
            correlation_id: None,
        }
    }

    // ── Orchestration Event Types ───────────────────────────────────

    /// Orchestration run started (decomposed goal into sub-tasks).
    pub fn orchestrate_started(
        run_id: &str,
        goal: &str,
        task_count: usize,
        workstream_count: usize,
        correlation_id: Option<String>,
    ) -> Self {
        Self {
            msg_type: "orchestrate_started".into(),
            payload: Some(serde_json::json!({
                "run_id": run_id,
                "goal": goal,
                "task_count": task_count,
                "workstream_count": workstream_count,
            })),
            correlation_id,
        }
    }

    /// A sub-task within an orchestration run has been assigned to an agent.
    pub fn orchestrate_task_assigned(
        run_id: &str,
        task_id: &str,
        agent_id: &str,
        goal: &str,
        correlation_id: Option<String>,
    ) -> Self {
        Self {
            msg_type: "orchestrate_task_assigned".into(),
            payload: Some(serde_json::json!({
                "run_id": run_id,
                "task_id": task_id,
                "agent_id": agent_id,
                "goal": goal,
            })),
            correlation_id,
        }
    }

    /// A sub-task within an orchestration run progressed.
    pub fn orchestrate_task_progress(
        run_id: &str,
        task_id: &str,
        phase: &str,
        iteration: u32,
        confidence: f64,
        correlation_id: Option<String>,
    ) -> Self {
        Self {
            msg_type: "orchestrate_task_progress".into(),
            payload: Some(serde_json::json!({
                "run_id": run_id,
                "task_id": task_id,
                "phase": phase,
                "iteration": iteration,
                "confidence": confidence,
            })),
            correlation_id,
        }
    }

    /// A sub-task completed successfully.
    pub fn orchestrate_task_complete(
        run_id: &str,
        task_id: &str,
        success: bool,
        summary: &str,
        correlation_id: Option<String>,
    ) -> Self {
        Self {
            msg_type: "orchestrate_task_complete".into(),
            payload: Some(serde_json::json!({
                "run_id": run_id,
                "task_id": task_id,
                "success": success,
                "summary": summary,
            })),
            correlation_id,
        }
    }

    /// A file lock was acquired by an agent.
    pub fn orchestrate_lock_acquired(
        run_id: &str,
        agent_id: &str,
        file_path: &str,
        lock_type: &str,
        correlation_id: Option<String>,
    ) -> Self {
        Self {
            msg_type: "orchestrate_lock_acquired".into(),
            payload: Some(serde_json::json!({
                "run_id": run_id,
                "agent_id": agent_id,
                "file_path": file_path,
                "lock_type": lock_type,
            })),
            correlation_id,
        }
    }

    /// A file lock was released by an agent.
    pub fn orchestrate_lock_released(
        run_id: &str,
        agent_id: &str,
        file_path: &str,
        correlation_id: Option<String>,
    ) -> Self {
        Self {
            msg_type: "orchestrate_lock_released".into(),
            payload: Some(serde_json::json!({
                "run_id": run_id,
                "agent_id": agent_id,
                "file_path": file_path,
            })),
            correlation_id,
        }
    }

    /// The entire orchestration run completed (all sub-tasks done).
    pub fn orchestrate_complete(
        run_id: &str,
        success: bool,
        total_tasks: usize,
        completed_tasks: usize,
        failed_tasks: usize,
        summary: &str,
        correlation_id: Option<String>,
    ) -> Self {
        Self {
            msg_type: "orchestrate_complete".into(),
            payload: Some(serde_json::json!({
                "run_id": run_id,
                "success": success,
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "summary": summary,
            })),
            correlation_id,
        }
    }
}

// ── Connection Manager ────────────────────────────────────────────────

/// Manages all active WebSocket connections and provides broadcast capability.
#[derive(Clone)]
pub struct WsConnectionManager {
    /// Active connections tracked by count (for capacity checks).
    connection_count: Arc<std::sync::atomic::AtomicUsize>,

    /// Broadcast channel — any message sent here goes to all connected clients.
    broadcast_tx: broadcast::Sender<WsMessage>,

    /// Start time for uptime calculation.
    start_time: Arc<std::time::Instant>,

    /// Optional orchestration store used to dispatch live control commands.
    control_store: Option<Arc<odin_orchestrator::persistence::SqliteOrchestrationStore>>,

    /// Optional shared secret required for remote control commands.
    control_token: Option<String>,
}

impl WsConnectionManager {
    /// Create a new connection manager.
    pub fn new(broadcast_capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(broadcast_capacity);
        Self {
            connection_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            broadcast_tx: tx,
            start_time: Arc::new(std::time::Instant::now()),
            control_store: None,
            control_token: None,
        }
    }

    /// Attach an orchestration store and optional control token for live dispatch.
    pub fn with_control(
        mut self,
        store: Arc<odin_orchestrator::persistence::SqliteOrchestrationStore>,
        control_token: Option<String>,
    ) -> Self {
        self.control_store = Some(store);
        self.control_token = control_token;
        self
    }

    /// Get the current uptime in seconds.
    pub fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Register a new connection — returns a broadcast receiver for this client.
    pub fn register(&self) -> broadcast::Receiver<WsMessage> {
        self.connection_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.broadcast_tx.subscribe()
    }

    /// Unregister a connection.
    pub fn unregister(&self) {
        let prev = self
            .connection_count
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        if prev == 0 {
            tracing::warn!("[WS] Unregister called with zero count (underflow?)");
        }
    }

    /// Broadcast a message to all connected clients.
    /// Returns the number of clients who received it.
    pub fn broadcast(&self, message: &WsMessage) -> usize {
        let count = self.broadcast_tx.receiver_count();
        if count > 0 {
            let _ = self.broadcast_tx.send(message.clone());
            tracing::debug!(
                "[WS] Broadcast message type '{}' to {count} clients",
                message.msg_type
            );
        }
        count
    }

    /// Get the number of connected clients.
    pub fn connection_count(&self) -> usize {
        self.connection_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Check if at connection limit.
    pub fn at_capacity(&self, config: &WsConfig) -> bool {
        self.connection_count() >= config.max_connections
    }

    /// Dispatch a live control command from an authorized WebSocket client.
    pub async fn dispatch_control(
        &self,
        kind: odin_orchestrator::RunControlKind,
        graph_id: &str,
        token: Option<&str>,
        reason: Option<String>,
        source: &str,
    ) -> Result<odin_orchestrator::RunControlCommand, String> {
        use odin_orchestrator::persistence::OrchestrationStore;
        use odin_orchestrator::{ControlAuth, RunControlCommand, authorize_control};

        match authorize_control(self.control_token.as_deref(), token) {
            ControlAuth::Allowed => {}
            ControlAuth::Denied(message) => return Err(message.to_string()),
        }
        let Some(store) = &self.control_store else {
            return Err("orchestration control store is not configured".into());
        };
        if graph_id.trim().is_empty() {
            return Err("graph_id is required".into());
        }
        // Reject control for unknown graphs so disconnected/stale owners fail closed.
        store
            .load_task_graph(graph_id)
            .await
            .map_err(|_| format!("graph '{graph_id}' not found"))?;
        let command = RunControlCommand::new(graph_id, kind, source, reason);
        store
            .enqueue_control(&command)
            .await
            .map_err(|error| format!("failed to enqueue control: {error}"))?;
        // Keep persisted status consistent with the live command.
        let status = match kind {
            odin_orchestrator::RunControlKind::Pause => "paused",
            odin_orchestrator::RunControlKind::Resume => "running",
            odin_orchestrator::RunControlKind::Cancel => "cancelled",
        };
        let _ = store.update_graph_status(graph_id, status).await;
        Ok(command)
    }
}

// ── WebSocket Handler ─────────────────────────────────────────────────

/// Axum handler for WebSocket upgrades.
/// Attach this to a route to enable WebSocket connections.
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    conn_mgr: Arc<WsConnectionManager>,
    config: Arc<WsConfig>,
) -> impl IntoResponse {
    if conn_mgr.at_capacity(&config) {
        return axum::response::Response::builder()
            .status(axum::http::StatusCode::SERVICE_UNAVAILABLE)
            .body(axum::body::Body::from("Max connections reached"))
            .unwrap();
    }

    ws.max_message_size(config.max_message_size)
        .on_upgrade(move |socket| handle_ws_connection(socket, conn_mgr))
}

/// Handle an individual WebSocket connection.
async fn handle_ws_connection(socket: WebSocket, conn_mgr: Arc<WsConnectionManager>) {
    let conn_id = uuid::Uuid::new_v4().to_string();
    let (mut sender, mut receiver) = socket.split();

    // Register for broadcasts
    let mut broadcast_rx = conn_mgr.register();

    tracing::info!(
        "[WS] Client connected: {conn_id} (total: {})",
        conn_mgr.connection_count()
    );

    // Send a welcome message
    let welcome = WsMessage {
        msg_type: "connected".into(),
        payload: Some(serde_json::json!({
            "connection_id": conn_id,
            "uptime_secs": conn_mgr.uptime_secs(),
            "active_connections": conn_mgr.connection_count(),
        })),
        correlation_id: None,
    };
    if let Ok(json) = serde_json::to_string(&welcome) {
        let _ = sender.send(Message::Text(json.into())).await;
    }

    // We need sender in both the broadcast forwarder and for pong responses.
    // Wrap sender in Arc<RwLock<Option<...>>> so both tasks can use it.
    let sender_ref = Arc::new(RwLock::new(Some(sender)));

    // Spawn broadcast forwarder — reads from broadcast channel, sends to this client
    let sender_for_broadcast = sender_ref.clone();
    let conn_id_for_task = conn_id.clone();
    let send_task = tokio::spawn(async move {
        loop {
            match broadcast_rx.recv().await {
                Ok(msg) => {
                    if let Ok(json) = serde_json::to_string(&msg) {
                        let mut s = sender_for_broadcast.write().await;
                        if let Some(ref mut s) = *s {
                            if s.send(Message::Text(json.into())).await.is_err() {
                                break; // Client disconnected
                            }
                        } else {
                            break;
                        }
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("[WS] Client {conn_id_for_task} lagged by {n} messages");
                    continue;
                }
                Err(broadcast::error::RecvError::Closed) => {
                    break;
                }
            }
        }
    });

    // Use conn_id_clone to avoid move issue with conn_id
    let conn_id_clone = conn_id.clone();

    // Read loop — handle incoming messages from client
    while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            Message::Text(text) => {
                if let Ok(ws_msg) = serde_json::from_str::<WsMessage>(&text) {
                    match ws_msg.msg_type.as_str() {
                        "ping" => {
                            let pong = WsMessage::pong();
                            if let Ok(json) = serde_json::to_string(&pong) {
                                let mut s = sender_ref.write().await;
                                if let Some(ref mut s) = *s {
                                    let _ = s.send(Message::Text(json.into())).await;
                                }
                            }
                        }
                        "task_submit" => {
                            conn_mgr.broadcast(&ws_msg);
                            tracing::info!("[WS] Task submitted by {conn_id_clone}");
                        }
                        "task_cancel" | "task_pause" | "task_resume" => {
                            let kind = match ws_msg.msg_type.as_str() {
                                "task_cancel" => odin_orchestrator::RunControlKind::Cancel,
                                "task_pause" => odin_orchestrator::RunControlKind::Pause,
                                _ => odin_orchestrator::RunControlKind::Resume,
                            };
                            let payload = ws_msg.payload.clone().unwrap_or(serde_json::json!({}));
                            let graph_id = payload
                                .get("graph_id")
                                .or_else(|| payload.get("run_id"))
                                .or_else(|| payload.get("task_id"))
                                .and_then(|value| value.as_str())
                                .unwrap_or_default()
                                .to_string();
                            let token = payload
                                .get("token")
                                .and_then(|value| value.as_str())
                                .map(str::to_string);
                            let reason = payload
                                .get("reason")
                                .and_then(|value| value.as_str())
                                .map(str::to_string);
                            match conn_mgr
                                .dispatch_control(
                                    kind,
                                    &graph_id,
                                    token.as_deref(),
                                    reason,
                                    &format!("ws:{conn_id_clone}"),
                                )
                                .await
                            {
                                Ok(command) => {
                                    let ack = WsMessage {
                                        msg_type: "control_ack".into(),
                                        payload: Some(serde_json::json!({
                                            "command_id": command.id,
                                            "graph_id": command.graph_id,
                                            "kind": command.kind.as_str(),
                                            "status": "enqueued",
                                        })),
                                        correlation_id: ws_msg.correlation_id.clone(),
                                    };
                                    conn_mgr.broadcast(&ack);
                                    tracing::info!(
                                        "[WS] Control {:?} enqueued for {} by {conn_id_clone}",
                                        kind,
                                        graph_id
                                    );
                                }
                                Err(error) => {
                                    let err = WsMessage {
                                        msg_type: "control_error".into(),
                                        payload: Some(serde_json::json!({
                                            "error": error,
                                            "graph_id": graph_id,
                                        })),
                                        correlation_id: ws_msg.correlation_id.clone(),
                                    };
                                    conn_mgr.broadcast(&err);
                                    tracing::warn!(
                                        "[WS] Control rejected from {conn_id_clone}: {error}"
                                    );
                                }
                            }
                        }
                        "status_query" => {
                            let status = WsMessage::status(
                                0,
                                0,
                                conn_mgr.connection_count(),
                                conn_mgr.uptime_secs(),
                            );
                            conn_mgr.broadcast(&status);
                        }
                        _ => {
                            tracing::debug!(
                                "[WS] Unknown message type from {conn_id_clone}: {}",
                                ws_msg.msg_type
                            );
                        }
                    }
                } else {
                    tracing::debug!("[WS] Invalid JSON received from {conn_id_clone}");
                }
            }
            Message::Close(_) => {
                break;
            }
            Message::Ping(data) => {
                let mut s = sender_ref.write().await;
                if let Some(ref mut s) = *s {
                    let _ = s.send(Message::Pong(data)).await;
                }
            }
            _ => {}
        }
    }

    // Cleanup
    send_task.abort();
    conn_mgr.unregister();
    tracing::info!(
        "[WS] Client disconnected: {conn_id} (total: {})",
        conn_mgr.connection_count()
    );
}

// ── Gateway (compatibility wrapper) ───────────────────────────────────

/// Compatibility wrapper for embedding the connection manager.
/// New code should attach [`ws_handler`] to its Axum router directly.
#[derive(Clone)]
pub struct WsGateway {
    /// Configuration.
    pub config: WsConfig,
    /// Connection manager.
    pub manager: Arc<WsConnectionManager>,
}

impl std::fmt::Debug for WsGateway {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WsGateway")
            .field("config", &self.config)
            .field("connections", &self.manager.connection_count())
            .finish()
    }
}

impl WsGateway {
    /// Create a new WebSocket gateway.
    pub fn new(config: WsConfig) -> Self {
        Self {
            config,
            manager: Arc::new(WsConnectionManager::new(256)),
        }
    }

    /// Start the WebSocket server.
    ///
    /// The actual WS upgrade handler must already be attached to an Axum router;
    /// this method only marks the embedded gateway lifecycle as started.
    pub async fn start(&self) -> OdinResult<()> {
        if !self.config.enabled {
            tracing::info!("[WS] Gateway disabled");
            return Ok(());
        }

        tracing::info!(
            "[WS] Gateway ready on {} (max {} connections, ping {}s)",
            self.config.addr,
            self.config.max_connections,
            self.config.ping_interval_secs,
        );

        Ok(())
    }

    /// Stop the WebSocket gateway.
    pub async fn stop(&self) -> OdinResult<()> {
        tracing::info!(
            "[WS] Gateway stopping ({} connections to close)",
            self.manager.connection_count()
        );
        Ok(())
    }

    /// Broadcast a message to all connected clients.
    pub async fn broadcast(&self, message: &WsMessage) -> OdinResult<()> {
        self.manager.broadcast(message);
        Ok(())
    }

    /// Get the number of connected clients.
    pub fn connection_count(&self) -> usize {
        self.manager.connection_count()
    }

    /// Get the connection manager (for embedding in Axum router).
    pub fn connection_manager(&self) -> Arc<WsConnectionManager> {
        self.manager.clone()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use odin_orchestrator::persistence::{OrchestrationStore, SqliteOrchestrationStore};
    use odin_orchestrator::task_graph::{TaskGraph, TaskGraphStatus};
    use odin_orchestrator::{RunControlKind, RunControlStatus};

    #[test]
    fn test_ws_config_default() {
        let config = WsConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.addr, "127.0.0.1:9178");
        assert_eq!(config.max_connections, 100);
        assert_eq!(config.ping_interval_secs, 30);
    }

    #[test]
    fn test_ws_gateway_creation() {
        let gateway = WsGateway::new(WsConfig::default());
        assert_eq!(gateway.connection_count(), 0);
    }

    #[tokio::test]
    async fn test_start_disabled_gateway() {
        let gateway = WsGateway::new(WsConfig::default());
        let result = gateway.start().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_start_enabled_gateway() {
        let config = WsConfig {
            enabled: true,
            ..Default::default()
        };
        let gateway = WsGateway::new(config);
        let result = gateway.start().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_stop_gateway() {
        let gateway = WsGateway::new(WsConfig::default());
        let result = gateway.stop().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_broadcast() {
        let gateway = WsGateway::new(WsConfig::default());
        let msg = WsMessage::pong();
        let result = gateway.broadcast(&msg).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_ws_message_serde() {
        let msg = WsMessage {
            msg_type: "task".into(),
            payload: Some(serde_json::json!({"goal": "write code"})),
            correlation_id: Some("req-1".into()),
        };

        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: WsMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.msg_type, "task");
        assert_eq!(
            deserialized.payload.unwrap()["goal"],
            serde_json::json!("write code")
        );
        assert_eq!(deserialized.correlation_id, Some("req-1".into()));
    }

    #[test]
    fn test_ws_message_task_started() {
        let msg = WsMessage::task_started("t1", "do thing", Some("corr-1".into()));
        assert_eq!(msg.msg_type, "task_started");
        let payload = msg.payload.unwrap();
        assert_eq!(payload["task_id"], "t1");
        assert_eq!(payload["goal"], "do thing");
    }

    #[test]
    fn test_ws_message_task_complete() {
        let msg = WsMessage::task_complete("t1", true, "done!", 5, 0.95, None);
        assert_eq!(msg.msg_type, "task_complete");
        let payload = msg.payload.unwrap();
        assert!(payload["success"].as_bool().unwrap());
        assert_eq!(payload["iterations"], 5);
    }

    #[test]
    fn test_ws_message_task_error() {
        let msg = WsMessage::task_error("t1", "boom", Some("corr-2".into()));
        assert_eq!(msg.msg_type, "task_error");
        let payload = msg.payload.unwrap();
        assert_eq!(payload["error"], "boom");
    }

    #[test]
    fn test_connection_manager_register_unregister() {
        let mgr = WsConnectionManager::new(16);
        assert_eq!(mgr.connection_count(), 0);

        let _rx = mgr.register();
        assert_eq!(mgr.connection_count(), 1);

        let _rx2 = mgr.register();
        assert_eq!(mgr.connection_count(), 2);

        mgr.unregister();
        assert_eq!(mgr.connection_count(), 1);

        mgr.unregister();
        assert_eq!(mgr.connection_count(), 0);
    }

    #[test]
    fn test_connection_manager_broadcast() {
        let mgr = WsConnectionManager::new(16);
        let msg = WsMessage::pong();
        // No receivers — broadcast returns 0
        let count = mgr.broadcast(&msg);
        assert_eq!(count, 0);

        // Register a receiver
        let mut rx = mgr.register();
        let count = mgr.broadcast(&msg);
        assert_eq!(count, 1);

        // Receiver should get the message
        let received = rx.try_recv();
        assert!(received.is_ok());
        assert_eq!(received.unwrap().msg_type, "pong");
    }

    #[test]
    fn test_at_capacity() {
        let mgr = WsConnectionManager::new(16);
        let config = WsConfig {
            max_connections: 1,
            ..Default::default()
        };

        assert!(!mgr.at_capacity(&config));
        let _rx = mgr.register();
        assert!(mgr.at_capacity(&config));
    }

    #[tokio::test]
    async fn websocket_control_requires_token_and_enqueues_for_known_graph() {
        let store = Arc::new(SqliteOrchestrationStore::new_in_memory().await.unwrap());
        store.initialize().await.unwrap();
        let mut graph = TaskGraph::new("controlled run");
        graph.status = TaskGraphStatus::Running;
        let graph_id = graph.id.to_string();
        store.save_task_graph(&graph).await.unwrap();

        let manager =
            WsConnectionManager::new(16).with_control(store.clone(), Some("secret".into()));
        let denied = manager
            .dispatch_control(
                RunControlKind::Cancel,
                &graph_id,
                Some("wrong"),
                None,
                "ws:test",
            )
            .await;
        assert!(denied.is_err());
        assert!(store.list_controls(&graph_id, 10).await.unwrap().is_empty());

        manager
            .dispatch_control(
                RunControlKind::Cancel,
                &graph_id,
                Some("secret"),
                Some("stop".into()),
                "ws:test",
            )
            .await
            .unwrap();
        let controls = store.list_controls(&graph_id, 10).await.unwrap();
        assert_eq!(controls.len(), 1);
        assert_eq!(controls[0].status, RunControlStatus::Pending);
        assert_eq!(
            store.load_task_graph(&graph_id).await.unwrap().status,
            TaskGraphStatus::Cancelled
        );
    }
}
