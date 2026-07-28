//! HTTP API server using Axum.
//!
//! Provides:
//! - `GET /health` — health check
//! - `POST /chat` — submit a task and receive results
//! - `GET /tools` — list all registered tools with schemas and capability tags
//! - `GET /tools/:name` — inspect one tool
//! - `POST /tools/validate` — run validation and return JSON report

use axum::{
    Json, Router,
    extract::{Path, Query},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use odin_core::config::ToolsConfig;
use odin_core::error::OdinResult;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing;

/// A boxed async handler for processing chat/task requests.
pub type TaskHandlerFn = Arc<
    dyn Fn(ChatRequest) -> Pin<Box<dyn Future<Output = OdinResult<ChatResponse>> + Send>>
        + Send
        + Sync,
>;

/// Shared state for the HTTP server.
#[derive(Clone)]
pub struct GatewayState {
    /// Optional handler for processing chat/task requests.
    pub task_handler: Option<TaskHandlerFn>,
    /// Whether the server has finished startup and is ready for traffic.
    pub ready: Arc<std::sync::atomic::AtomicBool>,
    /// Number of active tasks currently being processed.
    pub active_tasks: Arc<std::sync::atomic::AtomicU64>,
    /// Total tool calls since startup.
    pub total_tool_calls: Arc<std::sync::atomic::AtomicU64>,
    /// Total tool call errors since startup.
    pub total_tool_errors: Arc<std::sync::atomic::AtomicU64>,
    /// Total requests served.
    pub total_requests: Arc<std::sync::atomic::AtomicU64>,
    /// Optional WebSocket connection manager for broadcasting orchestration events.
    pub ws_manager: Option<Arc<crate::ws::WsConnectionManager>>,
    /// The live tool registry used by the task handler.
    pub tool_registry: Arc<odin_tools::ToolRegistry>,
    /// Correlated tool-call approval gate exposed by the approval API.
    pub approval_gate: Option<Arc<odin_permissions::ApprovalGate>>,
}

impl Default for GatewayState {
    fn default() -> Self {
        Self {
            task_handler: None,
            ready: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            active_tasks: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            total_tool_calls: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            total_tool_errors: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            total_requests: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            ws_manager: None,
            tool_registry: Arc::new(build_tool_registry(None)),
            approval_gate: None,
        }
    }
}

impl GatewayState {
    /// Mark the server as ready after all dependencies are loaded.
    pub fn mark_ready(&self) {
        self.ready.store(true, std::sync::atomic::Ordering::Release);
    }

    /// Check if the server is ready.
    pub fn is_ready(&self) -> bool {
        self.ready.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Broadcast an orchestration event to all connected WebSocket clients.
    pub fn broadcast_orchestration_event(&self, msg: &crate::ws::WsMessage) {
        if let Some(ref mgr) = self.ws_manager {
            let count = mgr.broadcast(msg);
            if count > 0 {
                tracing::debug!(
                    "[GATEWAY] Broadcast orchestration event '{}' to {count} WS clients",
                    msg.msg_type
                );
            }
        }
    }
}
// ── Request / Response Types ─────────────────────────────────────────

/// Incoming chat or task request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    /// The task goal or user message.
    pub task: String,

    /// Optional context for the task.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,

    /// Optional session ID for continuing a conversation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,

    /// Max iterations for this task.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_iterations: Option<u32>,
}

/// Chat or task response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// Whether the task was successful.
    pub success: bool,

    /// Summary of the result.
    pub summary: String,

    /// Number of iterations used.
    pub iterations: u32,

    /// Number of tool calls made.
    pub tool_calls: u32,

    /// Duration in milliseconds.
    pub duration_ms: u64,

    /// Confidence score (0.0 – 1.0).
    pub confidence: f64,

    /// Error message if unsuccessful.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Health check response with dependency status.
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_secs: u64,
    /// Whether all dependencies are loaded and the server is accepting traffic.
    pub ready: bool,
    /// Dependency statuses.
    pub dependencies: HealthDependencies,
}

/// Status of each dependency.
#[derive(Debug, Clone, Serialize)]
pub struct HealthDependencies {
    pub tools_loaded: bool,
    pub tool_count: usize,
    pub task_handler: bool,
}

// ── Tool API Response Types ──────────────────────────────────────────

/// A tool listed in the GET /tools response.
#[derive(Debug, Clone, Serialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub schema: odin_core::types::ToolSchema,
    pub is_safe: bool,
    pub requires_approval: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capability_tags: Vec<String>,
}

/// Response for GET /tools.
#[derive(Debug, Clone, Serialize)]
pub struct ToolsListResponse {
    pub total: usize,
    pub tools: Vec<ToolInfo>,
}

/// Aggregate validation report in JSON form.
#[derive(Debug, Clone, Serialize)]
pub struct ValidationReportResponse {
    pub passed: usize,
    pub failed: usize,
    pub total: usize,
    pub reports: Vec<odin_tools::ValidationReport>,
}

/// Doctor report response in JSON form.
#[derive(Debug, Clone, Serialize)]
pub struct DoctorReportResponse {
    pub healthy: bool,
    pub total_tools: usize,
    pub healthy_tools: usize,
    pub unhealthy_tools: usize,
    pub total_checks: usize,
    pub passed: usize,
    pub failed: usize,
    pub warnings: usize,
    pub tool_checks: Vec<odin_tools::ToolDoctorCheck>,
    pub ecosystem_checks: Vec<odin_tools::EcosystemCheck>,
}

/// Operator decision for a correlated pending tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalDecisionRequest {
    pub approved: bool,
    /// Must match the fingerprint returned with the pending request.
    pub argument_fingerprint: String,
}

/// Result of applying an approval decision.
#[derive(Debug, Clone, Serialize)]
pub struct ApprovalDecisionResponse {
    pub request_id: String,
    pub status: odin_permissions::ApprovalStatus,
}

// ── Route Handlers ───────────────────────────────────────────────────

/// Health check endpoint.
async fn health_handler(
    state: Arc<GatewayState>,
    start_time: Arc<std::time::Instant>,
) -> Json<HealthResponse> {
    let uptime = start_time.elapsed().as_secs();
    let tool_count = state.tool_registry.all_tools().len();
    Json(HealthResponse {
        status: if state.is_ready() { "ok" } else { "starting" }.into(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_secs: uptime,
        ready: state.is_ready(),
        dependencies: HealthDependencies {
            tools_loaded: tool_count > 0,
            tool_count,
            task_handler: state.task_handler.is_some(),
        },
    })
}

/// Metrics endpoint (Prometheus-compatible text format).
#[derive(Debug, Clone, Serialize)]
pub struct MetricsResponse {
    pub uptime_secs: u64,
    pub active_tasks: u64,
    pub total_requests: u64,
    pub total_tool_calls: u64,
    pub total_tool_errors: u64,
    pub tool_count: usize,
    pub tool_error_rate: f64,
}

async fn metrics_handler(
    state: Arc<GatewayState>,
    start_time: Arc<std::time::Instant>,
) -> Json<MetricsResponse> {
    let tool_calls = state
        .total_tool_calls
        .load(std::sync::atomic::Ordering::Acquire);
    let tool_errors = state
        .total_tool_errors
        .load(std::sync::atomic::Ordering::Acquire);
    let error_rate = if tool_calls > 0 {
        tool_errors as f64 / tool_calls as f64
    } else {
        0.0
    };

    Json(MetricsResponse {
        uptime_secs: start_time.elapsed().as_secs(),
        active_tasks: state
            .active_tasks
            .load(std::sync::atomic::Ordering::Acquire),
        total_requests: state
            .total_requests
            .load(std::sync::atomic::Ordering::Acquire),
        total_tool_calls: tool_calls,
        total_tool_errors: tool_errors,
        tool_count: state.tool_registry.all_tools().len(),
        tool_error_rate: error_rate,
    })
}

/// Chat/task endpoint.
async fn chat_handler(
    state: Arc<GatewayState>,
    start_time: Arc<std::time::Instant>,
    Json(request): Json<ChatRequest>,
) -> impl IntoResponse {
    match &state.task_handler {
        Some(handler) => match handler(request).await {
            Ok(response) => (StatusCode::OK, Json(response)).into_response(),
            Err(e) => {
                let error_resp = ChatResponse {
                    success: false,
                    summary: format!("Task execution failed: {e}"),
                    iterations: 0,
                    tool_calls: 0,
                    duration_ms: start_time.elapsed().as_millis() as u64,
                    confidence: 0.0,
                    error: Some(e.to_string()),
                };
                (StatusCode::INTERNAL_SERVER_ERROR, Json(error_resp)).into_response()
            }
        },
        None => {
            let error_resp = ChatResponse {
                success: false,
                summary: "No task handler configured".into(),
                iterations: 0,
                tool_calls: 0,
                duration_ms: 0,
                confidence: 0.0,
                error: Some("No task handler configured".into()),
            };
            (StatusCode::SERVICE_UNAVAILABLE, Json(error_resp)).into_response()
        }
    }
}

/// Build a tool registry with all built-in tools, filtered by
/// an optional [`ToolsConfig`]. When `config` is `None` (or the
/// enabled list is empty), all tools are registered.
fn build_tool_registry(config: Option<&ToolsConfig>) -> odin_tools::ToolRegistry {
    let registry = odin_tools::ToolRegistry::new();
    let sandbox = Arc::new(odin_tools::Sandbox::new(
        odin_core::types::PathBoundary::default(),
    ));

    // Helper to check whether a tool should be registered
    let tool_enabled = |name: &str| -> bool {
        let Some(tc) = config else {
            return true; // no config → all enabled
        };
        if !tc.enabled.is_empty() && !tc.enabled.iter().any(|e| e == name) {
            return false;
        }
        if tc.disabled.iter().any(|d| d == name) {
            return false;
        }
        true
    };

    macro_rules! try_reg {
        ($registry:expr, $tool:expr) => {
            if let Err(e) = $registry.register($tool) {
                tracing::warn!("[Gateway] Failed to register tool: {e}");
            }
        };
    }

    if tool_enabled("file_read") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::file::FileRead::new(sandbox.clone()))
        );
    }
    if tool_enabled("file_write") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::file::FileWrite::new(sandbox.clone()))
        );
    }
    if tool_enabled("shell") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::shell::Shell::new())
        );
    }
    if tool_enabled("web_fetch") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::web::WebFetch::new())
        );
    }
    if tool_enabled("web_search") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::web::WebSearch::new())
        );
    }
    if tool_enabled("http_request") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::web::HttpRequest::new())
        );
    }
    if tool_enabled("git") {
        try_reg!(registry, Box::new(odin_tools::builtins::git::Git::new()));
    }
    if tool_enabled("system_info") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::system::SystemInfo::new())
        );
    }
    if tool_enabled("disk_usage") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::system::DiskUsage::new())
        );
    }
    if tool_enabled("json_extract") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::data::JsonExtract::new())
        );
    }
    // Utility tools (Phase 4.0 expansion — 10 new tools)
    if tool_enabled("file_list") {
        try_reg!(registry, Box::new(odin_tools::builtins::utility::FileList));
    }
    if tool_enabled("file_delete") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::utility::FileDelete)
        );
    }
    if tool_enabled("file_exists") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::utility::FileExists)
        );
    }
    if tool_enabled("env_var") {
        try_reg!(registry, Box::new(odin_tools::builtins::utility::EnvVar));
    }
    if tool_enabled("time_now") {
        try_reg!(registry, Box::new(odin_tools::builtins::utility::TimeNow));
    }
    if tool_enabled("random_number") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::utility::RandomNumber)
        );
    }
    if tool_enabled("json_validate") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::utility::JsonValidate)
        );
    }
    if tool_enabled("text_search") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::utility::TextSearch)
        );
    }
    if tool_enabled("process_list") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::utility::ProcessList)
        );
    }
    if tool_enabled("network_ping") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::utility::NetworkPing)
        );
    }
    if tool_enabled("github_issue_create") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::github::GithubIssueCreate::new())
        );
    }
    if tool_enabled("github_issue_search") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::github::GithubIssueSearch::new())
        );
    }
    if tool_enabled("github_pr_create") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::github::GithubPrCreate::new())
        );
    }
    if tool_enabled("github_pr_status") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::github::GithubPrStatus::new())
        );
    }
    if tool_enabled("github_actions_status") {
        try_reg!(
            registry,
            Box::new(odin_tools::builtins::github::GithubActionsStatus::new())
        );
    }

    registry
}

/// GET /tools — list all registered tools with schemas and capability tags.
/// Supports ?tags=safe,read for filtering.
#[derive(Debug, Deserialize, Default)]
struct ToolsQuery {
    /// Comma-separated capability tags to filter by.
    #[serde(default)]
    tags: Option<String>,
}

async fn tools_list_handler(
    state: Arc<GatewayState>,
    Query(query): Query<ToolsQuery>,
) -> Json<ToolsListResponse> {
    let registry = &state.tool_registry;
    let schemas = registry.list_schemas();

    let filter_tags: Vec<String> = query
        .tags
        .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    let tools: Vec<ToolInfo> = schemas
        .into_iter()
        .filter_map(|schema| {
            let name = schema.function.name.clone();
            let tool = registry.get(&name)?;

            // Filter by tags if specified
            if !filter_tags.is_empty() {
                let tt = tool.capability_tags();
                if !filter_tags.iter().all(|ft| tt.contains(&ft.as_str())) {
                    return None;
                }
            }

            let is_safe = tool.is_safe();
            let requires_approval = tool.requires_approval();
            let capability_tags: Vec<String> = tool
                .capability_tags()
                .iter()
                .map(|s| s.to_string())
                .collect();

            Some(ToolInfo {
                name,
                description: tool.description().to_string(),
                schema,
                is_safe,
                requires_approval,
                capability_tags,
            })
        })
        .collect();

    let total = tools.len();
    Json(ToolsListResponse { total, tools })
}

/// GET /tools/:name — inspect one tool.
async fn tool_inspect_handler(
    state: Arc<GatewayState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let registry = &state.tool_registry;

    match registry.get(&name) {
        Some(tool) => {
            let schema = tool.schema();
            let capability_tags: Vec<String> = tool
                .capability_tags()
                .iter()
                .map(|s| s.to_string())
                .collect();
            let info = ToolInfo {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                schema,
                is_safe: tool.is_safe(),
                requires_approval: tool.requires_approval(),
                capability_tags,
            };
            (StatusCode::OK, Json(info)).into_response()
        }
        None => {
            let error = serde_json::json!({
                "error": format!("Tool '{}' not found", name)
            });
            (StatusCode::NOT_FOUND, Json(error)).into_response()
        }
    }
}

/// POST /tools/validate — run validation and return JSON report.
async fn tools_validate_handler(state: Arc<GatewayState>) -> Json<ValidationReportResponse> {
    let registry = &state.tool_registry;
    let reports = odin_tools::ToolValidator::validate_all(registry);

    let total = reports.len();
    let passed = reports.iter().filter(|r| r.failed.is_empty()).count();
    let failed = total - passed;

    Json(ValidationReportResponse {
        passed,
        failed,
        total,
        reports,
    })
}

/// POST /tools/doctor — run a comprehensive doctor check on all tools.
async fn tools_doctor_handler(state: Arc<GatewayState>) -> Json<DoctorReportResponse> {
    let registry = &state.tool_registry;
    let report = odin_tools::ToolDoctor::check(registry);

    Json(DoctorReportResponse {
        healthy: report.healthy,
        total_tools: report.summary.total_tools,
        healthy_tools: report.summary.healthy_tools,
        unhealthy_tools: report.summary.unhealthy_tools,
        total_checks: report.summary.total_checks,
        passed: report.summary.passed,
        failed: report.summary.failed,
        warnings: report.summary.warnings,
        tool_checks: report.tool_checks,
        ecosystem_checks: report.ecosystem_checks,
    })
}

/// GET /approvals — list redacted pending tool-call approvals.
async fn approvals_list_handler(
    state: Arc<GatewayState>,
) -> Json<Vec<odin_permissions::ApprovalRequest>> {
    match state.approval_gate.as_ref() {
        Some(gate) => Json(gate.pending_requests().await),
        None => Json(Vec::new()),
    }
}

/// POST /approvals/:id — approve or deny one exact pending call.
async fn approval_decision_handler(
    state: Arc<GatewayState>,
    Path(request_id): Path<String>,
    Json(decision): Json<ApprovalDecisionRequest>,
) -> impl IntoResponse {
    let Some(gate) = state.approval_gate.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "approval responder is not configured"})),
        )
            .into_response();
    };
    let Some(request) = gate.get_request(&request_id).await else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "approval request not found"})),
        )
            .into_response();
    };
    if request.status != odin_permissions::ApprovalStatus::Pending {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": "approval request is no longer pending"})),
        )
            .into_response();
    }

    let accepted = if decision.approved {
        gate.approve(&request_id, &decision.argument_fingerprint)
            .await
            .unwrap_or(false)
    } else {
        gate.deny(&request_id).await.unwrap_or(false)
    };
    let status = gate
        .get_request(&request_id)
        .await
        .map(|request| request.status)
        .unwrap_or(odin_permissions::ApprovalStatus::Denied);
    let response = ApprovalDecisionResponse { request_id, status };

    if accepted {
        (StatusCode::OK, Json(response)).into_response()
    } else {
        (StatusCode::CONFLICT, Json(response)).into_response()
    }
}

// ── Server ───────────────────────────────────────────────────────────

/// Run the HTTP server on the given address with graceful shutdown.
///
/// The `task_handler` is optional — if provided, it will be called
/// for every `/chat` request. Without one, the endpoint returns 503.
///
/// Listens for SIGTERM/SIGINT and drains active tasks before shutting down.
pub async fn run_http_server(
    addr: &str,
    task_handler: Option<TaskHandlerFn>,
    ws_manager: Option<Arc<crate::ws::WsConnectionManager>>,
    tool_registry: Option<Arc<odin_tools::ToolRegistry>>,
) -> OdinResult<()> {
    run_http_server_with_approvals(addr, task_handler, ws_manager, tool_registry, None).await
}

/// Run the HTTP server with a responder for correlated tool-call approvals.
pub async fn run_http_server_with_approvals(
    addr: &str,
    task_handler: Option<TaskHandlerFn>,
    ws_manager: Option<Arc<crate::ws::WsConnectionManager>>,
    tool_registry: Option<Arc<odin_tools::ToolRegistry>>,
    approval_gate: Option<Arc<odin_permissions::ApprovalGate>>,
) -> OdinResult<()> {
    let state: Arc<GatewayState> = Arc::new(GatewayState {
        task_handler,
        ws_manager,
        tool_registry: tool_registry.unwrap_or_else(|| Arc::new(build_tool_registry(None))),
        approval_gate,
        ..Default::default()
    });
    let start_time = Arc::new(std::time::Instant::now());

    // Mark server as ready after startup
    state.mark_ready();
    tracing::info!("[GATEWAY] Server ready — all dependencies loaded");

    let app = build_router(state.clone(), start_time.clone());

    let listener = TcpListener::bind(addr).await.map_err(|e| {
        odin_core::error::OdinError::Network(format!("Failed to bind to {addr}: {e}"))
    })?;

    tracing::info!("[GATEWAY] HTTP server listening on {addr}");

    // Graceful shutdown: drain active tasks on SIGTERM/SIGINT
    let shutdown_signal = graceful_shutdown_signal(state.clone());

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await
        .map_err(|e| odin_core::error::OdinError::Network(format!("Server error: {e}")))?;

    Ok(())
}

/// Signal handler for graceful shutdown: waits for SIGTERM/SIGINT, then
/// drains active tasks before returning.
async fn graceful_shutdown_signal(state: Arc<GatewayState>) {
    // Wait for shutdown signal
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigterm = signal(SignalKind::terminate()).ok();
        let mut sigint = signal(SignalKind::interrupt()).ok();
        tokio::select! {
            _ = async {
                if let Some(ref mut s) = sigterm { let _ = s.recv().await; }
                else { std::future::pending::<()>().await; }
            } => {},
            _ = async {
                if let Some(ref mut s) = sigint { let _ = s.recv().await; }
                else { std::future::pending::<()>().await; }
            } => {},
            _ = tokio::signal::ctrl_c() => {},
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }

    tracing::info!("[GATEWAY] Shutdown signal received, draining active tasks...");

    // Wait for active tasks to complete (with 30s timeout)
    let drain_start = std::time::Instant::now();
    loop {
        let active = state
            .active_tasks
            .load(std::sync::atomic::Ordering::Acquire);
        if active == 0 {
            break;
        }
        if drain_start.elapsed().as_secs() > 30 {
            tracing::warn!(
                "[GATEWAY] Draining timed out after 30s ({} active tasks remain)",
                active
            );
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    tracing::info!("[GATEWAY] Shutdown complete");
}

// ── Orchestration Helpers ─────────────────────────────────────────────

/// Get the Raven Agent state directory path.
fn dirs_state_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    std::path::PathBuf::from(home).join(".raven-agent")
}

/// Get a path within the Raven Agent state directory.
fn dirs_state_path(filename: &str) -> std::path::PathBuf {
    let base = dirs_state_dir();
    std::fs::create_dir_all(&base).ok();
    base.join(filename)
}

// ── Orchestration API Types ──────────────────────────────────────────

/// Request to orchestrate a goal with sub-agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrateRequest {
    /// The goal to decompose and orchestrate.
    pub goal: String,
    /// Max iterations per sub-agent.
    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,
}

fn default_max_iterations() -> u32 {
    100
}

/// Response from the orchestrate endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct OrchestrateResponse {
    /// The run ID for tracking this orchestration.
    pub run_id: String,
    /// The original goal.
    pub goal: String,
    /// Number of sub-tasks created.
    pub task_count: usize,
    /// Number of parallel workstreams detected.
    pub workstream_count: usize,
    /// The decomposed tasks.
    pub tasks: Vec<OrchestrateTaskInfo>,
    /// File lock summary.
    pub lock_summary: LockSummary,
}

/// Info about a single orchestrated task.
#[derive(Debug, Clone, Serialize)]
pub struct OrchestrateTaskInfo {
    pub label: String,
    pub goal: String,
    pub priority: u32,
    pub write_files: Vec<String>,
    pub read_files: Vec<String>,
    pub workstream_group: usize,
}

/// Summary of file lock state.
#[derive(Debug, Clone, Serialize)]
pub struct LockSummary {
    pub total_locked: usize,
    pub write_locked: usize,
    pub queued_writers: usize,
}

/// Status response for a specific orchestration run.
#[derive(Debug, Clone, Serialize)]
pub struct OrchestrateStatusResponse {
    pub run_id: String,
    pub goal: String,
    pub total_tasks: usize,
    pub tasks_done: usize,
    pub tasks_running: usize,
    pub tasks_failed: usize,
    pub conflicts: Vec<String>,
    pub complete: bool,
}

// ── Orchestration Handlers ───────────────────────────────────────────

/// POST /orchestrate — submit a goal for orchestration.
async fn orchestrate_handler(
    state: axum::extract::State<Arc<GatewayState>>,
    Json(body): Json<OrchestrateRequest>,
) -> impl IntoResponse {
    state
        .total_requests
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    use odin_orchestrator::Composer;
    use odin_orchestrator::persistence::{OrchestrationStore, SqliteOrchestrationStore};

    let mut composer = Composer::default();
    composer.intake(&body.goal);

    let mut graph = match composer.get_graph(&body.goal) {
        Some(g) => g.clone(),
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "Failed to create task graph"
                })),
            )
                .into_response();
        }
    };
    graph.status = odin_orchestrator::task_graph::TaskGraphStatus::Building;
    let run_id = graph.id;

    let groups = composer.detect_workstreams(&graph);

    // Persist the task graph to SQLite
    let db_path = dirs_state_path("orchestration.db");
    tracing::info!(run_id = %run_id, path = %db_path.display(), "Saving orchestration graph");
    let store = match SqliteOrchestrationStore::new(&db_path).await {
        Ok(store) => store,
        Err(error) => {
            tracing::error!(path = %db_path.display(), %error, "Failed to open orchestration store");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "Orchestration state is unavailable"})),
            )
                .into_response();
        }
    };
    if let Err(error) = store.initialize().await {
        tracing::error!(%error, "Failed to initialize orchestration store");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "Orchestration state could not be initialized"})),
        )
            .into_response();
    }
    if let Err(error) = store.save_task_graph(&graph).await {
        tracing::error!(%error, "Failed to persist orchestration graph");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "Orchestration plan could not be persisted"})),
        )
            .into_response();
    }
    tracing::info!(run_id = %run_id, "Orchestration graph saved");

    let tasks: Vec<OrchestrateTaskInfo> = graph
        .nodes
        .values()
        .map(|node| {
            let ws_group = groups
                .iter()
                .position(|g| g.contains(&node.id))
                .unwrap_or(0);
            OrchestrateTaskInfo {
                label: node.label.clone(),
                goal: node.goal.clone(),
                priority: node.priority,
                write_files: node.write_files.clone(),
                read_files: node.read_files.clone(),
                workstream_group: ws_group,
            }
        })
        .collect();

    let lock = composer.lock_summary();

    let response_goal = body.goal.clone();

    let response = OrchestrateResponse {
        run_id: run_id.to_string(),
        goal: body.goal,
        task_count: graph.nodes.len(),
        workstream_count: groups.len(),
        tasks,
        lock_summary: LockSummary {
            total_locked: lock.total_locked_files,
            write_locked: lock.write_locked_files,
            queued_writers: lock.queued_writers,
        },
    };

    // Broadcast orchestration started event to WebSocket clients
    let run_id_str = run_id.to_string();
    let task_count = graph.nodes.len();
    let ws_count = groups.len();
    state.broadcast_orchestration_event(&crate::ws::WsMessage::orchestrate_started(
        &run_id_str,
        &response_goal,
        task_count,
        ws_count,
        None,
    ));

    (StatusCode::OK, Json(response)).into_response()
}

/// GET /orchestrate/:id/status — check status of an orchestration run.
async fn orchestrate_status_handler(
    state: axum::extract::State<Arc<GatewayState>>,
    axum::extract::Path(run_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    state
        .total_requests
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    use odin_orchestrator::persistence::{OrchestrationStore, SqliteOrchestrationStore};

    let db_path = dirs_state_path("orchestration.db");
    let store = match SqliteOrchestrationStore::new(&db_path).await {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Failed to open store: {}", e)
                })),
            )
                .into_response();
        }
    };
    let _ = store.initialize().await;

    match store.load_task_graph(&run_id).await {
        Ok(graph) => {
            let total = graph.nodes.len();
            let done = graph
                .nodes
                .values()
                .filter(|n| n.status == odin_orchestrator::task_graph::TaskNodeStatus::Done)
                .count();
            let failed = graph
                .nodes
                .values()
                .filter(|n| n.status == odin_orchestrator::task_graph::TaskNodeStatus::Failed)
                .count();
            let running = total.saturating_sub(done + failed);
            let complete = matches!(
                graph.status,
                odin_orchestrator::task_graph::TaskGraphStatus::Complete
            );

            let response = OrchestrateStatusResponse {
                run_id: run_id.clone(),
                goal: graph.root_goal,
                total_tasks: total,
                tasks_done: done,
                tasks_running: running,
                tasks_failed: failed,
                conflicts: vec![],
                complete,
            };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(_) => {
            let response = OrchestrateStatusResponse {
                run_id,
                goal: "unknown".into(),
                total_tasks: 0,
                tasks_done: 0,
                tasks_running: 0,
                tasks_failed: 0,
                conflicts: vec![],
                complete: false,
            };
            (StatusCode::NOT_FOUND, Json(response)).into_response()
        }
    }
}

/// POST /orchestrate/:id/pause — pause an orchestration run.
async fn orchestrate_pause_handler(
    state: axum::extract::State<Arc<GatewayState>>,
    axum::extract::Path(run_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    state
        .total_requests
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    use odin_orchestrator::persistence::{OrchestrationStore, SqliteOrchestrationStore};

    let db_path = dirs_state_path("orchestration.db");
    match SqliteOrchestrationStore::new(&db_path).await {
        Ok(store) => {
            let _ = store.initialize().await;
            match store.update_graph_status(&run_id, "paused").await {
                Ok(()) => {
                    let command = odin_orchestrator::RunControlCommand::new(
                        &run_id,
                        odin_orchestrator::RunControlKind::Pause,
                        "http:orchestrate-pause",
                        None,
                    );
                    match store.enqueue_control(&command).await {
                        Ok(()) => (
                            StatusCode::OK,
                            Json(serde_json::json!({
                                "status": "paused",
                                "run_id": run_id,
                                "command_id": command.id
                            })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({
                                "error": format!("Failed to enqueue live pause: {}", e)
                            })),
                        )
                            .into_response(),
                    }
                }
                Err(e) => (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({
                        "error": format!("Not found: {}", e)
                    })),
                )
                    .into_response(),
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Store error: {}", e)
            })),
        )
            .into_response(),
    }
}

/// POST /orchestrate/:id/resume — resume a paused orchestration run.
async fn orchestrate_resume_handler(
    state: axum::extract::State<Arc<GatewayState>>,
    axum::extract::Path(run_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    state
        .total_requests
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    use odin_orchestrator::persistence::{OrchestrationStore, SqliteOrchestrationStore};

    let db_path = dirs_state_path("orchestration.db");
    match SqliteOrchestrationStore::new(&db_path).await {
        Ok(store) => {
            let _ = store.initialize().await;
            match store.update_graph_status(&run_id, "running").await {
                Ok(()) => {
                    let command = odin_orchestrator::RunControlCommand::new(
                        &run_id,
                        odin_orchestrator::RunControlKind::Resume,
                        "http:orchestrate-resume",
                        None,
                    );
                    match store.enqueue_control(&command).await {
                        Ok(()) => (
                            StatusCode::OK,
                            Json(serde_json::json!({
                                "status": "resumed",
                                "run_id": run_id,
                                "command_id": command.id
                            })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({
                                "error": format!("Failed to enqueue live resume: {}", e)
                            })),
                        )
                            .into_response(),
                    }
                }
                Err(e) => (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({
                        "error": format!("Not found: {}", e)
                    })),
                )
                    .into_response(),
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Store error: {}", e)
            })),
        )
            .into_response(),
    }
}

/// POST /orchestrate/:id/cancel — cancel an orchestration run.
async fn orchestrate_cancel_handler(
    state: axum::extract::State<Arc<GatewayState>>,
    axum::extract::Path(run_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    state
        .total_requests
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    use odin_orchestrator::persistence::{OrchestrationStore, SqliteOrchestrationStore};

    let db_path = dirs_state_path("orchestration.db");
    match SqliteOrchestrationStore::new(&db_path).await {
        Ok(store) => {
            let _ = store.initialize().await;
            match store.update_graph_status(&run_id, "cancelled").await {
                Ok(()) => {
                    let command = odin_orchestrator::RunControlCommand::new(
                        &run_id,
                        odin_orchestrator::RunControlKind::Cancel,
                        "http:orchestrate-cancel",
                        None,
                    );
                    match store.enqueue_control(&command).await {
                        Ok(()) => (
                            StatusCode::OK,
                            Json(serde_json::json!({
                                "status": "cancelled",
                                "run_id": run_id,
                                "command_id": command.id
                            })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({
                                "error": format!("Failed to enqueue live cancel: {}", e)
                            })),
                        )
                            .into_response(),
                    }
                }
                Err(e) => (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({
                        "error": format!("Not found: {}", e)
                    })),
                )
                    .into_response(),
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Store error: {}", e)
            })),
        )
            .into_response(),
    }
}

/// Build the Axum router, useful for embedding in larger apps.
pub fn build_router(state: Arc<GatewayState>, start_time: Arc<std::time::Instant>) -> Router {
    let mut router = Router::new()
        .route(
            "/health",
            get({
                let st = state.clone();
                let t0 = start_time.clone();
                move || health_handler(st.clone(), t0.clone())
            }),
        )
        .route(
            "/metrics",
            get({
                let st = state.clone();
                let t0 = start_time.clone();
                move || metrics_handler(st.clone(), t0.clone())
            }),
        )
        .route(
            "/chat",
            post({
                let st = state.clone();
                let t0 = start_time.clone();
                move |body| chat_handler(st.clone(), t0.clone(), body)
            }),
        )
        .route(
            "/tools",
            get({
                let st = state.clone();
                move |query| tools_list_handler(st.clone(), query)
            }),
        )
        .route(
            "/tools/{name}",
            get({
                let st = state.clone();
                move |path| tool_inspect_handler(st.clone(), path)
            }),
        )
        .route(
            "/tools/validate",
            post({
                let st = state.clone();
                move || tools_validate_handler(st.clone())
            }),
        )
        .route(
            "/tools/doctor",
            post({
                let st = state.clone();
                move || tools_doctor_handler(st.clone())
            }),
        )
        .route(
            "/approvals",
            get({
                let st = state.clone();
                move || approvals_list_handler(st.clone())
            }),
        )
        .route(
            "/approvals/{id}",
            post({
                let st = state.clone();
                move |path, body| approval_decision_handler(st.clone(), path, body)
            }),
        )
        .route(
            "/orchestrate",
            post({
                let st = state.clone();
                move |body| orchestrate_handler(axum::extract::State(st.clone()), body)
            }),
        )
        .route(
            "/orchestrate/{id}/status",
            get({
                let st = state.clone();
                move |path| orchestrate_status_handler(axum::extract::State(st.clone()), path)
            }),
        )
        .route(
            "/orchestrate/{id}/pause",
            post({
                let st = state.clone();
                move |path| orchestrate_pause_handler(axum::extract::State(st.clone()), path)
            }),
        )
        .route(
            "/orchestrate/{id}/resume",
            post({
                let st = state.clone();
                move |path| orchestrate_resume_handler(axum::extract::State(st.clone()), path)
            }),
        )
        .route(
            "/orchestrate/{id}/cancel",
            post({
                let st = state.clone();
                move |path| orchestrate_cancel_handler(axum::extract::State(st.clone()), path)
            }),
        );

    if let Some(manager) = state.ws_manager.clone() {
        let config = Arc::new(crate::ws::WsConfig {
            enabled: true,
            ..Default::default()
        });
        router = router.route(
            "/ws",
            get(move |ws| crate::ws::ws_handler(ws, manager.clone(), config.clone())),
        );
    }

    router
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .layer(tower_http::cors::CorsLayer::permissive())
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_request_serde() {
        let req = ChatRequest {
            task: "Write a test".into(),
            context: None,
            session_id: None,
            max_iterations: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        let deserialized: ChatRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.task, "Write a test");
    }

    #[test]
    fn test_chat_response_serde() {
        let resp = ChatResponse {
            success: true,
            summary: "Done".into(),
            iterations: 3,
            tool_calls: 5,
            duration_ms: 1000,
            confidence: 0.95,
            error: None,
        };

        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: ChatResponse = serde_json::from_str(&json).unwrap();
        assert!(deserialized.success);
        assert_eq!(deserialized.summary, "Done");
    }

    #[test]
    fn test_health_response_serde() {
        let resp = HealthResponse {
            status: "ok".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            uptime_secs: 42,
            ready: true,
            dependencies: HealthDependencies {
                tools_loaded: true,
                tool_count: 10,
                task_handler: true,
            },
        };

        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("ok"));
        assert!(json.contains("ready"));
        assert!(json.contains("dependencies"));
    }

    #[test]
    fn test_gateway_state_default() {
        let state = GatewayState::default();
        assert!(state.task_handler.is_none());
    }

    #[test]
    fn test_build_router_smoke() {
        let state = Arc::new(GatewayState::default());
        let start_time = Arc::new(std::time::Instant::now());
        let _router = build_router(state, start_time);
    }

    #[tokio::test]
    async fn test_handler_function() {
        let handler: TaskHandlerFn = Arc::new(|req: ChatRequest| {
            Box::pin(async move {
                Ok(ChatResponse {
                    success: true,
                    summary: format!("Handled: {}", req.task),
                    iterations: 1,
                    tool_calls: 0,
                    duration_ms: 0,
                    confidence: 1.0,
                    error: None,
                })
            })
        });

        let request = ChatRequest {
            task: "hello".into(),
            context: None,
            session_id: None,
            max_iterations: None,
        };

        let response = handler(request).await.unwrap();
        assert!(response.success);
        assert_eq!(response.summary, "Handled: hello");
    }

    #[test]
    fn test_tool_info_serde() {
        let info = ToolInfo {
            name: "file_read".into(),
            description: "Read file contents".into(),
            schema: odin_core::types::ToolSchema {
                schema_type: "function".into(),
                function: odin_core::types::FunctionSchema {
                    name: "file_read".into(),
                    description: "Read file contents".into(),
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                },
            },
            is_safe: true,
            requires_approval: false,
            capability_tags: vec!["filesystem".into(), "read".into()],
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("file_read"));
        assert!(json.contains("is_safe"));
        assert!(json.contains("filesystem"));
    }

    #[test]
    fn test_validation_report_response_serde() {
        let report = odin_tools::ValidationReport {
            tool_name: "test".into(),
            passed: vec!["name is non-empty".into()],
            failed: vec![],
            warnings: vec![],
            score: 1.0,
        };

        let resp = ValidationReportResponse {
            passed: 1,
            failed: 0,
            total: 1,
            reports: vec![report],
        };

        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("passed"));
        assert!(json.contains("test"));
    }

    #[test]
    fn test_tools_list_response_serde() {
        let resp = ToolsListResponse {
            total: 1,
            tools: vec![ToolInfo {
                name: "shell".into(),
                description: "Run shell commands".into(),
                schema: odin_core::types::ToolSchema {
                    schema_type: "function".into(),
                    function: odin_core::types::FunctionSchema {
                        name: "shell".into(),
                        description: "Run shell commands".into(),
                        parameters: serde_json::json!({"type": "object"}),
                    },
                },
                is_safe: false,
                requires_approval: true,
                capability_tags: vec!["dangerous".into()],
            }],
        };

        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("shell"));
        assert!(json.contains("dangerous"));
    }

    #[tokio::test]
    async fn remote_approval_handler_approves_correlated_call() {
        let gate = Arc::new(odin_permissions::ApprovalGate::new(false, 30));
        let request = gate
            .submit_request(
                uuid::Uuid::new_v4(),
                "shell".into(),
                r#"{"command":"echo ok"}"#.into(),
            )
            .await;
        let state = Arc::new(GatewayState {
            approval_gate: Some(gate.clone()),
            ..Default::default()
        });

        let response = approval_decision_handler(
            state,
            Path(request.id.clone()),
            Json(ApprovalDecisionRequest {
                approved: true,
                argument_fingerprint: request.argument_fingerprint,
            }),
        )
        .await
        .into_response();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            gate.get_request(&request.id).await.unwrap().status,
            odin_permissions::ApprovalStatus::Approved
        );
    }

    #[tokio::test]
    async fn remote_approval_handler_denies_correlated_call() {
        let gate = Arc::new(odin_permissions::ApprovalGate::new(false, 30));
        let request = gate
            .submit_request(uuid::Uuid::new_v4(), "file_write".into(), "{}".into())
            .await;
        let state = Arc::new(GatewayState {
            approval_gate: Some(gate.clone()),
            ..Default::default()
        });

        let response = approval_decision_handler(
            state,
            Path(request.id.clone()),
            Json(ApprovalDecisionRequest {
                approved: false,
                argument_fingerprint: request.argument_fingerprint,
            }),
        )
        .await
        .into_response();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            gate.get_request(&request.id).await.unwrap().status,
            odin_permissions::ApprovalStatus::Denied
        );
    }
}
