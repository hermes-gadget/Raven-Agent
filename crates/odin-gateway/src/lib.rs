//! Raven Agent gateway — HTTP, WebSocket, and Discord interfaces.
//!
//! Provides HTTP, Discord, and WebSocket interfaces for interacting
//! with the Raven agent system.

pub mod discord;
pub mod http;
pub mod ws;

pub use discord::DiscordConfig;
pub use discord::DiscordGateway;
pub use http::{
    ApprovalDecisionRequest, ApprovalDecisionResponse, ChatRequest, ChatResponse,
    DoctorReportResponse, GatewayState, HealthDependencies, HealthResponse, LockSummary,
    MetricsResponse, OrchestrateRequest, OrchestrateResponse, OrchestrateStatusResponse,
    OrchestrateTaskInfo, PendingApprovalResponse, TaskHandlerFn, ToolInfo, ToolsListResponse,
    ValidationReportResponse, build_management_router, build_router, run_http_server,
    run_http_server_with_auth, run_http_server_with_auth_and_budgets,
    run_http_server_with_management, run_http_server_with_management_auth,
    run_http_server_with_management_auth_and_budgets,
};
