//! Configuration types for Raven Agent.

use crate::types::{PathBoundary, PermissionRule};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Top-level configuration for a Raven Agent instance.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OdinConfig {
    /// General settings
    #[serde(default)]
    pub general: GeneralConfig,

    /// Model/provider configuration
    #[serde(default)]
    pub models: ModelsConfig,

    /// Agent loop configuration
    #[serde(default)]
    pub agent: AgentConfig,

    /// Tool configuration
    #[serde(default)]
    pub tools: ToolsConfig,

    /// Memory configuration
    #[serde(default)]
    pub memory: MemoryConfig,

    /// Safety and permissions
    #[serde(default)]
    pub safety: SafetyConfig,

    /// Audit/logging configuration
    #[serde(default)]
    pub audit: AuditConfig,

    /// Gateway configuration
    #[serde(default)]
    pub gateway: GatewayConfig,

    /// Scheduler configuration
    #[serde(default)]
    pub scheduler: SchedulerConfig,

    /// Central resource and payload budgets shared by all execution surfaces.
    #[serde(default)]
    pub budgets: ResourceBudgetConfig,
}

impl OdinConfig {
    /// Load configuration from a YAML file.
    pub fn load(path: &std::path::Path) -> Result<Self, crate::error::OdinError> {
        let contents = std::fs::read_to_string(path).map_err(|e| {
            crate::error::OdinError::Config(format!(
                "Failed to read config file {}: {}",
                path.display(),
                e
            ))
        })?;
        serde_yaml::from_str(&contents).map_err(|e| {
            crate::error::OdinError::Config(format!(
                "Failed to parse config file {}: {}",
                path.display(),
                e
            ))
        })
    }

    /// Save configuration to a YAML file.
    pub fn save(&self, path: &std::path::Path) -> Result<(), crate::error::OdinError> {
        let contents = serde_yaml::to_string(self).map_err(|e| {
            crate::error::OdinError::Config(format!("Failed to serialize config: {}", e))
        })?;
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|e| {
                crate::error::OdinError::Config(format!(
                    "Failed to create config directory {}: {}",
                    parent.display(),
                    e
                ))
            })?;
        }
        let temporary = path.with_extension(format!(
            "{}.tmp-{}",
            path.extension()
                .and_then(|extension| extension.to_str())
                .unwrap_or("yaml"),
            std::process::id()
        ));
        std::fs::write(&temporary, contents).map_err(|e| {
            crate::error::OdinError::Config(format!(
                "Failed to write temporary config file {}: {}",
                temporary.display(),
                e
            ))
        })?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&temporary, std::fs::Permissions::from_mode(0o600)).map_err(
                |e| {
                    crate::error::OdinError::Config(format!(
                        "Failed to restrict config file permissions {}: {}",
                        temporary.display(),
                        e
                    ))
                },
            )?;
        }

        std::fs::rename(&temporary, path).map_err(|e| {
            crate::error::OdinError::Config(format!(
                "Failed to atomically replace config file {}: {}",
                path.display(),
                e
            ))
        })
    }
}

/// Central limits applied to caller-controlled payloads and work.
///
/// Individual tools may impose a stricter limit, but they must never raise
/// these ceilings. Keeping the values in one serializable object prevents the
/// HTTP, CLI, scheduler, and loop surfaces from silently drifting apart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceBudgetConfig {
    /// Maximum model-requested tool calls in one ACT turn.
    #[serde(default = "default_budget_tool_calls")]
    pub max_tool_calls_per_turn: u32,
    /// Maximum tool calls across the complete task execution.
    #[serde(default = "default_budget_task_tool_calls")]
    pub max_tool_calls_per_task: u32,
    /// Maximum retained output from one tool invocation.
    #[serde(default = "default_budget_output_bytes")]
    pub max_tool_output_bytes: usize,
    /// Maximum wall-clock time for one tool invocation.
    #[serde(default = "default_budget_timeout_secs")]
    pub max_tool_timeout_secs: u64,
    /// Maximum network diagnostic packets requested by a caller.
    #[serde(default = "default_budget_ping_count")]
    pub max_network_ping_count: u32,
    /// Maximum request body accepted by an HTTP gateway surface.
    #[serde(default = "default_budget_request_bytes")]
    pub max_request_bytes: usize,
    /// Maximum context bytes accepted by an HTTP gateway surface.
    #[serde(default = "default_budget_context_bytes")]
    pub max_context_bytes: usize,
}

fn default_budget_tool_calls() -> u32 {
    10
}

fn default_budget_task_tool_calls() -> u32 {
    100
}

fn default_budget_output_bytes() -> usize {
    1024 * 1024
}

fn default_budget_timeout_secs() -> u64 {
    60
}

fn default_budget_ping_count() -> u32 {
    4
}

fn default_budget_request_bytes() -> usize {
    128 * 1024
}

fn default_budget_context_bytes() -> usize {
    64 * 1024
}

impl Default for ResourceBudgetConfig {
    fn default() -> Self {
        Self {
            max_tool_calls_per_turn: default_budget_tool_calls(),
            max_tool_calls_per_task: default_budget_task_tool_calls(),
            max_tool_output_bytes: default_budget_output_bytes(),
            max_tool_timeout_secs: default_budget_timeout_secs(),
            max_network_ping_count: default_budget_ping_count(),
            max_request_bytes: default_budget_request_bytes(),
            max_context_bytes: default_budget_context_bytes(),
        }
    }
}

impl OdinConfig {
    /// Return the effective budget after applying legacy per-agent/tool caps.
    pub fn effective_resource_budgets(&self) -> ResourceBudgetConfig {
        let mut budgets = self.budgets.clone();
        budgets.max_tool_calls_per_turn = budgets
            .max_tool_calls_per_turn
            .min(self.agent.max_tool_calls_per_turn);
        budgets.max_tool_timeout_secs = budgets
            .max_tool_timeout_secs
            .min(self.tools.default_timeout_secs.max(1));
        budgets
    }

    /// Validate the semantic ranges that cannot be expressed by serde alone.
    pub fn validate(&self) -> Result<(), crate::error::OdinError> {
        if self.budgets.max_tool_calls_per_turn == 0 || self.budgets.max_tool_calls_per_turn > 100 {
            return Err(crate::error::OdinError::Config(
                "budgets.max_tool_calls_per_turn must be between 1 and 100".into(),
            ));
        }
        if self.budgets.max_tool_calls_per_task == 0
            || self.budgets.max_tool_calls_per_task > 10_000
        {
            return Err(crate::error::OdinError::Config(
                "budgets.max_tool_calls_per_task must be between 1 and 10000".into(),
            ));
        }
        if self.budgets.max_tool_output_bytes == 0
            || self.budgets.max_tool_output_bytes > 16 * 1024 * 1024
        {
            return Err(crate::error::OdinError::Config(
                "budgets.max_tool_output_bytes must be between 1 and 16777216".into(),
            ));
        }
        if self.budgets.max_tool_timeout_secs == 0 || self.budgets.max_tool_timeout_secs > 300 {
            return Err(crate::error::OdinError::Config(
                "budgets.max_tool_timeout_secs must be between 1 and 300".into(),
            ));
        }
        if self.budgets.max_network_ping_count == 0 || self.budgets.max_network_ping_count > 32 {
            return Err(crate::error::OdinError::Config(
                "budgets.max_network_ping_count must be between 1 and 32".into(),
            ));
        }
        if self.budgets.max_request_bytes == 0 || self.budgets.max_request_bytes > 1024 * 1024 {
            return Err(crate::error::OdinError::Config(
                "budgets.max_request_bytes must be between 1 and 1048576".into(),
            ));
        }
        if self.budgets.max_context_bytes == 0 || self.budgets.max_context_bytes > 4 * 1024 * 1024 {
            return Err(crate::error::OdinError::Config(
                "budgets.max_context_bytes must be between 1 and 4194304".into(),
            ));
        }

        let budgets = self.effective_resource_budgets();
        if budgets.max_tool_calls_per_turn == 0 {
            return Err(crate::error::OdinError::Config(
                "the effective tool-call budget must be greater than zero".into(),
            ));
        }
        if self.audit.enabled && !self.audit.json_format {
            return Err(crate::error::OdinError::Config(
                "audit.json_format must be true for correlatable durable audit records".into(),
            ));
        }
        for (index, pattern) in self.safety.dangerous_commands.iter().enumerate() {
            regex::Regex::new(pattern).map_err(|error| {
                crate::error::OdinError::Config(format!(
                    "safety.dangerous_commands[{index}] is not a valid regex: {error}"
                ))
            })?;
        }
        if self.memory.max_entries == 0 {
            return Err(crate::error::OdinError::Config(
                "memory.max_entries must be greater than zero".into(),
            ));
        }
        Ok(())
    }
}

// ── General Config ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Human-readable name for this instance
    #[serde(default = "default_instance_name")]
    pub instance_name: String,

    /// Directory for Raven Agent data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_dir: Option<PathBuf>,

    /// Log level
    #[serde(default = "default_log_level")]
    pub log_level: String,

    /// Enable debug mode
    #[serde(default)]
    pub debug: bool,
}

fn default_instance_name() -> String {
    "raven".to_string()
}

fn default_log_level() -> String {
    "info".to_string()
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            instance_name: default_instance_name(),
            data_dir: None,
            log_level: default_log_level(),
            debug: false,
        }
    }
}

// ── Models Config ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    /// Default provider to use
    #[serde(default = "default_provider")]
    pub default_provider: String,

    /// Default model for the primary provider
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,

    /// Model used for the planning phase (can be a cheaper model)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub planning_model: Option<String>,

    /// Model used for critique/self-check (can be the same or different)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub critique_model: Option<String>,

    /// Stronger model for escalation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub escalation_model: Option<String>,

    /// Provider-specific configurations
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,
}

fn default_provider() -> String {
    "openai_compat".to_string()
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            default_provider: default_provider(),
            default_model: None,
            planning_model: None,
            critique_model: None,
            escalation_model: None,
            providers: HashMap::new(),
        }
    }
}

/// Configuration for a single provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider type: "openai_compat", "anthropic", "local"
    pub provider_type: String,

    /// Base URL for the API
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,

    /// API key (or env var name)
    #[serde(skip_serializing)]
    pub api_key: Option<String>,

    /// Environment variable name for the API key
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,

    /// Default model for this provider
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,

    /// Extra HTTP headers
    #[serde(default)]
    pub headers: HashMap<String, String>,

    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,

    /// Max retries for transient failures
    #[serde(default = "default_retries")]
    pub max_retries: u32,

    /// Ordered list of fallback provider names to try if this provider fails
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_chain: Option<Vec<String>>,

    /// Interval in seconds for periodic health checks (0 = disabled)
    #[serde(default)]
    pub health_check_interval_secs: u64,

    /// Number of consecutive failures before the circuit breaker opens (0 = disabled)
    #[serde(default)]
    pub circuit_breaker_threshold: u32,
}

fn default_timeout() -> u64 {
    120
}

fn default_retries() -> u32 {
    3
}

// ── Agent Config ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Maximum loop iterations per task
    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,

    /// Maximum tool calls per turn
    #[serde(default = "default_max_tool_calls")]
    pub max_tool_calls_per_turn: u32,

    /// Confidence threshold below which to escalate
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f64,

    /// Max retries for a single action
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,

    /// Enable goal decomposition for complex tasks
    #[serde(default = "default_true")]
    pub enable_decomposition: bool,

    /// Enable state summarization for context management
    #[serde(default = "default_true")]
    pub enable_summarization: bool,

    /// Token limit before triggering context compression
    #[serde(default = "default_context_limit")]
    pub context_limit: u32,

    /// Compress context to this ratio of the limit
    #[serde(default = "default_compression_ratio")]
    pub compression_ratio: f64,

    /// Directory for markdown skills (loaded by the skill registry)
    #[serde(default = "default_skills_dir")]
    pub skills_dir: String,
}

fn default_max_iterations() -> u32 {
    100
}
fn default_max_tool_calls() -> u32 {
    10
}
fn default_confidence_threshold() -> f64 {
    0.5
}
fn default_max_retries() -> u32 {
    3
}
fn default_true() -> bool {
    true
}
fn default_context_limit() -> u32 {
    32768
}
fn default_compression_ratio() -> f64 {
    0.5
}
fn default_skills_dir() -> String {
    "~/.config/raven/skills".into()
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: default_max_iterations(),
            max_tool_calls_per_turn: default_max_tool_calls(),
            confidence_threshold: default_confidence_threshold(),
            max_retries: default_max_retries(),
            enable_decomposition: default_true(),
            enable_summarization: default_true(),
            context_limit: default_context_limit(),
            compression_ratio: default_compression_ratio(),
            skills_dir: default_skills_dir(),
        }
    }
}

// ── Tools Config ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsConfig {
    /// Enabled tool names
    #[serde(default = "default_enabled_tools")]
    pub enabled: Vec<String>,

    /// Disabled tool names
    #[serde(default)]
    pub disabled: Vec<String>,

    /// Default timeout for tool execution (seconds)
    #[serde(default = "default_tool_timeout")]
    pub default_timeout_secs: u64,

    /// Filesystem boundary for tool execution
    #[serde(default)]
    pub path_boundary: PathBoundary,

    /// Enable sandboxing (container/chroot)
    #[serde(default)]
    pub sandbox_enabled: bool,

    /// Custom tool directories to scan
    #[serde(default)]
    pub tool_dirs: Vec<PathBuf>,

    /// MCP server configurations for loading external tools.
    #[serde(default)]
    pub mcp_servers: Vec<McpServerConfig>,
}

fn default_enabled_tools() -> Vec<String> {
    vec![
        "file_read".into(),
        "file_write".into(),
        "shell".into(),
        "web_search".into(),
        "web_fetch".into(),
        "git".into(),
        "github_issue_create".into(),
        "github_issue_search".into(),
        "github_pr_create".into(),
        "github_pr_status".into(),
        "github_actions_status".into(),
        "http_request".into(),
        "system_info".into(),
        "disk_usage".into(),
        "json_extract".into(),
    ]
}

fn default_tool_timeout() -> u64 {
    60
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled_tools(),
            disabled: vec![],
            default_timeout_secs: default_tool_timeout(),
            path_boundary: PathBoundary::default(),
            sandbox_enabled: false,
            tool_dirs: vec![],
            mcp_servers: vec![],
        }
    }
}

impl ToolsConfig {
    /// Built-in tools enabled after applying the disabled-name override.
    pub fn effective_enabled_tools(&self) -> Vec<String> {
        self.enabled
            .iter()
            .filter(|name| !self.disabled.contains(name))
            .cloned()
            .collect()
    }
}

// ── MCP Config ───────────────────────────────────────────────────────

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
    #[serde(default = "default_mcp_transport_type")]
    pub transport_type: String,

    /// URL for HTTP transport (required when transport_type is "http").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// Environment variables to set for the server process.
    #[serde(default)]
    pub env: std::collections::HashMap<String, String>,

    /// Whether this server is enabled.
    #[serde(default = "default_mcp_enabled")]
    pub enabled: bool,

    /// Capability tags to assign to all tools from this server.
    #[serde(default = "default_mcp_tags")]
    pub tags: Vec<String>,

    /// Require operator approval before any tool from this server runs.
    #[serde(default = "default_true")]
    pub requires_approval: bool,

    /// Mark every tool from this server as safe. Defaults to false because
    /// MCP metadata does not provide a portable safety classification.
    #[serde(default)]
    pub safe: bool,
}

fn default_mcp_transport_type() -> String {
    "stdio".into()
}

fn default_mcp_enabled() -> bool {
    true
}

fn default_mcp_tags() -> Vec<String> {
    vec!["mcp".into(), "external".into(), "dangerous".into()]
}

// ── Memory Config ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Enable persistent memory
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Database path (SQLite)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub db_path: Option<PathBuf>,

    /// Maximum entries in working memory
    #[serde(default = "default_max_entries")]
    pub max_entries: usize,

    /// Auto-save interval in seconds
    #[serde(default = "default_save_interval")]
    pub save_interval_secs: u64,
}

fn default_max_entries() -> usize {
    1000
}
fn default_save_interval() -> u64 {
    300
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            db_path: None,
            max_entries: default_max_entries(),
            save_interval_secs: default_save_interval(),
        }
    }
}

// ── Safety Config ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConfig {
    /// Enable command approval prompts
    #[serde(default = "default_true")]
    pub require_approval: bool,

    /// Commands that always require approval (regex patterns)
    #[serde(default = "default_dangerous_commands")]
    pub dangerous_commands: Vec<String>,

    /// Maximum rate of tool calls per minute
    #[serde(default = "default_max_rate")]
    pub max_rate_per_minute: u32,

    /// Custom permission rules
    #[serde(default)]
    pub permissions: Vec<PermissionRule>,
}

fn default_dangerous_commands() -> Vec<String> {
    vec![
        r"rm\s+-rf".into(),
        r"git\s+reset\s+--hard".into(),
        r"git\s+push\s+--force".into(),
        r"sudo\s+".into(),
        r"chmod\s+777".into(),
        r">\s*/dev/".into(),
        r"mkfs\.".into(),
        r"dd\s+if=".into(),
    ]
}

fn default_max_rate() -> u32 {
    60
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            require_approval: true,
            dangerous_commands: default_dangerous_commands(),
            max_rate_per_minute: default_max_rate(),
            permissions: vec![],
        }
    }
}

// ── Audit Config ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Enable audit logging
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Log file path
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_path: Option<PathBuf>,

    /// Log in JSON format
    #[serde(default = "default_true")]
    pub json_format: bool,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_path: None,
            json_format: true,
        }
    }
}

// ── Gateway Config ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    /// Enable the HTTP API
    #[serde(default)]
    pub http_enabled: bool,

    /// HTTP API listen address
    #[serde(default = "default_http_addr")]
    pub http_addr: String,

    /// Operator-only management API listen address.
    #[serde(default = "default_management_addr")]
    pub management_addr: String,

    /// Enable Discord integration
    #[serde(default)]
    pub discord_enabled: bool,

    /// Discord bot token (or env var name)
    #[serde(skip_serializing)]
    pub discord_token: Option<String>,

    /// Discord token env var name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub discord_token_env: Option<String>,

    /// Discord role allowed to invoke Raven commands.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discord_admin_role: Option<String>,

    /// Discord user IDs explicitly allowed to invoke Raven commands.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub discord_admin_user_ids: Vec<u64>,

    /// Restrict Discord commands to this guild ID when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discord_guild_id: Option<u64>,

    /// Permit direct-message commands from explicitly allowed user IDs.
    #[serde(default)]
    pub discord_allow_dms: bool,

    /// Optional shared operator credential for the management API and
    /// WebSocket pause/resume/cancel commands. When unset, `raven serve`
    /// generates an ephemeral credential and prints it once at startup.
    #[serde(default, skip_serializing)]
    pub control_token: Option<String>,

    /// Environment variable containing the shared operator credential.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_token_env: Option<String>,

    /// Bearer credential required by the public HTTP and WebSocket listener
    /// when configured. Prefer `public_auth_token_env` so the secret is never
    /// serialized into a config file.
    #[serde(default, skip_serializing)]
    pub public_auth_token: Option<String>,

    /// Environment variable containing the public listener credential.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub public_auth_token_env: Option<String>,

    /// Explicitly permit an unauthenticated non-loopback public listener.
    /// This is intentionally opt-in and should only be used behind another
    /// trusted authentication boundary.
    #[serde(default)]
    pub allow_insecure_non_loopback: bool,
}

fn default_http_addr() -> String {
    "127.0.0.1:9177".to_string()
}

fn default_management_addr() -> String {
    "127.0.0.1:9178".to_string()
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            http_enabled: false,
            http_addr: default_http_addr(),
            management_addr: default_management_addr(),
            discord_enabled: false,
            discord_token: None,
            discord_token_env: None,
            discord_admin_role: None,
            discord_admin_user_ids: Vec::new(),
            discord_guild_id: None,
            discord_allow_dms: false,
            control_token: None,
            control_token_env: None,
            public_auth_token: None,
            public_auth_token_env: None,
            allow_insecure_non_loopback: false,
        }
    }
}

// ── Scheduler Config ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Enable the scheduler
    #[serde(default)]
    pub enabled: bool,

    /// Check interval in seconds
    #[serde(default = "default_check_interval")]
    pub check_interval_secs: u64,

    /// Max concurrent jobs
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: u32,

    /// Database path for scheduler job persistence.
    /// Defaults to `~/.raven-agent/scheduler.db` if not set.
    #[serde(default)]
    pub db_path: Option<String>,

    /// Time to wait for in-flight jobs during shutdown before cancelling them.
    #[serde(default = "default_scheduler_shutdown_grace")]
    pub shutdown_grace_secs: u64,
}

fn default_check_interval() -> u64 {
    30
}
fn default_max_concurrent() -> u32 {
    5
}
fn default_scheduler_shutdown_grace() -> u64 {
    30
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            check_interval_secs: default_check_interval(),
            max_concurrent: default_max_concurrent(),
            db_path: None,
            shutdown_grace_secs: default_scheduler_shutdown_grace(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omitted_budget_and_audit_fields_use_safe_defaults() {
        let config: OdinConfig = serde_yaml::from_str("{}").unwrap();
        assert_eq!(config.budgets.max_tool_calls_per_task, 100);
        assert!(config.audit.json_format);
    }

    #[test]
    fn invalid_budget_and_safety_configuration_is_rejected() {
        let mut config = OdinConfig::default();
        config.budgets.max_tool_output_bytes = 0;
        assert!(config.validate().is_err());

        let mut config = OdinConfig::default();
        config.budgets.max_tool_calls_per_turn = 101;
        assert!(config.validate().is_err());

        let mut config = OdinConfig::default();
        config.safety.dangerous_commands = vec!["[unterminated".into()];
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_save_replaces_atomically_and_restricts_file_permissions() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.yaml");
        OdinConfig::default().save(&path).unwrap();
        assert!(path.exists());
        assert!(!path.with_extension("yaml.tmp").exists());

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn config_serialization_omits_direct_secret_values() {
        let mut config = OdinConfig::default();
        config.gateway.discord_token = Some("discord-secret".into());
        config.gateway.control_token = Some("operator-secret".into());
        config.gateway.public_auth_token = Some("public-secret".into());
        config.models.providers.insert(
            "hosted".into(),
            ProviderConfig {
                provider_type: "openai_compat".into(),
                base_url: None,
                api_key: Some("provider-secret".into()),
                api_key_env: Some("RAVEN_TEST_PROVIDER_KEY".into()),
                default_model: Some("model".into()),
                headers: HashMap::new(),
                timeout_secs: 120,
                max_retries: 3,
                fallback_chain: None,
                health_check_interval_secs: 0,
                circuit_breaker_threshold: 0,
            },
        );

        let serialized = serde_yaml::to_string(&config).unwrap();
        for secret in [
            "discord-secret",
            "operator-secret",
            "public-secret",
            "provider-secret",
        ] {
            assert!(!serialized.contains(secret));
        }
        assert!(serialized.contains("RAVEN_TEST_PROVIDER_KEY"));
    }
}
