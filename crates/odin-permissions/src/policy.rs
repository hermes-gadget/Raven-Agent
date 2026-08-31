//! Permission policy engine for Raven Agent.
//!
//! Implements [`PermissionEngine`] from odin-core, providing:
//! - Tool call allow/deny rules based on configured rules
//! - Shell command safety checking via regex patterns
//! - Rate limiting per agent/tool
//! - Path boundary validation for filesystem access

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use odin_core::ResolvedPathBoundary;
use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::AuditLogger;
use odin_core::types::{
    AgentId, AuditEntry, AuditEventType, AuditResult, PathBoundary, PermissionAction,
    PermissionRule,
};
use regex::Regex;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, trace, warn};

/// A rate-limit tracker for a single agent/tool pair.
#[derive(Debug, Clone)]
struct RateTracker {
    /// Timestamps of recent calls within the window.
    timestamps: VecDeque<DateTime<Utc>>,
    /// Maximum calls allowed per minute.
    max_per_minute: u32,
    /// Last time this principal/tool bucket was consulted.
    last_seen: DateTime<Utc>,
}

impl RateTracker {
    fn new(max_per_minute: u32) -> Self {
        Self {
            timestamps: VecDeque::new(),
            max_per_minute,
            last_seen: Utc::now(),
        }
    }

    /// Check if a call is allowed and record it if so.
    fn check_and_record(&mut self, now: &DateTime<Utc>) -> bool {
        // Remove timestamps older than 1 minute
        let cutoff = *now - chrono::TimeDelta::minutes(1);
        self.last_seen = *now;
        while self.timestamps.front().is_some_and(|time| *time <= cutoff) {
            self.timestamps.pop_front();
        }

        if self.timestamps.len() < self.max_per_minute as usize {
            self.timestamps.push_back(*now);
            true
        } else {
            false
        }
    }
}

/// The permission policy engine.
///
/// Evaluates tool calls and commands against configured allow/deny rules,
/// rate limits, and path boundaries.
pub struct PolicyEngine {
    /// Permission rules indexed by tool name.
    rules: HashMap<String, PermissionRule>,
    /// Dangerous command patterns (regex).
    dangerous_patterns: Vec<Regex>,
    /// Configuration error retained so the compatibility constructor fails
    /// closed at the first policy check instead of silently weakening safety.
    configuration_error: Option<String>,
    /// Path boundaries for filesystem operations.
    path_boundary: Result<ResolvedPathBoundary, String>,
    /// Rate limit trackers: key = "agent_id:tool_name"
    rate_trackers: Arc<RwLock<HashMap<String, RateTracker>>>,
    /// Default max rate per minute.
    default_max_rate: u32,
    /// Whether to require approval by default.
    require_approval: bool,
    /// Connected interactive approval responder, if any.
    approval_gate: Option<Arc<crate::approval::ApprovalGate>>,
    /// Optional audit sink for approval lifecycle decisions.
    audit_logger: Option<Arc<dyn AuditLogger>>,
}

const MAX_RATE_TRACKERS: usize = 4_096;
const RATE_TRACKER_IDLE_MINUTES: i64 = 10;

impl PolicyEngine {
    /// Create a new policy engine.
    pub fn new(
        rules: Vec<PermissionRule>,
        dangerous_commands: &[String],
        path_boundary: PathBoundary,
        default_max_rate: u32,
        require_approval: bool,
    ) -> Self {
        let rules_map: HashMap<String, PermissionRule> = rules
            .into_iter()
            .map(|r| (r.tool_name.clone(), r))
            .collect();

        let mut configuration_error = None;
        let dangerous_patterns: Vec<Regex> = dangerous_commands
            .iter()
            .filter_map(|pattern| match Regex::new(pattern) {
                Ok(regex) => Some(regex),
                Err(error) => {
                    warn!(
                        pattern = %pattern,
                        error = %error,
                        "Invalid dangerous command regex pattern; policy will fail closed"
                    );
                    configuration_error.get_or_insert_with(|| {
                        format!("invalid dangerous command regex '{pattern}': {error}")
                    });
                    None
                }
            })
            .collect();

        let path_boundary =
            ResolvedPathBoundary::new(&path_boundary).map_err(|error| error.to_string());

        Self {
            rules: rules_map,
            dangerous_patterns,
            configuration_error,
            path_boundary,
            rate_trackers: Arc::new(RwLock::new(HashMap::new())),
            default_max_rate,
            require_approval,
            approval_gate: None,
            audit_logger: None,
        }
    }

    /// Return a configuration error, if construction found an invalid safety
    /// rule. Keeping this check available makes direct embedders fail closed
    /// even when they do not use the top-level configuration validator.
    pub fn validate(&self) -> OdinResult<()> {
        if let Some(error) = &self.configuration_error {
            return Err(OdinError::Config(error.clone()));
        }
        Ok(())
    }

    /// Connect an interactive approval gate.
    pub fn with_approval_gate(mut self, gate: Arc<crate::approval::ApprovalGate>) -> Self {
        self.approval_gate = Some(gate);
        self
    }

    /// Connect an audit logger for approval decisions.
    pub fn with_audit_logger(mut self, logger: Arc<dyn AuditLogger>) -> Self {
        self.audit_logger = Some(logger);
        self
    }

    /// Check if a shell command matches any dangerous pattern.
    pub fn is_dangerous_command(&self, command: &str) -> bool {
        self.configuration_error.is_some()
            || self
                .dangerous_patterns
                .iter()
                .any(|re| re.is_match(command))
    }

    /// Whether tools marked as approval-required must be gated.
    pub fn requires_approval(&self) -> bool {
        self.require_approval
    }

    /// Validate that a path is within the allowed boundaries.
    pub fn check_path_boundary(&self, path: &std::path::Path, write: bool) -> OdinResult<()> {
        self.validate()?;
        let boundary = self.path_boundary.as_ref().map_err(|error| {
            OdinError::Validation(format!(
                "Invalid filesystem boundary configuration: {error}"
            ))
        })?;
        if write {
            boundary.check_write(path)?;
        } else {
            boundary.check_read(path)?;
        }
        Ok(())
    }

    /// Request approval while preserving the caller's session correlation ID.
    /// The trait-compatible method below remains available for older callers
    /// that do not have a session ID.
    pub async fn request_approval_for_session(
        &self,
        agent_id: AgentId,
        session_id: uuid::Uuid,
        action: &str,
        details: &str,
    ) -> OdinResult<bool> {
        self.validate()?;
        let redactor = crate::redact::SecretRedactor::full();
        let Some(gate) = self.approval_gate.as_ref() else {
            if let Some(logger) = self.audit_logger.as_ref() {
                logger
                    .log(AuditEntry {
                        id: uuid::Uuid::new_v4(),
                        timestamp: Utc::now(),
                        agent_id,
                        session_id,
                        event_type: AuditEventType::PermissionCheck,
                        action: redactor.redact(action),
                        details: serde_json::json!({
                            "details": redactor.redact(details),
                            "status": "denied_no_responder",
                        }),
                        result: AuditResult::Denied,
                    })
                    .await?;
            }
            warn!(
                action = %redactor.redact(action),
                details = %redactor.redact(details),
                "Approval requested but no approval responder is connected; denying"
            );
            return Ok(false);
        };

        let (request, status) = gate
            .request(agent_id, action.to_owned(), details.to_owned())
            .await;
        let approved = status == crate::approval::ApprovalStatus::Approved;

        if let Some(logger) = self.audit_logger.as_ref() {
            // Redact here as well as at the durable audit boundary so custom
            // AuditLogger implementations never receive raw approval details.
            let entry = AuditEntry {
                id: uuid::Uuid::new_v4(),
                timestamp: Utc::now(),
                agent_id,
                session_id,
                event_type: AuditEventType::PermissionCheck,
                action: redactor.redact(action),
                details: serde_json::json!({
                    "request_id": request.id,
                    "details": redactor.redact(details),
                    "argument_fingerprint": request.argument_fingerprint,
                    "status": status,
                }),
                result: if approved {
                    AuditResult::Success
                } else {
                    AuditResult::Denied
                },
            };
            logger.log(entry).await?;
        }

        warn!(
            action = %redactor.redact(action),
            details = %redactor.redact(details),
            ?status,
            "Approval request completed"
        );
        Ok(approved)
    }
}

fn http_request_requires_approval(args: &serde_json::Value) -> bool {
    let method = args
        .get("method")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if !matches!(
        method.to_ascii_uppercase().as_str(),
        "GET" | "HEAD" | "OPTIONS"
    ) {
        return true;
    }

    args.get("headers")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|headers| {
            headers.iter().any(|header| {
                let name = header
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_ascii_lowercase();
                matches!(
                    name.as_str(),
                    "authorization"
                        | "proxy-authorization"
                        | "cookie"
                        | "x-api-key"
                        | "x-auth-token"
                )
            })
        })
}

impl Default for PolicyEngine {
    fn default() -> Self {
        Self::new(
            vec![],
            &[
                // Destructive filesystem operations
                r"rm\s+-rf".into(),
                r"rm\s+-r\s+/".into(),
                r"mkfs\.".into(),
                r"dd\s+if=".into(),
                r">\s*/dev/".into(),
                // Privilege escalation
                r"sudo\s+".into(),
                r"su\s+-".into(),
                r"chown\s+root".into(),
                // Permission changes
                r"chmod\s+777".into(),
                r"chmod\s+-R\s+777".into(),
                // Destructive git operations
                r"git\s+reset\s+--hard".into(),
                r"git\s+push\s+--force".into(),
                r"git\s+push\s+--delete\s+origin".into(),
                r"git\s+clean\s+-fd".into(),
                // Network dangerous
                r"iptables\s+-F".into(),
                r"ufw\s+disable".into(),
                r"nc\s+-[lL]\s+-[pP]".into(),
                // System control
                r"shutdown\s+".into(),
                r"reboot\s+".into(),
                r"systemctl\s+stop\s+(docker|ssh|nginx|apache)".into(),
                // Fork bombs / resource exhaustion
                r":\(\)\s*\{".into(),
                r"while\s+true".into(),
            ],
            PathBoundary::default(),
            60,
            true,
        )
    }
}

#[async_trait]
impl odin_core::traits::PermissionEngine for PolicyEngine {
    /// Check if a tool call is allowed based on configured rules.
    async fn check_tool(
        &self,
        agent_id: AgentId,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> OdinResult<PermissionAction> {
        self.validate()?;
        // Check for explicit rules
        if let Some(rule) = self.rules.get(tool_name) {
            let action = if rule.action == PermissionAction::Deny {
                PermissionAction::Deny
            } else if rule.action == PermissionAction::AskUser
                || rule.require_approval
                || (tool_name == "http_request" && http_request_requires_approval(args))
            {
                PermissionAction::AskUser
            } else {
                PermissionAction::Allow
            };
            debug!(
                agent_id = %agent_id,
                tool = %tool_name,
                action = %action,
                rule_requires_approval = rule.require_approval,
                "Tool permission check via explicit rule"
            );
            return Ok(action);
        }

        if tool_name == "http_request" && http_request_requires_approval(args) {
            debug!(
                agent_id = %agent_id,
                tool = %tool_name,
                "State-changing or credential-bearing HTTP request requires approval"
            );
            return Ok(PermissionAction::AskUser);
        }

        // Tool metadata determines whether the default path needs approval.
        // Explicit rules still override this decision.
        debug!(
            agent_id = %agent_id,
            tool = %tool_name,
            "Tool permission check defaults to Allow"
        );
        Ok(PermissionAction::Allow)
    }

    /// Check if a shell command is allowed.
    async fn check_command(
        &self,
        agent_id: AgentId,
        command: &str,
    ) -> OdinResult<PermissionAction> {
        self.validate()?;
        if self.is_dangerous_command(command) {
            warn!(agent_id = %agent_id, "Dangerous command detected; approval required");
            return Ok(PermissionAction::AskUser);
        }

        debug!(
            agent_id = %agent_id,
            command = %command,
            "Command allowed"
        );
        Ok(PermissionAction::Allow)
    }

    /// Check rate limits for a specific agent and tool.
    async fn check_rate_limit(&self, agent_id: AgentId, tool_name: &str) -> OdinResult<bool> {
        self.validate()?;
        let key = format!("{}:{}", agent_id, tool_name);

        // Determine max rate for this tool
        let max_rate = self
            .rules
            .get(tool_name)
            .and_then(|r| r.max_rate_per_minute)
            .unwrap_or(self.default_max_rate);

        let now = Utc::now();
        let mut trackers = self.rate_trackers.write().await;
        let idle_cutoff = now - chrono::TimeDelta::minutes(RATE_TRACKER_IDLE_MINUTES);
        trackers.retain(|_, tracker| tracker.last_seen > idle_cutoff);
        if !trackers.contains_key(&key) && trackers.len() >= MAX_RATE_TRACKERS {
            warn!(
                agent_id = %agent_id,
                tool = %tool_name,
                max_trackers = MAX_RATE_TRACKERS,
                "Rate-limit tracker capacity reached; denying new principal bucket"
            );
            return Ok(false);
        }
        let tracker = trackers
            .entry(key.clone())
            .or_insert_with(|| RateTracker::new(max_rate));

        let allowed = tracker.check_and_record(&now);
        if allowed {
            trace!(
                agent_id = %agent_id,
                tool = %tool_name,
                "Rate limit check passed"
            );
        } else {
            warn!(
                agent_id = %agent_id,
                tool = %tool_name,
                max_rate = max_rate,
                "Rate limit exceeded"
            );
        }

        Ok(allowed)
    }

    /// Request user approval for an action.
    async fn request_approval(
        &self,
        agent_id: AgentId,
        action: &str,
        details: &str,
    ) -> OdinResult<bool> {
        self.request_approval_for_session(agent_id, uuid::Uuid::new_v4(), action, details)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use odin_core::traits::{AuditLogger, PermissionEngine};
    use uuid::Uuid;

    #[derive(Default)]
    struct CaptureAuditLogger {
        entries: tokio::sync::Mutex<Vec<AuditEntry>>,
    }

    #[async_trait]
    impl AuditLogger for CaptureAuditLogger {
        async fn log(&self, entry: AuditEntry) -> OdinResult<()> {
            self.entries.lock().await.push(entry);
            Ok(())
        }

        async fn query(
            &self,
            _agent_id: Option<AgentId>,
            _session_id: Option<odin_core::types::SessionId>,
            _event_type: Option<AuditEventType>,
            limit: usize,
        ) -> OdinResult<Vec<AuditEntry>> {
            Ok(self
                .entries
                .lock()
                .await
                .iter()
                .rev()
                .take(limit)
                .cloned()
                .collect())
        }

        async fn recent(&self, limit: usize) -> OdinResult<Vec<AuditEntry>> {
            self.query(None, None, None, limit).await
        }
    }

    fn test_agent() -> AgentId {
        Uuid::new_v4()
    }

    #[tokio::test]
    async fn test_default_allow_tool() {
        let engine = PolicyEngine::default();
        let agent = test_agent();
        let result = engine
            .check_tool(agent, "file_read", &serde_json::json!({}))
            .await
            .unwrap();
        assert_eq!(result, PermissionAction::Allow);
    }

    #[tokio::test]
    async fn test_explicit_rule_deny() {
        let rules = vec![PermissionRule {
            tool_name: "shell".to_string(),
            action: PermissionAction::Deny,
            require_approval: true,
            max_rate_per_minute: None,
        }];
        let engine = PolicyEngine::new(rules, &[], PathBoundary::default(), 60, true);
        let agent = test_agent();
        let result = engine
            .check_tool(agent, "shell", &serde_json::json!({}))
            .await
            .unwrap();
        assert_eq!(result, PermissionAction::Deny);
    }

    #[tokio::test]
    async fn test_explicit_rule_allow() {
        let rules = vec![PermissionRule {
            tool_name: "shell".to_string(),
            action: PermissionAction::Allow,
            require_approval: false,
            max_rate_per_minute: None,
        }];
        let engine = PolicyEngine::new(rules, &[], PathBoundary::default(), 60, false);
        let agent = test_agent();
        let result = engine
            .check_tool(agent, "shell", &serde_json::json!({}))
            .await
            .unwrap();
        assert_eq!(result, PermissionAction::Allow);
    }

    #[tokio::test]
    async fn explicit_rule_approval_is_enforced_regardless_of_global_default() {
        for global_require_approval in [false, true] {
            let engine = PolicyEngine::new(
                vec![PermissionRule {
                    tool_name: "shell".to_string(),
                    action: PermissionAction::Allow,
                    require_approval: true,
                    max_rate_per_minute: None,
                }],
                &[],
                PathBoundary::default(),
                60,
                global_require_approval,
            );
            let result = engine
                .check_tool(test_agent(), "shell", &serde_json::json!({}))
                .await
                .unwrap();
            assert_eq!(result, PermissionAction::AskUser);
        }
    }

    #[tokio::test]
    async fn explicit_deny_wins_over_approval_requirement() {
        let engine = PolicyEngine::new(
            vec![PermissionRule {
                tool_name: "shell".to_string(),
                action: PermissionAction::Deny,
                require_approval: true,
                max_rate_per_minute: None,
            }],
            &[],
            PathBoundary::default(),
            60,
            false,
        );
        let result = engine
            .check_tool(test_agent(), "shell", &serde_json::json!({}))
            .await
            .unwrap();
        assert_eq!(result, PermissionAction::Deny);
    }

    #[tokio::test]
    async fn mutating_and_credential_bearing_http_requests_require_approval() {
        let engine = PolicyEngine::new(vec![], &[], PathBoundary::default(), 60, false);
        let agent = test_agent();

        let get = engine
            .check_tool(
                agent,
                "http_request",
                &serde_json::json!({"method": "GET", "url": "https://example.com"}),
            )
            .await
            .unwrap();
        assert_eq!(get, PermissionAction::Allow);

        let post = engine
            .check_tool(
                agent,
                "http_request",
                &serde_json::json!({"method": "POST", "url": "https://example.com"}),
            )
            .await
            .unwrap();
        assert_eq!(post, PermissionAction::AskUser);

        let authorized_get = engine
            .check_tool(
                agent,
                "http_request",
                &serde_json::json!({
                    "method": "GET",
                    "url": "https://example.com",
                    "headers": [{"name": "Authorization", "value": "synthetic"}]
                }),
            )
            .await
            .unwrap();
        assert_eq!(authorized_get, PermissionAction::AskUser);
    }

    #[tokio::test]
    async fn test_dangerous_command_detection() {
        let engine = PolicyEngine::default();
        assert!(engine.is_dangerous_command("rm -rf /"));
        assert!(engine.is_dangerous_command("sudo apt install"));
        assert!(engine.is_dangerous_command("chmod 777 /etc"));
        assert!(engine.is_dangerous_command("dd if=/dev/zero of=/dev/sda"));
        assert!(!engine.is_dangerous_command("ls -la"));
        assert!(!engine.is_dangerous_command("echo hello"));
    }

    #[tokio::test]
    async fn invalid_dangerous_pattern_fails_closed() {
        let engine = PolicyEngine::new(
            vec![],
            &["[unterminated".into()],
            PathBoundary::default(),
            60,
            true,
        );
        assert!(engine.validate().is_err());
        assert!(engine.is_dangerous_command("echo harmless"));
        assert!(
            engine
                .check_command(test_agent(), "echo harmless")
                .await
                .is_err()
        );
        assert!(
            engine
                .check_path_boundary(std::path::Path::new("."), false)
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_command_permission() {
        let engine = PolicyEngine::default();
        let agent = test_agent();

        // Dangerous command should require approval
        let result = engine
            .check_command(agent, "rm -rf /important")
            .await
            .unwrap();
        assert_eq!(result, PermissionAction::AskUser);

        // Safe command should be allowed
        let result = engine.check_command(agent, "ls -la").await.unwrap();
        assert_eq!(result, PermissionAction::Allow);
    }

    #[tokio::test]
    async fn approval_audit_record_is_redacted() {
        let gate = Arc::new(crate::approval::ApprovalGate::new(false, 30));
        let logger = Arc::new(CaptureAuditLogger::default());
        let session_id = Uuid::new_v4();
        let engine = Arc::new(
            PolicyEngine::new(vec![], &[], PathBoundary::default(), 60, true)
                .with_approval_gate(gate.clone())
                .with_audit_logger(logger.clone()),
        );
        let mut requests = gate.subscribe();
        let secret = "ghp_abcdefghijklmnopqrstuvwxyz123456";
        let details = format!(r#"{{"token":"{secret}","email":"user@example.com"}}"#);
        let waiter = {
            let engine = engine.clone();
            tokio::spawn(async move {
                engine
                    .request_approval_for_session(test_agent(), session_id, "shell", &details)
                    .await
                    .unwrap()
            })
        };

        let request = requests.recv().await.unwrap();
        gate.approve(&request.id, &request.argument_fingerprint)
            .await
            .unwrap();
        assert!(waiter.await.unwrap());

        let entries = logger.recent(1).await.unwrap();
        let serialized = serde_json::to_string(&entries[0]).unwrap();
        assert_eq!(entries[0].session_id, session_id);
        assert!(!serialized.contains(secret));
        assert!(!serialized.contains("user@example.com"));
        assert!(serialized.contains("[REDACTED:"));
    }

    #[tokio::test]
    async fn approval_without_responder_is_audited_as_denied() {
        let logger = Arc::new(CaptureAuditLogger::default());
        let session_id = Uuid::new_v4();
        let engine = PolicyEngine::new(vec![], &[], PathBoundary::default(), 60, true)
            .with_audit_logger(logger.clone());

        assert!(
            !engine
                .request_approval_for_session(test_agent(), session_id, "shell", "{}")
                .await
                .unwrap()
        );
        let entries = logger.recent(1).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].session_id, session_id);
        assert_eq!(entries[0].result, AuditResult::Denied);
    }

    #[tokio::test]
    async fn test_rate_limit() {
        let engine = PolicyEngine::new(
            vec![],
            &[],
            PathBoundary::default(),
            3, // max 3 per minute
            false,
        );
        let agent = test_agent();

        // First 3 calls should pass
        assert!(engine.check_rate_limit(agent, "shell").await.unwrap());
        assert!(engine.check_rate_limit(agent, "shell").await.unwrap());
        assert!(engine.check_rate_limit(agent, "shell").await.unwrap());

        // 4th should fail
        assert!(!engine.check_rate_limit(agent, "shell").await.unwrap());
    }

    #[test]
    fn test_path_boundary_allowed() {
        let boundary = PathBoundary {
            allowed_read: vec!["/home/user".to_string()],
            allowed_write: vec!["/home/user".to_string()],
            denied: vec!["/etc/passwd".to_string()],
        };
        let engine = PolicyEngine::new(vec![], &[], boundary, 60, true);

        assert!(
            engine
                .check_path_boundary(std::path::Path::new("/home/user/docs/file.txt"), false)
                .is_ok()
        );
        assert!(
            engine
                .check_path_boundary(std::path::Path::new("/home/user/docs/file.txt"), true)
                .is_ok()
        );
    }

    #[test]
    fn test_path_boundary_denied() {
        let boundary = PathBoundary {
            allowed_read: vec!["/".to_string()],
            allowed_write: vec!["/home/user".to_string()],
            denied: vec!["/etc/passwd".to_string()],
        };
        let engine = PolicyEngine::new(vec![], &[], boundary, 60, true);

        assert!(
            engine
                .check_path_boundary(std::path::Path::new("/etc/passwd"), false)
                .is_err()
        );
    }

    #[test]
    fn test_path_boundary_outside() {
        let boundary = PathBoundary {
            allowed_read: vec!["/home/user".to_string()],
            allowed_write: vec!["/home/user".to_string()],
            denied: vec![],
        };
        let engine = PolicyEngine::new(vec![], &[], boundary, 60, true);

        assert!(
            engine
                .check_path_boundary(std::path::Path::new("/etc/config"), false)
                .is_err()
        );
        assert!(
            engine
                .check_path_boundary(std::path::Path::new("/etc/config"), true)
                .is_err()
        );
    }

    #[test]
    fn test_path_boundary_current_dir() {
        let boundary = PathBoundary {
            allowed_read: vec![".".to_string()],
            allowed_write: vec![".".to_string()],
            denied: vec![],
        };
        let engine = PolicyEngine::new(vec![], &[], boundary, 60, true);

        assert!(
            engine
                .check_path_boundary(std::path::Path::new("./src/main.rs"), false)
                .is_ok()
        );
        assert!(
            engine
                .check_path_boundary(std::path::Path::new("../outside"), false)
                .is_err()
        );
    }

    #[test]
    fn test_path_boundary_rejects_sibling_prefix() {
        let parent = tempfile::tempdir().unwrap();
        let allowed = parent.path().join("repo");
        let sibling = parent.path().join("repo-private");
        std::fs::create_dir_all(&allowed).unwrap();
        std::fs::create_dir_all(&sibling).unwrap();
        let engine = PolicyEngine::new(
            vec![],
            &[],
            PathBoundary {
                allowed_read: vec![allowed.display().to_string()],
                allowed_write: vec![allowed.display().to_string()],
                denied: vec![],
            },
            60,
            true,
        );

        assert!(engine.check_path_boundary(&allowed, false).is_ok());
        assert!(engine.check_path_boundary(&sibling, false).is_err());
    }

    #[test]
    fn test_home_denial_is_expanded() {
        let Some(home) = std::env::var_os("HOME") else {
            return;
        };
        let home = std::path::PathBuf::from(home);
        let engine = PolicyEngine::new(
            vec![],
            &[],
            PathBoundary {
                allowed_read: vec![home.display().to_string()],
                allowed_write: vec![home.display().to_string()],
                denied: vec!["~/.ssh".to_string()],
            },
            60,
            true,
        );
        assert!(
            engine
                .check_path_boundary(&home.join(".ssh/id_synthetic"), false)
                .is_err()
        );
    }
}
