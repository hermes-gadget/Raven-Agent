//! Interactive approval gate for potentially dangerous operations.
//!
//! Approval requests are correlated with a fingerprint of the exact action and
//! arguments. Only redacted request details leave this crate.

use chrono::{DateTime, Utc};
use odin_core::error::OdinResult;
use odin_core::types::AgentId;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{broadcast, watch};
use tracing::{debug, info, warn};

/// A single approval request safe to display to an operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    /// Unique request ID.
    pub id: String,
    /// The agent requesting approval.
    pub agent_id: AgentId,
    /// Redacted description of the action.
    pub action: String,
    /// Redacted arguments or context for the action.
    pub details: String,
    /// Fingerprint binding a decision to the original, unredacted arguments.
    pub argument_fingerprint: String,
    /// Current status of the request.
    pub status: ApprovalStatus,
    /// Time at which this request was created.
    pub created_at: DateTime<Utc>,
    /// Time at which this request will expire.
    pub expires_at: DateTime<Utc>,
}

/// Status of an approval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalStatus {
    /// Waiting for a response.
    Pending,
    /// Action was approved.
    Approved,
    /// Action was denied.
    Denied,
    /// Request expired.
    Expired,
}

struct PendingApproval {
    request: ApprovalRequest,
    decision_tx: watch::Sender<ApprovalStatus>,
}

struct ApprovalGateInner {
    pending: Mutex<HashMap<String, PendingApproval>>,
    auto_approve: Mutex<bool>,
    timeout: Duration,
    request_tx: broadcast::Sender<ApprovalRequest>,
}

/// Manages correlated interactive command/tool approval.
#[derive(Clone)]
pub struct ApprovalGate {
    inner: Arc<ApprovalGateInner>,
}

impl ApprovalGate {
    /// Create a new approval gate with a timeout in seconds.
    pub fn new(auto_approve: bool, timeout_secs: u64) -> Self {
        Self::with_timeout(auto_approve, Duration::from_secs(timeout_secs))
    }

    /// Create a gate with an explicit timeout.
    pub fn with_timeout(auto_approve: bool, timeout: Duration) -> Self {
        let (request_tx, _) = broadcast::channel(256);
        Self {
            inner: Arc::new(ApprovalGateInner {
                pending: Mutex::new(HashMap::new()),
                auto_approve: Mutex::new(auto_approve),
                timeout,
                request_tx,
            }),
        }
    }

    /// Subscribe to newly submitted requests.
    pub fn subscribe(&self) -> broadcast::Receiver<ApprovalRequest> {
        self.inner.request_tx.subscribe()
    }

    /// Fingerprint an action and its exact arguments.
    pub fn fingerprint(action: &str, details: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update((action.len() as u64).to_be_bytes());
        hasher.update(action.as_bytes());
        hasher.update((details.len() as u64).to_be_bytes());
        hasher.update(details.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Submit a new approval request and return its public, redacted form.
    pub async fn submit_request(
        &self,
        agent_id: AgentId,
        action: String,
        details: String,
    ) -> ApprovalRequest {
        let id = uuid::Uuid::new_v4().to_string();
        let created_at = Utc::now();
        let expires_at = created_at
            + chrono::TimeDelta::from_std(self.inner.timeout).unwrap_or(chrono::TimeDelta::MAX);
        let redactor = crate::redact::SecretRedactor::full();
        let request = ApprovalRequest {
            id: id.clone(),
            agent_id,
            action: redactor.redact(&action),
            details: redactor.redact(&details),
            argument_fingerprint: Self::fingerprint(&action, &details),
            status: ApprovalStatus::Pending,
            created_at,
            expires_at,
        };
        let (decision_tx, _) = watch::channel(ApprovalStatus::Pending);

        self.inner
            .pending
            .lock()
            .expect("approval gate lock poisoned")
            .insert(
                id.clone(),
                PendingApproval {
                    request: request.clone(),
                    decision_tx,
                },
            );
        let _ = self.inner.request_tx.send(request.clone());
        info!(request_id = %id, "Approval request submitted");
        request
    }

    /// Submit and wait for a decision.
    ///
    /// Dropping this future (for example when a client disconnects) denies the
    /// still-pending request.
    pub async fn request(
        &self,
        agent_id: AgentId,
        action: String,
        details: String,
    ) -> (ApprovalRequest, ApprovalStatus) {
        let request = self.submit_request(agent_id, action, details).await;
        let mut guard = PendingRequestGuard {
            gate: self.clone(),
            request_id: request.id.clone(),
            armed: true,
        };
        let status = self.await_decision(&request.id).await;
        guard.armed = false;
        (request, status)
    }

    /// Wait for an approval decision on a request.
    pub async fn await_decision(&self, request_id: &str) -> ApprovalStatus {
        if *self
            .inner
            .auto_approve
            .lock()
            .expect("approval auto-approve lock poisoned")
        {
            let fingerprint = self
                .get_request(request_id)
                .await
                .map(|request| request.argument_fingerprint);
            if let Some(fingerprint) = fingerprint {
                if self
                    .approve(request_id, &fingerprint)
                    .await
                    .unwrap_or(false)
                {
                    return ApprovalStatus::Approved;
                }
                return self
                    .get_request(request_id)
                    .await
                    .map(|request| request.status)
                    .unwrap_or(ApprovalStatus::Denied);
            }
            return ApprovalStatus::Denied;
        }

        let (mut decision_rx, remaining) = {
            let pending = self
                .inner
                .pending
                .lock()
                .expect("approval gate lock poisoned");
            let Some(entry) = pending.get(request_id) else {
                return ApprovalStatus::Denied;
            };
            (
                entry.decision_tx.subscribe(),
                entry
                    .request
                    .expires_at
                    .signed_duration_since(Utc::now())
                    .to_std(),
            )
        };

        let Ok(remaining) = remaining else {
            self.transition(request_id, ApprovalStatus::Expired);
            return ApprovalStatus::Expired;
        };

        let wait = async {
            loop {
                let status = *decision_rx.borrow_and_update();
                if status != ApprovalStatus::Pending {
                    return status;
                }
                if decision_rx.changed().await.is_err() {
                    return ApprovalStatus::Denied;
                }
            }
        };

        match tokio::time::timeout(remaining, wait).await {
            Ok(status) => status,
            Err(_) => {
                warn!(request_id = %request_id, "Approval request timed out");
                if self.transition(request_id, ApprovalStatus::Expired) {
                    ApprovalStatus::Expired
                } else {
                    self.get_request(request_id)
                        .await
                        .map(|request| request.status)
                        .unwrap_or(ApprovalStatus::Denied)
                }
            }
        }
    }

    /// Approve a pending request if the supplied argument fingerprint matches.
    ///
    /// A mismatch invalidates and denies the request.
    pub async fn approve(&self, request_id: &str, argument_fingerprint: &str) -> OdinResult<bool> {
        let matches = self
            .get_request(request_id)
            .await
            .filter(|request| request.status == ApprovalStatus::Pending)
            .is_some_and(|request| request.argument_fingerprint == argument_fingerprint);
        if !matches {
            warn!(request_id = %request_id, "Approval fingerprint mismatch or request is not pending");
            self.transition(request_id, ApprovalStatus::Denied);
            return Ok(false);
        }

        let changed = self.transition(request_id, ApprovalStatus::Approved);
        if changed {
            info!(request_id = %request_id, "Request approved");
        }
        Ok(changed)
    }

    /// Deny a pending request.
    pub async fn deny(&self, request_id: &str) -> OdinResult<bool> {
        Ok(self.transition(request_id, ApprovalStatus::Denied))
    }

    /// Deny every pending request, such as when a responder disconnects.
    pub fn disconnect(&self) -> usize {
        let ids: Vec<String> = self
            .inner
            .pending
            .lock()
            .expect("approval gate lock poisoned")
            .iter()
            .filter(|(_, entry)| entry.request.status == ApprovalStatus::Pending)
            .map(|(id, _)| id.clone())
            .collect();
        ids.iter()
            .filter(|id| self.transition(id, ApprovalStatus::Denied))
            .count()
    }

    /// Get all pending requests.
    pub async fn pending_requests(&self) -> Vec<ApprovalRequest> {
        self.expire_due_requests();
        self.inner
            .pending
            .lock()
            .expect("approval gate lock poisoned")
            .values()
            .filter(|entry| entry.request.status == ApprovalStatus::Pending)
            .map(|entry| entry.request.clone())
            .collect()
    }

    /// Get a specific request by ID.
    pub async fn get_request(&self, request_id: &str) -> Option<ApprovalRequest> {
        self.expire_due_requests();
        self.inner
            .pending
            .lock()
            .expect("approval gate lock poisoned")
            .get(request_id)
            .map(|entry| entry.request.clone())
    }

    /// Set auto-approve mode.
    pub async fn set_auto_approve(&self, enabled: bool) {
        *self
            .inner
            .auto_approve
            .lock()
            .expect("approval auto-approve lock poisoned") = enabled;
        info!(auto_approve = enabled, "Auto-approve mode changed");
    }

    /// Check if auto-approve is enabled.
    pub async fn is_auto_approve(&self) -> bool {
        *self
            .inner
            .auto_approve
            .lock()
            .expect("approval auto-approve lock poisoned")
    }

    /// Remove completed requests.
    pub async fn cleanup_expired(&self) -> usize {
        self.expire_due_requests();
        let mut pending = self
            .inner
            .pending
            .lock()
            .expect("approval gate lock poisoned");
        let before = pending.len();
        pending.retain(|_, entry| entry.request.status == ApprovalStatus::Pending);
        let cleaned = before - pending.len();
        if cleaned > 0 {
            debug!(count = cleaned, "Cleaned up completed approval requests");
        }
        cleaned
    }

    fn transition(&self, request_id: &str, status: ApprovalStatus) -> bool {
        let mut pending = self
            .inner
            .pending
            .lock()
            .expect("approval gate lock poisoned");
        let Some(entry) = pending.get_mut(request_id) else {
            return false;
        };
        if entry.request.status != ApprovalStatus::Pending {
            return false;
        }
        let effective_status = if Utc::now() >= entry.request.expires_at {
            ApprovalStatus::Expired
        } else {
            status
        };
        entry.request.status = effective_status;
        entry.decision_tx.send_replace(effective_status);
        effective_status == status
    }

    fn expire_due_requests(&self) -> usize {
        let now = Utc::now();
        let mut expired = 0;
        let mut pending = self
            .inner
            .pending
            .lock()
            .expect("approval gate lock poisoned");
        for entry in pending.values_mut().filter(|entry| {
            entry.request.status == ApprovalStatus::Pending && now >= entry.request.expires_at
        }) {
            entry.request.status = ApprovalStatus::Expired;
            entry.decision_tx.send_replace(ApprovalStatus::Expired);
            expired += 1;
        }
        expired
    }
}

struct PendingRequestGuard {
    gate: ApprovalGate,
    request_id: String,
    armed: bool,
}

impl Drop for PendingRequestGuard {
    fn drop(&mut self) {
        if self.armed {
            self.gate
                .transition(&self.request_id, ApprovalStatus::Denied);
        }
    }
}

impl Default for ApprovalGate {
    fn default() -> Self {
        Self::new(false, 30)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    async fn request(gate: &ApprovalGate, details: &str) -> ApprovalRequest {
        gate.submit_request(Uuid::new_v4(), "shell".into(), details.into())
            .await
    }

    #[tokio::test]
    async fn approves_matching_request() {
        let gate = ApprovalGate::new(false, 30);
        let request = request(&gate, r#"{"command":"echo ok"}"#).await;
        assert!(
            gate.approve(&request.id, &request.argument_fingerprint)
                .await
                .unwrap()
        );
        assert_eq!(
            gate.await_decision(&request.id).await,
            ApprovalStatus::Approved
        );
    }

    #[tokio::test]
    async fn denies_request() {
        let gate = ApprovalGate::new(false, 30);
        let request = request(&gate, "{}").await;
        assert!(gate.deny(&request.id).await.unwrap());
        assert_eq!(
            gate.await_decision(&request.id).await,
            ApprovalStatus::Denied
        );
    }

    #[tokio::test]
    async fn auto_approve_completes_request() {
        let gate = ApprovalGate::new(true, 30);
        let request = request(&gate, "{}").await;
        assert_eq!(
            gate.await_decision(&request.id).await,
            ApprovalStatus::Approved
        );
    }

    #[tokio::test]
    async fn expires_request() {
        let gate = ApprovalGate::with_timeout(false, Duration::from_millis(10));
        let request = request(&gate, "{}").await;
        assert_eq!(
            gate.await_decision(&request.id).await,
            ApprovalStatus::Expired
        );
    }

    #[tokio::test]
    async fn expired_request_cannot_be_approved_or_denied() {
        let gate = ApprovalGate::with_timeout(false, Duration::from_millis(1));
        let approval = request(&gate, "{}").await;
        tokio::time::sleep(Duration::from_millis(5)).await;

        assert!(
            !gate
                .approve(&approval.id, &approval.argument_fingerprint)
                .await
                .unwrap()
        );
        assert!(!gate.deny(&approval.id).await.unwrap());
        assert_eq!(
            gate.get_request(&approval.id).await.unwrap().status,
            ApprovalStatus::Expired
        );
    }

    #[tokio::test]
    async fn abandoned_pending_requests_expire_and_are_pruned() {
        let gate = ApprovalGate::with_timeout(false, Duration::from_millis(1));
        let approval = request(&gate, "{}").await;
        tokio::time::sleep(Duration::from_millis(5)).await;

        assert!(gate.pending_requests().await.is_empty());
        assert_eq!(
            gate.get_request(&approval.id).await.unwrap().status,
            ApprovalStatus::Expired
        );
        assert_eq!(gate.cleanup_expired().await, 1);
        assert!(gate.get_request(&approval.id).await.is_none());
    }

    #[tokio::test]
    async fn changed_arguments_invalidate_approval() {
        let gate = ApprovalGate::new(false, 30);
        let request = request(&gate, r#"{"command":"echo safe"}"#).await;
        let changed = ApprovalGate::fingerprint("shell", r#"{"command":"rm -rf /"}"#);
        assert!(!gate.approve(&request.id, &changed).await.unwrap());
        assert_eq!(
            gate.await_decision(&request.id).await,
            ApprovalStatus::Denied
        );
    }

    #[tokio::test]
    async fn concurrent_requests_are_correlated() {
        let gate = ApprovalGate::new(false, 30);
        let first = request(&gate, r#"{"command":"first"}"#).await;
        let second = request(&gate, r#"{"command":"second"}"#).await;

        gate.approve(&second.id, &second.argument_fingerprint)
            .await
            .unwrap();
        gate.deny(&first.id).await.unwrap();

        let (first_status, second_status) = tokio::join!(
            gate.await_decision(&first.id),
            gate.await_decision(&second.id)
        );
        assert_eq!(first_status, ApprovalStatus::Denied);
        assert_eq!(second_status, ApprovalStatus::Approved);
    }

    #[tokio::test]
    async fn disconnect_denies_pending_requests() {
        let gate = ApprovalGate::new(false, 30);
        let first = request(&gate, "{}").await;
        let second = request(&gate, "{}").await;
        assert_eq!(gate.disconnect(), 2);
        assert_eq!(gate.await_decision(&first.id).await, ApprovalStatus::Denied);
        assert_eq!(
            gate.await_decision(&second.id).await,
            ApprovalStatus::Denied
        );
    }

    #[tokio::test]
    async fn dropped_waiter_denies_request() {
        let gate = ApprovalGate::new(false, 30);
        let mut requests = gate.subscribe();
        let waiter = {
            let gate = gate.clone();
            tokio::spawn(async move {
                gate.request(Uuid::new_v4(), "shell".into(), "{}".into())
                    .await
            })
        };
        let request = requests.recv().await.unwrap();
        waiter.abort();
        let _ = waiter.await;
        assert_eq!(
            gate.get_request(&request.id).await.unwrap().status,
            ApprovalStatus::Denied
        );
    }

    #[tokio::test]
    async fn request_details_are_redacted() {
        let gate = ApprovalGate::new(false, 30);
        let secret = "ghp_abcdefghijklmnopqrstuvwxyz123456";
        let request = request(
            &gate,
            &format!(r#"{{"token":"{secret}","email":"user@example.com"}}"#),
        )
        .await;
        assert!(!request.details.contains(secret));
        assert!(!request.details.contains("user@example.com"));
        assert!(request.details.contains("[REDACTED:"));
    }

    #[tokio::test]
    async fn cleanup_removes_completed_requests() {
        let gate = ApprovalGate::new(false, 30);
        let request = request(&gate, "{}").await;
        gate.deny(&request.id).await.unwrap();
        assert_eq!(gate.cleanup_expired().await, 1);
    }
}
