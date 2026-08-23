//! Cross-process orchestration run control.
//!
//! Live owners (CLI `raven run`, TUI runner, or a host process) poll pending
//! control commands for a graph UUID. Other processes and authorized WebSocket
//! clients enqueue pause/resume/cancel commands against the same SQLite store.
//! Persisted graph status stays consistent with claimed commands.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Kind of live control request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunControlKind {
    Pause,
    Resume,
    Cancel,
}

impl RunControlKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pause => "pause",
            Self::Resume => "resume",
            Self::Cancel => "cancel",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "pause" => Some(Self::Pause),
            "resume" => Some(Self::Resume),
            "cancel" => Some(Self::Cancel),
            _ => None,
        }
    }
}

/// Lifecycle of a control command row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunControlStatus {
    Pending,
    Claimed,
    Applied,
}

impl RunControlStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Claimed => "claimed",
            Self::Applied => "applied",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "pending" => Some(Self::Pending),
            "claimed" => Some(Self::Claimed),
            "applied" => Some(Self::Applied),
            _ => None,
        }
    }
}

/// One durable control command targeted at a graph UUID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunControlCommand {
    pub id: Uuid,
    pub graph_id: String,
    pub kind: RunControlKind,
    pub reason: Option<String>,
    pub source: String,
    pub status: RunControlStatus,
    pub created_at: DateTime<Utc>,
    pub claimed_at: Option<DateTime<Utc>>,
    pub applied_at: Option<DateTime<Utc>>,
}

impl RunControlCommand {
    pub fn new(
        graph_id: impl Into<String>,
        kind: RunControlKind,
        source: impl Into<String>,
        reason: Option<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            graph_id: graph_id.into(),
            kind,
            reason,
            source: source.into(),
            status: RunControlStatus::Pending,
            created_at: Utc::now(),
            claimed_at: None,
            applied_at: None,
        }
    }
}

/// Result of attempting to authorize a remote control request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlAuth {
    Allowed,
    Denied(&'static str),
}

/// Authorize a remote control client.
///
/// Remote control fails closed unless a non-empty expected token is configured
/// and the caller supplies the exact token.
pub fn authorize_control(
    expected_token: Option<&str>,
    provided_token: Option<&str>,
) -> ControlAuth {
    match (expected_token, provided_token) {
        (Some(expected), Some(provided))
            if !expected.is_empty()
                && constant_time_eq(expected.as_bytes(), provided.as_bytes()) =>
        {
            ControlAuth::Allowed
        }
        _ => ControlAuth::Denied("missing or invalid control token"),
    }
}

fn constant_time_eq(expected: &[u8], provided: &[u8]) -> bool {
    if expected.len() != provided.len() {
        return false;
    }
    expected
        .iter()
        .zip(provided)
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        })
        == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_kind_roundtrips() {
        for kind in [
            RunControlKind::Pause,
            RunControlKind::Resume,
            RunControlKind::Cancel,
        ] {
            assert_eq!(RunControlKind::parse(kind.as_str()), Some(kind));
        }
    }

    #[test]
    fn authorize_control_fails_closed_without_token() {
        assert_eq!(
            authorize_control(None, None),
            ControlAuth::Denied("missing or invalid control token")
        );
        assert_eq!(
            authorize_control(Some(""), Some("")),
            ControlAuth::Denied("missing or invalid control token")
        );
        assert_eq!(
            authorize_control(Some("secret"), None),
            ControlAuth::Denied("missing or invalid control token")
        );
        assert_eq!(
            authorize_control(Some("secret"), Some("wrong")),
            ControlAuth::Denied("missing or invalid control token")
        );
        assert_eq!(
            authorize_control(Some("secret"), Some("secret")),
            ControlAuth::Allowed
        );
    }
}
