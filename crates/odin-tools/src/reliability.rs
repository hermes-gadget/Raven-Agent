//! Persistent per-tool reliability scoring for Raven Agent.
//!
//! Production tool attempts are stored as bounded, redacted samples. The store
//! deliberately contains only the tool name, outcome classification, duration,
//! and completion timestamp; arguments, output, and error text are never stored.

use chrono::{DateTime, Utc};
use odin_core::error::{OdinError, OdinResult};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, RwLock};
use std::time::Duration;

/// Configuration for the reliability tracker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityConfig {
    /// Maximum number of recent calls to retain per tool.
    pub window_size: usize,
    /// Half-life for exponential decay (older results count less).
    pub half_life: Duration,
    /// Default score for tools with no data.
    pub default_score: f64,
    /// Minimum score below which a tool is considered unreliable.
    pub alert_threshold: f64,
}

impl Default for ReliabilityConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            half_life: Duration::from_secs(3600),
            default_score: 0.5,
            alert_threshold: 0.7,
        }
    }
}

/// Redacted classification for a production tool attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReliabilityOutcome {
    Success,
    PolicyDenial,
    ValidationFailure,
    TransportFailure,
    ToolFailure,
}

impl ReliabilityOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::PolicyDenial => "policy_denial",
            Self::ValidationFailure => "validation_failure",
            Self::TransportFailure => "transport_failure",
            Self::ToolFailure => "tool_failure",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "success" => Some(Self::Success),
            "policy_denial" => Some(Self::PolicyDenial),
            "validation_failure" => Some(Self::ValidationFailure),
            "transport_failure" => Some(Self::TransportFailure),
            "tool_failure" => Some(Self::ToolFailure),
            _ => None,
        }
    }

    fn is_success(self) -> bool {
        self == Self::Success
    }
}

/// Classify a tool execution error without retaining its potentially sensitive text.
pub fn classify_tool_error(error: &OdinError) -> ReliabilityOutcome {
    match error {
        OdinError::PermissionDenied(_) => ReliabilityOutcome::PolicyDenial,
        OdinError::Validation(_) | OdinError::Serialization(_) => {
            ReliabilityOutcome::ValidationFailure
        }
        OdinError::Network(_)
        | OdinError::Timeout(_)
        | OdinError::RateLimit(_)
        | OdinError::Provider { .. } => ReliabilityOutcome::TransportFailure,
        OdinError::ProviderPermanent { .. } => ReliabilityOutcome::ToolFailure,
        OdinError::Tool {
            source: Some(source),
            ..
        } if source.is::<serde_json::Error>() => ReliabilityOutcome::ValidationFailure,
        _ => ReliabilityOutcome::ToolFailure,
    }
}

/// A single redacted tool call sample.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallRecord {
    pub outcome: ReliabilityOutcome,
    pub duration_ms: u64,
    pub timestamp: DateTime<Utc>,
}

impl CallRecord {
    pub fn new(outcome: ReliabilityOutcome, duration_ms: u64) -> Self {
        Self {
            outcome,
            duration_ms,
            timestamp: Utc::now(),
        }
    }

    pub fn success(duration_ms: u64) -> Self {
        Self::new(ReliabilityOutcome::Success, duration_ms)
    }

    pub fn failure(duration_ms: u64, _error: impl Into<String>) -> Self {
        Self::new(ReliabilityOutcome::ToolFailure, duration_ms)
    }

    fn age(&self) -> Duration {
        (Utc::now() - self.timestamp)
            .to_std()
            .unwrap_or(Duration::ZERO)
    }

    fn weight(&self, half_life: Duration) -> f64 {
        let hl_secs = half_life.as_secs_f64();
        if hl_secs <= 0.0 {
            return 1.0;
        }
        2.0_f64.powf(-self.age().as_secs_f64() / hl_secs)
    }
}

/// Reliability information for a single tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolReliability {
    pub tool_name: String,
    pub score: f64,
    pub total_calls: usize,
    pub success_count: usize,
    pub failure_count: usize,
    pub policy_denial_count: usize,
    pub validation_failure_count: usize,
    pub transport_failure_count: usize,
    pub tool_failure_count: usize,
    pub success_rate: f64,
    pub avg_duration_ms: f64,
    pub is_unreliable: bool,
    pub calls_until_mature: usize,
    pub latest_timestamp: Option<DateTime<Utc>>,
}

struct ReliabilityStore {
    connection: Mutex<Connection>,
}

impl ReliabilityStore {
    fn open(path: &Path) -> OdinResult<Self> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|error| {
                OdinError::Database(format!(
                    "failed to create reliability directory '{}': {error}",
                    parent.display()
                ))
            })?;
        }
        let connection = Connection::open(path).map_err(store_error)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
                .map_err(|error| OdinError::Io(std::io::Error::other(error)))?;
        }
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(store_error)?;
        connection
            .execute_batch(
                "PRAGMA journal_mode=WAL;
                 CREATE TABLE IF NOT EXISTS reliability_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    outcome TEXT NOT NULL CHECK (
                        outcome IN (
                            'success',
                            'policy_denial',
                            'validation_failure',
                            'transport_failure',
                            'tool_failure'
                        )
                    ),
                    duration_ms INTEGER NOT NULL,
                    timestamp TEXT NOT NULL
                 );
                 CREATE INDEX IF NOT EXISTS idx_reliability_tool_recent
                    ON reliability_samples(tool_name, id DESC);",
            )
            .map_err(store_error)?;
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    fn load(&self) -> OdinResult<HashMap<String, Vec<CallRecord>>> {
        let connection = self
            .connection
            .lock()
            .map_err(|error| OdinError::Database(format!("reliability lock poisoned: {error}")))?;
        let mut statement = connection
            .prepare(
                "SELECT tool_name, outcome, duration_ms, timestamp
                 FROM reliability_samples ORDER BY id ASC",
            )
            .map_err(store_error)?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, u64>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })
            .map_err(store_error)?;

        let mut records: HashMap<String, Vec<CallRecord>> = HashMap::new();
        for row in rows {
            let (tool_name, outcome, duration_ms, timestamp) = row.map_err(store_error)?;
            let Some(outcome) = ReliabilityOutcome::parse(&outcome) else {
                tracing::warn!(tool = %tool_name, "ignoring unknown reliability outcome");
                continue;
            };
            let timestamp = DateTime::parse_from_rfc3339(&timestamp)
                .map_err(|error| {
                    OdinError::Database(format!("invalid reliability timestamp: {error}"))
                })?
                .with_timezone(&Utc);
            records.entry(tool_name).or_default().push(CallRecord {
                outcome,
                duration_ms,
                timestamp,
            });
        }
        Ok(records)
    }

    fn append(&self, tool_name: &str, record: &CallRecord, window_size: usize) -> OdinResult<()> {
        let mut connection = self
            .connection
            .lock()
            .map_err(|error| OdinError::Database(format!("reliability lock poisoned: {error}")))?;
        let transaction = connection.transaction().map_err(store_error)?;
        transaction
            .execute(
                "INSERT INTO reliability_samples
                    (tool_name, outcome, duration_ms, timestamp)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    tool_name,
                    record.outcome.as_str(),
                    record.duration_ms,
                    record.timestamp.to_rfc3339()
                ],
            )
            .map_err(store_error)?;
        transaction
            .execute(
                "DELETE FROM reliability_samples
                 WHERE tool_name = ?1
                   AND id NOT IN (
                       SELECT id FROM reliability_samples
                       WHERE tool_name = ?1
                       ORDER BY id DESC LIMIT ?2
                   )",
                params![tool_name, window_size],
            )
            .map_err(store_error)?;
        transaction.commit().map_err(store_error)
    }

    fn reset(&self, tool_name: Option<&str>) -> OdinResult<()> {
        let connection = self
            .connection
            .lock()
            .map_err(|error| OdinError::Database(format!("reliability lock poisoned: {error}")))?;
        match tool_name {
            Some(tool_name) => connection
                .execute(
                    "DELETE FROM reliability_samples WHERE tool_name = ?1",
                    [tool_name],
                )
                .map_err(store_error)?,
            None => connection
                .execute("DELETE FROM reliability_samples", [])
                .map_err(store_error)?,
        };
        Ok(())
    }
}

fn store_error(error: rusqlite::Error) -> OdinError {
    OdinError::Database(format!("reliability store error: {error}"))
}

/// Thread-safe reliability tracker backed by an optional SQLite store.
pub struct ReliabilityTracker {
    config: ReliabilityConfig,
    records: RwLock<HashMap<String, Vec<CallRecord>>>,
    store: Option<std::sync::Arc<ReliabilityStore>>,
}

impl Default for ReliabilityTracker {
    fn default() -> Self {
        Self::new(ReliabilityConfig::default())
    }
}

impl ReliabilityTracker {
    pub fn new(config: ReliabilityConfig) -> Self {
        Self {
            config,
            records: RwLock::new(HashMap::new()),
            store: None,
        }
    }

    /// Open a tracker that reads and writes the shared bounded SQLite store.
    pub fn persistent(path: impl AsRef<Path>, config: ReliabilityConfig) -> OdinResult<Self> {
        let store = std::sync::Arc::new(ReliabilityStore::open(path.as_ref())?);
        let mut records = store.load()?;
        for entry in records.values_mut() {
            trim(entry, config.window_size);
        }
        Ok(Self {
            config,
            records: RwLock::new(records),
            store: Some(store),
        })
    }

    pub fn record_success(&self, tool_name: &str, duration_ms: u64) {
        self.record(tool_name, CallRecord::success(duration_ms));
    }

    pub fn record_failure(&self, tool_name: &str, duration_ms: u64, error: impl Into<String>) {
        self.record(tool_name, CallRecord::failure(duration_ms, error));
    }

    pub fn record_outcome(&self, tool_name: &str, outcome: ReliabilityOutcome, duration_ms: u64) {
        self.record(tool_name, CallRecord::new(outcome, duration_ms));
    }

    /// Record a sample while moving the synchronous SQLite append to Tokio's
    /// blocking pool. The in-memory score is updated immediately; the
    /// persistent sample is awaited so callers know the append task finished.
    pub async fn record_outcome_async(
        &self,
        tool_name: &str,
        outcome: ReliabilityOutcome,
        duration_ms: u64,
    ) {
        let record = CallRecord::new(outcome, duration_ms);
        self.record_in_memory(tool_name, record.clone());
        let Some(store) = self.store.as_ref().cloned() else {
            return;
        };
        let tool_name = tool_name.to_owned();
        let worker_tool_name = tool_name.clone();
        let window_size = self.config.window_size;
        let result = tokio::task::spawn_blocking(move || {
            store.append(&worker_tool_name, &record, window_size)
        })
        .await;
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                tracing::warn!(tool = %tool_name, %error, "failed to persist reliability sample");
            }
            Err(error) => {
                tracing::warn!(tool = %tool_name, %error, "reliability persistence worker failed");
            }
        }
    }

    pub fn record(&self, tool_name: &str, record: CallRecord) {
        if let Some(store) = &self.store
            && let Err(error) = store.append(tool_name, &record, self.config.window_size)
        {
            tracing::warn!(tool = %tool_name, %error, "failed to persist reliability sample");
        }
        self.record_in_memory(tool_name, record);
    }

    fn record_in_memory(&self, tool_name: &str, record: CallRecord) {
        let mut records = self
            .records
            .write()
            .expect("reliability tracker lock poisoned");
        let entry = records.entry(tool_name.to_string()).or_default();
        entry.push(record);
        trim(entry, self.config.window_size);
    }

    pub fn score(&self, tool_name: &str) -> f64 {
        let records = self
            .records
            .read()
            .expect("reliability tracker lock poisoned");
        records
            .get(tool_name)
            .map_or(self.config.default_score, |entry| self.compute_score(entry))
    }

    pub fn get(&self, tool_name: &str) -> ToolReliability {
        let records = self
            .records
            .read()
            .expect("reliability tracker lock poisoned");
        self.summarize(tool_name, records.get(tool_name).map(Vec::as_slice))
    }

    pub fn all(&self) -> Vec<ToolReliability> {
        let records = self
            .records
            .read()
            .expect("reliability tracker lock poisoned");
        let mut results: Vec<_> = records
            .iter()
            .filter(|(_, samples)| !samples.is_empty())
            .map(|(name, samples)| self.summarize(name, Some(samples)))
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.tool_name.cmp(&b.tool_name))
        });
        results
    }

    pub fn unreliable(&self) -> Vec<ToolReliability> {
        self.all().into_iter().filter(|r| r.is_unreliable).collect()
    }

    pub fn reset(&self, tool_name: &str) {
        if let Some(store) = &self.store
            && let Err(error) = store.reset(Some(tool_name))
        {
            tracing::warn!(tool = %tool_name, %error, "failed to reset persisted reliability");
        }
        self.records
            .write()
            .expect("reliability tracker lock poisoned")
            .remove(tool_name);
    }

    pub fn reset_all(&self) {
        if let Some(store) = &self.store
            && let Err(error) = store.reset(None)
        {
            tracing::warn!(%error, "failed to reset persisted reliability");
        }
        self.records
            .write()
            .expect("reliability tracker lock poisoned")
            .clear();
    }

    pub fn tool_count(&self) -> usize {
        self.records
            .read()
            .expect("reliability tracker lock poisoned")
            .values()
            .filter(|entry| !entry.is_empty())
            .count()
    }

    pub fn sample_count(&self) -> usize {
        self.records
            .read()
            .expect("reliability tracker lock poisoned")
            .values()
            .map(Vec::len)
            .sum()
    }

    fn summarize(&self, tool_name: &str, entry: Option<&[CallRecord]>) -> ToolReliability {
        let entry = entry.unwrap_or_default();
        let total = entry.len();
        let count = |outcome| entry.iter().filter(|r| r.outcome == outcome).count();
        let successes = count(ReliabilityOutcome::Success);
        let policy_denials = count(ReliabilityOutcome::PolicyDenial);
        let validation_failures = count(ReliabilityOutcome::ValidationFailure);
        let transport_failures = count(ReliabilityOutcome::TransportFailure);
        let tool_failures = count(ReliabilityOutcome::ToolFailure);
        let avg_ms = if total == 0 {
            0.0
        } else {
            entry.iter().map(|r| r.duration_ms as f64).sum::<f64>() / total as f64
        };
        let score = self.compute_score(entry);
        ToolReliability {
            tool_name: tool_name.to_string(),
            score,
            total_calls: total,
            success_count: successes,
            failure_count: total.saturating_sub(successes),
            policy_denial_count: policy_denials,
            validation_failure_count: validation_failures,
            transport_failure_count: transport_failures,
            tool_failure_count: tool_failures,
            success_rate: if total == 0 {
                0.0
            } else {
                successes as f64 / total as f64
            },
            avg_duration_ms: avg_ms,
            is_unreliable: total > 0 && score < self.config.alert_threshold,
            calls_until_mature: self.config.window_size.saturating_sub(total),
            latest_timestamp: entry.last().map(|record| record.timestamp),
        }
    }

    fn compute_score(&self, records: &[CallRecord]) -> f64 {
        if records.is_empty() {
            return self.config.default_score;
        }
        let (weighted_success, total_weight) =
            records.iter().fold((0.0, 0.0), |(success, total), record| {
                let weight = record.weight(self.config.half_life);
                (
                    success
                        + if record.outcome.is_success() {
                            weight
                        } else {
                            0.0
                        },
                    total + weight,
                )
            });
        if total_weight <= 0.0 {
            self.config.default_score
        } else {
            weighted_success / total_weight
        }
    }
}

fn trim(records: &mut Vec<CallRecord>, window_size: usize) {
    if records.len() > window_size {
        records.drain(0..records.len() - window_size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_state_is_explicit() {
        let tracker = ReliabilityTracker::default();
        assert_eq!(tracker.sample_count(), 0);
        assert!(tracker.all().is_empty());
        assert_eq!(tracker.score("missing"), 0.5);
    }

    #[test]
    fn classifications_remain_distinct() {
        let tracker = ReliabilityTracker::default();
        for outcome in [
            ReliabilityOutcome::Success,
            ReliabilityOutcome::PolicyDenial,
            ReliabilityOutcome::ValidationFailure,
            ReliabilityOutcome::TransportFailure,
            ReliabilityOutcome::ToolFailure,
        ] {
            tracker.record_outcome("tool", outcome, 10);
        }
        let info = tracker.get("tool");
        assert_eq!(info.total_calls, 5);
        assert_eq!(info.success_count, 1);
        assert_eq!(info.policy_denial_count, 1);
        assert_eq!(info.validation_failure_count, 1);
        assert_eq!(info.transport_failure_count, 1);
        assert_eq!(info.tool_failure_count, 1);
    }

    #[test]
    fn persistent_store_survives_restart_and_is_bounded() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("reliability.db");
        let config = ReliabilityConfig {
            window_size: 3,
            ..ReliabilityConfig::default()
        };
        {
            let tracker = ReliabilityTracker::persistent(&path, config.clone()).unwrap();
            tracker.record_outcome("tool", ReliabilityOutcome::PolicyDenial, 1);
            tracker.record_outcome("tool", ReliabilityOutcome::ValidationFailure, 2);
            tracker.record_outcome("tool", ReliabilityOutcome::TransportFailure, 3);
            tracker.record_outcome("tool", ReliabilityOutcome::Success, 4);
        }
        let tracker = ReliabilityTracker::persistent(&path, config).unwrap();
        let info = tracker.get("tool");
        assert_eq!(info.total_calls, 3);
        assert_eq!(info.policy_denial_count, 0);
        assert_eq!(info.validation_failure_count, 1);
        assert_eq!(info.transport_failure_count, 1);
        assert_eq!(info.success_count, 1);
        assert_eq!(info.avg_duration_ms, 3.0);
        assert!(info.latest_timestamp.is_some());
    }

    #[tokio::test]
    async fn async_record_persists_through_the_blocking_worker() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("reliability-async.db");
        let tracker = ReliabilityTracker::persistent(&path, ReliabilityConfig::default()).unwrap();

        tracker
            .record_outcome_async("tool", ReliabilityOutcome::TransportFailure, 7)
            .await;
        assert_eq!(tracker.get("tool").transport_failure_count, 1);

        drop(tracker);
        let reopened = ReliabilityTracker::persistent(&path, ReliabilityConfig::default()).unwrap();
        assert_eq!(reopened.get("tool").transport_failure_count, 1);
    }

    #[test]
    fn recent_successes_replace_old_failures() {
        let tracker = ReliabilityTracker::new(ReliabilityConfig {
            window_size: 2,
            ..ReliabilityConfig::default()
        });
        tracker.record_outcome("tool", ReliabilityOutcome::ToolFailure, 5);
        tracker.record_outcome("tool", ReliabilityOutcome::ToolFailure, 5);
        tracker.record_success("tool", 5);
        tracker.record_success("tool", 5);
        assert!(tracker.score("tool") > 0.99);
    }

    #[test]
    fn error_classification_does_not_use_messages() {
        assert_eq!(
            classify_tool_error(&OdinError::PermissionDenied("secret".into())),
            ReliabilityOutcome::PolicyDenial
        );
        assert_eq!(
            classify_tool_error(&OdinError::Validation("secret".into())),
            ReliabilityOutcome::ValidationFailure
        );
        assert_eq!(
            classify_tool_error(&OdinError::Timeout("secret".into())),
            ReliabilityOutcome::TransportFailure
        );
        assert_eq!(
            classify_tool_error(&OdinError::tool("tool", "secret")),
            ReliabilityOutcome::ToolFailure
        );
        assert_eq!(
            classify_tool_error(&OdinError::provider_permanent("provider", "bad request")),
            ReliabilityOutcome::ToolFailure
        );
    }
}
