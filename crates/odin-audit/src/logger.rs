//! Audit logger implementation for Raven Agent.
//!
//! Implements the [`AuditLogger`] trait from odin-core with support for:
//! - JSON file logging
//! - Querying by agent ID, session ID, and event type

use async_trait::async_trait;
use chrono::Utc;
use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::AuditLogger;
use odin_core::types::{AgentId, AuditEntry, AuditEventType, AuditResult, SessionId};
use odin_permissions::redact::SecretRedactor;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info};
use uuid::Uuid;

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

const MAX_AUDIT_QUERY_LIMIT: usize = 1_000;
const AUDIT_READ_CHUNK_BYTES: usize = 16 * 1024;
const MAX_AUDIT_LINE_BYTES: usize = 1024 * 1024;

fn default_history_size() -> usize {
    1_000
}

/// Configuration for the audit logger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLoggerConfig {
    /// Whether audit logging is enabled.
    pub enabled: bool,
    /// Path to a JSON log file (optional).
    pub file_path: Option<PathBuf>,
    /// Whether to log in JSON format.
    pub json_format: bool,
    /// Maximum entries to keep in memory before flushing.
    pub buffer_size: usize,
    /// Maximum recent entries retained in memory for queries.
    #[serde(default = "default_history_size")]
    pub history_size: usize,
    /// Backward-compatible configuration field. Sensitive values are always
    /// redacted; setting this to false no longer disables that safety boundary.
    pub mask_secrets: bool,
}

impl Default for AuditLoggerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            file_path: None,
            json_format: true,
            buffer_size: 100,
            history_size: default_history_size(),
            mask_secrets: true,
        }
    }
}

/// An in-memory audit entry for buffering.
#[derive(Debug, Clone)]
struct BufferedEntry {
    entry: AuditEntry,
}

/// Implementation of the [`AuditLogger`] trait.
///
/// Supports logging to:
/// 1. A JSON lines file (`file_path`)
/// 2. An in-memory buffer (always active for queries)
/// 3. A bounded in-memory history for low-latency queries
pub struct AuditLoggerImpl {
    /// Configuration.
    config: AuditLoggerConfig,
    /// In-memory buffer of recent entries.
    buffer: Arc<RwLock<Vec<BufferedEntry>>>,
    /// Entries retained for queries during this logger's lifetime.
    history: Arc<RwLock<VecDeque<AuditEntry>>>,
    /// File handle (opened lazily).
    file: Arc<Mutex<Option<std::fs::File>>>,
    /// Serializes flushes so a failed write can be retried without overlap.
    flush_lock: Arc<Mutex<()>>,
    /// Secret/PII redactor applied to every audit entry.
    redactor: SecretRedactor,
}

impl AuditLoggerImpl {
    /// Create a new audit logger, failing if a configured sink cannot open.
    pub fn try_new(mut config: AuditLoggerConfig) -> OdinResult<Self> {
        config.buffer_size = config.buffer_size.max(1);
        if config.enabled && config.file_path.is_some() && !config.json_format {
            return Err(OdinError::Config(
                "durable audit sinks must use JSON Lines format".into(),
            ));
        }
        let file = if config.enabled {
            if let Some(ref path) = config.file_path {
                let file = Self::open_file(path)?;
                info!(file_path = %path.display(), "Audit log file opened");
                Some(file)
            } else {
                None
            }
        } else {
            None
        };

        let redactor = SecretRedactor::full();

        Ok(Self {
            config,
            buffer: Arc::new(RwLock::new(Vec::new())),
            history: Arc::new(RwLock::new(VecDeque::new())),
            file: Arc::new(Mutex::new(file)),
            flush_lock: Arc::new(Mutex::new(())),
            redactor,
        })
    }

    /// Backward-compatible constructor for in-memory/test loggers.
    ///
    /// Production composition should use [`Self::try_new`] so a required
    /// durable sink failure aborts startup instead of silently degrading.
    pub fn new(config: AuditLoggerConfig) -> Self {
        Self::try_new(config).expect("audit logger sink must be openable")
    }

    /// Create a new audit logger with default configuration.
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self::new(AuditLoggerConfig::default())
    }

    /// Create the audit logger with a file-based output.
    pub fn with_file(path: impl Into<PathBuf>) -> Self {
        let config = AuditLoggerConfig {
            file_path: Some(path.into()),
            ..AuditLoggerConfig::default()
        };
        Self::new(config)
    }

    /// Helper to open the log file.
    fn open_file(path: &Path) -> OdinResult<std::fs::File> {
        // Create parent directories if needed
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|e| {
                OdinError::Io(std::io::Error::other(format!(
                    "Failed to create audit log directory '{}': {}",
                    parent.display(),
                    e
                )))
            })?;
        }

        let mut options = std::fs::OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        options.mode(0o600);
        let file = options.open(path).map_err(|e| {
            OdinError::Io(std::io::Error::other(format!(
                "Failed to open audit log file '{}': {}",
                path.display(),
                e
            )))
        })?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
                .map_err(OdinError::Io)?;
        }
        Ok(file)
    }

    /// Flush buffered entries to the file.
    async fn flush_to_file(&self) -> OdinResult<()> {
        let _flush_guard = self.flush_lock.lock().await;

        let mut file_guard = self.file.lock().await;
        let file = match file_guard.as_mut() {
            Some(file) => file,
            None => {
                // In-memory loggers are intentionally usable by unit tests and
                // embedders; production builders always configure a file sink.
                return Ok(());
            }
        };

        let original_len = file
            .metadata()
            .map_err(|error| {
                OdinError::Io(std::io::Error::other(format!(
                    "Failed to inspect audit log before flush: {error}"
                )))
            })?
            .len();

        // Serialize while the entries are still in the queue. If this fails,
        // the queue is untouched and the caller can retry safely.
        let (entries, encoded) = {
            let mut buffer = self.buffer.write().await;
            if buffer.is_empty() {
                return Ok(());
            }

            let encoded = self.serialize_entries(&buffer)?;
            let entries = buffer.drain(..).collect::<Vec<_>>();
            (entries, encoded)
        };
        let count = entries.len();

        let write_result = file
            .write_all(&encoded)
            .and_then(|_| file.flush())
            .and_then(|_| file.sync_data());

        if let Err(error) = write_result {
            let rollback_result = file
                .set_len(original_len)
                .and_then(|_| file.seek(SeekFrom::Start(original_len)))
                .and_then(|_| file.flush());

            let message = match rollback_result {
                Ok(()) => format!("Failed to write audit log: {error}"),
                Err(rollback_error) => format!(
                    "Failed to write audit log: {error}; rollback also failed: {rollback_error}"
                ),
            };
            let mut buffer = self.buffer.write().await;
            buffer.splice(0..0, entries);
            return Err(OdinError::Io(std::io::Error::other(message)));
        }

        debug!(count, "Flushed audit entries to file");
        Ok(())
    }

    fn serialize_entries(&self, entries: &[BufferedEntry]) -> OdinResult<Vec<u8>> {
        let mut encoded = Vec::new();
        for buffered in entries {
            if self.config.json_format {
                serde_json::to_writer(&mut encoded, &buffered.entry)
                    .map_err(OdinError::Serialization)?;
            } else {
                write!(
                    encoded,
                    "[{}] [{}] [{}] [{}] {}: {}",
                    buffered.entry.timestamp.to_rfc3339(),
                    buffered.entry.event_type,
                    buffered.entry.agent_id,
                    buffered.entry.session_id,
                    buffered.entry.action,
                    serde_json::to_string(&buffered.entry.details).unwrap_or_default(),
                )
                .map_err(|error| OdinError::Io(std::io::Error::other(error.to_string())))?;
            }
            encoded.push(b'\n');
        }
        Ok(encoded)
    }

    async fn persisted_entries(
        &self,
        agent_id: Option<AgentId>,
        session_id: Option<SessionId>,
        event_type: Option<AuditEventType>,
        limit: usize,
        excluded_ids: HashSet<Uuid>,
    ) -> OdinResult<Vec<AuditEntry>> {
        if !self.config.json_format || limit == 0 {
            return Ok(Vec::new());
        }
        let Some(path) = self.config.file_path.as_ref() else {
            return Ok(Vec::new());
        };

        let path = path.clone();
        tokio::task::spawn_blocking(move || {
            Self::read_persisted_reverse(
                &path,
                agent_id,
                session_id,
                event_type,
                limit,
                &excluded_ids,
            )
        })
        .await
        .map_err(|error| OdinError::Internal(format!("audit log reader task failed: {error}")))?
    }

    fn read_persisted_reverse(
        path: &Path,
        agent_id: Option<AgentId>,
        session_id: Option<SessionId>,
        event_type: Option<AuditEventType>,
        limit: usize,
        excluded_ids: &HashSet<Uuid>,
    ) -> OdinResult<Vec<AuditEntry>> {
        let mut file = match std::fs::File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(OdinError::Io(std::io::Error::other(format!(
                    "Failed to read audit log '{}': {error}",
                    path.display()
                ))));
            }
        };
        let mut position = file.metadata().map_err(OdinError::Io)?.len();
        let mut carry = Vec::new();
        let mut results = Vec::with_capacity(limit);

        while position > 0 && results.len() < limit {
            let read_size = position.min(AUDIT_READ_CHUNK_BYTES as u64) as usize;
            position -= read_size as u64;
            file.seek(SeekFrom::Start(position))
                .map_err(OdinError::Io)?;
            let mut chunk = vec![0; read_size];
            file.read_exact(&mut chunk).map_err(OdinError::Io)?;
            chunk.extend_from_slice(&carry);

            let mut line_end = chunk.len();
            while results.len() < limit {
                let Some(newline) = chunk[..line_end].iter().rposition(|byte| *byte == b'\n')
                else {
                    break;
                };
                Self::push_persisted_match(
                    &chunk[newline + 1..line_end],
                    agent_id,
                    session_id,
                    event_type,
                    excluded_ids,
                    &mut results,
                )?;
                line_end = newline;
            }

            carry = chunk[..line_end].to_vec();
            if carry.len() > MAX_AUDIT_LINE_BYTES {
                return Err(OdinError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "audit log contains an oversized entry",
                )));
            }
        }

        if position == 0 && results.len() < limit && !carry.is_empty() {
            Self::push_persisted_match(
                &carry,
                agent_id,
                session_id,
                event_type,
                excluded_ids,
                &mut results,
            )?;
        }
        Ok(results)
    }

    fn push_persisted_match(
        line: &[u8],
        agent_id: Option<AgentId>,
        session_id: Option<SessionId>,
        event_type: Option<AuditEventType>,
        excluded_ids: &HashSet<Uuid>,
        results: &mut Vec<AuditEntry>,
    ) -> OdinResult<()> {
        if line.iter().all(u8::is_ascii_whitespace) {
            return Ok(());
        }
        if line.len() > MAX_AUDIT_LINE_BYTES {
            return Err(OdinError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "audit log contains an oversized entry",
            )));
        }
        let entry: AuditEntry = serde_json::from_slice(line).map_err(OdinError::Serialization)?;
        if excluded_ids.contains(&entry.id)
            || agent_id.is_some_and(|id| entry.agent_id != id)
            || session_id.is_some_and(|id| entry.session_id != id)
            || event_type.is_some_and(|kind| entry.event_type != kind)
        {
            return Ok(());
        }
        results.push(entry);
        Ok(())
    }

    /// Add an audit entry to the in-memory buffer.
    async fn buffer_entry(&self, entry: AuditEntry) -> OdinResult<()> {
        if self.config.history_size > 0 {
            let mut history = self.history.write().await;
            history.push_back(entry.clone());
            while history.len() > self.config.history_size {
                history.pop_front();
            }
        }
        let buffered = BufferedEntry { entry };

        {
            let mut buffer = self.buffer.write().await;
            buffer.push(buffered);

            // An in-memory-only logger has no durable sink to protect. Keep
            // its historical bounded behavior, but never trim a durable
            // queue: a sink outage must return an error rather than discard
            // audit records that have not reached disk.
            if self.config.file_path.is_none() && buffer.len() > self.config.buffer_size * 2 {
                let excess = buffer.len() - self.config.buffer_size;
                buffer.drain(0..excess);
            }
        }

        // Flush if buffer is large enough. Errors are returned to the caller so
        // a sink outage cannot be mistaken for a durable audit write.
        if self.buffer.read().await.len() >= self.config.buffer_size
            && let Err(e) = self.flush_to_file().await
        {
            return Err(e);
        }

        Ok(())
    }
}

#[async_trait]
impl AuditLogger for AuditLoggerImpl {
    /// Log an audit entry.
    async fn log(&self, entry: AuditEntry) -> OdinResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Audit data always crosses a durable boundary, so redaction cannot be
        // disabled by configuration.
        let entry = self.redact_entry(entry);

        debug!(
            event_type = %entry.event_type,
            agent_id = %entry.agent_id,
            action = %entry.action,
            "Audit entry logged"
        );

        self.buffer_entry(entry).await
    }

    /// Query audit entries by agent, session, and event type.
    async fn query(
        &self,
        agent_id: Option<AgentId>,
        session_id: Option<SessionId>,
        event_type: Option<AuditEventType>,
        limit: usize,
    ) -> OdinResult<Vec<AuditEntry>> {
        let limit = limit.min(MAX_AUDIT_QUERY_LIMIT);
        if limit == 0 {
            return Ok(Vec::new());
        }
        let _file_guard = self.file.lock().await;
        let history = self.history.read().await;
        let history_ids: HashSet<_> = history.iter().map(|entry| entry.id).collect();
        let mut results: Vec<AuditEntry> = history
            .iter()
            .rev()
            .filter(|entry| {
                let mut matches = true;
                if let Some(ref aid) = agent_id {
                    matches = matches && entry.agent_id == *aid;
                }
                if let Some(ref sid) = session_id {
                    matches = matches && entry.session_id == *sid;
                }
                if let Some(ref et) = event_type {
                    matches = matches && entry.event_type == *et;
                }
                matches
            })
            .take(limit)
            .cloned()
            .collect();
        drop(history);

        if results.len() < limit {
            results.extend(
                self.persisted_entries(
                    agent_id,
                    session_id,
                    event_type,
                    limit - results.len(),
                    history_ids,
                )
                .await?,
            );
        }

        Ok(results)
    }

    /// Get the most recent entries.
    async fn recent(&self, limit: usize) -> OdinResult<Vec<AuditEntry>> {
        let limit = limit.min(MAX_AUDIT_QUERY_LIMIT);
        if limit == 0 {
            return Ok(Vec::new());
        }
        let _file_guard = self.file.lock().await;
        let history = self.history.read().await;
        let history_ids: HashSet<_> = history.iter().map(|entry| entry.id).collect();
        let mut results: Vec<AuditEntry> = history.iter().rev().take(limit).cloned().collect();
        drop(history);

        if results.len() < limit {
            results.extend(
                self.persisted_entries(None, None, None, limit - results.len(), history_ids)
                    .await?,
            );
        }

        Ok(results)
    }
}

impl AuditLoggerImpl {
    /// Force flush buffered entries to disk.
    pub async fn flush(&self) -> OdinResult<()> {
        self.flush_to_file().await
    }

    /// Rotate the log file (close and reopen).
    pub async fn rotate(&self, new_path: Option<PathBuf>) -> OdinResult<()> {
        let path = new_path.or_else(|| self.config.file_path.clone());
        let path = match path {
            Some(p) => p,
            None => return Err(OdinError::Config("No log file path configured".into())),
        };

        let new_file = Self::open_file(&path)?;
        let mut file_guard = self.file.lock().await;
        *file_guard = Some(new_file);

        info!(file_path = %path.display(), "Audit log file rotated");
        Ok(())
    }

    /// Get the number of buffered entries.
    pub async fn buffer_size(&self) -> usize {
        self.buffer.read().await.len()
    }

    /// Clear all buffered entries.
    pub async fn clear_buffer(&self) {
        self.buffer.write().await.clear();
        debug!("Audit buffer cleared");
    }

    /// Redact secrets and PII from an audit entry before logging.
    fn redact_entry(&self, mut entry: AuditEntry) -> AuditEntry {
        // Redact the action string (may contain command line args with secrets).
        entry.action = self.redactor.redact(&entry.action);

        // Redact the details JSON (may contain tool inputs/outputs with secrets).
        entry.details = self.redactor.redact_json(&entry.details);

        entry
    }
}

impl Drop for AuditLoggerImpl {
    fn drop(&mut self) {
        // Normal production writes use a one-entry buffer and graceful owners
        // call flush explicitly. This best-effort synchronous tail flush also
        // protects short-lived embedders that drop a logger without an async
        // shutdown hook.
        let Ok(_flush_guard) = self.flush_lock.try_lock() else {
            return;
        };
        let Ok(mut file_guard) = self.file.try_lock() else {
            return;
        };
        let Some(file) = file_guard.as_mut() else {
            return;
        };
        let Ok(mut buffer) = self.buffer.try_write() else {
            return;
        };
        if buffer.is_empty() {
            return;
        }
        let Ok(encoded) = self.serialize_entries(&buffer) else {
            return;
        };
        if file.write_all(&encoded).is_ok() {
            let _ = file.flush();
            let _ = file.sync_data();
            buffer.clear();
        }
    }
}

/// Create an audit entry builder for convenience.
pub fn audit_entry(
    agent_id: AgentId,
    session_id: SessionId,
    event_type: AuditEventType,
    action: impl Into<String>,
    details: serde_json::Value,
    result: AuditResult,
) -> AuditEntry {
    AuditEntry {
        id: Uuid::new_v4(),
        timestamp: Utc::now(),
        agent_id,
        session_id,
        event_type,
        action: action.into(),
        details,
        result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use odin_core::traits::AuditLogger;
    use odin_core::types::AuditEventType;
    use uuid::Uuid;

    fn make_entry(
        agent_id: AgentId,
        session_id: SessionId,
        event_type: AuditEventType,
    ) -> AuditEntry {
        AuditEntry {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            agent_id,
            session_id,
            event_type,
            action: "test_action".to_string(),
            details: serde_json::json!({"key": "value"}),
            result: AuditResult::Success,
        }
    }

    #[tokio::test]
    async fn test_log_and_recent() {
        let logger = AuditLoggerImpl::default();
        let agent_id = Uuid::new_v4();
        let session_id = Uuid::new_v4();

        let entry = make_entry(agent_id, session_id, AuditEventType::ToolCall);
        logger.log(entry).await.unwrap();

        let recent = logger.recent(10).await.unwrap();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].event_type, AuditEventType::ToolCall);
    }

    #[tokio::test]
    async fn test_query_by_agent() {
        let logger = AuditLoggerImpl::default();
        let agent1 = Uuid::new_v4();
        let agent2 = Uuid::new_v4();
        let session = Uuid::new_v4();

        logger
            .log(make_entry(agent1, session, AuditEventType::ToolCall))
            .await
            .unwrap();
        logger
            .log(make_entry(agent2, session, AuditEventType::ModelCall))
            .await
            .unwrap();

        let results = logger.query(Some(agent1), None, None, 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].agent_id, agent1);
    }

    #[tokio::test]
    async fn test_query_by_event_type() {
        let logger = AuditLoggerImpl::default();
        let agent_id = Uuid::new_v4();
        let session_id = Uuid::new_v4();

        logger
            .log(make_entry(agent_id, session_id, AuditEventType::ToolCall))
            .await
            .unwrap();
        logger
            .log(make_entry(agent_id, session_id, AuditEventType::ModelCall))
            .await
            .unwrap();
        logger
            .log(make_entry(agent_id, session_id, AuditEventType::ToolCall))
            .await
            .unwrap();

        let results = logger
            .query(None, None, Some(AuditEventType::ToolCall), 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_disabled_logger() {
        let config = AuditLoggerConfig {
            enabled: false,
            ..Default::default()
        };
        let logger = AuditLoggerImpl::new(config);
        let agent_id = Uuid::new_v4();
        let session_id = Uuid::new_v4();

        logger
            .log(make_entry(agent_id, session_id, AuditEventType::ToolCall))
            .await
            .unwrap();

        let recent = logger.recent(10).await.unwrap();
        assert_eq!(recent.len(), 0);
    }

    #[tokio::test]
    async fn test_file_logging() {
        let tmp_dir = std::env::temp_dir();
        let log_path = tmp_dir.join(format!("audit_test_{}.jsonl", Uuid::new_v4()));

        let config = AuditLoggerConfig {
            file_path: Some(log_path.clone()),
            json_format: true,
            ..AuditLoggerConfig::default()
        };

        let logger = AuditLoggerImpl::new(config);
        let agent_id = Uuid::new_v4();
        let session_id = Uuid::new_v4();

        logger
            .log(make_entry(
                agent_id,
                session_id,
                AuditEventType::SessionStart,
            ))
            .await
            .unwrap();

        // Flush to ensure it's written
        logger.flush().await.unwrap();

        // Read the file and verify
        let content = std::fs::read_to_string(&log_path).unwrap();
        assert!(content.contains("\"event_type\":\"session_start\""));
        assert!(content.contains(&agent_id.to_string()));

        let recent = logger.recent(10).await.unwrap();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].agent_id, agent_id);
        let queried = logger.query(Some(agent_id), None, None, 10).await.unwrap();
        assert_eq!(queried.len(), 1);

        // Cleanup
        let _ = std::fs::remove_file(&log_path);
    }

    #[test]
    fn startup_fails_when_audit_sink_cannot_be_opened() {
        let directory = std::env::temp_dir();
        let result = AuditLoggerImpl::try_new(AuditLoggerConfig {
            file_path: Some(directory),
            ..AuditLoggerConfig::default()
        });
        let Err(error) = result else {
            panic!("a directory is not a usable audit sink")
        };
        assert!(error.to_string().contains("audit log file"));
    }

    #[tokio::test]
    async fn durable_entries_are_json_and_correlatable() {
        let path = std::env::temp_dir().join(format!("audit-{}.jsonl", Uuid::new_v4()));
        let logger = AuditLoggerImpl::try_new(AuditLoggerConfig {
            file_path: Some(path.clone()),
            buffer_size: 2,
            ..AuditLoggerConfig::default()
        })
        .unwrap();
        let agent_id = Uuid::new_v4();
        let session_id = Uuid::new_v4();
        let first = make_entry(agent_id, session_id, AuditEventType::SessionStart);
        let second = make_entry(agent_id, session_id, AuditEventType::ToolCall);
        let first_id = first.id;
        let second_id = second.id;
        logger.log(first).await.unwrap();
        logger.log(second).await.unwrap();
        logger.flush().await.unwrap();

        let lines: Vec<AuditEntry> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].agent_id, agent_id);
        assert_eq!(lines[0].session_id, session_id);
        assert_eq!(lines[0].id, first_id);
        assert_eq!(lines[1].id, second_id);
        drop(logger);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn durable_text_format_is_rejected() {
        let result = AuditLoggerImpl::try_new(AuditLoggerConfig {
            file_path: Some(std::env::temp_dir().join(format!("audit-{}.log", Uuid::new_v4()))),
            json_format: false,
            ..AuditLoggerConfig::default()
        });
        let Err(error) = result else {
            panic!("durable text output must be rejected")
        };
        assert!(error.to_string().contains("JSON Lines"));
    }

    #[tokio::test]
    async fn test_audit_entry_builder() {
        let agent_id = Uuid::new_v4();
        let session_id = Uuid::new_v4();

        let entry = audit_entry(
            agent_id,
            session_id,
            AuditEventType::ConfigChange,
            "update_config",
            serde_json::json!({"setting": "value"}),
            AuditResult::Success,
        );

        assert_eq!(entry.event_type, AuditEventType::ConfigChange);
        assert_eq!(entry.action, "update_config");
        assert_eq!(entry.result, AuditResult::Success);
    }

    #[tokio::test]
    async fn test_buffer_trimming() {
        let config = AuditLoggerConfig {
            buffer_size: 5,
            enabled: true,
            ..Default::default()
        };
        let logger = AuditLoggerImpl::new(config);
        let agent_id = Uuid::new_v4();
        let session_id = Uuid::new_v4();

        // Add 12 entries — buffer should trim to ~5
        for _ in 0..12 {
            logger
                .log(make_entry(agent_id, session_id, AuditEventType::Decision))
                .await
                .unwrap();
        }

        let size = logger.buffer_size().await;
        assert!(size <= 10); // buffer_size * 2
    }

    #[tokio::test]
    async fn history_is_bounded_and_persisted_queries_read_from_the_tail() {
        let log_path = std::env::temp_dir().join(format!("audit_tail_{}.jsonl", Uuid::new_v4()));
        let logger = AuditLoggerImpl::new(AuditLoggerConfig {
            file_path: Some(log_path.clone()),
            buffer_size: 1,
            history_size: 2,
            ..AuditLoggerConfig::default()
        });
        let agent_id = Uuid::new_v4();
        let session_id = Uuid::new_v4();
        let mut ids = Vec::new();
        for index in 0..6 {
            let mut entry = make_entry(agent_id, session_id, AuditEventType::Decision);
            entry.action = format!("entry-{index}");
            ids.push(entry.id);
            logger.log(entry).await.unwrap();
        }

        assert_eq!(logger.history.read().await.len(), 2);
        let recent = logger.recent(4).await.unwrap();
        assert_eq!(
            recent.iter().map(|entry| entry.id).collect::<Vec<_>>(),
            ids.iter().rev().take(4).copied().collect::<Vec<_>>()
        );
        let queried = logger
            .query(Some(agent_id), None, Some(AuditEventType::Decision), 4)
            .await
            .unwrap();
        assert_eq!(queried.len(), 4);

        let _ = std::fs::remove_file(log_path);
    }

    #[tokio::test]
    async fn test_clear_buffer() {
        let logger = AuditLoggerImpl::default();
        let agent_id = Uuid::new_v4();
        let session_id = Uuid::new_v4();

        logger
            .log(make_entry(agent_id, session_id, AuditEventType::Error))
            .await
            .unwrap();
        assert_eq!(logger.buffer_size().await, 1);

        logger.clear_buffer().await;
        assert_eq!(logger.buffer_size().await, 0);
    }
}
