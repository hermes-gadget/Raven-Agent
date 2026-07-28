//! Scheduler persistence — `SchedulerStore` trait and SQLite implementation.
//!
//! Provides the abstraction for persisting scheduled jobs to durable storage,
//! and a concrete [`SqliteSchedulerStore`] backed by the same SQLite database
//! used by `odin-memory`.

use crate::job::{Job, JobId, Schedule};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use odin_core::error::OdinError;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

/// Durable outcome of one scheduled execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobRunStatus {
    Running,
    Succeeded,
    Failed,
}

impl JobRunStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
        }
    }

    fn parse(value: &str) -> odin_core::error::OdinResult<Self> {
        match value {
            "running" => Ok(Self::Running),
            "succeeded" => Ok(Self::Succeeded),
            "failed" => Ok(Self::Failed),
            other => Err(OdinError::Database(format!(
                "Invalid scheduler run status '{other}'"
            ))),
        }
    }
}

/// Durable scheduler execution record used by health and audit surfaces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRun {
    pub task_id: Uuid,
    pub job_id: JobId,
    pub job_name: String,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub status: JobRunStatus,
    pub error: Option<String>,
}

// ── SchedulerJobConfig ────────────────────────────────────────────────

/// Configuration for a scheduled job that executes an agent task.
///
/// When provided, the scheduler creates an `AgentTask` from this config and
/// submits it to the configured [`Runtime`](odin_runtime::Runtime) instead of
/// running a generic closure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerJobConfig {
    /// The task goal / instruction for the agent.
    pub task_goal: String,
    /// Maximum iterations for the agent loop.
    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,
}

fn default_max_iterations() -> u32 {
    100
}

impl SchedulerJobConfig {
    /// Create a new job config with the given task goal.
    pub fn new(task_goal: impl Into<String>) -> Self {
        Self {
            task_goal: task_goal.into(),
            max_iterations: 100,
        }
    }

    /// Set the maximum iterations for this job.
    pub fn with_max_iterations(mut self, max: u32) -> Self {
        self.max_iterations = max;
        self
    }
}

// ── PersistedJob ──────────────────────────────────────────────────────

/// A serialisable, closure-free snapshot of a [`Job`] used for persistence.
///
/// This is what flows through the store trait because `Job` itself contains
/// an `Arc<dyn Fn …>` which is not `Serialize`/`Deserialize`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedJob {
    /// Unique job identifier.
    pub id: JobId,
    /// Human-readable name.
    pub name: String,
    /// Raw cron expression.
    pub cron_expr: String,
    /// Optional task goal for runtime-driven execution.
    pub task_goal: Option<String>,
    /// Max iterations when executing via runtime.
    pub max_iterations: u32,
    /// Whether the job is enabled.
    pub enabled: bool,
    /// Timestamp of the last run.
    pub last_run: Option<DateTime<Utc>>,
    /// Next scheduled run time.
    pub next_run: Option<DateTime<Utc>>,
    /// Total number of runs.
    pub run_count: u64,
    /// When this job was created.
    pub created_at: DateTime<Utc>,
}

impl PersistedJob {
    /// Build a `PersistedJob` from a `Job` reference.
    pub fn from_job(job: &Job) -> Self {
        Self {
            id: job.id,
            name: job.name.clone(),
            cron_expr: job.schedule.expression.clone(),
            task_goal: job.task_goal.clone(),
            max_iterations: job.max_iterations,
            enabled: job.enabled,
            last_run: job.last_run,
            next_run: job.next_run,
            run_count: job.run_count,
            created_at: job.created_at,
        }
    }

    /// Convert back into a `Job` with an inert closure.
    ///
    /// Runtime-backed jobs are dispatched from `task_goal`; the inert closure
    /// is never executed for those jobs.
    pub fn into_job(self) -> Job {
        let schedule = Schedule::parse(&self.cron_expr)
            .unwrap_or_else(|_| Schedule::parse("* * * * *").unwrap());
        let mut job = Job {
            id: self.id,
            name: self.name,
            schedule,
            task: crate::job::noop_task(),
            enabled: self.enabled,
            last_run: self.last_run,
            next_run: self.next_run,
            last_task_id: None,
            run_count: self.run_count,
            max_concurrent: 1,
            running_count: 0,
            task_goal: self.task_goal,
            max_iterations: self.max_iterations,
            created_at: self.created_at,
        };
        // Ensure next_run is recalculated if missing
        if job.next_run.is_none() {
            job.calculate_next_run();
        }
        job
    }
}

// ── SchedulerStore Trait ──────────────────────────────────────────────

/// Abstraction for persisting scheduler jobs to durable storage.
#[async_trait]
pub trait SchedulerStore: Send + Sync {
    /// Save (insert or update) a job.
    async fn save_job(&self, job: &PersistedJob) -> odin_core::error::OdinResult<()>;

    /// Load all persisted jobs.
    async fn load_all_jobs(&self) -> odin_core::error::OdinResult<Vec<PersistedJob>>;

    /// Delete a job by its ID.
    async fn delete_job(&self, id: &JobId) -> odin_core::error::OdinResult<()>;

    /// Update the runtime state of a job (enabled flag, timestamps, run count).
    async fn update_job_state(
        &self,
        id: &JobId,
        enabled: bool,
        last_run: Option<DateTime<Utc>>,
        next_run: Option<DateTime<Utc>>,
        run_count: u64,
    ) -> odin_core::error::OdinResult<()>;

    /// Record the start of a scheduled execution.
    async fn record_run_started(&self, run: &JobRun) -> odin_core::error::OdinResult<()>;

    /// Record the terminal outcome of a scheduled execution.
    async fn record_run_finished(
        &self,
        task_id: &Uuid,
        status: JobRunStatus,
        error: Option<&str>,
    ) -> odin_core::error::OdinResult<()>;

    /// Return the most recent scheduler executions.
    async fn recent_runs(&self, limit: usize) -> odin_core::error::OdinResult<Vec<JobRun>>;
}

// ── SqliteSchedulerStore ──────────────────────────────────────────────

/// A [`SchedulerStore`] backed by SQLite, sharing the same connection pattern
/// as `odin_memory::SqliteMemoryStore`.
///
/// The `scheduler_jobs` table is created automatically in the constructor.
/// Pass the same `Arc<Mutex<Connection>>` from a `SqliteMemoryStore` to
/// share the same database file.
#[derive(Debug, Clone)]
pub struct SqliteSchedulerStore {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteSchedulerStore {
    /// Open (or create) a SQLite database at the given file path.
    pub fn new(path: &str) -> odin_core::error::OdinResult<Self> {
        let conn = Connection::open(path).map_err(|e| {
            OdinError::Database(format!("Failed to open scheduler database at {path}: {e}"))
        })?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        store.init_tables()?;
        tracing::info!(path = %path, "Opened SQLite scheduler store");
        Ok(store)
    }

    /// Create an in-memory SQLite database (useful for testing).
    pub fn in_memory() -> odin_core::error::OdinResult<Self> {
        let conn = Connection::open_in_memory().map_err(|e| {
            OdinError::Database(format!("Failed to open in-memory scheduler database: {e}"))
        })?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        store.init_tables()?;
        Ok(store)
    }

    /// Share an existing connection (e.g. from `SqliteMemoryStore`).
    ///
    /// Call `init_tables` separately if the connection wasn't already
    /// initialised.
    pub fn from_connection(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Create the `scheduler_jobs` table if it doesn't exist.
    pub fn init_tables(&self) -> odin_core::error::OdinResult<()> {
        let conn = self
            .conn
            .try_lock()
            .expect("store just created, no contention");

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS scheduler_jobs (
                id          TEXT PRIMARY KEY,
                name        TEXT    NOT NULL,
                cron_expr   TEXT    NOT NULL,
                task_goal   TEXT,
                max_iterations INTEGER NOT NULL DEFAULT 100,
                enabled     INTEGER NOT NULL DEFAULT 1,
                last_run    TEXT,
                next_run    TEXT,
                run_count   INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS scheduler_runs (
                task_id     TEXT PRIMARY KEY,
                job_id      TEXT NOT NULL,
                job_name    TEXT NOT NULL,
                started_at  TEXT NOT NULL,
                finished_at TEXT,
                status      TEXT NOT NULL,
                error       TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_scheduler_runs_started_at
                ON scheduler_runs(started_at DESC);",
        )
        .map_err(|e| OdinError::Database(format!("Failed to create scheduler_jobs table: {e}")))?;

        Ok(())
    }
}

#[async_trait]
impl SchedulerStore for SqliteSchedulerStore {
    async fn save_job(&self, job: &PersistedJob) -> odin_core::error::OdinResult<()> {
        let conn = self.conn.lock().await;

        conn.execute(
            "INSERT INTO scheduler_jobs
                (id, name, cron_expr, task_goal, max_iterations, enabled,
                 last_run, next_run, run_count, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(id) DO UPDATE SET
                name           = excluded.name,
                cron_expr      = excluded.cron_expr,
                task_goal      = excluded.task_goal,
                max_iterations = excluded.max_iterations,
                enabled        = excluded.enabled,
                last_run       = excluded.last_run,
                next_run       = excluded.next_run,
                run_count      = excluded.run_count",
            params![
                job.id.to_string(),
                job.name,
                job.cron_expr,
                job.task_goal,
                job.max_iterations,
                job.enabled as i32,
                job.last_run.map(|t| t.to_rfc3339()),
                job.next_run.map(|t| t.to_rfc3339()),
                job.run_count as i64,
                job.created_at.to_rfc3339(),
            ],
        )
        .map_err(|e| OdinError::Database(format!("Failed to save scheduler job: {e}")))?;

        Ok(())
    }

    async fn load_all_jobs(&self) -> odin_core::error::OdinResult<Vec<PersistedJob>> {
        let conn = self.conn.lock().await;

        let mut stmt = conn
            .prepare(
                "SELECT id, name, cron_expr, task_goal, max_iterations, enabled,
                        last_run, next_run, run_count, created_at
                 FROM scheduler_jobs
                 ORDER BY created_at ASC",
            )
            .map_err(|e| OdinError::Database(format!("Failed to prepare load statement: {e}")))?;

        let rows = stmt
            .query_map([], |row| {
                let id_str: String = row.get(0)?;
                let last_run_str: Option<String> = row.get(6)?;
                let next_run_str: Option<String> = row.get(7)?;
                let created_at_str: String = row.get(9)?;

                Ok(PersistedJobRaw {
                    id: id_str,
                    name: row.get(1)?,
                    cron_expr: row.get(2)?,
                    task_goal: row.get(3)?,
                    max_iterations: row.get::<_, i64>(4)? as u32,
                    enabled: row.get::<_, i64>(5)? != 0,
                    last_run: last_run_str,
                    next_run: next_run_str,
                    run_count: row.get::<_, i64>(8)? as u64,
                    created_at: created_at_str,
                })
            })
            .map_err(|e| OdinError::Database(format!("Failed to query scheduler jobs: {e}")))?;

        let mut jobs = Vec::new();
        for row in rows {
            let raw = row.map_err(|e| {
                OdinError::Database(format!("Error reading scheduler job row: {e}"))
            })?;

            let last_run = raw
                .last_run
                .as_deref()
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc));

            let next_run = raw
                .next_run
                .as_deref()
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc));

            let created_at = DateTime::parse_from_rfc3339(&raw.created_at)
                .map_err(|e| OdinError::Database(format!("Invalid created_at: {e}")))?
                .with_timezone(&Utc);

            let id = Uuid::parse_str(&raw.id)
                .map_err(|e| OdinError::Database(format!("Invalid job id '{}': {e}", raw.id)))?;

            jobs.push(PersistedJob {
                id,
                name: raw.name,
                cron_expr: raw.cron_expr,
                task_goal: raw.task_goal,
                max_iterations: raw.max_iterations,
                enabled: raw.enabled,
                last_run,
                next_run,
                run_count: raw.run_count,
                created_at,
            });
        }

        Ok(jobs)
    }

    async fn delete_job(&self, id: &JobId) -> odin_core::error::OdinResult<()> {
        let conn = self.conn.lock().await;

        let affected = conn
            .execute(
                "DELETE FROM scheduler_jobs WHERE id = ?1",
                params![id.to_string()],
            )
            .map_err(|e| OdinError::Database(format!("Failed to delete scheduler job: {e}")))?;

        if affected == 0 {
            tracing::warn!(job_id = %id, "Attempted to delete non-existent scheduler job");
        }

        Ok(())
    }

    async fn update_job_state(
        &self,
        id: &JobId,
        enabled: bool,
        last_run: Option<DateTime<Utc>>,
        next_run: Option<DateTime<Utc>>,
        run_count: u64,
    ) -> odin_core::error::OdinResult<()> {
        let conn = self.conn.lock().await;

        conn.execute(
            "UPDATE scheduler_jobs SET
                enabled   = ?1,
                last_run  = ?2,
                next_run  = ?3,
                run_count = ?4
             WHERE id = ?5",
            params![
                enabled as i32,
                last_run.map(|t| t.to_rfc3339()),
                next_run.map(|t| t.to_rfc3339()),
                run_count as i64,
                id.to_string(),
            ],
        )
        .map_err(|e| OdinError::Database(format!("Failed to update scheduler job state: {e}")))?;

        Ok(())
    }

    async fn record_run_started(&self, run: &JobRun) -> odin_core::error::OdinResult<()> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT OR REPLACE INTO scheduler_runs
                (task_id, job_id, job_name, started_at, finished_at, status, error)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                run.task_id.to_string(),
                run.job_id.to_string(),
                run.job_name,
                run.started_at.to_rfc3339(),
                run.finished_at.map(|value| value.to_rfc3339()),
                run.status.as_str(),
                run.error,
            ],
        )
        .map_err(|e| OdinError::Database(format!("Failed to record scheduler run: {e}")))?;
        Ok(())
    }

    async fn record_run_finished(
        &self,
        task_id: &Uuid,
        status: JobRunStatus,
        error: Option<&str>,
    ) -> odin_core::error::OdinResult<()> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE scheduler_runs
             SET finished_at = ?1, status = ?2, error = ?3
             WHERE task_id = ?4",
            params![
                Utc::now().to_rfc3339(),
                status.as_str(),
                error,
                task_id.to_string(),
            ],
        )
        .map_err(|e| OdinError::Database(format!("Failed to finish scheduler run: {e}")))?;
        Ok(())
    }

    async fn recent_runs(&self, limit: usize) -> odin_core::error::OdinResult<Vec<JobRun>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(
                "SELECT task_id, job_id, job_name, started_at, finished_at, status, error
                 FROM scheduler_runs
                 ORDER BY started_at DESC
                 LIMIT ?1",
            )
            .map_err(|e| OdinError::Database(format!("Failed to prepare run query: {e}")))?;
        let rows = stmt
            .query_map(params![limit as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, Option<String>>(6)?,
                ))
            })
            .map_err(|e| OdinError::Database(format!("Failed to query scheduler runs: {e}")))?;

        let mut runs = Vec::new();
        for row in rows {
            let (task_id, job_id, job_name, started_at, finished_at, status, error) =
                row.map_err(|e| OdinError::Database(format!("Failed to read scheduler run: {e}")))?;
            runs.push(JobRun {
                task_id: Uuid::parse_str(&task_id)
                    .map_err(|e| OdinError::Database(format!("Invalid task id: {e}")))?,
                job_id: Uuid::parse_str(&job_id)
                    .map_err(|e| OdinError::Database(format!("Invalid job id: {e}")))?,
                job_name,
                started_at: DateTime::parse_from_rfc3339(&started_at)
                    .map_err(|e| OdinError::Database(format!("Invalid started_at: {e}")))?
                    .with_timezone(&Utc),
                finished_at: finished_at
                    .map(|value| {
                        DateTime::parse_from_rfc3339(&value)
                            .map(|date| date.with_timezone(&Utc))
                            .map_err(|e| OdinError::Database(format!("Invalid finished_at: {e}")))
                    })
                    .transpose()?,
                status: JobRunStatus::parse(&status)?,
                error,
            });
        }
        Ok(runs)
    }
}

// ── Internal helper ───────────────────────────────────────────────────

/// Raw row from SQLite before parsing DateTime fields.
struct PersistedJobRaw {
    id: String,
    name: String,
    cron_expr: String,
    task_goal: Option<String>,
    max_iterations: u32,
    enabled: bool,
    last_run: Option<String>,
    next_run: Option<String>,
    run_count: u64,
    created_at: String,
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_persisted_job(name: &str, cron_expr: &str) -> PersistedJob {
        PersistedJob {
            id: Uuid::new_v4(),
            name: name.to_string(),
            cron_expr: cron_expr.to_string(),
            task_goal: Some(format!("Run {}", name)),
            max_iterations: 50,
            enabled: true,
            last_run: None,
            next_run: Some(Utc::now() + chrono::TimeDelta::hours(1)),
            run_count: 0,
            created_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_save_and_load_jobs() {
        let store = SqliteSchedulerStore::in_memory().unwrap();

        let job = make_persisted_job("test-job", "0 * * * *");
        store.save_job(&job).await.unwrap();

        let loaded = store.load_all_jobs().await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].name, "test-job");
        assert_eq!(loaded[0].cron_expr, "0 * * * *");
        assert_eq!(loaded[0].task_goal.as_deref(), Some("Run test-job"));
        assert_eq!(loaded[0].max_iterations, 50);
        assert!(loaded[0].enabled);
    }

    #[tokio::test]
    async fn test_save_update_and_load() {
        let store = SqliteSchedulerStore::in_memory().unwrap();

        let mut job = make_persisted_job("updatable", "*/5 * * * *");
        store.save_job(&job).await.unwrap();

        // Update some fields
        job.name = "updated-name".to_string();
        job.enabled = false;
        store.save_job(&job).await.unwrap();

        let loaded = store.load_all_jobs().await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].name, "updated-name");
        assert!(!loaded[0].enabled);
    }

    #[tokio::test]
    async fn test_delete_job() {
        let store = SqliteSchedulerStore::in_memory().unwrap();

        let job = make_persisted_job("delete-me", "0 9 * * *");
        store.save_job(&job).await.unwrap();
        assert_eq!(store.load_all_jobs().await.unwrap().len(), 1);

        store.delete_job(&job.id).await.unwrap();
        assert_eq!(store.load_all_jobs().await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_update_job_state() {
        let store = SqliteSchedulerStore::in_memory().unwrap();

        let job = make_persisted_job("state-test", "* * * * *");
        store.save_job(&job).await.unwrap();

        let last_run = Some(Utc::now());
        let next_run = Some(Utc::now() + chrono::TimeDelta::hours(2));
        store
            .update_job_state(&job.id, false, last_run, next_run, 5)
            .await
            .unwrap();

        let loaded = store.load_all_jobs().await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(!loaded[0].enabled);
        assert_eq!(loaded[0].run_count, 5);
        assert!(loaded[0].last_run.is_some());
        assert!(loaded[0].next_run.is_some());
    }

    #[tokio::test]
    async fn test_persisted_job_roundtrip() {
        let original = Job::new(
            "roundtrip",
            Schedule::parse("30 9 * * 1-5").unwrap(),
            crate::job::noop_task(),
        );

        let persisted = PersistedJob::from_job(&original);
        assert_eq!(persisted.name, "roundtrip");
        assert_eq!(persisted.cron_expr, "30 9 * * 1-5");
        assert_eq!(persisted.task_goal, None);

        let restored = persisted.into_job();
        assert_eq!(restored.id, original.id);
        assert_eq!(restored.name, original.name);
        assert_eq!(restored.schedule.expression, "30 9 * * 1-5");
        assert_eq!(restored.task_goal, original.task_goal);
        assert_eq!(restored.max_iterations, original.max_iterations);
        assert_eq!(restored.enabled, original.enabled);
        assert_eq!(restored.run_count, original.run_count);
    }

    #[tokio::test]
    async fn test_file_based_store() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("test_odin_scheduler_{}.db", Uuid::new_v4()));
        let path_str = path.to_str().unwrap().to_string();

        // Create store and insert data
        {
            let store = SqliteSchedulerStore::new(&path_str).unwrap();
            let job = make_persisted_job("persistent", "0 0 * * *");
            store.save_job(&job).await.unwrap();
            assert_eq!(store.load_all_jobs().await.unwrap().len(), 1);
        }

        // Re-open and verify data persists
        {
            let store = SqliteSchedulerStore::new(&path_str).unwrap();
            let loaded = store.load_all_jobs().await.unwrap();
            assert_eq!(loaded.len(), 1);
            assert_eq!(loaded[0].name, "persistent");
        }

        // Cleanup
        let _ = std::fs::remove_file(&path_str);
    }

    #[tokio::test]
    async fn test_in_memory_is_empty() {
        let store = SqliteSchedulerStore::in_memory().unwrap();
        let loaded = store.load_all_jobs().await.unwrap();
        assert!(loaded.is_empty());
    }
}
