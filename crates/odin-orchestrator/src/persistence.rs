//! SQLite persistence for orchestration state.
//!
//! Ensures task graphs, agent lifecycles, and file lock state survive restarts.
//! Uses SQLite via sqlx for durable storage.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx_core::query::query;
use sqlx_core::query_as::query_as;
use sqlx_sqlite::{
    SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions, SqliteSynchronous,
};
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;
use uuid::Uuid;

use crate::control::{RunControlCommand, RunControlKind, RunControlStatus};
use crate::lifecycle::{AgentLifecycle, AgentPhase};
use crate::task_graph::{TaskGraph, TaskGraphStatus};

/// Error type for orchestration storage operations.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx_core::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Invalid orchestration status: {0}")]
    InvalidStatus(String),
}

/// Trait for orchestration state persistence.
#[async_trait]
pub trait OrchestrationStore: Send + Sync {
    /// Save a task graph.
    async fn save_task_graph(&self, graph: &TaskGraph) -> Result<(), StoreError>;
    /// Load a task graph by its first node ID (or root).
    async fn load_task_graph(&self, root_id: &str) -> Result<TaskGraph, StoreError>;
    /// List all stored task graphs.
    async fn list_task_graphs(&self) -> Result<Vec<TaskGraphSummary>, StoreError>;
    /// List task graphs with "running" or "paused" status (unfinished runs).
    async fn find_unfinished_graphs(&self) -> Result<Vec<TaskGraphSummary>, StoreError>;

    /// Save an agent lifecycle.
    async fn save_agent_lifecycle(
        &self,
        graph_root_id: &str,
        lifecycle: &AgentLifecycle,
    ) -> Result<(), StoreError>;
    /// Load an agent lifecycle.
    async fn load_agent_lifecycle(&self, agent_id: Uuid) -> Result<AgentLifecycle, StoreError>;
    /// List all stored lifecycles.
    async fn list_agent_lifecycles(&self) -> Result<Vec<AgentLifecycleSummary>, StoreError>;

    /// Update the status of a stored task graph.
    async fn update_graph_status(&self, root_id: &str, status: &str) -> Result<(), StoreError>;
    /// Update the phase of a stored agent lifecycle.
    async fn update_lifecycle_phase(&self, agent_id: &str, phase: &str) -> Result<(), StoreError>;
    /// Delete a task graph and all its associated agent lifecycles.
    async fn delete_task_graph(&self, root_id: &str) -> Result<(), StoreError>;

    /// Save a file lock snapshot keyed by its graph/run identity.
    async fn save_lock_snapshot(&self, graph_id: &str, snapshot: &str) -> Result<(), StoreError>;
    /// Load the file lock snapshot for one graph/run.
    async fn load_lock_snapshot(&self, graph_id: &str) -> Result<Option<String>, StoreError>;
    /// Load the newest snapshot when an operator has not selected a run.
    async fn load_latest_lock_snapshot(&self) -> Result<Option<String>, StoreError>;

    /// Enqueue a live control command for a graph UUID.
    async fn enqueue_control(&self, command: &RunControlCommand) -> Result<(), StoreError>;
    /// Atomically claim all pending control commands for a graph.
    async fn claim_pending_controls(
        &self,
        graph_id: &str,
    ) -> Result<Vec<RunControlCommand>, StoreError>;
    /// Mark a claimed control command as applied by the owner process.
    async fn mark_control_applied(&self, command_id: Uuid) -> Result<(), StoreError>;
    /// List recent control commands for a graph (newest first).
    async fn list_controls(
        &self,
        graph_id: &str,
        limit: usize,
    ) -> Result<Vec<RunControlCommand>, StoreError>;

    /// Initialize the database (create tables if needed).
    async fn initialize(&self) -> Result<(), StoreError>;
}

/// Summary of a stored task graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskGraphSummary {
    pub run_id: String,
    pub root_goal: String,
    pub status: String,
    pub node_count: i64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Summary of a stored agent lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLifecycleSummary {
    pub agent_id: String,
    pub graph_root_id: Option<String>,
    pub phase: String,
    pub created_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
}

/// SQLite-backed orchestration store.
pub struct SqliteOrchestrationStore {
    pool: SqlitePool,
}

impl SqliteOrchestrationStore {
    /// Create a new SQLite store at the given path.
    pub async fn new(path: impl Into<PathBuf>) -> Result<Self, StoreError> {
        let path = path.into();
        let options = SqliteConnectOptions::new()
            .filename(&path)
            .create_if_missing(true)
            .busy_timeout(Duration::from_secs(5))
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .foreign_keys(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
                .map_err(|error| StoreError::Database(sqlx_core::Error::Io(error)))?;
        }

        Ok(Self { pool })
    }

    /// Create an in-memory store (for testing).
    pub async fn new_in_memory() -> Result<Self, StoreError> {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(
                SqliteConnectOptions::from_str("sqlite::memory:")?
                    .journal_mode(SqliteJournalMode::Memory)
                    .synchronous(SqliteSynchronous::Normal)
                    .foreign_keys(true),
            )
            .await?;

        Ok(Self { pool })
    }
}

#[async_trait]
impl OrchestrationStore for SqliteOrchestrationStore {
    async fn initialize(&self) -> Result<(), StoreError> {
        query(
            r#"
            CREATE TABLE IF NOT EXISTS orchestration_schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        query(
            r#"
            CREATE TABLE IF NOT EXISTS task_graphs (
                root_id TEXT PRIMARY KEY,
                root_goal TEXT NOT NULL,
                graph_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                node_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        query(
            r#"
            CREATE TABLE IF NOT EXISTS agent_lifecycles (
                agent_id TEXT PRIMARY KEY,
                graph_root_id TEXT,
                phase TEXT NOT NULL DEFAULT 'queued',
                lifecycle_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                finished_at TEXT
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        let lifecycle_columns =
            query_as::<_, (String,)>("SELECT name FROM pragma_table_info('agent_lifecycles')")
                .fetch_all(&self.pool)
                .await?;
        if !lifecycle_columns
            .iter()
            .any(|(name,)| name == "graph_root_id")
        {
            query("ALTER TABLE agent_lifecycles ADD COLUMN graph_root_id TEXT")
                .execute(&self.pool)
                .await?;
        }

        query(
            "CREATE INDEX IF NOT EXISTS idx_agent_lifecycles_graph_root
             ON agent_lifecycles(graph_root_id)",
        )
        .execute(&self.pool)
        .await?;

        query(
            r#"
            CREATE TABLE IF NOT EXISTS lock_snapshots (
                graph_id TEXT PRIMARY KEY,
                snapshot_json TEXT NOT NULL,
                saved_at TEXT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Migrate the pre-audit singleton snapshot table without losing the
        // last snapshot. New writes are keyed by graph/run identity so one
        // orchestration cannot overwrite another one's lock state.
        let lock_columns =
            query_as::<_, (String,)>("SELECT name FROM pragma_table_info('lock_snapshots')")
                .fetch_all(&self.pool)
                .await?;
        if !lock_columns.iter().any(|(name,)| name == "graph_id") {
            query("ALTER TABLE lock_snapshots RENAME TO lock_snapshots_legacy")
                .execute(&self.pool)
                .await?;
            query(
                r#"
                CREATE TABLE lock_snapshots (
                    graph_id TEXT PRIMARY KEY,
                    snapshot_json TEXT NOT NULL,
                    saved_at TEXT NOT NULL
                )
                "#,
            )
            .execute(&self.pool)
            .await?;
            query(
                "INSERT INTO lock_snapshots (graph_id, snapshot_json, saved_at)
                 SELECT 'legacy', snapshot_json, saved_at FROM lock_snapshots_legacy",
            )
            .execute(&self.pool)
            .await?;
            query("DROP TABLE lock_snapshots_legacy")
                .execute(&self.pool)
                .await?;
        }

        query(
            r#"
            CREATE TABLE IF NOT EXISTS run_controls (
                id TEXT PRIMARY KEY,
                graph_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                reason TEXT,
                source TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                claimed_at TEXT,
                applied_at TEXT
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_run_controls_graph_status
            ON run_controls(graph_id, status, created_at)
            "#,
        )
        .execute(&self.pool)
        .await?;

        query(
            "INSERT OR IGNORE INTO orchestration_schema_migrations (version, applied_at)
             VALUES (1, datetime('now'))",
        )
        .execute(&self.pool)
        .await?;
        query(
            "INSERT OR IGNORE INTO orchestration_schema_migrations (version, applied_at)
             VALUES (2, datetime('now'))",
        )
        .execute(&self.pool)
        .await?;
        query(
            "INSERT OR IGNORE INTO orchestration_schema_migrations (version, applied_at)
             VALUES (3, datetime('now'))",
        )
        .execute(&self.pool)
        .await?;

        // Keep terminal orchestration history bounded while preserving active
        // runs and recent diagnostics for operators.
        query(
            "DELETE FROM task_graphs
             WHERE status IN ('complete', 'failed', 'cancelled')
               AND updated_at < datetime('now', '-30 days')",
        )
        .execute(&self.pool)
        .await?;
        query(
            "DELETE FROM agent_lifecycles
             WHERE finished_at IS NOT NULL
               AND finished_at < datetime('now', '-30 days')",
        )
        .execute(&self.pool)
        .await?;
        query(
            "DELETE FROM run_controls
             WHERE status = 'applied'
               AND applied_at IS NOT NULL
               AND applied_at < datetime('now', '-30 days')",
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn save_task_graph(&self, graph: &TaskGraph) -> Result<(), StoreError> {
        let root_id = graph.id.to_string();
        let graph_json = serde_json::to_string(&graph)?;
        let now = Utc::now().to_rfc3339();
        let status = graph_status_label(graph.status);
        let node_count = graph.nodes.len() as i64;

        query(
            r#"
            INSERT INTO task_graphs (root_id, root_goal, graph_json, status, node_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(root_id) DO UPDATE SET
                graph_json = excluded.graph_json,
                status = excluded.status,
                node_count = excluded.node_count,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(&root_id)
        .bind(&graph.root_goal)
        .bind(&graph_json)
        .bind(status)
        .bind(node_count)
        .bind(&now)
        .bind(&now)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn load_task_graph(&self, root_id: &str) -> Result<TaskGraph, StoreError> {
        let row = query_as::<_, (String,)>(
            "SELECT graph_json FROM task_graphs WHERE root_id = ? OR root_goal = ? ORDER BY updated_at DESC LIMIT 1",
        )
                .bind(root_id)
                .bind(root_id)
                .fetch_optional(&self.pool)
                .await?
                .ok_or_else(|| {
                    StoreError::NotFound(format!("Task graph '{}' not found", root_id))
                })?;

        let graph: TaskGraph = serde_json::from_str(&row.0)?;
        Ok(graph)
    }

    async fn list_task_graphs(&self) -> Result<Vec<TaskGraphSummary>, StoreError> {
        let rows = query_as::<_, (String, String, String, i64, String, String)>(
            "SELECT root_id, root_goal, status, node_count, created_at, updated_at FROM task_graphs ORDER BY updated_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(
                |(run_id, goal, status, count, created, updated)| TaskGraphSummary {
                    run_id,
                    root_goal: goal,
                    status,
                    node_count: count,
                    created_at: DateTime::parse_from_rfc3339(&created)
                        .unwrap_or_default()
                        .with_timezone(&Utc),
                    updated_at: DateTime::parse_from_rfc3339(&updated)
                        .unwrap_or_default()
                        .with_timezone(&Utc),
                },
            )
            .collect())
    }

    async fn find_unfinished_graphs(&self) -> Result<Vec<TaskGraphSummary>, StoreError> {
        let rows = query_as::<_, (String, String, String, i64, String, String)>(
            "SELECT root_id, root_goal, status, node_count, created_at, updated_at FROM task_graphs WHERE status IN ('running', 'paused') ORDER BY updated_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(
                |(run_id, goal, status, count, created, updated)| TaskGraphSummary {
                    run_id,
                    root_goal: goal,
                    status,
                    node_count: count,
                    created_at: DateTime::parse_from_rfc3339(&created)
                        .unwrap_or_default()
                        .with_timezone(&Utc),
                    updated_at: DateTime::parse_from_rfc3339(&updated)
                        .unwrap_or_default()
                        .with_timezone(&Utc),
                },
            )
            .collect())
    }

    async fn save_agent_lifecycle(
        &self,
        graph_root_id: &str,
        lifecycle: &AgentLifecycle,
    ) -> Result<(), StoreError> {
        let agent_id = lifecycle.agent_id.to_string();
        let lifecycle_json = serde_json::to_string(&lifecycle)?;
        let phase = lifecycle.phase.label().to_string();
        let now = Utc::now().to_rfc3339();
        let finished = lifecycle.finished_at.map(|t| t.to_rfc3339());

        query(
            r#"
            INSERT INTO agent_lifecycles
                (agent_id, graph_root_id, phase, lifecycle_json, created_at, finished_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                graph_root_id = excluded.graph_root_id,
                phase = excluded.phase,
                lifecycle_json = excluded.lifecycle_json,
                finished_at = excluded.finished_at
            "#,
        )
        .bind(&agent_id)
        .bind(graph_root_id)
        .bind(&phase)
        .bind(&lifecycle_json)
        .bind(&now)
        .bind(&finished)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn load_agent_lifecycle(&self, agent_id: Uuid) -> Result<AgentLifecycle, StoreError> {
        let row = query_as::<_, (String,)>(
            "SELECT lifecycle_json FROM agent_lifecycles WHERE agent_id = ?",
        )
        .bind(agent_id.to_string())
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| StoreError::NotFound(format!("Agent lifecycle '{}' not found", agent_id)))?;

        let lifecycle: AgentLifecycle = serde_json::from_str(&row.0)?;
        Ok(lifecycle)
    }

    async fn list_agent_lifecycles(&self) -> Result<Vec<AgentLifecycleSummary>, StoreError> {
        let rows = query_as::<_, (String, Option<String>, String, String, Option<String>)>(
            "SELECT agent_id, graph_root_id, phase, created_at, finished_at FROM agent_lifecycles ORDER BY created_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(
                |(id, graph_root_id, phase, created, finished)| AgentLifecycleSummary {
                    agent_id: id,
                    graph_root_id,
                    phase,
                    created_at: DateTime::parse_from_rfc3339(&created)
                        .unwrap_or_default()
                        .with_timezone(&Utc),
                    finished_at: finished.map(|f| {
                        DateTime::parse_from_rfc3339(&f)
                            .unwrap_or_default()
                            .with_timezone(&Utc)
                    }),
                },
            )
            .collect())
    }

    async fn update_graph_status(&self, root_id: &str, status: &str) -> Result<(), StoreError> {
        let graph_status = match status.to_ascii_lowercase().as_str() {
            "building" => TaskGraphStatus::Building,
            "running" => TaskGraphStatus::Running,
            "paused" => TaskGraphStatus::Paused,
            "complete" | "completed" => TaskGraphStatus::Complete,
            "failed" => TaskGraphStatus::Failed,
            "cancelled" | "canceled" => TaskGraphStatus::Cancelled,
            _ => return Err(StoreError::InvalidStatus(status.to_string())),
        };
        let mut tx = self.pool.begin().await?;
        let row = query_as::<_, (String,)>(
            "SELECT graph_json FROM task_graphs WHERE root_id = ? OR root_goal = ? ORDER BY updated_at DESC LIMIT 1",
        )
        .bind(root_id)
        .bind(root_id)
        .fetch_optional(&mut *tx)
        .await?
        .ok_or_else(|| StoreError::NotFound(format!("Task graph '{}' not found", root_id)))?;
        let mut graph: TaskGraph = serde_json::from_str(&row.0)?;
        graph.status = graph_status;
        let graph_json = serde_json::to_string(&graph)?;
        let now = Utc::now().to_rfc3339();
        let rows = query(
            "UPDATE task_graphs SET status = ?, graph_json = ?, updated_at = ? WHERE root_id = ? OR root_goal = ?",
        )
                .bind(graph_status_label(graph_status))
                .bind(graph_json)
                .bind(&now)
                .bind(root_id)
                .bind(root_id)
                .execute(&mut *tx)
                .await?;

        if rows.rows_affected() == 0 {
            return Err(StoreError::NotFound(format!(
                "Task graph '{}' not found",
                root_id
            )));
        }
        tx.commit().await?;
        Ok(())
    }

    async fn update_lifecycle_phase(&self, agent_id: &str, phase: &str) -> Result<(), StoreError> {
        let target = parse_agent_phase(phase)?;
        let mut tx = self.pool.begin().await?;
        let row = query_as::<_, (String,)>(
            "SELECT lifecycle_json FROM agent_lifecycles WHERE agent_id = ?",
        )
        .bind(agent_id)
        .fetch_optional(&mut *tx)
        .await?
        .ok_or_else(|| StoreError::NotFound(format!("Agent lifecycle '{}' not found", agent_id)))?;

        let mut lifecycle: AgentLifecycle = serde_json::from_str(&row.0)?;
        lifecycle.transition(target, Some("Updated by persistence API".into()));
        let lifecycle_json = serde_json::to_string(&lifecycle)?;
        let finished = lifecycle.finished_at.map(|value| value.to_rfc3339());
        let rows = query(
            "UPDATE agent_lifecycles
             SET phase = ?, lifecycle_json = ?, finished_at = ?
             WHERE agent_id = ?",
        )
        .bind(target.label())
        .bind(lifecycle_json)
        .bind(finished)
        .bind(agent_id)
        .execute(&mut *tx)
        .await?;

        if rows.rows_affected() == 0 {
            return Err(StoreError::NotFound(format!(
                "Agent lifecycle '{}' not found",
                agent_id
            )));
        }
        tx.commit().await?;
        Ok(())
    }

    async fn delete_task_graph(&self, root_id: &str) -> Result<(), StoreError> {
        let mut tx = self.pool.begin().await?;
        // Delete associated lifecycles first, in the same transaction as the
        // graph so a partial cleanup cannot leave orphaned state.
        query("DELETE FROM agent_lifecycles WHERE graph_root_id = ?")
            .bind(root_id)
            .execute(&mut *tx)
            .await?;

        let rows = query("DELETE FROM task_graphs WHERE root_id = ?")
            .bind(root_id)
            .execute(&mut *tx)
            .await?;

        if rows.rows_affected() == 0 {
            return Err(StoreError::NotFound(format!(
                "Task graph '{}' not found",
                root_id
            )));
        }
        tx.commit().await?;
        Ok(())
    }

    async fn save_lock_snapshot(
        &self,
        graph_id: &str,
        snapshot_json: &str,
    ) -> Result<(), StoreError> {
        if graph_id.trim().is_empty() {
            return Err(StoreError::NotFound(
                "lock snapshot graph ID is empty".into(),
            ));
        }
        let now = Utc::now().to_rfc3339();
        query(
            "INSERT INTO lock_snapshots (graph_id, snapshot_json, saved_at) VALUES (?, ?, ?) ON CONFLICT(graph_id) DO UPDATE SET snapshot_json = excluded.snapshot_json, saved_at = excluded.saved_at",
        )
        .bind(graph_id)
        .bind(snapshot_json)
        .bind(&now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn load_lock_snapshot(&self, graph_id: &str) -> Result<Option<String>, StoreError> {
        let row =
            query_as::<_, (String,)>("SELECT snapshot_json FROM lock_snapshots WHERE graph_id = ?")
                .bind(graph_id)
                .fetch_optional(&self.pool)
                .await?;
        Ok(row.map(|r| r.0))
    }

    async fn load_latest_lock_snapshot(&self) -> Result<Option<String>, StoreError> {
        let row = query_as::<_, (String,)>(
            "SELECT snapshot_json FROM lock_snapshots ORDER BY saved_at DESC LIMIT 1",
        )
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|r| r.0))
    }

    async fn enqueue_control(&self, command: &RunControlCommand) -> Result<(), StoreError> {
        query(
            r#"
            INSERT INTO run_controls
                (id, graph_id, kind, reason, source, status, created_at, claimed_at, applied_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(command.id.to_string())
        .bind(&command.graph_id)
        .bind(command.kind.as_str())
        .bind(&command.reason)
        .bind(&command.source)
        .bind(command.status.as_str())
        .bind(command.created_at.to_rfc3339())
        .bind(command.claimed_at.map(|value| value.to_rfc3339()))
        .bind(command.applied_at.map(|value| value.to_rfc3339()))
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn claim_pending_controls(
        &self,
        graph_id: &str,
    ) -> Result<Vec<RunControlCommand>, StoreError> {
        let now = Utc::now().to_rfc3339();
        let rows = query_as::<
            _,
            (
                String,
                String,
                String,
                Option<String>,
                String,
                String,
                String,
                Option<String>,
                Option<String>,
            ),
        >(
            r#"
            UPDATE run_controls
            SET status = 'claimed', claimed_at = ?
            WHERE graph_id = ? AND status = 'pending'
            RETURNING
                id, graph_id, kind, reason, source, status,
                created_at, claimed_at, applied_at
            "#,
        )
        .bind(&now)
        .bind(graph_id)
        .fetch_all(&self.pool)
        .await?;

        let mut commands = Vec::with_capacity(rows.len());
        for row in rows {
            let id = Uuid::parse_str(&row.0).map_err(|error| {
                StoreError::InvalidStatus(format!("invalid control id: {error}"))
            })?;
            commands.push(parse_control_row(
                id, row.1, row.2, row.3, row.4, &row.5, row.6, row.7, row.8,
            )?);
        }
        commands.sort_by_key(|command| command.created_at);
        Ok(commands)
    }

    async fn mark_control_applied(&self, command_id: Uuid) -> Result<(), StoreError> {
        let now = Utc::now().to_rfc3339();
        let result = query(
            "UPDATE run_controls
             SET status = 'applied', applied_at = ?
             WHERE id = ? AND status = 'claimed'",
        )
        .bind(&now)
        .bind(command_id.to_string())
        .execute(&self.pool)
        .await?;
        if result.rows_affected() == 0 {
            return Err(StoreError::NotFound(format!(
                "control command '{command_id}' not found"
            )));
        }
        Ok(())
    }

    async fn list_controls(
        &self,
        graph_id: &str,
        limit: usize,
    ) -> Result<Vec<RunControlCommand>, StoreError> {
        let rows = query_as::<
            _,
            (
                String,
                String,
                String,
                Option<String>,
                String,
                String,
                String,
                Option<String>,
                Option<String>,
            ),
        >(
            r#"
            SELECT id, graph_id, kind, reason, source, status, created_at, claimed_at, applied_at
            FROM run_controls
            WHERE graph_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            "#,
        )
        .bind(graph_id)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter()
            .map(|row| {
                let id = Uuid::parse_str(&row.0).map_err(|error| {
                    StoreError::InvalidStatus(format!("invalid control id: {error}"))
                })?;
                parse_control_row(id, row.1, row.2, row.3, row.4, &row.5, row.6, row.7, row.8)
            })
            .collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn parse_control_row(
    id: Uuid,
    graph_id: String,
    kind: String,
    reason: Option<String>,
    source: String,
    status: &str,
    created_at: String,
    claimed_at: Option<String>,
    applied_at: Option<String>,
) -> Result<RunControlCommand, StoreError> {
    let kind = RunControlKind::parse(&kind)
        .ok_or_else(|| StoreError::InvalidStatus(format!("invalid control kind '{kind}'")))?;
    let status = RunControlStatus::parse(status)
        .ok_or_else(|| StoreError::InvalidStatus(format!("invalid control status '{status}'")))?;
    Ok(RunControlCommand {
        id,
        graph_id,
        kind,
        reason,
        source,
        status,
        created_at: DateTime::parse_from_rfc3339(&created_at)
            .map_err(|error| StoreError::InvalidStatus(format!("invalid created_at: {error}")))?
            .with_timezone(&Utc),
        claimed_at: claimed_at
            .map(|value| {
                DateTime::parse_from_rfc3339(&value)
                    .map(|date| date.with_timezone(&Utc))
                    .map_err(|error| {
                        StoreError::InvalidStatus(format!("invalid claimed_at: {error}"))
                    })
            })
            .transpose()?,
        applied_at: applied_at
            .map(|value| {
                DateTime::parse_from_rfc3339(&value)
                    .map(|date| date.with_timezone(&Utc))
                    .map_err(|error| {
                        StoreError::InvalidStatus(format!("invalid applied_at: {error}"))
                    })
            })
            .transpose()?,
    })
}

fn graph_status_label(status: TaskGraphStatus) -> &'static str {
    match status {
        TaskGraphStatus::Building => "building",
        TaskGraphStatus::Running => "running",
        TaskGraphStatus::Paused => "paused",
        TaskGraphStatus::Complete => "complete",
        TaskGraphStatus::Failed => "failed",
        TaskGraphStatus::Cancelled => "cancelled",
    }
}

fn parse_agent_phase(phase: &str) -> Result<AgentPhase, StoreError> {
    match phase.to_ascii_lowercase().as_str() {
        "queued" => Ok(AgentPhase::Queued),
        "running" => Ok(AgentPhase::Running),
        "blocked" => Ok(AgentPhase::Blocked),
        "waiting_for_lock" => Ok(AgentPhase::WaitingForLock),
        "reviewing" => Ok(AgentPhase::Reviewing),
        "done" => Ok(AgentPhase::Done),
        "failed" => Ok(AgentPhase::Failed),
        "cancelled" | "canceled" => Ok(AgentPhase::Cancelled),
        _ => Err(StoreError::InvalidStatus(phase.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::AgentPhase;
    use crate::task_graph::{TaskGraphStatus, TaskNode};

    async fn setup_store() -> SqliteOrchestrationStore {
        let store = SqliteOrchestrationStore::new_in_memory().await.unwrap();
        store.initialize().await.unwrap();
        store
    }

    #[tokio::test]
    async fn test_save_and_load_task_graph() {
        let store = setup_store().await;
        let mut graph = TaskGraph::new("test-root");
        let node = TaskNode {
            id: Uuid::new_v4(),
            label: "n1".into(),
            goal: "do stuff".into(),
            read_files: vec![],
            write_files: vec![],
            required_capabilities: vec![],
            priority: 0,
            status: crate::task_graph::TaskNodeStatus::Pending,
            result: None,
            agent_id: None,
        };
        graph.add_node(node);
        graph.status = TaskGraphStatus::Running;

        store.save_task_graph(&graph).await.unwrap();

        let loaded = store.load_task_graph(&graph.id.to_string()).await.unwrap();
        assert_eq!(loaded.root_goal, "test-root");
        assert_eq!(loaded.nodes.len(), 1);
        // Goal lookup remains available for records created before stable IDs.
        assert!(store.load_task_graph("test-root").await.is_ok());
    }

    #[tokio::test]
    async fn test_save_and_load_agent_lifecycle() {
        let store = setup_store().await;
        let graph = TaskGraph::new("lifecycle-graph");
        store.save_task_graph(&graph).await.unwrap();
        let agent_id = Uuid::new_v4();
        let mut lifecycle = AgentLifecycle::new(agent_id);
        lifecycle.start();
        lifecycle.complete();

        store
            .save_agent_lifecycle(&graph.id.to_string(), &lifecycle)
            .await
            .unwrap();

        let loaded = store.load_agent_lifecycle(agent_id).await.unwrap();
        assert_eq!(loaded.phase, AgentPhase::Done);
        assert_eq!(loaded.agent_id, agent_id);
    }

    #[tokio::test]
    async fn test_list_task_graphs() {
        let store = setup_store().await;
        let graph = TaskGraph::new("list-test");
        store.save_task_graph(&graph).await.unwrap();

        let graphs = store.list_task_graphs().await.unwrap();
        assert!(!graphs.is_empty());
        assert!(graphs.iter().any(|g| g.root_goal == "list-test"));
    }

    #[tokio::test]
    async fn test_list_agent_lifecycles() {
        let store = setup_store().await;
        let graph = TaskGraph::new("lifecycle-list-graph");
        store.save_task_graph(&graph).await.unwrap();
        let lifecycle = AgentLifecycle::new(Uuid::new_v4());
        store
            .save_agent_lifecycle(&graph.id.to_string(), &lifecycle)
            .await
            .unwrap();

        let lifecycles = store.list_agent_lifecycles().await.unwrap();
        assert!(!lifecycles.is_empty());
    }

    #[tokio::test]
    async fn test_lifecycle_graph_association_update_and_delete() {
        let store = setup_store().await;
        let graph_one = TaskGraph::new("graph-one");
        let graph_two = TaskGraph::new("graph-two");
        store.save_task_graph(&graph_one).await.unwrap();
        store.save_task_graph(&graph_two).await.unwrap();

        let first_id = Uuid::new_v4();
        let second_id = Uuid::new_v4();
        store
            .save_agent_lifecycle(&graph_one.id.to_string(), &AgentLifecycle::new(first_id))
            .await
            .unwrap();
        store
            .save_agent_lifecycle(&graph_two.id.to_string(), &AgentLifecycle::new(second_id))
            .await
            .unwrap();

        store
            .update_lifecycle_phase(&first_id.to_string(), "running")
            .await
            .unwrap();
        assert_eq!(
            store.load_agent_lifecycle(first_id).await.unwrap().phase,
            AgentPhase::Running
        );

        store
            .delete_task_graph(&graph_one.id.to_string())
            .await
            .unwrap();
        assert!(store.load_agent_lifecycle(first_id).await.is_err());
        assert!(store.load_agent_lifecycle(second_id).await.is_ok());
    }

    #[tokio::test]
    async fn test_not_found() {
        let store = setup_store().await;
        let result = store.load_task_graph("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_update_existing() {
        let store = setup_store().await;
        let mut graph = TaskGraph::new("update-test");
        graph.status = TaskGraphStatus::Running;
        store.save_task_graph(&graph).await.unwrap();

        // Update
        graph.status = TaskGraphStatus::Complete;
        store.save_task_graph(&graph).await.unwrap();

        let loaded = store.load_task_graph("update-test").await.unwrap();
        assert_eq!(loaded.status, TaskGraphStatus::Complete);
    }

    #[tokio::test]
    async fn test_status_update_keeps_summary_and_graph_in_sync() {
        let store = setup_store().await;
        let mut graph = TaskGraph::new("status-test");
        graph.status = TaskGraphStatus::Running;
        let run_id = graph.id.to_string();
        store.save_task_graph(&graph).await.unwrap();

        store.update_graph_status(&run_id, "paused").await.unwrap();

        let loaded = store.load_task_graph(&run_id).await.unwrap();
        assert_eq!(loaded.status, TaskGraphStatus::Paused);
        let summaries = store.list_task_graphs().await.unwrap();
        assert_eq!(summaries[0].run_id, run_id);
        assert_eq!(summaries[0].status, "paused");
        assert_eq!(store.find_unfinished_graphs().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn lock_snapshots_are_isolated_by_graph_and_have_latest_fallback() {
        let store = setup_store().await;
        store
            .save_lock_snapshot("run-a", "snapshot-a")
            .await
            .unwrap();
        store
            .save_lock_snapshot("run-b", "snapshot-b")
            .await
            .unwrap();

        assert_eq!(
            store.load_lock_snapshot("run-a").await.unwrap().as_deref(),
            Some("snapshot-a")
        );
        assert_eq!(
            store.load_lock_snapshot("run-b").await.unwrap().as_deref(),
            Some("snapshot-b")
        );
        assert_eq!(store.load_lock_snapshot("missing-run").await.unwrap(), None);
        assert_eq!(
            store.load_latest_lock_snapshot().await.unwrap().as_deref(),
            Some("snapshot-b")
        );
    }

    #[tokio::test]
    async fn control_commands_are_claimed_once_and_applied() {
        use crate::control::{RunControlCommand, RunControlKind, RunControlStatus};

        let store = setup_store().await;
        let graph_id = Uuid::new_v4().to_string();
        let command = RunControlCommand::new(
            &graph_id,
            RunControlKind::Cancel,
            "cli",
            Some("stop now".into()),
        );
        store.enqueue_control(&command).await.unwrap();

        let claimed = store.claim_pending_controls(&graph_id).await.unwrap();
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].kind, RunControlKind::Cancel);
        assert_eq!(claimed[0].status, RunControlStatus::Claimed);
        assert!(
            store
                .claim_pending_controls(&graph_id)
                .await
                .unwrap()
                .is_empty()
        );

        store.mark_control_applied(claimed[0].id).await.unwrap();
        let listed = store.list_controls(&graph_id, 10).await.unwrap();
        assert_eq!(listed[0].status, RunControlStatus::Applied);
    }

    #[tokio::test]
    async fn initialization_records_migrations_and_retires_old_terminal_state() {
        let store = setup_store().await;
        let old_graph = TaskGraph::new("old-terminal");
        let old_root = old_graph.id.to_string();
        query(
            "INSERT INTO task_graphs
             (root_id, root_goal, graph_json, status, node_count, created_at, updated_at)
             VALUES (?, ?, ?, 'complete', 0, ?, ?)",
        )
        .bind(&old_root)
        .bind(&old_graph.root_goal)
        .bind(serde_json::to_string(&old_graph).unwrap())
        .bind("2000-01-01T00:00:00+00:00")
        .bind("2000-01-01T00:00:00+00:00")
        .execute(&store.pool)
        .await
        .unwrap();
        let old_agent = Uuid::new_v4();
        query(
            "INSERT INTO agent_lifecycles
             (agent_id, graph_root_id, phase, lifecycle_json, created_at, finished_at)
             VALUES (?, ?, 'done', ?, ?, ?)",
        )
        .bind(old_agent.to_string())
        .bind(&old_root)
        .bind(serde_json::to_string(&AgentLifecycle::new(old_agent)).unwrap())
        .bind("2000-01-01T00:00:00+00:00")
        .bind("2000-01-01T00:00:00+00:00")
        .execute(&store.pool)
        .await
        .unwrap();

        store.initialize().await.unwrap();

        let migration_count: i64 = query_as::<_, (i64,)>(
            "SELECT COUNT(*) FROM orchestration_schema_migrations WHERE version IN (1, 2, 3)",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap()
        .0;
        let graph_count: i64 =
            query_as::<_, (i64,)>("SELECT COUNT(*) FROM task_graphs WHERE root_id = ?")
                .bind(&old_root)
                .fetch_one(&store.pool)
                .await
                .unwrap()
                .0;
        let lifecycle_count: i64 =
            query_as::<_, (i64,)>("SELECT COUNT(*) FROM agent_lifecycles WHERE agent_id = ?")
                .bind(old_agent.to_string())
                .fetch_one(&store.pool)
                .await
                .unwrap()
                .0;
        assert_eq!(migration_count, 3);
        assert_eq!(graph_count, 0);
        assert_eq!(lifecycle_count, 0);
    }

    #[tokio::test]
    async fn file_store_enables_wal() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("orchestration.db");
        let store = SqliteOrchestrationStore::new(&path).await.unwrap();
        store.initialize().await.unwrap();
        let (journal_mode,): (String,) = query_as("PRAGMA journal_mode")
            .fetch_one(&store.pool)
            .await
            .unwrap();
        assert_eq!(journal_mode, "wal");
    }
}
