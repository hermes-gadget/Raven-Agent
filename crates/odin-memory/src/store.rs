//! SQLite-backed implementation of the [`MemoryStore`] trait.
//!
//! `SqliteMemoryStore` stores memory entries in a local SQLite database
//! with full CRUD operations, text search via `LIKE`, and category filtering.

use crate::models::MemoryRow;
use async_trait::async_trait;
use odin_core::{MemoryCategory, MemoryEntry, OdinError, error::OdinResult, traits::MemoryStore};
use rusqlite::{Connection, params};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::instrument;

/// Persistent memory store backed by SQLite.
///
/// Synchronous SQLite work is moved to Tokio's blocking pool so slow queries
/// never hold an async mutex or block an executor worker.
#[derive(Debug, Clone)]
pub struct SqliteMemoryStore {
    conn: Arc<Mutex<Connection>>,
    retention_limit: usize,
}

impl SqliteMemoryStore {
    /// Open (or create) a SQLite database at the given file path.
    ///
    /// Runs table creation synchronously before returning.
    pub fn new(path: &str) -> OdinResult<Self> {
        let conn = Connection::open(path)
            .map_err(|e| OdinError::Database(format!("Failed to open database at {path}: {e}")))?;
        conn.busy_timeout(Duration::from_secs(5)).map_err(|e| {
            OdinError::Database(format!("Failed to configure database timeout: {e}"))
        })?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
            retention_limit: 1_000,
        };
        store.init_tables()?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
                .map_err(OdinError::Io)?;
        }
        tracing::info!(path = %path, "Opened SQLite memory store");
        Ok(store)
    }

    /// Create an in-memory SQLite database (useful for testing).
    pub fn in_memory() -> OdinResult<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| OdinError::Database(format!("Failed to open in-memory database: {e}")))?;
        conn.busy_timeout(Duration::from_secs(5)).map_err(|e| {
            OdinError::Database(format!("Failed to configure database timeout: {e}"))
        })?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
            retention_limit: 1_000,
        };
        store.init_tables()?;
        Ok(store)
    }

    /// Bound retained rows so long-lived memory databases cannot grow without
    /// limit. The newest rows by update time are retained.
    pub fn with_retention_limit(mut self, limit: usize) -> Self {
        self.retention_limit = limit.max(1);
        self
    }

    /// Initialise the database schema.
    fn init_tables(&self) -> OdinResult<()> {
        // Since `init_tables` is called from the constructor before the
        // Mutex can be contended, a blocking access is safe here.
        let conn = self
            .conn
            .lock()
            .map_err(|error| OdinError::Database(format!("Memory store lock poisoned: {error}")))?;

        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS memory_schema_migrations (
                version    INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_entries (
                id         TEXT PRIMARY KEY,
                content    TEXT    NOT NULL,
                category   TEXT    NOT NULL,
                created_at TEXT    NOT NULL,
                updated_at TEXT    NOT NULL,
                tags       TEXT    NOT NULL DEFAULT '[]',
                importance REAL    NOT NULL DEFAULT 0.0
            );

            CREATE INDEX IF NOT EXISTS idx_memory_category
                ON memory_entries (category);

            CREATE INDEX IF NOT EXISTS idx_memory_created
                ON memory_entries (created_at DESC);",
        )
        .map_err(|e| OdinError::Database(format!("Failed to initialise schema: {e}")))?;

        conn.execute(
            "INSERT OR IGNORE INTO memory_schema_migrations (version, applied_at)
             VALUES (?1, datetime('now'))",
            [1_i64],
        )
        .map_err(|e| {
            OdinError::Database(format!("Failed to record memory schema migration: {e}"))
        })?;

        Ok(())
    }

    async fn run_db<T, F>(&self, operation: F) -> OdinResult<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut Connection) -> OdinResult<T> + Send + 'static,
    {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().map_err(|error| {
                OdinError::Database(format!("Memory store lock poisoned: {error}"))
            })?;
            operation(&mut conn)
        })
        .await
        .map_err(|error| OdinError::Database(format!("SQLite worker failed: {error}")))?
    }
}

#[async_trait]
impl MemoryStore for SqliteMemoryStore {
    #[instrument(skip(self, entry), fields(entry_id = %entry.id))]
    async fn store(&self, entry: MemoryEntry) -> OdinResult<()> {
        let row = MemoryRow::from_entry(&entry);
        let retention_limit = self.retention_limit;
        self.run_db(move |conn| {
            let tx = conn.unchecked_transaction().map_err(|e| {
                OdinError::Database(format!("Failed to begin memory transaction: {e}"))
            })?;
            tx.execute(
                "INSERT INTO memory_entries (id, content, category, created_at, updated_at, tags, importance)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                 ON CONFLICT(id) DO UPDATE SET
                     content    = excluded.content,
                     category   = excluded.category,
                     updated_at = excluded.updated_at,
                     tags       = excluded.tags,
                     importance = excluded.importance",
                params![
                    row.id,
                    row.content,
                    row.category,
                    row.created_at,
                    row.updated_at,
                    row.tags,
                    row.importance,
                ],
            )
            .map_err(|e| OdinError::Database(format!("Failed to store memory entry: {e}")))?;

            tx.execute(
                "DELETE FROM memory_entries
                 WHERE id IN (
                     SELECT id FROM memory_entries
                     ORDER BY updated_at DESC, id DESC
                     LIMIT -1 OFFSET ?1
                 )",
                params![retention_limit as i64],
            )
            .map_err(|e| OdinError::Database(format!("Failed to enforce memory retention: {e}")))?;

            tx.commit().map_err(|e| {
                OdinError::Database(format!("Failed to commit memory transaction: {e}"))
            })?;

            Ok(())
        })
        .await
    }

    #[instrument(skip(self))]
    async fn get(&self, id: &str) -> OdinResult<Option<MemoryEntry>> {
        let id = id.to_string();
        self.run_db(move |conn| {
            let mut stmt = conn
                .prepare("SELECT id, content, category, created_at, updated_at, tags, importance FROM memory_entries WHERE id = ?1")
                .map_err(|e| OdinError::Database(format!("Failed to prepare get statement: {e}")))?;

            let result = stmt.query_row(params![id], |row| {
                Ok(MemoryRow {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    category: row.get(2)?,
                    created_at: row.get(3)?,
                    updated_at: row.get(4)?,
                    tags: row.get(5)?,
                    importance: row.get(6)?,
                })
            });

            match result {
                Ok(row) => {
                    let entry: MemoryEntry =
                        row.try_into().map_err(|e: String| OdinError::Database(e))?;
                    Ok(Some(entry))
                }
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(error) => Err(OdinError::Database(format!(
                    "Failed to read memory entry: {error}"
                ))),
            }
        })
        .await
    }

    #[instrument(skip(self))]
    async fn search(&self, query: &str, limit: usize) -> OdinResult<Vec<MemoryEntry>> {
        let query = query.to_string();
        self.run_db(move |conn| {
            let pattern = format!("%{}%", query.replace('%', "\\%").replace('_', "\\_"));
            let limit = limit as i64;
            let mut stmt = conn
                .prepare(
                    "SELECT id, content, category, created_at, updated_at, tags, importance
                     FROM memory_entries
                     WHERE content LIKE ?1 ESCAPE '\\'
                     ORDER BY updated_at DESC
                     LIMIT ?2",
                )
                .map_err(|e| {
                    OdinError::Database(format!("Failed to prepare search statement: {e}"))
                })?;

            let rows = stmt
                .query_map(params![pattern, limit], |row| {
                    Ok(MemoryRow {
                        id: row.get(0)?,
                        content: row.get(1)?,
                        category: row.get(2)?,
                        created_at: row.get(3)?,
                        updated_at: row.get(4)?,
                        tags: row.get(5)?,
                        importance: row.get(6)?,
                    })
                })
                .map_err(|e| OdinError::Database(format!("Failed to execute search: {e}")))?;

            let mut results = Vec::new();
            for row in rows {
                let row =
                    row.map_err(|e| OdinError::Database(format!("Error reading search row: {e}")))?;
                match MemoryEntry::try_from(row) {
                    Ok(entry) => results.push(entry),
                    Err(e) => tracing::warn!("Skipping malformed memory entry during search: {e}"),
                }
            }

            Ok(results)
        })
        .await
    }

    #[instrument(skip(self))]
    async fn list_by_category(
        &self,
        category: MemoryCategory,
        limit: usize,
    ) -> OdinResult<Vec<MemoryEntry>> {
        let category_str = serde_json::to_value(category)
            .map(|v| v.as_str().unwrap_or("fact").to_string())
            .unwrap_or_else(|_| "fact".to_string());
        self.run_db(move |conn| {
            let limit = limit as i64;
            let mut stmt = conn
                .prepare(
                    "SELECT id, content, category, created_at, updated_at, tags, importance
                     FROM memory_entries
                     WHERE category = ?1
                     ORDER BY created_at DESC
                     LIMIT ?2",
                )
                .map_err(|e| {
                    OdinError::Database(format!("Failed to prepare category statement: {e}"))
                })?;

            let rows = stmt
                .query_map(params![category_str, limit], |row| {
                    Ok(MemoryRow {
                        id: row.get(0)?,
                        content: row.get(1)?,
                        category: row.get(2)?,
                        created_at: row.get(3)?,
                        updated_at: row.get(4)?,
                        tags: row.get(5)?,
                        importance: row.get(6)?,
                    })
                })
                .map_err(|e| {
                    OdinError::Database(format!("Failed to execute category query: {e}"))
                })?;

            let mut results = Vec::new();
            for row in rows {
                let row = row
                    .map_err(|e| OdinError::Database(format!("Error reading category row: {e}")))?;
                match MemoryEntry::try_from(row) {
                    Ok(entry) => results.push(entry),
                    Err(e) => tracing::warn!("Skipping malformed memory entry: {e}"),
                }
            }

            Ok(results)
        })
        .await
    }

    #[instrument(skip(self))]
    async fn delete(&self, id: &str) -> OdinResult<()> {
        let id = id.to_string();
        self.run_db(move |conn| {
            let affected = conn
                .execute("DELETE FROM memory_entries WHERE id = ?1", params![id])
                .map_err(|e| OdinError::Database(format!("Failed to delete memory entry: {e}")))?;

            if affected == 0 {
                tracing::warn!(entry_id = %id, "Attempted to delete non-existent memory entry");
            }

            Ok(())
        })
        .await
    }

    #[instrument(skip(self))]
    async fn count(&self) -> OdinResult<usize> {
        self.run_db(|conn| {
            let total: i64 = conn
                .query_row("SELECT COUNT(*) FROM memory_entries", [], |row| row.get(0))
                .map_err(|e| OdinError::Database(format!("Failed to count entries: {e}")))?;

            Ok(total as usize)
        })
        .await
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use odin_core::MemoryEntry;
    use uuid::Uuid;

    fn make_entry(content: &str, category: MemoryCategory) -> MemoryEntry {
        let now = Utc::now();
        MemoryEntry {
            id: Uuid::new_v4().to_string(),
            content: content.to_string(),
            category,
            created_at: now,
            updated_at: now,
            tags: vec![],
            importance: 1.0,
        }
    }

    #[tokio::test]
    async fn test_store_and_get() {
        let store = SqliteMemoryStore::in_memory().unwrap();
        let entry = make_entry("Hello, world!", MemoryCategory::Fact);

        store.store(entry.clone()).await.unwrap();
        let retrieved = store.get(&entry.id).await.unwrap();

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Hello, world!");
    }

    #[tokio::test]
    async fn test_get_nonexistent() {
        let store = SqliteMemoryStore::in_memory().unwrap();
        let result = store.get("nonexistent-id").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_get_propagates_malformed_rows() {
        let store = SqliteMemoryStore::in_memory().unwrap();
        store
            .conn
            .lock()
            .unwrap()
            .execute(
                "INSERT INTO memory_entries
                 (id, content, category, created_at, updated_at, tags, importance)
                 VALUES ('bad', 'content', 'not-a-category', 'now', 'now', '[]', 1.0)",
                [],
            )
            .unwrap();

        let result = store.get("bad").await;
        assert!(result.is_err(), "malformed rows must not look like misses");
    }

    #[tokio::test]
    async fn test_search() {
        let store = SqliteMemoryStore::in_memory().unwrap();

        store
            .store(make_entry("Alice likes apples", MemoryCategory::Fact))
            .await
            .unwrap();
        store
            .store(make_entry("Bob prefers bananas", MemoryCategory::Fact))
            .await
            .unwrap();
        store
            .store(make_entry("Charlie codes in Rust", MemoryCategory::Fact))
            .await
            .unwrap();

        let results = store.search("apple", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("apple"));

        let results = store.search("rust", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Rust"));
    }

    #[tokio::test]
    async fn test_search_case_sensitive() {
        let store = SqliteMemoryStore::in_memory().unwrap();

        store
            .store(make_entry("Rust is great", MemoryCategory::Fact))
            .await
            .unwrap();

        // SQLite LIKE is case-insensitive for ASCII by default
        let results = store.search("rust", 10).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_list_by_category() {
        let store = SqliteMemoryStore::in_memory().unwrap();

        let pref = make_entry("Likes coffee", MemoryCategory::Preference);
        let fact = make_entry("Earth orbits the Sun", MemoryCategory::Fact);
        let entity = make_entry("Alice is a friend", MemoryCategory::Entity);

        store.store(pref).await.unwrap();
        store.store(fact).await.unwrap();
        store.store(entity).await.unwrap();

        let facts = store
            .list_by_category(MemoryCategory::Fact, 10)
            .await
            .unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].content, "Earth orbits the Sun");

        let prefs = store
            .list_by_category(MemoryCategory::Preference, 10)
            .await
            .unwrap();
        assert_eq!(prefs.len(), 1);
        assert_eq!(prefs[0].content, "Likes coffee");
    }

    #[tokio::test]
    async fn test_delete() {
        let store = SqliteMemoryStore::in_memory().unwrap();
        let entry = make_entry("Delete me", MemoryCategory::Fact);

        store.store(entry.clone()).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);

        store.delete(&entry.id).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_count() {
        let store = SqliteMemoryStore::in_memory().unwrap();

        assert_eq!(store.count().await.unwrap(), 0);

        store
            .store(make_entry("One", MemoryCategory::Fact))
            .await
            .unwrap();
        store
            .store(make_entry("Two", MemoryCategory::Fact))
            .await
            .unwrap();
        store
            .store(make_entry("Three", MemoryCategory::Fact))
            .await
            .unwrap();

        assert_eq!(store.count().await.unwrap(), 3);
    }

    #[tokio::test]
    async fn test_update_existing() {
        let store = SqliteMemoryStore::in_memory().unwrap();
        let mut entry = make_entry("Original content", MemoryCategory::Fact);
        let id = entry.id.clone();

        store.store(entry.clone()).await.unwrap();

        entry.content = "Updated content".to_string();
        entry.updated_at = Utc::now();
        store.store(entry).await.unwrap();

        let retrieved = store.get(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.content, "Updated content");
        assert_eq!(retrieved.category, MemoryCategory::Fact);
    }

    #[tokio::test]
    async fn test_empty_search() {
        let store = SqliteMemoryStore::in_memory().unwrap();
        let results = store.search("nonexistent", 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_file_based_store() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("test_odin_memory_{}.db", Uuid::new_v4()));
        let path_str = path.to_str().unwrap().to_string();

        // Create store and insert data
        {
            let store = SqliteMemoryStore::new(&path_str).unwrap();
            store
                .store(make_entry("Persistent data", MemoryCategory::Fact))
                .await
                .unwrap();
            assert_eq!(store.count().await.unwrap(), 1);
        }

        // Re-open and verify data persists
        {
            let store = SqliteMemoryStore::new(&path_str).unwrap();
            assert_eq!(store.count().await.unwrap(), 1);
            let results = store.search("Persistent", 10).await.unwrap();
            assert_eq!(results.len(), 1);
        }

        // Cleanup
        let _ = std::fs::remove_file(&path_str);
    }

    #[tokio::test]
    async fn retention_keeps_only_the_newest_entries() {
        let store = SqliteMemoryStore::in_memory()
            .unwrap()
            .with_retention_limit(2);
        let mut oldest = make_entry("oldest", MemoryCategory::Fact);
        oldest.updated_at = Utc::now() - chrono::TimeDelta::hours(2);
        let mut middle = make_entry("middle", MemoryCategory::Fact);
        middle.updated_at = Utc::now() - chrono::TimeDelta::hours(1);
        let newest = make_entry("newest", MemoryCategory::Fact);

        store.store(oldest).await.unwrap();
        store.store(middle).await.unwrap();
        store.store(newest).await.unwrap();

        assert_eq!(store.count().await.unwrap(), 2);
        let entries = store.search("", 10).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].content, "newest");
        assert_eq!(entries[1].content, "middle");
    }

    #[test]
    fn file_store_uses_wal_and_records_schema_version() {
        let path = std::env::temp_dir().join(format!("odin-memory-{}.db", Uuid::new_v4()));
        let store = SqliteMemoryStore::new(path.to_str().unwrap()).unwrap();
        let connection = store.conn.lock().unwrap();
        let journal_mode: String = connection
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .unwrap();
        let migration_count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM memory_schema_migrations WHERE version = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(journal_mode, "wal");
        assert_eq!(migration_count, 1);
        drop(connection);
        drop(store);
        let _ = std::fs::remove_file(path);
    }
}
