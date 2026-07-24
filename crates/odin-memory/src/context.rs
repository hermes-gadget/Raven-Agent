//! Helpers for packing retrieved memory into agent context budgets.

use odin_core::error::OdinResult;
use odin_core::traits::MemoryStore;
use odin_core::types::{MemoryCategory, MemoryEntry};
use uuid::Uuid;

/// Default character budget injected into a sub-agent task context.
pub const DEFAULT_CONTEXT_CHARS: usize = 4_000;

/// Retrieve relevant memory entries for a task goal within a character budget.
///
/// Returns `None` when no relevant content fits. Entries are ordered by the
/// store's search ranking and truncated to `budget_chars`.
pub async fn retrieve_task_context(
    store: &dyn MemoryStore,
    goal: &str,
    max_entries: usize,
    budget_chars: usize,
) -> OdinResult<Option<String>> {
    if max_entries == 0 || budget_chars == 0 || goal.trim().is_empty() {
        return Ok(None);
    }

    let terms: Vec<&str> = goal
        .split(|c: char| !c.is_alphanumeric())
        .filter(|term| term.len() >= 3)
        .take(6)
        .collect();
    let query = if terms.is_empty() {
        goal.chars().take(64).collect::<String>()
    } else {
        // Use the longest term as primary query for best LIKE matching
        terms.iter().max_by_key(|t| t.len()).unwrap().to_string()
    };

    let entries = store.search(&query, max_entries).await?;
    if entries.is_empty() {
        return Ok(None);
    }

    let mut packed = String::from("Relevant memory:\n");
    let mut used = packed.len();
    for entry in entries {
        let line = format!("- [{}] {}\n", entry.category, entry.content.trim());
        if used + line.len() > budget_chars {
            let remaining = budget_chars.saturating_sub(used);
            if remaining > 16 {
                packed.push_str(&line.chars().take(remaining.saturating_sub(1)).collect::<String>());
                packed.push('\n');
            }
            break;
        }
        packed.push_str(&line);
        used += line.len();
    }

    if packed.trim() == "Relevant memory:" {
        return Ok(None);
    }
    Ok(Some(packed))
}

/// Store a redacted task outcome with run/task/agent provenance tags.
pub async fn store_task_outcome(
    store: &dyn MemoryStore,
    content: impl Into<String>,
    run_id: &str,
    task_id: &str,
    agent_id: &str,
    importance: f32,
) -> OdinResult<()> {
    let content = content.into();
    if content.trim().is_empty() {
        return Ok(());
    }
    let now = chrono::Utc::now();
    let entry = MemoryEntry {
        id: Uuid::new_v4().to_string(),
        content,
        category: MemoryCategory::Event,
        created_at: now,
        updated_at: now,
        tags: vec![
            format!("run:{run_id}"),
            format!("task:{task_id}"),
            format!("agent:{agent_id}"),
            "orchestrated".into(),
        ],
        importance: importance.clamp(0.0, 1.0),
    };
    store.store(entry).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SqliteMemoryStore;
    use odin_core::types::MemoryCategory;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingStore {
        inner: SqliteMemoryStore,
        searches: AtomicUsize,
        stores: AtomicUsize,
    }

    impl CountingStore {
        fn new() -> Self {
            Self {
                inner: SqliteMemoryStore::in_memory().unwrap(),
                searches: AtomicUsize::new(0),
                stores: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl MemoryStore for CountingStore {
        async fn store(&self, entry: MemoryEntry) -> OdinResult<()> {
            self.stores.fetch_add(1, Ordering::SeqCst);
            self.inner.store(entry).await
        }
        async fn get(&self, id: &str) -> OdinResult<Option<MemoryEntry>> {
            self.inner.get(id).await
        }
        async fn search(&self, query: &str, limit: usize) -> OdinResult<Vec<MemoryEntry>> {
            self.searches.fetch_add(1, Ordering::SeqCst);
            self.inner.search(query, limit).await
        }
        async fn list_by_category(
            &self,
            category: MemoryCategory,
            limit: usize,
        ) -> OdinResult<Vec<MemoryEntry>> {
            self.inner.list_by_category(category, limit).await
        }
        async fn delete(&self, id: &str) -> OdinResult<()> {
            self.inner.delete(id).await
        }
        async fn count(&self) -> OdinResult<usize> {
            self.inner.count().await
        }
    }

    #[tokio::test]
    async fn retrieve_respects_budget_and_relevance() {
        let store = CountingStore::new();
        let now = chrono::Utc::now();
        store
            .store(MemoryEntry {
                id: "1".into(),
                content: "Configure mesh radio channel for SX1262 band".into(),
                category: MemoryCategory::Fact,
                created_at: now,
                updated_at: now,
                tags: vec![],
                importance: 0.8,
            })
            .await
            .unwrap();
        store
            .store(MemoryEntry {
                id: "2".into(),
                content: "Unrelated cooking recipe for pasta".into(),
                category: MemoryCategory::Fact,
                created_at: now,
                updated_at: now,
                tags: vec![],
                importance: 0.2,
            })
            .await
            .unwrap();

        let context = retrieve_task_context(&store, "configure mesh radio SX1262", 5, 200)
            .await
            .unwrap()
            .expect("expected memory hit");
        assert!(context.contains("SX1262"));
        assert!(!context.contains("pasta"));
        assert!(context.len() <= 200);
        assert_eq!(store.searches.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn store_outcome_tags_provenance() {
        let store = CountingStore::new();
        store_task_outcome(&store, "agent finished chat screen", "run-1", "task-1", "agent-1", 0.7)
            .await
            .unwrap();
        let hits = store.search("chat screen", 5).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].tags.iter().any(|tag| tag == "run:run-1"));
        assert!(hits[0].tags.iter().any(|tag| tag == "agent:agent-1"));
        assert_eq!(store.stores.load(Ordering::SeqCst), 1);
    }
}
