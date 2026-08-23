//! File-level locking for concurrent sub-agent execution.
//!
//! The FileLockManager ensures that while multiple sub-agents can read files
//! concurrently, only one agent can write to a file at a time. Conflicting
//! writes are queued. When the lock is released, the next writer is dequeued.
//!
//! IMPORTANT: Writes always use exclusive locks. Reads are concurrent.

use dashmap::DashMap;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;

/// Mode for a file lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockMode {
    /// Shared read lock — multiple agents can read concurrently.
    Read,
    /// Exclusive write lock — only one agent at a time.
    Write,
}

/// A file lock held by an agent.
#[derive(Debug, Clone)]
pub struct FileLock {
    /// The file path being locked (relative to workspace root).
    pub path: String,
    /// Lock mode.
    pub mode: LockMode,
    /// Agent ID holding the lock.
    pub agent_id: Uuid,
}

/// Manages file-level locks for parallel sub-agent execution.
///
/// Rules:
/// - Multiple agents can hold READ locks on the same file.
/// - Only ONE agent can hold a WRITE lock on a file.
/// - No READ locks are allowed while a WRITE lock is held.
/// - Writers are queued FIFO.
pub struct FileLockManager {
    /// Canonical workspace root used to turn aliases into one lock identity.
    workspace_root: Arc<PathBuf>,
    /// Currently held locks: file_path → Vec<FileLock>
    locks: Arc<DashMap<String, Vec<FileLock>>>,
    /// Queue of agents waiting for a write lock: file_path → VecDeque<(agent_id, queued_at)>
    write_queue: Arc<DashMap<String, VecDeque<QueuedWriter>>>,
}

/// A writer waiting in queue.
#[derive(Debug, Clone)]
struct QueuedWriter {
    agent_id: Uuid,
}

impl Default for FileLockManager {
    fn default() -> Self {
        Self::new()
    }
}

impl FileLockManager {
    /// Create a new empty file lock manager.
    pub fn new() -> Self {
        let root = std::env::current_dir()
            .ok()
            .and_then(|path| std::fs::canonicalize(path).ok())
            .unwrap_or_else(|| PathBuf::from("."));
        Self::from_root(root)
    }

    /// Create a lock manager rooted at a configured workspace.
    pub fn with_workspace_root(path: impl AsRef<Path>) -> Result<Self, String> {
        let root = std::fs::canonicalize(path.as_ref()).map_err(|error| {
            format!(
                "Unable to resolve file-lock workspace root '{}': {error}",
                path.as_ref().display()
            )
        })?;
        if !root.is_dir() {
            return Err(format!(
                "File-lock workspace root '{}' is not a directory",
                root.display()
            ));
        }
        Ok(Self::from_root(root))
    }

    fn from_root(root: PathBuf) -> Self {
        Self {
            workspace_root: Arc::new(root),
            locks: Arc::new(DashMap::new()),
            write_queue: Arc::new(DashMap::new()),
        }
    }

    /// Normalize a path against the workspace root for use in lifecycle state
    /// and lock operations.
    pub fn normalize_path(&self, raw: &str) -> Result<String, String> {
        let raw_path = Path::new(raw);
        let candidate = if raw_path.is_absolute() {
            raw_path.to_path_buf()
        } else {
            self.workspace_root.join(raw_path)
        };

        let mut missing = Vec::new();
        let mut current = candidate.as_path();
        while !current.exists() {
            let component = current
                .file_name()
                .ok_or_else(|| format!("Unable to resolve lock path '{}'", raw_path.display()))?;
            missing.push(component.to_os_string());
            current = current
                .parent()
                .ok_or_else(|| format!("Unable to resolve lock path '{}'", raw_path.display()))?;
        }

        let mut resolved = std::fs::canonicalize(current).map_err(|error| {
            format!(
                "Unable to resolve lock path '{}': {error}",
                raw_path.display()
            )
        })?;
        for component in missing.iter().rev() {
            resolved.push(component);
        }

        if !resolved.starts_with(self.workspace_root.as_path()) {
            return Err(format!(
                "Lock path '{}' is outside workspace root '{}'",
                raw,
                self.workspace_root.display()
            ));
        }

        let relative = resolved
            .strip_prefix(self.workspace_root.as_path())
            .map_err(|_| format!("Unable to make lock path '{}' workspace-relative", raw))?;
        let normalized = relative.to_string_lossy().to_string();
        Ok(if normalized.is_empty() {
            ".".into()
        } else {
            normalized
        })
    }

    /// Try to acquire a read lock on a file.
    /// Returns `Ok(())` if acquired, or `Err(msg)` if a write lock is held.
    pub fn acquire_read(&self, path: &str, agent_id: Uuid) -> Result<(), String> {
        let path = self.normalize_path(path)?;
        let mut entry = self.locks.entry(path.clone()).or_default();

        // A write lock held by this agent already grants read access, and
        // repeated read acquisition must not create duplicate holders.
        if entry
            .iter()
            .any(|lock| lock.agent_id == agent_id && lock.mode == LockMode::Write)
            || entry
                .iter()
                .any(|lock| lock.agent_id == agent_id && lock.mode == LockMode::Read)
        {
            return Ok(());
        }

        // A different agent's write lock excludes readers.
        if entry.iter().any(|lock| lock.mode == LockMode::Write) {
            return Err(format!(
                "Cannot acquire read lock on '{}': write lock held",
                path
            ));
        }

        entry.push(FileLock {
            path: path.to_string(),
            mode: LockMode::Read,
            agent_id,
        });

        tracing::debug!(agent_id = %agent_id, holders = entry.len(), "Read lock acquired");
        Ok(())
    }

    /// Try to acquire a write lock on a file.
    /// If a lock (read or write) is held, the writer is queued.
    /// Returns `Ok(())` if acquired immediately, or `Err(queued_message)` if queued.
    pub fn acquire_write(&self, path: &str, agent_id: Uuid) -> Result<(), String> {
        let path = self.normalize_path(path)?;
        let mut entry = self.locks.entry(path.clone()).or_default();

        // Re-acquiring a lock that was granted from the queue is idempotent.
        // Without this, a woken writer queues behind its own lock forever.
        if entry
            .iter()
            .any(|lock| lock.mode == LockMode::Write && lock.agent_id == agent_id)
        {
            return Ok(());
        }

        if entry.is_empty() {
            // No locks held — acquire immediately
            entry.push(FileLock {
                path: path.to_string(),
                mode: LockMode::Write,
                agent_id,
            });
            tracing::info!(agent_id = %agent_id, "Write lock acquired");
            Ok(())
        } else {
            // Locks held — queue this writer once.
            let mut queue = self.write_queue.entry(path.to_string()).or_default();
            if !queue.iter().any(|writer| writer.agent_id == agent_id) {
                queue.push_back(QueuedWriter { agent_id });
            }
            tracing::info!(agent_id = %agent_id, position = queue.len(), "Write lock queued");
            Err(format!(
                "Queued for write lock on '{}' (position: {})",
                path,
                queue.len()
            ))
        }
    }

    /// Release all locks held by an agent and remove its pending write requests.
    /// When a write lock is released, the next non-cancelled writer is granted the lock.
    pub fn release_all(&self, agent_id: Uuid) -> Vec<String> {
        let mut released_paths = Vec::new();

        // A cancelled/failed waiter must not receive a lock later.
        let queued_paths: Vec<String> = self
            .write_queue
            .iter()
            .map(|entry| entry.key().clone())
            .collect();
        for path in queued_paths {
            if let Some(mut queue) = self.write_queue.get_mut(&path) {
                queue.retain(|writer| writer.agent_id != agent_id);
                if queue.is_empty() {
                    drop(queue);
                    self.write_queue.remove(&path);
                }
            }
        }

        // Iterate over all locked files
        let paths: Vec<String> = self.locks.iter().map(|entry| entry.key().clone()).collect();

        for path in &paths {
            if let Some(mut entry) = self.locks.get_mut(path) {
                let had_lock = entry.iter().any(|lock| lock.agent_id == agent_id);
                if !had_lock {
                    continue;
                }
                let had_write = entry
                    .iter()
                    .any(|l| l.mode == LockMode::Write && l.agent_id == agent_id);
                entry.retain(|l| l.agent_id != agent_id);

                if entry.is_empty() {
                    // Keep the lock entry held while replacing the released
                    // holder. Removing it before granting would let another
                    // writer acquire the empty entry in between.
                    let (next, queue_empty) =
                        if let Some(mut queue) = self.write_queue.get_mut(path) {
                            let next = queue.pop_front();
                            let queue_empty = queue.is_empty();
                            drop(queue);
                            (next, queue_empty)
                        } else {
                            (None, false)
                        };
                    if queue_empty {
                        self.write_queue.remove(path);
                    }
                    if let Some(next) = next {
                        entry.push(FileLock {
                            path: path.clone(),
                            mode: LockMode::Write,
                            agent_id: next.agent_id,
                        });
                        tracing::info!(agent_id = %next.agent_id, "Queued write lock granted");
                    } else {
                        drop(entry);
                        self.locks.remove(path);
                    }
                }

                released_paths.push(path.clone());
                if had_write {
                    tracing::info!(agent_id = %agent_id, "Write lock released");
                }
            }
        }

        released_paths
    }

    /// Release a specific lock.
    pub fn release(&self, path: &str, agent_id: Uuid) {
        let Ok(path) = self.normalize_path(path) else {
            return;
        };
        if let Some(mut entry) = self.locks.get_mut(&path) {
            entry.retain(|l| l.agent_id != agent_id);
            if entry.is_empty() {
                // Keep the lock entry held through the handoff so a new
                // writer cannot observe an empty entry and acquire alongside
                // the queued writer.
                let (next, queue_empty) = if let Some(mut queue) = self.write_queue.get_mut(path) {
                    let next = queue.pop_front();
                    let queue_empty = queue.is_empty();
                    drop(queue);
                    (next, queue_empty)
                } else {
                    (None, false)
                };
                if queue_empty {
                    self.write_queue.remove(path);
                }
                if let Some(next) = next {
                    entry.push(FileLock {
                        path: path.to_string(),
                        mode: LockMode::Write,
                        agent_id: next.agent_id,
                    });
                } else {
                    drop(entry);
                    self.locks.remove(path);
                }
            }
        }
    }

    /// Check if a file has any locks.
    pub fn is_locked(&self, path: &str) -> bool {
        self.normalize_path(path)
            .map(|path| self.locks.contains_key(&path))
            .unwrap_or(false)
    }

    /// Check if a file has a write lock.
    pub fn has_write_lock(&self, path: &str) -> bool {
        let Ok(path) = self.normalize_path(path) else {
            return false;
        };
        self.locks
            .get(&path)
            .map(|entry| entry.iter().any(|l| l.mode == LockMode::Write))
            .unwrap_or(false)
    }

    /// Get all agents holding locks on a file.
    pub fn lock_holders(&self, path: &str) -> Vec<Uuid> {
        let Ok(path) = self.normalize_path(path) else {
            return Vec::new();
        };
        self.locks
            .get(&path)
            .map(|entry| entry.iter().map(|l| l.agent_id).collect())
            .unwrap_or_default()
    }

    /// Get the queue length for a file.
    pub fn queue_length(&self, path: &str) -> usize {
        self.normalize_path(path)
            .ok()
            .and_then(|path| self.write_queue.get(&path).map(|q| q.len()))
            .unwrap_or(0)
    }

    /// List all currently locked files.
    pub fn locked_files(&self) -> Vec<String> {
        self.locks.iter().map(|e| e.key().clone()).collect()
    }

    /// Summary of the lock manager state.
    pub fn summary(&self) -> FileLockSummary {
        let locked_count = self.locks.len();
        let queued_writers: usize = self.write_queue.iter().map(|q| q.len()).sum();
        let write_locked_count = self
            .locks
            .iter()
            .filter(|e| e.iter().any(|l| l.mode == LockMode::Write))
            .count();

        FileLockSummary {
            total_locked_files: locked_count,
            write_locked_files: write_locked_count,
            queued_writers,
        }
    }
}

/// Summary of the file lock manager state.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FileLockSummary {
    pub total_locked_files: usize,
    pub write_locked_files: usize,
    pub queued_writers: usize,
}

/// A serializable snapshot of the entire file lock state.
/// Used for persistence across restarts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LockSnapshot {
    /// Held locks: path → (mode, agent_id)
    pub held_locks: Vec<LockSnapshotEntry>,
    /// Write queue: path → queued agents
    pub write_queues: Vec<LockSnapshotQueue>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LockSnapshotEntry {
    pub path: String,
    pub mode: String, // "read" or "write"
    pub agent_id: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LockSnapshotQueue {
    pub path: String,
    pub queued_agents: Vec<String>,
}

impl FileLockManager {
    /// Take a snapshot of the current lock state.
    pub fn snapshot(&self) -> LockSnapshot {
        let held_locks: Vec<LockSnapshotEntry> = self
            .locks
            .iter()
            .flat_map(|entry| {
                let path = entry.key().clone();
                // Collect into a Vec first to avoid borrow issues with DashMap iterators
                let entries: Vec<LockSnapshotEntry> = entry
                    .value()
                    .iter()
                    .map(|lock| LockSnapshotEntry {
                        path: path.clone(),
                        mode: match lock.mode {
                            LockMode::Read => "read".to_string(),
                            LockMode::Write => "write".to_string(),
                        },
                        agent_id: lock.agent_id.to_string(),
                    })
                    .collect();
                entries
            })
            .collect();

        let write_queues: Vec<LockSnapshotQueue> = self
            .write_queue
            .iter()
            .map(|entry| LockSnapshotQueue {
                path: entry.key().clone(),
                queued_agents: entry
                    .value()
                    .iter()
                    .map(|q| q.agent_id.to_string())
                    .collect(),
            })
            .collect();

        LockSnapshot {
            held_locks,
            write_queues,
        }
    }

    /// Restore lock state from a snapshot. Existing locks are cleared first.
    pub fn restore_from(&self, snapshot: &LockSnapshot) {
        // Clear existing state
        self.locks.clear();
        self.write_queue.clear();

        // Restore held locks (Note: these are reconstructed as-is;
        // lock conflict checks happen naturally on re-acquisition)
        for entry in &snapshot.held_locks {
            if let Ok(agent_id) = Uuid::parse_str(&entry.agent_id) {
                match entry.mode.as_str() {
                    "write" => {
                        let _ = self.acquire_write(&entry.path, agent_id);
                    }
                    _ => {
                        let _ = self.acquire_read(&entry.path, agent_id);
                    }
                }
            }
        }

        // Restore write queues
        for queue in &snapshot.write_queues {
            let mut deque: VecDeque<QueuedWriter> = VecDeque::new();
            for agent_str in &queue.queued_agents {
                if let Ok(agent_id) = Uuid::parse_str(agent_str) {
                    deque.push_back(QueuedWriter { agent_id });
                }
            }
            if !deque.is_empty() {
                self.write_queue.insert(queue.path.clone(), deque);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_lock_concurrent() {
        let mgr = FileLockManager::new();
        let a1 = Uuid::new_v4();
        let a2 = Uuid::new_v4();

        assert!(mgr.acquire_read("test.txt", a1).is_ok());
        assert!(mgr.acquire_read("test.txt", a2).is_ok());

        let holders = mgr.lock_holders("test.txt");
        assert_eq!(holders.len(), 2);
    }

    #[test]
    fn test_write_lock_exclusive() {
        let mgr = FileLockManager::new();
        let a1 = Uuid::new_v4();
        let a2 = Uuid::new_v4();

        // First write succeeds
        assert!(mgr.acquire_write("test.txt", a1).is_ok());

        // Second write is queued
        let result = mgr.acquire_write("test.txt", a2);
        assert!(result.is_err()); // queued
        assert!(mgr.queue_length("test.txt") > 0);
    }

    #[test]
    fn test_read_blocked_by_write() {
        let mgr = FileLockManager::new();
        let writer = Uuid::new_v4();
        let reader = Uuid::new_v4();

        mgr.acquire_write("test.txt", writer).unwrap();
        let result = mgr.acquire_read("test.txt", reader);
        assert!(result.is_err());
    }

    #[test]
    fn test_release_grants_next_writer() {
        let mgr = FileLockManager::new();
        let a1 = Uuid::new_v4();
        let a2 = Uuid::new_v4();

        mgr.acquire_write("test.txt", a1).unwrap();
        let r2 = mgr.acquire_write("test.txt", a2);
        assert!(r2.is_err()); // queued

        // Release a1's lock
        mgr.release_all(a1);

        // a2 should now have the lock
        assert!(mgr.has_write_lock("test.txt"));
        let holders = mgr.lock_holders("test.txt");
        assert_eq!(holders, vec![a2]);
    }

    #[test]
    fn test_release_all_clears_multiple() {
        let mgr = FileLockManager::new();
        let agent = Uuid::new_v4();

        mgr.acquire_read("a.txt", agent).unwrap();
        mgr.acquire_read("b.txt", agent).unwrap();

        let paths = mgr.release_all(agent);
        assert_eq!(paths.len(), 2);
        assert!(!mgr.is_locked("a.txt"));
        assert!(!mgr.is_locked("b.txt"));
    }

    #[test]
    fn test_summary() {
        let mgr = FileLockManager::new();
        let a1 = Uuid::new_v4();
        let a2 = Uuid::new_v4();

        mgr.acquire_write("file1.txt", a1).unwrap();
        mgr.acquire_read("file2.txt", a2).unwrap();
        let _ = mgr.acquire_write("file1.txt", a2); // queued

        let summary = mgr.summary();
        assert_eq!(summary.total_locked_files, 2);
        assert_eq!(summary.write_locked_files, 1);
        assert_eq!(summary.queued_writers, 1);
    }

    #[test]
    fn test_read_after_write_release() {
        let mgr = FileLockManager::new();
        let writer = Uuid::new_v4();
        let reader = Uuid::new_v4();

        mgr.acquire_write("test.txt", writer).unwrap();
        mgr.release_all(writer);

        // Now reader should be able to acquire
        assert!(mgr.acquire_read("test.txt", reader).is_ok());
    }

    #[test]
    fn test_reacquiring_read_access_is_idempotent() {
        let mgr = FileLockManager::new();
        let agent = Uuid::new_v4();

        mgr.acquire_read("read.txt", agent).unwrap();
        mgr.acquire_read("read.txt", agent).unwrap();
        assert_eq!(
            mgr.snapshot()
                .held_locks
                .iter()
                .filter(|lock| lock.path == "read.txt")
                .count(),
            1
        );

        mgr.acquire_write("write.txt", agent).unwrap();
        mgr.acquire_read("write.txt", agent).unwrap();
        assert_eq!(
            mgr.snapshot()
                .held_locks
                .iter()
                .filter(|lock| lock.path == "write.txt")
                .count(),
            1
        );
    }

    #[test]
    fn test_reacquiring_granted_write_lock_is_idempotent() {
        let mgr = FileLockManager::new();
        let first = Uuid::new_v4();
        let queued = Uuid::new_v4();

        mgr.acquire_write("test.txt", first).unwrap();
        assert!(mgr.acquire_write("test.txt", queued).is_err());
        mgr.release_all(first);

        assert!(mgr.acquire_write("test.txt", queued).is_ok());
        assert_eq!(mgr.lock_holders("test.txt"), vec![queued]);
        assert_eq!(mgr.queue_length("test.txt"), 0);
    }

    #[test]
    fn test_repeated_write_attempt_does_not_duplicate_queue_entry() {
        let mgr = FileLockManager::new();
        let holder = Uuid::new_v4();
        let queued = Uuid::new_v4();

        mgr.acquire_write("test.txt", holder).unwrap();
        assert!(mgr.acquire_write("test.txt", queued).is_err());
        assert!(mgr.acquire_write("test.txt", queued).is_err());

        assert_eq!(mgr.queue_length("test.txt"), 1);
    }

    #[test]
    fn test_release_all_removes_waiting_writer() {
        let mgr = FileLockManager::new();
        let holder = Uuid::new_v4();
        let cancelled = Uuid::new_v4();

        mgr.acquire_write("test.txt", holder).unwrap();
        assert!(mgr.acquire_write("test.txt", cancelled).is_err());
        let released = mgr.release_all(cancelled);
        assert!(released.is_empty());
        assert_eq!(mgr.queue_length("test.txt"), 0);

        mgr.release_all(holder);
        assert!(!mgr.is_locked("test.txt"));
    }

    #[test]
    fn test_equivalent_path_spellings_share_one_lock() {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("file.txt");
        std::fs::write(&file, "data").unwrap();
        let manager = FileLockManager::with_workspace_root(root.path()).unwrap();
        let holder = Uuid::new_v4();
        let waiter = Uuid::new_v4();

        manager.acquire_write("file.txt", holder).unwrap();
        assert!(manager.acquire_read("./file.txt", waiter).is_err());
        assert!(
            manager
                .acquire_write(file.to_str().unwrap(), waiter)
                .is_err()
        );
        assert_eq!(manager.lock_holders("file.txt"), vec![holder]);
    }

    #[cfg(unix)]
    #[test]
    fn test_symlink_alias_shares_one_lock() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("file.txt");
        let alias = root.path().join("alias.txt");
        std::fs::write(&file, "data").unwrap();
        symlink(&file, &alias).unwrap();
        let manager = FileLockManager::with_workspace_root(root.path()).unwrap();
        let holder = Uuid::new_v4();
        let waiter = Uuid::new_v4();

        manager.acquire_write("file.txt", holder).unwrap();
        assert!(manager.acquire_read("alias.txt", waiter).is_err());
    }
}
