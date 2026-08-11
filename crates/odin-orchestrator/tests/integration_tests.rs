//! Integration tests for Raven Agent orchestration.
//!
//! Tests the full orchestration pipeline: decomposition, parallel execution,
//! file locking, conflict detection, and merge resolution.

use odin_orchestrator::{
    Composer,
    composer::ComposerConfig,
    file_lock::FileLockManager,
    lifecycle::AgentPhase,
    merge::{MergeStrategy, SubAgentResult},
    sub_agent::SubAgentConfigBuilder,
};
use std::sync::Arc;
use uuid::Uuid;

// ── Helpers ──────────────────────────────────────────────────────────

fn make_result_with_id(
    id: Uuid,
    name: &str,
    success: bool,
    files: Vec<&str>,
    summary: &str,
) -> SubAgentResult {
    SubAgentResult {
        agent_id: id,
        name: name.into(),
        summary: summary.into(),
        output: None,
        modified_files: files.iter().map(|s| s.to_string()).collect(),
        success,
        error: if success {
            None
        } else {
            Some("task failed".into())
        },
        duration_ms: 100,
    }
}

// ── Test: 5 Unrelated Tasks in One Prompt ────────────────────────────

#[test]
fn test_five_unrelated_tasks_decomposed() {
    let mut composer = Composer::default();
    let goal = "fix the CLI bug, improve docs, add tests for scheduler, check provider fallback, update README";

    composer.intake(goal);
    let graph = composer.get_graph(goal).unwrap();

    // Should decompose into 5 tasks
    assert_eq!(
        graph.nodes.len(),
        5,
        "Expected 5 sub-tasks, got {}",
        graph.nodes.len()
    );

    // All nodes should be independent (no edges between them)
    let groups = graph.independent_groups();
    assert!(!groups.is_empty(), "Should have at least one group");

    // All 5 nodes should be in the first group (no dependencies)
    assert_eq!(groups[0].len(), 5, "All 5 tasks should be parallelizable");
}

// ── Test: Parallel Execution with No File Conflicts ──────────────────

#[test]
fn test_parallel_safe_execution() {
    let mut composer = Composer::new(ComposerConfig {
        max_parallel: 10,
        default_max_iterations: 50,
        auto_merge: true,
        merge_strategy: MergeStrategy::Concatenate,
        ..Default::default()
    });

    // Register 3 agents with non-overlapping files
    let id_a = composer.register_agent(
        SubAgentConfigBuilder::new("task-a", "do a")
            .write_files(vec!["a.txt".into()])
            .build(),
    );
    let id_b = composer.register_agent(
        SubAgentConfigBuilder::new("task-b", "do b")
            .write_files(vec!["b.txt".into()])
            .build(),
    );
    let id_c = composer.register_agent(
        SubAgentConfigBuilder::new("task-c", "do c")
            .write_files(vec!["c.txt".into()])
            .build(),
    );

    // All three can start in parallel — no file conflicts
    assert!(composer.start_agent(id_a).is_ok());
    assert!(composer.start_agent(id_b).is_ok());
    assert!(composer.start_agent(id_c).is_ok());

    // Verify all are running
    assert_eq!(
        composer.get_agent(&id_a).unwrap().0.phase,
        AgentPhase::Running
    );
    assert_eq!(
        composer.get_agent(&id_b).unwrap().0.phase,
        AgentPhase::Running
    );
    assert_eq!(
        composer.get_agent(&id_c).unwrap().0.phase,
        AgentPhase::Running
    );

    // Complete all
    composer.complete_agent(
        id_a,
        make_result_with_id(id_a, "a", true, vec!["a.txt"], "A done"),
    );
    composer.complete_agent(
        id_b,
        make_result_with_id(id_b, "b", true, vec!["b.txt"], "B done"),
    );
    composer.complete_agent(
        id_c,
        make_result_with_id(id_c, "c", true, vec!["c.txt"], "C done"),
    );

    // Collect and merge
    let results = composer.collect_results();
    assert_eq!(results.len(), 3);

    let merged = composer.merge_results(results);
    assert!(merged.success, "Merge should succeed: {}", merged.summary);
    assert_eq!(merged.modified_files.len(), 3);
    assert!(merged.conflicts.is_empty(), "Should have no conflicts");
}

// ── Test: Overlapping File Edits → Queue Behavior ────────────────────

/// Test: Overlapping file edits → queueing verifies both lifecycle and agent phase
#[test]
fn test_overlapping_file_edits_queued() {
    let mut composer = Composer::default();

    // Two agents want to write to the same file
    let id_a = composer.register_agent(
        SubAgentConfigBuilder::new("writer-a", "write to shared")
            .write_files(vec!["shared.rs".into()])
            .build(),
    );
    let id_b = composer.register_agent(
        SubAgentConfigBuilder::new("writer-b", "also write to shared")
            .write_files(vec!["shared.rs".into()])
            .build(),
    );

    // First agent gets the lock
    assert!(
        composer.start_agent(id_a).is_ok(),
        "Agent A should get the write lock"
    );
    assert!(composer.file_locks().has_write_lock("shared.rs"));

    // Second agent is queued
    let result = composer.start_agent(id_b);
    assert!(result.is_err(), "Agent B should be queued, not started");
    assert!(
        result.unwrap_err().contains("Queued"),
        "Error should mention queuing"
    );

    // Verify the lifecycle shows WaitingForLock
    let (_agent_b, lifecycle_b) = composer.get_agent(&id_b).unwrap();
    assert_eq!(
        lifecycle_b.phase,
        AgentPhase::WaitingForLock,
        "Lifecycle should show WaitingForLock"
    );

    // Complete A → lock auto-granted to B in queue
    composer.complete_agent(
        id_a,
        make_result_with_id(id_a, "writer-a", true, vec!["shared.rs"], "A wrote"),
    );

    // Lock is now held by B (auto-granted from queue)
    assert!(
        composer.file_locks().has_write_lock("shared.rs"),
        "Lock should be held by B (auto-granted from queue)"
    );

    // B now holds the lock — transition it directly
    if let Some((agent, lifecycle)) = composer.get_agent_mut(&id_b) {
        agent.phase = AgentPhase::Running;
        lifecycle.start();
    }

    // Verify B is running
    let (_agent_b, lifecycle_b) = composer.get_agent(&id_b).unwrap();
    assert_eq!(
        lifecycle_b.phase,
        AgentPhase::Running,
        "B should be running after lock auto-grant"
    );
}

// ── Test: Concurrent Reads Allowed, Write Blocks Reads ───────────────

#[test]
fn test_concurrent_reads_write_blocks_reads() {
    let shared_locks = Arc::new(FileLockManager::new());
    let mut composer = Composer::default().with_file_locks(shared_locks.clone());

    let file = "data.txt";

    // Two readers can access concurrently
    let reader_a = composer.register_agent(
        SubAgentConfigBuilder::new("reader-a", "read data")
            .read_files(vec![file.into()])
            .build(),
    );
    let reader_b = composer.register_agent(
        SubAgentConfigBuilder::new("reader-b", "also read data")
            .read_files(vec![file.into()])
            .build(),
    );

    assert!(
        composer.start_agent(reader_a).is_ok(),
        "Reader A should start"
    );
    assert!(
        composer.start_agent(reader_b).is_ok(),
        "Reader B should start (concurrent read)"
    );

    // Now a writer comes along
    let writer = composer.register_agent(
        SubAgentConfigBuilder::new("writer", "write data")
            .write_files(vec![file.into()])
            .build(),
    );

    // Writer gets queued because readers hold read locks
    let result = composer.start_agent(writer);
    assert!(
        result.is_err(),
        "Writer should be queued while readers hold locks"
    );

    // After readers finish, writer should be able to proceed
    composer.complete_agent(
        reader_a,
        make_result_with_id(reader_a, "reader-a", true, vec![], "read"),
    );
    composer.complete_agent(
        reader_b,
        make_result_with_id(reader_b, "reader-b", true, vec![], "read"),
    );
}

// ── Test: Write Lock Queue Ordering ──────────────────────────────────

#[test]
fn test_write_lock_queue_fifo() {
    let locks = Arc::new(FileLockManager::new());
    let mut composer = Composer::default().with_file_locks(locks.clone());
    let file = "critical.rs";

    // First writer gets the lock
    let w1 = composer.register_agent(
        SubAgentConfigBuilder::new("w1", "write 1")
            .write_files(vec![file.into()])
            .build(),
    );
    assert!(composer.start_agent(w1).is_ok());
    assert_eq!(locks.queue_length(file), 0);

    // Second writer queued
    let w2 = composer.register_agent(
        SubAgentConfigBuilder::new("w2", "write 2")
            .write_files(vec![file.into()])
            .build(),
    );
    let _ = composer.start_agent(w2); // queued
    assert_eq!(locks.queue_length(file), 1);

    // Third writer queued
    let w3 = composer.register_agent(
        SubAgentConfigBuilder::new("w3", "write 3")
            .write_files(vec![file.into()])
            .build(),
    );
    let _ = composer.start_agent(w3); // queued
    assert_eq!(locks.queue_length(file), 2);

    // Release w1 (FIFO: w2 should get lock next)
    composer.complete_agent(
        w1,
        make_result_with_id(w1, "w1", true, vec![file], "w1 done"),
    );
    assert!(
        locks.has_write_lock(file),
        "w2 or w3 should now hold the lock"
    );
    assert_eq!(locks.queue_length(file), 1, "Queue should be down to 1");
}

// ── Test: Cancellation Releases Locks ────────────────────────────────

#[test]
fn test_cancellation_releases_locks() {
    let mut composer = Composer::default();
    let file = "important.rs";

    let agent = composer.register_agent(
        SubAgentConfigBuilder::new("worker", "do work")
            .write_files(vec![file.into()])
            .build(),
    );

    composer.start_agent(agent).unwrap();
    assert!(composer.file_locks().has_write_lock(file));

    // Cancel the agent
    composer.cancel_agent(agent, "user interrupted");

    let (_a, lc) = composer.get_agent(&agent).unwrap();
    assert_eq!(lc.phase, AgentPhase::Cancelled);
    assert!(
        !composer.file_locks().has_write_lock(file),
        "Lock should be released on cancel"
    );
}

// ── Test: Conflicting Edits Detected ─────────────────────────────────

#[test]
fn test_conflicting_edits_detected() {
    let mut composer = Composer::default();

    let id_a = composer.register_agent(
        SubAgentConfigBuilder::new("a", "edit shared")
            .write_files(vec!["shared.rs".into()])
            .build(),
    );
    let id_b = composer.register_agent(
        SubAgentConfigBuilder::new("b", "also edit shared")
            .write_files(vec!["shared.rs".into()])
            .build(),
    );

    // Start A, B gets queued
    composer.start_agent(id_a).unwrap();
    let _ = composer.start_agent(id_b); // queued — returns Err

    // Complete A first. The FileLockManager auto-grants the lock to B.
    composer.complete_agent(
        id_a,
        make_result_with_id(id_a, "a", true, vec!["shared.rs"], "A's changes"),
    );

    // B now holds the write lock (auto-granted). Start it.
    // We need to transition B directly since the lock is already held.
    if let Some((agent, lifecycle)) = composer.get_agent_mut(&id_b) {
        agent.phase = AgentPhase::Running;
        lifecycle.start();
    }
    composer.complete_agent(
        id_b,
        make_result_with_id(id_b, "b", true, vec!["shared.rs"], "B's changes"),
    );

    // Collect and merge — should detect that shared.rs was modified by both
    let results = composer.collect_results();
    let merged = composer.merge_results(results);

    // Both agents modified shared.rs — conflict detected
    let has_conflict = merged.conflicts.iter().any(|c| c.file == "shared.rs");
    assert!(has_conflict, "Should detect shared.rs conflict");
}

// ── Test: Pause and Resume ───────────────────────────────────────────

#[test]
fn test_pause_and_resume_all() {
    let mut composer = Composer::default();

    let id_a = composer.register_agent(
        SubAgentConfigBuilder::new("a", "task a")
            .write_files(vec!["a.txt".into()])
            .build(),
    );
    let id_b = composer.register_agent(
        SubAgentConfigBuilder::new("b", "task b")
            .write_files(vec!["b.txt".into()])
            .build(),
    );

    composer.start_agent(id_a).unwrap();
    composer.start_agent(id_b).unwrap();

    // Pause all
    composer.pause_all();
    assert_eq!(
        composer.get_agent(&id_a).unwrap().0.phase,
        AgentPhase::Blocked
    );
    assert_eq!(
        composer.get_agent(&id_b).unwrap().0.phase,
        AgentPhase::Blocked
    );

    // Resume all
    let resumed = composer.resume_all().unwrap();
    assert_eq!(resumed, 2);
    assert_eq!(
        composer.get_agent(&id_a).unwrap().0.phase,
        AgentPhase::Running
    );
    assert_eq!(
        composer.get_agent(&id_b).unwrap().0.phase,
        AgentPhase::Running
    );
}

// ── Test: Reprioritize Mid-Execution ─────────────────────────────────

#[test]
fn test_reprioritize_mid_execution() {
    let mut composer = Composer::default();

    let id = composer.register_agent(
        SubAgentConfigBuilder::new("low-prio", "do work")
            .priority(10)
            .build(),
    );

    composer.start_agent(id).unwrap();

    // Reprioritize to highest
    composer.reprioritize(id, 0).unwrap();
    assert_eq!(composer.get_agent(&id).unwrap().0.config.priority, 0);
}

// ── Test: Failed Sub-Agent Is Tracked ────────────────────────────────

#[test]
fn test_failed_sub_agent_tracking() {
    let mut composer = Composer::default();

    let id = composer.register_agent(SubAgentConfigBuilder::new("flaky", "might fail").build());

    composer.start_agent(id).unwrap();
    composer.fail_agent(id, "something broke");

    let (_agent, lifecycle) = composer.get_agent(&id).unwrap();
    assert_eq!(lifecycle.phase, AgentPhase::Failed);
    assert_eq!(lifecycle.error.as_deref(), Some("something broke"));

    // Verify it's counted as terminal
    assert!(lifecycle.phase.is_terminal());

    // Verify failure is reflected in results
    let results = composer.collect_results();
    let failed = results.iter().find(|r| r.agent_id == id).unwrap();
    assert!(!failed.success);
    assert_eq!(failed.error.as_deref(), Some("something broke"));
}

// ── Test: Persistence: Save and Restore Task Graph ───────────────────

#[tokio::test]
async fn test_persist_task_graph_survives_restart() {
    use odin_orchestrator::persistence::{OrchestrationStore, SqliteOrchestrationStore};
    use odin_orchestrator::task_graph::{TaskGraph, TaskNodeStatus};

    let store = SqliteOrchestrationStore::new_in_memory().await.unwrap();
    store.initialize().await.unwrap();

    // Create a task graph
    let mut graph = TaskGraph::new("persist-test");
    let node = odin_orchestrator::task_graph::TaskNode {
        id: Uuid::new_v4(),
        label: "n1".into(),
        goal: "do stuff".into(),
        read_files: vec!["README.md".into()],
        write_files: vec!["src/main.rs".into()],
        required_capabilities: vec!["filesystem".into()],
        priority: 1,
        status: TaskNodeStatus::Running,
        result: None,
        agent_id: Some(Uuid::new_v4()),
    };
    graph.add_node(node);

    // Save
    store.save_task_graph(&graph).await.unwrap();

    // Simulate restart — load from DB
    let loaded = store.load_task_graph("persist-test").await.unwrap();
    assert_eq!(loaded.root_goal, "persist-test");
    assert_eq!(loaded.nodes.len(), 1);
    let loaded_node = loaded.nodes.values().next().unwrap();
    assert_eq!(loaded_node.label, "n1");
    assert_eq!(loaded_node.read_files, vec!["README.md"]);
    assert_eq!(loaded_node.write_files, vec!["src/main.rs"]);
}

// ── Test: Persistence: Save and Restore Agent Lifecycle ──────────────

#[tokio::test]
async fn test_persist_agent_lifecycle_survives_restart() {
    use odin_orchestrator::lifecycle::{AgentLifecycle, AgentPhase};
    use odin_orchestrator::persistence::{OrchestrationStore, SqliteOrchestrationStore};

    let store = SqliteOrchestrationStore::new_in_memory().await.unwrap();
    store.initialize().await.unwrap();

    let agent_id = Uuid::new_v4();
    let mut lifecycle = AgentLifecycle::new(agent_id);
    lifecycle.start();
    lifecycle.wait_for_lock("src/main.rs");
    lifecycle.lock_acquired("src/main.rs");
    lifecycle.lock_released("src/main.rs");
    lifecycle.start(); // resume
    lifecycle.complete();

    // Save
    let graph_root_id = Uuid::new_v4().to_string();
    store
        .save_agent_lifecycle(&graph_root_id, &lifecycle)
        .await
        .unwrap();

    // Simulate restart
    let loaded = store.load_agent_lifecycle(agent_id).await.unwrap();
    assert_eq!(loaded.agent_id, agent_id);
    assert_eq!(loaded.phase, AgentPhase::Done);
    assert!(!loaded.history.is_empty());
    // Locks were released before completion
    assert!(loaded.held_locks.is_empty());
}

// ── Test: Final Answer Composition ───────────────────────────────────

#[test]
fn test_final_answer_composition_from_multiple_agents() {
    let mut composer = Composer::default();

    // Simulate 3 sub-agents completing with different results
    let id_a = composer.register_agent(
        SubAgentConfigBuilder::new("fix-cli", "fix CLI parsing")
            .write_files(vec!["src/cli.rs".into()])
            .build(),
    );
    let id_b = composer.register_agent(
        SubAgentConfigBuilder::new("add-docs", "improve docs")
            .write_files(vec!["README.md".into()])
            .build(),
    );
    let id_c = composer.register_agent(
        SubAgentConfigBuilder::new("add-tests", "add scheduler tests")
            .write_files(vec!["tests/scheduler.rs".into()])
            .build(),
    );

    // Start and complete all (no file conflicts, all parallel)
    for id in [id_a, id_b, id_c] {
        composer.start_agent(id).unwrap();
    }

    composer.complete_agent(
        id_a,
        make_result_with_id(
            id_a,
            "fix-cli",
            true,
            vec!["src/cli.rs"],
            "Fixed CLI parsing bug in arg parser",
        ),
    );
    composer.complete_agent(
        id_b,
        make_result_with_id(
            id_b,
            "add-docs",
            true,
            vec!["README.md"],
            "Added installation and config docs",
        ),
    );
    composer.complete_agent(
        id_c,
        make_result_with_id(
            id_c,
            "add-tests",
            true,
            vec!["tests/scheduler.rs"],
            "Added 5 scheduler integration tests",
        ),
    );

    // Collect and compose final answer
    let results = composer.collect_results();
    assert_eq!(results.len(), 3, "All 3 agents should be collected");

    let merged = composer.merge_results(results);
    assert!(merged.success, "Merge should succeed");

    // Verify the composed answer references all work
    assert!(
        merged.summary.contains("fix-cli"),
        "Summary should mention CLI fix"
    );
    assert!(
        merged.summary.contains("add-docs"),
        "Summary should mention docs"
    );
    assert!(
        merged.summary.contains("add-tests"),
        "Summary should mention tests"
    );

    // All 3 files should be in the modified list
    assert_eq!(merged.modified_files.len(), 3);
    assert!(merged.modified_files.contains(&"src/cli.rs".to_string()));
    assert!(merged.modified_files.contains(&"README.md".to_string()));
    assert!(
        merged
            .modified_files
            .contains(&"tests/scheduler.rs".to_string())
    );

    // No conflicts (different files)
    assert!(merged.conflicts.is_empty());
    // When no conflicts, no user input needed
    assert!(merged.conflicts.is_empty());
}

// ── Test: Orchestration with One Failing Agent ───────────────────────

#[test]
fn test_orchestration_with_one_failure() {
    let mut composer = Composer::default();

    let good = composer.register_agent(SubAgentConfigBuilder::new("good", "succeeds").build());
    let bad = composer.register_agent(SubAgentConfigBuilder::new("bad", "fails").build());

    composer.start_agent(good).unwrap();
    composer.start_agent(bad).unwrap();

    composer.complete_agent(
        good,
        make_result_with_id(good, "good", true, vec!["good.txt"], "Good done"),
    );
    composer.fail_agent(bad, "something went wrong");

    let results = composer.collect_results();
    let merged = composer.merge_results(results);

    assert!(!merged.success, "Overall should fail when any agent fails");
    assert!(merged.summary.contains("❌"), "Summary should show failure");
    assert!(
        merged.summary.contains("✅"),
        "Summary should also show success"
    );
}
