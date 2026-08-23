//! E2E test: scheduler persistence across restarts and enable/disable toggling.
//!
//! Tests that:
//! 1. Jobs survive a "restart" (new SqliteSchedulerStore reading the same DB file)
//! 2. Enabling/disabling a job persists across restarts

use chrono::Utc;
use odin_scheduler::store::{PersistedJob, SchedulerStore, SqliteSchedulerStore};
use uuid::Uuid;

/// Helper to create a PersistedJob for testing.
fn make_persisted_job(name: &str, cron_expr: &str) -> PersistedJob {
    PersistedJob {
        id: Uuid::new_v4(),
        name: name.to_string(),
        cron_expr: cron_expr.to_string(),
        task_goal: Some(format!("Run {}", name)),
        max_iterations: 100,
        agent_selector: Some("scheduler-agent".into()),
        enabled: true,
        last_run: None,
        next_run: Some(Utc::now() + chrono::TimeDelta::hours(1)),
        run_count: 0,
        created_at: Utc::now(),
    }
}

#[tokio::test]
async fn test_scheduler_persistence_survives_restart() {
    let dir = tempfile::tempdir().expect("tempdir must succeed");
    let db_path = dir.path().join("scheduler.db");
    let db_path_str = db_path.to_str().expect("valid UTF-8 path");

    // ── First session: add jobs ──────────────────────────────────────
    let store = SqliteSchedulerStore::new(db_path_str).expect("first store creation");
    let job1 = make_persisted_job("alpha", "0 * * * *");
    let job2 = make_persisted_job("beta", "*/5 * * * *");
    store.save_job(&job1).await.expect("save job1");
    store.save_job(&job2).await.expect("save job2");

    let jobs = store.load_all_jobs().await.expect("load all after save");
    assert_eq!(jobs.len(), 2, "should have 2 jobs in first session");

    // ── "Restart": new store, same DB file ───────────────────────────
    let store2 = SqliteSchedulerStore::new(db_path_str).expect("second store creation");
    let jobs2 = store2
        .load_all_jobs()
        .await
        .expect("load all after restart");

    assert_eq!(jobs2.len(), 2, "should still have 2 jobs after restart");
    let names: Vec<&str> = jobs2.iter().map(|j| j.name.as_str()).collect();
    assert!(names.contains(&"alpha"), "alpha should survive restart");
    assert!(names.contains(&"beta"), "beta should survive restart");

    // Verify full job details are intact
    let alpha = jobs2.iter().find(|j| j.name == "alpha").unwrap();
    assert_eq!(alpha.cron_expr, "0 * * * *");
    assert_eq!(alpha.task_goal.as_deref(), Some("Run alpha"));
    assert!(alpha.enabled, "job should be enabled");
}

#[tokio::test]
async fn test_scheduler_enable_disable_persists_across_restart() {
    let dir = tempfile::tempdir().expect("tempdir must succeed");
    let db_path = dir.path().join("scheduler.db");
    let db_path_str = db_path.to_str().expect("valid UTF-8 path");

    // ── First session: create a job ──────────────────────────────────
    let store = SqliteSchedulerStore::new(db_path_str).expect("store creation");
    let mut job = make_persisted_job("toggle-job", "0 9 * * 1-5");
    store.save_job(&job).await.expect("save job");
    assert!(job.enabled, "initially enabled");

    // Disable the job
    job.enabled = false;
    store
        .update_job_state(&job.id, false, job.last_run, job.next_run, job.run_count)
        .await
        .expect("disable job");

    // ── Restart and verify disabled state persists ───────────────────
    let store2 = SqliteSchedulerStore::new(db_path_str).expect("store after restart");
    let jobs = store2.load_all_jobs().await.expect("load jobs");
    assert_eq!(jobs.len(), 1, "should have 1 job");
    assert!(!jobs[0].enabled, "job should be disabled after restart");

    // Re-enable the job
    store2
        .update_job_state(
            &job.id,
            true,
            jobs[0].last_run,
            jobs[0].next_run,
            jobs[0].run_count,
        )
        .await
        .expect("re-enable job");

    // ── Verify enabled state after another restart ───────────────────
    let store3 = SqliteSchedulerStore::new(db_path_str).expect("store after re-enable");
    let jobs = store3.load_all_jobs().await.expect("load jobs");
    assert_eq!(jobs.len(), 1, "should still have 1 job");
    assert!(
        jobs[0].enabled,
        "job should be enabled after re-enable and restart"
    );
}
