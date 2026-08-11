//! Live cross-process orchestration control integration tests.

use odin_orchestrator::ControlAuth;
use odin_orchestrator::persistence::{OrchestrationStore, SqliteOrchestrationStore};
use odin_orchestrator::task_graph::{TaskGraph, TaskGraphStatus};
use odin_orchestrator::{RunControlCommand, RunControlKind, RunControlStatus, authorize_control};

#[tokio::test]
async fn second_process_can_cancel_owned_run_by_graph_uuid() {
    let store = SqliteOrchestrationStore::new_in_memory().await.unwrap();
    store.initialize().await.unwrap();

    let mut graph = TaskGraph::new("live-control-goal");
    graph.status = TaskGraphStatus::Running;
    let graph_id = graph.id.to_string();
    store.save_task_graph(&graph).await.unwrap();

    // Second process / WS client enqueues cancel.
    let command = RunControlCommand::new(
        &graph_id,
        RunControlKind::Cancel,
        "integration-test",
        Some("stop".into()),
    );
    store.enqueue_control(&command).await.unwrap();
    store
        .update_graph_status(&graph_id, "cancelled")
        .await
        .unwrap();

    // Owner process claims and applies.
    let claimed = store.claim_pending_controls(&graph_id).await.unwrap();
    assert_eq!(claimed.len(), 1);
    assert_eq!(claimed[0].kind, RunControlKind::Cancel);
    assert_eq!(claimed[0].status, RunControlStatus::Claimed);
    store.mark_control_applied(claimed[0].id).await.unwrap();

    let listed = store.list_controls(&graph_id, 10).await.unwrap();
    assert_eq!(listed[0].status, RunControlStatus::Applied);
    let loaded = store.load_task_graph(&graph_id).await.unwrap();
    assert_eq!(loaded.status, TaskGraphStatus::Cancelled);

    // Disconnected owner: second claim is empty (no double-apply).
    assert!(
        store
            .claim_pending_controls(&graph_id)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn control_authorization_rejects_bad_token() {
    assert_eq!(
        authorize_control(Some("secret"), Some("wrong")),
        ControlAuth::Denied("missing or invalid control token")
    );
    assert_eq!(
        authorize_control(Some("secret"), Some("secret")),
        ControlAuth::Allowed
    );
    assert_eq!(
        authorize_control(None, None),
        ControlAuth::Denied("missing or invalid control token")
    );
}

#[tokio::test]
async fn concurrent_owners_claim_each_command_once() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("orchestration.db");
    let first = SqliteOrchestrationStore::new(&path).await.unwrap();
    let second = SqliteOrchestrationStore::new(&path).await.unwrap();
    first.initialize().await.unwrap();

    let graph_id = uuid::Uuid::new_v4().to_string();
    first
        .enqueue_control(&RunControlCommand::new(
            &graph_id,
            RunControlKind::Pause,
            "integration-test",
            None,
        ))
        .await
        .unwrap();

    let (first_claim, second_claim) = tokio::join!(
        first.claim_pending_controls(&graph_id),
        second.claim_pending_controls(&graph_id)
    );
    let claimed = first_claim.unwrap().len() + second_claim.unwrap().len();
    assert_eq!(claimed, 1, "a control command must have exactly one owner");
}
