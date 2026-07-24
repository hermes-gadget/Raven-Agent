use async_trait::async_trait;
use chrono::Utc;
use odin_core::config::SchedulerConfig;
use odin_core::error::OdinResult;
use odin_core::traits::{LoopEngine, Provider};
use odin_core::types::{
    AgentTask, ChatResponse, CompletionOptions, LoopPhase, ModelInfo, StateSummary, TaskResult,
    ToolSchema,
};
use odin_runtime::{Agent, Runtime};
use odin_scheduler::{
    JobRunStatus, Scheduler, SchedulerJobConfig, SchedulerStore, SqliteSchedulerStore,
};
use std::sync::Arc;
use tokio::time::{Duration, sleep};

struct SuccessfulEngine;

#[async_trait]
impl LoopEngine for SuccessfulEngine {
    async fn execute_task(&self, task: &AgentTask) -> OdinResult<TaskResult> {
        Ok(TaskResult {
            task_id: task.id,
            success: true,
            summary: "completed by scheduler host".into(),
            iterations: 1,
            tool_calls: 0,
            duration_ms: 1,
            sub_tasks: vec![],
            confidence: 1.0,
            error: None,
        })
    }

    async fn execute_phase(
        &self,
        _phase: LoopPhase,
        _state: &mut odin_core::traits::LoopState,
    ) -> OdinResult<odin_core::traits::PhaseResult> {
        unreachable!("the test engine executes complete tasks")
    }

    fn state_summary(&self) -> StateSummary {
        StateSummary {
            goal: String::new(),
            current_phase: LoopPhase::Plan,
            completed_steps: vec![],
            pending_steps: vec![],
            last_action: None,
            last_result: None,
            errors: vec![],
            confidence: 1.0,
            token_usage: Default::default(),
        }
    }

    fn confidence(&self) -> odin_core::types::ConfidenceScore {
        odin_core::types::ConfidenceScore::new(1.0)
    }
}

struct UnusedProvider;

#[async_trait]
impl Provider for UnusedProvider {
    fn name(&self) -> &str {
        "scheduler-test"
    }

    async fn list_models(&self) -> OdinResult<Vec<ModelInfo>> {
        Ok(vec![])
    }

    async fn chat(
        &self,
        _model: &str,
        _messages: &[odin_core::types::Message],
        _tools: &[ToolSchema],
        _options: &CompletionOptions,
    ) -> OdinResult<ChatResponse> {
        unreachable!("the test loop does not call its provider")
    }

    async fn chat_stream(
        &self,
        _model: &str,
        _messages: &[odin_core::types::Message],
        _tools: &[ToolSchema],
        _options: &CompletionOptions,
    ) -> OdinResult<Box<dyn odin_core::traits::ChatStream>> {
        unreachable!("the test loop does not stream")
    }

    async fn health_check(&self) -> OdinResult<bool> {
        Ok(true)
    }
}

fn host_config(enabled: bool) -> SchedulerConfig {
    SchedulerConfig {
        enabled,
        check_interval_secs: 1,
        max_concurrent: 2,
        db_path: None,
    }
}

#[tokio::test]
async fn persisted_due_job_executes_after_host_restart() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("scheduler.db");
    let path = path.to_str().unwrap();

    let initial_store = Arc::new(SqliteSchedulerStore::new(path).unwrap());
    let initial = Scheduler::new(host_config(false)).with_store(initial_store.clone());
    let job_id = initial
        .add_job_with_config(
            "restart-job",
            "* * * * *",
            SchedulerJobConfig::new("run after restart"),
        )
        .await
        .unwrap();
    initial_store
        .update_job_state(
            &job_id,
            true,
            None,
            Some(Utc::now() - chrono::TimeDelta::minutes(5)),
            0,
        )
        .await
        .unwrap();
    drop(initial);
    drop(initial_store);

    let provider: Arc<dyn Provider> = Arc::new(UnusedProvider);
    let runtime = Arc::new(Runtime::new());
    runtime.register_agent(Agent::new(
        "scheduler-host-test",
        Arc::new(SuccessfulEngine),
        provider,
        vec![],
    ));
    let restarted = Scheduler::new(host_config(true))
        .with_store(Arc::new(SqliteSchedulerStore::new(path).unwrap()))
        .with_runtime(runtime);

    restarted.start().await.unwrap();
    sleep(Duration::from_millis(1_250)).await;
    restarted.stop().await.unwrap();

    let runs = restarted.recent_runs(10).await.unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].job_id, job_id);
    assert_eq!(runs[0].status, JobRunStatus::Succeeded);
    assert!(runs[0].finished_at.is_some());
}

#[tokio::test]
async fn disabled_host_loads_nothing_and_does_not_run() {
    let store = Arc::new(SqliteSchedulerStore::in_memory().unwrap());
    let scheduler = Scheduler::new(host_config(false)).with_store(store);
    scheduler.start().await.unwrap();
    sleep(Duration::from_millis(50)).await;

    assert!(!scheduler.is_running().await);
    assert_eq!(scheduler.job_count().await, 0);
    assert!(scheduler.recent_runs(10).await.unwrap().is_empty());
}
