//! Scheduler — cron-like job scheduling for Raven Agent.
//!
//! The [`Scheduler`] manages a collection of [`Job`]s, checks for
//! due jobs on a configurable tick interval, and spawns tasks via
//! Tokio. Supports optional persistence via [`SchedulerStore`] and
//! optional runtime-driven task execution via [`Runtime`].

use crate::job::{Job, JobId, JobTask, Schedule};
use crate::store::{JobRun, JobRunStatus, PersistedJob, SchedulerJobConfig, SchedulerStore};
use chrono::Utc;
use odin_core::config::SchedulerConfig;
use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::AuditLogger;
use odin_core::types::{AuditEntry, AuditEventType, AuditResult};
use odin_runtime::Runtime;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify, RwLock, Semaphore};
use tokio::time::{Duration, interval};
use tracing::{debug, info, trace, warn};
use uuid::Uuid;

async fn reserve_due_job(
    jobs: &Arc<RwLock<HashMap<JobId, Job>>>,
    job_id: JobId,
    now: chrono::DateTime<Utc>,
) -> Option<(Job, Uuid)> {
    let mut jobs = jobs.write().await;
    let job = jobs.get_mut(&job_id)?;
    if !job.is_due(&now) || (job.max_concurrent > 0 && job.running_count >= job.max_concurrent) {
        return None;
    }

    let task_id = Uuid::new_v4();
    job.mark_run(task_id);
    job.running_count += 1;
    Some((job.clone(), task_id))
}

struct RunningCountGuard {
    jobs: Arc<RwLock<HashMap<JobId, Job>>>,
    job_id: JobId,
    released: bool,
}

impl RunningCountGuard {
    fn new(jobs: Arc<RwLock<HashMap<JobId, Job>>>, job_id: JobId) -> Self {
        Self {
            jobs,
            job_id,
            released: false,
        }
    }

    async fn release(&mut self) {
        if self.released {
            return;
        }
        let mut jobs = self.jobs.write().await;
        if let Some(job) = jobs.get_mut(&self.job_id) {
            job.running_count = job.running_count.saturating_sub(1);
        }
        self.released = true;
    }
}

impl Drop for RunningCountGuard {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        let jobs = self.jobs.clone();
        let job_id = self.job_id;
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let mut jobs = jobs.write().await;
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.running_count = job.running_count.saturating_sub(1);
                }
            });
        }
    }
}

/// A cron-like job scheduler with optional persistence and runtime execution.
///
/// Manages a set of [`Job`]s, periodically checks for due jobs,
/// and spawns their tasks asynchronously. When a [`SchedulerStore`]
/// is provided, all job mutations are persisted automatically.
///
/// When a [`Runtime`] is provided, jobs with a `task_goal` will
/// submit [`AgentTask`]s to the runtime instead of running generic
/// closures.
pub struct Scheduler {
    /// The scheduled jobs, keyed by ID.
    jobs: Arc<RwLock<HashMap<JobId, Job>>>,
    /// Configuration for the scheduler.
    config: SchedulerConfig,
    /// Whether the scheduler loop is running.
    running: Arc<RwLock<bool>>,
    /// Handle to the spawned tick loop.
    tick_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
    /// Optional persistent store for job definitions and state.
    store: Option<Arc<dyn SchedulerStore>>,
    /// Optional runtime for submitting agent tasks.
    runtime: Option<Arc<Runtime>>,
    /// Optional structured audit sink for scheduler outcomes.
    audit_logger: Option<Arc<dyn AuditLogger>>,
    /// In-flight executions, drained during graceful shutdown.
    execution_handles: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>>,
    /// Wakes the tick loop immediately during shutdown.
    shutdown: Arc<Notify>,
}

impl Scheduler {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            config,
            running: Arc::new(RwLock::new(false)),
            tick_handle: Arc::new(RwLock::new(None)),
            store: None,
            runtime: None,
            audit_logger: None,
            execution_handles: Arc::new(Mutex::new(Vec::new())),
            shutdown: Arc::new(Notify::new()),
        }
    }

    /// Create a new scheduler with default configuration.
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self::new(SchedulerConfig::default())
    }

    /// Attach a persistent [`SchedulerStore`].
    ///
    /// When set, all job mutations are persisted automatically and
    /// persisted jobs are loaded into memory on [`start`](Self::start).
    pub fn with_store(mut self, store: Arc<dyn SchedulerStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Attach a [`Runtime`] for submitting agent tasks.
    ///
    /// When set, jobs that have a `task_goal` will create `AgentTask`s
    /// and submit them via the runtime instead of running generic closures.
    pub fn with_runtime(mut self, runtime: Arc<Runtime>) -> Self {
        self.runtime = Some(runtime);
        self
    }

    /// Attach an audit logger for durable scheduler start/outcome events.
    pub fn with_audit_logger(mut self, audit_logger: Arc<dyn AuditLogger>) -> Self {
        self.audit_logger = Some(audit_logger);
        self
    }

    /// Load persisted definitions without starting the host loop.
    pub async fn load_persisted(&self) -> OdinResult<usize> {
        let Some(store) = &self.store else {
            return Ok(0);
        };
        let persisted_jobs = store.load_all_jobs().await?;
        let count = persisted_jobs.len();
        let mut jobs = self.jobs.write().await;
        for persisted in persisted_jobs {
            jobs.insert(persisted.id, persisted.into_job());
        }
        Ok(count)
    }

    /// Return recent durable execution outcomes.
    pub async fn recent_runs(&self, limit: usize) -> OdinResult<Vec<JobRun>> {
        match &self.store {
            Some(store) => store.recent_runs(limit).await,
            None => Ok(Vec::new()),
        }
    }

    /// Add a job to the scheduler using a generic closure task.
    ///
    /// Returns the assigned job ID. If a store is configured, the job
    /// is persisted immediately.
    pub async fn add_job(&self, name: &str, schedule: &str, task: JobTask) -> OdinResult<JobId> {
        let sched = Schedule::parse(schedule)
            .map_err(|e| OdinError::Config(format!("Invalid schedule '{}': {}", schedule, e)))?;

        let job = Job::new(name, sched.clone(), task);
        let id = job.id;

        // Persist to store if configured
        if let Some(ref store) = self.store {
            let persisted = PersistedJob::from_job(&job);
            store.save_job(&persisted).await?;
        }

        self.jobs.write().await.insert(id, job);
        info!(job_id = %id, name = %name, schedule = %sched.expression, "Job added to scheduler");
        Ok(id)
    }

    /// Add a job with a [`SchedulerJobConfig`] instead of a closure.
    ///
    /// The job will be executed via the configured runtime. Attempting to run
    /// it without a runtime and registered agent returns an error.
    pub async fn add_job_with_config(
        &self,
        name: &str,
        schedule: &str,
        config: SchedulerJobConfig,
    ) -> OdinResult<JobId> {
        let sched = Schedule::parse(schedule)
            .map_err(|e| OdinError::Config(format!("Invalid schedule '{}': {}", schedule, e)))?;

        let mut job = Job::new(name, sched.clone(), crate::job::noop_task());
        job.task_goal = Some(config.task_goal);
        job.max_iterations = config.max_iterations;
        let id = job.id;

        // Persist to store if configured
        if let Some(ref store) = self.store {
            let persisted = PersistedJob::from_job(&job);
            store.save_job(&persisted).await?;
        }

        self.jobs.write().await.insert(id, job);
        info!(job_id = %id, name = %name, schedule = %sched.expression, "Job added to scheduler (with config)");
        Ok(id)
    }

    /// Remove a job from the scheduler by ID.
    pub async fn remove_job(&self, job_id: JobId) -> OdinResult<bool> {
        // Delete from store first
        if let Some(ref store) = self.store {
            store.delete_job(&job_id).await?;
        }

        let existed = self.jobs.write().await.remove(&job_id).is_some();
        if existed {
            info!(job_id = %job_id, "Job removed from scheduler");
        } else {
            warn!(job_id = %job_id, "Job not found for removal");
        }
        Ok(existed)
    }

    /// List all jobs currently registered.
    pub async fn list_jobs(&self) -> Vec<Job> {
        self.jobs.read().await.values().cloned().collect()
    }

    /// Get a specific job by ID.
    pub async fn get_job(&self, job_id: JobId) -> Option<Job> {
        self.jobs.read().await.get(&job_id).cloned()
    }

    /// Enable or disable a job.
    pub async fn set_job_enabled(&self, job_id: JobId, enabled: bool) -> OdinResult<bool> {
        let mut jobs = self.jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            job.enabled = enabled;

            // Persist to store
            if let Some(ref store) = self.store {
                let id = job.id;
                let run_count = job.run_count;
                let last_run = job.last_run;
                let next_run = job.next_run;
                // Drop lock before async call
                drop(jobs);
                store
                    .update_job_state(&id, enabled, last_run, next_run, run_count)
                    .await?;
                info!(job_id = %job_id, enabled = enabled, "Job enabled state changed");
                return Ok(true);
            }

            info!(job_id = %job_id, enabled = enabled, "Job enabled state changed");
            Ok(true)
        } else {
            warn!(job_id = %job_id, "Job not found for enable/disable");
            Ok(false)
        }
    }

    /// Run all pending jobs synchronously (for manual tick).
    ///
    /// Returns the number of jobs that were run.
    pub async fn run_pending(&self) -> OdinResult<usize> {
        let now = Utc::now();
        let due_ids: Vec<JobId> = {
            let jobs = self.jobs.read().await;
            jobs.iter()
                .filter(|(_, job)| job.is_due(&now))
                .map(|(id, _)| *id)
                .collect()
        };

        let mut count = 0;
        for job_id in due_ids {
            let job_snapshot = match self.jobs.read().await.get(&job_id).cloned() {
                Some(job) => job,
                None => continue,
            };
            // Resolve a real runtime agent before recording the run. A configured
            // runtime with no agent is an actionable error, not a successful no-op.
            if job_snapshot.task_goal.is_some() && self.runtime.is_none() {
                return Err(OdinError::Config(format!(
                    "Scheduled job '{}' requires a runtime, but none is configured",
                    job_snapshot.name
                )));
            }
            let runtime_agent =
                if let (Some(runtime), Some(_)) = (&self.runtime, &job_snapshot.task_goal) {
                    runtime.list_agents().first().map(|agent| agent.id)
                } else {
                    None
                };
            if self.runtime.is_some() && job_snapshot.task_goal.is_some() && runtime_agent.is_none()
            {
                return Err(OdinError::Internal(format!(
                    "Scheduled job '{}' cannot run because the runtime has no registered agent",
                    job_snapshot.name
                )));
            }

            // Recheck due state and reserve the execution under one write lock.
            let Some((job, task_id)) = reserve_due_job(&self.jobs, job_id, Utc::now()).await else {
                trace!(job_id = %job_id, "Skipping job: no longer due or at concurrency limit");
                continue;
            };
            count += 1;
            info!(
                job_id = %job_id,
                name = %job.name,
                task_id = %task_id,
                "Running scheduled job"
            );

            // Persist updated state
            if let Some(ref store) = self.store {
                let _ = store
                    .update_job_state(
                        &job_id,
                        job.enabled,
                        job.last_run,
                        job.next_run,
                        job.run_count,
                    )
                    .await;
            }

            if let Some(agent_id) = runtime_agent {
                // Runtime-driven execution path
                let runtime = self.runtime.clone().unwrap();
                let task_goal = job.task_goal.clone().unwrap();
                let max_iterations = job.max_iterations;
                let mut guard = RunningCountGuard::new(self.jobs.clone(), job_id);

                tokio::spawn(async move {
                    debug!(task_id = %task_id, "Job task starting via runtime");

                    // Create a minimal runtime task
                    let agent_task = odin_core::types::AgentTask {
                        id: task_id,
                        goal: task_goal,
                        context: None,
                        sub_tasks: vec![],
                        success_criteria: vec![],
                        max_iterations,
                        created_at: chrono::Utc::now(),
                    };

                    if let Err(error) = runtime.submit_task(&agent_id, &agent_task, None).await {
                        warn!(task_id = %task_id, "Scheduled runtime task failed: {error}");
                    }
                    guard.release().await;

                    debug!(task_id = %task_id, "Job task completed (runtime)");
                });
            } else {
                // Closure-based execution path (original behaviour)
                let task_fn = job.task.clone();
                let mut guard = RunningCountGuard::new(self.jobs.clone(), job_id);

                tokio::spawn(async move {
                    debug!(task_id = %task_id, "Job task started");
                    let start = std::time::Instant::now();
                    (task_fn)().await;
                    let duration = start.elapsed();
                    debug!(task_id = %task_id, duration_ms = duration.as_millis() as u64, "Job task completed");
                    guard.release().await;
                });
            }
        }

        Ok(count)
    }

    /// Start the scheduler loop that ticks on the configured interval.
    ///
    /// If a [`SchedulerStore`] is configured, all persisted jobs are
    /// loaded into memory before the loop starts.
    pub async fn start(&self) -> OdinResult<()> {
        if !self.config.enabled {
            info!("Scheduler is disabled; host loop will not start");
            return Ok(());
        }
        if self.config.check_interval_secs == 0 {
            return Err(OdinError::Config(
                "scheduler.check_interval_secs must be greater than zero".into(),
            ));
        }
        let mut running = self.running.write().await;
        if *running {
            warn!("Scheduler is already running");
            return Ok(());
        }
        *running = true;
        drop(running);

        // Load persisted jobs from store
        let loaded = match self.load_persisted().await {
            Ok(loaded) => loaded,
            Err(error) => {
                *self.running.write().await = false;
                return Err(error);
            }
        };
        info!(count = loaded, "Loaded persisted jobs into scheduler");

        let interval_secs = self.config.check_interval_secs;
        let jobs = self.jobs.clone();
        let running_flag = self.running.clone();
        let store = self.store.clone();
        let runtime = self.runtime.clone();
        let audit_logger = self.audit_logger.clone();
        let execution_handles = self.execution_handles.clone();
        let shutdown = self.shutdown.clone();
        let permits = Arc::new(Semaphore::new(self.config.max_concurrent.max(1) as usize));

        info!(
            check_interval_secs = interval_secs,
            "Scheduler loop starting"
        );

        let handle = tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(interval_secs));
            ticker.tick().await; // Skip the first immediate tick

            loop {
                tokio::select! {
                    _ = ticker.tick() => {}
                    _ = shutdown.notified() => {
                        info!("Scheduler loop received shutdown signal");
                        break;
                    }
                }

                let is_running = *running_flag.read().await;
                if !is_running {
                    info!("Scheduler loop stopped");
                    break;
                }

                let now = Utc::now();
                let due_ids: Vec<JobId> = {
                    let jobs_guard = jobs.read().await;
                    jobs_guard
                        .iter()
                        .filter(|(_, job)| job.is_due(&now))
                        .map(|(id, _)| *id)
                        .collect()
                };

                for job_id in due_ids {
                    if !*running_flag.read().await {
                        break;
                    }
                    let permit = match permits.clone().try_acquire_owned() {
                        Ok(permit) => permit,
                        Err(_) => {
                            trace!(job_id = %job_id, "Deferring job: host concurrency limit reached");
                            continue;
                        }
                    };
                    // Read a snapshot for runtime validation. Eligibility and
                    // reservation are rechecked atomically below.
                    let job_snapshot = match jobs.read().await.get(&job_id).cloned() {
                        Some(job) => job,
                        None => continue,
                    };
                    let task_goal = job_snapshot.task_goal.clone();

                    let runtime_dispatch = match (runtime.as_ref(), task_goal.as_ref()) {
                        (Some(runtime), Some(goal)) => match runtime.list_agents().first() {
                            Some(agent) => Some((runtime.clone(), agent.id, goal.clone())),
                            None => {
                                warn!(job_id = %job_id, "Scheduled job skipped: runtime has no registered agent");
                                continue;
                            }
                        },
                        (None, Some(_)) => {
                            warn!(job_id = %job_id, "Scheduled job skipped: no runtime is configured");
                            continue;
                        }
                        _ => None,
                    };

                    let Some((job, task_id)) = reserve_due_job(&jobs, job_id, Utc::now()).await
                    else {
                        trace!(job_id = %job_id, "Skipping job: no longer due or at concurrency limit");
                        continue;
                    };
                    let task_fn = job.task.clone();
                    let name = job.name.clone();
                    let max_iterations = job.max_iterations;
                    info!(
                        job_id = %job_id,
                        name = %name,
                        task_id = %task_id,
                        "Running scheduled job (loop)"
                    );

                    // Persist updated state
                    if let Some(ref store) = store {
                        if let Err(error) = store
                            .update_job_state(
                                &job_id,
                                job.enabled,
                                job.last_run,
                                job.next_run,
                                job.run_count,
                            )
                            .await
                        {
                            warn!(job_id = %job_id, "Failed to persist scheduler state: {error}");
                        }
                        let run = JobRun {
                            task_id,
                            job_id,
                            job_name: name.clone(),
                            started_at: Utc::now(),
                            finished_at: None,
                            status: JobRunStatus::Running,
                            error: None,
                        };
                        if let Err(error) = store.record_run_started(&run).await {
                            warn!(job_id = %job_id, "Failed to persist scheduler run: {error}");
                        }
                    }

                    if let Some(logger) = &audit_logger {
                        let _ = logger
                            .log(AuditEntry {
                                id: Uuid::new_v4(),
                                timestamp: Utc::now(),
                                agent_id: runtime_dispatch
                                    .as_ref()
                                    .map_or_else(Uuid::nil, |(_, agent_id, _)| *agent_id),
                                session_id: job_id,
                                event_type: AuditEventType::SessionStart,
                                action: "scheduler_job_started".into(),
                                details: serde_json::json!({
                                    "job_id": job_id,
                                    "job_name": name,
                                    "task_id": task_id,
                                }),
                                result: AuditResult::Pending,
                            })
                            .await;
                    }

                    let handle = if let Some((runtime, agent_id, goal)) = runtime_dispatch {
                        // Runtime-driven execution path in the loop
                        let store = store.clone();
                        let audit_logger = audit_logger.clone();
                        let completion = ExecutionCompletion {
                            store,
                            audit_logger,
                            running_guard: RunningCountGuard::new(jobs.clone(), job_id),
                            job_id,
                            job_name: name,
                            task_id,
                            agent_id,
                        };
                        tokio::spawn(async move {
                            let _permit = permit;
                            debug!(task_id = %task_id, "Job task started (loop, runtime)");

                            let agent_task = odin_core::types::AgentTask {
                                id: task_id,
                                goal,
                                context: None,
                                sub_tasks: vec![],
                                success_criteria: vec![],
                                max_iterations,
                                created_at: chrono::Utc::now(),
                            };

                            let outcome = runtime
                                .submit_task(&agent_id, &agent_task, None)
                                .await
                                .map(|result| result.success)
                                .map_err(|error| error.to_string());
                            completion.finish(outcome).await;
                        })
                    } else {
                        // Standard closure-based execution
                        let store = store.clone();
                        let audit_logger = audit_logger.clone();
                        let completion = ExecutionCompletion {
                            store,
                            audit_logger,
                            running_guard: RunningCountGuard::new(jobs.clone(), job_id),
                            job_id,
                            job_name: name,
                            task_id,
                            agent_id: Uuid::nil(),
                        };
                        tokio::spawn(async move {
                            let _permit = permit;
                            debug!(task_id = %task_id, "Job task started (loop)");
                            (task_fn)().await;
                            completion.finish(Ok(true)).await;
                        })
                    };

                    let mut handles = execution_handles.lock().await;
                    handles.retain(|existing| !existing.is_finished());
                    handles.push(handle);
                }
            }
        });

        *self.tick_handle.write().await = Some(handle);
        Ok(())
    }

    /// Stop the scheduler loop.
    pub async fn stop(&self) -> OdinResult<()> {
        let mut running = self.running.write().await;
        *running = false;
        drop(running);
        // `notify_one` retains a permit if the loop has not reached
        // `notified()` yet, so an immediate stop cannot miss the wake-up.
        self.shutdown.notify_one();

        if let Some(handle) = self.tick_handle.write().await.take() {
            info!("Scheduler stop requested, waiting for loop to finish");
            handle
                .await
                .map_err(|error| OdinError::Internal(format!("Scheduler loop failed: {error}")))?;
        }

        let handles = {
            let mut handles = self.execution_handles.lock().await;
            std::mem::take(&mut *handles)
        };
        for handle in handles {
            if let Err(error) = handle.await {
                warn!("Scheduled execution task failed to join: {error}");
            }
        }

        info!("Scheduler stopped");
        Ok(())
    }

    /// Check if the scheduler loop is running.
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }

    /// Get the total number of registered jobs.
    pub async fn job_count(&self) -> usize {
        self.jobs.read().await.len()
    }

    /// Get configuration.
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }
}

struct ExecutionCompletion {
    store: Option<Arc<dyn SchedulerStore>>,
    audit_logger: Option<Arc<dyn AuditLogger>>,
    running_guard: RunningCountGuard,
    job_id: JobId,
    job_name: String,
    task_id: Uuid,
    agent_id: Uuid,
}

impl ExecutionCompletion {
    async fn finish(mut self, outcome: Result<bool, String>) {
        let (status, error) = match outcome {
            Ok(true) => (JobRunStatus::Succeeded, None),
            Ok(false) => (
                JobRunStatus::Failed,
                Some("runtime returned an unsuccessful task result".to_string()),
            ),
            Err(error) => (JobRunStatus::Failed, Some(error)),
        };
        if let Some(store) = self.store
            && let Err(persist_error) = store
                .record_run_finished(&self.task_id, status, error.as_deref())
                .await
        {
            warn!(task_id = %self.task_id, "Failed to persist scheduler outcome: {persist_error}");
        }
        if let Some(logger) = self.audit_logger {
            let _ = logger
                .log(AuditEntry {
                    id: Uuid::new_v4(),
                    timestamp: Utc::now(),
                    agent_id: self.agent_id,
                    session_id: self.job_id,
                    event_type: AuditEventType::SessionEnd,
                    action: "scheduler_job_finished".into(),
                    details: serde_json::json!({
                        "job_id": self.job_id,
                        "job_name": self.job_name,
                        "task_id": self.task_id,
                        "error": error,
                    }),
                    result: if status == JobRunStatus::Succeeded {
                        AuditResult::Success
                    } else {
                        AuditResult::Failure
                    },
                })
                .await;
        }
        self.running_guard.release().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::SqliteSchedulerStore;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn enabled_config() -> SchedulerConfig {
        SchedulerConfig {
            enabled: true,
            check_interval_secs: 1,
            max_concurrent: 2,
            db_path: None,
        }
    }

    #[tokio::test]
    async fn test_add_and_list_jobs() {
        let sched = Scheduler::default();
        let task: JobTask = Arc::new(|| Box::pin(async {}));

        let id = sched.add_job("test", "* * * * *", task).await.unwrap();
        let jobs = sched.list_jobs().await;
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].id, id);
        assert_eq!(jobs[0].name, "test");
    }

    #[tokio::test]
    async fn test_remove_job() {
        let sched = Scheduler::default();
        let task: JobTask = Arc::new(|| Box::pin(async {}));

        let id = sched.add_job("test", "* * * * *", task).await.unwrap();
        assert!(sched.remove_job(id).await.unwrap());
        assert!(!sched.remove_job(id).await.unwrap()); // already gone
        assert_eq!(sched.list_jobs().await.len(), 0);
    }

    #[tokio::test]
    async fn test_run_pending() {
        let sched = Scheduler::default();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let task: JobTask = Arc::new(move || {
            let c = counter_clone.clone();
            Box::pin(async move {
                c.fetch_add(1, Ordering::SeqCst);
            })
        });

        // Use a schedule that matches every minute
        let id = sched.add_job("counter", "* * * * *", task).await.unwrap();

        // Manually set next_run to past to trigger immediate execution
        {
            let mut jobs = sched.jobs.write().await;
            if let Some(job) = jobs.get_mut(&id) {
                job.next_run = Some(Utc::now() - chrono::TimeDelta::minutes(1));
            }
        }

        let count = sched.run_pending().await.unwrap();
        assert_eq!(count, 1);

        // Give the spawned task time to run
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn concurrent_run_pending_reserves_a_due_job_once() {
        let sched = Arc::new(Scheduler::default());
        let counter = Arc::new(AtomicUsize::new(0));
        let task: JobTask = {
            let counter = counter.clone();
            Arc::new(move || {
                let counter = counter.clone();
                Box::pin(async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(50)).await;
                })
            })
        };
        let id = sched.add_job("once", "* * * * *", task).await.unwrap();
        sched.jobs.write().await.get_mut(&id).unwrap().next_run =
            Some(Utc::now() - chrono::TimeDelta::minutes(1));

        let (first, second) = tokio::join!(sched.run_pending(), sched.run_pending());
        assert_eq!(first.unwrap() + second.unwrap(), 1);
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(sched.get_job(id).await.unwrap().running_count, 0);
    }

    #[tokio::test]
    async fn panicking_job_releases_running_count() {
        let sched = Scheduler::default();
        let task: JobTask = Arc::new(|| Box::pin(async { panic!("job failed") }));
        let id = sched.add_job("panic", "* * * * *", task).await.unwrap();
        sched.jobs.write().await.get_mut(&id).unwrap().next_run =
            Some(Utc::now() - chrono::TimeDelta::minutes(1));

        assert_eq!(sched.run_pending().await.unwrap(), 1);
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(sched.get_job(id).await.unwrap().running_count, 0);
    }

    #[tokio::test]
    async fn test_enable_disable_job() {
        let sched = Scheduler::default();
        let task: JobTask = Arc::new(|| Box::pin(async {}));

        let id = sched.add_job("toggle", "* * * * *", task).await.unwrap();

        assert!(sched.set_job_enabled(id, false).await.unwrap());
        let job = sched.get_job(id).await.unwrap();
        assert!(!job.enabled);

        assert!(sched.set_job_enabled(id, true).await.unwrap());
        let job = sched.get_job(id).await.unwrap();
        assert!(job.enabled);
    }

    #[tokio::test]
    async fn test_invalid_schedule() {
        let sched = Scheduler::default();
        let task: JobTask = Arc::new(|| Box::pin(async {}));

        let result = sched.add_job("bad", "not-a-cron", task).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_start_stop() {
        let sched = Scheduler::new(enabled_config());
        assert!(!sched.is_running().await);

        sched.start().await.unwrap();
        assert!(sched.is_running().await);

        // Give it a moment to start ticking
        tokio::time::sleep(Duration::from_millis(50)).await;

        sched.stop().await.unwrap();
        assert!(!sched.is_running().await);
    }

    #[tokio::test]
    async fn immediate_stop_wakes_long_interval_loop() {
        let sched = Scheduler::new(SchedulerConfig {
            enabled: true,
            check_interval_secs: 3_600,
            max_concurrent: 1,
            db_path: None,
        });

        sched.start().await.unwrap();
        tokio::time::timeout(Duration::from_secs(1), sched.stop())
            .await
            .expect("stop should not wait for the next scheduler tick")
            .unwrap();
    }

    // ── Persistence Integration Tests ───────────────────────────────

    #[tokio::test]
    async fn test_add_job_with_store_persists() {
        let store = Arc::new(SqliteSchedulerStore::in_memory().unwrap());
        let sched = Scheduler::new(enabled_config()).with_store(store.clone());
        let task: JobTask = Arc::new(|| Box::pin(async {}));

        let id = sched
            .add_job("persist-test", "*/10 * * * *", task)
            .await
            .unwrap();
        let loaded = store.load_all_jobs().await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].name, "persist-test");
        assert_eq!(loaded[0].id, id);
    }

    #[tokio::test]
    async fn test_add_job_with_config_persists() {
        let store = Arc::new(SqliteSchedulerStore::in_memory().unwrap());
        let sched = Scheduler::new(enabled_config()).with_store(store.clone());

        let config = SchedulerJobConfig::new("Run my task").with_max_iterations(50);
        let _id = sched
            .add_job_with_config("config-test", "0 */6 * * *", config)
            .await
            .unwrap();

        let loaded = store.load_all_jobs().await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].name, "config-test");
        assert_eq!(loaded[0].task_goal.as_deref(), Some("Run my task"));
        assert_eq!(loaded[0].max_iterations, 50);
    }

    #[tokio::test]
    async fn test_remove_job_deletes_from_store() {
        let store = Arc::new(SqliteSchedulerStore::in_memory().unwrap());
        let sched = Scheduler::new(enabled_config()).with_store(store.clone());
        let task: JobTask = Arc::new(|| Box::pin(async {}));

        let id = sched.add_job("delete-me", "0 9 * * *", task).await.unwrap();
        assert_eq!(store.load_all_jobs().await.unwrap().len(), 1);

        sched.remove_job(id).await.unwrap();
        assert_eq!(store.load_all_jobs().await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_set_job_enabled_updates_store() {
        let store = Arc::new(SqliteSchedulerStore::in_memory().unwrap());
        let sched = Scheduler::default().with_store(store.clone());
        let task: JobTask = Arc::new(|| Box::pin(async {}));

        let id = sched.add_job("toggle-me", "* * * * *", task).await.unwrap();

        // Disable
        sched.set_job_enabled(id, false).await.unwrap();
        let loaded = store.load_all_jobs().await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(!loaded[0].enabled);
    }

    #[tokio::test]
    async fn test_start_loads_persisted_jobs() {
        let store = Arc::new(SqliteSchedulerStore::in_memory().unwrap());

        // Pre-populate the store directly
        let pj = PersistedJob {
            id: Uuid::new_v4(),
            name: "pre-loaded".to_string(),
            cron_expr: "0 * * * *".to_string(),
            task_goal: None,
            max_iterations: 100,
            enabled: true,
            last_run: None,
            next_run: Some(Utc::now() + chrono::TimeDelta::hours(1)),
            run_count: 0,
            created_at: Utc::now(),
        };
        store.save_job(&pj).await.unwrap();

        // Create scheduler with this store and start it
        let sched = Scheduler::new(enabled_config()).with_store(store.clone());
        assert_eq!(sched.job_count().await, 0);

        sched.start().await.unwrap();

        // After start, the persisted job should be loaded
        let jobs = sched.list_jobs().await;
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].name, "pre-loaded");

        sched.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_scheduler_restart_persistence() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("test_scheduler_restart_{}.db", Uuid::new_v4()));
        let path_str = path.to_str().unwrap().to_string();

        // First scheduler: add job via store
        let id = {
            let store = Arc::new(SqliteSchedulerStore::new(&path_str).unwrap());
            let sched = Scheduler::new(enabled_config()).with_store(store.clone());
            let task: JobTask = Arc::new(|| Box::pin(async {}));
            sched
                .add_job("restart-test", "*/5 * * * *", task)
                .await
                .unwrap()
        };

        // Second scheduler (simulating restart): verify job is loaded
        {
            let store = Arc::new(SqliteSchedulerStore::new(&path_str).unwrap());
            let sched = Scheduler::new(enabled_config()).with_store(store.clone());

            sched.start().await.unwrap();
            let jobs = sched.list_jobs().await;
            assert_eq!(jobs.len(), 1);
            assert_eq!(jobs[0].name, "restart-test");
            assert_eq!(jobs[0].id, id);

            sched.stop().await.unwrap();
        }

        // Cleanup
        let _ = std::fs::remove_file(&path_str);
    }

    #[tokio::test]
    async fn test_in_memory_path_still_works() {
        // Even with store configured, the in-memory-only path for other
        // operations should still work fine
        let store = Arc::new(SqliteSchedulerStore::in_memory().unwrap());
        let sched = Scheduler::default().with_store(store);
        let task: JobTask = Arc::new(|| Box::pin(async {}));

        let id = sched.add_job("mem-test", "0 0 * * *", task).await.unwrap();
        let job = sched.get_job(id).await.unwrap();
        assert_eq!(job.name, "mem-test");
        assert!(job.enabled);
    }

    #[tokio::test]
    async fn test_run_pending_with_store_updates_state() {
        let store = Arc::new(SqliteSchedulerStore::in_memory().unwrap());
        let sched = Scheduler::default().with_store(store.clone());
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let task: JobTask = Arc::new(move || {
            let c = counter_clone.clone();
            Box::pin(async move {
                c.fetch_add(1, Ordering::SeqCst);
            })
        });

        let id = sched
            .add_job("state-update", "* * * * *", task)
            .await
            .unwrap();

        // Set next_run to past
        {
            let mut jobs = sched.jobs.write().await;
            if let Some(job) = jobs.get_mut(&id) {
                job.next_run = Some(Utc::now() - chrono::TimeDelta::minutes(1));
            }
        }

        let count = sched.run_pending().await.unwrap();
        assert_eq!(count, 1);

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify store was updated
        let loaded = store.load_all_jobs().await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].run_count, 1);
        assert!(loaded[0].last_run.is_some());
        assert!(loaded[0].next_run.is_some());
    }

    #[tokio::test]
    async fn configured_job_fails_without_runtime_instead_of_noop() {
        let sched = Scheduler::default();
        let id = sched
            .add_job_with_config(
                "runtime-required",
                "* * * * *",
                SchedulerJobConfig::new("perform real work"),
            )
            .await
            .unwrap();
        sched.jobs.write().await.get_mut(&id).unwrap().next_run =
            Some(Utc::now() - chrono::TimeDelta::minutes(1));

        let error = sched.run_pending().await.unwrap_err();
        assert!(error.to_string().contains("requires a runtime"));
        assert_eq!(sched.get_job(id).await.unwrap().run_count, 0);
    }

    #[tokio::test]
    async fn disabled_scheduler_does_not_start() {
        let sched = Scheduler::default();
        sched.start().await.unwrap();
        assert!(!sched.is_running().await);
    }
}
