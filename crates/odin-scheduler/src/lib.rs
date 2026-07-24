//! `odin-scheduler` — cron-like job scheduling for Raven Agent.
//!
//! Provides a [`Scheduler`] that manages a collection of scheduled [`Job`]s,
//! parses cron-like expressions via [`Schedule`], and dispatches due jobs
//! to Tokio tasks.
//!
//! # Persistence
//!
//! The [`SchedulerStore`] trait and [`SqliteSchedulerStore`] implementation
//! provide optional SQLite-backed persistence for job definitions and
//! execution history.

pub mod job;
pub mod scheduler;
pub mod store;

pub use job::{CronField, Job, JobId, JobTask, Schedule, noop_task};
pub use scheduler::Scheduler;
pub use store::{
    JobRun, JobRunStatus, PersistedJob, SchedulerJobConfig, SchedulerStore, SqliteSchedulerStore,
};
