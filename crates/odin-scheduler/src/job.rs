//! Job and schedule types for the scheduler.
//!
//! Defines [`Job`] as a scheduled unit of work with a cron-like schedule,
//! and [`Schedule`] for parsing cron expressions.

use chrono::{DateTime, Datelike, Timelike, Utc};
use odin_core::types::TaskId;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use uuid::Uuid;

/// Unique identifier for a scheduled job.
pub type JobId = Uuid;

/// A task to execute — an async boxed closure.
pub type JobTask = Arc<dyn Send + Sync + Fn() -> Pin<Box<dyn Future<Output = ()> + Send>>>;

/// A scheduled job that runs on a cron-like schedule.
#[derive(Clone)]
pub struct Job {
    /// Unique job identifier.
    pub id: JobId,
    /// Human-readable name for this job.
    pub name: String,
    /// Cron-like schedule expression.
    pub schedule: Schedule,
    /// The async task to execute.
    pub task: JobTask,
    /// Whether this job is currently enabled.
    pub enabled: bool,
    /// When this job last ran (None if never run).
    pub last_run: Option<DateTime<Utc>>,
    /// When this job is scheduled to run next.
    pub next_run: Option<DateTime<Utc>>,
    /// Optional task ID from the last execution.
    pub last_task_id: Option<TaskId>,
    /// Number of times this job has run.
    pub run_count: u64,
    /// Maximum number of concurrent executions (0 = unlimited).
    pub max_concurrent: u32,
    /// Currently running task count.
    pub running_count: u32,
    /// Optional task goal for runtime-driven execution.
    ///
    /// When set, the scheduler creates an `AgentTask` from this goal
    /// and submits it via the runtime instead of running the generic closure.
    pub task_goal: Option<String>,
    /// Maximum iterations when executing via runtime.
    pub max_iterations: u32,
    /// When this job was created.
    pub created_at: DateTime<Utc>,
}

impl Job {
    /// Create a new job with the given name, schedule, and task.
    pub fn new(name: impl Into<String>, schedule: Schedule, task: JobTask) -> Self {
        let now = Utc::now();
        let next_run = schedule.next_occurrence(now);
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            schedule,
            task,
            enabled: true,
            last_run: None,
            next_run,
            last_task_id: None,
            run_count: 0,
            max_concurrent: 1,
            running_count: 0,
            task_goal: None,
            max_iterations: 100,
            created_at: now,
        }
    }

    /// Check if this job is due to run at the given time.
    pub fn is_due(&self, now: &DateTime<Utc>) -> bool {
        self.enabled && self.next_run.is_some_and(|t| t <= *now)
    }

    /// Calculate the next run time after the current one completes.
    pub fn calculate_next_run(&mut self) {
        let now = Utc::now();
        self.next_run = self.schedule.next_occurrence(now);
    }

    /// Mark the job as having run.
    pub fn mark_run(&mut self, task_id: TaskId) {
        self.last_run = Some(Utc::now());
        self.last_task_id = Some(task_id);
        self.run_count += 1;
        self.calculate_next_run();
    }
}

impl fmt::Debug for Job {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Job")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("schedule", &self.schedule)
            .field("enabled", &self.enabled)
            .field("last_run", &self.last_run)
            .field("next_run", &self.next_run)
            .field("run_count", &self.run_count)
            .field("task_goal", &self.task_goal)
            .field("max_iterations", &self.max_iterations)
            .finish()
    }
}

/// Create an inert task closure for use when deserialising persisted jobs.
///
/// This is used internally when loading jobs from the store; the actual
/// task (runtime-driven or closure-based) is set up by the scheduler.
pub fn noop_task() -> JobTask {
    Arc::new(|| Box::pin(async {}))
}

/// A cron-like schedule parsed from a 5-field expression.
///
/// Fields: minute, hour, day_of_month, month, day_of_week
/// All values are 0-indexed or 1-indexed per cron convention:
/// - minute: 0-59
/// - hour: 0-23
/// - day_of_month: 1-31
/// - month: 1-12
/// - day_of_week: 0-6 (0 = Sunday, 1 = Monday, ...)
///
/// Supports `*` (wildcard) in all fields.
/// Supports comma-separated lists: `1,3,5`
/// Supports ranges: `1-5`
/// Supports step values: `*/5`, `1-10/2`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schedule {
    /// Raw cron expression string.
    pub expression: String,
    /// Parsed minute field.
    pub minute: CronField,
    /// Parsed hour field.
    pub hour: CronField,
    /// Parsed day of month field.
    pub day_of_month: CronField,
    /// Parsed month field.
    pub month: CronField,
    /// Parsed day of week field.
    pub day_of_week: CronField,
}

impl Schedule {
    /// Parse a cron expression string into a Schedule.
    ///
    /// Supports standard 5-field cron expressions:
    /// ```ignore
    /// * * * * *     (every minute)
    /// */5 * * * *   (every 5 minutes)
    /// 0 * * * *     (every hour)
    /// 0 9 * * 1-5   (9 AM weekdays)
    /// ```
    pub fn parse(expression: &str) -> Result<Self, String> {
        let parts: Vec<&str> = expression.split_whitespace().collect();
        if parts.len() != 5 {
            return Err(format!(
                "Invalid cron expression '{}': expected 5 fields, got {}",
                expression,
                parts.len()
            ));
        }

        let minute = CronField::parse(parts[0], 0, 59)?;
        let hour = CronField::parse(parts[1], 0, 23)?;
        let day_of_month = CronField::parse(parts[2], 1, 31)?;
        let month = CronField::parse(parts[3], 1, 12)?;
        let day_of_week = CronField::parse(parts[4], 0, 6)?;

        Ok(Self {
            expression: expression.to_string(),
            minute,
            hour,
            day_of_month,
            month,
            day_of_week,
        })
    }

    /// Calculate the next occurrence of this schedule at or after `from`.
    pub fn next_occurrence(&self, from: DateTime<Utc>) -> Option<DateTime<Utc>> {
        // Start from the next minute to avoid immediate re-trigger
        let mut candidate = from
            .with_second(0)
            .and_then(|t| t.checked_add_signed(chrono::TimeDelta::minutes(1)))?;

        // Search up to 2 years into the future
        let deadline = from + chrono::TimeDelta::days(730);

        while candidate <= deadline {
            if self.month.matches(candidate.month() as i32)
                && self.day_of_month.matches(candidate.day() as i32)
                && self
                    .day_of_week
                    .matches(candidate.weekday().num_days_from_sunday() as i32)
                && self.hour.matches(candidate.hour() as i32)
                && self.minute.matches(candidate.minute() as i32)
            {
                return Some(candidate);
            }
            candidate = candidate.checked_add_signed(chrono::TimeDelta::minutes(1))?;
        }

        None
    }

    /// Check if the given datetime matches this schedule.
    pub fn matches(&self, dt: &DateTime<Utc>) -> bool {
        self.minute.matches(dt.minute() as i32)
            && self.hour.matches(dt.hour() as i32)
            && self.day_of_month.matches(dt.day() as i32)
            && self.month.matches(dt.month() as i32)
            && self
                .day_of_week
                .matches(dt.weekday().num_days_from_sunday() as i32)
    }
}

impl fmt::Display for Schedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expression)
    }
}

/// A parsed cron field supporting wildcards, lists, ranges, and steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CronField {
    /// Matches any value (`*`)
    Wildcard,
    /// Matches a specific value
    Value(i32),
    /// Matches any of the listed values (`1,3,5`)
    List(Vec<i32>),
    /// Matches a range (`1-5`)
    Range(i32, i32),
    /// Matches with a bounded step (`*/5`, `1-10/2`).
    Step(i32, i32, i32),
}

impl CronField {
    /// Parse a single cron field string.
    fn parse(input: &str, min: i32, max: i32) -> Result<Self, String> {
        let input = input.trim();

        if input == "*" {
            return Ok(CronField::Wildcard);
        }

        // Check for step: */5 or 1-10/2
        if let Some((base, step)) = input.split_once('/') {
            let step_val = step
                .parse::<i32>()
                .map_err(|_| format!("Invalid step value '{}' in field '{}'", step, input))?;
            if step_val <= 0 {
                return Err(format!("Step must be positive in field '{}'", input));
            }
            if base == "*" {
                return Ok(CronField::Step(min, max, step_val));
            }
            // Range with step: 1-10/2
            if let Some((start, end)) = base.split_once('-') {
                let start_val = start
                    .parse::<i32>()
                    .map_err(|_| format!("Invalid range start '{}' in field '{}'", start, input))?;
                let end_val = end
                    .parse::<i32>()
                    .map_err(|_| format!("Invalid range end '{}' in field '{}'", end, input))?;
                if start_val < min || end_val > max || start_val > end_val {
                    return Err(format!(
                        "Range {}-{} out of range [{}, {}] in field '{}'",
                        start_val, end_val, min, max, input
                    ));
                }
                return Ok(CronField::Step(start_val, end_val, step_val));
            }
            // Single value with step (unusual but supported)
            let val = base
                .parse::<i32>()
                .map_err(|_| format!("Invalid value '{}' in field '{}'", base, input))?;
            if val < min || val > max {
                return Err(format!(
                    "Value {} out of range [{}, {}] in field '{}'",
                    val, min, max, input
                ));
            }
            return Ok(CronField::Step(val, max, step_val));
        }

        // Check for list: 1,3,5
        if input.contains(',') {
            let values: Result<Vec<i32>, _> =
                input.split(',').map(|s| s.trim().parse::<i32>()).collect();
            let values = values.map_err(|_| format!("Invalid list value in field '{}'", input))?;
            // Validate range
            for &v in &values {
                if v < min || v > max {
                    return Err(format!(
                        "Value {} out of range [{}, {}] in field '{}'",
                        v, min, max, input
                    ));
                }
            }
            return Ok(CronField::List(values));
        }

        // Check for range: 1-5
        if let Some((start, end)) = input.split_once('-') {
            let start_val = start
                .parse::<i32>()
                .map_err(|_| format!("Invalid range start '{}' in field '{}'", start, input))?;
            let end_val = end
                .parse::<i32>()
                .map_err(|_| format!("Invalid range end '{}' in field '{}'", end, input))?;
            if start_val < min || end_val > max || start_val > end_val {
                return Err(format!(
                    "Range {}-{} out of range [{}, {}] in field '{}'",
                    start_val, end_val, min, max, input
                ));
            }
            return Ok(CronField::Range(start_val, end_val));
        }

        // Single value
        let val = input
            .parse::<i32>()
            .map_err(|_| format!("Invalid value '{}' in field '{}'", input, input))?;
        if val < min || val > max {
            return Err(format!(
                "Value {} out of range [{}, {}] in field '{}'",
                val, min, max, input
            ));
        }
        Ok(CronField::Value(val))
    }

    /// Check if the given value matches this field.
    fn matches(&self, value: i32) -> bool {
        match self {
            CronField::Wildcard => true,
            CronField::Value(v) => *v == value,
            CronField::List(values) => values.contains(&value),
            CronField::Range(start, end) => value >= *start && value <= *end,
            CronField::Step(start, end, step) => {
                value >= *start && value <= *end && (value - start) % step == 0
            }
        }
    }
}

// Note: The 'Field' variant was renamed to 'Value' — keep backward compat:
impl CronField {
    #[allow(dead_code)]
    fn field_value(&self) -> Option<i32> {
        match self {
            CronField::Value(v) => Some(*v),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_wildcard() {
        let sched = Schedule::parse("* * * * *").unwrap();
        assert!(matches!(sched.minute, CronField::Wildcard));
        assert!(matches!(sched.hour, CronField::Wildcard));
        assert!(matches!(sched.day_of_month, CronField::Wildcard));
        assert!(matches!(sched.month, CronField::Wildcard));
        assert!(matches!(sched.day_of_week, CronField::Wildcard));
    }

    #[test]
    fn test_parse_specific_minute() {
        let sched = Schedule::parse("30 * * * *").unwrap();
        assert!(matches!(sched.minute, CronField::Value(30)));
        assert!(matches!(sched.hour, CronField::Wildcard));
    }

    #[test]
    fn test_parse_step() {
        let sched = Schedule::parse("*/5 * * * *").unwrap();
        assert!(matches!(sched.minute, CronField::Step(0, 59, 5)));
    }

    #[test]
    fn test_ranged_step_respects_end_boundary() {
        let sched = Schedule::parse("1-10/2 * * * *").unwrap();
        assert!(matches!(sched.minute, CronField::Step(1, 10, 2)));
        assert!(
            sched.matches(
                &DateTime::parse_from_rfc3339("2025-01-01T00:09:00Z")
                    .unwrap()
                    .with_timezone(&Utc)
            )
        );
        assert!(
            !sched.matches(
                &DateTime::parse_from_rfc3339("2025-01-01T00:11:00Z")
                    .unwrap()
                    .with_timezone(&Utc)
            )
        );
        assert!(Schedule::parse("1-60/2 * * * *").is_err());
    }

    #[test]
    fn test_parse_list() {
        let sched = Schedule::parse("0 9,18 * * 1-5").unwrap();
        assert!(matches!(sched.hour, CronField::List(_)));
        assert!(matches!(sched.day_of_week, CronField::Range(1, 5)));
    }

    #[test]
    fn test_parse_invalid_expression() {
        assert!(Schedule::parse("invalid").is_err());
        assert!(Schedule::parse("* * * * * *").is_err()); // 6 fields
        assert!(Schedule::parse("60 * * * *").is_err()); // minute out of range
    }

    #[test]
    fn test_schedule_matches() {
        let sched = Schedule::parse("30 9 * * 1-5").unwrap();
        // Monday 9:30 AM
        let dt = DateTime::parse_from_rfc3339("2025-01-06T09:30:00Z")
            .unwrap()
            .with_timezone(&Utc);
        assert!(sched.matches(&dt));
        // Monday 9:31 AM (wrong minute)
        let dt2 = DateTime::parse_from_rfc3339("2025-01-06T09:31:00Z")
            .unwrap()
            .with_timezone(&Utc);
        assert!(!sched.matches(&dt2));
        // Saturday 9:30 AM (wrong day)
        let dt3 = DateTime::parse_from_rfc3339("2025-01-11T09:30:00Z")
            .unwrap()
            .with_timezone(&Utc);
        assert!(!sched.matches(&dt3));
    }

    #[test]
    fn test_job_creation() {
        let sched = Schedule::parse("0 * * * *").unwrap();
        let task: JobTask = Arc::new(|| Box::pin(async {}));
        let job = Job::new("test-job", sched.clone(), task);
        assert_eq!(job.name, "test-job");
        assert!(job.enabled);
        assert!(job.last_run.is_none());
        assert!(job.next_run.is_some());
        assert_eq!(job.run_count, 0);
    }

    #[test]
    fn test_job_is_due() {
        let sched = Schedule::parse("* * * * *").unwrap();
        let task: JobTask = Arc::new(|| Box::pin(async {}));
        let mut job = Job::new("due-test", sched, task);
        // next_run should be in the future, so it's not due "now" minus 1 minute
        let past = Utc::now() - chrono::TimeDelta::minutes(1);
        // But if next_run is set to the past...
        job.next_run = Some(past);
        assert!(job.is_due(&Utc::now()));

        job.enabled = false;
        assert!(!job.is_due(&Utc::now()));
    }

    #[test]
    fn test_job_mark_run() {
        let sched = Schedule::parse("0 * * * *").unwrap();
        let task: JobTask = Arc::new(|| Box::pin(async {}));
        let mut job = Job::new("mark-test", sched, task);
        let task_id = Uuid::new_v4();
        job.mark_run(task_id);
        assert!(job.last_run.is_some());
        assert_eq!(job.last_task_id, Some(task_id));
        assert_eq!(job.run_count, 1);
    }
}
