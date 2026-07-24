//! Odin Orchestrator — Multi-agent orchestration layer for Raven Agent.
//!
//! The orchestrator sits above the runtime and provides:
//! - **Composer**: User-facing agent that intakes goals, tracks intent, steers sub-agents
//! - **TaskGraph**: Parent goal → sub-goals → agents → files/tools → outputs
//! - **FileLockManager**: Safe concurrent file access with queue and merge resolution
//! - **MergeResolver**: Combines parallel sub-agent results into one coherent response
//! - **AgentLifecycle**: Full state machine for sub-agent execution
//! - **SubAgentPool**: Scoped agents with isolated tools, files, and permissions
//!
//! Default behavior: multi-agent orchestration. One user message → auto-split into
//! independent workstreams → spawn parallel sub-agents → merge results.

pub mod composer;
pub mod control;
pub mod file_lock;
pub mod lifecycle;
pub mod merge;
pub mod persistence;
pub mod progress;
pub mod sub_agent;
pub mod task_graph;

pub use composer::{
    Composer, capabilities_to_tools, default_agent_tools, default_capabilities_for_goal,
};
pub use control::{ControlAuth, RunControlCommand, RunControlKind, RunControlStatus, authorize_control};
pub use file_lock::{FileLock, FileLockManager};
pub use lifecycle::{AgentLifecycle, AgentPhase};
pub use merge::{MergeResolver, MergeStrategy};
pub use persistence::OrchestrationStore;
pub use progress::{ProgressTracker, WorkstreamStatus};
pub use sub_agent::{SubAgent, SubAgentConfig};
pub use task_graph::{TaskEdge, TaskGraph, TaskNode};
