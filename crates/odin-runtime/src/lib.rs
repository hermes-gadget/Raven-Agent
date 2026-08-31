//! Odin Runtime — Agent lifecycle, session management, and orchestration.
//!
//! The runtime is the backbone of Raven. It manages multiple agents across
//! sessions, tracks conversation state, and provides sub-agent spawning for
//! parallel task execution.

pub mod agent;
pub mod composition;
pub mod runtime;
pub mod session;

pub use agent::Agent;
pub use composition::{
    CompositionResources, EngineBuildOptions, ExecutionSurface, ProductionComposition,
    ResolvedComposition,
};
pub use runtime::Runtime;
pub use session::Session;
