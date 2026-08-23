//! Raven Agent core — foundation types, configuration, errors, and traits.
//!
//! This crate provides the shared vocabulary for the internal `odin-*` crates.
//! It has minimal dependencies and no feature flags — everything here
//! is always available.

pub mod config;
pub mod error;
pub mod path_security;
pub mod traits;
pub mod types;

// Re-export commonly used items
pub use config::OdinConfig;
pub use error::OdinError;
pub use path_security::ResolvedPathBoundary;
pub use types::*;
