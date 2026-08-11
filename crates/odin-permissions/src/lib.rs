//! `odin-permissions` — safety permissions and redaction for Raven Agent.
//!
//! Provides a complete permission system including:
//! - [`PolicyEngine`]: Evaluates tool calls and commands against allow/deny rules
//! - [`ApprovalGate`]: Interactive approval for dangerous operations
//! - [`SecretManager`]: Secure credential and secret handling
//! - [`SecretRedactor`]: Scans output for API keys/tokens and redacts them

pub mod approval;
pub mod policy;
pub mod redact;
pub mod secrets;

pub use approval::{ApprovalGate, ApprovalRequest, ApprovalStatus};
pub use policy::PolicyEngine;
pub use redact::SecretRedactor;
pub use secrets::{Secret, SecretManager, SecretMetadata};
