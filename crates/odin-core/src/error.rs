//! Error types for Raven Agent.

use crate::types::{LoopPhase, Severity};

/// The main error type for Raven Agent operations.
#[derive(Debug, thiserror::Error)]
pub enum OdinError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Provider error ({provider}): {message}")]
    Provider {
        provider: String,
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// A provider rejected the request or returned an invalid response. This
    /// is intentionally distinct from a transport/provider outage so fallback
    /// and orchestration code cannot retry it blindly.
    #[error("Non-retryable provider error ({provider}): {message}")]
    ProviderPermanent {
        provider: String,
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Model error: {0}")]
    Model(String),

    #[error("Tool error ({tool}): {message}")]
    Tool {
        tool: String,
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Rate limit exceeded: {0}")]
    RateLimit(String),

    #[error("Context limit exceeded: used {used} tokens, limit is {limit}")]
    ContextLimit { used: u32, limit: u32 },

    #[error("Loop error at phase {phase:?}: {message}")]
    Loop { phase: LoopPhase, message: String },

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("{0}")]
    Other(String),
}

impl OdinError {
    /// Classify the severity of this error.
    pub fn severity(&self) -> Severity {
        match self {
            OdinError::Config(_) => Severity::Error,
            OdinError::Provider { .. } => Severity::Error,
            OdinError::ProviderPermanent { .. } => Severity::Error,
            OdinError::Model(_) => Severity::Warning,
            OdinError::Tool { .. } => Severity::Warning,
            OdinError::PermissionDenied(_) => Severity::Warning,
            OdinError::RateLimit(_) => Severity::Warning,
            OdinError::ContextLimit { .. } => Severity::Warning,
            OdinError::Loop { .. } => Severity::Warning,
            OdinError::Validation(_) => Severity::Warning,
            OdinError::Serialization(_) => Severity::Error,
            OdinError::Io(_) => Severity::Error,
            OdinError::Network(_) => Severity::Error,
            OdinError::Database(_) => Severity::Error,
            OdinError::Timeout(_) => Severity::Warning,
            OdinError::Internal(_) => Severity::Critical,
            OdinError::Other(_) => Severity::Error,
        }
    }

    /// Whether this error is retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            OdinError::Provider { .. }
                | OdinError::RateLimit(_)
                | OdinError::Timeout(_)
                | OdinError::Network(_)
        )
    }

    /// Create a provider error.
    pub fn provider(provider: impl Into<String>, message: impl Into<String>) -> Self {
        OdinError::Provider {
            provider: provider.into(),
            message: message.into(),
            source: None,
        }
    }

    /// Create a provider error that must not be retried.
    pub fn provider_permanent(provider: impl Into<String>, message: impl Into<String>) -> Self {
        OdinError::ProviderPermanent {
            provider: provider.into(),
            message: message.into(),
            source: None,
        }
    }

    /// Create a tool error.
    pub fn tool(tool: impl Into<String>, message: impl Into<String>) -> Self {
        OdinError::Tool {
            tool: tool.into(),
            message: message.into(),
            source: None,
        }
    }

    /// Create a loop error.
    pub fn loop_error(phase: LoopPhase, message: impl Into<String>) -> Self {
        OdinError::Loop {
            phase,
            message: message.into(),
        }
    }
}

/// Convenience Result type.
pub type OdinResult<T> = Result<T, OdinError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryability_is_explicit_for_provider_failures() {
        assert!(OdinError::provider("p", "temporary outage").is_retryable());
        assert!(!OdinError::provider_permanent("p", "bad request").is_retryable());
        assert!(!OdinError::Validation("bad arguments".into()).is_retryable());
        assert!(!OdinError::Model("invalid model".into()).is_retryable());
    }
}
