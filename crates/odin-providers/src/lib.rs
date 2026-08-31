//! Odin Providers — Abstract model provider layer.
//!
//! Supports OpenAI-compatible APIs, Anthropic, and local models.
//! Designed so adding a new provider requires implementing one trait.

pub mod anthropic;
pub mod factory;
pub mod fallback;
pub mod local;
pub mod openai_compat;
pub mod registry;
pub mod traits;

pub use anthropic::AnthropicProvider;
pub use factory::create_provider;
pub use factory::create_provider_chain;
pub use local::LocalProvider;
pub use openai_compat::OpenAiCompatProvider;
pub use registry::ProviderRegistry;
pub use traits::ProviderExt;

const MAX_PROVIDER_ERROR_BODY_CHARS: usize = 4_096;

/// Convert a remote HTTP failure into a bounded, redacted typed error.
pub(crate) fn provider_http_error(
    provider: &str,
    status: u16,
    body: impl Into<String>,
) -> odin_core::error::OdinError {
    let body = odin_permissions::SecretRedactor::secrets_only().redact(&body.into());
    let body: String = body
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .take(MAX_PROVIDER_ERROR_BODY_CHARS)
        .collect();
    let body = if body.trim().is_empty() {
        "<empty response body>".to_string()
    } else {
        body.trim().to_string()
    };
    let message = format!("HTTP {status}: {body}");
    if status == 429 || (500..=599).contains(&status) {
        odin_core::error::OdinError::provider(provider, message)
    } else {
        odin_core::error::OdinError::provider_permanent(provider, message)
    }
}

#[cfg(test)]
mod tests {
    use super::provider_http_error;
    use odin_core::error::OdinError;

    #[test]
    fn remote_error_bodies_are_bounded_redacted_and_typed() {
        let error = provider_http_error(
            "test",
            400,
            format!(
                "Authorization: Bearer sk-12345678901234567890\n{}",
                "x".repeat(10_000)
            ),
        );
        assert!(matches!(error, OdinError::ProviderPermanent { .. }));
        let rendered = error.to_string();
        assert!(!rendered.contains("sk-12345678901234567890"));
        assert!(!rendered.contains('\n'));
        assert!(rendered.chars().count() <= 4_200);

        assert!(provider_http_error("test", 503, "overloaded").is_retryable());
    }
}
