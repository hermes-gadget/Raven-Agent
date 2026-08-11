//! E2E test: FallbackProvider with mock primary and fallback providers.
//!
//! Tests that:
//! 1. When the primary always fails, the fallback is used
//! 2. Circuit breaker opens after N consecutive failures
//! 3. Once the circuit is open, the failing provider is skipped

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::{ChatStream, Provider};
use odin_core::types::*;
use odin_providers::fallback::FallbackProvider;

// ── Mock: always-failing provider ──────────────────────────────────────

struct FailingProvider {
    name: String,
    call_count: Arc<AtomicUsize>,
}

impl FailingProvider {
    fn new(name: &str, call_count: Arc<AtomicUsize>) -> Self {
        Self {
            name: name.to_string(),
            call_count,
        }
    }
}

#[async_trait]
impl Provider for FailingProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn list_models(&self) -> OdinResult<Vec<ModelInfo>> {
        Ok(vec![])
    }

    async fn chat(
        &self,
        _model: &str,
        _messages: &[Message],
        _tools: &[ToolSchema],
        _options: &CompletionOptions,
    ) -> OdinResult<ChatResponse> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Err(OdinError::provider(&self.name, "Intentional mock failure"))
    }

    async fn chat_stream(
        &self,
        _model: &str,
        _messages: &[Message],
        _tools: &[ToolSchema],
        _options: &CompletionOptions,
    ) -> OdinResult<Box<dyn ChatStream>> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Err(OdinError::provider(&self.name, "Intentional mock failure"))
    }

    async fn health_check(&self) -> OdinResult<bool> {
        Ok(false)
    }
}

// ── Mock: always-succeeding provider ───────────────────────────────────

struct SuccessProvider {
    name: String,
    call_count: Arc<AtomicUsize>,
}

impl SuccessProvider {
    fn new(name: &str, call_count: Arc<AtomicUsize>) -> Self {
        Self {
            name: name.to_string(),
            call_count,
        }
    }
}

#[async_trait]
impl Provider for SuccessProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn list_models(&self) -> OdinResult<Vec<ModelInfo>> {
        Ok(vec![])
    }

    async fn chat(
        &self,
        _model: &str,
        _messages: &[Message],
        _tools: &[ToolSchema],
        _options: &CompletionOptions,
    ) -> OdinResult<ChatResponse> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(ChatResponse {
            message: Message::assistant("Mock fallback response"),
            usage: TokenUsage::default(),
            finish_reason: Some("stop".into()),
            model: "mock-model".into(),
        })
    }

    async fn chat_stream(
        &self,
        _model: &str,
        _messages: &[Message],
        _tools: &[ToolSchema],
        _options: &CompletionOptions,
    ) -> OdinResult<Box<dyn ChatStream>> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Err(OdinError::provider(
            &self.name,
            "Stream not implemented in mock",
        ))
    }

    async fn health_check(&self) -> OdinResult<bool> {
        Ok(true)
    }
}

// ── Helper to build the fallback chain ─────────────────────────────────

fn make_fallback(
    primary: Arc<dyn Provider>,
    fallback: Arc<dyn Provider>,
    threshold: u32,
) -> FallbackProvider {
    FallbackProvider::new(
        "test-chain",
        "primary",
        primary,
        vec![("fallback".to_string(), fallback)],
        threshold,
        0, // no health checks for these tests
    )
}

// ── Tests ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_fallback_used_when_primary_fails() {
    let primary_count = Arc::new(AtomicUsize::new(0));
    let fallback_count = Arc::new(AtomicUsize::new(0));

    let primary = Arc::new(FailingProvider::new("primary", primary_count.clone()));
    let fallback = Arc::new(SuccessProvider::new("fallback", fallback_count.clone()));

    let provider = make_fallback(primary, fallback, 3);

    let result = provider
        .chat(
            "gpt-4",
            &[Message::user("test")],
            &[],
            &CompletionOptions::default(),
        )
        .await;

    assert!(
        result.is_ok(),
        "chat should succeed via fallback when primary fails"
    );

    // Primary was called once and failed; fallback was called once and succeeded
    assert_eq!(
        primary_count.load(Ordering::SeqCst),
        1,
        "primary should have been called once"
    );
    assert_eq!(
        fallback_count.load(Ordering::SeqCst),
        1,
        "fallback should have been called once"
    );
}

#[tokio::test]
async fn test_circuit_breaker_opens_after_n_failures() {
    // threshold = 2 means circuit opens after 2 consecutive failures
    let threshold = 2u32;
    let primary_count = Arc::new(AtomicUsize::new(0));
    let fallback_count = Arc::new(AtomicUsize::new(0));

    let primary = Arc::new(FailingProvider::new("primary", primary_count.clone()));
    let fallback = Arc::new(SuccessProvider::new("fallback", fallback_count.clone()));

    let provider = make_fallback(primary, fallback, threshold);

    let opts = CompletionOptions::default();
    let msgs = &[Message::user("test")];

    // First call: primary fails → circuit records failure 1 → fallback succeeds
    let r1 = provider.chat("gpt-4", msgs, &[], &opts).await;
    assert!(r1.is_ok(), "call 1 should succeed (fallback)");

    // Second call: primary fails → circuit records failure 2 → circuit opens → fallback succeeds
    let r2 = provider.chat("gpt-4", msgs, &[], &opts).await;
    assert!(r2.is_ok(), "call 2 should succeed (fallback)");

    // Third call: primary is circuit-open → primary skipped → fallback succeeds
    let r3 = provider.chat("gpt-4", msgs, &[], &opts).await;
    assert!(
        r3.is_ok(),
        "call 3 should succeed (fallback, primary skipped)"
    );

    // Check circuit breaker states
    let states = provider.circuit_breaker_states().await;
    assert!(
        states.contains_key("primary"),
        "primary should have a circuit breaker state"
    );

    // circuit_breaker_states returns HashMap<String, (u32, bool)>
    let (failure_count, circuit_open) = states.get("primary").copied().unwrap_or_default();
    assert_eq!(
        failure_count, threshold,
        "primary should have {} failures",
        threshold
    );
    assert!(circuit_open, "primary circuit should be open");

    // The fallback should have been called 3 times (once per request)
    assert_eq!(
        fallback_count.load(Ordering::SeqCst),
        3,
        "fallback should have been called for all 3 requests"
    );
    // Primary was called 2 times (then skipped on 3rd)
    assert_eq!(
        primary_count.load(Ordering::SeqCst),
        2,
        "primary should have been called twice then circuit-open"
    );
}

#[tokio::test]
async fn test_circuit_breaker_state_persists_across_calls() {
    let threshold = 3u32;
    let primary_count = Arc::new(AtomicUsize::new(0));
    let fallback_count = Arc::new(AtomicUsize::new(0));

    let primary = Arc::new(FailingProvider::new("primary", primary_count.clone()));
    let fallback = Arc::new(SuccessProvider::new("fallback", fallback_count.clone()));

    let provider = make_fallback(primary, fallback, threshold);

    let opts = CompletionOptions::default();
    let msgs = &[Message::user("test")];

    // Make threshold-1 calls — circuit should still be closed
    for i in 0..threshold - 1 {
        let r = provider.chat("gpt-4", msgs, &[], &opts).await;
        assert!(r.is_ok(), "call {} should succeed via fallback", i + 1);
    }

    // Circuit should still be closed (only threshold-1 failures)
    let states = provider.circuit_breaker_states().await;
    let (failure_count, circuit_open) = states.get("primary").copied().unwrap_or_default();
    assert_eq!(
        failure_count,
        threshold - 1,
        "{} failures so far",
        threshold - 1
    );
    assert!(!circuit_open, "circuit should still be closed");

    // One more call triggers the circuit breaker
    let r = provider.chat("gpt-4", msgs, &[], &opts).await;
    assert!(r.is_ok(), "call {} should succeed via fallback", threshold);

    let states = provider.circuit_breaker_states().await;
    let (failure_count, circuit_open) = states.get("primary").copied().unwrap_or_default();
    assert_eq!(failure_count, threshold, "{} failures total", threshold);
    assert!(circuit_open, "circuit should now be open");
}
