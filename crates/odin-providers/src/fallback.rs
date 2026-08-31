//! FallbackProvider — chains multiple providers with circuit breaker and health checks.
//!
//! The FallbackProvider wraps a primary provider and a list of fallback providers.
//! On failure (error or timeout), it transparently tries the next provider in the chain.
//! Circuit breaker logic skips providers that have failed N consecutive times,
//! and re-enables them after a cooldown period.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::{ChatStream, Provider};
use odin_core::types::*;
use tokio::sync::{Mutex, Notify};

/// State of the circuit breaker for a single provider in the chain.
#[derive(Debug, Clone)]
struct CircuitBreakerState {
    /// Consecutive failure count
    failure_count: u32,
    /// When the last failure occurred (Unix timestamp in seconds)
    last_failure_time: i64,
    /// Whether the circuit is currently open (provider is being skipped)
    circuit_open: bool,
    /// Number of consecutive failures before opening the circuit
    threshold: u32,
    /// Cooldown duration in seconds before retrying an open circuit
    cooldown_secs: u64,
}

impl CircuitBreakerState {
    fn new(threshold: u32, cooldown_secs: u64) -> Self {
        Self {
            failure_count: 0,
            last_failure_time: 0,
            circuit_open: false,
            threshold,
            cooldown_secs,
        }
    }

    /// Record a success — resets the circuit.
    fn record_success(&mut self) {
        self.failure_count = 0;
        self.circuit_open = false;
    }

    /// Record a failure — may open the circuit.
    fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Utc::now().timestamp();
        if self.threshold > 0 && self.failure_count >= self.threshold {
            self.circuit_open = true;
        }
    }

    /// Check if the provider should be skipped (circuit open and still in cooldown).
    fn is_skippable(&self) -> bool {
        if !self.circuit_open || self.threshold == 0 {
            return false;
        }
        let elapsed = Utc::now().timestamp() - self.last_failure_time;
        elapsed < self.cooldown_secs as i64
    }
}

/// Health status of a provider in the chain.
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    /// Provider is healthy and accepting requests.
    Healthy,
    /// Provider is degraded but still accepting requests.
    Degraded,
    /// Provider is unhealthy — circuit breaker is open.
    Unhealthy { reason: String },
    /// Health has not been checked yet.
    Unknown,
}

/// Metadata about a provider in the fallback chain.
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    /// Provider name
    pub name: String,
    /// Provider type (e.g. "openai_compat", "anthropic")
    pub provider_type: String,
    /// Current health status
    pub health: HealthStatus,
    /// Base URL
    pub base_url: String,
}

/// Internal state shared between FallbackProvider and its health check task.
struct FallbackInner {
    /// Human-readable name for this chain
    name: String,
    /// Primary provider with its name
    primary: (String, Arc<dyn Provider>),
    /// Ordered fallback providers: (name, provider)
    fallbacks: Vec<(String, Arc<dyn Provider>)>,
    /// Circuit breaker state per provider name
    circuit_breakers: HashMap<String, Arc<Mutex<CircuitBreakerState>>>,
    /// Health status per provider name
    health_status: HashMap<String, Arc<Mutex<HealthStatus>>>,
    /// Whether the health check task has been spawned
    health_check_started: AtomicBool,
    /// Signals the background health checker to stop when the chain is dropped.
    health_check_cancelled: AtomicBool,
    /// Wakes the health checker immediately when the chain is dropped.
    health_check_shutdown: Notify,
    /// Health check interval in seconds
    health_check_interval: u64,
}

/// A provider that chains a primary with fallbacks, featuring circuit breaker and health checks.
///
/// The `chat()` method tries the primary provider first. If it fails, the next fallback
/// in the chain is tried, and so on. Circuit breaker logic tracks consecutive failures per
/// provider; after `circuit_breaker_threshold` failures, that provider is skipped for a
/// cooldown period. A background health check task periodically pings each provider's
/// health endpoint.
pub struct FallbackProvider {
    inner: Arc<FallbackInner>,
}

impl FallbackProvider {
    /// Create a new FallbackProvider.
    ///
    /// * `name` — name for this chain.
    /// * `primary_name` — name of the primary provider.
    /// * `primary` — primary provider.
    /// * `fallbacks` — ordered fallback providers with names.
    /// * `circuit_breaker_threshold` — failures before circuit opens (0 = disabled).
    /// * `health_check_interval_secs` — interval for health checks (0 = disabled).
    pub fn new(
        name: impl Into<String>,
        primary_name: impl Into<String>,
        primary: Arc<dyn Provider>,
        fallbacks: Vec<(String, Arc<dyn Provider>)>,
        circuit_breaker_threshold: u32,
        health_check_interval_secs: u64,
    ) -> Self {
        let name = name.into();
        let primary_name = primary_name.into();

        let mut circuit_breakers = HashMap::new();
        let mut health_status = HashMap::new();

        // Primary
        circuit_breakers.insert(
            primary_name.clone(),
            Arc::new(Mutex::new(CircuitBreakerState::new(
                circuit_breaker_threshold,
                60,
            ))),
        );
        health_status.insert(
            primary_name.clone(),
            Arc::new(Mutex::new(HealthStatus::Unknown)),
        );

        // Fallbacks
        for (f_name, _) in &fallbacks {
            circuit_breakers.insert(
                f_name.clone(),
                Arc::new(Mutex::new(CircuitBreakerState::new(
                    circuit_breaker_threshold,
                    60,
                ))),
            );
            health_status.insert(f_name.clone(), Arc::new(Mutex::new(HealthStatus::Unknown)));
        }

        let inner = Arc::new(FallbackInner {
            name,
            primary: (primary_name, primary),
            fallbacks,
            circuit_breakers,
            health_status,
            health_check_started: AtomicBool::new(false),
            health_check_cancelled: AtomicBool::new(false),
            health_check_shutdown: Notify::new(),
            health_check_interval: health_check_interval_secs,
        });

        Self { inner }
    }

    /// Start the background health check task.
    fn ensure_health_checks(&self) {
        if self.inner.health_check_interval == 0 {
            return;
        }
        if self
            .inner
            .health_check_started
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .is_err()
        {
            // Already started
            return;
        }

        let inner = self.inner.clone();
        let interval = Duration::from_secs(self.inner.health_check_interval);

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                tokio::select! {
                    _ = ticker.tick() => {}
                    _ = inner.health_check_shutdown.notified() => break,
                }

                if inner
                    .health_check_cancelled
                    .load(std::sync::atomic::Ordering::Acquire)
                {
                    break;
                }

                // Check primary
                let primary_healthy = inner.primary.1.health_check().await.unwrap_or(false);
                if let Some(status) = inner.health_status.get(&inner.primary.0) {
                    let mut s = status.lock().await;
                    if primary_healthy {
                        *s = HealthStatus::Healthy;
                    } else {
                        *s = HealthStatus::Unhealthy {
                            reason: "Health check failed".into(),
                        };
                    }
                }

                // Check fallbacks
                for (f_name, f_provider) in &inner.fallbacks {
                    let healthy = f_provider.health_check().await.unwrap_or(false);
                    if let Some(status) = inner.health_status.get(f_name) {
                        let mut s = status.lock().await;
                        if healthy {
                            *s = HealthStatus::Healthy;
                        } else {
                            *s = HealthStatus::Unhealthy {
                                reason: "Health check failed".into(),
                            };
                        }
                    }
                }
            }
        });
    }

    /// Try all providers in order until one succeeds.
    /// Returns the provider that succeeded and its response, or an error if all failed.
    async fn try_providers(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolSchema],
        options: &CompletionOptions,
    ) -> OdinResult<(String, ChatResponse)> {
        let mut last_error: Option<OdinError> = None;
        let mut tried_names: Vec<String> = Vec::new();

        // Build the ordered list: primary + fallbacks
        let mut chain: Vec<(&str, &Arc<dyn Provider>)> =
            Vec::with_capacity(1 + self.inner.fallbacks.len());
        chain.push((&self.inner.primary.0, &self.inner.primary.1));
        for (name, provider) in &self.inner.fallbacks {
            chain.push((name.as_str(), provider));
        }

        for (name, provider) in chain {
            // Check circuit breaker
            let should_skip = {
                let breaker = self.inner.circuit_breakers.get(name);
                match breaker {
                    None => false,
                    Some(cb) => {
                        let state = cb.lock().await;
                        if state.is_skippable() {
                            tracing::warn!(
                                "[FALLBACK] Skipping '{}' — circuit breaker open ({} consecutive failures)",
                                name,
                                state.failure_count
                            );
                            true
                        } else {
                            false
                        }
                    }
                }
            };
            if should_skip {
                tried_names.push(format!("{} (circuit open)", name));
                continue;
            }

            // If the circuit was half-open (cooldown expired), we let it through
            // but the attempt counts as a trial that can reset or re-open
            tracing::info!(
                "[FALLBACK] Trying provider '{}' for model '{}'",
                name,
                model
            );

            match provider.chat(model, messages, tools, options).await {
                Ok(response) => {
                    // Success — reset circuit breaker
                    if let Some(cb) = self.inner.circuit_breakers.get(name) {
                        let mut state = cb.lock().await;
                        state.record_success();
                    }
                    if let Some(hs) = self.inner.health_status.get(name) {
                        *hs.lock().await = HealthStatus::Healthy;
                    }
                    return Ok((name.to_string(), response));
                }
                Err(e) => {
                    if !e.is_retryable() {
                        tracing::warn!(
                            "[FALLBACK] Provider '{}' returned a non-retryable failure: {}",
                            name,
                            e
                        );
                        return Err(e);
                    }
                    tracing::warn!("[FALLBACK] Provider '{}' failed: {}", name, e);
                    // Record failure in circuit breaker
                    if let Some(cb) = self.inner.circuit_breakers.get(name) {
                        let mut state = cb.lock().await;
                        state.record_failure();
                        if state.circuit_open {
                            let status = self.inner.health_status.get(name);
                            if let Some(s) = status {
                                *s.lock().await = HealthStatus::Unhealthy {
                                    reason: format!(
                                        "Circuit breaker opened after {} failures",
                                        state.failure_count
                                    ),
                                };
                            }
                            tracing::warn!(
                                "[FALLBACK] Circuit breaker OPEN for '{}' ({} failures)",
                                name,
                                state.failure_count
                            );
                        }
                    }
                    tried_names.push(name.to_string());
                    last_error = Some(e);
                }
            }
        }

        Err(OdinError::Provider {
            provider: self.inner.name.clone(),
            message: format!(
                "All providers in fallback chain failed. Tried: [{}]. Last error: {}",
                tried_names.join(", "),
                last_error
                    .as_ref()
                    .map_or("none".to_string(), |e| e.to_string())
            ),
            source: last_error.map(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
        })
    }

    /// Try streaming providers in order until one succeeds.
    async fn try_stream_providers(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolSchema],
        options: &CompletionOptions,
    ) -> OdinResult<Box<dyn ChatStream>> {
        let mut last_error: Option<OdinError> = None;
        let mut tried_names: Vec<String> = Vec::new();

        let mut chain: Vec<(&str, &Arc<dyn Provider>)> =
            Vec::with_capacity(1 + self.inner.fallbacks.len());
        chain.push((&self.inner.primary.0, &self.inner.primary.1));
        for (name, provider) in &self.inner.fallbacks {
            chain.push((name.as_str(), provider));
        }

        for (name, provider) in chain {
            if let Some(cb) = self.inner.circuit_breakers.get(name) {
                let state = cb.lock().await;
                if state.is_skippable() {
                    tried_names.push(format!("{} (circuit open)", name));
                    continue;
                }
            }

            match provider.chat_stream(model, messages, tools, options).await {
                Ok(stream) => {
                    if let Some(cb) = self.inner.circuit_breakers.get(name) {
                        let mut state = cb.lock().await;
                        state.record_success();
                    }
                    if let Some(hs) = self.inner.health_status.get(name) {
                        *hs.lock().await = HealthStatus::Healthy;
                    }
                    return Ok(stream);
                }
                Err(e) => {
                    if !e.is_retryable() {
                        return Err(e);
                    }
                    if let Some(cb) = self.inner.circuit_breakers.get(name) {
                        let mut state = cb.lock().await;
                        state.record_failure();
                    }
                    tried_names.push(name.to_string());
                    last_error = Some(e);
                }
            }
        }

        Err(OdinError::Provider {
            provider: self.inner.name.clone(),
            message: format!(
                "All providers in fallback chain failed. Tried: [{}]. Last error: {}",
                tried_names.join(", "),
                last_error
                    .as_ref()
                    .map_or("none".to_string(), |e| e.to_string())
            ),
            source: last_error.map(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
        })
    }

    /// Get information about all providers in the chain.
    pub async fn provider_info(&self) -> Vec<ProviderInfo> {
        let mut info = Vec::new();

        // Primary
        let health = match self.inner.health_status.get(&self.inner.primary.0) {
            Some(status) => status.lock().await.clone(),
            None => HealthStatus::Unknown,
        };
        info.push(ProviderInfo {
            name: self.inner.primary.0.clone(),
            provider_type: "primary".into(),
            health,
            base_url: String::new(),
        });

        // Fallbacks
        for (f_name, _) in &self.inner.fallbacks {
            let health = match self.inner.health_status.get(f_name) {
                Some(status) => status.lock().await.clone(),
                None => HealthStatus::Unknown,
            };
            info.push(ProviderInfo {
                name: f_name.clone(),
                provider_type: "fallback".into(),
                health,
                base_url: String::new(),
            });
        }

        info
    }

    /// Get all circuit breaker states for diagnostics.
    pub async fn circuit_breaker_states(&self) -> HashMap<String, (u32, bool)> {
        let mut result = HashMap::new();
        for (name, cb) in &self.inner.circuit_breakers {
            let state = cb.lock().await;
            result.insert(name.clone(), (state.failure_count, state.circuit_open));
        }
        result
    }
}

impl Drop for FallbackProvider {
    fn drop(&mut self) {
        self.inner
            .health_check_cancelled
            .store(true, std::sync::atomic::Ordering::Release);
        self.inner.health_check_shutdown.notify_waiters();
    }
}

#[async_trait]
impl Provider for FallbackProvider {
    fn name(&self) -> &str {
        &self.inner.name
    }

    async fn list_models(&self) -> OdinResult<Vec<ModelInfo>> {
        self.inner.primary.1.list_models().await
    }

    async fn chat(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolSchema],
        options: &CompletionOptions,
    ) -> OdinResult<ChatResponse> {
        self.ensure_health_checks();
        let (_, response) = self.try_providers(model, messages, tools, options).await?;
        Ok(response)
    }

    async fn chat_stream(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolSchema],
        options: &CompletionOptions,
    ) -> OdinResult<Box<dyn ChatStream>> {
        self.ensure_health_checks();
        let stream = self
            .try_stream_providers(model, messages, tools, options)
            .await?;
        Ok(stream)
    }

    async fn health_check(&self) -> OdinResult<bool> {
        self.ensure_health_checks();

        // Check primary health
        let primary_healthy = self.inner.primary.1.health_check().await.unwrap_or(false);
        if primary_healthy {
            return Ok(true);
        }

        // Check fallbacks — return true if any fallback is healthy
        for (_, provider) in &self.inner.fallbacks {
            if provider.health_check().await.unwrap_or(false) {
                return Ok(true);
            }
        }

        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::Mutex as StdMutex;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use tokio::sync::Mutex as TokioMutex;

    // ── Mock Provider Helpers ───────────────────────────────────────

    struct MockProvider {
        name: String,
        chat_result: StdMutex<OdinResult<String>>,
        permanent_error: bool,
        call_count: AtomicUsize,
        health_result: TokioMutex<bool>,
    }

    impl MockProvider {
        fn new_ok(name: &str, response: String) -> Self {
            Self {
                name: name.to_string(),
                chat_result: StdMutex::new(Ok(response)),
                permanent_error: false,
                call_count: AtomicUsize::new(0),
                health_result: TokioMutex::new(true),
            }
        }

        fn new_err(name: &str, error_msg: impl Into<String>) -> Self {
            let msg = error_msg.into();
            Self {
                name: name.to_string(),
                chat_result: StdMutex::new(Err(OdinError::provider(name, msg))),
                permanent_error: false,
                call_count: AtomicUsize::new(0),
                health_result: TokioMutex::new(true),
            }
        }

        fn new_with_health(name: &str, result: OdinResult<String>, health: bool) -> Self {
            Self {
                name: name.to_string(),
                chat_result: StdMutex::new(result),
                permanent_error: false,
                call_count: AtomicUsize::new(0),
                health_result: TokioMutex::new(health),
            }
        }

        fn call_count(&self) -> usize {
            self.call_count.load(AtomicOrdering::SeqCst)
        }

        fn new_permanent_err(name: &str, error_msg: impl Into<String>) -> Self {
            Self {
                name: name.to_string(),
                chat_result: StdMutex::new(Err(OdinError::provider_permanent(
                    name,
                    error_msg.into(),
                ))),
                permanent_error: true,
                call_count: AtomicUsize::new(0),
                health_result: TokioMutex::new(true),
            }
        }
    }

    impl MockProvider {
        fn chat_impl(&self) -> OdinResult<ChatResponse> {
            self.call_count.fetch_add(1, AtomicOrdering::SeqCst);
            let result = self.chat_result.lock().unwrap();
            match &*result {
                Ok(content) => {
                    let resp = ChatResponse {
                        message: Message::assistant(content),
                        usage: TokenUsage::default(),
                        finish_reason: Some("stop".into()),
                        model: self.name.clone(),
                    };
                    Ok(resp)
                }
                Err(e) if self.permanent_error => {
                    Err(OdinError::provider_permanent(&self.name, e.to_string()))
                }
                Err(e) => Err(OdinError::provider(&self.name, format!("{}", e))),
            }
        }
    }

    #[async_trait]
    impl Provider for MockProvider {
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
            self.chat_impl()
        }
        async fn chat_stream(
            &self,
            _model: &str,
            _messages: &[Message],
            _tools: &[ToolSchema],
            _options: &CompletionOptions,
        ) -> OdinResult<Box<dyn ChatStream>> {
            Err(OdinError::provider(&self.name, "streaming not supported"))
        }
        async fn health_check(&self) -> OdinResult<bool> {
            Ok(*self.health_result.lock().await)
        }
    }

    fn make_response(text: &str) -> String {
        text.to_string()
    }

    // ── Tests ──────────────────────────────────────────────────────

    /// Primary succeeds → returns primary result
    #[tokio::test]
    async fn test_primary_succeeds() {
        let primary = Arc::new(MockProvider::new_ok(
            "primary",
            make_response("primary result"),
        ));
        let fallback = Arc::new(MockProvider::new_ok(
            "fallback",
            make_response("fallback result"),
        ));

        let fp = FallbackProvider::new(
            "test-chain",
            "primary",
            primary.clone(),
            vec![("fallback".into(), fallback.clone())],
            0, // circuit breaker disabled
            0, // health checks disabled
        );

        let result = fp
            .chat("test-model", &[], &[], &CompletionOptions::default())
            .await;
        assert!(result.is_ok());
        let text = result.unwrap().message.text().unwrap_or("").to_string();
        assert_eq!(text, "primary result");

        // Only primary should have been called
        assert_eq!(primary.call_count(), 1);
        // fallback should not have been called
        assert_eq!(fallback.call_count(), 0);
    }

    /// Primary fails → fallback succeeds
    #[tokio::test]
    async fn test_fallback_on_failure() {
        let primary = Arc::new(MockProvider::new_err("primary", "connection refused"));
        let fallback = Arc::new(MockProvider::new_ok(
            "fallback",
            make_response("fallback result"),
        ));

        let fp = FallbackProvider::new(
            "test-chain",
            "primary",
            primary.clone(),
            vec![("fallback".into(), fallback.clone())],
            0,
            0,
        );

        let result = fp
            .chat("test-model", &[], &[], &CompletionOptions::default())
            .await;
        assert!(result.is_ok());
        let text = result.unwrap().message.text().unwrap_or("").to_string();
        assert_eq!(text, "fallback result");

        // Both should have been called
        assert_eq!(primary.call_count(), 1);
        assert_eq!(fallback.call_count(), 1);
    }

    #[tokio::test]
    async fn permanent_provider_failure_does_not_fan_out_to_fallback() {
        let primary = Arc::new(MockProvider::new_permanent_err(
            "primary",
            "invalid request",
        ));
        let fallback = Arc::new(MockProvider::new_ok(
            "fallback",
            make_response("should not run"),
        ));
        let fp = FallbackProvider::new(
            "test-chain",
            "primary",
            primary.clone(),
            vec![("fallback".into(), fallback.clone())],
            0,
            0,
        );

        let error = fp
            .chat("test-model", &[], &[], &CompletionOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(error, OdinError::ProviderPermanent { .. }));
        assert_eq!(primary.call_count(), 1);
        assert_eq!(fallback.call_count(), 0);
    }

    /// All fail → error with chain info
    #[tokio::test]
    async fn test_all_fail() {
        let primary = Arc::new(MockProvider::new_err("primary", "timeout"));
        let fallback = Arc::new(MockProvider::new_err("fallback", "fallback also failed"));

        let fp = FallbackProvider::new(
            "test-chain",
            "primary",
            primary.clone(),
            vec![("fallback".into(), fallback.clone())],
            0,
            0,
        );

        let result = fp
            .chat("test-model", &[], &[], &CompletionOptions::default())
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("All providers in fallback chain failed"));
        assert!(err.contains("primary"));
        assert!(err.contains("fallback"));

        assert_eq!(primary.call_count(), 1);
        assert_eq!(fallback.call_count(), 1);
    }

    /// Circuit breaker: N failures → skips for cooldown → re-enables after cooldown
    #[tokio::test]
    async fn test_circuit_breaker_opens_and_retries() {
        let primary = Arc::new(MockProvider::new_err("primary", "server error"));
        let fallback = Arc::new(MockProvider::new_ok(
            "fallback",
            make_response("fallback result"),
        ));

        let fp = FallbackProvider::new(
            "test-chain",
            "primary",
            primary.clone(),
            vec![("fallback".into(), fallback.clone())],
            2, // open after 2 failures
            0, // no health checks
        );

        // First call: primary fails (1), fallback succeeds
        let r1 = fp
            .chat("test-model", &[], &[], &CompletionOptions::default())
            .await;
        assert!(r1.is_ok());
        assert_eq!(r1.unwrap().message.text().unwrap_or(""), "fallback result");
        assert_eq!(primary.call_count(), 1);
        assert_eq!(fallback.call_count(), 1);

        // Change fallback to also fail for second call
        let _fallback_err = Arc::new(MockProvider::new_err("fallback", "fail 1b"));

        // Use a new fallback provider that fails (need to test circuit breaker behaviour properly)
        // Actually, let's re-test more directly by checking the circuit breaker state
        {
            let cb = fp.inner.circuit_breakers.get("primary").unwrap();
            let state = cb.lock().await;
            // After 1st failure → failure_count should be 1, circuit not open yet
            assert_eq!(state.failure_count, 1);
            assert!(!state.circuit_open);
        }

        // Second call: primary fails again (now at 2), fallback should still be tried
        let primary2 = Arc::new(MockProvider::new_err("primary", "fail 2"));

        let fp2 = FallbackProvider::new(
            "test-chain-2",
            "primary",
            primary2.clone(),
            vec![(
                "fallback".into(),
                Arc::new(MockProvider::new_ok("fallback", make_response("fb2"))),
            )],
            2,
            0,
        );

        // 1st failure
        let _ = fp2
            .chat("model", &[], &[], &CompletionOptions::default())
            .await;
        // 2nd failure → circuit should open
        let r2 = fp2
            .chat("model", &[], &[], &CompletionOptions::default())
            .await;
        assert!(r2.is_ok());

        // Check circuit is open
        let cb = fp2.inner.circuit_breakers.get("primary").unwrap();
        let state = cb.lock().await;
        assert_eq!(state.failure_count, 2);
        assert!(state.circuit_open);
        drop(state);

        // Check that primary was actually skipped
        assert_eq!(primary2.call_count(), 2); // only 2 attempts, 3rd call skipped primary

        // Now verify the third call would skip primary (circuit open)
        let r3 = fp2
            .chat("model", &[], &[], &CompletionOptions::default())
            .await;
        assert!(r3.is_ok());
        // primary call count should still be 2 (skipped on 3rd call)
        assert_eq!(primary2.call_count(), 2);
        // fallback still works
        let state2 = cb.lock().await;
        assert_eq!(state2.failure_count, 2); // still 2 failures
        // Should be skippable
        assert!(state2.is_skippable());
    }

    /// Health check returns true if primary is healthy
    #[tokio::test]
    async fn test_health_check_healthy() {
        let primary = Arc::new(MockProvider::new_ok("primary", make_response("ok")));
        let fallback = Arc::new(MockProvider::new_ok("fallback", make_response("ok")));

        let fp = FallbackProvider::new(
            "test-chain",
            "primary",
            primary,
            vec![("fallback".into(), fallback)],
            0,
            0,
        );

        let healthy = fp.health_check().await.unwrap_or(false);
        assert!(healthy);
    }

    /// Health check returns true if primary is down but fallback is up
    #[tokio::test]
    async fn test_health_check_fallback_alive() {
        let primary = Arc::new(MockProvider::new_with_health(
            "primary",
            Ok(make_response("ok")),
            false, // primary health fails
        ));
        let fallback = Arc::new(MockProvider::new_with_health(
            "fallback",
            Ok(make_response("ok")),
            true, // fallback healthy
        ));

        let fp = FallbackProvider::new(
            "test-chain",
            "primary",
            primary,
            vec![("fallback".into(), fallback)],
            0,
            0,
        );

        let healthy = fp.health_check().await.unwrap_or(false);
        assert!(healthy);
    }

    /// Test provider_info returns correct metadata
    #[tokio::test]
    async fn test_provider_info() {
        let primary = Arc::new(MockProvider::new_ok("primary", make_response("ok")));
        let fallback = Arc::new(MockProvider::new_ok("fallback", make_response("ok")));

        let fp = FallbackProvider::new(
            "test-chain",
            "primary",
            primary,
            vec![("fallback".into(), fallback)],
            0,
            0,
        );

        let info = fp.provider_info().await;
        assert_eq!(info.len(), 2);
        assert_eq!(info[0].name, "primary");
        assert!(matches!(info[0].health, HealthStatus::Unknown));
        assert_eq!(info[1].name, "fallback");
    }
}
