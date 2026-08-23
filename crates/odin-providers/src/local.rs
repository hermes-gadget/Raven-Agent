//! Local model provider (llama.cpp, ollama, etc.)

use async_trait::async_trait;
use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::{ChatStream, Provider};
use odin_core::types::*;
use reqwest::Client;

/// Provider for locally-hosted models via Ollama or llama.cpp server.
pub struct LocalProvider {
    client: Client,
    base_url: String,
}

impl LocalProvider {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
        }
    }

    /// Create a provider for Ollama (default: http://localhost:11434).
    pub fn ollama() -> Self {
        Self::new("http://localhost:11434/v1")
    }

    /// Create a provider for llama.cpp server (default: http://localhost:8080).
    pub fn llama_cpp() -> Self {
        Self::new("http://localhost:8080/v1")
    }
}

#[async_trait]
impl Provider for LocalProvider {
    fn name(&self) -> &str {
        "local"
    }

    async fn list_models(&self) -> OdinResult<Vec<ModelInfo>> {
        // For local providers, we use the generic /models endpoint
        let resp = self
            .client
            .get(format!("{}/models", self.base_url))
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => {
                let json: serde_json::Value = r.json().await.unwrap_or_default();
                let models: Vec<ModelInfo> = json["data"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .map(|m| ModelInfo {
                                id: m["id"].as_str().unwrap_or("local-model").to_string(),
                                provider: "local".into(),
                                context_length: 8192,
                                supports_tools: true,
                                supports_vision: false,
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                if models.is_empty() {
                    Ok(vec![ModelInfo {
                        id: "local-model".into(),
                        provider: "local".into(),
                        context_length: 8192,
                        supports_tools: true,
                        supports_vision: false,
                    }])
                } else {
                    Ok(models)
                }
            }
            _ => {
                // Local provider might not be running — return a default model
                Ok(vec![ModelInfo {
                    id: "local-model".into(),
                    provider: "local".into(),
                    context_length: 8192,
                    supports_tools: true,
                    supports_vision: false,
                }])
            }
        }
    }

    async fn chat(
        &self,
        model: &str,
        messages: &[Message],
        _tools: &[ToolSchema],
        _options: &CompletionOptions,
    ) -> OdinResult<ChatResponse> {
        // Delegate to OpenAI-compatible API (most local servers support this)
        let openai_compat =
            crate::openai_compat::OpenAiCompatProvider::new("local-compat", &self.base_url, None);
        openai_compat.chat(model, messages, _tools, _options).await
    }

    async fn chat_stream(
        &self,
        _model: &str,
        _messages: &[Message],
        _tools: &[ToolSchema],
        _options: &CompletionOptions,
    ) -> OdinResult<Box<dyn ChatStream>> {
        Err(OdinError::provider(
            "local",
            "Streaming not yet implemented",
        ))
    }

    async fn health_check(&self) -> OdinResult<bool> {
        let response = self
            .client
            .get(format!("{}/models", self.base_url))
            .send()
            .await
            .map_err(|error| {
                OdinError::provider("local", format!("Health check failed: {error}"))
            })?;

        Ok(response.status().is_success())
    }
}
