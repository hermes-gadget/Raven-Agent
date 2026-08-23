//! OpenAI-compatible provider (works with OpenAI, Ollama, vLLM, Groq, DeepSeek, etc.)

use async_trait::async_trait;
use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::{ChatStream, Provider};
use odin_core::types::*;
use reqwest::Client;
use serde_json::Value;
use std::time::Duration;

/// Provider for any OpenAI-compatible API endpoint.
pub struct OpenAiCompatProvider {
    name: String,
    client: Client,
    base_url: String,
    api_key: Option<String>,
}

impl OpenAiCompatProvider {
    pub fn new(
        name: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        Self::new_with_timeout(name, base_url, api_key, None)
    }

    pub fn new_with_timeout(
        name: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<String>,
        timeout_secs: Option<u64>,
    ) -> Self {
        let client = timeout_secs
            .filter(|secs| *secs > 0)
            .and_then(|secs| {
                Client::builder()
                    .timeout(Duration::from_secs(secs))
                    .build()
                    .ok()
            })
            .unwrap_or_default();
        Self {
            name: name.into(),
            client,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key,
        }
    }

    fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    fn models_url(&self) -> String {
        format!("{}/models", self.base_url)
    }

    fn build_request(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolSchema],
        options: &CompletionOptions,
    ) -> OdinResult<Value> {
        if let Some(temperature) = options.temperature
            && !temperature.is_finite()
        {
            return Err(OdinError::Validation("temperature must be finite".into()));
        }
        if let Some(top_p) = options.top_p
            && !top_p.is_finite()
        {
            return Err(OdinError::Validation("top_p must be finite".into()));
        }

        let messages_json: Vec<Value> = messages
            .iter()
            .map(|m| {
                let mut map = serde_json::Map::new();
                map.insert(
                    "role".into(),
                    serde_json::to_value(m.role.to_string()).unwrap(),
                );

                match &m.content {
                    MessageContent::Text { content } => {
                        map.insert("content".into(), Value::String(content.clone()));
                    }
                    MessageContent::ToolCalls {
                        content,
                        tool_calls,
                    }
                    | MessageContent::AssistantWithTools {
                        content,
                        tool_calls,
                    } => {
                        if let Some(c) = content {
                            map.insert("content".into(), Value::String(c.clone()));
                        }
                        if !tool_calls.is_empty() {
                            let tc_json: Vec<Value> = tool_calls
                                .iter()
                                .map(|tc| {
                                    serde_json::json!({
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        }
                                    })
                                })
                                .collect();
                            map.insert("tool_calls".into(), Value::Array(tc_json));
                        }
                    }
                }

                if let Some(ref name) = m.name {
                    map.insert("name".into(), Value::String(name.clone()));
                }
                if let Some(ref tci) = m.tool_call_id {
                    map.insert("tool_call_id".into(), Value::String(tci.clone()));
                }

                Value::Object(map)
            })
            .collect();

        let mut body = serde_json::json!({
            "model": model,
            "messages": messages_json,
        });

        if !tools.is_empty() {
            let tools_json: Vec<Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": t.function,
                    })
                })
                .collect();
            body["tools"] = Value::Array(tools_json);
        }

        if let Some(temp) = options.temperature {
            body["temperature"] = Value::Number(
                serde_json::Number::from_f64(temp)
                    .ok_or_else(|| OdinError::Validation("temperature must be finite".into()))?,
            );
        }
        if let Some(mt) = options.max_tokens {
            body["max_tokens"] = Value::Number(mt.into());
        }
        if let Some(tp) = options.top_p {
            body["top_p"] = Value::Number(
                serde_json::Number::from_f64(tp)
                    .ok_or_else(|| OdinError::Validation("top_p must be finite".into()))?,
            );
        }

        Ok(body)
    }
}

#[async_trait]
impl Provider for OpenAiCompatProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn list_models(&self) -> OdinResult<Vec<ModelInfo>> {
        let mut req = self.client.get(self.models_url());
        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }

        let resp = req.send().await.map_err(|e| {
            OdinError::provider(&self.name, format!("Failed to list models: {}", e))
        })?;

        let json: Value = resp.json().await.map_err(|e| {
            OdinError::provider(&self.name, format!("Invalid models response: {}", e))
        })?;

        let models = json["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|m| ModelInfo {
                        id: m["id"].as_str().unwrap_or("unknown").to_string(),
                        provider: self.name.clone(),
                        context_length: 32768,
                        supports_tools: true,
                        supports_vision: false,
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }

    async fn chat(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolSchema],
        options: &CompletionOptions,
    ) -> OdinResult<ChatResponse> {
        let body = self.build_request(model, messages, tools, options)?;

        let mut req = self.client.post(self.chat_url()).json(&body);
        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }

        let resp = req
            .send()
            .await
            .map_err(|e| OdinError::provider(&self.name, format!("Chat request failed: {}", e)))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(OdinError::provider(
                &self.name,
                format!("HTTP {}: {}", status.as_u16(), text),
            ));
        }

        let json: Value = resp.json().await.map_err(|e| {
            OdinError::provider(&self.name, format!("Invalid chat response: {}", e))
        })?;

        let choice = &json["choices"][0];
        let msg = &choice["message"];

        // Try content first, fall back to reasoning_content for DeepSeek-style models
        let content = msg["content"]
            .as_str()
            .and_then(|s| {
                if s.is_empty() || s == "null" {
                    None
                } else {
                    Some(s.to_string())
                }
            })
            .or_else(|| msg["reasoning_content"].as_str().map(|s| s.to_string()));
        let tool_calls: Vec<ToolCall> = msg["tool_calls"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|tc| ToolCall {
                        id: tc["id"].as_str().unwrap_or("").to_string(),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: tc["function"]["name"].as_str().unwrap_or("").to_string(),
                            arguments: tc["function"]["arguments"]
                                .as_str()
                                .unwrap_or("{}")
                                .to_string(),
                        },
                    })
                    .collect()
            })
            .unwrap_or_default();

        let message = if !tool_calls.is_empty() {
            Message {
                role: Role::Assistant,
                content: MessageContent::ToolCalls {
                    content,
                    tool_calls,
                },
                name: None,
                tool_call_id: None,
            }
        } else {
            Message::assistant(content.unwrap_or_default())
        };

        let usage = json["usage"]
            .as_object()
            .map_or(TokenUsage::default(), |u| TokenUsage {
                prompt_tokens: u["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: u["total_tokens"].as_u64().unwrap_or(0) as u32,
            });

        Ok(ChatResponse {
            message,
            usage,
            finish_reason: choice["finish_reason"].as_str().map(|s| s.to_string()),
            model: json["model"].as_str().unwrap_or(model).to_string(),
        })
    }

    async fn chat_stream(
        &self,
        _model: &str,
        _messages: &[Message],
        _tools: &[ToolSchema],
        _options: &CompletionOptions,
    ) -> OdinResult<Box<dyn ChatStream>> {
        // Streaming not yet implemented; fall back to non-streaming
        Err(OdinError::provider(
            &self.name,
            "Streaming not yet implemented",
        ))
    }

    async fn health_check(&self) -> OdinResult<bool> {
        match self.list_models().await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_name() {
        let provider = OpenAiCompatProvider::new("test", "http://localhost:11434/v1", None);
        assert_eq!(provider.name(), "test");
    }

    #[test]
    fn test_build_request_basic() {
        let provider = OpenAiCompatProvider::new("test", "http://localhost:11434/v1", None);
        let messages = vec![Message::user("Hello")];
        let body = provider
            .build_request("llama3", &messages, &[], &CompletionOptions::default())
            .unwrap();

        assert_eq!(body["model"], "llama3");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "Hello");
    }

    #[test]
    fn test_build_request_with_tools() {
        let provider = OpenAiCompatProvider::new("test", "http://localhost:11434/v1", None);
        let schema = ToolSchema {
            schema_type: "function".into(),
            function: FunctionSchema {
                name: "test_tool".into(),
                description: "A test tool".into(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            },
        };
        let body = provider
            .build_request(
                "gpt-4",
                &[Message::user("Use the tool")],
                &[schema],
                &CompletionOptions::default(),
            )
            .unwrap();

        assert!(body["tools"].is_array());
        assert_eq!(body["tools"][0]["function"]["name"], "test_tool");
    }

    #[test]
    fn test_build_request_rejects_non_finite_sampling_options() {
        let provider = OpenAiCompatProvider::new("test", "http://localhost:11434/v1", None);
        let messages = vec![Message::user("Hello")];

        let mut options = CompletionOptions {
            temperature: Some(f64::NAN),
            ..Default::default()
        };
        assert!(
            provider
                .build_request("llama3", &messages, &[], &options)
                .is_err()
        );

        options.temperature = Some(0.7);
        options.top_p = Some(f64::INFINITY);
        assert!(
            provider
                .build_request("llama3", &messages, &[], &options)
                .is_err()
        );
    }
}
