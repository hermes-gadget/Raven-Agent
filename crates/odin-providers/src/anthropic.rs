//! Anthropic provider (Claude models).

use async_trait::async_trait;
use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::{ChatStream, Provider};
use odin_core::types::*;
use reqwest::Client;
use serde_json::Value;
use std::time::Duration;

pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Anthropic HTTP client configuration is valid"),
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com/v1".into(),
        }
    }

    fn convert_messages(messages: &[Message]) -> OdinResult<(Option<String>, Vec<Value>)> {
        let system = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .map(|m| m.text().unwrap_or("").to_string())
            .collect::<Vec<_>>()
            .join("\n");

        let system_msg = if system.is_empty() {
            None
        } else {
            Some(system)
        };

        let mut anthropic_msgs = Vec::new();
        for message in messages.iter().filter(|m| m.role != Role::System) {
            let role = if message.role == Role::Assistant {
                "assistant"
            } else {
                "user"
            };
            let mut content = Vec::new();

            if let Some(text) = message.text()
                && !text.is_empty()
            {
                content.push(serde_json::json!({"type": "text", "text": text}));
            }

            match message.role {
                Role::Assistant => {
                    for call in message.tool_calls() {
                        let input: Value =
                            serde_json::from_str(&call.function.arguments).map_err(|error| {
                                OdinError::provider(
                                    "anthropic",
                                    format!(
                                        "Tool call '{}' has invalid JSON arguments: {error}",
                                        call.function.name
                                    ),
                                )
                            })?;
                        content.push(serde_json::json!({
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.function.name,
                            "input": input,
                        }));
                    }
                }
                Role::Tool => {
                    content.clear();
                    content.push(serde_json::json!({
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id.as_deref().unwrap_or(""),
                        "content": message.text().unwrap_or(""),
                    }));
                }
                Role::User | Role::System => {}
            }

            anthropic_msgs.push(serde_json::json!({"role": role, "content": content}));
        }

        Ok((system_msg, anthropic_msgs))
    }

    fn build_request(
        model: &str,
        messages: &[Message],
        tools: &[ToolSchema],
        options: &CompletionOptions,
    ) -> OdinResult<Value> {
        let (system, anthropic_msgs) = Self::convert_messages(messages)?;
        let mut body = serde_json::json!({
            "model": model,
            "max_tokens": options.max_tokens.unwrap_or(4096),
            "messages": anthropic_msgs,
        });
        let object = body
            .as_object_mut()
            .expect("Anthropic request body is always an object");

        if let Some(system) = system {
            object.insert("system".into(), Value::String(system));
        }
        if !tools.is_empty() {
            object.insert(
                "tools".into(),
                Value::Array(
                    tools
                        .iter()
                        .map(|tool| {
                            serde_json::json!({
                                "name": tool.function.name,
                                "description": tool.function.description,
                                "input_schema": tool.function.parameters,
                            })
                        })
                        .collect(),
                ),
            );
            object.insert("tool_choice".into(), serde_json::json!({"type": "auto"}));
        }
        if let Some(temperature) = options.temperature {
            object.insert("temperature".into(), serde_json::json!(temperature));
        }
        if let Some(top_p) = options.top_p {
            object.insert("top_p".into(), serde_json::json!(top_p));
        }
        if let Some(stop) = &options.stop
            && !stop.is_empty()
        {
            object.insert("stop_sequences".into(), serde_json::json!(stop));
        }
        Ok(body)
    }

    fn parse_response(json: &Value, requested_model: &str) -> OdinResult<ChatResponse> {
        let blocks = json
            .get("content")
            .and_then(Value::as_array)
            .filter(|blocks| !blocks.is_empty())
            .ok_or_else(|| {
                OdinError::provider("anthropic", "Response is missing a non-empty content array")
            })?;
        let mut text_parts = Vec::new();
        let mut tool_calls = Vec::new();
        for block in blocks {
            match block["type"].as_str() {
                Some("text") => {
                    let text = block["text"].as_str().ok_or_else(|| {
                        OdinError::provider("anthropic", "Text content block is malformed")
                    })?;
                    text_parts.push(text.to_string());
                }
                Some("tool_use") => {
                    let id = block["id"].as_str().ok_or_else(|| {
                        OdinError::provider("anthropic", "Tool-use block is missing its id")
                    })?;
                    let name = block["name"].as_str().ok_or_else(|| {
                        OdinError::provider("anthropic", "Tool-use block is missing its name")
                    })?;
                    let input = block.get("input").ok_or_else(|| {
                        OdinError::provider("anthropic", "Tool-use block is missing its input")
                    })?;
                    tool_calls.push(ToolCall {
                        id: id.to_string(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: name.to_string(),
                            arguments: serde_json::to_string(input).map_err(|error| {
                                OdinError::provider(
                                    "anthropic",
                                    format!("Tool-use input could not be serialized: {error}"),
                                )
                            })?,
                        },
                    });
                }
                _ => {}
            }
        }

        if text_parts.is_empty() && tool_calls.is_empty() {
            return Err(OdinError::provider(
                "anthropic",
                "Response content contains no usable text or tool-use blocks",
            ));
        }

        let text = (!text_parts.is_empty()).then(|| text_parts.join("\n"));
        let message = if tool_calls.is_empty() {
            Message::assistant(text.unwrap_or_default())
        } else {
            Message {
                role: Role::Assistant,
                content: MessageContent::ToolCalls {
                    content: text,
                    tool_calls,
                },
                name: None,
                tool_call_id: None,
            }
        };
        let prompt_tokens = json["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32;
        let completion_tokens = json["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32;

        Ok(ChatResponse {
            message,
            usage: TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens.saturating_add(completion_tokens),
            },
            finish_reason: json["stop_reason"].as_str().map(str::to_string),
            model: json["model"]
                .as_str()
                .unwrap_or(requested_model)
                .to_string(),
        })
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    async fn list_models(&self) -> OdinResult<Vec<ModelInfo>> {
        Ok(vec![
            ModelInfo {
                id: "claude-sonnet-4-20250514".into(),
                provider: "anthropic".into(),
                context_length: 200000,
                supports_tools: true,
                supports_vision: true,
            },
            ModelInfo {
                id: "claude-haiku-3-5-20241022".into(),
                provider: "anthropic".into(),
                context_length: 200000,
                supports_tools: true,
                supports_vision: false,
            },
        ])
    }

    async fn chat(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolSchema],
        options: &CompletionOptions,
    ) -> OdinResult<ChatResponse> {
        let body = Self::build_request(model, messages, tools, options)?;

        let resp = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(|e| OdinError::provider("anthropic", format!("Request failed: {}", e)))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(OdinError::provider(
                "anthropic",
                format!("HTTP {}: {}", status.as_u16(), text),
            ));
        }

        let json: Value = resp
            .json()
            .await
            .map_err(|e| OdinError::provider("anthropic", format!("Invalid response: {}", e)))?;

        Self::parse_response(&json, model)
    }

    async fn chat_stream(
        &self,
        _model: &str,
        _messages: &[Message],
        _tools: &[ToolSchema],
        _options: &CompletionOptions,
    ) -> OdinResult<Box<dyn ChatStream>> {
        Err(OdinError::provider(
            "anthropic",
            "Streaming not yet implemented",
        ))
    }

    async fn health_check(&self) -> OdinResult<bool> {
        let response = self
            .client
            .get(format!("{}/models", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await
            .map_err(|error| {
                OdinError::provider("anthropic", format!("Health check failed: {error}"))
            })?;

        Ok(response.status().is_success())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn weather_tool() -> ToolSchema {
        ToolSchema {
            schema_type: "function".into(),
            function: FunctionSchema {
                name: "weather".into(),
                description: "Fetch weather".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }),
            },
        }
    }

    #[test]
    fn request_preserves_system_tools_options_and_tool_results() {
        let assistant_call = Message {
            role: Role::Assistant,
            content: MessageContent::ToolCalls {
                content: Some("I'll check.".into()),
                tool_calls: vec![ToolCall {
                    id: "call-1".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "weather".into(),
                        arguments: r#"{"city":"London"}"#.into(),
                    },
                }],
            },
            name: None,
            tool_call_id: None,
        };
        let messages = vec![
            Message::system("Be concise."),
            Message::user("Weather?"),
            assistant_call,
            Message::tool_result("call-1", r#"{"temp":19}"#),
        ];
        let options = CompletionOptions {
            temperature: Some(0.2),
            max_tokens: Some(512),
            top_p: Some(0.9),
            stop: Some(vec!["STOP".into()]),
            stream: None,
        };

        let request =
            AnthropicProvider::build_request("claude-test", &messages, &[weather_tool()], &options)
                .unwrap();

        assert_eq!(request["system"], "Be concise.");
        assert_eq!(request["max_tokens"], 512);
        assert_eq!(request["temperature"], 0.2);
        assert_eq!(request["top_p"], 0.9);
        assert_eq!(request["stop_sequences"][0], "STOP");
        assert_eq!(request["tools"][0]["name"], "weather");
        assert_eq!(request["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(request["messages"][1]["content"][1]["type"], "tool_use");
        assert_eq!(request["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(
            request["messages"][2]["content"][0]["tool_use_id"],
            "call-1"
        );
    }

    #[test]
    fn response_preserves_native_tool_use_and_usage() {
        let response = serde_json::json!({
            "model": "claude-test",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 12, "output_tokens": 7},
            "content": [
                {"type": "text", "text": "Checking now."},
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "weather",
                    "input": {"city": "London"}
                }
            ]
        });

        let parsed = AnthropicProvider::parse_response(&response, "fallback").unwrap();
        assert_eq!(parsed.model, "claude-test");
        assert_eq!(parsed.usage.prompt_tokens, 12);
        assert_eq!(parsed.usage.completion_tokens, 7);
        assert_eq!(parsed.usage.total_tokens, 19);
        assert_eq!(parsed.finish_reason.as_deref(), Some("tool_use"));
        assert_eq!(parsed.message.text(), Some("Checking now."));
        assert_eq!(parsed.message.tool_calls().len(), 1);
        assert_eq!(parsed.message.tool_calls()[0].function.name, "weather");
        assert_eq!(
            parsed.message.tool_calls()[0].function.arguments,
            r#"{"city":"London"}"#
        );
    }

    #[test]
    fn response_without_content_is_rejected() {
        let error = AnthropicProvider::parse_response(
            &serde_json::json!({"error": {"type": "authentication_error"}}),
            "fallback",
        )
        .unwrap_err();
        assert!(error.to_string().contains("content"));
    }

    #[tokio::test]
    async fn chat_rejects_http_errors_and_health_checks_status() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
                "error": {"type": "rate_limit_error", "message": "slow down"}
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let mut provider = AnthropicProvider::new("test-key");
        provider.base_url = server.uri();
        let error = provider
            .chat(
                "claude-test",
                &[Message::user("hello")],
                &[],
                &CompletionOptions::default(),
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("429"));
        assert!(!provider.health_check().await.unwrap());
    }
}
