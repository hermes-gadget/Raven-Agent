//! Agent — wraps a LoopEngine, Provider, and Tool collection into a runnable unit.

use async_trait::async_trait;
use odin_core::error::OdinResult;
use odin_core::traits::{LoopEngine, Provider, Tool};
use odin_core::types::{AgentId, AgentTask, TaskResult};
use std::sync::Arc;
use uuid::Uuid;

/// An agent instance composed of a loop engine, a provider, and tools.
///
/// This is the smallest runnable unit in the Raven ecosystem. Each agent
/// has its own identity, engine settings, and tool capabilities.
#[derive(Clone)]
pub struct Agent {
    /// Unique identifier for this agent.
    pub id: AgentId,

    /// Human-readable name.
    pub name: String,

    /// The loop engine that drives this agent.
    engine: Arc<dyn LoopEngine>,

    /// The model provider this agent uses.
    provider: Arc<dyn Provider>,

    /// Tools available to this agent.
    tools: Vec<Arc<dyn Tool>>,
}

impl Agent {
    /// Create a new agent with the given components.
    pub fn new(
        name: impl Into<String>,
        engine: Arc<dyn LoopEngine>,
        provider: Arc<dyn Provider>,
        tools: Vec<Arc<dyn Tool>>,
    ) -> Self {
        Self::new_with_id(Uuid::new_v4(), name, engine, provider, tools)
    }

    /// Create an agent with a caller-provided stable identity.
    ///
    /// Production composition uses the same ID for the agent and loop engine so
    /// authorization, rate limiting, and audit correlation refer to one principal.
    pub fn new_with_id(
        id: AgentId,
        name: impl Into<String>,
        engine: Arc<dyn LoopEngine>,
        provider: Arc<dyn Provider>,
        tools: Vec<Arc<dyn Tool>>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            engine,
            provider,
            tools,
        }
    }

    /// Execute a task through the agent's loop engine.
    pub async fn execute_task(&self, task: &AgentTask) -> OdinResult<TaskResult> {
        tracing::info!(agent_id = %self.id, task_id = %task.id, "Agent executing task");
        self.engine.execute_task(task).await
    }

    /// Get a reference to the provider.
    pub fn provider(&self) -> &Arc<dyn Provider> {
        &self.provider
    }

    /// Get a reference to the tools.
    pub fn tools(&self) -> &[Arc<dyn Tool>] {
        &self.tools
    }

    /// Get the loop engine reference.
    pub fn engine(&self) -> &Arc<dyn LoopEngine> {
        &self.engine
    }

    /// Check if the agent has a tool with the given name.
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.iter().any(|t| t.name() == name)
    }
}

/// Trait for creating agent configurations.
#[async_trait]
pub trait AgentFactory: Send + Sync {
    /// Create a new agent.
    async fn create_agent(&self, name: &str) -> OdinResult<Agent>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mock loop engine for testing.
    struct MockEngine;

    #[async_trait]
    impl LoopEngine for MockEngine {
        async fn execute_task(&self, task: &AgentTask) -> OdinResult<TaskResult> {
            Ok(TaskResult {
                task_id: task.id,
                success: true,
                summary: "Mock task completed".into(),
                iterations: 1,
                tool_calls: 0,
                duration_ms: 0,
                sub_tasks: vec![],
                confidence: 1.0,
                error: None,
            })
        }

        async fn execute_phase(
            &self,
            _phase: odin_core::types::LoopPhase,
            _state: &mut odin_core::traits::LoopState,
        ) -> OdinResult<odin_core::traits::PhaseResult> {
            Err(odin_core::error::OdinError::Other(
                "mock engine does not expose phases".into(),
            ))
        }

        fn state_summary(&self) -> odin_core::types::StateSummary {
            odin_core::types::StateSummary {
                goal: String::new(),
                current_phase: odin_core::types::LoopPhase::Plan,
                completed_steps: vec![],
                pending_steps: vec![],
                last_action: None,
                last_result: None,
                errors: vec![],
                confidence: 1.0,
                token_usage: odin_core::types::TokenUsage::default(),
            }
        }

        fn confidence(&self) -> odin_core::types::ConfidenceScore {
            odin_core::types::ConfidenceScore::new(1.0)
        }
    }

    /// A mock provider for testing.
    struct MockProvider;

    #[async_trait]
    impl Provider for MockProvider {
        fn name(&self) -> &str {
            "mock"
        }

        async fn list_models(&self) -> OdinResult<Vec<odin_core::types::ModelInfo>> {
            Ok(vec![])
        }

        async fn chat(
            &self,
            _model: &str,
            _messages: &[odin_core::types::Message],
            _tools: &[odin_core::types::ToolSchema],
            _options: &odin_core::types::CompletionOptions,
        ) -> OdinResult<odin_core::types::ChatResponse> {
            Err(odin_core::error::OdinError::Other(
                "mock provider chat is not used".into(),
            ))
        }

        async fn chat_stream(
            &self,
            _model: &str,
            _messages: &[odin_core::types::Message],
            _tools: &[odin_core::types::ToolSchema],
            _options: &odin_core::types::CompletionOptions,
        ) -> OdinResult<Box<dyn odin_core::traits::ChatStream>> {
            Err(odin_core::error::OdinError::Other(
                "mock provider does not stream".into(),
            ))
        }

        async fn health_check(&self) -> OdinResult<bool> {
            Ok(true)
        }
    }

    #[test]
    fn test_agent_creation() {
        let engine = Arc::new(MockEngine);
        let provider = Arc::new(MockProvider);
        let agent = Agent::new("test-agent", engine, provider, vec![]);

        assert!(!agent.id.to_string().is_empty());
        assert_eq!(agent.name, "test-agent");
        assert!(agent.tools().is_empty());
    }

    #[test]
    fn test_agent_tool_inventory() {
        let engine = Arc::new(MockEngine);
        let provider = Arc::new(MockProvider);

        struct MockTool;
        #[async_trait]
        impl Tool for MockTool {
            fn name(&self) -> &str {
                "mock-tool"
            }
            fn description(&self) -> &str {
                "A mock tool"
            }
            fn schema(&self) -> odin_core::types::ToolSchema {
                odin_core::types::ToolSchema {
                    schema_type: "function".into(),
                    function: odin_core::types::FunctionSchema {
                        name: "mock-tool".into(),
                        description: "mock".into(),
                        parameters: serde_json::json!({}),
                    },
                }
            }
            async fn execute(
                &self,
                _args: serde_json::Value,
                _context: &odin_core::traits::ToolContext,
            ) -> OdinResult<odin_core::types::ToolResult> {
                Err(odin_core::error::OdinError::Other(
                    "mock tool execution is not used".into(),
                ))
            }
        }

        let agent = Agent::new("test", engine, provider, vec![Arc::new(MockTool)]);
        assert_eq!(agent.tools().len(), 1);
        assert!(agent.has_tool("mock-tool"));
    }

    #[tokio::test]
    async fn test_agent_executes_task() {
        let engine = Arc::new(MockEngine);
        let provider = Arc::new(MockProvider);
        let agent = Agent::new("executor", engine, provider, vec![]);

        let task = AgentTask {
            id: Uuid::new_v4(),
            goal: "Test".into(),
            context: None,
            sub_tasks: vec![],
            success_criteria: vec![],
            max_iterations: 5,
            created_at: chrono::Utc::now(),
        };

        let result = agent.execute_task(&task).await.unwrap();
        assert!(result.success);
        assert_eq!(result.summary, "Mock task completed");
    }
}
