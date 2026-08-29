//! Validated production composition for every Raven execution surface.

use std::path::PathBuf;
use std::sync::Arc;

use odin_core::config::{OdinConfig, ProviderConfig};
use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::{AuditLogger, Provider, Tool};
use odin_core::types::AgentId;
use odin_loop::{LoopEngine, SmallModelProfile};

use crate::Agent;

/// Production callers supported by the shared composition root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionSurface {
    Cli,
    Tui,
    Http,
    Discord,
    Scheduler,
    Orchestration,
}

impl ExecutionSurface {
    pub const ALL: [Self; 6] = [
        Self::Cli,
        Self::Tui,
        Self::Http,
        Self::Discord,
        Self::Scheduler,
        Self::Orchestration,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cli => "cli",
            Self::Tui => "tui",
            Self::Http => "http",
            Self::Discord => "discord",
            Self::Scheduler => "scheduler",
            Self::Orchestration => "orchestration",
        }
    }
}

/// Shared runtime resources attached consistently to every composed engine.
#[derive(Clone, Default)]
pub struct CompositionResources {
    pub policy_engine: Option<Arc<odin_permissions::PolicyEngine>>,
    pub tool_registry: Option<Arc<odin_tools::ToolRegistry>>,
    pub audit_logger: Option<Arc<dyn AuditLogger>>,
    pub reliability_tracker: Option<Arc<odin_tools::ReliabilityTracker>>,
}

/// Per-execution overrides that do not change validated configuration.
#[derive(Clone, Default)]
pub struct EngineBuildOptions {
    pub max_iterations: Option<u32>,
    pub principal_id: Option<AgentId>,
    pub provider: Option<Arc<dyn Provider>>,
    pub tool_registry: Option<Arc<odin_tools::ToolRegistry>>,
    pub audit_logger: Option<Arc<dyn AuditLogger>>,
}

/// Immutable, inspectable output of startup validation and model resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedComposition {
    pub provider_name: String,
    pub provider_type: String,
    pub model_name: String,
    pub planning_model: Option<String>,
    pub critique_model: Option<String>,
    pub fallback_providers: Vec<String>,
    pub escalation_model: Option<String>,
    pub model_profile: Option<String>,
}

/// The single production composition root for providers and loop engines.
///
/// Construct this exactly once after loading configuration, then clone it into
/// request handlers or worker tasks. Provider fallbacks and model selection are
/// resolved here rather than independently by each execution surface.
#[derive(Clone)]
pub struct ProductionComposition {
    provider: Arc<dyn Provider>,
    resolved: ResolvedComposition,
    max_iterations: u32,
    confidence_threshold: f64,
    skill_registry: Option<Arc<odin_skills::SkillRegistry>>,
    model_profile: Option<SmallModelProfile>,
    resources: CompositionResources,
}

impl ProductionComposition {
    /// Validate configuration and construct the configured provider chain once.
    pub fn from_config(config: &OdinConfig) -> OdinResult<Self> {
        let (provider_name, provider_config, model_name, fallback_providers) =
            validate_and_resolve(config)?;
        let provider =
            odin_providers::create_provider_chain(provider_config, &config.models.providers)?;
        let model_profile = SmallModelProfile::for_model(&model_name);
        let skill_registry = load_skill_registry(&config.agent.skills_dir)?;

        let resolved = ResolvedComposition {
            provider_name: provider_name.to_string(),
            provider_type: provider_config.provider_type.clone(),
            model_name,
            planning_model: optional_model(&config.models.planning_model),
            critique_model: optional_model(&config.models.critique_model),
            fallback_providers,
            escalation_model: config
                .models
                .escalation_model
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string),
            model_profile: model_profile.as_ref().map(|profile| profile.id.clone()),
        };

        Ok(Self {
            provider,
            resolved,
            max_iterations: config.agent.max_iterations,
            confidence_threshold: config.agent.confidence_threshold,
            skill_registry,
            model_profile,
            resources: CompositionResources::default(),
        })
    }

    /// Attach process-lifetime resources after startup validation.
    pub fn with_resources(mut self, resources: CompositionResources) -> Self {
        self.resources = resources;
        self
    }

    pub fn provider(&self) -> Arc<dyn Provider> {
        self.provider.clone()
    }

    pub fn resolved(&self) -> &ResolvedComposition {
        &self.resolved
    }

    pub fn default_max_iterations(&self) -> u32 {
        self.max_iterations
    }

    /// Build a loop engine whose provider/model/configuration came from this root.
    pub fn build_engine(
        &self,
        surface: ExecutionSurface,
        options: EngineBuildOptions,
    ) -> LoopEngine {
        let provider = options.provider.unwrap_or_else(|| self.provider.clone());
        let max_iterations = options.max_iterations.unwrap_or(self.max_iterations);
        let principal_id = options.principal_id.unwrap_or_else(uuid::Uuid::new_v4);
        let tool_registry = options
            .tool_registry
            .or_else(|| self.resources.tool_registry.clone());
        let audit_logger = options
            .audit_logger
            .or_else(|| self.resources.audit_logger.clone());

        tracing::debug!(
            surface = surface.as_str(),
            provider = %self.resolved.provider_name,
            model = %self.resolved.model_name,
            "Building agent engine from validated composition"
        );

        let mut engine = LoopEngine::new()
            .with_principal_id(principal_id)
            .with_provider(provider)
            .with_model_name(self.resolved.model_name.clone())
            .with_max_iterations(max_iterations)
            .with_confidence_thresholds(
                self.confidence_threshold,
                self.confidence_threshold.max(0.8),
            );

        if let Some(planning_model) = self.resolved.planning_model.clone() {
            engine = engine.with_planning_model_name(planning_model);
        }
        if let Some(critique_model) = self.resolved.critique_model.clone() {
            engine = engine.with_critique_model_name(critique_model);
        }

        if let Some(policy_engine) = self.resources.policy_engine.clone() {
            engine = engine.with_policy_engine(policy_engine);
        }
        if let Some(tool_registry) = tool_registry {
            engine = engine.with_tool_registry(tool_registry);
        }
        if let Some(audit_logger) = audit_logger {
            engine = engine.with_audit_logger(audit_logger);
        }
        if let Some(reliability_tracker) = self.resources.reliability_tracker.clone() {
            engine = engine.with_reliability_tracker(reliability_tracker);
        }
        if let Some(skill_registry) = self.skill_registry.clone() {
            engine = engine.with_skill_registry(skill_registry);
        }
        if let Some(model_profile) = self.model_profile.clone() {
            engine = engine.with_small_model_profile(model_profile);
        }
        if let Some(escalation_model) = self.resolved.escalation_model.clone() {
            engine = engine
                .with_escalation_provider(self.provider.clone())
                .with_escalation_model_name(escalation_model);
        }

        engine
    }

    /// Build an agent and keep its runtime identity aligned with the engine.
    pub fn build_agent(
        &self,
        surface: ExecutionSurface,
        name: impl Into<String>,
        mut options: EngineBuildOptions,
    ) -> Agent {
        let principal_id = options.principal_id.unwrap_or_else(uuid::Uuid::new_v4);
        options.principal_id = Some(principal_id);
        let provider = options
            .provider
            .clone()
            .unwrap_or_else(|| self.provider.clone());
        let registry = options
            .tool_registry
            .clone()
            .or_else(|| self.resources.tool_registry.clone());
        let tools: Vec<Arc<dyn Tool>> = registry
            .as_ref()
            .map(|registry| {
                registry
                    .list_schemas()
                    .iter()
                    .filter_map(|schema| registry.get(&schema.function.name))
                    .collect()
            })
            .unwrap_or_default();
        let engine = self.build_engine(surface, options);

        Agent::new_with_id(principal_id, name, Arc::new(engine), provider, tools)
    }
}

fn validate_and_resolve(
    config: &OdinConfig,
) -> OdinResult<(&str, &ProviderConfig, String, Vec<String>)> {
    if config.agent.max_iterations == 0 {
        return Err(OdinError::Config(
            "agent.max_iterations must be greater than zero".into(),
        ));
    }
    if !(0.0..=1.0).contains(&config.agent.confidence_threshold) {
        return Err(OdinError::Config(
            "agent.confidence_threshold must be between 0.0 and 1.0".into(),
        ));
    }

    let provider_name = config.models.default_provider.trim();
    if provider_name.is_empty() {
        return Err(OdinError::Config(
            "models.default_provider must not be empty".into(),
        ));
    }
    let provider_config = config.models.providers.get(provider_name).ok_or_else(|| {
        OdinError::Config(format!(
            "Default provider '{provider_name}' is not present in models.providers"
        ))
    })?;

    let model_name = config
        .models
        .default_model
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .or_else(|| {
            provider_config
                .default_model
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
        })
        .ok_or_else(|| {
            OdinError::Config(format!(
                "No model configured for provider '{provider_name}'; set models.default_model or models.providers.{provider_name}.default_model"
            ))
        })?
        .to_string();

    for (field, value) in [
        (
            "models.planning_model",
            config.models.planning_model.as_deref(),
        ),
        (
            "models.critique_model",
            config.models.critique_model.as_deref(),
        ),
        (
            "models.escalation_model",
            config.models.escalation_model.as_deref(),
        ),
    ] {
        if value.is_some_and(|value| value.trim().is_empty()) {
            return Err(OdinError::Config(format!("{field} must not be empty")));
        }
    }

    let fallback_providers = provider_config.fallback_chain.clone().unwrap_or_default();
    let mut seen = std::collections::HashSet::new();
    for fallback_name in &fallback_providers {
        if fallback_name == provider_name {
            return Err(OdinError::Config(format!(
                "Provider '{provider_name}' cannot fall back to itself"
            )));
        }
        if !seen.insert(fallback_name) {
            return Err(OdinError::Config(format!(
                "Fallback provider '{fallback_name}' is configured more than once"
            )));
        }
        if !config.models.providers.contains_key(fallback_name) {
            return Err(OdinError::Config(format!(
                "Fallback provider '{fallback_name}' is not present in models.providers"
            )));
        }
    }

    Ok((
        provider_name,
        provider_config,
        model_name,
        fallback_providers,
    ))
}

fn load_skill_registry(path: &str) -> OdinResult<Option<Arc<odin_skills::SkillRegistry>>> {
    let path = PathBuf::from(shellexpand::tilde(path).to_string());
    if !path.exists() {
        return Ok(None);
    }
    odin_skills::SkillRegistry::load_from_dir(&path)
        .map(Arc::new)
        .map(Some)
}

fn optional_model(model: &Option<String>) -> Option<String> {
    model
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use odin_core::traits::ChatStream;
    use odin_core::traits::LoopEngine as _;
    use odin_core::types::{
        AgentTask, ChatResponse, CompletionOptions, Message, ModelInfo, TaskId, TokenUsage,
        ToolSchema,
    };
    use std::sync::Mutex;

    fn provider(provider_type: &str, model: Option<&str>) -> ProviderConfig {
        ProviderConfig {
            provider_type: provider_type.into(),
            base_url: Some("http://localhost:11434/v1".into()),
            api_key: None,
            api_key_env: None,
            default_model: model.map(str::to_string),
            headers: Default::default(),
            timeout_secs: 120,
            max_retries: 3,
            fallback_chain: None,
            health_check_interval_secs: 0,
            circuit_breaker_threshold: 0,
        }
    }

    fn valid_config() -> OdinConfig {
        let mut config = OdinConfig::default();
        config.models.default_provider = "primary".into();
        config.models.planning_model = Some("planning-model".into());
        config.models.critique_model = Some("critique-model".into());
        config.models.escalation_model = Some("escalation-model".into());
        let mut primary = provider("local", Some("qwen2.5-coder:7b"));
        primary.fallback_chain = Some(vec!["backup".into()]);
        config.models.providers.insert("primary".into(), primary);
        config
            .models
            .providers
            .insert("backup".into(), provider("local", Some("backup-model")));
        config
    }

    #[test]
    fn all_surfaces_share_one_resolved_provider_model_and_fallback_chain() {
        let composition = ProductionComposition::from_config(&valid_config()).unwrap();
        let expected = composition.resolved().clone();

        for surface in ExecutionSurface::ALL {
            let _engine = composition.build_engine(surface, EngineBuildOptions::default());
            assert_eq!(composition.resolved(), &expected, "surface {surface:?}");
        }
        assert_eq!(expected.provider_name, "primary");
        assert_eq!(expected.model_name, "qwen2.5-coder:7b");
        assert_eq!(expected.planning_model.as_deref(), Some("planning-model"));
        assert_eq!(expected.critique_model.as_deref(), Some("critique-model"));
        assert_eq!(
            expected.escalation_model.as_deref(),
            Some("escalation-model")
        );
        assert_eq!(expected.fallback_providers, vec!["backup"]);
        assert_eq!(
            expected.model_profile.as_deref(),
            Some("ollama-qwen2.5-coder-7b")
        );
    }

    #[test]
    fn startup_validation_rejects_missing_models_and_fallbacks() {
        let mut missing_model = valid_config();
        missing_model
            .models
            .providers
            .get_mut("primary")
            .unwrap()
            .default_model = None;
        let error = ProductionComposition::from_config(&missing_model)
            .err()
            .expect("missing model must fail");
        assert!(error.to_string().contains("No model configured"));

        let mut missing_fallback = valid_config();
        missing_fallback.models.providers.remove("backup");
        let error = ProductionComposition::from_config(&missing_fallback)
            .err()
            .expect("missing fallback must fail");
        assert!(error.to_string().contains("Fallback provider 'backup'"));
    }

    struct RecordingProvider {
        models: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl Provider for RecordingProvider {
        fn name(&self) -> &str {
            "recording"
        }

        async fn list_models(&self) -> OdinResult<Vec<ModelInfo>> {
            Ok(vec![])
        }

        async fn chat(
            &self,
            model: &str,
            messages: &[Message],
            _tools: &[ToolSchema],
            _options: &CompletionOptions,
        ) -> OdinResult<ChatResponse> {
            self.models.lock().unwrap().push(model.to_string());
            let prompt = messages.last().and_then(Message::text).unwrap_or_default();
            let content = if prompt.contains("Return only JSON using this schema") {
                r#"{"sub_tasks":[{"id":"task_1","description":"perform work"}]}"#
            } else if prompt.contains("Evaluate the last action") {
                "The action succeeded. Confidence: 0.90"
            } else if prompt.contains("Has the goal been achieved?") {
                "VERIFIED. Confidence: 0.90"
            } else {
                "I performed the requested work and recorded the result."
            };
            Ok(ChatResponse {
                message: Message::assistant(content),
                usage: TokenUsage::default(),
                finish_reason: Some("stop".into()),
                model: model.into(),
            })
        }

        async fn chat_stream(
            &self,
            _model: &str,
            _messages: &[Message],
            _tools: &[ToolSchema],
            _options: &CompletionOptions,
        ) -> OdinResult<Box<dyn ChatStream>> {
            Err(OdinError::Other(
                "recording provider does not stream".into(),
            ))
        }

        async fn health_check(&self) -> OdinResult<bool> {
            Ok(true)
        }
    }

    #[tokio::test]
    async fn every_surface_routes_phase_models_through_the_same_builder() {
        let composition = ProductionComposition::from_config(&valid_config()).unwrap();

        for surface in ExecutionSurface::ALL {
            let provider = Arc::new(RecordingProvider {
                models: Mutex::new(Vec::new()),
            });
            let engine = composition.build_engine(
                surface,
                EngineBuildOptions {
                    max_iterations: Some(2),
                    provider: Some(provider.clone()),
                    ..Default::default()
                },
            );
            let task = AgentTask {
                id: TaskId::new_v4(),
                goal: "perform work".into(),
                context: None,
                sub_tasks: vec![],
                success_criteria: vec![],
                max_iterations: 2,
                created_at: chrono::Utc::now(),
            };

            let _ = engine.execute_task(&task).await.unwrap();
            assert_eq!(
                provider.models.lock().unwrap().as_slice(),
                [
                    "planning-model",
                    "qwen2.5-coder:7b",
                    "critique-model",
                    "critique-model",
                ],
                "surface {surface:?}"
            );
        }
    }
}
