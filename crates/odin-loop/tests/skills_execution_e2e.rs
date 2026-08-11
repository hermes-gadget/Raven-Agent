//! E2E test: skills injected into the PLAN phase system prompt via LoopEngine.
//!
//! Creates a SkillRegistry with programmatic registration, attaches it to a
//! LoopEngine, and verifies the plan phase injects skills into the system prompt.

use std::sync::Arc;

use odin_core::types::*;
use odin_loop::phases::{Phase as _, PhaseContext};
use odin_loop::{ConfidenceScorer, GoalDecomposer, LoopEngine, PlanPhase, StateSummarizer};
use odin_skills::{Skill, SkillRegistry};

fn build_test_skills() -> SkillRegistry {
    let mut registry = SkillRegistry::new();

    registry.register(Skill {
        name: "code-review".into(),
        description: "Review code for bugs, style, and best practices".into(),
        content: "## Steps\n\n1. Read the diff\n2. Check for bugs\n3. Verify tests".into(),
        required_tools: vec!["file_read".into(), "git".into()],
        recommended_tools: vec![],
        enabled: true,
        source_path: None,
    });

    registry.register(Skill {
        name: "deploy-check".into(),
        description: "Verify deployment readiness".into(),
        content: "## Steps\n\n1. Check CI status\n2. Verify all tests pass\n3. Confirm environment config".into(),
        required_tools: vec!["shell".into()],
        recommended_tools: vec![],
        enabled: true,
        source_path: None,
    });

    registry
}

#[tokio::test]
async fn test_skills_injected_via_loop_engine() {
    let registry = Arc::new(build_test_skills());

    let _engine = LoopEngine::new()
        .with_skill_registry(registry.clone())
        .with_max_iterations(3);

    let task = AgentTask {
        id: TaskId::new_v4(),
        goal: "Deploy the latest build".to_string(),
        context: None,
        sub_tasks: vec![],
        success_criteria: vec![],
        max_iterations: 3,
        created_at: chrono::Utc::now(),
    };

    let mut state = odin_core::traits::LoopState {
        task: task.clone(),
        messages: vec![Message::system(
            "You are an AI agent. Follow the plan, execute carefully, and verify results.",
        )],
        tool_results: vec![],
        current_phase: LoopPhase::Plan,
        iteration: 0,
        retry_count: 0,
        history: vec![],
    };

    let context = PhaseContext {
        principal_id: uuid::Uuid::new_v4(),
        confidence_scorer: ConfidenceScorer::default(),
        decomposer: GoalDecomposer::default(),
        summarizer: StateSummarizer::default(),
        plan: None,
        model_name: String::new(),
        provider: None,
        escalation_provider: None,
        tool_registry: None,
        policy_engine: None,
        skill_registry: Some(registry.clone()),
        audit_logger: None,
        reliability_tracker: None,
        model_profile: None,
    };

    let plan_phase = PlanPhase::new(GoalDecomposer::default());
    let result = plan_phase.execute(&mut state, &context).await;
    assert!(result.is_ok(), "Plan phase should succeed");

    let first_msg = &state.messages[0];
    let text = first_msg.text().unwrap_or("");

    assert!(
        text.contains("Available Skills"),
        "System prompt must contain 'Available Skills' header"
    );

    assert!(
        text.contains("code-review"),
        "System prompt must mention 'code-review' skill by name"
    );
    assert!(
        text.contains("deploy-check"),
        "System prompt must mention 'deploy-check' skill by name"
    );

    assert!(
        text.contains("Review code for bugs, style, and best practices"),
        "System prompt must contain code-review description"
    );
    assert!(
        text.contains("Verify deployment readiness"),
        "System prompt must contain deploy-check description"
    );

    assert!(
        text.contains("[USE_SKILL: skill-name]"),
        "System prompt must explain how to invoke a skill"
    );
}
