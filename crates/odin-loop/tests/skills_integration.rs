//! Integration test: skills loaded from disk, injected into PLAN phase, queryable via load_skill.
//!
//! This test verifies the full skills pipeline:
//! 1. Skills are loaded from a temp directory
//! 2. They are injected into the PLAN phase system prompt
//! 3. They are queryable via load_skill

use std::sync::Arc;

use odin_core::traits::LoopEngine as _;
use odin_core::types::*;
use odin_loop::LoopEngine;
use odin_loop::phases::Phase as _;
use odin_skills::SkillRegistry;

/// A test skill markdown file
fn test_skill_content(name: &str, desc: &str, tools: &[&str]) -> String {
    let tools_yaml = if tools.is_empty() {
        String::new()
    } else {
        format!("required_tools:\n  - {}\n", tools.join("\n  - "))
    };
    format!(
        r#"---
name: {name}
description: {desc}
{tools_yaml}enabled: true
---

## {name} Instructions

Follow these steps for {name}.
"#
    )
}

fn setup_test_skills_dir() -> (tempfile::TempDir, SkillRegistry) {
    let dir = tempfile::tempdir().unwrap();

    std::fs::write(
        dir.path().join("code-review.md"),
        test_skill_content(
            "code-review",
            "Review code for bugs and style",
            &["file_read", "git"],
        ),
    )
    .unwrap();

    std::fs::write(
        dir.path().join("git-workflow.md"),
        test_skill_content(
            "git-workflow",
            "Standard git branch/commit/PR workflow",
            &["git", "shell"],
        ),
    )
    .unwrap();

    let registry = SkillRegistry::load_from_dir(dir.path()).unwrap();
    assert_eq!(registry.len(), 2);

    (dir, registry)
}

#[tokio::test]
async fn test_skills_loaded_and_injected_into_plan_phase() {
    let (_dir, registry) = setup_test_skills_dir();
    let registry = Arc::new(registry);

    let _engine = LoopEngine::new()
        .with_skill_registry(registry.clone())
        .with_max_iterations(3);

    let task = AgentTask {
        id: TaskId::new_v4(),
        goal: "Test skills injection".to_string(),
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

    let context = odin_loop::phases::PhaseContext {
        principal_id: uuid::Uuid::new_v4(),
        confidence_scorer: odin_loop::ConfidenceScorer::default(),
        decomposer: odin_loop::GoalDecomposer::default(),
        summarizer: odin_loop::StateSummarizer::default(),
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

    let plan_phase = odin_loop::PlanPhase::new(odin_loop::GoalDecomposer::default());
    let result = plan_phase.execute(&mut state, &context).await;
    assert!(result.is_ok(), "Plan phase should succeed");

    let first_msg = &state.messages[0];
    let text = first_msg.text().unwrap_or("");
    assert!(
        text.contains("Available Skills"),
        "System prompt should contain 'Available Skills', got: {}...",
        text.chars().take(200).collect::<String>()
    );
    assert!(
        text.contains("code-review"),
        "System prompt should mention 'code-review' skill"
    );
    assert!(
        text.contains("git-workflow"),
        "System prompt should mention 'git-workflow' skill"
    );
    assert!(
        text.contains("[USE_SKILL: skill-name]"),
        "System prompt should explain how to use skills"
    );
}

#[tokio::test]
async fn test_load_skill_after_execution() {
    let (_dir, registry) = setup_test_skills_dir();
    let registry = Arc::new(registry);

    let engine = LoopEngine::new()
        .with_skill_registry(registry.clone())
        .with_max_iterations(3);

    let content = engine.load_skill("code-review");
    assert!(content.is_some(), "Should find code-review skill");
    let loaded = content.unwrap();
    assert!(
        loaded.contains("## code-review Instructions"),
        "Should contain skill instructions"
    );

    let missing = engine.load_skill("nonexistent");
    assert!(missing.is_none(), "Non-existent skill should return None");
}

#[tokio::test]
async fn test_without_skills_registry_does_not_crash() {
    let engine = LoopEngine::new().with_max_iterations(3);

    let task = AgentTask {
        id: TaskId::new_v4(),
        goal: "Test no skills".to_string(),
        context: None,
        sub_tasks: vec![],
        success_criteria: vec![],
        max_iterations: 3,
        created_at: chrono::Utc::now(),
    };

    let result = engine.execute_task(&task).await;
    assert!(result.is_ok(), "Engine should work without skills registry");
}

#[tokio::test]
async fn test_skills_list_via_registry() {
    let (_dir, registry) = setup_test_skills_dir();

    let all = registry.all();
    assert_eq!(all.len(), 2);

    let names: Vec<&str> = all.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"code-review"));
    assert!(names.contains(&"git-workflow"));

    let code_review = registry.get("code-review").unwrap();
    assert_eq!(code_review.description, "Review code for bugs and style");
    assert!(
        code_review
            .required_tools
            .contains(&"file_read".to_string())
    );
    assert!(code_review.required_tools.contains(&"git".to_string()));
    assert!(code_review.enabled);
}
