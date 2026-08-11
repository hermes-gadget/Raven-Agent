//! Odin Tools — Tool system for AI agents.
//!
//! This crate provides the tool abstraction layer: a `ToolRegistry` that
//! manages registered tools, a `Sandbox` for filesystem boundary enforcement,
//! a `ToolCatalog` for category-based tool organization, and a set of built-in
//! tools (file, shell, web, git) implementing the `Tool` trait from `odin_core`.

pub mod builtins;
pub mod catalog;
pub mod dry_run;
pub mod reliability;
pub mod sandbox;
pub mod tool;
pub mod validator;

pub use catalog::{CatalogEntry, ToolCatalog};
pub use dry_run::{DryRunConfig, DryRunTool};
pub use reliability::*;
pub use sandbox::Sandbox;
pub use tool::ToolRegistry;
pub use validator::*;

/// Build Raven Agent's standard built-in tool registry.
///
/// When `enabled` is provided, only exact names in that list are registered.
/// Passing `None` registers the full catalog for inspection and diagnostics.
pub fn builtin_registry(
    sandbox: std::sync::Arc<Sandbox>,
    enabled: Option<&[String]>,
) -> odin_core::error::OdinResult<ToolRegistry> {
    use odin_core::traits::Tool;

    let registry = ToolRegistry::new();
    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(builtins::file::FileRead::new(sandbox.clone())),
        Box::new(builtins::file::FileWrite::new(sandbox.clone())),
        Box::new(builtins::shell::Shell::with_sandbox(sandbox.clone())),
        Box::new(builtins::git::Git::with_sandbox(sandbox.clone())),
        Box::new(builtins::web::WebFetch::new()),
        Box::new(builtins::web::WebSearch::new()),
        Box::new(builtins::web::HttpRequest::new()),
        Box::new(builtins::system::SystemInfo::new()),
        Box::new(builtins::system::DiskUsage::new()),
        Box::new(builtins::data::JsonExtract::new()),
        Box::new(builtins::github::GithubIssueCreate::default()),
        Box::new(builtins::github::GithubIssueSearch::default()),
        Box::new(builtins::github::GithubPrCreate::default()),
        Box::new(builtins::github::GithubPrStatus::default()),
        Box::new(builtins::github::GithubActionsStatus::default()),
        Box::new(builtins::utility::FileList::new(sandbox.clone())),
        Box::new(builtins::utility::FileDelete::new(sandbox.clone())),
        Box::new(builtins::utility::FileExists::new(sandbox)),
        Box::new(builtins::utility::EnvVar),
        Box::new(builtins::utility::TimeNow),
        Box::new(builtins::utility::RandomNumber),
        Box::new(builtins::utility::JsonValidate),
        Box::new(builtins::utility::TextSearch),
        Box::new(builtins::utility::ProcessList),
        Box::new(builtins::utility::NetworkPing),
    ];

    for tool in tools {
        if enabled.is_some_and(|names| !names.iter().any(|name| name == tool.name())) {
            continue;
        }
        registry.register(tool)?;
    }
    Ok(registry)
}
