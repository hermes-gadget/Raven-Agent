//! Tool registry — manages registered tools, adds/removes/gets tools, lists schemas.
//!
//! The [`ToolRegistry`] is the central point for tool management. It is
//! thread-safe (`Send + Sync`) and uses interior mutability so tools can
//! be registered dynamically.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::Tool;
use odin_core::types::ToolSchema;

/// Thread-safe registry of [`Tool`] instances.
///
/// Tools are stored as `Arc<dyn Tool>` so callers can clone the handle
/// cheaply without lifetime constraints.
pub struct ToolRegistry {
    tools: RwLock<HashMap<String, Arc<dyn Tool>>>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new tool.
    ///
    /// Returns an error if a tool with the same name is already registered.
    pub fn register(&self, tool: Box<dyn Tool>) -> OdinResult<()> {
        self.register_arc(Arc::from(tool))
    }

    /// Register an already shared tool handle.
    pub fn register_arc(&self, tool: Arc<dyn Tool>) -> OdinResult<()> {
        let name = tool.name().to_string();
        let mut tools = self
            .tools
            .write()
            .map_err(|e| OdinError::Internal(format!("ToolRegistry lock poisoned: {e}")))?;

        if tools.contains_key(&name) {
            return Err(OdinError::Tool {
                tool: "registry".into(),
                message: format!("Tool '{name}' is already registered"),
                source: None,
            });
        }

        tools.insert(name, tool);
        Ok(())
    }

    /// Build a registry restricted to the supplied names.
    ///
    /// An empty allow-list is intentionally fail-closed and produces an empty
    /// registry. Callers that need unrestricted access should pass the full
    /// catalog explicitly.
    pub fn scoped(&self, allowed: &[String]) -> OdinResult<Self> {
        let tools = self.all_tools();
        let known: HashSet<&str> = tools.iter().map(|tool| tool.name()).collect();
        let unknown: Vec<&str> = allowed
            .iter()
            .filter_map(|name| (!known.contains(name.as_str())).then_some(name.as_str()))
            .collect();
        if !unknown.is_empty() {
            return Err(OdinError::Validation(format!(
                "Unknown tool names in allow-list: {}",
                unknown.join(", ")
            )));
        }

        let scoped = Self::new();
        for tool in tools {
            if allowed.iter().any(|name| name == tool.name()) {
                scoped.register_arc(tool)?;
            }
        }
        Ok(scoped)
    }

    /// Get a registered tool by name.
    ///
    /// Returns `None` if no tool with that name exists.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        let tools = self
            .tools
            .read()
            .map_err(|e| OdinError::Internal(format!("ToolRegistry lock poisoned: {e}")))
            .ok()?;
        tools.get(name).cloned()
    }

    /// Remove a tool from the registry by name.
    ///
    /// Returns the tool if it was registered, or `None` otherwise.
    pub fn remove(&self, name: &str) -> Option<Arc<dyn Tool>> {
        let mut tools = self
            .tools
            .write()
            .map_err(|e| OdinError::Internal(format!("ToolRegistry lock poisoned: {e}")))
            .ok()?;
        tools.remove(name)
    }

    /// List schemas for all registered tools.
    pub fn list_schemas(&self) -> Vec<ToolSchema> {
        let tools = self
            .tools
            .read()
            .map_err(|e| OdinError::Internal(format!("ToolRegistry lock poisoned: {e}")))
            .ok();
        match tools {
            Some(t) => t.values().map(|t| t.schema()).collect(),
            None => vec![],
        }
    }

    /// Check whether a tool with the given name is registered.
    pub fn is_registered(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.read().map(|t| t.len()).unwrap_or(0)
    }

    /// Number of enabled (non-disabled) tools.
    ///
    /// Since disabled tools are never registered, this is equivalent to
    /// the total registered count.
    pub fn enabled_count(&self) -> usize {
        self.len()
    }

    /// Returns `true` if no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return all registered tools.
    pub fn all_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools
            .read()
            .map(|t| t.values().cloned().collect())
            .unwrap_or_default()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use chrono::Utc;
    use odin_core::error::OdinResult;
    use odin_core::traits::{Tool, ToolContext};
    use odin_core::types::{FunctionSchema, ToolResult, ToolSchema};
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Minimal test tool.
    struct EchoTool {
        name: String,
        called: AtomicBool,
    }

    impl EchoTool {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                called: AtomicBool::new(false),
            }
        }
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Echoes back the input"
        }

        fn schema(&self) -> ToolSchema {
            ToolSchema {
                schema_type: "function".into(),
                function: FunctionSchema {
                    name: self.name.clone(),
                    description: self.description().into(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Message to echo"
                            }
                        },
                        "required": ["message"]
                    }),
                },
            }
        }

        async fn execute(
            &self,
            args: serde_json::Value,
            _context: &ToolContext,
        ) -> OdinResult<ToolResult> {
            self.called.store(true, Ordering::SeqCst);
            let msg = args
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Ok(ToolResult {
                call_id: "test".into(),
                tool_name: self.name.clone(),
                success: true,
                output: msg,
                error: None,
                duration_ms: 0,
                timestamp: Utc::now(),
            })
        }
    }

    #[tokio::test]
    async fn test_register_and_get() {
        let registry = ToolRegistry::new();
        assert!(registry.is_empty());

        let tool = Box::new(EchoTool::new("echo"));
        registry.register(tool).unwrap();

        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
        assert!(registry.is_registered("echo"));
        assert!(registry.get("echo").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[tokio::test]
    async fn test_register_duplicate() {
        let registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool::new("echo"))).unwrap();
        let err = registry
            .register(Box::new(EchoTool::new("echo")))
            .unwrap_err();
        assert!(err.to_string().contains("already registered"));
    }

    #[tokio::test]
    async fn test_remove() {
        let registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool::new("echo"))).unwrap();
        let removed = registry.remove("echo");
        assert!(removed.is_some());
        assert!(registry.is_empty());
        assert!(registry.remove("echo").is_none());
    }

    #[tokio::test]
    async fn test_list_schemas() {
        let registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool::new("alpha"))).unwrap();
        registry.register(Box::new(EchoTool::new("beta"))).unwrap();
        let schemas = registry.list_schemas();
        assert_eq!(schemas.len(), 2);
        let names: Vec<&str> = schemas.iter().map(|s| s.function.name.as_str()).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
    }

    #[tokio::test]
    async fn test_execute_registered_tool() {
        let registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool::new("echo"))).unwrap();

        let tool = registry.get("echo").unwrap();
        let context = ToolContext {
            agent_id: Default::default(),
            session_id: Default::default(),
            working_dir: std::path::PathBuf::from("/tmp"),
            env: std::collections::HashMap::new(),
        };

        let args = serde_json::json!({"message": "hello world"});
        let result = tool.execute(args, &context).await.unwrap();
        assert!(result.success);
        assert_eq!(result.output, "hello world");
    }

    #[tokio::test]
    async fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ToolRegistry>();
    }

    #[test]
    fn test_scoped_registry_enforces_allow_list() {
        let registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool::new("alpha"))).unwrap();
        registry.register(Box::new(EchoTool::new("beta"))).unwrap();

        let scoped = registry.scoped(&["alpha".to_string()]).unwrap();
        assert!(scoped.is_registered("alpha"));
        assert!(!scoped.is_registered("beta"));
    }

    #[test]
    fn test_empty_scoped_registry_is_fail_closed() {
        let registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool::new("alpha"))).unwrap();

        let scoped = registry.scoped(&[]).unwrap();
        assert!(scoped.is_empty());
    }

    #[test]
    fn test_scoped_registry_rejects_unknown_tools() {
        let registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool::new("alpha"))).unwrap();

        let result = registry.scoped(&["missing".to_string()]);
        assert!(result.is_err());
        let error = result.err().unwrap();
        assert!(error.to_string().contains("missing"));
    }
}
