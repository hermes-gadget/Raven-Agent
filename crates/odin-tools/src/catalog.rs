//! Tool catalog — organizes tools by category for discovery and display.
//!
//! The [`ToolCatalog`] indexes registered tools by their capability tags,
//! providing fast lookup by category, tag, name, and safety classification.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::tool::ToolRegistry;

/// A tool entry in the catalog with full metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub name: String,
    pub description: String,
    pub category: String,
    pub tags: Vec<String>,
    pub is_safe: bool,
    pub is_dangerous: bool,
    pub requires_approval: bool,
}

/// Category grouping of tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryGroup {
    pub name: String,
    pub description: String,
    pub tools: Vec<CatalogEntry>,
}

/// The tool catalog — organized view of all registered tools.
///
/// Groups tools by their primary category tag and provides
/// fast lookup by name and tag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCatalog {
    /// All tools indexed by name for O(1) lookup.
    pub by_name: HashMap<String, CatalogEntry>,
    /// Tools grouped by primary category.
    pub by_category: HashMap<String, CategoryGroup>,
    /// All unique tags across all tools.
    pub all_tags: Vec<String>,
    /// Total number of tools in the catalog.
    pub total: usize,
}

impl ToolCatalog {
    /// Build a catalog from all tools registered in the given [`ToolRegistry`].
    ///
    /// Each tool is placed into the category derived from its first
    /// capability tag. Additional entries are created for secondary tags.
    pub fn from_registry(registry: &ToolRegistry) -> Self {
        let tools = registry.all_tools();
        let mut by_name = HashMap::with_capacity(tools.len());
        let mut by_category: HashMap<String, CategoryGroup> = HashMap::new();
        let mut all_tags: Vec<String> = Vec::new();

        for tool in &tools {
            let name = tool.name().to_string();
            let tags = tool.capability_tags();
            let primary_category = tags.first().cloned().unwrap_or_else(|| "other".into());

            // Collect all unique tags
            for tag in &tags {
                if !all_tags.contains(tag) {
                    all_tags.push(tag.clone());
                }
            }

            let entry = CatalogEntry {
                name: name.clone(),
                description: tool.description().to_string(),
                category: primary_category.clone(),
                tags: tags.clone(),
                is_safe: tool.is_safe(),
                is_dangerous: tool.is_dangerous(),
                requires_approval: tool.requires_approval(),
            };

            by_name.insert(name, entry.clone());

            // Add to primary category
            let group = by_category
                .entry(primary_category.clone())
                .or_insert_with(|| CategoryGroup {
                    name: primary_category.clone(),
                    description: category_description(&primary_category),
                    tools: Vec::new(),
                });
            group.tools.push(entry);
        }

        // Sort tools within each category alphabetically
        for group in by_category.values_mut() {
            group.tools.sort_by(|a, b| a.name.cmp(&b.name));
        }

        all_tags.sort();

        Self {
            total: tools.len(),
            by_name,
            by_category,
            all_tags,
        }
    }

    /// Look up a single tool by name.
    pub fn get(&self, name: &str) -> Option<&CatalogEntry> {
        self.by_name.get(name)
    }

    /// List all tools in a specific category.
    pub fn by_category(&self, category: &str) -> Option<&CategoryGroup> {
        self.by_category.get(category)
    }

    /// List all category names.
    pub fn categories(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.by_category.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// List tools matching a specific tag.
    pub fn by_tag(&self, tag: &str) -> Vec<&CatalogEntry> {
        self.by_name
            .values()
            .filter(|e| e.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// List all safe tools.
    pub fn safe_tools(&self) -> Vec<&CatalogEntry> {
        self.by_name.values().filter(|e| e.is_safe).collect()
    }

    /// List all dangerous tools.
    pub fn dangerous_tools(&self) -> Vec<&CatalogEntry> {
        self.by_name.values().filter(|e| e.is_dangerous).collect()
    }
}

/// Human-readable descriptions for well-known categories.
fn category_description(category: &str) -> String {
    match category {
        "filesystem" => "File system read/write operations".into(),
        "shell" => "Shell command execution".into(),
        "web" => "Web and HTTP operations".into(),
        "version-control" => "Version control (git) operations".into(),
        "github" => "GitHub API and repository operations".into(),
        "diagnostic" => "System diagnostic and health checks".into(),
        "data" => "Data transformation and query tools".into(),
        "mcp" => "External MCP connector tools".into(),
        "automation" => "Automated workflows and notifications".into(),
        "security" => "Security scanning and auditing tools".into(),
        "media" => "Image, audio, and video tools".into(),
        other => format!("{other} tools"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::file::{FileRead, FileWrite};
    use crate::builtins::git::Git;
    use crate::builtins::shell::Shell;
    use crate::builtins::web::{WebFetch, WebSearch};
    use crate::sandbox::Sandbox;
    use std::sync::Arc;

    fn builtin_registry() -> ToolRegistry {
        let registry = ToolRegistry::new();
        let sandbox = Arc::new(Sandbox::default());
        registry
            .register(Box::new(FileRead::new(sandbox.clone())))
            .unwrap();
        registry
            .register(Box::new(FileWrite::new(sandbox)))
            .unwrap();
        registry.register(Box::new(Shell::new())).unwrap();
        registry.register(Box::new(WebFetch::new())).unwrap();
        registry.register(Box::new(WebSearch::new())).unwrap();
        registry.register(Box::new(Git::new())).unwrap();
        registry
    }

    #[test]
    fn test_catalog_from_registry() {
        let registry = builtin_registry();
        let catalog = ToolCatalog::from_registry(&registry);

        assert_eq!(catalog.total, 6);
        assert_eq!(catalog.by_name.len(), 6);
    }

    #[test]
    fn test_catalog_categories() {
        let registry = builtin_registry();
        let catalog = ToolCatalog::from_registry(&registry);

        let cats = catalog.categories();
        // Expected: filesystem, shell, version-control, web
        assert!(
            cats.contains(&"filesystem"),
            "categories should include 'filesystem', got: {:?}",
            cats
        );
        assert!(
            cats.contains(&"shell"),
            "should include 'shell', got: {:?}",
            cats
        );
        assert!(
            cats.contains(&"web"),
            "should include 'web', got: {:?}",
            cats
        );
    }

    #[test]
    fn test_catalog_by_name() {
        let registry = builtin_registry();
        let catalog = ToolCatalog::from_registry(&registry);

        let entry = catalog.get("shell").expect("shell should be in catalog");
        assert_eq!(entry.name, "shell");
        assert!(entry.is_dangerous);
        assert!(entry.requires_approval);
    }

    #[test]
    fn test_catalog_by_category() {
        let registry = builtin_registry();
        let catalog = ToolCatalog::from_registry(&registry);

        let filesystem = catalog
            .by_category("filesystem")
            .expect("should have filesystem category");
        let names: Vec<&str> = filesystem.tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"file_read"));
        assert!(names.contains(&"file_write"));
    }

    #[test]
    fn test_catalog_by_tag() {
        let registry = builtin_registry();
        let catalog = ToolCatalog::from_registry(&registry);

        let dangerous = catalog.by_tag("dangerous");
        let names: Vec<&str> = dangerous.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"shell"));
        assert!(names.contains(&"git"));
    }

    #[test]
    fn test_catalog_safe_and_dangerous() {
        let registry = builtin_registry();
        let catalog = ToolCatalog::from_registry(&registry);

        let safe = catalog.safe_tools();
        let dangerous = catalog.dangerous_tools();

        // file_read, web_fetch, web_search are safe
        assert!(safe.iter().any(|e| e.name == "file_read"));
        assert!(safe.iter().any(|e| e.name == "web_fetch"));

        // shell, git, file_write are dangerous
        assert!(dangerous.iter().any(|e| e.name == "shell"));
        assert!(dangerous.iter().any(|e| e.name == "git"));
    }

    #[test]
    fn test_catalog_json_roundtrip() {
        let registry = builtin_registry();
        let catalog = ToolCatalog::from_registry(&registry);

        let json = serde_json::to_string(&catalog).unwrap();
        assert!(json.contains("file_read"));
        assert!(json.contains("by_category"));

        let parsed: ToolCatalog = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.total, catalog.total);
    }
}
