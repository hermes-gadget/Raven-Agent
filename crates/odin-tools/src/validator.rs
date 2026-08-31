//! Tool validation harness.
//!
//! Provides [`ToolValidator`] for running schema, argument, and
//! permission checks against tools implementing the [`Tool`] trait,
use std::collections::HashMap;

use odin_core::traits::Tool;
use odin_core::types::ToolSchema;
use serde::{Deserialize, Serialize};

use crate::tool::ToolRegistry;

/// Result of running one or more validation checks against a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Name of the tool that was validated.
    pub tool_name: String,
    /// Names of checks that passed.
    pub passed: Vec<String>,
    /// Names of checks that failed.
    pub failed: Vec<String>,
    /// Warning messages (non-blocking concerns).
    pub warnings: Vec<String>,
    /// Composite score from 0.0 (all failed) to 1.0 (all passed).
    pub score: f64,
}

impl ValidationReport {
    fn new(tool_name: &str) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            passed: Vec::new(),
            failed: Vec::new(),
            warnings: Vec::new(),
            score: 1.0,
        }
    }

    fn pass(&mut self, check: &str) {
        self.passed.push(check.to_string());
        self.recalc_score();
    }

    fn fail(&mut self, check: &str) {
        self.failed.push(check.to_string());
        self.recalc_score();
    }

    fn warn(&mut self, msg: &str) {
        self.warnings.push(msg.to_string());
    }

    fn recalc_score(&mut self) {
        let total = self.passed.len() + self.failed.len();
        if total == 0 {
            self.score = 1.0;
        } else {
            self.score = self.passed.len() as f64 / total as f64;
        }
    }
}

/// Validates tools for schema correctness, argument conformance, and
/// permission / safety policies.
pub struct ToolValidator;

impl ToolValidator {
    /// Validate the schema of a single tool.
    ///
    /// Checks performed:
    /// - Tool name is non-empty.
    /// - Tool description is non-empty.
    /// - Schema parameters are a valid JSON object.
    /// - Required parameters are documented (every entry in the
    ///   `required` array has a corresponding `properties` entry).
    pub fn validate_schema(tool: &dyn Tool) -> ValidationReport {
        let mut report = ValidationReport::new(tool.name());

        // ── name non-empty ──────────────────────────────────────
        if tool.name().is_empty() {
            report.fail("name is non-empty");
        } else {
            report.pass("name is non-empty");
        }

        // ── description non-empty ───────────────────────────────
        if tool.description().is_empty() {
            report.fail("description is non-empty");
        } else {
            report.pass("description is non-empty");
        }

        // ── schema parameters are valid JSON object ─────────────
        let schema = tool.schema();
        let params = &schema.function.parameters;
        if !params.is_object() {
            report.fail("schema parameters are a JSON object");
        } else {
            report.pass("schema parameters are a JSON object");

            // ── type field present if object ────────────────────
            let type_val = params.get("type").and_then(|v| v.as_str());
            match type_val {
                Some("object") => report.pass("schema type is 'object'"),
                _ => report.warn("schema type is not 'object' or is missing"),
            }

            // ── required params documented ─────────────────────
            let required = params
                .get("required")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|f| f.as_str().map(String::from))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            if required.is_empty() {
                report.warn("no required parameters declared in schema");
            } else {
                report.pass("required parameters are declared");

                let properties = params
                    .get("properties")
                    .and_then(|v| v.as_object())
                    .map(|m| m.keys().cloned().collect::<Vec<_>>())
                    .unwrap_or_default();

                let mut all_documented = true;
                for req in &required {
                    if !properties.contains(req) {
                        report.fail(&format!(
                            "required field '{}' has a corresponding property definition",
                            req
                        ));
                        all_documented = false;
                    }
                }
                if all_documented {
                    report.pass("all required parameters have property definitions");
                }
            }
        }

        report
    }

    /// Validate arguments against a tool's schema.
    ///
    /// Checks performed:
    /// - Arguments are a valid JSON object.
    /// - All required schema fields are present in the arguments.
    pub fn validate_args(tool: &dyn Tool, args: serde_json::Value) -> ValidationReport {
        let mut report = ValidationReport::new(tool.name());

        // Use the trait default validate_args internally
        match tool.validate_args(&args) {
            Ok(()) => {
                report.pass("arguments pass schema validation");
            }
            Err(e) => {
                report.fail(&format!("arguments pass schema validation: {e}"));
            }
        }

        report
    }

    /// Validate permissions and safety attributes of a tool.
    ///
    /// Checks performed:
    /// - If the tool has `"dangerous"` in its capability tags, then
    ///   `requires_approval()` must return `true` and `is_safe()`
    ///   must return `false`.
    /// - If the tool has `"safe"` in its capability tags, then
    ///   `is_safe()` must return `true`.
    /// - If the tool has `"filesystem"` in its capability tags, then
    ///   `is_safe()` should return `true` (sandbox-gated filesystem
    ///   tools are safe within boundaries).
    pub fn validate_permissions(tool: &dyn Tool) -> ValidationReport {
        let mut report = ValidationReport::new(tool.name());
        let tags = tool.capability_tags();
        let has_dangerous = tags.iter().any(|tag| tag == "dangerous");
        let has_safe = tags.iter().any(|tag| tag == "safe");
        let has_filesystem = tags.iter().any(|tag| tag == "filesystem");

        // ── dangerous tools ─────────────────────────────────────
        if has_dangerous {
            if tool.requires_approval() {
                report.pass("dangerous tool requires approval");
            } else {
                report.fail("dangerous tool requires approval");
            }
            if !tool.is_safe() {
                report.pass("dangerous tool is not marked safe");
            } else {
                report.fail("dangerous tool is not marked safe");
            }
        }

        // ── safe tools ──────────────────────────────────────────
        if has_safe {
            if tool.is_safe() {
                report.pass("safe tool is marked safe");
            } else {
                report.fail("safe tool is marked safe");
            }
        }

        // ── filesystem tools ────────────────────────────────────
        if has_filesystem {
            if tool.is_safe() {
                report.pass("filesystem tool is sandbox-safe");
            } else {
                report.warn("filesystem tool is not marked as sandbox-safe");
            }
        }

        // ── is_dangerous() consistency ──────────────────────────
        if tool.is_dangerous() && !has_dangerous {
            report.warn("tool is_dangerous() returns true but 'dangerous' tag is missing");
        }
        if has_dangerous && !tool.is_dangerous() {
            report.warn("tool has 'dangerous' tag but is_dangerous() returns false");
        }

        // ── capability tags non-empty ───────────────────────────
        if tags.is_empty() {
            report.warn("tool has no capability tags");
        } else {
            report.pass("capability tags are present");
        }

        report
    }

    /// Run all validation checks (schema, args, permissions) on every
    /// tool registered in the given [`ToolRegistry`].
    ///
    /// Returns one [`ValidationReport`] per tool, with checks merged
    /// from all three validators.  For argument validation, an empty
    /// JSON object `{}` is passed.
    pub fn validate_all(registry: &ToolRegistry) -> Vec<ValidationReport> {
        let tools = registry.all_tools();
        let mut reports = Vec::with_capacity(tools.len());

        for tool in &tools {
            let schema_report = Self::validate_schema(tool.as_ref());
            let args_report = Self::validate_args(tool.as_ref(), serde_json::json!({}));
            let perm_report = Self::validate_permissions(tool.as_ref());

            let mut combined = ValidationReport::new(tool.name());

            // Merge passed
            combined.passed.extend(schema_report.passed);
            combined.passed.extend(args_report.passed);
            combined.passed.extend(perm_report.passed);

            // Merge failed
            combined.failed.extend(schema_report.failed);
            combined.failed.extend(args_report.failed);
            combined.failed.extend(perm_report.failed);

            // Merge warnings
            combined.warnings.extend(schema_report.warnings);
            combined.warnings.extend(args_report.warnings);
            combined.warnings.extend(perm_report.warnings);

            combined.recalc_score();
            reports.push(combined);
        }

        // Append duplicate-detection warnings into the matching reports
        let dup_reports = Self::detect_duplicates(registry);
        for dup in &dup_reports {
            if let Some(report) = reports.iter_mut().find(|r| r.tool_name == dup.tool_name) {
                for w in &dup.warnings {
                    if !report.warnings.contains(w) {
                        report.warnings.push(w.clone());
                    }
                }
            } else {
                // Tool with duplicates not in the main list — add its report
                reports.push(dup.clone());
            }
        }

        reports
    }

    /// Detect duplicate tool registrations in a [`ToolRegistry`].
    ///
    /// Checks for:
    /// - Two or more tools registered with the same name
    /// - Two or more tools with identical function schemas (matching
    ///   description and parameters, regardless of name)
    ///
    /// Returns one [`ValidationReport`] per affected tool, with warnings
    /// describing the duplicate.
    pub fn detect_duplicates(registry: &ToolRegistry) -> Vec<ValidationReport> {
        let tools = registry.all_tools();
        let n = tools.len();
        let mut report_map: HashMap<String, ValidationReport> = HashMap::new();

        // ── Phase 1: check for duplicate names ─────────────────
        let mut name_groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, tool) in tools.iter().enumerate() {
            name_groups
                .entry(tool.name().to_string())
                .or_default()
                .push(i);
        }
        for (name, indices) in &name_groups {
            if indices.len() > 1 {
                let msg = format!(
                    "Duplicate tool name '{}': {} registrations found",
                    name,
                    indices.len()
                );
                for &i in indices {
                    report_map
                        .entry(tools[i].name().to_string())
                        .or_insert_with(|| {
                            let mut r = ValidationReport::new(tools[i].name());
                            r.fail("duplicate name or schema check");
                            r
                        })
                        .warn(&msg);
                }
            }
        }

        // ── Phase 2: check for identical function schemas ──────
        // Compare every pair — O(n²) but n is small in practice
        for i in 0..n {
            for j in (i + 1)..n {
                let ti = &tools[i];
                let tj = &tools[j];
                let si = ti.schema();
                let sj = tj.schema();

                if Self::schemas_identical(&si, &sj) {
                    let msg = format!(
                        "Duplicate schema: '{}' and '{}' have identical function schemas",
                        ti.name(),
                        tj.name()
                    );
                    report_map
                        .entry(ti.name().to_string())
                        .or_insert_with(|| {
                            let mut r = ValidationReport::new(ti.name());
                            r.fail("duplicate name or schema check");
                            r
                        })
                        .warn(&msg);
                    report_map
                        .entry(tj.name().to_string())
                        .or_insert_with(|| {
                            let mut r = ValidationReport::new(tj.name());
                            r.fail("duplicate name or schema check");
                            r
                        })
                        .warn(&msg);
                }
            }
        }

        // Finalise scores for every report we built
        for report in report_map.values_mut() {
            report.recalc_score();
        }

        report_map.into_values().collect()
    }

    /// Compare two tool schemas for semantic identity
    /// (matching description and parameters, ignoring the name field).
    fn schemas_identical(a: &ToolSchema, b: &ToolSchema) -> bool {
        a.function.description == b.function.description
            && a.function.parameters == b.function.parameters
    }
}

// ── Tool Doctor ─────────────────────────────────────────────────────

/// Result of running a comprehensive doctor check on the tool ecosystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorReport {
    /// Overall health status.
    pub healthy: bool,
    /// Per-tool check results.
    pub tool_checks: Vec<ToolDoctorCheck>,
    /// Ecosystem-wide checks.
    pub ecosystem_checks: Vec<EcosystemCheck>,
    /// Summary counts.
    pub summary: DoctorSummary,
}

/// Per-tool doctor check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDoctorCheck {
    pub tool_name: String,
    /// Individual checks run on this tool.
    pub checks: Vec<DoctorCheckItem>,
}

/// A single health check for a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorCheckItem {
    pub name: String,
    pub status: DoctorCheckStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DoctorCheckStatus {
    Pass,
    Fail,
    Warn,
}

/// Ecosystem-wide health check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcosystemCheck {
    pub name: String,
    pub status: DoctorCheckStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Summary counts for the doctor report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorSummary {
    pub total_tools: usize,
    pub healthy_tools: usize,
    pub unhealthy_tools: usize,
    pub total_checks: usize,
    pub passed: usize,
    pub failed: usize,
    pub warnings: usize,
}

impl DoctorReport {
    fn new() -> Self {
        Self {
            healthy: true,
            tool_checks: Vec::new(),
            ecosystem_checks: Vec::new(),
            summary: DoctorSummary {
                total_tools: 0,
                healthy_tools: 0,
                unhealthy_tools: 0,
                total_checks: 0,
                passed: 0,
                failed: 0,
                warnings: 0,
            },
        }
    }

    fn recalc(&mut self) {
        self.healthy = true;
        self.summary.passed = 0;
        self.summary.failed = 0;
        self.summary.warnings = 0;
        self.summary.total_checks = 0;

        // Collect items first to avoid borrow conflicts
        let items: Vec<DoctorCheckItem> = self
            .tool_checks
            .iter()
            .flat_map(|tc| tc.checks.iter().cloned())
            .collect();

        for c in &items {
            self.add_doctor_item_to_summary(c);
        }

        // Ecosystem checks
        let eco_statuses: Vec<DoctorCheckStatus> =
            self.ecosystem_checks.iter().map(|ec| ec.status).collect();
        for status in &eco_statuses {
            match status {
                DoctorCheckStatus::Pass => self.summary.passed += 1,
                DoctorCheckStatus::Fail => {
                    self.summary.failed += 1;
                    self.healthy = false;
                }
                DoctorCheckStatus::Warn => self.summary.warnings += 1,
            }
            self.summary.total_checks += 1;
        }

        self.summary.total_tools = self.tool_checks.len();
        self.summary.healthy_tools = self
            .tool_checks
            .iter()
            .filter(|tc| {
                tc.checks
                    .iter()
                    .all(|c| c.status != DoctorCheckStatus::Fail)
            })
            .count();
        self.summary.unhealthy_tools = self.summary.total_tools - self.summary.healthy_tools;
    }

    fn add_doctor_item_to_summary(&mut self, c: &DoctorCheckItem) {
        match c.status {
            DoctorCheckStatus::Pass => self.summary.passed += 1,
            DoctorCheckStatus::Fail => {
                self.summary.failed += 1;
                self.healthy = false;
            }
            DoctorCheckStatus::Warn => self.summary.warnings += 1,
        }
        self.summary.total_checks += 1;
    }
}

/// Comprehensive tool ecosystem health checker.
pub struct ToolDoctor;

impl ToolDoctor {
    /// Run a full doctor check on every tool in the registry plus
    /// ecosystem-wide health checks.
    pub fn check(registry: &ToolRegistry) -> DoctorReport {
        let mut report = DoctorReport::new();
        let tools = registry.all_tools();

        // ── Per-tool checks ──────────────────────────────────────
        for tool in &tools {
            let mut checks = Vec::new();

            // 1. Unique name (non-empty)
            let name = tool.name();
            checks.push(DoctorCheckItem {
                name: "unique name".into(),
                status: if name.is_empty() {
                    DoctorCheckStatus::Fail
                } else {
                    DoctorCheckStatus::Pass
                },
                detail: if name.is_empty() {
                    Some("Tool name is empty".into())
                } else {
                    None
                },
            });

            // 2. Description non-empty
            let desc = tool.description();
            checks.push(DoctorCheckItem {
                name: "description".into(),
                status: if desc.is_empty() {
                    DoctorCheckStatus::Fail
                } else {
                    DoctorCheckStatus::Pass
                },
                detail: if desc.is_empty() {
                    Some("Tool has no description".into())
                } else {
                    None
                },
            });

            // 3. Valid JSON schema
            let schema = tool.schema();
            let params = &schema.function.parameters;
            let schema_valid =
                params.is_object() && params.get("type").and_then(|v| v.as_str()) == Some("object");
            checks.push(DoctorCheckItem {
                name: "valid JSON schema".into(),
                status: if schema_valid {
                    DoctorCheckStatus::Pass
                } else {
                    DoctorCheckStatus::Fail
                },
                detail: if !schema_valid {
                    Some("Schema is not a valid JSON Schema object".into())
                } else {
                    None
                },
            });

            // 4. Required params documented
            if schema_valid {
                let required = params
                    .get("required")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);
                let has_props = params
                    .get("properties")
                    .and_then(|v| v.as_object())
                    .is_some();
                let all_documented = if required > 0 {
                    params
                        .get("required")
                        .and_then(|v| v.as_array())
                        .map(|a| {
                            let props = params.get("properties").and_then(|v| v.as_object());
                            match props {
                                Some(p) => a.iter().all(|f| {
                                    f.as_str().map(|s| p.contains_key(s)).unwrap_or(false)
                                }),
                                None => false,
                            }
                        })
                        .unwrap_or(false)
                } else {
                    true // no required params → nothing to document
                };
                checks.push(DoctorCheckItem {
                    name: "required params documented".into(),
                    status: if all_documented {
                        DoctorCheckStatus::Pass
                    } else if required > 0 && !has_props {
                        DoctorCheckStatus::Fail
                    } else {
                        DoctorCheckStatus::Warn
                    },
                    detail: if !all_documented && required > 0 {
                        Some(format!(
                            "{} required param(s) but not all have property definitions",
                            required
                        ))
                    } else if required == 0 {
                        Some("No required parameters declared".into())
                    } else {
                        None
                    },
                });
            } else {
                checks.push(DoctorCheckItem {
                    name: "required params documented".into(),
                    status: DoctorCheckStatus::Fail,
                    detail: Some("Cannot check — schema is invalid".into()),
                });
            }

            // 5. Capability tags present
            let tags = tool.capability_tags();
            checks.push(DoctorCheckItem {
                name: "capability tags".into(),
                status: if tags.is_empty() {
                    DoctorCheckStatus::Fail
                } else {
                    DoctorCheckStatus::Pass
                },
                detail: if tags.is_empty() {
                    Some("Tool has no capability tags".into())
                } else {
                    None
                },
            });

            // 6. Safety consistency
            let is_dangerous = tool.is_dangerous();
            let has_dangerous_tag = tags.iter().any(|tag| tag == "dangerous");
            let has_safe_tag = tags.iter().any(|tag| tag == "safe");

            if is_dangerous != has_dangerous_tag {
                checks.push(DoctorCheckItem {
                    name: "safety consistency".into(),
                    status: DoctorCheckStatus::Warn,
                    detail: Some(format!(
                        "is_dangerous()={is_dangerous} but 'dangerous' tag is {}",
                        if has_dangerous_tag {
                            "present"
                        } else {
                            "missing"
                        }
                    )),
                });
            } else if is_dangerous && !tool.requires_approval() {
                checks.push(DoctorCheckItem {
                    name: "dangerous requires approval".into(),
                    status: DoctorCheckStatus::Warn,
                    detail: Some("Tool is dangerous but does not require approval".into()),
                });
            } else {
                checks.push(DoctorCheckItem {
                    name: "safety consistency".into(),
                    status: DoctorCheckStatus::Pass,
                    detail: None,
                });
            }

            // 7. Is safe consistency
            if has_safe_tag && !tool.is_safe() {
                checks.push(DoctorCheckItem {
                    name: "safe tag consistency".into(),
                    status: DoctorCheckStatus::Warn,
                    detail: Some("Tool has 'safe' tag but is_safe() returns false".into()),
                });
            } else {
                checks.push(DoctorCheckItem {
                    name: "safe tag consistency".into(),
                    status: DoctorCheckStatus::Pass,
                    detail: None,
                });
            }

            // 8. Schema function name matches tool name
            let schema_name_matches = schema.function.name == name;
            checks.push(DoctorCheckItem {
                name: "schema name matches".into(),
                status: if schema_name_matches {
                    DoctorCheckStatus::Pass
                } else {
                    DoctorCheckStatus::Fail
                },
                detail: if !schema_name_matches {
                    Some(format!(
                        "Schema function name '{}' != tool name '{}'",
                        schema.function.name, name
                    ))
                } else {
                    None
                },
            });

            report.tool_checks.push(ToolDoctorCheck {
                tool_name: name.to_string(),
                checks,
            });
        }

        // ── Ecosystem-wide checks ────────────────────────────────
        // Duplicate detection
        let dup_reports = Self::check_duplicates(registry);
        let has_dupes = !dup_reports.is_empty();
        report.ecosystem_checks.push(EcosystemCheck {
            name: "duplicate tools".into(),
            status: if has_dupes {
                DoctorCheckStatus::Fail
            } else {
                DoctorCheckStatus::Pass
            },
            detail: if has_dupes {
                Some(format!(
                    "{} duplicate(s) detected: {}",
                    dup_reports.len(),
                    dup_reports
                        .iter()
                        .map(|r| r.tool_name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            } else {
                None
            },
        });

        // Tool count
        report.ecosystem_checks.push(EcosystemCheck {
            name: "tool count".into(),
            status: if tools.is_empty() {
                DoctorCheckStatus::Warn
            } else {
                DoctorCheckStatus::Pass
            },
            detail: Some(format!("{} tools registered", tools.len())),
        });

        report.recalc();
        report
    }

    fn check_duplicates(registry: &ToolRegistry) -> Vec<ValidationReport> {
        ToolValidator::detect_duplicates(registry)
    }

    /// Check if a single named tool passes all doctor checks.
    pub fn check_one(registry: &ToolRegistry, tool_name: &str) -> Option<DoctorReport> {
        let tool = registry.get(tool_name)?;
        let mut report = DoctorReport::new();

        let mut checks = Vec::new();

        // Same checks as above but for a single tool
        let name = tool.name();
        checks.push(DoctorCheckItem {
            name: "unique name".into(),
            status: if name.is_empty() {
                DoctorCheckStatus::Fail
            } else {
                DoctorCheckStatus::Pass
            },
            detail: None,
        });

        let desc = tool.description();
        checks.push(DoctorCheckItem {
            name: "description".into(),
            status: if desc.is_empty() {
                DoctorCheckStatus::Fail
            } else {
                DoctorCheckStatus::Pass
            },
            detail: None,
        });

        let schema = tool.schema();
        let params = &schema.function.parameters;
        let schema_valid =
            params.is_object() && params.get("type").and_then(|v| v.as_str()) == Some("object");
        checks.push(DoctorCheckItem {
            name: "valid JSON schema".into(),
            status: if schema_valid {
                DoctorCheckStatus::Pass
            } else {
                DoctorCheckStatus::Fail
            },
            detail: None,
        });

        let tags = tool.capability_tags();
        checks.push(DoctorCheckItem {
            name: "capability tags".into(),
            status: if tags.is_empty() {
                DoctorCheckStatus::Fail
            } else {
                DoctorCheckStatus::Pass
            },
            detail: None,
        });

        checks.push(DoctorCheckItem {
            name: "safety consistency".into(),
            status: DoctorCheckStatus::Pass,
            detail: None,
        });

        checks.push(DoctorCheckItem {
            name: "schema name matches".into(),
            status: if schema.function.name == name {
                DoctorCheckStatus::Pass
            } else {
                DoctorCheckStatus::Fail
            },
            detail: None,
        });

        report.tool_checks.push(ToolDoctorCheck {
            tool_name: name.to_string(),
            checks,
        });

        report.recalc();
        Some(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::data::JsonExtract;
    use crate::builtins::file::{FileRead, FileWrite};
    use crate::builtins::git::Git;
    use crate::builtins::shell::Shell;
    use crate::builtins::system::{DiskUsage, SystemInfo};
    use crate::builtins::web::{HttpRequest, WebFetch, WebSearch};
    use crate::sandbox::Sandbox;
    use std::sync::Arc;

    /// Create a registry with all built-in tools registered.
    fn builtin_registry() -> ToolRegistry {
        let registry = ToolRegistry::new();
        let sandbox = Arc::new(Sandbox::default());
        registry
            .register(Box::new(FileRead::new(sandbox.clone())))
            .unwrap();
        registry
            .register(Box::new(FileWrite::new(sandbox.clone())))
            .unwrap();
        registry.register(Box::new(Shell::new())).unwrap();
        registry.register(Box::new(WebFetch::new())).unwrap();
        registry.register(Box::new(WebSearch::new())).unwrap();
        registry.register(Box::new(HttpRequest::new())).unwrap();
        registry.register(Box::new(Git::new())).unwrap();
        registry.register(Box::new(SystemInfo::new())).unwrap();
        registry.register(Box::new(DiskUsage::new())).unwrap();
        registry.register(Box::new(JsonExtract::new())).unwrap();
        registry
    }

    // ── Schema Validation Tests ────────────────────────────────

    #[test]
    fn test_schema_all_builtins_pass() {
        let registry = builtin_registry();
        let reports = ToolValidator::validate_all(&registry);

        assert_eq!(reports.len(), 10, "expected 10 built-in tools");

        for report in &reports {
            // Every tool should have more passes than failures
            assert!(
                report.score >= 0.5,
                "Tool '{}' has low score {:.2}: {} passed, {} failed — {:?}",
                report.tool_name,
                report.score,
                report.passed.len(),
                report.failed.len(),
                report.failed,
            );
        }
    }

    #[test]
    fn test_schema_each_tool_basics() {
        let sandbox = Arc::new(Sandbox::default());

        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(FileRead::new(sandbox.clone())),
            Box::new(FileWrite::new(sandbox.clone())),
            Box::new(Shell::new()),
            Box::new(WebFetch::new()),
            Box::new(WebSearch::new()),
            Box::new(HttpRequest::new()),
            Box::new(Git::new()),
            Box::new(SystemInfo::new()),
            Box::new(DiskUsage::new()),
            Box::new(JsonExtract::new()),
        ];
        drop(sandbox); // release after all tools are created

        for tool in &tools {
            let report = ToolValidator::validate_schema(tool.as_ref());

            assert!(!tool.name().is_empty(), "tool name should be non-empty");
            assert!(
                !tool.description().is_empty(),
                "tool '{}' description should be non-empty",
                tool.name()
            );

            let schema = tool.schema();
            assert!(
                schema.function.parameters.is_object(),
                "tool '{}' parameters should be a JSON object",
                tool.name()
            );

            // Check that required fields are documented with property definitions
            if let Some(required) = schema
                .function
                .parameters
                .get("required")
                .and_then(|v| v.as_array())
            {
                for req_field in required {
                    let field_name = req_field.as_str().unwrap_or("");
                    assert!(
                        !field_name.is_empty(),
                        "tool '{}' has empty required field entry",
                        tool.name()
                    );
                    let has_prop = schema
                        .function
                        .parameters
                        .get("properties")
                        .and_then(|v| v.as_object())
                        .map(|props| props.contains_key(field_name))
                        .unwrap_or(false);
                    assert!(
                        has_prop,
                        "tool '{}' required field '{}' has no property definition",
                        tool.name(),
                        field_name
                    );
                }
            }

            // Verify no failed checks in basic schema validation
            assert!(
                report.failed.is_empty(),
                "tool '{}' has schema failures: {:?}",
                tool.name(),
                report.failed
            );
        }
    }

    // ── Arg Validation Tests ───────────────────────────────────

    #[test]
    fn test_args_valid_for_each_tool() {
        let sandbox = Arc::new(Sandbox::default());

        // Build expected required args for each tool
        let test_cases: Vec<(Box<dyn Tool>, serde_json::Value)> = vec![
            (
                Box::new(FileRead::new(sandbox.clone())),
                serde_json::json!({"path": "/tmp/test.txt"}),
            ),
            (
                Box::new(FileWrite::new(sandbox)),
                serde_json::json!({"path": "/tmp/test.txt", "content": "data"}),
            ),
            (
                Box::new(Shell::new()),
                serde_json::json!({"command": "echo hello"}),
            ),
            (
                Box::new(WebFetch::new()),
                serde_json::json!({"url": "https://example.com"}),
            ),
            (
                Box::new(WebSearch::new()),
                serde_json::json!({"query": "test"}),
            ),
            (
                Box::new(Git::new()),
                serde_json::json!({"command": "status"}),
            ),
        ];

        for (tool, args) in test_cases {
            let report = ToolValidator::validate_args(tool.as_ref(), args);
            assert!(
                report.failed.is_empty(),
                "tool '{}' arg validation failed: {:?}",
                tool.name(),
                report.failed
            );
        }
    }

    #[test]
    fn test_args_missing_required_fails() {
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(FileRead::new(Arc::new(Sandbox::default()))),
            Box::new(Shell::new()),
            Box::new(WebFetch::new()),
            Box::new(WebSearch::new()),
            Box::new(Git::new()),
        ];

        // Passing an empty object — all these tools have at least one
        // required field, so validation should fail.
        for tool in &tools {
            let report = ToolValidator::validate_args(tool.as_ref(), serde_json::json!({}));
            assert!(
                !report.failed.is_empty(),
                "tool '{}' should have failed with empty args",
                tool.name()
            );
        }
    }

    // ── Permission Validation Tests ────────────────────────────

    #[test]
    fn test_dangerous_tools_require_approval() {
        let sandbox = Arc::new(Sandbox::default());

        // Shell and Git are explicitly dangerous and require approval
        let explicit_dangerous: Vec<Box<dyn Tool>> =
            vec![Box::new(Shell::new()), Box::new(Git::new())];

        for tool in &explicit_dangerous {
            assert!(
                tool.capability_tags().iter().any(|tag| tag == "dangerous"),
                "tool '{}' should have 'dangerous' capability tag",
                tool.name()
            );
            assert!(
                tool.requires_approval(),
                "tool '{}' requires_approval() should be true",
                tool.name()
            );
        }

        // FileWrite has 'dangerous' tag but uses sandbox enforcement
        // instead of user-approval — it's dangerous by capability
        // but constrained by the sandbox boundary.
        let file_write = FileWrite::new(sandbox.clone());
        assert!(
            file_write
                .capability_tags()
                .iter()
                .any(|tag| tag == "dangerous"),
            "FileWrite should have 'dangerous' tag"
        );

        // FileRead is neither dangerous nor requires approval
        assert!(
            !FileRead::new(sandbox)
                .capability_tags()
                .iter()
                .any(|tag| tag == "dangerous"),
            "FileRead should not have 'dangerous' tag"
        );
    }

    #[test]
    fn test_safe_tools_marked_safe() {
        let sandbox = Arc::new(Sandbox::default());
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(FileRead::new(sandbox.clone())),
            Box::new(WebFetch::new()),
            Box::new(WebSearch::new()),
        ];

        for tool in &tools {
            assert!(
                tool.capability_tags().iter().any(|tag| tag == "safe"),
                "tool '{}' should have 'safe' capability tag",
                tool.name()
            );
            assert!(
                tool.is_safe(),
                "tool '{}' is_safe() should be true",
                tool.name()
            );
        }
    }

    #[test]
    fn test_permissions_report_dangerous() {
        let shell = Shell::new();
        let report = ToolValidator::validate_permissions(&shell);
        assert!(
            report
                .passed
                .iter()
                .any(|c| c.contains("requires approval")),
            "shell permission report should include approval check: {:?}",
            report.passed
        );
        assert!(
            report.failed.is_empty(),
            "shell permission report should have no failures: {:?}",
            report.failed
        );
    }

    #[test]
    fn test_permissions_report_safe() {
        let fetch = WebFetch::new();
        let report = ToolValidator::validate_permissions(&fetch);
        assert!(
            report.passed.iter().any(|c| c.contains("marked safe")),
            "WebFetch permission report should include safe check: {:?}",
            report.passed
        );
        assert!(
            report.failed.is_empty(),
            "WebFetch permission report should have no failures: {:?}",
            report.failed
        );
    }

    // ── Capability Tags Tests ──────────────────────────────────

    #[test]
    fn test_capability_tags_present_and_non_empty() {
        let sandbox = Arc::new(Sandbox::default());
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(FileRead::new(sandbox.clone())),
            Box::new(FileWrite::new(sandbox.clone())),
            Box::new(Shell::new()),
            Box::new(WebFetch::new()),
            Box::new(WebSearch::new()),
            Box::new(Git::new()),
        ];

        let expected: Vec<Vec<String>> = vec![
            vec!["filesystem".into(), "read".into(), "safe".into()],
            vec!["filesystem".into(), "write".into(), "dangerous".into()],
            vec!["shell".into(), "system".into(), "dangerous".into()],
            vec!["web".into(), "http".into(), "read".into(), "safe".into()],
            vec!["web".into(), "search".into(), "read".into(), "safe".into()],
            vec!["version-control".into(), "git".into(), "dangerous".into()],
        ];

        for (tool, expected_tags) in tools.iter().zip(expected.iter()) {
            let tags = tool.capability_tags();
            assert!(
                !tags.is_empty(),
                "tool '{}' should have non-empty capability tags",
                tool.name()
            );
            assert_eq!(
                tags,
                *expected_tags,
                "tool '{}' capability tags mismatch",
                tool.name()
            );
        }
    }

    // ── validate_all integration test ──────────────────────────

    #[test]
    fn test_validate_all_returns_reports_for_all_tools() {
        let registry = builtin_registry();
        let reports = ToolValidator::validate_all(&registry);

        assert_eq!(reports.len(), registry.len());

        for report in &reports {
            assert!(!report.tool_name.is_empty());
            // Each report should have at least some passed checks
            assert!(
                !report.passed.is_empty(),
                "tool '{}' has no passed checks",
                report.tool_name
            );
            // Score should be within [0, 1]
            assert!(
                (0.0..=1.0).contains(&report.score),
                "tool '{}' score {} is out of range",
                report.tool_name,
                report.score
            );
        }
    }

    #[test]
    fn test_validate_all_no_duplicate_tools() {
        let registry = builtin_registry();
        let reports = ToolValidator::validate_all(&registry);
        let names: Vec<&str> = reports.iter().map(|r| r.tool_name.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(names.len(), sorted.len(), "duplicate tool names in reports");
    }

    // ── Duplicate Detection Tests ──────────────────────────────

    /// A test tool with configurable name, description, and schema.
    struct DupTool {
        name: String,
        desc: String,
        params: serde_json::Value,
    }

    #[async_trait::async_trait]
    impl Tool for DupTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            &self.desc
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                schema_type: "function".into(),
                function: odin_core::types::FunctionSchema {
                    name: self.name.clone(),
                    description: self.desc.clone(),
                    parameters: self.params.clone(),
                },
            }
        }
        fn is_safe(&self) -> bool {
            true
        }
        fn requires_approval(&self) -> bool {
            false
        }
        fn capability_tags(&self) -> Vec<String> {
            vec!["test".into()]
        }
        async fn execute(
            &self,
            _args: serde_json::Value,
            _context: &odin_core::traits::ToolContext,
        ) -> odin_core::error::OdinResult<odin_core::types::ToolResult> {
            Ok(odin_core::types::ToolResult {
                call_id: "dup".into(),
                tool_name: self.name.clone(),
                success: true,
                output: "ok".into(),
                error: None,
                duration_ms: 0,
                timestamp: chrono::Utc::now(),
            })
        }
    }

    #[test]
    fn test_detect_duplicates_identical_schema() {
        let registry = ToolRegistry::new();
        registry
            .register(Box::new(DupTool {
                name: "tool_a".into(),
                desc: "Same description".into(),
                params: serde_json::json!({"type": "object", "properties": {}}),
            }))
            .unwrap();
        registry
            .register(Box::new(DupTool {
                name: "tool_b".into(),
                desc: "Same description".into(),
                params: serde_json::json!({"type": "object", "properties": {}}),
            }))
            .unwrap();

        let reports = ToolValidator::detect_duplicates(&registry);
        assert_eq!(reports.len(), 2, "both tools should be flagged");

        // Each report should warn about duplicate schema
        for r in &reports {
            assert!(
                r.warnings.iter().any(|w| w.contains("Duplicate schema")),
                "report for '{}' should contain 'Duplicate schema' warning: {:?}",
                r.tool_name,
                r.warnings
            );
        }
    }

    #[test]
    fn test_validate_all_merges_dup_warnings() {
        let registry = ToolRegistry::new();
        registry
            .register(Box::new(DupTool {
                name: "tool_x".into(),
                desc: "Shared description".into(),
                params: serde_json::json!({"type": "object", "properties": {}}),
            }))
            .unwrap();
        registry
            .register(Box::new(DupTool {
                name: "tool_y".into(),
                desc: "Shared description".into(),
                params: serde_json::json!({"type": "object", "properties": {}}),
            }))
            .unwrap();

        let reports = ToolValidator::validate_all(&registry);
        assert_eq!(reports.len(), 2);

        // Both reports should contain the duplicate-schema warning
        for r in &reports {
            assert!(
                r.warnings.iter().any(|w| w.contains("Duplicate schema")),
                "validate_all report for '{}' should contain duplicate warning: {:?}",
                r.tool_name,
                r.warnings
            );
        }
    }

    // ── Enable / Disable Filtering Tests ───────────────────────

    /// Helper that mirrors the tool_enabled closure used in CLI/Gateway.
    fn is_tool_allowed(enabled: &[String], disabled: &[String], name: &str) -> bool {
        if !enabled.is_empty() && !enabled.iter().any(|e| e == name) {
            return false;
        }
        if disabled.iter().any(|d| d == name) {
            return false;
        }
        true
    }

    #[test]
    fn test_enable_empty_list_means_all_enabled() {
        let enabled: Vec<String> = vec![];
        let disabled: Vec<String> = vec![];
        assert!(is_tool_allowed(&enabled, &disabled, "file_read"));
        assert!(is_tool_allowed(&enabled, &disabled, "shell"));
        assert!(is_tool_allowed(&enabled, &disabled, "custom_tool"));
    }

    #[test]
    fn test_enable_list_restricts_tools() {
        let enabled = vec!["file_read".into(), "shell".into()];
        let disabled: Vec<String> = vec![];
        assert!(is_tool_allowed(&enabled, &disabled, "file_read"));
        assert!(is_tool_allowed(&enabled, &disabled, "shell"));
        assert!(!is_tool_allowed(&enabled, &disabled, "git"));
        assert!(!is_tool_allowed(&enabled, &disabled, "web_fetch"));
    }

    #[test]
    fn test_disabled_list_overrides_enabled() {
        let enabled = vec!["file_read".into(), "shell".into(), "git".into()];
        let disabled = vec!["shell".into()];
        assert!(is_tool_allowed(&enabled, &disabled, "file_read"));
        assert!(!is_tool_allowed(&enabled, &disabled, "shell"));
        assert!(is_tool_allowed(&enabled, &disabled, "git"));
    }

    #[test]
    fn test_disabled_excludes_tool() {
        let enabled: Vec<String> = vec![]; // all enabled
        let disabled = vec!["shell".into()];
        assert!(is_tool_allowed(&enabled, &disabled, "file_read"));
        assert!(!is_tool_allowed(&enabled, &disabled, "shell"));
        assert!(is_tool_allowed(&enabled, &disabled, "web_fetch"));
    }

    #[test]
    fn test_enabled_count_returns_len() {
        let registry = builtin_registry();
        assert_eq!(registry.enabled_count(), registry.len());
        assert_eq!(registry.enabled_count(), 10);
    }

    // ── Tool Doctor Tests ──────────────────────────────────────

    #[test]
    fn test_doctor_all_builtins_healthy() {
        let registry = builtin_registry();
        let report = ToolDoctor::check(&registry);

        assert_eq!(report.tool_checks.len(), 10, "should check all 10 tools");
        assert!(report.healthy, "all builtins should pass doctor checks");
        assert_eq!(report.summary.total_tools, 10);
        assert_eq!(report.summary.healthy_tools, 10);
        assert_eq!(report.summary.unhealthy_tools, 0);
        assert_eq!(report.summary.failed, 0);
    }

    #[test]
    fn test_doctor_each_tool_has_checks() {
        let registry = builtin_registry();
        let report = ToolDoctor::check(&registry);

        for tc in &report.tool_checks {
            assert!(
                !tc.checks.is_empty(),
                "tool '{}' should have doctor checks",
                tc.tool_name
            );
            // Minimum checks: unique name, description, valid JSON schema,
            // required params documented, capability tags, safety consistency,
            // safe tag consistency, schema name matches
            assert!(
                tc.checks.len() >= 6,
                "tool '{}' has {} checks, expected at least 6",
                tc.tool_name,
                tc.checks.len()
            );
        }
    }

    #[test]
    fn test_doctor_ecosystem_checks_exist() {
        let registry = builtin_registry();
        let report = ToolDoctor::check(&registry);

        assert!(
            !report.ecosystem_checks.is_empty(),
            "should have ecosystem-wide checks"
        );
        assert!(
            report
                .ecosystem_checks
                .iter()
                .any(|ec| ec.name == "duplicate tools"),
            "should check for duplicates"
        );
        assert!(
            report
                .ecosystem_checks
                .iter()
                .any(|ec| ec.name == "tool count"),
            "should report tool count"
        );
    }

    #[test]
    fn test_doctor_detects_duplicates() {
        let registry = ToolRegistry::new();
        registry
            .register(Box::new(DupTool {
                name: "dup_a".into(),
                desc: "Same description".into(),
                params: serde_json::json!({"type": "object", "properties": {}}),
            }))
            .unwrap();
        registry
            .register(Box::new(DupTool {
                name: "dup_b".into(),
                desc: "Same description".into(),
                params: serde_json::json!({"type": "object", "properties": {}}),
            }))
            .unwrap();

        let report = ToolDoctor::check(&registry);

        // Should detect duplicates
        let dup_check = report
            .ecosystem_checks
            .iter()
            .find(|ec| ec.name == "duplicate tools")
            .expect("should have duplicate check");
        assert_eq!(
            dup_check.status,
            DoctorCheckStatus::Fail,
            "duplicates should be detected as failure"
        );
        assert!(
            !report.healthy,
            "report should be unhealthy with duplicates"
        );
    }

    #[test]
    fn test_doctor_summary_counts_match() {
        let registry = builtin_registry();
        let report = ToolDoctor::check(&registry);

        let summary = &report.summary;
        assert!(summary.passed > 0, "should have passed checks");
        assert!(summary.total_checks > 0, "should have total checks");
        assert_eq!(
            summary.passed + summary.failed + summary.warnings,
            summary.total_checks,
            "passed + failed + warnings should equal total_checks"
        );
        assert_eq!(
            summary.healthy_tools + summary.unhealthy_tools,
            summary.total_tools,
            "healthy + unhealthy should equal total"
        );
    }

    #[test]
    fn test_doctor_check_one_existing_tool() {
        let registry = builtin_registry();
        let report = ToolDoctor::check_one(&registry, "shell").expect("should find 'shell' tool");

        assert_eq!(report.tool_checks.len(), 1);
        assert_eq!(report.tool_checks[0].tool_name, "shell");
        assert!(report.summary.total_tools == 1);
    }

    #[test]
    fn test_doctor_check_one_nonexistent() {
        let registry = builtin_registry();
        let report = ToolDoctor::check_one(&registry, "nonexistent");
        assert!(report.is_none(), "should return None for nonexistent tool");
    }

    #[test]
    fn test_doctor_report_json_roundtrip() {
        let registry = builtin_registry();
        let report = ToolDoctor::check(&registry);

        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("tool_checks"));
        assert!(json.contains("ecosystem_checks"));
        assert!(json.contains("healthy"));

        // Verify it's valid JSON
        let _parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    }
}
