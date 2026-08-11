//! Canonical, component-aware filesystem boundary enforcement.

use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

use crate::error::{OdinError, OdinResult};
use crate::types::PathBoundary;

/// A [`PathBoundary`] resolved against one stable working directory.
///
/// Configured roots and requested paths pass through the same canonicalisation
/// pipeline. Existing symlinks are resolved, while create targets are anchored
/// at their nearest existing ancestor before their missing suffix is appended.
#[derive(Debug, Clone)]
pub struct ResolvedPathBoundary {
    allowed_read: Vec<PathBuf>,
    allowed_write: Vec<PathBuf>,
    denied: Vec<PathBuf>,
    base_dir: PathBuf,
}

impl ResolvedPathBoundary {
    /// Resolve all configured roots against the current directory.
    pub fn new(boundary: &PathBoundary) -> OdinResult<Self> {
        let base_dir = std::env::current_dir()
            .map_err(OdinError::Io)?
            .canonicalize()
            .map_err(OdinError::Io)?;

        let resolve_rules = |rules: &[String]| -> OdinResult<Vec<PathBuf>> {
            rules
                .iter()
                .map(|rule| {
                    let expanded = expand_home(rule)?;
                    resolve_from(&base_dir, &expanded)
                })
                .collect()
        };

        Ok(Self {
            allowed_read: resolve_rules(&boundary.allowed_read)?,
            allowed_write: resolve_rules(&boundary.allowed_write)?,
            denied: resolve_rules(&boundary.denied)?,
            base_dir,
        })
    }

    /// Resolve and authorize a read path.
    pub fn check_read(&self, path: &Path) -> OdinResult<PathBuf> {
        self.check(path, false)
    }

    /// Resolve and authorize a write/create path.
    pub fn check_write(&self, path: &Path) -> OdinResult<PathBuf> {
        self.check(path, true)
    }

    fn check(&self, path: &Path, write: bool) -> OdinResult<PathBuf> {
        let resolved = resolve_from(&self.base_dir, path)?;

        if let Some(rule) = self
            .denied
            .iter()
            .find(|rule| resolved == **rule || resolved.starts_with(rule))
        {
            return Err(OdinError::PermissionDenied(format!(
                "Path '{}' is denied by rule '{}'",
                resolved.display(),
                rule.display()
            )));
        }

        let allowed = if write {
            &self.allowed_write
        } else {
            &self.allowed_read
        };
        if allowed
            .iter()
            .any(|root| resolved == *root || resolved.starts_with(root))
        {
            return Ok(resolved);
        }

        Err(OdinError::PermissionDenied(format!(
            "Path '{}' is outside allowed {} boundaries",
            resolved.display(),
            if write { "write" } else { "read" }
        )))
    }
}

fn expand_home(rule: &str) -> OdinResult<PathBuf> {
    if rule == "~" || rule.starts_with("~/") {
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .ok_or_else(|| {
                OdinError::Validation(format!(
                    "Cannot resolve path boundary '{rule}': home directory is unavailable"
                ))
            })?;
        let suffix = rule.strip_prefix("~/").unwrap_or("");
        return Ok(PathBuf::from(home).join(suffix));
    }
    if rule.starts_with('~') {
        return Err(OdinError::Validation(format!(
            "Cannot resolve path boundary '{rule}': named-user expansion is unsupported"
        )));
    }
    Ok(PathBuf::from(rule))
}

fn resolve_from(base_dir: &Path, path: &Path) -> OdinResult<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    };

    if absolute
        .components()
        .any(|component| component == Component::ParentDir)
    {
        return Err(OdinError::PermissionDenied(format!(
            "Parent-directory aliases are not allowed in path '{}'",
            path.display()
        )));
    }

    if absolute.exists() {
        return absolute.canonicalize().map_err(OdinError::Io);
    }

    let mut cursor = absolute.as_path();
    let mut missing: Vec<OsString> = Vec::new();
    while !cursor.exists() {
        let name = cursor.file_name().ok_or_else(|| {
            OdinError::Validation(format!(
                "Cannot resolve path '{}': no existing ancestor",
                path.display()
            ))
        })?;
        missing.push(name.to_os_string());
        cursor = cursor.parent().ok_or_else(|| {
            OdinError::Validation(format!(
                "Cannot resolve path '{}': no existing ancestor",
                path.display()
            ))
        })?;
    }

    let mut resolved = cursor.canonicalize().map_err(OdinError::Io)?;
    for component in missing.into_iter().rev() {
        resolved.push(component);
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn sibling_prefix_is_not_inside_allowed_root() {
        let parent = tempfile::tempdir().unwrap();
        let allowed = parent.path().join("repo");
        let sibling = parent.path().join("repo-private");
        fs::create_dir_all(&allowed).unwrap();
        fs::create_dir_all(&sibling).unwrap();
        let boundary = PathBoundary {
            allowed_read: vec![allowed.display().to_string()],
            allowed_write: vec![allowed.display().to_string()],
            denied: vec![],
        };
        let resolved = ResolvedPathBoundary::new(&boundary).unwrap();
        assert!(resolved.check_read(&allowed).is_ok());
        assert!(resolved.check_read(&sibling).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn create_through_symlink_outside_root_is_denied() {
        use std::os::unix::fs::symlink;

        let parent = tempfile::tempdir().unwrap();
        let allowed = parent.path().join("allowed");
        let outside = parent.path().join("outside");
        fs::create_dir_all(&allowed).unwrap();
        fs::create_dir_all(&outside).unwrap();
        symlink(&outside, allowed.join("escape")).unwrap();

        let boundary = PathBoundary {
            allowed_read: vec![allowed.display().to_string()],
            allowed_write: vec![allowed.display().to_string()],
            denied: vec![],
        };
        let resolved = ResolvedPathBoundary::new(&boundary).unwrap();
        assert!(
            resolved
                .check_write(&allowed.join("escape/new-file"))
                .is_err()
        );
    }
}
