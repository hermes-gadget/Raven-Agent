//! Sandbox — validates file paths against [`PathBoundary`] rules.
//!
//! The [`Sandbox`] ensures that file operations performed by tools stay
//! within the allowed directories and do not access denied paths.

use std::path::{Path, PathBuf};

use odin_core::ResolvedPathBoundary;
use odin_core::error::{OdinError, OdinResult};
use odin_core::types::PathBoundary;

/// Filesystem boundary enforcer.
///
/// Wraps a [`PathBoundary`] and provides methods to check whether a given
/// path is allowed for reading or writing.
#[derive(Debug, Clone)]
pub struct Sandbox {
    boundary: PathBoundary,
    resolved: Result<ResolvedPathBoundary, String>,
}

impl Sandbox {
    /// Create a new sandbox from a [`PathBoundary`].
    pub fn new(boundary: PathBoundary) -> Self {
        let resolved = ResolvedPathBoundary::new(&boundary).map_err(|error| error.to_string());
        Self { boundary, resolved }
    }

    /// Borrow the underlying boundary configuration.
    pub fn boundary(&self) -> &PathBoundary {
        &self.boundary
    }

    /// Check whether `path` is allowed for reading.
    ///
    /// Returns the canonicalised path on success, or an error if the path
    /// is outside the allowed boundaries or falls in the denied list.
    pub fn check_read(&self, path: &Path) -> OdinResult<PathBuf> {
        self.resolved()?.check_read(path)
    }

    /// Check whether `path` is allowed for writing.
    ///
    /// For paths that don't exist yet, the parent directory is used for
    /// boundary checking.
    pub fn check_write(&self, path: &Path) -> OdinResult<PathBuf> {
        self.resolved()?.check_write(path)
    }

    fn resolved(&self) -> OdinResult<&ResolvedPathBoundary> {
        self.resolved.as_ref().map_err(|error| {
            OdinError::Validation(format!(
                "Invalid filesystem boundary configuration: {error}"
            ))
        })
    }
}

impl Default for Sandbox {
    fn default() -> Self {
        Self::new(PathBoundary::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_default_sandbox() {
        let sandbox = Sandbox::default();
        assert!(!sandbox.boundary().allowed_read.is_empty());
    }

    #[test]
    fn test_read_allowed_in_temp() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, "hello").unwrap();

        let boundary = PathBoundary {
            allowed_read: vec![dir.path().to_string_lossy().to_string()],
            allowed_write: vec![dir.path().to_string_lossy().to_string()],
            denied: vec![],
        };
        let sandbox = Sandbox::new(boundary);
        let result = sandbox.check_read(&file_path);
        assert!(result.is_ok(), "{:?}", result.err());
    }

    #[test]
    fn test_write_denied_outside_boundary() {
        let boundary = PathBoundary {
            allowed_read: vec!["/tmp".into()],
            allowed_write: vec!["/tmp".into()],
            denied: vec![],
        };
        let sandbox = Sandbox::new(boundary);
        // /etc is not in allowed_write
        let result = sandbox.check_write(Path::new("/etc/passwd"));
        assert!(result.is_err());
    }

    #[test]
    fn test_denied_path() {
        let boundary = PathBoundary {
            allowed_read: vec!["/".into()],
            allowed_write: vec!["/tmp".into()],
            denied: vec!["/etc/shadow".into()],
        };
        let sandbox = Sandbox::new(boundary);
        let result = sandbox.check_read(Path::new("/etc/shadow"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("denied"));
    }

    #[test]
    fn test_nonexistent_path_write() {
        let dir = tempfile::tempdir().unwrap();
        let new_file = dir.path().join("new_file.txt");
        // File doesn't exist yet
        assert!(!new_file.exists());

        let boundary = PathBoundary {
            allowed_read: vec![dir.path().to_string_lossy().to_string()],
            allowed_write: vec![dir.path().to_string_lossy().to_string()],
            denied: vec![],
        };
        let sandbox = Sandbox::new(boundary);
        let result = sandbox.check_write(&new_file);
        assert!(result.is_ok(), "{:?}", result.err());
    }

    #[test]
    fn test_sibling_prefix_is_denied() {
        let parent = tempfile::tempdir().unwrap();
        let allowed = parent.path().join("repo");
        let sibling = parent.path().join("repo-private");
        fs::create_dir_all(&allowed).unwrap();
        fs::create_dir_all(&sibling).unwrap();
        let sandbox = Sandbox::new(PathBoundary {
            allowed_read: vec![allowed.display().to_string()],
            allowed_write: vec![allowed.display().to_string()],
            denied: vec![],
        });

        assert!(sandbox.check_read(&allowed).is_ok());
        assert!(sandbox.check_read(&sibling).is_err());
    }
}
