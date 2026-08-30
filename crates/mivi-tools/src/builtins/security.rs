//! Filesystem security validation and safe path resolution.

use std::path::{Component, Path, PathBuf};

/// Safely resolve an untrusted relative path against a trusted workspace root.
/// Rejects '..', absolute paths, and Windows UNC prefixes, and verifies via canonicalize
/// that symlinks cannot escape the workspace directory.
pub fn safe_join(base: &Path, untrusted: &str) -> Result<PathBuf, String> {
    let rel = Path::new(untrusted);
    for component in rel.components() {
        match component {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir => {
                return Err(format!(
                    "Path traversal blocked: '..' is forbidden in '{}'",
                    untrusted
                ))
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(format!(
                    "Absolute or prefixed paths are forbidden in '{}'",
                    untrusted
                ))
            }
        }
    }

    let joined = base.join(rel);

    // If base directory exists on the filesystem, enforce canonical containment check
    if base.exists() {
        let canonical_base = base
            .canonicalize()
            .map_err(|e| format!("Failed to canonicalize base path {:?}: {}", base, e))?;

        if joined.exists() {
            let canonical_target = joined
                .canonicalize()
                .map_err(|e| format!("Failed to canonicalize target path {:?}: {}", joined, e))?;
            if !canonical_target.starts_with(&canonical_base) {
                return Err(format!(
                    "Symlink traversal blocked: path '{}' escapes workspace root",
                    untrusted
                ));
            }
        } else {
            let mut ancestor = joined.parent();
            while let Some(p) = ancestor {
                if p.exists() {
                    let canonical_ancestor = p.canonicalize().map_err(|e| {
                        format!("Failed to canonicalize ancestor directory {:?}: {}", p, e)
                    })?;
                    if !canonical_ancestor.starts_with(&canonical_base) {
                        return Err(format!(
                            "Symlink traversal blocked: parent directory of '{}' escapes workspace root",
                            untrusted
                        ));
                    }
                    break;
                }
                ancestor = p.parent();
            }
        }
    }

    Ok(joined)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_join_blocks_traversal() {
        let base = Path::new("/workspace");
        assert!(safe_join(base, "../../etc/passwd").is_err());
        assert!(safe_join(base, "/etc/shadow").is_err());
        assert!(safe_join(base, "valid/sub/path.txt").is_ok());
        assert!(safe_join(base, "./local_file.rs").is_ok());
    }
}
