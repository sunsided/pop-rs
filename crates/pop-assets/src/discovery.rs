//! Locate POP game assets on the host system.
//!
//! Today this is just the bundled level files — the source-tree path
//! `vendor/pop-apple2/04 Support/Levels/LEVEL0..LEVEL14`. The full
//! discovery surface (env vars, hash registry, system paths, file
//! picker) lands with #85.

use std::path::{Path, PathBuf};

/// Environment variable that overrides level-directory discovery.
pub const LEVELS_DIR_ENV: &str = "POP_LEVELS_DIR";

/// Number of bundled levels (LEVEL0 through LEVEL14 inclusive).
pub const BUNDLED_LEVEL_COUNT: u8 = 15;

/// Locate the directory containing the 15 `LEVEL{N}` files.
///
/// Looks in priority order:
///
/// 1. The `POP_LEVELS_DIR` environment variable (allows users to point
///    at an externally-installed copy).
/// 2. `<repo_root>/vendor/pop-apple2/04 Support/Levels/`, resolved
///    relative to this crate's manifest. Works for in-tree development
///    and `cargo run` from the workspace.
///
/// Returns `None` when nothing exists at either location.
#[must_use]
pub fn bundled_levels_dir() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(LEVELS_DIR_ENV) {
        let p = PathBuf::from(path);
        if levels_dir_looks_valid(&p) {
            return Some(p);
        }
    }
    let in_tree = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../vendor/pop-apple2/04 Support/Levels");
    if levels_dir_looks_valid(&in_tree) {
        return Some(in_tree);
    }
    None
}

/// `Some(path)` for level `n` (0..=14) inside the bundled directory,
/// or `None` if either the directory or the specific level file is
/// missing.
#[must_use]
pub fn bundled_level_path(n: u8) -> Option<PathBuf> {
    let dir = bundled_levels_dir()?;
    let p = dir.join(format!("LEVEL{n}"));
    p.is_file().then_some(p)
}

/// All bundled level paths in numerical order, skipping any that don't
/// exist on disk.
#[must_use]
pub fn bundled_level_paths() -> Vec<PathBuf> {
    (0..BUNDLED_LEVEL_COUNT).filter_map(bundled_level_path).collect()
}

/// A `Levels` directory is considered valid when it exists and contains
/// at least `LEVEL0` (the minimal smoke check — a partial install is
/// still usable, but an empty / wrong directory is not).
fn levels_dir_looks_valid(p: &Path) -> bool {
    p.is_dir() && p.join("LEVEL0").is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_tree_discovery_finds_all_bundled_levels() {
        let dir = bundled_levels_dir().expect("repo-local bundled level dir");
        assert!(dir.join("LEVEL0").is_file());
        assert!(dir.join("LEVEL14").is_file());
        let paths = bundled_level_paths();
        assert_eq!(paths.len(), usize::from(BUNDLED_LEVEL_COUNT));
    }

    #[test]
    fn bundled_level_path_returns_specific_files() {
        let p = bundled_level_path(1).expect("LEVEL1 present");
        assert!(p.ends_with("LEVEL1"));
    }

    #[test]
    fn missing_level_index_returns_none() {
        assert!(bundled_level_path(99).is_none());
    }
}
