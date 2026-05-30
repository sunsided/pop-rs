//! Locate POP game assets on the host system.
//!
//! POP's assets live in a directory tree that matches the layout of
//! Jordan Mechner's 1989 source release — under what the release calls
//! `04 Support/`, with sibling `Levels/` and `DRAZ/` subdirectories
//! holding the level binaries (#80) and sprite / screen data (#87)
//! respectively. We call that the **POP data root**, and discovery is
//! the job of finding one on the user's machine without prompting.
//!
//! # Priority order
//!
//! [`discover_data_roots`] walks these locations and returns every
//! valid hit, in this order:
//!
//! 1. `$POP_DATA_DIR` — explicit user override pointing at a data
//!    root.
//! 2. `$POP_LEVELS_DIR` — the legacy variable from #80 (#101). When set,
//!    we treat its parent directory as a candidate data root so the
//!    older "just point at Levels/" workflow keeps working.
//! 3. The repo-local `vendor/pop-apple2/04 Support/` tree, resolved
//!    relative to this crate's manifest. Works for in-tree development
//!    and `cargo run` from the workspace.
//! 4. The user's data dir (XDG `data_dir()` on Linux, `Application
//!    Support/` on macOS, `%APPDATA%` on Windows), namespaced as
//!    `pop-rs/data`.
//! 5. `~/Documents/Apple II/POP/` — a convention used by several
//!    Apple II emulator pipelines.
//!
//! Single-best lookup is [`primary_data_root`]. Asset-family helpers
//! (`levels_dir_in`, `draz_dir_in`) take a root and return the
//! subdirectory if it exists.
//!
//! # What's *not* here
//!
//! * **Disk-image parsing & hash registry.** Identifying canonical
//!   `.woz` / `.dsk` images by SHA-256 is gated on the disk-image
//!   reader (#84) — without parsing we can't normalise sector
//!   ordering and the hash would be format-fragile. The hash-registry
//!   API spec'd in #85 lands with that work.
//! * **System file picker.** `rfd`-backed `pick_interactive()` is a
//!   UI concern; it lands with the editor (#90).
//!
//! The legacy `bundled_levels_dir` / `bundled_level_paths` /
//! `bundled_level_path` functions stay as thin shims over the new API
//! so existing callers (`pop info`, the tests) keep compiling.

use std::path::{Path, PathBuf};

/// Environment variable that overrides POP data-root discovery.
///
/// Set this to the directory containing `Levels/` and `DRAZ/`
/// (typically `04 Support/` in a Mechner-style source layout). Takes
/// precedence over every other lookup.
pub const DATA_DIR_ENV: &str = "POP_DATA_DIR";

/// Legacy environment variable from #80 / #101. Points at the
/// `Levels/` directory directly. When set, its parent is treated as a
/// candidate data root so `Levels/` *and* a sibling `DRAZ/` both
/// resolve.
pub const LEVELS_DIR_ENV: &str = "POP_LEVELS_DIR";

/// Subdirectory name carrying the 15 `LEVEL{N}` binaries inside a POP
/// data root.
pub const LEVELS_SUBDIR: &str = "Levels";

/// Subdirectory name carrying the sprite / screen data
/// (`IMG.CHTAB*`, `IMG.BGTAB*`, hi-res screens) inside a POP data
/// root.
pub const DRAZ_SUBDIR: &str = "DRAZ";

/// Application namespace used when probing the user's
/// platform-standard data directory.
const APP_DIR_QUALIFIER: &str = "";
const APP_DIR_ORG: &str = "";
const APP_DIR_NAME: &str = "pop-rs";

/// Number of bundled levels (LEVEL0 through LEVEL14 inclusive).
pub const BUNDLED_LEVEL_COUNT: u8 = 15;

/// Where a discovered POP data root came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiscoverySource {
    /// `POP_DATA_DIR` environment variable.
    DataDirEnv,
    /// `POP_LEVELS_DIR` environment variable (parent directory used).
    LegacyLevelsEnv,
    /// `<repo>/vendor/pop-apple2/04 Support/` resolved from this
    /// crate's `CARGO_MANIFEST_DIR`.
    InTree,
    /// Platform-standard user data directory (XDG /
    /// `Application Support/` / `%APPDATA%`) namespaced as
    /// `pop-rs/data`.
    UserDataDir,
    /// `~/Documents/Apple II/POP/`.
    DocumentsConvention,
}

impl DiscoverySource {
    /// Short, lowercase tag suitable for diagnostics and CLI output.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::DataDirEnv => "env:POP_DATA_DIR",
            Self::LegacyLevelsEnv => "env:POP_LEVELS_DIR",
            Self::InTree => "in-tree",
            Self::UserDataDir => "user-data-dir",
            Self::DocumentsConvention => "documents",
        }
    }
}

/// A POP data root located by [`discover_data_roots`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataRoot {
    /// Filesystem path to the root (a directory containing at least
    /// `Levels/`).
    pub path: PathBuf,
    /// Where this hit came from.
    pub source: DiscoverySource,
}

/// Walk the discovery priority chain and return **every** valid data
/// root found, in priority order. Duplicate paths (e.g. when
/// `POP_DATA_DIR` and `POP_LEVELS_DIR/..` resolve to the same place)
/// are de-duplicated, keeping the earliest source.
#[must_use]
pub fn discover_data_roots() -> Vec<DataRoot> {
    let mut out: Vec<DataRoot> = Vec::new();
    let push = |path: PathBuf, source: DiscoverySource, sink: &mut Vec<DataRoot>| {
        if !data_root_looks_valid(&path) {
            return;
        }
        let canonical = path.canonicalize().unwrap_or(path);
        if sink.iter().any(|r| r.path == canonical) {
            return;
        }
        sink.push(DataRoot {
            path: canonical,
            source,
        });
    };

    if let Some(p) = std::env::var_os(DATA_DIR_ENV) {
        push(PathBuf::from(p), DiscoverySource::DataDirEnv, &mut out);
    }
    if let Some(p) = std::env::var_os(LEVELS_DIR_ENV) {
        if let Some(parent) = PathBuf::from(p).parent() {
            push(
                parent.to_path_buf(),
                DiscoverySource::LegacyLevelsEnv,
                &mut out,
            );
        }
    }
    push(in_tree_root(), DiscoverySource::InTree, &mut out);
    for p in user_data_dir_candidates() {
        push(p, DiscoverySource::UserDataDir, &mut out);
    }
    for p in documents_convention_candidates() {
        push(p, DiscoverySource::DocumentsConvention, &mut out);
    }

    out
}

/// The single best data root from [`discover_data_roots`], or `None`
/// if nothing was found.
#[must_use]
pub fn primary_data_root() -> Option<DataRoot> {
    discover_data_roots().into_iter().next()
}

/// `Some(<root>/Levels)` if `root` carries the levels subdirectory.
#[must_use]
pub fn levels_dir_in(root: &Path) -> Option<PathBuf> {
    let p = root.join(LEVELS_SUBDIR);
    levels_dir_looks_valid(&p).then_some(p)
}

/// `Some(<root>/DRAZ)` if `root` carries the DRAZ subdirectory.
#[must_use]
pub fn draz_dir_in(root: &Path) -> Option<PathBuf> {
    let p = root.join(DRAZ_SUBDIR);
    p.is_dir().then_some(p)
}

/// True when `root` is a plausible POP data root — at minimum it
/// must carry a `Levels/LEVEL0` file. (DRAZ is recommended but not
/// strictly required; a partial install with only level binaries is
/// still useful for `pop info`.)
#[must_use]
pub fn data_root_looks_valid(root: &Path) -> bool {
    root.is_dir() && levels_dir_looks_valid(&root.join(LEVELS_SUBDIR))
}

/// `Levels/` is valid when it exists and carries at least `LEVEL0`.
fn levels_dir_looks_valid(p: &Path) -> bool {
    p.is_dir() && p.join("LEVEL0").is_file()
}

fn in_tree_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor/pop-apple2/04 Support")
}

fn user_data_dir_candidates() -> Vec<PathBuf> {
    let Some(dirs) = directories::ProjectDirs::from(APP_DIR_QUALIFIER, APP_DIR_ORG, APP_DIR_NAME)
    else {
        return Vec::new();
    };
    vec![dirs.data_dir().join("data"), dirs.data_dir().to_path_buf()]
}

fn documents_convention_candidates() -> Vec<PathBuf> {
    let Some(user) = directories::UserDirs::new() else {
        return Vec::new();
    };
    let Some(docs) = user.document_dir() else {
        return Vec::new();
    };
    vec![docs.join("Apple II").join("POP")]
}

// ---------------------------------------------------------------------------
// Backwards-compat: thin shims over the new API for callers from #80 / #101.
// ---------------------------------------------------------------------------

/// Locate the directory containing the 15 `LEVEL{N}` files.
///
/// Implemented as `levels_dir_in(primary_data_root())` plus a
/// `POP_LEVELS_DIR` fast-path so existing scripts that point this env
/// var directly at a `Levels/` directory keep working without
/// requiring a sibling `DRAZ/`.
#[must_use]
pub fn bundled_levels_dir() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(LEVELS_DIR_ENV) {
        let p = PathBuf::from(path);
        if levels_dir_looks_valid(&p) {
            return Some(p);
        }
    }
    let root = primary_data_root()?;
    levels_dir_in(&root.path)
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
    let Some(dir) = bundled_levels_dir() else {
        return Vec::new();
    };
    (0..BUNDLED_LEVEL_COUNT)
        .filter_map(|n| {
            let p = dir.join(format!("LEVEL{n}"));
            p.is_file().then_some(p)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_tree_discovery_finds_a_data_root() {
        // Running tests from the workspace must always find the
        // vendor tree via the in-tree fallback. Other sources may or
        // may not be set; we just assert the in-tree hit shows up.
        let roots = discover_data_roots();
        assert!(
            roots.iter().any(|r| r.source == DiscoverySource::InTree),
            "discover_data_roots did not return an in-tree root: {roots:?}"
        );
        let primary = primary_data_root().expect("at least one data root");
        assert!(data_root_looks_valid(&primary.path));
    }

    #[test]
    fn in_tree_root_exposes_both_subdirs() {
        let primary = primary_data_root().expect("a primary data root");
        assert!(
            levels_dir_in(&primary.path).is_some(),
            "Levels/ missing from {}",
            primary.path.display()
        );
        assert!(
            draz_dir_in(&primary.path).is_some(),
            "DRAZ/ missing from {}",
            primary.path.display()
        );
    }

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

    #[test]
    fn data_root_looks_valid_rejects_empty_dir() {
        let tmp = std::env::temp_dir().join("pop_rs_discovery_test_empty");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        assert!(!data_root_looks_valid(&tmp));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn discovery_source_tags_are_distinct() {
        let all = [
            DiscoverySource::DataDirEnv,
            DiscoverySource::LegacyLevelsEnv,
            DiscoverySource::InTree,
            DiscoverySource::UserDataDir,
            DiscoverySource::DocumentsConvention,
        ];
        let mut tags: Vec<&str> = all.iter().map(|s| s.tag()).collect();
        tags.sort_unstable();
        let before = tags.len();
        tags.dedup();
        assert_eq!(tags.len(), before, "DiscoverySource tags must be unique");
    }
}
