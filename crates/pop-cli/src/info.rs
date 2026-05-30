//! `pop info` — dump a level file's structure.

use std::path::PathBuf;

use clap::Args as ClapArgs;
use pop_assets::{
    discovery,
    level::{Level, TileKind, ROOMS_PER_LEVEL, ROOM_HEIGHT, ROOM_WIDTH},
};

/// Arguments for the `info` subcommand.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Level file(s) to inspect. If empty, the bundled `LEVEL{N}` set
    /// is used (looked up via `POP_LEVELS_DIR` then the repo-local
    /// `vendor/pop-apple2/04 Support/Levels/` directory).
    #[arg(value_name = "PATH")]
    pub paths: Vec<PathBuf>,
}

/// Run the `info` subcommand against `args`.
///
/// # Errors
///
/// Bubbles up I/O and parse errors from the underlying readers, plus
/// an explicit error when no level paths were given and no bundled
/// directory could be located.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let paths = resolve_paths(args)?;

    for (i, path) in paths.iter().enumerate() {
        if i > 0 {
            println!();
        }
        let level = Level::from_file(path)?;
        print_level(path, &level);
    }
    Ok(())
}

fn resolve_paths(args: &Args) -> anyhow::Result<Vec<PathBuf>> {
    if !args.paths.is_empty() {
        return Ok(args.paths.clone());
    }
    let bundled = discovery::bundled_level_paths();
    if bundled.is_empty() {
        anyhow::bail!(
            "no level paths given and no bundled levels found \
             (try setting POP_LEVELS_DIR or passing a path)"
        );
    }
    Ok(bundled)
}

fn print_level(path: &std::path::Path, level: &Level) {
    println!("{}", path.display());
    println!("  rooms:    {ROOMS_PER_LEVEL} ({ROOM_WIDTH}x{ROOM_HEIGHT} tiles each)");
    println!(
        "  metadata: {} raw bytes (not yet decoded)",
        level.raw_metadata().len()
    );

    let hist = level.tile_kind_histogram();
    let mut entries: Vec<(TileKind, u32)> = (0..TileKind::COUNT)
        .filter_map(|i| TileKind::from_raw(i).map(|k| (k, hist[i as usize])))
        .filter(|(_, n)| *n > 0)
        .collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1));

    println!("  tile-kind histogram (non-zero, by count):");
    for (kind, count) in entries {
        println!("    {:>11}  {count:>4}", kind.short_name());
    }
}
