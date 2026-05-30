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
        "  screens used (INFO[0] - 1): {}",
        level.screen_count_plus_one().saturating_sub(1),
    );

    let prince = level.prince_start();
    if let Some((col, row)) = prince.col_row() {
        println!(
            "  prince start: room {} tile {} ({}, {}) face=0x{:02x}",
            prince.screen, prince.block, col, row, prince.face_raw,
        );
    } else {
        println!(
            "  prince start: room {} tile {} (out of range) face=0x{:02x}",
            prince.screen, prince.block, prince.face_raw,
        );
    }

    match level.sword_start() {
        None => println!("  sword start:  none"),
        Some(sword) => match sword.col_row() {
            Some((col, row)) => println!(
                "  sword start:  room {} tile {} ({}, {})",
                sword.screen, sword.block, col, row,
            ),
            None => println!(
                "  sword start:  room {} tile {} (out of range)",
                sword.screen, sword.block,
            ),
        },
    }

    let guards = level.guard_spawns();
    let guard_count = guards.iter().filter(|g| g.is_some()).count();
    println!("  guards:   {guard_count}");
    for (room_idx, guard) in guards.iter().enumerate() {
        if let Some(g) = guard {
            match g.col_row() {
                Some((col, row)) => println!(
                    "    room {:>2}: tile {:>2} ({}, {}) prog=0x{:02x} face=0x{:02x}",
                    room_idx + 1,
                    g.block,
                    col,
                    row,
                    g.prog,
                    g.face_raw,
                ),
                None => println!(
                    "    room {:>2}: tile {:>2} (out of range) prog=0x{:02x} face=0x{:02x}",
                    room_idx + 1,
                    g.block,
                    g.prog,
                    g.face_raw,
                ),
            }
        }
    }

    let hist = level.tile_kind_histogram();
    let mut entries: Vec<(TileKind, u32)> = (0..TileKind::COUNT)
        .filter_map(|i| TileKind::from_raw(i).map(|k| (k, hist[i as usize])))
        .filter(|(_, n)| *n > 0)
        .collect();
    entries.sort_by_key(|e| std::cmp::Reverse(e.1));

    println!("  tile-kind histogram (non-zero, by count):");
    for (kind, count) in entries {
        println!("    {:>11}  {count:>4}", kind.short_name());
    }
}
