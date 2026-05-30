//! `pop discover` — list POP data roots found on the host.
//!
//! Walks the priority chain in [`pop_assets::discovery`] and prints
//! every hit plus which asset families (`Levels/`, `DRAZ/`) are
//! present underneath each root. Useful for diagnosing
//! `POP_DATA_DIR` / `POP_LEVELS_DIR` overrides and for verifying that
//! an extracted install will be found by the rest of the toolkit.

use clap::Args as ClapArgs;
use pop_assets::discovery;

/// Arguments for the `discover` subcommand.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Print only the primary (highest-priority) data root.
    #[arg(long)]
    pub primary: bool,
}

/// Run the `discover` subcommand.
///
/// # Errors
///
/// Never fails today — kept as `Result<()>` for shape parity with the
/// other subcommands (so the `main` dispatch stays uniform).
#[allow(clippy::unnecessary_wraps)]
pub fn run(args: &Args) -> anyhow::Result<()> {
    let mut roots = discovery::discover_data_roots();
    if args.primary {
        roots.truncate(1);
    }

    if roots.is_empty() {
        println!("no POP data roots found");
        println!();
        println!("checked, in order:");
        println!("  1. ${}", discovery::DATA_DIR_ENV);
        println!("  2. ${} (parent dir)", discovery::LEVELS_DIR_ENV);
        println!(
            "  3. in-tree vendor/pop-apple2/04 Support/ (relative to pop-assets's CARGO_MANIFEST_DIR)"
        );
        println!("  4. platform user-data dir (pop-rs/data)");
        println!("  5. ~/Documents/Apple II/POP/");
        return Ok(());
    }

    println!("{} POP data root(s):", roots.len());
    for (i, root) in roots.iter().enumerate() {
        let levels = discovery::levels_dir_in(&root.path).is_some();
        let draz = discovery::draz_dir_in(&root.path).is_some();
        println!();
        println!("  [{}] {}", i + 1, root.path.display());
        println!("      source: {}", root.source.tag());
        println!(
            "      Levels/: {}    DRAZ/: {}",
            if levels { "yes" } else { "no" },
            if draz { "yes" } else { "no" },
        );
    }

    Ok(())
}
