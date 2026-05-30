//! `pop` — Prince of Persia (Apple II) toolkit.
//!
//! Subcommand tree:
//!
//! - `info` — identify and dump a level file's structure.
//! - `draz` — inspect / render POP's `DRAZ/` sprite and screen assets.
//! - `discover` — list POP data roots found on the host.
//! - `editor` — egui level browser (gated on the `editor` Cargo
//!   feature so headless / CI builds stay light).
//!
//! Future runtime subcommands (`play`, …) land behind their own Cargo
//! features the same way.

#![cfg_attr(not(test), warn(missing_docs))]

use clap::{Parser, Subcommand};

mod discover;
mod draz;
#[cfg(feature = "editor")]
mod editor;
mod info;

/// `pop` — Prince of Persia (Apple II) toolkit.
#[derive(Debug, Parser)]
#[command(name = "pop", version, about, long_about = None)]
struct Cli {
    /// Subcommand to run.
    #[command(subcommand)]
    cmd: Cmd,
}

/// Available subcommands.
#[derive(Debug, Subcommand)]
enum Cmd {
    /// Inspect a POP level file — print room layout summary and a
    /// tile-kind histogram.
    ///
    /// With no argument, scans every bundled `LEVEL{N}` file.
    Info(info::Args),
    /// Extract / preview POP's `DRAZ/` sprite and screen assets.
    Draz(draz::Args),
    /// List POP data roots found on the host (env vars, in-tree,
    /// system data dirs).
    Discover(discover::Args),
    /// Open the egui level browser.
    #[cfg(feature = "editor")]
    Editor(editor::Args),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Info(args) => info::run(&args),
        Cmd::Draz(args) => draz::run(&args),
        Cmd::Discover(args) => discover::run(&args),
        #[cfg(feature = "editor")]
        Cmd::Editor(args) => editor::run(&args),
    }
}
