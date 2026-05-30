//! `pop` — Prince of Persia (Apple II) toolkit.
//!
//! Subcommand tree:
//!
//! - `info` — identify and dump a level file's structure.
//! - `draz` — inspect / render POP's `DRAZ/` sprite and screen assets.
//! - `discover` — list POP data roots found on the host.
//!
//! Future subcommands (`editor`, `play`, …) land behind their own
//! Cargo features so headless / CI builds stay light.

#![cfg_attr(not(test), warn(missing_docs))]

use clap::{Parser, Subcommand};

mod discover;
mod draz;
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
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Info(args) => info::run(&args),
        Cmd::Draz(args) => draz::run(&args),
        Cmd::Discover(args) => discover::run(&args),
    }
}
