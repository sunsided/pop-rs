//! `pop draz` — extract / preview POP's `DRAZ/` sprite and screen assets.
//!
//! Three sub-actions:
//!
//! * `inspect <FILE>` — print a structural summary (image count, per-
//!   sprite dimensions). Auto-detects screens vs. image tables by size.
//! * `screen  <FILE> --out X.png` — render an 8 KiB hi-res page file
//!   to a PNG via [`pop_assets::hires::render`].
//! * `sprites <FILE> --out-dir DIR` — parse an `IMG.CHTAB*` /
//!   `IMG.BGTAB*` table and dump every sprite as `NNN.png` via
//!   [`pop_assets::hires::render_linear`].
//!
//! All rendering commands accept `--mode mono|ntsc` (default `ntsc`).

use std::path::{Path, PathBuf};

use clap::{Args as ClapArgs, Subcommand, ValueEnum};
use pop_assets::{
    draz::{image_table::ImageTable, screen},
    hires::{render, render_linear, Frame, RenderMode, HIRES_PAGE_BYTES},
};

/// Arguments for the `draz` subcommand.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Action to perform.
    #[command(subcommand)]
    pub action: Action,
}

/// Sub-actions of `pop draz`.
#[derive(Debug, Subcommand)]
pub enum Action {
    /// Print a structural summary of a DRAZ asset file. Auto-detects
    /// 8 KiB hi-res screens vs. variable-size image tables.
    Inspect(InspectArgs),
    /// Render an 8 KiB hi-res screen file to a PNG.
    Screen(ScreenArgs),
    /// Dump every sprite in an `IMG.CHTAB*` / `IMG.BGTAB*` table as
    /// individual PNGs into an output directory.
    Sprites(SpritesArgs),
}

/// Render colour mode, exposed on the CLI.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Mode {
    /// Pure black / white. Best for verifying raw sprite shape.
    Mono,
    /// Apple II NTSC 6-colour artifact palette (see
    /// [`pop_assets::hires`] for the color-cell approximation we use).
    Ntsc,
}

impl From<Mode> for RenderMode {
    fn from(m: Mode) -> Self {
        match m {
            Mode::Mono => RenderMode::Monochrome,
            Mode::Ntsc => RenderMode::NtscColor,
        }
    }
}

/// `pop draz inspect` arguments.
#[derive(Debug, ClapArgs)]
pub struct InspectArgs {
    /// Path to the DRAZ asset file.
    #[arg(value_name = "PATH")]
    pub path: PathBuf,
}

/// `pop draz screen` arguments.
#[derive(Debug, ClapArgs)]
pub struct ScreenArgs {
    /// Path to an 8 KiB hi-res page file (e.g. `DRAZ/SS/SS0`).
    #[arg(value_name = "PATH")]
    pub path: PathBuf,
    /// Output PNG path. Defaults to `<PATH>.png`.
    #[arg(short, long, value_name = "PNG")]
    pub out: Option<PathBuf>,
    /// Render mode. `ntsc` matches what an Apple II monitor would show;
    /// `mono` is best for inspecting raw bit patterns.
    #[arg(long, value_enum, default_value_t = Mode::Ntsc)]
    pub mode: Mode,
}

/// `pop draz sprites` arguments.
#[derive(Debug, ClapArgs)]
pub struct SpritesArgs {
    /// Path to an `IMG.CHTAB*` / `IMG.BGTAB*` image-table file.
    #[arg(value_name = "PATH")]
    pub path: PathBuf,
    /// Output directory. One PNG per sprite, named `000.png`,
    /// `001.png`, … in directory order (corresponding to image #1, #2,
    /// … in the 6502 sources).
    #[arg(short = 'o', long, value_name = "DIR")]
    pub out_dir: PathBuf,
    /// Render mode.
    #[arg(long, value_enum, default_value_t = Mode::Ntsc)]
    pub mode: Mode,
}

/// Run the `draz` subcommand.
///
/// # Errors
///
/// Bubbles up I/O, parse, and PNG-encode failures from the underlying
/// readers and writers.
pub fn run(args: &Args) -> anyhow::Result<()> {
    match &args.action {
        Action::Inspect(a) => run_inspect(a),
        Action::Screen(a) => run_screen(a),
        Action::Sprites(a) => run_sprites(a),
    }
}

fn run_inspect(args: &InspectArgs) -> anyhow::Result<()> {
    let bytes = std::fs::read(&args.path)?;
    println!("{}", args.path.display());
    println!("  size: {} bytes", bytes.len());
    if bytes.len() == HIRES_PAGE_BYTES {
        println!("  kind: 8 KiB Apple II hi-res screen");
        return Ok(());
    }
    match ImageTable::from_bytes(&bytes) {
        Ok(table) => {
            println!(
                "  kind: image table, {} sprites, base ${:04x}",
                table.images.len(),
                table.base_address,
            );
            let (max_w, max_h) = table.images.iter().fold((0u16, 0u16), |(mw, mh), img| {
                (mw.max(img.width_pixels()), mh.max(u16::from(img.height)))
            });
            println!("  max sprite size: {max_w} x {max_h} pixels");
            for (i, img) in table.images.iter().enumerate() {
                println!(
                    "    [{i:>3}] {:>2}x{:<3} px  ({}b x {} lines, {} bytes)",
                    img.width_pixels(),
                    img.height,
                    img.width_bytes,
                    img.height,
                    img.bitmap.len(),
                );
            }
        }
        Err(e) => {
            println!("  kind: unrecognised ({e})");
        }
    }
    Ok(())
}

fn run_screen(args: &ScreenArgs) -> anyhow::Result<()> {
    let bytes = screen::from_file(&args.path)?;
    let frame = render(&bytes, args.mode.into());
    let out = args
        .out
        .clone()
        .unwrap_or_else(|| default_png_path(&args.path));
    write_png(&out, &frame)?;
    println!("{} -> {}", args.path.display(), out.display());
    Ok(())
}

fn run_sprites(args: &SpritesArgs) -> anyhow::Result<()> {
    let table = ImageTable::from_file(&args.path)?;
    std::fs::create_dir_all(&args.out_dir)?;
    let width = digit_width(table.images.len());
    for (i, img) in table.images.iter().enumerate() {
        if img.width_bytes == 0 || img.height == 0 {
            // Empty placeholder slot (some POP sprites are 0×0).
            continue;
        }
        let Some(frame) = render_linear(&img.bitmap, img.width_bytes, img.height, args.mode.into())
        else {
            anyhow::bail!(
                "sprite {i}: bitmap length {} doesn't match {}x{} header",
                img.bitmap.len(),
                img.width_bytes,
                img.height,
            );
        };
        let out = args.out_dir.join(format!("{i:0width$}.png"));
        write_png(&out, &frame)?;
    }
    println!(
        "{} -> {} ({} sprites)",
        args.path.display(),
        args.out_dir.display(),
        table.images.len(),
    );
    Ok(())
}

fn default_png_path(input: &Path) -> PathBuf {
    let mut name = input.file_name().map_or_else(
        || std::ffi::OsString::from("screen"),
        std::ffi::OsStr::to_os_string,
    );
    name.push(".png");
    input.with_file_name(name)
}

fn digit_width(n: usize) -> usize {
    // Pad index strings so the directory sorts naturally.
    if n <= 1 {
        1
    } else {
        let mut digits = 0;
        let mut x = n - 1;
        while x > 0 {
            digits += 1;
            x /= 10;
        }
        digits.max(1)
    }
}

fn write_png(path: &Path, frame: &Frame) -> anyhow::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), frame.width, frame.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&frame.pixels)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digit_width_handles_edges() {
        assert_eq!(digit_width(0), 1);
        assert_eq!(digit_width(1), 1);
        assert_eq!(digit_width(2), 1);
        assert_eq!(digit_width(10), 1);
        assert_eq!(digit_width(11), 2);
        assert_eq!(digit_width(100), 2);
        assert_eq!(digit_width(101), 3);
    }

    #[test]
    fn default_png_path_appends_extension() {
        let p = default_png_path(Path::new("/tmp/SS0"));
        assert_eq!(p, PathBuf::from("/tmp/SS0.png"));
    }
}
