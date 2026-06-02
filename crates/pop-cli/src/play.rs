//! `pop play` — windowed game host (Path B, #79 / #91).
//!
//! The first runtime milestone: open a real window, run a frame loop,
//! and display a 280×192 [`pop_assets::hires::Frame`] scaled up with
//! nearest-neighbour pixels. The gameplay logic lives in the
//! [`pop_rs::World`] engine; this file is only the host shell — asset
//! loading, the eframe window, the input map, and the blit.
//!
//! Today it boots straight into a level and lets the arrow keys page
//! through its rooms, proving the window → loop → input → render path
//! end to end. The title-screen image and a real Prince hang off the
//! same pipeline in follow-ups.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::anyhow;
use clap::Args as ClapArgs;
use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, Stroke, TextureHandle, Vec2};

use pop_assets::bgdata::Biome;
use pop_assets::discovery;
use pop_assets::draz::image_table::ImageTable;
use pop_assets::hires::RenderMode;
use pop_assets::level::Level;
use pop_assets::scene::BiomeTables;
use pop_rs::backend::InputState;
use pop_rs::{PrinceDebug, World};

/// Arguments for the `play` subcommand.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// POP data root to load from. Overrides discovery. Should be a
    /// directory containing `Levels/` and `DRAZ/`.
    #[arg(value_name = "PATH")]
    pub path: Option<PathBuf>,
    /// Level to boot into (the `LEVEL{n}` file suffix, 0-based).
    /// Defaults to level 1, the first dungeon.
    #[arg(long, default_value_t = 1)]
    pub level: u8,
    /// Render in monochrome instead of the NTSC artifact palette.
    #[arg(long)]
    pub mono: bool,
}

/// Run the `play` subcommand.
///
/// # Errors
///
/// Bubbles up data-root discovery, level-parse, and sprite-load
/// failures, plus the eframe initialisation error.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let root = match &args.path {
        Some(p) => p.clone(),
        None => discovery::primary_data_root()
            .map(|r| r.path)
            .ok_or_else(|| anyhow!("no POP data root found; pass one as an argument"))?,
    };

    let levels_dir = discovery::levels_dir_in(&root)
        .ok_or_else(|| anyhow!("no Levels/ directory under {}", root.display()))?;
    let level_path = levels_dir.join(format!("LEVEL{}", args.level));
    let level = Level::from_file(&level_path)
        .map_err(|e| anyhow!("failed to load {}: {e}", level_path.display()))?;

    let biome = Biome::for_level(usize::from(args.level))
        .ok_or_else(|| anyhow!("LEVEL{} has no biome mapping", args.level))?;
    let tables = load_tables(&root, biome)?;

    let mode = if args.mono {
        RenderMode::Monochrome
    } else {
        RenderMode::NtscColor
    };
    let mut world = World::new(level, tables);
    let chtabs = load_chtabs(&root);
    if chtabs.iter().any(Option::is_some) {
        world = world.with_chtabs(chtabs);
    }
    let app = GameApp::new(world, mode);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Prince of Persia")
            .with_inner_size([
                f32::from(FRAME_W) * DEFAULT_SCALE,
                f32::from(FRAME_H) * DEFAULT_SCALE,
            ]),
        ..Default::default()
    };
    eframe::run_native("pop play", options, Box::new(|_cc| Ok(Box::new(app))))
        .map_err(|e| anyhow!("eframe failed to launch: {e}"))
}

/// Load a biome's sprite tables, attaching a dungeon fallback for the
/// tower biome whose 3.5"-rebuild tables are partly truncated (#112) —
/// same workaround the editor applies.
fn load_tables(root: &std::path::Path, biome: Biome) -> anyhow::Result<BiomeTables> {
    let tables = BiomeTables::load(root, biome)
        .map_err(|e| anyhow!("failed to load {} sprites: {e}", biome.name()))?;
    if biome == Biome::Tower {
        if let Ok(fallback) = BiomeTables::load(root, Biome::Dungeon) {
            return Ok(tables.with_fallback(fallback));
        }
    }
    Ok(tables)
}

/// Load the character sprite tables (`IMG.CHTAB1..8`) from `DRAZ/I` for the
/// animation engine to index by frame (`pop_assets::anim::frame_sprite`).
/// Slot `i` holds CHTAB `i+1`; a missing table is a `None` slot — non-fatal,
/// an unresolved frame just renders the bare scene rather than refusing to
/// start. CHTAB1-3 carry the kid; slot 4 (guard body) is left empty, and
/// 5-8 (shared / combat art) load best-effort for later use.
fn load_chtabs(root: &std::path::Path) -> Vec<Option<ImageTable>> {
    let Some(dir) = discovery::draz_dir_in(root).map(|d| d.join("I")) else {
        return Vec::new();
    };
    let load = |name: &str| ImageTable::from_file(dir.join(name)).ok();
    vec![
        load("IMG.CHTAB1"),
        load("IMG.CHTAB2"),
        load("IMG.CHTAB3"),
        // Slot 4 is the guard *body* (`IMG.CHTAB4.<variant>`, chosen per
        // guard type — `.GD` / `.FAT` / …), not the Prince's. Leave it empty
        // until the runtime spawns guards and knows which body to load.
        None,
        load("IMG.CHTAB5"),
        load("IMG.CHTAB6.A"),
        load("IMG.CHTAB7"),
        load("IMG.CHTAB8"),
    ]
}

/// POP's hi-res frame is 280×192; the window opens at this integer
/// scale and letterboxes any leftover space.
const FRAME_W: u16 = 280;
const FRAME_H: u16 = 192;
const DEFAULT_SCALE: f32 = 3.0;

/// Fixed logic-tick cadence. egui may call `update` at the display
/// refresh rate (often 60–144 Hz), but the game logic must advance at a
/// deterministic, host-independent rate, so `update` steps the world at
/// most once per `TICK`. ~18 Hz, close to POP's ~17 fps logic rate (#82);
/// pinned exactly when the controller's timing is tuned (#94).
const TICK: Duration = Duration::from_millis(55);

/// eframe application: owns the engine and the GPU texture for the
/// current frame, re-uploading only when the displayed room changes.
struct GameApp {
    world: World,
    mode: RenderMode,
    texture: Option<TextureHandle>,
    /// Wall-clock time of the last logic tick; `None` until the first.
    last_tick: Option<Instant>,
    /// Debug overlay (cell grid + the Prince's logical-x / collision / figure
    /// edges), toggled with `G`.
    debug: bool,
}

impl GameApp {
    fn new(world: World, mode: RenderMode) -> Self {
        Self {
            world,
            mode,
            texture: None,
            last_tick: None,
            debug: false,
        }
    }

    /// Re-render the current world state and upload it to the texture.
    fn refresh_texture(&mut self, ctx: &egui::Context) {
        let Some(frame) = self.world.render(self.mode) else {
            return;
        };
        let size = [
            usize::try_from(frame.width).unwrap_or(0),
            usize::try_from(frame.height).unwrap_or(0),
        ];
        let image = ColorImage::from_rgba_unmultiplied(size, &frame.pixels);
        self.texture = Some(ctx.load_texture("pop-frame", image, egui::TextureOptions::NEAREST));
    }
}

impl eframe::App for GameApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Read input inside the `ctx.input` closure; act on Escape
        // afterwards — calling `ctx` methods while the input lock is
        // held can deadlock.
        let (input, quit, toggle_debug) = ctx.input(|i| {
            let state = InputState {
                left: i.key_down(egui::Key::ArrowLeft),
                right: i.key_down(egui::Key::ArrowRight),
                up: i.key_down(egui::Key::ArrowUp),
                down: i.key_down(egui::Key::ArrowDown),
                shift: i.modifiers.shift,
            };
            (
                state,
                i.key_pressed(egui::Key::Escape),
                i.key_pressed(egui::Key::G),
            )
        });
        if quit {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        if toggle_debug {
            self.debug = !self.debug;
        }

        // Advance the world at a fixed cadence rather than at the host
        // repaint rate. egui can call `update` far more often than the
        // logic should step; `request_repaint_after(TICK)` below keeps
        // it ticking without busy-spinning. (Mirrors the editor's
        // ANIM_STEP gate.)
        let now = Instant::now();
        let waiting = self.last_tick.is_some_and(|t| now.duration_since(t) < TICK);
        if !waiting {
            self.last_tick = Some(now);
            self.world.tick(input);
            // The world is dynamic now (the Prince moves), so re-render
            // every tick — not just when the room changes.
            self.refresh_texture(ctx);
        }

        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(Color32::BLACK))
            .show(ctx, |ui| {
                let Some(texture) = &self.texture else { return };
                let avail = ui.available_size();
                let scale = (avail.x / f32::from(FRAME_W))
                    .min(avail.y / f32::from(FRAME_H))
                    .floor()
                    .max(1.0);
                let draw = Vec2::new(f32::from(FRAME_W) * scale, f32::from(FRAME_H) * scale);
                let origin = ui.min_rect().min + ((avail - draw) * 0.5).max(Vec2::ZERO);
                let rect = Rect::from_min_size(origin, draw);
                ui.painter().image(
                    texture.id(),
                    rect,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
                if self.debug {
                    draw_debug_overlay(ui, &self.world.prince_debug(), origin, draw, scale);
                }
            });

        // Wake again in time for the next logic tick (no busy-spin).
        ctx.request_repaint_after(TICK);
    }
}

/// Draw the debug overlay over the scaled frame: a dim-yellow cell grid, the
/// Prince's collision-box edges (cyan), his drawn-figure edges (magenta), his
/// logical centre x (green) and his feet line (white). Lets you see, in the
/// running game, exactly where his logical position and the cell boundaries
/// sit relative to the figure that's drawn.
#[allow(clippy::cast_precision_loss)]
fn draw_debug_overlay(ui: &egui::Ui, d: &PrinceDebug, origin: Pos2, draw: Vec2, scale: f32) {
    let painter = ui.painter();
    let (top, bot) = (origin.y, origin.y + draw.y);
    let sx = |px: i32| origin.x + px as f32 * scale;
    for c in 0..=i32::from(FRAME_W) / d.cell_w {
        painter.vline(
            sx(c * d.cell_w),
            top..=bot,
            Stroke::new(1.0, Color32::from_rgb(80, 80, 0)),
        );
    }
    let cyan = Color32::from_rgb(0, 210, 210);
    painter.vline(sx(d.x - d.half_w), top..=bot, Stroke::new(1.0, cyan));
    painter.vline(sx(d.x + d.half_w), top..=bot, Stroke::new(1.0, cyan));
    let magenta = Color32::from_rgb(255, 0, 255);
    painter.vline(sx(d.fig_left), top..=bot, Stroke::new(1.0, magenta));
    painter.vline(sx(d.fig_right), top..=bot, Stroke::new(1.0, magenta));
    painter.vline(
        sx(d.x),
        top..=bot,
        Stroke::new(1.5, Color32::from_rgb(0, 255, 0)),
    );
    let fy = origin.y + d.feet_y as f32 * scale;
    painter.hline(
        origin.x..=origin.x + draw.x,
        fy,
        Stroke::new(1.0, Color32::WHITE),
    );
}
