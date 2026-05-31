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
use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, TextureHandle, Vec2};

use pop_assets::bgdata::Biome;
use pop_assets::discovery;
use pop_assets::draz::image_table::ImageTable;
use pop_assets::hires::RenderMode;
use pop_assets::level::Level;
use pop_assets::scene::BiomeTables;
use pop_rs::backend::InputState;
use pop_rs::{KidArt, World};

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
    if let Some(art) = load_kid_art(&root) {
        world = world.with_kid_art(art);
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

/// CHTAB indices for the kid frames the renderer draws, traced from the
/// engine. FRAMEDEF maps a frame to `(table, image)` via `Fimage` /
/// `Fsword` (`CTRLSUBS.S decodeim`); POP image tables are 1-based, so the
/// 0-based index is `image - 1`. The run frames (`Fsword = 0`) live in
/// CHTAB1 alongside `stand`.
///
/// - `stand` = FRAMEDEF 15 (`$0f,9`) → CHTAB1 image 15 → index 14.
/// - `run` = SEQTABLE `runcyc1..8` = FRAMEDEF 7-14 → CHTAB1 images 7-14
///   → indices 6..14.
/// - `freefall` = FRAMEDEF 106 (`$36,$40`) → CHTAB2 image 54 → index 53.
const STAND_INDEX: usize = 14;
const FALL_INDEX: usize = 53;
const RUN_INDICES: std::ops::Range<usize> = 6..14;

/// Load the kid sprite set (`stand` + run cycle + `freefall`) from
/// `DRAZ/I`, or `None` if a table or frame is missing — non-fatal, the
/// host then renders the bare scene rather than refusing to start.
fn load_kid_art(root: &std::path::Path) -> Option<KidArt> {
    let dir = discovery::draz_dir_in(root)?.join("I");
    let chtab1 = ImageTable::from_file(dir.join("IMG.CHTAB1")).ok()?;
    let chtab2 = ImageTable::from_file(dir.join("IMG.CHTAB2")).ok()?;
    let run = RUN_INDICES
        .map(|i| chtab1.images.get(i).cloned())
        .collect::<Option<Vec<_>>>()?;
    Some(KidArt {
        stand: chtab1.images.get(STAND_INDEX)?.clone(),
        fall: chtab2.images.get(FALL_INDEX)?.clone(),
        run,
    })
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
}

impl GameApp {
    fn new(world: World, mode: RenderMode) -> Self {
        Self {
            world,
            mode,
            texture: None,
            last_tick: None,
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
        let (input, quit) = ctx.input(|i| {
            let state = InputState {
                left: i.key_down(egui::Key::ArrowLeft),
                right: i.key_down(egui::Key::ArrowRight),
                up: i.key_down(egui::Key::ArrowUp),
                down: i.key_down(egui::Key::ArrowDown),
                shift: i.modifiers.shift,
            };
            (state, i.key_pressed(egui::Key::Escape))
        });
        if quit {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
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
            });

        // Wake again in time for the next logic tick (no busy-spin).
        ctx.request_repaint_after(TICK);
    }
}
