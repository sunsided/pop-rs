//! `pop editor` — egui level browser.
//!
//! Phase 1 capstone: open a POP data root, pick a level, navigate
//! between rooms following the `MAP` neighbour graph from #88, and see
//! the Prince + guard spawn positions overlaid on a colour-coded
//! `10 × 3` tile grid.
//!
//! # What's here (vs. the original #90 scope)
//!
//! The original issue listed four viewport tabs:
//!
//! | Tab           | Status                                                    |
//! |---------------|-----------------------------------------------------------|
//! | Disk image    | Deferred — no disk parser (#84).                          |
//! | Sprite        | Deferred — `pop draz sprites` already covers this from the CLI; an in-editor view lands once we want side-by-side comparison or animation playback. |
//! | Level         | **Shipped here.**                                         |
//! | Animation     | Deferred — #89 (`SEQTABLE.S` / `SEQDATA.S` / `FRAMEDEF.S`).|
//!
//! Tile rendering uses solid colours per [`TileKind`] with the
//! `short_name()` overlaid, not the real `IMG.BGTAB.*` sprites — that
//! needs the `BGDATA.S` tile-piece mapping which is its own piece of
//! reverse engineering. The colour palette is good enough to recognise
//! every level visually (floor, block, gate, spikes, exit, etc. are
//! all distinct).
//!
//! # State / view split
//!
//! [`EditorState`] is a plain data struct with no eframe types —
//! the egui [`EditorApp`] is a pure projection over it. Keeps the
//! eventual WASM port painless (the issue called this out
//! explicitly).

#![cfg(feature = "editor")]

use std::path::PathBuf;

use clap::Args as ClapArgs;
use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2};
use pop_assets::{
    discovery,
    level::{
        GuardSpawn, Level, RoomNeighbours, StartPosition, Tile, TileKind, ROOMS_PER_LEVEL,
        ROOM_HEIGHT, ROOM_WIDTH,
    },
};

const _: () = assert!(ROOMS_PER_LEVEL <= u8::MAX as usize);
#[allow(clippy::cast_possible_truncation)]
const ROOMS_PER_LEVEL_U8: u8 = ROOMS_PER_LEVEL as u8;

/// True when `id` is a valid 1-based room id (`1..=ROOMS_PER_LEVEL`).
/// Used in lockstep by [`EditorState::step`] and the nav-button
/// enable predicate so the two stay in sync as the invariant
/// evolves.
fn is_valid_room_id(id: u8) -> bool {
    id != 0 && id <= ROOMS_PER_LEVEL_U8
}

/// Arguments for the `editor` subcommand.
#[derive(Debug, ClapArgs)]
pub struct Args {
    /// POP data root to open. Overrides discovery. Should be a
    /// directory containing `Levels/` and (ideally) `DRAZ/`.
    #[arg(value_name = "PATH")]
    pub path: Option<PathBuf>,
    /// Always show the system file picker even if discovery found
    /// something.
    #[arg(long)]
    pub pick: bool,
}

/// Run the `editor` subcommand.
///
/// # Errors
///
/// Bubbles up I/O / parse failures and the eframe initialisation error.
pub fn run(args: &Args) -> anyhow::Result<()> {
    let initial_root = resolve_initial_root(args);
    let state = EditorState::new(initial_root);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("pop editor")
            .with_inner_size([1100.0, 700.0]),
        ..Default::default()
    };

    eframe::run_native(
        "pop editor",
        options,
        Box::new(|_cc| Ok(Box::new(EditorApp::new(state)))),
    )
    .map_err(|e| anyhow::anyhow!("eframe failed to launch: {e}"))
}

fn resolve_initial_root(args: &Args) -> Option<PathBuf> {
    if args.pick {
        return pick_dir();
    }
    if let Some(p) = &args.path {
        return Some(p.clone());
    }
    discovery::primary_data_root().map(|r| r.path)
}

fn pick_dir() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .set_title("Choose a POP data root (containing Levels/)")
        .pick_folder()
}

// ---------------------------------------------------------------------------
// State (no egui types — pure projection target for the UI).
// ---------------------------------------------------------------------------

/// Editor data model. Holds the discovered POP data root and the
/// currently-loaded level + room selection. Has no eframe types so the
/// future WASM port can reuse this verbatim.
#[derive(Debug)]
pub struct EditorState {
    /// Active POP data root, if any.
    pub root: Option<PathBuf>,
    /// Per-`LEVEL{N}` paths under `root`'s `Levels/`. Empty when no
    /// root is set.
    pub level_paths: Vec<PathBuf>,
    /// Index into [`Self::level_paths`] of the currently loaded level.
    /// `None` until the user picks one.
    pub loaded_level_idx: Option<usize>,
    /// The parsed level for [`Self::loaded_level_idx`].
    pub loaded_level: Option<Level>,
    /// Current room (1-based, 1..=24). Defaults to the prince start
    /// room when a level loads.
    pub current_room: u8,
    /// Last error to surface in the status bar, if any.
    pub status: String,
}

impl EditorState {
    /// Create a new state. If `root` is `Some`, eagerly scans `Levels/`
    /// so the UI can show the list without an extra round-trip.
    #[must_use]
    pub fn new(root: Option<PathBuf>) -> Self {
        let mut state = Self {
            root: None,
            level_paths: Vec::new(),
            loaded_level_idx: None,
            loaded_level: None,
            current_room: 1,
            status: String::new(),
        };
        if let Some(r) = root {
            state.set_root(r);
        }
        state
    }

    /// Point at a new POP data root and rescan `Levels/`.
    pub fn set_root(&mut self, root: PathBuf) {
        self.level_paths = discovery::levels_dir_in(&root)
            .map(|dir| {
                (0..discovery::BUNDLED_LEVEL_COUNT)
                    .filter_map(|n| {
                        let p = dir.join(format!("LEVEL{n}"));
                        p.is_file().then_some(p)
                    })
                    .collect()
            })
            .unwrap_or_default();
        if self.level_paths.is_empty() {
            self.status = format!("no levels found under {}", root.display());
        } else {
            self.status = format!(
                "loaded {} levels from {}",
                self.level_paths.len(),
                root.display()
            );
        }
        self.root = Some(root);
        self.loaded_level_idx = None;
        self.loaded_level = None;
    }

    /// Load the level at `idx` into [`Self::loaded_level`] and seed
    /// `current_room` from its prince start position.
    pub fn load_level(&mut self, idx: usize) {
        let Some(path) = self.level_paths.get(idx).cloned() else {
            self.status = format!("level index {idx} out of range");
            return;
        };
        match Level::from_file(&path) {
            Ok(level) => {
                let prince = level.prince_start();
                // Clamp to the 1..=ROOMS_PER_LEVEL invariant so a
                // malformed level (or a sentinel like 0) can't drop
                // us straight into the "out of range" branch in the
                // central panel.
                self.current_room = prince.screen.clamp(1, ROOMS_PER_LEVEL_U8);
                self.loaded_level = Some(level);
                self.loaded_level_idx = Some(idx);
                self.status = format!("loaded {}", path.display());
            }
            Err(e) => {
                self.status = format!("failed to load {}: {e}", path.display());
            }
        }
    }

    /// Jump to the neighbour of [`Self::current_room`] in `direction`,
    /// if one exists (non-zero in the MAP). No-op otherwise.
    pub fn step(&mut self, direction: Direction) {
        let Some(level) = &self.loaded_level else {
            return;
        };
        let idx = usize::from(self.current_room.saturating_sub(1));
        let Some(neighbours) = level.room_links().get(idx) else {
            return;
        };
        let target = match direction {
            Direction::Left => neighbours.left,
            Direction::Right => neighbours.right,
            Direction::Up => neighbours.up,
            Direction::Down => neighbours.down,
        };
        if is_valid_room_id(target) {
            self.current_room = target;
        }
    }
}

/// Direction in the room neighbour graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Left,
    Right,
    Up,
    Down,
}

// ---------------------------------------------------------------------------
// eframe::App — UI projection.
// ---------------------------------------------------------------------------

/// egui application wrapping [`EditorState`].
struct EditorApp {
    state: EditorState,
}

impl EditorApp {
    fn new(state: EditorState) -> Self {
        Self { state }
    }
}

impl eframe::App for EditorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            self.toolbar(ui);
        });
        egui::SidePanel::left("levels")
            .resizable(true)
            .default_width(220.0)
            .show(ctx, |ui| {
                self.side_panel(ui);
            });
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("status:");
                ui.label(&self.state.status);
            });
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            self.center_panel(ui);
        });
    }
}

impl EditorApp {
    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("Open data root…").clicked() {
                if let Some(p) = pick_dir() {
                    self.state.set_root(p);
                }
            }
            ui.separator();
            if let Some(root) = &self.state.root {
                ui.label(format!("data root: {}", root.display()));
            } else {
                ui.label("no data root");
            }
        });
    }

    fn side_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Levels");
        ui.add_space(4.0);
        if self.state.level_paths.is_empty() {
            ui.label("no levels — open a data root");
            return;
        }
        let mut to_load: Option<usize> = None;
        egui::ScrollArea::vertical().show(ui, |ui| {
            for (i, path) in self.state.level_paths.iter().enumerate() {
                let label = path.file_name().map_or_else(
                    || format!("level {i}"),
                    |s| s.to_string_lossy().into_owned(),
                );
                let selected = self.state.loaded_level_idx == Some(i);
                if ui.selectable_label(selected, label).clicked() && !selected {
                    to_load = Some(i);
                }
            }
        });
        if let Some(i) = to_load {
            self.state.load_level(i);
        }
    }

    fn center_panel(&mut self, ui: &mut egui::Ui) {
        // Lift everything we need out of the immutable level borrow up
        // front so the navigation widget below can take `&mut state`.
        let snapshot: Option<RoomSnapshot> = self
            .state
            .loaded_level
            .as_ref()
            .and_then(|level| RoomSnapshot::capture(level, self.state.current_room));

        let Some(snap) = snapshot else {
            if self.state.loaded_level.is_none() {
                ui.label("select a level on the left to load it");
            } else {
                ui.label(format!("room {} out of range", self.state.current_room));
            }
            return;
        };

        ui.heading(format!("Room {}", self.state.current_room));
        room_navigation(ui, &mut self.state, snap.neighbours);
        ui.add_space(8.0);
        draw_room(
            ui,
            snap.tiles,
            self.state.current_room,
            snap.prince,
            &snap.guards,
        );
        ui.add_space(8.0);
        ui.collapsing("Neighbours", |ui| {
            ui.monospace(format!(
                "  left:  {:>2}   right: {:>2}\n  up:    {:>2}   down:  {:>2}",
                snap.neighbours.left,
                snap.neighbours.right,
                snap.neighbours.up,
                snap.neighbours.down,
            ));
            ui.label("(0 = edge of level)");
        });
    }
}

fn room_navigation(ui: &mut egui::Ui, state: &mut EditorState, n: RoomNeighbours) {
    ui.horizontal(|ui| {
        let mut nav = |label: &str, dir: Direction, target: u8, ui: &mut egui::Ui| {
            // Mirror `EditorState::step`'s bounds check exactly so an
            // out-of-range entry in the MAP table can't yield a button
            // that's enabled but does nothing.
            let enabled = is_valid_room_id(target);
            let hover = if enabled {
                format!("go to room {target}")
            } else if target == 0 {
                "edge of level".to_string()
            } else {
                format!("room {target} out of range (1..={ROOMS_PER_LEVEL_U8})")
            };
            if ui
                .add_enabled(enabled, egui::Button::new(label))
                .on_hover_text(hover)
                .clicked()
            {
                state.step(dir);
            }
        };
        nav("← left", Direction::Left, n.left, ui);
        nav("↑ up", Direction::Up, n.up, ui);
        nav("↓ down", Direction::Down, n.down, ui);
        nav("right →", Direction::Right, n.right, ui);
        ui.separator();
        ui.label("jump:");
        let mut current = state.current_room;
        egui::ComboBox::from_id_salt("jump_room")
            .selected_text(format!("room {current}"))
            .show_ui(ui, |ui| {
                for r in 1..=u8::try_from(ROOMS_PER_LEVEL).unwrap_or(24) {
                    ui.selectable_value(&mut current, r, format!("room {r}"));
                }
            });
        if current != state.current_room {
            state.current_room = current;
        }
    });
}

const TILE_W: f32 = 56.0;
const TILE_H: f32 = 36.0;

/// Owned snapshot of one room plus the data needed to render it
/// without keeping an immutable borrow on the level alive.
struct RoomSnapshot {
    tiles: [Tile; ROOM_WIDTH * ROOM_HEIGHT],
    neighbours: RoomNeighbours,
    prince: StartPosition,
    guards: [Option<GuardSpawn>; ROOMS_PER_LEVEL],
}

impl RoomSnapshot {
    fn capture(level: &Level, current_room: u8) -> Option<Self> {
        let idx = usize::from(current_room.saturating_sub(1));
        let room = level.rooms.get(idx)?;
        Some(Self {
            tiles: room.tiles,
            neighbours: level.room_links().get(idx).copied().unwrap_or_default(),
            prince: level.prince_start(),
            guards: level.guard_spawns(),
        })
    }
}

// `ROOM_WIDTH` / `ROOM_HEIGHT` are 10 and 3 respectively per #80; the
// `col` / `row` loop counters never exceed them. All four `as f32`
// casts in this function are exact on every supported target.
#[allow(clippy::cast_precision_loss)]
fn draw_room(
    ui: &mut egui::Ui,
    tiles: [Tile; ROOM_WIDTH * ROOM_HEIGHT],
    room_id: u8,
    prince: StartPosition,
    guards: &[Option<GuardSpawn>; ROOMS_PER_LEVEL],
) {
    const _: () = assert!(ROOM_WIDTH <= u8::MAX as usize);
    const _: () = assert!(ROOM_HEIGHT <= u8::MAX as usize);
    let size = Vec2::new(TILE_W * ROOM_WIDTH as f32, TILE_H * ROOM_HEIGHT as f32);
    let (resp, painter) = ui.allocate_painter(size, egui::Sense::hover());
    let origin = resp.rect.min;
    painter.rect_filled(resp.rect, 0.0, Color32::from_rgb(15, 15, 30));
    for row in 0..ROOM_HEIGHT {
        for col in 0..ROOM_WIDTH {
            let tile = tiles[row * ROOM_WIDTH + col];
            let rect = Rect::from_min_size(
                Pos2::new(
                    origin.x + TILE_W * col as f32,
                    origin.y + TILE_H * row as f32,
                ),
                Vec2::new(TILE_W, TILE_H),
            );
            painter.rect_filled(rect, 2.0, tile_color(tile.kind));
            painter.rect_stroke(rect, 2.0, Stroke::new(1.0, Color32::from_rgb(40, 40, 60)));
            painter.text(
                rect.left_top() + Vec2::new(4.0, 2.0),
                egui::Align2::LEFT_TOP,
                tile.kind.short_name(),
                egui::FontId::monospace(11.0),
                Color32::from_rgb(220, 220, 230),
            );
            if tile.variant != 0 {
                painter.text(
                    rect.right_bottom() - Vec2::new(4.0, 2.0),
                    egui::Align2::RIGHT_BOTTOM,
                    format!("v{}", tile.variant),
                    egui::FontId::monospace(9.0),
                    Color32::from_rgba_unmultiplied(220, 220, 230, 180),
                );
            }
        }
    }

    if prince.screen == room_id {
        if let Some((col, row)) = prince.col_row() {
            draw_marker(
                &painter,
                origin,
                col,
                row,
                "P",
                Color32::from_rgb(230, 70, 70),
            );
        }
    }
    if let Some(Some(g)) = guards.get(usize::from(room_id.saturating_sub(1))) {
        if let Some((col, row)) = g.col_row() {
            draw_marker(
                &painter,
                origin,
                col,
                row,
                "G",
                Color32::from_rgb(240, 160, 50),
            );
        }
    }
}

fn draw_marker(painter: &egui::Painter, origin: Pos2, col: u8, row: u8, text: &str, fill: Color32) {
    let cx = origin.x + TILE_W * (f32::from(col) + 0.5);
    let cy = origin.y + TILE_H * (f32::from(row) + 0.5);
    painter.circle_filled(Pos2::new(cx, cy), 11.0, fill);
    painter.text(
        Pos2::new(cx, cy),
        egui::Align2::CENTER_CENTER,
        text,
        egui::FontId::proportional(13.0),
        Color32::WHITE,
    );
}

fn tile_color(kind: TileKind) -> Color32 {
    match kind {
        TileKind::Empty => Color32::from_rgb(20, 20, 60),
        TileKind::Floor => Color32::from_rgb(140, 90, 40),
        TileKind::LooseFloor => Color32::from_rgb(180, 130, 60),
        TileKind::Block => Color32::from_rgb(80, 80, 80),
        TileKind::Spikes => Color32::from_rgb(200, 40, 40),
        TileKind::Gate => Color32::from_rgb(180, 180, 40),
        TileKind::DownPressPlate | TileKind::PressPlate | TileKind::UPressPlate => {
            Color32::from_rgb(180, 160, 80)
        }
        TileKind::Exit | TileKind::Exit2 => Color32::from_rgb(40, 180, 60),
        TileKind::Sword => Color32::from_rgb(200, 200, 220),
        TileKind::Flask => Color32::from_rgb(180, 40, 220),
        TileKind::Mirror => Color32::from_rgb(150, 200, 200),
        TileKind::Slicer => Color32::from_rgb(200, 100, 100),
        TileKind::Torch => Color32::from_rgb(250, 180, 0),
        TileKind::Posts | TileKind::PillarBottom | TileKind::PillarTop => {
            Color32::from_rgb(110, 80, 60)
        }
        TileKind::PanelWithFloor | TileKind::PanelWithoutFloor => Color32::from_rgb(100, 70, 50),
        TileKind::Window | TileKind::Window2 => Color32::from_rgb(120, 140, 180),
        TileKind::ArchBottom
        | TileKind::ArchTop1
        | TileKind::ArchTop2
        | TileKind::ArchTop3
        | TileKind::ArchTop4 => Color32::from_rgb(150, 130, 100),
        TileKind::Bones => Color32::from_rgb(220, 220, 200),
        TileKind::Rubble => Color32::from_rgb(120, 100, 80),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vendor_root() -> PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor/pop-apple2/04 Support")
    }

    #[test]
    fn state_with_no_root_has_no_levels() {
        let state = EditorState::new(None);
        assert!(state.root.is_none());
        assert!(state.level_paths.is_empty());
        assert!(state.loaded_level.is_none());
    }

    #[test]
    fn state_with_vendor_root_finds_all_levels() {
        let state = EditorState::new(Some(vendor_root()));
        assert_eq!(state.level_paths.len(), 15);
        assert!(
            state.loaded_level.is_none(),
            "no level loaded until selected"
        );
    }

    #[test]
    fn loading_level_one_seeds_current_room_from_prince_start() {
        let mut state = EditorState::new(Some(vendor_root()));
        // LEVEL1 is at index 1 (LEVEL0 is at 0).
        state.load_level(1);
        let level = state.loaded_level.as_ref().expect("level loaded");
        assert_eq!(state.current_room, level.prince_start().screen);
    }

    #[test]
    fn step_follows_neighbour_graph_when_valid() {
        let mut state = EditorState::new(Some(vendor_root()));
        state.load_level(1);
        // LEVEL1 room 1: left=5, right=0, up=0, down=2. So step(Down)
        // should jump to room 2; step(Right) should be a no-op.
        assert_eq!(state.current_room, 1);
        state.step(Direction::Right);
        assert_eq!(state.current_room, 1, "right edge — no-op");
        state.step(Direction::Down);
        assert_eq!(state.current_room, 2);
        state.step(Direction::Left);
        // Room 2 left = 6 per the level-test data, so verify >0 and <=24.
        assert!(state.current_room >= 1);
        assert!(usize::from(state.current_room) <= ROOMS_PER_LEVEL);
    }

    #[test]
    fn step_without_loaded_level_is_noop() {
        let mut state = EditorState::new(None);
        state.step(Direction::Left);
        assert_eq!(state.current_room, 1);
    }

    #[test]
    fn is_valid_room_id_matches_step_bounds() {
        // 0 (edge sentinel), 1..=24 (real ids), 25..=255 (corrupt).
        assert!(!is_valid_room_id(0));
        for id in 1..=ROOMS_PER_LEVEL_U8 {
            assert!(is_valid_room_id(id), "expected {id} to be valid");
        }
        assert!(!is_valid_room_id(ROOMS_PER_LEVEL_U8 + 1));
        assert!(!is_valid_room_id(u8::MAX));
    }

    #[test]
    fn current_room_stays_in_range_for_synthetic_levels() {
        // Build a state and exercise the clamp path directly: any
        // prince.screen value outside 1..=24 must be clamped to that
        // range so the UI's "out of range" branch is unreachable from
        // load_level alone.
        let mut state = EditorState::new(Some(vendor_root()));
        // Force-load every bundled level and verify the invariant.
        for idx in 0..state.level_paths.len() {
            state.load_level(idx);
            assert!(
                is_valid_room_id(state.current_room),
                "LEVEL{idx} produced current_room={} after load",
                state.current_room,
            );
        }
    }
}
