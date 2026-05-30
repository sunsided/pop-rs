//! `pop editor` — egui level browser.
//!
//! Phase 1 capstone, second cut: open a POP data root, pick a level,
//! and see **every room laid out spatially** in a pannable / zoomable
//! canvas, with Prince + guard spawn markers and a togglable
//! tile-name overlay.
//!
//! # Spatial layout
//!
//! POP's room neighbour graph (`MAP` from #88) doesn't carry explicit
//! room coordinates — neighbouring rooms are just linked by direction.
//! [`RoomLayout::compute`] walks the graph breadth-first from a seed
//! room (the prince start by default) and places each visited
//! neighbour at the appropriate `(±ROOM_WIDTH, 0)` /
//! `(0, ±ROOM_HEIGHT)` offset. Unreachable rooms get a [`None`] slot —
//! POP levels sometimes ship dead-code rooms that aren't connected to
//! the playable subgraph, and we shouldn't try to invent positions for
//! them.
//!
//! # What's here (vs. the original #90 scope)
//!
//! | Tab           | Status                                                    |
//! |---------------|-----------------------------------------------------------|
//! | Disk image    | Deferred — no disk parser (#84).                          |
//! | Sprite        | Deferred — `pop draz sprites` covers it from the CLI.     |
//! | Level         | **Shipped** — all-rooms pannable canvas, this PR.         |
//! | Animation     | Deferred — #89 (`SEQTABLE.S` / `SEQDATA.S` / `FRAMEDEF.S`).|
//!
//! Tile rendering still uses solid colours per [`TileKind`] with the
//! `short_name()` overlaid (optional). Real `IMG.BGTAB.*` sprite
//! compositing is the next iteration and slots cleanly into
//! [`draw_tile`] — swap the `painter.rect_filled` call for a textured
//! quad.
//!
//! # State / view split
//!
//! [`EditorState`] (data) holds no eframe types; [`EditorApp`] (view)
//! holds the eframe-specific pan / zoom state. The WASM port stays
//! cheap.

#![cfg(feature = "editor")]
// Canvas math casts tile coordinates (bounded i32 / usize) into f32
// pixel-space; these are all exact for any realistic level layout
// (max ~240×72 tiles).
#![allow(clippy::cast_precision_loss)]

use std::collections::VecDeque;
use std::path::PathBuf;

use clap::Args as ClapArgs;
use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2};
use pop_assets::{
    discovery,
    level::{Level, Tile, TileKind, ROOMS_PER_LEVEL, ROOM_HEIGHT, ROOM_WIDTH},
};

const _: () = assert!(ROOMS_PER_LEVEL <= u8::MAX as usize);
#[allow(clippy::cast_possible_truncation)]
const ROOMS_PER_LEVEL_U8: u8 = ROOMS_PER_LEVEL as u8;

/// True when `id` is a valid 1-based room id (`1..=ROOMS_PER_LEVEL`).
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
            .with_inner_size([1280.0, 800.0]),
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

/// Per-room spatial position, computed by [`RoomLayout::compute`].
/// Position is in **tile coordinates** (room origin = top-left tile of
/// the room). Rooms unreachable from the BFS seed are `None`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoomLayout {
    /// Per-room tile position, indexed `0..24` = POP room `1..24`.
    pub positions: [Option<(i32, i32)>; ROOMS_PER_LEVEL],
    /// Bounding box of all placed rooms, in tile coordinates.
    /// `((min_x, min_y), (max_x_exclusive, max_y_exclusive))`. `None`
    /// when no room was placed.
    pub bounds: Option<((i32, i32), (i32, i32))>,
}

impl RoomLayout {
    /// Compute the layout via BFS from `seed_room` (1-based).
    /// Unreachable rooms stay `None`.
    #[must_use]
    pub fn compute(level: &Level, seed_room: u8) -> Self {
        let mut positions: [Option<(i32, i32)>; ROOMS_PER_LEVEL] = [None; ROOMS_PER_LEVEL];
        if !is_valid_room_id(seed_room) {
            return Self {
                positions,
                bounds: None,
            };
        }

        let seed_idx = usize::from(seed_room - 1);
        positions[seed_idx] = Some((0, 0));
        let mut queue: VecDeque<u8> = VecDeque::new();
        queue.push_back(seed_room);

        let room_w = i32::try_from(ROOM_WIDTH).unwrap_or(10);
        let room_h = i32::try_from(ROOM_HEIGHT).unwrap_or(3);

        while let Some(room_id) = queue.pop_front() {
            let idx = usize::from(room_id - 1);
            let Some(pos) = positions[idx] else {
                continue;
            };
            let Some(neighbours) = level.room_links().get(idx) else {
                continue;
            };
            for (delta, neighbour_id) in [
                ((room_w, 0), neighbours.right),
                ((-room_w, 0), neighbours.left),
                ((0, room_h), neighbours.down),
                ((0, -room_h), neighbours.up),
            ] {
                if !is_valid_room_id(neighbour_id) {
                    continue;
                }
                let n_idx = usize::from(neighbour_id - 1);
                if positions[n_idx].is_some() {
                    // Already placed (either via a shorter BFS path or
                    // a cycle). POP levels are usually grid-consistent
                    // but we don't enforce it — first BFS hit wins.
                    continue;
                }
                positions[n_idx] = Some((pos.0 + delta.0, pos.1 + delta.1));
                queue.push_back(neighbour_id);
            }
        }

        let bounds = compute_bounds(&positions, room_w, room_h);
        Self { positions, bounds }
    }
}

fn compute_bounds(
    positions: &[Option<(i32, i32)>; ROOMS_PER_LEVEL],
    room_w: i32,
    room_h: i32,
) -> Option<((i32, i32), (i32, i32))> {
    let mut iter = positions.iter().filter_map(|p| p.as_ref());
    let first = *iter.next()?;
    let mut min = first;
    let mut max = (first.0 + room_w, first.1 + room_h);
    for &(x, y) in iter {
        if x < min.0 {
            min.0 = x;
        }
        if y < min.1 {
            min.1 = y;
        }
        if x + room_w > max.0 {
            max.0 = x + room_w;
        }
        if y + room_h > max.1 {
            max.1 = y + room_h;
        }
    }
    Some((min, max))
}

/// Editor data model. Holds the discovered POP data root and the
/// currently-loaded level, layout, and inspection state.
#[derive(Debug)]
pub struct EditorState {
    /// Active POP data root, if any.
    pub root: Option<PathBuf>,
    /// Per-`LEVEL{N}` paths under `root`'s `Levels/`. Empty when no
    /// root is set.
    pub level_paths: Vec<PathBuf>,
    /// Index into [`Self::level_paths`] of the currently loaded level.
    pub loaded_level_idx: Option<usize>,
    /// The parsed level for [`Self::loaded_level_idx`].
    pub loaded_level: Option<Level>,
    /// Spatial layout of `loaded_level`'s rooms. Recomputed on every
    /// level load.
    pub layout: Option<RoomLayout>,
    /// Last operation outcome, surfaced in the status bar.
    pub status: String,
}

impl EditorState {
    /// Construct a state, eagerly scanning `Levels/` if `root` is set.
    #[must_use]
    pub fn new(root: Option<PathBuf>) -> Self {
        let mut state = Self {
            root: None,
            level_paths: Vec::new(),
            loaded_level_idx: None,
            loaded_level: None,
            layout: None,
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
        self.layout = None;
    }

    /// Load the level at `idx` and compute its spatial layout.
    pub fn load_level(&mut self, idx: usize) {
        let Some(path) = self.level_paths.get(idx).cloned() else {
            self.status = format!("level index {idx} out of range");
            return;
        };
        match Level::from_file(&path) {
            Ok(level) => {
                let seed = level.prince_start().screen.clamp(1, ROOMS_PER_LEVEL_U8);
                let layout = RoomLayout::compute(&level, seed);
                let placed = layout.positions.iter().filter(|p| p.is_some()).count();
                self.status = format!(
                    "loaded {} ({}/{} rooms reachable from prince start)",
                    path.display(),
                    placed,
                    ROOMS_PER_LEVEL,
                );
                self.loaded_level = Some(level);
                self.loaded_level_idx = Some(idx);
                self.layout = Some(layout);
            }
            Err(e) => {
                self.status = format!("failed to load {}: {e}", path.display());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// eframe::App — UI projection.
// ---------------------------------------------------------------------------

/// egui application wrapping [`EditorState`] plus the
/// canvas-specific pan / zoom state.
struct EditorApp {
    state: EditorState,
    /// Canvas pan in screen pixels — the origin of the world (tile
    /// `(0, 0)`) sits at `panel_origin + pan` on screen.
    pan: Vec2,
    /// Canvas zoom in pixels-per-tile-pixel; effectively each tile is
    /// `(BASE_TILE_W * zoom, BASE_TILE_H * zoom)` pixels onscreen.
    zoom: f32,
    /// Toggle for the per-tile `short_name()` label overlay.
    show_labels: bool,
    /// Toggle for the per-room ID badge overlay.
    show_room_ids: bool,
    /// `Some((room, col, row))` when the mouse is hovering over a
    /// tile; surfaces in the status bar.
    hover: Option<(u8, u8, u8)>,
    /// Set on level load so the next frame can fit-to-view.
    pending_fit: bool,
}

/// Pixels per tile at zoom == 1.0.
const BASE_TILE_W: f32 = 26.0;
const BASE_TILE_H: f32 = 18.0;
/// Min / max zoom — guards against accidental zoom-to-infinity.
const MIN_ZOOM: f32 = 0.3;
const MAX_ZOOM: f32 = 8.0;

impl EditorApp {
    fn new(state: EditorState) -> Self {
        Self {
            state,
            pan: Vec2::ZERO,
            zoom: 1.0,
            show_labels: true,
            show_room_ids: true,
            hover: None,
            pending_fit: true,
        }
    }

    fn tile_w(&self) -> f32 {
        BASE_TILE_W * self.zoom
    }

    fn tile_h(&self) -> f32 {
        BASE_TILE_H * self.zoom
    }

    fn handle_pan_zoom(&mut self, ui: &mut egui::Ui, resp: &egui::Response, panel_rect: Rect) {
        // Pan: primary or middle drag.
        if resp.dragged_by(egui::PointerButton::Primary)
            || resp.dragged_by(egui::PointerButton::Middle)
        {
            self.pan += resp.drag_delta();
        }
        // Zoom: scroll wheel, centred on cursor.
        if let Some(cursor) = resp.hover_pos() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let old_zoom = self.zoom;
                let factor = (scroll * 0.005).exp2();
                self.zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
                if (self.zoom - old_zoom).abs() > f32::EPSILON {
                    // Keep the world point under the cursor stationary.
                    let cursor_local = cursor - panel_rect.min.to_vec2();
                    let world_under_cursor = (cursor_local.to_vec2() - self.pan) / old_zoom;
                    self.pan = cursor_local.to_vec2() - world_under_cursor * self.zoom;
                }
            }
        }
        // Arrow-key pan.
        ui.input(|i| {
            let pan_step = 24.0;
            if i.key_down(egui::Key::ArrowLeft) {
                self.pan.x += pan_step;
            }
            if i.key_down(egui::Key::ArrowRight) {
                self.pan.x -= pan_step;
            }
            if i.key_down(egui::Key::ArrowUp) {
                self.pan.y += pan_step;
            }
            if i.key_down(egui::Key::ArrowDown) {
                self.pan.y -= pan_step;
            }
        });
    }

    /// Reset pan + zoom so the entire layout fits inside `viewport`
    /// with a small margin.
    fn fit_to_view(&mut self, viewport: Rect) {
        let Some(layout) = &self.state.layout else {
            return;
        };
        let Some((bb_min, bb_max)) = layout.bounds else {
            return;
        };
        let world_w = (bb_max.0 - bb_min.0).max(1) as f32;
        let world_height = (bb_max.1 - bb_min.1).max(1) as f32;
        let margin = 24.0;
        let avail = Vec2::new(
            (viewport.width() - margin * 2.0).max(50.0),
            (viewport.height() - margin * 2.0).max(50.0),
        );
        let fit_zoom_w = avail.x / (world_w * BASE_TILE_W);
        let fit_zoom_h = avail.y / (world_height * BASE_TILE_H);
        self.zoom = fit_zoom_w.min(fit_zoom_h).clamp(MIN_ZOOM, MAX_ZOOM);
        // Centre the bounding box in the viewport.
        let center_world = Vec2::new(
            (bb_min.0 as f32 + bb_max.0 as f32) * 0.5 * BASE_TILE_W * self.zoom,
            (bb_min.1 as f32 + bb_max.1 as f32) * 0.5 * BASE_TILE_H * self.zoom,
        );
        let viewport_center =
            (viewport.min.to_vec2() + viewport.max.to_vec2()) * 0.5 - viewport.min.to_vec2();
        self.pan = viewport_center - center_world;
    }
}

impl eframe::App for EditorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| self.toolbar(ui));
        egui::SidePanel::left("levels")
            .resizable(true)
            .default_width(220.0)
            .show(ctx, |ui| self.side_panel(ui));
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| self.status_bar(ui));
        egui::CentralPanel::default().show(ctx, |ui| self.canvas(ui));
    }
}

impl EditorApp {
    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui.button("Open data root…").clicked() {
                if let Some(p) = pick_dir() {
                    self.state.set_root(p);
                }
            }
            ui.separator();
            ui.checkbox(&mut self.show_labels, "tile labels");
            ui.checkbox(&mut self.show_room_ids, "room IDs");
            ui.separator();
            if ui.button("Fit view").clicked() {
                self.pending_fit = true;
            }
            ui.separator();
            ui.label(format!("zoom: {:.2}×", self.zoom));
            ui.separator();
            if let Some(root) = &self.state.root {
                ui.label(format!("root: {}", root.display()));
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
            self.pending_fit = true;
        }
    }

    fn status_bar(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(format!("status: {}", self.state.status));
            ui.separator();
            if let Some((room, col, row)) = self.hover {
                ui.label(format!("hover: room {room} tile ({col}, {row})"));
            } else {
                ui.label("hover: —");
            }
        });
    }

    fn canvas(&mut self, ui: &mut egui::Ui) {
        // Lift everything we need out of the immutable level / layout
        // borrows up front so the rest of the function can mutate
        // self (pan, zoom, hover, …).
        let Some(snapshot) = LevelSnapshot::capture(&self.state) else {
            ui.centered_and_justified(|ui| {
                ui.label("select a level on the left to load it");
            });
            return;
        };

        let (resp, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let panel_rect = resp.rect;

        if self.pending_fit {
            self.fit_to_view(panel_rect);
            self.pending_fit = false;
        }
        self.handle_pan_zoom(ui, &resp, panel_rect);

        // Background.
        painter.rect_filled(panel_rect, 0.0, Color32::from_rgb(10, 10, 18));

        let cursor_local = resp
            .hover_pos()
            .map(|p| p.to_vec2() - panel_rect.min.to_vec2());
        self.hover = self.draw_layout(&painter, panel_rect, &snapshot, cursor_local);
    }

    fn draw_layout(
        &self,
        painter: &egui::Painter,
        panel_rect: Rect,
        snapshot: &LevelSnapshot,
        cursor_local: Option<Vec2>,
    ) -> Option<(u8, u8, u8)> {
        let tile_w = self.tile_w();
        let tile_h = self.tile_h();
        let prince = snapshot.prince;
        let mut hover = None;

        for (room_idx, slot) in snapshot.layout.positions.iter().enumerate() {
            let Some((rx, ry)) = *slot else { continue };
            let room_id = u8::try_from(room_idx + 1).unwrap_or(0);
            let tiles = snapshot.rooms[room_idx];
            draw_room(
                painter,
                panel_rect.min,
                self.pan,
                tile_w,
                tile_h,
                rx,
                ry,
                tiles,
                self.show_labels,
            );
            if self.show_room_ids {
                let badge_pos = panel_rect.min
                    + self.pan
                    + Vec2::new(rx as f32 * tile_w + 4.0, ry as f32 * tile_h + 2.0);
                painter.text(
                    badge_pos,
                    egui::Align2::LEFT_TOP,
                    format!("R{room_id}"),
                    egui::FontId::monospace((12.0 * self.zoom).clamp(8.0, 16.0)),
                    Color32::from_rgb(255, 230, 120),
                );
            }
            if prince.screen == room_id {
                if let Some((col, row)) = prince.col_row() {
                    draw_marker(
                        painter,
                        panel_rect.min,
                        self.pan,
                        tile_w,
                        tile_h,
                        rx + i32::from(col),
                        ry + i32::from(row),
                        "P",
                        Color32::from_rgb(230, 70, 70),
                    );
                }
            }
            if let Some((col, row)) = snapshot.guard_positions[room_idx] {
                draw_marker(
                    painter,
                    panel_rect.min,
                    self.pan,
                    tile_w,
                    tile_h,
                    rx + i32::from(col),
                    ry + i32::from(row),
                    "G",
                    Color32::from_rgb(240, 160, 50),
                );
            }

            if let Some(cl) = cursor_local {
                let room_x = self.pan.x + rx as f32 * tile_w;
                let room_y = self.pan.y + ry as f32 * tile_h;
                let dx = cl.x - room_x;
                let dy = cl.y - room_y;
                if dx >= 0.0
                    && dy >= 0.0
                    && dx < tile_w * ROOM_WIDTH as f32
                    && dy < tile_h * ROOM_HEIGHT as f32
                {
                    // Bounds-checked above; results lie in
                    // 0..ROOM_WIDTH / 0..ROOM_HEIGHT.
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let col = (dx / tile_w) as u8;
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let row = (dy / tile_h) as u8;
                    hover = Some((room_id, col, row));
                }
            }
        }

        // Outline placed rooms.
        for slot in &snapshot.layout.positions {
            let Some((rx, ry)) = *slot else { continue };
            let room_rect = Rect::from_min_size(
                panel_rect.min + self.pan + Vec2::new(rx as f32 * tile_w, ry as f32 * tile_h),
                Vec2::new(tile_w * ROOM_WIDTH as f32, tile_h * ROOM_HEIGHT as f32),
            );
            painter.rect_stroke(
                room_rect,
                0.0,
                Stroke::new(1.5, Color32::from_rgb(70, 70, 100)),
            );
        }
        hover
    }
}

/// Owned snapshot of the loaded level + layout, captured so the
/// canvas function can keep mutating `self` (pan, zoom, hover, …)
/// after the read.
struct LevelSnapshot {
    layout: RoomLayout,
    rooms: [[Tile; ROOM_WIDTH * ROOM_HEIGHT]; ROOMS_PER_LEVEL],
    prince: pop_assets::level::StartPosition,
    /// Per-room guard `(col, row)`, or `None` if no guard / out of range.
    guard_positions: [Option<(u8, u8)>; ROOMS_PER_LEVEL],
}

impl LevelSnapshot {
    fn capture(state: &EditorState) -> Option<Self> {
        let level = state.loaded_level.as_ref()?;
        let layout = state.layout.as_ref()?.clone();
        let mut rooms = [[Tile::default(); ROOM_WIDTH * ROOM_HEIGHT]; ROOMS_PER_LEVEL];
        for (i, room) in level.rooms.iter().enumerate() {
            rooms[i] = room.tiles;
        }
        let prince = level.prince_start();
        let mut guard_positions: [Option<(u8, u8)>; ROOMS_PER_LEVEL] = [None; ROOMS_PER_LEVEL];
        for (i, g) in level.guard_spawns().iter().enumerate() {
            if let Some(g) = g {
                guard_positions[i] = g.col_row();
            }
        }
        Some(Self {
            layout,
            rooms,
            prince,
            guard_positions,
        })
    }
}

#[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
fn draw_room(
    painter: &egui::Painter,
    panel_origin: Pos2,
    pan: Vec2,
    tile_w: f32,
    tile_h: f32,
    room_x_tiles: i32,
    room_top_tiles: i32,
    tiles: [Tile; ROOM_WIDTH * ROOM_HEIGHT],
    show_labels: bool,
) {
    const _: () = assert!(ROOM_WIDTH <= u8::MAX as usize);
    const _: () = assert!(ROOM_HEIGHT <= u8::MAX as usize);
    for row in 0..ROOM_HEIGHT {
        for col in 0..ROOM_WIDTH {
            let tile = tiles[row * ROOM_WIDTH + col];
            let x = panel_origin.x + pan.x + (room_x_tiles as f32 + col as f32) * tile_w;
            let y = panel_origin.y + pan.y + (room_top_tiles as f32 + row as f32) * tile_h;
            let rect = Rect::from_min_size(Pos2::new(x, y), Vec2::new(tile_w, tile_h));
            painter.rect_filled(rect, 1.0, tile_color(tile.kind));
            if show_labels && tile_w >= 18.0 {
                let font_px = (tile_h * 0.4).clamp(8.0, 14.0);
                painter.text(
                    rect.left_top() + Vec2::new(2.0, 1.0),
                    egui::Align2::LEFT_TOP,
                    tile.kind.short_name(),
                    egui::FontId::monospace(font_px),
                    Color32::from_rgb(220, 220, 230),
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
fn draw_marker(
    painter: &egui::Painter,
    panel_origin: Pos2,
    pan: Vec2,
    tile_w: f32,
    tile_h: f32,
    tile_x: i32,
    tile_y: i32,
    text: &str,
    fill: Color32,
) {
    let cx = panel_origin.x + pan.x + (tile_x as f32 + 0.5) * tile_w;
    let cy = panel_origin.y + pan.y + (tile_y as f32 + 0.5) * tile_h;
    let r = (tile_h * 0.35).clamp(6.0, 18.0);
    painter.circle_filled(Pos2::new(cx, cy), r, fill);
    painter.text(
        Pos2::new(cx, cy),
        egui::Align2::CENTER_CENTER,
        text,
        egui::FontId::proportional((r * 1.1).clamp(8.0, 18.0)),
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

    fn level_n(n: u8) -> Level {
        Level::from_file(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
                "../../vendor/pop-apple2/04 Support/Levels/LEVEL{n}"
            )),
        )
        .unwrap()
    }

    #[test]
    fn state_with_no_root_has_no_levels() {
        let state = EditorState::new(None);
        assert!(state.root.is_none());
        assert!(state.level_paths.is_empty());
        assert!(state.loaded_level.is_none());
        assert!(state.layout.is_none());
    }

    #[test]
    fn state_with_vendor_root_finds_all_levels() {
        let state = EditorState::new(Some(vendor_root()));
        assert_eq!(state.level_paths.len(), 15);
        assert!(state.layout.is_none(), "no layout until a level loads");
    }

    #[test]
    fn loading_level_one_computes_a_layout() {
        let mut state = EditorState::new(Some(vendor_root()));
        state.load_level(1);
        let layout = state.layout.as_ref().expect("layout computed");
        // LEVEL1's playable subgraph from prince start (room 1)
        // reaches many rooms — at minimum the prince room must be
        // placed at the origin.
        let prince_room = state.loaded_level.as_ref().unwrap().prince_start().screen;
        assert_eq!(layout.positions[usize::from(prince_room - 1)], Some((0, 0)));
        let reachable = layout.positions.iter().filter(|p| p.is_some()).count();
        assert!(
            reachable >= 10,
            "LEVEL1 should connect at least 10 rooms; got {reachable}"
        );
    }

    #[test]
    fn layout_places_right_neighbour_at_room_width_offset() {
        let level = level_n(1);
        let layout = RoomLayout::compute(&level, 1);
        // LEVEL1 room 1 right=0 (no right), down=2. Verify room 2
        // (down neighbour) is placed at (0, ROOM_HEIGHT).
        assert_eq!(layout.positions[0], Some((0, 0)));
        let room_h_i32 = i32::try_from(ROOM_HEIGHT).unwrap();
        assert_eq!(layout.positions[1], Some((0, room_h_i32)));
    }

    #[test]
    fn layout_bounds_cover_every_placed_room() {
        let level = level_n(1);
        let layout = RoomLayout::compute(&level, 1);
        let (min, max) = layout.bounds.expect("non-empty layout has bounds");
        let room_w = i32::try_from(ROOM_WIDTH).unwrap();
        let room_h = i32::try_from(ROOM_HEIGHT).unwrap();
        for slot in &layout.positions {
            if let Some((x, y)) = *slot {
                assert!(x >= min.0 && y >= min.1, "room ({x},{y}) below min {min:?}");
                assert!(
                    x + room_w <= max.0 && y + room_h <= max.1,
                    "room ({x},{y}) past max {max:?}",
                );
            }
        }
    }

    #[test]
    fn layout_with_invalid_seed_is_empty() {
        let level = level_n(1);
        let layout = RoomLayout::compute(&level, 0);
        assert!(layout.positions.iter().all(Option::is_none));
        assert!(layout.bounds.is_none());
        let layout = RoomLayout::compute(&level, 99);
        assert!(layout.positions.iter().all(Option::is_none));
    }

    #[test]
    fn every_bundled_level_layout_is_finite_and_in_bounds() {
        // Catches BFS bugs that could place a room at extreme
        // coordinates or never terminate.
        for n in 0u8..=14 {
            let level = level_n(n);
            let seed = level.prince_start().screen.clamp(1, ROOMS_PER_LEVEL_U8);
            let layout = RoomLayout::compute(&level, seed);
            if let Some(((min_x, min_y), (max_x, max_y))) = layout.bounds {
                // World extent shouldn't blow up to thousands of tiles —
                // at worst all 24 rooms in a strip → 24 * 10 = 240 wide
                // or 24 * 3 = 72 tall.
                assert!(
                    max_x - min_x <= 24 * i32::try_from(ROOM_WIDTH).unwrap(),
                    "LEVEL{n} layout width too large: {min_x}..{max_x}",
                );
                assert!(
                    max_y - min_y <= 24 * i32::try_from(ROOM_HEIGHT).unwrap(),
                    "LEVEL{n} layout height too large: {min_y}..{max_y}",
                );
            }
        }
    }

    #[test]
    fn is_valid_room_id_matches_bounds() {
        assert!(!is_valid_room_id(0));
        for id in 1..=ROOMS_PER_LEVEL_U8 {
            assert!(is_valid_room_id(id));
        }
        assert!(!is_valid_room_id(ROOMS_PER_LEVEL_U8 + 1));
        assert!(!is_valid_room_id(u8::MAX));
    }
}
