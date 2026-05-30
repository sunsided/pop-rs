//! Per-room scene compositor — turns one [`crate::level::Room`] into
//! a 280×192 RGBA frame using the biome's `IMG.BGTAB.*` sprites.
//!
//! Mirrors the Apple II background renderer in
//! `vendor/pop-apple2/01 POP Source/Source/FRAMEADV.S` (around the
//! `SURE` and `RedBlockSure` labels), but does the compositing in CPU
//! bytes — no $2000-page interleaving, no draw planes — and then
//! scan-converts via [`crate::hires::render_linear`] at the end.
//!
//! # What's modelled
//!
//! * Per-cell A/B/C/D/front pieces with the masks from
//!   [`crate::bgdata`].
//! * Left-neighbour and below-left-neighbour adjacency: when rendering
//!   cell (col, row) we pick up the left tile's B-section and the
//!   below-left tile's C-section, exactly as `RedBlockSure` does.
//! * Cross-room context: if the room being rendered has a left / below
//!   / above neighbour wired in the [`crate::level::Level`] graph, we
//!   read their edge tiles too so edges don't lose detail.
//! * The "ceiling row" D-pass that draws the bottom row of the room
//!   above (FRAMEADV.S:113-178).
//! * Biome-aware back-wall variants (`SPACE_B` / `FLOOR_B`) and
//!   conditional palace `B_STRIPE`.
//! * Multi-variant blocks / panels (`BLOCK_*` / `PANEL_*`) keyed off
//!   `Tile::variant`.
//!
//! # What's not modelled (yet)
//!
//! All animated / state-sensitive specials punted to a follow-up PR:
//!
//! * Loose floor in motion (`drawmd` / `drawlooseb`) — uses the at-rest
//!   floor look instead.
//! * Spikes / slicer animation frames (`drawma` / `drawspikea` /
//!   `drawslicera`) — uses the BGDATA `PIECE_A[k]` at-rest value.
//! * Gate bars at partial heights (`drawgateb` / `drawgatec`) — uses
//!   the closed-gate appearance.
//! * Depressed press-plate state — uses the up-state piece.
//! * Flask / sword movable A-pieces — uses the BGDATA piece.
//! * Half-piece climbup rendering — N/A for static scene browsing.
//!
//! Static rooms render correctly; dynamic objects look "frozen".

// Tile / pixel coordinates here are bounded (≤ 280×192 + small
// neighbour deltas); casts between i32 / usize / u8 are exact for the
// inputs this module ever sees.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use crate::bgdata::{
    Biome, PieceRef, BLOCK_B, BLOCK_BOT_ROW, BLOCK_C, BLOCK_D, BLOCK_FR, B_STRIPE,
    CELL_WIDTH_BYTES, D_HEIGHT, FLOOR_B, FLOOR_B_Y, FRONT_I, FRONT_X, FRONT_Y, MASK_A, MASK_B,
    PANEL_B, PANEL_B0_SENTINEL, PANEL_C, PANEL_C0_SENTINEL, PIECE_A, PIECE_A_Y, PIECE_B, PIECE_B_Y,
    PIECE_C, PIECE_D, ROOM_HEIGHT_PX, ROOM_WIDTH_BYTES, SPACE_B, SPACE_B_Y,
};
use crate::draz::image_table::{self, Image, ImageTable};
use crate::hires::{self, Frame, RenderMode};
use crate::level::{Level, Room, Tile, TileKind, ROOM_HEIGHT, ROOM_WIDTH};

/// The two `IMG.BGTAB.*` image tables for one biome.
///
/// Both are needed because piece-ID bit 7 selects between them
/// (`PieceRef::resolve`). For a given level: `table1` ↔
/// `IMG.BGTAB.{biome}1`, `table2` ↔ `IMG.BGTAB.{biome}2`.
#[derive(Clone, Debug)]
pub struct BiomeTables {
    /// The biome these tables belong to.
    pub biome: Biome,
    /// First image table (sprite IDs `1..=0x7E`).
    pub table1: ImageTable,
    /// Second image table (sprite IDs `0x81..=0xFE`).
    pub table2: ImageTable,
}

impl BiomeTables {
    /// Load both tables from `DRAZ/IP/IMG.BGTAB.{biome}{1,2}` under
    /// `data_root`. Errors bubble up unchanged from the
    /// [`crate::draz::image_table`] parser.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`image_table::ParseError`] if
    /// either table file is missing or malformed.
    pub fn load(
        data_root: &std::path::Path,
        biome: Biome,
    ) -> Result<Self, image_table::ParseError> {
        let (n1, n2) = biome.bgtab_filenames();
        let dir = data_root.join("DRAZ").join("IP");
        let table1 = ImageTable::from_file(dir.join(n1))?;
        let table2 = ImageTable::from_file(dir.join(n2))?;
        Ok(Self {
            biome,
            table1,
            table2,
        })
    }

    fn resolve(&self, piece_id: u8) -> Option<&Image> {
        match PieceRef::resolve(piece_id)? {
            PieceRef::Table1(i) => self.table1.images.get(usize::from(i)),
            PieceRef::Table2(i) => self.table2.images.get(usize::from(i)),
        }
    }
}

/// Render one room of a level to a 280×192 RGBA frame.
///
/// `room_id` is **1-based** (`1..=ROOMS_PER_LEVEL`) matching POP's
/// screen-number convention.
///
/// Returns `None` if `room_id` is out of range. Cross-room neighbour
/// data is looked up via `level.room_links()`; cells beyond the
/// rendered room's edges fall back to [`TileKind::Empty`] when no
/// neighbour exists.
#[must_use]
pub fn render_room(
    level: &Level,
    room_id: u8,
    bg: &BiomeTables,
    mode: RenderMode,
) -> Option<Frame> {
    let room_idx = usize::from(room_id).checked_sub(1)?;
    if room_idx >= level.rooms.len() {
        return None;
    }
    let ctx = RoomContext::from_level(level, room_idx);
    let canvas = Canvas::compose(&ctx, bg);
    canvas.into_frame(mode)
}

// ---------------------------------------------------------------------------
// Room neighbour context.
// ---------------------------------------------------------------------------

/// One room plus enough neighbour data to render its edges correctly.
struct RoomContext<'a> {
    room: &'a Room,
    /// Room visible immediately to the left, if any — its rightmost
    /// column provides PRECED for cells in column 0.
    left: Option<&'a Room>,
    /// Room visible immediately below — its top row provides BELOW for
    /// the bottom-row drawing pass.
    below: Option<&'a Room>,
    /// Room visible below-left — its top-right tile provides BELOW[0]
    /// for column 0 of the bottom row.
    below_left: Option<&'a Room>,
    /// Room visible above — its bottom row provides the "ceiling"
    /// D-pass (FRAMEADV.S:113-178).
    above: Option<&'a Room>,
}

impl<'a> RoomContext<'a> {
    fn from_level(level: &'a Level, room_idx: usize) -> Self {
        let room = &level.rooms[room_idx];
        let neighbours = level
            .room_links()
            .get(room_idx)
            .copied()
            .unwrap_or_default();
        let lookup = |id: u8| -> Option<&'a Room> {
            let i = usize::from(id).checked_sub(1)?;
            level.rooms.get(i)
        };
        let left = lookup(neighbours.left);
        let below = lookup(neighbours.down);
        let above = lookup(neighbours.up);
        let below_left = below.and_then(|_| {
            let below_idx = usize::from(neighbours.down).checked_sub(1)?;
            let below_neighbours = level.room_links().get(below_idx).copied()?;
            lookup(below_neighbours.left)
        });
        Self {
            room,
            left,
            below,
            below_left,
            above,
        }
    }

    /// Tile at `(col, row)` within the current room. Out-of-room
    /// indices return [`Tile::default()`] (Empty).
    fn tile(&self, col: i32, row: i32) -> Tile {
        if col >= 0
            && row >= 0
            && usize::try_from(col).is_ok_and(|c| c < ROOM_WIDTH)
            && usize::try_from(row).is_ok_and(|r| r < ROOM_HEIGHT)
        {
            self.room.tiles[(row as usize) * ROOM_WIDTH + (col as usize)]
        } else {
            Tile::default()
        }
    }

    /// Tile in the left neighbour at `(col, row)`, treating `col` as
    /// 0-based from the neighbour's left edge.
    fn left_tile(&self, col: usize, row: usize) -> Tile {
        self.left
            .map_or_else(Tile::default, |r| r.tiles[row * ROOM_WIDTH + col])
    }

    /// Top-row tile in the below-neighbour at `col`.
    fn below_top(&self, col: usize) -> Tile {
        self.below.map_or_else(Tile::default, |r| r.tiles[col])
    }

    /// Top-right tile in the below-left-neighbour (for BELOW[0] in the
    /// bottom row).
    fn below_left_top_right(&self) -> Tile {
        self.below_left
            .map_or_else(Tile::default, |r| r.tiles[ROOM_WIDTH - 1])
    }

    /// Bottom row of the above-neighbour, used for the "ceiling"
    /// D-pass at the top of the rendered screen.
    fn above_bottom_row(&self, col: usize) -> Tile {
        self.above.map_or_else(Tile::default, |r| {
            r.tiles[(ROOM_HEIGHT - 1) * ROOM_WIDTH + col]
        })
    }
}

// ---------------------------------------------------------------------------
// Hires canvas + blit primitives.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Opacity {
    /// `canvas[i] |= piece[i]` (additive).
    Or,
    /// `canvas[i] &= piece[i]` (mask-clear).
    And,
    /// `canvas[i] = piece[i]` (overwrite).
    Sta,
}

/// 40-byte × 192-row hires byte canvas. Y axis matches the screen
/// (`y=0` is the top scan-line).
struct Canvas {
    /// Linear hires bytes, indexed `y * ROOM_WIDTH_BYTES + x`.
    bytes: Box<[u8; (ROOM_WIDTH_BYTES as usize) * (ROOM_HEIGHT_PX as usize)]>,
}

impl Canvas {
    fn new() -> Self {
        Self {
            bytes: Box::new([0; (ROOM_WIDTH_BYTES as usize) * (ROOM_HEIGHT_PX as usize)]),
        }
    }

    /// Blit `sprite` at (`x_byte`, `y_bottom`) using `opacity`.
    /// `y_bottom` is the screen Y of the sprite's bottom row (matching
    /// POP's `YCO` convention); the sprite extends upward.
    fn blit(&mut self, sprite: &Image, x_byte: i32, y_bottom: i32, opacity: Opacity) {
        let w = usize::from(sprite.width_bytes);
        let h = i32::from(sprite.height);
        if w == 0 || h <= 0 {
            return;
        }
        for k in 0..h {
            // Sprite row k (0 = bottom) → screen y = y_bottom - k.
            let screen_y = y_bottom - k;
            if !(0..i32::from(ROOM_HEIGHT_PX)).contains(&screen_y) {
                continue;
            }
            let row_in_sprite = k as usize; // bottom-up source
            let src = &sprite.bitmap[row_in_sprite * w..(row_in_sprite + 1) * w];
            for (i, &byte) in src.iter().enumerate() {
                let dst_x = x_byte + i32::try_from(i).expect("sprite width fits in i32");
                if !(0..i32::from(ROOM_WIDTH_BYTES)).contains(&dst_x) {
                    continue;
                }
                let idx = (screen_y as usize) * usize::from(ROOM_WIDTH_BYTES) + (dst_x as usize);
                let dst = &mut self.bytes[idx];
                match opacity {
                    Opacity::Or => *dst |= byte,
                    Opacity::And => *dst &= byte,
                    Opacity::Sta => *dst = byte,
                }
            }
        }
    }

    /// Compose every cell of `ctx.room` onto a fresh canvas.
    fn compose(ctx: &RoomContext, bg: &BiomeTables) -> Self {
        let mut canvas = Self::new();
        // Three on-screen rows, top → bottom.
        for (row, &dy_byte) in BLOCK_BOT_ROW.iter().enumerate().take(ROOM_HEIGHT) {
            let dy = i32::from(dy_byte);
            let ay = dy - i32::from(D_HEIGHT);
            for col in 0..ROOM_WIDTH {
                let blockxco = (col as i32) * i32::from(CELL_WIDTH_BYTES);
                let me = ctx.tile(col as i32, row as i32);
                let left = if col == 0 {
                    ctx.left_tile(ROOM_WIDTH - 1, row)
                } else {
                    ctx.tile((col - 1) as i32, row as i32)
                };
                let below_left = if row + 1 >= ROOM_HEIGHT {
                    // Below-row off-screen: pull from below-neighbour.
                    if col == 0 {
                        ctx.below_left_top_right()
                    } else {
                        ctx.below_top(col - 1)
                    }
                } else if col == 0 {
                    ctx.left_tile(ROOM_WIDTH - 1, row + 1)
                } else {
                    ctx.tile((col - 1) as i32, (row + 1) as i32)
                };
                draw_block(&mut canvas, bg, me, left, below_left, blockxco, ay, dy);
            }
        }
        // "Ceiling" D-pass: bottom row of the above-neighbour, drawn at
        // the top of the screen (FRAMEADV.S:113-178).
        // Top-of-screen baselines for the "ceiling" D-pass — Dy=2
        // sits 2px below the top, Ay=-1 places A-pieces just off
        // the top of the visible area.
        let ceil_dy: i32 = 2;
        let ceil_above_y: i32 = -1;
        for col in 0..ROOM_WIDTH {
            let blockxco = (col as i32) * i32::from(CELL_WIDTH_BYTES);
            let me = ctx.above_bottom_row(col);
            let left = if col == 0 {
                Tile::default()
            } else {
                ctx.above_bottom_row(col - 1)
            };
            let below_left = if col == 0 {
                ctx.left_tile(ROOM_WIDTH - 1, 0)
            } else {
                ctx.tile((col - 1) as i32, 0)
            };
            draw_d_only(
                &mut canvas,
                bg,
                me,
                left,
                below_left,
                blockxco,
                ceil_above_y,
                ceil_dy,
            );
        }
        canvas
    }

    fn into_frame(self, mode: RenderMode) -> Option<Frame> {
        // hires::render_linear flips bottom-up; our canvas is top-down,
        // so pre-flip rows before handing off.
        let row_bytes = usize::from(ROOM_WIDTH_BYTES);
        let mut flipped = vec![0u8; self.bytes.len()];
        for y in 0..usize::from(ROOM_HEIGHT_PX) {
            let src = &self.bytes[y * row_bytes..(y + 1) * row_bytes];
            let dst_y = usize::from(ROOM_HEIGHT_PX) - 1 - y;
            flipped[dst_y * row_bytes..(dst_y + 1) * row_bytes].copy_from_slice(src);
        }
        hires::render_linear(&flipped, ROOM_WIDTH_BYTES, ROOM_HEIGHT_PX, mode)
    }
}

// ---------------------------------------------------------------------------
// Per-cell draw — mirrors FRAMEADV.S RedBlockSure.
// ---------------------------------------------------------------------------

/// Returns `true` if a tile's A-section blocks the C-section beneath
/// (the "checkc" predicate in `FRAMEADV.S:917`).
fn c_section_visible(me: TileKind) -> bool {
    matches!(
        me,
        TileKind::Empty
            | TileKind::PillarTop
            | TileKind::PanelWithoutFloor
            | TileKind::ArchTop1
            | TileKind::ArchTop2
            | TileKind::ArchTop3
            | TileKind::ArchTop4
    )
}

/// Returns `true` if the left tile's B-section "intrudes" into this
/// cell and we need to AND a mask before OR'ing the A-piece
/// (FRAMEADV.S:1134-1153).
fn left_intrudes(left: TileKind) -> bool {
    matches!(
        left,
        TileKind::PanelWithFloor
            | TileKind::PanelWithoutFloor
            | TileKind::PillarTop
            | TileKind::Block
    )
}

/// Full per-cell draw: C, B, D, A, Front. Order matches
/// `RedBlockSure` in `FRAMEADV.S`.
#[allow(clippy::too_many_arguments)]
fn draw_block(
    canvas: &mut Canvas,
    bg: &BiomeTables,
    me: Tile,
    left: Tile,
    below_left: Tile,
    blockxco: i32,
    ay: i32,
    dy: i32,
) {
    draw_c(canvas, bg, me.kind, below_left, blockxco, dy);
    draw_b(canvas, bg, left, below_left, blockxco, ay, dy, bg.biome);
    draw_d(canvas, bg, me, blockxco, dy);
    draw_a(canvas, bg, me, left.kind, blockxco, ay);
    draw_front(canvas, bg, me, blockxco, ay);
}

/// Reduced per-cell draw used for the ceiling pass — C, B, D, Front
/// only (no A — that would belong to the cell whose top we're
/// peeking at).
#[allow(clippy::too_many_arguments)]
fn draw_d_only(
    canvas: &mut Canvas,
    bg: &BiomeTables,
    me: Tile,
    left: Tile,
    below_left: Tile,
    blockxco: i32,
    ay: i32,
    dy: i32,
) {
    draw_c(canvas, bg, me.kind, below_left, blockxco, dy);
    draw_b(canvas, bg, left, below_left, blockxco, ay, dy, bg.biome);
    draw_d(canvas, bg, me, blockxco, dy);
    draw_front(canvas, bg, me, blockxco, ay);
}

fn draw_a(canvas: &mut Canvas, bg: &BiomeTables, me: Tile, left: TileKind, blockxco: i32, ay: i32) {
    if left_intrudes(left) {
        if let Some(mask) = bg.resolve(MASK_A[me.kind as usize]) {
            canvas.blit(mask, blockxco, ay, Opacity::And);
        }
    }
    let piece_id = PIECE_A[me.kind as usize];
    if let Some(piece) = bg.resolve(piece_id) {
        let y = ay + i32::from(PIECE_A_Y[me.kind as usize]);
        canvas.blit(piece, blockxco, y, Opacity::Or);
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_b(
    canvas: &mut Canvas,
    bg: &BiomeTables,
    left: Tile,
    below_left: Tile,
    blockxco: i32,
    ay: i32,
    dy: i32,
    biome: Biome,
) {
    // Block solid hides B-section (FRAMEADV.S:998-1000).
    if left.kind == TileKind::Block {
        // Variant-aware B-piece for solid blocks.
        let variant = usize::from(left.variant) % BLOCK_B.len();
        if let Some(piece) = bg.resolve(BLOCK_B[variant]) {
            let y = ay + i32::from(PIECE_B_Y[TileKind::Block as usize]);
            canvas.blit(piece, blockxco, y, Opacity::Or);
        }
        // No bstripe over solid blocks.
        return;
    }
    // Back-wall variants for space / floor (FRAMEADV.S:1055-1072).
    match left.kind {
        TileKind::Empty => {
            let v = usize::from(below_left.variant) % SPACE_B.len();
            if let Some(piece) = bg.resolve(SPACE_B[v]) {
                let y = ay + i32::from(SPACE_B_Y[v]);
                canvas.blit(piece, blockxco, y, Opacity::Or);
            }
            return;
        }
        TileKind::Floor => {
            let v = usize::from(below_left.variant) % FLOOR_B.len();
            if let Some(piece) = bg.resolve(FLOOR_B[v]) {
                let y = ay + i32::from(FLOOR_B_Y[v]);
                canvas.blit(piece, blockxco, y, Opacity::Or);
            }
            return;
        }
        _ => {}
    }
    let raw_b = PIECE_B[left.kind as usize];
    if raw_b == PANEL_B0_SENTINEL {
        let v = usize::from(left.variant) % PANEL_B.len();
        if let Some(piece) = bg.resolve(PANEL_B[v]) {
            let y = ay + i32::from(PIECE_B_Y[left.kind as usize]);
            canvas.blit(piece, blockxco, y, Opacity::Or);
        }
    } else if let Some(piece) = bg.resolve(raw_b) {
        let y = ay + i32::from(PIECE_B_Y[left.kind as usize]);
        canvas.blit(piece, blockxco, y, Opacity::Or);
    }
    // Palace-only diagonal stripe (FRAMEADV.S:1020-1037).
    if biome == Biome::Palace {
        if let Some(stripe) = bg.resolve(B_STRIPE[left.kind as usize]) {
            canvas.blit(stripe, blockxco, ay - 32, Opacity::Or);
        }
    }
    // Mask off the right portion of left's B-piece where current
    // tile's A-section will sit (FRAMEADV.S:domaskb).
    if let Some(mask) = bg.resolve(MASK_B[left.kind as usize]) {
        // domaskb uses Dy as YCO, not Ay (FRAMEADV.S:986).
        canvas.blit(mask, blockxco, dy, Opacity::And);
    }
}

fn draw_c(
    canvas: &mut Canvas,
    bg: &BiomeTables,
    me: TileKind,
    below_left: Tile,
    blockxco: i32,
    dy: i32,
) {
    if !c_section_visible(me) {
        return;
    }
    if below_left.kind == TileKind::Block {
        let v = usize::from(below_left.variant) % BLOCK_C.len();
        if let Some(piece) = bg.resolve(BLOCK_C[v]) {
            canvas.blit(piece, blockxco, dy, Opacity::Or);
        }
        return;
    }
    let raw_c = PIECE_C[below_left.kind as usize];
    if raw_c == PANEL_C0_SENTINEL {
        let v = usize::from(below_left.variant) % PANEL_C.len();
        if let Some(piece) = bg.resolve(PANEL_C[v]) {
            canvas.blit(piece, blockxco, dy, Opacity::Or);
        }
    } else if let Some(piece) = bg.resolve(raw_c) {
        canvas.blit(piece, blockxco, dy, Opacity::Or);
    }
}

fn draw_d(canvas: &mut Canvas, bg: &BiomeTables, me: Tile, blockxco: i32, dy: i32) {
    let opacity = if me.kind == TileKind::PanelWithoutFloor {
        Opacity::Or
    } else {
        Opacity::Sta
    };
    if me.kind == TileKind::Block {
        let v = usize::from(me.variant) % BLOCK_D.len();
        if let Some(piece) = bg.resolve(BLOCK_D[v]) {
            canvas.blit(piece, blockxco, dy, opacity);
        }
        return;
    }
    let piece_id = PIECE_D[me.kind as usize];
    if let Some(piece) = bg.resolve(piece_id) {
        canvas.blit(piece, blockxco, dy, opacity);
    }
}

fn draw_front(canvas: &mut Canvas, bg: &BiomeTables, me: Tile, blockxco: i32, ay: i32) {
    if me.kind == TileKind::Block {
        let v = usize::from(me.variant) % BLOCK_FR.len();
        if let Some(piece) = bg.resolve(BLOCK_FR[v]) {
            // Block front: Y offset is 0 per BGDATA (FRONT_Y[block]=0).
            canvas.blit(piece, blockxco, ay, Opacity::Sta);
        }
        return;
    }
    let piece_id = FRONT_I[me.kind as usize];
    if let Some(piece) = bg.resolve(piece_id) {
        let y = ay + i32::from(FRONT_Y[me.kind as usize]);
        let x = blockxco + i32::from(FRONT_X[me.kind as usize]);
        canvas.blit(piece, x, y, Opacity::Sta);
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn vendor_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor/pop-apple2/04 Support")
    }

    fn load_level(n: u8) -> Level {
        Level::from_file(vendor_root().join("Levels").join(format!("LEVEL{n}"))).unwrap()
    }

    #[test]
    fn dungeon_tables_load() {
        let tables = BiomeTables::load(&vendor_root(), Biome::Dungeon).expect("DUN1 + DUN2 load");
        // Both tables ship a non-trivial number of sprites. Exact
        // counts differ between BGTAB1 (the full piece set) and BGTAB2
        // (extras / overflow), so we only sanity-check the floor here
        // — a stricter pin would just track vendor-file fingerprints.
        assert!(tables.table1.images.len() >= 30);
        assert!(tables.table2.images.len() >= 10);
    }

    #[test]
    fn render_level1_room1_produces_full_size_frame() {
        // LEVEL1 room 1 is the prince start. Verify the framebuffer
        // comes back at the expected screen dimensions, isn't
        // all-zero, and is reproducible from the same inputs.
        let level = load_level(1);
        let tables = BiomeTables::load(&vendor_root(), Biome::Dungeon).unwrap();
        let frame =
            render_room(&level, 1, &tables, RenderMode::Monochrome).expect("room 1 renders");
        assert_eq!(frame.width, 280);
        assert_eq!(frame.height, 192);
        let nonblack = frame
            .pixels
            .chunks_exact(4)
            .filter(|p| p[0..3] != [0, 0, 0])
            .count();
        assert!(
            nonblack > 100,
            "expected non-trivial sprite coverage; got {nonblack} non-black pixels"
        );
        // Determinism: same call again hashes identically.
        let again = render_room(&level, 1, &tables, RenderMode::Monochrome).unwrap();
        assert_eq!(frame.pixels, again.pixels);
    }

    #[test]
    fn render_returns_none_for_out_of_range_room() {
        let level = load_level(1);
        let tables = BiomeTables::load(&vendor_root(), Biome::Dungeon).unwrap();
        assert!(render_room(&level, 0, &tables, RenderMode::Monochrome).is_none());
        assert!(render_room(&level, 25, &tables, RenderMode::Monochrome).is_none());
    }

    #[test]
    fn every_bundled_level_renders_room_one() {
        // Catches OOB indexing, missing biome files, and panics in the
        // BGDATA tables for any of the 15 bundled levels.
        for n in 0u8..=14 {
            let level = load_level(n);
            let biome = Biome::for_level(usize::from(n)).expect("bundled biome");
            let tables = BiomeTables::load(&vendor_root(), biome).expect("biome tables");
            let frame = render_room(&level, 1, &tables, RenderMode::Monochrome).expect("renders");
            assert_eq!(frame.width, 280);
            assert_eq!(frame.height, 192);
        }
    }

    #[test]
    fn ntsc_and_mono_modes_produce_different_output() {
        let level = load_level(1);
        let tables = BiomeTables::load(&vendor_root(), Biome::Dungeon).unwrap();
        let mono = render_room(&level, 1, &tables, RenderMode::Monochrome).unwrap();
        let ntsc = render_room(&level, 1, &tables, RenderMode::NtscColor).unwrap();
        assert_eq!(mono.pixels.len(), ntsc.pixels.len());
        assert_ne!(mono.pixels, ntsc.pixels);
    }
}
