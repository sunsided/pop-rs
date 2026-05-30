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
//!   [`crate::bgdata`]. `domaskb` (left's `MASK_B` AND'd into the
//!   current D-strip) runs at the end of `draw_c`, BEFORE `draw_b`,
//!   matching `FRAMEADV.S:906`.
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
//! * At-rest "movable" pieces: torch flame (`drawtorchb`), loose-floor
//!   B-edge (`drawlooseb`), closed-gate vertical bars (`drawgateb`),
//!   gate top piece into the cell above (`drawgatec` via `draw_mc`),
//!   and the exit-door stack + stairs + top-repair (`drawexitb`). The
//!   stairs piece is suppressed in the prince's start room, matching
//!   the `cmp KidStartScrn beq :nostairs` branch in `FRAMEADV.S:1635`.
//! * Loose-floor visual fix: `LOOSE_A[0]` + `LOOSE_D[0]` substituted
//!   for the empty `PIECE_A` / `PIECE_D` so the editor doesn't show
//!   blank cells where breakable tiles live.
//!
//! # What's not modelled (yet)
//!
//! All animated / state-sensitive specials punted to a follow-up PR:
//!
//! * Loose floor mid-fall animation frames (`LOOSE_A[1..]`).
//! * Spike / slicer animation frames (`drawma` / `drawspikea` /
//!   `drawslicera`) — uses the BGDATA `PIECE_A[k]` at-rest value.
//! * Gate bars at partial heights — gates always render fully closed.
//! * Depressed press-plate state — uses the up-state piece.
//! * Flask bubbles / sword gleam — uses the BGDATA piece.
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
    CELL_WIDTH_BYTES, DOOR, DOOR_MASK, D_HEIGHT, FLOOR_B, FLOOR_B_Y, FRONT_I, FRONT_X, FRONT_Y,
    GATE_8B, GATE_8C, GATE_B1, GATE_BOT_ORA, GATE_C_MASK, LOOSE_A, LOOSE_B, LOOSE_B_Y, LOOSE_D,
    MASK_A, MASK_B, PANEL_B, PANEL_B0_SENTINEL, PANEL_C, PANEL_C0_SENTINEL, PIECE_A, PIECE_A_Y,
    PIECE_B, PIECE_B_Y, PIECE_C, PIECE_D, ROOM_HEIGHT_PX, ROOM_WIDTH_BYTES, SPACE_B, SPACE_B_Y,
    STAIRS, TOP_REPAIR, TORCH_FLAME,
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
    // `drawexitb` skips stairs when the current room is the prince's
    // entry point (`FRAMEADV.S:1635` `cmp KidStartScrn beq :nostairs`).
    let draw_stairs = room_id != level.prince_start().screen;
    let canvas = Canvas::compose(&ctx, bg, draw_stairs);
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
        // Original `CLS` (`HIRES.S:206`) fills the hires page with
        // `lda #$80` ("black2") — bit 7 of every byte set. That bit
        // is the NTSC palette-select; for unlit pixels it doesn't
        // change `black-stays-black`, but it DOES affect downstream
        // `AND`/`OR` operations: `MASK_B`'s AND can preserve or
        // clear the palette bit per the mask sprite, and a later
        // `pieceb` OR then paints lit pixels with whichever palette
        // bit survives. Initialising to `0x00` instead silently
        // shifts NTSC artifact colours inside `domaskb` carved
        // regions — visible as the "triangular gaps" in the floor
        // strip alongside columns / arches.
        Self {
            bytes: Box::new(
                [0x80; (ROOM_WIDTH_BYTES as usize) * (ROOM_HEIGHT_PX as usize)],
            ),
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
    fn compose(ctx: &RoomContext, bg: &BiomeTables, draw_stairs: bool) -> Self {
        let mut canvas = Self::new();
        // Three on-screen rows, bottom → top — matches `FRAMEADV.S:62
        // SURE` (`ldy #2 :row sty rowno ... dey jmp :row`). Order
        // matters: row 1's tall sprites (e.g. red-biome
        // `SPACE_B[1]` 52-px window, drawn at `Ay − 20`) extend
        // upward into row 0's pixel area. With bottom-up rendering
        // row 0 is processed *after* row 1, so row 0's own `pieced`
        // (STA opacity at the bottom of its A-section) cleanly
        // overwrites any bleed-up before the frame is finalised.
        // Top-down rendering produced the visible "window bleeds
        // through floor" artefact in LV12 R19 / R20.
        for (row, &dy_byte) in BLOCK_BOT_ROW.iter().enumerate().take(ROOM_HEIGHT).rev() {
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
                draw_block(
                    &mut canvas,
                    bg,
                    me,
                    left,
                    below_left,
                    blockxco,
                    ay,
                    dy,
                    draw_stairs,
                );
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

/// Clamp a block-tile state byte to `0..2` (`numblox`). Matches the
/// `cpy #numblox bcc :1 ldy #0` clamp branches in
/// `FRAMEADV.S:1048` / `1118` / `:block` cases — out-of-range states
/// fall back to variant 0, **not** wrap-around.
fn block_variant(state: u8) -> usize {
    if usize::from(state) < BLOCK_B.len() {
        usize::from(state)
    } else {
        0
    }
}

/// Resolve a panel-tile state byte to a 0..`numpans` index. Returns
/// `None` for out-of-range — the original engine `bcs ]rts` in
/// `FRAMEADV.S:1041` skips the panel branch entirely rather than
/// clamping (so weird state bytes don't paint a stray default).
fn panel_index(state: u8) -> Option<usize> {
    let v = usize::from(state);
    if v < PANEL_B.len() {
        Some(v)
    } else {
        None
    }
}

/// Resolve a `space`-back-wall state byte. Skips out-of-range per
/// `FRAMEADV.S:1067 cpy #numbpans+1 bcs ]rts`.
fn space_index(state: u8) -> Option<usize> {
    let v = usize::from(state);
    if v < SPACE_B.len() {
        Some(v)
    } else {
        None
    }
}

/// Resolve a `floor`-back-wall state byte. Clamps to 0 per
/// `FRAMEADV.S:1056 cpy #numbpans+1 bcc :3 ldy #0`.
fn floor_index(state: u8) -> usize {
    let v = usize::from(state);
    if v < FLOOR_B.len() {
        v
    } else {
        0
    }
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

/// Full per-cell draw: C, mC, B, mB, D, mD, A, Front. Order matches
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
    draw_stairs: bool,
) {
    draw_c(canvas, bg, me.kind, below_left, blockxco, dy, left.kind);
    draw_mc(canvas, bg, me.kind, below_left.kind, blockxco, dy, ay);
    draw_b(canvas, bg, left, below_left, blockxco, ay, dy, bg.biome);
    draw_mb(canvas, bg, left.kind, blockxco, ay, dy, draw_stairs);
    draw_d(canvas, bg, me, blockxco, dy);
    draw_md(canvas, bg, me, blockxco, dy);
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
    draw_c(canvas, bg, me.kind, below_left, blockxco, dy, left.kind);
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
    // Loose floor's `PIECE_A` is empty in BGDATA (the game fills the
    // upper section via the left-neighbour's `pieceb` overflow). For
    // a static editor we want loose tiles to always look like a
    // floor, so substitute `LOOSE_A[0]` (= regular floor A-piece).
    let piece_id = if me.kind == TileKind::LooseFloor {
        LOOSE_A[0]
    } else {
        PIECE_A[me.kind as usize]
    };
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
    _below_left: Tile,
    blockxco: i32,
    ay: i32,
    dy: i32,
    biome: Biome,
) {
    // Block solid hides B-section (FRAMEADV.S:998-1000). State byte
    // (BLUESPEC, exposed as `Tile::modifier`) selects between two
    // block-B variants; FRAMEADV's `:block` branch clamps an
    // out-of-range state to 0 rather than skipping (FRAMEADV.S:1048).
    if left.kind == TileKind::Block {
        let v = block_variant(left.modifier);
        if let Some(piece) = bg.resolve(BLOCK_B[v]) {
            let y = ay + i32::from(PIECE_B_Y[TileKind::Block as usize]);
            canvas.blit(piece, blockxco, y, Opacity::Or);
        }
        return;
    }
    // Back-wall variants for space / floor (FRAMEADV.S:1055-1072).
    // Both branches key off `spreced` — the LEFT-neighbour's state
    // byte — not the cell-below-left's. Space SKIPS on out-of-range,
    // floor CLAMPS to 0; we mirror both exactly.
    match left.kind {
        TileKind::Empty => {
            if let Some(v) = space_index(left.modifier) {
                if let Some(piece) = bg.resolve(SPACE_B[v]) {
                    let y = ay + i32::from(SPACE_B_Y[v]);
                    canvas.blit(piece, blockxco, y, Opacity::Or);
                }
            }
            return;
        }
        TileKind::Floor => {
            let v = floor_index(left.modifier);
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
        // Panel branch SKIPS for out-of-range state (FRAMEADV.S:1041
        // `cpy #numpans bcs ]rts`).
        if let Some(v) = panel_index(left.modifier) {
            if let Some(piece) = bg.resolve(PANEL_B[v]) {
                let y = ay + i32::from(PIECE_B_Y[left.kind as usize]);
                canvas.blit(piece, blockxco, y, Opacity::Or);
            }
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
    // Note: `domaskb` (MASK_B AND'd at Dy) runs at the *end of drawc*
    // in `FRAMEADV.S:906`, BEFORE drawb fires. That ordering matters:
    // mask carves the canvas, then pieceb fills the carved area. We
    // mirror it via `draw_c`'s tail.
    let _ = dy;
}

/// `drawc` from `FRAMEADV.S:906` — when `me` is "see-through" (the
/// C-section behind it is visible), OR the below-left's C-piece into
/// the current cell and AND the left-neighbour's `MASK_B` over the
/// D-strip area to carve room for the left's `pieceb` overflow.
///
/// Both the C-piece draw AND the mask are gated on `c_section_visible`
/// — the original engine returns at `bcc ]rts` (FRAMEADV.S:908) before
/// `domaskb` runs when `checkc` says the C-section is hidden, because
/// the current tile's own `pieced` (drawn later with `Sta` opacity)
/// would otherwise overwrite the carved area, AND the mask is tall
/// enough to bleed *above* the floor strip into the A-section where
/// the overdraw never restores it — that bled-out area is what showed
/// up as a black rectangle to the left of solid-floor neighbours.
fn draw_c(
    canvas: &mut Canvas,
    bg: &BiomeTables,
    me: TileKind,
    below_left: Tile,
    blockxco: i32,
    dy: i32,
    left: TileKind,
) {
    if !c_section_visible(me) {
        return;
    }
    draw_c_piece(canvas, bg, below_left, blockxco, dy);
    if let Some(mask) = bg.resolve(MASK_B[left as usize]) {
        canvas.blit(mask, blockxco, dy, Opacity::And);
    }
}

fn draw_c_piece(canvas: &mut Canvas, bg: &BiomeTables, below_left: Tile, blockxco: i32, dy: i32) {
    if below_left.kind == TileKind::Block {
        let v = block_variant(below_left.modifier);
        if let Some(piece) = bg.resolve(BLOCK_C[v]) {
            canvas.blit(piece, blockxco, dy, Opacity::Or);
        }
        return;
    }
    let raw_c = PIECE_C[below_left.kind as usize];
    if raw_c == PANEL_C0_SENTINEL {
        if let Some(v) = panel_index(below_left.modifier) {
            if let Some(piece) = bg.resolve(PANEL_C[v]) {
                canvas.blit(piece, blockxco, dy, Opacity::Or);
            }
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
        let v = block_variant(me.modifier);
        if let Some(piece) = bg.resolve(BLOCK_D[v]) {
            canvas.blit(piece, blockxco, dy, opacity);
        }
        return;
    }
    // Loose floor's `PIECE_D` is empty in BGDATA; the game emits the
    // strip via `drawmd → drawloosed`. Use `LOOSE_D[0]` (= regular
    // floor D-strip) for the at-rest visual.
    let piece_id = if me.kind == TileKind::LooseFloor {
        LOOSE_D[0]
    } else {
        PIECE_D[me.kind as usize]
    };
    if let Some(piece) = bg.resolve(piece_id) {
        canvas.blit(piece, blockxco, dy, opacity);
    }
}

/// Movable B-piece — drawn into the CURRENT cell based on the
/// LEFT neighbour's kind. Mirrors `FRAMEADV.S:drawmb` for the few
/// tile kinds that have a visible at-rest spillover:
/// torch flame, loose-floor B-edge, closed-gate bar grill, and the
/// exit-door stairs / door stack.
fn draw_mb(
    canvas: &mut Canvas,
    bg: &BiomeTables,
    left: TileKind,
    blockxco: i32,
    ay: i32,
    dy: i32,
    draw_stairs: bool,
) {
    match left {
        TileKind::Torch => {
            // SETUPFLAME (`GAMEBG.S:735`): no flame on the leftmost
            // torch (would draw off the room's left edge), advance
            // X by 1 byte, drop Y by 43 px, frame 0 is at-rest.
            if blockxco == 0 {
                return;
            }
            if let Some(piece) = bg.resolve(TORCH_FLAME[0]) {
                canvas.blit(piece, blockxco + 1, ay - 43, Opacity::Sta);
            }
        }
        TileKind::LooseFloor => {
            // drawlooseb (`FRAMEADV.S:1388`) at state=0 → looseb at
            // Ay + LOOSE_B_Y[0] = Ay + 0.
            if let Some(piece) = bg.resolve(LOOSE_B) {
                canvas.blit(piece, blockxco, ay + i32::from(LOOSE_B_Y[0]), Opacity::Or);
            }
        }
        TileKind::Gate => {
            draw_gate_bars(canvas, bg, blockxco, ay);
        }
        TileKind::Exit => {
            draw_exit_door(canvas, bg, blockxco, ay, dy, draw_stairs);
        }
        _ => {}
    }
}

/// Movable D-piece — handles `drawmd` for the loose-floor at-rest
/// state. Loose's regular `draw_d` already substitutes `LOOSE_D[0]`,
/// so this is a no-op today; kept as a hook for the future animated
/// loose-floor frames.
fn draw_md(_canvas: &mut Canvas, _bg: &BiomeTables, _me: Tile, _blockxco: i32, _dy: i32) {}

/// Movable C-piece — fires only when CURRENT is a "see-through" kind
/// (space / panelwof / pillartop) and the BELOW-LEFT neighbour is a
/// gate. Mirrors `FRAMEADV.S:drawmc → drawgatec`: AND a fixed mask
/// over the current cell's D-strip, then OR the gate's top piece.
/// At state=0 (gate closed) the top piece is `GATE_8C[0]`.
fn draw_mc(
    canvas: &mut Canvas,
    bg: &BiomeTables,
    me: TileKind,
    below_left: TileKind,
    blockxco: i32,
    dy: i32,
    _ay: i32,
) {
    if !matches!(
        me,
        TileKind::Empty | TileKind::PanelWithoutFloor | TileKind::PillarTop
    ) {
        return;
    }
    if below_left != TileKind::Gate {
        return;
    }
    if let Some(mask) = bg.resolve(GATE_C_MASK) {
        canvas.blit(mask, blockxco, dy, Opacity::And);
    }
    if let Some(piece) = bg.resolve(GATE_8C[0]) {
        canvas.blit(piece, blockxco, dy, Opacity::Or);
    }
}

/// Draw the closed-gate vertical bar grill into the cell to the
/// right of a gate tile. Mirrors `FRAMEADV.S:drawgateb` at state=0
/// (gateposn = 1, gatebot = Ay − 1).
///
/// Stacks `GATE_B1` 8-pixel grill pieces from `gatebot − 12`
/// upward by 8 pixels each, stopping when the next piece's top
/// would rise above `blockthr = Ay − 59`. With state=0 this fits
/// six middle pieces between the bottom strip and a variable-height
/// `GATE_8B[height − 1]` top piece that fills the leftover space —
/// matching the `:done` tail of `FRAMEADV.S:drawgateb` (without that
/// top piece the rendered gate had a black horizontal gap at the top
/// edge inside the bars).
fn draw_gate_bars(canvas: &mut Canvas, bg: &BiomeTables, blockxco: i32, ay: i32) {
    let gate_bot = ay - 1;
    // Bottom strip: `gatebotORA` at gatebot − 2.
    if let Some(piece) = bg.resolve(GATE_BOT_ORA) {
        canvas.blit(piece, blockxco, gate_bot - 2, Opacity::Or);
    }
    // Middle grill pieces.
    let blockthr = ay - 59;
    let middle = bg.resolve(GATE_B1);
    let mut y = gate_bot - 12;
    while y - 7 >= blockthr && y >= 0 {
        if let Some(piece) = middle {
            canvas.blit(piece, blockxco, y, Opacity::Sta);
        }
        y -= 8;
    }
    // Variable-height top piece (`FRAMEADV.S:1889-1907`):
    //   desired_height = YCO − blockthr + 1, in 1..=8
    //   sprite        = GATE_8B[desired_height − 1]
    // `y` here is the YCO that *failed* the loop's `y-7 >= blockthr`
    // test, so it's the right starting point for the top piece.
    if y >= 0 {
        let desired_height = y - blockthr + 1;
        if (1..=8).contains(&desired_height) {
            let idx = usize::try_from(desired_height - 1).unwrap_or(0);
            if let Some(piece) = bg.resolve(GATE_8B[idx]) {
                canvas.blit(piece, blockxco, y, Opacity::Sta);
            }
        }
    }
}

/// Draw the closed-exit door + stairs + top-repair into the cell to
/// the right of an exit tile. Mirrors `FRAMEADV.S:drawexitb` at
/// state=0 (gateposn = 0, door at rest).
///
/// * Stairs sit at `(blockxco + 1, ay − 12)` — skipped when this
///   room is the prince's start (the entrance side has no stairs)
///   or when the cell hugs the right wall (`blockxco >= 36`,
///   matching the original's "can't protrude off R" guard).
/// * Door pieces stack from `ay − 14` downward by 4 px each, with
///   `DOOR_MASK` AND'd then `DOOR` OR'd, until the next piece's
///   bottom would dip below `blockthr = dy − 67`. Top-row exits
///   (where blockthr would wrap negative in the original 6502
///   byte arithmetic) skip the door stack entirely.
/// * `TOP_REPAIR` paints the cell strip above the door so the
///   wall edge reads cleanly.
fn draw_exit_door(
    canvas: &mut Canvas,
    bg: &BiomeTables,
    blockxco: i32,
    ay: i32,
    dy: i32,
    draw_stairs: bool,
) {
    let canvas_h = i32::from(ROOM_HEIGHT_PX);
    if draw_stairs && blockxco < 36 {
        if let Some(piece) = bg.resolve(STAIRS) {
            canvas.blit(piece, blockxco + 1, ay - 12, Opacity::Sta);
        }
    }
    let blockthr = dy - 67;
    if (0..canvas_h).contains(&blockthr) {
        // Both sprite refs are loop-invariant — resolve them once up
        // front so we don't re-index the BGTAB tables per slice.
        let door_mask = bg.resolve(DOOR_MASK);
        let door = bg.resolve(DOOR);
        // state=0 → gateposn=0; door top starts at `ay − 14`.
        let mut y = ay - 14;
        while y >= blockthr {
            if let Some(mask) = door_mask {
                canvas.blit(mask, blockxco, y, Opacity::And);
            }
            if let Some(piece) = door {
                canvas.blit(piece, blockxco, y, Opacity::Or);
            }
            y -= 4;
        }
    }
    let top_y = ay - 64;
    if (0..canvas_h).contains(&top_y) {
        if let Some(piece) = bg.resolve(TOP_REPAIR) {
            canvas.blit(piece, blockxco, top_y, Opacity::Sta);
        }
    }
}

fn draw_front(canvas: &mut Canvas, bg: &BiomeTables, me: Tile, blockxco: i32, ay: i32) {
    if me.kind == TileKind::Block {
        let v = block_variant(me.modifier);
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

    /// Synth a single-room mini-level for renderer-bug pinning. All
    /// rooms after the seed are blank so cross-room neighbour fetches
    /// land on `Empty` tiles cleanly.
    fn synth_level_with(tiles: [Tile; ROOM_WIDTH * ROOM_HEIGHT]) -> Level {
        // Base off a real level so headers + maps parse — then poke
        // the room 1 tiles. Reusing LEVEL0 also gives the dungeon
        // biome, so loading DUN1/DUN2 works.
        let mut level = load_level(0);
        level.rooms[0] = crate::level::Room { tiles };
        level
    }

    /// True if any pixel in the canvas rect `(x0..x1, y0..y1)` is
    /// non-black (i.e. the renderer drew something there).
    fn has_non_black(frame: &Frame, x0: u32, y0: u32, x1: u32, y1: u32) -> bool {
        let w = frame.width;
        for y in y0..y1 {
            for x in x0..x1 {
                let i = ((y * w + x) * 4) as usize;
                if frame.pixels[i..i + 3] != [0, 0, 0] {
                    return true;
                }
            }
        }
        false
    }

    #[test]
    fn loose_floor_renders_as_floor_at_rest() {
        // Regression: pre-fix the editor showed loose tiles as
        // completely empty cells (PIECE_A[loose] = 0, PIECE_D[loose]
        // = 0). The render now substitutes LOOSE_A[0] / LOOSE_D[0]
        // for the at-rest visual.
        let mut tiles = [Tile::default(); ROOM_WIDTH * ROOM_HEIGHT];
        // Single loose floor in the middle of the top row, with
        // Empty (= space) tiles to either side so left-neighbour
        // pieceb can't accidentally fill the cell.
        tiles[5] = Tile {
            kind: TileKind::LooseFloor,
            variant: 0,
            modifier: 0,
        };
        let level = synth_level_with(tiles);
        let tables = BiomeTables::load(&vendor_root(), Biome::Dungeon).unwrap();
        let frame = render_room(&level, 1, &tables, RenderMode::Monochrome).unwrap();
        // Loose tile sits at row 0, col 5. Cell occupies screen
        // (col*28..(col+1)*28, 2..65) (top row's D-strip at y=65).
        // The D-strip alone (3 px tall) would put any pixels in
        // y=62..65; the LOOSE_A substitution adds extra pixels above
        // that. Probe well above the strip — pre-fix this area was
        // all black.
        assert!(
            has_non_black(&frame, 140, 50, 168, 62),
            "loose-floor A-section should be visible above the D-strip"
        );
    }

    #[test]
    fn torch_emits_flame_in_right_cell() {
        // Regression: pre-fix torches had no flame because `drawmb`
        // wasn't wired up. The flame sprite belongs to the cell to
        // the right of the torch base.
        let mut tiles = [Tile::default(); ROOM_WIDTH * ROOM_HEIGHT];
        // Torch in middle row, col 3 — flame should appear in col 4
        // at roughly Ay − 43 above the row baseline.
        tiles[ROOM_WIDTH + 3] = Tile {
            kind: TileKind::Torch,
            variant: 0,
            modifier: 0,
        };
        let level = synth_level_with(tiles);
        let tables = BiomeTables::load(&vendor_root(), Biome::Dungeon).unwrap();
        let frame = render_room(&level, 1, &tables, RenderMode::Monochrome).unwrap();
        // Middle row Ay = 125, flame baseline at y = 82. Probe a
        // band around the right-of-torch cell.
        assert!(
            has_non_black(&frame, 112, 70, 168, 95),
            "torch flame should be visible in the cell to the right"
        );
    }

    #[test]
    fn gate_emits_bars_in_right_cell() {
        // Regression: pre-fix gates rendered as plain floor in the
        // right cell because `drawgateb` wasn't wired. At state=0
        // (closed) the bars fill most of the vertical cell.
        let mut tiles = [Tile::default(); ROOM_WIDTH * ROOM_HEIGHT];
        tiles[ROOM_WIDTH + 2] = Tile {
            kind: TileKind::Gate,
            variant: 0,
            modifier: 0,
        };
        let level = synth_level_with(tiles);
        let tables = BiomeTables::load(&vendor_root(), Biome::Dungeon).unwrap();
        let frame = render_room(&level, 1, &tables, RenderMode::Monochrome).unwrap();
        // Gate's right cell is col 3, middle row (Ay = 125). The
        // bars span roughly y = 70..125. Probe the middle of that
        // band.
        assert!(
            has_non_black(&frame, 84, 80, 112, 120),
            "closed-gate bars should fill the cell to the right"
        );
    }

    #[test]
    fn exit_emits_door_in_right_cell() {
        // Regression: pre-fix exit tiles rendered as plain floor
        // because `drawexitb` wasn't wired. At state=0 (closed) the
        // door stack fills the cell to the right vertically.
        let mut tiles = [Tile::default(); ROOM_WIDTH * ROOM_HEIGHT];
        // Bottom row, col 4 — door pieces in col 5.
        tiles[2 * ROOM_WIDTH + 4] = Tile {
            kind: TileKind::Exit,
            variant: 0,
            modifier: 0,
        };
        let level = synth_level_with(tiles);
        let tables = BiomeTables::load(&vendor_root(), Biome::Dungeon).unwrap();
        // Pick a room id different from the bundled prince start so
        // the stairs piece also draws (and we get the door + stairs
        // combo this test cares about).
        let start = level.prince_start().screen;
        let render_room_id = if start == 1 { 2 } else { 1 };
        let frame = render_room(&level, render_room_id, &tables, RenderMode::Monochrome).unwrap();
        // Bottom row Ay = 188; door stack spans roughly y = 124..174.
        // Probe the middle of that band, cell to right of the exit
        // (col 5 → x = 140..168).
        assert!(
            has_non_black(&frame, 140, 130, 168, 170),
            "closed exit door should fill the cell to the right"
        );
    }

    #[test]
    fn exit_stairs_present_outside_start_room_absent_inside() {
        // The `drawexitb` stairs piece (`STAIRS`) at `(blockxco + 1,
        // Ay − 12)` is drawn only when the current room is NOT the
        // prince's start (`FRAMEADV.S:1635`). Render the same exit
        // tile as both a start and a non-start room and confirm the
        // stairs band differs.
        let mut tiles = [Tile::default(); ROOM_WIDTH * ROOM_HEIGHT];
        tiles[2 * ROOM_WIDTH + 4] = Tile {
            kind: TileKind::Exit,
            variant: 0,
            modifier: 0,
        };
        let level = synth_level_with(tiles);
        let tables = BiomeTables::load(&vendor_root(), Biome::Dungeon).unwrap();
        // Derive the start room from the level header rather than
        // hard-coding it — that way an unrelated vendor / INFO header
        // change doesn't break this test silently.
        let start = level.prince_start().screen;
        let non_start = if start == 1 { 2 } else { 1 };
        let with_stairs = render_room(&level, non_start, &tables, RenderMode::Monochrome).unwrap();
        let without_stairs = render_room(&level, start, &tables, RenderMode::Monochrome).unwrap();
        // Whole-frame inequality: the start-room render skips the
        // stairs sprite, so some pixels must differ between the two.
        // Pinning a tighter pixel rect would couple the test to the
        // stairs sprite's exact shape, and a lit-pixel count would be
        // fragile too — `Opacity::Sta` *overwrites* canvas bytes, so
        // the stairs branch can clear previously-lit pixels just as
        // easily as it adds new ones.
        assert_ne!(
            with_stairs.pixels, without_stairs.pixels,
            "stairs flag should produce a visibly different render",
        );
    }
}
