//! Per-biome tile composition tables, ported from the 6502 source.
//!
//! Apple II POP's background renderer (`FRAMEADV.S` / `BGDATA.S`)
//! composes each room from small **piece sprites** — slices of floors,
//! walls, gates, posts, etc. — taken from two indexed
//! [`crate::draz::ImageTable`]s per biome (e.g. `IMG.BGTAB.DUN1` and
//! `IMG.BGTAB.DUN2`). Per [`crate::level::TileKind`] the engine knows
//! which sprite IDs to fetch and how to layer them; that mapping is
//! the data in this module.
//!
//! # Sprite ID convention
//!
//! Sprite-id byte values are interpreted per `GRAFIX.S:828` —
//!
//! ```text
//!   id == 0             → no piece (skip)
//!   id & 0x80 == 0      → index (id - 1) of BGTABLE1 (`IMG.BGTAB.{biome}1`)
//!   id & 0x80 != 0      → index ((id & 0x7F) - 1) of BGTABLE2 (`IMG.BGTAB.{biome}2`)
//! ```
//!
//! See [`PieceRef::resolve`] for the helper.
//!
//! # Layout
//!
//! Each piece array is indexed by [`crate::level::TileKind`] (so
//! `PIECE_A[TileKind::Floor as usize]` etc.). Comments at array head
//! match the original `BGDATA.S` annotation lines.
//!
//! Sections per `BGDATA.S:38-41`:
//! * A & B sit at `(blockxco, BlockBot - 3)` — the "wall/object" strip.
//! * C & D sit at `(blockxco, BlockBot)`     — the "floor" strip.
//! * Front pieces ride at `(blockxco + frontx, BlockBot - 3 + fronty)`.

use crate::level::TileKind;

/// 30 entries — one per [`TileKind`].
pub const TILE_KIND_COUNT: usize = 30;

const _: () = assert!(TILE_KIND_COUNT == TileKind::COUNT as usize);

// ---------------------------------------------------------------------------
// Per-tile piece tables — copied verbatim from BGDATA.S (vendor source).
// ---------------------------------------------------------------------------

/// A-section mask sprite, AND'd into the canvas before [`PIECE_A`].
pub const MASK_A: [u8; TILE_KIND_COUNT] = [
    0x00, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x00, 0x03, 0x03, 0x00, 0x03, 0x03, 0x03,
    0x03, 0x00, 0x00, 0x03, 0x00, 0x03, 0x00, 0x03, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00,
];

/// A-section piece sprite, OR'd into the canvas after [`MASK_A`].
pub const PIECE_A: [u8; TILE_KIND_COUNT] = [
    0x00, 0x01, 0x05, 0x07, 0x0a, 0x01, 0x01, 0x0a, 0x10, 0x00, 0x01, 0x00, 0x00, 0x14, 0x20, 0x4b,
    0x01, 0x00, 0x00, 0x01, 0x00, 0x97, 0x00, 0x01, 0x00, 0xa7, 0xa9, 0xaa, 0xac, 0xad,
];

/// Y offset (signed pixels) applied to the A-section sprite position.
pub const PIECE_A_Y: [i8; TILE_KIND_COUNT] = [
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -4, -4,
];

/// B-section mask. Drawn into the CURRENT cell using the
/// **left-neighbour**'s [`TileKind`], not current's.
pub const MASK_B: [u8; TILE_KIND_COUNT] = [
    0x00, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00, 0x04, 0x04, 0x04,
    0x00, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00, 0x00, 0x00, 0x00,
];

/// B-section piece. Same neighbour-rule as [`MASK_B`].
pub const PIECE_B: [u8; TILE_KIND_COUNT] = [
    0x00, 0x02, 0x06, 0x08, 0x0b, 0x1b, 0x02, 0x9e, 0x1a, 0x1c, 0x02, 0x00, 0x9e, 0x4a, 0x21, 0x1b,
    0x4d, 0x4e, 0x02, 0x51, 0x84, 0x98, 0x02, 0x91, 0x92, 0x02, 0x00, 0x00, 0x00, 0x00,
];

/// Signed Y offset for the B-section sprite.
pub const PIECE_B_Y: [i8; TILE_KIND_COUNT] = [
    0, 0, 0, 0, 0, 1, 0, 3, 0, 3, 0, 0, 3, 0, 0, -1, 0, 0, 0, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Palace-only "B-section stripe" overlay, added below [`PIECE_B`] at
/// `Ay - 32`. Indexed by **left-neighbour**'s [`TileKind`].
pub const B_STRIPE: [u8; TILE_KIND_COUNT] = [
    0x00, 0x47, 0x47, 0x00, 0x00, 0x47, 0x47, 0x00, 0x00, 0x00, 0x47, 0x47, 0x00, 0x00, 0x47, 0x47,
    0x00, 0x00, 0x47, 0x00, 0x00, 0x00, 0x47, 0x00, 0x00, 0x47, 0x00, 0x00, 0x00, 0x00,
];

/// C-section piece. Drawn into the CURRENT cell from the **below-left
/// neighbour**'s [`TileKind`].
pub const PIECE_C: [u8; TILE_KIND_COUNT] = [
    0x00, 0x00, 0x00, 0x09, 0x0c, 0x00, 0x00, 0x9f, 0x00, 0x1d, 0x00, 0x00, 0x9f, 0x00, 0x00, 0x00,
    0x4f, 0x50, 0x00, 0x00, 0x85, 0x00, 0x00, 0x93, 0x94, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// D-section piece. Drawn into the CURRENT cell from the CURRENT
/// tile's own [`TileKind`].
pub const PIECE_D: [u8; TILE_KIND_COUNT] = [
    0x00, 0x15, 0x15, 0x15, 0x15, 0x18, 0x19, 0x16, 0x15, 0x00, 0x15, 0x00, 0x17, 0x15, 0x2e, 0x4c,
    0x15, 0x15, 0x15, 0x15, 0x86, 0x15, 0x15, 0x15, 0x15, 0x15, 0xab, 0x00, 0x00, 0x00,
];

/// Front (post-layer) sprite. Drawn on top of everything in current
/// cell — used for posts, archtops, torches, blocks.
pub const FRONT_I: [u8; TILE_KIND_COUNT] = [
    0x00, 0x00, 0x00, 0x45, 0x46, 0x00, 0x00, 0x46, 0x48, 0x49, 0x87, 0x00, 0x46, 0x0f, 0x13, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x83, 0x00, 0x00, 0x00, 0x00, 0xa8, 0x00, 0xae, 0xae, 0xae,
];

/// Signed Y offset for [`FRONT_I`].
pub const FRONT_Y: [i8; TILE_KIND_COUNT] = [
    0, 0, 0, -1, 0, 0, 0, 0, -1, 3, -3, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -36, -36,
    -36,
];

/// X offset (hires bytes; always non-negative) for [`FRONT_I`].
pub const FRONT_X: [u8; TILE_KIND_COUNT] = [
    0x00, 0x00, 0x00, 0x01, 0x03, 0x00, 0x00, 0x03, 0x01, 0x01, 0x02, 0x00, 0x03, 0x01, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
];

// ---------------------------------------------------------------------------
// Back-wall panels for space / floor (BGDATA.S:175-179).
// ---------------------------------------------------------------------------

/// Back-wall variants drawn behind a `space` tile, indexed by
/// `state & 0b11` of the left-neighbour's space tile (one of 4).
pub const SPACE_B: [u8; 4] = [0x00, 0xa3, 0xa5, 0xa6];

/// Y offsets for [`SPACE_B`].
pub const SPACE_B_Y: [i8; 4] = [0, -20, -20, 0];

/// Back-wall variants for a `floor` tile, indexed by `state & 0b11`.
pub const FLOOR_B: [u8; 4] = [0x02, 0xa2, 0xa4, 0xa4];

/// Y offsets for [`FLOOR_B`].
pub const FLOOR_B_Y: [i8; 4] = [0, 0, 0, 0];

// ---------------------------------------------------------------------------
// Solid block variants (BGDATA.S:184-189).
// ---------------------------------------------------------------------------

/// Block B-piece variants, indexed by `state mod 2`.
pub const BLOCK_B: [u8; 2] = [0x84, 0x6f];
/// Block C-piece variants.
pub const BLOCK_C: [u8; 2] = [0x85, 0x85];
/// Block D-piece variants.
pub const BLOCK_D: [u8; 2] = [0x86, 0x86];
/// Block front-piece variants.
pub const BLOCK_FR: [u8; 2] = [0x83, 0x83];

// ---------------------------------------------------------------------------
// Panel-with-floor / -without-floor variants (BGDATA.S:161-166).
// ---------------------------------------------------------------------------

/// Panel B-piece variants, indexed by panel state (0..numpans=3).
pub const PANEL_B: [u8; 3] = [0x9e, 0x9a, 0x81];
/// Panel C-piece variants.
pub const PANEL_C: [u8; 3] = [0x9f, 0x9b, 0x82];
/// Sentinel for "default" panel B-piece value, per `BGDATA.S:161`.
pub const PANEL_B0_SENTINEL: u8 = 0x9e;
/// Sentinel for "default" panel C-piece value.
pub const PANEL_C0_SENTINEL: u8 = 0x9f;

// ---------------------------------------------------------------------------
// Loose floor (BGDATA.S:140-148) — frames cycle through the animation.
// ---------------------------------------------------------------------------

/// Per-frame A-piece for the loose-floor animation. Frame 0 is at rest.
pub const LOOSE_A: [u8; 11] = [
    0x01, 0x1e, 0x01, 0x1f, 0x1f, 0x01, 0x01, 0x01, 0x1f, 0x1f, 0x1f,
];
/// Per-frame B-piece Y offset.
pub const LOOSE_B_Y: [i8; 11] = [0, 1, 0, -1, -1, 0, 0, 0, -1, -1, -1];
/// Per-frame D-piece.
pub const LOOSE_D: [u8; 11] = [
    0x15, 0x2c, 0x15, 0x2d, 0x2d, 0x15, 0x15, 0x15, 0x2d, 0x2d, 0x2d,
];
/// Constant B-piece for loose floor.
pub const LOOSE_B: u8 = 0x1b;

// ---------------------------------------------------------------------------
// Torch flame frames (`GAMEBG.S:147`).
// ---------------------------------------------------------------------------

/// Per-frame sprite ID for the torch flame, drawn by `SETUPFLAME` in
/// the cell **to the right** of a torch tile, offset by
/// `(+1 hires byte, -43 px)` from `Ay`. Frame 0 is the at-rest flame.
pub const TORCH_FLAME: [u8; 12] = [
    0x52, 0x53, 0x54, 0x55, 0x56, 0x61, 0x62, 0x63, 0x64, 0x52, 0x54, 0x56,
];

// ---------------------------------------------------------------------------
// Gate pieces (`BGDATA.S:89-95`).
// ---------------------------------------------------------------------------

/// Gate bottom piece, drawn with `sta` opacity when the bottom sits
/// above the floor line.
pub const GATE_BOT_STA: u8 = 0x43;
/// Gate bottom piece, drawn with `ora` opacity when the bottom sits
/// at / below the floor line and needs to overlay the background.
pub const GATE_BOT_ORA: u8 = 0x44;
/// Gate middle "grill" piece (8 pixels tall). Stacked vertically by
/// `drawgateb` to fill the gate's full height.
pub const GATE_B1: u8 = 0x37;
/// Mask AND'd into the cell above-right when the cell below-left
/// holds a gate (`drawgatec`).
pub const GATE_C_MASK: u8 = 0x0d;
/// Eight C-section sprites of varying heights — `drawgatec`
/// indexes by `(state/4) mod 8`.
pub const GATE_8C: [u8; 8] = [0x2f, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36];
/// Eight B-section top-of-grill sprites (1-8 pixels tall), used as
/// the very top piece of `drawgateb`.
pub const GATE_8B: [u8; 8] = [0x3e, 0x3d, 0x3c, 0x3b, 0x3a, 0x39, 0x38, 0x37];

// ---------------------------------------------------------------------------
// Exit door pieces (`BGDATA.S:107-110`).
// ---------------------------------------------------------------------------

/// Staircase sprite drawn under an exit door — skipped on the prince's
/// start room (treated as the entrance, no stairs leaving).
pub const STAIRS: u8 = 0x6b;
/// One vertical slice of the closed exit door; `drawexitb` stacks
/// these from `Ay − 14` downward in 4-pixel steps.
pub const DOOR: u8 = 0x6c;
/// Mask AND'd into the canvas immediately before each [`DOOR`] piece.
pub const DOOR_MASK: u8 = 0x6d;
/// "Top repair" sprite drawn above the door stack so the wall edge
/// reads cleanly. Lives in the cell's C-section by Y coord but is
/// emitted from `drawexitb`'s tail (`FRAMEADV.S:1680`).
pub const TOP_REPAIR: u8 = 0x6e;

// ---------------------------------------------------------------------------
// Per-row layout constants (TABLES.S:33-42).
// ---------------------------------------------------------------------------

/// Bottom-of-screen Y coord, in display pixels (0 = top, 191 = bottom).
pub const SCRN_BOT: u8 = 191;
/// Each room row is this many pixels tall.
pub const BLOCK_HEIGHT: u8 = 63;
/// Floorpiece (D-section) thickness in pixels.
pub const D_HEIGHT: u8 = 3;
/// Width of one cell, in hires bytes (1 byte = 7 mono pixels).
pub const CELL_WIDTH_BYTES: u8 = 4;
/// Width of the whole room canvas, in hires bytes (10 columns * 4).
pub const ROOM_WIDTH_BYTES: u8 = 40;
/// Height of the whole room canvas, in pixels.
pub const ROOM_HEIGHT_PX: u8 = 192;

/// `BlockBot[row]` for the three on-screen rows. Index 0 = top row,
/// 2 = bottom row. Y axis matches the screen (0 = top, 191 = bottom).
pub const BLOCK_BOT_ROW: [u8; 3] = [
    SCRN_BOT - 2 * BLOCK_HEIGHT, // 65   — bottom of top row's D-section
    SCRN_BOT - BLOCK_HEIGHT,     // 128  — middle row
    SCRN_BOT,                    // 191  — bottom row
];

// ---------------------------------------------------------------------------
// Biome selection.
// ---------------------------------------------------------------------------

/// Which BGTAB set a level draws from. Mirrors the three values of
/// `MISC.S:772 bgset1` — the Apple II port ships exactly three biomes;
/// `IMG.BGTAB.{TWR1,TWR2}` in the vendor tree are unused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Biome {
    /// Levels 0-3 — `IMG.BGTAB.DUN1` + `IMG.BGTAB.DUN2`.
    Dungeon,
    /// Levels 4-6, 10-11, 14 — `IMG.BGTAB.PAL1` + `IMG.BGTAB.PAL2`.
    Palace,
    /// Levels 7-9, 12-13 — `IMG.BGTAB.RED1` + `IMG.BGTAB.RED2`.
    Red,
}

/// Per-level biome, indexed by **0-based** level number (LEVEL0..LEVEL14).
/// Verbatim from `MISC.S:772 bgset1` — `bgset2` is identical in that
/// source, so we only carry one copy.
pub const LEVEL_BIOME: [Biome; 15] = [
    Biome::Dungeon, // LEVEL0
    Biome::Dungeon, // LEVEL1
    Biome::Dungeon, // LEVEL2
    Biome::Dungeon, // LEVEL3
    Biome::Palace,  // LEVEL4
    Biome::Palace,  // LEVEL5
    Biome::Palace,  // LEVEL6
    Biome::Red,     // LEVEL7
    Biome::Red,     // LEVEL8
    Biome::Red,     // LEVEL9
    Biome::Palace,  // LEVEL10
    Biome::Palace,  // LEVEL11
    Biome::Red,     // LEVEL12
    Biome::Red,     // LEVEL13
    Biome::Palace,  // LEVEL14
];

impl Biome {
    /// Default biome for `level_index` (0-based, 0..15).
    /// Returns `None` for indices outside the bundled-level range.
    #[must_use]
    pub fn for_level(level_index: usize) -> Option<Self> {
        LEVEL_BIOME.get(level_index).copied()
    }

    /// `(table1, table2)` filenames inside a POP data root's
    /// `DRAZ/IP/` directory.
    #[must_use]
    pub fn bgtab_filenames(self) -> (&'static str, &'static str) {
        match self {
            Self::Dungeon => ("IMG.BGTAB.DUN1", "IMG.BGTAB.DUN2"),
            Self::Palace => ("IMG.BGTAB.PAL1", "IMG.BGTAB.PAL2"),
            Self::Red => ("IMG.BGTAB.RED1", "IMG.BGTAB.RED2"),
        }
    }

    /// Short ASCII name (`"DUN"`, `"PAL"`, `"RED"`), used in UI.
    #[must_use]
    pub const fn short_name(self) -> &'static str {
        match self {
            Self::Dungeon => "DUN",
            Self::Palace => "PAL",
            Self::Red => "RED",
        }
    }
}

/// Resolved reference into a biome's two BGTAB image tables.
///
/// Produced by [`PieceRef::resolve`] from a raw sprite-ID byte; encodes
/// the [`GRAFIX.S:828`](https://github.com/jmechner/Prince-of-Persia-Apple-II) bit-7 convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieceRef {
    /// Index into `IMG.BGTAB.{biome}1` (`0..126`).
    Table1(u8),
    /// Index into `IMG.BGTAB.{biome}2` (`0..126`).
    Table2(u8),
}

impl PieceRef {
    /// Decode a raw sprite-ID byte. Returns `None` for IDs whose
    /// 7-bit index is `0` — that's the "no piece" sentinel used
    /// throughout `BGDATA.S`, in either table half (`0x00` and
    /// `0x80`).
    #[must_use]
    pub const fn resolve(id: u8) -> Option<Self> {
        let index = id & 0x7F;
        if index == 0 {
            return None;
        }
        if id & 0x80 == 0 {
            Some(Self::Table1(index - 1))
        } else {
            Some(Self::Table2(index - 1))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piece_arrays_are_sized_one_per_tile_kind() {
        // Catches future additions to TileKind that forget to extend
        // the BGDATA tables.
        assert_eq!(MASK_A.len(), TILE_KIND_COUNT);
        assert_eq!(PIECE_A.len(), TILE_KIND_COUNT);
        assert_eq!(PIECE_A_Y.len(), TILE_KIND_COUNT);
        assert_eq!(MASK_B.len(), TILE_KIND_COUNT);
        assert_eq!(PIECE_B.len(), TILE_KIND_COUNT);
        assert_eq!(PIECE_B_Y.len(), TILE_KIND_COUNT);
        assert_eq!(B_STRIPE.len(), TILE_KIND_COUNT);
        assert_eq!(PIECE_C.len(), TILE_KIND_COUNT);
        assert_eq!(PIECE_D.len(), TILE_KIND_COUNT);
        assert_eq!(FRONT_I.len(), TILE_KIND_COUNT);
        assert_eq!(FRONT_Y.len(), TILE_KIND_COUNT);
        assert_eq!(FRONT_X.len(), TILE_KIND_COUNT);
    }

    #[test]
    fn piece_ref_resolves_table_select_bit() {
        assert_eq!(PieceRef::resolve(0), None);
        assert_eq!(PieceRef::resolve(1), Some(PieceRef::Table1(0)));
        assert_eq!(PieceRef::resolve(0x7E), Some(PieceRef::Table1(0x7D)));
        // Bit 7 set → table 2.
        assert_eq!(PieceRef::resolve(0x80 | 1), Some(PieceRef::Table2(0)));
        assert_eq!(PieceRef::resolve(0xa7), Some(PieceRef::Table2(0x26)));
        // `0x80` is bit-7-set with a zero 7-bit index — treat as
        // "no piece", same as raw `0`. Pre-fix this underflowed
        // `(id & 0x7F) - 1` in debug builds.
        assert_eq!(PieceRef::resolve(0x80), None);
    }

    #[test]
    fn biome_for_level_matches_bgset1() {
        // Spot checks of the per-level biome assignment.
        assert_eq!(Biome::for_level(0), Some(Biome::Dungeon));
        assert_eq!(Biome::for_level(3), Some(Biome::Dungeon));
        assert_eq!(Biome::for_level(4), Some(Biome::Palace));
        assert_eq!(Biome::for_level(7), Some(Biome::Red));
        assert_eq!(Biome::for_level(14), Some(Biome::Palace));
        assert_eq!(Biome::for_level(15), None);
    }

    #[test]
    fn biome_filenames_match_vendor_tree() {
        let (t1, t2) = Biome::Dungeon.bgtab_filenames();
        assert_eq!(t1, "IMG.BGTAB.DUN1");
        assert_eq!(t2, "IMG.BGTAB.DUN2");
        assert_eq!(Biome::Palace.bgtab_filenames().0, "IMG.BGTAB.PAL1");
        assert_eq!(Biome::Red.bgtab_filenames().0, "IMG.BGTAB.RED1");
    }

    #[test]
    fn block_bot_row_matches_table_constants() {
        // Sanity-check the row baselines against the originals
        // (TABLES.S:33-42 with BlockHeight=63, ScrnBot=191).
        assert_eq!(BLOCK_BOT_ROW[0], 65);
        assert_eq!(BLOCK_BOT_ROW[1], 128);
        assert_eq!(BLOCK_BOT_ROW[2], 191);
    }
}
