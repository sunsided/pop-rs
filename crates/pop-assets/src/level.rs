//! POP level binaries: parse the 2304-byte level files bundled under
//! `vendor/pop-apple2/04 Support/Levels/LEVEL0..LEVEL14`.
//!
//! # Format
//!
//! Each level file is exactly [`LEVEL_FILE_SIZE`] bytes, laid out as
//! six contiguous sections that mirror the `blueprnt` dummy section in
//! `vendor/pop-apple2/01 POP Source/Source/EQ.S:258-265`:
//!
//! | Range          | Bytes | Symbol     | Contents                                  |
//! |----------------|------:|------------|-------------------------------------------|
//! | `0..720`       |   720 | `BLUETYPE` | per-tile primary byte (24 rooms × 30)     |
//! | `720..1440`    |   720 | `BLUESPEC` | per-tile modifier byte (24 rooms × 30)    |
//! | `1440..1696`   |   256 | `LINKLOC`  | door/event source table (indexed by `BLUESPEC` byte of a door tile) |
//! | `1696..1952`   |   256 | `LINKMAP`  | door/event state table                     |
//! | `1952..2048`   |    96 | `MAP`      | per-room neighbour graph (24 rooms × 4 directions) |
//! | `2048..2304`   |   256 | `INFO`     | start positions, guard placements, header  |
//!
//! Rooms are numbered `1..=24` in POP itself (1-based — `MAP-4,x`,
//! `MAP-3,x`, … in `CTRLSUBS.S:GETLEFT/GETRIGHT/GETUP/GETDOWN` index
//! with `room * 4`). The `0` value in any neighbour slot means "no
//! room there — edge of the level". This module exposes rooms as a
//! 0-indexed `[Room; 24]` array; the parser writes file order
//! verbatim, and [`Level::room_links`] returns the neighbour graph
//! using 1-based room ids.
//!
//! Each `BLUETYPE` byte packs the tile kind into its low 5 bits and a
//! 3-bit *variant* into its high 3 bits:
//!
//! ```text
//! +-+-+-+-+-+-+-+-+
//! |v v v|k k k k k|
//! +-+-+-+-+-+-+-+-+
//!  ^^^^^ ^^^^^^^^^
//!  variant   kind
//! ```
//!
//! Tile kinds 0..29 are named per
//! `vendor/pop-apple2/01 POP Source/Source/BGDATA.S`. The `BLUESPEC`
//! modifier byte carries tile-state at level start (gate openness,
//! potion subtype, pressure-plate `LINKLOC` index, etc.) — kept as a
//! raw byte here; per-kind interpretation lives with the future
//! tile-interaction subsystem.
//!
//! # `LINKLOC` / `LINKMAP`
//!
//! These two 256-byte tables hold door / gate / pressure-plate event
//! data: a tile's `BLUESPEC` byte is a `LINKLOC` index for door-like
//! tiles, `LINKLOC` then indexes `LINKMAP`, which holds the actual
//! target-tile encoding. Exposed as raw byte slices for now — typed
//! `LinkEvent { source_room, source_tile, target_room, target_tile, … }`
//! decoding lands in a follow-up that cross-references `MOVER.S:430-510`
//! and `FRAMEADV.S:2069-2090`.
//!
//! # `INFO` substructure
//!
//! The 256-byte `INFO` section (per `EQ.S:272-287`):
//!
//! | INFO offset | Bytes | Symbol           | Contents                              |
//! |------------:|------:|------------------|---------------------------------------|
//! | `0`         |    64 | (unnamed)        | per-room header data — two 24-entry arrays + padding; not yet fully reverse-engineered, exposed as raw bytes |
//! | `64`        |     1 | `KidStartScrn`   | Prince start room (1-based)           |
//! | `65`        |     1 | `KidStartBlock`  | Prince start tile (`col + row*10`)    |
//! | `66`        |     1 | `KidStartFace`   | Prince start facing (raw; XOR'd with `$ff` into `CharFace` at game-load — see `SUBS.S:1516`) |
//! | `67`        |     1 | (padding)        |                                       |
//! | `68`        |     1 | `SwStartScrn`    | Sword start room (`0` = no sword)     |
//! | `69`        |     1 | `SwStartBlock`   | Sword start tile                      |
//! | `70`        |     1 | (padding)        |                                       |
//! | `71..95`    |    24 | `GdStartBlock`   | Guard start tile per room (`30` = no guard)   |
//! | `95..119`   |    24 | `GdStartFace`    | Guard start facing per room           |
//! | `119..143`  |    24 | `GdStartX`       | Guard sub-tile X per room             |
//! | `143..167`  |    24 | `GdStartSeqL`    | Guard animation seq pointer (lo) per room |
//! | `167..191`  |    24 | `GdStartProg`    | Guard "program" / skill per room      |
//! | `191..215`  |    24 | `GdStartSeqH`    | Guard animation seq pointer (hi) per room |
//! | `215..256`  |    41 | (padding)        |                                       |
//!
//! `INFO[0]` is `number-of-screens + 1` per the loop in `SUBS.S:1442`
//! (see [`Level::screen_count`]).

use std::path::Path;

use thiserror::Error;

/// Rooms per level. Constant across all POP levels.
pub const ROOMS_PER_LEVEL: usize = 24;

/// Tiles per room: a 10×3 grid.
pub const TILES_PER_ROOM: usize = 30;

/// Room grid width in tiles.
pub const ROOM_WIDTH: usize = 10;

/// Room grid height in tiles.
pub const ROOM_HEIGHT: usize = 3;

/// Total bytes in a level file.
pub const LEVEL_FILE_SIZE: usize = 2304;

// --- Section offsets, mirroring EQ.S:258-265 ----------------------------

const BLUETYPE_OFFSET: usize = 0;
const BLUESPEC_OFFSET: usize = 720;
const LINKLOC_OFFSET: usize = 1440;
const LINKMAP_OFFSET: usize = 1696;
const MAP_OFFSET: usize = 1952;
const INFO_OFFSET: usize = 2048;

const TILE_SECTION_LEN: usize = ROOMS_PER_LEVEL * TILES_PER_ROOM; // 720
const LINKLOC_LEN: usize = 256;
const LINKMAP_LEN: usize = 256;
const MAP_BYTES_PER_ROOM: usize = 4;
const MAP_LEN: usize = ROOMS_PER_LEVEL * MAP_BYTES_PER_ROOM; // 96
const INFO_LEN: usize = 256;

const _: () = assert!(BLUESPEC_OFFSET == BLUETYPE_OFFSET + TILE_SECTION_LEN);
const _: () = assert!(LINKLOC_OFFSET == BLUESPEC_OFFSET + TILE_SECTION_LEN);
const _: () = assert!(LINKMAP_OFFSET == LINKLOC_OFFSET + LINKLOC_LEN);
const _: () = assert!(MAP_OFFSET == LINKMAP_OFFSET + LINKMAP_LEN);
const _: () = assert!(INFO_OFFSET == MAP_OFFSET + MAP_LEN);
const _: () = assert!(LEVEL_FILE_SIZE == INFO_OFFSET + INFO_LEN);
const _: () = assert!(ROOM_WIDTH <= u8::MAX as usize);
const _: () = assert!(ROOM_HEIGHT <= u8::MAX as usize);
const _: () = assert!(TILES_PER_ROOM <= u8::MAX as usize);
#[allow(clippy::cast_possible_truncation)]
const ROOM_WIDTH_U8: u8 = ROOM_WIDTH as u8;
#[allow(clippy::cast_possible_truncation)]
const TILES_PER_ROOM_U8: u8 = TILES_PER_ROOM as u8;

// --- INFO sub-offsets, mirroring EQ.S:272-287 ---------------------------

/// Length of the unnamed header at the start of `INFO` (two 24-entry
/// per-room arrays + 16 bytes of padding, per the table in this
/// module's documentation).
pub const INFO_HEADER_LEN: usize = 64;
const INFO_KID_SCRN: usize = 64;
const INFO_KID_BLOCK: usize = 65;
const INFO_KID_FACE: usize = 66;
const INFO_SW_SCRN: usize = 68;
const INFO_SW_BLOCK: usize = 69;
const INFO_GUARD_BLOCK: usize = 71;
const INFO_GUARD_FACE: usize = 95;
const INFO_GUARD_X: usize = 119;
const INFO_GUARD_SEQ_LO: usize = 143;
const INFO_GUARD_PROG: usize = 167;
const INFO_GUARD_SEQ_HI: usize = 191;

/// Sentinel value in `GdStartBlock` (`INFO + 71 + room_idx`) that means
/// "no guard in this room" — `30` is one past the highest valid tile
/// index (`TILES_PER_ROOM - 1 == 29`).
pub const GUARD_BLOCK_NONE: u8 = TILES_PER_ROOM_U8;

/// Errors returned from [`Level::from_bytes`] and [`Level::from_file`].
#[derive(Debug, Error)]
pub enum ParseError {
    /// The buffer wasn't exactly [`LEVEL_FILE_SIZE`] bytes.
    #[error("level file must be {LEVEL_FILE_SIZE} bytes, got {0}")]
    WrongSize(usize),
    /// I/O failure while reading the file from disk.
    #[error("reading level file: {0}")]
    Io(#[from] std::io::Error),
}

/// One of POP's 30 tile kinds, named per `BGDATA.S`.
///
/// Bytes 0..=29 of the level's tile section map onto this enum's
/// discriminants directly via [`TileKind::from_raw`]. Higher bits of the
/// raw byte become the [`Tile::variant`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
#[allow(missing_docs)] // self-documenting per BGDATA.S
pub enum TileKind {
    /// Empty / blue background — no tile here.
    Empty = 0,
    Floor = 1,
    Spikes = 2,
    /// Vertical pillar segment — used for posts / columns.
    Posts = 3,
    Gate = 4,
    /// Down-acting pressure plate (closes the gate it points at).
    DownPressPlate = 5,
    /// Up-acting pressure plate (opens the gate it points at).
    PressPlate = 6,
    /// Wall panel with a floor below.
    PanelWithFloor = 7,
    PillarBottom = 8,
    PillarTop = 9,
    /// Potion flask.
    Flask = 10,
    LooseFloor = 11,
    /// Wall panel without a floor below.
    PanelWithoutFloor = 12,
    Mirror = 13,
    Rubble = 14,
    /// "U" pressure plate (used in palace-style levels).
    UPressPlate = 15,
    Exit = 16,
    Exit2 = 17,
    /// Chomper / slicer trap.
    Slicer = 18,
    Torch = 19,
    /// Wall block — the most common solid-wall tile.
    Block = 20,
    Bones = 21,
    Sword = 22,
    Window = 23,
    Window2 = 24,
    ArchBottom = 25,
    ArchTop1 = 26,
    ArchTop2 = 27,
    ArchTop3 = 28,
    ArchTop4 = 29,
}

impl TileKind {
    /// Number of distinct tile kinds, i.e. one past the largest valid
    /// discriminant. Used as the `tile_id` upper bound when decoding.
    pub const COUNT: u8 = 30;

    /// Decode a 5-bit tile id. `None` for values outside `0..30`
    /// (which can only appear if the input is corrupt — POP itself never
    /// produces an out-of-range tile id).
    #[must_use]
    pub fn from_raw(tile_id: u8) -> Option<Self> {
        Some(match tile_id {
            0 => Self::Empty,
            1 => Self::Floor,
            2 => Self::Spikes,
            3 => Self::Posts,
            4 => Self::Gate,
            5 => Self::DownPressPlate,
            6 => Self::PressPlate,
            7 => Self::PanelWithFloor,
            8 => Self::PillarBottom,
            9 => Self::PillarTop,
            10 => Self::Flask,
            11 => Self::LooseFloor,
            12 => Self::PanelWithoutFloor,
            13 => Self::Mirror,
            14 => Self::Rubble,
            15 => Self::UPressPlate,
            16 => Self::Exit,
            17 => Self::Exit2,
            18 => Self::Slicer,
            19 => Self::Torch,
            20 => Self::Block,
            21 => Self::Bones,
            22 => Self::Sword,
            23 => Self::Window,
            24 => Self::Window2,
            25 => Self::ArchBottom,
            26 => Self::ArchTop1,
            27 => Self::ArchTop2,
            28 => Self::ArchTop3,
            29 => Self::ArchTop4,
            _ => return None,
        })
    }

    /// Stable short name suitable for diagnostics and tile-histogram output.
    #[must_use]
    pub const fn short_name(self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::Floor => "floor",
            Self::Spikes => "spikes",
            Self::Posts => "posts",
            Self::Gate => "gate",
            Self::DownPressPlate => "dpressplate",
            Self::PressPlate => "pressplate",
            Self::PanelWithFloor => "panelwif",
            Self::PillarBottom => "pillarbot",
            Self::PillarTop => "pillartop",
            Self::Flask => "flask",
            Self::LooseFloor => "loose",
            Self::PanelWithoutFloor => "panelwof",
            Self::Mirror => "mirror",
            Self::Rubble => "rubble",
            Self::UPressPlate => "upressplate",
            Self::Exit => "exit",
            Self::Exit2 => "exit2",
            Self::Slicer => "slicer",
            Self::Torch => "torch",
            Self::Block => "block",
            Self::Bones => "bones",
            Self::Sword => "sword",
            Self::Window => "window",
            Self::Window2 => "window2",
            Self::ArchBottom => "archbot",
            Self::ArchTop1 => "archtop1",
            Self::ArchTop2 => "archtop2",
            Self::ArchTop3 => "archtop3",
            Self::ArchTop4 => "archtop4",
        }
    }
}

/// A single tile cell in a room: kind + variant + modifier byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    /// Tile kind (decoded from the low 5 bits of the primary byte).
    pub kind: TileKind,
    /// Tile variant (top 3 bits of the primary byte). Meaning is
    /// kind-specific — typically a graphic / behaviour sub-selection.
    pub variant: u8,
    /// Per-tile modifier byte from the `BLUESPEC` section. For door /
    /// gate / pressure-plate kinds this is a `LINKLOC` index; for
    /// loose-floor / potion / etc. it carries kind-specific state.
    /// Decoded shape lands with the tile-interaction subsystem.
    pub modifier: u8,
}

impl Default for Tile {
    fn default() -> Self {
        Self {
            kind: TileKind::Empty,
            variant: 0,
            modifier: 0,
        }
    }
}

/// A single 10×3 room: 30 tiles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Room {
    /// Tiles in source order: index `r * ROOM_WIDTH + c` is row `r`,
    /// column `c`, with row 0 at the top.
    pub tiles: [Tile; TILES_PER_ROOM],
}

impl Default for Room {
    fn default() -> Self {
        Self {
            tiles: [Tile::default(); TILES_PER_ROOM],
        }
    }
}

impl Room {
    /// Tile at `(col, row)` with `col < ROOM_WIDTH` and `row < ROOM_HEIGHT`.
    #[must_use]
    pub fn tile_at(&self, col: usize, row: usize) -> Option<&Tile> {
        if col < ROOM_WIDTH && row < ROOM_HEIGHT {
            Some(&self.tiles[row * ROOM_WIDTH + col])
        } else {
            None
        }
    }
}

/// One room's four neighbours, in the order `GETLEFT/RIGHT/UP/DOWN`
/// in `CTRLSUBS.S` reads them from the `MAP` table.
///
/// Each field is a **1-based** room id (`1..=24`), or `0` for "no room
/// there — edge of the level".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RoomNeighbours {
    /// Room id reached by walking left out of this one (1-based; 0 = edge).
    pub left: u8,
    /// Room id reached by walking right out of this one.
    pub right: u8,
    /// Room id reached by exiting through the top.
    pub up: u8,
    /// Room id reached by falling out of the bottom.
    pub down: u8,
}

/// A character's start position: which room, which tile within it, and
/// the raw facing byte from `INFO`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StartPosition {
    /// 1-based room id (`KidStartScrn` / `SwStartScrn` / etc.).
    pub screen: u8,
    /// Tile index within the room (`col + row * ROOM_WIDTH`). Use
    /// [`Self::col_row`] to split.
    pub block: u8,
    /// Raw `*StartFace` byte from `INFO`. The game-load path XORs this
    /// with `$ff` before storing into `CharFace` (`SUBS.S:1516`); the
    /// raw byte is preserved here so a future `Face` enum decoding can
    /// be added without changing the on-disk view.
    pub face_raw: u8,
}

impl StartPosition {
    /// Split [`Self::block`] into `(col, row)` with `col < ROOM_WIDTH`,
    /// `row < ROOM_HEIGHT`. Returns `None` if `block >= TILES_PER_ROOM`
    /// (which is also how the sword-start sentinel is detected).
    #[must_use]
    pub fn col_row(&self) -> Option<(u8, u8)> {
        if self.block >= TILES_PER_ROOM_U8 {
            return None;
        }
        Some((self.block % ROOM_WIDTH_U8, self.block / ROOM_WIDTH_U8))
    }
}

/// One guard's per-room spawn record from the `GdStart*` arrays in `INFO`.
///
/// `None` in [`Level::guard_spawns`] for any room whose
/// `GdStartBlock` is [`GUARD_BLOCK_NONE`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuardSpawn {
    /// Tile index within the room (`col + row * ROOM_WIDTH`). Always
    /// less than [`TILES_PER_ROOM`] when the spawn is present.
    pub block: u8,
    /// Raw `GdStartFace` byte.
    pub face_raw: u8,
    /// Sub-tile X offset (`GdStartX`) — pixel-level horizontal nudge
    /// applied at spawn.
    pub x: u8,
    /// Low byte of the animation sequence pointer (`GdStartSeqL`).
    pub seq_lo: u8,
    /// High byte of the animation sequence pointer (`GdStartSeqH`).
    pub seq_hi: u8,
    /// Guard "program" / skill level (`GdStartProg`).
    pub prog: u8,
}

impl GuardSpawn {
    /// Split [`Self::block`] into `(col, row)`.
    #[must_use]
    pub fn col_row(&self) -> Option<(u8, u8)> {
        if self.block >= TILES_PER_ROOM_U8 {
            return None;
        }
        Some((self.block % ROOM_WIDTH_U8, self.block / ROOM_WIDTH_U8))
    }
}

/// A parsed POP level: 24 rooms plus the structured INFO / link
/// sections from the on-disk blueprint.
///
/// Construct via [`Level::from_bytes`] or [`Level::from_file`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Level {
    /// The 24 rooms in file order. Index `i` corresponds to POP's
    /// 1-based room id `i + 1`.
    pub rooms: [Room; ROOMS_PER_LEVEL],
    /// Raw 256-byte `LINKLOC` table — door/event source records.
    /// Indexed by the modifier byte of a door-like tile. Decoded into
    /// typed `LinkEvent`s in a follow-up.
    link_loc: Box<[u8; LINKLOC_LEN]>,
    /// Raw 256-byte `LINKMAP` table — door/event state records.
    link_map: Box<[u8; LINKMAP_LEN]>,
    /// 24 × 4 room-neighbour graph from the `MAP` section. Pre-decoded
    /// into 1-based room ids (`0` = edge).
    neighbours: [RoomNeighbours; ROOMS_PER_LEVEL],
    /// Raw 256-byte `INFO` section. Structured access via
    /// [`Self::prince_start`], [`Self::sword_start`],
    /// [`Self::guard_spawns`].
    info: Box<[u8; INFO_LEN]>,
}

impl Level {
    /// Parse a level from a 2304-byte buffer.
    ///
    /// # Errors
    ///
    /// Returns [`ParseError::WrongSize`] if the input isn't exactly
    /// [`LEVEL_FILE_SIZE`] bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, ParseError> {
        if data.len() != LEVEL_FILE_SIZE {
            return Err(ParseError::WrongSize(data.len()));
        }

        let mut rooms = std::array::from_fn(|_| Room::default());
        for (room_idx, room) in rooms.iter_mut().enumerate() {
            let tile_base = BLUETYPE_OFFSET + room_idx * TILES_PER_ROOM;
            let mod_base = BLUESPEC_OFFSET + room_idx * TILES_PER_ROOM;

            for tile_idx in 0..TILES_PER_ROOM {
                let raw = data[tile_base + tile_idx];
                let kind = TileKind::from_raw(raw & 0x1f).unwrap_or(TileKind::Empty);
                let variant = raw >> 5;
                let modifier = data[mod_base + tile_idx];
                room.tiles[tile_idx] = Tile {
                    kind,
                    variant,
                    modifier,
                };
            }
        }

        let mut link_loc = Box::new([0u8; LINKLOC_LEN]);
        link_loc.copy_from_slice(&data[LINKLOC_OFFSET..LINKLOC_OFFSET + LINKLOC_LEN]);
        let mut link_map = Box::new([0u8; LINKMAP_LEN]);
        link_map.copy_from_slice(&data[LINKMAP_OFFSET..LINKMAP_OFFSET + LINKMAP_LEN]);
        let mut info = Box::new([0u8; INFO_LEN]);
        info.copy_from_slice(&data[INFO_OFFSET..INFO_OFFSET + INFO_LEN]);

        let mut neighbours = [RoomNeighbours::default(); ROOMS_PER_LEVEL];
        for (room_idx, slot) in neighbours.iter_mut().enumerate() {
            let base = MAP_OFFSET + room_idx * MAP_BYTES_PER_ROOM;
            *slot = RoomNeighbours {
                left: data[base],
                right: data[base + 1],
                up: data[base + 2],
                down: data[base + 3],
            };
        }

        Ok(Self {
            rooms,
            link_loc,
            link_map,
            neighbours,
            info,
        })
    }

    /// Read and parse a level file from disk.
    ///
    /// # Errors
    ///
    /// Returns [`ParseError::Io`] on I/O failure and
    /// [`ParseError::WrongSize`] if the file isn't [`LEVEL_FILE_SIZE`]
    /// bytes.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ParseError> {
        let data = std::fs::read(path)?;
        Self::from_bytes(&data)
    }

    /// Tile-kind histogram over every room.
    #[must_use]
    pub fn tile_kind_histogram(&self) -> [u32; TileKind::COUNT as usize] {
        let mut h = [0u32; TileKind::COUNT as usize];
        for room in &self.rooms {
            for tile in &room.tiles {
                h[tile.kind as usize] += 1;
            }
        }
        h
    }

    /// Per-room neighbour graph, indexed 0..24 (room `i + 1` in POP's
    /// 1-based numbering).
    #[must_use]
    pub fn room_links(&self) -> &[RoomNeighbours; ROOMS_PER_LEVEL] {
        &self.neighbours
    }

    /// Prince spawn record from `INFO.KidStartScrn/Block/Face`.
    #[must_use]
    pub fn prince_start(&self) -> StartPosition {
        StartPosition {
            screen: self.info[INFO_KID_SCRN],
            block: self.info[INFO_KID_BLOCK],
            face_raw: self.info[INFO_KID_FACE],
        }
    }

    /// Sword spawn record, or `None` if `SwStartScrn` is `0` (no
    /// sword on this level). Levels 1, 3, 4, 6, etc. ship a sword; the
    /// demo / cutscene levels don't.
    #[must_use]
    pub fn sword_start(&self) -> Option<StartPosition> {
        let screen = self.info[INFO_SW_SCRN];
        if screen == 0 {
            return None;
        }
        Some(StartPosition {
            screen,
            block: self.info[INFO_SW_BLOCK],
            // No SwStartFace byte; reuse the padding slot as 0 so the
            // shape matches `prince_start`.
            face_raw: 0,
        })
    }

    /// Per-room guard spawn, `None` for rooms whose `GdStartBlock` is
    /// [`GUARD_BLOCK_NONE`]. Indexed 0..24.
    #[must_use]
    pub fn guard_spawns(&self) -> [Option<GuardSpawn>; ROOMS_PER_LEVEL] {
        let mut out: [Option<GuardSpawn>; ROOMS_PER_LEVEL] = [None; ROOMS_PER_LEVEL];
        for (room_idx, slot) in out.iter_mut().enumerate() {
            let block = self.info[INFO_GUARD_BLOCK + room_idx];
            if block == GUARD_BLOCK_NONE {
                continue;
            }
            *slot = Some(GuardSpawn {
                block,
                face_raw: self.info[INFO_GUARD_FACE + room_idx],
                x: self.info[INFO_GUARD_X + room_idx],
                seq_lo: self.info[INFO_GUARD_SEQ_LO + room_idx],
                seq_hi: self.info[INFO_GUARD_SEQ_HI + room_idx],
                prog: self.info[INFO_GUARD_PROG + room_idx],
            });
        }
        out
    }

    /// `INFO[0]` — POP stores `(rooms used) + 1` here per the
    /// `SETINITIALS` loop in `SUBS.S:1441-1450`.
    #[must_use]
    pub fn screen_count_plus_one(&self) -> u8 {
        self.info[0]
    }

    /// First 64 bytes of `INFO` — two 24-entry per-room arrays plus 16
    /// bytes of padding. Field names not yet known; exposed raw so the
    /// editor can display / experiment. Always exactly
    /// [`INFO_HEADER_LEN`] (= 64) bytes.
    #[must_use]
    pub fn info_header(&self) -> &[u8] {
        &self.info[..INFO_HEADER_LEN]
    }

    /// Raw 256-byte `INFO` section.
    #[must_use]
    pub fn raw_info(&self) -> &[u8; INFO_LEN] {
        &self.info
    }

    /// Raw 256-byte `LINKLOC` table.
    #[must_use]
    pub fn link_loc(&self) -> &[u8; LINKLOC_LEN] {
        &self.link_loc
    }

    /// Raw 256-byte `LINKMAP` table.
    #[must_use]
    pub fn link_map(&self) -> &[u8; LINKMAP_LEN] {
        &self.link_map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bundled_level(n: u8) -> Vec<u8> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../vendor/pop-apple2/04 Support/Levels/LEVEL{n}"
        ));
        std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
    }

    fn level(n: u8) -> Level {
        Level::from_bytes(&bundled_level(n)).unwrap_or_else(|e| panic!("LEVEL{n}: {e}"))
    }

    #[test]
    fn wrong_size_rejected() {
        assert!(matches!(
            Level::from_bytes(&[0u8; 100]),
            Err(ParseError::WrongSize(100))
        ));
        assert!(matches!(
            Level::from_bytes(&[0u8; LEVEL_FILE_SIZE - 1]),
            Err(ParseError::WrongSize(_))
        ));
    }

    #[test]
    fn parses_every_bundled_level() {
        for n in 0u8..=14 {
            let lv = level(n);
            assert_eq!(lv.rooms.len(), ROOMS_PER_LEVEL);
            assert_eq!(lv.link_loc().len(), LINKLOC_LEN);
            assert_eq!(lv.link_map().len(), LINKMAP_LEN);
            assert_eq!(lv.raw_info().len(), INFO_LEN);
            assert_eq!(lv.room_links().len(), ROOMS_PER_LEVEL);
        }
    }

    #[test]
    fn tile_decoding_matches_known_room_zero_of_level_one() {
        let lv = level(1);
        let r0 = &lv.rooms[0];
        for col in 0..3 {
            assert_eq!(r0.tile_at(col, 0).unwrap().kind, TileKind::Empty);
        }
        assert_eq!(r0.tile_at(3, 0).unwrap().kind, TileKind::Floor);
        assert_eq!(r0.tile_at(6, 2).unwrap().kind, TileKind::LooseFloor);
        assert_eq!(r0.tile_at(6, 2).unwrap().variant, 0);
    }

    #[test]
    fn tile_decoding_split_is_low_5_bits() {
        let lv = level(1);
        let block_cell = lv.rooms[0].tile_at(8, 0).unwrap();
        assert_eq!(block_cell.kind, TileKind::Block);
        assert_eq!(block_cell.variant, 1);
    }

    #[test]
    fn level_one_histogram_has_a_plausible_shape() {
        let lv = level(1);
        let h = lv.tile_kind_histogram();
        let total: u32 = h.iter().sum();
        assert_eq!(total as usize, ROOMS_PER_LEVEL * TILES_PER_ROOM);
        assert!(h[TileKind::Floor as usize] > 10);
        assert!(h[TileKind::Block as usize] > 10);
        assert!(h[TileKind::Empty as usize] > 0);
    }

    #[test]
    fn from_file_round_trip() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor/pop-apple2/04 Support/Levels/LEVEL1");
        let from_file = Level::from_file(&path).unwrap();
        let from_bytes = Level::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(from_file, from_bytes);
    }

    #[test]
    fn tile_kind_round_trip_for_all_valid_ids() {
        for id in 0..TileKind::COUNT {
            let kind = TileKind::from_raw(id).expect("kind in range");
            assert_eq!(kind as u8, id);
        }
        assert!(TileKind::from_raw(TileKind::COUNT).is_none());
        assert!(TileKind::from_raw(255).is_none());
    }

    // --- Section-split tests --------------------------------------------

    #[test]
    fn level_one_prince_start_is_room_one() {
        // LEVEL1 hex at INFO+64..70: 01 00 ff 00 00 00 — KidStartScrn=1,
        // KidStartBlock=0, KidStartFace=0xff, SwStart{Scrn,Block}=0.
        let lv = level(1);
        let start = lv.prince_start();
        assert_eq!(start.screen, 1);
        assert_eq!(start.block, 0);
        assert_eq!(start.face_raw, 0xff);
        let (col, row) = start.col_row().unwrap();
        assert_eq!((col, row), (0, 0));
    }

    #[test]
    fn level_one_room_links_match_canonical_neighbours() {
        // From LEVEL1 MAP (offset 1952): room 1 = `05 00 00 02` —
        // left=5, right=0, up=0, down=2.
        let lv = level(1);
        let n = lv.room_links();
        assert_eq!(
            n[0],
            RoomNeighbours {
                left: 5,
                right: 0,
                up: 0,
                down: 2,
            }
        );
        // Room 4 in LEVEL1 = `13 0e 14 00` — left=19, right=14, up=20.
        assert_eq!(
            n[3],
            RoomNeighbours {
                left: 0x13,
                right: 0x0e,
                up: 0x14,
                down: 0,
            }
        );
        // All neighbour ids fit in 0..=24 (0 = edge).
        for (i, room_n) in n.iter().enumerate() {
            for (dir, v) in [
                ("left", room_n.left),
                ("right", room_n.right),
                ("up", room_n.up),
                ("down", room_n.down),
            ] {
                assert!(
                    usize::from(v) <= ROOMS_PER_LEVEL,
                    "room {} {dir} = {v} out of range",
                    i + 1
                );
            }
        }
    }

    #[test]
    fn level_one_has_guards_in_rooms_three_and_twenty_one() {
        // LEVEL1 GdStartBlock dump (INFO+71+0..23):
        //   1e 1e 11 1e 1e 1e 1e 1e 1e 1e 1e 1e 1e 1e 1e 1e
        //   1e 1e 1e 1e 06 1e 1e 1e
        // → guards in rooms 3 (block 0x11) and 21 (block 0x06).
        let lv = level(1);
        let guards = lv.guard_spawns();
        let present: Vec<usize> = guards
            .iter()
            .enumerate()
            .filter_map(|(i, g)| g.map(|_| i + 1))
            .collect();
        assert_eq!(present, vec![3, 21]);
        assert_eq!(guards[2].unwrap().block, 0x11);
        assert_eq!(guards[20].unwrap().block, 0x06);
    }

    #[test]
    fn level_three_has_no_guards_in_guard_table() {
        // LEVEL3 GdStartBlock is all 0x1e — skeleton levels don't use
        // the guard table the same way; documenting that we correctly
        // detect "no guards" rather than mis-decoding the sentinel.
        let lv = level(3);
        assert!(lv.guard_spawns().iter().all(Option::is_none));
    }

    #[test]
    fn sword_start_zero_screen_means_no_sword() {
        // LEVEL1 SwStartScrn = 0 in the on-disk bytes; absence of a
        // sword pickup must surface as None even though SwStartBlock
        // happens to be 0 too.
        let lv = level(1);
        assert_eq!(lv.sword_start(), None);
    }

    #[test]
    fn every_level_screen_count_is_at_least_one() {
        // INFO[0] is "rooms used + 1" — a level with at least one
        // playable room has INFO[0] >= 2.
        for n in 0u8..=14 {
            let lv = level(n);
            assert!(
                lv.screen_count_plus_one() >= 1,
                "LEVEL{n} INFO[0] = {}",
                lv.screen_count_plus_one()
            );
        }
    }
}
