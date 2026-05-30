//! POP level binaries: parse the 2304-byte level files bundled under
//! `vendor/pop-apple2/04 Support/Levels/LEVEL0..LEVEL14`.
//!
//! # Format
//!
//! Each level file is exactly [`LEVEL_FILE_SIZE`] bytes, laid out as
//! four contiguous sections:
//!
//! | Range          | Bytes | Contents                                  |
//! |----------------|------:|-------------------------------------------|
//! | `0..720`       |   720 | per-tile primary byte (24 rooms × 30)     |
//! | `720..1440`    |   720 | per-tile modifier byte (24 rooms × 30)    |
//! | `1440..1536`   |    96 | room links (24 rooms × 4 neighbours)      |
//! | `1536..2304`   |   768 | metadata: start position, guards, etc.    |
//!
//! Rooms are numbered `0..24` in file order and laid out as a 10 × 3
//! tile grid (10 columns, 3 rows; row 0 is the top).
//!
//! Each primary tile byte packs the tile kind into its low 5 bits and a
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
//! Tile kinds 0..29 are named per `vendor/pop-apple2/01 POP Source/Source/BGDATA.S`
//! (`space, floor, spikes, posts, gate, …, archtop4`). Modifier bytes
//! carry tile-state at level start (gate openness, potion subtype,
//! pressure-plate target, etc.) — kept as a raw byte here; per-kind
//! interpretation lives with the future tile-interaction subsystem
//! (#96 if we go with the reimplementation path, otherwise the lifted
//! `FRAMEADV.S` / `COLL.S` semantics).
//!
//! Room links and metadata are exposed as raw bytes for now: the
//! 96-byte link section uses a packed encoding that we'll cross-
//! reference against the lifted `LoadLevelX` path in a follow-up; the
//! 768-byte metadata block carries the start position, guard placements,
//! and trailer padding documented for the PC port that we adapt to the
//! Apple II layout as we use them.

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

const TILE_SECTION_LEN: usize = ROOMS_PER_LEVEL * TILES_PER_ROOM; // 720
const MODIFIER_SECTION_END: usize = 2 * TILE_SECTION_LEN; // 1440
const LINK_BYTES_PER_ROOM: usize = 4;
const LINK_SECTION_END: usize = MODIFIER_SECTION_END + ROOMS_PER_LEVEL * LINK_BYTES_PER_ROOM; // 1536

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
        // Exhaustive match keeps the lookup safe under `forbid(unsafe_code)`
        // while staying branch-predictor-friendly.
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
    /// Per-tile modifier byte from the level's modifier section
    /// (offsets 720..1440). Kept as a raw byte; the kind-specific
    /// interpretation (gate state, potion subtype, plate target, …)
    /// lives with the future tile-interaction subsystem.
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

/// Raw 4-byte room-link record for a single room.
///
/// The link bytes use a packed encoding (visible neighbour-room id in
/// the low bits plus some flag bits we haven't decoded yet — bytes like
/// `0x89` show up routinely, and POP only has 24 rooms). The decoded
/// neighbour-id API lands when we cross-reference `LoadLevelX` and the
/// `Mark*` helpers in `CTRLSUBS.S` / `FRAMEADV.S` — for now callers get
/// the raw bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RoomLinks {
    /// Raw link bytes, ordered as POP stores them on disk.
    pub raw: [u8; LINK_BYTES_PER_ROOM],
}

/// A single 10×3 room: 30 tiles plus its link record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Room {
    /// Tiles in source order: index `r * ROOM_WIDTH + c` is row `r`,
    /// column `c`, with row 0 at the top.
    pub tiles: [Tile; TILES_PER_ROOM],
    /// Raw link bytes for this room.
    pub links: RoomLinks,
}

impl Default for Room {
    fn default() -> Self {
        Self {
            tiles: [Tile::default(); TILES_PER_ROOM],
            links: RoomLinks::default(),
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

/// A parsed POP level: 24 rooms plus the unparsed metadata trailer.
///
/// Construct via [`Level::from_bytes`] or [`Level::from_file`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Level {
    /// The 24 rooms in file order.
    pub rooms: [Room; ROOMS_PER_LEVEL],
    /// Bytes 1536..2304 of the source file — the metadata trailer
    /// (start position, guard placements, level number, padding). Kept
    /// as a raw slice for now; structured accessors arrive as we
    /// reverse-engineer each field. See [`Level::raw_metadata`].
    metadata: Vec<u8>,
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
            let tile_base = room_idx * TILES_PER_ROOM;
            let mod_base = TILE_SECTION_LEN + tile_base;
            let link_base = MODIFIER_SECTION_END + room_idx * LINK_BYTES_PER_ROOM;

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

            room.links.raw.copy_from_slice(&data[link_base..link_base + LINK_BYTES_PER_ROOM]);
        }

        let metadata = data[LINK_SECTION_END..LEVEL_FILE_SIZE].to_vec();

        Ok(Self { rooms, metadata })
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

    /// Unparsed metadata trailer (bytes 1536..2304 of the source file).
    ///
    /// Will grow structured accessors as we decode the per-field layout.
    #[must_use]
    pub fn raw_metadata(&self) -> &[u8] {
        &self.metadata
    }

    /// Tile-kind histogram over every room, useful for "does this
    /// parse look sane?" smoke checks. Indexed by `TileKind` as
    /// `u8`, length [`TileKind::COUNT`].
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bundled_level(n: u8) -> Vec<u8> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../../vendor/pop-apple2/04 Support/Levels/LEVEL{n}"));
        std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
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
            let data = bundled_level(n);
            let lv = Level::from_bytes(&data).unwrap_or_else(|e| panic!("LEVEL{n}: {e}"));
            assert_eq!(lv.rooms.len(), ROOMS_PER_LEVEL);
            assert_eq!(lv.raw_metadata().len(), LEVEL_FILE_SIZE - LINK_SECTION_END);
        }
    }

    #[test]
    fn tile_decoding_matches_known_room_zero_of_level_one() {
        let data = bundled_level(1);
        let lv = Level::from_bytes(&data).unwrap();
        let r0 = &lv.rooms[0];

        // Top-left corner is the void above the first floor — three
        // empty tiles in row 0 columns 0..3.
        for col in 0..3 {
            assert_eq!(r0.tile_at(col, 0).unwrap().kind, TileKind::Empty);
        }
        // The first non-empty cell on row 0 is a floor tile.
        assert_eq!(r0.tile_at(3, 0).unwrap().kind, TileKind::Floor);
        // Row 2 has a loose floor at column 6 (raw byte 0x0b — `loose`
        // tile id 11, variant 0). This is the diagnostic feature we
        // recognise to confirm the (variant << 5) | kind split.
        assert_eq!(r0.tile_at(6, 2).unwrap().kind, TileKind::LooseFloor);
        assert_eq!(r0.tile_at(6, 2).unwrap().variant, 0);
    }

    #[test]
    fn tile_decoding_split_is_low_5_bits() {
        // Raw 0x34 should decode as block (20) with variant 1.
        let data = bundled_level(1);
        let lv = Level::from_bytes(&data).unwrap();
        let block_cell = lv.rooms[0].tile_at(8, 0).unwrap();
        assert_eq!(block_cell.kind, TileKind::Block);
        assert_eq!(block_cell.variant, 1);
    }

    #[test]
    fn level_one_histogram_has_a_plausible_shape() {
        // Smoke check: floor, block, empty should each show up in the
        // dozens to hundreds across 24 rooms; very rare tiles (mirror,
        // sword, bones) might be 0 or 1 per level.
        let data = bundled_level(1);
        let lv = Level::from_bytes(&data).unwrap();
        let h = lv.tile_kind_histogram();

        let total: u32 = h.iter().sum();
        assert_eq!(total as usize, ROOMS_PER_LEVEL * TILES_PER_ROOM);
        assert!(h[TileKind::Floor as usize] > 10, "level 1 has too few floor tiles");
        assert!(h[TileKind::Block as usize] > 10, "level 1 has too few block tiles");
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
}
