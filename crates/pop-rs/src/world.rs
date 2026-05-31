//! Minimal world model and frame loop (Path B, issue #92).
//!
//! First milestone: a windowed host needs *something* to tick and draw.
//! [`World`] holds a loaded level and its biome sprites, tracks a current
//! room and a frame counter, and renders that room to a [`Frame`] via the
//! existing [`pop_assets::scene`] compositor. The mode machine starts at
//! [`Mode::Playing`]; the title-screen asset path lands in a follow-up.
//!
//! This is deliberately thin — no Prince, no physics yet. It exists to
//! prove the host pipeline (window → loop → input → render) end to end
//! before gameplay subsystems (#93+) hang off [`World::tick`].

use pop_assets::bgdata::{BLOCK_BOT_ROW, CELL_WIDTH_BYTES};
use pop_assets::draz::image_table::Image;
use pop_assets::hires::{Frame, RenderMode};
use pop_assets::level::{Level, ROOMS_PER_LEVEL, ROOM_WIDTH};
use pop_assets::scene::{self, BiomeTables};
use pop_assets::sprite;

use crate::backend::InputState;

/// Top-level game mode. Expands toward the full
/// `Title → Attract → Demo → Playing → Paused → GameOver → Win` machine
/// from #92; only `Title` / `Playing` exist today.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Title screen (asset blit). Not yet wired to a decoded image — see
    /// the title-decode follow-up. A directional/shift press advances to
    /// [`Mode::Playing`].
    Title,
    /// In a level, browsing rooms. No Prince / physics yet.
    Playing,
}

/// High-level game state for one loaded level.
pub struct World {
    level: Level,
    tables: BiomeTables,
    /// Current room, 1-based (`1..=ROOMS_PER_LEVEL`).
    room_id: u8,
    /// Monotonic frame counter, advanced once per [`World::tick`].
    frame: u64,
    /// Current top-level mode.
    mode: Mode,
    /// Previous frame's input, for key-press edge detection.
    prev: InputState,
    /// Standing-Prince sprite, drawn in the start room when present.
    /// Loaded by the host (CHTAB image 15); `None` keeps the bare scene.
    kid: Option<Image>,
}

impl World {
    /// Build a world from a parsed level and its biome sprite tables.
    /// Starts in [`Mode::Playing`] at the prince's start room.
    #[must_use]
    pub fn new(level: Level, tables: BiomeTables) -> Self {
        let room_id = clamp_start_room(level.prince_start().screen);
        Self {
            level,
            tables,
            room_id,
            frame: 0,
            mode: Mode::Playing,
            prev: InputState::default(),
            kid: None,
        }
    }

    /// Attach the standing-Prince sprite (CHTAB image 15), drawn in the
    /// level's start room. Chainable; the host loads the image and passes
    /// it in so the engine stays free of file I/O.
    #[must_use]
    pub fn with_kid(mut self, sprite: Image) -> Self {
        self.kid = Some(sprite);
        self
    }

    /// Current room (1-based).
    #[must_use]
    pub fn room_id(&self) -> u8 {
        self.room_id
    }

    /// Frames elapsed since construction.
    #[must_use]
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Current mode.
    #[must_use]
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Advance one logic frame given the latest input.
    ///
    /// PR1 behaviour: left / right step through the level's rooms on
    /// key-press edges, so the host's input → state → render path is
    /// observable. Real per-frame gameplay (kid update, guards, tiles)
    /// hangs off here in later subsystems (#93+).
    pub fn tick(&mut self, input: InputState) {
        self.frame = self.frame.wrapping_add(1);
        match self.mode {
            Mode::Title => {
                if pressed(input) && !pressed(self.prev) {
                    self.mode = Mode::Playing;
                }
            }
            Mode::Playing => {
                if input.right && !self.prev.right {
                    self.room_id = step_room(self.room_id, 1);
                }
                if input.left && !self.prev.left {
                    self.room_id = step_room(self.room_id, -1);
                }
            }
        }
        self.prev = input;
    }

    /// Render the current frame to RGBA: the room scene, plus the
    /// standing Prince overlaid when [`Self::with_kid`] supplied a sprite
    /// and the current room is the level's start room.
    ///
    /// `None` only if the current room id is out of range for the loaded
    /// level (shouldn't happen for a valid level).
    #[must_use]
    pub fn render(&self, mode: RenderMode) -> Option<Frame> {
        let mut frame = scene::render_room(&self.level, self.room_id, &self.tables, mode)?;
        if let Some(img) = &self.kid {
            let start = self.level.prince_start();
            if start.screen == self.room_id {
                let (x, y) = kid_pixel_pos(start.block, img.width_bytes, img.height);
                sprite::overlay(&mut frame, img, x, y, mode, faces_right(start.face_raw));
            }
        }
        Some(frame)
    }
}

/// `true` if any directional or shift key is held.
fn pressed(i: InputState) -> bool {
    i.left || i.right || i.up || i.down || i.shift
}

/// Clamp a raw `KidStartScrn` into the `1..=ROOMS_PER_LEVEL` room
/// invariant (mirrors the editor's spawn clamp). A 0 or out-of-range
/// start room would otherwise make [`World::render`] return `None` and
/// the host open a blank window.
fn clamp_start_room(screen: u8) -> u8 {
    let last = u8::try_from(ROOMS_PER_LEVEL).unwrap_or(1).max(1);
    screen.clamp(1, last)
}

/// Top-left pixel position for the standing kid at his raw INFO start
/// tile `block` (`col + row * ROOM_WIDTH`). The sprite is centred in its
/// tile column; `y` puts its bottom scan-line on that row's floor
/// (`BLOCK_BOT_ROW`).
///
/// This is the *spawn* point, drawn faithfully — on LV1 that's an empty
/// top-left ledge the engine immediately drops the kid from (`SUBS.S`
/// `:special1 → jumpseq stepfall`). A static sprite can't show the fall;
/// the real spawn-and-drop lands with the movement controller (#94).
fn kid_pixel_pos(block: u8, width_bytes: u8, height: u8) -> (i32, i32) {
    let cols = i32::try_from(ROOM_WIDTH).unwrap_or(10).max(1);
    let col = i32::from(block) % cols;
    let row = (usize::from(block) / ROOM_WIDTH).min(BLOCK_BOT_ROW.len() - 1);
    let cell_w = i32::from(CELL_WIDTH_BYTES) * 7; // 28 px per tile column
    let sprite_w = i32::from(width_bytes) * 7;
    let floor_y = i32::from(BLOCK_BOT_ROW[row]);
    let x = col * cell_w + (cell_w - sprite_w) / 2;
    let y = floor_y - i32::from(height) + 1;
    (x, y)
}

/// Whether to mirror the kid sprite so he faces right.
///
/// CHTAB character frames are stored facing **left** ("normal"); POP
/// mirrors them to face right (`CTRLSUBS.S:354-355`). The runtime
/// `CharFace` is `KidStartFace ^ $ff` (`SUBS.S:1516`), and a non-negative
/// `CharFace` (high bit clear) means facing right. So we flip when the
/// high bit of `face_raw ^ $ff` is clear. LV1 spawns with
/// `KidStartFace = $ff` → `CharFace = 0` → faces right → mirror.
fn faces_right(face_raw: u8) -> bool {
    (face_raw ^ 0xff) & 0x80 == 0
}

/// Step a 1-based `room` by `delta`, wrapping within
/// `1..=ROOMS_PER_LEVEL`.
fn step_room(room: u8, delta: i32) -> u8 {
    let count = i32::try_from(ROOMS_PER_LEVEL).unwrap_or(1).max(1);
    let zero_based = i32::from(room) - 1;
    let next = (zero_based + delta).rem_euclid(count) + 1;
    u8::try_from(next).unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_start_room_keeps_invariant() {
        let last = u8::try_from(ROOMS_PER_LEVEL).unwrap();
        assert_eq!(clamp_start_room(0), 1); // floor
        assert_eq!(clamp_start_room(1), 1);
        assert_eq!(clamp_start_room(last), last);
        assert_eq!(clamp_start_room(last + 1), last); // ceil
        assert_eq!(clamp_start_room(u8::MAX), last); // ceil
    }

    #[test]
    fn step_room_wraps_both_directions() {
        let last = u8::try_from(ROOMS_PER_LEVEL).unwrap();
        assert_eq!(step_room(1, -1), last);
        assert_eq!(step_room(last, 1), 1);
        assert_eq!(step_room(5, 1), 6);
        assert_eq!(step_room(5, -1), 4);
    }

    #[test]
    fn world_renders_level_one_start_room() {
        use pop_assets::bgdata::Biome;
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor/pop-apple2/04 Support");
        let level = Level::from_file(root.join("Levels").join("LEVEL1")).expect("LEVEL1 loads");
        let start = level.prince_start().screen.max(1);
        let tables = BiomeTables::load(&root, Biome::Dungeon).expect("dungeon tables load");
        let world = World::new(level, tables);
        assert_eq!(world.room_id(), start);

        let frame = world
            .render(RenderMode::NtscColor)
            .expect("start room renders");
        assert_eq!((frame.width, frame.height), (280, 192));
        let lit = frame
            .pixels
            .chunks_exact(4)
            .filter(|p| p[0..3] != [0, 0, 0])
            .count();
        assert!(lit > 0, "rendered start room should have non-black pixels");
    }

    #[test]
    fn kid_pixel_pos_centers_in_column_and_sits_on_floor() {
        // 14 px (2-byte) wide × 41 px tall standing sprite.
        // block 0 → col 0, row 0; floor row 0 = 65.
        let (x, y) = kid_pixel_pos(0, 2, 41);
        assert_eq!(x, (28 - 14) / 2); // centred in the 28 px column
        assert_eq!(y, 65 - 41 + 1); // bottom scan-line on the floor
                                    // block 13 → col 3, row 1; floor row 1 = 128.
        let (x, y) = kid_pixel_pos(13, 2, 41);
        assert_eq!(x, 3 * 28 + (28 - 14) / 2);
        assert_eq!(y, 128 - 41 + 1);
    }

    #[test]
    fn faces_right_decodes_kidstartface() {
        // LV1: KidStartFace $ff → CharFace 0 → faces right → mirror.
        assert!(faces_right(0xff));
        // CharFace -1 (face_raw $00) → faces left → no mirror.
        assert!(!faces_right(0x00));
        // High bit of CharFace decides: $7f^$ff=$80 (left), $80^$ff=$7f (right).
        assert!(!faces_right(0x7f));
        assert!(faces_right(0x80));
    }

    #[test]
    fn kid_overlay_adds_pixels_in_start_room() {
        use pop_assets::bgdata::Biome;
        use pop_assets::draz::image_table::ImageTable;
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor/pop-apple2/04 Support");
        let level = Level::from_file(root.join("Levels").join("LEVEL1")).expect("LEVEL1 loads");
        let tables = BiomeTables::load(&root, Biome::Dungeon).expect("dungeon tables load");
        let chtab = ImageTable::from_file(root.join("DRAZ").join("I").join("IMG.CHTAB1"))
            .expect("CHTAB1 loads");
        // Image 15 (FRAMEDEF `stand`) = 0-based index 14.
        let kid = chtab.images.get(14).expect("stand frame").clone();

        let world = World::new(level, tables);
        let plain = world.render(RenderMode::Monochrome).expect("renders");
        let world = world.with_kid(kid);
        let kidded = world.render(RenderMode::Monochrome).expect("renders");

        assert_ne!(
            plain.pixels, kidded.pixels,
            "kid overlay should change the frame"
        );
        let lit = |f: &Frame| {
            f.pixels
                .chunks_exact(4)
                .filter(|p| p[0..3] != [0, 0, 0])
                .count()
        };
        assert!(lit(&kidded) > lit(&plain), "kid overlay adds lit pixels");
    }

    #[test]
    fn right_arrow_edge_advances_room() {
        use pop_assets::bgdata::Biome;
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor/pop-apple2/04 Support");
        let level = Level::from_file(root.join("Levels").join("LEVEL1")).expect("LEVEL1 loads");
        let tables = BiomeTables::load(&root, Biome::Dungeon).expect("dungeon tables load");
        let mut world = World::new(level, tables);
        let start = world.room_id();

        let held = InputState {
            right: true,
            ..InputState::default()
        };
        // First tick on a press edge advances exactly once...
        world.tick(held);
        let after = world.room_id();
        assert_eq!(after, step_room(start, 1));
        // ...and holding it without release does not advance again.
        world.tick(held);
        assert_eq!(world.room_id(), after);
    }
}
