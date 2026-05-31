//! World model, the player character, and the frame loop (Path B,
//! issues #92 / #94).
//!
//! [`World`] holds a loaded level + its biome sprites, a [`Mode`] machine,
//! the current room, and the [`Prince`]. `render` composites the room
//! scene ([`pop_assets::scene`]) and overlays the Prince on top.
//!
//! The Prince uses a clean pixel-space physics model (not POP's
//! `CharX`/`CharY` fixed-point coordinates) — Path B reads the original
//! for *behaviour* and reuses its *art*, but the runtime is fresh Rust.
//! Today's physics is just spawn + gravity: LV1 drops the kid in from the
//! top-left ledge (`SUBS.S :special1 → jumpseq stepfall`) and he falls to
//! the floor. The controller (run / turn / jump / climb) and the rest of
//! the per-frame subsystem order land next (#94 / #95).

use pop_assets::bgdata::{BLOCK_BOT_ROW, CELL_WIDTH_BYTES, ROOM_HEIGHT_PX, ROOM_WIDTH_BYTES};
use pop_assets::draz::image_table::Image;
use pop_assets::hires::{self, Frame, RenderMode};
use pop_assets::level::{Level, Room, TileKind, ROOMS_PER_LEVEL, ROOM_HEIGHT, ROOM_WIDTH};
use pop_assets::scene::{self, Anim, BiomeTables};
use pop_assets::sprite;

use crate::backend::InputState;

/// Downward acceleration applied each logic tick while the Prince is
/// airborne, in pixels per tick. Tuned for the ~12.5 Hz host step so the
/// LV1 drop-in reads as a quick fall; refined against POP's real fall
/// timing when the controller lands (#94).
const GRAVITY: i32 = 3;

/// Top-level game mode. Expands toward the full
/// `Title → Attract → Demo → Playing → Paused → GameOver → Win` machine
/// from #92; only `Title` / `Playing` exist today.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Title screen (asset blit). Not yet wired to a decoded image — see
    /// the title-decode follow-up. A directional/shift press advances to
    /// [`Mode::Playing`].
    Title,
    /// In a level. No combat / traps yet.
    Playing,
}

/// The kid sprites the host loads: the standing pose, the free-fall pose,
/// and the 8-frame run cycle. A full FRAMEDEF-driven frame set (turn /
/// jump / climb) and a sequence interpreter arrive with the rest of the
/// animation engine (#94 / #95).
pub struct KidArt {
    /// `stand` frame (FRAMEDEF 15 → CHTAB1 image 15).
    pub stand: Image,
    /// `freefall` frame (FRAMEDEF 106 → CHTAB2 image 54).
    pub fall: Image,
    /// The 8-frame run cycle (SEQTABLE `runcyc1..8` = FRAMEDEF 7-14 →
    /// CHTAB1 images 7-14), paired one-to-one with [`RUN_CHX`].
    pub run: Vec<Image>,
}

/// Per-frame horizontal step of the run cycle, in pixels — the `chx`
/// operands of SEQTABLE `runcyc1..8`. Indexed by the run phase.
const RUN_CHX: [i32; 8] = [5, 1, 2, 4, 5, 2, 3, 4];

/// Horizontal pixel bounds that keep the Prince on screen. No tile-level
/// wall / ledge collision yet (that's the next slice) — he just can't
/// walk off the 280 px room edge.
const X_MIN: i32 = 7;
const X_MAX: i32 = 273;

/// The player character's physical state in pixel space.
struct Prince {
    /// Room the kid is in (1-based). He's only drawn while this is the
    /// room on screen.
    room: u8,
    /// Horizontal centre, room pixels.
    x: i32,
    /// Feet line, room pixels — the sprite's bottom scan-line sits here.
    feet_y: i32,
    /// Vertical velocity, pixels per tick (down is positive).
    vy: i32,
    /// Floor line the kid lands on when the fall completes.
    landing_y: i32,
    /// `true` once he has landed.
    on_ground: bool,
    /// Facing right (sprite mirrored) vs left.
    facing_right: bool,
    /// Phase into the 8-frame run cycle ([`RUN_CHX`] / `KidArt::run`).
    run_phase: usize,
    /// `true` while he is stepping this tick (drives run vs stand frame).
    moving: bool,
}

impl Prince {
    /// Place the Prince at the level's INFO spawn and decide whether he
    /// starts grounded or has to fall to the floor below.
    fn spawn(level: &Level) -> Self {
        let start = level.prince_start();
        let cols = ROOM_WIDTH.max(1);
        let col = usize::from(start.block) % cols;
        let row = (usize::from(start.block) / cols).min(ROOM_HEIGHT - 1);
        let cell_w = i32::from(CELL_WIDTH_BYTES) * 7; // 28 px per tile column
        let x = i32::try_from(col).unwrap_or(0) * cell_w + cell_w / 2;
        let spawn_feet = i32::from(BLOCK_BOT_ROW[row]);
        let room = clamp_start_room(start.screen);
        let landing_y = level
            .rooms
            .get(usize::from(room) - 1)
            .map_or(spawn_feet, |r| settle_feet_y(r, col, row));
        Prince {
            room,
            x,
            feet_y: spawn_feet,
            vy: 0,
            landing_y,
            on_ground: spawn_feet >= landing_y,
            facing_right: faces_right(start.face_raw),
            run_phase: 0,
            moving: false,
        }
    }

    /// Advance one tick of gravity until he lands on `landing_y`.
    fn fall(&mut self) {
        self.vy += GRAVITY;
        self.feet_y += self.vy;
        if self.feet_y >= self.landing_y {
            self.feet_y = self.landing_y;
            self.vy = 0;
            self.on_ground = true;
        }
    }

    /// Run one tick in `dir` (`-1` left, `+1` right, `0` idle): face the
    /// way he moves, advance the run cycle, and step by that frame's
    /// `chx`, clamped to the room. `dir == 0` returns him to standing.
    fn walk(&mut self, dir: i32) {
        if dir == 0 {
            self.moving = false;
            self.run_phase = 0;
            return;
        }
        self.facing_right = dir > 0;
        self.run_phase = (self.run_phase + 1) % RUN_CHX.len();
        self.x = (self.x + RUN_CHX[self.run_phase] * dir).clamp(X_MIN, X_MAX);
        self.moving = true;
    }
}

/// High-level game state for one loaded level.
pub struct World {
    level: Level,
    tables: BiomeTables,
    /// Current room on screen, 1-based (`1..=ROOMS_PER_LEVEL`).
    room_id: u8,
    /// Monotonic frame counter, advanced once per [`World::tick`].
    frame: u64,
    /// Current top-level mode.
    mode: Mode,
    /// Previous frame's input, for key-press edge detection.
    prev: InputState,
    /// The player character.
    prince: Prince,
    /// Kid sprites, host-loaded; `None` renders the bare scene.
    art: Option<KidArt>,
}

impl World {
    /// Build a world from a parsed level and its biome sprite tables.
    /// Starts in [`Mode::Playing`] showing (and spawning the Prince in)
    /// his start room.
    #[must_use]
    pub fn new(level: Level, tables: BiomeTables) -> Self {
        let prince = Prince::spawn(&level);
        Self {
            room_id: prince.room,
            level,
            tables,
            frame: 0,
            mode: Mode::Playing,
            prev: InputState::default(),
            prince,
            art: None,
        }
    }

    /// Attach the kid sprite set (the host loads it so the engine stays
    /// free of file I/O). Chainable.
    #[must_use]
    pub fn with_kid_art(mut self, art: KidArt) -> Self {
        self.art = Some(art);
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

    /// `true` once the Prince has landed on the floor.
    #[must_use]
    pub fn prince_on_ground(&self) -> bool {
        self.prince.on_ground
    }

    /// Advance one logic frame given the latest input.
    ///
    /// In `Playing`: while airborne the Prince falls under gravity (no
    /// steering mid-air yet); once grounded, the left / right arrows run
    /// him that way. Guards / tiles / sound slot in here in order as later
    /// subsystems land (#94+).
    pub fn tick(&mut self, input: InputState) {
        self.frame = self.frame.wrapping_add(1);
        match self.mode {
            Mode::Title => {
                if pressed(input) && !pressed(self.prev) {
                    self.mode = Mode::Playing;
                }
            }
            Mode::Playing => {
                if self.prince.on_ground {
                    self.prince.walk(walk_dir(input));
                } else {
                    self.prince.fall();
                }
                // The room on screen follows the Prince.
                self.room_id = self.prince.room;
            }
        }
        self.prev = input;
    }

    /// Render the current frame to RGBA: the room scene plus the Prince,
    /// drawn when the room on screen is the one he's in.
    ///
    /// `None` only if the current room id is out of range for the loaded
    /// level (shouldn't happen for a valid level).
    #[must_use]
    pub fn render(&self, mode: RenderMode) -> Option<Frame> {
        // Composite the Prince into the room's hi-res *byte* buffer before
        // NTSC decode — that keeps his artifact colours tied to his true
        // screen column (an RGBA overlay would swap orange/blue on mirror).
        let mut bytes =
            scene::compose_room_bytes(&self.level, self.room_id, &self.tables, Anim::REST)?;
        if let Some(art) = &self.art {
            if self.room_id == self.prince.room {
                let img = if !self.prince.on_ground {
                    &art.fall
                } else if self.prince.moving {
                    art.run.get(self.prince.run_phase).unwrap_or(&art.stand)
                } else {
                    &art.stand
                };
                // Centre the sprite's byte span on the Prince's column;
                // sit its bottom scan-line on his feet.
                let byte_x = self.prince.x / 7 - i32::from(img.width_bytes) / 2;
                let top_y = self.prince.feet_y - i32::from(img.height) + 1;
                sprite::composite_hires(
                    &mut bytes[..],
                    usize::from(ROOM_WIDTH_BYTES),
                    usize::from(ROOM_HEIGHT_PX),
                    img,
                    byte_x,
                    top_y,
                    self.prince.facing_right,
                );
            }
        }
        hires::render_linear_topdown(&bytes[..], ROOM_WIDTH_BYTES, ROOM_HEIGHT_PX, mode)
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

/// Pixel row the kid's feet rest on when dropped straight down from
/// (`col`, `start_row`): the floor of the first floor-bearing tile below
/// him, or the top of the first solid block. Falls back to the room's
/// bottom floor if the column is open all the way down.
fn settle_feet_y(room: &Room, col: usize, start_row: usize) -> i32 {
    for r in start_row..ROOM_HEIGHT {
        let Some(kind) = room.tile_at(col, r).map(|t| t.kind) else {
            continue;
        };
        if tile_has_floor(kind) {
            return i32::from(BLOCK_BOT_ROW[r]);
        }
        if tile_is_solid(kind) {
            // Stand on the block's top edge = the floor line of the row
            // above it.
            return i32::from(BLOCK_BOT_ROW[r.saturating_sub(1)]);
        }
    }
    i32::from(BLOCK_BOT_ROW[ROOM_HEIGHT - 1])
}

/// `true` for tiles the kid stands *on*, feet at that row's floor line.
fn tile_has_floor(kind: TileKind) -> bool {
    matches!(
        kind,
        TileKind::Floor
            | TileKind::Spikes
            | TileKind::LooseFloor
            | TileKind::Rubble
            | TileKind::DownPressPlate
            | TileKind::PressPlate
            | TileKind::UPressPlate
            | TileKind::PanelWithFloor
            | TileKind::Flask
            | TileKind::Sword
            | TileKind::Bones
            | TileKind::Slicer
            | TileKind::Gate
            | TileKind::Exit
            | TileKind::Exit2
    )
}

/// `true` for solid blocks the kid stands on *top* of.
fn tile_is_solid(kind: TileKind) -> bool {
    matches!(kind, TileKind::Block | TileKind::PillarBottom)
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

/// Run direction from input: `+1` right, `-1` left, `0` idle. Right wins
/// if both are held.
fn walk_dir(input: InputState) -> i32 {
    if input.right {
        1
    } else if input.left {
        -1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vendor_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor/pop-apple2/04 Support")
    }

    fn load_level1() -> Level {
        Level::from_file(vendor_root().join("Levels").join("LEVEL1")).expect("LEVEL1 loads")
    }

    fn dungeon_tables() -> BiomeTables {
        use pop_assets::bgdata::Biome;
        BiomeTables::load(&vendor_root(), Biome::Dungeon).expect("dungeon tables load")
    }

    fn kid_art() -> KidArt {
        use pop_assets::draz::image_table::ImageTable;
        let dir = vendor_root().join("DRAZ").join("I");
        let chtab1 = ImageTable::from_file(dir.join("IMG.CHTAB1")).expect("CHTAB1 loads");
        let chtab2 = ImageTable::from_file(dir.join("IMG.CHTAB2")).expect("CHTAB2 loads");
        KidArt {
            stand: chtab1.images.get(14).expect("stand frame").clone(),
            fall: chtab2.images.get(53).expect("freefall frame").clone(),
            // Run frames 7-14 = CHTAB1 indices 6..14.
            run: (6..14).map(|i| chtab1.images[i].clone()).collect(),
        }
    }

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
    fn walk_dir_prefers_right() {
        let none = InputState::default();
        assert_eq!(walk_dir(none), 0);
        assert_eq!(
            walk_dir(InputState {
                right: true,
                ..none
            }),
            1
        );
        assert_eq!(walk_dir(InputState { left: true, ..none }), -1);
        assert_eq!(
            walk_dir(InputState {
                left: true,
                right: true,
                ..none
            }),
            1
        );
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
    fn lv1_prince_spawns_airborne_facing_right() {
        let prince = Prince::spawn(&load_level1());
        // Block 0 = col 0, row 0 (Empty ledge) → spawns above the floor.
        assert_eq!(prince.room, 1);
        assert!(!prince.on_ground, "LV1 spawns on an empty ledge → falls");
        assert!(prince.facing_right);
        // Settles onto the row-2 block top = the row-1 floor line.
        assert_eq!(prince.landing_y, i32::from(BLOCK_BOT_ROW[1]));
        assert!(prince.feet_y < prince.landing_y);
    }

    #[test]
    fn prince_falls_and_lands() {
        let mut world = World::new(load_level1(), dungeon_tables());
        assert!(!world.prince_on_ground());
        let landing = world.prince.landing_y;
        for _ in 0..30 {
            world.tick(InputState::default());
            if world.prince_on_ground() {
                break;
            }
        }
        assert!(
            world.prince_on_ground(),
            "Prince should land within 30 ticks"
        );
        assert_eq!(world.prince.feet_y, landing);
    }

    #[test]
    fn art_overlay_adds_pixels_in_start_room() {
        let world = World::new(load_level1(), dungeon_tables());
        let plain = world.render(RenderMode::Monochrome).expect("renders");
        let world = world.with_kid_art(kid_art());
        let drawn = world.render(RenderMode::Monochrome).expect("renders");
        assert_ne!(
            plain.pixels, drawn.pixels,
            "kid overlay should change the frame"
        );
        let lit = |f: &Frame| {
            f.pixels
                .chunks_exact(4)
                .filter(|p| p[0..3] != [0, 0, 0])
                .count()
        };
        assert!(lit(&drawn) > lit(&plain), "kid overlay adds lit pixels");
    }

    #[test]
    fn render_returns_full_size_frame() {
        let world = World::new(load_level1(), dungeon_tables()).with_kid_art(kid_art());
        let frame = world.render(RenderMode::NtscColor).expect("renders");
        assert_eq!((frame.width, frame.height), (280, 192));
    }

    #[test]
    fn grounded_prince_runs_right_on_arrow() {
        let mut world = World::new(load_level1(), dungeon_tables());
        // Let him land first (no steering mid-air).
        for _ in 0..30 {
            world.tick(InputState::default());
            if world.prince_on_ground() {
                break;
            }
        }
        assert!(world.prince_on_ground());
        let x0 = world.prince.x;
        let held = InputState {
            right: true,
            ..InputState::default()
        };
        world.tick(held);
        assert!(world.prince.moving, "arrow puts him in the run state");
        assert!(world.prince.facing_right);
        assert!(world.prince.x > x0, "he advances to the right");
        // Releasing returns him to standing.
        world.tick(InputState::default());
        assert!(!world.prince.moving);
    }
}
