//! World model, the player character, and the frame loop (Path B,
//! issues #92 / #94 / #120).
//!
//! [`World`] holds a loaded level + its biome sprites, a [`Mode`] machine,
//! the current room, and the [`Prince`]. `render` composites the room
//! scene ([`pop_assets::scene`]) and draws the Prince on top.
//!
//! The Prince uses a clean pixel-space physics model (not POP's
//! `CharX`/`CharY` fixed-point coordinates) — Path B reads the original
//! for *behaviour* and reuses its *art*, but the runtime is fresh Rust.
//! He spawns, drops in (LV1 `SUBS.S :special1 → stepfall`), runs / jumps /
//! steps, and now respects the tiles: he stands on floors, stops at solid
//! walls, and falls off ledges (first slice of #120). Still missing:
//! step-up, the faithful stepfall landing on the upper floor, room
//! transitions, and the foreground (front-piece) draw order (#120 / #121).

use pop_assets::bgdata::{BLOCK_BOT_ROW, CELL_WIDTH_BYTES, ROOM_HEIGHT_PX, ROOM_WIDTH_BYTES};
use pop_assets::draz::image_table::Image;
use pop_assets::hires::{self, Frame, RenderMode};
use pop_assets::level::{Level, Room, TileKind, ROOMS_PER_LEVEL, ROOM_HEIGHT, ROOM_WIDTH};
use pop_assets::scene::{self, Anim, BiomeTables};
use pop_assets::sprite;

use crate::backend::InputState;

/// Downward acceleration applied each logic tick while the Prince is
/// airborne, in pixels per tick. Tuned for the ~18 Hz host step; refined
/// against POP's real fall timing when the controller matures (#94).
const GRAVITY: i32 = 3;

/// Pixels from a row's block-bottom up to where the character stands
/// (`FloorY = BlockBot - VertDist`, `TABLES.S` "bottom of block to centre
/// plane"). Without it the Prince renders a touch too low.
const VERT_DIST: i32 = 10;

/// Tile column width in pixels (`CELL_WIDTH_BYTES * 7`).
const CELL_W: i32 = CELL_WIDTH_BYTES as i32 * 7;

/// Room width in pixels. Crossing it moves the Prince into the linked
/// neighbour room (see [`Level::room_links`]).
const ROOM_W: i32 = ROOM_WIDTH_BYTES as i32 * 7;

/// Per-frame horizontal step of the run cycle, in pixels — the `chx`
/// operands of SEQTABLE `runcyc1..8`. Indexed by the run phase.
const RUN_CHX: [i32; 8] = [5, 1, 2, 4, 5, 2, 3, 4];

/// Initial upward velocity of a jump, px per tick (negative = up). A
/// simple vertical hop; the running leap and ledge grabs come later (#120).
const JUMP_VY: i32 = -12;

/// Distance per tick of a careful step (SHIFT held), px — slower than the
/// run so he can edge up to gaps.
const STEP_PX: i32 = 2;

/// Half the Prince's collision width, px. His centre stops this far from a
/// wall — matched to [`VERT_DIST`] so he keeps about the same gap from a
/// wall as his feet keep from the bottom of the tile he stands on.
const COLLIDE_HALF: i32 = VERT_DIST;

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

/// The player character's physical state in pixel space.
struct Prince {
    /// Room the kid is in (1-based). He's only drawn while this is the
    /// room on screen.
    room: u8,
    /// Horizontal centre, room pixels.
    x: i32,
    /// Feet line, room pixels — the sprite's bottom scan-line sits here.
    feet_y: i32,
    /// Tile row he stands on while grounded (`0..ROOM_HEIGHT`).
    row: usize,
    /// Vertical velocity, pixels per tick (down is positive).
    vy: i32,
    /// Feet line he lands on when the current fall completes.
    landing_y: i32,
    /// Row he lands on when the current fall completes.
    landing_row: usize,
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
        let spawn_row = (usize::from(start.block) / cols).min(ROOM_HEIGHT - 1);
        let x = i32::try_from(col).unwrap_or(0) * CELL_W + CELL_W / 2;
        let room = clamp_start_room(start.screen);
        let spawn_feet = floor_y(spawn_row);
        let (landing_row, landing_y) = level
            .rooms
            .get(usize::from(room) - 1)
            .map_or((spawn_row, spawn_feet), |r| settle(r, col, spawn_row));
        let on_ground = spawn_feet >= landing_y;
        Prince {
            room,
            x,
            feet_y: if on_ground { landing_y } else { spawn_feet },
            row: if on_ground { landing_row } else { spawn_row },
            vy: 0,
            landing_y,
            landing_row,
            on_ground,
            facing_right: faces_right(start.face_raw),
            run_phase: 0,
            moving: false,
        }
    }

    /// Advance one airborne tick: gravity pulls `vy` down, move, and land
    /// when he's falling (`vy >= 0`) and reaches `landing_y`. Works for
    /// both a fall and the up-then-down arc of a jump.
    fn physics(&mut self) {
        self.vy += GRAVITY;
        self.feet_y += self.vy;
        if self.vy >= 0 && self.feet_y >= self.landing_y {
            self.feet_y = self.landing_y;
            self.row = self.landing_row;
            self.vy = 0;
            self.on_ground = true;
        }
    }

    /// Launch a standing jump — a vertical hop back onto the current floor.
    fn jump(&mut self) {
        self.landing_row = self.row;
        self.landing_y = self.feet_y;
        self.vy = JUMP_VY;
        self.on_ground = false;
        self.moving = false;
    }

    /// Run one tick in `dir` (`-1` left, `+1` right, `0` idle): face the
    /// way he moves, advance the run cycle, and step by that frame's
    /// `chx`. `dir == 0` returns him to standing.
    fn walk(&mut self, dir: i32, room: &Room) {
        if dir == 0 {
            self.moving = false;
            self.run_phase = 0;
            return;
        }
        self.run_phase = (self.run_phase + 1) % RUN_CHX.len();
        self.move_h(RUN_CHX[self.run_phase] * dir, room);
    }

    /// Take one careful step in `dir` (SHIFT held): a slow `STEP_PX` move
    /// with the run animation, for edging up to a gap.
    fn step(&mut self, dir: i32, room: &Room) {
        self.run_phase = (self.run_phase + 1) % RUN_CHX.len();
        self.move_h(STEP_PX * dir, room);
    }

    /// Move horizontally by `dx` px against `room`'s tiles: stop flush at a
    /// solid wall, follow the floor across flat ground, and start a fall
    /// when he walks off a ledge.
    fn move_h(&mut self, dx: i32, room: &Room) {
        let dir = dx.signum();
        self.facing_right = dir > 0;
        self.moving = true;

        let mut target_x = self.x + dx;

        // Collide on his *leading edge*, not his centre — his body reaches
        // the wall before his centre crosses the cell boundary. If the
        // column under the leading edge is solid, stop the edge flush
        // against the wall. (Off-room columns aren't walls — they're the
        // doorway to a neighbour, handled by `cross_horizontal`.)
        let lead = target_x + dir * COLLIDE_HALF;
        if (0..ROOM_W).contains(&lead) && is_solid_at(room, col_of(lead), self.row) {
            let wall = i32::try_from(col_of(lead)).unwrap_or(0);
            target_x = if dir > 0 {
                wall * CELL_W - COLLIDE_HALF
            } else {
                (wall + 1) * CELL_W + COLLIDE_HALF
            };
        }

        self.x = target_x;

        // Follow the floor in the (possibly new) column; fall if it
        // dropped. Skip while he's stepping across the room edge — the
        // neighbour room's floor takes over once `cross_horizontal` moves
        // him there.
        if (0..ROOM_W).contains(&self.x) {
            let (lrow, lfeet) = settle(room, col_of(self.x), self.row);
            if lrow == self.row {
                self.feet_y = lfeet;
            } else {
                self.on_ground = false;
                self.vy = 0;
                self.landing_row = lrow;
                self.landing_y = lfeet;
            }
        }
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

    /// Carry the Prince into a neighbour room when he steps off the left
    /// or right edge — but only if the neighbour's edge column is open at
    /// his row. A solid edge tile there is a wall (POP treats the shared
    /// boundary as a wall when either side's edge tile is solid), so he
    /// stops at the boundary instead of warping into it. No neighbour
    /// (`0`) is the level edge: also a wall.
    fn cross_horizontal_edge(&mut self) {
        let Some(&links) = self
            .level
            .room_links()
            .get(usize::from(self.prince.room).saturating_sub(1))
        else {
            return;
        };
        let row = self.prince.row;
        if self.prince.x >= ROOM_W {
            // Off the right edge → enter the right neighbour at its col 0.
            if links.right != 0 && self.room_col_open(links.right, 0, row) {
                self.prince.room = links.right;
                self.prince.x -= ROOM_W;
            } else {
                self.prince.x = ROOM_W - COLLIDE_HALF;
            }
        } else if self.prince.x < 0 {
            // Off the left edge → enter the left neighbour at its last col.
            if links.left != 0 && self.room_col_open(links.left, ROOM_WIDTH - 1, row) {
                self.prince.room = links.left;
                self.prince.x += ROOM_W;
            } else {
                self.prince.x = COLLIDE_HALF;
            }
        }
    }

    /// `true` if `(col, row)` of room id `room` is a non-solid tile the
    /// Prince could step into. Missing room → `false` (treat as a wall).
    fn room_col_open(&self, room: u8, col: usize, row: usize) -> bool {
        self.level
            .rooms
            .get(usize::from(room).saturating_sub(1))
            .is_some_and(|r| !is_solid_at(r, col, row))
    }

    /// Advance one logic frame given the latest input.
    ///
    /// In `Playing`, once grounded: Up jumps, SHIFT + arrow takes a
    /// careful step, a bare arrow runs him that way (stopping at walls,
    /// falling off ledges), nothing stands. While airborne he follows
    /// gravity (no steering mid-air yet).
    pub fn tick(&mut self, input: InputState) {
        self.frame = self.frame.wrapping_add(1);
        match self.mode {
            Mode::Title => {
                if pressed(input) && !pressed(self.prev) {
                    self.mode = Mode::Playing;
                }
            }
            Mode::Playing => {
                let room_idx = usize::from(self.prince.room).saturating_sub(1);
                if !self.prince.on_ground {
                    self.prince.physics();
                } else if input.up && !self.prev.up {
                    self.prince.jump();
                } else if let Some(room) = self.level.rooms.get(room_idx) {
                    let dir = walk_dir(input);
                    if input.shift && dir != 0 {
                        self.prince.step(dir, room);
                    } else {
                        self.prince.walk(dir, room);
                    }
                }
                // Carry him into a neighbour room if he stepped off an edge.
                self.cross_horizontal_edge();
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
                // The full scene (background + foreground) before the kid.
                let scene_bytes = bytes.clone();
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
                // Restore the foreground over the kid so columns / near
                // floor-edges occlude him (the original's drawfrnt-after-
                // drawchar order). A byte the foreground pass touched
                // (≠ the `0x80` "black2" CLS fill) reverts to the pre-kid
                // scene byte.
                if let Some(fg) = scene::compose_foreground_bytes(
                    &self.level,
                    self.room_id,
                    &self.tables,
                    Anim::REST,
                ) {
                    for (out, (&scene, &front)) in
                        bytes.iter_mut().zip(scene_bytes.iter().zip(fg.iter()))
                    {
                        if front != 0x80 {
                            *out = scene;
                        }
                    }
                }
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

/// Feet line (`FloorY`) for a tile row: the row's block-bottom lifted by
/// [`VERT_DIST`].
fn floor_y(row: usize) -> i32 {
    i32::from(BLOCK_BOT_ROW[row.min(ROOM_HEIGHT - 1)]) - VERT_DIST
}

/// Tile column containing pixel `x`, clamped to the room.
fn col_of(x: i32) -> usize {
    usize::try_from(x.max(0) / CELL_W)
        .unwrap_or(0)
        .min(ROOM_WIDTH - 1)
}

/// `true` if `(col, row)` is a solid tile (a wall the kid can't enter).
fn is_solid_at(room: &Room, col: usize, row: usize) -> bool {
    room.tile_at(col, row)
        .is_some_and(|t| tile_is_solid(t.kind))
}

/// Where the kid's feet rest when dropped straight down from
/// (`col`, `from_row`): `(row, feet_y)` of the first floor-bearing tile,
/// or the top of the first solid block, falling back to the bottom row.
fn settle(room: &Room, col: usize, from_row: usize) -> (usize, i32) {
    for r in from_row..ROOM_HEIGHT {
        let Some(kind) = room.tile_at(col, r).map(|t| t.kind) else {
            continue;
        };
        if tile_has_floor(kind) {
            return (r, floor_y(r));
        }
        if tile_is_solid(kind) {
            let top = r.saturating_sub(1);
            return (top, floor_y(top));
        }
    }
    (ROOM_HEIGHT - 1, floor_y(ROOM_HEIGHT - 1))
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

/// `true` for solid blocks the kid stands on *top* of (and can't walk
/// through).
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
            run: (6..14).map(|i| chtab1.images[i].clone()).collect(),
        }
    }

    fn landed_world() -> World {
        let mut world = World::new(load_level1(), dungeon_tables());
        for _ in 0..40 {
            world.tick(InputState::default());
            if world.prince_on_ground() {
                break;
            }
        }
        assert!(world.prince_on_ground());
        world
    }

    #[test]
    fn clamp_start_room_keeps_invariant() {
        let last = u8::try_from(ROOMS_PER_LEVEL).unwrap();
        assert_eq!(clamp_start_room(0), 1);
        assert_eq!(clamp_start_room(1), 1);
        assert_eq!(clamp_start_room(last), last);
        assert_eq!(clamp_start_room(last + 1), last);
        assert_eq!(clamp_start_room(u8::MAX), last);
    }

    #[test]
    fn floor_y_lifts_block_bottom_by_vertdist() {
        assert_eq!(floor_y(0), i32::from(BLOCK_BOT_ROW[0]) - VERT_DIST);
        assert_eq!(floor_y(1), i32::from(BLOCK_BOT_ROW[1]) - VERT_DIST);
        assert_eq!(floor_y(2), i32::from(BLOCK_BOT_ROW[2]) - VERT_DIST);
    }

    #[test]
    fn col_of_maps_pixels_to_tiles() {
        assert_eq!(col_of(0), 0);
        assert_eq!(col_of(27), 0);
        assert_eq!(col_of(28), 1);
        assert_eq!(col_of(10_000), ROOM_WIDTH - 1);
    }

    #[test]
    fn settle_stands_on_block_top_at_lv1_spawn() {
        // LV1 room 1 col 0 = Empty / Torch / Block → stand on the row-2
        // block top, i.e. row 1's floor line.
        let level = load_level1();
        let (row, feet) = settle(&level.rooms[0], 0, 0);
        assert_eq!(row, 1);
        assert_eq!(feet, floor_y(1));
    }

    #[test]
    fn faces_right_decodes_kidstartface() {
        assert!(faces_right(0xff));
        assert!(!faces_right(0x00));
        assert!(!faces_right(0x7f));
        assert!(faces_right(0x80));
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
    fn lv1_prince_spawns_airborne_and_lands_on_row1() {
        let prince = Prince::spawn(&load_level1());
        assert_eq!(prince.room, 1);
        assert!(!prince.on_ground, "LV1 spawns on an empty ledge → falls");
        assert!(prince.facing_right);
        assert_eq!(prince.landing_row, 1);
        assert_eq!(prince.landing_y, floor_y(1));
        assert!(prince.feet_y < prince.landing_y);
    }

    #[test]
    fn prince_falls_and_lands_on_floor_line() {
        let world = landed_world();
        assert_eq!(world.prince.row, 1);
        assert_eq!(world.prince.feet_y, floor_y(1));
    }

    #[test]
    fn grounded_prince_runs_right_on_arrow() {
        let mut world = landed_world();
        let x0 = world.prince.x;
        world.tick(InputState {
            right: true,
            ..InputState::default()
        });
        assert!(world.prince.facing_right);
        assert!(world.prince.x > x0, "he advances to the right");
        assert!(world.prince.moving);
        world.tick(InputState::default());
        assert!(!world.prince.moving);
    }

    #[test]
    fn running_off_a_ledge_starts_a_fall() {
        // Running right from the bottom-left, he reaches the col-4 gap
        // (row 1 Empty over row 2 Rubble) and drops to the lower floor.
        let mut world = landed_world();
        let mut fell = false;
        for _ in 0..120 {
            world.tick(InputState {
                right: true,
                ..InputState::default()
            });
            if !world.prince_on_ground() {
                fell = true;
                break;
            }
        }
        assert!(fell, "walking off the ledge should start a fall");
    }

    #[test]
    fn running_into_a_wall_stops_before_it() {
        // Running right, he drops to row 2 at the col-4 gap, then meets the
        // col-9 Block wall. His leading edge must never pass the wall's
        // left edge (`9 * CELL_W`), even though his centre stays left of it.
        let mut world = landed_world();
        let mut max_lead = 0;
        for _ in 0..300 {
            world.tick(InputState {
                right: true,
                ..InputState::default()
            });
            max_lead = max_lead.max(world.prince.x + COLLIDE_HALF);
        }
        assert!(
            max_lead <= 9 * CELL_W,
            "leading edge {max_lead} passed the wall at {}",
            9 * CELL_W
        );
        // And he actually got close to it (didn't stall early).
        assert!(max_lead >= 8 * CELL_W, "he should reach the wall");
    }

    #[test]
    fn running_into_a_left_wall_stops_before_it() {
        // Drop to row 2, then run back left into the col-3 block wall.
        let mut world = landed_world();
        for _ in 0..60 {
            world.tick(InputState {
                right: true,
                ..InputState::default()
            });
        }
        let mut min_lead = i32::MAX;
        for _ in 0..200 {
            world.tick(InputState {
                left: true,
                ..InputState::default()
            });
            min_lead = min_lead.min(world.prince.x - COLLIDE_HALF);
        }
        // His leading (left) edge can't pass the col-3 wall's right edge.
        assert!(
            min_lead >= 4 * CELL_W,
            "left edge {min_lead} passed the wall at {}",
            4 * CELL_W
        );
    }

    #[test]
    fn walled_neighbour_edge_blocks_transition() {
        // LV1 room 1's left link is room 5, but room 5's right column (col 9)
        // is a Block at row 1 — the shared boundary is a wall. Walking left
        // off the edge must NOT cross; he stops in room 1.
        let mut world = landed_world();
        assert_eq!(world.room_id(), 1);
        for _ in 0..30 {
            world.tick(InputState {
                left: true,
                ..InputState::default()
            });
        }
        assert_eq!(world.room_id(), 1, "room 5's edge wall blocks the crossing");
        assert!(world.prince.x >= 0, "he stays inside room 1");
    }

    #[test]
    fn open_neighbour_edge_transitions() {
        // Force him onto row 0 (room 5's col 9 there is a Gate, not solid)
        // just past room 1's left edge: the crossing into room 5 succeeds.
        let mut world = landed_world();
        world.prince.row = 0;
        world.prince.x = -1;
        world.cross_horizontal_edge();
        assert_eq!(world.prince.room, 5, "open boundary crosses into room 5");
        assert!(
            world.prince.x >= ROOM_W / 2,
            "he wraps to the right side of room 5"
        );
    }

    #[test]
    fn up_arrow_jumps_and_lands_back() {
        let mut world = landed_world();
        let floor = world.prince.feet_y;
        world.tick(InputState {
            up: true,
            ..InputState::default()
        });
        assert!(!world.prince_on_ground(), "Up launches a jump");
        world.tick(InputState::default());
        world.tick(InputState::default());
        assert!(world.prince.feet_y < floor, "the jump rises off the floor");
        for _ in 0..40 {
            world.tick(InputState::default());
            if world.prince_on_ground() {
                break;
            }
        }
        assert!(world.prince_on_ground(), "the jump lands");
        assert_eq!(world.prince.feet_y, floor, "back on the same floor");
    }

    #[test]
    fn shift_arrow_takes_a_careful_step() {
        let mut world = landed_world();
        let x0 = world.prince.x;
        world.tick(InputState {
            shift: true,
            right: true,
            ..InputState::default()
        });
        assert!(world.prince.facing_right);
        assert_eq!(world.prince.x - x0, STEP_PX, "careful step moves STEP_PX");
    }

    #[test]
    fn render_returns_full_size_frame() {
        let world = World::new(load_level1(), dungeon_tables()).with_kid_art(kid_art());
        let frame = world.render(RenderMode::NtscColor).expect("renders");
        assert_eq!((frame.width, frame.height), (280, 192));
    }
}
