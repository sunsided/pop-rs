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
//! He spawns, drops in, runs / jumps / careful-steps, climbs ledges,
//! walks between rooms, and falls / descends loose floors into the room
//! below — respecting walls, floors and ledges (#120). His drawn frame and
//! run displacement now come from the decoded SEQTABLE sequences through an
//! [`AnimCursor`] (the #95 playback engine — stand / run / fall / climb
//! wired; turn-in-place, faithful jump arcs and combat still to come). Loose
//! floors bob their own tile as they wobble, then give way; a crash or hard
//! landing jolts the screen (#96). Still rough: the falling-rubble mob +
//! rubble tile that round out #96, and the NTSC half-dot / figure-offset
//! blit (#121 / #123).

use pop_assets::bgdata::{BLOCK_BOT_ROW, CELL_WIDTH_BYTES, ROOM_HEIGHT_PX, ROOM_WIDTH_BYTES};
use pop_assets::draz::image_table::{Image, ImageTable};
use pop_assets::hires::{self, Frame, RenderMode};
use pop_assets::level::{Level, Room, Tile, TileKind, ROOMS_PER_LEVEL, ROOM_HEIGHT, ROOM_WIDTH};
use pop_assets::scene::{self, Anim, BiomeTables};
use pop_assets::sprite;

use crate::anim::AnimCursor;
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

/// Room height in pixels. Falling past it drops the Prince into the room
/// below (the `down` link).
const ROOM_H: i32 = ROOM_HEIGHT_PX as i32;

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

/// Ticks a jarred loose floor wobbles before it gives way. POP arms it with
/// a random `1..16`-tick countdown (`SUBS.S :trigloose` → `BREAKLOOSE1`); we
/// use a fixed short beat — long enough to feel the floor hesitate, short
/// enough that he can't run clear of a single loose tile before it drops.
const LOOSE_FALL_DELAY: u8 = 4;

/// Frames the picture jolts after a loose floor crashes down (`MOVER.S
/// SHAKEM`): each frame it rolls a couple of scan-lines, sign alternating.
const SHAKE_TICKS: u8 = 4;

/// Vertical roll of the crash / landing jolt, px.
const SHAKE_AMPLITUDE: i32 = 2;

/// Per-frame downward bob (px) of a loose floor's own tile while it
/// wobbles — POP's dedicated loose-block shudder (`FRAMEADV.S drawloosed`
/// offsets the block by `looseby`), local to the tile, not a screen roll.
const LOOSE_BOB: [usize; 4] = [0, 1, 2, 1];

/// Minimum downward speed at touchdown that thuds the floor (jolts the
/// screen, `MOVER.S SHAKEM`). Below it a gentle step-down lands quietly.
const THUD_VY: i32 = 6;

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

/// In-progress ledge climb: linearly carries the Prince from where he
/// stood up onto the ledge over the `climbup` animation's frames.
#[derive(Clone, Copy)]
struct ClimbState {
    /// Frame / interpolation step (`0..climb.len()`).
    phase: usize,
    from_x: i32,
    from_feet: i32,
    to_x: i32,
    to_feet: i32,
    to_row: usize,
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
    /// Animation cursor — the sequence + frame the engine plays for him
    /// (`stand` / `startrun` / `freefall` / `climbup`); the source of his
    /// drawn frame and per-frame run displacement (#95).
    cursor: AnimCursor,
    /// `Some` while climbing a ledge (overrides walk / fall).
    climb: Option<ClimbState>,
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
            .and_then(|r| settle(r, col, spawn_row))
            .unwrap_or((spawn_row, spawn_feet));
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
            cursor: AnimCursor::new("stand"),
            climb: None,
        }
    }

    /// Launch a standing jump — a vertical hop back onto the current floor.
    fn jump(&mut self) {
        self.landing_row = self.row;
        self.landing_y = self.feet_y;
        self.vy = JUMP_VY;
        self.on_ground = false;
    }

    /// One grounded locomotion tick in `dir` (`-1` left, `+1` right, `0`
    /// idle). Idle plays `stand`; otherwise he runs (or careful-steps with
    /// SHIFT held), facing `dir` and stepping by the run sequence's current
    /// per-frame `chx` (a fixed [`STEP_PX`] when careful). The cursor was
    /// advanced for this tick already, so its `current()` frame is the one to
    /// move and draw by.
    fn locomote(&mut self, dir: i32, careful: bool, room: &Room) {
        if dir == 0 {
            self.cursor.play("stand");
            return;
        }
        self.facing_right = dir > 0;
        self.cursor.play("startrun");
        let dx = if careful {
            STEP_PX
        } else {
            self.cursor.current().map_or(0, |f| f.dx)
        };
        self.move_h(dx * dir, room);
    }

    /// Move horizontally by `dx` px against `room`'s tiles: stop flush at a
    /// solid wall, follow the floor across flat ground, and start a fall
    /// when he walks off a ledge.
    fn move_h(&mut self, dx: i32, room: &Room) {
        let dir = dx.signum();
        // A zero-`dx` frame (the run cycle's windup) mustn't flip him to
        // facing left — keep the facing the caller set.
        if dir != 0 {
            self.facing_right = dir > 0;
        }

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
        // neighbour room's floor takes over once `cross_horizontal_edge`
        // moves him there (and settles him in the new room).
        if (0..ROOM_W).contains(&self.x) {
            self.settle_floor(room);
        }
    }

    /// Reconcile his feet against the floor of his current column in
    /// `room`: rest on it when one is there, otherwise begin a fall to the
    /// floor below. Used by `move_h` and right after a room crossing so the
    /// neighbour room's floor (or a fall) takes effect immediately.
    fn settle_floor(&mut self, room: &Room) {
        match settle(room, col_of(self.x), self.row) {
            // Floor right at his row: he stays grounded on it.
            Some((lrow, lfeet)) if lrow == self.row => {
                self.feet_y = lfeet;
                self.on_ground = true;
            }
            // A floor lower in this room: fall to it.
            Some((lrow, lfeet)) => {
                self.on_ground = false;
                self.vy = 0;
                self.landing_row = lrow;
                self.landing_y = lfeet;
            }
            // No floor in this column: fall past the bottom. `landing_y`
            // beyond the room signals `World::fall_step` to carry him into
            // the room below (or land him at the bottom if there's none).
            None => {
                self.on_ground = false;
                self.vy = 0;
                self.landing_row = ROOM_HEIGHT - 1;
                self.landing_y = ROOM_H + 1;
            }
        }
    }
}

/// A loose floor that's been jarred and is counting down to give way
/// (`MOVER.S` "trob": triggered object). The cell stays solid in the level
/// until [`World::advance_loose`] runs the countdown out.
struct LooseArm {
    room: u8,
    col: usize,
    row: usize,
    ticks_left: u8,
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
    /// Loose floors mid-wobble, counting down to break (`MOVER.S` trobs).
    loose: Vec<LooseArm>,
    /// Screen-jolt frames remaining after a loose floor crashed (`SHAKEM`).
    shake: u8,
    /// Character sprite tables (`IMG.CHTAB1..8`), host-loaded; slot `i` holds
    /// CHTAB `i+1`. The animation engine resolves a frame to one of these
    /// (`pop_assets::anim::frame_sprite`). Empty / missing → bare scene.
    chtabs: Vec<Option<ImageTable>>,
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
            loose: Vec::new(),
            shake: 0,
            chtabs: Vec::new(),
        }
    }

    /// Attach the character sprite tables (`IMG.CHTAB1..8`; the host loads
    /// them so the engine stays free of file I/O). Slot `i` is CHTAB `i+1`.
    /// Chainable.
    #[must_use]
    pub fn with_chtabs(mut self, chtabs: Vec<Option<ImageTable>>) -> Self {
        self.chtabs = chtabs;
        self
    }

    /// Resolve a 1-based animation frame id to its sprite image: decode the
    /// frame's `(chtab, index)` (`pop_assets::anim::frame_sprite`, the kid's
    /// view) and look it up in the loaded tables. `None` if the frame has no
    /// sprite or its table / image isn't loaded.
    fn frame_image(&self, frame: u8) -> Option<&Image> {
        let sprite = pop_assets::anim::frame_sprite(frame)?;
        let table = self
            .chtabs
            .get(usize::from(sprite.chtab).checked_sub(1)?)?
            .as_ref()?;
        table.images.get(sprite.index)
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
    /// Returns `true` if he actually crossed into a neighbour room.
    fn cross_horizontal_edge(&mut self) -> bool {
        let Some(&links) = self
            .level
            .room_links()
            .get(usize::from(self.prince.room).saturating_sub(1))
        else {
            return false;
        };
        let row = self.prince.row;
        let crossed = if self.prince.x >= ROOM_W {
            // Off the right edge → enter the right neighbour at its col 0.
            if links.right != 0 && self.room_col_open(links.right, 0, row) {
                self.prince.room = links.right;
                self.prince.x -= ROOM_W;
                true
            } else {
                self.prince.x = ROOM_W - COLLIDE_HALF;
                false
            }
        } else if self.prince.x < 0 {
            // Off the left edge → enter the left neighbour at its last col.
            if links.left != 0 && self.room_col_open(links.left, ROOM_WIDTH - 1, row) {
                self.prince.room = links.left;
                self.prince.x += ROOM_W;
                true
            } else {
                self.prince.x = COLLIDE_HALF;
                false
            }
        } else {
            false
        };
        // Reconcile his feet against the new room's entry column right away
        // — otherwise, if input is released this tick, `walk(0)` never
        // settles him and he'd hover over a floorless entry.
        if crossed {
            if let Some(room) = self
                .level
                .rooms
                .get(usize::from(self.prince.room).saturating_sub(1))
            {
                self.prince.settle_floor(room);
            }
        }
        crossed
    }

    /// One airborne tick: gravity, then carry him into the room below each
    /// time he falls past the bottom edge (the `down` link), until a floor
    /// in his column catches him. Also runs the up-then-down arc of a jump.
    fn fall_step(&mut self) {
        self.prince.vy += GRAVITY;
        self.prince.feet_y += self.prince.vy;

        // Descend through `down`-linked rooms while he's below the floor.
        // Bounded by the room count so a cyclic `down` chain in malformed
        // level data can't spin forever — it lands him at the bottom.
        for _ in 0..=ROOMS_PER_LEVEL {
            if self.prince.feet_y < ROOM_H {
                break;
            }
            let room_idx = usize::from(self.prince.room).saturating_sub(1);
            let down = self.level.room_links().get(room_idx).map_or(0, |l| l.down);
            if down == 0 {
                // Bottom of the level: land on the bottom-row floor.
                self.prince.feet_y = floor_y(ROOM_HEIGHT - 1);
                self.prince.vy = 0;
                self.prince.row = ROOM_HEIGHT - 1;
                self.prince.on_ground = true;
                return;
            }
            self.prince.room = down;
            self.prince.feet_y -= ROOM_H;
            // Re-aim the landing at the next floor *at or below his feet* in
            // the new room — scanning from the top could snap him up onto a
            // platform he's already fallen past when `vy` is large.
            let col = col_of(self.prince.x);
            let from_row = row_below_feet(self.prince.feet_y);
            if let Some((lrow, lfeet)) = self
                .level
                .rooms
                .get(usize::from(down).saturating_sub(1))
                .and_then(|r| settle(r, col, from_row))
            {
                self.prince.landing_row = lrow;
                self.prince.landing_y = lfeet;
            } else {
                // Open here too → keep falling into the next room.
                self.prince.landing_row = ROOM_HEIGHT - 1;
                self.prince.landing_y = ROOM_H + 1;
            }
        }

        // Still below a room floor after the bounded descent → a cyclic /
        // degenerate `down` chain. Clamp him to the bottom rather than fall
        // forever.
        if self.prince.feet_y >= ROOM_H {
            self.prince.feet_y = floor_y(ROOM_HEIGHT - 1);
            self.prince.vy = 0;
            self.prince.row = ROOM_HEIGHT - 1;
            self.prince.on_ground = true;
            return;
        }

        // Land once he reaches the floor he's aimed at in the current room.
        if self.prince.vy >= 0 && self.prince.feet_y >= self.prince.landing_y {
            self.prince.feet_y = self.prince.landing_y;
            self.prince.vy = 0;
            self.prince.row = self.prince.landing_row;
            self.prince.on_ground = true;
        }
    }

    /// Arm the loose floor the grounded Prince stands on, if any: his weight
    /// jars it (`SUBS.S :trigloose`) and it begins a [`LOOSE_FALL_DELAY`]-tick
    /// wobble. The cell stays solid until the countdown runs out in
    /// [`World::advance_loose`]; only then does it become a hole and take him
    /// with it. Idempotent per cell — re-arming a wobbling floor is a no-op.
    fn arm_loose_floor_under_feet(&mut self) {
        if !self.prince.on_ground {
            return;
        }
        let room = self.prince.room;
        let col = col_of(self.prince.x);
        let row = self.prince.row;
        let on_loose = self
            .level
            .rooms
            .get(usize::from(room).saturating_sub(1))
            .and_then(|r| r.tile_at(col, row))
            .is_some_and(|t| t.kind == TileKind::LooseFloor);
        if !on_loose {
            return;
        }
        let armed = self
            .loose
            .iter()
            .any(|l| l.room == room && l.col == col && l.row == row);
        if !armed {
            self.loose.push(LooseArm {
                room,
                col,
                row,
                ticks_left: LOOSE_FALL_DELAY,
            });
        }
    }

    /// Tick every wobbling loose floor (and the screen jolt). When a
    /// countdown reaches zero the cell becomes a hole (`Empty`) in the level
    /// data — renderer and floor physics both see it — and the picture jolts
    /// ([`SHAKE_TICKS`]). If the Prince is still standing on that very cell he
    /// drops through it; if he ran clear in time it just crumbles behind him.
    fn advance_loose(&mut self) {
        self.shake = self.shake.saturating_sub(1);
        let mut broken: Vec<(u8, usize, usize)> = Vec::new();
        self.loose.retain_mut(|l| {
            l.ticks_left = l.ticks_left.saturating_sub(1);
            if l.ticks_left == 0 {
                broken.push((l.room, l.col, l.row));
                false
            } else {
                true
            }
        });
        for (room, col, row) in broken {
            let room_idx = usize::from(room).saturating_sub(1);
            if let Some(r) = self.level.rooms.get_mut(room_idx) {
                r.tiles[row * ROOM_WIDTH + col] = Tile::default();
            }
            // Only jolt the picture if the crash is in the room on screen —
            // a floor giving way two rooms over shouldn't shake the view.
            if room == self.room_id {
                self.shake = SHAKE_TICKS;
            }
            let on_it = self.prince.on_ground
                && self.prince.room == room
                && self.prince.row == row
                && col_of(self.prince.x) == col;
            if on_it {
                self.drop_through_hole(room_idx, col, row);
            }
        }
    }

    /// Start the Prince falling through a fresh hole at `(col, row)` of room
    /// `room_idx`: aim at the next support below it in this room if there is
    /// one, else leave the column open so [`World::fall_step`] carries him
    /// into the room below.
    fn drop_through_hole(&mut self, room_idx: usize, col: usize, row: usize) {
        self.prince.on_ground = false;
        self.prince.vy = 0;
        if let Some((lrow, lfeet)) = self
            .level
            .rooms
            .get(room_idx)
            .and_then(|r| settle(r, col, row))
        {
            self.prince.landing_row = lrow;
            self.prince.landing_y = lfeet;
        } else {
            self.prince.landing_row = ROOM_HEIGHT - 1;
            self.prince.landing_y = ROOM_H + 1;
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

    /// Start a ledge climb if the grounded Prince faces a floor one row up
    /// in the next column. Returns `true` if it started. In-room only for
    /// now — climbing into the room above is a follow-up under #120.
    fn try_climb(&mut self) -> bool {
        if self.prince.row == 0 {
            return false;
        }
        let to_row = self.prince.row - 1;
        let dir = if self.prince.facing_right { 1 } else { -1 };
        let front = i32::try_from(col_of(self.prince.x)).unwrap_or(0) + dir;
        if !(0..i32::try_from(ROOM_WIDTH).unwrap_or(0)).contains(&front) {
            return false;
        }
        let front = usize::try_from(front).unwrap_or(0);
        let room_idx = usize::from(self.prince.room).saturating_sub(1);
        let Some(room) = self.level.rooms.get(room_idx) else {
            return false;
        };
        // A grabbable ledge is a floor *at* `to_row` he can stand on — not
        // a solid block (a solid at `to_row` would `saturating_sub` to
        // `(0, ..)` at the top row and read as a ledge into the wall, and a
        // solid lower down resolves to its top a row higher). Climbing onto
        // the top of a block is a follow-up (#120).
        let ledge = settle(room, front, to_row) == Some((to_row, floor_y(to_row)))
            && !is_solid_at(room, front, to_row);
        if !ledge {
            return false;
        }
        self.prince.climb = Some(ClimbState {
            phase: 0,
            from_x: self.prince.x,
            from_feet: self.prince.feet_y,
            to_x: i32::try_from(front).unwrap_or(0) * CELL_W + CELL_W / 2,
            to_feet: floor_y(to_row),
            to_row,
        });
        self.prince.on_ground = false;
        true
    }

    /// Advance one climb frame, interpolating his position up to the ledge;
    /// on the last frame he lands grounded on it.
    fn climb_step(&mut self) {
        let Some(mut c) = self.prince.climb else {
            return;
        };
        // Interpolate his position over the `climbup` animation's length so
        // the pull-up reaches the ledge as the last climb frame draws.
        let frames = crate::anim::sequence_len("climbup").max(1);
        c.phase += 1;
        if c.phase >= frames {
            self.prince.x = c.to_x;
            self.prince.feet_y = c.to_feet;
            self.prince.row = c.to_row;
            self.prince.on_ground = true;
            self.prince.climb = None;
        } else {
            let t = i32::try_from(c.phase).unwrap_or(0);
            let n = i32::try_from(frames).unwrap_or(1).max(1);
            self.prince.x = c.from_x + (c.to_x - c.from_x) * t / n;
            self.prince.feet_y = c.from_feet + (c.to_feet - c.from_feet) * t / n;
            self.prince.climb = Some(c);
        }
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
                let was_climbing = self.prince.climb.is_some();
                // Capture his fall speed before the step so a touchdown this
                // tick can jolt the floor by how hard he hit.
                let airborne = !self.prince.on_ground;
                let vy_before = self.prince.vy;
                // Play out the frame chosen last tick before this tick's state
                // re-selects a sequence — so a freshly switched sequence draws
                // its first frame, not its second.
                self.prince.cursor.advance();
                if was_climbing {
                    self.climb_step();
                    self.prince.cursor.play("climbup");
                } else if !self.prince.on_ground {
                    self.fall_step();
                    self.prince.cursor.play("freefall");
                } else if input.up && !self.prev.up {
                    // Up grabs a ledge above-in-front, else it's a jump.
                    if self.try_climb() {
                        self.prince.cursor.play("climbup");
                    } else {
                        self.prince.jump();
                        self.prince.cursor.play("freefall");
                    }
                } else if let Some(room) = self.level.rooms.get(room_idx) {
                    let dir = walk_dir(input);
                    self.prince.locomote(dir, input.shift, room);
                }
                // A hard landing thuds the floor — the impact jolt of a fall
                // or jump. A climb finishes with no downward speed, so it
                // stays quiet (`vy_before < THUD_VY`).
                if airborne && self.prince.on_ground && vy_before >= THUD_VY {
                    self.shake = SHAKE_TICKS;
                }
                // Room edges / loose floors don't apply mid-climb — including
                // the tick a climb starts or finishes (he hasn't stepped on
                // the landing tile yet).
                if !was_climbing && self.prince.climb.is_none() {
                    // Carry him into a neighbour room if he stepped off an edge.
                    let crossed = self.cross_horizontal_edge();
                    // A loose floor under his feet starts to wobble. Skip on
                    // the tick he just crossed a room edge — he hasn't stood on
                    // the destination's entry tile yet.
                    if !crossed {
                        self.arm_loose_floor_under_feet();
                    }
                }
                // Run out any wobbling loose floors (and the screen jolt);
                // one may give way and drop him this very tick.
                self.advance_loose();
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
        // Bob any loose floor that's wobbling — its own tile shudders (POP's
        // `drawloosed`), done before the Prince so he rides the steady floor
        // line rather than jittering with it.
        self.jiggle_loose_tiles(&mut bytes[..]);
        // Draw the Prince when the room on screen is the one he's in and his
        // animation cursor's current frame resolves to a loaded sprite.
        if self.room_id == self.prince.room {
            if let Some(img) = self
                .prince
                .cursor
                .current()
                .and_then(|f| self.frame_image(f.frame))
            {
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
        let frame =
            hires::render_linear_topdown(&bytes[..], ROOM_WIDTH_BYTES, ROOM_HEIGHT_PX, mode)?;
        Some(self.apply_shake(frame))
    }

    /// Bob each wobbling loose floor in the room on screen by shifting its
    /// own cell down a pixel or two ([`LOOSE_BOB`]) — the dedicated loose-
    /// block shudder POP draws (`FRAMEADV.S drawloosed`), local to the tile.
    /// Operates on the composed hi-res *byte* buffer (`ROOM_WIDTH_BYTES`
    /// wide, row 0 = top); a vertical shift keeps each column's artifact
    /// parity, so no colour swap. The exposed top band gets the `0x80` CLS
    /// fill — a thin gap that reads as the block dropping.
    fn jiggle_loose_tiles(&self, bytes: &mut [u8]) {
        let dy = LOOSE_BOB[usize::try_from(self.frame).unwrap_or(0) % LOOSE_BOB.len()];
        if dy == 0 {
            return;
        }
        let w = usize::from(ROOM_WIDTH_BYTES);
        for arm in self.loose.iter().filter(|l| l.room == self.room_id) {
            let row = arm.row.min(ROOM_HEIGHT - 1);
            let bot = usize::from(BLOCK_BOT_ROW[row]);
            let top = if row == 0 {
                0
            } else {
                usize::from(BLOCK_BOT_ROW[row - 1]) + 1
            };
            let x0 = arm.col * usize::from(CELL_WIDTH_BYTES);
            let x1 = (x0 + usize::from(CELL_WIDTH_BYTES)).min(w);
            // Descend so each source row is still original when it's read.
            for y in (top..=bot).rev() {
                for x in x0..x1 {
                    bytes[y * w + x] = if y >= top + dy {
                        bytes[(y - dy) * w + x]
                    } else {
                        0x80
                    };
                }
            }
        }
    }

    /// Vertical screen-roll this frame, px (0 when steady) — the brief jolt
    /// of an impact (a loose-floor crash or a hard landing), counted down by
    /// [`World::shake`]. The sign flips each frame so the picture shudders
    /// rather than slides. The loose-floor *wobble* is a per-tile bob
    /// ([`World::jiggle_loose_tiles`]), not a whole-screen roll.
    fn shake_dy(&self) -> i32 {
        if self.shake == 0 {
            0
        } else if self.shake % 2 == 0 {
            SHAKE_AMPLITUDE
        } else {
            -SHAKE_AMPLITUDE
        }
    }

    /// Roll `frame` vertically by [`World::shake_dy`], filling the exposed
    /// band with opaque black — the `MOVER.S SHAKEM` jolt after a loose floor
    /// crashes. A no-op while the screen is steady.
    fn apply_shake(&self, frame: Frame) -> Frame {
        let dy = self.shake_dy();
        if dy == 0 {
            return frame;
        }
        let w = usize::try_from(frame.width).unwrap_or(0);
        let h = usize::try_from(frame.height).unwrap_or(0);
        let stride = w * 4;
        let mut pixels = vec![0u8; frame.pixels.len()];
        for px in pixels.chunks_exact_mut(4) {
            px[3] = 255; // opaque black, so the exposed band isn't transparent
        }
        for y in 0..h {
            let Ok(src) = usize::try_from(i32::try_from(y).unwrap_or(0) - dy) else {
                continue; // rolled past the top edge
            };
            if src >= h {
                continue; // rolled past the bottom edge
            }
            pixels[y * stride..y * stride + stride]
                .copy_from_slice(&frame.pixels[src * stride..src * stride + stride]);
        }
        Frame {
            width: frame.width,
            height: frame.height,
            pixels,
        }
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

/// First tile row whose floor line is at or below `feet_y` — where a
/// downward floor scan should start so it never selects a platform the
/// Prince has already fallen past. `ROOM_HEIGHT` when he's below them all.
fn row_below_feet(feet_y: i32) -> usize {
    (0..ROOM_HEIGHT)
        .find(|&r| floor_y(r) >= feet_y)
        .unwrap_or(ROOM_HEIGHT)
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
/// (`col`, `from_row`): `Some((row, feet_y))` for the first floor-bearing
/// tile (or the top of the first solid block), or `None` when the column
/// is open all the way down — in which case the caller falls through to
/// the room below.
fn settle(room: &Room, col: usize, from_row: usize) -> Option<(usize, i32)> {
    for r in from_row..ROOM_HEIGHT {
        let Some(kind) = room.tile_at(col, r).map(|t| t.kind) else {
            continue;
        };
        if tile_has_floor(kind) {
            return Some((r, floor_y(r)));
        }
        if tile_is_solid(kind) {
            let top = r.saturating_sub(1);
            return Some((top, floor_y(top)));
        }
        // The bottom row is the room's floor base (POP convention: it
        // closes the room), so it supports decoration tiles standing on it
        // — pillars, torches, mirrors, arches, … . The only bottom cells
        // that are holes the kid drops through are `Empty` and the
        // explicitly floorless `PanelWithoutFloor`.
        if r == ROOM_HEIGHT - 1 && !matches!(kind, TileKind::Empty | TileKind::PanelWithoutFloor) {
            return Some((r, floor_y(r)));
        }
    }
    None
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

    fn chtabs() -> Vec<Option<ImageTable>> {
        let dir = vendor_root().join("DRAZ").join("I");
        let load = |name: &str| ImageTable::from_file(dir.join(name)).ok();
        // CHTAB1-3 cover the kid's stand / run / fall / climb frames.
        vec![load("IMG.CHTAB1"), load("IMG.CHTAB2"), load("IMG.CHTAB3")]
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
        assert_eq!(settle(&level.rooms[0], 0, 0), Some((1, floor_y(1))));
    }

    #[test]
    fn settle_treats_floorless_bottom_tiles_as_holes() {
        let mut world = landed_world();
        // LV1 room 1 col 5 row 2 is Posts — a decoration on the bottom-row
        // floor base, so it's standable.
        assert_eq!(settle(&world.level.rooms[0], 5, 2), Some((2, floor_y(2))));
        // PanelWithoutFloor at the bottom row is a hole (like Empty).
        world.level.rooms[0].tiles[2 * ROOM_WIDTH + 5] = Tile {
            kind: TileKind::PanelWithoutFloor,
            variant: 0,
            modifier: 0,
        };
        assert_eq!(settle(&world.level.rooms[0], 5, 2), None);
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
        // The run opens with a few windup frames (startrun's chx=0 lead-in),
        // so drive a handful of ticks before checking he's advanced.
        for _ in 0..8 {
            world.tick(InputState {
                right: true,
                ..InputState::default()
            });
        }
        assert!(world.prince.facing_right);
        assert!(world.prince.x > x0, "he advances to the right");
        assert_eq!(
            world.prince.cursor.name(),
            "startrun",
            "he's running the startrun cycle"
        );
        world.tick(InputState::default());
        assert_eq!(
            world.prince.cursor.name(),
            "stand",
            "releasing the arrow returns him to stand"
        );
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
    fn move_h_stops_at_the_right_wall() {
        // Row 2: col 8 is Floor, col 9 is a Block wall. Driving him right
        // must stop his leading edge at the wall (`9 * CELL_W`). Exercised
        // through `move_h` directly so the col-6 loose floor (which would
        // drop him out of the room) doesn't interfere.
        let mut world = landed_world();
        world.prince.row = 2;
        world.prince.x = 8 * CELL_W + CELL_W / 2;
        for _ in 0..30 {
            let room = &world.level.rooms[0];
            world.prince.move_h(5, room);
        }
        assert!(
            world.prince.x + COLLIDE_HALF <= 9 * CELL_W,
            "leading edge {} passed the wall at {}",
            world.prince.x + COLLIDE_HALF,
            9 * CELL_W
        );
    }

    #[test]
    fn move_h_stops_at_the_left_wall() {
        // Row 2: col 4 is Rubble, col 3 is a Block wall. Driving him left
        // must stop his leading edge at the wall's right side (`4 * CELL_W`).
        let mut world = landed_world();
        world.prince.row = 2;
        world.prince.x = 4 * CELL_W + CELL_W / 2;
        for _ in 0..30 {
            let room = &world.level.rooms[0];
            world.prince.move_h(-5, room);
        }
        assert!(
            world.prince.x - COLLIDE_HALF >= 4 * CELL_W,
            "leading edge {} passed the wall at {}",
            world.prince.x - COLLIDE_HALF,
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
        // The cross reconciles his feet against room 5's entry column
        // immediately (col 9 row 0 is a Gate — a floor), so he doesn't hover.
        assert!(world.prince_on_ground());
        assert_eq!(world.prince.feet_y, floor_y(0));
    }

    #[test]
    fn settle_floor_falls_over_a_gap() {
        // Room 1 col 4 row 1 is Empty (a gap) over Rubble at row 2: settling
        // there must start a fall, not leave him hovering.
        let mut world = landed_world();
        world.prince.row = 1;
        world.prince.x = 4 * CELL_W + CELL_W / 2;
        world.prince.on_ground = true;
        let room = &world.level.rooms[0];
        world.prince.settle_floor(room);
        assert!(!world.prince.on_ground, "no floor at row 1 → he falls");
        assert_eq!(world.prince.landing_row, 2);
    }

    #[test]
    fn falling_past_the_bottom_enters_the_room_below() {
        // Force a fall through an open column (as a broken loose floor would
        // leave): `landing_y` beyond the room means nothing caught him, so
        // he drops into room 1's down link (room 2).
        let mut world = landed_world();
        assert_eq!(world.room_id(), 1);
        world.prince.on_ground = false;
        world.prince.vy = 0;
        world.prince.feet_y = ROOM_H - 1;
        world.prince.landing_y = ROOM_H + 1;
        for _ in 0..50 {
            world.tick(InputState::default());
            if world.prince_on_ground() {
                break;
            }
        }
        assert_eq!(world.room_id(), 2, "room 1's down link is room 2");
        assert!(world.prince_on_ground(), "he lands in room 2");
        assert!(world.prince.feet_y < ROOM_H, "feet inside the new room");
    }

    #[test]
    fn row_below_feet_skips_floors_above() {
        // Floor lines: row 0 = 55, row 1 = 118, row 2 = 181.
        assert_eq!(row_below_feet(floor_y(0)), 0);
        assert_eq!(
            row_below_feet(floor_y(0) + 1),
            1,
            "past row 0 → start at row 1"
        );
        assert_eq!(row_below_feet(floor_y(1) + 1), 2);
        assert_eq!(
            row_below_feet(floor_y(2) + 1),
            ROOM_HEIGHT,
            "below them all"
        );
    }

    #[test]
    fn breaking_a_loose_floor_with_support_below_lands_in_room() {
        // Put a loose floor at col 0 row 1, which has a Block at row 2 under
        // it: breaking it drops him onto that block (in-room), not into the
        // room below.
        let mut world = landed_world();
        let loose = *world.level.rooms[0].tile_at(6, 2).unwrap(); // the col-6 loose tile
        world.level.rooms[0].tiles[ROOM_WIDTH] = loose; // col 0 row 1
        world.prince.row = 1;
        world.prince.x = CELL_W / 2;
        world.prince.on_ground = true;

        world.arm_loose_floor_under_feet();
        assert!(
            world.prince_on_ground(),
            "the floor still holds while it wobbles"
        );
        for _ in 0..LOOSE_FALL_DELAY {
            world.advance_loose();
        }
        assert!(
            !world.prince_on_ground(),
            "the floor gave way after the wobble"
        );
        assert_eq!(
            world.prince.landing_row, 1,
            "lands on the block below, in-room"
        );
        assert_eq!(world.prince.landing_y, floor_y(1));
        assert_ne!(world.prince.landing_y, ROOM_H + 1, "not a cross-room fall");
    }

    #[test]
    fn breaking_a_loose_floor_drops_him_to_the_room_below() {
        // Running right from the spawn he drops to row 2, reaches the col-6
        // loose floor, which gives way and drops him into room 2 (LV1 D=2).
        let mut world = landed_world();
        assert_eq!(world.room_id(), 1);
        let mut reached_room_2 = false;
        for _ in 0..300 {
            world.tick(InputState {
                right: true,
                ..InputState::default()
            });
            if world.room_id() == 2 {
                reached_room_2 = true;
                break;
            }
        }
        assert!(
            reached_room_2,
            "the col-6 loose floor should drop him into room 2"
        );
    }

    /// Stand the Prince squarely on the col-6 row-2 loose floor of room 1.
    fn on_loose_floor() -> World {
        let mut world = landed_world();
        world.prince.room = 1;
        world.prince.row = 2;
        world.prince.x = 6 * CELL_W + CELL_W / 2;
        world.prince.feet_y = floor_y(2);
        world.prince.on_ground = true;
        world
    }

    #[test]
    fn a_loose_floor_holds_through_its_wobble_then_gives_way() {
        let mut world = on_loose_floor();
        world.arm_loose_floor_under_feet();
        assert_eq!(world.loose.len(), 1, "his weight armed the loose floor");
        for t in 0..LOOSE_FALL_DELAY {
            assert!(
                world.prince_on_ground(),
                "still standing on tick {t} of the wobble"
            );
            world.advance_loose();
        }
        assert!(!world.prince_on_ground(), "after the wobble it drops him");
        assert!(world.loose.is_empty(), "the armed floor is consumed");
    }

    #[test]
    fn arming_a_wobbling_loose_floor_does_not_restack_it() {
        let mut world = on_loose_floor();
        world.arm_loose_floor_under_feet();
        world.arm_loose_floor_under_feet();
        assert_eq!(world.loose.len(), 1, "re-arming the same cell is a no-op");
    }

    #[test]
    fn a_loose_floor_crumbles_behind_a_prince_who_runs_clear() {
        let mut world = on_loose_floor();
        world.arm_loose_floor_under_feet();
        // He scrambles two tiles along the row before it gives way.
        world.prince.x = 8 * CELL_W + CELL_W / 2;
        for _ in 0..LOOSE_FALL_DELAY {
            world.advance_loose();
        }
        assert!(world.prince_on_ground(), "he stayed on solid ground");
        assert_ne!(
            world.level.rooms[0].tile_at(6, 2).unwrap().kind,
            TileKind::LooseFloor,
            "the loose floor still fell, behind him"
        );
    }

    #[test]
    fn a_crashing_loose_floor_jolts_the_screen() {
        let mut world = on_loose_floor();
        world.arm_loose_floor_under_feet();
        for _ in 0..LOOSE_FALL_DELAY {
            world.advance_loose();
        }
        assert!(world.shake > 0, "the crash starts the screen jolt");
        let jolted = world.render(RenderMode::Monochrome).expect("renders");
        world.shake = 0;
        let steady = world.render(RenderMode::Monochrome).expect("renders");
        assert_ne!(
            steady.pixels, jolted.pixels,
            "the jolt rolls the picture against the steady frame"
        );
    }

    #[test]
    fn a_wobbling_loose_floor_jiggles_only_its_tile() {
        let mut world = on_loose_floor();
        world.shake = 0; // ignore any leftover jolt from the drop-in landing
        world.frame = 2; // LOOSE_BOB[2] = 2 px down, so the bob is visible
        let rest = world.render(RenderMode::Monochrome).expect("renders");
        world.arm_loose_floor_under_feet();
        let wobbling = world.render(RenderMode::Monochrome).expect("renders");
        assert_eq!(
            world.shake_dy(),
            0,
            "the wobble no longer rolls the whole screen"
        );
        assert_ne!(
            rest.pixels, wobbling.pixels,
            "the loose tile itself bobs while it wobbles"
        );
    }

    #[test]
    fn a_hard_landing_thuds_the_screen() {
        let mut world = landed_world();
        // Launch a jump; it lands a few ticks later with downward speed.
        world.tick(InputState {
            up: true,
            ..InputState::default()
        });
        let mut thudded = false;
        for _ in 0..20 {
            world.tick(InputState::default());
            if world.shake > 0 {
                thudded = true;
                break;
            }
        }
        assert!(thudded, "landing the jump jolts the screen");
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
    fn up_at_a_ledge_climbs_to_the_floor_above() {
        let mut world = World::new(load_level1(), dungeon_tables()).with_chtabs(chtabs());
        for _ in 0..40 {
            world.tick(InputState::default());
            if world.prince_on_ground() {
                break;
            }
        }
        // Stand at row 1 col 2 (Floor), facing the row-0 col-3 Floor ledge.
        world.prince.row = 1;
        world.prince.x = 2 * CELL_W + CELL_W / 2;
        world.prince.feet_y = floor_y(1);
        world.prince.facing_right = true;
        world.prince.on_ground = true;

        world.tick(InputState {
            up: true,
            ..InputState::default()
        });
        assert!(world.prince.climb.is_some(), "Up at a ledge starts a climb");

        for _ in 0..30 {
            world.tick(InputState::default());
            if world.prince.climb.is_none() {
                break;
            }
        }
        assert!(world.prince.climb.is_none(), "the climb completes");
        assert_eq!(world.prince.row, 0, "he reaches the upper floor (row 0)");
        assert!(world.prince_on_ground());
        assert_eq!(world.prince.feet_y, floor_y(0));
    }

    #[test]
    fn up_facing_a_solid_top_tile_does_not_climb() {
        // Row 0 col 8 is a Block. Facing it from row 1 must NOT read as a
        // ledge (the top-row `saturating_sub` quirk) — Up falls back to a
        // jump instead of climbing into the wall.
        let mut world = World::new(load_level1(), dungeon_tables()).with_chtabs(chtabs());
        for _ in 0..40 {
            world.tick(InputState::default());
            if world.prince_on_ground() {
                break;
            }
        }
        world.prince.row = 1;
        world.prince.x = 7 * CELL_W + CELL_W / 2; // col 7 → front col 8 (Block)
        world.prince.facing_right = true;
        world.prince.on_ground = true;
        world.prince.feet_y = floor_y(1);
        world.tick(InputState {
            up: true,
            ..InputState::default()
        });
        assert!(
            world.prince.climb.is_none(),
            "no climb into a solid top tile"
        );
        assert!(!world.prince_on_ground(), "Up there jumps instead");
    }

    #[test]
    fn climbing_onto_a_loose_floor_does_not_shatter_on_arrival() {
        let mut world = World::new(load_level1(), dungeon_tables()).with_chtabs(chtabs());
        for _ in 0..40 {
            world.tick(InputState::default());
            if world.prince_on_ground() {
                break;
            }
        }
        // Make the climb target (row 0 col 3) a loose floor.
        let loose = *world.level.rooms[0].tile_at(6, 2).unwrap();
        world.level.rooms[0].tiles[3] = loose;
        world.prince.row = 1;
        world.prince.x = 2 * CELL_W + CELL_W / 2;
        world.prince.facing_right = true;
        world.prince.on_ground = true;
        world.prince.feet_y = floor_y(1);

        world.tick(InputState {
            up: true,
            ..InputState::default()
        });
        assert!(
            world.prince.climb.is_some(),
            "climb starts onto the loose floor"
        );
        for _ in 0..30 {
            world.tick(InputState::default());
            if world.prince.climb.is_none() {
                break;
            }
        }
        // The loose floor must not break on the arrival tick — he's grounded
        // on it, not already falling through.
        assert_eq!(world.prince.row, 0);
        assert!(
            world.prince_on_ground(),
            "loose floor doesn't shatter on the climb-landing tick"
        );
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
        let world = World::new(load_level1(), dungeon_tables()).with_chtabs(chtabs());
        let frame = world.render(RenderMode::NtscColor).expect("renders");
        assert_eq!((frame.width, frame.height), (280, 192));
    }
}
