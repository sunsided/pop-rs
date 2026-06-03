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
//! landing jolts the screen (#96). The drawn figure centres on his logical x
//! per frame, facing-aware, and his wall gap matches what's drawn (#123).
//! Still rough: the falling-rubble mob + rubble tile that round out #96, and
//! the NTSC half-dot / sub-byte figure parity (#121).

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

/// Impact speed (px/tick) below which a fall touchdown plays `softland`
/// rather than `medland`. Our gravity puts a ~1-row drop just under this and
/// a ~2-row drop just over, matching the original soft / medium split in
/// `CTRL.S hitflr` (`CharYVel < $16` soft, `< $21` medium). The fatal tier
/// (`>= $21` → `hardland` + death) needs HP, so a deep fall currently
/// recovers as `medland` (deferred to #96).
const LAND_SOFT_VY: i32 = 22;

/// Full careful-step stride, px — the distance an unobstructed `SHIFT` step
/// covers (`fullstep` spans 14 px). A step against a wall or a floor brink is
/// shortened to the clearance so he edges flush without overshooting
/// (`DoStepfwd` sizes the stride to `GETFWDDIST`).
const CAREFUL_STRIDE: i32 = 14;

// `forward_clearance` only looks one column ahead, which is sufficient only
// while a single stride can't skip past an adjacent cell — i.e. the stride is
// at most half a cell. Enforce that invariant at compile time.
const _: () = assert!(
    CAREFUL_STRIDE <= CELL_W / 2,
    "a careful stride longer than half a cell could step over a one-cell obstacle"
);

/// Careful-step sequences by stride length: `STEP_SEQS[n - 1]` covers ~`n` px
/// (`step1`..`step13`), picked from the forward clearance so the foot lands at
/// the wall / ledge edge.
const STEP_SEQS: [&str; 13] = [
    "step1", "step2", "step3", "step4", "step5", "step6", "step7", "step8", "step9", "step10",
    "step11", "step12", "step13",
];

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

/// Debug snapshot of the Prince's geometry for the `pop play` overlay
/// ([`World::prince_debug`]). Positions are pixels in the 280×192 frame
/// except `col`, which is a tile-column index.
#[derive(Clone, Copy, Debug)]
pub struct PrinceDebug {
    /// Logical centre x, px.
    pub x: i32,
    /// Feet line y, px.
    pub feet_y: i32,
    /// Tile column of his centre (index, not px). Not used by the overlay —
    /// handy for debug printing / external callers.
    pub col: usize,
    /// Wall-collision half-width (current frame), px.
    pub half_w: i32,
    /// Drawn figure's left / right inked edge, px (the actual figure span, not
    /// the wider sprite box).
    pub fig_left: i32,
    pub fig_right: i32,
    /// Cell width, px (for the grid).
    pub cell_w: i32,
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
    /// `true` once a jump arc (`standjump` / `runjump`) has lifted his feet off
    /// the floor — so a `runjump` knows it has actually leapt and should hand
    /// back to the run controller when it touches down (not on its grounded
    /// windup frames).
    mid_jump_air: bool,
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
            mid_jump_air: false,
        }
    }

    /// Begin a jump for `dir`: a running leap (`runjump`) when he's already
    /// running, otherwise a forward standing jump (`standjump`, facing `dir`
    /// if one is held). Mirrors `DoRunjump` / `DoStandjump`, which just select
    /// the sequence — the arc is scripted in the frames, so [`Self::jump_step`]
    /// carries it out. He stays grounded for the windup; the frames lift him.
    fn start_jump(&mut self, dir: i32) {
        self.mid_jump_air = false;
        if self.is_running() {
            self.cursor.play("runjump");
        } else {
            if dir != 0 {
                self.facing_right = dir > 0;
            }
            self.cursor.play("standjump");
        }
    }

    /// A `standjump` / `runjump` arc is playing — its frames script the whole
    /// trajectory, so [`Self::jump_step`] drives him, not gravity.
    fn in_jump(&self) -> bool {
        matches!(self.cursor.name(), "standjump" | "runjump")
    }

    /// Advance one scripted jump frame: move by the frame's `dx` (forward,
    /// facing-signed, wall-stopped via [`Self::glide`]) and `dy` (vertical),
    /// then reconcile with the floor. He rests when the descending arc meets
    /// his row's floor; a `runjump` that has actually leapt hands back to the
    /// run cycle on touchdown. If the arc carries him out over a gap (down to
    /// his launch line with no floor under him), he drops into a `freefall`.
    fn jump_step(&mut self, half_w: i32, room: &Room) {
        let Some(frame) = self.cursor.current() else {
            return;
        };
        let (dx, dy) = (frame.dx, frame.dy);
        let running_jump = self.cursor.name() == "runjump";
        // Horizontal only (wall-stopped) — *not* `glide`, whose floor-follow
        // would snap his feet back each tick and erase the `dy` arc.
        self.slide_x(dx * self.facing_sign(), half_w, room);
        self.feet_y += dy;

        let col = col_of(self.x);
        match settle(room, col, self.row) {
            // A floor at *his own row*, still below his feet → the hop's rise.
            Some((lrow, lfeet)) if lrow == self.row && self.feet_y < lfeet => {
                self.on_ground = false;
                self.mid_jump_air = true;
            }
            // A floor at his own row, reached → touch down on it.
            Some((lrow, lfeet)) if lrow == self.row => {
                self.feet_y = lfeet;
                let leapt = self.mid_jump_air;
                self.mid_jump_air = false;
                self.on_ground = true;
                // A running leap returns to the run controller; a standing
                // jump plays out its own recovery frames into `stand`. (Every
                // arc's `dy` nets to zero, so a `runjump` always returns to its
                // launch line within its aerial frames — landing here or
                // falling through the drop arm below — never hovering in its
                // `loops_to` run-cycle frames forever.)
                if running_jump && leapt {
                    self.cursor.play("startrun");
                }
            }
            // No floor at his row here — only a gap, or a floor a row or more
            // *below* (a drop he's leaping over). While still rising above his
            // launch line, keep arcing; once the arc brings him back down to it
            // he's over the drop, so fall the rest of the way (`settle_floor`
            // aims him at the lower floor, or beyond it). Without this a leap
            // over a tile that has a floor below — Empty-over-Rubble, a step
            // down — would hover through the whole jump and only fall after.
            _ => {
                if self.feet_y < floor_y(self.row) {
                    // Rising above his launch line — the hop has lifted him.
                    self.on_ground = false;
                    self.mid_jump_air = true;
                } else if self.mid_jump_air {
                    // The hop has played and the arc has brought him back down
                    // over the drop → fall the rest of the way (`settle_floor`
                    // aims him at the lower floor, or beyond it).
                    self.mid_jump_air = false;
                    self.settle_floor(room);
                    self.cursor.play("freefall");
                } else {
                    // Grounded wind-up that has shuffled forward over the edge
                    // but hasn't leapt yet — hold the jump; the hop comes next.
                    // (Without this the forward wind-up frames, still at his
                    // launch line, would trip the drop and abort the jump a
                    // half-tile in.)
                    self.on_ground = false;
                }
            }
        }
    }

    /// Touch down from a fall: ground him and play the landing cushion sized to
    /// his impact speed — `softland` for a gentle drop, `medland` for a harder
    /// one (`CTRL.S hitflr`). The fatal `hardland` tier needs HP and is folded
    /// into `medland` for now (#96).
    ///
    /// `softland` holds a near-stand frame and is deliberately *not* an
    /// [`Self::in_transition`] sequence, so the next tick's locomotion resumes
    /// at once (a soft landing doesn't lock control); `medland`'s stagger does
    /// own the cursor. Edge case: a fall through a bottom-row hole with no room
    /// below reaches here with barely-accumulated `vy`, so it reads as a
    /// `softland` and skips the thud — accepted until #96.
    fn land(&mut self) {
        self.on_ground = true;
        let seq = if self.vy < LAND_SOFT_VY {
            "softland"
        } else {
            "medland"
        };
        self.vy = 0;
        self.cursor.play(seq);
    }

    /// One grounded locomotion step in `dir` (`-1` left, `+1` right, `0`
    /// idle), once [`Self::select_locomotion`] has chosen this tick's
    /// sequence. A careful step plays out via [`Self::step_advance`]; a turn /
    /// runstop / runturn transition moves on its own frame displacement and
    /// flips facing on its `turn`-flagged frame; idle holds position; a bare
    /// `SHIFT`+dir starts a careful step; otherwise he steps by the run
    /// sequence's per-frame `chx`. The cursor was advanced for this tick
    /// already, so its `current()` frame is the one to move and draw by.
    fn locomote(&mut self, dir: i32, careful: bool, half_w: i32, room: &Room) {
        if self.in_step() {
            self.step_advance(room);
            return;
        }
        if self.in_transition() {
            // The `turn` / `runturn` sequences reverse facing on their flagged
            // frame; from there each transition's per-frame `dx` carries him in
            // the facing direction. Drive it with `glide` so a backward
            // (negative) frame edges him without flipping his facing.
            if let Some(frame) = self.cursor.current() {
                if frame.turn {
                    self.facing_right = !self.facing_right;
                    // `runturn` loops back onto this flip frame; apply its pivot
                    // recoil (the `dx = -14` carry, now in the new facing) this
                    // tick, then hand off to the run cycle the new way rather
                    // than re-triggering the turn on the loop.
                    if self.cursor.name() == "runturn" {
                        self.glide(frame.dx * self.facing_sign(), half_w, room);
                        self.cursor.play("startrun");
                        return;
                    }
                }
                self.glide(frame.dx * self.facing_sign(), half_w, room);
            }
            return;
        }
        if dir == 0 {
            // Idle — `select_locomotion` already set `stand`; hold position.
            return;
        }
        if careful {
            self.start_careful_step(dir, room);
            return;
        }
        let dx = self.cursor.current().map_or(0, |f| f.dx);
        self.move_h(dx * dir, half_w, room);
    }

    /// Pick this tick's locomotion sequence (and facing) for `dir`, unless a
    /// transition or careful step is mid-play (it owns the cursor until it
    /// chains back to `stand`). Releasing the stick out of a run skids to a stop
    /// (`runstop`), else he stands; `SHIFT`+dir faces the way he'll carefully
    /// step (the step itself is started in [`Self::locomote`], which has the
    /// room to size it); reversing mid-run plays the running turn-around
    /// (`runturn`); pressing *away* from a stand turns in place (`turn`);
    /// otherwise he runs (`startrun`). Mirrors the original `standing` /
    /// `arunning` dispatch. Split from [`Self::locomote`] so the caller can
    /// size collision from the *selected* frame, not the one just advanced off
    /// (#134).
    fn select_locomotion(&mut self, dir: i32, careful: bool) {
        if self.in_transition() || self.in_step() {
            return;
        }
        if dir == 0 {
            if self.is_running() {
                self.cursor.play("runstop");
            } else {
                self.cursor.play("stand");
            }
        } else if careful {
            // The careful step is started in `locomote`; just face it here.
            self.facing_right = dir > 0;
        } else if self.is_running() && (dir > 0) != self.facing_right {
            self.cursor.play("runturn");
        } else if (dir > 0) != self.facing_right {
            self.cursor.play("turn");
        } else {
            self.facing_right = dir > 0;
            self.cursor.play("startrun");
        }
    }

    /// A grounded transition that owns the cursor until it chains back to
    /// `stand` and must not be re-selected or interrupted: the `turn` /
    /// `runstop` / `runturn` transitions and the `medland` landing recovery
    /// (its long stagger plays out before he regains control).
    fn in_transition(&self) -> bool {
        matches!(
            self.cursor.name(),
            "turn" | "runstop" | "medland" | "runturn"
        )
    }

    /// A careful step (`SHIFT`) is playing out: a `step1`..`step13` stride, a
    /// `fullstep` (open field), or `testfoot` peeking over a brink. Owns the
    /// cursor until it chains to `stand`; driven by [`Self::step_advance`].
    fn in_step(&self) -> bool {
        matches!(
            self.cursor.name(),
            "step1"
                | "step2"
                | "step3"
                | "step4"
                | "step5"
                | "step6"
                | "step7"
                | "step8"
                | "step9"
                | "step10"
                | "step11"
                | "step12"
                | "step13"
                | "fullstep"
                | "testfoot"
        )
    }

    /// Forward px a careful step may cover before the next obstacle — a wall
    /// face or a floor brink — capped at a full stride. His leading edge stops
    /// flush at the cur/next cell boundary, leaving his toes at a ledge edge or
    /// against a wall while his body stays on the floored cell. Uses the fixed
    /// [`COLLIDE_HALF`] (not the per-frame figure width) so the stop point
    /// doesn't jitter as the step animation's frame widths vary. (`GETFWDDIST`,
    /// idiomatic.)
    fn forward_clearance(&self, room: &Room) -> i32 {
        let sign = self.facing_sign();
        let cur = i32::try_from(col_of(self.x)).unwrap_or(0);
        let next = cur + sign;
        if !(0..i32::try_from(ROOM_WIDTH).unwrap_or(0)).contains(&next) {
            // Room edge — let a full stride run into the edge crossing.
            return CAREFUL_STRIDE;
        }
        let next_u = usize::try_from(next).unwrap_or(0);
        if is_solid_at(room, next_u, self.row) || !has_floor(room, next_u, self.row) {
            // Wall or brink at the cur/next boundary: edge his leading side to it.
            let lead = self.x + sign * COLLIDE_HALF;
            let boundary = if sign > 0 {
                (cur + 1) * CELL_W
            } else {
                cur * CELL_W
            };
            (sign * (boundary - lead)).clamp(0, CAREFUL_STRIDE)
        } else {
            CAREFUL_STRIDE
        }
    }

    /// Begin a careful step toward `dir`: pick the stride that lands his foot
    /// at the wall / ledge edge (`step1`..`step13` by clearance), or `testfoot`
    /// to peek when he's already at the brink. The sequence then plays out via
    /// [`Self::step_advance`].
    fn start_careful_step(&mut self, dir: i32, room: &Room) {
        self.facing_right = dir > 0;
        let clearance = self.forward_clearance(room);
        if clearance <= 0 {
            // Already flush against a wall / brink — peek over it, don't step.
            self.cursor.play("testfoot");
        } else if clearance >= CAREFUL_STRIDE {
            // Open field: the full 14-px stride (`DoStepfwd`'s open step).
            self.cursor.play("fullstep");
        } else {
            // The branches above handle the extremes, so `clearance` is in
            // `1..=13` here — `step1`..`step13` land his foot at the obstacle.
            // (`try_from` rather than `as`: the latter trips `clippy::pedantic`
            // `cast_sign_loss`; the clamp makes the fallback unreachable.)
            let n = usize::try_from(clearance.clamp(1, 13)).unwrap_or(1);
            self.cursor.play(STEP_SEQS[n - 1]);
        }
    }

    /// Advance a careful step *monotonically* toward the obstacle by the
    /// current frame's stride magnitude, clamped to the remaining clearance.
    /// He never moves backward: a step sequence's backward foot-shuffle frames
    /// (e.g. `testfoot`'s retraction `dx`s) animate but must not drag his
    /// logical position back, or he oscillates against a wall / brink instead
    /// of settling at it. Moves via `slide_x` (no floor-follow), so a step
    /// can't carry him off a ledge. At zero clearance he holds (e.g. peeking).
    fn step_advance(&mut self, room: &Room) {
        let clearance = self.forward_clearance(room).max(0);
        if clearance == 0 {
            return;
        }
        let dx = self.cursor.current().map_or(0, |f| f.dx);
        let step = dx.abs().min(clearance);
        if step != 0 {
            self.slide_x(self.facing_sign() * step, COLLIDE_HALF, room);
        }
    }

    /// He is in the run cycle — the only state a released stick skids out of.
    fn is_running(&self) -> bool {
        self.cursor.name() == "startrun"
    }

    /// `+1` when facing right, `-1` when facing left.
    fn facing_sign(&self) -> i32 {
        if self.facing_right {
            1
        } else {
            -1
        }
    }

    /// Move horizontally by `dx` px against `room`'s tiles, facing the way he
    /// moves: stop flush at a solid wall, follow the floor across flat ground,
    /// and start a fall when he walks off a ledge. The run cycle always steps
    /// forward, so `dx`'s sign is his facing.
    fn move_h(&mut self, dx: i32, half_w: i32, room: &Room) {
        // A zero-`dx` frame (the run cycle's windup) mustn't flip him to
        // facing left — keep the facing the caller set.
        if dx != 0 {
            self.facing_right = dx > 0;
        }
        self.glide(dx, half_w, room);
    }

    /// Translate by `dx` world px, stopping flush at a solid wall on the
    /// leading edge — **without** changing facing or following the floor. The
    /// horizontal half of a move: [`Self::glide`] adds floor-following on top
    /// for grounded motion, while a jump arc drives the vertical itself and
    /// calls this directly so its per-frame `dy` isn't snapped back to the
    /// floor each tick ([`Self::jump_step`]).
    fn slide_x(&mut self, dx: i32, half_w: i32, room: &Room) {
        let dir = dx.signum();
        let mut target_x = self.x + dx;

        // Collide on the drawn figure's *leading edge*, `half_w` from his
        // centre (the current frame's figure half-width, #123) — his body
        // reaches the wall before his centre crosses the cell boundary. If
        // the column under the leading edge is solid, stop the edge flush
        // against the wall. (Off-room columns aren't walls — they're the
        // doorway to a neighbour, handled by `cross_horizontal`.) A zero-`dx`
        // frame has no leading edge and can't hit a wall, so skip the clamp —
        // its `dir == 0` would otherwise take the left-clamp arm and shove him
        // right.
        let lead = target_x + dir * half_w;
        if dir != 0 && (0..ROOM_W).contains(&lead) && is_solid_at(room, col_of(lead), self.row) {
            let wall = i32::try_from(col_of(lead)).unwrap_or(0);
            target_x = if dir > 0 {
                wall * CELL_W - half_w
            } else {
                (wall + 1) * CELL_W + half_w
            };
        }

        self.x = target_x;
    }

    /// Translate by `dx` world px against `room` **without** changing facing,
    /// then follow the floor of the column he lands in: rest on it, or begin a
    /// fall off a ledge. Used by the run step (via [`Self::move_h`]) and by the
    /// turn / runstop transitions, whose frames can edge *backward* (negative
    /// `dx`) without turning him around.
    fn glide(&mut self, dx: i32, half_w: i32, room: &Room) {
        self.slide_x(dx, half_w, room);

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

    /// The wall-collision half-width (px) for the current animation frame —
    /// half the drawn figure's width ([`figure_half_width`]), so the wall gap
    /// matches what's drawn (#123). [`COLLIDE_HALF`] when no sprite is loaded.
    fn current_figure_half_width(&self) -> i32 {
        self.prince
            .cursor
            .current()
            .and_then(|f| self.frame_image(f.frame))
            .map_or(COLLIDE_HALF, figure_half_width)
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

    /// Debug snapshot of the Prince's collision-vs-drawn-figure geometry for
    /// the `pop play` overlay (see [`PrinceDebug`] for field units).
    #[must_use]
    pub fn prince_debug(&self) -> PrinceDebug {
        let half_w = self.current_figure_half_width();
        // The actual inked figure edges, not the wider sprite box: take the
        // figure's byte span and map it through the same mirror `figure_byte_x`
        // / `composite_hires` use, so the magenta overlay lines hug the drawn
        // pixels.
        let (fig_left, fig_right) = self
            .prince
            .cursor
            .current()
            .and_then(|f| self.frame_image(f.frame))
            .and_then(|img| {
                let (lo, hi) = sprite::figure_byte_span(img)?;
                let (lo, hi) = (i32::from(lo), i32::from(hi));
                let w = i32::from(img.width_bytes);
                let bx = figure_byte_x(img, self.prince.x, self.prince.facing_right);
                let (a, b) = if self.prince.facing_right {
                    (w - 1 - hi, w - 1 - lo)
                } else {
                    (lo, hi)
                };
                Some(((bx + a) * 7, (bx + b) * 7 + 6))
            })
            .unwrap_or((self.prince.x - half_w, self.prince.x + half_w));
        PrinceDebug {
            x: self.prince.x,
            feet_y: self.prince.feet_y,
            col: col_of(self.prince.x),
            half_w,
            fig_left,
            fig_right,
            cell_w: CELL_W,
        }
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
                // Blocked edge (no neighbour, or its entry column is closed):
                // nudge him back inside by the constant box half-width, not the
                // per-frame figure `half_w`. These guards only stop him warping
                // off-room; the figure-extent wall stop is `move_h`'s job (#123).
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
                // Blocked left edge — see the right-edge note above: a
                // warp-guard clamp, not the figure-extent wall stop.
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
                self.prince.row = ROOM_HEIGHT - 1;
                self.prince.land();
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
            self.prince.row = ROOM_HEIGHT - 1;
            self.prince.land();
            return;
        }

        // Land once he reaches the floor he's aimed at in the current room.
        if self.prince.vy >= 0 && self.prince.feet_y >= self.prince.landing_y {
            self.prince.feet_y = self.prince.landing_y;
            self.prince.row = self.prince.landing_row;
            self.prince.land();
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
        // Move along an **L-shaped** path over the `climbup` animation's length:
        // rise straight up the wall first, then mount onto the ledge. A
        // straight diagonal would cut the floor corner — he'd visibly glide
        // *through* the ledge tile (the original pulls up vertically, then
        // settles forward onto the ledge).
        let frames = crate::anim::sequence_len("climbup").max(1);
        c.phase += 1;
        if c.phase >= frames {
            self.prince.x = c.to_x;
            self.prince.feet_y = c.to_feet;
            self.prince.row = c.to_row;
            self.prince.on_ground = true;
            self.prince.climb = None;
        } else {
            let p = i32::try_from(c.phase).unwrap_or(0);
            let n = i32::try_from(frames).unwrap_or(1).max(1);
            // Spend the first two-thirds rising at his launch column, the last
            // third stepping across onto the ledge.
            let rise = (n * 2 / 3).max(1);
            if p <= rise {
                self.prince.x = c.from_x;
                self.prince.feet_y = c.from_feet + (c.to_feet - c.from_feet) * p / rise;
            } else {
                self.prince.feet_y = c.to_feet;
                // `.max(1)`: the `p <= rise` branch + `p < frames` guard keep
                // `n - rise >= 1` today, but make the divisor safe regardless.
                self.prince.x = c.from_x + (c.to_x - c.from_x) * (p - rise) / (n - rise).max(1);
            }
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
                    // The cursor was set to `climbup` when the climb began and
                    // advances itself; don't re-play it here, or the tick it
                    // chains `climbup → stand` (as the climb lands) would snap
                    // the cursor back to climbup's first frame.
                    self.climb_step();
                } else if self.prince.in_jump() {
                    // A jump arc is in flight — its frames script his motion
                    // (horizontal + vertical), so drive it instead of gravity.
                    if let Some(room) = self.level.rooms.get(room_idx) {
                        let half_w = self.current_figure_half_width();
                        self.prince.jump_step(half_w, room);
                    } else {
                        // Room id out of range — abandon the arc into a fall
                        // rather than freeze the cursor mid-jump (and, for
                        // `runjump`, loop its run frames forever).
                        self.prince.on_ground = false;
                        self.prince.cursor.play("freefall");
                    }
                } else if !self.prince.on_ground {
                    self.fall_step();
                    // Keep falling as `freefall` — unless this step touched
                    // down, in which case `fall_step` already chose the landing
                    // cushion and must not be clobbered back to `freefall`.
                    if !self.prince.on_ground {
                        self.prince.cursor.play("freefall");
                    }
                } else if input.up && !self.prev.up && !self.prince.in_transition() {
                    // Up grabs a ledge above-in-front, else it launches a jump:
                    // a running leap, or a forward standing jump. A turn /
                    // runstop / medland transition owns the cursor, so Up
                    // can't interrupt it (it falls through to play out below).
                    if self.try_climb() {
                        self.prince.cursor.play("climbup");
                    } else {
                        self.prince.start_jump(walk_dir(input));
                    }
                } else if let Some(room) = self.level.rooms.get(room_idx) {
                    let dir = walk_dir(input);
                    // Choose the sequence first, then size collision from the
                    // frame we're about to move and draw by — not the one we
                    // just advanced off (a stand→run switch changes it) (#134).
                    self.prince.select_locomotion(dir, input.shift);
                    let half_w = self.current_figure_half_width();
                    self.prince.locomote(dir, input.shift, half_w, room);
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
                // Centre the figure (its true span within the box, mirror-
                // aware) on the Prince's column; sit its bottom scan-line on
                // his feet.
                let byte_x = figure_byte_x(img, self.prince.x, self.prince.facing_right);
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

/// Destination byte column for the left edge of a character sprite so its
/// drawn *figure* — its true span within the (often wider) box
/// ([`sprite::figure_byte_span`]), mirrored when `flip` — is centred on his
/// logical pixel `x`. The figure's pixel span is centred and **rounded to the
/// nearest byte the same way for both facings**, so an even-width box no longer
/// swings the figure half a byte left/right depending on which way he faces
/// (the bug behind the facing-dependent wall/ledge overlap). The remaining
/// sub-byte residual is the #121 preshift concern. Centres the whole box when
/// the sprite has no set pixels.
fn figure_byte_x(img: &Image, x: i32, flip: bool) -> i32 {
    let w = i32::from(img.width_bytes);
    let Some((lo, hi)) = sprite::figure_byte_span(img) else {
        return x / 7 - w / 2;
    };
    let (lo, hi) = (i32::from(lo), i32::from(hi));
    // Box byte columns the figure occupies once drawn (mirrored if `flip`).
    let (a, b) = if flip {
        (w - 1 - hi, w - 1 - lo)
    } else {
        (lo, hi)
    };
    // Centre of that span, in half-pixels relative to the box's left edge.
    let fig_centre_2px = (a + b) * 7 + 6;
    // byte_x = round((x - fig_centre_px) / 7), kept exact via half-pixels so
    // the rounding is identical for both facings (no half-byte swing).
    (2 * x - fig_centre_2px + 7).div_euclid(14)
}

/// Half the drawn figure's width in px — the per-frame wall-collision
/// half-extent ([`sprite::figure_byte_span`] width × 7, halved). Facing-
/// symmetric (the mirror doesn't change the width). [`COLLIDE_HALF`] for a
/// blank sprite.
fn figure_half_width(img: &Image) -> i32 {
    match sprite::figure_byte_span(img) {
        // Integer division truncates *down*: an odd-byte-span figure rounds
        // its half-extent toward the centre (a 3-byte span = 21px → 10, not
        // 10.5), so the wall stop is at most a half-pixel generous. Deliberate
        // byte-granular rounding; the sub-byte residual is the #121 concern.
        Some((lo, hi)) => (i32::from(hi) - i32::from(lo) + 1) * 7 / 2,
        None => COLLIDE_HALF,
    }
}

/// `true` if `(col, row)` is a solid tile (a wall the kid can't enter).
fn is_solid_at(room: &Room, col: usize, row: usize) -> bool {
    room.tile_at(col, row)
        .is_some_and(|t| tile_is_solid(t.kind))
}

/// `true` if `(col, row)` carries a floor a standing kid rests on — i.e.
/// [`settle`] finds support at his own `row` (not a lower one). Used to spot
/// the brink of the floor a careful step must not walk off.
fn has_floor(room: &Room, col: usize, row: usize) -> bool {
    settle(room, col, row).is_some_and(|(r, _)| r == row)
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
    fn figure_centres_on_x_for_both_facings() {
        use pop_assets::draz::image_table::Image;
        // 5-byte box; the figure occupies byte cols 1..=3 (odd width → an
        // exact byte centre), sitting off-centre within the box.
        let img = Image {
            width_bytes: 5,
            height: 1,
            bitmap: vec![0, 0b000_0001, 0b000_0001, 0b000_0001, 0],
        };
        let (lo, hi) = sprite::figure_byte_span(&img).expect("non-empty figure");
        let x = 140; // px — byte column 20.
                     // Replicate how `composite_hires` maps a source byte to a drawn column.
        let drawn_centre = |flip: bool| {
            let byte_x = figure_byte_x(&img, x, flip);
            let w = i32::from(img.width_bytes);
            let col = |c: i32| {
                if flip {
                    byte_x + (w - 1 - c)
                } else {
                    byte_x + c
                }
            };
            (col(i32::from(lo)) + col(i32::from(hi))) / 2
        };
        // The drawn figure lands on x's byte column regardless of facing —
        // the asymmetry the box-centring caused is gone.
        assert_eq!(
            drawn_centre(false),
            x / 7,
            "left-facing figure centred on x"
        );
        assert_eq!(
            drawn_centre(true),
            x / 7,
            "right-facing figure centred on x"
        );
    }

    #[test]
    fn even_width_box_does_not_swing_between_facings() {
        use pop_assets::draz::image_table::Image;
        // 4-byte box, figure on byte cols 1..=2 (even span → fractional byte
        // centre at 1.5). The old floored centring drew it half a byte left or
        // right depending on facing; `figure_byte_x` now rounds identically, so
        // the placement is the same whichever way he faces. (Sub-byte residual
        // is the #121 preshift concern.)
        let img = Image {
            width_bytes: 4,
            height: 1,
            bitmap: vec![0, 0b000_0001, 0b000_0001, 0],
        };
        assert_eq!(sprite::figure_byte_span(&img), Some((1, 2)));
        let x = 137; // px — deliberately off the byte grid.
        assert_eq!(
            figure_byte_x(&img, x, false),
            figure_byte_x(&img, x, true),
            "even-width box must not swing the figure between facings"
        );
        // The half-extent is facing-symmetric: 2 bytes = 14px → 7.
        assert_eq!(figure_half_width(&img), 7);
    }

    #[test]
    fn odd_span_half_extent_truncates_down() {
        use pop_assets::draz::image_table::Image;
        // 3-byte span = 21px wide; the half-extent truncates *down* to 10 (from
        // 10.5), making the wall stop at most a half-pixel generous (#123).
        let img = Image {
            width_bytes: 3,
            height: 1,
            bitmap: vec![0b000_0001, 0b000_0001, 0b000_0001],
        };
        assert_eq!(sprite::figure_byte_span(&img), Some((0, 2)));
        assert_eq!(figure_half_width(&img), 10);
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
        // Releasing the arrow now skids to a halt (`runstop`) before standing,
        // rather than snapping straight to `stand` (#95 slice 2).
        world.tick(InputState::default());
        assert_eq!(
            world.prince.cursor.name(),
            "runstop",
            "releasing mid-run skids to a stop"
        );
        // The skid chains to stand; run out its frames (+ the chain tick).
        for _ in 0..crate::anim::sequence_len("runstop") + 2 {
            world.tick(InputState::default());
        }
        assert_eq!(
            world.prince.cursor.name(),
            "stand",
            "the skid settles back to stand"
        );
    }

    #[test]
    fn pressing_away_from_facing_turns_in_place() {
        let mut world = landed_world();
        assert!(world.prince.facing_right, "lv1 spawns facing right");
        let x0 = world.prince.x;
        // Press left while facing right: he turns in place rather than
        // instantly bolting left.
        world.tick(InputState {
            left: true,
            ..InputState::default()
        });
        assert_eq!(
            world.prince.cursor.name(),
            "turn",
            "an opposite press starts a turn"
        );
        // The turn flips his facing and chains to stand, then he runs left.
        // Run out the turn's frames plus a margin for the chain + run start.
        for _ in 0..crate::anim::sequence_len("turn") + 4 {
            world.tick(InputState {
                left: true,
                ..InputState::default()
            });
        }
        assert!(!world.prince.facing_right, "he now faces left");
        assert_eq!(
            world.prince.cursor.name(),
            "startrun",
            "and then runs the new way"
        );
        assert!(
            (world.prince.x - x0).abs() < 2 * CELL_W,
            "the turn shuffles in place, it doesn't carry him across the room"
        );
    }

    #[test]
    fn releasing_mid_run_skids_forward_then_stands() {
        let mut world = landed_world();
        for _ in 0..8 {
            world.tick(InputState {
                right: true,
                ..InputState::default()
            });
        }
        assert_eq!(world.prince.cursor.name(), "startrun");
        let x_release = world.prince.x;
        world.tick(InputState::default());
        assert_eq!(world.prince.cursor.name(), "runstop", "the release skids");
        for _ in 0..crate::anim::sequence_len("runstop") + 2 {
            world.tick(InputState::default());
        }
        assert_eq!(
            world.prince.cursor.name(),
            "stand",
            "the skid settles to stand"
        );
        assert!(
            world.prince.x >= x_release,
            "the skid carried him forward, not backward"
        );
    }

    #[test]
    fn reversing_mid_run_plays_runturn_then_runs_the_new_way() {
        let mut world = landed_world();
        for _ in 0..8 {
            world.tick(InputState {
                right: true,
                ..InputState::default()
            });
        }
        assert_eq!(world.prince.cursor.name(), "startrun");
        assert!(world.prince.facing_right);
        // Pressing the opposite way now plays the running turn-around rather
        // than flipping his facing instantly (#95 slice 4).
        world.tick(InputState {
            left: true,
            ..InputState::default()
        });
        assert_eq!(
            world.prince.cursor.name(),
            "runturn",
            "reversing mid-run plays the turn-around"
        );
        // It flips his facing and hands back to the run cycle the new way.
        let mut faced_left_running = false;
        for _ in 0..20 {
            world.tick(InputState {
                left: true,
                ..InputState::default()
            });
            if !world.prince.facing_right && world.prince.cursor.name() == "startrun" {
                faced_left_running = true;
                break;
            }
        }
        assert!(faced_left_running, "he turns around and runs the new way");
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
            world.prince.move_h(5, COLLIDE_HALF, room);
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
            world.prince.move_h(-5, COLLIDE_HALF, room);
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
        // Drop him a couple of rows: a real fall lands with downward speed and
        // jolts the screen (a gentle jump arc no longer thuds — only a fall).
        world.prince.row = 0;
        world.prince.feet_y = floor_y(0);
        world.prince.on_ground = false;
        world.prince.vy = 0;
        world.prince.landing_row = 2;
        world.prince.landing_y = floor_y(2);
        let mut thudded = false;
        for _ in 0..40 {
            world.tick(InputState::default());
            if world.shake > 0 {
                thudded = true;
                break;
            }
        }
        assert!(thudded, "a multi-row fall jolts the screen on impact");
    }

    #[test]
    fn up_arrow_jumps_and_lands_back() {
        let mut world = landed_world();
        let floor = world.prince.feet_y;
        let x0 = world.prince.x;
        // Up from a stand launches a forward standing jump. Its windup is on
        // the ground, so he isn't airborne the instant Up is pressed.
        world.tick(InputState {
            up: true,
            ..InputState::default()
        });
        assert_eq!(
            world.prince.cursor.name(),
            "standjump",
            "Up launches a standing jump"
        );
        let mut rose = false;
        for _ in 0..30 {
            world.tick(InputState::default());
            if world.prince.feet_y < floor {
                rose = true;
            }
            if world.prince.cursor.name() == "stand" {
                break;
            }
        }
        assert!(rose, "the jump's hop rises off the floor");
        assert!(world.prince_on_ground(), "and lands back down");
        assert_eq!(world.prince.feet_y, floor, "on the same floor line");
        assert!(world.prince.x > x0, "the standing jump carries him forward");
        assert_eq!(world.prince.cursor.name(), "stand", "and recovers to stand");
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
        assert_eq!(
            world.prince.cursor.name(),
            "standjump",
            "Up there launches a jump instead"
        );
    }

    #[test]
    fn up_while_running_leaps_and_keeps_running() {
        let mut world = landed_world();
        // Flat floor to leap across (row 2), with the row above cleared so Up
        // launches a jump rather than grabbing a ledge.
        let floor_tile = *world.level.rooms[0].tile_at(8, 2).unwrap();
        let hole = Tile {
            kind: TileKind::PanelWithoutFloor,
            variant: 0,
            modifier: 0,
        };
        for c in 0..ROOM_WIDTH {
            world.level.rooms[0].tiles[2 * ROOM_WIDTH + c] = floor_tile;
            world.level.rooms[0].tiles[ROOM_WIDTH + c] = hole;
        }
        world.prince.row = 2;
        world.prince.x = CELL_W / 2;
        world.prince.feet_y = floor_y(2);
        world.prince.on_ground = true;
        world.prince.facing_right = true;
        // Get him running, then leap.
        for _ in 0..8 {
            world.tick(InputState {
                right: true,
                ..InputState::default()
            });
        }
        assert_eq!(world.prince.cursor.name(), "startrun");
        let x_launch = world.prince.x;
        world.tick(InputState {
            up: true,
            right: true,
            ..InputState::default()
        });
        assert_eq!(
            world.prince.cursor.name(),
            "runjump",
            "Up while running launches a running leap"
        );
        // Hold the direction; the leap lands and hands back to the run cycle.
        let mut grounded_running = false;
        for _ in 0..20 {
            world.tick(InputState {
                right: true,
                ..InputState::default()
            });
            if world.prince_on_ground() && world.prince.cursor.name() == "startrun" {
                grounded_running = true;
                break;
            }
        }
        assert!(grounded_running, "the leap lands and he resumes running");
        assert!(world.prince.x > x_launch, "the leap carried him forward");
    }

    #[test]
    fn fall_landing_cushion_scales_with_height() {
        // The landing animation reflects how far he dropped: a short fall lands
        // soft, a deeper one needs the medium recovery. (`hardland` + death is
        // deferred with HP, #96, so a deep fall currently folds into medland.)
        let land_after_drop = |rows: usize| -> &'static str {
            let mut world = landed_world();
            world.prince.row = 0;
            world.prince.feet_y = floor_y(0);
            world.prince.on_ground = false;
            world.prince.vy = 0;
            world.prince.landing_row = rows;
            world.prince.landing_y = floor_y(rows);
            for _ in 0..40 {
                world.tick(InputState::default());
                if world.prince_on_ground() {
                    break;
                }
            }
            world.prince.cursor.name()
        };
        assert_eq!(
            land_after_drop(1),
            "softland",
            "a one-row drop lands softly"
        );
        assert_eq!(
            land_after_drop(2),
            "medland",
            "a deeper drop needs the medium recovery"
        );
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
        let mut ran = false;
        for _ in 0..10 {
            world.tick(InputState {
                shift: true,
                right: true,
                ..InputState::default()
            });
            if world.prince.cursor.name() == "startrun" {
                ran = true;
            }
        }
        assert!(world.prince.facing_right);
        assert!(world.prince.x > x0, "careful steps edge him forward");
        assert!(
            !ran,
            "SHIFT walks carefully (step sequences), never breaks into a run"
        );
    }

    #[test]
    fn careful_step_stops_at_a_ledge_brink() {
        // Stand at row 1 col 3 — the last floored column before the col-4 gap —
        // facing the gap. Careful-stepping must edge up to the brink and stop,
        // never carrying him off (the whole point of SHIFT).
        let mut world = landed_world();
        world.prince.row = 1;
        world.prince.x = 3 * CELL_W + CELL_W / 2;
        world.prince.feet_y = floor_y(1);
        world.prince.on_ground = true;
        world.prince.facing_right = true;
        for _ in 0..20 {
            world.tick(InputState {
                shift: true,
                right: true,
                ..InputState::default()
            });
        }
        assert!(
            world.prince_on_ground(),
            "careful steps never walk him off the ledge"
        );
        assert_eq!(world.prince.row, 1, "he stays on the upper floor");
        assert!(
            world.prince.x <= 4 * CELL_W,
            "he stops at the brink rather than crossing it"
        );
    }

    #[test]
    fn render_returns_full_size_frame() {
        let world = World::new(load_level1(), dungeon_tables()).with_chtabs(chtabs());
        let frame = world.render(RenderMode::NtscColor).expect("renders");
        assert_eq!((frame.width, frame.height), (280, 192));
    }

    /// Build a grounded world with a custom row-2 layout (one tile kind per
    /// column-2 cell), the Prince standing at col 0 facing right.
    fn world_with_row2(tiles: &[TileKind]) -> World {
        let mut world = World::new(load_level1(), dungeon_tables()).with_chtabs(chtabs());
        for _ in 0..40 {
            world.tick(InputState::default());
            if world.prince_on_ground() {
                break;
            }
        }
        let floor = *world.level.rooms[0].tile_at(8, 2).unwrap();
        let block = *world.level.rooms[0].tile_at(8, 0).unwrap();
        for (c, &kind) in tiles.iter().enumerate() {
            let t = match kind {
                TileKind::PanelWithoutFloor => Tile {
                    kind: TileKind::PanelWithoutFloor,
                    variant: 0,
                    modifier: 0,
                },
                k if tile_is_solid(k) => block,
                _ => floor,
            };
            world.level.rooms[0].tiles[2 * ROOM_WIDTH + c] = t;
        }
        world.prince.room = 1;
        world.prince.row = 2;
        world.prince.x = CELL_W / 2;
        world.prince.feet_y = floor_y(2);
        world.prince.on_ground = true;
        world.prince.facing_right = true;
        world
    }

    #[test]
    fn careful_step_edges_up_to_a_ledge_not_short_of_it() {
        use TileKind::{Floor, PanelWithoutFloor as Gap};
        // Floored through col 5, a drop from col 6 on; brink at col5/col6.
        let mut world =
            world_with_row2(&[Floor, Floor, Floor, Floor, Floor, Floor, Gap, Gap, Gap, Gap]);
        for _ in 0..300 {
            world.tick(InputState {
                right: true,
                shift: true,
                ..InputState::default()
            });
        }
        let brink = 6 * CELL_W;
        // His leading edge (toes) reaches the brink — he isn't left a body-width
        // short of the drop as the un-closed step did.
        assert!(
            world.prince.x + COLLIDE_HALF >= brink - 1,
            "careful step stopped short of the ledge: x={}, brink={brink}",
            world.prince.x,
        );
        // But his centre stays on the floored cell, so he doesn't perch over the
        // gap (and never steps off).
        assert!(world.prince.x < brink, "his centre stays on the floor side");
        assert!(
            world.prince_on_ground(),
            "a careful step never walks him off the ledge"
        );
        assert_eq!(world.prince.row, 2);
    }

    #[test]
    fn careful_step_advances_monotonically_to_a_ledge() {
        use TileKind::{Floor, PanelWithoutFloor as Gap};
        let mut world =
            world_with_row2(&[Floor, Floor, Floor, Floor, Floor, Floor, Gap, Gap, Gap, Gap]);
        // Holding SHIFT+right, his x must never move backward — the careful
        // step settles at the brink instead of oscillating against it.
        let mut prev = world.prince.x;
        for _ in 0..300 {
            world.tick(InputState {
                right: true,
                shift: true,
                ..InputState::default()
            });
            assert!(
                world.prince.x >= prev,
                "careful step moved backward: {} -> {}",
                prev,
                world.prince.x
            );
            prev = world.prince.x;
        }
        assert!(world.prince_on_ground());
    }

    #[test]
    fn careful_step_stops_out_of_a_wall() {
        use TileKind::{Block, Floor};
        // Floored, with a Block wall at col 6.
        let mut world = world_with_row2(&[
            Floor, Floor, Floor, Floor, Floor, Floor, Block, Floor, Floor, Floor,
        ]);
        for _ in 0..300 {
            world.tick(InputState {
                right: true,
                shift: true,
                ..InputState::default()
            });
        }
        // He stops in the cell before the wall — centre never enters col 6.
        assert!(
            col_of(world.prince.x) < 6,
            "careful step entered the wall column: x={}",
            world.prince.x
        );
        assert!(world.prince_on_ground());
        assert_eq!(world.prince.row, 2);
    }

    #[test]
    fn careful_step_left_edges_up_to_a_ledge_without_stepping_off() {
        use TileKind::{Floor, PanelWithoutFloor as Gap};
        // Gap at cols 0..=2, floored from col 3 on; brink at col2/col3.
        let mut world = world_with_row2(&[
            Gap, Gap, Gap, Floor, Floor, Floor, Floor, Floor, Floor, Floor,
        ]);
        world.prince.x = 7 * CELL_W + CELL_W / 2;
        world.prince.facing_right = false;
        let mut prev = world.prince.x;
        for _ in 0..300 {
            world.tick(InputState {
                left: true,
                shift: true,
                ..InputState::default()
            });
            assert!(
                world.prince.x <= prev,
                "careful step moved backward (rightward): {prev} -> {}",
                world.prince.x
            );
            prev = world.prince.x;
        }
        let brink = 3 * CELL_W; // col2/col3 boundary
                                // Toes reach the brink, centre stays on the floored col-3 side, and he
                                // never steps off into the gap.
        assert!(
            world.prince.x - COLLIDE_HALF <= brink + 1,
            "left careful step stopped short of the ledge: x={}",
            world.prince.x
        );
        assert!(world.prince.x > brink, "his centre stays on the floor side");
        assert!(world.prince_on_ground(), "never steps off the left ledge");
        assert_eq!(world.prince.row, 2);
    }

    #[test]
    fn jump_over_a_step_down_falls_to_the_lower_floor() {
        // Row 1 floored through col 3, then a drop (Empty over a row-2 floor) —
        // a step-down, not a clean gap. A forward jump off the edge must fall to
        // the lower floor, not hover at his launch height through the whole arc.
        let mut world = World::new(load_level1(), dungeon_tables()).with_chtabs(chtabs());
        for _ in 0..40 {
            world.tick(InputState::default());
            if world.prince_on_ground() {
                break;
            }
        }
        let floor = *world.level.rooms[0].tile_at(8, 2).unwrap();
        let gap = Tile {
            kind: TileKind::PanelWithoutFloor,
            variant: 0,
            modifier: 0,
        };
        for c in 0..ROOM_WIDTH {
            world.level.rooms[0].tiles[c] = gap; // clear row 0 so `up` jumps, not climbs
            world.level.rooms[0].tiles[ROOM_WIDTH + c] = if c <= 3 { floor } else { gap };
            world.level.rooms[0].tiles[2 * ROOM_WIDTH + c] = floor;
        }
        world.prince.room = 1;
        world.prince.row = 1;
        let launch_x = 3 * CELL_W + CELL_W / 2;
        world.prince.x = launch_x;
        world.prince.feet_y = floor_y(1);
        world.prince.on_ground = true;
        world.prince.facing_right = true;
        world.tick(InputState {
            up: true,
            right: true,
            ..InputState::default()
        });
        let mut landed_below = false;
        for _ in 0..40 {
            world.tick(InputState::default());
            if world.prince_on_ground() && world.prince.row == 2 {
                landed_below = true;
                break;
            }
        }
        assert!(
            landed_below,
            "a jump over a step-down should fall to the lower floor"
        );
        // And he should leap the full arc forward first, not abort a half-tile
        // in during the wind-up.
        assert!(
            world.prince.x >= launch_x + CELL_W,
            "the jump barely advanced ({} from {launch_x}) — it aborted instead of leaping",
            world.prince.x
        );
    }

    #[test]
    fn climb_rises_before_mounting_no_diagonal_glide() {
        let mut world = World::new(load_level1(), dungeon_tables()).with_chtabs(chtabs());
        for _ in 0..40 {
            world.tick(InputState::default());
            if world.prince_on_ground() {
                break;
            }
        }
        // Stand row 1 col 2 facing the row-0 col-3 Floor ledge.
        world.prince.row = 1;
        world.prince.x = 2 * CELL_W + CELL_W / 2;
        world.prince.feet_y = floor_y(1);
        world.prince.facing_right = true;
        world.prince.on_ground = true;
        let from_x = world.prince.x;
        world.tick(InputState {
            up: true,
            ..InputState::default()
        });
        // The path is L-shaped: he must not drift sideways until he has risen
        // (nearly) to the ledge floor — a diagonal would cut the floor corner.
        let mut sideways_at_feet = None;
        for _ in 0..30 {
            world.tick(InputState::default());
            if world.prince.x != from_x && sideways_at_feet.is_none() {
                sideways_at_feet = Some(world.prince.feet_y);
            }
            if world.prince.climb.is_none() {
                break;
            }
        }
        // The climb must actually move him sideways (onto the front ledge), or
        // the L-path invariant below would pass vacuously.
        let feet = sideways_at_feet.expect("climb never stepped sideways — L-path not exercised");
        assert!(
            feet <= floor_y(0) + VERT_DIST,
            "he moved sideways mid-climb (diagonal glide): feet={feet}, ledge={}",
            floor_y(0)
        );
        assert_eq!(world.prince.row, 0, "he reaches the upper floor");
        assert!(world.prince_on_ground());
        assert_eq!(world.prince.feet_y, floor_y(0));
    }
}
