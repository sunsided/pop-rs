//! Character animation sequences — POP's `SEQTABLE` / `FRAMEDEF` decoded
//! into a typed list of motions the editor can preview (#89).
//!
//! Two vendored tables drive every character animation:
//!
//! * **`FRAMEDEF.S`** — one 5-byte record per frame:
//!   `(Fimage, Fsword, Fdx, Fdy, Fcheck)`. `Fimage`/`Fsword` pick a CHTAB
//!   sprite via `decodeim` ([`FrameDef::sprite`], `CTRLSUBS.S`); `Fdx`/`Fdy`
//!   are the per-frame blit offset.
//! * **`SEQTABLE.S`** — one byte stream per motion: positive bytes are
//!   frame ids (one displayed per tick), interleaved with negative
//!   *opcodes* (`chx`/`chy`/`act`/`goto`/`aboutface`/…). The `ANIMCHAR`
//!   interpreter (`COLL.S`) walks a stream until it hits a frame, draws it,
//!   and stops for that tick; `goto` chains/loops the streams.
//!
//! Both tables are baked in with `include_str!` and parsed once (lazily),
//! so the editor preview and any later runtime use share one decode. The
//! opcode set and operand counts are transcribed from `ANIMCHAR`
//! (`COLL.S`): `chx`/`chy`/`act`/`tap`/`effect` take one operand,
//! `setfall` two, `goto`/`ifwtless` a `dw` target, the rest none; a frame
//! byte ends the tick.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

const FRAMEDEF_SRC: &str =
    include_str!("../../../vendor/pop-apple2/01 POP Source/Source/FRAMEDEF.S");
const SEQTABLE_SRC: &str =
    include_str!("../../../vendor/pop-apple2/01 POP Source/Source/SEQTABLE.S");

/// One `FRAMEDEF.S` record: the sprite + offsets for a single frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameDef {
    /// `Fimage` — image index (bit 7 = high CHTAB bank select).
    pub image: u8,
    /// `Fsword` — sword image; bits 6/7 also select the CHTAB bank.
    pub sword: u8,
    /// `Fdx` — per-frame horizontal blit offset, px (facing-relative).
    pub dx: i8,
    /// `Fdy` — per-frame vertical blit offset, px.
    pub dy: i8,
    /// `Fcheck` — collision / draw flags.
    pub check: u8,
}

/// A resolved CHTAB sprite: which table and which 0-based image in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpriteRef {
    /// CHTAB number, 1-based (`1` = `IMG.CHTAB1`).
    pub chtab: u8,
    /// 0-based index into that table's `images`.
    pub index: usize,
}

impl FrameDef {
    /// `decodeim` (`CTRLSUBS.S`): pick the CHTAB sprite for this frame.
    ///
    /// `table = (Fimage.7 << 2) | (Fsword.7 << 1) | Fsword.6`,
    /// `image = Fimage & $7f` (POP image tables are 1-based, so the 0-based
    /// index is `image - 1`). `None` for a blank frame (`image == 0`).
    #[must_use]
    pub fn sprite(self) -> Option<SpriteRef> {
        let image = self.image & 0x7f;
        if image == 0 {
            return None;
        }
        let table = ((self.image >> 7) << 2) | ((self.sword >> 7) << 1) | ((self.sword >> 6) & 1);
        Some(SpriteRef {
            chtab: table + 1,
            index: usize::from(image - 1),
        })
    }
}

/// One step of a sequence: the frame to draw plus the motion / state the
/// interpreter applied reaching it (since the previous frame).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnimFrame {
    /// POP frame id (1-based `FRAMEDEF` index).
    pub frame: u8,
    /// Forward motion since the previous frame (`chx` sum), px.
    pub dx: i32,
    /// Vertical motion since the previous frame (`chy` sum), px.
    pub dy: i32,
    /// `CharAction` set by an `act` opcode in effect at this frame.
    pub action: Option<u8>,
    /// An `aboutface` (turn) happened just before this frame.
    pub turn: bool,
}

/// A named animation: the frames it plays and how it ends (self-loop or a
/// chain into another sequence).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnimSequence {
    /// Sequence id (the `SEQTABLE` `:N dw …` index).
    pub id: u8,
    /// Label name (`stand`, `startrun`, `climbup`, …).
    pub name: String,
    /// Frames in play order.
    pub frames: Vec<AnimFrame>,
    /// Frame index a closing `goto` loops back to, if it loops on itself.
    pub loops_to: Option<usize>,
    /// Sequence name a closing `goto` chains into, if not a self-loop.
    pub chains_to: Option<String>,
}

/// All `FRAMEDEF` records, 0-based (`frame_defs()[n]` is frame `n + 1`).
#[must_use]
pub fn frame_defs() -> &'static [FrameDef] {
    static DEFS: OnceLock<Vec<FrameDef>> = OnceLock::new();
    DEFS.get_or_init(parse_framedef)
}

/// The `FRAMEDEF` record for frame id `frame` (1-based), if defined.
#[must_use]
pub fn frame_def(frame: u8) -> Option<FrameDef> {
    frame_defs()
        .get(usize::from(frame).checked_sub(1)?)
        .copied()
}

/// The CHTAB sprite for frame id `frame` (1-based), if any.
#[must_use]
pub fn frame_sprite(frame: u8) -> Option<SpriteRef> {
    frame_def(frame)?.sprite()
}

/// All decoded animation sequences, in `SEQTABLE` id order.
#[must_use]
pub fn animations() -> &'static [AnimSequence] {
    static SEQS: OnceLock<Vec<AnimSequence>> = OnceLock::new();
    SEQS.get_or_init(parse_sequences)
}

// ---------------------------------------------------------------------------
// FRAMEDEF.S
// ---------------------------------------------------------------------------

fn parse_framedef() -> Vec<FrameDef> {
    let mut defs: Vec<FrameDef> = Vec::new();
    let mut in_table = false;
    for raw in FRAMEDEF_SRC.lines() {
        let line = strip_comment(raw);
        let trimmed = line.trim();
        if trimmed == "Fdef" {
            in_table = true;
            continue;
        }
        if !in_table {
            continue;
        }
        // The main frame table runs until the first `ds` padding, which
        // begins the `altset1`/`altset2`/`swordtab` alternates (each its own
        // `:1`-based table in a different format).
        if trimmed.starts_with("ds ") {
            break;
        }
        // Entries look like `:N db a,b,c,d,e`.
        let Some(rest) = trimmed.strip_prefix(':') else {
            continue;
        };
        let mut it = rest.split_whitespace();
        let Some(idx) = it.next().and_then(|n| n.parse::<usize>().ok()) else {
            continue;
        };
        if it.next() != Some("db") {
            continue;
        }
        let Some(ops) = it.next() else { continue };
        let vals: Vec<i32> = ops.split(',').map(eval).collect();
        if vals.len() < 5 {
            continue;
        }
        if defs.len() < idx {
            defs.resize(
                idx,
                FrameDef {
                    image: 0,
                    sword: 0,
                    dx: 0,
                    dy: 0,
                    check: 0,
                },
            );
        }
        defs[idx - 1] = FrameDef {
            image: u8::try_from(vals[0] & 0xff).unwrap_or(0),
            sword: u8::try_from(vals[1] & 0xff).unwrap_or(0),
            // `Fdx`/`Fdy` are signed bytes. The data writes them decimal
            // (`-2`), but reinterpret the low byte as two's complement so a
            // raw `$fe` would also decode to `-2` rather than failing to `0`.
            dx: byte_i8(vals[2]),
            dy: byte_i8(vals[3]),
            check: u8::try_from(vals[4] & 0xff).unwrap_or(0),
        };
    }
    defs
}

// ---------------------------------------------------------------------------
// SEQTABLE.S
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Opcode {
    Goto,
    Aboutface,
    Up,
    Down,
    Chx,
    Chy,
    Act,
    Setfall,
    Ifwtless,
    Die,
    Jaru,
    Jard,
    Effect,
    Tap,
    Nextlevel,
}

fn opcode(name: &str) -> Option<Opcode> {
    Some(match name {
        "goto" => Opcode::Goto,
        "aboutface" => Opcode::Aboutface,
        "up" => Opcode::Up,
        "down" => Opcode::Down,
        "chx" => Opcode::Chx,
        "chy" => Opcode::Chy,
        "act" => Opcode::Act,
        "setfall" => Opcode::Setfall,
        "ifwtless" => Opcode::Ifwtless,
        "die" => Opcode::Die,
        "jaru" => Opcode::Jaru,
        "jard" => Opcode::Jard,
        "effect" => Opcode::Effect,
        "tap" => Opcode::Tap,
        "nextlevel" => Opcode::Nextlevel,
        _ => return None,
    })
}

#[derive(Clone, Debug)]
enum Tok {
    /// A frame id or an opcode's integer operand.
    Int(i32),
    /// An opcode.
    Op(Opcode),
    /// A `dw LABEL` reference (a `goto` / `ifwtless` target).
    Word(String),
}

/// Local-label scope separator (cannot occur in a Merlin label).
const SCOPE: char = '\u{1}';

/// Assembled `SEQTABLE` token stream + the label / id maps it resolves to.
struct Program {
    tokens: Vec<Tok>,
    /// Global label -> token index.
    labels: HashMap<String, usize>,
    /// `scope\u{1}:local` -> token index.
    locals: HashMap<String, usize>,
    /// Globals in source order, for resolving a local from a `goto` site.
    globals: Vec<(usize, String)>,
    /// Sequence id -> entry label, from the `:N dw NAME` index block.
    id_table: Vec<(u8, String)>,
}

fn assemble() -> Program {
    let mut p = Program {
        tokens: Vec::new(),
        labels: HashMap::new(),
        locals: HashMap::new(),
        globals: Vec::new(),
        id_table: Vec::new(),
    };
    let mut cur_global = String::new();
    let mut bodies_started = false;

    for raw in SEQTABLE_SRC.lines() {
        let line = strip_comment(raw);
        if line.trim().is_empty() || line.trim_start().starts_with('*') {
            continue;
        }
        let has_label = !line.starts_with([' ', '\t']);
        let mut words = line.split_whitespace();
        let w0 = words.next().unwrap_or("");
        let (label, dir, ops) = if has_label {
            (Some(w0), words.next(), words.next())
        } else {
            (None, Some(w0), words.next())
        };

        // `name = value` constant (opcode definitions, org) — opcode set is
        // hard-coded from ANIMCHAR, so skip.
        if dir == Some("=") {
            continue;
        }

        // The `:N dw NAME` sequence-id index block, before the first body.
        if !bodies_started {
            if let Some(lbl) = label {
                if dir == Some("dw") {
                    if let Some(num) = lbl.strip_prefix(':').and_then(|n| n.parse::<u8>().ok()) {
                        if let Some(name) = ops {
                            p.id_table.push((num, name.to_string()));
                        }
                        continue;
                    }
                }
            }
        }

        if let Some(lbl) = label {
            let pos = p.tokens.len();
            if lbl.starts_with(':') || lbl.starts_with(']') {
                p.locals.insert(format!("{cur_global}{SCOPE}{lbl}"), pos);
            } else {
                cur_global = lbl.to_string();
                p.labels.insert(lbl.to_string(), pos);
                p.globals.push((pos, lbl.to_string()));
                bodies_started = true;
            }
        }

        match dir {
            Some("db") => {
                if let Some(ops) = ops {
                    for operand in ops.split(',') {
                        p.tokens.push(match opcode(operand) {
                            Some(op) => Tok::Op(op),
                            None => Tok::Int(eval(operand)),
                        });
                    }
                }
            }
            Some("dw") => {
                if let Some(ops) = ops {
                    p.tokens.push(Tok::Word(ops.to_string()));
                }
            }
            _ => {} // bare label, org, ds, … — nothing to emit.
        }
    }
    p
}

impl Program {
    /// Resolve a `goto`/`ifwtless` target to a token index. Globals by name;
    /// a `:local` against the global label enclosing the jump site.
    fn resolve(&self, target: &str, from: usize) -> Option<usize> {
        if target.starts_with(':') || target.starts_with(']') {
            let scope = self
                .globals
                .iter()
                .rev()
                .find(|(pos, _)| *pos <= from)
                .map_or("", |(_, name)| name.as_str());
            self.locals.get(&format!("{scope}{SCOPE}{target}")).copied()
        } else {
            self.labels.get(target).copied()
        }
    }

    fn int_at(&self, pos: usize) -> i32 {
        match self.tokens.get(pos) {
            Some(Tok::Int(n)) => *n,
            _ => 0,
        }
    }

    fn word_at(&self, pos: usize) -> Option<&str> {
        match self.tokens.get(pos) {
            Some(Tok::Word(s)) => Some(s),
            _ => None,
        }
    }
}

fn parse_sequences() -> Vec<AnimSequence> {
    let prog = assemble();
    let id_names: HashSet<&str> = prog.id_table.iter().map(|(_, n)| n.as_str()).collect();
    prog.id_table
        .iter()
        .filter_map(|(id, name)| {
            let start = *prog.labels.get(name)?;
            Some(walk(&prog, &id_names, *id, name, start))
        })
        .collect()
}

/// A sane upper bound on interpreter steps per sequence: every real one
/// reaches a frame-loop, chain, or terminal far sooner. Hitting it means a
/// parser desync or an unhandled opcode (a `debug_assert` fires).
const MAX_STEPS: usize = 4000;

/// Walk one sequence from `start`, accumulating frames until it loops on a
/// frame it already emitted, chains into another named sequence, or hits a
/// terminal opcode / the end of the stream.
fn walk(
    prog: &Program,
    id_names: &HashSet<&str>,
    id: u8,
    name: &str,
    start: usize,
) -> AnimSequence {
    let mut pos = start;
    let (mut dx, mut dy) = (0i32, 0i32);
    let mut action: Option<u8> = None;
    let mut turn = false;
    let mut frames: Vec<AnimFrame> = Vec::new();
    let mut frame_at: HashMap<usize, usize> = HashMap::new();
    let mut loops_to = None;
    let mut chains_to = None;

    // `goto` targets already taken — re-taking one can't reach a new frame
    // (a pure-opcode loop), so we stop instead of spinning to the cap.
    let mut goto_seen: HashSet<usize> = HashSet::new();
    let mut steps = 0;
    loop {
        if steps >= MAX_STEPS {
            debug_assert!(
                false,
                "animation `{name}` (#{id}) hit the {MAX_STEPS}-step cap — \
                 parser desync or unhandled opcode?"
            );
            break;
        }
        steps += 1;
        let Some(tok) = prog.tokens.get(pos) else {
            break;
        };
        match tok {
            Tok::Int(n) => {
                // Frame ids are bytes; 0 is the valid "blank" sentinel. An
                // out-of-range value means an opcode byte leaked in as a frame.
                debug_assert!(
                    (0..=255).contains(n),
                    "frame id {n} out of range in `{name}` (#{id})"
                );
                // Re-reaching a frame we already drew closes the loop (e.g.
                // `goto stand` lands on the `act` before frame 15, not the
                // frame itself).
                if let Some(&fi) = frame_at.get(&pos) {
                    loops_to = Some(fi);
                    break;
                }
                frame_at.insert(pos, frames.len());
                frames.push(AnimFrame {
                    frame: u8::try_from(*n).unwrap_or(0),
                    dx,
                    dy,
                    action,
                    turn,
                });
                dx = 0;
                dy = 0;
                turn = false;
                pos += 1;
            }
            Tok::Word(_) => {
                // A `dw` is only reached as a `goto`/`ifwtless` operand; a
                // bare one here means the assembler desynced.
                debug_assert!(false, "stray `dw` token at {pos} in `{name}` (#{id})");
                pos += 1;
            }
            Tok::Op(op) => match op {
                Opcode::Chx => {
                    dx += prog.int_at(pos + 1);
                    pos += 2;
                }
                Opcode::Chy => {
                    dy += prog.int_at(pos + 1);
                    pos += 2;
                }
                Opcode::Act => {
                    action = Some(u8::try_from(prog.int_at(pos + 1)).unwrap_or(0));
                    pos += 2;
                }
                // One operand each: `tap`/`effect` an id, `ifwtless` a `dw`
                // target. We always take the *not-weightless* fall-through
                // (skip the target): the preview has no weight state, and not
                // weightless is the in-game default. The weightless branch is
                // deliberately not explored.
                Opcode::Tap | Opcode::Effect | Opcode::Ifwtless => pos += 2,
                Opcode::Setfall => pos += 3,
                Opcode::Aboutface => {
                    turn = !turn;
                    pos += 1;
                }
                Opcode::Up | Opcode::Down | Opcode::Jaru | Opcode::Jard => pos += 1,
                Opcode::Die | Opcode::Nextlevel => break,
                Opcode::Goto => {
                    let Some(target) = prog.word_at(pos + 1).map(str::to_string) else {
                        break;
                    };
                    let Some(t) = prog.resolve(&target, pos) else {
                        break;
                    };
                    if let Some(&fi) = frame_at.get(&t) {
                        loops_to = Some(fi);
                        break;
                    } else if target != name && id_names.contains(target.as_str()) {
                        // A jump into a *different* named sequence ends this
                        // one; a jump to its own entry (e.g. `stand`) is a
                        // self-loop, caught when the frame is re-reached.
                        chains_to = Some(target);
                        break;
                    } else if !goto_seen.insert(t) {
                        break;
                    }
                    pos = t;
                }
            },
        }
    }

    AnimSequence {
        id,
        name: name.to_string(),
        frames,
        loops_to,
        chains_to,
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Drop a Merlin line comment (`;` to end of line).
fn strip_comment(line: &str) -> &str {
    line.split(';').next().unwrap_or("")
}

/// Evaluate a Merlin operand expression: a sum of `+`/`-` terms, each a
/// decimal or `$hex` literal (e.g. `$40+$20+6`). Unknown terms count as 0.
fn eval(expr: &str) -> i32 {
    let mut total = 0i32;
    let mut sign = 1i32;
    let mut term = String::new();
    for ch in expr.chars() {
        if ch == '+' || ch == '-' {
            total += sign * parse_num(&term);
            term.clear();
            sign = if ch == '-' { -1 } else { 1 };
        } else if !ch.is_whitespace() {
            term.push(ch);
        }
    }
    total + sign * parse_num(&term)
}

/// Reinterpret a parsed operand's low byte as a signed two's-complement
/// byte, so both a decimal `-2` and a raw `$fe` decode to `-2`.
fn byte_i8(v: i32) -> i8 {
    i8::from_ne_bytes([u8::try_from(v & 0xff).unwrap_or(0)])
}

fn parse_num(term: &str) -> i32 {
    let t = term.trim();
    if t.is_empty() {
        0
    } else if let Some(hex) = t.strip_prefix('$') {
        i32::from_str_radix(hex, 16).unwrap_or(0)
    } else {
        t.parse().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(name: &str) -> &'static AnimSequence {
        animations()
            .iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("sequence `{name}` not found"))
    }

    #[test]
    fn framedef_decodes_known_frames() {
        // stand = frame 15 = `$0f,9` -> CHTAB1 image 15 (index 14).
        let stand = frame_def(15).expect("frame 15");
        assert_eq!((stand.image, stand.sword), (0x0f, 9));
        assert_eq!(
            frame_sprite(15),
            Some(SpriteRef {
                chtab: 1,
                index: 14
            })
        );
        // freefall = frame 106 = `$36,$40` -> CHTAB2 image 54 (index 53).
        assert_eq!(
            frame_sprite(106),
            Some(SpriteRef {
                chtab: 2,
                index: 53
            })
        );
        // a climbup frame (Fsword $80) lands in CHTAB3.
        assert_eq!(
            frame_def(135).and_then(FrameDef::sprite).map(|s| s.chtab),
            Some(3)
        );
    }

    #[test]
    fn stand_is_a_single_looping_frame() {
        let stand = seq("stand");
        assert_eq!(stand.id, 2);
        assert_eq!(stand.frames.len(), 1);
        assert_eq!(stand.frames[0].frame, 15);
        assert_eq!(stand.frames[0].action, Some(0)); // `act,0`
        assert_eq!(stand.loops_to, Some(0)); // `goto stand`
    }

    #[test]
    fn startrun_runs_through_1_to_14_then_loops_into_the_cycle() {
        let run = seq("startrun");
        assert_eq!(run.id, 1);
        let ids: Vec<u8> = run.frames.iter().map(|f| f.frame).collect();
        assert_eq!(ids, (1..=14).collect::<Vec<_>>());
        assert_eq!(run.frames[0].action, Some(1)); // `act,1`
                                                   // `goto runcyc1` loops back to frame 7 (the start of the cycle).
        assert_eq!(run.loops_to, Some(6));
        assert_eq!(run.frames[6].frame, 7);
    }

    #[test]
    fn chx_lands_on_the_following_frame() {
        // `runstt4 db 4,chx,8` / `runstt5 db 5,chx,3`: the chx after frame 4
        // is the motion carried into frame 5.
        let run = seq("startrun");
        assert_eq!(run.frames[3].frame, 4);
        assert_eq!(run.frames[4].frame, 5);
        assert_eq!(run.frames[4].dx, 8);
    }

    #[test]
    fn every_id_table_entry_decodes() {
        // All 100+ sequences parse without panicking and most carry frames.
        let all = animations();
        assert!(
            all.len() > 100,
            "expected the full id table, got {}",
            all.len()
        );
        let with_frames = all.iter().filter(|s| !s.frames.is_empty()).count();
        assert!(with_frames > 80, "only {with_frames} sequences had frames");
    }
}
