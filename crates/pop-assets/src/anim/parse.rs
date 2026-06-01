//! The `FRAMEDEF.S` / `SEQTABLE.S` parser — **test-only**. It reads the
//! vendored Apple II source and produces owned tables, used by the data
//! generator ([`super::gen`]) and the drift test. The shipped library never
//! compiles this module (it reads the baked [`super::generated`] statics), so
//! the runtime build has no `include_str!` / vendor dependency.

use std::collections::{HashMap, HashSet};

use super::{AnimFrame, FrameDef};

const FRAMEDEF_SRC: &str =
    include_str!("../../../../vendor/pop-apple2/01 POP Source/Source/FRAMEDEF.S");
const SEQTABLE_SRC: &str =
    include_str!("../../../../vendor/pop-apple2/01 POP Source/Source/SEQTABLE.S");

/// An owned animation sequence — the parser's output before it's baked into
/// the `'static` [`super::AnimSequence`].
pub(super) struct OwnedSequence {
    pub(super) id: u8,
    pub(super) name: String,
    pub(super) frames: Vec<AnimFrame>,
    pub(super) loops_to: Option<usize>,
    pub(super) chains_to: Option<String>,
}

// ---------------------------------------------------------------------------
// FRAMEDEF.S
// ---------------------------------------------------------------------------

/// Parse every 5-byte `FRAMEDEF.S` table in order: the main `Fdef`, then the
/// `altset1` / `altset2` alternates (each its own `:1`-based block, separated
/// by `ds` padding). The 3-byte `swordtab` yields entries with < 5 values and
/// is skipped.
pub(super) fn parse_framedef() -> Vec<Vec<FrameDef>> {
    let mut sections: Vec<Vec<FrameDef>> = vec![Vec::new()];
    let mut started = false;
    for raw in FRAMEDEF_SRC.lines() {
        let line = strip_comment(raw);
        let trimmed = line.trim();
        if trimmed == "Fdef" {
            started = true;
            continue;
        }
        if !started {
            continue;
        }
        // `ds` padding ends one table and begins the next alternate set.
        if trimmed.starts_with("ds ") {
            sections.push(Vec::new());
            continue;
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
        let defs = sections.last_mut().expect("at least one section");
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
    sections
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

pub(super) fn parse_sequences() -> Vec<OwnedSequence> {
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
) -> OwnedSequence {
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

    OwnedSequence {
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
