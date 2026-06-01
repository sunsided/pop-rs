//! Character animation tables — POP's `SEQTABLE` / `FRAMEDEF` decoded into a
//! typed list of motions the editor can preview (#89).
//!
//! Two vendored tables drive every character animation:
//!
//! * **`FRAMEDEF.S`** — one 5-byte record per frame:
//!   `(Fimage, Fsword, Fdx, Fdy, Fcheck)`. `Fimage`/`Fsword` pick a CHTAB
//!   sprite via `decodeim` ([`FrameDef::sprite`], `CTRLSUBS.S`); `Fdx`/`Fdy`
//!   are the per-frame blit offset.
//! * **`SEQTABLE.S`** — one byte stream per motion: positive bytes are frame
//!   ids (one displayed per tick), interleaved with negative *opcodes*
//!   (`chx`/`chy`/`act`/`goto`/`aboutface`/…). The `ANIMCHAR` interpreter
//!   (`COLL.S`) walks a stream until it hits a frame, draws it, and stops for
//!   that tick; `goto` chains/loops the streams.
//!
//! The decoded data is **baked into committed Rust** ([`generated`], the
//! `// @generated` module) — the shipped library reads `static` tables and
//! has no `include_str!`, no runtime parse, and no `vendor/` dependency. The
//! `.S` parser lives only under `#[cfg(test)]` ([`parse`]); it feeds the code
//! generator ([`gen`]) and a drift test that fails CI if `generated.rs` is
//! stale. Regenerate with `task gen:anim`.

mod generated;

#[cfg(test)]
mod gen;
#[cfg(test)]
mod parse;

/// The `FrameDef::check` (`Fcheck`) bit layout — flag bits 5-7 plus a base-X
/// offset in bits 0-4. Named so the baked data reads semantically.
pub mod check {
    /// Bit 7 — odd-X pixel parity (the half-dot alignment, #121; `CTRLSUBS.S`
    /// `eor FCharFace`, "look only at the hibits").
    pub const ODD_X: u8 = 0x80;
    /// Bit 6 — `Fcheckmark` (`GAMEEQ.S`).
    pub const MARK: u8 = 0x40;
    /// Bit 5 — `Fthinmark`: draw the figure 3 px thinner each side
    /// (`CTRLSUBS.S` "set up sword").
    pub const THIN: u8 = 0x20;
    /// Bits 0-4 — base-X collision-offset mask (`GETBASEX`:
    /// `-(check & BASE_X)`, then `+ Fdx`).
    pub const BASE_X: u8 = 0x1f;
}

/// One `FRAMEDEF.S` record: the sprite + offsets for a single frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameDef {
    /// `Fimage` — 1-based image index in its CHTAB; bit 7 selects the high
    /// CHTAB bank. Decoded (with `sword`) by [`Self::sprite`], so kept as a
    /// raw byte here rather than split.
    pub image: u8,
    /// `Fsword` — sword image; bits 6/7 (with `Fimage` bit 7) select the
    /// CHTAB table. Decoded by [`Self::sprite`].
    pub sword: u8,
    /// `Fdx` — per-frame horizontal blit offset, px (facing-relative).
    pub dx: i8,
    /// `Fdy` — per-frame vertical blit offset, px.
    pub dy: i8,
    /// `Fcheck` — packed flags + offset; see the [`check`](crate::anim::check)
    /// module ([`ODD_X`](check::ODD_X) / [`MARK`](check::MARK) /
    /// [`THIN`](check::THIN) / [`BASE_X`](check::BASE_X)).
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
    /// Construct a frame record (the 5-byte `FRAMEDEF.S` field order).
    #[must_use]
    pub const fn new(image: u8, sword: u8, dx: i8, dy: i8, check: u8) -> Self {
        Self {
            image,
            sword,
            dx,
            dy,
            check,
        }
    }

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

impl AnimFrame {
    /// Construct one playback step.
    #[must_use]
    pub const fn new(frame: u8, dx: i32, dy: i32, action: Option<u8>, turn: bool) -> Self {
        Self {
            frame,
            dx,
            dy,
            action,
            turn,
        }
    }
}

/// A named animation: the frames it plays and how it ends (self-loop or a
/// chain into another sequence). Backed by `'static` generated data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnimSequence {
    /// Sequence id (the `SEQTABLE` `:N dw …` index).
    pub id: u8,
    /// Label name (`stand`, `startrun`, `climbup`, …).
    pub name: &'static str,
    /// Frames in play order.
    pub frames: &'static [AnimFrame],
    /// Frame index a closing `goto` loops back to, if it loops on itself.
    pub loops_to: Option<usize>,
    /// Sequence name a closing `goto` chains into, if not a self-loop.
    pub chains_to: Option<&'static str>,
}

impl AnimSequence {
    /// Construct a named animation.
    #[must_use]
    pub const fn new(
        id: u8,
        name: &'static str,
        frames: &'static [AnimFrame],
        loops_to: Option<usize>,
        chains_to: Option<&'static str>,
    ) -> Self {
        Self {
            id,
            name,
            frames,
            loops_to,
            chains_to,
        }
    }
}

/// The `FRAMEDEF.S` 5-byte tables in order: main `Fdef`, then the `altset1`
/// (chtable4) and `altset2` (chtable6) alternates.
fn sections() -> &'static [&'static [FrameDef]] {
    &generated::SECTIONS
}

/// All main `FRAMEDEF` records, 0-based (`frame_defs()[n]` is frame `n + 1`).
#[must_use]
pub fn frame_defs() -> &'static [FrameDef] {
    sections().first().copied().unwrap_or(&[])
}

/// The `FRAMEDEF` record for frame id `frame` (1-based), if defined.
#[must_use]
pub fn frame_def(frame: u8) -> Option<FrameDef> {
    frame_defs()
        .get(usize::from(frame).checked_sub(1)?)
        .copied()
}

/// The CHTAB sprite for frame id `frame` (1-based) as the **kid** sees it.
#[must_use]
pub fn frame_sprite(frame: u8) -> Option<SpriteRef> {
    frame_def(frame)?.sprite()
}

/// The `altset1` alternate frames (chtable4 = the guard body). Like the main
/// table, indexed by frame number (`:150..:189`, the "guy-N" guard poses).
fn altset1() -> &'static [FrameDef] {
    sections().get(1).copied().unwrap_or(&[])
}

/// The CHTAB sprite for frame id `frame` as a **guard** sees it
/// (`CharID = TypeGd`, `CTRLSUBS.S usealtsets`): the guard-range `CharPosn`
/// (`$96..=$bd`, with `$66..=$6a` shifted up by `$46`) is rewritten to the
/// altset1 entry of the *same* frame number, whose `Fsword = $c0+n` decodes
/// to chtable4 — the loaded guard body. Other frames stay on the shared main
/// set. So guard sequences render the guard, not the kid.
#[must_use]
pub fn guard_frame_sprite(frame: u8) -> Option<SpriteRef> {
    let mut a = frame;
    if (0x66..=0x6a).contains(&a) {
        a = a.wrapping_add(0x46);
    }
    if (0x96..=0xbd).contains(&a) {
        return altset1().get(usize::from(a).checked_sub(1)?)?.sprite();
    }
    frame_sprite(frame)
}

/// All decoded animation sequences, in `SEQTABLE` id order.
#[must_use]
pub fn animations() -> &'static [AnimSequence] {
    generated::SEQUENCES
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
    fn guard_frames_remap_to_the_guard_body() {
        // Kid frame 158 ("ready") is CHTAB5 (the kid); a guard's `usealtsets`
        // rewrites it to altset1 `:158` (guy-10), `Fsword=$c0+8` -> chtable4
        // (the loaded guard body) at the same image index.
        assert_eq!(frame_sprite(158).map(|s| s.chtab), Some(5));
        assert_eq!(
            guard_frame_sprite(158),
            Some(SpriteRef { chtab: 4, index: 7 })
        );
        // The `$66..=$6a` band shifts up by `$46` before the remap.
        assert_eq!(guard_frame_sprite(102).map(|s| s.chtab), Some(4));
        // Out-of-range frames (e.g. the shared run frame 7) stay on the main
        // set — a guard shares them with the kid.
        assert_eq!(guard_frame_sprite(7), frame_sprite(7));
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

    /// The baked `generated.rs` must equal a fresh parse of the vendored `.S`.
    /// Field-by-field (not text) so it never false-fails on formatting and
    /// pinpoints the drift. Regenerate with `task gen:anim`.
    #[test]
    fn generated_matches_a_fresh_parse() {
        const HINT: &str = "stale src/anim/generated.rs — regenerate with `task gen:anim`";

        // The parser yields the 3 used sections plus empty trailing `ds`
        // sections; only the first 3 are baked. Compare those.
        let parsed = super::parse::parse_framedef();
        for (i, b) in super::generated::SECTIONS.iter().enumerate() {
            let p = parsed.get(i).map_or(&[][..], Vec::as_slice);
            assert_eq!(p, *b, "FRAMEDEF section {i}: {HINT}");
        }

        let parsed = super::parse::parse_sequences();
        let baked = animations();
        assert_eq!(parsed.len(), baked.len(), "sequence count: {HINT}");
        for (p, b) in parsed.iter().zip(baked) {
            assert_eq!(p.id, b.id, "{HINT}");
            assert_eq!(p.name, b.name, "sequence `{}`: {HINT}", b.name);
            assert_eq!(
                p.frames.as_slice(),
                b.frames,
                "sequence `{}`: {HINT}",
                b.name
            );
            assert_eq!(p.loops_to, b.loops_to, "sequence `{}`: {HINT}", b.name);
            assert_eq!(
                p.chains_to.as_deref(),
                b.chains_to,
                "sequence `{}`: {HINT}",
                b.name
            );
        }
    }

    /// Rewrite `src/anim/generated.rs` from the vendored source. Inert unless
    /// `POP_ANIM_REGEN` is set, so a normal `cargo test` never writes; `task
    /// gen:anim` sets it (then runs `cargo fmt`).
    #[test]
    fn regenerate_anim_data() {
        if std::env::var_os("POP_ANIM_REGEN").is_none() {
            return;
        }
        let source = super::gen::render_generated();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/anim/generated.rs");
        std::fs::write(&path, source).expect("write generated.rs");
        eprintln!("regenerated {}", path.display());
    }
}
