//! Runtime animation cursor (#95): the playback engine that drives a
//! character's frames from the decoded SEQTABLE sequences
//! ([`pop_assets::anim`]).
//!
//! An [`AnimCursor`] points at one sequence and a frame within it. Each tick
//! the owner [`AnimCursor::advance`]s it (looping or chaining at the end per
//! the sequence's `loops_to` / `chains_to`) and reads [`AnimCursor::current`]
//! for the frame to draw and its per-frame `(dx, dy)` motion. The Prince's
//! displayed frame and run displacement come from here, replacing the
//! hand-picked CHTAB indices + transcribed `RUN_CHX` table in [`crate::world`].
//!
//! Slice 1 wires only the Prince's locomotion (stand / run / fall / climb);
//! faithful jump arcs, turn-in-place, combat chaining, and guard / trap
//! cursors are follow-ups — the engine itself is character-agnostic.

use std::collections::HashMap;
use std::sync::OnceLock;

use pop_assets::anim::{self, AnimFrame};

/// Sequence name → index into [`anim::animations`], built once. The decoded
/// table is `'static`, so the map can borrow its names.
fn name_index() -> &'static HashMap<&'static str, usize> {
    static IDX: OnceLock<HashMap<&'static str, usize>> = OnceLock::new();
    IDX.get_or_init(|| {
        anim::animations()
            .iter()
            .enumerate()
            .map(|(i, s)| (s.name, i))
            .collect()
    })
}

/// A playback cursor over one [`anim`] sequence: which sequence, and the
/// frame within it.
#[derive(Clone, Copy, Debug)]
pub struct AnimCursor {
    /// Index into [`anim::animations`].
    seq: usize,
    /// Frame index into the sequence's frame list.
    pos: usize,
}

impl AnimCursor {
    /// Start on the named sequence at its first frame. An unknown name falls
    /// back to sequence 0 (rather than panicking), so a stray name still
    /// yields a drawable frame.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            seq: name_index().get(name).copied().unwrap_or(0),
            pos: 0,
        }
    }

    /// The sequence currently playing, by name (`""` if the index is somehow
    /// out of range, which the constructors prevent).
    #[must_use]
    pub fn name(&self) -> &'static str {
        anim::animations().get(self.seq).map_or("", |s| s.name)
    }

    /// The frame to display (and move by) this tick, if the sequence has any.
    #[must_use]
    pub fn current(&self) -> Option<&'static AnimFrame> {
        anim::animations().get(self.seq)?.frames.get(self.pos)
    }

    /// Switch to `name` from its first frame — unless it is already playing,
    /// so a held sequence (e.g. a run) keeps looping instead of restarting
    /// every tick.
    pub fn play(&mut self, name: &str) {
        if self.name() != name {
            *self = Self::new(name);
        }
    }

    /// Advance one frame. At the end of the sequence: loop back to `loops_to`,
    /// else chain into `chains_to`, else hold on the last frame.
    pub fn advance(&mut self) {
        let Some(seq) = anim::animations().get(self.seq) else {
            return;
        };
        let len = seq.frames.len();
        if self.pos + 1 < len {
            self.pos += 1;
        } else if let Some(loop_to) = seq.loops_to {
            self.pos = loop_to.min(len.saturating_sub(1));
        } else if let Some(next) = seq.chains_to {
            *self = Self::new(next);
        }
        // else: a terminal sequence with no loop/chain — hold the last frame.
    }
}

/// Number of frames in the named sequence (0 if there is no such sequence) —
/// e.g. to size a position interpolation to the climb animation's length.
#[must_use]
pub fn sequence_len(name: &str) -> usize {
    name_index()
        .get(name)
        .and_then(|&i| anim::animations().get(i))
        .map_or(0, |s| s.frames.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stand_holds_a_single_frame() {
        let mut c = AnimCursor::new("stand");
        for _ in 0..5 {
            c.advance();
        }
        assert_eq!(c.current().map(|f| f.frame), Some(15));
        assert_eq!(c.name(), "stand");
    }

    #[test]
    fn freefall_holds_a_single_frame() {
        let mut c = AnimCursor::new("freefall");
        for _ in 0..5 {
            c.advance();
        }
        assert_eq!(c.current().map(|f| f.frame), Some(106));
    }

    #[test]
    fn startrun_loops_back_into_the_run_cycle() {
        let mut c = AnimCursor::new("startrun");
        assert_eq!(c.current().map(|f| f.frame), Some(1), "opens on frame 1");
        // 14 frames (indices 0..13); advancing off the end loops to index 6.
        for _ in 0..13 {
            c.advance();
        }
        assert_eq!(c.current().map(|f| f.frame), Some(14), "last accel frame");
        c.advance();
        assert_eq!(
            c.current().map(|f| f.frame),
            Some(7),
            "loops to the run-cycle start (index 6)"
        );
        assert_eq!(c.name(), "startrun");
    }

    #[test]
    fn turn_flips_then_chains_to_stand() {
        let mut c = AnimCursor::new("turn");
        assert!(
            c.current().is_some_and(|f| f.turn),
            "turn opens with the aboutface frame"
        );
        for _ in 0..sequence_len("turn") {
            c.advance();
        }
        assert_eq!(c.name(), "stand", "turn chains into stand when it finishes");
    }

    #[test]
    fn climbup_chains_to_stand() {
        let mut c = AnimCursor::new("climbup");
        for _ in 0..sequence_len("climbup") {
            c.advance();
        }
        assert_eq!(c.name(), "stand");
    }

    #[test]
    fn play_switches_only_on_change() {
        let mut c = AnimCursor::new("startrun");
        c.advance();
        c.advance();
        let mid = c.current().map(|f| f.frame);
        c.play("startrun"); // same sequence → keep position
        assert_eq!(c.current().map(|f| f.frame), mid);
        c.play("stand"); // different → reset to frame 0
        assert_eq!(c.current().map(|f| f.frame), Some(15));
    }

    #[test]
    fn sequence_len_counts_frames() {
        assert_eq!(sequence_len("startrun"), 14);
        assert_eq!(sequence_len("stand"), 1);
        assert_eq!(sequence_len("no-such-sequence"), 0);
    }
}
