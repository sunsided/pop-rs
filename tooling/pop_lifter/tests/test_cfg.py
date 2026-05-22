"""Basic-block construction tests. Pinned shapes for CHECKFLOOR
specifically — that's the routine the relooper targets and any
regression in block splitting would show up here first."""

from __future__ import annotations

from pathlib import Path

from pop_lifter.cfg import build_cfg
from pop_lifter.ir1 import If, Goto, Label, Return
from pop_lifter.pass0_parse import parse_files
from pop_lifter.pass1_lift import lift_file
from pop_lifter.pass2_struct import structure_module


def _lift_and_structure(source_dir: Path, name: str):
    ast = parse_files(
        [source_dir / "EQ.S", source_dir / "GAMEEQ.S", source_dir / "CTRL.S"],
        search_paths=[source_dir],
    )
    ctrl = next(f for f in ast.files if Path(f.path).name == "CTRL.S")
    ir1 = lift_file(ctrl, ast.equates, [name]).module
    return structure_module(ir1).find(name)


def test_checkfloor_block_count(source_dir):
    """CHECKFLOOR's IR2 has a known shape: 13 basic blocks once
    fusion + elision are done.

    Block roster (verified against the pinned IR2 artifact):
      B0: a = CharAction      ; if a == 6 → ]rts
      B1: if a != 5           → :2
      B2: a = CharPosn        ; if a == 0x6d → :ong
      B3: if a != 0xb9        → ]rts
      B4 (:ong): tail_call onground
      B5 (:2): if a == 4      → falling   (cross-module)
      B6: if a != 3           → :1
      B7: a = CharPosn        ; if a < 0x66 → ]rts
      B8: if a >= 0x6a        → ]rts
      B9: tail_call fallon
      B10 (:1): if a == 2     → ]rts
      B11: tail_call onground
      B12 (]rts): return
    """
    routine = _lift_and_structure(source_dir, "CHECKFLOOR")
    cfg = build_cfg(routine)
    assert len(cfg.blocks) == 13, (
        f"expected 13 basic blocks in CHECKFLOOR, got {len(cfg.blocks)}"
    )


def test_checkfloor_labels(source_dir):
    """The labelled blocks must match the IR2 label set."""
    cfg = build_cfg(_lift_and_structure(source_dir, "CHECKFLOOR"))
    labels = {b.label for b in cfg.blocks if b.label is not None}
    assert labels == {":ong", ":2", ":1", "]rts"}


def test_checkfloor_rts_block_has_five_predecessors(source_dir):
    """The ]rts block is the convergence point for every early-exit
    return. CHECKFLOOR has 5 of them: cmp/beq #6, cmp/bne #0xb9,
    cmp/bcc #0x66, cmp/bcs #0x6a, cmp/beq #2."""
    cfg = build_cfg(_lift_and_structure(source_dir, "CHECKFLOOR"))
    rts_id = cfg.label_to_block["]rts"]
    assert len(cfg.pred[rts_id]) == 5


def test_checkfloor_ong_block_has_two_predecessors(source_dir):
    """:ong is reached from B2's taken edge (`if a == 0x6d`) and from
    B3's fall-through."""
    cfg = build_cfg(_lift_and_structure(source_dir, "CHECKFLOOR"))
    ong_id = cfg.label_to_block[":ong"]
    assert len(cfg.pred[ong_id]) == 2


def test_terminator_kinds(source_dir):
    """Every block must end on a control-flow item."""
    cfg = build_cfg(_lift_and_structure(source_dir, "CHECKFLOOR"))
    for b in cfg.blocks:
        assert isinstance(b.terminator, (Return, Goto, If)), (
            f"block B{b.id} has unexpected terminator type "
            f"{type(b.terminator).__name__}"
        )


def test_returns_have_no_successors(source_dir):
    """Return / tail_call blocks should have an empty local successor
    list — control leaves the routine."""
    cfg = build_cfg(_lift_and_structure(source_dir, "CHECKFLOOR"))
    for b in cfg.blocks:
        if isinstance(b.terminator, Return):
            assert cfg.succ[b.id] == []
        elif isinstance(b.terminator, Goto) and b.terminator.kind == "tail_call":
            assert cfg.succ[b.id] == []


def test_empty_routine_gets_synthetic_return():
    """An empty routine body must produce a one-block CFG with a
    synthetic Return so downstream passes don't choke."""
    from pop_lifter.ir1 import Routine

    cfg = build_cfg(Routine(name="empty", entry_aliases=[], body=[]))
    assert len(cfg.blocks) == 1
    assert isinstance(cfg.blocks[0].terminator, Return)
