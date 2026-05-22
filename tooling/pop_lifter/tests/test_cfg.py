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


# ---- dominator / loop analysis


def test_checkfloor_entry_dominates_everything(source_dir):
    """The CFG entry must dominate every reachable block."""
    from pop_lifter.cfg import compute_idoms, dominates

    cfg = build_cfg(_lift_and_structure(source_dir, "CHECKFLOOR"))
    idom = compute_idoms(cfg)
    for b in cfg.blocks:
        if b.id in idom:
            assert dominates(idom, cfg.entry_id, b.id)


def test_checkfloor_has_no_back_edges(source_dir):
    """CHECKFLOOR is loop-free — no back-edges at all."""
    from pop_lifter.cfg import find_back_edges

    cfg = build_cfg(_lift_and_structure(source_dir, "CHECKFLOOR"))
    assert find_back_edges(cfg) == []


def test_chgshadposn_has_one_back_edge(source_dir):
    """AUTO.S `chgshadposn` has exactly one back-edge: from the
    `bpl :loop` tail to the `:loop` header."""
    from pop_lifter.cfg import find_back_edges
    from pop_lifter.pass0_parse import parse_files
    from pop_lifter.pass1_lift import lift_file
    from pop_lifter.pass2_struct import structure_module

    ast = parse_files(
        [source_dir / "EQ.S", source_dir / "GAMEEQ.S", source_dir / "AUTO.S"],
        search_paths=[source_dir],
    )
    auto = next(f for f in ast.files if Path(f.path).name == "AUTO.S")
    ir2 = structure_module(lift_file(auto, ast.equates, ["chgshadposn"]).module)
    cfg = build_cfg(ir2.find("chgshadposn"))
    back = find_back_edges(cfg)
    assert len(back) == 1, f"expected 1 back-edge, got {back}"
    src, dst = back[0]
    assert cfg.blocks[dst].label == ":loop"


def test_natural_loop_body_multi_block():
    """Natural-loop body for a genuine multi-block do-while: header
    has a conditional forward exit, tail is a separate block with the
    back-edge. The body should be exactly {header, tail}; the
    preheader (block 0) and the post-loop block must not be pulled
    in."""
    from pop_lifter.cfg import natural_loop_body
    from pop_lifter.ir1 import (
        Abs,
        Branch,
        CmpImm,
        Imm,
        Label,
        LoadImm,
        Reg,
        Return,
        Routine,
        SourceRef,
        StoreAbs,
    )

    src = SourceRef(file="syn", line=0, raw="")
    # B0: LoadImm (preheader, falls through to :hdr)
    # B1 (:hdr): StoreAbs to scratch, Branch(eq, :end) — header with
    #            a forward exit, which means it's NOT the tail.
    # B2: StoreAbs to scratch, Branch(ne, :hdr) — the tail.
    # B3 (:end): Return — post-loop.
    r = Routine(
        name="dw_multi",
        body=[
            LoadImm(reg=Reg.A, imm=Imm(value=0, text="#0"), src=src),
            Label(name=":hdr", src=src),
            StoreAbs(reg=Reg.A, target=Abs(name="s", addr=0x80), src=src),
            Branch(cond="eq", target=":end", src=src),
            StoreAbs(reg=Reg.A, target=Abs(name="t", addr=0x81), src=src),
            Branch(cond="ne", target=":hdr", src=src),
            Label(name=":end", src=src),
            Return(src=src),
        ],
    )
    cfg = build_cfg(r)
    hdr = cfg.label_to_block[":hdr"]
    end = cfg.label_to_block[":end"]
    # The tail is the block whose terminator targets the header.
    tail = next(
        b.id for b in cfg.blocks
        if isinstance(b.terminator, Branch) and b.terminator.target == ":hdr"
    )
    assert tail != hdr, "test setup error: tail and header collapsed"

    body = natural_loop_body(cfg, source=tail, header=hdr)
    assert body == {hdr, tail}, (
        f"expected loop body {{hdr, tail}} = {{{hdr}, {tail}}}, got {body}"
    )
    # Preheader (B0) and post-loop block (:end) must be excluded.
    assert 0 not in body
    assert end not in body


def test_natural_loop_body_self_loop():
    """Single-block do-while (`:hdr ... bne :hdr`) — the header IS
    the tail. Pinned to catch the bug where the reverse walk used to
    pull the preheader into the body when `source == header`."""
    from pop_lifter.cfg import natural_loop_body
    from pop_lifter.ir1 import (
        Branch,
        CmpImm,
        Imm,
        Label,
        LoadImm,
        Reg,
        Return,
        Routine,
        SourceRef,
    )

    src = SourceRef(file="syn", line=0, raw="")
    r = Routine(
        name="self",
        body=[
            LoadImm(reg=Reg.A, imm=Imm(value=0, text="#0"), src=src),
            Label(name=":hdr", src=src),
            CmpImm(reg=Reg.A, imm=Imm(value=5, text="#5"), src=src),
            Branch(cond="ne", target=":hdr", src=src),
            Return(src=src),
        ],
    )
    cfg = build_cfg(r)
    hdr = cfg.label_to_block[":hdr"]
    # In the self-loop case the back-edge source IS the header.
    body = natural_loop_body(cfg, source=hdr, header=hdr)
    assert body == {hdr}, (
        f"self-loop body should be just {{header}}, got {body}"
    )
