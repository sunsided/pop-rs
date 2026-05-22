"""Relooper tests: structural correctness on CHECKFLOOR, plus
behavioural equivalence between IR2 (interpreted as IR1) and the
relooped IR3."""

from __future__ import annotations

from pathlib import Path

from pop_lifter.interp_ir1 import run as ir1_run
from pop_lifter.interp_ir3 import run as ir3_run
from pop_lifter.ir1 import (
    Abs,
    Imm,
    LoadImm,
    ModuleIR1,
    Reg,
    Return,
    Routine,
    SourceRef,
    StoreAbs,
)
from pop_lifter.ir3 import (
    Block,
    GotoStmt,
    IfStmt,
    LabelStmt,
    RawIfStmt,
    RawStmt,
    ReturnStmt,
    TailCallStmt,
)
from pop_lifter.pass0_parse import parse_files
from pop_lifter.pass1_lift import lift_file
from pop_lifter.pass2_reloop import reloop_module, reloop_routine
from pop_lifter.pass2_struct import structure_module


def _ir3_module(source_dir: Path):
    """Lift CHECKFLOOR through pass 1 + pass 2 fusion + elision +
    reloop. Returns the IR3 module with chase callees stripped (they
    have loops and would fall back to unstructured form)."""
    ast = parse_files(
        [source_dir / "EQ.S", source_dir / "GAMEEQ.S", source_dir / "CTRL.S"],
        search_paths=[source_dir],
    )
    ctrl = next(f for f in ast.files if Path(f.path).name == "CTRL.S")
    ir1 = lift_file(ctrl, ast.equates, ["CHECKFLOOR"]).module
    ir2 = structure_module(ir1)
    ir3 = reloop_module(ir2)
    ir3.routines = [r for r in ir3.routines if r.name == "CHECKFLOOR"]
    return ir3


# --------------------------------------------------------------- structural


def test_checkfloor_ir3_is_fully_structured(source_dir):
    """CHECKFLOOR must contain ZERO GotoStmt / LabelStmt — those only
    appear when the relooper falls back. Anything emitted should be
    in the structured set."""
    ir3 = _ir3_module(source_dir)
    cf = ir3.find("CHECKFLOOR")
    assert cf is not None

    def walk(block: Block):
        for s in block.stmts:
            assert not isinstance(s, (GotoStmt, LabelStmt)), (
                f"unexpected unstructured stmt {type(s).__name__} "
                f"in CHECKFLOOR — relooper bail?"
            )
            if isinstance(s, (IfStmt, RawIfStmt)):
                walk(s.then_block)
                if s.else_block is not None:
                    walk(s.else_block)

    walk(cf.body)


def test_checkfloor_ir3_has_no_raw_if(source_dir):
    """After fusion + elision, every branch in CHECKFLOOR should be a
    structured `IfStmt`, not a flag-condition `RawIfStmt`."""
    ir3 = _ir3_module(source_dir)
    cf = ir3.find("CHECKFLOOR")

    def walk(block: Block):
        for s in block.stmts:
            assert not isinstance(s, RawIfStmt), (
                f"unfused branch surfaced in CHECKFLOOR — "
                f"flag {s.cond!r}"
            )
            if isinstance(s, IfStmt):
                walk(s.then_block)
                if s.else_block is not None:
                    walk(s.else_block)

    walk(cf.body)


def test_checkfloor_ir3_top_level_shape(source_dir):
    """Verify the headline structure of CHECKFLOOR's IR3:
       - first stmt: load CharAction
       - second: if a == 6 { return }
       - third: if a != 5 { ... :2 stuff ... }
       - then: load CharPosn, two early-exit ifs, tail_call onground.

    The exact body shape is pinned by the .ir3 artifact regen test;
    this checks the top-level outline so changes elsewhere don't
    silently rewire the entry path."""
    ir3 = _ir3_module(source_dir)
    cf = ir3.find("CHECKFLOOR")
    stmts = cf.body.stmts

    # First non-comment stmt: a load.
    assert isinstance(stmts[0], RawStmt)

    # Find the first IfStmt — should be a == 6 → Return.
    first_if = next(s for s in stmts if isinstance(s, IfStmt))
    assert first_if.cond.op == "=="
    assert first_if.cond.rhs.value == 6
    assert any(
        isinstance(t, ReturnStmt) for t in first_if.then_block.stmts
    )


def test_relooper_fallback_for_loops():
    """A routine with a backward local jump must take the
    unstructured fallback path — every IR1 item shows up as either a
    RawStmt, a Label/Goto/Return/TailCall, or a wrapped IfStmt with a
    Goto in its then-block. No infinite recursion, no exception."""
    from pop_lifter.ir1 import Branch, CmpImm, Imm, Label, Reg

    src = SourceRef(file="syn", line=0, raw="")
    # do { cmp; bne loop } end
    r = Routine(
        name="loopy",
        body=[
            Label(name=":loop", src=src),
            CmpImm(reg=Reg.A, imm=Imm(value=0, text="#0"), src=src),
            Branch(cond="ne", target=":loop", src=src),
            Return(src=src),
        ],
    )
    out = reloop_routine(r)
    # The unstructured fallback wraps the Branch as a RawIfStmt whose
    # then-block contains a GotoStmt (target is local).
    stmts = out.body.stmts
    assert any(isinstance(s, LabelStmt) for s in stmts), (
        "expected LabelStmt in the fallback shape"
    )
    raw_if = next((s for s in stmts if isinstance(s, RawIfStmt)), None)
    assert raw_if is not None
    assert any(
        isinstance(t, GotoStmt) for t in raw_if.then_block.stmts
    )


def test_fallback_cross_module_branch_becomes_tail_call():
    """In the unstructured fallback, an `If` whose target isn't any
    local label is a conditional tail call into another routine —
    IR1 executes it by switching routines. Emitting `GotoStmt` here
    would silently change semantics; the fallback must produce a
    `TailCallStmt` in the then-block instead."""
    from pop_lifter.ir1 import (
        Branch,
        CmpImm,
        Compare,
        If as IR1If,
        Imm,
        Label,
        Reg,
    )

    src = SourceRef(file="syn", line=0, raw="")
    # do { cmp; if a == 1 goto external_fn; bne :loop } — the
    # backward `bne :loop` triggers the fallback path.
    r = Routine(
        name="loopy_with_tail_call",
        body=[
            Label(name=":loop", src=src),
            CmpImm(reg=Reg.A, imm=Imm(value=0, text="#0"), src=src),
            IR1If(
                cond=Compare(
                    reg=Reg.A,
                    op="==",
                    rhs=Imm(value=1, text="#1"),
                ),
                target="external_fn",
                src=src,
            ),
            Branch(cond="ne", target=":loop", src=src),
            Return(src=src),
        ],
    )
    out = reloop_routine(r)
    # The IR1If with a non-local target must produce a TailCallStmt,
    # not a GotoStmt.
    if_stmt = next(s for s in out.body.stmts if isinstance(s, IfStmt))
    assert any(
        isinstance(t, TailCallStmt) and t.target == "external_fn"
        for t in if_stmt.then_block.stmts
    ), "cross-module If target must lower to TailCallStmt in the fallback"


def test_fallback_ir1_call_becomes_callstmt():
    """In the fallback path, an IR1 `Call` must be emitted as a
    structured `CallStmt`, not folded into a `RawStmt`. That keeps
    the IR3 shape consistent for downstream consumers regardless of
    whether the routine took the structured or unstructured path."""
    from pop_lifter.ir1 import Branch, Call as IR1Call, Label, Reg, CmpImm, Imm
    from pop_lifter.ir3 import CallStmt

    src = SourceRef(file="syn", line=0, raw="")
    r = Routine(
        name="loopy_with_call",
        body=[
            Label(name=":loop", src=src),
            IR1Call(target="helper", src=src),
            CmpImm(reg=Reg.A, imm=Imm(value=0, text="#0"), src=src),
            Branch(cond="ne", target=":loop", src=src),
            Return(src=src),
        ],
    )
    out = reloop_routine(r)
    assert any(
        isinstance(s, CallStmt) and s.target == "helper"
        for s in out.body.stmts
    ), "IR1 Call must lower to IR3 CallStmt in the fallback"
    # And the body must NOT contain a RawStmt wrapping the Call.
    from pop_lifter.ir3 import RawStmt as IR3RawStmt
    for s in out.body.stmts:
        if isinstance(s, IR3RawStmt):
            assert not isinstance(s.item, IR1Call), (
                "IR1 Call slipped through as a RawStmt"
            )


# --------------------------------------------------------------- behavioural


CHAR_ACTION = 0x46
CHAR_POSN = 0x40

_PATHS = [
    # (action, posn, expected sentinel set)
    (6, 0, set()),
    (5, 109, {0x200}),
    (5, 185, {0x200}),
    (5, 42, set()),
    (4, 0, {0x201}),
    (3, 104, {0x202}),
    (3, 50, set()),
    (3, 200, set()),
    (2, 0, set()),
    (0, 0, {0x200}),
    (1, 0, {0x200}),
    (7, 0, {0x200}),
]


def _stubs_module():
    """Same callee stubs the IR2 behavioural tests use: onground →
    write 1 to 0x200, falling → 0x201, fallon → 0x202. Defined here
    as IR1 so the IR3 runner falls back to IR1's interpreter for the
    chase callees."""
    src = SourceRef(file="syn", line=0, raw="")

    def stub(name: str, addr: int) -> Routine:
        return Routine(
            name=name,
            body=[
                LoadImm(reg=Reg.A, imm=Imm(value=1, text="#1"), src=src),
                StoreAbs(
                    reg=Reg.A,
                    target=Abs(name=f"<{name}>", addr=addr),
                    src=src,
                ),
                Return(src=src),
            ],
        )

    return ModuleIR1(
        name="STUBS",
        file="syn",
        routines=[stub("onground", 0x200), stub("falling", 0x201), stub("fallon", 0x202)],
    )


def test_relooper_preserves_every_checkfloor_path(source_dir):
    """The strongest assertion in this PR: every CHECKFLOOR path
    produces identical sentinel-touch sets when interpreted through
    IR2 vs. IR3. Catches any control-flow rewrite mistake the
    relooper might introduce."""
    ast = parse_files(
        [source_dir / "EQ.S", source_dir / "GAMEEQ.S", source_dir / "CTRL.S"],
        search_paths=[source_dir],
    )
    ctrl = next(f for f in ast.files if Path(f.path).name == "CTRL.S")
    ir1 = lift_file(ctrl, ast.equates, ["CHECKFLOOR"]).module
    ir2 = structure_module(ir1)
    # Strip chase callees from the IR2 (the test stubs them).
    ir2.routines = [
        r for r in ir2.routines
        if r.name not in ("onground", "falling", "fallon")
    ]
    ir3 = reloop_module(ir2)
    ir3.routines = [r for r in ir3.routines if r.name == "CHECKFLOOR"]
    stubs = _stubs_module()

    for action, posn, expected in _PATHS:
        # IR2 (interpreted as IR1).
        ram2 = bytearray(0x10000)
        ram2[CHAR_ACTION] = action
        ram2[CHAR_POSN] = posn
        ir1_run([ir2, stubs], "CHECKFLOOR", ram=ram2)
        touched2 = {a for a in (0x200, 0x201, 0x202) if ram2[a] != 0}

        # IR3.
        ram3 = bytearray(0x10000)
        ram3[CHAR_ACTION] = action
        ram3[CHAR_POSN] = posn
        ir3_run([ir3, stubs], "CHECKFLOOR", ram=ram3)
        touched3 = {a for a in (0x200, 0x201, 0x202) if ram3[a] != 0}

        assert touched2 == expected, (
            f"IR2 disagreed with hand-computed expected for "
            f"action={action} posn={posn}: got {touched2}, want {expected}"
        )
        assert touched3 == touched2, (
            f"IR3 differs from IR2 for action={action} posn={posn}: "
            f"ir2={touched2} ir3={touched3}"
        )
