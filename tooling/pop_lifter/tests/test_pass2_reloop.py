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
    ir1 = lift_file(ctrl, ast.symbols(), ["CHECKFLOOR"]).module
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


def _two_back_edges_routine(extra=None):
    """Build a routine with two back-edges to the same header — the
    loop-relooper recognises only single-back-edge simple do-while
    shapes, so this defeats it and forces the fallback path. Pre-
    pends an `extra` IR1 item just inside the loop body if supplied
    (used to inject Call / IR1If into the fallback shape)."""
    from pop_lifter.ir1 import Branch, CmpImm, Imm, Label, Reg

    src = SourceRef(file="syn", line=0, raw="")
    body = [
        Label(name=":loop", src=src),
        CmpImm(reg=Reg.A, imm=Imm(value=0, text="#0"), src=src),
        Branch(cond="eq", target=":loop", src=src),   # back-edge #1
    ]
    if extra is not None:
        body.append(extra)
    body.extend([
        CmpImm(reg=Reg.A, imm=Imm(value=1, text="#1"), src=src),
        Branch(cond="ne", target=":loop", src=src),   # back-edge #2
        Return(src=src),
    ])
    return Routine(name="loopy", body=body)


def test_relooper_fallback_for_unstructurable_loops():
    """Two back-edges to the same header — outside the simple-
    do-while shape the relooper recognises. The routine must take
    the unstructured fallback path and emit `LabelStmt` +
    `RawIfStmt`-with-`GotoStmt`."""
    out = reloop_routine(_two_back_edges_routine())
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
    `TailCallStmt` in the then-block instead. Uses the two-back-edge
    shape to force the fallback path."""
    from pop_lifter.ir1 import Compare, If as IR1If, Imm, Reg

    src = SourceRef(file="syn", line=0, raw="")
    cross_module_if = IR1If(
        cond=Compare(reg=Reg.A, op="==", rhs=Imm(value=2, text="#2")),
        target="external_fn",
        src=src,
    )
    out = reloop_routine(_two_back_edges_routine(extra=cross_module_if))
    # Among the IfStmts in the fallback, find the one whose then-block
    # references external_fn — its then-block must hold a TailCallStmt.
    matched = False
    for s in out.body.stmts:
        if isinstance(s, IfStmt):
            for t in s.then_block.stmts:
                if isinstance(t, TailCallStmt) and t.target == "external_fn":
                    matched = True
                    break
                if isinstance(t, GotoStmt) and t.target == "external_fn":
                    raise AssertionError(
                        "cross-module If target lowered to GotoStmt instead "
                        "of TailCallStmt — fallback semantics regression"
                    )
    assert matched, (
        "cross-module If target must lower to TailCallStmt in the fallback"
    )


def test_fallback_ir1_call_becomes_callstmt():
    """In the fallback path, an IR1 `Call` must be emitted as a
    structured `CallStmt`, not folded into a `RawStmt`. Uses the
    two-back-edge shape to force the fallback path."""
    from pop_lifter.ir1 import Call as IR1Call
    from pop_lifter.ir3 import CallStmt

    src = SourceRef(file="syn", line=0, raw="")
    out = reloop_routine(_two_back_edges_routine(
        extra=IR1Call(target="helper", src=src),
    ))
    assert any(
        isinstance(s, CallStmt) and s.target == "helper"
        for s in out.body.stmts
    ), "IR1 Call must lower to IR3 CallStmt in the fallback"
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


# --------------------------------------------------------------- loops


def _structured_loop_count(routine):
    """Count `LoopStmt` nodes anywhere in `routine`'s body."""
    from pop_lifter.ir3 import LoopStmt

    seen = 0
    def walk(stmts):
        nonlocal seen
        for s in stmts:
            if isinstance(s, LoopStmt):
                seen += 1
                walk(s.body.stmts)
            inner_then = getattr(s, "then_block", None)
            if inner_then is not None:
                walk(inner_then.stmts)
            inner_else = getattr(s, "else_block", None)
            if inner_else is not None:
                walk(inner_else.stmts)
    walk(routine.body.stmts)
    return seen


def test_chgshadposn_loop_structures(source_dir):
    """AUTO.S `chgshadposn` is the cleanest do-while in the codebase:
    `:loop ... dey ; bpl :loop`. After the loop-relooper slice it
    must contain exactly one `LoopStmt`, no GotoStmt/LabelStmt."""
    from pop_lifter.ir3 import GotoStmt as G, LabelStmt as L
    ast = parse_files(
        [source_dir / "EQ.S", source_dir / "GAMEEQ.S", source_dir / "AUTO.S"],
        search_paths=[source_dir],
    )
    auto = next(f for f in ast.files if Path(f.path).name == "AUTO.S")
    ir1 = lift_file(auto, ast.symbols(), ["chgshadposn"]).module
    ir3 = reloop_module(structure_module(ir1))
    r = ir3.find("chgshadposn")
    assert _structured_loop_count(r) == 1, (
        "chgshadposn should contain exactly one LoopStmt"
    )
    # No fallback markers.
    def walk(stmts):
        for s in stmts:
            assert not isinstance(s, (G, L)), (
                f"unexpected unstructured stmt {type(s).__name__}"
            )
            inner = getattr(s, "then_block", None)
            if inner is not None:
                walk(inner.stmts)
            inner = getattr(s, "else_block", None)
            if inner is not None:
                walk(inner.stmts)
            inner = getattr(s, "body", None)
            if inner is not None and hasattr(inner, "stmts"):
                walk(inner.stmts)
    walk(r.body.stmts)


def test_loop_exit_cond_is_inverted():
    """A do-while shape `:hdr ... bpl :hdr` continues while N=0
    (`pl`). The structured form has the exit guard at the bottom:
    `if mi { break }`. Verify the inversion."""
    from pop_lifter.ir1 import Branch, CmpImm, Imm, Label, Reg
    from pop_lifter.ir3 import BreakStmt, LoopStmt, RawIfStmt

    src = SourceRef(file="syn", line=0, raw="")
    r = Routine(
        name="dec_loop",
        body=[
            Label(name=":hdr", src=src),
            CmpImm(reg=Reg.A, imm=Imm(value=0, text="#0"), src=src),
            Branch(cond="pl", target=":hdr", src=src),
            Return(src=src),
        ],
    )
    out = reloop_routine(r)
    loop = next(s for s in out.body.stmts if isinstance(s, LoopStmt))
    guard = next(
        s for s in loop.body.stmts if isinstance(s, RawIfStmt)
    )
    assert guard.cond == "mi", (
        f"expected pl→mi inversion at loop bottom, got cond={guard.cond!r}"
    )
    assert any(isinstance(t, BreakStmt) for t in guard.then_block.stmts)


def test_loop_with_fused_if_inverts_compare():
    """Same as above, but the back-edge is a fused `If(Compare)`.
    Verify the Compare's op is inverted (e.g. `!=` → `==`)."""
    from pop_lifter.ir1 import (
        Compare,
        If as IR1If,
        Imm,
        Label,
        Reg,
    )

    src = SourceRef(file="syn", line=0, raw="")
    r = Routine(
        name="cmp_loop",
        body=[
            Label(name=":hdr", src=src),
            IR1If(
                cond=Compare(reg=Reg.A, op="!=", rhs=Imm(value=5, text="#5")),
                target=":hdr",
                src=src,
            ),
            Return(src=src),
        ],
    )
    out = reloop_routine(r)
    from pop_lifter.ir3 import LoopStmt
    loop = next(s for s in out.body.stmts if isinstance(s, LoopStmt))
    guard = next(s for s in loop.body.stmts if isinstance(s, IfStmt))
    assert guard.cond.op == "==", (
        f"expected != → == inversion, got {guard.cond.op!r}"
    )


def test_synthetic_counter_loop_runs_correctly():
    """Behavioural gate for the loop-relooper. Construct a do-while
    counter loop using only lifted ops, structure it, and execute via
    the IR3 interpreter. The loop:

        a = 0
        :loop:
        store a → 0x100 + a   (so we can see iteration count via mem)
        c = 0
        a = a + 1
        cmp a, #5
        if a != 5 goto :loop
        return

    Post-state: mem[0x100..0x105] should hold 0..4 (the loop writes
    a=0,1,2,3,4 before the cmp == 5 fires). mem[0x80] is the
    accumulator's final value (5).
    """
    from pop_lifter.ir1 import (
        AdcImm,
        Abs,
        Clc,
        CmpImm,
        Compare,
        If as IR1If,
        Imm,
        Label,
        LoadImm,
        Reg,
        Routine,
        StoreAbs,
        StoreIndexed,
    )

    src = SourceRef(file="syn", line=0, raw="")
    OUT = Abs(name="out", addr=0x100)

    body = [
        # x = 0  (used as the indexed-store offset; same value as a)
        LoadImm(reg=Reg.X, imm=Imm(value=0, text="#0"), src=src),
        LoadImm(reg=Reg.A, imm=Imm(value=0, text="#0"), src=src),
        Label(name=":loop", src=src),
        StoreIndexed(reg=Reg.A, base=OUT, index=Reg.X, src=src),
        # Bump x and a in lockstep.
        Clc(src=src),
        AdcImm(imm=Imm(value=1, text="#1"), src=src),
        # Reload x from a via a memory hop: write a to 0x80, then
        # read x from 0x80. (No `tax` in IR1 yet.)
        StoreAbs(
            reg=Reg.A,
            target=Abs(name="ctr", addr=0x80),
            src=src,
        ),
        # x = mem[0x80]
        # We don't have LoadAbs to X with a literal address... we do.
        # Use LoadAbs with reg=X.
    ]
    from pop_lifter.ir1 import LoadAbs
    body.append(LoadAbs(reg=Reg.X, source=Abs(name="ctr", addr=0x80), src=src))
    body.extend([
        IR1If(
            cond=Compare(reg=Reg.A, op="!=", rhs=Imm(value=5, text="#5")),
            target=":loop",
            src=src,
        ),
        Return(src=src),
    ])
    routine = Routine(name="counter", body=body)
    mod = ModuleIR1(name="SYN", file="syn", routines=[routine])
    ir3_mod = reloop_module(structure_module(mod))

    from pop_lifter.ir3 import LoopStmt
    counter = ir3_mod.find("counter")
    assert any(isinstance(s, LoopStmt) for s in counter.body.stmts), (
        "synthetic counter loop didn't structure as LoopStmt"
    )

    ram = bytearray(0x10000)
    ir3_run([ir3_mod], "counter", ram=ram)
    assert list(ram[0x100:0x105]) == [0, 1, 2, 3, 4], (
        f"loop did not iterate 5 times; got {list(ram[0x100:0x105])}"
    )
    assert ram[0x80] == 5, (
        f"final counter value should be 5, got {ram[0x80]}"
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
    ir1 = lift_file(ctrl, ast.symbols(), ["CHECKFLOOR"]).module
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


# --------------------------------------------------------------- merge dedup


def _flatten(stmts):
    """Yield every statement in `stmts`, descending into if/loop bodies."""
    for s in stmts:
        yield s
        for attr in ("then_block", "else_block", "body"):
            blk = getattr(s, attr, None)
            if blk is not None:
                yield from _flatten(blk.stmts)


def _diamond_module():
    """A diamond: `if a == 1` taken→A, fall-through→B, both reconverging
    at a shared tail T. The relooper's post-dominator merge should emit
    T exactly once (in the continuation after an `if/else`), not inline
    it into both arms."""
    from pop_lifter.ir1 import Compare, Goto, Label, LoadAbs
    from pop_lifter.ir1 import If as IR1If

    src = SourceRef(file="syn", line=0, raw="")
    M10 = Abs(name="m10", addr=0x10)
    M11 = Abs(name="m11", addr=0x11)
    INP = Abs(name="inp", addr=0x12)

    def store(val, target):
        return [
            LoadImm(reg=Reg.A, imm=Imm(value=val, text=f"#{val}"), src=src),
            StoreAbs(reg=Reg.A, target=target, src=src),
        ]

    body = [
        # Load the dispatch value from memory so a test can drive both arms.
        LoadAbs(reg=Reg.A, source=INP, src=src),
        IR1If(cond=Compare(reg=Reg.A, op="==", rhs=Imm(value=1, text="#1")),
              target=":taken", src=src),
        # fall-through arm B
        *store(0xB1, M10),
        Goto(target=":merge", kind="local", src=src),
        Label(name=":taken", src=src),
        *store(0xA1, M10),
        # implicit fall-through into :merge
        Label(name=":merge", src=src),
        *store(0xCC, M11),       # shared tail — must appear once
        Return(src=src),
    ]
    return ModuleIR1(name="SYN", file="syn", routines=[Routine(name="diamond", body=body)])


def test_postdom_merge_emits_shared_tail_once():
    mod = _diamond_module()
    ir2 = structure_module(mod)
    ir3 = reloop_module(ir2)
    diamond = ir3.find("diamond")

    # The conditional must structure with a real else-block (the merge
    # optimization fired) rather than inlining the continuation.
    ifs = [s for s in diamond.body.stmts if isinstance(s, IfStmt)]
    assert ifs and ifs[0].else_block is not None, (
        "expected an if/else from the post-dominator merge"
    )

    # The shared tail store (mem[0x11] = #0xCC) must be emitted exactly
    # once, not duplicated into both arms.
    tail = [
        s for s in _flatten(diamond.body.stmts)
        if isinstance(s, RawStmt) and isinstance(s.item, StoreAbs)
        and s.item.target.addr == 0x11
    ]
    assert len(tail) == 1, f"shared tail emitted {len(tail)} times, expected 1"


def _three_way_merge_module():
    """A 3-predecessor merge: `a==1`→M, `a==2`→M, else fall→M. M's body
    must be emitted exactly once (the old walker inlined it per path),
    here as a clean nested `if/else` since the merge is the tail."""
    from pop_lifter.ir1 import Compare, Label, LoadAbs
    from pop_lifter.ir1 import If as IR1If

    src = SourceRef(file="syn", line=0, raw="")
    SINK, OUT, INP = Abs(name="sink", addr=0x30), Abs(name="out", addr=0x31), Abs(name="inp", addr=0x32)

    def mark(reg_val, target):
        return [
            LoadImm(reg=Reg.X, imm=Imm(value=reg_val, text=f"#{reg_val}"), src=src),
            StoreAbs(reg=Reg.X, target=target, src=src),
        ]

    body = [
        LoadAbs(reg=Reg.A, source=INP, src=src),
        IR1If(cond=Compare(reg=Reg.A, op="==", rhs=Imm(value=1, text="#1")), target=":m", src=src),
        *mark(0xB1, SINK),
        IR1If(cond=Compare(reg=Reg.A, op="==", rhs=Imm(value=2, text="#2")), target=":m", src=src),
        *mark(0xB2, SINK),
        Label(name=":m", src=src),
        *mark(0xCC, OUT),
        Return(src=src),
    ]
    return ModuleIR1(name="SYN", file="syn", routines=[Routine(name="tri", body=body)])


def test_three_way_merge_emitted_once():
    mod = _three_way_merge_module()
    ir3 = reloop_module(structure_module(mod))
    tri = ir3.find("tri")
    # M's distinctive store (out = #0xCC) is emitted exactly once — the
    # dedup the relooper now guarantees (the old walker emitted it ~3x).
    outs = [
        s for s in _flatten(tri.body.stmts)
        if isinstance(s, RawStmt) and isinstance(s.item, StoreAbs)
        and s.item.target.addr == 0x31
    ]
    assert len(outs) == 1, f"merge tail emitted {len(outs)} times"


def test_three_way_merge_preserves_behaviour():
    mod = _three_way_merge_module()
    ir2 = structure_module(mod)
    ir3 = reloop_module(ir2)
    for inp, exp_sink, exp_out in [(1, 0x00, 0xCC), (2, 0xB1, 0xCC), (5, 0xB2, 0xCC)]:
        ram2 = bytearray(0x10000)
        ram2[0x32] = inp
        ir1_run([ir2], "tri", ram=ram2)
        ram3 = bytearray(0x10000)
        ram3[0x32] = inp
        ir3_run([ir3], "tri", ram=ram3)
        assert (ram3[0x30], ram3[0x31]) == (exp_sink, exp_out), f"ir3 wrong for inp={inp}"
        assert ram2[0x30:0x32] == ram3[0x30:0x32], f"ir1/ir3 differ for inp={inp}"


def test_postdom_merge_preserves_behaviour():
    mod = _diamond_module()
    ir2 = structure_module(mod)
    ir3 = reloop_module(ir2)

    # inp==1 exercises the taken arm (mem[0x10]=0xA1); any other value
    # exercises the else arm (mem[0x10]=0xB1). Both share the tail
    # mem[0x11]=0xCC. Check each against IR1 so neither arm regresses.
    for inp, exp_m10 in [(1, 0xA1), (0, 0xB1), (7, 0xB1)]:
        ram2 = bytearray(0x10000)
        ram2[0x12] = inp
        ir1_run([ir2], "diamond", ram=ram2)
        ram3 = bytearray(0x10000)
        ram3[0x12] = inp
        ir3_run([ir3], "diamond", ram=ram3)
        assert (ram3[0x10], ram3[0x11]) == (exp_m10, 0xCC), f"ir3 wrong for inp={inp}"
        assert ram2[0x10:0x12] == ram3[0x10:0x12], f"ir1/ir3 differ for inp={inp}"
