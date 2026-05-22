"""Pass 2 structurer tests: cmp+branch fusion correctness, behavioural
equivalence with pass 1, and the CHECKFLOOR pilot end-to-end."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pop_lifter.ir1 import (
    Branch,
    CmpAbs,
    CmpImm,
    Compare,
    If,
    LoadAbs,
)
from pop_lifter.interp_ir1 import run
from pop_lifter.pass0_parse import parse_files
from pop_lifter.pass1_lift import lift_file
from pop_lifter.pass2_struct import (
    fusion_stats,
    structure_module,
    structure_routine,
)


def _ir1_module(source_dir: Path, file: str, entries: list[str]):
    ast = parse_files(
        [source_dir / "EQ.S", source_dir / "GAMEEQ.S", source_dir / file],
        search_paths=[source_dir],
    )
    file_ast = next(f for f in ast.files if Path(f.path).name == file)
    return lift_file(file_ast, ast.equates, entries).module


# ---- fusion correctness (CmpImm / CmpAbs / Load* + branch suffix)


def test_cmp_imm_plus_beq_fuses_to_equality(source_dir):
    """`cmp a, #6 ; beq ]rts` becomes `if a == #6 goto ]rts`."""
    ir1 = _ir1_module(source_dir, "CTRL.S", ["CHECKFLOOR"])
    ir2 = structure_module(ir1)
    cf = ir2.find("CHECKFLOOR")
    assert cf is not None

    # Find the first If; in CHECKFLOOR it's the `cmp #6 ; beq ]rts` pair
    # at source lines 108-109.
    first_if = next(item for item in cf.body if isinstance(item, If))
    assert isinstance(first_if.cond, Compare)
    assert first_if.cond.op == "=="
    assert first_if.cond.rhs.value == 6
    assert first_if.target == "]rts"


def test_cmp_bcs_fuses_to_ge(source_dir):
    """`cmp #106 ; bcs ]rts` — C=1 after cmp means reg >= operand."""
    ir2 = structure_module(_ir1_module(source_dir, "CTRL.S", ["CHECKFLOOR"]))
    cf = ir2.find("CHECKFLOOR")
    matching = [
        item for item in cf.body
        if isinstance(item, If) and item.cond.op == ">=" and item.cond.rhs.value == 0x6a
    ]
    assert len(matching) == 1


def test_cmp_bcc_fuses_to_lt(source_dir):
    """`cmp #102 ; bcc ]rts` — C=0 after cmp means reg < operand."""
    ir2 = structure_module(_ir1_module(source_dir, "CTRL.S", ["CHECKFLOOR"]))
    cf = ir2.find("CHECKFLOOR")
    matching = [
        item for item in cf.body
        if isinstance(item, If) and item.cond.op == "<" and item.cond.rhs.value == 0x66
    ]
    assert len(matching) == 1


def test_bitwise_and_branch_fuses(source_dir):
    """`and #fcheckmark ; beq ]rts` in `onground` — the pass-1 long-
    tail slice lifts `and` as `Bitwise`, which `pass2_struct` treats
    as a flag-setter (Z/N reflect A's post-and value). The branch
    must fuse into `if a == 0 goto ]rts`."""
    ir2 = structure_module(_ir1_module(source_dir, "CTRL.S", ["CHECKFLOOR"]))
    onground = ir2.find("onground")
    assert onground is not None
    # Find the If immediately following an `and` against fcheckmark.
    from pop_lifter.ir1 import Bitwise, If
    for prev, item in zip(onground.body, onground.body[1:]):
        if (
            isinstance(prev, Bitwise)
            and prev.op == "and"
            and isinstance(item, If)
            and item.target == "]rts"
        ):
            assert item.cond.op == "=="
            assert item.cond.rhs.value == 0
            return
    raise AssertionError(
        "no `and ; beq ]rts` pair fused in onground — fusion regression?"
    )


def test_unfused_branch_left_alone():
    """When the preceding op is still `Unsupported` (e.g. an opcode
    not yet lifted by pass 1), the Branch must NOT fuse. We use a
    synthetic routine with an `???` predecessor to exercise that
    contract independently of which real opcodes happen to be
    lifted in any given commit."""
    from pop_lifter.ir1 import (
        Branch,
        Reg,
        Return,
        Routine,
        SourceRef,
        Unsupported,
    )
    src = SourceRef(file="syn", line=0, raw="")
    r = Routine(
        name="syn",
        body=[
            Unsupported(mnemonic="bit", operand="$80", src=src),
            Branch(cond="eq", target="]rts", src=src),
            Return(src=src),
        ],
    )
    out = structure_routine(r)
    # Branch must remain unfused — Unsupported predecessors don't
    # expose a known affected register.
    assert any(isinstance(item, Branch) for item in out.body)
    from pop_lifter.ir1 import If
    assert not any(isinstance(item, If) for item in out.body)


def test_structurer_is_idempotent(source_dir):
    """Running pass 2 a second time on its own output must be a no-op."""
    ir1 = _ir1_module(source_dir, "CTRL.S", ["CHECKFLOOR"])
    once = structure_module(ir1)
    twice = structure_module(once)
    cf_once = once.find("CHECKFLOOR")
    cf_twice = twice.find("CHECKFLOOR")
    assert [type(i).__name__ for i in cf_once.body] == \
           [type(i).__name__ for i in cf_twice.body]


def test_load_plus_bpl_fuses_to_sign_test():
    """`lda foo ; bpl L` — Z/N reflect the loaded value, so this is a
    `>= 0` sign test on the value just loaded."""
    from pop_lifter.ir1 import Abs, ModuleIR1, Reg, Return, Routine, SourceRef

    src = SourceRef(file="synthetic", line=0, raw="")
    r = Routine(
        name="f",
        body=[
            LoadAbs(reg=Reg.A, source=Abs(name="foo", addr=0x100), src=src),
            Branch(cond="pl", target="end", src=src),
            Return(src=src),
        ],
    )
    structured = structure_routine(r)
    ifs = [item for item in structured.body if isinstance(item, If)]
    assert len(ifs) == 1
    assert ifs[0].cond.op == ">=0"
    assert ifs[0].cond.rhs is None  # sign tests have no rhs


def test_fusion_does_not_cross_unrelated_op():
    """A `sta` between the cmp and the branch must NOT fuse — stores
    don't define flags but pass 2's pending-flag tracking has to forget
    on anything that isn't a flag-setter."""
    from pop_lifter.ir1 import (
        Abs,
        Imm,
        ModuleIR1,
        Reg,
        Return,
        Routine,
        SourceRef,
        StoreAbs,
    )

    src = SourceRef(file="synthetic", line=0, raw="")
    r = Routine(
        name="f",
        body=[
            CmpImm(reg=Reg.A, imm=Imm(value=5, text="#5"), src=src),
            StoreAbs(reg=Reg.A, target=Abs(name="sink", addr=0x100), src=src),
            Branch(cond="eq", target="end", src=src),
            Return(src=src),
        ],
    )
    structured = structure_routine(r)
    # The cmp+sta+branch trio must not fuse — Z is no longer guaranteed
    # to reflect (a == 5) by the time we hit the branch (well, sta
    # doesn't change Z either, but the structurer can't know that yet,
    # so it stays conservative).
    assert any(isinstance(item, Branch) for item in structured.body)
    assert not any(isinstance(item, If) for item in structured.body)


# ---- behavioural equivalence: pass-2 CHECKFLOOR matches pass-1 across
# every branch path.


CHAR_ACTION = 0x46
CHAR_POSN = 0x40


def _checkfloor_modules(source_dir: Path):
    """Lift CHECKFLOOR through pass 1 and pass 2, returning both
    modules plus a stub module providing onground/falling/fallon."""
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

    ir1 = _ir1_module(source_dir, "CTRL.S", ["CHECKFLOOR"])
    ir1.routines = [
        r for r in ir1.routines
        if r.name not in ("onground", "falling", "fallon")
    ]

    src = SourceRef(file="synthetic", line=0, raw="")

    def _stub(name: str, sentinel: int) -> Routine:
        return Routine(
            name=name,
            body=[
                LoadImm(reg=Reg.A, imm=Imm(value=1, text="#1"), src=src),
                StoreAbs(
                    reg=Reg.A,
                    target=Abs(name=f"<{name}>", addr=sentinel),
                    src=src,
                ),
                Return(src=src),
            ],
        )

    stubs = ModuleIR1(
        name="STUBS", file="synthetic",
        routines=[
            _stub("onground", 0x200),
            _stub("falling", 0x201),
            _stub("fallon", 0x202),
        ],
    )

    ir2 = structure_module(ir1)
    return ir1, ir2, stubs


_PATHS = [
    # (action, posn, expected sentinels touched)
    (6, 0, set()),                # hanging early-exit
    (5, 109, {0x200}),            # → onground (crouched)
    (5, 185, {0x200}),            # → onground (dead)
    (5, 42, set()),               # bumped, other pose → return
    (4, 0, {0x201}),              # freefall → falling
    (3, 104, {0x202}),            # action 3, posn in range → fallon
    (3, 50, set()),               # too low
    (3, 200, set()),              # too high
    (2, 0, set()),                # hanging
    (0, 0, {0x200}),              # default → onground
    (1, 0, {0x200}),
    (7, 0, {0x200}),
]


def test_pass2_matches_pass1_across_every_checkfloor_path(source_dir):
    ir1, ir2, stubs = _checkfloor_modules(source_dir)
    for action, posn, expected in _PATHS:
        # Pass 1 run
        ram1 = bytearray(0x10000)
        ram1[CHAR_ACTION] = action
        ram1[CHAR_POSN] = posn
        run([ir1, stubs], "CHECKFLOOR", ram=ram1)
        touched1 = {a for a in (0x200, 0x201, 0x202) if ram1[a] != 0}

        # Pass 2 run
        ram2 = bytearray(0x10000)
        ram2[CHAR_ACTION] = action
        ram2[CHAR_POSN] = posn
        run([ir2, stubs], "CHECKFLOOR", ram=ram2)
        touched2 = {a for a in (0x200, 0x201, 0x202) if ram2[a] != 0}

        assert touched1 == expected, (
            f"pass-1 disagreed with hand-computed expected for "
            f"action={action} posn={posn}: got {touched1}, want {expected}"
        )
        assert touched2 == touched1, (
            f"pass-2 differs from pass-1 for action={action} posn={posn}: "
            f"pass1={touched1} vs pass2={touched2}"
        )


def test_pass2_fusion_count_matches_expected(source_dir):
    """Pin the headline fusion number for CHECKFLOOR + chase callees:
    the lifter's regression test on this counter catches accidental
    structurer regressions even when paths still execute correctly."""
    ir2 = structure_module(_ir1_module(source_dir, "CTRL.S", ["CHECKFLOOR"]))
    fused, unfused = fusion_stats(ir2)
    # 56 fused / 9 unfused right now; pin within a tolerance so future
    # lifter improvements (recognising more flag-defining ops) don't
    # require updating this test on every push.
    assert 50 <= fused <= 70, f"unexpected fused-If count: {fused}"
    assert unfused <= 20, f"too many unfused Branches: {unfused}"


# ---- flag-liveness elision


def test_elision_drops_every_cmp_in_checkfloor(source_dir):
    """Every `cmp` in CHECKFLOOR proper feeds a fused `If` (no flag
    readers between them and the next overwrite), so all 14 of them
    should be dropped by the liveness sweep. The callees onground /
    fallon / etc. still have unfused branches (their `and` predecessors
    aren't lifted yet) so those keep their cmps — but CHECKFLOOR
    itself should end up cmp-free."""
    from pop_lifter.ir1 import CmpAbs, CmpImm

    ir2 = structure_module(_ir1_module(source_dir, "CTRL.S", ["CHECKFLOOR"]))
    cf = ir2.find("CHECKFLOOR")
    cmps_left = [
        item for item in cf.body if isinstance(item, (CmpImm, CmpAbs))
    ]
    assert cmps_left == [], (
        f"expected CHECKFLOOR to have zero cmp left after elision, got "
        f"{[(c.imm.value if hasattr(c, 'imm') else c.source.name) for c in cmps_left]}"
    )


def test_elision_keeps_cmp_before_unfused_branch():
    """When an opaque op sits between the `cmp` and the next branch,
    the sweep treats the opaque op as read-all/write-all — so the
    `cmp`'s Z/N/C might be consumed by it before being overwritten.
    Shape: `cmp #N ; <opaque> ; bne L`. The cmp must NOT be elided.

    (The trailing `Branch` here is incidental — even if it were
    removed the Unsupported's read-all alone keeps the cmp alive.)
    """
    from pop_lifter.ir1 import (
        Branch,
        CmpImm,
        Imm,
        Reg,
        Return,
        Routine,
        SourceRef,
        Unsupported,
    )

    src = SourceRef(file="synthetic", line=0, raw="")
    r = Routine(
        name="f",
        body=[
            CmpImm(reg=Reg.A, imm=Imm(value=5, text="#5"), src=src),
            Unsupported(mnemonic="???", operand="and #1", src=src),
            Branch(cond="ne", target="]rts", src=src),
            Return(src=src),
            # synthetic local target so the Branch resolves locally
            # (we won't actually reach it).
        ],
    )
    # We need a ]rts label for the Branch to be "local". Add one.
    from pop_lifter.ir1 import Label
    r = replace(r, body=[*r.body, Label(name="]rts", src=src), Return(src=src)])

    from pop_lifter.pass2_struct import _eliminate_dead_flags
    out = _eliminate_dead_flags(r, flag_demand={r.name: frozenset()})
    assert any(isinstance(item, CmpImm) for item in out.body), (
        "cmp before an opaque flag-reader must NOT be elided — its "
        "flags are still live"
    )


def test_elision_keeps_clc_before_adc():
    """`clc; adc #1` — the clc is alive because adc reads C. Without
    it, adc would inherit whatever C the caller set."""
    from pop_lifter.ir1 import (
        AdcImm,
        Clc,
        Imm,
        Return,
        Routine,
        SourceRef,
    )

    src = SourceRef(file="synthetic", line=0, raw="")
    r = Routine(
        name="f",
        body=[
            Clc(src=src),
            AdcImm(imm=Imm(value=1, text="#1"), src=src),
            Return(src=src),
        ],
    )
    from pop_lifter.pass2_struct import _eliminate_dead_flags
    out = _eliminate_dead_flags(r, flag_demand={r.name: frozenset()})
    assert any(isinstance(item, Clc) for item in out.body), (
        "clc feeding adc must not be elided"
    )


def test_elision_drops_clc_with_no_adc():
    """A bare `clc` with no carry-reader downstream is dead."""
    from pop_lifter.ir1 import Clc, Return, Routine, SourceRef

    src = SourceRef(file="synthetic", line=0, raw="")
    r = Routine(
        name="f",
        body=[Clc(src=src), Return(src=src)],
    )
    from pop_lifter.pass2_struct import _eliminate_dead_flags
    out = _eliminate_dead_flags(r, flag_demand={r.name: frozenset()})
    assert not any(isinstance(item, Clc) for item in out.body)


def test_elision_bails_on_backward_branch():
    """If a routine contains a backward local jump (loop), elision
    bails — the single-pass liveness sweep isn't fixed-point and
    might drop a flag-setter that's still live across the back-edge.
    Verify the cmp survives."""
    from pop_lifter.ir1 import (
        Branch,
        CmpImm,
        Imm,
        Label,
        Reg,
        Return,
        Routine,
        SourceRef,
    )

    src = SourceRef(file="synthetic", line=0, raw="")
    # do { ... cmp; bne loop } — backward branch from bne to :loop
    r = Routine(
        name="f",
        body=[
            Label(name=":loop", src=src),
            CmpImm(reg=Reg.A, imm=Imm(value=0, text="#0"), src=src),
            Branch(cond="ne", target=":loop", src=src),
            Return(src=src),
        ],
    )
    from pop_lifter.pass2_struct import _eliminate_dead_flags
    out = _eliminate_dead_flags(r, flag_demand={r.name: frozenset()})
    assert any(isinstance(item, CmpImm) for item in out.body), (
        "elision should bail on a loop and keep the cmp"
    )


def test_call_graph_keeps_callee_return_flag_setter():
    """Soundness gate for the call-graph fixed point: a callee whose
    Z is read by the caller (the `cmpspace` idiom: `call X ; if ne
    goto ...`) must retain its terminal `cmp`, even though the
    `cmp`'s flags appear locally dead going into Return.

    Two routines in the same module:

      fn caller {
        call X
        if a != #0 goto ]rts     ; reads Z from X's terminal cmp
        return
      :rts: return                ; (loose ]rts shape — synthetic)
      }
      fn X {
        cmp a, #0x14              ; sets Z; this is X's "return value"
        return
      }

    Without call-graph propagation, X's cmp would be elided
    (flag_demand[X] = ∅) and the caller's `if ne` would read stale Z.
    With propagation, the caller's live-OUT at `call X` includes Z,
    flag_demand[X] becomes {Z}, and the cmp survives.
    """
    from pop_lifter.ir1 import (
        Branch,
        Call,
        CmpImm,
        If,
        Imm,
        Label,
        ModuleIR1,
        Reg,
        Return,
        Routine,
        SourceRef,
    )

    src = SourceRef(file="synthetic", line=0, raw="")
    rts_label = Label(name="]rts", src=src)

    caller = Routine(
        name="caller",
        body=[
            Call(target="X", src=src),
            Branch(cond="ne", target="]rts", src=src),
            rts_label,
            Return(src=src),
        ],
    )
    callee = Routine(
        name="X",
        body=[
            CmpImm(reg=Reg.A, imm=Imm(value=0x14, text="#$14"), src=src),
            Return(src=src),
        ],
    )

    m = ModuleIR1(
        name="syn", file="synthetic", routines=[caller, callee]
    )
    out = structure_module(m)
    out_callee = out.find("X")
    assert any(isinstance(item, CmpImm) for item in out_callee.body), (
        "callee's terminal cmp must survive — its Z is read by caller"
    )


def test_call_graph_drops_callee_cmp_when_caller_ignores_flags():
    """Mirror of the previous test: same callee shape but the caller
    doesn't read the return Z. flag_demand[X] stays ∅ so X's cmp
    drops."""
    from pop_lifter.ir1 import (
        Call,
        CmpImm,
        Imm,
        ModuleIR1,
        Reg,
        Return,
        Routine,
        SourceRef,
    )

    src = SourceRef(file="synthetic", line=0, raw="")
    caller = Routine(
        name="caller",
        body=[Call(target="X", src=src), Return(src=src)],
    )
    callee = Routine(
        name="X",
        body=[
            CmpImm(reg=Reg.A, imm=Imm(value=0x14, text="#$14"), src=src),
            Return(src=src),
        ],
    )

    m = ModuleIR1(name="syn", file="synthetic", routines=[caller, callee])
    out = structure_module(m)
    assert not any(
        isinstance(item, CmpImm) for item in out.find("X").body
    ), "callee cmp must drop when no caller reads return flags"


def test_call_graph_propagates_through_tail_call():
    """`tail_call X` from R inherits R's flag_demand to X. If R's
    callers read Z, X must keep its terminal cmp."""
    from pop_lifter.ir1 import (
        Branch,
        Call,
        CmpImm,
        Goto,
        Imm,
        Label,
        ModuleIR1,
        Reg,
        Return,
        Routine,
        SourceRef,
    )

    src = SourceRef(file="synthetic", line=0, raw="")
    # outer calls R; reads Z afterward.
    outer = Routine(
        name="outer",
        body=[
            Call(target="R", src=src),
            Branch(cond="ne", target="]rts", src=src),
            Label(name="]rts", src=src),
            Return(src=src),
        ],
    )
    # R tail-calls X — so outer's Z-demand should flow through R to X.
    r = Routine(
        name="R",
        body=[Goto(target="X", kind="tail_call", src=src)],
    )
    x = Routine(
        name="X",
        body=[
            CmpImm(reg=Reg.A, imm=Imm(value=0x14, text="#$14"), src=src),
            Return(src=src),
        ],
    )
    m = ModuleIR1(name="syn", file="synthetic", routines=[outer, r, x])
    out = structure_module(m)
    assert any(isinstance(item, CmpImm) for item in out.find("X").body), (
        "demand should propagate outer → R → X via the tail_call edge"
    )


def test_elision_preserves_behavioural_equivalence(source_dir):
    """Combined fusion + elision must still produce identical sentinel-
    touch sets across every CHECKFLOOR path — the strongest end-to-end
    check that the rewrite is sound."""
    ir1, ir2, stubs = _checkfloor_modules(source_dir)
    for action, posn, expected in _PATHS:
        ram1 = bytearray(0x10000)
        ram1[CHAR_ACTION] = action
        ram1[CHAR_POSN] = posn
        run([ir1, stubs], "CHECKFLOOR", ram=ram1)
        touched1 = {a for a in (0x200, 0x201, 0x202) if ram1[a] != 0}

        ram2 = bytearray(0x10000)
        ram2[CHAR_ACTION] = action
        ram2[CHAR_POSN] = posn
        run([ir2, stubs], "CHECKFLOOR", ram=ram2)
        touched2 = {a for a in (0x200, 0x201, 0x202) if ram2[a] != 0}
        assert touched2 == touched1 == expected, (
            f"divergence for action={action} posn={posn}: "
            f"pass1={touched1} pass2={touched2} expected={expected}"
        )
