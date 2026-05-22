"""Pass 2 structurer tests: cmp+branch fusion correctness, behavioural
equivalence with pass 1, and the CHECKFLOOR pilot end-to-end."""

from __future__ import annotations

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


def test_unfused_branch_left_alone(source_dir):
    """`and #fcheckmark ; beq ]rts` in `onground` — the `and` isn't yet
    a recognised flag-setter, so the branch must remain as a Branch
    (not silently rewritten as a sign or load test)."""
    ir2 = structure_module(_ir1_module(source_dir, "CTRL.S", ["CHECKFLOOR"]))
    onground = ir2.find("onground")
    assert onground is not None
    # There should be at least one literal `Branch` left in onground;
    # its target is `]rts` and it follows an Unsupported `and`.
    raw_branches = [item for item in onground.body if isinstance(item, Branch)]
    assert any(b.target == "]rts" and b.cond == "eq" for b in raw_branches)


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
