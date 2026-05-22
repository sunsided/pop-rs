"""Pass-1 stack tests: `pha` / `pla` lift + interpret + fuse, plus
the headline interleaving test that `pha ; jsr X ; pla` preserves
A across a subroutine call."""

from __future__ import annotations

import pytest

from pop_lifter.interp_ir1 import InterpError, Trace, exec_atom, run
from pop_lifter.ir1 import (
    Abs,
    Branch,
    Call,
    If,
    Imm,
    LoadImm,
    ModuleIR1,
    Pha,
    Pla,
    Reg,
    Return,
    Routine,
    SourceRef,
    StoreAbs,
)
from pop_lifter.pass0_lex import Line
from pop_lifter.pass1_lift import _lift_instr
from pop_lifter.pass2_struct import structure_routine


def _line(mnemonic: str, operand: str | None = None) -> Line:
    return Line(
        file="syn",
        lineno=1,
        raw=f"  {mnemonic} {operand or ''}".rstrip(),
        label=None,
        mnemonic=mnemonic,
        operand=operand,
        comment=None,
    )


def _trace(a=0) -> Trace:
    return Trace(ram=bytearray(0x10000), a=a, x=0, y=0)


# ---- lifter dispatch


def test_pha_lifts_to_pha():
    instr = _lift_instr(_line("pha"), {}, set())
    assert isinstance(instr, Pha)


def test_pla_lifts_to_pla():
    instr = _lift_instr(_line("pla"), {}, set())
    assert isinstance(instr, Pla)


# ---- interpreter semantics


def test_pha_pla_roundtrip_preserves_a():
    """`pha ; pla` is a no-op on A. The stack ends empty."""
    src = SourceRef(file="syn", line=0, raw="")
    t = _trace(a=0x42)
    exec_atom(Pha(src=src), t, t.ram)
    assert t.value_stack == [0x42]
    # Clobber A to confirm the pop actually writes A.
    t.a = 0
    exec_atom(Pla(src=src), t, t.ram)
    assert t.a == 0x42
    assert t.value_stack == []


def test_pla_sets_zn_flags():
    """PLA's Z/N reflect the popped value, like any other load."""
    src = SourceRef(file="syn", line=0, raw="")
    t = _trace()
    t.value_stack.append(0x80)
    exec_atom(Pla(src=src), t, t.ram)
    assert t.a == 0x80
    assert t.n == 1
    assert t.z == 0


def test_pla_on_empty_stack_raises_interperror():
    """Pop from an empty stack must surface a clear error pointing
    at the unbalanced pha/pla rather than silently giving a junk
    value."""
    src = SourceRef(file="syn", line=0, raw="")
    t = _trace()
    with pytest.raises(InterpError, match="empty value stack"):
        exec_atom(Pla(src=src), t, t.ram)


def test_max_value_stack_depth_tracked():
    """`max_value_stack_depth` mirrors the existing `max_stack_depth`
    so tests can pin worst-case stack usage."""
    src = SourceRef(file="syn", line=0, raw="")
    t = _trace()
    exec_atom(Pha(src=src), t, t.ram)
    exec_atom(Pha(src=src), t, t.ram)
    exec_atom(Pha(src=src), t, t.ram)
    assert t.max_value_stack_depth == 3
    exec_atom(Pla(src=src), t, t.ram)
    # Depth doesn't decrease when popping — it records the peak.
    assert t.max_value_stack_depth == 3


# ---- the headline test: pha; jsr X; pla preserves A across a call
#      even when X itself does its own pha/pla


def test_pha_jsr_pla_preserves_a_even_when_callee_pushes_too():
    """The whole point of `pha ; jsr X ; pla` is "save A across this
    call". Our two-stack design (a Python list in `run` for JSR/RTS
    continuations, `Trace.value_stack` for PHA/PLA bytes — see
    `ir1.Pha`) only holds together if each routine balances its own
    PHA/PLA. This test exercises the worst-case interleaving:

        main:
          a = $42
          pha            ← push outer A
          jsr inner      ← inner does its own pha/pla
          pla            ← must restore outer A = $42
          sta $80        ← stash so we can assert on it
          rts

        inner:
          a = $99
          pha            ← inner's own save
          a = $00        ← clobber A
          pla            ← restore inner's saved A ($99)
          rts

    Real hardware would interleave 4 bytes on the stack at peak
    depth — bottom-to-top: outer's `pha` writes $42, then JSR
    writes ret_hi + ret_lo, then inner's `pha` writes $99. RTS
    pops the 2 return-address bytes (leaving $42 and $99 with
    $99 on top), inner's `pla` pops $99, control returns, and
    outer's `pla` pops $42. Our model: each PHA/PLA pair is
    independent, the JSR/RTS is on a separate Python list, and as
    long as inner is well-balanced (pushes and pops once) outer's
    `pla` gets back its $42.
    """
    src = SourceRef(file="syn", line=0, raw="")

    main = Routine(
        name="main",
        body=[
            LoadImm(reg=Reg.A, imm=Imm(value=0x42, text="#$42"), src=src),
            Pha(src=src),
            Call(target="inner", src=src),
            Pla(src=src),
            StoreAbs(reg=Reg.A, target=Abs(name="out", addr=0x80), src=src),
            Return(src=src),
        ],
    )
    inner = Routine(
        name="inner",
        body=[
            LoadImm(reg=Reg.A, imm=Imm(value=0x99, text="#$99"), src=src),
            Pha(src=src),
            LoadImm(reg=Reg.A, imm=Imm(value=0x00, text="#$00"), src=src),
            Pla(src=src),
            Return(src=src),
        ],
    )
    mod = ModuleIR1(name="SYN", file="syn", routines=[main, inner])
    trace = run(mod, "main")

    assert trace.ram[0x80] == 0x42, (
        f"outer A wasn't preserved across the call; got "
        f"${trace.ram[0x80]:02x} instead of $42"
    )
    # Both stacks must end empty if everyone balanced their frames.
    assert trace.value_stack == []


# ---- pass-2 fusion


def test_pla_then_beq_fuses_to_zero_test_on_a():
    """`pla ; beq L` — Z/N reflect the popped value (now in A), so
    this fuses into `if a == 0 goto L` exactly like Lda*."""
    src = SourceRef(file="syn", line=0, raw="")
    r = Routine(
        name="f",
        body=[
            Pla(src=src),
            Branch(cond="eq", target="]rts", src=src),
            Return(src=src),
        ],
    )
    out = structure_routine(r)
    ifs = [i for i in out.body if isinstance(i, If)]
    assert len(ifs) == 1
    assert ifs[0].cond.reg is Reg.A
    assert ifs[0].cond.op == "=="


def test_pha_between_cmp_and_branch_blocks_fusion():
    """On real hardware PHA preserves Z/N — it would be perfectly
    sound to fuse `cmp #5 ; pha ; beq L` into `if a == 5 goto L`.
    But pass 2's fuser is conservative: anything that isn't a
    recognised flag-setter resets the pending-flag tracker, and
    PHA isn't on that list (no `_affected_register` entry, no
    `_defines_flags` entry, no special "preserves flags" carve-out).
    The cmp+beq pair therefore does NOT fuse when PHA sits between
    them — the branch stays a raw Branch.

    Pinning this as intentional behaviour (not a regression to
    chase) so a future tightening that adds PHA to the "preserves
    Z/N" list will be a deliberate test update, not silent."""
    from pop_lifter.ir1 import CmpImm

    src = SourceRef(file="syn", line=0, raw="")
    r = Routine(
        name="f",
        body=[
            CmpImm(reg=Reg.A, imm=Imm(value=5, text="#5"), src=src),
            Pha(src=src),
            Branch(cond="eq", target="]rts", src=src),
            Return(src=src),
        ],
    )
    out = structure_routine(r)
    assert any(isinstance(i, Branch) for i in out.body)
    assert not any(isinstance(i, If) for i in out.body)
