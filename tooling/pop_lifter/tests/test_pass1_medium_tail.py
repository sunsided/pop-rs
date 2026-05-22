"""Pass-1 medium-tail tests: per-opcode lift + interpret + fuse
coverage for `sbc`, `lsr`, and `bit`."""

from __future__ import annotations

from pop_lifter.interp_ir1 import Trace, exec_atom
from pop_lifter.ir1 import (
    Abs,
    Bit,
    Branch,
    If,
    Imm,
    Lsr,
    Reg,
    Return,
    Routine,
    SbcAbs,
    SbcImm,
    SourceRef,
    Unsupported,
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


def _trace(a=0, x=0, y=0, c=0) -> Trace:
    return Trace(ram=bytearray(0x10000), a=a, x=x, y=y, c=c)


# ---- lifter dispatch


def test_sbc_imm_lifts_to_sbcimm():
    instr = _lift_instr(_line("sbc", "#$01"), {}, set())
    assert isinstance(instr, SbcImm)
    assert instr.imm.value == 1


def test_sbc_abs_lifts_to_sbcabs():
    instr = _lift_instr(_line("sbc", "$0200"), {}, set())
    assert isinstance(instr, SbcAbs)
    assert instr.source.addr == 0x200


def test_lsr_accumulator_lifts_to_lsr():
    """`lsr a` and `lsr` (no operand) both mean accumulator shift."""
    assert isinstance(_lift_instr(_line("lsr", "a"), {}, set()), Lsr)
    assert isinstance(_lift_instr(_line("lsr"), {}, set()), Lsr)


def test_lsr_memory_form_stays_unsupported():
    """`lsr addr` (memory shift) needs a separate node — pinned as
    Unsupported so a future overgeneralisation can't silently lift
    it into the accumulator form."""
    instr = _lift_instr(_line("lsr", "$80"), {}, set())
    assert isinstance(instr, Unsupported)


def test_bit_imm_lifts_to_bit_with_imm_source():
    instr = _lift_instr(_line("bit", "#$80"), {}, set())
    assert isinstance(instr, Bit)
    assert isinstance(instr.source, Imm)
    assert instr.source.value == 0x80


def test_bit_abs_lifts_to_bit_with_abs_source():
    instr = _lift_instr(_line("bit", "$03f0"), {}, set())
    assert isinstance(instr, Bit)
    assert isinstance(instr.source, Abs)
    assert instr.source.addr == 0x3f0


# ---- interpreter semantics


def test_sbc_with_carry_set_subtracts_cleanly():
    """`sec ; sbc #1` from A=$05 should give A=$04, C=1 (no borrow)."""
    src = SourceRef(file="syn", line=0, raw="")
    t = _trace(a=0x05, c=1)
    exec_atom(SbcImm(imm=Imm(value=1, text="#1"), src=src), t, t.ram)
    assert t.a == 0x04
    assert t.c == 1
    assert t.z == 0


def test_sbc_with_borrow_propagation():
    """`sbc #1` from A=$05 with C=0 (borrow set) gives A=$03 — the
    incoming borrow subtracts an extra 1."""
    src = SourceRef(file="syn", line=0, raw="")
    t = _trace(a=0x05, c=0)
    exec_atom(SbcImm(imm=Imm(value=1, text="#1"), src=src), t, t.ram)
    assert t.a == 0x03


def test_sbc_underflow_clears_carry_and_wraps():
    """`sec ; sbc #2` from A=$01 underflows: A wraps to $ff, C=0
    (a borrow occurred), N=1 (high bit of result)."""
    src = SourceRef(file="syn", line=0, raw="")
    t = _trace(a=0x01, c=1)
    exec_atom(SbcImm(imm=Imm(value=2, text="#2"), src=src), t, t.ram)
    assert t.a == 0xff
    assert t.c == 0
    assert t.n == 1


def test_sbc_abs_reads_memory_subtrahend():
    src = SourceRef(file="syn", line=0, raw="")
    t = _trace(a=0x10, c=1)
    t.ram[0x200] = 0x03
    exec_atom(SbcAbs(source=Abs(name="m", addr=0x200), src=src), t, t.ram)
    assert t.a == 0x0d


def test_lsr_shifts_right_and_captures_bit0_into_carry():
    """`lsr` of $05 → A=$02, C=1 (the lost bit 0)."""
    src = SourceRef(file="syn", line=0, raw="")
    t = _trace(a=0x05)
    exec_atom(Lsr(src=src), t, t.ram)
    assert t.a == 0x02
    assert t.c == 1
    assert t.z == 0
    assert t.n == 0           # lsr shifts in 0, so N is always 0


def test_lsr_of_zero_sets_z_clears_c():
    src = SourceRef(file="syn", line=0, raw="")
    t = _trace(a=0)
    exec_atom(Lsr(src=src), t, t.ram)
    assert t.a == 0
    assert t.c == 0
    assert t.z == 1


def test_bit_does_not_modify_a():
    """`bit` is a pure flag-setter — A must be unchanged."""
    src = SourceRef(file="syn", line=0, raw="")
    t = _trace(a=0x55)
    t.ram[0x100] = 0xaa
    exec_atom(Bit(source=Abs(name="m", addr=0x100), src=src), t, t.ram)
    assert t.a == 0x55


def test_bit_z_reflects_anded_value():
    """Z = (A AND operand) == 0."""
    src = SourceRef(file="syn", line=0, raw="")
    # 0x55 & 0xaa = 0
    t = _trace(a=0x55)
    t.ram[0x100] = 0xaa
    exec_atom(Bit(source=Abs(name="m", addr=0x100), src=src), t, t.ram)
    assert t.z == 1


def test_bit_n_comes_from_operand_bit7_not_anded_value():
    """N reflects bit 7 of the OPERAND, regardless of A. This is the
    quirk that makes `bit` useful for status-register probes — you
    can read a flag bit without changing A."""
    src = SourceRef(file="syn", line=0, raw="")
    # A=0 ⇒ AND result is 0 (Z=1). But operand bit 7 is set, so N=1
    # even though the AND result has no bits set.
    t = _trace(a=0x00)
    exec_atom(Bit(source=Imm(value=0x80, text="#$80"), src=src), t, t.ram)
    assert t.n == 1
    assert t.z == 1


# ---- pass-2 fusion


def test_sbc_then_beq_fuses_to_zero_test_on_a():
    """`sbc #imm ; beq L` → `if a == 0 goto L`. The SBC result lives
    in A, so Z reflects A's new value just like Adc."""
    src = SourceRef(file="syn", line=0, raw="")
    r = Routine(
        name="f",
        body=[
            SbcImm(imm=Imm(value=1, text="#1"), src=src),
            Branch(cond="eq", target="]rts", src=src),
            Return(src=src),
        ],
    )
    out = structure_routine(r)
    ifs = [i for i in out.body if isinstance(i, If)]
    assert len(ifs) == 1
    assert ifs[0].cond.reg is Reg.A
    assert ifs[0].cond.op == "=="
    assert ifs[0].cond.rhs.value == 0


def test_lsr_then_bcc_fuses_to_low_bit_zero():
    """`lsr ; bcc L` continues while old bit 0 was 0 — i.e. when
    the next branch reads C after the shift. With our pass-2 fusion
    rules (`bcs`→`>=`, `bcc`→`<` after `cmp`), only cmp+branch
    paths fuse on C; lsr+bcc doesn't because `_affected_register`
    for Lsr returns Reg.A and the LOAD_FUSE_OPS map only handles
    eq/ne/pl/mi. The branch must therefore remain unfused.

    This test pins the conservative-fallback contract so a future
    over-fusion that conflates Lsr's C-side-effect with cmp's
    can't slip through."""
    src = SourceRef(file="syn", line=0, raw="")
    r = Routine(
        name="f",
        body=[
            Lsr(src=src),
            Branch(cond="cc", target="]rts", src=src),
            Return(src=src),
        ],
    )
    out = structure_routine(r)
    assert any(isinstance(i, Branch) for i in out.body)
    assert not any(isinstance(i, If) for i in out.body)


def test_bit_then_beq_does_not_fuse():
    """`bit operand ; beq L` is *semantically* `if (a & operand) == 0
    goto L`, but our Compare form has no masked-equality variant.
    Fusing into `if a == 0 goto L` would be WRONG. The branch must
    stay a raw Branch until pass 3 introduces an expression-bearing
    Compare."""
    src = SourceRef(file="syn", line=0, raw="")
    r = Routine(
        name="f",
        body=[
            Bit(source=Abs(name="flag", addr=0x80), src=src),
            Branch(cond="eq", target="]rts", src=src),
            Return(src=src),
        ],
    )
    out = structure_routine(r)
    assert any(isinstance(i, Branch) for i in out.body)
    assert not any(isinstance(i, If) for i in out.body)


def test_bit_eligible_for_elision_when_flags_dead():
    """A `bit` with no downstream Z/N/V reader (i.e. flowing into a
    Return or into another full-flag-writer like a cmp) is dead and
    must be removed by the elision sweep.

    `structure_routine` only fuses cmp+branch — elision is a separate
    backward pass that `structure_module` chains on top. We invoke
    both here to exercise the full pass-2 pipeline that pop-lifter's
    CLI runs."""
    from pop_lifter.pass2_struct import _eliminate_dead_flags

    src = SourceRef(file="syn", line=0, raw="")
    r = Routine(
        name="f",
        body=[
            Bit(source=Imm(value=0x80, text="#$80"), src=src),
            # No flag reader between Bit and Return.
            Return(src=src),
        ],
    )
    fused = structure_routine(r)
    # `_eliminate_dead_flags` takes a `flag_demand` map (per-routine
    # exit-flag liveness). For a standalone routine with no callers
    # in the analysis the demand is empty.
    out = _eliminate_dead_flags(fused, flag_demand={})
    assert not any(isinstance(i, Bit) for i in out.body), (
        "dead Bit should be elided by pass 2's flag-liveness sweep"
    )
