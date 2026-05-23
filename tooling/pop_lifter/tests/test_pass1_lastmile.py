"""Pass-1 last-mile cleanup: Merlin `.` bitwise-OR operator,
`sbc (zp),y` indirect, and `inc/dec :label+N` self-modifying-code
operand bumps."""

from __future__ import annotations

from pop_lifter.interp_ir1 import Trace, exec_atom
from pop_lifter.ir1 import (
    Abs,
    DecTarget,
    IncTarget,
    IndirectY,
    LocalRef,
    SbcIndirect,
    SourceRef,
)
from pop_lifter.pass0_parse import eval_expr
from pop_lifter.pass0_lex import Line
from pop_lifter.pass1_lift import _lift_instr


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


# ---- Merlin `.` bitwise-OR operator


def test_dot_operator_is_bitwise_or():
    """Merlin's `.` is bitwise OR: `$40.$01` = `$41`."""
    assert eval_expr("$40.$01", {}) == 0x41


def test_dot_operator_with_symbol():
    """`sym.$80` ORs $80 into the symbol's value."""
    assert eval_expr("flag.$80", {"flag": 0x05}) == 0x85


def test_dot_operator_lifts_immediate():
    """`lda #sta.$40` no longer falls through to Unsupported — it
    resolves `sta` (a program label) ORed with $40. The exact value
    depends on the synthetic label address, so just assert the lift
    succeeds and the low byte reflects the OR."""
    from pop_lifter.ir1 import LoadImm
    instr = _lift_instr(_line("lda", "#flag.$40"), {"flag": 0x01}, set())
    assert isinstance(instr, LoadImm)
    # 0x01 | 0x40 = 0x41; immediate takes the low byte.
    assert instr.imm.value & 0xff == 0x41


def test_dot_does_not_break_plain_identifiers():
    """A symbol with no dot still resolves normally — the tokenizer
    change mustn't swallow or split ordinary identifiers."""
    assert eval_expr("CharX", {"CharX": 0x41}) == 0x41


# ---- sbc (zp),y


def test_sbc_indirect_lifts():
    instr = _lift_instr(_line("sbc", "(IMAGE),y"), {"IMAGE": 0x06}, set())
    assert isinstance(instr, SbcIndirect)
    assert instr.source.ptr.addr == 0x06


def test_sbc_indirect_subtracts_through_pointer():
    """`sbc (ptr),y` with C=1 subtracts the post-indexed byte from A."""
    src = SourceRef(file="syn", line=0, raw="sbc (ptr),y")
    t = Trace(ram=bytearray(0x10000), a=0x10, x=0, y=2, c=1)
    t.ram[0x06] = 0x00   # pointer lo
    t.ram[0x07] = 0x30   # pointer hi → $3000
    t.ram[0x3002] = 0x03  # operand at $3000 + y(2)
    exec_atom(
        SbcIndirect(source=IndirectY(ptr=Abs(name="ptr", addr=0x06)), src=src),
        t, t.ram,
    )
    assert t.a == 0x0d


# ---- inc/dec :label+N (SMC operand bump)


def test_inc_local_lifts_to_inctarget_localref():
    instr = _lift_instr(_line("inc", ":smod+2"), {}, set())
    assert isinstance(instr, IncTarget)
    assert isinstance(instr.target, LocalRef)
    assert instr.target.label == ":smod"
    assert instr.target.offset == 2


def test_dec_local_lifts_to_dectarget_localref():
    instr = _lift_instr(_line("dec", ":loop+2"), {}, set())
    assert isinstance(instr, DecTarget)
    assert isinstance(instr.target, LocalRef)
    assert instr.target.offset == 2


def test_inc_local_bumps_code_patch_slot():
    """`inc :smod+2` increments the SMC operand byte tracked in
    `code_patches`, seeding 0 if it wasn't patched yet."""
    src = SourceRef(file="syn", line=0, raw="inc :smod+2")
    t = Trace(ram=bytearray(0x10000), a=0, x=0, y=0)
    exec_atom(
        IncTarget(target=LocalRef(label=":smod", offset=2), src=src),
        t, t.ram,
    )
    assert t.code_patches[(":smod", 2)] == 1
    # A second bump continues from the stored value.
    exec_atom(
        IncTarget(target=LocalRef(label=":smod", offset=2), src=src),
        t, t.ram,
    )
    assert t.code_patches[(":smod", 2)] == 2


def test_inc_local_sets_zn_on_result():
    """The bump sets Z/N from the new value (a memory inc does too)."""
    src = SourceRef(file="syn", line=0, raw="inc :smod+2")
    t = Trace(ram=bytearray(0x10000), a=0, x=0, y=0)
    t.code_patches[(":smod", 2)] = 0xff
    exec_atom(
        IncTarget(target=LocalRef(label=":smod", offset=2), src=src),
        t, t.ram,
    )
    # 0xff + 1 = 0x00 → Z=1.
    assert t.code_patches[(":smod", 2)] == 0x00
    assert t.z == 1
