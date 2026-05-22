"""Pass-1 long-tail tests: per-opcode lift + interpret + fuse
coverage for the index inc/dec, register transfers, memory inc/dec
and bitwise families."""

from __future__ import annotations

from pop_lifter.interp_ir1 import Trace, exec_atom
from pop_lifter.ir1 import (
    Abs,
    Bitwise,
    Branch,
    DecTarget,
    If,
    Imm,
    IncTarget,
    Reg,
    Return,
    Routine,
    SourceRef,
    Transfer,
)
from pop_lifter.pass1_lift import _lift_instr
from pop_lifter.pass0_lex import Line
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


# ---- lifter coverage


def test_inx_lifts_to_inctarget_x():
    instr = _lift_instr(_line("inx"), {}, set())
    assert isinstance(instr, IncTarget)
    assert instr.target is Reg.X


def test_dey_lifts_to_dectarget_y():
    instr = _lift_instr(_line("dey"), {}, set())
    assert isinstance(instr, DecTarget)
    assert instr.target is Reg.Y


def test_inc_addr_lifts_to_inctarget_abs():
    instr = _lift_instr(_line("inc", "$0080"), {}, set())
    assert isinstance(instr, IncTarget)
    assert isinstance(instr.target, Abs)
    assert instr.target.addr == 0x80


def test_tax_lifts_to_transfer():
    instr = _lift_instr(_line("tax"), {}, set())
    assert isinstance(instr, Transfer)
    assert instr.src_reg is Reg.A and instr.dst_reg is Reg.X


def test_tya_lifts_to_transfer():
    instr = _lift_instr(_line("tya"), {}, set())
    assert isinstance(instr, Transfer)
    assert instr.src_reg is Reg.Y and instr.dst_reg is Reg.A


def test_and_imm_lifts_to_bitwise():
    instr = _lift_instr(_line("and", "#$0f"), {}, set())
    assert isinstance(instr, Bitwise)
    assert instr.op == "and"
    assert isinstance(instr.source, Imm)
    assert instr.source.value == 0x0f


def test_ora_abs_lifts_to_bitwise_abs():
    instr = _lift_instr(_line("ora", "$0200"), {}, set())
    assert isinstance(instr, Bitwise)
    assert instr.op == "or"
    assert isinstance(instr.source, Abs)
    assert instr.source.addr == 0x200


def test_eor_imm_lifts_to_bitwise():
    instr = _lift_instr(_line("eor", "#$ff"), {}, set())
    assert isinstance(instr, Bitwise)
    assert instr.op == "eor"
    assert instr.source.value == 0xff


# ---- interpreter coverage (exec_atom)


def _trace(a=0, x=0, y=0) -> Trace:
    return Trace(ram=bytearray(0x10000), a=a, x=x, y=y)


def test_inx_increments_x_and_sets_zn():
    t = _trace(x=0xff)
    src = SourceRef(file="syn", line=0, raw="")
    exec_atom(IncTarget(target=Reg.X, src=src), t, t.ram)
    assert t.x == 0          # 0xff + 1 = 0x100 -> 0x00
    assert t.z == 1
    assert t.n == 0


def test_dex_underflows_to_ff_and_sets_n():
    t = _trace(x=0)
    src = SourceRef(file="syn", line=0, raw="")
    exec_atom(DecTarget(target=Reg.X, src=src), t, t.ram)
    assert t.x == 0xff
    assert t.z == 0
    assert t.n == 1


def test_inc_addr_increments_memory():
    t = _trace()
    t.ram[0x80] = 0x41
    src = SourceRef(file="syn", line=0, raw="")
    exec_atom(IncTarget(target=Abs(name="m", addr=0x80), src=src), t, t.ram)
    assert t.ram[0x80] == 0x42


def test_tax_copies_a_to_x_and_sets_zn():
    t = _trace(a=0x80)
    src = SourceRef(file="syn", line=0, raw="")
    exec_atom(Transfer(src_reg=Reg.A, dst_reg=Reg.X, src=src), t, t.ram)
    assert t.x == 0x80
    assert t.n == 1
    assert t.z == 0


def test_and_imm_masks_a():
    t = _trace(a=0xff)
    src = SourceRef(file="syn", line=0, raw="")
    exec_atom(Bitwise(op="and", source=Imm(value=0x0f, text="#$0f"), src=src), t, t.ram)
    assert t.a == 0x0f
    assert t.z == 0


def test_eor_self_zeroes_a():
    t = _trace(a=0xc7)
    src = SourceRef(file="syn", line=0, raw="")
    exec_atom(Bitwise(op="eor", source=Imm(value=0xc7, text="#$c7"), src=src), t, t.ram)
    assert t.a == 0
    assert t.z == 1


def test_ora_abs_combines_a_with_memory():
    t = _trace(a=0x0f)
    t.ram[0x200] = 0x30
    src = SourceRef(file="syn", line=0, raw="")
    exec_atom(Bitwise(op="or", source=Abs(name="m", addr=0x200), src=src), t, t.ram)
    assert t.a == 0x3f


def test_inctarget_on_reg_a_raises_interperror():
    """`ina`/`dea` don't exist on stock NMOS 6502, so a hand-built
    IncTarget(Reg.A) must surface a clear InterpError rather than
    a KeyError into a register-lookup dict."""
    import pytest

    from pop_lifter.interp_ir1 import InterpError

    src = SourceRef(file="syn", line=0, raw="")
    t = _trace()
    with pytest.raises(InterpError, match="Reg.A is not a valid"):
        exec_atom(IncTarget(target=Reg.A, src=src), t, t.ram)
    with pytest.raises(InterpError, match="Reg.A is not a valid"):
        exec_atom(DecTarget(target=Reg.A, src=src), t, t.ram)


# ---- pass-2 fusion coverage


def test_bitwise_and_then_beq_fuses_to_zero_test():
    """`and #mask ; beq L` → `if a == 0 goto L` (A's post-and value)."""
    src = SourceRef(file="syn", line=0, raw="")
    r = Routine(
        name="f",
        body=[
            Bitwise(op="and", source=Imm(value=0x0f, text="#$0f"), src=src),
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


def test_dey_then_bpl_fuses_to_sign_test():
    """`dey ; bpl :loop` → `if y >= 0 goto :loop` (the classic
    counter-loop continue condition)."""
    src = SourceRef(file="syn", line=0, raw="")
    r = Routine(
        name="f",
        body=[
            DecTarget(target=Reg.Y, src=src),
            Branch(cond="pl", target=":loop", src=src),
            Return(src=src),
        ],
    )
    out = structure_routine(r)
    ifs = [i for i in out.body if isinstance(i, If)]
    assert len(ifs) == 1
    assert ifs[0].cond.reg is Reg.Y
    assert ifs[0].cond.op == ">=0"
    assert ifs[0].cond.rhs is None


def test_tax_then_bmi_fuses_to_sign_test_on_x():
    """`tax ; bmi L` reads N of the dst register (X), so the fusion
    must use `x < 0` (not `a < 0`)."""
    src = SourceRef(file="syn", line=0, raw="")
    r = Routine(
        name="f",
        body=[
            Transfer(src_reg=Reg.A, dst_reg=Reg.X, src=src),
            Branch(cond="mi", target="]rts", src=src),
            Return(src=src),
        ],
    )
    out = structure_routine(r)
    ifs = [i for i in out.body if isinstance(i, If)]
    assert len(ifs) == 1
    assert ifs[0].cond.reg is Reg.X
    assert ifs[0].cond.op == "<0"


def test_memory_inc_then_branch_does_not_fuse():
    """`inc addr ; bne :loop` — the test reads Z of a memory cell,
    not a register. The structurer has no Compare-on-memory form, so
    the branch stays unfused."""
    src = SourceRef(file="syn", line=0, raw="")
    r = Routine(
        name="f",
        body=[
            IncTarget(target=Abs(name="ctr", addr=0x80), src=src),
            Branch(cond="ne", target=":loop", src=src),
            Return(src=src),
        ],
    )
    out = structure_routine(r)
    assert any(isinstance(i, Branch) for i in out.body)
    assert not any(isinstance(i, If) for i in out.body)
