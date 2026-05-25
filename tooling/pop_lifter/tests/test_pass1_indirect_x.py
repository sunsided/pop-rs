"""Pass-1 pre-indexed indirect (`(zp,x)`) tests: parser + lifter +
interpreter coverage for `lda (PAC,x)`, the form POP's UNPACK blitter
uses. Distinct from `(zp),y`: X indexes the pointer *location*, not the
fetched pointer."""

from __future__ import annotations

from pop_lifter.interp_ir1 import Trace, exec_atom
from pop_lifter.ir1 import (
    Abs,
    IndirectX,
    LoadIndirect,
    Reg,
    SourceRef,
)
from pop_lifter.pass0_lex import Line
from pop_lifter.pass1_lift import _lift_instr, _parse_indirect_x


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


# ---- parser


def test_parse_indirect_x_resolves_via_equate():
    got = _parse_indirect_x("(PAC,x)", {"PAC": 0x00})
    assert got is not None
    assert got.ptr.name == "PAC" and got.ptr.addr == 0x00


def test_parse_indirect_x_accepts_uppercase_and_whitespace():
    assert _parse_indirect_x("(ptr,X)", {"ptr": 0x80}) is not None
    assert _parse_indirect_x("(ptr, x)", {"ptr": 0x80}) is not None


def test_parse_indirect_x_rejects_other_forms():
    # Post-indexed `(ptr),y` and indexed-absolute `name,x` are not this form.
    assert _parse_indirect_x("(ptr),y", {"ptr": 0x80}) is None
    assert _parse_indirect_x("name,x", {"name": 0x80}) is None


# ---- lifter dispatch


def test_lda_indirect_x_lifts_to_loadindirect():
    instr = _lift_instr(_line("lda", "(PAC,x)"), {"PAC": 0x00}, set())
    assert isinstance(instr, LoadIndirect)
    assert instr.reg is Reg.A
    assert isinstance(instr.source, IndirectX)
    assert instr.source.ptr.addr == 0x00


# ---- interpreter


def test_indirect_x_reads_through_pre_indexed_pointer():
    """X indexes the pointer location: with PAC=$00 and X=2, the pointer
    is read from mem[2]/mem[3]; if it holds $0305, `lda (PAC,x)` loads
    mem[$0305]."""
    t = Trace(ram=bytearray(0x10000), a=0, x=2, y=0)
    t.ram[0x02] = 0x05   # pointer low byte at PAC+X
    t.ram[0x03] = 0x03   # pointer high byte
    t.ram[0x0305] = 0x42
    src = SourceRef(file="syn", line=1, raw="lda (PAC,x)")
    exec_atom(LoadIndirect(reg=Reg.A, source=IndirectX(ptr=Abs(name="PAC", addr=0x00)), src=src), t, t.ram)
    assert t.a == 0x42


def test_indirect_x_wraps_in_zero_page():
    """The pointer location wraps within the zero page (PAC=$ff, X=1 →
    location $00)."""
    t = Trace(ram=bytearray(0x10000), a=0, x=1, y=0)
    t.ram[0x00] = 0x10   # location ($ff + 1) & 0xff == 0
    t.ram[0x01] = 0x20
    t.ram[0x2010] = 0x7e
    src = SourceRef(file="syn", line=1, raw="lda ($ff,x)")
    exec_atom(LoadIndirect(reg=Reg.A, source=IndirectX(ptr=Abs(name="p", addr=0xff)), src=src), t, t.ram)
    assert t.a == 0x7e
