"""Pass-1 indirect-indexed (`(zp),y`) tests: parser + lifter +
interpreter coverage for `lda`/`sta`/`cmp`/`and`/`ora`/`eor` against
the post-indexed indirect addressing mode."""

from __future__ import annotations

import pytest

from pop_lifter.interp_ir1 import InterpError, Trace, exec_atom
from pop_lifter.ir1 import (
    Abs,
    Bitwise,
    CmpIndirect,
    Imm,
    IndirectY,
    LoadIndirect,
    Reg,
    SourceRef,
    StoreIndirect,
)
from pop_lifter.pass0_lex import Line
from pop_lifter.pass1_lift import _lift_instr, _parse_indirect_y


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


def _trace(a=0, x=0, y=0) -> Trace:
    return Trace(ram=bytearray(0x10000), a=a, x=x, y=y)


# ---- parser


def test_parse_indirect_y_resolves_via_equate():
    """`(ztemp),y` → `IndirectY(ptr=Abs(name='ztemp', addr=0xf0))`."""
    equates = {"ztemp": 0xf0}
    got = _parse_indirect_y("(ztemp),y", equates)
    assert got is not None
    assert got.ptr.addr == 0xf0
    assert got.ptr.name == "ztemp"


def test_parse_indirect_y_accepts_uppercase_y():
    """Merlin source uses both `,y` and `,Y`. Both must parse."""
    got = _parse_indirect_y("(ptr),Y", {"ptr": 0x80})
    assert got is not None and got.ptr.addr == 0x80


def test_parse_indirect_y_accepts_whitespace_after_comma():
    """`(ptr), y` (whitespace before `y`) — some hand-written
    assembly uses it; the parser must accept it."""
    got = _parse_indirect_y("(ptr), y", {"ptr": 0x80})
    assert got is not None


def test_parse_indirect_y_rejects_non_indirect_form():
    """Plain `name,y` (indexed-absolute) and `(name,x)` (pre-indexed)
    must NOT match — they belong to different lifter paths."""
    assert _parse_indirect_y("name,y", {"name": 0x80}) is None
    assert _parse_indirect_y("(ptr,x)", {"ptr": 0x80}) is None


# ---- lifter dispatch


def test_lda_indirect_y_lifts_to_loadindirect():
    instr = _lift_instr(_line("lda", "(ztemp),y"), {"ztemp": 0xf0}, set())
    assert isinstance(instr, LoadIndirect)
    assert instr.reg is Reg.A
    assert instr.source.ptr.addr == 0xf0


def test_sta_indirect_y_lifts_to_storeindirect():
    instr = _lift_instr(_line("sta", "(dest),y"), {"dest": 0x82}, set())
    assert isinstance(instr, StoreIndirect)
    assert instr.target.ptr.addr == 0x82


def test_cmp_indirect_y_lifts_to_cmpindirect():
    instr = _lift_instr(_line("cmp", "(src),y"), {"src": 0x84}, set())
    assert isinstance(instr, CmpIndirect)
    assert instr.source.ptr.addr == 0x84


def test_and_indirect_y_lifts_to_bitwise_indirect():
    instr = _lift_instr(_line("and", "(mask),y"), {"mask": 0xfa}, set())
    assert isinstance(instr, Bitwise)
    assert instr.op == "and"
    assert isinstance(instr.source, IndirectY)
    assert instr.source.ptr.addr == 0xfa


def test_ldx_indirect_y_stays_unsupported():
    """The 6502 has no `ldx (ptr),y` — it doesn't exist as an
    addressing mode. The lifter must NOT silently fall back to an
    absolute parse; it must emit Unsupported."""
    from pop_lifter.ir1 import Unsupported
    instr = _lift_instr(_line("ldx", "(ptr),y"), {"ptr": 0x80}, set())
    assert isinstance(instr, Unsupported)


# ---- interpreter semantics


def test_load_indirect_dereferences_pointer_plus_y():
    """`(ptr),y` reads the 16-bit pointer at ptr.addr (little-endian)
    and adds Y to get the effective address."""
    src = SourceRef(file="syn", line=0, raw="")
    t = _trace(y=3)
    # Pointer at 0xf0/0xf1 → $1000.
    t.ram[0xf0] = 0x00
    t.ram[0xf1] = 0x10
    t.ram[0x1003] = 0x42
    exec_atom(
        LoadIndirect(reg=Reg.A, source=IndirectY(ptr=Abs(name="p", addr=0xf0)), src=src),
        t, t.ram,
    )
    assert t.a == 0x42
    assert t.z == 0
    assert t.n == 0


def test_load_indirect_sets_zn_on_loaded_value():
    """`lda (ptr),y` reads a byte and sets Z/N like any other load."""
    src = SourceRef(file="syn", line=0, raw="")
    t = _trace(y=0)
    t.ram[0xf0] = 0x00
    t.ram[0xf1] = 0x20
    t.ram[0x2000] = 0x80   # negative byte
    exec_atom(
        LoadIndirect(reg=Reg.A, source=IndirectY(ptr=Abs(name="p", addr=0xf0)), src=src),
        t, t.ram,
    )
    assert t.a == 0x80
    assert t.n == 1
    assert t.z == 0


def test_store_indirect_writes_through_pointer_plus_y():
    src = SourceRef(file="syn", line=0, raw="")
    t = _trace(a=0xab, y=5)
    t.ram[0xf2] = 0x40
    t.ram[0xf3] = 0x30
    exec_atom(
        StoreIndirect(reg=Reg.A, target=IndirectY(ptr=Abs(name="p", addr=0xf2)), src=src),
        t, t.ram,
    )
    assert t.ram[0x3045] == 0xab
    # The write must be recorded for diff tracking.
    assert t.writes[0x3045] == 0xab


def test_cmp_indirect_sets_carry_and_zero_correctly():
    src = SourceRef(file="syn", line=0, raw="")
    t = _trace(a=0x42, y=0)
    t.ram[0x80] = 0x00
    t.ram[0x81] = 0x40
    t.ram[0x4000] = 0x42
    exec_atom(
        CmpIndirect(reg=Reg.A, source=IndirectY(ptr=Abs(name="p", addr=0x80)), src=src),
        t, t.ram,
    )
    assert t.z == 1      # A == operand
    assert t.c == 1      # A >= operand


def test_indirect_y_wraps_effective_address_at_64k():
    """If the pointer + Y overflows 16 bits, the effective address
    must wrap modulo 64K — the 6502 has a 16-bit address bus."""
    src = SourceRef(file="syn", line=0, raw="")
    t = _trace(y=0x10)
    t.ram[0x80] = 0xf0   # pointer (lo, hi) = $fff0
    t.ram[0x81] = 0xff
    t.ram[0x0000] = 0x99   # $fff0 + $10 = $10000 → wraps to $0000
    exec_atom(
        LoadIndirect(reg=Reg.A, source=IndirectY(ptr=Abs(name="p", addr=0x80)), src=src),
        t, t.ram,
    )
    assert t.a == 0x99


def test_bitwise_and_indirect_y_combines_with_pointer_byte():
    """`and (ptr),y` AND's A with the byte at the post-indexed
    address. Used in POP's masked-sprite blits."""
    src = SourceRef(file="syn", line=0, raw="")
    t = _trace(a=0xf0, y=2)
    t.ram[0x84] = 0x00
    t.ram[0x85] = 0x50
    t.ram[0x5002] = 0x3c
    exec_atom(
        Bitwise(op="and", source=IndirectY(ptr=Abs(name="p", addr=0x84)), src=src),
        t, t.ram,
    )
    assert t.a == (0xf0 & 0x3c)
