"""Pass-1 indexed-absolute tests: `cmp`/`adc`/`sbc`/`and`/`ora`/
`eor` against `base,x` / `base,y`. Covers the lifter dispatch, the
interpreter semantics (including the synthetic-base gate + 16-bit
wrap), and pass-2 fusion where applicable."""

from __future__ import annotations

import pytest

from pop_lifter.interp_ir1 import InterpError, Trace, exec_atom
from pop_lifter.ir1 import (
    Abs,
    AdcIndexed,
    Bitwise,
    Branch,
    CmpIndexed,
    If,
    Imm,
    IndexedAbs,
    Reg,
    Return,
    Routine,
    SbcIndexed,
    SourceRef,
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


def test_cmp_indexed_x_lifts_to_cmpindexed():
    instr = _lift_instr(_line("cmp", "table,x"), {"table": 0x300}, set())
    assert isinstance(instr, CmpIndexed)
    assert instr.reg is Reg.A
    assert instr.base.addr == 0x300
    assert instr.index is Reg.X


def test_cmp_indexed_y_lifts_to_cmpindexed():
    instr = _lift_instr(_line("cmp", "list,y"), {"list": 0x400}, set())
    assert isinstance(instr, CmpIndexed)
    assert instr.index is Reg.Y


def test_cpx_indexed_stays_unsupported():
    """`cpx` / `cpy` have no indexed-absolute addressing on stock
    6502 — only zero-page indexed (which we don't parse here)."""
    from pop_lifter.ir1 import Unsupported
    instr = _lift_instr(_line("cpx", "tbl,y"), {"tbl": 0x100}, set())
    assert isinstance(instr, Unsupported)


def test_adc_indexed_lifts_to_adcindexed():
    instr = _lift_instr(_line("adc", "BarL,y"), {"BarL": 0x500}, set())
    assert isinstance(instr, AdcIndexed)
    assert instr.base.addr == 0x500
    assert instr.index is Reg.Y


def test_sbc_indexed_lifts_to_sbcindexed():
    instr = _lift_instr(_line("sbc", "BarR,y"), {"BarR": 0x600}, set())
    assert isinstance(instr, SbcIndexed)


def test_and_indexed_lifts_to_bitwise_indexed():
    instr = _lift_instr(_line("and", "mask,x"), {"mask": 0x700}, set())
    assert isinstance(instr, Bitwise)
    assert instr.op == "and"
    assert isinstance(instr.source, IndexedAbs)
    assert instr.source.base.addr == 0x700
    assert instr.source.index is Reg.X


def test_ora_indexed_lifts_to_bitwise_indexed():
    instr = _lift_instr(_line("ora", "list,y"), {"list": 0x800}, set())
    assert isinstance(instr, Bitwise)
    assert instr.op == "or"
    assert isinstance(instr.source, IndexedAbs)


def test_eor_indexed_lifts_to_bitwise_indexed():
    instr = _lift_instr(_line("eor", "key,x"), {"key": 0x900}, set())
    assert isinstance(instr, Bitwise)
    assert instr.op == "eor"
    assert isinstance(instr.source, IndexedAbs)


# ---- interpreter semantics


def test_cmp_indexed_compares_against_indexed_byte():
    """`cmp tbl,x` reads `mem[tbl + x]` and compares against A."""
    src = SourceRef(file="syn", line=0, raw="cmp tbl,x")
    t = _trace(a=0x42, x=3)
    t.ram[0x303] = 0x42
    exec_atom(
        CmpIndexed(reg=Reg.A, base=Abs(name="tbl", addr=0x300), index=Reg.X, src=src),
        t, t.ram,
    )
    assert t.z == 1
    assert t.c == 1


def test_adc_indexed_adds_indexed_byte_to_a():
    """`adc BarL,y` += `mem[BarL + y] + C`."""
    src = SourceRef(file="syn", line=0, raw="adc BarL,y")
    t = _trace(a=0x10, y=2, c=0)
    t.ram[0x502] = 0x05
    exec_atom(
        AdcIndexed(base=Abs(name="BarL", addr=0x500), index=Reg.Y, src=src),
        t, t.ram,
    )
    assert t.a == 0x15
    assert t.c == 0


def test_sbc_indexed_subtracts_indexed_byte_with_borrow():
    """`sbc BarR,y` with C=1 (no incoming borrow) subtracts cleanly."""
    src = SourceRef(file="syn", line=0, raw="sbc BarR,y")
    t = _trace(a=0x10, y=2, c=1)
    t.ram[0x602] = 0x03
    exec_atom(
        SbcIndexed(base=Abs(name="BarR", addr=0x600), index=Reg.Y, src=src),
        t, t.ram,
    )
    assert t.a == 0x0d
    assert t.c == 1


def test_and_indexed_masks_a_against_indexed_byte():
    src = SourceRef(file="syn", line=0, raw="and mask,x")
    t = _trace(a=0xff, x=4)
    t.ram[0x704] = 0x0f
    exec_atom(
        Bitwise(
            op="and",
            source=IndexedAbs(base=Abs(name="mask", addr=0x700), index=Reg.X),
            src=src,
        ),
        t, t.ram,
    )
    assert t.a == 0x0f


def test_indexed_addr_wraps_at_64k():
    """`cmp $fff0,x` with x=$30 wraps to `$0020`. Same wrap fix as
    LoadIndexed/StoreIndexed — pinned to prevent the synthetic-gate
    regression that previously misflagged `$fff0 + $30 = $10020` as
    a synthetic-label dereference."""
    src = SourceRef(file="syn", line=0, raw="cmp $fff0,x")
    t = _trace(a=0xab, x=0x30)
    t.ram[0x0020] = 0xab
    exec_atom(
        CmpIndexed(reg=Reg.A, base=Abs(name="hi", addr=0xfff0), index=Reg.X, src=src),
        t, t.ram,
    )
    assert t.z == 1


def test_indexed_with_synthetic_base_still_raises():
    """Synthetic-label base (≥ 0x10000) must raise — the gate is
    still in effect for the indexed forms."""
    src = SourceRef(file="syn", line=0, raw="cmp Label,x")
    t = _trace(a=0, x=0)
    with pytest.raises(InterpError, match="synthetic-label address"):
        exec_atom(
            CmpIndexed(reg=Reg.A, base=Abs(name="L", addr=0x10042), index=Reg.X, src=src),
            t, t.ram,
        )


# ---- pass-2 fusion


def test_cmp_indexed_then_bne_fuses_to_indexed_compare():
    """`cmp tbl,x ; bne :next` → `if a != *(tbl)[x] goto :next`."""
    src = SourceRef(file="syn", line=0, raw="")
    r = Routine(
        name="f",
        body=[
            CmpIndexed(reg=Reg.A, base=Abs(name="tbl", addr=0x300), index=Reg.X, src=src),
            Branch(cond="ne", target=":next", src=src),
            Return(src=src),
        ],
    )
    out = structure_routine(r)
    ifs = [i for i in out.body if isinstance(i, If)]
    assert len(ifs) == 1
    assert ifs[0].cond.op == "!="
    assert isinstance(ifs[0].cond.rhs, IndexedAbs)
    assert ifs[0].cond.rhs.base.addr == 0x300
    assert ifs[0].cond.rhs.index is Reg.X


def test_cmp_indexed_never_elided_even_when_flags_dead():
    """`CmpIndexed` reads through memory, and an indexed read can hit
    I/O space (Apple II `$c0xx` soft switches and friends). Even
    when Z/N/C are dead at exit, the elision sweep must keep the
    cmp — same conservatism as `Bit(Abs)` from PR #12. Without this
    pin a future "all cmps are pure" optimisation would silently
    delete the read and change program behavior."""
    from pop_lifter.pass2_struct import _eliminate_dead_flags

    src = SourceRef(file="syn", line=0, raw="cmp soft_switch,x")
    r = Routine(
        name="f",
        body=[
            CmpIndexed(
                reg=Reg.A,
                base=Abs(name="soft", addr=0xc030),
                index=Reg.X,
                src=src,
            ),
            # No flag reader downstream — Z/N/C are dead at exit.
            Return(src=src),
        ],
    )
    out = _eliminate_dead_flags(structure_routine(r), flag_demand={})
    assert any(isinstance(i, CmpIndexed) for i in out.body), (
        "CmpIndexed must survive elision even with dead flags — "
        "the indexed read itself may be side-effecting"
    )


def test_adc_indexed_then_branch_does_not_fuse():
    """Adc{Imm,Abs,Indexed} aren't on `_affected_register` — none of
    the adc variants participate in pass-2 fusion (adc reads AND
    writes C, and the existing fusion paths don't handle that
    pair). Pin the non-fusion to prevent silent over-extension."""
    src = SourceRef(file="syn", line=0, raw="")
    r = Routine(
        name="f",
        body=[
            AdcIndexed(base=Abs(name="t", addr=0x500), index=Reg.Y, src=src),
            Branch(cond="eq", target=":L", src=src),
            Return(src=src),
        ],
    )
    out = structure_routine(r)
    assert any(isinstance(i, Branch) for i in out.body)
    assert not any(isinstance(i, If) for i in out.body)
