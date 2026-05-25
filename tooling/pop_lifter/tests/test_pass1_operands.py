"""Operand-parsing slice: local-label loads/compares (`lda :sm`,
`ldx ]temp1`, `cpx :pitch+1`), indexed memory inc/dec (`inc joyX,x`),
and the Merlin `!` (EOR) expression operator (`lda #FinalDisk!1`).

Local (`:`) and Merlin-variable (`]`) names have no resolved address, so
StoreLocal / LoadLocal / CmpLocal share a keyed byte store — the interp's
`code_patches` side channel, the crate's `self.local` map — and a store
then load of the same name round-trips.
"""

from __future__ import annotations

from pop_lifter.interp_ir1 import Trace, exec_atom
from pop_lifter.ir1 import (
    Abs,
    CmpLocal,
    DecTarget,
    IncTarget,
    IndexedAbs,
    LoadLocal,
    Reg,
    SourceRef,
    StoreLocal,
)
from pop_lifter.pass0_parse import eval_expr
from pop_lifter.pass0_lex import Line
from pop_lifter.pass1_lift import _lift_instr
from pop_lifter.pass4_emit_rust import _emit_raw

_SRC = SourceRef(file="syn", line=0, raw="")


def _line(mnemonic: str, operand: str | None = None) -> Line:
    return Line(
        file="syn", lineno=1, raw=f"  {mnemonic} {operand or ''}".rstrip(),
        label=None, mnemonic=mnemonic, operand=operand, comment=None,
    )


def _trace(a=0, x=0, y=0) -> Trace:
    return Trace(ram=bytearray(0x10000), a=a, x=x, y=y)


# ---- Merlin `!` (EOR) operator


def test_bang_operator_is_eor():
    assert eval_expr("1!1", {}) == 0
    assert eval_expr("FinalDisk!1", {"FinalDisk": 1}) == 0
    assert eval_expr("$f0!$0f", {}) == 0xFF
    assert eval_expr("FinalDisk!1", {"FinalDisk": 0}) == 1


# ---- local-label loads / compares: lift dispatch


def test_lda_local_lifts_to_loadlocal():
    instr = _lift_instr(_line("lda", "]cleanflag"), {}, set())
    assert isinstance(instr, LoadLocal)
    assert instr.reg is Reg.A and instr.source_label == "]cleanflag" and instr.offset == 0


def test_ldx_local_with_offset():
    instr = _lift_instr(_line("ldx", ":pitch+1"), {}, set())
    assert isinstance(instr, LoadLocal)
    assert instr.reg is Reg.X and instr.source_label == ":pitch" and instr.offset == 1


def test_cpx_local_lifts_to_cmplocal():
    instr = _lift_instr(_line("cpx", ":pitch+1"), {}, set())
    assert isinstance(instr, CmpLocal)
    assert instr.reg is Reg.X and instr.source_label == ":pitch" and instr.offset == 1


# ---- local store/load round-trip through the interp side channel


def test_store_then_load_local_roundtrips():
    t = _trace(a=0x5A)
    exec_atom(StoreLocal(reg=Reg.A, target_label="]flag", offset=0, src=_SRC), t, t.ram)
    t.a = 0  # clobber
    exec_atom(LoadLocal(reg=Reg.A, source_label="]flag", offset=0, src=_SRC), t, t.ram)
    assert t.a == 0x5A
    assert t.z == 0 and t.n == 0


def test_load_local_unset_reads_zero_and_sets_z():
    t = _trace(a=0xFF)
    exec_atom(LoadLocal(reg=Reg.X, source_label=":never", offset=0, src=_SRC), t, t.ram)
    assert t.x == 0 and t.z == 1


def test_cmp_local_sets_carry_and_zero():
    t = _trace(x=0x10)
    exec_atom(StoreLocal(reg=Reg.X, target_label=":pitch", offset=1, src=_SRC), t, t.ram)
    t.x = 0x20
    exec_atom(CmpLocal(reg=Reg.X, source_label=":pitch", offset=1, src=_SRC), t, t.ram)
    # 0x20 >= 0x10 → C set, not equal → Z clear.
    assert t.c == 1 and t.z == 0


# ---- indexed memory inc/dec


def test_inc_indexed_lifts_and_interprets():
    instr = _lift_instr(_line("inc", "joyX,x"), {"joyX": 0x40}, set())
    assert isinstance(instr, IncTarget) and isinstance(instr.target, IndexedAbs)
    assert instr.target.index is Reg.X
    t = _trace(x=3)
    t.ram[0x43] = 0x09
    exec_atom(instr, t, t.ram)
    assert t.ram[0x43] == 0x0A


def test_dec_indexed_interprets():
    instr = _lift_instr(_line("dec", "tbl,y"), {"tbl": 0x50}, set())
    assert isinstance(instr, DecTarget) and isinstance(instr.target, IndexedAbs)
    t = _trace(y=2)
    t.ram[0x52] = 0x01
    exec_atom(instr, t, t.ram)
    assert t.ram[0x52] == 0x00 and t.z == 1


# ---- pass-4 lowering


def test_local_ops_lower_to_self_local():
    store = _emit_raw(StoreLocal(reg=Reg.A, target_label=":pitch", offset=1, src=_SRC))
    assert store == ['self.local.insert((":pitch", 1), self.reg.a);']
    load = _emit_raw(LoadLocal(reg=Reg.X, source_label="]flag", offset=0, src=_SRC))
    assert load[0] == 'self.reg.x = self.local.get(&("]flag", 0)).copied().unwrap_or(0);'
    cmp = _emit_raw(CmpLocal(reg=Reg.Y, source_label=":pitch", offset=0, src=_SRC))
    assert 'self.local.get(&(":pitch", 0)).copied().unwrap_or(0)' in cmp[0]


def test_inc_indexed_lowers_to_indexed_mem():
    out = _emit_raw(IncTarget(
        target=IndexedAbs(base=Abs(name="joyX", addr=0x40), index=Reg.X), src=_SRC,
    ))
    assert out == [
        "self.mem[0x0040 + self.reg.x as usize] = "
        "self.mem[0x0040 + self.reg.x as usize].wrapping_add(1);"
    ]
