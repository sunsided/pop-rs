"""Pass 3 — 16-bit expression recovery.

Two flavours of test:

* **Structural unit tests** pin recognition: the seven-instruction
  6502 idiom (`lda lo ; clc/sec ; adc/sbc lo_op ; sta lo_dst ;
  lda hi ; adc/sbc hi_op ; sta hi_dst`) becomes a `Wide16Stmt` after
  folding, while sequences that differ by one step (wrong carry set-up,
  intervening non-RawStmt, etc.) are left unchanged.
* **Behavioural equivalence** interprets the routine before and after
  folding and asserts byte-identical RAM — the structural rewrite must
  not change observable behaviour (add and subtract, with and without
  carry propagation).
"""

from __future__ import annotations

from pop_lifter.interp_ir3 import run as ir3_run
from pop_lifter.ir1 import (
    Abs,
    AdcAbs,
    AdcImm,
    Clc,
    Compare,
    Imm,
    LoadAbs,
    LoadImm,
    LoadIndexed,
    Reg,
    SbcAbs,
    SbcImm,
    Sec,
    SourceRef,
    StoreAbs,
)
from pop_lifter.ir3 import (
    Assign,
    Block,
    IfStmt,
    ModuleIR3,
    RawStmt,
    ReturnStmt,
    RoutineIR3,
    Wide16Stmt,
)
from pop_lifter.pass3_expr import fold_module, fold_routine, fold_stats, wide16_stats

SRC = SourceRef(file="syn", line=0, raw="")


# ---------------------------------------------------------------- helpers


def _raw(item) -> RawStmt:
    return RawStmt(item=item)


def _imm(v: int) -> Imm:
    return Imm(value=v, text=f"#{v:#04x}")


def _abs(name: str, addr: int) -> Abs:
    return Abs(name=name, addr=addr)


def _lda(name: str, addr: int) -> RawStmt:
    return _raw(LoadAbs(reg=Reg.A, source=_abs(name, addr), src=SRC))


def _sta(name: str, addr: int) -> RawStmt:
    return _raw(StoreAbs(reg=Reg.A, target=_abs(name, addr), src=SRC))


def _clc() -> RawStmt:
    return _raw(Clc(src=SRC))


def _sec() -> RawStmt:
    return _raw(Sec(src=SRC))


def _adc_imm(v: int) -> RawStmt:
    return _raw(AdcImm(imm=_imm(v), src=SRC))


def _adc_abs(name: str, addr: int) -> RawStmt:
    return _raw(AdcAbs(source=_abs(name, addr), src=SRC))


def _sbc_imm(v: int) -> RawStmt:
    return _raw(SbcImm(imm=_imm(v), src=SRC))


def _sbc_abs(name: str, addr: int) -> RawStmt:
    return _raw(SbcAbs(source=_abs(name, addr), src=SRC))


def _kill_a() -> RawStmt:
    return _raw(LoadImm(reg=Reg.A, imm=_imm(0), src=SRC))


def _ldy_imm(v: int) -> RawStmt:
    return _raw(LoadImm(reg=Reg.Y, imm=_imm(v), src=SRC))


def _fold(stmts: list) -> list:
    routine = RoutineIR3(name="syn", body=Block.of(stmts))
    return list(fold_routine(routine).body.stmts)


def _wide16_add_seq(
    lo_name: str, lo_addr: int,
    hi_name: str, hi_addr: int,
    lo_op_imm: int,
    hi_op_imm: int,
) -> list[RawStmt]:
    """7-instruction 16-bit add: `{hi:lo} += {hi_op:lo_op}`."""
    return [
        _lda(lo_name, lo_addr),
        _clc(),
        _adc_imm(lo_op_imm),
        _sta(lo_name, lo_addr),
        _lda(hi_name, hi_addr),
        _adc_imm(hi_op_imm),
        _sta(hi_name, hi_addr),
    ]


def _wide16_sub_seq(
    lo_name: str, lo_addr: int,
    hi_name: str, hi_addr: int,
    lo_op_imm: int,
    hi_op_imm: int,
) -> list[RawStmt]:
    """7-instruction 16-bit sub: `{hi:lo} -= {hi_op:lo_op}`."""
    return [
        _lda(lo_name, lo_addr),
        _sec(),
        _sbc_imm(lo_op_imm),
        _sta(lo_name, lo_addr),
        _lda(hi_name, hi_addr),
        _sbc_imm(hi_op_imm),
        _sta(hi_name, hi_addr),
    ]


# ---------------------------------------------------------------- structural tests


def test_wide16_add_recognised():
    """7-instruction add idiom collapses to a single `Wide16Stmt`."""
    stmts = _wide16_add_seq("lo", 0x10, "hi", 0x11, 0x01, 0x00)
    out = _fold(stmts)
    assert len(out) == 1
    w = out[0]
    assert isinstance(w, Wide16Stmt)
    assert w.op == "+"
    assert isinstance(w.lo_src, Abs) and w.lo_src.addr == 0x10
    assert isinstance(w.lo_op, Imm) and w.lo_op.value == 0x01
    assert isinstance(w.lo_dst, Abs) and w.lo_dst.addr == 0x10
    assert isinstance(w.hi_src, Abs) and w.hi_src.addr == 0x11
    assert isinstance(w.hi_op, Imm) and w.hi_op.value == 0x00
    assert isinstance(w.hi_dst, Abs) and w.hi_dst.addr == 0x11


def test_wide16_sub_recognised():
    """7-instruction subtract idiom collapses to `Wide16Stmt` with op='-'."""
    stmts = _wide16_sub_seq("lo", 0x10, "hi", 0x11, 0x05, 0x00)
    out = _fold(stmts)
    assert len(out) == 1
    w = out[0]
    assert isinstance(w, Wide16Stmt)
    assert w.op == "-"
    assert isinstance(w.lo_op, Imm) and w.lo_op.value == 0x05
    assert isinstance(w.hi_op, Imm) and w.hi_op.value == 0x00


def test_wide16_nonzero_hi_op():
    """Non-zero high byte operand is correctly captured."""
    stmts = [
        _lda("lo", 0x10),
        _clc(),
        _adc_abs("op_lo", 0x20),
        _sta("lo", 0x10),
        _lda("hi", 0x11),
        _adc_abs("op_hi", 0x21),
        _sta("hi", 0x11),
    ]
    out = _fold(stmts)
    assert len(out) == 1
    w = out[0]
    assert isinstance(w, Wide16Stmt)
    assert isinstance(w.lo_op, Abs) and w.lo_op.addr == 0x20
    assert isinstance(w.hi_op, Abs) and w.hi_op.addr == 0x21


def test_wide16_different_lo_dst():
    """lo_src ≠ lo_dst (non-write-back pattern) is still recognised."""
    stmts = [
        _lda("src", 0x10),
        _clc(),
        _adc_imm(8),
        _sta("dst", 0x20),
        _lda("src_hi", 0x11),
        _adc_imm(0),
        _sta("dst_hi", 0x21),
    ]
    out = _fold(stmts)
    assert len(out) == 1
    w = out[0]
    assert isinstance(w, Wide16Stmt)
    assert isinstance(w.lo_src, Abs) and w.lo_src.addr == 0x10
    assert isinstance(w.lo_dst, Abs) and w.lo_dst.addr == 0x20


def test_wide16_not_matched_when_clc_missing():
    """Bare adc + bare adc (no clc) doesn't match — the first adc
    depends on an unknown incoming carry and the pattern requires clc/sec
    at position i+1."""
    stmts = [
        _lda("lo", 0x10),
        _adc_imm(1),         # no clc at i+1 — _wide16_at checks for Clc/Sec
        _sta("lo", 0x10),
        _lda("hi", 0x11),
        _adc_imm(0),
        _sta("hi", 0x11),
        _kill_a(),           # ensure A dead for any potential 8-bit fold
    ]
    out = _fold(stmts)
    assert not any(isinstance(s, Wide16Stmt) for s in out)


def test_wide16_not_matched_when_clc_between_hi():
    """An explicit clc between the lo-store and hi-load resets carry —
    that's two independent 8-bit ops, not a 16-bit pair.  The extra clc
    at position i+4 of the 7-window causes `_is_reg_load` to return None
    (Clc isn't a load), so the idiom is rejected."""
    stmts = [
        _lda("lo", 0x10),
        _clc(),
        _adc_imm(1),
        _sta("lo", 0x10),
        _clc(),              # extra clc resets carry — NOT a bare carry-in adc
        _lda("hi", 0x11),
        _adc_imm(0),
        _sta("hi", 0x11),
    ]
    out = _fold(stmts)
    assert not any(isinstance(s, Wide16Stmt) for s in out)


def test_wide16_not_matched_when_non_rawstmt_inside():
    """An intervening non-RawStmt anywhere in the 7-window prevents
    recognition — `_wide16_at` requires all seven slots to be `RawStmt`."""
    guard = IfStmt(
        cond=Compare(reg=Reg.A, op="==", rhs=_imm(0)),
        then_block=Block.of([]),
        else_block=None,
        src=SRC,
    )
    stmts = [
        _lda("lo", 0x10),
        _clc(),
        _adc_imm(1),
        guard,              # non-RawStmt at i+3 — RawStmt check fails
        _lda("hi", 0x11),
        _adc_imm(0),
        _sta("hi", 0x11),
    ]
    out = _fold(stmts)
    assert not any(isinstance(s, Wide16Stmt) for s in out)


def test_wide16_not_matched_when_hi_adc_type_differs():
    """add-then-subtract mix (`clc + adc lo`, then `sbc hi`) is not a valid
    16-bit pair — `_wide16_at` requires both ops to be the same type."""
    stmts = [
        _lda("lo", 0x10),
        _clc(),
        _adc_imm(1),
        _sta("lo", 0x10),
        _lda("hi", 0x11),
        _sbc_imm(0),         # sbc instead of adc — type mismatch
        _sta("hi", 0x11),
    ]
    out = _fold(stmts)
    assert not any(isinstance(s, Wide16Stmt) for s in out)


def test_wide16_followed_by_other_folds():
    """Wide16 and a subsequent 8-bit copy in the same block both fire."""
    copy_stmts = [
        _lda("src", 0x30),
        _sta("dst", 0x40),
        _kill_a(),
    ]
    stmts = _wide16_add_seq("lo", 0x10, "hi", 0x11, 1, 0) + copy_stmts
    out = _fold(stmts)
    assert isinstance(out[0], Wide16Stmt)
    assert isinstance(out[1], Assign)


def test_wide16_stats():
    """wide16_stats counts Wide16Stmt nodes across the module."""
    seq = _wide16_add_seq("lo", 0x10, "hi", 0x11, 1, 0)
    module = ModuleIR3(
        name="test", file="syn",
        routines=[RoutineIR3(name="r", body=Block.of(seq))],
    )
    folded = fold_module(module)
    assert wide16_stats(folded) == 1
    assert fold_stats(folded) == 0  # no Assign nodes — only Wide16Stmt


# ---------------------------------------------------------------- behavioural tests


def _module(stmts: list) -> ModuleIR3:
    return ModuleIR3(
        name="test", file="syn",
        routines=[RoutineIR3(
            name="test",
            body=Block.of(stmts + [ReturnStmt(src=SRC)]),
        )],
    )


def _run_both(stmts: list, init: dict[int, int]):
    """Run the routine before and after folding; return both traces."""
    ram0 = bytearray(0x10000)
    for addr, val in init.items():
        ram0[addr] = val
    before = _module(stmts)
    after = fold_module(before)
    t_before = ir3_run([before], "test", ram=bytearray(ram0))
    t_after = ir3_run([after], "test", ram=bytearray(ram0))
    return t_before, t_after


def test_behaviour_add_no_carry():
    """16-bit add, no carry out of lo byte: {hi:lo} += {0:1}."""
    stmts = _wide16_add_seq("lo", 0x10, "hi", 0x11, 1, 0)
    tb, ta = _run_both(stmts, {0x10: 0x05, 0x11: 0x02})
    assert tb.ram[0x10] == ta.ram[0x10] == 0x06
    assert tb.ram[0x11] == ta.ram[0x11] == 0x02
    assert tb.a == ta.a
    assert tb.c == ta.c
    assert tb.z == ta.z
    assert tb.n == ta.n


def test_behaviour_add_with_carry():
    """16-bit add, carry propagates from lo to hi byte."""
    stmts = _wide16_add_seq("lo", 0x10, "hi", 0x11, 0xff, 0)
    tb, ta = _run_both(stmts, {0x10: 0x02, 0x11: 0x01})
    # lo: 0x02 + 0xff = 0x101 → lo_result=0x01, carry=1
    # hi: 0x01 + 0x00 + 1 = 0x02
    assert tb.ram[0x10] == ta.ram[0x10] == 0x01
    assert tb.ram[0x11] == ta.ram[0x11] == 0x02
    assert tb.a == ta.a
    assert tb.c == ta.c


def test_behaviour_sub_no_borrow():
    """16-bit subtract, no borrow from lo byte: {hi:lo} -= {0:2}."""
    stmts = _wide16_sub_seq("lo", 0x10, "hi", 0x11, 2, 0)
    tb, ta = _run_both(stmts, {0x10: 0x05, 0x11: 0x03})
    # lo: 0x05 - 0x02 = 0x03, no borrow (carry set)
    # hi: 0x03 - 0x00 - (1-1) = 0x03
    assert tb.ram[0x10] == ta.ram[0x10] == 0x03
    assert tb.ram[0x11] == ta.ram[0x11] == 0x03
    assert tb.a == ta.a
    assert tb.c == ta.c


def test_behaviour_sub_with_borrow():
    """16-bit subtract, borrow propagates from lo to hi byte."""
    stmts = _wide16_sub_seq("lo", 0x10, "hi", 0x11, 0x10, 0)
    tb, ta = _run_both(stmts, {0x10: 0x05, 0x11: 0x03})
    # lo: 0x05 - 0x10 = -0x0b → lo_result=0xf5, borrow (C=0)
    # hi: 0x03 - 0x00 - (1-0) = 0x02
    assert tb.ram[0x10] == ta.ram[0x10] == 0xf5
    assert tb.ram[0x11] == ta.ram[0x11] == 0x02
    assert tb.a == ta.a
    assert tb.c == ta.c


def test_behaviour_add_full_word():
    """16-bit add with non-zero hi operand: {hi:lo} += {op_hi:op_lo}."""
    stmts = [
        _lda("lo", 0x10),
        _clc(),
        _adc_abs("op_lo", 0x20),
        _sta("lo", 0x10),
        _lda("hi", 0x11),
        _adc_abs("op_hi", 0x21),
        _sta("hi", 0x11),
    ]
    tb, ta = _run_both(stmts, {0x10: 0x80, 0x11: 0x01, 0x20: 0x90, 0x21: 0x02})
    # lo: 0x80 + 0x90 = 0x110 → lo=0x10, carry=1
    # hi: 0x01 + 0x02 + 1 = 0x04
    assert tb.ram[0x10] == ta.ram[0x10] == 0x10
    assert tb.ram[0x11] == ta.ram[0x11] == 0x04
    assert tb.a == ta.a
    assert tb.c == ta.c


def test_behaviour_lo_writeback_aliases_hi_src():
    """lo_dst == lo_src (write-back), hi_src ≠ lo_dst — read ordering
    is: compute lo, write lo_dst, then read hi_src (adjacent address).
    Behavioural equivalence verifies the ordering is correct."""
    stmts = [
        _lda("fcharx", 0x71),
        _clc(),
        _adc_abs("ztemp", 0xe9),
        _sta("fcharx", 0x71),
        _lda("fcharx_hi", 0x72),
        _adc_imm(0),
        _sta("fcharx_hi", 0x72),
    ]
    tb, ta = _run_both(stmts, {0x71: 0x40, 0x72: 0x02, 0xe9: 0x03})
    # lo: 0x40 + 0x03 = 0x43, no carry
    # hi: 0x02 + 0x00 + 0 = 0x02
    assert tb.ram[0x71] == ta.ram[0x71] == 0x43
    assert tb.ram[0x72] == ta.ram[0x72] == 0x02
    assert tb.a == ta.a
    assert tb.c == ta.c


def test_behaviour_indexed_source():
    """lo_src and hi_src are Y-indexed (simulating the HIRES address-
    computation pattern); recognised as a Wide16Stmt and executed
    correctly regardless of Y."""
    # Prepend `ldy #3` so Y=3 for both before and after runs.
    stmts = [
        _ldy_imm(3),
        _raw(LoadIndexed(reg=Reg.A, base=_abs("YLO", 0x0100),
                         index=Reg.Y, src=SRC)),
        _clc(),
        _adc_abs("XCO", 0x01),
        _sta("BASE", 0xf0),
        _raw(LoadIndexed(reg=Reg.A, base=_abs("YHI", 0x0200),
                         index=Reg.Y, src=SRC)),
        _adc_abs("PAGE", 0x02),
        _sta("BASE_hi", 0xf1),
    ]
    # YLO[3]=0x28, XCO=0x14 → BASE=0x3c, no carry
    # YHI[3]=0x20, PAGE=0x40 → BASE_hi=0x60
    init = {0x0103: 0x28, 0x01: 0x14, 0x0203: 0x20, 0x02: 0x40}
    tb, ta = _run_both(stmts, init)
    assert tb.ram[0xf0] == ta.ram[0xf0]
    assert tb.ram[0xf1] == ta.ram[0xf1]
    assert tb.a == ta.a
    assert tb.c == ta.c
