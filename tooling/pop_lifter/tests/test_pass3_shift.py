"""Pass 3 — bit-shift expression recovery.

`lda X ; (asl)*n ; sta Y` and `lda X ; (lsr)*n ; sta Y` are folded into
a single `Assign` with a `BinExpr` when A and carry are both dead after
the store.  The shift count `n` becomes the `rhs` of the `BinExpr`.
"""

from __future__ import annotations

from pop_lifter.interp_ir3 import run as ir3_run
from pop_lifter.ir1 import (
    Abs,
    Asl,
    Imm,
    LoadAbs,
    LoadImm,
    Lsr,
    Reg,
    SourceRef,
    StoreAbs,
)
from pop_lifter.ir3 import (
    Assign,
    BinExpr,
    Block,
    ModuleIR3,
    RawStmt,
    ReturnStmt,
    RoutineIR3,
)
from pop_lifter.pass3_expr import fold_module, fold_routine

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


def _asl() -> RawStmt:
    return _raw(Asl(src=SRC))


def _lsr() -> RawStmt:
    return _raw(Lsr(src=SRC))


def _kill_a() -> RawStmt:
    return _raw(LoadImm(reg=Reg.A, imm=_imm(0), src=SRC))


def _kill_carry() -> RawStmt:
    from pop_lifter.ir1 import Clc
    return _raw(Clc(src=SRC))


def _fold(stmts: list) -> list:
    routine = RoutineIR3(name="syn", body=Block.of(stmts))
    return list(fold_routine(routine).body.stmts)


# ---------------------------------------------------------------- structural tests


def test_single_asl_folded():
    """lda X ; asl ; sta Y (A and carry dead) → Y = X << 1."""
    stmts = [_lda("X", 0x10), _asl(), _sta("Y", 0x20), _kill_a(), _kill_carry()]
    out = _fold(stmts)
    assert isinstance(out[0], Assign)
    src = out[0].source
    assert isinstance(src, BinExpr) and src.op == "<<"
    assert isinstance(src.lhs, Abs) and src.lhs.addr == 0x10
    assert isinstance(src.rhs, Imm) and src.rhs.value == 1
    assert isinstance(out[0].target, Abs) and out[0].target.addr == 0x20


def test_single_lsr_folded():
    """lda X ; lsr ; sta Y → Y = X >> 1."""
    stmts = [_lda("X", 0x10), _lsr(), _sta("Y", 0x20), _kill_a(), _kill_carry()]
    out = _fold(stmts)
    assert isinstance(out[0], Assign)
    src = out[0].source
    assert isinstance(src, BinExpr) and src.op == ">>"
    assert isinstance(src.rhs, Imm) and src.rhs.value == 1


def test_double_asl_folded():
    """lda X ; asl ; asl ; sta Y → Y = X << 2 (run of two asl)."""
    stmts = [_lda("X", 0x10), _asl(), _asl(), _sta("Y", 0x20), _kill_a(), _kill_carry()]
    out = _fold(stmts)
    assert isinstance(out[0], Assign)
    src = out[0].source
    assert isinstance(src, BinExpr) and src.op == "<<"
    assert isinstance(src.rhs, Imm) and src.rhs.value == 2


def test_triple_lsr_folded():
    """Three consecutive lsr collapse to >> 3."""
    stmts = [
        _lda("X", 0x10), _lsr(), _lsr(), _lsr(),
        _sta("Y", 0x20), _kill_a(), _kill_carry(),
    ]
    out = _fold(stmts)
    assert isinstance(out[0], Assign)
    assert isinstance(out[0].source, BinExpr)
    assert out[0].source.rhs.value == 3


def test_shift_not_folded_when_a_live():
    """If A is not dead after the store, the fold must not fire."""
    stmts = [
        _lda("X", 0x10),
        _asl(),
        _sta("Y", 0x20),
        # A is read here via sta Z (no kill_a)
        _sta("Z", 0x30),
    ]
    out = _fold(stmts)
    assert not any(isinstance(s, Assign) and isinstance(s.source, BinExpr) for s in out)


def test_shift_not_folded_when_carry_live():
    """If carry is live after the store, the fold must not fire."""
    from pop_lifter.ir1 import AdcImm
    stmts = [
        _lda("X", 0x10),
        _asl(),
        _sta("Y", 0x20),
        _kill_a(),
        # carry is consumed here (bare adc reads carry)
        _raw(AdcImm(imm=_imm(0), src=SRC)),
    ]
    out = _fold(stmts)
    assert not any(isinstance(s, Assign) and isinstance(s.source, BinExpr)
                   and s.source.op == "<<" for s in out)


def test_mixed_asl_lsr_not_merged():
    """asl followed by lsr — different directions, fold takes only the asl run."""
    stmts = [
        _lda("X", 0x10),
        _asl(),
        _lsr(),            # direction change — asl run ends here
        _sta("Y", 0x20),
        _kill_a(),
        _kill_carry(),
    ]
    out = _fold(stmts)
    # The run is length 1 (just one asl), then lsr is a separate RawStmt.
    # The fold may or may not fire depending on dead-after for the 1-asl case,
    # but the key invariant: no BinExpr with count > 1 may appear.
    for s in out:
        if isinstance(s, Assign) and isinstance(s.source, BinExpr):
            assert s.source.rhs.value == 1


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
    ram0 = bytearray(0x10000)
    for addr, val in init.items():
        ram0[addr] = val
    before = _module(stmts)
    after = fold_module(before)
    t_before = ir3_run([before], "test", ram=bytearray(ram0))
    t_after = ir3_run([after], "test", ram=bytearray(ram0))
    return t_before, t_after


def test_behaviour_asl_x1():
    """Single asl: result = X * 2 (mod 256)."""
    stmts = [_lda("X", 0x10), _asl(), _sta("Y", 0x20), _kill_a(), _kill_carry()]
    tb, ta = _run_both(stmts, {0x10: 0x15})
    assert tb.ram[0x20] == ta.ram[0x20] == 0x2a


def test_behaviour_asl_x2():
    """Double asl: result = X * 4 (mod 256), carry from final shift."""
    stmts = [_lda("X", 0x10), _asl(), _asl(), _sta("Y", 0x20), _kill_a(), _kill_carry()]
    tb, ta = _run_both(stmts, {0x10: 0x41})
    # 0x41 << 2 = 0x104 → 0x04 (mod 256)
    assert tb.ram[0x20] == ta.ram[0x20]


def test_behaviour_lsr_x1():
    """Single lsr: result = X // 2 (unsigned)."""
    stmts = [_lda("X", 0x10), _lsr(), _sta("Y", 0x20), _kill_a(), _kill_carry()]
    tb, ta = _run_both(stmts, {0x10: 0xfe})
    assert tb.ram[0x20] == ta.ram[0x20] == 0x7f


def test_behaviour_lsr_odd():
    """lsr of odd value: low bit goes to carry (which is dead), result = (X-1)//2."""
    stmts = [_lda("X", 0x10), _lsr(), _sta("Y", 0x20), _kill_a(), _kill_carry()]
    tb, ta = _run_both(stmts, {0x10: 0x07})
    assert tb.ram[0x20] == ta.ram[0x20] == 0x03
