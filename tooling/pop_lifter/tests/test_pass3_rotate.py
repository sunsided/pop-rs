"""Pass 3 — accumulator rotate expression recovery.

`lda X ; (rol)*n ; sta Y` and `lda X ; (ror)*n ; sta Y` are folded into
a single `Assign` with a `RotateExpr` when (a) there is exactly one store
target and (b) A and carry are both dead after the store.

Unlike the shift fold (`BinExpr << / >>`), carry is an *input* to each
rotation step: the `RotateExpr` interpreter reads the current carry from
the trace, feeding it into the first rotate and chaining carry through
successive steps. The final carry-out is not written back — it is dead by
the fold's soundness condition.

Single-store requirement mirrors the arithmetic / shift folds: the stored
value is not the source value, so a multi-store run would not be an
idempotent write-back.

Tests mirror the structure of test_pass3_shift.py:

* **Structural tests** pin the fold detection: simple pairs, multi-count
  runs, blocking conditions (A live, carry live, mixed direction).
* **Behavioural equivalence** interprets before and after folding from
  various initial carry states, asserting identical write sets.
"""

from __future__ import annotations

from pop_lifter.interp_ir3 import run as ir3_run
from pop_lifter.ir1 import (
    Abs,
    AdcImm,
    Clc,
    Imm,
    LoadAbs,
    LoadImm,
    Reg,
    Rol,
    Ror,
    Sec,
    SourceRef,
    StoreAbs,
)
from pop_lifter.ir3 import (
    Assign,
    Block,
    ModuleIR3,
    RawStmt,
    ReturnStmt,
    RotateExpr,
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


def _rol() -> RawStmt:
    return _raw(Rol(src=SRC))


def _ror() -> RawStmt:
    return _raw(Ror(src=SRC))


def _clc() -> RawStmt:
    return _raw(Clc(src=SRC))


def _sec() -> RawStmt:
    return _raw(Sec(src=SRC))


def _kill_a() -> RawStmt:
    return _raw(LoadImm(reg=Reg.A, imm=_imm(0), src=SRC))


def _fold(stmts: list) -> list:
    routine = RoutineIR3(name="syn", body=Block.of(stmts))
    return list(fold_routine(routine).body.stmts)


def _module(stmts: list) -> ModuleIR3:
    return ModuleIR3(
        name="test", file="syn",
        routines=[RoutineIR3(
            name="test",
            body=Block.of(stmts + [ReturnStmt(src=SRC)]),
        )],
    )


def _run_both(stmts: list, init: dict[int, int], *, carry: int = 0):
    """Run `stmts` before and after folding from initial RAM `init`.
    `carry` sets the initial trace carry via a leading `clc` (0) or
    `sec` (1) so differential tests cover both carry-in states."""
    ram0 = bytearray(0x10000)
    for addr, val in init.items():
        ram0[addr] = val
    carry_setup = [_sec() if carry else _clc()]
    before = _module(carry_setup + stmts)
    after = fold_module(before)
    t_before = ir3_run([before], "test", ram=bytearray(ram0))
    t_after = ir3_run([after], "test", ram=bytearray(ram0))
    return t_before, t_after


# ---------------------------------------------------------------- structural tests


def test_single_rol_folded():
    """lda X ; rol ; sta Y (A and carry dead) → Y = rotl(X, 1)."""
    stmts = [_lda("X", 0x10), _rol(), _sta("Y", 0x20), _kill_a(), _clc()]
    out = _fold(stmts)
    assert isinstance(out[0], Assign)
    src = out[0].source
    assert isinstance(src, RotateExpr) and src.op == "rotl"
    assert isinstance(src.operand, Abs) and src.operand.addr == 0x10
    assert src.count == 1
    assert isinstance(out[0].target, Abs) and out[0].target.addr == 0x20


def test_single_ror_folded():
    """lda X ; ror ; sta Y → Y = rotr(X, 1)."""
    stmts = [_lda("X", 0x10), _ror(), _sta("Y", 0x20), _kill_a(), _clc()]
    out = _fold(stmts)
    assert isinstance(out[0], Assign)
    src = out[0].source
    assert isinstance(src, RotateExpr) and src.op == "rotr"
    assert src.count == 1


def test_double_rol_folded():
    """Two consecutive rol collapse to rotl(X, 2)."""
    stmts = [_lda("X", 0x10), _rol(), _rol(), _sta("Y", 0x20), _kill_a(), _clc()]
    out = _fold(stmts)
    assert isinstance(out[0], Assign)
    src = out[0].source
    assert isinstance(src, RotateExpr) and src.op == "rotl"
    assert src.count == 2


def test_triple_ror_folded():
    """Three consecutive ror collapse to rotr(X, 3)."""
    stmts = [
        _lda("X", 0x10), _ror(), _ror(), _ror(),
        _sta("Y", 0x20), _kill_a(), _clc(),
    ]
    out = _fold(stmts)
    assert isinstance(out[0], Assign)
    assert isinstance(out[0].source, RotateExpr)
    assert out[0].source.count == 3


def test_rotate_not_folded_when_a_live():
    """If A is live after the store, the fold must not fire."""
    stmts = [
        _lda("X", 0x10), _rol(), _sta("Y", 0x20),
        # A is read again here (no kill_a) — fold blocked
        _sta("Z", 0x30),
    ]
    out = _fold(stmts)
    assert not any(isinstance(s, Assign) and isinstance(s.source, RotateExpr)
                   for s in out)


def test_rotate_not_folded_when_carry_live():
    """If carry is live after the store (consumed by a bare adc), no fold."""
    stmts = [
        _lda("X", 0x10), _rol(), _sta("Y", 0x20),
        _kill_a(),
        # carry is consumed here (adc reads carry without preceding clc/sec)
        _raw(AdcImm(imm=_imm(0), src=SRC)),
    ]
    out = _fold(stmts)
    assert not any(isinstance(s, Assign) and isinstance(s.source, RotateExpr)
                   for s in out)


def test_mixed_rol_ror_not_merged():
    """rol followed by ror — different directions, fold takes only the rol run."""
    stmts = [
        _lda("X", 0x10),
        _rol(),
        _ror(),          # direction change — rol run ends here
        _sta("Y", 0x20),
        _kill_a(),
        _clc(),
    ]
    out = _fold(stmts)
    for s in out:
        if isinstance(s, Assign) and isinstance(s.source, RotateExpr):
            assert s.source.count == 1   # at most the single-rol run


def test_rotate_multi_store_not_folded():
    """Two stores after the rotate — not an idempotent write-back, no fold."""
    stmts = [
        _lda("X", 0x10), _rol(),
        _sta("Y", 0x20), _sta("Z", 0x30),   # two store targets
        _kill_a(), _clc(),
    ]
    out = _fold(stmts)
    assert not any(isinstance(s, Assign) and isinstance(s.source, RotateExpr)
                   for s in out)


# ---------------------------------------------------------------- behavioural tests


def test_behaviour_rol_carry_in_0():
    """rol with carry-in = 0 acts like asl (shifts in 0)."""
    stmts = [_lda("X", 0x10), _rol(), _sta("Y", 0x20), _kill_a(), _clc()]
    tb, ta = _run_both(stmts, {0x10: 0x15}, carry=0)
    assert tb.writes == ta.writes
    assert ta.ram[0x20] == (0x15 << 1) & 0xFF


def test_behaviour_rol_carry_in_1():
    """rol with carry-in = 1 shifts left and inserts 1 in the LSB."""
    stmts = [_lda("X", 0x10), _rol(), _sta("Y", 0x20), _kill_a(), _clc()]
    tb, ta = _run_both(stmts, {0x10: 0x40}, carry=1)
    assert tb.writes == ta.writes
    # 0x40 << 1 = 0x80, | 1 = 0x81
    assert ta.ram[0x20] == 0x81


def test_behaviour_ror_carry_in_0():
    """ror with carry-in = 0 acts like lsr (shifts in 0 from MSB)."""
    stmts = [_lda("X", 0x10), _ror(), _sta("Y", 0x20), _kill_a(), _clc()]
    tb, ta = _run_both(stmts, {0x10: 0xfe}, carry=0)
    assert tb.writes == ta.writes
    assert ta.ram[0x20] == 0x7f


def test_behaviour_ror_carry_in_1():
    """ror with carry-in = 1 inserts 1 in the MSB."""
    stmts = [_lda("X", 0x10), _ror(), _sta("Y", 0x20), _kill_a(), _clc()]
    tb, ta = _run_both(stmts, {0x10: 0x02}, carry=1)
    assert tb.writes == ta.writes
    # 0x02 >> 1 = 0x01, | 0x80 = 0x81
    assert ta.ram[0x20] == 0x81


def test_behaviour_double_rol_carry_chain():
    """Two rol: carry threads through both steps correctly."""
    stmts = [_lda("X", 0x10), _rol(), _rol(), _sta("Y", 0x20), _kill_a(), _clc()]
    # 0xC0 rol1 (C=0): (0xC0<<1)|0 = 0x80, new_C=1
    # 0x80 rol2 (C=1): (0x80<<1)|1 = 0x01, new_C=1
    tb, ta = _run_both(stmts, {0x10: 0xC0}, carry=0)
    assert tb.writes == ta.writes
    assert ta.ram[0x20] == 0x01


def test_behaviour_lda_imm_rol():
    """Immediate source folds correctly: lda #k ; rol ; sta Y."""
    stmts = [
        _raw(LoadImm(reg=Reg.A, imm=_imm(0x42), src=SRC)),
        _rol(), _sta("Y", 0x20), _kill_a(), _clc(),
    ]
    tb, ta = _run_both(stmts, {}, carry=0)
    assert tb.writes == ta.writes
    assert ta.ram[0x20] == (0x42 << 1) & 0xFF
