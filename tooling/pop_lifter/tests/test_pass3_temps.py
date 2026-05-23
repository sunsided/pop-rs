"""Pass 3 — pha/pla scoped-temporary recovery.

A `push a ; … ; a = pop` bracket is matched into a `SaveTemp` /
`RestoreTemp` pair sharing a `slot`, naming the saved byte as a scoped
temporary. Matching nests like brackets within a flat block: plain ops
and balanced calls are spanned; a control transfer (or an `Unsupported`
op) is a barrier that forgoes the pair.

* **Structural tests** pin recognition: simple pairs, the swap idiom,
  nesting, call-spanning, and the non-firing cases (barrier between,
  unmatched push / pop, recursion into nested blocks).
* **Behavioural equivalence** interprets before and after recovery,
  asserting identical write sets (``Trace.writes``) — relabelling
  pha/pla must not change observable behaviour.
"""

from __future__ import annotations

from pop_lifter.interp_ir3 import run as ir3_run
from pop_lifter.ir1 import (
    Abs,
    Compare,
    Imm,
    LoadAbs,
    LoadImm,
    Pha,
    Pla,
    Reg,
    SourceRef,
    StoreAbs,
    Unsupported,
)
from pop_lifter.ir3 import (
    Block,
    CallStmt,
    IfStmt,
    ModuleIR3,
    RawStmt,
    RestoreTemp,
    ReturnStmt,
    RoutineIR3,
    SaveTemp,
)
from pop_lifter.pass3_temps import recover_routine, recover_temps, temp_stats

SRC = SourceRef(file="syn", line=0, raw="")


# ---------------------------------------------------------------- helpers


def _raw(item) -> RawStmt:
    return RawStmt(item=item)


def _imm(v: int) -> Imm:
    return Imm(value=v, text=f"#{v:#04x}")


def _abs(name: str, addr: int) -> Abs:
    return Abs(name=name, addr=addr)


def _lda_imm(v: int) -> RawStmt:
    return _raw(LoadImm(reg=Reg.A, imm=_imm(v), src=SRC))


def _lda(name: str, addr: int) -> RawStmt:
    return _raw(LoadAbs(reg=Reg.A, source=_abs(name, addr), src=SRC))


def _sta(name: str, addr: int) -> RawStmt:
    return _raw(StoreAbs(reg=Reg.A, target=_abs(name, addr), src=SRC))


def _pha() -> RawStmt:
    return _raw(Pha(src=SRC))


def _pla() -> RawStmt:
    return _raw(Pla(src=SRC))


def _unsupported() -> RawStmt:
    return _raw(Unsupported(mnemonic="php", operand=None, src=SRC))


def _recover(stmts: list) -> list:
    routine = RoutineIR3(name="syn", body=Block.of(stmts))
    return list(recover_routine(routine).body.stmts)


# ---------------------------------------------------------------- structural tests


def test_simple_pair_matched():
    """push a ; <neutral> ; a = pop → SaveTemp / RestoreTemp, same slot."""
    out = _recover([_pha(), _lda_imm(0x99), _sta("Z", 0x30), _pla()])
    assert isinstance(out[0], SaveTemp)
    assert isinstance(out[3], RestoreTemp)
    assert out[0].slot == out[3].slot
    # The spanned ops are untouched.
    assert isinstance(out[1], RawStmt) and isinstance(out[2], RawStmt)


def test_pair_spans_call():
    """A balanced call sits between the push and pop — still matched."""
    out = _recover([_pha(), CallStmt(target="cut", src=SRC), _pla()])
    assert isinstance(out[0], SaveTemp)
    assert isinstance(out[1], CallStmt)
    assert isinstance(out[2], RestoreTemp)
    assert out[0].slot == out[2].slot


def test_swap_idiom_matched():
    """push a ; load ; store ; a = pop ; store-saved."""
    out = _recover([
        _pha(),
        _lda("A1", 0x10),
        _sta("B1", 0x20),
        _pla(),
        _sta("A1", 0x10),
    ])
    assert isinstance(out[0], SaveTemp)
    assert isinstance(out[3], RestoreTemp)
    assert out[0].slot == out[3].slot


def test_nested_pairs_distinct_slots():
    """Two nested brackets get two distinct slots, inner popped first."""
    out = _recover([_pha(), _pha(), _pla(), _pla()])
    assert isinstance(out[0], SaveTemp)
    assert isinstance(out[1], SaveTemp)
    assert isinstance(out[2], RestoreTemp)
    assert isinstance(out[3], RestoreTemp)
    # LIFO: inner push (1) pairs with first pop (2); outer push (0) with (3).
    assert out[1].slot == out[2].slot
    assert out[0].slot == out[3].slot
    assert out[0].slot != out[1].slot


def test_no_match_across_control_barrier():
    """A push before an `if` is not paired with a pop after it."""
    guard = IfStmt(
        cond=Compare(reg=Reg.A, op="==", rhs=_imm(0)),
        then_block=Block.of([ReturnStmt(src=SRC)]),
        else_block=None,
        src=SRC,
    )
    out = _recover([_pha(), guard, _pla()])
    assert not any(isinstance(s, (SaveTemp, RestoreTemp)) for s in out)
    assert isinstance(out[0], RawStmt) and isinstance(out[0].item, Pha)
    assert isinstance(out[2], RawStmt) and isinstance(out[2].item, Pla)


def test_no_match_across_unsupported():
    """An Unsupported op (might touch the stack, e.g. php) is a barrier."""
    out = _recover([_pha(), _unsupported(), _pla()])
    assert not any(isinstance(s, (SaveTemp, RestoreTemp)) for s in out)


def test_unmatched_push_left_raw():
    """A push with no following pop stays a raw `pha`."""
    out = _recover([_pha(), _sta("Z", 0x30)])
    assert isinstance(out[0], RawStmt) and isinstance(out[0].item, Pha)
    assert not any(isinstance(s, SaveTemp) for s in out)


def test_unmatched_pop_left_raw():
    """A pop with no preceding push stays a raw `pla`."""
    out = _recover([_sta("Z", 0x30), _pla()])
    assert isinstance(out[1], RawStmt) and isinstance(out[1].item, Pla)
    assert not any(isinstance(s, RestoreTemp) for s in out)


def test_recursion_into_nested_block():
    """A pair entirely inside an `if` body is recovered."""
    guard = IfStmt(
        cond=Compare(reg=Reg.A, op="==", rhs=_imm(0)),
        then_block=Block.of([_pha(), _lda_imm(0x99), _pla()]),
        else_block=None,
        src=SRC,
    )
    out = _recover([guard])
    body = out[0].then_block.stmts
    assert isinstance(body[0], SaveTemp)
    assert isinstance(body[2], RestoreTemp)
    assert body[0].slot == body[2].slot


def test_slots_unique_across_nested_and_outer():
    """A top-level pair and a pair inside an `if` body get distinct slots
    (the slot counter is routine-wide). The two can't be a single
    bracket — the `if` is a barrier — so they sit side by side."""
    guard = IfStmt(
        cond=Compare(reg=Reg.A, op="==", rhs=_imm(0)),
        then_block=Block.of([_pha(), _lda_imm(1), _pla()]),
        else_block=None,
        src=SRC,
    )
    out = _recover([_pha(), _lda_imm(2), _pla(), guard])
    outer = [s for s in out if isinstance(s, SaveTemp)]
    inner = [s for s in out[3].then_block.stmts if isinstance(s, SaveTemp)]
    assert len(inner) == 1 and len(outer) == 1
    assert inner[0].slot != outer[0].slot


def test_temp_stats_counts_pairs():
    module = ModuleIR3(
        name="m", file="syn",
        routines=[RoutineIR3(name="r", body=Block.of([
            _pha(), _lda_imm(1), _pla(),
            _pha(), _lda_imm(2), _pla(),
        ]))],
    )
    assert temp_stats(recover_temps(module)) == 2


# ---------------------------------------------------------------- behavioural tests


def _module(stmts: list, *, extra_routines=()) -> ModuleIR3:
    routines = [RoutineIR3(
        name="test",
        body=Block.of(stmts + [ReturnStmt(src=SRC)]),
    )]
    routines.extend(extra_routines)
    return ModuleIR3(name="test", file="syn", routines=routines)


def _run_both(module: ModuleIR3, init: dict[int, int]):
    ram0 = bytearray(0x10000)
    for addr, val in init.items():
        ram0[addr] = val
    after = recover_temps(module)
    t_before = ir3_run([module], "test", ram=bytearray(ram0))
    t_after = ir3_run([after], "test", ram=bytearray(ram0))
    return t_before, t_after


def test_behaviour_save_restore():
    """A is clobbered between push/pop; the pop restores the saved byte."""
    stmts = [
        _lda_imm(0x11),     # A = 0x11
        _pha(),             # save 0x11
        _lda_imm(0x99),     # clobber A
        _sta("Z", 0x30),    # Z = 0x99
        _pla(),             # A = 0x11 restored
        _sta("Y", 0x20),    # Y = 0x11
    ]
    tb, ta = _run_both(_module(stmts), {})
    assert tb.writes == ta.writes
    assert tb.ram[0x20] == ta.ram[0x20] == 0x11
    assert tb.ram[0x30] == ta.ram[0x30] == 0x99


def test_behaviour_save_restore_across_call():
    """The pair brackets a call that clobbers A; A survives it."""
    clobber = RoutineIR3(
        name="clobber",
        body=Block.of([_lda_imm(0x99), ReturnStmt(src=SRC)]),
    )
    stmts = [
        _lda_imm(0x42),                      # A = 0x42
        _pha(),                              # save
        CallStmt(target="clobber", src=SRC),  # trashes A
        _pla(),                              # restore 0x42
        _sta("Y", 0x20),                     # Y = 0x42
    ]
    tb, ta = _run_both(_module(stmts, extra_routines=(clobber,)), {})
    assert tb.writes == ta.writes
    assert tb.ram[0x20] == ta.ram[0x20] == 0x42


def test_behaviour_nested():
    """Nested save/restore round-trips both bytes."""
    stmts = [
        _lda_imm(0xAA), _pha(),     # save 0xAA (outer)
        _lda_imm(0xBB), _pha(),     # save 0xBB (inner)
        _lda_imm(0x00),             # clobber
        _pla(), _sta("I", 0x21),    # inner restore → 0xBB
        _pla(), _sta("O", 0x22),    # outer restore → 0xAA
    ]
    tb, ta = _run_both(_module(stmts), {})
    assert tb.writes == ta.writes
    assert tb.ram[0x21] == ta.ram[0x21] == 0xBB
    assert tb.ram[0x22] == ta.ram[0x22] == 0xAA
