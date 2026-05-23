"""Pass 3 — do-while loop-condition recovery.

* **Structural unit tests** pin recognition: a `loop { … ; if Compare {
  break } }` becomes a `DoWhileStmt` whose condition is the negated exit
  test and whose body drops the guard; loops without a clean bottom
  `if Compare { break }` guard are left as `loop`s.
* **Behavioural equivalence** interprets a counted loop before and after
  recovery, asserting byte-identical RAM — the hoist must not change
  behaviour.
"""

from __future__ import annotations

from pop_lifter.interp_ir3 import run as ir3_run
from pop_lifter.ir1 import (
    Abs,
    Compare,
    DecTarget,
    Imm,
    IncTarget,
    LoadImm,
    Reg,
    SourceRef,
    StoreIndexed,
)
from pop_lifter.ir3 import (
    Block,
    BreakStmt,
    ContinueStmt,
    DoWhileStmt,
    IfStmt,
    LoopStmt,
    ModuleIR3,
    RawIfStmt,
    RawStmt,
    ReturnStmt,
    RoutineIR3,
)
from pop_lifter.pass3_loops import dowhile_stats, recover_loops, recover_routine

SRC = SourceRef(file="syn", line=0, raw="")


def _cmp(reg: Reg, op: str, k: int | None = None) -> Compare:
    rhs = None if k is None else Imm(value=k, text=f"#{k}")
    return Compare(reg=reg, op=op, rhs=rhs)


def _guard(cond: Compare) -> IfStmt:
    return IfStmt(cond=cond, then_block=Block.of([BreakStmt(src=SRC)]),
                  else_block=None, src=SRC)


def _loop(body: list) -> LoopStmt:
    return LoopStmt(body=Block.of(body), src=SRC)


def _recover_one(stmts: list):
    return recover_routine(RoutineIR3(name="t", body=Block.of(stmts))).body.stmts


# --------------------------------------------------------------- structural


def test_bottom_test_loop_becomes_dowhile():
    body = [RawStmt(DecTarget(target=Reg.Y, src=SRC)), _guard(_cmp(Reg.Y, "<0"))]
    out = _recover_one([_loop(body)])
    assert len(out) == 1 and isinstance(out[0], DoWhileStmt)
    dw = out[0]
    # Exit test `y < 0` becomes continue condition `y >= 0`.
    assert dw.cond.op == ">=0" and dw.cond.reg is Reg.Y
    # The guard is gone; the body keeps the rest.
    assert len(dw.body.stmts) == 1 and isinstance(dw.body.stmts[0], RawStmt)
    assert not any(isinstance(s, BreakStmt) for s in dw.body.stmts)


def test_all_exit_ops_negate():
    cases = [("==", "!="), ("!=", "=="), ("<", ">="), (">=", "<"),
             ("<0", ">=0"), (">=0", "<0")]
    for exit_op, cont_op in cases:
        k = None if exit_op.endswith("0") else 5
        out = _recover_one([_loop([_guard(_cmp(Reg.A, exit_op, k))])])
        assert isinstance(out[0], DoWhileStmt)
        assert out[0].cond.op == cont_op


def test_loop_without_bottom_guard_stays_loop():
    # Guard is at the top, not the bottom.
    out = _recover_one([_loop([_guard(_cmp(Reg.Y, "<0")),
                               RawStmt(DecTarget(target=Reg.Y, src=SRC))])])
    assert isinstance(out[0], LoopStmt)


def test_guard_with_extra_then_stmts_not_recovered():
    # then-block isn't a bare break.
    g = IfStmt(
        cond=_cmp(Reg.Y, "<0"),
        then_block=Block.of([RawStmt(DecTarget(target=Reg.X, src=SRC)), BreakStmt(src=SRC)]),
        else_block=None, src=SRC,
    )
    out = _recover_one([_loop([g])])
    assert isinstance(out[0], LoopStmt)


def test_guard_with_else_not_recovered():
    g = IfStmt(
        cond=_cmp(Reg.Y, "<0"),
        then_block=Block.of([BreakStmt(src=SRC)]),
        else_block=Block.of([RawStmt(DecTarget(target=Reg.X, src=SRC))]),
        src=SRC,
    )
    out = _recover_one([_loop([g])])
    assert isinstance(out[0], LoopStmt)


def test_rawif_guard_not_recovered():
    g = RawIfStmt(cond="mi", then_block=Block.of([BreakStmt(src=SRC)]),
                  else_block=None, src=SRC)
    out = _recover_one([_loop([g])])
    assert isinstance(out[0], LoopStmt)


def test_nested_loop_recovered():
    inner = _loop([RawStmt(DecTarget(target=Reg.X, src=SRC)), _guard(_cmp(Reg.X, "<0"))])
    outer = _loop([inner, RawStmt(DecTarget(target=Reg.Y, src=SRC)), _guard(_cmp(Reg.Y, "<0"))])
    out = _recover_one([outer])
    assert isinstance(out[0], DoWhileStmt)
    assert any(isinstance(s, DoWhileStmt) for s in out[0].body.stmts)


# --------------------------------------------------------------- behavioural


def test_dowhile_recovery_is_behaviour_preserving():
    """A counted loop writing `y` to `OUT[y]` for y = 3..0. Recovery to
    `do { … } while y >= 0` must produce identical RAM."""
    def routine() -> RoutineIR3:
        return RoutineIR3(name="counter", body=Block.of([
            RawStmt(LoadImm(reg=Reg.Y, imm=Imm(value=3, text="#3"), src=SRC)),
            _loop([
                RawStmt(StoreIndexed(
                    reg=Reg.Y, base=Abs(name="OUT", addr=0x300), index=Reg.Y, src=SRC)),
                RawStmt(DecTarget(target=Reg.Y, src=SRC)),
                _guard(_cmp(Reg.Y, "<0")),
            ]),
            ReturnStmt(src=SRC),
        ]))

    pre = ModuleIR3("M", "syn", [routine()])
    post = recover_loops(pre)
    assert dowhile_stats(post) == 1

    r1 = bytearray(0x10000)
    ir3_run([pre], "counter", ram=r1)
    r2 = bytearray(0x10000)
    ir3_run([post], "counter", ram=r2)
    assert r1 == r2
    assert list(r2[0x300:0x304]) == [0, 1, 2, 3]


def test_continue_restarts_body_without_testing_condition():
    """The subtle `DoWhileStmt` semantic: a `continue` restarts the body
    *without* re-testing the bottom condition (unlike a C/Rust `do { }
    while`). Here the body always `continue`s and the continue condition
    `y == 0` is false once `y` advances — a C-style do-while would test
    it and exit after one iteration, ours loops to the break guard."""
    body = [
        RawStmt(IncTarget(target=Reg.Y, src=SRC)),
        RawStmt(StoreIndexed(
            reg=Reg.Y, base=Abs(name="OUT", addr=0x300), index=Reg.Y, src=SRC)),
        _guard(_cmp(Reg.Y, ">=", 3)),       # bound: break once y >= 3
        ContinueStmt(src=SRC),              # always restart the body
    ]
    dw = DoWhileStmt(body=Block.of(body), cond=_cmp(Reg.Y, "==", 0), src=SRC)
    mod = ModuleIR3("M", "syn", [RoutineIR3(name="c", body=Block.of([dw, ReturnStmt(src=SRC)]))])

    ram = bytearray(0x10000)
    ir3_run([mod], "c", ram=ram)
    # Looped to the break guard (y reached 3), writing 1,2,3 — not just 1
    # (which is what testing the false condition after `continue` would give).
    assert list(ram[0x300:0x304]) == [0, 1, 2, 3]
