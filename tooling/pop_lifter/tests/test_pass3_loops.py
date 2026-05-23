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
    Unsupported,
)
from pop_lifter.ir3 import (
    Block,
    BreakStmt,
    CallStmt,
    ContinueStmt,
    DoWhileStmt,
    ForStmt,
    IfStmt,
    LoopStmt,
    ModuleIR3,
    RawIfStmt,
    RawStmt,
    RepeatStmt,
    ReturnStmt,
    RoutineIR3,
)
from pop_lifter.pass3_loops import (
    dowhile_stats,
    for_stats,
    recover_loops,
    recover_routine,
    repeat_stats,
)

SRC = SourceRef(file="syn", line=0, raw="")


def _cmp(reg: Reg, op: str, k: int | None = None) -> Compare:
    rhs = None if k is None else Imm(value=k, text=f"#{k}")
    return Compare(reg=reg, op=op, rhs=rhs)


def _guard(cond: Compare) -> IfStmt:
    return IfStmt(cond=cond, then_block=Block.of([BreakStmt(src=SRC)]),
                  else_block=None, src=SRC)


def _loop(body: list) -> LoopStmt:
    return LoopStmt(body=Block.of(body), src=SRC)


def _ldimm(reg: Reg, v: int) -> RawStmt:
    return RawStmt(LoadImm(reg=reg, imm=Imm(value=v, text=f"#{v}"), src=SRC))


def _dec(reg: Reg) -> RawStmt:
    return RawStmt(DecTarget(target=reg, src=SRC))


def _down_counter(init: int, body: list, reg: Reg = Reg.Y) -> list:
    """`reg = #init ; loop { *body* ; reg -= 1 ; if reg < 0 break }` —
    the dey/bpl down-counter shape recovery should promote to a for."""
    return [
        _ldimm(reg, init),
        _loop([*body, _dec(reg), _guard(_cmp(reg, "<0"))]),
    ]


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


# --------------------------------------------------------------- for recovery


def test_down_counter_becomes_for():
    body = [RawStmt(StoreIndexed(
        reg=Reg.Y, base=Abs(name="OUT", addr=0x300), index=Reg.Y, src=SRC))]
    out = _recover_one(_down_counter(6, body))
    assert len(out) == 1 and isinstance(out[0], ForStmt)
    f = out[0]
    assert f.var is Reg.Y and (f.start.value & 0xff) == 6 and f.step == -1
    # The init LoadImm and the trailing dey are subsumed.
    assert len(f.body.stmts) == 1 and isinstance(f.body.stmts[0], RawStmt)
    assert not any(isinstance(s, BreakStmt) for s in f.body.stmts)


def _not_a_for(out) -> bool:
    """A non-promoted counter keeps its init `LoadImm` and recovers a
    `DoWhileStmt` — never a `ForStmt`."""
    return (not any(isinstance(s, ForStmt) for s in out)
            and any(isinstance(s, DoWhileStmt) for s in out))


def test_negative_init_not_promoted_to_for():
    """init >= 0x80 (negative as a sign test) means the do-while body
    might run a different number of times than a top-tested for — left
    as a do-while."""
    assert _not_a_for(_recover_one(_down_counter(0x80, [_ldimm(Reg.A, 0)])))


def test_counter_rewritten_in_body_not_promoted():
    """If the counter is also written inside the body it isn't a clean
    induction variable — left as a do-while."""
    body = [_ldimm(Reg.Y, 9)]  # clobbers Y mid-body
    assert _not_a_for(_recover_one(_down_counter(6, body)))


def test_continue_in_body_not_promoted():
    out = _recover_one([
        _ldimm(Reg.Y, 6),
        _loop([ContinueStmt(src=SRC), _dec(Reg.Y), _guard(_cmp(Reg.Y, "<0"))]),
    ])
    assert _not_a_for(out)


def test_up_counter_becomes_for():
    """`x = #0 ; do { body ; x += 1 } while x != 5` ⇒ `for x in #0..#5`."""
    out = _recover_one([
        _ldimm(Reg.X, 0),
        _loop([RawStmt(StoreIndexed(
            reg=Reg.X, base=Abs(name="OUT", addr=0x300), index=Reg.X, src=SRC)),
            RawStmt(IncTarget(target=Reg.X, src=SRC)),
            _guard(_cmp(Reg.X, "==", 5))]),
    ])
    assert len(out) == 1 and isinstance(out[0], ForStmt)
    f = out[0]
    assert f.var is Reg.X and (f.start.value & 0xff) == 0 and f.step == 1
    assert f.cond.op == "!=" and (f.cond.rhs.value & 0xff) == 5


def test_up_counter_with_start_eq_bound_is_repeat_not_for():
    """`start == N` isn't a `start..N` range (it would be empty) — the
    counter wraps the full byte range instead, so it's recovered as a
    `repeat`, never a `for`."""
    out = _recover_one([
        _ldimm(Reg.X, 5),
        _loop([RawStmt(IncTarget(target=Reg.X, src=SRC)),
               _guard(_cmp(Reg.X, "==", 5))]),  # start == N → full wrap
    ])
    assert not any(isinstance(s, ForStmt) for s in out)
    assert any(isinstance(s, RepeatStmt) for s in out)


def test_up_counter_with_memory_bound_not_promoted():
    """A non-constant bound (`cpx mem`) can't be a fixed `start..N`
    range — left as a do-while."""
    out = _recover_one([
        _ldimm(Reg.X, 0),
        _loop([RawStmt(IncTarget(target=Reg.X, src=SRC)),
               IfStmt(cond=Compare(reg=Reg.X, op="==",
                                   rhs=Abs(name="bound", addr=0x90)),
                      then_block=Block.of([BreakStmt(src=SRC)]),
                      else_block=None, src=SRC)]),
    ])
    assert _not_a_for(out)


def test_call_in_body_not_promoted():
    """A `call` in the body may clobber X/Y (no proof the callee
    preserves it), so the counter isn't provably clean — not a for."""
    out = _recover_one(_down_counter(6, [CallStmt(target="foo", src=SRC)]))
    assert _not_a_for(out)


def test_unsupported_op_in_body_not_promoted():
    """An unmodelled opcode has unknown register effects — conservatively
    blocks promotion."""
    body = [RawStmt(Unsupported(mnemonic="wat", operand=None, src=SRC))]
    out = _recover_one(_down_counter(6, body))
    assert _not_a_for(out)


def test_symbolic_init_bound_preserved_in_dump():
    """A symbolic counter bound (`#numslots`) renders as the name, not
    its assembled byte, via `_fmt_imm`."""
    from pop_lifter import ir3 as ir3_mod
    init = Imm(value=0x06, text="#numslots")
    body = [RawStmt(StoreIndexed(
        reg=Reg.Y, base=Abs(name="OUT", addr=0x300), index=Reg.Y, src=SRC))]
    out = _recover_one([
        RawStmt(LoadImm(reg=Reg.Y, imm=init, src=SRC)),
        _loop([*body, _dec(Reg.Y), _guard(_cmp(Reg.Y, "<0"))]),
    ])
    assert isinstance(out[0], ForStmt)
    dump = "\n".join(ir3_mod._fmt_stmt(out[0], 0))
    assert "(0..=#numslots).rev()" in dump


def test_for_recovery_is_behaviour_preserving():
    """The recovered `for y in (0..=3).rev()` writing `y` to `OUT[y]`
    must produce identical RAM (and final register/flag state) to the
    original `loop`/do-while."""
    def routine() -> RoutineIR3:
        body = [RawStmt(StoreIndexed(
            reg=Reg.Y, base=Abs(name="OUT", addr=0x300), index=Reg.Y, src=SRC))]
        return RoutineIR3(name="counter", body=Block.of([
            *_down_counter(3, body),
            ReturnStmt(src=SRC),
        ]))

    pre = ModuleIR3("M", "syn", [routine()])
    post = recover_loops(pre)
    assert for_stats(post) == 1 and dowhile_stats(post) == 0

    r1 = bytearray(0x10000)
    t1 = ir3_run([pre], "counter", ram=r1)
    r2 = bytearray(0x10000)
    t2 = ir3_run([post], "counter", ram=r2)
    assert r1 == r2
    assert list(r2[0x300:0x304]) == [0, 1, 2, 3]
    # Final counter value (0xff after the last dey) must match too.
    assert t1.y == t2.y == 0xff


# --------------------------------------------------------------- delay loops


def test_full_wrap_delay_becomes_repeat():
    """`x = #0 ; do { x -= 1 } while x != 0` (exit value == start) wraps
    the full byte range — recovered as `repeat 0x100 {}`."""
    out = _recover_one([
        _ldimm(Reg.X, 0),
        _loop([_dec(Reg.X), _guard(_cmp(Reg.X, "==", 0))]),
    ])
    assert len(out) == 1 and isinstance(out[0], RepeatStmt)
    r = out[0]
    assert r.count == 0x100 and r.var is Reg.X and r.step == -1
    assert len(r.body.stmts) == 0  # the dex is the step, body is empty


def test_delay_with_nonmatching_exit_not_repeat():
    """`x = #5 ; do { dex } while x != 0` exits at 0 (!= start) — a
    down-counter, not a full wrap, so not a `repeat`."""
    out = _recover_one([
        _ldimm(Reg.X, 5),
        _loop([_dec(Reg.X), _guard(_cmp(Reg.X, "==", 0))]),
    ])
    assert not any(isinstance(s, RepeatStmt) for s in out)


def test_delay_body_reading_counter_not_repeat():
    """If the body reads the counter, the per-iteration value matters —
    `repeat` would lose it, so it's left as a do-while."""
    body = [RawStmt(StoreIndexed(
        reg=Reg.X, base=Abs(name="OUT", addr=0x300), index=Reg.X, src=SRC))]
    out = _recover_one([
        _ldimm(Reg.X, 0),
        _loop([*body, _dec(Reg.X), _guard(_cmp(Reg.X, "==", 0))]),
    ])
    assert not any(isinstance(s, RepeatStmt) for s in out)


def test_delay_recovery_is_behaviour_preserving():
    """`x = #0 ; do { *MARK += 1 ; dex } while x != 0` increments MARK
    256 times and leaves x = 0. Recovery to `repeat 0x100` must match."""
    from pop_lifter.ir1 import IncTarget as _Inc
    def routine() -> RoutineIR3:
        return RoutineIR3(name="delay", body=Block.of([
            _ldimm(Reg.X, 0),
            _loop([
                RawStmt(_Inc(target=Abs(name="MARK", addr=0x300), src=SRC)),
                _dec(Reg.X),
                _guard(_cmp(Reg.X, "==", 0)),
            ]),
            ReturnStmt(src=SRC),
        ]))

    pre = ModuleIR3("M", "syn", [routine()])
    post = recover_loops(pre)
    assert repeat_stats(post) == 1

    r1 = bytearray(0x10000)
    t1 = ir3_run([pre], "delay", ram=r1)
    r2 = bytearray(0x10000)
    t2 = ir3_run([post], "delay", ram=r2)
    assert r1 == r2
    assert r2[0x300] == 0x00  # incremented 256 times, wraps back to 0
    assert t1.x == t2.x == 0  # counter wrapped back to its start


# --------------------------------------------------------------- behavioural


def test_dowhile_recovery_is_behaviour_preserving():
    """A `>=`-bounded down-loop `x = 3 ; loop { OUT[x]=x ; x -= 1 ;
    if x < 1 break }` recovers to a `do { … } while x >= 1` (not a for —
    only `>= 0` / `!=` bounds are promoted) and must produce identical
    RAM."""
    def routine() -> RoutineIR3:
        return RoutineIR3(name="counter", body=Block.of([
            RawStmt(LoadImm(reg=Reg.X, imm=Imm(value=3, text="#3"), src=SRC)),
            _loop([
                RawStmt(StoreIndexed(
                    reg=Reg.X, base=Abs(name="OUT", addr=0x300), index=Reg.X, src=SRC)),
                RawStmt(DecTarget(target=Reg.X, src=SRC)),
                _guard(_cmp(Reg.X, "<", 1)),
            ]),
            ReturnStmt(src=SRC),
        ]))

    pre = ModuleIR3("M", "syn", [routine()])
    post = recover_loops(pre)
    assert dowhile_stats(post) == 1 and for_stats(post) == 0

    r1 = bytearray(0x10000)
    ir3_run([pre], "counter", ram=r1)
    r2 = bytearray(0x10000)
    ir3_run([post], "counter", ram=r2)
    assert r1 == r2
    assert list(r2[0x301:0x304]) == [1, 2, 3]


def test_up_counter_for_is_behaviour_preserving():
    """The recovered `for x in #0..#4` writing `x` to `OUT[x]` must
    produce identical RAM (and final register state x = 4) to the
    original up-counting loop."""
    def routine() -> RoutineIR3:
        return RoutineIR3(name="counter", body=Block.of([
            RawStmt(LoadImm(reg=Reg.X, imm=Imm(value=0, text="#0"), src=SRC)),
            _loop([
                RawStmt(StoreIndexed(
                    reg=Reg.X, base=Abs(name="OUT", addr=0x300), index=Reg.X, src=SRC)),
                RawStmt(IncTarget(target=Reg.X, src=SRC)),
                _guard(_cmp(Reg.X, "==", 4)),
            ]),
            ReturnStmt(src=SRC),
        ]))

    pre = ModuleIR3("M", "syn", [routine()])
    post = recover_loops(pre)
    assert for_stats(post) == 1 and dowhile_stats(post) == 0

    r1 = bytearray(0x10000)
    t1 = ir3_run([pre], "counter", ram=r1)
    r2 = bytearray(0x10000)
    t2 = ir3_run([post], "counter", ram=r2)
    assert r1 == r2
    assert list(r2[0x300:0x304]) == [0, 1, 2, 3]
    assert t1.x == t2.x == 4


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
