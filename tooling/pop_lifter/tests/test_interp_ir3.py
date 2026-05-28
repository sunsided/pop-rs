"""IR3 interpreter semantics that aren't covered by the pass-specific
tests — currently: a tail-call inside a *called* routine returns to the
caller, not to the top level.
"""

from __future__ import annotations

from pop_lifter.interp_ir3 import run as ir3_run
from pop_lifter.ir1 import Abs, Imm, LoadImm, Reg, SourceRef, StoreAbs
from pop_lifter.ir3 import (
    Block,
    CallStmt,
    ModuleIR3,
    RawStmt,
    ReturnStmt,
    RoutineIR3,
    TailCallStmt,
)

SRC = SourceRef(file="syn", line=0, raw="")


def _raw(item):
    return RawStmt(item=item)


def _lda(v: int):
    return _raw(LoadImm(reg=Reg.A, imm=Imm(value=v, text=f"#${v:02x}"), src=SRC))


def _sta(addr: int):
    return _raw(StoreAbs(reg=Reg.A, target=Abs(name=f"m{addr:x}", addr=addr), src=SRC))


def test_tail_call_inside_callee_returns_to_caller():
    """`main` does `jsr sub`; `sub` ends with `jmp tail` (a tail call);
    `tail` returns. On a 6502, `jsr sub` pushed a return address inside
    `main`, and `sub`'s `jmp` pushes nothing, so `tail`'s `rts` pops that
    address — execution resumes in `main` right after the `jsr`. Regression
    for the IR3 interpreter unwinding the tail-call past the call frame to
    the top-level loop, which abandoned the rest of `main`.

        main: jsr sub ; a=$aa ; sta $80 ; rts
        sub:  a=$01 ; jmp tail
        tail: sta $81 ; rts
    """
    main = RoutineIR3(name="main", body=Block.of([
        CallStmt(target="sub", src=SRC),
        _lda(0xAA),
        _sta(0x80),
        ReturnStmt(src=SRC),
    ]))
    sub = RoutineIR3(name="sub", body=Block.of([
        _lda(0x01),
        TailCallStmt(target="tail", src=SRC),
    ]))
    tail = RoutineIR3(name="tail", body=Block.of([
        _sta(0x81),
        ReturnStmt(src=SRC),
    ]))
    mod = ModuleIR3(name="SYN", file="syn", routines=[main, sub, tail])
    trace = ir3_run([mod], "main", ram=bytearray(0x10000))

    # `tail` ran (via the tail-call) ...
    assert trace.ram[0x81] == 0x01
    # ... AND `main` continued after the call rather than being abandoned.
    assert trace.ram[0x80] == 0xAA, (
        "main did not resume after the callee's tail-call — the tail-call "
        "unwound past the jsr boundary"
    )


def test_chained_tail_calls_inside_callee():
    """`sub` tail-calls `t1`, which tail-calls `t2`; the whole chain runs
    and control still returns to `main` after the `jsr sub`."""
    main = RoutineIR3(name="main", body=Block.of([
        CallStmt(target="sub", src=SRC),
        _lda(0xAA),
        _sta(0x80),
        ReturnStmt(src=SRC),
    ]))
    sub = RoutineIR3(name="sub", body=Block.of([TailCallStmt(target="t1", src=SRC)]))
    t1 = RoutineIR3(name="t1", body=Block.of([TailCallStmt(target="t2", src=SRC)]))
    t2 = RoutineIR3(name="t2", body=Block.of([_lda(0x07), _sta(0x81), ReturnStmt(src=SRC)]))
    mod = ModuleIR3(name="SYN", file="syn", routines=[main, sub, t1, t2])
    trace = ir3_run([mod], "main", ram=bytearray(0x10000))

    assert trace.ram[0x81] == 0x07   # chain reached t2
    assert trace.ram[0x80] == 0xAA   # main resumed after the call
