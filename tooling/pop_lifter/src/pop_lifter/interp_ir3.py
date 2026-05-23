"""IR3 interpreter — recursive walker over the structured statement
tree. Delegates per-opcode atom semantics to `interp_ir1.exec_atom`
so the IR3 and IR1 interpreters are guaranteed to agree on what
each load/store/cmp/etc. does.

What the walker handles itself:

* `RawStmt`: dispatched through `exec_atom`.
* `IfStmt` / `RawIfStmt`: evaluate cond, walk one branch.
* `ReturnStmt`: raises `_ReturnSignal` to unwind out of nested blocks
  back to the enclosing routine frame.
* `TailCallStmt`: raises `_TailCallSignal` to the top-level loop,
  which switches to the target routine.
* `CallStmt`: recurses into the callee (possibly an IR1 routine,
  resolved via the same alias index as IR1).
* `GotoStmt` / `LabelStmt`: only emitted when the relooper had to
  bail (loops / malformed input). The interpreter rejects them with
  `InterpError` so the test surface flags any structurer regression.
"""

from __future__ import annotations

from .interp_ir1 import (
    InterpError,
    Trace,
    _branch_taken,
    _eval_compare,
    _indexed_addr,
    _real_addr,
    _resolve,
    _resolve_indirect_y,
    exec_atom,
)
from .interp_ir1 import run as ir1_run
from .ir1 import Abs, Imm, IndexedAbs, IndirectY, ModuleIR1, Reg, Routine as IR1Routine
from .ir3 import (
    Assign,
    BinExpr,
    Block,
    BreakStmt,
    CallStmt,
    ContinueStmt,
    GotoStmt,
    IfStmt,
    LabelStmt,
    DoWhileStmt,
    LoopStmt,
    MatchStmt,
    ModuleIR3,
    RawIfStmt,
    RawStmt,
    ReturnStmt,
    RoutineIR3,
    Stmt,
    TailCallStmt,
)


class _ReturnSignal(Exception):
    """Raised by ReturnStmt to unwind to the enclosing call frame."""


class _TailCallSignal(Exception):
    """Carries a tail-call target up the recursion. The top-level
    `run` loop catches it and switches routines."""

    def __init__(self, target: str):
        super().__init__(target)
        self.target = target


class _BreakSignal(Exception):
    """Raised by BreakStmt to exit the innermost enclosing LoopStmt."""


class _ContinueSignal(Exception):
    """Raised by ContinueStmt to jump to the top of the innermost
    enclosing LoopStmt."""


def run(
    modules: list[ModuleIR3 | ModuleIR1],
    entry: str,
    *,
    ram: bytearray | None = None,
    aliases: dict[str, str] | None = None,
) -> Trace:
    """Execute `entry` in the given module set (mix of IR3 / IR1).

    Tail-call chaining is handled by an outer loop so deeply-nested
    cross-module tails don't stack the Python recursion. Calls and
    nested blocks recurse normally.
    """
    if ram is None:
        ram = bytearray(0x10000)
    trace = Trace(ram=ram, a=0, x=0, y=0)
    alias_idx = _alias_index(modules)
    if aliases:
        alias_idx.update(aliases)

    routine = _resolve(modules, alias_idx, entry)
    if routine is None:
        raise InterpError(f"entry {entry!r} not found in any module")

    while True:
        try:
            _exec_routine(routine, modules, alias_idx, trace)
            return trace
        except _TailCallSignal as tc:
            target = _resolve(modules, alias_idx, tc.target)
            if target is None:
                raise InterpError(
                    f"tail-call target {tc.target!r} not found in any module"
                )
            routine = target
            continue


def _alias_index(modules) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in modules:
        for r in m.routines:
            out[r.name] = r.name
            for a in getattr(r, "entry_aliases", []) or []:
                out[a] = r.name
    return out


def _exec_routine(routine, modules, aliases, trace: Trace) -> None:
    if isinstance(routine, RoutineIR3):
        try:
            _exec_block(routine.body, modules, aliases, trace)
        except _ReturnSignal:
            pass
        return
    if isinstance(routine, IR1Routine):
        # Delegate to the IR1 interpreter for IR1 leaves. We share the
        # RAM bytearray via `trace.ram`, so observable state stays in
        # sync.
        ir1_modules = [m for m in modules if isinstance(m, ModuleIR1)]
        # We need the IR1 routine reachable via the same alias map.
        # Re-running `ir1_run` with the same alias index keeps the
        # behaviour identical to "this routine in isolation".
        ir1_run(
            ir1_modules,
            routine.name,
            ram=trace.ram,
            a=trace.a, x=trace.x, y=trace.y, c=trace.c,
            aliases=aliases,
        )
        return
    raise InterpError(f"unknown routine type: {type(routine).__name__}")


def _exec_block(block: Block, modules, aliases, trace: Trace) -> None:
    for stmt in block.stmts:
        _exec_stmt(stmt, modules, aliases, trace)


def _assign_read(value, trace: Trace, src) -> int:
    """Read the source of a pass-3 `Assign` — an immediate or one of the
    memory-read forms — at the current register state. Mirrors the
    load-side of `exec_atom` so the fold's interpretation can't drift
    from the unfolded `lda` it replaced. `src` is the Assign's
    `SourceRef`, threaded into the address gates so a synthetic-label
    failure points at the right source line."""
    if isinstance(value, Imm):
        return value.value & 0xff
    if isinstance(value, IndexedAbs):
        return trace.ram[_indexed_addr(value, trace, src)]
    if isinstance(value, IndirectY):
        return trace.ram[_resolve_indirect_y(value, trace, trace.ram)]
    if isinstance(value, Abs):
        return trace.ram[_real_addr(value.addr, src)]
    if isinstance(value, BinExpr):
        # Folded `clc ; adc` / `sec ; sbc`: pure 8-bit add / subtract,
        # result wraps mod 256 (the carry set-up pinned the operation).
        lhs = _assign_read(value.lhs, trace, src)
        rhs = _assign_read(value.rhs, trace, src)
        if value.op == "+":
            return (lhs + rhs) & 0xff
        if value.op == "-":
            return (lhs - rhs) & 0xff
        raise InterpError(f"unknown BinExpr op: {value.op!r}")
    raise InterpError(f"unknown Assign source type: {type(value).__name__}")


def _assign_addr(target, trace: Trace, src) -> int:
    """Resolve the destination address of a pass-3 `Assign`. Mirrors the
    store-side of `exec_atom`. `src` is threaded into the address gates
    for the same diagnostics reason as `_assign_read`."""
    if isinstance(target, IndexedAbs):
        return _indexed_addr(target, trace, src)
    if isinstance(target, IndirectY):
        return _resolve_indirect_y(target, trace, trace.ram)
    if isinstance(target, Abs):
        return _real_addr(target.addr, src)
    raise InterpError(f"unknown Assign target type: {type(target).__name__}")


def _exec_stmt(stmt: Stmt, modules, aliases, trace: Trace) -> None:
    if isinstance(stmt, RawStmt):
        handled = exec_atom(stmt.item, trace, trace.ram)
        if not handled:
            raise InterpError(
                f"IR3 RawStmt wraps a non-atom item: {type(stmt.item).__name__}"
            )
        return
    if isinstance(stmt, Assign):
        # `target = source`, the dropped `lda`/`sta` round-trip. Read the
        # source then write the destination — A and the Z/N flags the
        # original load set are intentionally NOT touched (pass 3 only
        # folds when they're dead).
        value = _assign_read(stmt.source, trace, stmt.src)
        addr = _assign_addr(stmt.target, trace, stmt.src)
        trace.ram[addr] = value
        trace.writes[addr] = value
        return
    if isinstance(stmt, CallStmt):
        callee = _resolve(modules, aliases, stmt.target)
        if callee is None:
            raise InterpError(f"call target {stmt.target!r} not found")
        _exec_routine(callee, modules, aliases, trace)
        return
    if isinstance(stmt, TailCallStmt):
        raise _TailCallSignal(stmt.target)
    if isinstance(stmt, ReturnStmt):
        raise _ReturnSignal()
    if isinstance(stmt, IfStmt):
        if _eval_compare(stmt.cond, trace, trace.ram):
            _exec_block(stmt.then_block, modules, aliases, trace)
        elif stmt.else_block is not None:
            _exec_block(stmt.else_block, modules, aliases, trace)
        return
    if isinstance(stmt, MatchStmt):
        reg_val = {Reg.A: trace.a, Reg.X: trace.x, Reg.Y: trace.y}[stmt.reg]
        for arm in stmt.arms:
            if any((v.value & 0xff) == reg_val for v in arm.values):
                # The arm terminates (return / tail-call / break / ...),
                # so this raises the matching signal; if it somehow falls
                # off, returning here matches the no-match fall-through.
                _exec_block(arm.body, modules, aliases, trace)
                return
        return  # no arm matched — fall through to the next statement
    if isinstance(stmt, RawIfStmt):
        if _branch_taken(stmt.cond, trace):
            _exec_block(stmt.then_block, modules, aliases, trace)
        elif stmt.else_block is not None:
            _exec_block(stmt.else_block, modules, aliases, trace)
        return
    if isinstance(stmt, LoopStmt):
        # Bounded so a busted exit guard doesn't hang the interpreter.
        # 6502 routines we're modelling don't iterate more than ~1024
        # times in practice — a million is room to spare for tests.
        for _ in range(1_000_000):
            try:
                _exec_block(stmt.body, modules, aliases, trace)
            except _ContinueSignal:
                continue
            except _BreakSignal:
                break
            # Body fell off the bottom without break/continue —
            # natural iteration, keep looping.
        else:
            raise InterpError(
                "LoopStmt exceeded 1,000,000 iterations — exit guard bug?"
            )
        return
    if isinstance(stmt, DoWhileStmt):
        for _ in range(1_000_000):
            try:
                _exec_block(stmt.body, modules, aliases, trace)
            except _ContinueSignal:
                # A 6502 back-edge `continue` restarts the body without
                # re-testing the bottom guard.
                continue
            except _BreakSignal:
                break
            # Body fell through — test the continue condition at the
            # bottom, exactly where the original guard sat.
            if not _eval_compare(stmt.cond, trace, trace.ram):
                break
        else:
            raise InterpError(
                "DoWhileStmt exceeded 1,000,000 iterations — exit guard bug?"
            )
        return
    if isinstance(stmt, BreakStmt):
        raise _BreakSignal()
    if isinstance(stmt, ContinueStmt):
        raise _ContinueSignal()
    if isinstance(stmt, (GotoStmt, LabelStmt)):
        # The relooper currently emits these only for routines it
        # couldn't structure. Anything reaching the interpreter is a
        # bug in the structurer — fail loudly so the test surface
        # catches it.
        raise InterpError(
            f"IR3 interp doesn't resolve {type(stmt).__name__} — "
            f"unstructured fragment in this routine?"
        )
    raise InterpError(f"unknown IR3 stmt: {type(stmt).__name__}")
