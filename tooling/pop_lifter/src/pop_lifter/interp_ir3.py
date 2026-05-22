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
    _resolve,
    exec_atom,
)
from .interp_ir1 import run as ir1_run
from .ir1 import ModuleIR1, Routine as IR1Routine
from .ir3 import (
    Block,
    CallStmt,
    GotoStmt,
    IfStmt,
    LabelStmt,
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


def _exec_stmt(stmt: Stmt, modules, aliases, trace: Trace) -> None:
    if isinstance(stmt, RawStmt):
        handled = exec_atom(stmt.item, trace, trace.ram)
        if not handled:
            raise InterpError(
                f"IR3 RawStmt wraps a non-atom item: {type(stmt.item).__name__}"
            )
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
    if isinstance(stmt, RawIfStmt):
        if _branch_taken(stmt.cond, trace):
            _exec_block(stmt.then_block, modules, aliases, trace)
        elif stmt.else_block is not None:
            _exec_block(stmt.else_block, modules, aliases, trace)
        return
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
