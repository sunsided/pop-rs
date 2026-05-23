"""Pass 3 (semantic recovery): do-while loop-condition recovery.

The relooper emits every 6502 bottom-tested loop as a bare `loop` with
the exit test as the last statement of the body:

    loop {
        ...body...
        if y < 0 { break }
    }

This pass hoists that trailing guard into the loop header, dropping the
`if … { break }` boilerplate and naming the loop's continue condition
(the negation of the exit test):

    do {
        ...body...
    } while y >= 0

Only the textbook shape is recovered — a `LoopStmt` whose *last*
statement is an `if Compare { break }` with no `else`. The rewrite is
behaviour-preserving (`DoWhileStmt` keeps the original `loop`'s
continue/break semantics; see its docstring) and validated by the
differential interpreter.

This is the foundation for later induction-variable / `for` recovery,
which wants the exit condition exposed in the header rather than buried
at the bottom of the body.

**Ordering.** `DoWhileStmt` is a *late* readability node: the fold
(`pass3_expr`) and `match` recognition (`pass3_match`) walkers don't
model it, so loop recovery must run after them (the `loops` CLI command
runs `reloop → fold → recover_loops`). Re-running fold/match on the
output is unsupported; to keep that from being a silent footgun, the
fold's demand analysis treats any node it doesn't model conservatively
(as a register use) rather than transparently.
"""

from __future__ import annotations

from dataclasses import replace

from .ir3 import (
    Block,
    BreakStmt,
    DoWhileStmt,
    IfStmt,
    LoopStmt,
    MatchStmt,
    ModuleIR3,
    RawIfStmt,
    RoutineIR3,
    Stmt,
)

# Each comparison operator paired with its negation; an exit test of
# `y < 0` becomes a continue condition of `y >= 0`.
_NEGATE = {
    "==": "!=", "!=": "==",
    "<": ">=", ">=": "<",
    "<0": ">=0", ">=0": "<0",
}


def _negate(cond):
    return replace(cond, op=_NEGATE[cond.op])


def _bottom_exit_guard(body: Block):
    """If `body` ends in an `if Compare { break }` (no else, body is
    exactly a single `break`), return that guard `IfStmt`; else None."""
    if not body.stmts:
        return None
    last = body.stmts[-1]
    if (
        isinstance(last, IfStmt)
        and last.else_block is None
        and last.cond.op in _NEGATE
        and len(last.then_block.stmts) == 1
        and isinstance(last.then_block.stmts[0], BreakStmt)
    ):
        return last
    return None


def _recurse(stmt: Stmt) -> Stmt:
    """Recover loops inside a statement's nested blocks before the
    enclosing level inspects it."""
    if isinstance(stmt, (IfStmt, RawIfStmt)):
        return replace(
            stmt,
            then_block=recover_block(stmt.then_block),
            else_block=(
                recover_block(stmt.else_block)
                if stmt.else_block is not None else None
            ),
        )
    if isinstance(stmt, LoopStmt):
        return replace(stmt, body=recover_block(stmt.body))
    if isinstance(stmt, DoWhileStmt):
        return replace(stmt, body=recover_block(stmt.body))
    if isinstance(stmt, MatchStmt):
        return replace(
            stmt,
            arms=tuple(replace(a, body=recover_block(a.body)) for a in stmt.arms),
        )
    return stmt


def recover_block(block: Block) -> Block:
    out: list[Stmt] = []
    for s in block.stmts:
        s = _recurse(s)
        if isinstance(s, LoopStmt):
            guard = _bottom_exit_guard(s.body)
            if guard is not None:
                out.append(DoWhileStmt(
                    body=Block.of(list(s.body.stmts[:-1])),
                    cond=_negate(guard.cond),
                    src=s.src,
                ))
                continue
        out.append(s)
    return Block.of(out)


def recover_routine(routine: RoutineIR3) -> RoutineIR3:
    return replace(routine, body=recover_block(routine.body))


def recover_loops(module: ModuleIR3) -> ModuleIR3:
    return ModuleIR3(
        name=module.name,
        file=module.file,
        routines=[recover_routine(r) for r in module.routines],
    )


def dowhile_stats(module: ModuleIR3) -> int:
    """Total `DoWhileStmt`s recovered across the module."""
    def count(block: Block) -> int:
        total = 0
        for s in block.stmts:
            if isinstance(s, DoWhileStmt):
                total += 1
            if isinstance(s, MatchStmt):
                for arm in s.arms:
                    total += count(arm.body)
            for attr in ("then_block", "else_block", "body"):
                inner = getattr(s, attr, None)
                if inner is not None and hasattr(inner, "stmts"):
                    total += count(inner)
        return total
    return sum(count(r.body) for r in module.routines)
