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

Then, building on that, a counted loop with a clean induction variable
(written only by its step, no `continue`/calls) is promoted to a `for`:

* down-counter — `var = #N (0 <= N < 0x80) ; do { body ; var -= 1 }
  while var >= 0`  ⇒  `for var in (0..=N).rev() { body }`
  (the `ldy #N : … : dey : bpl` shape).
* up-counter — `var = #i ; do { body ; var += 1 } while var != #N`,
  with `i < N`  ⇒  `for var in i..N { body }`
  (the `ldx #i : … : inx : cpx #N : bne` shape).

And the full-wrap busy-wait `var = #INIT ; do { body ; var -= 1 }
while var != #INIT` (exit value equals the start, so the counter cycles
the whole byte range — 256 iterations) is recovered as a fixed-count
`repeat 0x100 { body }`, provided `var` is the counter only. Other
bound conditions (memory bounds, `>=`/`<` floors) are left as
`do`/`loop`.

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

from .ir1 import DecTarget, Imm, IncTarget, LoadImm, Unsupported
from .ir3 import (
    Block,
    BreakStmt,
    CallStmt,
    ContinueStmt,
    DoWhileStmt,
    ForStmt,
    GotoStmt,
    IfStmt,
    LabelStmt,
    LoopStmt,
    MatchStmt,
    ModuleIR3,
    RawIfStmt,
    RawStmt,
    RepeatStmt,
    RoutineIR3,
    Stmt,
    TailCallStmt,
)
from .pass3_expr import _reads_reg, _writes_reg

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


def _body_clobbers_counter(block: Block, reg) -> bool:
    """Could anything in `block` (recursively) change `reg` other than
    the loop's own step? A clean induction counter is written only by
    its `dey`/`dex`. Beyond explicit `RawStmt` writes this conservatively
    flags calls and unmodelled opcodes, which may clobber X/Y without our
    being able to prove otherwise — promoting such a loop to a `for`
    would misrepresent the iteration sequence."""
    for s in block.stmts:
        if isinstance(s, (CallStmt, TailCallStmt)):
            return True  # callee may use X/Y as scratch
        if isinstance(s, RawStmt):
            if isinstance(s.item, Unsupported) or _writes_reg(s.item, reg):
                return True
        for attr in ("then_block", "else_block", "body"):
            inner = getattr(s, attr, None)
            if inner is not None and hasattr(inner, "stmts") \
                    and _body_clobbers_counter(inner, reg):
                return True
        if isinstance(s, MatchStmt):
            for arm in s.arms:
                if _body_clobbers_counter(arm.body, reg):
                    return True
    return False


def _has_continue(block: Block) -> bool:
    for s in block.stmts:
        if isinstance(s, ContinueStmt):
            return True
        for attr in ("then_block", "else_block", "body"):
            inner = getattr(s, attr, None)
            if inner is not None and hasattr(inner, "stmts") and _has_continue(inner):
                return True
        if isinstance(s, MatchStmt):
            for arm in s.arms:
                if _has_continue(arm.body):
                    return True
    return False


def _counter_for(prev: Stmt, dw: DoWhileStmt):
    """If `prev` initialises the counter of do-while `dw` as a clean
    counted loop, return the `ForStmt`; else None. Two shapes:

    * down-counter: `var = #start (0 <= start < 0x80) ;
      do { body ; var -= 1 } while var >= 0`  →  step -1.
    * up-counter:   `var = #start ; do { body ; var += 1 }
      while var != #N`, with `start < N`      →  step +1.

    Both require `start` to satisfy the continue condition (so the loop
    runs at least once and the top-tested `for` matches), `var` written
    only by its step, and no `continue` / call / unmodelled op."""
    if not (isinstance(prev, RawStmt) and isinstance(prev.item, LoadImm)):
        return None
    reg = prev.item.reg
    if dw.cond.reg is not reg:
        return None
    body = dw.body.stmts
    if not body:
        return None
    step = body[-1]
    if not isinstance(step, RawStmt):
        return None
    start = prev.item.imm.value & 0xff

    if dw.cond.op == ">=0":
        # Down-counter to 0 via dey/dex.
        if not (isinstance(step.item, DecTarget) and step.item.target is reg):
            return None
        if start >= 0x80:
            return None  # start must be non-negative so the loop runs >= once
        direction = -1
    elif dw.cond.op == "!=" and isinstance(dw.cond.rhs, Imm):
        # Up-counter to a constant bound via inx/iny.
        if not (isinstance(step.item, IncTarget) and step.item.target is reg):
            return None
        if not (start < (dw.cond.rhs.value & 0xff)):
            return None  # start < N so `start..N` is the exact (non-wrapping) range
        direction = +1
    else:
        return None

    for_body = Block.of(list(body[:-1]))
    if _body_clobbers_counter(for_body, reg) or _has_continue(for_body):
        return None
    return ForStmt(var=reg, start=prev.item.imm, step=direction,
                   cond=dw.cond, body=for_body, src=dw.src)


def _delay_body_reads_reg(block: Block, reg) -> bool:
    """Could anything in `block` read `reg`? A `repeat` exposes no loop
    variable, so a delay body must not depend on the counter at all.
    Beyond explicit `RawStmt` reads this is conservative: any structured
    statement (an `IfStmt`/`MatchStmt` whose condition might inspect the
    counter, an `Assign` that might index by it, a nested loop, a call,
    …) is treated as a read. Only `RawStmt`s that don't read `reg` and
    pure control transfers are accepted."""
    for s in block.stmts:
        if isinstance(s, RawStmt):
            if _reads_reg(s.item, reg):
                return True
            continue
        if isinstance(s, (BreakStmt, ContinueStmt, GotoStmt, LabelStmt)):
            continue  # control transfer — reads no register
        return True  # any other node may read the counter
    return False


def _has_break(block: Block) -> bool:
    for s in block.stmts:
        if isinstance(s, BreakStmt):
            return True
        for attr in ("then_block", "else_block", "body"):
            inner = getattr(s, attr, None)
            if inner is not None and hasattr(inner, "stmts") and _has_break(inner):
                return True
        if isinstance(s, MatchStmt):
            for arm in s.arms:
                if _has_break(arm.body):
                    return True
    return False


def _delay_loop(prev: Stmt, dw: DoWhileStmt):
    """If `prev = var = #INIT` and `dw = do { body ; var ±= 1 } while
    var != #INIT` (the exit value equals the start), the counter wraps
    the full byte range — the body runs exactly 256 times. Return a
    `RepeatStmt` if `var` is the counter only (body never reads/writes
    it, no `break`/`continue`/calls); else None."""
    if not (isinstance(prev, RawStmt) and isinstance(prev.item, LoadImm)):
        return None
    reg = prev.item.reg
    if dw.cond.op != "!=" or dw.cond.reg is not reg \
            or not isinstance(dw.cond.rhs, Imm):
        return None
    if (dw.cond.rhs.value & 0xff) != (prev.item.imm.value & 0xff):
        return None  # exit value must equal the start → full 256-iteration wrap
    body = dw.body.stmts
    if not body:
        return None
    step = body[-1]
    if not isinstance(step, RawStmt):
        return None
    direction = (-1 if isinstance(step.item, DecTarget)
                 else 1 if isinstance(step.item, IncTarget) else None)
    if direction is None or step.item.target is not reg:
        return None
    rest = Block.of(list(body[:-1]))
    if (_body_clobbers_counter(rest, reg) or _delay_body_reads_reg(rest, reg)
            or _has_continue(rest) or _has_break(rest)):
        return None
    return RepeatStmt(count=0x100, var=reg, start=prev.item.imm,
                      step=direction, body=rest, src=dw.src)


def recover_block(block: Block) -> Block:
    out: list[Stmt] = []
    for s in block.stmts:
        s = _recurse(s)
        if isinstance(s, LoopStmt):
            guard = _bottom_exit_guard(s.body)
            if guard is not None:
                s = DoWhileStmt(
                    body=Block.of(list(s.body.stmts[:-1])),
                    cond=_negate(guard.cond),
                    src=s.src,
                )
        if isinstance(s, DoWhileStmt) and out:
            promoted = _counter_for(out[-1], s) or _delay_loop(out[-1], s)
            if promoted is not None:
                out[-1] = promoted  # subsume the preceding init LoadImm
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


def _stat(module: ModuleIR3, node_type) -> int:
    def count(block: Block) -> int:
        total = 0
        for s in block.stmts:
            if isinstance(s, node_type):
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


def dowhile_stats(module: ModuleIR3) -> int:
    """Total `DoWhileStmt`s left across the module (those not promoted to
    a `ForStmt`)."""
    return _stat(module, DoWhileStmt)


def for_stats(module: ModuleIR3) -> int:
    """Total `ForStmt`s (recovered counted loops) across the module."""
    return _stat(module, ForStmt)


def repeat_stats(module: ModuleIR3) -> int:
    """Total `RepeatStmt`s (recovered fixed-count delay loops)."""
    return _stat(module, RepeatStmt)
