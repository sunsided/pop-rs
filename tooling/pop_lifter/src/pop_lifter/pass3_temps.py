"""Pass 3 (semantic recovery): pha/pla scoped-temporary recovery.

The 6502 saves a register across a clobbering region by bracketing it
with a stack push/pop:

    push a            ; pha — stash A
    call cut          ; the callee may trash A
    a = pop           ; pla — A is back to its pre-call value

and the swap idiom uses the saved byte directly after restoring it:

    push a            ; save the value to relocate
    a = *(sortX-1 + x)
    *(sortX + x) = a
    a = pop           ; bring the saved value back
    *(sortX-1 + x) = a

This pass matches each `pha` with the `pla` that pops it and rewrites
the pair into a named scoped temporary:

    tmp0 = a          ; [save]
    call cut
    a = tmp0          ; [restore]

so the stack churn reads as an ordinary local — the shape pass 4 emits
as `let tmp0 = a; …; a = tmp0;`.

**Matching.**  Within one flat block, `pha`/`pla` nest like brackets:
a stack of open `pha` indices, each `pla` pops the most recent. Two
rules keep the pairing honest:

* **Stack-neutral statements are spanned.** Plain loads/stores/arith
  (`RawStmt`) and a non-tail `CallStmt` leave the stack as they found
  it (a callee balances its own pha/pla — see `ir1.Pha`), so a pair may
  bracket them. `Assign` / `Wide16Stmt` (folded memory writes) and an
  already-recovered `SaveTemp`/`RestoreTemp` are likewise neutral.
* **Control flow is a barrier.** A `Return` / `TailCall` / `If` / loop /
  `match` / `goto` / `break` / `continue`, or an `Unsupported` opcode
  (which might itself touch the stack, e.g. `php`), resets the open
  stack: a `pha` before it won't be matched with a `pla` after it. This
  is conservative — it only forgoes matches, never makes a wrong one.

`pha`/`pla` inside a nested block are matched within that block (the
recursion runs first); an unmatched `pha` or `pla` is left as the raw
op.

**Soundness.**  `slot` is a per-routine id used only to *name* the
temporary. The value still rides the single shared value stack
(`Trace.value_stack`), so the IR3 interpreter runs `SaveTemp`/
`RestoreTemp` as the exact `pha`/`pla` they replaced — behaviour is
preserved no matter how slots were numbered (and slots can't collide
across call frames the way a slot-keyed store would). The differential
interpreter test pins this.

**Ordering.**  Recovery runs last in the pass-3 chain (after fold and
loop recovery): it only relabels `pha`/`pla`, which those passes leave
untouched, and emits nodes they don't model.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import count
from typing import Iterator

from .ir1 import Pha, Phy, Pla, Unsupported
from .ir3 import (
    Assign,
    Block,
    CallStmt,
    DoWhileStmt,
    ForStmt,
    IfStmt,
    LoopStmt,
    MatchStmt,
    ModuleIR3,
    RawIfStmt,
    RawStmt,
    RepeatStmt,
    RestoreTemp,
    RoutineIR3,
    SaveTemp,
    Stmt,
    Wide16Stmt,
)


def _is_pha(s: Stmt) -> bool:
    return isinstance(s, RawStmt) and isinstance(s.item, Pha)


def _is_pla(s: Stmt) -> bool:
    return isinstance(s, RawStmt) and isinstance(s.item, Pla)


def _is_stack_barrier(s: Stmt) -> bool:
    """Does `s` break the straight-line region a pha/pla pair may span?
    Stack-neutral statements (plain ops, balanced calls, folded memory
    writes, already-recovered temps) return False; anything that
    transfers control — or an `Unsupported` op that might touch the
    stack itself — returns True."""
    if isinstance(s, RawStmt):
        # pha/pla are the bracket tokens, handled before this check.
        # A plain modelled op leaves the byte stack alone, so it's
        # spanned. But anything else that mutates the byte stack is a
        # barrier: a `pha` before it can't be soundly paired with a
        # `pla` after it, because that `pla` pops the *intervening*
        # value, not the saved A. That covers `Phy` (65C02 push-Y, e.g.
        # the `pha ; phy ; … ; pla` in `getparam`) and any `Unsupported`
        # op that might touch the stack (e.g. php/plp lowered as unknown).
        return isinstance(s.item, (Phy, Unsupported))
    if isinstance(s, (Assign, Wide16Stmt, SaveTemp, RestoreTemp, CallStmt)):
        return False
    # Return / TailCall / If / RawIf / Loop / DoWhile / For / Repeat /
    # Match / Goto / Label / Break / Continue — control leaves this
    # straight line, so a pha before can't be soundly paired past it.
    return True


def _recurse(stmt: Stmt, slots: Iterator[int]) -> Stmt:
    """Recover pairs inside a statement's nested blocks first, threading
    the routine-wide slot counter so every temporary is uniquely named."""
    if isinstance(stmt, (IfStmt, RawIfStmt)):
        return replace(
            stmt,
            then_block=_recover_block(stmt.then_block, slots),
            else_block=(
                _recover_block(stmt.else_block, slots)
                if stmt.else_block is not None else None
            ),
        )
    if isinstance(stmt, (LoopStmt, DoWhileStmt, ForStmt, RepeatStmt)):
        return replace(stmt, body=_recover_block(stmt.body, slots))
    if isinstance(stmt, MatchStmt):
        return replace(
            stmt,
            arms=tuple(
                replace(a, body=_recover_block(a.body, slots)) for a in stmt.arms
            ),
        )
    return stmt


def _recover_block(block: Block, slots: Iterator[int]) -> Block:
    stmts: list[Stmt] = [_recurse(s, slots) for s in block.stmts]
    open_pha: list[int] = []  # indices of unmatched `pha`, innermost last
    for idx, s in enumerate(stmts):
        if _is_pha(s):
            open_pha.append(idx)
        elif _is_pla(s):
            if open_pha:
                i = open_pha.pop()
                slot = next(slots)
                stmts[i] = SaveTemp(slot=slot, src=stmts[i].item.src)
                stmts[idx] = RestoreTemp(slot=slot, src=s.item.src)
            # else: unmatched pla (pops a byte pushed before a barrier /
            # the block start) — leave it as the raw op.
        elif _is_stack_barrier(s):
            open_pha.clear()  # can't pair a pha across a control transfer
    return Block.of(stmts)


def recover_routine(routine: RoutineIR3) -> RoutineIR3:
    return replace(routine, body=_recover_block(routine.body, count()))


def recover_temps(module: ModuleIR3) -> ModuleIR3:
    return ModuleIR3(
        name=module.name,
        file=module.file,
        routines=[recover_routine(r) for r in module.routines],
    )


def temp_stats(module: ModuleIR3) -> int:
    """Total scoped temporaries recovered (matched `pha`/`pla` pairs) —
    one `SaveTemp` per pair."""
    def count_block(block: Block) -> int:
        total = 0
        for s in block.stmts:
            if isinstance(s, SaveTemp):
                total += 1
            if isinstance(s, MatchStmt):
                for arm in s.arms:
                    total += count_block(arm.body)
            for attr in ("then_block", "else_block", "body"):
                inner = getattr(s, attr, None)
                if inner is not None and hasattr(inner, "stmts"):
                    total += count_block(inner)
        return total
    return sum(count_block(r.body) for r in module.routines)
