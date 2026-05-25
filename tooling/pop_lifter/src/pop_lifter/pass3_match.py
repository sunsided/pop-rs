"""Pass 3 (semantic recovery): `match` recognition.

The 6502 jump-table dispatch idiom — `cmp #K1 ; beq h1 ; cmp #K2 ;
beq h2 ; ...` — comes out of the relooper as a run of consecutive
`if reg == Ki { handler_i }` statements, each handler ending in a
terminator (the branch went somewhere). This pass collapses such a run
into a single structured `match`:

    if a == #space    { a = 0 ; return }
    if a == #pillartop { a = 0 ; return }
    if a == #block    { a = 0 ; return }
    ⇒
    match a {
      #space | #pillartop | #block => { a = 0 ; return }
    }

Recognition is deliberately strict so the rewrite is obviously
behaviour-preserving:

* A run is ≥2 *consecutive* `IfStmt`s, all comparing the **same**
  register with `==` against an **immediate**, with `else_block=None`.
* The keys must be **distinct** — then at most one arm matches, so the
  arm order doesn't matter.
* Every arm's body must **terminate** (last statement is a return /
  tail-call / break / continue) — so a matched arm never falls through
  into the next case, and the whole `match` falls through to the
  following statement only when nothing matched (the `if`-chain's exact
  behaviour; the chain's fall-through tail stays put as an implicit
  default). `goto` is excluded: it only occurs in unstructured
  routines, which this transform leaves alone.

Arms with structurally identical bodies are merged into one arm with
several keys (`K1 | K2 => ...`) — common because the asm shares one
handler label across keys, which the relooper tail-duplicates.

**Shared-handler dispatches.** A second shape: several keys branch to
*one* handler via a forward branch (`cmp Ki ; beq shared`), which the
relooper dedups into a nested negated chain followed by the shared tail
emitted once:

    if a != space { if a != block { if a < archtop1 { a = 1 ; return } } }
    a = 0 ; return                       ; shared tail T
    ⇒
    match a {
      #space | #block => { a = 0 ; return }    ; T, re-duplicated
    }
    if a < archtop1 { a = 1 ; return }   ; the innermost `rest`
    a = 0 ; return                       ; T re-appended (the implicit default)

Recognised when a run of ≥2 nested `if reg != Ki { ... }` (same reg,
distinct immediate keys, no else) wraps an innermost `rest`, and the
statements after the chain (the tail `T`) terminate — so the `match`
arm, which gets a copy of `T`, can't fall through. This *re-duplicates*
`T`, so it's restricted to terminating tails (small `return`/`tail_call`
bodies), not the exponential subtrees the relooper dedup removed.
"""

from __future__ import annotations

from dataclasses import replace

from .ir1 import Imm
from .ir3 import (
    Block,
    BreakStmt,
    ContinueStmt,
    IfStmt,
    LoopStmt,
    MatchArm,
    MatchStmt,
    ModuleIR3,
    RawIfStmt,
    ReturnStmt,
    RoutineIR3,
    Stmt,
    TailCallStmt,
)


def _terminates(block: Block) -> bool:
    """Does `block` end in a structured control transfer? Then a matched
    arm can't fall through into the next case. `GotoStmt` is *not*
    accepted: it only appears in routines the relooper couldn't
    structure (which the IR3 interpreter rejects anyway), so match
    recognition stays within fully-structured, interpretable code."""
    return bool(block.stmts) and isinstance(
        block.stmts[-1],
        (ReturnStmt, TailCallStmt, BreakStmt, ContinueStmt),
    )


def _dispatch_if(stmt: Stmt):
    """If `stmt` is a `reg == #imm` dispatch arm — an `IfStmt` with
    `==`, an immediate rhs, no else, and a terminating body — return
    `(reg, imm)`; else None."""
    if not isinstance(stmt, IfStmt):
        return None
    cond = stmt.cond
    if cond.op != "==" or not isinstance(cond.rhs, Imm) or stmt.else_block is not None:
        return None
    if not _terminates(stmt.then_block):
        return None
    return (cond.reg, cond.rhs)


def _negated_if(stmt: Stmt):
    """If `stmt` is a `reg != #imm` chain link — an `IfStmt` with `!=`, an
    immediate rhs, and no else — return `(reg, imm)`; else None. (Unlike
    `_dispatch_if`, the body needn't terminate: it holds the next chain
    link or the innermost `rest`.)"""
    if not isinstance(stmt, IfStmt):
        return None
    cond = stmt.cond
    if cond.op != "!=" or not isinstance(cond.rhs, Imm) or stmt.else_block is not None:
        return None
    return (cond.reg, cond.rhs)


def _shared_handler_chain(head: IfStmt, tail: list[Stmt]):
    """Recognise a shared-handler dispatch headed by `head` (a `reg != K`
    chain) whose enclosing-block tail is `tail`. Returns
    `(reg, keys, rest)` — the dispatch register, the distinct keys, and
    the innermost body — or None if `head`/`tail` don't form the shape.

    `tail` is the shared handler `T`; it must terminate so the rewritten
    `match` arm (a copy of `T`) can't fall through into the default."""
    first = _negated_if(head)
    if first is None or not tail or not _terminates(Block.of(tail)):
        return None
    reg, k0 = first
    keys = [k0]
    seen = {k0.value}
    cur = head
    while True:
        body = cur.then_block.stmts
        if len(body) == 1:
            link = _negated_if(body[0])
            if link is not None and link[0] is reg and link[1].value not in seen:
                cur = body[0]
                keys.append(link[1])
                seen.add(link[1].value)
                continue
        break
    if len(keys) < 2:
        return None
    return reg, keys, list(cur.then_block.stmts)


def _recurse(stmt: Stmt) -> Stmt:
    """Rewrite nested blocks of a single statement before scanning the
    current level (so arm bodies are themselves match-recognised)."""
    if isinstance(stmt, (IfStmt, RawIfStmt)):
        return replace(
            stmt,
            then_block=recognize_block(stmt.then_block),
            else_block=(
                recognize_block(stmt.else_block)
                if stmt.else_block is not None else None
            ),
        )
    if isinstance(stmt, LoopStmt):
        return replace(stmt, body=recognize_block(stmt.body))
    return stmt


def _build_match(reg, run: list[IfStmt]) -> MatchStmt:
    # Group arms by structurally identical body, preserving first-seen
    # order, so several keys sharing a handler become one `K1 | K2` arm.
    groups: list[tuple[Block, list[Imm]]] = []
    for ifs in run:
        body = ifs.then_block
        for g_body, keys in groups:
            if g_body == body:
                keys.append(ifs.cond.rhs)
                break
        else:
            groups.append((body, [ifs.cond.rhs]))
    arms = tuple(MatchArm(values=tuple(keys), body=body) for body, keys in groups)
    return MatchStmt(reg=reg, arms=arms, src=run[0].src)


def recognize_block(block: Block) -> Block:
    rec = [_recurse(s) for s in block.stmts]
    out: list[Stmt] = []
    i = 0
    n = len(rec)
    while i < n:
        info = _dispatch_if(rec[i])
        if info is not None:
            reg, first_key = info
            run = [rec[i]]
            keys = {first_key.value}
            j = i + 1
            while j < n:
                nxt = _dispatch_if(rec[j])
                if nxt is None or nxt[0] is not reg or nxt[1].value in keys:
                    break
                run.append(rec[j])
                keys.add(nxt[1].value)
                j += 1
            if len(run) >= 2:
                out.append(_build_match(reg, run))
                i = j
                continue

        # Shared-handler dispatch: a `reg != Ki` chain at `i` whose tail
        # (the rest of this block) is the shared terminating handler `T`.
        chain = _shared_handler_chain(rec[i], rec[i + 1:])
        if chain is not None:
            reg, keys, rest = chain
            tail = rec[i + 1:]
            out.append(MatchStmt(
                reg=reg,
                arms=(MatchArm(values=tuple(keys), body=Block.of(tail)),),
                src=rec[i].src,
            ))
            # Default (no key matched): the innermost `rest`, then `T`
            # re-appended — but only if `rest` can fall through. When
            # `rest` already terminates unconditionally, the original
            # never reached `T` either, so re-appending it would just be
            # dead code after a `return` (an `unreachable_code` warning).
            # Re-recognise so nested dispatches still fold.
            default = rest if _terminates(Block.of(rest)) else rest + tail
            out.extend(recognize_block(Block.of(default)).stmts)
            i = n
            continue

        out.append(rec[i])
        i += 1
    return Block.of(out)


def recognize_routine(routine: RoutineIR3) -> RoutineIR3:
    return replace(routine, body=recognize_block(routine.body))


def recognize_module(module: ModuleIR3) -> ModuleIR3:
    return ModuleIR3(
        name=module.name,
        file=module.file,
        routines=[recognize_routine(r) for r in module.routines],
    )


def match_stats(module: ModuleIR3) -> int:
    """Total `MatchStmt`s recognised across the module."""
    def count(block: Block) -> int:
        total = 0
        for s in block.stmts:
            if isinstance(s, MatchStmt):
                total += 1
                for arm in s.arms:
                    total += count(arm.body)
            for attr in ("then_block", "else_block", "body"):
                inner = getattr(s, attr, None)
                if inner is not None and hasattr(inner, "stmts"):
                    total += count(inner)
        return total
    return sum(count(r.body) for r in module.routines)
