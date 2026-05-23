"""Pass 3 (semantic recovery), slice 1: accumulator copy folding.

The structured IR3 from pass 2 still carries the 6502 accumulator
round-trip explicitly:

    a = *(ztemp)[y]          # load
    *(Char + y) = a          # store

Pass 3 collapses that into a direct memory-to-memory assignment,
dropping the intermediate `a`:

    *(Char + y) = *(ztemp)[y]

It also handles the constant-store idiom (`a = #1 ; *CharID = a` ⇒
`CharID = 1`) and runs of stores fed by one load (`a = #0 ; sta X ;
sta Y` ⇒ `X = 0 ; Y = 0`).

Scope (slice 1) — deliberately conservative, behaviour-preserving:

* **Accumulator only.** The X/Y copies (`ldx`/`stx`) are rarer; a
  later slice generalises.
* **Adjacent runs.** A foldable group is a `load A` immediately
  followed by one or more consecutive A-stores. Requiring adjacency
  sidesteps the source-clobbering question — nothing runs between
  the load and the stores, so a memory source can't be overwritten
  before the copy.
* **Dead-after check.** The load is only dropped when `A` is dead
  after the store run: the next thing that touches A writes it
  (reassignment), or — inside a loop body whose first A-touch is a
  write — control wraps back to that write. `Return` / `Call` /
  `TailCall` / conditions / arithmetic all count as A-reads, so a
  value that escapes via a call or return (A might be a return
  value) is never folded away.
* Anything ambiguous is left exactly as pass 2 produced it.

Out of scope (later slices): arithmetic expression trees
(`a = X ; clc ; adc #8 ; sta Y` ⇒ `Y = X + 8`), `match` recognition
from chained `if a == K`, SMC patch → operand-variable.
"""

from __future__ import annotations

from dataclasses import replace

from .ir1 import (
    AdcAbs,
    AdcImm,
    AdcIndexed,
    Asl,
    Bit,
    Bitwise,
    CmpAbs,
    CmpImm,
    CmpIndexed,
    CmpIndirect,
    LoadAbs,
    LoadImm,
    LoadIndexed,
    LoadIndirect,
    Lsr,
    Pha,
    Pla,
    Reg,
    Rol,
    Ror,
    SbcAbs,
    SbcImm,
    SbcIndexed,
    SbcIndirect,
    StoreAbs,
    StoreIndexed,
    StoreIndirect,
    StoreLocal,
    Transfer,
    Unsupported,
)
from .ir3 import (
    Assign,
    Block,
    BreakStmt,
    CallStmt,
    ContinueStmt,
    IfStmt,
    LoopStmt,
    ModuleIR3,
    RawIfStmt,
    RawStmt,
    ReturnStmt,
    RoutineIR3,
    Stmt,
    TailCallStmt,
)


# ---------------------------------------------------------------- A read/write classification


def _writes_a(item) -> bool:
    """True if the IR1 atom assigns a new value to the accumulator."""
    if isinstance(item, (LoadImm, LoadAbs, LoadIndexed, LoadIndirect)):
        return item.reg is Reg.A
    if isinstance(item, Transfer):
        return item.dst_reg is Reg.A
    if isinstance(item, (
        AdcImm, AdcAbs, AdcIndexed, SbcImm, SbcAbs, SbcIndexed, SbcIndirect,
        Bitwise, Asl, Lsr, Rol, Ror, Pla,
    )):
        # Accumulator arithmetic / shifts / pop all redefine A.
        return True
    return False


def _reads_a(item) -> bool:
    """True if the IR1 atom consumes the accumulator's current value.

    `Unsupported` opcodes count as a read: their semantics are unknown,
    so they might read A (or its flags). Treating them as a read makes
    them a hard barrier for liveness — the fold never steps past an
    opcode the lifter hasn't modelled yet."""
    if isinstance(item, Unsupported):
        return True
    if isinstance(item, (StoreAbs, StoreIndexed, StoreIndirect, StoreLocal)):
        return item.reg is Reg.A
    if isinstance(item, Transfer):
        return item.src_reg is Reg.A
    if isinstance(item, (CmpImm, CmpAbs, CmpIndexed, CmpIndirect)):
        return item.reg is Reg.A
    if isinstance(item, (
        AdcImm, AdcAbs, AdcIndexed, SbcImm, SbcAbs, SbcIndexed, SbcIndirect,
        Bitwise, Asl, Lsr, Rol, Ror, Bit, Pha,
    )):
        return True
    return False


def _is_a_load(item):
    """If `item` loads the accumulator from a value pass-3 can fold
    (immediate or a memory read), return that source value; else
    None. Note: `Transfer`/arithmetic write A but their "source"
    isn't a standalone copyable value, so they don't qualify."""
    if isinstance(item, LoadImm) and item.reg is Reg.A:
        return item.imm
    if isinstance(item, LoadAbs) and item.reg is Reg.A:
        return item.source
    if isinstance(item, LoadIndexed) and item.reg is Reg.A:
        from .ir1 import IndexedAbs
        return IndexedAbs(base=item.base, index=item.index)
    if isinstance(item, LoadIndirect) and item.reg is Reg.A:
        return item.source
    return None


def _a_store_target(item):
    """If `item` stores the accumulator to memory, return the store
    target (Abs / IndexedAbs / IndirectY); else None. `StoreLocal`
    (SMC operand patch) is excluded — its target is symbolic and an
    Assign can't represent it."""
    if isinstance(item, StoreAbs) and item.reg is Reg.A:
        return item.target
    if isinstance(item, StoreIndexed) and item.reg is Reg.A:
        from .ir1 import IndexedAbs
        return IndexedAbs(base=item.base, index=item.index)
    if isinstance(item, StoreIndirect) and item.reg is Reg.A:
        return item.target
    return None


# ---------------------------------------------------------------- A liveness over IR3


def _stmt_touches_a(stmt: Stmt) -> bool:
    """Conservative: does this statement (including any nested block)
    read or write A? Used by `_first_a_touch_is_write` to find the
    first A-touch in a loop body. (The forward deadness scan in
    `_a_dead_from` classifies each statement type directly rather than
    relying on this coarse "touches A" summary.)"""
    if isinstance(stmt, RawStmt):
        return _reads_a(stmt.item) or _writes_a(stmt.item)
    if isinstance(stmt, Assign):
        # Folded assigns never touch A (that's the point).
        return False
    if isinstance(stmt, IfStmt):
        if stmt.cond.reg is Reg.A:
            return True
        return _block_touches_a(stmt.then_block) or (
            stmt.else_block is not None and _block_touches_a(stmt.else_block)
        )
    if isinstance(stmt, RawIfStmt):
        # Raw flag-suffix condition doesn't name a register, but its
        # body might touch A.
        return _block_touches_a(stmt.then_block) or (
            stmt.else_block is not None and _block_touches_a(stmt.else_block)
        )
    if isinstance(stmt, LoopStmt):
        return _block_touches_a(stmt.body)
    # Call / TailCall / Return are handled as A-readers by the caller
    # (they might pass/return A); Break/Continue/Goto/Label don't
    # touch A.
    return False


def _block_touches_a(block: Block) -> bool:
    return any(_stmt_touches_a(s) for s in block.stmts)


def _first_a_touch_is_write(block: Block) -> bool:
    """Scanning a block from the top, is the first statement that
    touches A a *write*? If so, the block doesn't depend on the
    incoming A — used to decide that A is dead across a loop's
    back-edge (the next iteration overwrites it before reading)."""
    for s in block.stmts:
        if not _stmt_touches_a(s):
            continue
        if isinstance(s, RawStmt):
            if _writes_a(s.item) and not _reads_a(s.item):
                return True
            return False
        # A nested control-flow stmt touches A in a way we can't
        # cheaply classify — be conservative.
        return False
    return False  # never touches A → incoming A not required, but
    # nothing here proves a write either; safe default is "not a
    # clean write-first", so callers stay conservative.


# ---------------------------------------------------------------- the fold


def _a_dead_from(
    stmts: list[Stmt],
    idx: int,
    *,
    dead_fallthrough: bool,
    dead_break: bool,
    dead_continue: bool,
) -> bool:
    """Is A dead from `stmts[idx:]` onward — i.e. overwritten before
    being read on *every* control-flow path? Path-sensitive: each way
    out of the current statement list carries its own deadness:

    * `dead_fallthrough` — control runs off the end of this list.
    * `dead_break` — a `break` exits the innermost enclosing loop.
    * `dead_continue` — a `continue` re-enters the innermost loop.

    Per-statement rules:

    * `RawStmt` that writes A without reading → dead (True); that reads
      A → live (False); otherwise step past it.
    * `Call`/`TailCall`/`Return` → live (A may be an argument / return
      value).
    * `RawIfStmt` → live: the dropped load set Z/N, which the store run
      leaves intact, so a raw flag-branch may observe them. (An
      `IfStmt` uses a self-contained `Compare`, so its *condition*
      reads no flags — but each of its branches is still scanned.)
    * `IfStmt` whose condition reads A → live; otherwise A is dead past
      it only if dead on *both* the taken branch and the fall-through.
    * A nested `LoopStmt`, `Goto`/`Label`, or any unrecognised stmt →
      live (out of scope for slice 1; never step past it).

    Unlike the earlier `_stmt_touches_a`-based scan, this never steps
    past a compound statement on the strength of "its body doesn't
    touch A" alone — a nested flag-branch or early exit is honoured.
    """
    n = len(stmts)
    i = idx
    while i < n:
        s = stmts[i]
        if isinstance(s, RawStmt):
            if _reads_a(s.item):
                return False
            if _writes_a(s.item):
                return True
            i += 1
            continue
        if isinstance(s, Assign):
            i += 1  # folded copy touches neither A nor flags
            continue
        if isinstance(s, (CallStmt, TailCallStmt, ReturnStmt)):
            return False
        if isinstance(s, BreakStmt):
            return dead_break
        if isinstance(s, ContinueStmt):
            return dead_continue
        if isinstance(s, RawIfStmt):
            return False
        if isinstance(s, IfStmt):
            if s.cond.reg is Reg.A:
                return False
            # Deadness of everything after this `if` (the false edge,
            # and the then-block's own fall-through both land here).
            rest = _a_dead_from(
                stmts, i + 1,
                dead_fallthrough=dead_fallthrough,
                dead_break=dead_break,
                dead_continue=dead_continue,
            )
            then_dead = _a_dead_from(
                list(s.then_block.stmts), 0,
                dead_fallthrough=rest,
                dead_break=dead_break,
                dead_continue=dead_continue,
            )
            if s.else_block is not None:
                else_dead = _a_dead_from(
                    list(s.else_block.stmts), 0,
                    dead_fallthrough=rest,
                    dead_break=dead_break,
                    dead_continue=dead_continue,
                )
            else:
                else_dead = rest
            return then_dead and else_dead
        # LoopStmt / GotoStmt / LabelStmt / anything else: don't step
        # past it.
        return False
    return dead_fallthrough


def _fold_block(
    block: Block,
    *,
    dead_fallthrough: bool,
    dead_break: bool,
    dead_continue: bool,
) -> Block:
    """Fold accumulator copy runs in `block`. The three flags give the
    deadness of A at each edge out of this block — see `_a_dead_from`.
    They're threaded down so a fold inside a loop body knows whether A
    survives the break (post-loop) and back-edge (re-entry) paths."""
    stmts = list(block.stmts)
    out: list[Stmt] = []
    i = 0
    n = len(stmts)
    while i < n:
        stmt = stmts[i]

        # Recurse into nested blocks first. Nested `if` bodies fall
        # through to *after* the `if`; rather than recompute that edge
        # we pass `dead_fallthrough=False` (conservative — only costs
        # missed folds). `break`/`continue` inside the body still target
        # the same enclosing loop, so those flags pass through unchanged.
        if isinstance(stmt, (IfStmt, RawIfStmt)):
            stmt = replace(
                stmt,
                then_block=_fold_block(
                    stmt.then_block,
                    dead_fallthrough=False,
                    dead_break=dead_break,
                    dead_continue=dead_continue,
                ),
                else_block=(
                    _fold_block(
                        stmt.else_block,
                        dead_fallthrough=False,
                        dead_break=dead_break,
                        dead_continue=dead_continue,
                    )
                    if stmt.else_block is not None else None
                ),
            )
        elif isinstance(stmt, LoopStmt):
            # Re-entry (fall-off-bottom and `continue`): the next
            # iteration must overwrite A before reading it. Break: jump
            # to whatever follows the loop in this block.
            reentry_dead = _first_a_touch_is_write(stmt.body)
            post_loop_dead = _a_dead_from(
                stmts, i + 1,
                dead_fallthrough=dead_fallthrough,
                dead_break=dead_break,
                dead_continue=dead_continue,
            )
            stmt = replace(
                stmt,
                body=_fold_block(
                    stmt.body,
                    dead_fallthrough=reentry_dead,
                    dead_break=post_loop_dead,
                    dead_continue=reentry_dead,
                ),
            )

        # Try to start a copy run at a foldable A-load.
        if isinstance(stmt, RawStmt):
            source = _is_a_load(stmt.item)
            if source is not None:
                # Collect the maximal run of consecutive A-stores
                # immediately after the load.
                targets: list[tuple[object, object]] = []  # (target, src)
                j = i + 1
                while j < n and isinstance(stmts[j], RawStmt):
                    tgt = _a_store_target(stmts[j].item)
                    if tgt is None:
                        break
                    targets.append((tgt, stmts[j].item.src))
                    j += 1
                if targets and _a_dead_from(
                    stmts, j,
                    dead_fallthrough=dead_fallthrough,
                    dead_break=dead_break,
                    dead_continue=dead_continue,
                ):
                    # Fold: drop the load, replace each store with an
                    # Assign carrying the load's source.
                    for tgt, ssrc in targets:
                        out.append(Assign(target=tgt, source=source, src=ssrc))
                    i = j
                    continue

        out.append(stmt)
        i += 1
    return Block.of(out)


def fold_routine(routine: RoutineIR3) -> RoutineIR3:
    # The routine's top-level block falls through to `return` /
    # tail-call, where A might be a return value — so A is NOT dead
    # at the end. There's no enclosing loop, so break/continue are
    # dead-false too (they shouldn't appear at this level anyway).
    return replace(
        routine,
        body=_fold_block(
            routine.body,
            dead_fallthrough=False,
            dead_break=False,
            dead_continue=False,
        ),
    )


def fold_module(module: ModuleIR3) -> ModuleIR3:
    return ModuleIR3(
        name=module.name,
        file=module.file,
        routines=[fold_routine(r) for r in module.routines],
    )


def fold_stats(module: ModuleIR3) -> int:
    """Total `Assign` statements produced across the module — the
    headline 'how many copies did pass 3 fold' counter."""
    def count(block: Block) -> int:
        total = 0
        for s in block.stmts:
            if isinstance(s, Assign):
                total += 1
            inner = getattr(s, "then_block", None)
            if inner is not None:
                total += count(inner)
            inner = getattr(s, "else_block", None)
            if inner is not None:
                total += count(inner)
            inner = getattr(s, "body", None)
            if inner is not None and hasattr(inner, "stmts"):
                total += count(inner)
        return total
    return sum(count(r.body) for r in module.routines)
