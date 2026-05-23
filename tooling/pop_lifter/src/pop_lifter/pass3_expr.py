"""Pass 3 (semantic recovery): accumulator expression folding.

The structured IR3 from pass 2 still carries the 6502 accumulator
round-trip explicitly:

    a = *(ztemp)[y]          # load
    *(Char + y) = a          # store

Pass 3 collapses that into a direct memory-to-memory assignment,
dropping the intermediate `a`:

    *(Char + y) = *(ztemp)[y]

**Slice 1 — copy folding.** The constant-store idiom
(`a = #1 ; *CharID = a` ⇒ `CharID = 1`) and runs of stores fed by one
load (`a = #0 ; sta X ; sta Y` ⇒ `X = 0 ; Y = 0`).

**Slice 2 — arithmetic folding.** The compute-and-store idiom collapses
the load + carry set-up + add/subtract into a `BinExpr`:

    a = X ; clc ; adc #8 ; sta Y   ⇒   Y = X + 8
    a = X ; sec ; sbc #8 ; sta Y   ⇒   Y = X - 8

The explicit `clc`/`sec` pins the op to a pure 8-bit add / subtract.

Deliberately conservative, behaviour-preserving:

* **Accumulator only.** The X/Y copies (`ldx`/`stx`) are rarer; a
  later slice generalises.
* **Adjacent runs.** A foldable group is a `load A` immediately
  followed by its store(s) (with the `clc`/`sec` + `adc`/`sbc` in
  between for the arithmetic form). Requiring adjacency sidesteps the
  source-clobbering question — nothing runs between the load and the
  stores, so a memory operand can't be overwritten before the copy.
* **Dead-after check.** The load is only dropped when `A` is dead
  after the store run: the next thing that touches A writes it
  (reassignment), or — inside a loop body whose first A-touch is a
  write — control wraps back to that write. For an arithmetic fold the
  **carry** the add/sub set must *also* be dead (so a 16-bit add's
  high-byte `adc` is never folded — its carry feeds the next op). A
  single store only for the arithmetic form (the stored value isn't
  the source, so a multi-store run isn't an idempotent write-back).
  `Return` / `Call` / `TailCall` count as reads, so a value escaping
  via a call or return is never folded away.
* Anything ambiguous is left exactly as pass 2 produced it.

Out of scope (later slices): expression trees deeper than one op,
`match` recognition from chained `if a == K`, SMC patch →
operand-variable.
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
    Clc,
    CmpAbs,
    CmpImm,
    CmpIndexed,
    CmpIndirect,
    IndexedAbs,
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
    Sec,
    ShiftMem,
    StoreAbs,
    StoreIndexed,
    StoreIndirect,
    StoreLocal,
    Transfer,
    Unsupported,
)
from .ir3 import (
    Assign,
    BinExpr,
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
        return IndexedAbs(base=item.base, index=item.index)
    if isinstance(item, StoreIndirect) and item.reg is Reg.A:
        return item.target
    return None


def _store_run(stmts: list[Stmt], start: int):
    """Maximal run of consecutive A-stores beginning at `stmts[start]`,
    returned as `(list_of_(target, src), index_after_run)`."""
    run: list[tuple[object, object]] = []
    k = start
    n = len(stmts)
    while k < n and isinstance(stmts[k], RawStmt):
        tgt = _a_store_target(stmts[k].item)
        if tgt is None:
            break
        run.append((tgt, stmts[k].item.src))
        k += 1
    return run, k


# ---------------------------------------------------------------- carry read/write classification


def _reads_c(item) -> bool:
    """True if the IR1 atom consumes the carry flag. `adc`/`sbc` add it
    in; `rol`/`ror` (and their memory forms) rotate it through. Unknown
    opcodes count as a read so liveness never steps past them."""
    if isinstance(item, Unsupported):
        return True
    if isinstance(item, (
        AdcImm, AdcAbs, AdcIndexed, SbcImm, SbcAbs, SbcIndexed, SbcIndirect,
        Rol, Ror,
    )):
        return True
    if isinstance(item, ShiftMem):
        return item.op in ("rol", "ror")
    return False


def _writes_c(item) -> bool:
    """True if the IR1 atom redefines the carry flag without first
    depending on it (so it kills any previous carry). `clc`/`sec` set
    it; `cmp` and `asl`/`lsr` derive it fresh. (`adc`/`sbc`/`rol`/`ror`
    also write carry but *read* it first — they're handled as reads,
    which is the conservative call for liveness.)"""
    if isinstance(item, (Clc, Sec)):
        return True
    if isinstance(item, (CmpImm, CmpAbs, CmpIndexed, CmpIndirect)):
        return True
    if isinstance(item, (Asl, Lsr)):
        return True
    if isinstance(item, ShiftMem):
        return item.op in ("asl", "lsr")
    return False


# ---------------------------------------------------------------- arithmetic pattern


def _arith_operand(item):
    """Extract the add/sub operand value (`Imm` / `Abs` / `IndexedAbs` /
    `IndirectY`) from an `adc`/`sbc` atom."""
    if isinstance(item, (AdcImm, SbcImm)):
        return item.imm
    if isinstance(item, (AdcAbs, SbcAbs)):
        return item.source
    if isinstance(item, (AdcIndexed, SbcIndexed)):
        return IndexedAbs(base=item.base, index=item.index)
    if isinstance(item, SbcIndirect):
        return item.source
    return None


def _arith_after_load(stmts: list[Stmt], i: int):
    """If the load at `stmts[i]` is immediately followed by a carry
    set-up + add/subtract — `clc ; adc OP` or `sec ; sbc OP` — return
    `(op, operand, store_start_idx)`; else None.

    Requiring the explicit `clc`/`sec` pins the operation to a pure
    8-bit add / subtract (no dependence on the incoming carry), so the
    folded `BinExpr` is exact. A bare `adc`/`sbc` with no preceding
    carry set-up (e.g. the high byte of a 16-bit add) is left alone."""
    if i + 2 >= len(stmts):
        return None
    s1, s2 = stmts[i + 1], stmts[i + 2]
    if not (isinstance(s1, RawStmt) and isinstance(s2, RawStmt)):
        return None
    setup, arith = s1.item, s2.item
    if isinstance(setup, Clc) and isinstance(arith, (AdcImm, AdcAbs, AdcIndexed)):
        return ("+", _arith_operand(arith), i + 3)
    if isinstance(setup, Sec) and isinstance(
        arith, (SbcImm, SbcAbs, SbcIndexed, SbcIndirect)
    ):
        return ("-", _arith_operand(arith), i + 3)
    return None


def _cond_reads_a(cond) -> bool:
    return cond.reg is Reg.A


def _cond_reads_c(cond) -> bool:
    # An IR3 `Compare` re-derives `reg vs rhs` from registers — it never
    # reads the carry flag. (Carry branches that weren't fused to a
    # register compare stay as `RawIfStmt`, which the scan treats as a
    # hard barrier.)
    return False


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


def _dead_from(
    stmts: list[Stmt],
    idx: int,
    *,
    reads,
    writes,
    cond_reads,
    dead_fallthrough: bool,
    dead_break: bool,
    dead_continue: bool,
) -> bool:
    """Is the tracked resource (A, or the carry flag) dead from
    `stmts[idx:]` onward — i.e. overwritten before being read on
    *every* control-flow path?

    `reads` / `writes` classify a `RawStmt`'s atom; `cond_reads` says
    whether an `IfStmt`'s `Compare` consumes the resource. The scan is
    path-sensitive: each way out of the current statement list carries
    its own deadness:

    * `dead_fallthrough` — control runs off the end of this list.
    * `dead_break` — a `break` exits the innermost enclosing loop.
    * `dead_continue` — a `continue` re-enters the innermost loop.

    Per-statement rules:

    * `RawStmt` that `reads` the resource → live (False); that `writes`
      it (without reading) → dead (True); otherwise step past it.
    * `Call`/`TailCall`/`Return` → live (the resource may be an
      argument / return value).
    * `RawIfStmt` → live: it reads raw 6502 flags, which the dropped
      op may have set (Z/N for a load, plus carry for `adc`/`sbc`).
    * `IfStmt` whose condition reads the resource → live; otherwise
      dead past it only if dead on *both* the taken branch and the
      fall-through.
    * A nested `LoopStmt`, `Goto`/`Label`, or any unrecognised stmt →
      live (never step past it).

    This never steps past a compound statement on the strength of "its
    body doesn't touch the resource" alone — a nested flag-branch or
    early exit is honoured.
    """
    n = len(stmts)
    i = idx
    while i < n:
        s = stmts[i]
        if isinstance(s, RawStmt):
            if reads(s.item):
                return False
            if writes(s.item):
                return True
            i += 1
            continue
        if isinstance(s, Assign):
            i += 1  # a folded copy / expr touches neither A nor flags
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
            if cond_reads(s.cond):
                return False
            # Deadness of everything after this `if` (the false edge,
            # and the then-block's own fall-through both land here).
            kw = dict(reads=reads, writes=writes, cond_reads=cond_reads,
                      dead_break=dead_break, dead_continue=dead_continue)
            rest = _dead_from(stmts, i + 1, dead_fallthrough=dead_fallthrough, **kw)
            then_dead = _dead_from(
                list(s.then_block.stmts), 0, dead_fallthrough=rest, **kw
            )
            if s.else_block is not None:
                else_dead = _dead_from(
                    list(s.else_block.stmts), 0, dead_fallthrough=rest, **kw
                )
            else:
                else_dead = rest
            return then_dead and else_dead
        # LoopStmt / GotoStmt / LabelStmt / anything else: don't step
        # past it.
        return False
    return dead_fallthrough


def _a_dead_from(stmts, idx, *, dead_fallthrough, dead_break, dead_continue) -> bool:
    """A-liveness wrapper over `_dead_from`."""
    return _dead_from(
        stmts, idx,
        reads=_reads_a, writes=_writes_a, cond_reads=_cond_reads_a,
        dead_fallthrough=dead_fallthrough,
        dead_break=dead_break, dead_continue=dead_continue,
    )


def _c_dead_from(stmts, idx, *, dead_fallthrough, dead_break, dead_continue) -> bool:
    """Carry-liveness wrapper over `_dead_from`."""
    return _dead_from(
        stmts, idx,
        reads=_reads_c, writes=_writes_c, cond_reads=_cond_reads_c,
        dead_fallthrough=dead_fallthrough,
        dead_break=dead_break, dead_continue=dead_continue,
    )


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

        # Try to start a fold run at a foldable A-load.
        if isinstance(stmt, RawStmt):
            source = _is_a_load(stmt.item)
            if source is not None:
                edge = dict(
                    dead_fallthrough=dead_fallthrough,
                    dead_break=dead_break,
                    dead_continue=dead_continue,
                )

                # (1) Arithmetic fold: load ; clc/sec ; adc/sbc ; sta.
                # Single store only — the stored value isn't the source
                # value, so a multi-store run isn't an idempotent
                # write-back and could clobber the operand. A *and* the
                # carry the add/sub set must be dead afterwards.
                arith = _arith_after_load(stmts, i)
                if arith is not None and arith[1] is not None:
                    op, operand, store_start = arith
                    targets, j = _store_run(stmts, store_start)
                    if (
                        len(targets) == 1
                        and _a_dead_from(stmts, j, **edge)
                        and _c_dead_from(stmts, j, **edge)
                    ):
                        tgt, ssrc = targets[0]
                        out.append(Assign(
                            target=tgt,
                            source=BinExpr(op=op, lhs=source, rhs=operand),
                            src=ssrc,
                        ))
                        i = j
                        continue

                # (2) Pure copy fold: load ; sta [; sta ...]. A run of
                # stores fed by one load; sound for multiple stores
                # because each writes the (unchanged) source value.
                targets, j = _store_run(stmts, i + 1)
                if targets and _a_dead_from(stmts, j, **edge):
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
