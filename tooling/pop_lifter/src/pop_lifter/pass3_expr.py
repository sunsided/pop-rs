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
)
from .ir3 import (
    Assign,
    Block,
    IfStmt,
    LoopStmt,
    ModuleIR3,
    RawIfStmt,
    RawStmt,
    RoutineIR3,
    Stmt,
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
    """True if the IR1 atom consumes the accumulator's current value."""
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
    read or write A? Used to decide whether a forward scan can safely
    step past a statement when proving A dead."""
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


def _a_dead_after(stmts: list[Stmt], idx: int, *, dead_at_end: bool) -> bool:
    """Starting just after the store run (index `idx`), is A dead —
    i.e. overwritten before being read?

    * First statement that *writes* A (without reading) → dead.
    * First statement that *reads* A (store, arithmetic, condition,
      Call/TailCall/Return) → live.
    * A `RawIfStmt` → live. The dropped load sets Z/N, which the
      store run leaves untouched, so a raw flag-branch here could be
      reading the load's flags. (`IfStmt` uses a self-contained
      `Compare` and re-derives from registers, so it's exempt.)
    * A nested block that touches A → conservatively live.
    * Reached the end of the list → `dead_at_end` (True only for a
      loop body whose first A-touch is a write).
    """
    from .ir3 import CallStmt, ReturnStmt, TailCallStmt

    for s in stmts[idx:]:
        if isinstance(s, RawStmt):
            if _reads_a(s.item):
                return False
            if _writes_a(s.item):
                return True
            continue
        if isinstance(s, (CallStmt, TailCallStmt, ReturnStmt)):
            # A may be an argument / return value — treat as a read.
            return False
        if isinstance(s, RawIfStmt):
            # Raw flag-suffix branch (eq/ne/pl/mi/...) may observe the
            # Z/N the dropped load set — conservatively live.
            return False
        if isinstance(s, Assign):
            continue  # touches neither A nor flags
        # IfStmt / RawIfStmt / LoopStmt / Break / Continue / etc.
        if _stmt_touches_a(s):
            return False
        # Doesn't touch A — keep scanning (e.g. `if y < 0 { break }`).
        continue
    return dead_at_end


def _fold_block(block: Block, *, dead_at_end: bool) -> Block:
    """Fold accumulator copy runs in `block`. `dead_at_end` says
    whether A is dead when control falls off the end of this block
    (True for a loop body that overwrites A before reading it)."""
    stmts = list(block.stmts)
    out: list[Stmt] = []
    i = 0
    n = len(stmts)
    while i < n:
        stmt = stmts[i]

        # Recurse into nested blocks first.
        if isinstance(stmt, IfStmt):
            stmt = replace(
                stmt,
                then_block=_fold_block(stmt.then_block, dead_at_end=False),
                else_block=(
                    _fold_block(stmt.else_block, dead_at_end=False)
                    if stmt.else_block is not None else None
                ),
            )
        elif isinstance(stmt, RawIfStmt):
            stmt = replace(
                stmt,
                then_block=_fold_block(stmt.then_block, dead_at_end=False),
                else_block=(
                    _fold_block(stmt.else_block, dead_at_end=False)
                    if stmt.else_block is not None else None
                ),
            )
        elif isinstance(stmt, LoopStmt):
            body_dead_at_end = _first_a_touch_is_write(stmt.body)
            stmt = replace(
                stmt,
                body=_fold_block(stmt.body, dead_at_end=body_dead_at_end),
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
                if targets and _a_dead_after(stmts, j, dead_at_end=dead_at_end):
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
    # at the end. (Folds there happen only via an explicit
    # reassignment, never the fall-off-the-end rule.)
    return replace(routine, body=_fold_block(routine.body, dead_at_end=False))


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
