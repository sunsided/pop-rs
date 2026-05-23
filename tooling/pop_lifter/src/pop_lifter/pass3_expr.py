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

**Slice 3 — X/Y + interprocedural demand.** Copy folding generalises
from the accumulator to all three registers (`ldx SRC ; stx DST` ⇒
`DST = SRC`). And a register-demand call-graph fixed-point computes
which registers each routine reads before writing, so a `sta DST ; jmp
R` folds when `R` doesn't read the stored register — instead of
conservatively assuming every call/tail-call reads everything. (In POP
this mostly helps the accumulator: X/Y are pervasively live as index
registers, so X/Y copies rarely become dead.)

Deliberately conservative, behaviour-preserving:

* **All three registers** for the copy form; arithmetic stays
  accumulator-only (`adc`/`sbc` are A-only on the 6502).
* **Adjacent runs.** A foldable group is a register load immediately
  followed by its store(s) (with the `clc`/`sec` + `adc`/`sbc` in
  between for the arithmetic form). Requiring adjacency sidesteps the
  source-clobbering question — nothing runs between the load and the
  stores, so a memory operand can't be overwritten before the copy.
* **Dead-after check.** The load is only dropped when the register is
  dead after the store run: the next thing that touches it writes it
  (reassignment), or — inside a loop body whose first touch is a
  write — control wraps back to that write. X and Y are also read as
  *index* registers (`tbl,x` / `(zp),y`), which counts. For an
  arithmetic fold the **carry** the add/sub set must *also* be dead (so
  a 16-bit add's high-byte `adc` is never folded — its carry feeds the
  next op). A single store only for the arithmetic form (the stored
  value isn't the source, so a multi-store run isn't an idempotent
  write-back). A `Return` reads every register (a caller may read the
  result); a `Call`/`TailCall` reads only the registers its target
  demands.
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
    DecTarget,
    IncTarget,
    IndexedAbs,
    IndirectY,
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
    GotoStmt,
    IfStmt,
    LabelStmt,
    LoopStmt,
    ModuleIR3,
    RawIfStmt,
    RawStmt,
    ReturnStmt,
    RoutineIR3,
    Stmt,
    TailCallStmt,
)


# ---------------------------------------------------------------- resources


# A "resource" is something the fold must prove dead before it can drop
# a load: one of the three registers, or the carry flag (carry only
# matters for the arithmetic `clc ; adc` / `sec ; sbc` fold).
_REGS = (Reg.A, Reg.X, Reg.Y)
_CARRY = "carry"
_RESOURCES = (*_REGS, _CARRY)


# ---------------------------------------------------------------- register read/write classification


def _writes_reg(item, reg) -> bool:
    """True if the IR1 atom assigns a new value to register `reg`."""
    if isinstance(item, (LoadImm, LoadAbs, LoadIndexed, LoadIndirect)):
        return item.reg is reg
    if isinstance(item, Transfer):
        return item.dst_reg is reg
    if isinstance(item, (IncTarget, DecTarget)):
        return item.target is reg  # also reads it — see _reads_reg
    if reg is Reg.A and isinstance(item, (
        AdcImm, AdcAbs, AdcIndexed, SbcImm, SbcAbs, SbcIndexed, SbcIndirect,
        Bitwise, Asl, Lsr, Rol, Ror, Pla,
    )):
        # Accumulator arithmetic / shifts / pop all redefine A.
        return True
    return False


def _reads_reg(item, reg) -> bool:
    """True if the IR1 atom consumes register `reg`'s current value —
    including its use as an *index* register (`tbl,x` / `tbl,y`) or, for
    Y, the pointer index of `(zp),y`. Missing one of those would let the
    fold drop a still-live load, so the index cases are spelled out.

    `Unsupported` opcodes count as a read: their semantics are unknown,
    so they're a hard barrier — the fold never steps past an opcode the
    lifter hasn't modelled yet."""
    if isinstance(item, Unsupported):
        return True
    if isinstance(item, (StoreAbs, StoreIndexed, StoreIndirect, StoreLocal)) \
            and item.reg is reg:
        return True
    if isinstance(item, Transfer) and item.src_reg is reg:
        return True
    if isinstance(item, (CmpImm, CmpAbs, CmpIndexed, CmpIndirect)) and item.reg is reg:
        return True
    if isinstance(item, (IncTarget, DecTarget)) and item.target is reg:
        return True
    if reg is Reg.A and isinstance(item, (
        AdcImm, AdcAbs, AdcIndexed, SbcImm, SbcAbs, SbcIndexed, SbcIndirect,
        Bitwise, Asl, Lsr, Rol, Ror, Bit, Pha,
    )):
        return True
    if reg in (Reg.X, Reg.Y):
        # X / Y feed indexed addressing (`tbl,x` / `tbl,y`).
        if isinstance(item, (
            LoadIndexed, StoreIndexed, CmpIndexed, AdcIndexed, SbcIndexed,
        )) and item.index is reg:
            return True
        if isinstance(item, Bitwise) and isinstance(item.source, IndexedAbs) \
                and item.source.index is reg:
            return True
    if reg is Reg.Y:
        # `(zp),y` post-indexed indirect reads Y.
        if isinstance(item, (LoadIndirect, StoreIndirect, CmpIndirect, SbcIndirect)):
            return True
        if isinstance(item, Bitwise) and isinstance(item.source, IndirectY):
            return True
    return False


def _is_reg_load(item):
    """If `item` loads a register from a value pass-3 can fold (an
    immediate or a memory read), return `(reg, source_value)`; else
    None. `Transfer`/arithmetic write a register but their "source"
    isn't a standalone copyable value, so they don't qualify."""
    if isinstance(item, LoadImm):
        return (item.reg, item.imm)
    if isinstance(item, LoadAbs):
        return (item.reg, item.source)
    if isinstance(item, LoadIndexed):
        return (item.reg, IndexedAbs(base=item.base, index=item.index))
    if isinstance(item, LoadIndirect):
        return (item.reg, item.source)  # reg is always A for (zp),y
    return None


def _reg_store_target(item):
    """If `item` stores a register to memory, return `(reg, target)`
    (target an Abs / IndexedAbs / IndirectY); else None. `StoreLocal`
    (SMC operand patch) is excluded — its target is symbolic and an
    Assign can't represent it."""
    if isinstance(item, StoreAbs):
        return (item.reg, item.target)
    if isinstance(item, StoreIndexed):
        return (item.reg, IndexedAbs(base=item.base, index=item.index))
    if isinstance(item, StoreIndirect):
        return (item.reg, item.target)  # reg is always A for (zp),y
    return None


def _store_run(stmts: list[Stmt], start: int, reg):
    """Maximal run of consecutive stores of `reg` beginning at
    `stmts[start]`, returned as `(list_of_(target, src),
    index_after_run)`."""
    run: list[tuple[object, object]] = []
    k = start
    n = len(stmts)
    while k < n and isinstance(stmts[k], RawStmt):
        st = _reg_store_target(stmts[k].item)
        if st is None or st[0] is not reg:
            break
        run.append((st[1], stmts[k].item.src))
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


# ---------------------------------------------------------------- resource dispatch


def _resource_reads(item, res) -> bool:
    return _reads_c(item) if res is _CARRY else _reads_reg(item, res)


def _resource_writes(item, res) -> bool:
    return _writes_c(item) if res is _CARRY else _writes_reg(item, res)


def _resource_cond_reads(cond, res) -> bool:
    """Does an `IfStmt`'s `Compare` consume `res`? It re-derives `reg vs
    rhs` from registers, so it never reads the carry flag; for X/Y the
    rhs may be an indexed memory operand."""
    if res is _CARRY:
        return False
    if cond.reg is res:
        return True
    rhs = cond.rhs
    return res in (Reg.X, Reg.Y) and isinstance(rhs, IndexedAbs) and rhs.index is res


# ---------------------------------------------------------------- liveness over IR3


def _stmt_touches_resource(stmt: Stmt, res) -> bool:
    """Conservative: does this statement (including any nested block)
    read or write `res`? Used by `_first_resource_touch_is_write`. (The
    forward deadness scan in `_dead_from` classifies each statement type
    directly rather than relying on this coarse summary.)"""
    if isinstance(stmt, RawStmt):
        return _resource_reads(stmt.item, res) or _resource_writes(stmt.item, res)
    if isinstance(stmt, Assign):
        return False  # a folded copy / expr touches no register or flag
    if isinstance(stmt, (BreakStmt, ContinueStmt, GotoStmt, LabelStmt)):
        return False  # pure control transfer
    if isinstance(stmt, (CallStmt, TailCallStmt, ReturnStmt)):
        return True  # may pass / return the resource
    if isinstance(stmt, IfStmt):
        if _resource_cond_reads(stmt.cond, res):
            return True
        return _block_touches_resource(stmt.then_block, res) or (
            stmt.else_block is not None and _block_touches_resource(stmt.else_block, res)
        )
    if isinstance(stmt, RawIfStmt):
        return True  # reads a raw 6502 flag — conservatively touches
    if isinstance(stmt, LoopStmt):
        return _block_touches_resource(stmt.body, res)
    return True


def _block_touches_resource(block: Block, res) -> bool:
    return any(_stmt_touches_resource(s, res) for s in block.stmts)


def _first_resource_touch_is_write(block: Block, res) -> bool:
    """Scanning a block from the top, is the first statement that
    touches `res` a *write*? If so, the block doesn't depend on the
    incoming value of `res` — used to decide it's dead across a loop's
    back-edge (the next iteration overwrites it before reading)."""
    for s in block.stmts:
        if not _stmt_touches_resource(s, res):
            continue
        if isinstance(s, RawStmt):
            return _resource_writes(s.item, res) and not _resource_reads(s.item, res)
        # A nested control-flow stmt touches `res` in a way we can't
        # cheaply classify — be conservative.
        return False
    return False  # never touches `res` → incoming value not required,
    # but nothing here proves a write either; safe default is "not a
    # clean write-first", so callers stay conservative.


# ---------------------------------------------------------------- interprocedural register demand


_REG_SET = frozenset(_REGS)


def _has_unstructured(block: Block) -> bool:
    """Does `block` (recursively) contain a `Goto`/`Label` escape hatch?
    Such routines defeat the structured demand scan, so we treat them as
    demanding every register."""
    for s in block.stmts:
        if isinstance(s, (GotoStmt, LabelStmt)):
            return True
        for attr in ("then_block", "else_block", "body"):
            inner = getattr(s, attr, None)
            if inner is not None and hasattr(inner, "stmts") and _has_unstructured(inner):
                return True
    return False


def _stmt_demands(stmt: Stmt, reg, dm: dict):
    """Tri-state for a single statement: True = some path reads `reg`
    before writing it; False = every path writes `reg` before reading
    it (a kill); None = neither (the caller keeps scanning)."""
    if isinstance(stmt, RawStmt):
        if _reads_reg(stmt.item, reg):
            return True
        if _writes_reg(stmt.item, reg):
            return False
        return None
    if isinstance(stmt, Assign):
        return None
    if isinstance(stmt, ReturnStmt):
        return False  # path ends without reading reg from the entry state
    if isinstance(stmt, (BreakStmt, ContinueStmt)):
        return False  # exits / restarts the loop; no read here
    if isinstance(stmt, (GotoStmt, LabelStmt)):
        return True   # unstructured — conservatively a read
    if isinstance(stmt, CallStmt):
        # If the callee reads reg before writing it, reg is demanded.
        # Otherwise the call is transparent for demand (it may clobber
        # reg, but we don't track that — keep scanning).
        return True if reg in dm.get(stmt.target, _REG_SET) else None
    if isinstance(stmt, TailCallStmt):
        # Tail call ends the path: demanded iff the target reads reg.
        return reg in dm.get(stmt.target, _REG_SET)
    if isinstance(stmt, IfStmt):
        if _resource_cond_reads(stmt.cond, reg):
            return True
        return _branches_demand(stmt.then_block, stmt.else_block, reg, dm)
    if isinstance(stmt, RawIfStmt):
        # The cond reads a raw flag, not a register.
        return _branches_demand(stmt.then_block, stmt.else_block, reg, dm)
    if isinstance(stmt, LoopStmt):
        # The body runs at least once; its demand from the top decides.
        return _seq_demands(stmt.body.stmts, 0, reg, dm)
    return None


def _branches_demand(then_block: Block, else_block, reg, dm):
    t = _seq_demands(then_block.stmts, 0, reg, dm)
    e = _seq_demands(else_block.stmts, 0, reg, dm) if else_block is not None else None
    if t is True or e is True:
        return True
    # A kill only if *both* branches kill reg. With no `else`, the false
    # edge falls through past the `if`, so it isn't a kill.
    if t is False and else_block is not None and e is False:
        return False
    return None


def _seq_demands(stmts: list[Stmt], idx: int, reg, dm: dict):
    """Tri-state demand for `stmts[idx:]` — the first statement that
    determines `reg` (reads → True, writes → False); None if none do."""
    for k in range(idx, len(stmts)):
        r = _stmt_demands(stmts[k], reg, dm)
        if r is not None:
            return r
    return None


def _compute_register_demand(routines) -> dict:
    """Least fixed-point of each routine's register demand — the set of
    registers it may read before writing, so a caller must treat them as
    live. Keyed by every name a routine answers to (canonical name +
    entry aliases). Targets not in this set (cross-module / unknown)
    default to all registers when looked up, which is conservative."""
    alias_groups = [
        (r, (r.name, *getattr(r, "entry_aliases", []))) for r in routines
    ]
    unstructured = {r.name: _has_unstructured(r.body) for r in routines}
    dm: dict = {}
    for _, ns in alias_groups:
        for n in ns:
            dm[n] = frozenset()
    changed = True
    while changed:
        changed = False
        for r, ns in alias_groups:
            if unstructured[r.name]:
                new = _REG_SET
            else:
                new = frozenset(
                    reg for reg in _REGS
                    if _seq_demands(r.body.stmts, 0, reg, dm) is True
                )
            for n in ns:
                if dm[n] != new:
                    dm[n] = new
                    changed = True
    return dm


def _resource_demanded(target: str, res, demand) -> bool:
    """Does calling / tail-calling `target` read `res`? Carry isn't
    demand-analysed (pass 2 owns flag liveness), so it's always assumed
    read; an absent `demand` map or unknown target is conservative."""
    if res is _CARRY:
        return True
    if demand is None:
        return True
    return res in demand.get(target, _REG_SET)


# ---------------------------------------------------------------- the fold


def _dead_from(
    stmts: list[Stmt],
    idx: int,
    *,
    reads,
    writes,
    cond_reads,
    demand_reads,
    dead_fallthrough: bool,
    dead_break: bool,
    dead_continue: bool,
) -> bool:
    """Is the tracked resource (A / X / Y, or the carry flag) dead from
    `stmts[idx:]` onward — i.e. overwritten before being read on
    *every* control-flow path?

    `reads` / `writes` classify a `RawStmt`'s atom; `cond_reads` says
    whether an `IfStmt`'s `Compare` consumes the resource;
    `demand_reads(target)` says whether calling / tail-calling `target`
    reads it (from the interprocedural demand analysis). The scan is
    path-sensitive: each way out of the current statement list carries
    its own deadness:

    * `dead_fallthrough` — control runs off the end of this list.
    * `dead_break` — a `break` exits the innermost enclosing loop.
    * `dead_continue` — a `continue` re-enters the innermost loop.

    Per-statement rules:

    * `RawStmt` that `reads` the resource → live (False); that `writes`
      it (without reading) → dead (True); otherwise step past it.
    * `Return` → live (a caller may read whatever the routine returns).
    * `Call` whose target reads the resource → live; otherwise step
      past it (the call doesn't observe the dropped value).
    * `TailCall` whose target reads the resource → live; otherwise the
      path ends without a read → dead.
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
        if isinstance(s, ReturnStmt):
            return False
        if isinstance(s, CallStmt):
            if demand_reads(s.target):
                return False
            i += 1  # call doesn't read the resource — keep scanning
            continue
        if isinstance(s, TailCallStmt):
            return not demand_reads(s.target)
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
                      demand_reads=demand_reads,
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


# `Edges` is the per-resource deadness at the three ways out of the
# block currently being folded: control falling off the end, a `break`,
# and a `continue`. Each is a dict keyed by resource (A / X / Y / carry).
_Edges = tuple  # (dead_fallthrough, dead_break, dead_continue) of dicts


def _dead(stmts: list[Stmt], idx: int, res, edges, demand) -> bool:
    """Is `res` dead from `stmts[idx:]` onward, given this block's edge
    deadness `edges` and the module's register `demand` map? Binds the
    resource's read/write/cond/demand predicates and looks up its edge
    values."""
    dead_fallthrough, dead_break, dead_continue = edges
    return _dead_from(
        stmts, idx,
        reads=lambda it: _resource_reads(it, res),
        writes=lambda it: _resource_writes(it, res),
        cond_reads=lambda c: _resource_cond_reads(c, res),
        demand_reads=lambda tgt: _resource_demanded(tgt, res, demand),
        dead_fallthrough=dead_fallthrough[res],
        dead_break=dead_break[res],
        dead_continue=dead_continue[res],
    )


def _all_dead(value: bool) -> dict:
    return {res: value for res in _RESOURCES}


def _fold_block(block: Block, *, edges, demand) -> Block:
    """Fold register copy / arithmetic runs in `block`. `edges` is the
    per-resource deadness at this block's exits (fall-through, break,
    continue) — threaded down so a fold inside a loop body knows whether
    the register / carry survives the back-edge and break paths.
    `demand` is the interprocedural register-demand map."""
    dead_break, dead_continue = edges[1], edges[2]
    stmts = list(block.stmts)
    out: list[Stmt] = []
    i = 0
    n = len(stmts)
    while i < n:
        stmt = stmts[i]

        # Recurse into nested blocks first. Nested `if` bodies fall
        # through to *after* the `if`; rather than recompute that edge
        # we pass an all-live fall-through (conservative — only costs
        # missed folds). `break`/`continue` inside the body still target
        # the same enclosing loop, so those edges pass through unchanged.
        if isinstance(stmt, (IfStmt, RawIfStmt)):
            inner = (_all_dead(False), dead_break, dead_continue)
            stmt = replace(
                stmt,
                then_block=_fold_block(stmt.then_block, edges=inner, demand=demand),
                else_block=(
                    _fold_block(stmt.else_block, edges=inner, demand=demand)
                    if stmt.else_block is not None else None
                ),
            )
        elif isinstance(stmt, LoopStmt):
            # Re-entry (fall-off-bottom and `continue`): the next
            # iteration must overwrite the resource before reading it.
            # Break: jump to whatever follows the loop in this block.
            reentry = {
                res: _first_resource_touch_is_write(stmt.body, res)
                for res in _RESOURCES
            }
            post_loop = {
                res: _dead(stmts, i + 1, res, edges, demand) for res in _RESOURCES
            }
            stmt = replace(
                stmt,
                body=_fold_block(
                    stmt.body, edges=(reentry, post_loop, reentry), demand=demand
                ),
            )

        # Try to start a fold run at a foldable register load.
        if isinstance(stmt, RawStmt):
            load = _is_reg_load(stmt.item)
            if load is not None:
                reg, source = load

                # (1) Arithmetic fold (A only): load ; clc/sec ; adc/sbc
                # ; sta. Single store only — the stored value isn't the
                # source value, so a multi-store run isn't an idempotent
                # write-back and could clobber the operand. A *and* the
                # carry the add/sub set must be dead afterwards.
                if reg is Reg.A:
                    arith = _arith_after_load(stmts, i)
                    if arith is not None and arith[1] is not None:
                        op, operand, store_start = arith
                        targets, j = _store_run(stmts, store_start, Reg.A)
                        if (
                            len(targets) == 1
                            and _dead(stmts, j, Reg.A, edges, demand)
                            and _dead(stmts, j, _CARRY, edges, demand)
                        ):
                            tgt, ssrc = targets[0]
                            out.append(Assign(
                                target=tgt,
                                source=BinExpr(op=op, lhs=source, rhs=operand),
                                src=ssrc,
                            ))
                            i = j
                            continue

                # (2) Pure copy fold (A / X / Y): load ; sta [; sta ...].
                # A run of stores of the same register fed by one load;
                # sound for multiple stores because each writes the
                # (unchanged) source value.
                targets, j = _store_run(stmts, i + 1, reg)
                if targets and _dead(stmts, j, reg, edges, demand):
                    for tgt, ssrc in targets:
                        out.append(Assign(target=tgt, source=source, src=ssrc))
                    i = j
                    continue

        out.append(stmt)
        i += 1
    return Block.of(out)


def fold_routine(routine: RoutineIR3, demand: dict | None = None) -> RoutineIR3:
    # The routine's top-level block falls through to `return` /
    # tail-call, where a register might be a return value — so nothing
    # is dead at the end. No enclosing loop, so break/continue edges are
    # all-live too (they shouldn't appear at this level anyway).
    edges = (_all_dead(False), _all_dead(False), _all_dead(False))
    return replace(routine, body=_fold_block(routine.body, edges=edges, demand=demand))


def fold_module(module: ModuleIR3) -> ModuleIR3:
    # Interprocedural register demand lets a `sta DST ; jmp R` fold when
    # R doesn't read the stored register — computed across this module's
    # routines (cross-module targets stay conservative).
    demand = _compute_register_demand(module.routines)
    return ModuleIR3(
        name=module.name,
        file=module.file,
        routines=[fold_routine(r, demand) for r in module.routines],
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
