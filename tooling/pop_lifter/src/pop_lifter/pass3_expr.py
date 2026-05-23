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
    Imm,
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
    Wide16Stmt,
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


def _shift_after_load(stmts: list[Stmt], i: int):
    """If `stmts[i+1:]` begins with a maximal run of one or more `asl a`
    or `lsr a` (all the same direction), return `(op, count, store_start)`
    where `op` is `"<<"` or `">>"` and `count` is the number of shifts.
    Returns None if there is no shift immediately after the load.

    The caller is responsible for the dead-after-A and dead-after-carry
    checks: `asl`/`lsr` both write A and write carry (fresh, not a
    carry-through), so both must be dead past the following store for the
    fold to be sound."""
    n = len(stmts)
    if i + 1 >= n or not isinstance(stmts[i + 1], RawStmt):
        return None
    item1 = stmts[i + 1].item
    if isinstance(item1, Asl):
        shift_op = "<<"
    elif isinstance(item1, Lsr):
        shift_op = ">>"
    else:
        return None
    j = i + 1
    count = 0
    while j < n and isinstance(stmts[j], RawStmt):
        item = stmts[j].item
        if shift_op == "<<" and isinstance(item, Asl):
            count += 1
            j += 1
        elif shift_op == ">>" and isinstance(item, Lsr):
            count += 1
            j += 1
        else:
            break
    return (shift_op, count, j)


def _wide16_at(stmts: list[Stmt], i: int, lo_src):
    """If `stmts[i:i+7]` is exactly the 16-bit add/sub idiom, return a
    `Wide16Stmt`; otherwise None.

    Pattern (stmts[i] is already confirmed as a `LoadA` with source
    `lo_src` by the caller):

        stmts[i]   : lda LO_SRC           (caller's RawStmt load)
        stmts[i+1] : clc / sec
        stmts[i+2] : adc / sbc LO_OP
        stmts[i+3] : sta LO_DST
        stmts[i+4] : lda HI_SRC
        stmts[i+5] : adc / sbc HI_OP      (bare — no preceding carry set)
        stmts[i+6] : sta HI_DST

    All seven must be `RawStmt`.  The `adc`/`sbc` types at [i+2] and
    [i+5] must agree: add-pair (`clc` + two `adc`) or sub-pair (`sec` +
    two `sbc`).  `StoreLocal` (SMC patch) at [i+3] or [i+6] is rejected
    — `_reg_store_target` already excludes it.

    No dead-after check: this is a structural replacement that preserves
    all observable side-effects of the seven instructions."""
    if i + 6 >= len(stmts):
        return None
    for k in range(1, 7):
        if not isinstance(stmts[i + k], RawStmt):
            return None
    items = [stmts[i + k].item for k in range(1, 7)]
    # items[0..5] map to stmts[i+1..i+6]

    # [i+1] clc / sec
    if isinstance(items[0], Clc):
        op = "+"
        adc_types = (AdcImm, AdcAbs, AdcIndexed)
    elif isinstance(items[0], Sec):
        op = "-"
        adc_types = (SbcImm, SbcAbs, SbcIndexed, SbcIndirect)
    else:
        return None

    # [i+2] adc/sbc LO_OP
    if not isinstance(items[1], adc_types):
        return None
    lo_op = _arith_operand(items[1])
    if lo_op is None:
        return None

    # [i+3] sta LO_DST  (regular store — StoreLocal excluded by _reg_store_target)
    st3 = _reg_store_target(items[2])
    if st3 is None or st3[0] is not Reg.A:
        return None
    lo_dst = st3[1]

    # [i+4] lda HI_SRC
    ld4 = _is_reg_load(items[3])
    if ld4 is None or ld4[0] is not Reg.A:
        return None
    hi_src = ld4[1]

    # [i+5] bare adc/sbc HI_OP (same op type as [i+2], no carry set before)
    if not isinstance(items[4], adc_types):
        return None
    hi_op = _arith_operand(items[4])
    if hi_op is None:
        return None

    # [i+6] sta HI_DST
    st6 = _reg_store_target(items[5])
    if st6 is None or st6[0] is not Reg.A:
        return None
    hi_dst = st6[1]

    return Wide16Stmt(
        op=op,
        lo_src=lo_src,
        lo_op=lo_op,
        lo_dst=lo_dst,
        hi_src=hi_src,
        hi_op=hi_op,
        hi_dst=hi_dst,
        src=stmts[i].item.src,
    )


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
    if isinstance(stmt, Wide16Stmt):
        # Clobbers A and carry; may also read X/Y as index registers in
        # any of the six address slots (sources, operands, and destinations).
        if res is Reg.A or res is _CARRY:
            return True
        if res in (Reg.X, Reg.Y):
            for v in (stmt.lo_src, stmt.lo_op, stmt.lo_dst,
                      stmt.hi_src, stmt.hi_op, stmt.hi_dst):
                if isinstance(v, IndexedAbs) and v.index is res:
                    return True
                if isinstance(v, IndirectY) and res is Reg.Y:
                    return True
        return False
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
        if isinstance(s, Wide16Stmt):
            # First step: `lda lo_src` kills A; `clc`/`sec` kills carry.
            # Neither reads the incoming value of A or carry.
            if res is Reg.A or res is _CARRY:
                return True
            return False  # X/Y may be read as indexes; be conservative
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


def _stmt_demands(stmt: Stmt, reg, *, tail_dm, call_dm, ret):
    """Tri-state for a single statement: True = some path reads `reg`
    (or, when `ret` is True, lets it escape via a return) before writing
    it; False = every path writes `reg` before reading it (a kill);
    None = neither (the caller keeps scanning).

    `call_dm` is the *read*-demand map consulted for a non-tail `Call`
    (the callee returns, so only an actual read of `reg` matters).
    `tail_dm` is the map consulted for a `TailCall` (its eventual return
    escapes to *our* caller, so `tail_dm` is the escape-aware map)."""
    if isinstance(stmt, RawStmt):
        if _reads_reg(stmt.item, reg):
            return True
        if _writes_reg(stmt.item, reg):
            return False
        return None
    if isinstance(stmt, Wide16Stmt):
        # `lda lo_src` kills A without reading the incoming A; `clc`/`sec`
        # kills carry without reading it.  X/Y are read only when used as
        # index registers across all six address slots.
        if reg is Reg.A:
            return False  # killed
        if reg in (Reg.X, Reg.Y):
            for v in (stmt.lo_src, stmt.lo_op, stmt.lo_dst,
                      stmt.hi_src, stmt.hi_op, stmt.hi_dst):
                if isinstance(v, IndexedAbs) and v.index is reg:
                    return True
                if isinstance(v, IndirectY) and reg is Reg.Y:
                    return True
        return None  # doesn't affect this register — keep scanning
    if isinstance(stmt, Assign):
        return None
    if isinstance(stmt, ReturnStmt):
        # An unwritten reg escapes to the caller here. For the
        # escape-aware (`live`) map that's a use; for the read-only map
        # it isn't.
        return True if ret else False
    if isinstance(stmt, (BreakStmt, ContinueStmt)):
        return False  # stays within the routine; no read / escape here
    if isinstance(stmt, (GotoStmt, LabelStmt)):
        return True   # unstructured — conservatively a use
    if isinstance(stmt, CallStmt):
        # The callee returns to us; only a genuine read matters. If it
        # doesn't read reg the call is transparent (it may clobber reg,
        # which we don't track — keep scanning).
        return True if reg in call_dm.get(stmt.target, _REG_SET) else None
    if isinstance(stmt, TailCallStmt):
        # The path ends here; reg is used iff it's read *or escapes*
        # through the target — that's the escape-aware map.
        return reg in tail_dm.get(stmt.target, _REG_SET)
    if isinstance(stmt, IfStmt):
        if _resource_cond_reads(stmt.cond, reg):
            return True
        return _branches_demand(
            stmt.then_block, stmt.else_block, reg,
            tail_dm=tail_dm, call_dm=call_dm, ret=ret,
        )
    if isinstance(stmt, RawIfStmt):
        # The cond reads a raw flag, not a register.
        return _branches_demand(
            stmt.then_block, stmt.else_block, reg,
            tail_dm=tail_dm, call_dm=call_dm, ret=ret,
        )
    if isinstance(stmt, LoopStmt):
        # The body runs at least once; its demand from the top decides.
        return _seq_demands(stmt.body.stmts, 0, reg, tail_dm=tail_dm,
                            call_dm=call_dm, ret=ret)
    # Any other node type — notably the late readability passes' nodes
    # (`MatchStmt`, `DoWhileStmt`) — isn't modelled here. The fold runs
    # on reloop output, before those passes, so this is never hit in the
    # pipeline; but if it were, conservatively treat the register as used
    # (True), never transparent (None), so demand can't be under-reported
    # into an unsound fold.
    return True


def _branches_demand(then_block: Block, else_block, reg, *, tail_dm, call_dm, ret):
    kw = dict(tail_dm=tail_dm, call_dm=call_dm, ret=ret)
    t = _seq_demands(then_block.stmts, 0, reg, **kw)
    e = _seq_demands(else_block.stmts, 0, reg, **kw) if else_block is not None else None
    if t is True or e is True:
        return True
    # A kill only if *both* branches kill reg. With no `else`, the false
    # edge falls through past the `if`, so it isn't a kill.
    if t is False and else_block is not None and e is False:
        return False
    return None


def _seq_demands(stmts: list[Stmt], idx: int, reg, *, tail_dm, call_dm, ret):
    """Tri-state demand for `stmts[idx:]` — the first statement that
    determines `reg` (used → True, killed → False); None if none do."""
    for k in range(idx, len(stmts)):
        r = _stmt_demands(stmts[k], reg, tail_dm=tail_dm, call_dm=call_dm, ret=ret)
        if r is not None:
            return r
    return None


def _demand_fixpoint(routines, *, ret, call_dm) -> dict:
    """Fixed-point of each routine's register demand. `ret` says whether
    a register escaping via a `Return` counts as a use (the escape-aware
    `live` map) or not (the `read` map). `call_dm` is the read-demand map
    used for non-tail calls; pass None to consult the map being computed
    (only correct when `ret` is False, i.e. computing the read map).
    Keyed by every name a routine answers to (canonical + aliases);
    unknown / cross-module targets default to all registers."""
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
                cdm = dm if call_dm is None else call_dm
                new = frozenset(
                    reg for reg in _REGS
                    if _seq_demands(r.body.stmts, 0, reg,
                                    tail_dm=dm, call_dm=cdm, ret=ret) is True
                )
            for n in ns:
                if dm[n] != new:
                    dm[n] = new
                    changed = True
    return dm


def _compute_register_demand(routines):
    """Return `(read_demand, live_demand)`.

    * `read_demand[R]` — registers R may *read* before writing them.
      Used at a non-tail `Call`: the callee returns, so only a genuine
      read can observe the caller's register.
    * `live_demand[R]` — registers live at R's entry: read before
      written, *or* still unwritten when R returns (escaping to R's
      caller, who may read them). Used at a `TailCall`, whose return
      unwinds to *our* caller — so a target that merely preserves the
      register keeps it live, and the copy must not be folded.

    `not (reg in live_demand[R])` is exactly "R must clobber reg before
    returning", the soundness condition for dropping a `sta DST ; jmp R`.
    """
    read_demand = _demand_fixpoint(routines, ret=False, call_dm=None)
    live_demand = _demand_fixpoint(routines, ret=True, call_dm=read_demand)
    return read_demand, live_demand


def _resource_demanded(target: str, res, demand) -> bool:
    """Does the given demand map mark `res` as used by `target`? Carry
    isn't demand-analysed (pass 2 owns flag liveness), so it's always
    assumed used; an absent map or unknown target is conservative."""
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
    call_reads,
    tail_reads,
    dead_fallthrough: bool,
    dead_break: bool,
    dead_continue: bool,
) -> bool:
    """Is the tracked resource (A / X / Y, or the carry flag) dead from
    `stmts[idx:]` onward — i.e. overwritten before being read on
    *every* control-flow path?

    `reads` / `writes` classify a `RawStmt`'s atom; `cond_reads` says
    whether an `IfStmt`'s `Compare` consumes the resource;
    `call_reads(target)` says whether a non-tail call to `target`
    reads it, and `tail_reads(target)` whether the resource is still
    live when tail-calling `target` (read by it *or* escaping through
    its return). The scan is path-sensitive: each way out of the
    current statement list carries its own deadness:

    * `dead_fallthrough` — control runs off the end of this list.
    * `dead_break` — a `break` exits the innermost enclosing loop.
    * `dead_continue` — a `continue` re-enters the innermost loop.

    Per-statement rules:

    * `RawStmt` that `reads` the resource → live (False); that `writes`
      it (without reading) → dead (True); otherwise step past it.
    * `Return` → live (a caller may read whatever the routine returns).
    * `Call` whose target reads the resource → live; otherwise step
      past it (the call doesn't observe the dropped value).
    * `TailCall` → live unless the target *must clobber* the resource
      before returning (`not tail_reads`); a target that merely
      preserves it lets the value escape to our caller.
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
        if isinstance(s, Wide16Stmt):
            # Step 1 (lda lo_src) writes A without reading it; probe with
            # Pla — also writes A, reads nothing else.
            if writes(Pla(src=s.src)):
                return True  # A killed at step 1 — dead from here
            # Step 2 (clc/sec) writes carry without reading it first.
            if writes(Clc(src=s.src)):
                return True  # carry killed at step 2 — dead from here
            # Resource must be X or Y — live iff any indexed/indirect slot reads it.
            # Check all six address slots: sources, operands, and store destinations.
            for v in (s.lo_src, s.lo_op, s.lo_dst, s.hi_src, s.hi_op, s.hi_dst):
                if isinstance(v, IndexedAbs):
                    if reads(AdcIndexed(base=v.base, index=v.index, src=s.src)):
                        return False  # indexed read — X or Y is live
                elif isinstance(v, IndirectY):
                    if reads(StoreIndirect(reg=Reg.A, target=v, src=s.src)):
                        return False  # (zp),y read — Y is live
            i += 1  # doesn't touch this X/Y register — step past
            continue
        if isinstance(s, Assign):
            i += 1  # a folded copy / expr touches neither A nor flags
            continue
        if isinstance(s, ReturnStmt):
            return False
        if isinstance(s, CallStmt):
            if call_reads(s.target):
                return False
            i += 1  # call doesn't read the resource — keep scanning
            continue
        if isinstance(s, TailCallStmt):
            return not tail_reads(s.target)
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
                      call_reads=call_reads, tail_reads=tail_reads,
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
    deadness `edges` and the module's `demand` = `(read, live)` maps?
    Binds the resource's read/write/cond/demand predicates and looks up
    its edge values. A non-tail `Call` consults the read map; a
    `TailCall` consults the escape-aware `live` map."""
    read_demand, live_demand = (None, None) if demand is None else demand
    dead_fallthrough, dead_break, dead_continue = edges
    return _dead_from(
        stmts, idx,
        reads=lambda it: _resource_reads(it, res),
        writes=lambda it: _resource_writes(it, res),
        cond_reads=lambda c: _resource_cond_reads(c, res),
        call_reads=lambda tgt: _resource_demanded(tgt, res, read_demand),
        tail_reads=lambda tgt: _resource_demanded(tgt, res, live_demand),
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
                    # (0) 16-bit arithmetic: seven-instruction sequence.
                    # No dead-after check needed — structural replacement
                    # preserves all side-effects of the original seven ops.
                    wide = _wide16_at(stmts, i, source)
                    if wide is not None:
                        out.append(wide)
                        i += 7
                        continue

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

                    # (2) Shift fold: lda X ; (asl)*n / (lsr)*n ; sta Y.
                    # A single left/right shift run. Both A and carry must
                    # be dead after the store (asl/lsr produce fresh carry).
                    shift = _shift_after_load(stmts, i)
                    if shift is not None:
                        shift_op, count, store_start = shift
                        targets, j = _store_run(stmts, store_start, Reg.A)
                        if (
                            len(targets) == 1
                            and _dead(stmts, j, Reg.A, edges, demand)
                            and _dead(stmts, j, _CARRY, edges, demand)
                        ):
                            tgt, ssrc = targets[0]
                            out.append(Assign(
                                target=tgt,
                                source=BinExpr(
                                    op=shift_op,
                                    lhs=source,
                                    rhs=Imm(value=count, text=f"#{count}"),
                                ),
                                src=ssrc,
                            ))
                            i = j
                            continue

                # (3) Pure copy fold (A / X / Y): load ; sta [; sta ...].
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


def wide16_stats(module: ModuleIR3) -> int:
    """Total `Wide16Stmt` nodes produced across the module."""
    def count(block: Block) -> int:
        total = 0
        for s in block.stmts:
            if isinstance(s, Wide16Stmt):
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
