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

import sys

from dataclasses import dataclass

from .interp_ir1 import (
    InterpError,
    Trace,
    _branch_taken,
    _eval_compare,
    _indexed_addr,
    _real_addr,
    _resolve_indirect_y,
    _set_zn,
    exec_atom,
)
from .interp_ir1 import run as ir1_run
from .ir1 import (
    Abs,
    DecTarget,
    IncTarget,
    Imm,
    IndexedAbs,
    IndirectY,
    ModuleIR1,
    Reg,
    Routine as IR1Routine,
)
from .ir3 import (
    Assign,
    BinExpr,
    Block,
    BreakStmt,
    CallStmt,
    ContinueStmt,
    DispatchStmt,
    DoWhileStmt,
    ForStmt,
    GotoStateStmt,
    GotoStmt,
    IfStmt,
    LabeledBlock,
    LabelStmt,
    LoopStmt,
    MatchStmt,
    ModuleIR3,
    RawIfStmt,
    RawStmt,
    RepeatStmt,
    RestoreTemp,
    ReturnStmt,
    RotateExpr,
    RoutineIR3,
    SaveTemp,
    Stmt,
    TailCallStmt,
    Wide16Stmt,
)


# Live `jsr` nesting beyond which a run is treated as non-terminating
# (`InterpError`). Sized to the 6502's 256-byte stack (two bytes per
# return address); a real program never nests this deep, so hitting it
# means a call cycle the static name resolution can't break.
_MAX_CALL_DEPTH = 128


class _ReturnSignal(Exception):
    """Raised by ReturnStmt to unwind to the enclosing call frame."""


class _TailCallSignal(Exception):
    """Carries a tail-call target up the recursion. The top-level
    `run` loop catches it and switches routines."""

    def __init__(self, target: str):
        super().__init__(target)
        self.target = target


class _BreakSignal(Exception):
    """Raised by BreakStmt to exit the innermost enclosing LoopStmt."""


class _LabeledBreakSignal(Exception):
    """Raised by a labeled BreakStmt to unwind to the matching
    LabeledBlock (the relooper's structured forward jump to a merge
    node)."""

    def __init__(self, label: str):
        super().__init__(label)
        self.label = label


class _ContinueSignal(Exception):
    """Raised by ContinueStmt to jump to the top of the innermost
    enclosing LoopStmt."""


class _GotoStateSignal(Exception):
    """Raised by GotoStateStmt to transition the enclosing DispatchStmt
    to a new state — the dispatch-loop fallback's CFG edge."""

    def __init__(self, state: int):
        super().__init__(state)
        self.state = state


@dataclass(frozen=True)
class _Ctx:
    """Static per-run resolution config threaded through the walker.

    * `modules` — the loaded module set (for the IR1-leaf delegate path).
    * `entry_owners` — entry name (routine name + each `entry_alias`) →
      list of modules that define it. Mirrors `pass4_crate.build_name_map`.
    * `routine_by_module` — `(module_name, entry_name)` → routine, so a
      resolved `(owner, target)` pair lands on the exact routine the crate
      would call.
    * `alias_idx` — flat alias→canonical map, only for the IR1 delegate.
    """

    modules: tuple
    entry_owners: dict[str, list[str]]
    routine_by_module: dict[tuple[str, str], object]
    alias_idx: dict[str, str]


def _build_ctx(modules, alias_idx: dict[str, str]) -> _Ctx:
    entry_owners: dict[str, list[str]] = {}
    routine_by_module: dict[tuple[str, str], object] = {}
    for m in modules:
        for r in m.routines:
            for nm in (r.name, *(getattr(r, "entry_aliases", []) or [])):
                owners = entry_owners.setdefault(nm, [])
                if m.name not in owners:
                    owners.append(m.name)
                routine_by_module[(m.name, nm)] = r
    return _Ctx(tuple(modules), entry_owners, routine_by_module, alias_idx)


def _resolve_call_ctx(ctx: _Ctx, home_module: str | None, target: str):
    """Resolve a call to `target` made from `home_module` using the crate's
    policy: intra-module preferred, then unique cross-module, else `None`
    (external — no owner — or ambiguous — many owners). A `None` makes the
    caller raise `InterpError`, so such a routine is skipped rather than
    compared against the crate's no-op stub. Returns `(routine, owner)`."""
    owners = ctx.entry_owners.get(target)
    if not owners:
        return None
    if home_module in owners:
        owner = home_module
    elif len(owners) == 1:
        owner = owners[0]
    else:
        return None
    return ctx.routine_by_module[(owner, target)], owner


def _resolve_tail(ctx: _Ctx, current, home_module: str | None, target: str):
    """Resolve a tail-call target. A jump back into the *current* routine's
    own entry — its name or one of the loop-label entry aliases the
    relooper exposes (e.g. `:loop`) — is a self-loop, not a cross-routine
    call. Resolve it to the current routine directly, the way the crate's
    lexical `tail_call :loop` does, before the global name table — where a
    generic local label like `:loop` collides across routines and
    last-wins would send the jump to the wrong one. Returns
    `(routine, owner)` or `None`."""
    if current is not None and (
        target == current.name
        or target in (getattr(current, "entry_aliases", ()) or ())
    ):
        return current, home_module
    return _resolve_call_ctx(ctx, home_module, target)


def _resolve_entry(modules, entry: str):
    """Resolve the run's root entry. The root has no caller, so it keeps
    the historical first-module-wins rule (the differential harness orders
    the target segment first precisely so this picks it). Returns
    `(routine, owner_module)` or `(None, None)`."""
    for m in modules:
        r = m.find(entry)
        if r is not None:
            return r, m.name
    return None, None


def run(
    modules: list[ModuleIR3 | ModuleIR1],
    entry: str,
    *,
    ram: bytearray | None = None,
    a: int = 0,
    x: int = 0,
    y: int = 0,
    c: int = 0,
    aliases: dict[str, str] | None = None,
) -> Trace:
    """Execute `entry` in the given module set (mix of IR3 / IR1).

    Initial registers / carry default to zero; pass `a`/`x`/`y`/`c` to
    seed them (used by the differential harness to match a non-zero start
    state). Tail-call chaining is handled by an outer loop so deeply-
    nested cross-module tails don't stack the Python recursion. Calls and
    nested blocks recurse normally.
    """
    if ram is None:
        ram = bytearray(0x10000)
    trace = Trace(ram=ram, a=a & 0xff, x=x & 0xff, y=y & 0xff, c=c & 1)
    alias_idx = _alias_index(modules)
    if aliases:
        alias_idx.update(aliases)
    ctx = _build_ctx(modules, alias_idx)

    routine, owner = _resolve_entry(modules, entry)
    if routine is None:
        raise InterpError(f"entry {entry!r} not found in any module")
    trace.home_module = owner

    # The structured walker uses one Python frame per nested block/call, so
    # a legitimate `_MAX_CALL_DEPTH` chain needs headroom above CPython's
    # default limit. Raise it for the duration of the run (restored below)
    # and treat any overflow that still slips through as non-terminating
    # rather than a hard crash.
    prev_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(prev_limit, _MAX_CALL_DEPTH * 40 + 1000))
    try:
        while True:
            try:
                _exec_routine(routine, ctx, trace)
                return trace
            except _TailCallSignal as tc:
                nxt = _resolve_tail(ctx, routine, trace.home_module, tc.target)
                if nxt is None:
                    raise InterpError(
                        f"tail-call target {tc.target!r} unresolved from "
                        f"module {trace.home_module!r} (external/ambiguous)"
                    )
                routine, owner = nxt
                trace.home_module = owner
                continue
    except RecursionError as e:
        raise InterpError("Python recursion limit hit (non-terminating run)") from e
    finally:
        sys.setrecursionlimit(prev_limit)


def _alias_index(modules) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in modules:
        for r in m.routines:
            out[r.name] = r.name
            for a in getattr(r, "entry_aliases", []) or []:
                out[a] = r.name
    return out


def _exec_routine(routine, ctx: _Ctx, trace: Trace) -> None:
    if isinstance(routine, RoutineIR3):
        try:
            _exec_block(routine.body, ctx, trace)
        except _ReturnSignal:
            pass
        return
    if isinstance(routine, IR1Routine):
        # Delegate to the IR1 interpreter for IR1 leaves. We share the
        # RAM bytearray via `trace.ram`, so observable state stays in
        # sync.
        ir1_modules = [m for m in ctx.modules if isinstance(m, ModuleIR1)]
        # We need the IR1 routine reachable via the same alias map.
        # Re-running `ir1_run` with the same alias index keeps the
        # behaviour identical to "this routine in isolation".
        ir1_run(
            ir1_modules,
            routine.name,
            ram=trace.ram,
            a=trace.a, x=trace.x, y=trace.y, c=trace.c,
            aliases=ctx.alias_idx,
        )
        return
    raise InterpError(f"unknown routine type: {type(routine).__name__}")


def _exec_block(block: Block, ctx: _Ctx, trace: Trace) -> None:
    for stmt in block.stmts:
        _exec_stmt(stmt, ctx, trace)


def _assign_read(value, trace: Trace, src) -> int:
    """Read the source of a pass-3 `Assign` — an immediate or one of the
    memory-read forms — at the current register state. Mirrors the
    load-side of `exec_atom` so the fold's interpretation can't drift
    from the unfolded `lda` it replaced. `src` is the Assign's
    `SourceRef`, threaded into the address gates so a synthetic-label
    failure points at the right source line."""
    if isinstance(value, Imm):
        return value.value & 0xff
    if isinstance(value, IndexedAbs):
        return trace.ram[_indexed_addr(value, trace, src)]
    if isinstance(value, IndirectY):
        return trace.ram[_resolve_indirect_y(value, trace, trace.ram)]
    if isinstance(value, Abs):
        return trace.ram[_real_addr(value.addr, src)]
    if isinstance(value, BinExpr):
        # Folded arithmetic / shift: result wraps to 8 bits.
        lhs = _assign_read(value.lhs, trace, src)
        rhs = _assign_read(value.rhs, trace, src)
        if value.op == "+":
            return (lhs + rhs) & 0xff
        if value.op == "-":
            return (lhs - rhs) & 0xff
        if value.op == "<<":
            return (lhs << rhs) & 0xff
        if value.op == ">>":
            return (lhs >> rhs) & 0xff
        raise InterpError(f"unknown BinExpr op: {value.op!r}")
    if isinstance(value, RotateExpr):
        val = _assign_read(value.operand, trace, src)
        c = trace.c & 1
        for _ in range(value.count):
            if value.op == "rotl":
                new_c = (val >> 7) & 1
                val = ((val << 1) | c) & 0xff
            elif value.op == "rotr":
                new_c = val & 1
                val = ((val >> 1) | (c << 7)) & 0xff
            else:
                raise InterpError(f"unknown RotateExpr op: {value.op!r}")
            c = new_c
        # Carry-out not written back — dead by the fold's soundness condition.
        return val
    raise InterpError(f"unknown Assign source type: {type(value).__name__}")


def _assign_addr(target, trace: Trace, src) -> int:
    """Resolve the destination address of a pass-3 `Assign`. Mirrors the
    store-side of `exec_atom`. `src` is threaded into the address gates
    for the same diagnostics reason as `_assign_read`."""
    if isinstance(target, IndexedAbs):
        return _indexed_addr(target, trace, src)
    if isinstance(target, IndirectY):
        return _resolve_indirect_y(target, trace, trace.ram)
    if isinstance(target, Abs):
        return _real_addr(target.addr, src)
    raise InterpError(f"unknown Assign target type: {type(target).__name__}")


def _exec_stmt(stmt: Stmt, ctx: _Ctx, trace: Trace) -> None:
    if isinstance(stmt, RawStmt):
        handled = exec_atom(stmt.item, trace, trace.ram)
        if not handled:
            raise InterpError(
                f"IR3 RawStmt wraps a non-atom item: {type(stmt.item).__name__}"
            )
        return
    if isinstance(stmt, Wide16Stmt):
        # Replay the seven-instruction 16-bit add/sub faithfully:
        #   lda lo_src ; clc/sec ; adc/sbc lo_op ; sta lo_dst
        #   lda hi_src ; adc/sbc hi_op           ; sta hi_dst
        # Each step's side-effects (A, carry, Z/N, memory writes) must
        # match the original to pass differential tests.
        lo_src_val = _assign_read(stmt.lo_src, trace, stmt.src)
        lo_op_val = _assign_read(stmt.lo_op, trace, stmt.src)
        if stmt.op == "+":
            lo_sum = lo_src_val + lo_op_val          # clc → no carry-in
            lo_result = lo_sum & 0xFF
            lo_carry = (lo_sum >> 8) & 1
        else:
            lo_diff = lo_src_val - lo_op_val         # sec → borrow-in = 0
            lo_result = lo_diff & 0xFF
            lo_carry = 1 if lo_diff >= 0 else 0      # C=1 means no borrow
        lo_dst_addr = _assign_addr(stmt.lo_dst, trace, stmt.src)
        trace.ram[lo_dst_addr] = lo_result
        trace.writes[lo_dst_addr] = lo_result
        trace.a = lo_result                         # A after sta lo_dst
        # lda hi_src (reads AFTER lo_dst was written — matches 6502 order)
        hi_src_val = _assign_read(stmt.hi_src, trace, stmt.src)
        hi_op_val = _assign_read(stmt.hi_op, trace, stmt.src)
        if stmt.op == "+":
            hi_sum = hi_src_val + hi_op_val + lo_carry
            hi_result = hi_sum & 0xFF
            hi_carry = (hi_sum >> 8) & 1
        else:
            hi_diff = hi_src_val - hi_op_val - (1 - lo_carry)
            hi_result = hi_diff & 0xFF
            hi_carry = 1 if hi_diff >= 0 else 0
        hi_dst_addr = _assign_addr(stmt.hi_dst, trace, stmt.src)
        trace.ram[hi_dst_addr] = hi_result
        trace.writes[hi_dst_addr] = hi_result
        trace.a = hi_result                         # A after sta hi_dst
        trace.c = hi_carry
        _set_zn(trace, hi_result)
        return
    if isinstance(stmt, Assign):
        # `target = source`, the dropped `lda`/`sta` round-trip. Read the
        # source then write the destination — A and the Z/N flags the
        # original load set are intentionally NOT touched (pass 3 only
        # folds when they're dead).
        value = _assign_read(stmt.source, trace, stmt.src)
        addr = _assign_addr(stmt.target, trace, stmt.src)
        trace.ram[addr] = value
        trace.writes[addr] = value
        return
    if isinstance(stmt, SaveTemp):
        # Recovered `pha`. The slot is naming metadata; the value rides
        # the same shared value stack as `pha`/`pla`, so this is exactly
        # the push it replaced (and stays correct across call frames,
        # where slot ids would otherwise collide).
        trace.value_stack.append(trace.a & 0xff)
        if len(trace.value_stack) > trace.max_value_stack_depth:
            trace.max_value_stack_depth = len(trace.value_stack)
        return
    if isinstance(stmt, RestoreTemp):
        # Recovered `pla`: pop the shared value stack into A and set Z/N,
        # mirroring `pla` exactly.
        if not trace.value_stack:
            raise InterpError(
                f"RestoreTemp (tmp{stmt.slot}) on empty value stack at "
                f"{stmt.src.short()} ({stmt.src.raw!r}) — unbalanced save/restore?"
            )
        trace.a = trace.value_stack.pop()
        _set_zn(trace, trace.a)
        return
    if isinstance(stmt, CallStmt):
        resolved = _resolve_call_ctx(ctx, trace.home_module, stmt.target)
        if resolved is None:
            # External (no owner) or ambiguous (reused name in several
            # modules) from this caller — exactly what the crate emits as
            # an `ext` no-op stub. We can't faithfully reproduce that, so
            # skip the routine rather than risk a false divergence.
            raise InterpError(
                f"call target {stmt.target!r} unresolved from module "
                f"{trace.home_module!r} (external/ambiguous)"
            )
        callee, owner = resolved
        if trace.call_depth >= _MAX_CALL_DEPTH:
            # A real 6502 would overflow its 256-byte stack long before
            # this; an unbounded cycle here means the routine can't be run
            # to completion (commonly a bank-switch trampoline whose
            # soft-switch remap we don't model). Out of scope, not a crash.
            raise InterpError(
                f"call depth exceeded {_MAX_CALL_DEPTH} at {stmt.target!r} "
                "(recursion / stack overflow)"
            )
        # A `jsr` establishes a return boundary. If the callee tail-calls
        # (`jmp X`), X's `rts` returns to *this* caller — not further up —
        # so resolve the tail-call chain inside the call frame rather than
        # letting the signal unwind to the top-level loop (which would
        # abandon the rest of this routine). Mirrors `run`'s tail loop.
        # Calls made from inside the callee resolve relative to its owning
        # module, so set `home_module` accordingly across the frame.
        trace.call_depth += 1
        saved_home = trace.home_module
        try:
            while True:
                trace.home_module = owner
                try:
                    _exec_routine(callee, ctx, trace)
                    break
                except _TailCallSignal as tc:
                    nxt = _resolve_tail(ctx, callee, owner, tc.target)
                    if nxt is None:
                        raise InterpError(
                            f"tail-call target {tc.target!r} unresolved from "
                            f"module {owner!r} (external/ambiguous)"
                        )
                    callee, owner = nxt
        finally:
            trace.home_module = saved_home
            trace.call_depth -= 1
        return
    if isinstance(stmt, TailCallStmt):
        raise _TailCallSignal(stmt.target)
    if isinstance(stmt, ReturnStmt):
        raise _ReturnSignal()
    if isinstance(stmt, IfStmt):
        if _eval_compare(stmt.cond, trace, trace.ram):
            _exec_block(stmt.then_block, ctx, trace)
        elif stmt.else_block is not None:
            _exec_block(stmt.else_block, ctx, trace)
        return
    if isinstance(stmt, MatchStmt):
        reg_val = {Reg.A: trace.a, Reg.X: trace.x, Reg.Y: trace.y}[stmt.reg]
        for arm in stmt.arms:
            if any((v.value & 0xff) == reg_val for v in arm.values):
                # The arm terminates (return / tail-call / break / ...),
                # so this raises the matching signal; if it somehow falls
                # off, returning here matches the no-match fall-through.
                _exec_block(arm.body, ctx, trace)
                return
        return  # no arm matched — fall through to the next statement
    if isinstance(stmt, RawIfStmt):
        if _branch_taken(stmt.cond, trace):
            _exec_block(stmt.then_block, ctx, trace)
        elif stmt.else_block is not None:
            _exec_block(stmt.else_block, ctx, trace)
        return
    if isinstance(stmt, LoopStmt):
        # Bounded so a busted exit guard doesn't hang the interpreter.
        # 6502 routines we're modelling don't iterate more than ~1024
        # times in practice — a million is room to spare for tests.
        for _ in range(1_000_000):
            try:
                _exec_block(stmt.body, ctx, trace)
            except _ContinueSignal:
                continue
            except _BreakSignal:
                break
            # Body fell off the bottom without break/continue —
            # natural iteration, keep looping.
        else:
            raise InterpError(
                "LoopStmt exceeded 1,000,000 iterations — exit guard bug?"
            )
        return
    if isinstance(stmt, DoWhileStmt):
        for _ in range(1_000_000):
            try:
                _exec_block(stmt.body, ctx, trace)
            except _ContinueSignal:
                # A 6502 back-edge `continue` restarts the body without
                # re-testing the bottom guard.
                continue
            except _BreakSignal:
                break
            # Body fell through — test the continue condition at the
            # bottom, exactly where the original guard sat.
            if not _eval_compare(stmt.cond, trace, trace.ram):
                break
        else:
            raise InterpError(
                "DoWhileStmt exceeded 1,000,000 iterations — exit guard bug?"
            )
        return
    if isinstance(stmt, ForStmt):
        # Recovered counted loop. Replay the init `LoadImm`, the
        # `inx`/`dey` step, and the continue condition so register +
        # flag state matches the original do-while exactly.
        start = stmt.start.value & 0xff
        if stmt.var is Reg.A:
            trace.a = start
        elif stmt.var is Reg.X:
            trace.x = start
        else:
            trace.y = start
        _set_zn(trace, start)
        step_op = (DecTarget if stmt.step < 0 else IncTarget)(
            target=stmt.var, src=stmt.src)
        for _ in range(1_000_000):
            try:
                _exec_block(stmt.body, ctx, trace)
            except _ContinueSignal:
                # Matches the do-while back-edge: restart body, no step.
                continue
            except _BreakSignal:
                break
            exec_atom(step_op, trace, trace.ram)
            if not _eval_compare(stmt.cond, trace, trace.ram):
                break
        else:
            raise InterpError(
                "ForStmt exceeded 1,000,000 iterations — counter bug?"
            )
        return
    if isinstance(stmt, RepeatStmt):
        # Fixed-count busy-wait. Replay the init, body, and step `count`
        # times — the full byte wrap leaves `var` back at its start.
        if not (0 < stmt.count <= 1_000_000):
            raise InterpError(
                f"RepeatStmt count {stmt.count} out of range (0, 1,000,000]"
            )
        start = stmt.start.value & 0xff
        if stmt.var is Reg.A:
            trace.a = start
        elif stmt.var is Reg.X:
            trace.x = start
        else:
            trace.y = start
        _set_zn(trace, start)
        step_op = (DecTarget if stmt.step < 0 else IncTarget)(
            target=stmt.var, src=stmt.src)
        for _ in range(stmt.count):
            _exec_block(stmt.body, ctx, trace)
            exec_atom(step_op, trace, trace.ram)
        return
    if isinstance(stmt, LabeledBlock):
        try:
            _exec_block(stmt.body, ctx, trace)
        except _LabeledBreakSignal as sig:
            if sig.label != stmt.label:
                raise  # not ours — keep unwinding to the matching block
        return
    if isinstance(stmt, BreakStmt):
        if stmt.label is not None:
            raise _LabeledBreakSignal(stmt.label)
        raise _BreakSignal()
    if isinstance(stmt, ContinueStmt):
        raise _ContinueSignal()
    if isinstance(stmt, GotoStateStmt):
        raise _GotoStateSignal(stmt.state)
    if isinstance(stmt, DispatchStmt):
        # `loop { match pc { ... } }` fallback. Run the arm for the
        # current state; a GotoStateStmt inside it transitions `pc`, a
        # ReturnStmt / TailCallStmt unwinds past this loop entirely.
        arms = {arm.state: arm.body for arm in stmt.arms}
        pc = stmt.entry
        for _ in range(1_000_000):
            body = arms.get(pc)
            if body is None:
                raise InterpError(f"dispatch: no arm for state {pc}")
            try:
                _exec_block(body, ctx, trace)
            except _GotoStateSignal as sig:
                pc = sig.state
                continue
            # Every CFG block ends in a transition / return / tail-call,
            # so a dispatch arm never falls off its bottom.
            raise InterpError(
                f"dispatch arm {pc} fell through without a transition"
            )
        raise InterpError("DispatchStmt exceeded 1,000,000 iterations")
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
