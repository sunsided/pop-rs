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

from .interp_ir1 import (
    InterpError,
    Trace,
    _branch_taken,
    _eval_compare,
    _indexed_addr,
    _real_addr,
    _resolve,
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
    RestoreTemp,
    ReturnStmt,
    RotateExpr,
    RoutineIR3,
    SaveTemp,
    Stmt,
    TailCallStmt,
    Wide16Stmt,
)


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


class _ContinueSignal(Exception):
    """Raised by ContinueStmt to jump to the top of the innermost
    enclosing LoopStmt."""


def run(
    modules: list[ModuleIR3 | ModuleIR1],
    entry: str,
    *,
    ram: bytearray | None = None,
    aliases: dict[str, str] | None = None,
) -> Trace:
    """Execute `entry` in the given module set (mix of IR3 / IR1).

    Tail-call chaining is handled by an outer loop so deeply-nested
    cross-module tails don't stack the Python recursion. Calls and
    nested blocks recurse normally.
    """
    if ram is None:
        ram = bytearray(0x10000)
    trace = Trace(ram=ram, a=0, x=0, y=0)
    alias_idx = _alias_index(modules)
    if aliases:
        alias_idx.update(aliases)

    routine = _resolve(modules, alias_idx, entry)
    if routine is None:
        raise InterpError(f"entry {entry!r} not found in any module")

    while True:
        try:
            _exec_routine(routine, modules, alias_idx, trace)
            return trace
        except _TailCallSignal as tc:
            target = _resolve(modules, alias_idx, tc.target)
            if target is None:
                raise InterpError(
                    f"tail-call target {tc.target!r} not found in any module"
                )
            routine = target
            continue


def _alias_index(modules) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in modules:
        for r in m.routines:
            out[r.name] = r.name
            for a in getattr(r, "entry_aliases", []) or []:
                out[a] = r.name
    return out


def _exec_routine(routine, modules, aliases, trace: Trace) -> None:
    if isinstance(routine, RoutineIR3):
        try:
            _exec_block(routine.body, modules, aliases, trace)
        except _ReturnSignal:
            pass
        return
    if isinstance(routine, IR1Routine):
        # Delegate to the IR1 interpreter for IR1 leaves. We share the
        # RAM bytearray via `trace.ram`, so observable state stays in
        # sync.
        ir1_modules = [m for m in modules if isinstance(m, ModuleIR1)]
        # We need the IR1 routine reachable via the same alias map.
        # Re-running `ir1_run` with the same alias index keeps the
        # behaviour identical to "this routine in isolation".
        ir1_run(
            ir1_modules,
            routine.name,
            ram=trace.ram,
            a=trace.a, x=trace.x, y=trace.y, c=trace.c,
            aliases=aliases,
        )
        return
    raise InterpError(f"unknown routine type: {type(routine).__name__}")


def _exec_block(block: Block, modules, aliases, trace: Trace) -> None:
    for stmt in block.stmts:
        _exec_stmt(stmt, modules, aliases, trace)


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


def _exec_stmt(stmt: Stmt, modules, aliases, trace: Trace) -> None:
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
        callee = _resolve(modules, aliases, stmt.target)
        if callee is None:
            raise InterpError(f"call target {stmt.target!r} not found")
        _exec_routine(callee, modules, aliases, trace)
        return
    if isinstance(stmt, TailCallStmt):
        raise _TailCallSignal(stmt.target)
    if isinstance(stmt, ReturnStmt):
        raise _ReturnSignal()
    if isinstance(stmt, IfStmt):
        if _eval_compare(stmt.cond, trace, trace.ram):
            _exec_block(stmt.then_block, modules, aliases, trace)
        elif stmt.else_block is not None:
            _exec_block(stmt.else_block, modules, aliases, trace)
        return
    if isinstance(stmt, MatchStmt):
        reg_val = {Reg.A: trace.a, Reg.X: trace.x, Reg.Y: trace.y}[stmt.reg]
        for arm in stmt.arms:
            if any((v.value & 0xff) == reg_val for v in arm.values):
                # The arm terminates (return / tail-call / break / ...),
                # so this raises the matching signal; if it somehow falls
                # off, returning here matches the no-match fall-through.
                _exec_block(arm.body, modules, aliases, trace)
                return
        return  # no arm matched — fall through to the next statement
    if isinstance(stmt, RawIfStmt):
        if _branch_taken(stmt.cond, trace):
            _exec_block(stmt.then_block, modules, aliases, trace)
        elif stmt.else_block is not None:
            _exec_block(stmt.else_block, modules, aliases, trace)
        return
    if isinstance(stmt, LoopStmt):
        # Bounded so a busted exit guard doesn't hang the interpreter.
        # 6502 routines we're modelling don't iterate more than ~1024
        # times in practice — a million is room to spare for tests.
        for _ in range(1_000_000):
            try:
                _exec_block(stmt.body, modules, aliases, trace)
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
                _exec_block(stmt.body, modules, aliases, trace)
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
                _exec_block(stmt.body, modules, aliases, trace)
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
            _exec_block(stmt.body, modules, aliases, trace)
            exec_atom(step_op, trace, trace.ram)
        return
    if isinstance(stmt, BreakStmt):
        raise _BreakSignal()
    if isinstance(stmt, ContinueStmt):
        raise _ContinueSignal()
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
