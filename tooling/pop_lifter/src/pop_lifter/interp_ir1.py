"""IR1 interpreter — executes lifted IR1 against a 64K byte array.

This is the pass-1 half of the differential-test harness described in
`docs/architecture.md`: the emulator harness will snapshot 6502 RAM at
deterministic checkpoints, and this interpreter must produce the
identical post-state when run over the lifted IR1. Pass 2's structured
IR will use the same interpreter, so we keep the surface narrow and
explicit.

The interpreter is deliberately minimal:

* 64K main RAM (`bytearray`). Aux RAM and bank-switched language card
  pages will land alongside the SMC / hi-res work; the AUTO.S combat
  pilot only touches zero-page and the soft-switch image in main RAM.
* Pseudo-registers `a`, `x`, `y` (8-bit) and a single carry flag `c`.
  Z / N / V land alongside `cmp` and conditional branches in the
  CheckFloor slice.
* Explicit call stack — `jsr` pushes the return point, `rts` pops it.
  An `rts` at depth 0 returns from the run. `jmp` to a routine in any
  loaded module is treated as a tail call: the interpreter switches
  routines without growing the stack.
* Cross-module / jump-table aliasing — `run` accepts either a single
  module or a list, plus an optional `aliases: dict[str, str]` that
  maps Merlin jump-table slot names (e.g. `rnd`) to their concrete
  implementation labels (`RND`). The plan's `jumptables.py` will
  generate that map automatically from the dum blocks; here we accept
  it from the test harness.

`run` returns a `Trace` carrying the post-state plus the set of RAM
addresses that were written. Tests assert against that diff rather than
against the whole 64K so the assertions stay readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ir1 import (
    Abs,
    AdcAbs,
    AdcImm,
    AdcIndexed,
    Asl,
    Bit,
    Bitwise,
    Branch,
    Call,
    Clc,
    CmpAbs,
    CmpImm,
    CmpIndexed,
    CmpIndirect,
    Compare,
    DecTarget,
    FlagOp,
    Goto,
    If,
    Imm,
    IncTarget,
    IndexedAbs,
    IndirectX,
    IndirectY,
    Label,
    LoadAbs,
    LoadImm,
    LoadIndexed,
    LoadIndirect,
    LocalRef,
    Lsr,
    ModuleIR1,
    Nop,
    OpVarRef,
    Pha,
    Pla,
    Reg,
    Return,
    Rol,
    Ror,
    Routine,
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
    StoreOpAddr,
    StoreOpVar,
    Transfer,
    Unsupported,
)
# Re-use the parser's canonical synthetic-address base so the two
# modules can't drift. If the cutoff ever needs to move, it moves in
# `pass0_parse._LABEL_SENTINEL_BASE` and the interpreter follows
# automatically.
from .pass0_parse import _LABEL_SENTINEL_BASE


class InterpError(RuntimeError):
    """Anything the IR1 interpreter cannot honestly execute."""


@dataclass
class Trace:
    """The observable post-state of an interpreter run.

    Flag semantics follow the 6502 datasheet, restricted to the bits
    pass 1 actually needs:

    * `c` — carry. Set by `asl`, `adc`, `cmp`. Read by `adc`, `bcc/bcs`.
    * `z` — zero. Set by every defining op (`lda*/adc/asl/cmp`).
    * `n` — negative (bit 7 of result). Same set of definers as `z`.
    * `v` (overflow) is not modelled; pass 1 never emits `bvc`/`bvs`
      and Mechner's source doesn't either in the sections lifted so far.
    """

    ram: bytearray
    a: int
    x: int
    y: int
    c: int = 0          # carry
    z: int = 0          # zero
    n: int = 0          # negative
    i: int = 0          # interrupt-disable (set/cleared by sei/cli; unread)
    d: int = 0          # decimal mode (set/cleared by sed/cld; unread)
    writes: dict[int, int] = field(default_factory=dict)  # addr -> last value
    steps: int = 0
    max_stack_depth: int = 0
    # PHA/PLA byte stack — see `ir1.Pha` for the two-stack design
    # rationale. Kept distinct from the JSR/RTS call-stack tracking
    # (a local `stack` list inside `run()`) so each routine's pushed
    # bytes can interleave correctly with the call/return frames
    # around them. The call stack isn't a Trace field because it
    # only matters mid-execution; we expose its peak depth as
    # `max_stack_depth` for observability.
    value_stack: list[int] = field(default_factory=list)
    max_value_stack_depth: int = 0
    # Self-modifying-code / local-label store sink. `StoreLocal`
    # writes land here keyed by `(target_label, offset)` rather than
    # into `ram`, because local labels have no resolved address (we
    # don't assemble the program). Nothing reads it back yet — a
    # faithful SMC model that re-routes the patched instruction's
    # operand is future work — but the writes are observable so
    # tests can assert the patch happened. See `ir1.StoreLocal`.
    code_patches: dict[tuple[str, int], int] = field(default_factory=dict)
    # Self-modifying-code *operand variables* (the faithful model). A
    # recognised immediate patch (`StoreOpVar`) writes here keyed by
    # name, and an `Imm` carrying that `opvar` reads it back via
    # `_imm_value` — so the patched instruction actually sees the
    # rewritten operand. Distinct from `code_patches`, which stays the
    # opaque sink for unrecognised `StoreLocal` patches.
    operand_vars: dict[str, int] = field(default_factory=dict)
    # Self-modifying-code *address* operand variables (`StoreOpAddr`). The
    # low / high byte of a patched 16-bit absolute operand live in these
    # two maps, keyed by name; an `Abs` carrying that `opvar` composes its
    # effective base from them (see `_abs_base`), falling back to the
    # assembled `Abs.addr` byte for any half never patched.
    operand_addr_lo: dict[str, int] = field(default_factory=dict)
    operand_addr_hi: dict[str, int] = field(default_factory=dict)

    def diff_against(self, initial: bytes) -> dict[int, int]:
        """Return only the addresses whose byte differs from `initial`.
        Useful for differential assertions that ignore untouched RAM."""
        out: dict[int, int] = {}
        for i in range(min(len(initial), len(self.ram))):
            if self.ram[i] != initial[i]:
                out[i] = self.ram[i]
        return out


def _real_addr(addr: int, src) -> int:
    """Mask `addr` into a 16-bit RAM index, raising `InterpError` if
    the high bits flag it as a synthetic-label sentinel.

    Pass 0 puts every globally-scoped program label into
    `ProgramAST.labels` at `_LABEL_SENTINEL_BASE + i` (0x10000+), so
    the lifter can accept `lda #SymbolicLabel` / `ldx symbol_table,x`
    operands. Those addresses are fine to LIFT (the dump just shows
    the symbolic name), but you can't actually READ or WRITE through
    them — we haven't assembled the program, so the address is
    meaningless in terms of real memory layout.

    Calling convention for indexed accesses: validate the *base*
    here (`_real_addr(item.base.addr, item.src)`) and then apply
    the index + 16-bit wrap yourself:

        base = _real_addr(item.base.addr, item.src)
        addr = (base + (idx_val & 0xff)) & 0xffff

    That way a real high-page base (e.g. `$fff0,x` with `x=$30`)
    wraps to `$0020` instead of falsely tripping the synthetic gate
    on the un-wrapped `$10020` sum. Synthetic bases still raise."""
    if addr >= _LABEL_SENTINEL_BASE:
        where = src.short() if src is not None else "<unknown>"
        raw = repr(src.raw) if src is not None else ""
        raise InterpError(
            f"refusing to access synthetic-label address "
            f"{addr:#x} at {where} ({raw}). "
            f"This address came from pass 0's label table and was "
            f"never resolved to a real assembled location."
        )
    return addr & 0xffff


def _abs_base(operand, trace: Trace, src) -> int:
    """Resolve the base address of an `Abs` operand. For a self-modifying-
    code address operand (`opvar` set) compose the 16-bit base from the
    patched low / high bytes (`Trace.operand_addr_lo`/`_hi`), each falling
    back to the corresponding byte of the assembled `addr` when that half
    was never patched. Otherwise validate the assembled address through
    `_real_addr`."""
    if operand.opvar is not None:
        lo = trace.operand_addr_lo.get(operand.opvar, operand.addr & 0xff)
        hi = trace.operand_addr_hi.get(operand.opvar, (operand.addr >> 8) & 0xff)
        return ((hi << 8) | lo) & 0xffff
    return _real_addr(operand.addr, src)


def _set_zn(trace: Trace, value: int) -> None:
    """Update Z and N from an 8-bit result. Matches the 6502: Z is set
    when the value is exactly zero; N mirrors bit 7."""
    v = value & 0xff
    trace.z = 1 if v == 0 else 0
    trace.n = (v >> 7) & 1


def _imm_value(imm, trace: Trace) -> int:
    """The 8-bit value of an immediate operand. For a self-modifying-code
    operand variable (`imm.opvar` set) read the patched value from
    `trace.operand_vars`, falling back to the assembled `imm.value`
    before any patch; otherwise just the literal."""
    if imm.opvar is not None:
        return trace.operand_vars.get(imm.opvar, imm.value) & 0xff
    return imm.value & 0xff


def _eval_compare(cond: Compare, trace: Trace, ram: bytearray) -> bool:
    """Evaluate a structured Compare against the current register/RAM
    state. The IR2 form is self-contained — no flag inspection needed."""
    reg_val = {Reg.A: trace.a, Reg.X: trace.x, Reg.Y: trace.y}[cond.reg]
    if cond.op in ("<0", ">=0"):
        # Sign tests read the value as a signed byte. The 6502 N flag
        # is just bit 7 of the value, so the structured form mirrors
        # `(reg & 0x80) != 0`.
        is_negative = bool(reg_val & 0x80)
        return is_negative if cond.op == "<0" else not is_negative
    if cond.rhs is None:
        raise InterpError(
            f"Compare op {cond.op!r} requires a rhs but none was supplied"
        )
    if isinstance(cond.rhs, Imm):
        rhs = _imm_value(cond.rhs, trace)
    elif isinstance(cond.rhs, IndexedAbs):
        # Indexed-absolute Compare RHS: validate base then add the
        # 8-bit index. Same shape as the load/store/cmp dispatch in
        # `exec_atom`. Source is None because Compare doesn't carry
        # one; `_indexed_addr` degrades gracefully.
        rhs = ram[_indexed_addr(cond.rhs, trace, None)]
    elif isinstance(cond.rhs, Abs):
        # `_real_addr` raises on synthetic-label dereferences; we
        # pass `None` for the SourceRef because Compare doesn't
        # carry one and the error message degrades gracefully.
        rhs = ram[_real_addr(cond.rhs.addr, None)]
    else:
        raise InterpError(f"unknown Compare rhs type: {type(cond.rhs).__name__}")
    if cond.op == "==":
        return reg_val == rhs
    if cond.op == "!=":
        return reg_val != rhs
    if cond.op == "<":
        return reg_val < rhs
    if cond.op == ">=":
        return reg_val >= rhs
    raise InterpError(f"unknown Compare op: {cond.op!r}")


def _branch_taken(cond: str, trace: Trace) -> bool:
    """Evaluate a `Branch.cond` against the current flag state. Raises
    `InterpError` for conditions whose flags aren't tracked yet."""
    if cond == "eq":
        return trace.z == 1
    if cond == "ne":
        return trace.z == 0
    if cond == "cc":
        return trace.c == 0
    if cond == "cs":
        return trace.c == 1
    if cond == "pl":
        return trace.n == 0
    if cond == "mi":
        return trace.n == 1
    if cond in ("vc", "vs"):
        raise InterpError(
            f"branch condition {cond!r} reads the overflow flag, "
            f"which the interpreter does not track yet"
        )
    raise InterpError(f"unknown branch condition: {cond!r}")


def _find_label_index(routine: Routine, name: str) -> int | None:
    """Return the index of `Label(name=name)` inside `routine.body`, or
    `None` if the label isn't local to this routine. The caller decides
    whether a `None` is fatal or a cue to look across modules."""
    for idx, item in enumerate(routine.body):
        if isinstance(item, Label) and item.name == name:
            return idx
    return None


def _resolve(
    modules: list[ModuleIR1],
    aliases: dict[str, str],
    name: str,
) -> Routine | None:
    """Look up a routine entry by name across every loaded module,
    following the alias map. Returns `None` if nothing matches."""
    seen: set[str] = set()
    cur = name
    while cur not in seen:
        seen.add(cur)
        for m in modules:
            r = m.find(cur)
            if r is not None:
                return r
        if cur in aliases:
            cur = aliases[cur]
            continue
        break
    return None


def _resolve_indirect_y(ind: IndirectY, trace: Trace, ram: bytearray) -> int:
    """Compute the effective address for `(ptr),y`: read the 16-bit
    pointer from `mem[ptr.addr]` (lo) + `mem[ptr.addr+1]` (hi), then
    add Y. Returns a 16-bit address.

    See `IndirectY`'s docstring for the page-wrap caveat — we use
    `(addr + 1) & 0xffff` rather than the NMOS zero-page wrap, which
    matches POP's actual pointer layouts (never sitting at $ff).
    """
    # Validate the base pointer is a real (non-synthetic) address
    # first. The high-byte read at `base + 1` then uses ordinary
    # 16-bit wrap (`0xffff + 1 == 0x0000`) — without this two-step
    # the synthetic-check would falsely reject a `($ff),y` pair whose
    # high byte legitimately lives at $0000. (NMOS would actually
    # read $ff and $00 due to the famous page-wrap bug; we use the
    # cleaner 16-bit wrap, but POP's pointers never sit at $ff so it
    # doesn't matter for our inputs.)
    base = _real_addr(ind.ptr.addr, None)
    lo = ram[base]
    hi = ram[(base + 1) & 0xffff]
    return (((hi << 8) | lo) + (trace.y & 0xff)) & 0xffff


def _resolve_indirect_x(ind: IndirectX, trace: Trace, ram: bytearray) -> int:
    """Compute the effective address for `(ptr,x)` pre-indexed indirect:
    add X to the zero-page pointer location (with zero-page wrap), then
    read the 16-bit pointer there. Both pointer bytes wrap in zero page,
    matching the NMOS 6502."""
    loc = (ind.ptr.addr + trace.x) & 0xff
    lo = ram[loc]
    hi = ram[(loc + 1) & 0xff]
    return ((hi << 8) | lo) & 0xffff


def _indexed_addr(ix: IndexedAbs, trace: Trace, src) -> int:
    """Compute the effective address for an `base,x` / `base,y`
    indexed operand. Mirrors the load/store path: validate the base
    against the synthetic-label gate, then add the 8-bit index and
    wrap the result at 16 bits. See `_real_addr` for why we don't
    push the un-wrapped sum through the gate.
    """
    base = _real_addr(ix.base.addr, src)
    idx_val = trace.x if ix.index is Reg.X else trace.y
    return (base + (idx_val & 0xff)) & 0xffff


def exec_atom(item, trace: Trace, ram: bytearray) -> bool:
    """Execute a single non-control-flow IR1 atom against the supplied
    trace/RAM state. Returns True if the item was handled, False if
    it's a control-flow node the caller should dispatch itself.

    Used by both the IR1 interpreter loop below (via inlined dispatch
    — kept duplicated for hot-path clarity) and by the IR3 interpreter
    in `interp_ir3`, which only needs per-atom semantics and runs its
    own structured control flow.
    """
    if isinstance(item, Label):
        return True
    if isinstance(item, LoadImm):
        value = _imm_value(item.imm, trace)
        if item.reg is Reg.A:
            trace.a = value
        elif item.reg is Reg.X:
            trace.x = value
        else:
            trace.y = value
        _set_zn(trace, value)
        return True
    if isinstance(item, LoadAbs):
        value = ram[_abs_base(item.source, trace, item.src)]
        if item.reg is Reg.A:
            trace.a = value
        elif item.reg is Reg.X:
            trace.x = value
        else:
            trace.y = value
        _set_zn(trace, value)
        return True
    if isinstance(item, LoadIndexed):
        idx_val = trace.x if item.index is Reg.X else trace.y
        addr = ((_abs_base(item.base, trace, item.src) + (idx_val & 0xff)) & 0xffff)
        value = ram[addr]
        if item.reg is Reg.A:
            trace.a = value
        elif item.reg is Reg.X:
            trace.x = value
        else:
            trace.y = value
        _set_zn(trace, value)
        return True
    if isinstance(item, StoreAbs):
        value = {Reg.A: trace.a, Reg.X: trace.x, Reg.Y: trace.y}[item.reg]
        addr = _abs_base(item.target, trace, item.src)
        ram[addr] = value
        trace.writes[addr] = value
        return True
    if isinstance(item, StoreLocal):
        # SMC / local-label store. We don't have a real address for
        # the local label, so the write goes into the side-channel
        # `code_patches` dict rather than `ram`. No flag effects.
        value = {Reg.A: trace.a, Reg.X: trace.x, Reg.Y: trace.y}[item.reg]
        trace.code_patches[(item.target_label, item.offset)] = value
        return True
    if isinstance(item, StoreOpVar):
        # Recognised SMC immediate patch — the faithful model: write the
        # operand variable, which the patched `Imm` reads via `_imm_value`.
        value = {Reg.A: trace.a, Reg.X: trace.x, Reg.Y: trace.y}[item.reg]
        trace.operand_vars[item.name] = value
        return True
    if isinstance(item, StoreOpAddr):
        # Recognised SMC address patch — write the low / high byte of the
        # patched 16-bit operand, which the marked `Abs` reads via
        # `_abs_base`.
        value = {Reg.A: trace.a, Reg.X: trace.x, Reg.Y: trace.y}[item.reg]
        if item.half == "lo":
            trace.operand_addr_lo[item.name] = value
        elif item.half == "hi":
            trace.operand_addr_hi[item.name] = value
        else:
            raise InterpError(
                f"StoreOpAddr.half must be 'lo' or 'hi', got {item.half!r}"
            )
        return True
    if isinstance(item, StoreIndexed):
        value = {Reg.A: trace.a, Reg.X: trace.x, Reg.Y: trace.y}[item.reg]
        idx_val = trace.x if item.index is Reg.X else trace.y
        addr = ((_abs_base(item.base, trace, item.src) + (idx_val & 0xff)) & 0xffff)
        ram[addr] = value
        trace.writes[addr] = value
        return True
    if isinstance(item, Asl):
        old = trace.a & 0xff
        new = (old << 1) & 0xff
        trace.a = new
        trace.c = (old >> 7) & 1
        _set_zn(trace, new)
        return True
    if isinstance(item, Clc):
        trace.c = 0
        return True
    if isinstance(item, Sec):
        trace.c = 1
        return True
    if isinstance(item, Nop):
        return True
    if isinstance(item, FlagOp):
        # Records I/D for completeness; nothing reads them. POP runs
        # in binary mode (cld at boot) and we don't simulate IRQs.
        if item.flag == "I":
            trace.i = item.value
        elif item.flag == "D":
            trace.d = item.value
        else:
            raise InterpError(f"unknown FlagOp flag {item.flag!r}")
        return True
    if isinstance(item, AdcImm):
        total = (trace.a & 0xff) + _imm_value(item.imm, trace) + trace.c
        trace.a = total & 0xff
        trace.c = 1 if total > 0xff else 0
        _set_zn(trace, trace.a)
        return True
    if isinstance(item, AdcAbs):
        total = (trace.a & 0xff) + ram[_real_addr(item.source.addr, item.src)] + trace.c
        trace.a = total & 0xff
        trace.c = 1 if total > 0xff else 0
        _set_zn(trace, trace.a)
        return True
    if isinstance(item, (SbcImm, SbcAbs)):
        # 6502 SBC = A + ~operand + C. C=1 going in means "no borrow"
        # so the chain starts fresh; C=0 propagates a borrow from a
        # previous SBC. Going out, C=1 means "no borrow occurred"
        # (i.e. A >= operand + (1-C_in)).
        operand = (
            _imm_value(item.imm, trace) if isinstance(item, SbcImm)
            else ram[_real_addr(item.source.addr, item.src)]
        )
        total = (trace.a & 0xff) + ((operand ^ 0xff) & 0xff) + trace.c
        trace.a = total & 0xff
        trace.c = 1 if total > 0xff else 0
        _set_zn(trace, trace.a)
        return True
    if isinstance(item, SbcIndirect):
        # `sbc (ptr),y` — same borrow arithmetic as SbcAbs/SbcImm,
        # operand read from the post-indexed-indirect address.
        operand = ram[_resolve_indirect_y(item.source, trace, ram)]
        total = (trace.a & 0xff) + ((operand ^ 0xff) & 0xff) + trace.c
        trace.a = total & 0xff
        trace.c = 1 if total > 0xff else 0
        _set_zn(trace, trace.a)
        return True
    if isinstance(item, Lsr):
        old = trace.a & 0xff
        new = old >> 1
        trace.a = new
        trace.c = old & 1
        # `lsr` shifts in 0 from the top, so N is always 0; Z follows
        # the result. Re-using _set_zn would set N from the high bit
        # of `new`, which is correct because bit 7 of (>>1 result) is
        # always 0 anyway — but we spell it out for clarity.
        trace.n = 0
        trace.z = 1 if new == 0 else 0
        return True
    if isinstance(item, Rol):
        old = trace.a & 0xff
        new = ((old << 1) | (trace.c & 1)) & 0xff
        trace.a = new
        trace.c = (old >> 7) & 1
        _set_zn(trace, new)
        return True
    if isinstance(item, Ror):
        old = trace.a & 0xff
        new = ((old >> 1) | ((trace.c & 1) << 7)) & 0xff
        trace.a = new
        trace.c = old & 1
        _set_zn(trace, new)
        return True
    if isinstance(item, ShiftMem):
        # Memory shift/rotate. The flag effects mirror the accumulator
        # variants; just the operand lives in RAM.
        addr = _real_addr(item.target.addr, item.src)
        old = ram[addr]
        if item.op == "asl":
            new = (old << 1) & 0xff
            trace.c = (old >> 7) & 1
            _set_zn(trace, new)
        elif item.op == "lsr":
            new = old >> 1
            trace.c = old & 1
            trace.n = 0
            trace.z = 1 if new == 0 else 0
        elif item.op == "rol":
            new = ((old << 1) | (trace.c & 1)) & 0xff
            trace.c = (old >> 7) & 1
            _set_zn(trace, new)
        elif item.op == "ror":
            new = ((old >> 1) | ((trace.c & 1) << 7)) & 0xff
            trace.c = old & 1
            _set_zn(trace, new)
        else:
            raise InterpError(f"unknown ShiftMem op {item.op!r}")
        ram[addr] = new
        trace.writes[addr] = new
        return True
    if isinstance(item, Bit):
        operand = (
            _imm_value(item.source, trace) if isinstance(item.source, Imm)
            else ram[_real_addr(item.source.addr, item.src)]
        )
        # Z reflects (A AND operand). N and V come from bits 7 and 6
        # of the operand itself, NOT of the AND result — this is the
        # quirk that makes `bit` useful for status-register probes.
        # A is unchanged.
        trace.z = 1 if (trace.a & operand) == 0 else 0
        trace.n = (operand >> 7) & 1
        # V isn't tracked in Trace yet; nothing reads it. If a future
        # branch consumes V we'll plumb it through.
        return True
    if isinstance(item, Pha):
        trace.value_stack.append(trace.a & 0xff)
        if len(trace.value_stack) > trace.max_value_stack_depth:
            trace.max_value_stack_depth = len(trace.value_stack)
        return True
    if isinstance(item, Pla):
        if not trace.value_stack:
            raise InterpError(
                f"pla on empty value stack at {item.src.short()} "
                f"({item.src.raw!r}) — unbalanced pha/pla?"
            )
        trace.a = trace.value_stack.pop()
        _set_zn(trace, trace.a)
        return True
    if isinstance(item, (CmpImm, CmpAbs)):
        reg_val = {Reg.A: trace.a, Reg.X: trace.x, Reg.Y: trace.y}[item.reg]
        rhs = _imm_value(item.imm, trace) if isinstance(item, CmpImm) \
            else ram[_real_addr(item.source.addr, item.src)]
        diff = (reg_val - rhs) & 0xff
        trace.c = 1 if reg_val >= rhs else 0
        _set_zn(trace, diff)
        return True
    if isinstance(item, (IncTarget, DecTarget)):
        delta = 1 if isinstance(item, IncTarget) else -1
        if isinstance(item.target, Reg):
            # Only X and Y are valid 6502 inc/dec targets — there is
            # no `ina`/`dea` on the stock NMOS chip. The lifter only
            # emits these for X/Y, but if a future caller hand-builds
            # a node with `Reg.A` we want a clean InterpError rather
            # than a KeyError into a register-lookup dict.
            if item.target is Reg.X:
                cur = trace.x
            elif item.target is Reg.Y:
                cur = trace.y
            else:
                raise InterpError(
                    f"{type(item).__name__} on Reg.A is not a valid "
                    f"6502 operation (no ina/dea on NMOS)"
                )
            new = (cur + delta) & 0xff
            if item.target is Reg.X:
                trace.x = new
            else:
                trace.y = new
            _set_zn(trace, new)
        elif isinstance(item.target, OpVarRef):
            # Recognised SMC operand bump (`inc :smod+2` where `:smod` is
            # an operand-var site). Bump the same slot the matching store
            # writes and the patched instruction reads, so the faithful
            # model stays consistent. The store always precedes the bump
            # in practice, so the missing-slot seed (0) isn't normally hit.
            t = item.target
            if t.half is None:
                sink = trace.operand_vars
            elif t.half == "lo":
                sink = trace.operand_addr_lo
            elif t.half == "hi":
                sink = trace.operand_addr_hi
            else:
                raise InterpError(
                    f"OpVarRef.half must be None/'lo'/'hi', got {t.half!r}"
                )
            new = (sink.get(t.name, 0) + delta) & 0xff
            sink[t.name] = new
            _set_zn(trace, new)
        elif isinstance(item.target, LocalRef):
            # Unrecognised SMC operand bump. Route through the
            # code_patches side channel like an opaque StoreLocal; the
            # patched operand has no real address. We seed missing slots
            # at 0 so a bump-before-store reads as 0 (best-effort).
            key = (item.target.label, item.target.offset)
            new = (trace.code_patches.get(key, 0) + delta) & 0xff
            trace.code_patches[key] = new
            _set_zn(trace, new)
        else:
            addr = _real_addr(item.target.addr, item.src)
            new = (ram[addr] + delta) & 0xff
            ram[addr] = new
            trace.writes[addr] = new
            _set_zn(trace, new)
        return True
    if isinstance(item, Transfer):
        value = {Reg.A: trace.a, Reg.X: trace.x, Reg.Y: trace.y}[item.src_reg]
        if item.dst_reg is Reg.A:
            trace.a = value
        elif item.dst_reg is Reg.X:
            trace.x = value
        else:
            trace.y = value
        _set_zn(trace, value)
        return True
    if isinstance(item, Bitwise):
        if isinstance(item.source, Imm):
            rhs = _imm_value(item.source, trace)
        elif isinstance(item.source, IndirectY):
            rhs = ram[_resolve_indirect_y(item.source, trace, ram)]
        elif isinstance(item.source, IndexedAbs):
            rhs = ram[_indexed_addr(item.source, trace, item.src)]
        else:
            rhs = ram[_real_addr(item.source.addr, item.src)]
        if item.op == "and":
            trace.a = trace.a & rhs
        elif item.op == "or":
            trace.a = trace.a | rhs
        elif item.op == "eor":
            trace.a = trace.a ^ rhs
        else:
            raise InterpError(f"unknown Bitwise op {item.op!r}")
        _set_zn(trace, trace.a)
        return True
    if isinstance(item, CmpIndexed):
        # `cmp base,idx` — same semantics as CmpAbs but on the
        # indexed effective address. (cpx/cpy don't have an abs-
        # indexed addressing mode on stock 6502, so `item.reg` is
        # always Reg.A here; see the CmpIndexed dataclass docstring.)
        # Note: `_indexed_addr` validates `base` against the synthetic
        # gate and wraps the sum at 16 bits.
        reg_val = {Reg.A: trace.a, Reg.X: trace.x, Reg.Y: trace.y}[item.reg]
        rhs = ram[_indexed_addr(
            IndexedAbs(base=item.base, index=item.index), trace, item.src
        )]
        diff = (reg_val - rhs) & 0xff
        trace.c = 1 if reg_val >= rhs else 0
        _set_zn(trace, diff)
        return True
    if isinstance(item, AdcIndexed):
        rhs = ram[_indexed_addr(
            IndexedAbs(base=item.base, index=item.index), trace, item.src
        )]
        total = (trace.a & 0xff) + rhs + trace.c
        trace.a = total & 0xff
        trace.c = 1 if total > 0xff else 0
        _set_zn(trace, trace.a)
        return True
    if isinstance(item, SbcIndexed):
        # See SbcImm/SbcAbs for the `A + ~operand + C` borrow trick.
        rhs = ram[_indexed_addr(
            IndexedAbs(base=item.base, index=item.index), trace, item.src
        )]
        total = (trace.a & 0xff) + ((rhs ^ 0xff) & 0xff) + trace.c
        trace.a = total & 0xff
        trace.c = 1 if total > 0xff else 0
        _set_zn(trace, trace.a)
        return True
    if isinstance(item, LoadIndirect):
        if isinstance(item.source, IndirectX):
            addr = _resolve_indirect_x(item.source, trace, ram)
        else:
            addr = _resolve_indirect_y(item.source, trace, ram)
        value = ram[addr]
        trace.a = value
        _set_zn(trace, value)
        return True
    if isinstance(item, StoreIndirect):
        addr = _resolve_indirect_y(item.target, trace, ram)
        ram[addr] = trace.a
        trace.writes[addr] = trace.a
        return True
    if isinstance(item, CmpIndirect):
        addr = _resolve_indirect_y(item.source, trace, ram)
        rhs = ram[addr]
        diff = (trace.a - rhs) & 0xff
        trace.c = 1 if trace.a >= rhs else 0
        _set_zn(trace, diff)
        return True
    if isinstance(item, Unsupported):
        raise InterpError(
            f"refusing to execute unsupported opcode "
            f"{item.mnemonic!r} at {item.src.short()} ({item.src.raw!r})"
        )
    return False


def run(
    module: ModuleIR1 | list[ModuleIR1],
    entry: str,
    *,
    ram: bytearray | None = None,
    a: int = 0,
    x: int = 0,
    y: int = 0,
    c: int = 0,
    aliases: dict[str, str] | None = None,
    max_steps: int = 100_000,
) -> Trace:
    """Execute `entry` starting from the given register/RAM state.

    `module` may be a single `ModuleIR1` or a list — the latter is how
    cross-module calls (e.g. AUTO.S calling into GRAFIX.S's `RND`) get
    resolved. `aliases` adds an extra hop for jump-table slot names
    that don't appear as a label in any module body.

    Stops when an `rts` is executed at call-stack depth 0.
    """
    if ram is None:
        ram = bytearray(0x10000)
    elif len(ram) != 0x10000:
        raise ValueError(f"ram must be 64K, got {len(ram)} bytes")

    modules: list[ModuleIR1] = [module] if isinstance(module, ModuleIR1) else list(module)
    aliases = dict(aliases or {})

    routine = _resolve(modules, aliases, entry)
    if routine is None:
        names = [m.name for m in modules]
        raise InterpError(
            f"unknown entry {entry!r}; not found in any of {names} "
            f"(aliases: {aliases!r})"
        )

    trace = Trace(ram=ram, a=a & 0xff, x=x & 0xff, y=y & 0xff, c=c & 1)
    idx = 0
    body = routine.body
    # Each stack frame remembers the routine we were in and the index
    # of the instruction *after* the `jsr` so `rts` can resume there.
    stack: list[tuple[Routine, int]] = []

    while True:
        if trace.steps >= max_steps:
            raise InterpError(
                f"step limit ({max_steps}) reached in {routine.name!r}; "
                f"likely an unterminated loop"
            )
        if idx >= len(body):
            raise InterpError(
                f"routine {routine.name!r} ran past end of body; missing terminator?"
            )

        item = body[idx]
        trace.steps += 1

        if isinstance(item, Label):
            idx += 1
            continue

        if isinstance(item, LoadImm):
            value = _imm_value(item.imm, trace)
            if item.reg is Reg.A:
                trace.a = value
            elif item.reg is Reg.X:
                trace.x = value
            else:
                trace.y = value
            _set_zn(trace, value)
            idx += 1
            continue

        if isinstance(item, LoadAbs):
            value = ram[_abs_base(item.source, trace, item.src)]
            if item.reg is Reg.A:
                trace.a = value
            elif item.reg is Reg.X:
                trace.x = value
            else:
                trace.y = value
            _set_zn(trace, value)
            idx += 1
            continue

        if isinstance(item, LoadIndexed):
            idx_val = trace.x if item.index is Reg.X else trace.y
            addr = ((_abs_base(item.base, trace, item.src) + (idx_val & 0xff)) & 0xffff)
            value = ram[addr]
            if item.reg is Reg.A:
                trace.a = value
            elif item.reg is Reg.X:
                trace.x = value
            else:
                trace.y = value
            _set_zn(trace, value)
            idx += 1
            continue

        if isinstance(item, StoreAbs):
            value = {
                Reg.A: trace.a,
                Reg.X: trace.x,
                Reg.Y: trace.y,
            }[item.reg]
            addr = _abs_base(item.target, trace, item.src)
            ram[addr] = value
            trace.writes[addr] = value
            idx += 1
            continue

        if isinstance(item, StoreIndexed):
            value = {
                Reg.A: trace.a,
                Reg.X: trace.x,
                Reg.Y: trace.y,
            }[item.reg]
            idx_val = trace.x if item.index is Reg.X else trace.y
            addr = ((_abs_base(item.base, trace, item.src) + (idx_val & 0xff)) & 0xffff)
            ram[addr] = value
            trace.writes[addr] = value
            idx += 1
            continue

        if isinstance(item, Asl):
            old = trace.a & 0xff
            new = (old << 1) & 0xff
            trace.a = new
            trace.c = (old >> 7) & 1
            _set_zn(trace, new)
            idx += 1
            continue

        if isinstance(item, Clc):
            trace.c = 0
            idx += 1
            continue

        if isinstance(item, Sec):
            trace.c = 1
            idx += 1
            continue

        if isinstance(item, AdcImm):
            total = (trace.a & 0xff) + _imm_value(item.imm, trace) + trace.c
            trace.a = total & 0xff
            trace.c = 1 if total > 0xff else 0
            _set_zn(trace, trace.a)
            idx += 1
            continue

        if isinstance(item, AdcAbs):
            total = (trace.a & 0xff) + ram[_real_addr(item.source.addr, item.src)] + trace.c
            trace.a = total & 0xff
            trace.c = 1 if total > 0xff else 0
            _set_zn(trace, trace.a)
            idx += 1
            continue

        if isinstance(item, (CmpImm, CmpAbs)):
            reg_val = {Reg.A: trace.a, Reg.X: trace.x, Reg.Y: trace.y}[item.reg]
            if isinstance(item, CmpImm):
                rhs = _imm_value(item.imm, trace)
            else:
                rhs = ram[_real_addr(item.source.addr, item.src)]
            # 6502 CMP: compute reg - rhs without storing. C = no-borrow
            # (i.e. reg >= rhs); Z = (reg == rhs); N = bit 7 of result.
            diff = (reg_val - rhs) & 0xff
            trace.c = 1 if reg_val >= rhs else 0
            _set_zn(trace, diff)
            idx += 1
            continue

        if isinstance(item, Branch):
            if _branch_taken(item.cond, trace):
                # Local label takes precedence — common case (e.g.
                # `beq :2`). If absent, treat the branch as a
                # conditional tail call into another routine (e.g.
                # CHECKFLOOR's `beq falling`).
                local = _find_label_index(routine, item.target)
                if local is not None:
                    idx = local
                else:
                    target_routine = _resolve(modules, aliases, item.target)
                    if target_routine is None:
                        raise InterpError(
                            f"branch target {item.target!r} not found "
                            f"locally in {routine.name!r} or in any "
                            f"loaded module (aliases: {aliases!r})"
                        )
                    routine = target_routine
                    body = routine.body
                    idx = 0
            else:
                idx += 1
            continue

        if isinstance(item, If):
            # Structured (pass-2-fused) conditional. Same control-flow
            # shape as Branch — local label first, then cross-module
            # tail-call fallback — but the predicate is self-contained.
            if _eval_compare(item.cond, trace, ram):
                local = _find_label_index(routine, item.target)
                if local is not None:
                    idx = local
                else:
                    target_routine = _resolve(modules, aliases, item.target)
                    if target_routine is None:
                        raise InterpError(
                            f"if-target {item.target!r} not found locally "
                            f"in {routine.name!r} or in any loaded module "
                            f"(aliases: {aliases!r})"
                        )
                    routine = target_routine
                    body = routine.body
                    idx = 0
            else:
                idx += 1
            continue

        if isinstance(item, Call):
            target = _resolve(modules, aliases, item.target)
            if target is None:
                raise InterpError(
                    f"jsr target {item.target!r} not found in any loaded "
                    f"module (aliases: {aliases!r})"
                )
            stack.append((routine, idx + 1))
            trace.max_stack_depth = max(trace.max_stack_depth, len(stack))
            routine = target
            body = routine.body
            idx = 0
            continue

        if isinstance(item, Goto):
            if item.kind == "tail_call":
                target = _resolve(modules, aliases, item.target)
                if target is None:
                    raise InterpError(
                        f"tail-call target {item.target!r} not found in any "
                        f"loaded module (aliases: {aliases!r})"
                    )
                routine = target
                body = routine.body
                idx = 0
                continue
            # local goto
            local = _find_label_index(routine, item.target)
            if local is None:
                raise InterpError(
                    f"local goto target {item.target!r} not found in "
                    f"routine {routine.name!r}"
                )
            idx = local
            continue

        if isinstance(item, Return):
            if not stack:
                return trace
            routine, idx = stack.pop()
            body = routine.body
            continue

        if isinstance(item, Unsupported):
            raise InterpError(
                f"refusing to execute unsupported opcode "
                f"{item.mnemonic!r} at {item.src.short()} ({item.src.raw!r})"
            )

        # Atoms added by later slices (IncTarget / DecTarget / Transfer /
        # Bitwise / ...) are handled by the shared `exec_atom` helper.
        # That keeps both interpreters on a single per-opcode dispatch
        # so they can't drift; we only need bespoke handling in this
        # loop for the control-flow nodes above (Branch, If, Goto,
        # Call, Return).
        if exec_atom(item, trace, ram):
            idx += 1
            continue

        raise InterpError(f"unknown IR1 item: {type(item).__name__}")
