"""Pass 4: emit Rust source from IR3.

Earlier slices lowered module / routine scaffolding, leaf expressions
(folded `Assign`s, `return`), *all structured control flow*,
*data-movement `RawStmt` atoms* (loads, stores, transfers, bitwise,
inc/dec), and the *carry-arithmetic atoms* (`clc`/`sec`/`adc`/`sbc`/
shifts, over `self.c: u8`).

This slice adds *post-indexed indirect addressing* — the `(ptr),y`
form. The effective address is a 16-bit zero-page pointer fetch plus Y:

    self.mem[(self.mem[ptr] as usize
              | (self.mem[ptr + 1] as usize) << 8) + self.reg.y as usize]

lowered through one `_indirect_index` helper and reused by every site
that can carry an `IndirectY`:

* folded `Assign` source / target (e.g. `dst[y] = (ptr),y`);
* `LoadIndirect` (`lda (ptr),y`) / `StoreIndirect` (`sta (ptr),y`);
* `Bitwise` with a `(ptr),y` source (`and`/`ora`/`eor (ptr),y`);
* `SbcIndirect` (`sbc (ptr),y`), via the same `A + ~operand + C` trick.

The high byte renders as a `ptr + 1` operand so it resolves to
`sym::<ptr> + 1` when the base is named. No page wrap on the pointer
and no 16-bit wrap on `+ Y`, matching the interpreter's permissive rule
and the unmasked `IndexedAbs` lowering.

This slice also lowers the *flag-only comparison atoms* — the standalone
`cmp`/`cpx`/`cpy` and `bit` instructions pass 2 couldn't fuse into a
branch — over a provisional `self.z` / `self.n` model (joining
`self.c`):

* `CmpImm` / `CmpAbs` / `CmpIndexed` / `CmpIndirect` → `c = reg >= op`,
  `z = reg == op`, `n = (reg - op) >> 7`;
* `Bit` → `z = (a & op) == 0`, `n = op >> 7`. The `V` flag is *not*
  modeled (the lifter never tracks it: `bvc`/`bvs` surface as
  `Unsupported`), so `bit` writes only Z and N.

It also lowers `Wide16Stmt` — the recognised 16-bit `add`/`subtract`
idiom — to two chained byte ops over `u16`: the low byte's bit-8 feeds
the high byte (subtract via the `src + ~op + 1` identity), then `A` and
`C` take the high-byte result and carry-out, preserving both stores.

This slice also lowers the relooper's *dispatch fallback* — the
`DispatchStmt` (`loop { match pc { ... } }`) that pass 2 now emits for
routines it can't reduce to natural loops/conditionals (irreducible
flow, multi-back-edge loops, mid-body exits). Each numbered state is a
`match` arm holding the block's atoms; edges become `pc = <state>;`
transitions (`GotoStateStmt`). This replaces the old `GotoStmt` /
`LabelStmt` escape hatch, so those routines now emit valid structured
Rust instead of unresolved gotos.

This slice also lowers the *stack atoms* — the `pha`/`pla` that pass 3
couldn't fold into a scoped `SaveTemp`/`RestoreTemp` pair — over a
provisional `self.stack: Vec<u8>` that mirrors the interpreter's
`value_stack`:

* `Pha` → `self.stack.push(self.reg.a)`;
* `Pla` → `self.reg.a = self.stack.pop()…` plus Z/N from the popped byte.

It also lowers the *recognised SMC operand patches*. `pass3_smc` rewrites
runtime instruction-operand patches into a faithful model:

* immediate operand — `sta :label+1` patching the `#imm` of
  `lda/adc/sbc/cmp` → `StoreOpVar` + `Imm.opvar`: `self.<name> =
  self.<reg>` writes the operand variable, and the patched immediate
  reads `self.<name>`.
* 16-bit address operand — `sta :label+1`/`+2` patching the absolute
  operand of `lda/sta abs[,x/y]` → `StoreOpAddr` + `Abs.opvar`:
  `self.<name>_lo`/`_hi = self.<reg>` write the byte halves, and the
  patched access composes `((hi << 8) | lo)` as its base.

Still deferred:
* the opaque `StoreLocal` (opcode patches at offset 0, branch / jump
  target patches) and `LocalRef` inc/dec — self-modifying code with no
  consumer model yet.

State model: this slice emits a decomposed `Cpu` receiver and its
sub-structs (see `_emit_state_defs`) — registers `self.reg.{a,x,y}: u8`,
status flags `self.flags.{c,z,n,i,d}: bool` (V untracked), 64 KiB memory
`self.mem: Box<[u8; 0x10000]>`, the `pha`/`pla` value stack
`self.stack: Vec<u8>`, and recognised SMC operand variables under
`self.smc`. Each emitted file defines these types so it is self-contained
for state; resolving cross-module routine *calls* into one crate is a
separate later concern.
"""

from __future__ import annotations

import re

from .ir1 import (
    Abs,
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
    Compare,
    DecTarget,
    FlagOp,
    Imm,
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
    StoreOpAddr,
    StoreOpVar,
    Transfer,
)
from .ir3 import (
    Assign,
    BinExpr,
    BreakStmt,
    CallStmt,
    ContinueStmt,
    DispatchStmt,
    DoWhileStmt,
    ForStmt,
    GotoStateStmt,
    IfStmt,
    LabeledBlock,
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
    TailCallStmt,
    Wide16Stmt,
)

INDENT = "    "

# Statement types that produce real Rust code (not a comment placeholder).
_LOWERED_TYPES = (
    Assign, ReturnStmt,
    IfStmt, RawIfStmt, LoopStmt, DoWhileStmt, ForStmt, RepeatStmt,
    BreakStmt, ContinueStmt, MatchStmt,
    CallStmt, TailCallStmt,
    SaveTemp, RestoreTemp, Wide16Stmt, LabeledBlock,
    DispatchStmt, GotoStateStmt,
)


# ------------------------------------------------------------- method names


_RUST_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _mangle(name: str) -> str:
    """Map a 6502 routine/alias name to a valid Rust method identifier.

    Names that are already valid identifiers pass through unchanged (the
    common case), so existing output is untouched. Otherwise every
    non-identifier byte — the Merlin `:local` / `]macro-local` sigils,
    `$addr` call targets, `*`, `?` — is escaped as `_<hex>`. The escape is
    injective, so two distinct names never collide, and a leading digit is
    prefixed with `_`."""
    if _RUST_IDENT.fullmatch(name):
        return name
    out: list[str] = []
    for ch in name:
        if ch.isascii() and (ch.isalnum() or ch == "_"):
            out.append(ch)
        else:
            out.append(f"_{ord(ch):02x}")
    s = "".join(out) or "_"
    if s[0].isdigit():
        s = "_" + s
    return s


def _build_name_table(modules: list[ModuleIR3]) -> dict[str, str]:
    """Map every routine name *and entry alias* to that routine's
    canonical name, across all modules being emitted together. Call
    targets resolve through this so a `jsr`/`jmp` to an alias (e.g. the
    Merlin local `:next`, an alias of `ANIMCHAR`) emits the canonical
    method instead of an undefined alias method."""
    table: dict[str, str] = {}
    for module in modules:
        for r in module.routines:
            table.setdefault(r.name, r.name)
            for alias in r.entry_aliases:
                table.setdefault(alias, r.name)
    return table


def _method_name(target: str, names: dict[str, str] | None) -> str:
    """Resolve a call target to its canonical routine name (if known)
    then mangle it to a valid Rust method identifier."""
    canonical = names.get(target, target) if names else target
    return _mangle(canonical)


# ---------------------------------------------------------------- values


def _addr(addr: int) -> str:
    return f"0x{addr:04x}"


def _emit_imm(imm: Imm) -> str:
    # An opvar immediate is the runtime-patched operand of a
    # self-modifying-code instruction (recognised by `pass3_smc`); it
    # reads the provisional operand-variable field the matching
    # `StoreOpVar` writes, rather than the placeholder assembled byte.
    if imm.opvar is not None:
        return f"self.smc.{_mangle(imm.opvar)}"
    return f"0x{imm.value & 0xff:02x}"


def _imm_u8(imm: Imm) -> str:
    """Render an immediate as a `u8`-typed expression safe to bitwise-
    complement (`!`). A literal gets an explicit `_u8` suffix so `!`
    complements at byte width (a bare `!0xNN` would default to `i32` and
    cast wrong); an opvar immediate is already a `u8` field, so it's
    returned as-is — appending `_u8` there would mangle the field name
    into the invalid `self.<opvar>_u8`."""
    if imm.opvar is not None:
        return f"self.smc.{_mangle(imm.opvar)}"
    return f"0x{imm.value & 0xff:02x}_u8"


# A symbolic name is a Rust-ident base with an optional `+N` / `-N`
# offset (Merlin's `ztemp+1` form for a 16-bit pointer's high byte).
_SYM_NAME_RE = re.compile(r"^(?P<base>[A-Za-z_][A-Za-z0-9_]*)(?P<off>[+-]\d+)?$")


class SymTable:
    """Recovers symbolic RAM-address constants from the `Abs` operands a
    module references, so `self.mem[0x00a0]` can render as
    `self.mem[sym::PlayCount]` and keep the source's intent.

    Built in two phases because emission is the only place that knows
    which addresses are actually rendered as an index:

    1. *record*: emit the module once with a fresh table; every `Abs`
       reaching `index()` is noted and rendered as a bare literal.
    2. `finalize()`: resolve each `name(+off)` to a `sym::<base>` const
       whose value is `addr - off`. A base that resolves to two
       different addresses, or a name that isn't a clean ident, falls
       back to the literal so the output never lies.
    3. *render*: emit again; `index()` now returns the `sym::` form."""

    def __init__(self) -> None:
        self._seen: list[tuple[str, int]] = []
        self._consts: dict[str, int] = {}
        self._final = False

    @staticmethod
    def _parse(name: str) -> tuple[str, int] | None:
        m = _SYM_NAME_RE.match(name)
        if m is None:
            return None
        return m.group("base"), int(m.group("off") or 0)

    def index(self, a: Abs) -> str:
        """Render `a` as the `self.mem[...]` index expression."""
        if not self._final:
            self._seen.append((a.name, a.addr))
            return _addr(a.addr)
        parsed = self._parse(a.name)
        if parsed is not None:
            base, off = parsed
            if self._consts.get(base) == a.addr - off:
                if off == 0:
                    return f"sym::{base}"
                return f"sym::{base} {'+' if off > 0 else '-'} {abs(off)}"
        return _addr(a.addr)

    def finalize(self) -> None:
        candidates: dict[str, int] = {}
        conflicted: set[str] = set()
        for name, addr in self._seen:
            parsed = self._parse(name)
            if parsed is None:
                continue
            base, base_addr = parsed[0], addr - parsed[1]
            if base in candidates and candidates[base] != base_addr:
                conflicted.add(base)
            else:
                candidates.setdefault(base, base_addr)
        self._consts = {b: a for b, a in candidates.items() if b not in conflicted}
        self._final = True

    def render_block(self, indent: str) -> list[str]:
        if not self._consts:
            return []
        lines = [
            f"{indent}#[allow(non_upper_case_globals)]",
            f"{indent}mod sym {{",
        ]
        for name in sorted(self._consts):
            lines.append(f"{indent}    pub const {name}: usize = {_addr(self._consts[name])};")
        lines.append(f"{indent}}}")
        return lines


def _abs_index(a: Abs, syms: SymTable | None) -> str:
    """Render the index of `self.mem[<index>]` for an absolute address —
    a `sym::` reference when a symbol table resolves it, else the
    `0x..` literal.

    An SMC address operand (`a.opvar` set) instead composes its runtime
    16-bit base from the low / high byte fields the matching `StoreOpAddr`
    writes — but only for the halves actually patched (`a.opvar_halves`);
    an un-patched half bakes the assembled `addr` byte, matching the
    interpreter's per-byte fallback instead of reading an uninitialised
    field."""
    if a.opvar is not None:
        name = _mangle(a.opvar)
        hi = (f"self.smc.{name}_hi as usize" if "hi" in a.opvar_halves
              else f"0x{(a.addr >> 8) & 0xff:02x}")
        lo = (f"self.smc.{name}_lo as usize" if "lo" in a.opvar_halves
              else f"0x{a.addr & 0xff:02x}")
        return f"(({hi}) << 8 | ({lo}))"
    if syms is None:
        return _addr(a.addr)
    return syms.index(a)


def _offset_abs(a: Abs, delta: int) -> Abs:
    """Return `a` shifted by `delta` bytes, keeping a name that
    `SymTable` can still resolve. If `a.name` already parses as
    `base(+/-off)`, fold `delta` into that offset (so `ztemp+1` + 1
    becomes `ztemp+2`, and a net-zero offset drops back to `base`);
    otherwise append a fresh offset to the raw name."""
    parsed = SymTable._parse(a.name)
    addr = (a.addr + delta) & 0xFFFF
    if parsed is not None:
        base, off = parsed
        new_off = off + delta
        name = base if new_off == 0 else f"{base}{new_off:+d}"
    else:
        name = f"{a.name}{delta:+d}"
    return Abs(name=name, addr=addr)


def _indirect_index(iy: IndirectY, syms: SymTable | None) -> str:
    """Render the `self.mem[<index>]` index for a `(ptr),y` effective
    address: fetch the 16-bit pointer from the zero-page bytes at `ptr`
    (low) and `ptr + 1` (high), then add Y. The high byte is rendered via
    `_offset_abs(ptr, 1)` so it resolves to `sym::<base> + <off+1>` even
    when `ptr` itself carries an offset (`(ztemp+1),y`). Matches the
    interpreter's permissive rule (no page wrap on the pointer, no 16-bit
    wrap on `+ Y`) and the unmasked `IndexedAbs` form."""
    lo = f"self.mem[{_abs_index(iy.ptr, syms)}]"
    hi = f"self.mem[{_abs_index(_offset_abs(iy.ptr, 1), syms)}]"
    return f"({lo} as usize | ({hi} as usize) << 8) + self.reg.y as usize"


def _cmp_operand(item, syms: SymTable | None) -> str:
    """Render the compared byte for a `cmp`/`cpx`/`cpy` atom as a u8
    r-value, across the immediate / absolute / indexed / indirect
    addressing forms."""
    if isinstance(item, CmpImm):
        return _emit_imm(item.imm)
    if isinstance(item, CmpAbs):
        return f"self.mem[{_abs_index(item.source, syms)}]"
    if isinstance(item, CmpIndexed):
        return f"self.mem[{_abs_index(item.base, syms)} + self.reg.{item.index} as usize]"
    return f"self.mem[{_indirect_index(item.source, syms)}]"  # CmpIndirect


def _wide_term(operand, syms: SymTable | None, *, complement: bool) -> str:
    """Render one byte operand of a `Wide16Stmt` as a `u16` term. With
    `complement` (the subtract path), emit `!operand` at byte width so it
    feeds the `src + ~op + carry` identity. A literal immediate goes
    through `_imm_u8` for an explicit byte-width complement; every other
    operand (memory reads, opvar fields) is already `u8`, so a bare `!`
    is correct."""
    if complement:
        v = f"!{_imm_u8(operand)}" if isinstance(operand, Imm) else f"!{_emit_value(operand, syms)}"
    else:
        v = _emit_value(operand, syms)
    return f"({v} as u16)"


def _emit_value(v, syms: SymTable | None = None) -> str:
    """Render an `Assign` source or compare RHS as a Rust r-value."""
    if isinstance(v, Imm):
        return _emit_imm(v)
    if isinstance(v, Abs):
        return f"self.mem[{_abs_index(v, syms)}]"
    if isinstance(v, IndexedAbs):
        return f"self.mem[{_abs_index(v.base, syms)} + self.reg.{v.index} as usize]"
    if isinstance(v, IndirectY):
        return f"self.mem[{_indirect_index(v, syms)}]"
    if isinstance(v, BinExpr):
        return _emit_binexpr(v, syms)
    if isinstance(v, RotateExpr):
        # rotl/rotr read the carry flag, so they are methods on the CPU.
        return f"self.{v.op}({_emit_value(v.operand, syms)}, {v.count})"
    raise ValueError(f"unknown Assign source type: {type(v).__name__}")


def _emit_binexpr(v: BinExpr, syms: SymTable | None = None) -> str:
    lhs = _emit_value(v.lhs, syms)
    rhs = _emit_value(v.rhs, syms)
    if v.op == "+":
        return f"({lhs}).wrapping_add({rhs})"
    if v.op == "-":
        return f"({lhs}).wrapping_sub({rhs})"
    if v.op == "<<":
        return f"({lhs}).wrapping_shl({rhs} as u32)"
    if v.op == ">>":
        return f"({lhs}).wrapping_shr({rhs} as u32)"
    raise ValueError(f"unknown BinExpr op: {v.op!r}")


def _emit_target(target, syms: SymTable | None = None) -> str | None:
    """Render an `Assign` target as a Rust assignable place, or `None`
    when the destination form isn't lowered yet."""
    if isinstance(target, Abs):
        return f"self.mem[{_abs_index(target, syms)}]"
    if isinstance(target, IndexedAbs):
        return f"self.mem[{_abs_index(target.base, syms)} + self.reg.{target.index} as usize]"
    if isinstance(target, IndirectY):
        return f"self.mem[{_indirect_index(target, syms)}]"
    return None  # anything else: deferred


def _emit_compare(c: Compare, syms: SymTable | None = None) -> str:
    """Render a pass-3 `Compare` as a Rust boolean expression.

    * `rhs=None` is a sign test (N-flag): `op` is `">=0"` or `"<0"`.
      Cast the register to `i8` so the signed comparison is natural.
    * `rhs=Imm` with `==`/`!=` is a zero / equality test (unsigned ok).
    * `rhs=Imm` with `<`/`>=` comes from `cmp; bcc/bcs` — unsigned,
      so plain u8 comparison is correct.
    * `rhs=Abs` / `rhs=IndexedAbs`: compare against a memory byte."""
    reg = f"self.reg.{c.reg}"
    if c.rhs is None:
        # Sign test (N-flag): op is ">=0" (bpl) or "<0" (bmi). Validate
        # explicitly so an unexpected op surfaces rather than silently
        # emitting the wrong predicate.
        if c.op == ">=0":
            op_str = ">= 0"
        elif c.op == "<0":
            op_str = "< 0"
        else:
            raise ValueError(f"unexpected sign-test Compare op: {c.op!r}")
        return f"({reg} as i8) {op_str}"
    rhs = _emit_value(c.rhs, syms)
    return f"{reg} {c.op} {rhs}"


# A `RawIfStmt.cond` is a raw 6502 branch suffix — the flag test pass 2
# couldn't fuse into a `Compare`. Map each to the `bool` flag model
# (`self.flags.z`/`.c`/`.n`), mirroring `interp_ir1._branch_taken`.
# The overflow flag (`vs`/`vc`) isn't tracked anywhere in the lifter, so
# it has no entry; `_emit_branch_cond` flags it rather than guessing.
_BRANCH_COND_RS = {
    "eq": "self.flags.z",
    "ne": "!self.flags.z",
    "cs": "self.flags.c",
    "cc": "!self.flags.c",
    "mi": "self.flags.n",
    "pl": "!self.flags.n",
}


def _emit_branch_cond(cond: str) -> str:
    rs = _BRANCH_COND_RS.get(cond)
    if rs is not None:
        return rs
    # Overflow flag not modeled — keep the routine compiling but mark it.
    return f"false /* TODO(pass4): branch on {cond} (V flag not modeled) */"


# ---------------------------------------------------------------- raw atoms


_BITWISE_OPS = {"and": "&", "or": "|", "eor": "^"}


def _emit_raw(item, syms: SymTable | None = None) -> list[str] | None:
    """Lower an unfolded IR1 atom to Rust statement fragments (no
    indentation), or return `None` when the atom isn't lowered yet so
    the caller falls back to a `// raw:` comment.

    Covers register/memory moves, carry arithmetic (adc/sbc/shifts),
    `(ptr),y` indirect loads/stores/bitwise/sbc, the flag-only
    comparisons (cmp/cpx/cpy/bit), the unpaired stack `pha`/`pla`, and
    the recognised SMC operand patches (`StoreOpVar` immediate +
    `StoreOpAddr` 16-bit address). The opaque `StoreLocal` (offset-0
    opcode patches, branch / jump-target patches) stays deferred."""
    if isinstance(item, LoadImm):
        return [f"self.reg.{item.reg} = {_emit_imm(item.imm)};"]

    if isinstance(item, LoadAbs):
        return [f"self.reg.{item.reg} = self.mem[{_abs_index(item.source, syms)}];"]

    if isinstance(item, LoadIndexed):
        place = f"self.mem[{_abs_index(item.base, syms)} + self.reg.{item.index} as usize]"
        return [f"self.reg.{item.reg} = {place};"]

    if isinstance(item, LoadIndirect):
        return [f"self.reg.{item.reg} = self.mem[{_indirect_index(item.source, syms)}];"]

    if isinstance(item, StoreAbs):
        return [f"self.mem[{_abs_index(item.target, syms)}] = self.reg.{item.reg};"]

    if isinstance(item, StoreIndexed):
        place = f"self.mem[{_abs_index(item.base, syms)} + self.reg.{item.index} as usize]"
        return [f"{place} = self.reg.{item.reg};"]

    if isinstance(item, StoreIndirect):
        return [f"self.mem[{_indirect_index(item.target, syms)}] = self.reg.{item.reg};"]

    if isinstance(item, Transfer):
        return [f"self.reg.{item.dst_reg} = self.reg.{item.src_reg};"]

    if isinstance(item, Bitwise):
        op = _BITWISE_OPS.get(item.op)
        if op is None or not isinstance(item.source, (Imm, Abs, IndexedAbs, IndirectY)):
            return None
        return [f"self.reg.a {op}= {_emit_value(item.source, syms)};"]

    if isinstance(item, (IncTarget, DecTarget)):
        method = "wrapping_add" if isinstance(item, IncTarget) else "wrapping_sub"
        target = item.target
        if isinstance(target, Reg):
            return [f"self.reg.{target} = self.reg.{target}.{method}(1);"]
        if isinstance(target, Abs):
            place = f"self.mem[{_abs_index(target, syms)}]"
            return [f"{place} = {place}.{method}(1);"]
        return None  # LocalRef: self-modifying-code operand bump — deferred

    # ---- carry / flag operations ----------------------------------------

    if isinstance(item, Clc):
        return ["self.flags.c = false;"]

    if isinstance(item, Sec):
        return ["self.flags.c = true;"]

    if isinstance(item, FlagOp):
        flag = item.flag.lower()
        return [f"self.flags.{flag} = {'true' if item.value else 'false'};"]

    if isinstance(item, (AdcImm, AdcAbs, AdcIndexed)):
        if isinstance(item, AdcImm):
            rhs = f"({_emit_imm(item.imm)}) as u16"
        elif isinstance(item, AdcAbs):
            rhs = f"self.mem[{_abs_index(item.source, syms)}] as u16"
        else:
            rhs = f"self.mem[{_abs_index(item.base, syms)} + self.reg.{item.index} as usize] as u16"
        return [
            f"let _r = (self.reg.a as u16) + {rhs} + (self.flags.c as u16);",
            "self.reg.a = _r as u8;",
            "self.flags.c = (_r >> 8) != 0;",
        ]

    if isinstance(item, (SbcImm, SbcAbs, SbcIndexed, SbcIndirect)):
        # 6502 SBC uses the A + ~operand + C identity so the borrow
        # convention (C=1 means "no borrow") falls out naturally.
        if isinstance(item, SbcImm):
            # `_imm_u8` forces a literal complement to byte width (a bare
            # `!0xbd` defaults to i32 = -190 and casts to the wrong u16);
            # an opvar immediate is already a u8 field, so no `_u8`.
            rhs = f"(!{_imm_u8(item.imm)}) as u16"
        elif isinstance(item, SbcAbs):
            rhs = f"(!self.mem[{_abs_index(item.source, syms)}]) as u16"
        elif isinstance(item, SbcIndexed):
            rhs = f"(!self.mem[{_abs_index(item.base, syms)} + self.reg.{item.index} as usize]) as u16"
        else:
            rhs = f"(!self.mem[{_indirect_index(item.source, syms)}]) as u16"
        return [
            f"let _r = (self.reg.a as u16) + {rhs} + (self.flags.c as u16);",
            "self.reg.a = _r as u8;",
            "self.flags.c = (_r >> 8) != 0;",
        ]

    if isinstance(item, Asl):
        return [
            "self.flags.c = (self.reg.a >> 7) != 0;",
            "self.reg.a = self.reg.a.wrapping_shl(1);",
        ]

    if isinstance(item, Lsr):
        return [
            "self.flags.c = (self.reg.a & 1) != 0;",
            "self.reg.a = self.reg.a.wrapping_shr(1);",
        ]

    if isinstance(item, Rol):
        return [
            "let _c = self.reg.a >> 7;",
            "self.reg.a = self.reg.a.wrapping_shl(1) | (self.flags.c as u8);",
            "self.flags.c = _c != 0;",
        ]

    if isinstance(item, Ror):
        return [
            "let _c = self.reg.a & 1;",
            "self.reg.a = self.reg.a.wrapping_shr(1) | ((self.flags.c as u8) << 7);",
            "self.flags.c = _c != 0;",
        ]

    if isinstance(item, ShiftMem):
        place = f"self.mem[{_abs_index(item.target, syms)}]"
        if item.op == "asl":
            return [
                f"self.flags.c = ({place} >> 7) != 0;",
                f"{place} = {place}.wrapping_shl(1);",
            ]
        if item.op == "lsr":
            return [
                f"self.flags.c = ({place} & 1) != 0;",
                f"{place} = {place}.wrapping_shr(1);",
            ]
        if item.op == "rol":
            return [
                f"let _c = {place} >> 7;",
                f"{place} = {place}.wrapping_shl(1) | (self.flags.c as u8);",
                "self.flags.c = _c != 0;",
            ]
        if item.op == "ror":
            return [
                f"let _c = {place} & 1;",
                f"{place} = {place}.wrapping_shr(1) | ((self.flags.c as u8) << 7);",
                "self.flags.c = _c != 0;",
            ]

    # ---- flag-only comparisons ------------------------------------------

    if isinstance(item, (CmpImm, CmpAbs, CmpIndexed, CmpIndirect)):
        reg = f"self.reg.{item.reg}"
        return [
            f"let _o: u8 = {_cmp_operand(item, syms)};",
            f"self.flags.c = {reg} >= _o;",
            f"self.flags.z = {reg} == _o;",
            f"self.flags.n = ({reg}.wrapping_sub(_o) >> 7) != 0;",
        ]

    if isinstance(item, Bit):
        if isinstance(item.source, Imm):
            operand = _emit_imm(item.source)
        else:
            operand = f"self.mem[{_abs_index(item.source, syms)}]"
        return [
            f"let _o: u8 = {operand};",
            "self.flags.z = (self.reg.a & _o) == 0;",
            "self.flags.n = (_o >> 7) != 0;",
        ]

    # ---- stack ----------------------------------------------------------
    # The `pha`/`pla` pass 3 couldn't pair into a scoped `SaveTemp`/
    # `RestoreTemp` (unbalanced within the routine, or the pair straddles
    # a structured boundary). Lower over a provisional value stack —
    # `self.stack: Vec<u8>` — mirroring the interpreter's `value_stack`.
    # `pla` sets Z/N from the popped byte, matching the 6502 / interpreter.

    if isinstance(item, Pha):
        return ["self.stack.push(self.reg.a);"]

    if isinstance(item, Pla):
        return [
            "self.reg.a = self.stack.pop().expect(\"pla on empty stack\");",
            "self.flags.z = self.reg.a == 0;",
            "self.flags.n = (self.reg.a >> 7) != 0;",
        ]

    # ---- self-modifying code (recognised immediate-operand patch) -------
    # `pass3_smc` rewrote `sta :label+1` (patching the `#imm` byte of the
    # instruction at `:label`) into a `StoreOpVar`, and marked that
    # instruction's `Imm.opvar` so it reads the same provisional operand
    # variable (`self.<name>`) written here. The opaque `StoreLocal`
    # (address/branch patches) has no consumer model yet and stays raw.

    if isinstance(item, StoreOpVar):
        return [f"self.smc.{_mangle(item.name)} = self.reg.{item.reg};"]

    if isinstance(item, StoreOpAddr):
        # Patch the low / high byte of a recognised SMC address operand;
        # the marked `Abs` reads `self.<name>_lo`/`_hi` back via `_abs_index`.
        return [f"self.smc.{_mangle(item.name)}_{item.half} = self.reg.{item.reg};"]

    return None


def _emit_wide16(stmt: Wide16Stmt, syms: SymTable | None) -> list[str]:
    """Lower a 16-bit add/subtract to two chained byte ops (no
    indentation). `_lo` carries the low-byte result; its bit-8 is the
    carry into the high byte. Add uses a 0 carry-in (`clc`); subtract
    uses the `src + ~op + 1` identity (`sec`), so both reduce to `+`.
    Preserves the idiom's full effect: both stores, plus A = high byte
    and C = high carry-out (Z/N are not modelled for add/sbc)."""
    complement = stmt.op == "-"
    lo_carry_in = " + 1" if complement else ""
    return [
        f"let _lo = {_wide_term(stmt.lo_src, syms, complement=False)}"
        f" + {_wide_term(stmt.lo_op, syms, complement=complement)}{lo_carry_in};",
        f"{_emit_target(stmt.lo_dst, syms)} = _lo as u8;",
        f"let _hi = {_wide_term(stmt.hi_src, syms, complement=False)}"
        f" + {_wide_term(stmt.hi_op, syms, complement=complement)} + (_lo >> 8);",
        f"{_emit_target(stmt.hi_dst, syms)} = _hi as u8;",
        "self.reg.a = _hi as u8;",
        "self.flags.c = (_hi >> 8) != 0;",
    ]


# ---------------------------------------------------------------- statements


def _emit_stmt(
    stmt,
    indent: int,
    syms: SymTable | None = None,
    names: dict[str, str] | None = None,
) -> list[str]:
    pad = INDENT * indent

    if isinstance(stmt, Assign):
        place = _emit_target(stmt.target, syms)
        if place is None:
            return [f"{pad}// TODO(pass4): store via {type(stmt.target).__name__}"]
        return [f"{pad}{place} = {_emit_value(stmt.source, syms)};"]

    if isinstance(stmt, Wide16Stmt):
        return [f"{pad}{line}" for line in _emit_wide16(stmt, syms)]

    if isinstance(stmt, ReturnStmt):
        return [f"{pad}return;"]

    if isinstance(stmt, LabeledBlock):
        lines = [f"{pad}{stmt.label}: {{"]
        for s in stmt.body.stmts:
            lines.extend(_emit_stmt(s, indent + 1, syms, names))
        lines.append(f"{pad}}}")
        return lines

    if isinstance(stmt, BreakStmt):
        target = f" {stmt.label}" if stmt.label else ""
        return [f"{pad}break{target};"]

    if isinstance(stmt, ContinueStmt):
        return [f"{pad}continue;"]

    if isinstance(stmt, CallStmt):
        return [f"{pad}self.{_method_name(stmt.target, names)}();"]

    if isinstance(stmt, TailCallStmt):
        method = _method_name(stmt.target, names)
        return [f"{pad}self.{method}();", f"{pad}return;"]

    if isinstance(stmt, SaveTemp):
        return [f"{pad}let tmp{stmt.slot} = self.reg.a;"]

    if isinstance(stmt, RestoreTemp):
        return [f"{pad}self.reg.a = tmp{stmt.slot};"]

    if isinstance(stmt, IfStmt):
        lines = [f"{pad}if {_emit_compare(stmt.cond, syms)} {{"]
        for s in stmt.then_block.stmts:
            lines.extend(_emit_stmt(s, indent + 1, syms, names))
        if stmt.else_block is not None:
            lines.append(f"{pad}}} else {{")
            for s in stmt.else_block.stmts:
                lines.extend(_emit_stmt(s, indent + 1, syms, names))
        lines.append(f"{pad}}}")
        return lines

    if isinstance(stmt, RawIfStmt):
        lines = [f"{pad}if {_emit_branch_cond(stmt.cond)} {{"]
        for s in stmt.then_block.stmts:
            lines.extend(_emit_stmt(s, indent + 1, syms, names))
        if stmt.else_block is not None:
            lines.append(f"{pad}}} else {{")
            for s in stmt.else_block.stmts:
                lines.extend(_emit_stmt(s, indent + 1, syms, names))
        lines.append(f"{pad}}}")
        return lines

    if isinstance(stmt, LoopStmt):
        lines = [f"{pad}loop {{"]
        for s in stmt.body.stmts:
            lines.extend(_emit_stmt(s, indent + 1, syms, names))
        lines.append(f"{pad}}}")
        return lines

    if isinstance(stmt, DoWhileStmt):
        cond = _emit_compare(stmt.cond, syms)
        inner = INDENT * (indent + 1)
        lines = [f"{pad}loop {{"]
        for s in stmt.body.stmts:
            lines.extend(_emit_stmt(s, indent + 1, syms, names))
        lines += [
            f"{inner}if !({cond}) {{",
            f"{inner}    break;",
            f"{inner}}}",
            f"{pad}}}",
        ]
        return lines

    if isinstance(stmt, ForStmt):
        step_method = "wrapping_sub" if stmt.step < 0 else "wrapping_add"
        step_lit = f"0x{abs(stmt.step) & 0xff:02x}"
        cond = _emit_compare(stmt.cond, syms)
        inner = INDENT * (indent + 1)
        lines = [
            f"{pad}self.reg.{stmt.var} = {_emit_imm(stmt.start)};",
            f"{pad}loop {{",
        ]
        for s in stmt.body.stmts:
            lines.extend(_emit_stmt(s, indent + 1, syms, names))
        lines += [
            f"{inner}self.reg.{stmt.var} = self.reg.{stmt.var}.{step_method}({step_lit});",
            f"{inner}if !({cond}) {{",
            f"{inner}    break;",
            f"{inner}}}",
            f"{pad}}}",
        ]
        return lines

    if isinstance(stmt, RepeatStmt):
        step_method = "wrapping_sub" if stmt.step < 0 else "wrapping_add"
        step_lit = f"0x{abs(stmt.step) & 0xff:02x}"
        inner = INDENT * (indent + 1)
        lines = [
            f"{pad}self.reg.{stmt.var} = {_emit_imm(stmt.start)};",
            f"{pad}for _ in 0..{stmt.count}usize {{",
        ]
        for s in stmt.body.stmts:
            lines.extend(_emit_stmt(s, indent + 1, syms, names))
        lines += [
            f"{inner}self.reg.{stmt.var} = self.reg.{stmt.var}.{step_method}({step_lit});",
            f"{pad}}}",
        ]
        return lines

    if isinstance(stmt, MatchStmt):
        lines = [f"{pad}match self.reg.{stmt.reg} {{"]
        for arm in stmt.arms:
            vals = " | ".join(f"0x{v.value & 0xff:02x}" for v in arm.values)
            lines.append(f"{pad}    {vals} => {{")
            for s in arm.body.stmts:
                lines.extend(_emit_stmt(s, indent + 2, syms, names))
            lines.append(f"{pad}    }}")
        lines += [f"{pad}    _ => {{}}", f"{pad}}}"]
        return lines

    if isinstance(stmt, GotoStateStmt):
        return [f"{pad}pc = {stmt.state};"]

    if isinstance(stmt, DispatchStmt):
        inner = INDENT * (indent + 1)
        lines = [
            f"{pad}let mut pc: u32 = {stmt.entry};",
            f"{pad}loop {{",
            f"{inner}match pc {{",
        ]
        for arm in stmt.arms:
            lines.append(f"{inner}{INDENT}{arm.state} => {{")
            for s in arm.body.stmts:
                lines.extend(_emit_stmt(s, indent + 3, syms, names))
            lines.append(f"{inner}{INDENT}}}")
        lines += [
            f"{inner}{INDENT}_ => unreachable!(),",
            f"{inner}}}",
            f"{pad}}}",
        ]
        return lines

    if isinstance(stmt, RawStmt):
        lowered = _emit_raw(stmt.item, syms)
        if lowered is not None:
            return [f"{pad}{line}" for line in lowered]
        from .ir1 import format_item
        return [f"{pad}// raw: {format_item(stmt.item).strip()}"]

    return [f"{pad}// TODO(pass4): lower {type(stmt).__name__}"]


# ---------------------------------------------------------------- routines / module


def emit_routine(
    routine: RoutineIR3,
    indent: int = 1,
    syms: SymTable | None = None,
    names: dict[str, str] | None = None,
) -> list[str]:
    pad = INDENT * indent
    lines: list[str] = []
    if routine.entry_aliases:
        lines.append(f"{pad}// aliases: {', '.join(routine.entry_aliases)}")
    lines.append(f"{pad}fn {_mangle(routine.name)}(&mut self) {{")
    for s in routine.body.stmts:
        lines.extend(_emit_stmt(s, indent + 1, syms, names))
    lines.append(f"{pad}}}")
    return lines


_HEADER = [
    "// @generated by pop_lifter — DO NOT EDIT.",
    "//",
    "// Pass 4: module + routine scaffolding with leaf-expression,",
    "// control-flow, data-movement, carry-arithmetic, `(ptr),y` indirect,",
    "// cmp/bit flag, 16-bit (`Wide16`), dispatch-fallback, stack, and SMC",
    "// operand-patch lowering.",
    "//",
    "// State is decomposed into `Cpu { reg, flags, mem, stack, smc }`:",
    "// registers `self.reg.{a,x,y}: u8`, flags `self.flags.{c,z,n,i,d}:",
    "// bool`, memory `self.mem[addr]` (64 KiB), the `pha`/`pla` value",
    "// stack `self.stack: Vec<u8>`, and recognised SMC operand variables",
    "// `self.smc.<name>` (immediate) / `self.smc.<name>_lo`/`_hi`",
    "// (address). RAM addresses keep their source symbol names via the",
    "// `sym` constants below. Cross-module routine calls resolve once the",
    "// modules are compiled together; opcode / branch-target SMC stays",
    "// deferred as `// raw: …` comments.",
]


_REG_FIELDS = ("a", "x", "y")
_FLAG_FIELDS = ("c", "z", "n", "i", "d")


def _emit_state_defs(smc_fields: list[str]) -> list[str]:
    """Emit the decomposed CPU state types: `Cpu` and its `Regs` / `Flags`
    / `Smc` sub-structs. `smc_fields` are the recognised SMC operand-
    variable field names referenced by this file's routines (immediate
    `<name>` and address `<name>_lo`/`_hi`)."""
    regs = ", ".join(f"pub {r}: u8" for r in _REG_FIELDS)
    flags = ", ".join(f"pub {f}: bool" for f in _FLAG_FIELDS)
    lines = [
        "#[derive(Default)]",
        f"pub struct Regs {{ {regs} }}",
        "",
        "// 6502 status flags, modelled as booleans (V is not tracked).",
        "#[derive(Default)]",
        f"pub struct Flags {{ {flags} }}",
        "",
        "// Self-modifying-code operand variables: instruction operands the",
        "// routine patches at runtime, recovered by pass-3 SMC recognition.",
        "#[derive(Default)]",
        "pub struct Smc {",
    ]
    for f in smc_fields:
        lines.append(f"    pub {f}: u8,")
    lines += [
        "}",
        "",
        "pub struct Cpu {",
        "    pub reg: Regs,",
        "    pub flags: Flags,",
        "    pub mem: Box<[u8; 0x10000]>,",
        "    pub stack: Vec<u8>,",
        "    pub smc: Smc,",
        "}",
        "",
        "impl Cpu {",
        "    pub fn new() -> Self {",
        "        Cpu {",
        "            reg: Regs::default(),",
        "            flags: Flags::default(),",
        "            mem: Box::new([0u8; 0x10000]),",
        "            stack: Vec::new(),",
        "            smc: Smc::default(),",
        "        }",
        "    }",
        "}",
    ]
    return lines


def _record_syms(module: ModuleIR3, syms: SymTable) -> None:
    """Phase-1 recording pass: emit each routine into `syms` so every
    rendered `Abs` index is noted. The produced lines are discarded; only
    the table's `_seen` accumulation matters. Safe to call across several
    modules before a single `finalize()`."""
    for routine in module.routines:
        emit_routine(routine, indent=1, syms=syms)


def _emit_impl_block(
    module: ModuleIR3, syms: SymTable, names: dict[str, str] | None = None
) -> list[str]:
    lines = ["impl Cpu {"]
    for i, routine in enumerate(module.routines):
        if i:
            lines.append("")
        lines.extend(emit_routine(routine, indent=1, syms=syms, names=names))
    lines.append("}")
    return lines


def emit_module(module: ModuleIR3) -> str:
    """Emit one IR3 module as a standalone Rust source file."""
    return emit_modules([module])


def emit_modules(modules: list[ModuleIR3]) -> str:
    """Emit one or more IR3 modules into a single Rust source file.

    All modules share one `mod sym { ... }` block: a Rust file may hold
    only one module item of a given name, so emitting a `mod sym` per
    source (as `emit_module` did when its outputs were concatenated)
    produced a duplicate-module compile error. Symbols are recorded
    across every module before a single `finalize()`, so the global
    conflict rule (a base resolving to two addresses falls back to a
    literal everywhere) holds across files too. Each module keeps its own
    `// source:` line and `impl Cpu { ... }` block — several inherent
    `impl` blocks for one type are valid Rust.

    For a single module the output is byte-identical to the previous
    standalone form."""
    from .ir1 import _portable_path

    # Phase 1: record every rendered `Abs` across all modules into one
    # table, then resolve to named constants. The recorded output is
    # discarded.
    syms = SymTable()
    for module in modules:
        _record_syms(module, syms)
    syms.finalize()

    # Resolve call targets (incl. entry aliases) to canonical method names.
    names = _build_name_table(modules)

    # Phase 2: emit the `impl Cpu` blocks, then scan them for the SMC
    # operand-variable fields they reference so the `Smc` struct declares
    # exactly those.
    impl_lines: list[str] = []
    for i, module in enumerate(modules):
        if i:
            impl_lines.append("")
        impl_lines.extend(_emit_impl_block(module, syms, names))
    smc_fields = sorted(set(re.findall(r"self\.smc\.(\w+)", "\n".join(impl_lines))))

    lines = [*_HEADER, "//"]
    for module in modules:
        lines.append(f"// source: {_portable_path(module.file)}")
    lines.append("")
    lines.extend(_emit_state_defs(smc_fields))
    lines.append("")

    sym_block = syms.render_block("")
    if sym_block:
        lines.extend(sym_block)
        lines.append("")

    lines.extend(impl_lines)
    return "\n".join(lines) + "\n"


def lower_stats(module: ModuleIR3) -> tuple[int, int]:
    """Count top-level statements that produce real Rust code vs. those
    that produce a comment placeholder. Nested blocks under control-flow
    are not counted — only the top-level flat list of each routine."""
    lowered = deferred = 0
    for routine in module.routines:
        for s in routine.body.stmts:
            if isinstance(s, RawStmt):
                # A raw atom counts as lowered only when this slice
                # produces real Rust for it (data-movement atoms).
                if _emit_raw(s.item) is not None:
                    lowered += 1
                else:
                    deferred += 1
            elif isinstance(s, _LOWERED_TYPES):
                lowered += 1
            else:
                deferred += 1
    return lowered, deferred
