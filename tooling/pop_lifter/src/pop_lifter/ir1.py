"""IR1: a C-like, opcode-for-opcode lifting of 6502 assembly.

Goals for this stage:

* Stay close to the underlying 6502. Pseudo-registers `a, x, y`; flag
  pseudo-globals `c, z, n, v`. Every IR1 instruction maps back to exactly
  one Merlin source line via `SourceRef`.
* Resolve operand symbols against the equate table from pass 0 so each
  memory access carries both its name (`clrU`) and the concrete address
  (`$d6`).
* Keep control flow explicit and unstructured. Conditional branches lower
  to `branch(cond, label)`; unconditional jumps to `goto(label)`.

Pass 2 will reconstruct structured control flow, fold flag updates, fuse
parallel arrays, and recognise 16-bit add/sub patterns. None of that
happens here.

The instruction set covered by the current lifter is the minimum the
AUTO.S combat-button pilot needs (immediate load, absolute store,
unconditional jump, rts). Other opcodes appear as `Unsupported` so the
lifted IR1 still round-trips line-for-line with the source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Reg(str, Enum):
    A = "a"
    X = "x"
    Y = "y"

    def __str__(self) -> str:  # nicer repr in dumps
        return self.value


@dataclass(frozen=True)
class SourceRef:
    """Back-reference from an IR1 instruction to the source line it lifted from.

    The lifter promises that every emitted IR1 item has a `src` and that
    `src.line` matches the 1-based line number in `src.file`. Pass 4 uses
    this to generate the `// @generated` manifest required by the plan.
    """

    file: str
    line: int
    raw: str

    def short(self) -> str:
        # Normalise both separators by hand: on POSIX hosts `PurePath`
        # doesn't recognise `\\` as a separator, so a path produced on
        # a Windows checkout would otherwise come through verbatim.
        normalised = self.file.replace("\\", "/")
        return f"{normalised.rsplit('/', 1)[-1]}:{self.line}"


# ---------------------------------------------------------------- operands
#
# Operands are tagged so the interpreter can route each instruction to the
# right memory access without re-parsing the operand string. The `name`
# field carries the symbolic origin (`clrU`, future: `mob,x`, `(ptr),y`)
# for diagnostics and later passes.


@dataclass(frozen=True)
class Imm:
    """`#expr` — an immediate byte. Stored as a signed Python int so the
    Merlin `#-1` form survives without an extra masking step; the
    interpreter masks to 8 bits at the point of use."""

    value: int
    text: str  # original operand text, e.g. "#-1" or "#$ff"


@dataclass(frozen=True)
class Abs:
    """An absolute (or zero-page) address — e.g. `STA clrU`. We don't
    distinguish zero-page from absolute in IR1; the address value does."""

    name: str
    addr: int


# Other operand kinds (`IndexedX`, `IndexedY`, `IndY`, `IndX`) will land
# alongside `rndp` / `CheckFloor` in the next pilot slice. They are not
# emitted by the current lifter, so we omit them here to avoid speculative
# scaffolding.

Operand = Imm | Abs


# ---------------------------------------------------------------- instrs


@dataclass(frozen=True)
class Label:
    """A named program point inside a routine. The first instruction's
    entry labels live on `Routine.entry_aliases`; everything reachable
    only by branch/jump from inside the routine is a `Label` item in the
    body."""

    name: str
    src: SourceRef


@dataclass(frozen=True)
class LoadImm:
    """`lda/ldx/ldy #imm`. Sets N and Z; C/V untouched."""

    reg: Reg
    imm: Imm
    src: SourceRef


@dataclass(frozen=True)
class StoreAbs:
    """`sta/stx/sty addr`. No flag effects."""

    reg: Reg
    target: Abs
    src: SourceRef


@dataclass(frozen=True)
class Goto:
    """`jmp target` — unconditional control transfer.

    `kind` distinguishes:
    * `tail_call` — the target is another routine; the interpreter
      switches to executing it and inherits its return.
    * `local` — the target is a label inside the current routine.

    Pass 1 picks `tail_call` for jumps to global-style names and `local`
    for jumps to Merlin's local (`:foo`) or macro (`]foo`) label forms.
    Pass 2 will refine this against the full call graph.
    """

    target: str
    kind: str  # "tail_call" | "local"
    src: SourceRef


@dataclass(frozen=True)
class Return:
    """`rts` — return from subroutine."""

    src: SourceRef


@dataclass(frozen=True)
class Call:
    """`jsr target` — push return address, transfer to `target`. The
    interpreter implements this with an explicit call stack so step
    counts and stack depth stay observable. `target` is the symbolic
    label name; pass 1 records non-local names verbatim and lets the
    interpreter resolve them across modules / aliases at run time."""

    target: str
    src: SourceRef


@dataclass(frozen=True)
class LoadAbs:
    """`lda/ldx/ldy addr`. Loads an 8-bit value from memory into the
    named register. Sets N/Z conceptually; the interpreter doesn't
    track them yet because the rndp slice never reads them."""

    reg: Reg
    source: Abs
    src: SourceRef


@dataclass(frozen=True)
class Asl:
    """`asl a` — shift accumulator left one bit. New C = old bit 7;
    new A = (A << 1) & 0xff."""

    src: SourceRef


@dataclass(frozen=True)
class Clc:
    """`clc` — clear carry."""

    src: SourceRef


@dataclass(frozen=True)
class Sec:
    """`sec` — set carry."""

    src: SourceRef


@dataclass(frozen=True)
class AdcImm:
    """`adc #imm` — A = A + imm + C (mod 256); C = overflow."""

    imm: Imm
    src: SourceRef


@dataclass(frozen=True)
class AdcAbs:
    """`adc addr` — A = A + mem[addr] + C (mod 256); C = overflow."""

    source: Abs
    src: SourceRef


# ---------------------------------------------------------------- pass-1 medium-tail atoms


@dataclass(frozen=True)
class SbcImm:
    """`sbc #imm` — A = A - imm - (1 - C) (mod 256). C = 1 means "no
    borrow needed" (the result fit without underflow); C = 0 means a
    borrow happened. Sets N, Z, C (and V, which we don't currently
    model since no branch reads it).

    Implemented in the interpreter as the standard `A + ~operand + C`
    trick so the 6502's borrow convention falls out of the same
    arithmetic the chip itself uses."""

    imm: Imm
    src: SourceRef


@dataclass(frozen=True)
class SbcAbs:
    """`sbc addr` — same as `SbcImm` but reading the subtrahend from
    memory. Pairs with `Clc`/`Sec` + `Adc*` to form 16-bit math."""

    source: Abs
    src: SourceRef


@dataclass(frozen=True)
class Lsr:
    """`lsr a` / `lsr` — accumulator-only logical shift right.
    `C = old A bit 0`, `A = A >> 1`. N is always 0 (the shifted-in
    bit). Z = (A == 0).

    Memory `lsr addr` would need a separate node; none of the
    routines lifted so far emit it."""

    src: SourceRef


@dataclass(frozen=True)
class Bit:
    """`bit operand` — bit-test. Sets:

    * `Z = (A & operand) == 0` — fusable in principle but our
      `Compare` form has no masked-equality variant, so pass 2
      leaves `bit ; beq` unfused for now.
    * `N = bit 7 of operand` — *not* of `(A & operand)`. The N flag
      reflects the OPERAND, independent of A.
    * `V = bit 6 of operand` — same as N but for bit 6. (Like
      `SbcImm`, V is conceptually written but **not currently
      tracked** in `Trace`; nothing reads it yet. `bvc`/`bvs` aren't
      lifted, so a V-dependent test would surface as `Unsupported`
      rather than silently misbehave.)

    Crucially, `bit` does NOT modify A. `Bit(Imm)` is therefore a
    pure flag-setter, eligible for elision by `pass2_struct` when
    its outputs are dead. `Bit(Abs)` is **not** elided, because the
    memory read itself can be side-effecting — `bit $c0xx` is the
    classic Apple II soft-switch idiom (speaker click, page select,
    paddle reads)."""

    source: "Imm | Abs"
    src: SourceRef


@dataclass(frozen=True)
class Pha:
    """`pha` — push A onto the 6502 stack. The interpreter models
    the stack as a Python list (`Trace.value_stack`) rather than as
    bytes at `$0100..$01ff`. PHA pushes `A`; PLA pops the most
    recently pushed byte back into A.

    The two-stack design — `Trace.value_stack` for PHA/PLA bytes,
    plus a separate `stack` local in `interp_ir1.run` for JSR/RTS
    continuation tuples — is sound when:

    1. Every callee balances its own PHA/PLA over each control path,
       so a routine's PHA bytes don't leak into its caller's frame
       — true for POP's source.
    2. No code uses the `pha; rts` "computed jump via the stack"
       idiom — true for POP's source (we grepped to confirm).

    If a future input violates either assumption we'd need full
    byte-level stack emulation with a stack pointer and the actual
    `$0100..$01ff` page in `Trace.ram`. Documented here so the
    limitation is visible at the IR1-type level, not just buried in
    the interpreter."""

    src: SourceRef


@dataclass(frozen=True)
class Pla:
    """`pla` — pop the top of the stack into A. Sets Z/N on the
    popped value. Same two-stack caveat as `Pha`."""

    src: SourceRef


# ---------------------------------------------------------------- indirect addressing


@dataclass(frozen=True)
class IndirectY:
    """`(ptr),y` post-indexed indirect addressing. The CPU reads a
    16-bit pointer from `mem[ptr.addr]` (low byte) and
    `mem[ptr.addr + 1]` (high byte), then adds Y to that pointer and
    uses the result as the effective address.

    Notes:

    * On the NMOS 6502 the high-byte fetch wraps at the page boundary
      of the zero-page pointer (so `($ff),y` reads at $ff and $00).
      The interpreter follows the more permissive `(addr + 1) & 0xffff`
      rule because POP's pointers never sit at $ff and we don't want
      to surprise authors of synthetic test inputs. If the engine
      ends up exercising the wrap case we'll switch to the real
      semantics.

    * The 6502 has no `(ptr),x` form for absolute pointers, and POP's
      source never uses `(zp,x)` — pre-indexed indirect — so this
      single node type covers every indirect form we'll lift."""

    ptr: Abs


@dataclass(frozen=True)
class LoadIndirect:
    """`lda (ptr),y` — load A from the post-indexed indirect address.
    Sets Z/N on the loaded byte. The 6502 only has the `lda` form
    here (no `ldx`/`ldy` against `(ptr),y`), so `reg` is always A;
    we record it explicitly so the same `_affected_register` rule
    that handles `LoadAbs`/`LoadImm`/`LoadIndexed` also accepts this
    node — `lda (ptr),y ; beq L` fuses into `if a == 0 goto L`
    exactly like the other loads."""

    reg: Reg
    source: IndirectY
    src: SourceRef


@dataclass(frozen=True)
class StoreIndirect:
    """`sta (ptr),y` — store A at the post-indexed indirect address.
    Doesn't affect flags. As with `LoadIndirect`, only `sta` exists
    for this addressing mode on stock 6502."""

    reg: Reg
    target: IndirectY
    src: SourceRef


@dataclass(frozen=True)
class CmpIndirect:
    """`cmp (ptr),y` — compute A - mem[(ptr)+Y]; set Z/N/C without
    storing. Only one site uses this in POP, but lifting it keeps
    the dump consistent with the source."""

    reg: Reg
    source: IndirectY
    src: SourceRef


# ---------------------------------------------------------------- indexed-absolute addressing


@dataclass(frozen=True)
class IndexedAbs:
    """`base,x` or `base,y` — indexed-absolute addressing as a value
    type so it can appear anywhere `Abs` does in instruction source/
    target unions. Mirrors how `LoadIndexed` / `StoreIndexed` already
    embed `(base, index)` for the load/store opcodes, but exposes it
    as a value so non-load/store ops (`cmp tbl,x`, `and table,x`,
    `adc base,y`, `ora list,y`) can also carry indexed operands.

    `index` is always `Reg.X` or `Reg.Y` on the 6502; the IR
    enforces that at construction by the lifter (`_parse_indexed`)."""

    base: Abs
    index: Reg


@dataclass(frozen=True)
class CmpIndexed:
    """`cmp base,x` or `cmp base,y` — compare A against the byte at
    `mem[base + idx_reg]`. Sets Z/N/C without storing. Same fusion
    rules as `CmpAbs` (`bcs`/`bcc` → `>=`/`<`, `beq`/`bne` →
    `==`/`!=`), so chains like

        cmp tbl,x
        bne :next

    fuse cleanly into `if a != *tbl[x] goto :next`.

    Limited to `cmp` on stock 6502: `cpx`/`cpy` have no abs-indexed
    addressing mode (zero-page-only). `reg` is therefore always
    `Reg.A`; the lifter rejects `cpx tbl,y` / `cpy tbl,x` as
    `Unsupported` rather than synthesising a CmpIndexed for them.
    The field is kept for shape consistency with `CmpAbs` / `CmpImm`,
    not to imply cpx/cpy coverage."""

    reg: Reg
    base: Abs
    index: Reg
    src: SourceRef


@dataclass(frozen=True)
class AdcIndexed:
    """`adc base,x` or `,y` — A += mem[base + idx_reg] + C. Mirror of
    `AdcAbs` for the indexed addressing mode. Used by POP's
    arithmetic helpers (`adc BarL,y` etc.)."""

    base: Abs
    index: Reg
    src: SourceRef


@dataclass(frozen=True)
class SbcIndexed:
    """`sbc base,x` or `,y` — same shape as `AdcIndexed` but
    subtracts with borrow. Pairs with `SbcAbs` / `SbcImm` for the
    other addressing modes already lifted."""

    base: Abs
    index: Reg
    src: SourceRef


# ---------------------------------------------------------------- pass-1 long-tail atoms


@dataclass(frozen=True)
class IncTarget:
    """`inx` / `iny` / `inc addr` — add 1 to a register or memory
    location. Sets Z/N on the result; does NOT touch C. `target` is
    either a `Reg` (X or Y — `ina` doesn't exist on stock 6502) or
    an `Abs` for memory-resident counters."""

    target: "Reg | Abs"
    src: SourceRef


@dataclass(frozen=True)
class DecTarget:
    """`dex` / `dey` / `dec addr` — same as `IncTarget` but
    decrement. Sets Z/N; does NOT touch C."""

    target: "Reg | Abs"
    src: SourceRef


@dataclass(frozen=True)
class Transfer:
    """`tax` / `tay` / `txa` / `tya` — copy one register into another.
    Sets Z/N based on the destination value. The 6502 has no
    accumulator↔X+Y transfers via memory in this set; we never
    emit one for `tsx`/`txs` here since the stack pointer is
    modelled separately by the interpreter."""

    src_reg: Reg
    dst_reg: Reg
    src: SourceRef


@dataclass(frozen=True)
class Bitwise:
    """`and` / `ora` / `eor` against A. `op` ∈ {"and", "or", "eor"}.
    `source` is `Imm` (immediate operand), `Abs` (memory), or
    `IndirectY` (`(ptr),y` post-indexed indirect — added for the
    `and (ptr),y` / `ora (ptr),y` patterns POP uses for masked
    sprite blits). Updates A and Z/N; does NOT touch C.

    POP uses these heavily for flag-mask tests. The classic
    `and #fcheckmark ; beq ]rts` lowers to two IR items in order:

        a = a & fcheckmark            # Bitwise (mutates A, sets Z/N)
        if a == 0 goto ]rts           # If (reads Z of post-and A)

    Pass 2's fuser collapses the `Bitwise + Branch` pair into the
    structured `If` shown above. The fused Compare always tests
    against `0`; the mask is preserved by the prior `Bitwise` body
    item (still visible in the dump). Semantically this is the
    same as `if (a & fcheckmark) == 0 goto ]rts`, just expressed
    as two statements because pass-2's Compare doesn't have an
    embedded-expression form."""

    op: str
    # `IndexedAbs` extends the union for `and table,x` / `ora list,y`
    # / `eor mask,x` patterns POP uses for table-driven blits and
    # parallel-array bit checks.
    source: "Imm | Abs | IndirectY | IndexedAbs"
    src: SourceRef


@dataclass(frozen=True)
class LoadIndexed:
    """`lda/ldx/ldy base,idx` — load from `mem[base + idx_reg]`.

    The 6502 has separate forms for `,x` and `,y`; we just record the
    index register on the IR node. Loads also conceptually set Z/N
    based on the loaded value; the interpreter updates `Trace.z` /
    `Trace.n` so subsequent branches can read them."""

    reg: Reg
    base: Abs
    index: Reg          # always X or Y
    src: SourceRef


@dataclass(frozen=True)
class StoreIndexed:
    """`sta/stx/sty base,idx` — store register at `mem[base + idx_reg]`.
    Doesn't affect flags."""

    reg: Reg
    base: Abs
    index: Reg
    src: SourceRef


@dataclass(frozen=True)
class CmpImm:
    """`cmp #imm` / `cpx #imm` / `cpy #imm` — compute `reg - imm`,
    don't store; set Z, N, and C (C is set when there's no borrow,
    i.e. when `reg >= imm`)."""

    reg: Reg
    imm: Imm
    src: SourceRef


@dataclass(frozen=True)
class CmpAbs:
    """`cmp addr` / `cpx addr` / `cpy addr` — same semantics as
    `CmpImm`, but the operand is read from memory."""

    reg: Reg
    source: Abs
    src: SourceRef


@dataclass(frozen=True)
class Branch:
    """Conditional control transfer. `cond` is the 6502 flag combination
    encoded as the suffix of the original mnemonic — `eq` (Z=1),
    `ne` (Z=0), `cc` (C=0), `cs` (C=1), `pl` (N=0), `mi` (N=1),
    `vc` (V=0), `vs` (V=1).

    Pass 1 always emits `target` as a label name. The lifter places the
    `Label` items inside the routine body when it walks past a labeled
    code line, so `_find_label_index` in the interpreter can resolve
    them. Cross-routine branches in the upstream source are rare —
    they'll surface as InterpError until the lifter handles them.
    """

    cond: str
    target: str
    src: SourceRef


# ---------------------------------------------------------------- IR2 atoms
#
# Pass 2 fuses `CmpImm/CmpAbs + Branch` pairs into a single `If` node
# whose `Compare` carries an explicit operator. Same data, fewer items,
# and the interpreter can evaluate the condition without consulting
# flag state. The IR2 nodes live in the same union so a `Routine` can
# mix fused and unfused items where fusion isn't yet possible.


@dataclass(frozen=True)
class Compare:
    """Structured comparison: `reg <op> rhs`. `op` is one of `==`, `!=`,
    `<`, `>=` (unsigned), `<0`, `>=0` (sign tests with no rhs).

    `rhs` carries the right-hand operand. For sign tests (`<0`, `>=0`)
    it's `None`. For value comparisons it's an `Imm` (`cmp #k`) or an
    `Abs` (`cmp addr`)."""

    reg: Reg
    op: str
    # `IndexedAbs` joined the union so `cmp tbl,x ; bne :L` fuses
    # cleanly into `if a != *(tbl + x) goto :L`. Without it the
    # indexed cmp would stay unfused even when the addressing mode
    # is otherwise well-handled.
    rhs: "Imm | Abs | IndexedAbs | None"


@dataclass(frozen=True)
class If:
    """`if Compare goto target` — structured conditional branch.

    Replaces a `CmpImm/CmpAbs` followed immediately by a `Branch` in
    the lifted body. Semantics are identical to the original 6502
    sequence but the interpreter doesn't need to read Z/C from prior
    state — the Compare is self-contained."""

    cond: Compare
    target: str
    src: SourceRef


@dataclass(frozen=True)
class Unsupported:
    """An opcode the current lifter does not yet model. We keep it in
    the IR so dumps stay aligned with the source and so pass-2 reports
    can highlight exactly what still needs lifting."""

    mnemonic: str
    operand: str | None
    src: SourceRef


Instr = (
    LoadImm | StoreAbs | Goto | Return | Call
    | LoadAbs | Asl | Clc | Sec | AdcImm | AdcAbs
    | LoadIndexed | StoreIndexed | CmpImm | CmpAbs | Branch
    | If
    | IncTarget | DecTarget | Transfer | Bitwise
    | LoadIndirect | StoreIndirect | CmpIndirect
    | SbcImm | SbcAbs | Lsr | Bit
    | Pha | Pla
    | CmpIndexed | AdcIndexed | SbcIndexed
    | Unsupported
)
Item = Label | Instr


# ---------------------------------------------------------------- routines


@dataclass
class Routine:
    """A single lifted routine.

    `entry_aliases` lists every label *other than* `name` that resolves
    to this routine's first instruction. Merlin allows several labels on
    consecutive lines before a single opcode (e.g. `DoBlock` / `DoUp`);
    they collapse into one routine here. `body` interleaves `Label`
    markers and `Instr` nodes.
    """

    name: str
    entry_aliases: list[str] = field(default_factory=list)
    body: list[Item] = field(default_factory=list)

    def all_entry_names(self) -> set[str]:
        return {self.name, *self.entry_aliases}

    def label_names(self) -> set[str]:
        out = self.all_entry_names()
        for item in self.body:
            if isinstance(item, Label):
                out.add(item.name)
        return out


@dataclass
class ModuleIR1:
    """All routines lifted from a single `.S` file."""

    name: str  # e.g. "AUTO"
    file: str  # full path
    routines: list[Routine] = field(default_factory=list)

    def find(self, name: str) -> Routine | None:
        for r in self.routines:
            if name in r.all_entry_names():
                return r
        return None


# ---------------------------------------------------------------- pretty-print
#
# Used by the CLI and snapshot tests. Format is deliberately readable
# rather than canonical IR — pass 2 will define its own dump format.


_NUMERIC_IMM_RE = re.compile(
    # `#$ff`, `#%01010101`, `#-1`, `#42` — anything that's just a
    # number after the `#` (with optional `<`/`>` byte operator).
    r"^#[<>]?[\s]*[-+]?(?:\$[0-9a-fA-F]+|%[01]+|\d+)\s*$"
)


def _fmt_imm(imm: Imm) -> str:
    """Render an immediate operand for the IR1 dump.

    Pure-numeric Merlin sources (`#$06`, `#42`, `#-1`) get normalised
    to the `#0x..` hex form so the dump stays consistent. But when the
    text references a *symbol* (e.g. `#shadpos6a`, `#<MoreBytes`,
    `#>jumptable+1`), preserving that name is more useful than showing
    the synthetic-address byte that pass-0 happens to have assigned.
    Reviewers can still see the resolved value via `imm.value` in the
    object, but for visual scanning the symbol carries the engineering
    intent — what code in 1989 called the address by its name, not by
    its assembled offset.

    `.strip()` covers belt-and-suspenders whitespace from synthetic
    test fixtures; pass-0's lexer already removes `;` comments and
    leading/trailing whitespace from operands, so the call is a
    no-op in normal CLI usage.
    """
    if _NUMERIC_IMM_RE.match(imm.text or ""):
        return f"#{imm.value & 0xff:#04x}"
    return imm.text.strip()


def _fmt_abs(a: Abs) -> str:
    return f"{a.name}@{a.addr:#06x}"


def _fmt_compare(c: Compare) -> str:
    if c.rhs is None:
        # Sign test: `a < 0` or `a >= 0`.
        return f"{c.reg} {c.op}"
    if isinstance(c.rhs, Imm):
        return f"{c.reg} {c.op} {_fmt_imm(c.rhs)}"
    if isinstance(c.rhs, IndexedAbs):
        return f"{c.reg} {c.op} *({_fmt_abs(c.rhs.base)} + {c.rhs.index})"
    return f"{c.reg} {c.op} *{_fmt_abs(c.rhs)}"


def format_item(item: Item) -> str:
    if isinstance(item, Label):
        return f"{item.name}:"
    if isinstance(item, LoadImm):
        return f"  {item.reg} = {_fmt_imm(item.imm)}                  ; {item.src.short()}"
    if isinstance(item, LoadAbs):
        return f"  {item.reg} = *{_fmt_abs(item.source)}        ; {item.src.short()}"
    if isinstance(item, StoreAbs):
        return f"  *{_fmt_abs(item.target)} = {item.reg}            ; {item.src.short()}"
    if isinstance(item, Asl):
        return f"  a = a << 1                       ; {item.src.short()}"
    if isinstance(item, Clc):
        return f"  c = 0                            ; {item.src.short()}"
    if isinstance(item, Sec):
        return f"  c = 1                            ; {item.src.short()}"
    if isinstance(item, AdcImm):
        return f"  a = a + {_fmt_imm(item.imm)} + c              ; {item.src.short()}"
    if isinstance(item, AdcAbs):
        return f"  a = a + *{_fmt_abs(item.source)} + c    ; {item.src.short()}"
    if isinstance(item, SbcImm):
        return f"  a = a - {_fmt_imm(item.imm)} - (1 - c)        ; {item.src.short()}"
    if isinstance(item, SbcAbs):
        return f"  a = a - *{_fmt_abs(item.source)} - (1 - c) ; {item.src.short()}"
    if isinstance(item, Lsr):
        return f"  a = a >> 1                       ; {item.src.short()}"
    if isinstance(item, Bit):
        rhs = _fmt_imm(item.source) if isinstance(item.source, Imm) else f"*{_fmt_abs(item.source)}"
        return f"  bit {rhs}                ; {item.src.short()}"
    if isinstance(item, Pha):
        return f"  push a                           ; {item.src.short()}"
    if isinstance(item, Pla):
        return f"  a = pop                          ; {item.src.short()}"
    if isinstance(item, LoadIndexed):
        return (
            f"  {item.reg} = *({_fmt_abs(item.base)} + {item.index})"
            f"   ; {item.src.short()}"
        )
    if isinstance(item, StoreIndexed):
        return (
            f"  *({_fmt_abs(item.base)} + {item.index}) = {item.reg}"
            f"   ; {item.src.short()}"
        )
    if isinstance(item, CmpImm):
        return f"  cmp {item.reg}, {_fmt_imm(item.imm)}              ; {item.src.short()}"
    if isinstance(item, CmpAbs):
        return f"  cmp {item.reg}, *{_fmt_abs(item.source)}    ; {item.src.short()}"
    if isinstance(item, Branch):
        return f"  if {item.cond} goto {item.target}       ; {item.src.short()}"
    if isinstance(item, If):
        return f"  if {_fmt_compare(item.cond)} goto {item.target}  ; {item.src.short()}"
    if isinstance(item, Call):
        return f"  call {item.target}                  ; {item.src.short()}"
    if isinstance(item, Goto):
        kw = "tail_call" if item.kind == "tail_call" else "goto"
        return f"  {kw} {item.target}                ; {item.src.short()}"
    if isinstance(item, Return):
        return f"  return                           ; {item.src.short()}"
    if isinstance(item, IncTarget):
        tgt = item.target if isinstance(item.target, Reg) else f"*{_fmt_abs(item.target)}"
        return f"  {tgt} += 1                         ; {item.src.short()}"
    if isinstance(item, DecTarget):
        tgt = item.target if isinstance(item.target, Reg) else f"*{_fmt_abs(item.target)}"
        return f"  {tgt} -= 1                         ; {item.src.short()}"
    if isinstance(item, Transfer):
        return f"  {item.dst_reg} = {item.src_reg}                            ; {item.src.short()}"
    if isinstance(item, Bitwise):
        sym = {"and": "&", "or": "|", "eor": "^"}[item.op]
        if isinstance(item.source, Imm):
            rhs = _fmt_imm(item.source)
        elif isinstance(item.source, IndirectY):
            rhs = f"*({_fmt_abs(item.source.ptr)})[y]"
        elif isinstance(item.source, IndexedAbs):
            rhs = f"*({_fmt_abs(item.source.base)} + {item.source.index})"
        else:
            rhs = f"*{_fmt_abs(item.source)}"
        return f"  a = a {sym} {rhs}              ; {item.src.short()}"
    if isinstance(item, CmpIndexed):
        return (
            f"  cmp {item.reg}, *({_fmt_abs(item.base)} + {item.index})"
            f"   ; {item.src.short()}"
        )
    if isinstance(item, AdcIndexed):
        return (
            f"  a = a + *({_fmt_abs(item.base)} + {item.index}) + c"
            f"   ; {item.src.short()}"
        )
    if isinstance(item, SbcIndexed):
        return (
            f"  a = a - *({_fmt_abs(item.base)} + {item.index}) - (1 - c)"
            f"   ; {item.src.short()}"
        )
    if isinstance(item, LoadIndirect):
        return (
            f"  {item.reg} = *({_fmt_abs(item.source.ptr)})[y]"
            f"   ; {item.src.short()}"
        )
    if isinstance(item, StoreIndirect):
        return (
            f"  *({_fmt_abs(item.target.ptr)})[y] = {item.reg}"
            f"   ; {item.src.short()}"
        )
    if isinstance(item, CmpIndirect):
        return (
            f"  cmp {item.reg}, *({_fmt_abs(item.source.ptr)})[y]"
            f"   ; {item.src.short()}"
        )
    if isinstance(item, Unsupported):
        op = item.operand if item.operand else ""
        return f"  ??? {item.mnemonic} {op}            ; {item.src.short()}"
    raise TypeError(f"unknown IR1 item: {type(item)}")


def format_routine(r: Routine) -> str:
    header = f"fn {r.name}"
    if r.entry_aliases:
        header += f"  (aka {', '.join(r.entry_aliases)})"
    out = [header + " {"]
    for item in r.body:
        out.append(format_item(item))
    out.append("}")
    return "\n".join(out)


def _portable_path(file: str) -> str:
    """Strip everything up to and including `vendor/pop-apple2/` so the
    dumped path matches across machines/CI checkouts. Falls back to the
    basename if the marker isn't present.

    We replace backslashes with forward slashes before applying the
    marker logic so Windows-style inputs are handled even when the
    interpreter is running on POSIX (where `PurePath` won't recognise
    `\\` as a separator)."""
    posix = file.replace("\\", "/")
    marker = "vendor/pop-apple2/"
    idx = posix.find(marker)
    if idx >= 0:
        return posix[idx + len(marker):]
    return posix.rsplit("/", 1)[-1]


def format_module(m: ModuleIR1) -> str:
    parts = [f"; module {m.name} from {_portable_path(m.file)}"]
    for r in m.routines:
        parts.append("")
        parts.append(format_routine(r))
    return "\n".join(parts) + "\n"
