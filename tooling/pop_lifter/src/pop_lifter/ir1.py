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


def _fmt_imm(imm: Imm) -> str:
    return f"#{imm.value & 0xff:#04x}"


def _fmt_abs(a: Abs) -> str:
    return f"{a.name}@{a.addr:#06x}"


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
    if isinstance(item, Call):
        return f"  call {item.target}                  ; {item.src.short()}"
    if isinstance(item, Goto):
        kw = "tail_call" if item.kind == "tail_call" else "goto"
        return f"  {kw} {item.target}                ; {item.src.short()}"
    if isinstance(item, Return):
        return f"  return                           ; {item.src.short()}"
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
