"""IR3: structured statements reconstructed from the goto-flow IR2.

Pass 2's structurer (`pass2_reloop`) takes the basic-block CFG of an
IR2 routine and reshapes it into a tree of structured statements —
`Block`, `IfStmt`, `ReturnStmt`, `TailCallStmt`, etc. — that read
like ordinary procedural code.

Design choices for this first slice:

* **Statement-oriented**, not expression-oriented. Each non-control-
  flow IR1/IR2 atom (loads, stores, arithmetic) shows up wrapped in
  a `RawStmt`. Pass 3 will fold those into expressions; we don't try
  to here.
* **Escape hatches**: `GotoStmt` / `LabelStmt` exist so the relooper
  can fall back when it can't structure a region (loops, irreducible
  control flow). Anything emitting one of these is flagged for
  human review.
* **`Compare` re-used from IR2** for `IfStmt.cond`. That keeps the
  condition self-contained and means the IR3 interpreter doesn't
  need to read 6502 flag state.
* **`IfStmt.else_block` is optional.** The common shape produced by
  the relooper is `if cond { taken_action }` followed by the
  fall-through — early-exit returns, conditional tail calls. A
  full `else_block` only appears when the structurer recognises a
  symmetric two-way fork. For now we always emit `else_block=None`
  and emit the fall-through after the `IfStmt`; pass 3 / pass 4
  may rewrite into `if/else` shape when it improves readability.

Out of scope: loops (`while`/`for`), `match`/`switch` recognition,
type-folded expressions. Those land in pass 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ir1 import Compare, Item, SourceRef


# A `RawStmt` wraps a non-control-flow IR1/IR2 item. The relooper
# never inspects the inner item — it just passes it through. The IR3
# interpreter delegates the actual operation to the existing IR1
# interpreter machinery (per-opcode dispatch in `interp_ir1`).
@dataclass(frozen=True)
class RawStmt:
    item: Item


@dataclass(frozen=True)
class CallStmt:
    """`jsr target` — non-tail call. Translates to a regular function
    call at the Rust level."""

    target: str
    src: SourceRef


@dataclass(frozen=True)
class TailCallStmt:
    """`jmp target` where target is another routine. The pass-2
    relooper emits this for every tail call (unconditional `Goto
    kind=tail_call`) and for the taken edge of a cross-module
    conditional branch."""

    target: str
    src: SourceRef


@dataclass(frozen=True)
class ReturnStmt:
    src: SourceRef


@dataclass(frozen=True)
class IfStmt:
    """`if cond { then_block } [else { else_block }]`. The relooper
    currently emits `else_block=None` for every site; the fall-through
    edge follows the `IfStmt` in its enclosing `Block.stmts` list."""

    cond: Compare
    then_block: "Block"
    else_block: "Block | None"
    src: SourceRef


@dataclass(frozen=True)
class GotoStmt:
    """Escape hatch: the relooper couldn't structure this transfer
    (typically a loop back-edge or an irreducible CFG fragment). Pass
    3 will either restructure or fall back to a `loop { match pc {
    ... } }` dispatcher. For CHECKFLOOR this should never appear."""

    target: str
    src: SourceRef


@dataclass(frozen=True)
class LabelStmt:
    """Companion to `GotoStmt`. Same caveats."""

    name: str
    src: SourceRef


# `RawIfStmt` covers the case of an unfused `Branch` whose condition
# is a raw flag combination (e.g. `if eq goto X` where the preceding
# flag-setter wasn't lifted to a Compare). The relooper still emits
# the IR3 wrapping so the routine doesn't fall apart, but pass 3 will
# need to revisit these. Tracked separately so callers can flag the
# count.
@dataclass(frozen=True)
class RawIfStmt:
    cond: str               # flag suffix: "eq" | "ne" | "cs" | ...
    then_block: "Block"
    else_block: "Block | None"
    src: SourceRef


Stmt = (
    RawStmt | CallStmt | TailCallStmt | ReturnStmt
    | IfStmt | RawIfStmt | GotoStmt | LabelStmt
)


@dataclass(frozen=True)
class Block:
    stmts: tuple[Stmt, ...]

    @classmethod
    def of(cls, stmts: list[Stmt]) -> "Block":
        return cls(stmts=tuple(stmts))


@dataclass
class RoutineIR3:
    name: str
    entry_aliases: list[str] = field(default_factory=list)
    body: Block = field(default_factory=lambda: Block.of([]))


@dataclass
class ModuleIR3:
    name: str
    file: str
    routines: list[RoutineIR3]

    def find(self, name: str) -> RoutineIR3 | None:
        for r in self.routines:
            if r.name == name or name in r.entry_aliases:
                return r
        return None


# ---------------------------------------------------------------- format


def _fmt_compare(c: Compare) -> str:
    from .ir1 import Abs, Imm
    if c.rhs is None:
        return f"{c.reg} {c.op}"
    if isinstance(c.rhs, Imm):
        return f"{c.reg} {c.op} #{c.rhs.value:#04x}"
    if isinstance(c.rhs, Abs):
        return f"{c.reg} {c.op} *{c.rhs.name}@{c.rhs.addr:#06x}"
    return f"{c.reg} {c.op} {c.rhs!r}"


def _fmt_stmt(stmt: Stmt, indent: int) -> list[str]:
    pad = "  " * indent
    if isinstance(stmt, RawStmt):
        # Delegate body formatting to the IR1 formatter so the dump
        # stays consistent — same identifiers, same comments.
        from .ir1 import format_item
        line = format_item(stmt.item).lstrip()
        return [f"{pad}{line}"]
    if isinstance(stmt, CallStmt):
        return [f"{pad}call {stmt.target}                ; {stmt.src.short()}"]
    if isinstance(stmt, TailCallStmt):
        return [f"{pad}tail_call {stmt.target}           ; {stmt.src.short()}"]
    if isinstance(stmt, ReturnStmt):
        return [f"{pad}return                           ; {stmt.src.short()}"]
    if isinstance(stmt, IfStmt):
        lines = [f"{pad}if {_fmt_compare(stmt.cond)} {{    ; {stmt.src.short()}"]
        for s in stmt.then_block.stmts:
            lines.extend(_fmt_stmt(s, indent + 1))
        if stmt.else_block is not None:
            lines.append(f"{pad}}} else {{")
            for s in stmt.else_block.stmts:
                lines.extend(_fmt_stmt(s, indent + 1))
        lines.append(f"{pad}}}")
        return lines
    if isinstance(stmt, RawIfStmt):
        lines = [f"{pad}if {stmt.cond} {{                ; {stmt.src.short()}"]
        for s in stmt.then_block.stmts:
            lines.extend(_fmt_stmt(s, indent + 1))
        if stmt.else_block is not None:
            lines.append(f"{pad}}} else {{")
            for s in stmt.else_block.stmts:
                lines.extend(_fmt_stmt(s, indent + 1))
        lines.append(f"{pad}}}")
        return lines
    if isinstance(stmt, GotoStmt):
        return [f"{pad}goto {stmt.target}              ; {stmt.src.short()}"]
    if isinstance(stmt, LabelStmt):
        return [f"{pad}{stmt.name}:"]
    raise ValueError(f"unknown IR3 stmt: {stmt!r}")


def format_routine(routine: RoutineIR3) -> str:
    aliases = ""
    if routine.entry_aliases:
        aliases = " " + ", ".join(routine.entry_aliases)
    lines = [f"fn {routine.name}{aliases} {{"]
    for s in routine.body.stmts:
        lines.extend(_fmt_stmt(s, 1))
    lines.append("}")
    return "\n".join(lines)


def format_module(module: ModuleIR3) -> str:
    # Mirror ir1.format_module's portable-path behaviour so committed
    # artifacts diff identically across machines / CI checkouts.
    from .ir1 import _portable_path
    parts = [f"; module {module.name} from {_portable_path(module.file)}"]
    parts.append("")
    for r in module.routines:
        parts.append(format_routine(r))
        parts.append("")
    return "\n".join(parts)
