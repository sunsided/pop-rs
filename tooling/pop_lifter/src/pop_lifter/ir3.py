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
from typing import TYPE_CHECKING

from .ir1 import Compare, Item, SourceRef

if TYPE_CHECKING:
    # Referenced only in the string annotations on `Assign` / `BinExpr`
    # / `MatchStmt` (the concrete value types live in ir1; ir3 imports
    # them lazily inside the formatter to avoid a heavier import at
    # module load).
    from .ir1 import Abs, Imm, IndexedAbs, IndirectY, Reg


# A `RawStmt` wraps a non-control-flow IR1/IR2 item. The relooper
# never inspects the inner item — it just passes it through. The IR3
# interpreter delegates the actual operation to the existing IR1
# interpreter machinery (per-opcode dispatch in `interp_ir1`).
@dataclass(frozen=True)
class RawStmt:
    item: Item


@dataclass(frozen=True)
class BinExpr:
    """`lhs op rhs` — a pass-3 folded 8-bit arithmetic expression.
    Collapses the accumulator compute-and-store idiom
    `a = LHS ; clc ; adc RHS ; sta DST` (and the `sec ; sbc` subtract
    form) into a single `DST = LHS + RHS` assignment, dropping the
    load, the carry set-up, and the add/sub.

    `op` is `"+"` (clc + adc) or `"-"` (sec + sbc) — the carry set-up
    pins the operation to pure 8-bit add / subtract, with the result
    wrapping mod 256. `lhs` is the dropped load's value; `rhs` is the
    add/sub operand. Both are drawn from the same value forms as
    `Assign.source` (`Imm` / `Abs` / `IndexedAbs` / `IndirectY`).

    Produced only when A *and* the carry are dead after the store, so
    the dropped flag side-effects can't be observed (the IR3
    interpreter checks this via the differential tests)."""

    op: str  # "+" | "-"
    lhs: "Imm | Abs | IndexedAbs | IndirectY"
    rhs: "Imm | Abs | IndexedAbs | IndirectY"


@dataclass(frozen=True)
class Assign:
    """`target = source` — a pass-3 folded copy. Collapses the
    accumulator round-trip `a = SRC ; sta DST` (and the multi-store
    `a = #k ; sta X ; sta Y`) into a direct memory assignment,
    dropping the intermediate load.

    `target` is a store destination drawn from the IR1 value types
    (`Abs` for `sta addr`, `IndexedAbs` for `sta tbl,x`, `IndirectY`
    for `sta (ptr),y`). `source` is the value the dropped load
    produced — an `Imm` (`lda #k`) or one of the same memory-read
    forms (`Abs` / `IndexedAbs` / `IndirectY`) — or a `BinExpr` when
    a `clc ; adc` / `sec ; sbc` was folded in (slice 2).

    Produced only when the load's value flows *exclusively* into the
    store(s) and `A` is dead afterwards, so the fold is
    behaviour-preserving (the IR3 interpreter checks this via the
    differential tests)."""

    target: "Abs | IndexedAbs | IndirectY"
    source: "Imm | Abs | IndexedAbs | IndirectY | BinExpr"
    src: SourceRef


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
class LoopStmt:
    """`loop { body }` — an infinite loop with explicit exits via
    `BreakStmt`. The relooper produces this for the classic 6502
    do-while-with-counter pattern (`:hdr ... dex ; bpl :hdr`):

        loop {
            ...body...
            if !exit_cond { break; }
        }

    The exit guard sits at the bottom of `body` since 6502 do-while
    loops always evaluate the continue condition after the body.
    Pass 3 / pass 4 may rewrite into `for`/`while` shapes when an
    induction variable is recognised."""

    body: "Block"
    src: SourceRef


@dataclass(frozen=True)
class DoWhileStmt:
    """`do { body } while cond` — pass-3 loop-condition recovery. The
    relooper emits 6502 do-while loops as `loop { body ; if exit {
    break } }` (the exit test is always at the bottom). This hoists that
    trailing guard into the loop header, dropping the `if … { break }`
    boilerplate and naming the loop's continue condition (the negation
    of the exit test).

    Semantics match the original `loop` exactly: run `body`; on a
    `break` exit; on a `continue` restart `body` from the top *without*
    testing `cond` (the 6502 back-edge skips the bottom guard); on
    normal completion, repeat while `cond` holds. So `cond` is the
    *continue* condition — e.g. an exit test of `y < 0` becomes
    `while y >= 0`."""

    body: "Block"
    cond: Compare
    src: SourceRef


@dataclass(frozen=True)
class BreakStmt:
    """Exit the innermost enclosing `LoopStmt`."""
    src: SourceRef


@dataclass(frozen=True)
class ContinueStmt:
    """Jump to the top of the innermost enclosing `LoopStmt`."""
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


@dataclass(frozen=True)
class MatchArm:
    """One arm of a `MatchStmt`: `values => body`. Multiple constants
    map to one arm (`K1 | K2 => ...`) when the dispatch branched several
    keys to the same handler (the 6502 idiom shares one target via tail
    duplication, so the bodies come out identical)."""

    values: "tuple[Imm, ...]"
    body: "Block"


@dataclass(frozen=True)
class MatchStmt:
    """`match reg { K => body, ... }` — pass-3 recognition of the 6502
    jump-table dispatch idiom. Collapses a run of consecutive
    `if reg == K { terminating_body }` (distinct constant keys, each
    body ending in a return / tail-call / break / continue) into a
    single multi-way branch.

    Behaviour-preserving: the keys are distinct so at most one arm
    matches, and every arm terminates so there's no fall-through between
    cases. If no arm matches, control falls through to the statement
    after the `MatchStmt` — exactly the `if`-chain's behaviour, so the
    chain's fall-through tail stays where it was (an implicit default)."""

    reg: Reg
    arms: "tuple[MatchArm, ...]"
    src: SourceRef


Stmt = (
    RawStmt | Assign | CallStmt | TailCallStmt | ReturnStmt
    | IfStmt | RawIfStmt | LoopStmt | DoWhileStmt | MatchStmt
    | BreakStmt | ContinueStmt | GotoStmt | LabelStmt
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
    # Delegate Imm rendering to ir1's `_fmt_imm` so symbolic operand
    # names (e.g. `#block`) propagate from the original `cmp` into the
    # structured `if` form. Previously this hard-coded the numeric
    # format, which silently dropped the symbolic intent — a `cmp a,
    # #block` followed by `if a != #0x14 { ... }` would mis-render
    # because the imm was rebuilt with the value-only template.
    from .ir1 import Abs, Imm, IndexedAbs, _fmt_imm
    if c.rhs is None:
        return f"{c.reg} {c.op}"
    if isinstance(c.rhs, Imm):
        return f"{c.reg} {c.op} {_fmt_imm(c.rhs)}"
    if isinstance(c.rhs, IndexedAbs):
        return f"{c.reg} {c.op} *({c.rhs.base.name}@{c.rhs.base.addr:#06x} + {c.rhs.index})"
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
    if isinstance(stmt, Assign):
        from .ir1 import Abs, Imm, IndexedAbs, IndirectY, _fmt_abs, _fmt_imm

        def _loc(v) -> str:
            if isinstance(v, Imm):
                return _fmt_imm(v)
            if isinstance(v, IndexedAbs):
                return f"*({_fmt_abs(v.base)} + {v.index})"
            if isinstance(v, IndirectY):
                return f"*({_fmt_abs(v.ptr)})[y]"
            if isinstance(v, Abs):
                return f"*{_fmt_abs(v)}"
            if isinstance(v, BinExpr):
                return f"{_loc(v.lhs)} {v.op} {_loc(v.rhs)}"
            return repr(v)

        return [f"{pad}{_loc(stmt.target)} = {_loc(stmt.source)}    ; {stmt.src.short()}"]
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
    if isinstance(stmt, LoopStmt):
        lines = [f"{pad}loop {{                          ; {stmt.src.short()}"]
        for s in stmt.body.stmts:
            lines.extend(_fmt_stmt(s, indent + 1))
        lines.append(f"{pad}}}")
        return lines
    if isinstance(stmt, DoWhileStmt):
        lines = [f"{pad}do {{                            ; {stmt.src.short()}"]
        for s in stmt.body.stmts:
            lines.extend(_fmt_stmt(s, indent + 1))
        lines.append(f"{pad}}} while {_fmt_compare(stmt.cond)}")
        return lines
    if isinstance(stmt, MatchStmt):
        from .ir1 import _fmt_imm
        lines = [f"{pad}match {stmt.reg} {{    ; {stmt.src.short()}"]
        for arm in stmt.arms:
            keys = " | ".join(_fmt_imm(v) for v in arm.values)
            lines.append(f"{pad}  {keys} => {{")
            for s in arm.body.stmts:
                lines.extend(_fmt_stmt(s, indent + 2))
            lines.append(f"{pad}  }}")
        lines.append(f"{pad}}}")
        return lines
    if isinstance(stmt, BreakStmt):
        return [f"{pad}break                            ; {stmt.src.short()}"]
    if isinstance(stmt, ContinueStmt):
        return [f"{pad}continue                         ; {stmt.src.short()}"]
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
