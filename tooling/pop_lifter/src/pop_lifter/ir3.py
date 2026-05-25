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
    """`lhs op rhs` — a pass-3 folded 8-bit expression.

    Arithmetic form: collapses `a = LHS ; clc ; adc RHS ; sta DST`
    (and the `sec ; sbc` subtract form) into `DST = LHS + RHS`.
    `op` is `"+"` (clc + adc) or `"-"` (sec + sbc).

    Shift form: collapses `a = LHS ; (asl)*n ; sta DST` (or `lsr`)
    into `DST = LHS << n` (or `>>`). `op` is `"<<"` or `">>"`;
    `rhs` is an `Imm` holding the shift count.

    All forms: produced only when A *and* carry are dead after the
    store, so the dropped flag side-effects can't be observed."""

    op: str  # "+" | "-" | "<<" | ">>"
    lhs: "Imm | Abs | IndexedAbs | IndirectY"
    rhs: "Imm | Abs | IndexedAbs | IndirectY"


@dataclass(frozen=True)
class RotateExpr:
    """Accumulator rotate folded into a pass-3 expression.

    Collapses `lda X ; (rol)*n ; sta DST` (or `ror`) into
    `DST = rotl(X, n)` (or `rotr`). `op` is `"rotl"` (from
    consecutive `rol`) or `"rotr"` (from consecutive `ror`).
    `count` is the number of rotate instructions.

    Unlike `BinExpr` shifts, the carry flag is an *input*: at
    evaluation time the interpreter reads the current carry from
    the trace and feeds it into the first rotation step; successive
    steps use the carry produced by the previous step. The final
    carry-out is not written back — by the fold's soundness
    condition carry is dead after the store."""

    op: str   # "rotl" | "rotr"
    operand: "Imm | Abs | IndexedAbs | IndirectY"
    count: int


@dataclass(frozen=True)
class Wide16Stmt:
    """16-bit arithmetic — a pass-3 structural rewrite of the 6502 idiom:

        lda lo_src ; clc/sec ; adc/sbc lo_op ; sta lo_dst
        lda hi_src ; adc/sbc hi_op          ; sta hi_dst

    where the second `adc`/`sbc` has no preceding carry set-up — it
    consumes the carry produced by the low-byte operation.  `op` is
    `"+"` (clc + adc pair) or `"-"` (sec + sbc pair).

    Unlike the 8-bit `Assign`/`BinExpr` fold, no dead-after check is
    required: this is a *structural* replacement that preserves every
    side-effect of all seven instructions (both memory writes, the
    final A value = hi-byte result, and the final carry = carry out of
    the hi-byte operation).

    Combined semantics:
        lo_dst = lo_src ± lo_op            (lo byte, no carry-in)
        hi_dst = hi_src ± hi_op ± carry    (hi byte, carry from lo)
    i.e. `{hi_dst:lo_dst} = {hi_src:lo_src} ± {hi_op:lo_op}` — an
    exact 16-bit add (or subtract-with-borrow), result wrapping mod
    65536."""

    op: str  # "+" | "-"
    lo_src: "Imm | Abs | IndexedAbs | IndirectY"
    lo_op:  "Imm | Abs | IndexedAbs | IndirectY"
    lo_dst: "Abs | IndexedAbs | IndirectY"
    hi_src: "Imm | Abs | IndexedAbs | IndirectY"
    hi_op:  "Imm | Abs | IndexedAbs | IndirectY"
    hi_dst: "Abs | IndexedAbs | IndirectY"
    src: SourceRef


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
    source: "Imm | Abs | IndexedAbs | IndirectY | BinExpr | RotateExpr"
    src: SourceRef


@dataclass(frozen=True)
class SaveTemp:
    """`pha` recovered as a scoped-temporary save: stash A into temp
    `slot`. Pairs with a `RestoreTemp` of the same `slot` further down
    the same routine (pass-3 `pass3_temps` matches the pair under the
    stack's LIFO discipline).

    `slot` is a per-routine id used only to name the temporary in the
    dump / future Rust emission (`let tmp{slot} = a;`). The runtime
    value still rides the one shared value stack, so the interpreter
    treats this exactly like the `pha` it replaced — the rewrite is
    behaviour-preserving regardless of how slots were numbered."""

    slot: int
    src: SourceRef


@dataclass(frozen=True)
class RestoreTemp:
    """`pla` recovered as a scoped-temporary restore: `a = temp{slot}`,
    also setting Z/N from the restored value (matching `pla`). Pairs
    with the `SaveTemp` of the same `slot`. See `SaveTemp` for the
    slot-vs-stack rationale."""

    slot: int
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
class ForStmt:
    """A pass-3 recovered counted loop. Two 6502 idioms map here:

    * **down-counter** (`ldy #N : … : dey : bpl`) → `for var in
      (0..=start).rev()` — `step = -1`, continue condition `var >= 0`,
      `start` non-negative.
    * **up-counter** (`ldx #i : … : inx : cpx #N : bne`) → `for var in
      start..N` — `step = +1`, continue condition `var != N`, with
      `start < N` so the range doesn't wrap.

    Recognised from a `DoWhileStmt`

        var = #start       (immediately before the loop)
        do {
            body
            var ±= 1       (the step, last in the do-while body)
        } while <cond>

    where `var` is the loop counter only (never otherwise written in
    `body`, no `continue`, no calls) and the start value makes the loop
    run at least once (so the top-tested `for` matches the bottom test).

    The init `LoadImm` and the trailing `inx`/`dey` step are subsumed
    into this node; the interpreter replays the init, the step, and
    `cond` so register and flag state are identical to the do-while it
    replaced (e.g. a down-counter leaves `var = 0xff`)."""

    var: Reg
    start: Imm
    step: int          # +1 (up) or -1 (down)
    cond: Compare      # continue condition (the do-while's): `var >= 0` or `var != N`
    body: "Block"
    src: SourceRef


@dataclass(frozen=True)
class RepeatStmt:
    """`repeat count { body }` — a fixed-count loop, recovered from the
    6502 full-wrap busy-wait idiom:

        var = #INIT
        do { body ; var -= 1 } while var != #INIT

    The counter exits only when it cycles back to its start, so a
    `dex`/`dey` (or `inx`/`iny`) runs the body exactly 256 times
    regardless of `INIT`. Recognised only when `var` is the counter
    only (the body never reads or writes it, no `break`/`continue`/
    calls), so `count` fully captures the loop — the classic timing
    delay (`PAUSE`).

    The interpreter replays the init, body, and step `count` times, so
    `var` ends back at `INIT` and flag state matches the original."""

    count: int
    var: Reg
    start: Imm
    step: int          # +1 / -1
    body: "Block"
    src: SourceRef


@dataclass(frozen=True)
class BreakStmt:
    """Exit an enclosing block. With `label=None` it exits the innermost
    loop (`LoopStmt` / `DoWhileStmt`); with a `label` it exits the named
    `LabeledBlock` — the relooper's structured forward jump to a merge
    node (Rust `break 'label`)."""
    src: SourceRef
    label: str | None = None


@dataclass(frozen=True)
class LabeledBlock:
    """A named, single-entry scope the relooper wraps around a region so
    that a forward edge to the region's merge node becomes a structured
    `break 'label` to just after the block, letting the merge node be
    emitted exactly once (instead of duplicated into every predecessor).
    Lowers to a Rust labeled block: `'label: { <body> }`."""

    label: str
    body: "Block"
    src: SourceRef


@dataclass(frozen=True)
class ContinueStmt:
    """Jump to the top of the innermost enclosing loop statement
    (`LoopStmt` or, after recovery, `DoWhileStmt` — restarting its body
    without re-testing the bottom condition)."""
    src: SourceRef


@dataclass(frozen=True)
class GotoStmt:
    """Intermediate escape hatch for a transfer a structurer couldn't
    place (a loop back-edge or irreducible fragment). It never survives
    to a finished routine: `reloop_routine` rescans each structurer's
    output and, if any `GotoStmt`/`LabelStmt` remains, re-emits the
    whole routine as a `DispatchStmt` (`loop { match pc { ... } }`). The
    emitter and interpreter therefore reject it — reaching either is a
    structurer bug."""

    target: str
    src: SourceRef


@dataclass(frozen=True)
class LabelStmt:
    """Companion to `GotoStmt`. Same caveats."""

    name: str
    src: SourceRef


@dataclass(frozen=True)
class GotoStateStmt:
    """Set the enclosing `DispatchStmt`'s state variable to `state` and
    re-enter its `match`. Models one CFG edge as a state transition in
    the dispatch-loop fallback. Lowers to `pc = <state>;` (control then
    falls off the match arm, so the surrounding `loop` re-dispatches).
    Only ever appears inside a `DispatchStmt` arm."""

    state: int
    src: SourceRef


@dataclass(frozen=True)
class DispatchArm:
    """One state of a `DispatchStmt`: `state => { body }`. `body` runs
    the basic block's atoms then ends in a transition — a
    `GotoStateStmt`, an `IfStmt`/`RawIfStmt` whose arms transition, a
    `ReturnStmt`, or a `TailCallStmt`."""

    state: int
    body: "Block"


@dataclass(frozen=True)
class DispatchStmt:
    """`loop { match pc { state => {...} } }` — the relooper's universal
    fallback for routines whose control flow can't be reduced to natural
    loops/conditionals (irreducible CFGs, multi-back-edge loops, loops
    with mid-body exits). Each CFG basic block becomes one numbered
    state (its block id); every control-flow edge becomes a `pc = next`
    transition (`GotoStateStmt`). Replaces the older `GotoStmt` /
    `LabelStmt` escape hatch, so a fallback routine still emits valid
    structured Rust rather than unresolved gotos.

    `entry` is the starting state (the CFG entry block id). Behaviour is
    preserved 1-for-1 with the IR2 CFG: the dispatch visits blocks in
    exactly the order the original gotos/branches would have."""

    entry: int
    arms: "tuple[DispatchArm, ...]"
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
    RawStmt | Wide16Stmt | Assign | SaveTemp | RestoreTemp | CallStmt
    | TailCallStmt | ReturnStmt | IfStmt | RawIfStmt | LoopStmt | DoWhileStmt
    | ForStmt | RepeatStmt | MatchStmt | BreakStmt | ContinueStmt | GotoStmt
    | LabelStmt | LabeledBlock | GotoStateStmt | DispatchStmt
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
    if isinstance(stmt, Wide16Stmt):
        from .ir1 import Abs, Imm, IndexedAbs, IndirectY, _fmt_abs, _fmt_imm

        def _loc16(v) -> str:
            if isinstance(v, Imm):
                return _fmt_imm(v)
            if isinstance(v, IndexedAbs):
                return f"*({_fmt_abs(v.base)} + {v.index})"
            if isinstance(v, IndirectY):
                return f"*({_fmt_abs(v.ptr)})[y]"
            if isinstance(v, Abs):
                return f"*{_fmt_abs(v)}"
            return repr(v)

        op_c = f"{stmt.op}c"  # "+c" or "-c" — carry-in from the lo byte
        return [
            f"{pad}{_loc16(stmt.lo_dst)} = {_loc16(stmt.lo_src)} {stmt.op} {_loc16(stmt.lo_op)}    ; {stmt.src.short()}",
            f"{pad}{_loc16(stmt.hi_dst)} = {_loc16(stmt.hi_src)} {op_c} {_loc16(stmt.hi_op)}    ; [wide16]",
        ]
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
            if isinstance(v, RotateExpr):
                return f"{v.op}({_loc(v.operand)}, {v.count})"
            return repr(v)

        return [f"{pad}{_loc(stmt.target)} = {_loc(stmt.source)}    ; {stmt.src.short()}"]
    if isinstance(stmt, SaveTemp):
        return [f"{pad}tmp{stmt.slot} = a    ; {stmt.src.short()} [save]"]
    if isinstance(stmt, RestoreTemp):
        return [f"{pad}a = tmp{stmt.slot}    ; {stmt.src.short()} [restore]"]
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
    if isinstance(stmt, ForStmt):
        from .ir1 import _fmt_imm
        # Delegate bounds to ir1's immediate formatter so a symbolic
        # bound (`#numslots`, etc.) survives instead of its assembled byte.
        if stmt.step < 0:
            rng = f"(0..={_fmt_imm(stmt.start)}).rev()"
        else:
            rng = f"{_fmt_imm(stmt.start)}..{_fmt_imm(stmt.cond.rhs)}"
        lines = [f"{pad}for {stmt.var} in {rng} {{    ; {stmt.src.short()}"]
        for s in stmt.body.stmts:
            lines.extend(_fmt_stmt(s, indent + 1))
        lines.append(f"{pad}}}")
        return lines
    if isinstance(stmt, RepeatStmt):
        lines = [f"{pad}repeat {stmt.count:#06x} {{    ; {stmt.src.short()}"]
        for s in stmt.body.stmts:
            lines.extend(_fmt_stmt(s, indent + 1))
        lines.append(f"{pad}}}")
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
    if isinstance(stmt, LabeledBlock):
        lines = [f"{pad}{stmt.label}: {{"]
        for s in stmt.body.stmts:
            lines.extend(_fmt_stmt(s, indent + 1))
        lines.append(f"{pad}}}")
        return lines
    if isinstance(stmt, BreakStmt):
        target = f" {stmt.label}" if stmt.label else ""
        return [f"{pad}break{target}                            ; {stmt.src.short()}"]
    if isinstance(stmt, ContinueStmt):
        return [f"{pad}continue                         ; {stmt.src.short()}"]
    if isinstance(stmt, GotoStmt):
        return [f"{pad}goto {stmt.target}              ; {stmt.src.short()}"]
    if isinstance(stmt, LabelStmt):
        return [f"{pad}{stmt.name}:"]
    if isinstance(stmt, GotoStateStmt):
        return [f"{pad}pc = {stmt.state}                          ; {stmt.src.short()}"]
    if isinstance(stmt, DispatchStmt):
        lines = [f"{pad}dispatch pc = {stmt.entry} {{    ; {stmt.src.short()}"]
        for arm in stmt.arms:
            lines.append(f"{pad}  state {arm.state} => {{")
            for s in arm.body.stmts:
                lines.extend(_fmt_stmt(s, indent + 2))
            lines.append(f"{pad}  }}")
        lines.append(f"{pad}}}")
        return lines
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
