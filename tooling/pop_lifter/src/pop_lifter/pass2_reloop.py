"""Pass 2 (final phase): reloop the goto-flow IR2 into structured IR3.

Once `pass2_struct` has fused `cmp + branch` pairs and elided the
resulting dead flag-setters, every routine body is a sequence of
straight-line atoms separated by `If` / `Branch` / `Goto` / `Return`
terminators. The relooper takes that body, splits it into a basic-
block CFG (via `cfg.build_cfg`), and reshapes the CFG into a tree of
`Block` / `IfStmt` / `ReturnStmt` / `TailCallStmt` IR3 statements.

What this slice covers — enough for CHECKFLOOR's shape:

* **Early-exit `if`**: `if cond goto ]rts` (where ]rts is a `return`
  block) becomes `if cond { return }` inlined at the call site.
* **Conditional tail call** (cross-module taken edge of `If` /
  `Branch`): `if a == 4 goto falling` becomes
  `if a == 4 { tail_call falling; }`.
* **Local goto into a dominated block**: inlined as the continuation
  of the current block.
* **Unconditional `tail_call`** and **`return`**: emitted as the
  terminating statement of their enclosing block.
* **Unfused `Branch`**: wrapped in a `RawIfStmt` so the routine still
  structures. Pass 3 will revisit those once the lifter recognises
  the missing flag-setters.

Loop handling (this slice):

* **Simple do-while**: a single back-edge `(s, t)` where `t` is the
  loop header, the block range `[t..s]` is contiguous and contains
  no other exits, and `s`'s terminator is a conditional jump with
  one edge back to `t` (continue) and one edge forward (break) ⇒
  emitted as `LoopStmt { body... ; if !cond { break } }`. The exit
  guard sits at the bottom because 6502 do-while loops evaluate
  the continue condition after the body. Covers the classic 6502
  counter loop (`:hdr ... dex ; bpl :hdr`).

Loop handling (out of scope, still pending):

* Multiple back-edges to the same header (nested or unstructured).
* Loops with mid-body exits (`break` not at the bottom).
* While-style loops where the cond is checked at the TOP (rare in
  6502 — usually surfaces as a forward `bcc` over the body plus a
  `jmp` back at the bottom).
* Irreducible flow (back-edge target doesn't dominate the source).
  POP's combat / physics code has a few of these; they'll need the
  `loop { match pc { ... } }` dispatcher fallback from the plan.

When the relooper can't structure a routine (or any loop within it)
it falls back via `_wrap_unstructured` — see below. The fallback
walks the IR2 body in order and emits a 1-for-1 IR3 stream
(`Label`/`Goto`/`Return`/`TailCall`/`Call` map to their IR3
counterparts; `If`/`Branch` become a structured `IfStmt`/`RawIfStmt`
whose then-block holds a `GotoStmt` to the original target; atoms
become `RawStmt`). Correctness preserved, structure not improved.
* Post-dominator analysis. The current algorithm may emit a small
  amount of code duplication when a block is reached from both the
  taken edge of an `if` and the fall-through path of the next block.
  CHECKFLOOR's `:ong` (`tail_call onground`) is the only such case
  in the pilot — fine for now; a future pass will collapse it.
* `match` / `switch` recognition from chained `if a == K` —
  pass 3's job.
"""

from __future__ import annotations

from dataclasses import replace as dataclass_replace

from .cfg import (
    CFG,
    build_cfg,
    compute_idoms,
    dominates,
    find_back_edges,
    natural_loop_body,
)
from .ir1 import (
    Branch,
    Compare,
    Goto,
    If,
    Label,
    ModuleIR1,
    Return,
    Routine,
)
from .ir1 import (
    Call as IR1Call,
)
from .ir3 import (
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
)


# ----------------------------------------------------------------------
# Loop discovery — simple do-while shapes only.
#
# A *simple do-while* in IR2 is the classic 6502 counter loop:
#
#     :header           (block T)
#     ... body ...
#     ... more body ... (block S, terminator is `If`/`Branch` taken=T)
#     :after            (S+1, fall-through-out)
#
# Recognised if and only if:
#
# 1. There is exactly one back-edge `(S, T)`.
# 2. `T` dominates `S` (reducible).
# 3. `T..S` is contiguous in block order and S is the back-edge source.
# 4. `S`'s terminator is a conditional `If` or `Branch` with taken=T
#    (continue) and fall-through=S+1 (break).
# 5. No other block in `[T..S]` exits the loop (every successor stays
#    inside `[T..S]` or is the back-edge to T).
#
# When all five hold we emit a `LoopStmt` whose body is the
# restructured range `[T..S-1]` followed by the synthesized exit guard
# at the bottom — the natural Rust shape:
#
#     loop {
#         ...body...
#         if !cond { break; }
#     }


from typing import NamedTuple


class SimpleLoop(NamedTuple):
    header: int          # block id of the loop entry (= back-edge target)
    tail: int            # back-edge source — the conditional at the bottom
    exit_block: int      # block immediately after the tail (fall-through)
    # The exit guard at the bottom of the loop body. Recorded so the
    # relooper can emit it with the cond inverted.
    cond: object         # ir1.Compare or str (Branch cond suffix)
    cond_is_compare: bool


def _detect_simple_loops(cfg: CFG) -> tuple[dict[int, SimpleLoop], set[int]]:
    """Find every simple do-while loop in `cfg`. Returns a mapping from
    header block id to `SimpleLoop` plus the set of *all* blocks that
    belong to a recognised loop body (header through tail). Anything
    in `loop_blocks` is consumed when the parent walker reaches the
    header; nothing else should structure them."""
    if not cfg.blocks:
        return {}, set()
    try:
        idom = compute_idoms(cfg)
    except Exception:
        return {}, set()
    back = find_back_edges(cfg)
    loops: dict[int, SimpleLoop] = {}
    loop_blocks: set[int] = set()

    # Detect headers that are entered by more than one back-edge — we
    # don't structure those in this slice.
    header_counts: dict[int, int] = {}
    for _, h in back:
        header_counts[h] = header_counts.get(h, 0) + 1

    for src, hdr in back:
        if header_counts.get(hdr, 0) != 1:
            continue
        if not dominates(idom, hdr, src):
            continue
        # Contiguous block range?
        if src < hdr:
            continue
        body = set(range(hdr, src + 1))
        # Tail's terminator must be a conditional with taken=hdr
        # and fall-through=src+1 (= exit_block).
        term = cfg.blocks[src].terminator
        if not isinstance(term, (Branch, If)):
            continue
        if cfg.label_to_block.get(term.target) != hdr:
            continue
        exit_block = src + 1
        if exit_block >= len(cfg.blocks):
            # No fall-through — shape isn't a do-while.
            continue
        # Every block in the body must only have successors inside
        # `body` OR be exactly the back-edge to `hdr` OR — for the
        # tail — the exit_block. Anything else is a mid-body exit
        # which we don't structure yet.
        clean = True
        for b in body:
            for s in cfg.succ[b]:
                if s in body:
                    continue
                if b == src and s == exit_block:
                    continue
                clean = False
                break
            if not clean:
                break
        if not clean:
            continue
        # Make sure none of the blocks already belong to another
        # recognised loop (nested loops aren't handled in this slice).
        if any(b in loop_blocks for b in body):
            continue

        loops[hdr] = SimpleLoop(
            header=hdr,
            tail=src,
            exit_block=exit_block,
            cond=term.cond,
            cond_is_compare=isinstance(term, If),
        )
        loop_blocks.update(body)

    return loops, loop_blocks


def reloop_routine(routine: Routine) -> RoutineIR3:
    """Structure `routine` into an IR3 routine. Simple do-while loops
    are recognised and emitted as `LoopStmt`; anything else with a
    backward local jump falls back to `_wrap_unstructured`.
    """
    cfg = build_cfg(routine)

    # All back-edges must belong to a recognised simple loop — any
    # leftover back-edge means we can't soundly do a single-pass walk,
    # and the routine takes the fallback path.
    back = find_back_edges(cfg) if cfg.blocks else []
    loops, loop_blocks = _detect_simple_loops(cfg)
    if back and not all(
        src in loop_blocks and dst in loop_blocks for src, dst in back
    ):
        return _wrap_unstructured(routine)

    visiting: set[int] = set()
    body = _emit_block(
        cfg, cfg.entry_id,
        exit_id=None,
        visiting=visiting,
        loops=loops,
        loop_blocks=loop_blocks,
        active_loop_tail=None,
    )
    return RoutineIR3(
        name=routine.name,
        entry_aliases=list(routine.entry_aliases),
        body=body,
    )


def reloop_module(module: ModuleIR1) -> ModuleIR3:
    return ModuleIR3(
        name=module.name,
        file=module.file,
        routines=[reloop_routine(r) for r in module.routines],
    )


def is_unstructured(routine: RoutineIR3) -> bool:
    """True if `routine` came out of the fallback path — its body
    contains a `GotoStmt` or a `LabelStmt`. Reported by the CLI so
    callers can see at a glance how many routines pass 2 had to
    punt on."""
    def walk(stmts) -> bool:
        for s in stmts:
            if isinstance(s, (GotoStmt, LabelStmt)):
                return True
            inner_then = getattr(s, "then_block", None)
            if inner_then is not None and walk(inner_then.stmts):
                return True
            inner_else = getattr(s, "else_block", None)
            if inner_else is not None and walk(inner_else.stmts):
                return True
        return False
    return walk(routine.body.stmts)


def _wrap_unstructured(routine: Routine) -> RoutineIR3:
    """Fallback path for routines the relooper can't structure (loops,
    irreducible flow). Walks the IR2 body in order and emits a 1-for-1
    IR3 stream. The result preserves semantics — every IR1/IR2 atom
    has an IR3 representation — but the routine remains a sequence of
    gotos and labels rather than a tree of structured blocks.

    Per-item mapping:

    * `Label` → `LabelStmt`.
    * `Return` → `ReturnStmt`.
    * `Goto(tail_call)` → `TailCallStmt`. `Goto(local)` → `GotoStmt`.
    * `Call` → `CallStmt`. (Was previously folded into `RawStmt`,
       which left downstream consumers without a structured handle on
       the call.)
    * `If`/`Branch` whose target is **local** (some `Label` in this
       routine's body) → `IfStmt`/`RawIfStmt` whose then-block holds
       a `GotoStmt` to that label.
    * `If`/`Branch` whose target is **not local** → conditional
       tail-call: `IfStmt`/`RawIfStmt` whose then-block holds a
       `TailCallStmt`. IR1 executes a non-local branch by switching
       routines (see `interp_ir1`'s Branch/If handlers), so an
       unconditional `GotoStmt` here would silently change semantics
       for routines that take the fallback.
    * Anything else (loads/stores/arithmetic/cmp/clc/sec/...) →
       `RawStmt`.
    """
    local_labels = {
        item.name for item in routine.body if isinstance(item, Label)
    }
    stmts: list[Stmt] = []
    for item in routine.body:
        if isinstance(item, Label):
            stmts.append(LabelStmt(name=item.name, src=item.src))
        elif isinstance(item, Return):
            stmts.append(ReturnStmt(src=item.src))
        elif isinstance(item, IR1Call):
            stmts.append(CallStmt(target=item.target, src=item.src))
        elif isinstance(item, Goto):
            if item.kind == "tail_call":
                stmts.append(TailCallStmt(target=item.target, src=item.src))
            else:
                stmts.append(GotoStmt(target=item.target, src=item.src))
        elif isinstance(item, (If, Branch)):
            taken_is_local = item.target in local_labels
            then_stmt: Stmt = (
                GotoStmt(target=item.target, src=item.src)
                if taken_is_local
                else TailCallStmt(target=item.target, src=item.src)
            )
            then_block = Block.of([then_stmt])
            if isinstance(item, If):
                stmts.append(IfStmt(
                    cond=item.cond,
                    then_block=then_block,
                    else_block=None,
                    src=item.src,
                ))
            else:
                stmts.append(RawIfStmt(
                    cond=item.cond,
                    then_block=then_block,
                    else_block=None,
                    src=item.src,
                ))
        else:
            stmts.append(RawStmt(item=item))
    return RoutineIR3(
        name=routine.name,
        entry_aliases=list(routine.entry_aliases),
        body=Block.of(stmts),
    )


_INVERT_BRANCH_COND: dict[str, str] = {
    "eq": "ne", "ne": "eq",
    "cs": "cc", "cc": "cs",
    "pl": "mi", "mi": "pl",
    "vs": "vc", "vc": "vs",
}


_INVERT_COMPARE_OP: dict[str, str] = {
    "==": "!=", "!=": "==",
    "<": ">=", ">=": "<",
    "<0": ">=0", ">=0": "<0",
}


def _invert_loop_exit(cond, cond_is_compare: bool):
    """Invert the tail's continue-condition into a break-condition.
    The tail's `If`/`Branch` originally said "loop back to header if
    cond"; the structured form needs "break if NOT cond" at the
    bottom of the body."""
    if cond_is_compare:
        assert isinstance(cond, Compare)
        return dataclass_replace(cond, op=_INVERT_COMPARE_OP[cond.op])
    return _INVERT_BRANCH_COND[cond]


def _emit_block(
    cfg: CFG,
    bid: int | None,
    exit_id: int | None,
    visiting: set[int],
    *,
    loops: dict[int, SimpleLoop],
    loop_blocks: set[int],
    active_loop_tail: int | None,
) -> Block:
    """Recursively emit IR3 stmts for block `bid` up to (but not
    including) `exit_id`. `exit_id=None` means "emit until the
    routine exits".

    `loops` / `loop_blocks` carry the pre-computed simple-loop
    layout. When the walker hits a loop header, it switches into
    loop-body mode (`active_loop_tail` set) and emits the contained
    blocks as a `LoopStmt`. Inside the loop body, the tail's
    conditional jump becomes the bottom-of-loop break guard; any
    other back-edge to the same header surfaces as `continue`.

    Code duplication note: when a block is reached from two paths
    (e.g. CHECKFLOOR's `:ong` from both `B2.taken` and `B3.fall-
    through`), this algorithm emits its body once per visit. The
    duplication is structurally innocuous (each emission terminates
    via tail-call, return, or break) but pass 3 may merge them once
    it has post-dominator data.
    """
    stmts: list[Stmt] = []
    while bid is not None and bid != exit_id:
        # If we've walked into a loop header, materialise the whole
        # loop as a single `LoopStmt` and resume after its exit.
        if (
            bid in loops
            and active_loop_tail != loops[bid].tail
        ):
            loop = loops[bid]
            loop_body = _emit_loop_body(
                cfg, loop, visiting, loops, loop_blocks,
            )
            stmts.append(LoopStmt(body=loop_body, src=cfg.blocks[bid].terminator.src))
            bid = loop.exit_block
            continue

        if bid in visiting:
            label = cfg.blocks[bid].label or f"BB{bid}"
            stmts.append(GotoStmt(target=label, src=cfg.blocks[bid].terminator.src))
            break
        visiting.add(bid)
        block = cfg.blocks[bid]

        # Body atoms (loads, stores, arithmetic, calls).
        for item in block.body:
            if isinstance(item, IR1Call):
                stmts.append(CallStmt(target=item.target, src=item.src))
            else:
                stmts.append(RawStmt(item=item))

        t = block.terminator
        if isinstance(t, Return):
            stmts.append(ReturnStmt(src=t.src))
            visiting.discard(bid)
            break

        if isinstance(t, Goto):
            if t.kind == "tail_call":
                stmts.append(TailCallStmt(target=t.target, src=t.src))
                visiting.discard(bid)
                break
            visiting.discard(bid)
            target_id = cfg.label_to_block.get(t.target)
            if target_id is None:
                stmts.append(GotoStmt(target=t.target, src=t.src))
                break
            # Inside a loop body: an unconditional goto to the loop
            # header is `continue`.
            if active_loop_tail is not None and target_id == _loop_for_tail(loops, active_loop_tail).header:
                stmts.append(ContinueStmt(src=t.src))
                break
            bid = target_id
            continue

        if isinstance(t, (If, Branch)):
            taken_label = t.target
            taken_id = cfg.label_to_block.get(taken_label)
            ft_id = bid + 1 if bid + 1 < len(cfg.blocks) else None

            then_stmts: list[Stmt]
            if taken_id is None:
                then_stmts = [TailCallStmt(target=taken_label, src=t.src)]
            elif active_loop_tail is not None and taken_id == _loop_for_tail(loops, active_loop_tail).header:
                # Conditional back-edge inside a loop body → continue.
                then_stmts = [ContinueStmt(src=t.src)]
            else:
                then_stmts = list(_emit_block(
                    cfg, taken_id, exit_id=ft_id, visiting=visiting,
                    loops=loops, loop_blocks=loop_blocks,
                    active_loop_tail=active_loop_tail,
                ).stmts)

            if isinstance(t, If):
                stmts.append(IfStmt(
                    cond=t.cond,
                    then_block=Block.of(then_stmts),
                    else_block=None,
                    src=t.src,
                ))
            else:
                stmts.append(RawIfStmt(
                    cond=t.cond,
                    then_block=Block.of(then_stmts),
                    else_block=None,
                    src=t.src,
                ))

            visiting.discard(bid)
            bid = ft_id
            continue

        raise AssertionError(f"unexpected terminator: {t!r}")

    return Block.of(stmts)


def _loop_for_tail(loops: dict[int, SimpleLoop], tail: int) -> SimpleLoop:
    """Helper: find the loop whose tail is `tail`. The lookup is
    linear over `loops` but `loops` is small (at most a handful per
    routine), so this isn't worth indexing."""
    for loop in loops.values():
        if loop.tail == tail:
            return loop
    raise AssertionError(f"no loop with tail={tail}")


def _emit_loop_body(
    cfg: CFG,
    loop: SimpleLoop,
    visiting: set[int],
    loops: dict[int, SimpleLoop],
    loop_blocks: set[int],
) -> Block:
    """Emit the body of a `LoopStmt` for the simple do-while `loop`.

    The walker emits the header through (tail - 1) using the normal
    machinery, then appends the tail's body atoms followed by the
    inverted exit guard:

        ...header body...
        ...intermediate blocks...
        ...tail body atoms...
        if !continue_cond { break; }

    The conditional terminator on the tail isn't recursed through —
    we rewrite it inline as the bottom-of-loop break.
    """
    body_stmts: list[Stmt] = []

    # Walk the header up to (but not including) the tail. The exit_id
    # is the tail's id so the walker stops just before it; we then
    # emit the tail's body atoms manually so the conditional
    # terminator can be rewritten into the break guard.
    visiting_inner: set[int] = set()
    head_body = _emit_block(
        cfg, loop.header,
        exit_id=loop.tail,
        visiting=visiting_inner,
        loops=loops, loop_blocks=loop_blocks,
        active_loop_tail=loop.tail,
    )
    body_stmts.extend(head_body.stmts)

    # Emit the tail block's body atoms (loads/stores/etc.) and then
    # the synthesised exit guard.
    tail_block = cfg.blocks[loop.tail]
    for item in tail_block.body:
        if isinstance(item, IR1Call):
            body_stmts.append(CallStmt(target=item.target, src=item.src))
        else:
            body_stmts.append(RawStmt(item=item))

    term = tail_block.terminator
    inverted_cond = _invert_loop_exit(loop.cond, loop.cond_is_compare)
    break_block = Block.of([BreakStmt(src=term.src)])
    if loop.cond_is_compare:
        body_stmts.append(IfStmt(
            cond=inverted_cond,
            then_block=break_block,
            else_block=None,
            src=term.src,
        ))
    else:
        body_stmts.append(RawIfStmt(
            cond=inverted_cond,
            then_block=break_block,
            else_block=None,
            src=term.src,
        ))

    return Block.of(body_stmts)
