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
* Post-dominator merge. When a conditional's two arms reconverge at a
  block *after* the fall-through (its immediate post-dominator `M`),
  the walker emits each arm only up to `M` and the shared continuation
  once after the `if/else`. Without this the taken arm re-inlines the
  whole tail, which compounds exponentially across nested conditionals
  (one routine ballooned ~560x before this landed). Arms that instead
  diverge to separate exits with only *partial* joins (a block reached
  from some but not all paths) still duplicate — collapsing those needs
  labeled-block / multiple-block structuring, a later pass.
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
    find_dfs_back_edges,
    reverse_postorder,
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
    DoWhileStmt,
    ForStmt,
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


def _child_blocks(s: Stmt):
    """Sub-`Block`s of a statement, for recursive traversal."""
    blocks = []
    for attr in ("then_block", "else_block", "body"):
        b = getattr(s, attr, None)
        if b is not None:
            blocks.append(b)
    if isinstance(s, MatchStmt):
        blocks.extend(arm.body for arm in s.arms)
    return blocks


def _contains_break(stmts, label: str) -> bool:
    for s in stmts:
        if isinstance(s, BreakStmt) and s.label == label:
            return True
        if any(_contains_break(b.stmts, label) for b in _child_blocks(s)):
            return True
    return False


def _strip_tail_break(stmts: list, label: str) -> list:
    """Drop a `break <label>` in tail position — where falling through
    reaches exactly where the break would jump (just after the labeled
    block), so the break is redundant. Recurses through the tail of
    if/else arms and nested labeled blocks."""
    if not stmts:
        return stmts
    last = stmts[-1]
    if isinstance(last, BreakStmt) and last.label == label:
        return stmts[:-1]
    if isinstance(last, (IfStmt, RawIfStmt)):
        then_b = Block.of(_strip_tail_break(list(last.then_block.stmts), label))
        else_b = (
            Block.of(_strip_tail_break(list(last.else_block.stmts), label))
            if last.else_block is not None else None
        )
        return stmts[:-1] + [dataclass_replace(last, then_block=then_b, else_block=else_b)]
    if isinstance(last, LabeledBlock):
        inner = _strip_tail_break(list(last.body.stmts), label)
        return stmts[:-1] + [dataclass_replace(last, body=Block.of(inner))]
    return stmts


def _invert_if_cond(s):
    if isinstance(s, IfStmt):
        return dataclass_replace(s.cond, op=_INVERT_COMPARE_OP[s.cond.op])
    return _INVERT_BRANCH_COND[s.cond]


def _simplify_stmts(stmts) -> list:
    """Tidy structured output (bottom-up): unwrap labeled blocks whose
    breaks have all been elided, drop redundant tail breaks, and turn
    `if c {} else {X}` into `if !c {X}`."""
    out: list = []
    for s in stmts:
        out.extend(_simplify_stmt(s))
    return out


def _simplify_stmt(s) -> list:
    if isinstance(s, LabeledBlock):
        # Strip redundant tail breaks first, then simplify — so the
        # `if c { break } else { X }` shapes the strip empties out get
        # inverted to `if !c { X }` by the empty-then rule below.
        body = _simplify_stmts(_strip_tail_break(list(s.body.stmts), s.label))
        if not _contains_break(body, s.label):
            return body  # no break targets this label any more — unwrap it
        return [dataclass_replace(s, body=Block.of(body))]
    if isinstance(s, (IfStmt, RawIfStmt)):
        then_b = Block.of(_simplify_stmts(s.then_block.stmts))
        else_b = Block.of(_simplify_stmts(s.else_block.stmts)) if s.else_block is not None else None
        if not then_b.stmts and else_b is not None and else_b.stmts:
            return [dataclass_replace(s, cond=_invert_if_cond(s), then_block=else_b, else_block=None)]
        if else_b is not None and not else_b.stmts:
            else_b = None
        return [dataclass_replace(s, then_block=then_b, else_block=else_b)]
    if isinstance(s, (LoopStmt, DoWhileStmt, ForStmt, RepeatStmt)):
        return [dataclass_replace(s, body=Block.of(_simplify_stmts(s.body.stmts)))]
    if isinstance(s, MatchStmt):
        arms = tuple(
            dataclass_replace(a, body=Block.of(_simplify_stmts(a.body.stmts)))
            for a in s.arms
        )
        return [dataclass_replace(s, arms=arms)]
    return [s]


def _structure_acyclic(cfg: CFG) -> Block:
    """Structure a *reducible, acyclic* routine with no code duplication.

    Walks the dominator tree (`doTree`/`nodeWithin`, after Ramsey's
    "Beyond Relooper"). Every block with more than one predecessor is a
    *merge node*; it is emitted exactly once at its immediate dominator,
    wrapped so each predecessor reaches it by a structured `break
    'b<id>` out of a `LabeledBlock`. Non-merge blocks are inlined at
    their sole predecessor, so straight-line forks stay plain
    `if`/`else`. Merge children are nested highest-reverse-post-order
    outermost, so a forward `break` to any of them is always in scope.

    Only valid for acyclic CFGs (no back-edges): every merge node's
    immediate dominator then dominates all its predecessors, so emitting
    it at that point is sound. Loop-bearing routines keep the existing
    `_emit_block` walker."""
    from collections import defaultdict

    idom = compute_idoms(cfg)
    rank = {b: i for i, b in enumerate(reverse_postorder(cfg))}

    def _is_trivial_exit(b) -> bool:
        # A bare `return` / `tail_call` (no body) is cheaper to duplicate
        # at each predecessor than to hoist behind a label — and avoids
        # wrapping a whole routine just because it ends in one shared exit.
        if b.body:
            return False
        t = b.terminator
        return isinstance(t, Return) or (isinstance(t, Goto) and t.kind == "tail_call")

    merge = {
        b.id for b in cfg.blocks
        if len(cfg.pred[b.id]) > 1 and not _is_trivial_exit(b)
    }
    dom_children: dict[int, list[int]] = defaultdict(list)
    for b in cfg.blocks:
        if b.id != cfg.entry_id and b.id in idom:
            dom_children[idom[b.id]].append(b.id)

    def label_for(m: int) -> str:
        return f"'b{m}"

    def do_tree(x: int, context: list[int]) -> list[Stmt]:
        merges = sorted((c for c in dom_children[x] if c in merge), key=lambda c: rank[c])
        return node_within(x, merges, context)

    def node_within(x: int, merges: list[int], context: list[int]) -> list[Stmt]:
        if not merges:
            return emit_node(x, context)
        y = merges[-1]  # highest RPO → outermost block
        inner = node_within(x, merges[:-1], context + [y])
        block = LabeledBlock(label=label_for(y), body=Block.of(inner), src=cfg.blocks[y].terminator.src)
        return [block, *do_tree(y, context)]

    def succ(target_id: int, context: list[int], src) -> list[Stmt]:
        # Transfer to a known local block: break to it if it's a shared
        # merge node (emitted once elsewhere), else inline it here.
        if target_id in merge:
            return [BreakStmt(src=src, label=label_for(target_id))]
        return do_tree(target_id, context)

    def emit_node(x: int, context: list[int]) -> list[Stmt]:
        block = cfg.blocks[x]
        stmts: list[Stmt] = []
        for item in block.body:
            if isinstance(item, IR1Call):
                stmts.append(CallStmt(target=item.target, src=item.src))
            else:
                stmts.append(RawStmt(item=item))

        t = block.terminator
        if isinstance(t, Return):
            stmts.append(ReturnStmt(src=t.src))
            return stmts
        if isinstance(t, Goto):
            if t.kind == "tail_call":
                stmts.append(TailCallStmt(target=t.target, src=t.src))
                return stmts
            target_id = cfg.label_to_block.get(t.target)
            if target_id is None:
                stmts.append(GotoStmt(target=t.target, src=t.src))
                return stmts
            stmts.extend(succ(target_id, context, t.src))
            return stmts
        if isinstance(t, (If, Branch)):
            taken_id = cfg.label_to_block.get(t.target)
            ft_id = x + 1 if x + 1 < len(cfg.blocks) else None
            if taken_id is None:
                then_stmts: list[Stmt] = [TailCallStmt(target=t.target, src=t.src)]
            else:
                then_stmts = succ(taken_id, context, t.src)
            ft_stmts = succ(ft_id, context, t.src) if ft_id is not None else []
            node_cls = IfStmt if isinstance(t, If) else RawIfStmt
            if then_stmts and isinstance(then_stmts[-1], (ReturnStmt, TailCallStmt)):
                # The taken arm leaves the routine, so the fall-through is
                # not a join with it: keep it flat as a sibling (guard-
                # clause / dispatch-chain style) rather than nesting it in
                # an `else`. This preserves the chained-`if` shape the
                # `match` recogniser and readers expect.
                stmts.append(node_cls(
                    cond=t.cond, then_block=Block.of(then_stmts), else_block=None, src=t.src,
                ))
                stmts.extend(ft_stmts)
            else:
                # The taken arm rejoins later (a `break` to a merge): the
                # fall-through is the other arm, so emit a real `else`.
                stmts.append(node_cls(
                    cond=t.cond,
                    then_block=Block.of(then_stmts),
                    else_block=Block.of(ft_stmts) if ft_stmts else None,
                    src=t.src,
                ))
            return stmts
        raise AssertionError(f"unexpected terminator: {t!r}")

    return Block.of(_simplify_stmts(do_tree(cfg.entry_id, [])))


def reloop_routine(routine: Routine) -> RoutineIR3:
    """Structure `routine` into an IR3 routine. Simple do-while loops
    are recognised and emitted as `LoopStmt`; anything else with a
    backward local jump falls back to `_wrap_unstructured`.
    """
    cfg = build_cfg(routine)

    # Every cycle in the CFG must belong to a recognised simple loop.
    # We use `find_dfs_back_edges` (not `find_back_edges`) here so the
    # gate catches *irreducible* loops too — those have no dominator
    # back-edge but still represent cycles the linear walker can't
    # safely traverse. Without this stricter check the walker's
    # `visiting` escape hatch would fire mid-cycle and emit a
    # `GotoStmt` referencing a synthesised `BB{n}` label that nothing
    # downstream can resolve.
    cycle_edges = find_dfs_back_edges(cfg) if cfg.blocks else []
    loops, loop_blocks = _detect_simple_loops(cfg)
    if cycle_edges and not all(
        src in loop_blocks and dst in loop_blocks for src, dst in cycle_edges
    ):
        return _wrap_unstructured(routine)

    # Acyclic routines (no back-edges) are reducible: the dominator-tree
    # structurer emits every merge node once via labeled blocks, with no
    # code duplication. Loop-bearing routines keep the linear walker.
    if not cycle_edges:
        return RoutineIR3(
            name=routine.name,
            entry_aliases=list(routine.entry_aliases),
            body=_structure_acyclic(cfg),
        )

    ipostdoms = _compute_ipostdoms(cfg)
    visiting: set[int] = set()
    body = _emit_block(
        cfg, cfg.entry_id,
        exit_id=None,
        visiting=visiting,
        loops=loops,
        loop_blocks=loop_blocks,
        active_loop_tail=None,
        ipostdoms=ipostdoms,
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


_EXIT = -1  # virtual exit node for post-dominator analysis


def _block_leaves_routine(cfg: CFG, b) -> bool:
    """True if any control-flow edge out of block `b` leaves the routine
    — a `return`, a tail-call, a goto/branch to a label outside the
    routine, or a fall-through past the last block. Such blocks connect
    to the virtual exit in the reverse CFG."""
    t = b.terminator
    if isinstance(t, Return):
        return True
    if isinstance(t, Goto):
        if t.kind == "tail_call":
            return True
        return cfg.label_to_block.get(t.target) is None
    if isinstance(t, (If, Branch)):
        if cfg.label_to_block.get(t.target) is None:
            return True
        return b.id + 1 >= len(cfg.blocks)
    return True


def _compute_ipostdoms(cfg: CFG) -> dict[int, int]:
    """Immediate post-dominators (`block -> ipostdom`), computed as
    dominators on the reverse CFG from a virtual exit (`_EXIT`) that
    every routine-leaving block connects to.

    The relooper uses this to find where a conditional's two arms
    reconverge: emitting the shared continuation once after the
    `if/else` instead of inlining it into both arms is what keeps a
    branch-dense routine from blowing up by code duplication. A result
    of `_EXIT` means the arms diverge to separate exits (no shared
    continuation); blocks that can't reach an exit are omitted."""
    if not cfg.blocks:
        return {}
    exit_blocks = [b.id for b in cfg.blocks if _block_leaves_routine(cfg, b)]
    # Reverse CFG: forward edge u->v becomes v->u; `_EXIT` is the entry
    # and points at every exit block.
    rsucc: dict[int, list[int]] = {_EXIT: list(exit_blocks)}
    for b in cfg.blocks:
        rsucc[b.id] = list(cfg.pred[b.id])
    rpred: dict[int, list[int]] = {b.id: list(cfg.succ[b.id]) for b in cfg.blocks}
    for bid in exit_blocks:
        rpred[bid] = rpred[bid] + [_EXIT]

    # Reverse post-order over the reverse CFG (iterative, to avoid
    # recursion limits on large routines).
    order: list[int] = []
    seen: set[int] = {_EXIT}
    stack: list[tuple[int, object]] = [(_EXIT, iter(rsucc.get(_EXIT, [])))]
    while stack:
        node, it = stack[-1]
        for nxt in it:  # type: ignore[assignment]
            if nxt not in seen:
                seen.add(nxt)
                stack.append((nxt, iter(rsucc.get(nxt, []))))
                break
        else:
            order.append(node)
            stack.pop()
    rpo = list(reversed(order))
    rank = {n: i for i, n in enumerate(rpo)}

    idom: dict[int, int] = {_EXIT: _EXIT}

    def intersect(a: int, b: int) -> int:
        while a != b:
            while rank[a] > rank[b]:
                a = idom[a]
            while rank[b] > rank[a]:
                b = idom[b]
        return a

    changed = True
    while changed:
        changed = False
        for n in rpo:
            if n == _EXIT:
                continue
            preds = [p for p in rpred.get(n, []) if p in idom]
            if not preds:
                continue
            new = preds[0]
            for p in preds[1:]:
                new = intersect(p, new)
            if idom.get(n) != new:
                idom[n] = new
                changed = True

    return {b: d for b, d in idom.items() if b != _EXIT}


def _emit_block(
    cfg: CFG,
    bid: int | None,
    exit_id: int | None,
    visiting: set[int],
    *,
    loops: dict[int, SimpleLoop],
    loop_blocks: set[int],
    active_loop_tail: int | None,
    ipostdoms: dict[int, int],
) -> Block:
    """Recursively emit IR3 stmts for block `bid` up to (but not
    including) `exit_id`. `exit_id=None` means "emit until the
    routine exits".

    Used for *loop-bearing* routines; fully-acyclic ones go through
    `_structure_acyclic`, which structures merges with labeled blocks
    and no duplication.

    `loops` / `loop_blocks` carry the pre-computed simple-loop
    layout. When the walker hits a loop header, it switches into
    loop-body mode (`active_loop_tail` set) and emits the contained
    blocks as a `LoopStmt`. Inside the loop body, the tail's
    conditional jump becomes the bottom-of-loop break guard; any
    other back-edge to the same header surfaces as `continue`.

    Code duplication: when an `if`'s two arms reconverge at their
    immediate post-dominator `M` (`ipostdoms`), the continuation is
    emitted once after an `if/else` (see the `If`/`Branch` handling).
    A merge reached by *more* paths than that — a partial join — is
    still emitted once per path here; collapsing those needs the
    labeled-block structurer extended to loop bodies (a later pass).
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
                cfg, loop, visiting, loops, loop_blocks, ipostdoms,
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
            is_loop_continue = (
                taken_id is not None
                and active_loop_tail is not None
                and taken_id == _loop_for_tail(loops, active_loop_tail).header
            )

            # Merge optimization: when both arms reconverge at a block
            # `M` *after* the fall-through (the immediate post-dominator),
            # emit each arm only up to `M` and the shared continuation
            # once. Without this, the taken arm re-inlines the whole tail
            # (exit_id=ft_id is never hit), which compounds exponentially
            # across nested conditionals. The `M == ft_id` case is the
            # plain `if cond { taken } <fall-through>` shape handled below.
            merge = ipostdoms.get(bid)
            if (
                taken_id is not None
                and not is_loop_continue
                and ft_id is not None
                and merge is not None
                and merge != _EXIT
                and merge != ft_id
                and merge > bid
            ):
                then_stmts = list(_emit_block(
                    cfg, taken_id, exit_id=merge, visiting=visiting,
                    loops=loops, loop_blocks=loop_blocks,
                    active_loop_tail=active_loop_tail, ipostdoms=ipostdoms,
                ).stmts)
                else_stmts = list(_emit_block(
                    cfg, ft_id, exit_id=merge, visiting=visiting,
                    loops=loops, loop_blocks=loop_blocks,
                    active_loop_tail=active_loop_tail, ipostdoms=ipostdoms,
                ).stmts)
                node_cls = IfStmt if isinstance(t, If) else RawIfStmt
                stmts.append(node_cls(
                    cond=t.cond,
                    then_block=Block.of(then_stmts),
                    else_block=Block.of(else_stmts) if else_stmts else None,
                    src=t.src,
                ))
                visiting.discard(bid)
                bid = merge
                continue

            then_stmts: list[Stmt]
            if taken_id is None:
                then_stmts = [TailCallStmt(target=taken_label, src=t.src)]
            elif is_loop_continue:
                # Conditional back-edge inside a loop body → continue.
                then_stmts = [ContinueStmt(src=t.src)]
            else:
                then_stmts = list(_emit_block(
                    cfg, taken_id, exit_id=ft_id, visiting=visiting,
                    loops=loops, loop_blocks=loop_blocks,
                    active_loop_tail=active_loop_tail, ipostdoms=ipostdoms,
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
    ipostdoms: dict[int, int],
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
        active_loop_tail=loop.tail, ipostdoms=ipostdoms,
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
