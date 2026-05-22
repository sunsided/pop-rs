"""Per-routine basic-block construction.

A `BasicBlock` is a maximal sequence of straight-line IR1/IR2 items
terminated by exactly one control-flow item (`Return`, `Goto`,
`Branch`, `If`). `build_cfg` splits a `Routine.body` into such
blocks and records the local successor edges for each.

What "local" means: a successor edge that lands inside the same
routine. Returns, `Goto kind=tail_call`, and the taken edge of an
`If`/`Branch` whose target is cross-module all leave the routine
and don't appear in `succ`. The relooper inspects each block's
terminator directly for those cases.

Dominator analysis isn't computed yet — the current relooper handles
CHECKFLOOR's shape (reducible, no loops, mostly early-exit forks)
with a simple recursive walk over the block array. A future slice
that needs post-dominators or loop discovery will add the iterative
Cooper/Harvey/Kennedy implementation alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ir1 import (
    Branch,
    Goto,
    If,
    Item,
    Label,
    Return,
    Routine,
    SourceRef,
)


@dataclass
class BasicBlock:
    id: int
    label: str | None
    body: list[Item]
    terminator: Item            # Return | Goto | Branch | If


@dataclass
class CFG:
    routine: Routine
    blocks: list[BasicBlock]
    label_to_block: dict[str, int]
    succ: dict[int, list[int]]
    pred: dict[int, list[int]]

    @property
    def entry_id(self) -> int:
        return 0


def _is_terminator(item: Item) -> bool:
    return isinstance(item, (Return, Goto, Branch, If))


def _fallback_src(routine: Routine) -> SourceRef:
    for it in routine.body:
        s = getattr(it, "src", None)
        if s is not None:
            return s
    return SourceRef(file="<synthetic>", line=0, raw="")


def build_cfg(routine: Routine) -> CFG:
    """Slice `routine.body` into basic blocks.

    Block boundaries:

    * The first item starts block 0.
    * Every `Label` starts a new block. If the previous block didn't
      already end on a terminator, we synthesise a `Goto kind=local`
      to the new label — pass-1 bodies almost always have an explicit
      jump before a label, but the synthesis keeps the CFG well-formed
      even if a future input doesn't.
    * Every terminator (`Return`/`Goto`/`Branch`/`If`) closes the
      current block.

    A routine whose body is empty gets a single block containing a
    synthetic `Return` so downstream passes don't have to special-case
    empty inputs.
    """
    blocks: list[BasicBlock] = []
    label_to_block: dict[str, int] = {}

    cur_label: str | None = None
    cur_body: list[Item] = []

    def close(terminator: Item) -> None:
        nonlocal cur_label, cur_body
        bid = len(blocks)
        b = BasicBlock(
            id=bid,
            label=cur_label,
            body=cur_body,
            terminator=terminator,
        )
        blocks.append(b)
        if cur_label is not None:
            label_to_block[cur_label] = bid
        cur_label = None
        cur_body = []

    for item in routine.body:
        if isinstance(item, Label):
            if cur_body or cur_label is not None:
                # Implicit fall-through into the labelled block.
                close(Goto(target=item.name, kind="local", src=item.src))
            cur_label = item.name
            continue
        if _is_terminator(item):
            close(item)
            continue
        cur_body.append(item)

    if cur_body or cur_label is not None:
        # Routine body ended without an explicit terminator — synthesise
        # a Return so the CFG is well-formed. Real IR2 from pass 1 always
        # ends in an rts / jmp; this is a safety net.
        close(Return(src=_fallback_src(routine)))

    if not blocks:
        close(Return(src=_fallback_src(routine)))

    # Local successors.
    succ: dict[int, list[int]] = {b.id: [] for b in blocks}
    pred: dict[int, list[int]] = {b.id: [] for b in blocks}
    for b in blocks:
        for s in _local_successors(b, label_to_block, len(blocks)):
            succ[b.id].append(s)
            pred[s].append(b.id)

    return CFG(
        routine=routine,
        blocks=blocks,
        label_to_block=label_to_block,
        succ=succ,
        pred=pred,
    )


def _local_successors(
    b: BasicBlock,
    label_to_block: dict[str, int],
    total: int,
) -> list[int]:
    t = b.terminator
    if isinstance(t, Return):
        return []
    if isinstance(t, Goto):
        if t.kind == "tail_call":
            return []
        tid = label_to_block.get(t.target)
        return [tid] if tid is not None else []
    if isinstance(t, (Branch, If)):
        out: list[int] = []
        tid = label_to_block.get(t.target)
        if tid is not None:
            out.append(tid)
        # Fall-through to the positionally-next block.
        if b.id + 1 < total:
            out.append(b.id + 1)
        return out
    raise ValueError(f"unexpected terminator: {t!r}")


# ---------------------------------------------------------------- dominators


def reverse_postorder(cfg: CFG) -> list[int]:
    """Reverse post-order traversal of `cfg` from the entry block.
    Used as the iteration order for the dominator solver and as the
    primary scan order for the loop-relooper."""
    order: list[int] = []
    visited: set[int] = set()

    def dfs(b: int) -> None:
        visited.add(b)
        for s in cfg.succ[b]:
            if s not in visited:
                dfs(s)
        order.append(b)

    dfs(cfg.entry_id)
    return list(reversed(order))


def compute_idoms(cfg: CFG) -> dict[int, int]:
    """Cooper / Harvey / Kennedy iterative immediate-dominator
    computation. Returns a map `block_id -> idom_id`; the entry
    block's idom is itself (sentinel). Unreachable blocks aren't in
    the result.

    Reference: "A Simple, Fast Dominance Algorithm",
    Cooper, Harvey & Kennedy (Rice CS-TR-06-33870).
    """
    rpo = reverse_postorder(cfg)
    rpo_idx = {b: i for i, b in enumerate(rpo)}

    idom: dict[int, int] = {cfg.entry_id: cfg.entry_id}

    def intersect(b1: int, b2: int) -> int:
        finger1, finger2 = b1, b2
        while finger1 != finger2:
            while rpo_idx[finger1] > rpo_idx[finger2]:
                finger1 = idom[finger1]
            while rpo_idx[finger2] > rpo_idx[finger1]:
                finger2 = idom[finger2]
        return finger1

    changed = True
    while changed:
        changed = False
        for b in rpo:
            if b == cfg.entry_id:
                continue
            processed_preds = [p for p in cfg.pred[b] if p in idom]
            if not processed_preds:
                continue
            new_idom = processed_preds[0]
            for p in processed_preds[1:]:
                new_idom = intersect(p, new_idom)
            if idom.get(b) != new_idom:
                idom[b] = new_idom
                changed = True
    return idom


def dominates(idom: dict[int, int], a: int, b: int) -> bool:
    """True if `a` dominates `b` (every path from entry to `b` passes
    through `a`). `a` dominates itself by convention."""
    if a == b:
        return True
    cur = b
    while idom.get(cur, cur) != cur:
        cur = idom[cur]
        if cur == a:
            return True
    return False


# ---------------------------------------------------------------- loops


def find_back_edges(cfg: CFG) -> list[tuple[int, int]]:
    """Return all back-edges `(source, target)` where `target`
    dominates `source`. These are the canonical loop entries: each
    back-edge induces a natural loop with `target` as the header.

    Only catches *reducible* loops. For arbitrary-cycle detection
    (including irreducible flow that has no dominator back-edge),
    use `find_dfs_back_edges`.
    """
    idom = compute_idoms(cfg)
    edges: list[tuple[int, int]] = []
    for s, succs in cfg.succ.items():
        if s not in idom:
            continue
        for d in succs:
            if d in idom and dominates(idom, d, s):
                edges.append((s, d))
    return edges


def find_dfs_back_edges(cfg: CFG) -> list[tuple[int, int]]:
    """DFS classification of back-edges: any edge `(s, d)` where `d`
    is on the DFS stack when `s` is being explored. Catches *every*
    cycle in the CFG including irreducible ones (the kind without
    a dominator back-edge — Tarjan's "cross-into-SCC" shape).

    Pass 2's relooper uses this for the fallback gate: if the CFG
    has any cycle that doesn't fully belong to a recognised simple
    do-while, the routine takes the unstructured fallback. Sticking
    with `find_back_edges` for that gate would silently miss
    irreducible loops, which would then trip the walker's `visiting`
    escape hatch and emit `GotoStmt`s to synthesized labels that no
    interpreter can resolve.
    """
    visited: set[int] = set()
    on_stack: set[int] = set()
    edges: list[tuple[int, int]] = []

    def dfs(b: int) -> None:
        visited.add(b)
        on_stack.add(b)
        for s in cfg.succ.get(b, []):
            if s in on_stack:
                edges.append((b, s))
            elif s not in visited:
                dfs(s)
        on_stack.discard(b)

    if cfg.blocks:
        dfs(cfg.entry_id)
    return edges


def natural_loop_body(cfg: CFG, source: int, header: int) -> set[int]:
    """Body of the natural loop induced by back-edge `(source, header)`:
    the header plus every block that can reach `source` without going
    through the header. Computed as a reverse BFS from `source` along
    predecessor edges, stopping at `header`. The header itself is
    always included.

    Self-loop edge case: when `source == header` the loop is a single
    block (a back-edge from the header to itself). The body is just
    `{header}`; we don't seed the worklist, which would otherwise
    walk past the header into the preheader.
    """
    body: set[int] = {header}
    if source == header:
        return body
    body.add(source)
    worklist: list[int] = [source]
    while worklist:
        b = worklist.pop()
        for p in cfg.pred[b]:
            if p not in body:
                body.add(p)
                worklist.append(p)
    return body
