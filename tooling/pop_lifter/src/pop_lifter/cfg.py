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
