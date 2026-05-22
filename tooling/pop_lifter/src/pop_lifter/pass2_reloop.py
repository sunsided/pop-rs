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

Out of scope:

* Loops. The relooper bails out (returns the routine unchanged in IR1
  form via a single `RawStmt` wrapper) on any backward local jump.
  CHECKFLOOR has none.
* Post-dominator analysis. The current algorithm may emit a small
  amount of code duplication when a block is reached from both the
  taken edge of an `if` and the fall-through path of the next block.
  CHECKFLOOR's `:ong` (`tail_call onground`) is the only such case
  in the pilot — fine for now; a future pass will collapse it.
* `match` / `switch` recognition from chained `if a == K` —
  pass 3's job.
"""

from __future__ import annotations

from .cfg import CFG, BasicBlock, build_cfg
from .ir1 import (
    Branch,
    Goto,
    If,
    Label,
    ModuleIR1,
    Return,
    Routine,
    SourceRef,
)
from .ir1 import (
    Call as IR1Call,
)
from .ir3 import (
    Block,
    CallStmt,
    GotoStmt,
    IfStmt,
    LabelStmt,
    ModuleIR3,
    RawIfStmt,
    RawStmt,
    ReturnStmt,
    RoutineIR3,
    Stmt,
    TailCallStmt,
)


def _has_backward_local_jump(cfg: CFG) -> bool:
    """The relooper's only hard bail-out: if any local edge points
    backwards in block order it's (probably) a loop, and the simple
    recursive walk below would either infinite-loop or emit something
    wrong. CHECKFLOOR has none; combat/physics routines will need a
    loop-aware reloop pass in a later slice."""
    for src, succs in cfg.succ.items():
        for dst in succs:
            if dst <= src:
                return True
    return False


def reloop_routine(routine: Routine) -> RoutineIR3:
    """Structure `routine` into an IR3 routine. If the CFG can't be
    structured by this slice's algorithm (loop, malformed control
    flow), fall back to wrapping the whole IR2 body as a sequence of
    `RawStmt`s — correctness preserved, structure not improved.
    """
    cfg = build_cfg(routine)
    if _has_backward_local_jump(cfg):
        return _wrap_unstructured(routine)

    # `visiting` guards against accidental infinite recursion if the
    # CFG turns out to have a cycle we missed (shouldn't happen given
    # the bail-out, but cheap insurance).
    visiting: set[int] = set()
    body = _emit_block(cfg, cfg.entry_id, exit_id=None, visiting=visiting)
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


def _wrap_unstructured(routine: Routine) -> RoutineIR3:
    """Fallback: emit every item as either a RawStmt (atoms) or the
    closest IR3 control-flow shape (Label/Goto/Return/...). The
    routine isn't *structured* but is at least serialisable in IR3."""
    stmts: list[Stmt] = []
    for item in routine.body:
        if isinstance(item, Label):
            stmts.append(LabelStmt(name=item.name, src=item.src))
        elif isinstance(item, Return):
            stmts.append(ReturnStmt(src=item.src))
        elif isinstance(item, Goto):
            if item.kind == "tail_call":
                stmts.append(TailCallStmt(target=item.target, src=item.src))
            else:
                stmts.append(GotoStmt(target=item.target, src=item.src))
        elif isinstance(item, If):
            stmts.append(IfStmt(
                cond=item.cond,
                then_block=Block.of([GotoStmt(target=item.target, src=item.src)]),
                else_block=None,
                src=item.src,
            ))
        elif isinstance(item, Branch):
            stmts.append(RawIfStmt(
                cond=item.cond,
                then_block=Block.of([GotoStmt(target=item.target, src=item.src)]),
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


def _emit_block(
    cfg: CFG,
    bid: int | None,
    exit_id: int | None,
    visiting: set[int],
) -> Block:
    """Recursively emit IR3 stmts for block `bid` up to (but not
    including) `exit_id`. `exit_id=None` means "emit until the
    routine exits".

    Code duplication note: when a block is reached from two paths
    (e.g. CHECKFLOOR's `:ong` from both `B2.taken` and `B3.fall-
    through`), this algorithm emits its body once per visit. The
    duplication is structurally innocuous (each emission terminates
    via tail-call or return) but pass 3 may merge them once it has
    post-dominator data.
    """
    stmts: list[Stmt] = []
    while bid is not None and bid != exit_id:
        if bid in visiting:
            # Cycle — emit a goto as the escape hatch and stop.
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
            # local goto — continue the linear walk at the target.
            visiting.discard(bid)
            bid = cfg.label_to_block.get(t.target)
            if bid is None:
                # Cross-routine local goto (shouldn't happen given how
                # pass 1 emits, but be safe).
                stmts.append(GotoStmt(target=t.target, src=t.src))
                break
            continue

        if isinstance(t, (If, Branch)):
            taken_label = t.target
            taken_id = cfg.label_to_block.get(taken_label)
            ft_id = bid + 1 if bid + 1 < len(cfg.blocks) else None

            then_stmts: list[Stmt]
            if taken_id is None:
                # Cross-module: the taken edge is a conditional tail
                # call. The relooper emits a single TailCallStmt in
                # the then-branch; nothing inside the routine follows.
                then_stmts = [TailCallStmt(target=taken_label, src=t.src)]
            else:
                # Local: structure the taken subgraph until it
                # rejoins the fall-through (`exit=ft_id`). For early-
                # exit shapes this typically just emits a Return or
                # TailCall.
                then_stmts = list(_emit_block(
                    cfg, taken_id, exit_id=ft_id, visiting=visiting,
                ).stmts)

            if isinstance(t, If):
                stmts.append(IfStmt(
                    cond=t.cond,
                    then_block=Block.of(then_stmts),
                    else_block=None,
                    src=t.src,
                ))
            else:  # Branch
                stmts.append(RawIfStmt(
                    cond=t.cond,
                    then_block=Block.of(then_stmts),
                    else_block=None,
                    src=t.src,
                ))

            visiting.discard(bid)
            bid = ft_id
            continue

        # Unknown terminator — shouldn't happen.
        raise AssertionError(f"unexpected terminator: {t!r}")

    return Block.of(stmts)
