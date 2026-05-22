"""Pass 2: structured-IR (IR2) lowering.

Pass 2 takes the opcode-for-opcode IR1 produced by `pass1_lift` and
folds the most common `Cmp + Branch` idioms into a single structured
`If` node. The result still lives in the same `Routine` / `ModuleIR1`
types — the only IR2-specific node is `ir1.If`. That keeps the
interpreter, the dump format, and the cross-module call resolution
all on a single code path; pass-3 / pass-4 will fork the types when
they need to.

What this slice covers:

* Cmp + Branch fusion for the eq / ne / cc / cs conditions, which are
  the four "value comparison" forms used in CHECKFLOOR and the
  surrounding combat/physics code.
* `lda <abs>; beq/bne/bpl/bmi` ⇒ `if a == 0 / a != 0 / a >= 0 / a < 0
  goto target`. These don't have an explicit `cmp` because `lda` itself
  defines Z and N — but the structured form makes the intent obvious.

Out of scope for this slice (called out by the plan, still pending):

* The relooper — reconstructing `if`/`while`/`for` from arbitrary
  CFGs. We still emit unstructured `goto`s; pass 2 just makes their
  conditions self-describing.
* 16-bit add/sub pattern folding.
* Parallel-array fusion (`mob[x].{x,y,scrn,...}`).
* Flag-liveness analysis that elides dead Z/N updates.

Pass 2 is intentionally conservative: any sequence it doesn't
recognise passes through unchanged, so the structurer is always
behaviour-preserving. The IR1 interpreter executes either form
identically.
"""

from __future__ import annotations

from dataclasses import replace

from .ir1 import (
    Branch,
    CmpAbs,
    CmpImm,
    Compare,
    If,
    Imm,
    Item,
    LoadAbs,
    LoadImm,
    LoadIndexed,
    ModuleIR1,
    Routine,
)


# Mapping from a 6502 branch suffix to the structured operator it
# implies *when the most recent flag-setter was a `cmp`*. CMP sets:
#   * Z = (reg == rhs)
#   * C = (reg >= rhs)     (no-borrow, unsigned)
# so `beq` after `cmp` means `==`, `bcs` means `>=`, etc.
_CMP_FUSE_OPS: dict[str, str] = {
    "eq": "==",
    "ne": "!=",
    "cs": ">=",
    "cc": "<",
}

# After a `lda <abs>` or `ldx`/`ldy` etc., the next branch reads Z/N
# of the loaded value directly. `pl` / `mi` become sign tests; `eq` /
# `ne` become zero tests.
_LOAD_FUSE_OPS: dict[str, tuple[str, bool]] = {
    # cond -> (op, needs_rhs_imm0)
    "eq": ("==", True),    # rhs = #0 implicit
    "ne": ("!=", True),
    "pl": (">=0", False),
    "mi": ("<0", False),
}


_ZERO_IMM = Imm(value=0, text="#$00")


def _fuse_pair(prev: Item, branch: Branch) -> If | None:
    """Try to fuse `prev` (a flag-setter) with `branch` (a conditional
    transfer). Returns the `If` replacement or `None` if the pair
    doesn't match a known idiom."""
    cond_op = _CMP_FUSE_OPS.get(branch.cond)
    if isinstance(prev, CmpImm) and cond_op is not None:
        return If(
            cond=Compare(reg=prev.reg, op=cond_op, rhs=prev.imm),
            target=branch.target,
            src=branch.src,
        )
    if isinstance(prev, CmpAbs) and cond_op is not None:
        return If(
            cond=Compare(reg=prev.reg, op=cond_op, rhs=prev.source),
            target=branch.target,
            src=branch.src,
        )
    if isinstance(prev, (LoadAbs, LoadIndexed, LoadImm)):
        load_op = _LOAD_FUSE_OPS.get(branch.cond)
        if load_op is None:
            return None
        op, needs_zero_rhs = load_op
        rhs = _ZERO_IMM if needs_zero_rhs else None
        return If(
            cond=Compare(reg=prev.reg, op=op, rhs=rhs),
            target=branch.target,
            src=branch.src,
        )
    return None


def _defines_flags(item: Item) -> bool:
    """Heuristic: does this IR1 item define Z/N/C in a way the next
    branch might read? Conservative — only the obvious cases."""
    return isinstance(item, (CmpImm, CmpAbs, LoadImm, LoadAbs, LoadIndexed))


def structure_routine(routine: Routine) -> Routine:
    """Return a copy of `routine` with `Cmp + Branch` pairs fused into
    `If` nodes. The original routine is not mutated.

    The walk preserves item order and keeps the load that fed the
    comparison in place — pass 2 only collapses the `Cmp`/`Branch`
    suffix. That way `a = *foo` remains visible to subsequent
    consumers (parallel-array detection, flag-liveness elision)
    rather than being folded into the Compare prematurely.
    """
    new_body: list[Item] = []
    # The most recent item that *defined* Z/N/C and which the next
    # branch could legitimately consume. Reset every time we hit
    # something that doesn't define flags, so we never fuse across a
    # store or a tail-call.
    pending: Item | None = None

    for item in routine.body:
        if isinstance(item, Branch) and pending is not None:
            fused = _fuse_pair(pending, item)
            if fused is not None:
                # The flag-setter stays in the body (pass 2 doesn't yet
                # know whether it's still needed downstream). The Branch
                # is replaced by the structured If.
                new_body.append(fused)
                pending = None
                continue
        new_body.append(item)
        pending = item if _defines_flags(item) else None

    return replace(routine, body=new_body)


def structure_module(module: ModuleIR1) -> ModuleIR1:
    """Apply `structure_routine` to every routine in `module`. Returns
    a new `ModuleIR1`; the input is not mutated."""
    return ModuleIR1(
        name=module.name,
        file=module.file,
        routines=[structure_routine(r) for r in module.routines],
    )


def fusion_stats(module: ModuleIR1) -> tuple[int, int]:
    """Return (fused_if_count, unfused_branch_count) across all
    routines in the module. Used by the CLI to summarise how much of
    a routine's control flow pass 2 was able to lift to structured
    form."""
    fused = 0
    unfused = 0
    for r in module.routines:
        for item in r.body:
            if isinstance(item, If):
                fused += 1
            elif isinstance(item, Branch):
                unfused += 1
    return fused, unfused
