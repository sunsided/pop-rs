"""Pass 2: structured-IR (IR2) lowering.

Pass 2 takes the opcode-for-opcode IR1 produced by `pass1_lift` and
performs two transformations:

1. **Cmp + Branch fusion**. The most common 6502 idiom for a
   structured conditional —

      cmp #k
      beq target

   — becomes a single `If(Compare(reg, op, k), target)` node. The
   resulting `If` is self-contained: the interpreter can evaluate it
   without consulting Z/C from prior state. Pass 4's Rust emitter
   gets to write `if a == k { ... }` straight off the IR.

2. **Flag-liveness elision**. After fusion the original `cmp`
   instructions usually have *no* downstream readers — the `If`
   consumed the flags, and nothing else on the linear path between
   that `cmp` and the next flag-setter reads Z/N/C. A backward
   sweep finds those pure flag-defining instructions (`CmpImm`,
   `CmpAbs`, `Clc`, `Sec`) and drops them. The sweep handles forward
   local branches by remembering each label's live-in; backward
   branches (loops) defeat the single-pass analysis, so the routine
   bails out of elision in that case rather than risk an unsound
   delete. Unsupported items are treated as read-everything,
   write-everything.

Both transformations produce IR that lives in the same `Routine` /
`ModuleIR1` types — the only IR2-specific node is `ir1.If`. That
keeps the interpreter, the dump format, and the cross-module call
resolution all on a single code path; pass-3 / pass-4 will fork the
types when they need to.

What's already covered:

* Cmp + Branch fusion for eq / ne / cc / cs (the four CHECKFLOOR
  shapes).
* `lda <abs>; beq/bne/bpl/bmi` ⇒ `if a == 0 / a != 0 / a >= 0 / a < 0
  goto target`. These don't have an explicit `cmp` because `lda`
  itself defines Z and N.
* Flag-liveness elision for the four pure flag-writers.

Out of scope (called out by the plan, still pending):

* The relooper — reconstructing `if`/`while`/`for` from arbitrary
  CFGs. We still emit unstructured `goto`s; pass 2 just makes their
  conditions self-describing.
* 16-bit add/sub pattern folding.
* Parallel-array fusion (`mob[x].{x,y,scrn,...}`).

Pass 2 is intentionally conservative: any sequence it doesn't
recognise passes through unchanged, so the structurer is always
behaviour-preserving. The IR1 interpreter executes either form
identically.
"""

from __future__ import annotations

from dataclasses import replace

from .ir1 import (
    AdcAbs,
    AdcImm,
    Asl,
    Branch,
    Call,
    Clc,
    CmpAbs,
    CmpImm,
    Compare,
    Goto,
    If,
    Imm,
    Item,
    Label,
    LoadAbs,
    LoadImm,
    LoadIndexed,
    ModuleIR1,
    Return,
    Routine,
    Sec,
    Unsupported,
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


# ----------------------------------------------------------------------
# Flag-liveness elision.
#
# After fusion, a `cmp`'s only consumer is usually the `If` that
# replaced its old `Branch` partner — and `If` doesn't read flags. So
# the `cmp` is dead. The same applies to standalone `clc` / `sec`
# whose carry never reaches an `adc` or `bcs/bcc`.
#
# The analysis is a single backward sweep over the routine body. Each
# IR1 node is classified by what flags it reads and what flags it
# defines:
#
#   reader               : `Branch` (cond-specific), `AdcImm`/`AdcAbs`
#                          (reads C). `If` does NOT read flags.
#   writer (whole set)   : `CmpImm`/`CmpAbs` ⇒ {Z,N,C}.
#                          `Clc`/`Sec`       ⇒ {C}.
#                          `LoadImm`/`LoadAbs`/`LoadIndexed` ⇒ {Z,N}.
#                          `AdcImm`/`AdcAbs`/`Asl` ⇒ {Z,N,C}.
#   neutral              : `If`, `Goto`, `Return`, `StoreAbs`,
#                          `StoreIndexed`, `Label`.
#   barrier              : `Call` (clobbers all flags via the callee,
#                          and JSR doesn't consume flag arguments).
#                          `Goto kind=tail_call` and `Return`
#                          (control leaves the routine).
#                          `Unsupported` (assume reads-all,
#                          writes-all — safe).
#
# Elision rule: drop a writer iff none of the flags it defines are
# live at its exit AND the instruction has no other side-effect.
# That's true only for pure writers: `Cmp{Imm,Abs}`, `Clc`, `Sec`.
# `Lda*` is never elided here even when Z/N are dead — it still
# updates the register.
#
# Forward local branches contribute their target's live-IN to the
# branch site's live-OUT. Backward local branches (loops) defeat the
# single-pass analysis: when one is detected, the routine is left
# untouched. CHECKFLOOR has no loops; combat/physics will need the
# fixed-point analysis later.


_UNIVERSE = frozenset({"Z", "N", "C", "V"})


_COND_READS: dict[str, frozenset[str]] = {
    "eq": frozenset({"Z"}),
    "ne": frozenset({"Z"}),
    "cs": frozenset({"C"}),
    "cc": frozenset({"C"}),
    "pl": frozenset({"N"}),
    "mi": frozenset({"N"}),
    "vs": frozenset({"V"}),
    "vc": frozenset({"V"}),
}


def _has_backward_local_jump(body: list[Item]) -> bool:
    """Return True if any local branch / goto / if targets a label at
    or before the source instruction's index. A `True` answer disables
    elision for the routine — we can't soundly fold flag liveness
    without iterating to a fixed point.
    """
    label_idx: dict[str, int] = {
        item.name: i for i, item in enumerate(body) if isinstance(item, Label)
    }
    for i, item in enumerate(body):
        target: str | None = None
        if isinstance(item, Branch):
            target = item.target
        elif isinstance(item, If):
            target = item.target
        elif isinstance(item, Goto) and item.kind == "local":
            target = item.target
        if target is None:
            continue
        ti = label_idx.get(target)
        if ti is not None and ti <= i:
            return True
    return False


def _eliminate_dead_flags(routine: Routine) -> Routine:
    body = routine.body
    if _has_backward_local_jump(body):
        # Loops present — bail conservatively. The fusion done earlier
        # is still safe; we just keep the cmps.
        return routine

    label_idx = {
        item.name: i for i, item in enumerate(body) if isinstance(item, Label)
    }
    # Live-IN at each label, populated as the backward sweep walks
    # past the label.
    label_live_in: dict[str, frozenset[str]] = {}

    live: set[str] = set()
    drop: set[int] = set()

    for i in range(len(body) - 1, -1, -1):
        item = body[i]

        if isinstance(item, Return):
            live = set()
            continue

        if isinstance(item, Goto):
            if item.kind == "tail_call":
                # Control leaves the routine; nothing local depends on
                # flags from here on.
                live = set()
                continue
            # kind == "local"
            ti = label_idx.get(item.target)
            if ti is not None and ti > i:
                live = set(label_live_in.get(item.target, _UNIVERSE))
            else:
                # Forward jump out of routine (cross-module local-ish?
                # shouldn't happen given our backward-jump bail-out, but
                # be safe).
                live = set(_UNIVERSE)
            continue

        if isinstance(item, Branch):
            ti = label_idx.get(item.target)
            if ti is not None and ti > i:
                taken = set(label_live_in.get(item.target, _UNIVERSE))
            else:
                # Cross-routine conditional tail call. The taken edge
                # leaves the routine; its live contribution is empty.
                taken = set()
            # Live OUT = union of fall-through-live and taken-live.
            live = live | taken
            # Branch reads the condition's flags.
            live |= _COND_READS.get(item.cond, _UNIVERSE)
            continue

        if isinstance(item, If):
            ti = label_idx.get(item.target)
            if ti is not None and ti > i:
                taken = set(label_live_in.get(item.target, _UNIVERSE))
            else:
                taken = set()
            live = live | taken
            # `If` does NOT read flags — its Compare is self-contained.
            continue

        if isinstance(item, Call):
            # Callee clobbers all flags; JSR doesn't take flag args by
            # convention. So nothing before the call can keep a flag
            # alive through the call.
            live = set()
            continue

        if isinstance(item, Unsupported):
            # Treat as read-all, write-all — safe.
            live = set(_UNIVERSE)
            continue

        if isinstance(item, (CmpImm, CmpAbs)):
            defines = {"Z", "N", "C"}
            if not (live & defines):
                drop.add(i)
                # The cmp didn't satisfy any demand; live-in == live-out
                # (no flags consumed, no flags produced from this site's
                # perspective). Don't touch `live`.
            else:
                live -= defines
            continue

        if isinstance(item, Clc):
            if "C" not in live:
                drop.add(i)
            else:
                live.discard("C")
            continue

        if isinstance(item, Sec):
            if "C" not in live:
                drop.add(i)
            else:
                live.discard("C")
            continue

        if isinstance(item, (LoadImm, LoadAbs, LoadIndexed)):
            # Lda* writes Z/N; the load itself has a side-effect (the
            # register update) so we can't drop it even when Z/N are
            # dead.
            live -= {"Z", "N"}
            continue

        if isinstance(item, (AdcImm, AdcAbs)):
            # Adc writes Z,N,C and reads C.
            live -= {"Z", "N", "C"}
            live.add("C")
            continue

        if isinstance(item, Asl):
            live -= {"Z", "N", "C"}
            continue

        if isinstance(item, Label):
            # Transparent for fall-through; remember its live-in for
            # forward branches that target it.
            label_live_in[item.name] = frozenset(live)
            continue

        # StoreAbs, StoreIndexed: neutral.

    if not drop:
        return routine
    new_body = [it for i, it in enumerate(body) if i not in drop]
    return replace(routine, body=new_body)


def structure_module(module: ModuleIR1) -> ModuleIR1:
    """Apply fusion and flag-liveness elision to every routine in
    `module`. Returns a new `ModuleIR1`; the input is not mutated.

    The two transformations are sequenced per routine: fusion runs
    first (so the `If` consumers are visible to the liveness sweep),
    elision second (so any `cmp` whose `Branch` got absorbed and
    whose flags die before reaching another reader is dropped).
    """
    out: list[Routine] = []
    for r in module.routines:
        fused = structure_routine(r)
        out.append(_eliminate_dead_flags(fused))
    return ModuleIR1(name=module.name, file=module.file, routines=out)


def elision_stats(module: ModuleIR1) -> tuple[int, int]:
    """Return (remaining_cmp_count, remaining_setcarry_count). Used by
    the CLI to summarise how aggressive the elision pass was."""
    cmp = 0
    setc = 0
    for r in module.routines:
        for item in r.body:
            if isinstance(item, (CmpImm, CmpAbs)):
                cmp += 1
            elif isinstance(item, (Clc, Sec)):
                setc += 1
    return cmp, setc


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
