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
   `CmpAbs`, `Clc`, `Sec`) and drops them.

   Routine boundaries are handled by an explicit call-graph fixed
   point: ``flag_demand[R]`` is the set of flags some caller of R
   observes after R returns. At a `Return` site the liveness sweep
   uses `flag_demand[R]` as the live set; at a `tail_call X` site
   we propagate `flag_demand[X] ⊇ flag_demand[R]`; at a non-tail
   `call X` site we propagate `flag_demand[X] ⊇ live-OUT[call]`.
   The iteration starts with every routine's demand at `∅` and
   only grows monotonically, bounded by `{Z,N,C,V}`, so it
   terminates after a handful of rounds for any module.

   Cross-module call targets are treated optimistically: the
   propagation simply has no effect on a `flag_demand` entry that
   doesn't exist. Once those callees are themselves lifted into a
   later pass-2 run the same iteration will pick up their demands.
   Pass 2 still assumes callees don't *read* flag inputs (no
   `clc;jsr` carry-passing convention in POP) — that part is the
   one remaining hand-waved assumption.

   Backward branches (loops) defeat the single-pass per-routine
   sweep, so the routine bails out of elision in that case rather
   than risk an unsound delete. Unsupported items are treated as
   read-everything, write-everything.

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
    AdcIndexed,
    Asl,
    Bit,
    Bitwise,
    Branch,
    Call,
    Clc,
    CmpAbs,
    CmpImm,
    CmpIndexed,
    CmpLocal,
    Compare,
    DecTarget,
    Goto,
    If,
    Imm,
    IncTarget,
    Item,
    Label,
    LoadAbs,
    LoadImm,
    LoadIndexed,
    LoadIndirect,
    LoadLocal,
    Lsr,
    ModuleIR1,
    Pla,
    Return,
    Rol,
    Ror,
    Routine,
    SbcAbs,
    SbcImm,
    SbcIndexed,
    SbcIndirect,
    Sec,
    ShiftMem,
    Transfer,
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


def _affected_register(item: Item):
    """Return the register whose new value the most recent flag-setter
    *exposed* via Z/N. Used to fuse `<defining-op> ; b(eq|ne|pl|mi) L`
    into `if reg <op> 0 goto L`. `cmp` is excluded — its rhs is
    explicit, so `_fuse_pair` handles it separately.

    Returns `None` if the predecessor isn't one of the register-
    visible flag setters (e.g. memory inc/dec, which sets Z/N from
    a memory cell that Compare can't reference yet).
    """
    if isinstance(item, (LoadAbs, LoadIndexed, LoadImm, LoadIndirect, LoadLocal)):
        return item.reg
    if isinstance(item, Transfer):
        return item.dst_reg
    if isinstance(item, (IncTarget, DecTarget)):
        # Only register-target inc/dec exposes a flag-readable
        # register — memory inc/dec would need a Compare form that
        # references memory, which we don't have yet.
        from .ir1 import Reg
        if isinstance(item.target, Reg):
            return item.target
        return None
    if isinstance(item, Bitwise):
        from .ir1 import Reg
        return Reg.A
    if isinstance(item, (SbcImm, SbcAbs, SbcIndexed, SbcIndirect, Asl, Lsr, Rol, Ror)):
        # All eight define A's new value as their flag-side-effect,
        # so a subsequent `beq`/`bne`/`bpl`/`bmi` reads Z/N of A.
        # Asl belongs here too — `asl a ; beq L` fuses into
        # `if a == 0 goto L` exactly like Lsr/Rol/Ror do. Earlier
        # comments referring to "symmetry with Asl" were misleading
        # because Asl wasn't actually on this list; now it is, so
        # the claim holds.
        from .ir1 import Reg
        return Reg.A
    # Adc{Imm,Abs,Indexed} deliberately NOT here — Adc both reads
    # and writes C, and pass-2's existing fusion paths haven't been
    # extended to track that pair properly. Adding any of them would
    # need to land together to keep the contract consistent.
    if isinstance(item, Pla):
        # PLA sets Z/N from the popped value (which lands in A).
        # `pla ; beq L` is the canonical "did the saved A end up
        # zero?" idiom — fuses cleanly into `if a == 0 goto L`.
        from .ir1 import Reg
        return Reg.A
    # `Bit` is excluded on purpose. Its Z reflects `A & operand`, not
    # A's own value, and our Compare form has no masked-equality
    # variant — fusing `bit ; beq` would silently rewrite to
    # `if a == 0` which is the WRONG predicate. Leave it unfused
    # until pass 3 introduces an expression-bearing Compare.
    return None


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
    if isinstance(prev, CmpIndexed) and cond_op is not None:
        # `cmp tbl,x ; bne :next` → `if a != *(tbl)[x] goto :next`.
        # `Compare.rhs` accepts `Imm | Abs | IndexedAbs | None`; for
        # the indexed cmp we wrap the (base, index) pair so dumps
        # render `*(tbl + x)` and pass 3 / pass 4 can recognise the
        # indexed shape when they emit Rust subscripts.
        from .ir1 import IndexedAbs
        return If(
            cond=Compare(
                reg=prev.reg,
                op=cond_op,
                rhs=IndexedAbs(base=prev.base, index=prev.index),
            ),
            target=branch.target,
            src=branch.src,
        )
    affected = _affected_register(prev)
    if affected is not None:
        load_op = _LOAD_FUSE_OPS.get(branch.cond)
        if load_op is None:
            return None
        op, needs_zero_rhs = load_op
        rhs = _ZERO_IMM if needs_zero_rhs else None
        return If(
            cond=Compare(reg=affected, op=op, rhs=rhs),
            target=branch.target,
            src=branch.src,
        )
    return None


def _defines_flags(item: Item) -> bool:
    """Heuristic: does this IR1 item define Z/N/C in a way the next
    branch might read? Conservative — only the obvious cases.

    Pass-1 long-tail additions (`IncTarget`/`DecTarget`/`Transfer`/
    `Bitwise`) all define Z/N from their result, so a subsequent
    `beq`/`bne`/`bpl`/`bmi` can fuse with them. (None of them touch
    C, so `bcs`/`bcc` wouldn't make sense — pass-2 fusion already
    only treats `cmp` as the C-defining predecessor.)
    """
    return isinstance(
        item,
        (
            CmpImm, CmpAbs, CmpIndexed, CmpLocal,
            LoadImm, LoadAbs, LoadIndexed, LoadIndirect, LoadLocal,
            IncTarget, DecTarget, Transfer, Bitwise,
            SbcImm, SbcAbs, SbcIndexed, SbcIndirect,
            Asl, Lsr, Rol, Ror, Bit, ShiftMem,
            Pla,
            # CmpLocal defines Z/N/C like any compare; LoadLocal defines
            # Z/N like any load. LoadLocal fuses via `_affected_register`;
            # CmpLocal has no `Compare.rhs` form for a local operand, so
            # it stays unfused and its branch reads the flags it set.
            # Both are listed so an *earlier* comparison can't fuse across
            # them.
            # AdcIndexed: see `_affected_register` note — adc lacks
            # the symmetric fusion path the others have, so its
            # presence in the body is treated as fusion-opaque
            # rather than a flag-setter the next branch can consume.
            AdcIndexed,
        ),
    )


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


def _backward_sweep(
    routine: Routine,
    flag_demand: dict[str, frozenset[str]],
) -> tuple[set[int], list[tuple[str, frozenset[str]]]]:
    """Run the backward liveness sweep over `routine.body` and return:

    * the set of body indices eligible for elision (pure flag-writers
      whose flags are all dead at their exit), and
    * a list of `(callee_name, demanded_flags)` propagations the
      fixed-point loop should fold into the global `flag_demand`.

    `flag_demand[R]` carries each in-module routine's currently-known
    return-flag liveness. Routines not present in the map are treated
    as cross-module — propagations to them are still emitted, the
    fixed-point loop just drops them.

    The sweep bails entirely (returns `(set(), [])`) when the routine
    contains a backward local jump, since the single-pass analysis
    can't soundly reason across loop back-edges yet.
    """
    body = routine.body
    if _has_backward_local_jump(body):
        return set(), []

    label_idx = {
        item.name: i for i, item in enumerate(body) if isinstance(item, Label)
    }
    label_live_in: dict[str, frozenset[str]] = {}

    # `Return` and `tail_call` leave the routine — the flags arriving
    # there are observed by whoever wants R's output. That's exactly
    # `flag_demand[R]`.
    exit_live = flag_demand.get(routine.name, frozenset())

    live: set[str] = set()
    drop: set[int] = set()
    propagations: list[tuple[str, frozenset[str]]] = []

    for i in range(len(body) - 1, -1, -1):
        item = body[i]

        if isinstance(item, Return):
            live = set(exit_live)
            continue

        if isinstance(item, Goto):
            if item.kind == "tail_call":
                # Executing `R` via tail_call to X delivers X's return
                # flags to R's caller, so X inherits R's demand.
                propagations.append((item.target, frozenset(exit_live)))
                # Live IN at the tail_call site: assume X doesn't read
                # flag inputs (POP convention), so ∅ — see module doc.
                live = set()
                continue
            # kind == "local"
            ti = label_idx.get(item.target)
            if ti is not None and ti > i:
                live = set(label_live_in.get(item.target, _UNIVERSE))
            else:
                live = set(_UNIVERSE)
            continue

        if isinstance(item, Branch):
            ti = label_idx.get(item.target)
            if ti is not None and ti > i:
                taken = set(label_live_in.get(item.target, _UNIVERSE))
            else:
                # Cross-routine conditional tail call — same shape as
                # a tail_call on the taken edge.
                propagations.append((item.target, frozenset(exit_live)))
                taken = set()
            live = live | taken
            live |= _COND_READS.get(item.cond, _UNIVERSE)
            continue

        if isinstance(item, If):
            ti = label_idx.get(item.target)
            if ti is not None and ti > i:
                taken = set(label_live_in.get(item.target, _UNIVERSE))
            else:
                # Cross-module conditional tail call.
                propagations.append((item.target, frozenset(exit_live)))
                taken = set()
            live = live | taken
            # `If` itself does NOT read flags — its Compare is
            # self-contained.
            continue

        if isinstance(item, Call):
            # The callee delivers `live` (live-OUT at the call) to us,
            # so it inherits that demand. Then the callee clobbers all
            # flags and (by POP convention) reads none — live IN at
            # the call site is `∅`.
            propagations.append((item.target, frozenset(live)))
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

        if isinstance(item, (LoadImm, LoadAbs, LoadIndexed, LoadIndirect, LoadLocal)):
            # Lda* writes Z/N; the load itself has a side-effect (the
            # register update) so we can't drop it even when Z/N are
            # dead. LoadLocal also reads the local store — same rule.
            live -= {"Z", "N"}
            continue

        if isinstance(item, Pla):
            # PLA writes Z/N from the popped value, mutates A, AND
            # pops a byte off the value stack — never elidable, even
            # if the resulting flags are dead.
            live -= {"Z", "N"}
            continue

        # Pha is a pure store (push) — no flag effect, side-effecting
        # on the value stack. The walker treats it like StoreAbs:
        # falls through to leave `live` unchanged and avoid adding it
        # to `drop`.

        if isinstance(item, (IncTarget, DecTarget, Transfer, Bitwise)):
            # Pass-1 long-tail ops: all write Z/N from their result and
            # have an observable side effect (register or memory
            # update). Like Lda*, never elidable; just clears Z/N
            # from the live set going backward.
            live -= {"Z", "N"}
            continue

        if isinstance(item, (AdcImm, AdcAbs, AdcIndexed)):
            # Adc writes Z,N,C and reads C. Indexed form has the same
            # flag effect as the abs/imm variants.
            live -= {"Z", "N", "C"}
            live.add("C")
            continue

        if isinstance(item, (SbcImm, SbcAbs, SbcIndexed, SbcIndirect)):
            # Sbc is symmetric with Adc: writes Z,N,C; reads C (the
            # borrow flag chains through subsequent sbc's).
            live -= {"Z", "N", "C"}
            live.add("C")
            continue

        if isinstance(item, (CmpIndexed, CmpLocal)):
            # Same flag effect as CmpImm/CmpAbs (writes Z,N,C without
            # touching A/X/Y), but **never elided** even when those
            # flags are dead — an indexed memory read can hit I/O
            # space, and silently dropping the read would change
            # program behavior. Same rationale as `Bit(Abs)` in
            # PR #12. `CmpLocal` reads the local store and is the
            # unfused predecessor of a raw flag branch, so dropping it
            # would lose the branch's predicate. Concretely: we clear
            # Z/N/C from the live set (the cmp DID write them) but don't
            # add the index to `drop`, so the instruction stays.
            live -= {"Z", "N", "C"}
            continue

        if isinstance(item, (Asl, Lsr)):
            # Shifts write Z,N,C and mutate A. They don't *read* C
            # — the shifted-in bit is always 0 — so the backward
            # sweep just clears Z/N/C from `live`.
            live -= {"Z", "N", "C"}
            continue

        if isinstance(item, (Rol, Ror)):
            # Rotates write Z,N,C **and read C** (the carry rotates
            # in from the opposite end of the shifted byte). Going
            # backward: clear the writes first, then add C as a
            # read so a preceding `sec`/`clc` (or any other carry-
            # setter) stays alive. Without the explicit `live.add
            # ("C")` the elision pass would happily drop the
            # `sec` before `rol`/`ror` — silently changing program
            # semantics. Mirrors `Adc*` / `Sbc*` for the same
            # reason.
            live -= {"Z", "N", "C"}
            live.add("C")
            continue

        if isinstance(item, ShiftMem):
            # Memory shift/rotate. Same flag effect (writes Z,N,C)
            # plus a RAM write as the headline side effect; never
            # elidable.
            live -= {"Z", "N", "C"}
            if item.op in ("rol", "ror"):
                # Memory rotates also read C — same reason as the
                # accumulator forms above.
                live.add("C")
            continue

        if isinstance(item, Bit):
            # `Bit(Imm)` is a pure flag-setter — no memory access at
            # all — so the normal Cmp-style elision rules apply.
            #
            # `Bit(Abs)` reads a byte from memory. On the Apple II,
            # `bit $c0xx` is the canonical idiom for toggling soft-
            # switches (speaker, page select, paddle reads, etc.) —
            # the *read itself* is the observable side effect. Even
            # outside the $C0xx page we can't tell from pass 2 alone
            # whether a memory read is to a soft-switch or to plain
            # RAM, so the safe and uniform rule is "never elide
            # Bit(Abs)". Future work that knows the soft-switch range
            # can relax this; for now correctness trumps the loss of
            # ~43 elisions worst case.
            from .ir1 import Imm
            defines = {"Z", "N", "V"}
            if isinstance(item.source, Imm) and not (live & defines):
                drop.add(i)
            else:
                live -= defines
            continue

        if isinstance(item, Label):
            label_live_in[item.name] = frozenset(live)
            continue

        # StoreAbs, StoreIndexed: neutral.

    return drop, propagations


def _solve_flag_demand(
    routines: list[Routine],
) -> dict[str, frozenset[str]]:
    """Iterate the per-routine backward sweep until `flag_demand`
    reaches a fixed point. Each iteration only grows demands
    monotonically (bounded by `{Z,N,C,V}`), so convergence is
    guaranteed in O(|routines| × |flags|) rounds.

    Returns the final `flag_demand` mapping, including entries only
    for routines visible in this module. Entries for cross-module
    callees referenced by `call`/`tail_call`/conditional-tail edges
    aren't tracked here — propagations to them are silently dropped,
    matching the optimistic "external callers don't observe flags"
    assumption documented at module top.
    """
    in_module = {r.name for r in routines}
    flag_demand: dict[str, frozenset[str]] = {
        name: frozenset() for name in in_module
    }

    while True:
        new_demand = dict(flag_demand)
        for r in routines:
            _, propagations = _backward_sweep(r, flag_demand)
            for callee, demanded in propagations:
                if callee in new_demand:
                    new_demand[callee] = new_demand[callee] | demanded
        if new_demand == flag_demand:
            return flag_demand
        flag_demand = new_demand


def _eliminate_dead_flags(
    routine: Routine,
    flag_demand: dict[str, frozenset[str]],
) -> Routine:
    drop, _ = _backward_sweep(routine, flag_demand)
    if not drop:
        return routine
    new_body = [it for i, it in enumerate(routine.body) if i not in drop]
    return replace(routine, body=new_body)


def structure_module(module: ModuleIR1) -> ModuleIR1:
    """Apply fusion and flag-liveness elision to every routine in
    `module`. Returns a new `ModuleIR1`; the input is not mutated.

    Sequencing per module:

    1. Fuse `cmp + branch` pairs (so the `If` consumers are visible
       to the liveness sweep).
    2. Solve the call-graph fixed-point for `flag_demand[R]` — which
       flags each routine must deliver to its callers.
    3. Run the elision sweep with that demand map, dropping any pure
       flag-writer whose flags are dead at its exit.

    Entry routines (no in-module callers) have `flag_demand[R] = ∅`,
    matching the optimistic stance that external callers don't
    observe returned flag state. Routines whose callers in the same
    module read return flags (e.g. `cmpspace`'s `Z` consumed by an
    `if ne goto ...` immediately after a `call cmpspace`) get the
    appropriate demand propagated in and keep their terminal cmps.
    """
    fused = [structure_routine(r) for r in module.routines]
    flag_demand = _solve_flag_demand(fused)
    final = [_eliminate_dead_flags(r, flag_demand) for r in fused]
    return ModuleIR1(name=module.name, file=module.file, routines=final)


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
