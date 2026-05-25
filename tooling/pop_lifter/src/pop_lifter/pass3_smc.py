"""Pass 3 (semantic recovery): self-modifying-code operand variables.

POP's HIRES blitter patches instruction operands at runtime for speed:

        sta :smXCO+1      ; rewrite the immediate byte of :smXCO
        ...
    :smXCO:
        adc #$00          ; the #$00 is overwritten by the store above

Pass 1 lifts the patch store as an opaque `StoreLocal` keyed by
`(:smXCO, 1)` and the patched `adc` keeps its placeholder `#$00`, with
no connection between them — so the patch silently has no effect.

This pass recovers the connection for the common **immediate-operand**
case (`offset == 1`, patched instruction is `lda/ldx/ldy/adc/sbc/cmp
#imm`): it names an operand variable after the label, rewrites the
patch store into a `StoreOpVar`, and marks the patched immediate
(`Imm.opvar`) so the interpreter reads the rewritten value. The result
is a faithful, behaviour-correct model:

        opvar smXCO = a
        ...
    :smXCO:
        a = a + #{smXCO} + c

Out of scope (left as opaque `StoreLocal`): 16-bit address-operand
patches (`sta :smBASE+1 ; sta :smBASE+2 ; ... lda $0000,y`), branch
patches, and any patch whose target label / instruction this pass
can't resolve to an immediate operand.
"""

from __future__ import annotations

from dataclasses import replace

from .ir1 import (
    AdcImm,
    CmpImm,
    DecTarget,
    IncTarget,
    Label,
    LoadAbs,
    LoadImm,
    LoadIndexed,
    LocalRef,
    ModuleIR1,
    OpVarRef,
    Routine,
    SbcImm,
    StoreAbs,
    StoreIndexed,
    StoreLocal,
    StoreOpAddr,
    StoreOpVar,
)

# Immediate-operand instructions whose `#imm` byte sits at `label+1`.
_IMM_OPS = (LoadImm, AdcImm, SbcImm, CmpImm)

# Absolute-operand instructions whose 16-bit address sits at `label+1`
# (low byte) / `label+2` (high byte), mapped to the field holding the
# `Abs` operand to mark.
_ADDR_OP_FIELD = {
    LoadAbs: "source",
    StoreAbs: "target",
    LoadIndexed: "base",
    StoreIndexed: "base",
}


def _opvar_name(label: str) -> str:
    """Operand-variable name from a local label: strip Merlin's leading
    `:` / `]` local/macro sigils (`:smXCO` → `smXCO`)."""
    return label.lstrip(":]")


def _patched_instr_index(body: list, label_pos: dict, label: str):
    """Index of the first real instruction at `label` (skipping any
    stacked labels), or None if the label is unknown / ends the body."""
    i = label_pos.get(label)
    if i is None:
        return None
    j = i + 1
    while j < len(body) and isinstance(body[j], Label):
        j += 1
    return j if j < len(body) else None


def recognize_routine(routine: Routine) -> Routine:
    body = list(routine.body)
    label_pos = {it.name: i for i, it in enumerate(body) if isinstance(it, Label)}

    # Group every patch label's offsets and resolve the instruction it
    # labels, then classify:
    #   * immediate site — patched only at offset 1, labelled instruction
    #     has a `#imm` operand (`lda/adc/sbc/cmp #imm`);
    #   * address site — patched at offsets ⊆ {1, 2}, labelled instruction
    #     has a 16-bit `Abs` operand (`lda/sta abs[,x/y]`).
    # Anything else (opcode patch at offset 0, branch/jump-target patch,
    # unresolved label) is left as an opaque `StoreLocal` / `LocalRef`.
    # Both `sta :label+N` (StoreLocal) and `inc`/`dec :label+N`
    # (IncTarget/DecTarget on a LocalRef) count as patches.
    offsets_for: dict[str, set[int]] = {}
    idx_for: dict[str, int | None] = {}

    def _note(label: str, offset: int) -> None:
        offsets_for.setdefault(label, set()).add(offset)
        if label not in idx_for:
            idx_for[label] = _patched_instr_index(body, label_pos, label)

    for it in body:
        if isinstance(it, StoreLocal):
            _note(it.target_label, it.offset)
        elif isinstance(it, (IncTarget, DecTarget)) and isinstance(it.target, LocalRef):
            _note(it.target.label, it.target.offset)

    imm_var: dict[str, str] = {}            # label -> operand-var (immediate)
    imm_idx: dict[str, int] = {}            # label -> patched-instr index
    addr_var: dict[str, str] = {}           # label -> operand-var (address)
    addr_idx: dict[str, int] = {}
    for label, offsets in offsets_for.items():
        idx = idx_for[label]
        if idx is None:
            continue
        instr = body[idx]
        if offsets == {1} and isinstance(instr, _IMM_OPS):
            imm_var[label] = _opvar_name(label)
            imm_idx[label] = idx
        elif offsets <= {1, 2} and type(instr) in _ADDR_OP_FIELD:
            addr_var[label] = _opvar_name(label)
            addr_idx[label] = idx

    if not imm_var and not addr_var:
        return routine

    def _op_ref(label: str, offset: int) -> OpVarRef | None:
        """Map a recognised (label, offset) patch site to its operand-var
        reference: the whole immediate byte (`half=None`), or an address
        byte half."""
        if label in imm_var and offset == 1:
            return OpVarRef(name=imm_var[label], half=None)
        if label in addr_var and offset in (1, 2):
            return OpVarRef(name=addr_var[label], half="lo" if offset == 1 else "hi")
        return None

    new = list(body)
    # Rewrite the patch stores and the read-modify-write inc/dec bumps.
    for i, it in enumerate(new):
        if isinstance(it, StoreLocal):
            ref = _op_ref(it.target_label, it.offset)
            if isinstance(ref, OpVarRef) and ref.half is None:
                new[i] = StoreOpVar(reg=it.reg, name=ref.name, src=it.src)
            elif ref is not None:
                new[i] = StoreOpAddr(
                    reg=it.reg, name=ref.name, half=ref.half, src=it.src)
        elif isinstance(it, (IncTarget, DecTarget)) and isinstance(it.target, LocalRef):
            ref = _op_ref(it.target.label, it.target.offset)
            if ref is not None:
                new[i] = replace(it, target=ref)
    # Mark the patched immediate so the interpreter reads the variable.
    for label, idx in imm_idx.items():
        instr = new[idx]
        new[idx] = replace(instr, imm=replace(instr.imm, opvar=imm_var[label]))
    # Mark the patched address operand likewise, recording which byte
    # halves are patched so the emitter knows which to read from a field
    # and which to bake from the assembled address.
    for label, idx in addr_idx.items():
        instr = new[idx]
        field = _ADDR_OP_FIELD[type(instr)]
        operand = getattr(instr, field)
        halves = tuple(
            h for off, h in ((1, "lo"), (2, "hi")) if off in offsets_for[label]
        )
        new[idx] = replace(instr, **{field: replace(
            operand, opvar=addr_var[label], opvar_halves=halves)})

    return replace(routine, body=new)


def recognize_smc(module: ModuleIR1) -> ModuleIR1:
    return ModuleIR1(
        name=module.name,
        file=module.file,
        routines=[recognize_routine(r) for r in module.routines],
    )


def smc_var_count(module: ModuleIR1) -> int:
    """Number of distinct operand variables recognised — counted per
    routine (a local label is routine-scoped, so the same name in two
    routines is two variables) and summed. Counts both immediate
    (`StoreOpVar`) and 16-bit address (`StoreOpAddr`) operands; the two
    namespaces are disjoint per routine (a label patches one or the
    other), so a plain name-set union is exact. May be fewer than the
    number of patch *stores* (`smc_store_count`) when several stores
    rewrite the same operand (e.g. a low + high byte pair)."""
    return sum(
        len({
            it.name for it in r.body
            if isinstance(it, (StoreOpVar, StoreOpAddr))
        })
        for r in module.routines
    )


def smc_store_count(module: ModuleIR1) -> int:
    """Total patch *stores* produced across the module — both immediate
    (`StoreOpVar`) and address-byte (`StoreOpAddr`) patches."""
    return sum(
        1
        for r in module.routines
        for it in r.body
        if isinstance(it, (StoreOpVar, StoreOpAddr))
    )
