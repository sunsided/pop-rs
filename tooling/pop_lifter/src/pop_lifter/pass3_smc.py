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
    Label,
    LoadImm,
    ModuleIR1,
    Routine,
    SbcImm,
    StoreLocal,
    StoreOpVar,
)

# Immediate-operand instructions whose `#imm` byte sits at `label+1`.
_IMM_OPS = (LoadImm, AdcImm, SbcImm, CmpImm)


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

    # A label is a recognised immediate-SMC site if it's patched at
    # offset 1 and the instruction it labels has an immediate operand.
    opvar_for: dict[str, str] = {}      # label -> operand-variable name
    instr_idx_for: dict[str, int] = {}  # label -> index of patched instr
    for it in body:
        if isinstance(it, StoreLocal) and it.offset == 1:
            idx = _patched_instr_index(body, label_pos, it.target_label)
            if idx is None or not isinstance(body[idx], _IMM_OPS):
                continue
            opvar_for[it.target_label] = _opvar_name(it.target_label)
            instr_idx_for[it.target_label] = idx

    if not opvar_for:
        return routine

    new = list(body)
    # Rewrite every patch store of a recognised label into a StoreOpVar.
    for i, it in enumerate(new):
        if isinstance(it, StoreLocal) and it.offset == 1 and it.target_label in opvar_for:
            new[i] = StoreOpVar(reg=it.reg, name=opvar_for[it.target_label], src=it.src)
    # Mark the patched immediate so the interpreter reads the variable.
    for label, idx in instr_idx_for.items():
        instr = new[idx]
        new[idx] = replace(instr, imm=replace(instr.imm, opvar=opvar_for[label]))

    return replace(routine, body=new)


def recognize_smc(module: ModuleIR1) -> ModuleIR1:
    return ModuleIR1(
        name=module.name,
        file=module.file,
        routines=[recognize_routine(r) for r in module.routines],
    )


def smc_stats(module: ModuleIR1) -> int:
    """Total `StoreOpVar` patch stores produced across the module."""
    return sum(
        1
        for r in module.routines
        for it in r.body
        if isinstance(it, StoreOpVar)
    )
