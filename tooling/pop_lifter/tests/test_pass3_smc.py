"""Pass 3 — self-modifying-code operand-variable recognition.

* **Faithful interpretation** is the headline check: a synthetic routine
  patches an immediate at runtime, and after recognition the patched
  instruction actually reads the rewritten value (the opaque pre-pass
  model uses the stale placeholder instead).
* **Structural unit tests** pin what is and isn't recognised: an
  immediate `offset == 1` patch is rewritten; an `offset == 2` patch, a
  patch of a non-immediate instruction, and an unknown label stay
  opaque `StoreLocal`s.
"""

from __future__ import annotations

from pop_lifter.interp_ir1 import run as ir1_run
from pop_lifter.ir1 import (
    Abs,
    AdcImm,
    Clc,
    Imm,
    Label,
    LoadAbs,
    LoadImm,
    ModuleIR1,
    Reg,
    Return,
    Routine,
    SourceRef,
    StoreAbs,
    StoreIndexed,
    StoreLocal,
    StoreOpVar,
)
from pop_lifter.pass3_smc import (
    recognize_routine,
    recognize_smc,
    smc_store_count,
    smc_var_count,
)

SRC = SourceRef(file="syn", line=0, raw="")


def _imm(v: int) -> Imm:
    return Imm(value=v, text=f"#{v}")


def _smc_routine() -> Routine:
    """`a = #$42 ; sta :smL+1 ; a = #0 ; clc ; :smL adc #0 ; RESULT = a`.

    The store patches the immediate of the `adc` at `:smL`. Faithfully
    modelled, the `adc` then adds 0x42, not the placeholder 0."""
    return Routine(name="t", body=[
        LoadImm(reg=Reg.A, imm=_imm(0x42), src=SRC),
        StoreLocal(reg=Reg.A, target_label=":smL", offset=1, src=SRC),
        LoadImm(reg=Reg.A, imm=_imm(0x00), src=SRC),
        Clc(src=SRC),
        Label(name=":smL", src=SRC),
        AdcImm(imm=_imm(0x00), src=SRC),
        StoreAbs(reg=Reg.A, target=Abs(name="RESULT", addr=0x300), src=SRC),
        Return(src=SRC),
    ])


# --------------------------------------------------------------- faithful interpretation


def test_smc_patch_takes_effect_after_recognition():
    mod = ModuleIR1(name="M", file="syn", routines=[_smc_routine()])

    # Before recognition the patch is opaque: the `adc` uses its
    # placeholder #0, so RESULT = 0.
    raw = ir1_run(mod, "t", ram=bytearray(0x10000))
    assert raw.ram[0x300] == 0x00

    # After recognition the `adc` reads the operand variable, so the
    # 0x42 the store wrote actually lands: RESULT = 0x42.
    rec = recognize_smc(mod)
    out = ir1_run(rec, "t", ram=bytearray(0x10000))
    assert out.ram[0x300] == 0x42
    assert out.operand_vars["smL"] == 0x42


def test_recognition_rewrites_store_and_marks_immediate():
    rec = recognize_routine(_smc_routine())
    body = rec.body
    assert any(isinstance(it, StoreOpVar) and it.name == "smL" for it in body)
    assert not any(isinstance(it, StoreLocal) for it in body)
    adc = next(it for it in body if isinstance(it, AdcImm))
    assert adc.imm.opvar == "smL"
    # The placeholder value is preserved as the pre-patch fallback.
    assert adc.imm.value == 0x00


# --------------------------------------------------------------- not recognised


def test_offset_two_patch_stays_opaque():
    """`offset == 2` patches the high byte of a 16-bit address operand —
    out of scope; left as an opaque StoreLocal."""
    routine = Routine(name="t", body=[
        StoreLocal(reg=Reg.A, target_label=":smL", offset=2, src=SRC),
        Label(name=":smL", src=SRC),
        # lda $1234,y — a 3-byte address operand, not an immediate.
        StoreIndexed(reg=Reg.A, base=Abs(name="scr", addr=0x2000), index=Reg.Y, src=SRC),
        Return(src=SRC),
    ])
    rec = recognize_routine(routine)
    assert any(isinstance(it, StoreLocal) for it in rec.body)
    assert smc_store_count(ModuleIR1("M", "syn", [rec])) == 0


def test_patch_of_non_immediate_instruction_stays_opaque():
    """An `offset == 1` patch whose labelled instruction is a memory
    op (`lda abs`), not an immediate, isn't an operand-variable site."""
    routine = Routine(name="t", body=[
        StoreLocal(reg=Reg.A, target_label=":smL", offset=1, src=SRC),
        Label(name=":smL", src=SRC),
        LoadAbs(reg=Reg.A, source=Abs(name="src", addr=0x1234), src=SRC),
        Return(src=SRC),
    ])
    rec = recognize_routine(routine)
    assert any(isinstance(it, StoreLocal) for it in rec.body)
    assert not any(isinstance(it, StoreOpVar) for it in rec.body)


def test_unknown_label_stays_opaque():
    routine = Routine(name="t", body=[
        StoreLocal(reg=Reg.A, target_label=":nowhere", offset=1, src=SRC),
        Return(src=SRC),
    ])
    rec = recognize_routine(routine)
    assert any(isinstance(it, StoreLocal) for it in rec.body)
    assert not any(isinstance(it, StoreOpVar) for it in rec.body)


def test_multiple_patch_sites_share_one_operand_var():
    """Two stores patching the same label both become `StoreOpVar`s for
    the same operand variable, and the patched immediate is marked once."""
    routine = Routine(name="t", body=[
        StoreLocal(reg=Reg.A, target_label=":smL", offset=1, src=SRC),
        StoreLocal(reg=Reg.X, target_label=":smL", offset=1, src=SRC),
        Label(name=":smL", src=SRC),
        LoadImm(reg=Reg.A, imm=_imm(0x00), src=SRC),
        Return(src=SRC),
    ])
    rec = recognize_routine(routine)
    opvars = [it for it in rec.body if isinstance(it, StoreOpVar)]
    assert len(opvars) == 2 and all(it.name == "smL" for it in opvars)
    lda = next(it for it in rec.body if isinstance(it, LoadImm))
    assert lda.imm.opvar == "smL"
    # Two patch stores, but one operand variable.
    mod = ModuleIR1("M", "syn", [rec])
    assert smc_store_count(mod) == 2
    assert smc_var_count(mod) == 1
