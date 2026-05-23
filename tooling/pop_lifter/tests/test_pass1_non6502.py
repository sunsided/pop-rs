"""Pass-1 non-6502 cleanup: Merlin data pseudo-ops (rev/usr/da)
recognised as non-code, plus the simple status/no-op opcodes
(nop, sei/cli/sed/cld) and 65C02 `bra`."""

from __future__ import annotations

from pop_lifter.interp_ir1 import Trace, exec_atom
from pop_lifter.ir1 import FlagOp, Goto, Nop, SourceRef
from pop_lifter.pass0_lex import Line
from pop_lifter.pass0_parse import parse_files
from pop_lifter.pass1_lift import _lift_instr, discover_entries


def _line(mnemonic: str, operand: str | None = None, label: str | None = None) -> Line:
    return Line(
        file="syn",
        lineno=1,
        raw=f"{label or ''}  {mnemonic} {operand or ''}".strip(),
        label=label,
        mnemonic=mnemonic,
        operand=operand,
        comment=None,
    )


# ---- Merlin data pseudo-ops are non-code


def test_rev_lifts_to_none():
    """`rev "SKIP"` is data (a reversed string), not an opcode — the
    lifter returns None for it (same as `db`/`dw`)."""
    assert _lift_instr(_line("rev", '"SKIP"'), {}, set()) is None


def test_usr_and_da_lift_correctly():
    """`da` is inert data → None. `usr` is a code/data *generator*
    (`usr $a9,N,addr,*-org`), not literal data — it stays visible as
    `Unsupported` so the IR flags that an unexpanded generated block
    lives here, rather than silently dropping it like `rev`/`da`."""
    from pop_lifter.ir1 import Unsupported
    assert _lift_instr(_line("da", "label"), {}, set()) is None
    usr = _lift_instr(_line("usr", "$a9,16,$b00,*-org"), {}, set())
    assert isinstance(usr, Unsupported)
    assert usr.mnemonic == "usr"


def test_cheatcode_data_labels_not_discovered_as_entries(tmp_path):
    """A label attached to `rev` (POP's cheat-code table:
    `C_skip rev "SKIP"`) must NOT be treated as a routine entry —
    it's data. Previously these surfaced as all-`??? rev` routines."""
    f = tmp_path / "CHEAT.S"
    f.write_text(
        "RealRoutine  lda #$01\n"
        "             rts\n"
        'C_skip       rev "SKIP"\n'
        "             db 0\n"
        'C_devel      rev "POP"\n'
        "             db 0\n"
    )
    ast = parse_files([f], search_paths=[tmp_path])
    file_ast = next(x for x in ast.files if x.path.endswith("CHEAT.S"))
    entries = discover_entries(file_ast)
    assert "RealRoutine" in entries
    assert "C_skip" not in entries
    assert "C_devel" not in entries


# ---- nop / status flags


def test_nop_lifts_and_is_a_noop():
    instr = _lift_instr(_line("nop"), {}, set())
    assert isinstance(instr, Nop)
    # Executing it changes nothing.
    src = SourceRef(file="syn", line=0, raw="nop")
    t = Trace(ram=bytearray(0x10000), a=0x42, x=1, y=2, c=1, z=1, n=1)
    exec_atom(Nop(src=src), t, t.ram)
    assert (t.a, t.x, t.y, t.c, t.z, t.n) == (0x42, 1, 2, 1, 1, 1)


def test_sei_cli_set_clear_interrupt_flag():
    assert _lift_instr(_line("sei"), {}, set()) == FlagOp(
        flag="I", value=1, src=_lift_instr(_line("sei"), {}, set()).src
    )
    src = SourceRef(file="syn", line=0, raw="")
    t = Trace(ram=bytearray(0x10000), a=0, x=0, y=0)
    exec_atom(FlagOp(flag="I", value=1, src=src), t, t.ram)
    assert t.i == 1
    exec_atom(FlagOp(flag="I", value=0, src=src), t, t.ram)
    assert t.i == 0


def test_sed_cld_set_clear_decimal_flag():
    sed = _lift_instr(_line("sed"), {}, set())
    cld = _lift_instr(_line("cld"), {}, set())
    assert isinstance(sed, FlagOp) and sed.flag == "D" and sed.value == 1
    assert isinstance(cld, FlagOp) and cld.flag == "D" and cld.value == 0
    src = SourceRef(file="syn", line=0, raw="")
    t = Trace(ram=bytearray(0x10000), a=0, x=0, y=0)
    exec_atom(sed, t, t.ram)
    assert t.d == 1


def test_flagop_does_not_touch_zn_c():
    """sei/cli/sed/cld don't affect Z/N/C — confirm they're
    transparent to the flags pass-2 actually tracks."""
    src = SourceRef(file="syn", line=0, raw="")
    t = Trace(ram=bytearray(0x10000), a=0, x=0, y=0, c=1, z=1, n=1)
    exec_atom(FlagOp(flag="D", value=0, src=src), t, t.ram)
    assert (t.c, t.z, t.n) == (1, 1, 1)


# ---- bra (65C02 branch-always) → unconditional Goto


def test_bra_local_lifts_to_local_goto():
    instr = _lift_instr(_line("bra", ":loop"), {}, set())
    assert isinstance(instr, Goto)
    assert instr.kind == "local"
    assert instr.target == ":loop"


def test_bra_nonlocal_lifts_to_tail_call_goto():
    instr = _lift_instr(_line("bra", "elsewhere"), {}, set())
    assert isinstance(instr, Goto)
    assert instr.kind == "tail_call"
