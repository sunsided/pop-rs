"""Pass-1 self-modifying-code / local-label store tests: `sta
:label+N` lifts to `StoreLocal` (operand patches and plain local
stores), the parser handles the `:label` / `]label` / `+N` forms,
and the interpreter records writes in the `code_patches` side
channel."""

from __future__ import annotations

from pop_lifter.interp_ir1 import Trace, exec_atom
from pop_lifter.ir1 import Reg, SourceRef, StoreLocal
from pop_lifter.pass0_lex import Line
from pop_lifter.pass1_lift import _lift_instr, _parse_local_target


def _line(mnemonic: str, operand: str | None = None) -> Line:
    return Line(
        file="syn",
        lineno=1,
        raw=f"  {mnemonic} {operand or ''}".rstrip(),
        label=None,
        mnemonic=mnemonic,
        operand=operand,
        comment=None,
    )


# ---- parser


def test_parse_local_target_with_offset():
    assert _parse_local_target(":smXCO+1") == (":smXCO", 1)
    assert _parse_local_target(":92+2") == (":92", 2)


def test_parse_local_target_macro_local():
    """`]label` (Merlin macro-local) is also a local target."""
    assert _parse_local_target("]cleanflag") == ("]cleanflag", 0)


def test_parse_local_target_bare_no_offset():
    """`:buffer` with no offset → offset 0 (plain local-label store,
    not necessarily SMC)."""
    assert _parse_local_target(":buffer") == (":buffer", 0)


def test_parse_local_target_whitespace_around_plus():
    assert _parse_local_target(":sm + 2") == (":sm", 2)


def test_parse_local_target_rejects_non_local():
    """A normal absolute symbol (no `:`/`]` prefix) is not a local
    target — it goes through the StoreAbs path instead."""
    assert _parse_local_target("CharX") is None
    assert _parse_local_target("$0080") is None
    assert _parse_local_target("table,x") is None


# ---- lifter dispatch


def test_sta_local_offset_lifts_to_storelocal():
    """`sta :smXCO+1` — the canonical SMC operand patch."""
    instr = _lift_instr(_line("sta", ":smXCO+1"), {}, set())
    assert isinstance(instr, StoreLocal)
    assert instr.reg is Reg.A
    assert instr.target_label == ":smXCO"
    assert instr.offset == 1


def test_stx_local_offset_lifts_to_storelocal():
    instr = _lift_instr(_line("stx", ":smodCD+1"), {}, set())
    assert isinstance(instr, StoreLocal)
    assert instr.reg is Reg.X
    assert instr.offset == 1


def test_sta_bare_local_lifts_to_storelocal_offset_zero():
    instr = _lift_instr(_line("sta", ":buffer"), {}, set())
    assert isinstance(instr, StoreLocal)
    assert instr.offset == 0


def test_resolved_label_takes_storeabs_not_storelocal():
    """If the store target resolves as a normal absolute symbol, it
    must take the StoreAbs path — StoreLocal is only the fallback
    for unresolved local labels. (`:`/`]` prefixed names never
    resolve as equates, so this just confirms ordering: a plain
    name like `CharX` with an equate goes to StoreAbs.)"""
    from pop_lifter.ir1 import StoreAbs
    instr = _lift_instr(_line("sta", "CharX"), {"CharX": 0x41}, set())
    assert isinstance(instr, StoreAbs)
    assert instr.target.addr == 0x41


# ---- interpreter semantics


def test_storelocal_records_patch_in_side_channel():
    """`sta :smXCO+1` writes A into `code_patches[(":smXCO", 1)]`,
    NOT into ram — local labels have no resolved address."""
    src = SourceRef(file="syn", line=0, raw="sta :smXCO+1")
    t = Trace(ram=bytearray(0x10000), a=0x42, x=0, y=0)
    exec_atom(
        StoreLocal(reg=Reg.A, target_label=":smXCO", offset=1, src=src),
        t, t.ram,
    )
    assert t.code_patches[(":smXCO", 1)] == 0x42
    # RAM is untouched — the patch doesn't alias a low address.
    assert all(b == 0 for b in t.ram)


def test_storelocal_stx_records_x():
    src = SourceRef(file="syn", line=0, raw="stx :smodCD+1")
    t = Trace(ram=bytearray(0x10000), a=0, x=0x99, y=0)
    exec_atom(
        StoreLocal(reg=Reg.X, target_label=":smodCD", offset=1, src=src),
        t, t.ram,
    )
    assert t.code_patches[(":smodCD", 1)] == 0x99


def test_storelocal_has_no_flag_effect():
    """Stores don't touch flags — confirm Z/N/C are untouched."""
    src = SourceRef(file="syn", line=0, raw="sta :sm+1")
    t = Trace(ram=bytearray(0x10000), a=0x00, x=0, y=0, c=1, z=1, n=1)
    exec_atom(
        StoreLocal(reg=Reg.A, target_label=":sm", offset=1, src=src),
        t, t.ram,
    )
    assert t.c == 1 and t.z == 1 and t.n == 1
