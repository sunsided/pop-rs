"""65816 (IIgs-only) instructions are marked as explicit out-of-scope
platform stubs, distinct from generic unknown opcodes."""

from __future__ import annotations

from pop_lifter.ir1 import OPS_65816, SourceRef, Unsupported, format_item, is_65816
from pop_lifter.ir3 import RawStmt
from pop_lifter.pass4_emit_rust import _emit_stmt

SRC = SourceRef(file="syn", line=1, raw="")


def _unsup(mnemonic: str, operand: str | None = None) -> Unsupported:
    return Unsupported(mnemonic=mnemonic, operand=operand, src=SRC)


def test_is_65816_covers_native_ops_only():
    for m in ("xce", "rep", "sep", "mvn", "phb", "plb"):
        assert is_65816(m), m
    # 65C02 ops (present on enhanced IIe/IIc, modellable later) are NOT
    # treated as out-of-scope IIgs platform ops — including stp/wai, which
    # are WDC-65C02 instructions, not 65816-exclusive.
    for m in ("phy", "tsb", "trb", "stp", "wai", "lda", "sta"):
        assert not is_65816(m), m
    assert OPS_65816 >= {"xce", "rep", "sep", "mvn", "mvp", "phb", "plb"}


def test_ir1_dump_labels_65816_not_as_unknown():
    got = format_item(_unsup("xce")).strip()
    assert "65816/IIgs op (not modeled)" in got
    assert "???" not in got
    # A genuinely-unknown opcode keeps the `???` marker.
    assert format_item(_unsup("phy")).strip().startswith("??? phy")


def test_pass4_emits_65816_platform_stub():
    # The stub keeps the source ref so readers can jump back to the .S line.
    out = _emit_stmt(RawStmt(item=_unsup("mvn", "$E1,1")), 0)
    assert out == ["// 65816 (IIgs-only, not modeled): mvn $E1,1  ; syn:1"]
    # No operand → just the mnemonic.
    assert _emit_stmt(RawStmt(item=_unsup("xce")), 0) == [
        "// 65816 (IIgs-only, not modeled): xce  ; syn:1"
    ]


def test_ir1_dump_keeps_source_ref_for_65816():
    assert format_item(_unsup("xce")).strip().endswith("(not modeled) — syn:1")


def test_pass4_unknown_opcode_still_raw_comment():
    out = _emit_stmt(RawStmt(item=_unsup("phy")), 0)
    assert out[0].startswith("// raw: ??? phy")
