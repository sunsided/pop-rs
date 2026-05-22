"""Symbol-resolution tests: pass-0 label collection, low/high-byte
operators (`#<Label` / `#>Label`), and the interpreter's refusal to
dereference synthetic addresses."""

from __future__ import annotations

import pytest
from pathlib import Path

from pop_lifter.interp_ir1 import InterpError, Trace, _real_addr, exec_atom
from pop_lifter.ir1 import (
    Abs,
    LoadAbs,
    LoadIndexed,
    Reg,
    SourceRef,
    StoreAbs,
)
from pop_lifter.pass0_lex import Line
from pop_lifter.pass0_parse import (
    _LABEL_SENTINEL_BASE,
    parse_files,
)
from pop_lifter.pass1_lift import _lift_instr


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


# ---- pass-0 label collection


def test_global_labels_collected_with_sentinel_addresses(tmp_path: Path):
    """Every globally-scoped label across all parsed files ends up
    in `ProgramAST.labels` with a unique address ≥ 0x10000."""
    f = tmp_path / "FOO.S"
    f.write_text(
        "FirstLabel  lda  #$01\n"
        "            sta  $0080\n"
        "SecondLabel ldx  #$02\n"
    )
    ast = parse_files([f], search_paths=[tmp_path])
    assert "FirstLabel" in ast.labels
    assert "SecondLabel" in ast.labels
    assert ast.labels["FirstLabel"] >= _LABEL_SENTINEL_BASE
    assert ast.labels["SecondLabel"] >= _LABEL_SENTINEL_BASE
    # Each label gets a distinct address.
    assert ast.labels["FirstLabel"] != ast.labels["SecondLabel"]


def test_local_labels_not_collected(tmp_path: Path):
    """`:foo` / `]foo` local labels live in the lifter's per-routine
    namespace, not the cross-file symbol table."""
    f = tmp_path / "FOO.S"
    f.write_text(
        "Entry  lda  #$01\n"
        ":loop  dey\n"
        "       bpl  :loop\n"
        "]rts   rts\n"
    )
    ast = parse_files([f], search_paths=[tmp_path])
    assert "Entry" in ast.labels
    assert ":loop" not in ast.labels
    assert "]rts" not in ast.labels


def test_equate_labels_dont_double_count(tmp_path: Path):
    """`LABEL = EXPR` equate lines define `LABEL` in `equates`, not
    in `labels` — symbols() should still return a single entry, with
    the equate value (not the sentinel)."""
    f = tmp_path / "EQ.S"
    f.write_text("SCREEN = $2000\n")
    ast = parse_files([f], search_paths=[tmp_path])
    assert ast.equates.get("SCREEN") == 0x2000
    assert "SCREEN" not in ast.labels
    # Even if the merged table is consulted, equates win.
    assert ast.symbols()["SCREEN"] == 0x2000


def test_symbols_merge_equates_first(tmp_path: Path):
    """If somehow a name appears in both `equates` and `labels`,
    `symbols()` returns the equate's value (we trust real values
    over synthetic ones)."""
    f = tmp_path / "EQ.S"
    f.write_text("X = $42\n")
    ast = parse_files([f], search_paths=[tmp_path])
    # Force a collision (this shouldn't normally happen, but pin
    # the policy so future contributors don't get bitten).
    ast.labels["X"] = 0x10000
    assert ast.symbols()["X"] == 0x42


# ---- byte-operator handling in immediates


def test_immediate_with_low_byte_operator_resolves_to_low_byte():
    """`lda #<Label` — when `Label` resolves to a 16-bit value, the
    immediate is the low byte of that value."""
    instr = _lift_instr(
        _line("lda", "#<addr"),
        {"addr": 0x1234},
        set(),
    )
    assert instr.imm.value == 0x34


def test_immediate_with_high_byte_operator_resolves_to_high_byte():
    """`ldx #>Label` — same, but the high byte."""
    instr = _lift_instr(
        _line("ldx", "#>addr"),
        {"addr": 0x1234},
        set(),
    )
    assert instr.imm.value == 0x12


def test_immediate_with_synthetic_label_uses_synthetic_address_byte():
    """`lda #SymbolicLabel` against a label that pass 0 put at
    0x10000 (synthetic) gives the low byte of that synthetic
    address. The value isn't *meaningful* without an assembled
    binary, but the lift no longer falls through to Unsupported."""
    instr = _lift_instr(
        _line("lda", "#Label"),
        {"Label": 0x10042},
        set(),
    )
    # No <> operator → eval_expr returns the full value, then the
    # Imm's eventual `& 0xff` in the interpreter would take the low
    # byte. The Imm.value carries the full pre-mask number so the
    # dump can show the symbolic intent. We don't mask in
    # _parse_immediate when there's no byte operator.
    assert instr.imm.value == 0x10042


def test_immediate_with_low_byte_of_synthetic_label():
    """`lda #<SyntheticLabel` extracts the low byte at parse time."""
    instr = _lift_instr(
        _line("lda", "#<Label"),
        {"Label": 0x10042},
        set(),
    )
    assert instr.imm.value == 0x42


def test_immediate_with_high_byte_of_synthetic_label():
    """`ldx #>SyntheticLabel` — Merlin's `>` operator selects bits
    8–15 only: `(0x10042 >> 8) & 0xff` = `0x100 & 0xff` = **0x00**.
    Bit 16 (the `0x1` that distinguishes the synthetic range) is
    masked off because `>` is a single-byte operator. So the lifted
    immediate is 0x00 — same value the high byte of the underlying
    address would carry once pass 4 emits a real-binary form."""
    instr = _lift_instr(
        _line("ldx", "#>Label"),
        {"Label": 0x10042},
        set(),
    )
    assert instr.imm.value == 0x00      # (0x10042 >> 8) & 0xff = 0x00


# ---- interpreter refuses to dereference synthetic addresses


def test_real_addr_helper_passes_through_real_addresses():
    """Addresses below 0x10000 are masked to 16 bits and returned."""
    assert _real_addr(0x00, None) == 0x00       # zero page start
    assert _real_addr(0x80, None) == 0x80       # zero-page typical
    assert _real_addr(0xc000, None) == 0xc000   # I/O page
    assert _real_addr(0xffff, None) == 0xffff   # top of RAM — last
                                                 # real address before
                                                 # the synthetic cutoff


def test_real_addr_helper_rejects_synthetic_addresses():
    """Anything ≥ 0x10000 raises a clear InterpError pointing at the
    label-table origin."""
    with pytest.raises(InterpError, match="synthetic-label address"):
        _real_addr(0x10000, None)
    with pytest.raises(InterpError, match="synthetic-label address"):
        _real_addr(0x12345, None)


def test_load_through_synthetic_address_raises_interperror():
    """A LoadAbs whose source.addr is in the synthetic range fails
    loudly when the interpreter tries to dereference it."""
    src = SourceRef(file="syn", line=0, raw="lda Label")
    t = Trace(ram=bytearray(0x10000), a=0, x=0, y=0)
    with pytest.raises(InterpError, match="synthetic-label address"):
        exec_atom(
            LoadAbs(
                reg=Reg.A,
                source=Abs(name="Label", addr=0x10042),
                src=src,
            ),
            t,
            t.ram,
        )


def test_store_through_synthetic_address_raises_interperror():
    """Symmetric to load — synthetic addresses can't be written
    either. Same clear failure mode, not a silent low-RAM
    overwrite."""
    src = SourceRef(file="syn", line=0, raw="sta Label")
    t = Trace(ram=bytearray(0x10000), a=0xff, x=0, y=0)
    with pytest.raises(InterpError, match="synthetic-label address"):
        exec_atom(
            StoreAbs(
                reg=Reg.A,
                target=Abs(name="Label", addr=0x10042),
                src=src,
            ),
            t,
            t.ram,
        )


# ---- 16-bit wraparound must still work with real bases


def test_indexed_load_with_high_base_and_index_wraps_in_16_bits():
    """`lda $fff0,x` with `x = 0x30` should read from `$0020`
    (`0xfff0 + 0x30 = 0x10020` → wraps to `0x0020`). The synthetic-
    address gate runs on the *base* (which is real) — the wrapped
    effective address is computed afterwards. A naive check that
    applied `_real_addr` to the sum would have falsely flagged
    `0x10020` as synthetic."""
    src = SourceRef(file="syn", line=0, raw="lda $fff0,x")
    t = Trace(ram=bytearray(0x10000), a=0, x=0x30, y=0)
    t.ram[0x0020] = 0xab
    exec_atom(
        LoadIndexed(
            reg=Reg.A,
            base=Abs(name="hi", addr=0xfff0),
            index=Reg.X,
            src=src,
        ),
        t,
        t.ram,
    )
    assert t.a == 0xab, (
        f"expected wraparound to read 0xab from $0020, got {t.a:#04x}"
    )


def test_indirect_y_pointer_at_top_of_ram_wraps_high_byte():
    """`(ptr),y` where ptr = $ffff means the high byte of the
    16-bit pointer lives at $0000 (16-bit wrap). The synthetic gate
    validates the base ($ffff is real); the high-byte read at
    base+1 then wraps cleanly to $0000.

    Real NMOS chips have the page-wrap bug where ($ff),y reads the
    high byte from $00 instead of $100. We use the cleaner
    full-16-bit wrap (documented on `IndirectY`); the synthetic
    gate must not block this either way."""
    from pop_lifter.ir1 import IndirectY, LoadIndirect

    src = SourceRef(file="syn", line=0, raw="lda ($ffff),y")
    t = Trace(ram=bytearray(0x10000), a=0, x=0, y=0)
    t.ram[0xffff] = 0x34   # low byte of effective pointer
    t.ram[0x0000] = 0x12   # high byte at wrapped $0000
    t.ram[0x1234] = 0x77
    exec_atom(
        LoadIndirect(
            reg=Reg.A,
            source=IndirectY(ptr=Abs(name="p", addr=0xffff)),
            src=src,
        ),
        t,
        t.ram,
    )
    assert t.a == 0x77


def test_indexed_load_with_synthetic_base_still_rejected():
    """The wrap fix must not weaken the synthetic-address gate —
    a `LoadIndexed` whose *base* is synthetic still has to raise."""
    src = SourceRef(file="syn", line=0, raw="lda Label,x")
    t = Trace(ram=bytearray(0x10000), a=0, x=0, y=0)
    with pytest.raises(InterpError, match="synthetic-label address"):
        exec_atom(
            LoadIndexed(
                reg=Reg.A,
                base=Abs(name="Label", addr=0x10042),
                index=Reg.X,
                src=src,
            ),
            t,
            t.ram,
        )
