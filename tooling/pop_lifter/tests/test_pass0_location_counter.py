"""Pass-0 location-counter resolution: `LABEL = *-table` / `a-b` table
sizes that POP uses as immediates (`cpx #maxgatevel`, `ldx #CHECKEND`).

Pass 0 tracks a running program counter through code and data so these
equates resolve to a byte distance. The value is origin-independent (the
position terms cancel), which the resolver verifies by re-evaluating at a
shifted origin — so an address-valued equate (`label+1`) is left
unresolved rather than invented from the synthetic PC base.
"""

from __future__ import annotations

from pop_lifter.pass0_parse import (
    _count_operands,
    _directive_size,
    _instr_size,
    eval_expr,
    parse_files,
)


def _write(tmp_path, name: str, body: str):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


# ---- expression: `*` resolves only when a PC is supplied


def test_star_resolves_from_symbols():
    assert eval_expr("*-base-1", {"base": 10, "*": 19}) == 8


def test_star_unresolvable_without_pc():
    import pytest
    with pytest.raises(ValueError, match="current-PC"):
        eval_expr("*-base", {"base": 10})


# ---- instruction sizing


def test_instr_size_modes():
    assert _instr_size("clc", None, {}) == 1            # implied
    assert _instr_size("rts", None, {}) == 1
    assert _instr_size("asl", "a", {}) == 1             # accumulator
    assert _instr_size("lda", "#$EE", {}) == 2          # immediate
    assert _instr_size("bne", ":0", {}) == 2            # relative
    assert _instr_size("lda", "($40),y", {}) == 2       # zp indirect
    assert _instr_size("sta", "$C005", {}) == 3         # absolute
    assert _instr_size("lda", "$40", {}) == 2           # zero page
    assert _instr_size("jmp", "target", {}) == 3        # jmp always 3
    assert _instr_size("sta", "fwd,x", {}) == 3         # unresolved → abs


# ---- directive sizing


def test_directive_sizes():
    assert _count_operands("0,0,0,20,40,60,80,100,120") == 9
    assert _directive_size("db", "0,0,0,20,40,60,80,100,120", {}) == 9
    assert _directive_size("dw", "a,b,c", {}) == 6      # 3 words × 2
    assert _directive_size("hex", "22,00,00,E1", {}) == 4
    assert _directive_size("hex", "2200E1", {}) == 3
    assert _directive_size("ds", "16", {}) == 16
    assert _directive_size("lda", "$2000", {}) is None  # not a directive


# ---- end-to-end: a `*`-relative table-size equate resolves


def test_star_relative_table_size_resolves(tmp_path):
    src = _write(tmp_path, "T.S", "\n".join([
        " org $2000",
        "tbl db 0,1,2,3,4",   # 5 bytes
        "tblsize = *-tbl",
        "",
    ]))
    ast = parse_files([src])
    assert ast.equates["tblsize"] == 5


def test_label_difference_resolves(tmp_path):
    src = _write(tmp_path, "T.S", "\n".join([
        " org $3000",
        "start dw 1,2,3",     # 6 bytes
        "mid dw 4,5",         # 4 bytes
        "stop",
        "span = stop-start",
        "",
    ]))
    ast = parse_files([src])
    assert ast.equates["span"] == 10


def test_instruction_span_size_resolves(tmp_path):
    # Mirrors BOOT's `CHECKEND = *-CHECKER`: a routine's byte length.
    src = _write(tmp_path, "T.S", "\n".join([
        " org $800",
        "rtn lda #$EE",       # 2
        " sta $C005",         # 3
        " bne :0",            # 2
        ":0 rts",             # 1
        "rtnsize = *-rtn",
        "",
    ]))
    ast = parse_files([src])
    assert ast.equates["rtnsize"] == 8


# ---- safety: an address-valued equate is NOT resolved from the PC base


def test_origin_dependent_equate_stays_unresolved(tmp_path):
    # `foo = bar+1` is position-dependent (an address), so it must not be
    # invented from the synthetic PC base — it stays unresolved, exactly
    # as before the location-counter pass existed.
    src = _write(tmp_path, "T.S", "\n".join([
        " org $2000",
        "bar db 0,0",
        "foo = bar+1",
        "",
    ]))
    ast = parse_files([src])
    assert "foo" not in ast.equates


# ---- PR #60 review fixes


def test_dum_block_does_not_affect_main_pc(tmp_path):
    # A `dum` overlay emits no object bytes and lives in its own address
    # space, so it must neither advance the main PC nor record its field
    # labels for position equates. `after-tbl` counts only `tbl`'s bytes.
    src = _write(tmp_path, "T.S", "\n".join([
        " org $2000",
        "tbl db 0,1,2",       # 3 bytes
        " dum $00",
        "fieldA ds 8",        # dum field — separate address space
        " dend",
        "after",
        "sz = after-tbl",
        "",
    ]))
    ast = parse_files([src])
    assert ast.equates["sz"] == 3


def test_unknown_macro_advances_pc_by_zero(tmp_path):
    # A macro invocation has unknown byte size, so PC tracking treats it
    # as 0 bytes rather than guessing a 1/2/3-byte instruction size.
    src = _write(tmp_path, "T.S", "\n".join([
        " org $2000",
        "tbl db 0,1",         # 2 bytes
        " mymacro arg1,arg2",  # unknown mnemonic → 0 bytes
        "after",
        "sz = after-tbl",
        "",
    ]))
    ast = parse_files([src])
    assert ast.equates["sz"] == 2


def test_masked_address_equate_stays_unresolved(tmp_path):
    # A masked address (`tbl & $fff`) is non-affine in the position term,
    # so it must not resolve from the PC base — even though it would
    # survive a naive two-origin (0 / 0x1000) shift comparison. Affine
    # analysis rejects any bitwise use of a label.
    src = _write(tmp_path, "T.S", "\n".join([
        " org $2000",
        "tbl db 0,0",
        "masked = tbl&$fff",
        "",
    ]))
    ast = parse_files([src])
    assert "masked" not in ast.equates


def test_origin_independent_division_resolves(tmp_path):
    # `(*-tbl)/2` is origin-independent: the position terms cancel before
    # the divide, so the coefficient is 0 and it resolves to the entry
    # count (8 bytes of `dw` / 2).
    src = _write(tmp_path, "T.S", "\n".join([
        " org $2000",
        "tbl dw 0,0,0,0",     # 8 bytes
        "count = (*-tbl)/2",
        "",
    ]))
    ast = parse_files([src])
    assert ast.equates["count"] == 4


def test_equate_wins_over_label_pc_in_fallback(tmp_path):
    # When a name is both an equate and a label, the equate value wins in
    # the position-equate fallback (matching `symbols()` precedence). `w`
    # is the equate constant 3, so `*-w` is an address expression (not a
    # size) and is correctly left unresolved rather than using w's PC.
    src = _write(tmp_path, "T.S", "\n".join([
        "w = 3",
        " org $2000",
        "w db 0,0",
        "sz = *-w",
        "",
    ]))
    ast = parse_files([src])
    assert "sz" not in ast.equates
