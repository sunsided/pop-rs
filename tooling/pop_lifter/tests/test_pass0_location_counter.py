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
