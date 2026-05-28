from __future__ import annotations

from pathlib import Path

from pop_lifter.pass0_lex import lex_line


P = Path("test.S")


def L(raw: str):
    return lex_line(P, 1, raw)


def test_blank_line():
    assert L("").is_blank


def test_full_line_comment():
    line = L("* this is a comment")
    assert line.is_blank
    assert line.comment == "this is a comment"


def test_column1_label_with_opcode():
    line = L("rw18 = $d000")
    assert line.label == "rw18"
    assert line.mnemonic == "="
    assert line.operand == "$d000"


def test_indented_pseudoop():
    line = L(" dum master")
    assert line.label is None
    assert line.mnemonic == "dum"
    assert line.operand == "master"


def test_local_label():
    line = L(":loop pha")
    assert line.label == ":loop"
    assert line.mnemonic == "pha"
    assert line.operand is None


def test_macro_label():
    line = L("]rts rts")
    assert line.label == "]rts"
    assert line.mnemonic == "rts"


def test_label_with_ds():
    line = L("trobspace = $20")
    assert line.is_equate
    assert line.operand == "$20"


def test_inline_comment_stripped():
    line = L("unpack = $ea00 ;game only")
    assert line.label == "unpack"
    assert line.mnemonic == "="
    assert line.operand == "$ea00"
    assert line.comment == "game only"


def test_label_with_indented_operand_after_tab():
    line = L("_firstboot\tds\t3")
    assert line.label == "_firstboot"
    assert line.mnemonic == "ds"
    assert line.operand == "3"


def test_indented_ds_no_label():
    line = L(" ds 15")
    assert line.label is None
    assert line.mnemonic == "ds"
    assert line.operand == "15"


def test_space_delimited_comment_after_operand():
    # Merlin ends the operand at the first unquoted whitespace; the rest
    # is a comment even without a `;` (POP does this with jokey labels).
    line = L(" lda #99 \"stabbed\"")
    assert line.mnemonic == "lda"
    assert line.operand == "#99"
    assert line.comment == '"stabbed"'


def test_negative_immediate_with_space_comment():
    line = L(" lda #-5 impaled")
    assert line.mnemonic == "lda"
    assert line.operand == "#-5"
    assert line.comment == "impaled"


def test_quoted_operand_keeps_internal_spaces():
    # A quoted string operand (asc/dci/...) must keep its spaces — the
    # split only triggers on whitespace *outside* quotes.
    line = L(' asc "FOO BAR"')
    assert line.mnemonic == "asc"
    assert line.operand == '"FOO BAR"'
    assert line.comment is None


def test_indexed_operand_unaffected():
    line = L(" sta table,x")
    assert line.operand == "table,x"
    assert line.comment is None
