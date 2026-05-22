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
