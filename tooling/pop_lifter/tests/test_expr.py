from __future__ import annotations

from pop_lifter.pass0_parse import eval_expr


def test_decimal():
    assert eval_expr("42", {}) == 42


def test_hex():
    assert eval_expr("$ff", {}) == 0xFF
    assert eval_expr("$F880", {}) == 0xF880


def test_binary():
    assert eval_expr("%11000000", {}) == 0xC0
    assert eval_expr("%00011111", {}) == 0x1F


def test_arithmetic():
    assert eval_expr("24*30", {}) == 720
    assert eval_expr("$100-1", {}) == 255


def test_symbol():
    assert eval_expr("foo+1", {"foo": 7}) == 8
    assert eval_expr("maxpeel*2", {"maxpeel": 46}) == 92


def test_parens():
    assert eval_expr("(1+2)*3", {}) == 9


def test_unary_minus():
    assert eval_expr("-1", {}) == -1
    assert eval_expr("$100+-1", {}) == 255


def test_bitwise():
    assert eval_expr("$F0&$0F", {}) == 0
    assert eval_expr("$F0|$0F", {}) == 0xFF
