"""Pass 4 — Rust skeleton emitter: leaf expressions + control-flow lowering.

Slice 1 lowered module / routine scaffolding and leaf expressions
(folded `Assign`s, `return`). Slice 2 (this slice) additionally lowers
all structured control flow:

* `IfStmt` / `LoopStmt` / `DoWhileStmt` / `ForStmt` / `RepeatStmt`
* `BreakStmt` / `ContinueStmt`
* `MatchStmt`
* `CallStmt` / `TailCallStmt`
* `SaveTemp` / `RestoreTemp`

Still deferred: `RawStmt` (`// raw:`), `Wide16Stmt`, `RawIfStmt`,
`GotoStmt`/`LabelStmt` (`// TODO(pass4): …`).

All tests operate on synthetic IR3 (no source tree required).
"""

from __future__ import annotations

import pytest

from pop_lifter.ir1 import (
    Abs,
    Imm,
    IndexedAbs,
    IndirectY,
    LoadAbs,
    Reg,
    SourceRef,
    StoreAbs,
)
from pop_lifter.ir1 import Compare
from pop_lifter.ir3 import (
    Assign,
    BinExpr,
    Block,
    BreakStmt,
    CallStmt,
    ContinueStmt,
    DoWhileStmt,
    ForStmt,
    IfStmt,
    LoopStmt,
    MatchArm,
    MatchStmt,
    ModuleIR3,
    RawStmt,
    RepeatStmt,
    RestoreTemp,
    ReturnStmt,
    RotateExpr,
    RoutineIR3,
    SaveTemp,
    TailCallStmt,
)
from pop_lifter.pass4_emit_rust import (
    _emit_compare,
    _emit_stmt,
    _emit_value,
    emit_module,
    emit_routine,
    lower_stats,
)

SRC = SourceRef(file="syn", line=0, raw="")


# ---------------------------------------------------------------- helpers


def _imm(v: int) -> Imm:
    return Imm(value=v, text=f"#{v:#04x}")


def _abs(name: str, addr: int) -> Abs:
    return Abs(name=name, addr=addr)


def _emit_one(stmt) -> str:
    """Render a single statement at indent 0 (joined if multi-line)."""
    return "\n".join(_emit_stmt(stmt, 0))


def _routine(stmts: list, name: str = "r", aliases=None) -> RoutineIR3:
    return RoutineIR3(
        name=name,
        entry_aliases=list(aliases or []),
        body=Block.of(stmts),
    )


def _module(routines: list) -> ModuleIR3:
    return ModuleIR3(name="m", file="syn/AUTO.S", routines=routines)


# ---------------------------------------------------------------- value lowering


def test_value_imm():
    assert _emit_value(_imm(0x42)) == "0x42"


def test_value_imm_masks_to_byte():
    # `#-1` stores as signed -1; the emitted byte is masked to 0xff.
    assert _emit_value(Imm(value=-1, text="#-1")) == "0xff"


def test_value_abs_read():
    assert _emit_value(_abs("X", 0x10)) == "self.ram[0x0010]"


def test_value_indexed_read():
    iv = IndexedAbs(base=_abs("tbl", 0x0200), index=Reg.X)
    assert _emit_value(iv) == "self.ram[0x0200 + self.x as usize]"


def test_value_indirect_y_deferred():
    out = _emit_value(IndirectY(ptr=_abs("ptr", 0x20)))
    assert out == 'todo!("indirect (ptr),y read")'


def test_value_binexpr_add():
    e = BinExpr(op="+", lhs=_abs("X", 0x10), rhs=_imm(3))
    assert _emit_value(e) == "(self.ram[0x0010]).wrapping_add(0x03)"


def test_value_binexpr_sub():
    e = BinExpr(op="-", lhs=_abs("X", 0x10), rhs=_imm(1))
    assert _emit_value(e) == "(self.ram[0x0010]).wrapping_sub(0x01)"


def test_value_binexpr_shifts():
    shl = BinExpr(op="<<", lhs=_abs("X", 0x10), rhs=_imm(2))
    shr = BinExpr(op=">>", lhs=_abs("X", 0x10), rhs=_imm(1))
    assert _emit_value(shl) == "(self.ram[0x0010]).wrapping_shl(0x02 as u32)"
    assert _emit_value(shr) == "(self.ram[0x0010]).wrapping_shr(0x01 as u32)"


def test_value_rotate_is_method_call():
    rotl = RotateExpr(op="rotl", operand=_abs("X", 0x10), count=2)
    rotr = RotateExpr(op="rotr", operand=_abs("X", 0x10), count=1)
    assert _emit_value(rotl) == "self.rotl(self.ram[0x0010], 2)"
    assert _emit_value(rotr) == "self.rotr(self.ram[0x0010], 1)"


def test_value_unknown_binexpr_op_raises():
    with pytest.raises(ValueError, match="unknown BinExpr op"):
        _emit_value(BinExpr(op="?", lhs=_imm(1), rhs=_imm(2)))


# ---------------------------------------------------------------- statement lowering


def test_assign_imm_to_abs():
    a = Assign(target=_abs("Y", 0x20), source=_imm(1), src=SRC)
    assert _emit_one(a) == "self.ram[0x0020] = 0x01;"


def test_assign_abs_to_abs():
    a = Assign(target=_abs("Y", 0x20), source=_abs("X", 0x10), src=SRC)
    assert _emit_one(a) == "self.ram[0x0020] = self.ram[0x0010];"


def test_assign_indexed_target():
    tgt = IndexedAbs(base=_abs("tbl", 0x0200), index=Reg.Y)
    a = Assign(target=tgt, source=_imm(0), src=SRC)
    assert _emit_one(a) == "self.ram[0x0200 + self.y as usize] = 0x00;"


def test_assign_indirect_y_target_deferred():
    a = Assign(target=IndirectY(ptr=_abs("ptr", 0x20)), source=_imm(0), src=SRC)
    assert _emit_one(a) == "// TODO(pass4): store via IndirectY"


def test_return_stmt():
    assert _emit_one(ReturnStmt(src=SRC)) == "return;"


def test_raw_stmt_is_comment():
    raw = RawStmt(item=LoadAbs(reg=Reg.A, source=_abs("X", 0x10), src=SRC))
    out = _emit_one(raw)
    assert out.startswith("// raw: ")
    assert "X@0x0010" in out


def test_call_stmt_lowered():
    assert _emit_one(CallStmt(target="sub", src=SRC)) == "self.sub();"


def test_tail_call_stmt_lowered():
    lines = _emit_stmt(TailCallStmt(target="jump_target", src=SRC), 0)
    assert lines == ["self.jump_target();", "return;"]


def test_indent_is_four_spaces():
    a = Assign(target=_abs("Y", 0x20), source=_imm(1), src=SRC)
    assert _emit_stmt(a, 2) == ["        self.ram[0x0020] = 0x01;"]


# ---------------------------------------------------------------- routine / module framing


def test_routine_framing():
    a = Assign(target=_abs("Y", 0x20), source=_imm(1), src=SRC)
    lines = emit_routine(_routine([a, ReturnStmt(src=SRC)], name="do_thing"))
    assert lines[0] == "    fn do_thing(&mut self) {"
    assert lines[1] == "        self.ram[0x0020] = 0x01;"
    assert lines[2] == "        return;"
    assert lines[3] == "    }"


def test_routine_emits_aliases_comment():
    lines = emit_routine(_routine([ReturnStmt(src=SRC)], name="r", aliases=["alt1", "alt2"]))
    assert lines[0] == "    // aliases: alt1, alt2"
    assert lines[1] == "    fn r(&mut self) {"


def test_module_header_and_impl_block():
    out = emit_module(_module([_routine([ReturnStmt(src=SRC)], name="r")]))
    assert out.startswith("// @generated by pop_lifter")
    assert "// source: AUTO.S" in out
    assert "impl Cpu {" in out
    assert "    fn r(&mut self) {" in out
    assert out.endswith("}\n")


def test_module_separates_routines_with_blank_line():
    out = emit_module(_module([
        _routine([ReturnStmt(src=SRC)], name="a"),
        _routine([ReturnStmt(src=SRC)], name="b"),
    ]))
    assert "    }\n\n    fn b(&mut self) {" in out


# ---------------------------------------------------------------- stats


def test_lower_stats_counts_top_level():
    stmts = [
        Assign(target=_abs("Y", 0x20), source=_imm(1), src=SRC),  # lowered
        RawStmt(item=StoreAbs(reg=Reg.A, target=_abs("Z", 0x30), src=SRC)),  # deferred
        CallStmt(target="sub", src=SRC),  # lowered (CallStmt is now real code)
        ReturnStmt(src=SRC),  # lowered
    ]
    lowered, deferred = lower_stats(_module([_routine(stmts)]))
    assert (lowered, deferred) == (3, 1)


# ---------------------------------------------------------------- compare expressions


def test_compare_imm_eq():
    c = Compare(reg=Reg.A, op="==", rhs=_imm(0))
    assert _emit_compare(c) == "self.a == 0x00"


def test_compare_imm_ne():
    c = Compare(reg=Reg.X, op="!=", rhs=_imm(0x10))
    assert _emit_compare(c) == "self.x != 0x10"


def test_compare_abs_rhs():
    c = Compare(reg=Reg.A, op=">=", rhs=_abs("lim", 0x50))
    assert _emit_compare(c) == "self.a >= self.ram[0x0050]"


def test_compare_sign_test_bpl():
    # bpl fuses as op=">=0", rhs=None (N-flag clear = non-negative)
    c = Compare(reg=Reg.Y, op=">=0", rhs=None)
    assert _emit_compare(c) == "(self.y as i8) >= 0"


def test_compare_sign_test_bmi():
    c = Compare(reg=Reg.A, op="<0", rhs=None)
    assert _emit_compare(c) == "(self.a as i8) < 0"


# ---------------------------------------------------------------- control-flow lowering


def test_if_stmt_no_else():
    guard = IfStmt(
        cond=Compare(reg=Reg.A, op="==", rhs=_imm(0)),
        then_block=Block.of([ReturnStmt(src=SRC)]),
        else_block=None,
        src=SRC,
    )
    lines = _emit_stmt(guard, 0)
    assert lines == [
        "if self.a == 0x00 {",
        "    return;",
        "}",
    ]


def test_if_stmt_with_else():
    guard = IfStmt(
        cond=Compare(reg=Reg.A, op="!=", rhs=_imm(1)),
        then_block=Block.of([ReturnStmt(src=SRC)]),
        else_block=Block.of([BreakStmt(src=SRC)]),
        src=SRC,
    )
    lines = _emit_stmt(guard, 0)
    assert "} else {" in lines
    assert "    break;" in lines


def test_loop_stmt():
    inner = LoopStmt(body=Block.of([BreakStmt(src=SRC)]), src=SRC)
    lines = _emit_stmt(inner, 0)
    assert lines == ["loop {", "    break;", "}"]


def test_break_and_continue():
    assert _emit_one(BreakStmt(src=SRC)) == "break;"
    assert _emit_one(ContinueStmt(src=SRC)) == "continue;"


def test_do_while_stmt():
    dw = DoWhileStmt(
        body=Block.of([ReturnStmt(src=SRC)]),
        cond=Compare(reg=Reg.Y, op=">=0", rhs=None),
        src=SRC,
    )
    lines = _emit_stmt(dw, 0)
    assert lines[0] == "loop {"
    assert lines[1] == "    return;"
    assert lines[2] == "    if !((self.y as i8) >= 0) {"
    assert lines[3] == "        break;"
    assert lines[4] == "    }"
    assert lines[5] == "}"


def test_for_stmt_down_counter():
    fs = ForStmt(
        var=Reg.Y,
        start=_imm(6),
        step=-1,
        cond=Compare(reg=Reg.Y, op=">=0", rhs=None),
        body=Block.of([]),
        src=SRC,
    )
    lines = _emit_stmt(fs, 0)
    assert lines[0] == "self.y = 0x06;"
    assert lines[1] == "loop {"
    assert "    self.y = self.y.wrapping_sub(0x01);" in lines
    assert "    if !((self.y as i8) >= 0) {" in lines
    assert "        break;" in lines
    assert lines[-1] == "}"


def test_for_stmt_up_counter():
    fs = ForStmt(
        var=Reg.X,
        start=_imm(0),
        step=+1,
        cond=Compare(reg=Reg.X, op="!=", rhs=_imm(8)),
        body=Block.of([]),
        src=SRC,
    )
    lines = _emit_stmt(fs, 0)
    assert lines[0] == "self.x = 0x00;"
    assert "    self.x = self.x.wrapping_add(0x01);" in lines
    assert "    if !(self.x != 0x08) {" in lines


def test_repeat_stmt():
    rs = RepeatStmt(
        count=256,
        var=Reg.X,
        start=_imm(0),
        step=-1,
        body=Block.of([]),
        src=SRC,
    )
    lines = _emit_stmt(rs, 0)
    assert lines[0] == "self.x = 0x00;"
    assert lines[1] == "for _ in 0..256usize {"
    assert "    self.x = self.x.wrapping_sub(0x01);" in lines
    assert lines[-1] == "}"


def test_match_stmt():
    arm = MatchArm(
        values=(_imm(1), _imm(2)),
        body=Block.of([ReturnStmt(src=SRC)]),
    )
    ms = MatchStmt(reg=Reg.A, arms=(arm,), src=SRC)
    lines = _emit_stmt(ms, 0)
    assert lines[0] == "match self.a {"
    assert lines[1] == "    0x01 | 0x02 => {"
    assert lines[2] == "        return;"
    assert lines[3] == "    }"
    assert lines[4] == "    _ => {}"
    assert lines[5] == "}"


def test_save_restore_temp():
    assert _emit_one(SaveTemp(slot=0, src=SRC)) == "let tmp0 = self.a;"
    assert _emit_one(RestoreTemp(slot=0, src=SRC)) == "self.a = tmp0;"
    assert _emit_one(SaveTemp(slot=3, src=SRC)) == "let tmp3 = self.a;"
