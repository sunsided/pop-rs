"""Pass 4 — Rust skeleton emitter (first slice).

The emitter lays down module / routine scaffolding and lowers *leaf
expressions*: folded `Assign`s (over immediates, absolute / indexed
reads, `BinExpr`, `RotateExpr`) and bare `return`s. Everything else
surfaces honestly as a comment rather than being dropped:

* `RawStmt` → `// raw: <ir1 line>`
* control flow / calls / temps → `// TODO(pass4): lower <Kind>`

These tests pin the rendered Rust text for each lowered form, the
comment fallbacks, the module/routine framing, and the lowered-vs-
deferred statistics. They operate on synthetic IR3 (no source tree),
mirroring the other pass-3 unit tests.
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
from pop_lifter.ir3 import (
    Assign,
    BinExpr,
    Block,
    CallStmt,
    IfStmt,
    ModuleIR3,
    RawStmt,
    ReturnStmt,
    RotateExpr,
    RoutineIR3,
)
from pop_lifter.ir1 import Compare
from pop_lifter.pass4_emit_rust import (
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


def test_call_stmt_deferred():
    assert _emit_one(CallStmt(target="sub", src=SRC)) == "// TODO(pass4): lower CallStmt"


def test_control_flow_deferred():
    guard = IfStmt(
        cond=Compare(reg=Reg.A, op="==", rhs=_imm(0)),
        then_block=Block.of([ReturnStmt(src=SRC)]),
        else_block=None,
        src=SRC,
    )
    assert _emit_one(guard) == "// TODO(pass4): lower IfStmt"


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
        CallStmt(target="sub", src=SRC),  # deferred
        ReturnStmt(src=SRC),  # lowered
    ]
    lowered, deferred = lower_stats(_module([_routine(stmts)]))
    assert (lowered, deferred) == (2, 2)
