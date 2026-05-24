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
    AdcImm,
    Asl,
    Bitwise,
    Clc,
    CmpImm,
    DecTarget,
    Imm,
    IncTarget,
    IndexedAbs,
    IndirectY,
    LoadAbs,
    LoadImm,
    LoadIndexed,
    LoadIndirect,
    LocalRef,
    Reg,
    Rol,
    Ror,
    SbcIndirect,
    Sec,
    SourceRef,
    StoreAbs,
    StoreIndexed,
    StoreIndirect,
    Transfer,
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
    emit_modules,
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


def _module(routines: list, file: str = "syn/AUTO.S") -> ModuleIR3:
    return ModuleIR3(name="m", file=file, routines=routines)


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


def test_value_indirect_y():
    out = _emit_value(IndirectY(ptr=_abs("ptr", 0x20)))
    assert out == (
        "self.ram[(self.ram[0x0020] as usize "
        "| (self.ram[0x0021] as usize) << 8) + self.y as usize]"
    )


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


def test_assign_indirect_y_target():
    a = Assign(target=IndirectY(ptr=_abs("ptr", 0x20)), source=_imm(0), src=SRC)
    assert _emit_one(a) == (
        "self.ram[(self.ram[0x0020] as usize "
        "| (self.ram[0x0021] as usize) << 8) + self.y as usize] = 0x00;"
    )


def test_return_stmt():
    assert _emit_one(ReturnStmt(src=SRC)) == "return;"


def test_raw_load_abs_is_lowered():
    raw = RawStmt(item=LoadAbs(reg=Reg.A, source=_abs("X", 0x10), src=SRC))
    assert _emit_one(raw) == "self.a = self.ram[0x0010];"


def test_raw_deferred_atom_is_comment():
    # `cmp #imm` only sets Z/N/C flags with no register effect; that
    # flag model isn't lowered yet, so it stays as a `// raw:` comment.
    out = _emit_one(RawStmt(item=CmpImm(reg=Reg.A, imm=_imm(0), src=SRC)))
    assert out.startswith("// raw: ")
    assert "cmp" in out


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
        RawStmt(item=StoreAbs(reg=Reg.A, target=_abs("Z", 0x30), src=SRC)),  # lowered (data-movement)
        RawStmt(item=Clc(src=SRC)),  # lowered (carry op)
        RawStmt(item=CmpImm(reg=Reg.A, imm=_imm(0), src=SRC)),  # deferred (Z/N-only)
        CallStmt(target="sub", src=SRC),  # lowered
        ReturnStmt(src=SRC),  # lowered
    ]
    lowered, deferred = lower_stats(_module([_routine(stmts)]))
    assert (lowered, deferred) == (5, 1)


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


def test_compare_unexpected_sign_test_op_raises():
    # rhs=None must be a sign test ("<0"/">=0"); anything else surfaces.
    with pytest.raises(ValueError, match="unexpected sign-test Compare op"):
        _emit_compare(Compare(reg=Reg.A, op="==", rhs=None))


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


# ---------------------------------------------------------------- raw atom lowering


def _raw(item) -> str:
    return _emit_one(RawStmt(item=item))


def test_raw_load_imm():
    assert _raw(LoadImm(reg=Reg.X, imm=_imm(0x12), src=SRC)) == "self.x = 0x12;"


def test_raw_load_imm_opvar_deferred():
    # A self-modifying-code operand variable can't be lowered to a static
    # byte, so the whole load stays a `// raw:` comment.
    smc = Imm(value=0, text="#smXCO", opvar="smXCO")
    out = _raw(LoadImm(reg=Reg.A, imm=smc, src=SRC))
    assert out.startswith("// raw: ")


def test_raw_load_indexed():
    item = LoadIndexed(reg=Reg.A, base=_abs("tbl", 0x0200), index=Reg.Y, src=SRC)
    assert _raw(item) == "self.a = self.ram[0x0200 + self.y as usize];"


def test_raw_store_abs():
    assert _raw(StoreAbs(reg=Reg.Y, target=_abs("Z", 0x30), src=SRC)) == "self.ram[0x0030] = self.y;"


def test_raw_store_indexed():
    item = StoreIndexed(reg=Reg.A, base=_abs("tbl", 0x0200), index=Reg.X, src=SRC)
    assert _raw(item) == "self.ram[0x0200 + self.x as usize] = self.a;"


def test_raw_transfer():
    assert _raw(Transfer(src_reg=Reg.A, dst_reg=Reg.X, src=SRC)) == "self.x = self.a;"


def test_raw_bitwise_and_or_eor():
    assert _raw(Bitwise(op="and", source=_imm(0x0f), src=SRC)) == "self.a &= 0x0f;"
    assert _raw(Bitwise(op="or", source=_abs("M", 0x40), src=SRC)) == "self.a |= self.ram[0x0040];"
    eor_indexed = Bitwise(op="eor", source=IndexedAbs(base=_abs("t", 0x0200), index=Reg.X), src=SRC)
    assert _raw(eor_indexed) == "self.a ^= self.ram[0x0200 + self.x as usize];"


def test_raw_bitwise_indirect():
    item = Bitwise(op="and", source=IndirectY(ptr=_abs("ptr", 0x20)), src=SRC)
    assert _raw(item) == (
        "self.a &= self.ram[(self.ram[0x0020] as usize "
        "| (self.ram[0x0021] as usize) << 8) + self.y as usize];"
    )


def test_raw_inc_dec_register():
    assert _raw(IncTarget(target=Reg.X, src=SRC)) == "self.x = self.x.wrapping_add(1);"
    assert _raw(DecTarget(target=Reg.Y, src=SRC)) == "self.y = self.y.wrapping_sub(1);"


def test_raw_inc_dec_memory():
    assert _raw(IncTarget(target=_abs("M", 0x40), src=SRC)) == "self.ram[0x0040] = self.ram[0x0040].wrapping_add(1);"
    assert _raw(DecTarget(target=_abs("M", 0x40), src=SRC)) == "self.ram[0x0040] = self.ram[0x0040].wrapping_sub(1);"


def test_raw_inc_local_ref_deferred():
    # `inc :smod+2` bumps a self-modifying-code operand byte — deferred.
    item = IncTarget(target=LocalRef(label=":smod", offset=2), src=SRC)
    assert _raw(item).startswith("// raw: ")


def _indirect(name: str, addr: int) -> str:
    return (
        f"self.ram[(self.ram[{addr:#06x}] as usize "
        f"| (self.ram[{addr + 1:#06x}] as usize) << 8) + self.y as usize]"
    )


def test_raw_load_indirect():
    item = LoadIndirect(reg=Reg.A, source=IndirectY(ptr=_abs("ptr", 0x20)), src=SRC)
    assert _raw(item) == f"self.a = {_indirect('ptr', 0x20)};"


def test_raw_store_indirect():
    item = StoreIndirect(reg=Reg.A, target=IndirectY(ptr=_abs("ptr", 0x20)), src=SRC)
    assert _raw(item) == f"{_indirect('ptr', 0x20)} = self.a;"


def test_raw_sbc_indirect():
    item = SbcIndirect(source=IndirectY(ptr=_abs("ptr", 0x20)), src=SRC)
    lines = _emit_stmt(RawStmt(item=item), 0)
    assert lines[0] == (
        "let _r = (self.a as u16) + (!self.ram[(self.ram[0x0020] as usize "
        "| (self.ram[0x0021] as usize) << 8) + self.y as usize]) as u16 + (self.c as u16);"
    )
    assert lines[1] == "self.a = _r as u8;"
    assert lines[2] == "self.c = (_r >> 8) as u8;"


def test_indirect_high_byte_resolves_to_symbol():
    # The `ptr + 1` high byte must reuse the base's `sym::` const when the
    # low byte registers it, matching the `ztemp + 1` store form.
    lo = Assign(target=_abs("ztemp", 0xF0), source=_imm(0), src=SRC)
    load = RawStmt(item=LoadIndirect(reg=Reg.A, source=IndirectY(ptr=_abs("ztemp", 0xF0)), src=SRC))
    out = emit_module(_module([_routine([lo, load, ReturnStmt(src=SRC)], name="r")]))
    assert "self.a = self.ram[(self.ram[sym::ztemp] as usize " in out
    assert "(self.ram[sym::ztemp + 1] as usize) << 8) + self.y as usize];" in out


def test_indirect_ptr_with_offset_folds_into_high_byte():
    # An already-offset pointer `(ztemp+1),y` must keep symbol resolution:
    # the high byte is `ztemp+2`, not the unparseable `ztemp+1+1`.
    load = RawStmt(item=LoadIndirect(reg=Reg.A, source=IndirectY(ptr=_abs("ztemp+1", 0xF1)), src=SRC))
    out = emit_module(_module([_routine([load, ReturnStmt(src=SRC)], name="r")]))
    assert "self.a = self.ram[(self.ram[sym::ztemp + 1] as usize " in out
    assert "(self.ram[sym::ztemp + 2] as usize) << 8) + self.y as usize];" in out


# ---------------------------------------------------------------- symbolic address constants


def test_module_emits_sym_block_and_references():
    a = Assign(target=_abs("PlayCount", 0xa0), source=_imm(0), src=SRC)
    out = emit_module(_module([_routine([a, ReturnStmt(src=SRC)], name="r")]))
    assert "#[allow(non_upper_case_globals)]" in out
    assert "mod sym {" in out
    assert "    pub const PlayCount: usize = 0x00a0;" in out
    assert "self.ram[sym::PlayCount] = 0x00;" in out


def test_module_sym_offset_name():
    # `ztemp+1` resolves to the base `ztemp` const plus the offset, with
    # the base address recovered as 0xf1 - 1 = 0xf0.
    store = RawStmt(item=StoreAbs(reg=Reg.X, target=_abs("ztemp+1", 0xf1), src=SRC))
    out = emit_module(_module([_routine([store, ReturnStmt(src=SRC)], name="r")]))
    assert "    pub const ztemp: usize = 0x00f0;" in out
    assert "self.ram[sym::ztemp + 1] = self.x;" in out


def test_module_sym_conflict_falls_back_to_literal():
    # One name resolving to two addresses can't be a single const, so
    # both occurrences stay literal rather than emit a wrong constant.
    s1 = Assign(target=_abs("dup", 0x10), source=_imm(0), src=SRC)
    s2 = Assign(target=_abs("dup", 0x20), source=_imm(0), src=SRC)
    out = emit_module(_module([_routine([s1, s2, ReturnStmt(src=SRC)], name="r")]))
    assert "mod sym" not in out
    assert "self.ram[0x0010] = 0x00;" in out
    assert "self.ram[0x0020] = 0x00;" in out


def test_module_sym_unclean_name_falls_back_to_literal():
    # A name that isn't a valid Rust identifier can't become a const.
    s = Assign(target=_abs("we.ird", 0x30), source=_imm(0), src=SRC)
    out = emit_module(_module([_routine([s, ReturnStmt(src=SRC)], name="r")]))
    assert "mod sym" not in out
    assert "self.ram[0x0030] = 0x00;" in out


def test_module_without_named_addresses_has_no_sym_block():
    out = emit_module(_module([_routine([ReturnStmt(src=SRC)], name="r")]))
    assert "mod sym" not in out


# ---------------------------------------------------------------- carry-arithmetic atom lowering


def test_raw_clc_and_sec():
    assert _raw(Clc(src=SRC)) == "self.c = 0;"
    assert _raw(Sec(src=SRC)) == "self.c = 1;"


def test_raw_adc_imm():
    lines = _emit_stmt(RawStmt(item=AdcImm(imm=_imm(0xbd), src=SRC)), 0)
    assert lines[0] == "let _r = (self.a as u16) + (0xbd) as u16 + (self.c as u16);"
    assert lines[1] == "self.a = _r as u8;"
    assert lines[2] == "self.c = (_r >> 8) as u8;"


def test_raw_asl_and_lsr():
    asl_lines = _emit_stmt(RawStmt(item=Asl(src=SRC)), 0)
    assert asl_lines[0] == "self.c = self.a >> 7;"
    assert asl_lines[1] == "self.a = self.a.wrapping_shl(1);"
    lsr_lines = _emit_stmt(RawStmt(item=Ror(src=SRC)), 0)
    assert lsr_lines[0] == "let _c = self.a & 1;"
    assert lsr_lines[2] == "self.c = _c;"


def test_raw_rol_and_ror():
    rol = _emit_stmt(RawStmt(item=Rol(src=SRC)), 0)
    assert rol[0] == "let _c = self.a >> 7;"
    assert rol[1] == "self.a = self.a.wrapping_shl(1) | self.c;"
    assert rol[2] == "self.c = _c;"
    ror = _emit_stmt(RawStmt(item=Ror(src=SRC)), 0)
    assert ror[1] == "self.a = self.a.wrapping_shr(1) | (self.c << 7);"


# ---------------------------------------------------------------- multi-module emit


def test_emit_modules_single_matches_emit_module():
    # The single-module path must stay byte-identical to the standalone
    # form so the pilots don't churn.
    store = Assign(target=_abs("PlayCount", 0xA0), source=_imm(0), src=SRC)
    m = _module([_routine([store, ReturnStmt(src=SRC)], name="r")])
    assert emit_modules([m]) == emit_module(m)


def test_emit_modules_shares_one_sym_block():
    # Two modules with named addresses must not each emit a `mod sym`:
    # a Rust file may declare a module name only once.
    s1 = Assign(target=_abs("PlayCount", 0xA0), source=_imm(0), src=SRC)
    s2 = Assign(target=_abs("CharID", 0x4D), source=_imm(1), src=SRC)
    m1 = _module([_routine([s1, ReturnStmt(src=SRC)], name="a")], file="syn/A.S")
    m2 = _module([_routine([s2, ReturnStmt(src=SRC)], name="b")], file="syn/B.S")
    out = emit_modules([m1, m2])
    assert out.count("mod sym {") == 1
    # Both files contribute their own impl block and source line.
    assert out.count("impl Cpu {") == 2
    assert "// source: A.S" in out and "// source: B.S" in out
    # The merged block carries symbols from both modules.
    assert "pub const PlayCount: usize = 0x00a0;" in out
    assert "pub const CharID: usize = 0x004d;" in out


def test_emit_modules_conflicting_symbol_falls_back_to_literal():
    # The same base name resolving to two addresses across files is a
    # conflict: it must drop out of `mod sym` and stay literal everywhere.
    s1 = Assign(target=_abs("dup", 0x10), source=_imm(0), src=SRC)
    s2 = Assign(target=_abs("dup", 0x20), source=_imm(0), src=SRC)
    m1 = _module([_routine([s1, ReturnStmt(src=SRC)], name="a")], file="syn/A.S")
    m2 = _module([_routine([s2, ReturnStmt(src=SRC)], name="b")], file="syn/B.S")
    out = emit_modules([m1, m2])
    assert "sym::dup" not in out
    assert "self.ram[0x0010]" in out
    assert "self.ram[0x0020]" in out
