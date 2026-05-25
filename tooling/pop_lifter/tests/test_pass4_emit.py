"""Pass 4 — Rust skeleton emitter: leaf expressions + control-flow lowering.

Slice 1 lowered module / routine scaffolding and leaf expressions
(folded `Assign`s, `return`). Slice 2 (this slice) additionally lowers
all structured control flow:

* `IfStmt` / `LoopStmt` / `DoWhileStmt` / `ForStmt` / `RepeatStmt`
* `BreakStmt` / `ContinueStmt`
* `MatchStmt`
* `CallStmt` / `TailCallStmt`
* `SaveTemp` / `RestoreTemp`

Later slices added `RawStmt` atom lowering (data-movement, carry
arithmetic, `(ptr),y` indirect, cmp/bit flags) and `Wide16Stmt` 16-bit
arithmetic. Still deferred: the stack, SMC, `RawIfStmt`, and
`GotoStmt`/`LabelStmt` (`// raw:` / `// TODO(pass4): …`).

All tests operate on synthetic IR3 (no source tree required).
"""

from __future__ import annotations

import pytest

from pop_lifter.ir1 import (
    Abs,
    AdcImm,
    Asl,
    Bit,
    Bitwise,
    Clc,
    CmpAbs,
    CmpImm,
    CmpIndexed,
    CmpIndirect,
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
    Pha,
    Pla,
    Reg,
    Rol,
    Ror,
    SbcImm,
    SbcIndirect,
    Sec,
    SourceRef,
    StoreAbs,
    StoreIndexed,
    StoreIndirect,
    StoreLocal,
    StoreOpAddr,
    StoreOpVar,
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
    DispatchArm,
    DispatchStmt,
    DoWhileStmt,
    ForStmt,
    GotoStateStmt,
    IfStmt,
    LoopStmt,
    MatchArm,
    MatchStmt,
    ModuleIR3,
    RawIfStmt,
    RawStmt,
    RepeatStmt,
    RestoreTemp,
    ReturnStmt,
    RotateExpr,
    RoutineIR3,
    SaveTemp,
    TailCallStmt,
    Wide16Stmt,
)
from pop_lifter.pass4_emit_rust import (
    _emit_compare,
    _emit_stmt,
    _emit_value,
    _mangle,
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


def test_raw_pha_pushes_value_stack():
    # An unpaired `pha` (pass 3 couldn't fold it into a SaveTemp) lowers
    # over the provisional value stack.
    assert _emit_one(RawStmt(item=Pha(src=SRC))) == "self.stack.push(self.a);"


def test_raw_pla_pops_and_sets_flags():
    # `pla` pops into A and sets Z/N from the byte, matching the 6502.
    lines = _emit_stmt(RawStmt(item=Pla(src=SRC)), 0)
    assert lines == [
        'self.a = self.stack.pop().expect("pla on empty stack");',
        "self.z = (self.a == 0) as u8;",
        "self.n = self.a >> 7;",
    ]


def test_raw_deferred_atom_is_comment():
    # An SMC operand patch needs a consumer model that isn't lowered yet,
    # so it stays as a `// raw:` comment rather than emitting wrong code.
    out = _emit_one(RawStmt(item=StoreLocal(
        reg=Reg.A, target_label="smXCO", offset=1, src=SRC,
    )))
    assert out.startswith("// raw: ")


def test_call_stmt_lowered():
    assert _emit_one(CallStmt(target="sub", src=SRC)) == "self.sub();"


def test_tail_call_stmt_lowered():
    lines = _emit_stmt(TailCallStmt(target="jump_target", src=SRC), 0)
    assert lines == ["self.jump_target();", "return;"]


def test_mangle_passes_valid_idents_through():
    assert _mangle("ANIMCHAR") == "ANIMCHAR"
    assert _mangle("getseq") == "getseq"


def test_mangle_escapes_non_identifier_names():
    # Merlin sigils / raw-address / `*` targets aren't valid Rust idents;
    # each non-ident byte is escaped as `_<hex>` (`:`=3a, `]`=5d, `$`=24).
    assert _mangle(":next") == "_3anext"
    assert _mangle("]clr") == "_5dclr"
    assert _mangle("$c00") == "_24c00"
    assert _mangle(":slice?") == "_3aslice_3f"


def test_call_target_resolves_alias_to_canonical():
    # A call to an entry alias (`:next`, an alias of ANIMCHAR) must emit
    # the canonical method, not an undefined alias method.
    names = {"ANIMCHAR": "ANIMCHAR", ":next": "ANIMCHAR"}
    out = _emit_stmt(CallStmt(target=":next", src=SRC), 0, None, names)
    assert out == ["self.ANIMCHAR();"]


def test_call_target_unresolved_is_escaped():
    # A target with no known routine (raw address / cross-module) keeps
    # its name but is escaped to a valid identifier.
    out = _emit_stmt(CallStmt(target="$c00", src=SRC), 0, None, {})
    assert out == ["self._24c00();"]


def test_routine_fn_name_is_mangled():
    r = _routine([ReturnStmt(src=SRC)], name=":clr")
    assert emit_routine(r)[0] == "    fn _3aclr(&mut self) {"


def test_raw_if_lowers_flag_condition():
    # An unfused branch (`RawIfStmt`) lowers its flag suffix to the
    # provisional flag model. `eq` reads Z; the arms are emitted.
    stmt = RawIfStmt(
        cond="eq",
        then_block=Block.of([ReturnStmt(src=SRC)]),
        else_block=None,
        src=SRC,
    )
    assert _emit_stmt(stmt, 0) == ["if self.z != 0 {", "    return;", "}"]


def test_raw_if_all_tracked_conditions():
    want = {
        "eq": "self.z != 0", "ne": "self.z == 0",
        "cs": "self.c != 0", "cc": "self.c == 0",
        "mi": "self.n != 0", "pl": "self.n == 0",
    }
    for cond, rs in want.items():
        stmt = RawIfStmt(
            cond=cond, then_block=Block.of([ReturnStmt(src=SRC)]),
            else_block=None, src=SRC,
        )
        assert _emit_stmt(stmt, 0)[0] == f"if {rs} {{"


def test_goto_state_lowers_to_pc_assignment():
    assert _emit_one(GotoStateStmt(state=7, src=SRC)) == "pc = 7;"


def test_dispatch_emits_loop_match():
    # A two-state dispatcher: state 0 transitions to 1, state 1 returns.
    dispatch = DispatchStmt(
        entry=0,
        arms=(
            DispatchArm(state=0, body=Block.of([GotoStateStmt(state=1, src=SRC)])),
            DispatchArm(state=1, body=Block.of([ReturnStmt(src=SRC)])),
        ),
        src=SRC,
    )
    assert _emit_stmt(dispatch, 0) == [
        "let mut pc: u32 = 0;",
        "loop {",
        "    match pc {",
        "        0 => {",
        "            pc = 1;",
        "        }",
        "        1 => {",
        "            return;",
        "        }",
        "        _ => unreachable!(),",
        "    }",
        "}",
    ]


def test_dispatch_counts_as_lowered():
    dispatch = DispatchStmt(
        entry=0,
        arms=(DispatchArm(state=0, body=Block.of([ReturnStmt(src=SRC)])),),
        src=SRC,
    )
    lowered, deferred = lower_stats(_module([_routine([dispatch])]))
    assert (lowered, deferred) == (1, 0)


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
        RawStmt(item=CmpImm(reg=Reg.A, imm=_imm(0), src=SRC)),  # lowered (cmp flags)
        RawStmt(item=Pha(src=SRC)),  # lowered (stack)
        RawStmt(item=StoreLocal(  # deferred (SMC operand patch)
            reg=Reg.A, target_label="smXCO", offset=1, src=SRC)),
        CallStmt(target="sub", src=SRC),  # lowered
        ReturnStmt(src=SRC),  # lowered
    ]
    lowered, deferred = lower_stats(_module([_routine(stmts)]))
    assert (lowered, deferred) == (7, 1)


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


def test_raw_load_imm_opvar_reads_operand_var():
    # A recognised SMC operand immediate reads the provisional operand-
    # variable field that the matching StoreOpVar writes.
    smc = Imm(value=0, text="#smXCO", opvar="smXCO")
    out = _raw(LoadImm(reg=Reg.A, imm=smc, src=SRC))
    assert out == "self.a = self.smXCO;"


def test_raw_store_opvar_writes_operand_var():
    out = _raw(StoreOpVar(reg=Reg.A, name="smXCO", src=SRC))
    assert out == "self.smXCO = self.a;"


def test_raw_store_op_addr_writes_byte_halves():
    # The SMC address patch writes the low / high byte of the operand var.
    assert _raw(StoreOpAddr(reg=Reg.A, name="smBASE", half="lo", src=SRC)) == \
        "self.smBASE_lo = self.a;"
    assert _raw(StoreOpAddr(reg=Reg.X, name="smBASE", half="hi", src=SRC)) == \
        "self.smBASE_hi = self.x;"


def test_address_opvar_operand_composes_runtime_base():
    # An `Abs` marked with an address opvar reads its base from the
    # patched low/high byte fields instead of the assembled address.
    base = Abs(name="$2000", addr=0x2000, opvar="smBASE")
    item = LoadIndexed(reg=Reg.A, base=base, index=Reg.Y, src=SRC)
    assert _raw(item) == (
        "self.a = self.ram[((self.smBASE_hi as usize) << 8 "
        "| self.smBASE_lo as usize) + self.y as usize];"
    )


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


def test_raw_sbc_imm_complement_is_byte_width():
    # The immediate complement must be `_u8`: a bare `!0xbd` defaults to
    # i32 (-190) and casts to the wrong u16 (and can overflow-panic).
    lines = _emit_stmt(RawStmt(item=SbcImm(imm=_imm(0xbd), src=SRC)), 0)
    assert lines[0] == "let _r = (self.a as u16) + (!0xbd_u8) as u16 + (self.c as u16);"


def test_raw_sbc_imm_opvar_complement_has_no_u8_suffix():
    # An SMC opvar immediate is already a u8 field; complementing it must
    # be `!self.smXCO`, not the invalid `!self.smXCO_u8`.
    smc = Imm(value=0, text="#smXCO", opvar="smXCO")
    lines = _emit_stmt(RawStmt(item=SbcImm(imm=smc, src=SRC)), 0)
    assert lines[0] == "let _r = (self.a as u16) + (!self.smXCO) as u16 + (self.c as u16);"


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


# ---------------------------------------------------------------- cmp / bit flag lowering


def test_raw_cmp_imm():
    lines = _emit_stmt(RawStmt(item=CmpImm(reg=Reg.A, imm=_imm(0x05), src=SRC)), 0)
    assert lines == [
        "let _o: u8 = 0x05;",
        "self.c = (self.a >= _o) as u8;",
        "self.z = (self.a == _o) as u8;",
        "self.n = self.a.wrapping_sub(_o) >> 7;",
    ]


def test_raw_cmp_uses_compared_register():
    # cpx/cpy compare X/Y, not A.
    lines = _emit_stmt(RawStmt(item=CmpImm(reg=Reg.X, imm=_imm(0x01), src=SRC)), 0)
    assert lines[1] == "self.c = (self.x >= _o) as u8;"
    assert lines[3] == "self.n = self.x.wrapping_sub(_o) >> 7;"


def test_raw_cmp_abs_and_indexed():
    abs_lines = _emit_stmt(RawStmt(item=CmpAbs(reg=Reg.A, source=_abs("M", 0x40), src=SRC)), 0)
    assert abs_lines[0] == "let _o: u8 = self.ram[0x0040];"
    idx = CmpIndexed(reg=Reg.A, base=_abs("tbl", 0x0200), index=Reg.X, src=SRC)
    idx_lines = _emit_stmt(RawStmt(item=idx), 0)
    assert idx_lines[0] == "let _o: u8 = self.ram[0x0200 + self.x as usize];"


def test_raw_cmp_indirect():
    item = CmpIndirect(reg=Reg.A, source=IndirectY(ptr=_abs("ptr", 0x20)), src=SRC)
    lines = _emit_stmt(RawStmt(item=item), 0)
    assert lines[0] == (
        "let _o: u8 = self.ram[(self.ram[0x0020] as usize "
        "| (self.ram[0x0021] as usize) << 8) + self.y as usize];"
    )
    assert lines[1] == "self.c = (self.a >= _o) as u8;"


def test_raw_bit_imm_and_abs():
    imm_lines = _emit_stmt(RawStmt(item=Bit(source=_imm(0x80), src=SRC)), 0)
    assert imm_lines == [
        "let _o: u8 = 0x80;",
        "self.z = ((self.a & _o) == 0) as u8;",
        "self.n = _o >> 7;",
    ]
    abs_lines = _emit_stmt(RawStmt(item=Bit(source=_abs("sw", 0xC010), src=SRC)), 0)
    assert abs_lines[0] == "let _o: u8 = self.ram[0xc010];"


# ---------------------------------------------------------------- 16-bit (Wide16) arithmetic


def _wide16(op):
    # {hi:lo} = {ptr+1:ptr} op {0x33:0x44}  -> dst {dh:dl}
    return Wide16Stmt(
        op=op,
        lo_src=_abs("ptr", 0x20), hi_src=_abs("ptr+1", 0x21),
        lo_op=_imm(0x44), hi_op=_imm(0x33),
        lo_dst=_abs("dl", 0x30), hi_dst=_abs("dh", 0x31),
        src=SRC,
    )


def test_wide16_add():
    lines = _emit_stmt(_wide16("+"), 0)
    assert lines == [
        "let _lo = (self.ram[0x0020] as u16) + (0x44 as u16);",
        "self.ram[0x0030] = _lo as u8;",
        "let _hi = (self.ram[0x0021] as u16) + (0x33 as u16) + (_lo >> 8);",
        "self.ram[0x0031] = _hi as u8;",
        "self.a = _hi as u8;",
        "self.c = (_hi >> 8) as u8;",
    ]


def test_wide16_subtract_uses_complement_identity():
    # Subtract is `src + ~op + 1` (low) / `src + ~op + carry` (high);
    # the immediate operands complement at byte width (`_u8`).
    lines = _emit_stmt(_wide16("-"), 0)
    assert lines[0] == "let _lo = (self.ram[0x0020] as u16) + (!0x44_u8 as u16) + 1;"
    assert lines[2] == "let _hi = (self.ram[0x0021] as u16) + (!0x33_u8 as u16) + (_lo >> 8);"
    assert lines[4:] == ["self.a = _hi as u8;", "self.c = (_hi >> 8) as u8;"]


def test_wide16_memory_operand_not_u8_suffixed():
    # A memory op byte is already u8, so subtract emits `!self.ram[..]`
    # without the `_u8` suffix.
    stmt = Wide16Stmt(
        op="-",
        lo_src=_abs("a", 0x20), hi_src=_abs("a+1", 0x21),
        lo_op=_abs("b", 0x40), hi_op=_abs("b+1", 0x41),
        lo_dst=_abs("d", 0x30), hi_dst=_abs("d+1", 0x31),
        src=SRC,
    )
    lines = _emit_stmt(stmt, 0)
    assert lines[0] == "let _lo = (self.ram[0x0020] as u16) + (!self.ram[0x0040] as u16) + 1;"


def test_wide16_opvar_operand_complement_has_no_u8_suffix():
    # A Wide16 subtract whose operand is an SMC opvar immediate must
    # complement the u8 field directly (`!self.smXCO`), not `!self.smXCO_u8`.
    smc = Imm(value=0, text="#smXCO", opvar="smXCO")
    stmt = Wide16Stmt(
        op="-",
        lo_src=_abs("a", 0x20), hi_src=_abs("a+1", 0x21),
        lo_op=smc, hi_op=_imm(0x33),
        lo_dst=_abs("d", 0x30), hi_dst=_abs("d+1", 0x31),
        src=SRC,
    )
    lines = _emit_stmt(stmt, 0)
    assert lines[0] == "let _lo = (self.ram[0x0020] as u16) + (!self.smXCO as u16) + 1;"


def test_wide16_counts_as_lowered():
    stmts = [_wide16("+"), ReturnStmt(src=SRC)]
    lowered, deferred = lower_stats(_module([_routine(stmts)]))
    assert (lowered, deferred) == (2, 0)


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
