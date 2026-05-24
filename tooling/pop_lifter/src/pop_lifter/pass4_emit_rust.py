"""Pass 4: emit Rust source from IR3.

Earlier slices lowered module / routine scaffolding, leaf expressions
(folded `Assign`s, `return`), and *all structured control flow*
(`IfStmt` / `LoopStmt` / `DoWhileStmt` / `ForStmt` / `RepeatStmt`,
`BreakStmt` / `ContinueStmt`, `MatchStmt`, `CallStmt` / `TailCallStmt`,
`SaveTemp` / `RestoreTemp`).

This slice lowers the *data-movement `RawStmt` atoms* — the unfolded
IR1 instructions pass 3 couldn't collapse into an `Assign`, but which
still map directly onto a register/memory move with no carry- or
flag-flow to model:

* `LoadImm` / `LoadAbs` / `LoadIndexed` → `self.<reg> = <value>;`
* `StoreAbs` / `StoreIndexed` → `self.ram[<addr>] = self.<reg>;`
* `Transfer` (`tax`/`tay`/`txa`/`tya`) → `self.<dst> = self.<src>;`
* `Bitwise` (`and`/`ora`/`eor`) → `self.a &= <value>;` (`|=` / `^=`)
* `IncTarget` / `DecTarget` (reg or memory) → `<place> = <place>.wrapping_add(1);`

Still deferred — these stay as `// raw: …` (or `// TODO(pass4): …`)
comments until their model lands:

* carry/flag-bearing atoms (`Clc`/`Sec`/`AdcImm`/`SbcImm`/`Asl`/`Lsr`/
  `Rol`/`Ror`/`ShiftMem`/`Cmp*`/`Bit`) — need a processor-flag model;
* indirect addressing (`(ptr),y` loads/stores) — needs a pointer fetch;
* self-modifying code (`StoreLocal`/`StoreOpVar`, `LocalRef` inc/dec);
* the stack (`Pha`/`Pla`);
* `Wide16Stmt`, `RawIfStmt`, `GotoStmt` / `LabelStmt`.

Memory model and receiver (`Cpu` / `self.ram`) remain provisional
pending the Game/Renderer/Audio/Input design slice.
"""

from __future__ import annotations

from .ir1 import (
    Abs,
    Bitwise,
    Compare,
    DecTarget,
    Imm,
    IncTarget,
    IndexedAbs,
    IndirectY,
    LoadAbs,
    LoadImm,
    LoadIndexed,
    Reg,
    StoreAbs,
    StoreIndexed,
    Transfer,
)
from .ir3 import (
    Assign,
    BinExpr,
    BreakStmt,
    CallStmt,
    ContinueStmt,
    DoWhileStmt,
    ForStmt,
    IfStmt,
    LoopStmt,
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

INDENT = "    "

# Statement types that produce real Rust code (not a comment placeholder).
_LOWERED_TYPES = (
    Assign, ReturnStmt,
    IfStmt, LoopStmt, DoWhileStmt, ForStmt, RepeatStmt,
    BreakStmt, ContinueStmt, MatchStmt,
    CallStmt, TailCallStmt,
    SaveTemp, RestoreTemp,
)


# ---------------------------------------------------------------- values


def _addr(addr: int) -> str:
    return f"0x{addr:04x}"


def _emit_imm(imm: Imm) -> str:
    # The opvar (self-modifying-code) form names a runtime-patched byte;
    # lowering that to a mutable field is a later slice, so emit the
    # assembled value and leave the opvar intent to the IR3 dump.
    return f"0x{imm.value & 0xff:02x}"


def _emit_value(v) -> str:
    """Render an `Assign` source or compare RHS as a Rust r-value."""
    if isinstance(v, Imm):
        return _emit_imm(v)
    if isinstance(v, Abs):
        return f"self.ram[{_addr(v.addr)}]"
    if isinstance(v, IndexedAbs):
        return f"self.ram[{_addr(v.base.addr)} + self.{v.index} as usize]"
    if isinstance(v, IndirectY):
        # `(ptr),y` needs a 16-bit pointer fetch + Y — deferred.
        return f'todo!("indirect ({v.ptr.name}),y read")'
    if isinstance(v, BinExpr):
        return _emit_binexpr(v)
    if isinstance(v, RotateExpr):
        # rotl/rotr read the carry flag, so they are methods on the CPU.
        return f"self.{v.op}({_emit_value(v.operand)}, {v.count})"
    raise ValueError(f"unknown Assign source type: {type(v).__name__}")


def _emit_binexpr(v: BinExpr) -> str:
    lhs = _emit_value(v.lhs)
    rhs = _emit_value(v.rhs)
    if v.op == "+":
        return f"({lhs}).wrapping_add({rhs})"
    if v.op == "-":
        return f"({lhs}).wrapping_sub({rhs})"
    if v.op == "<<":
        return f"({lhs}).wrapping_shl({rhs} as u32)"
    if v.op == ">>":
        return f"({lhs}).wrapping_shr({rhs} as u32)"
    raise ValueError(f"unknown BinExpr op: {v.op!r}")


def _emit_target(target) -> str | None:
    """Render an `Assign` target as a Rust assignable place, or `None`
    when the destination form isn't lowered yet."""
    if isinstance(target, Abs):
        return f"self.ram[{_addr(target.addr)}]"
    if isinstance(target, IndexedAbs):
        return f"self.ram[{_addr(target.base.addr)} + self.{target.index} as usize]"
    return None  # IndirectY and anything else: deferred


def _emit_compare(c: Compare) -> str:
    """Render a pass-3 `Compare` as a Rust boolean expression.

    * `rhs=None` is a sign test (N-flag): `op` is `">=0"` or `"<0"`.
      Cast the register to `i8` so the signed comparison is natural.
    * `rhs=Imm` with `==`/`!=` is a zero / equality test (unsigned ok).
    * `rhs=Imm` with `<`/`>=` comes from `cmp; bcc/bcs` — unsigned,
      so plain u8 comparison is correct.
    * `rhs=Abs` / `rhs=IndexedAbs`: compare against a memory byte."""
    reg = f"self.{c.reg}"
    if c.rhs is None:
        # Sign test (N-flag): op is ">=0" (bpl) or "<0" (bmi). Validate
        # explicitly so an unexpected op surfaces rather than silently
        # emitting the wrong predicate.
        if c.op == ">=0":
            op_str = ">= 0"
        elif c.op == "<0":
            op_str = "< 0"
        else:
            raise ValueError(f"unexpected sign-test Compare op: {c.op!r}")
        return f"({reg} as i8) {op_str}"
    rhs = _emit_value(c.rhs)
    return f"{reg} {c.op} {rhs}"


# ---------------------------------------------------------------- raw atoms


_BITWISE_OPS = {"and": "&", "or": "|", "eor": "^"}


def _emit_raw(item) -> list[str] | None:
    """Lower an unfolded IR1 data-movement atom to Rust statement
    fragments (no indentation), or return `None` when the atom isn't
    lowered in this slice so the caller falls back to a `// raw:`
    comment.

    Covers the carry-/flag-free register and memory moves. Carry- or
    flag-bearing atoms (adc/sbc/shifts/cmp/bit/clc/sec), indirect
    addressing, self-modifying code, and the stack stay deferred."""
    if isinstance(item, LoadImm):
        # An opvar immediate is a runtime-patched SMC byte; lowering it
        # to its assembled value would be wrong, so defer the whole load.
        if item.imm.opvar is not None:
            return None
        return [f"self.{item.reg} = {_emit_imm(item.imm)};"]

    if isinstance(item, LoadAbs):
        return [f"self.{item.reg} = self.ram[{_addr(item.source.addr)}];"]

    if isinstance(item, LoadIndexed):
        place = f"self.ram[{_addr(item.base.addr)} + self.{item.index} as usize]"
        return [f"self.{item.reg} = {place};"]

    if isinstance(item, StoreAbs):
        return [f"self.ram[{_addr(item.target.addr)}] = self.{item.reg};"]

    if isinstance(item, StoreIndexed):
        place = f"self.ram[{_addr(item.base.addr)} + self.{item.index} as usize]"
        return [f"{place} = self.{item.reg};"]

    if isinstance(item, Transfer):
        return [f"self.{item.dst_reg} = self.{item.src_reg};"]

    if isinstance(item, Bitwise):
        op = _BITWISE_OPS.get(item.op)
        # `(ptr),y` sources need a pointer fetch — deferred with the rest
        # of indirect addressing.
        if op is None or not isinstance(item.source, (Imm, Abs, IndexedAbs)):
            return None
        return [f"self.a {op}= {_emit_value(item.source)};"]

    if isinstance(item, (IncTarget, DecTarget)):
        method = "wrapping_add" if isinstance(item, IncTarget) else "wrapping_sub"
        target = item.target
        if isinstance(target, Reg):
            return [f"self.{target} = self.{target}.{method}(1);"]
        if isinstance(target, Abs):
            place = f"self.ram[{_addr(target.addr)}]"
            return [f"{place} = {place}.{method}(1);"]
        return None  # LocalRef: self-modifying-code operand bump — deferred

    return None


# ---------------------------------------------------------------- statements


def _emit_stmt(stmt, indent: int) -> list[str]:
    pad = INDENT * indent

    if isinstance(stmt, Assign):
        place = _emit_target(stmt.target)
        if place is None:
            return [f"{pad}// TODO(pass4): store via {type(stmt.target).__name__}"]
        return [f"{pad}{place} = {_emit_value(stmt.source)};"]

    if isinstance(stmt, ReturnStmt):
        return [f"{pad}return;"]

    if isinstance(stmt, BreakStmt):
        return [f"{pad}break;"]

    if isinstance(stmt, ContinueStmt):
        return [f"{pad}continue;"]

    if isinstance(stmt, CallStmt):
        return [f"{pad}self.{stmt.target}();"]

    if isinstance(stmt, TailCallStmt):
        return [f"{pad}self.{stmt.target}();", f"{pad}return;"]

    if isinstance(stmt, SaveTemp):
        return [f"{pad}let tmp{stmt.slot} = self.a;"]

    if isinstance(stmt, RestoreTemp):
        return [f"{pad}self.a = tmp{stmt.slot};"]

    if isinstance(stmt, IfStmt):
        lines = [f"{pad}if {_emit_compare(stmt.cond)} {{"]
        for s in stmt.then_block.stmts:
            lines.extend(_emit_stmt(s, indent + 1))
        if stmt.else_block is not None:
            lines.append(f"{pad}}} else {{")
            for s in stmt.else_block.stmts:
                lines.extend(_emit_stmt(s, indent + 1))
        lines.append(f"{pad}}}")
        return lines

    if isinstance(stmt, LoopStmt):
        lines = [f"{pad}loop {{"]
        for s in stmt.body.stmts:
            lines.extend(_emit_stmt(s, indent + 1))
        lines.append(f"{pad}}}")
        return lines

    if isinstance(stmt, DoWhileStmt):
        cond = _emit_compare(stmt.cond)
        inner = INDENT * (indent + 1)
        lines = [f"{pad}loop {{"]
        for s in stmt.body.stmts:
            lines.extend(_emit_stmt(s, indent + 1))
        lines += [
            f"{inner}if !({cond}) {{",
            f"{inner}    break;",
            f"{inner}}}",
            f"{pad}}}",
        ]
        return lines

    if isinstance(stmt, ForStmt):
        step_method = "wrapping_sub" if stmt.step < 0 else "wrapping_add"
        step_lit = f"0x{abs(stmt.step) & 0xff:02x}"
        cond = _emit_compare(stmt.cond)
        inner = INDENT * (indent + 1)
        lines = [
            f"{pad}self.{stmt.var} = {_emit_imm(stmt.start)};",
            f"{pad}loop {{",
        ]
        for s in stmt.body.stmts:
            lines.extend(_emit_stmt(s, indent + 1))
        lines += [
            f"{inner}self.{stmt.var} = self.{stmt.var}.{step_method}({step_lit});",
            f"{inner}if !({cond}) {{",
            f"{inner}    break;",
            f"{inner}}}",
            f"{pad}}}",
        ]
        return lines

    if isinstance(stmt, RepeatStmt):
        step_method = "wrapping_sub" if stmt.step < 0 else "wrapping_add"
        step_lit = f"0x{abs(stmt.step) & 0xff:02x}"
        inner = INDENT * (indent + 1)
        lines = [
            f"{pad}self.{stmt.var} = {_emit_imm(stmt.start)};",
            f"{pad}for _ in 0..{stmt.count}usize {{",
        ]
        for s in stmt.body.stmts:
            lines.extend(_emit_stmt(s, indent + 1))
        lines += [
            f"{inner}self.{stmt.var} = self.{stmt.var}.{step_method}({step_lit});",
            f"{pad}}}",
        ]
        return lines

    if isinstance(stmt, MatchStmt):
        lines = [f"{pad}match self.{stmt.reg} {{"]
        for arm in stmt.arms:
            vals = " | ".join(f"0x{v.value & 0xff:02x}" for v in arm.values)
            lines.append(f"{pad}    {vals} => {{")
            for s in arm.body.stmts:
                lines.extend(_emit_stmt(s, indent + 2))
            lines.append(f"{pad}    }}")
        lines += [f"{pad}    _ => {{}}", f"{pad}}}"]
        return lines

    if isinstance(stmt, RawStmt):
        lowered = _emit_raw(stmt.item)
        if lowered is not None:
            return [f"{pad}{line}" for line in lowered]
        from .ir1 import format_item
        return [f"{pad}// raw: {format_item(stmt.item).strip()}"]

    return [f"{pad}// TODO(pass4): lower {type(stmt).__name__}"]


# ---------------------------------------------------------------- routines / module


def emit_routine(routine: RoutineIR3, indent: int = 1) -> list[str]:
    pad = INDENT * indent
    lines: list[str] = []
    if routine.entry_aliases:
        lines.append(f"{pad}// aliases: {', '.join(routine.entry_aliases)}")
    lines.append(f"{pad}fn {routine.name}(&mut self) {{")
    for s in routine.body.stmts:
        lines.extend(_emit_stmt(s, indent + 1))
    lines.append(f"{pad}}}")
    return lines


def emit_module(module: ModuleIR3) -> str:
    from .ir1 import _portable_path
    lines = [
        "// @generated by pop_lifter — DO NOT EDIT.",
        "//",
        "// Pass 4 skeleton slice: module + routine scaffolding with leaf-",
        "// expression, control-flow, and data-movement `RawStmt` lowering.",
        "// Carry/flag-bearing atoms, indirect addressing, self-modifying",
        "// code, the stack, `Wide16Stmt`, `RawIfStmt`, and `GotoStmt`/",
        "// `LabelStmt` are deferred to later slices; they appear as",
        "// `// TODO(pass4): …` or `// raw: …` comments.",
        "// The `Cpu` receiver and flat `self.ram` model are provisional,",
        "// pending the state/trait design slice.",
        "//",
        f"// source: {_portable_path(module.file)}",
        "",
        "impl Cpu {",
    ]
    for i, routine in enumerate(module.routines):
        if i:
            lines.append("")
        lines.extend(emit_routine(routine, indent=1))
    lines.append("}")
    return "\n".join(lines) + "\n"


def lower_stats(module: ModuleIR3) -> tuple[int, int]:
    """Count top-level statements that produce real Rust code vs. those
    that produce a comment placeholder. Nested blocks under control-flow
    are not counted — only the top-level flat list of each routine."""
    lowered = deferred = 0
    for routine in module.routines:
        for s in routine.body.stmts:
            if isinstance(s, RawStmt):
                # A raw atom counts as lowered only when this slice
                # produces real Rust for it (data-movement atoms).
                if _emit_raw(s.item) is not None:
                    lowered += 1
                else:
                    deferred += 1
            elif isinstance(s, _LOWERED_TYPES):
                lowered += 1
            else:
                deferred += 1
    return lowered, deferred
