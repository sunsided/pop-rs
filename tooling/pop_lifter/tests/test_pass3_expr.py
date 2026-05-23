"""Pass 3 — accumulator/register expression folding.

Three flavours of test:

* **Synthetic unit tests** drive the folder (`_fold_block` /
  `fold_routine`) over hand-built IR3 to pin the fold rules: copy folds
  (A/X/Y), arithmetic folds (`clc;adc` / `sec;sbc`), the carry- and
  register-liveness gates, and the interprocedural register demand that
  lets a copy fold across a `jmp`/`jsr` when the target doesn't read the
  register.
* **Behavioural equivalence** interprets hand-built and real routines
  (`chgshadposn`, `Cup`, plus synthetic caller/target pairs) before and
  after folding and asserts byte-identical RAM — the fold must not
  change observable behaviour.
"""

from __future__ import annotations

from pathlib import Path

from pop_lifter.interp_ir3 import run as ir3_run
from pop_lifter.ir1 import (
    Abs,
    AdcAbs,
    AdcImm,
    Clc,
    Compare,
    Imm,
    IndexedAbs,
    LoadAbs,
    LoadImm,
    ModuleIR1,
    Reg,
    Return,
    Routine,
    SbcImm,
    Sec,
    SourceRef,
    StoreAbs,
    Transfer,
)
from pop_lifter.ir3 import (
    Assign,
    BinExpr,
    Block,
    BreakStmt,
    CallStmt,
    IfStmt,
    LoopStmt,
    ModuleIR3,
    RawIfStmt,
    RawStmt,
    ReturnStmt,
    RoutineIR3,
    TailCallStmt,
)
from pop_lifter import ir3 as ir3_mod
from pop_lifter.pass0_parse import parse_files
from pop_lifter.pass1_lift import discover_entries, lift_file
from pop_lifter.pass2_reloop import reloop_module
from pop_lifter.pass2_struct import structure_module
from pop_lifter.pass3_expr import fold_module, fold_routine, fold_stats

SRC = SourceRef(file="syn", line=0, raw="")


def _raw(item) -> RawStmt:
    return RawStmt(item=item)


def _load_a_abs(name: str, addr: int) -> RawStmt:
    return _raw(LoadAbs(reg=Reg.A, source=Abs(name=name, addr=addr), src=SRC))


def _store_a_abs(name: str, addr: int) -> RawStmt:
    return _raw(StoreAbs(reg=Reg.A, target=Abs(name=name, addr=addr), src=SRC))


def _kill_a() -> RawStmt:
    """`lda #0` — a write that redefines A (and Z/N), proving the prior
    A value dead."""
    return _raw(LoadImm(reg=Reg.A, imm=Imm(value=0, text="#0"), src=SRC))


def _fold(stmts: list) -> list:
    routine = RoutineIR3(name="syn", body=Block.of(stmts))
    return list(fold_routine(routine).body.stmts)


# --------------------------------------------------------------- synthetic


def test_simple_copy_folds_when_a_dead():
    """`a = SRC ; *DST = a ; a = #0` — the round-trip collapses to
    `*DST = SRC`, dropping the load, because the trailing `a = #0`
    redefines A before any read."""
    out = _fold([_load_a_abs("SRC", 0x10), _store_a_abs("DST", 0x20), _kill_a()])
    assert len(out) == 2
    assign, tail = out
    assert isinstance(assign, Assign)
    assert isinstance(assign.target, Abs) and assign.target.addr == 0x20
    assert isinstance(assign.source, Abs) and assign.source.addr == 0x10
    # The reassigning load survives untouched (nothing to fold into it).
    assert isinstance(tail, RawStmt) and isinstance(tail.item, LoadImm)


def test_constant_multi_store_folds_to_one_assign_each():
    """`a = #1 ; *X = a ; *Y = a ; a = #0` ⇒ `*X = #1 ; *Y = #1`."""
    out = _fold([
        _raw(LoadImm(reg=Reg.A, imm=Imm(value=1, text="#1"), src=SRC)),
        _store_a_abs("X", 0x30),
        _store_a_abs("Y", 0x31),
        _kill_a(),
    ])
    assert len(out) == 3
    a0, a1, tail = out
    assert isinstance(a0, Assign) and a0.target.addr == 0x30
    assert isinstance(a1, Assign) and a1.target.addr == 0x31
    for a in (a0, a1):
        assert isinstance(a.source, Imm) and a.source.value == 1
    assert isinstance(tail, RawStmt)


def test_no_fold_when_a_live_via_return():
    """`a = SRC ; *DST = a ; return` — A might be a return value, so the
    load is NOT dropped."""
    out = _fold([_load_a_abs("SRC", 0x10), _store_a_abs("DST", 0x20), ReturnStmt(src=SRC)])
    assert [type(s) for s in out] == [RawStmt, RawStmt, ReturnStmt]
    assert not any(isinstance(s, Assign) for s in out)


def test_no_fold_when_a_live_via_call():
    """A trailing `call` may pass A as an argument — conservatively a
    read, so no fold."""
    out = _fold([
        _load_a_abs("SRC", 0x10),
        _store_a_abs("DST", 0x20),
        CallStmt(target="callee", src=SRC),
        _kill_a(),
    ])
    assert not any(isinstance(s, Assign) for s in out)


def test_transfer_source_is_not_foldable():
    """`txa ; *DST = a ; a = #0` — a `Transfer` writes A but its source
    isn't a standalone copyable value, so the store stays a raw store."""
    out = _fold([
        _raw(Transfer(src_reg=Reg.X, dst_reg=Reg.A, src=SRC)),
        _store_a_abs("DST", 0x20),
        _kill_a(),
    ])
    assert not any(isinstance(s, Assign) for s in out)


def test_load_with_no_following_store_is_left_alone():
    """A bare `a = SRC` with no store after it has nothing to fold."""
    out = _fold([_load_a_abs("SRC", 0x10), _kill_a()])
    assert not any(isinstance(s, Assign) for s in out)
    assert len(out) == 2


def test_no_fold_across_unsupported_opcode():
    """An `Unsupported` opcode has unknown semantics — it may read A or
    its flags — so liveness must not step past it. `a = SRC ; *DST = a ;
    ??? ; a = #0` stays unfolded. (Reviewer #20.)"""
    from pop_lifter.ir1 import Unsupported
    out = _fold([
        _load_a_abs("SRC", 0x10),
        _store_a_abs("DST", 0x20),
        _raw(Unsupported(mnemonic="wat", operand=None, src=SRC)),
        _kill_a(),
    ])
    assert not any(isinstance(s, Assign) for s in out)


def _if_y(then_stmts: list, cond_reg: Reg = Reg.Y) -> IfStmt:
    return IfStmt(
        cond=Compare(reg=cond_reg, op="==", rhs=Imm(value=0, text="#0")),
        then_block=Block.of(then_stmts),
        else_block=None,
        src=SRC,
    )


def test_no_fold_when_nested_if_can_return():
    """`a = SRC ; *DST = a ; if y == 0 { return } ; a = #0` must NOT
    fold: on the `y == 0` path control returns with A still holding the
    loaded value. The earlier `_stmt_touches_a` scan stepped past this
    `if` because its body 'doesn't touch A'. (Reviewer #20.)"""
    out = _fold([
        _load_a_abs("SRC", 0x10),
        _store_a_abs("DST", 0x20),
        _if_y([ReturnStmt(src=SRC)]),
        _kill_a(),
    ])
    assert not any(isinstance(s, Assign) for s in out)


def test_no_fold_when_nested_if_holds_raw_flag_branch():
    """A `RawIfStmt` nested inside an `if` body reads the Z/N flags the
    dropped load set — folding would change behaviour. The scan must
    not step past the outer `if` just because its body doesn't touch A.
    (Reviewer #20.)"""
    inner = RawIfStmt(
        cond="eq",
        then_block=Block.of([
            # stx — touches X/memory, not A — so the body 'doesn't touch A'.
            _raw(StoreAbs(reg=Reg.X, target=Abs(name="T", addr=0x40), src=SRC)),
        ]),
        else_block=None,
        src=SRC,
    )
    out = _fold([
        _load_a_abs("SRC", 0x10),
        _store_a_abs("DST", 0x20),
        _if_y([inner]),
        _kill_a(),
    ])
    assert not any(isinstance(s, Assign) for s in out)


def _loop_copy_routine(post_loop: list) -> RoutineIR3:
    """A chgshadposn-shaped loop: the copy run sits at the top of the
    body, the exit guard (`if ... { break }`) at the bottom. `post_loop`
    is spliced in after the loop — it decides whether the break path
    leaves A dead."""
    loop = LoopStmt(
        body=Block.of([
            _load_a_abs("SRC", 0x10),
            _store_a_abs("DST", 0x20),
            _if_y([BreakStmt(src=SRC)]),
        ]),
        src=SRC,
    )
    return RoutineIR3(name="syn", body=Block.of([loop, *post_loop]))


def test_loop_copy_folds_when_break_target_kills_a():
    """Break exits to code that overwrites A before reading it (`a =
    #0`), so the loop-body copy is dead on every path and folds."""
    routine = _loop_copy_routine(post_loop=[_kill_a(), ReturnStmt(src=SRC)])
    loop = fold_routine(routine).body.stmts[0]
    assert isinstance(loop, LoopStmt)
    assert any(isinstance(s, Assign) for s in loop.body.stmts)


def test_loop_copy_not_folded_when_break_target_reads_a():
    """If the post-loop code reads A before overwriting it (here a store
    of A), the loaded value escapes via the break path — no fold."""
    routine = _loop_copy_routine(
        post_loop=[_store_a_abs("OUT", 0x50), ReturnStmt(src=SRC)]
    )
    loop = fold_routine(routine).body.stmts[0]
    assert isinstance(loop, LoopStmt)
    assert not any(isinstance(s, Assign) for s in loop.body.stmts)


# --------------------------------------------------------------- arithmetic (slice 2)


def _clc() -> RawStmt:
    return _raw(Clc(src=SRC))


def _sec() -> RawStmt:
    return _raw(Sec(src=SRC))


def _adc_imm(v: int) -> RawStmt:
    return _raw(AdcImm(imm=Imm(value=v, text=f"#{v}"), src=SRC))


def _sbc_imm(v: int) -> RawStmt:
    return _raw(SbcImm(imm=Imm(value=v, text=f"#{v}"), src=SRC))


# A dead-after tail that also redefines carry (`lda #0 ; clc`) — an
# arithmetic fold needs *both* A and the carry the add/sub set to be
# dead, and a bare `lda #0` leaves carry live at the routine's
# fall-through.
def _kill_ac() -> list:
    return [_kill_a(), _clc()]


def test_arith_add_imm_folds():
    """`a = X ; clc ; adc #8 ; *Y = a ; ...` ⇒ `*Y = X + #8`."""
    out = _fold([
        _load_a_abs("X", 0x10), _clc(), _adc_imm(8),
        _store_a_abs("Y", 0x20), *_kill_ac(),
    ])
    assigns = [s for s in out if isinstance(s, Assign)]
    assert len(assigns) == 1
    expr = assigns[0].source
    assert isinstance(expr, BinExpr) and expr.op == "+"
    assert isinstance(expr.lhs, Abs) and expr.lhs.addr == 0x10
    assert isinstance(expr.rhs, Imm) and expr.rhs.value == 8
    assert assigns[0].target.addr == 0x20


def test_arith_sub_imm_folds():
    """`a = X ; sec ; sbc #8 ; *Y = a ; ...` ⇒ `*Y = X - #8`."""
    out = _fold([
        _load_a_abs("X", 0x10), _sec(), _sbc_imm(8),
        _store_a_abs("Y", 0x20), *_kill_ac(),
    ])
    assigns = [s for s in out if isinstance(s, Assign)]
    assert len(assigns) == 1
    assert isinstance(assigns[0].source, BinExpr) and assigns[0].source.op == "-"


def test_arith_memory_operand_folds():
    """`a = X ; clc ; adc Z ; *Y = a ; ...` ⇒ `*Y = X + Z`."""
    out = _fold([
        _load_a_abs("X", 0x10),
        _clc(),
        _raw(AdcAbs(source=Abs(name="Z", addr=0x30), src=SRC)),
        _store_a_abs("Y", 0x20),
        *_kill_ac(),
    ])
    assigns = [s for s in out if isinstance(s, Assign)]
    assert len(assigns) == 1
    expr = assigns[0].source
    assert isinstance(expr, BinExpr) and isinstance(expr.rhs, Abs)
    assert expr.rhs.addr == 0x30


def test_no_arith_fold_when_carry_live_at_return():
    """Even with A dead, if the carry the add set is never redefined
    before the routine falls through to `return`, the fold is refused
    (carry could be a return value)."""
    out = _fold([
        _load_a_abs("X", 0x10), _clc(), _adc_imm(8),
        _store_a_abs("Y", 0x20), _kill_a(),  # kills A but NOT carry
    ])
    assert not any(isinstance(s, Assign) for s in out)


def test_no_arith_fold_without_carry_setup():
    """A bare `adc` with no preceding `clc` depends on the incoming
    carry, so it can't be a pure `+` — left alone. A and carry are
    both dead afterwards (`*_kill_ac()`), so the missing `clc` is the
    *only* reason the fold is refused — a regression that dropped the
    carry-set-up requirement would make this fold and fail the test."""
    out = _fold([
        _load_a_abs("X", 0x10), _adc_imm(8), _store_a_abs("Y", 0x20), *_kill_ac(),
    ])
    assert not any(isinstance(s, Assign) for s in out)


def test_no_arith_fold_when_carry_live():
    """16-bit add idiom: the low byte's `adc` sets a carry the high
    byte's `adc` reads. Folding the low add would drop that carry —
    so the carry-liveness check must block it."""
    out = _fold([
        _load_a_abs("lo", 0x10), _clc(), _adc_imm(8), _store_a_abs("lo", 0x10),
        _load_a_abs("hi", 0x11), _adc_imm(0), _store_a_abs("hi", 0x11),
        _kill_a(),
    ])
    assert not any(isinstance(s, Assign) for s in out)


def test_no_arith_fold_multi_store():
    """`a = X ; clc ; adc #8 ; *Y = a ; *Z = a` — multi-store arithmetic
    isn't an idempotent write-back (a target could alias the operand),
    so the fold is refused. A and carry are both dead afterwards
    (`*_kill_ac()`), so the multi-store restriction is the *only*
    blocker — re-enabling multi-store arithmetic would fail this test."""
    out = _fold([
        _load_a_abs("X", 0x10), _clc(), _adc_imm(8),
        _store_a_abs("Y", 0x20), _store_a_abs("Z", 0x21), *_kill_ac(),
    ])
    assert not any(isinstance(s, Assign) for s in out)


# --------------------------------------------------------------- chgshadposn


def _chgshadposn_modules(source_dir: Path):
    """Lift chgshadposn through pass 1 + 2 + reloop, returning both the
    pre-fold IR3 module and the folded one."""
    ast = parse_files(
        [source_dir / "EQ.S", source_dir / "GAMEEQ.S", source_dir / "AUTO.S"],
        search_paths=[source_dir],
    )
    auto = next(f for f in ast.files if Path(f.path).name == "AUTO.S")
    ir1 = lift_file(auto, ast.symbols(), ["chgshadposn"]).module
    ir3 = reloop_module(structure_module(ir1))
    folded = fold_module(ir3)
    return ir3, folded


def _jumpseq_stub() -> ModuleIR1:
    """chgshadposn ends with `call jumpseq`. Stub it as an IR1 routine
    that records the accumulator it was handed (to 0x300) so the
    differential test also pins that A reaches the call unchanged."""
    body = [
        StoreAbs(reg=Reg.A, target=Abs(name="mark", addr=0x300), src=SRC),
        Return(src=SRC),
    ]
    return ModuleIR1(name="STUB", file="syn", routines=[Routine(name="jumpseq", body=body)])


def test_chgshadposn_fold_is_behaviour_preserving(source_dir):
    """Interpret the relooped IR3 and the folded IR3 from identical RAM;
    they must end byte-for-byte identical. This is the soundness gate on
    the fold."""
    ir3, folded = _chgshadposn_modules(source_dir)
    stub = _jumpseq_stub()

    def seed() -> bytearray:
        ram = bytearray(0x10000)
        # a = x = 0 at entry, so ztemp ends up pointing at 0x0000; the
        # loop copies mem[0x00..0x06] into Char[0..6]. Seed a pattern.
        for i in range(8):
            ram[i] = 0x10 + i
        return ram

    ram_pre = seed()
    ir3_run([ir3, stub], "chgshadposn", ram=ram_pre)

    ram_post = seed()
    ir3_run([folded, stub], "chgshadposn", ram=ram_post)

    assert ram_pre == ram_post, "fold changed observable RAM — unsound"


def test_chgshadposn_fold_structure(source_dir):
    """Pin the headline folds: the loop body becomes a single
    `Assign` (`*(Char + y) = *(ztemp)[y]`), a `CharID = #1` Assign
    appears at top level, and the `PlayCount` store before the
    `return` is NOT folded (A may be a return value)."""
    _, folded = _chgshadposn_modules(source_dir)
    routine = folded.find("chgshadposn")
    assert routine is not None

    # Exactly two folds across the routine: the loop copy + CharID.
    assert fold_stats(folded) == 2

    from pop_lifter.ir3 import LoopStmt

    loop = next(s for s in routine.body.stmts if isinstance(s, LoopStmt))
    loop_assigns = [s for s in loop.body.stmts if isinstance(s, Assign)]
    assert len(loop_assigns) == 1
    copy = loop_assigns[0]
    assert isinstance(copy.target, IndexedAbs) and copy.target.base.name == "Char"

    # Top-level CharID = #1 fold.
    top_assigns = [s for s in routine.body.stmts if isinstance(s, Assign)]
    assert any(
        isinstance(a.target, Abs) and a.target.name == "CharID"
        and isinstance(a.source, Imm) and a.source.value == 1
        for a in top_assigns
    )

    # The PlayCount store stays a raw store (blocked by the return).
    assert not any(
        isinstance(a.target, Abs) and a.target.name == "PlayCount"
        for a in top_assigns
    )


# --------------------------------------------------------------- Cup (arithmetic)


def _cup_modules(source_dir: Path):
    ast = parse_files(
        [source_dir / "EQ.S", source_dir / "GAMEEQ.S", source_dir / "AUTO.S"],
        search_paths=[source_dir],
    )
    auto = next(f for f in ast.files if Path(f.path).name == "AUTO.S")
    ir1 = lift_file(auto, ast.symbols(), ["Cup"]).module
    ir3 = reloop_module(structure_module(ir1))
    return ir3, fold_module(ir3)


def _getup_stub() -> ModuleIR1:
    """Cup opens with `a = CharScrn ; jsr getup ; sta CharScrn`. Stub
    getup to rewrite A (here `a = #0x55`) so the differential test
    exercises a non-trivial accumulator value flowing through the call."""
    body = [
        LoadImm(reg=Reg.A, imm=Imm(value=0x55, text="#$55"), src=SRC),
        Return(src=SRC),
    ]
    return ModuleIR1(name="STUB", file="syn", routines=[Routine(name="getup", body=body)])


def test_cup_fold_is_behaviour_preserving(source_dir):
    """Interpret Cup's relooped IR3 and folded IR3 (with the `CharBlockY
    += 3` arithmetic fold) from identical RAM — they must end
    byte-for-byte identical."""
    ir3, folded = _cup_modules(source_dir)
    stub = _getup_stub()

    def seed() -> bytearray:
        ram = bytearray(0x10000)
        ram[0x004b] = 0x12  # CharScrn
        ram[0x0045] = 0x40  # CharBlockY (+3 -> 0x43)
        ram[0x0042] = 0x90  # CharY      (+0xbd -> wraps)
        return ram

    ram_pre = seed()
    ir3_run([ir3, stub], "Cup", ram=ram_pre)
    ram_post = seed()
    ir3_run([folded, stub], "Cup", ram=ram_post)

    assert ram_pre == ram_post, "arithmetic fold changed observable RAM — unsound"
    # Sanity: the fold actually computed CharBlockY + 3.
    assert ram_post[0x0045] == 0x43


def test_cup_fold_structure(source_dir):
    """`CharBlockY = CharBlockY + #3` folds; the `CharScrn` copy (call
    between load and store) and the `CharY += #0xbd` (A live at the
    return) stay unfolded — exactly one fold."""
    _, folded = _cup_modules(source_dir)
    routine = folded.find("Cup")
    assert fold_stats(folded) == 1
    assign = next(s for s in routine.body.stmts if isinstance(s, Assign))
    assert isinstance(assign.source, BinExpr) and assign.source.op == "+"
    assert assign.target.name == "CharBlockY"
    assert assign.source.lhs.name == "CharBlockY"
    assert isinstance(assign.source.rhs, Imm) and assign.source.rhs.value == 3


# --------------------------------------------------------------- interprocedural demand


def _R(it) -> RawStmt:
    return RawStmt(item=it)


def _routine(name, stmts) -> RoutineIR3:
    return RoutineIR3(name=name, body=Block.of(stmts))


def _ldimm(reg, v):
    return _R(LoadImm(reg=reg, imm=Imm(value=v, text=f"#{v}"), src=SRC))


def _sta(reg, name, addr):
    return _R(StoreAbs(reg=reg, target=Abs(name=name, addr=addr), src=SRC))


def _diff_ram(unfolded: ModuleIR3, folded: ModuleIR3, entry: str):
    """Run `entry` in both modules from a zeroed RAM; return the two
    RAM images for an equality assertion."""
    from pop_lifter.interp_ir3 import run as ir3_run
    r1 = bytearray(0x10000)
    ir3_run([unfolded], entry, ram=r1)
    r2 = bytearray(0x10000)
    ir3_run([folded], entry, ram=r2)
    return r1, r2


def test_demand_tailcall_to_non_a_reader_folds():
    """`a = #0x42 ; *DST = a ; jmp target` folds to `*DST = #0x42` when
    `target` overwrites A before reading it (doesn't demand A). The
    SkelProg→GuardProg shape. Differential: identical RAM."""
    caller = _routine("caller", [
        _ldimm(Reg.A, 0x42), _sta(Reg.A, "DST", 0x300),
        TailCallStmt(target="target", src=SRC),
    ])
    target = _routine("target", [
        _ldimm(Reg.A, 0x99), _sta(Reg.A, "MARK", 0x301), ReturnStmt(src=SRC),
    ])
    mod = ModuleIR3(name="M", file="syn", routines=[caller, target])
    folded = fold_module(mod)
    cbody = folded.find("caller").body.stmts
    assert isinstance(cbody[0], Assign) and cbody[0].target.addr == 0x300
    assert isinstance(cbody[0].source, Imm) and cbody[0].source.value == 0x42
    r1, r2 = _diff_ram(mod, folded, "caller")
    assert r1 == r2 and r2[0x300] == 0x42 and r2[0x301] == 0x99


def test_demand_tailcall_to_a_reader_blocks_fold():
    """If `target` reads A before writing it (here `sta MARK` first), it
    demands A — so `a = #0x42 ; *DST = a ; jmp target` must NOT fold; the
    accumulator carries the value into the tail call."""
    caller = _routine("caller", [
        _ldimm(Reg.A, 0x42), _sta(Reg.A, "DST", 0x300),
        TailCallStmt(target="target", src=SRC),
    ])
    target = _routine("target", [
        _sta(Reg.A, "MARK", 0x301), ReturnStmt(src=SRC),  # reads A first
    ])
    mod = ModuleIR3(name="M", file="syn", routines=[caller, target])
    folded = fold_module(mod)
    assert not any(isinstance(s, Assign) for s in folded.find("caller").body.stmts)


def test_demand_tailcall_to_preserving_routine_blocks_fold():
    """A target that *preserves* A (never touches it) and returns lets
    the value escape to the original caller — so `a = #0x42 ; *DST = a ;
    jmp target` must NOT fold even though `target` doesn't *read* A.
    "Doesn't read before write" is not "must clobber before return", so
    the escape-aware `live` demand keeps A live here. (Reviewer #22.)"""
    caller = _routine("caller", [
        _ldimm(Reg.A, 0x42), _sta(Reg.A, "DST", 0x300),
        TailCallStmt(target="target", src=SRC),
    ])
    target = _routine("target", [
        _ldimm(Reg.X, 0x01), ReturnStmt(src=SRC),  # preserves A entirely
    ])
    mod = ModuleIR3(name="M", file="syn", routines=[caller, target])
    folded = fold_module(mod)
    # The `lda #0x42` survives — A escapes through target's return.
    assert not any(isinstance(s, Assign) for s in folded.find("caller").body.stmts)


def test_demand_tailcall_to_clobbering_routine_folds():
    """The dual: a target whose first act on A is a *write* (here it
    loads A) must-clobbers A before any return, so the value is provably
    dead — `a = #0x42 ; *DST = a ; jmp target` folds. This is the
    SkelProg/GuardProg shape, with the soundness condition made precise."""
    caller = _routine("caller", [
        _ldimm(Reg.A, 0x42), _sta(Reg.A, "DST", 0x300),
        TailCallStmt(target="target", src=SRC),
    ])
    target = _routine("target", [
        _R(LoadAbs(reg=Reg.A, source=Abs(name="OTHER", addr=0x301), src=SRC)),
        _sta(Reg.A, "MARK", 0x302), ReturnStmt(src=SRC),
    ])
    mod = ModuleIR3(name="M", file="syn", routines=[caller, target])
    folded = fold_module(mod)
    cbody = folded.find("caller").body.stmts
    assert isinstance(cbody[0], Assign) and cbody[0].source.value == 0x42


def test_demand_unknown_target_is_conservative():
    """A tail call to a target not in the module (cross-module / unknown)
    defaults to demanding every register — no fold."""
    caller = _routine("caller", [
        _ldimm(Reg.A, 0x42), _sta(Reg.A, "DST", 0x300),
        TailCallStmt(target="elsewhere", src=SRC),
    ])
    mod = ModuleIR3(name="M", file="syn", routines=[caller])
    folded = fold_module(mod)
    assert not any(isinstance(s, Assign) for s in folded.find("caller").body.stmts)


def test_demand_x_copy_across_tailcall_folds():
    """`x = #0x07 ; *DST = x ; jmp target` folds when `target` overwrites
    X before reading it. The X/Y generalisation, unblocked by demand."""
    caller = _routine("caller", [
        _ldimm(Reg.X, 0x07), _sta(Reg.X, "DST", 0x300),
        TailCallStmt(target="target", src=SRC),
    ])
    target = _routine("target", [
        _ldimm(Reg.X, 0x01), _sta(Reg.X, "OUT", 0x302), ReturnStmt(src=SRC),
    ])
    mod = ModuleIR3(name="M", file="syn", routines=[caller, target])
    folded = fold_module(mod)
    cbody = folded.find("caller").body.stmts
    assert isinstance(cbody[0], Assign) and cbody[0].source.value == 0x07
    r1, r2 = _diff_ram(mod, folded, "caller")
    assert r1 == r2 and r2[0x300] == 0x07


def test_demand_call_then_reassign_folds():
    """Non-tail `call` to a non-A-reader, then A reassigned: `a = #1 ;
    *DST = a ; jsr target ; a = #0 ; *DST2 = a` folds the first copy
    (A dead — target doesn't read it, then `a = #0` overwrites)."""
    caller = _routine("caller", [
        _ldimm(Reg.A, 1), _sta(Reg.A, "DST", 0x300),
        CallStmt(target="target", src=SRC),
        _ldimm(Reg.A, 0), _sta(Reg.A, "DST2", 0x302),
        ReturnStmt(src=SRC),
    ])
    target = _routine("target", [
        _ldimm(Reg.A, 5), _sta(Reg.A, "MARK", 0x301), ReturnStmt(src=SRC),
    ])
    mod = ModuleIR3(name="M", file="syn", routines=[caller, target])
    folded = fold_module(mod)
    assigns = [s for s in folded.find("caller").body.stmts if isinstance(s, Assign)]
    assert any(a.target.addr == 0x300 and a.source.value == 1 for a in assigns)
    r1, r2 = _diff_ram(mod, folded, "caller")
    assert r1 == r2 and r2[0x300] == 1 and r2[0x301] == 5 and r2[0x302] == 0


def test_demand_skelprog_folds_across_tailcall(source_dir):
    """Real-world: AUTO.S `SkelProg` is `a = #2 ; *CharSword = a ; jmp
    GuardProg`. GuardProg loads CharSword (overwrites A) before reading
    it, so it doesn't demand A — the constant store folds across the
    tail call to `*CharSword = #2`."""
    ast = parse_files(
        [source_dir / "EQ.S", source_dir / "GAMEEQ.S", source_dir / "AUTO.S"],
        search_paths=[source_dir],
    )
    auto = next(f for f in ast.files if Path(f.path).name == "AUTO.S")
    ir1 = lift_file(auto, ast.symbols(), ["SkelProg", "GuardProg"]).module
    folded = fold_module(reloop_module(structure_module(ir1)))
    body = folded.find("SkelProg").body.stmts
    assert isinstance(body[0], Assign)
    assert body[0].target.name == "CharSword"
    assert isinstance(body[0].source, Imm) and body[0].source.value == 2
    assert isinstance(body[1], TailCallStmt) and body[1].target == "GuardProg"


# ----------------------------------------- interprocedural carry demand


def _target_kills_carry(name="target") -> RoutineIR3:
    """A target whose first act is `clc` — kills carry before reading it."""
    return _routine(name, [_clc(), _ldimm(Reg.A, 0), ReturnStmt(src=SRC)])


def _target_reads_carry(name="target") -> RoutineIR3:
    """A target whose first act is `adc` — reads the incoming carry."""
    return _routine(name, [_adc_imm(0), _sta(Reg.A, "OUT", 0x400), ReturnStmt(src=SRC)])


def _target_preserves_carry(name="target") -> RoutineIR3:
    """A target that never touches carry — it passes through on return."""
    return _routine(name, [_ldimm(Reg.X, 1), ReturnStmt(src=SRC)])


def _arith_caller_jsr(name="caller") -> RoutineIR3:
    """`lda X ; clc ; adc #8 ; sta Y ; jsr target` — fold requires carry
    dead at the `jsr`. A is killed by the callee's `lda #0`."""
    return _routine(name, [
        _load_a_abs("X", 0x10), _clc(), _adc_imm(8),
        _store_a_abs("Y", 0x20),
        CallStmt(target="target", src=SRC),
    ])


def _arith_caller_jmp(name="caller") -> RoutineIR3:
    """`lda X ; clc ; adc #8 ; sta Y ; jmp target`."""
    return _routine(name, [
        _load_a_abs("X", 0x10), _clc(), _adc_imm(8),
        _store_a_abs("Y", 0x20),
        TailCallStmt(target="target", src=SRC),
    ])


def test_demand_carry_not_read_by_call_folds():
    """`clc ; adc #8 ; sta Y ; jsr target` folds to `Y = X + 8` when
    the callee kills carry before reading it (starts with `clc`)."""
    mod = ModuleIR3(name="M", file="syn",
                    routines=[_arith_caller_jsr(), _target_kills_carry()])
    folded = fold_module(mod)
    assigns = [s for s in folded.find("caller").body.stmts if isinstance(s, Assign)]
    assert len(assigns) == 1
    assert isinstance(assigns[0].source, BinExpr) and assigns[0].source.op == "+"


def test_demand_carry_read_by_call_blocks_fold():
    """`clc ; adc #8 ; sta Y ; jsr target` must NOT fold when the callee
    reads carry first (starts with `adc`)."""
    mod = ModuleIR3(name="M", file="syn",
                    routines=[_arith_caller_jsr(), _target_reads_carry()])
    folded = fold_module(mod)
    assert not any(isinstance(s, Assign) for s in folded.find("caller").body.stmts)


def test_demand_carry_not_read_by_tailcall_folds():
    """`clc ; adc #8 ; sta Y ; jmp target` folds when target kills carry."""
    target = _routine("target", [_clc(), _ldimm(Reg.A, 0), _ldimm(Reg.A, 1), ReturnStmt(src=SRC)])
    mod = ModuleIR3(name="M", file="syn",
                    routines=[_arith_caller_jmp(), target])
    folded = fold_module(mod)
    assigns = [s for s in folded.find("caller").body.stmts if isinstance(s, Assign)]
    assert len(assigns) == 1
    assert isinstance(assigns[0].source, BinExpr) and assigns[0].source.op == "+"


def test_demand_carry_preserved_by_tailcall_blocks_fold():
    """Target preserves carry (never touches it) and returns — carry
    escapes to the outer caller, so the fold is blocked."""
    mod = ModuleIR3(name="M", file="syn",
                    routines=[_arith_caller_jmp(), _target_preserves_carry()])
    folded = fold_module(mod)
    assert not any(isinstance(s, Assign) for s in folded.find("caller").body.stmts)


def test_demand_carry_unknown_target_blocks_fold():
    """A call to an out-of-module target conservatively demands carry."""
    caller = _routine("caller", [
        _load_a_abs("X", 0x10), _clc(), _adc_imm(8),
        _store_a_abs("Y", 0x20),
        CallStmt(target="external", src=SRC),
    ])
    mod = ModuleIR3(name="M", file="syn", routines=[caller])
    folded = fold_module(mod)
    assert not any(isinstance(s, Assign) for s in folded.find("caller").body.stmts)


# --------------------------------------------------------------- whole tree


def test_fold_whole_tree_is_robust_and_idempotent(source_dir):
    """Fold every relooped module across the whole source tree. Two
    guards: folding never crashes, and it's idempotent — re-folding an
    already-folded module is a no-op (an `Assign` carries no `A` to
    fold). Also asserts folding actually fires somewhere, so a future
    regression that silently disables the pass gets caught."""
    files = sorted(source_dir.glob("*.S"))
    base_order = [source_dir / "EQ.S", source_dir / "GAMEEQ.S"]
    base = [p for p in base_order if p.exists()]
    others = [p for p in files if p not in base]
    ast = parse_files([*base, *others], search_paths=[source_dir])
    symbols = ast.symbols()

    total_assigns = 0
    for src_path in files:
        file_ast = next(
            (f for f in ast.files if Path(f.path).resolve() == src_path.resolve()),
            None,
        )
        if file_ast is None:
            continue
        entries = discover_entries(file_ast)
        if not entries:
            continue
        ir1 = lift_file(file_ast, symbols, entries).module
        if not ir1.routines:
            continue
        ir3 = reloop_module(structure_module(ir1))
        folded = fold_module(ir3)
        total_assigns += fold_stats(folded)
        # Idempotent: folding the folded module changes nothing.
        twice = fold_module(folded)
        assert ir3_mod.format_module(twice) == ir3_mod.format_module(folded), (
            f"fold not idempotent on {src_path.name}"
        )

    assert total_assigns > 0, "folding fired nowhere across the tree"
