"""Pass-1 shifts/rotates: `rol a` / `ror a` (accumulator), plus
`asl/lsr/rol/ror addr` (memory). Covers lifter dispatch, interpreter
semantics (including the 16-bit shift idiom `asl lo ; rol hi`), and
pass-2 fusion of the accumulator rotate forms."""

from __future__ import annotations

from pop_lifter.interp_ir1 import Trace, exec_atom
from pop_lifter.ir1 import (
    Abs,
    Branch,
    If,
    Reg,
    Return,
    Rol,
    Ror,
    Routine,
    ShiftMem,
    SourceRef,
)
from pop_lifter.pass0_lex import Line
from pop_lifter.pass1_lift import _lift_instr
from pop_lifter.pass2_struct import structure_routine


def _line(mnemonic: str, operand: str | None = None) -> Line:
    return Line(
        file="syn",
        lineno=1,
        raw=f"  {mnemonic} {operand or ''}".rstrip(),
        label=None,
        mnemonic=mnemonic,
        operand=operand,
        comment=None,
    )


def _trace(a=0, c=0) -> Trace:
    return Trace(ram=bytearray(0x10000), a=a, x=0, y=0, c=c)


# ---- lifter dispatch


def test_rol_accumulator_lifts_to_rol():
    """`rol a` and `rol` (no operand) both mean accumulator rotate."""
    assert isinstance(_lift_instr(_line("rol", "a"), {}, set()), Rol)
    assert isinstance(_lift_instr(_line("rol"), {}, set()), Rol)


def test_ror_accumulator_lifts_to_ror():
    assert isinstance(_lift_instr(_line("ror", "a"), {}, set()), Ror)
    assert isinstance(_lift_instr(_line("ror"), {}, set()), Ror)


def test_asl_memory_lifts_to_shiftmem():
    instr = _lift_instr(_line("asl", "framepoint"), {"framepoint": 0x100}, set())
    assert isinstance(instr, ShiftMem)
    assert instr.op == "asl"
    assert instr.target.addr == 0x100


def test_rol_memory_lifts_to_shiftmem():
    instr = _lift_instr(_line("rol", "framepoint+1"), {"framepoint": 0x100}, set())
    assert isinstance(instr, ShiftMem)
    assert instr.op == "rol"
    assert instr.target.addr == 0x101


def test_ror_memory_lifts_to_shiftmem():
    instr = _lift_instr(_line("ror", "vblflag"), {"vblflag": 0x200}, set())
    assert isinstance(instr, ShiftMem)
    assert instr.op == "ror"


# ---- interpreter semantics


def test_rol_rotates_carry_in_from_bottom():
    """`rol a` with C=1, A=$01 → A=$03 (1<<1 | 1), C=0 (old bit 7)."""
    src = SourceRef(file="syn", line=0, raw="rol a")
    t = _trace(a=0x01, c=1)
    exec_atom(Rol(src=src), t, t.ram)
    assert t.a == 0x03
    assert t.c == 0


def test_rol_captures_bit7_into_carry():
    """`rol a` with C=0, A=$80 → A=$00 (high bit lost), C=1."""
    src = SourceRef(file="syn", line=0, raw="rol a")
    t = _trace(a=0x80, c=0)
    exec_atom(Rol(src=src), t, t.ram)
    assert t.a == 0x00
    assert t.c == 1
    assert t.z == 1


def test_ror_rotates_carry_in_from_top():
    """`ror a` with C=1, A=$02 → A=$81 (0b10 >> 1 | 0b10000000), C=0."""
    src = SourceRef(file="syn", line=0, raw="ror a")
    t = _trace(a=0x02, c=1)
    exec_atom(Ror(src=src), t, t.ram)
    assert t.a == 0x81
    assert t.c == 0
    assert t.n == 1


def test_ror_captures_bit0_into_carry():
    src = SourceRef(file="syn", line=0, raw="ror a")
    t = _trace(a=0x01, c=0)
    exec_atom(Ror(src=src), t, t.ram)
    assert t.a == 0x00
    assert t.c == 1
    assert t.z == 1


def test_shiftmem_asl_doubles_memory_byte():
    src = SourceRef(file="syn", line=0, raw="asl framepoint")
    t = _trace()
    t.ram[0x100] = 0x21
    exec_atom(ShiftMem(op="asl", target=Abs(name="fp", addr=0x100), src=src), t, t.ram)
    assert t.ram[0x100] == 0x42
    assert t.c == 0


def test_shiftmem_rol_uses_carry_from_trace():
    src = SourceRef(file="syn", line=0, raw="rol framepoint+1")
    t = _trace(c=1)
    t.ram[0x101] = 0x80
    exec_atom(ShiftMem(op="rol", target=Abs(name="fp+1", addr=0x101), src=src), t, t.ram)
    # 0x80 << 1 = 0x100 → 0x00; OR carry-in (1) → 0x01; new C = old bit 7 = 1.
    assert t.ram[0x101] == 0x01
    assert t.c == 1


def test_16bit_shift_idiom_doubles_two_bytes():
    """The classic POP 16-bit-shift pair: `asl lo ; rol hi` doubles
    the 16-bit value `hi:lo` in memory. Pin the round-trip so
    downstream code (pass 3's 16-bit fold) has a stable target to
    recognise."""
    src = SourceRef(file="syn", line=0, raw="")
    t = _trace()
    t.ram[0x100] = 0x80   # lo = 0x80
    t.ram[0x101] = 0x01   # hi = 0x01 → 16-bit value = 0x0180

    exec_atom(ShiftMem(op="asl", target=Abs(name="lo", addr=0x100), src=src), t, t.ram)
    # lo: 0x80 << 1 = 0x00, C=1 (the lost bit 7)
    assert t.ram[0x100] == 0x00
    assert t.c == 1

    exec_atom(ShiftMem(op="rol", target=Abs(name="hi", addr=0x101), src=src), t, t.ram)
    # hi: 0x01 << 1 = 0x02, OR carry-in 1 = 0x03; new C = old bit 7 = 0
    assert t.ram[0x101] == 0x03
    assert t.c == 0
    # 16-bit view: 0x0180 << 1 = 0x0300. We got hi:lo = 0x03:0x00 = 0x0300. ✓


# ---- pass-2 fusion


def test_rol_then_beq_fuses_to_zero_test_on_a():
    """`rol a ; beq L` — Rol's Z reflects A's new value, so the
    branch fuses cleanly the same way Asl/Lsr do."""
    src = SourceRef(file="syn", line=0, raw="")
    r = Routine(
        name="f",
        body=[
            Rol(src=src),
            Branch(cond="eq", target="]rts", src=src),
            Return(src=src),
        ],
    )
    out = structure_routine(r)
    ifs = [i for i in out.body if isinstance(i, If)]
    assert len(ifs) == 1
    assert ifs[0].cond.reg is Reg.A
    assert ifs[0].cond.op == "=="
    assert ifs[0].cond.rhs.value == 0
