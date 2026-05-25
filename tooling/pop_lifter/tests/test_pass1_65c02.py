"""Pass-1 (+ interp + emit) tests for the 65C02 opcodes POP uses:
`phy` (push Y) and `tsb`/`trb` (test-and-set / test-and-reset memory
bits). These appear only in the IIgs-gated speed / control-panel code
(`FASTSPEED`/`NORMSPEED`/`getparam`/`setparam`), but they're genuine
65C02 instructions, so the lifter models them rather than leaving `???`.
"""

from __future__ import annotations

from pop_lifter.interp_ir1 import Trace, exec_atom
from pop_lifter.ir1 import Abs, MemBitOp, Phy, SourceRef
from pop_lifter.pass0_lex import Line
from pop_lifter.pass1_lift import _lift_instr
from pop_lifter.pass4_emit_rust import _emit_raw

_SRC = SourceRef(file="syn", line=0, raw="")


def _line(mnemonic: str, operand: str | None = None) -> Line:
    return Line(
        file="syn", lineno=1, raw=f"  {mnemonic} {operand or ''}".rstrip(),
        label=None, mnemonic=mnemonic, operand=operand, comment=None,
    )


def _trace(a=0, y=0) -> Trace:
    return Trace(ram=bytearray(0x10000), a=a, x=0, y=y)


# ---- lifter dispatch


def test_phy_lifts_to_phy():
    assert isinstance(_lift_instr(_line("phy"), {}, set()), Phy)


def test_tsb_trb_lift_to_membitop():
    tsb = _lift_instr(_line("tsb", "$c036"), {}, set())
    assert isinstance(tsb, MemBitOp) and tsb.op == "tsb"
    assert tsb.target.addr == 0xC036
    trb = _lift_instr(_line("trb", "$c036"), {}, set())
    assert isinstance(trb, MemBitOp) and trb.op == "trb"
    assert trb.target.addr == 0xC036


# ---- interpreter semantics


def test_phy_pushes_y_onto_value_stack():
    t = _trace(y=0x55)
    exec_atom(Phy(src=_SRC), t, t.ram)
    assert t.value_stack == [0x55]
    assert t.max_value_stack_depth == 1


def test_tsb_sets_bits_and_z_from_old_value():
    # A selects bit 7; memory starts clear → Z set (no overlap), then the
    # bit is OR'd in.
    t = _trace(a=0x80)
    addr = 0xC036
    exec_atom(MemBitOp(op="tsb", target=Abs(name="reg", addr=addr), src=_SRC), t, t.ram)
    assert t.z == 1
    assert t.ram[addr] == 0x80


def test_trb_resets_bits_and_z_reflects_overlap():
    # A selects bit 7; memory has it set → Z clear (overlap), then the bit
    # is masked out, leaving the rest.
    t = _trace(a=0x80)
    addr = 0xC036
    t.ram[addr] = 0xFF
    exec_atom(MemBitOp(op="trb", target=Abs(name="reg", addr=addr), src=_SRC), t, t.ram)
    assert t.z == 0
    assert t.ram[addr] == 0x7F


# ---- pass-4 lowering


def test_phy_lowers_to_stack_push():
    assert _emit_raw(Phy(src=_SRC)) == ["self.stack.push(self.reg.y);"]


def test_tsb_trb_lower_to_test_then_rmw():
    tsb = _emit_raw(MemBitOp(op="tsb", target=Abs(name="r", addr=0xC036), src=_SRC))
    assert tsb == [
        "self.flags.z = (self.reg.a & self.mem[0xc036]) == 0;",
        "self.mem[0xc036] |= self.reg.a;",
    ]
    trb = _emit_raw(MemBitOp(op="trb", target=Abs(name="r", addr=0xC036), src=_SRC))
    assert trb[1] == "self.mem[0xc036] &= !self.reg.a;"
