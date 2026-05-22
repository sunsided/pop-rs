"""IR1 interpreter — executes lifted IR1 against a 64K byte array.

This is the pass-1 half of the differential-test harness described in
`docs/architecture.md`: the emulator harness will snapshot 6502 RAM at
deterministic checkpoints, and this interpreter must produce the
identical post-state when run over the lifted IR1. Pass 2's structured
IR will use the same interpreter, so we keep the surface narrow and
explicit.

The interpreter is deliberately minimal:

* 64K main RAM (`bytearray`). Aux RAM and bank-switched language card
  pages will land alongside the SMC / hi-res work; the AUTO.S combat
  pilot only touches zero-page and the soft-switch image in main RAM.
* Pseudo-registers `a`, `x`, `y` (8-bit). The pilot doesn't read flags,
  so `z`/`n`/`c`/`v` are not yet modelled on `Trace`; they land in the
  next slice along with `cmp` and conditional branches.
* No call stack — `jsr` lifts in a later slice. `jmp` to another routine
  is treated as a tail call: the interpreter switches routines and
  inherits the eventual `rts`.

`run` returns a `Trace` carrying the post-state plus the set of RAM
addresses that were written. Tests assert against that diff rather than
against the whole 64K so the assertions stay readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ir1 import (
    Goto,
    Label,
    LoadImm,
    ModuleIR1,
    Reg,
    Return,
    Routine,
    StoreAbs,
    Unsupported,
)


class InterpError(RuntimeError):
    """Anything the IR1 interpreter cannot honestly execute."""


@dataclass
class Trace:
    """The observable post-state of an interpreter run."""

    ram: bytearray
    a: int
    x: int
    y: int
    writes: dict[int, int] = field(default_factory=dict)  # addr -> last value
    steps: int = 0

    def diff_against(self, initial: bytes) -> dict[int, int]:
        """Return only the addresses whose byte differs from `initial`.
        Useful for differential assertions that ignore untouched RAM."""
        out: dict[int, int] = {}
        for i in range(min(len(initial), len(self.ram))):
            if self.ram[i] != initial[i]:
                out[i] = self.ram[i]
        return out


def _find_label_index(routine: Routine, name: str) -> int:
    for idx, item in enumerate(routine.body):
        if isinstance(item, Label) and item.name == name:
            return idx
    raise InterpError(
        f"local goto target {name!r} not found in routine {routine.name!r}"
    )


def run(
    module: ModuleIR1,
    entry: str,
    *,
    ram: bytearray | None = None,
    a: int = 0,
    x: int = 0,
    y: int = 0,
    max_steps: int = 100_000,
) -> Trace:
    """Execute `module`'s routine `entry` starting from the given
    register/RAM state. Stops on the first `rts` reached (after
    following tail-calls into other routines).
    """
    if ram is None:
        ram = bytearray(0x10000)
    elif len(ram) != 0x10000:
        raise ValueError(f"ram must be 64K, got {len(ram)} bytes")

    routine = module.find(entry)
    if routine is None:
        raise InterpError(f"unknown entry {entry!r} in module {module.name!r}")

    trace = Trace(ram=ram, a=a & 0xff, x=x & 0xff, y=y & 0xff)
    idx = 0
    body = routine.body

    while True:
        if trace.steps >= max_steps:
            raise InterpError(
                f"step limit ({max_steps}) reached in {routine.name!r}; "
                f"likely an unterminated loop"
            )
        if idx >= len(body):
            # A routine body that ran off the end without a terminator
            # would be a lifter bug. The pilot routines all end with
            # `rts` or a tail-call `jmp`, so this should be unreachable.
            raise InterpError(
                f"routine {routine.name!r} ran past end of body; missing terminator?"
            )

        item = body[idx]
        trace.steps += 1

        if isinstance(item, Label):
            idx += 1
            continue

        if isinstance(item, LoadImm):
            value = item.imm.value & 0xff
            if item.reg is Reg.A:
                trace.a = value
            elif item.reg is Reg.X:
                trace.x = value
            else:
                trace.y = value
            idx += 1
            continue

        if isinstance(item, StoreAbs):
            value = {
                Reg.A: trace.a,
                Reg.X: trace.x,
                Reg.Y: trace.y,
            }[item.reg]
            addr = item.target.addr & 0xffff
            ram[addr] = value
            trace.writes[addr] = value
            idx += 1
            continue

        if isinstance(item, Goto):
            if item.kind == "tail_call":
                target = module.find(item.target)
                if target is None:
                    raise InterpError(
                        f"tail-call target {item.target!r} not in module "
                        f"{module.name!r}; was it lifted?"
                    )
                routine = target
                body = routine.body
                idx = 0
                continue
            # local goto
            idx = _find_label_index(routine, item.target)
            continue

        if isinstance(item, Return):
            return trace

        if isinstance(item, Unsupported):
            raise InterpError(
                f"refusing to execute unsupported opcode "
                f"{item.mnemonic!r} at {item.src.short()} ({item.src.raw!r})"
            )

        raise InterpError(f"unknown IR1 item: {type(item).__name__}")
