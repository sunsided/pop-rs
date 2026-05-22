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
* Pseudo-registers `a`, `x`, `y` (8-bit) and a single carry flag `c`.
  Z / N / V land alongside `cmp` and conditional branches in the
  CheckFloor slice.
* Explicit call stack — `jsr` pushes the return point, `rts` pops it.
  An `rts` at depth 0 returns from the run. `jmp` to a routine in any
  loaded module is treated as a tail call: the interpreter switches
  routines without growing the stack.
* Cross-module / jump-table aliasing — `run` accepts either a single
  module or a list, plus an optional `aliases: dict[str, str]` that
  maps Merlin jump-table slot names (e.g. `rnd`) to their concrete
  implementation labels (`RND`). The plan's `jumptables.py` will
  generate that map automatically from the dum blocks; here we accept
  it from the test harness.

`run` returns a `Trace` carrying the post-state plus the set of RAM
addresses that were written. Tests assert against that diff rather than
against the whole 64K so the assertions stay readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ir1 import (
    AdcAbs,
    AdcImm,
    Asl,
    Call,
    Clc,
    Goto,
    Label,
    LoadAbs,
    LoadImm,
    ModuleIR1,
    Reg,
    Return,
    Routine,
    Sec,
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
    c: int = 0          # carry flag (0 or 1)
    writes: dict[int, int] = field(default_factory=dict)  # addr -> last value
    steps: int = 0
    max_stack_depth: int = 0

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


def _resolve(
    modules: list[ModuleIR1],
    aliases: dict[str, str],
    name: str,
) -> Routine | None:
    """Look up a routine entry by name across every loaded module,
    following the alias map. Returns `None` if nothing matches."""
    seen: set[str] = set()
    cur = name
    while cur not in seen:
        seen.add(cur)
        for m in modules:
            r = m.find(cur)
            if r is not None:
                return r
        if cur in aliases:
            cur = aliases[cur]
            continue
        break
    return None


def run(
    module: ModuleIR1 | list[ModuleIR1],
    entry: str,
    *,
    ram: bytearray | None = None,
    a: int = 0,
    x: int = 0,
    y: int = 0,
    c: int = 0,
    aliases: dict[str, str] | None = None,
    max_steps: int = 100_000,
) -> Trace:
    """Execute `entry` starting from the given register/RAM state.

    `module` may be a single `ModuleIR1` or a list — the latter is how
    cross-module calls (e.g. AUTO.S calling into GRAFIX.S's `RND`) get
    resolved. `aliases` adds an extra hop for jump-table slot names
    that don't appear as a label in any module body.

    Stops when an `rts` is executed at call-stack depth 0.
    """
    if ram is None:
        ram = bytearray(0x10000)
    elif len(ram) != 0x10000:
        raise ValueError(f"ram must be 64K, got {len(ram)} bytes")

    modules: list[ModuleIR1] = [module] if isinstance(module, ModuleIR1) else list(module)
    aliases = dict(aliases or {})

    routine = _resolve(modules, aliases, entry)
    if routine is None:
        names = [m.name for m in modules]
        raise InterpError(
            f"unknown entry {entry!r}; not found in any of {names} "
            f"(aliases: {aliases!r})"
        )

    trace = Trace(ram=ram, a=a & 0xff, x=x & 0xff, y=y & 0xff, c=c & 1)
    idx = 0
    body = routine.body
    # Each stack frame remembers the routine we were in and the index
    # of the instruction *after* the `jsr` so `rts` can resume there.
    stack: list[tuple[Routine, int]] = []

    while True:
        if trace.steps >= max_steps:
            raise InterpError(
                f"step limit ({max_steps}) reached in {routine.name!r}; "
                f"likely an unterminated loop"
            )
        if idx >= len(body):
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

        if isinstance(item, LoadAbs):
            value = ram[item.source.addr & 0xffff]
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

        if isinstance(item, Asl):
            old = trace.a & 0xff
            trace.a = (old << 1) & 0xff
            trace.c = (old >> 7) & 1
            idx += 1
            continue

        if isinstance(item, Clc):
            trace.c = 0
            idx += 1
            continue

        if isinstance(item, Sec):
            trace.c = 1
            idx += 1
            continue

        if isinstance(item, AdcImm):
            total = (trace.a & 0xff) + (item.imm.value & 0xff) + trace.c
            trace.a = total & 0xff
            trace.c = 1 if total > 0xff else 0
            idx += 1
            continue

        if isinstance(item, AdcAbs):
            total = (trace.a & 0xff) + ram[item.source.addr & 0xffff] + trace.c
            trace.a = total & 0xff
            trace.c = 1 if total > 0xff else 0
            idx += 1
            continue

        if isinstance(item, Call):
            target = _resolve(modules, aliases, item.target)
            if target is None:
                raise InterpError(
                    f"jsr target {item.target!r} not found in any loaded "
                    f"module (aliases: {aliases!r})"
                )
            stack.append((routine, idx + 1))
            trace.max_stack_depth = max(trace.max_stack_depth, len(stack))
            routine = target
            body = routine.body
            idx = 0
            continue

        if isinstance(item, Goto):
            if item.kind == "tail_call":
                target = _resolve(modules, aliases, item.target)
                if target is None:
                    raise InterpError(
                        f"tail-call target {item.target!r} not found in any "
                        f"loaded module (aliases: {aliases!r})"
                    )
                routine = target
                body = routine.body
                idx = 0
                continue
            # local goto
            idx = _find_label_index(routine, item.target)
            continue

        if isinstance(item, Return):
            if not stack:
                return trace
            routine, idx = stack.pop()
            body = routine.body
            continue

        if isinstance(item, Unsupported):
            raise InterpError(
                f"refusing to execute unsupported opcode "
                f"{item.mnemonic!r} at {item.src.short()} ({item.src.raw!r})"
            )

        raise InterpError(f"unknown IR1 item: {type(item).__name__}")
