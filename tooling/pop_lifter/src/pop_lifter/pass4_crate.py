"""Crate-assembly analysis (issue #47), slice 1: the program-wide
name → owning-module map and the call-resolution policy.

This is the *analysis foundation* for replacing the per-file `Cpu`
inspection tree with a single coherent crate. It performs no emission —
it answers, for every inter-routine `jsr`/`jmp` in the lifted program,
"which module owns the target?" — so the later emitter can namespace
routines as `mod auto`/`mod ctrl`/… free functions over a shared `Cpu`.

Why this is non-trivial: POP is built as separately-assembled overlay
segments that **reuse routine names**. The same name denotes *different*
routines in different segments — e.g. `DoAdvance` is defined
independently in both `AUTO.S` and `CTRL.S`. A single `impl Cpu` can't
hold both (`Cpu::DoAdvance` can't be two functions), and dedup would be
semantically wrong because the bodies genuinely differ. So a name maps
to a *set* of owning modules, and a call must be resolved relative to
the module it is made *from*.

Resolution policy (`resolve_call`), in priority order:

* **INTRA_MODULE** — the calling module defines a routine of that name.
  Overlay segments overwhelmingly call their own routines, so a target
  the caller defines resolves to the caller. This is the rule that
  dissolves the name-reuse collisions: `auto::DoAdvance`'s call to
  `DoAdvance` binds to `auto::DoAdvance`, never `ctrl::DoAdvance`.
* **CROSS_MODULE** — the caller doesn't define the name but exactly one
  *other* module does; the call binds to that module's routine.
* **EXTERNAL** — no lifted module defines the name. The target is ROM /
  firmware / monitor / a data label the lift doesn't cover; the emitter
  will stub it.
* **AMBIGUOUS** — the caller doesn't define the name and *more than one*
  other module does, so there's no unique owner. Left for the emitter
  to resolve with extra context (call-graph / segment-layout policy);
  flagged here rather than guessed.

A "name" here is a routine *entry* name (`Routine.name` or an
`entry_aliases` member) — the callable unit the free-function model
exposes. Body-internal labels (`:local`/`]macro`) are not callable
across routines and are deliberately excluded; a tail call to one would
surface as EXTERNAL.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from . import ir1
from .ir3 import (
    CallStmt,
    DispatchStmt,
    DoWhileStmt,
    ForStmt,
    IfStmt,
    LabeledBlock,
    LoopStmt,
    MatchStmt,
    ModuleIR3,
    RawIfStmt,
    RawStmt,
    RepeatStmt,
    RoutineIR3,
    TailCallStmt,
)


class CallDisposition(str, Enum):
    """How a call target resolves against the program's module set. See
    the module docstring for the policy each value encodes."""

    INTRA_MODULE = "intra_module"
    CROSS_MODULE = "cross_module"
    EXTERNAL = "external"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class CallSite:
    """One inter-routine transfer found in a routine body. `kind` is
    `"call"` (`jsr`/`CallStmt`) or `"tail_call"` (`jmp`-to-routine /
    `TailCallStmt`); `target` is the symbolic label name."""

    caller_module: str
    caller_routine: str
    kind: str
    target: str


@dataclass(frozen=True)
class Resolution:
    """The owning-module verdict for one call target made from
    `caller_module`. `owner` is the resolved module name for
    INTRA_MODULE / CROSS_MODULE and `None` otherwise. `candidates` lists
    every module defining `target`, sorted — empty for EXTERNAL, a
    singleton for CROSS_MODULE, the caller-included set for the others."""

    target: str
    caller_module: str
    disposition: CallDisposition
    owner: str | None
    candidates: tuple[str, ...]


def defined_entry_names(module: ModuleIR3) -> set[str]:
    """Every routine *entry* name the module defines (`name` plus
    `entry_aliases`) — the callable units, not body-internal labels."""
    names: set[str] = set()
    for r in module.routines:
        names.add(r.name)
        names.update(r.entry_aliases)
    return names


def build_name_map(modules: list[ModuleIR3]) -> dict[str, list[str]]:
    """Program-wide entry-name → sorted owning-module names. A name with
    more than one module is a reused-name collision (see `collisions`)."""
    owners: dict[str, set[str]] = {}
    for module in modules:
        for name in defined_entry_names(module):
            owners.setdefault(name, set()).add(module.name)
    return {name: sorted(mods) for name, mods in owners.items()}


def collisions(name_map: dict[str, list[str]]) -> dict[str, list[str]]:
    """The subset of `name_map` whose names are defined in more than one
    module — the overlay name-reuse collisions a single `impl Cpu`
    can't represent."""
    return {name: mods for name, mods in name_map.items() if len(mods) > 1}


def resolve_call(
    name_map: dict[str, list[str]], caller_module: str, target: str
) -> Resolution:
    """Resolve `target` called from `caller_module` per the policy in the
    module docstring (intra-module preferred, then unique cross-module,
    then external, with multi-owner cross-module flagged ambiguous)."""
    candidates = tuple(name_map.get(target, ()))
    if caller_module in candidates:
        return Resolution(
            target, caller_module, CallDisposition.INTRA_MODULE,
            caller_module, candidates,
        )
    if not candidates:
        return Resolution(
            target, caller_module, CallDisposition.EXTERNAL, None, candidates,
        )
    if len(candidates) == 1:
        return Resolution(
            target, caller_module, CallDisposition.CROSS_MODULE,
            candidates[0], candidates,
        )
    return Resolution(
        target, caller_module, CallDisposition.AMBIGUOUS, None, candidates,
    )


def _collect_calls(block, sink: list[tuple[str, str]]) -> None:
    """Depth-first walk of an IR3 `Block`, appending `(kind, target)` for
    every call / tail call — including ones still wrapped in a `RawStmt`
    (`ir1.Call` / `ir1.Goto(kind="tail_call")`) before relooping lifted
    them to `CallStmt` / `TailCallStmt`."""
    for stmt in block.stmts:
        if isinstance(stmt, CallStmt):
            sink.append(("call", stmt.target))
        elif isinstance(stmt, TailCallStmt):
            sink.append(("tail_call", stmt.target))
        elif isinstance(stmt, RawStmt):
            item = stmt.item
            if isinstance(item, ir1.Call):
                sink.append(("call", item.target))
            elif isinstance(item, ir1.Goto) and item.kind == "tail_call":
                sink.append(("tail_call", item.target))
        elif isinstance(stmt, (IfStmt, RawIfStmt)):
            _collect_calls(stmt.then_block, sink)
            if stmt.else_block is not None:
                _collect_calls(stmt.else_block, sink)
        elif isinstance(stmt, (LoopStmt, DoWhileStmt, ForStmt, RepeatStmt, LabeledBlock)):
            _collect_calls(stmt.body, sink)
        elif isinstance(stmt, (MatchStmt, DispatchStmt)):
            for arm in stmt.arms:
                _collect_calls(arm.body, sink)


def routine_call_sites(module: ModuleIR3, routine: RoutineIR3) -> list[CallSite]:
    """All inter-routine call sites in `routine`, in source order."""
    found: list[tuple[str, str]] = []
    _collect_calls(routine.body, found)
    return [
        CallSite(module.name, routine.name, kind, target)
        for kind, target in found
    ]


@dataclass(frozen=True)
class CrateCallGraph:
    """Result of `analyze_program`: the name map, the reused-name
    collisions, and a `Resolution` for every call site in the program."""

    name_map: dict[str, list[str]]
    collisions: dict[str, list[str]]
    call_sites: tuple[CallSite, ...]
    resolutions: tuple[Resolution, ...]

    def disposition_counts(self) -> dict[CallDisposition, int]:
        counts = dict.fromkeys(CallDisposition, 0)
        for r in self.resolutions:
            counts[r.disposition] += 1
        return counts

    def unresolved_targets(self) -> set[str]:
        """External + ambiguous targets — the call sites the emitter
        can't bind to a single lifted routine without more policy."""
        return {
            r.target
            for r in self.resolutions
            if r.disposition in (CallDisposition.EXTERNAL, CallDisposition.AMBIGUOUS)
        }


def analyze_program(modules: list[ModuleIR3]) -> CrateCallGraph:
    """Build the name map and resolve every call site in `modules`."""
    name_map = build_name_map(modules)
    coll = collisions(name_map)
    sites: list[CallSite] = []
    for module in modules:
        for routine in module.routines:
            sites.extend(routine_call_sites(module, routine))
    resolutions = tuple(
        resolve_call(name_map, s.caller_module, s.target) for s in sites
    )
    return CrateCallGraph(name_map, coll, tuple(sites), resolutions)
