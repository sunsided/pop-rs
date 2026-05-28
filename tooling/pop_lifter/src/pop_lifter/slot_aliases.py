"""Resolve segment jump-table call targets to lifted routines.

Each POP segment exposes its public entry points through a fixed *head
jump table*: right after `org org` the segment emits one `jmp <TARGET>`
per slot, in slot order. The matching `dum` overlay in `EQ.S` /
`GAMEEQ.S` (named after the segment, e.g. `dum auto`) labels those same
slots as 3-byte fields — `AutoCtrl ds 3`, `CheckStrike ds 3`, ... — so
the rest of the program can `jsr AutoCtrl` without knowing the segment's
load address.

The lifter names each routine after its *definition* label (the `jmp`
target, e.g. `AUTOCTRL`), so a `jsr AutoCtrl` call site resolves to
nothing. This module reconstructs the slot→target correspondence purely
by position — the i-th dum slot (offset `3*i`) is the i-th head `jmp` —
and attaches each slot label as an `entry_alias` on the target routine.
Reading the *actual* `jmp` operand (rather than fuzzy name matching)
handles renamed slots (`copyscrnMM` → `_copy2000`) and is correct by
construction.

A head jump table is the segment's *own* public API: each `jmp TARGET`
points to a routine defined in that same segment, so a slot only ever
binds to a routine the dum's segment lifts. Binding strictly within the
segment is what makes the result trustworthy — a head-jump target that
happens to share its name with an unrelated routine in another segment
(e.g. GRAFIX's `DOSTARTGAME` bank-switch trampoline vs. MASTER's real
`DOSTARTGAME`) must never be claimed by the wrong one.

Two kinds of slot are deliberately *not* aliased, because doing so
would be wrong rather than helpful:

* a slot whose head-jump target the segment doesn't actually lift (the
  real routine lives elsewhere or wasn't extracted) — the call stays
  external rather than latching onto a same-named routine in another
  segment;
* an alias name already owned by some routine (attaching it would make
  that name resolve to two modules — e.g. GRAFIX's `cls`/`lay`/`peel`
  bank-switch trampolines collide with HIRES's same-named routines).

These stay external, matching the pre-existing call-resolution policy.
"""

from __future__ import annotations

from pathlib import Path

from .ir3 import ModuleIR3
from .pass0_parse import ProgramAST


def _head_jump_targets(ast: ProgramAST, segment: str) -> list[str] | None:
    """The ordered `jmp` targets of `<segment>.S`'s head jump table, or
    `None` when the segment has no parsed file or no table.

    The table is the contiguous run of `jmp` instructions immediately
    after the `org org` directive; blank and comment lines within the
    run are skipped, and the first non-`jmp` code line (`lst`, `put`, …)
    ends it.
    """
    file_ast = next(
        (f for f in ast.files if Path(f.path).stem.lower() == segment.lower()),
        None,
    )
    if file_ast is None:
        return None

    start = next(
        (
            i
            for i, ln in enumerate(file_ast.lines)
            if ln.mnemonic == "org" and (ln.operand or "").strip() == "org"
        ),
        None,
    )
    if start is None:
        return None

    targets: list[str] = []
    for ln in file_ast.lines[start + 1:]:
        if ln.is_blank:
            continue
        if ln.mnemonic == "jmp":
            targets.append((ln.operand or "").strip())
        else:
            break
    return targets or None


def slot_alias_entries(ast: ProgramAST) -> list[tuple[str, str, str]]:
    """`(target, segment, alias)` for every head-jump-table slot whose
    position maps a dum label to a real `jmp` target.

    Only pure 3-byte-slot dum blocks are considered: a block with any
    non-3-byte field is a data overlay (or an alternate bank-switch
    overlay that shares the segment's load address), not the jump table,
    and index alignment wouldn't hold.
    """
    out: list[tuple[str, str, str]] = []
    for block in ast.dum_blocks:
        if any(f.size != 3 for f in block.fields):
            continue
        targets = _head_jump_targets(ast, block.start_expr)
        if not targets:
            continue
        seg = block.start_expr.upper()
        for fld in block.fields:
            if fld.name is None or fld.offset % 3:
                continue
            idx = fld.offset // 3
            if idx < len(targets):
                out.append((targets[idx], seg, fld.name))
    return out


def apply_slot_aliases(modules: list[ModuleIR3], ast: ProgramAST) -> int:
    """Attach head-jump-table slot labels as `entry_aliases` on their
    target routines, mutating `modules` in place. Returns the number of
    aliases added.

    A slot binds only to the routine its own segment defines under the
    head-jump target name; if that segment doesn't lift such a routine
    the slot stays external. An alias name already owned by some routine
    is also skipped, so attaching it can never turn a single name into a
    cross-module collision.
    """
    by_seg_name: dict[tuple[str, str], object] = {}
    owners: dict[str, set[str]] = {}
    for module in modules:
        for routine in module.routines:
            by_seg_name.setdefault((module.name, routine.name), routine)
            for name in (routine.name, *routine.entry_aliases):
                owners.setdefault(name, set()).add(module.name)

    applied = 0
    for target, seg, alias in slot_alias_entries(ast):
        routine = by_seg_name.get((seg, target))
        if routine is None:
            # The segment's head jump points at a routine it doesn't
            # itself lift — leave the call external rather than latch
            # onto a same-named routine in another segment.
            continue
        if alias == routine.name or alias in routine.entry_aliases:
            continue
        if owners.get(alias):
            # Already owned by some routine — aliasing here would make
            # the name resolve to two modules. Leave the call external.
            continue

        routine.entry_aliases.append(alias)
        owners.setdefault(alias, set()).add(seg)
        applied += 1
    return applied
