"""Ground-truth + invariant tests for segment jump-table alias
resolution (`slot_aliases`).

The slot→target map is reconstructed by *position* (i-th dum slot ↔
i-th head `jmp`), so the anchors here are concrete pairs read straight
out of the POP sources — including a renamed slot and a name defined in
two segments — plus the two invariants the resolver must never break:
it must shrink the unresolved-call set and must never turn a name into
a cross-module collision.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pop_lifter import pass4_crate as pc
from pop_lifter.cli import lift_all_modules
from pop_lifter.pass0_parse import parse_files
from pop_lifter.slot_aliases import (
    _head_jump_targets,
    apply_slot_aliases,
    slot_alias_entries,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = REPO_ROOT / "vendor" / "pop-apple2" / "01 POP Source" / "Source"


@pytest.fixture(scope="module")
def program_ast():
    files = sorted(SOURCE_DIR.glob("*.S"))
    base = [p for p in (SOURCE_DIR / "EQ.S", SOURCE_DIR / "GAMEEQ.S") if p.exists()]
    others = [p for p in files if p not in base]
    return parse_files([*base, *others], search_paths=[SOURCE_DIR])


# ---------------------------------------------------------------- parsing


def test_head_jump_targets_reads_the_table_in_order(program_ast):
    # AUTO.S's head table, verbatim from the source.
    assert _head_jump_targets(program_ast, "auto") == [
        "AUTOCTRL",
        "CHECKSTRIKE",
        "CHECKSTAB",
        "AUTOPLAYBACK",
        "CUTCHECK",
        "CUTGUARD",
        "ADDGUARD",
        "CUT",
    ]


def test_head_jump_targets_absent_segment_is_none(program_ast):
    # `misc` has a dum block but `MISC.S`'s table is unrelated/short; a
    # name with no `<seg>.S` file at all yields None.
    assert _head_jump_targets(program_ast, "savedgame") is None


# ---------------------------------------------------------------- mapping


def _entry_set(program_ast):
    return set(slot_alias_entries(program_ast))


def test_entries_include_plain_slot(program_ast):
    # auto slot 0 → AUTOCTRL, by position.
    assert ("AUTOCTRL", "AUTO", "AutoCtrl") in _entry_set(program_ast)


def test_entries_include_renamed_slot(program_ast):
    # hires slot 10 is labelled `_copy2000` but the head `jmp` targets a
    # routine since renamed `copyscrnMM`; position resolves it correctly.
    assert ("copyscrnMM", "HIRES", "_copy2000") in _entry_set(program_ast)


def test_entries_include_segment_shared_name(program_ast):
    # LOADLEVEL is defined in both GRAFIX and MASTER; the MASTER head
    # table claims this slot for MASTER.
    assert ("LOADLEVEL", "MASTER", "_loadlevel") in _entry_set(program_ast)


def test_short_table_does_not_overrun_into_data_slots(program_ast):
    # MISC's head table has only two `jmp`s, so dum slots past index 1
    # (e.g. `StabChar` at offset 21) must NOT be mapped to anything.
    aliases = {alias for _, _, alias in slot_alias_entries(program_ast)}
    assert "VanishChar" in aliases  # slot 0, in the table
    assert "StabChar" not in aliases  # slot 7, past the table


def test_bankswitch_overlay_block_is_ignored(program_ast):
    # MASTER's second dum block at $f880 leads with a 15-byte field — an
    # alternate loader overlay, not the 3-byte jump table — so its
    # labels (`_edreboot`, `_gobuild`, …) must never become aliases.
    aliases = {alias for _, _, alias in slot_alias_entries(program_ast)}
    for label in ("_edreboot", "_gobuild", "_gogame", "_writedir"):
        assert label not in aliases


# ---------------------------------------------------------------- apply


def test_apply_attaches_ground_truth_aliases():
    modules = lift_all_modules(SOURCE_DIR)  # aliases already applied here

    def routine(module: str, canonical: str):
        for m in modules:
            if m.name != module:
                continue
            for r in m.routines:
                if r.name == canonical:
                    return r
        return None

    assert routine("AUTO", "AUTOCTRL").entry_aliases == ["AutoCtrl"]
    # renamed slot lands on the renamed routine
    assert "_copy2000" in routine("HIRES", "copyscrnMM").entry_aliases
    # segment-shared name: the alias binds to MASTER's copy, not GRAFIX's
    assert "_loadlevel" in routine("MASTER", "LOADLEVEL").entry_aliases
    assert "_loadlevel" not in routine("GRAFIX", "LOADLEVEL").entry_aliases


def _lift_without_aliases(program_ast):
    """Re-run the pipeline but skip the alias step, so a before/after
    comparison is possible. Mirrors `lift_all_modules` minus the final
    `apply_slot_aliases` call."""
    from pop_lifter.cli import _recovered_module
    from pop_lifter.pass1_lift import discover_entries

    symbols = program_ast.symbols()
    modules = []
    for src_path in sorted(SOURCE_DIR.glob("*.S")):
        file_ast = next(
            (f for f in program_ast.files if Path(f.path).resolve() == src_path.resolve()),
            None,
        )
        if file_ast is None:
            continue
        entries = discover_entries(file_ast)
        if not entries:
            continue
        recovered = _recovered_module(file_ast, symbols, entries)
        if recovered is not None:
            modules.append(recovered)
    return modules


def test_apply_shrinks_unresolved_calls_without_new_collisions(program_ast):
    modules = _lift_without_aliases(program_ast)

    before = pc.analyze_program(modules)
    unresolved_before = before.unresolved_targets()
    collisions_before = set(before.collisions)

    apply_slot_aliases(modules, program_ast)

    after = pc.analyze_program(modules)
    unresolved_after = after.unresolved_targets()
    collisions_after = set(after.collisions)

    newly_resolved = unresolved_before - unresolved_after
    # The audited win is 251; assert a strong lower bound so the guard
    # survives unrelated lifting churn but still catches a regression to
    # "nothing resolves".
    assert len(newly_resolved) >= 240
    # Resolution only ever improves — nothing that resolved before may
    # become unresolved.
    assert unresolved_after <= unresolved_before
    # And the single hard invariant: never introduce a name collision.
    assert collisions_after == collisions_before
