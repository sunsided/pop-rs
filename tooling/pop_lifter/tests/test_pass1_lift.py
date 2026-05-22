"""Pass-1 lifter tests for the AUTO.S combat-button pilot.

These tests pin the IR1 shape produced by `pass1_lift.lift_file` for the
seven combat routines and the auxiliaries they tail-call into. The
pilot is the smallest set of routines that exercises multi-entry labels
(`DoBlock` / `DoUp` on consecutive lines), `#-1` / `#0` immediate
stores, unconditional cross-routine `jmp` (tail calls), and fall-through
into the shared `]rts` trampoline.
"""

from __future__ import annotations

from pathlib import Path

from pop_lifter.ir1 import (
    Goto,
    Label,
    LoadImm,
    Reg,
    Return,
    StoreAbs,
)
from pop_lifter.pass0_parse import parse_files
from pop_lifter.pass1_lift import lift_file


PILOT_ENTRIES = [
    "DoStrike", "DoBlock", "DoTurn",
    "DoStandup", "DoEngarde", "DoRelBtn", "DoRelease",
]


def _lift_auto(source_dir: Path):
    auto = source_dir / "AUTO.S"
    ast = parse_files(
        [source_dir / "EQ.S", source_dir / "GAMEEQ.S", auto],
        search_paths=[source_dir],
    )
    file_ast = next(f for f in ast.files if Path(f.path).name == "AUTO.S")
    return lift_file(file_ast, ast.equates, PILOT_ENTRIES)


def test_pilot_routines_lift_with_no_unsupported_ops(source_dir):
    report = _lift_auto(source_dir)
    assert report.unsupported == []


def test_pilot_routines_present(source_dir):
    module = _lift_auto(source_dir).module
    found = {r.name for r in module.routines}
    # Aliases collapse into a single routine — DoUp/DoDown/DoPress/
    # DoFwd/DoBack should not appear as separate names.
    assert found == {
        "DoStrike", "DoBlock", "DoTurn",
        "DoStandup", "DoEngarde", "DoRelBtn", "DoRelease",
        # Pulled in transitively as tail-call targets:
        "DoRetreat", "DoAdvance",
    }


def test_multi_entry_aliases_collapse(source_dir):
    module = _lift_auto(source_dir).module
    # `DoBlock` and `DoUp` are bare labels on consecutive lines that
    # both bind to the same `lda #-1` — the lifter should collapse them
    # into one routine.
    block = module.find("DoBlock")
    assert block is not None
    assert block.name == "DoBlock"
    assert block.entry_aliases == ["DoUp"]

    strike = module.find("DoStrike")
    assert strike is not None
    assert strike.entry_aliases == ["DoPress"]


def test_dostrike_body_shape(source_dir):
    module = _lift_auto(source_dir).module
    r = module.find("DoStrike")
    body = r.body
    # lda #-1; sta clrbtn; sta btn; rts
    assert len(body) == 4
    assert isinstance(body[0], LoadImm) and body[0].reg is Reg.A and body[0].imm.value & 0xff == 0xff
    assert isinstance(body[1], StoreAbs) and body[1].target.name == "clrbtn"
    assert isinstance(body[2], StoreAbs) and body[2].target.name == "btn"
    assert isinstance(body[3], Return)


def test_tail_call_kind_for_dostandup(source_dir):
    module = _lift_auto(source_dir).module
    r = module.find("DoStandup")
    last = r.body[-1]
    # The terminating `jmp DoBack` must be classified as a tail call so
    # the IR1 interpreter follows it into DoRetreat/DoBack.
    assert isinstance(last, Goto)
    assert last.kind == "tail_call"
    assert last.target == "DoBack"


def test_dorelbtn_falls_through_into_rts_trampoline(source_dir):
    module = _lift_auto(source_dir).module
    r = module.find("DoRelBtn")
    # The routine ends with the shared `]rts rts` line. The lifter
    # surfaces the `]rts` label as an internal Label item and emits the
    # rts as the terminator.
    kinds = [type(item).__name__ for item in r.body]
    assert kinds == ["LoadImm", "StoreAbs", "Label", "Return"]
    label = r.body[2]
    assert isinstance(label, Label)
    assert label.name == "]rts"


def test_source_refs_preserved(source_dir):
    module = _lift_auto(source_dir).module
    # Every emitted IR1 instruction must carry the source line it lifted
    # from — pass 4 relies on this for the @generated manifest.
    for routine in module.routines:
        for item in routine.body:
            assert item.src.line >= 1
            assert item.src.file.endswith("AUTO.S")
            assert item.src.raw.strip() != ""
