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
from pop_lifter.pass1_lift import discover_entries, lift_file


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
    return lift_file(file_ast, ast.symbols(), PILOT_ENTRIES)


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


def test_discover_entries_finds_pilot_labels(source_dir):
    auto = source_dir / "AUTO.S"
    ast = parse_files(
        [source_dir / "EQ.S", source_dir / "GAMEEQ.S", auto],
        search_paths=[source_dir],
    )
    file_ast = next(f for f in ast.files if Path(f.path).name == "AUTO.S")
    entries = discover_entries(file_ast)

    # Every entry must be a global-style label.
    for name in entries:
        assert not name.startswith(":")
        assert not name.startswith("]")

    # The combat pilot labels must all be discovered, including bare
    # labels on consecutive lines (`DoUp`, `DoDown`, `DoFwd`, `DoBack`,
    # `DoPress`).
    found = set(entries)
    for name in [
        "DoStrike", "DoPress", "DoBlock", "DoUp", "DoTurn", "DoDown",
        "DoStandup", "DoEngarde", "DoRelBtn", "DoRelease",
        "DoAdvance", "DoFwd", "DoRetreat", "DoBack",
    ]:
        assert name in found, f"discover_entries missed {name!r}"


def test_discover_entries_skips_data_only_labels(source_dir):
    # AUTO.S has `plus1 db -1,1` — a label bound to a data directive,
    # not a code instruction. The discovery pass must not return it as
    # an entry point.
    auto = source_dir / "AUTO.S"
    ast = parse_files([auto], search_paths=[source_dir])
    file_ast = next(f for f in ast.files if Path(f.path).name == "AUTO.S")
    entries = discover_entries(file_ast)
    assert "plus1" not in entries
    assert "minus1" not in entries


def test_discover_entries_skips_equate_only_files(source_dir):
    # EQ.S and GAMEEQ.S contain only equates and dum overlays. There
    # is nothing to lift, so discovery must return an empty list.
    for name in ("EQ.S", "GAMEEQ.S"):
        ast = parse_files([source_dir / name], search_paths=[source_dir])
        file_ast = next(f for f in ast.files if Path(f.path).name == name)
        assert discover_entries(file_ast) == []


# ---- rndp + RND slice: new opcodes (LoadAbs, ASL, CLC, AdcAbs, AdcImm, Call)


def _lift(source_dir: Path, file: str, entries: list[str]):
    ast = parse_files(
        [source_dir / "EQ.S", source_dir / "GAMEEQ.S", source_dir / file],
        search_paths=[source_dir],
    )
    file_ast = next(f for f in ast.files if Path(f.path).name == file)
    return _lift_to_module(file_ast, ast.symbols(), entries)


def _lift_to_module(file_ast, equates, entries):
    return lift_file(file_ast, equates, entries).module


def test_rndp_lift_shape(source_dir):
    """`rndp` is the smallest cross-module call site: `ldx guardprog`
    + `jmp rnd`. The lifter must produce LoadAbs(X) + tail_call goto."""
    from pop_lifter.ir1 import Goto, LoadAbs, Reg

    module = _lift(source_dir, "AUTO.S", ["rndp"])
    routine = module.find("rndp")
    assert routine is not None
    body = routine.body
    assert len(body) == 2
    assert isinstance(body[0], LoadAbs)
    assert body[0].reg is Reg.X
    assert body[0].source.name == "guardprog"
    assert isinstance(body[1], Goto)
    assert body[1].kind == "tail_call"
    assert body[1].target == "rnd"


def test_rnd_lift_shape(source_dir):
    """`RND` exercises every new arithmetic opcode: lda abs, asl a x2,
    clc, adc abs, clc, adc #imm, sta abs, rts."""
    from pop_lifter.ir1 import (
        AdcAbs,
        AdcImm,
        Asl,
        Clc,
        LoadAbs,
        Return,
        StoreAbs,
    )

    module = _lift(source_dir, "GRAFIX.S", ["RND"])
    routine = module.find("RND")
    assert routine is not None
    kinds = [type(item).__name__ for item in routine.body]
    # The trailing `]rts:` label gets surfaced before the final `rts`.
    assert kinds[:8] == [
        "LoadAbs", "Asl", "Asl", "Clc", "AdcAbs", "Clc", "AdcImm", "StoreAbs"
    ]
    assert kinds[-1] == "Return"
    body = routine.body
    assert isinstance(body[0], LoadAbs) and body[0].source.name == "RNDseed"
    assert isinstance(body[1], Asl)
    assert isinstance(body[3], Clc)
    assert isinstance(body[4], AdcAbs) and body[4].source.name == "RNDseed"
    assert isinstance(body[6], AdcImm) and body[6].imm.value == 23
    assert isinstance(body[7], StoreAbs) and body[7].target.name == "RNDseed"
    assert isinstance(body[-1], Return)


# ---- CheckFloor slice: cmp, conditional branches, ]rts trampoline


def test_checkfloor_lift_shape(source_dir):
    """`CHECKFLOOR` exercises the new cmp/branch surface plus the
    routine-extension and `]rts` macro-return synthesis that the
    lifter needed for this slice."""
    from pop_lifter.ir1 import (
        Branch,
        CmpImm,
        Goto,
        Label,
        LoadAbs,
        Return,
    )

    module = _lift(source_dir, "CTRL.S", ["CHECKFLOOR"])
    routine = module.find("CHECKFLOOR")
    assert routine is not None

    # The synthesized `]rts:` trampoline must appear as the last two
    # items so branches to `]rts` from within the body resolve.
    assert isinstance(routine.body[-1], Return)
    assert isinstance(routine.body[-2], Label)
    assert routine.body[-2].name == "]rts"

    # The body must reach past the first `jmp onground` — `:2` is
    # defined *after* it in source order and is the entry for the
    # action-4 / action-3 paths.
    label_names = [item.name for item in routine.body if isinstance(item, Label)]
    assert ":2" in label_names
    assert ":1" in label_names
    assert ":ong" in label_names

    # Spot-check key instructions: the first lda + cmp + beq triple.
    assert isinstance(routine.body[0], LoadAbs)
    assert routine.body[0].source.name == "CharAction"
    assert isinstance(routine.body[1], CmpImm)
    assert routine.body[1].imm.value == 6
    assert isinstance(routine.body[2], Branch)
    assert routine.body[2].cond == "eq" and routine.body[2].target == "]rts"

    # One of the tail-calls is `jmp onground` — confirmed via Goto.
    gotos = [item for item in routine.body if isinstance(item, Goto)]
    assert any(g.target == "onground" and g.kind == "tail_call" for g in gotos)


def test_jsr_lifts_to_call(source_dir):
    """Every `jsr X` becomes a `Call(X)` IR node. AUTOCTRL's first
    instruction is `jsr DoRelease`, which is a stable anchor across
    refactors of the surrounding control flow."""
    from pop_lifter.ir1 import Call

    module = _lift(source_dir, "AUTO.S", ["AUTOCTRL"])
    auto = module.find("AUTOCTRL")
    assert auto is not None
    calls = [item for item in auto.body if isinstance(item, Call)]
    assert any(c.target == "DoRelease" for c in calls), (
        "AUTOCTRL must lift `jsr DoRelease` (line 161) into a Call"
    )
