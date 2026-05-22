"""IR1 interpreter tests: execute the lifted pilot routines and check
the resulting RAM diff.

This is the pass-1 half of the differential test harness from
`docs/architecture.md`. A subsequent slice will replace the
hand-computed expected diffs here with snapshots from an emulator
harness; until then the values are pinned against what each routine
*literally* writes per its source.
"""

from __future__ import annotations

from pathlib import Path

from pop_lifter.interp_ir1 import run
from pop_lifter.pass0_parse import parse_files
from pop_lifter.pass1_lift import lift_file


# Addresses come from the equate files. Pinned here as concrete bytes so
# a regression in pass-0 equate resolution is also caught.
CLR_F = 0xD4
CLR_B = 0xD5
CLR_U = 0xD6
CLR_D = 0xD7
CLR_BTN = 0xD8
JSTKX = 0x18
JSTKY = 0x19
BTN = 0x3D

PILOT_ENTRIES = [
    "DoStrike", "DoBlock", "DoTurn",
    "DoStandup", "DoEngarde", "DoRelBtn", "DoRelease",
]


def _module(source_dir: Path):
    auto = source_dir / "AUTO.S"
    ast = parse_files(
        [source_dir / "EQ.S", source_dir / "GAMEEQ.S", auto],
        search_paths=[source_dir],
    )
    file_ast = next(f for f in ast.files if Path(f.path).name == "AUTO.S")
    return lift_file(file_ast, ast.equates, PILOT_ENTRIES).module


def _run_pilot(source_dir, entry):
    module = _module(source_dir)
    return run(module, entry)


def test_dostrike_writes_ff_to_clrbtn_and_btn(source_dir):
    trace = _run_pilot(source_dir, "DoStrike")
    assert trace.writes == {CLR_BTN: 0xff, BTN: 0xff}
    assert trace.a == 0xff


def test_dopress_alias_executes_dostrike(source_dir):
    # The alias must resolve to the same routine and produce the same
    # effects.
    a = _run_pilot(source_dir, "DoStrike").writes
    b = _run_pilot(source_dir, "DoPress").writes
    assert a == b


def test_doblock_writes_ff_to_clru_and_jstky(source_dir):
    trace = _run_pilot(source_dir, "DoBlock")
    assert trace.writes == {CLR_U: 0xff, JSTKY: 0xff}


def test_doturn_writes_ff_to_clrd_then_1_to_jstky(source_dir):
    trace = _run_pilot(source_dir, "DoTurn")
    # Two distinct values written; the JSTKY=1 store happens after the
    # second lda reloads A.
    assert trace.writes == {CLR_D: 0xff, JSTKY: 0x01}
    assert trace.a == 0x01


def test_dostandup_tail_calls_into_doback(source_dir):
    # DoStandup writes clrU=0xff itself, then tail-calls DoBack which
    # writes clrB=0xff and JSTKX=0x01.
    trace = _run_pilot(source_dir, "DoStandup")
    assert trace.writes == {
        CLR_U: 0xff,
        CLR_B: 0xff,
        JSTKX: 0x01,
    }


def test_doengarde_tail_calls_into_dofwd(source_dir):
    # DoEngarde writes clrD=0xff, then tail-calls DoFwd which writes
    # clrF=0xff and JSTKX=0xff.
    trace = _run_pilot(source_dir, "DoEngarde")
    assert trace.writes == {
        CLR_D: 0xff,
        CLR_F: 0xff,
        JSTKX: 0xff,
    }


def test_dorelbtn_zeroes_btn_only(source_dir):
    trace = _run_pilot(source_dir, "DoRelBtn")
    # The `]rts` label is just a marker; only the explicit store happens.
    assert trace.writes == {BTN: 0x00}
    assert trace.a == 0x00


def test_dorelease_zeroes_eight_buttons(source_dir):
    trace = _run_pilot(source_dir, "DoRelease")
    assert trace.writes == {
        CLR_F: 0x00, CLR_B: 0x00, CLR_U: 0x00, CLR_D: 0x00,
        CLR_BTN: 0x00, JSTKX: 0x00, JSTKY: 0x00, BTN: 0x00,
    }


def test_ram_outside_writes_is_untouched(source_dir):
    trace = _run_pilot(source_dir, "DoStrike")
    # Verify the interpreter doesn't scribble outside the addresses it
    # claims to have written. A simple checksum suffices.
    untouched = sum(trace.ram[i] for i in range(0x10000) if i not in trace.writes)
    assert untouched == 0
