"""Regen test for the checked-in pass-0 / pass-1 artifacts under `ir/`.

The lifter writes its intermediate output to disk so reviewers can
inspect (and diff) the generated AST and IR1 dumps directly. This test
re-generates the artifacts in-memory and asserts they match the
committed bytes — any drift between the lifter and the snapshot fails
CI, with a clear hint on how to regenerate.
"""

from __future__ import annotations

from pathlib import Path

from pop_lifter import ir1 as ir1_mod
from pop_lifter.pass0_parse import parse_files
from pop_lifter.pass1_lift import discover_entries, lift_file

REPO_ROOT = Path(__file__).resolve().parents[3]
IR_PASS0 = REPO_ROOT / "ir" / "pass0" / "equates.json"
IR_PILOT = REPO_ROOT / "ir" / "pilot" / "auto_combat.ir1"
IR_PILOT_RNDP = REPO_ROOT / "ir" / "pilot" / "rndp.ir1"
IR_PILOT_CHECKFLOOR = REPO_ROOT / "ir" / "pilot" / "checkfloor.ir1"
IR_RAW = REPO_ROOT / "ir" / "raw"

PILOT_ENTRIES = [
    "DoStrike", "DoBlock", "DoTurn",
    "DoStandup", "DoEngarde", "DoRelBtn", "DoRelease",
]


def _regen_equates_json(source_dir: Path) -> str:
    ast = parse_files(
        [source_dir / "EQ.S", source_dir / "GAMEEQ.S"],
        search_paths=[source_dir],
    )
    return ast.to_json() + "\n"


def _regen_auto_combat(source_dir: Path) -> str:
    ast = parse_files(
        [source_dir / "EQ.S", source_dir / "GAMEEQ.S", source_dir / "AUTO.S"],
        search_paths=[source_dir],
    )
    file_ast = next(f for f in ast.files if Path(f.path).name == "AUTO.S")
    report = lift_file(file_ast, ast.equates, PILOT_ENTRIES)
    return ir1_mod.format_module(report.module)


def test_pass0_equates_artifact_matches(source_dir):
    if not IR_PASS0.exists():
        # If the artifact is missing entirely, point the reviewer at the
        # regen command rather than emitting a generic failure.
        raise AssertionError(
            f"missing artifact {IR_PASS0}. regenerate with:\n"
            f"  pop-lifter dump-ast --out {IR_PASS0.relative_to(REPO_ROOT)}"
        )
    expected = IR_PASS0.read_text(encoding="utf-8")
    actual = _regen_equates_json(source_dir)
    assert actual == expected, (
        f"{IR_PASS0.relative_to(REPO_ROOT)} is stale. regenerate with:\n"
        f"  pop-lifter dump-ast --out {IR_PASS0.relative_to(REPO_ROOT)}"
    )


def test_pass1_pilot_artifact_matches(source_dir):
    if not IR_PILOT.exists():
        raise AssertionError(
            f"missing artifact {IR_PILOT}. regenerate with:\n"
            f"  pop-lifter lift AUTO.S "
            f"{' '.join('--entry ' + e for e in PILOT_ENTRIES)} "
            f"--out {IR_PILOT.relative_to(REPO_ROOT)}"
        )
    expected = IR_PILOT.read_text(encoding="utf-8")
    actual = _regen_auto_combat(source_dir)
    assert actual == expected, (
        f"{IR_PILOT.relative_to(REPO_ROOT)} is stale. regenerate with:\n"
        f"  pop-lifter lift AUTO.S "
        f"{' '.join('--entry ' + e for e in PILOT_ENTRIES)} "
        f"--out {IR_PILOT.relative_to(REPO_ROOT)}"
    )


def _regen_rndp_pilot(source_dir: Path) -> str:
    """Reproduce `pop-lifter lift AUTO.S GRAFIX.S --entry rndp --entry RND`.

    Builds the cross-module rndp/RND dump as a concatenation of the two
    module dumps separated by a single blank line, matching what the
    CLI writes.
    """
    ast = parse_files(
        [
            source_dir / "EQ.S",
            source_dir / "GAMEEQ.S",
            source_dir / "AUTO.S",
            source_dir / "GRAFIX.S",
        ],
        search_paths=[source_dir],
    )
    auto = next(f for f in ast.files if Path(f.path).name == "AUTO.S")
    grafix = next(f for f in ast.files if Path(f.path).name == "GRAFIX.S")
    m_auto = lift_file(auto, ast.equates, ["rndp"]).module
    m_grafix = lift_file(grafix, ast.equates, ["RND"]).module
    return "\n".join([
        ir1_mod.format_module(m_auto),
        ir1_mod.format_module(m_grafix),
    ])


def test_pass1_rndp_pilot_artifact_matches(source_dir):
    if not IR_PILOT_RNDP.exists():
        raise AssertionError(
            f"missing artifact {IR_PILOT_RNDP}. regenerate with:\n"
            f"  pop-lifter lift AUTO.S GRAFIX.S --entry rndp --entry RND "
            f"--out {IR_PILOT_RNDP.relative_to(REPO_ROOT)}"
        )
    expected = IR_PILOT_RNDP.read_text(encoding="utf-8")
    actual = _regen_rndp_pilot(source_dir)
    assert actual == expected, (
        f"{IR_PILOT_RNDP.relative_to(REPO_ROOT)} is stale. regenerate with:\n"
        f"  pop-lifter lift AUTO.S GRAFIX.S --entry rndp --entry RND "
        f"--out {IR_PILOT_RNDP.relative_to(REPO_ROOT)}"
    )


def _regen_checkfloor_pilot(source_dir: Path) -> str:
    ast = parse_files(
        [
            source_dir / "EQ.S",
            source_dir / "GAMEEQ.S",
            source_dir / "CTRL.S",
        ],
        search_paths=[source_dir],
    )
    ctrl = next(f for f in ast.files if Path(f.path).name == "CTRL.S")
    report = lift_file(ctrl, ast.equates, ["CHECKFLOOR"])
    return ir1_mod.format_module(report.module)


def test_pass1_checkfloor_pilot_artifact_matches(source_dir):
    if not IR_PILOT_CHECKFLOOR.exists():
        raise AssertionError(
            f"missing artifact {IR_PILOT_CHECKFLOOR}. regenerate with:\n"
            f"  pop-lifter lift CTRL.S --entry CHECKFLOOR "
            f"--out {IR_PILOT_CHECKFLOOR.relative_to(REPO_ROOT)}"
        )
    expected = IR_PILOT_CHECKFLOOR.read_text(encoding="utf-8")
    actual = _regen_checkfloor_pilot(source_dir)
    assert actual == expected, (
        f"{IR_PILOT_CHECKFLOOR.relative_to(REPO_ROOT)} is stale. "
        f"regenerate with:\n"
        f"  pop-lifter lift CTRL.S --entry CHECKFLOOR "
        f"--out {IR_PILOT_CHECKFLOOR.relative_to(REPO_ROOT)}"
    )


def _regen_raw_lift(source_dir: Path) -> dict[str, str]:
    """Reproduce what `pop-lifter lift-all` writes — one IR1 dump per
    code file, keyed by output filename (e.g. `AUTO.ir1`)."""
    files = sorted(source_dir.glob("*.S"))
    base_order = [source_dir / "EQ.S", source_dir / "GAMEEQ.S"]
    base = [p for p in base_order if p.exists()]
    others = [p for p in files if p not in base]
    ast = parse_files([*base, *others], search_paths=[source_dir])

    out: dict[str, str] = {}
    for src_path in files:
        file_ast = next(
            (f for f in ast.files if Path(f.path).resolve() == src_path.resolve()),
            None,
        )
        if file_ast is None:
            continue
        entries = discover_entries(file_ast)
        if not entries:
            continue
        report = lift_file(file_ast, ast.equates, entries)
        if not report.module.routines:
            continue
        out[f"{src_path.stem.upper()}.ir1"] = ir1_mod.format_module(report.module)
    return out


def test_pass1_raw_artifacts_match(source_dir):
    """The full per-file IR1 sweep under `ir/raw/` is the most direct
    "what does pass 1 see across the whole tree" view. Regenerating it
    must match what's checked in, byte for byte."""
    if not IR_RAW.is_dir():
        raise AssertionError(
            f"missing artifact dir {IR_RAW}. regenerate with:\n"
            f"  pop-lifter lift-all --out-dir {IR_RAW.relative_to(REPO_ROOT)}"
        )

    actual = _regen_raw_lift(source_dir)
    expected = {
        p.name: p.read_text(encoding="utf-8")
        for p in IR_RAW.glob("*.ir1")
    }

    extra_on_disk = set(expected) - set(actual)
    missing_on_disk = set(actual) - set(expected)
    regen_cmd = (
        f"  pop-lifter lift-all --out-dir {IR_RAW.relative_to(REPO_ROOT)}"
    )
    assert not extra_on_disk, (
        f"{sorted(extra_on_disk)} present in {IR_RAW} but not in regen output. "
        f"regenerate with:\n{regen_cmd}"
    )
    assert not missing_on_disk, (
        f"{sorted(missing_on_disk)} produced by regen but not in {IR_RAW}. "
        f"regenerate with:\n{regen_cmd}"
    )
    stale = sorted(name for name, body in actual.items() if expected[name] != body)
    assert not stale, (
        f"{stale} are stale under {IR_RAW.relative_to(REPO_ROOT)}. "
        f"regenerate with:\n{regen_cmd}"
    )
