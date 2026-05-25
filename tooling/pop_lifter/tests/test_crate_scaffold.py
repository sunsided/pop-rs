"""Regen + compile tests for the committed assembled crate under
`ir/crate/` (issue #47).

`emit-crate` assembles the whole lifted program into one coherent crate:
a shared `cpu` module (the single `Cpu` state), a shared `sym` module
(address constants), an `ext` module of external-call stubs, and one
module per POP source segment holding its routines as free functions
over `&mut Cpu`, plus `Cargo.toml`. Overlay name reuse is resolved per
calling module, so cross-segment same-named routines coexist.

* The regen test pins the tree byte-for-byte, like the other `ir/`
  artifacts.
* The compile test type-checks the whole crate through `src/lib.rs`
  under `rustc -D warnings` (skipped when rustc is absent) — not just
  each file, but the assembled crate with its cross-module call paths.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
IR_CRATE = REPO_ROOT / "ir" / "crate"


def _disk_tree(root: Path) -> dict[str, str]:
    """Every file under `root`, keyed by `/`-joined path relative to
    `root` (matching the keys `_emit_crate_artifacts` returns)."""
    out: dict[str, str] = {}
    for p in root.rglob("*"):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = p.read_text(encoding="utf-8")
    return out


def test_emit_crate_matches(source_dir):
    from pop_lifter.cli import _emit_crate_artifacts

    if not IR_CRATE.is_dir():
        raise AssertionError(
            f"missing artifact dir {IR_CRATE}. regenerate with:\n"
            f"  pop-lifter emit-crate --out-dir {IR_CRATE.relative_to(REPO_ROOT)}"
        )

    actual = _emit_crate_artifacts(source_dir)
    expected = _disk_tree(IR_CRATE)

    regen_cmd = f"  pop-lifter emit-crate --out-dir {IR_CRATE.relative_to(REPO_ROOT)}"
    extra_on_disk = set(expected) - set(actual)
    missing_on_disk = set(actual) - set(expected)
    assert not extra_on_disk, (
        f"{sorted(extra_on_disk)} present in {IR_CRATE} but not in regen output. "
        f"regenerate with:\n{regen_cmd}"
    )
    assert not missing_on_disk, (
        f"{sorted(missing_on_disk)} produced by regen but not in {IR_CRATE}. "
        f"regenerate with:\n{regen_cmd}"
    )
    stale = sorted(name for name, body in actual.items() if expected[name] != body)
    assert not stale, (
        f"{stale} are stale under {IR_CRATE.relative_to(REPO_ROOT)}. "
        f"regenerate with:\n{regen_cmd}"
    )


@pytest.mark.skipif(shutil.which("rustc") is None, reason="rustc not on PATH")
def test_emit_crate_compiles_under_deny_warnings(source_dir):
    """Type-check the whole assembled crate via `src/lib.rs` (rustc
    resolves the `mod` declarations to the sibling files). Denies all
    warnings, so the shared `Cpu`/`Smc`/`sym` defs, the free-function
    bodies, and every cross-module call path must be clean as one crate."""
    from pop_lifter.cli import _emit_crate_artifacts

    artifacts = _emit_crate_artifacts(source_dir)
    with tempfile.TemporaryDirectory() as d:
        for rel, content in artifacts.items():
            path = os.path.join(d, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        result = subprocess.run(
            [
                "rustc", "--edition", "2021", "--crate-type", "lib",
                "--emit=metadata", "-D", "warnings",
                "-o", os.path.join(d, "pop.rmeta"),
                os.path.join(d, "src", "lib.rs"),
            ],
            capture_output=True, text=True,
        )
    assert result.returncode == 0, (
        "crate scaffold failed `rustc -D warnings`:\n\n" + result.stderr.strip()
    )
