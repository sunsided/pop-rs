"""Compile-check the generated `emit-all` Rust tree under `rustc -D
warnings`.

Each emitted module file is self-contained for *state* (it defines its
own `Cpu`/`Regs`/`Flags`/`Smc` and `mod sym`), so the only externally
undefined symbols are cross-module routine *calls* — which we stub. The
check type-checks every file (`--emit=metadata`, no codegen) with all
warnings denied, including `unreachable_code`, so a regression that
emits dead code or otherwise-warning Rust fails the suite.

Skipped when `rustc` isn't on PATH (e.g. minimal CI images)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

import pytest

from pop_lifter.cli import _emit_all_artifacts

# Style lints the lifted, machine-shaped code intentionally trips. NB:
# `unreachable_code` is deliberately NOT allowed — the emitter must never
# produce dead code (see issue #53).
_ALLOWS = (
    "non_upper_case_globals", "non_snake_case", "dead_code",
    "unused_variables", "unused_mut", "unused_parens",
    "unused_assignments", "unused_comparisons",
)


def _harness(src: str) -> str:
    """Wrap one emitted module file so it type-checks standalone: allow
    the known-noisy style lints the lifted code trips (via crate-level
    `#![allow(...)]`), let `rustc -D warnings` deny every remaining
    warning, and stub the cross-module routine methods it calls but
    doesn't define."""
    defined = set(re.findall(r"fn (\w+)\(&mut self\)", src))
    called = set(re.findall(r"self\.(\w+)\(\)", src))
    stubs = sorted(called - defined)
    stub_impl = (
        "impl Cpu {\n"
        + "\n".join(f"    fn {m}(&mut self) {{}}" for m in stubs)
        + "\n}\n"
    )
    return f"#![allow({', '.join(_ALLOWS)})]\n{src}\n{stub_impl}"


@pytest.mark.skipif(shutil.which("rustc") is None, reason="rustc not on PATH")
def test_emit_all_tree_compiles_under_deny_warnings(source_dir):
    artifacts = {n: c for n, c in _emit_all_artifacts(source_dir) if n.endswith(".rs")}
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as d:
        for name, src in artifacts.items():
            path = os.path.join(d, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(_harness(src))
            result = subprocess.run(
                [
                    "rustc", "--edition", "2021", "--crate-type", "lib",
                    "--emit=metadata", "-D", "warnings",
                    "-o", os.path.join(d, name + ".rmeta"), path,
                ],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                failures.append(f"=== {name} ===\n{result.stderr.strip()}")
    assert not failures, (
        "generated Rust failed `rustc -D warnings`:\n\n" + "\n\n".join(failures)
    )
