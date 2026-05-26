"""Differential test: the emitted Rust crate vs. the IR3 interpreter.

The lifter's IR1↔IR3 interpreters are already cross-checked, but nothing
verified that the *emitted Rust* behaves like them — it only type-checks.
This runs a representative routine per segment through both the IR3
interpreter and the generated crate (via the `diffrun` harness binary,
which invokes a routine by name through `pop::dispatch`) from the same
zero initial state, and asserts the final registers + 64 KiB of RAM
agree. Status flags (carry/Z/N) are intentionally excluded: the emitter
elides dead flag writes, so exit-time flags can legitimately differ from
the interpreter (which always computes them) — registers and RAM are
never elided.

Pilots are routines confirmed to run to completion in the interpreter
(no external calls / synthetic-label access) and to match the crate; one
or more per segment. A broader sweep finds ~380/411 runnable routines
already agree — the ~31 divergences are tracked separately for
investigation.

Skipped when `cargo` is absent (e.g. minimal CI images), mirroring the
`rustc` compile-check test.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from pop_lifter.cli import lift_all_modules
from pop_lifter.interp_ir3 import run as ir3_run

REPO_ROOT = Path(__file__).resolve().parents[3]

# (segment module, source routine name) — a spread across every segment,
# each verified to match the crate exactly from the zero state.
_PILOTS = [
    ("auto", "AUTOCTRL"),
    ("boot", "CHECKER"),
    ("coll", "collide"),
    ("ctrl", "CHECKFLOOR"),
    ("ctrlsubs", "GETLEFT"),
    ("ctrlsubs", "INDEXBLOCK"),
    ("ctrlsubs", "GETDIST"),  # exercises jsr-then-callee-tail-call
    ("frameadv", "getprev"),
    ("gamebg", "getlevelno"),
    ("grafix", "RND"),
    ("grafix", "cvtpdl"),
    ("hires", "CROP"),
    ("master", "SetLevel"),
    ("misc", "STABCHAR"),
    ("mover", "gettimer"),
    ("sound", "ADDSOUND"),
    ("specialk", "addkey"),
    ("subs", "GRAVITY"),
    ("topctrl", "initgame"),
    ("unpack", "SNGEXPAND"),
]


@pytest.fixture(scope="session")
def diffrun_bin(source_dir) -> str:
    if shutil.which("cargo") is None:
        pytest.skip("cargo not on PATH")
    build = subprocess.run(
        ["cargo", "build", "-p", "pop-difftest", "--quiet"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    # The toolchain is present, so a build failure is a real regression in
    # the harness / crate wiring — fail rather than hide it behind a skip.
    if build.returncode != 0:
        pytest.fail(f"pop-difftest build failed:\n{build.stderr}")
    path = REPO_ROOT / "target" / "debug" / "diffrun"
    if not path.exists():
        pytest.fail(f"diffrun binary missing at {path}")
    return str(path)


@pytest.fixture(scope="session")
def ir3_modules(source_dir):
    return lift_all_modules(source_dir)


def _interp(ir3_modules, module: str, name: str):
    """Run `name` in the interpreter, resolving it in the *target* segment.
    Routine names aren't globally unique (e.g. `ADDSOUND` is in both
    `sound` and `specialk`), and the interpreter resolves an entry by the
    first module that defines it — so put the target segment first to
    match the routine the crate's `(module, name)` dispatch runs."""
    target = next((m for m in ir3_modules if m.name.lower() == module), None)
    assert target is not None, f"no segment module {module!r} in IR3 set"
    ordered = [target] + [m for m in ir3_modules if m is not target]
    return ir3_run(ordered, name, ram=bytearray(0x10000))


def _run_crate(diffrun_bin: str, module: str, name: str) -> tuple[int, int, int, bytes]:
    p = subprocess.run(
        [diffrun_bin, module, name, "0", "0", "0", "0"],
        input=b"", capture_output=True, timeout=15,
    )
    assert p.returncode == 0, (
        f"diffrun {module}/{name} exited {p.returncode}: {p.stderr.decode(errors='replace')}"
    )
    out = p.stdout
    assert len(out) == 4 + 0x10000, f"short diffrun output: {len(out)} bytes"
    # out[3] is the carry byte; flags are intentionally not compared — see
    # the assertion below.
    return out[0], out[1], out[2], bytes(out[4:])


@pytest.mark.parametrize("module,name", _PILOTS, ids=[f"{m}/{n}" for m, n in _PILOTS])
def test_emitted_routine_matches_interpreter(module, name, diffrun_bin, ir3_modules):
    trace = _interp(ir3_modules, module, name)
    a, x, y, ram = _run_crate(diffrun_bin, module, name)

    # Compare registers and RAM but NOT the status flags. The emitter
    # elides flag writes that are dead within a routine (sound under
    # pass-2's documented "callees don't read flag inputs" assumption),
    # so exit-time carry/Z/N can legitimately differ from the interpreter,
    # which always computes them. Registers and RAM are never elided.
    assert (a, x, y) == (trace.a, trace.x, trace.y), (
        f"{module}/{name}: final registers diverge — "
        f"interp=({trace.a},{trace.x},{trace.y}) crate=({a},{x},{y})"
    )
    if ram != bytes(trace.ram):
        ndiff = sum(1 for i in range(0x10000) if ram[i] != trace.ram[i])
        first = next(i for i in range(0x10000) if ram[i] != trace.ram[i])
        pytest.fail(
            f"{module}/{name}: RAM diverges at {ndiff} address(es); "
            f"first ${first:04x}: interp={trace.ram[first]} crate={ram[first]}"
        )
