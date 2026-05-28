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
(no external calls / synthetic-label access); one or more per segment,
plus a few that previously exposed emitter bugs. The seeded *sweep*
(`test_seeded_sweep_no_divergence`) then exercises every routine from
deterministic random register+RAM states and asserts the crate matches
the interpreter wherever the interpreter runs to completion.

Skipped when `cargo` is absent (e.g. minimal CI images), mirroring the
`rustc` compile-check test.
"""

from __future__ import annotations

import random
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

from pop_lifter.cli import lift_all_modules
from pop_lifter.interp_ir1 import InterpError
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
    ("hires", "LayGen"),     # load-Z/N live across jsr CROP (was a divergence)
    ("master", "SetLevel"),
    ("misc", "STABCHAR"),
    ("mover", "gettimer"),
    ("mover", "GETSPIKES"),  # load-Z/N live-out (was a divergence)
    ("grafix", "CVTX"),      # stale load-Z/N caused a non-terminating loop
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


def _interp(ir3_modules, module: str, name: str, regs=(0, 0, 0, 0)):
    """Run `name` in the interpreter, resolving it in the *target* segment.
    Routine names aren't globally unique (e.g. `ADDSOUND` is in both
    `sound` and `specialk`), and the interpreter resolves an entry by the
    first module that defines it — so put the target segment first to
    match the routine the crate's `(module, name)` dispatch runs."""
    target = next((m for m in ir3_modules if m.name.lower() == module), None)
    assert target is not None, f"no segment module {module!r} in IR3 set"
    ordered = [target] + [m for m in ir3_modules if m is not target]
    a, x, y, c = regs
    return ir3_run(ordered, name, ram=bytearray(0x10000), a=a, x=x, y=y, c=c)


def _run_crate(diffrun_bin, module, name, regs=(0, 0, 0, 0)) -> tuple[int, int, int, bytes]:
    p = subprocess.run(
        [diffrun_bin, module, name, *(str(v) for v in regs)],
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


def _assert_match(diffrun_bin, ir3_modules, module, name, regs):
    trace = _interp(ir3_modules, module, name, regs)
    a, x, y, ram = _run_crate(diffrun_bin, module, name, regs)

    # Compare registers and RAM but NOT the status flags. The emitter
    # elides flag writes that are dead within a routine (sound under
    # pass-2's documented "callees don't read flag inputs" assumption),
    # so exit-time carry/Z/N can legitimately differ from the interpreter,
    # which always computes them. Registers and RAM are never elided.
    assert (a, x, y) == (trace.a, trace.x, trace.y), (
        f"{module}/{name} regs={regs}: final registers diverge — "
        f"interp=({trace.a},{trace.x},{trace.y}) crate=({a},{x},{y})"
    )
    if ram != bytes(trace.ram):
        ndiff = sum(1 for i in range(0x10000) if ram[i] != trace.ram[i])
        first = next(i for i in range(0x10000) if ram[i] != trace.ram[i])
        pytest.fail(
            f"{module}/{name} regs={regs}: RAM diverges at {ndiff} address(es); "
            f"first ${first:04x}: interp={trace.ram[first]} crate={ram[first]}"
        )


@pytest.mark.parametrize("module,name", _PILOTS, ids=[f"{m}/{n}" for m, n in _PILOTS])
def test_emitted_routine_matches_interpreter(module, name, diffrun_bin, ir3_modules):
    _assert_match(diffrun_bin, ir3_modules, module, name, (0, 0, 0, 0))


# Non-zero initial registers (incl. edge values) exercise register-
# dependent paths and indexed addressing the zero state can't reach.
_REG_SEEDS = [(0x01, 0x02, 0x03, 1), (0xff, 0xff, 0xff, 0), (0x80, 0x7f, 0x01, 1)]


@pytest.mark.parametrize("module,name", _PILOTS, ids=[f"{m}/{n}" for m, n in _PILOTS])
@pytest.mark.parametrize("regs", _REG_SEEDS, ids=lambda r: "a%02x_x%02x_y%02x_c%d" % r)
def test_emitted_routine_matches_interpreter_seeded(module, name, regs, diffrun_bin, ir3_modules):
    # Some routines aren't runnable for a given start state (a seeded
    # index reaches a synthetic-label address the interpreter rejects);
    # skip those combos and compare the rest.
    try:
        _assert_match(diffrun_bin, ir3_modules, module, name, regs)
    except InterpError as e:
        pytest.skip(f"{module}/{name} not interpretable for regs={regs}: {e}")


# ---- seeded sweep over every routine -----------------------------------

_SWEEP_RNG_SEED = 0xC0FFEE
_SWEEP_SEEDS_PER_ROUTINE = 2
_INTERP_TIMEOUT_S = 0.5  # skip routines that loop on a wild RAM seed


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


def test_seeded_sweep_no_divergence(diffrun_bin, ir3_modules):
    """Property check: from deterministic random register+RAM start states,
    the emitted crate matches the IR3 interpreter for *every* routine the
    interpreter runs to completion.

    Per (routine, seed) we skip combos the interpreter can't run — a wild
    RAM seed often steers it into a synthetic-label address or an
    unresolved (vector-dispatched) call, or an unterminating loop (cut off
    by `_INTERP_TIMEOUT_S`). Those aren't statically runnable, so they're
    out of scope. But any *completed* comparison that diverges — or a crate
    panic / hang where the interpreter finished — fails the test. The
    skip set may shift with machine speed; the 0-divergence assertion does
    not.
    """
    if not hasattr(signal, "setitimer"):
        pytest.skip("signal.setitimer unavailable (non-POSIX)")

    prev = signal.signal(signal.SIGALRM, _on_alarm)
    divergences: list[str] = []
    compared = 0
    try:
        for module in ir3_modules:
            seg = module.name.lower()
            ordered = [module] + [m for m in ir3_modules if m is not module]
            for r in module.routines:
                for k in range(_SWEEP_SEEDS_PER_ROUTINE):
                    # Per-combo RNG keyed by (segment, name, k) so each
                    # start state is reproducible in isolation and does not
                    # depend on iteration order or which earlier combos were
                    # skipped (a machine-dependent timeout must not shift the
                    # seeds of later combos).
                    rng = random.Random(f"{_SWEEP_RNG_SEED}:{seg}:{r.name}:{k}")
                    a, x, y, c = (rng.randrange(256), rng.randrange(256),
                                  rng.randrange(256), rng.randrange(2))
                    ram0 = rng.randbytes(0x10000)
                    signal.setitimer(signal.ITIMER_REAL, _INTERP_TIMEOUT_S)
                    try:
                        trace = ir3_run(ordered, r.name, ram=bytearray(ram0),
                                        a=a, x=x, y=y, c=c)
                    except (_Timeout, InterpError):
                        # Interpreter can't run this routine from this state
                        # (synthetic-label access, unresolved vector call, or
                        # a loop past the timeout). Out of scope. Any *other*
                        # exception propagates and fails the test — it'd be a
                        # real interpreter/harness regression, not a skip.
                        continue
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)

                    tag = f"{seg}/{r.name} a={a} x={x} y={y} c={c}"
                    try:
                        p = subprocess.run(
                            [diffrun_bin, seg, r.name, str(a), str(x), str(y), str(c)],
                            input=ram0, capture_output=True, timeout=5,
                        )
                    except subprocess.TimeoutExpired:
                        # Crate looped where the interpreter finished — a
                        # termination divergence.
                        divergences.append(f"{tag}: crate hang (interp completed)")
                        continue
                    if p.returncode != 0:
                        # 3 = panic (e.g. an OOB index the interp wrapped),
                        # 2 = unknown dispatch; both are real divergences. The
                        # crate's panic message (if any) lands on stderr.
                        err = p.stderr.decode(errors="replace").strip()
                        divergences.append(
                            f"{tag}: diffrun exit {p.returncode}"
                            + (f" — {err}" if err else "")
                        )
                        continue
                    out = p.stdout
                    compared += 1
                    if ((out[0], out[1], out[2]) != (trace.a, trace.x, trace.y)
                            or bytes(out[4:]) != bytes(trace.ram)):
                        divergences.append(f"{tag}: state diverges")
    finally:
        signal.signal(signal.SIGALRM, prev)

    assert compared > 0, "seeded sweep compared nothing — harness/lift broken?"
    assert not divergences, (
        f"{len(divergences)} seeded divergence(s) over {compared} comparisons:\n"
        + "\n".join(divergences[:25])
    )
