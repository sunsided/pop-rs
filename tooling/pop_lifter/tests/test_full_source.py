"""Regression guard: parse every `.S` file in the upstream source and pin
the set of expected-deferred diagnostics."""

from __future__ import annotations

from pop_lifter.pass0_parse import parse_files


# Reasons we expect a Pass 0 diagnostic on a given source location. The
# location-counter equates (`*`-relative / label-difference table sizes)
# that used to land here now resolve via PC tracking. The two conditional
# `put ryellow1` / `put purple` includes that used to fail here are inside
# disabled `do ThreeFive ... else` blocks (ThreeFive=1), so conditional
# assembly now excludes them before any resolution is attempted — what
# remains is one genuinely-undefined symbol.
_EXPECTED_DIAGNOSTICS = {
    ("MISC.S", 150, "equate eval failed: undefined symbol: MOVEMEM"),
}


def test_full_source_parse(source_dir):
    files = sorted(source_dir.glob("*.S"))
    ast = parse_files(files, search_paths=[source_dir])

    # Coarse sanity bounds: the lifter should keep finding ~1.5k equates
    # and ~66 dum blocks. Tighten as later passes evolve.
    assert len(files) == 29
    assert 1400 <= len(ast.equates) <= 1600
    assert 60 <= len(ast.dum_blocks) <= 80

    actual = set()
    for d in ast.diagnostics:
        # Format: "<path>:<line>: <message> (<raw>)"
        path, rest = d.split(":", 1)
        lineno, rest = rest.split(":", 1)
        # Strip the trailing "(<raw>)" annotation
        msg = rest.split(" (", 1)[0].strip()
        filename = path.rsplit("/", 1)[-1]
        actual.add((filename, int(lineno), msg))

    unexpected = actual - _EXPECTED_DIAGNOSTICS
    missing = _EXPECTED_DIAGNOSTICS - actual
    assert not unexpected, f"new diagnostics not previously seen: {unexpected}"
    assert not missing, f"expected diagnostics no longer triggered: {missing}"
