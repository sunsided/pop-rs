"""Cross-platform portability of path strings in the lifter dumps.

`ProgramAST.to_json`, `SourceRef.short`, and `ir1._portable_path` all
rewrite source paths so the committed artifacts diff cleanly across
checkouts. They must work for both POSIX-style (`/`) and Windows-style
(`\\`) inputs.
"""

from __future__ import annotations

from pop_lifter.ir1 import SourceRef, _portable_path


def test_portable_path_strips_posix_vendor_prefix():
    assert _portable_path(
        "/home/x/pop-rs/vendor/pop-apple2/01 POP Source/Source/AUTO.S"
    ) == "01 POP Source/Source/AUTO.S"


def test_portable_path_strips_windows_vendor_prefix():
    assert _portable_path(
        r"C:\Users\x\pop-rs\vendor\pop-apple2\01 POP Source\Source\AUTO.S"
    ) == "01 POP Source/Source/AUTO.S"


def test_portable_path_falls_back_to_basename_posix():
    assert _portable_path("/tmp/standalone.S") == "standalone.S"


def test_portable_path_falls_back_to_basename_windows():
    assert _portable_path(r"C:\tmp\standalone.S") == "standalone.S"


def test_source_ref_short_handles_windows_separator():
    ref = SourceRef(
        file=r"C:\Users\x\pop-rs\vendor\pop-apple2\01 POP Source\Source\AUTO.S",
        line=42,
        raw=" lda #0",
    )
    assert ref.short() == "AUTO.S:42"
