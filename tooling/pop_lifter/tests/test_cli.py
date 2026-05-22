"""CLI surface tests. Pinned to user-visible behavior: exit codes and
stderr lines, not internal call shapes."""

from __future__ import annotations

from pop_lifter.cli import main


def test_missing_explicit_file_reports_and_exits_nonzero(tmp_path, capsys):
    # `--source-root` points at a directory that *has* the expected layout
    # so the CLI gets past its directory check and then validates the
    # explicit file arguments.
    src = tmp_path / "01 POP Source" / "Source"
    src.mkdir(parents=True)
    bogus = tmp_path / "does_not_exist.S"

    rc = main(["--source-root", str(tmp_path), "parse", str(bogus)])

    assert rc == 2
    err = capsys.readouterr().err
    assert "input file not found" in err
    assert str(bogus) in err


def test_partial_missing_files_reports_all_and_exits_nonzero(tmp_path, capsys):
    src = tmp_path / "01 POP Source" / "Source"
    src.mkdir(parents=True)
    real = tmp_path / "real.S"
    real.write_text("FOO = 1\n")
    bogus = tmp_path / "missing.S"

    rc = main(["--source-root", str(tmp_path), "parse", str(real), str(bogus)])

    # Even though one file exists, the CLI should refuse the run and
    # surface every missing file rather than silently lifting a partial
    # set.
    assert rc == 2
    err = capsys.readouterr().err
    assert str(bogus) in err
