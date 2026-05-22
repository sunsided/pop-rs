"""Command-line entry point for `pop-lifter`.

Subcommands:

* `parse`     — pass 0: lex/parse Merlin sources, print equates and dum
                blocks. The default input set is `EQ.S` + `GAMEEQ.S`.
* `dump-ast`  — pass 0: write the equates / dum-block AST to disk as
                JSON, so the checked-in artifacts in `ir/pass0/` stay
                inspectable and reviewable.
* `lift`      — pass 1: lift a single `.S` file's specified entry-point
                routines to IR1 and print the result.

`dump-ir1` is folded into `lift --out` rather than a separate verb.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import ir1 as ir1_mod
from . import ir3 as ir3_mod
from .pass0_parse import ProgramAST, parse_files
from .pass1_lift import discover_entries, lift_file
from .pass2_reloop import reloop_module
from .pass2_struct import elision_stats, fusion_stats, structure_module

DEFAULT_SOURCE_REL = Path("01 POP Source/Source")


# ---------------------------------------------------------------- shared

def _resolve_source_dir(args: argparse.Namespace) -> Path | None:
    root = Path(args.source_root)
    src_dir = root / DEFAULT_SOURCE_REL
    if not src_dir.is_dir():
        print(f"error: source dir not found: {src_dir}", file=sys.stderr)
        return None
    return src_dir


def _print_parse_text(ast: ProgramAST) -> None:
    print(f"# equates ({len(ast.equates)})")
    for k, v in sorted(ast.equates.items()):
        print(f"{k:>20} = ${v:04x}")
    print(f"\n# dum blocks ({len(ast.dum_blocks)})")
    for b in ast.dum_blocks:
        named = sum(1 for f in b.fields if f.name)
        print(
            f"  {b.start_expr:>16} @ ${b.start_addr:04x} "
            f"({len(b.fields)} fields, {named} named) "
            f"[{Path(b.file).name}:{b.line}]"
        )
    if ast.diagnostics:
        print(f"\n# diagnostics ({len(ast.diagnostics)})")
        for d in ast.diagnostics:
            print(f"  {d}")


# ---------------------------------------------------------------- parse

def _cmd_parse(args: argparse.Namespace) -> int:
    src_dir = _resolve_source_dir(args)
    if src_dir is None:
        return 2

    inputs: list[Path] = []
    if args.files:
        missing: list[str] = []
        for f in args.files:
            p = Path(f)
            if not p.is_file():
                missing.append(f)
            else:
                inputs.append(p)
        if missing:
            for f in missing:
                print(f"error: input file not found: {f}", file=sys.stderr)
            return 2
    else:
        for name in ("EQ.S", "GAMEEQ.S"):
            p = src_dir / name
            if p.exists():
                inputs.append(p)

    if not inputs:
        print("error: no input files", file=sys.stderr)
        return 2

    ast = parse_files(inputs, search_paths=[src_dir])
    if args.format == "json":
        print(ast.to_json())
    else:
        _print_parse_text(ast)
    return 0


# ---------------------------------------------------------------- dump-ast

def _cmd_dump_ast(args: argparse.Namespace) -> int:
    """Persist the pass-0 AST as JSON under `ir/pass0/`.

    Defaults to dumping `EQ.S + GAMEEQ.S` because they're the
    authoritative address book the rest of the lifter feeds on. The
    intent is for the JSON to be committed: reviewers can diff symbol
    movements directly, and CI's regen test will catch any drift between
    the parser and the snapshot.
    """
    src_dir = _resolve_source_dir(args)
    if src_dir is None:
        return 2

    inputs = [src_dir / name for name in ("EQ.S", "GAMEEQ.S") if (src_dir / name).exists()]
    if not inputs:
        print("error: EQ.S / GAMEEQ.S not found in source tree", file=sys.stderr)
        return 2

    ast = parse_files(inputs, search_paths=[src_dir])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(ast.to_json() + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({len(ast.equates)} equates, "
          f"{len(ast.dum_blocks)} dum blocks)")
    return 0


# ---------------------------------------------------------------- lift

def _cmd_lift(args: argparse.Namespace) -> int:
    src_dir = _resolve_source_dir(args)
    if src_dir is None:
        return 2

    # Resolve each input file: absolute paths used as-is; bare names are
    # tried under the source dir first, then cwd.
    file_paths: list[Path] = []
    for raw in args.file:
        p = Path(raw)
        if not p.is_absolute():
            candidate = src_dir / p
            if candidate.is_file():
                p = candidate
        if not p.is_file():
            print(f"error: input file not found: {raw}", file=sys.stderr)
            return 2
        file_paths.append(p)

    # Parse the equate base (EQ.S + GAMEEQ.S) plus every target file so
    # every operand symbol the lifter encounters resolves to a concrete
    # address. Each target file's own equates flow through the same
    # parse_files call.
    base = [src_dir / n for n in ("EQ.S", "GAMEEQ.S") if (src_dir / n).exists()]
    ast = parse_files([*base, *file_paths], search_paths=[src_dir])

    # For each file, lift only those `--entry` labels that this file
    # actually defines. This lets a single CLI invocation produce a
    # cross-module dump (e.g. AUTO.S's `rndp` + GRAFIX.S's `RND`)
    # without forcing the caller to know which entry lives where.
    dumps: list[tuple[Path, int, int, str]] = []
    handled: set[str] = set()
    for file_path in file_paths:
        file_ast = next(
            (f for f in ast.files if Path(f.path).resolve() == file_path.resolve()),
            None,
        )
        if file_ast is None:
            print(
                f"error: file {file_path} was not loaded by the parser",
                file=sys.stderr,
            )
            return 1
        defined = set(discover_entries(file_ast))
        local_entries = [e for e in args.entry if e in defined and e not in handled]
        if not local_entries:
            continue
        report = lift_file(file_ast, ast.equates, local_entries)
        if not report.module.routines:
            continue
        dumps.append((
            file_path,
            len(report.module.routines),
            len(report.unsupported),
            ir1_mod.format_module(report.module),
        ))
        handled.update(local_entries)

    missing = [e for e in args.entry if e not in handled]
    if missing:
        print(
            f"error: entries not found in any input file: {missing}",
            file=sys.stderr,
        )
        return 1

    text = "\n".join(d[3] for d in dumps)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        total_routines = sum(d[1] for d in dumps)
        total_unsupp = sum(d[2] for d in dumps)
        print(
            f"wrote {out_path} ({total_routines} routines across "
            f"{len(dumps)} module(s), {total_unsupp} unsupported ops)"
        )
    else:
        sys.stdout.write(text)
    return 0


# ---------------------------------------------------------------- struct (pass 2)

def _cmd_struct(args: argparse.Namespace) -> int:
    """Run pass 1 on the given files / entries and then pass 2's
    structurer. Writes the IR2 dump to `--out` (or stdout)."""
    src_dir = _resolve_source_dir(args)
    if src_dir is None:
        return 2

    file_paths: list[Path] = []
    for raw in args.file:
        p = Path(raw)
        if not p.is_absolute():
            candidate = src_dir / p
            if candidate.is_file():
                p = candidate
        if not p.is_file():
            print(f"error: input file not found: {raw}", file=sys.stderr)
            return 2
        file_paths.append(p)

    base = [src_dir / n for n in ("EQ.S", "GAMEEQ.S") if (src_dir / n).exists()]
    ast = parse_files([*base, *file_paths], search_paths=[src_dir])

    dumps: list[str] = []
    total_fused = 0
    total_unfused = 0
    total_cmp_left = 0
    total_setc_left = 0
    total_routines = 0
    handled: set[str] = set()
    for file_path in file_paths:
        file_ast = next(
            (f for f in ast.files if Path(f.path).resolve() == file_path.resolve()),
            None,
        )
        if file_ast is None:
            print(
                f"error: file {file_path} was not loaded by the parser",
                file=sys.stderr,
            )
            return 1
        defined = set(discover_entries(file_ast))
        local_entries = [e for e in args.entry if e in defined and e not in handled]
        if not local_entries:
            continue
        ir1_module = lift_file(file_ast, ast.equates, local_entries).module
        ir2_module = structure_module(ir1_module)
        f, u = fusion_stats(ir2_module)
        cmp_left, setc_left = elision_stats(ir2_module)
        total_fused += f
        total_unfused += u
        total_cmp_left += cmp_left
        total_setc_left += setc_left
        total_routines += len(ir2_module.routines)
        dumps.append(ir1_mod.format_module(ir2_module))
        handled.update(local_entries)

    missing = [e for e in args.entry if e not in handled]
    if missing:
        print(
            f"error: entries not found in any input file: {missing}",
            file=sys.stderr,
        )
        return 1

    text = "\n".join(dumps)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(
            f"wrote {out_path} ({total_routines} routines, "
            f"{total_fused} fused-if, {total_unfused} unfused-branch, "
            f"{total_cmp_left} cmp left, "
            f"{total_setc_left} clc/sec left)"
        )
    else:
        sys.stdout.write(text)
    return 0


# ---------------------------------------------------------------- reloop (pass 2 phase 2)


def _cmd_reloop(args: argparse.Namespace) -> int:
    """Run passes 1 + 2 (fusion/elision) + reloop (CFG → structured
    IR3) for the given files / entries. Writes the IR3 dump to `--out`
    (or stdout). Routines the relooper can't structure (loops,
    malformed CFGs) fall back to an unstructured wrapping so the
    routine still appears in the output — flagged by a higher
    `unstructured` count in the summary line."""
    src_dir = _resolve_source_dir(args)
    if src_dir is None:
        return 2

    file_paths: list[Path] = []
    for raw in args.file:
        p = Path(raw)
        if not p.is_absolute():
            candidate = src_dir / p
            if candidate.is_file():
                p = candidate
        if not p.is_file():
            print(f"error: input file not found: {raw}", file=sys.stderr)
            return 2
        file_paths.append(p)

    base = [src_dir / n for n in ("EQ.S", "GAMEEQ.S") if (src_dir / n).exists()]
    ast = parse_files([*base, *file_paths], search_paths=[src_dir])

    dumps: list[str] = []
    total_routines = 0
    handled: set[str] = set()
    for file_path in file_paths:
        file_ast = next(
            (f for f in ast.files if Path(f.path).resolve() == file_path.resolve()),
            None,
        )
        if file_ast is None:
            print(
                f"error: file {file_path} was not loaded by the parser",
                file=sys.stderr,
            )
            return 1
        defined = set(discover_entries(file_ast))
        local_entries = [e for e in args.entry if e in defined and e not in handled]
        if not local_entries:
            continue
        ir1_module = lift_file(file_ast, ast.equates, local_entries).module
        ir2_module = structure_module(ir1_module)
        ir3_module = reloop_module(ir2_module)
        total_routines += len(ir3_module.routines)
        dumps.append(ir3_mod.format_module(ir3_module))
        handled.update(local_entries)

    missing = [e for e in args.entry if e not in handled]
    if missing:
        print(
            f"error: entries not found in any input file: {missing}",
            file=sys.stderr,
        )
        return 1

    text = "\n".join(dumps)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"wrote {out_path} ({total_routines} routines)")
    else:
        sys.stdout.write(text)
    return 0


# ---------------------------------------------------------------- lift-all

def _cmd_lift_all(args: argparse.Namespace) -> int:
    """Mechanical sweep: parse every `.S` in the source tree, auto-
    discover entry points, lift each file's routines to IR1, and write
    one dump per file under `--out-dir` (default `ir/raw/`). Also emits
    `SUMMARY.md` with per-file routine / Unsupported counts.

    Files that contain zero global code labels (e.g. equate-only `EQ.S`
    / `GAMEEQ.S`) are skipped entirely — they have nothing to lift.
    """
    src_dir = _resolve_source_dir(args)
    if src_dir is None:
        return 2

    files = sorted(src_dir.glob("*.S"))
    if not files:
        print(f"error: no .S files in {src_dir}", file=sys.stderr)
        return 2

    # Parse the whole tree once so cross-file equate references resolve.
    # Parsing EQ.S and GAMEEQ.S first guarantees the address book is in
    # place before any code file is reached.
    base_order = [src_dir / "EQ.S", src_dir / "GAMEEQ.S"]
    base = [p for p in base_order if p.exists()]
    others = [p for p in files if p not in base]
    ast = parse_files([*base, *others], search_paths=[src_dir])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, int, int, int]] = []  # (file, routines, instrs, unsupported)
    skipped: list[str] = []

    for src_path in files:
        file_ast = next(
            (f for f in ast.files if Path(f.path).resolve() == src_path.resolve()),
            None,
        )
        if file_ast is None:
            print(
                f"warning: {src_path.name} was not loaded by the parser; "
                f"skipping",
                file=sys.stderr,
            )
            continue

        entries = discover_entries(file_ast)
        if not entries:
            skipped.append(src_path.name)
            continue

        report = lift_file(file_ast, ast.equates, entries)
        if not report.module.routines:
            skipped.append(src_path.name)
            continue

        text = ir1_mod.format_module(report.module)
        out_path = out_dir / f"{src_path.stem.upper()}.ir1"
        out_path.write_text(text, encoding="utf-8")

        # Count items for the summary (Label items don't count as
        # "instructions").
        instrs = sum(
            1
            for r in report.module.routines
            for item in r.body
            if not isinstance(item, ir1_mod.Label)
        )
        rows.append(
            (src_path.name, len(report.module.routines), instrs, len(report.unsupported))
        )

    # Write SUMMARY.md. The table doubles as a "how complete is pass 1"
    # snapshot — reviewers can scan the Unsupported column to see which
    # files are still mostly opaque to the lifter.
    summary_lines = [
        "# Pass-1 mechanical lift — per-file summary",
        "",
        f"Generated by `pop-lifter lift-all`. {len(rows)} files lifted, "
        f"{len(skipped)} skipped (no global code labels).",
        "",
        "| File | Routines | Instructions | Unsupported |",
        "|------|---------:|-------------:|------------:|",
    ]
    total_routines = 0
    total_instrs = 0
    total_unsupp = 0
    for name, n_routines, n_instrs, n_unsupp in sorted(rows):
        summary_lines.append(
            f"| `{name}` | {n_routines} | {n_instrs} | {n_unsupp} |"
        )
        total_routines += n_routines
        total_instrs += n_instrs
        total_unsupp += n_unsupp
    summary_lines.append(
        f"| **total** | **{total_routines}** | **{total_instrs}** | **{total_unsupp}** |"
    )
    if skipped:
        summary_lines += [
            "",
            "## Skipped (no liftable entry points)",
            "",
        ]
        for name in sorted(skipped):
            summary_lines.append(f"- `{name}`")
    summary_lines.append("")
    (out_dir / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(
        f"wrote {len(rows)} .ir1 files + SUMMARY.md under {out_dir} "
        f"(total {total_routines} routines, {total_instrs} instructions, "
        f"{total_unsupp} unsupported)"
    )
    return 0


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pop-lifter")
    parser.add_argument(
        "--source-root",
        default="vendor/pop-apple2",
        help="Path to the upstream Apple II source checkout.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_parse = sub.add_parser("parse", help="Pass 0: lex/parse Merlin sources.")
    p_parse.add_argument(
        "--format", choices=("text", "json"), default="text",
        help="Output format.",
    )
    p_parse.add_argument(
        "files", nargs="*",
        help="Specific .S files to parse (default: EQ.S + GAMEEQ.S).",
    )
    p_parse.set_defaults(func=_cmd_parse)

    p_dump = sub.add_parser(
        "dump-ast",
        help="Pass 0: write equates/dum-blocks JSON to disk.",
    )
    p_dump.add_argument(
        "--out", default="ir/pass0/equates.json",
        help="Output path for the JSON dump.",
    )
    p_dump.set_defaults(func=_cmd_dump_ast)

    p_lift = sub.add_parser(
        "lift",
        help="Pass 1: lift selected entry-point routines from one or more .S "
             "files to IR1.",
    )
    p_lift.add_argument(
        "file", nargs="+",
        help=".S file (absolute path or relative to the source dir). "
             "Multiple files may be passed; each --entry is routed to the "
             "file that defines it.",
    )
    p_lift.add_argument(
        "--entry", action="append", required=True,
        help="Routine entry-point label. May be passed multiple times.",
    )
    p_lift.add_argument(
        "--out", default=None,
        help="If given, write IR1 dump here instead of stdout.",
    )
    p_lift.set_defaults(func=_cmd_lift)

    p_struct = sub.add_parser(
        "struct",
        help="Pass 2: run pass 1 on the given file(s)/entries, then "
             "fuse cmp+branch pairs into structured `if` nodes.",
    )
    p_struct.add_argument(
        "file", nargs="+",
        help=".S file (absolute path or relative to the source dir).",
    )
    p_struct.add_argument(
        "--entry", action="append", required=True,
        help="Routine entry-point label. May be passed multiple times.",
    )
    p_struct.add_argument(
        "--out", default=None,
        help="If given, write IR2 dump here instead of stdout.",
    )
    p_struct.set_defaults(func=_cmd_struct)

    p_reloop = sub.add_parser(
        "reloop",
        help="Pass 2 phase 2: run pass 1, fuse cmp+branch, elide dead "
             "flags, then reloop the goto-flow IR2 into structured IR3.",
    )
    p_reloop.add_argument(
        "file", nargs="+",
        help=".S file (absolute path or relative to the source dir).",
    )
    p_reloop.add_argument(
        "--entry", action="append", required=True,
        help="Routine entry-point label. May be passed multiple times.",
    )
    p_reloop.add_argument(
        "--out", default=None,
        help="If given, write IR3 dump here instead of stdout.",
    )
    p_reloop.set_defaults(func=_cmd_reloop)

    p_lift_all = sub.add_parser(
        "lift-all",
        help="Pass 1: sweep every .S in the source tree, auto-discover "
             "entry points, write one IR1 dump per file plus SUMMARY.md.",
    )
    p_lift_all.add_argument(
        "--out-dir", default="ir/raw",
        help="Directory to write <MODULE>.ir1 files into.",
    )
    p_lift_all.set_defaults(func=_cmd_lift_all)

    args = parser.parse_args(argv)
    return args.func(args)
