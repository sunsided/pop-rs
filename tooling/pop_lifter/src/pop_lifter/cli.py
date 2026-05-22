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
from .pass0_parse import ProgramAST, parse_files
from .pass1_lift import lift_file

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

    file_path = Path(args.file)
    if not file_path.is_absolute():
        candidate = src_dir / file_path
        if candidate.is_file():
            file_path = candidate
    if not file_path.is_file():
        print(f"error: input file not found: {args.file}", file=sys.stderr)
        return 2

    # Parse the equate base (EQ.S + GAMEEQ.S) plus the target file so
    # every operand symbol the lifter encounters resolves to a concrete
    # address. The target file's own equates (e.g. AUTO.S's `flaskscrn`)
    # come in via the same parse_files call.
    base = [src_dir / n for n in ("EQ.S", "GAMEEQ.S") if (src_dir / n).exists()]
    ast = parse_files([*base, file_path], search_paths=[src_dir])

    target_str = str(file_path.resolve())
    file_ast = next(
        (f for f in ast.files if Path(f.path).resolve() == file_path.resolve()),
        None,
    )
    if file_ast is None:
        print(f"error: file {target_str} was not loaded by the parser", file=sys.stderr)
        return 1

    report = lift_file(file_ast, ast.equates, args.entry)

    text = ir1_mod.format_module(report.module)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(
            f"wrote {out_path} ({len(report.module.routines)} routines, "
            f"{len(report.unsupported)} unsupported ops)"
        )
    else:
        sys.stdout.write(text)
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
        help="Pass 1: lift selected entry-point routines from a .S file to IR1.",
    )
    p_lift.add_argument(
        "file",
        help=".S file (absolute path or relative to the source dir).",
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

    args = parser.parse_args(argv)
    return args.func(args)
