"""Command-line entry point for `pop-lifter`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pass0_parse import parse_files

DEFAULT_SOURCE_REL = Path("01 POP Source/Source")


def _cmd_parse(args: argparse.Namespace) -> int:
    root = Path(args.source_root)
    src_dir = root / DEFAULT_SOURCE_REL
    if not src_dir.is_dir():
        print(f"error: source dir not found: {src_dir}", file=sys.stderr)
        return 2

    inputs: list[Path] = []
    if args.files:
        inputs = [Path(f) for f in args.files]
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
    return 0


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

    args = parser.parse_args(argv)
    return args.func(args)
