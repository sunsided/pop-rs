"""Command-line entry point. Stub."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pop-lifter")
    parser.add_argument("--source-root", default="vendor/pop-apple2",
                        help="Path to the upstream Apple II source checkout.")
    parser.add_argument("--pass", dest="which_pass",
                        choices=["lex", "parse", "lift", "struct", "domain", "rust"],
                        required=True)
    parser.parse_args(argv)
    raise SystemExit("pop-lifter is not implemented yet")
