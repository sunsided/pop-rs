"""Pass 0a: line-oriented lexer for Merlin 16+ source.

Merlin's grammar is column-aware: a non-whitespace first character starts a
label; otherwise the line has no label. After (optional) label/whitespace
the next token is the mnemonic (an opcode, a pseudo-op, or `=` for an
equate). Anything after the mnemonic up to a `;` or end-of-line is the
operand. Lines beginning with `*` are full-line comments.

This module produces a flat `Line` per source line; expression evaluation
and semantic interpretation happen in `pass0_parse`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Line:
    """One source line, split into Merlin's column-positional fields."""

    file: Path
    lineno: int
    raw: str
    label: str | None
    mnemonic: str | None
    operand: str | None
    comment: str | None

    @property
    def is_blank(self) -> bool:
        return self.label is None and self.mnemonic is None

    @property
    def is_equate(self) -> bool:
        return self.mnemonic == "="


def lex_line(file: Path, lineno: int, raw: str) -> Line:
    text = raw.rstrip("\n").rstrip("\r")

    # Full-line comment. Merlin treats a column-1 `*` as a banner comment;
    # a column-1 `;` is also a comment in practice in this codebase (used
    # for annotating register usage above a routine).
    if text.startswith("*"):
        return Line(file, lineno, raw, None, None, None, text[1:].strip())
    if text.startswith(";"):
        return Line(file, lineno, raw, None, None, None, text[1:].strip())

    # Label is present iff column 1 is non-whitespace.
    label: str | None = None
    rest = text
    if text and not text[0].isspace():
        head, _, tail = text.partition(" ")
        # Tabs also delimit; partition on the first run of whitespace.
        if "\t" in head:
            head, _, tail2 = head.partition("\t")
            tail = tail2 + tail
        label = head
        rest = tail

    rest = rest.lstrip()

    # Trailing comment: `;` outside of a string. Merlin source in this
    # project never uses `;` inside strings in equate/data positions we
    # care about for pass 0, so a plain split is safe.
    comment: str | None = None
    if ";" in rest:
        rest, _, comment = rest.partition(";")
        rest = rest.rstrip()
        comment = comment.strip()

    if not rest:
        return Line(file, lineno, raw, label, None, None, comment)

    # The equate form `LABEL = EXPR` puts the `=` where the mnemonic
    # normally lives.
    if rest.startswith("="):
        return Line(file, lineno, raw, label, "=", rest[1:].strip() or None, comment)

    # Otherwise split mnemonic from operand on the first run of whitespace.
    parts = rest.split(None, 1)
    mnemonic = parts[0].lower()
    operand = parts[1].strip() if len(parts) == 2 else None
    return Line(file, lineno, raw, label, mnemonic, operand, comment)


def lex_file(path: Path) -> list[Line]:
    """Lex a single Merlin `.S` file into a list of `Line`."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return [lex_line(path, i + 1, raw) for i, raw in enumerate(text.splitlines())]
