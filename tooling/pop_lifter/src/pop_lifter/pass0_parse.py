"""Pass 0b: parse Merlin lines into a `ProgramAST`.

The parser handles the subset of Merlin 16+ needed to lift equate files
(`EQ.S`, `GAMEEQ.S`) and the equate-style headers of code files:

* `LABEL = EXPR` equates (with arithmetic on previously-defined symbols)
* `dum ADDR ... dend` overlay blocks, including nested / chained blocks
  where a new `dum` implicitly ends the previous one
* `ds N` inside a `dum` block: advance the location counter by `N`,
  defining `LABEL` (if any) at the current location
* `put NAME` includes (resolved as `NAME.S` in the current file's dir,
  case-insensitive)
* `lst on/off`, `tr on/off`, `xc`, `mx` are recognized and ignored

Other directives and opcodes are recorded but not evaluated; they remain
as raw `Line` objects in `FileAST.lines` for later passes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .pass0_lex import Line, lex_file


# ---------------------------------------------------------------- AST nodes


@dataclass
class Field:
    """One entry inside a `dum` block."""

    name: str | None
    offset: int
    size: int
    line: int


@dataclass
class DumBlock:
    """A Merlin `dum addr ... dend` overlay region.

    Each named `ds N` inside the block also produces a global equate
    (`name = start_addr + offset`); the block itself preserves the
    grouping so later passes can lift it to a struct or schema.
    """

    start_addr: int
    start_expr: str
    file: str
    line: int
    fields: list[Field] = field(default_factory=list)


@dataclass
class FileAST:
    path: str
    lines: list[Line] = field(default_factory=list)


@dataclass
class ProgramAST:
    equates: dict[str, int] = field(default_factory=dict)
    dum_blocks: list[DumBlock] = field(default_factory=list)
    files: list[FileAST] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        # File paths are rewritten to be repo-relative so the JSON dump
        # diffs cleanly across checkouts. The marker is the submodule
        # mount point under `vendor/pop-apple2/`; anything outside that
        # tree falls back to the basename, which is good enough for
        # ad-hoc test fixtures created in tmp dirs.
        def _portable(s: str) -> str:
            marker = "vendor/pop-apple2/"
            idx = s.find(marker)
            if idx >= 0:
                return s[idx + len(marker):]
            return s.rsplit("/", 1)[-1]

        def _default(obj: object) -> object:
            if isinstance(obj, Path):
                return _portable(str(obj))
            if isinstance(obj, Line):
                return {
                    "file": _portable(str(obj.file)),
                    "lineno": obj.lineno,
                    "label": obj.label,
                    "mnemonic": obj.mnemonic,
                    "operand": obj.operand,
                }
            raise TypeError(f"unhandled: {type(obj)}")

        # `asdict` on the dum blocks materialises a dict, so the `file`
        # field reaches `_default` as a plain string — rewrite those
        # in-place before serialising.
        dum_blocks_serialized: list[dict] = []
        for b in self.dum_blocks:
            d = asdict(b)
            d["file"] = _portable(d["file"])
            dum_blocks_serialized.append(d)

        return json.dumps(
            {
                "equates": dict(sorted(self.equates.items())),
                "dum_blocks": dum_blocks_serialized,
                "diagnostics": [_portable(d) for d in self.diagnostics],
            },
            indent=2,
            default=_default,
        )


# ---------------------------------------------------------------- expressions


_BIN_OPS = {"+", "-", "*", "/", "&", "|", "^"}


def _tokenize_expr(s: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c == "$":
            j = i + 1
            while j < len(s) and s[j] in "0123456789abcdefABCDEF":
                j += 1
            out.append(s[i:j])
            i = j
            continue
        if c == "%":
            j = i + 1
            while j < len(s) and s[j] in "01":
                j += 1
            out.append(s[i:j])
            i = j
            continue
        if c.isdigit():
            j = i + 1
            while j < len(s) and s[j].isdigit():
                j += 1
            out.append(s[i:j])
            i = j
            continue
        if c == '"':
            # Merlin: "X" in an expression is the ASCII value of X.
            if i + 2 < len(s) and s[i + 2] == '"':
                out.append(str(ord(s[i + 1])))
                i += 3
                continue
            raise ValueError(f"malformed char literal at {i}: {s!r}")
        if c == "*":
            # Current location counter. Pass 0 doesn't track PC outside
            # of `dum` blocks; expose it as a token so the caller can
            # emit a clearer diagnostic.
            out.append("*")
            i += 1
            continue
        if c.isalpha() or c == "_" or c == "]":
            # `]name` is a Merlin macro-style variable; treat as an
            # identifier so it can appear on either side of an equate.
            j = i + 1
            while j < len(s) and (s[j].isalnum() or s[j] in "_."):
                j += 1
            out.append(s[i:j])
            i = j
            continue
        if c in _BIN_OPS or c in "()":
            out.append(c)
            i += 1
            continue
        # Unknown character — abort expression cleanly.
        raise ValueError(f"unexpected character {c!r} in expression {s!r}")
    return out


class _ExprParser:
    """Recursive-descent evaluator. Operator precedence kept simple:
    `* / & | ^` bind tighter than `+ -`; left-associative."""

    def __init__(self, tokens: list[str], symbols: dict[str, int]) -> None:
        self.t = tokens
        self.i = 0
        self.symbols = symbols

    def _peek(self) -> str | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def _eat(self) -> str:
        tok = self.t[self.i]
        self.i += 1
        return tok

    def parse(self) -> int:
        value = self._sum()
        if self.i != len(self.t):
            raise ValueError(f"trailing tokens after expression: {self.t[self.i:]}")
        return value

    def _sum(self) -> int:
        v = self._term()
        while self._peek() in ("+", "-"):
            op = self._eat()
            r = self._term()
            v = v + r if op == "+" else v - r
        return v

    def _term(self) -> int:
        v = self._factor()
        while self._peek() in ("*", "/", "&", "|", "^"):
            op = self._eat()
            r = self._factor()
            if op == "*":
                v = v * r
            elif op == "/":
                v = v // r
            elif op == "&":
                v = v & r
            elif op == "|":
                v = v | r
            elif op == "^":
                v = v ^ r
        return v

    def _factor(self) -> int:
        tok = self._peek()
        if tok is None:
            raise ValueError("unexpected end of expression")
        if tok == "(":
            self._eat()
            v = self._sum()
            if self._eat() != ")":
                raise ValueError("expected ')'")
            return v
        if tok == "-":
            self._eat()
            return -self._factor()
        if tok == "+":
            self._eat()
            return self._factor()
        self._eat()
        if tok.startswith("$"):
            return int(tok[1:], 16)
        if tok.startswith("%"):
            return int(tok[1:], 2)
        if tok[0].isdigit():
            return int(tok, 10)
        if tok == "*":
            raise ValueError("current-PC operator `*` not resolvable in pass 0")
        if tok in self.symbols:
            return self.symbols[tok]
        raise ValueError(f"undefined symbol: {tok}")


def eval_expr(s: str, symbols: dict[str, int]) -> int:
    """Evaluate a Merlin expression against the current symbol table."""
    return _ExprParser(_tokenize_expr(s), symbols).parse()


# ---------------------------------------------------------------- parser


_IGNORED_DIRECTIVES = {"lst", "tr", "xc", "mx"}


@dataclass
class _DumState:
    block: DumBlock
    lc: int


class Parser:
    def __init__(self, search_paths: list[Path]) -> None:
        self.ast = ProgramAST()
        self.search_paths = search_paths
        self._included: set[Path] = set()

    def parse(self, path: Path) -> ProgramAST:
        self._parse_file(path.resolve())
        return self.ast

    # ---- internals ----

    def _parse_file(self, path: Path) -> None:
        if path in self._included:
            return
        self._included.add(path)
        lines = lex_file(path)
        file_ast = FileAST(path=str(path), lines=lines)
        self.ast.files.append(file_ast)

        dum: _DumState | None = None

        for line in lines:
            if line.is_blank:
                continue

            mnemonic = line.mnemonic

            # `LABEL = EXPR` equate
            if line.is_equate:
                if not line.label or not line.operand:
                    self._warn(line, "malformed equate")
                    continue
                try:
                    value = eval_expr(line.operand, self.ast.equates)
                except ValueError as exc:
                    self._warn(line, f"equate eval failed: {exc}")
                    continue
                self.ast.equates[line.label] = value
                continue

            if mnemonic in _IGNORED_DIRECTIVES:
                continue

            if mnemonic == "put":
                if not line.operand:
                    self._warn(line, "put with no operand")
                    continue
                self._handle_put(path, line.operand)
                continue

            if mnemonic == "dum":
                dum = self._open_dum(line, dum)
                continue

            if mnemonic == "dend":
                dum = None
                continue

            if mnemonic == "ds":
                if dum is None:
                    # `ds` outside a `dum` block advances the regular
                    # location counter; that's relevant only for files
                    # that emit code/data, not the equate files we
                    # focus on in pass 0.
                    continue
                self._handle_ds(line, dum)
                continue

            # Anything else (real opcodes, db/dw/hex/asc/dfb, org, jmp,
            # macros) is not yet processed by pass 0. The raw line stays
            # on `file_ast.lines` for later passes.

        # An open `dum` at end of file is implicitly closed.

    def _open_dum(self, line: Line, prev: _DumState | None) -> _DumState | None:
        del prev  # opening a new dum implicitly closes the previous one
        if not line.operand:
            # No operand: skip this block entirely. Returning `None`
            # leaves the parser in the "no open dum" state so any
            # following `ds` lines are silently dropped rather than
            # accumulating into a dangling block at address 0 (which
            # would also leak bogus equates into the symbol table).
            self._warn(line, "dum with no operand")
            return None
        try:
            addr = eval_expr(line.operand, self.ast.equates)
        except ValueError as exc:
            self._warn(line, f"dum addr eval failed: {exc}")
            addr = 0
        block = DumBlock(
            start_addr=addr,
            start_expr=line.operand,
            file=str(line.file),
            line=line.lineno,
        )
        self.ast.dum_blocks.append(block)
        return _DumState(block=block, lc=addr)

    def _handle_ds(self, line: Line, dum: _DumState) -> None:
        if not line.operand:
            self._warn(line, "ds with no operand")
            return
        try:
            size = eval_expr(line.operand, self.ast.equates)
        except ValueError as exc:
            self._warn(line, f"ds size eval failed: {exc}")
            return
        offset = dum.lc - dum.block.start_addr
        dum.block.fields.append(
            Field(name=line.label, offset=offset, size=size, line=line.lineno)
        )
        if line.label:
            # Inside a dum block, a labeled `ds` also creates a global
            # equate at the current location counter.
            self.ast.equates[line.label] = dum.lc
        dum.lc += size

    def _handle_put(self, current: Path, operand: str) -> None:
        # Strip a trailing comment if the lexer didn't already.
        name = operand.split(";", 1)[0].strip()
        target = self._resolve_include(current, name)
        if target is None:
            self._warn(
                Line(current, 0, "", None, "put", operand, None),
                f"could not resolve `put {name}`",
            )
            return
        self._parse_file(target)

    def _resolve_include(self, current: Path, name: str) -> Path | None:
        candidates: list[Path] = []
        roots = [current.parent, *self.search_paths]
        suffixes = (".S", ".s")
        for root in roots:
            for suf in suffixes:
                candidates.append(root / f"{name}{suf}")
                candidates.append(root / f"{name.upper()}{suf}")
                candidates.append(root / f"{name.lower()}{suf}")
        for c in candidates:
            if c.exists():
                return c.resolve()
        return None

    def _warn(self, line: Line, msg: str) -> None:
        self.ast.diagnostics.append(
            f"{line.file}:{line.lineno}: {msg} ({line.raw.strip()!r})"
        )


def parse_files(paths: list[Path], search_paths: list[Path] | None = None) -> ProgramAST:
    """Parse one or more `.S` files into a single `ProgramAST`. Files are
    processed in order; symbols from earlier files are visible to later
    ones, matching Merlin's flat global namespace."""
    parser = Parser(search_paths or [])
    for p in paths:
        parser.parse(p)
    return parser.ast
