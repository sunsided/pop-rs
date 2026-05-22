"""Pass 0b: parse tokens into a per-file AST and link into a `ProgramAST`.

TODO:
- Resolve `put FILE` includes transitively, depth-first.
- Build a flat global symbol table (Merlin namespace).
- Capture `dum addr ... dend` blocks as named struct layouts.
- Capture top-of-file `jmp` ladders as per-module ABI (jump-table slots).
- Resolve forward references via a fixpoint pass once all files are loaded.
"""

from __future__ import annotations
