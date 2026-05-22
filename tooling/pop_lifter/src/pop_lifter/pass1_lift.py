"""Pass 1: mechanical lift of 6502 opcodes to IR1 (C-like).

TODO:
- One IR1 function per discovered routine (reachable from a jump-table slot
  or a JSR site).
- Pseudo-registers: a, x, y. Flag pseudo-globals: c, z, n, v.
- Memory accesses become reads/writes of named globals via the symbol table.
- Branches emit labelled `goto`.
- Annotate parallel-array accesses (e.g. `lda mobx,x`) for pass 2 fusion.
- Recognize indirect-indexed `(ptr),y` and tag ptr as a 16-bit pair.
- Recognize 16-bit add/sub patterns; tag for pass 2 fold.
- Detect self-modifying code (stores into emitted code byte ranges) and
  rewrite both writer and reader through a synthesized variable.
"""

from __future__ import annotations
