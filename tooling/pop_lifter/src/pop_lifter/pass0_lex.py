"""Pass 0a: lex Merlin 16+ source into a token stream.

TODO:
- Token kinds: Comment, Label, LocalLabel, MacroLabel, Opcode, PseudoOp,
  Number, String, Operator, Newline, Eof.
- Recognized pseudo-ops: org, put, ds, db, dw, hex, dum, dend, equ, mac,
  asc, dfb, lup, ascii.
- Ignore (but track for source-mapping): lst on/off, tr on/off, xc, mx.
- Numbers: $hex, %binary, decimal; expression operators +-*/&|^.
- Labels: column-1 identifier; `:foo` local; `]foo` macro-style trampoline.
- Comments: line starting with `*`, or trailing after operand.
"""

from __future__ import annotations
