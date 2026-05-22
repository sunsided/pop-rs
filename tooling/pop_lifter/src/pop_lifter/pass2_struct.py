"""Pass 2: structure IR1 into IR2 (real control flow).

TODO:
- Build CFG per function; compute dominator tree.
- Relooper-style structurer: reconstruct if/else/while/for.
- Irreducible flow → `loop { match pc { ... } }`, flagged for review.
- Compare+branch fusion: `cmp #k; beq L` → `if a == k { goto L }`.
- Flag-liveness elision: drop dead `set_nz` updates.
- Fold tagged 16-bit add/sub pairs.
- Fuse confirmed parallel-array groups into arrays-of-structs.
"""

from __future__ import annotations
