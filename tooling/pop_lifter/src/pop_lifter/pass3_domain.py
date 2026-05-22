"""Pass 3: domain abstraction from IR2 → IR3.

TODO:
- Map GRAFIX / HIRES / SOUND / keyboard jump-table call sites to backend
  trait calls via a `policy/backend_map.toml` file.
- Lift data-only `.S` files (SEQTABLE, FRAMEDEF, MOVEDATA, HRPARAMS,
  HRTABLES, BGDATA, TABLES) to typed `static` arrays.
- Extract level and sprite binaries from the submodule into
  `assets/extracted/`.
"""

from __future__ import annotations
