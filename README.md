# pop-rs

A multi-pass effort to lift the original *Prince of Persia* (Apple II, 6502
assembly, Jordan Mechner, 1985–89) into a C-like intermediate representation
and then into a Rust port with a pluggable graphics/audio/input backend.

The original 6502 source is vendored as a git submodule at
[`vendor/pop-apple2/`](./vendor/pop-apple2). It is read by the lifter and is
**not** modified.

## Status

Pass 0 (Merlin lex/parse) and a first slice of Pass 1 (mechanical lift to IR1
plus a Python interpreter for differential testing) are working on the AUTO.S
combat-button pilot. Pass 2/3/4 not started.

Inspectable intermediate artifacts:

| Path                          | What it is                                                                                |
|-------------------------------|-------------------------------------------------------------------------------------------|
| `ir/pass0/equates.json`       | Pass-0 AST: every named address from `EQ.S` + `GAMEEQ.S`, plus the `dum`/`dend` overlays. |
| `ir/pilot/auto_combat.ir1`    | Pass-1 IR1: the combat-button routines from `AUTO.S` lifted opcode-for-opcode.            |

Regenerate after lifter changes:

```sh
pop-lifter dump-ast --out ir/pass0/equates.json
pop-lifter lift AUTO.S --entry DoStrike --entry DoBlock --entry DoTurn \
    --entry DoStandup --entry DoEngarde --entry DoRelBtn --entry DoRelease \
    --out ir/pilot/auto_combat.ir1
```

`tests/test_artifacts.py` re-generates them in-memory and fails CI on drift.

## Layout

```
vendor/pop-apple2/          original Apple II source (submodule, read-only)
tooling/pop_lifter/         Python lifter: Merlin .S -> IR1..IR3 -> Rust
crates/pop-rs/              the Rust port (Cargo workspace member)
  src/game.rs               central Game struct (formerly global memory)
  src/modules/              one module per .S file (ctrl, mover, coll, ...)
  src/backend/              Renderer / Audio / Input traits
  src/data/                 generated const tables (do not hand-edit)
ir/pilot/                   checked-in IR snapshots for pilot routines
assets/extracted/           binary game data extracted from upstream
tests/                      differential and golden-frame tests
docs/                       architecture and per-module notes
```

## The four lifter passes

| Pass | Input | Output | Purpose |
|------|-------|--------|---------|
| 0    | `.S`  | AST + symbol table | Parse Merlin syntax; resolve `put` includes, `dum`/`dend` overlays, jump tables |
| 1    | AST   | IR1 (C-like)       | Opcode-for-opcode lift: pseudo-registers `a,x,y`, flags `c,z,n,v`, named globals, branches as `goto` |
| 2    | IR1   | IR2                | CFG → structured control flow, 16-bit fold, parallel arrays → struct AoS |
| 3    | IR2   | IR3                | GRAFIX/HIRES/SOUND/disk → backend trait calls; data tables → typed consts |
| 4    | IR3   | Rust               | Emit `crates/pop-rs/src/modules/*.rs` and `src/data/*.rs` |

See [`docs/architecture.md`](./docs/architecture.md) for details.

## Building (eventually)

```sh
git clone --recurse-submodules https://github.com/sunsided/pop-rs.git
cd pop-rs
# lifter
pip install -e tooling/pop_lifter
# Rust port
cargo build
```

## Licensing

### This repository's own code (lifter + Rust port)

Licensed under the [European Union Public Licence v. 1.2 (EUPL-1.2)](./LICENSE).

SPDX-License-Identifier: `EUPL-1.2`

This covers everything under `tooling/`, `crates/`, `ir/`, `tests/`, `docs/`,
and the build/config files at the repository root. It does **not** cover the
vendored upstream source at `vendor/pop-apple2/`, nor the *Prince of Persia*
franchise itself.

### Vendored upstream source (`vendor/pop-apple2/`)

The Apple II source code is included as a git submodule pointing at
<https://github.com/widemeadows/Prince-of-Persia-Apple-II>. Per the upstream
README by Jordan Mechner:

> We extracted and posted the 6502 code because it was a piece of computer
> history that could be of interest to others, and because if we hadn't, it
> might have been lost for all time. We did this for fun, not profit. As the
> author and copyright holder of this source code, I personally have no
> problem with anyone studying it, modifying it, attempting to run it, etc.
> Please understand that this does NOT constitute a grant of rights of any
> kind in Prince of Persia, which is an ongoing Ubisoft game franchise.
> Ubisoft alone has the right to make and distribute Prince of Persia games.

This project exists for study, archival, and technical interest. It does
**not** grant rights in *Prince of Persia* as a franchise, trademark, or
product. Game assets (sprites, level data, sound) are not redistributed
here — the lifter extracts them from your local checkout of the upstream
submodule.

See [`LICENSE-VENDORED.md`](./LICENSE-VENDORED.md) for the full vendored
licensing statement.
