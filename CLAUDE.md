# pop-rs — orientation for Claude Code

Multi-pass effort to lift the original *Prince of Persia* (Apple II, 6502, Jordan
Mechner) into a C-like IR and then a Rust port. See `README.md` and
`docs/architecture.md` for the full design.

## Understanding original game logic: read the lifted Rust first

When you need to know how an original routine works, **do not jump straight to
the 6502 assembly.** Start with the lifted Rust. It preserves the original
control flow and routine names, so it gives a far better overview of what the
code does than raw 6502 ever will.

The lift is machine-generated and **may be incorrect** — use it to orient, then
drop down a layer to confirm any detail you rely on. Consult sources in this
order:

1. **Lifted Rust** — `ir/crate/src/<segment>.rs`. The assembled, buildable lift
   (issue #47): one module per POP source segment (`auto`, `ctrl`, `coll`,
   `mover`, …), each original routine a free function over a shared `Cpu`
   (`fn CHECKFLOOR(cpu: &mut Cpu)`). Cross-module `jsr`/`jmp` are resolved to
   real calls. **Start here.** (`ir/raw-rs/*.rs` is the flat per-module dump;
   `ir/raw-rs/SUMMARY.md` tracks lift coverage.)
2. **`.ir1`** — `ir/raw/<MODULE>.ir1` (whole-tree) and `ir/pilot/*.ir1`
   (curated slices). The C-like opcode-for-opcode IR: pseudo-registers
   `a,x,y`, flags `c,z,n,v`, named globals, branches as `goto`. Closer to the
   metal than the Rust; each line carries a `; FILE.S:line` provenance comment.
3. **Original assembly** — `vendor/pop-apple2/**/*.S` (Merlin source, vendored
   submodule, **read-only**). The ultimate ground truth. The `; FILE.S:line`
   comments in the IR point you at the exact source line.

In short: lifted Rust to understand, `.ir1` → original `.S` to verify.

The lift artifacts under `ir/` are **`@generated` — do not hand-edit them.** Fix
the lifter in `tooling/pop_lifter/` and regenerate (see `README.md` "Regenerate
after lifter changes").

## Don't confuse the lift with the runtime port

Two separate Rust bodies live in this repo:

- `ir/crate/` — the **lift** described above (mechanical, `Cpu`-based,
  `@generated`). Reference only; not the game you run.
- `crates/pop-rs/src/` — the **runtime port**. `world.rs` / `game.rs` are a
  hand-written *idiomatic* reimplementation ("Path B"): it reads the original
  for *behaviour* and reuses its *art*, but the runtime is fresh Rust. This is
  the live game. `src/modules/` + `src/data/` are reserved for the Pass-4
  generated port (mostly stubs today; do not hand-edit those either).

When implementing gameplay, write idiomatic Rust in `crates/pop-rs/`; use the
lift and the `.S` sources as the behavioural spec.

## Key paths

```
ir/crate/src/        lifted Rust (assembled crate)  ← read first
ir/raw-rs/           lifted Rust (flat per-module) + SUMMARY.md
ir/raw/, ir/pilot/   .ir1/.ir2/.ir3 IR snapshots
vendor/pop-apple2/   original Apple II 6502 source (submodule, read-only)
tooling/pop_lifter/  Python lifter: Merlin .S → IR1..IR3 → Rust
crates/pop-rs/       the runtime port (world.rs / game.rs = Path B)
crates/pop-assets/   asset extraction/decode; crates/pop-cli/ = `pop` binary
docs/                architecture & per-area notes
```

(Keep this file in sync when the structure changes.)
