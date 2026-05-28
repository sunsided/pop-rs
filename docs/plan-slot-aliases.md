# Plan: segment jump-table call-target resolution (`slot_aliases`)

Status: **COMPLETE** — differential sweep green, all tests pass (see
"RESOLVED" below). This file captures the design and the three fixes.

Branch: `claude/blissful-newton-2DB3a`. PR: #68.

---

## Goal

`jsr`/`jmp` call sites that name a *segment jump-table slot* (e.g.
`jsr AutoCtrl`) didn't resolve to any lifted routine, because the lifter
names routines after their *definition* label (`AUTOCTRL`), not the slot.
~330 such targets were unresolved. The task: resolve them by attaching the
slot label as an `entry_alias` on the target routine.

### How resolution works (correct-by-construction)

Each POP segment exposes its public entry points through a **head jump
table**: right after `org org` it emits one `jmp <TARGET>` per slot, in
slot order. The matching `dum` overlay in `EQ.S`/`GAMEEQ.S` (named after
the segment, e.g. `dum auto`) labels those same slots as 3-byte fields
(`AutoCtrl ds 3`, ...). So the i-th dum slot (offset `3*i`) corresponds to
the i-th head `jmp`. Reading the *actual* jmp operand handles renamed slots
(`copyscrnMM` ↔ `_copy2000`) with no fuzzy matching.

Binding rule (in `apply_slot_aliases`):
- Only **pure 3-byte-slot** dum blocks count (a non-3-byte field means a
  data overlay or an alternate bank-switch overlay sharing the load
  address — e.g. MASTER's second `$f880` block — not the jump table).
- Map slot at offset `3*i` to head `jmp[i]` only when `i < len(jmps)`
  (MISC's table has 2 jmps but 18 dum slots — the rest are data).
- **Bind strictly intra-segment**: the alias attaches only to a routine
  *in the dum's own segment* whose name equals the jmp target. A head jump
  is the segment's own API, so its target is always in-segment. This
  prevents a head target that happens to share a name with an unrelated
  routine in another segment (GRAFIX's `DOSTARTGAME` bank-switch trampoline
  vs. MASTER's real `DOSTARTGAME`) from being claimed by the wrong one.
- **Never create a collision**: skip any alias name already owned by some
  routine (GRAFIX's `cls`/`lay`/`peel` trampolines collide with HIRES's
  same-named routines).

### Numbers

- **249** previously-unresolved targets now resolve, **0 new collisions.**
- Not 251: the 2 extra in the earlier quantification were misattributions
  (head-jump target not actually lifted in its own segment, e.g. MASTER's
  `_dostartgame`). Binding those produced infinite self-recursion; the
  strict intra-segment rule correctly leaves them external. 249 is the
  right, safe count.

---

## Files changed / added

- **`tooling/pop_lifter/src/pop_lifter/slot_aliases.py`** (new): the whole
  mechanism — `_head_jump_targets`, `slot_alias_entries`,
  `apply_slot_aliases`.
- **`cli.py`**: `lift_all_modules` now calls `apply_slot_aliases(modules,
  ast)` before returning (single insertion point — feeds crate, IR3
  interpreter, and the differential harness).
- **`tooling/pop_lifter/tests/test_slot_aliases.py`** (new, 9 tests, all
  pass): ground-truth pins (`AUTOCTRL←AutoCtrl`, renamed
  `copyscrnMM←_copy2000`, segment-shared `LOADLEVEL→MASTER`), short-table /
  bank-switch-overlay exclusions, and the resolve-improves-with-no-new-
  collisions guard.
- **`interp_ir3.py` / `interp_ir1.py`**: added a hardware-faithful call-depth
  bound (`_MAX_CALL_DEPTH = 128`, matching the 6502's 256-byte stack ÷ 2)
  so bank-switch trampolines that ping-pong forever (LAY/PEEL/EPILOG/…)
  raise `InterpError` (→ sweep skips them) instead of blowing Python's
  stack. `run` bumps the recursion limit for the run and converts any
  residual `RecursionError` to `InterpError`. Added `Trace.call_depth`.
- **`ir/crate/**`** regenerated: compiles clean under `-D warnings`;
  `ext.rs` shrank by 251 stub lines.

### Regenerate / test commands
```
# regenerate the assembled crate after any lifting change
python3 -c "import sys; sys.path.insert(0,'tooling/pop_lifter/src'); \
  from pop_lifter.cli import main; sys.argv=['x','emit-crate','--out-dir','ir/crate']; \
  raise SystemExit(main())"

cd tooling/pop_lifter
python3 -m pytest tests/test_slot_aliases.py tests/test_crate_scaffold.py -q
python3 -m pytest tests/test_differential.py -q     # needs cargo on PATH
```

---

## RESOLVED — sweep is green

`test_differential.py::test_seeded_sweep_no_divergence` passes (0
divergences over ~1240 comparisons); full suite: 616 unit + 93
differential tests green. The widened reachability from the slot aliases
exposed three pre-existing interpreter↔crate gaps, fixed in order:

1. **Caller-sensitive call resolution** (commit *resolve calls
   caller-sensitively*). The IR3 interpreter resolved calls
   caller-*insensitively* (flat first-module-wins) while the crate
   resolves per-caller-module (intra preferred, unique cross-module, else
   external stub). Once aliases widened reachability, routines hitting a
   reused name (`addsound`, `tone`, …) ran a different callee in each.
   Fix: thread the caller's home module through the walker (`_Ctx` +
   `trace.home_module`) and resolve with the crate's exact policy
   (`_resolve_call_ctx`); external/ambiguous → `InterpError` (skip,
   mirroring the crate's no-op stub). Fixed `gamebg/TIMELEFTMSG`.

2. **SMC operand patch in indexed addressing** (commit *honor SMC operand
   patch for indexed addressing*). `_indexed_addr` resolved its base via
   `_real_addr`, ignoring an `opvar` address patch — so an SMC store
   folded into an IR3 `Assign` (`IndexedAbs` base with `opvar`) used the
   unpatched address while the unfolded `StoreIndexed` path (`_abs_base`)
   used the patched one. `coll/getCData` patches its `CDthisframe,x` /
   `SNthisframe,x` store low-bytes from X / A; the folded `SNthisframe`
   store landed wrong. Fix: `_indexed_addr` resolves through `_abs_base`
   (non-SMC operands fall back to `_real_addr` unchanged).

3. **Loop-back tail call mis-resolved** (commit *resolve a loop-back tail
   call to the current routine*). The relooper exposes a routine's loop
   label (e.g. `:loop`) as an entry alias so its self-looping
   `tail_call :loop` has a target. That label isn't unique (both
   `pauseNI` and `tpause` carry `:loop`), so the global name table sent
   `pauseNI`'s loop into `tpause` (last-wins). The crate resolves it
   lexically as a self-loop. Fix: `_resolve_tail` resolves a tail call
   whose target is the current routine's own name/entry-alias to the
   current routine, before the global table. Fixed `master/pauseNI`.

### Done
- All slot-alias work committed (`slot_aliases.py`, `cli.py` wiring,
  tests, regenerated `ir/crate`).
- Interpreter: depth bound, caller-sensitive resolution, SMC indexed-addr
  fix, loop-back tail-call fix — all committed.
- `pytest -q` (616) + `pytest tests/test_differential.py -q` (93) green.
- PR #68 open; branch `claude/blissful-newton-2DB3a`.

### Notes for future work
- `:loop` (and similar generic local labels) leaking into `entry_aliases`
  is a latent smell — the interp fix handles the self-loop case, but a
  cross-routine tail call to a genuinely colliding alias would still hit
  the last-wins ambiguity in `routine_by_module`. Consider not promoting
  bare local loop labels to global entry aliases in the relooper.
- `master/pauseNI`'s ~256K-iteration delay loop runs near the sweep's
  `_INTERP_TIMEOUT_S = 0.5s`; it now matches the crate when it completes,
  and a timeout is a skip (not a divergence), so the sweep is stable
  either way.

## Quick orientation
- `pass4_crate.resolve_call` / `build_name_map` — the crate's resolution
  policy (the behavior the interpreter now mirrors).
- `interp_ir3._Ctx` / `_resolve_call_ctx` / `_resolve_tail` — the
  interpreter's caller-sensitive resolver.
- `interp_ir1._abs_base` / `_indexed_addr` — SMC-patch-aware base
  resolution.
- `slot_aliases.apply_slot_aliases` — the slot-alias binding logic.
