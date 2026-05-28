# Plan: segment jump-table call-target resolution (`slot_aliases`)

Status as of this writing: **implemented and mostly green; blocked on one
scope decision** (interpreter vs. crate call-resolution parity). This file
captures everything needed to continue offline.

Branch: `claude/blissful-newton-2DB3a` (do not push elsewhere). No PR yet.

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

## THE OPEN DECISION (blocker)

The differential sweep (`test_differential.py::test_seeded_sweep_no_divergence`)
now runs **1176 comparisons** (more routines are reachable) and reports **4
state divergences** in 2 newly-reachable routines: `coll/getCData` and
`gamebg/TIMELEFTMSG`.

Traced precisely: **registers match**; only ~20 RAM bytes differ, all in
**sound buffers** (`vibes`/`SongCue`/`musicon` @ `$0316–$031f`, `$0360–$0369`).

Root cause is **pre-existing**, not a bug in the slot work:
- The IR3 interpreter resolves calls **caller-insensitively** — flat name
  lookup over a module-ordered list (`interp_ir1._resolve` + the global
  `_alias_index`). So `addsound`→SOUND, `tone`→GRAFIX.
- The emitted crate resolves **per-caller-module** (`pass4_crate.resolve_call`:
  intra-module preferred, unique cross-module, else `ext`). So
  `addsound`→SPECIALK, `tone`→SOUND.
- There are **111** such latent mismatches program-wide
  (`addsound`×32, `tone`×14, `DoEngarde`×6, `PlaySongI`×6, …). They were
  simply never both-reached in a completing run until the aliases widened
  reachability.

So the slot resolution is correct; it tripped over an existing
interpreter↔crate resolution gap.

### Options (pick one to proceed)

1. **Fix interpreter resolution to match the crate (RECOMMENDED).**
   Make the IR3 interpreter resolve each call relative to the *current
   routine's module* using `resolve_call` semantics (intra preferred,
   unique cross-module, else treat as external → `InterpError`/skip, the
   same as today's unresolved behavior). Eliminates all 111 latent
   mismatches and makes the differential harness sound.
   - Why it's safe: currently-passing routines already agree with the
     crate, so switching the interpreter to the crate's policy can't
     regress them; it only fixes the divergent ones.
   - Sketch: in `run`, build `name_map = build_name_map(modules)` plus a
     `(module, entry-name) → routine` index and a `routine → home module`
     map. Thread the home module through execution (simplest: carry it on
     `Trace` and save/restore around each routine invocation, since `Trace`
     is already threaded everywhere). New `_resolve_call_ctx(name_map,
     index, home_module, target)` returns `(routine, owner_module)` or
     `None` (ambiguous/external → caller raises `InterpError`). `CallStmt`
     and `TailCallStmt` use it and recurse with `home_module = owner`.
     `run` seeds `home_module` = entry routine's module. Keep external/
     ambiguous as skip (do **not** switch to no-op — that would widen the
     comparison set and risk surfacing unrelated pre-existing bugs).
   - Verify: differential sweep → 0 divergences; full suite green; crate
     regen unchanged (this is interpreter-only).

2. **Skip ambiguous-call routines (smaller unblock).** When a routine
   reaches a call that's ambiguous from its module, raise `InterpError` so
   the sweep skips it (like today's unresolved behavior) rather than
   comparing with wrong resolution. Same plumbing cost as option 1 (still
   needs current-module + name_map) but strictly less capable — it skips
   `getCData` instead of correctly resolving it. Only worth it if option 1
   feels too risky.

3. **Crate-only aliases.** Feed slot aliases to the crate but not the
   interpreter (separate path), so sweep reachability is unchanged. Loses
   interpreter validation of the newly-resolved calls and undercuts the
   "single insertion point" design. Not recommended.

**Recommendation: option 1.** It's the principled fix, aligns with the
differential harness's evident design intent (interp and crate should
agree), and is strictly an improvement.

---

## Remaining steps once the decision is made

1. Implement the chosen option (option 1: interpreter caller-sensitive
   resolution).
2. `pytest tests/test_differential.py -q` → expect 0 divergences.
3. Full suite: `cd tooling/pop_lifter && python3 -m pytest -q`.
4. Re-confirm `ir/crate` regen + `-D warnings` compile (interpreter change
   shouldn't touch the crate, but re-run `test_crate_scaffold.py`).
5. Commit: the slot-alias feature (`slot_aliases.py`, `cli.py` wiring,
   tests), the interpreter depth bound, the interpreter resolution fix, and
   the regenerated `ir/crate`. Suggested split: (a) slot-alias resolution +
   crate regen, (b) interpreter depth bound, (c) interpreter resolution
   parity. Then push `-u origin claude/blissful-newton-2DB3a`. **No PR
   unless explicitly asked.**

## Quick orientation for whoever picks this up
- `pass4_crate.resolve_call` / `build_name_map` — the crate's resolution
  policy (the target behavior).
- `pass4_emit_rust._make_call_render` (line ~1470) — how the crate emits
  intra (`fn(cpu)`) vs cross-module (`crate::owner::fn(cpu)`) vs
  `crate::ext::stub(cpu)`.
- `interp_ir1._resolve` (line ~323) — the interpreter's current flat
  resolver (what option 1 replaces for call sites).
- `slot_aliases.apply_slot_aliases` — the new binding logic.
