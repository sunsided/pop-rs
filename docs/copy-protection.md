# Copy protection — strip it during the lift

The vendored upstream (`vendor/pop-apple2`, a pin of
`widemeadows/Prince-of-Persia-Apple-II`) is the **3.5"-only** recovered
source. The 3.5" retail disk was not copy-protected, so the protection
*setup* code lives only in the 5.25" build under `04 Support/`, which is
out of scope. The protection *checks*, however, are scattered through the
in-scope game files. A faithful lift reproduces those checks against memory
that nothing ever writes, which silently steers the game into its own
anti-piracy sabotage paths.

## The `redherring` trap

| What | Where | In scope? |
|------|-------|-----------|
| `redherring` declared (`ds 1`) | `01 POP Source/Source/GAMEEQ.S:515` | yes |
| `redherring2` declared (`ds 1`) | `01 POP Source/Source/EQ.S:465` | yes |
| Both **written** | `04 Support/MakeDisk/S/MASTER525.S:193,336` | **no** (5.25" master) |
| 1st check reads them | `01 POP Source/Source/TOPCTRL.S:1559-1560` (`lda redherring` / `eor redherring2`) | yes |

In our lift the two variables are read but never assigned, so the
`eor` produces garbage and the "passed copy protection" branch
(`TOPCTRL.S:1567`) is not taken.

## The sabotage layers

The protection is staged: a check runs early, but the visible failure is
delayed several levels so a naive bypass looks fine at first. Named by the
"colour" comments in the source:

| Layer | Refs | Delayed effect (per Ferrie) |
|-------|------|------------------------------|
| 1st check (`redherring`) | `TOPCTRL.S:1557-1567` | gates later levels |
| `yellowcheck` / "Yellow" | `TOPCTRL.S:399,1703`, `GAMEBG.S:1144` | — |
| `purpleflag` / purple | `TOPCTRL.S:1549`, `GRAFIX.S:519` | — |
| `timebomb` (2nd-level) | `SUBS.S:1596` | corrupt gfx / crash later |
| crash payload | `SUBS.S:1121` (`lda #120 ;crash (copy protect)`) | hard crash |

Observed symptoms when bypassed the wrong way (from Peter Ferrie's
2-side 16-sector writeup): corrupted graphics on level 4, crash into text
mode on level 7, hang with corrupted graphics on level 14 (the reunion
scene).

## What we do about it

The source already gates protection code behind a Merlin assembly equate:

```
CopyProtect = 1        ; 01 POP Source/Source/GRAFIX.S:2
  do CopyProtect       ; ... GRAFIX.S:442, etc.
```

Lift strategy:

- Treat `CopyProtect` as **0** in pass 0, so `do CopyProtect` / `fin`
  conditional-assembly regions are dropped before they reach IR1.
- For protection checks *not* guarded by that equate (e.g. the
  `redherring` `eor` in `TOPCTRL.S`), excise the check and force the
  "passed" path.
- Add a difftest guard asserting no `redherring`, `yellowcheck`,
  `purpleflag`, `timebomb`, or the `#120` crash store survive into IR1.

## Data-completeness caveat

Ferrie also notes that "rebuilt from source" images had truncated graphics
data and a missing track `$11` on side B. Verify the vendored
`01 POP Source/Images/IMG.CHTAB*` and `01 POP Source/Levels/LEVEL*`
binaries are complete; if not, document a retail-image extraction path.

Ref: <https://pferrie.epizy.com/misc/lowlevel14.htm>
