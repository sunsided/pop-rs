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
2-side 16-sector writeup — see ref at the end): corrupted graphics on
level 4, crash into text mode on level 7, hang with corrupted graphics
on level 14 (the reunion scene).

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

Peter Ferrie also notes that "rebuilt from source" images shipped with
**truncated graphics data and a missing track `$11` on side B**. The
vendored tree is exactly such a rebuild, and the truncation is real and
visible in the renderer.

### Renderer-side fingerprint (PR [#109](https://github.com/sunsided/pop-rs/pull/109))

Interactive testing of the egui level browser surfaced diagonal black
gaps next to `LooseFloor` tiles in **red-biome** rooms (LV12 R12 / R19 /
R20, LV13 R17), and a generally sparser palette than the palace / dungeon
equivalents. The cause was not in the renderer — it reproduces
`FRAMEADV.S:SURE` / `RedBlockSure` against whatever bytes the loaded
`IMG.BGTAB.*` provides. The cause is the asset content:

| File | What's wrong | Evidence |
|------|--------------|----------|
| `04 Support/DRAZ/IP/IMG.BGTAB.RED1` | Sprite ID `0x1b` (= `looseb`, the `drawlooseb` spillover) is a 1×1 placeholder, single byte `0x80`. In `IMG.BGTAB.{DUN,PAL}1` the same sprite is 28×13 with the diagonal floor-edge pattern. | Raw directory walk: pointer `0x684f → 0x6852`, 3 bytes total (width=1, height=1, byte `0x80`). |
| `04 Support/DRAZ/IP/IMG.BGTAB.RED2` | Ships **50 sprites** where `IMG.BGTAB.{DUN,PAL}2` ship **126**. | `ImageTable::images.len()` at load time. |

Because `FRAMEADV.S:1388 drawlooseb` doesn't biome-check — it always
draws `looseb = $1b` from whichever `IMG.BGTAB.*1` is currently loaded —
the truncated `0x1b` produces a visible 7-pixel gap between every red-biome
`LooseFloor` cell. Faithful renderer, faithfully stripped asset.

### Workaround until the vendored tree is replaced

The renderer is correct against the data it's given. The clean fix is
**asset-side**: extract `IMG.BGTAB.*`, `IMG.CHTAB*`, and any other
truncated binaries from a known-good retail disk image (canonical
1989 Broderbund 5.25" `.woz`) and point `pop-cli editor` at a data root
containing those. Two avenues:

1. Once [#84](https://github.com/sunsided/pop-rs/issues/84) (disk-image
   reader for `.dsk` / `.nib` / `.woz`) lands, run `pop draz extract`
   against the retail `.woz` and write the result alongside the vendored
   tree.
2. Until then, manually extract from a retail disk via an existing
   utility (e.g. AppleSauce, CiderPress) and drop the resulting
   `IMG.BGTAB.{DUN,PAL,RED}{1,2}` into the data-root `DRAZ/IP/`
   directory.

Both follow the data-root discovery path from
[#105](https://github.com/sunsided/pop-rs/pull/105) and are transparent
to the editor.

### Detection at load time

`pop-assets` surfaces a heuristic diagnostic when loading a BGTAB
that fingerprints as the truncated 3.5" rebuild — see
[`crate::scene::BiomeTables::load_diagnostics`]. The check is
non-fatal (the editor continues to render with the truncated assets),
but it gives a clear signal to anyone wondering why their LV13 floors
have visible gaps.

### Tracking

* [#110](https://github.com/sunsided/pop-rs/issues/110) — closed as a
  duplicate of [#112](https://github.com/sunsided/pop-rs/issues/112);
  the symptom that surfaced this caveat.
* [#84](https://github.com/sunsided/pop-rs/issues/84) — disk-image
  reader needed for the long-term workaround.

Ref: <https://pferrie.epizy.com/misc/lowlevel14.htm> (Peter Ferrie,
"Old School Hacks of the New School Hacks #14" — POP protection
writeup).
