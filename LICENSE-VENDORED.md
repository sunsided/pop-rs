# Vendored upstream licensing

The directory [`vendor/pop-apple2/`](./vendor/pop-apple2) is a git submodule
pointing at <https://github.com/widemeadows/Prince-of-Persia-Apple-II>, which
mirrors the original *Prince of Persia* Apple II 6502 source code released
by Jordan Mechner.

**The EUPL-1.2 license in [`LICENSE`](./LICENSE) does NOT apply to anything
inside `vendor/pop-apple2/`.** That source is governed by the terms stated
in the upstream `README.md`, reproduced here verbatim:

---

> Some background: This archive contains the source code for the original
> Prince of Persia game that I wrote on the Apple II, in 6502 assembly
> language, between 1985-89. The game was first released by Broderbund
> Software in 1989, and is part of the ongoing Ubisoft game franchise.
>
> [...]
>
> We extracted and posted the 6502 code because it was a piece of computer
> history that could be of interest to others, and because if we hadn't, it
> might have been lost for all time. We did this for fun, not profit. As the
> author and copyright holder of this source code, I personally have no
> problem with anyone studying it, modifying it, attempting to run it, etc.
> Please understand that this does NOT constitute a grant of rights of any
> kind in Prince of Persia, which is an ongoing Ubisoft game franchise.
> Ubisoft alone has the right to make and distribute Prince of Persia games.
>
> -- Jordan Mechner (Updated September 2024)

---

## What this means in practice

1. The Apple II source in `vendor/pop-apple2/` is included for **study and
   archival** purposes only. It is not offered to you by this project under
   any open-source license.
2. The *Prince of Persia* name, characters, and franchise are the property
   of Ubisoft. Nothing in this repository grants any rights to them.
3. Game assets (sprite tables `IMG.CHTAB*`, background tables `IMG.BGTAB*`,
   level data `LEVEL0..14`) are **not** redistributed by `pop-rs`. The
   lifter reads them from your local checkout of the upstream submodule and
   converts them into in-memory data structures during build.
4. The lifter and Rust port in this repository are independently authored
   code; they describe and reimplement the original's behavior but contain
   no verbatim copy of the upstream source. Those original-authored parts
   are covered by the EUPL-1.2 license in `LICENSE`.

If you redistribute a binary built from this project, you are responsible
for ensuring you do not redistribute the upstream Apple II source or game
assets in a manner inconsistent with the statement above.
