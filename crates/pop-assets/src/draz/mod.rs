//! Sprite and screen extraction for POP's `DRAZ/` asset tree.
//!
//! POP's `04 Support/DRAZ/` directory holds two families of binary
//! asset files that drive the renderer:
//!
//! * **Whole hi-res screens** — exactly 8 KiB, the raw contents of one
//!   Apple II hi-res page (intro / cutscene / title art).  Lives under
//!   [`screen`]; pair with [`crate::hires::render`] for pixels.
//!
//! * **Image tables** — variable-size files named `IMG.CHTAB*`
//!   (character animation sheets: prince, guard, fat, shadow, …) and
//!   `IMG.BGTAB*` (background tile sheets, one set per biome: dungeon /
//!   palace / red / tower). Lives under [`image_table`]; each entry is
//!   a `(width_bytes, height, bitmap)` sprite that pairs with
//!   [`crate::hires::render_linear`].
//!
//! Masks and image pieces are stored as separate entries in the same
//! table (per `BGDATA.S` / `FRAMEDEF.S`); their composition (mask AND,
//! image OR) happens at draw time and is intentionally out of scope
//! here — extraction returns the raw sprite bitmaps, the editor / game
//! loop composes them later.

pub mod image_table;
pub mod screen;
