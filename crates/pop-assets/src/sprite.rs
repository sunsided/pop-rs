//! Character-sprite compositing onto a rendered [`Frame`].
//!
//! The background scene comes from [`crate::scene`]; characters (the
//! Prince, guards) are a separate layer blitted on top. The original
//! draws them through a precomputed mask; until that lands we use a
//! quick overlay that treats black sprite pixels as transparent. Good
//! enough for a static figure on the dark dungeon background — refine
//! with real POP masking when animation / combat needs pixel-exact
//! edges.

// Frame coordinates are bounded (≤ 280×192 + small sprite deltas);
// i32 ↔ u32 casts here are exact for the inputs this module sees.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use crate::draz::image_table::Image;
use crate::hires::{render_linear, Frame, RenderMode};

/// Blit `img` onto `dst` with its top-left at pixel `(x, y)`,
/// horizontally mirrored when `flip_h` (POP draws right-facing
/// characters mirrored — see [`crate::scene`] callers).
///
/// Black (`[0, 0, 0]`) source pixels are treated as transparent and
/// skipped; out-of-bounds pixels are clipped. No-op if the sprite has
/// zero size.
pub fn overlay(dst: &mut Frame, img: &Image, x: i32, y: i32, mode: RenderMode, flip_h: bool) {
    let Some(src) = render_linear(&img.bitmap, img.width_bytes, img.height, mode) else {
        return;
    };
    blit_non_black(dst, &src, x, y, flip_h);
}

/// Copy every non-black pixel of `src` onto `dst` at offset `(x, y)`,
/// clipping to `dst`'s bounds. When `flip_h`, source column `sx` lands at
/// destination column `x + (width - 1 - sx)`, mirroring the sprite.
fn blit_non_black(dst: &mut Frame, src: &Frame, x: i32, y: i32, flip_h: bool) {
    let dw = dst.width as i32;
    let dh = dst.height as i32;
    let sw = src.width as i32;
    for sy in 0..src.height as i32 {
        let dy = y + sy;
        if dy < 0 || dy >= dh {
            continue;
        }
        for sx in 0..sw {
            let dx = if flip_h { x + (sw - 1 - sx) } else { x + sx };
            if dx < 0 || dx >= dw {
                continue;
            }
            let si = ((sy * sw + sx) * 4) as usize;
            let px = &src.pixels[si..si + 4];
            if px[0] == 0 && px[1] == 0 && px[2] == 0 {
                continue; // transparent background pixel
            }
            let di = ((dy * dw + dx) * 4) as usize;
            dst.pixels[di..di + 4].copy_from_slice(px);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32, rgba: [u8; 4]) -> Frame {
        Frame {
            width,
            height,
            pixels: rgba
                .iter()
                .copied()
                .cycle()
                .take((width * height * 4) as usize)
                .collect(),
        }
    }

    #[test]
    fn blit_skips_black_and_clips() {
        let mut dst = solid(4, 4, [10, 20, 30, 255]);
        // 2×2 source: top row white, bottom row black (transparent).
        let src = Frame {
            width: 2,
            height: 2,
            pixels: vec![
                255, 255, 255, 255, 255, 255, 255, 255, // y=0 white
                0, 0, 0, 255, 0, 0, 0, 255, // y=1 black
            ],
        };
        // Place at (3,0): sx=1 lands at dx=4 (clipped off the right edge).
        blit_non_black(&mut dst, &src, 3, 0, false);

        let at = |x: usize, y: usize| {
            let i = (y * 4 + x) * 4;
            [dst.pixels[i], dst.pixels[i + 1], dst.pixels[i + 2]]
        };
        assert_eq!(at(3, 0), [255, 255, 255], "white copied at in-bounds pixel");
        assert_eq!(
            at(3, 1),
            [10, 20, 30],
            "black source pixel left transparent"
        );
        assert_eq!(at(0, 0), [10, 20, 30], "untouched pixel unchanged");
    }

    #[test]
    fn flip_mirrors_columns() {
        let mut plain = solid(2, 1, [0, 0, 0, 255]);
        let mut flipped = solid(2, 1, [0, 0, 0, 255]);
        // 2×1 source: left column white, right column black (transparent).
        let src = Frame {
            width: 2,
            height: 1,
            pixels: vec![255, 255, 255, 255, 0, 0, 0, 255],
        };
        blit_non_black(&mut plain, &src, 0, 0, false);
        blit_non_black(&mut flipped, &src, 0, 0, true);
        let col0 = |f: &Frame| [f.pixels[0], f.pixels[1], f.pixels[2]];
        let col1 = |f: &Frame| [f.pixels[4], f.pixels[5], f.pixels[6]];
        // Unflipped: white in col 0. Flipped: white mirrored to col 1.
        assert_eq!(col0(&plain), [255, 255, 255]);
        assert_eq!(col1(&flipped), [255, 255, 255]);
        assert_eq!(col0(&flipped), [0, 0, 0]);
    }
}
