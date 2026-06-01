//! Character-sprite compositing.
//!
//! The background scene comes from [`crate::scene`]; characters (the
//! Prince, guards) are a separate layer drawn on top. The original draws
//! them through a precomputed mask; until that lands we treat black /
//! empty sprite bytes as transparent — good enough for a figure on the
//! dark dungeon background.
//!
//! Two paths:
//! * [`composite_hires`] writes the sprite into the room's **hi-res byte
//!   buffer** *before* NTSC decode — the correct one for colour, since
//!   each pixel's artifact hue stays tied to its true screen column.
//! * [`overlay`] blits onto an already-decoded RGBA [`Frame`]; simpler,
//!   but flipping / shifting it swaps NTSC colours, so it's only for
//!   non-artifact use (mono, UI).

// Frame coordinates are bounded (≤ 280×192 + small sprite deltas);
// i32 ↔ u32 casts here are exact for the inputs this module sees.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use crate::draz::image_table::Image;
use crate::hires::{render_linear, Frame, RenderMode};

/// Composite a character sprite into a **top-down** hi-res byte buffer
/// (`dst`, `dst_width_bytes × dst_height`) at byte column `byte_x` and top
/// scan-line `top_y`.
///
/// The sprite bitmap is stored bottom-up (POP order); rows are flipped
/// into the top-down buffer. When `flip_h` the sprite is mirrored — byte
/// order reversed across its width and the 7 pixel bits reversed within
/// each byte (the palette/high bit is preserved). Sprite bytes with no
/// pixels set are transparent; others replace the destination byte
/// (figure on a dark background). Out-of-range bytes are clipped.
///
/// Compositing here — in byte space, before [`crate::hires`] decodes the
/// buffer — keeps every pixel's NTSC artifact colour tied to its true
/// screen column, unlike an RGBA [`overlay`] which swaps orange/blue when
/// the sprite is mirrored or shifted off an even-column boundary.
#[allow(clippy::verbose_bit_mask)] // `& 0x7f == 0` ("no pixels set") reads clearer
pub fn composite_hires(
    dst: &mut [u8],
    dst_width_bytes: usize,
    dst_height: usize,
    img: &Image,
    byte_x: i32,
    top_y: i32,
    flip_h: bool,
) {
    let w = usize::from(img.width_bytes);
    let h = usize::from(img.height);
    if w == 0 || h == 0 || img.bitmap.len() < w * h {
        return;
    }
    for sy in 0..h {
        // Bottom-up bitmap row `sy` is screen row `top_y + (h-1-sy)`.
        let screen_row = top_y + (h - 1 - sy) as i32;
        if screen_row < 0 || screen_row >= dst_height as i32 {
            continue;
        }
        let row_base = screen_row as usize * dst_width_bytes;
        for sx in 0..w {
            let dbx = if flip_h {
                byte_x + (w - 1 - sx) as i32
            } else {
                byte_x + sx as i32
            };
            if dbx < 0 || dbx >= dst_width_bytes as i32 {
                continue;
            }
            let mut b = img.bitmap[sy * w + sx];
            if flip_h {
                b = mirror_byte(b);
            }
            if b & 0x7f == 0 {
                continue; // no pixels set → transparent
            }
            dst[row_base + dbx as usize] = b;
        }
    }
}

/// Mirror one hi-res byte horizontally: reverse the 7 pixel bits (bit 0 =
/// leftmost pixel) and keep bit 7 (the palette / half-dot bit).
fn mirror_byte(b: u8) -> u8 {
    let reversed_7 = (b & 0x7f).reverse_bits() >> 1;
    (b & 0x80) | (reversed_7 & 0x7f)
}

/// The first and last **byte column** of `img` carrying any set pixel
/// (`b & 0x7f != 0`) on any row — the figure's true horizontal extent inside
/// its (often wider) sprite box. `None` for an empty / zero-size sprite.
///
/// Lets a caller centre the *figure* (not the box) on a character's logical x
/// and size his wall collision to what's actually drawn, so the gap to a wall
/// is the same whichever way he faces (#123). Uses the same `& 0x7f == 0`
/// transparency rule as [`composite_hires`].
#[must_use]
#[allow(clippy::verbose_bit_mask)]
pub fn figure_byte_span(img: &Image) -> Option<(u8, u8)> {
    let w = usize::from(img.width_bytes);
    let h = usize::from(img.height);
    if w == 0 || h == 0 || img.bitmap.len() < w * h {
        return None;
    }
    let mut span: Option<(u8, u8)> = None;
    for bx in 0..w {
        if (0..h).any(|sy| img.bitmap[sy * w + bx] & 0x7f != 0) {
            let bx = u8::try_from(bx).unwrap_or(u8::MAX);
            span = Some(match span {
                Some((lo, _)) => (lo, bx),
                None => (bx, bx),
            });
        }
    }
    span
}

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

    #[test]
    fn mirror_byte_reverses_pixels_keeps_palette() {
        // bit 0 (leftmost pixel) → bit 6 (rightmost).
        assert_eq!(mirror_byte(0b000_0001), 0b100_0000);
        assert_eq!(mirror_byte(0b100_0000), 0b000_0001);
        // Palette/high bit preserved; pixel bits reversed.
        assert_eq!(mirror_byte(0b1000_0001), 0b1100_0000);
        // Symmetric pattern is its own mirror.
        assert_eq!(mirror_byte(0b000_1000), 0b000_1000);
    }

    #[test]
    fn composite_hires_places_and_flips_bytes() {
        // 2-byte-wide, 1-row sprite. Bitmap is bottom-up; one row here.
        let img = Image {
            width_bytes: 2,
            height: 1,
            bitmap: vec![0b000_0001, 0b000_0000], // left byte has a pixel
        };
        // 4 bytes wide × 1 row top-down buffer.
        let mut buf = [0u8; 4];
        composite_hires(&mut buf, 4, 1, &img, 1, 0, false);
        assert_eq!(buf, [0, 0b000_0001, 0, 0], "byte 0 lands at column 1");

        // Flipped: byte order reverses (left byte → right column) and the
        // pixel bits mirror within the byte.
        let mut buf = [0u8; 4];
        composite_hires(&mut buf, 4, 1, &img, 1, 0, true);
        assert_eq!(
            buf,
            [0, 0, 0b100_0000, 0],
            "mirrored to column 2, pixel reversed"
        );
    }

    #[test]
    fn figure_byte_span_finds_the_used_columns() {
        // 4-byte-wide, 2-row sprite; only byte cols 1 and 2 carry pixels.
        let img = Image {
            width_bytes: 4,
            height: 2,
            bitmap: vec![
                0, 0b000_0001, 0, 0, // row 0: col 1
                0, 0, 0b100_0000, 0, // row 1: col 2
            ],
        };
        assert_eq!(figure_byte_span(&img), Some((1, 2)));

        // All-empty → None.
        let empty = Image {
            width_bytes: 2,
            height: 1,
            bitmap: vec![0, 0],
        };
        assert_eq!(figure_byte_span(&empty), None);

        // A byte with only the palette/high bit set has no pixels → empty.
        let hibit = Image {
            width_bytes: 1,
            height: 1,
            bitmap: vec![0x80],
        };
        assert_eq!(figure_byte_span(&hibit), None);
    }

    #[test]
    fn composite_hires_clips_and_skips_empty() {
        let img = Image {
            width_bytes: 1,
            height: 1,
            bitmap: vec![0b000_0000], // no pixels
        };
        let mut buf = [0xffu8; 2];
        composite_hires(&mut buf, 2, 1, &img, 0, 0, false);
        assert_eq!(buf, [0xff, 0xff], "empty sprite byte is transparent");
        // Out-of-range row is clipped (no panic).
        let solid = Image {
            width_bytes: 1,
            height: 1,
            bitmap: vec![0b000_0011],
        };
        composite_hires(&mut buf, 2, 1, &solid, 0, 5, false);
        assert_eq!(buf, [0xff, 0xff], "off-buffer write clipped");
    }
}
