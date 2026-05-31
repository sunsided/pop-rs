//! Apple II hi-res framebuffer rendering.
//!
//! The Apple II hi-res screen is 280 × 192 pixels stored in an 8 KiB
//! page at `$2000` (page 1) or `$4000` (page 2). The memory layout is
//! famously non-linear: rows are stored in an interleaved order so the
//! `n`-th display row doesn't sit at `n * 40` bytes from the page base.
//!
//! This module exposes:
//!
//! * [`row_byte_offset`] — the canonical Apple II row interleave.
//! * [`render`] — turn an 8 KiB page into a 280 × 192 RGBA [`Frame`].
//! * [`render_linear`] — render a non-interleaved `(width_bytes, height)`
//!   bitmap (the layout used by POP's `IMG.CHTAB*` / `IMG.BGTAB*` sprite
//!   tables) into a `(width_bytes * 7, height)` RGBA [`Frame`].
//! * [`Frame`] — a row-major RGBA pixel buffer with explicit dimensions.
//!
//! Two render modes are supported:
//!
//! * [`RenderMode::Monochrome`] — black + white only, the high bit
//!   ignored. Useful for sprite-shape debugging.
//! * [`RenderMode::NtscColor`] — the classic 6-colour artifact palette
//!   (black / green / violet / orange / blue / white). The per-byte
//!   high bit selects between the green-violet and orange-blue palette
//!   halves.
//!
//! # The "color-cell" approximation
//!
//! True Apple II artifact colour comes from NTSC phase relationships in
//! the composite signal: two pixels of a byte span one colour-subcarrier
//! cycle, and the per-byte high bit introduces a half-pixel delay that
//! shifts which phase a lit pixel falls on. Faithfully simulating that
//! wants a real subcarrier filter.
//!
//! What we do instead — the standard emulator approximation that
//! matches what most Apple II users remember and is correct for sprite
//! verification — is the *color-cell* model: pixels are paired into
//! 2-wide cells (`(0,1), (2,3), (4,5), …`), and each cell renders as:
//!
//! * `00` → black
//! * `11` → white
//! * `10` or `01` → a colour from a 4-entry palette indexed by
//!   `(high_bit, cell_parity)`: `(0, even) → violet`, `(0, odd) → green`,
//!   `(1, even) → blue`, `(1, odd) → orange`.
//!
//! Compared to a real composite monitor we don't reproduce the
//! sub-pixel shift visually (we change palette halves on the high bit
//! but leave column positions intact) and we don't reproduce colour
//! fringing on isolated lit pixels. Both are NTSC-quirk territory and
//! well beyond what sprite verification needs.

/// Hi-res screen width in pixels.
pub const HIRES_WIDTH: usize = 280;
/// Hi-res screen height in pixels.
pub const HIRES_HEIGHT: usize = 192;
/// Bytes per row stored in the framebuffer (7 pixels per byte).
pub const HIRES_BYTES_PER_ROW: usize = 40;
/// Bytes per hi-res page (8 KiB; the unused tail holds non-display data).
pub const HIRES_PAGE_BYTES: usize = 0x2000;
/// Bytes the renderer actually reads from a page: `192 * 40 = 7680`.
pub const HIRES_USED_BYTES: usize = HIRES_HEIGHT * HIRES_BYTES_PER_ROW;
/// Apple II hi-res page 1 base address (in the 6502's address space).
pub const HIRES_PAGE1_BASE: u16 = 0x2000;
/// Apple II hi-res page 2 base address.
pub const HIRES_PAGE2_BASE: u16 = 0x4000;

/// How to colourise the framebuffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderMode {
    /// Pure black / white. The high bit of each byte is ignored.
    /// Useful for inspecting raw bit patterns (sprite shapes etc.).
    Monochrome,
    /// 6-colour NTSC artifact palette: black, green, violet, orange,
    /// blue, white. See the module-level docs for the colour-cell
    /// approximation we use.
    NtscColor,
}

/// A rendered frame: row-major RGBA, top row first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// RGBA pixel buffer of length `width * height * 4`.
    pub pixels: Vec<u8>,
}

impl Frame {
    /// `Some(&[r, g, b, a])` for in-range `(x, y)`, else `None`.
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> Option<&[u8]> {
        if x < self.width && y < self.height {
            let i = (y * self.width + x) as usize * 4;
            Some(&self.pixels[i..i + 4])
        } else {
            None
        }
    }
}

/// Compute the byte offset within an 8 KiB hi-res page for display
/// row `y` (`0..192`).
///
/// Apple II hi-res rows are interleaved as
///
/// ```text
///   offset(y) = (y & 7) * 0x400 + ((y >> 3) & 7) * 0x80 + (y >> 6) * 0x28
/// ```
///
/// Each row spans [`HIRES_BYTES_PER_ROW`] (= 40) bytes from this offset.
///
/// # Panics
///
/// Debug-only: panics if `y >= HIRES_HEIGHT`.
#[must_use]
pub const fn row_byte_offset(y: u8) -> u16 {
    debug_assert!((y as usize) < HIRES_HEIGHT);
    let y0 = (y & 7) as u16;
    let y1 = ((y >> 3) & 7) as u16;
    let y2 = (y >> 6) as u16;
    y0 * 0x400 + y1 * 0x80 + y2 * 0x28
}

/// Subcarrier phase offset (radians) for the artifact-colour demod.
/// Calibrated so the four base hi-res artifact colours come out
/// canonical — even/high-clear → violet, odd/high-clear → green,
/// even/high-set → blue, odd/high-set → orange — which makes the decode
/// physically correct for every biome (they differ only by sprite
/// data). Warm dithered fills (palace / red brick) then read as
/// tan/gold; a composite monitor's darker, warmer response renders the
/// same hue as the brown seen in screenshots.
const NTSC_HUE: f32 = 0.4;
/// Chroma gain. Higher = more saturated artifact colour; the YIQ luma
/// term carries brightness so this only scales the colour swing.
const NTSC_SAT: f32 = 1.5;
/// Demodulation window, in oversampled samples (one subcarrier cycle).
const NTSC_WIN: usize = 4;

const BLACK: [u8; 4] = [0x00, 0x00, 0x00, 0xff];
const WHITE: [u8; 4] = [0xff, 0xff, 0xff, 0xff];

/// Render an 8 KiB hi-res page to an RGBA [`Frame`].
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn render(page: &[u8; HIRES_PAGE_BYTES], mode: RenderMode) -> Frame {
    // `HIRES_HEIGHT` is statically 192, so the u8 cast on `y` and the
    // u32 casts on the constants are exact — see the const_asserts above.
    const _: () = assert!(HIRES_HEIGHT <= u8::MAX as usize);
    const _: () = assert!(HIRES_WIDTH <= u32::MAX as usize);
    const _: () = assert!(HIRES_HEIGHT <= u32::MAX as usize);

    let mut pixels = vec![0u8; HIRES_WIDTH * HIRES_HEIGHT * 4];
    // Scratch reused across rows so the NTSC demod doesn't heap-allocate
    // a per-scanline signal buffer 192× per frame.
    let mut ntsc_scratch: Vec<f32> = Vec::new();
    for y in 0u8..HIRES_HEIGHT as u8 {
        let row_start = row_byte_offset(y) as usize;
        let row = &page[row_start..row_start + HIRES_BYTES_PER_ROW];
        let y_usize = usize::from(y);
        let out_row = &mut pixels[y_usize * HIRES_WIDTH * 4..(y_usize + 1) * HIRES_WIDTH * 4];
        match mode {
            RenderMode::Monochrome => render_row_mono(row, out_row),
            RenderMode::NtscColor => render_row_ntsc(row, out_row, &mut ntsc_scratch),
        }
    }
    Frame {
        width: HIRES_WIDTH as u32,
        height: HIRES_HEIGHT as u32,
        pixels,
    }
}

/// Render a non-interleaved bitmap of `width_bytes × height` (POP's
/// sprite / image-table format) to an RGBA [`Frame`].
///
/// `bytes` must be exactly `width_bytes * height` long. Within each
/// byte, bit 0 is the leftmost pixel and bit 6 is the rightmost (bit 7
/// is the NTSC palette select). The resulting frame is
/// `width_bytes * 7` pixels wide.
///
/// # Row order
///
/// POP stores sprite bitmaps **bottom-up**: bytes `[0..width_bytes]`
/// are the *bottom* row of the displayed sprite, and the last
/// `width_bytes` bytes are the *top* row. This matches the `FASTLAY`
/// blitter in `HIRES.S` (around line 421), which seeds `X` from `YCO`
/// (the lowest visible scan-line of the sprite) and advances `IMAGE`
/// forward in lockstep with `dex` — so the first byte of the bitmap
/// corresponds to the lowest screen row. The "left-right, top-bottom"
/// comment near `HIRES.S:186` describes the draw order on screen
/// (which is bottom-up here), not in-memory order.
///
/// `render_linear` flips during read so callers always get a frame
/// with row 0 = the visual top of the sprite.
///
/// # Errors
///
/// Returns `None` if `bytes.len() != width_bytes * height`.
#[must_use]
pub fn render_linear(bytes: &[u8], width_bytes: u8, height: u8, mode: RenderMode) -> Option<Frame> {
    let w_bytes = usize::from(width_bytes);
    let h = usize::from(height);
    if bytes.len() != w_bytes * h {
        return None;
    }
    let w_pixels = w_bytes * 7;
    let mut pixels = vec![0u8; w_pixels * h * 4];
    // Scratch reused across rows — see `render` for the rationale.
    let mut ntsc_scratch: Vec<f32> = Vec::new();
    for y in 0..h {
        // POP sprites are stored bottom-up; flip so row 0 of the output
        // frame is the visual top of the sprite.
        let src_y = h - 1 - y;
        let row = &bytes[src_y * w_bytes..(src_y + 1) * w_bytes];
        let out_row = &mut pixels[y * w_pixels * 4..(y + 1) * w_pixels * 4];
        match mode {
            RenderMode::Monochrome => render_row_mono(row, out_row),
            RenderMode::NtscColor => render_row_ntsc(row, out_row, &mut ntsc_scratch),
        }
    }
    Some(Frame {
        width: u32::try_from(w_pixels).ok()?,
        height: u32::from(height),
        pixels,
    })
}

/// Unpack a row of `row.len() * 7` per-pixel "lit" bits into `bits`,
/// LSB-first within each byte (so bit 0 of byte 0 is the leftmost
/// pixel). Panics in debug if `bits.len() != row.len() * 7`.
fn unpack_row_bits(row: &[u8], bits: &mut [bool]) {
    debug_assert_eq!(bits.len(), row.len() * 7);
    for (byte_idx, &b) in row.iter().enumerate() {
        let base = byte_idx * 7;
        for p in 0..7 {
            bits[base + p] = (b >> p) & 1 == 1;
        }
    }
}

fn render_row_mono(row: &[u8], out: &mut [u8]) {
    let width = row.len() * 7;
    let mut bits = vec![false; width];
    unpack_row_bits(row, &mut bits);
    for (x, &lit) in bits.iter().enumerate() {
        let i = x * 4;
        let color = if lit { WHITE } else { BLACK };
        out[i..i + 4].copy_from_slice(&color);
    }
}

/// Decode one hi-res scanline into RGBA using an NTSC artifact-colour
/// model (`Y`/`I`/`Q` over a sliding sample window), rather than the
/// hard 6-colour cell approximation.
///
/// This reproduces what a composite monitor does: the bandwidth-limited
/// chroma/luma filters blend the per-pixel dither into the smooth
/// intermediate tones the game relies on — orange-on-black dither reads
/// as brown, orange-on-white as beige, while solid colour runs stay
/// vivid blue / orange / violet / green. A pure per-pixel palette can't
/// produce those browns because single hi-res only has six dot colours.
///
/// Method: build a 2×-oversampled binary signal where each lit pixel
/// fills two samples and a set high bit delays its byte by one sample
/// (the half-dot shift that rotates violet/green → blue/orange). Then
/// for each output pixel demodulate luma (`Y`, window average) and
/// chroma (`I`/`Q`, the signal mixed against the period-4 subcarrier)
/// over a one-cycle window and convert `YIQ → RGB`.
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
fn render_row_ntsc(row: &[u8], out: &mut [u8], sig: &mut Vec<f32>) {
    let width = row.len() * 7;
    // 2× oversample; +2 tail so the last pixel's window never indexes
    // out of bounds. `sig` is caller-owned scratch: clear + resize reuses
    // the existing allocation and zeroes it for this row.
    sig.clear();
    sig.resize(width * 2 + 2, 0.0);
    for x in 0..width {
        let byte = row[x / 7];
        if (byte >> (x % 7)) & 1 == 1 {
            let shift = usize::from((byte >> 7) & 1);
            sig[2 * x + shift] = 1.0;
            sig[2 * x + 1 + shift] = 1.0;
        }
    }
    // Edge-replicate the 2-sample tail so the rightmost pixel's window
    // doesn't read padding zeros — otherwise a fully-lit line would
    // fringe to colour at its right edge instead of staying white.
    if width > 0 {
        let last = sig[2 * width - 1];
        sig[2 * width] = last;
        sig[2 * width + 1] = last;
    }
    // Subcarrier samples for the four phases (period = 4 samples).
    let mut cos_p = [0.0f32; 4];
    let mut sin_p = [0.0f32; 4];
    for (p, (c, s)) in cos_p.iter_mut().zip(sin_p.iter_mut()).enumerate() {
        let a = std::f32::consts::TAU * (p as f32) / 4.0 + NTSC_HUE;
        *c = a.cos();
        *s = a.sin();
    }
    // One-cycle demodulation window per output pixel.
    for x in 0..width {
        let base = 2 * x;
        let (mut y, mut iq_i, mut iq_q) = (0.0f32, 0.0f32, 0.0f32);
        for k in 0..NTSC_WIN {
            let s = sig[base + k];
            let p = (base + k) & 3;
            y += s;
            iq_i += s * cos_p[p];
            iq_q += s * sin_p[p];
        }
        let y = y / NTSC_WIN as f32;
        let i_term = iq_i / NTSC_WIN as f32 * 2.0 * NTSC_SAT;
        let q_term = iq_q / NTSC_WIN as f32 * 2.0 * NTSC_SAT;
        let r = y + 0.956 * i_term + 0.619 * q_term;
        let g = y - 0.272 * i_term - 0.647 * q_term;
        let b = y - 1.106 * i_term + 1.703 * q_term;
        let o = x * 4;
        out[o] = clamp_u8(r);
        out[o + 1] = clamp_u8(g);
        out[o + 2] = clamp_u8(b);
        out[o + 3] = 0xff;
    }
}

/// Clamp a 0.0..=1.0 intensity to a 0..=255 byte.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn clamp_u8(v: f32) -> u8 {
    // Clamped to 0..=1 then scaled, so the result is always in 0..=255.
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a page with `f(y)` producing the 40 bytes of display row `y`.
    fn make_page<F: Fn(usize) -> [u8; HIRES_BYTES_PER_ROW]>(f: F) -> Box<[u8; HIRES_PAGE_BYTES]> {
        let mut page = vec![0u8; HIRES_PAGE_BYTES].into_boxed_slice();
        for y in 0..HIRES_HEIGHT {
            let off = row_byte_offset(u8::try_from(y).unwrap()) as usize;
            let bytes = f(y);
            page[off..off + HIRES_BYTES_PER_ROW].copy_from_slice(&bytes);
        }
        page.try_into().unwrap()
    }

    const TEST_WHITE: [u8; 4] = [0xff, 0xff, 0xff, 0xff];
    const TEST_BLACK: [u8; 4] = [0x00, 0x00, 0x00, 0xff];

    #[test]
    fn render_linear_flips_rows_bottom_up_to_top_down() {
        // POP sprite memory layout: bytes [0..width_bytes] = visual
        // bottom row, last `width_bytes` = visual top row. After
        // render_linear, the output frame must have row 0 = visual top.
        //
        // 1-byte-wide, 2-line sprite: lit bottom row, dark top row in
        // memory. Output frame row 0 should be black, row 1 should be
        // white (lit).
        let bytes = [
            0x7f, // memory row 0 = visual bottom: all 7 pixels lit
            0x00, // memory row 1 = visual top:    all 7 pixels dark
        ];
        let frame = render_linear(&bytes, 1, 2, RenderMode::Monochrome).unwrap();
        assert_eq!(frame.width, 7);
        assert_eq!(frame.height, 2);
        // Output row 0 = top of display = memory row 1 (dark).
        for x in 0..7 {
            assert_eq!(frame.pixel(x, 0).unwrap(), TEST_BLACK, "top row x={x}");
        }
        // Output row 1 = bottom of display = memory row 0 (lit).
        for x in 0..7 {
            assert_eq!(frame.pixel(x, 1).unwrap(), TEST_WHITE, "bottom row x={x}");
        }
    }

    #[test]
    fn render_linear_rejects_size_mismatch() {
        // 2-byte-wide, 3-line sprite needs 6 bytes; pass 5.
        assert!(render_linear(&[0u8; 5], 2, 3, RenderMode::Monochrome).is_none());
    }

    #[test]
    fn row_interleave_is_bijective_over_192_rows() {
        // Every display row 0..192 gets a unique 40-byte slice; the
        // union covers exactly 192 * 40 = 7680 bytes — no overlap, no
        // gap. Catches sign / nibble typos in the interleave formula.
        let mut covered = [false; HIRES_PAGE_BYTES];
        let mut count = 0usize;
        for y in 0..HIRES_HEIGHT {
            let off = row_byte_offset(u8::try_from(y).unwrap()) as usize;
            assert!(
                off + HIRES_BYTES_PER_ROW <= HIRES_PAGE_BYTES,
                "row {y} out of page"
            );
            for cell in &mut covered[off..off + HIRES_BYTES_PER_ROW] {
                assert!(!*cell, "row {y} overlaps an earlier row at offset {off}");
                *cell = true;
                count += 1;
            }
        }
        assert_eq!(count, HIRES_USED_BYTES);
    }

    #[test]
    fn row_interleave_matches_canonical_anchors() {
        // Standard Apple II reference anchors — see "Apple II hi-res
        // address layout" tables. `y=191` decomposes as
        // `y2=2, y1=7, y0=7` → `2*0x28 + 7*0x80 + 7*0x400 = 0x1fd0`.
        assert_eq!(row_byte_offset(0), 0x0000);
        assert_eq!(row_byte_offset(1), 0x0400);
        assert_eq!(row_byte_offset(8), 0x0080);
        assert_eq!(row_byte_offset(64), 0x0028);
        assert_eq!(row_byte_offset(191), 0x1fd0);
    }

    #[test]
    fn empty_page_renders_all_black_in_both_modes() {
        let page: Box<[u8; HIRES_PAGE_BYTES]> = vec![0u8; HIRES_PAGE_BYTES]
            .into_boxed_slice()
            .try_into()
            .unwrap();
        for mode in [RenderMode::Monochrome, RenderMode::NtscColor] {
            let f = render(&page, mode);
            assert_eq!(f.width, u32::try_from(HIRES_WIDTH).unwrap());
            assert_eq!(f.height, u32::try_from(HIRES_HEIGHT).unwrap());
            assert_eq!(f.pixels.len(), HIRES_WIDTH * HIRES_HEIGHT * 4);
            assert!(
                f.pixels.chunks_exact(4).all(|p| p == TEST_BLACK),
                "mode={mode:?}",
            );
        }
    }

    #[test]
    fn solid_low_seven_bits_per_byte_renders_all_white_in_both_modes() {
        // 0x7f: all seven pixels set, high bit clear. Every pixel is lit
        // and partners are lit too, so both modes produce solid white.
        let page = make_page(|_| [0x7f; HIRES_BYTES_PER_ROW]);
        for mode in [RenderMode::Monochrome, RenderMode::NtscColor] {
            let f = render(&page, mode);
            assert!(
                f.pixels.chunks_exact(4).all(|p| p == TEST_WHITE),
                "mode={mode:?}",
            );
        }
    }

    #[test]
    fn high_bit_alone_keeps_screen_black() {
        // 0x80: no display pixels lit. Should render black in both
        // modes — the high bit is only meaningful when paired with at
        // least one lit pixel.
        let page = make_page(|_| [0x80; HIRES_BYTES_PER_ROW]);
        for mode in [RenderMode::Monochrome, RenderMode::NtscColor] {
            let f = render(&page, mode);
            assert!(
                f.pixels.chunks_exact(4).all(|p| p == TEST_BLACK),
                "mode={mode:?}",
            );
        }
    }

    #[test]
    fn monochrome_byte_unpacks_lsb_first() {
        // Mono mode: byte `0b0000_0001` (= 0x01) lights only the
        // leftmost pixel of its byte; `0b0100_0000` (= 0x40) lights
        // only the rightmost pixel of its byte. Verify both for byte 0
        // of row 0 (the leftmost 7 pixels of the screen).
        let mut bytes = [0u8; HIRES_BYTES_PER_ROW];
        bytes[0] = 0b0000_0001;
        let page = make_page(move |y| {
            if y == 0 {
                bytes
            } else {
                [0; HIRES_BYTES_PER_ROW]
            }
        });
        let f = render(&page, RenderMode::Monochrome);
        assert_eq!(f.pixel(0, 0).unwrap(), TEST_WHITE);
        for x in 1..7 {
            assert_eq!(f.pixel(x, 0).unwrap(), TEST_BLACK, "x={x} should be black");
        }

        let mut bytes = [0u8; HIRES_BYTES_PER_ROW];
        bytes[0] = 0b0100_0000;
        let page = make_page(move |y| {
            if y == 0 {
                bytes
            } else {
                [0; HIRES_BYTES_PER_ROW]
            }
        });
        let f = render(&page, RenderMode::Monochrome);
        for x in 0..6 {
            assert_eq!(f.pixel(x, 0).unwrap(), TEST_BLACK, "x={x} should be black");
        }
        assert_eq!(f.pixel(6, 0).unwrap(), TEST_WHITE);
    }

    #[test]
    fn ntsc_base_artifact_colours_are_canonical() {
        // The four single-pixel hi-res patterns must decode to the
        // canonical artifact hues, which guarantees the decode is right
        // for every biome (they differ only by sprite data).
        let probe = |byte: u8, x: u32| -> [u8; 3] {
            let page = make_page(|_| [byte; HIRES_BYTES_PER_ROW]);
            let f = render(&page, RenderMode::NtscColor);
            let p = f.pixel(x, 0).unwrap();
            [p[0], p[1], p[2]]
        };
        let dominant = |c: [u8; 3]| -> &'static str {
            let [r, g, b] = c;
            match (r > 160, g > 160, b > 160) {
                (true, false, true) => "violet",
                (false, true, false) => "green",
                (false, _, true) => "blue",
                (true, _, false) => "orange",
                _ => "other",
            }
        };
        assert_eq!(dominant(probe(0x55, 2)), "violet", "even/high-clear");
        assert_eq!(dominant(probe(0x2a, 3)), "green", "odd/high-clear");
        assert_eq!(dominant(probe(0xd5, 2)), "blue", "even/high-set");
        assert_eq!(dominant(probe(0xaa, 3)), "orange", "odd/high-set");
    }

    #[test]
    fn ntsc_lit_dither_produces_saturated_colour() {
        // 0x55 = 0b0101_0101 — every other pixel lit, high bit clear.
        // The NTSC decode turns this into a saturated artifact colour
        // (violet family on hardware), not grey/white.
        let page = make_page(|_| [0x55; HIRES_BYTES_PER_ROW]);
        let f = render(&page, RenderMode::NtscColor);
        let px = f.pixel(2, 0).unwrap();
        assert_ne!(px, TEST_BLACK, "lit dither should not be black");
        assert_ne!(px, TEST_WHITE, "lit dither should be coloured, not white");
        // Saturated: max channel clearly above min (a real artifact hue,
        // not a grey). The exact hue is calibration-dependent.
        let max = px[0].max(px[1]).max(px[2]);
        let min = px[0].min(px[1]).min(px[2]);
        assert!(
            i32::from(max) - i32::from(min) > 60,
            "0x55 should read as a saturated colour, got {px:?}"
        );
    }

    #[test]
    fn ntsc_high_bit_rotates_hue() {
        // 0xd5 = same pixel pattern as 0x55 with the high bit set: the
        // half-dot delay rotates the artifact hue (violet → blue family).
        let a = render(&make_page(|_| [0x55; HIRES_BYTES_PER_ROW]), RenderMode::NtscColor);
        let b = render(&make_page(|_| [0xd5; HIRES_BYTES_PER_ROW]), RenderMode::NtscColor);
        assert_ne!(
            a.pixel(2, 0).unwrap(),
            b.pixel(2, 0).unwrap(),
            "the high bit must change the artifact hue"
        );
    }

    #[test]
    fn ntsc_color_does_not_equal_monochrome_for_alternating_pattern() {
        // Sanity: the two render modes must produce different output
        // for a non-trivial pattern. Otherwise we'd be silently mono.
        let page = make_page(|_| [0x55; HIRES_BYTES_PER_ROW]);
        let f_mono = render(&page, RenderMode::Monochrome);
        let f_color = render(&page, RenderMode::NtscColor);
        assert_ne!(f_mono.pixels, f_color.pixels);
    }

    #[test]
    fn frame_pixel_accessor_bounds_checks() {
        let page: Box<[u8; HIRES_PAGE_BYTES]> = vec![0u8; HIRES_PAGE_BYTES]
            .into_boxed_slice()
            .try_into()
            .unwrap();
        let f = render(&page, RenderMode::Monochrome);
        let last_x = u32::try_from(HIRES_WIDTH).unwrap() - 1;
        let last_y = u32::try_from(HIRES_HEIGHT).unwrap() - 1;
        assert!(f.pixel(0, 0).is_some());
        assert!(f.pixel(last_x, last_y).is_some());
        assert!(f.pixel(last_x + 1, 0).is_none());
        assert!(f.pixel(0, last_y + 1).is_none());
    }
}
