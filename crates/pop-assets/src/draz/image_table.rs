//! POP image-table files: indexed sprite sheets used for animated
//! characters (`IMG.CHTAB*`) and biome backgrounds (`IMG.BGTAB*`).
//!
//! # On-disk format
//!
//! These files implement the table layout that `HIRES.S` consumes via
//! the `setimage` / `GETWIDTH` / `LAY` calls. Per-image bitmaps follow
//! the comment in `HIRES.S` (lines 182–186):
//!
//! ```text
//!   Byte 0:    width  (# of bytes; 1 byte = 7 pixels)
//!   Byte 1:    height (# of lines)
//!   Byte 2-n:  image bytes (read left-right, top-bottom)
//! ```
//!
//! The "top-bottom" wording is misleading: the actual blitter
//! (`FASTLAY`, `HIRES.S:421`) walks screen Y *up* from `YCO` (the
//! lowest visible scan-line) while advancing `IMAGE` *forward* through
//! the bitmap. That means the **first row of bytes in memory
//! corresponds to the bottom row of the displayed sprite** — i.e. the
//! bitmap is stored bottom-up. The extracted [`Image::bitmap`] keeps
//! this on-disk order verbatim; [`crate::hires::render_linear`] flips
//! at render time so callers get a top-down RGBA frame.
//!
//! The file as a whole wraps `N` such blobs behind a count + pointer
//! directory laid out at load address `$6000`:
//!
//! ```text
//!   Byte 0:                   N — image count
//!   Bytes 1 .. 1+2(N+1):      (N+1) little-endian 16-bit pointers
//!                             slots[0..N-1] = absolute addr of image 1..N
//!                             slot[N]       = fence-post (end of image N)
//!   Bytes 1+2(N+1) ..:        image blobs, packed
//! ```
//!
//! Image numbering at the 6502 source level is 1-based; slot index 0
//! in the on-disk directory holds the pointer for image 1. We expose
//! that directly via [`ImageTable::images`] as a `Vec<Image>` indexed
//! from 0 — callers translating from FRAMEDEF.S / BGDATA.S piece IDs
//! subtract 1.
//!
//! The base load address is inferred from the first pointer rather
//! than hardcoded, so the parser stays correct if a future POP variant
//! relocates the table.
//!
//! # What's *not* here
//!
//! Masks and image pieces live side-by-side in the same table; the
//! `BGDATA.S` / `FRAMEDEF.S` metadata tables decide which entry is a
//! mask vs. an image and how to compose them. That composition lands
//! with the editor / runtime layers — this module just extracts the
//! raw bitmaps.

use std::path::Path;

use thiserror::Error;

/// Errors returned from [`ImageTable::from_bytes`] and
/// [`ImageTable::from_file`].
#[derive(Debug, Error)]
pub enum ParseError {
    /// The file was shorter than the minimum viable header (count byte
    /// + one pointer slot).
    #[error("image table truncated: only {0} bytes")]
    TooSmall(usize),
    /// The pointer directory extends past the end of the file.
    #[error("image table pointer directory truncated: count={count}, file={file_len} bytes")]
    DirectoryTruncated {
        /// Image count declared in the header byte.
        count: usize,
        /// Total file length.
        file_len: usize,
    },
    /// The first pointer in the directory is smaller than the size of
    /// the header it implicitly defines (i.e. the file is internally
    /// inconsistent — base address can't be reconciled with directory
    /// size).
    #[error("first pointer 0x{first_pointer:04x} can't address byte {first_data_offset}")]
    BaseUnderflow {
        /// Raw absolute pointer for image 1 from the directory.
        first_pointer: u16,
        /// Byte offset where image data must begin (= `1 + 2(N+1)`).
        first_data_offset: usize,
    },
    /// A directory entry pointed below the inferred base address —
    /// directory pointers must be monotonically non-decreasing from
    /// `base_address`. Catches non-monotone / corrupt directories
    /// before they get silently mapped to byte 0 (the count byte).
    #[error(
        "image {image_index}: pointer 0x{pointer:04x} below inferred base 0x{base_address:04x}"
    )]
    PointerBelowBase {
        /// 0-based index of the offending directory entry.
        image_index: usize,
        /// Raw absolute pointer from the directory.
        pointer: u16,
        /// Inferred load base (from the first pointer).
        base_address: u16,
    },
    /// An image's `(width, height)` header pointed past the end of the
    /// file. Carries the offending image index (0-based) and the
    /// computed end offset.
    #[error(
        "image {image_index}: bitmap extends to byte {end} past end-of-file ({file_len} bytes)"
    )]
    ImageOutOfRange {
        /// 0-based index of the offending image.
        image_index: usize,
        /// Computed end offset (exclusive) within the file.
        end: usize,
        /// Total file length.
        file_len: usize,
    },
    /// I/O failure while reading the file from disk.
    #[error("reading image table file: {0}")]
    Io(#[from] std::io::Error),
}

/// One sprite from an image table.
///
/// Width is in bytes (each byte = 7 horizontal pixels in Apple II
/// hi-res); height is in scan-lines. The bitmap is exactly
/// `width_bytes * height` bytes, preserving POP's on-disk **bottom-up**
/// row order — bytes `[0..width_bytes]` are the bottom scan-line of the
/// displayed sprite, the last `width_bytes` bytes are the top
/// scan-line (see the module-level docs and `HIRES.S:421`). Hand to
/// [`crate::hires::render_linear`] for a top-down RGBA frame (it flips
/// during read).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    /// Image width in hi-res bytes (= pixels / 7).
    pub width_bytes: u8,
    /// Image height in scan-lines.
    pub height: u8,
    /// Linear bitmap, length `width_bytes * height`.
    pub bitmap: Vec<u8>,
}

impl Image {
    /// Width in pixels (= `width_bytes * 7`).
    #[must_use]
    pub fn width_pixels(&self) -> u16 {
        u16::from(self.width_bytes) * 7
    }
}

/// A parsed POP image table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageTable {
    /// Sprites in the order they appear in the directory. Index `i`
    /// here corresponds to image number `i + 1` in the 6502 sources.
    pub images: Vec<Image>,
    /// Inferred load base of the directory. Universally `$6000` in
    /// the bundled assets; surfaced for diagnostics.
    pub base_address: u16,
}

impl ImageTable {
    /// Parse a byte buffer in the on-disk image-table layout.
    ///
    /// # Errors
    ///
    /// See [`ParseError`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ParseError> {
        if bytes.len() < 3 {
            return Err(ParseError::TooSmall(bytes.len()));
        }
        let count = usize::from(bytes[0]);
        let directory_bytes = 2 * (count + 1);
        let first_data_offset = 1 + directory_bytes;
        if bytes.len() < first_data_offset {
            return Err(ParseError::DirectoryTruncated {
                count,
                file_len: bytes.len(),
            });
        }

        let first_pointer = read_u16_le(bytes, 1);
        let base_address = first_pointer
            .checked_sub(u16::try_from(first_data_offset).unwrap_or(u16::MAX))
            .ok_or(ParseError::BaseUnderflow {
                first_pointer,
                first_data_offset,
            })?;

        let mut images = Vec::with_capacity(count);
        for i in 0..count {
            let ptr = read_u16_le(bytes, 1 + 2 * i);
            let offset_in_table =
                ptr.checked_sub(base_address)
                    .ok_or(ParseError::PointerBelowBase {
                        image_index: i,
                        pointer: ptr,
                        base_address,
                    })?;
            let start = usize::from(offset_in_table);
            if start + 2 > bytes.len() {
                return Err(ParseError::ImageOutOfRange {
                    image_index: i,
                    end: start + 2,
                    file_len: bytes.len(),
                });
            }
            let width_bytes = bytes[start];
            let height = bytes[start + 1];
            let size = usize::from(width_bytes) * usize::from(height);
            let end = start + 2 + size;
            if end > bytes.len() {
                return Err(ParseError::ImageOutOfRange {
                    image_index: i,
                    end,
                    file_len: bytes.len(),
                });
            }
            images.push(Image {
                width_bytes,
                height,
                bitmap: bytes[start + 2..end].to_vec(),
            });
        }

        Ok(Self {
            images,
            base_address,
        })
    }

    /// Read and parse an image-table file from disk.
    ///
    /// # Errors
    ///
    /// See [`ParseError`].
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ParseError> {
        let bytes = std::fs::read(path.as_ref())?;
        Self::from_bytes(&bytes)
    }
}

fn read_u16_le(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn draz_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor/pop-apple2/04 Support/DRAZ")
    }

    #[test]
    fn too_small_rejected() {
        assert!(matches!(
            ImageTable::from_bytes(&[]),
            Err(ParseError::TooSmall(0))
        ));
        assert!(matches!(
            ImageTable::from_bytes(&[0, 0]),
            Err(ParseError::TooSmall(2))
        ));
    }

    #[test]
    fn truncated_directory_rejected() {
        // Says 10 images (= 22-byte directory + 1 count byte = 23
        // required) but only 5 bytes.
        let buf = [10u8, 0, 0, 0, 0];
        let err = ImageTable::from_bytes(&buf).unwrap_err();
        assert!(matches!(
            err,
            ParseError::DirectoryTruncated {
                count: 10,
                file_len: 5
            }
        ));
    }

    #[test]
    fn synthetic_two_image_round_trip() {
        // Two images: 1×1 (1 data byte) and 2×1 (2 data bytes).
        // Base 0x4000, directory size = 1 + 2*(2+1) = 7 bytes.
        //  byte 0      : count = 2
        //  bytes 1..7  : pointers
        //    image 1 at $4007, image 2 at $400A, fence at $400D
        //  bytes 7..9  : image 1 header (w=1, h=1) + 1 byte data
        //  bytes 10..13: image 2 header (w=2, h=1) + 2 bytes data
        let mut buf = vec![2u8];
        buf.extend_from_slice(&0x4007u16.to_le_bytes());
        buf.extend_from_slice(&0x400Au16.to_le_bytes());
        buf.extend_from_slice(&0x400Du16.to_le_bytes());
        buf.extend_from_slice(&[1, 1, 0xAA]); // image 1
        buf.extend_from_slice(&[2, 1, 0xBB, 0xCC]); // image 2

        let table = ImageTable::from_bytes(&buf).unwrap();
        assert_eq!(table.base_address, 0x4000);
        assert_eq!(table.images.len(), 2);
        assert_eq!(
            table.images[0],
            Image {
                width_bytes: 1,
                height: 1,
                bitmap: vec![0xAA],
            }
        );
        assert_eq!(
            table.images[1],
            Image {
                width_bytes: 2,
                height: 1,
                bitmap: vec![0xBB, 0xCC],
            }
        );
    }

    #[test]
    fn non_monotone_directory_rejected() {
        // Two-image directory where the second pointer is below the
        // base address derived from the first. Pre-fix, this was
        // silently mapped to offset 0 and produced garbage sprites.
        //  byte 0      : count = 2
        //  bytes 1..7  : pointers
        //    pointer 1 = $4007 (sets base = $4000), pointer 2 = $3FFF
        //    (below base), fence = $400D
        let mut buf = vec![2u8];
        buf.extend_from_slice(&0x4007u16.to_le_bytes());
        buf.extend_from_slice(&0x3FFFu16.to_le_bytes());
        buf.extend_from_slice(&0x400Du16.to_le_bytes());
        buf.extend_from_slice(&[1, 1, 0xAA]); // image 1 OK
        buf.extend_from_slice(&[2, 1, 0xBB, 0xCC]);

        let err = ImageTable::from_bytes(&buf).unwrap_err();
        assert!(
            matches!(
                err,
                ParseError::PointerBelowBase {
                    image_index: 1,
                    pointer: 0x3FFF,
                    base_address: 0x4000,
                }
            ),
            "unexpected: {err:?}"
        );
    }

    #[test]
    fn vendor_chtab1_parses_with_sane_sprites() {
        let table =
            ImageTable::from_file(draz_dir().join("I/IMG.CHTAB1")).expect("IMG.CHTAB1 parses");
        assert_eq!(table.base_address, 0x6000);
        assert_eq!(table.images.len(), 0x41); // 65, from header byte
        for (i, img) in table.images.iter().enumerate() {
            // Apple II hi-res screen is 40 bytes wide × 192 lines.
            // Real sprites are well below that.
            assert!(
                img.width_bytes <= 40,
                "image {i}: width_bytes {} > 40",
                img.width_bytes
            );
            assert!(img.height <= 192, "image {i}: height {} > 192", img.height);
            assert_eq!(
                img.bitmap.len(),
                usize::from(img.width_bytes) * usize::from(img.height),
                "image {i}: bitmap length mismatch"
            );
        }
    }

    #[test]
    fn vendor_bgtab_pal1_parses() {
        let table = ImageTable::from_file(draz_dir().join("IP/IMG.BGTAB.PAL1"))
            .expect("IMG.BGTAB.PAL1 parses");
        assert_eq!(table.base_address, 0x6000);
        assert_eq!(table.images.len(), 0x7e); // 126
    }

    #[test]
    fn vendor_bgtab_dun1_parses() {
        let table = ImageTable::from_file(draz_dir().join("IP/IMG.BGTAB.DUN1"))
            .expect("IMG.BGTAB.DUN1 parses");
        assert_eq!(table.base_address, 0x6000);
        assert_eq!(table.images.len(), 0x7e); // 126
    }
}
