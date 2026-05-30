//! 8 KiB hi-res screen dumps from `DRAZ/I/`, `DRAZ/SS/`, and `DRAZ/IP/`.
//!
//! These files are byte-for-byte copies of one Apple II hi-res page
//! (8192 bytes, the same layout you'd get by snapshotting `$2000` or
//! `$4000` at runtime). Pair [`from_file`] / [`from_bytes`] with
//! [`crate::hires::render`] to produce a 280 × 192 RGBA frame.
//!
//! ```text
//!   let bytes  = pop_assets::draz::screen::from_file("vendor/.../SS/SS0")?;
//!   let frame  = pop_assets::hires::render(&bytes, RenderMode::NtscColor);
//! ```
//!
//! Heuristic ID: any file under `04 Support/DRAZ/` whose length is
//! exactly [`crate::hires::HIRES_PAGE_BYTES`] is a screen dump in this
//! sense. (A handful of `.OLD` and `TEST` variants in the same folders
//! also fit.)

use std::path::Path;

use thiserror::Error;

use crate::hires::HIRES_PAGE_BYTES;

/// Errors returned from [`from_bytes`] and [`from_file`].
#[derive(Debug, Error)]
pub enum ParseError {
    /// The buffer wasn't exactly [`HIRES_PAGE_BYTES`] bytes.
    #[error("hi-res screen file must be {HIRES_PAGE_BYTES} bytes, got {0}")]
    WrongSize(usize),
    /// I/O failure while reading the file from disk.
    #[error("reading screen file: {0}")]
    Io(#[from] std::io::Error),
}

/// Validate a byte slice as an 8 KiB hi-res page and return it as a
/// boxed fixed-size array suitable for [`crate::hires::render`].
///
/// # Errors
///
/// [`ParseError::WrongSize`] if `bytes.len() != HIRES_PAGE_BYTES`.
pub fn from_bytes(bytes: &[u8]) -> Result<Box<[u8; HIRES_PAGE_BYTES]>, ParseError> {
    if bytes.len() != HIRES_PAGE_BYTES {
        return Err(ParseError::WrongSize(bytes.len()));
    }
    let mut arr: Box<[u8; HIRES_PAGE_BYTES]> = Box::new([0u8; HIRES_PAGE_BYTES]);
    arr.copy_from_slice(bytes);
    Ok(arr)
}

/// Read and validate a hi-res screen file from disk.
///
/// # Errors
///
/// [`ParseError::Io`] on read failure; [`ParseError::WrongSize`] if the
/// file isn't exactly [`HIRES_PAGE_BYTES`] bytes.
pub fn from_file(path: impl AsRef<Path>) -> Result<Box<[u8; HIRES_PAGE_BYTES]>, ParseError> {
    let bytes = std::fs::read(path.as_ref())?;
    from_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn draz_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor/pop-apple2/04 Support/DRAZ")
    }

    #[test]
    fn wrong_size_rejected() {
        let err = from_bytes(&[0u8; 100]).unwrap_err();
        assert!(matches!(err, ParseError::WrongSize(100)));
    }

    #[test]
    fn ss0_is_a_valid_screen() {
        let p = draz_dir().join("SS/SS0");
        let arr = from_file(&p).expect("SS0 readable");
        assert_eq!(arr.len(), HIRES_PAGE_BYTES);
    }
}
