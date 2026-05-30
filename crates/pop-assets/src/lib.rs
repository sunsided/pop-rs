//! Asset extraction for Prince of Persia (Apple II).
//!
//! Pure-data library — file-format readers for the bundled game files
//! and (eventually) on-disk resources. No UI, no host I/O beyond
//! `std::fs`.
//!
//! Today's surface is just the level-binary reader [`level`]. Disk-image
//! parsing, sprite extraction, and animation-sequence decoding land
//! incrementally as the Phase 1 sub-issues of the umbrella tracker (#79)
//! are taken.

#![cfg_attr(not(test), warn(missing_docs))]

pub mod discovery;
pub mod hires;
pub mod level;
