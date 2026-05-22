//! Central game state.
//!
//! `Game` will eventually hold every byte that the original 6502 build
//! addressed as a global. Pass 4 of the lifter rewrites global memory
//! accesses into field accesses on this struct.

#[derive(Default)]
pub struct Game {
    // Fields are emitted by the lifter; this is intentionally empty for now.
}

impl Game {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}
