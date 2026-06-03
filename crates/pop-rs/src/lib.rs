//! Rust port of the original *Prince of Persia* (Apple II, 1989) game logic.
//!
//! See the repository `README.md` and `docs/architecture.md` for design.

mod anim;
pub mod backend;
pub mod data;
pub mod game;
pub mod modules;
pub mod world;

pub use game::Game;
pub use world::{Mode, PrinceDebug, World};
