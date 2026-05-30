//! Pluggable hardware backend.
//!
//! The original Apple II graphics, sound, and input are reached through
//! these traits. Pass 3 of the lifter rewrites GRAFIX/HIRES/SOUND/keyboard
//! calls into trait method calls.

pub trait Renderer {
    fn blit_sprite(&mut self, sprite: SpriteId, x: i16, y: i16);
    fn clear(&mut self);
    fn flip(&mut self);
}

pub trait Audio {
    fn play(&mut self, sound: SoundId);
}

pub trait Input {
    fn poll(&mut self) -> InputState;
}

#[derive(Copy, Clone, Debug)]
pub struct SpriteId(pub u16);

#[derive(Copy, Clone, Debug)]
pub struct SoundId(pub u16);

// Directional + modifier flags are independent and can be held
// simultaneously, so a flat set of bools is the right shape here — a
// state machine / enum would misrepresent the domain.
#[allow(clippy::struct_excessive_bools)]
#[derive(Copy, Clone, Debug, Default)]
pub struct InputState {
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub shift: bool,
}
