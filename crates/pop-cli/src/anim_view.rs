//! Animation preview pane for the editor (#89).
//!
//! A floating window with two modes:
//!
//! * **Characters** — POP's decoded [`pop_assets::anim`] sequences. The
//!   figure is mirrored to face the way it travels (byte-space flip, so NTSC
//!   colours stay put — the #121 half-dot). A `mirror` toggle flips facing
//!   and travel together. A *body* picker swaps the CHTAB4 guard sprite
//!   (guard / fat / skeleton / shadow / vizier) so guard sequences render.
//! * **Tiles** — the animated tile pieces (torch flame, loose floor, spike,
//!   slicer, …) cycled through their phases via the scene compositor
//!   ([`pop_assets::scene::Anim`]) over a one-tile [`Level::preview_tile`].
//!
//! Both modes share the play / pause / step / loop / speed controls.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use eframe::egui::{
    self, Align2, Color32, ColorImage, FontId, Pos2, Rect, Sense, TextureHandle, Vec2,
};

use pop_assets::anim::{self, AnimSequence, SpriteRef};
use pop_assets::bgdata::Biome;
use pop_assets::discovery;
use pop_assets::draz::image_table::{Image, ImageTable};
use pop_assets::hires::{render_linear, RenderMode};
use pop_assets::level::{Level, Tile, TileKind};
use pop_assets::scene::{self, Anim, BiomeTables};

/// On-screen magnification of the previewed character sprite.
const PREVIEW_SCALE: f32 = 2.0;
/// Where the previewed tile sits in the synthetic room.
const TILE_COL: usize = 4;
const TILE_ROW: usize = 1;
/// Room frame dimensions (for fitting the tile preview).
const ROOM_W: f32 = 280.0;
const ROOM_H: f32 = 192.0;

/// The animatable / browsable tile kinds offered in Tiles mode.
const TILE_KINDS: &[(&str, TileKind)] = &[
    ("torch", TileKind::Torch),
    ("loose floor", TileKind::LooseFloor),
    ("spikes", TileKind::Spikes),
    ("slicer", TileKind::Slicer),
    ("gate", TileKind::Gate),
    ("sword", TileKind::Sword),
    ("exit", TileKind::Exit),
    ("flask", TileKind::Flask),
];

/// Which body fills the CHTAB4 slot (guard sprite variant).
#[derive(Clone, Copy, PartialEq, Eq)]
enum GuardKind {
    None,
    Guard,
    Fat,
    Skeleton,
    Shadow,
    Vizier,
}

impl GuardKind {
    const ALL: [GuardKind; 6] = [
        Self::None,
        Self::Guard,
        Self::Fat,
        Self::Skeleton,
        Self::Shadow,
        Self::Vizier,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::None => "Prince",
            Self::Guard => "Guard",
            Self::Fat => "Fat",
            Self::Skeleton => "Skeleton",
            Self::Shadow => "Shadow",
            Self::Vizier => "Vizier",
        }
    }

    /// The `IMG.CHTAB4.*` body file, or `None` for the Prince (no CHTAB4).
    fn chtab4_file(self) -> Option<&'static str> {
        Some(match self {
            Self::None => return None,
            Self::Guard => "IMG.CHTAB4.GD",
            Self::Fat => "IMG.CHTAB4.FAT",
            Self::Skeleton => "IMG.CHTAB4.SKEL",
            Self::Shadow => "IMG.CHTAB4.SHAD",
            Self::Vizier => "IMG.CHTAB4.VIZ",
        })
    }
}

/// Which kind of thing the pane is previewing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PreviewMode {
    Characters,
    Tiles,
}

/// Editor animation preview window state.
#[allow(clippy::struct_excessive_bools)] // independent UI toggles
pub struct AnimViewer {
    /// Whether the window is shown (toggled from the toolbar).
    pub open: bool,
    mode: PreviewMode,
    // Shared playback.
    playing: bool,
    looping: bool,
    speed_ms: u64,
    last_step: Option<Instant>,
    // Characters.
    /// CHTAB sprite tables, indexed by `chtab - 1` (`None` if absent).
    tables: Vec<Option<ImageTable>>,
    /// `(root, guard)` the CHTABs were loaded for, to reload on a change.
    chtab_key: Option<(PathBuf, GuardKind)>,
    guard: GuardKind,
    selected: usize,
    frame_pos: usize,
    user_flip: bool,
    /// Sprite textures keyed `(frame id, ntsc, mirror)`.
    cache: HashMap<(u8, bool, bool), TextureHandle>,
    // Tiles.
    bg: Option<BiomeTables>,
    bg_root: Option<PathBuf>,
    selected_tile: usize,
    tick: u32,
    /// Tile-frame texture for the displayed `(tile, tick, ntsc)`.
    tile_cache: Option<(usize, u32, bool, TextureHandle)>,
}

impl Default for AnimViewer {
    fn default() -> Self {
        Self {
            open: false,
            mode: PreviewMode::Characters,
            playing: true,
            looping: true,
            speed_ms: 90,
            last_step: None,
            tables: Vec::new(),
            chtab_key: None,
            guard: GuardKind::None,
            selected: 0,
            frame_pos: 0,
            user_flip: false,
            cache: HashMap::new(),
            bg: None,
            bg_root: None,
            selected_tile: 0,
            tick: 0,
            tile_cache: None,
        }
    }
}

impl AnimViewer {
    /// Draw the window if open. `ntsc` selects the colour mode; `root` is the
    /// editor's current data root (sprites + biome tables).
    pub fn ui(&mut self, ctx: &egui::Context, ntsc: bool, root: Option<&Path>) {
        if !self.open {
            return;
        }
        self.ensure_loaded(root);
        let mut open = self.open;
        egui::Window::new("Animations")
            .open(&mut open)
            .default_size([560.0, 380.0])
            .resizable(true)
            .show(ctx, |ui| self.window_ui(ui, ntsc));
        self.open = open;
        // Keep ticking only while still open (the × may have just closed it).
        if self.open && self.playing {
            ctx.request_repaint_after(Duration::from_millis(self.speed_ms));
        }
    }

    /// (Re)load the biome tables (on root change) and the CHTAB sprites (on
    /// root / guard change). Each commits its key only once it actually loads,
    /// so files dropped in later under the same root are still picked up.
    fn ensure_loaded(&mut self, root: Option<&Path>) {
        let Some(root) = root else { return };
        if self.bg_root.as_deref() != Some(root) {
            if let Ok(bg) = BiomeTables::load(root, Biome::Dungeon) {
                self.bg = Some(bg);
                self.bg_root = Some(root.to_path_buf());
                self.tile_cache = None;
            }
        }
        let key = (root.to_path_buf(), self.guard);
        if self.chtab_key.as_ref() != Some(&key) {
            if let Some(dir) = discovery::draz_dir_in(root).map(|d| d.join("I")) {
                self.tables = load_chtabs(&dir, self.guard);
                self.chtab_key = Some(key);
                self.cache.clear();
            }
        }
    }

    fn window_ui(&mut self, ui: &mut egui::Ui, ntsc: bool) {
        egui::TopBottomPanel::top("anim_mode").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .selectable_label(self.mode == PreviewMode::Characters, "Characters")
                    .clicked()
                {
                    self.mode = PreviewMode::Characters;
                }
                if ui
                    .selectable_label(self.mode == PreviewMode::Tiles, "Tiles")
                    .clicked()
                {
                    self.mode = PreviewMode::Tiles;
                }
                if self.mode == PreviewMode::Characters {
                    ui.separator();
                    ui.label("body:");
                    let prev = self.guard;
                    egui::ComboBox::from_id_salt("guard-body")
                        .selected_text(self.guard.label())
                        .show_ui(ui, |ui| {
                            for g in GuardKind::ALL {
                                ui.selectable_value(&mut self.guard, g, g.label());
                            }
                        });
                    // The Prince's own sequences (run / stand / …) always
                    // render the kid from CHTAB1-3; the guard art lives in the
                    // *guard* sequences (CHTAB5+). So when the body switches,
                    // jump the selection to a matching sequence — otherwise
                    // picking "Guard" on a Prince sequence shows no change.
                    if self.guard != prev {
                        self.jump_to_body_sequence();
                    }
                }
            });
        });
        match self.mode {
            PreviewMode::Characters => self.characters_ui(ui, ntsc),
            PreviewMode::Tiles => self.tiles_ui(ui, ntsc),
        }
    }

    // --- Characters --------------------------------------------------------

    fn characters_ui(&mut self, ui: &mut egui::Ui, ntsc: bool) {
        let seqs = anim::animations();
        if seqs.is_empty() {
            ui.label("No animation data parsed.");
            return;
        }
        self.selected = self.selected.min(seqs.len() - 1);

        egui::SidePanel::left("anim_list")
            .default_width(170.0)
            .show_inside(ui, |ui| {
                ui.label(format!("{} sequences", seqs.len()));
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for (i, s) in seqs.iter().enumerate() {
                        let label = format!("#{:>3}  {}  ({}f)", s.id, s.name, s.frames.len());
                        if ui.selectable_label(self.selected == i, label).clicked()
                            && self.selected != i
                        {
                            self.selected = i;
                            self.frame_pos = 0;
                            self.last_step = None;
                        }
                    }
                });
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let seq = &seqs[self.selected]; // 'static, doesn't tie up `self`
            egui::TopBottomPanel::bottom("anim_footer").show_inside(ui, |ui| {
                self.playback_controls(ui, true);
                self.char_meta(ui, seq);
            });
            egui::CentralPanel::default().show_inside(ui, |ui| {
                self.advance();
                self.char_preview(ui, ntsc, seq);
            });
        });
    }

    /// The CHTAB sprite for `frame`, resolved for the selected body — the
    /// `usealtsets` guard remap (to the chtable4 body) when a guard is picked.
    fn sprite_ref(&self, frame: u8) -> Option<SpriteRef> {
        if self.guard == GuardKind::None {
            anim::frame_sprite(frame)
        } else {
            anim::guard_frame_sprite(frame)
        }
    }

    /// Jump the selection to a sequence that suits the chosen body: the most
    /// guard-bodied one (most frames remapping to chtable4) for a guard, else
    /// `stand` for the kid.
    fn jump_to_body_sequence(&mut self) {
        let seqs = anim::animations();
        let target = if self.guard == GuardKind::None {
            seqs.iter().position(|s| s.name == "stand")
        } else {
            let guard_frames = |s: &AnimSequence| {
                s.frames
                    .iter()
                    .filter(|f| anim::guard_frame_sprite(f.frame).is_some_and(|sp| sp.chtab == 4))
                    .count()
            };
            seqs.iter()
                .enumerate()
                .max_by_key(|(_, s)| guard_frames(s))
                .filter(|(_, s)| guard_frames(s) > 0)
                .map(|(i, _)| i)
        };
        if let Some(i) = target {
            self.selected = i;
            self.frame_pos = 0;
            self.last_step = None;
        }
    }

    #[allow(clippy::cast_precision_loss)] // small POP pixel deltas
    fn char_preview(&mut self, ui: &mut egui::Ui, ntsc: bool, seq: &AnimSequence) {
        let (rect, _) = ui.allocate_exact_size(ui.available_size(), Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, Color32::from_gray(28));
        let base_y = rect.bottom() - 28.0;
        painter.hline(rect.x_range(), base_y, (1.0, Color32::from_gray(60)));

        let Some(frame) = seq.frames.get(self.frame_pos).copied() else {
            return;
        };
        // Face the figure the way it travels (mirror the left-facing art when
        // net chx is positive); the `mirror` toggle flips both together.
        let net_dx: i32 = seq.frames.iter().map(|f| f.dx).sum();
        let mirror = (net_dx > 0) ^ self.user_flip;
        let sx = if self.user_flip { -1.0 } else { 1.0 };

        let key = (frame.frame, ntsc, mirror);
        if !self.cache.contains_key(&key) {
            if let Some(tex) = self.render_sprite(ui.ctx(), frame.frame, ntsc, mirror) {
                self.cache.insert(key, tex);
            }
        }
        let Some(tex) = self.cache.get(&key) else {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                format!("frame {} — CHTAB sprite not loaded", frame.frame),
                FontId::proportional(13.0),
                Color32::GRAY,
            );
            return;
        };

        let mut off = Vec2::ZERO;
        for f in &seq.frames[..=self.frame_pos] {
            off += Vec2::new(f.dx as f32 * sx, f.dy as f32) * PREVIEW_SCALE;
        }
        let size = tex.size_vec2() * PREVIEW_SCALE;
        let feet = Pos2::new(rect.center().x + off.x, base_y + off.y);
        let img_rect = Rect::from_min_size(Pos2::new(feet.x - size.x / 2.0, feet.y - size.y), size);
        painter.image(
            tex.id(),
            img_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
    }

    fn char_meta(&mut self, ui: &mut egui::Ui, seq: &AnimSequence) {
        let frame = seq.frames.get(self.frame_pos).copied();
        let fid = frame.map_or(0, |f| f.frame);
        let sprite = self.sprite_ref(fid).map_or_else(
            || "no sprite".to_string(),
            |s| format!("CHTAB{} #{}", s.chtab, s.index),
        );
        ui.label(format!(
            "{} (#{}) • {}/{} • id {fid} • {sprite}",
            seq.name,
            seq.id,
            self.frame_pos + 1,
            seq.frames.len().max(1),
        ));
        let mut bits: Vec<String> = Vec::new();
        if let Some(f) = frame {
            bits.push(format!("dx {} dy {}", f.dx, f.dy));
            if f.turn {
                bits.push("turn".to_string());
            }
            if let Some(a) = f.action {
                bits.push(format!("act {a}"));
            }
        }
        if let Some(l) = seq.loops_to {
            bits.push(format!("loops→{}", l + 1));
        }
        if let Some(c) = &seq.chains_to {
            bits.push(format!("chains→{c}"));
        }
        ui.label(bits.join(" • "));
    }

    // --- Tiles -------------------------------------------------------------

    fn tiles_ui(&mut self, ui: &mut egui::Ui, ntsc: bool) {
        self.selected_tile = self.selected_tile.min(TILE_KINDS.len() - 1);
        egui::SidePanel::left("tile_list")
            .default_width(170.0)
            .show_inside(ui, |ui| {
                ui.label("tiles");
                ui.separator();
                for (i, (name, _)) in TILE_KINDS.iter().enumerate() {
                    if ui
                        .selectable_label(self.selected_tile == i, *name)
                        .clicked()
                        && self.selected_tile != i
                    {
                        self.selected_tile = i;
                        self.tick = 0;
                        self.last_step = None;
                    }
                }
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::TopBottomPanel::bottom("tile_footer").show_inside(ui, |ui| {
                self.playback_controls(ui, false);
                let (name, kind) = TILE_KINDS[self.selected_tile];
                ui.label(format!("{name} • tick {} • {kind:?}", self.tick));
            });
            egui::CentralPanel::default().show_inside(ui, |ui| {
                self.advance();
                self.tile_preview(ui, ntsc);
            });
        });
    }

    fn tile_preview(&mut self, ui: &mut egui::Ui, ntsc: bool) {
        let (rect, _) = ui.allocate_exact_size(ui.available_size(), Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, Color32::from_gray(20));

        if self.bg.is_none() {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "biome tables not loaded",
                FontId::proportional(13.0),
                Color32::GRAY,
            );
            return;
        }

        let key = (self.selected_tile, self.tick, ntsc);
        if self.tile_cache.as_ref().map(|(t, k, n, _)| (*t, *k, *n)) != Some(key) {
            self.tile_cache = self
                .render_tile(ui.ctx(), self.selected_tile, self.tick, ntsc)
                .map(|tex| (key.0, key.1, key.2, tex));
        }
        let Some((_, _, _, tex)) = &self.tile_cache else {
            return;
        };
        // Fit the 280×192 room frame into the preview, preserving aspect.
        let avail = rect.size();
        let scale = (avail.x / ROOM_W).min(avail.y / ROOM_H).max(0.1);
        let size = Vec2::new(ROOM_W * scale, ROOM_H * scale);
        let frame_rect = Rect::from_center_size(rect.center(), size);
        painter.image(
            tex.id(),
            frame_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
    }

    /// Render the selected tile, animated at `tick`, into a texture.
    fn render_tile(
        &self,
        ctx: &egui::Context,
        tile_idx: usize,
        tick: u32,
        ntsc: bool,
    ) -> Option<TextureHandle> {
        let bg = self.bg.as_ref()?;
        let &(_, kind) = TILE_KINDS.get(tile_idx)?;
        let tile = Tile {
            kind,
            variant: 0,
            modifier: 0,
        };
        let level = Level::preview_tile(tile, TILE_COL, TILE_ROW);
        let mode = if ntsc {
            RenderMode::NtscColor
        } else {
            RenderMode::Monochrome
        };
        let frame = scene::render_room_animated(&level, 1, bg, mode, Anim { tick, traps: true })?;
        let size = [
            usize::try_from(frame.width).ok()?,
            usize::try_from(frame.height).ok()?,
        ];
        let image = ColorImage::from_rgba_unmultiplied(size, &frame.pixels);
        Some(ctx.load_texture(
            format!("anim-tile-{tile_idx}-{tick}-{ntsc}"),
            image,
            egui::TextureOptions::NEAREST,
        ))
    }

    // --- Shared playback ---------------------------------------------------

    fn playback_controls(&mut self, ui: &mut egui::Ui, show_mirror: bool) {
        ui.horizontal(|ui| {
            let play = if self.playing { "⏸" } else { "▶" };
            if ui.button(play).clicked() {
                self.playing = !self.playing;
                self.last_step = None;
            }
            if ui.button("⏮").on_hover_text("step back").clicked() {
                self.playing = false;
                self.step_manual(-1);
            }
            if ui.button("⏭").on_hover_text("step forward").clicked() {
                self.playing = false;
                self.step_manual(1);
            }
            ui.checkbox(&mut self.looping, "loop");
            if show_mirror {
                ui.checkbox(&mut self.user_flip, "mirror")
                    .on_hover_text("Flip the figure's facing and travel direction");
            }
            ui.add(egui::Slider::new(&mut self.speed_ms, 30..=400).text("ms/frame"));
        });
    }

    /// Advance one step if the per-frame interval has elapsed while playing.
    fn advance(&mut self) {
        if !self.playing {
            return;
        }
        let now = Instant::now();
        let due = self.last_step.map_or(true, |t| {
            now.duration_since(t).as_millis() >= u128::from(self.speed_ms)
        });
        if due {
            self.last_step = Some(now);
            match self.mode {
                PreviewMode::Characters => self.tick_char(),
                PreviewMode::Tiles => self.tick = self.tick.wrapping_add(1),
            }
        }
    }

    /// One character playback step: next frame, looping at the end.
    fn tick_char(&mut self) {
        let seqs = anim::animations();
        let Some(seq) = seqs.get(self.selected) else {
            return;
        };
        let len = seq.frames.len();
        if len == 0 {
            return;
        }
        if self.frame_pos + 1 >= len {
            if self.looping {
                self.frame_pos = seq.loops_to.unwrap_or(0).min(len - 1);
            } else {
                self.playing = false;
            }
        } else {
            self.frame_pos += 1;
        }
    }

    fn step_manual(&mut self, dir: i32) {
        match self.mode {
            PreviewMode::Characters => {
                let seqs = anim::animations();
                let Some(seq) = seqs.get(self.selected) else {
                    return;
                };
                let len = seq.frames.len();
                if len == 0 {
                    return;
                }
                self.frame_pos = if dir > 0 {
                    (self.frame_pos + 1) % len
                } else {
                    (self.frame_pos + len - 1) % len
                };
            }
            PreviewMode::Tiles => {
                self.tick = if dir > 0 {
                    self.tick.wrapping_add(1)
                } else {
                    self.tick.wrapping_sub(1)
                };
            }
        }
    }

    /// Render one frame's CHTAB sprite to an egui texture, mirrored if
    /// `mirror`. `None` if the frame has no sprite or its CHTAB isn't loaded.
    fn render_sprite(
        &self,
        ctx: &egui::Context,
        frame: u8,
        ntsc: bool,
        mirror: bool,
    ) -> Option<TextureHandle> {
        let sprite = self.sprite_ref(frame)?;
        let table = self
            .tables
            .get(usize::from(sprite.chtab).checked_sub(1)?)?
            .as_ref()?;
        let img = table.images.get(sprite.index)?;
        if img.width_bytes == 0 || img.height == 0 {
            return None;
        }
        let mode = if ntsc {
            RenderMode::NtscColor
        } else {
            RenderMode::Monochrome
        };
        // Mirror in hi-res *byte* space (reverse byte order + flip each byte's
        // 7 pixels), not in RGBA — a pixel flip would swap the NTSC colours.
        let mirrored;
        let bytes = if mirror {
            mirrored = mirror_bitmap(img);
            &mirrored[..]
        } else {
            &img.bitmap[..]
        };
        let rendered = render_linear(bytes, img.width_bytes, img.height, mode)?;
        let size = [
            usize::try_from(rendered.width).ok()?,
            usize::try_from(rendered.height).ok()?,
        ];
        let image = ColorImage::from_rgba_unmultiplied(size, &rendered.pixels);
        // Name carries the full key so mirror / ntsc variants don't alias.
        Some(ctx.load_texture(
            format!("anim-frame-{frame}-{ntsc}-{mirror}"),
            image,
            egui::TextureOptions::NEAREST,
        ))
    }
}

/// Load the CHTAB sprite tables into slots 1..8. Slot 4 (the body) is the
/// selected guard variant, or empty for the Prince.
fn load_chtabs(dir: &Path, guard: GuardKind) -> Vec<Option<ImageTable>> {
    let load = |name: &str| ImageTable::from_file(dir.join(name)).ok();
    vec![
        load("IMG.CHTAB1"),
        load("IMG.CHTAB2"),
        load("IMG.CHTAB3"),
        guard.chtab4_file().and_then(load),
        load("IMG.CHTAB5"),
        load("IMG.CHTAB6.A"),
        load("IMG.CHTAB7"),
        None,
    ]
}

/// Mirror a CHTAB sprite bitmap horizontally: reverse the byte order within
/// each row and flip every byte's 7 pixel bits (keeping bit 7, the palette /
/// half-dot bit). Matches `pop_assets::sprite::composite_hires`'s `flip_h`.
fn mirror_bitmap(img: &Image) -> Vec<u8> {
    let w = usize::from(img.width_bytes);
    let h = usize::from(img.height);
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            out[y * w + (w - 1 - x)] = mirror_byte(img.bitmap[y * w + x]);
        }
    }
    out
}

/// Reverse the 7 pixel bits of a hi-res byte (bit 0 = leftmost), keeping
/// bit 7.
fn mirror_byte(b: u8) -> u8 {
    let mut out = b & 0x80;
    for i in 0..7 {
        if b & (1 << i) != 0 {
            out |= 1 << (6 - i);
        }
    }
    out
}
