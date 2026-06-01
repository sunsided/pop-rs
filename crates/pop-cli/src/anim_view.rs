//! Animation preview pane for the editor (#89).
//!
//! A floating window that plays POP's decoded character animations
//! ([`pop_assets::anim`]): pick a sequence on the left, watch it animate on
//! the right with play / pause / step / loop and a speed control. Each
//! frame's sprite is resolved `FRAMEDEF → decodeim → CHTAB` and the parsed
//! `chx`/`chy` deltas drift the figure so the locomotion shows.
//!
//! `chx` is facing-relative and the in-game facing isn't in SEQTABLE, so the
//! figure is mirrored to face the way it travels (byte-space flip, so NTSC
//! colours stay put, unlike an RGBA flip — the #121 half-dot). A `mirror`
//! toggle flips facing and travel together when the absolute left/right
//! should swap (e.g. a stair climb up-right). Guard tables (CHTAB4+) preview
//! when present; otherwise the frame shows a "CHTAB not loaded" note.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use eframe::egui::{
    self, Align2, Color32, ColorImage, FontId, Pos2, Rect, Sense, TextureHandle, Vec2,
};

use pop_assets::anim::{self, AnimSequence};
use pop_assets::discovery;
use pop_assets::draz::image_table::{Image, ImageTable};
use pop_assets::hires::{render_linear, RenderMode};

/// On-screen magnification of the previewed sprite.
const PREVIEW_SCALE: f32 = 2.0;

/// Editor animation preview window state.
#[allow(clippy::struct_excessive_bools)] // independent UI toggles
pub struct AnimViewer {
    /// Whether the window is shown (toggled from the toolbar).
    pub open: bool,
    /// Data root the CHTAB tables were loaded from, to reload on a change.
    loaded_root: Option<PathBuf>,
    /// CHTAB sprite tables, indexed by `chtab - 1` (`None` if absent).
    tables: Vec<Option<ImageTable>>,
    /// Selected sequence (index into [`anim::animations`]).
    selected: usize,
    /// Current frame within the selected sequence.
    frame_pos: usize,
    playing: bool,
    looping: bool,
    /// Milliseconds per frame while playing.
    speed_ms: u64,
    last_step: Option<Instant>,
    /// Manually flip the figure's facing (and its travel) from the
    /// auto-pick, e.g. to send a stair climb up-right instead of up-left.
    user_flip: bool,
    /// Rendered sprite textures, keyed `(frame id, ntsc, mirror)`. Reused
    /// across frames / loops; cleared when the data root changes.
    cache: HashMap<(u8, bool, bool), TextureHandle>,
}

impl Default for AnimViewer {
    fn default() -> Self {
        Self {
            open: false,
            loaded_root: None,
            tables: Vec::new(),
            selected: 0,
            frame_pos: 0,
            playing: true,
            looping: true,
            speed_ms: 90,
            last_step: None,
            user_flip: false,
            cache: HashMap::new(),
        }
    }
}

impl AnimViewer {
    /// Draw the window if open. `ntsc` selects the colour mode; `root` is the
    /// editor's current data root (for the CHTAB sprites).
    pub fn ui(&mut self, ctx: &egui::Context, ntsc: bool, root: Option<&Path>) {
        if !self.open {
            return;
        }
        self.ensure_tables(root);
        let mut open = self.open;
        egui::Window::new("Animations")
            .open(&mut open)
            .default_size([540.0, 340.0])
            .resizable(true)
            .show(ctx, |ui| self.window_ui(ui, ntsc));
        self.open = open;
        // Keep ticking only while still open (the × may have just closed it).
        if self.open && self.playing {
            ctx.request_repaint_after(Duration::from_millis(self.speed_ms));
        }
    }

    /// Load CHTAB1..8 from `root/DRAZ/I` once per data root.
    fn ensure_tables(&mut self, root: Option<&Path>) {
        let Some(root) = root else { return };
        if self.loaded_root.as_deref() == Some(root) {
            return;
        }
        // Commit (and stop retrying) only once the DRAZ dir actually exists,
        // so sprites dropped in later, under an unchanged root, get picked up.
        let Some(dir) = discovery::draz_dir_in(root).map(|d| d.join("I")) else {
            return;
        };
        self.loaded_root = Some(root.to_path_buf());
        self.cache.clear();
        self.tables = (1..=8u8)
            .map(|n| ImageTable::from_file(dir.join(format!("IMG.CHTAB{n}"))).ok())
            .collect();
    }

    fn window_ui(&mut self, ui: &mut egui::Ui, ntsc: bool) {
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
                            self.reset();
                        }
                    }
                });
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            // `animations()` is 'static, so the borrow doesn't tie up `self`.
            let seq = &seqs[self.selected];
            // Controls + metadata hug the bottom (compact); the preview gets
            // all the remaining height, so tall cross-cell anims stay visible.
            egui::TopBottomPanel::bottom("anim_footer").show_inside(ui, |ui| {
                self.controls(ui, seq);
                self.meta(ui, seq);
            });
            egui::CentralPanel::default().show_inside(ui, |ui| {
                self.advance(seq);
                self.preview(ui, ntsc, seq);
            });
        });
    }

    fn controls(&mut self, ui: &mut egui::Ui, seq: &AnimSequence) {
        ui.horizontal(|ui| {
            let play = if self.playing { "⏸" } else { "▶" };
            if ui.button(play).clicked() {
                self.playing = !self.playing;
                self.last_step = None;
            }
            if ui.button("⏮").on_hover_text("step back").clicked() {
                self.playing = false;
                self.step_manual(-1, seq);
            }
            if ui.button("⏭").on_hover_text("step forward").clicked() {
                self.playing = false;
                self.step_manual(1, seq);
            }
            ui.checkbox(&mut self.looping, "loop");
            ui.checkbox(&mut self.user_flip, "mirror")
                .on_hover_text("Flip the figure's facing and travel direction");
            ui.add(egui::Slider::new(&mut self.speed_ms, 30..=400).text("ms/frame"));
        });
    }

    /// Advance one frame if the per-frame interval has elapsed while playing.
    fn advance(&mut self, seq: &AnimSequence) {
        if !self.playing || seq.frames.is_empty() {
            return;
        }
        let now = Instant::now();
        let due = self.last_step.map_or(true, |t| {
            now.duration_since(t).as_millis() >= u128::from(self.speed_ms)
        });
        if due {
            self.last_step = Some(now);
            self.tick(seq);
        }
    }

    /// One playback step: to the next frame, looping at the end (to
    /// `loops_to`, else frame 0) or stopping when not looping.
    fn tick(&mut self, seq: &AnimSequence) {
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

    fn step_manual(&mut self, dir: i32, seq: &AnimSequence) {
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

    #[allow(clippy::cast_precision_loss)] // small POP pixel deltas
    fn preview(&mut self, ui: &mut egui::Ui, ntsc: bool, seq: &AnimSequence) {
        let (rect, _) = ui.allocate_exact_size(ui.available_size(), Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, Color32::from_gray(28));
        // A ground line near the bottom for a sense of the feet baseline.
        let base_y = rect.bottom() - 28.0;
        painter.hline(rect.x_range(), base_y, (1.0, Color32::from_gray(60)));

        let Some(frame) = seq.frames.get(self.frame_pos).copied() else {
            return;
        };
        // `chx` is facing-relative and the in-game facing isn't in SEQTABLE,
        // so face the figure the way it actually travels: mirror POP's
        // left-facing art when the net chx is positive (rightward). The
        // `mirror` toggle flips both facing and travel together, so it never
        // moonwalks — only the absolute left/right swaps.
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

        // Drift by the chx/chy accumulated to this frame so locomotion shows;
        // the sprite is already mirrored to face this way. Bounded since
        // `frame_pos` wraps at the loop.
        let mut off = Vec2::ZERO;
        for f in &seq.frames[..=self.frame_pos] {
            off += Vec2::new(f.dx as f32 * sx, f.dy as f32) * PREVIEW_SCALE;
        }
        let size = tex.size_vec2() * PREVIEW_SCALE;
        // Feet near the baseline, drifting with the locomotion offset.
        let feet = Pos2::new(rect.center().x + off.x, base_y + off.y);
        let img_rect = Rect::from_min_size(Pos2::new(feet.x - size.x / 2.0, feet.y - size.y), size);
        painter.image(
            tex.id(),
            img_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
    }

    fn meta(&mut self, ui: &mut egui::Ui, seq: &AnimSequence) {
        let frame = seq.frames.get(self.frame_pos).copied();
        let fid = frame.map_or(0, |f| f.frame);
        let sprite = anim::frame_sprite(fid).map_or_else(
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

    fn reset(&mut self) {
        self.frame_pos = 0;
        self.last_step = None;
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
        let sprite = anim::frame_sprite(frame)?;
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
        // Name the texture by the full cache key — `mirror`/`ntsc` change the
        // pixels for the same frame id, so they must disambiguate the label.
        Some(ctx.load_texture(
            format!("anim-frame-{frame}-{ntsc}-{mirror}"),
            image,
            egui::TextureOptions::NEAREST,
        ))
    }
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
