//! Animation preview pane for the editor (#89).
//!
//! A floating window that plays POP's decoded character animations
//! ([`pop_assets::anim`]): pick a sequence on the left, watch it animate on
//! the right with play / pause / step / loop and a speed control. Each
//! frame's sprite is resolved `FRAMEDEF → decodeim → CHTAB` and the parsed
//! `chx`/`chy` deltas drift the figure so the locomotion shows.
//!
//! Sprites render in their native (left-facing) orientation; a faithful
//! mirrored facing needs the byte-space flip the runtime uses (the NTSC
//! half-dot, #121), so it's left for later. Guard tables (CHTAB4+) preview
//! too when present; otherwise the frame shows a "CHTAB missing" note.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use eframe::egui::{
    self, Align2, Color32, ColorImage, FontId, Pos2, Rect, Sense, TextureHandle, Vec2,
};

use pop_assets::anim::{self, AnimSequence};
use pop_assets::discovery;
use pop_assets::draz::image_table::ImageTable;
use pop_assets::hires::{render_linear, RenderMode};

/// On-screen magnification of the previewed sprite.
const PREVIEW_SCALE: f32 = 2.0;

/// Editor animation preview window state.
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
    /// Cached texture for the displayed `(frame id, ntsc)` pair.
    cached: Option<(u8, bool, TextureHandle)>,
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
            cached: None,
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
        if self.playing {
            ctx.request_repaint_after(Duration::from_millis(self.speed_ms));
        }
    }

    /// Load CHTAB1..8 from `root/DRAZ/I` once per data root.
    fn ensure_tables(&mut self, root: Option<&Path>) {
        let Some(root) = root else { return };
        if self.loaded_root.as_deref() == Some(root) {
            return;
        }
        self.loaded_root = Some(root.to_path_buf());
        self.tables.clear();
        self.cached = None;
        if let Some(dir) = discovery::draz_dir_in(root).map(|d| d.join("I")) {
            for n in 1..=8u8 {
                let table = ImageTable::from_file(dir.join(format!("IMG.CHTAB{n}"))).ok();
                self.tables.push(table);
            }
        }
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
            self.controls(ui, seq);
            self.advance(seq);
            self.preview(ui, ntsc, seq);
            self.meta(ui, seq);
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
        let (rect, _) =
            ui.allocate_exact_size(Vec2::new(ui.available_width(), 180.0), Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, Color32::from_gray(28));
        // A ground line near the bottom for a sense of the feet baseline.
        let base_y = rect.bottom() - 24.0;
        painter.hline(rect.x_range(), base_y, (1.0, Color32::from_gray(60)));

        let Some(frame) = seq.frames.get(self.frame_pos).copied() else {
            return;
        };
        if self.cached.as_ref().map(|(f, m, _)| (*f, *m)) != Some((frame.frame, ntsc)) {
            self.cached = self
                .render_sprite(ui.ctx(), frame.frame, ntsc)
                .map(|t| (frame.frame, ntsc, t));
        }

        let Some((_, _, tex)) = &self.cached else {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                format!("frame {} — CHTAB sprite not loaded", frame.frame),
                FontId::proportional(13.0),
                Color32::GRAY,
            );
            return;
        };

        // Drift the figure by the chx/chy accumulated up to this frame, so a
        // run cycle visibly travels. Bounded: frame_pos wraps at the loop.
        let mut off = Vec2::ZERO;
        for f in &seq.frames[..=self.frame_pos] {
            off += Vec2::new(f.dx as f32, f.dy as f32) * PREVIEW_SCALE;
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
        ui.separator();
        let frame = seq.frames.get(self.frame_pos).copied();
        let fid = frame.map_or(0, |f| f.frame);
        let sprite = anim::frame_sprite(fid).map_or_else(
            || " • no sprite".to_string(),
            |s| format!(" • CHTAB{} #{}", s.chtab, s.index),
        );
        ui.label(format!(
            "frame {}/{} • id {fid}{sprite}",
            self.frame_pos + 1,
            seq.frames.len().max(1),
        ));
        if let Some(f) = frame {
            let turn = if f.turn { " • turn" } else { "" };
            let act = f.action.map_or(String::new(), |a| format!(" • act {a}"));
            ui.label(format!("dx {} dy {}{turn}{act}", f.dx, f.dy));
        }
        let mut tail: Vec<String> = Vec::new();
        if let Some(l) = seq.loops_to {
            tail.push(format!("loops to frame {}", l + 1));
        }
        if let Some(c) = &seq.chains_to {
            tail.push(format!("→ chains to {c}"));
        }
        if !tail.is_empty() {
            ui.label(tail.join("   "));
        }
    }

    fn reset(&mut self) {
        self.frame_pos = 0;
        self.last_step = None;
        self.cached = None;
    }

    /// Render one frame's CHTAB sprite to an egui texture, or `None` if the
    /// frame has no sprite or its CHTAB table isn't loaded.
    fn render_sprite(&self, ctx: &egui::Context, frame: u8, ntsc: bool) -> Option<TextureHandle> {
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
        let rendered = render_linear(&img.bitmap, img.width_bytes, img.height, mode)?;
        let size = [
            usize::try_from(rendered.width).ok()?,
            usize::try_from(rendered.height).ok()?,
        ];
        let image = ColorImage::from_rgba_unmultiplied(size, &rendered.pixels);
        Some(ctx.load_texture(
            format!("anim-frame-{frame}"),
            image,
            egui::TextureOptions::NEAREST,
        ))
    }
}
