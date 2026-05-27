//! egui port of app.py: Generate / Import / FH6 Tools / Tutorial tabs,
//! process picker, live log and geometry preview.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};

use crate::backend;
use crate::generate::{list_profiles, parse_profile};
use crate::procs::{find_game_processes, GameProc};

use egui::{Color32, RichText};

const ACCENT: Color32 = Color32::from_rgb(99, 160, 255);
const OK: Color32 = Color32::from_rgb(80, 200, 130);
const WARN: Color32 = Color32::from_rgb(240, 170, 70);
const MUTED: Color32 = Color32::from_rgb(150, 156, 168);
const BORDER: Color32 = Color32::from_rgb(46, 50, 62);
const BG_HEADER: Color32 = Color32::from_rgb(24, 27, 34);
const BG_CENTRAL: Color32 = Color32::from_rgb(18, 20, 26);
const BG_LOG: Color32 = Color32::from_rgb(13, 14, 18);

fn apply_theme(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();
    v.panel_fill = Color32::from_rgb(18, 20, 26);
    v.window_fill = Color32::from_rgb(24, 27, 34);
    v.extreme_bg_color = Color32::from_rgb(13, 14, 18);
    v.faint_bg_color = Color32::from_rgb(30, 33, 42);
    v.override_text_color = Some(Color32::from_rgb(222, 226, 232));
    v.selection.bg_fill = Color32::from_rgb(46, 78, 132);
    v.selection.stroke = egui::Stroke::new(1.0, ACCENT);
    v.hyperlink_color = ACCENT;
    let r = egui::Rounding::same(7.0);
    v.widgets.noninteractive.rounding = r;
    v.widgets.inactive.rounding = r;
    v.widgets.hovered.rounding = r;
    v.widgets.active.rounding = r;
    v.widgets.open.rounding = r;
    v.widgets.inactive.bg_fill = Color32::from_rgb(38, 42, 53);
    v.widgets.inactive.weak_bg_fill = Color32::from_rgb(38, 42, 53);
    v.widgets.hovered.bg_fill = Color32::from_rgb(54, 60, 76);
    v.widgets.hovered.weak_bg_fill = Color32::from_rgb(54, 60, 76);
    v.window_rounding = egui::Rounding::same(10.0);
    ctx.set_visuals(v);

    let mut s = (*ctx.style()).clone();
    s.spacing.item_spacing = egui::vec2(10.0, 8.0);
    s.spacing.button_padding = egui::vec2(12.0, 7.0);
    s.spacing.window_margin = egui::Margin::same(14.0);
    s.spacing.interact_size.y = 30.0;
    ctx.set_style(s);
}

fn primary_button(text: &str) -> egui::Button<'static> {
    egui::Button::new(RichText::new(text.to_owned()).strong().color(Color32::WHITE)).fill(ACCENT)
}

fn status_pill(ui: &mut egui::Ui, status: &str, busy: bool) {
    let (txt, col) = if busy {
        (status.to_string(), WARN)
    } else if status == "Done" {
        ("Done".to_string(), OK)
    } else {
        ("Ready".to_string(), OK)
    };
    egui::Frame::none()
        .fill(Color32::from_rgb(32, 36, 46))
        .rounding(egui::Rounding::same(999.0))
        .inner_margin(egui::Margin::symmetric(10.0, 4.0))
        .show(ui, |ui| {
            ui.label(RichText::new(txt).color(col).strong());
        });
}

enum Msg {
    Log(String),
    Status(String),
    Preview(egui::ColorImage),
    GenerateDone,
}

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Generate,
    Import,
}

pub struct App {
    tab: Tab,
    processes: Vec<GameProc>,
    proc_idx: usize,
    profiles: Vec<(String, PathBuf)>,
    profile_idx: usize,
    last_profile_idx: usize,
    image: Option<PathBuf>,
    jsons: Vec<PathBuf>,
    layer_count: String,
    // Runtime-editable generation parameters (seeded from the chosen preset).
    e_res: String,
    e_shapes: String,
    e_random: String,
    e_mutated: String,
    e_posterize: String,
    e_save_every: String,
    e_opacity: String,
    bg_transparent: bool,
    edge_mask: bool,
    log: Vec<String>,
    status: String,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    tex: Option<egui::TextureHandle>,
    busy: bool,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        apply_theme(&cc.egui_ctx);
        let (tx, rx) = channel();
        let profiles = list_profiles(&settings_dir());
        let processes = find_game_processes();
        let profile_idx = profiles.len().min(3).saturating_sub(1);
        let mut app = Self {
            tab: Tab::Generate,
            processes,
            proc_idx: 0,
            profile_idx,
            last_profile_idx: profile_idx,
            profiles,
            image: None,
            jsons: Vec::new(),
            layer_count: String::new(),
            e_res: String::new(),
            e_shapes: String::new(),
            e_random: String::new(),
            e_mutated: String::new(),
            e_posterize: String::new(),
            e_save_every: String::new(),
            e_opacity: String::new(),
            bg_transparent: false,
            edge_mask: false,
            log: Vec::new(),
            status: "Ready".into(),
            tx,
            rx,
            tex: None,
            busy: false,
        };
        app.load_preset();
        app
    }

    /// Seed the runtime editor from the currently selected preset .ini.
    fn load_preset(&mut self) {
        if let Some((_, path)) = self.profiles.get(self.profile_idx) {
            let p = parse_profile(path);
            self.e_res = p.max_resolution.to_string();
            self.e_shapes = p.stop_at.to_string();
            self.e_random = p.random_samples.to_string();
            self.e_mutated = p.mutated_samples.to_string();
            self.e_posterize = p.posterize_levels.to_string();
            self.e_save_every = p.save_every.to_string();
            if self.e_opacity.trim().is_empty() {
                self.e_opacity = "255".to_string();
            }
        }
    }

    fn log_line(&mut self, s: impl Into<String>) {
        self.log.push(s.into());
        if self.log.len() > 4000 {
            self.log.drain(0..1000);
        }
    }

    fn selected_pid(&self) -> Option<u32> {
        self.processes.get(self.proc_idx).map(|p| p.pid)
    }
    fn selected_game(&self) -> String {
        self.processes
            .get(self.proc_idx)
            .map(|p| p.profile_key.clone())
            .unwrap_or_else(|| "fh6".into())
    }

    fn spawn<F>(&mut self, f: F)
    where
        F: FnOnce(Sender<Msg>) + Send + 'static,
    {
        if self.busy {
            self.log_line("Busy: another task is running.");
            return;
        }
        self.busy = true;
        self.status = "Running".into();
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            f(tx.clone());
            let _ = tx.send(Msg::Status("Ready".into()));
            let _ = tx.send(Msg::GenerateDone);
        });
    }

    fn start_generate(&mut self) {
        let Some(img) = self.image.clone() else {
            self.log_line("No image selected.");
            return;
        };
        let exe = std::env::current_exe().unwrap_or_default();
        let params = [
            "--max-resolution".to_string(),
            self.e_res.trim().to_string(),
            "--stop-at".to_string(),
            self.e_shapes.trim().to_string(),
            "--random-samples".to_string(),
            self.e_random.trim().to_string(),
            "--mutated-samples".to_string(),
            self.e_mutated.trim().to_string(),
            "--posterize-levels".to_string(),
            self.e_posterize.trim().to_string(),
            "--save-every".to_string(),
            self.e_save_every.trim().to_string(),
            "--opacity".to_string(),
            self.e_opacity.trim().to_string(),
            "--background".to_string(),
            if self.bg_transparent { "transparent" } else { "white" }.to_string(),
        ];
        self.spawn(move |tx| {
            let _ = tx.send(Msg::Log(format!("Generating: {}", img.display())));
            if let Some(ci) = render_source(&img) {
                let _ = tx.send(Msg::Preview(ci));
            }
            let before = generated_jsons(&img);
            let mut child = match std::process::Command::new(&exe)
                .arg("generate")
                .arg(&img)
                .args(&params)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
            {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(Msg::Log(format!("Failed to start generator: {e}")));
                    return;
                }
            };
            use std::io::{BufRead, BufReader};
            let stdout = child.stdout.take();
            let stderr = child.stderr.take();
            let tx2 = tx.clone();
            let reader = std::thread::spawn(move || {
                if let Some(o) = stdout {
                    for line in BufReader::new(o).lines().map_while(Result::ok) {
                        let _ = tx2.send(Msg::Log(line));
                    }
                }
            });
            let txe = tx.clone();
            let ereader = std::thread::spawn(move || {
                if let Some(e) = stderr {
                    for line in BufReader::new(e).lines().map_while(Result::ok) {
                        let _ = txe.send(Msg::Log(format!("[stderr] {line}")));
                    }
                }
            });
            let img2 = img.clone();
            let tx3 = tx.clone();
            let mut last = PathBuf::new();
            while child.try_wait().ok().flatten().is_none() {
                std::thread::sleep(std::time::Duration::from_millis(800));
                if let Some(j) = generated_jsons(&img2).first().cloned() {
                    if j != last {
                        last = j.clone();
                        if let Some(ci) = render_geometry(&j) {
                            let _ = tx3.send(Msg::Preview(ci));
                        }
                    }
                }
            }
            let _ = reader.join();
            let _ = ereader.join();
            let status = child.wait();
            let mut produced = false;
            for j in generated_jsons(&img) {
                if !before.contains(&j) {
                    produced = true;
                    let _ = tx.send(Msg::Log(format!("Generated: {}", j.display())));
                    if let Some(ci) = render_geometry(&j) {
                        let _ = tx.send(Msg::Preview(ci));
                    } else {
                        let _ = tx.send(Msg::Log(
                            "Preview render failed (JSON unreadable or too large).".into(),
                        ));
                    }
                }
            }
            match status {
                Ok(s) if !s.success() => {
                    let _ = tx.send(Msg::Log(format!(
                        "Generator exited with error ({s}). See [stderr] above."
                    )));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Log(format!("Could not wait for generator: {e}")));
                }
                _ => {}
            }
            if !produced {
                let _ = tx.send(Msg::Log(
                    "No JSON was produced — generation failed (check [stderr]/log above).".into(),
                ));
            }
        });
    }

    fn start_import(&mut self) {
        if self.jsons.is_empty() {
            self.log_line("No JSON files selected.");
            return;
        }
        let jsons = self.jsons.clone();
        let game = self.selected_game();
        let pid = self.selected_pid();
        let layer_count: Option<u32> = self.layer_count.trim().parse().ok();
        let edge_mask = self.edge_mask;
        self.spawn(move |tx| {
            let mut log = |s: String| {
                let _ = tx.send(Msg::Log(s));
            };
            // Addresses are always auto-resolved for FH6 (no manual entry).
            let (mut ca, mut ta): (Option<u64>, Option<u64>) = (None, None);
            // Effective count = whatever the locator actually validated, which
            // can differ by a few from what the user typed (FH6 counts ungroup
            // differently). The importer must write that many slots.
            let mut eff_count = layer_count;
            if ca.is_none() && ta.is_none() && game == "fh6" {
                if let Some(s) = backend::saved_session() {
                    let pid_ok = pid.map(|p| p == s.pid).unwrap_or(true);
                    let near = layer_count
                        .map(|lc| (lc as i64 - s.layer_count as i64).abs() <= 3)
                        .unwrap_or(false);
                    if pid_ok && near {
                        ca = Some(s.count_address);
                        ta = Some(s.table_address);
                        eff_count = Some(s.layer_count);
                        log(format!(
                            "Using saved FH6 session: count=0x{:x} table=0x{:x} layers={}",
                            s.count_address, s.table_address, s.layer_count
                        ));
                    } else if pid_ok {
                        log("Saved session layer count is far from entered; re-locating.".into());
                    }
                }
                if ca.is_none() {
                    if let Some(lc) = layer_count {
                        log("Auto-locating FH6 session before import...".into());
                        match backend::run_auto_locate(Some(&game), pid, lc, &mut log) {
                            Ok(s) => {
                                ca = Some(s.count_address);
                                ta = Some(s.table_address);
                                eff_count = Some(s.layer_count);
                            }
                            Err(e) => {
                                log(format!("Auto-locate failed: {e}"));
                                return;
                            }
                        }
                    }
                }
            }
            for j in jsons {
                log(format!("Importing {}", j.display()));
                match backend::run_import(&j, Some(&game), pid, ca, ta, eff_count, edge_mask, &mut log) {
                    Ok(()) => log("Import OK".into()),
                    Err(e) => {
                        log(format!("Import failed: {e}"));
                        return;
                    }
                }
            }
        });
    }

    fn drain(&mut self, ctx: &egui::Context) {
        while let Ok(m) = self.rx.try_recv() {
            match m {
                Msg::Log(s) => self.log_line(s),
                Msg::Status(s) => self.status = s,
                Msg::GenerateDone => self.busy = false,
                Msg::Preview(ci) => {
                    self.tex =
                        Some(ctx.load_texture("preview", ci, egui::TextureOptions::LINEAR));
                }
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain(ctx);
        ctx.request_repaint_after(std::time::Duration::from_millis(200));

        egui::TopBottomPanel::top("hdr")
            .frame(egui::Frame::none()
                .fill(BG_HEADER)
                .inner_margin(egui::Margin::symmetric(16.0, 10.0))
                .stroke(egui::Stroke::new(1.0, BORDER)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("forza").size(22.0).strong().color(ACCENT));
                    ui.label(RichText::new("painter").size(22.0).strong());
                    ui.label(RichText::new("FH6").size(12.0).color(MUTED));
                });
                ui.label(
                    RichText::new("GPU geometry generator + live vinyl importer")
                    .size(12.0)
                    .color(MUTED),
                );
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label("Game process");
                    let cur = self
                        .processes
                        .get(self.proc_idx)
                        .map(|p| p.label())
                        .unwrap_or_else(|| "No game".into());
                    egui::ComboBox::from_id_source("proc")
                        .selected_text(cur)
                        .show_ui(ui, |ui| {
                            for (i, p) in self.processes.iter().enumerate() {
                                ui.selectable_value(&mut self.proc_idx, i, p.label());
                            }
                        });
                    if ui.button("Refresh").clicked() {
                        self.processes = find_game_processes();
                        self.proc_idx = 0;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        status_pill(ui, &self.status, self.busy);
                    });
                });
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    let mut tab_btn = |ui: &mut egui::Ui, t: Tab, label: &str| {
                        let sel = self.tab == t;
                        let txt = if sel {
                            RichText::new(label).strong().color(Color32::WHITE)
                        } else {
                            RichText::new(label).color(MUTED)
                        };
                        let b = egui::Button::new(txt)
                            .fill(if sel { ACCENT } else { Color32::TRANSPARENT });
                        if ui.add(b).clicked() {
                            self.tab = t;
                        }
                    };
                    tab_btn(ui, Tab::Generate, "Generate");
                    tab_btn(ui, Tab::Import, "Import");
                });
            });

        egui::TopBottomPanel::bottom("log")
            .resizable(true)
            .default_height(170.0)
            .min_height(90.0)
            .frame(egui::Frame::none()
                .fill(BG_LOG)
                .inner_margin(egui::Margin::symmetric(16.0, 10.0))
                .stroke(egui::Stroke::new(1.0, BORDER)))
            .show(ctx, |ui| {
                ui.label(RichText::new("LOGS").color(MUTED).strong().size(11.0));
                ui.add_space(4.0);
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for l in &self.log {
                            ui.add(
                                egui::Label::new(RichText::new(l).monospace().size(12.0))
                                    .wrap(),
                            );
                        }
                    });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::none()
                .fill(BG_CENTRAL)
                .inner_margin(egui::Margin::symmetric(18.0, 14.0)))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| match self.tab {
                        Tab::Generate => self.ui_generate(ui),
                        Tab::Import => self.ui_import(ui),
                    });
            });
    }
}

impl App {
    fn ui_generate(&mut self, ui: &mut egui::Ui) {
        // Selecting a preset reseeds the runtime editor.
        if self.profile_idx != self.last_profile_idx {
            self.last_profile_idx = self.profile_idx;
            self.load_preset();
        }

        ui.horizontal(|ui| {
            if ui.button("Choose image").clicked() {
                if let Some(f) = rfd::FileDialog::new()
                    .add_filter("Images", &["png", "jpg", "jpeg", "bmp"])
                    .pick_file()
                {
                    self.image = Some(f);
                }
            }
            ui.label("Preset");
            let label = self
                .profiles
                .get(self.profile_idx)
                .map(|p| p.0.clone())
                .unwrap_or_default();
            egui::ComboBox::from_id_source("profile")
                .width(240.0)
                .selected_text(label)
                .show_ui(ui, |ui| {
                    for (i, p) in self.profiles.iter().enumerate() {
                        ui.selectable_value(&mut self.profile_idx, i, &p.0);
                    }
                });
            if ui.button("Reset").clicked() {
                self.load_preset();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(
                        !self.busy && self.image.is_some(),
                        primary_button("Start generating"),
                    )
                    .clicked()
                {
                    self.start_generate();
                }
            });
        });
        if let Some(img) = &self.image {
            ui.label(
                RichText::new(format!("Image: {}", img.file_name().unwrap_or_default().to_string_lossy()))
                    .color(MUTED)
                    .size(12.0),
            );
        }

        ui.add_space(10.0);
        ui.columns(2, |c| {
            // Left: runtime parameter editor.
            c[0].label(
                RichText::new("GENERATION SETTINGS")
                    .color(MUTED)
                    .strong()
                    .size(11.0),
            );
            c[0].add_space(4.0);
            egui::Frame::group(c[0].style()).show(&mut c[0], |ui| {
                let row = |ui: &mut egui::Ui, label: &str, v: &mut String, hint: &str| {
                    ui.horizontal(|ui| {
                        ui.add_sized([150.0, 20.0], egui::Label::new(label));
                        ui.add(egui::TextEdit::singleline(v).desired_width(90.0));
                        ui.label(RichText::new(hint).color(MUTED).size(11.0));
                    });
                };
                row(ui, "Resolution", &mut self.e_res, "px");
                row(ui, "Shapes (stopAt)", &mut self.e_shapes, "≈ template");
                row(ui, "Random samples", &mut self.e_random, "↑ quality");
                row(ui, "Mutated samples", &mut self.e_mutated, "↑ quality");
                row(ui, "Posterize levels", &mut self.e_posterize, "2-256");
                row(ui, "Save every", &mut self.e_save_every, "shapes");
                row(ui, "Opacity", &mut self.e_opacity, "255 = solid");
                ui.horizontal(|ui| {
                    ui.add_sized([150.0, 20.0], egui::Label::new("Background"));
                    egui::ComboBox::from_id_source("bg_mode")
                        .selected_text(if self.bg_transparent {
                            "Transparent (subject only)"
                        } else {
                            "White (full image)"
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.bg_transparent,
                                false,
                                "White (full image)",
                            );
                            ui.selectable_value(
                                &mut self.bg_transparent,
                                true,
                                "Transparent (subject only)",
                            );
                        });
                });
            });
            // Right: preview.
            c[1].label(RichText::new("PREVIEW").color(MUTED).strong().size(11.0));
            c[1].add_space(4.0);
            egui::Frame::group(c[1].style())
                .fill(BG_LOG)
                .show(&mut c[1], |ui| {
                    ui.set_min_height(360.0);
                    ui.centered_and_justified(|ui| {
                        if let Some(t) = &self.tex {
                            ui.add(egui::Image::new(t).max_width(460.0));
                        } else {
                            ui.label(
                                RichText::new("Preview appears here while generating.")
                                .color(MUTED),
                            );
                        }
                    });
                });
        });
    }

    fn ui_import(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("Add JSON").clicked() {
                if let Some(fs) = rfd::FileDialog::new()
                    .add_filter("Geometry JSON", &["json"])
                    .pick_files()
                {
                    for f in fs {
                        if !self.jsons.contains(&f) {
                            self.jsons.push(f);
                        }
                    }
                }
            }
            if ui.button("Use generated JSON").clicked() {
                if let Some(img) = self.image.clone() {
                    if let Some(j) = generated_jsons(&img).first().cloned() {
                        if !self.jsons.contains(&j) {
                            self.jsons.push(j);
                        }
                    }
                }
            }
        });
        ui.add_space(6.0);
        ui.label(
            RichText::new("GEOMETRY JSON")
                .color(MUTED)
                .strong()
                .size(11.0),
        );
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_min_height(120.0);
            if self.jsons.is_empty() {
                ui.label(RichText::new("No JSON added.").color(MUTED));
            }
            for j in &self.jsons.clone() {
                ui.horizontal(|ui| {
                    if ui.small_button("x").clicked() {
                        self.jsons.retain(|p| p != j);
                    }
                    ui.label(j.file_name().unwrap_or_default().to_string_lossy());
                });
            }
        });

        ui.add_space(10.0);
        ui.horizontal(|ui| {
            ui.add_sized([180.0, 20.0], egui::Label::new("Template layer count"));
            ui.add(egui::TextEdit::singleline(&mut self.layer_count).desired_width(120.0));
        });
        ui.checkbox(
            &mut self.edge_mask,
            "Edge clip mask (FH5 only — shows as black frame on FH6)",
        );
        ui.add_space(4.0);
        ui.label(
            RichText::new("Enter the EXACT ungrouped layer count. The live address is auto-located. Run as administrator if OpenProcess fails.")
            .color(MUTED)
            .size(12.0),
        );
        ui.add_space(10.0);
        if ui
            .add_enabled(
                !self.busy && !self.jsons.is_empty(),
                primary_button("Import into game"),
            )
            .clicked()
        {
            self.start_import();
        }
    }
}

fn settings_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_default().join("settings")
}


/// app.py generated_jsons: <dir>/<stem>/**.json + <dir>/<stem>*.json, newest first.
fn generated_jsons(image: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let parent = image.parent().unwrap_or_else(|| Path::new("."));
    let stem = image.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let folder = parent.join(&stem);
    if folder.is_dir() {
        collect_json(&folder, &mut out);
    }
    if let Ok(rd) = std::fs::read_dir(parent) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().map(|x| x == "json").unwrap_or(false)
                && p.file_name()
                    .map(|n| n.to_string_lossy().starts_with(&stem))
                    .unwrap_or(false)
            {
                out.push(p);
            }
        }
    }
    out.sort_by_key(|p| {
        std::cmp::Reverse(
            std::fs::metadata(p)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0),
        )
    });
    out.dedup();
    out
}

fn collect_json(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                collect_json(&p, out);
            } else if p.extension().map(|x| x == "json").unwrap_or(false) {
                out.push(p);
            }
        }
    }
}

fn render_source(path: &Path) -> Option<egui::ColorImage> {
    // Composite onto white so the thumbnail matches what generation uses
    // (transparent PNGs are flattened, not shown see-through).
    let img = image::open(path).ok()?.to_rgba8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut buf = Vec::with_capacity(w * h * 4);
    for p in img.pixels() {
        let a = p[3] as f32 / 255.0;
        let mix = |c: u8| (c as f32 * a + 255.0 * (1.0 - a)).round() as u8;
        buf.extend_from_slice(&[mix(p[0]), mix(p[1]), mix(p[2]), 255]);
    }
    Some(egui::ColorImage::from_rgba_unmultiplied([w, h], &buf))
}

/// app.py render_geometry_json: background rect + filled rotated ellipses.
fn render_geometry(path: &Path) -> Option<egui::ColorImage> {
    let text = std::fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let shapes = v.get("shapes")?.as_array()?;
    let head = shapes.first()?;
    let hd = head.get("data")?.as_array()?;
    let w = hd[2].as_f64()? as usize;
    let h = hd[3].as_f64()? as usize;
    if w == 0 || h == 0 || w * h > 64_000_000 {
        return None;
    }
    let hc = head.get("color")?.as_array()?;
    let bg = [
        hc[0].as_i64().unwrap_or(0) as u8,
        hc[1].as_i64().unwrap_or(0) as u8,
        hc[2].as_i64().unwrap_or(0) as u8,
        255,
    ];
    let mut buf = vec![0u8; w * h * 4];
    for px in buf.chunks_exact_mut(4) {
        px.copy_from_slice(&bg);
    }
    for sh in &shapes[1..] {
        if sh.get("type").and_then(|t| t.as_i64()) != Some(16) {
            continue;
        }
        let d = sh.get("data").and_then(|x| x.as_array())?;
        let c = sh.get("color").and_then(|x| x.as_array())?;
        let (cx, cy) = (d[0].as_f64()? as f32, d[1].as_f64()? as f32);
        let (rw, rh) = (d[2].as_f64()?.max(1.0) as f32, d[3].as_f64()?.max(1.0) as f32);
        let ang = (-90.0 + d[4].as_f64()? as f32).to_radians();
        let col = [
            c[0].as_i64().unwrap_or(0) as u8,
            c[1].as_i64().unwrap_or(0) as u8,
            c[2].as_i64().unwrap_or(0) as u8,
        ];
        let (ax, ay) = (rh, rw); // cv2 axes=(h,w)
        let rad = ax.max(ay);
        let x0 = ((cx - rad).floor().max(0.0)) as usize;
        let y0 = ((cy - rad).floor().max(0.0)) as usize;
        let x1 = ((cx + rad).ceil().min(w as f32 - 1.0)) as usize;
        let y1 = ((cy + rad).ceil().min(h as f32 - 1.0)) as usize;
        let (cs, sn) = (ang.cos(), ang.sin());
        for py in y0..=y1 {
            for px in x0..=x1 {
                let dx = px as f32 - cx;
                let dy = py as f32 - cy;
                let u = dx * cs + dy * sn;
                let vv = -dx * sn + dy * cs;
                if (u * u) / (ax * ax) + (vv * vv) / (ay * ay) <= 1.0 {
                    let i = (py * w + px) * 4;
                    buf[i] = col[0];
                    buf[i + 1] = col[1];
                    buf[i + 2] = col[2];
                    buf[i + 3] = 255;
                }
            }
        }
    }
    Some(egui::ColorImage::from_rgba_unmultiplied([w, h], &buf))
}

