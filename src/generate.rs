// GPU-accelerated ellipse geometrizer.
//
// Drop-in replacement for the bundled `forza-painter-geometrize.exe`:
//   * argv[1] = image path (the Python app passes only the image path)
//   * one line on stdin = 1-based profile index into sorted settings/*.ini
//     (same ordering as app.py `sorted(SETTINGS_DIR.glob("*.ini"))`)
//   * emits {"shapes":[bg, ellipse...]} JSON that main.py / app.py consume.
//
// The hill-climb is the primitive/geometrize energy, but every candidate is
// scored on the GPU in parallel (see shaders.wgsl).

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use wgpu::util::DeviceExt;

const WG: u32 = 64; // must match shaders.wgsl
const MAX_WORKGROUPS: usize = 65_535;
const CLIMB_WIDTH: usize = 256; // mutations evaluated per (GPU-resident) climb round
// Uniform dynamic-offset slots; stride must be a multiple of the device's
// min_uniform_buffer_offset_alignment (>=256 everywhere).
const SLOT: u64 = 256;
const S_EVAL_RAND: u64 = 0;
const S_REDUCE_INIT: u64 = 1;
const S_EVAL_CLIMB: u64 = 2;
const S_REDUCE_FOLD: u64 = 3;
const S_MISC: u64 = 4; // commit / advance / end_shape (flags unused)

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    dims: [u32; 4],  // w, h, n_candidates, flag
    cfg: [f32; 4],   // alpha, posterize_levels, max_r, PI
    extra: [u32; 4], // shape_mode (0=ellipse 1=rect 2=mixed), _, _, _
}

#[derive(Clone, Copy)]
struct Ellipse {
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    rot: f32, // radians
}

pub struct Profile {
    pub max_resolution: u32,
    pub random_samples: usize,
    pub mutated_samples: usize,
    pub posterize_levels: u32,
    pub stop_at: usize,
    pub save_every: usize,
    pub save_at: Vec<usize>,
    /// Transparent mode: only the opaque subject gets shapes, no background.
    pub transparent_bg: bool,
    /// Per-shape opacity 1..=255. 255 = solid sticker (no see-through, no
    /// car-colour matching); 128 = classic blended geometrize.
    pub shape_alpha: u8,
    /// 0 = ellipses, 1 = rectangles, 2 = mixed.
    pub shape_mode: u8,
}

/// Run generation with explicit (runtime-edited) parameters.
pub fn generate_with_profile(image_path: &Path, profile: &Profile) {
    pollster::block_on(run(image_path, profile));
}

/// CLI/bat entry point: choose a profile by the 1-based index read from stdin
/// (same ordering as the Python `sorted(settings/*.ini)`), then generate.
pub fn cli_generate(image_path: PathBuf) {
    let settings_dir = std::env::current_dir().unwrap_or_default().join("settings");
    let profiles = list_profiles(&settings_dir);
    if profiles.is_empty() {
        eprintln!("No settings/*.ini profiles found next to the working directory.");
        std::process::exit(1);
    }
    println!("Profiles:");
    for (i, (name, _)) in profiles.iter().enumerate() {
        println!("  {} {}", i + 1, name);
    }
    let _ = std::io::stdout().flush();

    let mut line = String::new();
    let _ = std::io::stdin().read_to_string(&mut line);
    let index: usize = line
        .lines()
        .next()
        .and_then(|s| s.trim().parse().ok())
        .filter(|&n: &usize| n >= 1 && n <= profiles.len())
        .unwrap_or_else(|| {
            profiles
                .iter()
                .position(|(n, _)| n == "_default.ini")
                .map(|p| p + 1)
                .unwrap_or(1)
        });
    let (profile_name, profile_path) = &profiles[index - 1];
    println!("Using profile {}: {}", index, profile_name);
    let profile = parse_profile(profile_path);
    println!(
        "stopAt={} randomSamples={} mutatedSamples={} maxResolution={} posterizeLevels={}",
        profile.stop_at,
        profile.random_samples,
        profile.mutated_samples,
        profile.max_resolution,
        profile.posterize_levels
    );
    println!("Note: GPU build ignores maxThreads/preview keys (GPU does the scoring).");
    let _ = std::io::stdout().flush();

    pollster::block_on(run(&image_path, &profile));
}

pub fn list_profiles(dir: &Path) -> Vec<(String, PathBuf)> {
    let mut v: Vec<(String, PathBuf)> = match fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "ini").unwrap_or(false))
            .map(|p| (p.file_name().unwrap().to_string_lossy().into_owned(), p))
            .collect(),
        Err(_) => Vec::new(),
    };
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

pub fn parse_profile(path: &Path) -> Profile {
    let text = fs::read_to_string(path).unwrap_or_default();
    let get = |key: &str| -> Option<String> {
        for raw in text.lines() {
            let l = raw.trim();
            if l.is_empty() || l.starts_with('#') {
                continue;
            }
            if let Some((k, val)) = l.split_once('=') {
                if k.trim().eq_ignore_ascii_case(key) {
                    return Some(val.trim().to_string());
                }
            }
        }
        None
    };
    let num = |s: Option<String>, d: usize| -> usize {
        s.and_then(|x| x.parse().ok()).unwrap_or(d)
    };
    let save_at = get("saveAt")
        .map(|s| {
            s.split(',')
                .filter_map(|x| x.trim().parse::<usize>().ok())
                .collect()
        })
        .unwrap_or_default();
    Profile {
        max_resolution: num(get("maxResolution"), 1000) as u32,
        random_samples: num(get("randomSamples"), 1500).max(1),
        mutated_samples: num(get("mutatedSamples"), 600),
        posterize_levels: num(get("posterizeLevels"), 256).clamp(2, 256) as u32,
        stop_at: num(get("stopAt"), 3000).max(1),
        save_every: num(get("saveEvery"), 50).max(1),
        save_at,
        transparent_bg: false,
        shape_alpha: 255,
        shape_mode: 0,
    }
}

async fn run(image_path: &Path, profile: &Profile) {
    let rgba = match image::open(image_path) {
        Ok(i) => i.to_rgba8(),
        Err(e) => {
            eprintln!("Failed to open image {}: {}", image_path.display(), e);
            std::process::exit(1);
        }
    };
    let (ow, oh) = (rgba.width(), rgba.height());

    // ---- GPU setup -------------------------------------------------------
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("No compatible GPU adapter found");
    println!("GPU: {}", adapter.get_info().name);
    let limits = adapter.limits();
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("geometrize"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .expect("request_device failed");

    // ---- resize so the image fits maxResolution and GPU buffer limits ----
    let cap = (limits.max_storage_buffer_binding_size as usize / 16)
        .min(limits.max_buffer_size as usize / 16)
        .min(MAX_WORKGROUPS * WG as usize); // commit is a 1-D dispatch over pixels
    let mut scale = (profile.max_resolution as f32 / ow.max(oh) as f32).min(1.0);
    loop {
        let w = ((ow as f32 * scale).round() as u32).max(1);
        let h = ((oh as f32 * scale).round() as u32).max(1);
        if (w as usize) * (h as usize) <= cap || scale < 0.02 {
            break;
        }
        scale *= 0.85;
    }
    let w = ((ow as f32 * scale).round() as u32).max(1);
    let h = ((oh as f32 * scale).round() as u32).max(1);
    let resized = image::imageops::resize(&rgba, w, h, image::imageops::FilterType::Triangle);
    println!("Working resolution: {}x{} (source {}x{})", w, h, ow, oh);
    let _ = std::io::stdout().flush();

    // Alpha handling. White mode: flatten onto white, every pixel scored.
    // Transparent mode: keep an alpha mask in target.w (1 = score this pixel,
    // 0 = skip), so the geometrizer only places shapes on the opaque subject
    // and transparent areas stay bare (no background, no stray ellipses).
    let transparent = profile.transparent_bg;
    let npix = (w as usize) * (h as usize);
    let mut target = vec![0f32; npix * 4];
    let mut avg = [0f64; 3];
    let mut counted = 0u64;
    for (i, px) in resized.pixels().enumerate() {
        let a = px[3] as f32 / 255.0;
        // Composite over white for clean subject-edge colour either way.
        let r = (px[0] as f32 * a + 255.0 * (1.0 - a)) / 255.0;
        let g = (px[1] as f32 * a + 255.0 * (1.0 - a)) / 255.0;
        let b = (px[2] as f32 * a + 255.0 * (1.0 - a)) / 255.0;
        target[i * 4] = r;
        target[i * 4 + 1] = g;
        target[i * 4 + 2] = b;
        let scored = !transparent || a >= 0.5;
        target[i * 4 + 3] = if scored { 1.0 } else { 0.0 };
        if scored {
            avg[0] += r as f64;
            avg[1] += g as f64;
            avg[2] += b as f64;
            counted += 1;
        }
    }
    let denom = counted.max(1) as f64;
    let bg = [
        (avg[0] / denom) as f32,
        (avg[1] / denom) as f32,
        (avg[2] / denom) as f32,
    ];
    if transparent {
        let masked = npix as u64 - counted;
        println!(
            "Transparent mode ON: {} / {} px are subject (scored), {} masked as background (bg alpha 0).",
            counted, npix, masked
        );
        if masked == 0 {
            println!(
                "WARNING: 0 px masked -> the source image has NO transparent pixels (alpha is fully opaque). \
Transparent mode can only skip real alpha; supply a cut-out PNG with transparency."
            );
        }
        let _ = std::io::stdout().flush();
    } else {
        println!("Transparent mode OFF (White): whole image is filled.");
        let _ = std::io::stdout().flush();
    }
    let mut canvas = vec![0f32; npix * 4];
    for i in 0..npix {
        canvas[i * 4] = bg[0];
        canvas[i * 4 + 1] = bg[1];
        canvas[i * 4 + 2] = bg[2];
        canvas[i * 4 + 3] = 1.0;
    }

    // ---- buffers ---------------------------------------------------------
    let max_cand = profile.random_samples.max(CLIMB_WIDTH).min(MAX_WORKGROUPS);
    let stop_at = profile.stop_at;
    let target_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("target"),
        contents: bytemuck::cast_slice(&target),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let canvas_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("canvas"),
        contents: bytemuck::cast_slice(&canvas),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let results_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("results"),
        size: (max_cand * 3 * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let best_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bestbuf"),
        size: 3 * 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let state_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("state"),
        contents: bytemuck::cast_slice(&[0x853C49E6u32, 0, 0, 0]), // [seed, round, shape, _]
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let shapes_bytes = (stop_at * 3 * 16) as u64;
    let shapes_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shapes_out"),
        size: shapes_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: shapes_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: SLOT * 5,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Indirect commit: commit_args writes cargs (3xu32) + cbox (4xu32);
    // cargs is copied into the dedicated INDIRECT buffer cdisp so the
    // indirect-arg buffer is never also bound as storage.
    let cargs_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cargs"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let cbox_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cbox"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let cdisp_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cdisp"),
        size: 16,
        usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uni_size = std::num::NonZeroU64::new(std::mem::size_of::<Params>() as u64);
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, false),
            storage_entry(2, false),
            storage_entry(3, false),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: uni_size,
                },
                count: None,
            },
            storage_entry(5, false),
            storage_entry(6, false),
            storage_entry(7, false),
            storage_entry(8, false),
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: target_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: canvas_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: results_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: best_buf.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &params_buf,
                    offset: 0,
                    size: uni_size,
                }),
            },
            wgpu::BindGroupEntry { binding: 5, resource: state_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: shapes_out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: cargs_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: cbox_buf.as_entire_binding() },
        ],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("kernels"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let p_eval = make_pipeline(&device, &pl, &shader, "evaluate");
    let p_reduce = make_pipeline(&device, &pl, &shader, "reduce");
    let p_advance = make_pipeline(&device, &pl, &shader, "advance");
    let p_end = make_pipeline(&device, &pl, &shader, "end_shape");
    let p_commit_args = make_pipeline(&device, &pl, &shader, "commit_args");
    let p_commit = make_pipeline(&device, &pl, &shader, "commit");

    // ---- one-time uniform slots -----------------------------------------
    let levels = profile.posterize_levels as f32;
    let max_r = ((w.min(h) as f32) * 0.5).max(3.0);
    let alpha = (profile.shape_alpha.max(1) as f32) / 255.0;
    let cfg = [alpha, levels, max_r, std::f32::consts::PI];
    let extra = [profile.shape_mode as u32, 0, 0, 0];
    let n_rand = profile.random_samples.min(MAX_WORKGROUPS) as u32;
    let slot = |dims: [u32; 4]| Params { dims, cfg, extra };
    let slots = [
        (S_EVAL_RAND, slot([w, h, n_rand, 0])),
        (S_REDUCE_INIT, slot([w, h, n_rand, 0])),
        (S_EVAL_CLIMB, slot([w, h, CLIMB_WIDTH as u32, 1])),
        (S_REDUCE_FOLD, slot([w, h, CLIMB_WIDTH as u32, 1])),
        (S_MISC, slot([w, h, 0, 0])),
    ];
    for (i, p) in slots {
        queue.write_buffer(&params_buf, i * SLOT, bytemuck::bytes_of(&p));
    }

    // ---- main loop (GPU-resident: no CPU in the per-shape search) --------
    let climb_rounds = (profile.mutated_samples / CLIMB_WIDTH).max(1);
    let save_set: std::collections::HashSet<usize> = profile.save_at.iter().cloned().collect();
    let out_dir = output_dir(image_path);
    let _ = fs::create_dir_all(&out_dir);
    let stem = image_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "out".into());
    let off = |s: u64| (s * SLOT) as u32;
    // (shape, rgb, is_rect)
    let mut shapes: Vec<(Ellipse, [u8; 3], bool)> = Vec::with_capacity(stop_at);

    for shape_i in 0..stop_at {
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let pass = |enc: &mut wgpu::CommandEncoder,
                    pipe: &wgpu::ComputePipeline,
                    slot: u64,
                    groups: u32| {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_pipeline(pipe);
            cp.set_bind_group(0, &bind_group, &[off(slot)]);
            cp.dispatch_workgroups(groups, 1, 1);
        };
        pass(&mut enc, &p_eval, S_EVAL_RAND, n_rand);
        pass(&mut enc, &p_reduce, S_REDUCE_INIT, 1);
        for _ in 0..climb_rounds {
            pass(&mut enc, &p_advance, S_MISC, 1);
            pass(&mut enc, &p_eval, S_EVAL_CLIMB, CLIMB_WIDTH as u32);
            pass(&mut enc, &p_reduce, S_REDUCE_FOLD, 1);
        }
        pass(&mut enc, &p_commit_args, S_MISC, 1);
        enc.copy_buffer_to_buffer(&cargs_buf, 0, &cdisp_buf, 0, 12);
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_pipeline(&p_commit);
            cp.set_bind_group(0, &bind_group, &[off(S_MISC)]);
            cp.dispatch_workgroups_indirect(&cdisp_buf, 0);
        }
        pass(&mut enc, &p_end, S_MISC, 1);
        queue.submit(Some(enc.finish()));

        let n = shape_i + 1;
        if n == stop_at || n % profile.save_every == 0 || save_set.contains(&n) {
            let bytes = (n * 3 * 16) as u64;
            let mut enc2 =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc2.copy_buffer_to_buffer(&shapes_out_buf, 0, &read_buf, 0, bytes);
            queue.submit(Some(enc2.finish()));

            let slice = read_buf.slice(..bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            device.poll(wgpu::Maintain::Wait);
            let _ = rx.recv();
            let data = slice.get_mapped_range();
            let f: &[f32] = bytemuck::cast_slice(&data);
            shapes.clear();
            for s in 0..n {
                let b = s * 12;
                let e = Ellipse {
                    cx: f[b + 1],
                    cy: f[b + 2],
                    rx: f[b + 3],
                    ry: f[b + 4],
                    rot: f[b + 5],
                };
                let col = [
                    (f[b + 6] * 255.0).round().clamp(0.0, 255.0) as u8,
                    (f[b + 7] * 255.0).round().clamp(0.0, 255.0) as u8,
                    (f[b + 8] * 255.0).round().clamp(0.0, 255.0) as u8,
                ];
                let is_rect = f[b + 10] >= 0.5;
                shapes.push((e, col, is_rect));
            }
            drop(data);
            read_buf.unmap();

            let path = out_dir.join(format!("{}_{}.json", stem, n));
            let bg_alpha = if profile.transparent_bg { 0 } else { 255 };
            write_json(&path, w, h, bg, bg_alpha, profile.shape_alpha, &shapes);
            println!("Saved {} ({} shapes)", path.display(), n);
            let _ = std::io::stdout().flush();
        }
    }

    println!("DONE!");
    if profile.transparent_bg {
        println!("Transparent sticker: no background, no car-colour matching needed.");
    } else if profile.shape_alpha >= 255 {
        println!("Opaque sticker: solid shapes, looks the same on any car colour.");
    } else {
        let (hue, sat, val) = rgb_to_hsv(bg[0], bg[1], bg[2]);
        println!(
            "Blended shapes (alpha {}): for a clean look set the car colour to HSV\n{:.2},{:.2},{:.2}\n(or use Background=Transparent / Opacity=255 to skip colour matching).",
            profile.shape_alpha, hue, sat, val
        );
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn make_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(layout),
        module: shader,
        entry_point: entry,
        compilation_options: Default::default(),
        cache: None,
    })
}

fn output_dir(image_path: &Path) -> PathBuf {
    let parent = image_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = image_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "out".into());
    parent.join(stem)
}

fn write_json(
    path: &Path,
    w: u32,
    h: u32,
    bg: [f32; 3],
    bg_alpha: u8,
    shape_alpha: u8,
    shapes: &[(Ellipse, [u8; 3], bool)],
) {
    let mut s = String::with_capacity(shapes.len() * 64 + 128);
    s.push_str("{\"shapes\":[");
    s.push_str(&format!(
        "{{\"type\":8,\"data\":[0,0,{},{}],\"color\":[{},{},{},{}]}}",
        w,
        h,
        (bg[0] * 255.0).round() as i32,
        (bg[1] * 255.0).round() as i32,
        (bg[2] * 255.0).round() as i32,
        bg_alpha
    ));
    for (e, c, is_rect) in shapes {
        let mut deg = e.rot.to_degrees() % 360.0;
        if deg < 0.0 {
            deg += 360.0;
        }
        // type 16 = rotated ellipse (id 102), type 1 = rectangle (id 101);
        // the importer maps both.
        let ty = if *is_rect { 1 } else { 16 };
        s.push_str(&format!(
            ",{{\"type\":{},\"data\":[{},{},{},{},{}],\"color\":[{},{},{},{}]}}",
            ty,
            e.cx.round() as i32,
            e.cy.round() as i32,
            e.rx.round().max(1.0) as i32,
            e.ry.round().max(1.0) as i32,
            deg.round() as i32,
            c[0],
            c[1],
            c[2],
            shape_alpha
        ));
    }
    s.push_str("]}");
    if let Ok(mut f) = fs::File::create(path) {
        let _ = f.write_all(s.as_bytes());
    }
}

fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let v = max;
    let s = if max <= 0.0 { 0.0 } else { d / max };
    let h = if d == 0.0 {
        0.0
    } else if max == r {
        ((g - b) / d).rem_euclid(6.0)
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    } / 6.0;
    (h, s, v)
}
