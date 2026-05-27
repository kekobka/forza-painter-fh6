// forza-painter — single-exe Forza Horizon vinyl helper.
//
//   forza-painter                         -> launch the GUI
//   forza-painter generate <image>        -> GPU geometry (profile index on stdin)
//   forza-painter import <geometry.json> [--game fh6] [--pid N]
//                 [--count-address 0x..] [--table-address 0x..]
//   forza-painter diagnose      [--game fh6] [--pid N]
//   forza-painter auto-locate   --layer-count K [--game fh6] [--pid N]
//
// On non-Windows hosts only `generate` and the GUI shell build; import/probe
// need live game-process memory and are Windows-only.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod backend;
mod generate;
mod gui;
mod procs;
mod profiles;

#[cfg(windows)]
mod winmem;
#[cfg(windows)]
mod importer;
#[cfg(windows)]
mod probe;

use std::path::PathBuf;

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn parse_hex(s: &str) -> Option<u64> {
    let s = s.trim();
    let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    u64::from_str_radix(s, 16).ok()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("");

    match cmd {
        "generate" => {
            let Some(img) = args.get(2) else {
                eprintln!("usage: forza-painter generate <image> [--max-resolution N --stop-at N --random-samples N --mutated-samples N --posterize-levels N --save-every N]");
                std::process::exit(1);
            };
            let img = PathBuf::from(img);
            let keys = [
                "--max-resolution",
                "--stop-at",
                "--random-samples",
                "--mutated-samples",
                "--posterize-levels",
                "--save-every",
                "--background",
                "--opacity",
                "--shapes",
            ];
            if keys.iter().any(|k| flag(&args, k).is_some()) {
                let n = |name: &str, d: usize| {
                    flag(&args, name).and_then(|s| s.trim().parse().ok()).unwrap_or(d)
                };
                let profile = generate::Profile {
                    max_resolution: n("--max-resolution", 1000) as u32,
                    random_samples: n("--random-samples", 1500).max(1),
                    mutated_samples: n("--mutated-samples", 600),
                    posterize_levels: n("--posterize-levels", 256).clamp(2, 256) as u32,
                    stop_at: n("--stop-at", 3000).max(1),
                    save_every: n("--save-every", 50).max(1),
                    save_at: Vec::new(),
                    transparent_bg: flag(&args, "--background")
                        .map(|s| s.trim().eq_ignore_ascii_case("transparent"))
                        .unwrap_or(false),
                    shape_alpha: flag(&args, "--opacity")
                        .and_then(|s| s.trim().parse::<u32>().ok())
                        .unwrap_or(255)
                        .clamp(1, 255) as u8,
                    shape_mode: match flag(&args, "--shapes").as_deref() {
                        Some("rect") => 1,
                        Some("mixed") => 2,
                        _ => 0,
                    },
                };
                generate::generate_with_profile(&img, &profile);
            } else {
                generate::cli_generate(img);
            }
        }
        "import" => {
            let Some(json) = args.get(2) else {
                eprintln!("usage: forza-painter import <geometry.json> [--game fh6] [--pid N]");
                std::process::exit(1);
            };
            let game = flag(&args, "--game");
            let pid = flag(&args, "--pid").and_then(|p| p.parse().ok());
            let ca = flag(&args, "--count-address").and_then(|s| parse_hex(&s));
            let ta = flag(&args, "--table-address").and_then(|s| parse_hex(&s));
            let exp = flag(&args, "--layer-count").and_then(|s| s.parse().ok());
            let edge_mask = args.iter().any(|a| a == "--edge-mask");
            let mut log = |s: String| println!("{s}");
            match backend::run_import(
                std::path::Path::new(json),
                game.as_deref(),
                pid,
                ca,
                ta,
                exp,
                edge_mask,
                &mut log,
            ) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            }
        }
        "diagnose" => {
            let game = flag(&args, "--game");
            let pid = flag(&args, "--pid").and_then(|p| p.parse().ok());
            let mut log = |s: String| println!("{s}");
            backend::run_diagnose(game.as_deref(), pid, &mut log);
        }
        "auto-locate" => {
            let game = flag(&args, "--game");
            let pid = flag(&args, "--pid").and_then(|p| p.parse().ok());
            let lc: u32 = flag(&args, "--layer-count")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            if lc == 0 {
                eprintln!("usage: forza-painter auto-locate --layer-count K");
                std::process::exit(1);
            }
            let mut log = |s: String| println!("{s}");
            if let Err(e) = backend::run_auto_locate(game.as_deref(), pid, lc, &mut log) {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
        "" => {
            let opts = eframe::NativeOptions {
                viewport: egui::ViewportBuilder::default()
                    .with_inner_size([1060.0, 800.0])
                    .with_min_inner_size([900.0, 620.0]),
                ..Default::default()
            };
            if let Err(e) = eframe::run_native(
                "forza-painter",
                opts,
                Box::new(|cc| Ok(Box::new(gui::App::new(cc)))),
            ) {
                eprintln!("GUI failed to start: {e}");
                std::process::exit(1);
            }
        }
        other => {
            eprintln!("Unknown command '{other}'. Run with no arguments for the GUI.");
            std::process::exit(1);
        }
    }
}
