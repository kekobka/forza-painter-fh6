//! Importer — faithful port of main.py (load_geometry / draw_memory_shape /
//! calculate_CLivery / diagnose_livery). Windows-only.
//!
//! Writes generated geometry into a live Forza Horizon vinyl-editor process.
//! Every offset/formula mirrors main.py exactly; behaviour is reported through
//! the `log` callback so the GUI/CLI can stream it like the Python did.

use std::path::Path;

use crate::procs::resolve_pid;
use crate::profiles::{get_profile, GameProfile, FH6_DISCOVERED_TABLE_POINTER_DELTA};
use crate::winmem::Proc;

struct Shape {
    type_id: i64,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    rot_deg: f64,
    color: [u8; 4],
    is_mask: bool,
}

pub struct ImportArgs {
    pub game: Option<String>,
    pub pid: Option<u32>,
    pub layer_count_address: Option<u64>,
    pub layer_table_address: Option<u64>,
    /// The exact ungrouped layer count the user entered; used to reject a
    /// false-positive located address whose value doesn't match.
    pub expected_count: Option<u32>,
    /// Add the 4 FH5-era clip rectangles. Off by default: on FH6 the mask
    /// flag does not clip and they just show as a black frame.
    pub edge_mask: bool,
}

fn as_i64(v: &serde_json::Value) -> i64 {
    v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)).unwrap_or(0)
}
fn as_f64(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}
fn as_u8(v: &serde_json::Value) -> u8 {
    as_i64(v).clamp(0, 255) as u8
}

/// main.py calculate_CLivery
fn calculate_clivery(p: &Proc, pr: &GameProfile, log: &mut dyn FnMut(String)) -> Option<u64> {
    let base = match p.base_address() {
        Ok(b) => b,
        Err(e) => {
            log(format!("Unable to open process. Try running as administrator. {e}"));
            return None;
        }
    };
    log(format!("Attempting to scan for {} livery address:", pr.label));
    let mut pre_a: i64 = -1;
    let mut sig_hit: &[u8] = &[];
    'outer: for &(off, size) in pr.scan_regions {
        let start = base + off;
        for sig in pr.signature_patterns {
            log(format!(
                "Scanning {} Base+{:x}..Base+{:x}",
                pr.label,
                off,
                off + size
            ));
            let rel = p.scan_block(start, size, sig);
            if rel != -1 {
                pre_a = (start as i64) + rel;
                sig_hit = sig;
                break 'outer;
            }
        }
    }
    if pre_a == -1 {
        log(format!(
            "Unsupported {} version and cannot find a matching pattern.",
            pr.label
        ));
        log("FH6 support may need a new signature or offsets for this game build.".into());
        return None;
    }
    let pre_a = pre_a as u64;
    log(format!(
        "Signature {} found at Base+{:x}",
        sig_hit.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" "),
        pre_a - base
    ));
    if p.read_u64(pre_a) != p.read_u64(pre_a + pr.validation_mirror_offset) {
        log(format!(
            "Matching signature failed validation at Base+{:x}",
            pre_a - base
        ));
        return None;
    }
    let addr_a = p.deref(pre_a + pr.livery_root_pointer_offset);
    log(format!(
        "Found livery root pointer at Base+{:x}",
        pre_a + pr.livery_root_pointer_offset - base
    ));
    let addr_b = p.deref(addr_a + pr.editor_pointer_offset);
    if addr_b == 0 {
        log("Create Vinyl Group menu not detected".into());
        return None;
    }
    let clivery = p.deref(addr_b + pr.livery_pointer_offset);
    if clivery == 0 {
        log("Create Vinyl Group menu not detected".into());
        return None;
    }
    Some(clivery)
}

/// main.py diagnose_livery — read-only signature/pointer-chain check.
pub fn diagnose(args: &ImportArgs, log: &mut dyn FnMut(String)) -> bool {
    let (pid, key) = match resolve_pid(args.game.as_deref(), args.pid) {
        Some(v) => v,
        None => {
            log("No supported Forza Horizon process is running.".into());
            return false;
        }
    };
    let pr = match get_profile(&key) {
        Some(p) => p,
        None => {
            log(format!("Unsupported game profile '{key}'."));
            return false;
        }
    };
    let p = match Proc::open(pid) {
        Ok(p) => p,
        Err(e) => {
            log(e);
            return false;
        }
    };
    let base = match p.base_address() {
        Ok(b) => b,
        Err(e) => {
            log(format!("Unable to open process {pid}. Try running as administrator. {e}"));
            return false;
        }
    };
    log(format!("{} diagnostics for pid {}", pr.label, pid));
    log(format!("Base address: 0x{base:x}"));
    let mut found_valid = false;
    for &(off, size) in pr.scan_regions {
        let start = base + off;
        for sig in pr.signature_patterns {
            let rel = p.scan_block(start, size, sig);
            if rel == -1 {
                log(format!("No match in Base+{:x}..Base+{:x}", off, off + size));
                continue;
            }
            let abs = start + rel as u64;
            log(format!(
                "Signature {} found at Base+{:x}",
                sig.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" "),
                abs - base
            ));
            let mirror_equal =
                p.read_u64(abs) == p.read_u64(abs + pr.validation_mirror_offset);
            let root = p.deref(abs + pr.livery_root_pointer_offset);
            let editor = if root != 0 {
                p.deref(root + pr.editor_pointer_offset)
            } else {
                0
            };
            let livery = if editor != 0 {
                p.deref(editor + pr.livery_pointer_offset)
            } else {
                0
            };
            log(format!(
                "Validation mirror: {}",
                if mirror_equal { "OK" } else { "FAILED" }
            ));
            log(format!("Root pointer: 0x{root:x}"));
            log(format!("Editor pointer: 0x{editor:x}"));
            log(format!("Livery pointer: 0x{livery:x}"));
            found_valid = found_valid
                || (mirror_equal && root != 0 && editor != 0 && livery != 0);
        }
    }
    if !found_valid {
        log("No validated livery pointer chain found.".into());
    }
    found_valid
}

/// main.py draw_memory_shape — returns false on the grouped-vinyl write error.
fn draw_memory_shape(
    p: &Proc,
    pr: &GameProfile,
    s: &Shape,
    index: usize,
    layer_addresses: &[u64],
    livery_count: i64,
    log: &mut dyn FnMut(String),
) -> bool {
    if index as i64 >= livery_count || index >= layer_addresses.len() {
        return true;
    }
    let addr = layer_addresses[index];

    let do_writes = || -> Result<(), String> {
        let mut pos = Vec::with_capacity(8);
        pos.extend_from_slice(&(s.x as f32).to_le_bytes());
        pos.extend_from_slice(&(-(s.y as f32)).to_le_bytes());
        p.write(addr + pr.layer_position_offset, &pos)?;

        let div: f32 = if s.type_id == 16 { 63.0 } else { 127.0 };
        let mut scale = Vec::with_capacity(8);
        scale.extend_from_slice(&((s.w as f32) / div).to_le_bytes());
        scale.extend_from_slice(&((s.h as f32) / div).to_le_bytes());
        p.write(addr + pr.layer_scale_offset, &scale)?;

        let rot = (360.0 - s.rot_deg as f32).to_le_bytes();
        p.write(addr + pr.layer_rotation_offset, &rot)?;

        p.write(addr + pr.layer_color_offset, &s.color)?;

        if s.type_id == 16 {
            p.write(addr + pr.layer_shape_id_offset, &[102u8])?;
        } else if s.type_id == 1 {
            p.write(addr + pr.layer_shape_id_offset, &[101u8])?;
        }
        p.write(addr + pr.layer_mask_offset, &[if s.is_mask { 1u8 } else { 0u8 }])?;
        Ok(())
    };

    if do_writes().is_err() {
        log(format!("Write failed at slot {}.", index + 1));
        if index < 16 {
            // Failing this early almost never means a real group; it means the
            // located table address is wrong (false positive — common with a
            // small template, since the scanned layer count is not unique).
            log("This is most likely a WRONG located table (false positive), not a grouped vinyl.".into());
            log("Use an ungrouped template with >=500 layers (1000+ reliable on FH6),".into());
            log("enter the EXACT current layer count, and re-run Auto-locate.".into());
        } else {
            log("Possible grouped vinyl: make sure the template is ungrouped,".into());
            log("and that you are in the Vinyl Group Editor (not applying a vinyl/livery).".into());
        }
        return false;
    }
    true
}

/// main.py load_geometry. Returns Ok(()) on success.
pub fn load_geometry(
    path: &Path,
    args: &ImportArgs,
    log: &mut dyn FnMut(String),
) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("Not a valid file: {e}"))?;
    let data: serde_json::Value =
        serde_json::from_str(&text).map_err(|_| "Not a valid .json file".to_string())?;
    let shapes_json = data
        .get("shapes")
        .and_then(|s| s.as_array())
        .ok_or("Not a valid generated geometry .json file")?;
    if shapes_json.is_empty() {
        return Err("No shapes were loaded. Check your exported geometry .json".into());
    }
    let head = &shapes_json[0];
    let head_data = head.get("data").and_then(|d| d.as_array());
    let head_color = head.get("color").and_then(|c| c.as_array());
    let valid = head_data.map(|d| d.len() == 4).unwrap_or(false)
        && as_i64(head.get("type").unwrap_or(&serde_json::Value::Null)) > 0
        && head_color.map(|c| c.len() == 4).unwrap_or(false);
    if !valid {
        return Err("Not a valid generated geometry .json file".into());
    }
    let hd = head_data.unwrap();
    let image_w = as_i64(&hd[2]);
    let image_h = as_i64(&hd[3]);
    let hc = head_color.unwrap();
    let (bg_r, bg_g, bg_b, bg_a) = (as_u8(&hc[0]), as_u8(&hc[1]), as_u8(&hc[2]), as_u8(&hc[3]));

    let mut shapes: Vec<Shape> = Vec::new();
    if bg_a > 0 {
        shapes.push(Shape {
            type_id: 1,
            x: (image_w / 2) as f64,
            y: (image_h / 2) as f64,
            w: image_w as f64,
            h: image_h as f64,
            rot_deg: 0.0,
            color: [bg_r, bg_g, bg_b, bg_a],
            is_mask: false,
        });
    }
    for sh in &shapes_json[1..] {
        let t = as_i64(sh.get("type").unwrap_or(&serde_json::Value::Null));
        // 16 = rotated ellipse, 1 = rectangle (the generator can emit both).
        if t == 16 || t == 1 {
            let d = sh.get("data").and_then(|d| d.as_array()).ok_or("bad shape data")?;
            let c = sh.get("color").and_then(|c| c.as_array()).ok_or("bad shape color")?;
            shapes.push(Shape {
                type_id: t,
                x: as_f64(&d[0]),
                y: as_f64(&d[1]),
                w: as_f64(&d[2]),
                h: as_f64(&d[3]),
                rot_deg: as_f64(&d[4]),
                color: [as_u8(&c[0]), as_u8(&c[1]), as_u8(&c[2]), as_u8(&c[3])],
                is_mask: false,
            });
        } else {
            return Err(
                "Unsupported shape in geometry file (only ellipse=16 / rect=1).".into(),
            );
        }
    }
    if shapes.is_empty() {
        return Err("No shapes were loaded. Check your exported geometry .json".into());
    }

    let game = args.game.clone().unwrap_or_else(|| "fh6".into());
    let (pid, key) = resolve_pid(Some(&game), args.pid)
        .ok_or("No supported Forza Horizon process is running")?;
    let pr = get_profile(&key).ok_or_else(|| format!("Unsupported game '{key}'"))?;
    log(format!("{} (pid {})", pr.label, pid));
    let p = Proc::open(pid)?;

    let is_fh6 = key == "fh6";
    let current_livery_count: i64;
    let clivery_layer_table: u64;
    if let Some(count_addr) = args.layer_count_address {
        current_livery_count = if let Some(exp) = args.expected_count {
            // FH6: the live count is a u16 inside a bigger struct, so trust the
            // exact count the user entered (upstream --layer-count-value).
            log(format!("Using template layer count {exp} (address 0x{count_addr:x})"));
            exp as i64
        } else if is_fh6 {
            let c = p.read_u16(count_addr) as i64;
            log(format!("Manual FH6 layer count address 0x{count_addr:x} -> {c} (u16)"));
            c
        } else {
            let c = p.read_u32(count_addr) as i64;
            log(format!("Manual layer count address 0x{count_addr:x} -> {c}"));
            c
        };
        let table = if let Some(t) = args.layer_table_address {
            t
        } else {
            let field = count_addr + FH6_DISCOVERED_TABLE_POINTER_DELTA;
            let t = p.deref(field);
            log(format!("Manual table pointer field 0x{field:x} -> 0x{t:x}"));
            t
        };
        clivery_layer_table = table;
    } else {
        let clivery = calculate_clivery(&p, &pr, log)
            .ok_or("Could not resolve CLivery via signature chain")?;
        log(format!("CLivery found at {clivery:x}"));
        let clivery_group = p.deref(clivery + pr.livery_group_offset);
        if clivery_group == 0 {
            return Err("cLiveryGroup is invalid. You are probably not in `Create Vinyl Group` menu.".into());
        }
        log(format!("CLiveryGroup found at {clivery_group:x}"));
        current_livery_count = p.read_u32(clivery_group + pr.livery_count_offset) as i64;
        clivery_layer_table = p.deref(clivery_group + pr.layer_table_offset);
    }

    if current_livery_count < 100 {
        log("READ THE INSTRUCTIONS".into());
        log("Load an UNGROUPED vinyl group of spheres (>=100; 500-3000 recommended) first,".into());
        log("and enter the EXACT current layer count.".into());
        return Err("Layer count below 100".into());
    }
    if clivery_layer_table == 0 {
        return Err("cLiveryLayer table is invalid. You are probably not in `Create Vinyl Group` menu.".into());
    }
    log(format!("CLiveryLayer table found at {clivery_layer_table:x}"));

    if shapes.len() > 2996 {
        shapes.truncate(2996);
    }
    let cap = (current_livery_count - 4) as usize;
    if shapes.len() > cap {
        shapes.truncate(cap);
    }
    // The 4 masking rectangles (exact main.py formulas).
    let push_mask = |v: &mut Vec<Shape>, x: i64, y: i64, w: i64, h: i64| {
        v.push(Shape {
            type_id: 1,
            x: x as f64,
            y: y as f64,
            w: w as f64,
            h: h as f64,
            rot_deg: 0.0,
            color: [0, 0, 0, 255],
            is_mask: true,
        });
    };
    if args.edge_mask {
        push_mask(&mut shapes, -(image_w / 4), image_h / 2, image_w / 2, (image_h as f64 * 1.5) as i64);
        push_mask(&mut shapes, image_w + image_w / 4, image_h / 2, image_w / 2, (image_h as f64 * 1.5) as i64);
        push_mask(&mut shapes, image_w / 2, -(image_h / 4), image_w + image_w, image_h / 2);
        push_mask(&mut shapes, image_w / 2, image_h + image_h / 4, image_w + image_w, image_h / 2);
    } else {
        let _ = &push_mask;
    }

    // Resolve all layer pointers in one read (native.read_pointer_table).
    let raw = p.read(clivery_layer_table, current_livery_count as usize * 8);
    let usable = raw.len() - (raw.len() % 8);
    let layer_addresses: Vec<u64> = raw[..usable]
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // Write every template slot. Slots past our shapes are hidden, so a larger
    // template does not leave its original spheres visible (the "circles stayed
    // white" symptom) after importing a smaller JSON.
    let clear = Shape {
        type_id: 1,
        x: -100000.0,
        y: -100000.0,
        w: 0.0,
        h: 0.0,
        rot_deg: 0.0,
        color: [0, 0, 0, 0],
        is_mask: true,
    };
    let total = current_livery_count as usize;
    for i in 0..total {
        let s = shapes.get(i).unwrap_or(&clear);
        if i == 0 || (i + 1) % 100 == 0 || i + 1 == total {
            log(format!("Writing layer {}/{}", i + 1, total));
        }
        if !draw_memory_shape(&p, &pr, s, i, &layer_addresses, current_livery_count, log) {
            return Err("Import aborted: write failed (wrong located table / grouped vinyl / wrong editor state). See log above.".into());
        }
    }

    log("DONE!".into());
    let (hh, ss, vv) = rgb_to_hsv(
        bg_r as f32 / 255.0,
        bg_g as f32 / 255.0,
        bg_b as f32 / 255.0,
    );
    log(format!(
        "The ideal background color for the car is:\n{hh:.2},{ss:.2},{vv:.2}"
    ));
    Ok(())
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
