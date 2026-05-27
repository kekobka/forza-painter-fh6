//! FH6 runtime locator — port of the path app.py actually drives in
//! fh6_probe.py: scan memory for the live layer count, score nearby pointers
//! as candidate layer tables, persist the winning count/table pair. Plus the
//! read-only `inspect_table` helper. Windows-only.

use std::path::{Path, PathBuf};

use crate::procs::resolve_pid;
use crate::profiles::{get_profile, GameProfile};
use crate::winmem::{is_user_pointer, Proc};

#[derive(Clone)]
pub struct Session {
    pub pid: u32,
    pub layer_count: u32,
    pub group_address: u64,
    pub count_address: u64,
    pub table_pointer_field: u64,
    pub table_address: u64,
    pub score: i64,
}

pub fn session_path() -> PathBuf {
    PathBuf::from("fh6-session.json")
}

pub fn load_session() -> Option<Session> {
    let text = std::fs::read_to_string(session_path()).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    Some(Session {
        pid: v.get("pid")?.as_u64()? as u32,
        layer_count: v.get("layer_count")?.as_u64()? as u32,
        group_address: v.get("group_address").and_then(|x| x.as_u64()).unwrap_or(0),
        count_address: v.get("count_address")?.as_u64()?,
        table_pointer_field: v.get("table_pointer_field").and_then(|x| x.as_u64()).unwrap_or(0),
        table_address: v.get("table_address")?.as_u64()?,
        score: v.get("score").and_then(|s| s.as_i64()).unwrap_or(0),
    })
}

fn write_session(s: &Session, process: &str) {
    let p = session_path();
    if let Some(dir) = p.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let json = serde_json::json!({
        "type": "fh6_session_location_v1",
        "pid": s.pid,
        "process": process,
        "layer_count": s.layer_count,
        "group_address": s.group_address,
        "count_address": s.count_address,
        "table_pointer_field": s.table_pointer_field,
        "table_address": s.table_address,
        "score": s.score,
        "locator": "u16_group_layout",
    });
    let _ = std::fs::write(p, serde_json::to_vec_pretty(&json).unwrap_or_default());
}

fn plausible(v: f32) -> bool {
    v == v && (-100000.0..100000.0).contains(&v)
}

fn read_float_pair(p: &Proc, addr: u64) -> Option<(f32, f32)> {
    let b = p.read(addr, 8);
    if b.len() != 8 {
        return None;
    }
    Some((
        f32::from_le_bytes(b[0..4].try_into().unwrap()),
        f32::from_le_bytes(b[4..8].try_into().unwrap()),
    ))
}

/// fh6_probe.score_layer_pointer — returns (score, human checks).
fn score_layer_pointer(p: &Proc, ptr: u64, pr: &GameProfile) -> (i64, Vec<String>) {
    let mut checks = Vec::new();
    if !is_user_pointer(ptr) {
        return (0, checks);
    }
    let mut score = 0;
    if let Some((a, b)) = read_float_pair(p, ptr + pr.layer_position_offset) {
        if (-10000.0..10000.0).contains(&a) && (-10000.0..10000.0).contains(&b) {
            score += 1;
            checks.push(format!("pos={a:.2},{b:.2}"));
        }
    }
    if let Some((a, b)) = read_float_pair(p, ptr + pr.layer_scale_offset) {
        if a.abs() < 10000.0 && b.abs() < 10000.0 {
            score += 1;
            checks.push(format!("scale={a:.3},{b:.3}"));
        }
    }
    let color = p.read(ptr + pr.layer_color_offset, 4);
    if color.len() == 4 {
        score += 1;
        checks.push(format!(
            "color={:02x} {:02x} {:02x} {:02x}",
            color[0], color[1], color[2], color[3]
        ));
    }
    let shape = p.read(ptr + pr.layer_shape_id_offset, 1);
    if shape.len() == 1 && matches!(shape[0], 0 | 1 | 2 | 100 | 101 | 102) {
        score += 1;
        checks.push(format!("shape={}", shape[0]));
    }
    let mask = p.read(ptr + pr.layer_mask_offset, 1);
    if mask.len() == 1 && matches!(mask[0], 0 | 1) {
        score += 1;
        checks.push(format!("mask={}", mask[0]));
    }
    let _ = plausible(0.0);
    (score, checks)
}

/// fh6_probe.strict_layer_pointer — a confident, fully-formed layer struct.
fn strict_layer_pointer(p: &Proc, ptr: u64, pr: &GameProfile) -> bool {
    if !p.is_private_writable(ptr) {
        return false;
    }
    let pos = match read_float_pair(p, ptr + pr.layer_position_offset) {
        Some(v) => v,
        None => return false,
    };
    if !(-10000.0..10000.0).contains(&pos.0) || !(-10000.0..10000.0).contains(&pos.1) {
        return false;
    }
    let scale = match read_float_pair(p, ptr + pr.layer_scale_offset) {
        Some(v) => v,
        None => return false,
    };
    if scale.0.abs() >= 10000.0 || scale.1.abs() >= 10000.0 {
        return false;
    }
    if scale.0.abs() < 0.0001 && scale.1.abs() < 0.0001 {
        return false;
    }
    let color = p.read(ptr + pr.layer_color_offset, 4);
    if color.len() != 4 || !matches!(color[3], 0 | 255) {
        return false;
    }
    let shape = p.read(ptr + pr.layer_shape_id_offset, 1);
    if shape.len() != 1 || !matches!(shape[0], 0 | 1 | 2 | 100 | 101 | 102) {
        return false;
    }
    let mask = p.read(ptr + pr.layer_mask_offset, 1);
    mask.len() == 1 && matches!(mask[0], 0 | 1)
}

type Samples = Vec<(usize, u64, i64, Vec<String>)>;

/// fh6_probe.score_table — strict: every sampled entry must be a private
/// writable pointer; enough entries must look layer-like and be distinct.
fn score_table(p: &Proc, pr: &GameProfile, table: u64, sample_count: i64) -> (i64, Samples) {
    if !p.is_private_writable(table) {
        return (0, Vec::new());
    }
    let sample_total = sample_count.clamp(0, 64) as usize;
    let mut pointers = Vec::with_capacity(sample_total);
    let mut scores: Samples = Vec::new();
    let mut layer_like = 0usize;
    let mut total = 0i64;
    for index in 0..sample_total {
        let ptr = p.deref(table + (index as u64) * 8);
        if !p.is_private_writable(ptr) {
            return (0, Vec::new());
        }
        pointers.push(ptr);
        let (score, checks) = score_layer_pointer(p, ptr, pr);
        total += score;
        if score >= 3 {
            layer_like += 1;
        }
        if score > 0 && index < 8 {
            scores.push((index, ptr, score, checks));
        }
    }
    if sample_total >= 16 {
        let unique = pointers.iter().collect::<std::collections::HashSet<_>>().len();
        if unique < std::cmp::max(8, sample_total * 3 / 4) {
            return (0, Vec::new());
        }
        if layer_like < std::cmp::max(8, sample_total / 2) {
            return (0, Vec::new());
        }
    }
    (total + layer_like as i64, scores)
}

/// fh6_probe.validate_table_layer_coverage — the table really must cover the
/// whole template with layer-like, mostly-strict entries.
fn validate_table_layer_coverage(
    p: &Proc,
    pr: &GameProfile,
    table: u64,
    layer_count: u32,
) -> (bool, usize) {
    if !p.is_private_writable(table) {
        return (false, 0);
    }
    let required = (layer_count as usize).min(3000);
    let scan_limit = 3000.min(std::cmp::max(required + 512, required * 2));
    let mut valid = 0usize;
    let mut strict_valid = 0usize;
    let mut seen = std::collections::HashSet::new();
    for index in 0..scan_limit {
        let ptr = p.deref(table + (index as u64) * 8);
        if seen.contains(&ptr) {
            continue;
        }
        if p.is_private_writable(ptr) {
            let (score, _) = score_layer_pointer(p, ptr, pr);
            if score >= 3 {
                seen.insert(ptr);
                valid += 1;
                if strict_layer_pointer(p, ptr, pr) {
                    strict_valid += 1;
                }
                if valid >= required {
                    let strict_required =
                        required.min(std::cmp::max(32, required / 4));
                    return (strict_valid >= strict_required, strict_valid);
                }
            }
        }
    }
    (false, strict_valid)
}

/// fh6_probe.auto_locate_count_table — the FH6 `u16_group_layout` locator:
/// scan private writable memory for the u16 layer count, reconstruct the
/// CLiveryGroup at count-0x5A, follow group+0x78 to the real layer table, and
/// only accept it if it actually covers the whole template.
pub fn auto_locate(
    game: Option<&str>,
    pid_override: Option<u32>,
    layer_count: u32,
    log: &mut dyn FnMut(String),
) -> Result<Session, String> {
    let (pid, key) =
        resolve_pid(game, pid_override).ok_or("No supported Forza Horizon process is running")?;
    let pr = get_profile(&key).ok_or_else(|| format!("Unsupported game '{key}'"))?;
    let p = Proc::open(pid)?;
    log(format!(
        "Auto-locating FH6 layer group for count {layer_count} (u16_group_layout)..."
    ));

    let started = std::time::Instant::now();
    let max_seconds = 45u64;
    let max_candidates = 200_000usize;
    let mut candidates = 0usize;

    // The live FH6 count can be off-by-one from what the user enters
    // (ungroup counting), so try a small neighbourhood and lock onto the
    // first value that yields a validated layer group.
    let mut order: Vec<i64> = Vec::new();
    for d in [0i64, -1, 1, -2, 2, -3, 3] {
        let v = layer_count as i64 + d;
        if v >= 1 && !order.contains(&v) {
            order.push(v);
        }
    }

    let mut matched: u32 = layer_count;
    // (score, group, count_addr, table_field, table, valid_entries, samples)
    let mut groups: Vec<(i64, u64, u64, u64, u64, usize, Samples)> = Vec::new();

    'counts: for lc_i in order {
        let lc = lc_i as u32;
        let pattern = (lc as u16).to_le_bytes();
        groups.clear();
        if started.elapsed().as_secs() > max_seconds {
            log(format!("Stopped FH6 scan after {max_seconds}s."));
            break;
        }
        log(format!("Scanning for layer count {lc}..."));
        'regions: for (base, size) in p.regions(true) {
            if started.elapsed().as_secs() > max_seconds {
                log(format!("Stopped FH6 scan after {max_seconds}s."));
                break;
            }
            if size == 0 || size > 256 * 1024 * 1024 {
                continue;
            }
            let mem = p.read(base, size as usize);
            if mem.len() != size as usize {
                continue;
            }
            let mut start = 0usize;
            while let Some(rel) = mem[start..].windows(2).position(|w| w == pattern) {
                let pos = start + rel;
                start = pos + 1;
                candidates += 1;
                if candidates > max_candidates {
                    log(format!("Stopped FH6 scan after {max_candidates} count hits."));
                    break 'regions;
                }
                let count_address = base + pos as u64;
                if count_address < pr.livery_count_offset {
                    continue;
                }
                let group_address = count_address - pr.livery_count_offset;
                if group_address < base {
                    continue;
                }
                let table_field = group_address + pr.layer_table_offset;
                let table = p.read_u64(table_field);
                if !is_user_pointer(table) || !p.is_private_writable(table) {
                    continue;
                }
                let (score, samples) = score_table(&p, &pr, table, (lc as i64).min(64));
                if score <= 0 {
                    continue;
                }
                let (ok, valid_entries) =
                    validate_table_layer_coverage(&p, &pr, table, lc);
                if !ok {
                    continue;
                }
                log(format!(
                    "layout candidate group=0x{group_address:x} count=0x{count_address:x} table=0x{table:x} validated={valid_entries}/{lc}"
                ));
                groups.push((
                    score + 60,
                    group_address,
                    count_address,
                    table_field,
                    table,
                    valid_entries,
                    samples,
                ));
            }
            if !groups.is_empty() {
                break;
            }
        }
        if !groups.is_empty() {
            matched = lc;
            break 'counts;
        }
    }

    groups.sort_by(|a, b| b.0.cmp(&a.0));
    let winner = groups.first().ok_or(
        "No safe FH6 layer group found near that count. Use an UNGROUPED template and enter the EXACT current layer count, then re-run Auto-locate.",
    )?;
    if matched != layer_count {
        log(format!(
            "Locked onto actual layer count {matched} (you entered {layer_count})."
        ));
    }
    for (sc, g, c, tf, t, ve, samples) in groups.iter().take(5) {
        log(format!(
            "candidate score={sc} group=0x{g:x} count=0x{c:x} tableField=0x{tf:x} table=0x{t:x} validated={ve}"
        ));
        for (idx, ptr, ls, checks) in samples.iter().take(4) {
            log(format!("  table[{idx}] ptr=0x{ptr:x} score={ls} {}", checks.join("; ")));
        }
    }
    let s = Session {
        pid,
        layer_count: matched,
        group_address: winner.1,
        count_address: winner.2,
        table_pointer_field: winner.3,
        table_address: winner.4,
        score: winner.0,
    };
    write_session(&s, &format!("pid{pid}"));
    log(format!(
        "Saved FH6 session: layer_count={} group=0x{:x} count=0x{:x} table=0x{:x} score={}",
        s.layer_count, s.group_address, s.count_address, s.table_address, s.score
    ));
    Ok(s)
}

/// fh6_probe.inspect_table — read-only dump of the first entries of a table.
pub fn inspect_table(
    game: Option<&str>,
    pid_override: Option<u32>,
    table_address: u64,
    layer_count: u32,
    max_layers: usize,
    log: &mut dyn FnMut(String),
) -> Result<(), String> {
    let (pid, key) =
        resolve_pid(game, pid_override).ok_or("No supported Forza Horizon process is running")?;
    let pr = get_profile(&key).ok_or_else(|| format!("Unsupported game '{key}'"))?;
    let p = Proc::open(pid)?;
    log(format!(
        "Inspecting table=0x{table_address:x} layers={layer_count}"
    ));
    let end = (layer_count as usize).min(max_layers);
    let mut valid = 0;
    for index in 0..end {
        let entry = table_address + (index as u64) * 8;
        let ptr = p.deref(entry);
        log(format!("table[{index}] @0x{entry:x} -> 0x{ptr:x}"));
        if !is_user_pointer(ptr) {
            continue;
        }
        let (sc, checks) = score_layer_pointer(&p, ptr, &pr);
        if sc > 0 {
            valid += 1;
        }
        log(format!("  layer score={sc} {}", checks.join("; ")));
        let raw = p.read(ptr, 0x80);
        for (row, chunk) in raw.chunks(16).enumerate() {
            let hex: Vec<String> = chunk.iter().map(|b| format!("{b:02x}")).collect();
            log(format!("  +0x{:03x}: {}", row * 16, hex.join(" ")));
        }
    }
    log(format!("Valid scored layer entries: {valid}"));
    Ok(())
}

pub fn session_pid_is_live(s: &Session) -> bool {
    crate::procs::find_game_processes()
        .iter()
        .any(|g| g.pid == s.pid)
}

pub fn _unused(_: &Path) {}
