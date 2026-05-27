//! Thin platform boundary so the GUI never touches Windows-only types.
//! On Windows these call the real importer/probe; elsewhere they degrade
//! gracefully (the generator still works cross-platform).

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct Located {
    pub pid: u32,
    pub count_address: u64,
    pub table_address: u64,
    pub layer_count: u32,
}

#[cfg(windows)]
pub fn run_diagnose(game: Option<&str>, pid: Option<u32>, log: &mut dyn FnMut(String)) -> bool {
    crate::importer::diagnose(
        &crate::importer::ImportArgs {
            game: game.map(|s| s.to_string()),
            pid,
            layer_count_address: None,
            layer_table_address: None,
            expected_count: None,
            edge_mask: false,
        },
        log,
    )
}

#[cfg(windows)]
pub fn run_auto_locate(
    game: Option<&str>,
    pid: Option<u32>,
    layer_count: u32,
    log: &mut dyn FnMut(String),
) -> Result<Located, String> {
    let s = crate::probe::auto_locate(game, pid, layer_count, log)?;
    Ok(Located {
        pid: s.pid,
        count_address: s.count_address,
        table_address: s.table_address,
        layer_count: s.layer_count,
    })
}

#[cfg(windows)]
#[allow(dead_code)]
pub fn run_inspect(
    game: Option<&str>,
    pid: Option<u32>,
    table: u64,
    count: u32,
    log: &mut dyn FnMut(String),
) -> Result<(), String> {
    crate::probe::inspect_table(game, pid, table, count, 12, log)
}

#[cfg(windows)]
#[allow(clippy::too_many_arguments)]
pub fn run_import(
    json: &std::path::Path,
    game: Option<&str>,
    pid: Option<u32>,
    count_address: Option<u64>,
    table_address: Option<u64>,
    expected_count: Option<u32>,
    edge_mask: bool,
    log: &mut dyn FnMut(String),
) -> Result<(), String> {
    crate::importer::load_geometry(
        json,
        &crate::importer::ImportArgs {
            game: game.map(|s| s.to_string()),
            pid,
            layer_count_address: count_address,
            layer_table_address: table_address,
            expected_count,
            edge_mask,
        },
        log,
    )
}

#[cfg(windows)]
pub fn saved_session() -> Option<Located> {
    let s = crate::probe::load_session()?;
    if crate::probe::session_pid_is_live(&s) {
        Some(Located {
            pid: s.pid,
            count_address: s.count_address,
            table_address: s.table_address,
            layer_count: s.layer_count,
        })
    } else {
        None
    }
}

// ---- non-Windows stubs: compile + run the generator/GUI anywhere ----------
#[cfg(not(windows))]
const NOT_WIN: &str = "Import/probe is Windows-only (needs live game process memory).";

#[cfg(not(windows))]
pub fn run_diagnose(_: Option<&str>, _: Option<u32>, log: &mut dyn FnMut(String)) -> bool {
    log(NOT_WIN.into());
    false
}
#[cfg(not(windows))]
pub fn run_auto_locate(
    _: Option<&str>,
    _: Option<u32>,
    _: u32,
    log: &mut dyn FnMut(String),
) -> Result<Located, String> {
    log(NOT_WIN.into());
    Err(NOT_WIN.into())
}
#[cfg(not(windows))]
#[allow(dead_code)]
pub fn run_inspect(
    _: Option<&str>,
    _: Option<u32>,
    _: u64,
    _: u32,
    log: &mut dyn FnMut(String),
) -> Result<(), String> {
    log(NOT_WIN.into());
    Err(NOT_WIN.into())
}
#[cfg(not(windows))]
#[allow(clippy::too_many_arguments)]
pub fn run_import(
    _: &std::path::Path,
    _: Option<&str>,
    _: Option<u32>,
    _: Option<u64>,
    _: Option<u64>,
    _: Option<u32>,
    _: bool,
    log: &mut dyn FnMut(String),
) -> Result<(), String> {
    log(NOT_WIN.into());
    Err(NOT_WIN.into())
}
#[cfg(not(windows))]
pub fn saved_session() -> Option<Located> {
    None
}
