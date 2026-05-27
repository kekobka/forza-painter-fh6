//! Portable process discovery (app.py game_processes / main.py get_pid).

use crate::profiles::all_profiles;

#[derive(Clone)]
pub struct GameProc {
    pub pid: u32,
    pub name: String,
    pub profile_key: String,
}

impl GameProc {
    pub fn label(&self) -> String {
        format!("{} pid {}", self.name, self.pid)
    }
}

/// Every running process whose name matches a known Forza profile.
pub fn find_game_processes() -> Vec<GameProc> {
    let profiles = all_profiles();
    let sys = sysinfo::System::new_all();
    let mut out = Vec::new();
    for (pid, proc_) in sys.processes() {
        let name = proc_.name().to_string_lossy().to_string();
        let lname = name.to_lowercase();
        for p in &profiles {
            if p.process_names.iter().any(|n| n.to_lowercase() == lname) {
                out.push(GameProc {
                    pid: pid.as_u32(),
                    name: name.clone(),
                    profile_key: p.key.to_string(),
                });
                break;
            }
        }
    }
    out
}

/// Resolve a pid + profile key the way main.py get_pid does: an explicit pid
/// wins, otherwise the first running matching process for `prefer_key`.
pub fn resolve_pid(prefer_key: Option<&str>, pid_override: Option<u32>) -> Option<(u32, String)> {
    let found = find_game_processes();
    if let Some(pid) = pid_override {
        if let Some(g) = found.iter().find(|g| g.pid == pid) {
            return Some((pid, g.profile_key.clone()));
        }
        // Unknown process but caller forced a pid: trust the requested profile.
        return prefer_key.map(|k| (pid, k.to_string()));
    }
    if let Some(k) = prefer_key {
        if let Some(g) = found.iter().find(|g| g.profile_key == k) {
            return Some((g.pid, g.profile_key.clone()));
        }
    }
    found.first().map(|g| (g.pid, g.profile_key.clone()))
}
