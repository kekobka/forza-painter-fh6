//! Game memory profile — a faithful port of game_profiles.py.
//!
//! Every offset here is a byte-exact copy of the Python constants. Do not
//! "tidy" these values: they are the live Forza Horizon vinyl-editor memory
//! layout and the importer writes into a running game using them.

#[derive(Clone)]
pub struct GameProfile {
    pub key: &'static str,
    pub label: &'static str,
    pub process_names: &'static [&'static str],
    pub signature_patterns: &'static [&'static [u8]],
    pub scan_regions: &'static [(u64, u64)],
    pub validation_mirror_offset: u64,
    pub livery_root_pointer_offset: u64,
    pub editor_pointer_offset: u64,
    pub livery_pointer_offset: u64,
    pub livery_group_offset: u64,
    pub livery_count_offset: u64,
    pub layer_table_offset: u64,
    pub layer_position_offset: u64,
    pub layer_scale_offset: u64,
    pub layer_rotation_offset: u64,
    pub layer_color_offset: u64,
    pub layer_mask_offset: u64,
    pub layer_shape_id_offset: u64,
}

/// main.py: FH6_DISCOVERED_TABLE_POINTER_DELTA
pub const FH6_DISCOVERED_TABLE_POINTER_DELTA: u64 = 0x1E;

const KNOWN_LIVERY_SIGNATURE: &[u8] = &[0x12, 0x47, 0x9B, 0x13, 0x29, 0xD9, 0xA2, 0xB1];
const KNOWN_SIGS: &[&[u8]] = &[KNOWN_LIVERY_SIGNATURE];
const COMMON_SCAN_REGIONS: &[(u64, u64)] = &[
    (0x06000000, 0x02000000),
    (0x08000000, 0x02000000),
    (0x0A000000, 0x02000000),
];

const fn base(
    key: &'static str,
    label: &'static str,
    process_names: &'static [&'static str],
) -> GameProfile {
    GameProfile {
        key,
        label,
        process_names,
        signature_patterns: KNOWN_SIGS,
        scan_regions: COMMON_SCAN_REGIONS,
        validation_mirror_offset: 0x70,
        livery_root_pointer_offset: 0xB8,
        editor_pointer_offset: 0xA58,
        livery_pointer_offset: 0x8,
        livery_group_offset: 0x20,
        livery_count_offset: 0x5A,
        layer_table_offset: 0x78,
        layer_position_offset: 0x18,
        layer_scale_offset: 0x28,
        layer_rotation_offset: 0x50,
        layer_color_offset: 0x74,
        layer_mask_offset: 0x78,
        layer_shape_id_offset: 0x7A,
    }
}

pub fn get_profile(key: &str) -> Option<GameProfile> {
    match key.to_lowercase().as_str() {
        "fh6" => Some(base(
            "fh6",
            "Forza Horizon 6",
            &["ForzaHorizon6.exe", "ForzaHorizon6-Win64-Shipping.exe"],
        )),
        "fh5" => Some(base("fh5", "Forza Horizon 5", &["ForzaHorizon5.exe"])),
        _ => None,
    }
}

pub fn all_profiles() -> Vec<GameProfile> {
    vec![get_profile("fh6").unwrap(), get_profile("fh5").unwrap()]
}
