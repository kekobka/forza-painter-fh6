// Statically parse + validate the WGSL so a shader error is caught at
// `cargo test` time instead of only when a GPU runs it.
#[test]
fn wgsl_is_valid() {
    let src = include_str!("../src/shaders.wgsl");
    let module = naga::front::wgsl::parse_str(src).expect("WGSL failed to parse");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .expect("WGSL failed naga validation");
}
