# forza-painter (Rust, single exe)

From-scratch Rust rewrite: **GPU** geometry generator + live-game importer +
egui GUI, in one `forza-painter.exe`. No Python, no bundled closed-source
generator.

## Build (Windows, MSVC toolchain)

```powershell
# install rustup from https://rustup.rs (MSVC default), then, in the repo root:
cargo build --release
Copy-Item -Force target\release\forza-painter.exe forza-painter.exe
# (cmd.exe instead of PowerShell: copy target\release\forza-painter.exe forza-painter.exe)
```

Run `forza-painter.exe` from the repo root so it finds `settings/*.ini`
(presets) in the working directory. Run as administrator if the log shows
`OpenProcess` failures.

## Use

Launch with no arguments for the GUI (two tabs):

- **Generate** — add image(s), pick a preset, tweak the runtime editor
  (resolution / shapes / random / mutated / posterize / save-every /
  background White vs Transparent), Start generating.
- **Import** — add the generated JSON (or "Use generated JSON"), enter the
  exact ungrouped template layer count, Import. The live address is
  auto-located; no manual addresses.

CLI:

```
forza-painter generate <image> [--max-resolution N --stop-at N --random-samples N
                                --mutated-samples N --posterize-levels N
                                --save-every N --background white|transparent]
forza-painter import <geom.json> [--game fh6] [--pid N] [--layer-count N]
forza-painter diagnose    [--game fh6] [--pid N]
forza-painter auto-locate --layer-count K [--game fh6] [--pid N]
```

`generate` with no flags falls back to choosing a `settings/*.ini` preset by
1-based index read from stdin (legacy/bat compatibility).

## How it works

- **Generator** — GPU-resident primitive/geometrize hill-climb
  (`shaders.wgsl`): candidates and the hill-climb are generated in-shader from
  a counter-based RNG, scored in parallel (one workgroup per candidate,
  moment-sums over only covered pixels), reduced to the running best,
  committed and recorded entirely on the GPU. The CPU only reads results back
  at save points, so a fast GPU stays saturated. Transparent mode keeps an
  alpha mask in `target.w`; the shader skips `w<0.5` so only the opaque
  subject gets shapes (no background, bare car shows through).
- **Importer** (`importer.rs` / `winmem.rs` / `profiles.rs`) — faithful port
  of the upstream Python (`main.py` / `native.py` / `game_profiles.py`) with
  byte-exact offsets.
- **FH6 locator** (`probe.rs`) — the `u16_group_layout` locator: scans
  MEM_PRIVATE writable memory for the u16 layer count, reconstructs the
  CLiveryGroup at `count-0x5A`, follows `group+0x78` to the layer table, and
  accepts it only after a strict coverage check.

## Verification

`cargo check` (Linux + `x86_64-pc-windows-gnu`) and `naga` WGSL validation run
in CI-of-record here. GPU output quality and the live-game import are verified
on Windows against Forza Horizon 6 (confirmed working by the maintainer).

## Not implemented

The advanced FH6 memory snapshot/compare troubleshooting
(`--save-memory-snapshot` / `--compare-memory-snapshot` in the old
`fh6_probe.py`) is not reimplemented. The normal workflow — generate, import,
auto-locate, diagnose, inspect-table — is fully covered.
