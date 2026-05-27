// GPU-resident geometrizer kernels.
//
// The whole per-shape search (random sampling + hill-climb) runs on the GPU
// with no CPU in the loop: candidates are generated in-shader from a
// counter-based RNG, scored in parallel, reduced to the running best, and the
// winner is committed and recorded into `shapes_out`. The CPU only reads
// `shapes_out` back at save points. Same primitive/geometrize energy and
// optimal-colour formula as the original CPU-orchestrated version.

struct Params {
    dims: vec4<u32>,  // w, h, n_candidates, flag (eval: 0=random 1=climb; reduce: 0=init 1=fold)
    cfg: vec4<f32>,   // alpha, posterize_levels, max_r, PI
    extra: vec4<u32>, // shape_mode (0=ellipse 1=rect 2=mixed), _, _, _
};

@group(0) @binding(0) var<storage, read>       target_img: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> canvas:     array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> results:    array<vec4<f32>>; // 3 vec4 / candidate
@group(0) @binding(3) var<storage, read_write> bestbuf:    array<vec4<f32>>; // 3 vec4 (running best)
@group(0) @binding(4) var<uniform>             params:     Params;
@group(0) @binding(5) var<storage, read_write> state:      array<u32>;       // [seed, round, shape]
@group(0) @binding(6) var<storage, read_write> shapes_out: array<vec4<f32>>; // 3 vec4 / committed shape
@group(0) @binding(7) var<storage, read_write> cargs:      array<u32>;       // indirect dispatch [x,y,z]
@group(0) @binding(8) var<storage, read_write> cbox:       array<u32>;       // commit bbox [ox,oy,bw,bh]

const WG: u32 = 64u;
const RWG: u32 = 256u;
const BIG: f32 = 1.0e30;
// Per-pixel cost for covering a masked (transparent) pixel. ~ the max
// per-pixel SSE so a shape may only graze the subject edge, not bleed.
const PEN: f32 = 3.0;

var<workgroup> shared_red: array<f32, WG>;
var<workgroup> red_val: array<f32, RWG>;
var<workgroup> red_idx: array<u32, RWG>;

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn next_f(st: ptr<function, u32>) -> f32 {
    *st = pcg(*st);
    return f32(*st) * 2.3283064365386963e-10;
}
fn next_g(st: ptr<function, u32>) -> f32 {
    let u1 = max(next_f(st), 1.0e-7);
    let u2 = next_f(st);
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}

// Primitive kind for a candidate: 0 = ellipse, 1 = rectangle.
fn shape_kind(cand: u32) -> u32 {
    let mode = params.extra.x;
    if (mode == 0u) { return 0u; }
    if (mode == 1u) { return 1u; }
    let hsh = pcg(state[0] ^ (state[2] * 0x27D4EB2Fu) ^ (cand * 0x165667B1u) ^ 0x9E3779B9u);
    return hsh & 1u;
}

// AABB half-extents for a kind/rx/ry rotated by (ct,st).
fn shape_half(kind: u32, rx: f32, ry: f32, ct: f32, st: f32) -> vec2<f32> {
    if (kind == 1u) {
        return vec2<f32>(abs(rx * ct) + abs(ry * st), abs(rx * st) + abs(ry * ct));
    }
    return vec2<f32>(sqrt(rx * rx * ct * ct + ry * ry * st * st),
                     sqrt(rx * rx * st * st + ry * ry * ct * ct));
}

// Is local (uu,vv) inside the shape?
fn shape_inside(kind: u32, uu: f32, vv: f32, rx: f32, ry: f32) -> bool {
    if (kind == 1u) {
        return abs(uu) <= rx && abs(vv) <= ry;
    }
    return (uu * uu) / (rx * rx) + (vv * vv) / (ry * ry) <= 1.0;
}

fn gen_candidate(cand: u32) -> array<f32, 5> {
    let w = f32(params.dims.x);
    let h = f32(params.dims.y);
    let max_r = params.cfg.z;
    var rs: u32 = pcg(state[0]
        ^ (state[2] * 0x9E3779B1u)
        ^ (state[1] * 0x85EBCA77u)
        ^ (cand * 0xC2B2AE35u));
    if (params.dims.w == 0u) {
        let cx = next_f(&rs) * w;
        let cy = next_f(&rs) * h;
        let rx = 1.0 + next_f(&rs) * (max_r - 1.0);
        let ry = 1.0 + next_f(&rs) * (max_r - 1.0);
        let rot = next_f(&rs) * params.cfg.w;
        return array<f32, 5>(cx, cy, rx, ry, rot);
    }
    // climb: mutate the running best (bestbuf)
    let bcx = bestbuf[0].y;
    let bcy = bestbuf[0].z;
    let brx = bestbuf[0].w;
    let bry = bestbuf[1].x;
    let brot = bestbuf[1].y;
    let cx = clamp(bcx + next_g(&rs) * (w * 0.05 + 1.0), 0.0, w - 1.0);
    let cy = clamp(bcy + next_g(&rs) * (h * 0.05 + 1.0), 0.0, h - 1.0);
    let rx = clamp(brx * (1.0 + next_g(&rs) * 0.15), 1.0, max_r * 2.0);
    let ry = clamp(bry * (1.0 + next_g(&rs) * 0.15), 1.0, max_r * 2.0);
    let rot = brot + next_g(&rs) * 0.35;
    return array<f32, 5>(cx, cy, rx, ry, rot);
}

@compute @workgroup_size(WG)
fn evaluate(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_index) lid: u32) {
    let cand = wid.x;
    if (cand >= params.dims.z) {
        return;
    }
    let w = params.dims.x;
    let h = params.dims.y;
    let e = gen_candidate(cand);
    let cx = e[0];
    let cy = e[1];
    let rx = max(e[2], 0.5);
    let ry = max(e[3], 0.5);
    let ct = cos(e[4]);
    let st = sin(e[4]);
    let kind = shape_kind(cand);

    let half = shape_half(kind, rx, ry, ct, st);
    let hx = half.x;
    let hy = half.y;
    let x0 = clamp(i32(floor(cx - hx)), 0, i32(w) - 1);
    let y0 = clamp(i32(floor(cy - hy)), 0, i32(h) - 1);
    let x1 = clamp(i32(ceil(cx + hx)), 0, i32(w) - 1);
    let y1 = clamp(i32(ceil(cy + hy)), 0, i32(h) - 1);
    let bw = u32(x1 - x0 + 1);
    let bh = u32(y1 - y0 + 1);
    let total = bw * bh;

    // acc[0..15] = moment sums over scored pixels; acc[16] = count of covered
    // pixels that are masked (transparent), used as a spill penalty.
    var acc: array<f32, 17>;
    for (var i = 0u; i < 17u; i = i + 1u) { acc[i] = 0.0; }
    var p = lid;
    loop {
        if (p >= total) { break; }
        let px = u32(x0) + (p % bw);
        let py = u32(y0) + (p / bw);
        let dx = f32(px) - cx;
        let dy = f32(py) - cy;
        let uu = dx * ct + dy * st;
        let vv = -dx * st + dy * ct;
        if (shape_inside(kind, uu, vv, rx, ry)) {
            let idx = py * w + px;
            let T4 = target_img[idx];
            // target.w is the score mask: 0 = transparent pixel, skip it so
            // shapes are never placed/scored over transparent areas.
            if (T4.w >= 0.5) {
            let T = T4.xyz;
            let C = canvas[idx].xyz;
            acc[0] = acc[0] + 1.0;
            acc[1] = acc[1] + T.x; acc[2] = acc[2] + T.y; acc[3] = acc[3] + T.z;
            acc[4] = acc[4] + C.x; acc[5] = acc[5] + C.y; acc[6] = acc[6] + C.z;
            acc[7] = acc[7] + C.x * C.x; acc[8] = acc[8] + C.y * C.y; acc[9] = acc[9] + C.z * C.z;
            acc[10] = acc[10] + T.x * T.x; acc[11] = acc[11] + T.y * T.y; acc[12] = acc[12] + T.z * T.z;
            acc[13] = acc[13] + C.x * T.x; acc[14] = acc[14] + C.y * T.y; acc[15] = acc[15] + C.z * T.z;
            } else {
                acc[16] = acc[16] + 1.0;
            }
        }
        p = p + WG;
    }

    var red: array<f32, 17>;
    for (var k = 0u; k < 17u; k = k + 1u) {
        shared_red[lid] = acc[k];
        workgroupBarrier();
        var stride = WG / 2u;
        loop {
            if (stride == 0u) { break; }
            if (lid < stride) {
                shared_red[lid] = shared_red[lid] + shared_red[lid + stride];
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        red[k] = shared_red[0];
        workgroupBarrier();
    }

    if (lid == 0u) {
        let base = cand * 3u;
        let cnt = red[0];
        if (cnt < 1.0) {
            results[base + 0u] = vec4<f32>(BIG, cx, cy, rx);
            results[base + 1u] = vec4<f32>(ry, e[4], 0.0, 0.0);
            results[base + 2u] = vec4<f32>(0.0, 0.0, f32(kind), 0.0);
            return;
        }
        let a = params.cfg.x;
        let levels = params.cfg.y;
        var col: array<f32, 3>;
        var delta = 0.0;
        for (var c = 0u; c < 3u; c = c + 1u) {
            let st_ = red[1u + c];
            let sc = red[4u + c];
            let scc = red[7u + c];
            let stt = red[10u + c];
            let sct = red[13u + c];
            var kk = (st_ - (1.0 - a) * sc) / (a * cnt);
            kk = clamp(kk, 0.0, 1.0);
            kk = round(kk * (levels - 1.0)) / (levels - 1.0);
            let new_c = (1.0 - a) * (1.0 - a) * scc
                + a * a * kk * kk * cnt
                + stt
                + 2.0 * (1.0 - a) * a * kk * sc
                - 2.0 * (1.0 - a) * sct
                - 2.0 * a * kk * st_;
            let old_c = scc - 2.0 * sct + stt;
            delta = delta + (new_c - old_c);
            col[c] = kk;
        }
        // Spill penalty: every covered transparent pixel costs PEN, so a shape
        // only wins if it tightly fits the subject instead of bleeding into
        // the masked background.
        delta = delta + red[16u] * PEN;
        results[base + 0u] = vec4<f32>(delta, cx, cy, rx);
        results[base + 1u] = vec4<f32>(ry, e[4], col[0], col[1]);
        results[base + 2u] = vec4<f32>(col[2], 1.0, f32(kind), 0.0);
    }
}

@compute @workgroup_size(RWG)
fn reduce(@builtin(local_invocation_index) lid: u32) {
    let n = params.dims.z;
    var best_v = BIG;
    var best_i = 0u;
    var c = lid;
    loop {
        if (c >= n) { break; }
        let d = results[c * 3u].x;
        if (d < best_v) {
            best_v = d;
            best_i = c;
        }
        c = c + RWG;
    }
    red_val[lid] = best_v;
    red_idx[lid] = best_i;
    workgroupBarrier();
    var stride = RWG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride && red_val[lid + stride] < red_val[lid]) {
            red_val[lid] = red_val[lid + stride];
            red_idx[lid] = red_idx[lid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid == 0u) {
        let b = red_idx[0] * 3u;
        let cand0 = results[b + 0u];
        let cand1 = results[b + 1u];
        let cand2 = results[b + 2u];
        // flag 1 = fold: keep the previous running best if it is still better
        if (params.dims.w == 1u && bestbuf[0].x <= cand0.x) {
            return;
        }
        bestbuf[0] = cand0;
        bestbuf[1] = cand1;
        bestbuf[2] = cand2;
    }
}

@compute @workgroup_size(1)
fn advance() {
    state[1] = state[1] + 1u;
}

@compute @workgroup_size(1)
fn end_shape() {
    let s = state[2];
    shapes_out[s * 3u + 0u] = bestbuf[0];
    shapes_out[s * 3u + 1u] = bestbuf[1];
    shapes_out[s * 3u + 2u] = bestbuf[2];
    state[2] = s + 1u;
    state[1] = 0u;
}

// One thread: turn the winning ellipse into a bounding box + an indirect
// dispatch size, so `commit` only touches the box, not the whole canvas.
@compute @workgroup_size(1)
fn commit_args() {
    let w = params.dims.x;
    let h = params.dims.y;
    let cx = bestbuf[0].y;
    let cy = bestbuf[0].z;
    let rx = max(bestbuf[0].w, 0.5);
    let ry = max(bestbuf[1].x, 0.5);
    let ct = cos(bestbuf[1].y);
    let st = sin(bestbuf[1].y);
    let kind = u32(bestbuf[2].z + 0.5);
    let half = shape_half(kind, rx, ry, ct, st);
    let hx = half.x;
    let hy = half.y;
    let x0 = clamp(i32(floor(cx - hx)), 0, i32(w) - 1);
    let y0 = clamp(i32(floor(cy - hy)), 0, i32(h) - 1);
    let x1 = clamp(i32(ceil(cx + hx)), 0, i32(w) - 1);
    let y1 = clamp(i32(ceil(cy + hy)), 0, i32(h) - 1);
    let bw = u32(max(x1 - x0 + 1, 1));
    let bh = u32(max(y1 - y0 + 1, 1));
    cbox[0] = u32(x0);
    cbox[1] = u32(y0);
    cbox[2] = bw;
    cbox[3] = bh;
    cargs[0] = (bw * bh + WG - 1u) / WG;
    cargs[1] = 1u;
    cargs[2] = 1u;
}

@compute @workgroup_size(WG)
fn commit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bw = cbox[2];
    let bh = cbox[3];
    let t = gid.x;
    if (t >= bw * bh) {
        return;
    }
    let w = params.dims.x;
    let px = cbox[0] + (t % bw);
    let py = cbox[1] + (t / bw);
    let p = py * w + px;

    let cx = bestbuf[0].y;
    let cy = bestbuf[0].z;
    let rx = max(bestbuf[0].w, 0.5);
    let ry = max(bestbuf[1].x, 0.5);
    let ct = cos(bestbuf[1].y);
    let st = sin(bestbuf[1].y);
    let a = params.cfg.x;
    let col = vec3<f32>(bestbuf[1].z, bestbuf[1].w, bestbuf[2].x);
    let kind = u32(bestbuf[2].z + 0.5);

    let dx = f32(px) - cx;
    let dy = f32(py) - cy;
    let uu = dx * ct + dy * st;
    let vv = -dx * st + dy * ct;
    if (shape_inside(kind, uu, vv, rx, ry)) {
        let cc = canvas[p].xyz;
        canvas[p] = vec4<f32>(mix(cc, col, a), 1.0);
    }
}
