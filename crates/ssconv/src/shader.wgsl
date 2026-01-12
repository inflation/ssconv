// Rec.2100 PQ-encoded BT.2020 input, outputs sRGB with BT.2446A tone mapping.
struct Params {
    width: u32,
    height: u32,
    gain_width: u32,
    gain_height: u32,
};

const HDR_PEAK_NITS: f32 = 1000.0;
const SDR_WHITE_NITS: f32 = 203.0;
const INV_65535: f32 = 1.0 / 65535.0;
const MIN_CONTENT_BOOST: f32 = 1.0;
const MAX_CONTENT_BOOST: f32 = HDR_PEAK_NITS / SDR_WHITE_NITS;
const INV_LOG_BOOST_RANGE: f32 = 1.0 / (log2(MAX_CONTENT_BOOST) - log2(MIN_CONTENT_BOOST));
const INV_GAMMA: f32 = 1.0;

// Packed u16 RGBA; each u32 holds two channels (low/high 16).
@group(0) @binding(0) var<storage, read> input_pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_pixels: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn pack_rgba8(color: vec3<f32>, a: f32) -> u32 {
    let r = u32(clamp(color.r, 0.0, 1.0) * 255.0 + 0.5);
    let g = u32(clamp(color.g, 0.0, 1.0) * 255.0 + 0.5);
    let b = u32(clamp(color.b, 0.0, 1.0) * 255.0 + 0.5);
    let a8 = u32(clamp(a, 0.0, 1.0) * 255.0 + 0.5);
    return r | (g << 8) | (b << 16) | (a8 << 24);
}

fn pq_eotf(v: f32) -> f32 {
    let m1 = 0.1593017578125;
    let m2 = 78.84375;
    let c1 = 0.8359375;
    let c2 = 18.8515625;
    let c3 = 18.6875;
    let v_p = pow(v, 1.0 / m2);
    let num = max(v_p - c1, 0.0);
    let den = c2 - c3 * v_p;
    let l = pow(num / den, 1.0 / m1);
    return clamp(l, 0.0, 1.0) * 10000.0;
}

fn bt2020_to_srgb_linear(rgb_nits: vec3<f32>) -> vec3<f32> {
    // WGSL matrices are column-major; values below are transposed.
    let m = mat3x3<f32>(
        0.016605, -0.001246, -0.000182,
        -0.005876, 0.011329, -0.001006,
        -0.000728, -0.000083, 0.011187
    );
    return m * rgb_nits;
}

fn srgb_oetf(v: f32) -> f32 {
    if (v <= 0.0031308) {
        return 12.92 * v;
    }
    return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

fn bt1886_eotf(v: f32, min_nits: f32, max_nits: f32) -> f32 {
    let lb = pow(min_nits, 1.0 / 2.4);
    let lw = pow(max_nits, 1.0 / 2.4);
    return pow((lw - lb) * v + lb, 2.4);
}

fn bt2446a_tonemap_nits(
    x_nits: f32,
    in_max: f32,
    out_max: f32,
    phdr: f32,
    psdr: f32,
    inv_log_phdr: f32,
    inv_psdr_minus1: f32,
) -> f32 {
    var x = clamp(x_nits / in_max, 0.0, 1.0);
    x = pow(x, 1.0 / 2.4);
    x = log(1.0 + (phdr - 1.0) * x) * inv_log_phdr;

    if (x <= 0.7399) {
        x = 1.0770 * x;
    } else if (x < 0.9909) {
        x = (-1.1510 * x + 2.7811) * x - 0.6302;
    } else {
        x = 0.5 * x + 0.5;
    }

    x = (pow(psdr, x) - 1.0) * inv_psdr_minus1;
    return bt1886_eotf(x, 0.0, out_max);
}

fn decode_rgba_u16(idx: u32) -> vec4<f32> {
    let base = idx * 2u;
    let packed0 = input_pixels[base];
    let packed1 = input_pixels[base + 1u];
    let r = f32(packed0 & 0xFFFFu) * INV_65535;
    let g = f32(packed0 >> 16u) * INV_65535;
    let b = f32(packed1 & 0xFFFFu) * INV_65535;
    let a = f32(packed1 >> 16u) * INV_65535;
    return vec4<f32>(r, g, b, a);
}

fn gain_encode(base_luma: f32, hdr_luma: f32) -> f32 {
    let safe_base = max(base_luma, 1e-6);
    let safe_hdr = max(hdr_luma, 1e-6);
    let gain = safe_hdr / safe_base;
    let log_gain = log2(clamp(gain, MIN_CONTENT_BOOST, MAX_CONTENT_BOOST));
    let normalized = (log_gain - log2(MIN_CONTENT_BOOST)) * INV_LOG_BOOST_RANGE;
    return pow(clamp(normalized, 0.0, 1.0), INV_GAMMA);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let idx = gid.y * params.width + gid.x;
    let encoded = decode_rgba_u16(idx);
    let phdr = 1.0 + 32.0 * pow(HDR_PEAK_NITS / 10000.0, 1.0 / 2.4);
    let psdr = 1.0 + 32.0 * pow(SDR_WHITE_NITS / 10000.0, 1.0 / 2.4);
    let inv_log_phdr = 1.0 / log(phdr);
    let inv_psdr_minus1 = 1.0 / (psdr - 1.0);

    let linear_nits = vec3<f32>(
        pq_eotf(encoded.r),
        pq_eotf(encoded.g),
        pq_eotf(encoded.b)
    );
    let luma = dot(linear_nits, vec3<f32>(0.2627, 0.6780, 0.0593));
    let mapped_luma = bt2446a_tonemap_nits(
        luma,
        HDR_PEAK_NITS,
        SDR_WHITE_NITS,
        phdr,
        psdr,
        inv_log_phdr,
        inv_psdr_minus1,
    );
    let scale = select(mapped_luma / luma, 0.0, luma <= 1e-6);
    let mapped_nits = linear_nits * scale;

    let srgb_linear = bt2020_to_srgb_linear(mapped_nits);
    let srgb = vec3<f32>(
        srgb_oetf(clamp(srgb_linear.r, 0.0, 1.0)),
        srgb_oetf(clamp(srgb_linear.g, 0.0, 1.0)),
        srgb_oetf(clamp(srgb_linear.b, 0.0, 1.0))
    );
    output_pixels[idx] = pack_rgba8(srgb, encoded.a);
}

@compute @workgroup_size(16, 16, 1)
fn gain_map(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.gain_width || gid.y >= params.gain_height) {
        return;
    }

    let base_x = gid.x * 2u;
    let base_y = gid.y * 2u;
    let max_x = params.width - 1u;
    let max_y = params.height - 1u;

    let x0 = min(base_x, max_x);
    let y0 = min(base_y, max_y);
    let x1 = min(base_x + 1u, max_x);
    let y1 = min(base_y + 1u, max_y);

    let idx00 = y0 * params.width + x0;
    let idx01 = y0 * params.width + x1;
    let idx10 = y1 * params.width + x0;
    let idx11 = y1 * params.width + x1;

    let e00 = decode_rgba_u16(idx00);
    let e01 = decode_rgba_u16(idx01);
    let e10 = decode_rgba_u16(idx10);
    let e11 = decode_rgba_u16(idx11);

    let linear00 = vec3<f32>(pq_eotf(e00.r), pq_eotf(e00.g), pq_eotf(e00.b));
    let linear01 = vec3<f32>(pq_eotf(e01.r), pq_eotf(e01.g), pq_eotf(e01.b));
    let linear10 = vec3<f32>(pq_eotf(e10.r), pq_eotf(e10.g), pq_eotf(e10.b));
    let linear11 = vec3<f32>(pq_eotf(e11.r), pq_eotf(e11.g), pq_eotf(e11.b));

    let hdr_luma = 0.25 * (
        dot(linear00, vec3<f32>(0.2627, 0.6780, 0.0593)) +
        dot(linear01, vec3<f32>(0.2627, 0.6780, 0.0593)) +
        dot(linear10, vec3<f32>(0.2627, 0.6780, 0.0593)) +
        dot(linear11, vec3<f32>(0.2627, 0.6780, 0.0593))
    );

    let phdr = 1.0 + 32.0 * pow(HDR_PEAK_NITS / 10000.0, 1.0 / 2.4);
    let psdr = 1.0 + 32.0 * pow(SDR_WHITE_NITS / 10000.0, 1.0 / 2.4);
    let inv_log_phdr = 1.0 / log(phdr);
    let inv_psdr_minus1 = 1.0 / (psdr - 1.0);

    let mapped_luma = bt2446a_tonemap_nits(
        hdr_luma,
        HDR_PEAK_NITS,
        SDR_WHITE_NITS,
        phdr,
        psdr,
        inv_log_phdr,
        inv_psdr_minus1,
    );

    let gain_value = gain_encode(mapped_luma, hdr_luma);
    let gain_u8 = u32(clamp(gain_value, 0.0, 1.0) * 255.0 + 0.5);
    let out_idx = gid.y * params.gain_width + gid.x;
    output_pixels[out_idx] = gain_u8;
}
