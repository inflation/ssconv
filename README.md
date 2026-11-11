# ssconv

Convert HDR JPEG XL screenshots from [ReShade](https://github.com/crosire/reshade)
using the experimental Rust decoder [jxl-rs](https://github.com/libjxl/jxl-rs)
(`jxl` 0.8.0).

```sh
# SDR-compatible JPEG with an embedded HDR gain map
cargo run --release -p ssconv-cli -- screenshot.jxl screenshot.jpg

# Floating-point linear HDR, without tone mapping
cargo run --release -p ssconv-cli -- screenshot.jxl screenshot.exr
```

## Gain-map JPEG output

`.jpg` and `.jpeg` produce an Ultra HDR JPEG: ordinary viewers show a tone-mapped
SDR base image; compatible HDR viewers reconstruct HDR using the embedded RGB
gain map. The file includes MPF, an sRGB ICC profile, Adobe HDR gain-map XMP, and
ISO 21496-1 metadata. JPEG container/metadata assembly uses `ultrahdr-rs` 0.3.5;
JPEG compression uses `image`'s encoder, without native codec dependencies.

Pixels are converted to linear BT.709 and normalized to 203 cd/m² SDR white.
Defaults follow ReShade's screenshot encodings:

| Source transfer | cd/m² represented by linearized 1.0 |
| --- | --- |
| Linear scRGB | 80 |
| Rec. 2020 PQ | 10,000 |
| Rec. 2020 HLG | 1,000 (system gamma 1.2) |
| Other / untagged SDR | 203 |

For other linear conventions, HLG display peaks, or ICC-only HDR profiles without
CICP, override the absolute scale:

```sh
cargo run --release -p ssconv-cli -- input.jxl output.jpg --reference-white-nits 203
```

The base image uses a hue-preserving Reinhard shoulder. The gain map is computed
against the decoded lossy base JPEG, stored at full resolution with RGB channels,
and encoded at JPEG quality 95. Gain bounds are shared between the map encoding
and its metadata, including for flat black / SDR-only images.

JPEG export composites alpha over black and clips negative RGB after conversion
to BT.709. It cannot preserve colors outside that gamut; use EXR when you need
negative samples or alpha. No automatic content exposure adjustment is applied.

## Decoding and EXR output

Both bare JPEG XL codestreams and containers are recognized by signature, with a
case-insensitive `.jxl` extension fallback. The first visible frame is decoded as
32-bit floating-point RGBA, preserving half-float scRGB highlights and the
precision of integer PQ/HLG screenshots. Standard native D65 color encodings are
represented directly as CICP profiles; embedded ICC profiles are retained. This
avoids applying display-oriented CLUTs from synthesized HDR ICC profiles.

EXR is linear BT.709 / D65 and preserves alpha and negative out-of-gamut values.
It does not tone map or rescale luminance: scRGB retains its original scale,
PQ linearizes to 1.0 = 10,000 cd/m², and HLG uses the color engine's display-linear
transfer. `--reference-white-nits` applies only to JPEG output. Other input
formats supported by `image` can also be converted.

The library exposes `load_image(path)`,
`SourceImage::convert(&destination_profile) -> Result<Rgba32FImage, SsconvError>`,
and `SourceImage::encode_gain_map(GainMapOptions) -> Result<Vec<u8>, SsconvError>`.
`SourceImage::colorspace` is the authoritative source profile; the contained
`DynamicImage`'s default CICP metadata may differ.

## Validation

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Tests cover synthetic ReShade-style scRGB/PQ/HLG JXL fixtures, raw/container JXL,
alpha, negative values, highlights, truncated input, EXR output, and gain-map JPEG
output. Reconstruction checks decode both JPEGs and apply the metadata to verify
HDR luminance across all three transfer functions. Real ReShade captures and
HDR viewer interoperability have not yet been validated.

Tests do not require the ignored `samples/` folder. Fixture generation is in
`ssconv-core/tests/fixtures/generate.c` and requires libjxl development headers
and library (generated with libjxl 0.11.2).
