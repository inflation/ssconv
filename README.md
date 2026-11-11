# ssconv

Image colorspace conversion in Rust, including HDR JPEG XL screenshots from
[ReShade](https://github.com/crosire/reshade).

```sh
cargo run --release -p ssconv-cli -- screenshot.jxl screenshot.exr
```

The CLI converts to floating-point, linear BT.709 / D65 OpenEXR, preserving alpha,
negative out-of-gamut values, and HDR highlights. Output must have an `.exr`
extension. Existing image formats supported by the `image` crate also work as
inputs.

JPEG XL decoding supports both raw codestreams and containers, detected by file
signature (with a case-insensitive `.jxl` extension fallback). ReShade's linear
scRGB half-float screenshots retain floating-point samples; Rec. 2020 PQ and HLG
screenshots retain 16-bit precision until conversion. The decoder's rendered ICC
profile describes the pixels, including when JXL stores a native color encoding
instead of an embedded ICC profile. HDR JXL is never treated as untagged sRGB.

Conversion does not tone map or rescale luminance to a shared reference white.
Linear scRGB keeps its original scale (1.0 = 80 cd/m²); PQ linearizes to a scale
where 1.0 = 10,000 cd/m². HLG uses the color engine's display-linear transfer
(default system gamma 1.2). Set the exposure/reference scale appropriately in the
application reading the EXR. SDR export and JXL encoding are not implemented.

The library exposes `load_image(path)` and
`SourceImage::convert(&destination_profile) -> Result<Rgba32FImage, SsconvError>`
for conversion to other profiles. `SourceImage::colorspace` is the authoritative
source profile; the contained `DynamicImage`'s default CICP metadata may differ.

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Tests include tiny synthetic HDR fixtures with ReShade's color encodings, raw
and container JXL, alpha, negative values and highlights, PQ/HLG linearization,
and truncated input. They do not require files in the ignored `samples/` folder.
Fixture generation is documented in `ssconv-core/tests/fixtures/generate.c` and
requires libjxl development headers and library (generated with libjxl 0.11.2).
