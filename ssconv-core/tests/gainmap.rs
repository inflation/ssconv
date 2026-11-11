use image::{DynamicImage, Rgba, Rgba32FImage};
use moxcms::{
    CicpColorPrimaries, CicpProfile, ColorProfile, MatrixCoefficients, TransferCharacteristics,
};
use ssconv_core::{GainMapOptions, SourceImage};
use ultrahdr_rs::Decoder;

fn source(transfer: TransferCharacteristics, level: f32, alpha: f32) -> SourceImage {
    SourceImage {
        image: DynamicImage::ImageRgba32F(Rgba32FImage::from_pixel(
            32,
            32,
            Rgba([level, level, level, alpha]),
        )),
        colorspace: ColorProfile::new_from_cicp(CicpProfile {
            color_primaries: CicpColorPrimaries::Bt709,
            transfer_characteristics: transfer,
            matrix_coefficients: MatrixCoefficients::Identity,
            full_range: true,
        }),
    }
}

fn reconstruct(bytes: &[u8]) -> f64 {
    let decoder = Decoder::new(bytes).unwrap();
    assert!(decoder.is_ultrahdr());
    let metadata = decoder.metadata().unwrap();
    metadata.validate().unwrap();
    assert!(metadata.alternate_hdr_headroom > 0.0);
    let base = image::load_from_memory(decoder.primary_jpeg().unwrap())
        .unwrap()
        .to_rgb8();
    let map = image::load_from_memory(decoder.gainmap_jpeg().unwrap())
        .unwrap()
        .to_rgb8();
    assert_eq!(map.dimensions(), base.dimensions());
    let value = f64::from(base.get_pixel(16, 16).0[0]) / 255.0;
    let linear = if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    };
    let gain = f64::from(map.get_pixel(16, 16).0[0]) / 255.0;
    let log_gain =
        metadata.gain_map_min[0] + gain * (metadata.gain_map_max[0] - metadata.gain_map_min[0]);
    (linear + metadata.base_offset[0]) * log_gain.exp2() - metadata.alternate_offset[0]
}

#[test]
fn hdr_reconstruction_uses_absolute_luminance_for_each_transfer() {
    // All three inputs represent 1000 cd/m². HLG inverse with system gamma 1.2.
    let hlg_scene = 1.0_f32; // HLG peak 1000 cd/m² by default.
    for (transfer, level) in [
        (TransferCharacteristics::Linear, 12.5),
        (TransferCharacteristics::Smpte2084, 0.751_827_1),
        (TransferCharacteristics::Hlg, hlg_scene),
    ] {
        let encoded = source(transfer, level, 1.0)
            .encode_gain_map(GainMapOptions::default())
            .unwrap();
        let decoder = Decoder::new(&encoded).unwrap();
        assert!(
            (decoder.metadata().unwrap().alternate_hdr_headroom.exp2() / (1000.0 / 203.0) - 1.0)
                .abs()
                < 0.001
        );
        let actual = reconstruct(&encoded);
        let expected = 1000.0 / 203.0;
        assert!(
            (actual / expected - 1.0).abs() < 0.04,
            "{transfer:?}: {actual} != {expected}"
        );
        // Both primary and secondary carry ISO signaling; XMP supports older readers.
        assert_eq!(
            encoded
                .windows(b"urn:iso:std:iso:ts:21496:-1".len())
                .filter(|w| *w == b"urn:iso:std:iso:ts:21496:-1")
                .count(),
            2
        );
        assert!(encoded.windows(b"hdrgm".len()).any(|w| w == b"hdrgm"));
        assert_eq!(image::load_from_memory(&encoded).unwrap().width(), 32);
    }
}

#[test]
fn alpha_composites_over_black_and_white_override_is_applied() {
    let encoded = source(TransferCharacteristics::Linear, 8.0, 0.5)
        .encode_gain_map(GainMapOptions {
            reference_white_nits: Some(203.0),
            ..GainMapOptions::default()
        })
        .unwrap();
    assert!((reconstruct(&encoded) / 4.0 - 1.0).abs() < 0.04);
}

#[test]
fn black_sdr_and_invalid_inputs_are_handled() {
    for level in [0.0, 0.2, -1.0] {
        let encoded = source(TransferCharacteristics::Linear, level, 1.0)
            .encode_gain_map(GainMapOptions::default())
            .unwrap();
        assert!(reconstruct(&encoded).is_finite());
    }
    let image = source(TransferCharacteristics::Linear, 1.0, 1.0);
    for value in [0.0, -1.0, f32::INFINITY, f32::NAN] {
        assert!(
            image
                .encode_gain_map(GainMapOptions {
                    reference_white_nits: Some(value),
                    ..GainMapOptions::default()
                })
                .is_err()
        );
    }
    assert!(
        image
            .encode_gain_map(GainMapOptions {
                quality: 0,
                ..GainMapOptions::default()
            })
            .is_err()
    );
    assert!(
        source(TransferCharacteristics::Linear, f32::NAN, 1.0)
            .encode_gain_map(GainMapOptions::default())
            .is_err()
    );
}
