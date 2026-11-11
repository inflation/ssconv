use image::{DynamicImage, GenericImageView};
use moxcms::{CicpColorPrimaries, CicpProfile, MatrixCoefficients, TransferCharacteristics};
use ssconv_core::{ColorProfile, load_image};

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn linear() -> ColorProfile {
    ColorProfile::new_from_cicp(CicpProfile {
        color_primaries: CicpColorPrimaries::Bt709,
        transfer_characteristics: TransferCharacteristics::Linear,
        matrix_coefficients: MatrixCoefficients::Identity,
        full_range: true,
    })
}

#[test]
fn scrgb_preserves_negative_and_hdr_float_samples() {
    let source = load_image(fixture("scrgb.jxl")).unwrap();
    assert!(matches!(source.image, DynamicImage::ImageRgba32F(_)));
    assert_eq!(source.image.dimensions(), (2, 1));
    let converted = source.convert(&linear()).unwrap();
    for (actual, expected) in converted
        .as_raw()
        .iter()
        .zip([-0.25, 2.0, 12.5, 1.0, 0.0, 0.5, 1.0, 1.0])
    {
        assert!((actual - expected).abs() < 0.001, "{actual} != {expected}");
    }
}

#[test]
fn container_preserves_float_alpha() {
    let source = load_image(fixture("scrgb-alpha.jxl")).unwrap();
    assert!(matches!(source.image, DynamicImage::ImageRgba32F(_)));
    let converted = source.convert(&linear()).unwrap();
    assert_eq!(converted.get_pixel(0, 0).0[3], 0.5);
    assert_eq!(converted.get_pixel(1, 0).0[3], 1.0);
}

#[test]
fn pq_and_hlg_use_rec2020_instead_of_srgb() {
    for (file, expected) in [("pq.jxl", 0.009_225), ("hlg.jxl", 0.050_701)] {
        let source = load_image(fixture(file)).unwrap();
        assert!(matches!(source.image, DynamicImage::ImageRgba32F(_)));
        let converted = source.convert(&linear()).unwrap();
        // Equal encoded RGB is neutral. PQ 0.5 is ~92 nits / 10000;
        // HLG 0.5 is ~(1/12)^1.2 display-linear at the default peak,
        // rather than sRGB's ~0.214.
        for actual in &converted.get_pixel(0, 0).0[..3] {
            assert!(
                (actual - expected).abs() < 0.0001,
                "{file}: {actual} != {expected}"
            );
        }
        // A saturated BT.2020 green lies outside the BT.709 gamut.
        assert!(converted.get_pixel(1, 0).0[0] < 0.0);
        assert!(converted.get_pixel(1, 0).0[1] > 1.0);
    }
}

#[test]
fn signature_sniffing_works_without_jxl_extension() {
    let path = std::env::temp_dir().join(format!("ssconv-{}-jxl.data", std::process::id()));
    std::fs::copy(fixture("hlg.jxl"), &path).unwrap();
    let loaded = load_image(&path);
    std::fs::remove_file(path).unwrap();
    assert_eq!(loaded.unwrap().image.dimensions(), (2, 1));
}

#[test]
fn truncated_jxl_returns_error() {
    let path = std::env::temp_dir().join(format!("ssconv-{}-truncated.JXL", std::process::id()));
    std::fs::write(&path, [0xff, 0x0a, 0]).unwrap();
    let loaded = load_image(&path);
    std::fs::remove_file(path).unwrap();
    assert!(loaded.is_err());
}
