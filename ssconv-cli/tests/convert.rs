use std::{path::Path, process::Command};

#[test]
fn jxl_to_exr_preserves_hdr_and_alpha() {
    let input =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../ssconv-core/tests/fixtures/scrgb-alpha.jxl");
    let output = std::env::temp_dir().join(format!("ssconv-cli-{}.exr", std::process::id()));
    let result = Command::new(env!("CARGO_BIN_EXE_ssconv-cli"))
        .args([input.as_os_str(), output.as_os_str()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let image = image::open(&output).unwrap().to_rgba32f();
    std::fs::remove_file(output).unwrap();
    assert_eq!(image.dimensions(), (2, 1));
    for (actual, expected) in image
        .get_pixel(0, 0)
        .0
        .into_iter()
        .zip([-0.25, 2.0, 12.5, 0.5])
    {
        assert!((actual - expected).abs() < 0.001);
    }
}

#[test]
fn invalid_arguments_fail_with_usage() {
    let output = Command::new(env!("CARGO_BIN_EXE_ssconv-cli"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Usage:"));
}

#[test]
fn jxl_to_gain_map_jpeg_contains_two_decodable_images() {
    let input = Path::new(env!("CARGO_MANIFEST_DIR")).join("../ssconv-core/tests/fixtures/pq.jxl");
    let output = std::env::temp_dir().join(format!("ssconv-cli-{}.jpg", std::process::id()));
    let result = Command::new(env!("CARGO_BIN_EXE_ssconv-cli"))
        .args([input.as_os_str(), output.as_os_str()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let bytes = std::fs::read(&output).unwrap();
    std::fs::remove_file(output).unwrap();
    let decoder = ultrahdr_rs::Decoder::new(&bytes).unwrap();
    assert!(decoder.is_ultrahdr());
    for jpeg in [
        decoder.primary_jpeg().unwrap(),
        decoder.gainmap_jpeg().unwrap(),
    ] {
        let image = image::load_from_memory(jpeg).unwrap();
        assert_eq!((image.width(), image.height()), (2, 1));
    }
}
