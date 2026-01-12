use std::fs;
use std::path::Path;
use std::process::Command;

use image::GenericImageView;

#[test]
fn converts_sample_png_to_jpeg() {
    let input = Path::new("samples").join("sample.png");
    let output_dir = Path::new("target").join("test-output");
    let output = output_dir.join("sample_srgb.jpg");

    fs::create_dir_all(&output_dir).expect("create output directory");

    let status = Command::new(env!("CARGO_BIN_EXE_ssconv"))
        .arg(&input)
        .arg(&output)
        .status()
        .expect("run converter");
    assert!(status.success(), "converter exited with failure");

    let metadata = fs::metadata(&output).expect("output file metadata");
    assert!(metadata.len() > 0, "output file is empty");

    let in_img = image::open(&input).expect("open input image");
    let out_img = image::open(&output).expect("open output image");
    assert_eq!(
        in_img.dimensions(),
        out_img.dimensions(),
        "output size should match input size"
    );
}
