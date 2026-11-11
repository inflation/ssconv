fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let (Some(input), Some(output), None) = (args.next(), args.next(), args.next()) else {
        return Err("Usage: ssconv-cli <input image> <output.exr>".into());
    };
    let output = std::path::Path::new(&output);
    if !output
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("exr"))
    {
        return Err("Output must be .exr to preserve linear HDR samples".into());
    }
    let source = ssconv_core::load_image(input)?;
    let linear = moxcms::ColorProfile::new_from_cicp(moxcms::CicpProfile {
        color_primaries: moxcms::CicpColorPrimaries::Bt709,
        transfer_characteristics: moxcms::TransferCharacteristics::Linear,
        matrix_coefficients: moxcms::MatrixCoefficients::Identity,
        full_range: true,
    });
    source
        .convert(&linear)?
        .save_with_format(output, image::ImageFormat::OpenExr)?;
    Ok(())
}
