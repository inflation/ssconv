use ssconv_core::{GainMapOptions, load_image};

const USAGE: &str =
    "Usage: ssconv-cli <input image> <output.jpg|output.exr> [--reference-white-nits N]";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    let mut options = GainMapOptions::default();
    let (input, output) = match args.as_slice() {
        [input, output] => (input, output),
        [input, output, flag, value] if flag == "--reference-white-nits" => {
            options.reference_white_nits =
                Some(value.to_str().ok_or("Invalid reference white")?.parse()?);
            (input, output)
        }
        _ => return Err(USAGE.into()),
    };
    let output = std::path::Path::new(output);
    let extension = output
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default();
    let jpeg = extension.eq_ignore_ascii_case("jpg") || extension.eq_ignore_ascii_case("jpeg");
    let exr = extension.eq_ignore_ascii_case("exr");
    if !jpeg && !exr {
        return Err(
            "Output must be .jpg/.jpeg (ISO Gain Map gain map) or .exr (linear HDR)".into(),
        );
    }
    if exr && options.reference_white_nits.is_some() {
        return Err("--reference-white-nits applies to gain-map JPEG output only".into());
    }
    let source = load_image(input)?;
    if jpeg {
        std::fs::write(output, source.encode_gain_map(options)?)?;
    } else {
        let linear = moxcms::ColorProfile::new_from_cicp(moxcms::CicpProfile {
            color_primaries: moxcms::CicpColorPrimaries::Bt709,
            transfer_characteristics: moxcms::TransferCharacteristics::Linear,
            matrix_coefficients: moxcms::MatrixCoefficients::Identity,
            full_range: true,
        });
        source
            .convert(&linear)?
            .save_with_format(output, image::ImageFormat::OpenExr)?;
    }
    Ok(())
}
