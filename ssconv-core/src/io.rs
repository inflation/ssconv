use std::path::Path;

use image::{DynamicImage, ImageDecoder, ImageReader};
use moxcms::ColorProfile;
use snafu::ResultExt;

use crate::error::{ColorspaceDetectionSnafu, ImageDecodeSnafu, IoSnafu, SsconvError};

/// A decoded image together with its detected colorspace.
///
/// Colorspace priority: ICC profile > CICP metadata > sRGB fallback.
pub struct SourceImage {
    pub image: DynamicImage,
    pub colorspace: ColorProfile,
}

/// Load an image from `path`, detecting its colorspace.
///
/// # Errors
///
/// Returns an error if the file cannot be opened, the format cannot be determined,
/// the image data is invalid, or an embedded ICC profile cannot be parsed.
pub fn load_image<P: AsRef<Path>>(path: P) -> Result<SourceImage, SsconvError> {
    let reader = ImageReader::open(&path)
        .and_then(ImageReader::with_guessed_format)
        .context(IoSnafu)?;

    let (icc_bytes, image) = reader
        .into_decoder()
        .and_then(|mut decoder| {
            let icc = decoder.icc_profile()?;
            DynamicImage::from_decoder(decoder).map(|img| (icc, img))
        })
        .context(ImageDecodeSnafu)?;

    let colorspace = detect_colorspace(icc_bytes, image.color_space())?;

    Ok(SourceImage { image, colorspace })
}

/// Detect the colorspace from ICC bytes and/or CICP metadata.
///
/// Priority: ICC profile > CICP metadata > sRGB fallback.
fn detect_colorspace(
    icc_bytes: Option<Vec<u8>>,
    cicp: image::metadata::Cicp,
) -> Result<ColorProfile, SsconvError> {
    use image::metadata::{CicpColorPrimaries, CicpTransferCharacteristics};

    if let Some(bytes) = icc_bytes {
        return ColorProfile::new_from_slice(&bytes).context(ColorspaceDetectionSnafu);
    }

    if matches!(cicp.primaries, CicpColorPrimaries::Unspecified)
        || matches!(cicp.transfer, CicpTransferCharacteristics::Unspecified)
        || (cicp.primaries == CicpColorPrimaries::SRgb
            && cicp.transfer == CicpTransferCharacteristics::SRgb)
    {
        return Ok(ColorProfile::new_srgb());
    }

    // Both enums are #[repr(u8)] and follow ITU-T H.273, so the u8 values match moxcms.
    let primaries = moxcms::CicpColorPrimaries::try_from(cicp.primaries as u8)
        .context(ColorspaceDetectionSnafu)?;
    let transfer = moxcms::TransferCharacteristics::try_from(cicp.transfer as u8)
        .context(ColorspaceDetectionSnafu)?;

    // DynamicImage only tracks primaries and transfer; matrix is always Identity for RGB.
    Ok(ColorProfile::new_from_cicp(moxcms::CicpProfile {
        color_primaries: primaries,
        transfer_characteristics: transfer,
        matrix_coefficients: moxcms::MatrixCoefficients::Identity,
        full_range: true,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_rec2020_sample() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../samples/rec2020.png");
        let loaded = load_image(path).expect("should load rec2020.png");
        assert!(
            loaded.colorspace.cicp.is_some() || loaded.colorspace.red_trc.is_some(),
            "rec2020.png should have a non-trivial color profile"
        );
    }
}
