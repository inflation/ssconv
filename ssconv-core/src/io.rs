use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader};
use moxcms::{CicpColorPrimaries, ColorProfile, TransferCharacteristics};
use snafu::ResultExt;

use crate::error::{ColorspaceDetectionSnafu, ImageDecodeSnafu, IoSnafu, SsconvError};

/// A decoded image together with its detected colorspace.
///
/// Colorspace priority: ICC profile > CICP metadata > sRGB fallback.
#[derive(Debug)]
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
    // image 0.25 has no built-in JPEG XL decoder. Sniff both codestream and
    // container signatures, so renamed screenshots work as well as .jxl files.
    let mut file = BufReader::new(File::open(&path).context(IoSnafu)?);
    let signature = file.fill_buf().context(IoSnafu)?;
    if signature.starts_with(&[0xff, 0x0a])
        || signature.starts_with(b"\0\0\0\x0cJXL \r\n\x87\n")
        || path
            .as_ref()
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("jxl"))
    {
        let mut decoder =
            jxl_oxide::integration::JxlDecoder::new(file).context(ImageDecodeSnafu)?;
        // This is the profile of the *rendered* pixels, including JXL's native
        // color encoding when no ICC was embedded. Never assume sRGB here.
        let icc = decoder.icc_profile().context(ImageDecodeSnafu)?;
        let image = DynamicImage::from_decoder(decoder).context(ImageDecodeSnafu)?;
        let colorspace = detect_colorspace(icc, image.color_space(), None)?;
        return Ok(SourceImage { image, colorspace });
    }

    let reader = ImageReader::open(&path)
        .and_then(ImageReader::with_guessed_format)
        .context(IoSnafu)?;

    let format = reader.format();

    let (icc_bytes, image) = reader
        .into_decoder()
        .and_then(|mut decoder| {
            let icc = decoder.icc_profile()?;
            DynamicImage::from_decoder(decoder).map(|img| (icc, img))
        })
        .context(ImageDecodeSnafu)?;

    let colorspace = detect_colorspace(icc_bytes, image.color_space(), format)?;

    Ok(SourceImage { image, colorspace })
}

/// Detect the colorspace from ICC bytes and/or CICP metadata.
///
/// Priority: ICC profile > CICP metadata > sRGB fallback.
fn detect_colorspace(
    icc_bytes: Option<Vec<u8>>,
    cicp: image::metadata::Cicp,
    format: Option<ImageFormat>,
) -> Result<ColorProfile, SsconvError> {
    if let Some(bytes) = icc_bytes {
        return ColorProfile::new_from_slice(&bytes).context(ColorspaceDetectionSnafu);
    }

    // HDR and EXR files store scene-referred linear light with BT.709 primaries.
    if matches!(format, Some(ImageFormat::Hdr | ImageFormat::OpenExr)) {
        return Ok(ColorProfile::new_from_cicp(moxcms::CicpProfile {
            color_primaries: CicpColorPrimaries::Bt709,
            transfer_characteristics: TransferCharacteristics::Linear,
            matrix_coefficients: moxcms::MatrixCoefficients::Identity,
            full_range: true,
        }));
    }

    // Both enums are #[repr(u8)] and follow ITU-T H.273, so the u8 values match moxcms.
    let (primaries, transfer) = CicpColorPrimaries::try_from(cicp.primaries as u8)
        .and_then(|p| TransferCharacteristics::try_from(cicp.transfer as u8).map(|t| (p, t)))
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
    fn detects_linear_hdr_and_exr() {
        for format in [ImageFormat::Hdr, ImageFormat::OpenExr] {
            let profile =
                detect_colorspace(None, image::metadata::Cicp::SRGB, Some(format)).unwrap();
            assert_eq!(
                profile.cicp.unwrap().transfer_characteristics,
                TransferCharacteristics::Linear
            );
        }
    }

    #[test]
    fn icc_takes_priority_and_invalid_icc_is_reported() {
        assert!(
            detect_colorspace(
                Some(vec![0; 4]),
                image::metadata::Cicp::SRGB,
                Some(ImageFormat::Hdr)
            )
            .is_err()
        );
        let profile = detect_colorspace(None, image::metadata::Cicp::SRGB, None).unwrap();
        assert_eq!(
            profile.cicp.unwrap().transfer_characteristics,
            TransferCharacteristics::Srgb
        );
    }
}
