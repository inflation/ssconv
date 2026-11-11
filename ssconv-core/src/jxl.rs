use std::io::{BufReader, Read, Seek};

use image::{DynamicImage, Rgba32FImage};
use jxl::api::{
    Event, JxlColorEncoding, JxlColorProfile, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
    JxlPixelFormat, JxlPrimaries, JxlTransferFunction, JxlWhitePoint,
};
use moxcms::{
    CicpColorPrimaries, CicpProfile, ColorProfile, MatrixCoefficients, TransferCharacteristics,
};
use snafu::ResultExt;
use zerocopy::IntoBytes;

use crate::{
    SourceImage, SsconvError,
    error::{ColorspaceDetectionSnafu, JxlDecodeSnafu},
};

fn invalid(message: &str) -> SsconvError {
    SsconvError::InvalidImage {
        message: message.into(),
    }
}

pub(crate) fn load(mut input: BufReader<impl Read + Seek>) -> Result<SourceImage, SsconvError> {
    let mut decoder = JxlDecoder::new(JxlDecoderOptions::default());
    if decoder
        .process(&mut input, None, None)
        .context(JxlDecodeSnafu)?
        != Event::BasicInfo
    {
        return Err(invalid("truncated JPEG XL header"));
    }
    let info = decoder
        .basic_info()
        .ok_or_else(|| invalid("missing JPEG XL header"))?
        .clone();
    decoder
        .set_pixel_format(JxlPixelFormat::rgba_f32(info.extra_channels.len()))
        .context(JxlDecodeSnafu)?;
    let profile = decoder
        .output_color_profile()
        .ok_or_else(|| invalid("missing JPEG XL color profile"))?;
    let colorspace = color_profile(profile)?;
    let (width, height) = (
        u32::try_from(info.size.0).map_err(|_| invalid("image too wide"))?,
        u32::try_from(info.size.1).map_err(|_| invalid("image too tall"))?,
    );
    let samples = info
        .size
        .0
        .checked_mul(info.size.1)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| invalid("JPEG XL dimensions overflow"))?;
    // Match image's usual 512 MiB allocation limit before allocating output.
    if samples > 512 * 1024 * 1024 / size_of::<f32>() || samples == 0 {
        return Err(invalid("JPEG XL exceeds output memory limit"));
    }
    let mut pixels = vec![0.0_f32; samples];
    let mut output = JxlOutputBuffer::new(
        pixels.as_mut_slice().as_mut_bytes(),
        height as usize,
        width as usize * 16,
    );
    loop {
        match decoder
            .process(&mut input, Some(std::slice::from_mut(&mut output)), None)
            .context(JxlDecodeSnafu)?
        {
            Event::FrameHeader => {}
            Event::FrameComplete { .. } => break,
            _ => return Err(invalid("truncated JPEG XL frame")),
        }
    }
    let image = Rgba32FImage::from_raw(width, height, pixels)
        .ok_or_else(|| invalid("invalid JPEG XL pixel buffer"))?;
    Ok(SourceImage {
        image: DynamicImage::ImageRgba32F(image),
        colorspace,
    })
}

fn color_profile(profile: &JxlColorProfile) -> Result<ColorProfile, SsconvError> {
    // Use the native encoding for standard D65 RGB. A synthesized HDR ICC can
    // contain a display-oriented CLUT; applying that would change luminance.
    if let JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
        white_point: JxlWhitePoint::D65,
        primaries,
        transfer_function,
        ..
    }) = profile
    {
        let primaries = match primaries {
            JxlPrimaries::SRGB => Some(CicpColorPrimaries::Bt709),
            JxlPrimaries::BT2100 => Some(CicpColorPrimaries::Bt2020),
            JxlPrimaries::P3 => Some(CicpColorPrimaries::Smpte432),
            JxlPrimaries::Chromaticities { .. } => None,
        };
        let transfer = match transfer_function {
            JxlTransferFunction::BT709 => Some(TransferCharacteristics::Bt709),
            JxlTransferFunction::Linear => Some(TransferCharacteristics::Linear),
            JxlTransferFunction::SRGB => Some(TransferCharacteristics::Srgb),
            JxlTransferFunction::PQ => Some(TransferCharacteristics::Smpte2084),
            JxlTransferFunction::HLG => Some(TransferCharacteristics::Hlg),
            _ => None,
        };
        if let (Some(color_primaries), Some(transfer_characteristics)) = (primaries, transfer) {
            return Ok(ColorProfile::new_from_cicp(CicpProfile {
                color_primaries,
                transfer_characteristics,
                matrix_coefficients: MatrixCoefficients::Identity,
                full_range: true,
            }));
        }
    }
    let icc = profile
        .try_as_icc()
        .ok_or_else(|| invalid("JPEG XL color encoding has no usable ICC profile"))?;
    ColorProfile::new_from_slice(icc.as_ref()).context(ColorspaceDetectionSnafu)
}
