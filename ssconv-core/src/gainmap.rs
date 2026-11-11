use image::{ExtendedColorType, RgbImage, codecs::jpeg::JpegEncoder};
use moxcms::{
    CicpColorPrimaries, CicpProfile, ColorProfile, MatrixCoefficients, TransferCharacteristics,
};
use snafu::ResultExt;
use ultrahdr_rs::{ColorGamut, GainMapMetadata, encode_ultrahdr};

use crate::{
    SourceImage, SsconvError,
    error::{GainMapSnafu, ImageDecodeSnafu, ImageEncodeSnafu},
};

/// Settings for an SDR-compatible Ultra HDR JPEG with an embedded RGB gain map.
#[derive(Debug, Clone, Copy)]
pub struct GainMapOptions {
    /// Override cd/m² represented by 1.0 after linearization. Automatic defaults:
    /// scRGB/linear = 80, PQ = 10000, HLG = 1000, other profiles = 203.
    pub reference_white_nits: Option<f32>,
    /// JPEG quality for both base and gain map, from 1 to 100.
    pub quality: u8,
}

impl Default for GainMapOptions {
    fn default() -> Self {
        Self {
            reference_white_nits: None,
            quality: 95,
        }
    }
}

impl SourceImage {
    /// Encode a tone-mapped SDR JPEG plus an embedded HDR gain map (XMP and ISO
    /// 21496-1 metadata). Alpha is composited over black and negative RGB is
    /// clipped after conversion to BT.709. SDR white is 203 cd/m².
    ///
    /// # Errors
    /// Returns an error for invalid options, non-finite pixels, color conversion,
    /// or JPEG / gain-map encoding failures.
    pub fn encode_gain_map(&self, options: GainMapOptions) -> Result<Vec<u8>, SsconvError> {
        let reference = options.reference_white_nits.unwrap_or_else(|| {
            match self
                .colorspace
                .cicp
                .as_ref()
                .map(|c| c.transfer_characteristics)
            {
                Some(TransferCharacteristics::Linear) => 80.0,
                Some(TransferCharacteristics::Smpte2084) => 10_000.0,
                Some(TransferCharacteristics::Hlg) => 1_000.0,
                _ => 203.0,
            }
        });
        if !reference.is_finite() || reference <= 0.0 || !(1..=100).contains(&options.quality) {
            return Err(invalid(
                "reference white must be finite and positive; quality must be 1..=100",
            ));
        }
        let linear = ColorProfile::new_from_cicp(CicpProfile {
            color_primaries: CicpColorPrimaries::Bt709,
            transfer_characteristics: TransferCharacteristics::Linear,
            matrix_coefficients: MatrixCoefficients::Identity,
            full_range: true,
        });
        let converted = self.convert(&linear)?;
        let (width, height) = converted.dimensions();
        if width == 0 || height == 0 {
            return Err(invalid("cannot encode an empty image"));
        }
        let mut hdr = Vec::with_capacity(converted.as_raw().len() / 4);
        let mut base = RgbImage::new(width, height);
        let mut peak = 1.0_f64;
        for (pixel, sdr) in converted.pixels().zip(base.pixels_mut()) {
            if pixel.0.iter().any(|v| !v.is_finite()) {
                return Err(invalid("HDR pixels must be finite"));
            }
            let alpha = f64::from(pixel.0[3].clamp(0.0, 1.0));
            let rgb: [f64; 3] = std::array::from_fn(|c| {
                f64::from(pixel.0[c].max(0.0)) * alpha * f64::from(reference) / 203.0
            });
            let maximum = rgb.into_iter().fold(0.0_f64, f64::max);
            peak = peak.max(maximum);
            // Hue-preserving Reinhard shoulder; highlights live in the gain map.
            sdr.0 = rgb.map(|v| byte(linear_to_srgb(v / (1.0 + maximum))));
            hdr.push(rgb);
        }
        let base_jpeg = jpeg(&base, options.quality)?;
        // Compute against the actual lossy base that viewers will reconstruct.
        let decoded_base = image::load_from_memory(&base_jpeg)
            .context(ImageDecodeSnafu)?
            .to_rgb8();
        let offset = 1.0 / 64.0;
        let mut gains = Vec::with_capacity(hdr.len());
        let (mut minimum, mut maximum) = (0.0_f64, 0.0_f64);
        for (rgb, sdr) in hdr.iter().zip(decoded_base.pixels()) {
            let gain: [f64; 3] = std::array::from_fn(|c| {
                let base = srgb_to_linear(f64::from(sdr.0[c]) / 255.0);
                ((rgb[c] + offset) / (base + offset)).log2()
            });
            for value in gain {
                minimum = minimum.min(value);
                maximum = maximum.max(value);
            }
            gains.push(gain);
        }
        // Keep flat black and SDR-only inputs well-defined (no zero divisor).
        maximum = maximum.max(minimum + 0.000_001);
        let mut gainmap = RgbImage::new(width, height);
        for (gain, pixel) in gains.iter().zip(gainmap.pixels_mut()) {
            pixel.0 = gain.map(|v| byte((v - minimum) / (maximum - minimum)));
        }
        let mut metadata = GainMapMetadata::default();
        metadata.gain_map_min = [minimum; 3];
        metadata.gain_map_max = [maximum; 3];
        metadata.alternate_hdr_headroom = peak.log2().max(0.000_001);
        encode_ultrahdr(
            &base_jpeg,
            &jpeg(&gainmap, options.quality)?,
            &metadata,
            ColorGamut::Bt709,
        )
        .context(GainMapSnafu)
    }
}

fn invalid(message: &str) -> SsconvError {
    SsconvError::InvalidImage {
        message: message.into(),
    }
}

fn jpeg(image: &RgbImage, quality: u8) -> Result<Vec<u8>, SsconvError> {
    let mut bytes = Vec::new();
    JpegEncoder::new_with_quality(&mut bytes, quality)
        .encode(
            image.as_raw(),
            image.width(),
            image.height(),
            ExtendedColorType::Rgb8,
        )
        .context(ImageEncodeSnafu)?;
    Ok(bytes)
}

fn linear_to_srgb(v: f64) -> f64 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn byte(value: f64) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}
