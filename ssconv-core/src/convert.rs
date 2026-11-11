use image::Rgba32FImage;
use moxcms::{ColorProfile, Layout, TransformOptions};
use snafu::ResultExt;

use crate::{SourceImage, SsconvError, error::ColorspaceConversionSnafu};

impl SourceImage {
    /// Convert pixels to a destination profile without quantizing HDR samples.
    ///
    /// Floating-point values outside 0..=1 and alpha are preserved. This performs
    /// color conversion only, without tone mapping or luminance rescaling.
    ///
    /// # Errors
    /// Returns an error if the color transform cannot be constructed or applied.
    pub fn convert(&self, destination: &ColorProfile) -> Result<Rgba32FImage, SsconvError> {
        let source = self.image.to_rgba32f();
        let mut output = Rgba32FImage::new(source.width(), source.height());
        let transform = self
            .colorspace
            .create_transform_f32(
                Layout::Rgba,
                destination,
                Layout::Rgba,
                TransformOptions {
                    allow_extended_range_rgb_xyz: true,
                    prefer_fixed_point: false,
                    ..TransformOptions::default()
                },
            )
            .context(ColorspaceConversionSnafu)?;
        transform
            .transform(source.as_raw(), output.as_mut())
            .context(ColorspaceConversionSnafu)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;
    use moxcms::{CicpColorPrimaries, CicpProfile, MatrixCoefficients, TransferCharacteristics};

    #[test]
    fn linear_conversion_preserves_hdr_and_alpha() {
        let profile = ColorProfile::new_from_cicp(CicpProfile {
            color_primaries: CicpColorPrimaries::Bt709,
            transfer_characteristics: TransferCharacteristics::Linear,
            matrix_coefficients: MatrixCoefficients::Identity,
            full_range: true,
        });
        let source = SourceImage {
            image: DynamicImage::ImageRgba32F(
                Rgba32FImage::from_raw(1, 1, vec![-0.25, 2.0, 12.5, 0.5]).unwrap(),
            ),
            colorspace: profile.clone(),
        };
        let converted = source.convert(&profile).unwrap();
        for (actual, expected) in converted.as_raw().iter().zip([-0.25, 2.0, 12.5, 0.5]) {
            assert!((actual - expected).abs() < 0.001, "{actual} != {expected}");
        }
    }
}
