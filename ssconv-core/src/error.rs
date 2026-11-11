use snafu::prelude::*;

#[derive(Snafu, Debug)]
#[snafu(visibility(pub(crate)))]
pub enum SsconvError {
    #[snafu(display("IO error: {source}"))]
    Io { source: std::io::Error },
    #[snafu(display("Failed to decode image: {source}"))]
    ImageDecode { source: image::ImageError },
    #[snafu(display("Failed to encode image: {source}"))]
    ImageEncode { source: image::ImageError },
    #[snafu(display("Failed to detect colorspace: {source}"))]
    ColorspaceDetection { source: moxcms::CmsError },
    #[snafu(display("Failed to convert colorspace: {source}"))]
    ColorspaceConversion { source: moxcms::CmsError },
    #[snafu(display("JPEG XL decoding failed: {source}"))]
    JxlDecode { source: jxl::error::Error },
    #[snafu(display("Gain map encoding failed: {source}"))]
    GainMap { source: ultrahdr_rs::Error },
    #[snafu(display("Invalid image: {message}"))]
    InvalidImage { message: String },
}
