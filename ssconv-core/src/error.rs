use snafu::prelude::*;

#[derive(Snafu, Debug)]
#[snafu(visibility(pub(crate)))]
pub enum SsconvError {
    #[snafu(display("IO error: {source}"))]
    Io { source: std::io::Error },
    #[snafu(display("Failed to decode image: {source}"))]
    ImageDecode { source: image::ImageError },
    #[snafu(display("Failed to detect colorspace: {source}"))]
    ColorspaceDetection { source: moxcms::CmsError },
}
