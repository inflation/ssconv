use std::path::Path;

use image::{GenericImageView, ImageReader, metadata::Cicp};
use snafu::ResultExt;

use crate::error::{ImageOpenSnafu, SsconvError};

pub fn load_image<P: AsRef<Path>>(path: P) -> Result<image::DynamicImage, SsconvError> {
    image::open(&path).context(ImageOpenSnafu)
}
