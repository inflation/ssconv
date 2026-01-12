use std::path::Path;

use color_eyre::eyre::{Result, WrapErr};

fn main() -> Result<()> {
    color_eyre::install().wrap_err("install error reporting")?;

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <input.png> (Rec.2100 PQ)", args[0]);
        std::process::exit(2);
    }

    let input_path = Path::new(&args[1]);
    let output_path = input_path.with_extension(".jpg");

    ssconv::convert_png_to_jpeg(input_path, &output_path).wrap_err("convert input image")?;

    Ok(())
}
