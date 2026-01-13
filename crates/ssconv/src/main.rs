use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, WrapErr};

fn main() -> Result<()> {
    color_eyre::install().wrap_err("install error reporting")?;

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        let exe = std::env::args()
            .next()
            .unwrap_or_else(|| "ssconv".to_string());
        eprintln!(
            "Usage: {exe} <input.png> [output.jpg]\n       {exe} <input1.png> <output1.jpg> ...\n       {exe} <input1.png> <input2.png> ..."
        );
        std::process::exit(2);
    }

    let jobs = parse_jobs(&args);
    let mut converter = ssconv::Converter::new().wrap_err("create converter")?;
    for (input_path, output_path) in jobs {
        converter
            .convert_png_to_jpeg(&input_path, &output_path)
            .wrap_err_with(|| {
                format!(
                    "convert {} -> {}",
                    input_path.display(),
                    output_path.display()
                )
            })?;
    }

    Ok(())
}

fn parse_jobs(args: &[String]) -> Vec<(PathBuf, PathBuf)> {
    if args.len() == 2 && is_jpeg_path(&args[1]) {
        return vec![(PathBuf::from(&args[0]), PathBuf::from(&args[1]))];
    }

    if args.len() > 1 && args.len() % 2 == 0 && args.chunks(2).all(|pair| is_jpeg_path(&pair[1])) {
        return args
            .chunks(2)
            .map(|pair| (PathBuf::from(&pair[0]), PathBuf::from(&pair[1])))
            .collect();
    }

    args.iter()
        .map(|input| {
            let input_path = PathBuf::from(input);
            let output_path = input_path.with_extension("jpg");
            (input_path, output_path)
        })
        .collect()
}

fn is_jpeg_path(path: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg"))
        .unwrap_or(false)
}
