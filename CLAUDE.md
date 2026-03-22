# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`ssconv` is a Rust workspace for image colorspace conversion. It reads images, detects their colorspace (priority: ICC profile > CICP > fallback sRGB), and converts between colorspaces (e.g., Rec. 2020).

## Workspace Structure

- **`ssconv-core`** — library crate with shared logic: error types (`snafu`), image I/O (`image` crate), and ICC profile parsing (`moxcms`)
- **`ssconv-cli`** — binary crate that uses `ssconv-core` and `moxcms` (avx512) for colorspace conversion

## Commands

```sh
# Build
cargo build

# Build release
cargo build --release

# Run CLI (requires samples/rec2020.png — excluded via .gitignore)
cargo run -p ssconv-cli

# Run tests
cargo test

# Run tests for a specific package
cargo test -p ssconv-core

# Lint (clippy configured at workspace level with pedantic warnings)
cargo clippy

# Check without building
cargo check
```

## Colorspace Detection Architecture

- Colorspace priority: ICC profile > CICP > sRGB fallback — stored as `ColorProfile` directly on `SourceImage`
- `moxcms` in both crates: `ssconv-core` (no features) and `ssconv-cli` (avx512, conversion)
- `ColorProfile::new_from_cicp(CicpProfile)` synthesizes a profile from CICP; `new_srgb()` for fallback
- `DynamicImage::color_space()` returns `image::metadata::Cicp` (primaries + transfer only; matrix/full_range are fixed)
- `image` crate uses `moxcms 0.7.4` internally; `ssconv-core` uses `moxcms 0.8.0` — both coexist in the build
- CICP enums in `image` and `moxcms` are both `#[repr(u8)]` per ITU-T H.273 — cast via `as u8` + `TryFrom<u8>`

## Key Dependencies

- `image` — image decoding/encoding
- `moxcms` — ICC profile parsing and colorspace conversion (with `avx512` feature enabled in CLI)
- `snafu` — ergonomic error handling with context selectors
- `zerocopy` — zero-copy type casting for pixel data

## Error Handling

Errors use `snafu` with the `Snafu` derive macro. Error variants live in `ssconv-core/src/error.rs`. Use `snafu::ResultExt` and the generated `*Snafu` context selectors (e.g., `IoSnafu`, `ImageDecodeSnafu`, `ColorspaceDetectionSnafu`).

## Version Control

This project uses **jj** (Jujutsu) for version control. Use `jj` commands instead of `git`:

```sh
# Show status
jj status

# Show log
jj log

# Describe the current change
jj describe -m "message"

# Create a new change
jj new

# Create a bookmark (branch)
jj bookmark create <name>

# Push to remote
jj git push
```

## Notes

- `samples/` is gitignored — test images must be added locally
- Clippy pedantic lints are enabled workspace-wide; address warnings before committing
