use std::path::Path;
use std::sync::mpsc;

use color_eyre::eyre::{Result, WrapErr, eyre};
use image::Rgb;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
    gain_width: u32,
    gain_height: u32,
}

struct GpuBuffers {
    width: u32,
    height: u32,
    gain_width: u32,
    gain_height: u32,
    pixel_count: usize,
    gain_pixel_count: usize,
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    gain_buffer: wgpu::Buffer,
    gain_readback_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    gain_bind_group: wgpu::BindGroup,
}

impl GpuBuffers {
    fn new(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        width: u32,
        height: u32,
    ) -> Self {
        let pixel_count = (width * height) as usize;
        let gain_width = width.div_ceil(2);
        let gain_height = height.div_ceil(2);
        let gain_pixel_count = (gain_width * gain_height) as usize;
        let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("input-rgba"),
            size: (pixel_count * 4 * std::mem::size_of::<u16>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output-rgba"),
            size: (pixel_count * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output-readback"),
            size: (pixel_count * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let gain_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gain-map"),
            size: (gain_pixel_count * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let gain_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gain-map-readback"),
            size: (gain_pixel_count * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = Params {
            width,
            height,
            gain_width,
            gain_height,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = create_bind_group(
            device,
            bind_group_layout,
            &input_buffer,
            &output_buffer,
            &params_buffer,
        );
        let gain_bind_group = create_bind_group(
            device,
            bind_group_layout,
            &input_buffer,
            &gain_buffer,
            &params_buffer,
        );

        Self {
            width,
            height,
            gain_width,
            gain_height,
            pixel_count,
            gain_pixel_count,
            input_buffer,
            output_buffer,
            readback_buffer,
            gain_buffer,
            gain_readback_buffer,
            params_buffer,
            bind_group,
            gain_bind_group,
        }
    }
}

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    gain_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    buffers: Option<GpuBuffers>,
}

impl GpuContext {
    async fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .wrap_err("request GPU adapter")?;
        let adapter_info = adapter.get_info();
        println!(
            "Using GPU: {} (type: {:?}, backend: {:?})",
            adapter_info.name, adapter_info.device_type, adapter_info.backend
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ssconv-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::default(),
            })
            .await
            .wrap_err("request GPU device")?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rec2100-to-srgb"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bind_group_layout = create_bind_group_layout(&device);
        let pipeline = create_pipeline(&device, &bind_group_layout, &shader, "main");
        let gain_pipeline = create_pipeline(&device, &bind_group_layout, &shader, "gain_map");

        Ok(Self {
            device,
            queue,
            pipeline,
            gain_pipeline,
            bind_group_layout,
            buffers: None,
        })
    }

    fn convert(&mut self, input_rgba: &[u16], width: u32, height: u32) -> Result<ConvertOutput> {
        validate_input(input_rgba, width, height).wrap_err("validate input buffer")?;
        self.ensure_buffers(width, height);

        let device = &self.device;
        let queue = &self.queue;
        let pipeline = &self.pipeline;
        let gain_pipeline = &self.gain_pipeline;
        let buffers = self
            .buffers
            .as_mut()
            .ok_or_else(|| eyre!("GPU buffers not initialized"))?;
        upload_input(queue, buffers, input_rgba, width, height);
        let submission = dispatch_compute(
            device,
            queue,
            pipeline,
            gain_pipeline,
            buffers,
            width,
            height,
        );
        let (sdr_rgb, gain_map) =
            readback_output(device, buffers, submission).wrap_err("read back GPU output")?;

        Ok(ConvertOutput {
            sdr_rgb,
            gain_map,
            gain_width: buffers.gain_width,
            gain_height: buffers.gain_height,
        })
    }

    fn ensure_buffers(&mut self, width: u32, height: u32) {
        let rebuild = match self.buffers.as_ref() {
            Some(buffers) => buffers.width != width || buffers.height != height,
            None => true,
        };
        if rebuild {
            self.buffers = Some(GpuBuffers::new(
                &self.device,
                &self.bind_group_layout,
                width,
                height,
            ));
        }
    }
}

struct ConvertOutput {
    sdr_rgb: Vec<u8>,
    gain_map: Vec<u8>,
    gain_width: u32,
    gain_height: u32,
}

pub struct Converter {
    gpu: GpuContext,
}

impl Converter {
    /// Creates a GPU-backed converter instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU adapter or device cannot be initialized.
    pub fn new() -> Result<Self> {
        let gpu = pollster::block_on(GpuContext::new()).wrap_err("initialize GPU context")?;
        Ok(Self { gpu })
    }

    /// Converts a PNG image at `input_path` to a JPEG at `output_path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the input image cannot be opened, the GPU conversion
    /// fails, or the output image cannot be written.
    pub fn convert_png_to_jpeg(&mut self, input_path: &Path, output_path: &Path) -> Result<()> {
        let image = image::open(input_path)
            .wrap_err_with(|| format!("open input image {}", input_path.display()))?;
        let image_rgba = image.into_rgba16();
        let (width, height) = image_rgba.dimensions();
        let input_rgba = image_rgba.into_raw();

        let result = self
            .gpu
            .convert(&input_rgba, width, height)
            .wrap_err("convert image on GPU")?;
        println!(
            "Gain map: {}x{} ({} bytes)",
            result.gain_width,
            result.gain_height,
            result.gain_map.len()
        );

        let out_img: image::ImageBuffer<Rgb<u8>, Vec<u8>> =
            image::ImageBuffer::from_vec(width, height, result.sdr_rgb)
                .ok_or_else(|| eyre!("failed to create output image buffer"))?;
        out_img
            .save(output_path)
            .wrap_err_with(|| format!("save output image {}", output_path.display()))?;

        Ok(())
    }
}

/// Converts a PNG image at `input_path` to a JPEG at `output_path`.
///
/// # Errors
///
/// Returns an error if the converter cannot be created or the conversion
/// fails.
pub fn convert_png_to_jpeg(input_path: &Path, output_path: &Path) -> Result<()> {
    let mut converter = Converter::new().wrap_err("create converter")?;
    converter
        .convert_png_to_jpeg(input_path, output_path)
        .wrap_err("convert PNG to JPEG")?;
    Ok(())
}

fn upload_input(
    queue: &wgpu::Queue,
    buffers: &GpuBuffers,
    input_rgba: &[u16],
    width: u32,
    height: u32,
) {
    let params = Params {
        width,
        height,
        gain_width: buffers.gain_width,
        gain_height: buffers.gain_height,
    };
    queue.write_buffer(&buffers.params_buffer, 0, bytemuck::bytes_of(&params));
    queue.write_buffer(
        &buffers.input_buffer,
        0,
        bytemuck::cast_slice::<u16, u8>(input_rgba),
    );
}

fn dispatch_compute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    gain_pipeline: &wgpu::ComputePipeline,
    buffers: &GpuBuffers,
    width: u32,
    height: u32,
) -> wgpu::SubmissionIndex {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("command-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &buffers.bind_group, &[]);
        let workgroup_size = 16;
        let dispatch_x = width.div_ceil(workgroup_size);
        let dispatch_y = height.div_ceil(workgroup_size);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gain-map-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(gain_pipeline);
        pass.set_bind_group(0, &buffers.gain_bind_group, &[]);
        let workgroup_size = 16;
        let dispatch_x = buffers.gain_width.div_ceil(workgroup_size);
        let dispatch_y = buffers.gain_height.div_ceil(workgroup_size);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
    encoder.copy_buffer_to_buffer(
        &buffers.output_buffer,
        0,
        &buffers.readback_buffer,
        0,
        (buffers.pixel_count * std::mem::size_of::<u32>()) as u64,
    );
    encoder.copy_buffer_to_buffer(
        &buffers.gain_buffer,
        0,
        &buffers.gain_readback_buffer,
        0,
        (buffers.gain_pixel_count * std::mem::size_of::<u32>()) as u64,
    );
    queue.submit(std::iter::once(encoder.finish()))
}

fn readback_output(
    device: &wgpu::Device,
    buffers: &GpuBuffers,
    submission: wgpu::SubmissionIndex,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let output_slice = buffers.readback_buffer.slice(..);
    let gain_slice = buffers.gain_readback_buffer.slice(..);
    let (tx, rx) = mpsc::channel();
    let (gain_tx, gain_rx) = mpsc::channel();
    output_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    gain_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = gain_tx.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .wrap_err("wait for GPU readback")?;
    rx.recv()
        .wrap_err("receive GPU map callback")?
        .wrap_err("map readback buffer")?;
    gain_rx
        .recv()
        .wrap_err("receive gain map callback")?
        .wrap_err("map gain map buffer")?;

    let output_data = output_slice.get_mapped_range();
    let pixels_u32: &[u32] = bytemuck::cast_slice(&output_data);
    let sdr_rgb = pack_rgb(pixels_u32, buffers.pixel_count);
    drop(output_data);
    buffers.readback_buffer.unmap();

    let gain_data = gain_slice.get_mapped_range();
    let gain_u32: &[u32] = bytemuck::cast_slice(&gain_data);
    let gain_map = pack_gain(gain_u32, buffers.gain_pixel_count);
    drop(gain_data);
    buffers.gain_readback_buffer.unmap();

    Ok((sdr_rgb, gain_map))
}

fn validate_input(input_rgba: &[u16], width: u32, height: u32) -> Result<()> {
    let expected_len = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(4);
    if input_rgba.len() != expected_len {
        return Err(eyre!(
            "input buffer length {} does not match {}x{} RGBA",
            input_rgba.len(),
            width,
            height
        ));
    }
    Ok(())
}

fn pack_rgb(pixels_u32: &[u32], pixel_count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixel_count * 3);
    for packed in pixels_u32 {
        let r = (packed & 0xFF) as u8;
        let g = ((packed >> 8) & 0xFF) as u8;
        let b = ((packed >> 16) & 0xFF) as u8;
        out.extend_from_slice(&[r, g, b]);
    }
    out
}

fn pack_gain(gain_u32: &[u32], gain_pixel_count: usize) -> Vec<u8> {
    let mut gain_map = Vec::with_capacity(gain_pixel_count);
    for packed in gain_u32 {
        gain_map.push((packed & 0xFF) as u8);
    }
    gain_map
}

fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bind-group-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input_buffer: &wgpu::Buffer,
    output_buffer: &wgpu::Buffer,
    params_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind-group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    })
}

fn create_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline-layout"),
        bind_group_layouts: &[bind_group_layout],
        immediate_size: 0,
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute-pipeline"),
        layout: Some(&pipeline_layout),
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}
