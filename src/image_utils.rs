// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

#[cfg(feature = "image")]
use image::RgbaImage;
use vulkano::{
    buffer::{AllocateBufferError, Buffer, BufferCreateInfo, BufferUsage},
    device::Queue,
    image::{
        view::{ImageView, ImageViewCreateInfo},
        AllocateImageError, Image, ImageCreateInfo, ImageType, ImageUsage,
    },
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryAllocator, MemoryTypeFilter},
    Validated, VulkanError,
};
use vulkano_taskgraph::{
    command_buffer::CopyBufferToImageInfo,
    graph::ExecuteError,
    resource::{AccessTypes, Flight, HostAccessType, ImageLayoutType, Resources},
    Id,
};

/// Egui uses a premultiplied color space which allows encoding of additive
/// colors (see documentation of [`egui::Color32`] for more details).
/// - [`AlphaMode::Straight`] is what you want for most semi-transparent images. This will
/// premultiply the image data you provide which can be an expensive operation for very large images.
/// To improve performance you can do the conversion manually (e.g. multithreading) and set this to [`AlphaMode::Premultiplied`] (hence the name).
/// - [`AlphaMode::Premultiplied`] encodes additive transparency, where as pixels approach an alpha of zero they become additive.
///
#[derive(Debug, Clone, Copy, Default)]
pub enum AlphaMode {
    #[default]
    Straight,
    Premultiplied,
}

/// Multiplies a byte array of (presumably) pixels with the alpha channel.
pub fn premultiply_rgba(pixels: &mut [u8]) {
    for px in pixels.chunks_exact_mut(4) {
        let a = px[3] as u32;
        px[0] = ((px[0] as u32 * a + 127) / 255) as u8;
        px[1] = ((px[1] as u32 * a + 127) / 255) as u8;
        px[2] = ((px[2] as u32 * a + 127) / 255) as u8;
    }
}

#[derive(Debug)]
pub enum ImageCreationError {
    Vulkan(Validated<VulkanError>),
    AllocateBuffer(Validated<AllocateBufferError>),
    AllocateImage(Validated<AllocateImageError>),
    ExecuteError(ExecuteError),
}

/// Creates an image resource and uploads data to it from raw byte data.
///
/// # Safety
///
/// - The user must ensure the queue supports transfer operations.
pub unsafe fn immutable_texture_from_bytes<W: 'static + ?Sized>(
    queue: &Arc<Queue>,
    resources: &Arc<Resources>,
    flight_id: Id<Flight>,
    staging_allocator: Option<&Arc<dyn MemoryAllocator>>,
    byte_data: &[u8],
    dimensions: [u32; 2],
    alpha_mode: AlphaMode,
    format: vulkano::format::Format,
) -> Result<(Id<Image>, Arc<ImageView>), ImageCreationError> {
    let premultiplied = if let AlphaMode::Straight = alpha_mode {
        let mut v = byte_data.to_vec();
        premultiply_rgba(&mut v);
        Some(v)
    } else {
        None
    };
    let byte_data: &[u8] = premultiplied.as_deref().unwrap_or(byte_data);

    let texture_data_buffer = {
        let buffer_create_info =
            BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() };
        let allocation_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        };
        let device_layout = DeviceLayout::new_unsized::<[u8]>(byte_data.len() as u64).unwrap();

        if let Some(staging_allocator) = staging_allocator {
            let texture_buffer = Buffer::new(
                staging_allocator,
                &buffer_create_info,
                &allocation_info,
                device_layout,
            )
            .map_err(ImageCreationError::AllocateBuffer)?;
            resources.add_buffer(texture_buffer)
        } else {
            resources
                .create_buffer(&buffer_create_info, &allocation_info, device_layout)
                .map_err(ImageCreationError::AllocateBuffer)?
        }
    };

    let texture_id = resources
        .create_image(
            &ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format,
                extent: [dimensions[0], dimensions[1], 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            &AllocationCreateInfo::default(),
        )
        .map_err(ImageCreationError::AllocateImage)?;

    let image = resources.image(texture_id).unwrap().image().clone();
    let image_view = ImageView::new(&image, &ImageViewCreateInfo::from_image(&image))
        .map_err(ImageCreationError::Vulkan)?;

    let flight = resources.flight(flight_id).unwrap();
    flight.wait(None).unwrap();

    // SAFETY:
    // * The resources are not being accessed by any other task graph execution.
    // * The user must ensure the queue supports image transfer operations.
    unsafe {
        vulkano_taskgraph::execute(
            &queue.clone(),
            &resources.clone(),
            flight_id,
            |builder, task_context| {
                let write_buffer = task_context.write_buffer::<[u8]>(texture_data_buffer, ..)?;
                write_buffer.copy_from_slice(byte_data);

                builder
                    .copy_buffer_to_image(&CopyBufferToImageInfo {
                        src_buffer: texture_data_buffer,
                        dst_image: texture_id,
                        ..Default::default()
                    })
                    .unwrap();

                Ok(())
            },
            [(texture_data_buffer, HostAccessType::Write)],
            [(texture_data_buffer, AccessTypes::COPY_TRANSFER_READ)],
            [(texture_id, AccessTypes::COPY_TRANSFER_WRITE, ImageLayoutType::Optimal)],
        )
    }
    .map_err(ImageCreationError::ExecuteError)?;

    // Queue destruction of staging buffer
    let mut batch = resources.create_deferred_batch();
    batch.destroy_buffer(texture_data_buffer);

    // SAFETY: The buffer isn't used by any other flights.
    unsafe {
        batch.enqueue_with_flights([flight_id]);
    }

    if staging_allocator.is_some() {
        // Wait to ensure the staging allocator is reset.
        flight.wait(None).unwrap();
    }

    Ok((texture_id, image_view))
}

#[cfg(feature = "image")]
/// Creates an image resource and uploads data to it from the file bytes.
///
/// # Safety
///
/// - The user must ensure the queue supports transfer operations.
pub unsafe fn immutable_texture_from_file<W: 'static + ?Sized>(
    queue: &Arc<Queue>,
    resources: &Arc<Resources>,
    flight_id: Id<Flight>,
    staging_allocator: Option<&Arc<dyn MemoryAllocator>>,
    file_bytes: &[u8],
    alpha_mode: AlphaMode,
    format: vulkano::format::Format,
) -> Result<(Id<Image>, Arc<ImageView>), ImageCreationError> {
    use image::GenericImageView;

    let img = image::load_from_memory(file_bytes).expect("Failed to load image from bytes");
    let rgba = if let Some(rgba) = img.as_rgba8() {
        rgba.to_owned().to_vec()
    } else {
        // Convert rgb to rgba
        let rgb = img.as_rgb8().unwrap().to_owned();
        let mut raw_data = vec![];
        for val in rgb.chunks(3) {
            raw_data.push(val[0]);
            raw_data.push(val[1]);
            raw_data.push(val[2]);
            raw_data.push(255);
        }
        let new_rgba = RgbaImage::from_raw(rgb.width(), rgb.height(), raw_data).unwrap();
        new_rgba.to_vec()
    };
    let dimensions = img.dimensions();

    immutable_texture_from_bytes::<W>(
        queue,
        resources,
        flight_id,
        staging_allocator,
        &rgba,
        [dimensions.0, dimensions.1],
        alpha_mode,
        format,
    )
}
