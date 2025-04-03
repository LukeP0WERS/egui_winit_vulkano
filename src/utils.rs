// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

#[cfg(feature = "image")]
use image::RgbaImage;
use vulkano::{
    buffer::{AllocateBufferError, BufferCreateInfo, BufferUsage},
    image::{ImageCreateInfo, ImageType, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    Validated, VulkanError,
};
use vulkano_taskgraph::{command_buffer::CopyBufferToImageInfo, graph::ExecuteError, resource::{AccessTypes, ImageLayoutType}};
use vulkano_util::resource_access::{GlobalImageCreateError, GlobalImageCreateInfo, GlobalImageTracker, ResourceAccess};

#[derive(Debug)]
pub enum ImageCreationError {
    Vulkan(Validated<VulkanError>),
    AllocateBuffer(Validated<AllocateBufferError>),
    CreateGlobalImage(Validated<GlobalImageCreateError>),
    ExecuteError(ExecuteError),
}

pub fn immutable_texture_from_bytes<W: 'static + ?Sized>(
    access: &ResourceAccess,
    byte_data: &[u8],
    dimensions: [u32; 2],
    format: vulkano::format::Format,
) -> Result<GlobalImageTracker, ImageCreationError> {
    let texture_data_buffer = access.buffer_from_slice(
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        byte_data,
    ).map_err(ImageCreationError::AllocateBuffer)?;

    let texture_id = access.create_global_image::<W>(
        None,
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format,
            extent: [dimensions[0], dimensions[1], 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
        GlobalImageCreateInfo::sampled(),
    ).map_err(ImageCreationError::CreateGlobalImage)?;

    unsafe {
        vulkano_taskgraph::execute(
            &access.queue(),
            &access.resources(),
            access.flight_id(),
            |builder, _task_context| {
                builder.copy_buffer_to_image(&CopyBufferToImageInfo {
                    src_buffer: texture_data_buffer,
                    dst_image: texture_id.id(),
                    ..Default::default()
                }).unwrap();
                
                Ok(())
            },
            [],
            [(texture_data_buffer, AccessTypes::COPY_TRANSFER_READ)],
            [(texture_id.id(), AccessTypes::COPY_TRANSFER_WRITE, ImageLayoutType::Optimal)],
        )
    }.map_err(ImageCreationError::ExecuteError)?;

    let resources = access.resources();
    let flight = resources.flight(access.flight_id()).unwrap();
    flight.wait(None).unwrap();

    Ok(texture_id)
}

#[cfg(feature = "image")]
pub fn immutable_texture_from_file<W: 'static + ?Sized>(
    access: &ResourceAccess,
    file_bytes: &[u8],
    format: vulkano::format::Format,
) -> Result<GlobalImageTracker, ImageCreationError> {
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
    immutable_texture_from_bytes::<W>(access, &rgba, [dimensions.0, dimensions.1], format)
}