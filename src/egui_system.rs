// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{fmt::Debug, marker::PhantomData, ops::Range, sync::Arc};

use egui::{ahash::AHashMap, epaint::Primitive, ClippedPrimitive, Rect, TexturesDelta};
use vulkano::{
    buffer::{
        AllocateBufferError, Buffer, BufferContents, BufferCreateInfo, BufferUsage, IndexType,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator,
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
        DescriptorImageInfo, DescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue, QueueFlags},
    format::{Format, NumericFormat},
    image::{
        sampler::{
            ComponentMapping, ComponentSwizzle, Filter, Sampler, SamplerAddressMode,
            SamplerCreateInfo, SamplerMipmapMode,
        },
        view::{ImageView, ImageViewCreateInfo},
        AllocateImageError, Image, ImageAspects, ImageCreateInfo, ImageLayout,
        ImageSubresourceLayers, ImageType, ImageUsage, SampleCount,
    },
    instance::debug::DebugUtilsLabel,
    memory::{
        allocator::{AllocationCreateInfo, DeviceLayout, MemoryAllocator, MemoryTypeFilter},
        DeviceAlignment,
    },
    pipeline::{
        graphics::{
            color_blend::{
                AttachmentBlend, BlendFactor, ColorBlendAttachmentState, ColorBlendState,
            },
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineSubpassType,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Scissor, Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, Subpass},
    shader::ShaderStages,
    swapchain::{Surface, Swapchain},
    Validated, VulkanError,
};
use vulkano_taskgraph::{
    command_buffer::{BufferImageCopy, CopyBufferToImageInfo, RecordingCommandBuffer},
    descriptor_set::{SampledImageId, SamplerId},
    graph::{AttachmentInfo, ExecutableTaskGraph, NodeId, TaskGraph},
    resource::{AccessTypes, Flight, HostAccessType, ImageLayoutType, Resources},
    Id, QueueFamilyType, Task, TaskContext, TaskError, TaskResult,
};
use winit::{event_loop::ActiveEventLoop, raw_window_handle::HandleError, window::Window};

const MAX_QUADS: usize = 0x10000;
const VERTICES_PER_QUAD: usize = 4;
const INDICES_PER_QUAD: usize = 6;
const MAX_VERTICES: usize = MAX_QUADS * VERTICES_PER_QUAD;
const MAX_INDICES: usize = MAX_QUADS * INDICES_PER_QUAD;

use egui::epaint::Vertex as EpaintVertex;

#[cfg(feature = "image")]
use crate::image_utils::immutable_texture_from_file;
use crate::{image_utils::ImageCreationError, immutable_texture_from_bytes};

type Index = u32;

const VERTEX_ALIGN: DeviceAlignment = DeviceAlignment::of::<EguiVertex>();
const INDEX_ALIGN: DeviceAlignment = DeviceAlignment::of::<Index>();

/// Should match vertex definition of egui
#[repr(C)]
#[derive(BufferContents, Vertex)]
pub struct EguiVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
    #[format(R32G32_SFLOAT)]
    pub tex_coords: [f32; 2],
    #[format(R8G8B8A8_UNORM)]
    pub color: [u8; 4],
}

pub struct EguiSystemConfig {
    /// When true the bindless system will be used for rendering as long as the bindless context
    /// exists for [Resources]. If not descriptor sets will be bound for the images instead.
    pub use_bindless: bool,
    /// Optional debug utils to be associated with the work done when rendering egui.
    pub debug_utils: Option<DebugUtilsLabel>,
}

impl Default for EguiSystemConfig {
    fn default() -> Self {
        Self { use_bindless: true, debug_utils: None }
    }
}

#[derive(Clone)]
pub enum EguiTexture {
    /// Can be used even if bindless is enabled, and will be converted to a bindless texture if it is.
    Raw { image_view: Arc<ImageView>, sampler: Arc<Sampler> },
    /// Must only be used if bindless is enabled.
    Bindless { sampled_image_id: SampledImageId, sampler_id: SamplerId },
}

/// Your task graph's world type needs to implement this to expose data
/// needed during [RenderEguiTask] execution.
pub trait RenderEguiWorld<W: 'static + RenderEguiWorld<W> + ?Sized> {
    fn get_egui_system(&self) -> &EguiSystem<W>;
    fn get_swapchain_id(&self) -> Id<Swapchain>;
}

pub enum EguiSystemError {
    PresentationNotSupported,
    TransferNotSupported,
    ImageCreationError(ImageCreationError),
    HandleError(Validated<HandleError>),
    Vulkan(Validated<VulkanError>),
    AllocateBuffer(Validated<AllocateBufferError>),
    AllocateImage(Validated<AllocateImageError>),
}

impl Debug for EguiSystemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PresentationNotSupported => {
                f.write_str("The physical device must support presentation to the event loop!")
            }
            Self::TransferNotSupported => {
                f.write_str("The queue provided must support transfer operations!")
            }
            Self::ImageCreationError(err) => err.fmt(f),
            Self::HandleError(err) => err.fmt(f),
            Self::Vulkan(err) => err.fmt(f),
            Self::AllocateImage(err) => err.fmt(f),
            Self::AllocateBuffer(err) => err.fmt(f),
        }
    }
}

/// `EguiSystem` is a rendering backend for egui which is meant to contain it's state and provide a
/// means of integrating egui with an existing taskgraph. There are three functions which must be called
/// to properly fully initialize `EguiSystem` after it has been created:
///
/// - [`render_egui`] This must be called during task graph construction, it creates a taskgraph node forrendering egui and
/// returns it's NodeId for synchronization.
/// - [`create_task_pipeline`] This must be called after task graph construction and requires access to `ExecutableTaskGraph`.
/// - [`update_task_draw_data`] This should be called at the end every frame to update textures and mesh data.
///
/// You need to use this with automatic render pass creation and it will render directly to the swapchain.
pub struct EguiSystem<W: 'static + RenderEguiWorld<W> + ?Sized> {
    queue: Arc<Queue>,
    resources: Arc<Resources>,
    flight_id: Id<Flight>,
    staging_allocator: Option<Arc<dyn MemoryAllocator>>,

    use_bindless: bool,
    debug_utils: Option<DebugUtilsLabel>,
    output_in_linear_colorspace: bool,

    pub egui_ctx: egui::Context,
    pub egui_winit: egui_winit::State,
    surface: Arc<Surface>,

    shapes: Vec<egui::epaint::ClippedShape>,
    textures_delta: egui::TexturesDelta,

    vertex_buffer_ids: Vec<Id<Buffer>>,
    index_buffer_ids: Vec<Id<Buffer>>,

    font_sampler: Arc<Sampler>,
    font_sampler_id: Option<SamplerId>,

    descriptor_set_allocator: Option<Arc<StandardDescriptorSetAllocator>>,
    descriptor_set_layout: Option<Arc<DescriptorSetLayout>>,
    texture_descriptor_sets: Option<AHashMap<egui::TextureId, Arc<DescriptorSet>>>,

    texture_ids: AHashMap<egui::TextureId, (Id<Image>, EguiTexture)>,
    next_native_tex_id: u64,

    egui_node_id: Option<NodeId>,

    _marker: PhantomData<fn() -> W>,
}

impl<W: 'static + RenderEguiWorld<W> + ?Sized> EguiSystem<W> {
    /// Creates a new EguiSystem for rendering to the task graph.
    /// - `event_loop`: The physical device that the [`Queue`] was created with must support
    /// presentation to the event loop provided.
    /// - `queue`: You must verify that the [`Queue`] supports image transfer operations.
    /// - `staging_allocator`: can be optionally provided for temporary allocation of staging buffers.
    /// A `BumpAllocator` works well here if its properly managed with the rest of your program.
    ///
    pub fn new(
        event_loop: &ActiveEventLoop,
        surface: &Arc<Surface>,
        queue: &Arc<Queue>,
        resources: &Arc<Resources>,
        flight_id: Id<Flight>,
        staging_allocator: Option<&Arc<impl MemoryAllocator>>,
        swapchain_format: Format,
        config: EguiSystemConfig,
    ) -> Result<Self, EguiSystemError> {
        let use_bindless = config.use_bindless && resources.bindless_context().is_some();
        let debug_utils = config.debug_utils;

        let output_in_linear_colorspace =
            if let Some(numeric_format) = swapchain_format.numeric_format_color() {
                numeric_format == NumericFormat::SRGB
            } else {
                false
            };

        let physical_device = queue.device().physical_device();
        let queue_index = queue.queue_family_index();
        let max_texture_side = physical_device.properties().max_image_dimension2_d as usize;

        let presentation_support = physical_device
            .presentation_support(queue_index, event_loop)
            .map_err(|err| EguiSystemError::HandleError(err))?;
        if !presentation_support {
            return Err(EguiSystemError::PresentationNotSupported);
        }

        let queue_properties = &physical_device.queue_family_properties()[queue_index as usize];

        if !queue_properties.queue_flags.intersects(QueueFlags::TRANSFER) {
            return Err(EguiSystemError::TransferNotSupported);
        }

        let egui_ctx: egui::Context = Default::default();

        let theme = match egui_ctx.theme() {
            egui::Theme::Dark => winit::window::Theme::Dark,
            egui::Theme::Light => winit::window::Theme::Light,
        };

        let egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            egui_ctx.viewport_id(),
            event_loop,
            Some(surface_window(surface).scale_factor() as f32),
            Some(theme),
            Some(max_texture_side),
        );

        let font_sampler = Sampler::new(queue.device(), &SamplerCreateInfo {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            address_mode: [SamplerAddressMode::ClampToEdge; 3],
            mipmap_mode: SamplerMipmapMode::Linear,
            ..Default::default()
        })
        .map_err(|err| EguiSystemError::Vulkan(err))?;

        let font_sampler_id = if use_bindless {
            let bcx = resources.bindless_context().unwrap();

            Some(bcx.global_set().add_sampler(font_sampler.clone()))
        } else {
            None
        };

        let flight = resources.flight(flight_id).unwrap();
        let frames_in_flight = flight.frame_count();

        let create_buffer_ids = |create_info, allocation_info, layout| {
            let mut buffer_ids = vec![];
            for _ in 0..frames_in_flight {
                let buffer_id = resources
                    .create_buffer(&create_info, &allocation_info, layout)
                    .map_err(|err| EguiSystemError::AllocateBuffer(err))?;
                buffer_ids.push(buffer_id);
            }
            Ok(buffer_ids)
        };

        let vertex_buffer_ids = create_buffer_ids(
            BufferCreateInfo { usage: BufferUsage::VERTEX_BUFFER, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DeviceLayout::from_size_alignment(
                (MAX_VERTICES * size_of::<EguiVertex>()) as u64,
                VERTEX_ALIGN.into(),
            )
            .unwrap(),
        )?;

        let index_buffer_ids = create_buffer_ids(
            BufferCreateInfo { usage: BufferUsage::INDEX_BUFFER, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DeviceLayout::from_size_alignment(
                (MAX_INDICES * size_of::<Index>()) as u64,
                INDEX_ALIGN.into(),
            )
            .unwrap(),
        )?;

        let (descriptor_set_allocator, descriptor_set_layout, texture_descriptor_sets) =
            if !use_bindless {
                let descriptor_set_allocator =
                    StandardDescriptorSetAllocator::new(queue.device(), &Default::default()).into();

                let descriptor_set_layout =
                    DescriptorSetLayout::new(queue.device(), &DescriptorSetLayoutCreateInfo {
                        bindings: &[DescriptorSetLayoutBinding {
                            stages: ShaderStages::FRAGMENT,
                            ..DescriptorSetLayoutBinding::new(DescriptorType::CombinedImageSampler)
                        }],
                        ..Default::default()
                    })
                    .map_err(|err| EguiSystemError::Vulkan(err))?;

                let texture_descriptor_sets = AHashMap::default();

                (
                    Some(descriptor_set_allocator),
                    Some(descriptor_set_layout),
                    Some(texture_descriptor_sets),
                )
            } else {
                (None, None, None)
            };

        Ok(Self {
            queue: queue.clone(),
            resources: resources.clone(),
            flight_id,
            staging_allocator: staging_allocator.map(|x| x.clone().as_dyn()),

            debug_utils,
            use_bindless,
            output_in_linear_colorspace,

            egui_ctx,
            egui_winit,
            surface: surface.clone(),

            shapes: vec![],
            textures_delta: Default::default(),

            vertex_buffer_ids,
            index_buffer_ids,

            font_sampler,
            font_sampler_id,

            descriptor_set_allocator,
            descriptor_set_layout,
            texture_descriptor_sets,
            texture_ids: AHashMap::default(),

            next_native_tex_id: 0,

            egui_node_id: None,

            _marker: PhantomData,
        })
    }

    /// Creates RenderEguiTask and adds it to task graph for rendering
    pub fn render_egui(
        &mut self,
        task_graph: &mut TaskGraph<W>,
        virtual_swapchain_id: Id<Swapchain>,
        virtual_framebuffer_id: Id<Framebuffer>,
    ) -> NodeId {
        for (vertex_id, index_id) in self.vertex_buffer_ids.iter().zip(&self.index_buffer_ids) {
            task_graph.add_host_buffer_access(*vertex_id, HostAccessType::Write);
            task_graph.add_host_buffer_access(*index_id, HostAccessType::Write);
        }

        // Initialize RenderEguiTask
        let mut task_node_builder = task_graph.create_task_node(
            "Render Egui",
            QueueFamilyType::Graphics,
            RenderEguiTask::new(),
        );

        for (vertex_id, index_id) in self.vertex_buffer_ids.iter().zip(&self.index_buffer_ids) {
            task_node_builder
                .buffer_access(*vertex_id, AccessTypes::VERTEX_ATTRIBUTE_READ)
                .buffer_access(*index_id, AccessTypes::INDEX_READ);
        }

        task_node_builder.framebuffer(virtual_framebuffer_id).color_attachment(
            virtual_swapchain_id.current_image_id(),
            AccessTypes::COLOR_ATTACHMENT_WRITE | AccessTypes::COLOR_ATTACHMENT_READ,
            ImageLayoutType::Optimal,
            &AttachmentInfo { ..Default::default() },
        );
        let node_id = task_node_builder.build();
        self.egui_node_id = Some(node_id);

        node_id
    }

    /// Creates the graphics pipeline for the task node, this **must** be called after taskgraph construction.
    pub fn create_task_pipeline(
        &mut self,
        task_graph: &mut ExecutableTaskGraph<W>,
        resources: &Arc<Resources>,
        device: &Arc<Device>,
    ) -> Result<(), EguiSystemError> {
        let node_id = self.get_node_id();

        let egui_node = task_graph.task_node_mut(node_id).unwrap();
        let subpass = egui_node.subpass().unwrap().clone();
        egui_node
            .task_mut()
            .downcast_mut::<RenderEguiTask<W>>()
            .unwrap()
            .create_pipeline(resources, device, &subpass, self.use_bindless)
            .map_err(|err| EguiSystemError::Vulkan(err))?;

        Ok(())
    }

    /// Extracts the draw data for the frame, updates textures, and sends mesh primitive data required for rendering
    /// to [RenderEguiTask].
    pub fn update_task_draw_data(&mut self, task_graph: &mut ExecutableTaskGraph<W>) {
        let (clipped_meshes, textures_delta) = self.extract_draw_data_at_frame_end();

        self.update_textures(&textures_delta.set).unwrap();

        for &id in &textures_delta.free {
            self.unregister_image(id);
        }

        let node_id = self.get_node_id();

        let egui_node = task_graph.task_node_mut(node_id).unwrap();
        egui_node
            .task_mut()
            .downcast_mut::<RenderEguiTask<W>>()
            .unwrap()
            .set_clipped_meshes(clipped_meshes);
    }

    fn get_node_id(&self) -> NodeId {
        self.egui_node_id.expect(
            "RenderEguiTask must be initialized by calling render_egui during task graph \
             construction.",
        )
    }

    /// Registers a user texture. User texture needs to be unregistered when it is no longer needed.
    /// If bindless is in use raw textures can still be provided as they will be converted to bindless.
    pub fn register_image(
        &mut self,
        image: Id<Image>,
        egui_texture: EguiTexture,
    ) -> Result<egui::TextureId, EguiSystemError> {
        let egui_texture = self.convert_egui_texture(egui_texture);

        let id = egui::TextureId::User(self.next_native_tex_id);
        self.next_native_tex_id += 1;

        if !self.use_bindless {
            if let EguiTexture::Raw { ref image_view, ref sampler } = egui_texture {
                let descriptor_set = self
                    .sampled_image_descriptor_set(image_view, sampler)
                    .map_err(|err| EguiSystemError::Vulkan(err))?;
                self.texture_descriptor_sets.as_mut().unwrap().insert(id, descriptor_set);
            }
        }

        self.texture_ids.insert(id, (image, egui_texture));

        Ok(id)
    }

    /// Registers a user image to be used by egui
    /// - `image_file_bytes`: e.g. include_bytes!("./assets/tree.png")
    /// - `format`: e.g. vulkano::format::Format::R8G8B8A8Unorm
    ///
    /// This differs from [`EguiSystem::register_user_image_from_bytes`] in that it
    /// automatically converts into the rgba8 format with [`image::load_from_memory`]
    /// and [`image::DynamicImage::as_rgba8`].
    ///
    #[cfg(feature = "image")]
    pub fn register_user_image(
        &mut self,
        image_file_bytes: &[u8],
        format: vulkano::format::Format,
        sampler_create_info: SamplerCreateInfo,
    ) -> Result<egui::TextureId, EguiSystemError> {
        // SAFETY: [`EguiSystem`] cannot be successfully created with a queue that
        // doesn't support transfer operations.
        let (image_id, image_view) = unsafe {
            immutable_texture_from_file::<W>(
                &self.queue,
                &self.resources,
                self.flight_id,
                self.staging_allocator.as_ref(),
                image_file_bytes,
                format,
            )
            .map_err(|err| EguiSystemError::ImageCreationError(err))?
        };

        let sampler = Sampler::new(self.queue.device(), &sampler_create_info)
            .map_err(|err| EguiSystemError::Vulkan(err))?;

        let egui_texture = self.get_egui_texture(image_view, sampler);
        let texture_id = self.register_image(image_id, egui_texture)?;

        Ok(texture_id)
    }

    /// Registers a user image to be used by egui from raw file bytes.
    /// - `image_file_bytes`: e.g. include_bytes!("./assets/tree.png")
    /// - `format`: e.g. vulkano::format::Format::R8G8B8A8Unorm
    ///
    pub fn register_user_image_from_bytes(
        &mut self,
        image_byte_data: &[u8],
        dimensions: [u32; 2],
        format: vulkano::format::Format,
        sampler_create_info: SamplerCreateInfo,
    ) -> Result<egui::TextureId, EguiSystemError> {
        // SAFETY: [`EguiSystem`] cannot be successfully created with a queue that
        // doesn't support transfer operations.
        let (image_id, image_view) = unsafe {
            immutable_texture_from_bytes::<W>(
                &self.queue,
                &self.resources,
                self.flight_id,
                self.staging_allocator.as_ref(),
                image_byte_data,
                dimensions,
                format,
            )
            .map_err(|err| EguiSystemError::ImageCreationError(err))?
        };

        let sampler = Sampler::new(self.queue.device(), &sampler_create_info)
            .map_err(|err| EguiSystemError::Vulkan(err))?;

        let egui_texture = self.get_egui_texture(image_view, sampler);
        let texture_id = self.register_image(image_id, egui_texture)?;

        Ok(texture_id)
    }

    /// Unregister user texture.
    pub fn unregister_image(&mut self, texture_id: egui::TextureId) {
        if !self.use_bindless {
            self.texture_descriptor_sets.as_mut().unwrap().remove(&texture_id);
        }
        self.texture_ids.remove(&texture_id);
    }

    pub fn get_egui_texture(
        &self,
        image_view: Arc<ImageView>,
        sampler: Arc<Sampler>,
    ) -> EguiTexture {
        if self.use_bindless {
            let bcx = self.resources.bindless_context().unwrap();

            let sampled_image_id =
                bcx.global_set().add_sampled_image(image_view, ImageLayout::General);
            let sampler_id = bcx.global_set().add_sampler(sampler);

            EguiTexture::Bindless { sampled_image_id, sampler_id }
        } else {
            EguiTexture::Raw { image_view, sampler }
        }
    }

    fn convert_egui_texture(&self, egui_texture: EguiTexture) -> EguiTexture {
        if let EguiTexture::Raw { ref image_view, ref sampler } = egui_texture {
            if self.use_bindless {
                let bcx = self.resources.bindless_context().unwrap();

                let sampled_image_id =
                    bcx.global_set().add_sampled_image(image_view.clone(), ImageLayout::General);
                let sampler_id = bcx.global_set().add_sampler(sampler.clone());

                EguiTexture::Bindless { sampled_image_id, sampler_id }
            } else {
                egui_texture
            }
        } else {
            assert!(self.use_bindless, "Bindless must be enabled for EguiSystem!");

            egui_texture
        }
    }

    fn sampled_image_descriptor_set(
        &self,
        image_view: &Arc<ImageView>,
        sampler: &Arc<Sampler>,
    ) -> Result<Arc<DescriptorSet>, Validated<VulkanError>> {
        assert!(!self.use_bindless, "Bindless must be disabled for descriptor sets to be created");

        DescriptorSet::new(
            &self.descriptor_set_allocator.as_ref().unwrap().clone(),
            &self.descriptor_set_layout.as_ref().unwrap().clone(),
            &[WriteDescriptorSet::image(0, &DescriptorImageInfo {
                image_view: Some(&image_view.clone()),
                sampler: Some(&sampler.clone()),
                image_layout: ImageLayout::ShaderReadOnlyOptimal,
            })],
            &[],
        )
    }

    fn image_size_bytes(&self, delta: &egui::epaint::ImageDelta) -> usize {
        let image = &delta.image;
        image.width() * image.height() * image.bytes_per_pixel()
    }

    /// Write a single texture delta using the provided staging region and commandbuffer
    fn update_texture_within(
        &mut self,
        id: egui::TextureId,
        delta: &egui::epaint::ImageDelta,
        buffer_id: Id<Buffer>,
        range: Range<u64>,
    ) -> Result<(), ImageCreationError> {
        // Extract pixel data from egui, writing into our region of the stage buffer.
        let (format, bytes) = match &delta.image {
            egui::ImageData::Color(image) => {
                assert_eq!(
                    image.width() * image.height(),
                    image.pixels.len(),
                    "Mismatch between texture size and texel count"
                );

                let bytes: Vec<u8> =
                    image.pixels.iter().flat_map(|color| color.to_array()).collect();

                (Format::R8G8B8A8_SRGB, bytes)
            }
        };

        let extent = [delta.image.width() as u32, delta.image.height() as u32, 1];

        // Copy texture data to existing image if delta pos exists (e.g. font changed)
        let (is_new_image, (new_image_id, new_egui_texture)) = if delta.pos.is_some() {
            let Some(existing_image) = self.texture_ids.get(&id) else {
                // Egui wants us to update this texture but we don't have it to begin with!
                panic!("attempt to write into non-existing image");
            };

            (false, existing_image.clone())
        } else {
            // Otherwise save the newly created image

            let create_info = &ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format,
                extent,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            };
            let allocation_info = &AllocationCreateInfo::default();

            let new_image_id = self
                .resources
                .create_image(create_info, allocation_info)
                .map_err(ImageCreationError::AllocateImage)?;

            // Swizzle packed font images up to a full premul white.
            let component_mapping = match format {
                Format::R8G8_UNORM => ComponentMapping {
                    r: ComponentSwizzle::Red,
                    g: ComponentSwizzle::Red,
                    b: ComponentSwizzle::Red,
                    a: ComponentSwizzle::Green,
                },
                _ => ComponentMapping::identity(),
            };

            let image = self.resources.image(new_image_id).unwrap().image().clone();
            let image_view = ImageView::new(&image, &ImageViewCreateInfo {
                component_mapping,
                ..ImageViewCreateInfo::from_image(&image)
            })
            .map_err(ImageCreationError::Vulkan)?;

            let new_egui_texture = if self.use_bindless {
                let bcx = self.resources.bindless_context().unwrap();

                let sampled_image_id =
                    bcx.global_set().add_sampled_image(image_view, ImageLayout::General);
                let sampler_id = self.font_sampler_id.unwrap();

                EguiTexture::Bindless { sampled_image_id, sampler_id }
            } else {
                let sampler = self.font_sampler.clone();

                EguiTexture::Raw { image_view, sampler }
            };

            (true, (new_image_id, new_egui_texture))
        };

        let flight = self.resources.flight(self.flight_id).unwrap();
        flight.wait(None).unwrap();

        // SAFETY:
        // * The resources are not being accessed by any other task graph execution.
        // * [`EguiSystem`] cannot be successfully created with a queue that doesn't support
        // transfer operations.
        unsafe {
            vulkano_taskgraph::execute(
                &self.queue,
                &self.resources,
                self.flight_id,
                |builder, task_context| {
                    let write = task_context.write_buffer::<[u8]>(buffer_id, range.clone())?;
                    write.copy_from_slice(&bytes);

                    if is_new_image {
                        // Defer upload of data
                        builder.copy_buffer_to_image(&CopyBufferToImageInfo {
                            src_buffer: buffer_id,
                            dst_image: new_image_id,
                            regions: &[BufferImageCopy {
                                buffer_offset: range.start,
                                image_extent: extent,
                                image_subresource: ImageSubresourceLayers {
                                    aspects: ImageAspects::COLOR,
                                    mip_level: 0,
                                    base_array_layer: 0,
                                    layer_count: None,
                                },
                                ..Default::default()
                            }],
                            ..Default::default()
                        })?;
                    } else {
                        let pos = delta.pos.unwrap();
                        // Defer upload of data
                        builder.copy_buffer_to_image(&CopyBufferToImageInfo {
                            src_buffer: buffer_id,
                            dst_image: new_image_id,
                            regions: &[BufferImageCopy {
                                buffer_offset: range.start,
                                image_offset: [pos[0] as u32, pos[1] as u32, 0],
                                image_extent: extent,
                                // Always use the whole image (no arrays or mips are performed)
                                image_subresource: ImageSubresourceLayers {
                                    aspects: ImageAspects::COLOR,
                                    mip_level: 0,
                                    base_array_layer: 0,
                                    layer_count: None,
                                },
                                ..Default::default()
                            }],
                            ..Default::default()
                        })?;
                    }

                    Ok(())
                },
                [(buffer_id, HostAccessType::Write)],
                [(buffer_id, AccessTypes::COPY_TRANSFER_READ)],
                [(new_image_id, AccessTypes::COPY_TRANSFER_WRITE, ImageLayoutType::Optimal)],
            )
        }
        .map_err(|err| ImageCreationError::ExecuteError(err))?;

        if is_new_image {
            if !self.use_bindless {
                if let EguiTexture::Raw { ref image_view, ref sampler } = new_egui_texture {
                    let descriptor_set = self
                        .sampled_image_descriptor_set(image_view, sampler)
                        .map_err(|err| ImageCreationError::Vulkan(err))?;
                    self.texture_descriptor_sets.as_mut().unwrap().insert(id, descriptor_set);
                }
            }
            self.texture_ids.insert(id, (new_image_id, new_egui_texture));
        }

        Ok(())
    }

    /// Write the entire texture delta for this frame.
    fn update_textures(
        &mut self,
        sets: &[(egui::TextureId, egui::epaint::ImageDelta)],
    ) -> Result<(), EguiSystemError> {
        if sets.is_empty() {
            return Ok(());
        }

        // Allocate enough memory to upload every delta at once.
        let total_size_bytes =
            sets.iter().map(|(_, set)| self.image_size_bytes(set)).sum::<usize>() * 4;

        let total_size_bytes = total_size_bytes as u64;
        let Ok(total_size_bytes) = vulkano::NonZeroDeviceSize::try_from(total_size_bytes) else {
            return Ok(());
        };

        let buffer_id = {
            let create_info =
                &BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() };
            let allocation_info = &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            };
            let layout = DeviceLayout::new(total_size_bytes, DeviceAlignment::MIN).unwrap();

            if let Some(staging_allocator) = self.staging_allocator.as_ref() {
                let buffer = Buffer::new(staging_allocator, create_info, allocation_info, layout)
                    .map_err(|err| EguiSystemError::AllocateBuffer(err))?;
                self.resources.add_buffer(buffer)
            } else {
                self.resources
                    .create_buffer(create_info, allocation_info, layout)
                    .map_err(|err| EguiSystemError::AllocateBuffer(err))?
            }
        };

        // Keep track of where to write the next image to into the staging buffer.
        let mut past_buffer_end = 0;

        for (id, delta) in sets {
            let image_size_bytes = self.image_size_bytes(delta) as u64;
            let range = past_buffer_end..(past_buffer_end + image_size_bytes);

            // Bump for next loop
            past_buffer_end += image_size_bytes;

            if let Some(err) = self.update_texture_within(*id, delta, buffer_id, range).err() {
                panic!("Failed to create new image for id: {:?}, with error: {:?}", id, err);
            }
        }

        // Queue destruction of staging buffer
        let mut batch = self.resources.create_deferred_batch();
        batch.destroy_buffer(buffer_id);

        // SAFETY: The buffer isn't used by any other flights.
        unsafe {
            batch.enqueue_with_flights([self.flight_id]);
        }

        if self.staging_allocator.is_some() {
            // Wait to ensure the staging allocator is reset.
            let flight = self.resources.flight(self.flight_id).unwrap();
            flight.wait(None).unwrap();
        }

        Ok(())
    }

    /// Returns the pixels per point of the window of this gui.
    fn pixels_per_point(&self) -> f32 {
        egui_winit::pixels_per_point(&self.egui_ctx, surface_window(&self.surface))
    }

    /// Updates context state by winit window event.
    /// Returns `true` if egui wants exclusive use of this event
    /// (e.g. a mouse click on an egui window, or entering text into a text field).
    /// For instance, if you use egui for a game, you want to first call this
    /// and only when this returns `false` pass on the events to your game.
    ///
    /// Note that egui uses `tab` to move focus between elements, so this will always return `true` for tabs.
    pub fn update(&mut self, winit_event: &winit::event::WindowEvent) -> bool {
        self.egui_winit.on_window_event(surface_window(&self.surface), winit_event).consumed
    }

    /// Begins Egui frame & determines what will be drawn later. This must be called before draw, and after `update` (winit event).
    pub fn immediate_ui(&mut self) -> egui::Context {
        let raw_input = self.egui_winit.take_egui_input(surface_window(&self.surface));
        self.egui_ctx.begin_pass(raw_input);
        self.egui_ctx.clone()
    }

    fn extract_draw_data_at_frame_end(&mut self) -> (Vec<ClippedPrimitive>, TexturesDelta) {
        self.end_frame();
        let shapes = std::mem::take(&mut self.shapes);
        let textures_delta = std::mem::take(&mut self.textures_delta);
        let clipped_meshes = self.egui_ctx.tessellate(shapes, self.pixels_per_point());

        (clipped_meshes, textures_delta)
    }

    fn end_frame(&mut self) {
        let egui::FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point: _,
            viewport_output: _,
        } = self.egui_ctx.end_pass();

        self.egui_winit.handle_platform_output(surface_window(&self.surface), platform_output);
        self.shapes = shapes;
        self.textures_delta = textures_delta;
    }

    /// Access egui's context (which can be used to e.g. set fonts, visuals etc)
    pub fn context(&self) -> egui::Context {
        self.egui_ctx.clone()
    }

    /// Uploads all meshes in bulk. They will be available in the same order, packed.
    /// None if no vertices or no indices.
    fn upload_meshes(
        &self,
        task_context: &mut TaskContext<'_>,
        clipped_meshes: &[ClippedPrimitive],
    ) -> Result<Option<(Id<Buffer>, Id<Buffer>)>, TaskError> {
        // Iterator over only the meshes, no user callbacks.
        let meshes = clipped_meshes.iter().filter_map(|mesh| match &mesh.primitive {
            Primitive::Mesh(m) => Some(m),
            _ => None,
        });

        // Calculate counts of each mesh, and total bytes for combined data
        let mut total_vertices = 0;
        let mut total_indices = 0;

        for mesh in meshes.clone() {
            total_vertices += mesh.vertices.len();
            total_indices += mesh.indices.len();
        }
        if total_indices == 0 || total_vertices == 0 {
            return Ok(None);
        }

        // We must put the items with stricter align *first* in the packed buffer.
        // Correct at time of writing, but assert in case that changes.
        assert!(VERTEX_ALIGN >= INDEX_ALIGN);

        let frame = task_context.current_frame_index() as usize;

        let vertex_buffer = self.vertex_buffer_ids[frame];
        let vertices = task_context.write_buffer::<[EpaintVertex]>(
            vertex_buffer,
            0..(total_vertices.min(MAX_VERTICES) * size_of::<EpaintVertex>()) as u64,
        )?;

        vertices
            .iter_mut()
            .zip(meshes.clone().flat_map(|m| &m.vertices).copied())
            .for_each(|(into, from)| *into = from);

        let index_buffer = self.index_buffer_ids[frame];
        let indices = task_context.write_buffer::<[Index]>(
            self.index_buffer_ids[frame],
            0..(total_indices.min(MAX_INDICES) * size_of::<Index>()) as u64,
        )?;

        indices
            .iter_mut()
            .zip(meshes.flat_map(|m| &m.indices).copied())
            .for_each(|(into, from)| *into = from);

        Ok(Some((vertex_buffer, index_buffer)))
    }
}

pub struct RenderEguiTask<W: 'static + RenderEguiWorld<W> + ?Sized> {
    pipeline: Option<Arc<GraphicsPipeline>>,
    clipped_meshes: Option<Vec<ClippedPrimitive>>,
    _marker: PhantomData<fn() -> W>,
}

impl<W: 'static + RenderEguiWorld<W> + ?Sized> RenderEguiTask<W> {
    pub fn new() -> RenderEguiTask<W> {
        RenderEguiTask::<W> { pipeline: None, clipped_meshes: None, _marker: PhantomData }
    }

    pub fn create_pipeline(
        &mut self,
        resources: &Arc<Resources>,
        device: &Arc<Device>,
        subpass: &Subpass,
        use_bindless: bool,
    ) -> Result<(), Validated<VulkanError>> {
        self.pipeline = Some({
            let (vs, fs) = if use_bindless {
                (
                    render_egui_bindless_vs::load(device).unwrap().entry_point("main").unwrap(),
                    render_egui_bindless_fs::load(device).unwrap().entry_point("main").unwrap(),
                )
            } else {
                (
                    render_egui_vs::load(device).unwrap().entry_point("main").unwrap(),
                    render_egui_fs::load(device).unwrap().entry_point("main").unwrap(),
                )
            };

            let blend = AttachmentBlend {
                src_color_blend_factor: BlendFactor::One,
                src_alpha_blend_factor: BlendFactor::OneMinusDstAlpha,
                dst_alpha_blend_factor: BlendFactor::One,
                ..AttachmentBlend::alpha()
            };

            let blend_state = ColorBlendState {
                attachments: &[ColorBlendAttachmentState {
                    blend: Some(blend),
                    ..Default::default()
                }],
                ..ColorBlendState::default()
            };

            let stages =
                &[PipelineShaderStageCreateInfo::new(&vs), PipelineShaderStageCreateInfo::new(&fs)];

            let layout = if use_bindless {
                let bcx = resources.bindless_context().unwrap();

                bcx.pipeline_layout_from_stages(stages)?
            } else {
                PipelineLayout::from_stages(device, stages)?
            };

            GraphicsPipeline::new(device, None, &GraphicsPipelineCreateInfo {
                stages,
                vertex_input_state: Some(&EguiVertex::per_vertex().definition(&vs).unwrap()),
                input_assembly_state: Some(&InputAssemblyState::default()),
                viewport_state: Some(&ViewportState::default()),
                rasterization_state: Some(&RasterizationState::default()),
                multisample_state: Some(&MultisampleState {
                    rasterization_samples: subpass.num_samples().unwrap_or(SampleCount::Sample1),
                    ..Default::default()
                }),
                color_blend_state: Some(&blend_state),
                dynamic_state: &[DynamicState::Viewport, DynamicState::Scissor],
                subpass: Some(PipelineSubpassType::BeginRenderPass(subpass)),
                ..GraphicsPipelineCreateInfo::new(&layout)
            })?
        });

        Ok(())
    }

    pub fn set_clipped_meshes(&mut self, clipped_meshes: Vec<ClippedPrimitive>) {
        self.clipped_meshes = Some(clipped_meshes);
    }
}

impl<W: 'static + RenderEguiWorld<W> + ?Sized> Task for RenderEguiTask<W> {
    type World = W;

    unsafe fn execute(
        &self,
        builder: &mut RecordingCommandBuffer<'_>,
        task_context: &mut TaskContext<'_>,
        render_context: &Self::World,
    ) -> TaskResult {
        let egui_system = render_context.get_egui_system();
        let swapchain_id = render_context.get_swapchain_id();
        let Some(ref clipped_meshes) = self.clipped_meshes else {
            return Ok(());
        };

        let swapchain_state = task_context.swapchain(swapchain_id)?;
        let Some(image_index) = swapchain_state.current_image_index() else {
            return Ok(());
        };
        let swapchain_extent = swapchain_state.images()[image_index as usize].extent();
        let extent = [swapchain_extent[0], swapchain_extent[1]];

        let Some(ref pipeline) = self.pipeline else {
            panic!("Pipeline must be created before task is executed!");
        };

        if let Some(debug_utils_label) = &egui_system.debug_utils {
            builder.as_raw().begin_debug_utils_label(debug_utils_label)?;
        }

        let scale_factor = egui_system.pixels_per_point();
        let screen_size = [extent[0] as f32 / scale_factor, extent[1] as f32 / scale_factor];
        let output_in_linear_colorspace = egui_system.output_in_linear_colorspace.into();

        let mesh_buffers = egui_system.upload_meshes(task_context, clipped_meshes)?;

        // Current position of renderbuffers, advances as meshes are consumed.
        let mut vertex_cursor = 0;
        let mut index_cursor = 0;
        // Some of our state is immutable and only changes
        // if a user callback thrashes it, rebind all when this is set:
        let mut needs_full_rebind = true;
        // Track resources that change from call-to-call.
        // egui already makes the optimization that draws with identical resources are merged into one,
        // so every mesh changes usually one or possibly both of these.
        let mut current_rect = None;
        let mut current_texture = None;

        for ClippedPrimitive { clip_rect, primitive } in clipped_meshes {
            match primitive {
                Primitive::Mesh(mesh) => {
                    // Nothing to draw if we don't have vertices & indices
                    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                        // Consume the mesh and skip it.
                        index_cursor += mesh.indices.len() as u32;
                        vertex_cursor += mesh.vertices.len() as u32;
                        continue;
                    }

                    // Reset overall state, if needed.
                    // Only happens on first mesh, and after a user callback which does unknowable
                    // things to the command buffer's state.
                    if needs_full_rebind {
                        needs_full_rebind = false;

                        // Bind combined meshes.
                        let Some((vertex_buffer, index_buffer)) = mesh_buffers.clone() else {
                            // Only None if there are no mesh calls, but here we are in a mesh call!
                            unreachable!()
                        };

                        builder.set_viewport(0, &[Viewport {
                            extent: [extent[0] as f32, extent[1] as f32],
                            ..Viewport::new()
                        }])?;
                        builder.bind_pipeline_graphics(pipeline)?;

                        builder
                            .bind_index_buffer(index_buffer, 0, None, IndexType::U32)?
                            .bind_vertex_buffers(0, &[vertex_buffer], &[0], &[], &[])?;
                    }
                    // Find and bind image, if different.
                    if current_texture != Some(mesh.texture_id) {
                        let Some(texture_id) = egui_system.texture_ids.get(&mesh.texture_id) else {
                            eprintln!("This texture no longer exists {:?}", mesh.texture_id);
                            continue;
                        };
                        current_texture = Some(mesh.texture_id);

                        if egui_system.use_bindless {
                            if let EguiTexture::Bindless { sampled_image_id, sampler_id } =
                                texture_id.1
                            {
                                builder.as_raw().push_constants(
                                    pipeline.layout(),
                                    0,
                                    &render_egui_bindless_fs::PushConstants {
                                        texture_id: sampled_image_id,
                                        sampler_id,
                                        screen_size,
                                        output_in_linear_colorspace,
                                    },
                                )?;
                            } else {
                                panic!(
                                    "Raw textures must be converted to bindless before rendering."
                                );
                            }
                        } else {
                            let texture_descriptor_sets =
                                egui_system.texture_descriptor_sets.as_ref().unwrap();

                            let Some(descriptor_set) =
                                texture_descriptor_sets.get(&mesh.texture_id)
                            else {
                                eprintln!(
                                    "Descriptor set could not be found for this texture {:?}",
                                    mesh.texture_id
                                );
                                continue;
                            };

                            builder.as_raw().bind_descriptor_sets(
                                PipelineBindPoint::Graphics,
                                self.pipeline.as_ref().unwrap().layout(),
                                0,
                                &[descriptor_set.as_raw()],
                                &[],
                            )?;

                            builder.as_raw().push_constants(
                                pipeline.layout(),
                                0,
                                &render_egui_fs::PushConstants {
                                    screen_size,
                                    output_in_linear_colorspace,
                                },
                            )?;
                        }
                    };
                    // Calculate and set scissor, if different
                    if current_rect != Some(*clip_rect) {
                        current_rect = Some(*clip_rect);
                        let new_scissor = get_rect_scissor(scale_factor, extent, *clip_rect);

                        builder.set_scissor(0, &[new_scissor])?;
                    }

                    // All set up to draw!
                    builder.draw_indexed(
                        mesh.indices.len() as u32,
                        1,
                        index_cursor,
                        vertex_cursor as i32,
                        0,
                    )?;

                    // Consume this mesh for next iteration
                    index_cursor += mesh.indices.len() as u32;
                    vertex_cursor += mesh.vertices.len() as u32;
                }
                Primitive::Callback(_) => {
                    panic!("Callbacks are not currently supported by the task graph.")
                }
            }
        }

        if egui_system.debug_utils.is_some() {
            builder.as_raw().end_debug_utils_label()?;
        }

        Ok(())
    }
}

fn get_rect_scissor(scale_factor: f32, framebuffer_dimensions: [u32; 2], rect: Rect) -> Scissor {
    let min = rect.min;
    let min = egui::Pos2 { x: min.x * scale_factor, y: min.y * scale_factor };
    let min = egui::Pos2 {
        x: min.x.clamp(0.0, framebuffer_dimensions[0] as f32),
        y: min.y.clamp(0.0, framebuffer_dimensions[1] as f32),
    };
    let max = rect.max;
    let max = egui::Pos2 { x: max.x * scale_factor, y: max.y * scale_factor };
    let max = egui::Pos2 {
        x: max.x.clamp(min.x, framebuffer_dimensions[0] as f32),
        y: max.y.clamp(min.y, framebuffer_dimensions[1] as f32),
    };
    Scissor {
        offset: [min.x.round() as u32, min.y.round() as u32],
        extent: [(max.x.round() - min.x) as u32, (max.y.round() - min.y) as u32],
    }
}

//helper to retrieve Window from surface object
fn surface_window(surface: &Surface) -> &Window {
    surface.object().unwrap().downcast_ref::<Window>().unwrap()
}

// Bindful shaders:

mod render_egui_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "./src/render_egui/egui_vs.glsl",
    }
}

mod render_egui_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "./src/render_egui/egui_fs.glsl",
    }
}

// Bindless shaders:

mod render_egui_bindless_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "./src/render_egui/egui_vs.glsl",
        define: [("BINDLESS", "")],
    }
}

mod render_egui_bindless_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "./src/render_egui/egui_fs.glsl",
        define: [("BINDLESS", "")],
    }
}
