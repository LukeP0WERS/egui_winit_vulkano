// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{marker::PhantomData, ops::Range, sync::Arc};

use egui::{ahash::AHashMap, epaint::Primitive, ClippedPrimitive, Rect, TexturesDelta};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator,
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
        DescriptorImageInfo, DescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    format::{Format, NumericFormat},
    image::{
        sampler::{
            ComponentMapping, ComponentSwizzle, Filter, Sampler, SamplerAddressMode,
            SamplerCreateInfo, SamplerMipmapMode,
        },
        view::{ImageView, ImageViewCreateInfo},
        Image, ImageAspects, ImageCreateInfo, ImageLayout, ImageSubresourceLayers,
        ImageType, ImageUsage, SampleCount,
    },
    instance::debug::DebugUtilsLabel,
    memory::{
        allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
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
    DeviceSize,
};
use vulkano_taskgraph::{
    command_buffer::{BufferImageCopy, CopyBufferToImageInfo, RecordingCommandBuffer},
    descriptor_set::{SampledImageId, SamplerId},
    graph::{AttachmentInfo, ExecutableTaskGraph, NodeId, TaskGraph},
    resource::{AccessTypes, Flight, HostAccessType, ImageLayoutType, Resources},
    Id, QueueFamilyType, Task, TaskContext, TaskResult,
};
use winit::{event_loop::ActiveEventLoop, window::Window};

const VERTICES_PER_QUAD: DeviceSize = 4;
const VERTEX_BUFFER_SIZE: DeviceSize = 1024 * 1024 * VERTICES_PER_QUAD;
const INDEX_BUFFER_SIZE: DeviceSize = 1024 * 1024 * 2;

use egui::epaint::Vertex as EpaintVertex;

#[cfg(feature = "image")]
use crate::utils::immutable_texture_from_file;
use crate::{immutable_texture_from_bytes, utils::ImageCreationError};

type Index = u32;

const VERTEX_ALIGN: DeviceAlignment = DeviceAlignment::of::<EpaintVertex>();
const INDEX_ALIGN: DeviceAlignment = DeviceAlignment::of::<Index>();

type VertexBuffer = Subbuffer<[egui::epaint::Vertex]>;
type IndexBuffer = Subbuffer<[u32]>;

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

/// `EguiSystem` is a rendering backend for egui which is meant to contain it's state and provide a
/// means of integrating egui with an existing taskgraph. There are three functions which must be called
/// to properly fully initialize `EguiSystem` after it has been created:
///
/// - [`render_egui`] This must be called during task graph construction, it creates a taskgraph node for rendering egui and returns it's NodeId for synchronization.
/// - [`create_task_pipeline`] This must be called after task graph construction and requires access to `ExecutableTaskGraph`.
/// - [`update_task_draw_data`] This should be called at the end every frame to update textures and mesh data.
///
/// You need to use this with automatic render pass creation and it will render directly to the swapchain.
pub struct EguiSystem<W: 'static + RenderEguiWorld<W> + ?Sized> {
    queue: Arc<Queue>,
    resources: Arc<Resources>,
    flight_id: Id<Flight>,

    use_bindless: bool,
    debug_utils: Option<DebugUtilsLabel>,
    output_in_linear_colorspace: bool,

    pub egui_ctx: egui::Context,
    pub egui_winit: egui_winit::State,
    surface: Arc<Surface>,

    shapes: Vec<egui::epaint::ClippedShape>,
    textures_delta: egui::TexturesDelta,

    vertex_index_buffer: Arc<Buffer>,

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
    pub fn new(
        event_loop: &ActiveEventLoop,
        surface: &Arc<Surface>,
        queue: &Arc<Queue>,
        resources: &Arc<Resources>,
        flight_id: Id<Flight>,
        swapchain_format: Format,
        config: EguiSystemConfig,
    ) -> Self {
        let use_bindless = config.use_bindless && resources.bindless_context().is_some();
        let debug_utils = config.debug_utils;

        let output_in_linear_colorspace =
            swapchain_format.numeric_format_color().unwrap() == NumericFormat::SRGB;

        let max_texture_side =
            queue.device().physical_device().properties().max_image_dimension2_d as usize;

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
        .unwrap();

        let font_sampler_id = if use_bindless {
            let bcx = resources.bindless_context().unwrap();

            Some(bcx.global_set().add_sampler(font_sampler.clone()))
        } else {
            None
        };

        let vertex_index_buffer = Buffer::new(
            resources.memory_allocator(),
            &BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER | BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DeviceLayout::from_size_alignment(
                INDEX_BUFFER_SIZE + VERTEX_BUFFER_SIZE,
                VERTEX_ALIGN.max(INDEX_ALIGN).into(),
            )
            .unwrap(),
        )
        .unwrap();

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
                    .unwrap();

                let texture_descriptor_sets = AHashMap::default();

                (
                    Some(descriptor_set_allocator),
                    Some(descriptor_set_layout),
                    Some(texture_descriptor_sets),
                )
            } else {
                (None, None, None)
            };

        Self {
            queue: queue.clone(),
            resources: resources.clone(),
            flight_id,

            debug_utils,
            use_bindless,
            output_in_linear_colorspace,

            egui_ctx,
            egui_winit,
            surface: surface.clone(),

            shapes: vec![],
            textures_delta: Default::default(),

            vertex_index_buffer,

            font_sampler,
            font_sampler_id,

            descriptor_set_allocator,
            descriptor_set_layout,
            texture_descriptor_sets,
            texture_ids: AHashMap::default(),

            next_native_tex_id: 0,

            egui_node_id: None,

            _marker: PhantomData,
        }
    }

    /// Creates RenderEguiTask and adds it to task graph for rendering
    pub fn render_egui(
        &mut self,
        task_graph: &mut TaskGraph<W>,
        virtual_swapchain_id: Id<Swapchain>,
        virtual_framebuffer_id: Id<Framebuffer>,
    ) -> NodeId {
        // Initialize RenderEguiTask
        let node_id = task_graph
            .create_task_node("Render Egui", QueueFamilyType::Graphics, RenderEguiTask::new())
            .framebuffer(virtual_framebuffer_id)
            .color_attachment(
                virtual_swapchain_id.current_image_id(),
                AccessTypes::COLOR_ATTACHMENT_WRITE | AccessTypes::COLOR_ATTACHMENT_READ,
                ImageLayoutType::Optimal,
                &AttachmentInfo { ..Default::default() },
            )
            .build();

        self.egui_node_id = Some(node_id);

        node_id
    }

    /// Creates the graphics pipeline for the task node, this **must** be called after taskgraph construction.
    pub fn create_task_pipeline(
        &mut self,
        task_graph: &mut ExecutableTaskGraph<W>,
        resources: &Arc<Resources>,
        device: &Arc<Device>,
    ) {
        let node_id = self.get_node_id();

        let egui_node = task_graph.task_node_mut(node_id).unwrap();
        let subpass = egui_node.subpass().unwrap().clone();
        egui_node.task_mut().downcast_mut::<RenderEguiTask<W>>().unwrap().create_pipeline(
            resources,
            device,
            &subpass,
            self.use_bindless,
        );
    }

    /// Extracts the draw data for the frame, updates textures, and sends mesh primitive data required for rendering
    /// to [RenderEguiTask].
    pub fn update_task_draw_data(&mut self, task_graph: &mut ExecutableTaskGraph<W>) {
        let (clipped_meshes, textures_delta) = self.extract_draw_data_at_frame_end();

        self.update_textures(&textures_delta.set);

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
    ) -> egui::TextureId {
        let egui_texture = self.convert_egui_texture(egui_texture);

        let id = egui::TextureId::User(self.next_native_tex_id);
        self.next_native_tex_id += 1;

        if !self.use_bindless {
            if let EguiTexture::Raw { ref image_view, ref sampler } = egui_texture {
                let descriptor_set = self.sampled_image_descriptor_set(image_view, sampler);
                self.texture_descriptor_sets.as_mut().unwrap().insert(id, descriptor_set);
            }
        }

        self.texture_ids.insert(id, (image, egui_texture));

        id
    }

    /// Registers a user image to be used by egui
    /// - `image_file_bytes`: e.g. include_bytes!("./assets/tree.png")
    /// - `format`: e.g. vulkano::format::Format::R8G8B8A8Unorm
    #[cfg(feature = "image")]
    pub fn register_user_image(
        &mut self,
        image_file_bytes: &[u8],
        format: vulkano::format::Format,
        sampler_create_info: SamplerCreateInfo,
    ) -> Result<egui::TextureId, ImageCreationError> {
        let (image_id, image_view) = immutable_texture_from_file::<W>(
            &self.queue,
            &self.resources,
            self.flight_id,
            image_file_bytes,
            format,
        )?;

        let sampler = Sampler::new(self.queue.device(), &sampler_create_info).unwrap();

        let egui_texture = self.get_egui_texture(image_view, sampler);

        Ok(self.register_image(image_id, egui_texture))
    }

    pub fn register_user_image_from_bytes(
        &mut self,
        image_byte_data: &[u8],
        dimensions: [u32; 2],
        format: vulkano::format::Format,
        sampler_create_info: SamplerCreateInfo,
    ) -> Result<egui::TextureId, ImageCreationError> {
        let (image_id, image_view) = immutable_texture_from_bytes::<W>(
            &self.queue,
            &self.resources,
            self.flight_id,
            image_byte_data,
            dimensions,
            format,
        )?;

        let sampler = Sampler::new(self.queue.device(), &sampler_create_info).unwrap();

        let egui_texture = self.get_egui_texture(image_view, sampler);

        Ok(self.register_image(image_id, egui_texture))
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
    ) -> Arc<DescriptorSet> {
        assert!(!self.use_bindless, "Bindless must be disabled for descriptor sets to be created");

        DescriptorSet::new(
            self.descriptor_set_allocator.as_ref().unwrap().clone(),
            self.descriptor_set_layout.as_ref().unwrap().clone(),
            [WriteDescriptorSet::image(0, DescriptorImageInfo {
                image_view: Some(image_view.clone()),
                sampler: Some(sampler.clone()),
                image_layout: ImageLayout::ShaderReadOnlyOptimal,
            })],
            [],
        )
        .unwrap()
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

                let bytes: Vec<u8> = image.pixels.iter().flat_map(|color| color.to_array()).collect();

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
            let new_image_id = self
                .resources
                .create_image(
                    &ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format,
                        extent,
                        usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                        ..Default::default()
                    },
                    &AllocationCreateInfo::default(),
                )
                .map_err(ImageCreationError::AllocateImage)?;

            //Swizzle packed font images up to a full premul white.
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

        //let flight = self.resources.flight(self.flight_id).unwrap();
        //flight.wait(None).unwrap();

        unsafe { vulkano_taskgraph::execute(
            &self.queue,
            &self.resources,
            self.flight_id,
            |builder, task_context| {
                let write = task_context.write_buffer::<[u8]>(buffer_id, range.clone())?;
                write.copy_from_slice(&bytes);

                if is_new_image {
                    // Defer upload of data
                    builder
                        .copy_buffer_to_image(&CopyBufferToImageInfo {
                            src_buffer: buffer_id,
                            dst_image: new_image_id,
                            regions: &[BufferImageCopy {
                                buffer_offset: range.start,
                                image_extent: extent,
                                image_subresource: ImageSubresourceLayers {
                                    aspects: ImageAspects::COLOR,
                                    mip_level: 0,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                },
                                ..Default::default()
                            }],
                            ..Default::default()
                        })
                        .unwrap();
                } else {
                    let pos = delta.pos.unwrap();
                    // Defer upload of data
                    builder
                        .copy_buffer_to_image(&CopyBufferToImageInfo {
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
                                    layer_count: 1,
                                },
                                ..Default::default()
                            }],
                            ..Default::default()
                        })
                        .unwrap();
                }

                Ok(())
            },
            [(buffer_id, HostAccessType::Write)],
            [(buffer_id, AccessTypes::COPY_TRANSFER_READ)],
            [(new_image_id, AccessTypes::COPY_TRANSFER_WRITE, ImageLayoutType::Optimal)],
        ) }.unwrap();

        let flight = self.resources.flight(self.flight_id).unwrap();
        flight.wait(None).unwrap();

        if is_new_image {
            if !self.use_bindless {
                if let EguiTexture::Raw { ref image_view, ref sampler } = new_egui_texture {
                    let descriptor_set = self.sampled_image_descriptor_set(image_view, sampler);
                    self.texture_descriptor_sets.as_mut().unwrap().insert(id, descriptor_set);
                }
            }
            self.texture_ids.insert(id, (new_image_id, new_egui_texture));
        }

        Ok(())
    }

    /// Write the entire texture delta for this frame.
    pub fn update_textures(&mut self, sets: &[(egui::TextureId, egui::epaint::ImageDelta)]) {
        // Allocate enough memory to upload every delta at once.
        let total_size_bytes =
            sets.iter().map(|(_, set)| self.image_size_bytes(set)).sum::<usize>() * 4;
        // Infallible - unless we're on a 128 bit machine? :P
        let total_size_bytes = u64::try_from(total_size_bytes).unwrap();
        let Ok(total_size_bytes) = vulkano::NonZeroDeviceSize::try_from(total_size_bytes) else {
            // Nothing to upload!
            return;
        };
        let buffer_id = self
            .resources
            .create_buffer(
                &BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                // Bytes, align of one, infallible.
                DeviceLayout::new(total_size_bytes, DeviceAlignment::MIN).unwrap(),
            )
            .unwrap();

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
        clipped_meshes: &[ClippedPrimitive],
    ) -> Option<(VertexBuffer, IndexBuffer)> {
        // Iterator over only the meshes, no user callbacks.
        let meshes = clipped_meshes.iter().filter_map(|mesh| match &mesh.primitive {
            Primitive::Mesh(m) => Some(m),
            _ => None,
        });

        // Calculate counts of each mesh, and total bytes for combined data
        let (total_vertices, total_size_bytes) = {
            let mut total_vertices = 0;
            let mut total_indices = 0;

            for mesh in meshes.clone() {
                total_vertices += mesh.vertices.len();
                total_indices += mesh.indices.len();
            }
            if total_indices == 0 || total_vertices == 0 {
                return None;
            }

            let total_size_bytes = total_vertices * std::mem::size_of::<EpaintVertex>()
                + total_indices * std::mem::size_of::<Index>();
            (
                total_vertices,
                // Infallible! Checked above.
                u64::try_from(total_size_bytes).unwrap(),
            )
        };

        assert!(total_size_bytes <= self.vertex_index_buffer.size());

        // Allocate a buffer which can hold both packed arrays:
        let buffer = Subbuffer::new(self.vertex_index_buffer.clone());

        // We must put the items with stricter align *first* in the packed buffer.
        // Correct at time of writing, but assert in case that changes.
        assert!(VERTEX_ALIGN >= INDEX_ALIGN);
        let (vertices, indices) = {
            let partition_bytes =
                total_vertices as u64 * std::mem::size_of::<EpaintVertex>() as u64;
            (
                // Slice the start as vertices
                buffer.clone().slice(..partition_bytes).reinterpret::<[EpaintVertex]>(),
                // Take the rest, reinterpret as indices.
                buffer.slice(partition_bytes..).reinterpret::<[Index]>(),
            )
        };

        // We have to upload in two mapping steps to avoid trivial but ugly unsafe.
        {
            let mut vertex_write = vertices.write().unwrap();
            vertex_write
                .iter_mut()
                .zip(meshes.clone().flat_map(|m| &m.vertices).copied())
                .for_each(|(into, from)| *into = from);
        }
        {
            let mut index_write = indices.write().unwrap();
            index_write
                .iter_mut()
                .zip(meshes.flat_map(|m| &m.indices).copied())
                .for_each(|(into, from)| *into = from);
        }

        Some((vertices, indices))
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
    ) {
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

                bcx.pipeline_layout_from_stages(stages).unwrap()
            } else {
                PipelineLayout::from_stages(device, stages).unwrap()
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
            })
            .unwrap()
        });
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
        let clipped_meshes = self.clipped_meshes.as_ref().unwrap();

        let swapchain_state = task_context.swapchain(swapchain_id)?;
        let image_index = swapchain_state.current_image_index().unwrap();
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

        let mesh_buffers = egui_system.upload_meshes(clipped_meshes);

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
                        let Some((vertices, indices)) = mesh_buffers.clone() else {
                            // Only None if there are no mesh calls, but here we are in a mesh call!
                            unreachable!()
                        };

                        builder.set_viewport(0, &[Viewport {
                            extent: [extent[0] as f32, extent[1] as f32],
                            ..Viewport::new()
                        }])?;
                        builder.bind_pipeline_graphics(pipeline)?;
                        builder
                            .as_raw()
                            .bind_index_buffer(&vulkano::buffer::IndexBuffer::U32(indices))?
                            .bind_vertex_buffers(0, &[vertices.into_bytes()])?;
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
