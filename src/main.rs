mod camera;

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, RenderingAttachmentInfo, RenderingInfo,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Features, QueueCreateInfo,
    },
    image::{view::ImageView, ImageAccess, ImageUsage, SwapchainImage},
    impl_vertex,
    instance::{Instance, InstanceCreateInfo},
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            rasterization::{RasterizationState, FrontFace},
            render_pass::PipelineRenderingCreateInfo,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{LoadOp, StoreOp},
    swapchain::{
        acquire_next_image, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture},
    Version,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct Vertex {
    position: [f32; 2],
}
impl_vertex!(Vertex, position);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct InstanceData {
    position_offset: [f32; 2],
    scale: f32,
}
impl_vertex!(InstanceData, position_offset, scale);

fn main() {
    let required_extensions = vulkano_win::required_extensions();

    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        enumerate_portability: true,
        ..Default::default()
    })
    .unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    center_window(surface.window());

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.api_version() >= Version::V1_3)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false))
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .expect("No suitable physical device found");

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            enabled_features: Features {
                dynamic_rendering: true,
                ..Features::none()
            },
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let surface_capabilities = physical_device
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        let image_format = Some(
            physical_device
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count,
                image_format,
                image_extent: surface.window().inner_size().into(),
                image_usage: ImageUsage::color_attachment(),
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            },
        )
        .unwrap()
    };

    let vertices = [
        Vertex {
            position: [-0.5, 0.0],
        },
        Vertex {
            position: [0.0, 0.5],
        },
        Vertex {
            position: [0.5, 0.0],
        },
    ];
    let vertex_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, vertices)
            .unwrap();

    let uniform_buffer = CpuBufferPool::<vs::ty::Data>::new(device.clone(), BufferUsage::all());

    let instances = {
        let rows = 10;
        let cols = 10;
        let mut data = Vec::new();

        for c in 0..cols {
            for r in 0..rows {
                let half_cell_w = 0.5 / cols as f32;
                let half_cell_h = 0.5 / rows as f32;

                let x = half_cell_w + (c as f32 / cols as f32) * 2.0 - 1.0;
                let y = half_cell_h + (r as f32 / cols as f32) * 2.0 - 1.0;

                let position_offset = [x, y];
                let scale = 0.5;

                data.push(InstanceData {
                    position_offset,
                    scale,
                });
            }
        }

        data
    };
    let instance_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, instances)
            .unwrap();

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    let pipeline = GraphicsPipeline::start()
        .render_pass(PipelineRenderingCreateInfo {
            color_attachment_formats: vec![Some(swapchain.image_format())],
            ..Default::default()
        })
        .vertex_input_state(
            BuffersDefinition::new()
                .vertex::<Vertex>()
                .instance::<InstanceData>(),
        )
        .input_assembly_state(InputAssemblyState::new())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .rasterization_state(
            RasterizationState::new().front_face(FrontFace::Clockwise),
        )
        .build(device.clone())
        .unwrap();

    let mut camera = camera::Camera {
        fov_y: 45.0,
        z_near: 0.1,
        z_far: 100.0,
        pos: cgmath::Vector3::new(0.0, 0.0, -2.0),
        target: cgmath::Vector3::new(0.0, 0.0, 0.0),
        up: cgmath::Vector3::new(0.0, 1.0, 0.0),
    };
    camera.rotate_y(cgmath::Deg(40.0));
    camera.rotate_x(cgmath::Deg(-25.0));

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let mut attachment_image_views = window_size_dependent_setup(&images, &mut viewport);

    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            recreate_swapchain = true;
        }
        Event::RedrawEventsCleared => {
            previous_frame_end.as_mut().unwrap().cleanup_finished();

            if recreate_swapchain {
                let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                    image_extent: surface.window().inner_size().into(),
                    ..swapchain.create_info()
                }) {
                    Ok(r) => r,
                    Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                    Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                };

                swapchain = new_swapchain;
                attachment_image_views = window_size_dependent_setup(&new_images, &mut viewport);
                recreate_swapchain = false;
            }

            let uniform_buffer_subbuffer = {
                let window_size = surface.window().inner_size();
                let aspect = window_size.width as f32 / window_size.height as f32;
                let view_proj = camera.build_view_projection_matrix(aspect);
                let uniform_data = vs::ty::Data {
                    view_proj: view_proj.into(),
                };

                uniform_buffer.next(uniform_data).unwrap()
            };
            let layout = pipeline.layout().set_layouts().get(0).unwrap();
            let set = PersistentDescriptorSet::new(
                layout.clone(),
                [WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer)],
            )
            .unwrap();

            let (image_num, suboptimal, acquire_future) =
                match acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            builder
                .begin_rendering(RenderingInfo {
                    color_attachments: vec![Some(RenderingAttachmentInfo {
                        load_op: LoadOp::Clear,
                        store_op: StoreOp::Store,
                        clear_value: Some([0.1, 0.1, 0.2, 1.0].into()),
                        ..RenderingAttachmentInfo::image_view(
                            attachment_image_views[image_num].clone(),
                        )
                    })],
                    ..Default::default()
                })
                .unwrap()
                .set_viewport(0, [viewport.clone()])
                .bind_pipeline_graphics(pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    pipeline.layout().clone(),
                    0,
                    set.clone(),
                )
                .bind_vertex_buffers(0, (vertex_buffer.clone(), instance_buffer.clone()))
                .draw(
                    vertex_buffer.len() as u32,
                    instance_buffer.len() as u32,
                    0,
                    0,
                )
                .unwrap()
                .end_rendering()
                .unwrap();

            let command_buffer = builder.build().unwrap();

            let future = previous_frame_end
                .take()
                .unwrap()
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => {
                    previous_frame_end = Some(future.boxed());
                }
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
                Err(e) => {
                    println!("Failed to flush future: {:?}", e);
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
            }
        }
        _ => (),
    });
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    viewport: &mut Viewport,
) -> Vec<Arc<ImageView<SwapchainImage<Window>>>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).unwrap())
        .collect::<Vec<_>>()
}

fn center_window(window: &Window) {
    if let Some(monitor) = window.current_monitor() {
        let screen_size = monitor.size();
        let window_size = window.outer_size();

        window.set_outer_position(winit::dpi::PhysicalPosition {
            x: screen_size.width.saturating_sub(window_size.width) as f64 / 2.
                + monitor.position().x as f64,
            y: screen_size.height.saturating_sub(window_size.height) as f64 / 2.
                + monitor.position().y as f64,
        });
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
        src: "
            #version 450

            layout(location = 0) in vec2 position;

            layout(location = 1) in vec2 position_offset;
            layout(location = 2) in float scale;

            layout(set = 0, binding = 0) uniform Data {
                mat4 view_proj;
            } uniforms;

            void main() {
                gl_Position = uniforms.view_proj * vec4(position * scale + position_offset, 0.0, 1.0);
            }
        "
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 450

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        "
    }
}
