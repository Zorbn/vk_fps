mod camera;
mod state;
mod input;
mod rect;
mod ray;

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess, CpuBufferPool},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    image::{view::ImageView, ImageAccess, ImageUsage, SwapchainImage, AttachmentImage},
    impl_vertex,
    instance::{Instance, InstanceCreateInfo},
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState}, rasterization::{RasterizationState, FrontFace, CullMode}, depth_stencil::DepthStencilState,
        },
        GraphicsPipeline, PipelineBindPoint, Pipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    single_pass_renderpass,
    swapchain::{
        acquire_next_image, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture, FenceSignalFuture}, descriptor_set::{WriteDescriptorSet, PersistentDescriptorSet}, format::Format,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent, VirtualKeyCode, ElementState, MouseButton, DeviceEvent, KeyboardInput},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};
use cgmath::prelude::*;

use crate::{camera::Camera, input::Input, state::State, rect::Rect, ray::Ray};

// TODO:
// Reduce imports.
// Combine camera, etc into player struct.

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct Vertex {
    position: [f32; 3],
}
impl_vertex!(Vertex, position);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct InstanceData {
    position_offset: [f32; 3],
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
        .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
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
                // present_mode: PresentMode::Immediate,
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
        // Forward face
        Vertex {
            position: [-0.5, -0.5, -0.5],
        },
        Vertex {
            position: [0.5, 0.5, -0.5],
        },
        Vertex {
            position: [0.5, -0.5, -0.5],
        },
        Vertex {
            position: [-0.5, -0.5, -0.5],
        },
        Vertex {
            position: [-0.5, 0.5, -0.5],
        },
        Vertex {
            position: [0.5, 0.5, -0.5],
        },
        // Left face
        Vertex {
            position: [-0.5, -0.5, -0.5],
        },
        Vertex {
            position: [-0.5, -0.5, 0.5],
        },
        Vertex {
            position: [-0.5, 0.5, 0.5],
        },
        Vertex {
            position: [-0.5, -0.5, -0.5],
        },
        Vertex {
            position: [-0.5, 0.5, 0.5],
        },
        Vertex {
            position: [-0.5, 0.5, -0.5],
        },
        // Right face
        Vertex {
            position: [0.5, -0.5, -0.5],
        },
        Vertex {
            position: [0.5, 0.5, 0.5],
        },
        Vertex {
            position: [0.5, -0.5, 0.5],
        },
        Vertex {
            position: [0.5, -0.5, -0.5],
        },
        Vertex {
            position: [0.5, 0.5, -0.5],
        },
        Vertex {
            position: [0.5, 0.5, 0.5],
        },
    ];
    let vertex_buffer = {
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, vertices).unwrap()
    };

    let vertices_2 = [
        Vertex {
            position: [-0.5, 0.0, 0.0],
        },
        Vertex {
            position: [0.0, 0.0, -0.5],
        },
        Vertex {
            position: [0.5, 0.0, 0.0],
        },
    ];
    let vertex_buffer_2 = {
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, vertices_2).unwrap()
    };

    let test_rect = Rect {
        min: cgmath::Vector2::new(-0.5, -0.5),
        max: cgmath::Vector2::new(0.5, 0.5),
    };

    let uniform_buffer = CpuBufferPool::<vs::ty::Data>::new(device.clone(), BufferUsage::all());

    let instances = {
        let rows = 1;
        let cols = 1;
        let n_instances = rows * cols;
        let mut data = Vec::new();

        for c in 0..cols {
            for r in 0..rows {
                let half_cell_w = 0.5 / cols as f32;
                let half_cell_h = 0.5 / rows as f32;

                // let x = half_cell_w + (c as f32 / cols as f32) * 2.0 - 1.0;
                // let y = half_cell_h + (r as f32 / rows as f32) * 2.0 - 1.0;

                let n_i = c + r * cols;

                // let z = if n_i == 0 || n_i == n_instances - 1 {
                //     -0.5
                // } else {
                //     0.0
                // };

                let x = 0.0;
                let y = 0.0;
                let z = 0.0;

                println!("{}, {}", x, z);

                let position_offset = [x, y, z];
                let scale = 1.0;

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

    let render_pass = single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16_UNORM,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    )
    .unwrap();

    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(
            BuffersDefinition::new()
                .vertex::<Vertex>()
                .instance::<InstanceData>(),
        )
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .rasterization_state(RasterizationState::new().front_face(FrontFace::Clockwise).cull_mode(CullMode::Back))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap();

    let mut camera = Camera {
        fov_y: 45.0,
        z_near: 0.1,
        z_far: 100.0,
        pos: cgmath::Vector3::new(0.0, 0.0, -2.0),
        target: cgmath::Vector3::new(0.0, 0.0, 0.0),
        up: cgmath::Vector3::unit_y(),
    };
    let mut camera_rotation = cgmath::Vector2::new(0.0, 0.0);

    let mut state = State {
        input_handler: Input::new(),
    };

    let mut is_focused = false;

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };
    let mut framebuffers = window_size_dependent_setup(device.clone(), &images, render_pass.clone(), &mut viewport);
    let mut recreate_swapchain = false;

    let frames_in_flight = images.len();
    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
    let mut previous_fence_num = 0;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { ref event, .. } => {
                state.input(event);

                match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(_) => recreate_swapchain = true,
                    WindowEvent::MouseInput {
                        button, .. 
                    } => match button {
                        MouseButton::Left => {
                            if is_focused {
                                let ray_pos = camera.pos;
                                let ray_dir = camera.get_forward(true);
                                let ray = Ray {
                                    pos: cgmath::Vector2::new(ray_pos.x, ray_pos.z),
                                    dir: cgmath::Vector2::new(ray_dir.x, ray_dir.z),
                                };

                                println!("Hit? {}", ray.intersects(&test_rect));
                            } else {
                                is_focused = set_locked_cursor(surface.window(), true);
                            }
                        }
                        _ => (),
                    },
                    _ => (),
                }
            }
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseMotion { delta } => {
                    if is_focused {
                        camera_rotation.x -= delta.1 as f32 / 5.0;
                        camera_rotation.y += delta.0 as f32 / 5.0;
                        camera.reset_rotation();
                        camera.rotate_x(cgmath::Deg(camera_rotation.x));
                        camera.rotate_y(cgmath::Deg(camera_rotation.y));
                    }
                }
                _ => (),
            },
            Event::RedrawEventsCleared => {
                // Update:
                if state.input_handler.was_key_pressed(VirtualKeyCode::Escape) {
                    is_focused = set_locked_cursor(surface.window(), false);
                }

                let mut forward_dir = 0.0;
                let mut right_dir = 0.0;

                if state.input_handler.is_key_held(VirtualKeyCode::W) {
                    forward_dir += 1.0;
                }

                if state.input_handler.is_key_held(VirtualKeyCode::S) {
                    forward_dir -= 1.0;
                }

                if state.input_handler.is_key_held(VirtualKeyCode::D) {
                    right_dir += 1.0;
                }

                if state.input_handler.is_key_held(VirtualKeyCode::A) {
                    right_dir -= 1.0;
                }

                if right_dir != 0.0 || forward_dir != 0.0 {
                    let vec_dir = cgmath::Vector2::new(right_dir, forward_dir).normalize();

                    camera.move_right(vec_dir.x * 0.01, true);
                    camera.move_forward(vec_dir.y * 0.01, true);
                }

                state.input_handler.update();

                // Draw:
                let dimensions = surface.window().inner_size();
                if dimensions.width == 0 || dimensions.height == 0 {
                    return;
                }

                if recreate_swapchain {
                    let (new_swapchain, new_images) =
                        match swapchain.recreate(SwapchainCreateInfo {
                            image_extent: dimensions.into(),
                            ..swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    swapchain = new_swapchain;
                    framebuffers = window_size_dependent_setup(
                        device.clone(),
                        &new_images,
                        render_pass.clone(),
                        &mut viewport,
                    );
                    recreate_swapchain = false;
                }

                let uniform_buffer_subbuffer = {
                    let view_size = swapchain.image_extent();
                    let aspect = view_size[0] as f32 / view_size[1] as f32;
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
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![
                                Some([0.0, 0.0, 1.0, 1.0].into()),
                                Some(1f32.into()),
                            ],
                            ..RenderPassBeginInfo::framebuffer(framebuffers[image_num].clone())
                        },
                        SubpassContents::Inline,
                    )
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
                    // Test 2nd draw call.
                    .bind_vertex_buffers(0, (vertex_buffer_2.clone(), instance_buffer.clone()))
                    .draw(
                        vertex_buffer_2.len() as u32,
                        instance_buffer.len() as u32,
                        0,
                        0,
                    )
                    .unwrap()
                    .end_render_pass()
                    .unwrap();
                let command_buffer = builder.build().unwrap();

                if let Some(image_fence) = &fences[image_num] {
                    image_fence.wait(None).unwrap();
                }

                let previous_future = match fences[previous_fence_num].clone() {
                    None => {
                        let mut now = sync::now(device.clone());
                        now.cleanup_finished();

                        now.boxed()
                    }
                    Some(fence) => fence.boxed(),
                };

                let future = previous_future
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                fences[image_num] = match future {
                    Ok(value) => Some(Arc::new(value)),
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        None
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        None
                    }
                };

                previous_fence_num = image_num;
            }
            _ => (),
        }
    });
}

fn window_size_dependent_setup(
    device: Arc<Device>,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(device.clone(), dimensions, Format::D16_UNORM).unwrap(),
    )
    .unwrap();

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
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

fn set_locked_cursor(window: &Window, is_locked: bool) -> bool {
    let _ = window.set_cursor_grab(is_locked);
    _ = window.set_cursor_visible(!is_locked);

    is_locked
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

            layout(location = 0) in vec3 position;

            layout(location = 1) in vec3 position_offset;
            layout(location = 2) in float scale;

            layout(set = 0, binding = 0) uniform Data {
                mat4 view_proj;
            } uniforms;

            layout(location = 3) out float instanceId;

            void main() {
                instanceId = gl_InstanceIndex;
                gl_Position = uniforms.view_proj * vec4(position * scale + position_offset, 1.0);
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

            layout(location = 3) in float instanceId;

            void main() {
                f_color = vec4(instanceId/100.0, 0.0, 0.0, 1.0);
            }
        "
    }
}
