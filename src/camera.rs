use cgmath::prelude::*;

#[rustfmt::skip]
const GL_TO_VK_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::<f32>::new(
    -1.0,  0.0,  0.0,  0.0,
     0.0, -1.0,  0.0,  0.0,
     0.0,  0.0,  0.5,  0.5,
     0.0,  0.0,  0.0,  1.0,
);

pub struct Camera {
    pub fov_y: f32,
    pub z_near: f32,
    pub z_far: f32,
    pub pos: cgmath::Vector3<f32>,
    pub target: cgmath::Vector3<f32>,
    pub up: cgmath::Vector3<f32>,
}

impl Camera {
    pub fn build_view_projection_matrix(&self, aspect: f32) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(
            cgmath::Point3::<f32>::from_vec(self.pos),
            cgmath::Point3::<f32>::from_vec(self.target),
            self.up,
        );

        let proj = GL_TO_VK_MATRIX
            * cgmath::perspective(cgmath::Deg(self.fov_y), aspect, self.z_near, self.z_far);

        proj * view
    }
}
