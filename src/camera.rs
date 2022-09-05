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

    pub fn move_forward(&mut self, delta: f32, flatten: bool) {
        let delta_vec = self.get_forward(flatten) * delta;
        self.pan(delta_vec);
    }

    pub fn move_right(&mut self, delta: f32, flatten: bool) {
        let delta_vec = self.get_right(flatten) * delta;
        self.pan(delta_vec);
    }

    pub fn get_right(&self, flatten: bool) -> cgmath::Vector3<f32> {
        let forward = self.get_forward(flatten);

        cgmath::Vector3::new(forward.z, forward.y, -forward.x)
    }

    pub fn get_forward(&self, flatten: bool) -> cgmath::Vector3<f32> {
        let forward = (self.target - self.pos).normalize();

        if flatten {
            cgmath::Vector3::new(forward.x, 0.0, forward.z)
        } else {
            forward
        }
    }

    pub fn pan(&mut self, delta: cgmath::Vector3<f32>) {
        self.pos += delta;
        self.target += delta;
    }

    pub fn reset_rotation(&mut self) {
        self.target = self.pos + cgmath::Vector3::<f32>::unit_z();
    }

    pub fn rotate_y(&mut self, delta: cgmath::Deg<f32>) {
        self.rotate_on_axis(0, 2, delta);
    }

    pub fn rotate_x(&mut self, delta: cgmath::Deg<f32>) {
        self.rotate_on_axis(1, 2, delta);
    }

    fn rotate_on_axis(&mut self, h_axis: usize, v_axis: usize, delta: cgmath::Deg<f32>) {
        let delta_rad = cgmath::Rad::from(-delta);
        let sin = delta_rad.sin();
        let cos = delta_rad.cos();

        self.target[h_axis] -= self.pos[h_axis];
        self.target[v_axis] -= self.pos[v_axis];

        let h_new = self.target[h_axis] * cos - self.target[v_axis] * sin;
        let v_new = self.target[h_axis] * sin + self.target[v_axis] * cos;

        self.target[h_axis] = h_new + self.pos[h_axis];
        self.target[v_axis] = v_new + self.pos[v_axis];
    }
}
