use crate::input::Input;

pub struct State {
    pub input_handler: Input,
}

impl State {
    pub fn input(&mut self, event: &winit::event::WindowEvent) {
        match event {
            winit::event::WindowEvent::KeyboardInput {
                input:
                    winit::event::KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                self.input_handler.key_state_changed(*keycode, *state);
            }
            _ => {}
        }
    }
}
