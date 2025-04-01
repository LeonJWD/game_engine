use std::{sync::Arc, time::Duration};

use crate::render_state::RenderState;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

#[derive(Default)]
pub struct App {
    state: Option<RenderState>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let state = pollster::block_on(RenderState::new(window.clone()));

        self.state = Some(state);

        window.request_redraw();
    }
    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let state = self.state.as_mut().unwrap();
        if let DeviceEvent::MouseMotion { delta } = event {
            state.camera_controller.proccess_mouse(delta.0, delta.1);
        }
    }
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let state = self.state.as_mut().unwrap();
        if event == WindowEvent::CloseRequested {
            println!("The close button was pressed. stopping");
            event_loop.exit();
        }

        match event {
            WindowEvent::RedrawRequested => {
                state.render();
                state.get_window().request_redraw();
            }
            WindowEvent::Resized(size) => {
                state.resize(size);
            }

            _ => (),
        }
        state.input(&event);
        state.update(Duration::from_millis(10));
    }
}
