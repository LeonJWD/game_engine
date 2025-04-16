use std::{sync::Arc, time::Duration};

use crate::camera::PlayerController;
use crate::simulation_state::{self, tickable};
use crate::{render_state::RenderState, simulation_state::SimulationState};
use winit::event::KeyEvent;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

pub struct App {
    render_state: Option<RenderState>,
    simulation_state: Option<SimulationState>,
    last_simulation: std::time::Instant,
}

impl Default for App {
    fn default() -> Self {
        Self {
            render_state: None,
            simulation_state: None,
            last_simulation: std::time::Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let render_state = pollster::block_on(RenderState::new(window.clone()));

        self.render_state = Some(render_state);

        let simulation_state =
            SimulationState::new(&self.render_state.as_ref().unwrap().loaded_objects);
        self.simulation_state = Some(simulation_state);

        let simulation_state = self.simulation_state.as_mut().unwrap();
        let render_state = self.render_state.as_mut().unwrap();

        let player = &mut simulation_state.player;

        player.update_render_state(render_state);

        //render_state.render();

        window.request_redraw();
    }
    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let simulation_state = self.simulation_state.as_mut().unwrap();
        if let DeviceEvent::MouseMotion { delta } = event {
            //TODO: ?move from render_state to simulation_state?
            simulation_state
                .player_controller
                .proccess_mouse(delta.0, delta.1);
            println!("Mouse input");
        }
    }
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let render_state = self.render_state.as_mut().unwrap();
        let simulation_state = self.simulation_state.as_mut().unwrap();
        if event == WindowEvent::CloseRequested {
            println!("The close button was pressed. stopping");
            event_loop.exit();
        }

        match event {
            WindowEvent::RedrawRequested => {
                render_state.render();
                render_state.get_window().request_redraw();
                //println!("redraw!")
            }
            WindowEvent::Resized(size) => {
                render_state.resize(size);
            }

            _ => (),
        }

        let input_happened = simulation_state.input(&event);
        let player = &mut simulation_state.player;
        //   if input_happened {
        simulation_state.player_controller.update_player(player);
        player.tick(self.last_simulation.elapsed(), &simulation_state.bvh);
        player.update_render_state(render_state);
        //render_state.update_camera();
        //println!("input happened");
        //    }
        self.last_simulation = std::time::Instant::now();
    }
}

impl App {
    //input handling
    //mouse buttons add speed to simulationState
    //new camera update function -> replaces CameraController.update
}
