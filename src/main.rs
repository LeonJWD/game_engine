use anyhow::Ok;
use app::App;
use winit::event_loop::{ControlFlow, EventLoop};

mod app;
mod camera;
mod model;
mod object_loader;
mod render_state;
mod texture;
mod worldLoader;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
    Ok(())
}
