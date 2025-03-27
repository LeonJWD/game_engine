use std::sync::Arc;

use anyhow::Ok;
use app::App;
use fs_extra::copy_items;
use fs_extra::dir::CopyOptions;
use std::env;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

mod app;
mod app_state;
mod model;
mod texture;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
    Ok(())
}
