
use pollster::FutureExt;

#[cfg(test)]
use super::*;
use crate::{object_loader::LoadedObects, simulation_state::SimulationState, world_loader};

fn load_world() -> world_loader::World {
    world_loader::World::new("res/testworld.json")
}

#[test]
fn load_test_world() {
    let _world = load_world();
}
async fn load_objects() -> LoadedObects {
    let world = load_world();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .unwrap();
    let texture_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
    world.load_obj_models(&device, &queue, &texture_bind_group_layout)
}

#[test]
fn load_world_onjects() {
    let _obj = load_objects().block_on();
}

#[test]
fn create_simulationstate() {
    let objs = load_objects().block_on();
    time_test!("SimulationState creation time");
    let _s = SimulationState::new(&objs);
}
