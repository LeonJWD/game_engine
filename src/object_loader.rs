
use crate::app_state;
use crate::model::{Model};
#[derive(Debug)]
pub struct objectLoaderDescriptor {
    pub path: String,
    pub position: cgmath::Vector3<f32>,
    pub rotation: cgmath::Quaternion<f32>,
}
#[derive(Debug)]
pub struct loadedObject {
    instance: app_state::Instance,
    model: Model,
}
#[derive(Debug)]
pub struct LoadedObects {
    pub instances: Vec<app_state::Instance>,
    pub models: Vec<Model>,
    pub instance_ranges: Vec<std::ops::Range<u32>>,
}

pub fn load_object(
    desc: &objectLoaderDescriptor,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> loadedObject {
    let obj_model = Model::load_model(&desc.path, device, queue, layout).unwrap();
    let instance = app_state::Instance {
        position: desc.position,
        rotation: desc.rotation,
    };
    loadedObject {
        model: obj_model,
        instance,
    }
}
pub fn load_models(
    descs: Vec<objectLoaderDescriptor>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> LoadedObects {
    //TODO: prevent multiple file system loads

    let mut instances = Vec::new();
    let mut models = Vec::new();
    let mut instance_ranges = Vec::new();

    let mut sorted_descs = descs;
    sorted_descs.sort_by_key(|o| o.path.clone());
    //println!("{:?}", sorted_descs);

    let mut objs = Vec::new();
    for desc in sorted_descs {
        objs.push(load_object(&desc, device, queue, layout));
    }
    let mut start = 0;
    let mut end = 0;

    instances.push(objs[0].instance);
    for i in 1..objs.len() {
        end += 1;
        instances.push(objs[i].instance);
        if objs[i - 1].model.name != objs[i].model.name {
            models.push(objs[i - 1].model.clone());
            instance_ranges.push(start..end - 1);
            start = end;
        }
    }
    if end as usize != objs.len() {
        instance_ranges.push(start..end + 1);
        models.push(objs[objs.len() - 1].model.clone());
    }
    println!("{:?}", instance_ranges);

    LoadedObects {
        instances,
        models,
        instance_ranges,
    }
    //load each model as instance
}
