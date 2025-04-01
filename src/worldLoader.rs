use crate::{
    model, object_loader,
    render_state::{self, LightUniform},
};
use cgmath::{self, Rotation3};

#[derive(Debug)]
pub struct World {
    objects: Vec<object_loader::objectLoaderDescriptor>,
    lights: render_state::LightUniform,
}
impl World {
    pub fn lights(&self) -> render_state::LightUniform {
        self.lights
    }
    pub fn load_obj_models(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
    ) -> object_loader::LoadedObects {
        object_loader::load_models(&self.objects, device, queue, layout)
    }

    pub fn new(path: &str) -> Self {
        //load json from path
        let file = std::fs::read_to_string(path).unwrap();
        let json: json::JsonValue = json::parse(&file).unwrap();

        let objects = &json["objects"];

        let mut obj_desc = Vec::new();

        for object in objects.members() {
            let pos = &object["position"];

            let position = cgmath::Vector3 {
                x: pos[0].as_f32().unwrap(),
                y: pos[1].as_f32().unwrap(),
                z: pos[2].as_f32().unwrap(),
            };

            let rotation = cgmath::Quaternion::from_axis_angle(
                position,
                cgmath::Deg(object["rotation"].as_f32().unwrap()),
            );
            let desc = object_loader::objectLoaderDescriptor {
                path: object["path"].to_string(),
                position,
                rotation,
            };

            obj_desc.push(desc);
        }
        let lights = &json["lights"];

        let mut position = [[0.0; 4]; render_state::MAX_LIGHTS];
        let mut color = [[0.0; 4]; render_state::MAX_LIGHTS];

        let mut i = 0;

        for light in lights.members() {
            let pos_1 = light["position"][0].as_f32().unwrap();
            let pos_2 = light["position"][1].as_f32().unwrap();
            let pos_3 = light["position"][2].as_f32().unwrap();
            position[i][0] = pos_1;
            position[i][1] = pos_2;
            position[i][2] = pos_3;

            let col_1 = light["color"][0].as_f32().unwrap();
            let col_2 = light["color"][1].as_f32().unwrap();
            let col_3 = light["color"][2].as_f32().unwrap();
            color[i][0] = col_1;
            color[i][1] = col_2;
            color[i][2] = col_3;
            i += 1;
        }
        let lights = LightUniform::new(position, color, i.try_into().unwrap());

        Self {
            lights,
            objects: obj_desc,
        }
    }
}
