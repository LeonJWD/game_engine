use std::cell::RefMut;

use crate::{
    camera::{Camera, Projection}, object_loader, render_state::{self, LightUniform, SpotLights}
};
use cgmath::{self, Rad, Rotation3};

#[derive(Debug)]
pub struct World {
    objects: Vec<object_loader::objectLoaderDescriptor>,
    spot_lights: SpotLights,
}
impl World {
    pub fn spot_lights(&self) -> SpotLights {
        self.spot_lights.clone()
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

            //TODO: correct rotation around x y and z axis

            let rotation = cgmath::Quaternion::from_axis_angle(
                position,
                cgmath::Deg(object["rotation"].as_f32().unwrap()),
            ); //Error: seperate x,y and z
            let desc = object_loader::objectLoaderDescriptor {
                path: object["path"].to_string(),
                position,
                rotation,
            };

            obj_desc.push(desc);
        }
        let lights = &json["spot_lights"];

        let mut i = 0;

        let mut colors=Vec::new();
        let mut cameras=Vec::new();
        let mut projections=Vec::new();

        for light in lights.members() {
            let mut position = [0.0; 3];
        let mut color = [0.0; 3];
            let pos_1 = light["position"][0].as_f32().unwrap();
            let pos_2 = light["position"][1].as_f32().unwrap();
            let pos_3 = light["position"][2].as_f32().unwrap();
            position[0] = pos_1;
            position[1] = pos_2;
            position[2] = pos_3;

            let col_1 = light["color"][0].as_f32().unwrap();
            let col_2 = light["color"][1].as_f32().unwrap();
            let col_3 = light["color"][2].as_f32().unwrap();
            color[0] = col_1;
            color[1] = col_2;
            color[2] = col_3;
            i += 1;

            let yaw=light["yaw"].as_f32().unwrap();
            let pitch=light["pitch"].as_f32().unwrap();

            let width=json["spot_light_width"].as_u32().unwrap();
            let height=json["spot_light_width"].as_u32().unwrap();
            let fovy=light["fovy"].as_f32().unwrap();
            let znear =light["znear"].as_f32().unwrap();
            let zfar =light["zfar"].as_f32().unwrap();

            let proj= Projection::new(width, height, Rad(fovy), znear, zfar);

            let camera=Camera::new(position, Rad(yaw) , Rad(pitch));
            cameras.push(camera);
            colors.push(color);
            projections.push(proj);

        }


        Self {
            spot_lights:SpotLights::new(colors, cameras, projections),
            objects: obj_desc,
        }
    }
}
