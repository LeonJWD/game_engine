use core::num;
use std::{cmp::max, io::SeekFrom, ops::Bound};

use cgmath::{
    Deg, EuclideanSpace, InnerSpace, MetricSpace, Point3, Quaternion, Rad, Rotation, Rotation3,
    Vector3,
};
use json::object;
use winit::event::{KeyEvent, MouseButton, WindowEvent};
use winit::keyboard::PhysicalKey;

use crate::camera::{self, PlayerController};
use crate::{
    model::{self, ModelVertex, Vertex},
    object_loader::LoadedObects,
    render_state::{self, Instance, RenderState},
};

pub struct SimulationState {
    pub player: Player,
    //instances: Vec<render_state::Instance>,
    //obj_models: Vec<model::Model>,
    //pub instance_ranges: Vec<std::ops::Range<u32>>,
    // tickable_Instances: f32, //TODO: type of instances that have behaviour
    pub bvh: BVH,
    pub player_controller: PlayerController,
}
impl SimulationState {
    pub fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => self.player_controller.process_keyboard(*key, *state),
            WindowEvent::MouseWheel { delta, .. } => {
                self.player_controller.process_scroll(delta);
                true
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                // self.mouse_pressed = *state == ElementState::Pressed;
                true
            }

            _ => false,
        }
    }
    pub fn new(objects: &LoadedObects) -> Self {
        //load the static objects data recieved from parameters
        //apply instance offset and rotations to vertices of model meshes
        //create a new bvh
        //when implemented -> load physics objects

        //iterate through the different kinds of objects
        let mut triangles = Vec::new();
        for i in 0..objects.instance_ranges.len() {
            let range = objects.instance_ranges[i].clone();
            let model = &objects.models[i as usize];
            for j in range {
                let instance = objects.instances[j as usize];

                //get of model and apply instance offset and rotation

                for mesh in &model.meshes {
                    for i in mesh.indices.chunks(3) {
                        //get vertex positions and normals
                        let a = Point3::from(mesh.vertices[i[0] as usize].position);
                        let b = Point3::from(mesh.vertices[i[1] as usize].position);
                        let c = Point3::from(mesh.vertices[i[2] as usize].position);

                        let n1 = Vector3::from(mesh.vertices[i[0] as usize].normal);
                        let n2 = Vector3::from(mesh.vertices[i[1] as usize].normal);
                        let n3 = Vector3::from(mesh.vertices[i[2] as usize].normal);

                        //interpolate vertex normals
                        let normal = (n1 + n2 + n3) / 3.0;
                        //apply instance offset and rotation to vertices

                        let a = instance.rotation.rotate_point(a);
                        let b = instance.rotation.rotate_point(b);
                        let c = instance.rotation.rotate_point(c);
                        let a = a + instance.position;
                        let b = b + instance.position;
                        let c = c + instance.position;

                        let triangle = Triangle {
                            a: a,
                            b: b,
                            c: c,
                            normal: normal,
                        };
                        triangles.push(triangle);
                    }
                }
            }
        }

        //create a bvh from the triangles
        let bvh = BVH::new(triangles);

        //create a player object
        let player = Player::new();

        Self {
            player,
            bvh,
            player_controller: PlayerController::new(4.1, 5.1),
        }
    }
}

#[derive(Clone, Copy)]
pub struct BoundingVolume {
    center: cgmath::Vector3<f32>,
    size: f32,
    reference_triangle: Option<Triangle>,
}
impl collidable for BoundingVolume {
    fn collides_width(&self, other: &Self) -> bool {
        let dist = self.center.distance2(other.center);
        if dist <= self.size * self.size {
            return true;
        }
        if dist <= other.size * other.size {
            return true;
        }
        false
    }
}
pub struct BVH {
    volumes: Vec<BoundingVolume>,
    children_relations: Vec<Vec<usize>>,
    parent_relations: Vec<usize>,
    head: usize,
}

#[derive(Clone, Copy)]
struct Triangle {
    a: Point3<f32>,
    b: Point3<f32>,
    c: Point3<f32>,
    normal: Vector3<f32>,
}
impl BVH {
    pub fn new(triangles: Vec<Triangle>) -> Self {
        //TODO: check if correct

        //create a Bounding volume for each Vertex
        //merge two volumes and create a larger one (clustering) until all in one
        let mut volumes = Vec::new();
        let mut children_relations = Vec::new();

        //create volumes for each vertex
        for triangle in triangles {
            let middle = (triangle.a.to_vec() + triangle.b.to_vec() + triangle.c.to_vec()) / 3.0;
            let dist_1 = (triangle.a - middle).distance([0.0, 0.0, 0.0].into());
            let dist_2 = (triangle.a - middle).distance([0.0, 0.0, 0.0].into());
            let dist_3 = (triangle.a - middle).distance([0.0, 0.0, 0.0].into());
            let dist = f32::max(dist_1, f32::max(dist_2, dist_3));
            let volume = BoundingVolume {
                center: middle,
                size: dist,
                reference_triangle: Some(triangle),
            };

            children_relations.push(Vec::new());
            volumes.push(volume);
        }
        //hierachical clustering

        //create distance matrix
        let mut dist_matrix = Vec::with_capacity(volumes.len());
        let mut index_vec = Vec::with_capacity(volumes.len());

        
        for i in 0..volumes.len() {
            let mut dist_vec = Vec::with_capacity(volumes.len());
            index_vec.push(i);

            for j in 0..volumes.len() {
                let dist = volumes[i].center.distance(volumes[j].center);
                dist_vec.push(dist);
            }
            dist_matrix.push(dist_vec);
        }
        while dist_matrix.len() >= 1 {
            //get minimum distance
            let mut min_i = 0;
            let mut min_j = 0;
            let mut min = f32::MAX;
            for i in 0..dist_matrix.len() {
                for j in i..dist_matrix.len() {
                    if i != j && dist_matrix[i][j] < min {
                        min_i = i;
                        min_j = j;
                        min = dist_matrix[i][j];
                    }
                }
            }
            //create a bounding volume between volumes with minimum distance

            let volume_a = &volumes[index_vec[min_i]];
            let volume_b = &volumes[index_vec[min_j]];
            //add new volume to bvh,
            let new_center = (volume_a.center + volume_b.center) / 2.0;
            let new_radius = (volume_a.center.distance(volume_b.center))
                + f32::max(volume_a.size, volume_b.size);

            let new_volume = BoundingVolume {
                center: new_center,
                size: new_radius,
                reference_triangle: None,
            };
            volumes.push(new_volume);
            //create parent/child references
            let new_index = volumes.len() - 1;
            children_relations.push(Vec::new());
            children_relations[new_index].push(index_vec[min_i]);
            children_relations[new_index].push(index_vec[min_j]);

            //update index_vec

            index_vec[min_i] = new_index;
            index_vec.remove(min_j);

            //update dists
            //TODO: more efficient since it is not nessescary to loop over all elements
            for i in 0..dist_matrix.len() {

                    //fill in the distance matrix with the merged distances
                
                        if i != min_i {
                            dist_matrix[i][min_i] =
                                f32::max(dist_matrix[i][min_i], dist_matrix[min_j][i]);
                        }
                }
                dist_matrix[min_i][min_i]=0.0;

            println!("dist_matrix: {:?}", dist_matrix.len());
            //remove the seccond volume from the matrix
            dist_matrix.remove(min_j);
            for i in 0..dist_matrix.len() {
                dist_matrix[i].remove(min_j);
            }
        } //repeat until matrix is 1x1

        Self {
            volumes: volumes.clone(),
            children_relations: children_relations,
            parent_relations: Vec::new(),
            head: volumes.len() - 1,
        }
    }
}

pub trait tickable {
    fn tick(&mut self, dt: std::time::Duration, bvh: &BVH);
    fn update_render_state(&mut self, render_state: &mut RenderState);
}
pub trait collidable {
    fn collides_width(&self, other: &Self) -> bool;
}

pub struct Player {
    pub position: cgmath::Point3<f32>,
    pub yaw: Rad<f32>,
    pub pitch: Rad<f32>,
    pub yaw_input: Rad<f32>,
    pub pitch_input: Rad<f32>,
    pub speed: cgmath::Vector3<f32>,
    pub movement_input: Vector3<f32>,
    pub gravity: cgmath::Vector3<f32>,
    pub bounding_volume: BoundingVolume,
}
impl Player {
    fn new() -> Self {
        return Self {
            position: Point3::from([0.0, 5.0, 10.0]),
            yaw: Rad { 0: -90.0 },
            pitch: Rad { 0: -20.0 },
            yaw_input: Rad { 0: 0.0 },
            pitch_input: Rad { 0: 0.0 },
            speed: Vector3::from([0.0, 0.0, 0.0]),
            movement_input: Vector3::from([0.0, 0.0, 0.0]),
            gravity: Vector3::from([0.0, 0.0, 0.0]),
            bounding_volume: BoundingVolume {
                center: Vector3::from([0.0, 0.0, 0.0]),
                size: 1.0,
                reference_triangle: None,
            },
        };
    }

    fn get_collision_volume(&self, other_index: usize, bvh: &BVH) -> Option<BoundingVolume> {
        //go into the BVH
        //TODO: move this code to the bvh struct
        let mut queue = Vec::new();
        queue.push(other_index);

        while (!queue.is_empty()) {
            let current = queue.pop().unwrap();
            if self.bounding_volume.collides_width(&bvh.volumes[current]) {
                let children = &bvh.children_relations[current];
                if children.len() != 0 {
                    queue.append(&mut children.clone());
                } else {
                    return Some(bvh.volumes[current].clone());
                }
            }
        }
        return None;
    }

    fn handle_collision(&mut self, bvh: &BVH) {
        let collided_object = self.get_collision_volume(bvh.head, bvh);
        //collision detected
        if collided_object.is_some() {
            //reflect speed along normal
            let vert_normal =
                Vector3::from(collided_object.unwrap().reference_triangle.unwrap().normal);
            self.speed = self.speed - 2.0 * (self.speed.dot(vert_normal)) * vert_normal;
            //TODO: move along normal out of object
        }
    }
}

impl tickable for Player {
    fn tick(&mut self, dt: std::time::Duration, bvh: &BVH) {
        //println!("{:?}",dt.as_secs_f32());
        self.speed += self.gravity * dt.as_secs_f32();
        //collison check with all objects
        //self.handle_collision(bvh);
        self.position += self.speed * dt.as_secs_f32();
        self.position += self.movement_input * dt.as_secs_f32();
        //TODO: take rotationspeed as input from controller instead direct rotation to be able to apply dt
        self.movement_input = [0.0, 0.0, 0.0].into();
        self.speed *= 0.5;
        self.yaw += self.yaw_input * dt.as_secs_f32();
        self.pitch += self.pitch_input * dt.as_secs_f32();
        self.yaw_input.0 = 0.0;
        self.pitch_input.0 = 0.0;
        //println!("{:?}", self.position);
    }
    fn update_render_state(&mut self, render_state: &mut RenderState) {
        render_state.camera.position = self.position;
        render_state.camera.yaw = self.yaw;
        render_state.camera.pitch = self.pitch;
        render_state.update_camera();
    }
}
