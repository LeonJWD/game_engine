use cgmath::*;
use std::f32::consts::FRAC_PI_2;
use winit::dpi::PhysicalPosition;
use winit::event::*;
use winit::keyboard::KeyCode;

use crate::simulation_state::Player;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.0001;

#[derive(Debug, Clone,Copy)]
pub struct Camera {
    pub position: Point3<f32>,
    pub yaw: Rad<f32>,
    pub pitch: Rad<f32>,
}
impl Camera {
    pub fn new<V: Into<Point3<f32>>, Y: Into<Rad<f32>>, P: Into<Rad<f32>>>(
        position: V,
        yaw: Y,
        pitch: P,
    ) -> Self {
        Self {
            position: position.into(),
            yaw: yaw.into(),
            pitch: pitch.into(),
        }
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        let (sin_pitch, cos_pitch) = self.pitch.0.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.0.sin_cos();

        Matrix4::look_to_rh(
            self.position,
            Vector3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize(),
            Vector3::unit_y(),
        )
    }
}
#[derive(Debug, Clone,Copy)]
pub struct Projection {
    aspect: f32,
    fovy: Rad<f32>,
    znear: f32,
    zfar: f32,
}

impl Projection {
    pub fn new<F: Into<Rad<f32>>>(width: u32, height: u32, fovy: F, znear: f32, zfar: f32) -> Self {
        Self {
            aspect: width as f32,
            fovy: fovy.into(),
            znear,
            zfar,
        }
    }
    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }
    pub fn calc_matrix(&self) -> Matrix4<f32> {
        OPENGL_TO_WGPU_MATRIX * perspective(self.fovy, self.aspect, self.znear, self.zfar)
    }
}

pub struct PlayerController {
    ammount_left: f32,
    ammout_right: f32,
    ammount_forward: f32,
    ammount_backward: f32,
    amount_up: f32,
    ammount_down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    scroll: f32,
    speed: f32,
    sensitivity: f32,
}
impl PlayerController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            ammount_left: 0.0,
            ammout_right: 0.0,
            ammount_forward: 0.0,
            ammount_backward: 0.0,
            amount_up: 0.0,
            ammount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            scroll: 0.0,
            speed,
            sensitivity,
        }
    }

    pub fn process_keyboard(&mut self, key: KeyCode, state: ElementState) -> bool {
        let ammount = if state == ElementState::Pressed {
            1.0
        } else {
            0.0
        };
        match key {
            KeyCode::KeyW | KeyCode::ArrowUp => {
                self.ammount_forward = ammount;
                true
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                self.ammount_backward = ammount;
                true
            }
            KeyCode::KeyA => {
                self.ammount_left = ammount;
                true
            }
            KeyCode::KeyD => {
                self.ammout_right = ammount;
                true
            }
            KeyCode::Space => {
                self.amount_up = ammount;
                true
            }
            KeyCode::ShiftLeft => {
                self.ammount_down = ammount;
                true
            }
            _ => false,
        }
    }
    pub fn proccess_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.rotate_horizontal = mouse_dx as f32;
        self.rotate_vertical = mouse_dy as f32;
    }
    pub fn process_scroll(&mut self, delta: &MouseScrollDelta) {
        self.scroll = -match delta {
            MouseScrollDelta::LineDelta(_, scroll) => scroll * 100.0,
            MouseScrollDelta::PixelDelta(PhysicalPosition { y: scroll, .. }) => *scroll as f32,
        };
    }
    pub fn update_player(&mut self, player: &mut Player) {
        let (yaw_sin, yaw_cos) = player.yaw.0.sin_cos();

        let forward = Vector3::new(yaw_cos, 0.0, yaw_sin).normalize();
        let right = Vector3::new(-yaw_sin, 0.0, yaw_cos).normalize();
        let mut speed = Vector3::new(0.0, 0.0, 0.0);
        speed += forward * (self.ammount_forward - self.ammount_backward) * self.speed;
        speed += right * (self.ammout_right - self.ammount_left) * self.speed;

        let (pitch_sin, pitch_cos) = player.pitch.0.sin_cos();
        let scrollward =
            Vector3::new(pitch_cos * yaw_cos, pitch_sin, pitch_sin * yaw_sin).normalize();
        speed += scrollward * self.scroll * self.speed * self.sensitivity;
        self.scroll = 0.0;

        speed.y += (self.amount_up - self.ammount_down) * self.speed;

        player.movement_input = speed;

        player.yaw_input += Rad(self.rotate_horizontal) * self.sensitivity;
        player.pitch_input += Rad(-self.rotate_vertical) * self.sensitivity;

        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        if player.pitch <= -Rad(SAFE_FRAC_PI_2) {
            player.pitch = -Rad(SAFE_FRAC_PI_2);
        } else if player.pitch > Rad(SAFE_FRAC_PI_2) {
            player.pitch = Rad(SAFE_FRAC_PI_2);
        }
    }
}
