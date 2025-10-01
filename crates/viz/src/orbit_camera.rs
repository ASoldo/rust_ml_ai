use bevy::input::ButtonInput;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;

const ORBIT_SENSITIVITY: f32 = 0.01;
const ZOOM_SENSITIVITY: f32 = 0.8;
const MIN_PITCH: f32 = -1.5;
const MAX_PITCH: f32 = 1.5;

pub struct OrbitCameraPlugin;

impl Plugin for OrbitCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_camera)
            .add_systems(Update, control_orbit_camera);
    }
}

#[derive(Component)]
pub struct OrbitCamera {
    pub focus: Vec3,
    radius: f32,
    yaw: f32,
    pitch: f32,
    min_radius: f32,
    max_radius: f32,
}

impl OrbitCamera {
    fn new(focus: Vec3, translation: Vec3) -> Self {
        let offset = translation - focus;
        let radius = offset.length();
        let yaw = offset.z.atan2(offset.x);
        let pitch = (offset.y / radius).asin();

        Self {
            focus,
            radius,
            yaw,
            pitch,
            min_radius: 1.5,
            max_radius: 45.0,
        }
    }
}

fn spawn_camera(mut commands: Commands) {
    let translation = Vec3::new(-8.0, 6.0, 8.0);
    let focus = Vec3::ZERO;

    commands.spawn((
        Name::new("Viewer Camera"),
        Camera3d::default(),
        Transform::from_translation(translation).looking_at(focus, Vec3::Y),
        OrbitCamera::new(focus, translation),
    ));
}

fn control_orbit_camera(
    mut motion_events: MessageReader<MouseMotion>,
    mut scroll_events: MessageReader<MouseWheel>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut query: Query<(&mut Transform, &mut OrbitCamera)>,
) {
    let mut rotation_delta = Vec2::ZERO;
    for ev in motion_events.read() {
        rotation_delta += ev.delta;
    }
    if !mouse_buttons.pressed(MouseButton::Right) {
        rotation_delta = Vec2::ZERO;
    }

    let mut scroll = 0.0;
    for ev in scroll_events.read() {
        scroll += ev.y;
    }

    for (mut transform, mut orbit) in query.iter_mut() {
        if rotation_delta.length_squared() > 0.0 {
            orbit.yaw -= rotation_delta.x * ORBIT_SENSITIVITY;
            orbit.pitch =
                (orbit.pitch - rotation_delta.y * ORBIT_SENSITIVITY).clamp(MIN_PITCH, MAX_PITCH);
        }

        if scroll.abs() > f32::EPSILON {
            orbit.radius = (orbit.radius - scroll * ZOOM_SENSITIVITY)
                .clamp(orbit.min_radius, orbit.max_radius);
        }

        let (sin_pitch, cos_pitch) = orbit.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = orbit.yaw.sin_cos();

        let offset = Vec3::new(
            orbit.radius * cos_pitch * cos_yaw,
            orbit.radius * sin_pitch,
            orbit.radius * cos_pitch * sin_yaw,
        );

        transform.translation = orbit.focus + offset;
        transform.look_at(orbit.focus, Vec3::Y);
    }
}
