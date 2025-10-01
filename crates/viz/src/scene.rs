use bevy::math::primitives::Plane3d;
use bevy::prelude::*;

use crate::stream::{StreamMaterial, StreamTexture};

const PLANE_WIDTH: f32 = 12.0;
const PLANE_ASPECT_RATIO: f32 = 16.0 / 9.0;

pub fn spawn_environment(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    stream_texture: Res<StreamTexture>,
) {
    let plane_mesh = meshes.add(Mesh::from(
        Plane3d::default()
            .mesh()
            .size(PLANE_WIDTH, PLANE_WIDTH / PLANE_ASPECT_RATIO),
    ));

    let stream_material = materials.add(StandardMaterial {
        base_color_texture: Some(stream_texture.handle.clone()),
        base_color: Color::WHITE,
        unlit: true,
        cull_mode: None,
        ..default()
    });

    commands.insert_resource(StreamMaterial {
        handle: stream_material.clone(),
    });

    commands.spawn((
        Name::new("Camera Feed Plane"),
        Mesh3d(plane_mesh),
        MeshMaterial3d(stream_material.clone()),
        Transform::from_xyz(0.0, 0.0, 0.0).with_rotation(Quat::from_euler(
            EulerRot::YXZ,
            0.0,
            0.0,
            std::f32::consts::PI,
        )),
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::VISIBLE,
        ViewVisibility::default(),
    ));

    commands.spawn((
        Name::new("Key Light"),
        DirectionalLight {
            illuminance: 3000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(8.0, 10.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::VISIBLE,
        ViewVisibility::default(),
    ));

    commands.spawn((
        Name::new("Fill Light"),
        DirectionalLight {
            illuminance: 1500.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(-6.0, 6.0, -4.0).looking_at(Vec3::ZERO, Vec3::Y),
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::VISIBLE,
        ViewVisibility::default(),
    ));
}
