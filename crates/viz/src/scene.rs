//! Scene graph assembly for the visualization environment.

use bevy::math::primitives::Plane3d;
use bevy::prelude::*;

use crate::map::MapTexture;
use crate::stream::{StreamMaterial, StreamTexture};

/// Width of the video plane in world units.
pub const PLANE_WIDTH: f32 = 12.0;
/// Aspect ratio for the video plane (16:9).
pub const PLANE_ASPECT_RATIO: f32 = 16.0 / 9.0;
/// Width of the ground map square.
pub const MAP_WIDTH: f32 = 120.0;

/// Origin of the camera rig.
pub const RIG_ROOT: Vec3 = Vec3::ZERO;
/// Base platform height.
pub const RIG_HEIGHT: f32 = 2.2;
/// Additional mast height above the platform.
pub const CAMERA_MAST_EXTRA: f32 = 3.5;
/// Total camera head height.
pub const CAMERA_HEAD_HEIGHT: f32 = RIG_HEIGHT + CAMERA_MAST_EXTRA;
/// Distance between the camera head and the video plane.
pub const PLANE_DISTANCE: f32 = 3.6;

#[derive(Component)]
/// Marker component tagging the plane that displays the live stream.
pub struct CameraFeedPlane;

/// Create the 3D environment, map, lighting, and camera feed plane.
pub fn spawn_environment(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    stream_texture: Res<StreamTexture>,
    map_texture: Res<MapTexture>,
) {
    let map_mesh = meshes.add(Mesh::from(
        Plane3d::default().mesh().size(MAP_WIDTH, MAP_WIDTH),
    ));
    let map_material = materials.add(StandardMaterial {
        base_color_texture: Some(map_texture.handle.clone()),
        base_color: Color::WHITE,
        perceptual_roughness: 1.0,
        metallic: 0.0,
        unlit: true,
        cull_mode: None,
        ..default()
    });

    commands.spawn((
        Name::new("Ground Map"),
        Mesh3d(map_mesh),
        MeshMaterial3d(map_material),
        Transform::from_translation(RIG_ROOT + Vec3::new(0.0, -0.01, 0.0)),
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::VISIBLE,
        ViewVisibility::default(),
    ));

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

    let rig_tip = RIG_ROOT + Vec3::Y * CAMERA_HEAD_HEIGHT;
    let plane_center = RIG_ROOT + Vec3::new(0.0, CAMERA_HEAD_HEIGHT, -PLANE_DISTANCE);
    let plane_normal = (rig_tip - plane_center).normalize_or_zero();
    let plane_rotation = if plane_normal.length_squared() > 0.0 {
        Quat::from_rotation_arc(Vec3::Y, plane_normal)
    } else {
        Quat::IDENTITY
    };
    let mut plane_transform =
        Transform::from_translation(plane_center).with_rotation(plane_rotation);
    plane_transform.scale.x *= -1.0;

    commands.spawn((
        Name::new("Camera Feed Plane"),
        Mesh3d(plane_mesh),
        MeshMaterial3d(stream_material.clone()),
        plane_transform,
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::VISIBLE,
        ViewVisibility::default(),
        CameraFeedPlane,
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
