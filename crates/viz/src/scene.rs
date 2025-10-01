use bevy::asset::RenderAssetUsages;
use bevy::math::primitives::Cuboid;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::stream::{StreamMaterial, StreamTexture};

const PLANE_WIDTH: f32 = 10.0;
const PLANE_ASPECT_RATIO: f32 = 16.0 / 9.0;

pub fn spawn_environment(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    stream_texture: Res<StreamTexture>,
) {
    let cube_mesh = meshes.add(Cuboid::new(
        PLANE_WIDTH,
        PLANE_WIDTH / PLANE_ASPECT_RATIO,
        0.2,
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
        Name::new("Camera Feed Cube"),
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(stream_material),
        Transform::from_xyz(0.0, 0.0, 0.0).with_rotation(Quat::from_euler(
            EulerRot::YXZ,
            0.5,
            0.3,
            0.0,
        )),
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::VISIBLE,
        ViewVisibility::default(),
    ));

    let (width, height) = (256u32, 256u32);
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            data.push(x as u8);
            data.push(y as u8);
            data.push(((x + y) % 256) as u8);
            data.push(255);
        }
    }

    let gradient_image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );

    let gradient_handle = images.add(gradient_image);
    let gradient_material = materials.add(StandardMaterial {
        base_color_texture: Some(gradient_handle),
        base_color: Color::WHITE,
        unlit: true,
        cull_mode: None,
        ..default()
    });

    commands.spawn((
        Name::new("Gradient Cube"),
        Mesh3d(cube_mesh),
        MeshMaterial3d(gradient_material),
        Transform::from_xyz(14.0, 0.0, 0.0).with_rotation(Quat::from_euler(
            EulerRot::YXZ,
            -0.4,
            -0.2,
            0.0,
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
