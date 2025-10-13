use bevy::asset::{AssetMode, AssetPlugin};
use bevy::prelude::*;
use bevy::window::{PresentMode, Window, WindowPlugin, WindowResolution};

mod gizmos;
mod map;
mod orbit_camera;
mod scene;
mod stream;

pub use orbit_camera::OrbitCameraPlugin;

pub struct VizAppPlugin;

impl Plugin for VizAppPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ClearColor(Color::srgb(0.02, 0.02, 0.05)))
            .insert_resource(AmbientLight {
                color: Color::srgb(0.7, 0.7, 0.8),
                brightness: 350.0,
                affects_lightmapped_meshes: true,
            })
            .add_plugins(
                DefaultPlugins
                    .set(WindowPlugin {
                        primary_window: Some(Window {
                            title: "Viz Sensor Viewer".into(),
                            resolution: WindowResolution::new(1440, 900),
                            present_mode: PresentMode::AutoVsync,
                            ..default()
                        }),
                        ..default()
                    })
                    .set(AssetPlugin {
                        mode: AssetMode::Unprocessed,
                        ..default()
                    }),
            )
            .add_plugins(stream::StreamTexturePlugin)
            .add_plugins(map::MapTexturePlugin)
            .add_plugins(gizmos::CameraRigGizmosPlugin)
            .add_plugins(OrbitCameraPlugin)
            .add_systems(
                Startup,
                scene::spawn_environment
                    .after(stream::StreamSetupSet)
                    .after(map::MapSetupSet),
            )
            .add_systems(Update, stream::debug_texture_sample);
    }
}

/// Launches the standalone Bevy visualization app.
pub fn run() {
    App::new().add_plugins(VizAppPlugin).run();
}
