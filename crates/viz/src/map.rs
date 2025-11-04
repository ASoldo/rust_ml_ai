//! Static map loader that fetches a tile and exposes it as a Bevy texture.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use image::GenericImageView;
use image::imageops::FilterType;
use reqwest::blocking::Client;
use std::collections::HashMap;

/// Plugin responsible for fetching and registering the static map texture.
pub struct MapTexturePlugin;

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
/// Startup system ordering for map setup.
pub struct MapSetupSet;

#[derive(Resource, Clone)]
/// Configuration for the static map request. Values can be overridden through
/// environment variables.
pub struct StaticMapConfig {
    pub latitude: f64,
    pub longitude: f64,
    pub zoom: u8,
    pub size: u32,
    pub provider_url: Option<String>,
}

impl Default for StaticMapConfig {
    fn default() -> Self {
        let latitude = std::env::var("VIZ_MAP_LAT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(45.81298654949797);
        let longitude = std::env::var("VIZ_MAP_LON")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(15.977990737614029);
        let zoom = std::env::var("VIZ_MAP_ZOOM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(17);
        let size = std::env::var("VIZ_MAP_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        let provider_url = std::env::var("VIZ_MAP_URL").ok();

        Self {
            latitude,
            longitude,
            zoom,
            size,
            provider_url,
        }
    }
}

#[derive(Resource, Clone)]
/// Wrapper resource storing the map texture handle.
pub struct MapTexture {
    pub handle: Handle<Image>,
}

impl Plugin for MapTexturePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<StaticMapConfig>()
            .configure_sets(Startup, MapSetupSet)
            .add_systems(Startup, setup_map_texture.in_set(MapSetupSet));
    }
}

/// Fetch the map texture and insert it into the Bevy world.
fn setup_map_texture(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    config: Res<StaticMapConfig>,
) {
    let placeholder = Image::new_fill(
        Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[32, 48, 64, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    let handle = images.add(placeholder);

    match fetch_map_image(&config) {
        Ok(image) => {
            if let Some(existing) = images.get_mut(&handle) {
                *existing = image;
            }
            info!(
                "Loaded static map at ({}, {}) with zoom {}",
                config.latitude, config.longitude, config.zoom
            );
        }
        Err(err) => {
            warn!("Failed to fetch static map: {err}");
        }
    }

    commands.insert_resource(MapTexture { handle });
}

/// Download and resize the map tile according to the provided config.
fn fetch_map_image(config: &StaticMapConfig) -> Result<Image, String> {
    let url_template = config
        .provider_url
        .clone()
        .unwrap_or_else(|| "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png".into());
    let url = build_url_from_template(&url_template, config);

    let client = Client::new();
    let response = client
        .get(&url)
        .send()
        .map_err(|err| format!("request error: {err}"))?;
    if !response.status().is_success() {
        return Err(format!("map server responded with {}", response.status()));
    }
    let bytes = response
        .bytes()
        .map_err(|err| format!("failed to read map response: {err}"))?
        .to_vec();
    let mut dynamic = image::load_from_memory(&bytes)
        .map_err(|err| format!("failed to decode map image: {err}"))?;
    let target_size = config.size;
    if target_size > 0 {
        let (width, height) = dynamic.dimensions();
        if width != target_size || height != target_size {
            dynamic = dynamic.resize_exact(target_size, target_size, FilterType::Lanczos3);
        }
    }
    let rgba = dynamic.into_rgba8();
    let (width, height) = rgba.dimensions();

    let mut image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        rgba.into_raw(),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    image.copy_on_resize = false;
    Ok(image)
}

/// Interpolate a template URL using configuration values and slippy-map maths.
fn build_url_from_template(template: &str, config: &StaticMapConfig) -> String {
    let zoom = config.zoom.min(22);
    let n = 1_i64 << zoom;
    let x_raw = ((config.longitude + 180.0) / 360.0 * (n as f64)).floor() as i64;
    let lat_rad = config.latitude.to_radians();
    let y_raw = ((1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / std::f64::consts::PI) / 2.0
        * (n as f64))
        .floor() as i64;

    let x_tile = ((x_raw % n) + n) % n;
    let y_tile = y_raw.clamp(0, n - 1);
    let retina_suffix = if config.size >= 512 { "@2x" } else { "" };

    let mut values: HashMap<&str, String> = HashMap::new();
    values.insert("lat", config.latitude.to_string());
    values.insert("lon", config.longitude.to_string());
    values.insert("size", config.size.to_string());
    values.insert("z", zoom.to_string());
    values.insert("zoom", zoom.to_string());
    values.insert("x", x_tile.to_string());
    values.insert("y", y_tile.to_string());
    values.insert("r", retina_suffix.to_string());
    if template.contains("{s}") {
        values.insert("s", "a".to_string());
    }

    let mut url = template.to_string();
    for (key, value) in &values {
        url = url.replace(&format!("{{{key}}}"), value);
    }
    url
}
