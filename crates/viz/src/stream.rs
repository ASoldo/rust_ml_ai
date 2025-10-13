use std::fs::File;
use std::io::{BufReader, Read};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::Duration;

use bevy::asset::RenderAssetUsages;
use bevy::log::prelude::*;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use reqwest::blocking::{Client, Response};

const CHUNK_SIZE: usize = 64 * 1024;
static FIRST_FRAME_DUMPED: AtomicBool = AtomicBool::new(false);

pub struct StreamTexturePlugin;

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub struct StreamSetupSet;

#[derive(Resource, Clone)]
pub struct MjpegStreamConfig {
    pub source: String,
    pub reopen_delay: Duration,
    pub idle_sleep: Duration,
}

impl Default for MjpegStreamConfig {
    fn default() -> Self {
        let source = std::env::var("VIZ_STREAM_PATH")
            .unwrap_or_else(|_| "http://127.0.0.1:8080/stream.mjpg".into());
        Self {
            source,
            reopen_delay: Duration::from_secs(1),
            idle_sleep: Duration::from_millis(16),
        }
    }
}

#[derive(Resource, Clone)]
pub struct StreamTexture {
    pub handle: Handle<Image>,
}

#[derive(Resource, Clone)]
pub struct StreamMaterial {
    pub handle: Handle<StandardMaterial>,
}

#[derive(Resource)]
struct FrameReceiver {
    rx: Mutex<Receiver<DecodedFrame>>,
}

struct DecodedFrame {
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    frame_len: usize,
}

impl Plugin for StreamTexturePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MjpegStreamConfig>()
            .configure_sets(Startup, StreamSetupSet)
            .add_systems(Startup, setup_stream_texture.in_set(StreamSetupSet))
            .add_systems(Update, update_stream_texture);
    }
}

fn setup_stream_texture(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    config: Res<MjpegStreamConfig>,
) {
    let placeholder = Image::new_fill(
        Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );

    let image_handle = images.add(placeholder);
    let (raw_tx, raw_rx) = mpsc::channel();
    let (decoded_tx, decoded_rx) = mpsc::channel();

    commands.insert_resource(StreamTexture {
        handle: image_handle.clone(),
    });
    commands.insert_resource(FrameReceiver {
        rx: Mutex::new(decoded_rx),
    });

    let reader_config = config.clone();
    thread::Builder::new()
        .name("mjpeg-stream-reader".into())
        .spawn(move || run_stream_reader(reader_config, raw_tx))
        .expect("failed to spawn mjpeg stream reader thread");

    thread::Builder::new()
        .name("mjpeg-frame-decoder".into())
        .spawn(move || run_frame_decoder(raw_rx, decoded_tx))
        .expect("failed to spawn mjpeg frame decoder thread");
}

pub fn debug_texture_sample(
    stream_texture: Option<Res<StreamTexture>>,
    images: Res<Assets<Image>>,
) {
    let Some(stream_texture) = stream_texture else {
        return;
    };
    if let Some(image) = images.get(&stream_texture.handle) {
        if let Some(data) = &image.data {
            if data.len() >= 4 {
                trace!(
                    "sample pixel rgba=({}, {}, {}, {}) size={:?}",
                    data[0], data[1], data[2], data[3], image.texture_descriptor.size
                );
            }
        }
    }
}

fn update_stream_texture(
    mut images: ResMut<Assets<Image>>,
    stream_texture: Option<Res<StreamTexture>>,
    receiver: Option<Res<FrameReceiver>>,
    stream_material: Option<Res<StreamMaterial>>,
    mut materials: Option<ResMut<Assets<StandardMaterial>>>,
) {
    let (Some(stream_texture), Some(receiver)) = (stream_texture, receiver) else {
        return;
    };

    let mut latest_frame: Option<DecodedFrame> = None;

    let guard = match receiver.rx.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            warn!(error = %poisoned, "MJPEG receiver mutex poisoned");
            return;
        }
    };

    loop {
        match guard.try_recv() {
            Ok(frame) => latest_frame = Some(frame),
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                warn!("MJPEG stream disconnected");
                return;
            }
        }
    }

    drop(guard);

    let Some(frame) = latest_frame else {
        return;
    };

    apply_frame_to_image(
        &mut images,
        &stream_texture.handle,
        frame.width,
        frame.height,
        frame.rgba,
        frame.frame_len,
    );

    if let (Some(stream_material), Some(materials)) = (stream_material, materials.as_mut()) {
        if let Some(material) = materials.get_mut(&stream_material.handle) {
            if material.base_color_texture.is_none() {
                material.base_color_texture = Some(stream_texture.handle.clone());
            }
        }
    }
}

fn run_frame_decoder(rx: Receiver<Vec<u8>>, tx: Sender<DecodedFrame>) {
    while let Ok(mut frame) = rx.recv() {
        while let Ok(newer) = rx.try_recv() {
            frame = newer;
        }

        let frame_len = frame.len();
        match image::load_from_memory(&frame) {
            Ok(dynamic) => {
                if !FIRST_FRAME_DUMPED.swap(true, Ordering::Relaxed) {
                    if let Err(err) = std::fs::write("viz_first_frame.jpg", &frame) {
                        warn!("Failed to dump first MJPEG frame: {err}");
                    } else {
                        info!(
                            "Dumped first MJPEG frame ({} bytes) to viz_first_frame.jpg",
                            frame_len
                        );
                    }
                }

                let rgba = dynamic.into_rgba8();
                let (width, height) = rgba.dimensions();
                let data = rgba.into_raw();

                if tx
                    .send(DecodedFrame {
                        rgba: data,
                        width,
                        height,
                        frame_len,
                    })
                    .is_err()
                {
                    return;
                }
            }
            Err(err) => warn!(
                "Failed to decode MJPEG frame ({} bytes): {err}",
                frame_len,
                err = err
            ),
        }
    }
}

fn apply_frame_to_image(
    images: &mut Assets<Image>,
    handle: &Handle<Image>,
    width: u32,
    height: u32,
    data: Vec<u8>,
    frame_len: usize,
) {
    use bevy::render::render_resource::TextureUsages;

    let extent = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let expected_len = (width as usize) * (height as usize) * 4;

    if let Some(image) = images.get_mut(handle) {
        let needs_rebuild = image.texture_descriptor.size != extent
            || image.texture_descriptor.dimension != TextureDimension::D2
            || image.texture_descriptor.format != TextureFormat::Rgba8UnormSrgb
            || image
                .data
                .as_ref()
                .map(|buffer| buffer.len() != expected_len)
                .unwrap_or(true);

        if needs_rebuild {
            let mut replacement = Image::new(
                extent,
                TextureDimension::D2,
                data,
                TextureFormat::Rgba8UnormSrgb,
                RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
            );
            replacement.texture_descriptor.usage =
                TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
            replacement.copy_on_resize = false;
            *image = replacement;
        } else if let Some(existing) = image.data.as_mut() {
            existing.copy_from_slice(&data);
        } else {
            image.data = Some(data);
        }

        image.asset_usage = RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD;
        image.texture_descriptor.size = extent;
        image.texture_descriptor.dimension = TextureDimension::D2;
        image.texture_descriptor.format = TextureFormat::Rgba8UnormSrgb;
        image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
        image.copy_on_resize = false;

        trace!(
            "Updated stream texture from MJPEG frame ({} bytes) → size {}x{}, data {} bytes",
            frame_len, width, height, expected_len
        );
    } else {
        let mut image = Image::new(
            extent,
            TextureDimension::D2,
            data,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
        );
        image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
        image.copy_on_resize = false;

        if let Err(err) = images.insert(handle.id(), image) {
            warn!("Failed to initialize stream texture: {err:?}");
        }
    }
}

fn run_stream_reader(config: MjpegStreamConfig, tx: Sender<Vec<u8>>) {
    let is_http = config.source.starts_with("http://") || config.source.starts_with("https://");
    let client = if is_http {
        match Client::builder().build() {
            Ok(client) => Some(client),
            Err(err) => {
                warn!("Failed to build HTTP client for {}: {err}", config.source);
                return;
            }
        }
    } else {
        None
    };

    loop {
        if let Some(client) = client.as_ref() {
            match client.get(&config.source).send() {
                Ok(response) => {
                    info!("Connected to MJPEG stream {}", config.source);
                    stream_from_http(response, &tx, &config);
                    info!("MJPEG stream {} ended, retrying", config.source);
                }
                Err(err) => {
                    warn!("Waiting for MJPEG stream {}: {err}", config.source);
                    thread::sleep(config.reopen_delay);
                }
            }
            continue;
        }

        match File::open(&config.source) {
            Ok(file) => {
                info!("Opened MJPEG stream {}", config.source);
                stream_from_file(file, &tx, &config);
                info!("MJPEG stream {} ended, retrying", config.source);
            }
            Err(err) => {
                warn!("Waiting for MJPEG stream {}: {err}", config.source);
                thread::sleep(config.reopen_delay);
            }
        }
    }
}

fn stream_from_file(file: File, tx: &Sender<Vec<u8>>, config: &MjpegStreamConfig) {
    let reader = BufReader::new(file);
    stream_from_reader(reader, tx, config);
}

fn stream_from_http(response: Response, tx: &Sender<Vec<u8>>, config: &MjpegStreamConfig) {
    if !response.status().is_success() {
        warn!(
            "MJPEG stream {} responded with status {}",
            config.source,
            response.status()
        );
        thread::sleep(config.reopen_delay);
        return;
    }

    let reader = BufReader::new(response);
    stream_from_reader(reader, tx, config);
}

fn stream_from_reader<R: Read>(mut reader: R, tx: &Sender<Vec<u8>>, config: &MjpegStreamConfig) {
    let mut buffer = Vec::with_capacity(CHUNK_SIZE * 2);
    let mut chunk = [0u8; CHUNK_SIZE];

    loop {
        match reader.read(&mut chunk) {
            Ok(0) => {
                trace!("MJPEG read returned 0 bytes");
                thread::sleep(config.idle_sleep);
                return;
            }
            Ok(n) => {
                trace!("MJPEG read {} bytes", n);
                buffer.extend_from_slice(&chunk[..n]);
                while let Some(frame) = extract_frame(&mut buffer) {
                    trace!("Extracted MJPEG frame of {} bytes", frame.len());
                    if tx.send(frame).is_err() {
                        return;
                    }
                }
            }
            Err(err) => {
                warn!("MJPEG read error: {err}");
                thread::sleep(config.reopen_delay);
                return;
            }
        }
    }
}

fn extract_frame(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
    let Some(start) = find_marker(buffer, &[0xFF, 0xD8]) else {
        if buffer.len() > CHUNK_SIZE {
            buffer.clear();
        }
        return None;
    };

    if start > 0 {
        buffer.drain(..start);
    }

    let Some(end) = find_marker(buffer, &[0xFF, 0xD9]) else {
        return None;
    };

    let frame_end = (end + 2).min(buffer.len());
    let frame = buffer[..frame_end].to_vec();
    buffer.drain(..frame_end);
    Some(frame)
}

fn find_marker(buffer: &[u8], marker: &[u8]) -> Option<usize> {
    buffer
        .windows(marker.len())
        .position(|window| window == marker)
}
