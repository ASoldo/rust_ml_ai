use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    sync::{Arc, Mutex, Once},
    time::Duration,
};

use actix_web::web::Bytes;
use actix_web::{App, HttpResponse, HttpServer, http::header, web};
use anyhow::{Context, Result, anyhow, bail};
use async_stream::stream;
use crossbeam_channel::{Receiver, Sender, TrySendError};
use gpu_kernels::VisionRuntime;
use image::{DynamicImage, ImageBuffer, Rgba, codecs::jpeg::JpegEncoder};
use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
use ml_core::{
    detector::Detector,
    tch::{Cuda, Device},
};
use serde_json::to_string;
use tracing::{debug, error, info, warn};
use video_ingest::{self, Frame, FrameFormat};

#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub camera_uri: String,
    pub model_path: PathBuf,
    pub width: i32,
    pub height: i32,
    pub verbose: bool,
    pub use_cpu: bool,
    pub use_nvdec: bool,
    pub detector_width: i32,
    pub detector_height: i32,
    pub jpeg_quality: i32,
}

impl VisionConfig {
    pub fn from_args(args: &[String]) -> Result<Self> {
        if args.len() < 6 {
            bail!(
                "Usage: cargo run -p cuda-app --features with-tch -- vision <camera-uri> <model-path> <width> <height> [--cpu] [--nvdec] [--verbose] [--detector-width <px>] [--detector-height <px>] [--jpeg-quality <1-100>]"
            );
        }

        let camera_uri = args[2].clone();
        let model_path = PathBuf::from(&args[3]);
        let width = args[4]
            .parse::<i32>()
            .with_context(|| "width must be an integer".to_string())?;
        let height = args[5]
            .parse::<i32>()
            .with_context(|| "height must be an integer".to_string())?;
        let mut verbose = false;
        let mut use_cpu = false;
        let mut use_nvdec = false;
        let mut detector_width = width;
        let mut detector_height = height;
        let mut jpeg_quality = 85;

        let mut idx = 6;
        while idx < args.len() {
            match args[idx].as_str() {
                "--verbose" => {
                    verbose = true;
                    idx += 1;
                }
                "--cpu" => {
                    use_cpu = true;
                    idx += 1;
                }
                "--nvdec" => {
                    use_nvdec = true;
                    idx += 1;
                }
                "--detector-width" => {
                    idx += 1;
                    let value = args
                        .get(idx)
                        .ok_or_else(|| anyhow!("--detector-width requires a value"))?
                        .parse::<i32>()
                        .with_context(|| {
                            "--detector-width must be a positive integer".to_string()
                        })?;
                    if value <= 0 {
                        bail!("--detector-width must be a positive integer");
                    }
                    detector_width = value;
                    idx += 1;
                }
                "--detector-height" => {
                    idx += 1;
                    let value = args
                        .get(idx)
                        .ok_or_else(|| anyhow!("--detector-height requires a value"))?
                        .parse::<i32>()
                        .with_context(|| {
                            "--detector-height must be a positive integer".to_string()
                        })?;
                    if value <= 0 {
                        bail!("--detector-height must be a positive integer");
                    }
                    detector_height = value;
                    idx += 1;
                }
                "--jpeg-quality" => {
                    idx += 1;
                    let value = args
                        .get(idx)
                        .ok_or_else(|| anyhow!("--jpeg-quality requires a value"))?
                        .parse::<i32>()
                        .with_context(|| {
                            "--jpeg-quality must be an integer between 1 and 100".to_string()
                        })?;
                    if !(1..=100).contains(&value) {
                        bail!("--jpeg-quality must be an integer between 1 and 100");
                    }
                    jpeg_quality = value;
                    idx += 1;
                }
                other => bail!("Unrecognised flag: {other}"),
            }
        }

        if use_cpu && use_nvdec {
            bail!("--cpu and --nvdec are mutually exclusive");
        }

        Ok(Self {
            camera_uri,
            model_path,
            width,
            height,
            verbose,
            use_cpu,
            use_nvdec,
            detector_width,
            detector_height,
            jpeg_quality,
        })
    }
}

pub fn run_from_args(args: &[String]) -> Result<()> {
    let config = VisionConfig::from_args(args)?;
    run(config)
}

pub fn run(config: VisionConfig) -> Result<()> {
    if !config.use_cpu {
        load_torch_cuda_runtime(config.verbose);
    }

    let device = if config.use_cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available()
    };

    let cuda_available = Cuda::is_available();
    let cuda_devices = Cuda::device_count();
    info!(
        "CUDA available: {} (devices: {})",
        cuda_available, cuda_devices
    );

    let receiver = if config.use_nvdec {
        video_ingest::spawn_nvdec_h264_reader(&config.camera_uri, (config.width, config.height))
            .with_context(|| "Failed to start NVDEC capture".to_string())?
    } else {
        video_ingest::spawn_camera_reader(&config.camera_uri, (config.width, config.height))
            .with_context(|| "Failed to start capture".to_string())?
    };

    let shared: SharedFrame = Arc::new(Mutex::new(None));
    let tracker = Arc::new(Mutex::new(SimpleTracker::default()));
    let (work_tx, work_rx) = crossbeam_channel::bounded::<FrameTask>(3);
    let (encode_tx, encode_rx) = crossbeam_channel::bounded::<EncodeJob>(3);

    let detector_init = DetectorInit {
        model_path: config.model_path.clone(),
        device,
        input_size: (config.detector_width as i64, config.detector_height as i64),
    };

    let (init_tx, init_rx) = crossbeam_channel::bounded::<std::result::Result<String, String>>(1);
    let encode_handle = spawn_encode_worker(shared.clone(), encode_rx);
    let processing_handle = spawn_processing_worker(
        detector_init,
        tracker.clone(),
        work_rx,
        config.verbose,
        config.jpeg_quality,
        init_tx,
        encode_tx.clone(),
    );

    match init_rx.recv() {
        Ok(Ok(message)) => info!("{message}"),
        Ok(Err(err)) => {
            error!("{err}");
            drop(work_tx);
            drop(encode_tx);
            let _ = processing_handle.join();
            let _ = encode_handle.join();
            bail!(err);
        }
        Err(err) => {
            drop(work_tx);
            drop(encode_tx);
            let _ = processing_handle.join();
            let _ = encode_handle.join();
            bail!("Processing thread failed to initialise detector: {err}");
        }
    }

    spawn_preview_server(shared.clone()).context("Failed to start preview server")?;

    info!("HTTP preview available at http://127.0.0.1:8080/frame.jpg and /stream.mjpg");

    let running = Arc::new(AtomicBool::new(true));
    {
        let running = running.clone();
        if let Err(err) = ctrlc::set_handler(move || {
            running.store(false, Ordering::SeqCst);
        }) {
            warn!("Failed to install Ctrl+C handler: {err}");
        }
    }

    if config.verbose {
        info!("Running vision pipeline — press Ctrl+C to stop");
    }

    let mut frame_number: u64 = 0;
    let mut smoothed_fps: f32 = 0.0;
    let mut last_instant = std::time::Instant::now();
    let mut dropped_frames: u64 = 0;

    while running.load(Ordering::Relaxed) {
        match receiver.recv() {
            Ok(frame) => match frame {
                Ok(frame) => {
                    frame_number = frame_number.wrapping_add(1);
                    let now = std::time::Instant::now();
                    let elapsed = now.duration_since(last_instant).as_secs_f32();
                    last_instant = now;
                    if elapsed > 0.0 {
                        let instant = 1.0 / elapsed;
                        smoothed_fps = if smoothed_fps == 0.0 {
                            instant
                        } else {
                            0.9 * smoothed_fps + 0.1 * instant
                        };
                    }

                    let task = FrameTask {
                        frame,
                        frame_number,
                        fps: smoothed_fps,
                    };
                    match work_tx.try_send(task) {
                        Ok(()) => {}
                        Err(TrySendError::Full(_)) => {
                            dropped_frames = dropped_frames.wrapping_add(1);
                            if config.verbose {
                                warn!(
                                    "Dropping frame #{frame_number} (processing backlog, dropped total: {})",
                                    dropped_frames
                                );
                            }
                        }
                        Err(TrySendError::Disconnected(_)) => {
                            error!("Processing thread terminated unexpectedly");
                            break;
                        }
                    }
                }
                Err(err) => {
                    error!("Capture error: {err}");
                    break;
                }
            },
            Err(err) => {
                error!("Frame channel closed: {err}");
                break;
            }
        }
    }

    info!("Stopping vision pipeline");

    drop(work_tx);
    let _ = processing_handle.join();
    drop(encode_tx);
    let _ = encode_handle.join();

    Ok(())
}

type SharedFrame = Arc<Mutex<Option<FramePacket>>>;

struct ServerState {
    latest: SharedFrame,
}

#[derive(Clone)]
struct FramePacket {
    jpeg: Vec<u8>,
    detections: Vec<DetectionSummary>,
    timestamp_ms: i64,
    frame_number: u64,
    fps: f32,
}

#[derive(Clone, serde::Serialize)]
struct DetectionSummary {
    class: String,
    score: f32,
    bbox: [f32; 4],
    track_id: i64,
}

#[derive(serde::Serialize)]
struct DetectionsResponse<'a> {
    timestamp_ms: i64,
    frame_number: u64,
    fps: f32,
    detections: &'a [DetectionSummary],
}

#[derive(Default)]
struct SimpleTracker {
    next_id: i64,
}

struct FrameTask {
    frame: Frame,
    frame_number: u64,
    fps: f32,
}

struct DetectorInit {
    model_path: PathBuf,
    device: Device,
    input_size: (i64, i64),
}

struct GpuEncodeJob {
    runtime: Arc<Mutex<VisionRuntime>>,
    width: i32,
    height: i32,
    summaries: Vec<DetectionSummary>,
    timestamp_ms: i64,
    frame_number: u64,
    fps: f32,
    jpeg_quality: i32,
}

enum EncodeJob {
    Cpu(FramePacket),
    Gpu(GpuEncodeJob),
}

fn spawn_processing_worker(
    detector_init: DetectorInit,
    tracker: Arc<Mutex<SimpleTracker>>,
    work_rx: Receiver<FrameTask>,
    verbose: bool,
    jpeg_quality: i32,
    init_tx: Sender<std::result::Result<String, String>>,
    encode_tx: Sender<EncodeJob>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let detector = match Detector::new(
            &detector_init.model_path,
            detector_init.device,
            detector_init.input_size,
        ) {
            Ok(det) => {
                let message = format!(
                    "Detector loaded on {:?} (vision runtime enabled: {})",
                    det.device(),
                    det.uses_gpu_runtime()
                );
                let _ = init_tx.send(Ok(message));
                det
            }
            Err(err) => {
                let _ = init_tx.send(Err(format!("Failed to load detector: {err}")));
                return;
            }
        };
        let vision_runtime = detector.vision_runtime();
        for task in work_rx {
            match process_frame(
                task.frame_number,
                task.fps,
                &detector,
                &task.frame,
                &tracker,
                verbose,
                jpeg_quality,
                vision_runtime.clone(),
            ) {
                Ok(job) => {
                    if encode_tx.send(job).is_err() {
                        error!("Encode channel closed, stopping processing worker");
                        break;
                    }
                }
                Err(err) => error!("Frame processing error: {err}"),
            }
        }
    })
}

fn spawn_encode_worker(
    shared: SharedFrame,
    encode_rx: Receiver<EncodeJob>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        for job in encode_rx {
            let packet_result = match job {
                EncodeJob::Cpu(packet) => Ok(packet),
                EncodeJob::Gpu(task) => encode_gpu_frame(task),
            };

            match packet_result {
                Ok(packet) => {
                    if let Ok(mut guard) = shared.lock() {
                        *guard = Some(packet);
                    }
                }
                Err(err) => error!("Encode stage error: {err}"),
            }
        }
    })
}

fn process_frame(
    frame_number: u64,
    fps: f32,
    detector: &Detector,
    frame: &Frame,
    tracker: &Arc<Mutex<SimpleTracker>>,
    verbose: bool,
    jpeg_quality: i32,
    vision: Option<Arc<Mutex<VisionRuntime>>>,
) -> Result<EncodeJob> {
    if !matches!(frame.format, FrameFormat::Bgr8) {
        bail!("unsupported frame format");
    }
    let tensor = detector
        .bgr_to_tensor(&frame.data, frame.width, frame.height)
        .with_context(|| "Failed to prepare tensor from frame")?;
    let detections = detector
        .infer(&tensor)
        .with_context(|| "Detector inference failed")?;
    if verbose {
        if detections.detections.is_empty() {
            debug!("frame #{frame_number}: no detections");
        } else {
            debug!(
                "frame #{frame_number}: {} detection(s)",
                detections.detections.len()
            );
            for (idx, det) in detections.detections.iter().enumerate() {
                debug!(
                    "  #{idx}: class={} conf={:.3} bbox={:?}",
                    det.class_id, det.score, det.bbox
                );
            }
        }
    }
    let mut summaries = Vec::with_capacity(detections.detections.len());
    let mut label_positions = Vec::with_capacity(detections.detections.len());
    let mut boxes_px = Vec::with_capacity(detections.detections.len());
    let (detector_w, detector_h) = detector.input_size();
    let scale_x = if detector_w > 0 {
        frame.width as f32 / detector_w as f32
    } else {
        1.0
    };
    let scale_y = if detector_h > 0 {
        frame.height as f32 / detector_h as f32
    } else {
        1.0
    };

    for det in &detections.detections {
        let left = (det.bbox[0] * scale_x).clamp(0.0, (frame.width - 1) as f32);
        let top = (det.bbox[1] * scale_y).clamp(0.0, (frame.height - 1) as f32);
        let right = (det.bbox[2] * scale_x).clamp(0.0, (frame.width - 1) as f32);
        let bottom = (det.bbox[3] * scale_y).clamp(0.0, (frame.height - 1) as f32);

        let left_i = left.round() as i32;
        let top_i = top.round() as i32;
        let right_i = right.round() as i32;
        let bottom_i = bottom.round() as i32;

        boxes_px.push([left_i, top_i, right_i, bottom_i]);
        label_positions.push((left_i, (top_i - 12).max(0)));

        let label = match det.class_id {
            0 => "FACE",
            1 => "PERSON",
            _ => "OBJECT",
        };

        summaries.push(DetectionSummary {
            class: label.to_string(),
            score: det.score,
            bbox: [left, top, right, bottom],
            track_id: 0,
        });
    }

    assign_tracks(tracker, &mut summaries);

    if let Some(runtime) = vision {
        let labels: Vec<String> = summaries
            .iter()
            .map(|summary| {
                format!(
                    "{} {} {:.0}%",
                    summary.class,
                    summary.track_id,
                    summary.score * 100.0
                )
            })
            .collect();
        Ok(EncodeJob::Gpu(annotate_frame_gpu(
            &runtime,
            frame,
            frame_number,
            fps,
            summaries.clone(),
            &boxes_px,
            &label_positions,
            &labels,
            jpeg_quality,
        )?))
    } else {
        let packet = annotate_frame_cpu(frame, frame_number, fps, summaries, jpeg_quality)?;
        Ok(EncodeJob::Cpu(packet))
    }
}

fn spawn_preview_server(shared: SharedFrame) -> Result<()> {
    let server_shared = shared.clone();
    std::thread::Builder::new()
        .name("vision-preview-server".into())
        .spawn(move || {
            if let Err(err) = actix_web::rt::System::new().block_on(async move {
                HttpServer::new(move || {
                    App::new()
                        .app_data(web::Data::new(ServerState {
                            latest: server_shared.clone(),
                        }))
                        .route("/atak", web::get().to(atak_route))
                        .route("/", web::get().to(index_route))
                        .route("/frame.jpg", web::get().to(frame_handler))
                        .route("/stream.mjpg", web::get().to(stream_handler))
                        .route("/detections", web::get().to(detections_handler))
                        .route(
                            "/stream_detections",
                            web::get().to(stream_detections_handler),
                        )
                })
                .bind(("0.0.0.0", 8080))?
                .run()
                .await
            }) {
                error!("HTTP server error: {err}");
            }
        })
        .context("Failed to spawn preview server thread")?;
    Ok(())
}

async fn frame_handler(state: web::Data<ServerState>) -> HttpResponse {
    let guard = match state.latest.lock() {
        Ok(guard) => guard,
        Err(err) => return HttpResponse::InternalServerError().body(err.to_string()),
    };
    if let Some(ref packet) = *guard {
        HttpResponse::Ok()
            .content_type("image/jpeg")
            .body(packet.jpeg.clone())
    } else {
        HttpResponse::NoContent().finish()
    }
}

async fn stream_handler(state: web::Data<ServerState>) -> HttpResponse {
    let state = state.clone();
    let stream = stream! {
        let mut interval = actix_web::rt::time::interval(Duration::from_millis(33));
        loop {
            interval.tick().await;
            let frame = state
                .latest
                .lock()
                .ok()
                .and_then(|guard| guard.clone());
            if let Some(packet) = frame {
                let mut payload = Vec::with_capacity(packet.jpeg.len() + 64);
                payload.extend_from_slice(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n");
                payload.extend_from_slice(&packet.jpeg);
                payload.extend_from_slice(b"\r\n");
                yield Ok::<Bytes, actix_web::Error>(Bytes::from(payload));
            }
        }
    };

    HttpResponse::Ok()
        .insert_header((header::ACCESS_CONTROL_ALLOW_ORIGIN, "*"))
        .insert_header((header::ACCESS_CONTROL_ALLOW_HEADERS, "*"))
        .insert_header((header::ACCESS_CONTROL_ALLOW_METHODS, "GET"))
        .insert_header((header::ACCESS_CONTROL_EXPOSE_HEADERS, "Content-Type"))
        .append_header(("Cache-Control", "no-cache"))
        .append_header(("Content-Type", "multipart/x-mixed-replace; boundary=frame"))
        .streaming(stream)
}

async fn index_route() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(crate::hud_html::HUD_INDEX_HTML)
}

async fn atak_route() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(crate::atak::HUD_ATAK_HTML)
}

async fn detections_handler(state: web::Data<ServerState>) -> HttpResponse {
    let guard = match state.latest.lock() {
        Ok(guard) => guard,
        Err(err) => return HttpResponse::InternalServerError().body(err.to_string()),
    };
    if let Some(ref packet) = *guard {
        HttpResponse::Ok().json(DetectionsResponse {
            timestamp_ms: packet.timestamp_ms,
            frame_number: packet.frame_number,
            fps: packet.fps,
            detections: &packet.detections,
        })
    } else {
        HttpResponse::NoContent().finish()
    }
}

async fn stream_detections_handler(state: web::Data<ServerState>) -> HttpResponse {
    let state = state.clone();
    let stream = stream! {
        let mut interval = actix_web::rt::time::interval(Duration::from_millis(250));
        loop {
            interval.tick().await;
            let snapshot = state
                .latest
                .lock()
                .ok()
                .and_then(|guard| guard.clone());
            if let Some(packet) = snapshot {
                let payload = DetectionsResponse {
                    timestamp_ms: packet.timestamp_ms,
                    frame_number: packet.frame_number,
                    fps: packet.fps,
                    detections: &packet.detections,
                };
                match to_string(&payload) {
                    Ok(json) => {
                        let mut sse_chunk = String::with_capacity(json.len() + 7);
                        sse_chunk.push_str("data: ");
                        sse_chunk.push_str(&json);
                        sse_chunk.push_str("\n\n");
                        yield Ok::<Bytes, actix_web::Error>(Bytes::from(sse_chunk));
                    }
                    Err(err) => {
                        let error_chunk = format!("event: error\ndata: {}\n\n", err);
                        yield Ok::<Bytes, actix_web::Error>(Bytes::from(error_chunk));
                    }
                }
            } else {
                yield Ok::<Bytes, actix_web::Error>(Bytes::from_static(b": keep-alive\n\n"));
            }
        }
    };

    HttpResponse::Ok()
        .insert_header((header::ACCESS_CONTROL_ALLOW_ORIGIN, "*"))
        .insert_header((header::ACCESS_CONTROL_ALLOW_HEADERS, "*"))
        .insert_header((header::ACCESS_CONTROL_ALLOW_METHODS, "GET"))
        .insert_header((header::ACCESS_CONTROL_EXPOSE_HEADERS, "Content-Type"))
        .append_header(("Cache-Control", "no-cache"))
        .append_header(("Content-Type", "text/event-stream"))
        .append_header(("Connection", "keep-alive"))
        .streaming(stream)
}

fn annotate_frame_cpu(
    frame: &Frame,
    frame_number: u64,
    fps: f32,
    summaries: Vec<DetectionSummary>,
    jpeg_quality: i32,
) -> Result<FramePacket> {
    let width = frame.width as u32;
    let height = frame.height as u32;
    let rgba = bgr_to_rgba(&frame.data);
    let mut image = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_vec(width, height, rgba)
        .ok_or_else(|| anyhow!("failed to convert frame into image buffer"))?;

    for summary in &summaries {
        let left = summary.bbox[0].clamp(0.0, (width - 1) as f32);
        let top = summary.bbox[1].clamp(0.0, (height - 1) as f32);
        let right = summary.bbox[2].clamp(0.0, (width - 1) as f32);
        let bottom = summary.bbox[3].clamp(0.0, (height - 1) as f32);
        draw_rectangle(
            &mut image,
            left.round() as i32,
            top.round() as i32,
            right.round() as i32,
            bottom.round() as i32,
            Rgba([0, 255, 0, 255]),
        );
    }

    for summary in &summaries {
        let left = summary.bbox[0].clamp(0.0, (width - 1) as f32);
        let top = summary.bbox[1].clamp(0.0, (height - 1) as f32);
        let label_text = format!(
            "{} {} {:.0}%",
            summary.class,
            summary.track_id,
            summary.score * 100.0
        );
        let label_x = left.round() as i32;
        let label_y = (top.round() as i32 - 12).max(0);
        let text_width = label_text.chars().count() as i32 * 6;
        fill_rect(
            &mut image,
            label_x,
            label_y,
            label_x + text_width,
            label_y + 8,
            Rgba([0, 0, 0, 180]),
        );
        draw_label(
            &mut image,
            label_x,
            label_y,
            &label_text,
            Rgba([0, 255, 0, 255]),
        );
    }

    let info = format!("FRAME {:06}  FPS {:4.1}", frame_number, fps);
    let info_width = (info.chars().count() as i32 * 6).min(width as i32);
    let info_x = (width as i32 - info_width - 4).max(0);
    let info_y = (height as i32 - 12).max(0);
    fill_rect(
        &mut image,
        info_x,
        info_y,
        info_x + info_width + 4,
        info_y + 8,
        Rgba([0, 0, 0, 180]),
    );
    draw_label(
        &mut image,
        info_x + 2,
        info_y,
        &info,
        Rgba([255, 255, 255, 255]),
    );

    let rgb = DynamicImage::ImageRgba8(image).to_rgb8();
    let mut buffer = Vec::new();
    let quality = jpeg_quality.clamp(1, 100) as u8;
    JpegEncoder::new_with_quality(&mut buffer, quality)
        .encode_image(&rgb)
        .map_err(|err| anyhow!("JPEG encode failed: {err}"))?;

    Ok(FramePacket {
        jpeg: buffer,
        detections: summaries,
        timestamp_ms: frame.timestamp_ms,
        frame_number,
        fps,
    })
}

fn annotate_frame_gpu(
    runtime: &Arc<Mutex<VisionRuntime>>,
    frame: &Frame,
    frame_number: u64,
    fps: f32,
    summaries: Vec<DetectionSummary>,
    boxes_px: &[[i32; 4]],
    label_positions: &[(i32, i32)],
    labels: &[String],
    jpeg_quality: i32,
) -> Result<GpuEncodeJob> {
    let width = frame.width;
    let height = frame.height;

    {
        let mut guard = runtime
            .lock()
            .map_err(|_| anyhow!("vision runtime poisoned"))?;

        let mut boxes_flat = Vec::with_capacity(boxes_px.len() * 4);
        for b in boxes_px {
            boxes_flat.extend_from_slice(b);
        }

        let mut label_positions_flat = Vec::with_capacity(label_positions.len() * 2);
        for (x, y) in label_positions {
            label_positions_flat.push(*x);
            label_positions_flat.push(*y);
        }

        let mut offsets = Vec::with_capacity(labels.len());
        let mut lengths = Vec::with_capacity(labels.len());
        let mut chars = Vec::new();
        for text in labels {
            offsets.push(chars.len() as i32);
            let upper = text.to_uppercase();
            chars.extend_from_slice(upper.as_bytes());
            lengths.push(upper.len() as i32);
        }

        let info = format!("FRAME {:06}  FPS {:4.1}", frame_number, fps).to_uppercase();
        let info_width = ((info.chars().count() as i32) * 6).min(width as i32);
        let info_x = (width as i32 - info_width - 4).max(0);
        let info_y = (height as i32 - 12).max(0);

        guard
            .annotate(
                width,
                height,
                &boxes_flat,
                &label_positions_flat,
                &offsets,
                &lengths,
                &chars,
                info.as_bytes(),
                (info_x, info_y),
            )
            .map_err(|err| anyhow!("annotation kernel failed: {err}"))?;
    }

    Ok(GpuEncodeJob {
        runtime: runtime.clone(),
        width,
        height,
        summaries,
        timestamp_ms: frame.timestamp_ms,
        frame_number,
        fps,
        jpeg_quality,
    })
}

fn encode_gpu_frame(job: GpuEncodeJob) -> Result<FramePacket> {
    let GpuEncodeJob {
        runtime,
        width,
        height,
        summaries,
        timestamp_ms,
        frame_number,
        fps,
        jpeg_quality,
    } = job;

    let mut guard = runtime
        .lock()
        .map_err(|_| anyhow!("vision runtime poisoned"))?;
    let quality = jpeg_quality.clamp(1, 100);
    let buffer = guard
        .encode_jpeg(width, height, quality)
        .map_err(|err| anyhow!("nvjpeg encode failed: {err}"))?;

    Ok(FramePacket {
        jpeg: buffer,
        detections: summaries,
        timestamp_ms,
        frame_number,
        fps,
    })
}

fn bgr_to_rgba(input: &[u8]) -> Vec<u8> {
    let pixels = input.len() / 3;
    let mut output = Vec::with_capacity(pixels * 4);
    for chunk in input.chunks_exact(3) {
        output.push(chunk[2]);
        output.push(chunk[1]);
        output.push(chunk[0]);
        output.push(255);
    }
    output
}

fn draw_rectangle(
    image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
    color: Rgba<u8>,
) {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let left = left.clamp(0, width.saturating_sub(1));
    let right = right.clamp(0, width.saturating_sub(1));
    let top = top.clamp(0, height.saturating_sub(1));
    let bottom = bottom.clamp(0, height.saturating_sub(1));

    for x in left..=right {
        if top >= 0 && top < height {
            *image.get_pixel_mut(x as u32, top as u32) = color;
        }
        if bottom >= 0 && bottom < height {
            *image.get_pixel_mut(x as u32, bottom as u32) = color;
        }
    }
    for y in top..=bottom {
        if left >= 0 && left < width {
            *image.get_pixel_mut(left as u32, y as u32) = color;
        }
        if right >= 0 && right < width {
            *image.get_pixel_mut(right as u32, y as u32) = color;
        }
    }
}

fn fill_rect(
    image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
    color: Rgba<u8>,
) {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let left = left.clamp(0, width.saturating_sub(1));
    let right = right.clamp(0, width.saturating_sub(1));
    let top = top.clamp(0, height.saturating_sub(1));
    let bottom = bottom.clamp(0, height.saturating_sub(1));

    for y in top..=bottom {
        for x in left..=right {
            *image.get_pixel_mut(x as u32, y as u32) = color;
        }
    }
}

fn draw_label(
    image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    mut x: i32,
    y: i32,
    text: &str,
    color: Rgba<u8>,
) {
    let height = image.height() as i32;
    let baseline = y;
    for ch in text.chars().flat_map(|c| c.to_uppercase()) {
        if let Some(glyph) = glyph_bits(ch) {
            for (row, pattern) in glyph.iter().enumerate() {
                let py = baseline + row as i32;
                if py < 0 || py >= height {
                    continue;
                }
                for col in 0..5 {
                    if (pattern >> (4 - col)) & 1 == 1 {
                        let px = x + col as i32;
                        if px >= 0 && px < image.width() as i32 {
                            *image.get_pixel_mut(px as u32, py as u32) = color;
                        }
                    }
                }
            }
            x += 6;
        } else {
            x += 6;
        }
    }
}

fn glyph_bits(ch: char) -> Option<[u8; 7]> {
    match ch {
        'A' => Some([
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ]),
        'C' => Some([
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ]),
        'E' => Some([
            0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b11111,
        ]),
        'F' => Some([
            0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b10000,
        ]),
        'M' => Some([
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ]),
        'O' => Some([
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ]),
        'P' => Some([
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ]),
        'R' => Some([
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ]),
        'S' => Some([
            0b01111, 0b10000, 0b01110, 0b00001, 0b00001, 0b10001, 0b01110,
        ]),
        'N' => Some([
            0b10001, 0b11001, 0b10101, 0b10101, 0b10011, 0b10001, 0b10001,
        ]),
        '0' => Some([
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ]),
        '1' => Some([
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ]),
        '2' => Some([
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ]),
        '3' => Some([
            0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
        ]),
        '4' => Some([
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ]),
        '5' => Some([
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ]),
        '6' => Some([
            0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ]),
        '7' => Some([
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ]),
        '8' => Some([
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ]),
        '9' => Some([
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
        ]),
        '%' => Some([
            0b10001, 0b10010, 0b00100, 0b01000, 0b10010, 0b10001, 0b00000,
        ]),
        '.' => Some([0, 0, 0, 0, 0, 0b00110, 0b00110]),
        ' ' => Some([0, 0, 0, 0, 0, 0, 0]),
        _ => None,
    }
}

fn assign_tracks(tracker: &Arc<Mutex<SimpleTracker>>, detections: &mut [DetectionSummary]) {
    if let Ok(mut tracker) = tracker.lock() {
        for det in detections {
            det.track_id = tracker.next_id;
            tracker.next_id += 1;
        }
    }
}

fn load_torch_cuda_runtime(verbose: bool) {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let mut handles = Vec::new();
        for lib in [
            "libtorch_cuda.so",
            "libtorch_cuda_cu.so",
            "libtorch_cuda_cpp.so",
        ] {
            match unsafe { Library::open(Some(lib), RTLD_NOW | RTLD_GLOBAL) } {
                Ok(handle) => {
                    if verbose {
                        info!("Loaded {lib}");
                    }
                    handles.push(handle);
                }
                Err(err) => {
                    if verbose {
                        warn!("Warning: failed to load {lib}: {err}");
                    }
                }
            }
        }
        Box::leak(Box::new(handles));
    });
}
