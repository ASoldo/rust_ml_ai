use gpu_kernels::add_vectors;
use ml_core::sample_inputs;

#[cfg(feature = "with-tch")]
use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    sync::{Arc, Mutex},
    time::Duration,
};

#[cfg(feature = "with-tch")]
use ml_core::{
    DetectionBatch, TrainingConfig, detector::Detector, predict_image_file, tch::Device,
    train_mnist,
};

#[cfg(feature = "with-tch")]
use video_ingest::{self, Frame};

#[cfg(feature = "with-tch")]
use actix_web::{App, HttpResponse, HttpServer, web};

#[cfg(feature = "with-tch")]
use actix_web::web::Bytes;

#[cfg(feature = "with-tch")]
use anyhow::{Result as AnyResult, anyhow};

#[cfg(feature = "with-tch")]
use async_stream::stream;

#[cfg(feature = "with-tch")]
use ctrlc;

#[cfg(feature = "with-tch")]
use image::{DynamicImage, ImageBuffer, Rgba, codecs::jpeg::JpegEncoder};

const ELEMENT_COUNT: usize = 16;

#[cfg(feature = "with-tch")]
type SharedFrame = Arc<Mutex<Option<FramePacket>>>;

#[cfg(feature = "with-tch")]
struct ServerState {
    latest: SharedFrame,
}

#[cfg(feature = "with-tch")]
#[cfg(feature = "with-tch")]
#[derive(Clone)]
struct FramePacket {
    jpeg: Vec<u8>,
    detections: Vec<DetectionSummary>,
    timestamp_ms: i64,
}

#[cfg(feature = "with-tch")]
#[derive(Clone, serde::Serialize)]
struct DetectionSummary {
    class: String,
    score: f32,
    bbox: [f32; 4],
    track_id: i64,
}

#[cfg(feature = "with-tch")]
#[derive(Default)]
struct SimpleTracker {
    next_id: i64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if handle_digits_commands(&args) {
        return;
    }

    run_gpu_add_demo();
}

fn run_gpu_add_demo() {
    let (a, b) = sample_inputs(ELEMENT_COUNT);

    println!("Input A: {:?}", a);
    println!("Input B: {:?}", b);

    match add_vectors(&a, &b) {
        Ok(sum) => println!("Sum on GPU: {:?}", sum),
        Err(err) => {
            eprintln!("Failed to launch GPU kernel: {err}");
            eprintln!(
                "Hint: ensure an NVIDIA driver is installed and accessible (try `nvidia-smi`)."
            );
        }
    }
}

#[cfg(feature = "with-tch")]
fn handle_digits_commands(args: &[String]) -> bool {
    match args.get(1).map(|s| s.as_str()) {
        Some("mnist-train") => {
            run_mnist_training(args);
            true
        }
        Some("mnist-predict") => {
            run_mnist_prediction(args);
            true
        }
        Some("vision-demo") => {
            run_vision_demo(args);
            true
        }
        Some("mnist-help") => {
            print_mnist_help();
            true
        }
        _ => false,
    }
}

#[cfg(not(feature = "with-tch"))]
fn handle_digits_commands(_args: &[String]) -> bool {
    false
}

#[cfg(feature = "with-tch")]
fn run_mnist_training(args: &[String]) {
    if args.len() < 4 {
        eprintln!(
            "Usage: cargo run -p cuda-app --features with-tch -- mnist-train <data-dir> <model-out> [epochs] [batch-size] [learning-rate] [--cpu]"
        );
        return;
    }

    let data_dir = PathBuf::from(&args[2]);
    let model_out = PathBuf::from(&args[3]);
    let epochs = args.get(4).and_then(|s| s.parse::<i64>().ok()).unwrap_or(5);
    let batch_size = args
        .get(5)
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(128);
    let learning_rate = args
        .get(6)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1e-3);
    let use_cpu = args.iter().any(|arg| arg == "--cpu");

    let mut config = TrainingConfig::new(&data_dir, &model_out);
    config.epochs = epochs;
    config.batch_size = batch_size;
    config.learning_rate = learning_rate;
    if use_cpu {
        config.device = Device::Cpu;
    }

    match train_mnist(&config) {
        Ok(report) => {
            println!(
                "Training finished — epochs: {}, final loss: {:.4}, test accuracy: {:.2}%",
                report.epochs,
                report.final_loss,
                report.test_accuracy * 100.0
            );
        }
        Err(err) => {
            eprintln!("Failed to train MNIST classifier: {err}");
            eprintln!(
                "Hint: download the MNIST dataset (t10k/train .idx files) into the provided data directory."
            );
        }
    }
}

#[cfg(feature = "with-tch")]
fn run_mnist_prediction(args: &[String]) {
    if args.len() < 4 {
        eprintln!(
            "Usage: cargo run -p cuda-app --features with-tch -- mnist-predict <model-path> <image-path> [--cpu]"
        );
        return;
    }

    let model_path = PathBuf::from(&args[2]);
    let image_path = PathBuf::from(&args[3]);
    let use_cpu = args.iter().any(|arg| arg == "--cpu");
    let device = if use_cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available()
    };

    match predict_image_file(&model_path, &image_path, Some(device)) {
        Ok(prediction) => {
            println!("Predicted digit: {}", prediction.digit);
            println!("Class probabilities:");
            for (digit, prob) in prediction.probabilities.iter().enumerate() {
                println!("  {digit}: {prob:.3}");
            }
        }
        Err(err) => {
            eprintln!("Failed to run prediction: {err}");
            eprintln!(
                "Hint: make sure the model path points to a `.ot` file produced by mnist-train \n        and that the image is a 28x28 grayscale PNG or JPEG."
            );
        }
    }
}

#[cfg(feature = "with-tch")]
fn print_mnist_help() {
    println!("MNIST helper commands:");
    println!(
        "  mnist-train <data-dir> <model-out> [epochs] [batch-size] [learning-rate] [--cpu]\n      Train the digit classifier using files in <data-dir> and save weights to <model-out>."
    );
    println!(
        "  mnist-predict <model-path> <image-path> [--cpu]\n      Load a trained model and classify a 28x28 grayscale image."
    );
    println!(
        "  vision-demo <camera-uri> <model-path> <width> <height> [--cpu]\n      Stream frames from the camera and run the TorchScript detector (stubbed)."
    );
    println!("  mnist-help\n      Show this message.");
}

#[cfg(feature = "with-tch")]
fn run_vision_demo(args: &[String]) {
    if args.len() < 6 {
        eprintln!(
            "Usage: cargo run -p cuda-app --features with-tch -- vision-demo <camera-uri> <model-path> <width> <height> [--cpu]"
        );
        return;
    }

    let camera_uri = args[2].clone();
    let model_path = PathBuf::from(&args[3]);
    let width = match args[4].parse::<i32>() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("width must be an integer");
            return;
        }
    };
    let height = match args[5].parse::<i32>() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("height must be an integer");
            return;
        }
    };
    let use_cpu = args.iter().any(|arg| arg == "--cpu");
    let device = if use_cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available()
    };

    let detector = match Detector::new(&model_path, device, (width as i64, height as i64)) {
        Ok(det) => det,
        Err(err) => {
            eprintln!("Failed to load detector: {err}");
            return;
        }
    };

    let receiver = match video_ingest::spawn_camera_reader(&camera_uri, (width, height)) {
        Ok(rx) => rx,
        Err(err) => {
            eprintln!("Failed to start capture: {err}");
            return;
        }
    };

    let shared = Arc::new(Mutex::new(None));
    let tracker = Arc::new(Mutex::new(SimpleTracker::default()));
    if let Err(err) = spawn_preview_server(shared.clone()) {
        eprintln!("Failed to start preview server: {err}");
    } else {
        println!("HTTP preview available at http://127.0.0.1:8080/frame.jpg and /stream.mjpg");
    }

    let running = Arc::new(AtomicBool::new(true));
    {
        let running = running.clone();
        if let Err(err) = ctrlc::set_handler(move || {
            running.store(false, Ordering::SeqCst);
        }) {
            eprintln!("Failed to install Ctrl+C handler: {err}");
        }
    }

    println!("Running vision demo — press Ctrl+C to stop");
    let mut frame_index: usize = 0;
    while running.load(Ordering::Relaxed) {
        match receiver.recv() {
            Ok(frame) => match frame {
                Ok(frame) => {
                    match process_frame(frame_index, &detector, &frame, &tracker) {
                        Ok(packet) => {
                            if let Ok(mut guard) = shared.lock() {
                                *guard = Some(packet);
                            }
                        }
                        Err(err) => eprintln!("Frame processing error: {err}"),
                    }
                    frame_index = frame_index.wrapping_add(1);
                }
                Err(err) => {
                    eprintln!("Capture error: {err}");
                    break;
                }
            },
            Err(err) => {
                eprintln!("Frame channel closed: {err}");
                break;
            }
        }
        if !running.load(Ordering::Relaxed) {
            break;
        }
    }

    println!("Stopping vision demo");
}

#[cfg(feature = "with-tch")]
fn process_frame(
    index: usize,
    detector: &Detector,
    frame: &Frame,
    tracker: &Arc<Mutex<SimpleTracker>>,
) -> AnyResult<FramePacket> {
    let tensor = detector.rgba_to_tensor(&frame.data, frame.width, frame.height)?;
    let detections = detector.infer(&tensor)?;
    if detections.detections.is_empty() {
        println!("frame #{index}: no detections");
    } else {
        println!(
            "frame #{index}: {} detection(s)",
            detections.detections.len()
        );
        for (idx, det) in detections.detections.iter().enumerate() {
            println!(
                "  #{idx}: class={} conf={:.3} bbox={:?}",
                det.class_id, det.score, det.bbox
            );
        }
    }
    annotate_frame(frame, &detections, tracker)
}

#[cfg(feature = "with-tch")]
fn spawn_preview_server(shared: SharedFrame) -> std::io::Result<()> {
    let server_shared = shared.clone();
    std::thread::spawn(move || {
        if let Err(err) = actix_web::rt::System::new().block_on(async move {
            HttpServer::new(move || {
                App::new()
                    .app_data(web::Data::new(ServerState {
                        latest: server_shared.clone(),
                    }))
                    .route("/", web::get().to(index_route))
                    .route("/frame.jpg", web::get().to(frame_handler))
                    .route("/stream.mjpg", web::get().to(stream_handler))
                    .route("/detections", web::get().to(detections_handler))
            })
            .bind(("0.0.0.0", 8080))?
            .run()
            .await
        }) {
            eprintln!("HTTP server error: {err}");
        }
    });
    Ok(())
}

#[cfg(feature = "with-tch")]
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

#[cfg(feature = "with-tch")]
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
        .append_header(("Cache-Control", "no-cache"))
        .append_header(("Content-Type", "multipart/x-mixed-replace; boundary=frame"))
        .streaming(stream)
}

#[cfg(feature = "with-tch")]
async fn index_route() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(
            "<html><body><h3>Vision Preview</h3><img src=\"/stream.mjpg\" width=\"640\" /></body></html>",
        )
}

#[cfg(feature = "with-tch")]
async fn detections_handler(state: web::Data<ServerState>) -> HttpResponse {
    let guard = match state.latest.lock() {
        Ok(guard) => guard,
        Err(err) => return HttpResponse::InternalServerError().body(err.to_string()),
    };
    if let Some(ref packet) = *guard {
        #[derive(serde::Serialize)]
        struct ResponsePayload<'a> {
            timestamp_ms: i64,
            detections: &'a [DetectionSummary],
        }

        HttpResponse::Ok().json(ResponsePayload {
            timestamp_ms: packet.timestamp_ms,
            detections: &packet.detections,
        })
    } else {
        HttpResponse::NoContent().finish()
    }
}

#[cfg(feature = "with-tch")]
fn annotate_frame(
    frame: &Frame,
    detections: &DetectionBatch,
    tracker: &Arc<Mutex<SimpleTracker>>,
) -> AnyResult<FramePacket> {
    let width = frame.width as u32;
    let height = frame.height as u32;
    let mut image = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, frame.data.clone())
        .ok_or_else(|| anyhow!("failed to convert frame into image buffer"))?;

    let mut summaries = Vec::with_capacity(detections.detections.len());

    for det in &detections.detections {
        let left = det.bbox[0].clamp(0.0, (width - 1) as f32);
        let top = det.bbox[1].clamp(0.0, (height - 1) as f32);
        let right = det.bbox[2].clamp(0.0, (width - 1) as f32);
        let bottom = det.bbox[3].clamp(0.0, (height - 1) as f32);
        draw_rectangle(
            &mut image,
            left.round() as i32,
            top.round() as i32,
            right.round() as i32,
            bottom.round() as i32,
            Rgba([0, 255, 0, 255]),
        );

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

        let label_x = left.round() as i32;
        let label_y = (top.round() as i32 - 12).max(0);
        let text_width = label.chars().count() as i32 * 6;
        fill_rect(
            &mut image,
            label_x,
            label_y,
            label_x + text_width,
            label_y + 8,
            Rgba([0, 0, 0, 180]),
        );
        draw_label(&mut image, label_x, label_y, label, Rgba([0, 255, 0, 255]));
    }

    let rgb = DynamicImage::ImageRgba8(image).to_rgb8();
    let mut buffer = Vec::new();
    JpegEncoder::new_with_quality(&mut buffer, 85).encode_image(&rgb)?;
    assign_tracks(tracker, &mut summaries);

    Ok(FramePacket {
        jpeg: buffer,
        detections: summaries,
        timestamp_ms: frame.timestamp_ms,
    })
}

#[cfg(feature = "with-tch")]
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

#[cfg(feature = "with-tch")]
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

#[cfg(feature = "with-tch")]
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

#[cfg(feature = "with-tch")]
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
        ' ' => Some([0, 0, 0, 0, 0, 0, 0]),
        _ => None,
    }
}

#[cfg(feature = "with-tch")]
fn assign_tracks(tracker: &Arc<Mutex<SimpleTracker>>, detections: &mut [DetectionSummary]) {
    if let Ok(mut tracker) = tracker.lock() {
        for det in detections {
            det.track_id = tracker.next_id;
            tracker.next_id += 1;
        }
    }
}
