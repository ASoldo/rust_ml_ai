//! Pipeline supervisor tying together capture, processing, encoding, and the
//! preview server.
//!
//! The pipeline is responsible for wiring channels between stages, keeping
//! watchdog state in sync, and handling restarts when components stall.

use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc, Mutex, Once,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use crossbeam_channel::{Receiver, TrySendError};
use ml_core::tch::{Cuda, Device};
use tracing::{Span, debug, error, warn};

use crate::vision::{
    SourceKind, VisionConfig,
    data::{FRAME_HISTORY_CAPACITY, FrameHistory, SharedFrame},
    encoding::{EncodeJob, spawn_encode_worker},
    processing::{DetectorInit, FrameTask, SimpleTracker, spawn_processing_worker},
    runtime::load_torch_cuda_runtime,
    server::spawn_preview_server,
    telemetry,
    watchdog::{HealthComponent, PipelineHealth, WatchdogState, spawn_watchdog},
};

/// Run the vision pipeline, automatically restarting on recoverable faults.
pub fn run(config: VisionConfig) -> Result<()> {
    static CTRL_HANDLER: Once = Once::new();

    let shutdown = Arc::new(AtomicBool::new(false));
    let handler_shutdown = shutdown.clone();
    CTRL_HANDLER.call_once(move || {
        if let Err(err) = ctrlc::set_handler({
            let handler_shutdown = handler_shutdown.clone();
            move || {
                handler_shutdown.store(true, Ordering::SeqCst);
            }
        }) {
            warn!("Failed to install Ctrl+C handler: {err}");
        }
    });

    let mut attempt: u32 = 0;
    loop {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }

        match run_pipeline_once(config.clone(), shutdown.clone()) {
            Ok(PipelineOutcome::Graceful) => break,
            Ok(PipelineOutcome::Restart(reason)) => {
                attempt = attempt.saturating_add(1);
                warn!("Pipeline watchdog requested restart (reason: {reason}), attempt #{attempt}");
                thread::sleep(Duration::from_secs(1));
            }
            Err(err) => {
                error!("Vision pipeline error: {err:?}");
                if shutdown.load(Ordering::SeqCst) {
                    break;
                }
                attempt = attempt.saturating_add(1);
                thread::sleep(Duration::from_secs(1));
            }
        }
    }

    Ok(())
}

/// Result of a single pipeline run attempt.
enum PipelineOutcome {
    Graceful,
    Restart(&'static str),
}

/// Execute the pipeline once, returning whether to exit or restart.
fn run_pipeline_once(config: VisionConfig, shutdown: Arc<AtomicBool>) -> Result<PipelineOutcome> {
    if shutdown.load(Ordering::SeqCst) {
        return Ok(PipelineOutcome::Graceful);
    }

    let _telemetry_guard = telemetry::enter_runtime(&config.telemetry);
    let _ = telemetry::init_metrics_recorder();
    let pipeline_span = tracing::info_span!(
        "vision.pipeline",
        source = %config.camera_uri,
        width = config.width,
        height = config.height,
        detector_width = config.detector_width,
        detector_height = config.detector_height,
        workers = config.processor_workers,
        batch_size = config.batch_size,
        use_cpu = config.use_cpu,
        nvdec = config.use_nvdec,
        gpu = tracing::field::Empty,
        codec = "mjpeg"
    );
    let _pipeline_span_guard = pipeline_span.enter();

    if !config.use_cpu {
        load_torch_cuda_runtime(config.verbose);
    }

    let device = if config.use_cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available()
    };
    pipeline_span.record(
        "gpu",
        &tracing::field::display(format_args!("{:?}", device)),
    );

    let cuda_available = Cuda::is_available();
    let cuda_devices = Cuda::device_count();
    debug!(
        "CUDA available: {} (devices: {})",
        cuda_available, cuda_devices
    );

    debug!(
        "Capture source: {} ({:?})",
        config.camera_uri, config.source_kind
    );
    debug!(
        "Detector input size: {}x{}",
        config.detector_width, config.detector_height
    );
    if config.detector_width != config.width || config.detector_height != config.height {
        debug!(
            "Resizing frames for detector (capture {}x{} → detector {}x{})",
            config.width, config.height, config.detector_width, config.detector_height
        );
    }
    debug!(
        "Processing workers: {} (batch size: {})",
        config.processor_workers, config.batch_size
    );

    let receiver = match config.source_kind {
        SourceKind::Rtsp => {
            match video_ingest::spawn_rtsp_reader(
                &config.camera_uri,
                (config.width, config.height),
                config.use_nvdec,
            ) {
                Ok(rx) => rx,
                Err(err) if config.use_nvdec => {
                    warn!(
                        "RTSP NVDEC setup failed ({}); falling back to software decode",
                        err
                    );
                    video_ingest::spawn_rtsp_reader(
                        &config.camera_uri,
                        (config.width, config.height),
                        false,
                    )
                    .with_context(|| {
                        "Failed to start RTSP capture (software fallback)".to_string()
                    })?
                }
                Err(err) => {
                    return Err(err).with_context(|| "Failed to start RTSP capture".to_string());
                }
            }
        }
        SourceKind::Udp => {
            match video_ingest::spawn_udp_reader(
                &config.camera_uri,
                (config.width, config.height),
                config.use_nvdec,
            ) {
                Ok(rx) => rx,
                Err(err) if config.use_nvdec => {
                    warn!(
                        "UDP NVDEC setup failed ({}); falling back to software decode",
                        err
                    );
                    video_ingest::spawn_udp_reader(
                        &config.camera_uri,
                        (config.width, config.height),
                        false,
                    )
                    .with_context(|| {
                        "Failed to start UDP capture (software fallback)".to_string()
                    })?
                }
                Err(err) => {
                    return Err(err).with_context(|| "Failed to start UDP capture".to_string());
                }
            }
        }
        SourceKind::Device => {
            if config.use_nvdec {
                match video_ingest::spawn_nvdec_h264_reader(
                    &config.camera_uri,
                    (config.width, config.height),
                ) {
                    Ok(rx) => rx,
                    Err(err) => {
                        warn!(
                            "NVDEC capture failed ({}); falling back to V4L software capture",
                            err
                        );
                        video_ingest::spawn_camera_reader(
                            &config.camera_uri,
                            (config.width, config.height),
                        )
                        .with_context(|| "Failed to start capture".to_string())?
                    }
                }
            } else {
                video_ingest::spawn_camera_reader(&config.camera_uri, (config.width, config.height))
                    .with_context(|| "Failed to start capture".to_string())?
            }
        }
    };

    let shared: SharedFrame = Arc::new(Mutex::new(None));
    let history: FrameHistory =
        Arc::new(Mutex::new(VecDeque::with_capacity(FRAME_HISTORY_CAPACITY)));
    let tracker = Arc::new(Mutex::new(SimpleTracker::default()));
    let processing_queue = std::cmp::max(
        3,
        config
            .processor_workers
            .saturating_mul(config.batch_size)
            .saturating_mul(2),
    );
    let encoding_queue = std::cmp::max(3, config.processor_workers.saturating_mul(2));
    let (work_tx, work_rx) = crossbeam_channel::bounded::<FrameTask>(processing_queue);
    let (encode_tx, encode_rx) = crossbeam_channel::bounded::<EncodeJob>(encoding_queue);
    let (frame_done_tx, frame_done_rx) = crossbeam_channel::unbounded::<u64>();
    let mut inflight_spans = HashMap::new();

    let detector_init = DetectorInit {
        model_path: config.model_path.clone(),
        device,
        input_size: (config.detector_width as i64, config.detector_height as i64),
    };

    let (init_tx, init_rx) =
        crossbeam_channel::bounded::<std::result::Result<String, String>>(config.processor_workers);

    let health = Arc::new(PipelineHealth::new());
    let pipeline_running = Arc::new(AtomicBool::new(true));
    let watchdog_state = Arc::new(WatchdogState::new());

    let watchdog_handle = spawn_watchdog(
        health.clone(),
        pipeline_running.clone(),
        shutdown.clone(),
        watchdog_state.clone(),
    );

    let encode_handle = spawn_encode_worker(
        shared.clone(),
        history.clone(),
        encode_rx,
        frame_done_tx,
        health.clone(),
        pipeline_running.clone(),
    );
    let mut processing_handles = Vec::with_capacity(config.processor_workers);
    for worker_index in 0..config.processor_workers {
        let worker_rx = work_rx.clone();
        let worker_init_tx = init_tx.clone();
        let worker_encode_tx = encode_tx.clone();
        let handle = spawn_processing_worker(
            detector_init.clone(),
            tracker.clone(),
            worker_rx,
            config.verbose,
            config.jpeg_quality,
            worker_init_tx,
            worker_encode_tx,
            health.clone(),
            pipeline_running.clone(),
            shutdown.clone(),
            config.batch_size,
            worker_index,
        );
        processing_handles.push(handle);
    }
    drop(init_tx);
    drop(work_rx);

    let mut init_messages = Vec::new();
    for _ in 0..config.processor_workers {
        match init_rx.recv() {
            Ok(Ok(message)) => init_messages.push(message),
            Ok(Err(err)) => {
                pipeline_running.store(false, Ordering::SeqCst);
                drop(work_tx);
                drop(encode_tx);
                for handle in processing_handles.drain(..) {
                    let _ = handle.join();
                }
                let _ = encode_handle.join();
                let _ = watchdog_handle.join();
                bail!(err);
            }
            Err(err) => {
                pipeline_running.store(false, Ordering::SeqCst);
                drop(work_tx);
                drop(encode_tx);
                for handle in processing_handles.drain(..) {
                    let _ = handle.join();
                }
                let _ = encode_handle.join();
                let _ = watchdog_handle.join();
                bail!("Processing thread failed to initialise detector: {err}");
            }
        }
    }
    for message in init_messages {
        debug!("{message}");
        println!("{message}");
    }

    let preview_server = spawn_preview_server(shared.clone(), history.clone())
        .context("Failed to start preview server")?;

    debug!("HTTP preview available at http://127.0.0.1:8080/frame.jpg and /stream.mjpg");
    println!("HTTP preview available at http://127.0.0.1:8080/frame.jpg and /stream.mjpg");
    if config.verbose {
        debug!("Running vision pipeline — press Ctrl+C to stop");
        println!("Running vision pipeline — press Ctrl+C to stop");
    }

    let mut frame_number: u64 = 0;
    let mut smoothed_fps: f32 = 0.0;
    let mut last_instant = Instant::now();
    let mut dropped_frames: u64 = 0;
    let mut restart_reason: Option<&'static str> = None;

    while pipeline_running.load(Ordering::Relaxed) {
        drain_completed_frame_spans(&frame_done_rx, &mut inflight_spans);
        if shutdown.load(Ordering::Relaxed) {
            pipeline_running.store(false, Ordering::SeqCst);
            break;
        }

        let frame_result = tracing::info_span!("capture.recv").in_scope(|| receiver.recv());
        match frame_result {
            Ok(frame) => match frame {
                Ok(frame) => {
                    let next_frame_number = frame_number.wrapping_add(1);
                    let frame_span = tracing::info_span!(
                        "frame",
                        frame = next_frame_number,
                        width = frame.width,
                        height = frame.height,
                        timestamp = frame.timestamp_ms
                    );
                    let task_span = frame_span.clone();
                    let _frame_guard = frame_span.enter();
                    health.beat(HealthComponent::Capture);
                    frame_number = next_frame_number;

                    let capture_stage_start = Instant::now();

                    let now = Instant::now();
                    let elapsed = now.duration_since(last_instant).as_secs_f32();
                    last_instant = now;
                    if elapsed > 0.0 {
                        let instant = 1.0 / elapsed;
                        smoothed_fps = if smoothed_fps == 0.0 {
                            instant
                        } else {
                            0.9 * smoothed_fps + 0.1 * instant
                        };
                        metrics::histogram!("vision_capture_frame_interval_seconds")
                            .record(elapsed as f64);
                    }
                    metrics::gauge!("vision_pipeline_fps").set(smoothed_fps as f64);

                    let capture_span = tracing::info_span!(
                        "capture.frame",
                        frame = frame_number,
                        fps = smoothed_fps,
                        timestamp = frame.timestamp_ms
                    );
                    let _capture_guard = capture_span.enter();

                    if frame_number % 30 == 0 {
                        debug!(
                            "Capture heartbeat: frame #{}, {:.1} fps, ts={}",
                            frame_number, smoothed_fps, frame.timestamp_ms
                        );
                    }

                    let task = FrameTask {
                        frame,
                        frame_number,
                        fps: smoothed_fps,
                        enqueued_at: Instant::now(),
                        span: task_span,
                    };
                    let _dispatch_guard = tracing::info_span!(
                        "capture.dispatch",
                        frame = frame_number,
                        queue_depth = work_tx.len()
                    )
                    .entered();

                    match work_tx.try_send(task) {
                        Ok(()) => {
                            inflight_spans.insert(frame_number, frame_span.clone());
                            metrics::gauge!("vision_queue_depth", "queue" => "processing")
                                .set(work_tx.len() as f64);
                        }
                        Err(TrySendError::Full(_)) => {
                            dropped_frames = dropped_frames.wrapping_add(1);
                            metrics::counter!("vision_capture_dropped_frames_total").increment(1);
                            metrics::gauge!("vision_queue_depth", "queue" => "processing")
                                .set(work_tx.len() as f64);
                            tracing::info_span!(
                                "frame.drop",
                                frame = frame_number,
                                queue_depth = work_tx.len(),
                                dropped_total = dropped_frames
                            )
                            .in_scope(|| {});
                            if config.verbose {
                                warn!(
                                    "Dropping frame #{frame_number} (processing backlog, dropped total: {})",
                                    dropped_frames
                                );
                            }
                        }
                        Err(TrySendError::Disconnected(_)) => {
                            error!("Processing thread terminated unexpectedly");
                            restart_reason = Some("processing channel disconnected");
                            metrics::histogram!("vision_stage_latency_seconds", "stage" => "capture")
                                .record(capture_stage_start.elapsed().as_secs_f64());
                            pipeline_running.store(false, Ordering::SeqCst);
                            break;
                        }
                    }

                    metrics::histogram!("vision_stage_latency_seconds", "stage" => "capture")
                        .record(capture_stage_start.elapsed().as_secs_f64());
                }
                Err(err) => {
                    error!("Capture error: {err}");
                    restart_reason = Some("capture error");
                    pipeline_running.store(false, Ordering::SeqCst);
                    break;
                }
            },
            Err(err) => {
                error!("Frame channel closed: {err}");
                restart_reason = Some("capture channel closed");
                pipeline_running.store(false, Ordering::SeqCst);
                break;
            }
        }
    }

    drain_completed_frame_spans(&frame_done_rx, &mut inflight_spans);
    inflight_spans.clear();

    debug!("Stopping vision pipeline");
    println!("Stopping vision pipeline");

    pipeline_running.store(false, Ordering::SeqCst);
    drop(work_tx);
    for handle in processing_handles {
        let _ = handle.join();
    }
    drop(encode_tx);
    let _ = encode_handle.join();
    let _ = watchdog_handle.join();
    preview_server.stop();

    if watchdog_state.is_triggered() {
        let reason = watchdog_state
            .reason()
            .map(|component| component.label())
            .unwrap_or("watchdog");
        return Ok(PipelineOutcome::Restart(reason));
    }

    if let Some(reason) = restart_reason {
        return Ok(PipelineOutcome::Restart(reason));
    }

    if shutdown.load(Ordering::SeqCst) {
        return Ok(PipelineOutcome::Graceful);
    }

    Ok(PipelineOutcome::Graceful)
}

fn drain_completed_frame_spans(rx: &Receiver<u64>, inflight: &mut HashMap<u64, Span>) {
    while let Ok(frame_number) = rx.try_recv() {
        if inflight.remove(&frame_number).is_none() {
            tracing::debug!(
                frame = frame_number,
                "received frame completion without in-flight span"
            );
        }
    }
}
