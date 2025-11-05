//! Encoder stage handling both CPU and GPU JPEG generation.
//!
//! The encoding stage consumes `EncodeJob`s produced by processing workers. CPU
//! jobs already include the fully annotated frame whereas GPU jobs require
//! invoking NVJPEG via the shared `VisionRuntime`.

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Instant,
};

use anyhow::{Result, anyhow};
use crossbeam_channel::Receiver;
use gpu_kernels::VisionRuntime;
use tracing::{Span, error};

use crate::vision::{
    data::{DetectionSummary, FRAME_HISTORY_CAPACITY, FrameHistory, FramePacket, SharedFrame},
    telemetry,
    watchdog::{HealthComponent, PipelineHealth},
};

/// GPU encoding request emitted by the processing workers.
pub(crate) struct GpuEncodeJob {
    pub(crate) runtime: Arc<Mutex<VisionRuntime>>,
    pub(crate) width: i32,
    pub(crate) height: i32,
    pub(crate) summaries: Vec<DetectionSummary>,
    pub(crate) timestamp_ms: i64,
    pub(crate) frame_number: u64,
    pub(crate) fps: f32,
    pub(crate) jpeg_quality: i32,
}

/// Payload sent to the encoder thread. CPU jobs contain a ready-to-serve
/// `FramePacket` while GPU jobs carry the metadata required to finalise the
/// annotated frame via NVJPEG.
pub(crate) enum EncodeJob {
    Cpu { packet: FramePacket, span: Span },
    Gpu { job: GpuEncodeJob, span: Span },
}

/// Spawn the dedicated encoder thread.
///
/// The worker listens for `EncodeJob`s, updates health metrics, and keeps the
/// shared/latest frame buffers in sync for the HTTP preview server.
pub(crate) fn spawn_encode_worker(
    shared: SharedFrame,
    history: FrameHistory,
    encode_rx: Receiver<EncodeJob>,
    health: Arc<PipelineHealth>,
    running: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    telemetry::spawn_thread("vision-encoding", move || {
        let worker_span =
            tracing::info_span!("encoding.worker", codec = "nvjpeg", queue = "encoding");
        let _worker_guard = worker_span.enter();
        let depth_probe = encode_rx.clone();
        for job in encode_rx {
            if !running.load(Ordering::Relaxed) {
                break;
            }
            metrics::gauge!("vision_queue_depth", "queue" => "encoding")
                .set(depth_probe.len() as f64);

            let encode_start = Instant::now();
            let path_label: &'static str;
            let packet_result = match job {
                EncodeJob::Cpu { packet, span } => {
                    path_label = "cpu";
                    Ok((packet, span))
                }
                EncodeJob::Gpu { job, span } => {
                    path_label = "gpu";
                    span.in_scope(|| encode_gpu_frame(job))
                        .map(|packet| (packet, span))
                }
            };

            match packet_result {
                Ok((packet, span)) => {
                    let _frame_guard = span.enter();
                    let job_span = tracing::info_span!(
                        "encoding.job",
                        path = path_label,
                        frame = packet.frame_number,
                        fps = packet.fps
                    );
                    let _job_guard = job_span.enter();

                    health.beat(HealthComponent::Encoder);
                    tracing::info_span!("encoding.sink", stage = "history").in_scope(|| {
                        if let Ok(mut guard) = history.lock() {
                            guard.push_back(packet.clone());
                            if guard.len() > FRAME_HISTORY_CAPACITY {
                                guard.pop_front();
                            }
                        }
                    });
                    tracing::info_span!("encoding.sink", stage = "latest").in_scope(|| {
                        if let Ok(mut guard) = shared.lock() {
                            *guard = Some(packet);
                        }
                    });

                    let elapsed = encode_start.elapsed().as_secs_f64();
                    metrics::histogram!("vision_stage_latency_seconds", "stage" => "encoding")
                        .record(elapsed);
                    metrics::histogram!("vision_encoding_seconds", "path" => path_label)
                        .record(elapsed);
                    if path_label == "gpu" {
                        metrics::histogram!("vision_gpu_encode_seconds").record(elapsed);
                    }
                }
                Err(err) => {
                    error!("Encode stage error: {err}");
                    metrics::counter!("vision_encoding_errors_total", "path" => path_label)
                        .increment(1);
                    let elapsed = encode_start.elapsed().as_secs_f64();
                    metrics::histogram!("vision_stage_latency_seconds", "stage" => "encoding")
                        .record(elapsed);
                    metrics::histogram!("vision_encoding_seconds", "path" => path_label)
                        .record(elapsed);
                    running.store(false, Ordering::SeqCst);
                    break;
                }
            }
        }
    })
    .expect("failed to spawn encoding worker")
}

/// Encode a GPU-annotated frame into a `FramePacket`.
pub(crate) fn encode_gpu_frame(job: GpuEncodeJob) -> Result<FramePacket> {
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

    let nvjpeg_span = tracing::info_span!(
        "encoding.nvjpeg",
        frame = frame_number,
        width = width,
        height = height
    );
    let _nvjpeg_guard = nvjpeg_span.enter();
    let encode_start = Instant::now();

    let mut guard = runtime
        .lock()
        .map_err(|_| anyhow!("vision runtime poisoned"))?;
    let quality = jpeg_quality.clamp(1, 100);
    let buffer = guard
        .encode_jpeg(width, height, quality)
        .map_err(|err| anyhow!("nvjpeg encode failed: {err}"))?;

    metrics::histogram!("vision_gpu_nvjpeg_seconds").record(encode_start.elapsed().as_secs_f64());

    Ok(FramePacket {
        jpeg: buffer,
        detections: summaries,
        timestamp_ms,
        frame_number,
        fps,
    })
}
