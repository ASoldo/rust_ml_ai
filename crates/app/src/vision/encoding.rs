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
};

use anyhow::{Result, anyhow};
use crossbeam_channel::Receiver;
use gpu_kernels::VisionRuntime;
use tracing::error;

use crate::vision::{
    data::{DetectionSummary, FRAME_HISTORY_CAPACITY, FrameHistory, FramePacket, SharedFrame},
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
    Cpu(FramePacket),
    Gpu(GpuEncodeJob),
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
    thread::spawn(move || {
        for job in encode_rx {
            if !running.load(Ordering::Relaxed) {
                break;
            }
            let packet_result = match job {
                EncodeJob::Cpu(packet) => Ok(packet),
                EncodeJob::Gpu(task) => encode_gpu_frame(task),
            };

            match packet_result {
                Ok(packet) => {
                    health.beat(HealthComponent::Encoder);
                    if let Ok(mut guard) = history.lock() {
                        guard.push_back(packet.clone());
                        if guard.len() > FRAME_HISTORY_CAPACITY {
                            guard.pop_front();
                        }
                    }
                    if let Ok(mut guard) = shared.lock() {
                        *guard = Some(packet);
                    }
                }
                Err(err) => {
                    error!("Encode stage error: {err}");
                    running.store(false, Ordering::SeqCst);
                    break;
                }
            }
        }
    })
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
