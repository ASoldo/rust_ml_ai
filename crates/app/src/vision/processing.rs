//! Detector worker threads responsible for batching frames and preparing
//! annotation jobs.
//!
//! Processing workers own the heavy lifting: they convert frames into tensors,
//! run inference, consolidate detections, and delegate to CPU or GPU annotation
//! paths before handing jobs to the encoder.

use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

use anyhow::{Context, Result, bail};
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use gpu_kernels::VisionRuntime;
use ml_core::{
    DetectionBatch,
    detector::Detector,
    tch::{Device, Tensor},
};
use tracing::{debug, error};
use video_ingest::{Frame, FrameFormat};

use crate::vision::{
    annotation::{annotate_frame_cpu, annotate_frame_gpu},
    data::DetectionSummary,
    encoding::EncodeJob,
    watchdog::{HealthComponent, PipelineHealth},
};

/// Unit of work consumed by processing threads.
pub(crate) struct FrameTask {
    pub(crate) frame: Frame,
    pub(crate) frame_number: u64,
    pub(crate) fps: f32,
}

#[derive(Clone)]
/// Initialiser payload shared across processing workers.
pub(crate) struct DetectorInit {
    pub(crate) model_path: PathBuf,
    pub(crate) device: Device,
    pub(crate) input_size: (i64, i64),
}

#[derive(Default)]
/// Minimal tracker assigning monotonic IDs to detections for HUD display.
pub(crate) struct SimpleTracker {
    next_id: i64,
}

/// Spawn a processing worker thread that owns a detector instance.
///
/// Each worker drains the frame queue, batches tensors, and hands completed
/// annotation jobs to the encoder stage.
pub(crate) fn spawn_processing_worker(
    detector_init: DetectorInit,
    tracker: Arc<Mutex<SimpleTracker>>,
    work_rx: Receiver<FrameTask>,
    verbose: bool,
    jpeg_quality: i32,
    init_tx: Sender<std::result::Result<String, String>>,
    encode_tx: Sender<EncodeJob>,
    health: Arc<PipelineHealth>,
    running: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    batch_size: usize,
    worker_index: usize,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let detector = match Detector::new(
            &detector_init.model_path,
            detector_init.device,
            detector_init.input_size,
        ) {
            Ok(det) => match init_tx.send(Ok(format!(
                "worker #{worker_index}: detector loaded on {:?} (vision runtime enabled: {})",
                det.device(),
                det.uses_gpu_runtime()
            ))) {
                Ok(_) => det,
                Err(_) => return,
            },
            Err(err) => {
                let _ = init_tx.send(Err(format!(
                    "worker #{worker_index}: failed to load detector: {err}"
                )));
                return;
            }
        };
        drop(init_tx);

        let vision_runtime = detector.vision_runtime();
        let max_batch = batch_size.max(1);

        'outer: loop {
            if shutdown.load(Ordering::Relaxed) || !running.load(Ordering::Relaxed) {
                break;
            }

            let first_task = match work_rx.recv() {
                Ok(task) => task,
                Err(_) => break,
            };

            let mut batch = Vec::with_capacity(max_batch);
            batch.push(first_task);

            if max_batch > 1 {
                while batch.len() < max_batch {
                    match work_rx.try_recv() {
                        Ok(task) => batch.push(task),
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => break,
                    }
                }
            }

            match process_frame_batch(
                &detector,
                &tracker,
                batch,
                verbose,
                jpeg_quality,
                vision_runtime.clone(),
            ) {
                Ok(jobs) => {
                    for job in jobs {
                        health.beat(HealthComponent::Processor);
                        if encode_tx.send(job).is_err() {
                            error!("Encode channel closed, stopping processing worker");
                            running.store(false, Ordering::SeqCst);
                            break 'outer;
                        }
                    }
                }
                Err(err) => {
                    error!("Frame processing error: {err:?}");
                    running.store(false, Ordering::SeqCst);
                    break;
                }
            }
        }
    })
}

/// Convert a batch of `FrameTask`s into encode jobs by running detector
/// inference, annotation, and routing to CPU/GPU encode paths.
pub(crate) fn process_frame_batch(
    detector: &Detector,
    tracker: &Arc<Mutex<SimpleTracker>>,
    mut tasks: Vec<FrameTask>,
    verbose: bool,
    jpeg_quality: i32,
    vision: Option<Arc<Mutex<VisionRuntime>>>,
) -> Result<Vec<EncodeJob>> {
    if tasks.is_empty() {
        return Ok(Vec::new());
    }

    let mut tensors = Vec::with_capacity(tasks.len());
    for task in tasks.iter() {
        let frame = &task.frame;
        if !matches!(frame.format, FrameFormat::Bgr8) {
            bail!("unsupported frame format");
        }
        let tensor = detector
            .bgr_to_tensor(&frame.data, frame.width, frame.height)
            .with_context(|| "Failed to prepare tensor from frame")?;
        tensors.push(tensor.copy());
    }

    let batched_input = if tensors.len() == 1 {
        tensors.remove(0)
    } else {
        Tensor::cat(&tensors, 0)
    };

    let detection_batches = detector
        .infer_batch(&batched_input)
        .with_context(|| "Detector inference failed")?;

    if detection_batches.len() != tasks.len() {
        bail!(
            "Detector returned {} batch(es) but {} frame(s) were submitted",
            detection_batches.len(),
            tasks.len()
        );
    }

    let mut jobs = Vec::with_capacity(tasks.len());
    for (task, detections) in tasks.drain(..).zip(detection_batches.into_iter()) {
        jobs.push(finalize_frame(
            detector,
            tracker,
            &task,
            detections,
            verbose,
            jpeg_quality,
            vision.clone(),
        )?);
    }

    Ok(jobs)
}

/// Finalise an individual frame by drawing detections and packaging an
/// `EncodeJob`.
fn finalize_frame(
    detector: &Detector,
    tracker: &Arc<Mutex<SimpleTracker>>,
    task: &FrameTask,
    detections: DetectionBatch,
    verbose: bool,
    jpeg_quality: i32,
    vision: Option<Arc<Mutex<VisionRuntime>>>,
) -> Result<EncodeJob> {
    let frame = &task.frame;
    if verbose {
        if detections.detections.is_empty() {
            debug!("frame #{}: no detections", task.frame_number);
        } else {
            debug!(
                "frame #{}: {} detection(s)",
                task.frame_number,
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
            task.frame_number,
            task.fps,
            summaries.clone(),
            &boxes_px,
            &label_positions,
            &labels,
            jpeg_quality,
        )?))
    } else {
        let packet =
            annotate_frame_cpu(frame, task.frame_number, task.fps, summaries, jpeg_quality)?;
        Ok(EncodeJob::Cpu(packet))
    }
}

/// Assign incremental track identifiers to detections for downstream HUD use.
fn assign_tracks(tracker: &Arc<Mutex<SimpleTracker>>, detections: &mut [DetectionSummary]) {
    if let Ok(mut tracker) = tracker.lock() {
        for det in detections {
            det.track_id = tracker.next_id;
            tracker.next_id += 1;
        }
    }
}
