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
    time::Instant,
};

use anyhow::{Context, Result, bail};
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use gpu_kernels::VisionRuntime;
use ml_core::{
    DetectionBatch,
    detector::Detector,
    tch::{Cuda, Device, Tensor},
};
use tracing::{Span, debug, error};
use video_ingest::{Frame, FrameFormat};

use crate::vision::{
    annotation::{annotate_frame_cpu, annotate_frame_gpu},
    data::DetectionSummary,
    encoding::EncodeJob,
    telemetry,
    watchdog::{HealthComponent, PipelineHealth},
};

/// Unit of work consumed by processing threads.
pub(crate) struct FrameTask {
    pub(crate) frame: Frame,
    pub(crate) frame_number: u64,
    pub(crate) fps: f32,
    pub(crate) enqueued_at: Instant,
    pub(crate) span: Span,
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
    telemetry::spawn_thread(format!("vision-processing-{worker_index}"), move || {
        let worker_span = tracing::info_span!(
            "processing.worker",
            worker = worker_index,
            batch = batch_size,
            device = tracing::field::Empty
        );
        let _worker_guard = worker_span.enter();

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

        worker_span.record(
            "device",
            &tracing::field::display(format_args!("{:?}", detector.device())),
        );

        let vision_runtime = detector.vision_runtime();
        let max_batch = batch_size.max(1);

        'outer: loop {
            if shutdown.load(Ordering::Relaxed) || !running.load(Ordering::Relaxed) {
                break;
            }

            let first_task = match work_rx.recv() {
                Ok(task) => {
                    metrics::gauge!("vision_queue_depth", "queue" => "processing")
                        .set(work_rx.len() as f64);
                    task
                }
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
                worker_index,
            ) {
                Ok(jobs) => {
                    for job in jobs {
                        health.beat(HealthComponent::Processor);
                        let frame_id = match &job {
                            EncodeJob::Cpu { packet, .. } => packet.frame_number,
                            EncodeJob::Gpu { job, .. } => job.frame_number,
                        };
                        let send_result = tracing::info_span!(
                            "processing.encode_send",
                            worker = worker_index,
                            frame = frame_id
                        )
                        .in_scope(|| encode_tx.send(job));
                        if send_result.is_err() {
                            error!("Encode channel closed, stopping processing worker");
                            running.store(false, Ordering::SeqCst);
                            break 'outer;
                        }
                        metrics::gauge!("vision_queue_depth", "queue" => "encoding")
                            .set(encode_tx.len() as f64);
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
    .expect("failed to spawn processing worker")
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
    worker_index: usize,
) -> Result<Vec<EncodeJob>> {
    if tasks.is_empty() {
        return Ok(Vec::new());
    }

    let batch_size = tasks.len();
    let batch_span = tracing::info_span!(
        "processing.batch",
        worker = worker_index,
        frames = batch_size
    );
    let _batch_guard = batch_span.enter();

    metrics::gauge!("vision_processing_batch_frames").set(batch_size as f64);

    let stage_start = Instant::now();
    let tensor_start = Instant::now();

    let mut tensors = Vec::with_capacity(batch_size);
    for task in tasks.iter() {
        let frame = &task.frame;
        if !matches!(frame.format, FrameFormat::Bgr8) {
            bail!("unsupported frame format");
        }
        let tensor = tracing::info_span!(
            "processing.preprocess",
            worker = worker_index,
            frame = task.frame_number
        )
        .in_scope(|| {
            detector
                .bgr_to_tensor(&frame.data, frame.width, frame.height)
                .with_context(|| "Failed to prepare tensor from frame")
        })?;
        tensors.push(tensor.copy());
    }

    metrics::histogram!("vision_processing_tensor_seconds")
        .record(tensor_start.elapsed().as_secs_f64());

    let batched_input = if tensors.len() == 1 {
        tensors.remove(0)
    } else {
        Tensor::cat(&tensors, 0)
    };

    let inference_start = Instant::now();
    let detection_batches = tracing::info_span!(
        "processing.infer",
        worker = worker_index,
        frames = batch_size
    )
    .in_scope(|| {
        let outputs = detector
            .infer_batch(&batched_input)
            .with_context(|| "Detector inference failed")?;
        if detector.uses_gpu_runtime() {
            if let Device::Cuda(device_index) = detector.device() {
                let _ = Cuda::synchronize(device_index as i64);
            }
        }
        Ok::<_, anyhow::Error>(outputs)
    })?;
    metrics::histogram!("vision_processing_inference_seconds")
        .record(inference_start.elapsed().as_secs_f64());

    if detection_batches.len() != tasks.len() {
        bail!(
            "Detector returned {} batch(es) but {} frame(s) were submitted",
            detection_batches.len(),
            tasks.len()
        );
    }

    let mut jobs = Vec::with_capacity(tasks.len());
    let annotation_start = Instant::now();
    for (task, detections) in tasks.drain(..).zip(detection_batches.into_iter()) {
        let flow_guard = task.span.enter();
        let frame_span = tracing::info_span!(
            "processing.frame",
            worker = worker_index,
            frame = task.frame_number,
            fps = task.fps
        );
        let _frame_guard = frame_span.enter();

        let queue_delay = task.enqueued_at.elapsed();
        metrics::histogram!("vision_processing_queue_delay_seconds")
            .record(queue_delay.as_secs_f64());

        jobs.push(finalize_frame(
            detector,
            tracker,
            &task,
            detections,
            verbose,
            jpeg_quality,
            vision.clone(),
            task.span.clone(),
        )?);
        drop(flow_guard);
    }

    metrics::histogram!("vision_processing_annotation_seconds")
        .record(annotation_start.elapsed().as_secs_f64());

    metrics::histogram!("vision_stage_latency_seconds", "stage" => "processing")
        .record(stage_start.elapsed().as_secs_f64());

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
    trace_span: Span,
) -> Result<EncodeJob> {
    let frame = &task.frame;
    let path_label = if vision.is_some() { "gpu" } else { "cpu" };
    let finalize_span = tracing::info_span!(
        "processing.finalize",
        frame = task.frame_number,
        path = path_label
    );
    let _span_guard = finalize_span.enter();
    let finalize_start = Instant::now();

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
    {
        let _nms_guard = tracing::info_span!("processing.nms", frame = task.frame_number).entered();
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
    }

    let job = if let Some(runtime) = vision {
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
        let gpu_job = tracing::info_span!(
            "processing.annotate",
            path = "gpu",
            frame = task.frame_number
        )
        .in_scope(|| {
            annotate_frame_gpu(
                &runtime,
                frame,
                task.frame_number,
                task.fps,
                summaries.clone(),
                &boxes_px,
                &label_positions,
                &labels,
                jpeg_quality,
            )
            .map(|job| EncodeJob::Gpu {
                job,
                span: trace_span.clone(),
            })
        })?;
        gpu_job
    } else {
        let cpu_packet = tracing::info_span!(
            "processing.annotate",
            path = "cpu",
            frame = task.frame_number
        )
        .in_scope(|| {
            annotate_frame_cpu(frame, task.frame_number, task.fps, summaries, jpeg_quality).map(
                |packet| EncodeJob::Cpu {
                    packet,
                    span: trace_span.clone(),
                },
            )
        })?;
        cpu_packet
    };

    metrics::histogram!("vision_processing_finalize_seconds", "path" => path_label)
        .record(finalize_start.elapsed().as_secs_f64());

    Ok(job)
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
