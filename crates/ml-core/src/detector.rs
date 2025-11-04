//! TorchScript object detector wrapper with optional CUDA acceleration.
//!
//! The detector normalises input frames, runs the scripted model, and optionally
//! leverages the CUDA kernels from `gpu-kernels` for preprocessing and NMS.

use std::{
    convert::TryFrom,
    path::Path,
    sync::{Arc, Mutex},
};

use anyhow::{Result, anyhow};
use gpu_kernels::{PreprocessOutput, VisionRuntime};
use tch::{self, Device, Kind, Tensor, no_grad};

/// Single detection returned by the detector.
#[derive(Debug, Clone, Default)]
pub struct Detection {
    /// Bounding box stored as `[x1, y1, x2, y2]` in input-image pixels.
    pub bbox: [f32; 4],
    pub score: f32,
    pub class_id: i64,
}

/// Batched detections for a single frame.
#[derive(Debug, Clone, Default)]
pub struct DetectionBatch {
    pub detections: Vec<Detection>,
}

/// TorchScript-backed detector wrapper.
pub struct Detector {
    module: tch::CModule,
    device: Device,
    input_size: (i64, i64),
    confidence_threshold: f32,
    vision: Option<Arc<Mutex<VisionRuntime>>>,
}

impl Detector {
    /// Load a TorchScript module and prepare it for GPU execution.
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        device: Device,
        input_size: (i64, i64),
    ) -> Result<Self> {
        let mut module = tch::CModule::load_on_device(model_path, device)?;
        module.set_eval();
        let vision = match device {
            Device::Cuda(index) => {
                let device_index = i32::try_from(index)
                    .map_err(|_| anyhow!("CUDA device index {index} is out of range for i32"))?;
                let runtime = VisionRuntime::new(device_index)
                    .map_err(|err| anyhow!("failed to initialise vision runtime: {err}"))?;
                Some(Arc::new(Mutex::new(runtime)))
            }
            _ => None,
        };
        Ok(Self {
            module,
            device,
            input_size,
            confidence_threshold: 0.25,
            vision,
        })
    }

    /// Override the confidence threshold used for filtering detections.
    pub fn with_confidence_threshold(mut self, confidence: f32) -> Self {
        self.confidence_threshold = confidence;
        self
    }

    pub fn input_size(&self) -> (i64, i64) {
        self.input_size
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn uses_gpu_runtime(&self) -> bool {
        self.vision.is_some()
    }

    /// Converts an interleaved BGR frame into a normalised tensor on the chosen device.
    pub fn bgr_to_tensor(&self, bgr: &[u8], width: i32, height: i32) -> Result<Tensor> {
        let expected = (width as usize) * (height as usize) * 3;
        if bgr.len() != expected {
            anyhow::bail!(
                "unexpected frame buffer size: got {} bytes, expected {}",
                bgr.len(),
                expected
            );
        }

        if let (Device::Cuda(_), Some(runtime)) = (&self.device, &self.vision) {
            let (target_w, target_h) = (self.input_size.0 as i32, self.input_size.1 as i32);
            let mut guard = runtime
                .lock()
                .map_err(|_| anyhow::anyhow!("vision runtime poisoned"))?;
            let PreprocessOutput {
                tensor_ptr,
                width: out_w,
                height: out_h,
                ..
            } = guard
                .preprocess_bgr(bgr, width, height, target_w, target_h)
                .map_err(|err| anyhow!("preprocess kernel failed: {err}"))?;
            let size = [1, 3, out_h as i64, out_w as i64];
            let tensor = unsafe {
                Tensor::from_blob(
                    tensor_ptr as *const u8,
                    &size,
                    &[],
                    Kind::Float,
                    self.device,
                )
            };
            Ok(tensor)
        } else {
            let (in_w, in_h) = self.input_size;
            let mut tensor = Tensor::from_slice(bgr)
                .to_kind(Kind::Float)
                .view([height as i64, width as i64, 3])
                .permute([2, 0, 1])
                .unsqueeze(0)
                / 255.0;

            if (width as i64, height as i64) != (in_w, in_h) {
                tensor = tensor.upsample_bilinear2d(&[in_h, in_w], false, None, None);
            }

            Ok(tensor.to_device(self.device))
        }
    }

    /// Executes the TorchScript module and performs basic confidence filtering.
    pub fn infer(&self, input: &Tensor) -> Result<DetectionBatch> {
        let mut batches = self.infer_batch(input)?;
        Ok(batches.pop().unwrap_or_default())
    }

    /// Batched variant of [`infer`]; accepts an input tensor with batch dimension.
    pub fn infer_batch(&self, input: &Tensor) -> Result<Vec<DetectionBatch>> {
        let output = no_grad(|| self.module.forward_ts(&[input]))?;
        let shape = output.size();
        if shape.len() != 3 {
            anyhow::bail!("unexpected detector output shape: {shape:?}");
        }
        let batch = shape[0];
        let channels = shape[1];
        if channels < 5 {
            anyhow::bail!(
                "detector output requires at least 5 channels (x,y,w,h,conf), got {channels}"
            );
        }

        let outputs = output.permute([0, 2, 1]).contiguous();
        let mut results = Vec::with_capacity(batch as usize);

        for idx in 0..batch {
            let preds = outputs.get(idx);
            let detections = match self.device {
                Device::Cuda(_) => {
                    if let Some(rows) = self.process_gpu_single(preds.copy())? {
                        self.rows_to_detections(rows, false)?
                    } else {
                        Vec::new()
                    }
                }
                _ => {
                    let rows: Vec<Vec<f32>> =
                        Vec::<Vec<f32>>::try_from(&preds.to_device(Device::Cpu))?;
                    let mut detections = self.rows_to_detections(rows, true)?;
                    apply_nms(&mut detections, 0.45);
                    detections
                }
            };
            results.push(DetectionBatch { detections });
        }

        Ok(results)
    }

    /// Run GPU-side confidence filtering and NMS for a single batch element.
    fn process_gpu_single(&self, preds: Tensor) -> Result<Option<Vec<Vec<f32>>>> {
        let device = self.device;
        if !matches!(device, Device::Cuda(_)) {
            return Ok(None);
        }
        let scores = preds.select(1, 4);
        let mask = scores.ge(self.confidence_threshold as f64);
        let mut indices = mask.nonzero();
        if indices.numel() == 0 {
            return Ok(Some(Vec::new()));
        }
        indices = indices.squeeze_dim(1);
        let filtered = preds.index_select(0, &indices);
        let scores = filtered.select(1, 4);
        let mut order = scores.argsort(-1, true);
        if order.dim() == 0 {
            order = order.unsqueeze(0);
        }
        let mut ordered = filtered.index_select(0, &order);
        let limit = ordered.size()[0].min(512);
        ordered = ordered.narrow(0, 0, limit);

        if limit == 0 {
            return Ok(Some(Vec::new()));
        }

        let xs = ordered.select(1, 0);
        let ys = ordered.select(1, 1);
        let ws = ordered.select(1, 2);
        let hs = ordered.select(1, 3);
        let half_w = &ws / 2.0;
        let half_h = &hs / 2.0;
        let x1 = &xs - &half_w;
        let y1 = &ys - &half_h;
        let x2 = &xs + &half_w;
        let y2 = &ys + &half_h;
        let boxes = Tensor::stack(&[x1, y1, x2, y2], 1).contiguous();
        let num_boxes = boxes.size()[0] as usize;

        let boxes_ptr = boxes.data_ptr() as u64;
        if let Some(runtime) = &self.vision {
            let mut guard = runtime
                .lock()
                .map_err(|_| anyhow::anyhow!("vision runtime poisoned"))?;
            let keep_flags = guard
                .run_nms(boxes_ptr, num_boxes, 0.45)
                .map_err(|err| anyhow!("nms kernel failed: {err}"))?;
            let kept_indices: Vec<i64> = keep_flags
                .iter()
                .enumerate()
                .filter_map(|(idx, &flag)| (flag != 0).then_some(idx as i64))
                .collect();
            if kept_indices.is_empty() {
                return Ok(Some(Vec::new()));
            }
            let index_tensor = Tensor::from_slice(&kept_indices).to_device(device);
            let selected = ordered.index_select(0, &index_tensor);
            let rows: Vec<Vec<f32>> = Vec::<Vec<f32>>::try_from(&selected.to_device(Device::Cpu))?;
            Ok(Some(rows))
        } else {
            Ok(Some(Vec::new()))
        }
    }

    /// Convert raw detector rows into typed `Detection`s.
    fn rows_to_detections(
        &self,
        rows: Vec<Vec<f32>>,
        apply_threshold: bool,
    ) -> Result<Vec<Detection>> {
        let mut detections = Vec::new();
        for row in rows {
            if row.len() < 5 {
                continue;
            }
            let score = row[4];
            if apply_threshold && score < self.confidence_threshold {
                continue;
            }
            let bbox = xywh_to_corners(row[0], row[1], row[2], row[3], self.input_size);
            let class_id = if row.len() > 5 { row[5] as i64 } else { 0 };
            detections.push(Detection {
                bbox,
                score,
                class_id,
            });
            if detections.len() >= 512 {
                break;
            }
        }
        Ok(detections)
    }

    /// Expose the optional CUDA `VisionRuntime` handle used by annotation code.
    pub fn vision_runtime(&self) -> Option<Arc<Mutex<VisionRuntime>>> {
        self.vision.clone()
    }
}

/// Convert `[x, y, w, h]` detections into `[x1, y1, x2, y2]` corners.
fn xywh_to_corners(x: f32, y: f32, w: f32, h: f32, input_size: (i64, i64)) -> [f32; 4] {
    let (width, height) = (input_size.0 as f32, input_size.1 as f32);
    let half_w = w / 2.0;
    let half_h = h / 2.0;
    let mut x1 = x - half_w;
    let mut y1 = y - half_h;
    let mut x2 = x + half_w;
    let mut y2 = y + half_h;
    x1 = x1.clamp(0.0, width - 1.0);
    y1 = y1.clamp(0.0, height - 1.0);
    x2 = x2.clamp(0.0, width - 1.0);
    y2 = y2.clamp(0.0, height - 1.0);
    [x1, y1, x2, y2]
}

/// Apply greedy non-maximum suppression on CPU.
fn apply_nms(detections: &mut Vec<Detection>, iou_threshold: f32) {
    detections.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut result: Vec<Detection> = Vec::with_capacity(detections.len());

    for det in detections.drain(..) {
        let mut should_keep = true;
        for kept in &result {
            if iou(&det.bbox, &kept.bbox) > iou_threshold {
                should_keep = false;
                break;
            }
        }
        if should_keep {
            result.push(det);
        }
    }

    *detections = result;
}

/// Intersection-over-union helper used by CPU NMS.
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let intersection = inter_w * inter_h;
    if intersection <= 0.0 {
        return 0.0;
    }
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    intersection / (area_a + area_b - intersection + 1e-6)
}
