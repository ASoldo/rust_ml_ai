use std::{convert::TryFrom, path::Path};

use anyhow::Result;
use tch::{self, Device, Kind, Tensor};

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
}

impl Detector {
    /// Load a TorchScript module and prepare it for GPU execution.
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        device: Device,
        input_size: (i64, i64),
    ) -> Result<Self> {
        let module = tch::CModule::load_on_device(model_path, device)?;
        Ok(Self {
            module,
            device,
            input_size,
            confidence_threshold: 0.25,
        })
    }

    /// Override the confidence threshold used for filtering detections.
    pub fn with_confidence_threshold(mut self, confidence: f32) -> Self {
        self.confidence_threshold = confidence;
        self
    }

    /// Converts an RGBA frame (height, width) into a normalized tensor ready for inference.
    pub fn rgba_to_tensor(&self, rgba: &[u8], width: i32, height: i32) -> Result<Tensor> {
        let expected = (width as usize) * (height as usize) * 4;
        if rgba.len() != expected {
            anyhow::bail!(
                "unexpected frame buffer size: got {} bytes, expected {}",
                rgba.len(),
                expected
            );
        }

        let (in_w, in_h) = self.input_size;

        let mut tensor = Tensor::from_slice(rgba)
            .to_kind(Kind::Float)
            .view([height as i64, width as i64, 4])
            .narrow(2, 0, 3)
            .permute([2, 0, 1])
            .unsqueeze(0)
            / 255.0;

        if (width as i64, height as i64) != (in_w, in_h) {
            tensor = tensor.upsample_bilinear2d(&[in_h, in_w], false, None, None);
        }

        let tensor = tensor.to_device(self.device);

        Ok(tensor)
    }

    /// Executes the TorchScript module and performs basic confidence filtering.
    pub fn infer(&self, input: &Tensor) -> Result<DetectionBatch> {
        let output = self.module.forward_ts(&[input])?;
        let shape = output.size();
        if shape.len() != 3 {
            anyhow::bail!("unexpected detector output shape: {shape:?}");
        }
        let batch = shape[0];
        let channels = shape[1];
        let _num_preds = shape[2];
        if batch != 1 {
            anyhow::bail!("detector expected batch=1 but received {batch}");
        }
        if channels < 5 {
            anyhow::bail!(
                "detector output requires at least 5 channels (x,y,w,h,conf), got {channels}"
            );
        }

        let preds = output
            .to_device(Device::Cpu)
            .squeeze_dim(0)
            .permute([1, 0])
            .contiguous();

        let rows: Vec<Vec<f32>> = Vec::<Vec<f32>>::try_from(&preds)?;

        let mut detections = Vec::new();
        for row in rows {
            if row.len() < 5 {
                continue;
            }
            let score = row[4];
            if score < self.confidence_threshold {
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

        apply_nms(&mut detections, 0.45);

        Ok(DetectionBatch { detections })
    }
}

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
