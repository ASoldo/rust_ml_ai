use std::{convert::TryFrom, path::Path};

use anyhow::Result;
use tch::{self, Device, Kind, Tensor};

/// Single detection returned by the detector.
#[derive(Debug, Clone, Default)]
pub struct Detection {
    pub bbox_xywh: [f32; 4],
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
        if (width as i64, height as i64) != (in_w, in_h) {
            anyhow::bail!(
                "frame size {width}x{height} does not match detector input {in_w}x{in_h}"
            );
        }

        let tensor = Tensor::from_slice(rgba)
            .to_device(self.device)
            .to_kind(Kind::Float)
            .view([1, in_h, in_w, 4])
            .narrow(3, 0, 3)
            .permute([0, 3, 1, 2])
            / 255.0;

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
            let bbox = [row[0], row[1], row[2], row[3]];
            let class_id = if row.len() > 5 { row[5] as i64 } else { 0 };
            detections.push(Detection {
                bbox_xywh: bbox,
                score,
                class_id,
            });
            if detections.len() >= 512 {
                break;
            }
        }

        Ok(DetectionBatch { detections })
    }
}
