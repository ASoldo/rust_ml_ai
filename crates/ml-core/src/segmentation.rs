//! YOLO26 instance masks, kept at prototype resolution until image-space drawing.
use crate::DetectionBatch;
use anyhow::{Result, bail};
use std::sync::Arc;
use tch::{CModule, Device, IValue, Kind, Tensor, no_grad};

#[derive(Debug, Clone)]
pub struct PersonMask {
    /// Box in detector pixels, independent of the pose detector's box.
    pub bbox: [f32; 4],
    pub score: f32,
    pub width: usize,
    pub height: usize,
    pub input_size: (i64, i64),
    /// Unthresholded prototype logits. Bilinear resize precedes threshold/crop,
    /// matching Ultralytics process_mask(upsample=True).
    pub logits: Vec<f32>,
}

pub(crate) fn infer(
    module: &CModule,
    input: &Tensor,
    threshold: f32,
) -> Result<Vec<Vec<Arc<PersonMask>>>> {
    let (rows, protos): (Tensor, Tensor) =
        no_grad(|| module.forward_is(&[IValue::Tensor(input.shallow_clone())]))?.try_into()?;
    let shape = rows.size();
    let ps = protos.size();
    let ins = input.size();
    if shape != [ins[0], 300, 38] || ps != [ins[0], 32, 160, 160] || ins[2..] != [640, 640] {
        bail!(
            "segmentation export requires [B,300,38] rows and [B,32,160,160] prototypes; got {shape:?}, {ps:?}"
        );
    }
    let mut batches = Vec::new();
    for b in 0..shape[0] {
        let predictions = rows.get(b);
        let indices = predictions
            .select(1, 4)
            .ge(threshold as f64)
            .logical_and(&predictions.select(1, 5).eq(0))
            .nonzero()
            .squeeze_dim(1);
        // The end-to-end export is score ordered. Bound mask memory/work even
        // in crowded scenes; unmatched people retain their boxes and pose.
        let count = indices.size()[0].min(16);
        if count == 0 {
            batches.push(Vec::new());
            continue;
        }
        let selected = predictions.index_select(0, &indices.narrow(0, 0, count));
        let logits = no_grad(|| {
            selected
                .narrow(1, 6, 32)
                .matmul(&protos.get(b).view([32, -1]))
        })
        .to_device(Device::Cpu)
        .to_kind(Kind::Float)
        .contiguous();
        let metadata = Vec::<Vec<f32>>::try_from(&selected.narrow(1, 0, 6).to_device(Device::Cpu))?;
        let values = Vec::<Vec<f32>>::try_from(&logits)?;
        let masks = metadata
            .into_iter()
            .zip(values)
            .filter_map(|(r, logits)| {
                if !r.iter().all(|v| v.is_finite())
                    || r[4] > 1.0
                    || r[2] <= r[0]
                    || r[3] <= r[1]
                    || !logits.iter().all(|v| v.is_finite())
                {
                    return None;
                }
                Some(Arc::new(PersonMask {
                    bbox: [r[0], r[1], r[2], r[3]],
                    score: r[4],
                    width: 160,
                    height: 160,
                    input_size: (640, 640),
                    logits,
                }))
            })
            .collect();
        batches.push(masks);
    }
    Ok(batches)
}

/// Associate each mask to at most one current person pose using raw geometry.
/// Never borrow a previous frame's mask or invent a mask for an unmatched box.
pub fn attach_person_masks(batch: &mut DetectionBatch, masks: Vec<Arc<PersonMask>>) {
    let mut candidates = Vec::new();
    for (d, detection) in batch
        .detections
        .iter()
        .enumerate()
        .filter(|(_, d)| d.class_id == 1)
    {
        for (m, mask) in masks.iter().enumerate() {
            let overlap = iou(detection.bbox, mask.bbox);
            if overlap >= 0.5 {
                candidates.push((overlap, d, m));
            }
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut used_d = vec![false; batch.detections.len()];
    let mut used_m = vec![false; masks.len()];
    for (_, d, m) in candidates {
        if !used_d[d] && !used_m[m] {
            batch.detections[d].mask = Some(masks[m].clone());
            used_d[d] = true;
            used_m[m] = true;
        }
    }
}

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let intersection =
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let area = |r: [f32; 4]| (r[2] - r[0]).max(0.0) * (r[3] - r[1]).max(0.0);
    intersection / (area(a) + area(b) - intersection).max(1e-6)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Detection;
    #[test]
    fn masks_match_people_one_to_one_never_faces_or_distant_boxes() {
        let make = |class_id, bbox| Detection {
            class_id,
            bbox,
            ..Default::default()
        };
        let mut batch = DetectionBatch {
            detections: vec![
                make(0, [0., 0., 10., 10.]),
                make(1, [0., 0., 10., 10.]),
                make(1, [1., 0., 11., 10.]),
                make(1, [100., 100., 200., 200.]),
            ],
        };
        let mask = Arc::new(PersonMask {
            bbox: [0., 0., 10., 10.],
            score: 0.9,
            width: 1,
            height: 1,
            input_size: (640, 640),
            logits: vec![1.],
        });
        attach_person_masks(&mut batch, vec![mask]);
        assert!(batch.detections[1].mask.is_some());
        assert_eq!(
            batch.detections.iter().filter(|d| d.mask.is_some()).count(),
            1
        );
    }
}
