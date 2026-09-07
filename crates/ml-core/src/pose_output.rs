//! Explicit YOLO26 NMS-free COCO pose output contract.

pub(crate) struct PoseDetection {
    pub bbox: [f32; 4],
    pub score: f32,
    pub keypoints: Vec<[f32; 3]>,
}

pub(crate) fn decode_pose_row(row: &[f32], threshold: f32) -> Option<PoseDetection> {
    if row.len() != 57
        || row[..6].iter().any(|v| !v.is_finite())
        || row[4] < threshold
        || row[4] > 1.0
        || row[5] != 0.0
        || row[2] <= row[0]
        || row[3] <= row[1]
    {
        return None;
    }
    let keypoints = row[6..]
        .chunks_exact(3)
        .map(|p| {
            if p.iter().all(|v| v.is_finite()) && (0.0..=1.0).contains(&p[2]) {
                [p[0], p[1], p[2]]
            } else {
                [0.0, 0.0, 0.0]
            }
        })
        .collect();
    Some(PoseDetection {
        bbox: [row[0], row[1], row[2], row[3]],
        score: row[4],
        keypoints,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn person_row() -> Vec<f32> {
        let mut row = vec![0.0; 57];
        row[..9].copy_from_slice(&[100.0, 50.0, 300.0, 600.0, 0.9, 0.0, 200.0, 70.0, 0.8]);
        row
    }

    #[test]
    fn preserves_xyxy_and_all_seventeen_points() {
        let p = decode_pose_row(&person_row(), 0.25).unwrap();
        assert_eq!(p.bbox, [100.0, 50.0, 300.0, 600.0]);
        assert_eq!(p.keypoints.len(), 17);
        assert_eq!(p.keypoints[0], [200.0, 70.0, 0.8]);
        assert_eq!(p.score, 0.9);
    }

    #[test]
    fn rejects_other_layouts_classes_and_invalid_boxes() {
        assert!(decode_pose_row(&[0.0; 56], 0.25).is_none());
        for (index, value) in [(4, 0.1), (4, f32::NAN), (5, 1.0), (2, 90.0)] {
            let mut row = person_row();
            row[index] = value;
            assert!(decode_pose_row(&row, 0.25).is_none());
        }
        let mut row = person_row();
        row[6] = f32::NAN;
        assert_eq!(decode_pose_row(&row, 0.25).unwrap().keypoints[0], [0.0; 3]);
    }
}
