//! Subtle current-frame instance silhouettes beneath the standard pose skeleton.
use super::data::DetectionSummary;
use ml_core::segmentation::PersonMask;

pub(crate) fn draw(
    detections: &[DetectionSummary],
    width: i32,
    height: i32,
    mut blend: impl FnMut(i32, i32, [u8; 3], u8),
) {
    for mask in detections
        .iter()
        .filter(|d| d.class == "PERSON")
        .filter_map(|d| d.silhouette.as_ref())
    {
        draw_mask(mask, width, height, &mut blend);
    }
}

fn draw_mask(
    mask: &PersonMask,
    width: i32,
    height: i32,
    blend: &mut impl FnMut(i32, i32, [u8; 3], u8),
) {
    if width <= 0
        || height <= 0
        || mask.width == 0
        || mask.height == 0
        || mask.width.checked_mul(mask.height) != Some(mask.logits.len())
        || mask.input_size.0 <= 0
        || mask.input_size.1 <= 0
    {
        return;
    }
    let sx = width as f32 / mask.input_size.0 as f32;
    let sy = height as f32 / mask.input_size.1 as f32;
    let bbox = [
        mask.bbox[0] * sx,
        mask.bbox[1] * sy,
        mask.bbox[2] * sx,
        mask.bbox[3] * sy,
    ];
    if !bbox.iter().all(|v| v.is_finite()) {
        return;
    }
    let dx = mask.width as f32 / width as f32;
    let dy = mask.height as f32 / height as f32;
    for y in (bbox[1].floor() as i32).max(0)..(bbox[3].ceil() as i32).min(height) {
        let py = ((y as f32 + 0.5) * dy - 0.5).clamp(0.0, (mask.height - 1) as f32);
        let y0 = py.floor() as usize;
        let y1 = (y0 + 1).min(mask.height - 1);
        let fy = py - y0 as f32;
        for x in (bbox[0].floor() as i32).max(0)..(bbox[2].ceil() as i32).min(width) {
            let px = ((x as f32 + 0.5) * dx - 0.5).clamp(0.0, (mask.width - 1) as f32);
            let x0 = px.floor() as usize;
            let x1 = (x0 + 1).min(mask.width - 1);
            let fx = px - x0 as f32;
            let a = mask.logits[y0 * mask.width + x0];
            let b = mask.logits[y0 * mask.width + x1];
            let c = mask.logits[y1 * mask.width + x0];
            let d = mask.logits[y1 * mask.width + x1];
            let top = a + (b - a) * fx;
            let bottom = c + (d - c) * fx;
            let value = top + (bottom - top) * fy;
            let gx = ((b - a) * (1.0 - fy) + (d - c) * fy) * dx;
            let gy = (bottom - top) * dy;
            // First-order distance to the zero-logit contour gives a one-pixel
            // antialiased edge. Thresholding follows bilinear resize, and the
            // independent segmentation box crops the result at image resolution.
            let distance = value / (gx * gx + gy * gy).sqrt().max(1e-5);
            let crop = (x as f32 + 0.5 - bbox[0])
                .min(bbox[2] - x as f32 - 0.5)
                .min(y as f32 + 0.5 - bbox[1])
                .min(bbox[3] - y as f32 - 0.5);
            if crop < 0.0 {
                continue;
            }
            let edge_distance = distance.min(crop);
            let fill = (edge_distance + 0.5).clamp(0.0, 1.0) * 18.0;
            let edge = (1.0 - edge_distance.abs()).clamp(0.0, 1.0) * 118.0;
            let alpha = fill.max(edge).round() as u8;
            if alpha > 0 {
                blend(x, y, super::pose::POSE_GREEN, alpha);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn raster(mask: &PersonMask, w: i32, h: i32) -> Vec<u8> {
        let mut pixels = vec![0; (w * h) as usize];
        draw_mask(mask, w, h, &mut |x, y, _, a| {
            pixels[(y * w + x) as usize] = a
        });
        pixels
    }
    #[test]
    fn mask_resizes_to_capture_aspect_and_crops_without_box_fill() {
        let mut mask = PersonMask {
            bbox: [1., 0., 3., 4.],
            score: 0.9,
            width: 4,
            height: 4,
            input_size: (4, 4),
            logits: vec![10.; 16],
        };
        let p = raster(&mask, 8, 4);
        assert_eq!(p[0], 0);
        assert!(p[3] > 0);
        assert_eq!(p[6], 0);
        assert!(p[3 * 8 + 3] > 0);
        mask.logits.fill(-10.);
        assert!(raster(&mask, 8, 4).iter().all(|p| *p == 0));
    }
    #[test]
    fn mask_preserves_background_holes_and_rejects_invalid_storage() {
        let mut mask = PersonMask {
            bbox: [0., 0., 5., 5.],
            score: 0.9,
            width: 5,
            height: 5,
            input_size: (5, 5),
            logits: vec![10.; 25],
        };
        for y in 1..4 {
            for x in 1..4 {
                mask.logits[y * 5 + x] = -10.;
            }
        }
        assert_eq!(raster(&mask, 5, 5)[12], 0);
        mask.logits.pop();
        assert!(raster(&mask, 5, 5).iter().all(|p| *p == 0));
    }
}
