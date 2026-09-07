//! Draw only visible COCO joints and links, in capture-image coordinates.

use super::data::DetectionSummary;

pub(crate) const POSE_GREEN: [u8; 3] = [82, 230, 123];
const POSE_LINE_WIDTH: f32 = 2.0;

pub(crate) const KEYPOINT_NAMES: [&str; 17] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
];

const LINKS: [(usize, usize); 19] = [
    (1, 2), // Ultralytics COCO skeleton also links the two eyes.
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
];

/// Sparse drawing avoids another image conversion on the GPU encode path:
/// its owned BGR snapshot already exists before NVJPEG encoding.
pub(crate) fn draw_poses(
    detections: &[DetectionSummary],
    width: i32,
    height: i32,
    mut pixel: impl FnMut(i32, i32, [u8; 3], u8),
) {
    for det in detections {
        if det.keypoints.len() != 17 {
            continue;
        }
        let point = |i: usize| {
            let raw = &det.keypoints[i];
            let p = det.display_keypoints.get(i).unwrap_or(raw);
            (raw.x >= 0.0
                && raw.x < width as f32
                && raw.y >= 0.0
                && raw.y < height as f32
                && p.confidence >= 0.5
                && p.x.is_finite()
                && p.y.is_finite()
                && p.x >= 0.0
                && p.x < width as f32
                && p.y >= 0.0
                && p.y < height as f32)
                .then(|| (p.x as i32, p.y as i32))
        };
        for (a, b) in LINKS {
            if let (Some((x0, y0)), Some((x1, y1))) = (point(a), point(b)) {
                line(x0, y0, x1, y1, |x, y| {
                    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                        if x + dx >= 0 && y + dy >= 0 && x + dx < width && y + dy < height {
                            pixel(x + dx, y + dy, [9, 18, 14], 150);
                        }
                    }
                });
            }
        }
        // Draw every halo first so intersecting bones retain their green core.
        for (a, b) in LINKS {
            if let (Some((x0, y0)), Some((x1, y1))) = (point(a), point(b)) {
                antialiased_line(x0, y0, x1, y1, |x, y, alpha| {
                    if x >= 0 && y >= 0 && x < width && y < height {
                        pixel(x, y, POSE_GREEN, alpha);
                    }
                });
            }
        }
        for i in 0..17 {
            if let Some((x, y)) = point(i) {
                for dy in -3..=3 {
                    for dx in -3..=3 {
                        if x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height {
                            let radius = ((dx * dx + dy * dy) as f32).sqrt();
                            if radius <= 2.8 {
                                pixel(x + dx, y + dy, [9, 18, 14], 180);
                                let coverage = (1.0 - (radius - 1.8).abs()).clamp(0.0, 1.0);
                                if coverage > 0.0 {
                                    pixel(x + dx, y + dy, POSE_GREEN, (coverage * 255.0) as u8);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Coverage along the major axis keeps subpixel edges smooth without a full image pass.
fn antialiased_line(x0: i32, y0: i32, x1: i32, y1: i32, mut pixel: impl FnMut(i32, i32, u8)) {
    let steep = (y1 - y0).abs() > (x1 - x0).abs();
    let ((mut a, mut b), (mut c, mut d)) = if steep {
        ((y0, x0), (y1, x1))
    } else {
        ((x0, y0), (x1, y1))
    };
    if a > c {
        std::mem::swap(&mut a, &mut c);
        std::mem::swap(&mut b, &mut d);
    }
    let gradient = if c == a {
        0.0
    } else {
        (d - b) as f32 / (c - a) as f32
    };
    // Correct for slope so diagonal bones have the same apparent thickness.
    let half_width = POSE_LINE_WIDTH * 0.5 * (1.0 + gradient * gradient).sqrt();
    let extent = (half_width + 0.5).ceil() as i32;
    for major in a..=c {
        let minor = b as f32 + (major - a) as f32 * gradient;
        let base = minor.floor() as i32;
        for offset in -extent..=extent {
            let distance = ((base + offset) as f32 - minor).abs();
            let alpha = ((half_width + 0.5 - distance).clamp(0.0, 1.0) * 255.0) as u8;
            if alpha > 0 {
                if steep {
                    pixel(base + offset, major, alpha);
                } else {
                    pixel(major, base + offset, alpha);
                }
            }
        }
    }
}

fn line(mut x: i32, mut y: i32, end_x: i32, end_y: i32, mut pixel: impl FnMut(i32, i32)) {
    let dx = (end_x - x).abs();
    let sx = if x < end_x { 1 } else { -1 };
    let dy = -(end_y - y).abs();
    let sy = if y < end_y { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        pixel(x, y);
        if x == end_x && y == end_y {
            break;
        }
        let e = 2 * error;
        if e >= dy {
            error += dy;
            x += sx;
        }
        if e <= dx {
            error += dx;
            y += sy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::data::PoseKeypoint;

    #[test]
    fn hidden_or_offscreen_joints_do_not_draw_fake_limbs() {
        let mut det = DetectionSummary {
            class: "PERSON".into(),
            score: 0.9,
            bbox: [0.0, 0.0, 50.0, 50.0],
            track_id: 1,
            track_state: "TRACK",
            display_bbox: [0.0, 0.0, 50.0, 50.0],
            display_keypoints: Vec::new(),
            person_track_id: None,
            silhouette_score: None,
            silhouette: None,
            keypoints: KEYPOINT_NAMES
                .iter()
                .map(|name| PoseKeypoint {
                    name,
                    x: 10.0,
                    y: 10.0,
                    confidence: 0.1,
                })
                .collect(),
        };
        let mut pixels = Vec::new();
        draw_poses(&[det.clone()], 50, 50, |x, y, _, _| pixels.push((x, y)));
        assert!(pixels.is_empty());
        det.keypoints[5].confidence = 0.9;
        det.keypoints[6].confidence = 0.9;
        det.keypoints[6].x = 30.0;
        det.keypoints[7].confidence = 0.9;
        det.keypoints[7].x = -20.0;
        draw_poses(&[det], 50, 50, |x, y, _, _| pixels.push((x, y)));
        assert!(pixels.contains(&(20, 10)));
        assert!(
            pixels
                .iter()
                .all(|(x, y)| (0..50).contains(x) && (0..50).contains(y))
        );
    }
}
