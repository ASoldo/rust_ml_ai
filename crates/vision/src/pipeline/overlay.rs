//! Sparse image-space annotations. Color denotes observation state, never affiliation.

use super::data::DetectionSummary;

const INK: [u8; 3] = [178, 242, 191];
const DIM: [u8; 3] = [162, 196, 171];
const CAUTION: [u8; 3] = [240, 196, 104];
const BACK: [u8; 3] = [9, 18, 14];

pub(crate) fn draw(
    bgr: &mut [u8],
    width: i32,
    height: i32,
    frame: u64,
    fps: f32,
    timestamp_ms: i64,
    detections: &[DetectionSummary],
) {
    let mut canvas = Canvas { bgr, width, height };
    super::silhouette::draw(detections, width, height, |x, y, rgb, alpha| {
        canvas.blend(x, y, rgb, alpha)
    });
    super::pose::draw_poses(detections, width, height, |x, y, rgb, alpha| {
        canvas.blend(x, y, rgb, alpha)
    });
    let scale = if width >= 960 { 2 } else { 1 };
    let margin = 8 * scale;
    let people = detections.iter().filter(|d| d.class == "PERSON").count();
    let faces = detections.iter().filter(|d| d.class == "FACE").count();
    let seconds = timestamp_ms.div_euclid(1000).rem_euclid(86400);
    // Small corner readouts preserve the image between them, including at its edges.
    let source = "VISION / CAM 01".to_string();
    let count = format!("PERSON {people:02}  FACE {faces:02}");
    let clock = format!(
        "{:02}:{:02}:{:02}Z  {:04.1} FPS",
        seconds / 3600,
        seconds / 60 % 60,
        seconds % 60,
        fps
    );
    let sequence = format!("FR {frame:06}");
    let chip_h = 7 * scale + 10;
    let mut occupied = Vec::new();
    for (text, right, bottom) in [
        (&source, false, false),
        (&count, true, false),
        (&clock, false, true),
        (&sequence, true, true),
    ] {
        let w = (text.len() as i32 * 6 * scale + 12).min(width);
        let x = if right { width - margin - w } else { margin }.clamp(0, (width - w).max(0));
        let y = if bottom {
            height - margin - chip_h
        } else {
            margin
        }
        .clamp(0, (height - chip_h).max(0));
        let rect = [x, y, x + w, y + chip_h];
        if occupied.iter().any(|b| overlap(rect, *b) > 0) {
            continue;
        }
        occupied.push(rect);
        canvas.panel(rect, 160);
        canvas.text(x + 6, y + 5, text, scale, if right { DIM } else { INK });
    }
    for det in detections {
        let color = if det.track_state == "TRACK" {
            super::pose::POSE_GREEN
        } else {
            CAUTION
        };
        canvas.brackets(det.display_bbox, color, det.class == "PERSON");
    }
    let face_regions: Vec<_> = detections
        .iter()
        .filter(|d| d.class == "FACE")
        .map(|d| d.display_bbox.map(|v| v.round() as i32))
        .collect();
    let joint_regions: Vec<_> = detections
        .iter()
        .flat_map(|d| d.keypoints.iter())
        .filter(|p| visible_point(p, width, height))
        .map(|p| {
            [
                p.x as i32 - 4,
                p.y as i32 - 4,
                p.x as i32 + 5,
                p.y as i32 + 5,
            ]
        })
        .collect();
    for class in ["PERSON", "FACE", "OBJECT"] {
        for det in detections.iter().filter(|d| d.class == class) {
            let color = if det.track_state == "TRACK" {
                super::pose::POSE_GREEN
            } else {
                CAUTION
            };
            let id = det.person_track_id.unwrap_or(det.track_id);
            let label = format!("{} T{:03} {:02.0}%", det.class, id, det.score * 100.0);
            let detail = if det.class == "PERSON" {
                let visible = det
                    .keypoints
                    .iter()
                    .filter(|p| visible_point(p, width, height))
                    .count();
                format!("{}  POSE {:02}/17", det.track_state, visible)
            } else {
                det.track_state.into()
            };
            let w = ((label.len().max(detail.len()) as i32 * 6 * scale) + 12).min(width);
            let h = 17 * scale + 10;
            let bounds @ [left, top, right, bottom] = det.display_bbox.map(|v| v.round() as i32);
            let candidates = [
                (left, top - h - 6),
                (right + 8, top),
                (left - w - 8, top),
                (left, bottom + 6),
                (right + 8, bottom - h),
                (left - w - 8, bottom - h),
                (left + 5, top + 5),
                (width - w - margin, margin + chip_h + 5),
                (margin, margin + chip_h + 5),
                (margin, height - margin - chip_h - h - 5),
            ];
            let mut best = None;
            for (x, y) in candidates {
                let x = x.clamp(0, (width - w).max(0));
                let y = y.clamp(0, (height - h).max(0));
                let rect = [x, y, x + w, y + h];
                let gap_x = (bounds[0] - rect[2]).max(rect[0] - bounds[2]).max(0);
                let gap_y = (bounds[1] - rect[3]).max(rect[1] - bounds[3]).max(0);
                // Distant detached tags can appear to describe a different person.
                // Suppress a crowded label; its detection remains in the API.
                if gap_x + gap_y > 32 {
                    continue;
                }
                let cost = occupied.iter().map(|b| overlap(rect, *b) * 12).sum::<i32>()
                    + face_regions
                        .iter()
                        .map(|b| overlap(rect, *b) * 8)
                        .sum::<i32>()
                    + joint_regions
                        .iter()
                        .map(|b| overlap(rect, *b) * 6)
                        .sum::<i32>()
                    + overlap(rect, bounds);
                if best.map(|(_, c)| cost < c).unwrap_or(true) {
                    best = Some((rect, cost));
                }
            }
            if let Some((rect, cost)) = best {
                if cost > w * h * 3 {
                    continue;
                }
                // A short leader anchors displaced labels without crossing the subject.
                canvas.leader(rect, bounds, color);
                canvas.panel(rect, 194);
                canvas.fill([rect[0], rect[1], rect[0] + 2, rect[3]], color);
                canvas.text(rect[0] + 6, rect[1] + 5, &label, scale, INK);
                canvas.text(rect[0] + 6, rect[1] + 5 + 10 * scale, &detail, scale, DIM);
                occupied.push(rect);
            }
        }
    }
}

fn visible_point(p: &&super::data::PoseKeypoint, width: i32, height: i32) -> bool {
    p.confidence >= 0.5
        && p.x.is_finite()
        && p.y.is_finite()
        && p.x >= 0.0
        && p.x < width as f32
        && p.y >= 0.0
        && p.y < height as f32
}

fn overlap(a: [i32; 4], b: [i32; 4]) -> i32 {
    (a[2].min(b[2]) - a[0].max(b[0])).max(0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0)
}

struct Canvas<'a> {
    bgr: &'a mut [u8],
    width: i32,
    height: i32,
}
impl Canvas<'_> {
    fn blend(&mut self, x: i32, y: i32, rgb: [u8; 3], alpha: u8) {
        if x < 0 || y < 0 || x >= self.width || y >= self.height {
            return;
        }
        let p = (y as usize * self.width as usize + x as usize) * 3;
        if p + 3 > self.bgr.len() {
            return;
        }
        for (channel, value) in [rgb[2], rgb[1], rgb[0]].into_iter().enumerate() {
            self.bgr[p + channel] = ((self.bgr[p + channel] as u32 * (255 - alpha as u32)
                + value as u32 * alpha as u32
                + 127)
                / 255) as u8;
        }
    }
    fn panel(&mut self, b: [i32; 4], alpha: u8) {
        for y in b[1].max(0)..b[3].min(self.height) {
            for x in b[0].max(0)..b[2].min(self.width) {
                self.blend(x, y, BACK, alpha);
            }
        }
    }
    fn leader(&mut self, label: [i32; 4], bounds: [i32; 4], color: [u8; 3]) {
        let [l, t, r, b] = bounds;
        if label[0] >= r && label[0] - r <= 32 {
            let y = (label[1] + label[3]) / 2;
            if y >= t && y <= b {
                for x in r..label[0] {
                    self.blend(x, y, color, 180);
                }
            }
        } else if label[2] <= l && l - label[2] <= 32 {
            let y = (label[1] + label[3]) / 2;
            if y >= t && y <= b {
                for x in label[2]..l {
                    self.blend(x, y, color, 180);
                }
            }
        } else if label[3] <= t && t - label[3] <= 32 {
            let x = (label[0] + 6).clamp(l, r.max(l));
            for y in label[3]..t {
                self.blend(x, y, color, 180);
            }
        } else if label[1] >= b && label[1] - b <= 32 {
            let x = (label[0] + 6).clamp(l, r.max(l));
            for y in b..label[1] {
                self.blend(x, y, color, 180);
            }
        }
    }
    fn pixel(&mut self, x: i32, y: i32, rgb: [u8; 3]) {
        if x < 0 || y < 0 || x >= self.width || y >= self.height {
            return;
        }
        let p = (y as usize * self.width as usize + x as usize) * 3;
        if p + 3 <= self.bgr.len() {
            self.bgr[p..p + 3].copy_from_slice(&[rgb[2], rgb[1], rgb[0]]);
        }
    }
    fn fill(&mut self, b: [i32; 4], rgb: [u8; 3]) {
        for y in b[1].max(0)..b[3].min(self.height) {
            for x in b[0].max(0)..b[2].min(self.width) {
                self.pixel(x, y, rgb);
            }
        }
    }
    fn brackets(&mut self, b: [f32; 4], rgb: [u8; 3], person: bool) {
        let [l, t, r, b] = [
            b[0].clamp(0.0, (self.width - 1).max(0) as f32),
            b[1].clamp(0.0, (self.height - 1).max(0) as f32),
            b[2].clamp(0.0, (self.width - 1).max(0) as f32),
            b[3].clamp(0.0, (self.height - 1).max(0) as f32),
        ]
        .map(|v| v.round() as i32);
        let n = if person {
            ((r - l).min(b - t) / 6).clamp(5, 18)
        } else {
            ((r - l).min(b - t) / 4).clamp(3, 8)
        };
        for (x, y, sx, sy) in [(l, t, 1, 1), (r, t, -1, 1), (l, b, 1, -1), (r, b, -1, -1)] {
            for i in 0..n {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        self.blend(x + sx * i + dx, y + dy, BACK, 130);
                        self.blend(x + dx, y + sy * i + dy, BACK, 130);
                    }
                }
            }
            for i in 0..n {
                self.pixel(x + sx * i, y, rgb);
                self.pixel(x, y + sy * i, rgb);
            }
        }
    }
    fn text(&mut self, mut x: i32, y: i32, text: &str, scale: i32, color: [u8; 3]) {
        for ch in text.chars() {
            let glyph = glyph(ch);
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..5 {
                    if bits & (1 << (4 - col)) != 0 {
                        self.fill(
                            [
                                x + col * scale,
                                y + row as i32 * scale,
                                x + (col + 1) * scale,
                                y + (row as i32 + 1) * scale,
                            ],
                            color,
                        );
                    }
                }
            }
            x += 6 * scale;
        }
    }
}

fn glyph(c: char) -> [u8; 7] {
    const LETTERS: [[u8; 7]; 26] = [
        [14, 17, 17, 31, 17, 17, 17],
        [30, 17, 17, 30, 17, 17, 30],
        [14, 17, 16, 16, 16, 17, 14],
        [30, 17, 17, 17, 17, 17, 30],
        [31, 16, 16, 30, 16, 16, 31],
        [31, 16, 16, 30, 16, 16, 16],
        [14, 17, 16, 23, 17, 17, 15],
        [17, 17, 17, 31, 17, 17, 17],
        [14, 4, 4, 4, 4, 4, 14],
        [7, 2, 2, 2, 18, 18, 12],
        [17, 18, 20, 24, 20, 18, 17],
        [16, 16, 16, 16, 16, 16, 31],
        [17, 27, 21, 21, 17, 17, 17],
        [17, 25, 21, 19, 17, 17, 17],
        [14, 17, 17, 17, 17, 17, 14],
        [30, 17, 17, 30, 16, 16, 16],
        [14, 17, 17, 17, 21, 18, 13],
        [30, 17, 17, 30, 20, 18, 17],
        [15, 16, 16, 14, 1, 1, 30],
        [31, 4, 4, 4, 4, 4, 4],
        [17, 17, 17, 17, 17, 17, 14],
        [17, 17, 17, 17, 17, 10, 4],
        [17, 17, 17, 21, 21, 21, 10],
        [17, 17, 10, 4, 10, 17, 17],
        [17, 17, 10, 4, 4, 4, 4],
        [31, 1, 2, 4, 8, 16, 31],
    ];
    const NUMBERS: [[u8; 7]; 10] = [
        [14, 17, 19, 21, 25, 17, 14],
        [4, 12, 4, 4, 4, 4, 14],
        [14, 17, 1, 2, 4, 8, 31],
        [30, 1, 1, 14, 1, 1, 30],
        [2, 6, 10, 18, 31, 2, 2],
        [31, 16, 16, 30, 1, 1, 30],
        [6, 8, 16, 30, 17, 17, 14],
        [31, 1, 2, 4, 8, 8, 8],
        [14, 17, 17, 14, 17, 17, 14],
        [14, 17, 17, 15, 1, 2, 12],
    ];
    match c {
        'A'..='Z' => LETTERS[c as usize - 'A' as usize],
        '0'..='9' => NUMBERS[c as usize - '0' as usize],
        ':' => [0, 4, 4, 0, 4, 4, 0],
        '.' => [0, 0, 0, 0, 0, 6, 6],
        '/' => [1, 2, 2, 4, 8, 8, 16],
        '+' => [0, 4, 4, 31, 4, 4, 0],
        '-' => [0, 0, 0, 31, 0, 0, 0],
        '%' => [17, 2, 4, 8, 17, 0, 0],
        _ => [0; 7],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn edge_labels_and_small_frames_do_not_write_out_of_bounds() {
        for (w, h) in [(1, 1), (160, 96), (640, 480)] {
            let mut image = vec![80; (w * h * 3) as usize];
            let det = DetectionSummary {
                class: "PERSON".into(),
                score: 0.8,
                bbox: [0., 0., w as f32 - 1., h as f32 - 1.],
                display_bbox: [0., 0., w as f32 - 1., h as f32 - 1.],
                track_id: 1,
                track_state: "TRACK",
                keypoints: vec![],
                display_keypoints: vec![],
                person_track_id: None,
                silhouette_score: None,
                silhouette: None,
            };
            draw(&mut image, w, h, 1, 15., 1000, &[det]);
            assert_eq!(image.len(), (w * h * 3) as usize);
        }
    }
}
