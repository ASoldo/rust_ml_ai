//! Bounded, motion-only association for the current camera session.
//! No appearance embeddings, identity inference, or drawing of missing detections.

use super::data::{DetectionSummary, PoseKeypoint};

const MAX_GAP_MS: i64 = 1000;
const MAX_TRACKS: usize = 256;

struct Track {
    id: i64,
    class: String,
    bbox: [f32; 4],
    display_bbox: [f32; 4],
    points: Vec<PoseKeypoint>,
    velocity: [f32; 4],
    timestamp_ms: i64,
    hits: u32,
}

#[derive(Default)]
pub(crate) struct Tracker {
    next_id: i64,
    tracks: Vec<Track>,
    last_frame: u64,
    last_timestamp_ms: i64,
    dimensions: (i32, i32),
}

impl Tracker {
    pub(crate) fn update(
        &mut self,
        frame: u64,
        timestamp_ms: i64,
        dimensions: (i32, i32),
        detections: &mut Vec<DetectionSummary>,
    ) {
        // Suppress only very high-overlap predictions within the same class.
        // Faces inside person boxes remain independent detections.
        detections.sort_by(|a, b| b.score.total_cmp(&a.score));
        let mut keep: Vec<DetectionSummary> = Vec::with_capacity(detections.len());
        for det in detections.drain(..) {
            if !keep
                .iter()
                .any(|k| k.class == det.class && iou(k.bbox, det.bbox) > 0.8)
            {
                keep.push(det);
            }
        }
        *detections = keep;
        // Parallel workers may finish out of order. Such frames must not move
        // an existing track backwards or corrupt future association.
        if frame <= self.last_frame && self.last_frame != 0 {
            for det in detections {
                self.next_id += 1;
                det.track_id = self.next_id;
                det.track_state = "UNTRACKED";
            }
            return;
        }
        if dimensions != self.dimensions || timestamp_ms < self.last_timestamp_ms {
            self.tracks.clear();
        }
        self.last_frame = frame;
        self.last_timestamp_ms = timestamp_ms;
        self.dimensions = dimensions;
        self.tracks
            .retain(|t| timestamp_ms - t.timestamp_ms <= MAX_GAP_MS);

        let mut pairs = Vec::new();
        for (di, det) in detections.iter().enumerate() {
            for (ti, track) in self.tracks.iter().enumerate() {
                if track.class != det.class {
                    continue;
                }
                let dt = ((timestamp_ms - track.timestamp_ms) as f32 / 1000.0).clamp(0.0, 0.35);
                let motion = std::array::from_fn(|i| track.bbox[i] + track.velocity[i] * dt);
                // Network bursts make arrival-time velocity noisy. The most
                // recent observed box remains a valid association hypothesis.
                let predicted = if iou(motion, det.bbox) > iou(track.bbox, det.bbox) {
                    motion
                } else {
                    track.bbox
                };
                let overlap = iou(predicted, det.bbox);
                let w = (predicted[2] - predicted[0]).max(1.0);
                let h = (predicted[3] - predicted[1]).max(1.0);
                let dx =
                    ((det.bbox[0] + det.bbox[2] - predicted[0] - predicted[2]) * 0.5 / w).abs();
                let dy =
                    ((det.bbox[1] + det.bbox[3] - predicted[1] - predicted[3]) * 0.5 / h).abs();
                let ratio = area(det.bbox) / area(predicted).max(1.0);
                if overlap >= 0.15 && dx < 0.75 && dy < 0.75 && (0.35..=2.85).contains(&ratio) {
                    pairs.push((overlap - 0.1 * (dx + dy), di, ti));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut assigned = vec![false; detections.len()];
        let mut used = vec![false; self.tracks.len()];
        for (_, di, ti) in pairs {
            if assigned[di] || used[ti] {
                continue;
            }
            assigned[di] = true;
            used[ti] = true;
            let det = &mut detections[di];
            let track = &mut self.tracks[ti];
            let dt = ((timestamp_ms - track.timestamp_ms) as f32 / 1000.0).max(0.001);
            // Less smoothing after a gap or a large motion to avoid lagging boxes.
            let alpha = if dt > 0.2 || iou(track.bbox, det.bbox) < 0.5 {
                1.0
            } else {
                0.7
            };
            for i in 0..4 {
                let extent = if i % 2 == 0 {
                    track.bbox[2] - track.bbox[0]
                } else {
                    track.bbox[3] - track.bbox[1]
                }
                .max(1.0);
                let velocity = ((det.bbox[i] - track.bbox[i]) / dt.max(1.0 / 60.0))
                    .clamp(-extent * 3.0, extent * 3.0);
                track.velocity[i] = 0.5 * track.velocity[i] + 0.5 * velocity;
                det.display_bbox[i] = alpha * det.bbox[i] + (1.0 - alpha) * track.display_bbox[i];
            }
            for (point, previous) in det.display_keypoints.iter_mut().zip(&track.points) {
                // Confidence always comes from this frame. Never resurrect a hidden joint.
                if point.confidence >= 0.5 && previous.confidence >= 0.5 {
                    point.x = alpha * point.x + (1.0 - alpha) * previous.x;
                    point.y = alpha * point.y + (1.0 - alpha) * previous.y;
                }
            }
            track.bbox = det.bbox;
            track.display_bbox = det.display_bbox;
            track.points = det.display_keypoints.clone();
            track.timestamp_ms = timestamp_ms;
            track.hits = track.hits.saturating_add(1);
            det.track_id = track.id;
            det.track_state = if track.hits >= 3 { "TRACK" } else { "NEW" };
        }
        for (i, det) in detections.iter_mut().enumerate() {
            if assigned[i] {
                continue;
            }
            self.next_id += 1;
            det.track_id = self.next_id;
            det.track_state = "NEW";
            if self.tracks.len() < MAX_TRACKS {
                self.tracks.push(Track {
                    id: det.track_id,
                    class: det.class.clone(),
                    bbox: det.bbox,
                    display_bbox: det.display_bbox,
                    points: det.display_keypoints.clone(),
                    velocity: [0.0; 4],
                    timestamp_ms,
                    hits: 1,
                });
            }
        }
        // Link a face only when one current person box plausibly contains it.
        let people: Vec<_> = detections
            .iter()
            .filter(|d| d.class == "PERSON")
            .map(|d| (d.track_id, d.bbox))
            .collect();
        for face in detections.iter_mut().filter(|d| d.class == "FACE") {
            let x = (face.bbox[0] + face.bbox[2]) * 0.5;
            let y = (face.bbox[1] + face.bbox[3]) * 0.5;
            let candidates: Vec<_> = people
                .iter()
                .filter(|(_, b)| {
                    x >= b[0]
                        && x <= b[2]
                        && y >= b[1]
                        && y <= b[3]
                        && face.bbox[3] - face.bbox[1] < (b[3] - b[1]) * 0.55
                })
                .collect();
            if candidates.len() == 1 {
                face.person_track_id = Some(candidates[0].0);
            }
        }
    }
}

fn area(b: [f32; 4]) -> f32 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let intersection = area([
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]);
    intersection / (area(a) + area(b) - intersection).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn det(class: &str, bbox: [f32; 4]) -> DetectionSummary {
        DetectionSummary {
            class: class.into(),
            score: 0.8,
            bbox,
            display_bbox: bbox,
            track_id: 0,
            track_state: "NEW",
            keypoints: vec![],
            display_keypoints: vec![],
            person_track_id: None,
            silhouette_score: None,
            silhouette: None,
        }
    }
    #[test]
    fn stable_ids_survive_brief_misses_without_drawing_ghosts() {
        let mut tracker = Tracker::default();
        let mut a = vec![det("PERSON", [10., 10., 100., 200.])];
        tracker.update(1, 1000, (640, 480), &mut a);
        let id = a[0].track_id;
        let mut empty = vec![];
        tracker.update(2, 1067, (640, 480), &mut empty);
        assert!(empty.is_empty());
        let mut b = vec![det("PERSON", [15., 10., 105., 200.])];
        tracker.update(3, 1134, (640, 480), &mut b);
        assert_eq!(b[0].track_id, id);
        let mut c = vec![det("PERSON", [15., 10., 105., 200.])];
        tracker.update(4, 3000, (640, 480), &mut c);
        assert_ne!(c[0].track_id, id);
    }
    #[test]
    fn duplicates_are_removed_without_suppressing_faces_or_other_people() {
        let mut tracker = Tracker::default();
        let mut rows = vec![
            det("PERSON", [10., 10., 110., 300.]),
            det("PERSON", [12., 10., 112., 300.]),
            det("FACE", [35., 20., 65., 60.]),
            det("PERSON", [200., 10., 300., 300.]),
        ];
        tracker.update(1, 1000, (640, 480), &mut rows);
        assert_eq!(rows.len(), 3);
        let person = rows.iter().find(|r| r.class == "PERSON").unwrap();
        let face = rows.iter().find(|r| r.class == "FACE").unwrap();
        assert_eq!(face.person_track_id, Some(person.track_id));
        assert_ne!(face.track_id, person.track_id);
    }
    #[test]
    fn stale_worker_results_do_not_rewind_tracks() {
        let mut tracker = Tracker::default();
        let mut rows = vec![det("PERSON", [10., 10., 110., 300.])];
        tracker.update(10, 1000, (640, 480), &mut rows);
        let id = rows[0].track_id;
        let mut stale = vec![det("PERSON", [200., 10., 300., 300.])];
        tracker.update(9, 934, (640, 480), &mut stale);
        assert_eq!(stale[0].track_state, "UNTRACKED");
        tracker.update(11, 1067, (640, 480), &mut rows);
        assert_eq!(rows[0].track_id, id);
    }

    #[test]
    fn burst_arrivals_do_not_launch_motion_prediction_away_from_the_subject() {
        let mut tracker = Tracker::default();
        let mut id = 0;
        for (i, timestamp) in [1000, 1001, 1002, 1067].into_iter().enumerate() {
            let x = i as f32 * 10.0;
            let mut rows = vec![det("PERSON", [x, 0., x + 100., 200.])];
            tracker.update(i as u64 + 1, timestamp, (640, 480), &mut rows);
            if i == 0 {
                id = rows[0].track_id;
            }
            assert_eq!(rows[0].track_id, id);
        }
    }

    #[test]
    fn smoothing_keeps_raw_measurements_and_current_joint_confidence() {
        let mut tracker = Tracker::default();
        let mut a = vec![det("PERSON", [10., 10., 110., 300.])];
        a[0].keypoints = vec![PoseKeypoint {
            name: "nose",
            x: 30.,
            y: 30.,
            confidence: 0.9,
        }];
        a[0].display_keypoints = a[0].keypoints.clone();
        tracker.update(1, 1000, (640, 480), &mut a);
        let mut b = vec![det("PERSON", [20., 10., 120., 300.])];
        b[0].keypoints = vec![PoseKeypoint {
            name: "nose",
            x: 40.,
            y: 30.,
            confidence: 0.1,
        }];
        b[0].display_keypoints = b[0].keypoints.clone();
        tracker.update(2, 1067, (640, 480), &mut b);
        assert_eq!(b[0].bbox[0], 20.);
        assert!(b[0].display_bbox[0] > 10. && b[0].display_bbox[0] < 20.);
        assert_eq!(b[0].keypoints[0].x, 40.);
        assert_eq!(b[0].display_keypoints[0].confidence, 0.1);
    }
}
