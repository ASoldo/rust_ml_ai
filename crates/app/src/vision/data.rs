use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use serde::Serialize;

pub(crate) const FRAME_HISTORY_CAPACITY: usize = 64;

#[derive(Clone)]
pub(crate) struct FramePacket {
    pub(crate) jpeg: Vec<u8>,
    pub(crate) detections: Vec<DetectionSummary>,
    pub(crate) timestamp_ms: i64,
    pub(crate) frame_number: u64,
    pub(crate) fps: f32,
}

#[derive(Clone, Serialize)]
pub(crate) struct DetectionSummary {
    pub(crate) class: String,
    pub(crate) score: f32,
    pub(crate) bbox: [f32; 4],
    pub(crate) track_id: i64,
}

#[derive(Serialize)]
pub(crate) struct DetectionsResponse<'a> {
    pub(crate) timestamp_ms: i64,
    pub(crate) frame_number: u64,
    pub(crate) fps: f32,
    pub(crate) detections: &'a [DetectionSummary],
}

pub(crate) type SharedFrame = Arc<Mutex<Option<FramePacket>>>;
pub(crate) type FrameHistory = Arc<Mutex<VecDeque<FramePacket>>>;
