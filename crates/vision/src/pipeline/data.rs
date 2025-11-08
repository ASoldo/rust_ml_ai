//! Shared data structures passed between pipeline stages and exposed via the
//! HTTP API.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use serde::Serialize;

/// Number of annotated frames retained for history queries.
pub(crate) const FRAME_HISTORY_CAPACITY: usize = 64;

#[derive(Clone)]
/// Serialized packet produced by the encoder stage.
pub(crate) struct FramePacket {
    /// MJPEG payload ready for HTTP streaming.
    pub(crate) jpeg: Vec<u8>,
    /// Detection summaries used by the HUD and API payloads.
    pub(crate) detections: Vec<DetectionSummary>,
    /// Capture timestamp in milliseconds since UNIX epoch.
    pub(crate) timestamp_ms: i64,
    /// Monotonic frame identifier.
    pub(crate) frame_number: u64,
    /// Smoothed instantaneous FPS for diagnostics.
    pub(crate) fps: f32,
}

#[derive(Clone, Serialize)]
/// High-level detection metadata returned to clients.
pub(crate) struct DetectionSummary {
    /// Human readable class label.
    pub(crate) class: String,
    /// Confidence score from the detector.
    pub(crate) score: f32,
    /// Bounding box in detector-space coordinates `[left, top, right, bottom]`.
    pub(crate) bbox: [f32; 4],
    /// Assigned tracking identifier (simple monotonic scheme).
    pub(crate) track_id: i64,
}

#[derive(Serialize)]
/// JSON payload streamed over `/detections` and SSE endpoints.
pub(crate) struct DetectionsResponse<'a> {
    /// Timestamps and metadata corresponding to the frame the detections belong to.
    pub(crate) timestamp_ms: i64,
    pub(crate) frame_number: u64,
    pub(crate) fps: f32,
    pub(crate) detections: &'a [DetectionSummary],
}

/// Shared pointer to the most recent encoded frame.
pub(crate) type SharedFrame = Arc<Mutex<Option<FramePacket>>>;
/// FIFO of encoded frames used to satisfy history queries.
pub(crate) type FrameHistory = Arc<Mutex<VecDeque<FramePacket>>>;
