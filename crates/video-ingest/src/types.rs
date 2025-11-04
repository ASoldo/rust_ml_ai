//! Shared data types exposed by the video ingest layer.

use anyhow::Error;
use thiserror::Error;

/// Raw RGBA frame captured from a video source.
pub struct Frame {
    /// Frame pixel buffer in the layout declared by [`FrameFormat`].
    pub data: Vec<u8>,
    /// Frame width in pixels.
    pub width: i32,
    /// Frame height in pixels.
    pub height: i32,
    /// Capture timestamp in milliseconds.
    pub timestamp_ms: i64,
    /// Format descriptor explaining how to interpret [`Frame::data`].
    pub format: FrameFormat,
}

#[derive(Clone, Copy)]
/// Supported pixel formats emitted by the capture layer.
pub enum FrameFormat {
    /// Packed BGR (24-bit) used by OpenCV and FFmpeg readers.
    Bgr8,
}

#[derive(Debug, Error)]
/// Errors that can arise while configuring or driving capture pipelines.
pub enum CaptureError {
    #[error("failed to open video source {uri:?}")]
    Open { uri: String },
    #[error(transparent)]
    Other(#[from] Error),
}
