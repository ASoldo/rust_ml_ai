use anyhow::Error;
use thiserror::Error;

/// Raw RGBA frame captured from a video source.
pub struct Frame {
    pub data: Vec<u8>,
    pub width: i32,
    pub height: i32,
    pub timestamp_ms: i64,
    pub format: FrameFormat,
}

#[derive(Clone, Copy)]
pub enum FrameFormat {
    Bgr8,
}

#[derive(Debug, Error)]
pub enum CaptureError {
    #[error("failed to open video source {uri:?}")]
    Open { uri: String },
    #[error(transparent)]
    Other(#[from] Error),
}
