//! Unified API for spawning video capture pipelines.
//!
//! The crate hides the details of OpenCV- and FFmpeg-backed ingest paths and
//! exposes a consistent channel-based interface to upstream callers.

mod camera;
mod ffmpeg;
mod types;

/// Spawn a threaded V4L / OpenCV capture loop pulling BGR frames.
pub use camera::spawn_camera_reader;
/// Spawn FFmpeg NVDEC capture for direct device inputs.
pub use ffmpeg::{spawn_nvdec_h264_reader, spawn_rtsp_reader, spawn_udp_reader};
/// Types shared between capture implementations.
pub use types::{CaptureError, Frame, FrameFormat};
