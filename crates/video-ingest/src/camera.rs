//! OpenCV-backed camera capture pipeline.

use std::thread;

use anyhow::Result;
use chrono::Utc;
use crossbeam_channel::{Receiver, Sender, bounded};
use opencv::{
    core::{self, MatTraitConstManual},
    prelude::*,
    videoio::{self, VideoCapture, VideoCaptureTrait},
};

use crate::types::{CaptureError, Frame, FrameFormat};

/// Spawns a background thread that continually captures frames from the provided `uri`.
///
/// Frames are resized to `target_size` (width, height) and converted to RGBA before being
/// forwarded over the returned [`Receiver`]. The buffer is intentionally small to backpressure
/// the capture loop when downstream consumers fall behind.
pub fn spawn_camera_reader(
    uri: &str,
    target_size: (i32, i32),
) -> Result<Receiver<Result<Frame, CaptureError>>> {
    let (tx, rx) = bounded(2);
    let uri = uri.to_string();

    thread::spawn(move || {
        if let Err(err) = capture_loop(&uri, target_size, tx.clone()) {
            let _ = tx.send(Err(err));
        }
    });

    Ok(rx)
}

/// Main capture loop executed on the background thread.
fn capture_loop(
    uri: &str,
    target_size: (i32, i32),
    tx: Sender<Result<Frame, CaptureError>>,
) -> Result<(), CaptureError> {
    let mut cap = open_video_capture(uri)?;

    configure_camera(&mut cap, target_size, 60.0);

    let mut frame = Mat::default();
    let mut scratch = Mat::default();
    let (target_w, target_h) = target_size;

    loop {
        cap.read(&mut frame)
            .map_err(|e| CaptureError::Other(e.into()))?;

        if frame
            .size()
            .map_err(|e| CaptureError::Other(e.into()))?
            .width
            <= 0
        {
            continue;
        }

        let size = frame.size().map_err(|e| CaptureError::Other(e.into()))?;

        let working = if size.width != target_w || size.height != target_h {
            opencv::imgproc::resize(
                &frame,
                &mut scratch,
                core::Size {
                    width: target_w,
                    height: target_h,
                },
                0.0,
                0.0,
                opencv::imgproc::INTER_LINEAR,
            )
            .map_err(|e| CaptureError::Other(e.into()))?;
            &scratch
        } else {
            &frame
        };

        let data = working
            .data_bytes()
            .map_err(|e| CaptureError::Other(e.into()))?
            .to_vec();

        let timestamp_ms = Utc::now().timestamp_millis();

        if tx
            .send(Ok(Frame {
                data,
                width: target_w,
                height: target_h,
                timestamp_ms,
                format: FrameFormat::Bgr8,
            }))
            .is_err()
        {
            break;
        }
    }

    Ok(())
}

/// Parse a `/dev/videoX` style URI and return the zero-based index if present.
pub(crate) fn parse_device_index(uri: &str) -> Option<i32> {
    if let Ok(index) = uri.parse::<i32>() {
        return Some(index);
    }
    if let Some(stripped) = uri.strip_prefix("/dev/video") {
        if stripped.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(index) = stripped.parse::<i32>() {
                return Some(index);
            }
        }
    }
    None
}

/// Attempt to open a camera input either by index or URI.
fn open_video_capture(uri: &str) -> Result<VideoCapture, CaptureError> {
    if let Some(index) = parse_device_index(uri) {
        for backend in [videoio::CAP_V4L, videoio::CAP_ANY] {
            match VideoCapture::new(index, backend) {
                Ok(cap) => {
                    if cap.is_opened().map_err(|e| CaptureError::Other(e.into()))? {
                        return Ok(cap);
                    }
                }
                Err(err) => {
                    eprintln!(
                        "video-ingest: failed to open device #{index} with backend {backend}: {err}"
                    );
                }
            }
        }
    }

    for backend in [videoio::CAP_V4L, videoio::CAP_ANY] {
        match VideoCapture::from_file(uri, backend) {
            Ok(cap) => {
                if cap.is_opened().map_err(|e| CaptureError::Other(e.into()))? {
                    return Ok(cap);
                }
            }
            Err(err) => {
                eprintln!("video-ingest: failed to open {uri} with backend {backend}: {err}");
            }
        }
    }

    Err(CaptureError::Open {
        uri: uri.to_string(),
    })
}

/// Apply common capture settings (resolution, fps, preferred pixel format).
fn configure_camera(cap: &mut VideoCapture, target_size: (i32, i32), fps: f64) {
    let mut fourcc_set = false;
    if let Ok(mjpg) = videoio::VideoWriter::fourcc('M', 'J', 'P', 'G') {
        if matches!(cap.set(videoio::CAP_PROP_FOURCC, mjpg as f64), Ok(true)) {
            fourcc_set = true;
        }
    }
    if !fourcc_set {
        if let Ok(yuyv) = videoio::VideoWriter::fourcc('Y', 'U', 'Y', 'V') {
            let _ = cap.set(videoio::CAP_PROP_FOURCC, yuyv as f64);
        }
    }
    let _ = cap.set(videoio::CAP_PROP_FRAME_WIDTH, target_size.0 as f64);
    let _ = cap.set(videoio::CAP_PROP_FRAME_HEIGHT, target_size.1 as f64);
    let _ = cap.set(videoio::CAP_PROP_FPS, fps);
}
