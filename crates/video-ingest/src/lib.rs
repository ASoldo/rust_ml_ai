use std::thread;

use anyhow::Result;
use chrono::Utc;
use crossbeam_channel::{Receiver, Sender, bounded};
use opencv::{
    core::{self, MatTraitConstManual},
    imgproc,
    prelude::*,
    videoio::{self, VideoCapture, VideoCaptureTrait},
};
use thiserror::Error;

/// Raw RGBA frame captured from a video source.
pub struct Frame {
    pub data: Vec<u8>,
    pub width: i32,
    pub height: i32,
    pub timestamp_ms: i64,
}

#[derive(Debug, Error)]
pub enum CaptureError {
    #[error("failed to open video source {uri:?}")]
    Open { uri: String },
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

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

fn capture_loop(
    uri: &str,
    target_size: (i32, i32),
    tx: Sender<Result<Frame, CaptureError>>,
) -> Result<(), CaptureError> {
    let mut cap = open_video_capture(uri)?;

    configure_camera(&mut cap, target_size, 60.0);

    // Best-effort: turn off autofocus/auto-exposure so the lens stops hunting.
    // let _ = cap.set(videoio::CAP_PROP_AUTOFOCUS, 0.0);
    // let _ = cap.set(videoio::CAP_PROP_FOCUS, 0.0);
    // For UVC cameras, 1.0 typically selects manual exposure mode.
    // let _ = cap.set(videoio::CAP_PROP_AUTO_EXPOSURE, 1.0);

    let mut frame = Mat::default();
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

        let mut working = Mat::default();
        let size = frame.size().map_err(|e| CaptureError::Other(e.into()))?;

        if size.width != target_w || size.height != target_h {
            imgproc::resize(
                &frame,
                &mut working,
                core::Size {
                    width: target_w,
                    height: target_h,
                },
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )
            .map_err(|e| CaptureError::Other(e.into()))?;
        } else {
            working = frame.clone();
        }

        let mut rgba = Mat::default();
        let hint = core::get_default_algorithm_hint().map_err(|e| CaptureError::Other(e.into()))?;

        imgproc::cvt_color(&working, &mut rgba, imgproc::COLOR_BGR2RGBA, 0, hint)
            .map_err(|e| CaptureError::Other(e.into()))?;

        let data = rgba
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
            }))
            .is_err()
        {
            break;
        }
    }

    Ok(())
}

fn parse_device_index(uri: &str) -> Option<i32> {
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

fn configure_camera(cap: &mut VideoCapture, target_size: (i32, i32), fps: f64) {
    if let Ok(fourcc) = videoio::VideoWriter::fourcc('M', 'J', 'P', 'G') {
        let _ = cap.set(videoio::CAP_PROP_FOURCC, fourcc as f64);
    }
    let _ = cap.set(videoio::CAP_PROP_FRAME_WIDTH, target_size.0 as f64);
    let _ = cap.set(videoio::CAP_PROP_FRAME_HEIGHT, target_size.1 as f64);
    let _ = cap.set(videoio::CAP_PROP_FPS, fps);
}
