use std::{
    io::Read,
    process::{Child, Command, Stdio},
    thread,
};

use anyhow::{Result, anyhow};
use chrono::Utc;
use crossbeam_channel::{Receiver, Sender, bounded};
use opencv::{
    core::{self, MatTraitConstManual},
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

/// Spawns an FFmpeg process that uses NVDEC (via CUDA) to decode an H.264 stream and
/// yields BGR8 frames via a background thread.
pub fn spawn_nvdec_h264_reader(
    uri: &str,
    target_size: (i32, i32),
) -> Result<Receiver<Result<Frame, CaptureError>>> {
    let (tx, rx) = bounded(2);
    let uri = uri.to_string();
    let scale_arg = format!("scale={}:{}", target_size.0, target_size.1);

    let (is_v4l, ffmpeg_uri) = if let Some(index) = parse_device_index(&uri) {
        (true, format!("/dev/video{index}"))
    } else if uri.starts_with("/dev/video") {
        (true, uri.clone())
    } else {
        (false, uri.clone())
    };

    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-hwaccel")
        .arg("cuda")
        .arg("-hwaccel_output_format")
        .arg("cuda")
        .arg("-c:v")
        .arg("h264_cuvid")
        .stderr(Stdio::inherit());

    if is_v4l {
        cmd.arg("-f")
            .arg("video4linux2")
            .arg("-input_format")
            .arg("h264");
    }

    cmd.arg("-i")
        .arg(&ffmpeg_uri)
        .arg("-vf")
        .arg(&scale_arg)
        .arg("-pix_fmt")
        .arg("bgr24")
        .arg("-f")
        .arg("rawvideo")
        .arg("-")
        .stdout(Stdio::piped());

    let mut child = cmd.spawn().map_err(|err| CaptureError::Other(err.into()))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| CaptureError::Other(anyhow!("failed to capture ffmpeg stdout")))?;

    thread::spawn(move || {
        if let Err(err) = nvdec_loop(stdout, child, target_size, tx.clone()) {
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

fn nvdec_loop(
    mut stdout: impl Read,
    mut child: Child,
    target_size: (i32, i32),
    tx: Sender<Result<Frame, CaptureError>>,
) -> Result<(), CaptureError> {
    let frame_bytes = (target_size.0 as usize) * (target_size.1 as usize) * 3;
    let mut buffer = vec![0u8; frame_bytes];

    loop {
        match stdout.read_exact(&mut buffer) {
            Ok(()) => {
                let timestamp_ms = Utc::now().timestamp_millis();
                if tx
                    .send(Ok(Frame {
                        data: buffer.clone(),
                        width: target_size.0,
                        height: target_size.1,
                        timestamp_ms,
                        format: FrameFormat::Bgr8,
                    }))
                    .is_err()
                {
                    break;
                }
            }
            Err(err) => {
                if tx.send(Err(CaptureError::Other(err.into()))).is_err() {
                    break;
                }
                break;
            }
        }
    }

    let _ = child.kill();
    Ok(())
}
