use std::{
    io::{Read, Write},
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
        .arg("h264_cuvid");

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
        .arg("-");

    spawn_ffmpeg_reader(cmd, target_size, 3, None)
}

pub fn spawn_rtsp_reader(
    uri: &str,
    target_size: (i32, i32),
    use_nvdec: bool,
) -> Result<Receiver<Result<Frame, CaptureError>>> {
    let scale_arg = format!("scale={}:{}", target_size.0, target_size.1);
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-rtsp_transport")
        .arg("tcp")
        .arg("-fflags")
        .arg("nobuffer")
        .arg("-flags")
        .arg("low_delay")
        .arg("-max_delay")
        .arg("0");

    if use_nvdec {
        cmd.arg("-hwaccel")
            .arg("cuda")
            .arg("-hwaccel_output_format")
            .arg("cuda")
            .arg("-c:v")
            .arg("h264_cuvid");
    }

    cmd.arg("-i")
        .arg(uri)
        .arg("-vf")
        .arg(&scale_arg)
        .arg("-pix_fmt")
        .arg("bgr24")
        .arg("-f")
        .arg("rawvideo")
        .arg("-");

    spawn_ffmpeg_reader(cmd, target_size, 4, None)
}

pub fn spawn_udp_reader(
    uri: &str,
    target_size: (i32, i32),
    use_nvdec: bool,
) -> Result<Receiver<Result<Frame, CaptureError>>> {
    let scale_arg = format!("scale={}:{}", target_size.0, target_size.1);
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-protocol_whitelist")
        .arg("file,udp,rtp,fd,pipe")
        .arg("-fflags")
        .arg("+genpts+discardcorrupt")
        .arg("-flags")
        .arg("low_delay");

    if use_nvdec {
        cmd.arg("-hwaccel")
            .arg("cuda")
            .arg("-hwaccel_output_format")
            .arg("cuda")
            .arg("-c:v")
            .arg("h264_cuvid");
    }

    cmd.arg("-f").arg("sdp").arg("-i").arg("-");

    cmd.arg("-an")
        .arg("-vf")
        .arg(&scale_arg)
        .arg("-pix_fmt")
        .arg("bgr24")
        .arg("-f")
        .arg("rawvideo")
        .arg("-");

    let sdp = build_udp_sdp(uri).map_err(|err| anyhow!("{err}"))?;
    spawn_ffmpeg_reader(cmd, target_size, 4, Some(sdp))
}

fn build_udp_sdp(uri: &str) -> Result<String, CaptureError> {
    let without_scheme = uri.strip_prefix("udp://").unwrap_or(uri);
    let mut parts = without_scheme.splitn(2, '?');
    let endpoint = parts.next().unwrap_or("");
    let query = parts.next();
    let mut host = "0.0.0.0";
    let mut port_str = endpoint;
    let mut sprop: Option<String> = None;
    let mut payload: Option<String> = None;

    if endpoint.is_empty() {
        return Err(CaptureError::Other(anyhow!(
            "udp source must include host:port, e.g. udp://127.0.0.1:5000"
        )));
    }

    if let Some((h, p)) = endpoint.rsplit_once(':') {
        if !h.is_empty() {
            host = h;
        }
        port_str = p;
    }

    let port: u16 = port_str
        .parse()
        .map_err(|_| CaptureError::Other(anyhow!("invalid UDP port in source URI")))?;

    if let Some(query) = query {
        for pair in query.split('&') {
            let mut kv = pair.splitn(2, '=');
            let key = kv.next().unwrap_or("");
            let value = kv.next().unwrap_or("");
            match key {
                "sprop" | "sprop-parameter-sets" => {
                    sprop = Some(value.to_string());
                }
                "payload" | "pt" => {
                    payload = Some(value.to_string());
                }
                _ => {}
            }
        }
    }

    let payload = payload.unwrap_or_else(|| "96".to_string());

    let mut sdp = String::new();
    use std::fmt::Write as _;
    writeln!(&mut sdp, "v=0").ok();
    writeln!(&mut sdp, "o=- 0 0 IN IP4 {host}").ok();
    writeln!(&mut sdp, "s=vision-udp").ok();
    writeln!(&mut sdp, "c=IN IP4 {host}").ok();
    writeln!(&mut sdp, "t=0 0").ok();
    writeln!(&mut sdp, "m=video {port} RTP/AVP {payload}").ok();
    writeln!(&mut sdp, "a=rtpmap:{payload} H264/90000").ok();
    if let Some(sprop) = sprop {
        writeln!(
            &mut sdp,
            "a=fmtp:{payload} packetization-mode=1; sprop-parameter-sets={}",
            sprop
        )
        .ok();
    } else {
        writeln!(&mut sdp, "a=fmtp:{payload} packetization-mode=1").ok();
    }

    Ok(sdp)
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
    // Prefer MJPG since many UVC devices expose higher frame rates for the compressed stream.
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

fn spawn_ffmpeg_reader(
    mut cmd: Command,
    target_size: (i32, i32),
    queue_size: usize,
    stdin_payload: Option<String>,
) -> Result<Receiver<Result<Frame, CaptureError>>> {
    let (tx, rx) = bounded(queue_size);
    if stdin_payload.is_some() {
        cmd.stdin(Stdio::piped());
    } else {
        cmd.stdin(Stdio::null());
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::inherit());

    let mut child = cmd.spawn().map_err(|err| CaptureError::Other(err.into()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| CaptureError::Other(anyhow!("failed to capture ffmpeg stdout")))?;

    if let Some(payload) = stdin_payload {
        if let Some(mut stdin) = child.stdin.take() {
            if let Err(err) = stdin.write_all(payload.as_bytes()) {
                let _ = child.kill();
                return Err(CaptureError::Other(err.into()).into());
            }
        }
    }

    thread::spawn(move || {
        let tx_clone = tx.clone();
        match ffmpeg_loop(stdout, child, target_size, tx_clone) {
            Ok(()) => {}
            Err(err) => {
                let _ = tx.send(Err(err));
            }
        }
    });

    Ok(rx)
}

fn ffmpeg_loop(
    mut stdout: impl Read,
    mut child: Child,
    target_size: (i32, i32),
    tx: Sender<Result<Frame, CaptureError>>,
) -> Result<(), CaptureError> {
    let frame_bytes = (target_size.0 as usize) * (target_size.1 as usize) * 3;
    let mut buffer = vec![0u8; frame_bytes];
    let mut result = Ok(());

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
                result = Err(CaptureError::Other(err.into()));
                break;
            }
        }
    }

    let _ = child.kill();
    result
}
