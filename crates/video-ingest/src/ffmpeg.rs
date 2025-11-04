use std::{
    fmt::Write,
    io::{Read, Write as IoWrite},
    process::{Child, Command, Stdio},
    thread,
};

use anyhow::{Result, anyhow};
use chrono::Utc;
use crossbeam_channel::{Receiver, Sender, bounded};

use crate::{
    camera::parse_device_index,
    types::{CaptureError, Frame, FrameFormat},
};

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
