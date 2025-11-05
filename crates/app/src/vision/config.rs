//! Configuration parsing for the vision pipeline.
//!
//! This module owns translation of CLI arguments into a `VisionConfig` struct
//! which downstream stages use without re-parsing flags.

use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};
use clap::Args;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Ingress transport used to source frames.
pub enum SourceKind {
    /// Local V4L devices or pre-recorded files.
    Device,
    /// Real-time streaming protocol feeds.
    Rtsp,
    /// UDP socket carrying H.264 via RTP.
    Udp,
}

impl SourceKind {
    /// Infer the transport kind from a URI.
    pub(crate) fn from_uri(uri: &str) -> Self {
        if uri.starts_with("rtsp://") || uri.starts_with("rtsps://") {
            SourceKind::Rtsp
        } else if uri.starts_with("udp://") {
            SourceKind::Udp
        } else {
            SourceKind::Device
        }
    }
}

#[derive(Clone, Debug)]
/// Canonical configuration shared by every stage in the pipeline.
pub struct VisionConfig {
    /// Camera URI or device identifier.
    pub camera_uri: String,
    /// Source transport used to acquire frames.
    pub source_kind: SourceKind,
    /// TorchScript model path used by the detector workers.
    pub model_path: PathBuf,
    /// Capture width streamed by the ingest component.
    pub width: i32,
    /// Capture height streamed by the ingest component.
    pub height: i32,
    /// Emit verbose logging (frame drops, detection details).
    pub verbose: bool,
    /// Force CPU inference and annotation.
    pub use_cpu: bool,
    /// Attempt NVDEC capture or decoding when supported.
    pub use_nvdec: bool,
    /// Logical detector width, may differ from capture width when resizing.
    pub detector_width: i32,
    /// Logical detector height, may differ from capture height when resizing.
    pub detector_height: i32,
    /// JPEG quality used by CPU and GPU encoders.
    pub jpeg_quality: i32,
    /// Number of parallel detector workers.
    pub processor_workers: usize,
    /// Batch size processed per worker iteration.
    pub batch_size: usize,
    /// Telemetry and instrumentation options.
    pub telemetry: TelemetryOptions,
}

#[derive(Clone, Debug, Default)]
/// Optional telemetry knobs for tracing and runtime inspection.
pub struct TelemetryOptions {
    /// Write a Chrome trace JSON file capturing pipeline spans.
    pub chrome_trace_path: Option<PathBuf>,
    /// Enable the Tokio console subscriber for live task/queue inspection.
    pub enable_tokio_console: bool,
}

/// CLI arguments accepted by the `vision` subcommand.
#[derive(Debug, Args)]
pub struct VisionCliArgs {
    /// Camera URI or device identifier.
    #[arg(value_name = "CAMERA_URI")]
    pub camera_uri: Option<String>,
    /// TorchScript model path.
    #[arg(value_name = "MODEL_PATH")]
    pub model_path: Option<PathBuf>,
    /// Capture width in pixels.
    #[arg(value_name = "WIDTH")]
    pub width: Option<i32>,
    /// Capture height in pixels.
    #[arg(value_name = "HEIGHT")]
    pub height: Option<i32>,

    /// Camera URI flag (overrides positional).
    #[arg(long = "source", value_name = "URI")]
    pub source_uri: Option<String>,
    /// TorchScript model path flag (overrides positional).
    #[arg(long = "model", value_name = "PATH")]
    pub model_path_flag: Option<PathBuf>,
    /// Capture width flag (overrides positional).
    #[arg(long = "width", value_name = "PX")]
    pub width_flag: Option<i32>,
    /// Capture height flag (overrides positional).
    #[arg(long = "height", value_name = "PX")]
    pub height_flag: Option<i32>,
    /// Enable verbose logging (frame drops, detections).
    #[arg(long = "verbose", action = clap::ArgAction::SetTrue)]
    pub verbose: bool,
    /// Force CPU inference and annotation.
    #[arg(long = "cpu", action = clap::ArgAction::SetTrue, conflicts_with = "use_nvdec")]
    pub use_cpu: bool,
    /// Attempt NVDEC capture when supported.
    #[arg(long = "nvdec", action = clap::ArgAction::SetTrue)]
    pub use_nvdec: bool,
    /// Detector input width in pixels.
    #[arg(long = "detector-width", value_name = "PX")]
    pub detector_width: Option<i32>,
    /// Detector input height in pixels.
    #[arg(long = "detector-height", value_name = "PX")]
    pub detector_height: Option<i32>,
    /// JPEG quality used by the encoder (1-100).
    #[arg(long = "jpeg-quality", value_name = "QUALITY")]
    pub jpeg_quality: Option<i32>,
    /// Number of detector workers.
    #[arg(long = "processors", value_name = "N")]
    pub processors: Option<usize>,
    /// Detector batch size.
    #[arg(long = "batch-size", value_name = "N")]
    pub batch_size: Option<usize>,
    /// Emit Chrome trace JSON for post-mortem analysis.
    #[arg(long = "chrome-trace", value_name = "PATH")]
    pub chrome_trace: Option<PathBuf>,
    /// Enable the Tokio console instrumentation server.
    #[arg(long = "tokio-console", action = clap::ArgAction::SetTrue)]
    pub tokio_console: bool,
}

impl TryFrom<VisionCliArgs> for VisionConfig {
    type Error = anyhow::Error;

    fn try_from(args: VisionCliArgs) -> Result<Self> {
        let camera_uri = args.source_uri.or(args.camera_uri).ok_or_else(|| {
            anyhow!("Missing source. Provide --source <uri> or positional <camera-uri>.")
        })?;
        let model_path = args.model_path_flag.or(args.model_path).ok_or_else(|| {
            anyhow!("Missing model path. Provide --model <path> or positional <model-path>.")
        })?;
        let width = args
            .width_flag
            .or(args.width)
            .ok_or_else(|| anyhow!("Missing width. Provide --width <px> or positional <width>."))?;
        let height = args.height_flag.or(args.height).ok_or_else(|| {
            anyhow!("Missing height. Provide --height <px> or positional <height>.")
        })?;

        if width <= 0 || height <= 0 {
            bail!("Capture width and height must be positive integers");
        }

        let (detector_width, detector_height) = match (args.detector_width, args.detector_height) {
            (Some(w), Some(h)) if w > 0 && h > 0 => (w, h),
            (Some(w), None) if w > 0 => (w, height),
            (None, Some(h)) if h > 0 => (width, h),
            (None, None) => Self::default_detector_size(width, height),
            (Some(_), Some(_)) => bail!("Detector dimensions must be positive integers"),
            (Some(_), None) | (None, Some(_)) => {
                bail!("Detector dimensions must be positive integers")
            }
        };

        if args.use_cpu && args.use_nvdec {
            bail!("--cpu and --nvdec are mutually exclusive");
        }

        let jpeg_quality = args.jpeg_quality.unwrap_or(85);
        if !(1..=100).contains(&jpeg_quality) {
            bail!("--jpeg-quality must be an integer between 1 and 100");
        }

        let processor_workers = args.processors.unwrap_or(1);
        if processor_workers == 0 {
            bail!("--processors must be at least 1");
        }

        let batch_size = args.batch_size.unwrap_or(1);
        if batch_size == 0 {
            bail!("--batch-size must be at least 1");
        }

        let telemetry = TelemetryOptions {
            chrome_trace_path: args.chrome_trace,
            enable_tokio_console: args.tokio_console,
        };

        let source_kind = SourceKind::from_uri(&camera_uri);

        Ok(Self {
            camera_uri,
            source_kind,
            model_path,
            width,
            height,
            verbose: args.verbose,
            use_cpu: args.use_cpu,
            use_nvdec: args.use_nvdec,
            detector_width,
            detector_height,
            jpeg_quality,
            processor_workers,
            batch_size,
            telemetry,
        })
    }
}

impl VisionConfig {
    /// Derive a square detector size rounded up to 32-pixel alignment.
    fn default_detector_size(width: i32, height: i32) -> (i32, i32) {
        let max_dim = width.max(height).max(32);
        let aligned = ((max_dim + 31) / 32) * 32;
        (aligned, aligned)
    }
}
