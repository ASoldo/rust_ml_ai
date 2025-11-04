use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceKind {
    Device,
    Rtsp,
    Udp,
}

impl SourceKind {
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
pub struct VisionConfig {
    pub camera_uri: String,
    pub source_kind: SourceKind,
    pub model_path: PathBuf,
    pub width: i32,
    pub height: i32,
    pub verbose: bool,
    pub use_cpu: bool,
    pub use_nvdec: bool,
    pub detector_width: i32,
    pub detector_height: i32,
    pub jpeg_quality: i32,
    pub processor_workers: usize,
    pub batch_size: usize,
}

const VISION_USAGE: &str = "Usage: cargo run -p vision --features with-tch -- \
vision [--source <uri>] [--model <path>] [--width <px>] [--height <px>] \
[--cpu] [--nvdec] [--verbose] [--detector-width <px>] [--detector-height <px>] \
[--jpeg-quality <1-100>] [--processors <n>] [--batch-size <n>]\n\nPositional form is \
also supported: vision <uri> <model-path> <width> <height> [...flags...]";

impl VisionConfig {
    pub fn from_args(args: &[String]) -> Result<Self> {
        if args.len() < 3 {
            bail!(VISION_USAGE);
        }

        let mut camera_uri: Option<String> = None;
        let mut model_path: Option<PathBuf> = None;
        let mut width: Option<i32> = None;
        let mut height: Option<i32> = None;
        let mut verbose = false;
        let mut use_cpu = false;
        let mut use_nvdec = false;
        let mut detector_width: Option<i32> = None;
        let mut detector_height: Option<i32> = None;
        let mut jpeg_quality: Option<i32> = None;
        let mut processor_workers: Option<usize> = None;
        let mut batch_size: Option<usize> = None;
        let mut positional: Vec<String> = Vec::new();

        let mut idx = 2;
        while idx < args.len() {
            match args[idx].as_str() {
                "--source" => {
                    idx += 1;
                    let value = args
                        .get(idx)
                        .ok_or_else(|| anyhow!("--source requires a value"))?
                        .clone();
                    camera_uri = Some(value);
                    idx += 1;
                }
                "--model" => {
                    idx += 1;
                    let value = args
                        .get(idx)
                        .ok_or_else(|| anyhow!("--model requires a value"))?
                        .clone();
                    model_path = Some(PathBuf::from(value));
                    idx += 1;
                }
                "--width" => {
                    idx += 1;
                    let value = args
                        .get(idx)
                        .ok_or_else(|| anyhow!("--width requires a value"))?
                        .parse::<i32>()
                        .with_context(|| "--width must be an integer".to_string())?;
                    width = Some(value);
                    idx += 1;
                }
                "--height" => {
                    idx += 1;
                    let value = args
                        .get(idx)
                        .ok_or_else(|| anyhow!("--height requires a value"))?
                        .parse::<i32>()
                        .with_context(|| "--height must be an integer".to_string())?;
                    height = Some(value);
                    idx += 1;
                }
                "--verbose" => {
                    verbose = true;
                    idx += 1;
                }
                "--cpu" => {
                    use_cpu = true;
                    idx += 1;
                }
                "--nvdec" => {
                    use_nvdec = true;
                    idx += 1;
                }
                "--detector-width" => {
                    idx += 1;
                    let value = args
                        .get(idx)
                        .ok_or_else(|| anyhow!("--detector-width requires a value"))?
                        .parse::<i32>()
                        .with_context(|| {
                            "--detector-width must be a positive integer".to_string()
                        })?;
                    if value <= 0 {
                        bail!("--detector-width must be a positive integer");
                    }
                    detector_width = Some(value);
                    idx += 1;
                }
                "--detector-height" => {
                    idx += 1;
                    let value = args
                        .get(idx)
                        .ok_or_else(|| anyhow!("--detector-height requires a value"))?
                        .parse::<i32>()
                        .with_context(|| {
                            "--detector-height must be a positive integer".to_string()
                        })?;
                    if value <= 0 {
                        bail!("--detector-height must be a positive integer");
                    }
                    detector_height = Some(value);
                    idx += 1;
                }
                "--processors" => {
                    idx += 1;
                    let value = args
                        .get(idx)
                        .ok_or_else(|| anyhow!("--processors requires a value"))?
                        .parse::<usize>()
                        .with_context(|| "--processors must be a positive integer".to_string())?;
                    if value == 0 {
                        bail!("--processors must be at least 1");
                    }
                    processor_workers = Some(value);
                    idx += 1;
                }
                "--batch-size" => {
                    idx += 1;
                    let value = args
                        .get(idx)
                        .ok_or_else(|| anyhow!("--batch-size requires a value"))?
                        .parse::<usize>()
                        .with_context(|| "--batch-size must be a positive integer".to_string())?;
                    if value == 0 {
                        bail!("--batch-size must be at least 1");
                    }
                    batch_size = Some(value);
                    idx += 1;
                }
                "--jpeg-quality" => {
                    idx += 1;
                    let value = args
                        .get(idx)
                        .ok_or_else(|| anyhow!("--jpeg-quality requires a value"))?
                        .parse::<i32>()
                        .with_context(|| {
                            "--jpeg-quality must be an integer between 1 and 100".to_string()
                        })?;
                    if !(1..=100).contains(&value) {
                        bail!("--jpeg-quality must be an integer between 1 and 100");
                    }
                    jpeg_quality = Some(value);
                    idx += 1;
                }
                arg if arg.starts_with('-') => {
                    bail!("Unrecognised flag: {arg}");
                }
                other => {
                    positional.push(other.to_string());
                    idx += 1;
                }
            }
        }

        let mut positional = positional.into_iter();
        if camera_uri.is_none() {
            camera_uri = positional.next();
        }
        if model_path.is_none() {
            if let Some(path) = positional.next() {
                model_path = Some(PathBuf::from(path));
            }
        }
        if width.is_none() {
            if let Some(value) = positional.next() {
                width = Some(
                    value
                        .parse::<i32>()
                        .with_context(|| "width must be an integer".to_string())?,
                );
            }
        }
        if height.is_none() {
            if let Some(value) = positional.next() {
                height = Some(
                    value
                        .parse::<i32>()
                        .with_context(|| "height must be an integer".to_string())?,
                );
            }
        }

        let camera_uri = camera_uri.ok_or_else(|| {
            anyhow!("Missing source. Provide --source <uri> or positional <camera-uri>.")
        })?;
        let model_path = model_path.ok_or_else(|| {
            anyhow!("Missing model path. Provide --model <path> or positional <model-path>.")
        })?;
        let width = width
            .ok_or_else(|| anyhow!("Missing width. Provide --width <px> or positional <width>."))?;
        let height = height.ok_or_else(|| {
            anyhow!("Missing height. Provide --height <px> or positional <height>.")
        })?;

        if use_cpu && use_nvdec {
            bail!("--cpu and --nvdec are mutually exclusive");
        }

        let (detector_width, detector_height) = match (detector_width, detector_height) {
            (Some(w), Some(h)) => (w, h),
            (Some(w), None) => (w, height),
            (None, Some(h)) => (width, h),
            (None, None) => Self::default_detector_size(width, height),
        };
        let jpeg_quality = jpeg_quality.unwrap_or(85);
        let source_kind = SourceKind::from_uri(&camera_uri);
        let processor_workers = processor_workers.unwrap_or(1);
        let batch_size = batch_size.unwrap_or(1);

        Ok(Self {
            camera_uri,
            source_kind,
            model_path,
            width,
            height,
            verbose,
            use_cpu,
            use_nvdec,
            detector_width,
            detector_height,
            jpeg_quality,
            processor_workers,
            batch_size,
        })
    }

    fn default_detector_size(width: i32, height: i32) -> (i32, i32) {
        let max_dim = width.max(height).max(32);
        let aligned = ((max_dim + 31) / 32) * 32;
        (aligned, aligned)
    }
}
