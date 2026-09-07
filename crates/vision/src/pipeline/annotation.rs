//! One sparse BGR overlay renderer shared by CPU and GPU JPEG encoding.
use crate::pipeline::{
    data::{DetectionSummary, FramePacket},
    encoding::GpuEncodeJob,
};
use anyhow::{Result, anyhow};
use gpu_kernels::VisionRuntime;
use image::{ImageBuffer, Rgb, codecs::jpeg::JpegEncoder};
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};
use video_ingest::Frame;

fn annotated_pixels(
    frame: &Frame,
    frame_number: u64,
    fps: f32,
    summaries: &[DetectionSummary],
) -> Result<Vec<u8>> {
    let expected = (frame.width as usize)
        .checked_mul(frame.height as usize)
        .and_then(|n| n.checked_mul(3));
    if frame.width <= 0 || frame.height <= 0 || expected != Some(frame.data.len()) {
        return Err(anyhow!("invalid packed BGR capture frame"));
    }
    let mut pixels = frame.data.clone();
    super::overlay::draw(
        &mut pixels,
        frame.width,
        frame.height,
        frame_number,
        fps,
        frame.timestamp_ms,
        summaries,
    );
    Ok(pixels)
}

pub(crate) fn annotate_frame_cpu(
    frame: &Frame,
    frame_number: u64,
    fps: f32,
    mut summaries: Vec<DetectionSummary>,
    jpeg_quality: i32,
) -> Result<FramePacket> {
    let start = Instant::now();
    let mut pixels = annotated_pixels(frame, frame_number, fps, &summaries)?;
    for p in pixels.chunks_exact_mut(3) {
        p.swap(0, 2);
    }
    let rgb =
        ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(frame.width as u32, frame.height as u32, pixels)
            .ok_or_else(|| anyhow!("invalid RGB annotation frame"))?;
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, jpeg_quality.clamp(1, 100) as u8)
        .encode_image(&rgb)?;
    metrics::histogram!("vision_annotation_seconds","path"=>"cpu")
        .record(start.elapsed().as_secs_f64());
    for summary in &mut summaries {
        summary.silhouette = None;
    }
    Ok(FramePacket {
        jpeg,
        detections: summaries,
        timestamp_ms: frame.timestamp_ms,
        frame_number,
        fps,
    })
}

pub(crate) fn annotate_frame_gpu(
    runtime: &Arc<Mutex<VisionRuntime>>,
    frame: &Frame,
    frame_number: u64,
    fps: f32,
    mut summaries: Vec<DetectionSummary>,
    jpeg_quality: i32,
) -> Result<GpuEncodeJob> {
    let start = Instant::now();
    // The capture frame is already owned CPU BGR. Sparse labels/lines need no
    // CUDA upload/download round trip; the queued job still uses GPU NVJPEG.
    let annotated_bgr = annotated_pixels(frame, frame_number, fps, &summaries)?;
    metrics::histogram!("vision_annotation_seconds","path"=>"gpu")
        .record(start.elapsed().as_secs_f64());
    for summary in &mut summaries {
        summary.silhouette = None;
    }
    Ok(GpuEncodeJob {
        runtime: runtime.clone(),
        annotated_bgr,
        width: frame.width,
        height: frame.height,
        summaries,
        timestamp_ms: frame.timestamp_ms,
        frame_number,
        fps,
        jpeg_quality,
    })
}

#[cfg(test)]
mod gpu_regression_tests {
    use super::*;
    use crate::pipeline::encoding::encode_gpu_frame;
    use video_ingest::FrameFormat;

    #[test]
    #[ignore = "requires CUDA and NVJPEG; run explicitly on a GPU host"]
    fn queued_gpu_frame_preserves_capture_geometry_and_owns_pixels() {
        let runtime = Arc::new(Mutex::new(VisionRuntime::new(0).unwrap()));
        let (width, height) = (160, 96);
        let mut pixels = Vec::new();
        for y in 0..height {
            for _ in 0..width {
                pixels.extend_from_slice(if y < height / 2 {
                    &[0, 0, 255]
                } else {
                    &[255, 0, 0]
                });
            }
        }
        let frame = Frame {
            data: pixels,
            width,
            height,
            timestamp_ms: 123,
            format: FrameFormat::Bgr8,
        };
        runtime
            .lock()
            .unwrap()
            .preprocess_bgr(&frame.data, width, height, 160, 160)
            .unwrap();
        let queued = annotate_frame_gpu(&runtime, &frame, 1, 10.0, vec![], 95).unwrap();

        let later = Frame {
            data: [0, 255, 0].repeat((width * height) as usize),
            width,
            height,
            timestamp_ms: 456,
            format: FrameFormat::Bgr8,
        };
        let _later_job = annotate_frame_gpu(&runtime, &later, 2, 10.0, vec![], 95).unwrap();
        let packet = std::thread::spawn(move || encode_gpu_frame(queued).unwrap())
            .join()
            .unwrap();
        assert_eq!((packet.frame_number, packet.timestamp_ms), (1, 123));
        let decoded = image::load_from_memory(&packet.jpeg).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (160, 96));
        // Sample the image bands between the corner readouts. This regression
        // checks capture geometry and queued frame ownership, not HUD styling.
        let top = decoded.get_pixel(80, 32).0;
        let bottom = decoded.get_pixel(80, 64).0;
        assert!(top[0] > 220 && top[1] < 30 && top[2] < 30, "top: {top:?}");
        assert!(
            bottom[2] > 220 && bottom[0] < 30 && bottom[1] < 30,
            "bottom: {bottom:?}"
        );
    }
}
