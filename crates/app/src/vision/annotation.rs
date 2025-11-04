use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use gpu_kernels::VisionRuntime;
use image::{DynamicImage, ImageBuffer, Rgba, codecs::jpeg::JpegEncoder};

use crate::vision::{
    data::{DetectionSummary, FramePacket},
    encoding::GpuEncodeJob,
};
use video_ingest::Frame;

pub(crate) fn annotate_frame_cpu(
    frame: &Frame,
    frame_number: u64,
    fps: f32,
    summaries: Vec<DetectionSummary>,
    jpeg_quality: i32,
) -> Result<FramePacket> {
    let width = frame.width as u32;
    let height = frame.height as u32;
    let rgba = bgr_to_rgba(&frame.data);
    let mut image = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_vec(width, height, rgba)
        .ok_or_else(|| anyhow!("failed to convert frame into image buffer"))?;

    for summary in &summaries {
        let left = summary.bbox[0].clamp(0.0, (width - 1) as f32);
        let top = summary.bbox[1].clamp(0.0, (height - 1) as f32);
        let right = summary.bbox[2].clamp(0.0, (width - 1) as f32);
        let bottom = summary.bbox[3].clamp(0.0, (height - 1) as f32);
        draw_rectangle(
            &mut image,
            left.round() as i32,
            top.round() as i32,
            right.round() as i32,
            bottom.round() as i32,
            Rgba([0, 255, 0, 255]),
        );
    }

    for summary in &summaries {
        let left = summary.bbox[0].clamp(0.0, (width - 1) as f32);
        let top = summary.bbox[1].clamp(0.0, (height - 1) as f32);
        let label_text = format!(
            "{} {} {:.0}%",
            summary.class,
            summary.track_id,
            summary.score * 100.0
        );
        let label_x = left.round() as i32;
        let label_y = (top.round() as i32 - 12).max(0);
        let text_width = label_text.chars().count() as i32 * 6;
        fill_rect(
            &mut image,
            label_x,
            label_y,
            label_x + text_width,
            label_y + 8,
            Rgba([0, 0, 0, 180]),
        );
        draw_label(
            &mut image,
            label_x,
            label_y,
            &label_text,
            Rgba([0, 255, 0, 255]),
        );
    }

    let info = format!("FRAME {:06}  FPS {:4.1}", frame_number, fps);
    let info_width = (info.chars().count() as i32 * 6).min(width as i32);
    let info_x = (width as i32 - info_width - 4).max(0);
    let info_y = (height as i32 - 12).max(0);
    fill_rect(
        &mut image,
        info_x,
        info_y,
        info_x + info_width + 4,
        info_y + 8,
        Rgba([0, 0, 0, 180]),
    );
    draw_label(
        &mut image,
        info_x + 2,
        info_y,
        &info,
        Rgba([255, 255, 255, 255]),
    );

    let rgb = DynamicImage::ImageRgba8(image).to_rgb8();
    let mut buffer = Vec::new();
    let quality = jpeg_quality.clamp(1, 100) as u8;
    JpegEncoder::new_with_quality(&mut buffer, quality)
        .encode_image(&rgb)
        .map_err(|err| anyhow!("JPEG encode failed: {err}"))?;

    Ok(FramePacket {
        jpeg: buffer,
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
    summaries: Vec<DetectionSummary>,
    boxes_px: &[[i32; 4]],
    label_positions: &[(i32, i32)],
    labels: &[String],
    jpeg_quality: i32,
) -> Result<GpuEncodeJob> {
    let width = frame.width;
    let height = frame.height;

    {
        let mut guard = runtime
            .lock()
            .map_err(|_| anyhow!("vision runtime poisoned"))?;

        let mut boxes_flat = Vec::with_capacity(boxes_px.len() * 4);
        for b in boxes_px {
            boxes_flat.extend_from_slice(b);
        }

        let mut label_positions_flat = Vec::with_capacity(label_positions.len() * 2);
        for (x, y) in label_positions {
            label_positions_flat.push(*x);
            label_positions_flat.push(*y);
        }

        let mut offsets = Vec::with_capacity(labels.len());
        let mut lengths = Vec::with_capacity(labels.len());
        let mut chars = Vec::new();
        for text in labels {
            offsets.push(chars.len() as i32);
            let upper = text.to_uppercase();
            chars.extend_from_slice(upper.as_bytes());
            lengths.push(upper.len() as i32);
        }

        let info = format!("FRAME {:06}  FPS {:4.1}", frame_number, fps).to_uppercase();
        let info_width = ((info.chars().count() as i32) * 6).min(width as i32);
        let info_x = (width as i32 - info_width - 4).max(0);
        let info_y = (height as i32 - 12).max(0);

        guard
            .annotate(
                width,
                height,
                &boxes_flat,
                &label_positions_flat,
                &offsets,
                &lengths,
                &chars,
                info.as_bytes(),
                (info_x, info_y),
            )
            .map_err(|err| anyhow!("annotation kernel failed: {err}"))?;
    }

    Ok(GpuEncodeJob {
        runtime: runtime.clone(),
        width,
        height,
        summaries,
        timestamp_ms: frame.timestamp_ms,
        frame_number,
        fps,
        jpeg_quality,
    })
}

fn bgr_to_rgba(input: &[u8]) -> Vec<u8> {
    let pixels = input.len() / 3;
    let mut output = Vec::with_capacity(pixels * 4);
    for chunk in input.chunks_exact(3) {
        output.push(chunk[2]);
        output.push(chunk[1]);
        output.push(chunk[0]);
        output.push(255);
    }
    output
}

fn draw_rectangle(
    image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
    color: Rgba<u8>,
) {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let left = left.clamp(0, width.saturating_sub(1));
    let right = right.clamp(0, width.saturating_sub(1));
    let top = top.clamp(0, height.saturating_sub(1));
    let bottom = bottom.clamp(0, height.saturating_sub(1));

    for x in left..=right {
        if top >= 0 && top < height {
            *image.get_pixel_mut(x as u32, top as u32) = color;
        }
        if bottom >= 0 && bottom < height {
            *image.get_pixel_mut(x as u32, bottom as u32) = color;
        }
    }
    for y in top..=bottom {
        if left >= 0 && left < width {
            *image.get_pixel_mut(left as u32, y as u32) = color;
        }
        if right >= 0 && right < width {
            *image.get_pixel_mut(right as u32, y as u32) = color;
        }
    }
}

fn fill_rect(
    image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
    color: Rgba<u8>,
) {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let left = left.clamp(0, width.saturating_sub(1));
    let right = right.clamp(0, width.saturating_sub(1));
    let top = top.clamp(0, height.saturating_sub(1));
    let bottom = bottom.clamp(0, height.saturating_sub(1));

    for y in top..=bottom {
        for x in left..=right {
            *image.get_pixel_mut(x as u32, y as u32) = color;
        }
    }
}

fn draw_label(
    image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    mut x: i32,
    y: i32,
    text: &str,
    color: Rgba<u8>,
) {
    let height = image.height() as i32;
    let baseline = y;
    for ch in text.chars().flat_map(|c| c.to_uppercase()) {
        if let Some(glyph) = glyph_bits(ch) {
            for (row, pattern) in glyph.iter().enumerate() {
                let py = baseline + row as i32;
                if py < 0 || py >= height {
                    continue;
                }
                for col in 0..5 {
                    if (pattern >> (4 - col)) & 1 == 1 {
                        let px = x + col as i32;
                        if px >= 0 && px < image.width() as i32 {
                            *image.get_pixel_mut(px as u32, py as u32) = color;
                        }
                    }
                }
            }
            x += 6;
        } else {
            x += 6;
        }
    }
}

fn glyph_bits(ch: char) -> Option<[u8; 7]> {
    match ch {
        'A' => Some([
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ]),
        'C' => Some([
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ]),
        'E' => Some([
            0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b11111,
        ]),
        'F' => Some([
            0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b10000,
        ]),
        'M' => Some([
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ]),
        'O' => Some([
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ]),
        'P' => Some([
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ]),
        'R' => Some([
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ]),
        'S' => Some([
            0b01111, 0b10000, 0b01110, 0b00001, 0b00001, 0b10001, 0b01110,
        ]),
        'N' => Some([
            0b10001, 0b11001, 0b10101, 0b10101, 0b10011, 0b10001, 0b10001,
        ]),
        '0' => Some([
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ]),
        '1' => Some([
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ]),
        '2' => Some([
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ]),
        '3' => Some([
            0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
        ]),
        '4' => Some([
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ]),
        '5' => Some([
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ]),
        '6' => Some([
            0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ]),
        '7' => Some([
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ]),
        '8' => Some([
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ]),
        '9' => Some([
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
        ]),
        '%' => Some([
            0b10001, 0b10010, 0b00100, 0b01000, 0b10010, 0b10001, 0b00000,
        ]),
        '.' => Some([0, 0, 0, 0, 0, 0b00110, 0b00110]),
        ' ' => Some([0, 0, 0, 0, 0, 0, 0]),
        _ => None,
    }
}
