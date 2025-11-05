//! GPU runtime for preprocessing frames, drawing overlays, and encoding JPEGs.

use std::{ffi::c_int, ptr, sync::Arc};

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
};
use cudarc::nvrtc::{self, Ptx};
use nvjpeg_sys::*;

use crate::Result;
use tracing::{info_span, trace_span};

/// Holds CUDA resources and scratch buffers for image preprocessing and annotation.
pub struct VisionRuntime {
    device_index: i32,
    _context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    preprocess_fn: CudaFunction,
    nms_fn: CudaFunction,
    annotate_fn: CudaFunction,
    input_bgr: Option<CudaSlice<u8>>,
    resized_bgr: Option<CudaSlice<u8>>,
    tensor_buffer: Option<CudaSlice<f32>>,
    keep_flags: Option<CudaSlice<i32>>,
    boxes_scratch: Option<CudaSlice<i32>>,
    label_positions: Option<CudaSlice<i32>>,
    label_offsets: Option<CudaSlice<i32>>,
    label_lengths: Option<CudaSlice<i32>>,
    label_chars: Option<CudaSlice<u8>>,
    info_text: Option<CudaSlice<u8>>,
    nvjpeg_handle: nvjpegHandle_t,
    nvjpeg_encoder_state: nvjpegEncoderState_t,
    nvjpeg_encoder_params: nvjpegEncoderParams_t,
    jpeg_quality: c_int,
}

// The underlying CUDA and NVJPEG handles are only ever accessed behind a `Mutex` in the
// application layer, so it is safe to mark the runtime as `Send`/`Sync` for cross-thread use.
unsafe impl Send for VisionRuntime {}
unsafe impl Sync for VisionRuntime {}

/// Result of the preprocessing stage.
pub struct PreprocessOutput {
    /// Device pointer to the tensor buffer (GPU address space).
    pub tensor_ptr: u64,
    /// Number of `f32` elements in the tensor buffer.
    pub tensor_len: usize,
    /// Width of the resized output frame.
    pub width: i32,
    /// Height of the resized output frame.
    pub height: i32,
}

impl VisionRuntime {
    /// Compile kernels and allocate the CUDA context for the given device.
    pub fn new(device_index: i32) -> Result<Self> {
        let init_span = info_span!("vision.runtime.init", device = device_index);
        let _init_guard = init_span.enter();

        let context = info_span!("vision.cuda.context", device = device_index)
            .in_scope(|| CudaContext::new(device_index as usize))?;
        let ptx = info_span!("vision.cuda.compile_ptx", device = device_index)
            .in_scope(Self::build_ptx)?;
        let module = info_span!("vision.cuda.module_load", device = device_index)
            .in_scope(|| context.load_module(ptx))?;
        let preprocess_fn = info_span!("vision.cuda.load_kernel", name = "bgr_resize_normalize")
            .in_scope(|| module.load_function("bgr_resize_normalize"))?;
        let nms_fn = info_span!("vision.cuda.load_kernel", name = "nms_suppress")
            .in_scope(|| module.load_function("nms_suppress"))?;
        let annotate_fn = info_span!("vision.cuda.load_kernel", name = "annotate_overlays")
            .in_scope(|| module.load_function("annotate_overlays"))?;
        let stream = context.default_stream();

        let stream_ptr = stream.cu_stream() as cudaStream_t;
        let mut nv_handle: nvjpegHandle_t = ptr::null_mut();
        info_span!("vision.nvjpeg.create").in_scope(|| unsafe {
            Self::check_nvjpeg(nvjpegCreateSimple(&mut nv_handle), "nvjpegCreateSimple")
        })?;

        let mut encoder_state: nvjpegEncoderState_t = ptr::null_mut();
        info_span!("vision.nvjpeg.encoder_state").in_scope(|| unsafe {
            Self::check_nvjpeg(
                nvjpegEncoderStateCreate(nv_handle, &mut encoder_state, stream_ptr),
                "nvjpegEncoderStateCreate",
            )
        })?;

        let mut encoder_params: nvjpegEncoderParams_t = ptr::null_mut();
        info_span!("vision.nvjpeg.encoder_params").in_scope(|| unsafe {
            Self::check_nvjpeg(
                nvjpegEncoderParamsCreate(nv_handle, &mut encoder_params, stream_ptr),
                "nvjpegEncoderParamsCreate",
            )
        })?;

        let default_quality: c_int = 85;
        unsafe {
            info_span!("vision.nvjpeg.configure", step = "encoding").in_scope(|| {
                Self::check_nvjpeg(
                    nvjpegEncoderParamsSetEncoding(
                        encoder_params,
                        nvjpegJpegEncoding_t_NVJPEG_ENCODING_BASELINE_DCT,
                        stream_ptr,
                    ),
                    "nvjpegEncoderParamsSetEncoding",
                )
            })?;
            info_span!("vision.nvjpeg.configure", step = "sampling").in_scope(|| {
                Self::check_nvjpeg(
                    nvjpegEncoderParamsSetSamplingFactors(
                        encoder_params,
                        nvjpegChromaSubsampling_t_NVJPEG_CSS_444,
                        stream_ptr,
                    ),
                    "nvjpegEncoderParamsSetSamplingFactors",
                )
            })?;
            info_span!("vision.nvjpeg.configure", step = "huffman").in_scope(|| {
                Self::check_nvjpeg(
                    nvjpegEncoderParamsSetOptimizedHuffman(encoder_params, 1, stream_ptr),
                    "nvjpegEncoderParamsSetOptimizedHuffman",
                )
            })?;
            info_span!("vision.nvjpeg.configure", step = "quality").in_scope(|| {
                Self::check_nvjpeg(
                    nvjpegEncoderParamsSetQuality(encoder_params, default_quality, stream_ptr),
                    "nvjpegEncoderParamsSetQuality",
                )
            })?;
        }

        Ok(Self {
            device_index,
            _context: context,
            stream,
            _module: module,
            preprocess_fn,
            nms_fn,
            annotate_fn,
            input_bgr: None,
            resized_bgr: None,
            tensor_buffer: None,
            keep_flags: None,
            boxes_scratch: None,
            label_positions: None,
            label_offsets: None,
            label_lengths: None,
            label_chars: None,
            info_text: None,
            nvjpeg_handle: nv_handle,
            nvjpeg_encoder_state: encoder_state,
            nvjpeg_encoder_params: encoder_params,
            jpeg_quality: default_quality,
        })
    }

    /// Encode the annotated frame buffer to JPEG using NVJPEG.
    pub fn encode_jpeg(&mut self, width: i32, height: i32, quality: c_int) -> Result<Vec<u8>> {
        let span = info_span!(
            "vision.gpu.encode_jpeg",
            width = width,
            height = height,
            quality = quality
        );
        let _guard = span.enter();

        if self.resized_bgr.is_none() {
            return Err("no annotated frame available".into());
        }

        let stream_ptr = self.stream.cu_stream() as cudaStream_t;

        if quality != self.jpeg_quality {
            unsafe {
                info_span!("vision.nvjpeg.configure", step = "quality_update").in_scope(|| {
                    Self::check_nvjpeg(
                        nvjpegEncoderParamsSetQuality(
                            self.nvjpeg_encoder_params,
                            quality,
                            stream_ptr,
                        ),
                        "nvjpegEncoderParamsSetQuality",
                    )
                })?;
            }
            self.jpeg_quality = quality;
        }

        let bgr = self.resized_bgr.as_ref().unwrap();
        let (device_ptr, device_sync) = bgr.device_ptr(&self.stream);
        let mut image = nvjpegImage_t::new();
        image.channel[0] = (device_ptr as usize) as *mut ::std::os::raw::c_uchar;
        image.pitch[0] = (width as usize) * 3;

        unsafe {
            info_span!("vision.nvjpeg.encode", stage = "launch").in_scope(|| {
                Self::check_nvjpeg(
                    nvjpegEncodeImage(
                        self.nvjpeg_handle,
                        self.nvjpeg_encoder_state,
                        self.nvjpeg_encoder_params,
                        &image,
                        nvjpegInputFormat_t_NVJPEG_INPUT_BGRI,
                        width,
                        height,
                        stream_ptr,
                    ),
                    "nvjpegEncodeImage",
                )
            })?;
        }
        drop(device_sync);

        let mut length: usize = 0;
        unsafe {
            info_span!("vision.nvjpeg.encode", stage = "query_size").in_scope(|| {
                Self::check_nvjpeg(
                    nvjpegEncodeRetrieveBitstream(
                        self.nvjpeg_handle,
                        self.nvjpeg_encoder_state,
                        ptr::null_mut(),
                        &mut length,
                        stream_ptr,
                    ),
                    "nvjpegEncodeRetrieveBitstream(size)",
                )
            })?;
        }

        let mut output = vec![0u8; length];
        unsafe {
            info_span!("vision.nvjpeg.encode", stage = "download").in_scope(|| {
                Self::check_nvjpeg(
                    nvjpegEncodeRetrieveBitstream(
                        self.nvjpeg_handle,
                        self.nvjpeg_encoder_state,
                        output.as_mut_ptr(),
                        &mut length,
                        stream_ptr,
                    ),
                    "nvjpegEncodeRetrieveBitstream(data)",
                )
            })?;
        }
        output.truncate(length);
        trace_span!("vision.cuda.stream_sync", op = "nvjpeg")
            .in_scope(|| self.stream.synchronize())?;
        Ok(output)
    }

    /// Compile CUDA kernels for preprocessing, annotation, and NMS.
    fn build_ptx() -> Result<Ptx> {
        const SOURCE: &str = include_str!("vision_kernels.cu");
        info_span!("vision.cuda.nvrtc", kernel = "vision").in_scope(|| {
            nvrtc::compile_ptx(SOURCE)
                .map_err(|err| Box::new(err) as Box<dyn std::error::Error + Send + Sync>)
        })
    }

    /// Map NVJPEG error codes into `Result`.
    fn check_nvjpeg(status: nvjpegStatus_t, label: &str) -> Result<()> {
        if status == nvjpegStatus_t_NVJPEG_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(format!("{label} failed with status {status}").into())
        }
    }

    /// Ensure the input staging buffer is large enough for the upcoming frame.
    fn ensure_input_buffer(&mut self, len: usize) -> Result<&mut CudaSlice<u8>> {
        let needs = match self.input_bgr {
            Some(ref buf) if buf.len() >= len => false,
            _ => true,
        };
        if needs {
            self.input_bgr = Some(self.stream.alloc_zeros::<u8>(len)?);
        }
        Ok(self.input_bgr.as_mut().unwrap())
    }

    /// Ensure the resized BGR buffer is allocated.
    fn ensure_resized_buffer(&mut self, len: usize) -> Result<&mut CudaSlice<u8>> {
        let needs = match self.resized_bgr {
            Some(ref buf) if buf.len() >= len => false,
            _ => true,
        };
        if needs {
            self.resized_bgr = Some(self.stream.alloc_zeros::<u8>(len)?);
        }
        Ok(self.resized_bgr.as_mut().unwrap())
    }

    /// Ensure the tensor output buffer is allocated.
    fn ensure_tensor_buffer(&mut self, len: usize) -> Result<&mut CudaSlice<f32>> {
        let needs = match self.tensor_buffer {
            Some(ref buf) if buf.len() >= len => false,
            _ => true,
        };
        if needs {
            self.tensor_buffer = Some(self.stream.alloc_zeros::<f32>(len)?);
        }
        Ok(self.tensor_buffer.as_mut().unwrap())
    }

    /// Ensure the NMS keep-flag buffer is allocated.
    fn ensure_keep_flags(&mut self, len: usize) -> Result<&mut CudaSlice<i32>> {
        let needs = match self.keep_flags {
            Some(ref buf) if buf.len() >= len => false,
            _ => true,
        };
        if needs {
            self.keep_flags = Some(self.stream.alloc_zeros::<i32>(len)?);
        }
        Ok(self.keep_flags.as_mut().unwrap())
    }

    /// Ensure the boxes scratch buffer is allocated.
    fn ensure_boxes_scratch(&mut self, len: usize) -> Result<&mut CudaSlice<i32>> {
        let needs = match self.boxes_scratch {
            Some(ref buf) if buf.len() >= len => false,
            _ => true,
        };
        if needs {
            self.boxes_scratch = Some(self.stream.alloc_zeros::<i32>(len)?);
        }
        Ok(self.boxes_scratch.as_mut().unwrap())
    }

    /// Ensure the label positions buffer is allocated.
    fn ensure_label_positions(&mut self, len: usize) -> Result<&mut CudaSlice<i32>> {
        let needs = match self.label_positions {
            Some(ref buf) if buf.len() >= len => false,
            _ => true,
        };
        if needs {
            self.label_positions = Some(self.stream.alloc_zeros::<i32>(len)?);
        }
        Ok(self.label_positions.as_mut().unwrap())
    }

    /// Ensure the label offsets buffer is allocated.
    fn ensure_label_offsets(&mut self, len: usize) -> Result<&mut CudaSlice<i32>> {
        let needs = match self.label_offsets {
            Some(ref buf) if buf.len() >= len => false,
            _ => true,
        };
        if needs {
            self.label_offsets = Some(self.stream.alloc_zeros::<i32>(len)?);
        }
        Ok(self.label_offsets.as_mut().unwrap())
    }

    /// Ensure the label lengths buffer is allocated.
    fn ensure_label_lengths(&mut self, len: usize) -> Result<&mut CudaSlice<i32>> {
        let needs = match self.label_lengths {
            Some(ref buf) if buf.len() >= len => false,
            _ => true,
        };
        if needs {
            self.label_lengths = Some(self.stream.alloc_zeros::<i32>(len)?);
        }
        Ok(self.label_lengths.as_mut().unwrap())
    }

    /// Ensure the label character buffer is allocated.
    fn ensure_label_chars(&mut self, len: usize) -> Result<&mut CudaSlice<u8>> {
        let needs = match self.label_chars {
            Some(ref buf) if buf.len() >= len => false,
            _ => true,
        };
        if needs {
            self.label_chars = Some(self.stream.alloc_zeros::<u8>(len)?);
        }
        Ok(self.label_chars.as_mut().unwrap())
    }

    /// Ensure the info text buffer is allocated.
    fn ensure_info_text(&mut self, len: usize) -> Result<&mut CudaSlice<u8>> {
        let needs = match self.info_text {
            Some(ref buf) if buf.len() >= len => false,
            _ => true,
        };
        if needs {
            self.info_text = Some(self.stream.alloc_zeros::<u8>(len)?);
        }
        Ok(self.info_text.as_mut().unwrap())
    }

    /// Uploads BGR pixels, resizes/normalises them, and stores the tensor-ready buffer on the GPU.
    pub fn preprocess_bgr(
        &mut self,
        bgr: &[u8],
        in_width: i32,
        in_height: i32,
        out_width: i32,
        out_height: i32,
    ) -> Result<PreprocessOutput> {
        let span = info_span!(
            "vision.gpu.preprocess",
            width = in_width,
            height = in_height,
            out_width = out_width,
            out_height = out_height,
            bytes = bgr.len()
        );
        let _guard = span.enter();

        let num_bytes_in = bgr.len();
        let num_pixels_out = (out_width * out_height) as usize;
        let tensor_len = num_pixels_out * 3;
        let bgr_out_bytes = num_pixels_out * 3;

        self.ensure_input_buffer(num_bytes_in)?;
        self.ensure_resized_buffer(bgr_out_bytes)?;
        self.ensure_tensor_buffer(tensor_len)?;

        let stream = self.stream.clone();
        let preprocess_fn = self.preprocess_fn.clone();

        {
            let input_slice = self.input_bgr.as_mut().unwrap();
            trace_span!(
                "vision.cuda.memcpy",
                direction = "htod",
                bytes = num_bytes_in
            )
            .in_scope(|| stream.memcpy_htod(bgr, input_slice))?;
        }

        let input_view = self.input_bgr.as_ref().unwrap().as_view();
        let mut tensor_view = {
            let tensor = self.tensor_buffer.as_mut().unwrap();
            tensor.as_view_mut()
        };
        let mut output_view = {
            let output = self.resized_bgr.as_mut().unwrap();
            output.as_view_mut()
        };

        let launch = LaunchConfig::for_num_elems(num_pixels_out as u32);
        trace_span!("vision.cuda.kernel", name = "bgr_resize_normalize").in_scope(|| unsafe {
            stream
                .launch_builder(&preprocess_fn)
                .arg(&input_view)
                .arg(&in_width)
                .arg(&in_height)
                .arg(&out_width)
                .arg(&out_height)
                .arg(&mut tensor_view)
                .arg(&mut output_view)
                .launch(launch)
        })?;

        drop(tensor_view);
        drop(output_view);

        trace_span!("vision.cuda.stream_sync", op = "preprocess")
            .in_scope(|| stream.synchronize())?;

        let tensor_buf = self.tensor_buffer.as_ref().unwrap();
        let (tensor_ptr, _sync) = tensor_buf.device_ptr(&stream);

        Ok(PreprocessOutput {
            tensor_ptr,
            tensor_len,
            width: out_width,
            height: out_height,
        })
    }

    /// Runs in-place NMS on an array of boxes stored on the device.
    pub fn run_nms(
        &mut self,
        boxes_ptr: u64,
        num_boxes: usize,
        iou_threshold: f32,
    ) -> Result<Vec<i32>> {
        let span = info_span!(
            "vision.gpu.nms",
            boxes = num_boxes,
            threshold = iou_threshold
        );
        let _guard = span.enter();

        let stream = self.stream.clone();
        let nms_fn = self.nms_fn.clone();
        self.ensure_keep_flags(num_boxes)?;
        let launch = LaunchConfig::for_num_elems(1);
        {
            let keep_slice = self.keep_flags.as_mut().unwrap();
            trace_span!("vision.cuda.kernel", name = "nms_suppress").in_scope(|| unsafe {
                stream
                    .launch_builder(&nms_fn)
                    .arg(&boxes_ptr)
                    .arg(&(num_boxes as i32))
                    .arg(&iou_threshold)
                    .arg(keep_slice)
                    .launch(launch)
            })?;
        }
        trace_span!("vision.cuda.stream_sync", op = "nms").in_scope(|| stream.synchronize())?;
        let mut result = trace_span!(
            "vision.cuda.memcpy",
            direction = "dtoh",
            bytes = num_boxes * std::mem::size_of::<i32>()
        )
        .in_scope(|| stream.memcpy_dtov(self.keep_flags.as_ref().unwrap()))?;
        result.truncate(num_boxes);
        Ok(result)
    }

    /// Annotates the resized BGR buffer with detection overlays.
    #[allow(clippy::too_many_arguments)]
    pub fn annotate(
        &mut self,
        width: i32,
        height: i32,
        boxes: &[i32],
        label_positions: &[i32],
        label_offsets: &[i32],
        label_lengths: &[i32],
        label_chars: &[u8],
        info_text: &[u8],
        info_origin: (i32, i32),
    ) -> Result<()> {
        let span = info_span!(
            "vision.gpu.annotate",
            width = width,
            height = height,
            boxes = boxes.len() / 4,
            labels = label_offsets.len()
        );
        let _guard = span.enter();

        if self.resized_bgr.is_none() {
            return Ok(());
        }

        let stream = self.stream.clone();
        let annotate_fn = self.annotate_fn.clone();

        self.ensure_boxes_scratch(boxes.len())?;
        self.ensure_label_positions(label_positions.len())?;
        self.ensure_label_offsets(label_offsets.len())?;
        self.ensure_label_lengths(label_lengths.len())?;
        self.ensure_label_chars(label_chars.len())?;
        self.ensure_info_text(info_text.len())?;

        {
            let buf = self.boxes_scratch.as_mut().unwrap();
            trace_span!(
                "vision.cuda.memcpy",
                direction = "htod",
                bytes = boxes.len() * std::mem::size_of::<i32>()
            )
            .in_scope(|| stream.memcpy_htod(boxes, buf))?;
        }
        {
            let buf = self.label_positions.as_mut().unwrap();
            trace_span!(
                "vision.cuda.memcpy",
                direction = "htod",
                bytes = label_positions.len() * std::mem::size_of::<i32>()
            )
            .in_scope(|| stream.memcpy_htod(label_positions, buf))?;
        }
        {
            let buf = self.label_offsets.as_mut().unwrap();
            trace_span!(
                "vision.cuda.memcpy",
                direction = "htod",
                bytes = label_offsets.len() * std::mem::size_of::<i32>()
            )
            .in_scope(|| stream.memcpy_htod(label_offsets, buf))?;
        }
        {
            let buf = self.label_lengths.as_mut().unwrap();
            trace_span!(
                "vision.cuda.memcpy",
                direction = "htod",
                bytes = label_lengths.len() * std::mem::size_of::<i32>()
            )
            .in_scope(|| stream.memcpy_htod(label_lengths, buf))?;
        }
        {
            let buf = self.label_chars.as_mut().unwrap();
            trace_span!(
                "vision.cuda.memcpy",
                direction = "htod",
                bytes = label_chars.len()
            )
            .in_scope(|| stream.memcpy_htod(label_chars, buf))?;
        }
        {
            let buf = self.info_text.as_mut().unwrap();
            trace_span!(
                "vision.cuda.memcpy",
                direction = "htod",
                bytes = info_text.len()
            )
            .in_scope(|| stream.memcpy_htod(info_text, buf))?;
        }

        let mut output_view = {
            let output = self.resized_bgr.as_mut().unwrap();
            output.as_view_mut()
        };
        let boxes_view = self.boxes_scratch.as_ref().unwrap().as_view();
        let label_pos_view = self.label_positions.as_ref().unwrap().as_view();
        let offsets_view = self.label_offsets.as_ref().unwrap().as_view();
        let lengths_view = self.label_lengths.as_ref().unwrap().as_view();
        let chars_view = self.label_chars.as_ref().unwrap().as_view();
        let info_view = self.info_text.as_ref().unwrap().as_view();

        let num_boxes = (boxes.len() / 4) as i32;
        let launch = LaunchConfig::for_num_elems(num_boxes.max(1) as u32);

        trace_span!("vision.cuda.kernel", name = "annotate_overlays").in_scope(|| unsafe {
            stream
                .launch_builder(&annotate_fn)
                .arg(&mut output_view)
                .arg(&width)
                .arg(&height)
                .arg(&boxes_view)
                .arg(&label_pos_view)
                .arg(&offsets_view)
                .arg(&lengths_view)
                .arg(&chars_view)
                .arg(&num_boxes)
                .arg(&info_view)
                .arg(&(info_text.len() as i32))
                .arg(&info_origin.0)
                .arg(&info_origin.1)
                .launch(launch)
        })?;

        drop(output_view);
        Ok(())
    }

    /// Downloads the resized BGR frame into host memory.
    /// Download the resized BGR buffer into host memory.
    pub fn download_bgr(&self, width: i32, height: i32) -> Result<Vec<u8>> {
        let span = info_span!("vision.gpu.download", width = width, height = height);
        let _guard = span.enter();
        let len = (width as usize) * (height as usize) * 3;
        if let Some(ref buf) = self.resized_bgr {
            let mut host = vec![0u8; len];
            trace_span!("vision.cuda.memcpy", direction = "dtoh", bytes = len)
                .in_scope(|| self.stream.memcpy_dtoh(buf, host.as_mut_slice()))?;
            trace_span!("vision.cuda.stream_sync", op = "download")
                .in_scope(|| self.stream.synchronize())?;
            Ok(host)
        } else {
            Ok(vec![0u8; len])
        }
    }

    /// Return the CUDA device index owning the runtime.
    pub fn device_index(&self) -> i32 {
        self.device_index
    }
}

impl Drop for VisionRuntime {
    fn drop(&mut self) {
        unsafe {
            if !self.nvjpeg_encoder_params.is_null() {
                let _ = nvjpegEncoderParamsDestroy(self.nvjpeg_encoder_params);
                self.nvjpeg_encoder_params = ptr::null_mut();
            }
            if !self.nvjpeg_encoder_state.is_null() {
                let _ = nvjpegEncoderStateDestroy(self.nvjpeg_encoder_state);
                self.nvjpeg_encoder_state = ptr::null_mut();
            }
            if !self.nvjpeg_handle.is_null() {
                let _ = nvjpegDestroy(self.nvjpeg_handle);
                self.nvjpeg_handle = ptr::null_mut();
            }
        }
    }
}
