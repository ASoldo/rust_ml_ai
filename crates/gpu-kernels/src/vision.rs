use std::sync::Arc;

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
};
use cudarc::nvrtc::{self, Ptx};

use crate::Result;

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
    resized_rgba: Option<CudaSlice<u8>>,
    tensor_buffer: Option<CudaSlice<f32>>,
    keep_flags: Option<CudaSlice<i32>>,
    boxes_scratch: Option<CudaSlice<i32>>,
    label_positions: Option<CudaSlice<i32>>,
    label_offsets: Option<CudaSlice<i32>>,
    label_lengths: Option<CudaSlice<i32>>,
    label_chars: Option<CudaSlice<u8>>,
    info_text: Option<CudaSlice<u8>>,
}

/// Result of the preprocessing stage.
pub struct PreprocessOutput {
    pub tensor_ptr: u64,
    pub tensor_len: usize,
    pub width: i32,
    pub height: i32,
}

impl VisionRuntime {
    /// Compile kernels and allocate the CUDA context for the given device.
    pub fn new(device_index: i32) -> Result<Self> {
        let context = CudaContext::new(device_index as usize)?;
        let module = context.load_module(Self::build_ptx()?)?;
        let preprocess_fn = module.load_function("bgr_resize_normalize")?;
        let nms_fn = module.load_function("nms_suppress")?;
        let annotate_fn = module.load_function("annotate_overlays")?;
        let stream = context.default_stream();

        Ok(Self {
            device_index,
            _context: context,
            stream,
            _module: module,
            preprocess_fn,
            nms_fn,
            annotate_fn,
            input_bgr: None,
            resized_rgba: None,
            tensor_buffer: None,
            keep_flags: None,
            boxes_scratch: None,
            label_positions: None,
            label_offsets: None,
            label_lengths: None,
            label_chars: None,
            info_text: None,
        })
    }

    fn build_ptx() -> Result<Ptx> {
        const SOURCE: &str = include_str!("vision_kernels.cu");
        nvrtc::compile_ptx(SOURCE)
            .map_err(|err| Box::new(err) as Box<dyn std::error::Error + Send + Sync>)
    }

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

    fn ensure_resized_buffer(&mut self, len: usize) -> Result<&mut CudaSlice<u8>> {
        let needs = match self.resized_rgba {
            Some(ref buf) if buf.len() >= len => false,
            _ => true,
        };
        if needs {
            self.resized_rgba = Some(self.stream.alloc_zeros::<u8>(len)?);
        }
        Ok(self.resized_rgba.as_mut().unwrap())
    }

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

    /// Uploads RGBA pixels, resizes/normalises them, and stores the tensor-ready buffer on the GPU.
    pub fn preprocess_bgr(
        &mut self,
        bgr: &[u8],
        in_width: i32,
        in_height: i32,
        out_width: i32,
        out_height: i32,
    ) -> Result<PreprocessOutput> {
        let num_pixels_in = bgr.len();
        let num_pixels_out = (out_width * out_height) as usize;
        let tensor_len = num_pixels_out * 3;
        let rgba_out_len = num_pixels_out * 4;

        self.ensure_input_buffer(num_pixels_in)?;
        self.ensure_resized_buffer(rgba_out_len)?;
        self.ensure_tensor_buffer(tensor_len)?;

        let stream = self.stream.clone();
        let preprocess_fn = self.preprocess_fn.clone();

        {
            let input_slice = self.input_bgr.as_mut().unwrap();
            stream.memcpy_htod(bgr, input_slice)?;
        }

        let input_view = self.input_bgr.as_ref().unwrap().as_view();
        let mut tensor_view = {
            let tensor = self.tensor_buffer.as_mut().unwrap();
            tensor.as_view_mut()
        };
        let mut output_view = {
            let output = self.resized_rgba.as_mut().unwrap();
            output.as_view_mut()
        };

        let launch = LaunchConfig::for_num_elems(num_pixels_out as u32);
        unsafe {
            stream
                .launch_builder(&preprocess_fn)
                .arg(&input_view)
                .arg(&in_width)
                .arg(&in_height)
                .arg(&out_width)
                .arg(&out_height)
                .arg(&mut tensor_view)
                .arg(&mut output_view)
                .launch(launch)?;
        }

        drop(tensor_view);
        drop(output_view);

        stream.synchronize()?;

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
        let stream = self.stream.clone();
        let nms_fn = self.nms_fn.clone();
        self.ensure_keep_flags(num_boxes)?;
        let launch = LaunchConfig::for_num_elems(1);
        {
            let keep_slice = self.keep_flags.as_mut().unwrap();
            unsafe {
                stream
                    .launch_builder(&nms_fn)
                    .arg(&boxes_ptr)
                    .arg(&(num_boxes as i32))
                    .arg(&iou_threshold)
                    .arg(keep_slice)
                    .launch(launch)?;
            }
        }
        stream.synchronize()?;
        let result = stream.memcpy_dtov(self.keep_flags.as_ref().unwrap())?;
        Ok(result)
    }

    /// Annotates the resized RGBA buffer with detection overlays.
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
        if self.resized_rgba.is_none() {
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
            stream.memcpy_htod(boxes, buf)?;
        }
        {
            let buf = self.label_positions.as_mut().unwrap();
            stream.memcpy_htod(label_positions, buf)?;
        }
        {
            let buf = self.label_offsets.as_mut().unwrap();
            stream.memcpy_htod(label_offsets, buf)?;
        }
        {
            let buf = self.label_lengths.as_mut().unwrap();
            stream.memcpy_htod(label_lengths, buf)?;
        }
        {
            let buf = self.label_chars.as_mut().unwrap();
            stream.memcpy_htod(label_chars, buf)?;
        }
        {
            let buf = self.info_text.as_mut().unwrap();
            stream.memcpy_htod(info_text, buf)?;
        }

        let mut output_view = {
            let output = self.resized_rgba.as_mut().unwrap();
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

        unsafe {
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
                .launch(launch)?;
        }

        drop(output_view);

        stream.synchronize()?;
        Ok(())
    }

    /// Downloads the resized RGBA frame into host memory.
    pub fn download_rgba(&self, width: i32, height: i32) -> Result<Vec<u8>> {
        let len = (width as usize) * (height as usize) * 4;
        if let Some(ref buf) = self.resized_rgba {
            let mut host = vec![0u8; len];
            self.stream.memcpy_dtoh(buf, host.as_mut_slice())?;
            self.stream.synchronize()?;
            Ok(host)
        } else {
            Ok(vec![0u8; len])
        }
    }

    pub fn device_index(&self) -> i32 {
        self.device_index
    }
}
