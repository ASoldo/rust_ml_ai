set shell := ["zsh", "-lc"]

CARGO := "cargo"
BIN := "vision"
WITH_TCH_FEATURE := "with-tch"
MNIST_DATA_DIR := "data/mnist"
MNIST_MODEL_PATH := "models/mnist-linear.ot"
VISION_MODEL_PATH := "models/yolov12n-face.torchscript"

default:
    @just --list

check:
    {{CARGO}} check --workspace

build:
    {{CARGO}} build --workspace

build-release:
    {{CARGO}} build --workspace --release

fmt:
    {{CARGO}} fmt --all

lint:
    {{CARGO}} clippy --workspace --all-targets -- -D warnings

test:
    {{CARGO}} test --workspace

run:
    {{CARGO}} run -p {{BIN}}

run-release:
    {{CARGO}} run --release -p {{BIN}}

mnist-help:
    {{CARGO}} run -p {{BIN}} --features {{WITH_TCH_FEATURE}} -- mnist-help

mnist-train data_dir=MNIST_DATA_DIR model_out=MNIST_MODEL_PATH epochs='5' batch_size='128' learning_rate='0.001' device='':
    {{CARGO}} run -p {{BIN}} --features {{WITH_TCH_FEATURE}} -- mnist-train {{data_dir}} {{model_out}} {{epochs}} {{batch_size}} {{learning_rate}} {{device}}

mnist-predict image_path model_path=MNIST_MODEL_PATH device='':
    {{CARGO}} run -p {{BIN}} --features {{WITH_TCH_FEATURE}} -- mnist-predict {{model_path}} {{image_path}} {{device}}

# vision
vision camera='/dev/video0' model=VISION_MODEL_PATH width='640' height='640' flags='':
    {{CARGO}} run --release -p {{BIN}} --features {{WITH_TCH_FEATURE}} -- vision {{camera}} {{model}} {{width}} {{height}} {{flags}} --processors 2 --batch-size 2

vision-batch camera='/dev/video0' model=VISION_MODEL_PATH width='640' height='640' flags='':
    {{CARGO}} run --release -p {{BIN}} --features {{WITH_TCH_FEATURE}} -- \
    vision {{camera}} {{model}} {{width}} {{height}} {{flags}} --processors 4 --batch-size 2

vision-nvdec camera='/dev/video0' model=VISION_MODEL_PATH width='640' height='640' flags='--nvdec':
    {{CARGO}} run --release -p {{BIN}} --features {{WITH_TCH_FEATURE}} -- vision {{camera}} {{model}} {{width}} {{height}} {{flags}}

vision-gdb:
    sudo gdb -p $(pgrep -f 'vision /dev/video0')

vision-udp:
    cargo run --release -p vision --features with-tch -- \
    vision --source 'udp://127.0.0.1:5000?sprop=Z/QAFpGWgKA9sBagIMDIAAADAAgAAAMA9HixdQ==,aO8xkhk=&payload=96' \
    --model models/yolov12n-face.torchscript \
    --width 640 --height 640 \
    --verbose

vision-ptz:
    cargo run --release -p vision --features with-tch -- \
    vision --source "udp://127.0.0.1:5000?sprop=Z/QAFpGWgKA9sBagIMDIAAADAAgAAAMA9HixdQ==,aO8xkhk=&payload=96" \
    --model models/yolov12n-face.torchscript \
    --width 640 --height 640 \
    --verbose  --processors 2 --batch-size 2

gstream:
    gst-launch-1.0 -v \
    v4l2src device=/dev/video0 ! \
    videoconvert ! \
    x264enc tune=zerolatency bitrate=2000 speed-preset=superfast ! \
    rtph264pay config-interval=1 pt=96 ! \
    udpsink host=127.0.0.1 port=5000

gstreamer:
    gst-launch-1.0 -v \
    rtspsrc location="rtsp://root:root@192.168.0.90/axis-media/media.amp?resolution=1920x1080&fps=60" \
    latency=100 protocols=udp timeout=5000000 ! \
    rtph264depay ! h264parse config-interval=-1 ! \
    video/x-h264,stream-format=byte-stream,alignment=au ! \
    rtph264pay pt=96 config-interval=1 ! \
    udpsink host=127.0.0.1 port=5000 sync=false async=false

check-cuda:
    mkdir -p target
    /opt/cuda/bin/nvcc tools/check_cuda.cu -o target/check_cuda
    target/check_cuda
