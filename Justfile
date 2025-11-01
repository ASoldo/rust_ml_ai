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

vision camera='/dev/video0' model=VISION_MODEL_PATH width='640' height='640' flags='':
    {{CARGO}} run --release -p {{BIN}} --features {{WITH_TCH_FEATURE}} -- vision {{camera}} {{model}} {{width}} {{height}} {{flags}}

vision-nvdec camera='/dev/video0' model=VISION_MODEL_PATH width='640' height='640' flags='--nvdec':
    {{CARGO}} run --release -p {{BIN}} --features {{WITH_TCH_FEATURE}} -- vision {{camera}} {{model}} {{width}} {{height}} {{flags}}

check-cuda:
    mkdir -p target
    /opt/cuda/bin/nvcc tools/check_cuda.cu -o target/check_cuda
    target/check_cuda
