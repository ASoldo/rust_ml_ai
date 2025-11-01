# syntax=docker/dockerfile:1.6

ARG CUDA_BASE=13.0.1-devel-ubuntu22.04

FROM nvidia/cuda:${CUDA_BASE} AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      ca-certificates \
      pkg-config \
      libssl-dev \
      cmake \
      git \
      unzip \
      libopencv-dev \
      libavcodec-dev \
      libavformat-dev \
      libavutil-dev \
      libswscale-dev \
      libgtk-3-dev \
      llvm-dev \
      libclang-dev \
      clang \
      v4l-utils \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/local/cuda /opt/cuda

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --profile default --default-toolchain stable && \
    /root/.cargo/bin/rustup component add rustfmt clippy

ENV PATH=/root/.cargo/bin:${PATH}

ARG LIBTORCH_URL=https://download.pytorch.org/libtorch/cu129/libtorch-shared-with-deps-2.8.0%2Bcu129.zip
RUN curl -L "${LIBTORCH_URL}" -o /tmp/libtorch.zip && \
    unzip /tmp/libtorch.zip -d /opt && \
    rm /tmp/libtorch.zip

ENV CUDA_HOME=/opt/cuda \
    LIBTORCH=/opt/libtorch \
    LIBTORCH_BYPASS_VERSION_CHECK=0 \
    PATH=/opt/cuda/bin:/opt/nsight-systems/2025.3.2/target-linux-x64:/opt/nsight-compute/2025.3.1:${PATH}

ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LIBTORCH}/lib:${LD_LIBRARY_PATH} \
    LIBRARY_PATH=${CUDA_HOME}/lib64:${LIBRARY_PATH}

WORKDIR /workspace

COPY . .

RUN cargo fetch

RUN cargo build --release -p vision --features with-tch

FROM nvidia/cuda:${CUDA_BASE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libopencv-dev \
      libavcodec-dev \
      libavformat-dev \
      libavutil-dev \
      libswscale-dev \
      libgtk-3-0 \
      v4l-utils \
      libssl3 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/local/cuda /opt/cuda

COPY --from=builder /opt/libtorch /opt/libtorch

ENV CUDA_HOME=/opt/cuda \
    LIBTORCH=/opt/libtorch \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    PATH=/opt/cuda/bin:/opt/nsight-systems/2025.3.2/target-linux-x64:/opt/nsight-compute/2025.3.1:${PATH}

ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LIBTORCH}/lib:${LD_LIBRARY_PATH} \
    LIBRARY_PATH=${CUDA_HOME}/lib64:${LIBRARY_PATH}

WORKDIR /app

COPY --from=builder /workspace/target/release/vision /usr/local/bin/vision
COPY README.md ./
COPY html ./html
COPY images ./images

RUN mkdir -p data models && ldconfig

ENTRYPOINT ["vision"]
