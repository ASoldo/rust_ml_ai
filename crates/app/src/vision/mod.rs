//! End-to-end vision pipeline that captures frames, runs ML inference, and
//! exposes annotated previews over HTTP.
//!
//! The module is split into focused submodules:
//! - `config`: CLI configuration parsing.
//! - `pipeline`: Orchestrates the capture → process → encode loop.
//! - `processing`: Detector workers and CPU/GPU annotation hand-off.
//! - `encoding`: JPEG encode handling for CPU/GPU paths.
//! - `server`: Actix Web preview endpoints.
//! - `watchdog`: Health monitoring for pipeline components.
//! - `runtime`: CUDA runtime loader glue.
//! - `data`: Shared structs passed between stages.
//! - `annotation`: Drawing primitives shared by CPU/GPU encoders.

/// Re-export pipeline settings so callers can configure runs without reaching
/// into submodules.
pub use config::{SourceKind, VisionCliArgs, VisionConfig};
/// Launch the vision pipeline with a ready-made configuration.
pub use pipeline::run;

mod annotation;
mod config;
mod data;
mod encoding;
mod pipeline;
mod processing;
mod runtime;
mod server;
mod telemetry;
mod watchdog;
