pub use config::{SourceKind, VisionConfig};
pub use pipeline::run_from_args;

mod annotation;
mod config;
mod data;
mod encoding;
mod pipeline;
mod processing;
mod runtime;
mod server;
mod watchdog;
