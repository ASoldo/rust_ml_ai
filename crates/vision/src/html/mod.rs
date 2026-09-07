//! Embedded static HTML assets served by the preview web UI.
//!
//! The assets are kept as `&'static str` so they can be bundled directly inside
//! the binary without filesystem lookups.

pub mod atak;
pub mod hud_html;

/// The operator console has one source asset, embedded in the executable.
pub const OPERATOR_HTML: &str = include_str!("../../../../html/operator.html");
