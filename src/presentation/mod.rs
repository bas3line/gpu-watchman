//! Terminal, JSON, HTTP, and Prometheus presentation adapters.

mod benchmark;
mod benchmark_compare;
mod canary;
mod processes;
mod prometheus;
mod runtime;
pub mod server;
pub mod terminal;
mod text;

pub use benchmark::render as render_saturation_benchmark;
pub use benchmark_compare::render as render_saturation_comparison;
pub use canary::render as render_canary;
pub use processes::{PROCESS_VIEW_VERSION, render as render_process_view};
pub use prometheus::encode as prometheus;
pub use runtime::render as render_runtime_fingerprint;
pub use server::Exporter;
pub use terminal::{OutputFormat, colorize, render, render_processes};
pub(crate) use text::{safe_inline, safe_multiline};
