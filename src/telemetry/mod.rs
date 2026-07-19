//! Hardware and operating-system telemetry adapters.

mod command;
mod host;
pub mod nvidia;
mod process;
pub(crate) mod runtime;

pub use host::local as local_host;
pub use nvidia::NvidiaCollector;
