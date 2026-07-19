//! Stateless health rules and stateful trend analysis.

pub mod health;
pub mod trend;

pub use health::{
    AnalyzerConfig, analyze_endpoints, analyze_gpus, analyze_sources, finalize, recommendation,
};
pub use trend::Tracker;
