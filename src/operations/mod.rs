//! Operational workflows that consume or persist reports.

pub mod bundle;
pub mod canary;
pub mod compare;
pub mod doctor;
pub mod history;
pub(crate) mod report_input;
pub mod rollout;
pub mod runtime;
pub mod saturation;
pub mod saturation_compare;
pub(crate) mod workload;
