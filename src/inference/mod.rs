//! Inference-runtime discovery, probing, and metric normalization.

mod openai;
pub mod probe;

pub(crate) use openai::{
    MAX_REPORTED_PROMPT_TOKENS_PER_REQUEST, OpenAiClient, OpenAiClientOptions,
    request_body_memory_budget, worker_memory_budget,
};
pub use probe::{ProbeOptions, collect, parse_metrics, resolve_metrics_url};
