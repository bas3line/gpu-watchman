//! Strict, versioned operational-profile schema.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// The only configuration schema understood by this release.
pub const CONFIG_VERSION: u32 = 1;

/// A duration encoded as a human-readable TOML string such as `500ms` or `5s`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct HumanDuration(pub Duration);

impl fmt::Display for HumanDuration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        humantime::format_duration(self.0).fmt(formatter)
    }
}

impl From<Duration> for HumanDuration {
    fn from(value: Duration) -> Self {
        Self(value)
    }
}

impl From<HumanDuration> for Duration {
    fn from(value: HumanDuration) -> Self {
        value.0
    }
}

impl FromStr for HumanDuration {
    type Err = humantime::DurationError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        humantime::parse_duration(value).map(Self)
    }
}

impl Serialize for HumanDuration {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for HumanDuration {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

/// Finding severity that makes a monitoring command unhealthy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProfileFailOn {
    Never,
    Warning,
    Critical,
}

/// Complete version-1 configuration document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigV1 {
    pub config_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_profile: Option<String>,
    #[serde(default)]
    pub profiles: BTreeMap<String, ProfileV1>,
}

impl Default for ConfigV1 {
    fn default() -> Self {
        Self {
            config_version: CONFIG_VERSION,
            default_profile: None,
            profiles: BTreeMap::new(),
        }
    }
}

/// Command-scoped settings in one named operational profile.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProfileV1 {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub monitor: Option<MonitorProfileV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service: Option<ServiceProfileV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canary: Option<CanaryProfileV1>,
}

/// Settings shared by snapshot, top, process, and service collection.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MonitorProfileV1 {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collection: Option<MonitorCollectionProfileV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference: Option<MonitorInferenceProfileV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub health: Option<MonitorHealthProfileV1>,
}

/// NVIDIA collection inputs and limits.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MonitorCollectionProfileV1 {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command_timeout: Option<HumanDuration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpus: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvidia_smi: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collect_xid: Option<bool>,
}

/// Passive inference-runtime metrics probes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MonitorInferenceProfileV1 {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub urls: Option<Vec<String>>,
    /// File reference only; inline bearer credentials are intentionally absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_file: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<HumanDuration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_insecure_http: Option<bool>,
}

/// Monitoring findings and exit-policy thresholds.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MonitorHealthProfileV1 {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fail_on: Option<ProfileFailOn>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_sources: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_warning_percent: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_critical_percent: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature_warning_c: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature_critical_c: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_cache_warning_percent: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_cache_critical_percent: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub process_growth_warning_mib: Option<i64>,
}

/// Settings used only by the explicit `serve` workflow.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServiceProfileV1 {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub listen: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interval: Option<HumanDuration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history_file: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub freshness: Option<HumanDuration>,
    /// File reference only; inline HTTP API credentials are intentionally absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_token_file: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quiet: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_remote_listen: Option<bool>,
}

/// Bounded OpenAI-compatible active inference validation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CanaryProfileV1 {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// File reference only; inline API keys are intentionally absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_file: Option<PathBuf>,
    /// Synthetic prompts are referenced rather than embedded in configuration output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_file: Option<PathBuf>,
    /// Non-secret identity required when `prompt_file` selects a custom workload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workload_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concurrency: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<HumanDuration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_insecure_http: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slo: Option<CanarySloProfileV1>,
}

/// Canary correctness, availability, latency, and throughput gates.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CanarySloProfileV1 {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expect: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_ttft: Option<HumanDuration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_e2e: Option<HumanDuration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_output_tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_success_percent: Option<f64>,
}
