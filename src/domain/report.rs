//! Stable domain model shared by collection, analysis, storage, and APIs.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 3;
pub const MAX_RUNTIME_HISTOGRAM_BUCKETS: usize = 64;
pub const MAX_RUNTIME_HISTOGRAM_SERIES: usize = 32;

/// Availability of one independently collected telemetry source.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SourceState {
    #[default]
    Ok,
    Partial,
    Unavailable,
    Skipped,
}

impl SourceState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Partial => "partial",
            Self::Unavailable => "unavailable",
            Self::Skipped => "skipped",
        }
    }
}

impl std::fmt::Display for SourceState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Evidence describing whether a telemetry source was actually observed.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceStatus {
    pub name: String,
    pub state: SourceState,
    pub duration_ms: u64,
    pub records: u64,
    #[serde(default, skip_serializing_if = "is_false")]
    pub required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl SourceStatus {
    pub fn ok(name: impl Into<String>, duration_ms: u64, records: usize) -> Self {
        Self {
            name: name.into(),
            state: SourceState::Ok,
            duration_ms,
            records: u64::try_from(records).unwrap_or(u64::MAX),
            required: false,
            error: None,
        }
    }

    pub fn failed(
        name: impl Into<String>,
        state: SourceState,
        duration_ms: u64,
        records: usize,
        error: impl Into<String>,
    ) -> Self {
        debug_assert!(state != SourceState::Ok);
        Self {
            name: name.into(),
            state,
            duration_ms,
            records: u64::try_from(records).unwrap_or(u64::MAX),
            required: false,
            error: Some(error.into()),
        }
    }

    pub fn skipped(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            state: SourceState::Skipped,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Gpu {
    pub index: i32,
    pub name: String,
    pub uuid: String,
    pub driver: String,
    pub pci_bus_id: String,
    pub performance_state: String,
    pub temperature_c: i32,
    pub fan_percent: i32,
    pub power_draw_w: f64,
    pub power_limit_w: f64,
    pub graphics_clock_mhz: i32,
    pub memory_clock_mhz: i32,
    pub max_graphics_clock_mhz: i32,
    pub max_memory_clock_mhz: i32,
    pub memory_total_mib: i64,
    pub memory_used_mib: i64,
    pub memory_free_mib: i64,
    pub gpu_util_percent: i32,
    pub memory_util_percent: i32,
    pub pcie_gen_current: i32,
    pub pcie_gen_max: i32,
    pub pcie_width_current: i32,
    pub pcie_width_max: i32,
    pub compute_mode: String,
    pub persistence_mode: bool,
    pub ecc_enabled: bool,
    pub ecc_corrected_volatile: i64,
    pub ecc_uncorrected_volatile: i64,
    pub retired_pages: i64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub mig_mode: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub throttle_reasons: Vec<String>,
    #[serde(default)]
    pub processes: Vec<GpuProcess>,
}

impl Gpu {
    pub fn memory_percent(&self) -> i64 {
        percent(self.memory_used_mib, self.memory_total_mib)
    }

    pub fn is_active(&self) -> bool {
        self.gpu_util_percent > 0 || self.memory_used_mib > 0 || !self.processes.is_empty()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuProcess {
    pub pid: u32,
    pub name: String,
    pub memory_mib: i64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub owner: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub command: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub cgroup: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kubernetes_pod_uid: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Critical,
    Warning,
    #[default]
    Info,
}

impl Severity {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Critical => "critical",
            Self::Warning => "warning",
            Self::Info => "info",
        }
    }

    pub const fn rank(self) -> u8 {
        match self {
            Self::Critical => 0,
            Self::Warning => 1,
            Self::Info => 2,
        }
    }
}

impl std::fmt::Display for Severity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Finding {
    /// `None` denotes a host or inference-runtime finding.
    #[serde(rename = "gpu_index", with = "gpu_index_serde")]
    pub gpu_index: Option<i32>,
    pub severity: Severity,
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub recommendation: String,
}

impl Finding {
    pub fn new(
        gpu_index: Option<i32>,
        severity: Severity,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        let code = code.into();
        Self {
            gpu_index,
            severity,
            recommendation: String::new(),
            code,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RuntimeCounters {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requests_completed_total: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_errors_total: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_total: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_tokens_total: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preemptions_total: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_latency_seconds_sum: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_latency_seconds_count: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_to_first_token_seconds_sum: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_to_first_token_seconds_count: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_per_output_token_seconds_sum: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_per_output_token_seconds_count: Option<f64>,
    #[serde(default, skip_serializing_if = "RuntimeHistograms::is_empty")]
    pub histograms: RuntimeHistograms,
}

impl RuntimeCounters {
    pub fn is_empty(&self) -> bool {
        self.requests_completed_total.is_none()
            && self.request_errors_total.is_none()
            && self.prompt_tokens_total.is_none()
            && self.generation_tokens_total.is_none()
            && self.preemptions_total.is_none()
            && self.request_latency_seconds_sum.is_none()
            && self.request_latency_seconds_count.is_none()
            && self.time_to_first_token_seconds_sum.is_none()
            && self.time_to_first_token_seconds_count.is_none()
            && self.time_per_output_token_seconds_sum.is_none()
            && self.time_per_output_token_seconds_count.is_none()
            && self.histograms.is_empty()
    }
}

/// One normalized cumulative bucket from a classic Prometheus histogram.
///
/// A `None` upper bound represents the required `+Inf` bucket without placing
/// a non-finite float in JSON reports.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RuntimeHistogramBucket {
    pub upper_bound_seconds: Option<f64>,
    pub cumulative_count: f64,
}

#[derive(Clone, Default, PartialEq)]
struct RuntimeHistogramSeriesState {
    series: BTreeMap<String, Vec<RuntimeHistogramBucket>>,
}

impl std::fmt::Debug for RuntimeHistogramSeriesState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeHistogramSeriesState")
            .field("series_count", &self.series.len())
            .field(
                "bucket_count",
                &self.series.values().map(Vec::len).sum::<usize>(),
            )
            .finish()
    }
}

/// A bounded, normalized cumulative histogram snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeHistogram {
    /// Number of compatible source series aggregated into this snapshot.
    /// Live interval derivation also checks bounded private identities, while
    /// persisted snapshots omit those identities and cannot be re-differenced.
    #[serde(default = "default_histogram_series_count")]
    pub series_count: usize,
    #[serde(default)]
    pub buckets: Vec<RuntimeHistogramBucket>,
    /// Bounded live-only state used to detect per-series resets and churn. The
    /// keys may include endpoint labels, so this field is neither serialized
    /// nor exposed by its redacted `Debug` implementation.
    #[serde(skip)]
    private_series: RuntimeHistogramSeriesState,
}

impl PartialEq for RuntimeHistogram {
    fn eq(&self, other: &Self) -> bool {
        self.series_count == other.series_count && self.buckets == other.buckets
    }
}

impl RuntimeHistogram {
    #[cfg(test)]
    pub(crate) fn from_cumulative_buckets(buckets: Vec<RuntimeHistogramBucket>) -> Option<Self> {
        let mut series = BTreeMap::new();
        series.insert(String::new(), buckets);
        Self::from_series(series)
    }

    pub(crate) fn from_series(
        mut series: BTreeMap<String, Vec<RuntimeHistogramBucket>>,
    ) -> Option<Self> {
        if !(1..=MAX_RUNTIME_HISTOGRAM_SERIES).contains(&series.len()) {
            return None;
        }
        for buckets in series.values_mut() {
            buckets.sort_by(compare_runtime_histogram_buckets);
        }
        let buckets = aggregate_histogram_series(&series)?;
        let histogram = Self {
            series_count: series.len(),
            buckets,
            private_series: RuntimeHistogramSeriesState { series },
        };
        histogram.is_valid().then_some(histogram)
    }

    /// Difference two compatible cumulative snapshots and estimate interval
    /// quantiles using the same linear-within-bucket model as classic
    /// Prometheus histograms.
    pub fn interval(&self, after: &Self) -> Option<RuntimeHistogramInterval> {
        if !self.is_valid()
            || !after.is_valid()
            || self.series_count != after.series_count
            || self.buckets.len() != after.buckets.len()
        {
            return None;
        }

        if self.private_series.series.len() != self.series_count
            || after.private_series.series.len() != after.series_count
        {
            // Serialized snapshots deliberately omit private identities, so
            // they cannot safely be differenced even when each has one series.
            return None;
        }
        let mut interval_series = BTreeMap::new();
        for ((before_key, before_buckets), (after_key, after_buckets)) in self
            .private_series
            .series
            .iter()
            .zip(&after.private_series.series)
        {
            if before_key != after_key {
                return None;
            }
            interval_series.insert(
                before_key.clone(),
                difference_histogram_buckets(before_buckets, after_buckets)?,
            );
        }
        let cumulative = aggregate_histogram_series(&interval_series)?;

        let samples = cumulative.last()?.cumulative_count;
        if samples <= 0.0 {
            return None;
        }
        let p50_ms = histogram_quantile_seconds(&cumulative, 0.50)? * 1_000.0;
        let p95_ms = histogram_quantile_seconds(&cumulative, 0.95)? * 1_000.0;
        let p99_ms = histogram_quantile_seconds(&cumulative, 0.99)? * 1_000.0;
        if ![samples, p50_ms, p95_ms, p99_ms]
            .into_iter()
            .all(f64::is_finite)
        {
            return None;
        }
        Some(RuntimeHistogramInterval {
            samples,
            p50_ms,
            p95_ms,
            p99_ms,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    fn is_valid(&self) -> bool {
        if !(1..=MAX_RUNTIME_HISTOGRAM_SERIES).contains(&self.series_count)
            || !valid_cumulative_histogram(&self.buckets)
        {
            return false;
        }
        if self.private_series.series.is_empty() {
            return true;
        }
        self.private_series.series.len() == self.series_count
            && aggregate_histogram_series(&self.private_series.series)
                .is_some_and(|aggregate| aggregate == self.buckets)
    }
}

fn valid_cumulative_histogram(buckets: &[RuntimeHistogramBucket]) -> bool {
    if !(2..=MAX_RUNTIME_HISTOGRAM_BUCKETS).contains(&buckets.len())
        || buckets
            .last()
            .is_none_or(|bucket| bucket.upper_bound_seconds.is_some())
    {
        return false;
    }

    let mut previous_bound = None;
    let mut previous_count = 0.0;
    for (index, bucket) in buckets.iter().enumerate() {
        if !bucket.cumulative_count.is_finite()
            || bucket.cumulative_count < previous_count
            || bucket.cumulative_count < 0.0
        {
            return false;
        }
        previous_count = bucket.cumulative_count;

        match bucket.upper_bound_seconds {
            Some(bound) => {
                if !bound.is_finite()
                    || bound < 0.0
                    || previous_bound.is_some_and(|previous| bound <= previous)
                {
                    return false;
                }
                previous_bound = Some(bound);
            }
            None if index + 1 == buckets.len() && previous_bound.is_some() => {}
            None => return false,
        }
    }
    true
}

fn aggregate_histogram_series(
    series: &BTreeMap<String, Vec<RuntimeHistogramBucket>>,
) -> Option<Vec<RuntimeHistogramBucket>> {
    if !(1..=MAX_RUNTIME_HISTOGRAM_SERIES).contains(&series.len()) {
        return None;
    }
    let mut aggregate: Option<Vec<RuntimeHistogramBucket>> = None;
    for buckets in series.values() {
        if !valid_cumulative_histogram(buckets) {
            return None;
        }
        let Some(current) = aggregate.as_mut() else {
            aggregate = Some(buckets.clone());
            continue;
        };
        if current.len() != buckets.len() {
            return None;
        }
        for (aggregate_bucket, series_bucket) in current.iter_mut().zip(buckets) {
            if aggregate_bucket.upper_bound_seconds != series_bucket.upper_bound_seconds {
                return None;
            }
            aggregate_bucket.cumulative_count += series_bucket.cumulative_count;
            if !aggregate_bucket.cumulative_count.is_finite() {
                return None;
            }
        }
    }
    aggregate.filter(|buckets| valid_cumulative_histogram(buckets))
}

fn difference_histogram_buckets(
    before: &[RuntimeHistogramBucket],
    after: &[RuntimeHistogramBucket],
) -> Option<Vec<RuntimeHistogramBucket>> {
    if !valid_cumulative_histogram(before)
        || !valid_cumulative_histogram(after)
        || before.len() != after.len()
    {
        return None;
    }
    let mut cumulative = Vec::with_capacity(before.len());
    let mut previous_delta = 0.0;
    for (before_bucket, after_bucket) in before.iter().zip(after) {
        if before_bucket.upper_bound_seconds != after_bucket.upper_bound_seconds
            || after_bucket.cumulative_count < before_bucket.cumulative_count
        {
            return None;
        }
        let delta = after_bucket.cumulative_count - before_bucket.cumulative_count;
        if !delta.is_finite() || delta < previous_delta {
            return None;
        }
        previous_delta = delta;
        cumulative.push(RuntimeHistogramBucket {
            upper_bound_seconds: before_bucket.upper_bound_seconds,
            cumulative_count: delta,
        });
    }
    valid_cumulative_histogram(&cumulative).then_some(cumulative)
}

fn compare_runtime_histogram_buckets(
    left: &RuntimeHistogramBucket,
    right: &RuntimeHistogramBucket,
) -> std::cmp::Ordering {
    match (left.upper_bound_seconds, right.upper_bound_seconds) {
        (Some(left), Some(right)) => left.total_cmp(&right),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

const fn default_histogram_series_count() -> usize {
    1
}

fn histogram_quantile_seconds(buckets: &[RuntimeHistogramBucket], quantile: f64) -> Option<f64> {
    let samples = buckets.last()?.cumulative_count;
    let rank = quantile * samples;
    let index = buckets
        .iter()
        .position(|bucket| bucket.cumulative_count >= rank)?;
    let bucket = &buckets[index];
    let Some(upper_bound) = bucket.upper_bound_seconds else {
        // A quantile in the open-ended bucket is conservatively represented by
        // the highest finite boundary, matching Prometheus' classic-histogram
        // behavior rather than emitting infinity.
        return buckets.get(index.checked_sub(1)?)?.upper_bound_seconds;
    };
    let (lower_bound, lower_count) = if index == 0 {
        (0.0, 0.0)
    } else {
        let previous = &buckets[index - 1];
        (previous.upper_bound_seconds?, previous.cumulative_count)
    };
    let bucket_count = bucket.cumulative_count - lower_count;
    if bucket_count <= 0.0 {
        return Some(upper_bound);
    }
    Some(lower_bound + (upper_bound - lower_bound) * (rank - lower_count) / bucket_count)
}

/// The only histogram families retained in a report. Field names and count are
/// fixed; endpoint-controlled metric and label names never enter this model.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RuntimeHistograms {
    #[serde(default, skip_serializing_if = "RuntimeHistogram::is_empty")]
    pub request_latency: RuntimeHistogram,
    #[serde(default, skip_serializing_if = "RuntimeHistogram::is_empty")]
    pub time_to_first_token: RuntimeHistogram,
    #[serde(default, skip_serializing_if = "RuntimeHistogram::is_empty")]
    pub time_per_output_token: RuntimeHistogram,
    #[serde(default, skip_serializing_if = "RuntimeHistogram::is_empty")]
    pub queue_time: RuntimeHistogram,
}

impl RuntimeHistograms {
    pub fn is_empty(&self) -> bool {
        self.request_latency.is_empty()
            && self.time_to_first_token.is_empty()
            && self.time_per_output_token.is_empty()
            && self.queue_time.is_empty()
    }
}

/// Tail-latency estimates derived from requests observed strictly within one
/// collection interval.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RuntimeHistogramInterval {
    pub samples: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RuntimeRates {
    pub interval_seconds: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requests_per_second: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_errors_per_second: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_per_second: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_tokens_per_second: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preemptions_per_second: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_request_latency_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_time_to_first_token_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_time_per_output_token_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_latency: Option<RuntimeHistogramInterval>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_to_first_token: Option<RuntimeHistogramInterval>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_per_output_token: Option<RuntimeHistogramInterval>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_time: Option<RuntimeHistogramInterval>,
}

impl RuntimeRates {
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Endpoint {
    pub url: String,
    pub metrics_url: String,
    pub reachable: bool,
    #[serde(default, skip_serializing_if = "is_zero_u16")]
    pub status_code: u16,
    pub latency_ms: u64,
    pub kind: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub runtime: String,
    pub metric_samples: usize,
    #[serde(default, skip_serializing_if = "is_false")]
    pub metrics_truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requests_running: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requests_waiting: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_cache_usage_percent: Option<f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metrics: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "RuntimeCounters::is_empty")]
    pub counters: RuntimeCounters,
    #[serde(default, skip_serializing_if = "RuntimeRates::is_empty")]
    pub rates: RuntimeRates,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub failure: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Host {
    pub hostname: String,
    pub os: String,
    pub arch: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Summary {
    pub gpus: usize,
    pub active_gpus: usize,
    pub processes: usize,
    pub endpoints: usize,
    pub endpoints_up: usize,
    pub vram_used_mib: i64,
    pub vram_total_mib: i64,
    pub critical_findings: usize,
    pub warning_findings: usize,
    pub info_findings: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Report {
    pub schema_version: u32,
    pub collected_at: DateTime<Utc>,
    pub collection_duration_ms: u64,
    pub host: Host,
    pub status: String,
    pub summary: Summary,
    #[serde(default)]
    pub gpus: Vec<Gpu>,
    #[serde(default)]
    pub findings: Vec<Finding>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub topology: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub xid_events: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub endpoints: Vec<Endpoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<SourceStatus>,
}

impl Default for Report {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            collected_at: Utc::now(),
            collection_duration_ms: 0,
            host: Host::default(),
            status: "unknown".to_owned(),
            summary: Summary::default(),
            gpus: Vec::new(),
            findings: Vec::new(),
            topology: String::new(),
            xid_events: Vec::new(),
            endpoints: Vec::new(),
            sources: Vec::new(),
        }
    }
}

pub fn percent(value: i64, total: i64) -> i64 {
    if total <= 0 {
        return 0;
    }
    value.saturating_mul(100) / total
}

#[allow(clippy::trivially_copy_pass_by_ref)]
const fn is_zero_u16(value: &u16) -> bool {
    *value == 0
}

#[allow(clippy::trivially_copy_pass_by_ref)]
const fn is_false(value: &bool) -> bool {
    !*value
}

mod gpu_index_serde {
    use serde::{Deserialize, Deserializer, Serializer};

    #[allow(clippy::ref_option, clippy::trivially_copy_pass_by_ref)]
    pub fn serialize<S>(value: &Option<i32>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_i32(value.unwrap_or(-1))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<i32>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = i32::deserialize(deserializer)?;
        Ok((value >= 0).then_some(value))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        Finding, MAX_RUNTIME_HISTOGRAM_BUCKETS, Report, RuntimeCounters, RuntimeHistogram,
        RuntimeHistogramBucket, RuntimeRates, SCHEMA_VERSION, Severity, SourceState, SourceStatus,
    };

    fn histogram(buckets: &[(Option<f64>, f64)]) -> RuntimeHistogram {
        RuntimeHistogram::from_cumulative_buckets(
            buckets
                .iter()
                .map(
                    |(upper_bound_seconds, cumulative_count)| RuntimeHistogramBucket {
                        upper_bound_seconds: *upper_bound_seconds,
                        cumulative_count: *cumulative_count,
                    },
                )
                .collect(),
        )
        .unwrap()
    }

    fn multi_histogram(series: &[(&str, [f64; 3])]) -> RuntimeHistogram {
        RuntimeHistogram::from_series(
            series
                .iter()
                .map(|(identity, counts)| {
                    (
                        (*identity).to_owned(),
                        vec![
                            RuntimeHistogramBucket {
                                upper_bound_seconds: Some(0.1),
                                cumulative_count: counts[0],
                            },
                            RuntimeHistogramBucket {
                                upper_bound_seconds: Some(1.0),
                                cumulative_count: counts[1],
                            },
                            RuntimeHistogramBucket {
                                upper_bound_seconds: None,
                                cumulative_count: counts[2],
                            },
                        ],
                    )
                })
                .collect::<BTreeMap<_, _>>(),
        )
        .unwrap()
    }

    #[test]
    fn host_findings_use_the_external_minus_one_gpu_index() {
        let finding = Finding::new(None, Severity::Warning, "HOST_TEST", "host issue");
        let encoded = serde_json::to_value(&finding).unwrap();

        assert_eq!(encoded["gpu_index"], -1);

        let decoded: Finding = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.gpu_index, None);
    }

    #[test]
    fn gpu_findings_round_trip_their_index() {
        let finding = Finding::new(Some(3), Severity::Critical, "GPU_TEST", "gpu issue");
        let encoded = serde_json::to_value(&finding).unwrap();

        assert_eq!(encoded["gpu_index"], 3);

        let decoded: Finding = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.gpu_index, Some(3));
    }

    #[test]
    fn old_reports_without_source_evidence_remain_readable() {
        let report: Report = serde_json::from_str(
            r#"{"schema_version":2,"status":"healthy","gpus":[],"findings":[]}"#,
        )
        .unwrap();

        assert!(report.sources.is_empty());
        assert_eq!(report.schema_version, 2);
        assert_eq!(Report::default().schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn source_state_uses_stable_machine_names() {
        let source = SourceStatus::failed(
            "nvidia.processes",
            SourceState::Partial,
            12,
            3,
            "graphics query failed",
        );
        let encoded = serde_json::to_value(&source).unwrap();

        assert_eq!(encoded["state"], "partial");
        assert_eq!(encoded["records"], 3);
        assert_eq!(encoded["error"], "graphics query failed");
        assert!(encoded.get("required").is_none());
    }

    #[test]
    fn derives_bounded_interval_quantiles_from_cumulative_histograms() {
        let before = histogram(&[
            (Some(0.1), 10.0),
            (Some(0.5), 50.0),
            (Some(1.0), 90.0),
            (None, 100.0),
        ]);
        let after = histogram(&[
            (Some(0.1), 20.0),
            (Some(0.5), 80.0),
            (Some(1.0), 140.0),
            (None, 160.0),
        ]);

        let interval = before.interval(&after).unwrap();
        assert!((interval.samples - 60.0).abs() < f64::EPSILON);
        assert!((interval.p50_ms - 500.0).abs() < f64::EPSILON);
        assert!((interval.p95_ms - 1_000.0).abs() < f64::EPSILON);
        assert!((interval.p99_ms - 1_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn rejects_resets_boundary_changes_and_nonmonotonic_interval_deltas() {
        let before = histogram(&[(Some(0.1), 5.0), (Some(1.0), 10.0), (None, 12.0)]);
        let reset = histogram(&[(Some(0.1), 1.0), (Some(1.0), 2.0), (None, 3.0)]);
        assert!(before.interval(&reset).is_none());

        let changed = histogram(&[(Some(0.2), 6.0), (Some(1.0), 12.0), (None, 15.0)]);
        assert!(before.interval(&changed).is_none());

        let mut series_churn = histogram(&[(Some(0.1), 6.0), (Some(1.0), 12.0), (None, 15.0)]);
        series_churn.series_count = 2;
        assert!(before.interval(&series_churn).is_none());

        let inconsistent = histogram(&[(Some(0.1), 9.0), (Some(1.0), 12.0), (None, 14.0)]);
        assert!(before.interval(&inconsistent).is_none());
    }

    #[test]
    fn rejects_masked_per_series_resets_and_equal_count_replacement_churn() {
        let before = multi_histogram(&[
            ("private-a", [5.0, 10.0, 10.0]),
            ("private-b", [5.0, 10.0, 10.0]),
        ]);
        let masked_reset = multi_histogram(&[
            ("private-a", [1.0, 2.0, 2.0]),
            ("private-b", [20.0, 30.0, 30.0]),
        ]);
        assert!(
            masked_reset.buckets.last().unwrap().cumulative_count
                > before.buckets.last().unwrap().cumulative_count
        );
        assert!(before.interval(&masked_reset).is_none());

        let replacement = multi_histogram(&[
            ("private-a", [8.0, 15.0, 15.0]),
            ("private-c", [8.0, 15.0, 15.0]),
        ]);
        assert_eq!(before.series_count, replacement.series_count);
        assert!(before.interval(&replacement).is_none());
        let debug = format!("{before:?}");
        assert!(!debug.contains("private-a"));
        assert!(!debug.contains("private-b"));
    }

    #[test]
    fn deserialized_snapshots_fail_closed_without_private_identity() {
        let before = multi_histogram(&[
            ("private-a", [1.0, 2.0, 2.0]),
            ("private-b", [1.0, 2.0, 2.0]),
        ]);
        let after = multi_histogram(&[
            ("private-a", [2.0, 4.0, 4.0]),
            ("private-b", [2.0, 4.0, 4.0]),
        ]);
        assert!(before.interval(&after).is_some());

        let before: RuntimeHistogram =
            serde_json::from_str(&serde_json::to_string(&before).unwrap()).unwrap();
        let after: RuntimeHistogram =
            serde_json::from_str(&serde_json::to_string(&after).unwrap()).unwrap();
        assert!(before.interval(&after).is_none());

        let before = histogram(&[(Some(0.1), 1.0), (None, 2.0)]);
        let after = histogram(&[(Some(0.1), 2.0), (None, 4.0)]);
        let before: RuntimeHistogram =
            serde_json::from_str(&serde_json::to_string(&before).unwrap()).unwrap();
        let after: RuntimeHistogram =
            serde_json::from_str(&serde_json::to_string(&after).unwrap()).unwrap();
        assert!(before.interval(&after).is_none());
    }

    #[test]
    fn rejects_invalid_or_oversized_cumulative_histograms() {
        assert!(
            RuntimeHistogram::from_cumulative_buckets(vec![RuntimeHistogramBucket {
                upper_bound_seconds: Some(1.0),
                cumulative_count: 1.0,
            }])
            .is_none()
        );
        assert!(
            RuntimeHistogram::from_cumulative_buckets(vec![
                RuntimeHistogramBucket {
                    upper_bound_seconds: Some(0.1),
                    cumulative_count: 2.0,
                },
                RuntimeHistogramBucket {
                    upper_bound_seconds: Some(1.0),
                    cumulative_count: 1.0,
                },
                RuntimeHistogramBucket {
                    upper_bound_seconds: None,
                    cumulative_count: 2.0,
                },
            ])
            .is_none()
        );
        let oversized = (0_u32..=u32::try_from(MAX_RUNTIME_HISTOGRAM_BUCKETS).unwrap())
            .map(|index| RuntimeHistogramBucket {
                upper_bound_seconds: (usize::try_from(index).unwrap()
                    < MAX_RUNTIME_HISTOGRAM_BUCKETS)
                    .then_some(f64::from(index)),
                cumulative_count: f64::from(index),
            })
            .collect();
        assert!(RuntimeHistogram::from_cumulative_buckets(oversized).is_none());
    }

    #[test]
    fn older_counter_and_rate_payloads_default_new_histogram_fields() {
        let counters: RuntimeCounters = serde_json::from_str("{}").unwrap();
        let rates: RuntimeRates =
            serde_json::from_str(r#"{"interval_seconds":2.0,"mean_request_latency_ms":12.0}"#)
                .unwrap();
        let legacy_histogram: RuntimeHistogram = serde_json::from_str(
            r#"{"buckets":[{"upper_bound_seconds":0.1,"cumulative_count":1.0},{"upper_bound_seconds":null,"cumulative_count":1.0}]}"#,
        )
        .unwrap();

        assert!(counters.histograms.is_empty());
        assert!(rates.request_latency.is_none());
        assert!(rates.time_to_first_token.is_none());
        assert!(rates.time_per_output_token.is_none());
        assert!(rates.queue_time.is_none());
        assert_eq!(rates.mean_request_latency_ms, Some(12.0));
        assert_eq!(legacy_histogram.series_count, 1);
    }
}
