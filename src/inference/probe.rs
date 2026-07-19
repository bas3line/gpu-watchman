//! Bounded, authenticated inference-runtime metrics probing and normalization.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Read as _;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use reqwest::blocking::Client;
use reqwest::header::{ACCEPT, AUTHORIZATION, HeaderMap, HeaderValue, USER_AGENT};
use reqwest::redirect::Policy;
use url::Url;

use crate::domain::{
    Endpoint, MAX_RUNTIME_HISTOGRAM_BUCKETS, MAX_RUNTIME_HISTOGRAM_SERIES, RuntimeHistogram,
    RuntimeHistogramBucket, RuntimeHistograms,
};
use crate::security::url_is_loopback;

pub const MAX_PROBE_TARGETS: usize = 32;
const MAX_PROBE_BODY_BYTES: usize = 8 << 20;
const MAX_PROBE_CONCURRENCY: usize = 8;
const MAX_PROBE_BODY_BUDGET_BYTES: usize = 32 << 20;
const MAX_PROCESSED_METRIC_FAMILY_BYTES: usize = 128 << 10;
const MAX_METRIC_FAMILY_BYTES: usize = 256;
const MAX_HISTOGRAM_LABELS_PER_SAMPLE: usize = 32;
const MAX_HISTOGRAM_SAMPLE_KEY_BYTES: usize = 2 << 10;
const MAX_HISTOGRAM_SERIES_KEY_BYTES: usize = 1 << 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HistogramKind {
    RequestLatency,
    TimeToFirstToken,
    TimePerOutputToken,
    QueueTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HistogramRuntime {
    Vllm,
    Tgi,
    Triton,
    Ollama,
    Sglang,
    TensorRtLlm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HistogramPrefix {
    TensorRtLlm,
    NvInference,
    TritonColon,
    TritonUnderscore,
    Trtllm,
    OllamaColon,
    OllamaUnderscore,
    SglangColon,
    SglangUnderscore,
    VllmColon,
    VllmUnderscore,
    TgiColon,
    TgiUnderscore,
}

impl HistogramPrefix {
    const fn discriminator(self) -> char {
        match self {
            Self::TensorRtLlm => 'a',
            Self::NvInference => 'b',
            Self::TritonColon => 'c',
            Self::TritonUnderscore => 'd',
            Self::Trtllm => 'e',
            Self::OllamaColon => 'f',
            Self::OllamaUnderscore => 'g',
            Self::SglangColon => 'h',
            Self::SglangUnderscore => 'i',
            Self::VllmColon => 'j',
            Self::VllmUnderscore => 'k',
            Self::TgiColon => 'l',
            Self::TgiUnderscore => 'm',
        }
    }
}

impl HistogramRuntime {
    const fn discriminator(self) -> char {
        match self {
            Self::Vllm => 'v',
            Self::Tgi => 'g',
            Self::Triton => 'r',
            Self::Ollama => 'o',
            Self::Sglang => 's',
            Self::TensorRtLlm => 't',
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HistogramSchema {
    E2eRequestLatency,
    RequestDuration,
    RequestLatency,
    TimeToFirstToken,
    TimeToFirstTokenDuration,
    Ttft,
    TimePerOutputToken,
    InterTokenLatency,
    InterTokenDuration,
    Tpot,
    RequestQueueTime,
    QueueTime,
    RequestQueueDuration,
    WaitingTime,
}

impl HistogramSchema {
    const fn discriminator(self) -> char {
        match self {
            Self::E2eRequestLatency => 'a',
            Self::RequestDuration => 'b',
            Self::RequestLatency => 'c',
            Self::TimeToFirstToken => 'd',
            Self::TimeToFirstTokenDuration => 'e',
            Self::Ttft => 'f',
            Self::TimePerOutputToken => 'g',
            Self::InterTokenLatency => 'h',
            Self::InterTokenDuration => 'i',
            Self::Tpot => 'j',
            Self::RequestQueueTime => 'k',
            Self::QueueTime => 'l',
            Self::RequestQueueDuration => 'm',
            Self::WaitingTime => 'n',
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RecognizedHistogram {
    kind: HistogramKind,
    schema: HistogramSchema,
    runtime: HistogramRuntime,
    prefix: HistogramPrefix,
}

#[derive(Debug, Default)]
struct HistogramAccumulator {
    schema: Option<HistogramSchema>,
    series: BTreeMap<String, Vec<RuntimeHistogramBucket>>,
    invalid: bool,
}

impl HistogramAccumulator {
    /// Returns `true` only when a resource limit rejected the sample.
    fn observe(
        &mut self,
        schema: HistogramSchema,
        runtime: HistogramRuntime,
        prefix: HistogramPrefix,
        key: &str,
        value: f64,
    ) -> bool {
        if self.invalid {
            return false;
        }
        if self.schema.is_some_and(|current| current != schema) {
            self.invalid = true;
            return false;
        }
        self.schema = Some(schema);
        if value < 0.0 {
            self.invalid = true;
            return false;
        }
        let (label_key, upper_bound) = match parse_histogram_sample_key(key) {
            Ok(sample) => sample,
            Err(HistogramSampleError::Invalid) => {
                self.invalid = true;
                return false;
            }
            Err(HistogramSampleError::LimitExceeded) => {
                self.invalid = true;
                return true;
            }
        };
        let mut series_key = String::with_capacity(label_key.len().saturating_add(5));
        series_key.push(runtime.discriminator());
        series_key.push(prefix.discriminator());
        series_key.push(schema.discriminator());
        series_key.push(':');
        series_key.push_str(&label_key);

        if !self.series.contains_key(&series_key)
            && self.series.len() >= MAX_RUNTIME_HISTOGRAM_SERIES
        {
            self.invalid = true;
            return true;
        }
        let buckets = self.series.entry(series_key).or_default();
        if buckets
            .iter()
            .any(|bucket| bucket.upper_bound_seconds == upper_bound)
        {
            self.invalid = true;
            return false;
        }
        if buckets.len() >= MAX_RUNTIME_HISTOGRAM_BUCKETS {
            self.invalid = true;
            return true;
        }
        buckets.push(RuntimeHistogramBucket {
            upper_bound_seconds: upper_bound,
            cumulative_count: value,
        });
        false
    }

    fn invalidate(&mut self) {
        self.invalid = true;
    }

    fn finish(self) -> RuntimeHistogram {
        self.try_finish().unwrap_or_default()
    }

    fn try_finish(self) -> Option<RuntimeHistogram> {
        if self.invalid || self.schema.is_none() || self.series.is_empty() {
            return None;
        }
        RuntimeHistogram::from_series(self.series)
    }
}

#[derive(Debug, Default)]
struct HistogramCollector {
    request_latency: HistogramAccumulator,
    time_to_first_token: HistogramAccumulator,
    time_per_output_token: HistogramAccumulator,
    queue_time: HistogramAccumulator,
}

impl HistogramCollector {
    fn accumulator_mut(&mut self, kind: HistogramKind) -> &mut HistogramAccumulator {
        match kind {
            HistogramKind::RequestLatency => &mut self.request_latency,
            HistogramKind::TimeToFirstToken => &mut self.time_to_first_token,
            HistogramKind::TimePerOutputToken => &mut self.time_per_output_token,
            HistogramKind::QueueTime => &mut self.queue_time,
        }
    }

    fn observe(&mut self, histogram: RecognizedHistogram, key: &str, value: f64) -> bool {
        self.accumulator_mut(histogram.kind).observe(
            histogram.schema,
            histogram.runtime,
            histogram.prefix,
            key,
            value,
        )
    }

    fn invalidate(&mut self, kind: HistogramKind) {
        self.accumulator_mut(kind).invalidate();
    }

    fn finish(self) -> RuntimeHistograms {
        RuntimeHistograms {
            request_latency: self.request_latency.finish(),
            time_to_first_token: self.time_to_first_token.finish(),
            time_per_output_token: self.time_per_output_token.finish(),
            queue_time: self.queue_time.finish(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HistogramSampleError {
    Invalid,
    LimitExceeded,
}

#[derive(Clone)]
pub struct ProbeOptions {
    pub timeout: Duration,
    pub max_body_bytes: usize,
    pub max_samples: usize,
    pub max_concurrency: usize,
    pub max_targets: usize,
    pub bearer_token: Option<String>,
    pub allow_insecure_http: bool,
}

impl std::fmt::Debug for ProbeOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProbeOptions")
            .field("timeout", &self.timeout)
            .field("max_body_bytes", &self.max_body_bytes)
            .field("max_samples", &self.max_samples)
            .field("max_concurrency", &self.max_concurrency)
            .field("max_targets", &self.max_targets)
            .field("bearer_token_configured", &self.bearer_token.is_some())
            .field("allow_insecure_http", &self.allow_insecure_http)
            .finish()
    }
}

impl Default for ProbeOptions {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(3),
            max_body_bytes: 4 << 20,
            max_samples: 10_000,
            max_concurrency: 4,
            max_targets: MAX_PROBE_TARGETS,
            bearer_token: None,
            allow_insecure_http: false,
        }
    }
}

pub fn collect(targets: &[String], options: &ProbeOptions) -> Vec<Endpoint> {
    let targets = targets
        .iter()
        .map(|target| target.trim())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if targets.is_empty() {
        return Vec::new();
    }
    if let Err(error) = validate_probe_options(options) {
        return failed_probe_set(error);
    }
    if targets.len() > options.max_targets {
        return failed_probe_set(format!(
            "probe target count {} exceeds the configured limit of {}",
            targets.len(),
            options.max_targets
        ));
    }
    if options.bearer_token.is_some()
        && let Err(error) = validate_shared_bearer_origin(&targets)
    {
        return failed_probe_set(error);
    }

    let mut headers = HeaderMap::new();
    headers.insert(ACCEPT, HeaderValue::from_static("text/plain"));
    headers.insert(USER_AGENT, HeaderValue::from_static("gpu-watchman"));
    if let Some(token) = options.bearer_token.as_deref() {
        let Ok(value) = HeaderValue::from_str(&format!("Bearer {token}")) else {
            return targets
                .into_iter()
                .map(|target| {
                    failed_endpoint(
                        &target,
                        &target,
                        "probe bearer token contains invalid HTTP header characters",
                    )
                })
                .collect();
        };
        headers.insert(AUTHORIZATION, value);
    }
    let Ok(client) = Client::builder()
        .timeout(options.timeout)
        .connect_timeout(options.timeout)
        .redirect(Policy::none())
        .no_proxy()
        .default_headers(headers)
        .build()
    else {
        return targets
            .into_iter()
            .map(|target| {
                failed_endpoint(&target, &target, "could not build inference probe client")
            })
            .collect();
    };

    let results = Arc::new(Mutex::new(vec![None; targets.len()]));
    let concurrency = options.max_concurrency.max(1);
    for batch in (0..targets.len()).collect::<Vec<_>>().chunks(concurrency) {
        std::thread::scope(|scope| {
            for &index in batch {
                let client = client.clone();
                let target = targets[index].clone();
                let results = Arc::clone(&results);
                scope.spawn(move || {
                    let endpoint = collect_one(&client, &target, options);
                    if let Ok(mut guard) = results.lock() {
                        guard[index] = Some(endpoint);
                    }
                });
            }
        });
    }

    Arc::try_unwrap(results)
        .ok()
        .and_then(|results| results.into_inner().ok())
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .map(|(index, result)| {
            result.unwrap_or_else(|| {
                failed_endpoint(&targets[index], &targets[index], "probe worker failed")
            })
        })
        .collect()
}

fn collect_one(client: &Client, target: &str, options: &ProbeOptions) -> Endpoint {
    let metrics_url = match resolve_metrics_url(target) {
        Ok(url) => url,
        Err(error) => return failed_endpoint(target, target, error),
    };
    if let Err(error) = validate_transport_security(&metrics_url, options.allow_insecure_http) {
        return failed_endpoint(target, metrics_url.as_str(), error);
    }
    let safe_target = redact_url(target);
    let safe_metrics_url = redact_url(metrics_url.as_str());
    let mut endpoint = Endpoint {
        url: safe_target,
        metrics_url: safe_metrics_url,
        kind: "prometheus".to_owned(),
        ..Endpoint::default()
    };
    let started = Instant::now();
    let response = match client.get(metrics_url).send() {
        Ok(response) => response,
        Err(error) => {
            endpoint.latency_ms = elapsed_millis(started);
            let message = if error.is_timeout() {
                "inference probe timed out"
            } else if error.is_connect() {
                "inference probe connection failed"
            } else {
                "inference probe request failed"
            };
            message.clone_into(&mut endpoint.failure);
            return endpoint;
        }
    };
    endpoint.latency_ms = elapsed_millis(started);
    endpoint.status_code = response.status().as_u16();
    if !response.status().is_success() {
        endpoint.failure = response.status().to_string();
        return endpoint;
    }
    let limit = u64::try_from(options.max_body_bytes.saturating_add(1)).unwrap_or(u64::MAX);
    let body = match std::io::read_to_string(response.take(limit)) {
        Ok(body) => body,
        Err(error) => {
            endpoint.failure = format!("read metrics: {error}");
            return endpoint;
        }
    };
    endpoint.latency_ms = elapsed_millis(started);
    if body.len() > options.max_body_bytes {
        endpoint.failure = format!("metrics response exceeds {} bytes", options.max_body_bytes);
        return endpoint;
    }
    endpoint.reachable = true;
    parse_metrics_bounded(
        &mut endpoint,
        &body,
        options.max_samples,
        MAX_PROCESSED_METRIC_FAMILY_BYTES,
    );
    endpoint
}

fn validate_transport_security(url: &Url, allow_insecure_http: bool) -> Result<(), &'static str> {
    if url.scheme() == "http" && !url_is_loopback(url) && !allow_insecure_http {
        return Err("remote cleartext HTTP probe requires explicit insecure-transport opt-in");
    }
    Ok(())
}

pub fn resolve_metrics_url(target: &str) -> Result<Url, String> {
    let mut url =
        Url::parse(target.trim()).map_err(|error| format!("invalid endpoint URL: {error}"))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err("endpoint URL must use http or https".to_owned());
    }
    if url.host_str().is_none() {
        return Err("endpoint URL must include a host".to_owned());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("endpoint URL must not include user information".to_owned());
    }
    if url.path().is_empty() || url.path() == "/" {
        url.set_path("/metrics");
    }
    Ok(url)
}

pub fn parse_metrics(endpoint: &mut Endpoint, body: &str, max_samples: usize) {
    parse_metrics_bounded(
        endpoint,
        body,
        max_samples,
        MAX_PROCESSED_METRIC_FAMILY_BYTES,
    );
}

fn parse_metrics_bounded(
    endpoint: &mut Endpoint,
    body: &str,
    max_samples: usize,
    max_processed_family_bytes: usize,
) {
    let mut processed_family_bytes = 0_usize;
    let mut histogram_collector = HistogramCollector::default();
    let mut complete_scan = true;
    for line in body.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, raw_value)) = split_metric_sample(line) else {
            continue;
        };
        let family = metric_family(key);
        if !valid_metric_family(family) {
            continue;
        }
        let runtime = runtime_for_metric(family);
        if runtime.is_empty() {
            continue;
        }
        let recognized_histogram = recognize_histogram(family);
        let Ok(value) = raw_value.parse::<f64>() else {
            if let Some(histogram) = recognized_histogram {
                histogram_collector.invalidate(histogram.kind);
            }
            continue;
        };
        if !value.is_finite() {
            if let Some(histogram) = recognized_histogram {
                histogram_collector.invalidate(histogram.kind);
            }
            continue;
        }
        if endpoint.metric_samples >= max_samples {
            endpoint.metrics_truncated = true;
            complete_scan = false;
            break;
        }
        let Some(next_processed_family_bytes) = processed_family_bytes.checked_add(family.len())
        else {
            endpoint.metrics_truncated = true;
            complete_scan = false;
            break;
        };
        if next_processed_family_bytes > max_processed_family_bytes {
            endpoint.metrics_truncated = true;
            complete_scan = false;
            break;
        }
        processed_family_bytes = next_processed_family_bytes;
        endpoint.metric_samples += 1;
        add_runtime(&mut endpoint.runtime, runtime);
        if recognized_histogram
            .is_some_and(|histogram| histogram_collector.observe(histogram, key, value))
        {
            endpoint.metrics_truncated = true;
        }
        normalize(endpoint, family, value);
    }
    endpoint.counters.histograms = if complete_scan {
        histogram_collector.finish()
    } else {
        RuntimeHistograms::default()
    };
    if endpoint.metric_samples == 0 {
        "generic".clone_into(&mut endpoint.kind);
    } else {
        endpoint.kind.clone_from(&endpoint.runtime);
        // Endpoint-controlled names and labels are never serialized. A hostile
        // runtime could otherwise reflect credentials or tenant data through a
        // syntactically valid, runtime-prefixed metric family.
        endpoint.metrics.clear();
    }
}

fn valid_metric_family(family: &str) -> bool {
    if family.is_empty() || family.len() > MAX_METRIC_FAMILY_BYTES {
        return false;
    }
    let mut bytes = family.bytes();
    bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || matches!(byte, b'_' | b':'))
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b':'))
}

fn validate_probe_options(options: &ProbeOptions) -> Result<(), String> {
    if options.max_body_bytes == 0 || options.max_body_bytes > MAX_PROBE_BODY_BYTES {
        return Err(format!(
            "probe body limit must be between 1 byte and {MAX_PROBE_BODY_BYTES} bytes"
        ));
    }
    if options.max_samples == 0 {
        return Err("probe sample limit must be positive".to_owned());
    }
    if options.max_concurrency == 0 || options.max_concurrency > MAX_PROBE_CONCURRENCY {
        return Err(format!(
            "probe concurrency must be between 1 and {MAX_PROBE_CONCURRENCY}"
        ));
    }
    if options
        .max_body_bytes
        .checked_mul(options.max_concurrency)
        .is_none_or(|budget| budget > MAX_PROBE_BODY_BUDGET_BYTES)
    {
        return Err(format!(
            "probe concurrency multiplied by body limit must not exceed {MAX_PROBE_BODY_BUDGET_BYTES} bytes"
        ));
    }
    if options.max_targets == 0 || options.max_targets > MAX_PROBE_TARGETS {
        return Err(format!(
            "probe target limit must be between 1 and {MAX_PROBE_TARGETS}"
        ));
    }
    Ok(())
}

/// A legacy probe token is installed as a client-wide default header, so it
/// may only be used for targets that resolve to one exact URL origin. This
/// validation happens before the client is built or any request is sent.
fn validate_shared_bearer_origin(targets: &[String]) -> Result<(), String> {
    let mut expected_origin = None;
    for target in targets {
        let url = resolve_metrics_url(target).map_err(|_| {
            "a shared probe bearer token requires every target to have a valid HTTP(S) origin"
                .to_owned()
        })?;
        let origin = url.origin().ascii_serialization();
        if expected_origin
            .as_ref()
            .is_some_and(|expected| expected != &origin)
        {
            return Err(
                "a shared probe bearer token cannot be sent to multiple URL origins; run one authenticated origin per invocation"
                    .to_owned(),
            );
        }
        expected_origin = Some(origin);
    }
    Ok(())
}

fn failed_probe_set(failure: impl Into<String>) -> Vec<Endpoint> {
    vec![Endpoint {
        url: "<probe-set>".to_owned(),
        metrics_url: "<probe-set>".to_owned(),
        kind: "prometheus".to_owned(),
        failure: failure.into(),
        ..Endpoint::default()
    }]
}

fn metric_family(key: &str) -> &str {
    key.split_once('{').map_or(key, |(family, _)| family)
}

fn split_metric_sample(line: &str) -> Option<(&str, &str)> {
    let mut quoted = false;
    let mut escaped = false;
    let mut separator = None;
    for (index, byte) in line.bytes().enumerate() {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
        } else if byte == b'"' {
            quoted = true;
        } else if byte.is_ascii_whitespace() {
            separator = Some(index);
            break;
        }
    }
    let separator = separator?;
    let key = &line[..separator];
    let remainder = line[separator..].trim_start();
    let value_end = remainder
        .bytes()
        .position(|byte| byte.is_ascii_whitespace())
        .unwrap_or(remainder.len());
    let raw_value = &remainder[..value_end];
    (!key.is_empty() && !raw_value.is_empty()).then_some((key, raw_value))
}

fn recognize_histogram_prefix(family: &str) -> Option<(HistogramRuntime, HistogramPrefix, &str)> {
    [
        (
            "tensorrt_llm_",
            HistogramRuntime::TensorRtLlm,
            HistogramPrefix::TensorRtLlm,
        ),
        (
            "nv_inference_",
            HistogramRuntime::Triton,
            HistogramPrefix::NvInference,
        ),
        (
            "triton:",
            HistogramRuntime::Triton,
            HistogramPrefix::TritonColon,
        ),
        (
            "triton_",
            HistogramRuntime::Triton,
            HistogramPrefix::TritonUnderscore,
        ),
        (
            "trtllm_",
            HistogramRuntime::TensorRtLlm,
            HistogramPrefix::Trtllm,
        ),
        (
            "ollama:",
            HistogramRuntime::Ollama,
            HistogramPrefix::OllamaColon,
        ),
        (
            "ollama_",
            HistogramRuntime::Ollama,
            HistogramPrefix::OllamaUnderscore,
        ),
        (
            "sglang:",
            HistogramRuntime::Sglang,
            HistogramPrefix::SglangColon,
        ),
        (
            "sglang_",
            HistogramRuntime::Sglang,
            HistogramPrefix::SglangUnderscore,
        ),
        ("vllm:", HistogramRuntime::Vllm, HistogramPrefix::VllmColon),
        (
            "vllm_",
            HistogramRuntime::Vllm,
            HistogramPrefix::VllmUnderscore,
        ),
        ("tgi:", HistogramRuntime::Tgi, HistogramPrefix::TgiColon),
        (
            "tgi_",
            HistogramRuntime::Tgi,
            HistogramPrefix::TgiUnderscore,
        ),
    ]
    .into_iter()
    .find_map(|(prefix, runtime, prefix_kind)| {
        family
            .strip_prefix(prefix)
            .map(|suffix| (runtime, prefix_kind, suffix))
    })
}

fn recognize_histogram(family: &str) -> Option<RecognizedHistogram> {
    let (runtime, prefix_kind, suffix) = recognize_histogram_prefix(family)?;

    let (kind, schema) = match suffix {
        "e2e_request_latency_seconds_bucket" => (
            HistogramKind::RequestLatency,
            HistogramSchema::E2eRequestLatency,
        ),
        "request_duration_seconds_bucket" => (
            HistogramKind::RequestLatency,
            HistogramSchema::RequestDuration,
        ),
        "request_latency_seconds_bucket" => (
            HistogramKind::RequestLatency,
            HistogramSchema::RequestLatency,
        ),
        "time_to_first_token_seconds_bucket" => (
            HistogramKind::TimeToFirstToken,
            HistogramSchema::TimeToFirstToken,
        ),
        "time_to_first_token_duration_seconds_bucket" => (
            HistogramKind::TimeToFirstToken,
            HistogramSchema::TimeToFirstTokenDuration,
        ),
        "ttft_seconds_bucket" => (HistogramKind::TimeToFirstToken, HistogramSchema::Ttft),
        "time_per_output_token_seconds_bucket" => (
            HistogramKind::TimePerOutputToken,
            HistogramSchema::TimePerOutputToken,
        ),
        "inter_token_latency_seconds_bucket" => (
            HistogramKind::TimePerOutputToken,
            HistogramSchema::InterTokenLatency,
        ),
        "inter_token_duration_seconds_bucket" => (
            HistogramKind::TimePerOutputToken,
            HistogramSchema::InterTokenDuration,
        ),
        "tpot_seconds_bucket" => (HistogramKind::TimePerOutputToken, HistogramSchema::Tpot),
        "request_queue_time_seconds_bucket" => {
            (HistogramKind::QueueTime, HistogramSchema::RequestQueueTime)
        }
        "queue_time_seconds_bucket" => (HistogramKind::QueueTime, HistogramSchema::QueueTime),
        "request_queue_duration_seconds_bucket" => (
            HistogramKind::QueueTime,
            HistogramSchema::RequestQueueDuration,
        ),
        "waiting_time_seconds_bucket" => (HistogramKind::QueueTime, HistogramSchema::WaitingTime),
        _ => return None,
    };
    Some(RecognizedHistogram {
        kind,
        schema,
        runtime,
        prefix: prefix_kind,
    })
}

fn parse_histogram_sample_key(key: &str) -> Result<(String, Option<f64>), HistogramSampleError> {
    if key.len() > MAX_HISTOGRAM_SAMPLE_KEY_BYTES {
        return Err(HistogramSampleError::LimitExceeded);
    }
    let Some(open) = key.find('{') else {
        return Err(HistogramSampleError::Invalid);
    };
    if !key.ends_with('}') || open + 1 >= key.len() {
        return Err(HistogramSampleError::Invalid);
    }
    let labels = &key[open + 1..key.len() - 1];
    let mut cursor = 0_usize;
    let mut parsed_labels = Vec::new();
    let mut upper_bound = None;
    let mut saw_upper_bound = false;
    let mut label_count = 0_usize;

    while cursor < labels.len() {
        if label_count >= MAX_HISTOGRAM_LABELS_PER_SAMPLE {
            return Err(HistogramSampleError::LimitExceeded);
        }
        label_count += 1;
        let name_start = cursor;
        let Some(first) = labels.as_bytes().get(cursor).copied() else {
            return Err(HistogramSampleError::Invalid);
        };
        if !(first.is_ascii_alphabetic() || first == b'_') {
            return Err(HistogramSampleError::Invalid);
        }
        cursor += 1;
        while labels
            .as_bytes()
            .get(cursor)
            .is_some_and(u8::is_ascii_alphanumeric)
            || labels.as_bytes().get(cursor) == Some(&b'_')
        {
            cursor += 1;
        }
        let name = &labels[name_start..cursor];
        if labels.as_bytes().get(cursor) != Some(&b'=')
            || labels.as_bytes().get(cursor + 1) != Some(&b'"')
        {
            return Err(HistogramSampleError::Invalid);
        }
        cursor += 2;
        let value_start = cursor;
        let mut escaped = false;
        while let Some(byte) = labels.as_bytes().get(cursor).copied() {
            if byte == b'"' && !escaped {
                break;
            }
            if byte == b'\n' || byte == b'\r' {
                return Err(HistogramSampleError::Invalid);
            }
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            }
            cursor += 1;
        }
        if escaped || labels.as_bytes().get(cursor) != Some(&b'"') {
            return Err(HistogramSampleError::Invalid);
        }
        let raw_value = &labels[value_start..cursor];
        cursor += 1;

        if name == "le" {
            if saw_upper_bound {
                return Err(HistogramSampleError::Invalid);
            }
            saw_upper_bound = true;
            upper_bound = parse_histogram_upper_bound(raw_value)?;
        } else {
            if parsed_labels.iter().any(|(existing, _)| *existing == name) {
                return Err(HistogramSampleError::Invalid);
            }
            parsed_labels.push((name, raw_value));
        }

        if cursor == labels.len() {
            break;
        }
        if labels.as_bytes().get(cursor) != Some(&b',') {
            return Err(HistogramSampleError::Invalid);
        }
        cursor += 1;
        if cursor == labels.len() {
            break;
        }
    }
    if !saw_upper_bound {
        return Err(HistogramSampleError::Invalid);
    }

    Ok((canonical_histogram_series_key(parsed_labels)?, upper_bound))
}

fn parse_histogram_upper_bound(raw_value: &str) -> Result<Option<f64>, HistogramSampleError> {
    if raw_value == "+Inf" {
        return Ok(None);
    }
    let bound = raw_value
        .parse::<f64>()
        .map_err(|_| HistogramSampleError::Invalid)?;
    if !bound.is_finite() || bound < 0.0 {
        return Err(HistogramSampleError::Invalid);
    }
    Ok(Some(if bound == 0.0 { 0.0 } else { bound }))
}

fn canonical_histogram_series_key(
    mut parsed_labels: Vec<(&str, &str)>,
) -> Result<String, HistogramSampleError> {
    parsed_labels.sort_unstable_by_key(|(name, _)| *name);
    let mut series_key = String::new();
    for (name, value) in parsed_labels {
        let _ = write!(series_key, "{}:{name}{}:{value}", name.len(), value.len());
        if series_key.len() > MAX_HISTOGRAM_SERIES_KEY_BYTES {
            return Err(HistogramSampleError::LimitExceeded);
        }
    }
    Ok(series_key)
}

fn runtime_for_metric(family: &str) -> &'static str {
    let family = family.to_ascii_lowercase();
    if family.starts_with("vllm:") || family.starts_with("vllm_") {
        "vllm"
    } else if family.starts_with("tgi:") || family.starts_with("tgi_") {
        "tgi"
    } else if family.starts_with("triton:")
        || family.starts_with("triton_")
        || family.starts_with("nv_inference_")
        || family.starts_with("nv_gpu_")
    {
        "triton"
    } else if family.starts_with("ollama:") || family.starts_with("ollama_") {
        "ollama"
    } else if family.starts_with("sglang:") || family.starts_with("sglang_") {
        "sglang"
    } else if family.starts_with("tensorrt_llm_") || family.starts_with("trtllm_") {
        "tensorrt-llm"
    } else {
        ""
    }
}

fn add_runtime(current: &mut String, runtime: &str) {
    if current.is_empty() {
        current.push_str(runtime);
    } else if !current.split('+').any(|item| item == runtime) {
        current.push('+');
        current.push_str(runtime);
    }
}

fn normalize(endpoint: &mut Endpoint, family: &str, value: f64) {
    let family = family.to_ascii_lowercase();
    if contains_any(
        &family,
        &[
            "num_requests_running",
            "requests_running",
            "batch_current_size",
        ],
    ) {
        *endpoint.requests_running.get_or_insert(0.0) += value;
    } else if contains_any(
        &family,
        &[
            "num_requests_waiting",
            "requests_waiting",
            "queue_size",
            "pending_request_count",
        ],
    ) {
        *endpoint.requests_waiting.get_or_insert(0.0) += value;
    } else if contains_any(
        &family,
        &[
            "gpu_cache_usage_perc",
            "kv_cache_usage_perc",
            "kv_cache_usage_percent",
            "kv_cache_utilization",
            "kv_cache_usage_ratio",
        ],
    ) {
        let percent = if (0.0..=1.0).contains(&value) {
            value * 100.0
        } else {
            value
        };
        if endpoint
            .kv_cache_usage_percent
            .is_none_or(|current| percent > current)
        {
            endpoint.kv_cache_usage_percent = Some(percent);
        }
    }
    normalize_counters(endpoint, &family, value);
}

fn normalize_counters(endpoint: &mut Endpoint, family: &str, value: f64) {
    let counters = &mut endpoint.counters;
    if contains_any(
        family,
        &[
            "request_success_total",
            "requests_success_total",
            "nv_inference_request_success",
        ],
    ) || family.ends_with("request_count")
        && !contains_any(family, &["pending_request_count", "waiting_request_count"])
    {
        add_counter(&mut counters.requests_completed_total, value);
    }
    if contains_any(
        family,
        &[
            "request_failure_total",
            "request_errors_total",
            "requests_failed_total",
            "nv_inference_request_failure",
        ],
    ) {
        add_counter(&mut counters.request_errors_total, value);
    }
    if contains_any(
        family,
        &[
            "prompt_tokens_total",
            "input_tokens_total",
            "request_input_length_sum",
        ],
    ) {
        add_counter(&mut counters.prompt_tokens_total, value);
    }
    if contains_any(
        family,
        &[
            "generation_tokens_total",
            "generated_tokens_total",
            "output_tokens_total",
        ],
    ) {
        add_counter(&mut counters.generation_tokens_total, value);
    }
    if family.contains("preemption") && family.ends_with("_total") {
        add_counter(&mut counters.preemptions_total, value);
    }
    if contains_any(
        family,
        &[
            "e2e_request_latency_seconds_sum",
            "request_duration_seconds_sum",
        ],
    ) {
        add_counter(&mut counters.request_latency_seconds_sum, value);
    }
    if contains_any(
        family,
        &[
            "e2e_request_latency_seconds_count",
            "request_duration_seconds_count",
        ],
    ) {
        add_counter(&mut counters.request_latency_seconds_count, value);
    }
    if family.contains("time_to_first_token_seconds_sum") {
        add_counter(&mut counters.time_to_first_token_seconds_sum, value);
    }
    if family.contains("time_to_first_token_seconds_count") {
        add_counter(&mut counters.time_to_first_token_seconds_count, value);
    }
    if contains_any(
        family,
        &[
            "time_per_output_token_seconds_sum",
            "inter_token_latency_seconds_sum",
        ],
    ) {
        add_counter(&mut counters.time_per_output_token_seconds_sum, value);
    }
    if contains_any(
        family,
        &[
            "time_per_output_token_seconds_count",
            "inter_token_latency_seconds_count",
        ],
    ) {
        add_counter(&mut counters.time_per_output_token_seconds_count, value);
    }
}

fn add_counter(counter: &mut Option<f64>, value: f64) {
    *counter.get_or_insert(0.0) += value;
}

fn contains_any(value: &str, candidates: &[&str]) -> bool {
    candidates.iter().any(|candidate| value.contains(candidate))
}

fn failed_endpoint(target: &str, metrics_url: &str, failure: impl Into<String>) -> Endpoint {
    Endpoint {
        url: redact_url(target),
        metrics_url: redact_url(metrics_url),
        kind: "prometheus".to_owned(),
        failure: failure.into(),
        ..Endpoint::default()
    }
}

fn redact_url(raw: &str) -> String {
    let Ok(mut url) = Url::parse(raw) else {
        return "<invalid URL>".to_owned();
    };
    if !url.username().is_empty() || url.password().is_some() {
        let _ = url.set_password(None);
        let _ = url.set_username("");
    }
    if url.query().is_some() {
        url.set_query(Some("REDACTED"));
    }
    url.set_fragment(None);
    url.to_string()
}

fn elapsed_millis(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::*;

    #[test]
    fn normalizes_multiple_inference_runtimes() {
        let mut endpoint = Endpoint::default();
        parse_metrics(
            &mut endpoint,
            "vllm:num_requests_running 2\nvllm:num_requests_waiting 3\nvllm:gpu_cache_usage_perc 0.91\nsglang:num_requests_running 1\nNaN_metric NaN\n",
            100,
        );
        assert_eq!(endpoint.requests_running, Some(3.0));
        assert_eq!(endpoint.requests_waiting, Some(3.0));
        assert_eq!(endpoint.kv_cache_usage_percent, Some(91.0));
        assert_eq!(endpoint.runtime, "vllm+sglang");
    }

    #[test]
    fn resolves_and_redacts_urls() {
        assert_eq!(
            resolve_metrics_url("http://localhost:8000")
                .unwrap()
                .as_str(),
            "http://localhost:8000/metrics"
        );
        let redacted =
            redact_url("https://:secret@example.test/metrics?private-key=abc#private-fragment");
        assert_eq!(redacted, "https://example.test/metrics?REDACTED");
        assert!(!redacted.contains("secret"));
        assert!(!redacted.contains("abc"));
        assert!(!redacted.contains("private-key"));
        assert!(!redacted.contains("private-fragment"));
        assert!(resolve_metrics_url("https://user:secret@example.test").is_err());
    }

    #[test]
    fn remote_cleartext_requires_opt_in_before_network_access() {
        let remote = Url::parse("http://inference.example.invalid:8000").unwrap();
        let local = Url::parse("http://127.0.0.1:8000").unwrap();
        assert!(validate_transport_security(&remote, false).is_err());
        assert!(validate_transport_security(&remote, true).is_ok());
        assert!(validate_transport_security(&local, false).is_ok());
    }

    #[test]
    fn explicit_blank_library_target_is_a_failed_endpoint() {
        let endpoints = collect(&["   ".to_owned()], &ProbeOptions::default());
        assert_eq!(endpoints.len(), 1);
        assert!(!endpoints[0].reachable);
        assert!(endpoints[0].failure.contains("invalid endpoint URL"));
    }

    #[test]
    fn shared_bearer_token_is_scoped_to_one_exact_origin() {
        assert!(
            validate_shared_bearer_origin(&[
                "https://inference.example/metrics".to_owned(),
                "https://inference.example/custom/metrics".to_owned(),
            ])
            .is_ok()
        );
        assert!(
            validate_shared_bearer_origin(&[
                "https://inference.example/metrics".to_owned(),
                "https://other.example/metrics".to_owned(),
            ])
            .is_err()
        );
        assert!(validate_shared_bearer_origin(&["not a URL".to_owned()]).is_err());
    }

    #[test]
    fn multi_origin_token_fails_before_network_access() {
        let options = ProbeOptions {
            bearer_token: Some("private-token".to_owned()),
            ..ProbeOptions::default()
        };
        let endpoints = collect(
            &[
                "https://one.example.invalid/metrics".to_owned(),
                "https://two.example.invalid/metrics".to_owned(),
            ],
            &options,
        );
        assert_eq!(endpoints.len(), 1);
        assert_eq!(endpoints[0].url, "<probe-set>");
        assert!(endpoints[0].failure.contains("multiple URL origins"));
        assert!(!endpoints[0].failure.contains("private-token"));
    }

    #[test]
    fn probe_set_and_retained_metric_memory_are_bounded() {
        let options = ProbeOptions {
            max_targets: 1,
            ..ProbeOptions::default()
        };
        let endpoints = collect(
            &[
                "https://one.example.invalid".to_owned(),
                "https://two.example.invalid".to_owned(),
            ],
            &options,
        );
        assert_eq!(endpoints.len(), 1);
        assert!(endpoints[0].failure.contains("target count"));

        let mut endpoint = Endpoint::default();
        parse_metrics_bounded(
            &mut endpoint,
            "vllm:first_metric 1\nvllm:second_metric 2\n",
            100,
            "vllm:first_metric".len(),
        );
        assert_eq!(endpoint.metric_samples, 1);
        assert!(endpoint.metrics_truncated);
    }

    #[test]
    fn caps_metric_cardinality() {
        let mut endpoint = Endpoint::default();
        parse_metrics(&mut endpoint, "vllm:a{x=\"1\"} 1\nvllm:b{x=\"2\"} 2\n", 1);
        assert_eq!(endpoint.metric_samples, 1);
        assert!(endpoint.metrics_truncated);
    }

    #[test]
    fn aggregates_runtime_counters_across_metric_labels() {
        let mut endpoint = Endpoint::default();
        parse_metrics(
            &mut endpoint,
            "vllm:request_success_total{model=\"a\"} 10\n\
             vllm:request_success_total{model=\"b\"} 5\n\
             vllm:generation_tokens_total{model=\"a\"} 1000\n\
             vllm:e2e_request_latency_seconds_sum 12\n\
             vllm:e2e_request_latency_seconds_count 15\n",
            100,
        );

        assert_eq!(endpoint.counters.requests_completed_total, Some(15.0));
        assert_eq!(endpoint.counters.generation_tokens_total, Some(1_000.0));
        assert_eq!(endpoint.counters.request_latency_seconds_sum, Some(12.0));
        assert_eq!(endpoint.counters.request_latency_seconds_count, Some(15.0));
    }

    #[test]
    fn report_metrics_never_retain_endpoint_controlled_names_or_labels() {
        let mut endpoint = Endpoint::default();
        parse_metrics(
            &mut endpoint,
            "vllm:request_success_total{token=\"private-token\",tenant=\"acme\"} 1\n\
             vllm:stolen_BEARER_VALUE 2\n",
            10,
        );

        assert!(endpoint.metrics.is_empty());
        let serialized = serde_json::to_string(&endpoint).unwrap();
        assert!(!serialized.contains("private-token"));
        assert!(!serialized.contains("acme"));
        assert!(!serialized.contains("tenant"));
        assert!(!serialized.contains("BEARER_VALUE"));
    }

    #[test]
    fn debug_output_redacts_bearer_credentials() {
        let options = ProbeOptions {
            bearer_token: Some("probe-private-token".to_owned()),
            ..ProbeOptions::default()
        };

        let debug = format!("{options:?}");
        assert!(debug.contains("bearer_token_configured: true"));
        assert!(!debug.contains("probe-private-token"));
    }

    #[test]
    fn retains_fixed_histograms_and_aggregates_compatible_series() {
        let mut endpoint = Endpoint::default();
        parse_metrics(
            &mut endpoint,
            "vllm:e2e_request_latency_seconds_bucket{model=\"private-a\",le=\"0.1\"} 2\n\
             vllm:e2e_request_latency_seconds_bucket{le=\"1\",model=\"private-a\"} 5\n\
             vllm:e2e_request_latency_seconds_bucket{model=\"private-a\",le=\"+Inf\"} 6\n\
             vllm:e2e_request_latency_seconds_bucket{model=\"private-b\",le=\"0.1\"} 1\n\
             vllm:e2e_request_latency_seconds_bucket{model=\"private-b\",le=\"1\"} 3\n\
             vllm:e2e_request_latency_seconds_bucket{model=\"private-b\",le=\"+Inf\"} 4\n\
             vllm:time_to_first_token_seconds_bucket{le=\"0.05\"} 2\n\
             vllm:time_to_first_token_seconds_bucket{le=\"+Inf\"} 2\n\
             vllm:time_per_output_token_seconds_bucket{le=\"0.01\"} 3\n\
             vllm:time_per_output_token_seconds_bucket{le=\"+Inf\"} 3\n\
             vllm:request_queue_time_seconds_bucket{le=\"0.1\"} 1\n\
             vllm:request_queue_time_seconds_bucket{le=\"+Inf\"} 1\n",
            100,
        );

        let request = &endpoint.counters.histograms.request_latency.buckets;
        assert_eq!(request.len(), 3);
        assert_eq!(endpoint.counters.histograms.request_latency.series_count, 2);
        assert_eq!(request[0].upper_bound_seconds, Some(0.1));
        assert!((request[0].cumulative_count - 3.0).abs() < f64::EPSILON);
        assert!((request[1].cumulative_count - 8.0).abs() < f64::EPSILON);
        assert_eq!(request[2].upper_bound_seconds, None);
        assert!((request[2].cumulative_count - 10.0).abs() < f64::EPSILON);
        assert!(!endpoint.counters.histograms.time_to_first_token.is_empty());
        assert!(
            !endpoint
                .counters
                .histograms
                .time_per_output_token
                .is_empty()
        );
        assert!(!endpoint.counters.histograms.queue_time.is_empty());

        let serialized = serde_json::to_string(&endpoint).unwrap();
        assert!(!serialized.contains("private-a"));
        assert!(!serialized.contains("private-b"));
        assert!(!serialized.contains("model"));
    }

    #[test]
    fn fixed_runtime_discriminators_keep_compatible_series_distinct() {
        let mut endpoint = Endpoint::default();
        parse_metrics(
            &mut endpoint,
            "vllm:e2e_request_latency_seconds_bucket{le=\"0.1\"} 1\n\
             vllm:e2e_request_latency_seconds_bucket{le=\"+Inf\"} 2\n\
             sglang:e2e_request_latency_seconds_bucket{le=\"0.1\"} 3\n\
             sglang:e2e_request_latency_seconds_bucket{le=\"+Inf\"} 4\n",
            100,
        );

        let histogram = &endpoint.counters.histograms.request_latency;
        assert_eq!(histogram.series_count, 2);
        assert!((histogram.buckets[0].cumulative_count - 4.0).abs() < f64::EPSILON);
        assert!((histogram.buckets[1].cumulative_count - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn interval_identity_rejects_fixed_schema_switches() {
        let mut before = Endpoint::default();
        parse_metrics(
            &mut before,
            "vllm:e2e_request_latency_seconds_bucket{le=\"0.1\"} 1\n\
             vllm:e2e_request_latency_seconds_bucket{le=\"+Inf\"} 2\n",
            10,
        );
        let mut after = Endpoint::default();
        parse_metrics(
            &mut after,
            "vllm:request_duration_seconds_bucket{le=\"0.1\"} 2\n\
             vllm:request_duration_seconds_bucket{le=\"+Inf\"} 4\n",
            10,
        );

        assert!(
            before
                .counters
                .histograms
                .request_latency
                .interval(&after.counters.histograms.request_latency)
                .is_none()
        );

        let mut prefix_after = Endpoint::default();
        parse_metrics(
            &mut prefix_after,
            "vllm_e2e_request_latency_seconds_bucket{le=\"0.1\"} 2\n\
             vllm_e2e_request_latency_seconds_bucket{le=\"+Inf\"} 4\n",
            10,
        );
        assert!(
            before
                .counters
                .histograms
                .request_latency
                .interval(&prefix_after.counters.histograms.request_latency)
                .is_none()
        );
    }

    #[test]
    fn histogram_parser_accepts_spaces_escapes_and_trailing_label_commas() {
        let mut endpoint = Endpoint::default();
        parse_metrics(
            &mut endpoint,
            r#"vllm:e2e_request_latency_seconds_bucket{model="tenant one",note="quote\" slash\\",le="0.1",} 1
vllm:e2e_request_latency_seconds_bucket{model="tenant one",note="quote\" slash\\",le="+Inf",} 2
"#,
            10,
        );

        let histogram = &endpoint.counters.histograms.request_latency;
        assert_eq!(histogram.series_count, 1);
        assert_eq!(histogram.buckets.len(), 2);
        let serialized = serde_json::to_string(&endpoint).unwrap();
        let debug = format!("{endpoint:?}");
        assert!(!serialized.contains("tenant one"));
        assert!(!serialized.contains("quote"));
        assert!(!debug.contains("tenant one"));
        assert!(!debug.contains("quote"));
    }

    #[test]
    fn histogram_label_limit_counts_the_le_label() {
        let accepted = (0..MAX_HISTOGRAM_LABELS_PER_SAMPLE - 1)
            .map(|index| format!("label_{index}=\"x\""))
            .collect::<Vec<_>>()
            .join(",");
        let excessive = (0..MAX_HISTOGRAM_LABELS_PER_SAMPLE)
            .map(|index| format!("label_{index}=\"x\""))
            .collect::<Vec<_>>()
            .join(",");
        let body = format!(
            "vllm:e2e_request_latency_seconds_bucket{{le=\"0.1\",{accepted}}} 1\n\
             vllm:e2e_request_latency_seconds_bucket{{le=\"+Inf\",{accepted}}} 1\n\
             vllm:time_to_first_token_seconds_bucket{{le=\"0.1\",{excessive}}} 1\n\
             vllm:time_to_first_token_seconds_bucket{{le=\"+Inf\",{excessive}}} 1\n"
        );
        let mut endpoint = Endpoint::default();
        parse_metrics(&mut endpoint, &body, 100);

        assert!(!endpoint.counters.histograms.request_latency.is_empty());
        assert!(endpoint.counters.histograms.time_to_first_token.is_empty());
        assert!(endpoint.metrics_truncated);
    }

    #[test]
    fn rejects_histogram_series_with_mismatched_boundaries() {
        let mut endpoint = Endpoint::default();
        parse_metrics(
            &mut endpoint,
            "vllm:e2e_request_latency_seconds_bucket{model=\"a\",le=\"0.1\"} 1\n\
             vllm:e2e_request_latency_seconds_bucket{model=\"a\",le=\"1\"} 2\n\
             vllm:e2e_request_latency_seconds_bucket{model=\"a\",le=\"+Inf\"} 3\n\
             vllm:e2e_request_latency_seconds_bucket{model=\"b\",le=\"0.2\"} 1\n\
             vllm:e2e_request_latency_seconds_bucket{model=\"b\",le=\"1\"} 2\n\
             vllm:e2e_request_latency_seconds_bucket{model=\"b\",le=\"+Inf\"} 3\n",
            100,
        );

        assert!(endpoint.counters.histograms.request_latency.is_empty());
    }

    #[test]
    fn rejects_missing_infinity_nonmonotonic_and_nonfinite_histograms() {
        let mut endpoint = Endpoint::default();
        parse_metrics(
            &mut endpoint,
            "vllm:e2e_request_latency_seconds_bucket{le=\"0.1\"} 1\n\
             vllm:e2e_request_latency_seconds_bucket{le=\"1\"} 2\n\
             vllm:time_to_first_token_seconds_bucket{le=\"0.1\"} 2\n\
             vllm:time_to_first_token_seconds_bucket{le=\"+Inf\"} 1\n\
             vllm:time_per_output_token_seconds_bucket{le=\"0.1\"} NaN\n\
             vllm:time_per_output_token_seconds_bucket{le=\"+Inf\"} 2\n\
             vllm:request_queue_time_seconds_bucket{le=\"NaN\"} 1\n\
             vllm:request_queue_time_seconds_bucket{le=\"+Inf\"} 1\n",
            100,
        );

        assert!(endpoint.counters.histograms.request_latency.is_empty());
        assert!(endpoint.counters.histograms.time_to_first_token.is_empty());
        assert!(
            endpoint
                .counters
                .histograms
                .time_per_output_token
                .is_empty()
        );
        assert!(endpoint.counters.histograms.queue_time.is_empty());
    }

    #[test]
    fn histogram_bucket_and_series_retention_are_bounded() {
        let mut body = String::new();
        for bucket in 0..=MAX_RUNTIME_HISTOGRAM_BUCKETS {
            let _ = writeln!(
                body,
                "vllm:e2e_request_latency_seconds_bucket{{le=\"{bucket}\"}} {bucket}"
            );
        }
        body.push_str("vllm:e2e_request_latency_seconds_bucket{le=\"+Inf\"} 100\n");
        for series in 0..=MAX_RUNTIME_HISTOGRAM_SERIES {
            let _ = writeln!(
                body,
                "vllm:time_to_first_token_seconds_bucket{{model=\"{series}\",le=\"0.1\"}} 1"
            );
            let _ = writeln!(
                body,
                "vllm:time_to_first_token_seconds_bucket{{model=\"{series}\",le=\"+Inf\"}} 1"
            );
        }
        let mut endpoint = Endpoint::default();
        parse_metrics(&mut endpoint, &body, 1_000);

        assert!(endpoint.metrics_truncated);
        assert!(endpoint.counters.histograms.request_latency.is_empty());
        assert!(endpoint.counters.histograms.time_to_first_token.is_empty());
    }

    #[test]
    fn partial_scrapes_never_retain_histogram_snapshots() {
        let mut endpoint = Endpoint::default();
        parse_metrics(
            &mut endpoint,
            "vllm:e2e_request_latency_seconds_bucket{le=\"0.1\"} 1\n\
             vllm:e2e_request_latency_seconds_bucket{le=\"1\"} 2\n\
             vllm:e2e_request_latency_seconds_bucket{le=\"+Inf\"} 3\n",
            2,
        );

        assert!(endpoint.metrics_truncated);
        assert!(endpoint.counters.histograms.request_latency.is_empty());
    }

    #[test]
    fn only_exact_fixed_histogram_schemas_are_retained() {
        let mut endpoint = Endpoint::default();
        parse_metrics(
            &mut endpoint,
            "vllm:tenant_e2e_request_latency_seconds_bucket{tenant=\"private\",le=\"0.1\"} 1\n\
             vllm:tenant_e2e_request_latency_seconds_bucket{tenant=\"private\",le=\"+Inf\"} 1\n",
            10,
        );

        assert!(endpoint.counters.histograms.is_empty());
        assert!(
            !serde_json::to_string(&endpoint)
                .unwrap()
                .contains("private")
        );
    }
}
