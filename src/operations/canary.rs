//! Bounded active inference canary orchestration, aggregation, and SLO gates.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};
use chrono::Utc;

use crate::domain::{
    CANARY_VERSION, CanaryAttempt, CanaryDistribution, CanaryGate, CanaryPlan, CanaryPolicy,
    CanaryReport, CanaryStatus, CanarySummary, valid_canary_workload_id,
};
use crate::inference::{
    OpenAiClient, OpenAiClientOptions, request_body_memory_budget, worker_memory_budget,
};
use crate::operations::workload::execute_batch;

pub const DEFAULT_CANARY_PROMPT: &str = "Reply with exactly: gpu-watchman-ok";
pub const DEFAULT_CANARY_EXPECTATION: &str = "gpu-watchman-ok";
pub use crate::domain::{
    DEFAULT_CANARY_WORKLOAD_ID, MAX_CANARY_WORKLOAD_ID_BYTES as MAX_WORKLOAD_ID_BYTES,
};

const MAX_REQUEST_COUNT: u32 = 10_000;
const MAX_CONCURRENCY: u32 = 64;
const MAX_COMPLETION_TOKENS: u32 = 65_536;
const MAX_REQUESTED_COMPLETION_TOKENS: u64 = 1_000_000;
const MAX_PROMPT_BYTES: usize = 1 << 20;
const MAX_EXPECTATION_BYTES: usize = 64 << 10;
const MAX_IDENTITY_BYTES: usize = 64 << 10;
const MAX_API_KEY_BYTES: usize = 64 << 10;
const DEFAULT_RESPONSE_BYTES: usize = 1 << 20;
const MAX_RESPONSE_BYTES: usize = 8 << 20;
const MAX_AGGREGATE_CANARY_MEMORY_BYTES: u64 = 768 << 20;
const RETAINED_ATTEMPT_HEAP_BUDGET_BYTES: u64 = 576;
const AGGREGATION_BYTES_PER_ATTEMPT: u64 = 8;
const INPUT_COPY_BUDGET: u64 = 3;
const MAX_REQUEST_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const MAX_PLANNED_DURATION: Duration = Duration::from_secs(15 * 60);

/// Optional SLO gates. Selected latency and throughput gates fail closed when
/// a successful request did not expose the required measurement.
#[derive(Debug, Clone)]
pub struct CanaryThresholds {
    pub min_success_percent: f64,
    pub max_ttft: Option<Duration>,
    pub max_e2e: Option<Duration>,
    pub min_output_tokens_per_second: Option<f64>,
}

impl Default for CanaryThresholds {
    fn default() -> Self {
        Self {
            min_success_percent: 100.0,
            max_ttft: None,
            max_e2e: None,
            min_output_tokens_per_second: None,
        }
    }
}

/// Complete input for a bounded OpenAI-compatible canary run.
///
/// `Debug` is intentionally not derived because this type owns request
/// credentials and prompt content.
#[derive(Clone)]
pub struct CanaryOptions {
    pub base_url: String,
    pub model: String,
    pub workload_id: String,
    pub api_key: Option<String>,
    pub prompt: String,
    pub expectation: Option<String>,
    pub max_tokens: u32,
    pub count: u32,
    pub concurrency: u32,
    pub timeout: Duration,
    pub max_body_bytes: usize,
    pub stream: bool,
    pub allow_insecure_http: bool,
    pub thresholds: CanaryThresholds,
}

impl Default for CanaryOptions {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            model: String::new(),
            workload_id: DEFAULT_CANARY_WORKLOAD_ID.to_owned(),
            api_key: None,
            prompt: DEFAULT_CANARY_PROMPT.to_owned(),
            expectation: Some(DEFAULT_CANARY_EXPECTATION.to_owned()),
            max_tokens: 16,
            count: 1,
            concurrency: 1,
            timeout: Duration::from_secs(30),
            max_body_bytes: DEFAULT_RESPONSE_BYTES,
            stream: true,
            allow_insecure_http: false,
            thresholds: CanaryThresholds::default(),
        }
    }
}

/// Run the configured requests and return a report even when service or SLO
/// validation fails. Only invalid local configuration returns `Err`.
pub fn run(options: &CanaryOptions) -> Result<CanaryReport> {
    validate(options)?;
    let client = Arc::new(OpenAiClient::new(OpenAiClientOptions {
        base_url: options.base_url.clone(),
        model: options.model.clone(),
        api_key: options.api_key.clone(),
        prompt: options.prompt.clone(),
        expectation: options.expectation.clone(),
        max_tokens: options.max_tokens,
        timeout: options.timeout,
        max_body_bytes: options.max_body_bytes,
        stream: options.stream,
        allow_insecure_http: options.allow_insecure_http,
    })?);
    let started_at = Utc::now();
    let execution = execute_batch(
        usize::try_from(options.count).unwrap_or(usize::MAX),
        usize::try_from(options.concurrency).unwrap_or(usize::MAX),
        |index| client.execute(u32::try_from(index).unwrap_or(u32::MAX)),
    )?;
    let duration = execution.duration;
    let attempts = execution.results;
    let summary = summarize(&attempts, duration);
    let policy = canary_policy(options);
    let gates = canonical_gates(&summary, &attempts, &policy)?;
    let status = if gates.iter().all(|gate| gate.passed) {
        CanaryStatus::Pass
    } else {
        CanaryStatus::Fail
    };

    Ok(CanaryReport {
        canary_version: CANARY_VERSION,
        started_at,
        duration_ms: elapsed_millis(duration),
        status,
        workload_id: options.workload_id.clone(),
        target: client.target(),
        plan: CanaryPlan {
            count: options.count,
            concurrency: options.concurrency,
            max_tokens: options.max_tokens,
            timeout_ms: duration_millis_ceil(options.timeout),
            response_limit_bytes: options.max_body_bytes,
        },
        policy: Some(policy),
        summary,
        gates,
        attempts,
    })
}

fn validate(options: &CanaryOptions) -> Result<()> {
    if options.base_url.trim().is_empty() || options.base_url.len() > MAX_IDENTITY_BYTES {
        bail!("canary base URL must contain 1 byte to 64 KiB");
    }
    if options.model.trim().is_empty() || options.model.len() > MAX_IDENTITY_BYTES {
        bail!("canary model must contain 1 byte to 64 KiB");
    }
    if !valid_canary_workload_id(&options.workload_id) {
        bail!(
            "canary workload ID must start with an ASCII letter or digit and contain at most \
             {MAX_WORKLOAD_ID_BYTES} bytes using letters, digits, dot, underscore, colon, slash, or hyphen"
        );
    }
    if options
        .api_key
        .as_ref()
        .is_some_and(|value| value.is_empty() || value.len() > MAX_API_KEY_BYTES)
    {
        bail!("canary API key must contain 1 byte to 64 KiB");
    }
    if options.prompt.is_empty() || options.prompt.len() > MAX_PROMPT_BYTES {
        bail!("canary prompt must contain 1 byte to 1 MiB");
    }
    if options
        .expectation
        .as_ref()
        .is_some_and(|value| value.is_empty() || value.len() > MAX_EXPECTATION_BYTES)
    {
        bail!("canary expectation must contain 1 byte to 64 KiB");
    }
    if options.max_tokens == 0 || options.max_tokens > MAX_COMPLETION_TOKENS {
        bail!("canary max tokens must be between 1 and {MAX_COMPLETION_TOKENS}");
    }
    if options.count == 0 || options.count > MAX_REQUEST_COUNT {
        bail!("canary count must be between 1 and {MAX_REQUEST_COUNT}");
    }
    if options.concurrency == 0
        || options.concurrency > options.count
        || options.concurrency > MAX_CONCURRENCY
    {
        bail!("canary concurrency must be between 1 and min(count, {MAX_CONCURRENCY})");
    }
    let requested_tokens = u64::from(options.count).saturating_mul(u64::from(options.max_tokens));
    if requested_tokens > MAX_REQUESTED_COMPLETION_TOKENS {
        bail!(
            "canary count multiplied by max tokens must not exceed {MAX_REQUESTED_COMPLETION_TOKENS}"
        );
    }
    if options.timeout.is_zero() || options.timeout > MAX_REQUEST_TIMEOUT {
        bail!("canary timeout must be greater than zero and at most 5 minutes");
    }
    let waves = options.count.div_ceil(options.concurrency);
    let planned_duration = options.timeout.checked_mul(waves).unwrap_or(Duration::MAX);
    if planned_duration > MAX_PLANNED_DURATION {
        bail!("canary request waves multiplied by timeout must not exceed 15 minutes");
    }
    validate_resource_budget(options)?;
    let thresholds = &options.thresholds;
    if !thresholds.min_success_percent.is_finite()
        || !(0.0..=100.0).contains(&thresholds.min_success_percent)
    {
        bail!("minimum success percent must be finite and between zero and 100");
    }
    if thresholds.max_ttft.is_some_and(|value| value.is_zero())
        || thresholds.max_e2e.is_some_and(|value| value.is_zero())
    {
        bail!("canary duration thresholds must be greater than zero");
    }
    if thresholds
        .min_output_tokens_per_second
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        bail!("minimum output token rate must be finite and non-negative");
    }
    if !options.stream
        && (thresholds.max_ttft.is_some() || thresholds.min_output_tokens_per_second.is_some())
    {
        bail!("TTFT and output token rate gates require streaming responses");
    }
    Ok(())
}

fn validate_resource_budget(options: &CanaryOptions) -> Result<()> {
    if options.max_body_bytes == 0 || options.max_body_bytes > MAX_RESPONSE_BYTES {
        bail!("canary response limit must be between 1 byte and 8 MiB");
    }
    let Some(memory_bytes) = estimated_canary_memory_bytes(options) else {
        bail!("canary working-set estimate overflowed; reduce the request plan");
    };
    if memory_bytes > MAX_AGGREGATE_CANARY_MEMORY_BYTES {
        bail!(
            "canary estimated working set ({memory_bytes} bytes) must not exceed \
             {MAX_AGGREGATE_CANARY_MEMORY_BYTES} bytes; reduce concurrency, response limit, \
             prompt, or expectation size"
        );
    }
    Ok(())
}

/// Conservative admission estimate for allocations retained by the run and
/// multiplied by its worker pool. Three input copies cover the caller, client
/// construction, and JSON/header materialization. Each retained result covers
/// its inline representation, the bounded failure string, allocator slack,
/// and one temporary aggregation sample.
fn estimated_canary_memory_bytes(options: &CanaryOptions) -> Option<u64> {
    let expectation_bytes = options.expectation.as_ref().map_or(0, String::len);
    let api_key_bytes = options.api_key.as_ref().map_or(0, String::len);
    let input_bytes = u64::try_from(options.base_url.len())
        .ok()?
        .checked_add(u64::try_from(options.model.len()).ok()?)?
        .checked_add(u64::try_from(options.workload_id.len()).ok()?)?
        .checked_add(u64::try_from(options.prompt.len()).ok()?)?
        .checked_add(u64::try_from(expectation_bytes).ok()?)?
        .checked_add(u64::try_from(api_key_bytes).ok()?)?;
    let request_body_bytes = request_body_memory_budget(options.model.len(), options.prompt.len())?;
    let worker = worker_memory_budget(
        request_body_bytes,
        options.max_body_bytes,
        expectation_bytes,
        options.stream,
    )?;
    let attempt_bytes = u64::try_from(std::mem::size_of::<Option<CanaryAttempt>>())
        .ok()?
        .checked_add(RETAINED_ATTEMPT_HEAP_BUDGET_BYTES)?
        .checked_add(AGGREGATION_BYTES_PER_ATTEMPT)?;
    let retained_bytes = u64::from(options.count).checked_mul(attempt_bytes)?;
    let persistent_bytes = input_bytes
        .checked_mul(INPUT_COPY_BUDGET)?
        .checked_add(request_body_bytes)?
        .checked_add(retained_bytes)?;
    persistent_bytes.checked_add(u64::from(options.concurrency).checked_mul(worker)?)
}

fn summarize(attempts: &[CanaryAttempt], duration: Duration) -> CanarySummary {
    let attempted = u32::try_from(attempts.len()).unwrap_or(u32::MAX);
    let duration_seconds = duration.as_secs_f64();
    let achieved_requests_per_second = if duration_seconds > 0.0 {
        f64::from(attempted) / duration_seconds
    } else {
        0.0
    };
    recompute_summary(attempts, achieved_requests_per_second)
}

/// Rebuild every attempt-derived summary field with the production percentile
/// implementation. The supplied achieved rate is retained because the report's
/// rounded millisecond duration cannot reproduce the original sub-millisecond
/// wall-clock measurement exactly.
pub(crate) fn recompute_summary(
    attempts: &[CanaryAttempt],
    achieved_requests_per_second: f64,
) -> CanarySummary {
    let attempted = u32::try_from(attempts.len()).unwrap_or(u32::MAX);
    let succeeded = u32::try_from(attempts.iter().filter(|attempt| attempt.success).count())
        .unwrap_or(u32::MAX);
    CanarySummary {
        attempted,
        succeeded,
        failed: attempted.saturating_sub(succeeded),
        success_percent: if attempted == 0 {
            0.0
        } else {
            f64::from(succeeded) * 100.0 / f64::from(attempted)
        },
        achieved_requests_per_second,
        prompt_tokens_total: attempts
            .iter()
            .filter(|attempt| attempt.success)
            .filter_map(|attempt| attempt.prompt_tokens)
            .fold(0, u64::saturating_add),
        completion_tokens_total: attempts
            .iter()
            .filter(|attempt| attempt.success)
            .filter_map(|attempt| attempt.completion_tokens)
            .fold(0, u64::saturating_add),
        headers_ms: distribution(
            attempts
                .iter()
                .filter(|attempt| attempt.success)
                .filter_map(|attempt| attempt.headers_ms),
        ),
        ttft_ms: distribution(
            attempts
                .iter()
                .filter(|attempt| attempt.success)
                .filter_map(|attempt| attempt.ttft_ms),
        ),
        e2e_ms: distribution(
            attempts
                .iter()
                .filter(|attempt| attempt.success)
                .filter_map(|attempt| attempt.e2e_ms),
        ),
        output_tokens_per_second: distribution(
            attempts
                .iter()
                .filter(|attempt| attempt.success)
                .filter_map(|attempt| attempt.output_tokens_per_second),
        ),
    }
}

/// Compare a retained aggregate with one rebuilt from its serialized attempts.
///
/// JSON preserves each individual `f64`, but rebuilding a mean after a report
/// round-trip can differ from the producer's mean by a handful of ULPs. Keep
/// every selected observation exact and allow that narrow drift only for the
/// derived mean.
pub(crate) fn summary_evidence_matches(retained: &CanarySummary, rebuilt: &CanarySummary) -> bool {
    retained.attempted == rebuilt.attempted
        && retained.succeeded == rebuilt.succeeded
        && retained.failed == rebuilt.failed
        && retained.success_percent.to_bits() == rebuilt.success_percent.to_bits()
        && retained.achieved_requests_per_second.to_bits()
            == rebuilt.achieved_requests_per_second.to_bits()
        && retained.prompt_tokens_total == rebuilt.prompt_tokens_total
        && retained.completion_tokens_total == rebuilt.completion_tokens_total
        && distribution_evidence_matches(retained.headers_ms.as_ref(), rebuilt.headers_ms.as_ref())
        && distribution_evidence_matches(retained.ttft_ms.as_ref(), rebuilt.ttft_ms.as_ref())
        && distribution_evidence_matches(retained.e2e_ms.as_ref(), rebuilt.e2e_ms.as_ref())
        && distribution_evidence_matches(
            retained.output_tokens_per_second.as_ref(),
            rebuilt.output_tokens_per_second.as_ref(),
        )
}

pub(crate) fn distribution_evidence_matches(
    retained: Option<&CanaryDistribution>,
    rebuilt: Option<&CanaryDistribution>,
) -> bool {
    match (retained, rebuilt) {
        (None, None) => true,
        (Some(retained), Some(rebuilt)) => {
            retained.samples == rebuilt.samples
                && retained.min.to_bits() == rebuilt.min.to_bits()
                && within_ulps(retained.mean, rebuilt.mean, 16)
                && retained.p50.to_bits() == rebuilt.p50.to_bits()
                && retained.p95.to_bits() == rebuilt.p95.to_bits()
                && retained.p99.to_bits() == rebuilt.p99.to_bits()
                && retained.max.to_bits() == rebuilt.max.to_bits()
        }
        _ => false,
    }
}

pub(crate) fn within_ulps(left: f64, right: f64, max_ulps: u64) -> bool {
    if left.to_bits() == right.to_bits() {
        return true;
    }
    if !left.is_finite() || !right.is_finite() {
        return false;
    }
    ordered_float_bits(left).abs_diff(ordered_float_bits(right)) <= max_ulps
}

fn ordered_float_bits(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits & (1_u64 << 63) == 0 {
        bits | (1_u64 << 63)
    } else {
        !bits
    }
}

#[allow(clippy::cast_precision_loss)]
pub(crate) fn distribution(values: impl Iterator<Item = f64>) -> Option<CanaryDistribution> {
    let mut values = values.filter(|value| value.is_finite()).collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let sum = values.iter().sum::<f64>();
    Some(CanaryDistribution {
        samples: values.len(),
        min: values[0],
        mean: sum / values.len() as f64,
        p50: nearest_rank(&values, 50),
        p95: nearest_rank(&values, 95),
        p99: nearest_rank(&values, 99),
        max: values[values.len() - 1],
    })
}

fn nearest_rank(values: &[f64], percentile: usize) -> f64 {
    let rank = percentile.saturating_mul(values.len()).div_ceil(100);
    values[rank.saturating_sub(1).min(values.len() - 1)]
}

#[cfg(test)]
fn evaluate_gates(
    summary: &CanarySummary,
    attempts: &[CanaryAttempt],
    thresholds: &CanaryThresholds,
) -> Vec<CanaryGate> {
    canonical_gates(
        summary,
        attempts,
        &CanaryPolicy {
            min_success_percent: thresholds.min_success_percent,
            max_ttft_ms: thresholds
                .max_ttft
                .map(|value| value.as_secs_f64() * 1_000.0),
            max_e2e_ms: thresholds
                .max_e2e
                .map(|value| value.as_secs_f64() * 1_000.0),
            min_output_tokens_per_second: thresholds.min_output_tokens_per_second,
            expectation_configured: false,
        },
    )
    .expect("validated canary thresholds must produce a canonical policy")
}

pub(crate) fn canonical_gates(
    summary: &CanarySummary,
    attempts: &[CanaryAttempt],
    policy: &CanaryPolicy,
) -> Result<Vec<CanaryGate>> {
    validate_policy(policy)?;
    let mut gates = vec![
        CanaryGate {
            name: "min_successful_requests".to_owned(),
            operator: ">=".to_owned(),
            observed: Some(f64::from(summary.succeeded)),
            threshold: 1.0,
            passed: summary.succeeded > 0,
            detail: String::new(),
        },
        CanaryGate {
            name: "min_success_percent".to_owned(),
            operator: ">=".to_owned(),
            observed: Some(summary.success_percent),
            threshold: policy.min_success_percent,
            passed: summary.success_percent >= policy.min_success_percent,
            detail: String::new(),
        },
    ];
    if let Some(threshold) = policy.max_ttft_ms {
        gates.push(maximum_gate(
            "max_ttft_ms",
            summary.ttft_ms.as_ref(),
            summary.succeeded,
            threshold,
        ));
    }
    if let Some(threshold) = policy.max_e2e_ms {
        gates.push(maximum_gate(
            "max_e2e_ms",
            summary.e2e_ms.as_ref(),
            summary.succeeded,
            threshold,
        ));
    }
    if let Some(threshold) = policy.min_output_tokens_per_second {
        let distribution = summary.output_tokens_per_second.as_ref();
        let complete = distribution.is_some_and(|value| {
            value.samples == usize::try_from(summary.succeeded).unwrap_or(usize::MAX)
        });
        gates.push(CanaryGate {
            name: "min_output_tokens_per_second".to_owned(),
            operator: ">=".to_owned(),
            observed: distribution.map(|value| value.min),
            threshold,
            passed: complete && distribution.is_some_and(|value| value.min >= threshold),
            detail: missing_metric_detail(
                distribution.map_or(0, |value| value.samples),
                summary.succeeded,
            ),
        });
    }
    debug_assert_eq!(
        usize::try_from(summary.attempted).ok(),
        Some(attempts.len())
    );
    Ok(gates)
}

pub(crate) fn validate_policy(policy: &CanaryPolicy) -> Result<()> {
    if !policy.min_success_percent.is_finite()
        || !(0.0..=100.0).contains(&policy.min_success_percent)
    {
        bail!("canary policy minimum success percent is invalid");
    }
    if policy
        .max_ttft_ms
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
        || policy
            .max_e2e_ms
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        bail!("canary policy latency threshold is invalid");
    }
    if policy
        .min_output_tokens_per_second
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        bail!("canary policy output-token-rate threshold is invalid");
    }
    Ok(())
}

fn canary_policy(options: &CanaryOptions) -> CanaryPolicy {
    CanaryPolicy {
        min_success_percent: options.thresholds.min_success_percent,
        max_ttft_ms: options
            .thresholds
            .max_ttft
            .map(|value| value.as_secs_f64() * 1_000.0),
        max_e2e_ms: options
            .thresholds
            .max_e2e
            .map(|value| value.as_secs_f64() * 1_000.0),
        min_output_tokens_per_second: options.thresholds.min_output_tokens_per_second,
        expectation_configured: options.expectation.is_some(),
    }
}

fn maximum_gate(
    name: &str,
    distribution: Option<&CanaryDistribution>,
    succeeded: u32,
    threshold: f64,
) -> CanaryGate {
    let complete = distribution
        .is_some_and(|value| value.samples == usize::try_from(succeeded).unwrap_or(usize::MAX));
    CanaryGate {
        name: name.to_owned(),
        operator: "<=".to_owned(),
        observed: distribution.map(|value| value.max),
        threshold,
        passed: complete && distribution.is_some_and(|value| value.max <= threshold),
        detail: missing_metric_detail(distribution.map_or(0, |value| value.samples), succeeded),
    }
}

fn missing_metric_detail(samples: usize, succeeded: u32) -> String {
    let succeeded = usize::try_from(succeeded).unwrap_or(usize::MAX);
    if samples == succeeded {
        String::new()
    } else {
        format!(
            "measurement unavailable for {} successful request(s)",
            succeeded.saturating_sub(samples)
        )
    }
}

fn elapsed_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn duration_millis_ceil(duration: Duration) -> u64 {
    let nanos = duration.as_nanos();
    let millis = nanos.div_ceil(1_000_000);
    u64::try_from(millis).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn successful_attempt(index: u32, ttft_ms: f64, e2e_ms: f64) -> CanaryAttempt {
        CanaryAttempt {
            index,
            success: true,
            status_code: 200,
            headers_ms: Some(10.0),
            ttft_ms: Some(ttft_ms),
            e2e_ms: Some(e2e_ms),
            prompt_tokens: Some(4),
            completion_tokens: Some(3),
            output_tokens_per_second: Some(20.0),
            model: "model".to_owned(),
            finish_reason: "stop".to_owned(),
            expectation_met: Some(true),
            failure: None,
        }
    }

    #[test]
    fn distributions_use_exact_nearest_rank_percentiles() {
        let attempts = (0..20)
            .map(|index| successful_attempt(index, f64::from(index + 1), 100.0))
            .collect::<Vec<_>>();
        let summary = summarize(&attempts, Duration::from_secs(2));
        let ttft = summary.ttft_ms.unwrap();

        assert_eq!(ttft.samples, 20);
        assert!((ttft.p50 - 10.0).abs() < f64::EPSILON);
        assert!((ttft.p95 - 19.0).abs() < f64::EPSILON);
        assert!((ttft.p99 - 20.0).abs() < f64::EPSILON);
        assert!((summary.achieved_requests_per_second - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn retained_summary_allows_only_narrow_mean_round_trip_drift() {
        let attempts = (0..20)
            .map(|index| successful_attempt(index, f64::from(index + 1), 100.0))
            .collect::<Vec<_>>();
        let rebuilt = summarize(&attempts, Duration::from_secs(2));
        let mut retained = rebuilt.clone();
        let mean = retained.ttft_ms.as_ref().unwrap().mean;
        retained.ttft_ms.as_mut().unwrap().mean = f64::from_bits(mean.to_bits() + 8);
        assert!(summary_evidence_matches(&retained, &rebuilt));

        retained.ttft_ms.as_mut().unwrap().mean = f64::from_bits(mean.to_bits() + 17);
        assert!(!summary_evidence_matches(&retained, &rebuilt));

        retained.ttft_ms.as_mut().unwrap().mean = mean;
        retained.ttft_ms.as_mut().unwrap().p95 += 1.0;
        assert!(!summary_evidence_matches(&retained, &rebuilt));
    }

    #[test]
    fn selected_measurement_gates_fail_closed() {
        let mut attempt = successful_attempt(0, 50.0, 100.0);
        attempt.output_tokens_per_second = None;
        let summary = summarize(std::slice::from_ref(&attempt), Duration::from_secs(1));
        let gates = evaluate_gates(
            &summary,
            &[attempt],
            &CanaryThresholds {
                max_ttft: Some(Duration::from_millis(100)),
                min_output_tokens_per_second: Some(10.0),
                ..CanaryThresholds::default()
            },
        );

        assert!(
            gates
                .iter()
                .find(|gate| gate.name == "max_ttft_ms")
                .unwrap()
                .passed
        );
        let token_gate = gates
            .iter()
            .find(|gate| gate.name == "min_output_tokens_per_second")
            .unwrap();
        assert!(!token_gate.passed);
        assert!(token_gate.detail.contains("unavailable"));
    }

    #[test]
    fn validation_bounds_active_load() {
        let mut options = CanaryOptions {
            base_url: "http://localhost:8000/v1".to_owned(),
            model: "model".to_owned(),
            ..CanaryOptions::default()
        };
        assert!(validate(&options).is_ok());
        options.concurrency = 2;
        assert!(validate(&options).is_err());
        options.concurrency = 1;
        options.count = 10_000;
        options.max_tokens = 101;
        assert!(validate(&options).is_err());

        options.count = 9;
        options.concurrency = 9;
        options.max_tokens = 1;
        options.max_body_bytes = 8 << 20;
        assert!(validate(&options).is_err());
    }

    #[test]
    fn memory_admission_counts_persistent_worker_and_retained_allocations() {
        let options = CanaryOptions {
            base_url: "http://localhost:8000/v1".to_owned(),
            model: "model".to_owned(),
            api_key: Some("token".to_owned()),
            prompt: "prompt".to_owned(),
            expectation: Some("ready".to_owned()),
            max_tokens: 1,
            count: 2,
            concurrency: 2,
            max_body_bytes: 100,
            ..CanaryOptions::default()
        };
        let input_bytes = options.base_url.len()
            + options.model.len()
            + options.workload_id.len()
            + options.api_key.as_ref().unwrap().len()
            + options.prompt.len()
            + options.expectation.as_ref().unwrap().len();
        let request =
            request_body_memory_budget(options.model.len(), options.prompt.len()).unwrap();
        let worker = worker_memory_budget(
            request,
            options.max_body_bytes,
            options.expectation.as_ref().unwrap().len(),
            options.stream,
        )
        .unwrap();
        let attempt = u64::try_from(std::mem::size_of::<Option<CanaryAttempt>>()).unwrap()
            + RETAINED_ATTEMPT_HEAP_BUDGET_BYTES
            + AGGREGATION_BYTES_PER_ATTEMPT;
        let expected = u64::try_from(input_bytes).unwrap() * INPUT_COPY_BUDGET
            + request
            + u64::from(options.count) * attempt
            + u64::from(options.concurrency) * worker;

        assert_eq!(estimated_canary_memory_bytes(&options), Some(expected));
    }

    #[test]
    fn memory_admission_allows_one_maximum_response_but_rejects_parallel_parser_peaks() {
        let mut options = CanaryOptions {
            base_url: "http://localhost:8000/v1".to_owned(),
            model: "model".to_owned(),
            max_tokens: 1,
            max_body_bytes: MAX_RESPONSE_BYTES,
            ..CanaryOptions::default()
        };
        assert!(validate(&options).is_ok());

        options.count = 2;
        options.concurrency = 2;
        let error = validate(&options).unwrap_err().to_string();
        assert!(error.contains("estimated working set"));
    }

    #[test]
    fn memory_admission_bounds_default_response_concurrency() {
        let mut options = CanaryOptions {
            base_url: "http://localhost:8000/v1".to_owned(),
            model: "model".to_owned(),
            max_tokens: 1,
            count: 11,
            concurrency: 11,
            ..CanaryOptions::default()
        };
        assert!(validate(&options).is_ok());

        options.count = 12;
        options.concurrency = 12;
        assert!(validate(&options).is_err());
    }

    #[test]
    fn validation_bounds_library_api_keys() {
        let options = CanaryOptions {
            base_url: "http://localhost:8000/v1".to_owned(),
            model: "model".to_owned(),
            api_key: Some("x".repeat(MAX_API_KEY_BYTES + 1)),
            ..CanaryOptions::default()
        };

        assert!(validate(&options).is_err());
    }

    #[test]
    fn validation_rejects_unreportable_workload_identity() {
        let options = CanaryOptions {
            base_url: "http://localhost:8000/v1".to_owned(),
            model: "model".to_owned(),
            workload_id: "contains whitespace".to_owned(),
            ..CanaryOptions::default()
        };

        assert!(validate(&options).is_err());
    }

    #[test]
    fn non_stream_mode_rejects_stream_only_gates() {
        let options = CanaryOptions {
            base_url: "http://localhost:8000/v1".to_owned(),
            model: "model".to_owned(),
            stream: false,
            thresholds: CanaryThresholds {
                max_ttft: Some(Duration::from_millis(100)),
                ..CanaryThresholds::default()
            },
            ..CanaryOptions::default()
        };

        assert!(validate(&options).is_err());
    }

    #[test]
    fn validation_caps_request_and_planned_wall_time() {
        let mut options = CanaryOptions {
            base_url: "http://localhost:8000/v1".to_owned(),
            model: "model".to_owned(),
            timeout: Duration::from_secs(5 * 60 + 1),
            ..CanaryOptions::default()
        };
        assert!(validate(&options).is_err());

        options.timeout = Duration::from_secs(30);
        options.count = 31;
        assert!(validate(&options).is_err());

        options.concurrency = 2;
        assert!(validate(&options).is_ok());
    }

    #[test]
    fn sub_millisecond_plan_durations_round_up() {
        assert_eq!(duration_millis_ceil(Duration::from_nanos(1)), 1);
        assert_eq!(duration_millis_ceil(Duration::from_millis(1)), 1);
    }

    #[test]
    fn zero_percent_policy_still_requires_one_successful_request() {
        let attempt = CanaryAttempt::failed(
            0,
            crate::domain::CanaryFailureStage::Transport,
            "connection failed",
        );
        let summary = summarize(std::slice::from_ref(&attempt), Duration::from_secs(1));
        let gates = evaluate_gates(
            &summary,
            &[attempt],
            &CanaryThresholds {
                min_success_percent: 0.0,
                ..CanaryThresholds::default()
            },
        );

        assert!(
            !gates
                .iter()
                .find(|gate| gate.name == "min_successful_requests")
                .unwrap()
                .passed
        );
    }

    #[test]
    fn bounded_workers_preserve_attempt_order() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let active = AtomicUsize::new(0);
        let maximum = AtomicUsize::new(0);
        let execution = execute_batch(8, 3, |index| {
            let current = active.fetch_add(1, Ordering::SeqCst) + 1;
            maximum.fetch_max(current, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(
                u64::try_from(8 - index).unwrap_or(u64::MAX),
            ));
            active.fetch_sub(1, Ordering::SeqCst);
            successful_attempt(u32::try_from(index).unwrap_or(u32::MAX), 10.0, 20.0)
        })
        .unwrap();

        assert_eq!(
            execution
                .results
                .iter()
                .map(|attempt| attempt.index)
                .collect::<Vec<_>>(),
            (0..8).collect::<Vec<_>>()
        );
        assert!((2..=3).contains(&maximum.load(Ordering::SeqCst)));
    }
}
