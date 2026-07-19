//! Bounded closed-loop saturation benchmarking and exact-stage deployment gates.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use chrono::Utc;

use crate::domain::{
    CanaryAttempt, CanaryDistribution, CanaryFailureStage, SATURATION_BENCHMARK_NONCLAIMS,
    SATURATION_BENCHMARK_VERSION, SaturationAbortReason, SaturationAssessment,
    SaturationAssessmentStatus, SaturationAttempt, SaturationBenchmarkReport,
    SaturationFailureStageCounts, SaturationGate, SaturationGateKind, SaturationGateOperator,
    SaturationGateReason, SaturationGateStatus, SaturationLoadModel, SaturationPhaseResult,
    SaturationPhaseStatus, SaturationPlan, SaturationPolicy, SaturationRoute, SaturationRunStatus,
    SaturationScalingEvidence, SaturationSchedule, SaturationSignal, SaturationStageOrder,
    SaturationStageResult, SaturationStageSummary, SaturationTarget, SaturationVerification,
    SaturationVerificationReason, SaturationVerificationStatus, SaturationWarmupScope,
    SaturationWorkerStart, valid_canary_workload_id,
};
use crate::inference::{
    MAX_REPORTED_PROMPT_TOKENS_PER_REQUEST, OpenAiClient, OpenAiClientOptions,
    request_body_memory_budget, worker_memory_budget,
};
use crate::operations::canary::{
    DEFAULT_CANARY_EXPECTATION, DEFAULT_CANARY_PROMPT, DEFAULT_CANARY_WORKLOAD_ID, distribution,
    distribution_evidence_matches, within_ulps,
};
use crate::operations::workload::execute_batch;

pub const DEFAULT_RESPONSE_LIMIT_BYTES: usize = 128 << 10;
pub const DEFAULT_MAX_TOKENS: u32 = 128;
pub const DEFAULT_WARMUP_REQUESTS_PER_WORKER: u32 = 2;
pub const DEFAULT_REQUESTS_PER_WORKER: u32 = 20;
pub const MINIMUM_LATENCY_SAMPLES: u32 = 20;

const MAX_STAGES: usize = 8;
const MAX_CONCURRENCY: u32 = 64;
const MAX_WARMUP_REQUESTS_PER_WORKER: u32 = 10;
const MAX_REQUESTS_PER_WORKER: u32 = 100;
const MAX_ATTEMPTS: u64 = 10_000;
const MAX_COMPLETION_TOKENS: u32 = 65_536;
const MAX_REQUESTED_COMPLETION_TOKENS: u64 = 1_000_000;
const MAX_PROMPT_BYTES: usize = 1 << 20;
const MAX_EXPECTATION_BYTES: usize = 64 << 10;
const MAX_IDENTITY_BYTES: usize = 64 << 10;
const MAX_API_KEY_BYTES: usize = 64 << 10;
const MAX_RESPONSE_BYTES: usize = 8 << 20;
const MAX_TOTAL_PROMPT_BYTES: u64 = 64 << 20;
const MAX_TOTAL_RESPONSE_LIMIT_BYTES: u64 = 2 << 30;
const MAX_AGGREGATE_MEMORY_BYTES: u64 = 768 << 20;
const RETAINED_ATTEMPT_BUDGET_BYTES: u64 = 1_024;
const INPUT_COPY_BUDGET: u64 = 3;
const MAX_REQUEST_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const MAX_PLANNED_DURATION: Duration = Duration::from_secs(30 * 60);
const SIGNAL_MAX_MARGINAL_SCALING_EFFICIENCY_PERCENT: f64 = 5.0;
const SIGNAL_MIN_P95_LATENCY_INFLATION_PERCENT: f64 = 20.0;

/// Optional gates and the load-abort guard for a saturation run.
#[derive(Debug, Clone)]
pub struct SaturationThresholds {
    pub max_error_percent: f64,
    pub max_p95_ttft: Option<Duration>,
    pub max_p95_e2e: Option<Duration>,
    pub min_successful_requests_per_second: Option<f64>,
    pub min_completion_token_goodput_per_second: Option<f64>,
    pub abort_error_percent: f64,
}

impl Default for SaturationThresholds {
    fn default() -> Self {
        Self {
            max_error_percent: 1.0,
            max_p95_ttft: None,
            max_p95_e2e: None,
            min_successful_requests_per_second: None,
            min_completion_token_goodput_per_second: None,
            abort_error_percent: 50.0,
        }
    }
}

/// Complete input for one bounded single-process saturation benchmark.
///
/// `Debug` is intentionally omitted because this type owns request content and
/// an optional credential.
#[derive(Clone)]
pub struct SaturationOptions {
    pub base_url: String,
    pub model: String,
    pub workload_id: String,
    pub api_key: Option<String>,
    pub prompt: String,
    pub expectation: Option<String>,
    pub concurrency_stages: Vec<u32>,
    pub warmup_requests_per_worker: u32,
    pub requests_per_worker: u32,
    pub max_tokens: u32,
    pub timeout: Duration,
    pub max_body_bytes: usize,
    pub stream: bool,
    pub allow_insecure_http: bool,
    pub verify_concurrency: Option<u32>,
    pub thresholds: SaturationThresholds,
}

impl Default for SaturationOptions {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            model: String::new(),
            workload_id: DEFAULT_CANARY_WORKLOAD_ID.to_owned(),
            api_key: None,
            prompt: DEFAULT_CANARY_PROMPT.to_owned(),
            expectation: Some(DEFAULT_CANARY_EXPECTATION.to_owned()),
            concurrency_stages: vec![1],
            warmup_requests_per_worker: DEFAULT_WARMUP_REQUESTS_PER_WORKER,
            requests_per_worker: DEFAULT_REQUESTS_PER_WORKER,
            max_tokens: DEFAULT_MAX_TOKENS,
            timeout: Duration::from_secs(10),
            max_body_bytes: DEFAULT_RESPONSE_LIMIT_BYTES,
            stream: true,
            allow_insecure_http: false,
            verify_concurrency: None,
            thresholds: SaturationThresholds::default(),
        }
    }
}

/// Execute bounded excluded warmups and measurements at every explicit stage.
///
/// Invalid local configuration fails before any request is sent. Once active
/// traffic begins, a safety abort still returns a complete report so operators
/// retain the bounded evidence that caused the stop.
pub fn run(options: &SaturationOptions) -> Result<SaturationBenchmarkReport> {
    let admission = validate(options)?;
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
    let canary_target = client.target();
    let target = SaturationTarget {
        url: canary_target.url,
        route: SaturationRoute::ChatCompletions,
        model: canary_target.model,
        stream: canary_target.stream,
    };
    let policy = policy(options);
    let started_at = Utc::now();
    let started = Instant::now();
    let (warmups, stages, abort_reason) = execute_schedule(options, &client, &policy)?;

    let status = if abort_reason.is_some() {
        SaturationRunStatus::Aborted
    } else {
        SaturationRunStatus::Complete
    };
    let assessment = assess(&stages, &policy);
    let verification = verify(options.verify_concurrency, &stages);
    let duration = started.elapsed();

    Ok(SaturationBenchmarkReport {
        saturation_benchmark_version: SATURATION_BENCHMARK_VERSION,
        started_at,
        duration_ns: elapsed_nanos(duration),
        duration_ms: elapsed_millis(duration),
        status,
        abort_reason,
        workload_id: options.workload_id.clone(),
        target,
        plan: SaturationPlan {
            concurrency_stages: options.concurrency_stages.clone(),
            warmup_requests_per_worker: options.warmup_requests_per_worker,
            requests_per_worker: options.requests_per_worker,
            planned_attempts: u32::try_from(admission.planned_attempts).unwrap_or(u32::MAX),
            max_tokens: options.max_tokens,
            timeout_ns: elapsed_nanos(options.timeout),
            timeout_ms: duration_millis_ceil(options.timeout),
            response_limit_bytes: options.max_body_bytes,
            schedule: SaturationSchedule {
                load_model: SaturationLoadModel::ClosedLoopFixedConcurrency,
                stage_order: SaturationStageOrder::ExplicitAscending,
                warmup_scope: SaturationWarmupScope::EachStageExcluded,
                worker_start: SaturationWorkerStart::SimultaneousBarrier,
            },
        },
        policy,
        warmups,
        stages,
        assessment,
        verification,
        nonclaims: SATURATION_BENCHMARK_NONCLAIMS.to_vec(),
    })
}

fn execute_warmup(
    options: &SaturationOptions,
    client: &Arc<OpenAiClient>,
    policy: &SaturationPolicy,
    concurrency: u32,
) -> Result<(SaturationPhaseResult, Option<SaturationAbortReason>)> {
    let count = concurrency
        .checked_mul(options.warmup_requests_per_worker)
        .expect("validated warmup count must fit u32");
    let execution = execute_batch(
        usize::try_from(count).unwrap_or(usize::MAX),
        usize::try_from(concurrency).unwrap_or(usize::MAX),
        |index| client.execute(u32::try_from(index).unwrap_or(u32::MAX)),
    )?;
    let mut warmup = summarize_warmup(&execution.results, concurrency, count, execution.duration);
    let abort_reason = warmup_abort_reason(&warmup, policy.abort_error_percent);
    warmup.status = if abort_reason.is_some() {
        SaturationPhaseStatus::Aborted
    } else {
        SaturationPhaseStatus::Complete
    };
    Ok((warmup, abort_reason))
}

fn execute_schedule(
    options: &SaturationOptions,
    client: &Arc<OpenAiClient>,
    policy: &SaturationPolicy,
) -> Result<(
    Vec<SaturationPhaseResult>,
    Vec<SaturationStageResult>,
    Option<SaturationAbortReason>,
)> {
    let mut warmups = Vec::with_capacity(options.concurrency_stages.len());
    let mut stages = Vec::with_capacity(options.concurrency_stages.len());
    for &concurrency in &options.concurrency_stages {
        let (warmup, warmup_abort) = execute_warmup(options, client, policy, concurrency)?;
        warmups.push(warmup);
        if warmup_abort.is_some() {
            return Ok((warmups, stages, warmup_abort));
        }
        let planned_requests = concurrency
            .checked_mul(options.requests_per_worker)
            .expect("validated measured count must fit u32");
        let execution = execute_batch(
            usize::try_from(planned_requests).unwrap_or(usize::MAX),
            usize::try_from(concurrency).unwrap_or(usize::MAX),
            |index| client.execute(u32::try_from(index).unwrap_or(u32::MAX)),
        )?;
        let attempts = execution
            .results
            .into_iter()
            .map(SaturationAttempt::from)
            .collect::<Vec<_>>();
        let summary = summarize_stage(&attempts, execution.duration, options.max_tokens);
        let gates = canonical_gates(&summary, policy);
        let abort_reason = stage_abort_reason(&summary, policy.abort_error_percent);
        stages.push(SaturationStageResult {
            status: SaturationPhaseStatus::Complete,
            concurrency,
            planned_requests,
            duration_ns: elapsed_nanos(execution.duration),
            duration_ms: elapsed_millis(execution.duration),
            summary,
            gates,
            attempts,
        });
        if abort_reason.is_some() {
            return Ok((warmups, stages, abort_reason));
        }
    }
    Ok((warmups, stages, None))
}

#[derive(Debug, Clone, Copy)]
struct Admission {
    planned_attempts: u64,
}

fn validate(options: &SaturationOptions) -> Result<Admission> {
    validate_identity_and_request(options)?;
    validate_schedule(options)?;
    validate_thresholds(options)?;

    let concurrency_sum = options
        .concurrency_stages
        .iter()
        .try_fold(0_u64, |sum, value| sum.checked_add(u64::from(*value)))
        .ok_or_else(|| anyhow::anyhow!("benchmark concurrency sum overflowed"))?;
    let attempts_per_concurrency = u64::from(options.warmup_requests_per_worker)
        .checked_add(u64::from(options.requests_per_worker))
        .ok_or_else(|| anyhow::anyhow!("benchmark requests-per-worker sum overflowed"))?;
    let planned_attempts = concurrency_sum
        .checked_mul(attempts_per_concurrency)
        .ok_or_else(|| anyhow::anyhow!("benchmark total request count overflowed"))?;
    if planned_attempts > MAX_ATTEMPTS {
        bail!("benchmark warmup and measured attempts must not exceed {MAX_ATTEMPTS}");
    }
    let requested_tokens = planned_attempts
        .checked_mul(u64::from(options.max_tokens))
        .ok_or_else(|| anyhow::anyhow!("benchmark completion-token budget overflowed"))?;
    if requested_tokens > MAX_REQUESTED_COMPLETION_TOKENS {
        bail!(
            "benchmark attempts multiplied by max tokens must not exceed {MAX_REQUESTED_COMPLETION_TOKENS}"
        );
    }
    let prompt_bytes = planned_attempts
        .checked_mul(u64::try_from(options.prompt.len()).unwrap_or(u64::MAX))
        .ok_or_else(|| anyhow::anyhow!("benchmark aggregate prompt budget overflowed"))?;
    if prompt_bytes > MAX_TOTAL_PROMPT_BYTES {
        bail!("benchmark aggregate prompt bytes must not exceed 64 MiB");
    }
    let response_bytes = planned_attempts
        .checked_mul(u64::try_from(options.max_body_bytes).unwrap_or(u64::MAX))
        .ok_or_else(|| anyhow::anyhow!("benchmark aggregate response budget overflowed"))?;
    if response_bytes > MAX_TOTAL_RESPONSE_LIMIT_BYTES {
        bail!("benchmark attempts multiplied by response limit must not exceed 2 GiB");
    }

    let waves = u64::try_from(options.concurrency_stages.len())
        .unwrap_or(u64::MAX)
        .checked_mul(attempts_per_concurrency)
        .ok_or_else(|| anyhow::anyhow!("benchmark planned wave count overflowed"))?;
    let waves = u32::try_from(waves)
        .map_err(|_| anyhow::anyhow!("benchmark planned wave count is too large"))?;
    let planned_duration = options.timeout.checked_mul(waves).unwrap_or(Duration::MAX);
    if planned_duration > MAX_PLANNED_DURATION {
        bail!("benchmark request waves multiplied by timeout must not exceed 30 minutes");
    }
    validate_resource_budget(options, planned_attempts)?;
    Ok(Admission { planned_attempts })
}

fn validate_identity_and_request(options: &SaturationOptions) -> Result<()> {
    if options.base_url.trim().is_empty() || options.base_url.len() > MAX_IDENTITY_BYTES {
        bail!("benchmark base URL must contain 1 byte to 64 KiB");
    }
    if options.model.trim().is_empty() || options.model.len() > MAX_IDENTITY_BYTES {
        bail!("benchmark model must contain 1 byte to 64 KiB");
    }
    if !valid_canary_workload_id(&options.workload_id) {
        bail!("benchmark workload ID is not a bounded privacy-safe identity");
    }
    if options
        .api_key
        .as_ref()
        .is_some_and(|value| value.is_empty() || value.len() > MAX_API_KEY_BYTES)
    {
        bail!("benchmark API key must contain 1 byte to 64 KiB");
    }
    if options.prompt.is_empty() || options.prompt.len() > MAX_PROMPT_BYTES {
        bail!("benchmark prompt must contain 1 byte to 1 MiB");
    }
    if options
        .expectation
        .as_ref()
        .is_some_and(|value| value.is_empty() || value.len() > MAX_EXPECTATION_BYTES)
    {
        bail!("benchmark expectation must contain 1 byte to 64 KiB");
    }
    if options.max_tokens == 0 || options.max_tokens > MAX_COMPLETION_TOKENS {
        bail!("benchmark max tokens must be between 1 and {MAX_COMPLETION_TOKENS}");
    }
    if options.timeout.is_zero() || options.timeout > MAX_REQUEST_TIMEOUT {
        bail!("benchmark timeout must be greater than zero and at most 5 minutes");
    }
    if options.max_body_bytes == 0 || options.max_body_bytes > MAX_RESPONSE_BYTES {
        bail!("benchmark response limit must be between 1 byte and 8 MiB");
    }
    Ok(())
}

fn validate_schedule(options: &SaturationOptions) -> Result<()> {
    if options.concurrency_stages.is_empty() || options.concurrency_stages.len() > MAX_STAGES {
        bail!("benchmark requires between 1 and {MAX_STAGES} concurrency stages");
    }
    if options.concurrency_stages[0] != 1 {
        bail!("benchmark concurrency stages must start at 1 for a bounded correctness warmup");
    }
    if options
        .concurrency_stages
        .iter()
        .any(|value| *value == 0 || *value > MAX_CONCURRENCY)
    {
        bail!("benchmark concurrency stages must be between 1 and {MAX_CONCURRENCY}");
    }
    if options
        .concurrency_stages
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
    {
        bail!("benchmark concurrency stages must be unique and strictly increasing");
    }
    if options.warmup_requests_per_worker == 0
        || options.warmup_requests_per_worker > MAX_WARMUP_REQUESTS_PER_WORKER
    {
        bail!(
            "benchmark warmup requests per worker must be between 1 and {MAX_WARMUP_REQUESTS_PER_WORKER}"
        );
    }
    if options.requests_per_worker == 0 || options.requests_per_worker > MAX_REQUESTS_PER_WORKER {
        bail!(
            "benchmark measured requests per worker must be between 1 and {MAX_REQUESTS_PER_WORKER}"
        );
    }
    if options
        .verify_concurrency
        .is_some_and(|value| !options.concurrency_stages.contains(&value))
    {
        bail!("--verify-concurrency must name one exact --concurrency-stages value");
    }
    Ok(())
}

fn validate_thresholds(options: &SaturationOptions) -> Result<()> {
    let thresholds = &options.thresholds;
    if !thresholds.max_error_percent.is_finite()
        || !(0.0..100.0).contains(&thresholds.max_error_percent)
    {
        bail!("benchmark maximum error percent must be finite and in [0, 100)");
    }
    if !thresholds.abort_error_percent.is_finite()
        || thresholds.abort_error_percent <= thresholds.max_error_percent
        || thresholds.abort_error_percent > 100.0
    {
        bail!(
            "benchmark abort error percent must be finite, above maximum error percent, and at most 100"
        );
    }
    if thresholds
        .max_p95_ttft
        .is_some_and(|value| value.is_zero() || value > MAX_REQUEST_TIMEOUT)
        || thresholds
            .max_p95_e2e
            .is_some_and(|value| value.is_zero() || value > MAX_REQUEST_TIMEOUT)
    {
        bail!("benchmark latency thresholds must be positive and at most 5 minutes");
    }
    if thresholds
        .min_successful_requests_per_second
        .is_some_and(|value| !value.is_finite() || value < 0.0)
        || thresholds
            .min_completion_token_goodput_per_second
            .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        bail!("benchmark goodput thresholds must be finite and non-negative");
    }
    if !options.stream && thresholds.max_p95_ttft.is_some() {
        bail!("benchmark p95 TTFT gate requires streaming responses");
    }
    Ok(())
}

fn validate_resource_budget(options: &SaturationOptions, planned_attempts: u64) -> Result<()> {
    let expectation_bytes = options.expectation.as_ref().map_or(0, String::len);
    let api_key_bytes = options.api_key.as_ref().map_or(0, String::len);
    let input_bytes = u64::try_from(options.base_url.len())
        .ok()
        .and_then(|value| value.checked_add(u64::try_from(options.model.len()).ok()?))
        .and_then(|value| value.checked_add(u64::try_from(options.workload_id.len()).ok()?))
        .and_then(|value| value.checked_add(u64::try_from(options.prompt.len()).ok()?))
        .and_then(|value| value.checked_add(u64::try_from(expectation_bytes).ok()?))
        .and_then(|value| value.checked_add(u64::try_from(api_key_bytes).ok()?))
        .ok_or_else(|| anyhow::anyhow!("benchmark input memory estimate overflowed"))?;
    let request_body_bytes = request_body_memory_budget(options.model.len(), options.prompt.len())
        .ok_or_else(|| anyhow::anyhow!("benchmark request memory estimate overflowed"))?;
    let worker = worker_memory_budget(
        request_body_bytes,
        options.max_body_bytes,
        expectation_bytes,
        options.stream,
    )
    .ok_or_else(|| anyhow::anyhow!("benchmark worker memory estimate overflowed"))?;
    let retained = planned_attempts
        .checked_mul(RETAINED_ATTEMPT_BUDGET_BYTES)
        .ok_or_else(|| anyhow::anyhow!("benchmark retained-evidence estimate overflowed"))?;
    let persistent = input_bytes
        .checked_mul(INPUT_COPY_BUDGET)
        .and_then(|value| value.checked_add(request_body_bytes))
        .and_then(|value| value.checked_add(retained))
        .ok_or_else(|| anyhow::anyhow!("benchmark persistent memory estimate overflowed"))?;
    let max_concurrency = u64::from(*options.concurrency_stages.last().unwrap_or(&0));
    let estimate = max_concurrency
        .checked_mul(worker)
        .and_then(|value| value.checked_add(persistent))
        .ok_or_else(|| anyhow::anyhow!("benchmark working-set estimate overflowed"))?;
    if estimate > MAX_AGGREGATE_MEMORY_BYTES {
        bail!(
            "benchmark estimated working set ({estimate} bytes) must not exceed {MAX_AGGREGATE_MEMORY_BYTES} bytes; reduce concurrency, response limit, prompt, or expectation size"
        );
    }
    Ok(())
}

fn policy(options: &SaturationOptions) -> SaturationPolicy {
    SaturationPolicy {
        max_error_percent: options.thresholds.max_error_percent,
        max_p95_ttft_ms: options
            .thresholds
            .max_p95_ttft
            .map(|value| value.as_secs_f64() * 1_000.0),
        max_p95_e2e_ms: options
            .thresholds
            .max_p95_e2e
            .map(|value| value.as_secs_f64() * 1_000.0),
        min_successful_requests_per_second: options.thresholds.min_successful_requests_per_second,
        min_completion_token_goodput_per_second: options
            .thresholds
            .min_completion_token_goodput_per_second,
        abort_error_percent: options.thresholds.abort_error_percent,
        minimum_latency_samples: MINIMUM_LATENCY_SAMPLES,
        signal_max_marginal_scaling_efficiency_percent:
            SIGNAL_MAX_MARGINAL_SCALING_EFFICIENCY_PERCENT,
        signal_min_p95_latency_inflation_percent: SIGNAL_MIN_P95_LATENCY_INFLATION_PERCENT,
        expectation_configured: options.expectation.is_some(),
    }
}

fn summarize_warmup(
    attempts: &[CanaryAttempt],
    concurrency: u32,
    planned_requests: u32,
    duration: Duration,
) -> SaturationPhaseResult {
    let attempted = u32::try_from(attempts.len()).unwrap_or(u32::MAX);
    let succeeded = u32::try_from(attempts.iter().filter(|attempt| attempt.success).count())
        .unwrap_or(u32::MAX);
    let failed = attempted.saturating_sub(succeeded);
    let safe_attempts = attempts
        .iter()
        .map(SaturationAttempt::from)
        .collect::<Vec<_>>();
    SaturationPhaseResult {
        status: SaturationPhaseStatus::Complete,
        concurrency,
        planned_requests,
        attempted,
        succeeded,
        failed,
        error_percent: error_percent(failed, attempted),
        duration_ns: elapsed_nanos(duration),
        duration_ms: elapsed_millis(duration),
        failure_stage_counts: SaturationFailureStageCounts::from_attempts(&safe_attempts),
    }
}

pub(crate) fn summarize_stage(
    attempts: &[SaturationAttempt],
    duration: Duration,
    max_tokens: u32,
) -> SaturationStageSummary {
    let attempted = u32::try_from(attempts.len()).unwrap_or(u32::MAX);
    let succeeded_attempts = attempts
        .iter()
        .filter(|attempt| attempt.success)
        .collect::<Vec<_>>();
    let succeeded = u32::try_from(succeeded_attempts.len()).unwrap_or(u32::MAX);
    let failed = attempted.saturating_sub(succeeded);
    let seconds = duration.as_secs_f64();
    let attempted_rps = rate(u64::from(attempted), seconds);
    let successful_rps = rate(u64::from(succeeded), seconds);
    let prompt_token_samples = u32::try_from(
        succeeded_attempts
            .iter()
            .filter(|attempt| {
                attempt
                    .prompt_tokens
                    .is_some_and(valid_reported_prompt_tokens)
            })
            .count(),
    )
    .unwrap_or(u32::MAX);
    let completion_token_samples = u32::try_from(
        succeeded_attempts
            .iter()
            .filter(|attempt| {
                attempt
                    .completion_tokens
                    .is_some_and(|tokens| valid_reported_completion_tokens(tokens, max_tokens))
            })
            .count(),
    )
    .unwrap_or(u32::MAX);
    let prompt_tokens_observed_total = succeeded_attempts
        .iter()
        .filter_map(|attempt| attempt.prompt_tokens)
        .filter(|tokens| valid_reported_prompt_tokens(*tokens))
        .try_fold(0_u64, u64::checked_add)
        .unwrap_or(0);
    let completion_tokens_observed_total = succeeded_attempts
        .iter()
        .filter_map(|attempt| attempt.completion_tokens)
        .filter(|tokens| valid_reported_completion_tokens(*tokens, max_tokens))
        .try_fold(0_u64, u64::checked_add)
        .unwrap_or(0);
    let total_tokens_observed_total = prompt_tokens_observed_total
        .checked_add(completion_tokens_observed_total)
        .unwrap_or(0);
    let completion_token_usage_complete = succeeded > 0 && completion_token_samples == succeeded;
    let total_token_usage_complete =
        completion_token_usage_complete && prompt_token_samples == succeeded;
    SaturationStageSummary {
        attempted,
        succeeded,
        failed,
        error_percent: error_percent(failed, attempted),
        attempted_requests_per_second: attempted_rps,
        successful_requests_per_second: successful_rps,
        prompt_token_samples,
        completion_token_samples,
        prompt_tokens_observed_total,
        completion_tokens_observed_total,
        total_tokens_observed_total,
        completion_token_usage_complete,
        total_token_usage_complete,
        completion_token_goodput_per_second: completion_token_usage_complete
            .then(|| rate(completion_tokens_observed_total, seconds)),
        total_token_goodput_per_second: total_token_usage_complete
            .then(|| rate(total_tokens_observed_total, seconds)),
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
        failure_stage_counts: SaturationFailureStageCounts::from_attempts(attempts),
    }
}

fn valid_reported_prompt_tokens(tokens: u64) -> bool {
    (1..=MAX_REPORTED_PROMPT_TOKENS_PER_REQUEST).contains(&tokens)
}

fn valid_reported_completion_tokens(tokens: u64, max_tokens: u32) -> bool {
    (1..=u64::from(max_tokens)).contains(&tokens)
}

pub(crate) fn canonical_gates(
    summary: &SaturationStageSummary,
    policy: &SaturationPolicy,
) -> Vec<SaturationGate> {
    let mut gates = vec![
        SaturationGate {
            kind: SaturationGateKind::SuccessfulRequests,
            operator: SaturationGateOperator::GreaterThanOrEqual,
            observed: Some(f64::from(summary.succeeded)),
            threshold: 1.0,
            status: pass_or_fail(summary.succeeded > 0),
            reason: None,
            samples: None,
            required_samples: None,
        },
        SaturationGate {
            kind: SaturationGateKind::ErrorPercent,
            operator: SaturationGateOperator::LessThanOrEqual,
            observed: Some(summary.error_percent),
            threshold: policy.max_error_percent,
            status: pass_or_fail(summary.error_percent <= policy.max_error_percent),
            reason: None,
            samples: None,
            required_samples: None,
        },
    ];
    if let Some(threshold) = policy.max_p95_ttft_ms {
        gates.push(latency_gate(
            SaturationGateKind::P95TtftMs,
            summary.ttft_ms.as_ref(),
            summary.succeeded,
            threshold,
            policy.minimum_latency_samples,
        ));
    }
    if let Some(threshold) = policy.max_p95_e2e_ms {
        gates.push(latency_gate(
            SaturationGateKind::P95E2eMs,
            summary.e2e_ms.as_ref(),
            summary.succeeded,
            threshold,
            policy.minimum_latency_samples,
        ));
    }
    if let Some(threshold) = policy.min_successful_requests_per_second {
        gates.push(SaturationGate {
            kind: SaturationGateKind::SuccessfulRequestsPerSecond,
            operator: SaturationGateOperator::GreaterThanOrEqual,
            observed: Some(summary.successful_requests_per_second),
            threshold,
            status: pass_or_fail(summary.successful_requests_per_second >= threshold),
            reason: None,
            samples: None,
            required_samples: None,
        });
    }
    if let Some(threshold) = policy.min_completion_token_goodput_per_second {
        let observed = summary.completion_token_goodput_per_second;
        let (status, reason) = if summary.succeeded == 0 {
            (
                SaturationGateStatus::NotEvaluable,
                Some(SaturationGateReason::NoSuccessfulRequests),
            )
        } else if !summary.completion_token_usage_complete {
            (
                SaturationGateStatus::NotEvaluable,
                Some(SaturationGateReason::IncompleteUsageEvidence),
            )
        } else {
            (
                pass_or_fail(observed.is_some_and(|value| value >= threshold)),
                None,
            )
        };
        gates.push(SaturationGate {
            kind: SaturationGateKind::CompletionTokenGoodputPerSecond,
            operator: SaturationGateOperator::GreaterThanOrEqual,
            observed,
            threshold,
            status,
            reason,
            samples: Some(summary.completion_token_samples),
            required_samples: Some(summary.succeeded),
        });
    }
    gates
}

fn latency_gate(
    kind: SaturationGateKind,
    distribution: Option<&CanaryDistribution>,
    succeeded: u32,
    threshold: f64,
    minimum_samples: u32,
) -> SaturationGate {
    let samples = distribution.map_or(0, |value| u32::try_from(value.samples).unwrap_or(u32::MAX));
    let observed = distribution.map(|value| value.p95);
    let (status, reason) = if succeeded == 0 {
        (
            SaturationGateStatus::NotEvaluable,
            Some(SaturationGateReason::NoSuccessfulRequests),
        )
    } else if samples < minimum_samples {
        (
            SaturationGateStatus::NotEvaluable,
            Some(SaturationGateReason::InsufficientSamples),
        )
    } else {
        (
            pass_or_fail(observed.is_some_and(|value| value <= threshold)),
            None,
        )
    };
    SaturationGate {
        kind,
        operator: SaturationGateOperator::LessThanOrEqual,
        observed,
        threshold,
        status,
        reason,
        samples: Some(samples),
        required_samples: Some(minimum_samples),
    }
}

const fn pass_or_fail(passed: bool) -> SaturationGateStatus {
    if passed {
        SaturationGateStatus::Pass
    } else {
        SaturationGateStatus::Fail
    }
}

fn warmup_abort_reason(
    warmup: &SaturationPhaseResult,
    abort_error_percent: f64,
) -> Option<SaturationAbortReason> {
    if warmup.succeeded == 0 {
        Some(SaturationAbortReason::WarmupNoSuccessfulRequests)
    } else if warmup.error_percent >= abort_error_percent {
        Some(SaturationAbortReason::WarmupErrorRateLimitExceeded)
    } else {
        None
    }
}

fn stage_abort_reason(
    summary: &SaturationStageSummary,
    abort_error_percent: f64,
) -> Option<SaturationAbortReason> {
    if summary.succeeded == 0 {
        Some(SaturationAbortReason::StageNoSuccessfulRequests)
    } else if summary.error_percent >= abort_error_percent {
        Some(SaturationAbortReason::StageErrorRateLimitExceeded)
    } else {
        None
    }
}

pub(crate) fn verify(
    requested_concurrency: Option<u32>,
    stages: &[SaturationStageResult],
) -> SaturationVerification {
    let Some(requested_concurrency) = requested_concurrency else {
        return SaturationVerification::default();
    };
    let Some(stage) = stages
        .iter()
        .find(|stage| stage.concurrency == requested_concurrency)
    else {
        return SaturationVerification {
            requested_concurrency: Some(requested_concurrency),
            status: SaturationVerificationStatus::NotEvaluable,
            reason: Some(SaturationVerificationReason::StageNotRun),
        };
    };
    if stage
        .gates
        .iter()
        .any(|gate| gate.status == SaturationGateStatus::Fail)
    {
        SaturationVerification {
            requested_concurrency: Some(requested_concurrency),
            status: SaturationVerificationStatus::Fail,
            reason: None,
        }
    } else if stage
        .gates
        .iter()
        .any(|gate| gate.status == SaturationGateStatus::NotEvaluable)
    {
        SaturationVerification {
            requested_concurrency: Some(requested_concurrency),
            status: SaturationVerificationStatus::NotEvaluable,
            reason: Some(SaturationVerificationReason::GateNotEvaluable),
        }
    } else {
        SaturationVerification {
            requested_concurrency: Some(requested_concurrency),
            status: SaturationVerificationStatus::Pass,
            reason: None,
        }
    }
}

pub(crate) fn assess(
    stages: &[SaturationStageResult],
    policy: &SaturationPolicy,
) -> SaturationAssessment {
    let highest_accepted_tested_concurrency = highest_accepted_concurrency(stages);
    if stages.len() < 2 {
        return SaturationAssessment {
            status: SaturationAssessmentStatus::NotEvaluable,
            signal: None,
            first_signal_concurrency: None,
            highest_accepted_tested_concurrency,
            scaling_evidence: Vec::new(),
        };
    }

    let baseline = &stages[0];
    let mut scaling_evidence = Vec::with_capacity(stages.len() - 1);
    let mut signal = None;
    let mut evidence_gap = false;
    for pair in stages.windows(2) {
        let previous = &pair[0];
        let current = &pair[1];
        let marginal_rps_gain = percent_change(
            previous.summary.successful_requests_per_second,
            current.summary.successful_requests_per_second,
        );
        let marginal_scaling_efficiency = marginal_scaling_efficiency(previous, current);
        let scaling_efficiency = scaling_efficiency(baseline, current);
        let ttft_inflation = distribution_inflation(
            previous.summary.ttft_ms.as_ref(),
            current.summary.ttft_ms.as_ref(),
            policy.minimum_latency_samples,
        );
        let e2e_inflation = distribution_inflation(
            previous.summary.e2e_ms.as_ref(),
            current.summary.e2e_ms.as_ref(),
            policy.minimum_latency_samples,
        );
        scaling_evidence.push(SaturationScalingEvidence {
            baseline_concurrency: baseline.concurrency,
            previous_concurrency: previous.concurrency,
            concurrency: current.concurrency,
            marginal_successful_rps_gain_percent: marginal_rps_gain,
            marginal_scaling_efficiency_percent: marginal_scaling_efficiency,
            scaling_efficiency_percent: scaling_efficiency,
            p95_ttft_inflation_percent: ttft_inflation,
            p95_e2e_inflation_percent: e2e_inflation,
        });
        let Some(marginal_scaling_efficiency) = marginal_scaling_efficiency else {
            evidence_gap = true;
            continue;
        };
        if marginal_scaling_efficiency > policy.signal_max_marginal_scaling_efficiency_percent {
            continue;
        }
        if current.summary.error_percent > policy.max_error_percent {
            if signal.is_none() {
                signal = Some((
                    SaturationSignal::ThroughputPlateauWithErrorRate,
                    current.concurrency,
                ));
            }
            continue;
        }
        let has_latency_evidence = ttft_inflation.is_some() || e2e_inflation.is_some();
        let latency_signal = ttft_inflation
            .is_some_and(|value| value >= policy.signal_min_p95_latency_inflation_percent)
            || e2e_inflation
                .is_some_and(|value| value >= policy.signal_min_p95_latency_inflation_percent);
        if latency_signal {
            if signal.is_none() {
                signal = Some((
                    SaturationSignal::ThroughputPlateauWithLatencyInflation,
                    current.concurrency,
                ));
            }
            continue;
        }
        if !has_latency_evidence {
            evidence_gap = true;
        }
    }

    let (status, signal, first_signal_concurrency) = if let Some((signal, concurrency)) = signal {
        (
            SaturationAssessmentStatus::SignalObserved,
            Some(signal),
            Some(concurrency),
        )
    } else if evidence_gap {
        (SaturationAssessmentStatus::NotEvaluable, None, None)
    } else {
        (
            SaturationAssessmentStatus::NoSignalInTestedStages,
            None,
            None,
        )
    };
    SaturationAssessment {
        status,
        signal,
        first_signal_concurrency,
        highest_accepted_tested_concurrency,
        scaling_evidence,
    }
}

fn highest_accepted_concurrency(stages: &[SaturationStageResult]) -> Option<u32> {
    stages
        .iter()
        .filter(|stage| {
            !stage.gates.is_empty()
                && stage
                    .gates
                    .iter()
                    .all(|gate| gate.status == SaturationGateStatus::Pass)
        })
        .map(|stage| stage.concurrency)
        .max()
}

fn scaling_efficiency(
    baseline: &SaturationStageResult,
    current: &SaturationStageResult,
) -> Option<f64> {
    let baseline_rps = baseline.summary.successful_requests_per_second;
    let current_rps = current.summary.successful_requests_per_second;
    if baseline_rps <= 0.0 || baseline.concurrency == 0 || current.concurrency == 0 {
        return None;
    }
    let throughput_ratio = current_rps / baseline_rps;
    let concurrency_ratio = f64::from(current.concurrency) / f64::from(baseline.concurrency);
    finite((throughput_ratio / concurrency_ratio) * 100.0)
}

fn marginal_scaling_efficiency(
    previous: &SaturationStageResult,
    current: &SaturationStageResult,
) -> Option<f64> {
    let rps_gain = percent_change(
        previous.summary.successful_requests_per_second,
        current.summary.successful_requests_per_second,
    )?;
    let concurrency_gain = percent_change(
        f64::from(previous.concurrency),
        f64::from(current.concurrency),
    )?;
    if concurrency_gain <= 0.0 {
        return None;
    }
    finite((rps_gain / concurrency_gain) * 100.0)
}

fn distribution_inflation(
    previous: Option<&CanaryDistribution>,
    current: Option<&CanaryDistribution>,
    minimum_samples: u32,
) -> Option<f64> {
    let previous = previous?;
    let current = current?;
    let minimum_samples = usize::try_from(minimum_samples).unwrap_or(usize::MAX);
    if previous.samples < minimum_samples || current.samples < minimum_samples {
        return None;
    }
    percent_change(previous.p95, current.p95)
}

fn percent_change(previous: f64, current: f64) -> Option<f64> {
    if !previous.is_finite() || !current.is_finite() || previous <= 0.0 {
        return None;
    }
    finite(((current / previous) - 1.0) * 100.0)
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

#[allow(clippy::cast_precision_loss)]
fn rate(total: u64, seconds: f64) -> f64 {
    if seconds > 0.0 && seconds.is_finite() {
        finite(total as f64 / seconds).unwrap_or(0.0)
    } else {
        0.0
    }
}

fn error_percent(failed: u32, attempted: u32) -> f64 {
    if attempted == 0 {
        0.0
    } else {
        f64::from(failed) * 100.0 / f64::from(attempted)
    }
}

/// Reconstruct and validate all integrity evidence retained in a saved report.
///
/// Errors are deliberately fixed prose so callers never echo attacker-controlled
/// report values. Passing validation proves internal consistency, not report
/// authenticity.
pub(crate) fn validate_report_semantics(
    report: &SaturationBenchmarkReport,
) -> Result<(), &'static str> {
    if report.saturation_benchmark_version != SATURATION_BENCHMARK_VERSION {
        return Err("saturation report version is unsupported");
    }
    if !valid_canary_workload_id(&report.workload_id) {
        return Err("saturation workload identity is invalid");
    }
    validate_saved_target(&report.target)?;
    validate_saved_plan(&report.plan)?;
    validate_saved_policy(&report.policy, report.target.stream)?;
    if report.nonclaims != SATURATION_BENCHMARK_NONCLAIMS {
        return Err("saturation report nonclaims are not canonical");
    }
    if report.duration_ns == 0 || report.duration_ms != report.duration_ns / 1_000_000 {
        return Err("saturation run duration is invalid");
    }

    validate_saved_schedule(report)?;

    let phase_duration = report
        .warmups
        .iter()
        .map(|phase| u128::from(phase.duration_ns))
        .chain(
            report
                .stages
                .iter()
                .map(|stage| u128::from(stage.duration_ns)),
        )
        .try_fold(0_u128, u128::checked_add)
        .ok_or("saturation phase durations overflow")?;
    if u128::from(report.duration_ns) < phase_duration {
        return Err("saturation run duration is shorter than its phases");
    }

    if !assessment_evidence_matches(&report.assessment, &assess(&report.stages, &report.policy)) {
        return Err("saturation assessment is not canonical");
    }
    if report
        .verification
        .requested_concurrency
        .is_some_and(|value| !report.plan.concurrency_stages.contains(&value))
    {
        return Err("saturation verification stage is outside the plan");
    }
    if report.verification != verify(report.verification.requested_concurrency, &report.stages) {
        return Err("saturation verification is not canonical");
    }
    Ok(())
}

fn assessment_evidence_matches(
    retained: &SaturationAssessment,
    rebuilt: &SaturationAssessment,
) -> bool {
    retained.status == rebuilt.status
        && retained.signal == rebuilt.signal
        && retained.first_signal_concurrency == rebuilt.first_signal_concurrency
        && retained.highest_accepted_tested_concurrency
            == rebuilt.highest_accepted_tested_concurrency
        && retained.scaling_evidence.len() == rebuilt.scaling_evidence.len()
        && retained
            .scaling_evidence
            .iter()
            .zip(&rebuilt.scaling_evidence)
            .all(|(retained, rebuilt)| {
                retained.baseline_concurrency == rebuilt.baseline_concurrency
                    && retained.previous_concurrency == rebuilt.previous_concurrency
                    && retained.concurrency == rebuilt.concurrency
                    && option_float_bits_equal(
                        retained.marginal_successful_rps_gain_percent,
                        rebuilt.marginal_successful_rps_gain_percent,
                    )
                    && option_float_bits_equal(
                        retained.marginal_scaling_efficiency_percent,
                        rebuilt.marginal_scaling_efficiency_percent,
                    )
                    && option_float_bits_equal(
                        retained.scaling_efficiency_percent,
                        rebuilt.scaling_efficiency_percent,
                    )
                    && option_float_bits_equal(
                        retained.p95_ttft_inflation_percent,
                        rebuilt.p95_ttft_inflation_percent,
                    )
                    && option_float_bits_equal(
                        retained.p95_e2e_inflation_percent,
                        rebuilt.p95_e2e_inflation_percent,
                    )
            })
}

fn validate_saved_target(target: &SaturationTarget) -> Result<(), &'static str> {
    if target.model.trim().is_empty()
        || target.model.len() > MAX_IDENTITY_BYTES
        || target.url.is_empty()
        || target.url.len() > MAX_IDENTITY_BYTES
        || target.route != SaturationRoute::ChatCompletions
    {
        return Err("saturation target identity is invalid");
    }
    let url = url::Url::parse(&target.url).map_err(|_| "saturation target URL is invalid")?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.origin().ascii_serialization() != target.url
    {
        return Err("saturation target URL is not a canonical redacted origin");
    }
    Ok(())
}

fn validate_saved_plan(plan: &SaturationPlan) -> Result<(), &'static str> {
    if plan.concurrency_stages.is_empty()
        || plan.concurrency_stages.len() > MAX_STAGES
        || plan.concurrency_stages.first() != Some(&1)
        || plan
            .concurrency_stages
            .iter()
            .any(|value| *value == 0 || *value > MAX_CONCURRENCY)
        || plan
            .concurrency_stages
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || !(1..=MAX_WARMUP_REQUESTS_PER_WORKER).contains(&plan.warmup_requests_per_worker)
        || !(1..=MAX_REQUESTS_PER_WORKER).contains(&plan.requests_per_worker)
        || !(1..=MAX_COMPLETION_TOKENS).contains(&plan.max_tokens)
        || plan.timeout_ns == 0
        || plan.timeout_ns > elapsed_nanos(MAX_REQUEST_TIMEOUT)
        || plan.timeout_ms != plan.timeout_ns.div_ceil(1_000_000)
        || plan.response_limit_bytes == 0
        || plan.response_limit_bytes > MAX_RESPONSE_BYTES
        || plan.schedule
            != (SaturationSchedule {
                load_model: SaturationLoadModel::ClosedLoopFixedConcurrency,
                stage_order: SaturationStageOrder::ExplicitAscending,
                warmup_scope: SaturationWarmupScope::EachStageExcluded,
                worker_start: SaturationWorkerStart::SimultaneousBarrier,
            })
    {
        return Err("saturation plan is outside producer safety limits");
    }

    let concurrency_sum = plan
        .concurrency_stages
        .iter()
        .try_fold(0_u64, |sum, value| sum.checked_add(u64::from(*value)))
        .ok_or("saturation plan concurrency sum overflowed")?;
    let requests_per_concurrency = u64::from(plan.warmup_requests_per_worker)
        .checked_add(u64::from(plan.requests_per_worker))
        .ok_or("saturation plan request count overflowed")?;
    let attempts = concurrency_sum
        .checked_mul(requests_per_concurrency)
        .ok_or("saturation plan attempt count overflowed")?;
    if attempts == 0
        || attempts > MAX_ATTEMPTS
        || u64::from(plan.planned_attempts) != attempts
        || attempts
            .checked_mul(u64::from(plan.max_tokens))
            .is_none_or(|tokens| tokens > MAX_REQUESTED_COMPLETION_TOKENS)
        || attempts
            .checked_mul(u64::try_from(plan.response_limit_bytes).unwrap_or(u64::MAX))
            .is_none_or(|bytes| bytes > MAX_TOTAL_RESPONSE_LIMIT_BYTES)
    {
        return Err("saturation plan aggregate bounds are invalid");
    }
    let waves = u64::try_from(plan.concurrency_stages.len())
        .unwrap_or(u64::MAX)
        .checked_mul(requests_per_concurrency)
        .ok_or("saturation plan wave count overflowed")?;
    if u128::from(plan.timeout_ns)
        .checked_mul(u128::from(waves))
        .is_none_or(|duration| duration > MAX_PLANNED_DURATION.as_nanos())
    {
        return Err("saturation plan exceeds the wall-time safety bound");
    }
    Ok(())
}

fn validate_saved_policy(policy: &SaturationPolicy, stream: bool) -> Result<(), &'static str> {
    if !policy.max_error_percent.is_finite()
        || !(0.0..100.0).contains(&policy.max_error_percent)
        || !policy.abort_error_percent.is_finite()
        || policy.abort_error_percent <= policy.max_error_percent
        || policy.abort_error_percent > 100.0
        || policy.minimum_latency_samples != MINIMUM_LATENCY_SAMPLES
        || policy.signal_max_marginal_scaling_efficiency_percent
            != SIGNAL_MAX_MARGINAL_SCALING_EFFICIENCY_PERCENT
        || policy.signal_min_p95_latency_inflation_percent
            != SIGNAL_MIN_P95_LATENCY_INFLATION_PERCENT
    {
        return Err("saturation policy is outside producer bounds");
    }
    if [policy.max_p95_ttft_ms, policy.max_p95_e2e_ms]
        .into_iter()
        .flatten()
        .any(|value| !value.is_finite() || value <= 0.0 || value > 300_000.0)
        || [
            policy.min_successful_requests_per_second,
            policy.min_completion_token_goodput_per_second,
        ]
        .into_iter()
        .flatten()
        .any(|value| !value.is_finite() || value < 0.0)
        || (!stream && policy.max_p95_ttft_ms.is_some())
    {
        return Err("saturation policy contains invalid optional gates");
    }
    Ok(())
}

fn validate_saved_schedule(report: &SaturationBenchmarkReport) -> Result<(), &'static str> {
    let plan = &report.plan;
    let mut terminal_reason = None;
    for (position, concurrency) in plan.concurrency_stages.iter().copied().enumerate() {
        let warmup = report
            .warmups
            .get(position)
            .ok_or("saturation warmups do not form a reached plan prefix")?;
        validate_saved_warmup(warmup, concurrency, plan.warmup_requests_per_worker)?;
        if let Some(reason) = warmup_abort_reason(warmup, report.policy.abort_error_percent) {
            if warmup.status != SaturationPhaseStatus::Aborted
                || report.warmups.len() != position + 1
                || report.stages.len() != position
            {
                return Err("saturation warmup abort sequence is inconsistent");
            }
            terminal_reason = Some(reason);
            break;
        }
        if warmup.status != SaturationPhaseStatus::Complete {
            return Err("saturation non-aborting warmup is not complete");
        }

        let stage = report
            .stages
            .get(position)
            .ok_or("saturation stages do not form a reached plan prefix")?;
        validate_saved_stage(stage, concurrency, plan.requests_per_worker, report)?;
        if let Some(reason) = stage_abort_reason(&stage.summary, report.policy.abort_error_percent)
        {
            if report.warmups.len() != position + 1 || report.stages.len() != position + 1 {
                return Err("saturation stage abort sequence is inconsistent");
            }
            terminal_reason = Some(reason);
            break;
        }
    }

    match terminal_reason {
        Some(reason) => {
            if report.status != SaturationRunStatus::Aborted || report.abort_reason != Some(reason)
            {
                return Err("saturation abort status or reason is inconsistent");
            }
        }
        None => {
            if report.warmups.len() != plan.concurrency_stages.len()
                || report.stages.len() != plan.concurrency_stages.len()
                || report.status != SaturationRunStatus::Complete
                || report.abort_reason.is_some()
            {
                return Err("saturation completion status or schedule is inconsistent");
            }
        }
    }
    Ok(())
}

fn validate_saved_warmup(
    warmup: &SaturationPhaseResult,
    concurrency: u32,
    requests_per_worker: u32,
) -> Result<(), &'static str> {
    let planned = concurrency
        .checked_mul(requests_per_worker)
        .ok_or("saturation warmup request count overflowed")?;
    if warmup.concurrency != concurrency
        || warmup.planned_requests != planned
        || warmup.attempted != planned
        || warmup.succeeded.checked_add(warmup.failed) != Some(warmup.attempted)
        || failure_count_total(warmup.failure_stage_counts) != Some(warmup.failed)
        || warmup.error_percent.to_bits()
            != error_percent(warmup.failed, warmup.attempted).to_bits()
        || warmup.duration_ns == 0
        || warmup.duration_ms != warmup.duration_ns / 1_000_000
    {
        return Err("saturation warmup evidence is internally inconsistent");
    }
    Ok(())
}

fn validate_saved_stage(
    stage: &SaturationStageResult,
    concurrency: u32,
    requests_per_worker: u32,
    report: &SaturationBenchmarkReport,
) -> Result<(), &'static str> {
    let planned = concurrency
        .checked_mul(requests_per_worker)
        .ok_or("saturation stage request count overflowed")?;
    if stage.status != SaturationPhaseStatus::Complete
        || stage.concurrency != concurrency
        || stage.planned_requests != planned
        || stage.attempts.len() != usize::try_from(planned).unwrap_or(usize::MAX)
        || stage.duration_ns == 0
        || stage.duration_ms != stage.duration_ns / 1_000_000
    {
        return Err("saturation stage shape is internally inconsistent");
    }
    validate_worker_duration_lower_bound(stage, requests_per_worker)?;
    for (expected_index, attempt) in stage.attempts.iter().enumerate() {
        if attempt.index != u32::try_from(expected_index).unwrap_or(u32::MAX) {
            return Err("saturation attempt indexes are not canonical");
        }
        validate_saved_attempt(
            attempt,
            report.target.stream,
            report.policy.expectation_configured,
            report.plan.max_tokens,
        )?;
    }
    let reconstructed = summarize_stage(
        &stage.attempts,
        Duration::from_nanos(stage.duration_ns),
        report.plan.max_tokens,
    );
    if !stage_summary_evidence_matches(&stage.summary, &reconstructed) {
        return Err("saturation stage summary does not match attempts");
    }
    if !gates_evidence_match(
        &stage.gates,
        &canonical_gates(&reconstructed, &report.policy),
    ) {
        return Err("saturation stage gates are not canonical");
    }
    Ok(())
}

fn stage_summary_evidence_matches(
    retained: &SaturationStageSummary,
    rebuilt: &SaturationStageSummary,
) -> bool {
    retained.attempted == rebuilt.attempted
        && retained.succeeded == rebuilt.succeeded
        && retained.failed == rebuilt.failed
        && retained.error_percent.to_bits() == rebuilt.error_percent.to_bits()
        && within_ulps(
            retained.attempted_requests_per_second,
            rebuilt.attempted_requests_per_second,
            16,
        )
        && within_ulps(
            retained.successful_requests_per_second,
            rebuilt.successful_requests_per_second,
            16,
        )
        && retained.prompt_token_samples == rebuilt.prompt_token_samples
        && retained.completion_token_samples == rebuilt.completion_token_samples
        && retained.prompt_tokens_observed_total == rebuilt.prompt_tokens_observed_total
        && retained.completion_tokens_observed_total == rebuilt.completion_tokens_observed_total
        && retained.total_tokens_observed_total == rebuilt.total_tokens_observed_total
        && retained.completion_token_usage_complete == rebuilt.completion_token_usage_complete
        && retained.total_token_usage_complete == rebuilt.total_token_usage_complete
        && option_float_bits_equal(
            retained.completion_token_goodput_per_second,
            rebuilt.completion_token_goodput_per_second,
        )
        && option_float_bits_equal(
            retained.total_token_goodput_per_second,
            rebuilt.total_token_goodput_per_second,
        )
        && distribution_evidence_matches(retained.headers_ms.as_ref(), rebuilt.headers_ms.as_ref())
        && distribution_evidence_matches(retained.ttft_ms.as_ref(), rebuilt.ttft_ms.as_ref())
        && distribution_evidence_matches(retained.e2e_ms.as_ref(), rebuilt.e2e_ms.as_ref())
        && distribution_evidence_matches(
            retained.output_tokens_per_second.as_ref(),
            rebuilt.output_tokens_per_second.as_ref(),
        )
        && retained.failure_stage_counts == rebuilt.failure_stage_counts
}

fn option_float_bits_equal(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => within_ulps(left, right, 16),
        _ => false,
    }
}

fn gates_evidence_match(retained: &[SaturationGate], rebuilt: &[SaturationGate]) -> bool {
    retained.len() == rebuilt.len()
        && retained.iter().zip(rebuilt).all(|(retained, rebuilt)| {
            retained.kind == rebuilt.kind
                && retained.operator == rebuilt.operator
                && option_float_bits_equal(retained.observed, rebuilt.observed)
                && retained.threshold.to_bits() == rebuilt.threshold.to_bits()
                && retained.status == rebuilt.status
                && retained.reason == rebuilt.reason
                && retained.samples == rebuilt.samples
                && retained.required_samples == rebuilt.required_samples
        })
}

fn validate_saved_attempt(
    attempt: &SaturationAttempt,
    stream: bool,
    expectation_configured: bool,
    max_tokens: u32,
) -> Result<(), &'static str> {
    for metric in [
        attempt.headers_ms,
        attempt.ttft_ms,
        attempt.e2e_ms,
        attempt.output_tokens_per_second,
    ] {
        if metric.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err("saturation attempt contains an invalid measurement");
        }
    }
    if attempt
        .headers_ms
        .zip(attempt.e2e_ms)
        .is_some_and(|(headers, e2e)| headers > e2e)
        || attempt
            .ttft_ms
            .zip(attempt.e2e_ms)
            .is_some_and(|(ttft, e2e)| ttft > e2e)
        || (!stream && (attempt.ttft_ms.is_some() || attempt.output_tokens_per_second.is_some()))
        || attempt
            .prompt_tokens
            .is_some_and(|tokens| !valid_reported_prompt_tokens(tokens))
        || attempt
            .completion_tokens
            .is_some_and(|tokens| !valid_reported_completion_tokens(tokens, max_tokens))
        || attempt.success != attempt.failure_stage.is_none()
    {
        return Err("saturation attempt fields are inconsistent");
    }

    if attempt.success {
        validate_saved_completed_attempt(attempt, stream, expectation_configured, true)?;
    } else {
        let stage = attempt
            .failure_stage
            .ok_or("failed saturation attempt lacks a failure stage")?;
        match stage {
            CanaryFailureStage::Transport => {
                if attempt.status_code != 0
                    || attempt.headers_ms.is_some()
                    || attempt.e2e_ms.is_none()
                {
                    return Err("saturation transport failure is not canonical");
                }
            }
            CanaryFailureStage::Http => {
                if attempt.status_code == 0
                    || (200..300).contains(&attempt.status_code)
                    || attempt.headers_ms.is_none()
                    || attempt.e2e_ms.is_none()
                {
                    return Err("saturation HTTP failure is not canonical");
                }
            }
            CanaryFailureStage::Protocol | CanaryFailureStage::EmptyOutput => {
                if !(200..300).contains(&attempt.status_code)
                    || attempt.headers_ms.is_none()
                    || attempt.e2e_ms.is_none()
                {
                    return Err("saturation response failure is not canonical");
                }
            }
            CanaryFailureStage::Expectation => {
                validate_saved_completed_attempt(attempt, stream, expectation_configured, false)?;
            }
        }
        if stage != CanaryFailureStage::Expectation
            && (attempt.ttft_ms.is_some()
                || attempt.prompt_tokens.is_some()
                || attempt.completion_tokens.is_some()
                || attempt.output_tokens_per_second.is_some()
                || attempt.expectation_met.is_some())
        {
            return Err("saturation failed attempt retains non-canonical evidence");
        }
    }

    validate_saved_output_rate_shape(attempt, stream)
}

fn validate_saved_completed_attempt(
    attempt: &SaturationAttempt,
    stream: bool,
    expectation_configured: bool,
    success: bool,
) -> Result<(), &'static str> {
    if !(200..300).contains(&attempt.status_code)
        || attempt.headers_ms.is_none()
        || attempt.e2e_ms.is_none()
        || (stream && attempt.ttft_ms.is_none())
        || attempt.expectation_met
            != if expectation_configured {
                Some(success)
            } else {
                None
            }
        || (!success
            && (!expectation_configured
                || attempt.failure_stage != Some(CanaryFailureStage::Expectation)))
    {
        return Err("saturation completed attempt is not canonical");
    }
    Ok(())
}

fn validate_saved_output_rate_shape(
    attempt: &SaturationAttempt,
    stream: bool,
) -> Result<(), &'static str> {
    let expected = stream
        && attempt.completion_tokens.is_some_and(|tokens| tokens >= 2)
        && attempt
            .ttft_ms
            .zip(attempt.e2e_ms)
            .is_some_and(|(ttft, e2e)| e2e > ttft);
    match (attempt.output_tokens_per_second, expected) {
        (None, false) => Ok(()),
        (Some(rate), true) if rate > 0.0 => Ok(()),
        _ => Err("saturation output-token rate shape is not canonical"),
    }
}

fn validate_worker_duration_lower_bound(
    stage: &SaturationStageResult,
    requests_per_worker: u32,
) -> Result<(), &'static str> {
    let worker_width = usize::try_from(requests_per_worker)
        .map_err(|_| "saturation worker request width is invalid")?;
    let mut longest_worker_ns = 0_u64;
    for worker_attempts in stage.attempts.chunks_exact(worker_width) {
        let worker_ns = worker_attempts.iter().try_fold(0_u64, |total, attempt| {
            let elapsed_ms = attempt
                .e2e_ms
                .ok_or("saturation attempt lacks end-to-end timing")?;
            let lower_bound_ns = conservative_nanoseconds(elapsed_ms)
                .ok_or("saturation end-to-end timing cannot be represented")?;
            total
                .checked_add(lower_bound_ns)
                .ok_or("saturation worker timing sum overflowed")
        })?;
        longest_worker_ns = longest_worker_ns.max(worker_ns);
    }
    if longest_worker_ns > stage.duration_ns {
        return Err("saturation stage duration is shorter than a closed-loop worker");
    }
    Ok(())
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn conservative_nanoseconds(milliseconds: f64) -> Option<u64> {
    let nanoseconds = milliseconds * 1_000_000.0;
    if nanoseconds.is_finite() && nanoseconds >= 0.0 && nanoseconds <= u64::MAX as f64 {
        Some(nanoseconds.floor() as u64)
    } else {
        None
    }
}

fn failure_count_total(counts: SaturationFailureStageCounts) -> Option<u32> {
    counts
        .transport
        .checked_add(counts.http)?
        .checked_add(counts.protocol)?
        .checked_add(counts.empty_output)?
        .checked_add(counts.expectation)
}

fn elapsed_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn elapsed_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn duration_millis_ceil(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos().div_ceil(1_000_000)).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{CanaryFailureStage, SaturationFailureStageCounts};

    fn successful_attempt(index: u32, ttft_ms: f64, e2e_ms: f64) -> SaturationAttempt {
        SaturationAttempt {
            index,
            success: true,
            status_code: 200,
            headers_ms: Some(1.0),
            ttft_ms: Some(ttft_ms),
            e2e_ms: Some(e2e_ms),
            prompt_tokens: Some(4),
            completion_tokens: Some(3),
            output_tokens_per_second: Some(30.0),
            expectation_met: Some(true),
            failure_stage: None,
        }
    }

    fn stage(concurrency: u32, successful_rps: f64, p95_ms: f64) -> SaturationStageResult {
        SaturationStageResult {
            status: SaturationPhaseStatus::Complete,
            concurrency,
            planned_requests: 20,
            duration_ns: 1_000_000_000,
            duration_ms: 1_000,
            summary: SaturationStageSummary {
                attempted: 20,
                succeeded: 20,
                successful_requests_per_second: successful_rps,
                attempted_requests_per_second: successful_rps,
                ttft_ms: Some(CanaryDistribution {
                    samples: 20,
                    min: p95_ms,
                    mean: p95_ms,
                    p50: p95_ms,
                    p95: p95_ms,
                    p99: p95_ms,
                    max: p95_ms,
                }),
                e2e_ms: Some(CanaryDistribution {
                    samples: 20,
                    min: p95_ms,
                    mean: p95_ms,
                    p50: p95_ms,
                    p95: p95_ms,
                    p99: p95_ms,
                    max: p95_ms,
                }),
                ..SaturationStageSummary::default()
            },
            gates: vec![SaturationGate {
                kind: SaturationGateKind::SuccessfulRequests,
                operator: SaturationGateOperator::GreaterThanOrEqual,
                observed: Some(20.0),
                threshold: 1.0,
                status: SaturationGateStatus::Pass,
                reason: None,
                samples: None,
                required_samples: None,
            }],
            attempts: Vec::new(),
        }
    }

    #[test]
    fn schedule_validation_is_explicit_and_monotonic() {
        let mut options = SaturationOptions {
            base_url: "http://127.0.0.1:8000/v1".to_owned(),
            model: "model".to_owned(),
            concurrency_stages: vec![1, 2, 4],
            max_tokens: 1,
            requests_per_worker: 1,
            ..SaturationOptions::default()
        };
        assert!(validate(&options).is_ok());
        options.concurrency_stages = vec![2, 4];
        assert!(validate(&options).is_err());
        options.concurrency_stages = vec![1, 4, 4];
        assert!(validate(&options).is_err());
        options.concurrency_stages = vec![1; 9];
        assert!(validate(&options).is_err());
    }

    #[test]
    fn saved_stage_allows_only_narrow_aggregate_mean_drift() {
        let attempts = (0..20)
            .map(|index| successful_attempt(index, f64::from(index + 1), 100.0))
            .collect::<Vec<_>>();
        let rebuilt = summarize_stage(&attempts, Duration::from_secs(1), 8);
        let mut retained = rebuilt.clone();
        let mean = retained.ttft_ms.as_ref().unwrap().mean;
        retained.ttft_ms.as_mut().unwrap().mean = f64::from_bits(mean.to_bits() + 4);
        assert!(stage_summary_evidence_matches(&retained, &rebuilt));

        retained.ttft_ms.as_mut().unwrap().p99 += 1.0;
        assert!(!stage_summary_evidence_matches(&retained, &rebuilt));
    }

    #[test]
    fn saved_assessment_allows_only_narrow_derived_float_drift() {
        let stages = vec![stage(1, 1.0, 100.0), stage(2, 2.0, 110.0)];
        let rebuilt = assess(&stages, &SaturationPolicy::default());
        let mut retained = rebuilt.clone();
        let efficiency = retained.scaling_evidence[0]
            .scaling_efficiency_percent
            .unwrap();
        retained.scaling_evidence[0].scaling_efficiency_percent =
            Some(f64::from_bits(efficiency.to_bits() + 8));
        assert!(assessment_evidence_matches(&retained, &rebuilt));

        retained.scaling_evidence[0].scaling_efficiency_percent =
            Some(f64::from_bits(efficiency.to_bits() + 17));
        assert!(!assessment_evidence_matches(&retained, &rebuilt));
    }

    #[test]
    fn failed_requests_never_inflate_successful_goodput() {
        let mut attempts = vec![successful_attempt(0, 10.0, 20.0)];
        attempts.push(SaturationAttempt {
            index: 1,
            success: false,
            failure_stage: Some(CanaryFailureStage::Transport),
            ..SaturationAttempt::default()
        });
        let summary = summarize_stage(&attempts, Duration::from_secs(1), 128);

        assert!((summary.attempted_requests_per_second - 2.0).abs() < f64::EPSILON);
        assert!((summary.successful_requests_per_second - 1.0).abs() < f64::EPSILON);
        assert_eq!(summary.completion_token_goodput_per_second, Some(3.0));
        assert!((summary.error_percent - 50.0).abs() < f64::EPSILON);
        assert_eq!(
            summary.failure_stage_counts,
            SaturationFailureStageCounts {
                transport: 1,
                ..SaturationFailureStageCounts::default()
            }
        );
    }

    #[test]
    fn token_goodput_gate_fails_closed_on_partial_usage() {
        let mut attempts = (0..20)
            .map(|index| successful_attempt(index, 10.0, 20.0))
            .collect::<Vec<_>>();
        attempts[0].completion_tokens = None;
        let summary = summarize_stage(&attempts, Duration::from_secs(1), 128);
        let gates = canonical_gates(
            &summary,
            &SaturationPolicy {
                min_completion_token_goodput_per_second: Some(1.0),
                minimum_latency_samples: MINIMUM_LATENCY_SAMPLES,
                ..SaturationPolicy::default()
            },
        );
        let gate = gates
            .iter()
            .find(|gate| gate.kind == SaturationGateKind::CompletionTokenGoodputPerSecond)
            .unwrap();

        assert_eq!(gate.status, SaturationGateStatus::NotEvaluable);
        assert_eq!(
            gate.reason,
            Some(SaturationGateReason::IncompleteUsageEvidence)
        );
        assert!(gate.observed.is_none());
    }

    #[test]
    fn latency_gate_requires_twenty_finite_success_samples() {
        let attempts = (0..19)
            .map(|index| successful_attempt(index, 10.0, 20.0))
            .collect::<Vec<_>>();
        let summary = summarize_stage(&attempts, Duration::from_secs(1), 128);
        let gate = latency_gate(
            SaturationGateKind::P95E2eMs,
            summary.e2e_ms.as_ref(),
            summary.succeeded,
            100.0,
            MINIMUM_LATENCY_SAMPLES,
        );

        assert_eq!(gate.status, SaturationGateStatus::NotEvaluable);
        assert_eq!(gate.reason, Some(SaturationGateReason::InsufficientSamples));
    }

    #[test]
    fn implausible_usage_cannot_create_goodput_or_pass_its_gate() {
        let mut attempt = successful_attempt(0, 10.0, 20.0);
        attempt.prompt_tokens = Some(u64::MAX);
        attempt.completion_tokens = Some(u64::MAX);
        let summary = summarize_stage(&[attempt], Duration::from_secs(1), 128);

        assert_eq!(summary.prompt_token_samples, 0);
        assert_eq!(summary.completion_token_samples, 0);
        assert_eq!(summary.prompt_tokens_observed_total, 0);
        assert_eq!(summary.completion_tokens_observed_total, 0);
        assert_eq!(summary.total_tokens_observed_total, 0);
        assert!(!summary.completion_token_usage_complete);
        assert!(!summary.total_token_usage_complete);
        assert_eq!(summary.completion_token_goodput_per_second, None);
        assert_eq!(summary.total_token_goodput_per_second, None);

        let gate = canonical_gates(
            &summary,
            &SaturationPolicy {
                min_completion_token_goodput_per_second: Some(1.0),
                ..SaturationPolicy::default()
            },
        )
        .into_iter()
        .find(|gate| gate.kind == SaturationGateKind::CompletionTokenGoodputPerSecond)
        .unwrap();
        assert_eq!(gate.status, SaturationGateStatus::NotEvaluable);
        assert_eq!(
            gate.reason,
            Some(SaturationGateReason::IncompleteUsageEvidence)
        );
        assert_eq!(gate.observed, None);
    }

    #[test]
    fn assessment_reports_only_an_observed_plateau_with_pressure() {
        let policy = SaturationPolicy {
            max_error_percent: 1.0,
            signal_max_marginal_scaling_efficiency_percent: 5.0,
            signal_min_p95_latency_inflation_percent: 20.0,
            ..SaturationPolicy::default()
        };
        let report = assess(&[stage(1, 10.0, 100.0), stage(2, 10.4, 125.0)], &policy);

        assert_eq!(report.status, SaturationAssessmentStatus::SignalObserved);
        assert_eq!(
            report.signal,
            Some(SaturationSignal::ThroughputPlateauWithLatencyInflation)
        );
        assert_eq!(report.first_signal_concurrency, Some(2));
    }

    #[test]
    fn assessment_requires_minimum_samples_and_keeps_all_scaling_rows() {
        let policy = SaturationPolicy {
            max_error_percent: 1.0,
            minimum_latency_samples: 20,
            signal_max_marginal_scaling_efficiency_percent: 5.0,
            signal_min_p95_latency_inflation_percent: 20.0,
            ..SaturationPolicy::default()
        };
        let mut first = stage(1, 10.0, 100.0);
        let mut second = stage(2, 10.4, 125.0);
        first.summary.ttft_ms.as_mut().unwrap().samples = 19;
        first.summary.e2e_ms.as_mut().unwrap().samples = 19;
        second.summary.ttft_ms.as_mut().unwrap().samples = 19;
        second.summary.e2e_ms.as_mut().unwrap().samples = 19;
        let undersampled = assess(&[first, second], &policy);
        assert_eq!(
            undersampled.status,
            SaturationAssessmentStatus::NotEvaluable
        );
        assert!(undersampled.signal.is_none());
        assert!(
            undersampled.scaling_evidence[0]
                .p95_e2e_inflation_percent
                .is_none()
        );

        let complete = assess(
            &[
                stage(1, 10.0, 100.0),
                stage(2, 10.4, 125.0),
                stage(4, 10.5, 150.0),
            ],
            &policy,
        );
        assert_eq!(complete.status, SaturationAssessmentStatus::SignalObserved);
        assert_eq!(complete.first_signal_concurrency, Some(2));
        assert_eq!(complete.scaling_evidence.len(), 2);
    }

    #[test]
    fn assessment_normalizes_throughput_gain_by_the_adjacent_concurrency_step() {
        let policy = SaturationPolicy {
            max_error_percent: 1.0,
            minimum_latency_samples: 20,
            signal_max_marginal_scaling_efficiency_percent: 5.0,
            signal_min_p95_latency_inflation_percent: 20.0,
            ..SaturationPolicy::default()
        };
        let report = assess(&[stage(20, 200.0, 100.0), stage(21, 210.0, 125.0)], &policy);

        assert_eq!(
            report.status,
            SaturationAssessmentStatus::NoSignalInTestedStages
        );
        assert!(report.signal.is_none());
        assert_eq!(report.scaling_evidence.len(), 1);
        let evidence = &report.scaling_evidence[0];
        assert!((evidence.marginal_successful_rps_gain_percent.unwrap() - 5.0).abs() < 1e-9);
        assert!((evidence.marginal_scaling_efficiency_percent.unwrap() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn assessment_handles_large_steps_regressions_and_missing_rps_fail_closed() {
        let policy = SaturationPolicy {
            max_error_percent: 1.0,
            minimum_latency_samples: 20,
            signal_max_marginal_scaling_efficiency_percent: 5.0,
            signal_min_p95_latency_inflation_percent: 20.0,
            ..SaturationPolicy::default()
        };

        let ideal_large_step = assess(&[stage(1, 10.0, 100.0), stage(8, 80.0, 125.0)], &policy);
        assert_eq!(
            ideal_large_step.status,
            SaturationAssessmentStatus::NoSignalInTestedStages
        );
        assert!(
            (ideal_large_step.scaling_evidence[0]
                .marginal_scaling_efficiency_percent
                .unwrap()
                - 100.0)
                .abs()
                < 1e-9
        );

        let regression = assess(&[stage(1, 10.0, 100.0), stage(2, 9.0, 125.0)], &policy);
        assert_eq!(
            regression.signal,
            Some(SaturationSignal::ThroughputPlateauWithLatencyInflation)
        );
        assert!(
            regression.scaling_evidence[0]
                .marginal_scaling_efficiency_percent
                .unwrap()
                < 0.0
        );

        let missing_baseline = assess(&[stage(1, 0.0, 100.0), stage(2, 1.0, 125.0)], &policy);
        assert_eq!(
            missing_baseline.status,
            SaturationAssessmentStatus::NotEvaluable
        );
        assert!(
            missing_baseline.scaling_evidence[0]
                .marginal_scaling_efficiency_percent
                .is_none()
        );
    }

    #[test]
    fn admission_counts_warmup_at_every_stage() {
        let options = SaturationOptions {
            base_url: "http://127.0.0.1:8000/v1".to_owned(),
            model: "model".to_owned(),
            concurrency_stages: vec![1, 2],
            warmup_requests_per_worker: 1,
            requests_per_worker: 1,
            max_tokens: 1,
            ..SaturationOptions::default()
        };

        assert_eq!(validate(&options).unwrap().planned_attempts, 6);
    }

    #[test]
    fn output_rate_shape_accepts_exact_duration_math_lost_in_millisecond_floats() {
        let attempt = SaturationAttempt {
            index: 0,
            success: true,
            status_code: 200,
            headers_ms: Some(1.0),
            ttft_ms: Some(299_999.999_999),
            e2e_ms: Some(300_000.0),
            prompt_tokens: Some(1),
            completion_tokens: Some(2),
            output_tokens_per_second: Some(1_000_000_000.0),
            expectation_met: Some(true),
            failure_stage: None,
        };

        assert!(validate_saved_output_rate_shape(&attempt, true).is_ok());
    }

    #[test]
    fn verification_distinguishes_failure_from_missing_evidence() {
        let mut failed = stage(1, 1.0, 1.0);
        failed.gates[0].status = SaturationGateStatus::Fail;
        assert_eq!(
            verify(Some(1), std::slice::from_ref(&failed)).status,
            SaturationVerificationStatus::Fail
        );
        failed.gates[0].status = SaturationGateStatus::NotEvaluable;
        assert_eq!(
            verify(Some(1), std::slice::from_ref(&failed)).status,
            SaturationVerificationStatus::NotEvaluable
        );
        assert_eq!(
            verify(Some(2), &[failed]).reason,
            Some(SaturationVerificationReason::StageNotRun)
        );
    }
}
