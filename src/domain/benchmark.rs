//! Stable, privacy-preserving contract for bounded saturation benchmarks.
//!
//! Saturation Benchmark v1 describes a single-process, closed-loop concurrency
//! ladder. It deliberately does not claim open-loop arrival control,
//! distributed load generation, production capacity, GPU occupancy, or an
//! untested breakpoint. Prompts, generated content, credentials, response
//! bodies, arbitrary server errors, and per-attempt model strings are never
//! retained.

use std::fmt;
use std::marker::PhantomData;

use chrono::{DateTime, Utc};
use serde::de::{Error as _, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use super::canary::{CanaryAttempt, CanaryDistribution, CanaryFailureStage};

/// Current version of the standalone saturation benchmark report contract.
pub const SATURATION_BENCHMARK_VERSION: u32 = 1;

/// Maximum number of measured concurrency points in one benchmark.
pub const MAX_SATURATION_STAGES: usize = 8;

/// Maximum number of retained measured attempts in one stage or report.
pub const MAX_SATURATION_ATTEMPTS: usize = 10_000;

/// Maximum number of policy gates attached to a measured stage.
pub const MAX_SATURATION_GATES_PER_STAGE: usize = 6;

/// Maximum number of fixed scaling comparisons in a report.
pub const MAX_SATURATION_SCALING_EVIDENCE: usize = MAX_SATURATION_STAGES - 1;

/// Maximum number of fixed nonclaims in a report.
pub const MAX_SATURATION_NONCLAIMS: usize = 16;

/// Maximum retained operator-supplied workload identity size.
pub const MAX_SATURATION_WORKLOAD_ID_BYTES: usize = 128;

/// Maximum retained target URL or model identity size.
pub const MAX_SATURATION_IDENTITY_BYTES: usize = 64 << 10;

/// Whether the complete requested schedule ran.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationRunStatus {
    Complete,
    #[default]
    Aborted,
}

impl SaturationRunStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Aborted => "aborted",
        }
    }
}

/// Fixed reason why a benchmark stopped before completing every stage.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationAbortReason {
    WarmupNoSuccessfulRequests,
    WarmupErrorRateLimitExceeded,
    StageNoSuccessfulRequests,
    StageErrorRateLimitExceeded,
}

/// The only active inference route supported by Saturation Benchmark v1.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationRoute {
    #[default]
    ChatCompletions,
}

/// Redacted endpoint identity. Credentials, prompts, and response data are absent.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SaturationTarget {
    #[serde(deserialize_with = "deserialize_identity_string")]
    pub url: String,
    pub route: SaturationRoute,
    #[serde(deserialize_with = "deserialize_identity_string")]
    pub model: String,
    pub stream: bool,
}

/// Request-arrival semantics used by every v1 measured stage.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationLoadModel {
    #[default]
    ClosedLoopFixedConcurrency,
}

/// Ordering rule for explicitly selected concurrency points.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationStageOrder {
    #[default]
    ExplicitAscending,
}

/// Scope and measurement treatment of warmup traffic.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationWarmupScope {
    #[default]
    EachStageExcluded,
}

/// How workers begin requests within each phase.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationWorkerStart {
    #[default]
    SimultaneousBarrier,
}

/// Explicit scheduling semantics for a bounded benchmark.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SaturationSchedule {
    pub load_model: SaturationLoadModel,
    pub stage_order: SaturationStageOrder,
    pub warmup_scope: SaturationWarmupScope,
    pub worker_start: SaturationWorkerStart,
}

/// Exact workload shape used for a saturation run.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SaturationPlan {
    #[serde(deserialize_with = "deserialize_concurrency_stages")]
    pub concurrency_stages: Vec<u32>,
    pub warmup_requests_per_worker: u32,
    pub requests_per_worker: u32,
    /// Warmup plus measured requests in the complete bounded schedule.
    pub planned_attempts: u32,
    pub max_tokens: u32,
    /// Exact request timeout used by the benchmark. This is the canonical value.
    pub timeout_ns: u64,
    /// Rounded-up millisecond timeout retained for display and compatibility.
    pub timeout_ms: u64,
    pub response_limit_bytes: usize,
    pub schedule: SaturationSchedule,
}

/// Exact thresholds and evidence requirements used for evaluation.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SaturationPolicy {
    pub max_error_percent: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_p95_ttft_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_p95_e2e_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_successful_requests_per_second: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_completion_token_goodput_per_second: Option<f64>,
    pub expectation_configured: bool,
    pub abort_error_percent: f64,
    pub minimum_latency_samples: u32,
    pub signal_max_marginal_scaling_efficiency_percent: f64,
    pub signal_min_p95_latency_inflation_percent: f64,
}

/// Completion state for a warmup or measured stage.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationPhaseStatus {
    Complete,
    #[default]
    Aborted,
}

/// Fixed failure-stage counts without retaining arbitrary failure messages.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SaturationFailureStageCounts {
    pub transport: u32,
    pub http: u32,
    pub protocol: u32,
    pub empty_output: u32,
    pub expectation: u32,
}

impl SaturationFailureStageCounts {
    /// Count one fixed failure stage.
    pub const fn increment(&mut self, stage: CanaryFailureStage) {
        match stage {
            CanaryFailureStage::Transport => self.transport = self.transport.saturating_add(1),
            CanaryFailureStage::Http => self.http = self.http.saturating_add(1),
            CanaryFailureStage::Protocol => self.protocol = self.protocol.saturating_add(1),
            CanaryFailureStage::EmptyOutput => {
                self.empty_output = self.empty_output.saturating_add(1);
            }
            CanaryFailureStage::Expectation => {
                self.expectation = self.expectation.saturating_add(1);
            }
        }
    }

    /// Count fixed failure stages from privacy-safe benchmark attempts.
    pub fn from_attempts(attempts: &[SaturationAttempt]) -> Self {
        let mut counts = Self::default();
        for attempt in attempts {
            if let Some(stage) = attempt.failure_stage {
                counts.increment(stage);
            }
        }
        counts
    }
}

/// Aggregate evidence from one exact-stage warmup, excluded from measurements.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SaturationPhaseResult {
    pub status: SaturationPhaseStatus,
    pub concurrency: u32,
    pub planned_requests: u32,
    pub attempted: u32,
    pub succeeded: u32,
    pub failed: u32,
    pub error_percent: f64,
    /// Exact elapsed phase duration used for any derived rates.
    pub duration_ns: u64,
    /// Truncated millisecond duration retained for display.
    pub duration_ms: u64,
    pub failure_stage_counts: SaturationFailureStageCounts,
}

/// Aggregated measured evidence for one concurrency point.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SaturationStageSummary {
    pub attempted: u32,
    pub succeeded: u32,
    pub failed: u32,
    pub error_percent: f64,
    pub attempted_requests_per_second: f64,
    pub successful_requests_per_second: f64,
    pub prompt_token_samples: u32,
    pub completion_token_samples: u32,
    pub prompt_tokens_observed_total: u64,
    pub completion_tokens_observed_total: u64,
    pub total_tokens_observed_total: u64,
    pub completion_token_usage_complete: bool,
    pub total_token_usage_complete: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_token_goodput_per_second: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_token_goodput_per_second: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers_ms: Option<CanaryDistribution>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<CanaryDistribution>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub e2e_ms: Option<CanaryDistribution>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_per_second: Option<CanaryDistribution>,
    pub failure_stage_counts: SaturationFailureStageCounts,
}

/// One request result with all arbitrary server- and model-provided text removed.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SaturationAttempt {
    pub index: u32,
    pub success: bool,
    #[serde(default, skip_serializing_if = "is_zero_u16")]
    pub status_code: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub e2e_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_per_second: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expectation_met: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_stage: Option<CanaryFailureStage>,
}

impl From<&CanaryAttempt> for SaturationAttempt {
    fn from(attempt: &CanaryAttempt) -> Self {
        Self {
            index: attempt.index,
            success: attempt.success,
            status_code: attempt.status_code,
            headers_ms: attempt.headers_ms,
            ttft_ms: attempt.ttft_ms,
            e2e_ms: attempt.e2e_ms,
            prompt_tokens: attempt.prompt_tokens,
            completion_tokens: attempt.completion_tokens,
            output_tokens_per_second: attempt.output_tokens_per_second,
            expectation_met: attempt.expectation_met,
            failure_stage: attempt.failure.as_ref().map(|failure| failure.stage),
        }
    }
}

impl From<CanaryAttempt> for SaturationAttempt {
    fn from(attempt: CanaryAttempt) -> Self {
        Self::from(&attempt)
    }
}

/// Fixed service-level policy gate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationGateKind {
    SuccessfulRequests,
    ErrorPercent,
    P95TtftMs,
    P95E2eMs,
    SuccessfulRequestsPerSecond,
    CompletionTokenGoodputPerSecond,
}

/// Comparison used by a fixed policy gate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationGateOperator {
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Gate outcome, including absence of the evidence needed to evaluate it.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationGateStatus {
    Pass,
    Fail,
    #[default]
    NotEvaluable,
}

/// Fixed reason a gate could not be evaluated.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationGateReason {
    NoSuccessfulRequests,
    InsufficientSamples,
    IncompleteUsageEvidence,
}

/// One auditable gate attached to a measured stage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SaturationGate {
    pub kind: SaturationGateKind,
    pub operator: SaturationGateOperator,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed: Option<f64>,
    pub threshold: f64,
    pub status: SaturationGateStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<SaturationGateReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_samples: Option<u32>,
}

/// Measured evidence from one exact tested concurrency point.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SaturationStageResult {
    pub status: SaturationPhaseStatus,
    pub concurrency: u32,
    pub planned_requests: u32,
    /// Exact elapsed stage duration used for all derived rates.
    pub duration_ns: u64,
    /// Truncated millisecond duration retained for display.
    pub duration_ms: u64,
    pub summary: SaturationStageSummary,
    #[serde(
        default,
        deserialize_with = "deserialize_gates",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub gates: Vec<SaturationGate>,
    #[serde(default, deserialize_with = "deserialize_attempts")]
    pub attempts: Vec<SaturationAttempt>,
}

/// Scaling evidence comparing a measured stage with earlier tested points.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SaturationScalingEvidence {
    pub baseline_concurrency: u32,
    pub previous_concurrency: u32,
    pub concurrency: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub marginal_successful_rps_gain_percent: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub marginal_scaling_efficiency_percent: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scaling_efficiency_percent: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p95_ttft_inflation_percent: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p95_e2e_inflation_percent: Option<f64>,
}

/// Scope-safe result of the fixed saturation-signal heuristic.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationAssessmentStatus {
    SignalObserved,
    NoSignalInTestedStages,
    #[default]
    NotEvaluable,
}

/// Fixed observed saturation signal. It is evidence, not a capacity estimate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationSignal {
    ThroughputPlateauWithErrorRate,
    ThroughputPlateauWithLatencyInflation,
}

/// Assessment over only the explicitly tested concurrency points.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SaturationAssessment {
    pub status: SaturationAssessmentStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal: Option<SaturationSignal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_signal_concurrency: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub highest_accepted_tested_concurrency: Option<u32>,
    #[serde(
        default,
        deserialize_with = "deserialize_scaling_evidence",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub scaling_evidence: Vec<SaturationScalingEvidence>,
}

/// Outcome of an optional exact-stage deployment gate.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationVerificationStatus {
    #[default]
    NotRequested,
    Pass,
    Fail,
    NotEvaluable,
}

/// Fixed reason an explicitly requested verification was not evaluable.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationVerificationReason {
    StageNotRun,
    GateNotEvaluable,
}

/// Verification result for one explicitly requested concurrency stage.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SaturationVerification {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_concurrency: Option<u32>,
    pub status: SaturationVerificationStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<SaturationVerificationReason>,
}

/// Explicit limits on what can be inferred from this benchmark.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationNonclaim {
    ClosedLoopMayHideCoordinatedOmission,
    SingleLoadGeneratorMayBeTheBottleneck,
    TestedConcurrencyPointsOnly,
    ConcurrencyIsNotBatchSizeOrGpuOccupancy,
    SyntheticWorkloadMayBenefitFromCaching,
    ExternalTrafficIsNotIsolated,
    TokenGoodputUsesEndpointReportedUsage,
    HighestAcceptedTestedConcurrencyIsNotProductionCapacityOrRecommendation,
    NoProductionSlaCertification,
    NotDistributedOpenLoopSoakOrBreakpointTesting,
}

/// Canonical v1 nonclaims. Producers should emit this complete ordered list.
pub const SATURATION_BENCHMARK_NONCLAIMS: [SaturationNonclaim; 10] = [
    SaturationNonclaim::ClosedLoopMayHideCoordinatedOmission,
    SaturationNonclaim::SingleLoadGeneratorMayBeTheBottleneck,
    SaturationNonclaim::TestedConcurrencyPointsOnly,
    SaturationNonclaim::ConcurrencyIsNotBatchSizeOrGpuOccupancy,
    SaturationNonclaim::SyntheticWorkloadMayBenefitFromCaching,
    SaturationNonclaim::ExternalTrafficIsNotIsolated,
    SaturationNonclaim::TokenGoodputUsesEndpointReportedUsage,
    SaturationNonclaim::HighestAcceptedTestedConcurrencyIsNotProductionCapacityOrRecommendation,
    SaturationNonclaim::NoProductionSlaCertification,
    SaturationNonclaim::NotDistributedOpenLoopSoakOrBreakpointTesting,
];

/// Complete standalone result from a bounded saturation benchmark.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SaturationBenchmarkReport {
    pub saturation_benchmark_version: u32,
    pub started_at: DateTime<Utc>,
    /// Exact elapsed run duration. This is the canonical duration.
    pub duration_ns: u64,
    /// Truncated millisecond duration retained for display.
    pub duration_ms: u64,
    pub status: SaturationRunStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abort_reason: Option<SaturationAbortReason>,
    #[serde(deserialize_with = "deserialize_workload_id")]
    pub workload_id: String,
    pub target: SaturationTarget,
    pub plan: SaturationPlan,
    pub policy: SaturationPolicy,
    #[serde(deserialize_with = "deserialize_warmups")]
    pub warmups: Vec<SaturationPhaseResult>,
    #[serde(deserialize_with = "deserialize_stages")]
    pub stages: Vec<SaturationStageResult>,
    pub assessment: SaturationAssessment,
    pub verification: SaturationVerification,
    #[serde(deserialize_with = "deserialize_nonclaims")]
    pub nonclaims: Vec<SaturationNonclaim>,
}

fn deserialize_workload_id<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_string::<D, MAX_SATURATION_WORKLOAD_ID_BYTES>(deserializer)
}

fn deserialize_identity_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_string::<D, MAX_SATURATION_IDENTITY_BYTES>(deserializer)
}

fn deserialize_bounded_string<'de, D, const MAX: usize>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    struct BoundedStringVisitor<const MAX: usize>;

    impl<const MAX: usize> Visitor<'_> for BoundedStringVisitor<MAX> {
        type Value = String;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "a UTF-8 string no longer than {MAX} bytes")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            if value.len() > MAX {
                return Err(E::custom(
                    "string exceeds its saturation report safety limit",
                ));
            }
            Ok(value.to_owned())
        }

        fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            if value.len() > MAX {
                return Err(E::custom(
                    "string exceeds its saturation report safety limit",
                ));
            }
            Ok(value)
        }
    }

    deserializer.deserialize_string(BoundedStringVisitor::<MAX>)
}

fn deserialize_concurrency_stages<'de, D>(deserializer: D) -> Result<Vec<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, u32, MAX_SATURATION_STAGES>(deserializer)
}

fn deserialize_gates<'de, D>(deserializer: D) -> Result<Vec<SaturationGate>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, SaturationGate, MAX_SATURATION_GATES_PER_STAGE>(deserializer)
}

fn deserialize_attempts<'de, D>(deserializer: D) -> Result<Vec<SaturationAttempt>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, SaturationAttempt, MAX_SATURATION_ATTEMPTS>(deserializer)
}

fn deserialize_stages<'de, D>(deserializer: D) -> Result<Vec<SaturationStageResult>, D::Error>
where
    D: Deserializer<'de>,
{
    let stages =
        deserialize_bounded_vec::<D, SaturationStageResult, MAX_SATURATION_STAGES>(deserializer)?;
    let total_attempts = stages.iter().try_fold(0_usize, |total, stage| {
        total.checked_add(stage.attempts.len())
    });
    if total_attempts.is_none_or(|total| total > MAX_SATURATION_ATTEMPTS) {
        return Err(D::Error::custom(
            "attempts exceed the saturation report safety limit",
        ));
    }
    Ok(stages)
}

fn deserialize_warmups<'de, D>(deserializer: D) -> Result<Vec<SaturationPhaseResult>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, SaturationPhaseResult, MAX_SATURATION_STAGES>(deserializer)
}

fn deserialize_scaling_evidence<'de, D>(
    deserializer: D,
) -> Result<Vec<SaturationScalingEvidence>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, SaturationScalingEvidence, MAX_SATURATION_SCALING_EVIDENCE>(
        deserializer,
    )
}

fn deserialize_nonclaims<'de, D>(deserializer: D) -> Result<Vec<SaturationNonclaim>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, SaturationNonclaim, MAX_SATURATION_NONCLAIMS>(deserializer)
}

fn deserialize_bounded_vec<'de, D, T, const MAX: usize>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct BoundedVecVisitor<T, const MAX: usize>(PhantomData<T>);

    impl<'de, T, const MAX: usize> Visitor<'de> for BoundedVecVisitor<T, MAX>
    where
        T: Deserialize<'de>,
    {
        type Value = Vec<T>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "a sequence with no more than {MAX} entries")
        }

        fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            if sequence.size_hint().is_some_and(|size| size > MAX) {
                return Err(A::Error::custom(
                    "sequence exceeds its saturation report safety limit",
                ));
            }
            let mut values = Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(MAX));
            while let Some(value) = sequence.next_element()? {
                if values.len() == MAX {
                    return Err(A::Error::custom(
                        "sequence exceeds its saturation report safety limit",
                    ));
                }
                values.push(value);
            }
            Ok(values)
        }
    }

    deserializer.deserialize_seq(BoundedVecVisitor::<T, MAX>(PhantomData))
}

#[allow(clippy::trivially_copy_pass_by_ref)]
const fn is_zero_u16(value: &u16) -> bool {
    *value == 0
}

#[cfg(test)]
mod tests {
    use super::super::canary::CanaryFailure;
    use super::*;

    #[test]
    fn attempt_conversion_drops_all_arbitrary_server_text() {
        let attempt = CanaryAttempt {
            index: 7,
            success: false,
            status_code: 503,
            headers_ms: Some(2.0),
            ttft_ms: None,
            e2e_ms: Some(8.0),
            prompt_tokens: Some(3),
            completion_tokens: Some(5),
            output_tokens_per_second: Some(10.0),
            model: "secret-response-model".to_owned(),
            finish_reason: "secret-finish-detail".to_owned(),
            expectation_met: Some(false),
            failure: Some(CanaryFailure {
                stage: CanaryFailureStage::Http,
                message: "secret-server-error".to_owned(),
            }),
        };

        let encoded = serde_json::to_string(&SaturationAttempt::from(attempt)).unwrap();

        assert!(!encoded.contains("secret-response-model"));
        assert!(!encoded.contains("secret-finish-detail"));
        assert!(!encoded.contains("secret-server-error"));
        assert!(!encoded.contains("model"));
        assert!(!encoded.contains("finish_reason"));
        assert!(!encoded.contains("message"));
        assert!(encoded.contains("\"failure_stage\":\"http\""));
    }

    #[test]
    fn enum_spellings_are_stable_and_scope_safe() {
        assert_eq!(
            serde_json::to_string(&SaturationRunStatus::Complete).unwrap(),
            "\"complete\""
        );
        assert_eq!(
            serde_json::to_string(&SaturationGateStatus::NotEvaluable).unwrap(),
            "\"not_evaluable\""
        );
        assert_eq!(
            serde_json::to_string(&SaturationVerificationStatus::NotRequested).unwrap(),
            "\"not_requested\""
        );
        assert_eq!(
            serde_json::to_string(&SaturationAssessmentStatus::NoSignalInTestedStages).unwrap(),
            "\"no_signal_in_tested_stages\""
        );
        assert_eq!(
            serde_json::to_string(&SaturationAbortReason::WarmupNoSuccessfulRequests).unwrap(),
            "\"warmup_no_successful_requests\""
        );
        assert_eq!(SATURATION_BENCHMARK_VERSION, 1);
    }

    #[test]
    fn fixed_nonclaims_make_the_v1_scope_explicit() {
        let encoded = serde_json::to_value(SATURATION_BENCHMARK_NONCLAIMS).unwrap();

        assert_eq!(encoded.as_array().unwrap().len(), 10);
        assert!(
            encoded
                .as_array()
                .unwrap()
                .contains(&serde_json::Value::String(
                    "tested_concurrency_points_only".to_owned()
                ))
        );
        assert!(
            encoded
                .as_array()
                .unwrap()
                .contains(&serde_json::Value::String(
                    "not_distributed_open_loop_soak_or_breakpoint_testing".to_owned()
                ))
        );
    }

    #[test]
    fn deserialization_rejects_oversized_identity_and_stage_vectors() {
        let target = serde_json::json!({
            "url": "x".repeat(MAX_SATURATION_IDENTITY_BYTES + 1),
            "route": "chat_completions",
            "model": "model",
            "stream": false
        });
        assert!(serde_json::from_value::<SaturationTarget>(target).is_err());

        let plan = serde_json::json!({
            "concurrency_stages": vec![1; MAX_SATURATION_STAGES + 1],
            "warmup_requests_per_worker": 1,
            "requests_per_worker": 1,
            "planned_attempts": 1,
            "max_tokens": 1,
            "timeout_ns": 1_000_000,
            "timeout_ms": 1,
            "response_limit_bytes": 1,
            "schedule": {
                "load_model": "closed_loop_fixed_concurrency",
                "stage_order": "explicit_ascending",
                "warmup_scope": "each_stage_excluded",
                "worker_start": "simultaneous_barrier"
            }
        });
        assert!(serde_json::from_value::<SaturationPlan>(plan).is_err());
    }

    #[test]
    fn failure_counts_only_retain_fixed_stages() {
        let attempts = [
            SaturationAttempt {
                failure_stage: Some(CanaryFailureStage::Transport),
                ..SaturationAttempt::default()
            },
            SaturationAttempt {
                failure_stage: Some(CanaryFailureStage::Protocol),
                ..SaturationAttempt::default()
            },
            SaturationAttempt {
                failure_stage: Some(CanaryFailureStage::Transport),
                ..SaturationAttempt::default()
            },
        ];

        assert_eq!(
            SaturationFailureStageCounts::from_attempts(&attempts),
            SaturationFailureStageCounts {
                transport: 2,
                protocol: 1,
                ..SaturationFailureStageCounts::default()
            }
        );
    }
}
