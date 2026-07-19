//! Stable, privacy-safe contract for offline saturation benchmark comparison.
//!
//! Saturation Comparison v1 compares only canonical, like-for-like saved
//! Saturation Benchmark v1 reports. It intentionally excludes endpoint origins,
//! raw attempts, prompts, generated content, credentials, and arbitrary errors.

use std::fmt;
use std::marker::PhantomData;

use chrono::{DateTime, Utc};
use serde::de::{Error, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use super::{
    MAX_SATURATION_IDENTITY_BYTES, MAX_SATURATION_STAGES, MAX_SATURATION_WORKLOAD_ID_BYTES,
    SaturationPlan, SaturationPolicy, SaturationRoute, SaturationRunStatus,
};

/// Current version of the standalone saturation comparison contract.
pub const SATURATION_COMPARISON_VERSION: u32 = 1;

/// Minimum relevant samples required on each side of every quantitative gate.
pub const MIN_SATURATION_COMPARISON_SAMPLES: u32 = 20;

const MAX_COMPARISON_COMPATIBILITY_CHECKS: usize = 20;
const MAX_COMPARISON_GATES_PER_STAGE: usize = 5;
const MAX_COMPARISON_NONCLAIMS: usize = 13;

/// Operator-selected stage-by-stage regression thresholds.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SaturationComparisonPolicy {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_p95_ttft_regression_percent: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_p95_e2e_regression_percent: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_successful_requests_per_second_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_completion_token_goodput_per_second_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_error_percent_increase_points: Option<f64>,
    pub minimum_stage_samples: u32,
}

impl Default for SaturationComparisonPolicy {
    fn default() -> Self {
        Self {
            max_p95_ttft_regression_percent: None,
            max_p95_e2e_regression_percent: None,
            min_successful_requests_per_second_ratio: None,
            min_completion_token_goodput_per_second_ratio: None,
            max_error_percent_increase_points: None,
            minimum_stage_samples: MIN_SATURATION_COMPARISON_SAMPLES,
        }
    }
}

/// Endpoint-free identity required for a like-for-like comparison.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SaturationComparisonIdentity {
    pub saturation_benchmark_version: u32,
    #[serde(deserialize_with = "deserialize_workload_id")]
    pub workload_id: String,
    #[serde(deserialize_with = "deserialize_identity_string")]
    pub model: String,
    pub route: SaturationRoute,
    pub stream: bool,
    pub plan: SaturationPlan,
    pub policy: SaturationPolicy,
}

/// One exact condition required before quantitative evidence is comparable.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationCompatibilityField {
    BaselineIntegrity,
    CandidateIntegrity,
    ReportVersion,
    TimestampOrder,
    BaselineStatus,
    CandidateStatus,
    WorkloadId,
    Model,
    Route,
    Stream,
    ConcurrencyStages,
    WarmupRequestsPerWorker,
    RequestsPerWorker,
    MaxTokens,
    TimeoutNanoseconds,
    ResponseLimitBytes,
    Schedule,
    Policy,
}

/// Fixed reason a compatibility condition failed.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationCompatibilityReason {
    InvalidReportEvidence,
    UnsupportedVersion,
    CandidatePredatesBaseline,
    RunDidNotComplete,
    InvalidIdentity,
    ValuesDiffer,
}

/// Result of one fixed compatibility condition.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct SaturationCompatibilityCheck {
    pub field: SaturationCompatibilityField,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<SaturationCompatibilityReason>,
}

/// Aggregate outcome of the benchmark's own stage policy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationStagePolicyStatus {
    Pass,
    Fail,
    NotEvaluable,
}

/// Outcome of comparing the source benchmark policy at one exact stage.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationPolicyComparisonStatus {
    Pass,
    Regression,
    NotEvaluable,
}

/// Bounded metrics retained for one side of one exact concurrency point.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SaturationComparableStageEvidence {
    /// Exact denominator used for every rate in this stage.
    pub duration_ns: u64,
    pub attempted: u32,
    pub succeeded: u32,
    pub error_percent: f64,
    pub successful_requests_per_second: f64,
    pub ttft_samples: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p95_ttft_ms: Option<f64>,
    pub e2e_samples: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p95_e2e_ms: Option<f64>,
    pub completion_token_samples: u32,
    pub completion_token_usage_complete: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_token_goodput_per_second: Option<f64>,
}

/// Quantitative metric selected for every exact concurrency point.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationComparisonGateKind {
    P95TtftRegressionPercent,
    P95E2eRegressionPercent,
    SuccessfulRequestsPerSecondRatio,
    CompletionTokenGoodputPerSecondRatio,
    ErrorPercentIncreasePoints,
}

/// Comparison operator applied to a stage gate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationComparisonGateOperator {
    LessThanOrEqual,
    GreaterThanOrEqual,
}

/// Outcome of one selected comparison gate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationComparisonGateStatus {
    Pass,
    Fail,
    NotEvaluable,
}

/// Fixed missing-evidence reason for a comparison gate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationComparisonGateReason {
    BaselineNoSuccessfulRequests,
    CandidateNoSuccessfulRequests,
    BaselineMeasurementUnavailable,
    CandidateMeasurementUnavailable,
    BaselineInsufficientSamples,
    CandidateInsufficientSamples,
    BaselineUsageIncomplete,
    CandidateUsageIncomplete,
    BaselineMeasurementZero,
    NonFiniteComparison,
}

/// One selected metric gate at one exact concurrency point.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SaturationComparisonGate {
    pub kind: SaturationComparisonGateKind,
    pub operator: SaturationComparisonGateOperator,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed: Option<f64>,
    pub threshold: f64,
    pub status: SaturationComparisonGateStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<SaturationComparisonGateReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_samples: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_samples: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_samples: Option<u32>,
}

/// Like-for-like evidence and selected gates at one exact tested point.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SaturationStageComparison {
    pub concurrency: u32,
    pub baseline: SaturationComparableStageEvidence,
    pub candidate: SaturationComparableStageEvidence,
    pub baseline_policy_status: SaturationStagePolicyStatus,
    pub candidate_policy_status: SaturationStagePolicyStatus,
    pub policy_comparison_status: SaturationPolicyComparisonStatus,
    #[serde(
        default,
        deserialize_with = "deserialize_comparison_gates",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub gates: Vec<SaturationComparisonGate>,
}

/// Explicit limits on what can be inferred from an offline comparison.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationComparisonNonclaim {
    RecomputedConsistencyDoesNotAuthenticateSavedReports,
    MatchingWorkloadIdAndExpectationFlagDoNotProveHiddenInputsMatch,
    ExactLikeForLikeTestedStagesOnly,
    ClosedLoopMayHideCoordinatedOmission,
    SingleLoadGeneratorMayBeTheBottleneck,
    TokenGoodputUsesEndpointReportedUsage,
    RuntimeHardwareNetworkAndExternalTrafficAreNotControlled,
    NoCausalAttribution,
    NoStatisticalSignificanceOrConfidenceInterval,
    OperatorThresholdsAreNotUniversalSlos,
    NoProductionCapacityOrRecommendation,
    NoSlaCertification,
    EndpointOriginsIntentionallyExcluded,
}

/// Canonical v1 comparison limitations.
pub const SATURATION_COMPARISON_NONCLAIMS: [SaturationComparisonNonclaim; 13] = [
    SaturationComparisonNonclaim::RecomputedConsistencyDoesNotAuthenticateSavedReports,
    SaturationComparisonNonclaim::MatchingWorkloadIdAndExpectationFlagDoNotProveHiddenInputsMatch,
    SaturationComparisonNonclaim::ExactLikeForLikeTestedStagesOnly,
    SaturationComparisonNonclaim::ClosedLoopMayHideCoordinatedOmission,
    SaturationComparisonNonclaim::SingleLoadGeneratorMayBeTheBottleneck,
    SaturationComparisonNonclaim::TokenGoodputUsesEndpointReportedUsage,
    SaturationComparisonNonclaim::RuntimeHardwareNetworkAndExternalTrafficAreNotControlled,
    SaturationComparisonNonclaim::NoCausalAttribution,
    SaturationComparisonNonclaim::NoStatisticalSignificanceOrConfidenceInterval,
    SaturationComparisonNonclaim::OperatorThresholdsAreNotUniversalSlos,
    SaturationComparisonNonclaim::NoProductionCapacityOrRecommendation,
    SaturationComparisonNonclaim::NoSlaCertification,
    SaturationComparisonNonclaim::EndpointOriginsIntentionallyExcluded,
];

/// Authoritative machine-readable outcome of the complete comparison.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SaturationComparisonStatus {
    Pass,
    Regression,
    NotEvaluable,
}

/// Complete standalone result from one offline saturation comparison.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SaturationComparisonReport {
    pub saturation_comparison_version: u32,
    pub baseline_started_at: DateTime<Utc>,
    pub candidate_started_at: DateTime<Utc>,
    pub baseline_status: SaturationRunStatus,
    pub candidate_status: SaturationRunStatus,
    pub baseline: SaturationComparisonIdentity,
    pub candidate: SaturationComparisonIdentity,
    pub comparison_policy: SaturationComparisonPolicy,
    pub compatible: bool,
    #[serde(deserialize_with = "deserialize_compatibility_checks")]
    pub compatibility: Vec<SaturationCompatibilityCheck>,
    #[serde(
        default,
        deserialize_with = "deserialize_stage_comparisons",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub stages: Vec<SaturationStageComparison>,
    pub status: SaturationComparisonStatus,
    /// True only when comparable evidence demonstrates a regression.
    pub regression: bool,
    #[serde(deserialize_with = "deserialize_comparison_nonclaims")]
    pub nonclaims: Vec<SaturationComparisonNonclaim>,
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
            write!(formatter, "a UTF-8 string with no more than {MAX} bytes")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: Error,
        {
            if value.len() > MAX {
                return Err(E::custom("string exceeds its comparison safety limit"));
            }
            Ok(value.to_owned())
        }

        fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
        where
            E: Error,
        {
            if value.len() > MAX {
                return Err(E::custom("string exceeds its comparison safety limit"));
            }
            Ok(value)
        }
    }

    deserializer.deserialize_string(BoundedStringVisitor::<MAX>)
}

fn deserialize_compatibility_checks<'de, D>(
    deserializer: D,
) -> Result<Vec<SaturationCompatibilityCheck>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, SaturationCompatibilityCheck, MAX_COMPARISON_COMPATIBILITY_CHECKS>(
        deserializer,
    )
}

fn deserialize_stage_comparisons<'de, D>(
    deserializer: D,
) -> Result<Vec<SaturationStageComparison>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, SaturationStageComparison, MAX_SATURATION_STAGES>(deserializer)
}

fn deserialize_comparison_gates<'de, D>(
    deserializer: D,
) -> Result<Vec<SaturationComparisonGate>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, SaturationComparisonGate, MAX_COMPARISON_GATES_PER_STAGE>(
        deserializer,
    )
}

fn deserialize_comparison_nonclaims<'de, D>(
    deserializer: D,
) -> Result<Vec<SaturationComparisonNonclaim>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, SaturationComparisonNonclaim, MAX_COMPARISON_NONCLAIMS>(
        deserializer,
    )
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
                    "sequence exceeds its comparison safety limit",
                ));
            }
            let mut values = Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(MAX));
            while let Some(value) = sequence.next_element()? {
                if values.len() == MAX {
                    return Err(A::Error::custom(
                        "sequence exceeds its comparison safety limit",
                    ));
                }
                values.push(value);
            }
            Ok(values)
        }
    }

    deserializer.deserialize_seq(BoundedVecVisitor::<T, MAX>(PhantomData))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_nonclaims_cover_statistical_and_operational_scope() {
        let value = serde_json::to_value(SATURATION_COMPARISON_NONCLAIMS).unwrap();
        let entries = value.as_array().unwrap();

        assert_eq!(entries.len(), 13);
        assert!(
            entries
                .iter()
                .any(|entry| { entry == "no_statistical_significance_or_confidence_interval" })
        );
        assert!(
            entries
                .iter()
                .any(|entry| { entry == "endpoint_origins_intentionally_excluded" })
        );
        assert!(entries.iter().any(|entry| {
            entry == "matching_workload_id_and_expectation_flag_do_not_prove_hidden_inputs_match"
        }));
        assert!(entries.iter().any(|entry| {
            entry == "recomputed_consistency_does_not_authenticate_saved_reports"
        }));
        assert!(
            entries
                .iter()
                .any(|entry| entry == "closed_loop_may_hide_coordinated_omission")
        );
        assert!(
            entries
                .iter()
                .any(|entry| entry == "token_goodput_uses_endpoint_reported_usage")
        );
    }
}
