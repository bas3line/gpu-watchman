//! Privacy-safe, fail-closed comparison of saved saturation benchmark reports.

use std::path::Path;

use anyhow::{Result, bail};

use crate::domain::{
    MIN_SATURATION_COMPARISON_SAMPLES, SATURATION_BENCHMARK_VERSION,
    SATURATION_COMPARISON_NONCLAIMS, SATURATION_COMPARISON_VERSION, SaturationBenchmarkReport,
    SaturationComparableStageEvidence, SaturationComparisonGate, SaturationComparisonGateKind,
    SaturationComparisonGateOperator, SaturationComparisonGateReason,
    SaturationComparisonGateStatus, SaturationComparisonIdentity, SaturationComparisonPolicy,
    SaturationComparisonReport, SaturationComparisonStatus, SaturationCompatibilityCheck,
    SaturationCompatibilityField, SaturationCompatibilityReason, SaturationGateStatus,
    SaturationPolicyComparisonStatus, SaturationRunStatus, SaturationStageComparison,
    SaturationStagePolicyStatus, valid_canary_workload_id,
};
use crate::operations::report_input::load_json_or_final_ndjson;
use crate::operations::saturation::validate_report_semantics;

const MAX_SATURATION_REPORT_BYTES: u64 = 64 * 1024 * 1024;

/// Load one complete JSON benchmark or the final non-empty NDJSON record.
pub fn load_saturation_report(path: &Path) -> Result<SaturationBenchmarkReport> {
    let report: SaturationBenchmarkReport = load_json_or_final_ndjson(
        path,
        "saturation benchmark report",
        MAX_SATURATION_REPORT_BYTES,
    )?;
    if report.saturation_benchmark_version != SATURATION_BENCHMARK_VERSION {
        bail!(
            "saturation benchmark report {} uses an unsupported version (this build supports version {SATURATION_BENCHMARK_VERSION})",
            path.display()
        );
    }
    Ok(report)
}

/// Compare two saved, exact-ladder reports under explicit stage-by-stage gates.
pub fn compare(
    baseline: &SaturationBenchmarkReport,
    candidate: &SaturationBenchmarkReport,
    policy: &SaturationComparisonPolicy,
) -> Result<SaturationComparisonReport> {
    validate_comparison_policy(policy)?;

    let baseline_integrity = validate_report_semantics(baseline);
    let candidate_integrity = validate_report_semantics(candidate);
    let compatibility = compatibility_checks(
        baseline,
        candidate,
        &baseline_integrity,
        &candidate_integrity,
    );
    let compatible = compatibility.iter().all(|check| check.passed);
    let stages = if compatible {
        baseline
            .stages
            .iter()
            .zip(&candidate.stages)
            .map(|(baseline, candidate)| compare_stage(baseline, candidate, policy))
            .collect()
    } else {
        Vec::new()
    };
    let status = comparison_status(compatible, &stages);

    Ok(SaturationComparisonReport {
        saturation_comparison_version: SATURATION_COMPARISON_VERSION,
        baseline_started_at: baseline.started_at,
        candidate_started_at: candidate.started_at,
        baseline_status: baseline.status,
        candidate_status: candidate.status,
        baseline: identity(baseline),
        candidate: identity(candidate),
        comparison_policy: policy.clone(),
        compatible,
        compatibility,
        stages,
        status,
        regression: status == SaturationComparisonStatus::Regression,
        nonclaims: SATURATION_COMPARISON_NONCLAIMS.to_vec(),
    })
}

fn validate_comparison_policy(policy: &SaturationComparisonPolicy) -> Result<()> {
    if policy.minimum_stage_samples != MIN_SATURATION_COMPARISON_SAMPLES {
        bail!(
            "saturation comparison requires exactly {MIN_SATURATION_COMPARISON_SAMPLES} relevant samples per side"
        );
    }
    for (label, value) in [
        (
            "maximum p95 TTFT regression percent",
            policy.max_p95_ttft_regression_percent,
        ),
        (
            "maximum p95 end-to-end regression percent",
            policy.max_p95_e2e_regression_percent,
        ),
        (
            "minimum successful request-rate ratio",
            policy.min_successful_requests_per_second_ratio,
        ),
        (
            "minimum completion-token goodput ratio",
            policy.min_completion_token_goodput_per_second_ratio,
        ),
    ] {
        if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
            bail!("{label} must be finite and non-negative");
        }
    }
    if policy
        .max_error_percent_increase_points
        .is_some_and(|value| !value.is_finite() || !(0.0..=100.0).contains(&value))
    {
        bail!("maximum error-percent increase must be finite and between zero and 100 points");
    }
    Ok(())
}

fn identity(report: &SaturationBenchmarkReport) -> SaturationComparisonIdentity {
    SaturationComparisonIdentity {
        saturation_benchmark_version: report.saturation_benchmark_version,
        workload_id: report.workload_id.clone(),
        model: report.target.model.clone(),
        route: report.target.route,
        stream: report.target.stream,
        plan: report.plan.clone(),
        policy: report.policy.clone(),
    }
}

fn compatibility_checks(
    baseline: &SaturationBenchmarkReport,
    candidate: &SaturationBenchmarkReport,
    baseline_integrity: &Result<(), &'static str>,
    candidate_integrity: &Result<(), &'static str>,
) -> Vec<SaturationCompatibilityCheck> {
    vec![
        condition(
            SaturationCompatibilityField::BaselineIntegrity,
            baseline_integrity.is_ok(),
            SaturationCompatibilityReason::InvalidReportEvidence,
        ),
        condition(
            SaturationCompatibilityField::CandidateIntegrity,
            candidate_integrity.is_ok(),
            SaturationCompatibilityReason::InvalidReportEvidence,
        ),
        condition(
            SaturationCompatibilityField::ReportVersion,
            baseline.saturation_benchmark_version == SATURATION_BENCHMARK_VERSION
                && candidate.saturation_benchmark_version == SATURATION_BENCHMARK_VERSION
                && baseline.saturation_benchmark_version == candidate.saturation_benchmark_version,
            SaturationCompatibilityReason::UnsupportedVersion,
        ),
        condition(
            SaturationCompatibilityField::TimestampOrder,
            candidate.started_at >= baseline.started_at,
            SaturationCompatibilityReason::CandidatePredatesBaseline,
        ),
        condition(
            SaturationCompatibilityField::BaselineStatus,
            baseline.status == SaturationRunStatus::Complete,
            SaturationCompatibilityReason::RunDidNotComplete,
        ),
        condition(
            SaturationCompatibilityField::CandidateStatus,
            candidate.status == SaturationRunStatus::Complete,
            SaturationCompatibilityReason::RunDidNotComplete,
        ),
        identity_condition(
            SaturationCompatibilityField::WorkloadId,
            valid_canary_workload_id(&baseline.workload_id)
                && valid_canary_workload_id(&candidate.workload_id),
            baseline.workload_id == candidate.workload_id,
        ),
        identity_condition(
            SaturationCompatibilityField::Model,
            !baseline.target.model.trim().is_empty() && !candidate.target.model.trim().is_empty(),
            baseline.target.model == candidate.target.model,
        ),
        equality(
            SaturationCompatibilityField::Route,
            baseline.target.route == candidate.target.route,
        ),
        equality(
            SaturationCompatibilityField::Stream,
            baseline.target.stream == candidate.target.stream,
        ),
        equality(
            SaturationCompatibilityField::ConcurrencyStages,
            baseline.plan.concurrency_stages == candidate.plan.concurrency_stages,
        ),
        equality(
            SaturationCompatibilityField::WarmupRequestsPerWorker,
            baseline.plan.warmup_requests_per_worker == candidate.plan.warmup_requests_per_worker,
        ),
        equality(
            SaturationCompatibilityField::RequestsPerWorker,
            baseline.plan.requests_per_worker == candidate.plan.requests_per_worker,
        ),
        equality(
            SaturationCompatibilityField::MaxTokens,
            baseline.plan.max_tokens == candidate.plan.max_tokens,
        ),
        equality(
            SaturationCompatibilityField::TimeoutNanoseconds,
            baseline.plan.timeout_ns == candidate.plan.timeout_ns,
        ),
        equality(
            SaturationCompatibilityField::ResponseLimitBytes,
            baseline.plan.response_limit_bytes == candidate.plan.response_limit_bytes,
        ),
        equality(
            SaturationCompatibilityField::Schedule,
            baseline.plan.schedule == candidate.plan.schedule,
        ),
        equality(
            SaturationCompatibilityField::Policy,
            baseline.policy == candidate.policy,
        ),
    ]
}

const fn condition(
    field: SaturationCompatibilityField,
    passed: bool,
    failed_reason: SaturationCompatibilityReason,
) -> SaturationCompatibilityCheck {
    SaturationCompatibilityCheck {
        field,
        passed,
        reason: if passed { None } else { Some(failed_reason) },
    }
}

fn identity_condition(
    field: SaturationCompatibilityField,
    valid: bool,
    equal: bool,
) -> SaturationCompatibilityCheck {
    condition(
        field,
        valid && equal,
        if valid {
            SaturationCompatibilityReason::ValuesDiffer
        } else {
            SaturationCompatibilityReason::InvalidIdentity
        },
    )
}

const fn equality(
    field: SaturationCompatibilityField,
    equal: bool,
) -> SaturationCompatibilityCheck {
    condition(field, equal, SaturationCompatibilityReason::ValuesDiffer)
}

fn compare_stage(
    baseline: &crate::domain::SaturationStageResult,
    candidate: &crate::domain::SaturationStageResult,
    policy: &SaturationComparisonPolicy,
) -> SaturationStageComparison {
    let baseline_evidence = comparable_evidence(baseline);
    let candidate_evidence = comparable_evidence(candidate);
    let baseline_policy_status = stage_policy_status(baseline);
    let candidate_policy_status = stage_policy_status(candidate);
    let policy_comparison_status =
        compare_stage_policy(baseline_policy_status, candidate_policy_status);
    let mut gates = Vec::new();
    if let Some(threshold) = policy.max_p95_ttft_regression_percent {
        gates.push(latency_regression_gate(
            SaturationComparisonGateKind::P95TtftRegressionPercent,
            &baseline_evidence,
            &candidate_evidence,
            baseline_evidence.p95_ttft_ms,
            candidate_evidence.p95_ttft_ms,
            baseline_evidence.ttft_samples,
            candidate_evidence.ttft_samples,
            threshold,
        ));
    }
    if let Some(threshold) = policy.max_p95_e2e_regression_percent {
        gates.push(latency_regression_gate(
            SaturationComparisonGateKind::P95E2eRegressionPercent,
            &baseline_evidence,
            &candidate_evidence,
            baseline_evidence.p95_e2e_ms,
            candidate_evidence.p95_e2e_ms,
            baseline_evidence.e2e_samples,
            candidate_evidence.e2e_samples,
            threshold,
        ));
    }
    if let Some(threshold) = policy.min_successful_requests_per_second_ratio {
        gates.push(rps_ratio_gate(
            &baseline_evidence,
            &candidate_evidence,
            threshold,
        ));
    }
    if let Some(threshold) = policy.min_completion_token_goodput_per_second_ratio {
        gates.push(completion_goodput_ratio_gate(
            &baseline_evidence,
            &candidate_evidence,
            threshold,
        ));
    }
    if let Some(threshold) = policy.max_error_percent_increase_points {
        gates.push(error_increase_gate(
            &baseline_evidence,
            &candidate_evidence,
            threshold,
        ));
    }
    SaturationStageComparison {
        concurrency: baseline.concurrency,
        baseline: baseline_evidence,
        candidate: candidate_evidence,
        baseline_policy_status,
        candidate_policy_status,
        policy_comparison_status,
        gates,
    }
}

fn comparable_evidence(
    stage: &crate::domain::SaturationStageResult,
) -> SaturationComparableStageEvidence {
    SaturationComparableStageEvidence {
        duration_ns: stage.duration_ns,
        attempted: stage.summary.attempted,
        succeeded: stage.summary.succeeded,
        error_percent: stage.summary.error_percent,
        successful_requests_per_second: stage.summary.successful_requests_per_second,
        ttft_samples: stage
            .summary
            .ttft_ms
            .as_ref()
            .map_or(0, |value| u32::try_from(value.samples).unwrap_or(u32::MAX)),
        p95_ttft_ms: stage.summary.ttft_ms.as_ref().map(|value| value.p95),
        e2e_samples: stage
            .summary
            .e2e_ms
            .as_ref()
            .map_or(0, |value| u32::try_from(value.samples).unwrap_or(u32::MAX)),
        p95_e2e_ms: stage.summary.e2e_ms.as_ref().map(|value| value.p95),
        completion_token_samples: stage.summary.completion_token_samples,
        completion_token_usage_complete: stage.summary.completion_token_usage_complete,
        completion_token_goodput_per_second: stage.summary.completion_token_goodput_per_second,
    }
}

fn stage_policy_status(
    stage: &crate::domain::SaturationStageResult,
) -> SaturationStagePolicyStatus {
    if stage.summary.attempted < MIN_SATURATION_COMPARISON_SAMPLES {
        SaturationStagePolicyStatus::NotEvaluable
    } else if stage
        .gates
        .iter()
        .any(|gate| gate.status == SaturationGateStatus::Fail)
    {
        SaturationStagePolicyStatus::Fail
    } else if stage
        .gates
        .iter()
        .any(|gate| gate.status == SaturationGateStatus::NotEvaluable)
    {
        SaturationStagePolicyStatus::NotEvaluable
    } else {
        SaturationStagePolicyStatus::Pass
    }
}

const fn compare_stage_policy(
    baseline: SaturationStagePolicyStatus,
    candidate: SaturationStagePolicyStatus,
) -> SaturationPolicyComparisonStatus {
    match (baseline, candidate) {
        (SaturationStagePolicyStatus::Pass, SaturationStagePolicyStatus::Pass) => {
            SaturationPolicyComparisonStatus::Pass
        }
        (SaturationStagePolicyStatus::Pass, SaturationStagePolicyStatus::Fail) => {
            SaturationPolicyComparisonStatus::Regression
        }
        _ => SaturationPolicyComparisonStatus::NotEvaluable,
    }
}

#[allow(clippy::too_many_arguments)]
fn latency_regression_gate(
    kind: SaturationComparisonGateKind,
    baseline_evidence: &SaturationComparableStageEvidence,
    candidate_evidence: &SaturationComparableStageEvidence,
    baseline: Option<f64>,
    candidate: Option<f64>,
    baseline_samples: u32,
    candidate_samples: u32,
    threshold: f64,
) -> SaturationComparisonGate {
    let reason = if baseline_evidence.succeeded == 0 {
        Some(SaturationComparisonGateReason::BaselineNoSuccessfulRequests)
    } else if candidate_evidence.succeeded == 0 {
        Some(SaturationComparisonGateReason::CandidateNoSuccessfulRequests)
    } else if baseline.is_none() {
        Some(SaturationComparisonGateReason::BaselineMeasurementUnavailable)
    } else if candidate.is_none() {
        Some(SaturationComparisonGateReason::CandidateMeasurementUnavailable)
    } else if baseline_samples < MIN_SATURATION_COMPARISON_SAMPLES {
        Some(SaturationComparisonGateReason::BaselineInsufficientSamples)
    } else if candidate_samples < MIN_SATURATION_COMPARISON_SAMPLES {
        Some(SaturationComparisonGateReason::CandidateInsufficientSamples)
    } else if baseline.is_some_and(|value| value <= 0.0) {
        Some(SaturationComparisonGateReason::BaselineMeasurementZero)
    } else {
        None
    };
    evaluated_gate(
        kind,
        SaturationComparisonGateOperator::LessThanOrEqual,
        baseline,
        candidate,
        threshold,
        reason,
        Some(baseline_samples),
        Some(candidate_samples),
        Some(MIN_SATURATION_COMPARISON_SAMPLES),
        |before, after| ((after / before) - 1.0) * 100.0,
    )
}

fn rps_ratio_gate(
    baseline: &SaturationComparableStageEvidence,
    candidate: &SaturationComparableStageEvidence,
    threshold: f64,
) -> SaturationComparisonGate {
    let reason = if baseline.attempted < MIN_SATURATION_COMPARISON_SAMPLES {
        Some(SaturationComparisonGateReason::BaselineInsufficientSamples)
    } else if candidate.attempted < MIN_SATURATION_COMPARISON_SAMPLES {
        Some(SaturationComparisonGateReason::CandidateInsufficientSamples)
    } else if baseline.succeeded == 0 {
        Some(SaturationComparisonGateReason::BaselineNoSuccessfulRequests)
    } else if baseline.successful_requests_per_second <= 0.0 {
        Some(SaturationComparisonGateReason::BaselineMeasurementZero)
    } else {
        None
    };
    evaluated_gate(
        SaturationComparisonGateKind::SuccessfulRequestsPerSecondRatio,
        SaturationComparisonGateOperator::GreaterThanOrEqual,
        Some(baseline.successful_requests_per_second),
        Some(candidate.successful_requests_per_second),
        threshold,
        reason,
        Some(baseline.attempted),
        Some(candidate.attempted),
        Some(MIN_SATURATION_COMPARISON_SAMPLES),
        |before, after| after / before,
    )
}

fn completion_goodput_ratio_gate(
    baseline: &SaturationComparableStageEvidence,
    candidate: &SaturationComparableStageEvidence,
    threshold: f64,
) -> SaturationComparisonGate {
    let reason = if baseline.succeeded == 0 {
        Some(SaturationComparisonGateReason::BaselineNoSuccessfulRequests)
    } else if candidate.succeeded == 0 {
        Some(SaturationComparisonGateReason::CandidateNoSuccessfulRequests)
    } else if !baseline.completion_token_usage_complete {
        Some(SaturationComparisonGateReason::BaselineUsageIncomplete)
    } else if !candidate.completion_token_usage_complete {
        Some(SaturationComparisonGateReason::CandidateUsageIncomplete)
    } else if baseline.completion_token_samples < MIN_SATURATION_COMPARISON_SAMPLES {
        Some(SaturationComparisonGateReason::BaselineInsufficientSamples)
    } else if candidate.completion_token_samples < MIN_SATURATION_COMPARISON_SAMPLES {
        Some(SaturationComparisonGateReason::CandidateInsufficientSamples)
    } else if baseline.completion_token_goodput_per_second.is_none() {
        Some(SaturationComparisonGateReason::BaselineMeasurementUnavailable)
    } else if candidate.completion_token_goodput_per_second.is_none() {
        Some(SaturationComparisonGateReason::CandidateMeasurementUnavailable)
    } else if baseline
        .completion_token_goodput_per_second
        .is_some_and(|value| value <= 0.0)
    {
        Some(SaturationComparisonGateReason::BaselineMeasurementZero)
    } else {
        None
    };
    evaluated_gate(
        SaturationComparisonGateKind::CompletionTokenGoodputPerSecondRatio,
        SaturationComparisonGateOperator::GreaterThanOrEqual,
        baseline.completion_token_goodput_per_second,
        candidate.completion_token_goodput_per_second,
        threshold,
        reason,
        Some(baseline.completion_token_samples),
        Some(candidate.completion_token_samples),
        Some(MIN_SATURATION_COMPARISON_SAMPLES),
        |before, after| after / before,
    )
}

fn error_increase_gate(
    baseline: &SaturationComparableStageEvidence,
    candidate: &SaturationComparableStageEvidence,
    threshold: f64,
) -> SaturationComparisonGate {
    let reason = if baseline.attempted < MIN_SATURATION_COMPARISON_SAMPLES {
        Some(SaturationComparisonGateReason::BaselineInsufficientSamples)
    } else if candidate.attempted < MIN_SATURATION_COMPARISON_SAMPLES {
        Some(SaturationComparisonGateReason::CandidateInsufficientSamples)
    } else {
        None
    };
    evaluated_gate(
        SaturationComparisonGateKind::ErrorPercentIncreasePoints,
        SaturationComparisonGateOperator::LessThanOrEqual,
        Some(baseline.error_percent),
        Some(candidate.error_percent),
        threshold,
        reason,
        Some(baseline.attempted),
        Some(candidate.attempted),
        Some(MIN_SATURATION_COMPARISON_SAMPLES),
        |before, after| after - before,
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluated_gate(
    kind: SaturationComparisonGateKind,
    operator: SaturationComparisonGateOperator,
    baseline: Option<f64>,
    candidate: Option<f64>,
    threshold: f64,
    unavailable_reason: Option<SaturationComparisonGateReason>,
    baseline_samples: Option<u32>,
    candidate_samples: Option<u32>,
    required_samples: Option<u32>,
    evaluate: impl FnOnce(f64, f64) -> f64,
) -> SaturationComparisonGate {
    let (observed, status, reason) = if let Some(reason) = unavailable_reason {
        (
            None,
            SaturationComparisonGateStatus::NotEvaluable,
            Some(reason),
        )
    } else if let (Some(baseline), Some(candidate)) = (baseline, candidate) {
        let observed = evaluate(baseline, candidate);
        if observed.is_finite() {
            let tolerance = observed.abs().max(threshold.abs()).max(1.0) * 1.0e-9;
            let passed = match operator {
                SaturationComparisonGateOperator::LessThanOrEqual => {
                    observed <= threshold + tolerance
                }
                SaturationComparisonGateOperator::GreaterThanOrEqual => {
                    observed + tolerance >= threshold
                }
            };
            (
                Some(observed),
                if passed {
                    SaturationComparisonGateStatus::Pass
                } else {
                    SaturationComparisonGateStatus::Fail
                },
                None,
            )
        } else {
            (
                None,
                SaturationComparisonGateStatus::NotEvaluable,
                Some(SaturationComparisonGateReason::NonFiniteComparison),
            )
        }
    } else {
        (
            None,
            SaturationComparisonGateStatus::NotEvaluable,
            Some(if baseline.is_none() {
                SaturationComparisonGateReason::BaselineMeasurementUnavailable
            } else {
                SaturationComparisonGateReason::CandidateMeasurementUnavailable
            }),
        )
    };
    SaturationComparisonGate {
        kind,
        operator,
        baseline,
        candidate,
        observed,
        threshold,
        status,
        reason,
        baseline_samples,
        candidate_samples,
        required_samples,
    }
}

fn comparison_status(
    compatible: bool,
    stages: &[SaturationStageComparison],
) -> SaturationComparisonStatus {
    if !compatible {
        return SaturationComparisonStatus::NotEvaluable;
    }
    if stages.iter().any(|stage| {
        stage.policy_comparison_status == SaturationPolicyComparisonStatus::Regression
            || stage
                .gates
                .iter()
                .any(|gate| gate.status == SaturationComparisonGateStatus::Fail)
    }) {
        return SaturationComparisonStatus::Regression;
    }
    if stages.iter().any(|stage| {
        stage.policy_comparison_status == SaturationPolicyComparisonStatus::NotEvaluable
            || stage
                .gates
                .iter()
                .any(|gate| gate.status == SaturationComparisonGateStatus::NotEvaluable)
    }) {
        SaturationComparisonStatus::NotEvaluable
    } else {
        SaturationComparisonStatus::Pass
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use chrono::{TimeZone, Utc};

    use super::*;
    use crate::domain::{
        SATURATION_BENCHMARK_NONCLAIMS, SaturationAttempt, SaturationFailureStageCounts,
        SaturationLoadModel, SaturationPhaseResult, SaturationPhaseStatus, SaturationPlan,
        SaturationPolicy, SaturationRoute, SaturationSchedule, SaturationStageOrder,
        SaturationStageResult, SaturationTarget, SaturationVerification, SaturationWarmupScope,
        SaturationWorkerStart,
    };
    use crate::operations::saturation::{assess, canonical_gates, summarize_stage};

    fn timestamp(seconds: i64) -> chrono::DateTime<Utc> {
        Utc.timestamp_opt(seconds, 0).single().unwrap()
    }

    fn successful_attempt(index: u32, e2e_ms: f64) -> SaturationAttempt {
        let ttft_ms = 10.0;
        SaturationAttempt {
            index,
            success: true,
            status_code: 200,
            headers_ms: Some(1.0),
            ttft_ms: Some(ttft_ms),
            e2e_ms: Some(e2e_ms),
            prompt_tokens: Some(4),
            completion_tokens: Some(3),
            output_tokens_per_second: Some(2_000.0 / (e2e_ms - ttft_ms)),
            expectation_met: Some(true),
            failure_stage: None,
        }
    }

    fn report(
        started_at: chrono::DateTime<Utc>,
        origin: &str,
        stage_duration_ns: u64,
        e2e_ms: f64,
        failures: u32,
    ) -> SaturationBenchmarkReport {
        let policy = SaturationPolicy {
            max_error_percent: 90.0,
            max_p95_ttft_ms: None,
            max_p95_e2e_ms: None,
            min_successful_requests_per_second: None,
            min_completion_token_goodput_per_second: None,
            expectation_configured: true,
            abort_error_percent: 100.0,
            minimum_latency_samples: 20,
            signal_max_marginal_scaling_efficiency_percent: 5.0,
            signal_min_p95_latency_inflation_percent: 20.0,
        };
        let attempts = (0..20)
            .map(|index| {
                if index < failures {
                    SaturationAttempt {
                        index,
                        success: false,
                        status_code: 0,
                        e2e_ms: Some(5.0),
                        failure_stage: Some(crate::domain::CanaryFailureStage::Transport),
                        ..SaturationAttempt::default()
                    }
                } else {
                    successful_attempt(index, e2e_ms)
                }
            })
            .collect::<Vec<_>>();
        let summary = summarize_stage(&attempts, Duration::from_nanos(stage_duration_ns), 8);
        let stage = SaturationStageResult {
            status: SaturationPhaseStatus::Complete,
            concurrency: 1,
            planned_requests: 20,
            duration_ns: stage_duration_ns,
            duration_ms: stage_duration_ns / 1_000_000,
            gates: canonical_gates(&summary, &policy),
            summary,
            attempts,
        };
        let warmup_duration_ns = 10_000_000;
        let stages = vec![stage];
        SaturationBenchmarkReport {
            saturation_benchmark_version: SATURATION_BENCHMARK_VERSION,
            started_at,
            duration_ns: warmup_duration_ns + stage_duration_ns,
            duration_ms: (warmup_duration_ns + stage_duration_ns) / 1_000_000,
            status: SaturationRunStatus::Complete,
            abort_reason: None,
            workload_id: "builtin-v1".to_owned(),
            target: SaturationTarget {
                url: origin.to_owned(),
                route: SaturationRoute::ChatCompletions,
                model: "served-model".to_owned(),
                stream: true,
            },
            plan: SaturationPlan {
                concurrency_stages: vec![1],
                warmup_requests_per_worker: 1,
                requests_per_worker: 20,
                planned_attempts: 21,
                max_tokens: 8,
                timeout_ns: 10_000_000_000,
                timeout_ms: 10_000,
                response_limit_bytes: 128 << 10,
                schedule: SaturationSchedule {
                    load_model: SaturationLoadModel::ClosedLoopFixedConcurrency,
                    stage_order: SaturationStageOrder::ExplicitAscending,
                    warmup_scope: SaturationWarmupScope::EachStageExcluded,
                    worker_start: SaturationWorkerStart::SimultaneousBarrier,
                },
            },
            policy: policy.clone(),
            warmups: vec![SaturationPhaseResult {
                status: SaturationPhaseStatus::Complete,
                concurrency: 1,
                planned_requests: 1,
                attempted: 1,
                succeeded: 1,
                failed: 0,
                error_percent: 0.0,
                duration_ns: warmup_duration_ns,
                duration_ms: warmup_duration_ns / 1_000_000,
                failure_stage_counts: SaturationFailureStageCounts::default(),
            }],
            assessment: assess(&stages, &policy),
            stages,
            verification: SaturationVerification::default(),
            nonclaims: SATURATION_BENCHMARK_NONCLAIMS.to_vec(),
        }
    }

    fn recanonicalize(report: &mut SaturationBenchmarkReport) {
        for stage in &mut report.stages {
            stage.summary = summarize_stage(
                &stage.attempts,
                Duration::from_nanos(stage.duration_ns),
                report.plan.max_tokens,
            );
            stage.gates = canonical_gates(&stage.summary, &report.policy);
        }
        report.assessment = assess(&report.stages, &report.policy);
    }

    #[test]
    fn exact_nanosecond_reports_validate_and_timing_tampering_is_rejected() {
        let baseline = report(
            timestamp(1),
            "https://baseline.example",
            1_000_000_001,
            20.0,
            0,
        );
        assert!(validate_report_semantics(&baseline).is_ok());

        let mut tampered = baseline.clone();
        tampered.stages[0].summary.successful_requests_per_second += 0.001;
        assert!(validate_report_semantics(&tampered).is_err());

        let mut rounded = baseline.clone();
        rounded.stages[0].duration_ms += 1;
        assert!(validate_report_semantics(&rounded).is_err());

        let mut forged_rate = baseline;
        forged_rate.stages[0].duration_ns = 100_000_000;
        forged_rate.stages[0].duration_ms = 100;
        forged_rate.duration_ns = 110_000_000;
        forged_rate.duration_ms = 110;
        recanonicalize(&mut forged_rate);
        assert!(validate_report_semantics(&forged_rate).is_err());
    }

    #[test]
    fn different_origins_are_allowed_and_never_enter_comparison_output() {
        let baseline = report(
            timestamp(1),
            "https://private-baseline.example",
            1_000_000_000,
            20.0,
            0,
        );
        let candidate = report(
            timestamp(2),
            "https://private-candidate.example",
            1_000_000_000,
            20.0,
            0,
        );
        let comparison = compare(
            &baseline,
            &candidate,
            &SaturationComparisonPolicy::default(),
        )
        .unwrap();

        assert!(comparison.compatible);
        assert_eq!(comparison.status, SaturationComparisonStatus::Pass);
        let json = serde_json::to_string(&comparison).unwrap();
        for forbidden in [
            "private-baseline.example",
            "private-candidate.example",
            "\"attempts\":",
            "\"target\":",
        ] {
            assert!(!json.contains(forbidden));
        }
    }

    #[test]
    fn quantitative_regressions_fail_every_selected_stage_gate() {
        let baseline = report(
            timestamp(1),
            "https://baseline.example",
            1_000_000_000,
            20.0,
            0,
        );
        let candidate = report(
            timestamp(2),
            "https://candidate.example",
            2_000_000_000,
            30.0,
            0,
        );
        let comparison = compare(
            &baseline,
            &candidate,
            &SaturationComparisonPolicy {
                max_p95_ttft_regression_percent: Some(0.0),
                max_p95_e2e_regression_percent: Some(10.0),
                min_successful_requests_per_second_ratio: Some(0.9),
                min_completion_token_goodput_per_second_ratio: Some(0.9),
                max_error_percent_increase_points: Some(0.0),
                ..SaturationComparisonPolicy::default()
            },
        )
        .unwrap();

        assert_eq!(comparison.status, SaturationComparisonStatus::Regression);
        assert!(comparison.regression);
        let gates = &comparison.stages[0].gates;
        assert_eq!(gates.len(), 5);
        assert_eq!(gates[0].status, SaturationComparisonGateStatus::Pass);
        assert_eq!(gates[1].status, SaturationComparisonGateStatus::Fail);
        assert_eq!(gates[2].status, SaturationComparisonGateStatus::Fail);
        assert_eq!(gates[3].status, SaturationComparisonGateStatus::Fail);
        assert_eq!(gates[4].status, SaturationComparisonGateStatus::Pass);
    }

    #[test]
    fn incomplete_candidate_usage_is_not_evaluable_never_a_pass() {
        let baseline = report(
            timestamp(1),
            "https://baseline.example",
            1_000_000_000,
            20.0,
            0,
        );
        let mut candidate = report(
            timestamp(2),
            "https://candidate.example",
            1_000_000_000,
            20.0,
            0,
        );
        candidate.stages[0].attempts[0].completion_tokens = None;
        candidate.stages[0].attempts[0].output_tokens_per_second = None;
        recanonicalize(&mut candidate);
        assert!(validate_report_semantics(&candidate).is_ok());

        let comparison = compare(
            &baseline,
            &candidate,
            &SaturationComparisonPolicy {
                min_completion_token_goodput_per_second_ratio: Some(0.9),
                ..SaturationComparisonPolicy::default()
            },
        )
        .unwrap();
        assert_eq!(comparison.status, SaturationComparisonStatus::NotEvaluable);
        assert!(!comparison.regression);
        assert_eq!(
            comparison.stages[0].gates[0].reason,
            Some(SaturationComparisonGateReason::CandidateUsageIncomplete)
        );
    }

    #[test]
    fn source_policy_failure_is_a_typed_regression() {
        let mut baseline = report(
            timestamp(1),
            "https://baseline.example",
            1_000_000_000,
            20.0,
            0,
        );
        let mut candidate = report(
            timestamp(2),
            "https://candidate.example",
            1_000_000_000,
            20.0,
            1,
        );
        baseline.policy.max_error_percent = 1.0;
        candidate.policy.max_error_percent = 1.0;
        recanonicalize(&mut baseline);
        recanonicalize(&mut candidate);

        let comparison = compare(
            &baseline,
            &candidate,
            &SaturationComparisonPolicy::default(),
        )
        .unwrap();
        assert_eq!(comparison.status, SaturationComparisonStatus::Regression);
        assert_eq!(
            comparison.stages[0].policy_comparison_status,
            SaturationPolicyComparisonStatus::Regression
        );
    }

    #[test]
    fn incompatibility_is_not_evaluable_not_a_fabricated_regression() {
        let baseline = report(
            timestamp(2),
            "https://baseline.example",
            1_000_000_000,
            20.0,
            0,
        );
        let candidate = report(
            timestamp(1),
            "https://candidate.example",
            1_000_000_000,
            20.0,
            0,
        );
        let comparison = compare(
            &baseline,
            &candidate,
            &SaturationComparisonPolicy::default(),
        )
        .unwrap();

        assert!(!comparison.compatible);
        assert_eq!(comparison.status, SaturationComparisonStatus::NotEvaluable);
        assert!(!comparison.regression);
        assert!(comparison.stages.is_empty());
    }

    #[test]
    fn comparison_thresholds_are_finite_and_bounded() {
        let invalid = SaturationComparisonPolicy {
            max_p95_e2e_regression_percent: Some(f64::NAN),
            ..SaturationComparisonPolicy::default()
        };
        assert!(validate_comparison_policy(&invalid).is_err());
        let invalid = SaturationComparisonPolicy {
            max_error_percent_increase_points: Some(100.1),
            ..SaturationComparisonPolicy::default()
        };
        assert!(validate_comparison_policy(&invalid).is_err());
    }

    #[test]
    fn mathematical_gate_boundaries_tolerate_binary_rounding_only() {
        let gate = evaluated_gate(
            SaturationComparisonGateKind::P95E2eRegressionPercent,
            SaturationComparisonGateOperator::LessThanOrEqual,
            Some(100.0),
            Some(110.0),
            10.0,
            None,
            Some(20),
            Some(20),
            Some(20),
            |before, after| ((after / before) - 1.0) * 100.0,
        );
        assert_eq!(gate.status, SaturationComparisonGateStatus::Pass);

        let gate = evaluated_gate(
            SaturationComparisonGateKind::SuccessfulRequestsPerSecondRatio,
            SaturationComparisonGateOperator::GreaterThanOrEqual,
            Some(10.0),
            Some(9.0),
            0.9,
            None,
            Some(20),
            Some(20),
            Some(20),
            |before, after| after / before,
        );
        assert_eq!(gate.status, SaturationComparisonGateStatus::Pass);

        let gate = evaluated_gate(
            SaturationComparisonGateKind::ErrorPercentIncreasePoints,
            SaturationComparisonGateOperator::LessThanOrEqual,
            Some(0.1),
            Some(0.3),
            0.2,
            None,
            Some(20),
            Some(20),
            Some(20),
            |before, after| after - before,
        );
        assert_eq!(gate.status, SaturationComparisonGateStatus::Pass);
    }
}
