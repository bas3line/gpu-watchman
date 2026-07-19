//! Offline comparison and regression gating for saved active-canary reports.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::domain::{
    CANARY_VERSION, CanaryAttempt, CanaryDistribution, CanaryFailureStage, CanaryPolicy,
    CanaryReport, CanaryStatus, MAX_CANARY_ATTEMPTS, MAX_CANARY_FAILURE_BYTES, MAX_CANARY_GATES,
    MAX_CANARY_IDENTITY_BYTES, MAX_CANARY_RETAINED_STRING_BYTES, valid_canary_workload_id,
};
use crate::operations::canary::{
    canonical_gates, recompute_summary, summary_evidence_matches, validate_policy,
};
use crate::operations::report_input::load_json_or_final_ndjson;
use crate::presentation::safe_inline;

pub const ROLLOUT_VERSION: u32 = 1;
pub const MIN_ROLLOUT_METRIC_SAMPLES: usize = 20;

const MAX_CANARY_REPORT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CONCURRENCY: u32 = 64;
const MAX_COMPLETION_TOKENS: u32 = 65_536;
const MAX_REQUESTED_COMPLETION_TOKENS: u64 = 1_000_000;
const MAX_TIMEOUT_MS: u64 = 5 * 60 * 1_000;
const MAX_PLANNED_DURATION_MS: u64 = 15 * 60 * 1_000;
const MAX_RESPONSE_LIMIT_BYTES: usize = 8 << 20;
const CANONICAL_ROUTE: &str = "chat_completions";

/// Optional rollout gates. A selected metric must be complete and valid in
/// both reports; unavailable evidence never passes a gate.
#[derive(Debug, Clone, Default)]
pub struct RolloutThresholds {
    pub max_p95_ttft_regression_percent: Option<f64>,
    pub max_p95_e2e_regression_percent: Option<f64>,
    pub min_p50_output_tokens_per_second_ratio: Option<f64>,
    pub max_success_percent_drop: Option<f64>,
}

/// Non-secret request identity retained in the comparison evidence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RolloutCanaryIdentity {
    pub canary_version: u32,
    pub workload_id: String,
    pub model: String,
    pub route: String,
    pub stream: bool,
    pub count: u32,
    pub concurrency: u32,
    pub max_tokens: u32,
    pub timeout_ms: u64,
    pub response_limit_bytes: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<CanaryPolicy>,
}

/// One compatibility condition required before metrics can be trusted as a
/// like-for-like rollout comparison.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RolloutCompatibilityCheck {
    pub field: String,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub detail: String,
}

/// One selected quantitative rollout gate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RolloutGate {
    pub name: String,
    pub operator: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed: Option<f64>,
    pub threshold: f64,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub detail: String,
}

/// Stable output contract for one offline baseline/candidate comparison.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CanaryRolloutComparison {
    pub rollout_version: u32,
    pub baseline_started_at: DateTime<Utc>,
    pub candidate_started_at: DateTime<Utc>,
    pub baseline_status: String,
    pub candidate_status: String,
    pub baseline: RolloutCanaryIdentity,
    pub candidate: RolloutCanaryIdentity,
    pub compatible: bool,
    pub compatibility: Vec<RolloutCompatibilityCheck>,
    pub gates: Vec<RolloutGate>,
    pub regression: bool,
}

/// Load a complete JSON canary report or the final non-empty record in an
/// NDJSON file. Reads are regular-file checked and hard bounded.
pub fn load_canary_report(path: &Path) -> Result<CanaryReport> {
    let report = load_json_or_final_ndjson(path, "canary report", MAX_CANARY_REPORT_BYTES)?;
    validate_canary_version(&report, path)?;
    Ok(report)
}

fn validate_canary_version(report: &CanaryReport, path: &Path) -> Result<()> {
    if report.canary_version == 0 || report.canary_version > CANARY_VERSION {
        bail!(
            "canary report {} uses unsupported canary version {} (this build supports 1..={})",
            path.display(),
            report.canary_version,
            CANARY_VERSION
        );
    }
    Ok(())
}

/// Compare two saved reports under explicit, fail-closed rollout gates.
pub fn compare(
    baseline: &CanaryReport,
    candidate: &CanaryReport,
    thresholds: &RolloutThresholds,
) -> Result<CanaryRolloutComparison> {
    validate_thresholds(thresholds)?;
    validate_compare_version(baseline, "baseline")?;
    validate_compare_version(candidate, "candidate")?;
    let baseline_integrity = validate_report_semantics(baseline);
    let candidate_integrity = validate_report_semantics(candidate);
    let compatibility = compatibility_checks(
        baseline,
        candidate,
        &baseline_integrity,
        &candidate_integrity,
    );
    let compatible = compatibility.iter().all(|check| check.passed);
    let gates = if baseline_integrity.is_ok() && candidate_integrity.is_ok() {
        evaluate_gates(baseline, candidate, thresholds)
    } else {
        unavailable_gates(thresholds)
    };
    let regression = !compatible || gates.iter().any(|gate| !gate.passed);

    Ok(CanaryRolloutComparison {
        rollout_version: ROLLOUT_VERSION,
        baseline_started_at: baseline.started_at,
        candidate_started_at: candidate.started_at,
        baseline_status: baseline.status.as_str().to_owned(),
        candidate_status: candidate.status.as_str().to_owned(),
        baseline: identity(baseline),
        candidate: identity(candidate),
        compatible,
        compatibility,
        gates,
        regression,
    })
}

fn validate_compare_version(report: &CanaryReport, label: &str) -> Result<()> {
    if report.canary_version == 0 || report.canary_version > CANARY_VERSION {
        bail!(
            "{label} canary report uses an unsupported version (supported versions: 1..={CANARY_VERSION})"
        );
    }
    Ok(())
}

fn validate_thresholds(thresholds: &RolloutThresholds) -> Result<()> {
    for (name, threshold) in [
        (
            "maximum p95 TTFT regression percent",
            thresholds.max_p95_ttft_regression_percent,
        ),
        (
            "maximum p95 E2E regression percent",
            thresholds.max_p95_e2e_regression_percent,
        ),
        (
            "minimum p50 output-token-rate ratio",
            thresholds.min_p50_output_tokens_per_second_ratio,
        ),
    ] {
        if threshold.is_some_and(|value| !value.is_finite() || value < 0.0) {
            bail!("{name} must be finite and non-negative");
        }
    }
    if thresholds
        .max_success_percent_drop
        .is_some_and(|value| !value.is_finite() || !(0.0..=100.0).contains(&value))
    {
        bail!("maximum success-percent drop must be finite and between zero and 100");
    }
    Ok(())
}

fn identity(report: &CanaryReport) -> RolloutCanaryIdentity {
    RolloutCanaryIdentity {
        canary_version: report.canary_version,
        workload_id: bounded_identity_value(
            &report.workload_id,
            crate::domain::MAX_CANARY_WORKLOAD_ID_BYTES,
        ),
        model: bounded_identity_value(&report.target.model, MAX_CANARY_IDENTITY_BYTES),
        route: bounded_identity_value(&report.target.route, MAX_CANARY_RETAINED_STRING_BYTES),
        stream: report.target.stream,
        count: report.plan.count,
        concurrency: report.plan.concurrency,
        max_tokens: report.plan.max_tokens,
        timeout_ms: report.plan.timeout_ms,
        response_limit_bytes: report.plan.response_limit_bytes,
        policy: report
            .policy
            .as_ref()
            .filter(|policy| validate_policy(policy).is_ok())
            .cloned(),
    }
}

fn bounded_identity_value(value: &str, maximum: usize) -> String {
    if value.len() <= maximum {
        value.to_owned()
    } else {
        "<invalid>".to_owned()
    }
}

fn compatibility_checks(
    baseline: &CanaryReport,
    candidate: &CanaryReport,
    baseline_integrity: &Result<(), &'static str>,
    candidate_integrity: &Result<(), &'static str>,
) -> Vec<RolloutCompatibilityCheck> {
    vec![
        integrity_compatibility("baseline_integrity", baseline_integrity),
        integrity_compatibility("candidate_integrity", candidate_integrity),
        condition_compatibility(
            "canary_version",
            baseline.canary_version == CANARY_VERSION
                && candidate.canary_version == CANARY_VERSION
                && baseline.canary_version == candidate.canary_version,
            "reports must use the same current canary version",
        ),
        condition_compatibility(
            "timestamp_order",
            candidate.started_at >= baseline.started_at,
            "candidate timestamp is before baseline",
        ),
        condition_compatibility(
            "baseline_status",
            baseline.status == CanaryStatus::Pass,
            "baseline canary status is fail",
        ),
        condition_compatibility(
            "candidate_status",
            candidate.status == CanaryStatus::Pass,
            "candidate canary status is fail",
        ),
        workload_compatibility(&baseline.workload_id, &candidate.workload_id),
        string_compatibility("model", &baseline.target.model, &candidate.target.model),
        string_compatibility("route", &baseline.target.route, &candidate.target.route),
        equality_compatibility(
            "stream",
            &baseline.target.stream,
            &candidate.target.stream,
            false,
        ),
        equality_compatibility(
            "count",
            &baseline.plan.count,
            &candidate.plan.count,
            baseline.plan.count == 0 || candidate.plan.count == 0,
        ),
        equality_compatibility(
            "concurrency",
            &baseline.plan.concurrency,
            &candidate.plan.concurrency,
            baseline.plan.concurrency == 0 || candidate.plan.concurrency == 0,
        ),
        equality_compatibility(
            "max_tokens",
            &baseline.plan.max_tokens,
            &candidate.plan.max_tokens,
            baseline.plan.max_tokens == 0 || candidate.plan.max_tokens == 0,
        ),
        equality_compatibility(
            "timeout_ms",
            &baseline.plan.timeout_ms,
            &candidate.plan.timeout_ms,
            baseline.plan.timeout_ms == 0 || candidate.plan.timeout_ms == 0,
        ),
        equality_compatibility(
            "response_limit_bytes",
            &baseline.plan.response_limit_bytes,
            &candidate.plan.response_limit_bytes,
            baseline.plan.response_limit_bytes == 0 || candidate.plan.response_limit_bytes == 0,
        ),
        equality_compatibility(
            "policy",
            &baseline.policy,
            &candidate.policy,
            baseline.policy.is_none() || candidate.policy.is_none(),
        ),
    ]
}

fn integrity_compatibility(
    field: &str,
    result: &Result<(), &'static str>,
) -> RolloutCompatibilityCheck {
    match result {
        Ok(()) => condition_compatibility(field, true, ""),
        Err(detail) => condition_compatibility(field, false, detail),
    }
}

fn condition_compatibility(
    field: &str,
    passed: bool,
    failed_detail: &str,
) -> RolloutCompatibilityCheck {
    RolloutCompatibilityCheck {
        field: field.to_owned(),
        passed,
        detail: if passed {
            String::new()
        } else {
            failed_detail.to_owned()
        },
    }
}

fn string_compatibility(field: &str, baseline: &str, candidate: &str) -> RolloutCompatibilityCheck {
    let detail = if baseline.is_empty() || candidate.is_empty() {
        "missing identity in one or both reports".to_owned()
    } else if baseline == candidate {
        String::new()
    } else {
        "values differ".to_owned()
    };
    RolloutCompatibilityCheck {
        field: field.to_owned(),
        passed: detail.is_empty(),
        detail,
    }
}

fn workload_compatibility(baseline: &str, candidate: &str) -> RolloutCompatibilityCheck {
    if !valid_canary_workload_id(baseline) || !valid_canary_workload_id(candidate) {
        return RolloutCompatibilityCheck {
            field: "workload_id".to_owned(),
            passed: false,
            detail: "missing or invalid non-secret workload identity in one or both reports"
                .to_owned(),
        };
    }
    string_compatibility("workload_id", baseline, candidate)
}

fn equality_compatibility<T: PartialEq>(
    field: &str,
    baseline: &T,
    candidate: &T,
    invalid: bool,
) -> RolloutCompatibilityCheck {
    let detail = if invalid {
        "invalid zero value in one or both reports".to_owned()
    } else if baseline == candidate {
        String::new()
    } else {
        "values differ".to_owned()
    };
    RolloutCompatibilityCheck {
        field: field.to_owned(),
        passed: detail.is_empty(),
        detail,
    }
}

fn validate_report_semantics(report: &CanaryReport) -> Result<(), &'static str> {
    if report.canary_version != CANARY_VERSION {
        return Err("legacy canary reports lack current policy and integrity evidence");
    }
    if !valid_canary_workload_id(&report.workload_id) {
        return Err("workload identity is missing or invalid");
    }
    validate_target(report)?;
    validate_plan(report)?;
    let policy = report
        .policy
        .as_ref()
        .ok_or("canary policy evidence is missing")?;
    if validate_policy(policy).is_err() {
        return Err("canary policy evidence is invalid");
    }
    if !report.target.stream
        && (policy.max_ttft_ms.is_some() || policy.min_output_tokens_per_second.is_some())
    {
        return Err("non-stream canary policy contains stream-only gates");
    }
    if report.attempts.len() != usize::try_from(report.plan.count).unwrap_or(usize::MAX) {
        return Err("attempt count does not match the canary plan");
    }
    if report.attempts.len() > MAX_CANARY_ATTEMPTS || report.gates.len() > MAX_CANARY_GATES {
        return Err("canary evidence exceeds producer sequence limits");
    }
    for (expected_index, attempt) in report.attempts.iter().enumerate() {
        if attempt.index != u32::try_from(expected_index).unwrap_or(u32::MAX) {
            return Err("attempt indexes are not the exact ordered plan sequence");
        }
        validate_attempt(attempt, report.target.stream, policy.expectation_configured)?;
    }
    if !report.summary.achieved_requests_per_second.is_finite()
        || report.summary.achieved_requests_per_second < 0.0
    {
        return Err("achieved request rate is non-finite or negative");
    }
    if !achieved_rate_matches_duration(report) {
        return Err("achieved request rate is inconsistent with report duration");
    }
    let reconstructed = recompute_summary(
        &report.attempts,
        report.summary.achieved_requests_per_second,
    );
    if !summary_evidence_matches(&report.summary, &reconstructed) {
        return Err("summary counts, tokens, or distributions do not match attempts");
    }
    let expected_gates = canonical_gates(&reconstructed, &report.attempts, policy)
        .map_err(|_| "canary policy cannot produce canonical gates")?;
    if report.gates != expected_gates {
        return Err("canary gates are missing, duplicated, unknown, or non-canonical");
    }
    let expected_status = if expected_gates.iter().all(|gate| gate.passed) {
        CanaryStatus::Pass
    } else {
        CanaryStatus::Fail
    };
    if report.status != expected_status {
        return Err("canary status does not match canonical gate results");
    }
    Ok(())
}

fn validate_target(report: &CanaryReport) -> Result<(), &'static str> {
    if report.target.url.is_empty()
        || report.target.url.len() > MAX_CANARY_IDENTITY_BYTES
        || report.target.model.trim().is_empty()
        || report.target.model.len() > MAX_CANARY_IDENTITY_BYTES
        || report.target.route != CANONICAL_ROUTE
    {
        return Err("canary target identity is invalid");
    }
    let url = url::Url::parse(&report.target.url).map_err(|_| "canary target URL is invalid")?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.origin().ascii_serialization() != report.target.url
    {
        return Err("canary target URL is not a canonical redacted origin");
    }
    Ok(())
}

fn validate_plan(report: &CanaryReport) -> Result<(), &'static str> {
    let plan = &report.plan;
    if plan.count == 0
        || usize::try_from(plan.count).unwrap_or(usize::MAX) > MAX_CANARY_ATTEMPTS
        || plan.concurrency == 0
        || plan.concurrency > plan.count
        || plan.concurrency > MAX_CONCURRENCY
        || plan.max_tokens == 0
        || plan.max_tokens > MAX_COMPLETION_TOKENS
        || u64::from(plan.count).saturating_mul(u64::from(plan.max_tokens))
            > MAX_REQUESTED_COMPLETION_TOKENS
        || plan.timeout_ms == 0
        || plan.timeout_ms > MAX_TIMEOUT_MS
        || plan.response_limit_bytes == 0
        || plan.response_limit_bytes > MAX_RESPONSE_LIMIT_BYTES
    {
        return Err("canary plan is outside producer safety limits");
    }
    let waves = plan.count.div_ceil(plan.concurrency);
    if plan
        .timeout_ms
        .checked_mul(u64::from(waves))
        .is_none_or(|duration| duration > MAX_PLANNED_DURATION_MS)
    {
        return Err("canary plan exceeds the producer wall-time limit");
    }
    Ok(())
}

fn validate_attempt(
    attempt: &CanaryAttempt,
    stream: bool,
    expectation_configured: bool,
) -> Result<(), &'static str> {
    if attempt.model.len() > MAX_CANARY_IDENTITY_BYTES
        || attempt.finish_reason.len() > MAX_CANARY_RETAINED_STRING_BYTES
        || !attempt.model.is_empty()
        || !matches!(
            attempt.finish_reason.as_str(),
            "" | "stop" | "length" | "tool_calls" | "content_filter" | "function_call"
        )
    {
        return Err("attempt retained identity strings are not producer-canonical");
    }
    for metric in [
        attempt.headers_ms,
        attempt.ttft_ms,
        attempt.e2e_ms,
        attempt.output_tokens_per_second,
    ] {
        if metric.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err("attempt contains a non-finite or negative metric");
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
    {
        return Err("attempt timing order is inconsistent");
    }
    if !stream && (attempt.ttft_ms.is_some() || attempt.output_tokens_per_second.is_some()) {
        return Err("non-stream attempt contains streaming-only metrics");
    }
    if attempt.output_tokens_per_second.is_some()
        && (!stream
            || attempt.ttft_ms.is_none()
            || attempt.e2e_ms.is_none()
            || attempt.completion_tokens.is_none_or(|tokens| tokens < 2))
    {
        return Err("attempt output-token rate lacks authoritative prerequisites");
    }
    if attempt.success != attempt.failure.is_none() {
        return Err("attempt success and failure evidence disagree");
    }
    if attempt.success {
        validate_completed_attempt(attempt, stream, expectation_configured, true)?;
        return Ok(());
    }
    let failure = attempt
        .failure
        .as_ref()
        .ok_or("failed attempt lacks failure evidence")?;
    if failure.message.is_empty() || failure.message.len() > MAX_CANARY_FAILURE_BYTES {
        return Err("attempt failure message is missing or oversized");
    }
    match failure.stage {
        CanaryFailureStage::Transport => {
            if attempt.status_code != 0 {
                return Err("transport failure has an HTTP status");
            }
        }
        CanaryFailureStage::Http => {
            if attempt.status_code == 0 || (200..300).contains(&attempt.status_code) {
                return Err("HTTP failure status is invalid");
            }
        }
        CanaryFailureStage::Protocol | CanaryFailureStage::EmptyOutput => {
            if !(200..300).contains(&attempt.status_code) {
                return Err("response failure lacks a successful HTTP status");
            }
        }
        CanaryFailureStage::Expectation => {
            validate_completed_attempt(attempt, stream, expectation_configured, false)?;
            return Ok(());
        }
    }
    if attempt.ttft_ms.is_some()
        || attempt.prompt_tokens.is_some()
        || attempt.completion_tokens.is_some()
        || attempt.output_tokens_per_second.is_some()
        || !attempt.finish_reason.is_empty()
        || attempt.expectation_met.is_some()
    {
        return Err("failed attempt retains fields the producer discards");
    }
    Ok(())
}

fn validate_completed_attempt(
    attempt: &CanaryAttempt,
    stream: bool,
    expectation_configured: bool,
    success: bool,
) -> Result<(), &'static str> {
    if !(200..300).contains(&attempt.status_code)
        || attempt.headers_ms.is_none()
        || attempt.e2e_ms.is_none()
        || (stream && attempt.ttft_ms.is_none())
    {
        return Err("completed attempt lacks canonical transport timing evidence");
    }
    let expected_marker = if expectation_configured {
        Some(success)
    } else {
        None
    };
    if attempt.expectation_met != expected_marker {
        return Err("attempt expectation evidence disagrees with canary policy");
    }
    if !success
        && (!expectation_configured
            || attempt
                .failure
                .as_ref()
                .is_none_or(|failure| failure.stage != CanaryFailureStage::Expectation))
    {
        return Err("expectation failure is not canonical for the configured policy");
    }
    Ok(())
}

fn achieved_rate_matches_duration(report: &CanaryReport) -> bool {
    if report.duration_ms == 0 {
        return true;
    }
    let count = f64::from(report.plan.count) * 1_000.0;
    #[allow(clippy::cast_precision_loss)]
    let upper = count / report.duration_ms as f64;
    #[allow(clippy::cast_precision_loss)]
    let lower = count / report.duration_ms.saturating_add(1) as f64;
    let tolerance = upper.abs().max(1.0) * 1.0e-9;
    report.summary.achieved_requests_per_second + tolerance >= lower
        && report.summary.achieved_requests_per_second <= upper + tolerance
}

fn unavailable_gates(thresholds: &RolloutThresholds) -> Vec<RolloutGate> {
    let mut gates = Vec::new();
    for (name, operator, threshold) in [
        (
            "max_p95_ttft_regression_percent",
            "<=",
            thresholds.max_p95_ttft_regression_percent,
        ),
        (
            "max_p95_e2e_regression_percent",
            "<=",
            thresholds.max_p95_e2e_regression_percent,
        ),
        (
            "min_p50_output_tokens_per_second_ratio",
            ">=",
            thresholds.min_p50_output_tokens_per_second_ratio,
        ),
        (
            "max_success_percent_drop",
            "<=",
            thresholds.max_success_percent_drop,
        ),
    ] {
        if let Some(threshold) = threshold {
            gates.push(RolloutGate {
                name: name.to_owned(),
                operator: operator.to_owned(),
                baseline: None,
                candidate: None,
                observed: None,
                threshold,
                passed: false,
                detail: "canary report integrity validation failed".to_owned(),
            });
        }
    }
    gates
}

fn evaluate_gates(
    baseline: &CanaryReport,
    candidate: &CanaryReport,
    thresholds: &RolloutThresholds,
) -> Vec<RolloutGate> {
    let mut gates = Vec::new();
    if let Some(threshold) = thresholds.max_p95_ttft_regression_percent {
        gates.push(regression_percent_gate(
            "max_p95_ttft_regression_percent",
            distribution_metric(
                baseline,
                |summary| summary.ttft_ms.as_ref(),
                |value| value.p95,
            ),
            distribution_metric(
                candidate,
                |summary| summary.ttft_ms.as_ref(),
                |value| value.p95,
            ),
            threshold,
        ));
    }
    if let Some(threshold) = thresholds.max_p95_e2e_regression_percent {
        gates.push(regression_percent_gate(
            "max_p95_e2e_regression_percent",
            distribution_metric(
                baseline,
                |summary| summary.e2e_ms.as_ref(),
                |value| value.p95,
            ),
            distribution_metric(
                candidate,
                |summary| summary.e2e_ms.as_ref(),
                |value| value.p95,
            ),
            threshold,
        ));
    }
    if let Some(threshold) = thresholds.min_p50_output_tokens_per_second_ratio {
        gates.push(ratio_gate(
            "min_p50_output_tokens_per_second_ratio",
            distribution_metric(
                baseline,
                |summary| summary.output_tokens_per_second.as_ref(),
                |value| value.p50,
            ),
            distribution_metric(
                candidate,
                |summary| summary.output_tokens_per_second.as_ref(),
                |value| value.p50,
            ),
            threshold,
        ));
    }
    if let Some(threshold) = thresholds.max_success_percent_drop {
        gates.push(success_drop_gate(baseline, candidate, threshold));
    }
    gates
}

fn distribution_metric(
    report: &CanaryReport,
    select: impl FnOnce(&crate::domain::CanarySummary) -> Option<&CanaryDistribution>,
    value: impl FnOnce(&CanaryDistribution) -> f64,
) -> Result<f64, &'static str> {
    if report.summary.succeeded == 0 {
        return Err("report has no successful requests");
    }
    let distribution = select(&report.summary).ok_or("measurement is missing")?;
    let succeeded = usize::try_from(report.summary.succeeded).unwrap_or(usize::MAX);
    if distribution.samples != succeeded {
        return Err("measurement coverage is incomplete");
    }
    if distribution.samples < MIN_ROLLOUT_METRIC_SAMPLES {
        return Err("measurement has fewer than 20 complete samples");
    }
    let values = [
        distribution.min,
        distribution.mean,
        distribution.p50,
        distribution.p95,
        distribution.p99,
        distribution.max,
    ];
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("distribution contains a non-finite or negative measurement");
    }
    if distribution.min > distribution.mean
        || distribution.mean > distribution.max
        || distribution.min > distribution.p50
        || distribution.p50 > distribution.p95
        || distribution.p95 > distribution.p99
        || distribution.p99 > distribution.max
    {
        return Err("distribution ordering is inconsistent");
    }
    let value = value(distribution);
    Ok(value)
}

fn regression_percent_gate(
    name: &str,
    baseline: Result<f64, &'static str>,
    candidate: Result<f64, &'static str>,
    threshold: f64,
) -> RolloutGate {
    evaluated_gate(
        name,
        "<=",
        baseline,
        candidate,
        threshold,
        |before, after| (after - before) * 100.0 / before,
    )
}

fn ratio_gate(
    name: &str,
    baseline: Result<f64, &'static str>,
    candidate: Result<f64, &'static str>,
    threshold: f64,
) -> RolloutGate {
    evaluated_gate(
        name,
        ">=",
        baseline,
        candidate,
        threshold,
        |before, after| after / before,
    )
}

fn evaluated_gate(
    name: &str,
    operator: &str,
    baseline: Result<f64, &'static str>,
    candidate: Result<f64, &'static str>,
    threshold: f64,
    evaluate: impl FnOnce(f64, f64) -> f64,
) -> RolloutGate {
    let before = baseline.ok();
    let after = candidate.ok();
    let (observed, detail) = match (baseline, candidate) {
        (Err(reason), _) => (None, format!("baseline {reason}")),
        (_, Err(reason)) => (None, format!("candidate {reason}")),
        (Ok(0.0), Ok(_)) => (None, "baseline measurement is zero".to_owned()),
        (Ok(before), Ok(after)) => {
            let observed = evaluate(before, after);
            if observed.is_finite() {
                (Some(observed), String::new())
            } else {
                (None, "computed comparison is non-finite".to_owned())
            }
        }
    };
    let tolerance = 1.0e-9 * threshold.abs().max(1.0);
    let passed = detail.is_empty()
        && observed.is_some_and(|value| match operator {
            "<=" => value <= threshold + tolerance,
            ">=" => value + tolerance >= threshold,
            _ => false,
        });
    RolloutGate {
        name: name.to_owned(),
        operator: operator.to_owned(),
        baseline: before.filter(|value| value.is_finite()),
        candidate: after.filter(|value| value.is_finite()),
        observed,
        threshold,
        passed,
        detail,
    }
}

fn success_drop_gate(
    baseline: &CanaryReport,
    candidate: &CanaryReport,
    threshold: f64,
) -> RolloutGate {
    let before = valid_success_percent(baseline);
    let after = valid_success_percent(candidate);
    evaluated_gate(
        "max_success_percent_drop",
        "<=",
        before,
        after,
        threshold,
        |before, after| before - after,
    )
}

fn valid_success_percent(report: &CanaryReport) -> Result<f64, &'static str> {
    if report.summary.attempted == 0 {
        return Err("report has no attempted requests");
    }
    let value = report.summary.success_percent;
    if !value.is_finite() || !(0.0..=100.0).contains(&value) {
        return Err("measurement is non-finite or outside zero to 100");
    }
    Ok(value)
}

/// Render a compact human-readable representation of the stable comparison.
pub fn render_text(comparison: &CanaryRolloutComparison) -> String {
    let mut output = String::new();
    let verdict = if comparison.regression {
        "REGRESSION"
    } else {
        "PASS"
    };
    let _ = writeln!(output, "GPU Watchman canary rollout  {verdict}");
    let _ = writeln!(
        output,
        "Contract       rollout v{}",
        comparison.rollout_version
    );
    let _ = writeln!(
        output,
        "Window         {} -> {}",
        comparison.baseline_started_at, comparison.candidate_started_at
    );
    let _ = writeln!(
        output,
        "Status         {} -> {}",
        safe_inline(&comparison.baseline_status),
        safe_inline(&comparison.candidate_status)
    );
    let _ = writeln!(
        output,
        "Baseline       workload={} | model={} | route={} | stream={} | count={} | concurrency={} | max tokens={} | timeout={}ms | response cap={}",
        safe_inline(&comparison.baseline.workload_id),
        safe_inline(&comparison.baseline.model),
        safe_inline(&comparison.baseline.route),
        comparison.baseline.stream,
        comparison.baseline.count,
        comparison.baseline.concurrency,
        comparison.baseline.max_tokens,
        comparison.baseline.timeout_ms,
        comparison.baseline.response_limit_bytes
    );
    let _ = writeln!(
        output,
        "Candidate      workload={} | model={} | route={} | stream={} | count={} | concurrency={} | max tokens={} | timeout={}ms | response cap={}",
        safe_inline(&comparison.candidate.workload_id),
        safe_inline(&comparison.candidate.model),
        safe_inline(&comparison.candidate.route),
        comparison.candidate.stream,
        comparison.candidate.count,
        comparison.candidate.concurrency,
        comparison.candidate.max_tokens,
        comparison.candidate.timeout_ms,
        comparison.candidate.response_limit_bytes
    );
    let _ = writeln!(
        output,
        "Policy         baseline {} | candidate {}",
        render_policy(comparison.baseline.policy.as_ref()),
        render_policy(comparison.candidate.policy.as_ref())
    );
    let _ = writeln!(
        output,
        "Compatibility  {}",
        if comparison.compatible {
            "PASS"
        } else {
            "FAIL"
        }
    );
    for check in &comparison.compatibility {
        let _ = writeln!(
            output,
            "  {:<4} {:<16} {}",
            if check.passed { "PASS" } else { "FAIL" },
            check.field,
            safe_inline(&check.detail)
        );
    }
    output.push_str("Gates\n");
    if comparison.gates.is_empty() {
        output.push_str("  none selected\n");
    }
    for gate in &comparison.gates {
        let observed = gate
            .observed
            .map_or_else(|| "unavailable".to_owned(), |value| format!("{value:.3}"));
        let _ = writeln!(
            output,
            "  {:<4} {:<48} observed {} {} {:.3} {}",
            if gate.passed { "PASS" } else { "FAIL" },
            gate.name,
            observed,
            gate.operator,
            gate.threshold,
            safe_inline(&gate.detail)
        );
    }
    output
}

fn render_policy(policy: Option<&CanaryPolicy>) -> String {
    policy.map_or_else(
        || "missing".to_owned(),
        |policy| {
            format!(
                "success>={:.3}% ttft<={} e2e<={} output-tps>={} expectation={}",
                policy.min_success_percent,
                optional_number(policy.max_ttft_ms),
                optional_number(policy.max_e2e_ms),
                optional_number(policy.min_output_tokens_per_second),
                policy.expectation_configured
            )
        },
    )
}

fn optional_number(value: Option<f64>) -> String {
    value.map_or_else(|| "off".to_owned(), |value| format!("{value:.3}"))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::domain::{CanaryPlan, CanaryTarget};

    fn successful_attempt(index: u32, ttft_ms: f64, e2e_ms: f64, tps: f64) -> CanaryAttempt {
        CanaryAttempt {
            index,
            success: true,
            status_code: 200,
            headers_ms: Some(ttft_ms.min(10.0)),
            ttft_ms: Some(ttft_ms),
            e2e_ms: Some(e2e_ms),
            prompt_tokens: Some(4),
            completion_tokens: Some(3),
            output_tokens_per_second: Some(tps),
            model: String::new(),
            finish_reason: "stop".to_owned(),
            expectation_met: None,
            failure: None,
        }
    }

    fn report_with(
        count: u32,
        succeeded: u32,
        ttft_ms: f64,
        e2e_ms: f64,
        tps: f64,
    ) -> CanaryReport {
        let attempts = (0..count)
            .map(|index| {
                if index < succeeded {
                    successful_attempt(index, ttft_ms, e2e_ms, tps)
                } else {
                    CanaryAttempt::failed(index, CanaryFailureStage::Transport, "connection failed")
                }
            })
            .collect::<Vec<_>>();
        let policy = CanaryPolicy {
            min_success_percent: 90.0,
            max_ttft_ms: None,
            max_e2e_ms: None,
            min_output_tokens_per_second: None,
            expectation_configured: false,
        };
        let duration_ms = 50_000;
        let achieved_rate = f64::from(count) * 1_000.0 / 50_000.0;
        let summary = recompute_summary(&attempts, achieved_rate);
        let gates = canonical_gates(&summary, &attempts, &policy).unwrap();
        let status = if gates.iter().all(|gate| gate.passed) {
            CanaryStatus::Pass
        } else {
            CanaryStatus::Fail
        };
        CanaryReport {
            canary_version: CANARY_VERSION,
            started_at: "2026-07-18T00:00:00Z".parse().unwrap(),
            duration_ms,
            status,
            workload_id: "synthetic-chat-v1".to_owned(),
            target: CanaryTarget {
                url: "https://inference.example".to_owned(),
                route: CANONICAL_ROUTE.to_owned(),
                model: "served-model".to_owned(),
                stream: true,
            },
            plan: CanaryPlan {
                count,
                concurrency: 4.min(count),
                max_tokens: 32,
                timeout_ms: 30_000,
                response_limit_bytes: 1 << 20,
            },
            policy: Some(policy),
            summary,
            gates,
            attempts,
        }
    }

    fn report() -> CanaryReport {
        report_with(100, 100, 100.0, 500.0, 50.0)
    }

    fn rebuild(report: &mut CanaryReport) {
        let duration_seconds =
            std::time::Duration::from_millis(report.duration_ms.max(1)).as_secs_f64();
        let achieved_rate = f64::from(report.plan.count) / duration_seconds;
        report.summary = recompute_summary(&report.attempts, achieved_rate);
        report.gates = canonical_gates(
            &report.summary,
            &report.attempts,
            report.policy.as_ref().unwrap(),
        )
        .unwrap();
        report.status = if report.gates.iter().all(|gate| gate.passed) {
            CanaryStatus::Pass
        } else {
            CanaryStatus::Fail
        };
    }

    fn all_thresholds() -> RolloutThresholds {
        RolloutThresholds {
            max_p95_ttft_regression_percent: Some(10.0),
            max_p95_e2e_regression_percent: Some(10.0),
            min_p50_output_tokens_per_second_ratio: Some(0.9),
            max_success_percent_drop: Some(1.0),
        }
    }

    #[test]
    fn like_for_like_reports_pass_all_selected_gates() {
        let baseline = report();
        let mut candidate = report_with(100, 99, 105.0, 525.0, 47.5);
        candidate.started_at = "2026-07-18T00:05:00Z".parse().unwrap();

        let comparison = compare(&baseline, &candidate, &all_thresholds()).unwrap();

        assert_eq!(comparison.rollout_version, ROLLOUT_VERSION);
        assert!(comparison.compatible);
        assert!(!comparison.regression);
        assert!(comparison.gates.iter().all(|gate| gate.passed));
        assert!(render_text(&comparison).contains("canary rollout  PASS"));
    }

    #[test]
    fn each_compatibility_dimension_fails_closed() {
        let baseline = report();
        let mut candidates = Vec::new();
        let mut workload = report();
        workload.workload_id = "different-v1".to_owned();
        candidates.push(("workload_id", workload));
        let mut model = report();
        model.target.model = "different-model".to_owned();
        candidates.push(("model", model));
        let mut stream = report();
        stream.target.stream = false;
        candidates.push(("stream", stream));
        let mut count = report_with(99, 99, 100.0, 500.0, 50.0);
        count.started_at = "2026-07-18T00:05:00Z".parse().unwrap();
        candidates.push(("count", count));
        let mut concurrency = report();
        concurrency.plan.concurrency = 1;
        candidates.push(("concurrency", concurrency));
        let mut max_tokens = report();
        max_tokens.plan.max_tokens = 64;
        candidates.push(("max_tokens", max_tokens));
        let mut timeout = report();
        timeout.plan.timeout_ms = 20_000;
        candidates.push(("timeout_ms", timeout));
        let mut response_limit = report();
        response_limit.plan.response_limit_bytes = 2 << 20;
        candidates.push(("response_limit_bytes", response_limit));
        let mut route = report();
        route.target.route = "other_route".to_owned();
        candidates.push(("route", route));

        for (field, candidate) in candidates {
            let comparison = compare(&baseline, &candidate, &RolloutThresholds::default()).unwrap();
            assert!(!comparison.compatible, "{field}");
            assert!(comparison.regression, "{field}");
            assert!(
                !comparison
                    .compatibility
                    .iter()
                    .find(|check| check.field == field)
                    .unwrap()
                    .passed,
                "{field}"
            );
        }

        let mut different = report();
        different.workload_id = "different-v1".to_owned();
        different.target.model = "different-model".to_owned();
        let output =
            render_text(&compare(&baseline, &different, &RolloutThresholds::default()).unwrap());
        assert!(output.contains("workload=synthetic-chat-v1 | model=served-model"));
        assert!(output.contains("workload=different-v1 | model=different-model"));
        let value = serde_json::to_value(
            compare(&baseline, &different, &RolloutThresholds::default()).unwrap(),
        )
        .unwrap();
        assert_eq!(value["baseline"]["route"], CANONICAL_ROUTE);
        assert_eq!(value["baseline"]["count"], 100);
        assert_eq!(value["baseline"]["timeout_ms"], 30_000);
        assert_eq!(value["baseline"]["response_limit_bytes"], 1 << 20);
        assert_eq!(value["baseline"]["policy"]["min_success_percent"], 90.0);
    }

    #[test]
    fn version_one_missing_workload_identity_is_incompatible() {
        let mut baseline = report();
        baseline.canary_version = 1;
        baseline.workload_id.clear();
        baseline.policy = None;
        let mut candidate = baseline.clone();
        candidate.started_at = "2026-07-18T00:05:00Z".parse().unwrap();

        let comparison = compare(&baseline, &candidate, &RolloutThresholds::default()).unwrap();

        assert!(!comparison.compatible);
        assert!(comparison.regression);
        assert!(
            comparison
                .compatibility
                .iter()
                .find(|check| check.field == "workload_id")
                .unwrap()
                .detail
                .contains("missing")
        );
    }

    #[test]
    fn selected_gates_detect_regressions() {
        let baseline = report();
        let candidate = report_with(100, 95, 125.0, 600.0, 40.0);

        let comparison = compare(&baseline, &candidate, &all_thresholds()).unwrap();

        assert!(comparison.compatible);
        assert!(comparison.regression);
        assert!(comparison.gates.iter().all(|gate| !gate.passed));
    }

    #[test]
    fn exact_regression_boundary_passes_without_float_noise() {
        let baseline = report();
        let candidate = report_with(100, 100, 110.0, 500.0, 50.0);

        let comparison = compare(
            &baseline,
            &candidate,
            &RolloutThresholds {
                max_p95_ttft_regression_percent: Some(10.0),
                ..RolloutThresholds::default()
            },
        )
        .unwrap();

        assert!(comparison.gates[0].passed);
        assert!((comparison.gates[0].observed.unwrap() - 10.0).abs() < 1.0e-12);
    }

    #[test]
    fn failed_or_time_reversed_candidate_regresses_without_metric_gates() {
        let baseline = report();
        let mut candidate = report_with(100, 89, 100.0, 500.0, 50.0);
        candidate.started_at = "2026-07-17T23:59:59Z".parse().unwrap();

        let comparison = compare(&baseline, &candidate, &RolloutThresholds::default()).unwrap();

        assert!(comparison.gates.is_empty());
        assert!(!comparison.compatible);
        assert!(comparison.regression);
        assert!(
            !comparison
                .compatibility
                .iter()
                .find(|check| check.field == "candidate_status")
                .unwrap()
                .passed
        );
        assert!(
            !comparison
                .compatibility
                .iter()
                .find(|check| check.field == "timestamp_order")
                .unwrap()
                .passed
        );
    }

    #[test]
    fn missing_incomplete_nonfinite_and_zero_baselines_fail_closed() {
        let candidate = report();
        let mut cases = Vec::new();
        let mut missing = report();
        for attempt in &mut missing.attempts {
            attempt.output_tokens_per_second = None;
        }
        rebuild(&mut missing);
        cases.push(missing);
        let too_few = report_with(19, 19, 100.0, 500.0, 50.0);
        cases.push(too_few);
        let mut nonfinite = report();
        nonfinite.attempts[0].ttft_ms = Some(f64::NAN);
        cases.push(nonfinite);
        let zero = report_with(100, 100, 0.0, 500.0, 50.0);
        cases.push(zero);

        for (index, baseline) in cases.into_iter().enumerate() {
            let comparison = compare(
                &baseline,
                &candidate,
                &RolloutThresholds {
                    max_p95_ttft_regression_percent: (index != 0).then_some(10.0),
                    min_p50_output_tokens_per_second_ratio: (index == 0).then_some(0.9),
                    ..RolloutThresholds::default()
                },
            )
            .unwrap();
            assert!(comparison.regression);
            assert!(!comparison.gates[0].passed);
            assert!(comparison.gates[0].observed.is_none());
        }

        let zero_success = report_with(100, 0, 100.0, 500.0, 50.0);
        let comparison = compare(
            &zero_success,
            &candidate,
            &RolloutThresholds {
                max_success_percent_drop: Some(0.0),
                ..RolloutThresholds::default()
            },
        )
        .unwrap();
        assert!(!comparison.gates[0].passed);
        assert!(comparison.gates[0].detail.contains("zero"));
    }

    #[test]
    fn policy_weakening_and_gate_tampering_are_never_green() {
        let baseline = report();
        let mut weakened = report();
        weakened.policy.as_mut().unwrap().min_success_percent = 80.0;
        rebuild(&mut weakened);
        let comparison = compare(&baseline, &weakened, &RolloutThresholds::default()).unwrap();
        assert!(comparison.regression);
        assert!(
            !comparison
                .compatibility
                .iter()
                .find(|check| check.field == "policy")
                .unwrap()
                .passed
        );

        let mut forged = report();
        forged.gates[1].threshold = 0.0;
        forged.gates[1].passed = true;
        let comparison = compare(&baseline, &forged, &RolloutThresholds::default()).unwrap();
        assert!(comparison.regression);
        assert!(
            !comparison
                .compatibility
                .iter()
                .find(|check| check.field == "candidate_integrity")
                .unwrap()
                .passed
        );
    }

    #[test]
    fn forged_attempts_summaries_and_distributions_are_rejected() {
        let baseline = report();
        let mut no_attempts = report();
        no_attempts.attempts.clear();
        assert!(
            compare(&baseline, &no_attempts, &RolloutThresholds::default())
                .unwrap()
                .regression
        );

        let mut forged_summary = report();
        forged_summary.summary.succeeded = 0;
        forged_summary.summary.failed = 100;
        forged_summary.summary.success_percent = 0.0;
        assert!(
            compare(&baseline, &forged_summary, &RolloutThresholds::default())
                .unwrap()
                .regression
        );

        let mut forged_distribution = report();
        forged_distribution.summary.ttft_ms.as_mut().unwrap().p95 += 1.0;
        assert!(
            compare(
                &baseline,
                &forged_distribution,
                &RolloutThresholds::default()
            )
            .unwrap()
            .regression
        );

        let mut duplicate_index = report();
        duplicate_index.attempts[1].index = 0;
        assert!(
            compare(&baseline, &duplicate_index, &RolloutThresholds::default())
                .unwrap()
                .regression
        );
    }

    #[test]
    fn loads_pretty_json_and_final_ndjson_record_with_hard_bounds() {
        let directory = tempfile::tempdir().unwrap();
        let pretty = directory.path().join("canary.json");
        fs::write(&pretty, serde_json::to_string_pretty(&report()).unwrap()).unwrap();
        assert_eq!(
            load_canary_report(&pretty).unwrap().workload_id,
            "synthetic-chat-v1"
        );

        let ndjson = directory.path().join("canary.ndjson");
        let mut second = report();
        second.workload_id = "synthetic-chat-v2".to_owned();
        fs::write(
            &ndjson,
            format!(
                "{}\n{}\n",
                serde_json::to_string(&report()).unwrap(),
                serde_json::to_string(&second).unwrap()
            ),
        )
        .unwrap();
        assert_eq!(
            load_canary_report(&ndjson).unwrap().workload_id,
            "synthetic-chat-v2"
        );

        let oversized = directory.path().join("oversized.json");
        fs::File::create(&oversized)
            .unwrap()
            .set_len(MAX_CANARY_REPORT_BYTES + 1)
            .unwrap();
        assert!(load_canary_report(&oversized).is_err());
        assert!(load_canary_report(directory.path()).is_err());
    }

    #[test]
    fn rejects_future_versions_and_invalid_thresholds() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("future.json");
        let mut future = report();
        future.canary_version = CANARY_VERSION + 1;
        fs::write(&path, serde_json::to_string(&future).unwrap()).unwrap();
        assert!(load_canary_report(&path).is_err());
        assert!(
            compare(&report(), &future, &RolloutThresholds::default())
                .unwrap_err()
                .to_string()
                .contains("unsupported version")
        );
        assert!(
            compare(
                &report(),
                &report(),
                &RolloutThresholds {
                    min_p50_output_tokens_per_second_ratio: Some(f64::INFINITY),
                    ..RolloutThresholds::default()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn bounded_deserializers_reject_compact_sequence_and_string_bombs() {
        let report = report();
        let attempt = serde_json::to_value(&report.attempts[0]).unwrap();
        let mut value = serde_json::to_value(&report).unwrap();
        value["attempts"] = serde_json::Value::Array(vec![attempt; MAX_CANARY_ATTEMPTS + 1]);
        assert!(
            serde_json::from_str::<CanaryReport>(&serde_json::to_string(&value).unwrap()).is_err()
        );

        let gate = serde_json::to_value(&report.gates[0]).unwrap();
        let mut value = serde_json::to_value(&report).unwrap();
        value["gates"] = serde_json::Value::Array(vec![gate; MAX_CANARY_GATES + 1]);
        assert!(
            serde_json::from_str::<CanaryReport>(&serde_json::to_string(&value).unwrap()).is_err()
        );

        for (path, oversized) in [
            (
                "workload_id",
                "x".repeat(crate::domain::MAX_CANARY_WORKLOAD_ID_BYTES + 1),
            ),
            ("target.model", "x".repeat(MAX_CANARY_IDENTITY_BYTES + 1)),
            (
                "attempts.0.finish_reason",
                "x".repeat(MAX_CANARY_RETAINED_STRING_BYTES + 1),
            ),
            (
                "attempts.0.failure.message",
                "x".repeat(MAX_CANARY_FAILURE_BYTES + 1),
            ),
        ] {
            let mut value = serde_json::to_value(&report).unwrap();
            match path {
                "workload_id" => value["workload_id"] = oversized.into(),
                "target.model" => value["target"]["model"] = oversized.into(),
                "attempts.0.finish_reason" => {
                    value["attempts"][0]["finish_reason"] = oversized.into();
                }
                "attempts.0.failure.message" => {
                    value["attempts"][0]["failure"] = serde_json::json!({
                        "stage": "transport",
                        "message": oversized,
                    });
                }
                _ => unreachable!(),
            }
            let body = serde_json::to_string(&value).unwrap();
            let error = serde_json::from_str::<CanaryReport>(&body)
                .unwrap_err()
                .to_string();
            assert!(error.contains("safety limit"), "{path}: {error}");
            assert!(
                !error.contains(&"x".repeat(64)),
                "source excerpt leaked for {path}"
            );
        }
    }
}
