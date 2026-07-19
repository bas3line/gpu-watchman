//! Privacy-safe human-readable presentation for offline saturation comparisons.

use std::fmt::Write as _;

use chrono::SecondsFormat;

use crate::domain::{
    SaturationComparableStageEvidence, SaturationComparisonGate, SaturationComparisonGateKind,
    SaturationComparisonGateOperator, SaturationComparisonGateReason,
    SaturationComparisonGateStatus, SaturationComparisonNonclaim, SaturationComparisonReport,
    SaturationComparisonStatus, SaturationCompatibilityCheck, SaturationCompatibilityField,
    SaturationCompatibilityReason, SaturationPolicyComparisonStatus, SaturationRoute,
    SaturationRunStatus, SaturationStagePolicyStatus,
};
use crate::presentation::safe_inline;

/// Render an auditable comparison without endpoint origins, artifact paths, or attempts.
pub fn render(report: &SaturationComparisonReport) -> String {
    let mut output = String::new();
    let _ = writeln!(
        output,
        "GPU Watchman saturation comparison v{}  {}",
        report.saturation_comparison_version,
        comparison_status(report.status)
    );
    let _ = writeln!(
        output,
        "Outcome        compatible={} | regression={}",
        yes_no(report.compatible),
        yes_no(report.regression)
    );
    let _ = writeln!(
        output,
        "Runs           baseline={} ({}) | candidate={} ({})",
        report
            .baseline_started_at
            .to_rfc3339_opts(SecondsFormat::AutoSi, true),
        run_status(report.baseline_status),
        report
            .candidate_started_at
            .to_rfc3339_opts(SecondsFormat::AutoSi, true),
        run_status(report.candidate_status)
    );
    let _ = writeln!(
        output,
        "Baseline ID    model={} | workload={} | route={} | stream={}",
        safe_inline(&report.baseline.model),
        safe_inline(&report.baseline.workload_id),
        route(report.baseline.route),
        yes_no(report.baseline.stream)
    );
    let _ = writeln!(
        output,
        "Candidate ID   model={} | workload={} | route={} | stream={}",
        safe_inline(&report.candidate.model),
        safe_inline(&report.candidate.workload_id),
        route(report.candidate.route),
        yes_no(report.candidate.stream)
    );

    render_comparison_policy(&mut output, report);
    render_compatibility(&mut output, report);
    render_stages(&mut output, report);
    render_nonclaims(&mut output, report);
    output
}

fn render_comparison_policy(output: &mut String, report: &SaturationComparisonReport) {
    output.push_str("\nSELECTED COMPARISON GATES\n");
    let policy = &report.comparison_policy;
    let mut selected = 0_u8;

    if let Some(threshold) = policy.max_p95_ttft_regression_percent {
        selected = selected.saturating_add(1);
        let _ = writeln!(output, "  p95 TTFT regression <= {}", percent(threshold));
    }
    if let Some(threshold) = policy.max_p95_e2e_regression_percent {
        selected = selected.saturating_add(1);
        let _ = writeln!(output, "  p95 E2E regression <= {}", percent(threshold));
    }
    if let Some(threshold) = policy.min_successful_requests_per_second_ratio {
        selected = selected.saturating_add(1);
        let _ = writeln!(
            output,
            "  successful request rate ratio >= {}",
            ratio(threshold)
        );
    }
    if let Some(threshold) = policy.min_completion_token_goodput_per_second_ratio {
        selected = selected.saturating_add(1);
        let _ = writeln!(
            output,
            "  completion-token goodput ratio >= {}",
            ratio(threshold)
        );
    }
    if let Some(threshold) = policy.max_error_percent_increase_points {
        selected = selected.saturating_add(1);
        let _ = writeln!(
            output,
            "  error-rate increase <= {}",
            percentage_points(threshold)
        );
    }
    if selected == 0 {
        output.push_str("  none; source benchmark policy transitions are still compared\n");
    }
    let _ = writeln!(
        output,
        "  every quantitative gate requires {} relevant sample(s) on each side",
        policy.minimum_stage_samples
    );
}

fn render_compatibility(output: &mut String, report: &SaturationComparisonReport) {
    let passed = report
        .compatibility
        .iter()
        .filter(|check| check.passed)
        .count();
    let _ = writeln!(
        output,
        "\nCOMPATIBILITY  {} ({passed}/{} exact checks passed)",
        if report.compatible { "PASS" } else { "FAIL" },
        report.compatibility.len()
    );
    for check in &report.compatibility {
        render_compatibility_check(output, *check);
    }
}

fn render_compatibility_check(output: &mut String, check: SaturationCompatibilityCheck) {
    let status = if check.passed { "PASS" } else { "FAIL" };
    let reason = check.reason.map_or_else(String::new, |reason| {
        format!(" | {}", compatibility_reason(reason))
    });
    let _ = writeln!(
        output,
        "  {status:<4} {:<34}{reason}",
        compatibility_field(check.field)
    );
}

fn render_stages(output: &mut String, report: &SaturationComparisonReport) {
    output.push_str("\nEXACT-STAGE COMPARISON\n");
    if report.stages.is_empty() {
        if report.compatible {
            output.push_str("  none: no exact measured stage evidence was retained\n");
        } else {
            output.push_str(
                "  none: quantitative comparison is not evaluable because exact compatibility was not established\n",
            );
        }
    }

    for stage in &report.stages {
        let _ = writeln!(
            output,
            "\n  CONCURRENCY {}  SOURCE POLICY baseline={} candidate={} comparison={}",
            stage.concurrency,
            stage_policy_status(stage.baseline_policy_status),
            stage_policy_status(stage.candidate_policy_status),
            policy_comparison_status(stage.policy_comparison_status)
        );
        render_stage_evidence(output, "baseline ", &stage.baseline);
        render_stage_evidence(output, "candidate", &stage.candidate);

        if stage.gates.is_empty() {
            output.push_str("    gates: none selected for this stage\n");
        } else {
            output.push_str("    gates:\n");
            for gate in &stage.gates {
                render_gate(output, gate);
            }
        }
    }

    output.push_str(
        "\nN/E means required comparable evidence was unavailable or incomplete; it is never a pass.\n",
    );
    output.push_str(
        "A source-policy REGRESSION means the baseline stage passed its benchmark policy and the candidate stage did not.\n",
    );
}

fn render_stage_evidence(
    output: &mut String,
    side: &str,
    evidence: &SaturationComparableStageEvidence,
) {
    let _ = writeln!(
        output,
        "    {side}: ok={}/{} | error={} | successful={} | duration={} ns",
        evidence.succeeded,
        evidence.attempted,
        percent(evidence.error_percent),
        requests_per_second(evidence.successful_requests_per_second),
        evidence.duration_ns
    );
    let _ = writeln!(
        output,
        "              p95 TTFT={} ({} samples) | p95 E2E={} ({} samples)",
        optional_milliseconds(evidence.p95_ttft_ms),
        evidence.ttft_samples,
        optional_milliseconds(evidence.p95_e2e_ms),
        evidence.e2e_samples
    );
    let _ = writeln!(
        output,
        "              completion goodput={} | token samples={} | usage={}",
        optional_tokens_per_second(evidence.completion_token_goodput_per_second),
        evidence.completion_token_samples,
        if evidence.completion_token_usage_complete {
            "complete"
        } else {
            "incomplete"
        }
    );
}

fn render_gate(output: &mut String, gate: &SaturationComparisonGate) {
    let status = gate_status(gate.status);
    let baseline = gate.baseline.map_or_else(
        || "unavailable".to_owned(),
        |value| gate_side_value(gate.kind, value),
    );
    let candidate = gate.candidate.map_or_else(
        || "unavailable".to_owned(),
        |value| gate_side_value(gate.kind, value),
    );
    let observed = gate.observed.map_or_else(
        || "unavailable".to_owned(),
        |value| gate_comparison_value(gate.kind, value),
    );
    let threshold = gate_comparison_value(gate.kind, gate.threshold);
    let _ = writeln!(
        output,
        "      {status:<4} {} | baseline={baseline} candidate={candidate}",
        gate_name(gate.kind)
    );
    let _ = writeln!(
        output,
        "           observed={observed} {} threshold={threshold}",
        gate_operator(gate.operator)
    );

    if gate.baseline_samples.is_some()
        || gate.candidate_samples.is_some()
        || gate.required_samples.is_some()
    {
        let _ = writeln!(
            output,
            "           samples baseline={} candidate={} required={}",
            optional_samples(gate.baseline_samples),
            optional_samples(gate.candidate_samples),
            optional_samples(gate.required_samples)
        );
    }
    if let Some(reason) = gate.reason {
        let _ = writeln!(output, "           reason={}", gate_reason(reason));
    }
}

fn render_nonclaims(output: &mut String, report: &SaturationComparisonReport) {
    output.push_str("\nLIMITS (FIXED)\n");
    for nonclaim in &report.nonclaims {
        let _ = writeln!(output, "  - {}", nonclaim_text(*nonclaim));
    }
}

const fn comparison_status(status: SaturationComparisonStatus) -> &'static str {
    match status {
        SaturationComparisonStatus::Pass => "PASS",
        SaturationComparisonStatus::Regression => "REGRESSION",
        SaturationComparisonStatus::NotEvaluable => "NOT EVALUABLE",
    }
}

const fn run_status(status: SaturationRunStatus) -> &'static str {
    match status {
        SaturationRunStatus::Complete => "COMPLETE",
        SaturationRunStatus::Aborted => "ABORTED",
    }
}

const fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

const fn route(value: SaturationRoute) -> &'static str {
    match value {
        SaturationRoute::ChatCompletions => "chat_completions",
    }
}

const fn compatibility_field(field: SaturationCompatibilityField) -> &'static str {
    match field {
        SaturationCompatibilityField::BaselineIntegrity => "baseline report integrity",
        SaturationCompatibilityField::CandidateIntegrity => "candidate report integrity",
        SaturationCompatibilityField::ReportVersion => "report version",
        SaturationCompatibilityField::TimestampOrder => "timestamp order",
        SaturationCompatibilityField::BaselineStatus => "baseline run complete",
        SaturationCompatibilityField::CandidateStatus => "candidate run complete",
        SaturationCompatibilityField::WorkloadId => "workload identity",
        SaturationCompatibilityField::Model => "model identity",
        SaturationCompatibilityField::Route => "inference route",
        SaturationCompatibilityField::Stream => "streaming mode",
        SaturationCompatibilityField::ConcurrencyStages => "concurrency stages",
        SaturationCompatibilityField::WarmupRequestsPerWorker => "warmup requests per worker",
        SaturationCompatibilityField::RequestsPerWorker => "measured requests per worker",
        SaturationCompatibilityField::MaxTokens => "maximum completion tokens",
        SaturationCompatibilityField::TimeoutNanoseconds => "request timeout (nanoseconds)",
        SaturationCompatibilityField::ResponseLimitBytes => "response limit (bytes)",
        SaturationCompatibilityField::Schedule => "load schedule semantics",
        SaturationCompatibilityField::Policy => "source benchmark policy",
    }
}

const fn compatibility_reason(reason: SaturationCompatibilityReason) -> &'static str {
    match reason {
        SaturationCompatibilityReason::InvalidReportEvidence => {
            "saved report evidence failed consistency validation"
        }
        SaturationCompatibilityReason::UnsupportedVersion => "saved report version is unsupported",
        SaturationCompatibilityReason::CandidatePredatesBaseline => {
            "candidate timestamp predates baseline"
        }
        SaturationCompatibilityReason::RunDidNotComplete => "saved benchmark run did not complete",
        SaturationCompatibilityReason::InvalidIdentity => "comparison identity is invalid",
        SaturationCompatibilityReason::ValuesDiffer => "values differ",
    }
}

const fn stage_policy_status(status: SaturationStagePolicyStatus) -> &'static str {
    match status {
        SaturationStagePolicyStatus::Pass => "PASS",
        SaturationStagePolicyStatus::Fail => "FAIL",
        SaturationStagePolicyStatus::NotEvaluable => "N/E",
    }
}

const fn policy_comparison_status(status: SaturationPolicyComparisonStatus) -> &'static str {
    match status {
        SaturationPolicyComparisonStatus::Pass => "PASS",
        SaturationPolicyComparisonStatus::Regression => "REGRESSION",
        SaturationPolicyComparisonStatus::NotEvaluable => "N/E",
    }
}

const fn gate_name(kind: SaturationComparisonGateKind) -> &'static str {
    match kind {
        SaturationComparisonGateKind::P95TtftRegressionPercent => "p95 TTFT regression",
        SaturationComparisonGateKind::P95E2eRegressionPercent => "p95 E2E regression",
        SaturationComparisonGateKind::SuccessfulRequestsPerSecondRatio => {
            "successful request rate ratio"
        }
        SaturationComparisonGateKind::CompletionTokenGoodputPerSecondRatio => {
            "completion-token goodput ratio"
        }
        SaturationComparisonGateKind::ErrorPercentIncreasePoints => "error-rate increase",
    }
}

const fn gate_operator(operator: SaturationComparisonGateOperator) -> &'static str {
    match operator {
        SaturationComparisonGateOperator::LessThanOrEqual => "<=",
        SaturationComparisonGateOperator::GreaterThanOrEqual => ">=",
    }
}

const fn gate_status(status: SaturationComparisonGateStatus) -> &'static str {
    match status {
        SaturationComparisonGateStatus::Pass => "PASS",
        SaturationComparisonGateStatus::Fail => "FAIL",
        SaturationComparisonGateStatus::NotEvaluable => "N/E",
    }
}

const fn gate_reason(reason: SaturationComparisonGateReason) -> &'static str {
    match reason {
        SaturationComparisonGateReason::BaselineNoSuccessfulRequests => {
            "baseline has no successful requests"
        }
        SaturationComparisonGateReason::CandidateNoSuccessfulRequests => {
            "candidate has no successful requests"
        }
        SaturationComparisonGateReason::BaselineMeasurementUnavailable => {
            "baseline measurement is unavailable"
        }
        SaturationComparisonGateReason::CandidateMeasurementUnavailable => {
            "candidate measurement is unavailable"
        }
        SaturationComparisonGateReason::BaselineInsufficientSamples => {
            "baseline has fewer than the required samples"
        }
        SaturationComparisonGateReason::CandidateInsufficientSamples => {
            "candidate has fewer than the required samples"
        }
        SaturationComparisonGateReason::BaselineUsageIncomplete => {
            "baseline token-usage evidence is incomplete"
        }
        SaturationComparisonGateReason::CandidateUsageIncomplete => {
            "candidate token-usage evidence is incomplete"
        }
        SaturationComparisonGateReason::BaselineMeasurementZero => {
            "baseline measurement is zero, so a ratio cannot be computed"
        }
        SaturationComparisonGateReason::NonFiniteComparison => "comparison result is not finite",
    }
}

const fn nonclaim_text(nonclaim: SaturationComparisonNonclaim) -> &'static str {
    match nonclaim {
        SaturationComparisonNonclaim::RecomputedConsistencyDoesNotAuthenticateSavedReports => {
            "Recomputed consistency does not authenticate or prove provenance of saved reports."
        }
        SaturationComparisonNonclaim::MatchingWorkloadIdAndExpectationFlagDoNotProveHiddenInputsMatch => {
            "Matching workload identity and expectation flag do not prove hidden prompts, expectations, or request inputs match."
        }
        SaturationComparisonNonclaim::ExactLikeForLikeTestedStagesOnly => {
            "Only exact, like-for-like concurrency points in complete runs are compared."
        }
        SaturationComparisonNonclaim::ClosedLoopMayHideCoordinatedOmission => {
            "Closed-loop load can hide coordinated omission and arrival-rate tail behavior."
        }
        SaturationComparisonNonclaim::SingleLoadGeneratorMayBeTheBottleneck => {
            "A single load generator can itself be the throughput or latency bottleneck."
        }
        SaturationComparisonNonclaim::TokenGoodputUsesEndpointReportedUsage => {
            "Token goodput uses plausible endpoint-reported usage, not independently measured GPU work."
        }
        SaturationComparisonNonclaim::RuntimeHardwareNetworkAndExternalTrafficAreNotControlled => {
            "Runtime, hardware, network, and external traffic are not controlled by this offline comparison."
        }
        SaturationComparisonNonclaim::NoCausalAttribution => {
            "Observed differences do not establish causal attribution."
        }
        SaturationComparisonNonclaim::NoStatisticalSignificanceOrConfidenceInterval => {
            "No statistical-significance test or confidence interval is provided."
        }
        SaturationComparisonNonclaim::OperatorThresholdsAreNotUniversalSlos => {
            "Operator-selected thresholds are not universal SLOs."
        }
        SaturationComparisonNonclaim::NoProductionCapacityOrRecommendation => {
            "This comparison does not establish production capacity or make a deployment recommendation."
        }
        SaturationComparisonNonclaim::NoSlaCertification => {
            "This comparison is not SLA certification."
        }
        SaturationComparisonNonclaim::EndpointOriginsIntentionallyExcluded => {
            "Endpoint origins are intentionally excluded and are neither displayed nor compared."
        }
    }
}

fn gate_side_value(kind: SaturationComparisonGateKind, value: f64) -> String {
    match kind {
        SaturationComparisonGateKind::P95TtftRegressionPercent
        | SaturationComparisonGateKind::P95E2eRegressionPercent => milliseconds(value),
        SaturationComparisonGateKind::SuccessfulRequestsPerSecondRatio => {
            requests_per_second(value)
        }
        SaturationComparisonGateKind::CompletionTokenGoodputPerSecondRatio => {
            tokens_per_second(value)
        }
        SaturationComparisonGateKind::ErrorPercentIncreasePoints => percent(value),
    }
}

fn gate_comparison_value(kind: SaturationComparisonGateKind, value: f64) -> String {
    match kind {
        SaturationComparisonGateKind::P95TtftRegressionPercent
        | SaturationComparisonGateKind::P95E2eRegressionPercent => percent(value),
        SaturationComparisonGateKind::SuccessfulRequestsPerSecondRatio
        | SaturationComparisonGateKind::CompletionTokenGoodputPerSecondRatio => ratio(value),
        SaturationComparisonGateKind::ErrorPercentIncreasePoints => percentage_points(value),
    }
}

fn milliseconds(value: f64) -> String {
    finite_with_unit(value, "ms")
}

fn percent(value: f64) -> String {
    finite_with_unit(value, "%")
}

fn percentage_points(value: f64) -> String {
    finite_with_unit(value, "percentage points")
}

fn ratio(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.6}x")
    } else {
        "unavailable".to_owned()
    }
}

fn requests_per_second(value: f64) -> String {
    finite_with_unit(value, "req/s")
}

fn tokens_per_second(value: f64) -> String {
    finite_with_unit(value, "token/s")
}

fn finite_with_unit(value: f64, unit: &str) -> String {
    if value.is_finite() {
        format!("{value:.3} {unit}")
    } else {
        "unavailable".to_owned()
    }
}

fn optional_milliseconds(value: Option<f64>) -> String {
    value.map_or_else(|| "unavailable".to_owned(), milliseconds)
}

fn optional_tokens_per_second(value: Option<f64>) -> String {
    value.map_or_else(|| "unavailable".to_owned(), tokens_per_second)
}

fn optional_samples(value: Option<u32>) -> String {
    value.map_or_else(|| "-".to_owned(), |samples| samples.to_string())
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use super::*;
    use crate::domain::{
        SATURATION_BENCHMARK_VERSION, SATURATION_COMPARISON_NONCLAIMS,
        SATURATION_COMPARISON_VERSION, SaturationComparisonIdentity, SaturationComparisonPolicy,
        SaturationPlan, SaturationPolicy, SaturationSchedule, SaturationStageComparison,
    };

    fn evidence(
        duration_ns: u64,
        requests_per_second: f64,
        p95_e2e_ms: f64,
    ) -> SaturationComparableStageEvidence {
        SaturationComparableStageEvidence {
            duration_ns,
            attempted: 20,
            succeeded: 20,
            error_percent: 0.0,
            successful_requests_per_second: requests_per_second,
            ttft_samples: 20,
            p95_ttft_ms: Some(25.0),
            e2e_samples: 20,
            p95_e2e_ms: Some(p95_e2e_ms),
            completion_token_samples: 20,
            completion_token_usage_complete: true,
            completion_token_goodput_per_second: Some(640.0),
        }
    }

    fn report() -> SaturationComparisonReport {
        let identity = SaturationComparisonIdentity {
            saturation_benchmark_version: SATURATION_BENCHMARK_VERSION,
            workload_id: "builtin-v1".to_owned(),
            model: "served-model\n\u{202e}safe".to_owned(),
            route: SaturationRoute::ChatCompletions,
            stream: true,
            plan: SaturationPlan {
                concurrency_stages: vec![4],
                warmup_requests_per_worker: 1,
                requests_per_worker: 5,
                planned_attempts: 24,
                max_tokens: 32,
                timeout_ns: 5_000_000_000,
                timeout_ms: 5_000,
                response_limit_bytes: 1_048_576,
                schedule: SaturationSchedule::default(),
            },
            policy: SaturationPolicy::default(),
        };
        SaturationComparisonReport {
            saturation_comparison_version: SATURATION_COMPARISON_VERSION,
            baseline_started_at: Utc.with_ymd_and_hms(2026, 7, 18, 12, 0, 0).unwrap(),
            candidate_started_at: Utc.with_ymd_and_hms(2026, 7, 19, 12, 0, 0).unwrap(),
            baseline_status: SaturationRunStatus::Complete,
            candidate_status: SaturationRunStatus::Complete,
            baseline: identity.clone(),
            candidate: identity,
            comparison_policy: SaturationComparisonPolicy {
                max_p95_e2e_regression_percent: Some(10.0),
                min_successful_requests_per_second_ratio: Some(0.95),
                ..SaturationComparisonPolicy::default()
            },
            compatible: true,
            compatibility: vec![SaturationCompatibilityCheck {
                field: SaturationCompatibilityField::BaselineIntegrity,
                passed: true,
                reason: None,
            }],
            stages: vec![SaturationStageComparison {
                concurrency: 4,
                baseline: evidence(2_000_000_000, 10.0, 100.0),
                candidate: evidence(2_100_000_000, 9.5, 112.0),
                baseline_policy_status: SaturationStagePolicyStatus::Pass,
                candidate_policy_status: SaturationStagePolicyStatus::Fail,
                policy_comparison_status: SaturationPolicyComparisonStatus::Regression,
                gates: vec![
                    SaturationComparisonGate {
                        kind: SaturationComparisonGateKind::P95E2eRegressionPercent,
                        operator: SaturationComparisonGateOperator::LessThanOrEqual,
                        baseline: Some(100.0),
                        candidate: Some(112.0),
                        observed: Some(12.0),
                        threshold: 10.0,
                        status: SaturationComparisonGateStatus::Fail,
                        reason: None,
                        baseline_samples: Some(20),
                        candidate_samples: Some(20),
                        required_samples: Some(20),
                    },
                    SaturationComparisonGate {
                        kind: SaturationComparisonGateKind::SuccessfulRequestsPerSecondRatio,
                        operator: SaturationComparisonGateOperator::GreaterThanOrEqual,
                        baseline: Some(10.0),
                        candidate: Some(9.5),
                        observed: Some(0.95),
                        threshold: 0.95,
                        status: SaturationComparisonGateStatus::Pass,
                        reason: None,
                        baseline_samples: None,
                        candidate_samples: None,
                        required_samples: None,
                    },
                ],
            }],
            status: SaturationComparisonStatus::Regression,
            regression: true,
            nonclaims: SATURATION_COMPARISON_NONCLAIMS.to_vec(),
        }
    }

    #[test]
    fn renders_exact_units_policy_transition_and_selected_gates() {
        let output = render(&report());

        assert!(output.contains("saturation comparison v1  REGRESSION"));
        assert!(output.contains("model=served-model safe"));
        assert!(output.contains(
            "CONCURRENCY 4  SOURCE POLICY baseline=PASS candidate=FAIL comparison=REGRESSION"
        ));
        assert!(output.contains("duration=2000000000 ns"));
        assert!(output.contains("p95 E2E=100.000 ms (20 samples)"));
        assert!(output.contains("completion goodput=640.000 token/s"));
        assert!(output.contains("observed=12.000 % <= threshold=10.000 %"));
        assert!(output.contains("observed=0.950000x >= threshold=0.950000x"));
        assert!(!output.contains('\u{202e}'));
        assert!(!output.contains('\u{1b}'));
    }

    #[test]
    fn explains_not_evaluable_and_never_emits_artifact_details() {
        let mut report = report();
        report.status = SaturationComparisonStatus::NotEvaluable;
        report.regression = false;
        report.compatible = false;
        report.compatibility = vec![SaturationCompatibilityCheck {
            field: SaturationCompatibilityField::CandidateIntegrity,
            passed: false,
            reason: Some(SaturationCompatibilityReason::InvalidReportEvidence),
        }];
        report.stages.clear();

        let output = render(&report);
        assert!(output.contains("NOT EVALUABLE"));
        assert!(output.contains("saved report evidence failed consistency validation"));
        assert!(output.contains("N/E means required comparable evidence was unavailable"));
        assert!(output.contains("Endpoint origins are intentionally excluded"));
        for forbidden in [
            "https://inference.example",
            "/tmp/baseline.json",
            "Authorization:",
            "attempt #",
            "response body",
        ] {
            assert!(!output.contains(forbidden));
        }
    }

    #[test]
    fn renders_fixed_not_evaluable_gate_reason_and_sample_counts() {
        let mut report = report();
        let gate = &mut report.stages[0].gates[0];
        gate.status = SaturationComparisonGateStatus::NotEvaluable;
        gate.observed = None;
        gate.reason = Some(SaturationComparisonGateReason::CandidateInsufficientSamples);
        gate.candidate_samples = Some(7);

        let output = render(&report);
        assert!(output.contains("N/E  p95 E2E regression"));
        assert!(output.contains("samples baseline=20 candidate=7 required=20"));
        assert!(output.contains("reason=candidate has fewer than the required samples"));
        assert!(output.contains("it is never a pass"));
    }
}
