//! Human-readable presentation for bounded saturation benchmark reports.

use std::fmt::Write as _;

use crate::domain::{
    SaturationAssessmentStatus, SaturationBenchmarkReport, SaturationGateStatus,
    SaturationRunStatus, SaturationSignal, SaturationVerificationStatus,
};
use crate::presentation::safe_inline;

/// Render a compact, auditable saturation ladder without request or response content.
pub fn render(report: &SaturationBenchmarkReport) -> String {
    let mut output = String::new();
    let status = match report.status {
        SaturationRunStatus::Complete => "COMPLETE",
        SaturationRunStatus::Aborted => "ABORTED",
    };
    let _ = writeln!(output, "GPU Watchman saturation benchmark  {status}");
    let _ = writeln!(
        output,
        "Target   {} | model={} | workload={} | stream={}",
        safe_inline(&report.target.url),
        safe_inline(&report.target.model),
        safe_inline(&report.workload_id),
        if report.target.stream { "yes" } else { "no" }
    );
    let stages = report
        .plan
        .concurrency_stages
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let _ = writeln!(
        output,
        "Plan     concurrency [{stages}] | {} measured request(s)/worker/stage | {} warmup request(s)/worker/stage | {} max tokens",
        report.plan.requests_per_worker,
        report.plan.warmup_requests_per_worker,
        report.plan.max_tokens
    );
    let _ = writeln!(
        output,
        "Bounds   {} planned attempt(s) | timeout {} | response cap {} | {} elapsed",
        report.plan.planned_attempts,
        elapsed_duration(report.plan.timeout_ms),
        bytes(report.plan.response_limit_bytes),
        elapsed_duration(report.duration_ms)
    );
    let _ = writeln!(
        output,
        "Warmup   {} excluded exact-stage phase(s)",
        report.warmups.len()
    );
    for warmup in &report.warmups {
        let _ = writeln!(
            output,
            "         concurrency {}: {}/{} passed | {:.1}% errors",
            warmup.concurrency, warmup.succeeded, warmup.attempted, warmup.error_percent
        );
    }
    if let Some(reason) = report.abort_reason {
        let _ = writeln!(output, "Abort    {}", abort_reason(reason));
    }

    render_stage_table(&mut output, report);
    render_assessment(&mut output, report);
    render_verification(&mut output, report);
    render_failures(&mut output, report);
    output.push_str(
        "\nSCOPE    Closed-loop, fixed-concurrency, single-process load from this client.\n",
    );
    output.push_str(
        "         Only the listed points were tested; coordinated omission and client/network bottlenecks remain possible.\n",
    );
    output.push_str(
        "         Highest accepted tested concurrency is not production capacity, a recommendation, GPU occupancy, or SLA certification.\n",
    );
    output
}

fn render_stage_table(output: &mut String, report: &SaturationBenchmarkReport) {
    output.push_str(
        "\n CONC   OK/TOTAL    ERROR   SUCCESS RPS   P95 TTFT    P95 E2E   COMPLETION TOK/S   GATES\n",
    );
    for stage in &report.stages {
        let p95_ttft = stage
            .summary
            .ttft_ms
            .as_ref()
            .map_or_else(|| "-".to_owned(), |value| duration(value.p95));
        let p95_e2e = stage
            .summary
            .e2e_ms
            .as_ref()
            .map_or_else(|| "-".to_owned(), |value| duration(value.p95));
        let token_goodput = stage
            .summary
            .completion_token_goodput_per_second
            .map_or_else(|| "-".to_owned(), compact_number);
        let gate_state = if stage
            .gates
            .iter()
            .any(|gate| gate.status == SaturationGateStatus::Fail)
        {
            "FAIL"
        } else if stage
            .gates
            .iter()
            .any(|gate| gate.status == SaturationGateStatus::NotEvaluable)
        {
            "N/E"
        } else {
            "PASS"
        };
        let _ = writeln!(
            output,
            "{:>5} {:>4}/{:<5} {:>7.1}% {:>13} {:>10} {:>10} {:>18}   {gate_state}",
            stage.concurrency,
            stage.summary.succeeded,
            stage.summary.attempted,
            stage.summary.error_percent,
            compact_number(stage.summary.successful_requests_per_second),
            p95_ttft,
            p95_e2e,
            token_goodput
        );
    }
    if report.stages.is_empty() {
        output.push_str("    - no measured stage ran\n");
    }
}

fn render_assessment(output: &mut String, report: &SaturationBenchmarkReport) {
    let assessment = &report.assessment;
    let status = match assessment.status {
        SaturationAssessmentStatus::SignalObserved => "SIGNAL OBSERVED",
        SaturationAssessmentStatus::NoSignalInTestedStages => "NO SIGNAL IN TESTED STAGES",
        SaturationAssessmentStatus::NotEvaluable => "NOT EVALUABLE",
    };
    let _ = writeln!(output, "\nASSESSMENT  {status}");
    if let (Some(signal), Some(concurrency)) =
        (assessment.signal, assessment.first_signal_concurrency)
    {
        let label = match signal {
            SaturationSignal::ThroughputPlateauWithErrorRate => {
                "throughput plateau with elevated error rate"
            }
            SaturationSignal::ThroughputPlateauWithLatencyInflation => {
                "throughput plateau with p95 latency inflation"
            }
        };
        let _ = writeln!(
            output,
            "  First observed at concurrency {concurrency}: {label}"
        );
    }
    if let Some(concurrency) = assessment.highest_accepted_tested_concurrency {
        let _ = writeln!(
            output,
            "  Highest tested point passing configured stage gates: {concurrency} (not a capacity recommendation)"
        );
    }
}

fn render_verification(output: &mut String, report: &SaturationBenchmarkReport) {
    let verification = &report.verification;
    let status = match verification.status {
        SaturationVerificationStatus::NotRequested => "NOT REQUESTED",
        SaturationVerificationStatus::Pass => "PASS",
        SaturationVerificationStatus::Fail => "FAIL",
        SaturationVerificationStatus::NotEvaluable => "NOT EVALUABLE",
    };
    if let Some(concurrency) = verification.requested_concurrency {
        let _ = writeln!(
            output,
            "VERIFICATION  {status} at exact concurrency {concurrency}"
        );
    } else {
        let _ = writeln!(output, "VERIFICATION  {status}");
    }
}

fn render_failures(output: &mut String, report: &SaturationBenchmarkReport) {
    let failed = report
        .stages
        .iter()
        .filter(|stage| stage.summary.failed > 0)
        .collect::<Vec<_>>();
    if failed.is_empty() {
        return;
    }
    output.push_str("\nFAILURE STAGES\n");
    for stage in failed {
        let counts = stage.summary.failure_stage_counts;
        let _ = writeln!(
            output,
            "  concurrency {}: transport={} http={} protocol={} empty_output={} expectation={}",
            stage.concurrency,
            counts.transport,
            counts.http,
            counts.protocol,
            counts.empty_output,
            counts.expectation
        );
    }
    output.push_str("  Complete privacy-safe per-attempt evidence: use --format json or ndjson.\n");
}

const fn abort_reason(reason: crate::domain::SaturationAbortReason) -> &'static str {
    match reason {
        crate::domain::SaturationAbortReason::WarmupNoSuccessfulRequests => {
            "warmup produced no successful request"
        }
        crate::domain::SaturationAbortReason::WarmupErrorRateLimitExceeded => {
            "warmup reached the configured error-rate abort limit"
        }
        crate::domain::SaturationAbortReason::StageNoSuccessfulRequests => {
            "a measured stage produced no successful request"
        }
        crate::domain::SaturationAbortReason::StageErrorRateLimitExceeded => {
            "a measured stage reached the configured error-rate abort limit"
        }
    }
}

fn duration(milliseconds: f64) -> String {
    if milliseconds >= 1_000.0 {
        format!("{:.2}s", milliseconds / 1_000.0)
    } else {
        format!("{milliseconds:.0}ms")
    }
}

fn elapsed_duration(milliseconds: u64) -> String {
    if milliseconds >= 1_000 {
        format!(
            "{:.2}s",
            std::time::Duration::from_millis(milliseconds).as_secs_f64()
        )
    } else {
        format!("{milliseconds}ms")
    }
}

fn bytes(value: usize) -> String {
    let value_f64 = f64::from(u32::try_from(value).unwrap_or(u32::MAX));
    if value >= 1 << 20 {
        format!("{:.1} MiB", value_f64 / f64::from(1 << 20))
    } else if value >= 1 << 10 {
        format!("{:.0} KiB", value_f64 / f64::from(1 << 10))
    } else {
        format!("{value} B")
    }
}

fn compact_number(value: f64) -> String {
    if !value.is_finite() {
        return "-".to_owned();
    }
    if value.abs() >= 1_000.0 {
        format!("{value:.0}")
    } else if value.abs() >= 100.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;

    use super::*;
    use crate::domain::{
        SATURATION_BENCHMARK_NONCLAIMS, SATURATION_BENCHMARK_VERSION, SaturationAssessment,
        SaturationLoadModel, SaturationPhaseResult, SaturationPlan, SaturationPolicy,
        SaturationRoute, SaturationSchedule, SaturationStageOrder, SaturationTarget,
        SaturationVerification, SaturationWarmupScope, SaturationWorkerStart,
    };

    #[test]
    fn text_output_is_explicit_about_scope_and_never_has_request_content_fields() {
        let report = SaturationBenchmarkReport {
            saturation_benchmark_version: SATURATION_BENCHMARK_VERSION,
            started_at: Utc::now(),
            duration_ns: 5_000_000,
            duration_ms: 5,
            status: SaturationRunStatus::Complete,
            abort_reason: None,
            workload_id: "builtin-v1".to_owned(),
            target: SaturationTarget {
                url: "http://127.0.0.1:8000".to_owned(),
                route: SaturationRoute::ChatCompletions,
                model: "served-model".to_owned(),
                stream: true,
            },
            plan: SaturationPlan {
                concurrency_stages: vec![1],
                warmup_requests_per_worker: 1,
                requests_per_worker: 1,
                planned_attempts: 2,
                max_tokens: 1,
                timeout_ns: 1_000_000_000,
                timeout_ms: 1_000,
                response_limit_bytes: 1_024,
                schedule: SaturationSchedule {
                    load_model: SaturationLoadModel::ClosedLoopFixedConcurrency,
                    stage_order: SaturationStageOrder::ExplicitAscending,
                    warmup_scope: SaturationWarmupScope::EachStageExcluded,
                    worker_start: SaturationWorkerStart::SimultaneousBarrier,
                },
            },
            policy: SaturationPolicy::default(),
            warmups: vec![SaturationPhaseResult {
                concurrency: 1,
                planned_requests: 1,
                attempted: 1,
                succeeded: 1,
                duration_ns: 1_000_000,
                duration_ms: 1,
                ..SaturationPhaseResult::default()
            }],
            stages: Vec::new(),
            assessment: SaturationAssessment::default(),
            verification: SaturationVerification::default(),
            nonclaims: SATURATION_BENCHMARK_NONCLAIMS.to_vec(),
        };

        let text = render(&report);
        assert!(text.contains("Closed-loop, fixed-concurrency"));
        assert!(text.contains("not production capacity"));
        assert!(text.contains("VERIFICATION  NOT REQUESTED"));
        for forbidden in ["prompt=", "expect=", "credential=", "generated_content="] {
            assert!(!text.contains(forbidden));
        }
    }
}
