//! Human-readable presentation for active inference canary reports.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::domain::{
    CanaryDistribution, CanaryFailureStage, CanaryGate, CanaryReport, CanaryStatus,
};
use crate::presentation::safe_inline;

const MAX_FAILURE_GROUPS: usize = 20;
const MAX_FAILURE_EXAMPLES: usize = 5;

/// Render a compact, copyable canary summary without prompt or response data.
pub fn render(report: &CanaryReport) -> String {
    let mut output = String::new();
    let status = match report.status {
        CanaryStatus::Pass => "PASS",
        CanaryStatus::Fail => "FAIL",
    };
    let _ = writeln!(output, "GPU Watchman canary  {status}");
    let _ = writeln!(
        output,
        "Target   {} | model={} | workload={} | stream={}",
        safe_inline(&report.target.url),
        safe_inline(&report.target.model),
        safe_inline(&report.workload_id),
        if report.target.stream { "yes" } else { "no" }
    );
    let _ = writeln!(
        output,
        "Plan     {} request(s) | concurrency {} | max tokens {} | timeout {} | response cap {}",
        report.plan.count,
        report.plan.concurrency,
        report.plan.max_tokens,
        elapsed_duration(report.plan.timeout_ms),
        bytes(report.plan.response_limit_bytes)
    );
    let _ = writeln!(
        output,
        "Results  {}/{} passed | {:.1}% | {} req/s | {} elapsed",
        report.summary.succeeded,
        report.summary.attempted,
        report.summary.success_percent,
        compact_number(report.summary.achieved_requests_per_second),
        elapsed_duration(report.duration_ms)
    );

    output.push_str("\nMETRIC          SAMPLES       P50       P95       P99       MAX\n");
    render_distribution(
        &mut output,
        "Headers",
        report.summary.headers_ms.as_ref(),
        true,
    );
    render_distribution(&mut output, "TTFT", report.summary.ttft_ms.as_ref(), true);
    render_distribution(&mut output, "E2E", report.summary.e2e_ms.as_ref(), true);
    render_distribution(
        &mut output,
        "Output tok/s",
        report.summary.output_tokens_per_second.as_ref(),
        false,
    );

    output.push_str("\nGATES\n");
    for gate in &report.gates {
        render_gate(&mut output, gate);
    }

    render_failures(&mut output, report);
    output
}

fn render_distribution(
    output: &mut String,
    name: &str,
    distribution: Option<&CanaryDistribution>,
    milliseconds: bool,
) {
    let Some(distribution) = distribution else {
        let _ = writeln!(
            output,
            "{name:<15} {:>7} {:>9} {:>9} {:>9} {:>9}",
            0, "-", "-", "-", "-"
        );
        return;
    };
    let format_value = |value| {
        if milliseconds {
            duration(value)
        } else {
            compact_number(value)
        }
    };
    let _ = writeln!(
        output,
        "{name:<15} {:>7} {:>9} {:>9} {:>9} {:>9}",
        distribution.samples,
        format_value(distribution.p50),
        format_value(distribution.p95),
        format_value(distribution.p99),
        format_value(distribution.max)
    );
}

fn render_failures(output: &mut String, report: &CanaryReport) {
    let mut groups = BTreeMap::<(&str, &str), (usize, Vec<u32>)>::new();
    let mut total = 0_usize;
    for attempt in &report.attempts {
        let Some(failure) = attempt.failure.as_ref() else {
            continue;
        };
        total = total.saturating_add(1);
        let group = groups
            .entry((failure_stage(failure.stage), failure.message.as_str()))
            .or_default();
        group.0 = group.0.saturating_add(1);
        if group.1.len() < MAX_FAILURE_EXAMPLES {
            group.1.push(attempt.index.saturating_add(1));
        }
    }
    if groups.is_empty() {
        return;
    }

    let mut groups = groups.into_iter().collect::<Vec<_>>();
    groups.sort_by(|left, right| right.1.0.cmp(&left.1.0).then_with(|| left.0.cmp(&right.0)));
    let _ = writeln!(
        output,
        "\nFAILURES  {total} attempt(s) | {} unique group(s)",
        groups.len()
    );
    for ((stage, message), (count, examples)) in groups.iter().take(MAX_FAILURE_GROUPS) {
        let shown_examples = examples.len();
        let examples = examples
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", #");
        let omitted_examples = count.saturating_sub(shown_examples);
        let suffix = if omitted_examples == 0 {
            String::new()
        } else {
            format!(", +{omitted_examples} more")
        };
        let _ = writeln!(
            output,
            "  {stage} x{count} (#{examples}{suffix}): {}",
            safe_inline(message)
        );
    }
    let omitted_groups = groups.len().saturating_sub(MAX_FAILURE_GROUPS);
    if omitted_groups > 0 {
        let _ = writeln!(output, "  ... {omitted_groups} additional group(s) omitted");
    }
    output.push_str(
        "  Complete per-attempt evidence: rerun with --format json or --format ndjson.\n",
    );
}

fn render_gate(output: &mut String, gate: &CanaryGate) {
    let state = if gate.passed { "PASS" } else { "FAIL" };
    let observed = gate
        .observed
        .map_or_else(|| "unavailable".to_owned(), |value| gate_value(gate, value));
    let detail = if gate.detail.is_empty() {
        String::new()
    } else {
        format!(" | {}", safe_inline(&gate.detail))
    };
    let _ = writeln!(
        output,
        "  {state:<4} {:<36} {observed} {} {}{detail}",
        safe_inline(&gate.name),
        safe_inline(&gate.operator),
        gate_value(gate, gate.threshold)
    );
}

fn gate_value(gate: &CanaryGate, value: f64) -> String {
    if gate.name.ends_with("_ms") {
        duration(value)
    } else if gate.name.ends_with("_percent") {
        format!("{}%", compact_number(value))
    } else if gate.name.ends_with("_per_second") {
        format!("{}/s", compact_number(value))
    } else {
        compact_number(value)
    }
}

fn failure_stage(stage: CanaryFailureStage) -> &'static str {
    match stage {
        CanaryFailureStage::Transport => "transport",
        CanaryFailureStage::Http => "http",
        CanaryFailureStage::Protocol => "protocol",
        CanaryFailureStage::EmptyOutput => "empty_output",
        CanaryFailureStage::Expectation => "expectation",
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

fn compact_number(value: f64) -> String {
    if value.abs() >= 100.0 {
        format!("{value:.0}")
    } else if value.abs() >= 10.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
    }
}

fn bytes(value: usize) -> String {
    const MIB: usize = 1024 * 1024;
    const KIB: usize = 1024;
    if value.is_multiple_of(MIB) {
        format!("{} MiB", value / MIB)
    } else if value.is_multiple_of(KIB) {
        format!("{} KiB", value / KIB)
    } else {
        format!("{value} bytes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{CANARY_VERSION, CanaryPlan, CanarySummary, CanaryTarget};
    use chrono::Utc;

    #[test]
    fn renderer_shows_metrics_and_never_needs_request_content() {
        let report = CanaryReport {
            canary_version: CANARY_VERSION,
            started_at: Utc::now(),
            duration_ms: 500,
            status: CanaryStatus::Pass,
            workload_id: "builtin-v1".to_owned(),
            policy: None,
            target: CanaryTarget {
                url: "http://localhost/v1/chat/completions".to_owned(),
                route: "chat_completions".to_owned(),
                model: "model".to_owned(),
                stream: true,
            },
            plan: CanaryPlan {
                count: 1,
                concurrency: 1,
                max_tokens: 16,
                timeout_ms: 30_000,
                response_limit_bytes: 8 * 1024 * 1024,
            },
            summary: CanarySummary {
                attempted: 1,
                succeeded: 1,
                success_percent: 100.0,
                ttft_ms: Some(CanaryDistribution {
                    samples: 1,
                    min: 20.0,
                    mean: 20.0,
                    p50: 20.0,
                    p95: 20.0,
                    p99: 20.0,
                    max: 20.0,
                }),
                ..CanarySummary::default()
            },
            gates: Vec::new(),
            attempts: Vec::new(),
        };

        let output = render(&report);
        assert!(output.contains("GPU Watchman canary  PASS"));
        assert!(output.contains("TTFT"));
        assert!(output.contains("20ms"));
    }

    #[test]
    fn renderer_keeps_missing_metric_rows_aligned() {
        let report = CanaryReport {
            canary_version: CANARY_VERSION,
            started_at: Utc::now(),
            duration_ms: 1,
            status: CanaryStatus::Fail,
            workload_id: "builtin-v1".to_owned(),
            policy: None,
            target: CanaryTarget::default(),
            plan: CanaryPlan::default(),
            summary: CanarySummary::default(),
            gates: Vec::new(),
            attempts: Vec::new(),
        };

        let output = render(&report);
        let headers = output
            .lines()
            .find(|line| line.starts_with("Headers"))
            .unwrap();
        assert_eq!(
            headers.split_whitespace().collect::<Vec<_>>(),
            ["Headers", "0", "-", "-", "-", "-"]
        );
    }

    #[test]
    fn renderer_groups_and_caps_failures_with_machine_output_pointer() {
        let mut attempts = (0..8)
            .map(|index| {
                crate::domain::CanaryAttempt::failed(
                    index,
                    CanaryFailureStage::Transport,
                    "connection failed",
                )
            })
            .collect::<Vec<_>>();
        attempts.extend((8..40).map(|index| {
            crate::domain::CanaryAttempt::failed(
                index,
                CanaryFailureStage::Protocol,
                format!("protocol failure {index}"),
            )
        }));
        let report = CanaryReport {
            canary_version: CANARY_VERSION,
            started_at: Utc::now(),
            duration_ms: 1,
            status: CanaryStatus::Fail,
            workload_id: "builtin-v1".to_owned(),
            policy: None,
            target: CanaryTarget::default(),
            plan: CanaryPlan::default(),
            summary: CanarySummary {
                attempted: 40,
                failed: 40,
                ..CanarySummary::default()
            },
            gates: Vec::new(),
            attempts,
        };

        let output = render(&report);
        assert!(output.contains("FAILURES  40 attempt(s) | 33 unique group(s)"));
        assert!(output.contains("transport x8 (#1, #2, #3, #4, #5, +3 more): connection failed"));
        assert!(output.contains("13 additional group(s) omitted"));
        assert!(output.contains("rerun with --format json or --format ndjson"));
        assert!(!output.contains("protocol failure 39"));
    }
}
