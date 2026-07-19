//! Offline before/after comparison and regression gating for Watchman reports.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::domain::{
    Endpoint, Finding, Gpu, Report, SCHEMA_VERSION, Severity, SourceState, SourceStatus,
};
use crate::presentation::safe_inline;
use crate::security::open_read_nonblocking;

const MAX_REPORT_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ReportComparison {
    pub comparison_version: u32,
    pub baseline_at: DateTime<Utc>,
    pub current_at: DateTime<Utc>,
    pub elapsed_seconds: i64,
    pub baseline_status: String,
    pub current_status: String,
    pub summary: SummaryDelta,
    pub gpus: Vec<GpuDelta>,
    pub endpoints: Vec<EndpointDelta>,
    pub sources: Vec<SourceDelta>,
    pub new_findings: Vec<Finding>,
    pub resolved_findings: Vec<Finding>,
    pub regression: bool,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct SummaryDelta {
    pub gpus: i64,
    pub active_gpus: i64,
    pub processes: i64,
    pub endpoints_up: i64,
    pub vram_used_mib: i64,
    pub critical_findings: i64,
    pub warning_findings: i64,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChangeKind {
    Added,
    Removed,
    Changed,
    Unchanged,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GpuDelta {
    pub id: String,
    pub name: String,
    pub index: i32,
    pub change: ChangeKind,
    pub memory_used_mib: i64,
    pub gpu_util_percent: i32,
    pub temperature_c: i32,
    pub power_draw_w: f64,
    pub processes: i64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct EndpointDelta {
    pub url: String,
    pub runtime: String,
    pub change: ChangeKind,
    pub reachable_before: bool,
    pub reachable_after: bool,
    pub latency_ms: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requests_running: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requests_waiting: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_cache_usage_percent: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requests_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_errors_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preemptions_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_time_to_first_token_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SourceDelta {
    pub name: String,
    pub change: ChangeKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_before: Option<SourceState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_after: Option<SourceState>,
    pub required_before: bool,
    pub required_after: bool,
    pub duration_ms: i64,
    pub records: i64,
}

/// Load either a pretty JSON report or the final report in an NDJSON history file.
pub fn load_report(path: &Path) -> Result<Report> {
    let mut file = open_read_nonblocking(path, false)
        .with_context(|| format!("open report {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("inspect report {}", path.display()))?;
    if !metadata.is_file() {
        bail!("report {} is not a regular file", path.display());
    }
    if metadata.len() > MAX_REPORT_BYTES {
        bail!(
            "report {} is larger than the {} MiB safety limit",
            path.display(),
            MAX_REPORT_BYTES / (1024 * 1024)
        );
    }
    let mut bytes = Vec::with_capacity(
        usize::try_from(metadata.len().min(MAX_REPORT_BYTES)).unwrap_or_default(),
    );
    Read::by_ref(&mut file)
        .take(MAX_REPORT_BYTES + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("read report {}", path.display()))?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_REPORT_BYTES {
        bail!(
            "report {} is larger than the {} MiB safety limit",
            path.display(),
            MAX_REPORT_BYTES / (1024 * 1024)
        );
    }
    let body = String::from_utf8(bytes)
        .with_context(|| format!("report {} is not UTF-8", path.display()))?;
    let report = serde_json::from_str(&body).or_else(|full_error| {
        let last = body
            .lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .ok_or(full_error)?;
        serde_json::from_str(last)
    })?;
    validate_schema(&report, path)?;
    Ok(report)
}

fn validate_schema(report: &Report, path: &Path) -> Result<()> {
    if report.schema_version == 0 || report.schema_version > SCHEMA_VERSION {
        bail!(
            "report {} uses unsupported schema version {} (this build supports 1..={})",
            path.display(),
            report.schema_version,
            SCHEMA_VERSION
        );
    }
    Ok(())
}

pub fn compare(baseline: &Report, current: &Report) -> ReportComparison {
    let new_findings = finding_difference(&current.findings, &baseline.findings);
    let resolved_findings = finding_difference(&baseline.findings, &current.findings);
    let gpus = compare_gpus(&baseline.gpus, &current.gpus);
    let endpoints = compare_endpoints(&baseline.endpoints, &current.endpoints);
    let sources = compare_sources(&baseline.sources, &current.sources);
    let regression = new_findings
        .iter()
        .any(|finding| finding.severity <= Severity::Warning)
        || gpus.iter().any(|gpu| gpu.change == ChangeKind::Removed)
        || endpoints.iter().any(|endpoint| {
            (endpoint.change == ChangeKind::Added || endpoint.reachable_before)
                && !endpoint.reachable_after
        })
        || sources.iter().any(source_is_regression);

    ReportComparison {
        comparison_version: 2,
        baseline_at: baseline.collected_at,
        current_at: current.collected_at,
        elapsed_seconds: current
            .collected_at
            .signed_duration_since(baseline.collected_at)
            .num_seconds(),
        baseline_status: baseline.status.clone(),
        current_status: current.status.clone(),
        summary: SummaryDelta {
            gpus: usize_delta(current.summary.gpus, baseline.summary.gpus),
            active_gpus: usize_delta(current.summary.active_gpus, baseline.summary.active_gpus),
            processes: usize_delta(current.summary.processes, baseline.summary.processes),
            endpoints_up: usize_delta(current.summary.endpoints_up, baseline.summary.endpoints_up),
            vram_used_mib: current
                .summary
                .vram_used_mib
                .saturating_sub(baseline.summary.vram_used_mib),
            critical_findings: usize_delta(
                current.summary.critical_findings,
                baseline.summary.critical_findings,
            ),
            warning_findings: usize_delta(
                current.summary.warning_findings,
                baseline.summary.warning_findings,
            ),
        },
        gpus,
        endpoints,
        sources,
        new_findings,
        resolved_findings,
        regression,
    }
}

fn finding_difference(source: &[Finding], other: &[Finding]) -> Vec<Finding> {
    let other_keys = other.iter().map(finding_key).collect::<BTreeSet<_>>();
    source
        .iter()
        .filter(|finding| !other_keys.contains(&finding_key(finding)))
        .cloned()
        .collect()
}

fn finding_key(finding: &Finding) -> (Option<i32>, Severity, &str) {
    (finding.gpu_index, finding.severity, finding.code.as_str())
}

fn compare_gpus(baseline: &[Gpu], current: &[Gpu]) -> Vec<GpuDelta> {
    let before = baseline
        .iter()
        .map(|gpu| (gpu_key(gpu), gpu))
        .collect::<BTreeMap<_, _>>();
    let after = current
        .iter()
        .map(|gpu| (gpu_key(gpu), gpu))
        .collect::<BTreeMap<_, _>>();
    before
        .keys()
        .chain(after.keys())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|key| gpu_delta(key, before.get(key).copied(), after.get(key).copied()))
        .collect()
}

fn gpu_delta(key: &str, before: Option<&Gpu>, after: Option<&Gpu>) -> GpuDelta {
    let reference = after.or(before).expect("GPU key must exist in one report");
    let mut delta = GpuDelta {
        id: key.to_owned(),
        name: reference.name.clone(),
        index: reference.index,
        change: match (before, after) {
            (None, Some(_)) => ChangeKind::Added,
            (Some(_), None) => ChangeKind::Removed,
            _ => ChangeKind::Changed,
        },
        memory_used_mib: after.map_or(0, |gpu| gpu.memory_used_mib)
            - before.map_or(0, |gpu| gpu.memory_used_mib),
        gpu_util_percent: after.map_or(0, |gpu| gpu.gpu_util_percent)
            - before.map_or(0, |gpu| gpu.gpu_util_percent),
        temperature_c: after.map_or(0, |gpu| gpu.temperature_c)
            - before.map_or(0, |gpu| gpu.temperature_c),
        power_draw_w: after.map_or(0.0, |gpu| gpu.power_draw_w)
            - before.map_or(0.0, |gpu| gpu.power_draw_w),
        processes: usize_delta(
            after.map_or(0, |gpu| gpu.processes.len()),
            before.map_or(0, |gpu| gpu.processes.len()),
        ),
    };
    if before.is_some()
        && after.is_some()
        && delta.memory_used_mib == 0
        && delta.gpu_util_percent == 0
        && delta.temperature_c == 0
        && delta.power_draw_w.abs() < f64::EPSILON
        && delta.processes == 0
    {
        delta.change = ChangeKind::Unchanged;
    }
    delta
}

fn gpu_key(gpu: &Gpu) -> String {
    if gpu.uuid.is_empty() {
        format!("index:{}", gpu.index)
    } else {
        gpu.uuid.clone()
    }
}

fn compare_endpoints(baseline: &[Endpoint], current: &[Endpoint]) -> Vec<EndpointDelta> {
    let before = baseline
        .iter()
        .map(|endpoint| (endpoint.url.as_str(), endpoint))
        .collect::<BTreeMap<_, _>>();
    let after = current
        .iter()
        .map(|endpoint| (endpoint.url.as_str(), endpoint))
        .collect::<BTreeMap<_, _>>();
    before
        .keys()
        .chain(after.keys())
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|url| endpoint_delta(url, before.get(url).copied(), after.get(url).copied()))
        .collect()
}

fn endpoint_delta(url: &str, before: Option<&Endpoint>, after: Option<&Endpoint>) -> EndpointDelta {
    let reference = after
        .or(before)
        .expect("endpoint key must exist in one report");
    let mut delta = EndpointDelta {
        url: url.to_owned(),
        runtime: reference.runtime.clone(),
        change: match (before, after) {
            (None, Some(_)) => ChangeKind::Added,
            (Some(_), None) => ChangeKind::Removed,
            _ => ChangeKind::Changed,
        },
        reachable_before: before.is_some_and(|endpoint| endpoint.reachable),
        reachable_after: after.is_some_and(|endpoint| endpoint.reachable),
        latency_ms: i64_delta(
            after.map_or(0, |endpoint| endpoint.latency_ms),
            before.map_or(0, |endpoint| endpoint.latency_ms),
        ),
        requests_running: option_delta(
            before.and_then(|endpoint| endpoint.requests_running),
            after.and_then(|endpoint| endpoint.requests_running),
        ),
        requests_waiting: option_delta(
            before.and_then(|endpoint| endpoint.requests_waiting),
            after.and_then(|endpoint| endpoint.requests_waiting),
        ),
        kv_cache_usage_percent: option_delta(
            before.and_then(|endpoint| endpoint.kv_cache_usage_percent),
            after.and_then(|endpoint| endpoint.kv_cache_usage_percent),
        ),
        requests_per_second: option_delta(
            before.and_then(|endpoint| endpoint.rates.requests_per_second),
            after.and_then(|endpoint| endpoint.rates.requests_per_second),
        ),
        generation_tokens_per_second: option_delta(
            before.and_then(|endpoint| endpoint.rates.generation_tokens_per_second),
            after.and_then(|endpoint| endpoint.rates.generation_tokens_per_second),
        ),
        request_errors_per_second: option_delta(
            before.and_then(|endpoint| endpoint.rates.request_errors_per_second),
            after.and_then(|endpoint| endpoint.rates.request_errors_per_second),
        ),
        preemptions_per_second: option_delta(
            before.and_then(|endpoint| endpoint.rates.preemptions_per_second),
            after.and_then(|endpoint| endpoint.rates.preemptions_per_second),
        ),
        mean_time_to_first_token_ms: option_delta(
            before.and_then(|endpoint| endpoint.rates.mean_time_to_first_token_ms),
            after.and_then(|endpoint| endpoint.rates.mean_time_to_first_token_ms),
        ),
    };
    if before.is_some()
        && after.is_some()
        && delta.reachable_before == delta.reachable_after
        && delta.latency_ms == 0
        && optional_is_zero(delta.requests_running)
        && optional_is_zero(delta.requests_waiting)
        && optional_is_zero(delta.kv_cache_usage_percent)
        && optional_is_zero(delta.requests_per_second)
        && optional_is_zero(delta.generation_tokens_per_second)
        && optional_is_zero(delta.request_errors_per_second)
        && optional_is_zero(delta.preemptions_per_second)
        && optional_is_zero(delta.mean_time_to_first_token_ms)
    {
        delta.change = ChangeKind::Unchanged;
    }
    delta
}

fn compare_sources(baseline: &[SourceStatus], current: &[SourceStatus]) -> Vec<SourceDelta> {
    let before = baseline
        .iter()
        .map(|source| (source.name.as_str(), source))
        .collect::<BTreeMap<_, _>>();
    let after = current
        .iter()
        .map(|source| (source.name.as_str(), source))
        .collect::<BTreeMap<_, _>>();
    before
        .keys()
        .chain(after.keys())
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|name| source_delta(name, before.get(name).copied(), after.get(name).copied()))
        .collect()
}

fn source_delta(
    name: &str,
    before: Option<&SourceStatus>,
    after: Option<&SourceStatus>,
) -> SourceDelta {
    let mut delta = SourceDelta {
        name: name.to_owned(),
        change: match (before, after) {
            (None, Some(_)) => ChangeKind::Added,
            (Some(_), None) => ChangeKind::Removed,
            _ => ChangeKind::Changed,
        },
        state_before: before.map(|source| source.state),
        state_after: after.map(|source| source.state),
        required_before: before.is_some_and(|source| source.required),
        required_after: after.is_some_and(|source| source.required),
        duration_ms: i64_delta(
            after.map_or(0, |source| source.duration_ms),
            before.map_or(0, |source| source.duration_ms),
        ),
        records: i64_delta(
            after.map_or(0, |source| source.records),
            before.map_or(0, |source| source.records),
        ),
    };
    if before.is_some()
        && after.is_some()
        && delta.state_before == delta.state_after
        && delta.required_before == delta.required_after
        && delta.duration_ms == 0
        && delta.records == 0
    {
        delta.change = ChangeKind::Unchanged;
    }
    delta
}

fn source_is_regression(source: &SourceDelta) -> bool {
    if source.change == ChangeKind::Removed {
        return true;
    }
    let degraded = source
        .state_after
        .is_some_and(|state| state != SourceState::Ok);
    (source.required_after || source.state_before == Some(SourceState::Ok)) && degraded
}

fn optional_is_zero(value: Option<f64>) -> bool {
    value.is_none_or(|value| value.abs() < f64::EPSILON)
}

fn option_delta(before: Option<f64>, after: Option<f64>) -> Option<f64> {
    match (before, after) {
        (Some(before), Some(after)) => Some(after - before),
        (None, Some(after)) => Some(after),
        (Some(before), None) => Some(-before),
        (None, None) => None,
    }
}

fn usize_delta(after: usize, before: usize) -> i64 {
    i64_delta(
        after.try_into().unwrap_or(u64::MAX),
        before.try_into().unwrap_or(u64::MAX),
    )
}

fn i64_delta(after: u64, before: u64) -> i64 {
    let delta = i128::from(after)
        .saturating_sub(i128::from(before))
        .clamp(i128::from(i64::MIN), i128::from(i64::MAX));
    i64::try_from(delta).unwrap_or(if delta.is_negative() {
        i64::MIN
    } else {
        i64::MAX
    })
}

pub fn render_text(comparison: &ReportComparison) -> String {
    let mut output = String::new();
    let verdict = if comparison.regression {
        "REGRESSION"
    } else {
        "NO REGRESSION"
    };
    let _ = writeln!(output, "GPU Watchman comparison  {verdict}");
    let _ = writeln!(
        output,
        "Window      {} -> {} ({}s)",
        comparison.baseline_at, comparison.current_at, comparison.elapsed_seconds
    );
    let _ = writeln!(
        output,
        "Status      {} -> {}",
        safe_inline(&comparison.baseline_status),
        safe_inline(&comparison.current_status)
    );
    let _ = writeln!(
        output,
        "Summary     GPUs {:+}, active {:+}, processes {:+}, endpoints up {:+}",
        comparison.summary.gpus,
        comparison.summary.active_gpus,
        comparison.summary.processes,
        comparison.summary.endpoints_up
    );
    let _ = writeln!(
        output,
        "Pressure    VRAM {:+} MiB, critical {:+}, warnings {:+}",
        comparison.summary.vram_used_mib,
        comparison.summary.critical_findings,
        comparison.summary.warning_findings
    );
    if !comparison.gpus.is_empty() {
        output.push_str("\nGPU changes\n");
        for gpu in &comparison.gpus {
            let _ = writeln!(
                output,
                "  GPU {} {:<7?} VRAM {:+} MiB | util {:+}% | temp {:+}C | power {:+.1}W | procs {:+}",
                gpu.index,
                gpu.change,
                gpu.memory_used_mib,
                gpu.gpu_util_percent,
                gpu.temperature_c,
                gpu.power_draw_w,
                gpu.processes
            );
        }
    }
    render_source_changes(&mut output, &comparison.sources);
    if !comparison.endpoints.is_empty() {
        output.push_str("\nEndpoint changes\n");
        for endpoint in &comparison.endpoints {
            let _ = writeln!(
                output,
                "  {} [{}] up {} -> {} | latency {:+}ms | running {} | waiting {} | KV {}",
                safe_inline(&endpoint.url),
                safe_inline(&endpoint.runtime),
                endpoint.reachable_before,
                endpoint.reachable_after,
                endpoint.latency_ms,
                display_optional(endpoint.requests_running),
                display_optional(endpoint.requests_waiting),
                display_optional(endpoint.kv_cache_usage_percent)
            );
            if endpoint.requests_per_second.is_some()
                || endpoint.generation_tokens_per_second.is_some()
                || endpoint.request_errors_per_second.is_some()
                || endpoint.preemptions_per_second.is_some()
                || endpoint.mean_time_to_first_token_ms.is_some()
            {
                let _ = writeln!(
                    output,
                    "    interval deltas: req/s {} | generation tok/s {} | errors/s {} | preempt/s {} | TTFT {}ms",
                    display_optional(endpoint.requests_per_second),
                    display_optional(endpoint.generation_tokens_per_second),
                    display_optional(endpoint.request_errors_per_second),
                    display_optional(endpoint.preemptions_per_second),
                    display_optional(endpoint.mean_time_to_first_token_ms)
                );
            }
        }
    }
    render_findings(&mut output, "New findings", &comparison.new_findings);
    render_findings(
        &mut output,
        "Resolved findings",
        &comparison.resolved_findings,
    );
    output
}

fn render_source_changes(output: &mut String, sources: &[SourceDelta]) {
    if sources.is_empty() {
        return;
    }
    output.push_str("\nTelemetry source changes\n");
    for source in sources {
        let before = source
            .state_before
            .map_or_else(|| "absent".to_owned(), |state| state.to_string());
        let after = source
            .state_after
            .map_or_else(|| "absent".to_owned(), |state| state.to_string());
        let _ = writeln!(
            output,
            "  {} {:?} {before} -> {after} | required {} -> {} | duration {:+}ms | records {:+}",
            safe_inline(&source.name),
            source.change,
            source.required_before,
            source.required_after,
            source.duration_ms,
            source.records,
        );
    }
}

fn display_optional(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_owned(), |value| format!("{value:+.1}"))
}

fn render_findings(output: &mut String, title: &str, findings: &[Finding]) {
    if findings.is_empty() {
        return;
    }
    let _ = writeln!(output, "\n{title}");
    for finding in findings {
        let target = finding
            .gpu_index
            .map_or_else(|| "host/runtime".to_owned(), |index| format!("GPU {index}"));
        let _ = writeln!(
            output,
            "  [{}] {} {}: {}",
            finding.severity,
            target,
            safe_inline(&finding.code),
            safe_inline(&finding.message)
        );
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::domain::{SCHEMA_VERSION, Summary};

    fn report() -> Report {
        Report {
            schema_version: SCHEMA_VERSION,
            status: "healthy".to_owned(),
            summary: Summary {
                gpus: 1,
                active_gpus: 1,
                endpoints: 1,
                endpoints_up: 1,
                ..Summary::default()
            },
            gpus: vec![Gpu {
                index: 0,
                uuid: "GPU-a".to_owned(),
                name: "Example".to_owned(),
                memory_used_mib: 1_000,
                processes: vec![crate::domain::GpuProcess::default()],
                ..Gpu::default()
            }],
            endpoints: vec![Endpoint {
                url: "http://runtime:8000".to_owned(),
                runtime: "vllm".to_owned(),
                reachable: true,
                latency_ms: 10,
                requests_waiting: Some(0.0),
                ..Endpoint::default()
            }],
            sources: vec![SourceStatus::ok("nvidia.processes", 10, 1)],
            ..Report::default()
        }
    }

    #[test]
    fn detects_new_findings_and_endpoint_outages_as_regressions() {
        let baseline = report();
        let mut current = report();
        current.status = "critical".to_owned();
        current.summary.endpoints_up = 0;
        current.summary.critical_findings = 1;
        current.gpus[0].memory_used_mib = 2_500;
        current.endpoints[0].reachable = false;
        current.findings.push(Finding::new(
            Some(0),
            Severity::Critical,
            "gpu-test",
            "regressed",
        ));

        let comparison = compare(&baseline, &current);

        assert!(comparison.regression);
        assert_eq!(comparison.summary.vram_used_mib, 0);
        assert_eq!(comparison.gpus[0].memory_used_mib, 1_500);
        assert!(!comparison.endpoints[0].reachable_after);
        assert_eq!(comparison.new_findings.len(), 1);
    }

    #[test]
    fn telemetry_source_degradation_is_a_regression() {
        let baseline = report();
        let mut current = report();
        current.sources[0] = SourceStatus::failed(
            "nvidia.processes",
            SourceState::Partial,
            20,
            0,
            "compute query failed",
        );

        let comparison = compare(&baseline, &current);

        assert!(comparison.regression);
        assert_eq!(comparison.comparison_version, 2);
        assert_eq!(comparison.sources[0].state_before, Some(SourceState::Ok));
        assert_eq!(
            comparison.sources[0].state_after,
            Some(SourceState::Partial)
        );
    }

    #[test]
    fn loads_the_last_report_from_ndjson() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("history.ndjson");
        let first = report();
        let mut second = report();
        second.status = "warning".to_owned();
        fs::write(
            &path,
            format!(
                "{}\n{}\n",
                serde_json::to_string(&first).unwrap(),
                serde_json::to_string(&second).unwrap()
            ),
        )
        .unwrap();

        assert_eq!(load_report(&path).unwrap().status, "warning");
    }
}
