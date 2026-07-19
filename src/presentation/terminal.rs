//! Human-readable and machine-readable report rendering.

use std::fmt::Write as _;

use crate::domain::{
    Endpoint, Finding, Gpu, GpuProcess, Report, RuntimeHistogramInterval, Severity, SourceState,
    SourceStatus,
};
use crate::presentation::{safe_inline, safe_multiline};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Json,
    Ndjson,
}

pub fn render(report: &Report, format: OutputFormat, details: bool, show_all: bool) -> String {
    match format {
        OutputFormat::Json => {
            let mut output = serde_json::to_string_pretty(report).unwrap_or_else(|error| {
                format!(r#"{{"error":"could not encode report: {error}"}}"#)
            });
            output.push('\n');
            output
        }
        OutputFormat::Ndjson => {
            let mut output = serde_json::to_string(report).unwrap_or_else(|error| {
                format!(r#"{{"error":"could not encode report: {error}"}}"#)
            });
            output.push('\n');
            output
        }
        OutputFormat::Text => render_text(report, details, show_all),
    }
}

/// Add ANSI styling to an already aligned human-readable report.
///
/// Styling after layout preserves table widths. Callers never apply it to
/// JSON or NDJSON output.
pub fn colorize(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    for line in input.split_inclusive('\n') {
        let content = line.strip_suffix('\n').unwrap_or(line);
        if matches!(
            content,
            "GPUS"
                | "SOURCES"
                | "ENDPOINTS"
                | "FINDINGS"
                | "GPU DETAILS"
                | "TOPOLOGY"
                | "XID EVENTS"
        ) {
            let _ = write!(output, "\x1b[1;36m{content}\x1b[0m");
        } else {
            for chunk in content.split_inclusive(char::is_whitespace) {
                output.push_str(&colorize_chunk(chunk));
            }
        }
        if line.ends_with('\n') {
            output.push('\n');
        }
    }
    output
}

fn colorize_chunk(chunk: &str) -> String {
    let Some(start) = chunk.find(char::is_alphanumeric) else {
        return chunk.to_owned();
    };
    let end = chunk
        .char_indices()
        .rev()
        .find(|(_, character)| character.is_alphanumeric())
        .map_or(start, |(index, character)| index + character.len_utf8());
    let word = &chunk[start..end];
    let code = match word {
        "HEALTHY" | "ACTIVE" | "UP" | "OK" | "PASS" | "FITS" => "1;32",
        "WARNING" | "PARTIAL" | "WARN" => "1;33",
        "CRITICAL" | "UNAVAILABLE" | "DOWN" | "FAIL" => "1;31",
        "INFO" => "1;34",
        "IDLE" | "SKIPPED" => "2",
        _ => return chunk.to_owned(),
    };
    format!(
        "{}\x1b[{code}m{}\x1b[0m{}",
        &chunk[..start],
        word,
        &chunk[end..]
    )
}

pub fn render_text(report: &Report, details: bool, show_all: bool) -> String {
    let mut output = String::new();
    let summary = &report.summary;
    let _ = writeln!(
        output,
        "GPU Watchman  {}  {} ({}/{})  collected={}  duration={}ms",
        report.status.to_ascii_uppercase(),
        value_or_dash(&report.host.hostname),
        value_or_dash(&report.host.os),
        value_or_dash(&report.host.arch),
        report.collected_at.format("%Y-%m-%d %H:%M:%S UTC"),
        report.collection_duration_ms,
    );
    let _ = writeln!(
        output,
        "GPUs {} total / {} active  |  VRAM {} ({:.0}%)  |  processes {}  |  endpoints {}/{} up  |  findings {} critical / {} warning",
        summary.gpus,
        summary.active_gpus,
        memory_pair(summary.vram_used_mib, summary.vram_total_mib),
        percent(summary.vram_used_mib, summary.vram_total_mib),
        summary.processes,
        summary.endpoints_up,
        summary.endpoints,
        summary.critical_findings,
        summary.warning_findings,
    );

    let mut gpus = report
        .gpus
        .iter()
        .filter(|gpu| show_all || gpu_visible(gpu, &report.findings))
        .collect::<Vec<_>>();
    gpus.sort_by_key(|gpu| gpu.index);
    if gpus.is_empty() {
        output.push_str(
            "\nNo GPU rows shown. Healthy idle GPUs may be hidden; use --all to include them.\n",
        );
    } else {
        output.push_str("\nGPUS\n");
        output.push_str("GPU  STATE     NAME                     VRAM            GPU   MEM   TEMP   POWER       TOP PROCESS\n");
        for gpu in &gpus {
            let _ = writeln!(
                output,
                "{:<4} {:<9} {:<24} {:<15} {:>4}% {:>4}% {:>5}  {:<11} {}",
                gpu.index,
                gpu_state(gpu, &report.findings),
                truncate(&value_or_dash(&gpu.name), 24),
                memory_pair(gpu.memory_used_mib, gpu.memory_total_mib),
                gpu.gpu_util_percent,
                gpu.memory_util_percent,
                temperature(gpu.temperature_c),
                power(gpu.power_draw_w, gpu.power_limit_w),
                top_process(&gpu.processes),
            );
        }
    }

    render_sources(&mut output, &report.sources, details);
    render_endpoints(&mut output, &report.endpoints);
    render_findings(&mut output, &report.findings, details);
    if details {
        render_details(&mut output, &gpus, &report.topology, &report.xid_events);
    }
    output
}

pub fn render_processes(report: &Report) -> String {
    let mut output = format!(
        "GPU Watchman processes  {}  {}  collected={}\n",
        report.status.to_ascii_uppercase(),
        value_or_dash(&report.host.hostname),
        report.collected_at.format("%Y-%m-%d %H:%M:%S UTC"),
    );
    let mut processes = report
        .gpus
        .iter()
        .flat_map(|gpu| gpu.processes.iter().map(move |process| (gpu, process)))
        .collect::<Vec<_>>();
    processes.sort_by(|(left_gpu, left_process), (right_gpu, right_process)| {
        left_gpu
            .index
            .cmp(&right_gpu.index)
            .then(right_process.memory_mib.cmp(&left_process.memory_mib))
            .then(left_process.pid.cmp(&right_process.pid))
    });
    if processes.is_empty() {
        output.push_str("\nNo GPU processes reported. Inspect SOURCES for accounting coverage.\n");
    } else {
        output.push_str(
            "\nGPU  PID       VRAM       OWNER        POD                    CONTAINER       PROCESS\n",
        );
        for (gpu, process) in processes {
            let _ = writeln!(
                output,
                "{:<4} {:<9} {:>9}  {:<12} {:<22} {:<15} {}",
                gpu.index,
                process.pid,
                memory_value(process.memory_mib),
                truncate(&value_or_dash(&process.owner), 12),
                truncate(process.kubernetes_pod_uid.as_deref().unwrap_or("-"), 22),
                truncate(process.container_id.as_deref().unwrap_or("-"), 15),
                truncate(&value_or_dash(&process.name), 28),
            );
            if !process.command.is_empty() {
                let _ = writeln!(output, "     command: {}", single_line(&process.command));
            }
        }
    }
    render_sources(&mut output, &report.sources, false);
    render_findings(&mut output, &report.findings, false);
    output
}

fn render_sources(output: &mut String, sources: &[SourceStatus], details: bool) {
    let mut sources = sources
        .iter()
        .filter(|source| details || source.state != SourceState::Ok)
        .collect::<Vec<_>>();
    if sources.is_empty() {
        return;
    }
    sources.sort_by(|left, right| left.name.cmp(&right.name));
    output.push_str("\nSOURCES\n");
    output.push_str("STATE        REQUIRED  DURATION  RECORDS  SOURCE\n");
    for source in sources {
        let _ = writeln!(
            output,
            "{:<12} {:<8} {:>7}ms {:>8}  {}",
            source.state.as_str().to_ascii_uppercase(),
            if source.required { "yes" } else { "no" },
            source.duration_ms,
            source.records,
            single_line(&source.name),
        );
        if let Some(error) = source.error.as_deref() {
            let _ = writeln!(output, "  Error: {}", single_line(error));
        }
    }
}

fn render_endpoints(output: &mut String, endpoints: &[Endpoint]) {
    if endpoints.is_empty() {
        return;
    }
    let mut endpoints = endpoints.iter().collect::<Vec<_>>();
    endpoints.sort_by(|left, right| left.metrics_url.cmp(&right.metrics_url));
    output.push_str("\nENDPOINTS\n");
    output.push_str("STATE  RUNTIME        HTTP  LATENCY  RUN   WAIT  KV%    METRICS  ENDPOINT\n");
    for endpoint in endpoints {
        let _ = writeln!(
            output,
            "{:<6} {:<14} {:<5} {:>6}ms {:>5} {:>5} {:>6} {:>8}  {}",
            if endpoint.reachable { "UP" } else { "DOWN" },
            truncate(
                if endpoint.runtime.is_empty() {
                    "unknown"
                } else {
                    &endpoint.runtime
                },
                14
            ),
            if endpoint.status_code == 0 {
                "-".to_owned()
            } else {
                endpoint.status_code.to_string()
            },
            endpoint.latency_ms,
            optional_number(endpoint.requests_running, false),
            optional_number(endpoint.requests_waiting, false),
            optional_number(endpoint.kv_cache_usage_percent, true),
            format!(
                "{}{}",
                endpoint.metric_samples,
                if endpoint.metrics_truncated { "+" } else { "" }
            ),
            value_or_dash(&endpoint.metrics_url),
        );
        if !endpoint.failure.is_empty() {
            let _ = writeln!(output, "       Failure: {}", single_line(&endpoint.failure));
        }
        if !endpoint.rates.is_empty() {
            let _ = writeln!(
                output,
                "       Rates: req/s {} | prompt tok/s {} | generation tok/s {} | errors/s {} | preempt/s {} | TTFT {}ms | TPOT {}ms",
                optional_rate(endpoint.rates.requests_per_second),
                optional_rate(endpoint.rates.prompt_tokens_per_second),
                optional_rate(endpoint.rates.generation_tokens_per_second),
                optional_rate(endpoint.rates.request_errors_per_second),
                optional_rate(endpoint.rates.preemptions_per_second),
                optional_rate(endpoint.rates.mean_time_to_first_token_ms),
                optional_rate(endpoint.rates.mean_time_per_output_token_ms),
            );
            if endpoint.rates.request_latency.is_some()
                || endpoint.rates.time_to_first_token.is_some()
                || endpoint.rates.time_per_output_token.is_some()
                || endpoint.rates.queue_time.is_some()
            {
                let _ = writeln!(
                    output,
                    "       Tails p50/p95/p99: request {} | TTFT {} | TPOT {} | queue {}",
                    interval_quantiles(endpoint.rates.request_latency.as_ref()),
                    interval_quantiles(endpoint.rates.time_to_first_token.as_ref()),
                    interval_quantiles(endpoint.rates.time_per_output_token.as_ref()),
                    interval_quantiles(endpoint.rates.queue_time.as_ref()),
                );
            }
        }
    }
}

fn render_findings(output: &mut String, findings: &[Finding], details: bool) {
    let mut findings = findings
        .iter()
        .filter(|finding| details || finding.severity != Severity::Info)
        .collect::<Vec<_>>();
    if findings.is_empty() {
        return;
    }
    findings.sort_by(|left, right| {
        left.severity
            .rank()
            .cmp(&right.severity.rank())
            .then(left.gpu_index.cmp(&right.gpu_index))
            .then(left.code.cmp(&right.code))
    });
    output.push_str("\nFINDINGS\n");
    for finding in findings {
        let target = finding
            .gpu_index
            .map_or_else(|| "HOST".to_owned(), |index| format!("GPU {index}"));
        let _ = writeln!(
            output,
            "  {:<8} {:<6} {:<28} {}",
            finding.severity.as_str().to_ascii_uppercase(),
            target,
            single_line(&finding.code),
            single_line(&finding.message),
        );
        if !finding.recommendation.is_empty() {
            let _ = writeln!(
                output,
                "           Next: {}",
                single_line(&finding.recommendation)
            );
        }
    }
}

fn render_details(output: &mut String, gpus: &[&Gpu], topology: &str, xid_events: &[String]) {
    if !gpus.is_empty() {
        output.push_str("\nGPU DETAILS\n");
    }
    for gpu in gpus {
        let _ = writeln!(
            output,
            "GPU {}  {}  uuid={}  bus={}",
            gpu.index,
            value_or_dash(&gpu.name),
            value_or_dash(&gpu.uuid),
            value_or_dash(&gpu.pci_bus_id),
        );
        let _ = writeln!(
            output,
            "  clocks {}/{} MHz core, {}/{} MHz memory | PCIe Gen {}/{} x{}/{} | P-state {}",
            gpu.graphics_clock_mhz,
            gpu.max_graphics_clock_mhz,
            gpu.memory_clock_mhz,
            gpu.max_memory_clock_mhz,
            gpu.pcie_gen_current,
            gpu.pcie_gen_max,
            gpu.pcie_width_current,
            gpu.pcie_width_max,
            value_or_dash(&gpu.performance_state),
        );
        let _ = writeln!(
            output,
            "  driver {} | ECC {} | persistence {} | compute mode {} | MIG {}",
            value_or_dash(&gpu.driver),
            if gpu.ecc_enabled {
                "enabled"
            } else {
                "disabled"
            },
            if gpu.persistence_mode {
                "enabled"
            } else {
                "disabled"
            },
            value_or_dash(&gpu.compute_mode),
            value_or_dash(&gpu.mig_mode),
        );
        if !gpu.throttle_reasons.is_empty() {
            let _ = writeln!(
                output,
                "  throttle reasons: {}",
                single_line(&gpu.throttle_reasons.join(", "))
            );
        }
        for process in &gpu.processes {
            let _ = writeln!(
                output,
                "  PID {:<8} {:<24} {:>9}  owner={}  container={}  pod={}",
                process.pid,
                truncate(&value_or_dash(&process.name), 24),
                memory_value(process.memory_mib),
                value_or_dash(&process.owner),
                single_line(process.container_id.as_deref().unwrap_or("-")),
                single_line(process.kubernetes_pod_uid.as_deref().unwrap_or("-")),
            );
            if !process.command.is_empty() {
                let _ = writeln!(output, "    command: {}", single_line(&process.command));
            }
        }
    }
    if !topology.is_empty() {
        let _ = writeln!(output, "\nTOPOLOGY\n{}", safe_multiline(topology.trim()));
    }
    if !xid_events.is_empty() {
        output.push_str("\nXID EVENTS\n");
        for event in xid_events {
            let _ = writeln!(output, "  {}", single_line(event));
        }
    }
}

fn gpu_visible(gpu: &Gpu, findings: &[Finding]) -> bool {
    gpu.is_active()
        || findings.iter().any(|finding| {
            finding.gpu_index == Some(gpu.index) && finding.severity != Severity::Info
        })
}

fn gpu_state(gpu: &Gpu, findings: &[Finding]) -> &'static str {
    let mut state = if gpu.is_active() { "ACTIVE" } else { "IDLE" };
    for finding in findings
        .iter()
        .filter(|finding| finding.gpu_index == Some(gpu.index))
    {
        match finding.severity {
            Severity::Critical => return "CRITICAL",
            Severity::Warning => state = "WARNING",
            Severity::Info => {}
        }
    }
    state
}

fn top_process(processes: &[GpuProcess]) -> String {
    let Some(process) = processes.iter().max_by_key(|process| process.memory_mib) else {
        return "-".to_owned();
    };
    truncate(
        &format!(
            "{}[{}] {}",
            if process.name.is_empty() {
                "unknown"
            } else {
                &process.name
            },
            process.pid,
            memory_value(process.memory_mib)
        ),
        32,
    )
}

fn optional_number(value: Option<f64>, percent: bool) -> String {
    value.filter(|value| value.is_finite()).map_or_else(
        || "-".to_owned(),
        |value| {
            if percent {
                format!("{value:.1}%")
            } else {
                format!("{value:.0}")
            }
        },
    )
}

fn optional_rate(value: Option<f64>) -> String {
    value.filter(|value| value.is_finite()).map_or_else(
        || "-".to_owned(),
        |value| {
            let magnitude = value.abs();
            if magnitude >= 100.0 {
                format!("{value:.0}")
            } else if magnitude >= 10.0 {
                format!("{value:.1}")
            } else if magnitude >= 1.0 {
                format!("{value:.2}")
            } else {
                format!("{value:.3}")
            }
        },
    )
}

fn interval_quantiles(value: Option<&RuntimeHistogramInterval>) -> String {
    let Some(value) = value.filter(|value| {
        [value.samples, value.p50_ms, value.p95_ms, value.p99_ms]
            .into_iter()
            .all(|number| number.is_finite() && number >= 0.0)
    }) else {
        return "-".to_owned();
    };
    format!(
        "{}/{}/{}ms n={}",
        optional_rate(Some(value.p50_ms)),
        optional_rate(Some(value.p95_ms)),
        optional_rate(Some(value.p99_ms)),
        optional_rate(Some(value.samples)),
    )
}

fn value_or_dash(value: &str) -> String {
    let value = single_line(value);
    if value.is_empty() {
        "-".to_owned()
    } else {
        value
    }
}

fn single_line(value: &str) -> String {
    safe_inline(value)
}

fn truncate(value: &str, width: usize) -> String {
    let value = single_line(value);
    let characters = value.chars().collect::<Vec<_>>();
    if characters.len() <= width {
        return value;
    }
    if width <= 3 {
        return characters.into_iter().take(width).collect();
    }
    characters
        .into_iter()
        .take(width - 3)
        .chain(['.', '.', '.'])
        .collect()
}

#[allow(clippy::cast_precision_loss)]
fn memory_pair(used: i64, total: i64) -> String {
    if used >= 1_024 || total >= 1_024 {
        format!(
            "{:.1}/{:.1} GiB",
            used as f64 / 1_024.0,
            total as f64 / 1_024.0
        )
    } else {
        format!("{used}/{total} MiB")
    }
}

#[allow(clippy::cast_precision_loss)]
fn memory_value(mib: i64) -> String {
    if mib >= 1_024 {
        format!("{:.1} GiB", mib as f64 / 1_024.0)
    } else {
        format!("{mib} MiB")
    }
}

fn temperature(celsius: i32) -> String {
    if celsius < 0 {
        "-".to_owned()
    } else {
        format!("{celsius}C")
    }
}

fn power(draw: f64, limit: f64) -> String {
    if draw <= 0.0 && limit <= 0.0 {
        "-".to_owned()
    } else if limit <= 0.0 {
        format!("{draw:.0}W")
    } else {
        format!("{draw:.0}/{limit:.0}W")
    }
}

#[allow(clippy::cast_precision_loss)]
fn percent(value: i64, total: i64) -> f64 {
    if total <= 0 {
        0.0
    } else {
        value as f64 * 100.0 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Finding, Host, SourceState, SourceStatus, Summary};

    #[test]
    fn compact_report_keeps_host_findings_without_visible_gpus() {
        let report = Report {
            host: Host {
                hostname: "node-7".to_owned(),
                os: "linux".to_owned(),
                arch: "x86_64".to_owned(),
            },
            status: "warning".to_owned(),
            summary: Summary {
                gpus: 1,
                warning_findings: 1,
                ..Summary::default()
            },
            gpus: vec![Gpu {
                name: "H100".to_owned(),
                memory_total_mib: 80_000,
                ..Gpu::default()
            }],
            findings: vec![Finding::new(
                None,
                Severity::Warning,
                "inference-endpoint-down",
                "endpoint unavailable",
            )],
            ..Report::default()
        };
        let output = render_text(&report, false, false);
        assert!(output.contains("node-7"));
        assert!(output.contains("No GPU rows shown"));
        assert!(output.contains("inference-endpoint-down"));
    }

    #[test]
    fn source_table_hides_healthy_evidence_until_details_are_requested() {
        let report = Report {
            sources: vec![
                SourceStatus::ok("nvidia.inventory", 10, 2),
                SourceStatus::failed(
                    "nvidia.processes",
                    SourceState::Unavailable,
                    20,
                    0,
                    "permission\ndenied",
                ),
            ],
            ..Report::default()
        };

        let compact = render_text(&report, false, true);
        assert!(compact.contains("SOURCES"));
        assert!(compact.contains("nvidia.processes"));
        assert!(compact.contains("permission denied"));
        assert!(!compact.contains("nvidia.inventory"));

        let detailed = render_text(&report, true, true);
        assert!(detailed.contains("nvidia.inventory"));
        assert!(detailed.contains("nvidia.processes"));
    }

    #[test]
    fn color_is_applied_after_layout_to_status_words_and_headings() {
        let plain = "GPU Watchman  WARNING\n\nGPUS\n0    CRITICAL\n";
        let styled = colorize(plain);

        assert!(styled.contains("\x1b[1;33mWARNING\x1b[0m"));
        assert!(styled.contains("\x1b[1;36mGPUS\x1b[0m"));
        assert!(styled.contains("\x1b[1;31mCRITICAL\x1b[0m"));
        assert_eq!(
            styled
                .replace("\x1b[1;33m", "")
                .replace("\x1b[1;36m", "")
                .replace("\x1b[1;31m", "")
                .replace("\x1b[0m", ""),
            plain
        );
    }

    #[test]
    fn low_runtime_rates_do_not_round_down_to_zero() {
        assert_eq!(optional_rate(Some(0.4)), "0.400");
        assert_eq!(optional_rate(Some(4.25)), "4.25");
        assert_eq!(optional_rate(Some(42.25)), "42.2");
        assert_eq!(optional_rate(None), "-");
    }

    #[test]
    fn interval_tail_quantiles_are_visible_in_the_control_room_view() {
        let report = Report {
            endpoints: vec![Endpoint {
                url: "http://runtime/metrics".to_owned(),
                metrics_url: "http://runtime/metrics".to_owned(),
                reachable: true,
                rates: crate::domain::RuntimeRates {
                    time_to_first_token: Some(RuntimeHistogramInterval {
                        samples: 20.0,
                        p50_ms: 100.0,
                        p95_ms: 250.0,
                        p99_ms: 400.0,
                    }),
                    ..crate::domain::RuntimeRates::default()
                },
                ..Endpoint::default()
            }],
            ..Report::default()
        };

        let output = render_text(&report, false, true);
        assert!(output.contains("Tails p50/p95/p99"));
        assert!(output.contains("TTFT 100/250/400ms n=20.0"));
    }

    #[test]
    fn informational_findings_are_available_without_crowding_the_default_view() {
        let report = Report {
            findings: vec![Finding::new(
                None,
                Severity::Info,
                "gpu-saturated",
                "expected saturation",
            )],
            ..Report::default()
        };

        assert!(!render_text(&report, false, true).contains("gpu-saturated"));
        assert!(render_text(&report, true, true).contains("gpu-saturated"));
    }
}
