//! Bounded-cardinality Prometheus encoding for finalized reports.

use std::fmt::Write as _;

use chrono::{DateTime, Utc};

use crate::domain::{Report, Severity};

#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub fn encode(report: Option<&Report>, last_success: Option<DateTime<Utc>>) -> String {
    let mut output = String::new();
    for (name, help) in METRIC_METADATA {
        let _ = writeln!(output, "# HELP {name} {help}");
        let _ = writeln!(output, "# TYPE {name} gauge");
    }
    metric(
        &mut output,
        "gpu_watchman_report_available",
        "",
        f64::from(report.is_some()),
    );
    let Some(report) = report else {
        return output;
    };
    metric(
        &mut output,
        "gpu_watchman_report_status",
        &labels(&[("status", &report.status)]),
        1.0,
    );
    for severity in [Severity::Critical, Severity::Warning, Severity::Info] {
        metric(
            &mut output,
            "gpu_watchman_findings",
            &labels(&[("severity", severity.as_str())]),
            report
                .findings
                .iter()
                .filter(|finding| finding.severity == severity)
                .count() as f64,
        );
    }
    metric(
        &mut output,
        "gpu_watchman_collection_duration_seconds",
        "",
        report.collection_duration_ms as f64 / 1_000.0,
    );
    for source in &report.sources {
        let source_labels = labels(&[("source", &source.name)]);
        metric(
            &mut output,
            "gpu_watchman_source_status",
            &labels(&[("source", &source.name), ("status", source.state.as_str())]),
            1.0,
        );
        metric(
            &mut output,
            "gpu_watchman_source_required",
            &source_labels,
            f64::from(source.required),
        );
        metric(
            &mut output,
            "gpu_watchman_source_duration_seconds",
            &source_labels,
            source.duration_ms as f64 / 1_000.0,
        );
        metric(
            &mut output,
            "gpu_watchman_source_records",
            &source_labels,
            source.records as f64,
        );
    }
    if let Some(last_success) = last_success {
        metric(
            &mut output,
            "gpu_watchman_last_success_timestamp_seconds",
            "",
            last_success.timestamp_millis() as f64 / 1_000.0,
        );
        metric(
            &mut output,
            "gpu_watchman_last_success_age_seconds",
            "",
            Utc::now()
                .signed_duration_since(last_success)
                .num_milliseconds()
                .max(0) as f64
                / 1_000.0,
        );
    }
    for gpu in &report.gpus {
        let index = gpu.index.to_string();
        let gpu_labels = labels(&[("gpu", &index), ("uuid", &gpu.uuid), ("name", &gpu.name)]);
        for (name, value) in [
            ("gpu_watchman_vram_used_mib", gpu.memory_used_mib as f64),
            ("gpu_watchman_vram_total_mib", gpu.memory_total_mib as f64),
            ("gpu_watchman_vram_free_mib", gpu.memory_free_mib as f64),
            (
                "gpu_watchman_utilization_percent",
                f64::from(gpu.gpu_util_percent),
            ),
            (
                "gpu_watchman_memory_utilization_percent",
                f64::from(gpu.memory_util_percent),
            ),
            (
                "gpu_watchman_temperature_celsius",
                f64::from(gpu.temperature_c),
            ),
            ("gpu_watchman_power_watts", gpu.power_draw_w),
            ("gpu_watchman_power_limit_watts", gpu.power_limit_w),
            ("gpu_watchman_processes", gpu.processes.len() as f64),
            (
                "gpu_watchman_ecc_uncorrected",
                gpu.ecc_uncorrected_volatile as f64,
            ),
            ("gpu_watchman_retired_pages", gpu.retired_pages as f64),
        ] {
            metric(&mut output, name, &gpu_labels, value);
        }
        for process in &gpu.processes {
            let pid = process.pid.to_string();
            let process_labels = labels(&[
                ("gpu", &index),
                ("uuid", &gpu.uuid),
                ("name", &gpu.name),
                ("pid", &pid),
                ("process", &process.name),
                ("owner", &process.owner),
            ]);
            metric(
                &mut output,
                "gpu_watchman_process_vram_mib",
                &process_labels,
                process.memory_mib as f64,
            );
        }
    }
    for endpoint in &report.endpoints {
        let runtime = if endpoint.runtime.is_empty() {
            "unknown"
        } else {
            &endpoint.runtime
        };
        let endpoint_labels = labels(&[("url", &endpoint.url), ("runtime", runtime)]);
        metric(
            &mut output,
            "gpu_watchman_inference_endpoint_up",
            &endpoint_labels,
            f64::from(endpoint.reachable),
        );
        metric(
            &mut output,
            "gpu_watchman_inference_endpoint_latency_seconds",
            &endpoint_labels,
            endpoint.latency_ms as f64 / 1_000.0,
        );
        for (name, value) in [
            (
                "gpu_watchman_inference_requests_running",
                endpoint.requests_running,
            ),
            (
                "gpu_watchman_inference_requests_waiting",
                endpoint.requests_waiting,
            ),
            (
                "gpu_watchman_inference_kv_cache_usage_percent",
                endpoint.kv_cache_usage_percent,
            ),
        ] {
            if let Some(value) = value.filter(|value| value.is_finite()) {
                metric(&mut output, name, &endpoint_labels, value);
            }
        }
        for (name, value) in [
            (
                "gpu_watchman_inference_requests_per_second",
                endpoint.rates.requests_per_second,
            ),
            (
                "gpu_watchman_inference_request_errors_per_second",
                endpoint.rates.request_errors_per_second,
            ),
            (
                "gpu_watchman_inference_prompt_tokens_per_second",
                endpoint.rates.prompt_tokens_per_second,
            ),
            (
                "gpu_watchman_inference_generation_tokens_per_second",
                endpoint.rates.generation_tokens_per_second,
            ),
            (
                "gpu_watchman_inference_preemptions_per_second",
                endpoint.rates.preemptions_per_second,
            ),
            (
                "gpu_watchman_inference_mean_request_latency_seconds",
                endpoint
                    .rates
                    .mean_request_latency_ms
                    .map(|value| value / 1_000.0),
            ),
            (
                "gpu_watchman_inference_mean_time_to_first_token_seconds",
                endpoint
                    .rates
                    .mean_time_to_first_token_ms
                    .map(|value| value / 1_000.0),
            ),
            (
                "gpu_watchman_inference_mean_time_per_output_token_seconds",
                endpoint
                    .rates
                    .mean_time_per_output_token_ms
                    .map(|value| value / 1_000.0),
            ),
        ] {
            if let Some(value) = value.filter(|value| value.is_finite()) {
                metric(&mut output, name, &endpoint_labels, value);
            }
        }
        for (kind, interval) in [
            ("request_latency", endpoint.rates.request_latency.as_ref()),
            (
                "time_to_first_token",
                endpoint.rates.time_to_first_token.as_ref(),
            ),
            (
                "time_per_output_token",
                endpoint.rates.time_per_output_token.as_ref(),
            ),
            ("queue_time", endpoint.rates.queue_time.as_ref()),
        ] {
            let Some(interval) = interval else { continue };
            if interval.samples <= 0.0
                || ![
                    interval.samples,
                    interval.p50_ms,
                    interval.p95_ms,
                    interval.p99_ms,
                ]
                .into_iter()
                .all(|value| value.is_finite() && value >= 0.0)
            {
                continue;
            }
            let kind_labels =
                labels(&[("url", &endpoint.url), ("runtime", runtime), ("kind", kind)]);
            metric(
                &mut output,
                "gpu_watchman_inference_latency_interval_samples",
                &kind_labels,
                interval.samples,
            );
            for (quantile, milliseconds) in [
                ("0.5", interval.p50_ms),
                ("0.95", interval.p95_ms),
                ("0.99", interval.p99_ms),
            ] {
                metric(
                    &mut output,
                    "gpu_watchman_inference_latency_quantile_seconds",
                    &labels(&[
                        ("url", &endpoint.url),
                        ("runtime", runtime),
                        ("kind", kind),
                        ("quantile", quantile),
                    ]),
                    milliseconds / 1_000.0,
                );
            }
        }
    }
    output
}

fn metric(output: &mut String, name: &str, labels: &str, value: f64) {
    if labels.is_empty() {
        let _ = writeln!(output, "{name} {value}");
    } else {
        let _ = writeln!(output, "{name}{{{labels}}} {value}");
    }
}

fn labels(values: &[(&str, &str)]) -> String {
    values
        .iter()
        .map(|(name, value)| format!(r#"{name}="{}""#, escape_label(value)))
        .collect::<Vec<_>>()
        .join(",")
}

fn escape_label(value: &str) -> String {
    value
        .replace('\\', r"\\")
        .replace('\n', r"\n")
        .replace('"', r#"\""#)
}

const METRIC_METADATA: &[(&str, &str)] = &[
    (
        "gpu_watchman_report_available",
        "Whether a successful collection is available.",
    ),
    (
        "gpu_watchman_report_status",
        "Current report status as a labeled gauge.",
    ),
    ("gpu_watchman_findings", "Current findings by severity."),
    (
        "gpu_watchman_collection_duration_seconds",
        "Last collection duration.",
    ),
    (
        "gpu_watchman_last_success_timestamp_seconds",
        "Unix timestamp of last successful collection.",
    ),
    (
        "gpu_watchman_last_success_age_seconds",
        "Age of last successful collection.",
    ),
    (
        "gpu_watchman_source_status",
        "Current telemetry source state as a labeled gauge.",
    ),
    (
        "gpu_watchman_source_required",
        "Whether a telemetry source is required by policy.",
    ),
    (
        "gpu_watchman_source_duration_seconds",
        "Duration of the latest telemetry source collection.",
    ),
    (
        "gpu_watchman_source_records",
        "Records produced by the latest telemetry source collection.",
    ),
    ("gpu_watchman_vram_used_mib", "GPU memory currently used."),
    ("gpu_watchman_vram_total_mib", "Total GPU memory."),
    ("gpu_watchman_vram_free_mib", "GPU memory currently free."),
    (
        "gpu_watchman_utilization_percent",
        "GPU compute utilization.",
    ),
    (
        "gpu_watchman_memory_utilization_percent",
        "GPU memory-controller utilization.",
    ),
    ("gpu_watchman_temperature_celsius", "GPU temperature."),
    ("gpu_watchman_power_watts", "GPU power draw."),
    (
        "gpu_watchman_power_limit_watts",
        "Configured GPU power limit.",
    ),
    ("gpu_watchman_processes", "Processes reported on the GPU."),
    (
        "gpu_watchman_ecc_uncorrected",
        "Volatile uncorrected ECC error count.",
    ),
    ("gpu_watchman_retired_pages", "Retired GPU memory pages."),
    (
        "gpu_watchman_process_vram_mib",
        "GPU memory attributed to a process.",
    ),
    (
        "gpu_watchman_inference_endpoint_up",
        "Inference metrics endpoint reachability.",
    ),
    (
        "gpu_watchman_inference_endpoint_latency_seconds",
        "Inference metrics scrape latency.",
    ),
    (
        "gpu_watchman_inference_requests_running",
        "Normalized running inference requests.",
    ),
    (
        "gpu_watchman_inference_requests_waiting",
        "Normalized queued inference requests.",
    ),
    (
        "gpu_watchman_inference_kv_cache_usage_percent",
        "Normalized KV-cache utilization.",
    ),
    (
        "gpu_watchman_inference_requests_per_second",
        "Completed inference requests per second over the last collection interval.",
    ),
    (
        "gpu_watchman_inference_request_errors_per_second",
        "Inference request errors per second over the last collection interval.",
    ),
    (
        "gpu_watchman_inference_prompt_tokens_per_second",
        "Prompt tokens processed per second over the last collection interval.",
    ),
    (
        "gpu_watchman_inference_generation_tokens_per_second",
        "Generation tokens emitted per second over the last collection interval.",
    ),
    (
        "gpu_watchman_inference_preemptions_per_second",
        "Request preemptions per second over the last collection interval.",
    ),
    (
        "gpu_watchman_inference_mean_request_latency_seconds",
        "Mean request latency observed during the last collection interval.",
    ),
    (
        "gpu_watchman_inference_mean_time_to_first_token_seconds",
        "Mean time to first token observed during the last collection interval.",
    ),
    (
        "gpu_watchman_inference_mean_time_per_output_token_seconds",
        "Mean time per output token observed during the last collection interval.",
    ),
    (
        "gpu_watchman_inference_latency_interval_samples",
        "Samples used for an inference latency histogram over the last collection interval.",
    ),
    (
        "gpu_watchman_inference_latency_quantile_seconds",
        "Estimated inference latency quantile over the last collection interval.",
    ),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        Endpoint, Finding, Gpu, GpuProcess, RuntimeHistogramInterval, RuntimeRates, SourceState,
        SourceStatus,
    };

    #[test]
    fn includes_gpu_process_and_inference_state() {
        let report = Report {
            status: "warning".to_owned(),
            gpus: vec![Gpu {
                index: 0,
                uuid: "GPU-1".to_owned(),
                name: "H100 \"SXM\"".to_owned(),
                memory_used_mib: 42,
                processes: vec![GpuProcess {
                    pid: 9,
                    name: "vllm".to_owned(),
                    memory_mib: 40,
                    ..GpuProcess::default()
                }],
                ..Gpu::default()
            }],
            endpoints: vec![Endpoint {
                url: "http://vllm:8000".to_owned(),
                reachable: true,
                runtime: "vllm".to_owned(),
                requests_waiting: Some(2.0),
                rates: RuntimeRates {
                    request_latency: Some(RuntimeHistogramInterval {
                        samples: 20.0,
                        p50_ms: 125.0,
                        p95_ms: 500.0,
                        p99_ms: 900.0,
                    }),
                    ..RuntimeRates::default()
                },
                ..Endpoint::default()
            }],
            findings: vec![Finding::new(
                None,
                Severity::Warning,
                "inference-queue",
                "queued",
            )],
            sources: vec![SourceStatus {
                name: "nvidia.processes".to_owned(),
                state: SourceState::Partial,
                duration_ms: 25,
                records: 1,
                required: true,
                error: Some("secret diagnostic detail".to_owned()),
            }],
            ..Report::default()
        };
        let output = encode(Some(&report), Some(Utc::now()));
        assert!(output.contains("gpu_watchman_vram_used_mib"));
        assert!(output.contains(r#"name="H100 \"SXM\"""#));
        assert!(output.contains("gpu_watchman_process_vram_mib"));
        assert!(output.contains("gpu_watchman_inference_requests_waiting"));
        assert!(output.contains(
            r#"gpu_watchman_source_status{source="nvidia.processes",status="partial"} 1"#
        ));
        assert!(output.contains(r#"gpu_watchman_source_required{source="nvidia.processes"} 1"#));
        assert!(
            output.contains(
                r#"gpu_watchman_source_duration_seconds{source="nvidia.processes"} 0.025"#
            )
        );
        assert!(output.contains(
            r#"gpu_watchman_inference_latency_interval_samples{url="http://vllm:8000",runtime="vllm",kind="request_latency"} 20"#
        ));
        assert!(output.contains(
            r#"gpu_watchman_inference_latency_quantile_seconds{url="http://vllm:8000",runtime="vllm",kind="request_latency",quantile="0.95"} 0.5"#
        ));
        assert!(!output.contains("secret diagnostic detail"));
    }

    #[test]
    fn advertises_unavailable_before_first_report() {
        let output = encode(None, None);
        assert!(output.contains("gpu_watchman_report_available 0"));
        assert!(!output.contains("gpu_watchman_vram_used_mib{"));
    }
}
