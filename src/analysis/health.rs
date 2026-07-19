//! Deterministic health rules and actionable inference-operations findings.

use crate::domain::{
    Endpoint, Finding, Gpu, Report, SCHEMA_VERSION, Severity, SourceState, SourceStatus, Summary,
};

#[derive(Debug, Clone)]
pub struct AnalyzerConfig {
    pub vram_warning_percent: i64,
    pub vram_critical_percent: i64,
    pub temperature_warning_c: i32,
    pub temperature_critical_c: i32,
    pub kv_cache_warning_percent: f64,
    pub kv_cache_critical_percent: f64,
    pub process_growth_warning_mib: i64,
    pub endpoint_latency_warning_ms: u64,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            vram_warning_percent: 90,
            vram_critical_percent: 99,
            temperature_warning_c: 82,
            temperature_critical_c: 90,
            kv_cache_warning_percent: 85.0,
            kv_cache_critical_percent: 95.0,
            process_growth_warning_mib: 256,
            endpoint_latency_warning_ms: 2_000,
        }
    }
}

#[allow(clippy::too_many_lines)]
pub fn analyze_gpus(gpus: &[Gpu], config: &AnalyzerConfig) -> Vec<Finding> {
    let mut findings = Vec::new();
    for gpu in gpus {
        let memory_percent = gpu.memory_percent();
        let mut add = |severity, code, message| {
            findings.push(Finding::new(Some(gpu.index), severity, code, message));
        };

        if gpu.memory_total_mib <= 0 {
            add(
                Severity::Critical,
                "memory-unavailable",
                "VRAM capacity could not be read".to_owned(),
            );
        } else if memory_percent >= config.vram_critical_percent {
            add(
                Severity::Critical,
                "vram-critical",
                format!(
                    "VRAM is {memory_percent}% full ({}/{} MiB)",
                    gpu.memory_used_mib, gpu.memory_total_mib
                ),
            );
        } else if memory_percent >= config.vram_warning_percent {
            add(
                Severity::Warning,
                "vram-high",
                format!("VRAM is {memory_percent}% full"),
            );
        }

        if gpu.memory_used_mib > 256 && gpu.processes.is_empty() {
            add(
                Severity::Warning,
                "unattributed-vram",
                format!(
                    "{} MiB VRAM is used but no compute or graphics process was reported",
                    gpu.memory_used_mib
                ),
            );
        }
        if gpu.temperature_c >= config.temperature_critical_c {
            add(
                Severity::Critical,
                "temperature-critical",
                format!("GPU temperature is {}C", gpu.temperature_c),
            );
        } else if gpu.temperature_c >= config.temperature_warning_c {
            add(
                Severity::Warning,
                "temperature-high",
                format!("GPU temperature is {}C", gpu.temperature_c),
            );
        }
        if gpu.power_limit_w > 0.0 && gpu.power_draw_w / gpu.power_limit_w >= 0.98 {
            add(
                Severity::Info,
                "power-limit",
                "power draw is at the configured limit".to_owned(),
            );
        }
        if gpu.gpu_util_percent >= 95 {
            add(
                Severity::Info,
                "gpu-saturated",
                format!("GPU utilization is {}%", gpu.gpu_util_percent),
            );
        }
        if gpu.gpu_util_percent < 5 && memory_percent >= 50 {
            add(
                Severity::Info,
                "vram-reserved",
                "substantial VRAM is allocated while the GPU is idle".to_owned(),
            );
        }
        if gpu.pcie_gen_max > 0
            && gpu.pcie_gen_current < gpu.pcie_gen_max
            && gpu.gpu_util_percent >= 50
        {
            add(
                Severity::Warning,
                "pcie-generation",
                format!(
                    "PCIe is Gen {}; GPU supports Gen {}",
                    gpu.pcie_gen_current, gpu.pcie_gen_max
                ),
            );
        }
        if gpu.pcie_width_max > 0
            && gpu.pcie_width_current < gpu.pcie_width_max
            && gpu.gpu_util_percent >= 50
        {
            add(
                Severity::Warning,
                "pcie-width",
                format!(
                    "PCIe link is x{}; GPU supports x{}",
                    gpu.pcie_width_current, gpu.pcie_width_max
                ),
            );
        }
        if gpu.ecc_uncorrected_volatile > 0 {
            add(
                Severity::Critical,
                "ecc-uncorrected",
                format!(
                    "{} uncorrected volatile ECC errors",
                    gpu.ecc_uncorrected_volatile
                ),
            );
        }
        if gpu.ecc_corrected_volatile > 0 {
            add(
                Severity::Warning,
                "ecc-corrected",
                format!(
                    "{} corrected volatile ECC errors",
                    gpu.ecc_corrected_volatile
                ),
            );
        }
        if gpu.retired_pages > 0 {
            add(
                Severity::Critical,
                "retired-pages",
                format!("{} retired memory pages", gpu.retired_pages),
            );
        }
        if !gpu.throttle_reasons.is_empty() {
            add(
                Severity::Warning,
                "clock-throttled",
                format!("active clock throttle: {}", gpu.throttle_reasons.join(", ")),
            );
        }
        if gpu.mig_mode.eq_ignore_ascii_case("enabled") {
            add(
                Severity::Info,
                "mig-enabled",
                "MIG is enabled; inspect instances before attributing whole-GPU VRAM".to_owned(),
            );
        }
        if !gpu.compute_mode.is_empty() && !gpu.compute_mode.eq_ignore_ascii_case("default") {
            add(
                Severity::Info,
                "compute-mode",
                format!("non-default compute mode: {}", gpu.compute_mode),
            );
        }
        if !gpu.processes.is_empty() {
            add(
                Severity::Info,
                "processes",
                format!("{} process(es) using this GPU", gpu.processes.len()),
            );
        }
        // Fanless/passively cooled datacenter cards legitimately report fan N/A.
        if gpu.temperature_c < 0 {
            add(
                Severity::Warning,
                "sensor-unavailable",
                "GPU temperature sensor is unavailable".to_owned(),
            );
        }
    }
    findings
}

pub fn analyze_endpoints(endpoints: &[Endpoint], config: &AnalyzerConfig) -> Vec<Finding> {
    let mut findings = Vec::new();
    for endpoint in endpoints {
        if !endpoint.reachable {
            findings.push(Finding::new(
                None,
                Severity::Warning,
                "inference-endpoint-down",
                format!("{}: {}", endpoint.url, endpoint.failure),
            ));
            continue;
        }
        if endpoint.runtime.is_empty() {
            findings.push(Finding::new(
                None,
                Severity::Info,
                "inference-metrics-unrecognized",
                format!(
                    "{} is reachable, but no supported inference runtime metrics were found",
                    endpoint.url
                ),
            ));
        }
        if endpoint.latency_ms >= config.endpoint_latency_warning_ms {
            findings.push(Finding::new(
                None,
                Severity::Warning,
                "inference-metrics-slow",
                format!(
                    "{} metrics scrape took {} ms",
                    endpoint.url, endpoint.latency_ms
                ),
            ));
        }
        if let Some(waiting) = endpoint.requests_waiting.filter(|value| value.is_finite())
            && waiting > 0.0
        {
            let (severity, code, suffix) = if endpoint
                .requests_running
                .is_some_and(|running| running <= 0.0)
            {
                (
                    Severity::Critical,
                    "inference-stalled",
                    " while no requests are running",
                )
            } else {
                (Severity::Warning, "inference-queue", "")
            };
            findings.push(Finding::new(
                None,
                severity,
                code,
                format!(
                    "{} has {waiting:.0} request(s) waiting{suffix}",
                    endpoint.url
                ),
            ));
        }
        analyze_runtime_rates(endpoint, &mut findings);
        if let Some(usage) = endpoint
            .kv_cache_usage_percent
            .filter(|value| value.is_finite())
        {
            if usage >= config.kv_cache_critical_percent {
                findings.push(Finding::new(
                    None,
                    Severity::Critical,
                    "kv-cache-critical",
                    format!("{} KV cache is {usage:.1}% full", endpoint.url),
                ));
            } else if usage >= config.kv_cache_warning_percent {
                findings.push(Finding::new(
                    None,
                    Severity::Warning,
                    "kv-cache-high",
                    format!("{} KV cache is {usage:.1}% full", endpoint.url),
                ));
            }
        }
    }
    findings
}

/// Turn explicit telemetry requirements into actionable health findings.
pub fn analyze_sources(sources: &[SourceStatus], required_sources: &[String]) -> Vec<Finding> {
    let required = required_sources
        .iter()
        .map(|name| name.trim())
        .filter(|name| !name.is_empty())
        .collect::<std::collections::BTreeSet<_>>();
    let mut findings = Vec::new();
    for name in required {
        let Some(source) = sources.iter().find(|source| source.name == name) else {
            findings.push(Finding::new(
                None,
                Severity::Critical,
                "telemetry-source-required",
                format!("required telemetry source {name} was not reported"),
            ));
            continue;
        };
        let detail = source
            .error
            .as_deref()
            .map_or_else(String::new, |error| format!(": {error}"));
        match source.state {
            SourceState::Ok => {}
            SourceState::Partial => findings.push(Finding::new(
                None,
                Severity::Warning,
                "telemetry-source-partial",
                format!("required telemetry source {name} is partial{detail}"),
            )),
            SourceState::Unavailable => findings.push(Finding::new(
                None,
                Severity::Critical,
                "telemetry-source-required",
                format!("required telemetry source {name} is unavailable{detail}"),
            )),
            SourceState::Skipped => findings.push(Finding::new(
                None,
                Severity::Critical,
                "telemetry-source-required",
                format!("required telemetry source {name} was skipped"),
            )),
        }
    }
    findings
}

fn analyze_runtime_rates(endpoint: &Endpoint, findings: &mut Vec<Finding>) {
    if let Some(errors) = endpoint
        .rates
        .request_errors_per_second
        .filter(|value| value.is_finite() && *value > 0.0)
    {
        findings.push(Finding::new(
            None,
            Severity::Warning,
            "inference-errors",
            format!(
                "{} is reporting {errors:.2} request error(s)/s",
                endpoint.url
            ),
        ));
    }
    if let Some(preemptions) = endpoint
        .rates
        .preemptions_per_second
        .filter(|value| value.is_finite() && *value > 0.0)
    {
        findings.push(Finding::new(
            None,
            Severity::Warning,
            "inference-preemptions",
            format!(
                "{} is reporting {preemptions:.2} preemption(s)/s",
                endpoint.url
            ),
        ));
    }
}

pub fn finalize(report: &mut Report) {
    report.schema_version = SCHEMA_VERSION;
    let mut summary = Summary {
        gpus: report.gpus.len(),
        endpoints: report.endpoints.len(),
        ..Summary::default()
    };
    for gpu in &report.gpus {
        summary.vram_used_mib += gpu.memory_used_mib;
        summary.vram_total_mib += gpu.memory_total_mib;
        summary.processes += gpu.processes.len();
        summary.active_gpus += usize::from(gpu.is_active());
    }
    summary.endpoints_up = report
        .endpoints
        .iter()
        .filter(|endpoint| endpoint.reachable)
        .count();
    let mut status = "healthy";
    for finding in &mut report.findings {
        if finding.recommendation.is_empty() {
            finding.recommendation = recommendation(&finding.code).to_owned();
        }
        match finding.severity {
            Severity::Critical => {
                summary.critical_findings += 1;
                status = "critical";
            }
            Severity::Warning => {
                summary.warning_findings += 1;
                if status == "healthy" {
                    status = "warning";
                }
            }
            Severity::Info => summary.info_findings += 1,
        }
    }
    status.clone_into(&mut report.status);
    report.summary = summary;
}

pub fn recommendation(code: &str) -> &'static str {
    match code {
        "memory-unavailable" => {
            "Verify driver health with nvidia-smi and inspect recent NVIDIA Xid events."
        }
        "vram-critical" => {
            "Drain or reduce the workload before the next allocation fails; inspect the listed VRAM owners."
        }
        "vram-high" => {
            "Check batch size, context length, KV-cache policy, and the largest VRAM owners."
        }
        "unattributed-vram" => {
            "Compare host and container PIDs, then check driver accounting and stale CUDA contexts."
        }
        "temperature-critical" => {
            "Throttle or drain the workload and inspect chassis airflow immediately."
        }
        "temperature-high" => {
            "Inspect cooling, fan behavior, ambient temperature, and sustained power draw."
        }
        "power-limit" => {
            "Confirm the workload is compute-bound and that the configured power cap is intentional."
        }
        "pcie-generation" | "pcie-width" => {
            "Check slot placement, lane allocation, BIOS settings, risers, and link errors under load."
        }
        "ecc-uncorrected" | "retired-pages" => {
            "Drain the GPU and follow the vendor hardware remediation procedure."
        }
        "ecc-corrected" => {
            "Track whether the counter grows and schedule hardware inspection if it does."
        }
        "clock-throttled" => {
            "Correlate the throttle reason with temperature, power, and host hardware telemetry."
        }
        "sensor-unavailable" => {
            "Check whether this GPU exposes the sensor and verify driver permissions."
        }
        "vram-growth" => {
            "Watch several samples and inspect the process for an unbounded cache or allocation leak."
        }
        "xid-events" => {
            "Inspect the original kernel log entries and NVIDIA Xid guidance before reusing the GPU."
        }
        "inference-endpoint-down" => {
            "Verify the metrics URL, runtime process, network path, credentials, and probe timeout."
        }
        "inference-metrics-unrecognized" => {
            "Expose vLLM, TGI, Triton, TensorRT-LLM, SGLang, or Ollama Prometheus metrics."
        }
        "inference-metrics-slow" => {
            "Check runtime load and metrics cardinality; keep scrape latency below the watch interval."
        }
        "inference-queue" => {
            "Check scheduler saturation, batch limits, KV-cache headroom, and arrival rate."
        }
        "inference-stalled" => {
            "Inspect runtime logs and the scheduler; queued work is making no forward progress."
        }
        "inference-errors" => {
            "Inspect runtime error labels and logs, then correlate failures with load and resource pressure."
        }
        "inference-preemptions" => {
            "Reduce KV-cache pressure, context, or concurrency and inspect runtime scheduling policy."
        }
        "telemetry-source-partial" => {
            "Inspect the source error and restore complete telemetry coverage before trusting an all-clear."
        }
        "telemetry-source-required" => {
            "Restore or enable the required telemetry source, then rerun GPU Watchman."
        }
        "kv-cache-high" => {
            "Reduce context or concurrency, enable a safer cache policy, or add capacity."
        }
        "kv-cache-critical" => {
            "Reduce incoming load or context pressure before requests are preempted or rejected."
        }
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{GpuProcess, RuntimeRates};

    #[test]
    fn hardware_faults_are_actionable_and_finalized() {
        let gpu = Gpu {
            index: 2,
            memory_total_mib: 80_000,
            memory_used_mib: 79_500,
            temperature_c: 91,
            ecc_uncorrected_volatile: 2,
            retired_pages: 1,
            processes: vec![GpuProcess {
                pid: 7,
                ..GpuProcess::default()
            }],
            ..Gpu::default()
        };
        let mut report = Report {
            gpus: vec![gpu],
            ..Report::default()
        };
        report.findings = analyze_gpus(&report.gpus, &AnalyzerConfig::default());
        finalize(&mut report);
        assert_eq!(report.status, "critical");
        assert!(report.summary.critical_findings >= 4);
        assert!(report.findings.iter().all(
            |finding| !finding.recommendation.is_empty() || finding.severity == Severity::Info
        ));
    }

    #[test]
    fn endpoint_stall_and_cache_pressure_are_detected() {
        let endpoint = Endpoint {
            url: "http://vllm:8000".to_owned(),
            reachable: true,
            runtime: "vllm".to_owned(),
            requests_running: Some(0.0),
            requests_waiting: Some(3.0),
            kv_cache_usage_percent: Some(98.0),
            rates: RuntimeRates {
                request_errors_per_second: Some(1.0),
                preemptions_per_second: Some(0.5),
                ..RuntimeRates::default()
            },
            ..Endpoint::default()
        };
        let findings = analyze_endpoints(&[endpoint], &AnalyzerConfig::default());
        assert!(
            findings
                .iter()
                .any(|finding| finding.code == "inference-stalled")
        );
        assert!(
            findings
                .iter()
                .any(|finding| finding.code == "kv-cache-critical")
        );
        assert!(
            findings
                .iter()
                .any(|finding| finding.code == "inference-errors")
        );
        assert!(
            findings
                .iter()
                .any(|finding| finding.code == "inference-preemptions")
        );
    }

    #[test]
    fn required_sources_fail_closed_without_penalizing_optional_sources() {
        let sources = vec![
            SourceStatus::failed(
                "nvidia.processes",
                SourceState::Partial,
                10,
                2,
                "graphics accounting failed",
            ),
            SourceStatus::failed(
                "kernel.xid",
                SourceState::Unavailable,
                20,
                0,
                "permission denied",
            ),
            SourceStatus::failed(
                "nvidia.topology",
                SourceState::Unavailable,
                5,
                0,
                "not supported",
            ),
        ];
        let findings = analyze_sources(
            &sources,
            &[
                "nvidia.processes".to_owned(),
                "kernel.xid".to_owned(),
                "missing.source".to_owned(),
            ],
        );

        assert_eq!(findings.len(), 3);
        assert!(findings.iter().any(|finding| {
            finding.code == "telemetry-source-partial"
                && finding.message.contains("graphics accounting failed")
        }));
        assert!(findings.iter().any(|finding| {
            finding.code == "telemetry-source-required" && finding.message.contains("kernel.xid")
        }));
        assert!(findings.iter().any(|finding| {
            finding.code == "telemetry-source-required"
                && finding.message.contains("missing.source")
        }));
        assert!(
            !findings
                .iter()
                .any(|finding| finding.message.contains("nvidia.topology"))
        );
    }
}
