//! Environment preflight checks for drivers, telemetry, identity, and endpoints.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Duration;

use serde::Serialize;

use crate::domain::{Endpoint, Report, SourceState};
use crate::inference::{ProbeOptions, collect};
use crate::presentation::safe_inline;
use crate::telemetry::NvidiaCollector;

#[derive(Debug, Clone, Serialize)]
pub struct Check {
    pub name: String,
    pub status: CheckStatus,
    pub detail: String,
    pub remedy: String,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CheckStatus {
    Pass,
    Warn,
    Fail,
}

pub fn run(
    command: &Path,
    timeout: Duration,
    probes: &[String],
    probe_token: Option<String>,
    allow_insecure_http: bool,
) -> Vec<Check> {
    let collector = NvidiaCollector::new(command, timeout).without_xid();
    let mut checks = vec![driver_check(&collector)];
    match collector.collect() {
        Ok(report) => {
            checks.extend(telemetry_checks(&report));
            if let Some(check) = process_attribution_check(&report) {
                checks.push(check);
            }
        }
        Err(error) => checks.push(Check {
            name: "gpu-telemetry".to_owned(),
            status: CheckStatus::Fail,
            detail: error.to_string(),
            remedy:
                "Check device visibility, container GPU passthrough, and nvidia-smi permissions."
                    .to_owned(),
        }),
    }
    checks.extend(
        collect(
            probes,
            &ProbeOptions {
                timeout,
                bearer_token: probe_token,
                allow_insecure_http,
                ..ProbeOptions::default()
            },
        )
        .iter()
        .map(endpoint_check),
    );
    checks
}

/// Build doctor evidence from a report already collected by another workflow.
pub fn from_report(command: &Path, timeout: Duration, report: &Report) -> Vec<Check> {
    let collector = NvidiaCollector::new(command, timeout).without_xid();
    let mut checks = vec![driver_check(&collector)];
    checks.extend(telemetry_checks(report));
    if let Some(check) = process_attribution_check(report) {
        checks.push(check);
    }
    checks.extend(report.endpoints.iter().map(endpoint_check));
    checks
}

fn driver_check(collector: &NvidiaCollector) -> Check {
    match collector.driver_version() {
        Ok(version) if !version.is_empty() => Check {
            name: "nvidia-driver".to_owned(),
            status: CheckStatus::Pass,
            detail: format!("nvidia-smi responded; driver {version}"),
            remedy: String::new(),
        },
        Ok(_) => Check {
            name: "nvidia-driver".to_owned(),
            status: CheckStatus::Warn,
            detail: "nvidia-smi responded without a driver version".to_owned(),
            remedy: "Run nvidia-smi directly and inspect the driver installation.".to_owned(),
        },
        Err(error) => Check {
            name: "nvidia-driver".to_owned(),
            status: CheckStatus::Fail,
            detail: error.to_string(),
            remedy: "Install or repair the NVIDIA driver and place nvidia-smi in PATH.".to_owned(),
        },
    }
}

fn telemetry_checks(report: &Report) -> Vec<Check> {
    let mut checks = vec![Check {
        name: "gpu-telemetry".to_owned(),
        status: CheckStatus::Pass,
        detail: format!(
            "{} GPU(s), {} process record(s), {} telemetry source(s)",
            report.gpus.len(),
            report
                .gpus
                .iter()
                .map(|gpu| gpu.processes.len())
                .sum::<usize>(),
            report.sources.len(),
        ),
        remedy: String::new(),
    }];
    checks.extend(report.sources.iter().map(|source| {
        let status = if source.state == SourceState::Ok {
            CheckStatus::Pass
        } else {
            CheckStatus::Warn
        };
        Check {
            name: format!("telemetry-source:{}", source.name),
            status,
            detail: format!(
                "state={}, records={}, duration={}ms{}",
                source.state,
                source.records,
                source.duration_ms,
                source
                    .error
                    .as_deref()
                    .map_or_else(String::new, |error| format!(", {error}"))
            ),
            remedy: if status == CheckStatus::Pass {
                String::new()
            } else {
                "Check command permissions, container mounts, and host telemetry access.".to_owned()
            },
        }
    }));
    checks
}

fn process_attribution_check(report: &Report) -> Option<Check> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    let processes = report
        .gpus
        .iter()
        .flat_map(|gpu| gpu.processes.iter())
        .collect::<Vec<_>>();
    let attributed = processes
        .iter()
        .filter(|process| {
            !process.owner.is_empty() || !process.command.is_empty() || !process.cgroup.is_empty()
        })
        .count();
    let (status, detail, remedy) = if processes.is_empty() {
        (
            CheckStatus::Warn,
            "no live GPU process was available, so workload attribution was not exercised"
                .to_owned(),
            "Run doctor while a representative GPU workload is active, or inspect the nvidia.processes source explicitly."
                .to_owned(),
        )
    } else if attributed == processes.len() {
        (
            CheckStatus::Pass,
            format!(
                "owner, command, or cgroup evidence was recovered for all {} GPU process record(s)",
                processes.len()
            ),
            String::new(),
        )
    } else {
        (
            CheckStatus::Warn,
            format!(
                "workload attribution was recovered for {attributed}/{} GPU process record(s)",
                processes.len()
            ),
            "Expose the host PID namespace and a read-only host /proc view; check hidepid and process ownership permissions."
                .to_owned(),
        )
    };
    Some(Check {
        name: "process-attribution".to_owned(),
        status,
        detail,
        remedy,
    })
}

fn endpoint_check(endpoint: &Endpoint) -> Check {
    let status =
        if !endpoint.reachable || endpoint.metric_samples == 0 || endpoint.runtime.is_empty() {
            CheckStatus::Fail
        } else if endpoint.metrics_truncated {
            CheckStatus::Warn
        } else {
            CheckStatus::Pass
        };
    Check {
        name: format!("inference-endpoint:{}", endpoint.url),
        status,
        detail: if endpoint.reachable {
            format!(
                "HTTP {}, runtime={}, {} samples{}, {} ms, metrics={}",
                endpoint.status_code,
                if endpoint.runtime.is_empty() {
                    "unrecognized"
                } else {
                    &endpoint.runtime
                },
                endpoint.metric_samples,
                if endpoint.metrics_truncated { "+" } else { "" },
                endpoint.latency_ms,
                endpoint.metrics_url,
            )
        } else {
            endpoint.failure.clone()
        },
        remedy: match status {
            CheckStatus::Pass => String::new(),
            CheckStatus::Warn => {
                "Reduce endpoint metric cardinality or expose a runtime-filtered metrics view."
                    .to_owned()
            }
            CheckStatus::Fail => {
                "Verify the final metrics URL, supported runtime families, credentials, service routing, and network policy."
                    .to_owned()
            }
        },
    }
}

pub fn render_text(checks: &[Check]) -> String {
    let mut output = String::from("GPU Watchman doctor\n");
    for check in checks {
        let state = match check.status {
            CheckStatus::Pass => "PASS",
            CheckStatus::Warn => "WARN",
            CheckStatus::Fail => "FAIL",
        };
        let _ = writeln!(
            output,
            "  {state:<4}  {:<28} {}",
            safe_inline(&check.name),
            safe_inline(&check.detail)
        );
        if !check.remedy.is_empty() && check.status != CheckStatus::Pass {
            let _ = writeln!(output, "        Next: {}", safe_inline(&check.remedy));
        }
    }
    output
}

pub fn failed(checks: &[Check]) -> bool {
    checks.iter().any(|check| check.status == CheckStatus::Fail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reachable_endpoint_without_recognized_metrics_fails_closed() {
        let endpoint = Endpoint {
            url: "http://127.0.0.1:8000".to_owned(),
            metrics_url: "http://127.0.0.1:8000/metrics".to_owned(),
            reachable: true,
            status_code: 200,
            ..Endpoint::default()
        };

        let check = endpoint_check(&endpoint);
        assert_eq!(check.status, CheckStatus::Fail);
        assert!(check.detail.contains("runtime=unrecognized"));
        assert!(check.detail.contains("/metrics"));
    }

    #[test]
    fn truncated_recognized_metrics_warn_instead_of_claiming_complete_success() {
        let endpoint = Endpoint {
            runtime: "vllm".to_owned(),
            reachable: true,
            status_code: 200,
            metric_samples: 10_000,
            metrics_truncated: true,
            ..Endpoint::default()
        };

        assert_eq!(endpoint_check(&endpoint).status, CheckStatus::Warn);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn process_attribution_uses_gpu_workload_evidence_not_self_proc() {
        let empty = process_attribution_check(&Report::default()).unwrap();
        assert_eq!(empty.status, CheckStatus::Warn);
        assert!(empty.detail.contains("not exercised"));

        let mut report = Report::default();
        report.gpus.push(crate::domain::Gpu {
            processes: vec![crate::domain::GpuProcess {
                pid: 42,
                name: "python".to_owned(),
                ..crate::domain::GpuProcess::default()
            }],
            ..crate::domain::Gpu::default()
        });
        let missing = process_attribution_check(&report).unwrap();
        assert_eq!(missing.status, CheckStatus::Warn);
        assert!(missing.detail.contains("0/1"));

        report.gpus[0].processes[0].owner = "inference".to_owned();
        let attributed = process_attribution_check(&report).unwrap();
        assert_eq!(attributed.status, CheckStatus::Pass);
        assert!(attributed.detail.contains("all 1"));
    }
}
