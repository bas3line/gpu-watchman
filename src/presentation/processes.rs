//! Stable text and machine-readable GPU process views.

use chrono::{DateTime, Utc};
use serde::Serialize;

use super::terminal::{OutputFormat, render_processes};
use crate::domain::{Host, Report, SourceStatus};

/// Schema version for the process-specific JSON and NDJSON view.
pub const PROCESS_VIEW_VERSION: u32 = 1;

const PROCESS_SOURCE: &str = "nvidia.processes";

/// Render a process-specific view of a complete collection report.
///
/// Text preserves the operator-oriented process table. JSON and NDJSON use a
/// compact, versioned contract and never fall back to the full report schema.
pub fn render(report: &Report, format: OutputFormat) -> Result<String, serde_json::Error> {
    if format == OutputFormat::Text {
        return Ok(render_processes(report));
    }

    let view = ProcessView::from_report(report);
    let mut output = match format {
        OutputFormat::Json => serde_json::to_string_pretty(&view)?,
        OutputFormat::Ndjson => serde_json::to_string(&view)?,
        OutputFormat::Text => unreachable!("text output returns before serialization"),
    };
    output.push('\n');
    Ok(output)
}

#[derive(Debug, Serialize)]
struct ProcessView<'a> {
    #[serde(rename = "process_view_version")]
    version: u32,
    collected_at: &'a DateTime<Utc>,
    host: ProcessHostView<'a>,
    source: ProcessSourceView<'a>,
    process_count: usize,
    processes: Vec<ProcessRecord<'a>>,
}

impl<'a> ProcessView<'a> {
    fn from_report(report: &'a Report) -> Self {
        let mut processes = report
            .gpus
            .iter()
            .flat_map(|gpu| {
                gpu.processes.iter().map(move |process| ProcessRecord {
                    gpu_index: gpu.index,
                    gpu_uuid: &gpu.uuid,
                    gpu_name: &gpu.name,
                    pid: process.pid,
                    name: &process.name,
                    memory_mib: process.memory_mib,
                    owner: &process.owner,
                    command: &process.command,
                    cgroup: &process.cgroup,
                    container_id: process.container_id.as_deref(),
                    kubernetes_pod_uid: process.kubernetes_pod_uid.as_deref(),
                })
            })
            .collect::<Vec<_>>();
        processes.sort_by(|left, right| {
            left.gpu_index
                .cmp(&right.gpu_index)
                .then_with(|| right.memory_mib.cmp(&left.memory_mib))
                .then_with(|| left.pid.cmp(&right.pid))
                .then_with(|| left.name.cmp(right.name))
                .then_with(|| left.command.cmp(right.command))
        });

        Self {
            version: PROCESS_VIEW_VERSION,
            collected_at: &report.collected_at,
            host: ProcessHostView::from_host(&report.host),
            source: ProcessSourceView::from_report(report),
            process_count: processes.len(),
            processes,
        }
    }
}

#[derive(Debug, Serialize)]
struct ProcessHostView<'a> {
    hostname: &'a str,
    os: &'a str,
    arch: &'a str,
}

impl<'a> ProcessHostView<'a> {
    fn from_host(host: &'a Host) -> Self {
        Self {
            hostname: &host.hostname,
            os: &host.os,
            arch: &host.arch,
        }
    }
}

#[derive(Debug, Serialize)]
struct ProcessSourceView<'a> {
    name: &'static str,
    state: &'static str,
    complete: bool,
    duration_ms: Option<u64>,
    records: Option<u64>,
    required: bool,
    error: Option<&'a str>,
}

impl<'a> ProcessSourceView<'a> {
    fn from_report(report: &'a Report) -> Self {
        report
            .sources
            .iter()
            .find(|source| source.name == PROCESS_SOURCE)
            .map_or_else(Self::missing, Self::from_status)
    }

    const fn missing() -> Self {
        Self {
            name: PROCESS_SOURCE,
            state: "missing",
            complete: false,
            duration_ms: None,
            records: None,
            required: false,
            error: None,
        }
    }

    fn from_status(source: &'a SourceStatus) -> Self {
        Self {
            name: PROCESS_SOURCE,
            state: source.state.as_str(),
            complete: source.state == crate::domain::SourceState::Ok,
            duration_ms: Some(source.duration_ms),
            records: Some(source.records),
            required: source.required,
            error: source.error.as_deref(),
        }
    }
}

#[derive(Debug, Serialize)]
struct ProcessRecord<'a> {
    gpu_index: i32,
    gpu_uuid: &'a str,
    gpu_name: &'a str,
    pid: u32,
    name: &'a str,
    memory_mib: i64,
    owner: &'a str,
    command: &'a str,
    cgroup: &'a str,
    container_id: Option<&'a str>,
    kubernetes_pod_uid: Option<&'a str>,
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone as _;
    use serde_json::{Value, json};

    use super::*;
    use crate::domain::{Gpu, GpuProcess, SourceState};

    fn report() -> Report {
        Report {
            collected_at: Utc.with_ymd_and_hms(2026, 7, 18, 10, 20, 30).unwrap(),
            host: Host {
                hostname: "inference-01".to_owned(),
                os: "linux".to_owned(),
                arch: "x86_64".to_owned(),
            },
            gpus: vec![
                Gpu {
                    index: 1,
                    uuid: "GPU-BBBB".to_owned(),
                    name: "NVIDIA H100".to_owned(),
                    processes: vec![GpuProcess {
                        pid: 30,
                        name: "worker-c".to_owned(),
                        memory_mib: 300,
                        owner: String::new(),
                        command: "worker --rank 1".to_owned(),
                        cgroup: "/kubepods/c".to_owned(),
                        container_id: None,
                        kubernetes_pod_uid: Some("pod-c".to_owned()),
                    }],
                    ..Gpu::default()
                },
                Gpu {
                    index: 0,
                    uuid: "GPU-AAAA".to_owned(),
                    name: "NVIDIA H100".to_owned(),
                    processes: vec![
                        GpuProcess {
                            pid: 20,
                            name: "worker-b".to_owned(),
                            memory_mib: 200,
                            owner: "bob".to_owned(),
                            command: "worker --rank 0".to_owned(),
                            cgroup: "/kubepods/b".to_owned(),
                            container_id: Some("container-b".to_owned()),
                            kubernetes_pod_uid: None,
                        },
                        GpuProcess {
                            pid: 10,
                            name: "worker-a".to_owned(),
                            memory_mib: 400,
                            owner: "alice".to_owned(),
                            command: "worker --rank 0 --leader".to_owned(),
                            cgroup: "/kubepods/a".to_owned(),
                            container_id: Some("container-a".to_owned()),
                            kubernetes_pod_uid: Some("pod-a".to_owned()),
                        },
                    ],
                    ..Gpu::default()
                },
            ],
            sources: vec![
                SourceStatus::ok("nvidia.inventory", 4, 2),
                SourceStatus {
                    name: PROCESS_SOURCE.to_owned(),
                    state: SourceState::Partial,
                    duration_ms: 12,
                    records: 3,
                    required: true,
                    error: Some("graphics accounting unavailable".to_owned()),
                },
            ],
            ..Report::default()
        }
    }

    #[test]
    fn json_is_a_versioned_process_only_contract_with_stable_ordering() {
        let value: Value = serde_json::from_str(
            &render(&report(), OutputFormat::Json).expect("process JSON must serialize"),
        )
        .unwrap();

        assert_eq!(value["process_view_version"], 1);
        assert_eq!(value["collected_at"], "2026-07-18T10:20:30Z");
        assert_eq!(
            value["host"],
            json!({"hostname": "inference-01", "os": "linux", "arch": "x86_64"})
        );
        assert_eq!(
            value["source"],
            json!({
                "name": "nvidia.processes",
                "state": "partial",
                "complete": false,
                "duration_ms": 12,
                "records": 3,
                "required": true,
                "error": "graphics accounting unavailable"
            })
        );
        assert_eq!(value["process_count"], 3);
        let records = value["processes"].as_array().unwrap();
        assert_eq!(records[0]["gpu_index"], 0);
        assert_eq!(records[0]["pid"], 10);
        assert_eq!(records[1]["gpu_index"], 0);
        assert_eq!(records[1]["pid"], 20);
        assert_eq!(records[2]["gpu_index"], 1);
        assert_eq!(records[2]["pid"], 30);
        assert_eq!(records[0]["gpu_uuid"], "GPU-AAAA");
        assert_eq!(records[0]["gpu_name"], "NVIDIA H100");
        assert_eq!(records[0]["owner"], "alice");
        assert_eq!(records[0]["command"], "worker --rank 0 --leader");
        assert_eq!(records[0]["cgroup"], "/kubepods/a");
        assert_eq!(records[0]["container_id"], "container-a");
        assert_eq!(records[0]["kubernetes_pod_uid"], "pod-a");
        assert!(value.get("schema_version").is_none());
        assert!(value.get("gpus").is_none());
        assert!(value.get("endpoints").is_none());
        assert!(value.get("findings").is_none());
        assert!(value.get("summary").is_none());
    }

    #[test]
    fn ndjson_is_one_compact_process_view_record() {
        let report = report();
        let json: Value =
            serde_json::from_str(&render(&report, OutputFormat::Json).unwrap()).unwrap();
        let output = render(&report, OutputFormat::Ndjson).unwrap();

        assert!(output.ends_with('\n'));
        assert_eq!(output.lines().count(), 1);
        assert_eq!(serde_json::from_str::<Value>(&output).unwrap(), json);
    }

    #[test]
    fn missing_source_is_explicit_and_text_delegates_to_the_existing_view() {
        let report = Report {
            sources: Vec::new(),
            ..report()
        };
        let value: Value =
            serde_json::from_str(&render(&report, OutputFormat::Json).unwrap()).unwrap();

        assert_eq!(value["source"]["state"], "missing");
        assert_eq!(value["source"]["complete"], false);
        assert!(value["source"]["duration_ms"].is_null());
        assert!(value["source"]["records"].is_null());
        assert_eq!(
            render(&report, OutputFormat::Text).unwrap(),
            render_processes(&report)
        );
    }

    #[test]
    fn complete_source_and_optional_identity_fields_are_preserved() {
        let mut report = report();
        report.sources[1] = SourceStatus::ok(PROCESS_SOURCE, 8, 3);
        let value: Value =
            serde_json::from_str(&render(&report, OutputFormat::Json).unwrap()).unwrap();

        assert_eq!(value["source"]["state"], "ok");
        assert_eq!(value["source"]["complete"], true);
        let third = &value["processes"][2];
        assert_eq!(third["owner"], "");
        assert!(third["container_id"].is_null());
        assert_eq!(third["kubernetes_pod_uid"], "pod-c");
    }
}
