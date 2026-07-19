#![cfg(unix)]

use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::fd::OwnedFd;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::thread::JoinHandle;

use gpu_watchman::domain::{
    CANARY_VERSION, CanaryAttempt, CanaryDistribution, CanaryFailureStage, CanaryGate, CanaryPlan,
    CanaryPolicy, CanaryReport, CanaryStatus, CanarySummary, CanaryTarget, Finding, Report,
    SATURATION_BENCHMARK_NONCLAIMS, SATURATION_BENCHMARK_VERSION, SaturationAssessment,
    SaturationAssessmentStatus, SaturationAttempt, SaturationBenchmarkReport,
    SaturationFailureStageCounts, SaturationGate, SaturationGateKind, SaturationGateOperator,
    SaturationGateStatus, SaturationLoadModel, SaturationPhaseResult, SaturationPhaseStatus,
    SaturationPlan, SaturationPolicy, SaturationRoute, SaturationRunStatus, SaturationSchedule,
    SaturationStageOrder, SaturationStageResult, SaturationStageSummary, SaturationTarget,
    SaturationVerification, SaturationWarmupScope, SaturationWorkerStart, Severity,
};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_gpu-watchman"))
        .args(args)
        .output()
        .unwrap()
}

fn assert_compact_ndjson(output: &Output) -> serde_json::Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    let stdout = std::str::from_utf8(&output.stdout).unwrap();
    assert!(stdout.ends_with('\n'));
    assert_eq!(
        stdout.matches('\n').count(),
        1,
        "NDJSON output must contain exactly one compact record: {stdout:?}"
    );
    serde_json::from_str(stdout).unwrap()
}

fn fake_nvidia_smi() -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("nvidia-smi");
    write_fake_nvidia_smi(&path, "GPU-test", "NVIDIA H100");
    (directory, path)
}

fn write_fake_nvidia_smi(path: &std::path::Path, gpu_uuid: &str, gpu_name: &str) {
    let script = r#"#!/bin/sh
case "$1" in
  --query-gpu=index,*)
    printf '%s\n' '0,NVIDIA H100,GPU-test,580.1,0000:01:00.0,P0,75,N/A,500,700,1800,2000,2100,2500,81920,40960,40960,97,80,5,5,16,16,Default,Enabled,Enabled,0,0'
    ;;
  --query-compute-apps=*)
    printf '%s\n' 'GPU-test,4242,vllm,40000'
    ;;
  --query-graphics-apps=*)
    ;;
  --query-gpu=uuid,retired_pages.single_bit_ecc.count)
    printf '%s\n' 'GPU-test,0'
    ;;
  --query-gpu=uuid,retired_pages.double_bit_ecc.count)
    printf '%s\n' 'GPU-test,0'
    ;;
  --query-gpu=uuid,*)
    printf '%s\n' 'GPU-test,Disabled,Not Active,Not Active,Not Active,Not Active,Not Active'
    ;;
  topo)
    printf '%s\n' 'GPU0 CPU Affinity' 'GPU0 X 0-31'
    ;;
  --query-gpu=driver_version)
    printf '%s\n' '580.1'
    ;;
  *)
    printf '%s\n' "unexpected fake nvidia-smi invocation: $*" >&2
    exit 1
    ;;
esac
"#
    .replace("GPU-test", gpu_uuid)
    .replace("NVIDIA H100", gpu_name);
    fs::write(path, script).unwrap();
    let mut permissions = fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions).unwrap();
}

fn write_process_incomplete_nvidia_smi(path: &std::path::Path) {
    write_fake_nvidia_smi(path, "GPU-test", "NVIDIA H100");
    let script = fs::read_to_string(path)
        .unwrap()
        .replace("printf '%s\\n' 'GPU-test,4242,vllm,40000'", "exit 42")
        .replace(
            "  --query-graphics-apps=*)\n    ;;",
            "  --query-graphics-apps=*)\n    exit 42\n    ;;",
        );
    fs::write(path, script).unwrap();
}

fn invocation_marking_nvidia_smi() -> (tempfile::TempDir, PathBuf, PathBuf) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("nvidia-smi");
    let marker = directory.path().join("nvidia-smi.invoked");
    fs::write(
        &path,
        r#"#!/bin/sh
: > "$0.invoked"
exit 97
"#,
    )
    .unwrap();
    let mut permissions = fs::metadata(&path).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&path, permissions).unwrap();
    (directory, path, marker)
}

fn write_private_file(path: &std::path::Path, body: &str) {
    fs::write(path, body).unwrap();
    let mut permissions = fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o600);
    fs::set_permissions(path, permissions).unwrap();
}

fn write_private_config(path: &std::path::Path, body: &str) {
    write_private_file(path, body);
}

fn write_test_safetensors(path: &std::path::Path) {
    let mut header = serde_json::to_vec(&serde_json::json!({
        "__metadata__": {"private_note": "must-not-leak"},
        "private.layer.weight": {
            "dtype": "BF16",
            "shape": [2, 2],
            "data_offsets": [0, 8]
        }
    }))
    .unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut artifact = Vec::new();
    artifact.extend_from_slice(&u64::try_from(header.len()).unwrap().to_le_bytes());
    artifact.extend_from_slice(&header);
    artifact.extend_from_slice(&[0_u8; 8]);
    fs::write(path, artifact).unwrap();
}

fn write_sparse_test_safetensors(path: &std::path::Path, payload_bytes: u64) {
    let mut header = serde_json::to_vec(&serde_json::json!({
        "__metadata__": {"private_note": "must-not-leak"},
        "private.large.weight": {
            "dtype": "U8",
            "shape": [payload_bytes],
            "data_offsets": [0, payload_bytes]
        }
    }))
    .unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut artifact = Vec::new();
    artifact.extend_from_slice(&u64::try_from(header.len()).unwrap().to_le_bytes());
    artifact.extend_from_slice(&header);
    fs::write(path, artifact).unwrap();
    fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_len(8 + u64::try_from(header.len()).unwrap() + payload_bytes)
        .unwrap();
}

fn rollout_canary(
    workload_id: &str,
    ttft_p95: f64,
    e2e_p95: f64,
    output_tps_p50: f64,
    succeeded: u32,
) -> CanaryReport {
    assert!(succeeded <= 100);
    let distribution = |value| CanaryDistribution {
        samples: usize::try_from(succeeded).unwrap(),
        min: value,
        mean: value,
        p50: value,
        p95: value,
        p99: value,
        max: value,
    };
    let attempts = (0..100)
        .map(|index| {
            if index < succeeded {
                CanaryAttempt {
                    index,
                    success: true,
                    status_code: 200,
                    headers_ms: Some(ttft_p95.min(10.0)),
                    ttft_ms: Some(ttft_p95),
                    e2e_ms: Some(e2e_p95),
                    prompt_tokens: Some(4),
                    completion_tokens: Some(3),
                    output_tokens_per_second: Some(output_tps_p50),
                    model: String::new(),
                    finish_reason: "stop".to_owned(),
                    expectation_met: None,
                    failure: None,
                }
            } else {
                CanaryAttempt::failed(index, CanaryFailureStage::Transport, "connection failed")
            }
        })
        .collect();
    let policy = CanaryPolicy {
        min_success_percent: 90.0,
        max_ttft_ms: None,
        max_e2e_ms: None,
        min_output_tokens_per_second: None,
        expectation_configured: false,
    };
    let success_percent = f64::from(succeeded);
    let gates = vec![
        CanaryGate {
            name: "min_successful_requests".to_owned(),
            operator: ">=".to_owned(),
            observed: Some(f64::from(succeeded)),
            threshold: 1.0,
            passed: succeeded > 0,
            detail: String::new(),
        },
        CanaryGate {
            name: "min_success_percent".to_owned(),
            operator: ">=".to_owned(),
            observed: Some(success_percent),
            threshold: policy.min_success_percent,
            passed: success_percent >= policy.min_success_percent,
            detail: String::new(),
        },
    ];
    CanaryReport {
        canary_version: CANARY_VERSION,
        started_at: chrono::Utc::now(),
        duration_ms: 50_000,
        status: CanaryStatus::Pass,
        workload_id: workload_id.to_owned(),
        target: CanaryTarget {
            url: "https://inference.example".to_owned(),
            route: "chat_completions".to_owned(),
            model: "served-model".to_owned(),
            stream: true,
        },
        plan: CanaryPlan {
            count: 100,
            concurrency: 4,
            max_tokens: 32,
            timeout_ms: 30_000,
            response_limit_bytes: 1 << 20,
        },
        policy: Some(policy),
        summary: CanarySummary {
            attempted: 100,
            succeeded,
            failed: 100 - succeeded,
            success_percent,
            achieved_requests_per_second: 2.0,
            prompt_tokens_total: u64::from(succeeded) * 4,
            completion_tokens_total: u64::from(succeeded) * 3,
            headers_ms: Some(distribution(ttft_p95.min(10.0))),
            ttft_ms: Some(distribution(ttft_p95)),
            e2e_ms: Some(distribution(e2e_p95)),
            output_tokens_per_second: Some(distribution(output_tps_p50)),
        },
        gates,
        attempts,
    }
}

#[allow(clippy::too_many_lines)]
fn saved_saturation_benchmark(
    started_at: chrono::DateTime<chrono::Utc>,
    origin: &str,
    stage_duration_ns: u64,
) -> SaturationBenchmarkReport {
    let distribution = |value| CanaryDistribution {
        samples: 20,
        min: value,
        mean: value,
        p50: value,
        p95: value,
        p99: value,
        max: value,
    };
    let attempts = (0..20)
        .map(|index| SaturationAttempt {
            index,
            success: true,
            status_code: 200,
            headers_ms: Some(1.0),
            ttft_ms: Some(10.0),
            e2e_ms: Some(20.0),
            prompt_tokens: Some(4),
            completion_tokens: Some(3),
            output_tokens_per_second: Some(200.0),
            expectation_met: Some(true),
            failure_stage: None,
        })
        .collect();
    let seconds = std::time::Duration::from_nanos(stage_duration_ns).as_secs_f64();
    let summary = SaturationStageSummary {
        attempted: 20,
        succeeded: 20,
        failed: 0,
        error_percent: 0.0,
        attempted_requests_per_second: 20.0 / seconds,
        successful_requests_per_second: 20.0 / seconds,
        prompt_token_samples: 20,
        completion_token_samples: 20,
        prompt_tokens_observed_total: 80,
        completion_tokens_observed_total: 60,
        total_tokens_observed_total: 140,
        completion_token_usage_complete: true,
        total_token_usage_complete: true,
        completion_token_goodput_per_second: Some(60.0 / seconds),
        total_token_goodput_per_second: Some(140.0 / seconds),
        headers_ms: Some(distribution(1.0)),
        ttft_ms: Some(distribution(10.0)),
        e2e_ms: Some(distribution(20.0)),
        output_tokens_per_second: Some(distribution(200.0)),
        failure_stage_counts: SaturationFailureStageCounts::default(),
    };
    let policy = SaturationPolicy {
        max_error_percent: 90.0,
        max_p95_ttft_ms: None,
        max_p95_e2e_ms: None,
        min_successful_requests_per_second: None,
        min_completion_token_goodput_per_second: None,
        expectation_configured: true,
        abort_error_percent: 100.0,
        minimum_latency_samples: 20,
        signal_max_marginal_scaling_efficiency_percent: 5.0,
        signal_min_p95_latency_inflation_percent: 20.0,
    };
    let stage = SaturationStageResult {
        status: SaturationPhaseStatus::Complete,
        concurrency: 1,
        planned_requests: 20,
        duration_ns: stage_duration_ns,
        duration_ms: stage_duration_ns / 1_000_000,
        gates: vec![
            SaturationGate {
                kind: SaturationGateKind::SuccessfulRequests,
                operator: SaturationGateOperator::GreaterThanOrEqual,
                observed: Some(20.0),
                threshold: 1.0,
                status: SaturationGateStatus::Pass,
                reason: None,
                samples: None,
                required_samples: None,
            },
            SaturationGate {
                kind: SaturationGateKind::ErrorPercent,
                operator: SaturationGateOperator::LessThanOrEqual,
                observed: Some(0.0),
                threshold: 90.0,
                status: SaturationGateStatus::Pass,
                reason: None,
                samples: None,
                required_samples: None,
            },
        ],
        summary,
        attempts,
    };
    let warmup_duration_ns = 10_000_000;
    SaturationBenchmarkReport {
        saturation_benchmark_version: SATURATION_BENCHMARK_VERSION,
        started_at,
        duration_ns: warmup_duration_ns + stage_duration_ns,
        duration_ms: (warmup_duration_ns + stage_duration_ns) / 1_000_000,
        status: SaturationRunStatus::Complete,
        abort_reason: None,
        workload_id: "builtin-v1".to_owned(),
        target: SaturationTarget {
            url: origin.to_owned(),
            route: SaturationRoute::ChatCompletions,
            model: "served-model".to_owned(),
            stream: true,
        },
        plan: SaturationPlan {
            concurrency_stages: vec![1],
            warmup_requests_per_worker: 1,
            requests_per_worker: 20,
            planned_attempts: 21,
            max_tokens: 8,
            timeout_ns: 10_000_000_000,
            timeout_ms: 10_000,
            response_limit_bytes: 128 << 10,
            schedule: SaturationSchedule {
                load_model: SaturationLoadModel::ClosedLoopFixedConcurrency,
                stage_order: SaturationStageOrder::ExplicitAscending,
                warmup_scope: SaturationWarmupScope::EachStageExcluded,
                worker_start: SaturationWorkerStart::SimultaneousBarrier,
            },
        },
        policy,
        warmups: vec![SaturationPhaseResult {
            status: SaturationPhaseStatus::Complete,
            concurrency: 1,
            planned_requests: 1,
            attempted: 1,
            succeeded: 1,
            failed: 0,
            error_percent: 0.0,
            duration_ns: warmup_duration_ns,
            duration_ms: warmup_duration_ns / 1_000_000,
            failure_stage_counts: SaturationFailureStageCounts::default(),
        }],
        stages: vec![stage],
        assessment: SaturationAssessment {
            status: SaturationAssessmentStatus::NotEvaluable,
            signal: None,
            first_signal_concurrency: None,
            highest_accepted_tested_concurrency: Some(1),
            scaling_evidence: Vec::new(),
        },
        verification: SaturationVerification::default(),
        nonclaims: SATURATION_BENCHMARK_NONCLAIMS.to_vec(),
    }
}

fn clean_config_command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_gpu-watchman"));
    for variable in [
        "GPU_WATCHMAN_CONFIG",
        "GPU_WATCHMAN_PROFILE",
        "GPU_WATCHMAN_NVIDIA_SMI",
        "GPU_WATCHMAN_PROBE_TOKEN",
        "GPU_WATCHMAN_PROBE_TOKEN_FILE",
    ] {
        command.env_remove(variable);
    }
    command
}

fn operational_profile_fixture() -> (tempfile::TempDir, PathBuf) {
    let directory = tempfile::tempdir().unwrap();
    let profile_driver = directory.path().join("profile-smi");
    write_fake_nvidia_smi(&profile_driver, "GPU-profile", "Profile GPU");
    let config = directory.path().join("profiles.toml");
    write_private_config(
        &config,
        r#"config_version = 1
default_profile = "ops"

[profiles.ops.monitor.collection]
nvidia_smi = "./profile-smi"
collect_xid = false

[profiles.ops.monitor.inference]
urls = ["http://127.0.0.1:9"]
token_file = "missing-profile-token"
timeout = "100ms"
"#,
    );
    (directory, config)
}

fn serve_openai_once(status: &str, content_type: &str, body: &str) -> (String, JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let status = status.to_owned();
    let content_type = content_type.to_owned();
    let body = body.to_owned();
    let task = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let bytes = stream.read(&mut buffer).unwrap();
            if bytes == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..bytes]);
            let Some(headers_end) = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|index| index + 4)
            else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..headers_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    line.split_once(':').and_then(|(name, value)| {
                        if name.eq_ignore_ascii_case("content-length") {
                            Some(value.trim().parse::<usize>().unwrap())
                        } else {
                            None
                        }
                    })
                })
                .unwrap_or(0);
            if request.len() >= headers_end + content_length {
                break;
            }
        }

        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).unwrap();
        stream.flush().unwrap();
        String::from_utf8(request).unwrap()
    });
    (format!("http://{address}/v1"), task)
}

fn serve_openai_many(
    count: usize,
    status: &str,
    content_type: &str,
    body: &str,
) -> (String, JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let status = status.to_owned();
    let content_type = content_type.to_owned();
    let body = body.to_owned();
    let task = std::thread::spawn(move || {
        (0..count)
            .map(|_| {
                let (mut stream, _) = listener.accept().unwrap();
                let request = read_http_request(&mut stream);
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
                request
            })
            .collect()
    });
    (format!("http://{address}/v1"), task)
}

fn serve_openai_sequence(responses: Vec<(&str, &str, &str)>) -> (String, JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let responses = responses
        .into_iter()
        .map(|(status, content_type, body)| {
            (status.to_owned(), content_type.to_owned(), body.to_owned())
        })
        .collect::<Vec<_>>();
    let task = std::thread::spawn(move || {
        responses
            .into_iter()
            .map(|(status, content_type, body)| {
                let (mut stream, _) = listener.accept().unwrap();
                let request = read_http_request(&mut stream);
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
                request
            })
            .collect()
    });
    (format!("http://{address}/v1"), task)
}

fn read_http_request(stream: &mut TcpStream) -> String {
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .unwrap();
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let bytes = stream.read(&mut buffer).unwrap();
        if bytes == 0 {
            break;
        }
        request.extend_from_slice(&buffer[..bytes]);
        let Some(headers_end) = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|index| index + 4)
        else {
            continue;
        };
        let headers = String::from_utf8_lossy(&request[..headers_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                line.split_once(':').and_then(|(name, value)| {
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
            })
            .unwrap_or(0);
        if request.len() >= headers_end + content_length {
            break;
        }
    }
    String::from_utf8(request).unwrap()
}

#[test]
fn snapshot_executes_the_driver_protocol_and_emits_the_stable_schema() {
    let (_directory, nvidia_smi) = fake_nvidia_smi();
    let output = Command::new(env!("CARGO_BIN_EXE_gpu-watchman"))
        .args([
            "snapshot",
            "--nvidia-smi",
            nvidia_smi.to_str().unwrap(),
            "--format",
            "json",
            "--no-xid",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Report = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report.schema_version, 3);
    assert_eq!(report.gpus.len(), 1);
    assert_eq!(report.gpus[0].name, "NVIDIA H100");
    assert_eq!(report.gpus[0].processes[0].name, "vllm");
    assert_eq!(report.gpus[0].fan_percent, -1);
    assert!(
        !report
            .findings
            .iter()
            .any(|finding| finding.code == "sensor-unavailable")
    );
}

#[test]
fn capacity_command_is_available_without_gpu_hardware() {
    let output = Command::new(env!("CARGO_BIN_EXE_gpu-watchman"))
        .args([
            "capacity",
            "--params",
            "7",
            "--gpu-vram",
            "24",
            "--layers",
            "32",
            "--kv-heads",
            "8",
            "--head-dim",
            "128",
            "--weight-bits",
            "4",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("capacity report v3"));
    assert!(text.contains("KV cache"));
}

#[test]
fn capacity_artifact_floor_is_path_free_monotonic_and_can_change_fit_to_false() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory
        .path()
        .join("private-capacity-checkpoint.safetensors");
    let payload_bytes = 4_u64 * 1024 * 1024 * 1024;
    write_sparse_test_safetensors(&path, payload_bytes);
    let common = [
        "capacity",
        "--params",
        "1",
        "--weight-bits",
        "8",
        "--gpu-vram",
        "8",
        "--layers",
        "1",
        "--kv-heads",
        "1",
        "--head-dim",
        "1",
        "--context",
        "1",
        "--runtime-overhead",
        "0",
        "--weight-overhead-percent",
        "0",
        "--artifact",
        path.to_str().unwrap(),
        "--format",
        "json",
    ];

    let fitting = run(&common);
    assert!(
        fitting.status.success(),
        "{}",
        String::from_utf8_lossy(&fitting.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&fitting.stdout).unwrap();
    assert_eq!(report["capacity_version"], 3);
    assert_eq!(report["artifact"]["source_artifact_version"], 1);
    assert_eq!(report["artifact"]["serialized_tensor_bytes"], payload_bytes);
    assert_eq!(report["artifact"]["selected_as_base_weight_floor"], true);
    assert_eq!(
        report["weights"]["base_weight_basis"],
        "artifact_residency_floor"
    );
    assert_eq!(report["fits"], true);
    let encoded = String::from_utf8(fitting.stdout).unwrap();
    assert!(!encoded.contains("private-capacity-checkpoint"));
    assert!(!encoded.contains("private.large"));
    assert!(!encoded.contains("must-not-leak"));

    let mut strengthened_args = common.to_vec();
    strengthened_args.splice(
        strengthened_args.len() - 2..strengthened_args.len() - 2,
        ["--artifact-residency-multiplier", "2"],
    );
    let strengthened = run(&strengthened_args);
    assert_eq!(strengthened.status.code(), Some(2));
    let report: serde_json::Value = serde_json::from_slice(&strengthened.stdout).unwrap();
    assert_eq!(report["artifact"]["residency_multiplier"], 2.0);
    assert_eq!(report["fits"], false);

    let help = run(&["capacity", "--help"]);
    let help = String::from_utf8(help.stdout).unwrap();
    assert!(help.contains("--artifact <PATH>"));
    assert!(help.contains("--artifact-residency-multiplier"));

    let missing_artifact = run(&[
        "capacity",
        "--params",
        "1",
        "--gpu-vram",
        "8",
        "--layers",
        "1",
        "--kv-heads",
        "1",
        "--head-dim",
        "1",
        "--artifact-residency-multiplier",
        "2",
    ]);
    assert_eq!(missing_artifact.status.code(), Some(1));
    assert!(missing_artifact.stdout.is_empty());
    assert!(String::from_utf8_lossy(&missing_artifact.stderr).contains("--artifact"));
}

#[test]
fn artifact_inspect_is_path_free_versioned_and_available_without_gpu_hardware() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("private-checkpoint.safetensors");
    write_test_safetensors(&path);

    let text = run(&["artifact", "inspect", path.to_str().unwrap()]);
    assert!(
        text.status.success(),
        "{}",
        String::from_utf8_lossy(&text.stderr)
    );
    assert!(text.stderr.is_empty());
    let text = String::from_utf8(text.stdout).unwrap();
    assert!(text.contains("artifact report v1"));
    assert!(text.contains("BF16"));
    assert!(!text.contains("private-checkpoint"));
    assert!(!text.contains("private.layer"));
    assert!(!text.contains("must-not-leak"));

    let json = run(&[
        "artifact",
        "inspect",
        path.to_str().unwrap(),
        "--format",
        "json",
    ]);
    assert!(json.status.success());
    let report: serde_json::Value = serde_json::from_slice(&json.stdout).unwrap();
    assert_eq!(report["artifact_version"], 1);
    assert_eq!(report["artifact_format"], "safetensors");
    assert_eq!(report["layout"], "single_file");
    assert_eq!(report["summary"]["tensor_count"], 1);
    assert_eq!(report["summary"]["serialized_tensor_bytes"], 8);
    assert_eq!(report["dtypes"][0]["dtype"], "BF16");
    assert_eq!(
        report["verification"]["tensor_payload_contents_read"],
        false
    );
    let encoded = String::from_utf8(json.stdout).unwrap();
    assert!(!encoded.contains("private-checkpoint"));
    assert!(!encoded.contains("private.layer"));
    assert!(!encoded.contains("must-not-leak"));

    let ndjson = run(&[
        "artifact",
        "inspect",
        path.to_str().unwrap(),
        "--format",
        "ndjson",
    ]);
    let report = assert_compact_ndjson(&ndjson);
    assert_eq!(report["artifact_version"], 1);

    let help = run(&["artifact", "inspect", "--help"]);
    assert!(help.status.success());
    let help = String::from_utf8(help.stdout).unwrap();
    assert!(help.contains("Artifact Report v1"));
    assert!(help.contains("--format"));
}

#[test]
fn artifact_inspect_rejects_malformed_metadata_without_echoing_it() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("malformed.safetensors");
    let secret = "PRIVATE_HEADER_CONTENT_MUST_NOT_LEAK";
    let header = format!("{{\"{secret}\":");
    let mut artifact = Vec::new();
    artifact.extend_from_slice(&u64::try_from(header.len()).unwrap().to_le_bytes());
    artifact.extend_from_slice(header.as_bytes());
    fs::write(&path, artifact).unwrap();

    let output = run(&["artifact", "inspect", path.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("valid bounded JSON"));
    assert!(!error.contains(secret));

    let capacity = run(&[
        "capacity",
        "--params",
        "1",
        "--gpu-vram",
        "8",
        "--layers",
        "1",
        "--kv-heads",
        "1",
        "--head-dim",
        "1",
        "--artifact",
        path.to_str().unwrap(),
    ]);
    assert_eq!(capacity.status.code(), Some(1));
    assert!(capacity.stdout.is_empty());
    let error = String::from_utf8(capacity.stderr).unwrap();
    assert!(error.contains("valid bounded JSON"));
    assert!(!error.contains(secret));
}

#[test]
fn runtime_fingerprint_is_versioned_private_and_uses_completeness_exit_status() {
    let pid = std::process::id().to_string();
    let output = run(&["runtime", "inspect", "--pid", &pid, "--format", "json"]);
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
            panic!(
                "runtime JSON must be emitted before completeness gating: {error}; stderr={}",
                String::from_utf8_lossy(&output.stderr)
            )
        });
    assert_eq!(report["runtime_fingerprint_version"], 1);
    assert_eq!(report["target_count"], 1);
    assert_eq!(report["processes"][0]["pid"], std::process::id());
    assert_eq!(report["assessment"]["compatibility"], "not_evaluated");
    assert_eq!(
        output.status.code(),
        Some(if report["complete"] == true { 0 } else { 2 })
    );

    let encoded = String::from_utf8(output.stdout).unwrap();
    for forbidden in [
        "hostname",
        "api_key",
        "HF_TOKEN",
        "/private/",
        "/proc/",
        "raw_command",
        "model_identity",
    ] {
        assert!(
            !encoded.contains(forbidden),
            "runtime report leaked forbidden token {forbidden}"
        );
    }

    let allowed = run(&[
        "runtime",
        "inspect",
        "--pid",
        &pid,
        "--allow-incomplete",
        "--format",
        "ndjson",
    ]);
    let allowed = assert_compact_ndjson(&allowed);
    assert_eq!(allowed["runtime_fingerprint_version"], 1);
    assert_eq!(allowed["assessment"]["compatibility"], "not_evaluated");

    let duplicate = run(&["runtime", "inspect", "--pid", &pid, "--pid", &pid]);
    assert_eq!(duplicate.status.code(), Some(1));
    assert!(duplicate.stdout.is_empty());
    assert!(String::from_utf8_lossy(&duplicate.stderr).contains("duplicate"));

    let zero = run(&["runtime", "inspect", "--pid", "0"]);
    assert_eq!(zero.status.code(), Some(1));
    assert!(zero.stdout.is_empty());

    let mut excessive = Command::new(env!("CARGO_BIN_EXE_gpu-watchman"));
    excessive.args(["runtime", "inspect"]);
    for value in 1..=33 {
        excessive.arg("--pid").arg(value.to_string());
    }
    let excessive = excessive.output().unwrap();
    assert_eq!(excessive.status.code(), Some(1));
    assert!(excessive.stdout.is_empty());
    assert!(String::from_utf8_lossy(&excessive.stderr).contains("at most 32"));

    let help = run(&["runtime", "inspect", "--help"]);
    assert!(help.status.success());
    let help = String::from_utf8(help.stdout).unwrap();
    assert!(help.contains("--pid <PID>"));
    assert!(help.contains("--allow-incomplete"));
    assert!(help.contains("Runtime Fingerprint v1"));
}

#[test]
fn runtime_fingerprint_never_executes_external_discovery_tools() {
    let (directory, _nvidia_smi, marker) = invocation_marking_nvidia_smi();
    let mut markers = vec![marker];
    for name in [
        "python",
        "python3",
        "pip",
        "pip3",
        "ldconfig",
        "vllm",
        "text-generation-launcher",
        "tritonserver",
        "sglang",
        "trtllm-serve",
    ] {
        let path = directory.path().join(name);
        fs::write(
            &path,
            r#"#!/bin/sh
: > "$0.invoked"
exit 97
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).unwrap();
        markers.push(path.with_extension("invoked"));
    }
    let pid = std::process::id().to_string();
    let output = Command::new(env!("CARGO_BIN_EXE_gpu-watchman"))
        .args([
            "runtime",
            "inspect",
            "--pid",
            &pid,
            "--allow-incomplete",
            "--format",
            "json",
        ])
        .env("PATH", directory.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    for marker in markers {
        assert!(
            !marker.exists(),
            "runtime inspect must not execute external discovery tools"
        );
    }
}

#[test]
fn compare_command_can_gate_a_regression() {
    let directory = tempfile::tempdir().unwrap();
    let baseline_path = directory.path().join("baseline.json");
    let current_path = directory.path().join("current.json");
    let baseline = Report::default();
    let mut current = Report::default();
    current.findings.push(Finding::new(
        None,
        Severity::Critical,
        "runtime-regression",
        "the runtime regressed",
    ));
    fs::write(&baseline_path, serde_json::to_vec(&baseline).unwrap()).unwrap();
    fs::write(&current_path, serde_json::to_vec(&current).unwrap()).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_gpu-watchman"))
        .args([
            "compare",
            baseline_path.to_str().unwrap(),
            current_path.to_str().unwrap(),
            "--fail-on-regression",
            "--format",
            "json",
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    let comparison: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(comparison["regression"], true);
    assert_eq!(comparison["new_findings"][0]["code"], "runtime-regression");
}

#[test]
fn machine_workflows_emit_compact_parseable_ndjson() {
    let capacity = run(&[
        "capacity",
        "--params",
        "7",
        "--gpu-vram",
        "24",
        "--layers",
        "32",
        "--kv-heads",
        "8",
        "--head-dim",
        "128",
        "--weight-bits",
        "4",
        "--format",
        "ndjson",
    ]);
    let capacity = assert_compact_ndjson(&capacity);
    assert_eq!(capacity["capacity_version"], 3);
    assert_eq!(capacity["input"]["parameter_source"], "explicit_override");
    assert_eq!(capacity["topology"]["world_size"], 1);
    assert!(capacity["fits"].is_boolean());

    let (_driver_directory, nvidia_smi) = fake_nvidia_smi();
    let doctor = run(&[
        "doctor",
        "--nvidia-smi",
        nvidia_smi.to_str().unwrap(),
        "--format",
        "ndjson",
    ]);
    let doctor = assert_compact_ndjson(&doctor);
    assert!(doctor.as_array().is_some_and(|checks| !checks.is_empty()));

    let report_directory = tempfile::tempdir().unwrap();
    let history_path = report_directory.path().join("history.ndjson");
    let report = Report::default();
    fs::write(
        &history_path,
        format!("{}\n", serde_json::to_string(&report).unwrap()),
    )
    .unwrap();
    let history = run(&[
        "history",
        history_path.to_str().unwrap(),
        "--format",
        "ndjson",
    ]);
    let history = assert_compact_ndjson(&history);
    assert_eq!(history["records"], 1);

    let baseline_path = report_directory.path().join("baseline.json");
    let current_path = report_directory.path().join("current.json");
    let encoded_report = serde_json::to_vec(&report).unwrap();
    fs::write(&baseline_path, &encoded_report).unwrap();
    fs::write(&current_path, encoded_report).unwrap();
    let comparison = run(&[
        "compare",
        baseline_path.to_str().unwrap(),
        current_path.to_str().unwrap(),
        "--format",
        "ndjson",
    ]);
    let comparison = assert_compact_ndjson(&comparison);
    assert_eq!(comparison["comparison_version"], 2);
    assert_eq!(comparison["regression"], false);
}

#[test]
fn capacity_rejects_non_finite_and_non_positive_inputs_with_usage_exit() {
    let invalid_cases = [
        (
            "non-finite parameters",
            "NaN",
            "4",
            "24",
            "128",
            "1",
            "0.9",
            "4",
        ),
        ("infinite VRAM", "7", "4", "inf", "128", "1", "0.9", "4"),
        (
            "zero weight precision",
            "7",
            "0",
            "24",
            "128",
            "1",
            "0.9",
            "4",
        ),
        ("zero head dimension", "7", "4", "24", "0", "1", "0.9", "4"),
        ("zero concurrency", "7", "4", "24", "128", "0", "0.9", "4"),
        (
            "utilization above one",
            "7",
            "4",
            "24",
            "128",
            "1",
            "1.01",
            "4",
        ),
        (
            "non-finite overhead",
            "7",
            "4",
            "24",
            "128",
            "1",
            "0.9",
            "NaN",
        ),
    ];

    for (name, params, weight_bits, gpu_vram, head_dim, concurrency, utilization, overhead) in
        invalid_cases
    {
        let output = run(&[
            "capacity",
            "--params",
            params,
            "--weight-bits",
            weight_bits,
            "--gpu-vram",
            gpu_vram,
            "--layers",
            "32",
            "--kv-heads",
            "8",
            "--head-dim",
            head_dim,
            "--concurrency",
            concurrency,
            "--utilization",
            utilization,
            "--runtime-overhead",
            overhead,
        ]);
        assert_eq!(
            output.status.code(),
            Some(1),
            "{name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty(), "{name}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("error: invalid value"),
            "{name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn completions_treat_a_closed_output_pipe_as_success() {
    let (reader, writer) = UnixStream::pair().unwrap();
    drop(reader);
    let writer: OwnedFd = writer.into();
    let output = Command::new(env!("CARGO_BIN_EXE_gpu-watchman"))
        .args(["completions", "bash"])
        .stdout(Stdio::from(writer))
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panicked"));
    assert!(!stderr.contains("Broken pipe"));
}

#[test]
fn capacity_loads_hugging_face_model_geometry() {
    let directory = tempfile::tempdir().unwrap();
    let config = directory.path().join("config.json");
    fs::write(
        &config,
        r#"{
          "model_type": "llama",
          "num_parameters": 7000000000,
          "num_hidden_layers": 32,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "hidden_size": 4096
        }"#,
    )
    .unwrap();

    let missing_expert_weight = run(&[
        "capacity",
        "--model-config",
        config.to_str().unwrap(),
        "--params",
        "46.7",
        "--gpu-vram",
        "80",
        "--weight-bits",
        "4",
        "--tp",
        "2",
        "--ep",
        "2",
        "--gpus",
        "2",
    ]);
    assert_eq!(missing_expert_weight.status.code(), Some(1));
    assert!(missing_expert_weight.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&missing_expert_weight.stderr)
            .contains("expert_count and expert_weight_percent")
    );

    let output = run(&[
        "capacity",
        "--model-config",
        config.to_str().unwrap(),
        "--gpu-vram",
        "24",
        "--weight-bits",
        "4",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("7.000B parameters"));
    assert!(stdout.contains("model_type=llama"));
    assert!(stdout.contains("Parameter count comes from config.json"));
}

#[test]
fn capacity_data_parallelism_never_creates_a_false_fit() {
    let output = run(&[
        "capacity",
        "--params",
        "70",
        "--weight-bits",
        "4",
        "--weight-overhead-percent",
        "0",
        "--gpu-vram",
        "24",
        "--layers",
        "8",
        "--kv-heads",
        "8",
        "--head-dim",
        "64",
        "--context",
        "1024",
        "--concurrency",
        "8",
        "--tp",
        "1",
        "--dp",
        "8",
        "--gpus",
        "8",
        "--format",
        "json",
    ]);

    assert_eq!(output.status.code(), Some(2));
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["capacity_version"], 3);
    assert_eq!(report["topology"]["data_parallel_size"], 8);
    assert_eq!(report["topology"]["world_size"], 8);
    assert_eq!(report["fits"], false);
    assert!(
        report["weights"]["weight_memory_gib_worst_rank"]
            .as_f64()
            .unwrap()
            > 32.0
    );
}

#[test]
fn capacity_rejects_an_expected_gpu_world_size_mismatch() {
    let output = run(&[
        "capacity",
        "--params",
        "7",
        "--gpu-vram",
        "80",
        "--layers",
        "8",
        "--kv-heads",
        "8",
        "--head-dim",
        "64",
        "--gpus",
        "8",
        "--tp",
        "2",
        "--pp",
        "2",
        "--dp",
        "3",
    ]);

    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("expected GPU count (8)"), "{stderr}");
    assert!(stderr.contains("world size (12)"), "{stderr}");
}

#[test]
fn capacity_reports_kv_head_replication_when_tp_exceeds_kv_heads() {
    let conservative = run(&[
        "capacity",
        "--params",
        "7",
        "--weight-bits",
        "4",
        "--gpu-vram",
        "80",
        "--layers",
        "8",
        "--kv-heads",
        "8",
        "--head-dim",
        "64",
        "--context",
        "1024",
        "--concurrency",
        "32",
        "--tp",
        "16",
        "--gpus",
        "16",
        "--format",
        "json",
    ]);
    assert!(
        conservative.status.success(),
        "{}",
        String::from_utf8_lossy(&conservative.stderr)
    );
    let conservative: serde_json::Value = serde_json::from_slice(&conservative.stdout).unwrap();
    assert_eq!(
        conservative["topology"]["kv_heads_per_tensor_parallel_rank_upper_bound"],
        8
    );
    assert_eq!(
        conservative["topology"]["kv_head_replication_upper_bound"],
        16.0
    );

    let output = run(&[
        "capacity",
        "--params",
        "7",
        "--weight-bits",
        "4",
        "--gpu-vram",
        "80",
        "--layers",
        "8",
        "--kv-heads",
        "8",
        "--head-dim",
        "64",
        "--context",
        "1024",
        "--concurrency",
        "32",
        "--tp",
        "16",
        "--max-kv-heads-per-rank",
        "1",
        "--gpus",
        "16",
        "--format",
        "json",
    ]);

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        report["topology"]["kv_heads_per_tensor_parallel_rank_upper_bound"],
        1
    );
    assert_eq!(report["topology"]["kv_head_replication_upper_bound"], 2.0);
}

#[test]
fn capacity_parameter_override_unlocks_moe_config_with_provenance() {
    let directory = tempfile::tempdir().unwrap();
    let config = directory.path().join("config.json");
    fs::write(
        &config,
        r#"{
          "model_type": "mixtral",
          "num_hidden_layers": 32,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "hidden_size": 4096,
          "num_local_experts": 8
        }"#,
    )
    .unwrap();

    let output = run(&[
        "capacity",
        "--model-config",
        config.to_str().unwrap(),
        "--params",
        "46.7",
        "--gpu-vram",
        "80",
        "--weight-bits",
        "4",
        "--tp",
        "2",
        "--ep",
        "2",
        "--gpus",
        "2",
        "--expert-weight-percent",
        "75",
        "--format",
        "json",
    ]);

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["input"]["parameter_source"], "explicit_override");
    assert_eq!(report["input"]["model_type"], "mixtral");
    assert_eq!(report["input"]["parameters_billion"], 46.7);
    assert_eq!(report["topology"]["expert_parallel_size"], 2);
}

#[test]
fn capacity_legacy_gpus_maps_to_tp_and_records_the_assumption() {
    let output = run(&[
        "capacity",
        "--params",
        "7",
        "--weight-bits",
        "4",
        "--gpu-vram",
        "24",
        "--layers",
        "32",
        "--kv-heads",
        "8",
        "--head-dim",
        "128",
        "--gpus",
        "2",
        "--format",
        "json",
    ]);

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["topology"]["tensor_parallel_size"], 2);
    assert_eq!(report["topology"]["world_size"], 2);
    assert_eq!(report["topology"]["expected_gpu_count"], 2);
    assert!(
        report["assumptions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| { item["code"] == "legacy_gpus_interpreted_as_tp" })
    );
}

#[test]
fn capacity_geometry_overrides_drive_dense_parameter_derivation() {
    let directory = tempfile::tempdir().unwrap();
    let config = directory.path().join("config.json");
    fs::write(
        &config,
        r#"{
          "model_type": "mistral",
          "num_hidden_layers": 2,
          "num_attention_heads": 8,
          "num_key_value_heads": 2,
          "hidden_size": 1024,
          "intermediate_size": 4096,
          "vocab_size": 32000,
          "tie_word_embeddings": true
        }"#,
    )
    .unwrap();

    let output = run(&[
        "capacity",
        "--model-config",
        config.to_str().unwrap(),
        "--layers",
        "4",
        "--kv-heads",
        "4",
        "--head-dim",
        "64",
        "--gpu-vram",
        "24",
        "--weight-bits",
        "4",
        "--format",
        "json",
    ]);

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["input"]["parameter_source"], "dense_estimate");
    assert_eq!(report["input"]["layers"], 4);
    assert_eq!(report["input"]["kv_heads"], 4);
    assert_eq!(report["input"]["head_dim"], 64);
}

#[test]
fn capacity_tp_sharding_credit_requires_an_explicit_rank_bound() {
    let base = [
        "capacity",
        "--params",
        "70",
        "--weight-bits",
        "4",
        "--weight-overhead-percent",
        "0",
        "--runtime-overhead",
        "0",
        "--gpu-vram",
        "24",
        "--layers",
        "8",
        "--kv-heads",
        "8",
        "--head-dim",
        "64",
        "--context",
        "1024",
        "--tp",
        "2",
        "--gpus",
        "2",
        "--format",
        "json",
    ];
    let conservative = run(&base);
    assert_eq!(conservative.status.code(), Some(2));
    let conservative: serde_json::Value = serde_json::from_slice(&conservative.stdout).unwrap();
    assert!(
        conservative["weights"]["shared_weight_memory_gib_worst_rank"]
            .as_f64()
            .unwrap()
            > 32.0
    );

    let mut bounded = base.to_vec();
    bounded.splice(
        bounded.len() - 2..bounded.len() - 2,
        ["--max-shared-rank-weight-percent", "50"],
    );
    let bounded = run(&bounded);
    assert!(
        bounded.status.success(),
        "{}",
        String::from_utf8_lossy(&bounded.stderr)
    );
    let bounded: serde_json::Value = serde_json::from_slice(&bounded.stdout).unwrap();
    let worst = bounded["weights"]["shared_weight_memory_gib_worst_rank"]
        .as_f64()
        .unwrap();
    assert!((16.0..17.0).contains(&worst));
    assert_eq!(
        bounded["topology"]["worst_shared_rank_weight_percent"],
        50.0
    );
}

#[test]
fn capacity_pp_credit_requires_independent_weight_and_layer_bounds() {
    let base = [
        "capacity",
        "--params",
        "70",
        "--weight-bits",
        "4",
        "--weight-overhead-percent",
        "0",
        "--runtime-overhead",
        "0",
        "--gpu-vram",
        "24",
        "--layers",
        "8",
        "--kv-heads",
        "8",
        "--head-dim",
        "64",
        "--context",
        "1024",
        "--pp",
        "2",
        "--gpus",
        "2",
        "--format",
        "json",
    ];
    let conservative = run(&base);
    assert_eq!(conservative.status.code(), Some(2));
    let conservative: serde_json::Value = serde_json::from_slice(&conservative.stdout).unwrap();
    assert_eq!(conservative["topology"]["worst_pipeline_stage_layers"], 8);
    assert_eq!(
        conservative["topology"]["worst_pipeline_stage_component_weight_percent"],
        100.0
    );

    let mut bounded = base.to_vec();
    bounded.splice(
        bounded.len() - 2..bounded.len() - 2,
        [
            "--max-stage-component-weight-percent",
            "50",
            "--max-stage-layers",
            "4",
        ],
    );
    let bounded = run(&bounded);
    assert!(
        bounded.status.success(),
        "{}",
        String::from_utf8_lossy(&bounded.stderr)
    );
    let bounded: serde_json::Value = serde_json::from_slice(&bounded.stdout).unwrap();
    assert_eq!(bounded["topology"]["worst_pipeline_stage_layers"], 4);
    assert_eq!(
        bounded["topology"]["worst_pipeline_stage_component_weight_percent"],
        50.0
    );
}

#[test]
fn capacity_model_config_machine_output_does_not_leak_path_or_unknown_fields() {
    let directory = tempfile::Builder::new()
        .prefix("do-not-leak-config-path-token")
        .tempdir()
        .unwrap();
    let config = directory.path().join("config.json");
    fs::write(
        &config,
        r#"{
          "model_type": "llama",
          "num_parameters": 7000000000,
          "num_hidden_layers": 32,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "hidden_size": 4096,
          "private_token": "do-not-leak-config-secret"
        }"#,
    )
    .unwrap();

    let output = run(&[
        "capacity",
        "--model-config",
        config.to_str().unwrap(),
        "--gpu-vram",
        "24",
        "--weight-bits",
        "4",
        "--format",
        "json",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(!stdout.contains("do-not-leak-config-path-token"));
    assert!(!stdout.contains("do-not-leak-config-secret"));
}

#[test]
fn required_telemetry_source_fails_closed_with_report_evidence() {
    let (_directory, nvidia_smi) = fake_nvidia_smi();
    let output = run(&[
        "snapshot",
        "--nvidia-smi",
        nvidia_smi.to_str().unwrap(),
        "--format",
        "json",
        "--no-xid",
        "--require-source",
        "xid",
    ]);
    assert_eq!(output.status.code(), Some(2));
    let report: Report = serde_json::from_slice(&output.stdout).unwrap();
    let xid = report
        .sources
        .iter()
        .find(|source| source.name == "kernel.xid")
        .unwrap();
    assert!(xid.required);
    assert_eq!(xid.state, gpu_watchman::domain::SourceState::Skipped);
    assert!(
        report
            .findings
            .iter()
            .any(|finding| finding.code == "telemetry-source-required")
    );
}

#[test]
fn color_is_explicit_and_machine_output_remains_ansi_free() {
    let (_directory, nvidia_smi) = fake_nvidia_smi();
    let driver = nvidia_smi.to_str().unwrap();
    let text = run(&[
        "snapshot",
        "--nvidia-smi",
        driver,
        "--all",
        "--no-xid",
        "--color",
        "always",
    ]);
    assert!(text.status.success());
    assert!(text.stdout.windows(2).any(|window| window == b"\x1b["));

    let json = run(&[
        "snapshot",
        "--nvidia-smi",
        driver,
        "--no-xid",
        "--format",
        "json",
    ]);
    assert!(json.status.success());
    assert!(!json.stdout.windows(2).any(|window| window == b"\x1b["));
    serde_json::from_slice::<Report>(&json.stdout).unwrap();
}

#[test]
fn ps_view_surfaces_gpu_process_ownership_without_a_full_dump() {
    let (_directory, nvidia_smi) = fake_nvidia_smi();
    let output = run(&[
        "ps",
        "--nvidia-smi",
        nvidia_smi.to_str().unwrap(),
        "--color",
        "never",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("GPU Watchman processes"));
    assert!(stdout.contains("4242"));
    assert!(stdout.contains("vllm"));
    assert!(!stdout.contains("GPU DETAILS"));
}

#[test]
fn ps_machine_output_uses_the_dedicated_process_contract() {
    for format in ["json", "ndjson"] {
        let (_directory, nvidia_smi) = fake_nvidia_smi();
        let output = run(&[
            "ps",
            "--nvidia-smi",
            nvidia_smi.to_str().unwrap(),
            "--format",
            format,
        ]);
        assert!(
            output.status.success(),
            "{format}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stderr.is_empty(), "{format}");

        let stdout = std::str::from_utf8(&output.stdout).unwrap();
        assert!(stdout.ends_with('\n'), "{format}");
        if format == "ndjson" {
            assert_eq!(
                stdout.matches('\n').count(),
                1,
                "NDJSON must contain one compact process record: {stdout:?}"
            );
        }
        let view: serde_json::Value = serde_json::from_str(stdout).unwrap();
        let top_level_keys = view
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            top_level_keys,
            std::collections::BTreeSet::from([
                "collected_at",
                "host",
                "process_count",
                "process_view_version",
                "processes",
                "source",
            ]),
            "{format} unexpectedly exposed fields from the full report"
        );
        assert_eq!(view["process_view_version"], 1);
        assert_eq!(view["process_count"], 1);
        assert_eq!(view["source"]["name"], "nvidia.processes");
        assert_eq!(view["source"]["complete"], true);
        assert_eq!(view["processes"][0]["gpu_index"], 0);
        assert_eq!(view["processes"][0]["gpu_uuid"], "GPU-test");
        assert_eq!(view["processes"][0]["pid"], 4242);
        assert_eq!(view["processes"][0]["name"], "vllm");
        assert_eq!(view["processes"][0]["memory_mib"], 40_000);
    }
}

#[test]
fn ps_requires_complete_process_telemetry_unless_explicitly_best_effort() {
    let directory = tempfile::tempdir().unwrap();
    let nvidia_smi = directory.path().join("nvidia-smi");
    write_process_incomplete_nvidia_smi(&nvidia_smi);
    let driver = nvidia_smi.to_str().unwrap();

    let strict = run(&["ps", "--nvidia-smi", driver, "--format", "json"]);
    assert_eq!(strict.status.code(), Some(2));
    let evidence: serde_json::Value = serde_json::from_slice(&strict.stdout).unwrap();
    assert_eq!(evidence["source"]["name"], "nvidia.processes");
    assert_eq!(evidence["source"]["complete"], false);

    let best_effort = run(&[
        "ps",
        "--nvidia-smi",
        driver,
        "--allow-incomplete",
        "--format",
        "json",
    ]);
    assert!(
        best_effort.status.success(),
        "{}",
        String::from_utf8_lossy(&best_effort.stderr)
    );
    let evidence: serde_json::Value = serde_json::from_slice(&best_effort.stdout).unwrap();
    assert_eq!(evidence["source"]["complete"], false);
}

#[test]
fn workflow_specific_flags_are_rejected_before_driver_collection() {
    let cases: &[(&str, &[&str], &str)] = &[
        ("snapshot", &["--listen", "127.0.0.1:0"], "--listen"),
        ("snapshot", &["--watch", "1s"], "--watch"),
        (
            "snapshot",
            &["--format", "json", "--color", "always"],
            "text output",
        ),
        ("top", &["--listen", "127.0.0.1:0"], "--listen"),
        ("top", &["--format", "json"], "json"),
        ("ps", &["--probe", "http://127.0.0.1:8000"], "--probe"),
        ("ps", &["--watch", "1s"], "--watch"),
        ("serve", &["--emit", "json"], "json"),
    ];

    for (workflow, invalid_args, rejected_value) in cases {
        let (_directory, nvidia_smi, marker) = invocation_marking_nvidia_smi();
        let mut args = vec![
            (*workflow).to_owned(),
            "--nvidia-smi".to_owned(),
            nvidia_smi.to_string_lossy().into_owned(),
        ];
        args.extend(invalid_args.iter().map(|value| (*value).to_owned()));

        let output = Command::new(env!("CARGO_BIN_EXE_gpu-watchman"))
            .args(&args)
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(1),
            "{workflow} accepted {invalid_args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty(), "{workflow} {invalid_args:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(rejected_value),
            "{workflow} {invalid_args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            !marker.exists(),
            "{workflow} invoked nvidia-smi before rejecting {invalid_args:?}"
        );
    }
}

#[test]
fn hardened_cli_rejects_config_and_profile_for_non_profile_aware_commands() {
    let cases: &[(&[&str], &str)] = &[
        (
            &["--config", "/definitely/missing/config.toml", "version"],
            "--config and --profile apply only",
        ),
        (
            &["--profile", "ops", "doctor"],
            "--config and --profile apply only",
        ),
        (
            &[
                "--config",
                "/definitely/missing/config.toml",
                "artifact",
                "inspect",
                "/definitely/missing/model.safetensors",
            ],
            "--config and --profile apply only",
        ),
    ];

    for (args, expected) in cases {
        let output = clean_config_command().args(*args).output().unwrap();
        assert_eq!(
            output.status.code(),
            Some(1),
            "accepted unsupported profile flags {args:?}"
        );
        assert!(output.stdout.is_empty(), "{args:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "{args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn hardened_cli_rejects_text_only_flags_in_machine_output() {
    let directory = tempfile::tempdir().unwrap();
    let history = directory.path().join("quiet-history.ndjson");
    let cases = [
        (
            "snapshot",
            vec!["--format", "json", "--all"],
            "apply only to text output",
        ),
        (
            "snapshot",
            vec!["--format", "ndjson", "--details"],
            "apply only to text output",
        ),
        (
            "snapshot",
            vec!["--format", "json", "--color", "never"],
            "apply only to text output",
        ),
        (
            "snapshot",
            vec![
                "--history",
                history.to_str().unwrap(),
                "--quiet",
                "--format",
                "json",
            ],
            "--format has no effect",
        ),
        (
            "top",
            vec!["--format", "ndjson", "--no-clear"],
            "apply only to text output",
        ),
        (
            "top",
            vec!["--format", "ndjson", "--details"],
            "apply only to text output",
        ),
        (
            "top",
            vec!["--format", "ndjson", "--color", "never"],
            "apply only to text output",
        ),
        (
            "ps",
            vec!["--format", "json", "--color", "never"],
            "--color applies only to text output",
        ),
    ];

    for (workflow, invalid_args, expected) in cases {
        let (_driver_directory, nvidia_smi, marker) = invocation_marking_nvidia_smi();
        let mut command = Command::new(env!("CARGO_BIN_EXE_gpu-watchman"));
        command
            .arg(workflow)
            .arg("--nvidia-smi")
            .arg(&nvidia_smi)
            .args(&invalid_args);
        let output = command.output().unwrap();

        assert_eq!(
            output.status.code(),
            Some(1),
            "{workflow} accepted {invalid_args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty(), "{workflow} {invalid_args:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "{workflow} {invalid_args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            !marker.exists(),
            "{workflow} invoked nvidia-smi before rejecting {invalid_args:?}"
        );
    }
    assert!(!history.exists());
}

#[test]
fn hardened_cli_rejects_probe_options_without_a_probe() {
    let cases: &[(&str, &[&str])] = &[
        ("doctor", &["--probe-token", "cli-private"]),
        (
            "doctor",
            &["--probe-token-file", "/definitely/missing/probe-token"],
        ),
        ("doctor", &["--no-probe-auth"]),
        ("doctor", &["--allow-insecure-http"]),
        ("bundle", &["--probe-token", "cli-private"]),
        (
            "bundle",
            &["--probe-token-file", "/definitely/missing/probe-token"],
        ),
        ("bundle", &["--no-probe-auth"]),
        ("bundle", &["--allow-insecure-http"]),
    ];

    for (workflow, invalid_args) in cases {
        let (directory, nvidia_smi, marker) = invocation_marking_nvidia_smi();
        let bundle_path = directory.path().join("must-not-exist.json");
        let mut command = clean_config_command();
        command
            .arg(workflow)
            .arg("--nvidia-smi")
            .arg(&nvidia_smi)
            .args(*invalid_args);
        if *workflow == "bundle" {
            command.arg("--output").arg(&bundle_path);
        }
        let output = command.output().unwrap();

        assert_eq!(
            output.status.code(),
            Some(1),
            "{workflow} accepted {invalid_args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty(), "{workflow} {invalid_args:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("probe authentication and transport options require --probe"),
            "{workflow} {invalid_args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!marker.exists(), "{workflow} collected before rejection");
        assert!(!bundle_path.exists(), "bundle was written before rejection");
        assert!(!String::from_utf8_lossy(&output.stderr).contains("cli-private"));
    }
}

#[test]
fn hardened_cli_requires_remote_service_opt_in_and_authentication() {
    let cases: &[(&[&str], &str)] = &[
        (&["--listen", "0.0.0.0:0"], "requires --allow-remote-listen"),
        (
            &["--listen", "0.0.0.0:0", "--allow-remote-listen"],
            "requires --api-token or --api-token-file",
        ),
    ];

    for (service_args, expected) in cases {
        let (_directory, nvidia_smi, marker) = invocation_marking_nvidia_smi();
        let output = Command::new(env!("CARGO_BIN_EXE_gpu-watchman"))
            .arg("serve")
            .arg("--nvidia-smi")
            .arg(&nvidia_smi)
            .args(*service_args)
            .output()
            .unwrap();

        assert_eq!(
            output.status.code(),
            Some(1),
            "accepted unsafe service args {service_args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty(), "{service_args:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "{service_args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!marker.exists(), "service collected before rejection");
    }
}

#[test]
fn config_init_is_private_and_never_overwrites_an_existing_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("gpu-watchman.toml");

    let first = clean_config_command()
        .args(["config", "init"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert!(first.stderr.is_empty());
    assert!(String::from_utf8_lossy(&first.stdout).contains("created private configuration"));
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    let original = fs::read(&path).unwrap();
    assert!(String::from_utf8_lossy(&original).contains("config_version = 1"));

    let second = clean_config_command()
        .args(["config", "init"])
        .arg(&path)
        .output()
        .unwrap();
    assert_eq!(second.status.code(), Some(1));
    assert!(second.stdout.is_empty());
    assert!(String::from_utf8_lossy(&second.stderr).contains("create configuration"));
    assert_eq!(fs::read(&path).unwrap(), original);
}

#[test]
fn config_validate_and_show_filter_and_redact_toml_and_json() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("profiles.toml");
    write_private_config(
        &path,
        r#"config_version = 1
default_profile = "hidden"

[profiles.selected.monitor.inference]
urls = ["https://metrics.example.invalid/path?access_token=query-private#fragment-private"]
token_file = "secrets/probe-token"

[profiles.selected.service]
history_file = "state/history.ndjson"
api_token_file = "secrets/service-token"

[profiles.selected.canary]
base_url = "https://canary.example.invalid/v1?api_key=canary-query-private#canary-fragment-private"
model = "selected-model"
api_key_file = "secrets/canary-key"

[profiles.hidden.canary]
base_url = "https://hidden.example.invalid/v1?hidden=hidden-query-private#hidden-fragment-private"
model = "hidden-model"
"#,
    );

    let validated = clean_config_command()
        .arg("--config")
        .arg(&path)
        .args(["--profile", "selected", "config", "validate"])
        .output()
        .unwrap();
    assert!(
        validated.status.success(),
        "{}",
        String::from_utf8_lossy(&validated.stderr)
    );
    let validation = String::from_utf8(validated.stdout).unwrap();
    assert!(validation.contains("2 profile(s)"));
    assert!(validation.contains("profile \"selected\" selected"));
    let canonical_directory = fs::canonicalize(directory.path()).unwrap();

    for format in ["toml", "json"] {
        let shown = clean_config_command()
            .arg("--config")
            .arg(&path)
            .args([
                "--profile",
                "selected",
                "config",
                "show",
                "--format",
                format,
            ])
            .output()
            .unwrap();
        assert!(
            shown.status.success(),
            "{format}: {}",
            String::from_utf8_lossy(&shown.stderr)
        );
        assert!(shown.stderr.is_empty(), "{format}");
        let stdout = String::from_utf8(shown.stdout).unwrap();
        assert!(stdout.ends_with('\n'), "{format}");
        for private in [
            "query-private",
            "fragment-private",
            "canary-query-private",
            "canary-fragment-private",
            "hidden-query-private",
            "hidden-fragment-private",
            "hidden-model",
        ] {
            assert!(!stdout.contains(private), "{format} exposed {private}");
        }

        let value = if format == "toml" {
            let value: toml::Value = toml::from_str(&stdout).unwrap();
            serde_json::to_value(value).unwrap()
        } else {
            serde_json::from_str(&stdout).unwrap()
        };
        assert_eq!(value["config_version"], 1);
        assert_eq!(value["default_profile"], "selected");
        assert_eq!(value["profiles"].as_object().unwrap().len(), 1);
        assert!(value["profiles"].get("hidden").is_none());
        assert_eq!(
            value["profiles"]["selected"]["monitor"]["inference"]["urls"][0],
            "https://metrics.example.invalid/path?REDACTED"
        );
        assert_eq!(
            value["profiles"]["selected"]["canary"]["base_url"],
            "https://canary.example.invalid/v1?REDACTED"
        );
        assert_eq!(
            value["profiles"]["selected"]["monitor"]["inference"]["token_file"],
            canonical_directory
                .join("secrets/probe-token")
                .to_string_lossy()
                .as_ref()
        );
    }
}

#[test]
fn config_rejects_userinfo_without_echoing_credentials() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("userinfo.toml");
    let private = "userinfo-password-private";
    write_private_config(
        &path,
        &format!(
            r#"config_version = 1
[profiles.bad.monitor.inference]
urls = ["https://operator:{private}@metrics.example.invalid/path?query-private#fragment-private"]
"#
        ),
    );

    let output = clean_config_command()
        .arg("--config")
        .arg(&path)
        .args(["config", "validate"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("must not include URL user information"));
    for private in [private, "query-private", "fragment-private"] {
        assert!(!stderr.contains(private));
    }
}

#[test]
fn config_rejects_unknown_fields_versions_and_unsafe_permissions() {
    let directory = tempfile::tempdir().unwrap();
    let cases = [
        (
            "version.toml",
            "config_version = 2\n",
            "unsupported config_version 2",
        ),
        (
            "unknown.toml",
            concat!(
                "config_version = 1\n",
                "[profiles.local.monitor.collection]\n",
                "unexpected_setting = true\n"
            ),
            "configuration contains an unknown field",
        ),
    ];
    for (name, body, expected_error) in cases {
        let path = directory.path().join(name);
        write_private_config(&path, body);
        let output = clean_config_command()
            .arg("--config")
            .arg(&path)
            .args(["config", "validate"])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1), "{name}");
        assert!(output.stdout.is_empty(), "{name}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected_error),
            "{name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let unsafe_path = directory.path().join("unsafe.toml");
    write_private_config(&unsafe_path, "config_version = 1\n");
    let mut permissions = fs::metadata(&unsafe_path).unwrap().permissions();
    permissions.set_mode(0o620);
    fs::set_permissions(&unsafe_path, permissions).unwrap();
    let output = clean_config_command()
        .arg("--config")
        .arg(&unsafe_path)
        .args(["config", "validate"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("must not be writable by group or other users")
    );
}

#[test]
fn config_paths_and_operational_precedence_are_deterministic() {
    let (directory, config) = operational_profile_fixture();
    let environment_driver = directory.path().join("environment-smi");
    let cli_driver = directory.path().join("cli-smi");
    write_fake_nvidia_smi(&environment_driver, "GPU-environment", "Environment GPU");
    write_fake_nvidia_smi(&cli_driver, "GPU-cli", "CLI GPU");

    let process_gpu_name = |output: Output, layer: &str| {
        assert!(
            output.status.success(),
            "{layer}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let view: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        view["processes"][0]["gpu_name"]
            .as_str()
            .unwrap()
            .to_owned()
    };

    let profile = clean_config_command()
        .arg("--config")
        .arg(&config)
        .args(["ps", "--format", "json"])
        .output()
        .unwrap();
    assert_eq!(process_gpu_name(profile, "profile"), "Profile GPU");

    let environment = clean_config_command()
        .env("GPU_WATCHMAN_NVIDIA_SMI", &environment_driver)
        .arg("--config")
        .arg(&config)
        .args(["ps", "--format", "json"])
        .output()
        .unwrap();
    assert_eq!(
        process_gpu_name(environment, "environment"),
        "Environment GPU"
    );

    let cli = clean_config_command()
        .env("GPU_WATCHMAN_NVIDIA_SMI", &environment_driver)
        .arg("--config")
        .arg(&config)
        .args(["ps", "--nvidia-smi"])
        .arg(&cli_driver)
        .args(["--format", "json"])
        .output()
        .unwrap();
    assert_eq!(process_gpu_name(cli, "CLI"), "CLI GPU");
}

#[test]
fn credential_precedence_crosses_inline_and_file_forms_without_network() {
    let (directory, config) = operational_profile_fixture();

    let missing_profile_credential = clean_config_command()
        .arg("--config")
        .arg(&config)
        .args(["snapshot", "--no-xid", "--format", "json"])
        .output()
        .unwrap();
    assert_eq!(missing_profile_credential.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&missing_profile_credential.stderr)
            .contains("missing-profile-token")
    );

    let environment_inline = "environment-inline-private";
    let environment_over_profile = clean_config_command()
        .env("GPU_WATCHMAN_PROBE_TOKEN", environment_inline)
        .arg("--config")
        .arg(&config)
        .args(["snapshot", "--no-xid", "--format", "json"])
        .output()
        .unwrap();
    assert!(
        environment_over_profile.status.success(),
        "{}",
        String::from_utf8_lossy(&environment_over_profile.stderr)
    );
    assert!(
        !String::from_utf8_lossy(&environment_over_profile.stdout).contains(environment_inline)
    );
    assert!(
        !String::from_utf8_lossy(&environment_over_profile.stderr).contains(environment_inline)
    );

    let cli_inline = "cli-inline-private";
    let cli_over_environment_file = clean_config_command()
        .env(
            "GPU_WATCHMAN_PROBE_TOKEN_FILE",
            directory.path().join("missing-environment-token"),
        )
        .arg("--config")
        .arg(&config)
        .args([
            "snapshot",
            "--probe-token",
            cli_inline,
            "--no-xid",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(
        cli_over_environment_file.status.success(),
        "{}",
        String::from_utf8_lossy(&cli_over_environment_file.stderr)
    );
    assert!(!String::from_utf8_lossy(&cli_over_environment_file.stdout).contains(cli_inline));
    assert!(!String::from_utf8_lossy(&cli_over_environment_file.stderr).contains(cli_inline));
}

#[test]
fn hardened_cli_doctor_credential_precedence_crosses_forms() {
    let (_driver_directory, nvidia_smi) = fake_nvidia_smi();
    let secrets = tempfile::tempdir().unwrap();
    let missing_environment_file = secrets.path().join("missing-environment-token");
    let cli_token = "doctor-cli-private";
    let (probe, server) =
        serve_openai_once("200 OK", "text/plain", "vllm:num_requests_running 0\n");

    let output = clean_config_command()
        .env("GPU_WATCHMAN_PROBE_TOKEN_FILE", &missing_environment_file)
        .args(["doctor", "--probe", &probe, "--probe-token", cli_token])
        .arg("--nvidia-smi")
        .arg(&nvidia_smi)
        .args(["--format", "json"])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let request = server.join().unwrap();
    assert!(
        request
            .to_ascii_lowercase()
            .contains("authorization: bearer doctor-cli-private")
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains(cli_token));
    assert!(!String::from_utf8_lossy(&output.stderr).contains(cli_token));
    assert!(
        !String::from_utf8_lossy(&output.stderr)
            .contains(missing_environment_file.to_str().unwrap())
    );
}

#[test]
fn doctor_rejects_a_reachable_endpoint_without_runtime_metrics() {
    let (_driver_directory, nvidia_smi) = fake_nvidia_smi();
    let (probe, server) = serve_openai_once("200 OK", "text/html", "<h1>healthy</h1>\n");
    let probe = probe.strip_suffix("/v1").unwrap().to_owned();

    let output = clean_config_command()
        .args(["doctor", "--probe", &probe])
        .arg("--nvidia-smi")
        .arg(&nvidia_smi)
        .args(["--format", "json"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    assert!(output.stderr.is_empty());
    let checks: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let endpoint = checks
        .as_array()
        .unwrap()
        .iter()
        .find(|check| {
            check["name"]
                .as_str()
                .is_some_and(|name| name.starts_with("inference-endpoint:"))
        })
        .unwrap();
    assert_eq!(endpoint["status"], "fail");
    assert!(endpoint["detail"].as_str().unwrap().contains("/metrics"));
    server.join().unwrap();
}

#[test]
fn unmatched_gpu_selector_fails_instead_of_emitting_a_healthy_empty_report() {
    let (_directory, nvidia_smi) = fake_nvidia_smi();
    let output = run(&[
        "snapshot",
        "--nvidia-smi",
        nvidia_smi.to_str().unwrap(),
        "--no-xid",
        "--gpu",
        "GPU-stale",
        "--format",
        "json",
    ]);

    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("GPU-stale"));
    assert!(stderr.contains("0 (GPU-test)"));
}

#[test]
fn hardened_cli_bundle_credential_precedence_and_redaction() {
    let (_driver_directory, nvidia_smi) = fake_nvidia_smi();
    let directory = tempfile::tempdir().unwrap();
    let output_path = directory.path().join("support.json");
    let token_path = directory.path().join("probe-token");
    let cli_file_token = "bundle-cli-file-private";
    let environment_token = "bundle-environment-inline-private";
    write_private_file(&token_path, &format!("{cli_file_token}\n"));
    let (probe, server) =
        serve_openai_once("200 OK", "text/plain", "vllm:num_requests_running 0\n");

    let output = clean_config_command()
        .env("GPU_WATCHMAN_PROBE_TOKEN", environment_token)
        .args(["bundle", "--output"])
        .arg(&output_path)
        .args(["--nvidia-smi"])
        .arg(&nvidia_smi)
        .args(["--probe", &probe, "--probe-token-file"])
        .arg(&token_path)
        .arg("--no-xid")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let request = server.join().unwrap();
    assert!(
        request
            .to_ascii_lowercase()
            .contains("authorization: bearer bundle-cli-file-private")
    );
    assert!(!request.contains(environment_token));
    let body = fs::read_to_string(output_path).unwrap();
    assert!(!body.contains(cli_file_token));
    assert!(!body.contains(environment_token));
    let bundle: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(bundle["report"]["schema_version"], 3);
    assert!(
        bundle["checks"]
            .as_array()
            .is_some_and(|checks| !checks.is_empty())
    );
}

#[test]
fn canary_streams_without_gpu_hardware_and_emits_compact_ndjson() {
    let stream_body = concat!(
        "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",",
        "\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},",
        "\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",",
        "\"model\":\"test-model\",\"choices\":[{\"index\":0,",
        "\"delta\":{\"content\":\"gpu-watchman-ok\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",",
        "\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{},",
        "\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":6,",
        "\"completion_tokens\":3,\"total_tokens\":9}}\n\n",
        "data: [DONE]\n\n"
    );
    let (base_url, server) = serve_openai_once("200 OK", "text/event-stream", stream_body);
    let secrets = tempfile::tempdir().unwrap();
    let api_key_path = secrets.path().join("inference-api-key");
    write_private_file(&api_key_path, "local-test-key\n");
    let output = Command::new(env!("CARGO_BIN_EXE_gpu-watchman"))
        .args([
            "canary",
            "--base-url",
            &base_url,
            "--model",
            "test-model",
            "--api-key-file",
            api_key_path.to_str().unwrap(),
            "--format",
            "ndjson",
        ])
        .env("GPU_WATCHMAN_NVIDIA_SMI", "/definitely/missing/nvidia-smi")
        .env("HTTP_PROXY", "http://127.0.0.1:9")
        .env("ALL_PROXY", "http://127.0.0.1:9")
        .env_remove("NO_PROXY")
        .env_remove("no_proxy")
        .output()
        .unwrap();
    let request = server.join().unwrap();

    let report = assert_compact_ndjson(&output);
    assert_eq!(report["canary_version"], 2);
    assert_eq!(report["workload_id"], "builtin-v1");
    assert_eq!(report["status"], "pass");
    assert_eq!(report["summary"]["succeeded"], 1);
    assert_eq!(report["target"]["stream"], true);
    assert_eq!(report["attempts"][0]["expectation_met"], true);
    assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"));
    assert!(
        request
            .to_ascii_lowercase()
            .contains("authorization: bearer local-test-key")
    );
    let request_body: serde_json::Value =
        serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(request_body["stream"], true);
    assert_eq!(request_body["stream_options"]["include_usage"], true);
    assert_eq!(
        request_body["messages"][0]["content"],
        "Reply with exactly: gpu-watchman-ok"
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains("local-test-key"));
    assert!(!String::from_utf8_lossy(&output.stderr).contains("nvidia-smi"));
}

#[test]
fn canary_custom_prompt_has_no_implicit_built_in_expectation() {
    let response = concat!(
        "{\"id\":\"chatcmpl-custom\",\"object\":\"chat.completion\",",
        "\"model\":\"response-model-private\",\"choices\":[{\"index\":0,",
        "\"message\":{\"role\":\"assistant\",\"content\":\"a custom response\"},",
        "\"finish_reason\":\"finish-private\"}],\"usage\":{\"prompt_tokens\":3,",
        "\"completion_tokens\":3,\"total_tokens\":6}}"
    );
    let (base_url, server) = serve_openai_once("200 OK", "application/json", response);
    let prompts = tempfile::tempdir().unwrap();
    let prompt_path = prompts.path().join("prompt.txt");
    fs::write(&prompt_path, "perform a custom health check").unwrap();
    let output = run(&[
        "canary",
        "--base-url",
        &base_url,
        "--model",
        "test-model",
        "--prompt-file",
        prompt_path.to_str().unwrap(),
        "--workload-id",
        "custom-health-v1",
        "--no-stream",
        "--format",
        "json",
    ]);
    let request = server.join().unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["status"], "pass");
    assert_eq!(report["workload_id"], "custom-health-v1");
    assert!(report["attempts"][0].get("expectation_met").is_none());
    let request_body: serde_json::Value =
        serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(request_body["stream"], false);
    assert!(request_body.get("stream_options").is_none());
    assert_eq!(
        request_body["messages"][0]["content"],
        "perform a custom health check"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.contains("perform a custom health check"));
    assert!(!stdout.contains("a custom response"));
    assert!(!stdout.contains("response-model-private"));
    assert!(!stdout.contains("finish-private"));
    assert_eq!(report["target"]["model"], "test-model");
    assert!(report["attempts"][0].get("model").is_none());
    assert!(report["attempts"][0].get("finish_reason").is_none());
}

#[test]
fn canary_custom_prompt_requires_a_privacy_safe_workload_identity() {
    let private_prompt = "private-custom-prompt-that-must-not-be-echoed";
    let output = run(&[
        "canary",
        "--model",
        "test-model",
        "--prompt",
        private_prompt,
    ]);

    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("requires an explicit --workload-id"));
    assert!(!stderr.contains(private_prompt));

    let invalid = run(&[
        "canary",
        "--model",
        "test-model",
        "--prompt",
        "synthetic",
        "--workload-id",
        "contains whitespace",
    ]);
    assert_eq!(invalid.status.code(), Some(1));
}

#[test]
fn saturation_benchmark_emits_bounded_goodput_evidence_and_passes_an_exact_stage_gate() {
    let body = concat!(
        "{\"id\":\"chatcmpl-benchmark\",\"object\":\"chat.completion\",",
        "\"model\":\"response-model-private\",\"choices\":[{\"index\":0,",
        "\"message\":{\"role\":\"assistant\",\"content\":\"gpu-watchman-ok\"},",
        "\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":6,",
        "\"completion_tokens\":3,\"total_tokens\":9}}"
    );
    let (base_url, server) = serve_openai_many(2, "200 OK", "application/json", body);
    let output = run(&[
        "benchmark",
        "saturation",
        "--base-url",
        &base_url,
        "--model",
        "test-model",
        "--concurrency-stages",
        "1",
        "--warmup-requests-per-worker",
        "1",
        "--requests-per-worker",
        "1",
        "--max-tokens",
        "3",
        "--no-stream",
        "--verify-concurrency",
        "1",
        "--max-error-percent",
        "0",
        "--min-successful-requests-per-second",
        "0",
        "--min-completion-token-goodput-per-second",
        "0",
        "--format",
        "ndjson",
    ]);
    let requests = server.join().unwrap();

    let report = assert_compact_ndjson(&output);
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .all(|request| request.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"))
    );
    assert_eq!(report["saturation_benchmark_version"], 1);
    assert_eq!(report["status"], "complete");
    assert!(report["duration_ns"].as_u64().unwrap() > 0);
    assert_eq!(report["plan"]["planned_attempts"], 2);
    assert_eq!(report["plan"]["timeout_ns"], 10_000_000_000_u64);
    assert_eq!(report["warmups"][0]["succeeded"], 1);
    assert!(report["warmups"][0]["duration_ns"].as_u64().unwrap() > 0);
    assert_eq!(report["stages"][0]["summary"]["succeeded"], 1);
    assert!(report["stages"][0]["duration_ns"].as_u64().unwrap() > 0);
    assert_eq!(
        report["stages"][0]["summary"]["completion_token_usage_complete"],
        true
    );
    assert_eq!(
        report["stages"][0]["summary"]["completion_tokens_observed_total"],
        3
    );
    assert_eq!(report["verification"]["status"], "pass");
    assert_eq!(report["assessment"]["status"], "not_evaluable");
    assert!(report["stages"][0]["attempts"][0].get("model").is_none());
    assert!(
        report["stages"][0]["attempts"][0]
            .get("finish_reason")
            .is_none()
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains("response-model-private"));
}

#[test]
fn saturation_benchmark_warms_and_measures_every_explicit_stage() {
    let body = concat!(
        "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",",
        "\"content\":\"gpu-watchman-ok\"},\"finish_reason\":\"stop\"}],",
        "\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}"
    );
    let (base_url, server) = serve_openai_many(6, "200 OK", "application/json", body);
    let output = run(&[
        "benchmark",
        "saturation",
        "--base-url",
        &base_url,
        "--model",
        "test-model",
        "--concurrency-stages",
        "1,2",
        "--warmup-requests-per-worker",
        "1",
        "--requests-per-worker",
        "1",
        "--no-stream",
        "--verify-concurrency",
        "2",
        "--format",
        "json",
    ]);
    let requests = server.join().unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(requests.len(), 6);
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["plan"]["planned_attempts"], 6);
    assert_eq!(report["warmups"].as_array().unwrap().len(), 2);
    assert_eq!(report["warmups"][0]["concurrency"], 1);
    assert_eq!(report["warmups"][0]["planned_requests"], 1);
    assert_eq!(report["warmups"][1]["concurrency"], 2);
    assert_eq!(report["warmups"][1]["planned_requests"], 2);
    assert_eq!(report["stages"].as_array().unwrap().len(), 2);
    assert_eq!(report["stages"][0]["planned_requests"], 1);
    assert_eq!(report["stages"][1]["planned_requests"], 2);
    assert_eq!(report["verification"]["status"], "pass");
}

#[test]
fn saturation_benchmark_second_stage_warmup_abort_prevents_its_measurement() {
    let success = concat!(
        "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",",
        "\"content\":\"gpu-watchman-ok\"},\"finish_reason\":\"stop\"}],",
        "\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}"
    );
    let failure = "{\"error\":{\"message\":\"private-stage-two-failure\"}}";
    let responses = vec![
        ("200 OK", "application/json", success),
        ("200 OK", "application/json", success),
        ("503 Service Unavailable", "application/json", failure),
        ("503 Service Unavailable", "application/json", failure),
    ];
    let (base_url, server) = serve_openai_sequence(responses);
    let output = run(&[
        "benchmark",
        "saturation",
        "--base-url",
        &base_url,
        "--model",
        "test-model",
        "--concurrency-stages",
        "1,2",
        "--warmup-requests-per-worker",
        "1",
        "--requests-per-worker",
        "1",
        "--no-stream",
        "--verify-concurrency",
        "2",
        "--format",
        "json",
    ]);
    let requests = server.join().unwrap();

    assert_eq!(output.status.code(), Some(2));
    assert_eq!(requests.len(), 4);
    let stdout = String::from_utf8(output.stdout).unwrap();
    let report: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(report["status"], "aborted");
    assert_eq!(report["abort_reason"], "warmup_no_successful_requests");
    assert_eq!(report["warmups"].as_array().unwrap().len(), 2);
    assert_eq!(report["warmups"][0]["status"], "complete");
    assert_eq!(report["warmups"][1]["status"], "aborted");
    assert_eq!(report["warmups"][1]["concurrency"], 2);
    assert_eq!(report["stages"].as_array().unwrap().len(), 1);
    assert_eq!(report["stages"][0]["concurrency"], 1);
    assert_eq!(report["verification"]["status"], "not_evaluable");
    assert_eq!(report["verification"]["reason"], "stage_not_run");
    assert!(!stdout.contains("private-stage-two-failure"));
}

#[test]
fn saturation_benchmark_percentile_gate_fails_closed_with_too_few_samples() {
    let body = concat!(
        "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",",
        "\"content\":\"gpu-watchman-ok\"},\"finish_reason\":\"stop\"}],",
        "\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}"
    );
    let (base_url, server) = serve_openai_many(2, "200 OK", "application/json", body);
    let output = run(&[
        "benchmark",
        "saturation",
        "--base-url",
        &base_url,
        "--model",
        "test-model",
        "--concurrency-stages",
        "1",
        "--warmup-requests-per-worker",
        "1",
        "--requests-per-worker",
        "1",
        "--no-stream",
        "--verify-concurrency",
        "1",
        "--max-p95-e2e",
        "1s",
        "--format",
        "json",
    ]);
    let _requests = server.join().unwrap();

    assert_eq!(output.status.code(), Some(2));
    assert!(output.stderr.is_empty());
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["status"], "complete");
    assert_eq!(report["verification"]["status"], "not_evaluable");
    let gate = report["stages"][0]["gates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|gate| gate["kind"] == "p95_e2e_ms")
        .unwrap();
    assert_eq!(gate["status"], "not_evaluable");
    assert_eq!(gate["reason"], "insufficient_samples");
    assert_eq!(gate["samples"], 1);
    assert_eq!(gate["required_samples"], 20);
}

#[test]
fn saturation_benchmark_aborts_after_failed_warmup_without_leaking_private_inputs() {
    let response_private = "benchmark-response-private";
    let (base_url, server) = serve_openai_once(
        "401 Unauthorized",
        "application/json",
        &format!("{{\"error\":{{\"message\":\"{response_private}\"}}}}"),
    );
    let query_private = "benchmark-query-private";
    let target = format!("{base_url}?access_token={query_private}");
    let private_values = [
        "benchmark-api-key-private",
        "benchmark-prompt-private",
        "benchmark-expectation-private",
        response_private,
        query_private,
    ];
    let output = run(&[
        "benchmark",
        "saturation",
        "--base-url",
        &target,
        "--model",
        "test-model",
        "--api-key",
        private_values[0],
        "--prompt",
        private_values[1],
        "--workload-id",
        "benchmark-auth-v1",
        "--expect",
        private_values[2],
        "--concurrency-stages",
        "1,2",
        "--warmup-requests-per-worker",
        "1",
        "--requests-per-worker",
        "1",
        "--no-stream",
        "--verify-concurrency",
        "2",
        "--format",
        "json",
    ]);
    let _request = server.join().unwrap();

    assert_eq!(output.status.code(), Some(2));
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let report: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(report["status"], "aborted");
    assert_eq!(report["abort_reason"], "warmup_no_successful_requests");
    assert_eq!(report["warmups"][0]["failure_stage_counts"]["http"], 1);
    assert!(report["stages"].as_array().unwrap().is_empty());
    assert_eq!(report["verification"]["status"], "not_evaluable");
    assert_eq!(report["verification"]["reason"], "stage_not_run");
    for private in private_values {
        assert!(
            !stdout.contains(private),
            "serialized private value: {private}"
        );
    }
}

#[test]
fn benchmark_compare_emits_private_compact_evidence_and_enforces_regressions() {
    let directory = tempfile::tempdir().unwrap();
    let baseline_path = directory.path().join("baseline-benchmark.json");
    let candidate_path = directory.path().join("candidate-benchmark.ndjson");
    let baseline = saved_saturation_benchmark(
        chrono::Utc::now(),
        "https://private-baseline.example",
        1_000_000_000,
    );
    let candidate = saved_saturation_benchmark(
        baseline.started_at + chrono::Duration::seconds(1),
        "https://private-candidate.example",
        2_000_000_000,
    );
    fs::write(
        &baseline_path,
        serde_json::to_string_pretty(&baseline).unwrap(),
    )
    .unwrap();
    fs::write(
        &candidate_path,
        format!(
            "{}\n{}\n",
            serde_json::to_string(&baseline).unwrap(),
            serde_json::to_string(&candidate).unwrap()
        ),
    )
    .unwrap();

    let args = [
        "benchmark",
        "compare",
        baseline_path.to_str().unwrap(),
        candidate_path.to_str().unwrap(),
        "--min-successful-rps-ratio",
        "0.9",
        "--format",
        "ndjson",
    ];
    let output = run(&args);
    let comparison = assert_compact_ndjson(&output);
    assert_eq!(comparison["saturation_comparison_version"], 1);
    assert_eq!(comparison["status"], "regression");
    assert_eq!(comparison["compatible"], true);
    assert_eq!(comparison["regression"], true);
    assert_eq!(comparison["stages"][0]["gates"][0]["status"], "fail");
    let stdout = String::from_utf8_lossy(&output.stdout);
    for forbidden in [
        "private-baseline.example",
        "private-candidate.example",
        "\"attempts\":",
        "\"target\":",
    ] {
        assert!(
            !stdout.contains(forbidden),
            "leaked comparison field: {forbidden}"
        );
    }

    let output = run(&[
        "benchmark",
        "compare",
        baseline_path.to_str().unwrap(),
        candidate_path.to_str().unwrap(),
        "--min-successful-rps-ratio",
        "0.9",
        "--fail-on-regression",
        "--format",
        "ndjson",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stderr.is_empty());
    let comparison: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(comparison["status"], "regression");
}

#[test]
fn benchmark_compare_fails_closed_for_incompatibility_and_rejects_profiles() {
    let directory = tempfile::tempdir().unwrap();
    let baseline_path = directory.path().join("baseline.json");
    let candidate_path = directory.path().join("candidate.json");
    let baseline = saved_saturation_benchmark(
        chrono::Utc::now(),
        "https://baseline.example",
        1_000_000_000,
    );
    let mut candidate = saved_saturation_benchmark(
        baseline.started_at + chrono::Duration::seconds(1),
        "https://candidate.example",
        1_000_000_000,
    );
    candidate.workload_id = "different-workload-v1".to_owned();
    fs::write(&baseline_path, serde_json::to_vec(&baseline).unwrap()).unwrap();
    fs::write(&candidate_path, serde_json::to_vec(&candidate).unwrap()).unwrap();

    let output = run(&[
        "benchmark",
        "compare",
        baseline_path.to_str().unwrap(),
        candidate_path.to_str().unwrap(),
        "--fail-on-regression",
        "--format",
        "json",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stderr.is_empty());
    let comparison: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(comparison["status"], "not_evaluable");
    assert_eq!(comparison["compatible"], false);
    assert_eq!(comparison["regression"], false);
    assert!(comparison.get("stages").is_none());

    let output = clean_config_command()
        .args([
            "--config",
            "must-not-be-loaded.toml",
            "benchmark",
            "compare",
            baseline_path.to_str().unwrap(),
            candidate_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("benchmark saturation"));
    assert!(!stderr.contains("must-not-be-loaded"));
}

#[test]
fn benchmark_compare_decode_and_version_errors_never_echo_report_values() {
    let directory = tempfile::tempdir().unwrap();
    let baseline_path = directory.path().join("baseline.json");
    let candidate_path = directory.path().join("candidate.json");
    let baseline = saved_saturation_benchmark(
        chrono::Utc::now(),
        "https://baseline.example",
        1_000_000_000,
    );
    fs::write(&baseline_path, serde_json::to_vec(&baseline).unwrap()).unwrap();

    let private = "private-invalid-report-value-that-must-not-leak";
    fs::write(
        &candidate_path,
        format!("{{\"saturation_benchmark_version\":\"{private}\"}}"),
    )
    .unwrap();
    let output = run(&[
        "benchmark",
        "compare",
        baseline_path.to_str().unwrap(),
        candidate_path.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("line"));
    assert!(!stderr.contains(private));

    let mut future = baseline;
    future.saturation_benchmark_version = SATURATION_BENCHMARK_VERSION + 1;
    fs::write(&candidate_path, serde_json::to_vec(&future).unwrap()).unwrap();
    let output = run(&[
        "benchmark",
        "compare",
        baseline_path.to_str().unwrap(),
        candidate_path.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unsupported version"));
}

#[test]
fn rollout_compares_saved_canaries_and_emits_stable_compact_ndjson() {
    let directory = tempfile::tempdir().unwrap();
    let baseline_path = directory.path().join("baseline.json");
    let candidate_path = directory.path().join("candidate.ndjson");
    let baseline = rollout_canary("chat-smoke-v1", 100.0, 500.0, 50.0, 100);
    let candidate = rollout_canary("chat-smoke-v1", 105.0, 525.0, 47.5, 99);
    fs::write(
        &baseline_path,
        serde_json::to_string_pretty(&baseline).unwrap(),
    )
    .unwrap();
    fs::write(
        &candidate_path,
        format!(
            "{}\n{}\n",
            serde_json::to_string(&baseline).unwrap(),
            serde_json::to_string(&candidate).unwrap()
        ),
    )
    .unwrap();

    let output = run(&[
        "rollout",
        baseline_path.to_str().unwrap(),
        candidate_path.to_str().unwrap(),
        "--max-p95-ttft-regression-percent",
        "10",
        "--max-p95-e2e-regression-percent",
        "10",
        "--min-output-tps-ratio",
        "0.9",
        "--max-success-drop-percent",
        "1",
        "--fail-on-regression",
        "--format",
        "ndjson",
    ]);

    let comparison = assert_compact_ndjson(&output);
    assert_eq!(comparison["rollout_version"], 1);
    assert_eq!(comparison["compatible"], true);
    assert_eq!(comparison["regression"], false);
    assert_eq!(comparison["baseline"]["workload_id"], "chat-smoke-v1");
    assert_eq!(comparison["gates"].as_array().unwrap().len(), 4);
    assert!(
        comparison["gates"]
            .as_array()
            .unwrap()
            .iter()
            .all(|gate| gate["passed"] == true)
    );
}

#[test]
fn rollout_exits_two_for_metric_regression_or_incompatible_identity() {
    let directory = tempfile::tempdir().unwrap();
    let baseline_path = directory.path().join("baseline.json");
    let candidate_path = directory.path().join("candidate.json");
    let baseline = rollout_canary("chat-smoke-v1", 100.0, 500.0, 50.0, 100);
    let candidate = rollout_canary("chat-smoke-v1", 130.0, 650.0, 35.0, 90);
    fs::write(&baseline_path, serde_json::to_string(&baseline).unwrap()).unwrap();
    fs::write(&candidate_path, serde_json::to_string(&candidate).unwrap()).unwrap();

    let output = run(&[
        "rollout",
        baseline_path.to_str().unwrap(),
        candidate_path.to_str().unwrap(),
        "--max-p95-ttft-regression-percent",
        "10",
        "--max-p95-e2e-regression-percent",
        "10",
        "--min-output-tps-ratio",
        "0.9",
        "--max-success-drop-percent",
        "1",
        "--fail-on-regression",
        "--format",
        "json",
    ]);
    assert_eq!(output.status.code(), Some(2));
    let comparison: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(comparison["compatible"], true);
    assert_eq!(comparison["regression"], true);
    assert!(
        comparison["gates"]
            .as_array()
            .unwrap()
            .iter()
            .all(|gate| gate["passed"] == false)
    );

    let mut incompatible = rollout_canary("different-workload-v1", 100.0, 500.0, 50.0, 100);
    incompatible.target.model = "different-model".to_owned();
    fs::write(
        &candidate_path,
        serde_json::to_string(&incompatible).unwrap(),
    )
    .unwrap();
    let output = run(&[
        "rollout",
        baseline_path.to_str().unwrap(),
        candidate_path.to_str().unwrap(),
        "--fail-on-regression",
        "--format",
        "json",
    ]);
    assert_eq!(output.status.code(), Some(2));
    let comparison: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(comparison["compatible"], false);
    assert_eq!(comparison["regression"], true);
    assert!(comparison["gates"].as_array().unwrap().is_empty());
}

#[test]
fn canary_failure_emits_a_redacted_report_and_exits_two() {
    let generated_output = "generated-output-private";
    let (base_url, server) = serve_openai_once(
        "401 Unauthorized",
        "application/json",
        &format!("{{\"error\":{{\"message\":\"{generated_output}\"}}}}"),
    );
    let target = format!("{base_url}?access_token=query-credential-private");
    let output = Command::new(env!("CARGO_BIN_EXE_gpu-watchman"))
        .args([
            "canary",
            "--base-url",
            &target,
            "--model",
            "test-model",
            "--api-key",
            "canary-api-key-private",
            "--prompt",
            "canary-prompt-private",
            "--workload-id",
            "synthetic-auth-failure-v1",
            "--expect",
            "expected-private-marker",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    let _request = server.join().unwrap();

    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8(output.stdout).unwrap();
    let report: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(report["status"], "fail");
    assert_eq!(report["summary"]["failed"], 1);
    assert_eq!(report["attempts"][0]["failure"]["stage"], "http");
    for private in [
        "canary-api-key-private",
        "canary-prompt-private",
        "expected-private-marker",
        generated_output,
        "query-credential-private",
    ] {
        assert!(
            !stdout.contains(private),
            "serialized private value: {private}"
        );
        assert!(!stderr.contains(private), "stderr private value: {private}");
    }
}

#[test]
fn canary_token_rate_gate_fails_closed_without_authoritative_usage() {
    let stream_body = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"gpu-watchman-ok\"},",
        "\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let (base_url, server) = serve_openai_once("200 OK", "text/event-stream", stream_body);
    let output = run(&[
        "canary",
        "--base-url",
        &base_url,
        "--model",
        "test-model",
        "--min-output-tokens-per-second",
        "1",
        "--format",
        "json",
    ]);
    let _request = server.join().unwrap();

    assert_eq!(output.status.code(), Some(2));
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["status"], "fail");
    assert_eq!(report["summary"]["succeeded"], 1);
    let gate = report["gates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|gate| gate["name"] == "min_output_tokens_per_second")
        .unwrap();
    assert_eq!(gate["passed"], false);
    assert!(gate.get("observed").is_none());
    assert!(gate["detail"].as_str().unwrap().contains("unavailable"));
}

#[test]
fn canary_rejects_concurrency_above_count_as_local_setup_failure() {
    let output = run(&[
        "canary",
        "--model",
        "test-model",
        "--count",
        "1",
        "--concurrency",
        "2",
        "--format",
        "json",
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("canary concurrency"));
}

#[test]
fn canary_rejects_url_userinfo_without_echoing_it() {
    let private = "password-only-private";
    let output = run(&[
        "canary",
        "--base-url",
        &format!("http://:{private}@127.0.0.1:1/v1"),
        "--model",
        "test-model",
        "--format",
        "json",
    ]);

    assert_eq!(output.status.code(), Some(1));
    assert!(!String::from_utf8_lossy(&output.stdout).contains(private));
    assert!(!String::from_utf8_lossy(&output.stderr).contains(private));
    assert!(String::from_utf8_lossy(&output.stderr).contains("must not include credentials"));
}

#[test]
fn canary_requires_explicit_opt_in_for_remote_cleartext_http() {
    let output = run(&[
        "canary",
        "--base-url",
        "http://inference.example.invalid/v1",
        "--model",
        "test-model",
    ]);

    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("--allow-insecure-http"));
}

#[test]
fn canary_rejects_stream_only_gates_in_non_stream_mode() {
    let output = run(&[
        "canary",
        "--model",
        "test-model",
        "--no-stream",
        "--max-ttft",
        "1s",
    ]);

    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("cannot be used with"));
}

#[test]
fn canary_rejects_a_long_running_serial_plan_before_network_access() {
    let output = run(&[
        "canary",
        "--model",
        "test-model",
        "--count",
        "31",
        "--timeout",
        "30s",
    ]);

    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("15 minutes"));
}
