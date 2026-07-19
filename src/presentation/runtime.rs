//! Stable, privacy-preserving text for Runtime Fingerprint reports.

use std::fmt::Write as _;

use chrono::SecondsFormat;

use crate::domain::{
    RuntimeArchitecture, RuntimeAssessmentReason, RuntimeCanonicalDtype,
    RuntimeCanonicalQuantization, RuntimeCompatibilityState, RuntimeDriverEvidence,
    RuntimeDriverSource, RuntimeDriverState, RuntimeEngine, RuntimeEngineEvidenceKind,
    RuntimeFingerprintReport, RuntimeFramework, RuntimeFrameworkEvidenceKind,
    RuntimeLaunchEvidenceKind, RuntimeLaunchObservation, RuntimeLibraryEvidence,
    RuntimeLibraryEvidenceKind, RuntimeLibraryFamily, RuntimeLibraryVersionEvidence,
    RuntimeNumericVersion, RuntimeObservation, RuntimeObservationState, RuntimeOperatingSystem,
    RuntimeProcessFingerprint, RuntimeProcessIdentityState, RuntimeProcessSource,
    RuntimeProcessSourceEvidence, RuntimeSourceReason, RuntimeSourceState,
};

/// Render a deterministic, path-free Runtime Fingerprint v1 summary.
///
/// Every string written here is fixed by this renderer. The report contract
/// contributes only enums, bounded numeric values, and a UTC timestamp; raw
/// command lines, paths, model identities, hostnames, and diagnostics cannot
/// enter this output.
pub fn render(report: &RuntimeFingerprintReport) -> String {
    let mut output = String::new();
    let _ = writeln!(
        output,
        "GPU Watchman runtime fingerprint v{}",
        report.runtime_fingerprint_version
    );
    let _ = writeln!(
        output,
        "compatibility: {}",
        compatibility_state(report.assessment.compatibility)
    );
    let _ = writeln!(
        output,
        "completeness: {} | target_count={} | process_records={}",
        if report.complete {
            "COMPLETE"
        } else {
            "INCOMPLETE"
        },
        report.target_count,
        report.processes.len()
    );
    let _ = writeln!(
        output,
        "collected_at: {}",
        report
            .collected_at
            .to_rfc3339_opts(SecondsFormat::AutoSi, true)
    );
    let _ = writeln!(
        output,
        "host_class: operating_system={} architecture={}",
        operating_system(report.host.operating_system),
        architecture(report.host.architecture)
    );

    render_driver(&mut output, &report.driver);

    output.push_str("\nPROCESSES\n");
    if report.processes.is_empty() {
        output.push_str("  none\n");
    } else {
        let mut processes = report.processes.iter().collect::<Vec<_>>();
        processes.sort_by_key(|process| process.pid);
        for process in processes {
            render_process(&mut output, process);
        }
    }

    output.push_str("\nASSESSMENT REASONS\n");
    if report.assessment.reasons.is_empty() {
        output.push_str("  none recorded; compatibility remains NOT EVALUATED\n");
    } else {
        for reason in &report.assessment.reasons {
            let _ = writeln!(
                output,
                "  [{}] {}",
                assessment_reason(*reason),
                assessment_caveat(*reason)
            );
        }
    }
    output.push_str(
        "CAVEAT: observation evidence only; engine/framework candidates, mapped names, and launch declarations are not a compatibility verdict.\n",
    );
    output
}

fn render_driver(output: &mut String, driver: &RuntimeDriverEvidence) {
    output.push_str("\nDRIVER EVIDENCE\n");
    let _ = writeln!(
        output,
        "  state={} source={}",
        driver_state(driver.state),
        driver_source(driver.source)
    );
    if driver.kernel_module_versions.is_empty() {
        output.push_str("  kernel_module_versions: none\n");
    } else {
        let versions = driver
            .kernel_module_versions
            .iter()
            .map(format_numeric_version)
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(output, "  kernel_module_versions: {versions}");
    }
    output.push_str(
        "  caveat: kernel-module evidence is not CUDA user-mode driver, toolkit, or runtime compatibility evidence.\n",
    );
}

fn render_process(output: &mut String, process: &RuntimeProcessFingerprint) {
    let _ = writeln!(
        output,
        "\nPID {}  identity={}",
        process.pid,
        process_identity_state(process.identity_state)
    );
    output.push_str("  sources:\n");
    if process.sources.is_empty() {
        output.push_str("    none\n");
    } else {
        for source in &process.sources {
            render_process_source(output, source);
        }
    }

    let engines = process
        .engine_candidates
        .iter()
        .map(|candidate| {
            format!(
                "{}({})",
                engine(candidate.engine),
                engine_evidence_kind(candidate.evidence)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(output, "  engine_candidates: {}", none_if_empty(&engines));

    let frameworks = process
        .framework_candidates
        .iter()
        .map(|candidate| {
            format!(
                "{}({})",
                framework(candidate.framework),
                framework_evidence_kind(candidate.evidence)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(
        output,
        "  framework_candidates: {}",
        none_if_empty(&frameworks)
    );

    output.push_str("  mapped_libraries:\n");
    if process.mapped_libraries.is_empty() {
        output.push_str("    none\n");
    } else {
        for library in &process.mapped_libraries {
            render_library(output, library);
        }
    }

    render_launch_observation(output, &process.launch_observation);
}

fn render_process_source(output: &mut String, evidence: &RuntimeProcessSourceEvidence) {
    let reason = evidence.reason.map_or("-", source_reason);
    let _ = writeln!(
        output,
        "    {:<12} state={:<11} reason={:<25} records={}",
        process_source(evidence.source),
        source_state(evidence.state),
        reason,
        evidence.records
    );
}

fn render_library(output: &mut String, library: &RuntimeLibraryEvidence) {
    let soname_major = library
        .soname_major
        .map_or_else(|| "-".to_owned(), |value| value.to_string());
    let filename_version = library
        .filename_version
        .as_ref()
        .map_or_else(|| "-".to_owned(), format_numeric_version);
    let _ = writeln!(
        output,
        "    {} evidence={} soname_major={} filename_version={} version_evidence={}",
        library_family(library.family),
        library_evidence_kind(library.evidence),
        soname_major,
        filename_version,
        library_version_evidence(library.version_evidence)
    );
}

fn render_launch_observation(output: &mut String, launch: &RuntimeLaunchObservation) {
    let _ = writeln!(
        output,
        "  launch_observations: evidence={} (declarations only)",
        launch_evidence_kind(launch.evidence)
    );
    render_observation(
        output,
        "model_reference_present",
        &launch.model_reference_present,
        std::string::ToString::to_string,
    );
    render_observation(
        output,
        "tensor_parallel_size",
        &launch.tensor_parallel_size,
        u32::to_string,
    );
    render_observation(
        output,
        "pipeline_parallel_size",
        &launch.pipeline_parallel_size,
        u32::to_string,
    );
    render_observation(
        output,
        "data_parallel_size",
        &launch.data_parallel_size,
        u32::to_string,
    );
    render_observation(
        output,
        "context_token_limit",
        &launch.context_token_limit,
        u64::to_string,
    );
    render_observation(output, "declared_dtype", &launch.declared_dtype, |value| {
        canonical_dtype(*value).to_owned()
    });
    render_observation(
        output,
        "declared_quantization",
        &launch.declared_quantization,
        |value| canonical_quantization(*value).to_owned(),
    );
    render_observation(
        output,
        "declared_kv_cache_dtype",
        &launch.declared_kv_cache_dtype,
        |value| canonical_dtype(*value).to_owned(),
    );
}

fn render_observation<T>(
    output: &mut String,
    name: &str,
    observation: &RuntimeObservation<T>,
    render_value: impl Fn(&T) -> String,
) {
    let value = match (observation.state, observation.value.as_ref()) {
        (RuntimeObservationState::Observed, Some(value)) => render_value(value),
        (RuntimeObservationState::Observed, None) => "missing".to_owned(),
        (_, Some(_)) => "ignored_inconsistent".to_owned(),
        (_, None) => "-".to_owned(),
    };
    let _ = writeln!(
        output,
        "    {name:<27} state={:<16} value={value}",
        observation_state(observation.state)
    );
}

fn format_numeric_version(version: &RuntimeNumericVersion) -> String {
    match (version.minor, version.patch, version.build) {
        (None, None, None) => version.major.to_string(),
        (Some(minor), None, None) => format!("{}.{minor}", version.major),
        (Some(minor), Some(patch), None) => format!("{}.{minor}.{patch}", version.major),
        (Some(minor), Some(patch), Some(build)) => {
            format!("{}.{minor}.{patch}.{build}", version.major)
        }
        (minor, patch, build) => format!(
            "{}.{}.{}.{}",
            version.major,
            optional_component(minor),
            optional_component(patch),
            optional_component(build)
        ),
    }
}

fn optional_component(component: Option<u32>) -> String {
    component.map_or_else(|| "?".to_owned(), |value| value.to_string())
}

fn none_if_empty(value: &str) -> &str {
    if value.is_empty() { "none" } else { value }
}

const fn operating_system(value: RuntimeOperatingSystem) -> &'static str {
    match value {
        RuntimeOperatingSystem::Linux => "linux",
        RuntimeOperatingSystem::Unsupported => "unsupported",
    }
}

const fn architecture(value: RuntimeArchitecture) -> &'static str {
    match value {
        RuntimeArchitecture::X86_64 => "x86_64",
        RuntimeArchitecture::Aarch64 => "aarch64",
        RuntimeArchitecture::Powerpc64Le => "powerpc64_le",
        RuntimeArchitecture::S390x => "s390x",
        RuntimeArchitecture::Riscv64 => "riscv64",
        RuntimeArchitecture::Unknown => "unknown",
    }
}

const fn driver_state(value: RuntimeDriverState) -> &'static str {
    match value {
        RuntimeDriverState::Observed => "observed",
        RuntimeDriverState::NotPresent => "not_present",
        RuntimeDriverState::Conflict => "conflict",
        RuntimeDriverState::Unavailable => "unavailable",
    }
}

const fn driver_source(value: RuntimeDriverSource) -> &'static str {
    match value {
        RuntimeDriverSource::None => "none",
        RuntimeDriverSource::SysModule => "sys_module",
        RuntimeDriverSource::ProcDriver => "proc_driver",
        RuntimeDriverSource::CorroboratedFixedFiles => "corroborated_fixed_files",
        RuntimeDriverSource::ConflictingFixedFiles => "conflicting_fixed_files",
    }
}

const fn process_identity_state(value: RuntimeProcessIdentityState) -> &'static str {
    match value {
        RuntimeProcessIdentityState::Stable => "stable",
        RuntimeProcessIdentityState::Exited => "exited",
        RuntimeProcessIdentityState::Reused => "reused",
        RuntimeProcessIdentityState::Unavailable => "unavailable",
    }
}

const fn process_source(value: RuntimeProcessSource) -> &'static str {
    match value {
        RuntimeProcessSource::Identity => "identity",
        RuntimeProcessSource::CommandLine => "command_line",
        RuntimeProcessSource::MemoryMaps => "memory_maps",
    }
}

const fn source_state(value: RuntimeSourceState) -> &'static str {
    match value {
        RuntimeSourceState::Observed => "observed",
        RuntimeSourceState::NotPresent => "not_present",
        RuntimeSourceState::Skipped => "skipped",
        RuntimeSourceState::Unavailable => "unavailable",
    }
}

const fn source_reason(value: RuntimeSourceReason) -> &'static str {
    match value {
        RuntimeSourceReason::PermissionDenied => "permission_denied",
        RuntimeSourceReason::NotFound => "not_found",
        RuntimeSourceReason::ProcessExited => "process_exited",
        RuntimeSourceReason::PidReused => "pid_reused",
        RuntimeSourceReason::ChangedDuringCollection => "changed_during_collection",
        RuntimeSourceReason::LimitExceeded => "limit_exceeded",
        RuntimeSourceReason::Malformed => "malformed",
        RuntimeSourceReason::UnsupportedPlatform => "unsupported_platform",
    }
}

const fn engine(value: RuntimeEngine) -> &'static str {
    match value {
        RuntimeEngine::Vllm => "vllm",
        RuntimeEngine::Tgi => "tgi",
        RuntimeEngine::Triton => "triton",
        RuntimeEngine::Sglang => "sglang",
        RuntimeEngine::TensorRtLlm => "tensorrt_llm",
    }
}

const fn engine_evidence_kind(value: RuntimeEngineEvidenceKind) -> &'static str {
    match value {
        RuntimeEngineEvidenceKind::ArgvExecutable => "argv_executable",
        RuntimeEngineEvidenceKind::ArgvPythonModule => "argv_python_module",
    }
}

const fn framework(value: RuntimeFramework) -> &'static str {
    match value {
        RuntimeFramework::Pytorch => "pytorch",
        RuntimeFramework::Tensorflow => "tensorflow",
        RuntimeFramework::OnnxRuntime => "onnx_runtime",
        RuntimeFramework::TensorRt => "tensorrt",
    }
}

const fn framework_evidence_kind(value: RuntimeFrameworkEvidenceKind) -> &'static str {
    match value {
        RuntimeFrameworkEvidenceKind::MappedLibraryName => "mapped_library_name",
    }
}

const fn library_family(value: RuntimeLibraryFamily) -> &'static str {
    match value {
        RuntimeLibraryFamily::CudaDriver => "cuda_driver",
        RuntimeLibraryFamily::CudaRuntime => "cuda_runtime",
        RuntimeLibraryFamily::Cublas => "cublas",
        RuntimeLibraryFamily::Cudnn => "cudnn",
        RuntimeLibraryFamily::Nccl => "nccl",
        RuntimeLibraryFamily::TensorRt => "tensorrt",
        RuntimeLibraryFamily::TorchCore => "torch_core",
        RuntimeLibraryFamily::TorchCuda => "torch_cuda",
        RuntimeLibraryFamily::Tensorflow => "tensorflow",
        RuntimeLibraryFamily::OnnxRuntimeCuda => "onnxruntime_cuda",
    }
}

const fn library_evidence_kind(value: RuntimeLibraryEvidenceKind) -> &'static str {
    match value {
        RuntimeLibraryEvidenceKind::MappedLibraryName => "mapped_library_name",
    }
}

const fn library_version_evidence(value: RuntimeLibraryVersionEvidence) -> &'static str {
    match value {
        RuntimeLibraryVersionEvidence::NotObserved => "not_observed",
        RuntimeLibraryVersionEvidence::MappedFilenameOnly => "mapped_filename_only",
    }
}

const fn launch_evidence_kind(value: RuntimeLaunchEvidenceKind) -> &'static str {
    match value {
        RuntimeLaunchEvidenceKind::ProcessCommandLine => "process_command_line",
    }
}

const fn observation_state(value: RuntimeObservationState) -> &'static str {
    match value {
        RuntimeObservationState::Observed => "observed",
        RuntimeObservationState::PresentUnparsed => "present_unparsed",
        RuntimeObservationState::Ambiguous => "ambiguous",
        RuntimeObservationState::NotObserved => "not_observed",
    }
}

const fn canonical_dtype(value: RuntimeCanonicalDtype) -> &'static str {
    match value {
        RuntimeCanonicalDtype::Auto => "auto",
        RuntimeCanonicalDtype::Fp64 => "fp64",
        RuntimeCanonicalDtype::Fp32 => "fp32",
        RuntimeCanonicalDtype::Fp16 => "fp16",
        RuntimeCanonicalDtype::Bf16 => "bf16",
        RuntimeCanonicalDtype::Fp8E4m3 => "fp8_e4m3",
        RuntimeCanonicalDtype::Fp8E5m2 => "fp8_e5m2",
        RuntimeCanonicalDtype::Int8 => "int8",
    }
}

const fn canonical_quantization(value: RuntimeCanonicalQuantization) -> &'static str {
    match value {
        RuntimeCanonicalQuantization::Aqlm => "aqlm",
        RuntimeCanonicalQuantization::Awq => "awq",
        RuntimeCanonicalQuantization::BitsAndBytes => "bitsandbytes",
        RuntimeCanonicalQuantization::BitsAndBytesFp4 => "bitsandbytes_fp4",
        RuntimeCanonicalQuantization::BitsAndBytesNf4 => "bitsandbytes_nf4",
        RuntimeCanonicalQuantization::CompressedTensors => "compressed_tensors",
        RuntimeCanonicalQuantization::Eetq => "eetq",
        RuntimeCanonicalQuantization::Exl2 => "exl2",
        RuntimeCanonicalQuantization::Fp8 => "fp8",
        RuntimeCanonicalQuantization::Gptq => "gptq",
        RuntimeCanonicalQuantization::Marlin => "marlin",
        RuntimeCanonicalQuantization::NvFp4 => "nvfp4",
    }
}

const fn compatibility_state(value: RuntimeCompatibilityState) -> &'static str {
    match value {
        RuntimeCompatibilityState::NotEvaluated => "NOT EVALUATED",
    }
}

const fn assessment_reason(value: RuntimeAssessmentReason) -> &'static str {
    match value {
        RuntimeAssessmentReason::KernelDriverIsNotUserModeDriver => {
            "kernel_driver_is_not_user_mode_driver"
        }
        RuntimeAssessmentReason::MappedNamesAreNotPackageVersions => {
            "mapped_names_are_not_package_versions"
        }
        RuntimeAssessmentReason::ArgvIsNotEffectiveConfiguration => {
            "argv_is_not_effective_configuration"
        }
        RuntimeAssessmentReason::NoModelArtifactBinding => "no_model_artifact_binding",
        RuntimeAssessmentReason::MemoryMapsAreNonAtomic => "memory_maps_are_non_atomic",
    }
}

const fn assessment_caveat(value: RuntimeAssessmentReason) -> &'static str {
    match value {
        RuntimeAssessmentReason::KernelDriverIsNotUserModeDriver => {
            "Kernel-module versions do not identify CUDA user-mode driver, toolkit, or runtime versions."
        }
        RuntimeAssessmentReason::MappedNamesAreNotPackageVersions => {
            "Mapped-library names and filename numbers do not prove package versions or loaded symbols."
        }
        RuntimeAssessmentReason::ArgvIsNotEffectiveConfiguration => {
            "Launch arguments are declarations, not effective runtime configuration."
        }
        RuntimeAssessmentReason::NoModelArtifactBinding => {
            "No model artifact is bound to these observations."
        }
        RuntimeAssessmentReason::MemoryMapsAreNonAtomic => {
            "Memory-map inspection is a non-atomic process snapshot."
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use chrono::{TimeZone as _, Utc};

    use super::*;
    use crate::domain::{
        RUNTIME_FINGERPRINT_VERSION, RuntimeAssessment, RuntimeEngineEvidence,
        RuntimeFingerprintHost, RuntimeFrameworkEvidence,
    };

    fn observed_process(pid: u32) -> RuntimeProcessFingerprint {
        RuntimeProcessFingerprint {
            pid,
            identity_state: RuntimeProcessIdentityState::Stable,
            sources: [
                RuntimeProcessSourceEvidence {
                    source: RuntimeProcessSource::Identity,
                    state: RuntimeSourceState::Observed,
                    reason: None,
                    records: 1,
                },
                RuntimeProcessSourceEvidence {
                    source: RuntimeProcessSource::CommandLine,
                    state: RuntimeSourceState::Observed,
                    reason: None,
                    records: 7,
                },
                RuntimeProcessSourceEvidence {
                    source: RuntimeProcessSource::MemoryMaps,
                    state: RuntimeSourceState::Unavailable,
                    reason: Some(RuntimeSourceReason::PermissionDenied),
                    records: 0,
                },
            ]
            .into_iter()
            .collect(),
            engine_candidates: [RuntimeEngineEvidence {
                engine: RuntimeEngine::Vllm,
                evidence: RuntimeEngineEvidenceKind::ArgvPythonModule,
            }]
            .into_iter()
            .collect(),
            framework_candidates: [RuntimeFrameworkEvidence {
                framework: RuntimeFramework::Pytorch,
                evidence: RuntimeFrameworkEvidenceKind::MappedLibraryName,
            }]
            .into_iter()
            .collect(),
            mapped_libraries: [RuntimeLibraryEvidence {
                family: RuntimeLibraryFamily::CudaRuntime,
                evidence: RuntimeLibraryEvidenceKind::MappedLibraryName,
                soname_major: Some(12),
                filename_version: Some(RuntimeNumericVersion::major_minor_patch(12, 8, 90)),
                version_evidence: RuntimeLibraryVersionEvidence::MappedFilenameOnly,
            }]
            .into_iter()
            .collect(),
            launch_observation: RuntimeLaunchObservation {
                evidence: RuntimeLaunchEvidenceKind::ProcessCommandLine,
                model_reference_present: RuntimeObservation::observed(true),
                tensor_parallel_size: RuntimeObservation::observed(8),
                pipeline_parallel_size: RuntimeObservation::ambiguous(),
                data_parallel_size: RuntimeObservation::present_unparsed(),
                context_token_limit: RuntimeObservation::observed(32_768),
                declared_dtype: RuntimeObservation::observed(RuntimeCanonicalDtype::Bf16),
                declared_quantization: RuntimeObservation::observed(
                    RuntimeCanonicalQuantization::Awq,
                ),
                declared_kv_cache_dtype: RuntimeObservation::not_observed(),
            },
        }
    }

    fn report() -> RuntimeFingerprintReport {
        RuntimeFingerprintReport {
            runtime_fingerprint_version: RUNTIME_FINGERPRINT_VERSION,
            collected_at: Utc.with_ymd_and_hms(2026, 7, 18, 10, 20, 30).unwrap(),
            host: RuntimeFingerprintHost {
                operating_system: RuntimeOperatingSystem::Linux,
                architecture: RuntimeArchitecture::X86_64,
            },
            target_count: 2,
            complete: false,
            driver: RuntimeDriverEvidence {
                state: RuntimeDriverState::Conflict,
                source: RuntimeDriverSource::ConflictingFixedFiles,
                kernel_module_versions: [
                    RuntimeNumericVersion::major_minor_patch(580, 95, 5),
                    RuntimeNumericVersion::major_minor_patch(550, 54, 14),
                ]
                .into_iter()
                .collect(),
            },
            processes: vec![observed_process(42), RuntimeProcessFingerprint::new(7)],
            assessment: RuntimeAssessment::default(),
        }
    }

    #[test]
    fn rendering_is_deterministic_complete_and_conspicuously_no_verdict() {
        let report = report();
        let first = render(&report);
        let second = render(&report);

        assert_eq!(first, second);
        assert!(
            first
                .starts_with("GPU Watchman runtime fingerprint v1\ncompatibility: NOT EVALUATED\n")
        );
        assert!(first.contains("completeness: INCOMPLETE | target_count=2 | process_records=2"));
        assert!(first.contains(
            "state=conflict source=conflicting_fixed_files\n  kernel_module_versions: 550.54.14, 580.95.5"
        ));
        assert!(first.contains("memory_maps  state=unavailable reason=permission_denied"));
        assert!(first.contains("engine_candidates: vllm(argv_python_module)"));
        assert!(first.contains("framework_candidates: pytorch(mapped_library_name)"));
        assert!(first.contains(
            "cuda_runtime evidence=mapped_library_name soname_major=12 filename_version=12.8.90 version_evidence=mapped_filename_only"
        ));
        assert!(first.contains("tensor_parallel_size        state=observed         value=8"));
        assert!(first.contains("pipeline_parallel_size      state=ambiguous        value=-"));
        assert!(first.contains("[kernel_driver_is_not_user_mode_driver]"));

        let pid_7 = first.find("PID 7  identity=unavailable").unwrap();
        let pid_42 = first.find("PID 42  identity=stable").unwrap();
        assert!(pid_7 < pid_42, "renderer must sort processes by PID");
    }

    #[test]
    fn renderer_exposes_only_fixed_privacy_safe_vocabulary() {
        let output = render(&report());

        for forbidden in [
            "/srv/private/model",
            "secret-hostname",
            "--api-key",
            "HF_TOKEN",
            "tenant-model-name",
            "raw_command",
            "raw_path",
            "free_form_error",
        ] {
            assert!(
                !output.contains(forbidden),
                "rendered forbidden token {forbidden}"
            );
        }
        assert!(!output.contains("hostname:"));
        assert!(!output.contains("model_identity"));
        assert!(!output.contains("command_line_value"));
    }

    #[test]
    fn empty_report_stays_incomplete_and_non_optimistic() {
        let mut report = RuntimeFingerprintReport {
            collected_at: Utc.with_ymd_and_hms(2026, 7, 18, 10, 20, 30).unwrap(),
            ..RuntimeFingerprintReport::default()
        };
        report.assessment.reasons = BTreeSet::new();

        let output = render(&report);
        assert!(output.contains("compatibility: NOT EVALUATED"));
        assert!(output.contains("completeness: INCOMPLETE | target_count=0 | process_records=0"));
        assert!(output.contains("kernel_module_versions: none"));
        assert!(output.contains("PROCESSES\n  none"));
        assert!(output.contains("none recorded; compatibility remains NOT EVALUATED"));
    }
}
