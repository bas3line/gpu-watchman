//! Stable, privacy-preserving evidence for a local inference-runtime process.
//!
//! Runtime Fingerprint v1 is deliberately an observation contract, not a
//! compatibility verdict. It contains no hostname, filesystem path, command
//! line, environment value, model identity, arbitrary diagnostic, or other
//! free-form string. Collectors must reduce transient local data to the fixed
//! enums and bounded numeric values defined here before constructing a report.

use std::collections::BTreeSet;

use chrono::{DateTime, Utc};
use serde::Serialize;

/// Current version of the standalone Runtime Fingerprint report contract.
pub const RUNTIME_FINGERPRINT_VERSION: u32 = 1;

/// Maximum number of explicitly selected processes in one v1 report.
pub const MAX_RUNTIME_FINGERPRINT_TARGETS: usize = 32;

/// Maximum number of distinct recognized mapped-library facts per process.
pub const MAX_RUNTIME_FINGERPRINT_LIBRARIES_PER_PROCESS: usize = 64;

/// Operating-system classification retained by Runtime Fingerprint v1.
///
/// V1 collection is Linux-specific. Other operating-system names are not
/// retained because they are neither actionable evidence nor needed to
/// explain the fixed unsupported-platform result.
#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeOperatingSystem {
    Linux,
    #[default]
    Unsupported,
}

/// Fixed machine-architecture classification retained by the report.
#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeArchitecture {
    X86_64,
    Aarch64,
    Powerpc64Le,
    S390x,
    Riscv64,
    #[default]
    Unknown,
}

/// Path-free and hostname-free host classification.
#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimeFingerprintHost {
    pub operating_system: RuntimeOperatingSystem,
    pub architecture: RuntimeArchitecture,
}

/// A numeric version token with at most four ordered components.
///
/// This representation intentionally cannot retain suffixes, paths, package
/// names, build labels, or arbitrary source text. Collectors should reject an
/// input with zero or more than four numeric components.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimeNumericVersion {
    pub major: u32,
    pub minor: Option<u32>,
    pub patch: Option<u32>,
    pub build: Option<u32>,
}

impl RuntimeNumericVersion {
    /// Construct a one-component numeric version.
    pub const fn major(major: u32) -> Self {
        Self {
            major,
            minor: None,
            patch: None,
            build: None,
        }
    }

    /// Construct a two-component numeric version.
    pub const fn major_minor(major: u32, minor: u32) -> Self {
        Self {
            major,
            minor: Some(minor),
            patch: None,
            build: None,
        }
    }

    /// Construct a three-component numeric version.
    pub const fn major_minor_patch(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor: Some(minor),
            patch: Some(patch),
            build: None,
        }
    }

    /// Construct a four-component numeric version.
    pub const fn major_minor_patch_build(major: u32, minor: u32, patch: u32, build: u32) -> Self {
        Self {
            major,
            minor: Some(minor),
            patch: Some(patch),
            build: Some(build),
        }
    }

    /// Construct a version from one to four numeric components.
    pub fn from_components(components: &[u32]) -> Option<Self> {
        match components {
            [major] => Some(Self::major(*major)),
            [major, minor] => Some(Self::major_minor(*major, *minor)),
            [major, minor, patch] => Some(Self::major_minor_patch(*major, *minor, *patch)),
            [major, minor, patch, build] => Some(Self::major_minor_patch_build(
                *major, *minor, *patch, *build,
            )),
            _ => None,
        }
    }
}

/// Availability of fixed NVIDIA kernel-driver evidence.
#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeDriverState {
    Observed,
    NotPresent,
    Conflict,
    #[default]
    Unavailable,
}

/// Fixed local source used for NVIDIA kernel-module version evidence.
#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeDriverSource {
    #[default]
    None,
    SysModule,
    ProcDriver,
    CorroboratedFixedFiles,
    ConflictingFixedFiles,
}

/// NVIDIA kernel-module evidence, distinct from CUDA UMD/toolkit/runtime data.
///
/// `kernel_module_versions` contains one value when observed or corroborated,
/// multiple values on a conflict, and no values when not present or
/// unavailable. A collector must never choose one conflicting version.
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct RuntimeDriverEvidence {
    pub state: RuntimeDriverState,
    pub source: RuntimeDriverSource,
    pub kernel_module_versions: BTreeSet<RuntimeNumericVersion>,
}

/// Stability of the process identity represented by a numeric PID.
#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeProcessIdentityState {
    Stable,
    Exited,
    Reused,
    #[default]
    Unavailable,
}

/// One bounded process source used by Runtime Fingerprint v1.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeProcessSource {
    Identity,
    CommandLine,
    MemoryMaps,
}

/// Availability of one process evidence source.
#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeSourceState {
    Observed,
    NotPresent,
    Skipped,
    #[default]
    Unavailable,
}

/// Fixed failure or incompleteness reason for a process evidence source.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeSourceReason {
    PermissionDenied,
    NotFound,
    ProcessExited,
    PidReused,
    ChangedDuringCollection,
    LimitExceeded,
    Malformed,
    UnsupportedPlatform,
}

/// Explicit state for a single bounded process evidence source.
#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimeProcessSourceEvidence {
    pub source: RuntimeProcessSource,
    pub state: RuntimeSourceState,
    pub reason: Option<RuntimeSourceReason>,
    pub records: u64,
}

/// Inference engines recognized from conservative fixed entrypoint patterns.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RuntimeEngine {
    #[serde(rename = "vllm")]
    Vllm,
    #[serde(rename = "tgi")]
    Tgi,
    #[serde(rename = "triton")]
    Triton,
    #[serde(rename = "sglang")]
    Sglang,
    #[serde(rename = "tensorrt_llm")]
    TensorRtLlm,
}

/// Kind of privacy-safe evidence supporting an engine candidate.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeEngineEvidenceKind {
    ArgvExecutable,
    ArgvPythonModule,
}

/// A candidate engine identity; this never asserts the engine is authoritative.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimeEngineEvidence {
    pub engine: RuntimeEngine,
    pub evidence: RuntimeEngineEvidenceKind,
}

/// Frameworks inferred only from recognized mapped-library names.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RuntimeFramework {
    #[serde(rename = "pytorch")]
    Pytorch,
    #[serde(rename = "tensorflow")]
    Tensorflow,
    #[serde(rename = "onnx_runtime")]
    OnnxRuntime,
    #[serde(rename = "tensorrt")]
    TensorRt,
}

/// Kind of privacy-safe evidence supporting a framework candidate.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeFrameworkEvidenceKind {
    MappedLibraryName,
}

/// A candidate framework identity; package presence or version is not implied.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimeFrameworkEvidence {
    pub framework: RuntimeFramework,
    pub evidence: RuntimeFrameworkEvidenceKind,
}

/// Fixed mapped-library families retained by Runtime Fingerprint v1.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RuntimeLibraryFamily {
    #[serde(rename = "cuda_driver")]
    CudaDriver,
    #[serde(rename = "cuda_runtime")]
    CudaRuntime,
    #[serde(rename = "cublas")]
    Cublas,
    #[serde(rename = "cudnn")]
    Cudnn,
    #[serde(rename = "nccl")]
    Nccl,
    #[serde(rename = "tensorrt")]
    TensorRt,
    #[serde(rename = "torch_core")]
    TorchCore,
    #[serde(rename = "torch_cuda")]
    TorchCuda,
    #[serde(rename = "tensorflow")]
    Tensorflow,
    #[serde(rename = "onnxruntime_cuda")]
    OnnxRuntimeCuda,
}

/// Source of a mapped-library family observation.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeLibraryEvidenceKind {
    MappedLibraryName,
}

/// Strength of any numeric version retained from a mapped filename.
#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeLibraryVersionEvidence {
    #[default]
    NotObserved,
    MappedFilenameOnly,
}

/// Privacy-safe evidence for one recognized mapped-library family.
///
/// Filename-derived numbers are name-only evidence. They do not prove a
/// package version, toolkit version, loaded symbol set, or compatibility.
#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimeLibraryEvidence {
    pub family: RuntimeLibraryFamily,
    pub evidence: RuntimeLibraryEvidenceKind,
    pub soname_major: Option<u32>,
    pub filename_version: Option<RuntimeNumericVersion>,
    pub version_evidence: RuntimeLibraryVersionEvidence,
}

/// State of one typed launch setting observed in the current process argv.
#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeObservationState {
    Observed,
    PresentUnparsed,
    Ambiguous,
    #[default]
    NotObserved,
}

/// Typed launch observation with an explicit absence/ambiguity state.
///
/// Raw argument text is never retained. Only `Observed` carries a value;
/// collectors should use the constructors to preserve that invariant.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimeObservation<T> {
    pub state: RuntimeObservationState,
    pub value: Option<T>,
}

impl<T> RuntimeObservation<T> {
    /// Construct a successfully parsed typed observation.
    pub const fn observed(value: T) -> Self {
        Self {
            state: RuntimeObservationState::Observed,
            value: Some(value),
        }
    }

    /// Record that a supported flag was present but its value was not retained.
    pub const fn present_unparsed() -> Self {
        Self {
            state: RuntimeObservationState::PresentUnparsed,
            value: None,
        }
    }

    /// Record duplicate, conflicting, or otherwise ambiguous observations.
    pub const fn ambiguous() -> Self {
        Self {
            state: RuntimeObservationState::Ambiguous,
            value: None,
        }
    }

    /// Record that no supported declaration was observed.
    pub const fn not_observed() -> Self {
        Self {
            state: RuntimeObservationState::NotObserved,
            value: None,
        }
    }
}

impl<T> Default for RuntimeObservation<T> {
    fn default() -> Self {
        Self::not_observed()
    }
}

/// Canonical dtype tokens accepted from fixed engine-specific argv parsers.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RuntimeCanonicalDtype {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "fp64")]
    Fp64,
    #[serde(rename = "fp32")]
    Fp32,
    #[serde(rename = "fp16")]
    Fp16,
    #[serde(rename = "bf16")]
    Bf16,
    #[serde(rename = "fp8_e4m3")]
    Fp8E4m3,
    #[serde(rename = "fp8_e5m2")]
    Fp8E5m2,
    #[serde(rename = "int8")]
    Int8,
}

/// Canonical quantization families accepted by fixed engine-specific parsers.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RuntimeCanonicalQuantization {
    #[serde(rename = "aqlm")]
    Aqlm,
    #[serde(rename = "awq")]
    Awq,
    #[serde(rename = "bitsandbytes")]
    BitsAndBytes,
    #[serde(rename = "bitsandbytes_fp4")]
    BitsAndBytesFp4,
    #[serde(rename = "bitsandbytes_nf4")]
    BitsAndBytesNf4,
    #[serde(rename = "compressed_tensors")]
    CompressedTensors,
    #[serde(rename = "eetq")]
    Eetq,
    #[serde(rename = "exl2")]
    Exl2,
    #[serde(rename = "fp8")]
    Fp8,
    #[serde(rename = "gptq")]
    Gptq,
    #[serde(rename = "marlin")]
    Marlin,
    #[serde(rename = "nvfp4")]
    NvFp4,
}

/// Fixed source of v1 launch-setting observations.
#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeLaunchEvidenceKind {
    #[default]
    ProcessCommandLine,
}

/// Typed, privacy-safe settings observed in the current process argv.
///
/// Values describe declarations only. They do not prove engine defaults,
/// effective configuration, model compatibility, or successfully initialized
/// runtime state.
#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimeLaunchObservation {
    pub evidence: RuntimeLaunchEvidenceKind,
    pub model_reference_present: RuntimeObservation<bool>,
    pub tensor_parallel_size: RuntimeObservation<u32>,
    pub pipeline_parallel_size: RuntimeObservation<u32>,
    pub data_parallel_size: RuntimeObservation<u32>,
    pub context_token_limit: RuntimeObservation<u64>,
    pub declared_dtype: RuntimeObservation<RuntimeCanonicalDtype>,
    pub declared_quantization: RuntimeObservation<RuntimeCanonicalQuantization>,
    pub declared_kv_cache_dtype: RuntimeObservation<RuntimeCanonicalDtype>,
}

/// One target process and its bounded runtime observations.
///
/// Sets serialize in enum/value order. The enclosing report's process vector
/// is canonically ordered by ascending PID.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RuntimeProcessFingerprint {
    pub pid: u32,
    pub identity_state: RuntimeProcessIdentityState,
    pub sources: BTreeSet<RuntimeProcessSourceEvidence>,
    pub engine_candidates: BTreeSet<RuntimeEngineEvidence>,
    pub framework_candidates: BTreeSet<RuntimeFrameworkEvidence>,
    pub mapped_libraries: BTreeSet<RuntimeLibraryEvidence>,
    pub launch_observation: RuntimeLaunchObservation,
}

impl RuntimeProcessFingerprint {
    /// Construct a target whose evidence has not yet been collected.
    pub const fn new(pid: u32) -> Self {
        Self {
            pid,
            identity_state: RuntimeProcessIdentityState::Unavailable,
            sources: BTreeSet::new(),
            engine_candidates: BTreeSet::new(),
            framework_candidates: BTreeSet::new(),
            mapped_libraries: BTreeSet::new(),
            launch_observation: RuntimeLaunchObservation {
                evidence: RuntimeLaunchEvidenceKind::ProcessCommandLine,
                model_reference_present: RuntimeObservation::not_observed(),
                tensor_parallel_size: RuntimeObservation::not_observed(),
                pipeline_parallel_size: RuntimeObservation::not_observed(),
                data_parallel_size: RuntimeObservation::not_observed(),
                context_token_limit: RuntimeObservation::not_observed(),
                declared_dtype: RuntimeObservation::not_observed(),
                declared_quantization: RuntimeObservation::not_observed(),
                declared_kv_cache_dtype: RuntimeObservation::not_observed(),
            },
        }
    }
}

/// Compatibility result allowed by Runtime Fingerprint v1.
#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeCompatibilityState {
    #[default]
    NotEvaluated,
}

/// Fixed reasons that v1 evidence cannot support a compatibility verdict.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeAssessmentReason {
    KernelDriverIsNotUserModeDriver,
    MappedNamesAreNotPackageVersions,
    ArgvIsNotEffectiveConfiguration,
    NoModelArtifactBinding,
    MemoryMapsAreNonAtomic,
}

/// Explicit no-verdict assessment accompanying every v1 report.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RuntimeAssessment {
    pub compatibility: RuntimeCompatibilityState,
    pub reasons: BTreeSet<RuntimeAssessmentReason>,
}

impl Default for RuntimeAssessment {
    fn default() -> Self {
        Self {
            compatibility: RuntimeCompatibilityState::NotEvaluated,
            reasons: [
                RuntimeAssessmentReason::KernelDriverIsNotUserModeDriver,
                RuntimeAssessmentReason::MappedNamesAreNotPackageVersions,
                RuntimeAssessmentReason::ArgvIsNotEffectiveConfiguration,
                RuntimeAssessmentReason::NoModelArtifactBinding,
                RuntimeAssessmentReason::MemoryMapsAreNonAtomic,
            ]
            .into_iter()
            .collect(),
        }
    }
}

/// Versioned, deterministic, privacy-preserving local runtime evidence.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RuntimeFingerprintReport {
    pub runtime_fingerprint_version: u32,
    pub collected_at: DateTime<Utc>,
    pub host: RuntimeFingerprintHost,
    pub target_count: u32,
    pub complete: bool,
    pub driver: RuntimeDriverEvidence,
    /// Canonically sorted by ascending PID before serialization.
    pub processes: Vec<RuntimeProcessFingerprint>,
    pub assessment: RuntimeAssessment,
}

impl RuntimeFingerprintReport {
    /// Restore the deterministic v1 process ordering after report assembly.
    pub fn canonicalize(&mut self) {
        self.processes.sort_by_key(|process| process.pid);
    }
}

impl Default for RuntimeFingerprintReport {
    fn default() -> Self {
        Self {
            runtime_fingerprint_version: RUNTIME_FINGERPRINT_VERSION,
            collected_at: Utc::now(),
            host: RuntimeFingerprintHost::default(),
            target_count: 0,
            complete: false,
            driver: RuntimeDriverEvidence::default(),
            processes: Vec::new(),
            assessment: RuntimeAssessment::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone as _;
    use serde_json::{Value, json};

    use super::*;

    #[test]
    fn default_report_is_versioned_incomplete_and_never_claims_compatibility() {
        let report = RuntimeFingerprintReport::default();

        assert_eq!(report.runtime_fingerprint_version, 1);
        assert!(!report.complete);
        assert_eq!(report.target_count, 0);
        assert!(report.processes.is_empty());
        assert_eq!(report.driver.state, RuntimeDriverState::Unavailable);
        assert_eq!(report.driver.source, RuntimeDriverSource::None);
        assert!(report.driver.kernel_module_versions.is_empty());
        assert_eq!(
            report.assessment.compatibility,
            RuntimeCompatibilityState::NotEvaluated
        );
        assert_eq!(report.assessment.reasons.len(), 5);
    }

    #[test]
    fn typed_observations_keep_missing_unparsed_and_ambiguous_values_explicit() {
        assert_eq!(
            serde_json::to_value(RuntimeObservation::observed(8_u32)).unwrap(),
            json!({"state": "observed", "value": 8})
        );
        assert_eq!(
            serde_json::to_value(RuntimeObservation::<u32>::not_observed()).unwrap(),
            json!({"state": "not_observed", "value": null})
        );
        assert_eq!(
            serde_json::to_value(RuntimeObservation::<RuntimeCanonicalDtype>::present_unparsed())
                .unwrap(),
            json!({"state": "present_unparsed", "value": null})
        );
        assert_eq!(
            serde_json::to_value(RuntimeObservation::<RuntimeCanonicalQuantization>::ambiguous())
                .unwrap(),
            json!({"state": "ambiguous", "value": null})
        );
    }

    #[test]
    fn serialization_is_deterministic_and_contains_only_reduced_evidence() {
        let mut driver_versions = BTreeSet::new();
        driver_versions.insert(RuntimeNumericVersion::major_minor_patch(580, 95, 5));

        let mut sources = BTreeSet::new();
        sources.insert(RuntimeProcessSourceEvidence {
            source: RuntimeProcessSource::MemoryMaps,
            state: RuntimeSourceState::Observed,
            reason: None,
            records: 9,
        });
        sources.insert(RuntimeProcessSourceEvidence {
            source: RuntimeProcessSource::CommandLine,
            state: RuntimeSourceState::Observed,
            reason: None,
            records: 7,
        });

        let mut engine_candidates = BTreeSet::new();
        engine_candidates.insert(RuntimeEngineEvidence {
            engine: RuntimeEngine::Vllm,
            evidence: RuntimeEngineEvidenceKind::ArgvPythonModule,
        });

        let mut framework_candidates = BTreeSet::new();
        framework_candidates.insert(RuntimeFrameworkEvidence {
            framework: RuntimeFramework::Pytorch,
            evidence: RuntimeFrameworkEvidenceKind::MappedLibraryName,
        });

        let mut mapped_libraries = BTreeSet::new();
        mapped_libraries.insert(RuntimeLibraryEvidence {
            family: RuntimeLibraryFamily::CudaRuntime,
            evidence: RuntimeLibraryEvidenceKind::MappedLibraryName,
            soname_major: Some(12),
            filename_version: Some(RuntimeNumericVersion::major_minor_patch(12, 8, 90)),
            version_evidence: RuntimeLibraryVersionEvidence::MappedFilenameOnly,
        });

        let process = RuntimeProcessFingerprint {
            pid: 42,
            identity_state: RuntimeProcessIdentityState::Stable,
            sources,
            engine_candidates,
            framework_candidates,
            mapped_libraries,
            launch_observation: RuntimeLaunchObservation {
                model_reference_present: RuntimeObservation::observed(true),
                tensor_parallel_size: RuntimeObservation::observed(8),
                declared_dtype: RuntimeObservation::observed(RuntimeCanonicalDtype::Bf16),
                declared_quantization: RuntimeObservation::observed(
                    RuntimeCanonicalQuantization::Awq,
                ),
                ..RuntimeLaunchObservation::default()
            },
        };

        let report = RuntimeFingerprintReport {
            runtime_fingerprint_version: RUNTIME_FINGERPRINT_VERSION,
            collected_at: Utc.with_ymd_and_hms(2026, 7, 18, 10, 20, 30).unwrap(),
            host: RuntimeFingerprintHost {
                operating_system: RuntimeOperatingSystem::Linux,
                architecture: RuntimeArchitecture::X86_64,
            },
            target_count: 1,
            complete: true,
            driver: RuntimeDriverEvidence {
                state: RuntimeDriverState::Observed,
                source: RuntimeDriverSource::CorroboratedFixedFiles,
                kernel_module_versions: driver_versions,
            },
            processes: vec![process],
            assessment: RuntimeAssessment::default(),
        };

        let first = serde_json::to_string(&report).unwrap();
        let second = serde_json::to_string(&report).unwrap();
        assert_eq!(first, second);

        let value: Value = serde_json::from_str(&first).unwrap();
        assert_eq!(value["runtime_fingerprint_version"], 1);
        assert_eq!(
            value["host"],
            json!({
                "operating_system": "linux",
                "architecture": "x86_64"
            })
        );
        assert_eq!(
            value["driver"]["kernel_module_versions"][0],
            json!({
                "major": 580,
                "minor": 95,
                "patch": 5,
                "build": null
            })
        );
        assert_eq!(
            value["processes"][0]["engine_candidates"][0]["engine"],
            "vllm"
        );
        assert_eq!(
            value["processes"][0]["launch_observation"]["declared_dtype"],
            json!({"state": "observed", "value": "bf16"})
        );
        assert_eq!(value["assessment"]["compatibility"], "not_evaluated");
    }

    #[test]
    fn report_contract_has_no_field_capable_of_retaining_sensitive_free_form_data() {
        let mut report = RuntimeFingerprintReport {
            collected_at: Utc.with_ymd_and_hms(2026, 7, 18, 10, 20, 30).unwrap(),
            target_count: 2,
            processes: vec![
                RuntimeProcessFingerprint::new(20),
                RuntimeProcessFingerprint::new(10),
            ],
            ..RuntimeFingerprintReport::default()
        };
        report.canonicalize();

        assert_eq!(
            report
                .processes
                .iter()
                .map(|process| process.pid)
                .collect::<Vec<_>>(),
            vec![10, 20]
        );

        let output = serde_json::to_string(&report).unwrap();
        for forbidden in [
            "\"hostname\":",
            "\"path\":",
            "\"command\":",
            "\"environment\":",
            "\"model_name\":",
            "\"revision\":",
            "\"tokenizer\":",
            "\"api_key\":",
            "\"secret\":",
            "\"error\":",
            "\"message\":",
            "/srv/private/model",
            "HF_TOKEN",
        ] {
            assert!(
                !output.contains(forbidden),
                "serialized forbidden token {forbidden}"
            );
        }
    }

    #[test]
    fn numeric_versions_reject_empty_and_overlong_component_lists() {
        assert_eq!(RuntimeNumericVersion::from_components(&[]), None);
        assert_eq!(
            RuntimeNumericVersion::from_components(&[12, 8, 90]),
            Some(RuntimeNumericVersion::major_minor_patch(12, 8, 90))
        );
        assert_eq!(
            RuntimeNumericVersion::from_components(&[1, 2, 3, 4, 5]),
            None
        );
    }
}
