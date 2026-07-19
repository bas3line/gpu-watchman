//! Runtime Fingerprint v1 orchestration and completeness policy.

use std::collections::BTreeSet;

use anyhow::{Result, bail};
use chrono::{DateTime, Utc};

use crate::domain::{
    MAX_RUNTIME_FINGERPRINT_TARGETS, RUNTIME_FINGERPRINT_VERSION, RuntimeArchitecture,
    RuntimeAssessment, RuntimeDriverEvidence, RuntimeFingerprintHost, RuntimeFingerprintReport,
    RuntimeOperatingSystem, RuntimeProcessFingerprint, RuntimeProcessIdentityState,
    RuntimeProcessSource, RuntimeSourceState,
};
use crate::telemetry::runtime::{inspect_driver, inspect_process};

/// Inspect the explicitly selected local processes and return bounded runtime evidence.
///
/// Collection failures are represented in the report's typed evidence rather than
/// returned as errors. Errors are reserved for invalid target selection, allowing the
/// caller to serialize an incomplete report before choosing its exit status.
pub fn inspect(pids: &[u32]) -> Result<RuntimeFingerprintReport> {
    validate_targets(pids)?;

    let mut canonical_pids = pids.to_vec();
    canonical_pids.sort_unstable();
    let collected_at = Utc::now();
    let host = local_host_class();
    let driver = inspect_driver();
    let processes = canonical_pids
        .iter()
        .copied()
        .map(inspect_process)
        .collect::<Vec<_>>();

    Ok(assemble_report(
        collected_at,
        host,
        &canonical_pids,
        driver,
        processes,
    ))
}

fn assemble_report(
    collected_at: DateTime<Utc>,
    host: RuntimeFingerprintHost,
    requested_pids: &[u32],
    driver: RuntimeDriverEvidence,
    processes: Vec<RuntimeProcessFingerprint>,
) -> RuntimeFingerprintReport {
    let complete = collection_is_complete(host, requested_pids, &processes);
    let mut report = RuntimeFingerprintReport {
        runtime_fingerprint_version: RUNTIME_FINGERPRINT_VERSION,
        collected_at,
        host,
        target_count: u32::try_from(requested_pids.len())
            .expect("validated runtime target count fits in u32"),
        complete,
        driver,
        processes,
        assessment: RuntimeAssessment::default(),
    };
    report.canonicalize();
    report
}

fn validate_targets(pids: &[u32]) -> Result<()> {
    if pids.is_empty() {
        bail!("runtime inspection requires at least one explicit PID");
    }
    if pids.len() > MAX_RUNTIME_FINGERPRINT_TARGETS {
        bail!("runtime inspection accepts at most {MAX_RUNTIME_FINGERPRINT_TARGETS} explicit PIDs");
    }

    let mut unique = BTreeSet::new();
    for &pid in pids {
        if pid == 0 {
            bail!("runtime inspection PID must be greater than zero");
        }
        if !unique.insert(pid) {
            bail!("duplicate runtime inspection PID {pid}");
        }
    }
    Ok(())
}

fn local_host_class() -> RuntimeFingerprintHost {
    let operating_system = if cfg!(target_os = "linux") {
        RuntimeOperatingSystem::Linux
    } else {
        RuntimeOperatingSystem::Unsupported
    };
    let architecture = match std::env::consts::ARCH {
        "x86_64" => RuntimeArchitecture::X86_64,
        "aarch64" => RuntimeArchitecture::Aarch64,
        "powerpc64" if cfg!(target_endian = "little") => RuntimeArchitecture::Powerpc64Le,
        "s390x" => RuntimeArchitecture::S390x,
        "riscv64" => RuntimeArchitecture::Riscv64,
        _ => RuntimeArchitecture::Unknown,
    };
    RuntimeFingerprintHost {
        operating_system,
        architecture,
    }
}

fn collection_is_complete(
    host: RuntimeFingerprintHost,
    requested_pids: &[u32],
    processes: &[RuntimeProcessFingerprint],
) -> bool {
    if host.operating_system != RuntimeOperatingSystem::Linux
        || processes.len() != requested_pids.len()
    {
        return false;
    }

    let requested = requested_pids.iter().copied().collect::<BTreeSet<_>>();
    if requested.len() != requested_pids.len() {
        return false;
    }
    let returned = processes
        .iter()
        .map(|process| process.pid)
        .collect::<BTreeSet<_>>();
    if returned != requested || returned.len() != processes.len() {
        return false;
    }

    processes.iter().all(process_is_complete)
}

fn process_is_complete(process: &RuntimeProcessFingerprint) -> bool {
    const REQUIRED_SOURCES: [RuntimeProcessSource; 3] = [
        RuntimeProcessSource::Identity,
        RuntimeProcessSource::CommandLine,
        RuntimeProcessSource::MemoryMaps,
    ];

    process.identity_state == RuntimeProcessIdentityState::Stable
        && process.sources.len() == REQUIRED_SOURCES.len()
        && REQUIRED_SOURCES.iter().all(|required| {
            let mut matching = process
                .sources
                .iter()
                .filter(|evidence| evidence.source == *required);
            matching.next().is_some_and(|evidence| {
                evidence.state == RuntimeSourceState::Observed && evidence.reason.is_none()
            }) && matching.next().is_none()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        RuntimeDriverSource, RuntimeDriverState, RuntimeProcessSourceEvidence, RuntimeSourceReason,
    };

    fn linux_host() -> RuntimeFingerprintHost {
        RuntimeFingerprintHost {
            operating_system: RuntimeOperatingSystem::Linux,
            architecture: RuntimeArchitecture::X86_64,
        }
    }

    fn complete_process(pid: u32) -> RuntimeProcessFingerprint {
        let mut process = RuntimeProcessFingerprint::new(pid);
        process.identity_state = RuntimeProcessIdentityState::Stable;
        process.sources = [
            RuntimeProcessSource::Identity,
            RuntimeProcessSource::CommandLine,
            RuntimeProcessSource::MemoryMaps,
        ]
        .into_iter()
        .map(|source| RuntimeProcessSourceEvidence {
            source,
            state: RuntimeSourceState::Observed,
            reason: None,
            records: 1,
        })
        .collect();
        process
    }

    #[test]
    fn target_validation_rejects_empty_zero_duplicate_and_excessive_input() {
        assert!(validate_targets(&[]).is_err());
        assert!(validate_targets(&[0]).is_err());
        assert!(validate_targets(&[7, 7]).is_err());

        let excessive =
            (1..=u32::try_from(MAX_RUNTIME_FINGERPRINT_TARGETS).unwrap() + 1).collect::<Vec<_>>();
        assert!(validate_targets(&excessive).is_err());
        assert!(validate_targets(&[9, 2, 17]).is_ok());
    }

    #[test]
    fn complete_requires_linux_stable_identity_and_each_observed_source() {
        let process = complete_process(42);
        assert!(collection_is_complete(
            linux_host(),
            &[42],
            std::slice::from_ref(&process)
        ));

        let unsupported = RuntimeFingerprintHost {
            operating_system: RuntimeOperatingSystem::Unsupported,
            architecture: RuntimeArchitecture::X86_64,
        };
        assert!(!collection_is_complete(
            unsupported,
            &[42],
            std::slice::from_ref(&process)
        ));

        let mut unstable = process.clone();
        unstable.identity_state = RuntimeProcessIdentityState::Unavailable;
        assert!(!collection_is_complete(linux_host(), &[42], &[unstable]));

        let mut missing = process.clone();
        missing
            .sources
            .retain(|source| source.source != RuntimeProcessSource::MemoryMaps);
        assert!(!collection_is_complete(linux_host(), &[42], &[missing]));

        let mut unavailable = process;
        unavailable.sources.replace(RuntimeProcessSourceEvidence {
            source: RuntimeProcessSource::CommandLine,
            state: RuntimeSourceState::Unavailable,
            reason: Some(RuntimeSourceReason::PermissionDenied),
            records: 0,
        });
        unavailable.sources.retain(|source| {
            source.source != RuntimeProcessSource::CommandLine
                || source.state == RuntimeSourceState::Unavailable
        });
        assert!(!collection_is_complete(linux_host(), &[42], &[unavailable]));
    }

    #[test]
    fn absent_or_unavailable_driver_does_not_make_stable_processes_incomplete() {
        for state in [
            RuntimeDriverState::NotPresent,
            RuntimeDriverState::Unavailable,
        ] {
            let driver = RuntimeDriverEvidence {
                state,
                source: RuntimeDriverSource::None,
                kernel_module_versions: BTreeSet::new(),
            };
            let report = assemble_report(
                Utc::now(),
                linux_host(),
                &[81],
                driver,
                vec![complete_process(81)],
            );

            assert!(report.complete);
            assert_eq!(report.driver.state, state);
        }
    }

    #[test]
    fn completeness_never_accepts_wrong_duplicate_or_extra_process_records() {
        let process = complete_process(11);
        assert!(!collection_is_complete(
            linux_host(),
            &[11],
            &[complete_process(12)]
        ));
        assert!(!collection_is_complete(
            linux_host(),
            &[11, 12],
            &[process.clone(), process.clone()]
        ));
        assert!(!collection_is_complete(
            linux_host(),
            &[11],
            &[process, complete_process(12)]
        ));
    }

    #[test]
    fn duplicate_source_entries_cannot_look_complete() {
        let mut process = complete_process(23);
        process.sources.insert(RuntimeProcessSourceEvidence {
            source: RuntimeProcessSource::MemoryMaps,
            state: RuntimeSourceState::Unavailable,
            reason: Some(RuntimeSourceReason::ChangedDuringCollection),
            records: 0,
        });

        assert!(!collection_is_complete(linux_host(), &[23], &[process]));
    }
}
