//! Bounded, evidence-only local inference-runtime inspection.
//!
//! Linux collection is anchored to one `/proc/<pid>` directory descriptor and
//! never executes a target, driver utility, interpreter, or package manager.
//! Raw command lines, paths, map records, and operating-system errors are
//! reduced to the fixed Runtime Fingerprint contract before this module
//! returns.

// Pure Linux parsers are compiled by tests on other Unix hosts so the rustix
// descriptor path stays type-checked; they are intentionally unused there.
#![cfg_attr(not(target_os = "linux"), allow(dead_code, unused_imports))]

use std::collections::BTreeSet;

use crate::domain::{
    MAX_RUNTIME_FINGERPRINT_LIBRARIES_PER_PROCESS, RuntimeCanonicalDtype,
    RuntimeCanonicalQuantization, RuntimeDriverEvidence, RuntimeDriverSource, RuntimeDriverState,
    RuntimeEngine, RuntimeEngineEvidence, RuntimeEngineEvidenceKind, RuntimeFramework,
    RuntimeFrameworkEvidence, RuntimeFrameworkEvidenceKind, RuntimeLaunchObservation,
    RuntimeLibraryEvidence, RuntimeLibraryEvidenceKind, RuntimeLibraryFamily,
    RuntimeLibraryVersionEvidence, RuntimeNumericVersion, RuntimeObservation,
    RuntimeObservationState, RuntimeProcessFingerprint, RuntimeProcessIdentityState,
    RuntimeProcessSource, RuntimeProcessSourceEvidence, RuntimeSourceReason, RuntimeSourceState,
};

const MAX_STAT_BYTES: usize = 8 * 1024;
const MAX_CMDLINE_BYTES: usize = 256 * 1024;
const MAX_CMDLINE_ARGS: usize = 4096;
const MAX_ARG_BYTES: usize = 16 * 1024;
const MAX_MAPS_BYTES: usize = 8 * 1024 * 1024;
const MAX_MAP_RECORDS: usize = 65_536;
const MAX_MAP_LINE_BYTES: usize = 16 * 1024;
const MAX_DRIVER_BYTES: usize = 4 * 1024;
const MAX_VERSION_TOKEN_BYTES: usize = 64;

/// Inspect fixed NVIDIA kernel-module version files without invoking a command.
#[cfg(target_os = "linux")]
pub(crate) fn inspect_driver() -> RuntimeDriverEvidence {
    inspect_driver_linux()
}

#[cfg(any(target_os = "linux", all(test, unix)))]
fn inspect_driver_linux() -> RuntimeDriverEvidence {
    let sys = read_fixed_file(b"/sys/module/nvidia/version", MAX_DRIVER_BYTES)
        .and_then(|bytes| parse_sys_driver_version(&bytes));
    let proc = read_fixed_file(b"/proc/driver/nvidia/version", MAX_DRIVER_BYTES)
        .and_then(|bytes| parse_proc_driver_version(&bytes));

    match (sys, proc) {
        (FixedEvidence::Observed(left), FixedEvidence::Observed(right)) if left == right => {
            RuntimeDriverEvidence {
                state: RuntimeDriverState::Observed,
                source: RuntimeDriverSource::CorroboratedFixedFiles,
                kernel_module_versions: [left].into_iter().collect(),
            }
        }
        (FixedEvidence::Observed(left), FixedEvidence::Observed(right)) => RuntimeDriverEvidence {
            state: RuntimeDriverState::Conflict,
            source: RuntimeDriverSource::ConflictingFixedFiles,
            kernel_module_versions: [left, right].into_iter().collect(),
        },
        (FixedEvidence::Observed(version), _) => RuntimeDriverEvidence {
            state: RuntimeDriverState::Observed,
            source: RuntimeDriverSource::SysModule,
            kernel_module_versions: [version].into_iter().collect(),
        },
        (_, FixedEvidence::Observed(version)) => RuntimeDriverEvidence {
            state: RuntimeDriverState::Observed,
            source: RuntimeDriverSource::ProcDriver,
            kernel_module_versions: [version].into_iter().collect(),
        },
        (FixedEvidence::Missing, FixedEvidence::Missing) => RuntimeDriverEvidence {
            state: RuntimeDriverState::NotPresent,
            source: RuntimeDriverSource::None,
            kernel_module_versions: BTreeSet::new(),
        },
        _ => RuntimeDriverEvidence::default(),
    }
}

/// Produce fixed unsupported evidence on non-Linux hosts.
#[cfg(not(target_os = "linux"))]
pub(crate) fn inspect_driver() -> RuntimeDriverEvidence {
    RuntimeDriverEvidence::default()
}

/// Inspect one explicitly selected process through an anchored procfs descriptor.
#[cfg(target_os = "linux")]
pub(crate) fn inspect_process(pid: u32) -> RuntimeProcessFingerprint {
    inspect_process_linux(pid)
}

#[cfg(any(target_os = "linux", all(test, unix)))]
fn inspect_process_linux(pid: u32) -> RuntimeProcessFingerprint {
    let directory = match open_process_directory(pid) {
        Ok(directory) => directory,
        Err(reason) => return unavailable_process(pid, reason),
    };

    let first_identity = read_process_child(&directory, b"stat", MAX_STAT_BYTES)
        .and_then(|bytes| parse_proc_stat(&bytes, pid));
    let Ok(first_start_time) = first_identity else {
        let reason = first_identity.unwrap_err();
        let final_identity = read_process_child(&directory, b"stat", MAX_STAT_BYTES)
            .and_then(|bytes| parse_proc_stat(&bytes, pid));
        let reason = final_identity.err().unwrap_or(reason);
        return unavailable_process(pid, reason);
    };

    // These are collected into temporaries. Nothing derived from argv or maps
    // is published until the final identity bookend proves the same process.
    let cmdline_first = read_process_child(&directory, b"cmdline", MAX_CMDLINE_BYTES);
    let maps = read_process_child(&directory, b"maps", MAX_MAPS_BYTES);
    let cmdline_second = read_process_child(&directory, b"cmdline", MAX_CMDLINE_BYTES);
    let final_identity = read_process_child(&directory, b"stat", MAX_STAT_BYTES)
        .and_then(|bytes| parse_proc_stat(&bytes, pid));

    let final_start_time = match final_identity {
        Ok(start_time) => start_time,
        Err(RuntimeSourceReason::NotFound | RuntimeSourceReason::ProcessExited) => {
            return identity_lost_process(pid, RuntimeProcessIdentityState::Exited);
        }
        Err(reason) => return unavailable_process(pid, reason),
    };
    if first_start_time != final_start_time {
        return identity_lost_process(pid, RuntimeProcessIdentityState::Reused);
    }

    let (command_source, command) = reduce_cmdline(cmdline_first, cmdline_second);
    let (maps_source, mapped_libraries, framework_candidates) = reduce_maps(maps);

    let mut process = RuntimeProcessFingerprint::new(pid);
    process.identity_state = RuntimeProcessIdentityState::Stable;
    process.sources = [
        source_evidence(
            RuntimeProcessSource::Identity,
            RuntimeSourceState::Observed,
            None,
            2,
        ),
        command_source,
        maps_source,
    ]
    .into_iter()
    .collect();
    if let Some(command) = command {
        process.engine_candidates = command.engine_candidates;
        process.launch_observation = command.launch_observation;
    }
    process.mapped_libraries = mapped_libraries;
    process.framework_candidates = framework_candidates;
    process
}

/// Produce fixed unsupported evidence on non-Linux hosts.
#[cfg(not(target_os = "linux"))]
pub(crate) fn inspect_process(pid: u32) -> RuntimeProcessFingerprint {
    unavailable_process(pid, RuntimeSourceReason::UnsupportedPlatform)
}

fn unavailable_process(pid: u32, reason: RuntimeSourceReason) -> RuntimeProcessFingerprint {
    let identity_state = match reason {
        RuntimeSourceReason::NotFound | RuntimeSourceReason::ProcessExited => {
            RuntimeProcessIdentityState::Exited
        }
        RuntimeSourceReason::PidReused => RuntimeProcessIdentityState::Reused,
        _ => RuntimeProcessIdentityState::Unavailable,
    };
    let mut process = RuntimeProcessFingerprint::new(pid);
    process.identity_state = identity_state;
    process.sources = [
        source_evidence(
            RuntimeProcessSource::Identity,
            RuntimeSourceState::Unavailable,
            Some(reason),
            0,
        ),
        source_evidence(
            RuntimeProcessSource::CommandLine,
            RuntimeSourceState::Skipped,
            Some(reason),
            0,
        ),
        source_evidence(
            RuntimeProcessSource::MemoryMaps,
            RuntimeSourceState::Skipped,
            Some(reason),
            0,
        ),
    ]
    .into_iter()
    .collect();
    process
}

#[cfg(any(target_os = "linux", all(test, unix)))]
fn identity_lost_process(
    pid: u32,
    identity_state: RuntimeProcessIdentityState,
) -> RuntimeProcessFingerprint {
    let reason = if identity_state == RuntimeProcessIdentityState::Reused {
        RuntimeSourceReason::PidReused
    } else {
        RuntimeSourceReason::ProcessExited
    };
    unavailable_process(pid, reason)
}

fn source_evidence(
    source: RuntimeProcessSource,
    state: RuntimeSourceState,
    reason: Option<RuntimeSourceReason>,
    records: u64,
) -> RuntimeProcessSourceEvidence {
    RuntimeProcessSourceEvidence {
        source,
        state,
        reason,
        records,
    }
}

#[cfg(any(target_os = "linux", all(test, unix)))]
fn reduce_cmdline(
    first: Result<Vec<u8>, RuntimeSourceReason>,
    second: Result<Vec<u8>, RuntimeSourceReason>,
) -> (RuntimeProcessSourceEvidence, Option<ParsedCommandLine>) {
    let (first, second) = match (first, second) {
        (Ok(first), Ok(second)) => (first, second),
        (Err(reason), _) | (_, Err(reason)) => {
            return (
                source_evidence(
                    RuntimeProcessSource::CommandLine,
                    RuntimeSourceState::Unavailable,
                    Some(reason),
                    0,
                ),
                None,
            );
        }
    };
    if first != second {
        return (
            source_evidence(
                RuntimeProcessSource::CommandLine,
                RuntimeSourceState::Unavailable,
                Some(RuntimeSourceReason::ChangedDuringCollection),
                0,
            ),
            None,
        );
    }
    match parse_cmdline(&first) {
        Ok(parsed) => {
            let state = if parsed.arg_count == 0 {
                RuntimeSourceState::NotPresent
            } else {
                RuntimeSourceState::Observed
            };
            (
                source_evidence(
                    RuntimeProcessSource::CommandLine,
                    state,
                    None,
                    parsed.arg_count,
                ),
                Some(parsed),
            )
        }
        Err(reason) => (
            source_evidence(
                RuntimeProcessSource::CommandLine,
                RuntimeSourceState::Unavailable,
                Some(reason),
                0,
            ),
            None,
        ),
    }
}

#[cfg(any(target_os = "linux", all(test, unix)))]
fn reduce_maps(
    maps: Result<Vec<u8>, RuntimeSourceReason>,
) -> (
    RuntimeProcessSourceEvidence,
    BTreeSet<RuntimeLibraryEvidence>,
    BTreeSet<RuntimeFrameworkEvidence>,
) {
    let maps = match maps {
        Ok(maps) => maps,
        Err(reason) => {
            return (
                source_evidence(
                    RuntimeProcessSource::MemoryMaps,
                    RuntimeSourceState::Unavailable,
                    Some(reason),
                    0,
                ),
                BTreeSet::new(),
                BTreeSet::new(),
            );
        }
    };
    match parse_maps(&maps) {
        Ok(parsed) => (
            source_evidence(
                RuntimeProcessSource::MemoryMaps,
                RuntimeSourceState::Observed,
                None,
                parsed.records,
            ),
            parsed.libraries,
            parsed.frameworks,
        ),
        Err(reason) => (
            source_evidence(
                RuntimeProcessSource::MemoryMaps,
                RuntimeSourceState::Unavailable,
                Some(reason),
                0,
            ),
            BTreeSet::new(),
            BTreeSet::new(),
        ),
    }
}

#[cfg(any(target_os = "linux", all(test, unix)))]
fn open_process_directory(pid: u32) -> Result<std::fs::File, RuntimeSourceReason> {
    use rustix::fs::{Mode, OFlags};

    let path = format!("/proc/{pid}");
    rustix::fs::open(
        path,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map(Into::into)
    .map_err(map_errno)
}

#[cfg(any(target_os = "linux", all(test, unix)))]
fn read_process_child(
    directory: &std::fs::File,
    name: &[u8],
    limit: usize,
) -> Result<Vec<u8>, RuntimeSourceReason> {
    use rustix::fs::{Mode, OFlags};

    let descriptor = rustix::fs::openat(
        directory,
        name,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map_err(map_errno)?;
    read_bounded(std::fs::File::from(descriptor), limit)
}

#[cfg(any(target_os = "linux", all(test, unix)))]
fn read_fixed_file(path: &[u8], limit: usize) -> FixedEvidence<Vec<u8>> {
    use rustix::fs::{Mode, OFlags};

    let descriptor = match rustix::fs::open(
        path,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
    ) {
        Ok(descriptor) => descriptor,
        Err(error) if map_errno(error) == RuntimeSourceReason::NotFound => {
            return FixedEvidence::Missing;
        }
        Err(_) => return FixedEvidence::Unavailable,
    };
    match read_bounded(std::fs::File::from(descriptor), limit) {
        Ok(bytes) => FixedEvidence::Observed(bytes),
        Err(_) => FixedEvidence::Unavailable,
    }
}

#[cfg(any(target_os = "linux", all(test, unix)))]
fn read_bounded(mut file: std::fs::File, limit: usize) -> Result<Vec<u8>, RuntimeSourceReason> {
    use std::io::Read as _;

    let mut bytes = Vec::with_capacity(limit.min(8 * 1024));
    file.by_ref()
        .take(u64::try_from(limit).unwrap_or(u64::MAX).saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| map_io_error(&error))?;
    if bytes.len() > limit {
        return Err(RuntimeSourceReason::LimitExceeded);
    }
    Ok(bytes)
}

#[cfg(any(target_os = "linux", all(test, unix)))]
fn map_errno(error: rustix::io::Errno) -> RuntimeSourceReason {
    map_os_error(error.raw_os_error())
}

#[cfg(any(target_os = "linux", all(test, unix)))]
fn map_io_error(error: &std::io::Error) -> RuntimeSourceReason {
    error
        .raw_os_error()
        .map_or(RuntimeSourceReason::Malformed, map_os_error)
}

#[cfg(any(target_os = "linux", all(test, unix)))]
fn map_os_error(code: i32) -> RuntimeSourceReason {
    match code {
        libc::EACCES | libc::EPERM => RuntimeSourceReason::PermissionDenied,
        libc::ENOENT => RuntimeSourceReason::NotFound,
        libc::ESRCH => RuntimeSourceReason::ProcessExited,
        _ => RuntimeSourceReason::Malformed,
    }
}

fn parse_proc_stat(bytes: &[u8], expected_pid: u32) -> Result<u64, RuntimeSourceReason> {
    let open = bytes
        .iter()
        .position(|byte| *byte == b'(')
        .ok_or(RuntimeSourceReason::Malformed)?;
    let close = bytes
        .iter()
        .rposition(|byte| *byte == b')')
        .filter(|close| *close > open)
        .ok_or(RuntimeSourceReason::Malformed)?;
    let pid = parse_ascii_u32(trim_ascii(&bytes[..open])).ok_or(RuntimeSourceReason::Malformed)?;
    if pid != expected_pid {
        return Err(RuntimeSourceReason::Malformed);
    }
    let fields = bytes[close + 1..]
        .split(u8::is_ascii_whitespace)
        .filter(|field| !field.is_empty())
        .collect::<Vec<_>>();
    if fields.len() < 20 || fields[0].len() != 1 {
        return Err(RuntimeSourceReason::Malformed);
    }
    parse_ascii_u64(fields[19]).ok_or(RuntimeSourceReason::Malformed)
}

#[derive(Debug)]
struct ParsedCommandLine {
    arg_count: u64,
    engine_candidates: BTreeSet<RuntimeEngineEvidence>,
    launch_observation: RuntimeLaunchObservation,
}

fn parse_cmdline(bytes: &[u8]) -> Result<ParsedCommandLine, RuntimeSourceReason> {
    let args = split_cmdline(bytes)?;
    let (engine_candidates, launch_start) = recognize_engines(&args);
    let engines = engine_candidates
        .iter()
        .map(|evidence| evidence.engine)
        .collect::<BTreeSet<_>>();
    let launch_observation = if let ([engine], Some(launch_start)) = (
        engines.iter().copied().collect::<Vec<_>>().as_slice(),
        launch_start,
    ) {
        parse_launch(*engine, &args, launch_start)
    } else {
        RuntimeLaunchObservation::default()
    };
    Ok(ParsedCommandLine {
        arg_count: u64::try_from(args.len()).unwrap_or(u64::MAX),
        engine_candidates,
        launch_observation,
    })
}

fn split_cmdline(bytes: &[u8]) -> Result<Vec<&[u8]>, RuntimeSourceReason> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    let mut args = bytes.split(|byte| *byte == 0).collect::<Vec<_>>();
    if bytes.last() == Some(&0) {
        args.pop();
    }
    if args.len() > MAX_CMDLINE_ARGS || args.iter().any(|arg| arg.len() > MAX_ARG_BYTES) {
        return Err(RuntimeSourceReason::LimitExceeded);
    }
    Ok(args)
}

fn recognize_engines(args: &[&[u8]]) -> (BTreeSet<RuntimeEngineEvidence>, Option<usize>) {
    let mut candidates = BTreeSet::new();
    let Some(executable) = args.first().copied().map(basename) else {
        return (candidates, None);
    };
    let mut launch_start = None;
    if let Some(engine) = executable_engine(executable) {
        candidates.insert(RuntimeEngineEvidence {
            engine,
            evidence: RuntimeEngineEvidenceKind::ArgvExecutable,
        });
        launch_start = Some(1);
    }
    if is_python_executable(executable)
        && let Some((engine, module_arguments_start)) = python_entrypoint_engine(args)
    {
        candidates.insert(RuntimeEngineEvidence {
            engine,
            evidence: RuntimeEngineEvidenceKind::ArgvPythonModule,
        });
        launch_start = Some(module_arguments_start);
    }
    (candidates, launch_start)
}

fn executable_engine(value: &[u8]) -> Option<RuntimeEngine> {
    match value {
        b"vllm" => Some(RuntimeEngine::Vllm),
        b"text-generation-launcher" | b"text-generation-server" => Some(RuntimeEngine::Tgi),
        b"tritonserver" => Some(RuntimeEngine::Triton),
        b"sglang" => Some(RuntimeEngine::Sglang),
        b"trtllm-serve" => Some(RuntimeEngine::TensorRtLlm),
        _ => None,
    }
}

fn is_python_executable(value: &[u8]) -> bool {
    value == b"python"
        || value == b"python3"
        || value
            .strip_prefix(b"python3.")
            .is_some_and(|suffix| !suffix.is_empty() && suffix.iter().all(u8::is_ascii_digit))
}

fn python_entrypoint_engine(args: &[&[u8]]) -> Option<(RuntimeEngine, usize)> {
    let mut index = 1;
    while let Some(argument) = args.get(index).copied() {
        match argument {
            b"-m" => {
                let engine = args
                    .get(index + 1)
                    .copied()
                    .and_then(python_module_engine)?;
                return Some((engine, index + 2));
            }
            // These select a different CPython interface (command, stdin, or
            // script) or terminate interpreter-option parsing.
            b"-c" | b"-" | b"--" => return None,
            // Options whose value is the next argv item.
            b"-W" | b"-X" | b"--check-hash-based-pycs" => {
                args.get(index + 1)?;
                index += 2;
            }
            // Supported no-value CPython options that may precede `-m`.
            b"-b" | b"-bb" | b"-B" | b"-d" | b"-E" | b"-i" | b"-I" | b"-O" | b"-OO" | b"-P"
            | b"-q" | b"-s" | b"-S" | b"-u" | b"-v" | b"-x" => {
                index += 1;
            }
            // CPython also accepts attached -W/-X values. Unknown options are
            // not guessed because strict entrypoint evidence is preferable to
            // a false engine candidate.
            value if (value.starts_with(b"-W") || value.starts_with(b"-X")) && value.len() > 2 => {
                index += 1;
            }
            _ => return None,
        }
    }
    None
}

fn python_module_engine(value: &[u8]) -> Option<RuntimeEngine> {
    match value {
        b"vllm.entrypoints.openai.api_server" => Some(RuntimeEngine::Vllm),
        b"sglang.launch_server" | b"sglang.srt.entrypoints.http_server" => {
            Some(RuntimeEngine::Sglang)
        }
        b"tensorrt_llm.commands.serve" => Some(RuntimeEngine::TensorRtLlm),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
enum LaunchField {
    Model,
    TensorParallel,
    PipelineParallel,
    DataParallel,
    Context,
    Dtype,
    Quantization,
    KvCacheDtype,
}

fn parse_launch(
    engine: RuntimeEngine,
    args: &[&[u8]],
    launch_start: usize,
) -> RuntimeLaunchObservation {
    let mut launch = RuntimeLaunchObservation::default();
    let launch_end = args
        .iter()
        .enumerate()
        .skip(launch_start)
        .find_map(|(index, argument)| (*argument == b"--").then_some(index))
        .unwrap_or(args.len());
    let args = &args[..launch_end];
    let mut index = launch_start;
    while index < args.len() {
        let argument = args[index];
        let (flag, inline) = split_flag(argument);
        let Some(field) = launch_field(engine, flag) else {
            index += 1;
            continue;
        };
        let mut consumed_next = false;
        let value = inline.or_else(|| {
            args.get(index + 1).copied().filter(|candidate| {
                if candidate.starts_with(b"-") {
                    false
                } else {
                    consumed_next = true;
                    true
                }
            })
        });
        apply_launch_value(&mut launch, field, value);
        index += usize::from(consumed_next) + 1;
    }

    if engine == RuntimeEngine::Vllm {
        for pair in args[launch_start..].windows(2) {
            if pair[0] == b"serve" && !pair[1].is_empty() && !pair[1].starts_with(b"-") {
                observe_value(&mut launch.model_reference_present, Some(true));
            }
        }
    }
    launch
}

fn split_flag(argument: &[u8]) -> (&[u8], Option<&[u8]>) {
    argument
        .iter()
        .position(|byte| *byte == b'=')
        .map_or((argument, None), |position| {
            (&argument[..position], Some(&argument[position + 1..]))
        })
}

fn launch_field(engine: RuntimeEngine, flag: &[u8]) -> Option<LaunchField> {
    match engine {
        RuntimeEngine::Vllm => match flag {
            b"--model" => Some(LaunchField::Model),
            b"--tensor-parallel-size" => Some(LaunchField::TensorParallel),
            b"--pipeline-parallel-size" => Some(LaunchField::PipelineParallel),
            b"--data-parallel-size" => Some(LaunchField::DataParallel),
            b"--max-model-len" => Some(LaunchField::Context),
            b"--dtype" => Some(LaunchField::Dtype),
            b"--quantization" => Some(LaunchField::Quantization),
            b"--kv-cache-dtype" => Some(LaunchField::KvCacheDtype),
            _ => None,
        },
        RuntimeEngine::Tgi => match flag {
            b"--model-id" => Some(LaunchField::Model),
            b"--num-shard" => Some(LaunchField::TensorParallel),
            b"--max-total-tokens" => Some(LaunchField::Context),
            b"--dtype" => Some(LaunchField::Dtype),
            b"--quantize" => Some(LaunchField::Quantization),
            _ => None,
        },
        RuntimeEngine::Triton => match flag {
            b"--model-repository" => Some(LaunchField::Model),
            _ => None,
        },
        RuntimeEngine::Sglang => match flag {
            b"--model-path" => Some(LaunchField::Model),
            b"--tp-size" | b"--tensor-parallel-size" => Some(LaunchField::TensorParallel),
            b"--pp-size" | b"--pipeline-parallel-size" => Some(LaunchField::PipelineParallel),
            b"--dp-size" | b"--data-parallel-size" => Some(LaunchField::DataParallel),
            b"--context-length" | b"--max-model-len" => Some(LaunchField::Context),
            b"--dtype" => Some(LaunchField::Dtype),
            b"--quantization" => Some(LaunchField::Quantization),
            b"--kv-cache-dtype" => Some(LaunchField::KvCacheDtype),
            _ => None,
        },
        RuntimeEngine::TensorRtLlm => match flag {
            b"--model" | b"--model-path" => Some(LaunchField::Model),
            b"--tp_size" | b"--tp-size" => Some(LaunchField::TensorParallel),
            b"--pp_size" | b"--pp-size" => Some(LaunchField::PipelineParallel),
            b"--max_seq_len" | b"--max-seq-len" => Some(LaunchField::Context),
            b"--dtype" => Some(LaunchField::Dtype),
            b"--kv_cache_dtype" | b"--kv-cache-dtype" => Some(LaunchField::KvCacheDtype),
            _ => None,
        },
    }
}

fn apply_launch_value(
    launch: &mut RuntimeLaunchObservation,
    field: LaunchField,
    value: Option<&[u8]>,
) {
    match field {
        LaunchField::Model => observe_value(
            &mut launch.model_reference_present,
            value.filter(|value| !value.is_empty()).map(|_| true),
        ),
        LaunchField::TensorParallel => observe_value(
            &mut launch.tensor_parallel_size,
            value.and_then(parse_positive_ascii_u32),
        ),
        LaunchField::PipelineParallel => observe_value(
            &mut launch.pipeline_parallel_size,
            value.and_then(parse_positive_ascii_u32),
        ),
        LaunchField::DataParallel => observe_value(
            &mut launch.data_parallel_size,
            value.and_then(parse_positive_ascii_u32),
        ),
        LaunchField::Context => observe_value(
            &mut launch.context_token_limit,
            value.and_then(parse_positive_ascii_u64),
        ),
        LaunchField::Dtype => {
            observe_value(&mut launch.declared_dtype, value.and_then(parse_dtype));
        }
        LaunchField::Quantization => observe_value(
            &mut launch.declared_quantization,
            value.and_then(parse_quantization),
        ),
        LaunchField::KvCacheDtype => observe_value(
            &mut launch.declared_kv_cache_dtype,
            value.and_then(parse_dtype),
        ),
    }
}

fn observe_value<T: Copy>(observation: &mut RuntimeObservation<T>, value: Option<T>) {
    if observation.state != RuntimeObservationState::NotObserved {
        *observation = RuntimeObservation::ambiguous();
    } else if let Some(value) = value {
        *observation = RuntimeObservation::observed(value);
    } else {
        *observation = RuntimeObservation::present_unparsed();
    }
}

fn parse_dtype(value: &[u8]) -> Option<RuntimeCanonicalDtype> {
    if eq_ascii(value, b"auto") {
        Some(RuntimeCanonicalDtype::Auto)
    } else if eq_ascii(value, b"fp64") || eq_ascii(value, b"float64") || eq_ascii(value, b"double")
    {
        Some(RuntimeCanonicalDtype::Fp64)
    } else if eq_ascii(value, b"fp32") || eq_ascii(value, b"float32") {
        Some(RuntimeCanonicalDtype::Fp32)
    } else if eq_ascii(value, b"fp16") || eq_ascii(value, b"float16") || eq_ascii(value, b"half") {
        Some(RuntimeCanonicalDtype::Fp16)
    } else if eq_ascii(value, b"bf16") || eq_ascii(value, b"bfloat16") {
        Some(RuntimeCanonicalDtype::Bf16)
    } else if eq_ascii(value, b"fp8_e4m3") || eq_ascii(value, b"fp8_e4m3fn") {
        Some(RuntimeCanonicalDtype::Fp8E4m3)
    } else if eq_ascii(value, b"fp8_e5m2") {
        Some(RuntimeCanonicalDtype::Fp8E5m2)
    } else if eq_ascii(value, b"int8") {
        Some(RuntimeCanonicalDtype::Int8)
    } else {
        None
    }
}

fn parse_quantization(value: &[u8]) -> Option<RuntimeCanonicalQuantization> {
    if eq_ascii(value, b"aqlm") {
        Some(RuntimeCanonicalQuantization::Aqlm)
    } else if eq_ascii(value, b"awq") {
        Some(RuntimeCanonicalQuantization::Awq)
    } else if eq_ascii(value, b"bitsandbytes") || eq_ascii(value, b"bnb") {
        Some(RuntimeCanonicalQuantization::BitsAndBytes)
    } else if eq_ascii(value, b"bitsandbytes_fp4") || eq_ascii(value, b"bnb_fp4") {
        Some(RuntimeCanonicalQuantization::BitsAndBytesFp4)
    } else if eq_ascii(value, b"bitsandbytes_nf4") || eq_ascii(value, b"bnb_nf4") {
        Some(RuntimeCanonicalQuantization::BitsAndBytesNf4)
    } else if eq_ascii(value, b"compressed-tensors") || eq_ascii(value, b"compressed_tensors") {
        Some(RuntimeCanonicalQuantization::CompressedTensors)
    } else if eq_ascii(value, b"eetq") {
        Some(RuntimeCanonicalQuantization::Eetq)
    } else if eq_ascii(value, b"exl2") {
        Some(RuntimeCanonicalQuantization::Exl2)
    } else if eq_ascii(value, b"fp8") {
        Some(RuntimeCanonicalQuantization::Fp8)
    } else if eq_ascii(value, b"gptq") {
        Some(RuntimeCanonicalQuantization::Gptq)
    } else if eq_ascii(value, b"marlin") {
        Some(RuntimeCanonicalQuantization::Marlin)
    } else if eq_ascii(value, b"nvfp4") {
        Some(RuntimeCanonicalQuantization::NvFp4)
    } else {
        None
    }
}

fn eq_ascii(left: &[u8], right: &[u8]) -> bool {
    left.eq_ignore_ascii_case(right)
}

#[derive(Debug)]
struct ParsedMaps {
    records: u64,
    libraries: BTreeSet<RuntimeLibraryEvidence>,
    frameworks: BTreeSet<RuntimeFrameworkEvidence>,
}

fn parse_maps(bytes: &[u8]) -> Result<ParsedMaps, RuntimeSourceReason> {
    let mut records = 0_usize;
    let mut libraries = BTreeSet::new();
    for line in bytes.split(|byte| *byte == b'\n') {
        if line.is_empty() {
            continue;
        }
        if line.len() > MAX_MAP_LINE_BYTES {
            return Err(RuntimeSourceReason::LimitExceeded);
        }
        records = records
            .checked_add(1)
            .ok_or(RuntimeSourceReason::LimitExceeded)?;
        if records > MAX_MAP_RECORDS {
            return Err(RuntimeSourceReason::LimitExceeded);
        }
        let path = maps_pathname(line)?;
        let Some(path) = path else {
            continue;
        };
        let path = path.strip_suffix(b" (deleted)").unwrap_or(path);
        let name = basename(path);
        if let Some(library) = recognize_library(name) {
            libraries.insert(library);
            if libraries.len() > MAX_RUNTIME_FINGERPRINT_LIBRARIES_PER_PROCESS {
                return Err(RuntimeSourceReason::LimitExceeded);
            }
        }
    }
    let frameworks = frameworks_from_libraries(&libraries);
    Ok(ParsedMaps {
        records: u64::try_from(records).unwrap_or(u64::MAX),
        libraries,
        frameworks,
    })
}

fn maps_pathname(line: &[u8]) -> Result<Option<&[u8]>, RuntimeSourceReason> {
    let mut cursor = 0;
    for _ in 0..5 {
        while line.get(cursor).is_some_and(u8::is_ascii_whitespace) {
            cursor += 1;
        }
        let start = cursor;
        while line
            .get(cursor)
            .is_some_and(|byte| !byte.is_ascii_whitespace())
        {
            cursor += 1;
        }
        if cursor == start {
            return Err(RuntimeSourceReason::Malformed);
        }
    }
    while line.get(cursor).is_some_and(u8::is_ascii_whitespace) {
        cursor += 1;
    }
    if cursor == line.len() {
        Ok(None)
    } else {
        Ok(Some(&line[cursor..]))
    }
}

fn recognize_library(name: &[u8]) -> Option<RuntimeLibraryEvidence> {
    const STEMS: &[(&[u8], RuntimeLibraryFamily)] = &[
        (b"libcuda.so", RuntimeLibraryFamily::CudaDriver),
        (b"libcudart.so", RuntimeLibraryFamily::CudaRuntime),
        (b"libcublas.so", RuntimeLibraryFamily::Cublas),
        (b"libcublasLt.so", RuntimeLibraryFamily::Cublas),
        (b"libcudnn.so", RuntimeLibraryFamily::Cudnn),
        (b"libcudnn_adv.so", RuntimeLibraryFamily::Cudnn),
        (b"libcudnn_cnn.so", RuntimeLibraryFamily::Cudnn),
        (b"libcudnn_graph.so", RuntimeLibraryFamily::Cudnn),
        (b"libcudnn_heuristic.so", RuntimeLibraryFamily::Cudnn),
        (b"libcudnn_ops.so", RuntimeLibraryFamily::Cudnn),
        (b"libcudnn_adv_infer.so", RuntimeLibraryFamily::Cudnn),
        (b"libcudnn_adv_train.so", RuntimeLibraryFamily::Cudnn),
        (b"libcudnn_cnn_infer.so", RuntimeLibraryFamily::Cudnn),
        (b"libcudnn_cnn_train.so", RuntimeLibraryFamily::Cudnn),
        (b"libcudnn_ops_infer.so", RuntimeLibraryFamily::Cudnn),
        (b"libcudnn_ops_train.so", RuntimeLibraryFamily::Cudnn),
        (b"libnccl.so", RuntimeLibraryFamily::Nccl),
        (b"libnvinfer.so", RuntimeLibraryFamily::TensorRt),
        (b"libnvinfer_plugin.so", RuntimeLibraryFamily::TensorRt),
        (b"libnvonnxparser.so", RuntimeLibraryFamily::TensorRt),
        (b"libtorch.so", RuntimeLibraryFamily::TorchCore),
        (b"libtorch_cpu.so", RuntimeLibraryFamily::TorchCore),
        (b"libc10.so", RuntimeLibraryFamily::TorchCore),
        (b"libtorch_cuda.so", RuntimeLibraryFamily::TorchCuda),
        (b"libc10_cuda.so", RuntimeLibraryFamily::TorchCuda),
        (
            b"libtensorflow_framework.so",
            RuntimeLibraryFamily::Tensorflow,
        ),
        (b"libtensorflow_cc.so", RuntimeLibraryFamily::Tensorflow),
        (
            b"libonnxruntime_providers_cuda.so",
            RuntimeLibraryFamily::OnnxRuntimeCuda,
        ),
    ];
    let (stem, family) = STEMS.iter().find(|(stem, _)| {
        name == *stem
            || name
                .strip_prefix(*stem)
                .is_some_and(|s| s.starts_with(b"."))
    })?;
    let suffix = &name[stem.len()..];
    let version = if suffix.is_empty() {
        None
    } else {
        let token = suffix.strip_prefix(b".")?;
        if token.len() > MAX_VERSION_TOKEN_BYTES {
            return None;
        }
        parse_numeric_version(token)
    };
    if !suffix.is_empty() && version.is_none() {
        return None;
    }
    Some(RuntimeLibraryEvidence {
        family: *family,
        evidence: RuntimeLibraryEvidenceKind::MappedLibraryName,
        soname_major: version.map(|version| version.major),
        filename_version: version,
        version_evidence: if version.is_some() {
            RuntimeLibraryVersionEvidence::MappedFilenameOnly
        } else {
            RuntimeLibraryVersionEvidence::NotObserved
        },
    })
}

fn frameworks_from_libraries(
    libraries: &BTreeSet<RuntimeLibraryEvidence>,
) -> BTreeSet<RuntimeFrameworkEvidence> {
    libraries
        .iter()
        .filter_map(|library| {
            let framework = match library.family {
                RuntimeLibraryFamily::TorchCore | RuntimeLibraryFamily::TorchCuda => {
                    RuntimeFramework::Pytorch
                }
                RuntimeLibraryFamily::Tensorflow => RuntimeFramework::Tensorflow,
                RuntimeLibraryFamily::OnnxRuntimeCuda => RuntimeFramework::OnnxRuntime,
                RuntimeLibraryFamily::TensorRt => RuntimeFramework::TensorRt,
                _ => return None,
            };
            Some(RuntimeFrameworkEvidence {
                framework,
                evidence: RuntimeFrameworkEvidenceKind::MappedLibraryName,
            })
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FixedEvidence<T> {
    Observed(T),
    Missing,
    Unavailable,
}

impl<T> FixedEvidence<T> {
    fn and_then<U>(self, convert: impl FnOnce(T) -> FixedEvidence<U>) -> FixedEvidence<U> {
        match self {
            Self::Observed(value) => convert(value),
            Self::Missing => FixedEvidence::Missing,
            Self::Unavailable => FixedEvidence::Unavailable,
        }
    }
}

fn parse_sys_driver_version(bytes: &[u8]) -> FixedEvidence<RuntimeNumericVersion> {
    let token = trim_ascii(bytes);
    if token.is_empty() || token.len() > MAX_VERSION_TOKEN_BYTES {
        return FixedEvidence::Unavailable;
    }
    parse_numeric_version(token).map_or(FixedEvidence::Unavailable, FixedEvidence::Observed)
}

fn parse_proc_driver_version(bytes: &[u8]) -> FixedEvidence<RuntimeNumericVersion> {
    for line in bytes.split(|byte| *byte == b'\n') {
        let line = trim_ascii(line);
        let Some(remainder) = line.strip_prefix(b"NVRM version:") else {
            continue;
        };
        for token in remainder
            .split(u8::is_ascii_whitespace)
            .filter(|token| !token.is_empty())
        {
            if token.len() <= MAX_VERSION_TOKEN_BYTES
                && token.contains(&b'.')
                && let Some(version) = parse_numeric_version(token)
            {
                return FixedEvidence::Observed(version);
            }
        }
        return FixedEvidence::Unavailable;
    }
    FixedEvidence::Unavailable
}

fn parse_numeric_version(value: &[u8]) -> Option<RuntimeNumericVersion> {
    if value.is_empty() || value.len() > MAX_VERSION_TOKEN_BYTES {
        return None;
    }
    let components = value
        .split(|byte| *byte == b'.')
        .map(parse_ascii_u32)
        .collect::<Option<Vec<_>>>()?;
    RuntimeNumericVersion::from_components(&components)
}

fn parse_positive_ascii_u32(value: &[u8]) -> Option<u32> {
    parse_ascii_u32(value).filter(|value| *value > 0)
}

fn parse_positive_ascii_u64(value: &[u8]) -> Option<u64> {
    parse_ascii_u64(value).filter(|value| *value > 0)
}

fn parse_ascii_u32(value: &[u8]) -> Option<u32> {
    parse_ascii_u64(value).and_then(|value| u32::try_from(value).ok())
}

fn parse_ascii_u64(value: &[u8]) -> Option<u64> {
    if value.is_empty() || !value.iter().all(u8::is_ascii_digit) {
        return None;
    }
    value.iter().try_fold(0_u64, |accumulator, byte| {
        accumulator
            .checked_mul(10)?
            .checked_add(u64::from(byte - b'0'))
    })
}

fn basename(value: &[u8]) -> &[u8] {
    value
        .iter()
        .rposition(|byte| *byte == b'/')
        .map_or(value, |position| &value[position + 1..])
}

fn trim_ascii(mut value: &[u8]) -> &[u8] {
    while value.first().is_some_and(u8::is_ascii_whitespace) {
        value = &value[1..];
    }
    while value.last().is_some_and(u8::is_ascii_whitespace) {
        value = &value[..value.len() - 1];
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmdline_splitting_preserves_empty_arguments_without_a_trailing_phantom() {
        assert_eq!(split_cmdline(b"").unwrap(), Vec::<&[u8]>::new());
        assert_eq!(split_cmdline(b"foo").unwrap(), vec![b"foo".as_slice()]);
        assert_eq!(split_cmdline(b"foo\0").unwrap(), vec![b"foo".as_slice()]);
        assert_eq!(
            split_cmdline(b"foo\0bar\0").unwrap(),
            vec![b"foo".as_slice(), b"bar".as_slice()]
        );
        assert_eq!(split_cmdline(b"\0").unwrap(), vec![b"".as_slice()]);
        assert_eq!(
            split_cmdline(b"foo\0\0bar").unwrap(),
            vec![b"foo".as_slice(), b"".as_slice(), b"bar".as_slice()]
        );
    }

    #[test]
    fn cmdline_and_maps_parsers_enforce_count_length_and_fact_limits() {
        let mut too_many_args = Vec::new();
        for _ in 0..=MAX_CMDLINE_ARGS {
            too_many_args.extend_from_slice(b"x\0");
        }
        assert_eq!(
            split_cmdline(&too_many_args),
            Err(RuntimeSourceReason::LimitExceeded)
        );
        assert_eq!(
            split_cmdline(&vec![b'x'; MAX_ARG_BYTES + 1]),
            Err(RuntimeSourceReason::LimitExceeded)
        );

        let overlong_line = vec![b'x'; MAX_MAP_LINE_BYTES + 1];
        assert!(matches!(
            parse_maps(&overlong_line),
            Err(RuntimeSourceReason::LimitExceeded)
        ));
        assert!(matches!(
            parse_maps(b"one two three four\n"),
            Err(RuntimeSourceReason::Malformed)
        ));

        let mut too_many_facts = Vec::new();
        for version in 1..=MAX_RUNTIME_FINGERPRINT_LIBRARIES_PER_PROCESS + 1 {
            too_many_facts.extend_from_slice(
                format!("7f00-7f01 r-xp 00000000 00:00 0 /x/libcudart.so.{version}\n").as_bytes(),
            );
        }
        assert!(matches!(
            parse_maps(&too_many_facts),
            Err(RuntimeSourceReason::LimitExceeded)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn changed_cmdline_discards_all_argv_derived_evidence() {
        let (source, parsed) = reduce_cmdline(
            Ok(b"vllm\0--dtype\0bf16\0".to_vec()),
            Ok(b"vllm\0--dtype\0fp16\0".to_vec()),
        );
        assert_eq!(source.state, RuntimeSourceState::Unavailable);
        assert_eq!(
            source.reason,
            Some(RuntimeSourceReason::ChangedDuringCollection)
        );
        assert!(parsed.is_none());
    }

    #[test]
    fn engine_and_launch_parsing_is_typed_and_duplicate_flags_are_ambiguous() {
        let parsed = parse_cmdline(
            b"/usr/bin/python3\0-m\0vllm.entrypoints.openai.api_server\0--model\0/private/model\0--tensor-parallel-size=8\0--dtype\0bf16\0--quantization\0awq\0--kv-cache-dtype\0nonsense\0",
        )
        .unwrap();
        assert!(parsed.engine_candidates.contains(&RuntimeEngineEvidence {
            engine: RuntimeEngine::Vllm,
            evidence: RuntimeEngineEvidenceKind::ArgvPythonModule,
        }));
        assert_eq!(
            parsed.launch_observation.model_reference_present,
            RuntimeObservation::observed(true)
        );
        assert_eq!(
            parsed.launch_observation.tensor_parallel_size,
            RuntimeObservation::observed(8)
        );
        assert_eq!(
            parsed.launch_observation.declared_dtype,
            RuntimeObservation::observed(RuntimeCanonicalDtype::Bf16)
        );
        assert_eq!(
            parsed.launch_observation.declared_quantization,
            RuntimeObservation::observed(RuntimeCanonicalQuantization::Awq)
        );
        assert_eq!(
            parsed.launch_observation.declared_kv_cache_dtype,
            RuntimeObservation::present_unparsed()
        );

        let duplicate =
            parse_cmdline(b"vllm\0--tensor-parallel-size\08\0--tensor-parallel-size=8\0").unwrap();
        assert_eq!(
            duplicate.launch_observation.tensor_parallel_size,
            RuntimeObservation::ambiguous()
        );
    }

    #[test]
    fn python_interface_and_argument_terminators_prevent_false_engine_evidence() {
        let unknown = parse_cmdline(b"python3\0-m\0private.module\0--dtype\0bf16\0").unwrap();
        assert!(unknown.engine_candidates.is_empty());
        assert_eq!(
            unknown.launch_observation.declared_dtype,
            RuntimeObservation::not_observed()
        );

        let script_argument = parse_cmdline(
            b"python3\0worker.py\0-m\0vllm.entrypoints.openai.api_server\0--dtype\0bf16\0",
        )
        .unwrap();
        assert!(script_argument.engine_candidates.is_empty());

        let command_argument = parse_cmdline(
            b"python3\0-c\0print(1)\0-m\0vllm.entrypoints.openai.api_server\0--dtype\0bf16\0",
        )
        .unwrap();
        assert!(command_argument.engine_candidates.is_empty());

        let optioned_module = parse_cmdline(
            b"python3\0-u\0-X\0dev\0-m\0vllm.entrypoints.openai.api_server\0--dtype\0bf16\0",
        )
        .unwrap();
        assert_eq!(optioned_module.engine_candidates.len(), 1);
        assert_eq!(
            optioned_module.launch_observation.declared_dtype,
            RuntimeObservation::observed(RuntimeCanonicalDtype::Bf16)
        );

        let interpreter_option_value =
            parse_cmdline(b"python3\0-X\0--dtype=bf16\0-m\0vllm.entrypoints.openai.api_server\0")
                .unwrap();
        assert_eq!(interpreter_option_value.engine_candidates.len(), 1);
        assert_eq!(
            interpreter_option_value.launch_observation.declared_dtype,
            RuntimeObservation::not_observed()
        );

        let terminated =
            parse_cmdline(b"vllm\0--\0--tensor-parallel-size\08\0--dtype\0bf16\0").unwrap();
        assert_eq!(
            terminated.launch_observation.tensor_parallel_size,
            RuntimeObservation::not_observed()
        );
        assert_eq!(
            terminated.launch_observation.declared_dtype,
            RuntimeObservation::not_observed()
        );
    }

    #[test]
    fn maps_parser_handles_spaces_deleted_suffixes_and_strict_library_names() {
        let maps = b"7f00-7f01 r-xp 00000000 00:00 0 /private path/libcudart.so.12.8.90 (deleted)\n7f01-7f02 r--p 00000000 00:00 0 /x/libtorch_cuda.so\n7f02-7f03 r--p 00000000 00:00 0 /x/libcuda.so.backup\n7f03-7f04 rw-p 00000000 00:00 0\n";
        let parsed = parse_maps(maps).unwrap();
        assert_eq!(parsed.records, 4);
        assert!(parsed.libraries.contains(&RuntimeLibraryEvidence {
            family: RuntimeLibraryFamily::CudaRuntime,
            evidence: RuntimeLibraryEvidenceKind::MappedLibraryName,
            soname_major: Some(12),
            filename_version: Some(RuntimeNumericVersion::major_minor_patch(12, 8, 90)),
            version_evidence: RuntimeLibraryVersionEvidence::MappedFilenameOnly,
        }));
        assert!(parsed.frameworks.contains(&RuntimeFrameworkEvidence {
            framework: RuntimeFramework::Pytorch,
            evidence: RuntimeFrameworkEvidenceKind::MappedLibraryName,
        }));
        assert!(
            !parsed
                .libraries
                .iter()
                .any(|library| library.family == RuntimeLibraryFamily::CudaDriver)
        );
    }

    #[test]
    fn stat_parser_uses_the_final_comm_delimiter_and_field_22() {
        let stat = b"42 (odd ) name) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 999 20\n";
        assert_eq!(parse_proc_stat(stat, 42).unwrap(), 999);
        assert!(parse_proc_stat(stat, 41).is_err());
    }

    #[test]
    fn driver_parsers_retain_only_strict_numeric_versions() {
        assert_eq!(
            parse_sys_driver_version(b"580.95.05\n"),
            FixedEvidence::Observed(RuntimeNumericVersion::major_minor_patch(580, 95, 5))
        );
        assert_eq!(
            parse_proc_driver_version(
                b"NVRM version: NVIDIA UNIX Open Kernel Module for x86_64 580.95.05 Release Build\nGCC version: 13.2.0\n"
            ),
            FixedEvidence::Observed(RuntimeNumericVersion::major_minor_patch(580, 95, 5))
        );
        assert_eq!(
            parse_sys_driver_version(b"580.95.private"),
            FixedEvidence::Unavailable
        );
    }
}
