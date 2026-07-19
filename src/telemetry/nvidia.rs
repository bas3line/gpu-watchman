//! Read-only NVIDIA telemetry collection through the stable `nvidia-smi` CLI.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::domain::{Gpu, GpuProcess, Report, SourceState, SourceStatus};
use crate::telemetry::command;
use crate::telemetry::process;
use anyhow::{Context, Result, bail};
use chrono::Utc;

pub const SOURCE_INVENTORY: &str = "nvidia.inventory";
pub const SOURCE_PROCESSES: &str = "nvidia.processes";
pub const SOURCE_OPTIONAL: &str = "nvidia.optional";
pub const SOURCE_TOPOLOGY: &str = "nvidia.topology";
pub const SOURCE_XID: &str = "kernel.xid";

const MAX_SOURCE_ERROR_BYTES: usize = 512;

const GPU_FIELDS: &[&str] = &[
    "index",
    "name",
    "uuid",
    "driver_version",
    "pci.bus_id",
    "pstate",
    "temperature.gpu",
    "fan.speed",
    "power.draw",
    "power.limit",
    "clocks.current.graphics",
    "clocks.current.memory",
    "clocks.max.graphics",
    "clocks.max.memory",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
    "utilization.memory",
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "pcie.link.width.max",
    "compute_mode",
    "persistence_mode",
    "ecc.mode.current",
    "ecc.errors.corrected.volatile.total",
    "ecc.errors.uncorrected.volatile.total",
];

const RETIRED_PAGE_FIELDS: &[&str] = &[
    "retired_pages.single_bit_ecc.count",
    "retired_pages.double_bit_ecc.count",
];

const OPTIONAL_FIELDS: &[(&str, &str)] = &[
    ("mig.mode.current", ""),
    ("clocks_event_reasons.sw_power_cap", "software power cap"),
    (
        "clocks_event_reasons.hw_thermal_slowdown",
        "hardware thermal slowdown",
    ),
    (
        "clocks_event_reasons.sw_thermal_slowdown",
        "software thermal slowdown",
    ),
    (
        "clocks_event_reasons.hw_power_brake_slowdown",
        "external power brake",
    ),
    ("clocks_event_reasons.hw_slowdown", "hardware slowdown"),
];

#[derive(Debug, Default)]
struct TopologyCache {
    key: String,
    value: String,
}

#[derive(Debug)]
struct Observed<T> {
    value: T,
    source: SourceStatus,
}

impl<T: Default> Observed<T> {
    fn unavailable(name: &str, error: impl std::fmt::Display) -> Self {
        Self {
            value: T::default(),
            source: SourceStatus::failed(name, SourceState::Unavailable, 0, 0, source_error(error)),
        }
    }
}

#[derive(Debug, Default)]
struct Parsed<T> {
    value: T,
    rejected: usize,
}

#[derive(Debug)]
pub struct NvidiaCollector {
    command: PathBuf,
    command_timeout: Duration,
    topology: Mutex<TopologyCache>,
    collect_xid: bool,
}

impl Default for NvidiaCollector {
    fn default() -> Self {
        Self::new("nvidia-smi", Duration::from_secs(3))
    }
}

impl NvidiaCollector {
    pub fn new(command: impl Into<PathBuf>, command_timeout: Duration) -> Self {
        Self {
            command: command.into(),
            command_timeout,
            topology: Mutex::new(TopologyCache::default()),
            collect_xid: cfg!(target_os = "linux"),
        }
    }

    #[must_use]
    pub fn without_xid(mut self) -> Self {
        self.collect_xid = false;
        self
    }

    pub fn command(&self) -> &Path {
        &self.command
    }

    pub fn collect(&self) -> Result<Report> {
        let inventory_started = Instant::now();
        let query = format!("--query-gpu={}", GPU_FIELDS.join(","));
        let output = self
            .run(&[&query, "--format=csv,noheader,nounits"])
            .with_context(|| {
                format!(
                    "could not query GPUs with {}; install a working NVIDIA driver and ensure nvidia-smi is in PATH",
                    self.command.display()
                )
            })?;
        let mut gpus = parse_gpus(&output)?;
        if gpus.is_empty() {
            bail!("nvidia-smi returned no GPUs");
        }
        let inventory = SourceStatus::ok(
            SOURCE_INVENTORY,
            elapsed_millis(inventory_started),
            gpus.len(),
        );

        // Once inventory is known, all optional sources are independent. Run
        // them together so an inaccessible log source or optional query cannot
        // multiply the collection deadline.
        let (processes, optional, topology, xid_events) = std::thread::scope(|scope| {
            let process_task = scope.spawn(|| self.collect_processes());
            let optional_task = scope.spawn(|| self.collect_optional(gpus.len()));
            let topology_task = scope.spawn(|| self.collect_topology(&gpus));
            let xid_task = scope.spawn(|| {
                if self.collect_xid {
                    self.collect_xid_events()
                } else {
                    Observed {
                        value: Vec::new(),
                        source: SourceStatus::skipped(SOURCE_XID),
                    }
                }
            });
            (
                process_task
                    .join()
                    .unwrap_or_else(|_| Observed::unavailable(SOURCE_PROCESSES, "worker panicked")),
                optional_task
                    .join()
                    .unwrap_or_else(|_| Observed::unavailable(SOURCE_OPTIONAL, "worker panicked")),
                topology_task
                    .join()
                    .unwrap_or_else(|_| Observed::unavailable(SOURCE_TOPOLOGY, "worker panicked")),
                xid_task
                    .join()
                    .unwrap_or_else(|_| Observed::unavailable(SOURCE_XID, "worker panicked")),
            )
        });
        for gpu in &mut gpus {
            gpu.processes = processes.value.get(&gpu.uuid).cloned().unwrap_or_default();
            if let Some(details) = optional.value.get(&gpu.uuid) {
                gpu.mig_mode.clone_from(&details.mig_mode);
                gpu.throttle_reasons.clone_from(&details.throttle_reasons);
                gpu.retired_pages = details.retired_pages;
            }
        }
        let mut sources = vec![
            inventory,
            processes.source,
            optional.source,
            topology.source,
            xid_events.source,
        ];
        sources.sort_by(|left, right| left.name.cmp(&right.name));

        Ok(Report {
            collected_at: Utc::now(),
            topology: topology.value,
            xid_events: xid_events.value,
            gpus,
            sources,
            ..Report::default()
        })
    }

    pub fn driver_version(&self) -> Result<String> {
        self.run(&[
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ])
        .map(|output| output.lines().next().unwrap_or_default().trim().to_owned())
    }

    fn run(&self, args: &[&str]) -> Result<String> {
        command::run(&self.command, args, self.command_timeout)
    }

    fn collect_processes(&self) -> Observed<HashMap<String, Vec<GpuProcess>>> {
        let started = Instant::now();
        let mut by_gpu: HashMap<String, Vec<GpuProcess>> = HashMap::new();
        let mut seen = HashSet::new();
        let outputs = std::thread::scope(|scope| {
            let compute = scope.spawn(|| self.collect_process_kind("compute"));
            let graphics = scope.spawn(|| self.collect_process_kind("graphics"));
            [
                (
                    "compute",
                    compute
                        .join()
                        .map_err(|_| "worker panicked".to_owned())
                        .and_then(|result| result.map_err(source_error)),
                ),
                (
                    "graphics",
                    graphics
                        .join()
                        .map_err(|_| "worker panicked".to_owned())
                        .and_then(|result| result.map_err(source_error)),
                ),
            ]
        });
        let mut successes = 0;
        let mut rejected = 0;
        let mut errors = Vec::new();
        let local_users = process::local_users();
        for (kind, output) in outputs {
            match output {
                Ok(output) => {
                    successes += 1;
                    let parsed = parse_processes(&output);
                    rejected += parsed.rejected;
                    for (uuid, mut process) in parsed.value {
                        if !seen.insert((uuid.clone(), process.pid)) {
                            continue;
                        }
                        process::enrich(&mut process, &local_users);
                        by_gpu.entry(uuid).or_default().push(process);
                    }
                }
                Err(error) => errors.push(format!("{kind}: {error}")),
            }
        }
        if rejected > 0 {
            errors.push(format!("rejected {rejected} malformed process row(s)"));
        }
        for processes in by_gpu.values_mut() {
            processes.sort_by(|left, right| {
                right
                    .memory_mib
                    .cmp(&left.memory_mib)
                    .then(left.pid.cmp(&right.pid))
            });
        }
        let records = by_gpu.values().map(Vec::len).sum::<usize>();
        let state = if successes == 0 {
            SourceState::Unavailable
        } else if successes < 2 || rejected > 0 {
            SourceState::Partial
        } else {
            SourceState::Ok
        };
        let source = if state == SourceState::Ok {
            SourceStatus::ok(SOURCE_PROCESSES, elapsed_millis(started), records)
        } else {
            SourceStatus::failed(
                SOURCE_PROCESSES,
                state,
                elapsed_millis(started),
                records,
                source_error(errors.join("; ")),
            )
        };
        Observed {
            value: by_gpu,
            source,
        }
    }

    fn collect_process_kind(&self, kind: &str) -> Result<String> {
        let query = format!("--query-{kind}-apps=gpu_uuid,pid,process_name,used_memory");
        self.run(&[&query, "--format=csv,noheader,nounits"])
    }

    fn collect_optional(&self, expected_records: usize) -> Observed<HashMap<String, OptionalGpu>> {
        let started = Instant::now();
        let fields = std::iter::once("uuid")
            .chain(OPTIONAL_FIELDS.iter().map(|(field, _)| *field))
            .collect::<Vec<_>>();
        let query = format!("--query-gpu={}", fields.join(","));
        let output = match self.run(&[&query, "--format=csv,noheader,nounits"]) {
            Ok(output) => output,
            Err(error) => {
                return Observed {
                    value: HashMap::new(),
                    source: SourceStatus::failed(
                        SOURCE_OPTIONAL,
                        SourceState::Unavailable,
                        elapsed_millis(started),
                        0,
                        source_error(error),
                    ),
                };
            }
        };
        let mut parsed = parse_optional(&output);
        let mut errors = Vec::new();
        let mut retired_successes = 0;
        let mut retired_rejected = 0;
        for field in RETIRED_PAGE_FIELDS {
            let query = format!("--query-gpu=uuid,{field}");
            match self.run(&[&query, "--format=csv,noheader,nounits"]) {
                Ok(output) => {
                    let retired = parse_retired_pages(&output);
                    retired_rejected += retired.rejected;
                    if retired.rejected == 0 && retired.value.len() == expected_records {
                        retired_successes += 1;
                    } else {
                        errors.push(format!(
                            "{field}: expected {expected_records} GPU record(s), received {}, rejected {} malformed row(s)",
                            retired.value.len(),
                            retired.rejected
                        ));
                    }
                    for (uuid, pages) in retired.value {
                        let retired_pages = parsed
                            .value
                            .get(&uuid)
                            .map_or(pages, |details| details.retired_pages.saturating_add(pages));
                        parsed.value.entry(uuid).or_default().retired_pages = retired_pages;
                    }
                }
                Err(error) => errors.push(format!("{field}: {}", source_error(error))),
            }
        }
        let records = parsed.value.len();
        let source = if parsed.rejected == 0
            && retired_rejected == 0
            && retired_successes == RETIRED_PAGE_FIELDS.len()
            && records == expected_records
        {
            SourceStatus::ok(SOURCE_OPTIONAL, elapsed_millis(started), records)
        } else {
            if parsed.rejected > 0 {
                errors.push(format!(
                    "rejected {} malformed optional row(s)",
                    parsed.rejected
                ));
            }
            if records != expected_records {
                errors.push(format!(
                    "expected {expected_records} GPU record(s), received {records}"
                ));
            }
            SourceStatus::failed(
                SOURCE_OPTIONAL,
                SourceState::Partial,
                elapsed_millis(started),
                records,
                errors.join("; "),
            )
        };
        Observed {
            value: parsed.value,
            source,
        }
    }

    fn collect_topology(&self, gpus: &[Gpu]) -> Observed<String> {
        let started = Instant::now();
        let mut uuids = gpus.iter().map(|gpu| gpu.uuid.as_str()).collect::<Vec<_>>();
        uuids.sort_unstable();
        let key = uuids.join("\0");
        if let Ok(cache) = self.topology.lock()
            && cache.key == key
            && !cache.key.is_empty()
        {
            let records = cache.value.lines().count();
            return Observed {
                source: topology_source(elapsed_millis(started), records),
                value: cache.value.clone(),
            };
        }
        let value = match self.run(&["topo", "-m"]) {
            Ok(output) => output.trim().to_owned(),
            Err(error) => {
                return Observed {
                    value: String::new(),
                    source: SourceStatus::failed(
                        SOURCE_TOPOLOGY,
                        SourceState::Unavailable,
                        elapsed_millis(started),
                        0,
                        source_error(error),
                    ),
                };
            }
        };
        if let Ok(mut cache) = self.topology.lock() {
            cache.key = key;
            cache.value.clone_from(&value);
        }
        let records = value.lines().count();
        let source = topology_source(elapsed_millis(started), records);
        Observed { value, source }
    }

    fn collect_xid_events(&self) -> Observed<Vec<String>> {
        let started = Instant::now();
        let candidates: [(&str, &[&str]); 2] = [
            ("journalctl", &["-k", "-n", "300", "--no-pager"]),
            ("dmesg", &["--color=never"]),
        ];
        let mut errors = Vec::new();
        for (command, args) in candidates {
            match command::run(Path::new(command), args, self.command_timeout) {
                Ok(output) => {
                    let events = output
                        .lines()
                        .filter(|line| line.contains("Xid"))
                        .map(str::trim)
                        .map(str::to_owned)
                        .collect::<Vec<_>>();
                    return Observed {
                        source: SourceStatus::ok(SOURCE_XID, elapsed_millis(started), events.len()),
                        value: events,
                    };
                }
                Err(error) => errors.push(format!("{command}: {}", source_error(error))),
            }
        }
        Observed {
            value: Vec::new(),
            source: SourceStatus::failed(
                SOURCE_XID,
                SourceState::Unavailable,
                elapsed_millis(started),
                0,
                source_error(errors.join("; ")),
            ),
        }
    }
}

#[derive(Debug, Default)]
struct OptionalGpu {
    mig_mode: String,
    throttle_reasons: Vec<String>,
    retired_pages: i64,
}

fn topology_source(duration_ms: u64, records: usize) -> SourceStatus {
    if records == 0 {
        SourceStatus::failed(
            SOURCE_TOPOLOGY,
            SourceState::Partial,
            duration_ms,
            0,
            "command returned no topology rows",
        )
    } else {
        SourceStatus::ok(SOURCE_TOPOLOGY, duration_ms, records)
    }
}

pub fn parse_gpus(csv_text: &str) -> Result<Vec<Gpu>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .trim(csv::Trim::All)
        .from_reader(csv_text.as_bytes());
    let mut gpus = Vec::new();
    for row in reader.records() {
        let row = row.context("parse nvidia-smi GPU CSV")?;
        if row.len() != GPU_FIELDS.len() {
            bail!(
                "nvidia-smi returned {} GPU fields; expected {}",
                row.len(),
                GPU_FIELDS.len()
            );
        }
        let value = |index| clean(row.get(index).unwrap_or_default());
        gpus.push(Gpu {
            index: integer(value(0)),
            name: value(1).to_owned(),
            uuid: value(2).to_owned(),
            driver: value(3).to_owned(),
            pci_bus_id: value(4).to_owned(),
            performance_state: value(5).to_owned(),
            temperature_c: sensor_integer(value(6)),
            fan_percent: sensor_integer(value(7)),
            power_draw_w: decimal(value(8)),
            power_limit_w: decimal(value(9)),
            graphics_clock_mhz: integer(value(10)),
            memory_clock_mhz: integer(value(11)),
            max_graphics_clock_mhz: integer(value(12)),
            max_memory_clock_mhz: integer(value(13)),
            memory_total_mib: integer64(value(14)),
            memory_used_mib: integer64(value(15)),
            memory_free_mib: integer64(value(16)),
            gpu_util_percent: integer(value(17)),
            memory_util_percent: integer(value(18)),
            pcie_gen_current: integer(value(19)),
            pcie_gen_max: integer(value(20)),
            pcie_width_current: integer(value(21)),
            pcie_width_max: integer(value(22)),
            compute_mode: value(23).to_owned(),
            persistence_mode: enabled(value(24)),
            ecc_enabled: enabled(value(25)),
            ecc_corrected_volatile: integer64(value(26)),
            ecc_uncorrected_volatile: integer64(value(27)),
            ..Gpu::default()
        });
    }
    Ok(gpus)
}

fn parse_processes(csv_text: &str) -> Parsed<Vec<(String, GpuProcess)>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .trim(csv::Trim::All)
        .flexible(true)
        .from_reader(csv_text.as_bytes());
    let mut parsed: Parsed<Vec<(String, GpuProcess)>> = Parsed::default();
    for row in reader.records() {
        let Ok(row) = row else {
            parsed.rejected += 1;
            continue;
        };
        if row.len() < 4 {
            parsed.rejected += 1;
            continue;
        }
        let uuid = clean(row.get(0).unwrap_or_default());
        let Some(pid) = clean(row.get(1).unwrap_or_default()).parse::<u32>().ok() else {
            parsed.rejected += 1;
            continue;
        };
        if uuid.is_empty() {
            parsed.rejected += 1;
            continue;
        }
        parsed.value.push((
            uuid.to_owned(),
            GpuProcess {
                pid,
                name: clean(row.get(2).unwrap_or_default()).to_owned(),
                memory_mib: integer64(clean(row.get(3).unwrap_or_default())),
                ..GpuProcess::default()
            },
        ));
    }
    parsed
}

fn parse_optional(csv_text: &str) -> Parsed<HashMap<String, OptionalGpu>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .trim(csv::Trim::All)
        .from_reader(csv_text.as_bytes());
    let mut parsed: Parsed<HashMap<String, OptionalGpu>> = Parsed::default();
    for row in reader.records() {
        let Ok(row) = row else {
            parsed.rejected += 1;
            continue;
        };
        if row.len() != OPTIONAL_FIELDS.len() + 1 {
            parsed.rejected += 1;
            continue;
        }
        let Some(uuid) = row.get(0).map(clean).filter(|value| !value.is_empty()) else {
            parsed.rejected += 1;
            continue;
        };
        let mut details = OptionalGpu {
            mig_mode: optional_value(row.get(1).unwrap_or_default()).to_owned(),
            ..OptionalGpu::default()
        };
        for (index, (_, label)) in OPTIONAL_FIELDS.iter().enumerate().skip(1) {
            if optional_value(row.get(index + 1).unwrap_or_default()).eq_ignore_ascii_case("active")
            {
                details.throttle_reasons.push((*label).to_owned());
            }
        }
        parsed.value.insert(uuid.to_owned(), details);
    }
    parsed
}

fn parse_retired_pages(csv_text: &str) -> Parsed<HashMap<String, i64>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .trim(csv::Trim::All)
        .from_reader(csv_text.as_bytes());
    let mut parsed: Parsed<HashMap<String, i64>> = Parsed::default();
    for row in reader.records() {
        let Ok(row) = row else {
            parsed.rejected += 1;
            continue;
        };
        if row.len() != 2 {
            parsed.rejected += 1;
            continue;
        }
        let Some(uuid) = row.get(0).map(clean).filter(|value| !value.is_empty()) else {
            parsed.rejected += 1;
            continue;
        };
        parsed.value.insert(
            uuid.to_owned(),
            integer64(clean(row.get(1).unwrap_or_default())),
        );
    }
    parsed
}

fn elapsed_millis(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn source_error(error: impl std::fmt::Display) -> String {
    let mut value = error
        .to_string()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if value.len() <= MAX_SOURCE_ERROR_BYTES {
        return value;
    }
    let suffix = "...";
    let mut end = MAX_SOURCE_ERROR_BYTES - suffix.len();
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value.truncate(end);
    value.push_str(suffix);
    value
}

fn clean(value: &str) -> &str {
    value.trim().trim_matches('"')
}

fn optional_value(value: &str) -> &str {
    let value = clean(value);
    if value.eq_ignore_ascii_case("N/A") || value.eq_ignore_ascii_case("[Not Supported]") {
        ""
    } else {
        value
    }
}

fn integer(value: &str) -> i32 {
    optional_value(value).parse().unwrap_or_default()
}

fn integer64(value: &str) -> i64 {
    optional_value(value).parse().unwrap_or_default()
}

fn sensor_integer(value: &str) -> i32 {
    if optional_value(value).is_empty() {
        -1
    } else {
        integer(value)
    }
}

fn decimal(value: &str) -> f64 {
    optional_value(value).parse().unwrap_or_default()
}

fn enabled(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "enabled" | "on" | "yes" | "1"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_row() -> String {
        [
            "0",
            "NVIDIA H100",
            "GPU-123",
            "580.1",
            "0000:01:00.0",
            "P0",
            "75",
            "42",
            "500.5",
            "700",
            "1800",
            "2000",
            "2100",
            "2500",
            "81920",
            "40960",
            "40960",
            "99",
            "80",
            "5",
            "5",
            "16",
            "16",
            "Default",
            "Enabled",
            "Enabled",
            "1",
            "0",
        ]
        .join(",")
    }

    #[test]
    fn parses_full_gpu_inventory() {
        let gpus = parse_gpus(&gpu_row()).unwrap();
        let gpu = &gpus[0];
        assert_eq!(gpu.name, "NVIDIA H100");
        assert_eq!(gpu.memory_total_mib, 81_920);
        assert_eq!(gpu.retired_pages, 0);
        assert!(gpu.ecc_enabled);
    }

    #[test]
    fn rejects_driver_schema_drift() {
        let error = parse_gpus("0,H100").unwrap_err().to_string();
        assert!(error.contains("expected 28"));
    }

    #[test]
    fn source_errors_are_single_line_and_byte_bounded() {
        let error = format!("first line\n{}\r\nlast line", "é".repeat(400));
        let sanitized = source_error(error);

        assert!(sanitized.len() <= MAX_SOURCE_ERROR_BYTES);
        assert!(!sanitized.contains(['\n', '\r']));
        assert!(sanitized.starts_with("first line "));
        assert!(sanitized.ends_with("..."));
    }

    #[cfg(unix)]
    #[test]
    fn collection_preserves_partial_and_unavailable_source_evidence() {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let command = directory.path().join("nvidia-smi");
        fs::write(
            &command,
            format!(
                r#"#!/bin/sh
case "$1" in
  --query-gpu=index,*)
    printf '%s\n' '{}'
    ;;
  --query-compute-apps=*)
    printf '%s\n' 'compute accounting failed' 'permission denied' >&2
    exit 1
    ;;
  --query-graphics-apps=*)
    exit 0
    ;;
  --query-gpu=uuid,*)
    printf '%s\n' 'optional fields unavailable' >&2
    exit 1
    ;;
  topo)
    printf '%s\n' 'GPU0 CPU Affinity' 'GPU0 X 0-31'
    ;;
  *)
    exit 1
    ;;
esac
"#,
                gpu_row()
            ),
        )
        .unwrap();
        let mut permissions = fs::metadata(&command).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&command, permissions).unwrap();
        fs::File::open(&command).unwrap().sync_all().unwrap();
        std::thread::sleep(Duration::from_millis(10));

        let report = NvidiaCollector::new(&command, Duration::from_secs(5))
            .without_xid()
            .collect()
            .unwrap();
        let source = |name: &str| {
            report
                .sources
                .iter()
                .find(|source| source.name == name)
                .unwrap()
        };

        assert_eq!(source(SOURCE_INVENTORY).state, SourceState::Ok);
        assert_eq!(source(SOURCE_INVENTORY).records, 1);
        assert_eq!(source(SOURCE_PROCESSES).state, SourceState::Partial);
        assert_eq!(source(SOURCE_PROCESSES).records, 0);
        assert!(
            source(SOURCE_PROCESSES)
                .error
                .as_deref()
                .unwrap()
                .contains("permission denied")
        );
        assert!(
            !source(SOURCE_PROCESSES)
                .error
                .as_deref()
                .unwrap()
                .contains("compute accounting failed")
        );
        assert_eq!(source(SOURCE_OPTIONAL).state, SourceState::Unavailable);
        assert!(
            !source(SOURCE_OPTIONAL)
                .error
                .as_deref()
                .unwrap()
                .contains("optional fields unavailable")
        );
        assert_eq!(source(SOURCE_TOPOLOGY).state, SourceState::Ok);
        assert_eq!(source(SOURCE_TOPOLOGY).records, 2);
        assert_eq!(source(SOURCE_XID).state, SourceState::Skipped);
        assert!(
            report
                .sources
                .windows(2)
                .all(|pair| pair[0].name <= pair[1].name)
        );
    }

    #[cfg(unix)]
    #[test]
    fn collection_tolerates_an_unsupported_retired_page_field() {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let command = directory.path().join("nvidia-smi");
        fs::write(
            &command,
            format!(
                r#"#!/bin/sh
case "$1" in
  --query-gpu=index,*)
    printf '%s\n' '{}'
    ;;
  --query-compute-apps=*|--query-graphics-apps=*)
    exit 0
    ;;
  --query-gpu=uuid,retired_pages.single_bit_ecc.count)
    printf '%s\n' 'GPU-123,2'
    ;;
  --query-gpu=uuid,retired_pages.double_bit_ecc.count)
    printf '%s\n' 'Field "retired_pages.double_bit_ecc.count" is not a valid field to query.'
    exit 2
    ;;
  --query-gpu=uuid,*)
    printf '%s\n' 'GPU-123,Disabled,Not Active,Not Active,Not Active,Not Active,Not Active'
    ;;
  topo)
    printf '%s\n' 'GPU0 CPU Affinity' 'GPU0 X 0-31'
    ;;
  *)
    exit 1
    ;;
esac
"#,
                gpu_row()
            ),
        )
        .unwrap();
        let mut permissions = fs::metadata(&command).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&command, permissions).unwrap();

        let report = NvidiaCollector::new(&command, Duration::from_secs(5))
            .without_xid()
            .collect()
            .unwrap();
        let inventory = report
            .sources
            .iter()
            .find(|source| source.name == SOURCE_INVENTORY)
            .unwrap();
        let optional = report
            .sources
            .iter()
            .find(|source| source.name == SOURCE_OPTIONAL)
            .unwrap();

        assert_eq!(inventory.state, SourceState::Ok);
        assert_eq!(optional.state, SourceState::Partial);
        assert_eq!(report.gpus[0].retired_pages, 2);
        assert!(
            optional
                .error
                .as_deref()
                .unwrap()
                .contains("driver query field is unavailable")
        );
    }
}
