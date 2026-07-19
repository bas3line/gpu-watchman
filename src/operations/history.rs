//! Secure NDJSON persistence and offline operational history analysis.

use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::domain::{
    MAX_RUNTIME_HISTOGRAM_BUCKETS, MAX_RUNTIME_HISTOGRAM_SERIES, Report, SCHEMA_VERSION, Severity,
    SourceState,
};
use crate::presentation::safe_inline;
use crate::security::{open_read_nonblocking, reject_permissive_acl};

const MAX_HISTORY_RECORD_BYTES: usize = 64 << 20;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HistorySummary {
    pub records: usize,
    pub first_sample: Option<DateTime<Utc>>,
    pub last_sample: Option<DateTime<Utc>>,
    pub hosts: Vec<String>,
    pub max_vram_used_mib: Option<i64>,
    pub vram_used_samples: usize,
    pub max_gpu_util_percent: Option<i32>,
    pub gpu_util_samples: usize,
    pub max_temperature_c: Option<i32>,
    pub temperature_samples: usize,
    pub max_requests_waiting: Option<f64>,
    pub requests_waiting_samples: usize,
    pub max_kv_cache_usage_percent: Option<f64>,
    pub kv_cache_usage_samples: usize,
    pub max_requests_per_second: Option<f64>,
    pub requests_per_second_samples: usize,
    pub max_generation_tokens_per_second: Option<f64>,
    pub generation_tokens_per_second_samples: usize,
    pub max_request_errors_per_second: Option<f64>,
    pub request_errors_per_second_samples: usize,
    pub max_time_to_first_token_ms: Option<f64>,
    pub time_to_first_token_samples: usize,
    pub endpoint_samples: usize,
    pub endpoint_successes: usize,
    pub endpoint_availability_percent: Option<f64>,
    pub source_samples: usize,
    pub source_ok_samples: usize,
    pub source_availability_percent: Option<f64>,
    pub source_non_ok_counts: BTreeMap<String, usize>,
    pub critical_samples: usize,
    pub warning_samples: usize,
    pub finding_counts: BTreeMap<String, usize>,
}

pub fn append(path: &Path, report: &Report) -> Result<()> {
    let mut writer = BoundedWriter::new(MAX_HISTORY_RECORD_BYTES.saturating_sub(1));
    if let Err(error) = serde_json::to_writer(&mut writer, report) {
        if writer.exceeded {
            bail!("history record exceeds the 64 MiB limit");
        }
        return Err(error).context("encode history report");
    }
    let mut record = writer.bytes;
    record.push(b'\n');
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let mut file = options
        .open(path)
        .with_context(|| format!("open history {}", path.display()))?;
    validate_append_target(path, &file)?;
    file.write_all(&record).context("write history record")?;
    file.flush().context("flush history")?;
    Ok(())
}

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: usize,
    exceeded: bool,
}

impl BoundedWriter {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(limit.min(64 * 1024)),
            limit,
            exceeded: false,
        }
    }
}

impl Write for BoundedWriter {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        if buffer.len() > self.limit.saturating_sub(self.bytes.len()) {
            self.exceeded = true;
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "history record exceeds the 64 MiB limit",
            ));
        }
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn validate_append_target(path: &Path, file: &File) -> Result<()> {
    let metadata = file
        .metadata()
        .with_context(|| format!("inspect history {}", path.display()))?;
    if !metadata.is_file() {
        bail!("history path {} is not a regular file", path.display());
    }
    if std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect history path {}", path.display()))?
        .file_type()
        .is_symlink()
    {
        bail!(
            "history path {} must not be a symbolic link",
            path.display()
        );
    }
    validate_private_history_file(path, &metadata)
}

#[cfg(unix)]
fn validate_private_history_file(path: &Path, metadata: &std::fs::Metadata) -> Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let path_metadata = std::fs::metadata(path)
        .with_context(|| format!("inspect resolved history {}", path.display()))?;
    if metadata.dev() != path_metadata.dev() || metadata.ino() != path_metadata.ino() {
        bail!("history path changed while it was being opened");
    }
    if metadata.permissions().mode() & 0o077 != 0 {
        bail!(
            "history file {} must not grant group or other permissions",
            path.display()
        );
    }
    let current_uid = uzers::get_current_uid();
    if !matches!(metadata.uid(), 0) && metadata.uid() != current_uid {
        bail!(
            "history file {} must be owned by the current user or root",
            path.display()
        );
    }
    reject_permissive_acl(path, "history file")?;
    Ok(())
}

#[cfg(not(unix))]
fn validate_private_history_file(_: &Path, _: &std::fs::Metadata) -> Result<()> {
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
pub fn analyze(path: &Path) -> Result<HistorySummary> {
    let file = open_read_nonblocking(path, false)
        .with_context(|| format!("open history {}", path.display()))?;
    if !file
        .metadata()
        .with_context(|| format!("inspect history {}", path.display()))?
        .is_file()
    {
        bail!("history path {} is not a regular file", path.display());
    }
    let mut summary = HistorySummary::default();
    let mut hosts = HashSet::new();
    let mut reader = BufReader::new(file);
    let mut line = Vec::new();
    let mut line_number = 0_usize;
    let mut schema_version = None;
    while read_bounded_line(&mut reader, &mut line, MAX_HISTORY_RECORD_BYTES)
        .with_context(|| format!("read history line {}", line_number + 1))?
    {
        line_number += 1;
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let report: Report = serde_json::from_slice(&line)
            .with_context(|| format!("parse history line {line_number}"))?;
        validate_report(&report, line_number, &mut schema_version)?;
        observe_report(&mut summary, &mut hosts, &report);
    }
    if summary.records == 0 {
        bail!("history {} contains no reports", path.display());
    }
    summary.hosts = hosts.into_iter().collect();
    summary.hosts.sort();
    if summary.endpoint_samples > 0 {
        summary.endpoint_availability_percent =
            Some(summary.endpoint_successes as f64 * 100.0 / summary.endpoint_samples as f64);
    }
    if summary.source_samples > 0 {
        summary.source_availability_percent =
            Some(summary.source_ok_samples as f64 * 100.0 / summary.source_samples as f64);
    }
    Ok(summary)
}

#[allow(clippy::cast_precision_loss)]
fn observe_report(summary: &mut HistorySummary, hosts: &mut HashSet<String>, report: &Report) {
    summary.records += 1;
    summary.first_sample = Some(summary.first_sample.map_or(report.collected_at, |current| {
        current.min(report.collected_at)
    }));
    summary.last_sample = Some(summary.last_sample.map_or(report.collected_at, |current| {
        current.max(report.collected_at)
    }));
    if !report.host.hostname.is_empty() {
        hosts.insert(report.host.hostname.clone());
    }
    for gpu in &report.gpus {
        update_max(
            &mut summary.max_vram_used_mib,
            &mut summary.vram_used_samples,
            Some(gpu.memory_used_mib),
        );
        update_max(
            &mut summary.max_gpu_util_percent,
            &mut summary.gpu_util_samples,
            (gpu.gpu_util_percent >= 0).then_some(gpu.gpu_util_percent),
        );
        update_max(
            &mut summary.max_temperature_c,
            &mut summary.temperature_samples,
            (gpu.temperature_c >= 0).then_some(gpu.temperature_c),
        );
    }
    for endpoint in &report.endpoints {
        summary.endpoint_samples += 1;
        summary.endpoint_successes += usize::from(endpoint.reachable);
        if let Some(waiting) = endpoint.requests_waiting.filter(|value| value.is_finite()) {
            update_max(
                &mut summary.max_requests_waiting,
                &mut summary.requests_waiting_samples,
                Some(waiting),
            );
        }
        if let Some(usage) = endpoint
            .kv_cache_usage_percent
            .filter(|value| value.is_finite())
        {
            update_max(
                &mut summary.max_kv_cache_usage_percent,
                &mut summary.kv_cache_usage_samples,
                Some(usage),
            );
        }
        update_max(
            &mut summary.max_requests_per_second,
            &mut summary.requests_per_second_samples,
            endpoint.rates.requests_per_second,
        );
        update_max(
            &mut summary.max_generation_tokens_per_second,
            &mut summary.generation_tokens_per_second_samples,
            endpoint.rates.generation_tokens_per_second,
        );
        update_max(
            &mut summary.max_request_errors_per_second,
            &mut summary.request_errors_per_second_samples,
            endpoint.rates.request_errors_per_second,
        );
        update_max(
            &mut summary.max_time_to_first_token_ms,
            &mut summary.time_to_first_token_samples,
            endpoint.rates.mean_time_to_first_token_ms,
        );
    }
    for source in &report.sources {
        summary.source_samples += 1;
        if source.state == crate::domain::SourceState::Ok {
            summary.source_ok_samples += 1;
        } else {
            *summary
                .source_non_ok_counts
                .entry(format!("{}:{}", source.name, source.state))
                .or_default() += 1;
        }
    }
    let mut has_critical = false;
    let mut has_warning = false;
    for finding in &report.findings {
        *summary
            .finding_counts
            .entry(finding.code.clone())
            .or_default() += 1;
        has_critical |= finding.severity == Severity::Critical;
        has_warning |= finding.severity == Severity::Warning;
    }
    summary.critical_samples += usize::from(has_critical);
    summary.warning_samples += usize::from(has_warning);
}

fn validate_report(
    report: &Report,
    line_number: usize,
    history_schema_version: &mut Option<u32>,
) -> Result<()> {
    if report.schema_version == 0 || report.schema_version > SCHEMA_VERSION {
        bail!(
            "history line {line_number} uses unsupported schema version {} (this build supports 1..={SCHEMA_VERSION})",
            report.schema_version
        );
    }
    if let Some(expected) = history_schema_version {
        if report.schema_version != *expected {
            bail!(
                "history line {line_number} mixes schema version {} with version {expected} from earlier records",
                report.schema_version
            );
        }
    } else {
        *history_schema_version = Some(report.schema_version);
    }

    validate_summary(report, line_number)?;
    validate_gpus(report, line_number)?;
    validate_endpoints(report, line_number)?;
    validate_sources(report, line_number)?;
    Ok(())
}

fn validate_summary(report: &Report, line_number: usize) -> Result<()> {
    let processes = report
        .gpus
        .iter()
        .try_fold(0_usize, |count, gpu| count.checked_add(gpu.processes.len()))
        .ok_or_else(|| anyhow::anyhow!("history line {line_number} process count overflows"))?;
    let vram_used_mib = report
        .gpus
        .iter()
        .try_fold(0_i64, |total, gpu| total.checked_add(gpu.memory_used_mib));
    let vram_total_mib = report
        .gpus
        .iter()
        .try_fold(0_i64, |total, gpu| total.checked_add(gpu.memory_total_mib));
    let Some(vram_used_mib) = vram_used_mib else {
        bail!("history line {line_number} used VRAM total overflows");
    };
    let Some(vram_total_mib) = vram_total_mib else {
        bail!("history line {line_number} total VRAM overflows");
    };
    let active_gpus = report.gpus.iter().filter(|gpu| gpu.is_active()).count();
    let endpoints_up = report
        .endpoints
        .iter()
        .filter(|endpoint| endpoint.reachable)
        .count();
    let critical_findings = report
        .findings
        .iter()
        .filter(|finding| finding.severity == Severity::Critical)
        .count();
    let warning_findings = report
        .findings
        .iter()
        .filter(|finding| finding.severity == Severity::Warning)
        .count();
    let info_findings = report
        .findings
        .iter()
        .filter(|finding| finding.severity == Severity::Info)
        .count();

    let summary_matches = report.summary.gpus == report.gpus.len()
        && report.summary.active_gpus == active_gpus
        && report.summary.processes == processes
        && report.summary.endpoints == report.endpoints.len()
        && report.summary.endpoints_up == endpoints_up
        && report.summary.vram_used_mib == vram_used_mib
        && report.summary.vram_total_mib == vram_total_mib
        && report.summary.critical_findings == critical_findings
        && report.summary.warning_findings == warning_findings
        && report.summary.info_findings == info_findings;
    if !summary_matches {
        bail!("history line {line_number} has a summary that does not match its records");
    }

    let expected_status = if critical_findings > 0 {
        "critical"
    } else if warning_findings > 0 {
        "warning"
    } else {
        "healthy"
    };
    let is_empty_default = report.status == "unknown"
        && report.gpus.is_empty()
        && report.endpoints.is_empty()
        && report.findings.is_empty();
    if report.status != expected_status && !is_empty_default {
        bail!(
            "history line {line_number} status {:?} does not match derived status {expected_status:?}",
            report.status
        );
    }
    Ok(())
}

fn validate_gpus(report: &Report, line_number: usize) -> Result<()> {
    let mut indexes = HashSet::new();
    for gpu in &report.gpus {
        if gpu.index < 0 || !indexes.insert(gpu.index) {
            bail!(
                "history line {line_number} has an invalid or duplicate GPU index {}",
                gpu.index
            );
        }
        if gpu.memory_total_mib < 0
            || gpu.memory_used_mib < 0
            || gpu.memory_free_mib < 0
            || (gpu.memory_total_mib > 0
                && (gpu.memory_used_mib > gpu.memory_total_mib
                    || gpu.memory_free_mib > gpu.memory_total_mib))
        {
            bail!("history line {line_number} has invalid GPU memory values");
        }
        if gpu.temperature_c < -1
            || !(-1..=100).contains(&gpu.fan_percent)
            || !(-1..=100).contains(&gpu.gpu_util_percent)
            || !(-1..=100).contains(&gpu.memory_util_percent)
        {
            bail!("history line {line_number} has invalid GPU sensor values");
        }
        if !gpu.power_draw_w.is_finite()
            || gpu.power_draw_w < 0.0
            || !gpu.power_limit_w.is_finite()
            || gpu.power_limit_w < 0.0
        {
            bail!("history line {line_number} has invalid GPU power values");
        }
        if gpu.processes.iter().any(|process| process.memory_mib < 0) {
            bail!("history line {line_number} has invalid process memory values");
        }
    }
    Ok(())
}

fn validate_endpoints(report: &Report, line_number: usize) -> Result<()> {
    for endpoint in &report.endpoints {
        validate_endpoint(endpoint, line_number)?;
    }
    Ok(())
}

fn validate_endpoint(endpoint: &crate::domain::Endpoint, line_number: usize) -> Result<()> {
    validate_optional_nonnegative(endpoint.requests_running, line_number, "requests_running")?;
    validate_optional_nonnegative(endpoint.requests_waiting, line_number, "requests_waiting")?;
    if endpoint
        .kv_cache_usage_percent
        .is_some_and(|value| !value.is_finite() || !(0.0..=100.0).contains(&value))
    {
        bail!("history line {line_number} has invalid kv_cache_usage_percent");
    }
    if endpoint.metrics.values().any(|value| !value.is_finite()) {
        bail!("history line {line_number} has a non-finite runtime metric");
    }
    validate_runtime_counters(&endpoint.counters, line_number)?;
    validate_runtime_rates(&endpoint.rates, line_number)
}

fn validate_runtime_counters(
    counters: &crate::domain::RuntimeCounters,
    line_number: usize,
) -> Result<()> {
    for (name, value) in [
        (
            "requests_completed_total",
            counters.requests_completed_total,
        ),
        ("request_errors_total", counters.request_errors_total),
        ("prompt_tokens_total", counters.prompt_tokens_total),
        ("generation_tokens_total", counters.generation_tokens_total),
        ("preemptions_total", counters.preemptions_total),
        (
            "request_latency_seconds_sum",
            counters.request_latency_seconds_sum,
        ),
        (
            "request_latency_seconds_count",
            counters.request_latency_seconds_count,
        ),
        (
            "time_to_first_token_seconds_sum",
            counters.time_to_first_token_seconds_sum,
        ),
        (
            "time_to_first_token_seconds_count",
            counters.time_to_first_token_seconds_count,
        ),
        (
            "time_per_output_token_seconds_sum",
            counters.time_per_output_token_seconds_sum,
        ),
        (
            "time_per_output_token_seconds_count",
            counters.time_per_output_token_seconds_count,
        ),
    ] {
        validate_optional_nonnegative(value, line_number, name)?;
    }
    for (name, histogram) in [
        ("request_latency", &counters.histograms.request_latency),
        (
            "time_to_first_token",
            &counters.histograms.time_to_first_token,
        ),
        (
            "time_per_output_token",
            &counters.histograms.time_per_output_token,
        ),
        ("queue_time", &counters.histograms.queue_time),
    ] {
        validate_histogram(histogram, line_number, name)?;
    }
    Ok(())
}

fn validate_runtime_rates(rates: &crate::domain::RuntimeRates, line_number: usize) -> Result<()> {
    if !rates.interval_seconds.is_finite() || rates.interval_seconds < 0.0 {
        bail!("history line {line_number} has an invalid runtime interval");
    }
    let scalar_rates = [
        ("requests_per_second", rates.requests_per_second),
        ("request_errors_per_second", rates.request_errors_per_second),
        ("prompt_tokens_per_second", rates.prompt_tokens_per_second),
        (
            "generation_tokens_per_second",
            rates.generation_tokens_per_second,
        ),
        ("preemptions_per_second", rates.preemptions_per_second),
        ("mean_request_latency_ms", rates.mean_request_latency_ms),
        (
            "mean_time_to_first_token_ms",
            rates.mean_time_to_first_token_ms,
        ),
        (
            "mean_time_per_output_token_ms",
            rates.mean_time_per_output_token_ms,
        ),
    ];
    for (name, value) in scalar_rates {
        validate_optional_nonnegative(value, line_number, name)?;
    }
    let intervals = [
        ("request_latency", rates.request_latency.as_ref()),
        ("time_to_first_token", rates.time_to_first_token.as_ref()),
        (
            "time_per_output_token",
            rates.time_per_output_token.as_ref(),
        ),
        ("queue_time", rates.queue_time.as_ref()),
    ];
    let has_rates = scalar_rates.iter().any(|(_, value)| value.is_some())
        || intervals.iter().any(|(_, interval)| interval.is_some());
    if has_rates && rates.interval_seconds <= 0.0 {
        bail!("history line {line_number} has runtime rates without a positive interval");
    }
    for (name, interval) in intervals {
        if let Some(interval) = interval {
            let values = [
                interval.samples,
                interval.p50_ms,
                interval.p95_ms,
                interval.p99_ms,
            ];
            if !values.into_iter().all(f64::is_finite)
                || interval.samples <= 0.0
                || interval.p50_ms < 0.0
                || interval.p50_ms > interval.p95_ms
                || interval.p95_ms > interval.p99_ms
            {
                bail!("history line {line_number} has an invalid {name} quantile interval");
            }
        }
    }
    Ok(())
}

fn validate_optional_nonnegative(value: Option<f64>, line_number: usize, name: &str) -> Result<()> {
    if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
        bail!("history line {line_number} has an invalid {name}");
    }
    Ok(())
}

fn validate_histogram(
    histogram: &crate::domain::RuntimeHistogram,
    line_number: usize,
    name: &str,
) -> Result<()> {
    if histogram.buckets.is_empty() {
        // A wholly omitted histogram deserializes through `Default` with zero
        // series; an explicitly empty histogram uses the serde default of one.
        if histogram.series_count > 1 {
            bail!("history line {line_number} has an invalid empty {name} histogram");
        }
        return Ok(());
    }
    if !(1..=MAX_RUNTIME_HISTOGRAM_SERIES).contains(&histogram.series_count)
        || !(2..=MAX_RUNTIME_HISTOGRAM_BUCKETS).contains(&histogram.buckets.len())
        || histogram
            .buckets
            .last()
            .is_none_or(|bucket| bucket.upper_bound_seconds.is_some())
    {
        bail!("history line {line_number} has an invalid {name} histogram shape");
    }
    let mut previous_bound = None;
    let mut previous_count = 0.0;
    for (index, bucket) in histogram.buckets.iter().enumerate() {
        if !bucket.cumulative_count.is_finite()
            || bucket.cumulative_count < 0.0
            || bucket.cumulative_count < previous_count
        {
            bail!("history line {line_number} has invalid {name} histogram counts");
        }
        previous_count = bucket.cumulative_count;
        match bucket.upper_bound_seconds {
            Some(bound)
                if bound.is_finite()
                    && bound >= 0.0
                    && previous_bound.is_none_or(|previous| bound > previous) =>
            {
                previous_bound = Some(bound);
            }
            None if index + 1 == histogram.buckets.len() && previous_bound.is_some() => {}
            _ => bail!("history line {line_number} has invalid {name} histogram bounds"),
        }
    }
    Ok(())
}

fn validate_sources(report: &Report, line_number: usize) -> Result<()> {
    let mut source_names = HashSet::new();
    for source in &report.sources {
        if source.name.trim().is_empty() || !source_names.insert(source.name.as_str()) {
            bail!("history line {line_number} has an empty or duplicate source name");
        }
        match source.state {
            SourceState::Ok if source.error.is_some() => {
                bail!("history line {line_number} has an error on an OK source");
            }
            SourceState::Partial | SourceState::Unavailable
                if source.error.as_deref().is_none_or(str::is_empty) =>
            {
                bail!("history line {line_number} has a failed source without an error");
            }
            SourceState::Skipped if source.records != 0 => {
                bail!("history line {line_number} has records on a skipped source");
            }
            _ => {}
        }
    }
    Ok(())
}

fn read_bounded_line<R: BufRead>(
    reader: &mut R,
    output: &mut Vec<u8>,
    limit: usize,
) -> std::io::Result<bool> {
    output.clear();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(!output.is_empty());
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let consumed = newline.map_or(available.len(), |position| position + 1);
        if output.len().saturating_add(consumed) > limit {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "history record exceeds the 64 MiB limit",
            ));
        }
        output.extend_from_slice(&available[..consumed]);
        reader.consume(consumed);
        if newline.is_some() {
            return Ok(true);
        }
    }
}

fn update_max<T: Copy + PartialOrd>(
    current: &mut Option<T>,
    samples: &mut usize,
    candidate: Option<T>,
) {
    if let Some(candidate) = candidate {
        *samples = samples.saturating_add(1);
        *current =
            Some(current.map_or(
                candidate,
                |value| {
                    if candidate > value { candidate } else { value }
                },
            ));
    }
}

pub fn render_text(summary: &HistorySummary) -> String {
    let range = match (summary.first_sample, summary.last_sample) {
        (Some(first), Some(last)) => format!("{} to {}", first.to_rfc3339(), last.to_rfc3339()),
        _ => "no samples".to_owned(),
    };
    let max_vram =
        format_integer_metric(summary.max_vram_used_mib, " MiB", summary.vram_used_samples);
    let max_gpu_util =
        format_integer_metric(summary.max_gpu_util_percent, "%", summary.gpu_util_samples);
    let max_temperature =
        format_integer_metric(summary.max_temperature_c, "C", summary.temperature_samples);
    let endpoint_availability = format_float_metric(
        summary.endpoint_availability_percent,
        "%",
        summary.endpoint_samples,
        2,
    );
    let max_queued = format_float_metric(
        summary.max_requests_waiting,
        "",
        summary.requests_waiting_samples,
        0,
    );
    let max_kv_cache = format_float_metric(
        summary.max_kv_cache_usage_percent,
        "%",
        summary.kv_cache_usage_samples,
        1,
    );
    let source_availability = format_float_metric(
        summary.source_availability_percent,
        "%",
        summary.source_samples,
        2,
    );
    let max_requests_per_second = format_float_metric(
        summary.max_requests_per_second,
        " req/s",
        summary.requests_per_second_samples,
        2,
    );
    let max_generation_tokens_per_second = format_float_metric(
        summary.max_generation_tokens_per_second,
        " generation tok/s",
        summary.generation_tokens_per_second_samples,
        2,
    );
    let max_request_errors_per_second = format_float_metric(
        summary.max_request_errors_per_second,
        " errors/s",
        summary.request_errors_per_second_samples,
        2,
    );
    let max_time_to_first_token = format_float_metric(
        summary.max_time_to_first_token_ms,
        "ms mean TTFT",
        summary.time_to_first_token_samples,
        2,
    );
    let mut output = format!(
        "GPU Watchman history\n\
         Samples      {} ({range})\n\
         Hosts        {}\n\
         GPU peaks    {max_vram} VRAM | {max_gpu_util} utilization | {max_temperature}\n\
         Inference    {endpoint_availability} endpoint availability | {max_queued} max queued | {max_kv_cache} max KV cache\n\
         Sources      {source_availability} complete | {} non-OK source samples\n\
         Runtime peak {max_requests_per_second} | {max_generation_tokens_per_second} | {max_request_errors_per_second} | {max_time_to_first_token}\n\
         Health       {} critical samples | {} warning samples\n",
        summary.records,
        if summary.hosts.is_empty() {
            "-".to_owned()
        } else {
            summary
                .hosts
                .iter()
                .map(|host| safe_inline(host))
                .collect::<Vec<_>>()
                .join(", ")
        },
        summary
            .source_samples
            .saturating_sub(summary.source_ok_samples),
        summary.critical_samples,
        summary.warning_samples,
    );
    if !summary.finding_counts.is_empty() {
        output.push_str("\nFindings\n");
        let mut findings = summary.finding_counts.iter().collect::<Vec<_>>();
        findings.sort_by(|left, right| right.1.cmp(left.1).then(left.0.cmp(right.0)));
        for (code, count) in findings {
            let _ = writeln!(output, "  {count:>6}  {}", safe_inline(code));
        }
    }
    if !summary.source_non_ok_counts.is_empty() {
        output.push_str("\nNon-OK telemetry sources\n");
        for (source, count) in &summary.source_non_ok_counts {
            let _ = writeln!(output, "  {count:>6}  {}", safe_inline(source));
        }
    }
    output
}

fn format_integer_metric<T: std::fmt::Display>(
    value: Option<T>,
    suffix: &str,
    samples: usize,
) -> String {
    value.map_or_else(
        || format!("N/A (n={samples})"),
        |value| format!("{value}{suffix} (n={samples})"),
    )
}

fn format_float_metric(
    value: Option<f64>,
    suffix: &str,
    samples: usize,
    precision: usize,
) -> String {
    value.map_or_else(
        || format!("N/A (n={samples})"),
        |value| format!("{value:.precision$}{suffix} (n={samples})"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Endpoint, Finding, Gpu, Host, SourceStatus};

    fn write_reports(path: &Path, reports: &[Report]) {
        let mut body = String::new();
        for report in reports {
            writeln!(body, "{}", serde_json::to_string(report).unwrap()).unwrap();
        }
        std::fs::write(path, body).unwrap();
    }

    #[test]
    fn bounded_line_reader_rejects_unterminated_growth() {
        let mut reader = std::io::Cursor::new(b"12345".as_slice());
        let mut output = Vec::new();
        let error = read_bounded_line(&mut reader, &mut output, 4).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert!(output.len() <= 4);
    }

    #[test]
    fn accepts_and_summarizes_valid_ndjson_history() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("history.ndjson");
        let mut report = Report {
            host: Host {
                hostname: "node-1".to_owned(),
                ..Host::default()
            },
            gpus: vec![Gpu {
                memory_used_mib: 42_000,
                temperature_c: 81,
                gpu_util_percent: 99,
                ..Gpu::default()
            }],
            findings: vec![Finding::new(
                Some(0),
                Severity::Warning,
                "temperature-high",
                "hot",
            )],
            sources: vec![
                SourceStatus::ok("nvidia.inventory", 1, 1),
                SourceStatus::failed(
                    "nvidia.processes",
                    SourceState::Partial,
                    1,
                    0,
                    "compute query failed",
                ),
            ],
            ..Report::default()
        };
        crate::analysis::health::finalize(&mut report);
        append(&path, &report).unwrap();
        report.gpus[0].memory_used_mib = 50_000;
        crate::analysis::health::finalize(&mut report);
        append(&path, &report).unwrap();
        let summary = analyze(&path).unwrap();
        assert_eq!(summary.records, 2);
        assert_eq!(summary.max_vram_used_mib, Some(50_000));
        assert_eq!(summary.vram_used_samples, 2);
        assert_eq!(summary.finding_counts["temperature-high"], 2);
        assert_eq!(summary.source_availability_percent, Some(50.0));
        assert_eq!(summary.source_non_ok_counts["nvidia.processes:partial"], 2);
    }

    #[test]
    fn rejects_empty_or_whitespace_only_history() {
        for body in ["", "\n \t\r\n"] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("empty.ndjson");
            std::fs::write(&path, body).unwrap();

            let error = analyze(&path).unwrap_err().to_string();
            assert!(error.contains("contains no reports"), "{error}");
        }
    }

    #[test]
    fn rejects_zero_and_future_schema_versions() {
        for schema_version in [0, SCHEMA_VERSION + 1] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("unsupported.ndjson");
            let report = Report {
                schema_version,
                ..Report::default()
            };
            write_reports(&path, &[report]);

            let error = analyze(&path).unwrap_err().to_string();
            assert!(error.contains("unsupported schema version"), "{error}");
        }
    }

    #[test]
    fn rejects_mixed_supported_schema_versions() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("mixed.ndjson");
        let current = Report::default();
        let previous = Report {
            schema_version: SCHEMA_VERSION - 1,
            ..Report::default()
        };
        write_reports(&path, &[previous, current]);

        let error = analyze(&path).unwrap_err().to_string();
        assert!(error.contains("mixes schema version"), "{error}");
    }

    #[test]
    fn rejects_structurally_inconsistent_reports() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("inconsistent.ndjson");
        let mut report = Report::default();
        report.summary.gpus = 1;
        write_reports(&path, &[report]);

        let error = analyze(&path).unwrap_err().to_string();
        assert!(error.contains("summary that does not match"), "{error}");
    }

    #[test]
    fn unavailable_gpu_and_runtime_metrics_remain_null_with_zero_samples() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("unavailable.ndjson");
        let mut report = Report {
            endpoints: vec![Endpoint {
                url: "http://127.0.0.1:8000".to_owned(),
                reachable: true,
                ..Endpoint::default()
            }],
            ..Report::default()
        };
        crate::analysis::health::finalize(&mut report);
        write_reports(&path, &[report]);

        let summary = analyze(&path).unwrap();
        assert_eq!(summary.max_vram_used_mib, None);
        assert_eq!(summary.vram_used_samples, 0);
        assert_eq!(summary.max_gpu_util_percent, None);
        assert_eq!(summary.gpu_util_samples, 0);
        assert_eq!(summary.max_requests_per_second, None);
        assert_eq!(summary.requests_per_second_samples, 0);
        assert_eq!(summary.endpoint_availability_percent, Some(100.0));
        assert_eq!(summary.endpoint_samples, 1);

        let machine = serde_json::to_value(&summary).unwrap();
        assert!(machine["max_vram_used_mib"].is_null());
        assert_eq!(machine["vram_used_samples"], 0);
        assert!(machine["max_requests_per_second"].is_null());
        assert_eq!(machine["requests_per_second_samples"], 0);

        let text = render_text(&summary);
        assert!(text.contains("GPU peaks    N/A (n=0)"), "{text}");
        assert!(text.contains("Runtime peak N/A (n=0)"), "{text}");
    }

    #[cfg(unix)]
    #[test]
    fn append_rejects_public_or_symbolic_link_targets_before_writing() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let directory = tempfile::tempdir().unwrap();
        let public = directory.path().join("public.ndjson");
        std::fs::write(&public, b"existing\n").unwrap();
        let mut permissions = std::fs::metadata(&public).unwrap().permissions();
        permissions.set_mode(0o644);
        std::fs::set_permissions(&public, permissions).unwrap();
        assert!(append(&public, &Report::default()).is_err());
        assert_eq!(std::fs::read(&public).unwrap(), b"existing\n");

        let target = directory.path().join("target.ndjson");
        std::fs::write(&target, b"target\n").unwrap();
        let mut permissions = std::fs::metadata(&target).unwrap().permissions();
        permissions.set_mode(0o600);
        std::fs::set_permissions(&target, permissions).unwrap();
        let link = directory.path().join("linked.ndjson");
        symlink(&target, &link).unwrap();
        assert!(append(&link, &Report::default()).is_err());
        assert_eq!(std::fs::read(&target).unwrap(), b"target\n");
    }
}
