//! Snapshot and continuous-monitor execution lifecycle.

use std::io::IsTerminal;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};

use super::{
    CliFormat, EXIT_UNHEALTHY, FailOn, MonitorArgs, MonitorView, resolve_secret, write_stdout,
};
use crate::analysis::{AnalyzerConfig, Tracker, finalize};
use crate::domain::{Report, Severity, SourceState};
use crate::inference::ProbeOptions;
use crate::operations::history;
use crate::presentation::{Exporter, colorize, render, render_process_view};
use crate::security::listen_is_loopback;
use crate::telemetry::NvidiaCollector;
use crate::{CycleOptions, collect_cycle};

pub(super) fn run(args: &MonitorArgs) -> Result<u8> {
    validate_args(args)?;
    let interval = args
        .watch
        .or_else(|| args.listen.as_ref().map(|_| Duration::from_secs(5)));
    let options = cycle_options(args)?;
    let collector = collector(args);
    let mut exporter = start_exporter(args)?;
    let mut tracker = Tracker::new(args.process_growth_mib);
    let running = Arc::new(AtomicBool::new(true));
    let signal_state = Arc::clone(&running);
    ctrlc::set_handler(move || signal_state.store(false, Ordering::SeqCst))
        .context("install signal handler")?;
    let interactive = std::io::stdout().is_terminal();
    let mut first = true;
    let mut failure_streak = 0_u32;
    let mut repeated_error = 0_u64;
    let mut last_error = String::new();
    while running.load(Ordering::SeqCst) {
        match collect_cycle(&collector, &mut tracker, &options) {
            Ok(mut report) => {
                if failure_streak > 0 {
                    eprintln!(
                        "gpu-watchman: collection recovered after {failure_streak} failed attempt(s)"
                    );
                    failure_streak = 0;
                    repeated_error = 0;
                    last_error.clear();
                }
                filter_gpus(&mut report, &args.gpu)?;
                if let Some(exporter) = exporter.as_mut() {
                    exporter.set_report(&report);
                }
                if let Some(path) = args.history.as_deref() {
                    history::append(path, &report)?;
                }
                if !args.quiet {
                    let mut output = String::new();
                    if !first && interactive && !args.no_clear && args.format == CliFormat::Text {
                        output.push_str("\x1b[H\x1b[2J");
                    }
                    let rendered = if args.view == MonitorView::Processes {
                        render_process_view(&report, args.format.into())?
                    } else {
                        render(&report, args.format.into(), args.details, args.all)
                    };
                    if args.format == CliFormat::Text && args.color.enabled(interactive) {
                        output.push_str(&colorize(&rendered));
                    } else {
                        output.push_str(&rendered);
                    }
                    if !write_stdout(&output)? {
                        return Ok(0);
                    }
                }
                if (interval.is_none() && required_source_failed(&report))
                    || failed_threshold(&report, args.fail_on)
                {
                    return Ok(EXIT_UNHEALTHY);
                }
            }
            Err(error) if interval.is_some() => {
                failure_streak = failure_streak.saturating_add(1);
                let error = format!("{error:#}");
                if error == last_error {
                    repeated_error = repeated_error.saturating_add(1);
                    if repeated_error.is_power_of_two() {
                        eprintln!(
                            "gpu-watchman: collection still failing ({repeated_error} repeated): {error}"
                        );
                    }
                } else {
                    eprintln!("gpu-watchman: collection failed: {error}");
                    last_error = error;
                    repeated_error = 0;
                }
            }
            Err(error) => return Err(error),
        }
        first = false;
        let Some(interval) = interval else {
            break;
        };
        interruptible_wait(retry_delay(interval, failure_streak), &running);
    }
    Ok(0)
}

fn cycle_options(args: &MonitorArgs) -> Result<CycleOptions> {
    let analyzer = AnalyzerConfig {
        vram_warning_percent: args.vram_warning,
        vram_critical_percent: args.vram_critical,
        temperature_warning_c: args.temperature_warning,
        temperature_critical_c: args.temperature_critical,
        kv_cache_warning_percent: args.kv_cache_warning,
        kv_cache_critical_percent: args.kv_cache_critical,
        process_growth_warning_mib: args.process_growth_mib,
        ..AnalyzerConfig::default()
    };
    Ok(CycleOptions {
        probe_urls: args.probe.clone(),
        probe: ProbeOptions {
            timeout: args.probe_timeout,
            bearer_token: resolve_secret(
                args.probe_token.clone(),
                args.probe_token_file.as_deref(),
            )?,
            allow_insecure_http: args.probe_allow_insecure_http,
            ..ProbeOptions::default()
        },
        analyzer,
        required_sources: args.require_source.clone(),
    })
}

fn collector(args: &MonitorArgs) -> NvidiaCollector {
    let collector = NvidiaCollector::new(&args.nvidia_smi, args.command_timeout);
    if args.no_xid {
        collector.without_xid()
    } else {
        collector
    }
}

fn start_exporter(args: &MonitorArgs) -> Result<Option<Exporter>> {
    match args.listen.as_deref() {
        Some(address) => {
            let token = resolve_secret(args.api_token.clone(), args.api_token_file.as_deref())?;
            let exporter = if let Some(token) = token {
                Exporter::start(address, args.freshness, Some(token))?
            } else if args.allow_unauthenticated_api {
                Exporter::start_unauthenticated_loopback(address, args.freshness)?
            } else {
                bail!("HTTP API authentication policy was not resolved");
            };
            eprintln!(
                "gpu-watchman: API on http://{} (/livez, /metrics, /healthz, /api/v1/report)",
                exporter.address()
            );
            Ok(Some(exporter))
        }
        None => Ok(None),
    }
}

fn validate_args(args: &MonitorArgs) -> Result<()> {
    if args
        .watch
        .is_some_and(|interval| interval < Duration::from_millis(500))
    {
        bail!("--watch must be at least 500ms to avoid overloading the driver");
    }
    if args.command_timeout.is_zero() || args.probe_timeout.is_zero() {
        bail!("timeouts must be positive");
    }
    if let Some(listen) = args.listen.as_deref() {
        let loopback = listen_is_loopback(listen).map_err(anyhow::Error::msg)?;
        let has_authentication = args.api_token.is_some() || args.api_token_file.is_some();
        if !loopback && !args.allow_remote_listen {
            bail!("non-loopback --listen requires --allow-remote-listen");
        }
        if !loopback && !has_authentication {
            bail!("non-loopback --listen requires --api-token or --api-token-file");
        }
        if loopback && !has_authentication && !args.allow_unauthenticated_api {
            bail!(
                "loopback HTTP API requires --api-token or --api-token-file; use --no-api-auth to explicitly accept local-user access"
            );
        }
    }
    if (args.watch.is_some() || args.listen.is_some())
        && !args.quiet
        && args.format == CliFormat::Json
    {
        bail!("--format json represents one snapshot; use --format ndjson for repeated output");
    }
    if !(0..=100).contains(&args.vram_warning)
        || !(0..=100).contains(&args.vram_critical)
        || args.vram_warning >= args.vram_critical
    {
        bail!("VRAM thresholds must satisfy 0 <= warning < critical <= 100");
    }
    if args.temperature_warning >= args.temperature_critical {
        bail!("temperature warning must be below critical");
    }
    if args.kv_cache_warning >= args.kv_cache_critical {
        bail!("KV-cache warning must be below critical");
    }
    Ok(())
}

fn filter_gpus(report: &mut Report, filters: &[String]) -> Result<()> {
    if filters.is_empty() {
        return Ok(());
    }

    let matches = |filter: &str, gpu: &crate::domain::Gpu| {
        let filter = filter.trim();
        filter == gpu.index.to_string() || filter.eq_ignore_ascii_case(&gpu.uuid)
    };
    let unmatched = filters
        .iter()
        .filter(|filter| !report.gpus.iter().any(|gpu| matches(filter, gpu)))
        .map(|filter| filter.trim())
        .collect::<Vec<_>>();
    if !unmatched.is_empty() {
        let available = report
            .gpus
            .iter()
            .map(|gpu| format!("{} ({})", gpu.index, gpu.uuid))
            .collect::<Vec<_>>();
        if available.is_empty() {
            bail!(
                "--gpu selector(s) matched no GPU: {}; collected inventory contains no GPUs",
                unmatched.join(", ")
            );
        }
        bail!(
            "--gpu selector(s) matched no GPU: {}; available GPUs: {}",
            unmatched.join(", "),
            available.join(", ")
        );
    }

    report
        .gpus
        .retain(|gpu| filters.iter().any(|filter| matches(filter, gpu)));
    let indexes = report
        .gpus
        .iter()
        .map(|gpu| gpu.index)
        .collect::<std::collections::HashSet<_>>();
    report.findings.retain(|finding| {
        finding
            .gpu_index
            .is_none_or(|index| indexes.contains(&index))
    });
    finalize(report);
    Ok(())
}

fn failed_threshold(report: &Report, threshold: FailOn) -> bool {
    report.findings.iter().any(|finding| match threshold {
        FailOn::Never => false,
        FailOn::Warning => matches!(finding.severity, Severity::Warning | Severity::Critical),
        FailOn::Critical => finding.severity == Severity::Critical,
    })
}

fn required_source_failed(report: &Report) -> bool {
    report
        .sources
        .iter()
        .any(|source| source.required && source.state != SourceState::Ok)
        || report.findings.iter().any(|finding| {
            matches!(
                finding.code.as_str(),
                "telemetry-source-required" | "telemetry-source-partial"
            )
        })
}

fn interruptible_wait(duration: Duration, running: &AtomicBool) {
    let mut remaining = duration;
    while !remaining.is_zero() && running.load(Ordering::SeqCst) {
        let slice = remaining.min(Duration::from_millis(100));
        std::thread::sleep(slice);
        remaining = remaining.saturating_sub(slice);
    }
}

fn retry_delay(interval: Duration, failure_streak: u32) -> Duration {
    if failure_streak <= 1 {
        return interval;
    }
    let multiplier = 1_u32 << (failure_streak - 1).min(5);
    interval
        .saturating_mul(multiplier)
        .min(interval.max(Duration::from_secs(30)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Finding, Gpu};

    #[test]
    fn gpu_filter_keeps_host_findings_and_rebuilds_summary() {
        let mut report = Report {
            gpus: vec![
                Gpu {
                    index: 0,
                    ..Gpu::default()
                },
                Gpu {
                    index: 1,
                    ..Gpu::default()
                },
            ],
            findings: vec![
                Finding::new(Some(0), Severity::Warning, "zero", "zero"),
                Finding::new(None, Severity::Warning, "host", "host"),
            ],
            ..Report::default()
        };
        filter_gpus(&mut report, &["1".to_owned()]).unwrap();
        assert_eq!(report.gpus.len(), 1);
        assert_eq!(report.gpus[0].index, 1);
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].code, "host");
    }

    #[test]
    fn gpu_filter_matches_multiple_indexes_and_case_insensitive_uuids() {
        let mut report = Report {
            gpus: vec![
                Gpu {
                    index: 0,
                    uuid: "GPU-AAAA".to_owned(),
                    ..Gpu::default()
                },
                Gpu {
                    index: 1,
                    uuid: "GPU-BBBB".to_owned(),
                    ..Gpu::default()
                },
                Gpu {
                    index: 2,
                    uuid: "GPU-CCCC".to_owned(),
                    ..Gpu::default()
                },
            ],
            ..Report::default()
        };

        filter_gpus(&mut report, &["0".to_owned(), "gpu-bbbb".to_owned()]).unwrap();

        assert_eq!(
            report.gpus.iter().map(|gpu| gpu.index).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn gpu_filter_rejects_any_unmatched_selector_without_mutating_report() {
        let mut report = Report {
            gpus: vec![
                Gpu {
                    index: 0,
                    uuid: "GPU-AAAA".to_owned(),
                    ..Gpu::default()
                },
                Gpu {
                    index: 1,
                    uuid: "GPU-BBBB".to_owned(),
                    ..Gpu::default()
                },
            ],
            findings: vec![Finding::new(Some(0), Severity::Warning, "zero", "zero")],
            ..Report::default()
        };

        let error =
            filter_gpus(&mut report, &["0".to_owned(), "GPU-stale".to_owned()]).unwrap_err();

        assert_eq!(report.gpus.len(), 2);
        assert_eq!(report.findings.len(), 1);
        assert_eq!(
            error.to_string(),
            "--gpu selector(s) matched no GPU: GPU-stale; available GPUs: 0 (GPU-AAAA), 1 (GPU-BBBB)"
        );
    }

    #[test]
    fn repeated_collection_failures_back_off_without_shrinking_long_intervals() {
        assert_eq!(
            retry_delay(Duration::from_secs(1), 1),
            Duration::from_secs(1)
        );
        assert_eq!(
            retry_delay(Duration::from_secs(1), 4),
            Duration::from_secs(8)
        );
        assert_eq!(
            retry_delay(Duration::from_secs(1), 20),
            Duration::from_secs(30)
        );
        assert_eq!(
            retry_delay(Duration::from_secs(60), 4),
            Duration::from_secs(60)
        );
    }

    #[test]
    fn remote_listener_requires_explicit_scope_and_authentication() {
        let mut loopback = MonitorArgs {
            listen: Some("127.0.0.1:9400".to_owned()),
            ..MonitorArgs::default()
        };
        assert!(validate_args(&loopback).is_err());
        loopback.allow_unauthenticated_api = true;
        assert!(validate_args(&loopback).is_ok());
        loopback.allow_unauthenticated_api = false;
        loopback.api_token = Some("private".to_owned());
        assert!(validate_args(&loopback).is_ok());

        let mut remote = MonitorArgs {
            listen: Some("0.0.0.0:9400".to_owned()),
            ..MonitorArgs::default()
        };
        assert!(validate_args(&remote).is_err());
        remote.allow_remote_listen = true;
        assert!(validate_args(&remote).is_err());
        remote.api_token = Some("private".to_owned());
        assert!(validate_args(&remote).is_ok());

        let invalid = MonitorArgs {
            listen: Some("not-an-address".to_owned()),
            ..MonitorArgs::default()
        };
        assert!(validate_args(&invalid).is_err());
    }
}
