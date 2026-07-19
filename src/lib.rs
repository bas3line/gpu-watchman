//! GPU Watchman core library.
//!
//! Collection is read-only. The library never changes clocks, power limits,
//! compute modes, MIG configuration, or running workloads.

pub mod analysis;
pub mod application;
pub mod config;
pub mod domain;
pub mod inference;
pub mod operations;
pub mod planning;
pub mod presentation;
pub mod telemetry;

mod security;

use std::time::Instant;

use analysis::{
    AnalyzerConfig, Tracker, analyze_endpoints, analyze_gpus, analyze_sources, finalize,
};
use anyhow::Result;
use domain::{Finding, Report, Severity};
use inference::{ProbeOptions, collect as collect_endpoints};
use telemetry::NvidiaCollector;

/// Options shared by one-shot snapshots and watch/serve collection cycles.
#[derive(Debug, Clone, Default)]
pub struct CycleOptions {
    pub probe_urls: Vec<String>,
    pub probe: ProbeOptions,
    pub analyzer: AnalyzerConfig,
    /// Stable telemetry source names that must be completely available.
    pub required_sources: Vec<String>,
}

/// Collects and finalizes a complete report while updating process trends.
pub fn collect_cycle(
    collector: &NvidiaCollector,
    tracker: &mut Tracker,
    options: &CycleOptions,
) -> Result<Report> {
    let started = Instant::now();
    let (mut report, endpoints) = std::thread::scope(|scope| {
        let probe_task = scope.spawn(|| collect_endpoints(&options.probe_urls, &options.probe));
        let report = collector.collect();
        let endpoints = probe_task.join().unwrap_or_default();
        report.map(|report| (report, endpoints))
    })?;
    report.host = telemetry::local_host();
    report.endpoints = endpoints;
    for source in &mut report.sources {
        source.required = options
            .required_sources
            .iter()
            .any(|required| required.trim() == source.name);
    }
    tracker.observe_endpoints(&mut report.endpoints, report.collected_at);
    report.findings = analyze_gpus(&report.gpus, &options.analyzer);
    report
        .findings
        .extend(analyze_sources(&report.sources, &options.required_sources));
    report
        .findings
        .extend(tracker.observe(&report.gpus, report.collected_at));
    if !report.xid_events.is_empty() {
        report.findings.push(Finding::new(
            None,
            Severity::Critical,
            "xid-events",
            format!(
                "{} NVIDIA Xid driver event(s) found in accessible kernel logs",
                report.xid_events.len()
            ),
        ));
    }
    report
        .findings
        .extend(analyze_endpoints(&report.endpoints, &options.analyzer));
    report.collection_duration_ms =
        u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    finalize(&mut report);
    Ok(report)
}
