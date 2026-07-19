//! Stateful trend detection across collection cycles.

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};

use crate::domain::{Endpoint, Finding, Gpu, RuntimeCounters, RuntimeRates, Severity};

#[derive(Debug, Clone)]
struct Sample {
    memory_mib: i64,
    at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct EndpointSample {
    counters: RuntimeCounters,
    at: DateTime<Utc>,
}

#[derive(Debug, Default)]
pub struct Tracker {
    processes: HashMap<String, Sample>,
    endpoints: HashMap<String, EndpointSample>,
    growth_warning_mib: i64,
}

impl Tracker {
    pub fn new(growth_warning_mib: i64) -> Self {
        Self {
            processes: HashMap::new(),
            endpoints: HashMap::new(),
            growth_warning_mib,
        }
    }

    pub fn observe(&mut self, gpus: &[Gpu], now: DateTime<Utc>) -> Vec<Finding> {
        let mut findings = Vec::new();
        let mut seen = HashSet::new();
        for gpu in gpus {
            for process in &gpu.processes {
                let key = format!("{}/{}", gpu.uuid, process.pid);
                seen.insert(key.clone());
                if let Some(previous) = self.processes.get(&key) {
                    let elapsed_ms = now.signed_duration_since(previous.at).num_milliseconds();
                    let growth = process.memory_mib - previous.memory_mib;
                    if elapsed_ms >= 1_000 && growth >= self.growth_threshold() {
                        findings.push(Finding::new(
                            Some(gpu.index),
                            Severity::Warning,
                            "vram-growth",
                            format!(
                                "process {} ({}) grew by {growth} MiB since the prior sample",
                                process.pid, process.name
                            ),
                        ));
                    }
                }
                self.processes.insert(
                    key,
                    Sample {
                        memory_mib: process.memory_mib,
                        at: now,
                    },
                );
            }
        }
        self.processes.retain(|key, _| seen.contains(key));
        findings
    }

    /// Convert monotonic runtime counters into interval rates and latency means.
    pub fn observe_endpoints(&mut self, endpoints: &mut [Endpoint], now: DateTime<Utc>) {
        let mut seen = HashSet::new();
        for endpoint in endpoints {
            let key = if endpoint.metrics_url.is_empty() {
                endpoint.url.clone()
            } else {
                endpoint.metrics_url.clone()
            };
            seen.insert(key.clone());
            if !endpoint.reachable || endpoint.counters.is_empty() {
                continue;
            }
            if let Some(previous) = self.endpoints.get(&key) {
                let elapsed = now
                    .signed_duration_since(previous.at)
                    .to_std()
                    .map_or(0.0, |duration| duration.as_secs_f64());
                if elapsed > 0.0 {
                    endpoint.rates = derive_rates(&previous.counters, &endpoint.counters, elapsed);
                }
            }
            self.endpoints.insert(
                key,
                EndpointSample {
                    counters: endpoint.counters.clone(),
                    at: now,
                },
            );
        }
        self.endpoints.retain(|key, _| seen.contains(key));
    }

    fn growth_threshold(&self) -> i64 {
        if self.growth_warning_mib > 0 {
            self.growth_warning_mib
        } else {
            256
        }
    }
}

fn derive_rates(
    before: &RuntimeCounters,
    after: &RuntimeCounters,
    elapsed_seconds: f64,
) -> RuntimeRates {
    RuntimeRates {
        interval_seconds: elapsed_seconds,
        requests_per_second: rate(
            before.requests_completed_total,
            after.requests_completed_total,
            elapsed_seconds,
        ),
        request_errors_per_second: rate(
            before.request_errors_total,
            after.request_errors_total,
            elapsed_seconds,
        ),
        prompt_tokens_per_second: rate(
            before.prompt_tokens_total,
            after.prompt_tokens_total,
            elapsed_seconds,
        ),
        generation_tokens_per_second: rate(
            before.generation_tokens_total,
            after.generation_tokens_total,
            elapsed_seconds,
        ),
        preemptions_per_second: rate(
            before.preemptions_total,
            after.preemptions_total,
            elapsed_seconds,
        ),
        mean_request_latency_ms: histogram_mean_ms(
            before.request_latency_seconds_sum,
            before.request_latency_seconds_count,
            after.request_latency_seconds_sum,
            after.request_latency_seconds_count,
        ),
        mean_time_to_first_token_ms: histogram_mean_ms(
            before.time_to_first_token_seconds_sum,
            before.time_to_first_token_seconds_count,
            after.time_to_first_token_seconds_sum,
            after.time_to_first_token_seconds_count,
        ),
        mean_time_per_output_token_ms: histogram_mean_ms(
            before.time_per_output_token_seconds_sum,
            before.time_per_output_token_seconds_count,
            after.time_per_output_token_seconds_sum,
            after.time_per_output_token_seconds_count,
        ),
        request_latency: before
            .histograms
            .request_latency
            .interval(&after.histograms.request_latency),
        time_to_first_token: before
            .histograms
            .time_to_first_token
            .interval(&after.histograms.time_to_first_token),
        time_per_output_token: before
            .histograms
            .time_per_output_token
            .interval(&after.histograms.time_per_output_token),
        queue_time: before
            .histograms
            .queue_time
            .interval(&after.histograms.queue_time),
    }
}

fn rate(before: Option<f64>, after: Option<f64>, elapsed_seconds: f64) -> Option<f64> {
    monotonic_delta(before, after).map(|delta| delta / elapsed_seconds)
}

fn histogram_mean_ms(
    before_sum: Option<f64>,
    before_count: Option<f64>,
    after_sum: Option<f64>,
    after_count: Option<f64>,
) -> Option<f64> {
    let sum = monotonic_delta(before_sum, after_sum)?;
    let count = monotonic_delta(before_count, after_count)?;
    (count > 0.0).then_some(sum / count * 1_000.0)
}

fn monotonic_delta(before: Option<f64>, after: Option<f64>) -> Option<f64> {
    let (Some(before), Some(after)) = (before, after) else {
        return None;
    };
    (before.is_finite() && after.is_finite() && after >= before).then_some(after - before)
}

#[cfg(test)]
mod tests {
    use chrono::Duration;

    use super::*;
    use crate::domain::{GpuProcess, RuntimeHistogram, RuntimeHistogramBucket};

    fn histogram(counts: [f64; 3]) -> RuntimeHistogram {
        RuntimeHistogram::from_cumulative_buckets(vec![
            RuntimeHistogramBucket {
                upper_bound_seconds: Some(0.1),
                cumulative_count: counts[0],
            },
            RuntimeHistogramBucket {
                upper_bound_seconds: Some(1.0),
                cumulative_count: counts[1],
            },
            RuntimeHistogramBucket {
                upper_bound_seconds: None,
                cumulative_count: counts[2],
            },
        ])
        .unwrap()
    }

    #[test]
    fn detects_process_growth_and_forgets_departed_processes() {
        let now = Utc::now();
        let mut tracker = Tracker::new(256);
        let mut gpu = Gpu {
            uuid: "GPU-1".to_owned(),
            processes: vec![GpuProcess {
                pid: 10,
                name: "server".to_owned(),
                memory_mib: 1_000,
                ..GpuProcess::default()
            }],
            ..Gpu::default()
        };
        assert!(tracker.observe(&[gpu.clone()], now).is_empty());
        gpu.processes[0].memory_mib = 1_300;
        assert_eq!(
            tracker.observe(&[gpu], now + Duration::seconds(2))[0].code,
            "vram-growth"
        );
        tracker.observe(&[], now + Duration::seconds(3));
        assert!(tracker.processes.is_empty());
    }

    #[test]
    fn derives_runtime_rates_and_ignores_counter_resets() {
        let now = Utc::now();
        let mut tracker = Tracker::new(256);
        let mut endpoint = Endpoint {
            url: "http://runtime/metrics".to_owned(),
            reachable: true,
            counters: RuntimeCounters {
                requests_completed_total: Some(100.0),
                generation_tokens_total: Some(1_000.0),
                time_to_first_token_seconds_sum: Some(20.0),
                time_to_first_token_seconds_count: Some(100.0),
                histograms: crate::domain::RuntimeHistograms {
                    request_latency: histogram([10.0, 80.0, 100.0]),
                    ..crate::domain::RuntimeHistograms::default()
                },
                ..RuntimeCounters::default()
            },
            ..Endpoint::default()
        };
        tracker.observe_endpoints(std::slice::from_mut(&mut endpoint), now);
        endpoint.counters.requests_completed_total = Some(120.0);
        endpoint.counters.generation_tokens_total = Some(1_400.0);
        endpoint.counters.time_to_first_token_seconds_sum = Some(24.0);
        endpoint.counters.time_to_first_token_seconds_count = Some(120.0);
        endpoint.counters.histograms.request_latency = histogram([15.0, 95.0, 120.0]);
        tracker.observe_endpoints(
            std::slice::from_mut(&mut endpoint),
            now + Duration::seconds(2),
        );

        assert_eq!(endpoint.rates.requests_per_second, Some(10.0));
        assert_eq!(endpoint.rates.generation_tokens_per_second, Some(200.0));
        assert_eq!(endpoint.rates.mean_time_to_first_token_ms, Some(200.0));
        let request_latency = endpoint.rates.request_latency.as_ref().unwrap();
        assert!((request_latency.samples - 20.0).abs() < f64::EPSILON);
        assert!((request_latency.p95_ms - 1_000.0).abs() < f64::EPSILON);

        endpoint.counters.requests_completed_total = Some(1.0);
        endpoint.counters.histograms.request_latency = histogram([1.0, 2.0, 3.0]);
        tracker.observe_endpoints(
            std::slice::from_mut(&mut endpoint),
            now + Duration::seconds(4),
        );
        assert_eq!(endpoint.rates.requests_per_second, None);
        assert!(endpoint.rates.request_latency.is_none());
    }
}
