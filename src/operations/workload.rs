//! Generic bounded closed-loop workload execution.
//!
//! The executor deliberately owns only scheduling. Callers retain control of
//! request semantics and may use a result type such as `Result<T, E>` when an
//! individual operation can fail.

use std::fmt;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// A completed batch, ordered by the deterministic index passed to the
/// workload closure.
#[derive(Debug)]
pub(crate) struct BatchExecution<T> {
    pub(crate) duration: Duration,
    pub(crate) results: Vec<T>,
}

/// Failures owned by the workload scheduler rather than by an individual
/// workload invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorkloadExecutionError {
    /// A non-empty batch cannot make progress without an active worker.
    ZeroConcurrency,
    /// The operating system refused to create one of the bounded workers.
    ThreadSpawnFailed,
    /// A workload invocation unwound instead of returning a value.
    WorkerPanicked { index: Option<usize> },
    /// The scheduler completed without producing exactly one value per index.
    IncompleteResults { expected: usize, actual: usize },
}

impl fmt::Display for WorkloadExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroConcurrency => {
                formatter.write_str("non-empty workload batch requires concurrency above zero")
            }
            Self::ThreadSpawnFailed => {
                formatter.write_str("failed to create a bounded workload worker")
            }
            Self::WorkerPanicked { index: Some(index) } => {
                write!(formatter, "workload invocation {index} panicked")
            }
            Self::WorkerPanicked { index: None } => {
                formatter.write_str("bounded workload worker panicked")
            }
            Self::IncompleteResults { expected, actual } => write!(
                formatter,
                "bounded workload returned {actual} results; expected {expected}"
            ),
        }
    }
}

impl std::error::Error for WorkloadExecutionError {}

/// Execute exactly `count` indexed operations with at most `concurrency`
/// closed-loop workers.
///
/// Worker threads are created before the measured interval. They wait on a
/// shared start gate, so the interval begins immediately before all workers
/// are released. Deterministic index ranges are split as evenly as possible;
/// when count is an exact multiple of concurrency, every worker executes the
/// same number of operations. Each worker stores results in a preallocated
/// thread-local vector; the vectors are merged and sorted after every worker
/// has finished. Reported duration ends at the last workload invocation and
/// excludes thread joining and result merging.
///
/// An empty batch succeeds without creating workers, including when
/// `concurrency` is zero. A non-empty batch with zero concurrency is rejected.
/// When `count` is below `concurrency`, only `count` worker threads are created.
///
/// Invocation panics are converted to [`WorkloadExecutionError`] when the
/// binary's panic strategy supports unwinding. Errors returned *as values* by
/// the workload closure remain in the ordered results for the caller to
/// interpret.
pub(crate) fn execute_batch<T, F>(
    count: usize,
    concurrency: usize,
    execute: F,
) -> Result<BatchExecution<T>, WorkloadExecutionError>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    if count == 0 {
        return Ok(BatchExecution {
            duration: Duration::ZERO,
            results: Vec::new(),
        });
    }
    if concurrency == 0 {
        return Err(WorkloadExecutionError::ZeroConcurrency);
    }

    let worker_count = count.min(concurrency);
    let start_gate = StartGate::default();

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);

        for slot in 0..worker_count {
            let execute = &execute;
            let start_gate = &start_gate;
            let (start_index, end_index) = worker_range(count, worker_count, slot);
            let builder = std::thread::Builder::new().name(format!("gpu-watchman-workload-{slot}"));
            let handle = builder.spawn_scoped(scope, move || {
                let mut results = Vec::with_capacity(end_index.saturating_sub(start_index));
                let Some(started) = start_gate.worker_ready_and_wait() else {
                    return WorkerResults::default();
                };

                let mut first_panic = None;
                for index in start_index..end_index {
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute(index)))
                    {
                        Ok(result) => results.push((index, result)),
                        Err(_) => {
                            first_panic =
                                Some(first_panic.map_or(index, |prior: usize| prior.min(index)));
                        }
                    }
                }
                let duration = started.elapsed();

                WorkerResults {
                    results,
                    first_panic,
                    duration,
                }
            });

            if let Ok(handle) = handle {
                handles.push(handle);
            } else {
                start_gate.cancel();
                for handle in handles {
                    let _ = handle.join();
                }
                return Err(WorkloadExecutionError::ThreadSpawnFailed);
            }
        }

        let mut indexed_results = Vec::with_capacity(count);
        start_gate.wait_until_ready(worker_count);
        start_gate.release();

        let mut first_panic = None;
        let mut worker_panicked = false;
        let mut duration = Duration::ZERO;
        for handle in handles {
            match handle.join() {
                Ok(worker) => {
                    duration = duration.max(worker.duration);
                    indexed_results.extend(worker.results);
                    if let Some(index) = worker.first_panic {
                        first_panic =
                            Some(first_panic.map_or(index, |prior: usize| prior.min(index)));
                    }
                }
                Err(_) => worker_panicked = true,
            }
        }

        if worker_panicked {
            return Err(WorkloadExecutionError::WorkerPanicked { index: None });
        }
        if let Some(index) = first_panic {
            return Err(WorkloadExecutionError::WorkerPanicked { index: Some(index) });
        }

        indexed_results.sort_unstable_by_key(|(index, _)| *index);
        if indexed_results.len() != count {
            return Err(WorkloadExecutionError::IncompleteResults {
                expected: count,
                actual: indexed_results.len(),
            });
        }
        let results = indexed_results
            .into_iter()
            .map(|(_, result)| result)
            .collect();

        Ok(BatchExecution { duration, results })
    })
}

fn worker_range(count: usize, worker_count: usize, slot: usize) -> (usize, usize) {
    let base = count / worker_count;
    let remainder = count % worker_count;
    let extra_before = slot.min(remainder);
    let start = slot.saturating_mul(base).saturating_add(extra_before);
    let length = base + usize::from(slot < remainder);
    (start, start.saturating_add(length))
}

#[derive(Debug)]
struct WorkerResults<T> {
    results: Vec<(usize, T)>,
    first_panic: Option<usize>,
    duration: Duration,
}

impl<T> Default for WorkerResults<T> {
    fn default() -> Self {
        Self {
            results: Vec::new(),
            first_panic: None,
            duration: Duration::ZERO,
        }
    }
}

#[derive(Debug, Default)]
struct StartGate {
    state: Mutex<StartState>,
    changed: Condvar,
}

#[derive(Debug, Default)]
struct StartState {
    ready: usize,
    started_at: Option<Instant>,
    cancelled: bool,
}

impl StartGate {
    fn worker_ready_and_wait(&self) -> Option<Instant> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.ready = state.ready.saturating_add(1);
        self.changed.notify_all();
        while state.started_at.is_none() && !state.cancelled {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state.started_at
    }

    fn wait_until_ready(&self, worker_count: usize) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.ready < worker_count {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    fn release(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.started_at = Some(Instant::now());
        self.changed.notify_all();
    }

    fn cancel(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.cancelled = true;
        self.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier, Mutex};
    use std::time::{Duration, Instant};

    use super::{WorkloadExecutionError, execute_batch};

    #[test]
    fn empty_batch_is_an_explicit_success() {
        let execution = execute_batch::<usize, _>(0, 0, |_| unreachable!("no work exists"))
            .expect("empty batch should succeed");

        assert_eq!(execution.duration, Duration::ZERO);
        assert!(execution.results.is_empty());
    }

    #[test]
    fn non_empty_batch_rejects_zero_concurrency() {
        let error = execute_batch(1, 0, |index| index).expect_err("zero workers cannot progress");

        assert_eq!(error, WorkloadExecutionError::ZeroConcurrency);
    }

    #[test]
    fn returns_every_index_once_in_index_order() {
        let seen = (0..257).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
        let execution = execute_batch(257, 8, |index| {
            seen[index].fetch_add(1, Ordering::Relaxed);
            if index % 7 == 0 {
                std::thread::yield_now();
            }
            index * 3
        })
        .expect("batch should complete");

        assert_eq!(
            execution.results,
            (0..257).map(|index| index * 3).collect::<Vec<_>>()
        );
        assert!(seen.iter().all(|count| count.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn exact_multiple_assigns_the_same_request_count_to_every_worker() {
        let counts = Mutex::new(BTreeMap::<String, usize>::new());
        let execution = execute_batch(12, 4, |index| {
            let name = std::thread::current()
                .name()
                .unwrap_or("unnamed")
                .to_owned();
            let mut counts = counts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *counts.entry(name).or_default() += 1;
            index
        })
        .expect("batch should complete");

        assert_eq!(execution.results, (0..12).collect::<Vec<_>>());
        let counts = counts
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(counts.len(), 4);
        assert!(counts.values().all(|count| *count == 3));
    }

    #[test]
    fn concurrency_is_capped_and_count_below_limit_is_supported() {
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let execution = execute_batch(3, 64, |index| {
            let current = active.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(current, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(10));
            active.fetch_sub(1, Ordering::SeqCst);
            index
        })
        .expect("small batch should complete");

        assert_eq!(execution.results, vec![0, 1, 2]);
        assert_eq!(peak.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn peak_concurrency_never_exceeds_the_requested_ceiling() {
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let execution = execute_batch(32, 4, |index| {
            let current = active.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(current, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(2));
            active.fetch_sub(1, Ordering::SeqCst);
            index
        })
        .expect("bounded batch should complete");

        assert_eq!(execution.results.len(), 32);
        assert_eq!(peak.load(Ordering::SeqCst), 4);
        assert_eq!(active.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn simultaneous_gate_releases_all_active_workers() {
        let worker_count = 4;
        let callback_gate = Arc::new(Barrier::new(worker_count));
        let arrivals = Arc::new(Mutex::new(Vec::with_capacity(worker_count)));
        let outer_started = Instant::now();
        let execution = execute_batch(worker_count, worker_count, {
            let callback_gate = Arc::clone(&callback_gate);
            let arrivals = Arc::clone(&arrivals);
            move |index| {
                arrivals
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(Instant::now());
                callback_gate.wait();
                index
            }
        })
        .expect("barrier batch should complete");
        let wall_duration = outer_started.elapsed();

        let arrivals = arrivals
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let first = arrivals.iter().min().expect("first arrival");
        let last = arrivals.iter().max().expect("last arrival");
        assert!(last.duration_since(*first) < Duration::from_secs(1));
        assert!(execution.duration <= wall_duration);
        assert_eq!(execution.results, vec![0, 1, 2, 3]);
    }

    #[test]
    fn closure_results_remain_caller_owned_values() {
        let execution = execute_batch(5, 2, |index| {
            if index == 2 {
                Err("request failed")
            } else {
                Ok(index)
            }
        })
        .expect("a returned error is still a workload result");

        assert_eq!(
            execution.results,
            vec![Ok(0), Ok(1), Err("request failed"), Ok(3), Ok(4)]
        );
    }

    #[test]
    fn invocation_panics_become_scheduler_errors() {
        let error = execute_batch(12, 4, |index| {
            assert_ne!(index, 7, "synthetic panic");
            index
        })
        .expect_err("panic should be converted");

        assert_eq!(
            error,
            WorkloadExecutionError::WorkerPanicked { index: Some(7) }
        );
    }
}
