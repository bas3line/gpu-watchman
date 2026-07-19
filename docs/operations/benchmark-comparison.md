# Saturation benchmark comparison

Use **benchmark compare** to turn two saved Saturation Benchmark v1 ladders into an offline, privacy-safe regression decision. Run the same explicit workload and ladder before and after a model, engine, kernel, quantization, or serving-configuration change.

~~~sh
gpu-watchman benchmark saturation --model served-model \
  --concurrency-stages 1,2,4,8 --requests-per-worker 20 \
  --format json > baseline.json

# Apply the serving change, then run the identical command into candidate.json.

gpu-watchman benchmark compare baseline.json candidate.json \
  --max-p95-ttft-regression-percent 10 \
  --max-p95-e2e-regression-percent 10 \
  --min-successful-rps-ratio 0.95 \
  --min-completion-token-goodput-ratio 0.95 \
  --max-error-percent-increase 1 \
  --fail-on-regression --format ndjson
~~~

The comparator accepts one JSON object or the final non-empty object in NDJSON. Files are regular-file, UTF-8, size, growth, and stable-read checked. Decode diagnostics contain line and column only; source values, endpoints, paths, attempts, and payloads are never copied into comparison output.

## Admission and formulas

Both reports must use the current schema, be complete and internally reconstructable, have baseline then candidate timestamps, and match on workload ID, model, route, stream mode, full plan/schedule, and source policy. Endpoint origins may differ. Every selected gate is applied independently at every exact concurrency point:

~~~text
latency regression percent = ((candidate / baseline) - 1) * 100
RPS or token-goodput ratio = candidate / baseline
error increase points      = candidate error percent - baseline error percent
~~~

At least 20 relevant samples are required in each report for every quantitative decision. A missing stage, missing authoritative token usage, undersampling, zero ratio baseline, incomplete report, or incompatible identity is **not_evaluable**. It cannot silently pass and is not mislabeled as an observed regression.

## CI policy

Without **--fail-on-regression**, a successfully emitted result exits 0 regardless of its status. This supports exploratory reporting. With enforcement, only **status: pass** exits 0; **regression** and **not_evaluable** exit 2. Input, decoding, version, and usage failures exit 1.

Keep the source reports as private operational artifacts. Internal consistency proves that retained evidence agrees with itself, not that a saved file is authentic or that hidden prompt content matched. The experiment remains closed-loop, single-generator evidence affected by coordinated omission, caching, external traffic, client/network bottlenecks, and endpoint-reported usage. It does not prove causality, production capacity, GPU decode capacity, statistical significance, cost, or an SLA.
