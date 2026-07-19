# Report comparison and rollout gates

This command compares hardware/telemetry reports. To compare saved active inference canaries and their latency/throughput distributions, use [**gpu-watchman rollout**](rollout.md).

The **compare** workflow turns two Watchman reports into a deterministic operational delta. It is intended for rollout validation, incident before/after checks, and configuration experiments—not load generation.

## Basic use

~~~sh
gpu-watchman compare known-good.json candidate.json
gpu-watchman compare known-good.json candidate.json --format json
gpu-watchman compare known-good.json candidate.json --fail-on-regression
~~~

Both inputs may be pretty JSON snapshots or NDJSON history files. For NDJSON, the final non-empty record is selected. Inputs larger than 64 MiB and schema versions newer than this binary supports are rejected.

Comparison output currently uses **comparison_version: 2**, which adds telemetry-source deltas to the original GPU, endpoint, pressure, and finding changes.

## Identity and delta rules

- GPUs match by UUID. If UUID is absent, **index:N** is the fallback key.
- Inference endpoints match by their already-redacted report URL.
- Telemetry sources match by stable source name.
- Findings match by GPU target, severity, and stable code. A changed message alone does not create a new incident.
- Removed and added resources remain explicit instead of being folded into zero values.
- GPU deltas cover VRAM, utilization, temperature, power, and process count.
- Endpoint deltas cover reachability, scrape latency, running/waiting requests, KV-cache pressure, request/generation throughput, errors, preemptions, and mean TTFT.
- Source deltas cover state, required-policy membership, duration, and record count. Error text is deliberately excluded from identity and policy matching.

## Regression policy

The comparison sets **regression: true** when any of these occurs:

1. a warning or critical finding is new;
2. a GPU present in the baseline is missing;
3. a previously reachable endpoint is unreachable;
4. a newly configured endpoint is unreachable on its first candidate report.
5. a telemetry source disappears, changes from **ok** to a non-OK state, or is required and non-OK.

Resolved findings, reduced queue depth, lower pressure, and newly available endpoints are shown but do not fail the gate. Raw utilization or temperature growth alone is not considered a regression unless it crosses a health rule and creates a finding; this avoids failing on ordinary traffic differences.

With **--fail-on-regression**, a regression exits 2. Parsing or schema errors exit 1.

## Deployment example

~~~sh
gpu-watchman snapshot --format json \
  --probe http://127.0.0.1:8000 > rollout-before.json

# Deploy, load the model, and allow caches/graphs to warm.

gpu-watchman snapshot --format json \
  --probe http://127.0.0.1:8000 > rollout-after.json

gpu-watchman compare rollout-before.json rollout-after.json \
  --format json --fail-on-regression > rollout-comparison.json
~~~

Always compare equivalent workload phases. Comparing an idle baseline with a fully loaded candidate will correctly report pressure deltas, but those deltas are not a performance conclusion. Pair this health gate with your normal request-level benchmark and SLO tooling.
