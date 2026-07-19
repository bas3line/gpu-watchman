# Canary rollout comparison

**gpu-watchman rollout** performs an offline, like-for-like comparison of two saved
**CanaryReport** JSON values. It never contacts either inference endpoint.

~~~sh
gpu-watchman rollout baseline-canary.json candidate-canary.json \
  --max-p95-ttft-regression-percent 10 \
  --max-p95-e2e-regression-percent 10 \
  --min-output-tps-ratio 0.90 \
  --max-success-drop-percent 1 \
  --fail-on-regression --format json
~~~

Inputs may be complete JSON documents or NDJSON files; for NDJSON, the final
non-empty record is selected. Each input must be a regular UTF-8 file and is capped
at 64 MiB. Decoding also caps attempts at 10,000, gates at five, and retained
identity/diagnostic strings at their canary-contract limits. Canary versions newer
than this build are rejected directly rather than converted into a comparison.

## Workload identity

Canary version 2 adds **workload_id**, a non-secret operator identity for the exact
synthetic request. The built-in prompt always uses **builtin-v1**. A custom
**--prompt** or **--prompt-file** requires an explicit **--workload-id** composed of
at most 128 ASCII letters, digits, dots, underscores, colons, slashes, or hyphens.
GPU Watchman never hashes, serializes, or otherwise derives identity from prompt
content.

Version-1 reports remain readable, but their missing workload and policy evidence
makes them incompatible with rollout comparison. Re-run both canaries with the same
explicit workload ID and policy instead of guessing what an older run contained.

Canary version 2 also records the privacy-safe policy that produced the status:
minimum success percentage, optional maximum TTFT and end-to-end milliseconds,
optional minimum output-token rate, and whether content expectation checking was
configured. Expectation text itself is never retained.

Before evaluating quantitative gates, rollout validates each report from its raw
attempts. Attempt count and ordered indexes must exactly match the plan; individual
success/failure, timing, token, stream, expectation, and failure-stage evidence must
be coherent. Summary counts, token totals, and distributions are recomputed with the
production nearest-rank percentile implementation. The canonical canary gates and
top-level status are then rebuilt from that summary and the recorded policy. Forged
or stale summaries, distributions, gates, and statuses therefore fail compatibility.

Rollout then requires:

- the same current canary contract version on both sides;
- passing baseline and candidate canary status;
- a candidate timestamp that is not before the baseline timestamp;
- identical, non-empty workload ID, model, and chat-completions route;
- identical stream mode, request count, concurrency, maximum completion tokens,
  timeout, and response-size limit;
- exactly identical recorded canary policy.

Any failed compatibility check sets **compatible: false** and **regression: true**.
The JSON evidence includes both privacy-safe identities and policy fields. Endpoint
URLs may differ between baseline and candidate and are deliberately excluded, as
are prompts, generated text, expectation text, and credentials.

## Quantitative gates

All four quantitative gates are opt-in:

| Option | Calculation | Pass condition |
| --- | --- | --- |
| **--max-p95-ttft-regression-percent P** | **(candidate p95 - baseline p95) / baseline p95 × 100** | value is at most **P** |
| **--max-p95-e2e-regression-percent P** | **(candidate p95 - baseline p95) / baseline p95 × 100** | value is at most **P** |
| **--min-output-tps-ratio R** | **candidate p50 / baseline p50** | ratio is at least **R** |
| **--max-success-drop-percent P** | **baseline success percent - candidate success percent** | drop is at most **P** percentage points |

A selected latency or output-rate gate requires at least **20 complete successful
samples in each report**, and the distribution sample count must equal that report's
successful-request count. Every selected distribution value must be finite,
non-negative, and consistently ordered. A selected baseline metric must be greater
than zero. Missing, partial, too-small, non-finite, negative, inconsistent, or
zero-baseline evidence fails closed and is emitted as an unavailable gate with a
specific reason.

When no quantitative option is selected, rollout still enforces every compatibility,
status, timestamp, and summary-integrity check. This supports an identity/status-only
admission gate without silently accepting a failed candidate.

## Output and exit codes

Text, pretty JSON, and compact one-record NDJSON are supported. Machine output uses
the stable **rollout_version: 1** contract and contains baseline/candidate identities,
compatibility checks, selected gates, and the final regression boolean. Each identity
records canary version, workload ID, model, route, stream mode, count, concurrency,
maximum tokens, timeout, response-size limit, and the canary policy.

| Code | Meaning |
| --- | --- |
| 0 | Comparison completed; or a regression was reported without **--fail-on-regression** |
| 1 | Usage, input, decoding, unsupported-version, or threshold validation failed |
| 2 | **--fail-on-regression** was selected and identity/status compatibility or a selected gate failed |
