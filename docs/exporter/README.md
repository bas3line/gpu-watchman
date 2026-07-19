# Prometheus and HTTP API

## Start

~~~sh
gpu-watchman serve --interval 5s \
  --listen 127.0.0.1:9400 \
  --api-token-file /run/secrets/watchman-api-token \
  --probe-token-file /run/secrets/runtime-metrics-token \
  --probe http://localhost:8000
~~~

The service is quiet by default and collects every five seconds when **--interval** is omitted. Add **--emit text** or **--emit ndjson** only when the report stream should also be written to stdout.

## Routes

| Route | Response |
| --- | --- |
| **/** | Service name, version, and route discovery |
| **/livez** | Process/server liveness; deliberately unauthenticated and independent of collection |
| **/metrics** | Prometheus text exposition |
| **/healthz** | Freshness and last report status |
| **/api/v1/report** | Immutable latest schema-versioned JSON report |

Only GET is allowed. Unknown paths return 404. Before the first successful collection, metrics advertises report availability zero while health and report return 503. Health also returns 503 when the last success is older than **--freshness**. Liveness returns 200 whenever the embedded server can answer and is suitable for a supervisor; it does not claim GPU collection is healthy.

Metrics, report, and discovery require a token from **--api-token**, **--api-token-file**, **GPU_WATCHMAN_API_TOKEN**, or **GPU_WATCHMAN_API_TOKEN_FILE**:

~~~text
Authorization: Bearer TOKEN
~~~

**/livez** and **/healthz** remain unauthenticated so Kubernetes and process supervisors can distinguish a dead server from an unavailable/stale collector without embedding API credentials in probe definitions. Neither response contains GPU, process, endpoint, or host identity.

For deliberate single-user debugging only, **--no-api-auth** clears inherited credentials and explicitly permits an unauthenticated loopback listener. Without either a resolved token or that flag, even loopback startup fails closed; this prevents a privileged observer from becoming a process-data oracle for other local users. Unauthenticated non-loopback listeners are never allowed.

GPU Watchman does not terminate TLS. Bind to loopback by default. A non-loopback address requires both **--allow-remote-listen** (or the profile equivalent) and an API token; the process refuses to bind if either half is missing. Use an authenticated TLS reverse proxy or service mesh when traffic crosses a trust boundary.

The packaged Kubernetes file is an isolated base: Watchman binds pod loopback and no Service or NetworkPolicy is created. A separate production overlay must inject an authenticated mTLS proxy/mesh, opt Watchman into the non-loopback listener, expose only the proxy, add proxy-owned health probes/rate limits, and restrict ingress to the authenticated scraper. NetworkPolicy narrows reachability; it does not encrypt the bearer token.

The embedded HTTP/1.1 server is deliberately small and bounded: eight workers, an eight-connection queue, no more than two active-or-queued connections per peer IP, a 750 ms absolute request-header deadline, a two-second response deadline, an 8 KiB request-line limit, 72 KiB/64-header limits, and a 2 MiB response-body ceiling. It accepts one GET per connection and always closes the connection. Put any remote listener behind a rate-limiting proxy; these limits protect the observer but are not a replacement for edge admission control. An explicitly unauthenticated loopback listener additionally requires an exact Host authority matching the bound loopback address and port, preventing DNS-rebinding and HTTP/1.0 Host-bypass access.

Prefer token-file options for mounted Kubernetes/systemd secrets because the token does not appear in the command line. **--probe-token-file** and **GPU_WATCHMAN_PROBE_TOKEN_FILE** similarly authenticate runtime metrics probes. Token files are trimmed when read, conflict with their inline-token counterpart, and their contents are never serialized.

Passive runtime probes require HTTPS for non-loopback targets unless **--allow-insecure-http** explicitly accepts cleartext transport. They reject URL user information, refuse redirects, and bypass ambient proxy variables for every target, so a probe bearer token is sent only to the configured origin.

## Metric families

Every family includes HELP and TYPE metadata.

### Observer

- **gpu_watchman_report_available**
- **gpu_watchman_report_status{status}**
- **gpu_watchman_findings{severity}**
- **gpu_watchman_collection_duration_seconds**
- **gpu_watchman_last_success_timestamp_seconds**
- **gpu_watchman_last_success_age_seconds**

### Collection sources

Source metrics use the stable **source** label. The status family additionally carries the current **status** value: **ok**, **partial**, **unavailable**, or **skipped**.

- **gpu_watchman_source_status{source,status}** — 1 for the state observed in the latest report
- **gpu_watchman_source_required{source}** — 1 when **--require-source** made the source fail-closed
- **gpu_watchman_source_duration_seconds{source}** — time spent collecting that source
- **gpu_watchman_source_records{source}** — usable records retained from the source

The stable source values are **nvidia.inventory**, **nvidia.processes**, **nvidia.optional**, **nvidia.topology**, and **kernel.xid**. A legitimate zero-record result, such as no active GPU processes, remains **ok**. Sources that expect per-GPU rows or a topology matrix become **partial** when those records are missing. Bounded diagnostic classifications are available in the JSON report and terminal, but are deliberately excluded from Prometheus to avoid secrets, unbounded labels, and cardinality churn. Raw driver-wrapper stderr is never serialized.

In **snapshot** and **top** terminal output, the **SOURCES** section shows non-OK sources by default. Add their **--details** option to include healthy sources as well.

### GPU

GPU labels are **gpu**, **uuid**, and **name**.

- **gpu_watchman_vram_used_mib**
- **gpu_watchman_vram_total_mib**
- **gpu_watchman_vram_free_mib**
- **gpu_watchman_utilization_percent**
- **gpu_watchman_memory_utilization_percent**
- **gpu_watchman_temperature_celsius**
- **gpu_watchman_power_watts**
- **gpu_watchman_power_limit_watts**
- **gpu_watchman_processes**
- **gpu_watchman_ecc_uncorrected**
- **gpu_watchman_retired_pages**
- **gpu_watchman_process_vram_mib{pid,process,owner}**

Per-process metrics are operationally valuable but increase cardinality as PIDs churn. Drop the family at ingestion if your metrics budget cannot accommodate it; full identity remains in the report API and history.

### Inference

Endpoint metrics use **url** and normalized **runtime** labels.

- **gpu_watchman_inference_endpoint_up**
- **gpu_watchman_inference_endpoint_latency_seconds**
- **gpu_watchman_inference_requests_running**
- **gpu_watchman_inference_requests_waiting**
- **gpu_watchman_inference_kv_cache_usage_percent**
- **gpu_watchman_inference_requests_per_second**
- **gpu_watchman_inference_request_errors_per_second**
- **gpu_watchman_inference_prompt_tokens_per_second**
- **gpu_watchman_inference_generation_tokens_per_second**
- **gpu_watchman_inference_preemptions_per_second**
- **gpu_watchman_inference_mean_request_latency_seconds**
- **gpu_watchman_inference_mean_time_to_first_token_seconds**
- **gpu_watchman_inference_mean_time_per_output_token_seconds**
- **gpu_watchman_inference_latency_interval_samples{kind}**
- **gpu_watchman_inference_latency_quantile_seconds{kind,quantile}**

The fixed **kind** values are **request_latency**, **time_to_first_token**, **time_per_output_token**, and **queue_time**; fixed quantiles are **0.5**, **0.95**, and **0.99**. Rate and interval-latency families appear after two compatible samples. Counter or histogram resets, changed bucket boundaries, incomplete histograms, missing runtime families, non-finite values, and unavailable normalized values are omitted, never exported as invalid or negative Prometheus samples. These interval estimates are gauges, not cumulative native histograms.

## Prometheus

~~~yaml
scrape_configs:
  - job_name: gpu-watchman
    scrape_interval: 15s
    authorization:
      credentials_file: /run/secrets/watchman-api-token
    static_configs:
      - targets: ["node-07:9400"]
~~~

Import [the example alert rules](../../packaging/prometheus/alerts.yaml) and tune thresholds/durations to your scheduler and workload.

The packaged Kubernetes manifest intentionally has no annotation-based scrape discovery because annotations cannot carry the bearer token. Configure Prometheus discovery, a PodMonitor, or a ServiceMonitor to read the mounted Secret and emit the authorization header; otherwise those scrapes correctly receive HTTP 401.
