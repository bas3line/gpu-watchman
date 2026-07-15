# Prometheus Exporter

Start the exporter with `-listen`. If no `-watch` interval is supplied, Watchman uses five seconds so the latest report stays fresh.

```sh
gpu-watchman -all -watch 5s -listen :9400
curl http://localhost:9400/metrics
```

```mermaid
flowchart LR
  Collect[Collection cycle] --> Report[Latest complete report]
  Report --> Server[In-process HTTP server]
  Prometheus -->|GET /metrics| Server
  Server -->|text/plain 0.0.4| Prometheus
```

## Metrics

All GPU metrics use `gpu`, `uuid`, and `name` labels. Per-process VRAM adds `pid` and `process`. Endpoint availability uses a `url` label.

| Metric | Unit | Meaning |
| --- | --- | --- |
| `gpu_watchman_vram_used_mib` | MiB | Allocated GPU memory |
| `gpu_watchman_vram_total_mib` | MiB | Total GPU memory |
| `gpu_watchman_utilization_percent` | percent | GPU utilization from driver |
| `gpu_watchman_temperature_celsius` | Celsius | GPU temperature |
| `gpu_watchman_power_watts` | watts | Current power draw |
| `gpu_watchman_processes` | count | Reported compute/graphics process count |
| `gpu_watchman_ecc_uncorrected` | count | Volatile uncorrected ECC error count |
| `gpu_watchman_retired_pages` | count | Retired memory page count |
| `gpu_watchman_process_vram_mib` | MiB | Per-process GPU allocation |
| `gpu_watchman_inference_endpoint_up` | 0 or 1 | Model endpoint reachability |

## Scrape Configuration

```yaml
scrape_configs:
  - job_name: gpu-watchman
    static_configs:
      - targets: ["node-07:9400"]
```

## Limits

The exporter only serves the latest completed report. It does not retain a time series itself; Prometheus should scrape it. It does not expose every raw driver value, only the stable operational metrics listed above.
