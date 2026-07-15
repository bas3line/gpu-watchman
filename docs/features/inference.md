# Inference Probes And VRAM Growth

## Model Endpoint Probe

`-probe` accepts comma-separated base URLs. For every non-empty URL, Watchman appends `/metrics`, makes a GET request with a three-second timeout, and records the endpoint outcome.

```mermaid
sequenceDiagram
  participant W as Watchman
  participant M as Model server
  W->>M: GET BASE_URL/metrics
  alt 2xx or 3xx response
    M-->>W: Prometheus text
    W->>W: retain vllm/triton/tgi/ollama samples
  else request or read failure
    W->>W: warning: inference-endpoint-down
  end
```

```sh
gpu-watchman -watch 5s \
  -probe http://vllm:8000,http://triton:8002
```

Watchman preserves the metric key, including labels, and only retains keys containing `vllm`, `triton`, `tgi`, or `ollama`. A reachable endpoint without one of those keys is reported as `generic`.

## VRAM Growth Detection

The tracker is in memory and exists only for the current process. It is active when repeated collection occurs with `-watch`.

```mermaid
flowchart LR
  A[Sample GPU UUID + PID + MiB] --> B{Prior sample at least 1 s old?}
  B -->|no| C[Store sample]
  B -->|yes| D{Growth at least 256 MiB?}
  D -->|yes| E[warning: vram-growth]
  D -->|no| C
  E --> C
```

The process key is `<GPU UUID>/<PID>`. If a process disappears from a later sample, its stored baseline is removed. A `vram-growth` finding does not establish the cause of the allocation; it identifies an increase worth investigating.
