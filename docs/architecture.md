# Architecture

GPU Watchman is organized into three runtime layers: collection, analysis, and presentation. The CLI owns the optional endpoint probe, Prometheus server, in-memory trend tracker, and NDJSON writer.

```mermaid
flowchart LR
  Driver[nvidia-smi\nread-only] --> Collector[Collector\nGPU, process, topology, optional fields]
  Kernel[Kernel log access\njournalctl or dmesg] --> Collector
  Collector --> Analyzer[Analyzer\nhealth findings]
  Collector --> Tracker[Tracker\nper-process VRAM samples]
  Tracker --> Findings[Report findings]
  Analyzer --> Findings
  Probe[Optional model endpoint\nGET /metrics] --> Findings
  Collector --> Report[Complete report]
  Findings --> Report
  Report --> Text[Terminal text]
  Report --> JSON[JSON stdout]
  Report --> Exporter[Prometheus /metrics]
  Report --> History[NDJSON history]
```

## One Collection Cycle

```mermaid
sequenceDiagram
  participant CLI as gpu-watchman
  participant SMI as nvidia-smi
  participant API as Model endpoint
  participant P as Prometheus
  CLI->>SMI: query GPU fields
  SMI-->>CLI: inventory and telemetry CSV
  CLI->>SMI: query compute and graphics processes
  SMI-->>CLI: process VRAM CSV
  CLI->>SMI: optional MIG and throttle fields
  SMI-->>CLI: value or unsupported response
  opt -probe configured
    CLI->>API: GET /metrics (3 s timeout)
    API-->>CLI: supported inference metric samples
  end
  CLI->>CLI: analyze + compare trend samples
  opt -listen configured
    P->>CLI: GET /metrics
    CLI-->>P: latest complete report metrics
  end
```

## Optional Capability Rules

The base GPU query is required. MIG mode, throttle reasons, topology, Xid events, ECC counters, and retired-page counters depend on the GPU, driver version, host permissions, and NVIDIA feature support. Unsupported optional fields are omitted; they do not prevent the rest of the report from being collected.
