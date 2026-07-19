# GPU Watchman documentation

GPU Watchman is an inference-aware, read-only node observer written in Rust.

## Start here

1. [Install and release](installation.md)
2. [CLI reference](reference/cli.md)
3. [Operational profiles](reference/config.md)
4. [Architecture and collection flow](architecture.md)
5. [Report schema and compatibility](reference/report-schema.md)
6. [Production deployment runbook](operations/deployment.md)
7. [Troubleshooting and capability boundaries](troubleshooting.md)

## Feature guides

- [VRAM ownership and workload attribution](features/ownership.md)
- [Hardware health and transport](features/health.md)
- [Inference canaries, saturation benchmarks, probes, local runtime fingerprints, safetensors artifacts, capacity, and VRAM growth](features/inference.md)
- [Prometheus and HTTP API](exporter/README.md)
- [History and JSON operations](operations/history.md)
- [Before/after comparison and rollout gates](operations/comparison.md)
- [Active-canary baseline/candidate rollout comparison](operations/rollout.md)
- [Saturation benchmark baseline/candidate comparison](operations/benchmark-comparison.md)

## Contributors

- [Crate structure, extension rules, and quality gates](development.md)

GPU Watchman does not mutate hardware or workloads. Explicitly invoked **canary** and **benchmark saturation** commands do send bounded synthetic requests and can consume serving capacity or billable tokens; every passive workflow avoids that traffic. Linux-only **runtime inspect** reads bounded procfs and fixed NVIDIA kernel-module files for explicitly selected PIDs, executes no child utility, emits a path/hostname/environment/model/raw-argv-free Runtime Fingerprint v1 report, and always leaves compatibility **not_evaluated**. Artifact inspection reads bounded safetensors/index metadata but never tensor payload contents. It bounds strings during decoding, rereads the index and shard headers for same-descriptor consistency, accepts only supported dtype identifiers, proves exact shape-bit-to-payload length for every reported tensor, and records whether directory access was descriptor-anchored on the host platform. Capacity Report v3 remains an offline calculation with explicit assumptions. Only **capacity --artifact PATH** performs a fresh inspection and may use verified serialized tensor bytes as a stronger base-weight floor; its retained artifact evidence is path-free, and it grants no runtime placement or resident-memory proof. None of the passive or planning workflows allocates model memory or runs a stress workload.
