# GPU Watchman documentation

GPU Watchman is an inference-aware, read-only node observer written in Rust.

## Hosted resources

- [Product documentation](https://docs.yshubham.com/v2/products/watchman) is the canonical human-facing guide.
- [Tools registry](https://tools.yshubham.com/) hosts the verified installer and agent setup.
- [Watchman skill](https://github.com/bas3line/rool-repo/tree/main/skills/watchman) adds the operational workflow to compatible coding agents.
- [Forge mirror](https://git.yshubham.com/bas3line/watchman) mirrors the public source.
- [Platform status](https://status.yshubham.com/) reports public-service health.

## Start here

1. [Install and release](https://docs.yshubham.com/v2/watchman/reference#install-and-releases)
2. [CLI reference](https://docs.yshubham.com/v2/watchman/reference#cli-workflows)
3. [Operational profiles](https://docs.yshubham.com/v2/watchman/reference#operational-profiles)
4. [Architecture and collection flow](https://docs.yshubham.com/v2/watchman/reference#architecture-and-development)
5. [Report schema and compatibility](https://docs.yshubham.com/v2/watchman/reference#report-contract)
6. [Production deployment runbook](https://docs.yshubham.com/v2/watchman/rollouts-and-deployment)
7. [Troubleshooting and capability boundaries](https://docs.yshubham.com/v2/watchman/rollouts-and-deployment#troubleshoot-evidence-first)

## Feature guides

- [VRAM ownership and workload attribution](https://docs.yshubham.com/v2/watchman/health-and-collection)
- [Hardware health and transport](https://docs.yshubham.com/v2/watchman/health-and-collection)
- [Inference canaries, saturation benchmarks, probes, local runtime fingerprints, safetensors artifacts, capacity, and VRAM growth](https://docs.yshubham.com/v2/watchman/inference-and-capacity)
- [Prometheus and HTTP API](https://docs.yshubham.com/v2/watchman/reference#prometheus-and-http-api)
- [History and JSON operations](https://docs.yshubham.com/v2/watchman/rollouts-and-deployment#configuration-and-history)
- [Before/after comparison and rollout gates](https://docs.yshubham.com/v2/watchman/rollouts-and-deployment#compare-like-with-like)
- [Active-canary baseline/candidate rollout comparison](https://docs.yshubham.com/v2/watchman/rollouts-and-deployment#compare-like-with-like)
- [Saturation benchmark baseline/candidate comparison](https://docs.yshubham.com/v2/watchman/rollouts-and-deployment#compare-like-with-like)

## Contributors

- [Crate structure, extension rules, and quality gates](https://docs.yshubham.com/v2/watchman/reference#architecture-and-development)

GPU Watchman does not mutate hardware or workloads. Explicitly invoked **canary** and **benchmark saturation** commands do send bounded synthetic requests and can consume serving capacity or billable tokens; every passive workflow avoids that traffic. Linux-only **runtime inspect** reads bounded procfs and fixed NVIDIA kernel-module files for explicitly selected PIDs, executes no child utility, emits a path/hostname/environment/model/raw-argv-free Runtime Fingerprint v1 report, and always leaves compatibility **not_evaluated**. Artifact inspection reads bounded safetensors/index metadata but never tensor payload contents. It bounds strings during decoding, rereads the index and shard headers for same-descriptor consistency, accepts only supported dtype identifiers, proves exact shape-bit-to-payload length for every reported tensor, and records whether directory access was descriptor-anchored on the host platform. Capacity Report v3 remains an offline calculation with explicit assumptions. Only **capacity --artifact PATH** performs a fresh inspection and may use verified serialized tensor bytes as a stronger base-weight floor; its retained artifact evidence is path-free, and it grants no runtime placement or resident-memory proof. None of the passive or planning workflows allocates model memory or runs a stress workload.
