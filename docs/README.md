# GPU Watchman Documentation

GPU Watchman is a compact, read-only CLI for checking NVIDIA GPU health and ownership on inference nodes. It queries the installed driver utility, analyzes the result locally, and can export the latest report to Prometheus or NDJSON.

## Start Here

1. [Install](installation.md)
2. [CLI reference](reference/cli.md)
3. [Architecture and collection flow](architecture.md)

## Feature Guides

- [VRAM ownership and process attribution](features/ownership.md)
- [GPU health, memory integrity, and transport](features/health.md)
- [Inference probes and VRAM-growth detection](features/inference.md)

## Operations

- [Prometheus exporter](exporter/README.md)
- [Report history and JSON](operations/history.md)
- [Troubleshooting and capability boundaries](troubleshooting.md)

## What It Does Not Do

GPU Watchman does not modify GPU configuration, reset a GPU, run CUDA stress tests, or estimate model-specific KV-cache capacity. It reports driver data and explicit analysis rules only.
