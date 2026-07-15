# GPU Watchman

Small, dependency-free NVIDIA GPU and inference-runtime watchman for macOS and Linux. It calls the NVIDIA driver utility already installed on the host, so it does not ship or link NVML.

```sh
go build -trimpath -ldflags='-s -w' -o gpu-watchman ./cmd/gpu-watchman
./gpu-watchman -all
./gpu-watchman -format json
./gpu-watchman -watch 2s
./gpu-watchman -watch 5s -listen :9400 -history /var/log/gpu-watchman.ndjson
./gpu-watchman -watch 5s -probe http://localhost:8000
```

`nvidia-smi` must be installed and accessible in `PATH`. The tool never changes GPU configuration.

## Layers

1. `internal/collector`: runs and parses read-only NVIDIA driver queries.
2. `internal/analyzer`: turns raw readings into actionable health findings.
3. `internal/render`: produces compact terminal or JSON reports.

## Inference Operations

The collector attributes both compute and graphics VRAM processes. On Linux, it also enriches each process with the OS user, command line, and cgroup path, which can identify a container or Kubernetes workload. It samples live runs to flag a process that grows by at least 256 MiB between samples.

`-probe` fetches a metrics endpoint from a vLLM, Triton, TGI, or Ollama deployment and records its exposed inference metrics. It deliberately does not guess vendor-specific semantic names: the endpoint's original metric keys are retained in JSON.

`-listen` starts a dependency-free Prometheus endpoint at `/metrics`. It exports GPU VRAM, utilization, temperature, power, ECC, retired pages, process count, per-process VRAM, and inference endpoint availability. `-history` writes one complete report per line as a 0600-permission NDJSON file.

## Hardware Features

1. GPU inventory
2. Model names
3. GPU UUIDs
4. Driver version
5. Performance state
6. GPU temperature
7. Fan percentage
8. Power draw
9. Power-limit visibility
10. Core clock
11. Memory clock
12. Maximum core clock
13. Maximum memory clock
14. Total VRAM
15. Used VRAM
16. Free VRAM
17. GPU utilization
18. Memory-controller utilization
19. PCIe current generation
20. PCIe maximum generation
21. PCIe current link width
22. PCIe maximum link width
23. Compute mode
24. Persistence mode
25. ECC status
26. Corrected ECC errors
27. Uncorrected ECC errors
28. Retired VRAM pages
29. Per-process PID
30. Per-process name
31. Per-process VRAM ownership
32. VRAM pressure warnings
33. Unattributed VRAM detection
34. Thermal health warnings
35. Power-limit detection
36. Idle VRAM-reservation detection
37. PCIe link health checks
38. ECC and memory-fault alerts
39. Live refresh mode
40. Machine-readable JSON export

41. Graphics-process accounting
42. Linux user, command, and cgroup attribution
43. MIG mode visibility
44. Active clock-throttle reasons
45. GPU topology report (NVLink/PCIe matrix when exposed by the driver)
46. NVIDIA Xid event detection from accessible kernel logs
47. Per-process VRAM growth alerts
48. Inference endpoint reachability and latency
49. vLLM, Triton, TGI, and Ollama metrics capture
50. Prometheus scrape endpoint
51. Append-only NDJSON history

## Size target

The module uses only the Go standard library. Build with the documented flags to strip debug metadata; the resulting executable is verified below 10-15 MB on the development platform. Runtime memory depends on the Go runtime, collection interval, endpoint payloads, and the host `nvidia-smi` process, so it cannot be guaranteed across operating systems and GPU driver versions.

## Limits

GPU Watchman is read-only. It does not perform a CUDA stress test, reset a GPU, or infer model-specific KV-cache capacity from unrelated counters. Those actions require an explicit workload profile or vendor/runtime SDK and should be added as opt-in modules, not silently run by a node-health agent. Optional fields depend on GPU model, driver version, permissions, and whether MIG/NVLink are present.
