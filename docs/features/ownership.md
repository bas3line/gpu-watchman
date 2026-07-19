# VRAM ownership and workload attribution

GPU Watchman queries both NVIDIA compute and graphics accounting. Duplicate GPU UUID and PID pairs are kept once.

On Linux, each PID is enriched from procfs with:

- host user name or numeric UID;
- null-separated command line converted to readable text;
- full cgroup membership;
- likely container ID;
- likely Kubernetes pod UID.

~~~mermaid
flowchart LR
  C[Compute accounting] --> D[Deduplicate UUID and PID]
  G[Graphics accounting] --> D
  D --> P[Linux procfs]
  P --> O[PID user command cgroup container pod]
~~~

Container and pod extraction is best effort because cgroup formats differ across systemd, cgroup versions, Docker, containerd, CRI-O, and Kubernetes releases. The original cgroup string is retained for verification.

## Investigation

~~~sh
gpu-watchman ps
gpu-watchman snapshot --all --details
gpu-watchman snapshot --gpu 0,GPU-aaaaaaaa --format json
~~~

Processes are sorted by VRAM use in detailed output. JSON preserves full command and cgroup values.

The focused **ps** command requires the **nvidia.processes** telemetry source by
default. It still emits its text or machine-readable evidence when the driver
accounting queries are partial or unavailable, but exits 2 so an empty table
cannot be mistaken for proof that no workload owns VRAM. Use
**--allow-incomplete** only for an explicitly best-effort investigation; the
machine contract continues to report **complete: false** and the source state.

## Unattributed VRAM

**unattributed-vram** appears when more than 256 MiB is used but driver accounting reports no process. Possible causes include:

- a host/container PID namespace mismatch;
- permissions or accounting limitations;
- display or graphics contexts;
- stale CUDA contexts;
- MIG visibility differences;
- a process exiting between inventory and accounting.

It is a warning, not proof of a leak.

## Data sensitivity

Commands and cgroups can reveal model paths, tenant identifiers, and deployment names. History files and bundles are mode 0600 on Unix. Review a bundle before sharing it outside the incident team.
