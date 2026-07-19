# Troubleshooting and boundaries

## Start with doctor

~~~sh
gpu-watchman doctor
gpu-watchman doctor --probe http://localhost:8000 \
  --probe-token-file /run/secrets/runtime-metrics-token \
  --format json
~~~

Doctor distinguishes driver access, available telemetry coverage, procfs attribution, and endpoint failures and prints the next check for each problem. It emits one check per schema-v3 source: **PASS** for **ok**, **WARN** for **partial**, **unavailable**, or **skipped**, and **FAIL** when the required base GPU collection cannot run. Doctor does not claim that every optional GPU capability exists.

## nvidia-smi not found

Confirm the driver utility in the same service/container environment:

~~~sh
command -v nvidia-smi
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv,noheader
~~~

Use **--nvidia-smi PATH** or **GPU_WATCHMAN_NVIDIA_SMI** when the utility is not on PATH.

## Container sees no GPUs

- Confirm NVIDIA Container Toolkit and the NVIDIA runtime class.
- Expose **NVIDIA_VISIBLE_DEVICES=all** and the **utility** driver capability.
- Do not rely on an ordinary GPU resource request for a node-wide observer; device plugins may expose only the allocated device.
- Use host PID namespace if host process attribution is required.
- Verify the non-root account can access injected device nodes.

## Missing owner, command, container, or pod

Attribution is Linux-only and best effort. Inspect **nvidia.processes** in the terminal **SOURCES** section or JSON **sources** array first. An **ok** source with zero records means the NVIDIA process queries completed; **partial** or **unavailable** includes a bounded reason. Then confirm procfs exposes the target PID and that hidepid, PID namespaces, or security policy do not obscure status, command line, or cgroups.

## Missing topology, MIG, throttle, ECC, or Xid

These capabilities vary by GPU model, driver, permissions, virtualization, and container environment. Use **--details** to show every source, including healthy ones. The canonical evidence is:

- **nvidia.optional** for MIG mode and throttle-reason fields;
- **nvidia.topology** for the topology matrix;
- **kernel.xid** for kernel-log inspection.

Xid lookup uses journalctl and then dmesg with bounded timeouts. Use **--no-xid** when kernel-log denial is expected; the source then says **skipped**, not **ok**. ECC and retired-page values are part of base GPU inventory but may be unsupported values on some devices.

To turn missing evidence into policy instead of an observation, use a canonical source or alias:

~~~sh
gpu-watchman snapshot --require-source processes,topology
gpu-watchman snapshot --require-source kernel.xid --format json
~~~

A required partial, unavailable, skipped, or absent source exits 2 even when **--fail-on never**. The finding explains which source failed and includes a finalized remediation recommendation.

## Endpoint is generic

The endpoint is reachable but no recognized runtime metric families were present. Inspect:

~~~sh
curl -fsS http://runtime:port/metrics
~~~

GPU Watchman recognizes vLLM, TGI, Triton, TensorRT-LLM, SGLang, and Ollama prefixes. A proxy login page or OpenMetrics endpoint with another prefix is intentionally reported as generic.

## Remote cleartext probe is rejected

Non-loopback passive probes require HTTPS by default. Prefer the final HTTPS metrics URL. For an explicitly trusted test network only, add **--allow-insecure-http**; the same policy can be stored as **monitor.inference.allow_insecure_http = true**. Passive probes do not follow redirects or inherit **HTTP_PROXY**, **HTTPS_PROXY**, or **ALL_PROXY**, so configure direct reachability to the exact target.

## Health endpoint is stale

Check **gpu_watchman_last_success_age_seconds**, **gpu_watchman_source_status**, source duration metrics, service stderr, driver latency, and endpoint timeouts. Optional collections run in parallel, but the required inventory must succeed. Increase **--command-timeout** only after confirming the driver is slow rather than wedged.

## Authenticated probes or API return 401

For inference endpoints, use **--probe-token-file PATH** or **GPU_WATCHMAN_PROBE_TOKEN_FILE**. Monitor, snapshot, doctor, and bundle workflows support the file form. For the embedded HTTP service, use **--api-token-file PATH** or **GPU_WATCHMAN_API_TOKEN_FILE**. File options conflict with their inline-token equivalents, reject empty or larger-than-64-KiB files, trim surrounding whitespace, require visible-ASCII values, validate the opened target plus every path/ancestor owner and write boundary, and never serialize the token. Root/current-user ownership with mode 0600 is the simplest local policy; Kubernetes-style root-owned mode-0440 projected symlinks are accepted when their full path is trusted.

Prefer a read-only mounted secret file over an inline CLI token. URL user information is rejected and complete queries are redacted from reports. GPU Watchman does not provide TLS; keep the listener on loopback or use **--allow-remote-listen** together with an API token behind an authenticated TLS proxy when crossing a trust boundary. Prometheus must send the same bearer token when scraping **/metrics**; Kubernetes discovery annotations alone cannot supply it.

## Explicit non-goals

GPU Watchman does not:

- reset GPUs or terminate processes;
- change clocks, power, compute mode, persistence, or MIG;
- run CUDA stress tests;
- guarantee capacity-planner results;
- replace DCGM diagnostics or vendor RMA procedures;
- provide a time-series database or retention service (optional NDJSON history remains file based);
- terminate TLS.

These boundaries reduce mutation and credential risk when the observer is deployed continuously, but operators must still validate host permissions, secret mounts, network exposure, and resource limits for their environment.
