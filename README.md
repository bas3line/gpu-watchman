# GPU Watchman

GPU Watchman is a Rust control-room utility for operating AI inference nodes. Hardware collection is read-only; the explicitly invoked **canary** and **benchmark saturation** workflows send bounded synthetic inference requests. It joins GPU health, VRAM ownership, container and Kubernetes attribution, inference-runtime telemetry, privacy-safe local runtime fingerprints, active OpenAI-compatible validation and saturation evidence, exact-ladder offline benchmark comparison, local safetensors artifact inspection, Prometheus export, history and rollout comparison, and model-capacity planning in one binary.

It is designed for the question inference engineers actually ask during an incident: **is the GPU healthy, who owns its memory, is the serving runtime making progress, and what should I do next?**

## Why I open-sourced it

GPU Watchman started as a private tool I built to diagnose and operate real AI inference infrastructure. I am open-sourcing it because the same problems—unclear VRAM ownership, incomplete runtime telemetry, difficult capacity decisions, and weak rollout evidence—show up for inference engineers everywhere. My goal is to make this a practical, trustworthy utility that the community can use, audit, and improve for real production work. If it saves someone time during a GPU incident or helps them ship a safer inference deployment, it is doing its job.

## Install

Install the latest hosted release as **watchman**:

~~~sh
curl -fsSL https://tools.yshubham.com/watchman/install.sh | sh
watchman version
~~~

The hosted installer verifies the release checksum and supports macOS and Linux on x86-64 and ARM64.

Install the Watchman skill globally for every agent supported by the Skills CLI:

~~~sh
npx skills add bas3line/rool-repo \
  --skill watchman --agent '*' --global --yes
~~~

To build from source instead:

~~~sh
cargo build --locked --release
install -m 0755 target/release/gpu-watchman /usr/local/bin/watchman
~~~

## Quick start

~~~sh
# Full node snapshot
watchman snapshot --all --details

# Focused VRAM ownership and workload identity
watchman ps

# Inspect bounded local runtime evidence for explicit process IDs
watchman runtime inspect --pid 4242 --pid 4243 --format json

# Exercise a real OpenAI-compatible completion and enforce rollout SLOs
watchman canary --base-url http://127.0.0.1:8000/v1 \
  --model served-model --api-key-file /run/secrets/inference-api-key \
  --max-ttft 2s --max-e2e 10s

# Measure an explicit closed-loop concurrency ladder and gate one tested point
watchman benchmark saturation --model served-model \
  --concurrency-stages 1,2,4,8 --verify-concurrency 8 \
  --max-error-percent 1 --max-p95-ttft 2s --max-p95-e2e 10s

# Compare complete saved ladders without sending inference traffic
watchman benchmark compare baseline-benchmark.json candidate-benchmark.json \
  --max-p95-e2e-regression-percent 10 \
  --min-successful-rps-ratio 0.95 --fail-on-regression

# Live control-room view (watch is an alias for top)
watchman top --watch 5s \
  --probe http://vllm:8000,http://triton:8002,http://sglang:30000 \
  --allow-insecure-http

# Production observer: Prometheus, health API, report API, and history
# (serve is quiet unless --emit text|ndjson is requested)
watchman serve --interval 5s \
  --listen 127.0.0.1:9400 \
  --api-token-file /run/secrets/watchman-api-token \
  --probe http://vllm:8000 \
  --allow-insecure-http \
  --probe-token-file /run/secrets/runtime-metrics-token \
  --history /var/lib/gpu-watchman/history.ndjson
~~~

**top** has the visible **watch** alias; **serve** has the visible **exporter** alias. Every subcommand exposes only options that affect its workflow. Running without a subcommand performs one default snapshot; use **snapshot** explicitly when passing collection or output flags.

Repeatable operations can use a strict versioned profile. Configuration is never auto-discovered: pass **--config PATH** (or **GPU_WATCHMAN_CONFIG**) and optionally **--profile NAME** (or **GPU_WATCHMAN_PROFILE**). Values resolve in the order **built-in defaults < profile < environment < CLI**.

~~~sh
watchman config init gpu-watchman.toml
watchman --config gpu-watchman.toml config validate
watchman --config gpu-watchman.toml --profile local serve --no-api-auth
~~~

The generated local profile uses **fail_on = "never"** so a long-running observer keeps exporting the incident that made the node unhealthy. Opt into **--fail-on** for one-shot rollout gates, or deliberately for a supervisor policy that should terminate and restart a continuous process.

GPU collection workflows require Linux or macOS, a working NVIDIA driver, and **nvidia-smi** in PATH. **runtime inspect** is Linux-only and reads bounded evidence for explicit PIDs without requiring an NVIDIA driver or executing **nvidia-smi**, an engine, an interpreter, or a package manager; other platforms emit typed incomplete evidence. The hardware-independent **canary**, **benchmark saturation**, **artifact inspect**, and **capacity** workflows do not require a local GPU or NVIDIA driver. GPU Watchman never changes clocks, power limits, compute modes, MIG state, or workloads.

## What it does

- Inventories NVIDIA GPUs, driver, clocks, power, temperature, VRAM, utilization, ECC, retired pages, PCIe links, MIG mode, throttle reasons, and topology.
- Attributes compute and graphics VRAM to PIDs, users, commands, cgroups, container IDs, and Kubernetes pod UIDs.
- Provides a focused **watchman ps** ownership table for incident triage.
- Emits standalone **runtime_fingerprint_version: 1** evidence for up to 32 explicit local PIDs, with typed engine/framework candidates, mapped-library families, launch declarations, fixed NVIDIA kernel-module evidence, and an explicit **not_evaluated** compatibility assessment. The report contains no path, hostname, environment value, model identity, or raw argument.
- Detects memory pressure, thermal danger, ECC faults, page retirement, link degradation under load, throttling, Xid events, unattributed VRAM, and per-process VRAM growth.
- Records per-source coverage, latency, record counts, and bounded errors for inventory, process accounting, optional GPU fields, topology, and kernel Xid logs. Selected sources can be made fail-closed.
- Probes Prometheus metrics from vLLM, TGI, Triton, TensorRT-LLM, SGLang, and Ollama concurrently with driver collection, with HTTPS-by-default remote transport, no redirects, and no ambient proxy inheritance.
- Normalizes running/queued requests and KV pressure, then derives request/token throughput, error/preemption rates, interval means, and p50/p95/p99 request-latency, TTFT, TPOT, and queue-time estimates from bounded monotonic runtime histograms.
- Runs bounded OpenAI-compatible chat-completion canaries with streaming TTFT, end-to-end latency, expected-output checks, authoritative output-token throughput, stable non-secret workload identity, concurrency, and CI-friendly SLO gates.
- Runs an explicit closed-loop concurrency ladder with excluded exact-stage warmup, simultaneous worker release, fixed samples per worker, attempted and successful request rates, p95 latency, authoritative completion-token goodput, fail-closed evidence gates, a bounded saturation signal, and optional exact-stage deployment verification.
- Compares complete saved saturation ladders offline at every exact stage, reconstructing source evidence before p95 latency, successful-RPS, completion-token-goodput, error-point, and source-policy gates produce typed **pass**, **regression**, or **not_evaluable** outcomes.
- Compares saved baseline/candidate canaries offline with fail-closed identity checks and opt-in p95 TTFT, p95 end-to-end, p50 output-throughput-ratio, and success-drop rollout gates.
- Serves **/livez**, **/metrics**, **/healthz**, and **/api/v1/report** through a bounded HTTP server. Authentication is required by default even on loopback; **--no-api-auth** is an explicit loopback-only debugging exception, while a non-loopback listener also requires **--allow-remote-listen**.
- Writes stable schema-versioned NDJSON and summarizes runtime/source availability, peaks, non-OK telemetry sources, and recurring findings with **watchman history**.
- Compares before/after reports and can fail a deployment gate when GPUs or telemetry sources disappear, sources degrade, endpoints go down, or new warning/critical findings appear.
- Inspects one safetensors file or a sharded safetensors index without reading tensor payloads, emitting exact serialized tensor/header/container bytes, dtype summaries, and fail-closed metadata/dtype/offset/index verification without paths or tensor names.
- Emits fail-closed **capacity_version: 3** evidence for TP/PP/DP/EP topology, optional verified-artifact weight floors, explicit placement upper bounds, worst-rank weight/KV-cache memory, headroom, and full-context concurrency without requiring GPU hardware.
- Derives model geometry from a local Hugging Face **config.json**, records parameter-count provenance, requires explicit resident counts and expert metadata for detected MoE models, and limits dense auto-estimation to audited standard layouts with exact **model_type** values **llama** or **mistral**.
- Produces 0600-permission incident bundles and completions for Bash, Zsh, Fish, PowerShell, and Elvish.
- Loads explicit, strict **config_version = 1** operational profiles for snapshot, top, serve, ps, canary, and benchmark target/workload workflows, with source-aware overrides and no implicit file discovery.

## Inference workflows

~~~sh
# Validate the node and serving endpoints
watchman doctor --probe http://vllm:8000 \
  --allow-insecure-http \
  --probe-token-file /run/secrets/runtime-metrics-token

# Verify a real generation path before admitting traffic
watchman canary \
  --base-url https://candidate.example/v1 --model served-model \
  --api-key-file /run/secrets/inference-api-key \
  --count 5 --concurrency 2 \
  --max-ttft 2s --max-e2e 10s --min-success-percent 100

# Characterize only these closed-loop concurrency points, then verify point 8
watchman benchmark saturation \
  --base-url https://candidate.example/v1 --model served-model \
  --api-key-file /run/secrets/inference-api-key \
  --concurrency-stages 1,2,4,8 \
  --warmup-requests-per-worker 2 --requests-per-worker 20 \
  --verify-concurrency 8 --max-error-percent 1 \
  --max-p95-ttft 2s --max-p95-e2e 10s \
  --min-completion-token-goodput-per-second 200

# Gate a candidate ladder against the same benchmark identity and source policy
watchman benchmark compare baseline-benchmark.json candidate-benchmark.json \
  --max-p95-ttft-regression-percent 10 \
  --max-p95-e2e-regression-percent 10 \
  --min-successful-rps-ratio 0.95 \
  --min-completion-token-goodput-ratio 0.95 \
  --max-error-percent-increase 1 --fail-on-regression

# Inspect local runtime declarations and mapped-library evidence
watchman runtime inspect --pid 4242 --pid 4243 --format json

# Inspect checkpoint storage metadata without loading tensor payloads
watchman artifact inspect /models/served-model/model.safetensors.index.json \
  --format json

# Estimate a 70B 4-bit model on two 80 GiB GPUs
watchman capacity \
  --params 70 --weight-bits 4 \
  --tp 2 --gpu-vram 80 --max-shared-rank-weight-percent 50 \
  --layers 80 --kv-heads 8 --head-dim 128 \
  --max-kv-heads-per-rank 4 --context 32768 --concurrency 8

# Or derive geometry from config and strengthen the weight floor with the checkpoint
watchman capacity --model-config ./config.json \
  --artifact /models/served-model/model.safetensors.index.json \
  --artifact-residency-multiplier 1.25 \
  --gpu-vram 80 --tp 2 --weight-bits 4 \
  --max-shared-rank-weight-percent 50 \
  --context 32768 --concurrency 8

# Analyze a production capture
watchman history /var/lib/gpu-watchman/history.ndjson

# Gate a rollout against a known-good report
watchman compare before.json after.json --fail-on-regression

# Gate a candidate canary against the exact same synthetic workload
watchman rollout baseline-canary.json candidate-canary.json \
  --max-p95-ttft-regression-percent 10 \
  --max-p95-e2e-regression-percent 10 \
  --min-output-tps-ratio 0.9 \
  --max-success-drop-percent 1 --fail-on-regression

# Create an incident handoff without overwriting an existing file
watchman bundle --output node-17-support.json \
  --probe http://vllm:8000 \
  --allow-insecure-http
~~~

The canary is a small correctness and SLO probe, not a capacity or load benchmark. Its built-in prompt contains no customer data, asks for exactly **gpu-watchman-ok**, and reports workload identity **builtin-v1**. Custom prompts require an explicit non-secret **--workload-id** (or a paired profile identity); prompt content is never stored or hashed. Output-token throughput is reported only for streamed responses with authoritative completion-token usage and at least two completion tokens; GPU Watchman never guesses token counts from text. Non-positive completion usage or a count above the request's **max_tokens** is discarded as implausible, so it cannot satisfy a throughput gate. Prefer **--api-key-file** and **--prompt-file** because direct **--api-key**, **--prompt**, and **--expect** values can be visible in the process list and shell history. Treat expectation markers and workload IDs as non-sensitive. Remote endpoints require HTTPS unless **--allow-insecure-http** explicitly opts into cleartext; loopback requests bypass ambient proxy variables. Credentials, prompts, and generated content are excluded from output. Response retention defaults to 1 MiB per canary attempt, and plans whose conservative aggregate working-set estimate exceeds 768 MiB are rejected. The estimate includes concurrent request clones, bounded response and JSON-parser amplification, expectation-matcher state, and retained attempt evidence. Each canary request timeout is capped at 5 minutes, and request waves multiplied by that timeout cannot exceed 15 minutes.

Saturation Benchmark v1 is a bounded, single-process, closed-loop fixed-concurrency experiment—not an open-loop arrival-rate, distributed, soak, or adaptive-breakpoint test. Stages must be explicit, unique, strictly increasing, start at one, contain at most eight points, and never exceed concurrency 64. Every exact point receives an excluded warmup; each worker then executes exactly **requests-per-worker** measured requests after a simultaneous release. The default 10-second request timeout and 128 KiB response cap let the advertised ladder fit its bounded wave budget and reach useful concurrency under the same conservative 768 MiB working-set ceiling. Total attempts, requested completion tokens, aggregate prompt/response budgets, timeout, and worst-case request waves are bounded before network access. A warmup or measured-stage abort stops higher load after zero successes or the configured severe error rate. **--verify-concurrency N** gates only that exact tested point and exits 2 on failure or missing evidence; p95 gates and the saturation heuristic require at least 20 finite successful samples per adjacent point, and completion-token goodput requires plausible authoritative usage for every successful request. The descriptive plateau heuristic normalizes RPS growth by the exact adjacent concurrency growth, so narrow stage increments are not mistaken for saturation. Non-positive token counts, prompt counts above 10,000,000 per request, and completion counts above requested **max_tokens** are omitted and make affected token evidence incomplete. A plateau signal alone does not fail the command. Closed-loop coordinated omission, a bottleneck in this load generator or its network, prefix caching, and unrelated traffic can bias results. The highest accepted tested point is neither production capacity nor a deployment recommendation, and concurrency is not server batch size or GPU occupancy.

Benchmark Comparison v1 is an offline gate over two saved Saturation Benchmark v1 reports. Both reports must be internally reconstructable, complete, ordered in time, and identical in workload ID, requested model, route, stream mode, full plan/schedule, and source policy. Endpoint origins may differ and are deliberately absent from comparison output. Every selected quantitative gate is evaluated at every exact concurrency point and requires at least 20 relevant samples per side; missing usage, incompatible identity, zero ratio baselines, and undersampling become **not_evaluable**, never a pass. Canonical **duration_ns** and **timeout_ns** make source rate denominators exactly reproducible; millisecond fields remain display conveniences. Internal consistency does not authenticate a coherently fabricated file, and matching workload IDs/expectation booleans do not prove hidden prompt text matches.

Runtime Fingerprint v1 is observation evidence, never a compatibility verdict. On Linux, **runtime inspect** pins each **/proc/PID** directory, bookends collection with process start-time reads, rereads **cmdline** for exact stability, and bounds the non-atomic **maps** observation. It retains only fixed engine/framework/library enums, strictly numeric mapped-filename and kernel-module version components, typed launch declarations, and a boolean saying whether a model reference was present. Paths, hostnames, environment values, model identities, raw argv, map lines, and operating-system diagnostics are never serialized. Driver absence does not make otherwise complete process evidence incomplete. An incomplete report is printed before the command exits 2 by default; **--allow-incomplete** changes only that exit to 0.

Artifact Report v1 proves bounded checkpoint-storage facts only. It validates bounded safetensors/index metadata, an explicit dtype allowlist, exact **elements × dtype bits** for every tensor, complete no-hole payload offsets, sharded-index membership, and declared total size while reading no tensor payload bytes. Bounded string visitors reject oversized names, dtypes, shard names, and metadata during decoding before tool-owned retention. The complete bounded index and each shard's length-prefix/header are reread from the same descriptor and compared; Unix file snapshots also compare device, inode, length, and nanosecond modification/change times. Sub-byte tensors must occupy a whole number of bytes and unknown dtypes fail closed; every successful v1 report therefore has zero shape/payload-unverified tensors. Paths, shard filenames, and tensor names are used transiently for validation but never serialized. On Unix, selected files and shards are opened relative to pinned directory descriptors with final-component no-follow protection; machine output records whether that descriptor anchoring was available. Serialized checkpoint bytes do not establish runtime GPU residency, quantization workspaces, or TP/PP/DP/EP placement.

Capacity Report v3 models explicit tensor, pipeline, data, and expert parallelism with **--tp**, **--pp**, **--dp**, and **--ep**, then evaluates the rank with the largest modeled placement. Topology degrees are not proof of even byte or layer placement: by default the worst rank is charged 100% of shared weights, 100% of routed-expert weights when present, all KV heads, and—for PP—100% of every weight class plus all layers. Supply only independently deployment/runtime-evidenced upper bounds through **--max-shared-rank-weight-percent**, **--max-expert-rank-weight-percent**, **--max-kv-heads-per-rank**, **--max-stage-component-weight-percent**, and **--max-stage-layers**.

**--artifact PATH** explicitly inspects a safetensors file, index, or directory and uses its verified serialized tensor bytes only to strengthen the logical base-weight floor. With **P = parameters_billion × 1e9 × weight_bits / 8** and **A = ceil(serialized_tensor_bytes × artifact_residency_multiplier)**, v3 selects **max(P, A)** and applies **--weight-overhead-percent** once afterward. The multiplier is finite and in **[1, 1000]**; it defaults to 1 only when **--artifact** is present and cannot be supplied alone. The product is rounded upward to a whole byte, and raw plus adjusted artifact bytes are capped at **8,000,000,000,000,000**. Machine output records a path-free artifact aggregate—including whether descriptor anchoring was available—and the selected **parameter_precision_estimate** or **artifact_residency_floor** basis. Artifact evidence never replaces required parameter/config geometry or MoE expert metadata.

**--gpus** remains compatible: without any topology-degree flag it means legacy tensor parallelism; with any of those four degree flags it becomes an exact assertion that the GPU count equals **TP × PP × DP**. Data parallelism never divides per-rank weights or KV cache, and **--concurrency** is resident on every DP replica.

Capacity output is an engineering estimate, not an allocation guarantee. Weight overhead is conservatively charged without TP/EP sharding. Activation memory, CUDA graphs, allocator fragmentation, communication workspaces, speculative/draft-model allocations, and checkpoint-format conversion into resident runtime memory are not modeled. Artifact bytes enter capacity only through explicit **--artifact** and can never reduce the parameter/precision estimate. They still do not prove loaded residency, expert composition, or TP/PP/DP/EP placement; calibrate the multiplier and overheads against the deployed loader and runtime.

## Operator contract

- Exit 0: command completed; any selected health, canary, benchmark verification/comparison, capacity, rollout, or runtime-completeness policy passed. **benchmark compare** also exits 0 after reporting a non-pass outcome when enforcement was not requested. A benchmark saturation signal alone does not change this exit. For **runtime inspect**, this includes an incomplete report only when **--allow-incomplete** is explicit.
- Exit 1: usage, validation, collection, local file, setup, or encoding failure prevented the requested workflow from running. Invalid, duplicate, zero, or more than 32 runtime PID selections fail here before a report is emitted.
- Exit 2: the overall canary policy failed; a saturation benchmark aborted or its requested exact-stage verification failed/not-evaluable; enforced **benchmark compare** evidence regressed, was incompatible, or was not evaluable; the selected **--fail-on** threshold was crossed; doctor failed; capacity did not fit; a selected comparison regressed; a one-shot required source was incomplete; or **runtime inspect** emitted incomplete process evidence without **--allow-incomplete**. Completed evidence is emitted before this exit. Continuous monitors keep exporting source violations unless a **--fail-on** threshold stops them.
- Hardware JSON reports use **schema_version: 3**, including explicit telemetry-source coverage. Runtime inspection uses **runtime_fingerprint_version: 1**; saturation benchmarks use **saturation_benchmark_version: 1**. Host-level hardware findings retain **gpu_index: -1** for compatibility.
- **--format ndjson** is always compact one-record-per-line output. Repeated pretty JSON is rejected; use NDJSON for streams.
- **--color auto|always|never** affects human text only. Auto honors TTY detection, **NO_COLOR**, and **TERM=dumb**.
- URL user information is rejected, and complete query strings are redacted. Passive probes refuse redirects and ambient proxies; non-loopback cleartext HTTP requires **--allow-insecure-http**. Passive probes and the HTTP API use **GPU_WATCHMAN_PROBE_TOKEN[_FILE]** and **GPU_WATCHMAN_API_TOKEN[_FILE]**. Active canaries use **GPU_WATCHMAN_INFERENCE_URL**, **GPU_WATCHMAN_INFERENCE_MODEL**, and **GPU_WATCHMAN_INFERENCE_API_KEY[_FILE]**.
- A shared passive-probe bearer token is admitted only when every probe URL has the same exact scheme, host, and port. GPU Watchman rejects a multi-origin authenticated probe set before opening a connection; run separate invocations for separately authenticated origins.

## Deploy

- [Installation and releases](docs/installation.md)
- [CLI reference](docs/reference/cli.md)
- [Operational profile reference](docs/reference/config.md)
- [Architecture](docs/architecture.md)
- [Inference canary, runtime, artifact, and capacity guide](docs/features/inference.md)
- [Report comparison and rollout gates](docs/operations/comparison.md)
- [Active-canary rollout comparison](docs/operations/rollout.md)
- [Saturation benchmark baseline/candidate comparison](docs/operations/benchmark-comparison.md)
- [Report schema reference](docs/reference/report-schema.md)
- [Prometheus/API reference](docs/exporter/README.md)
- [systemd unit](packaging/systemd/gpu-watchman.service)
- [Kubernetes DaemonSet](packaging/kubernetes/daemonset.yaml)
- [Prometheus alert rules](packaging/prometheus/alerts.yaml)
- [Security policy](SECURITY.md)

The Kubernetes manifest is an isolated safe base: it expects an NVIDIA runtime class, targets labeled GPU nodes, exposes all node GPUs through the runtime’s utility capability, mounts the **gpu-watchman-api** Secret, binds only pod loopback, and deliberately creates no Service. Replace its fail-closed image digest and add a separate authenticated-mTLS proxy/mesh overlay before remote scraping.

## Develop

~~~sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
cargo build --locked --release
~~~

The crate is organized by responsibility rather than command: stable data contracts, telemetry adapters, inference probes, analysis, operational workflows, planning, presentation, and the CLI application layer. See the [architecture guide](docs/architecture.md) and [development guide](docs/development.md) before adding a backend or rule.
