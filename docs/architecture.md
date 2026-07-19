# Architecture

GPU Watchman is a Rust library plus a thin CLI. The library separates collection, normalization, analysis, trend state, presentation, and serving so hardware parsers and policy rules can be tested without GPU hardware.

~~~mermaid
flowchart LR
  SMI[nvidia-smi] --> C[NVIDIA collector]
  PROC[Linux procfs] --> C
  RPROC[Explicit Linux procfs PIDs] --> FINGERPRINT[Bounded runtime fingerprint collector]
  RDRIVER[Fixed NVIDIA kernel-module files] --> FINGERPRINT
  LOG[journalctl or dmesg] --> C
  RUNTIME[Inference metrics] --> P[Bounded concurrent probes]
  OPENAI[OpenAI-compatible chat API] --> CANARY[Bounded active canary]
  OPENAI --> BENCH[Bounded saturation ladder]
  ARTIFACT[Local safetensors file or index] --> INSPECT[Bounded metadata-only inspector]
  MODEL[Local model config or explicit geometry] --> CAPACITY[Topology-aware capacity planner]
  PROFILE[Explicit versioned profile] --> APP[Typed workflow resolver]
  APP --> C
  APP --> P
  APP --> CANARY
  APP --> BENCH
  APP --> INSPECT
  APP --> CAPACITY
  APP --> FINGERPRINT
  INSPECT -->|explicit --artifact verified byte floor| CAPACITY
  C --> A[Configurable analyzer]
  P --> A
  C --> T[Process trend tracker]
  T --> A
  A --> R[Schema v3 report]
  R --> TEXT[Operator text]
  R --> JSON[JSON and NDJSON]
  R --> API[Metrics health report API]
  R --> H[History and support bundle]
  CANARY --> CR[Canary v2 result]
  CR --> TEXT
  CR --> JSON
  BENCH --> BR[Saturation Benchmark v1]
  BR --> TEXT
  BR --> JSON
  INSPECT --> AR[Artifact Report v1]
  AR --> TEXT
  AR --> JSON
  CAPACITY --> CP[Capacity Report v3]
  CP --> TEXT
  CP --> JSON
  FINGERPRINT --> RF[Runtime Fingerprint v1]
  RF --> TEXT
  RF --> JSON
~~~

## Collection cycle

1. One required CSV query collects the stable GPU inventory.
2. Process accounting, optional fields, topology, and Xid logs run in parallel.
3. Compute and graphics process queries also run in parallel and are de-duplicated by GPU UUID plus PID.
4. Inference endpoints run concurrently with the entire driver collection.
5. Each external driver/log command has its own deadline; every probe has a timeout, response-size limit, sample limit, and global concurrency bound.
6. Each hardware/log collector returns its value together with source state, duration, record count, and a bounded error when applicable.
7. The analyzer and in-memory trend tracker emit findings. Analysis finalization attaches recommendations and computes health status and summary exactly once.

The collector continuously drains child stdout and stderr, preventing large process lists or logs from deadlocking full pipe buffers. It retains at most 8 MiB of stdout and 64 KiB of stderr internally. Before a failure enters a report or support bundle, stderr is reduced to a bounded operational classification such as permission denied, unsupported field, no devices, or omitted; arbitrary wrapper diagnostics are never serialized. Pipe draining shares the command's hard deadline, and Unix commands run in a dedicated process group that is terminated on timeout, so a wrapper descendant that inherits the pipes cannot hang collection. Topology is cached until the sorted GPU UUID inventory changes.

## Optional capability rules

The inventory query is required. Process accounting, MIG mode, throttle reasons, topology, kernel logs, ECC counters, and retired pages vary by model, driver, permissions, and host. Failure of an optional source never discards valid base inventory, but it is not silently converted into an indistinguishable empty result: schema-v3 reports record **ok**, **partial**, **unavailable**, or **skipped** source evidence.

Operators can promote any named source into a fail-closed requirement with **--require-source**. A partial required source creates a warning finding; an unavailable, skipped, or absent required source creates a critical finding. Either condition exits 2 for a one-shot collection even when the general **--fail-on** policy is **never**. Continuous workflows keep publishing the violation unless **--fail-on** requests termination.

An unavailable fan sensor is not unhealthy because many datacenter GPUs are passively cooled. An unavailable temperature sensor is a warning.

## Data contract

Reports use **schema_version 3** and include:

- UTC collection time and duration;
- host, status, and summary envelope;
- available GPU and process records;
- actionable findings with recommendations;
- optional topology and Xid entries;
- endpoint status and fixed-schema normalized inference signals, including bounded cumulative histogram snapshots and interval p50/p95/p99 latency estimates, with endpoint-controlled Prometheus family names and labels discarded;
- per-source state, duration, record count, requirement flag, and bounded diagnostic error.

Host findings serialize **gpu_index: -1** as part of the stable external schema. In Rust they are represented as an optional device index.

Active canary output is intentionally a separate **canary_version: 2** contract. It contains a non-secret workload identity, privacy-safe effective policy, aggregate distributions, canonical gates, and bounded per-attempt evidence rather than pretending to be a hardware snapshot. Hardware history and **compare** therefore do not accept canary results; the offline **rollout** workflow emits its own **rollout_version: 1** baseline/candidate contract.

Active concurrency characterization emits **saturation_benchmark_version: 1**. It records the explicit closed-loop schedule, aggregate excluded warmup for every reached point, measured per-stage attempts/distributions/goodput/gates, adjacent-point scaling evidence, a bounded heuristic assessment, optional exact-stage verification, and fixed nonclaims. Its independent measured-attempt type cannot retain response model, finish reason, arbitrary failure message, prompt, expectation, credential, or generated content. It is not accepted by hardware history, compare, rollout, or capacity planning.

The focused **ps** machine view is a third contract, **process_view_version: 1**. It contains only collection identity, process-source completeness, and flattened process ownership records; it is not silently substituted with the full report schema.

Explicit local process inspection emits **runtime_fingerprint_version: 1**. It contains path-free and hostname-free host classification, fixed NVIDIA kernel-module evidence, typed source completeness, engine/framework candidates, mapped-library families and numeric filename-version evidence, typed launch declarations, and an assessment whose compatibility state is always **not_evaluated**. It excludes raw command lines, paths, environment values, model identities, map records, and arbitrary diagnostics. It is not a hardware report and is not accepted by history or comparison workflows.

Local safetensors inspection emits **artifact_version: 1**. Artifact Report v1 contains only aggregate serialized-storage facts, dtype summaries, verification evidence, and caveats. It excludes paths, shard filenames, tensor names, arbitrary safetensors metadata, and payload contents. It is not a hardware report, a capacity report, or accepted by history/comparison workflows.

Hardware-independent planning has its own **capacity_version: 3** contract. It retains the effective model and topology input, safe parameter-count provenance, optional path-free aggregate artifact evidence, the selected parameter-versus-artifact weight basis, validated placement evidence, total and worst-rank weight/KV derivations, worst-rank memory and headroom, the fit decision, stable assumption codes, and caveats. Reports without artifacts still use version 3 with a null artifact field. Capacity output is neither a schema-v3 hardware sample nor accepted by hardware history/comparison workflows.

## Crate boundaries

~~~text
src/
├── application/   CLI parsing, dispatch, monitor lifecycle, exit policy
├── config/        strict profile schema, bounded I/O, validation, safe display
├── domain/        stable hardware, runtime, process, endpoint, finding contracts
├── telemetry/     NVIDIA adapter, bounded runtime evidence, process attribution
├── inference/     bounded passive probes plus active OpenAI request/stream parsing
├── analysis/      health rules, recommendations, cross-cycle trend state
├── operations/    runtime/canary/benchmark orchestration, bounded worker pool, history/comparison
├── planning/      bounded artifact inspection, model derivation, worst-rank capacity math
├── presentation/  terminal/JSON rendering and HTTP/Prometheus serving
├── lib.rs         collection-cycle orchestration
└── main.rs        thin executable adapter
~~~

Dependencies point inward toward **domain**. Telemetry and inference adapters create domain records; analysis consumes them; presentation and operations consume finalized reports. Host identity I/O lives in **telemetry::host**, outside the stable domain types. Findings are created as domain data, while recommendation policy is attached by the analysis finalizer. The executable contains no collection logic. This keeps parsers and rules testable without a GPU and gives future AMD, Intel, DCGM, or runtime adapters a defined home.

The configuration layer performs no GPU, endpoint, or secret-file access. It validates one explicit versioned document, resolves file references against that document's directory, and returns a selected partial profile. The application layer alone merges built-ins, profile fields, environment inputs, and typed CLI overrides before invoking a workflow.

## Runtime-fingerprint boundary

**gpu-watchman runtime inspect** accepts only explicitly supplied positive PIDs, rejects duplicates, and admits at most 32 targets. It never scans the process table. Runtime Fingerprint v1 collection is Linux-only; a non-Linux invocation emits fixed **unsupported_platform** identity, command-line, and memory-map source evidence for each target and is incomplete.

For each Linux target, the collector opens **/proc/PID** once with **O_DIRECTORY|O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK**. It opens only the fixed children **stat**, **cmdline**, and **maps** relative to that descriptor with **openat(O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK)**. The process start time from **stat** is read before and after all other observations; an exit or changed start time discards transient evidence and becomes **process_exited** or **pid_reused**. **cmdline** is read twice and interpreted only when both bounded byte sequences match exactly. **maps** is one explicitly non-atomic observation; the report's assessment preserves that limitation.

Reads and parsing are bounded: each **stat** read is at most 8 KiB; each **cmdline** read is at most 256 KiB with 4,096 arguments and 16 KiB per argument; **maps** is at most 8 MiB with 65,536 records and 16 KiB per record; at most 64 distinct recognized mapped-library facts survive per process. Each fixed driver file is limited to 4 KiB and each retained numeric version token to 64 bytes. Limit, permission, absence, identity-race, malformed-input, and unsupported-platform results reduce to fixed reason codes rather than operating-system text.

Engine recognition is restricted to **vllm**, **tgi**, **triton**, **sglang**, and **tensorrt_llm** entrypoint patterns. Framework candidates are restricted to **pytorch**, **tensorflow**, **onnx_runtime**, and **tensorrt**, and arise only from fixed mapped-library families. Only after exactly one engine identity is recognized does the collector reduce supported argv flags into typed tensor/pipeline/data parallel sizes, context limit, canonical dtype, canonical quantization, KV-cache dtype, and **model_reference_present**. Duplicate declarations become **ambiguous**, unsupported values become **present_unparsed**, and absence remains **not_observed**; argument strings and model references are discarded.

The collector never starts a child process. In particular, it does not execute **nvidia-smi**, an inference engine, Python, a package manager, **ldconfig**, or a version command. Driver evidence comes only from bounded no-follow reads of **/sys/module/nvidia/version** and **/proc/driver/nvidia/version** and describes the NVIDIA kernel module, not the CUDA user-mode driver, toolkit, or runtime. Missing or unavailable driver evidence does not make stable process evidence incomplete. Because mapped filenames are not package versions, argv declarations are not effective configuration, memory maps are non-atomic, and no model artifact is bound, every v1 assessment is **compatibility: not_evaluated**.

Completeness requires Linux plus a stable identity and observed identity, command-line, and memory-map sources for every requested PID. The complete or incomplete report is always rendered before policy is applied. Incomplete evidence exits 2 by default; **--allow-incomplete** changes only that exit to 0. Invalid target selection and serialization/encoding failures are fatal exit-1 errors.

## Artifact-inspection boundary

The artifact inspector resolves one explicit **.safetensors** file, one **.safetensors.index.json**, or an unambiguous directory. Directory enumeration, index bytes, per-shard and combined header bytes, shard/tensor/dtype counts, tensor-name storage, shard-name length, tensor rank, dtype length, header metadata entries/strings/bytes, and index metadata entries/key length all have hard ceilings enforced during resolution, bounded parsing, and aggregation. Directory discovery preserves raw names: non-UTF-8 entries count toward the enumeration ceiling but are not candidates, while an explicit non-UTF-8 filename fails. A directory input must be a non-symlink directory; every selected input/candidate file and final shard component must be a non-symlink regular file. Safe index shard names are single local filename components.

On Unix, source resolution establishes an **AnchoredDirectory** backed by an open directory descriptor. A directory input is opened with **O_DIRECTORY|O_NOFOLLOW**; a direct file's parent is opened as a directory descriptor. Enumeration reads from that descriptor, while selected candidates, indexes, and shards are opened with descriptor-relative **openat(O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_NOFOLLOW)** and checked through the returned file descriptor. The initially inspected device/inode identity of a direct file or directory input must equal the opened descriptor's identity. This rejects replacement during resolution, keeps subsequent member opens attached to the directory that was inspected, and sets **verification.directory_descriptors_anchored** to true.

**Non-Unix caveat:** **AnchoredDirectory** stores a path instead. The fallback rejoins member names, checks final-component symlink metadata, opens the path, and verifies the resulting descriptor is regular, but it does not have Unix's descriptor-relative anchoring against directory rename/replacement races or its device/inode identity checks. Reports make that weaker platform boundary explicit with **directory_descriptors_anchored: false**. Non-Unix before/after snapshots compare length and **modified()** instead of Unix modification/change timestamps.

For each shard, the inspector reads the eight-byte header-length prefix and bounded JSON header, never the tensor payload. Bounded serde string visitors enforce limits before retaining or cloning tensor/header names, dtype identifiers, safe shard names, and metadata keys; header metadata values are reduced to bounded byte lengths rather than retained. Duplicate keys, unknown tensor-descriptor fields, malformed metadata, unsafe identifiers, arithmetic overflow, out-of-range offsets, overlap, holes, uncovered payload bytes, and consistency changes fail the workflow. Header **__metadata__** is restricted to 4,096 string-to-string entries, 8 MiB per key/value, and 8 MiB of cumulative key/value bytes per header. Index **metadata** is duplicate-free, limited to 64 entries, and limits each key to 1,024 UTF-8 bytes.

The inspector rewinds and rereads the complete bounded index from the same descriptor, requiring byte-for-byte equality and the same end-of-file. It also rewinds and rereads each shard's prefix/header and compares those bytes. Initial and final lengths must agree; on Unix, device, inode, and both modification/change timestamps—including nanoseconds—must also agree. These checks detect same-length metadata changes and ordinary same-length file mutations while preserving the core boundary: no tensor payload byte is read or checksummed.

Dtype admission is also fail-closed. The parser recognizes only its current safetensors identifier allowlist and assigns each identifier an exact 4-, 6-, 8-, 16-, 32-, or 64-bit width. Checked **elements × bits** must be divisible by eight and equal the offset length, so sub-byte tensors receive the same shape/payload proof rather than offset-only evidence. Unknown identifiers and non-byte-aligned sub-byte totals fail. Successful v1 reports therefore mark every dtype verified, set the verified tensor count equal to the total, and keep the reserved unverified tensor count at zero.

A sharded index is read through its own bound. Its weight map must match shard headers exactly in both directions and map every tensor to the opened shard; declared **total_size** must equal the summed serialized tensor bytes. Only aggregate byte/count/dtype and verification evidence survives report construction. Paths, shard and tensor identities, and user metadata remain transient, and payload checksums are explicitly false because payload contents are never read.

Artifact storage evidence is deliberately separate from runtime placement. A checkpoint shard is not a TP, PP, DP, or EP rank, and serialized tensor bytes do not establish resident GPU/CPU bytes, expert composition, dequantization buffers, activations, allocator fragmentation, or workspaces. Nothing is automatically transferred between workflows. Only an explicit capacity **--artifact PATH** triggers a fresh bounded inspection and offers verified serialized tensor bytes as a base-weight floor; no Artifact Report field is converted into a capacity placement bound, and saved report JSON is not accepted as the artifact path.

## Capacity-planning boundary

The application resolves compatibility syntax into an explicit placement before calling **planning/**. Without topology flags, legacy **--gpus N** becomes TP=N; once any of **--tp**, **--pp**, **--dp**, or **--ep** is present, **--gpus** is an optional assertion that N equals **TP × PP × DP**. EP overlays the TP×DP ranks in each pipeline stage and does not multiply world size.

The model-config loader first performs bounded local parsing into opaque, debug-redacted evidence. The application then supplies all explicit parameter/layer/KV-head/head-dimension overrides together, and dependent geometry validation and dense parameter estimation run only after that merge. The serialized input records a bounded model-type identifier plus **explicit_override**, **config_declared**, or **dense_estimate** provenance; capacity v3 rejects unknown provenance, and the local config path and raw config object are never retained.

MoE detection examines expert-count aliases, known MoE model types, and nested sparse-routing markers. Detected MoE evidence requires an explicit total-resident parameter override and expert count/weight placement metadata; config parameter fields are not trusted because they can describe only active weights. Dense auto-estimation is limited to standard gated, bias-free layouts with exact **model_type** values **llama** or **mistral**. Other dense families need an explicit or non-MoE config-declared total, while sparse evidence always follows the stricter MoE path.

Capacity v3 can strengthen its logical base-weight estimate with **--artifact PATH**. The application invokes the bounded Artifact Report v1 inspector and the planner revalidates the report version, safetensors format/layout consistency, the supported dtype bit widths and aggregate element-to-byte geometry, aggregate count/container totals, complete shape-to-payload coverage, required file/header/offset/index evidence, and payload-read/checksum flags. **--artifact-residency-multiplier** requires that path, is finite in **[1, 1000]**, and defaults to 1 only when the path is present. With **P = parameters_billion × 1e9 × weight_bits / 8** and **A = ceil(serialized_tensor_bytes × multiplier)**, the base is **B = max(P, A)**; without an artifact it is **B = P**. The represented multiplier is decomposed into its binary significand and exponent so checked integer multiplication and ceiling division—not rounded floating multiplication—produce the upward whole-byte floor. Raw and adjusted artifact values are each capped at **8,000,000,000,000,000 bytes**, and weight overhead is applied once after selecting **B**. Equality retains the parameter basis. This maximum makes artifact evidence monotonic: it can never reduce memory or improve fit.

The retained artifact projection contains version, format/layout, shard/tensor/element counts, serialized bytes, descriptor-anchoring evidence, multiplier, adjusted logical floor, and selection status. It retains no path, shard/tensor identity, dtype breakdown, arbitrary metadata, or payload. Descriptor anchoring proves only the local file-resolution boundary; neither it nor any artifact aggregate proves loaded residency, runtime format expansion, expert composition, or placement.

The remaining arithmetic is deliberately based on the heaviest modeled rank rather than a world-size average. TP, PP, and EP declare topology but do not prove byte, layer, or KV-head balance. Shared and expert rank fractions therefore default independently to 100%; a PP stage defaults independently to 100% of shared, expert, and overhead bytes plus every layer; and a TP rank defaults to all KV heads. **--max-shared-rank-weight-percent**, **--max-expert-rank-weight-percent**, **--max-stage-component-weight-percent**, **--max-stage-layers**, and **--max-kv-heads-per-rank** can tighten only the placement dimension they name and require independent deployment/runtime evidence. Artifact shard structure and metadata never back these bounds. Weight overhead receives no TP/EP rank-sharding credit, and DP never divides per-rank memory or per-replica concurrency.

KV evidence preserves four different questions rather than overloading one total: logical bytes per sequence, a physical-per-DP-replica upper bound after possible head replication, the upper bound for one concurrency slot across every DP replica, and bytes on the worst rank. The topology retains the complete-head per-rank upper bound and a floating replication upper bound, so uneven placement does not require or imply exact KV-head divisibility.

This planner boundary is intentionally pure and hardware-independent. It validates topology, expert divisibility, conservative bound ranges, optional artifact evidence, and finite arithmetic, emits assumptions with stable codes, and returns a worst-rank estimate. Artifact input never replaces parameter/config geometry: detected MoE still requires an explicit total-resident parameter count plus expert count and expert-weight percentage. The planner does not model loaded allocation, model activation memory, CUDA graphs, speculative/draft models, allocator behavior, kernels, multimodal encoders, or communication workspaces; those remain explicit calibration caveats rather than hidden precision.

## Failure and concurrency model

The required NVIDIA inventory query decides whether a cycle succeeds. Optional telemetry preserves any usable value and emits source evidence on success, partial parsing, permission failure, unsupported capability, or deliberate skipping; it never erases valid inventory. A legitimate zero-record result, such as no active GPU processes, remains **ok** rather than being guessed unavailable. Sources that expect per-GPU rows or a topology matrix become **partial** when those records are missing. Endpoint failure becomes a report finding rather than a failed GPU collection.

The cycle has two concurrency layers:

1. all inference endpoints run behind a global concurrency cap while GPU collection proceeds;
2. process accounting, optional GPU fields, topology, and kernel-log collection run together after inventory, with compute and graphics process queries parallelized again.

No task holds a report lock during I/O. Child stdout/stderr are drained concurrently, requests and bodies are bounded, and the report is finalized only after all available evidence has been joined.

The active canary has its own bounded worker pool. Attempts occupy stable indexes even when requests finish out of order. Streaming TTFT begins at the first non-empty generated-content delta, not response headers or role-only events; token throughput requires authoritative server usage. The built-in prompt uses workload identity **builtin-v1**; custom prompt content requires an explicit non-secret identity and is never stored or hashed. Prompt text, generated content, credentials, and query values never enter its result contract.

Canary and saturation traffic share one generic closed-loop scheduler and the same OpenAI request/parser boundary. For each batch, a bounded fixed worker set and its result buffers are created before the measured interval, waits on a simultaneous start gate, receives an even deterministic index range, retains results thread-locally, and merges once in index order. Reported duration ends at the last invocation rather than including thread joins or result merging. The saturation workflow builds one HTTP client shared by every exact-stage warmup and measured stage, preventing client reconstruction from becoming a stage artifact; exact-stage warmup exercises the reused pool outside measurement to reduce, but cannot prove the elimination of, cold connection setup. Stages are sequential and ascending; every warmup and measured stage completes before the severe error/no-success guard decides whether higher load may begin. This scheduler deliberately does not implement open-loop arrival times or distributed coordination.

Runtime fingerprint collection is sequential over at most 32 explicit PIDs. Per-process collection is descriptor-anchored and bookended, but **maps** remains a non-atomic procfs view. Collection failures stay in typed source evidence so an incomplete report can be emitted before the CLI applies exit-2 policy; only invalid target selection or output failure prevents that evidence from being printed.

The rollout comparator is fully offline. It performs bounded regular-file reads and
bounded canary decoding, selects the final non-empty NDJSON record when needed, and
rejects non-current or unequal canary versions. It validates ordered raw attempts,
recomputes summaries and production nearest-rank distributions, and rebuilds
canonical gates/status before proving workload/model/route/stream/count/concurrency/
max-token/timeout/response-limit/policy compatibility. Endpoint URL may differ. It
also requires passing, time-ordered canaries. Selected tail-latency and throughput
gates need 20 complete finite samples per report and a non-zero baseline;
unavailable evidence is a regression rather than an implicit pass.

## Extension rules

- Add hardware/vendor integrations under **telemetry/** and return domain records instead of vendor structs.
- Add serving-system integrations under **inference/** and normalize only gauge-like operational symptoms; preserve bounded raw families for diagnosis.
- Add deterministic, side-effect-free findings under **analysis/** with a recommendation and tests for healthy and unhealthy cases.
- Add offline workflows under **operations/** and memory estimators under **planning/**.
- Never make a presentation adapter responsible for collection or policy.

## Security boundaries

- Collection is read-only and invokes only explicit utilities.
- Profile loading is opt-in, bounded to 1 MiB, rejects unknown fields, omits TOML source excerpts from errors, and stores only secret-file references. On Unix it also rejects a symlink as the selected file, owners other than the invoking user or root, group/world-writable files, and untrusted writable ancestors except sticky shared directories. Referenced credential files receive their own opened-inode, owner, mode, size, value, component, and canonical-ancestor checks while still permitting trusted projected-secret symlinks.
- Driver commands have null stdin and bounded execution.
- Runtime fingerprinting executes no child command and reads no **environ**, **fd**, **root**, **mem**, **map_files**, cgroup, or process-table discovery source. It retains only fixed enums, booleans, bounded counts, PIDs, strictly numeric version components, OS/architecture class, and a timestamp; paths, hostnames, environment values, model identities, argv, map lines, and arbitrary diagnostics are excluded by the report types.
- Artifact inspection is local, metadata-only, and bounded; it rejects oversize strings during decoding, final symlinks, unsafe index shard names, and metadata/header/index mutation, retains no paths/tensor names, and never reads or checksums tensor payload contents. Unix member opens are descriptor-relative and no-follow; the report explicitly exposes the non-Unix path-based fallback through **directory_descriptors_anchored**.
- Endpoint bodies and sample counts are capped.
- Active canary/benchmark and passive-probe URLs containing user information are rejected, and serialized targets retain only safe origins with complete queries redacted.
- Probe/API secrets are read from arguments, environment, or the corresponding **--probe-token-file** and **--api-token-file** inputs, but never serialized.
- **canary** is an active request that can consume serving capacity and billable tokens. Its count, concurrency, request size, response size, per-request timeout, and planned request-wave duration are bounded; response retention defaults to 1 MiB per attempt, and a conservative 768 MiB aggregate working-set admission budget covers request clones, response/parser amplification, matcher tables, and retained attempts. Use synthetic prompts and file-backed credentials. Remote HTTP is rejected unless **--allow-insecure-http** explicitly opts into cleartext; prefer HTTPS. Loopback canary requests disable ambient HTTP proxies so local prompts and credentials cannot be forwarded accidentally.
- **benchmark saturation** is a larger active request and can consume serving capacity and billable tokens. Explicit stage count/concurrency, per-worker samples, total attempts/tokens/prompt/response budgets, timeout waves, and a conservative 768 MiB working set are admitted before network access. Exact-stage excluded warmups and severe error guards bound escalation and measurement cold-start bias, but operators remain responsible for using synthetic traffic and an isolated test window. The machine contract includes fixed limitations so closed-loop results cannot silently become a distributed-capacity or production-SLA claim.
- Passive runtime probes require HTTPS for non-loopback targets unless **--allow-insecure-http** explicitly opts in, refuse redirects, and disable ambient proxies for every target. This prevents a metrics bearer token from being forwarded outside the configured origin.
- History and bundles use mode 0600 on Unix.
- The HTTP service supports bearer authentication but not TLS. A non-loopback bind requires both explicit **--allow-remote-listen** consent and an API token; put it behind an authenticated TLS proxy when crossing a trust boundary.
