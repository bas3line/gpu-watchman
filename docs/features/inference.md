# Inference canaries, saturation benchmarks, runtimes, artifacts, capacity, and trends

## Active OpenAI-compatible canary

**gpu-watchman canary** verifies the request path that users depend on: it sends bounded chat-completion requests to an OpenAI-compatible **/v1** API, checks generated content, measures time to first token/content (TTFT) and end-to-end latency, and can fail a rollout on selected SLOs.

~~~sh
gpu-watchman canary \
  --base-url http://127.0.0.1:8000/v1 \
  --model served-model \
  --max-ttft 2s --max-e2e 10s
~~~

The base URL defaults to **http://127.0.0.1:8000/v1**. The required **--model** value, or **GPU_WATCHMAN_INFERENCE_MODEL**, is passed through as the OpenAI model identifier. By default the canary sends one streamed request with a 30-second timeout and 16 maximum completion tokens. Its privacy-safe built-in prompt is **Reply with exactly: gpu-watchman-ok**, the result must contain **gpu-watchman-ok**, and the report uses workload identity **builtin-v1**. HTTPS is required for remote endpoints; loopback HTTP is allowed for local development, while **--allow-insecure-http** is an explicit cleartext opt-in for a trusted remote test environment.

Custom synthetic requests can come from the command line or a file:

~~~sh
gpu-watchman canary \
  --base-url https://candidate.example/v1 \
  --model served-model \
  --api-key-file /run/secrets/inference-api-key \
  --prompt-file ./synthetic-canary-prompt.txt \
  --workload-id synthetic-ready-v1 \
  --expect READY --max-tokens 32
~~~

**--api-key** and **--api-key-file** are mutually exclusive, as are **--prompt** and **--prompt-file**. A custom CLI prompt requires an explicit non-secret **--workload-id**; a profile prompt file must be paired with **canary.workload_id**. The identity is a label for operator-controlled synthetic content, never a copy or hash of that content. A custom prompt with no **--expect** needs non-empty output but does not inherit the built-in **gpu-watchman-ok** expectation. An explicit expectation is checked for both streamed and non-streamed responses.

### SLO gates and concurrency

~~~sh
gpu-watchman canary \
  --base-url https://candidate.example/v1 --model served-model \
  --api-key-file /run/secrets/inference-api-key \
  --count 10 --concurrency 2 \
  --max-ttft 2s --max-e2e 10s \
  --min-output-tokens-per-second 20 \
  --min-success-percent 100 --format json
~~~

**--count** controls total samples and **--concurrency** limits simultaneous work; both default to one, and concurrency remains implementation-bounded. Streaming is on by default and can be disabled with **--no-stream**. **--max-ttft** and **--max-e2e** enforce the worst successful-request measurement. Request, HTTP, timeout, bounded-body, stream/protocol, empty-output, and expectation failures count against the aggregate **--min-success-percent**, which defaults to 100; at least one request must always succeed even when the percentage threshold is zero. **--max-ttft** and **--min-output-tokens-per-second** require streaming timing, so the CLI rejects either option when combined with **--no-stream** as an exit-1 usage error.

Output-token throughput is calculated as **(completion tokens - 1) / (end-to-end time - TTFT)** and only from authoritative OpenAI-compatible completion-token usage returned by the server. It therefore requires a streamed response and at least two completion tokens. Non-positive completion usage or a count above the request's configured **max_tokens** is discarded as implausible. Throughput is never estimated from response bytes, whitespace, or a local tokenizer. A selected **--min-output-tokens-per-second** gate fails closed when plausible usage, streaming timing, or enough tokens are absent, because an unavailable measurement is not evidence that the SLO passed.

Exit 0 means the aggregate content/success policy and every selected SLO passed. Individual failed attempts may be tolerated when **--min-success-percent** explicitly allows them. Exit 1 means invalid usage or a local setup/file/output failure prevented the workflow from running. Exit 2 means the overall canary policy failed; text or machine-readable evidence is still emitted for CI diagnosis.

The canary caps accepted response bodies, request concurrency, per-request completion tokens, and the total requested completion-token budget. Response retention defaults to 1 MiB per attempt. Before opening a connection, admission control rejects plans whose conservative working-set estimate exceeds 768 MiB. The estimate is **persistent inputs and retained attempt slots + concurrency × worker budget**. A worker budget includes an encoded request clone with capacity growth, 64 times the response limit for worst-case JSON `Value`/collection amplification, capacity for one raw JSON body or a streaming line plus assembled event, the expectation needle plus its `usize` prefix table, and 1 MiB of fixed transport/parser allowance. This is a conservative admission estimate rather than an operating-system memory quota. A request timeout cannot exceed 5 minutes, and **ceil(count / concurrency) × timeout** cannot exceed 15 minutes, so a mistaken sample plan cannot occupy the CLI for hours. Each response must represent one completion choice, and any supplied choice index must be zero; alternatives cannot be concatenated into a false expectation match. It is a correctness and lightweight SLO smoke test; use the bounded saturation workflow below when multiple explicit concurrency points are required.

API keys are excluded from output; prefer **--api-key-file** because direct arguments may be visible in the local process list and shell history. Prefer **--prompt-file** for custom input for the same reason, and keep direct **--expect** markers non-sensitive because no file-backed expectation option exists. The default prompt contains no production data. Custom prompts are transmitted to the configured endpoint and may enter server logs, so use synthetic content and never customer inputs. Base URLs containing user information are rejected, serialized targets retain only the safe origin, and redirects are not followed. Remote cleartext HTTP additionally requires **--allow-insecure-http** because it can expose prompts and bearer credentials on the network; prefer HTTPS and configure the final chat-completions base URL directly. Loopback requests explicitly bypass ambient **HTTP_PROXY** and **ALL_PROXY** settings instead of forwarding local credentials through a proxy.

## Bounded saturation benchmark

**gpu-watchman benchmark saturation** measures only an explicit closed-loop concurrency ladder and can enforce deployment gates at one exact tested point:

~~~sh
gpu-watchman benchmark saturation \
  --base-url https://candidate.example/v1 --model served-model \
  --api-key-file /run/secrets/inference-api-key \
  --concurrency-stages 1,2,4,8 \
  --warmup-requests-per-worker 2 --requests-per-worker 20 \
  --verify-concurrency 8 --max-error-percent 1 \
  --max-p95-ttft 2s --max-p95-e2e 10s \
  --min-completion-token-goodput-per-second 200 --format json
~~~

Every schedule starts at one, is unique and strictly increasing, and contains at most eight points up to concurrency 64. Every point receives an excluded warmup at its exact concurrency, exercising the reused client and pool to reduce cold connection setup without proving that every measured transport path was pre-opened. A measured stage creates and preallocates its fixed worker set before timing, releases it through a simultaneous barrier, and assigns exactly **requests-per-worker** attempts to every worker through one reused HTTP client. Workers immediately replace only their own completed requests, so this is closed-loop fixed concurrency—not a target arrival rate. Machine evidence separates attempted RPS from successful-request goodput; only successful requests contribute authoritative completion-token goodput.

The optional p95 gates require at least 20 finite successful samples. The completion-token goodput gate requires every successful request to carry plausible authoritative endpoint usage. Non-positive token counts, prompt counts above 10,000,000 per request, and completion counts above configured **max_tokens** are omitted from samples and totals, making the affected usage evidence incomplete. Missing or implausible evidence becomes **not_evaluable**. **--verify-concurrency** must match one listed point and turns that point's gates into exit policy: pass is exit 0, while fail or not-evaluable is exit 2 after the report is emitted. Independently, zero successes or the severe error rate in an exact-stage warmup or measured phase stops higher load and exits 2 with the partial bounded evidence; a warmup abort prevents that point's measurement. The descriptive signal normalizes adjacent successful-RPS growth by concurrency growth: **(RPS gain % / concurrency gain %) × 100** must be at most 5%, together with at least 20% p95 latency inflation or excessive errors. It does not fail the run by itself; a latency-inflation comparison requires at least 20 finite successful samples in both adjacent stages.

The plan is rejected before network access when it exceeds any fixed ceiling: 10,000 total warmup-plus-measured attempts, 1,000,000 requested completion tokens, 64 MiB of aggregate prompt bytes, 2 GiB of aggregate response-limit bytes, 30 minutes of worst-case request waves, 8 MiB per response, or the conservative 768 MiB working-set admission. The benchmark defaults to a 10-second timeout and 128 KiB response cap so its advertised ladder fits the time budget and useful concurrency can fit the parser-aware memory budget. It reuses canary profile target/workload/transport settings but never inherits its count, concurrency, maximum tokens, timeout, or SLO thresholds other than the expectation paired with unchanged profile prompt content.

Saturation Benchmark v1 deliberately makes narrow claims. Closed-loop scheduling has coordinated omission; this single generator, client, or network can bottleneck before the endpoint; repeated synthetic prompts may benefit from prefix caching; and unrelated traffic is neither isolated nor attributed. Only listed points were tested. Concurrency is not server batch size, GPU occupancy, production RPS, maximum capacity, or an SLA. Endpoint-reported token goodput is not raw GPU decode capacity. The report makes no distributed, open-loop, adaptive-breakpoint, soak, cost, production-p99, or deployment-recommendation claim. See the [complete CLI and exit contract](../reference/cli.md#benchmark-saturation) and [machine schema](../reference/report-schema.md#saturation-benchmark-v1).

## Local runtime fingerprint

**gpu-watchman runtime inspect** produces bounded Runtime Fingerprint v1 evidence for one to 32 explicit local process IDs:

~~~sh
gpu-watchman runtime inspect --pid 4242 --pid 4243 --format json
gpu-watchman runtime inspect --pid 4242 --allow-incomplete --format ndjson
~~~

Collection is Linux-only and never scans for candidate processes. Each target must be a positive PID, duplicate PIDs are rejected, and more than 32 targets is an exit-1 usage error. On a non-Linux host, the command still emits the versioned report with **unsupported_platform** process-source reasons, but the report is incomplete.

On Linux, the collector pins **/proc/PID** as one no-follow directory descriptor and opens only **stat**, **cmdline**, and **maps** relative to it. It compares the process start time from a **stat** read before and after collection to detect exit or PID reuse. It reads **cmdline** twice and retains argv-derived deductions only when both bounded byte sequences match exactly. **maps** is read once and remains explicitly non-atomic; this limitation is a fixed assessment reason rather than a hidden guarantee.

The hard process bounds are:

| Input | Limit |
| --- | --- |
| Explicit PIDs | 32 |
| One **stat** read | 8 KiB |
| One **cmdline** read | 256 KiB |
| Command-line arguments / one argument | 4,096 / 16 KiB |
| One **maps** read | 8 MiB |
| Map records / one map record | 65,536 / 16 KiB |
| Distinct recognized mapped-library facts per process | 64 |
| One fixed driver-file read | 4 KiB |
| One retained numeric version token | 64 bytes |

No child process is created: the workflow does not execute **nvidia-smi**, an engine, Python, a package manager, **ldconfig**, or any version command. NVIDIA driver evidence is limited to no-follow reads of **/sys/module/nvidia/version**, with **/proc/driver/nvidia/version** as corroboration or fallback. It is kernel-module evidence only, not a CUDA user-mode driver, toolkit, or runtime version. Driver absence or unavailability does not make otherwise stable process evidence incomplete.

Engine candidates are fixed to **vllm**, **tgi**, **triton**, **sglang**, and **tensorrt_llm**. Framework candidates are fixed to **pytorch**, **tensorflow**, **onnx_runtime**, and **tensorrt** and are derived only from recognized mapped-library names. The mapped-library allowlist has fixed families for the CUDA driver/runtime, cuBLAS, cuDNN, NCCL, TensorRT, Torch core/CUDA, TensorFlow, and ONNX Runtime CUDA. Filename-derived versions contain at most four strictly numeric components and are labeled **mapped_filename_only**; they are not package-version evidence.

When exactly one engine identity is recognized, supported launch flags are reduced to typed observations for tensor, pipeline, and data parallel sizes; context-token limit; canonical dtype, quantization, and KV-cache dtype; and the boolean **model_reference_present**. The model reference itself is discarded. Duplicate declarations become **ambiguous**, a present but unsupported/unparseable value becomes **present_unparsed**, and an absent declaration is **not_observed**. These fields describe argv declarations, not defaults or effective initialized state.

The report schema cannot carry filesystem paths, hostname, environment values, model identity, raw argv, map records, package labels, or arbitrary operating-system errors. It retains only fixed enums and evidence kinds, PIDs, bounded counts, OS/architecture class, booleans, strictly numeric versions, and the collection timestamp. Fixed source reason codes are **permission_denied**, **not_found**, **process_exited**, **pid_reused**, **changed_during_collection**, **limit_exceeded**, **malformed**, and **unsupported_platform**.

Compatibility is always **not_evaluated**. Runtime Fingerprint v1 does not assert compatible or incompatible because a kernel-module version is not a user-mode driver version, mapped names are not package versions, argv is not effective configuration, maps is non-atomic, and no model artifact is bound to the process. A complete report requires Linux, stable identity, and observed identity, command-line, and memory-map sources for every target. The report is printed before exit policy: complete evidence exits 0; incomplete evidence exits 2 by default; **--allow-incomplete** changes only that incomplete exit to 0. Fatal PID selection, validation, or serialization/encoding failures exit 1.

## Runtime probes

**--probe** accepts comma-separated HTTP or HTTPS base URLs and full metrics URLs. A base URL resolves to **/metrics**. Probes run concurrently with GPU collection. Loopback HTTP is allowed, while a non-loopback HTTP target requires **--allow-insecure-http** (or **monitor.inference.allow_insecure_http = true**) to acknowledge cleartext transport; HTTPS is the production default.

Supported runtime metric prefixes:

| Runtime | Recognized prefixes |
| --- | --- |
| vLLM | vllm: and vllm_ |
| Hugging Face TGI | tgi: and tgi_ |
| NVIDIA Triton | triton_, nv_inference_, and nv_gpu_ |
| TensorRT-LLM | tensorrt_llm_ and trtllm_ |
| SGLang | sglang: and sglang_ |
| Ollama | ollama: and ollama_ |

GPU Watchman processes recognized metric families into normalized signals:

- currently running requests;
- queued/waiting requests;
- maximum KV-cache utilization;
- monotonic request, error, prompt-token, generation-token, and preemption counters;
- interval request/error/preemption rates and prompt/generation token throughput;
- interval means for request latency, time to first token (TTFT), and time per output token (TPOT) when histogram sum/count families are available;
- interval p50/p95/p99 request latency, TTFT, TPOT, and queue time from supported classic Prometheus bucket families.

Counters are not treated as gauges merely because they share a runtime prefix. The first sample records a baseline; the second and later watch samples derive rates from monotonic deltas and elapsed wall time. A counter reset or missing family produces no rate instead of a negative or fabricated value. Non-finite values are ignored.

Histogram retention is deliberately fixed and bounded: only recognized latency bucket families enter four normalized fields, with at most 64 buckets and 32 compatible series per kind. Each series must have finite non-negative cumulative counts, strictly increasing finite boundaries, and a final **+Inf** bucket. Canonical series identities—including the fixed runtime/family/schema and non-**le** labels—remain only in bounded live memory until the next interval is derived; their debug view is redacted and Serde never writes them to reports. Serialized snapshots contain only aggregate fixed-schema buckets and a numeric series count. Quantiles require two complete scrapes with the exact same private series identities and boundaries, then difference every series before aggregation. A reset, alias or series replacement, boundary change, incomplete scan, or invalid bucket omits the interval estimate instead of fabricating one. Quantiles use the classic Prometheus linear-within-bucket estimate and conservatively cap an open-ended result at the highest finite boundary.

The default probe bounds are 4 MiB per response, 10,000 processed samples, four concurrent requests, 32 targets, 32 MiB of aggregate in-flight response capacity, 128 KiB of processed metric-family names, and a three-second timeout. Individual family names are limited to 256 ASCII Prometheus-name bytes. Hard ceilings are 8 MiB per response and eight concurrent requests; invalid combinations fail before network access.

~~~sh
gpu-watchman top --watch 5s \
  --probe https://vllm-a.example/metrics,https://triton.example/metrics \
  --probe-token-file /run/secrets/runtime-metrics-token
~~~

**snapshot**, **top**/**watch**, and **serve**/**exporter** accept the full passive-probe group. **doctor** and **bundle** also accept **--probe** plus direct or file-backed probe credentials and the cleartext opt-in. Use **--probe-token TOKEN** or **GPU_WATCHMAN_PROBE_TOKEN** for a direct value; use **--probe-token-file PATH** or **GPU_WATCHMAN_PROBE_TOKEN_FILE** for a file-backed secret. Direct and file-backed inputs at the same precedence level are mutually exclusive, CLI inputs win over environment inputs, and tokens are never serialized.

Passive probes reject URL user information, do not follow redirects, and disable ambient proxy configuration for all targets. A configured bearer token is admitted only when every probe URL resolves to one exact origin (the same scheme, host, and port); a multi-origin authenticated set is rejected before network access so a credential cannot cross service boundaries. Complete queries are redacted in text, JSON, history, bundles, and metrics labels. Endpoint-controlled Prometheus family names, label names, and label values are never retained in reports; only fixed-schema normalized counters/gauges and bounded sample counts are serialized, preventing a compromised endpoint from reflecting secrets or tenant data into operational artifacts. Configure the final metrics URL directly rather than relying on a redirect.

## Runtime findings

| Code | Severity | Condition |
| --- | --- | --- |
| inference-endpoint-down | warning | Request, HTTP, size, or read failure |
| inference-metrics-unrecognized | info | Reachable response has no supported runtime families |
| inference-metrics-slow | warning | Scrape reaches the configured latency threshold |
| inference-queue | warning | One or more requests wait |
| inference-stalled | critical | Requests wait while none are running |
| inference-errors | warning | Request error counter increased during the interval |
| inference-preemptions | warning | Runtime preemption counter increased during the interval |
| kv-cache-high | warning | KV cache reaches the warning threshold |
| kv-cache-critical | critical | KV cache reaches the critical threshold |

The normalization is intentionally symptom-oriented. Runtime-specific counters remain in JSON for deeper analysis. A one-shot snapshot cannot calculate a rate because it has no prior sample; use **top**/**watch**, **serve**/**exporter**, or history collection for interval signals.

## VRAM growth

Continuous mode tracks each GPU UUID and PID in memory. If a process gains at least the configured **--process-growth-mib** over samples separated by one second, it emits **vram-growth**. A missing process is immediately removed from the tracker.

This is an investigation signal, not proof of a leak: model loading, CUDA graph capture, prefix caching, and legitimate cache growth can all increase VRAM.

## Safetensors artifact inspection

**gpu-watchman artifact inspect** validates checkpoint-storage metadata without loading tensor payloads:

~~~sh
# One safetensors container
gpu-watchman artifact inspect ./model.safetensors

# A Hugging Face sharded index, or its unambiguous containing directory
gpu-watchman artifact inspect ./model.safetensors.index.json --format json
gpu-watchman artifact inspect ./checkpoint-directory --format ndjson
~~~

The input can be one **.safetensors** file, one **.safetensors.index.json** file, or a directory that resolves unambiguously. A directory with one index selects that index; without an index it must contain exactly one safetensors file. No file, multiple standalone safetensors files, or multiple indexes fail closed. Directory discovery examines raw filenames: non-UTF-8 entries consume the 4,096-entry budget but are skipped as candidates, while an explicitly supplied non-UTF-8 filename fails validation. Final-component symlinks are rejected for the input, discovered candidates, and every opened shard; shards must be regular files.

On Unix, the inspector pins the selected input directory or direct file's parent as an open directory descriptor. Directory enumeration uses that descriptor, and each candidate, index, and shard is opened relative to it with **openat**, **O_NOFOLLOW**, **O_NONBLOCK**, and **O_CLOEXEC** before the opened descriptor is checked as a regular file. A directory supplied directly is itself opened with **O_DIRECTORY** and **O_NOFOLLOW**. For a direct file or directory input, the initially inspected device/inode identity must match the opened descriptor, so replacement during resolution fails. Successful reports record **directory_descriptors_anchored: true**.

**Non-Unix caveat:** the fallback stores the directory path, rejoins each selected filename, rejects a symlink through a metadata check, opens the path, and checks the resulting file is regular. It cannot provide the same descriptor-relative directory anchoring, device/inode identity checks, or rename-race resistance, so successful reports record **directory_descriptors_anchored: false**. Its before/after file snapshot compares length and the platform **modified()** value rather than Unix modification/change timestamps.

Artifact Report v1 has **artifact_version: 1**, **artifact_format: safetensors**, and **layout: single_file|sharded_index**. Its summary reports exact integers for tensor count/elements, serialized tensor payload bytes, complete safetensors-container file bytes, header-region bytes, and the eight-byte length prefixes. Indexed reports additionally record exact index-file bytes and the index's declared **total_size**. Dtype summaries aggregate tensor count, element count, and serialized bytes in deterministic dtype order.

Inspection enforces these hard ceilings during source resolution, bounded parsing, and aggregation:

| Input | Limit |
| --- | --- |
| Index JSON | 64 MiB |
| One safetensors header region | 100,000,000 bytes |
| Combined shard header regions | 256 MiB |
| Directory entries examined | 4,096 |
| Shards referenced by one index | 1,024 |
| Tensors | 500,000 |
| One tensor name / all tensor names | 1,024 bytes / 64 MiB |
| One shard filename | 255 bytes |
| Tensor rank | 32 dimensions |
| Distinct dtype groups | 64 aggregation guard; the current allowlist is smaller |
| One dtype identifier | 32 UTF-8 bytes, then exact allowlist validation |
| Header **__metadata__** entries | 4,096 per header |
| One header metadata key or value | 8 MiB UTF-8 bytes |
| Header metadata key + value bytes | 8 MiB cumulative per header |
| Index **metadata** entries | 64 |
| One index metadata key | 1,024 UTF-8 bytes |

Headers must be duplicate-free bounded JSON and use only ASCII-space trailing padding. Dedicated decode visitors enforce the per-string limits before GPU Watchman retains or clones header/tensor names, dtypes, shard names, and metadata keys. Header metadata values are validated as bounded strings but retained only as their byte lengths. Header **__metadata__** must be a duplicate-free string-to-string map; its 8 MiB cumulative budget is the UTF-8 byte length of keys and values, excluding JSON syntax. Index metadata is also duplicate-free, must contain integer **total_size**, and may contain other bounded entries that are ignored after validation.

Artifact Report v1 currently accepts only these exact safetensors dtype identifiers:

| Bits per element | Accepted identifiers |
| --- | --- |
| 4 | **F4** |
| 6 | **F6_E2M3**, **F6_E3M2** |
| 8 | **BOOL**, **U8**, **I8**, **F8_E5M2**, **F8_E4M3**, **F8_E8M0**, **F8_E4M3FNUZ**, **F8_E5M2FNUZ** |
| 16 | **U16**, **I16**, **F16**, **BF16** |
| 32 | **U32**, **I32**, **F32** |
| 64 | **U64**, **I64**, **F64**, **C64** |

Every tensor must pass checked **element count × dtype bits** arithmetic. The result must be divisible by eight—including for **F4** and the six-bit formats—and that exact byte count must equal the tensor's data-offset length. Unknown identifiers, bit-size overflow, non-byte-aligned sub-byte totals, and shape/offset mismatches fail the complete inspection. Tensor offsets must also stay within the shard payload, be non-overlapping and hole-free, and cover the complete payload.

Consequently, every dtype item in a successful Artifact Report v1 has **shape_payload_bytes_verified: true**, **verification.shape_payload_bytes_verified_tensors** equals **summary.tensor_count**, and **verification.shape_payload_bytes_unverified_tensors** is 0. The unverified counter is reserved in the schema; current v1 inspection never uses offset-only evidence to admit a tensor.

For a sharded index, every weight-map tensor must appear in exactly the named shard, no shard tensor may be absent from the index, no indexed tensor may be missing, and **metadata.total_size** must equal the summed serialized tensor bytes. Shard names are limited to one safe local filename component.

Consistency checks use the already-open descriptor. The bounded full index is rewound, reread byte-for-byte, compared with the retained first read, and checked for the same end-of-file. Each shard's eight-byte length prefix and bounded header are likewise rewound, reread, and compared; tensor payload bytes remain unread. Initial and final file lengths must match. On Unix, the same descriptor's device, inode, modification seconds/nanoseconds, and change seconds/nanoseconds must also match, detecting ordinary same-length mutations that a size-only check would miss. This is consistency evidence, not a payload checksum.

Tensor payload contents are never read, materialized, decoded, or checksummed. Paths, shard filenames, tensor names, and safetensors metadata values are used transiently for validation but never enter text, JSON, or NDJSON output. Dtype labels and aggregate counts/bytes are retained. Consequently, Artifact Report v1 proves serialized checkpoint layout only: shard containers do not prove runtime TP/PP/DP/EP placement, and serialized tensor bytes do not prove resident GPU/CPU memory, activation memory, allocator state, or runtime workspaces.

## Capacity planner

The capacity command is hardware-independent. Capacity Report v3 models an explicit deployment topology and reports the rank with the largest modeled memory placement. It can optionally inspect a safetensors artifact and use verified serialized tensor bytes as a stronger base-weight floor. Topology degrees and artifact shard structure never earn placement credit: any tighter placement must be supplied as an independently deployment/runtime-evidenced upper bound. A manual estimate can be written as:

~~~sh
gpu-watchman capacity \
  --params 70 --weight-bits 4 \
  --tp 2 --gpu-vram 80 --utilization 0.9 \
  --layers 80 --kv-heads 8 --head-dim 128 \
  --max-shared-rank-weight-percent 50 \
  --max-kv-heads-per-rank 4 \
  --context 32768 --concurrency 8 \
  --kv-bits 16 --runtime-overhead 6 \
  --weight-overhead-percent 8
~~~

Or it can derive parameters, layers, KV heads, and head dimension from a local Hugging Face **config.json**:

~~~sh
gpu-watchman capacity --model-config ./config.json \
  --tp 2 --gpu-vram 80 --utilization 0.9 \
  --weight-bits 4 --kv-bits 16 \
  --max-shared-rank-weight-percent 50 \
  --context 32768 --concurrency 8 \
  --runtime-overhead 6
~~~

Add **--artifact PATH** to inspect the exact checkpoint and strengthen the weight floor:

~~~sh
gpu-watchman capacity --model-config ./config.json \
  --artifact ./model.safetensors.index.json \
  --artifact-residency-multiplier 1.25 \
  --tp 2 --gpu-vram 80 --utilization 0.9 \
  --weight-bits 4 --kv-bits 16 \
  --max-shared-rank-weight-percent 50 \
  --context 32768 --concurrency 8 \
  --runtime-overhead 6
~~~

**--artifact PATH** accepts the same local **.safetensors** file, **.safetensors.index.json**, or unambiguous directory as **artifact inspect**. It performs a fresh bounded inspection; it does not accept a saved Artifact Report JSON file. **--artifact-residency-multiplier M** requires **--artifact**, must be finite and in **[1, 1000]**, and defaults to 1 only when an artifact is supplied. Supplying the multiplier without an artifact is an exit-1 usage error.

Artifact input strengthens only the base-weight calculation. Let **P** be parameter/precision bytes and **A** be the artifact-derived floor:

~~~text
P = parameters_billion × 1,000,000,000 × weight_bits / 8
A = ceil(serialized_tensor_bytes × artifact_residency_multiplier)
B = max(P, A) when an artifact is supplied; otherwise B = P
O = B × weight_overhead_percent / 100
~~~

The artifact product is rounded upward to a whole byte. The raw serialized total and multiplier-adjusted total must each be no greater than **8,000,000,000,000,000 bytes**; invalid, non-finite, overflowing, or larger evidence fails with exit 1 before a report is emitted. The artifact wins only when **A > P**; equality retains the **parameter_precision_estimate** basis. Because **B** is a maximum and the multiplier cannot be below 1, adding artifact evidence can only preserve or increase modeled memory and can only preserve or reduce the fit result. Weight overhead is applied once, after selecting the larger base.

The artifact is not a source of model geometry. **--params** remains required unless **--model-config** supplies or safely derives it, and layer/KV-head/head-dimension geometry still comes from explicit flags or that config. Detected MoE models still require an explicit total-resident **--params** plus expert-count and expert-weight-percentage metadata. Serialized shards and tensor metadata do not reveal shared/expert composition or justify TP, PP, DP, EP, stage, expert, layer, or KV-head placement bounds.

The topology controls describe ranks, not byte placement:

- **--tp** is the tensor-parallel degree. By default one worst TP rank is charged all shared base-weight bytes from the heaviest stage and all KV heads. TP does not imply even tensor or KV sharding.
- **--pp** is the pipeline-parallel degree. By default one stage is charged 100% of each weight class and every transformer layer. PP does not imply an even byte or layer split.
- **--dp** is the number of full data-parallel replicas. DP increases **TP × PP × DP** world size but never divides per-rank weights or KV cache. **--concurrency** means full-context sequences resident on every DP replica, not a global total divided by DP.
- **--ep** overlays expert placement within the **TP × DP** ranks of each pipeline stage; it does not multiply world size. Expert count must be divisible by EP, and EP must divide **TP × DP**, but those count relationships do not prove equal expert byte sizes.

Prefer these explicit topology flags for new automation. For command-line compatibility, **--gpus N** without any **--tp**, **--pp**, **--dp**, or **--ep** flag is interpreted as **--tp N --pp 1 --dp 1 --ep 1** and records the **legacy_gpus_interpreted_as_tp** assumption. Once any topology degree is explicit, **--gpus N** instead asserts that N exactly equals **TP × PP × DP**; a mismatch is rejected rather than averaged into a plausible-looking result.

Placement upper bounds are independent and fail closed:

- **--max-shared-rank-weight-percent** bounds shared bytes from the heaviest stage on the worst TP rank. It defaults to 100% and must be between **100 / TP** and 100.
- **--max-expert-rank-weight-percent** independently bounds routed-expert bytes on the worst EP rank. It defaults to 100%, must be between **100 / EP** and 100, and requires paired **--expert-count**/**--expert-weight-percent** metadata.
- **--max-stage-component-weight-percent** independently bounds each of shared, routed-expert, and overhead bytes on the heaviest PP stage. It defaults to 100% and must be between **100 / PP** and 100; it is not a bound on only their combined total.
- **--max-stage-layers** bounds layers on the heaviest PP stage for worst-rank KV cache. It defaults to all model layers and must be between **ceil(layers / PP)** and the full layer count.
- **--max-kv-heads-per-rank** bounds complete KV heads retained by the worst TP rank. It defaults to all KV heads and must be between **ceil(KV heads / TP)** and the full KV-head count. No exact divisibility is required or inferred.

Use these flags only when deployment/runtime evidence independently justifies the bound. Artifact shard structure, tensor metadata, and descriptor anchoring never do. Omitting one preserves its conservative 100%-or-all default independently of every other bound.

The model-config loader requires model type, layer count, attention-head count, and hidden size. It uses **num_key_value_heads** when present or conservatively assumes one KV head per attention head. It uses **head_dim** when present or derives hidden size divided by attention heads. Explicit **--params**, **--layers**, **--kv-heads**, and **--head-dim** values are merged before dependent validation and parameter estimation, so a geometry override changes any audited dense estimate that depends on it.

The loader treats expert-count aliases, known MoE model types, and nested sparse-routing markers as MoE evidence. A detected MoE config requires an explicit **--params** value representing total resident parameters; **num_parameters** or **parameter_count** from that config is rejected because it may describe active rather than resident weights. Expert count and expert-weight percentage placement metadata are also required. A recognized config expert count may supply the count, while a CLI **--expert-count** must match it; **--expert-weight-percent** remains explicit.

For a non-MoE config, **num_parameters** or **parameter_count** is accepted but not verified against checkpoint files. Without either field or **--params**, automatic dense estimation is audited only for exact **model_type: "llama"** and **model_type: "mistral"** configs using the standard gated, bias-free layout. Unknown model families, unsupported projection/bias layouts, or sparse markers fail closed instead of receiving an optimistic estimate. Machine output records **input.parameter_source** as **explicit_override**, **config_declared**, or **dense_estimate**, plus a bounded safe **input.model_type** or null; current capacity-v3 estimation rejects unknown provenance. Config input is local-only, regular-file checked, valid JSON, and limited to 8 MiB.

Let **B** be the selected logical base-weight bytes above, **O** the configured logical weight-overhead bytes, **E** the routed-expert fraction, **p** the stage-component upper-bound fraction, **s** the shared-rank upper-bound fraction, and **x** the expert-rank upper-bound fraction. The worst-rank placement estimate is:

~~~text
worst-rank weights = p × B × (1 - E) × s
                   + p × B × E × x
                   + p × O
~~~

All three placement fractions default to 1.0. The planner never substitutes **1 / TP**, **1 / EP**, or **1 / PP** automatically. Weight metadata/dequantization overhead is not rank-sharded, though the explicit stage-component bound applies to it independently. DP never appears in this per-rank formula. KV memory per sequence on the worst rank is:

~~~text
2 × maximum stage layers × maximum KV heads per rank
  × head dimension × context tokens × KV bits / 8
~~~

The topology exposes **kv_heads_per_tensor_parallel_rank_upper_bound** and the floating **kv_head_replication_upper_bound = max heads per rank × TP / logical KV heads**. Machine evidence distinguishes logical KV bytes per sequence, a physical-per-DP-replica upper bound after head replication, the upper bound for one concurrency slot resident across every DP replica, and worst-rank bytes. The planner multiplies worst-rank KV bytes by per-replica concurrency, adds per-rank runtime overhead, applies the usable-memory fraction to the smallest configured GPU, and calculates approximate full-context concurrency per DP replica. Physical **--gpu-vram** is always required.

JSON and NDJSON use a separate **capacity_version: 3** contract containing the effective **input**, optional path-free aggregate **artifact** evidence, validated **topology**, **weights**, **kv_cache**, **memory**, **fits**, stable coded **assumptions**, and **caveats**. **weights.base_weight_basis** is **parameter_precision_estimate** or **artifact_residency_floor**; the weight object also retains the parameter-derived estimate, nullable adjusted artifact floor, selected base, overhead, logical class totals, and worst-rank class totals. When no artifact is supplied, **artifact** and **weights.artifact_adjusted_weight_floor_gib_logical** are null and the parameter basis is selected. Logical weight totals use **\*_gib_logical** names so they cannot be mistaken for allocated or summed physical memory. It is not a schema-v3 hardware report.

The capacity artifact object contains only Artifact Report version/format/layout, shard/tensor/element counts, serialized tensor bytes, whether directory access was descriptor-anchored, multiplier, adjusted logical GiB floor, and whether that floor was selected. Descriptor anchoring is file-resolution evidence, not placement evidence. The object contains no input path, shard or tensor names, dtype breakdown, arbitrary metadata, or payload data. This privacy-safe aggregate still describes serialized storage, not loaded allocation: it cannot prove runtime precision conversion, dequantization, duplication, quantization workspaces, GPU/CPU residency, expert composition, or TP/PP/DP/EP placement. Choose the multiplier and overhead allowances to cover known loader expansion; neither is measured automatically.

The planner still does not model activation memory, CUDA graph captures, speculative/draft-model allocations, allocator fragmentation, kernels, multimodal encoders, or communication workspaces. **--runtime-overhead** and **--weight-overhead-percent** are explicit conservative allowances, not measurements of those allocations. Calibrate them from the exact runtime, checkpoint format, parallel layout, sequence mix, and known-good deployment before using the result for admission control.

Counts and weight/KV precisions must be positive, floating-point inputs must be finite, utilization must be in **(0, 1]**, and overheads must be non-negative. Invalid usage, config or artifact inspection failure, invalid artifact evidence, out-of-range multiplier, arithmetic/ceiling failure, or invalid geometry exits 1 without a capacity report. A valid request that fits exits 0. A valid request that does not fit exits 2 and still prints the complete estimate; strengthening the artifact multiplier can therefore change a fit from exit 0 to non-fit exit 2, never the reverse.

## Rollout verification

Capture a known-good node report and compare it with the candidate after warm-up:

~~~sh
gpu-watchman snapshot --format json \
  --probe http://runtime:8000 --allow-insecure-http > before.json
# deploy and warm the candidate
gpu-watchman canary --base-url https://runtime.example/v1 --model served-model \
  --count 5 --concurrency 1 --max-ttft 2s --max-e2e 10s
gpu-watchman snapshot --format json \
  --probe http://runtime:8000 --allow-insecure-http > after.json
gpu-watchman compare before.json after.json --fail-on-regression
~~~

The hardware report comparison is a passive health regression gate: it detects endpoint loss, new findings, device loss, and pressure deltas. Saved canaries can also be compared directly under the same request identity:

~~~sh
gpu-watchman canary --base-url https://baseline.example/v1 --model served-model \
  --count 20 --concurrency 2 --format json > baseline-canary.json
gpu-watchman canary --base-url https://candidate.example/v1 --model served-model \
  --count 20 --concurrency 2 --format json > candidate-canary.json
gpu-watchman rollout baseline-canary.json candidate-canary.json \
  --max-p95-ttft-regression-percent 10 \
  --max-p95-e2e-regression-percent 10 \
  --min-output-tps-ratio 0.9 \
  --max-success-drop-percent 1 --fail-on-regression
~~~

The active comparator requires the same current canary version and identical model,
workload ID, route, stream mode, count, concurrency, maximum tokens, timeout,
response limit, and recorded canary policy; endpoint URL may differ. It validates
ordered raw attempts and rebuilds token totals, nearest-rank distributions,
canonical gates, and status before comparing them. A selected latency or throughput
distribution needs at least 20 complete successful samples in each report; missing,
partial, non-finite, negative, inconsistent, or zero-baseline evidence fails closed.
See the [canary rollout guide](../operations/rollout.md). Neither canary workflow
replaces a sustained load or capacity benchmark.
