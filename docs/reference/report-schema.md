# Report schema

Every snapshot, watch record, history entry, report API response, and support bundle report uses the same Rust domain model. The current external contract is **schema_version 3**. Version 3 adds explicit telemetry-source evidence so an empty value is no longer the only indication that an optional collector failed or was skipped.

**gpu-watchman canary** is deliberately separate: active inference results use **canary_version: 2** with a non-secret workload identity, request distributions, gates, and privacy-safe attempt evidence. They are not hardware reports and are not valid input to **history** or **compare**; **rollout** is their dedicated offline comparator.

Canary v2 adds the top-level **workload_id** and privacy-safe **policy** evidence. The policy records minimum success percentage, optional maximum TTFT/end-to-end milliseconds, optional minimum output-token rate, and only a boolean saying whether an expectation was configured. The built-in prompt always emits **builtin-v1**; custom prompts require an explicit ID rather than storing or hashing prompt content. Version-1 canary JSON remains deserializable through a default empty workload ID and missing policy, but rollout always treats v1 as incompatible. The remaining envelope fields are **started_at**, **duration_ms**, **status**, redacted **target**, bounded **plan**, aggregate **summary**, canonical **gates**, and privacy-safe **attempts**. Prompt text, generated text, expectation text, credentials, and target query values never enter the contract.

Canary decoding caps attempt and gate sequences at 10,000 and five respectively and bounds every retained identity or diagnostic string. For rollout admission, attempt indexes and count must exactly match the plan; per-attempt metrics and failure evidence are validated; summary counts, token totals, and nearest-rank distributions are rebuilt from successful attempts; and canonical gates plus status are rebuilt from the recorded policy. Serialized summaries are evidence to verify, not trusted substitutes for the attempts.

**gpu-watchman rollout** emits its own stable **rollout_version: 1** object. It contains baseline/candidate timestamps, statuses, privacy-safe identities, explicit compatibility checks, selected quantitative gates, **compatible**, and **regression**. Each identity includes the exact current canary version, workload ID, model, route, stream mode, count, concurrency, maximum tokens, timeout, response-size limit, and policy. URL may differ across environments and is not an identity field. The rollout result never embeds the input reports or endpoint URLs.

**gpu-watchman benchmark saturation** emits a separate **saturation_benchmark_version: 1** object. It is active measured evidence, not a schema-v3 hardware report, canary result, rollout comparison, runtime fingerprint, artifact report, or capacity report. **gpu-watchman benchmark compare** is its dedicated offline consumer.

**gpu-watchman ps --format json|ndjson** is also a focused contract rather than a full hardware report. It uses **process_view_version: 1** and is not valid input to **history** or **compare**. The top-level fields are **process_view_version**, **collected_at**, **host**, **source**, **process_count**, and **processes**. **source** always identifies **nvidia.processes** and makes missing, partial, unavailable, or complete collection evidence explicit. Each deterministically ordered process record flattens physical GPU index/UUID/name together with PID, process name, VRAM MiB, owner, command, cgroup, container ID, and Kubernetes pod UID. Optional identity fields remain present as null when unavailable.

## Saturation Benchmark v1

**gpu-watchman benchmark saturation --format json|ndjson** emits:

| Field | Meaning |
| --- | --- |
| **saturation_benchmark_version** | Contract version; currently 1 |
| **started_at** / **duration_ns** / **duration_ms** | UTC start, canonical active-run nanoseconds, and floored display milliseconds |
| **status** | **complete** or **aborted** |
| **abort_reason** | Optional fixed warmup/stage no-success or error-limit reason |
| **workload_id** | Bounded non-secret identity for operator-controlled synthetic content |
| **target** | Safe origin, fixed **chat_completions** route, requested model, and stream mode |
| **plan** | Explicit concurrency stages, sample counts, total planned attempts, request limits, and scheduling semantics |
| **policy** | Exact stage gates, abort guard, evidence minimums, fixed **signal_max_marginal_scaling_efficiency_percent** and latency-inflation thresholds, and expectation-presence boolean |
| **warmups** | Ordered aggregate exact-stage traffic excluded from measurement |
| **stages** | Ordered measured results for the points that actually ran |
| **assessment** | Bounded adjacent-point signal, accepted tested point, and scaling evidence |
| **verification** | **not_requested**, **pass**, **fail**, or **not_evaluable** for an exact selected stage |
| **nonclaims** | Complete ordered fixed list of v1 inference limitations |

**plan.schedule** is fixed to **closed_loop_fixed_concurrency**, **explicit_ascending**, **each_stage_excluded**, and **simultaneous_barrier**. **concurrency_stages** therefore represents only measured points selected by the operator; no unlisted point or interpolated breakpoint was tested. **planned_attempts** includes every planned exact-stage warmup plus all measured attempts even when a safety abort prevented later stages. **plan.timeout_ns** is canonical; **timeout_ms** is the same timeout rounded up for display. Warmup, stage, and report **duration_ns** values are canonical; their **duration_ms** values are floored display conveniences.

Every entry in **warmups** contains phase status, exact concurrency, planned/attempted/succeeded/failed counts, error percentage, duration, and fixed failure-stage counts. It intentionally retains no request-level response evidence. A warmup may exist without a matching measured stage when it caused a safety abort. A measured stage contains phase status, exact concurrency, planned request count, duration, **summary**, **gates**, and deterministically indexed privacy-safe **attempts**.

**summary** contains:

- attempted/succeeded/failed counts and error percentage;
- attempted RPS and successful-request RPS as distinct fields;
- prompt/completion usage sample counts and partial observed totals, excluding non-positive or implausibly large endpoint counts;
- completion-usage and total-usage completeness booleans;
- nullable aggregate completion-token and total-token goodput, present only for complete evidence;
- nullable exact finite-sample distributions for headers, TTFT, end-to-end latency, and authoritative per-request output rate;
- fixed transport, HTTP, protocol, empty-output, and expectation failure counts.

Every distribution contains **samples**, **min**, **mean**, **p50**, **p95**, **p99**, and **max**. Percentiles use nearest rank. A **SaturationAttempt** contains only index, success, optional status/timing/usage/rate/expectation booleans, and an optional fixed failure-stage enum. It has no field capable of retaining the canary attempt's response model, finish reason, or arbitrary failure message.

Usage is plausible only when it is positive, prompt usage is at most 10,000,000 tokens per request, and completion usage is at most the requested **plan.max_tokens**. Values outside those bounds are omitted before attempt serialization and independently rejected during aggregation. They contribute to neither sample counts nor totals, force the relevant completeness boolean false, and make a selected completion-goodput gate **not_evaluable** rather than creating an inflated pass.

Gate **kind** is one of **successful_requests**, **error_percent**, **p95_ttft_ms**, **p95_e2e_ms**, **successful_requests_per_second**, or **completion_token_goodput_per_second**. The operator is fixed **greater_than_or_equal** or **less_than_or_equal**. Gate status is **pass**, **fail**, or **not_evaluable**; fixed missing-evidence reasons are **no_successful_requests**, **insufficient_samples**, and **incomplete_usage_evidence**. Optional sample and required-sample counts make fail-closed latency/token decisions auditable.

The assessment status is **signal_observed**, **no_signal_in_tested_stages**, or **not_evaluable**. A signal is **throughput_plateau_with_error_rate** or **throughput_plateau_with_latency_inflation**. Each adjacent-point scaling record identifies baseline, previous, and current concurrency and may contain raw marginal successful-RPS gain, **marginal_scaling_efficiency_percent**, scaling efficiency from the first point, and p95 TTFT/end-to-end inflation. Marginal scaling efficiency is **(RPS gain % / concurrency gain %) × 100**; 100% is proportional growth, while the fixed policy threshold is at most 5%. A latency-inflation value requires at least 20 finite successful samples in both adjacent stages. Missing denominators or all qualifying latency evidence stay null and can make the assessment not evaluable. **highest_accepted_tested_concurrency** is the largest actually run point whose complete configured gate set passed; it is explicitly not production capacity or a recommendation.

Verification includes nullable **requested_concurrency**, status, and optional fixed **stage_not_run** or **gate_not_evaluable** reason. The command exits 2 after emitting this object when the run aborted or requested verification is fail/not-evaluable. An observed saturation signal alone does not change exit status.

Canonical v1 nonclaims state that closed-loop scheduling may hide coordinated omission; the single generator can bottleneck; only listed points were tested; concurrency is not batch size/GPU occupancy; synthetic input may benefit from caching; external traffic is not isolated; token goodput uses endpoint usage; the highest accepted point is not production capacity/recommendation; no production SLA is certified; and this is not distributed, open-loop, soak, or adaptive-breakpoint testing.

The contract has no credential, prompt, expectation text, generated content, response body, response model, finish reason, arbitrary server diagnostic, URL query, URL fragment, filesystem path, or local-host identity. The configured endpoint's safe scheme/host/port origin is intentionally present. Workload IDs are limited to 128 bytes; target URL/model identities to 64 KiB. Decode-time sequence caps are eight stages, six gates per stage, 10,000 measured attempts across stages, seven scaling comparisons, and sixteen nonclaims. Unknown incompatible semantics require a benchmark-version increment.

## Benchmark Comparison v1

**gpu-watchman benchmark compare --format json|ndjson** emits a standalone **saturation_comparison_version: 1** object. It contains privacy-safe baseline/candidate identities, the exact selected comparison policy, typed compatibility checks, one ordered result per exact concurrency stage, authoritative overall **status**, a convenience **regression** boolean, and fixed nonclaims. It never embeds source reports, endpoints, paths, prompts, generated output, credentials, or request attempts.

Inputs are compatible only when both current-version reports are complete and internally reconstructed, the candidate timestamp is not earlier than the baseline, and workload ID, model, route, stream mode, full plan/schedule, and source policy match. Exact endpoint origin is deliberately not an identity field. Internal reconstruction detects inconsistent evidence but does not authenticate a coherently fabricated report.

Each selected latency gate records candidate-versus-baseline percent change. Successful-RPS and completion-token-goodput gates record candidate/baseline ratios. The error gate records candidate minus baseline percentage points. All selected gates are evaluated at every exact stage and require at least 20 relevant samples in each report; missing or incomplete evidence, a missing stage, zero ratio baseline, or incompatible identity produces typed **not_evaluable** evidence. Overall status is **regression** when an evaluable gate demonstrates a violation, **not_evaluable** when no violation is demonstrated but any required comparison cannot be evaluated, and otherwise **pass**. The **regression** boolean is true only for the first case.

## Runtime Fingerprint v1

**gpu-watchman runtime inspect --pid PID --format json|ndjson** emits a standalone **runtime_fingerprint_version: 1** object. It is not a schema-v3 hardware report and is not valid input to **history**, **compare**, **rollout**, Artifact Report, or Capacity Report workflows.

| Field | Meaning |
| --- | --- |
| **runtime_fingerprint_version** | Runtime-fingerprint contract version; currently 1 |
| **collected_at** | UTC RFC 3339 collection timestamp |
| **host** | Path-free **operating_system** (**linux** or **unsupported**) and fixed **architecture** class |
| **target_count** | Number of unique explicit PID targets, at most 32 |
| **complete** | True only on Linux when every requested PID has stable identity and all three process sources are observed |
| **driver** | Fixed NVIDIA kernel-module evidence; independent of process completeness |
| **processes** | One deterministic ascending-PID process fingerprint per requested target |
| **assessment** | Explicit no-verdict compatibility state and fixed reasons |

**driver.state** is **observed**, **not_present**, **conflict**, or **unavailable**. **driver.source** is **none**, **sys_module**, **proc_driver**, **corroborated_fixed_files**, or **conflicting_fixed_files**. **kernel_module_versions** contains only ordered one-to-four-component numeric versions read from **/sys/module/nvidia/version** and/or **/proc/driver/nvidia/version**. The files are read directly with no-follow protection and a 4 KiB bound; no **nvidia-smi** or other child command runs. A version here describes the NVIDIA kernel module, not the CUDA user-mode driver, toolkit, runtime, or compatibility. Missing or unavailable driver evidence does not make stable process evidence incomplete.

Each process has:

| Field | Meaning |
| --- | --- |
| **pid** | Explicit numeric target |
| **identity_state** | **stable**, **exited**, **reused**, or **unavailable** |
| **sources** | Fixed evidence for **identity**, **command_line**, and **memory_maps** |
| **engine_candidates** | Fixed candidate enum plus **argv_executable** or **argv_python_module** evidence |
| **framework_candidates** | Fixed candidate enum plus **mapped_library_name** evidence |
| **mapped_libraries** | Fixed family, mapped-name evidence, optional numeric SONAME/filename version, and version-evidence strength |
| **launch_observation** | Typed engine-specific argv declarations, never raw argument text |

Each source entry contains **source**, **state**, optional **reason**, and bounded numeric **records**. Source state is **observed**, **not_present**, **skipped**, or **unavailable**. Reasons are closed enums: **permission_denied**, **not_found**, **process_exited**, **pid_reused**, **changed_during_collection**, **limit_exceeded**, **malformed**, and **unsupported_platform**. No operating-system error text is retained.

Engine candidates are **vllm**, **tgi**, **triton**, **sglang**, and **tensorrt_llm**. Framework candidates are **pytorch**, **tensorflow**, **onnx_runtime**, and **tensorrt**. Mapped-library families are **cuda_driver**, **cuda_runtime**, **cublas**, **cudnn**, **nccl**, **tensorrt**, **torch_core**, **torch_cuda**, **tensorflow**, and **onnxruntime_cuda**. Numeric mapped-filename versions contain at most four components and are explicitly marked **mapped_filename_only**; absence is **not_observed**. Neither form is package-version evidence.

**launch_observation** has **evidence: process_command_line** and typed fields **model_reference_present**, **tensor_parallel_size**, **pipeline_parallel_size**, **data_parallel_size**, **context_token_limit**, **declared_dtype**, **declared_quantization**, and **declared_kv_cache_dtype**. Each is an object with **state** and nullable **value**. State is **observed**, **present_unparsed**, **ambiguous**, or **not_observed**. Raw values are interpreted only after exactly one engine identity is recognized. Duplicate declarations become ambiguous; unknown or malformed values are present-unparsed; absence is not-observed. The model reference itself is never retained.

Collection is bounded to 32 explicit unique positive PIDs. Per PID, each **stat** read is limited to 8 KiB, each of two **cmdline** reads to 256 KiB, argv to 4,096 entries and 16 KiB per entry, **maps** to 8 MiB, 65,536 records, and 16 KiB per record, and retained mapped-library evidence to 64 distinct facts. Numeric version tokens are limited to 64 bytes. On Linux, **/proc/PID** is pinned once as a no-follow directory descriptor and **stat**, **cmdline**, and **maps** are opened relative to it. Start-time **stat** reads bookend collection, the two cmdline byte sequences must match, and maps remains a one-time non-atomic observation.

The schema contains no filesystem path, hostname, environment value, model identity, raw argv, map record, package label, cgroup, credential, or arbitrary diagnostic string. **assessment.compatibility** is always **not_evaluated** in v1. Its fixed reasons are **kernel_driver_is_not_user_mode_driver**, **mapped_names_are_not_package_versions**, **argv_is_not_effective_configuration**, **no_model_artifact_binding**, and **memory_maps_are_non_atomic**. Consumers must not infer **compatible** or **incompatible** from candidates or versions.

Completeness is process evidence only: every requested target must be on Linux, have **identity_state: stable**, and have exactly one observed, reason-free identity, command-line, and memory-map source. The driver state is deliberately excluded. An incomplete report is serialized before the default exit 2. **--allow-incomplete** preserves identical report content and changes only that policy exit to 0. Invalid target selection and serialization failures exit 1 without converting the failure to incomplete process evidence.

## Artifact Report v1

**gpu-watchman artifact inspect PATH --format json|ndjson** emits a self-contained **artifact_version: 1** object after all selected safetensors metadata passes validation. It is not a schema-v3 hardware report, Capacity Report, or valid input to **history** or **compare**.

| Field | Meaning |
| --- | --- |
| **artifact_version** | Artifact-contract compatibility version; currently 1 |
| **artifact_format** | **safetensors** |
| **layout** | **single_file** or **sharded_index** |
| **summary** | Exact aggregate serialized-storage byte and tensor/element counts |
| **dtypes** | Deterministically ordered aggregate dtype summaries |
| **verification** | Explicit statements about file/header/offset/index/shape verification and payload non-inspection |
| **caveats** | Human-readable storage-versus-runtime limitations; not stable automation keys |

**summary** contains:

| Field | Meaning |
| --- | --- |
| **shard_files** | Number of inspected safetensors container files; one for **single_file** |
| **tensor_count** | Tensor descriptors validated across all headers |
| **tensor_elements** | Checked sum of shape element products |
| **serialized_tensor_bytes** | Exact sum of tensor data-offset lengths; offsets must cover every shard payload without overlap or holes |
| **serialized_shard_file_bytes** | Exact sum of complete safetensors file lengths, excluding a separate index file |
| **safetensors_header_json_bytes** | Sum of header regions declared by each length prefix, including permitted ASCII-space padding |
| **safetensors_length_prefix_bytes** | Eight bytes per inspected shard |
| **index_file_bytes** | Exact bounded index-file length for **sharded_index**; null for **single_file** |
| **declared_total_size_bytes** | Index **metadata.total_size** after equality with **serialized_tensor_bytes** is proved; null for **single_file** |

Each **dtypes** item contains **dtype**, **tensor_count**, **tensor_elements**, **serialized_bytes**, and **shape_payload_bytes_verified**. Current Artifact Report v1 accepts only the exact dtype identifiers below and verifies checked **elements × bits / 8** against every tensor's offset length. Sub-byte totals must be divisible by eight; unknown identifiers fail instead of producing offset-only evidence.

| Bits per element | Accepted identifiers |
| --- | --- |
| 4 | **F4** |
| 6 | **F6_E2M3**, **F6_E3M2** |
| 8 | **BOOL**, **U8**, **I8**, **F8_E5M2**, **F8_E4M3**, **F8_E8M0**, **F8_E4M3FNUZ**, **F8_E5M2FNUZ** |
| 16 | **U16**, **I16**, **F16**, **BF16** |
| 32 | **U32**, **I32**, **F32** |
| 64 | **U64**, **I64**, **F64**, **C64** |

Therefore **shape_payload_bytes_verified** is true for every dtype group in every successful current-v1 report.

**verification** contains:

| Field | Successful-report meaning |
| --- | --- |
| **regular_shard_files** | Every shard was opened as a regular file |
| **final_symlinks_rejected** | Final-component input/candidate/shard symlinks were rejected |
| **directory_descriptors_anchored** | True when member enumeration/opens were relative to pinned directory descriptors; true on Unix and false on the current non-Unix fallback |
| **headers_validated** | Every bounded safetensors header passed strict structural validation and its same-descriptor prefix/header reread matched |
| **data_offsets_complete_without_holes** | Sorted offsets exactly cover each complete payload without overlap or gaps |
| **index_membership_validated** | True after exact weight-map/shard membership validation; null for **single_file** |
| **declared_total_size_validated** | True after index total-size equality; null for **single_file** |
| **shape_payload_bytes_verified_tensors** | Equals **summary.tensor_count** in every successful current-v1 report |
| **shape_payload_bytes_unverified_tensors** | Reserved; always 0 in every successful current-v1 report |
| **tensor_payload_contents_read** | Always false in Artifact Report v1 |
| **payload_checksum_validated** | Always false in Artifact Report v1 |

Header **__metadata__** is admitted only as a duplicate-free string-to-string map with at most 4,096 entries, at most 8 MiB per key or value, and 8 MiB of cumulative UTF-8 key/value bytes per header. Index **metadata** is duplicate-free, capped at 64 entries and 1,024 UTF-8 bytes per key, and must contain integer **total_size**. Bounded decode visitors apply these limits—as well as 1,024-byte tensor/header names, 32-byte dtype names, and 255-byte shard names—before retaining or cloning strings. Header metadata values are not retained, only their validated byte lengths. These are parser admission limits; metadata identities and values are not serialized.

Successful inspection also requires a consistent metadata snapshot. The bounded full index is reread byte-for-byte from its same descriptor and must reach the same end-of-file; each shard's eight-byte prefix and header is likewise reread and compared. Initial/final lengths must match. On Unix, device, inode, modification seconds/nanoseconds, and change seconds/nanoseconds must also match. Payload bytes remain unread, so this is not checksum evidence.

On Unix, **directory_descriptors_anchored: true** means the selected directory or parent was pinned as an open descriptor, direct file/directory inputs passed an initial-versus-opened device/inode identity check, enumeration used the pinned descriptor, and candidate/index/shard files were opened relative to it with **openat** and final-component **O_NOFOLLOW**. **Non-Unix caveat:** the fallback rejoins names to a stored path and performs symlink and regular-file checks, but lacks equivalent descriptor-relative anchoring and the Unix identity/timestamp fingerprint, so the field is false; its snapshot compares length and **modified()**.

No report field contains the input path, index/shard filename, tensor name, weight-map key, or arbitrary safetensors metadata value. Those identities exist only during bounded validation. A sharded report proves exact storage membership and byte accounting, not runtime TP/PP/DP/EP placement; serialized bytes are not GPU/CPU residency, activation memory, or workspace estimates. See the [artifact inspection guide](../features/inference.md#safetensors-artifact-inspection) for input resolution and hard ceilings.

## Capacity Report v3

**gpu-watchman capacity --format json|ndjson** emits a self-contained **capacity_version: 3** object. Version 3 adds optional verified-artifact base-weight floors and explicit basis evidence. A report without **--artifact** still uses version 3, with **artifact: null**. Capacity reports are not schema-v3 hardware reports and are not valid input to **history** or **compare**.

| Field | Meaning |
| --- | --- |
| **capacity_version** | Capacity-contract compatibility version; currently 3 |
| **input** | Effective model geometry, precision, topology degrees, VRAM policy, per-replica workload, expert metadata, and model provenance used by the calculation |
| **artifact** | Null without **--artifact**; otherwise path-free aggregate Artifact Report v1 evidence and the multiplier-adjusted serialized-byte floor |
| **topology** | Validated TP/PP/DP/EP ranks, **world_size**, explicit worst-rank/stage upper bounds, floating KV-head replication upper bound, and optional expected-GPU assertion |
| **weights** | Base-weight basis and both candidate floors, logical base/shared/expert/overhead totals, plus shared, expert, overhead, and combined memory on the worst modeled rank |
| **kv_cache** | Logical, physical-per-DP-replica, deployment-slot, worst-rank, and requested worst-rank KV evidence |
| **memory** | Per-rank runtime allowance, usable VRAM, worst-rank estimate/headroom, and approximate full-context capacity per DP replica |
| **fits** | Whether worst-rank estimated memory is no greater than usable memory on the configured smallest GPU |
| **assumptions** | Stable snake-case code plus human explanation for every arithmetic assumption that applies |
| **caveats** | Human limitations and config-derived warnings; not stable automation keys |

When present, **artifact** has exactly these fields:

| Field | Meaning |
| --- | --- |
| **source_artifact_version** | Source contract version; currently 1 |
| **artifact_format** | Verified source format; currently **safetensors** |
| **layout** | **single_file** or **sharded_index** |
| **shard_files** | Number of verified safetensors container files |
| **tensor_count** | Aggregate verified tensor count |
| **tensor_elements** | Aggregate checked tensor-element count |
| **serialized_tensor_bytes** | Exact verified tensor-payload bytes before the residency multiplier |
| **directory_descriptors_anchored** | Whether the source inspection used descriptor-anchored directory/member access on this platform; file-resolution evidence only, never placement evidence |
| **residency_multiplier** | Finite multiplier in **[1, 1000]**, defaulting to 1 only because an artifact was supplied |
| **adjusted_residency_floor_gib_logical** | Upward whole-byte-rounded artifact floor converted to logical GiB |
| **selected_as_base_weight_floor** | True only when the adjusted artifact floor is strictly greater than the parameter/precision estimate |

This is a deliberately path-free projection, not an embedded Artifact Report. It omits the input path, shard and tensor names, dtype breakdown, weight-map identities, arbitrary safetensors metadata, headers, and payload contents. **directory_descriptors_anchored** proves only how local files were resolved during inspection. The capacity command freshly inspects the path supplied to **--artifact**; it does not accept a saved Artifact Report JSON file.

The **weights** object has exactly these fields:

| Field | Meaning |
| --- | --- |
| **base_weight_basis** | **parameter_precision_estimate** or **artifact_residency_floor** |
| **parameter_derived_weight_memory_gib_logical** | Candidate **P** derived from the effective parameter count and weight precision |
| **artifact_adjusted_weight_floor_gib_logical** | Candidate **A** after multiplication and upward whole-byte rounding, or null without an artifact |
| **base_weight_memory_gib_logical** | Selected **B**, the maximum of **P** and **A**, or **P** alone without an artifact |
| **weight_overhead_memory_gib_logical** | Operator-configured overhead **O**, applied once after selecting **B** |
| **weight_memory_gib_logical** | Logical base plus logical weight overhead |
| **shared_weight_memory_gib_logical** | Logical non-expert portion of selected **B** |
| **expert_weight_memory_gib_logical** | Logical routed-expert portion of selected **B** |
| **shared_weight_memory_gib_worst_rank** | Shared component on the heaviest modeled rank after explicit stage/rank bounds |
| **expert_weight_memory_gib_worst_rank** | Expert component on the heaviest modeled rank after explicit stage/rank bounds |
| **weight_overhead_memory_gib_worst_rank** | Overhead on the heaviest modeled stage; no automatic TP/EP sharding credit |
| **weight_memory_gib_worst_rank** | Sum of the three worst-rank weight components |

The base-weight calculation is:

~~~text
P = parameters_billion × 1,000,000,000 × weight_bits / 8
A = ceil(serialized_tensor_bytes × artifact_residency_multiplier)
B = max(P, A) with an artifact; otherwise B = P
O = B × weight_overhead_percent / 100
~~~

The artifact product is rounded upward to a whole byte before it becomes the logical GiB floor. The implementation decomposes the represented multiplier into its binary significand and exponent, then performs checked **u128** multiplication and exact ceiling division; it does not rely on a rounded floating-point product followed by **ceil**. Both raw **serialized_tensor_bytes** and the multiplier-adjusted product must be no greater than **8,000,000,000,000,000 bytes**. Non-finite, overflowing, inconsistent, or larger evidence is rejected. Equality **A = P** retains **base_weight_basis: parameter_precision_estimate** and **selected_as_base_weight_floor: false**. Consequently, adding valid artifact evidence can only preserve or increase memory and can only preserve or worsen **fits**; it never weakens the parameter/precision estimate. Weight overhead is applied once after the maximum, not once to each candidate.

**input.parameter_source** is **explicit_override**, **config_declared**, or **dense_estimate**. Current capacity-v3 estimation rejects unknown provenance. **input.model_type** is a bounded safe identifier from config.json or null; neither the config path nor raw config content is serialized. Explicit parameter/layer/KV-head/head-dimension overrides are applied before dependent validation and dense estimation, so the retained source and effective geometry describe the calculation actually performed.

Artifact evidence does not replace those inputs. **--params** remains required unless **--model-config** supplies or safely derives it, and layer/KV-head/head-dimension geometry must still be supplied explicitly or by that config. Detected MoE evidence—expert-count aliases, known MoE model types, or nested sparse-routing markers—still requires an explicit total-resident parameter override plus expert count and expert-weight percentage metadata. A config-declared parameter count is rejected for MoE because it may describe active rather than resident weights. Dense auto-estimation is limited to standard gated, bias-free layouts with exact **model_type** values **llama** or **mistral**; unknown dense families need an explicit or non-MoE config-declared total, and sparse evidence always takes the stricter MoE path.

The validated world size is **TP × PP × DP**. EP overlays the **TP × DP** ranks in a pipeline stage and does not multiply world size. When any topology degree is explicit, a supplied **--gpus** becomes **topology.expected_gpu_count** and must exactly equal world size. Without explicit topology, legacy **--gpus N** resolves to TP=N and emits **legacy_gpus_interpreted_as_tp**.

Capacity v3 never infers placement balance from a topology degree or artifact layout. Null **input.max_shared_rank_weight_percent** and **input.max_expert_rank_weight_percent** values resolve independently to 100%; their minimum legal explicit values are **100 / TP** and **100 / EP**. A null **input.max_pipeline_stage_component_weight_percent** resolves to 100% and any explicit value independently applies to shared, expert, and overhead classes; its minimum is **100 / PP**. A null **input.max_pipeline_stage_layers** resolves to all layers and an explicit value may be no lower than **ceil(layers / PP)**. Effective values appear as **topology.worst_shared_rank_weight_percent**, nullable **topology.worst_expert_rank_weight_percent**, **topology.worst_pipeline_stage_component_weight_percent**, and **topology.worst_pipeline_stage_layers**. Weight overhead receives no TP/EP rank-sharding credit. DP never divides per-rank memory, and requested/reported concurrency is per DP replica. Tighter placement bounds require independent deployment/runtime evidence; artifact shards, tensor metadata, and descriptor anchoring never justify them.

A null **input.max_kv_heads_per_rank** resolves to all KV heads; an explicit value may be between **ceil(KV heads / TP)** and the full head count, with no exact divisibility required or inferred. **topology.kv_heads_per_tensor_parallel_rank_upper_bound** records the effective complete-head bound. Floating **topology.kv_head_replication_upper_bound** is **max heads per rank × TP / logical KV heads**.

KV evidence deliberately separates **kv_cache_mib_per_sequence_logical**, **kv_cache_mib_per_sequence_physical_per_data_parallel_replica_upper_bound**, **kv_cache_mib_per_concurrency_slot_across_deployment_upper_bound**, **kv_cache_mib_per_sequence_worst_rank**, and **kv_cache_gib_requested_worst_rank**. One concurrency slot across the deployment means one full-context sequence resident on every DP replica.

Current stable assumption codes are **parameter_precision_weight_estimate**, **artifact_serialized_bytes_are_residency_floor**, **artifact_does_not_prove_runtime_placement**, **conservative_shared_weight_placement**, **explicit_shared_rank_weight_upper_bound**, **weight_overhead_is_unsharded**, **conservative_pipeline_stage_weights**, **explicit_pipeline_stage_component_weight_upper_bound**, **conservative_pipeline_stage_layers**, **explicit_pipeline_stage_layer_upper_bound**, **kv_cache_uses_layer_placement**, **conservative_kv_head_placement**, **explicit_kv_head_placement_upper_bound**, **data_parallel_concurrency_is_per_replica**, **conservative_expert_weight_placement**, **explicit_expert_rank_weight_upper_bound**, **uniform_gpu_memory**, **runtime_overhead_is_per_rank**, and **legacy_gpus_interpreted_as_tp**. The two **artifact_*** codes appear only when artifact evidence is present. Consumers should branch on the code and treat the detail as display text.

Capacity v3 is an arithmetic planning contract, not observed allocation evidence. Serialized tensor bytes are a storage-derived lower floor, not proof of loaded GPU/CPU residency, runtime precision, conversion/dequantization expansion, duplicated tensors, expert composition, or TP/PP/DP/EP, stage, layer, expert, or KV-head placement. The planner also does not model activation memory, CUDA graph captures, speculative/draft-model allocations, allocator fragmentation, kernels, multimodal encoders, or communication workspaces. The artifact multiplier, runtime allowance, and weight overhead are operator-supplied calibration inputs.

Invalid usage, a multiplier without **--artifact**, config/artifact inspection failure, inconsistent artifact evidence, out-of-range multiplier, invalid geometry, or arithmetic/ceiling failure exits 1 without a capacity report. A valid fit exits 0. A valid non-fit exits 2 after emitting the complete report, so a stronger artifact floor may change exit 0 to exit 2 but never exit 2 to exit 0.

## Envelope

| Field | Meaning |
| --- | --- |
| **schema_version** | Integer compatibility version; reject versions newer than the reader supports |
| **collected_at** | UTC RFC 3339 timestamp for the completed hardware collection |
| **collection_duration_ms** | Wall-clock time for the joined GPU and inference cycle |
| **host** | Hostname, operating system, and architecture |
| **status** | **healthy**, **warning**, or **critical**, derived from finalized findings |
| **summary** | Counts and fleet-level totals derived from the full report |
| **gpus** | Available per-device inventory and process ownership |
| **findings** | Deterministic symptoms, severity, code, message, and recommendation |
| **topology** | Optional raw NVIDIA topology matrix |
| **xid_events** | Optional recent accessible kernel-log lines containing NVIDIA Xid events |
| **endpoints** | Inference endpoint reachability and normalized metrics |
| **sources** | Evidence for each independently collected hardware/log telemetry source |

Unknown additive fields should be ignored by consumers. Required fields retain stable JSON names. New incompatible semantics require a schema-version increment.

## GPU and process records

A GPU record includes identity and driver data; performance state; temperature and fan sensors; power draw and limit; graphics/memory clocks and maxima; total/used/free VRAM; GPU and memory utilization; PCIe generation and width; compute and persistence modes; ECC mode and volatile counters; retired pages; MIG mode; active throttle reasons; and attributed processes.

Each process includes PID, process name, VRAM, and best-effort Linux owner, command, cgroup, container ID, and Kubernetes pod UID. Missing enrichment is encoded as an empty string or absent optional value, not fabricated identity.

## Telemetry-source evidence

Each source record separates “the query succeeded and returned no records” from “the query could not be used”:

~~~json
{
  "name": "nvidia.processes",
  "state": "partial",
  "duration_ms": 18,
  "records": 3,
  "required": true,
  "error": "graphics: nvidia-smi exited with exit status 1: permission denied"
}
~~~

| Field | Meaning |
| --- | --- |
| **name** | Stable source identifier |
| **state** | **ok**, **partial**, **unavailable**, or **skipped** |
| **duration_ms** | Wall time spent obtaining the source in this cycle; a cache hit or skipped source may be zero |
| **records** | Successfully retained records or lines; zero is valid when the source state is **ok** |
| **required** | Whether **--require-source** made this source part of the health policy; omitted when false |
| **error** | Optional bounded diagnostic; omitted on clean success |

The stable source names are:

- **nvidia.inventory** — required base GPU inventory;
- **nvidia.processes** — joined compute and graphics process accounting;
- **nvidia.optional** — MIG mode and clock-throttle reason fields;
- **nvidia.topology** — the NVIDIA topology matrix;
- **kernel.xid** — journalctl/dmesg Xid inspection.

Source state has precise semantics:

- **ok**: the source command completed and its usable records were retained, including a legitimate zero-record result;
- **partial**: some evidence was retained, but a companion query or one or more rows failed;
- **unavailable**: the source was attempted but produced no trusted result;
- **skipped**: collection was intentionally disabled, for example **kernel.xid** under **--no-xid**.

Collector error strings are collapsed to one line and limited to 512 bytes before entering a report. They are operational diagnostics, not stable automation keys, and they are never used as Prometheus labels. Source records default to an empty list when schema-v2 JSON is read, preserving old report compatibility.

**--require-source** accepts the stable names and the short aliases **inventory**, **processes**, **optional**, **topology**, and **xid**. A required partial source creates a warning finding; a required unavailable, skipped, or absent source creates a critical finding. A one-shot collection fails closed with exit 2 for either state, independently of **--fail-on**. Continuous watch and service workflows retain and export the unhealthy report unless **--fail-on** explicitly requests termination.

## Findings

~~~json
{
  "gpu_index": 0,
  "severity": "critical",
  "code": "temperature-critical",
  "message": "GPU temperature is 92C",
  "recommendation": "Inspect cooling, airflow, workload power, and neighboring devices immediately."
}
~~~

Host- or runtime-wide findings use **gpu_index: -1** in JSON. Internally this is **None**, preventing a host symptom from being confused with GPU zero. Finding codes are the stable automation key; messages may gain detail over time. Collection creates findings without policy coupling; the analysis finalizer attaches the recommendation and rebuilds report status and summary.

## Endpoint records

Endpoint records expose only redacted URLs. URL user information is removed and query values are replaced before serialization. Bearer tokens are never stored.

Normalized fields are:

- **reachable**, HTTP status, collection latency, runtime kind, and retained sample count;
- **requests_running** and **requests_waiting**;
- maximum **kv_cache_usage_percent**;
- optional monotonic **counters** for completed/error requests, prompt/generation tokens, preemptions, latency histogram sums/counts, and bounded fixed-schema cumulative histogram buckets;
- optional interval **rates** for request/token throughput, errors/preemptions, mean request latency/TTFT/TPOT, and **samples**, **p50_ms**, **p95_ms**, and **p99_ms** objects for request latency, TTFT, TPOT, and queue time;
- fixed-schema normalized runtime counters and gauges plus bounded processed-sample evidence, with all endpoint-controlled Prometheus family names and labels discarded;
- a redacted failure description when the endpoint cannot be used.

Absent numeric data is omitted rather than replaced with zero, because “runtime did not expose this signal” and “measured zero” are different states.

Serialized histogram snapshots expose only fixed fields: aggregate **buckets** plus **series_count**. Raw metric names and labels are never serialized. Continuous collection retains a bounded, debug-redacted private series identity map for one previous sample so resets, alias changes, and label-series churn are rejected before interval buckets are aggregated.

## Summary invariants

The finalizer rebuilds summary and status after filtering. Consumers can rely on:

- **summary.gpus == gpus.length**;
- process count equals the sum of visible GPU process records;
- endpoint totals and up counts match endpoint records;
- severity counts match findings;
- status is critical when any critical finding exists, warning when no critical but at least one warning exists, and healthy otherwise.

## JSON and NDJSON

**--format json** produces one indented object. **--format ndjson** and **--history** produce one compact object per line. History readers report a malformed record with its one-based line number. Hardware comparison and canary rollout comparison accept either a complete JSON object or the final non-empty NDJSON line, with a 64 MiB per-input safety limit.

**gpu-watchman history** validates every record before aggregation and rejects an empty stream, unsupported or mixed schema versions, and summaries/statuses or telemetry values that disagree with contained evidence. Its peak and availability fields are nullable and paired with explicit per-signal sample counts: `null` with zero samples means unavailable, while numeric zero with a non-zero count means an observed idle value.

## Compatibility workflow

When changing the domain model:

1. add fields with Serde defaults when old data has an unambiguous meaning;
2. keep secret-bearing inputs out of domain types;
3. update the report API, JSON tests, schema guide, history, comparison, and bundle readers together;
4. increment **SCHEMA_VERSION** only for a change an older consumer cannot safely interpret;
5. retain a fixture or round-trip test for every compatibility rule.
