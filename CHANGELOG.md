# Changelog

All notable GPU Watchman changes are recorded here. Version tags follow semantic versioning.

## [0.8.3] - 2026-07-19

### Fixed

- Linux amd64 and arm64 release archives now contain statically linked musl binaries instead of inheriting GLIBC 2.39 from the Ubuntu 24.04 build runner. Release CI inspects every Linux binary and rejects dynamic interpreters or shared-library dependencies before signing and publication, restoring compatibility with Debian 12 and other supported Linux hosts.
- The pinned Rust 1.88 toolchain is installed explicitly in verification and packaging jobs, removing ambiguous action inputs and keeping the declared MSRV enforceable.
- NVIDIA command-fixture tests use a bounded five-second process deadline so parallel, resource-constrained release runners do not produce one-second scheduling flakes; production collection defaults remain unchanged.

## [0.8.2] - 2026-07-19

### Fixed

- Linux release verification installs the pinned **cargo-audit 0.22.2** dependency graph with its lockfile, keeping the security gate compatible with Watchman's Rust 1.88 MSRV instead of resolving newer transitive crates that require Rust 1.89 or 1.96.

## [0.8.1] - 2026-07-19

### Fixed

- NVIDIA Blackwell and newer driver inventories no longer fail when a retired-page query field is unsupported. Core inventory is collected independently, supported retired-page counters are retained, and unavailable optional fields remain explicit partial evidence.
- Streaming TTFT begins at the first non-empty reasoning delta used by reasoning-capable OpenAI-compatible models. Reasoning is never treated as final content and cannot satisfy the configured output expectation.
- Saved canary and saturation reports now survive JSON round-trips without rejecting their own derived floating-point means, rates, gates, or scaling evidence. Integrity validation permits only a tightly bounded 16-ULP serialization drift while counts, tokens, identities, selected observations, statuses, and larger changes remain fail-closed.
- Release packaging and installer verification now publish a clean **watchman** command without macOS AppleDouble metadata in Linux archives.
- Full CLI fixtures track the independent retired-page query contract; filesystem-sensitive artifact and executable-script tests are deterministic under parallel Linux CI.

### Changed

- Generated quick-start help consistently uses the public **watchman** command.

## [0.8.0] - 2026-07-18

### Added

- Strict, explicit **config_version = 1** TOML operational profiles for monitor collection, inference probes, health policy, the HTTP service, and active canary SLOs. Profiles support deterministic config-relative paths, source-aware overrides, named/default selection, and no file auto-discovery.
- **gpu-watchman config init|validate|show** for private create-new initialization, complete offline validation, and redacted TOML/JSON inspection without reading referenced secret or prompt files.
- Stable process-only JSON/NDJSON contract for **gpu-watchman ps**, including schema version, collection identity, process-source completeness, and deterministic flattened process records.
- Explicit clear and inverse controls for profile-backed lists, credentials, GPU filters, Xid collection, service history/output, canary prompts, stream and transport policy, and optional SLO gates.
- A bounded authenticated HTTP/1.1 exporter, private history/support artifacts, an isolated GPU-node-scoped Kubernetes base for explicit mTLS overlays, SHA-pinned GitHub Actions, Dependabot maintenance, and cross-platform tag/version publication gates.
- Offline **gpu-watchman rollout** comparison for saved canary JSON/NDJSON, with stable rollout-v1 text/machine evidence, raw-attempt reconstruction, fail-closed identity/policy/status compatibility, opt-in p95 TTFT and end-to-end regression gates, p50 authoritative output-rate ratio, success-percent drop, and exit-2 CI policy.
- Fixed-schema bounded runtime histograms with interval p50/p95/p99 request-latency, TTFT, TPOT, and queue-time estimates in reports and Prometheus.
- Bounded **gpu-watchman artifact inspect PATH** for single-file and sharded-index safetensors metadata. Artifact Report v1 records exact serialized tensor/header/container/index bytes, tensor/element and deterministic dtype summaries, strict offset/index/declared-size verification, and exact shape-bit coverage without loading or checksumming tensor payloads or serializing paths, shard filenames, or tensor names. Decode-time visitors bound strings before retention; header metadata is capped at 4,096 entries, 8 MiB per key/value, and 8 MiB cumulative key/value bytes per header, while index metadata is capped at 64 entries and 1,024 bytes per key. Unknown dtypes, non-byte-aligned sub-byte tensors, and shape/payload mismatches fail closed. The full bounded index and every shard prefix/header are reread from the same descriptor; Unix snapshots additionally compare device/inode and nanosecond modification/change times. Unix inspection uses descriptor-anchored **openat/O_NOFOLLOW** member opens and reports that platform evidence explicitly; the non-Unix path-based fallback reports it as unavailable. Non-UTF-8 directory entries are ignored as candidates while explicit non-UTF-8 filenames fail. The storage evidence makes no runtime TP/PP/DP/EP placement or resident-memory claim.
- Topology-aware **capacity_version: 3** planning with explicit TP/PP/DP/EP ranks, exact world-size assertions, independent deployment/runtime-evidenced shared/expert/stage/layer/KV-head placement upper bounds, logical-versus-physical memory evidence, floating KV-head replication bounds, stable assumption codes, and per-DP-replica concurrency. Optional **--artifact PATH** freshly inspects safetensors evidence and selects **max(parameter × precision bytes, ceil(serialized tensor bytes × residency multiplier))** before applying weight overhead once. The finite multiplier is in **[1, 1000]**, defaults to 1 only with an artifact, and raw/adjusted artifact evidence is capped at **8,000,000,000,000,000 bytes**. Reports retain only path-free aggregate artifact and descriptor-anchoring evidence plus an explicit weight basis. Artifact input can never reduce modeled memory, replace required parameter/config geometry or MoE metadata, or prove loaded residency, expert composition, or TP/PP/DP/EP placement.
- Linux-only **gpu-watchman runtime inspect** for one to 32 explicit PIDs, with standalone **runtime_fingerprint_version: 1** text/JSON/NDJSON evidence. Procfs collection pins each target directory, bookends identity through **stat**, requires double-read **cmdline** stability, bounds non-atomic **maps**, and retains only fixed engine/framework/library candidates, typed launch declarations, numeric filename versions, and fixed NVIDIA kernel-module evidence. It executes no child utility and serializes no path, hostname, environment value, model identity, raw argv, map record, or arbitrary diagnostic. Compatibility is always **not_evaluated**; driver absence does not make process evidence incomplete. Incomplete reports are emitted before default exit 2, while **--allow-incomplete** changes only that exit to 0 and fatal input errors remain exit 1.
- Bounded **gpu-watchman benchmark saturation** with standalone **saturation_benchmark_version: 1** text/JSON/NDJSON evidence. Explicit unique ascending concurrency stages start at one, warm up every exact point outside measurement, reuse one OpenAI client, and run fixed closed-loop workers behind a simultaneous-start barrier. Stage evidence separates attempted from successful RPS, requires complete plausible authoritative usage for aggregate completion/total-token goodput, rejects non-positive or implausibly large token telemetry, uses nearest-rank latency distributions, retains only privacy-safe attempts and fixed failure stages, and evaluates fail-closed gates. **--verify-concurrency** turns one exact listed point into an exit-2 deployment policy; severe error/no-success guards apply to warmups and measurements, abort before higher load, and preserve a report. The adjacent-point saturation heuristic normalizes RPS growth by concurrency growth, is descriptive, and never extrapolates capacity. Plans are pre-admitted against stage/concurrency/sample, 10,000-attempt, 1,000,000-token, 64 MiB prompt, 2 GiB response-limit, 30-minute wave, and conservative 768 MiB working-set ceilings. Ten fixed nonclaims cover coordinated omission, generator/network bottlenecks, caching, external traffic, endpoint usage, GPU/batch non-equivalence, no untested breakpoint, and no production recommendation or SLA claim.
- Offline **gpu-watchman benchmark compare** with stable **saturation_comparison_version: 1** text/JSON/NDJSON evidence, deep source-report reconstruction, exact-ladder identity checks, endpoint-free output, and per-stage p95 TTFT, p95 end-to-end, successful-RPS, completion-token-goodput, and error-rate regression gates. Every selected gate requires 20 relevant samples per side; missing, incompatible, incomplete, undersampled, and zero-baseline evidence is explicitly **not_evaluable**. Optional enforcement maps every non-pass result to exit 2.

### Changed

- Passive probe bearer credentials are now fail-closed to one exact URL origin, Doctor verifies attribution evidence on collected GPU workload PIDs, and **ps** requires complete process telemetry unless **--allow-incomplete** is explicit.
- History analysis now rejects empty, unsupported, mixed-version, and structurally inconsistent report streams. Missing peak and availability signals are nullable with explicit sample counts instead of fabricated zeros.
- Replaced the shared monitor-flag surface with typed **snapshot**, **top**, **serve**, and **ps** arguments. Invalid cross-workflow flags now fail during CLI parsing before driver collection, network access, or listener binding. Running with no subcommand remains a default snapshot, while snapshot options belong to the explicit command.
- **serve** is quiet by default and uses **--emit text|ndjson** for an optional stdout stream. Deployment manifests and service examples now use only valid service flags.
- Operational values resolve as built-ins, selected profile, environment, then CLI. Direct and file-backed credentials use source-aware precedence, while conflicting forms at the same level fail closed.
- Configuration loading is bounded to a 1 MiB regular UTF-8 file, keeps bare driver commands on PATH, and never accepts inline tokens, API keys, or prompts. Unix loading rejects a symlink as the selected file, untrusted ownership, group/world-writable files, and untrusted writable ancestors except sticky shared directories. Decode diagnostics omit source excerpts, URL user information is rejected, and safe display removes fragments plus complete queries.
- Passive inference probes now reject non-loopback cleartext HTTP unless **--allow-insecure-http** (or the matching profile field) explicitly opts in. They reject URL user information, refuse redirects, and bypass ambient proxy settings so bearer tokens cannot be forwarded to an unintended proxy or redirect target.
- Every **serve** listener now requires an API bearer token by default. **--no-api-auth** is an explicit loopback-only debugging exception with strict Host-authority validation; non-loopback listeners require both **--allow-remote-listen** and a token. The server bounds workers, queued/per-peer connections, parsing, deadlines, and responses, while **/livez** plus identity-free **/healthz** remain unauthenticated for supervisors.
- Driver-wrapper stderr is reduced to bounded operational classifications before it enters reports or bundles; arbitrary wrapper diagnostics and possible secret material are no longer serialized.
- Driver children no longer inherit any **GPU_WATCHMAN_*** environment values, process owner attribution uses a bounded local **/etc/passwd** snapshot instead of potentially blocking NSS, and all human renderers remove terminal/bidirectional controls from external strings.
- Config, prompt, secret, history, comparison, and model inputs use descriptor-based bounded reads with nonblocking FIFO/device resistance. Unix trust validation checks ownership, modes, path identity, and permission-granting macOS ACLs where relevant.
- Passive probes cap targets, concurrency, aggregate body capacity, and processed metric-family bytes. Endpoint-controlled Prometheus family names and labels are discarded before serialization, and debug formatting reports only whether credentials are configured.
- Active canary admission now accounts conservatively for request clones, expectation matchers, response/parser amplification, retained attempts, and persistent inputs rather than budgeting response bodies alone.
- Active canary and saturation execution now share a generic bounded closed-loop scheduler with pre-created start-gated workers, balanced deterministic index ranges, thread-local result vectors, one ordered merge, and scheduler panic/spawn/incomplete-result errors instead of a mutex acquisition after every request.
- Saturation reports now retain canonical timeout and duration nanoseconds alongside rounded display milliseconds, allowing offline rate reconstruction without millisecond-denominator drift.
- Active canary output is now **canary_version: 2** with a bounded non-secret **workload_id** and privacy-safe effective policy evidence. The built-in prompt uses **builtin-v1**; custom CLI or profile prompts require an explicit paired identity, while version-1 reports remain readable and always fail rollout compatibility. Prompt and expectation text are never stored or hashed.
- Rollout validates bounded ordered attempts, recomputes token totals and production nearest-rank distributions, and rebuilds canonical gates/status before comparison. Reports must share the exact current version, model, workload, route, stream mode, count, concurrency, maximum tokens, timeout, response limit, and policy; URL may differ. Selected latency and throughput gates additionally require at least 20 complete successful samples per report, finite non-negative ordered distributions, complete sample coverage, and non-zero baselines.
- Runtime histogram parsing now aggregates only compatible bounded series, requires a complete finite monotonic bucket set with **+Inf**, never serializes endpoint-controlled names or labels, and omits interval quantiles on resets, alias/series churn, boundary changes, incomplete scrapes, or invalid evidence.
- Release archives, checksums, containers, and the versioned installer are attested; actions and multi-architecture container bases are immutable-pinned. Per-tag publication is serialized, reuses only drafts, refuses to replace a public release, verifies the exact asset set, and publishes only after every architecture succeeds.
- Capacity topology degrees no longer imply even placement: shared/expert rank bytes default to 100%, PP defaults every weight class and all layers to the worst stage, and TP defaults all KV heads to the worst rank unless the corresponding explicit upper bound is supplied. DP never reduces per-rank memory, weight overhead is not TP/EP rank-sharded, KV heads need not divide TP exactly, and logical totals are distinguished from physical replica/deployment/worst-rank evidence.
- Model-config overrides are applied before dependent derivation, with parameter provenance and effective safe model type retained in machine output. Detected MoE/sparse configs require an explicit total-resident parameter count plus expert placement metadata and reject config-declared counts; dense auto-estimation is fail-closed outside audited standard bias-free layouts with exact **model_type** **llama** or **mistral**.

## [0.7.0] - 2026-07-18

### Added

- Bounded **gpu-watchman canary** workflow for real OpenAI-compatible chat completions, with streaming by default, TTFT and end-to-end timing, expected-output checks, count/concurrency controls, and text/JSON/NDJSON evidence.
- CI-friendly gates for maximum TTFT, maximum end-to-end latency, minimum authoritative output-token throughput, and aggregate success percentage.
- File-backed API credentials and prompts through **--api-key-file** and **--prompt-file**, plus a deterministic privacy-safe built-in **gpu-watchman-ok** request.

### Changed

- Canary output-token throughput is derived only from server-reported completion-token usage; a selected throughput gate fails closed when authoritative usage is unavailable.
- Canary request concurrency, per-request tokens, total requested tokens, accepted response bodies, per-request timeouts, and maximum planned request-wave duration are bounded; response retention defaults to 1 MiB per attempt with a 64 MiB aggregate concurrency budget. Remote cleartext HTTP requires explicit opt-in, loopback requests bypass ambient proxies, URL user information is rejected, report targets retain only the safe origin, redirects are refused, and custom prompts are treated as transmitted data rather than safe telemetry. Canary responses must contain one completion choice, whose index must be zero when supplied, preventing expectation matches across unrelated alternatives. Passive probes now strip all URL user information and complete query strings from reports.
- File-backed credentials must be regular UTF-8 files and are read through a 64 KiB race-safe bound; direct secret values now use the same size ceiling.
- Retained child-process stdout is capped at 8 MiB and stderr at 64 KiB while both pipes are still fully drained. The hard deadline now also covers inherited pipes, and Unix process groups are terminated so a wrapper descendant cannot hang collection or grow retained output indefinitely.
- Every **--gpu** selector must now match the collected inventory; stale UUIDs and typos fail explicitly instead of producing a misleading healthy zero-GPU report.
- Exit 2 now covers completed canary evaluations whose overall success/content/SLO policy fails; individual failed attempts may be tolerated by **--min-success-percent**, while invalid usage and local setup/file/output failures remain exit 1.

## [0.6.0] - 2026-07-18

### Added

- Explicit **snapshot**, **top**/**watch**, and **serve**/**exporter** workflows while retaining the compatible no-subcommand interface.
- Focused **ps** workflow for GPU VRAM ownership, user, container, Kubernetes pod, and command identity.
- TTY-aware **--color auto|always|never** text styling with **NO_COLOR** and **TERM=dumb** support.
- File-backed probe and API bearer tokens through CLI options and **GPU_WATCHMAN_PROBE_TOKEN_FILE**/**GPU_WATCHMAN_API_TOKEN_FILE**.
- Hugging Face **config.json** capacity input with geometry derivation, explicit overrides, bounded local reads, dense-parameter estimation, and surfaced caveats.
- Schema v3 telemetry-source evidence for inventory, process accounting, optional fields, topology, and kernel Xid logs.
- Fail-closed **--require-source** policy with short/canonical aliases and report evidence; one-shot violations exit 2 while continuous monitors keep publishing.
- History summaries for source completeness and per-source/state non-OK counts.
- Comparison v2 source deltas; removed sources and **ok** to non-OK transitions are rollout regressions.

### Changed

- JSON and NDJSON are now consistent across snapshot, doctor, capacity, history, and compare: NDJSON is always one compact value per line, and repeated pretty JSON is rejected.
- Capacity numeric inputs now reject zero, negative, non-finite, out-of-range utilization, and invalid percentage values before estimation.
- History stores the selected, finalized report after **--gpu** filtering, matching stdout and the exporter.
- CLI usage failures map to documented exit 1; health gates, required-source violations, doctor failures, capacity non-fit, and selected regressions use exit 2.
- Continuous collection suppresses repeated identical error spam, applies bounded retry backoff, and reports recovery.
- Closed stdout pipes terminate cleanly instead of producing a panic traceback.
- Support bundles reuse one collection for report and doctor evidence, and can explicitly skip Xid logs.

## [0.5.0] - 2026-07-18

### Added

- Report comparison for JSON and NDJSON snapshots with GPU, endpoint, pressure, throughput, and finding deltas.
- **--fail-on-regression** rollout gate for new unhealthy findings, removed GPUs, and endpoint outages.
- Runtime counter normalization and interval request, error, preemption, prompt-token, and generation-token rates.
- Interval mean request latency, time to first token, and time per output token when supported histogram families exist.
- Runtime error and preemption findings with actionable recommendations.
- Dedicated unauthenticated **/livez** route separated from collection freshness at **/healthz**.
- Full schema, comparison, deployment, development, and extension documentation.
- Hard child-process deadline test and CLI comparison end-to-end test.

### Changed

- Reorganized the Rust crate into application, domain, telemetry, inference, analysis, operations, planning, and presentation subsystems.
- Split Linux process attribution, command execution, Prometheus encoding, monitor lifecycle, and support bundles into focused modules.
- History now summarizes peak request/token throughput, error rate, and mean TTFT.
- Prometheus exports interval runtime throughput, errors, preemptions, and latency means.
- Kubernetes liveness uses **/livez** so stale telemetry does not restart a functioning observer.

### Removed

- Entire superseded Go implementation, Go tests/module metadata, compiled Go artifact, legacy documentation, and migration-only CLI shims.

## [0.4.0] - 2026-07-18

### Added

- Initial library-first Rust rewrite with NVIDIA telemetry, inference probes, health analysis, history, capacity planning, Prometheus/API serving, support bundles, deployment manifests, CI, and multi-platform releases.

[0.8.0]: https://github.com/bas3line/gpu-watchman/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/bas3line/gpu-watchman/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/bas3line/gpu-watchman/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/bas3line/gpu-watchman/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/bas3line/gpu-watchman/releases/tag/v0.4.0
