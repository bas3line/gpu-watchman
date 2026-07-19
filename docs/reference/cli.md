# CLI reference

Run **gpu-watchman --help** or **gpu-watchman COMMAND --help** for the generated reference. Version 0.8 uses typed, workflow-specific arguments: each command accepts only options that affect that workflow. Invalid combinations fail as usage errors before GPU collection or an inference request starts.

## Global profile options

| Option | Environment | Purpose |
| --- | --- | --- |
| **--config PATH** | **GPU_WATCHMAN_CONFIG** | Load one explicit, strict TOML configuration document |
| **--profile NAME** | **GPU_WATCHMAN_PROFILE** | Select a named profile; otherwise the document's **default_profile** is used |

GPU Watchman never searches the current directory, home directory, or system directories for configuration. Pass **--config** or set **GPU_WATCHMAN_CONFIG** to opt in. **--profile** requires a configuration document. Global options may appear before or after a subcommand.

Only **snapshot**, **top**/**watch**, **serve**/**exporter**, **ps**, **canary**, and **benchmark saturation** consume operational profiles. The benchmark reuses only the profile's canary target, model, credential file, prompt/workload identity, stream/transport choice, and expectation paired with unchanged profile prompt content; its stages, sample plan, maximum tokens, timeout, response cap, and benchmark gates remain explicit benchmark inputs/defaults. Running **gpu-watchman** with no subcommand performs a default snapshot and consumes the selected monitor profile, but command-specific flags belong after the explicit **snapshot** command. **doctor**, **runtime**, **artifact**, **capacity**, **history**, **compare**, **benchmark compare**, **rollout**, **bundle**, and **completions** do not consume profile settings; supplying global **--config** or **--profile** with one of those commands is rejected instead of silently doing nothing.

Effective values follow **built-in defaults < selected profile < environment < CLI**. Environment precedence applies where an environment equivalent exists. Explicit clear and inverse flags let automation override a profile rather than being trapped by a stored true value or non-empty list. See the [configuration reference](config.md) for the complete version-1 schema and security contract.

## Collection workflows

| Command | Behavior |
| --- | --- |
| **gpu-watchman snapshot** | Collect one point-in-time report |
| **gpu-watchman top** | Live terminal view; defaults to a 2-second refresh and includes idle GPUs |
| **gpu-watchman watch** | Visible alias for **top** |
| **gpu-watchman serve** | Quiet continuous collection plus HTTP export; defaults to **127.0.0.1:9400**, a 5-second cycle, and required API authentication |
| **gpu-watchman exporter** | Visible alias for **serve** |
| **gpu-watchman ps** | Point-in-time GPU process, VRAM, owner, container, pod, and command table |
| **gpu-watchman canary** | Send bounded OpenAI-compatible chat completions and gate correctness, success, TTFT, end-to-end latency, and output-token throughput |
| **gpu-watchman benchmark saturation** | Measure an explicit closed-loop concurrency ladder and optionally verify one exact tested stage |
| **gpu-watchman benchmark compare** | Compare two saved saturation ladders offline with exact-stage regression gates |
| **gpu-watchman rollout** | Compare saved baseline/candidate canaries under like-for-like identity and quantitative regression gates |
| **gpu-watchman runtime inspect** | Inspect bounded, privacy-safe local runtime evidence for up to 32 explicit PIDs |
| **gpu-watchman artifact inspect** | Validate local safetensors storage metadata without loading tensor payloads |
| **gpu-watchman** | Run one default snapshot; use the explicit command to pass snapshot options |

~~~sh
gpu-watchman snapshot --all --details
gpu-watchman top --probe http://localhost:8000
gpu-watchman watch --watch 5s --no-clear
gpu-watchman ps --gpu 0
gpu-watchman canary --model served-model --max-ttft 2s --max-e2e 10s
gpu-watchman benchmark saturation --model served-model \
  --concurrency-stages 1,2,4,8 --verify-concurrency 8
gpu-watchman benchmark compare baseline-benchmark.json candidate-benchmark.json \
  --max-p95-e2e-regression-percent 10 --fail-on-regression
gpu-watchman rollout baseline-canary.json candidate-canary.json \
  --max-p95-ttft-regression-percent 10 --fail-on-regression
gpu-watchman runtime inspect --pid 4242 --pid 4243 --format json
gpu-watchman artifact inspect ./model.safetensors.index.json --format json
gpu-watchman serve --api-token-file /run/secrets/watchman-api-token
gpu-watchman exporter --listen 0.0.0.0:9400 \
  --allow-remote-listen \
  --api-token-file /run/secrets/watchman-api-token \
  --emit ndjson
gpu-watchman --config /etc/gpu-watchman/config.toml --profile production serve
~~~

**serve** is quiet by default. Add **--emit text** or **--emit ndjson** only when reports should also stream to stdout. **--no-emit** (with **--quiet** as a compatibility alias) explicitly overrides a profile that enables output. A bearer token is required even on loopback unless **--no-api-auth** explicitly accepts local-user access for a debugging run; non-loopback listeners always require authentication.

**ps** is intentionally point-in-time. Its text, JSON, and NDJSON outputs use a dedicated process-view contract rather than the complete monitor report.

## Typed collection options

The collection, inference-probe, and health-policy groups are available only where they have an effect:

| Option | Commands | Default or purpose |
| --- | --- | --- |
| **--gpu INDEX\|UUID,...** | snapshot, top, serve, ps | Restrict physical GPUs; all when unset |
| **--all-gpus** | snapshot, top, serve, ps | Clear a GPU selection inherited from a profile |
| **--nvidia-smi PATH** | snapshot, top, serve, ps | Driver utility or wrapper; **nvidia-smi** on PATH by default |
| **--driver-timeout DURATION** | snapshot, top, serve, ps | Driver-command deadline; defaults to 3s; **--command-timeout** and **--timeout** are aliases |
| **--no-xid** / **--xid** | snapshot, top, serve | Skip Xid collection or force it on when a profile disables it |
| **--probe URL,...** | snapshot, top, serve | Probe inference base or metrics URLs |
| **--no-probe** | snapshot, top, serve | Clear endpoints inherited from a profile |
| **--probe-token TOKEN** / **--probe-token-file PATH** | snapshot, top, serve | Authenticate passive probes; prefer a file |
| **--no-probe-auth** | snapshot, top, serve | Clear probe credentials inherited from a profile or environment |
| **--probe-timeout DURATION** | snapshot, top, serve | Deadline per inference endpoint; defaults to 3s |
| **--allow-insecure-http** / **--deny-insecure-http** | snapshot, top, serve | Permit remote cleartext probe transport, or override a profile and require HTTPS |
| **--fail-on never\|warning\|critical** | snapshot, top, serve | Exit 2 at the selected finding severity; defaults to never |
| **--require-source SOURCE,...** | snapshot, top, serve | Make named telemetry sources mandatory |
| **--no-require-source** | snapshot, top, serve | Clear required sources inherited from a profile |
| threshold flags | snapshot, top, serve | Override VRAM, temperature, and KV-cache health thresholds |
| **--process-growth-mib N** | top, serve | Per-process VRAM-growth warning; defaults to 256 MiB |

Durations accept values such as **500ms**, **5s**, **2m**, and **1h** and must be nonzero. Every **--gpu** selector must match a collected index or UUID; stale selectors fail before output, history, or service state is updated.

Passive probes allow loopback HTTP but require HTTPS for a non-loopback URL unless **--allow-insecure-http** explicitly accepts cleartext exposure. They reject URL user information, never follow redirects, and bypass ambient proxy configuration for every target. Point **--probe** directly at the final metrics endpoint; the cleartext opt-in does not permit credential forwarding through redirects or proxies. Probe authentication, timeout, or transport flags without an active endpoint are usage errors rather than silent no-ops.

### Snapshot output

| Option | Default | Purpose |
| --- | --- | --- |
| **--format text\|json\|ndjson** | text | Human view, one pretty JSON report, or one compact report |
| **--color auto\|always\|never** | auto | Text color policy |
| **--all** | false | Include healthy idle devices in text |
| **--details** | false | Include detailed hardware, topology, source, and finding evidence |
| **--history PATH** | off | Append the finalized report as private NDJSON |
| **--quiet** | false | Write only the history record; requires **--history** |

### Top runtime and output

| Option | Default | Purpose |
| --- | --- | --- |
| **--interval DURATION** | 2s | Refresh interval, at least 500ms; **--watch** is an alias |
| **--history PATH** | off | Append complete samples as private NDJSON |
| **--no-clear** | false | Append terminal samples instead of redrawing |
| **--details** | false | Include detailed evidence |
| **--format text\|ndjson** | text | Interactive text or compact stream; pretty JSON is not a live format |
| **--color auto\|always\|never** | auto | Text color policy |

### Serve runtime and output

| Option | Default | Purpose |
| --- | --- | --- |
| **--listen ADDR** | 127.0.0.1:9400 | Serve **/livez**, **/metrics**, **/healthz**, and **/api/v1/report** |
| **--allow-remote-listen** | false | Permit a non-loopback bind; API authentication is also mandatory |
| **--deny-remote-listen** | false | Override a profile and require a loopback bind |
| **--interval DURATION** | 5s | Collection interval, at least 500ms; **--watch** is an alias |
| **--history PATH** / **--no-history** | off | Append private NDJSON, or clear a profile history path |
| **--freshness DURATION** | 2m | Age after which **/healthz** reports stale collection |
| **--api-token TOKEN** / **--api-token-file PATH** | required by default | Protect metrics, discovery, and report routes |
| **--no-api-auth** | false | Clear inherited credentials and explicitly permit an unauthenticated loopback debugging API |
| **--emit text\|ndjson** | off | Also stream every report to stdout; the service is quiet by default |
| **--no-emit** | off | Disable profile-configured stdout streaming; **--quiet** is an alias |

A loopback listener requires a resolved API token unless **--no-api-auth** explicitly accepts access from other local users for that run. A non-loopback listener is rejected unless **--allow-remote-listen** or the equivalent profile field opts in and an API token resolves from CLI, environment, or profile. This is a two-part guard: neither the allow flag nor **--no-api-auth** can expose process and host telemetry remotely. The embedded server does not terminate TLS, so place remote listeners behind an authenticated, rate-limiting TLS boundary. **/livez** and identity-free **/healthz** remain unauthenticated; discovery, metrics, and the full report require the bearer token.

### Process-list output

**ps** accepts only **--gpu**, **--all-gpus**, **--nvidia-smi**, **--driver-timeout**, **--allow-incomplete**, **--format text|json|ndjson**, and **--color auto|always|never**. It does not probe inference endpoints, evaluate general monitor health thresholds, collect Xid events, write history, or start a server. The **nvidia.processes** source is required by default: incomplete accounting is still rendered but exits 2. **--allow-incomplete** explicitly restores a best-effort exit 0 while preserving incomplete source evidence in machine output.

### Health thresholds

| Option | Default |
| --- | --- |
| **--vram-warning** | 90 percent |
| **--vram-critical** | 99 percent |
| **--temperature-warning** | 82 C |
| **--temperature-critical** | 90 C |
| **--kv-cache-warning** | 85 percent |
| **--kv-cache-critical** | 95 percent |
| **--process-growth-mib** | 256 MiB |

Percentages must be finite and between 0 and 100; warning thresholds must be below critical thresholds. Process growth must be positive.

### Telemetry source coverage

Schema v3 reports evidence for five independently observed sources:

| Short alias | Canonical report name | Evidence |
| --- | --- | --- |
| **inventory** | **nvidia.inventory** | Required GPU inventory query |
| **processes** | **nvidia.processes** | Compute and graphics process queries plus parsed records |
| **optional** | **nvidia.optional** | MIG and throttle-reason fields |
| **topology** | **nvidia.topology** | NVIDIA topology matrix |
| **xid** | **kernel.xid** | Accessible kernel Xid log inspection |

Each source records **state** (**ok**, **partial**, **unavailable**, or **skipped**), duration, record count, whether it was required, and a bounded error when applicable. Short and canonical names are both accepted:

~~~sh
gpu-watchman snapshot --details \
  --require-source inventory,processes,topology

# Intentionally proves the fail-closed path: skipped Xid is required.
gpu-watchman snapshot --no-xid --require-source xid --format json
~~~

A one-shot report with a required source in any state other than **ok** exits 2. The report is still emitted with source evidence and a **telemetry-source-required** or **telemetry-source-partial** finding. Continuous **top**/**watch** and **serve**/**exporter** keep exporting and recording the violation instead of stopping.

### Color and accessibility

**--color auto** styles text only when stdout is a terminal. It honors **NO_COLOR** and **TERM=dumb**. Status remains visible as text, so color is never the only signal. Use **--color never --no-clear** for screen readers or durable terminal logs.

### Authentication and environment

Direct-token and token-file options are mutually exclusive for the same secret. Token inputs are limited to 64 KiB; files must be regular UTF-8 files, are read through a race-safe bound, are trimmed, and are rejected when empty. File-based secrets avoid exposing bearer values in process arguments. Clear flags take precedence over profile and environment credentials.

| Variable | Equivalent option |
| --- | --- |
| **GPU_WATCHMAN_CONFIG** | global **--config** |
| **GPU_WATCHMAN_PROFILE** | global **--profile** |
| **GPU_WATCHMAN_NVIDIA_SMI** | **--nvidia-smi** |
| **GPU_WATCHMAN_PROBE_TOKEN** | **--probe-token** |
| **GPU_WATCHMAN_PROBE_TOKEN_FILE** | **--probe-token-file** |
| **GPU_WATCHMAN_API_TOKEN** | **--api-token** |
| **GPU_WATCHMAN_API_TOKEN_FILE** | **--api-token-file** |
| **GPU_WATCHMAN_INFERENCE_URL** | **canary/benchmark saturation --base-url** |
| **GPU_WATCHMAN_INFERENCE_MODEL** | **canary/benchmark saturation --model** |
| **GPU_WATCHMAN_INFERENCE_API_KEY** | **canary/benchmark saturation --api-key** |
| **GPU_WATCHMAN_INFERENCE_API_KEY_FILE** | **canary/benchmark saturation --api-key-file** |
| **NO_COLOR** | Disables automatic text color when set |

Probe credentials work with monitor workflows, **doctor**, and **bundle**. The current probe credential is shared client-wide and is therefore admitted only when every configured probe resolves to one exact URL origin (the same scheme, host, and port); a multi-origin authenticated set fails before any request. Run separate authenticated invocations for separate origins. CLI credential inputs take precedence over either environment form even when one source uses a direct token and the other uses a file. **--no-probe-auth** clears environment credentials for doctor and bundle as well as profile/environment credentials for monitor commands. Canary credentials are scoped to the active chat-completions request. API authentication applies to **/metrics**, **/**, and **/api/v1/report**; minimal **/livez** and **/healthz** probes remain unauthenticated. File-backed credentials are bounded, ownership/mode checked, and may use a validated projected-secret symlink. Tokens, URL credentials, query values, prompts, and generated content are not serialized.

## Machine output

- **--format json** writes one indented JSON value and is intended for point-in-time output.
- **--format ndjson** writes exactly one compact JSON value and newline per result. In **top**, every cycle is a separate line.
- **serve --emit ndjson** writes one compact report per cycle; without **--emit**, serve writes no reports to stdout.
- **ps --format json|ndjson** emits the stable process-view contract, not a generic full report.
- **doctor**, **canary**, **benchmark saturation**, **runtime inspect**, **artifact inspect**, **capacity**, **history**, **compare**, and **rollout** honor the same JSON/NDJSON distinction.
- Machine formats never contain color or progress text; operational messages and errors use stderr. Arbitrary stderr from an **nvidia-smi** wrapper is not copied into reports or bundles: it is reduced to a bounded operational classification.

## config

Creates, validates, and safely inspects explicit operational profiles:

~~~sh
gpu-watchman config init gpu-watchman.toml
gpu-watchman --config gpu-watchman.toml config validate
gpu-watchman --config gpu-watchman.toml --profile production config show
gpu-watchman --config gpu-watchman.toml config show --format json
~~~

**init** uses create-new semantics and mode 0600 on Unix. **validate** enforces the strict versioned schema without contacting hardware or endpoints. Unix loading rejects a symlink as the selected file, untrusted ownership, unsafe file permissions, and untrusted writable ancestors. Configuration URLs containing user information are rejected. **show** normalizes config-relative paths, removes fragments, redacts complete queries, and never reads referenced secret or prompt files. See the [operational profile reference](config.md) for the schema, precedence, permission, and clear-override contracts.

## doctor

Validates driver response, GPU telemetry, every source status, Linux process attribution, and configured inference endpoints.

~~~sh
gpu-watchman doctor \
  --probe https://vllm.example/metrics \
  --probe-token-file /run/secrets/runtime-metrics-token \
  --format json
~~~

Exit 2 means at least one required check failed. A reachable HTTP endpoint still fails when it exposes zero recognized runtime samples, preventing a wrong path, health page, or HTML response from receiving a false PASS. A recognized response truncated by the probe safety bounds is a visible warning. Source limitations are warnings and remain visible without failing doctor. Linux process attribution is checked against collected GPU workload PIDs; when no live process exists, Doctor reports that attribution was not exercised instead of claiming a pass from its own procfs entry.

Doctor accepts **--allow-insecure-http** only as an explicit opt-in for non-loopback cleartext probe URLs. **--probe-token**, **--probe-token-file**, **--no-probe-auth**, and the transport opt-in require at least one **--probe** URL; otherwise the command rejects the no-op combination. Passive doctor probes do not follow redirects or use ambient proxies.

## canary

Sends real chat-completion requests to an OpenAI-compatible API and emits one aggregate result with per-request evidence. It is intended for deployment smoke tests and lightweight SLO gates:

~~~sh
gpu-watchman canary --base-url URL [--allow-insecure-http] --model NAME \
  [--api-key TOKEN | --api-key-file PATH] \
  [--prompt TEXT | --prompt-file PATH] [--workload-id ID] [--expect TEXT] \
  [--max-tokens N] [--count N] [--concurrency N] \
  [--timeout DURATION] [--no-stream] \
  [--max-ttft DURATION] [--max-e2e DURATION] \
  [--min-output-tokens-per-second N] \
  [--min-success-percent 0..100] \
  [--format text|json|ndjson]
~~~

For example:

~~~sh
gpu-watchman canary \
  --base-url https://candidate.example/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --api-key-file /run/secrets/inference-api-key \
  --count 10 --concurrency 2 \
  --max-ttft 2s --max-e2e 10s \
  --min-output-tokens-per-second 20 \
  --min-success-percent 100 --format json
~~~

| Option | Default | Purpose |
| --- | --- | --- |
| **--base-url URL** | **http://127.0.0.1:8000/v1** | OpenAI-compatible API base; the canary calls its chat-completions route |
| **--allow-insecure-http** | false | Permit cleartext HTTP to a non-loopback endpoint; prefer HTTPS |
| **--deny-insecure-http** | false | Override a profile and reject non-loopback cleartext HTTP |
| **--model NAME** | required from CLI, environment, or profile | Served model identifier sent in every request |
| **--api-key TOKEN** | off | Bearer credential sent to the API; prefer the file form because command arguments may be visible locally |
| **--api-key-file PATH** | off | Read the bearer credential from a bounded local file; conflicts with **--api-key** |
| **--no-api-key** | false | Clear profile or environment API-key inputs |
| **--prompt TEXT** | built-in safe prompt | Custom user prompt; conflicts with **--prompt-file** |
| **--prompt-file PATH** | off | Read a custom prompt from a bounded local file |
| **--default-prompt** | false | Ignore a profile prompt file and use the built-in prompt |
| **--workload-id ID** | **builtin-v1** for the built-in prompt | Non-secret identity required for custom prompt content; CLI overrides a paired profile identity |
| **--expect TEXT** | prompt-dependent | Require generated output to contain this text |
| **--no-expect** | false | Clear an expectation inherited from the profile |
| **--max-tokens N** | 16 | Maximum completion tokens per request, bounded to 65,536 |
| **--count N** | 1 | Total requests to execute |
| **--concurrency N** | 1 | Maximum simultaneous requests, capped at 64 and never above count |
| **--timeout DURATION** | 30s | Deadline for each request, including its streamed body; maximum 5m |
| **--stream** | true unless a profile disables it | Force streaming when a profile disables it |
| **--no-stream** | false | Request one non-streamed response; conflicts with **--max-ttft** and **--min-output-tokens-per-second** |
| **--max-ttft DURATION** | off | Fail requests whose time to first generated content exceeds the limit |
| **--no-max-ttft** | false | Clear a profile TTFT gate |
| **--max-e2e DURATION** | off | Fail requests whose complete response exceeds the limit |
| **--no-max-e2e** | false | Clear a profile end-to-end gate |
| **--min-output-tokens-per-second N** | off | Require streamed, authoritative decode throughput; **--min-tps** is an alias |
| **--no-min-tps** | false | Clear a profile output-token-rate gate |
| **--min-success-percent 0..100** | 100 | Minimum percentage of requests that must pass; at least one success is always required |
| **--format text\|json\|ndjson** | text | Human summary, one pretty JSON result, or one compact JSON result |

The default prompt is **Reply with exactly: gpu-watchman-ok**, its implicit expectation is **gpu-watchman-ok**, and its stable workload identity is **builtin-v1**. A custom CLI **--prompt** or **--prompt-file** requires an explicit **--workload-id**; a profile **prompt_file** must be paired with **canary.workload_id**, which the CLI flag may override. IDs are non-secret, bounded labels. GPU Watchman never hashes or stores prompt content to create an identity. When custom input has no CLI or profile expectation, any non-empty generated output satisfies the content check. An explicit **--expect** always applies to the selected prompt; use **--no-expect** to clear a profile expectation.

Streaming is enabled by default so the first generated content yields a meaningful TTFT. Every request also records end-to-end latency and whether its content expectation passed. HTTP, transport, timeout, response-size, stream/protocol, empty-output, and expectation failures count against **--min-success-percent**. Selected latency and throughput bounds are separate fail-closed aggregate gates over the successful-request measurements. **--max-ttft** and **--min-output-tokens-per-second** require streaming timing, so the CLI rejects either option when combined with **--no-stream** as an exit-1 usage error.

Canary v2 machine output records the privacy-safe effective policy: minimum success
percentage, optional TTFT/end-to-end/output-rate thresholds, and whether an
expectation was configured. It does not retain expectation text. Attempts and gates
are bounded so a saved result can be validated and compared offline.

Output-token throughput is authoritative only when the server returns OpenAI-compatible completion-token usage. It measures decode rate as **(completion tokens - 1) / (end-to-end time - TTFT)**, so it requires a streamed response and at least two completion tokens. GPU Watchman does not estimate tokens from bytes, words, or a local tokenizer. If the gate is selected and usage, streaming timing, or enough completion tokens are unavailable, it fails closed instead of receiving a fabricated rate.

The canary bounds request concurrency, accepted response bodies, and the total requested completion-token budget (**count × max tokens** cannot exceed 1,000,000). It retains at most 1 MiB from each response by default, with an internal per-response ceiling of 8 MiB. Plans are rejected when a conservative 768 MiB aggregate working-set estimate—covering concurrent request clones, raw response buffers, worst-case JSON/parser amplification, expectation matcher tables, and retained attempt evidence—would be exceeded. Each timeout is at most 5 minutes, and **ceil(count / concurrency) × timeout** cannot exceed 15 minutes. Each response must contain exactly one completion choice, and any supplied choice index must be zero, so unrelated alternatives cannot combine into a false content match. It is deliberately a small verification workload, not a saturation or latency-distribution benchmark; use **benchmark saturation** for the bounded local concurrency ladder below.

API keys are never included in text or machine output. Prefer **--api-key-file** so the secret is not exposed in process arguments. Prefer **--prompt-file** for custom input because direct **--prompt** values can be visible in the process list and shell history; **--expect** is also an argument, so keep expectation markers non-sensitive. The built-in prompt contains no tenant, customer, model-input, or workload data; custom prompt content is transmitted to the configured server and may enter its logs, so use synthetic input. Base URLs containing user information are rejected, and serialized targets retain only the safe origin. HTTP is allowed by default only for loopback hosts; a remote HTTP endpoint requires **--allow-insecure-http**, which can expose prompts and bearer credentials on the network and should be reserved for explicitly trusted test environments. Loopback requests bypass ambient proxy variables so local credentials and prompts stay local.

Canary exit codes are workflow-specific:

| Code | Meaning |
| --- | --- |
| 0 | The requests ran and the content, success-percentage, and every selected SLO gate passed |
| 1 | Invalid usage, input/secret file failure, URL/setup failure, or output encoding prevented the canary workflow from running |
| 2 | The overall success/content/SLO policy failed; individual failed attempts can be tolerated by **--min-success-percent** when every other gate passes, and a structured result is still emitted |

## benchmark saturation

Runs a bounded single-process, closed-loop fixed-concurrency ladder against the same OpenAI-compatible chat-completions path as the canary:

~~~sh
gpu-watchman benchmark saturation --model NAME \
  --concurrency-stages 1,2,4,8 \
  [--base-url URL] [--api-key KEY | --api-key-file PATH] \
  [--prompt TEXT | --prompt-file PATH] [--workload-id ID] \
  [--expect TEXT | --no-expect] \
  [--warmup-requests-per-worker N] [--requests-per-worker N] \
  [--max-tokens N] [--timeout DURATION] [--response-limit-bytes N] \
  [--no-stream] [--verify-concurrency N] \
  [--max-error-percent PERCENT] \
  [--max-p95-ttft DURATION] [--max-p95-e2e DURATION] \
  [--min-successful-requests-per-second N] \
  [--min-completion-token-goodput-per-second N] \
  [--abort-error-percent PERCENT] [--format text|json|ndjson]
~~~

| Option | Default | Purpose |
| --- | --- | --- |
| **--concurrency-stages N[,N...]** | required | Explicit unique, strictly increasing points; must start at 1, with at most eight points and a maximum of 64 |
| **--warmup-requests-per-worker N** | 2 | Unmeasured requests per worker before every exact stage; range 1-10 |
| **--requests-per-worker N** | 20 | Measured requests each worker completes at every point; range 1-100 |
| **--base-url URL** | **http://127.0.0.1:8000/v1** | OpenAI-compatible base; profile/environment precedence matches canary |
| **--model NAME** | required from CLI, environment, or profile | Requested model identity |
| **--api-key / --api-key-file / --no-api-key** | off | Credential source or explicit clearing; prefer the private file form |
| **--prompt / --prompt-file / --default-prompt** | built-in deterministic prompt | Synthetic request content; custom content requires **--workload-id** |
| **--workload-id ID** | **builtin-v1** for built-in input | Bounded non-secret workload identity, never a copy or hash of the prompt |
| **--expect TEXT / --no-expect** | prompt/profile dependent | Correctness check applied to warmup and every measured request |
| **--max-tokens N** | 128 | Requested completion-token ceiling per attempt; maximum 65,536 |
| **--timeout DURATION** | 10s | Per-request deadline; maximum 5m |
| **--response-limit-bytes N** | 131,072 | Accepted response cap per request; range 1 byte-8 MiB |
| **--stream / --no-stream** | streaming, or selected profile value | Response mode; p95 TTFT requires streaming |
| **--verify-concurrency N** | off | Turn the gates at this exact listed point into an exit-code deployment policy |
| **--max-error-percent P** | 1 | Accepted measured error percentage; finite and in **[0, 100)** |
| **--max-p95-ttft DURATION** | off | Optional p95 TTFT gate; at least 20 finite successful samples required |
| **--max-p95-e2e DURATION** | off | Optional p95 end-to-end gate; at least 20 finite successful samples required |
| **--min-successful-requests-per-second N** | off | Optional successful-request goodput gate; failed attempts never inflate it |
| **--min-completion-token-goodput-per-second N** | off | Optional aggregate completion-token goodput gate; every successful request must contain authoritative usage |
| **--abort-error-percent P** | 50 | Stop after an exact-stage warmup or measured phase reaches this error rate; a warmup abort prevents that point's measurement; must exceed **--max-error-percent** and be at most 100 |
| **--format text\|json\|ndjson** | text | Human ladder, one pretty JSON report, or one compact JSON report |

Every exact stage first runs **concurrency × warmup-requests-per-worker** excluded requests, exercising the reused client and pool at that point to reduce cold connection setup without claiming every measured transport path was pre-opened. A measured stage then runs exactly **requests-per-worker** requests on every worker. The fixed OS worker set and result buffers are created before timing, released through one simultaneous-start barrier, and each worker sends its next request only after its previous one finishes. Reported stage duration ends at the final request completion and excludes thread joining/result merging. Results use deterministic fixed worker ranges and are merged into attempt-index order. One HTTP client and connection pool are reused for the whole run. This is closed-loop scheduling and therefore can hide queueing through coordinated omission; there is no target-arrival-rate generator.

For measured stage wall time **T**, the report separates:

~~~text
attempted_requests_per_second = attempted / T
successful_requests_per_second = succeeded / T
completion_token_goodput_per_second = successful authoritative completion tokens / T
~~~

Failed requests can increase attempted rate but never successful request or token goodput. Prompt and completion usage sample counts, partial observed totals, and explicit completeness booleans make missing endpoint usage visible. Only positive counts are usable; prompt usage above 10,000,000 tokens per request and completion usage above that request's configured **max_tokens** are treated as implausible endpoint telemetry, omitted from attempt evidence and totals, and counted as missing. Completion-token goodput exists only when every successful request supplied plausible authoritative completion usage; total-token goodput additionally requires complete plausible prompt usage. Latency and per-request output-rate distributions use the production nearest-rank percentile implementation over finite successful samples.

Every stage always evaluates at least-one-success and maximum-error gates. Selected latency, successful-RPS, and token-goodput gates are added to every stage, making **highest_accepted_tested_concurrency** auditable. A latency gate with fewer than 20 finite successful samples and a token gate with incomplete authoritative usage are **not_evaluable**, never silently passed. **--verify-concurrency** must name a listed point: it passes only when every gate there passes, fails on an observed violation, and is not evaluable when required evidence is missing or a safety abort prevented that stage from running.

The descriptive saturation heuristic examines only adjacent listed points. It first normalizes successful-RPS growth by the exact concurrency growth: **marginal scaling efficiency = (RPS gain % / concurrency gain %) × 100**. A value of 100% means throughput grew proportionally to the step; the signal requires efficiency at or below 5% and either p95 TTFT/end-to-end latency inflation of at least 20% or error percentage above the configured maximum. This prevents narrow stages such as 20→21 from looking saturated merely because their absolute percentage step is small. A latency-inflation comparison requires at least 20 finite successful samples in both adjacent stages. Missing throughput or all qualifying latency evidence produces **not_evaluable** instead of “unsaturated.” Observing the signal does not itself fail the command. **highest_accepted_tested_concurrency** means only that this exact observed point passed the configured stage gates; it is not maximum capacity, a production recommendation, or an extrapolation.

Admission happens before the client is built or a request is sent. The complete plan is capped at 10,000 warmup-plus-measured attempts, 1,000,000 requested completion tokens, 64 MiB of aggregate raw prompt bytes, 2 GiB of aggregate response-limit bytes, 30 minutes of worst-case sequential request waves, and a conservative 768 MiB peak working-set estimate. The memory estimate includes persistent inputs, retained evidence, concurrent request clones, bounded response buffers, worst-case JSON parser amplification, and expectation matching. The same strict URL, HTTPS-by-default remote transport, no-redirect, loopback no-proxy, response parsing, one-choice, file trust, and privacy rules as canary apply.

Saturation Benchmark v1 retains a safe target origin, requested model, non-secret workload ID, explicit plan/policy, one aggregate warmup per reached point, measured stage summaries/gates, privacy-safe measured attempts, scaling evidence, verification, and ten fixed nonclaims. It never retains a credential, prompt, expectation, generated text, response body, response model, finish reason, URL query/fragment, or arbitrary server error. Repeated synthetic prompts may exercise prefix caching; unrelated traffic is not isolated; the client or network may bottleneck first; concurrency is neither server batch size nor GPU occupancy; endpoint-reported token goodput is not raw GPU decode capacity. The workflow makes no distributed, open-loop, soak, adaptive-breakpoint, cost, production-p99, or SLA-certification claim.

| Code | Meaning |
| --- | --- |
| 0 | The full explicit schedule completed and verification was not requested or passed; a saturation signal alone is still exit 0 |
| 1 | Usage, plan admission, input/secret file, URL/client setup, scheduler, serialization, or output failure prevented a valid report |
| 2 | A privacy-safe report was emitted, then the run is considered unhealthy because it aborted or requested verification failed/was not evaluable |

## benchmark compare

Reconstructs and compares two saved Saturation Benchmark v1 JSON reports, or the final non-empty value in each NDJSON file, without contacting an inference endpoint:

~~~sh
gpu-watchman benchmark compare BASELINE CANDIDATE \
  [--max-p95-ttft-regression-percent PERCENT] \
  [--max-p95-e2e-regression-percent PERCENT] \
  [--min-successful-rps-ratio RATIO] \
  [--min-completion-token-goodput-ratio RATIO] \
  [--max-error-percent-increase POINTS] \
  [--fail-on-regression] [--format text|json|ndjson]
~~~

Both reports must be complete, internally reconstructable, ordered baseline then candidate, and identical in workload ID, requested model, route, stream mode, full plan/schedule, and source policy. Origins may differ and are excluded from the result. Every selected gate runs at every exact concurrency point and needs at least 20 relevant samples per report. Missing stages or usage, undersampling, zero ratio baselines, incompatible inputs, and incomplete reports are **not_evaluable**, not passes. See the [benchmark comparison runbook](../operations/benchmark-comparison.md) for formulas and CI examples.

Without **--fail-on-regression**, every valid emitted comparison exits 0 so operators can inspect its typed **pass**, **regression**, or **not_evaluable** status. With enforcement enabled, only **pass** exits 0; **regression** and **not_evaluable** exit 2. Read, decode, version, and usage failures exit 1 without emitting a fabricated comparison.

## runtime inspect

Inspects one to 32 explicitly selected local process IDs and emits the standalone Runtime Fingerprint v1 contract:

~~~sh
gpu-watchman runtime inspect --pid PID [--pid PID...] \
  [--allow-incomplete] [--format text|json|ndjson]
~~~

| Option | Default | Purpose |
| --- | --- | --- |
| **--pid PID** | required | Positive local PID; repeat for at most 32 unique targets |
| **--allow-incomplete** | false | Preserve incomplete evidence but change its policy exit from 2 to 0 |
| **--format text\|json\|ndjson** | text | Human summary, one pretty JSON report, or one compact JSON report |

The workflow is Linux-only and never discovers targets automatically. On Linux it opens each **/proc/PID** once as a no-follow directory descriptor; opens only **stat**, **cmdline**, and **maps** relative to that descriptor; compares start time from **stat** before and after collection; and requires two bounded **cmdline** reads to match exactly. The one **maps** read remains non-atomic. Bounds are 8 KiB per **stat** read, 256 KiB per **cmdline** read, 4,096 arguments, 16 KiB per argument, 8 MiB of maps, 65,536 map records, 16 KiB per map record, and 64 retained recognized mapped-library facts per process.

It starts no child command and does not use **nvidia-smi**, engine or interpreter version commands, a package manager, or **ldconfig**. Kernel-module evidence comes only from bounded no-follow reads of **/sys/module/nvidia/version** and **/proc/driver/nvidia/version**. Those reads are limited to 4 KiB, retained numeric version tokens are limited to 64 bytes, and their output is not labeled as a CUDA user-mode driver, toolkit, or runtime version. Driver absence does not make complete process evidence incomplete.

The machine contract contains fixed engine candidates (**vllm**, **tgi**, **triton**, **sglang**, **tensorrt_llm**), framework candidates (**pytorch**, **tensorflow**, **onnx_runtime**, **tensorrt**), fixed mapped-library families, and typed argv declarations. Launch fields are parsed only for one recognized engine identity and include TP/PP/DP sizes, context-token limit, declared dtype/quantization/KV-cache dtype, and a model-reference-presence boolean. Duplicate declarations are **ambiguous**; unsupported values are **present_unparsed**; missing declarations are **not_observed**.

Runtime Fingerprint v1 serializes no path, hostname, environment value, model identity, raw argv, raw map record, or arbitrary diagnostic. Compatibility is always **not_evaluated**: candidate identities, mapped filenames, kernel-module evidence, and launch declarations cannot prove compatibility. Complete evidence exits 0. Incomplete evidence—including **unsupported_platform** on non-Linux—still prints the report and exits 2; **--allow-incomplete** changes only that exit to 0. Missing/zero, duplicate, or more than 32 PID values and serialization/encoding failures are fatal exit-1 errors and do not become incomplete reports.

## artifact inspect

Inspects bounded safetensors metadata and emits Artifact Report v1:

~~~sh
gpu-watchman artifact inspect PATH [--format text|json|ndjson]
~~~

**PATH** may be one **.safetensors** file, one **.safetensors.index.json** file, or an unambiguous directory. A directory containing exactly one index selects the sharded-index layout; without an index, exactly one safetensors file is required. Non-UTF-8 directory entries count toward the enumeration bound but are skipped during candidate discovery; an explicit non-UTF-8 filename fails. Multiple indexes, multiple standalone files without an index, final-component symlinks, non-regular shards, unsafe index shard names, and unsupported suffixes fail with exit 1.

The inspector reads only the bounded index, each eight-byte length prefix, and safetensors header regions. Bounded decode visitors validate strings before retaining/cloning them: tensor/header names are at most 1,024 UTF-8 bytes, dtype names 32, shard names 255, each header metadata key/value 8 MiB, and each index metadata key 1,024 bytes. It validates duplicate-free metadata, tensor descriptor shape/dtype/offset structure, complete hole-free payload offsets, and—for indexed artifacts—exact tensor-to-shard membership plus **metadata.total_size**. Header **__metadata__** is limited to 4,096 string-to-string entries and 8 MiB of cumulative key/value bytes per header; index **metadata** is limited to 64 entries.

Only the current supported safetensors identifiers are accepted: **F4**; **F6_E2M3**, **F6_E3M2**; **BOOL**, **U8**, **I8**, **F8_E5M2**, **F8_E4M3**, **F8_E8M0**, **F8_E4M3FNUZ**, **F8_E5M2FNUZ**; **U16**, **I16**, **F16**, **BF16**; **U32**, **I32**, **F32**; and **U64**, **I64**, **F64**, **C64**. Every tensor's checked **elements × dtype bits** must be divisible by eight and exactly match its offset length. This includes sub-byte dtypes; an unknown dtype, a non-byte-aligned sub-byte total, or any mismatch fails with exit 1. Successful v1 output has every dtype's **shape_payload_bytes_verified** set to true, verified tensor count equal to total tensor count, and the reserved **shape_payload_bytes_unverified_tensors** set to 0.

The full bounded index and every shard length-prefix/header are reread from the same descriptor and compared byte-for-byte; payload contents are still not read. Unix before/after snapshots additionally require equal device, inode, length, and nanosecond modification/change times, so ordinary same-length mutations fail.

On Unix, directory enumeration and candidate/index/shard opens are anchored to open directory descriptors; members are opened relative to the pinned descriptor with **openat** and **O_NOFOLLOW**, and machine output records **directory_descriptors_anchored: true**. **Non-Unix caveat:** the fallback performs symlink and regular-file checks on rejoined paths but cannot provide descriptor-relative directory anchoring or the Unix identity/timestamp fingerprint, so the field is false; its snapshot compares length and **modified()**.

Text and machine output contain no input path, shard filename, tensor name, or arbitrary safetensors metadata value. They report aggregate exact storage bytes, tensor/element counts, deterministic dtype summaries, and verification booleans. Tensor payload contents are not read or checksummed. Artifact shards are storage containers, not evidence of runtime TP/PP/DP/EP placement, and serialized bytes are not a resident-memory estimate. See the [artifact inspection guide](../features/inference.md#safetensors-artifact-inspection) and [Artifact Report v1 schema](report-schema.md#artifact-report-v1) for hard limits and exact fields.

## capacity

Calculates topology-aware worst-rank weights, KV cache, runtime overhead, headroom, and approximate full-context concurrency per data-parallel replica. Physical **--gpu-vram** is always required and should describe the smallest-memory rank in the placement.

Manual geometry:

~~~sh
gpu-watchman capacity \
  --params 70 --weight-bits 4 \
  --tp 2 --gpu-vram 80 --utilization 0.9 \
  --layers 80 --kv-heads 8 --head-dim 128 \
  --max-shared-rank-weight-percent 50 \
  --max-kv-heads-per-rank 4 \
  --context 32768 --concurrency 8 --runtime-overhead 6
~~~

Hugging Face geometry:

~~~sh
gpu-watchman capacity --model-config ./config.json \
  --tp 2 --gpu-vram 80 --weight-bits 4 \
  --max-shared-rank-weight-percent 50 \
  --context 32768 --concurrency 8
~~~

Hugging Face geometry with a verified artifact floor:

~~~sh
gpu-watchman capacity --model-config ./config.json \
  --artifact ./model.safetensors.index.json \
  --artifact-residency-multiplier 1.25 \
  --tp 2 --gpu-vram 80 --weight-bits 4 \
  --max-shared-rank-weight-percent 50 \
  --context 32768 --concurrency 8
~~~

Artifact options:

| Option | Meaning |
| --- | --- |
| **--artifact PATH** | Inspect a safetensors file, sharded index, or unambiguous directory and use its verified serialized tensor bytes only as a possible stronger base-weight floor |
| **--artifact-residency-multiplier M** | Multiply verified serialized tensor bytes before selecting the floor; requires **--artifact**, must be finite and in **[1, 1000]**, and defaults to 1 only when **--artifact** is present |

The artifact path is inspected directly with the same bounded validation as **artifact inspect**; a saved Artifact Report JSON file is not accepted. Artifact input does not supply parameters, layers, KV heads, head dimension, or MoE metadata. **--params** is still required unless **--model-config** provides or safely derives it, geometry must still be complete, and detected MoE models still require an explicit total-resident **--params** plus expert count and expert-weight percentage.

For **P = parameters_billion × 1,000,000,000 × weight_bits / 8**, the optional artifact floor is **A = ceil(serialized_tensor_bytes × artifact_residency_multiplier)**. Capacity v3 chooses **B = max(P, A)**, or **B = P** without an artifact, then applies **--weight-overhead-percent** exactly once to **B**. The artifact product is rounded upward to a whole byte. Raw and adjusted artifact evidence must each be no greater than **8,000,000,000,000,000 bytes**. The artifact basis is selected only when **A > P**; equality keeps the parameter/precision basis. Therefore an artifact can preserve or increase the estimate and preserve or worsen fit, but can never reduce modeled memory or make a non-fit fit.

Topology options:

| Option | Placement meaning |
| --- | --- |
| **--tp N** | Tensor-parallel rank count; does not by itself prove shared-weight or KV-head sharding |
| **--pp N** | Pipeline-parallel stage count; defaults to 1, must not exceed the layer count, and does not prove byte or layer balance |
| **--dp N** | Full data-parallel replica count; defaults to 1 and never divides per-rank weights, KV cache, or requested concurrency |
| **--ep N** | Expert-parallel degree overlaid within each stage's **TP × DP** ranks; defaults to 1 and does not multiply world size |
| **--gpus N** | Legacy TP count when no topology degree is supplied; otherwise an exact **TP × PP × DP** world-size assertion |
| **--expert-count N** | Total routed expert count; must be paired with **--expert-weight-percent** and divisible by EP |
| **--expert-weight-percent P** | Percentage of base weight bytes belonging to routed experts; must be paired with **--expert-count** |
| **--max-shared-rank-weight-percent P** | Shared bytes from the heaviest stage on the worst TP rank; defaults to 100% and must be in **[100 / TP, 100]** |
| **--max-expert-rank-weight-percent P** | Routed-expert bytes on the worst EP rank; defaults to 100%, must be in **[100 / EP, 100]**, and requires expert metadata |
| **--max-stage-component-weight-percent P** | Independent upper bound for each shared, expert, and overhead class on the heaviest PP stage; defaults to 100% and must be in **[100 / PP, 100]** |
| **--max-stage-layers N** | Transformer layers on the heaviest PP stage; defaults to all layers and must be in **[ceil(layers / PP), layers]** |
| **--max-kv-heads-per-rank N** | Complete KV heads on the worst TP rank; defaults to all heads and must be in **[ceil(KV heads / TP), KV heads]** |

With explicit topology, unspecified degrees default to 1. World size is **TP × PP × DP**; EP overlays existing ranks, must divide **TP × DP**, and does not add GPUs. For new automation prefer **--tp**. A legacy **--gpus 2** invocation remains equivalent to **--tp 2 --pp 1 --dp 1 --ep 1** and records that assumption, while **--gpus 8 --tp 2 --pp 2 --dp 3** is rejected because the topology requires 12 ranks.

Every placement upper bound is independent. Without explicit bounds, the worst rank is charged 100% of shared weights, 100% of routed-expert weights when present, all KV heads, and—for PP—100% of each weight class plus all layers. GPU Watchman never converts TP, PP, or EP counts into optimistic **1 / degree** byte fractions. The stage-component percentage applies separately to shared, expert, and unsharded overhead bytes rather than only their combined total. **--max-stage-layers** and **--max-kv-heads-per-rank** independently bound worst-rank KV geometry; KV heads need not divide TP exactly. Supply a tighter value only when independent deployment/runtime evidence supports it; artifact structure and metadata never do. **--concurrency** remains resident on every DP replica.

**--model-config** derives layers, KV heads, head dimension, and—only when safe—parameter count from a local Hugging Face **config.json**. Explicit **--params**, **--layers**, **--kv-heads**, and **--head-dim** are applied before dependent validation and parameter estimation, so geometry overrides cannot leave a stale audited dense estimate.

A detected MoE config—by expert-count alias, known model type, or nested sparse-routing marker—requires explicit total-resident **--params** plus expert-count and expert-weight-percentage metadata. Config-declared parameter fields are rejected for MoE because they may report active rather than resident weights. A config expert count can supply **--expert-count**; an explicitly supplied count must match it. **--expert-weight-percent** remains required. For non-MoE configs, automatic parameter estimation is audited only for exact **model_type** values **llama** and **mistral** with a standard gated, bias-free layout. Unknown dense families are not auto-estimated but may use an explicit or non-MoE config-declared total; sparse markers always take the stricter MoE path. The file is local-only, regular-file checked, valid JSON, and limited to 8 MiB.

Machine output is a separate **capacity_version: 3** object. It retains the effective **input**, nullable path-free aggregate **artifact** evidence, validated **topology**, logical and worst-rank **weights**, four distinct **kv_cache** views, worst-rank **memory**, **fits**, stable coded **assumptions**, and **caveats**. **weights.base_weight_basis** is **parameter_precision_estimate** or **artifact_residency_floor**. The weight evidence separately records the parameter-derived estimate, nullable adjusted artifact floor, selected base, post-selection overhead, logical shared/expert totals, and worst-rank components. Without **--artifact**, **artifact** and **weights.artifact_adjusted_weight_floor_gib_logical** are null and the parameter basis is used. Weight totals use **\*_gib_logical** names. KV evidence distinguishes logical bytes per sequence, physical bytes per DP replica after the configured head-replication upper bound, one concurrency slot across the deployment, and worst-rank bytes. **topology.kv_heads_per_tensor_parallel_rank_upper_bound** and floating **topology.kv_head_replication_upper_bound** make the selected placement explicit. **input.parameter_source** is **explicit_override**, **config_declared**, or **dense_estimate**; current capacity-v3 estimation rejects unknown provenance. **input.model_type** is a bounded safe config identifier or null. This object is not a schema-v3 hardware report.

Capacity artifact evidence contains aggregate source version, format, layout, shard/tensor/element counts, serialized tensor bytes, descriptor-anchoring availability, multiplier, adjusted logical floor, and selection status. Descriptor anchoring is file-resolution evidence only. The object never serializes the input path, shard or tensor names, dtype details, arbitrary metadata, or payload data. The artifact proves serialized storage only—not loaded GPU/CPU residency, conversion or dequantization expansion, duplicate allocations, TP/PP/DP/EP placement, stage/layer/KV-head placement, or shared/expert composition. It grants no placement credit and does not replace independently deployment/runtime-evidenced placement bounds.

The estimate does not model activation memory, CUDA graph captures, speculative/draft-model allocations, allocator fragmentation, kernels, multimodal encoders, or communication workspaces. Calibrate **--artifact-residency-multiplier**, **--runtime-overhead**, and **--weight-overhead-percent** against the exact loader, runtime, and artifact before using the result as an admission gate.

All counts and precisions must be positive; floating-point inputs must be finite; utilization must be in **(0, 1]**. Invalid usage, multiplier-without-artifact, config/artifact inspection failure, invalid artifact evidence, arithmetic/ceiling failure, or invalid geometry exits 1 without emitting a capacity report. A valid fit exits 0. A valid non-fit exits 2 after emitting the complete estimate; increasing a valid artifact multiplier can change exit 0 to exit 2, never exit 2 to exit 0.

## history

Reads the NDJSON emitted by **--history** or **--format ndjson** and summarizes sample range, hosts, GPU peaks, endpoint availability, queue/KV pressure, runtime throughput, telemetry-source completeness/non-OK counts, unhealthy samples, and finding frequency.

~~~sh
gpu-watchman history reports.ndjson
gpu-watchman history reports.ndjson --format ndjson
~~~

See the [history guide](../operations/history.md) for selection and rotation semantics.

## compare

Compares two JSON reports, or the final non-empty report in each NDJSON file. Comparison version 2 matches GPUs by UUID with index fallback, endpoints by redacted URL, and telemetry sources by canonical name.

~~~sh
gpu-watchman compare baseline.json candidate.json
gpu-watchman compare baseline.json candidate.json \
  --format ndjson --fail-on-regression
~~~

The regression gate exits 2 for a new warning/critical finding, removed GPU/source, telemetry source changing from **ok** to non-OK, previously healthy endpoint going down, or newly configured endpoint that is already down. Inputs are limited to 64 MiB and unsupported future schemas fail closed.

## rollout

Compares saved active-canary JSON or the final non-empty record in each canary NDJSON file:

~~~sh
gpu-watchman rollout BASELINE_CANARY CANDIDATE_CANARY \
  [--max-p95-ttft-regression-percent PERCENT] \
  [--max-p95-e2e-regression-percent PERCENT] \
  [--min-output-tps-ratio RATIO] \
  [--max-success-drop-percent POINTS] \
  [--fail-on-regression] [--format text|json|ndjson]
~~~

Both inputs must use the same current canary version. Model, workload identity,
chat-completions route, stream mode, request count, concurrency, maximum completion
tokens, timeout, response-size limit, and recorded canary policy must match exactly.
Endpoint URL may differ. Both canaries must pass and the candidate timestamp cannot
precede the baseline. Each report is reconstructed from its ordered raw attempts:
summary/token/distribution values, canonical gates, and status must match exactly.
Those checks always run, even when no quantitative gate is selected.

The latency gates compare p95 percentage regression; the output-throughput gate compares the candidate/baseline p50 authoritative per-request rate; and the success gate compares percentage-point drop. Each selected latency or throughput distribution needs at least 20 complete successful samples per report, finite non-negative ordered values, complete measurement coverage, and a non-zero baseline. Missing or invalid evidence fails closed. **--min-p50-output-tps-ratio** and **--max-success-percent-drop** remain visible aliases for the shorter primary flags.

Inputs are bounded to 64 MiB regular UTF-8 files. Output uses **rollout_version: 1**. **--fail-on-regression** exits 2 for an incompatible identity/status or any failed selected gate; without it, the result is still emitted with **regression: true** and exit 0. See the [canary rollout guide](../operations/rollout.md) for formulas and compatibility details.

## bundle

Creates a new 0600 JSON support file with a complete report, doctor checks, versions, and generation time. It refuses to overwrite an existing path.

~~~sh
gpu-watchman bundle --output node-07-support.json \
  --probe http://localhost:8000 \
  --probe-token-file /run/secrets/runtime-metrics-token \
  --no-xid
~~~

The bundle reuses one collection for its report and embedded doctor evidence, so endpoint probes and driver telemetry are not collected twice. It accepts **--allow-insecure-http** for an explicitly trusted non-loopback cleartext target and follows the same no-redirect, no-ambient-proxy policy as every passive probe. Probe credentials and transport options require an active **--probe** target. Use **--no-xid** when kernel-log access is intentionally unavailable. Review process commands and cgroups before sharing a bundle; arbitrary driver-wrapper stderr is classified rather than embedded verbatim.

## completions

~~~sh
gpu-watchman completions bash
gpu-watchman completions zsh
gpu-watchman completions fish
gpu-watchman completions powershell
gpu-watchman completions elvish
~~~

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Command completed; any selected health, canary, capacity, rollout, or runtime-completeness policy passed |
| 1 | Usage, validation, collection, local file, setup, or encoding error prevented the workflow from running |
| 2 | Canary/SLO failure, health threshold crossed, one-shot required source incomplete, runtime fingerprint incomplete without **--allow-incomplete**, doctor failure, capacity non-fit, or selected hardware/canary rollout regression |

SIGINT and SIGTERM end continuous mode cleanly with exit 0. A closed stdout pipe is also treated as a normal consumer exit.
