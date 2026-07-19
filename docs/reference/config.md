# Operational profiles

GPU Watchman can load one explicit TOML document for repeatable node monitoring, service, and active-canary workflows. Configuration is opt-in: the binary performs no file auto-discovery.

~~~sh
# Either form selects the same file.
gpu-watchman --config /etc/gpu-watchman/config.toml --profile production serve
GPU_WATCHMAN_CONFIG=/etc/gpu-watchman/config.toml \
  GPU_WATCHMAN_PROFILE=production gpu-watchman serve
~~~

**--config PATH** and **GPU_WATCHMAN_CONFIG** are the only ways to load a document. GPU Watchman does not search the working directory, **$HOME**, **/etc**, XDG directories, or platform-specific defaults. **--profile NAME** or **GPU_WATCHMAN_PROFILE** selects a profile; without either, the document's **default_profile** is selected when present, and built-in defaults apply when it is absent. Requesting a profile without a configuration file, naming a missing profile, or naming a default that does not exist is an error.

## Manage configuration

~~~sh
# Create a starter file; defaults to ./gpu-watchman.toml when no path is given.
gpu-watchman config init gpu-watchman.toml

# Parse and semantically validate the complete document.
gpu-watchman --config gpu-watchman.toml config validate

# Inspect normalized settings without reading referenced secret or prompt files.
gpu-watchman --config gpu-watchman.toml config show
gpu-watchman --config gpu-watchman.toml --profile production \
  config show --format json
~~~

**config init** uses create-new semantics and never overwrites an existing file. On Unix it creates the file with mode 0600. A path supplied directly to **config init** takes precedence over the global **--config** destination, and **--profile** is not valid with initialization.

**config validate** rejects malformed TOML, unknown keys, unsupported versions, invalid profile names, missing defaults, invalid URLs, unsafe probe transport or listener exposure, empty path references, unsafe thresholds, contradictory streaming SLOs, and canary plans outside the same request/time budgets enforced at runtime. It validates references but does not contact a GPU, endpoint, or referenced secret file.

**config show** emits normalized TOML, or JSON with **--format json**. When a profile is selected, only that profile is shown. It never opens referenced token, API-key, or prompt files. Configuration URLs containing user information are rejected during loading. For accepted URLs, fragments are removed and the entire query is replaced with **REDACTED** so query names and values cannot leak. File paths remain visible because operators need to verify which references will be used.

## Version-1 schema

Every document must declare the exact schema version:

~~~toml
config_version = 1
default_profile = "production"
~~~

Unknown fields are errors at every level; they are not ignored as possible typos. This release accepts only **config_version = 1**. A document may contain multiple entries below **profiles.NAME**, with profile names beginning with an ASCII letter or digit and containing only ASCII letters, digits, dots, underscores, or hyphens.

The complete section hierarchy is:

~~~text
profiles.NAME.monitor.collection
profiles.NAME.monitor.inference
profiles.NAME.monitor.health
profiles.NAME.service
profiles.NAME.canary
profiles.NAME.canary.slo
~~~

Here is a complete example. Omit fields that should use lower-precedence defaults.

~~~toml
config_version = 1
default_profile = "production"

[profiles.production.monitor.collection]
command_timeout = "3s"
gpus = ["0", "GPU-aaaaaaaa"]
nvidia_smi = "nvidia-smi"
collect_xid = true

[profiles.production.monitor.inference]
urls = ["http://127.0.0.1:8000"]
token_file = "secrets/runtime-metrics-token"
timeout = "3s"
allow_insecure_http = false

[profiles.production.monitor.health]
fail_on = "critical"
required_sources = ["inventory", "processes", "topology"]
vram_warning_percent = 90
vram_critical_percent = 99
temperature_warning_c = 82
temperature_critical_c = 90
kv_cache_warning_percent = 85
kv_cache_critical_percent = 95
process_growth_warning_mib = 256

[profiles.production.service]
listen = "127.0.0.1:9400"
allow_remote_listen = false
interval = "5s"
history_file = "state/history.ndjson"
freshness = "2m"
api_token_file = "secrets/watchman-api-token"
quiet = true

[profiles.production.canary]
base_url = "https://inference.example/v1"
model = "served-model"
api_key_file = "secrets/inference-api-key"
prompt_file = "prompts/synthetic-smoke-test.txt"
workload_id = "synthetic-smoke-v1"
max_tokens = 16
count = 5
concurrency = 2
timeout = "30s"
stream = true
allow_insecure_http = false

[profiles.production.canary.slo]
expect = "gpu-watchman-ok"
max_ttft = "2s"
max_e2e = "10s"
min_output_tokens_per_second = 20
min_success_percent = 100
~~~

### monitor.collection

These fields configure NVIDIA collection for **snapshot**, **top**, and **serve**. **ps** consumes only **gpus**, **nvidia_smi**, and **command_timeout** from this section; it deliberately skips Xid collection. The no-subcommand default snapshot also consumes monitor settings.

| Field | CLI override |
| --- | --- |
| **command_timeout** | **--driver-timeout** |
| **gpus** | **--gpu**; **--all-gpus** clears the list |
| **nvidia_smi** | **--nvidia-smi** or **GPU_WATCHMAN_NVIDIA_SMI** |
| **collect_xid** | **--xid** / **--no-xid** |

### monitor.inference

These fields configure passive runtime probes for **snapshot**, **top**, and **serve**.

| Field | CLI override |
| --- | --- |
| **urls** | **--probe**; **--no-probe** clears the list |
| **token_file** | **--probe-token-file** or direct **--probe-token**; **--no-probe-auth** clears profile and environment credentials |
| **timeout** | **--probe-timeout** |
| **allow_insecure_http** | **--allow-insecure-http** / **--deny-insecure-http** |

The environment credential equivalents are **GPU_WATCHMAN_PROBE_TOKEN** and **GPU_WATCHMAN_PROBE_TOKEN_FILE**. Supplying both forms at the same precedence level is an error.

HTTPS is required for non-loopback probe URLs unless **allow_insecure_http = true** explicitly accepts cleartext transport. Loopback HTTP remains available for node-local runtimes. Passive probes reject URL user information, never follow redirects, and never inherit ambient proxy settings; configure the final metrics URL directly. When a probe token resolves from any precedence layer, every configured URL must share one exact scheme, host, and port or collection fails before network access. Use separate GPU Watchman instances for separately authenticated origins. The cleartext opt-in does not weaken URL, redirect, response-size, timeout, or credential handling.

### monitor.health

These fields configure findings and exit policy for **snapshot**, **top**, and **serve**. **required_sources** accepts **inventory**, **processes**, **optional**, **topology**, and **xid**, or their canonical report names. **--no-require-source** clears the profile list. The other fields map directly to **--fail-on**, **--vram-warning**, **--vram-critical**, **--temperature-warning**, **--temperature-critical**, **--kv-cache-warning**, **--kv-cache-critical**, and **--process-growth-mib**. Process-growth detection applies to continuous **top** and **serve** samples.

The generated local starter uses **fail_on = "never"** so **serve** keeps publishing during an incident. A non-never policy terminates continuous **top** or **serve** when the selected severity appears; use that only when the supervisor restart behavior is intentional. One-shot **snapshot --fail-on** is the safer rollout-gate shape.

### service

This section applies only to **serve**/**exporter**. It never turns **snapshot** or **top** into a server.

| Field | CLI override |
| --- | --- |
| **listen** | **--listen** |
| **allow_remote_listen** | **--allow-remote-listen** / **--deny-remote-listen** |
| **interval** | **--interval** or its **--watch** alias |
| **history_file** | **--history**; **--no-history** clears it |
| **freshness** | **--freshness** |
| **api_token_file** | **--api-token-file** or direct **--api-token**; **--no-api-auth** clears credentials and explicitly permits unauthenticated loopback debugging |
| **quiet** | **--emit text\|ndjson** enables stdout; **--no-emit** disables it |

Serve is quiet when **quiet** is omitted. Setting **quiet = false** emits text; a CLI **--emit ndjson** can choose the machine stream instead. **GPU_WATCHMAN_API_TOKEN** and **GPU_WATCHMAN_API_TOKEN_FILE** override the profile token reference. The **--quiet** spelling remains an alias for **--no-emit**, but new deployments should use the explicit output switch.

Loopback is the default service boundary, but it is not assumed to be a local-user identity boundary: startup still requires a resolved API token unless the operator passes **--no-api-auth** for an explicit debugging exception. A non-loopback **listen** value is valid only when **allow_remote_listen = true**, and runtime startup always requires a resolved API token from the profile, environment, or CLI. The opt-in does not provide TLS; use an authenticated, rate-limiting TLS reverse proxy or service mesh when traffic crosses a trust boundary. **--deny-remote-listen** restores the loopback-only policy even when the profile opts in.

### canary and canary.slo

These sections primarily configure the explicitly invoked **canary** workflow. **canary** sets its request target and bounded sampling plan; **canary.slo** sets correctness, availability, latency, and authoritative output-token-rate gates. **benchmark saturation** reuses **base_url**, **model**, **api_key_file**, **prompt_file** with **workload_id**, **stream**, **allow_insecure_http**, and **slo.expect** only when that profile prompt remains selected. Benchmark stages, sample counts, maximum tokens, timeout, response cap, error/latency/goodput gates, and abort threshold are benchmark-specific and never inherited from the canary profile.

| Field | CLI or environment override |
| --- | --- |
| **base_url** | **--base-url** or **GPU_WATCHMAN_INFERENCE_URL** |
| **model** | **--model** or **GPU_WATCHMAN_INFERENCE_MODEL** |
| **api_key_file** | **--api-key-file**, direct **--api-key**, or the **GPU_WATCHMAN_INFERENCE_API_KEY[_FILE]** pair |
| **prompt_file** | **--prompt-file** or direct **--prompt** |
| **workload_id** | **--workload-id** |
| **max_tokens** | **--max-tokens** |
| **count** | **--count** |
| **concurrency** | **--concurrency** |
| **timeout** | **--timeout** |
| **stream** | **--stream** / **--no-stream** |
| **allow_insecure_http** | **--allow-insecure-http** / **--deny-insecure-http** |
| **slo.expect** | **--expect** / **--no-expect** |
| **slo.max_ttft** | **--max-ttft** / **--no-max-ttft** |
| **slo.max_e2e** | **--max-e2e** / **--no-max-e2e** |
| **slo.min_output_tokens_per_second** | **--min-output-tokens-per-second** or **--min-tps**; **--no-min-tps** clears it |
| **slo.min_success_percent** | **--min-success-percent** |

CLI values override individual profile fields. **prompt_file** and its non-secret **workload_id** must be configured together; a CLI custom prompt requires its own **--workload-id** so it cannot inherit an identity or expectation belonging to different profile content. **--default-prompt** likewise restores the built-in **builtin-v1** identity and **gpu-watchman-ok** expectation unless **--expect** or **--no-expect** is explicit. Inverse and clear controls include **--deny-insecure-http**, **--no-api-key**, **--default-prompt**, **--no-expect**, **--stream**/**--no-stream**, **--no-max-ttft**, **--no-max-e2e**, and **--no-min-tps**. The environment equivalents for target and credentials are **GPU_WATCHMAN_INFERENCE_URL**, **GPU_WATCHMAN_INFERENCE_MODEL**, **GPU_WATCHMAN_INFERENCE_API_KEY**, and **GPU_WATCHMAN_INFERENCE_API_KEY_FILE**.

## Precedence and list behavior

For every profile-aware workflow, effective values are resolved in this order:

~~~text
built-in default < selected profile < environment < command line
~~~

Lists replace lower-precedence lists; they are not merged. For example, **--probe URL** replaces **monitor.inference.urls**, and **--gpu 1** replaces **monitor.collection.gpus**. Use the command-specific clear switch to restore an empty list. Boolean inverse flags explicitly override either true or false profile values.

This precedence is source-aware for credentials: a CLI direct value or CLI file wins over either environment form, and either environment form wins over a profile file reference. A clear flag wins over all three. Conflicting direct and file forms at the same level fail closed.

## Paths, permissions, and secrets

Configuration files are bounded to 1 MiB and must be regular UTF-8 files. On Unix, the selected path itself cannot be a symlink; its owner must be either the invoking user or root; and the file cannot be writable by its group or by other users. After opening, GPU Watchman verifies that the handle still identifies the canonical file.

Every existing canonical ancestor must likewise be owned by the invoking user or root. Group- or world-writable ancestors are rejected unless they use the sticky bit, which permits standard shared directories such as **/tmp** without trusting arbitrary replacement by another user. These checks make the configuration document a policy input rather than merely a convenient settings file. Keep ownership and write access narrow even though secret contents are deliberately stored elsewhere.

Referenced credential files are separately limited to 64 KiB, must be regular UTF-8 files owned by root or the invoking user, and may be group-readable but cannot be group-writable, group-executable, or accessible to other users. Symlinks are allowed for projected-secret mechanisms such as Kubernetes because GPU Watchman validates and reads the already-opened target handle. Credential values must be non-empty visible ASCII without whitespace.

Parser and schema diagnostics report an error category and byte offset but deliberately omit the source excerpt. This prevents an accidentally pasted inline credential or malformed sensitive value from being reflected into CI logs or terminal diagnostics.

Relative paths inside a profile are resolved against the directory containing the configuration file, not the process working directory. This applies to token files, API-key files, prompt files, history files, and an **nvidia_smi** value containing a path separator. A bare command such as **nvidia-smi** remains a PATH lookup. CLI and environment paths keep their normal caller-relative semantics.

The schema has no inline fields for probe tokens, service API tokens, canary API keys, or canary prompts. It accepts only **token_file**, **api_token_file**, **api_key_file**, and **prompt_file** references. **workload_id** is intentionally inline and visible because it must be a non-secret label, never prompt content or a prompt hash. Do not put credentials in URL queries: runtime results redact them, and **config show** redacts the complete query. Treat **canary.slo.expect** as non-sensitive because it is stored and displayed as configuration.

Profiles do not support includes, inheritance, interpolation, shell expansion, or implicit environment substitution. This keeps the selected operational contract finite and auditable.
