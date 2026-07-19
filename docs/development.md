# Development and extension guide

GPU Watchman is a library-first Rust project. The binary entrypoint delegates immediately to the application layer; collection, parsing, analysis, and rendering remain independently testable.

## Module map

| Module | Owns | Must not own |
| --- | --- | --- |
| **config** | Strict profile schema, bounded loading, semantic validation, safe rendering | Hardware/network access or secret contents |
| **domain** | Stable report contracts and schema version | I/O, policy, CLI parsing |
| **telemetry** | Vendor commands, parsers, procfs attribution | Rendering or health severity |
| **inference** | Passive HTTP probing, active request protocols, runtime normalization | GPU collection or alert thresholds |
| **analysis** | Findings, recommendations, cross-cycle state | Network/filesystem presentation |
| **operations** | Canary orchestration, doctor, history, bundle, comparison workflows | Vendor-specific parsing |
| **planning** | Offline memory/capacity math | Live allocation or mutation |
| **presentation** | Text/JSON, HTTP, Prometheus | Collection policy |
| **application** | Clap commands, lifecycle, exit codes | Vendor parsing internals |

Dependencies should flow toward **domain**. If two adapters need the same behavior, extract a small domain-neutral helper rather than making them depend on each other.

## Adding a telemetry backend

1. Put the adapter under **src/telemetry/**.
2. Define which command/API is required and which capabilities are optional.
3. Bound every command, null stdin, and drain output without pipe deadlock.
4. Normalize into domain GPU/process records; do not leak backend-only structs into analysis.
5. Add parser fixtures that run without hardware and an end-to-end wrapper test for the exact command protocol.
6. Document permissions, unavailable values, and backend-specific limitations.

## Adding an inference runtime

1. Recognize a stable metric prefix/family, not a product name guessed from the URL.
2. Normalize only operational gauges whose semantics are known: running, waiting, and KV pressure.
3. Keep response bytes, sample count, concurrency, timeout, and processed metric-family bytes bounded; never retain endpoint-controlled family names or labels.
4. Redact URL user information and complete query strings on every error/report path.
5. Add fixtures for exposition quirks, non-finite samples, labels, histograms/counters, and truncation.
6. Add health-rule documentation only when the normalized signal has a deterministic interpretation.

Active request protocols additionally need fragmented-stream fixtures, first-content timing tests, authoritative token-usage handling, bounded worker scheduling, and proof that prompts, generated content, credentials, and untrusted response metadata cannot enter output or errors.

## Adding a health rule

Rules must be deterministic and read-only. Each finding needs a stable code, severity, evidence-rich message, concise recommendation, healthy test, unhealthy test, and a statement about false positives. Rules should account for operating state—for example, PCIe downgrade only matters under meaningful load and a missing fan sensor is normal on passive datacenter GPUs.

## Quality gates

~~~sh
./scripts/check-repository.sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --all-targets
cargo build --locked --release
cargo package --locked --allow-dirty
~~~

The repository hygiene gate checks files and directories for case-insensitive **copy**/**duplicate** names, numbered sync-conflict suffixes such as **README 2.md** and **README (2).md**, editor backup/swap/temp remnants, case-folded path collisions, platform metadata, reintroduced Go sources or module/workspace files, and obvious executables or build artifacts outside **target/**. The maintained **install.sh** installer and shell utilities under **scripts/** are the only expected executable files.

Also validate shell syntax, workflow/manifests, report/Prometheus cardinality, secret redaction, timeouts, optional-capability behavior, and a real NVIDIA node before release.

The crate forbids unsafe Rust and enables Clippy's pedantic set. Do not silence a lint when a small structural improvement expresses the invariant more clearly.

## Release contract

- Rust MSRV is pinned in **rust-toolchain.toml** and **Cargo.toml**.
- Cargo dependency resolution is locked.
- Release builds use thin LTO, one codegen unit, abort-on-panic, and symbol stripping.
- Linux/macOS amd64/arm64 archives use installer-compatible names and SHA-256 sidecars.
- Container and binary workflows must remain reproducible from a version tag.
