use std::io::{ErrorKind, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::builder::styling::{AnsiColor, Styles};
use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::Shell;
use serde::Serialize;

use crate::config::{
    CanaryProfileV1, MonitorHealthProfileV1, ProfileFailOn, ProfileV1, init_config, load_config,
    safe_toml, select_profile, validate_profile_name,
};
use crate::inference::probe::MAX_PROBE_TARGETS;
use crate::operations::{
    bundle, canary, compare, doctor, history, rollout, runtime, saturation, saturation_compare,
};
use crate::planning::capacity::{
    CapacityAssumption, CapacityAssumptionCode, CapacityInput, MAX_ARTIFACT_RESIDENCY_MULTIPLIER,
    estimate, estimate_with_artifact, render_text as render_capacity,
};
use crate::planning::{
    ModelGeometryOverrides, ModelParameterSource, inspect_artifact, load_model_config_evidence,
    model_geometry_from_evidence, render_artifact_text,
};
use crate::presentation::{OutputFormat, render_runtime_fingerprint};
use crate::security::{open_read_nonblocking, reject_permissive_acl};

mod monitor;

const EXIT_FAILURE: u8 = 1;
const EXIT_UNHEALTHY: u8 = 2;
const CLI_STYLES: Styles = Styles::styled()
    .header(AnsiColor::Cyan.on_default().bold())
    .usage(AnsiColor::Cyan.on_default().bold())
    .literal(AnsiColor::Green.on_default().bold())
    .placeholder(AnsiColor::Yellow.on_default())
    .error(AnsiColor::Red.on_default().bold())
    .valid(AnsiColor::Green.on_default())
    .invalid(AnsiColor::Yellow.on_default());

#[derive(Parser)]
#[command(
    name = "gpu-watchman",
    version,
    about = "Inference-aware GPU operations, active validation, observability, and capacity planning",
    long_about = "Read-only NVIDIA GPU diagnostics plus explicitly invoked, bounded inference validation and benchmarking. Run with no subcommand for a snapshot, use top for a live view, or serve for a production metrics API.",
    styles = CLI_STYLES,
    after_help = "QUICK START:\n  watchman snapshot --all --details\n  watchman ps\n  watchman top --probe http://localhost:8000\n  watchman runtime inspect --pid 4242\n  watchman canary --model meta-llama/Llama-3.1-8B-Instruct\n  watchman benchmark saturation --model served-model --concurrency-stages 1,2,4,8\n  watchman benchmark compare baseline-benchmark.json candidate-benchmark.json --min-successful-rps-ratio 0.95 --fail-on-regression\n  watchman rollout baseline-canary.json candidate-canary.json --fail-on-regression\n  watchman artifact inspect /models/llama\n  watchman serve --listen 127.0.0.1:9400 --no-api-auth\n  watchman config init gpu-watchman.toml\n  watchman capacity --model-config config.json --artifact /models/llama --gpu-vram 80 --tp 2 --gpus 2 --weight-bits 4"
)]
struct Cli {
    /// Load a profile for snapshot/top/serve/ps/canary/benchmark saturation/config; rejected elsewhere.
    #[arg(
        long,
        global = true,
        env = "GPU_WATCHMAN_CONFIG",
        hide_env_values = true,
        value_name = "PATH"
    )]
    config: Option<PathBuf>,

    /// Select a profile for a profile-aware command; rejected by other workflows.
    #[arg(
        long,
        global = true,
        env = "GPU_WATCHMAN_PROFILE",
        hide_env_values = true,
        value_name = "NAME",
        value_parser = parse_profile_name
    )]
    profile: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Capture one complete point-in-time report.
    Snapshot(SnapshotArgs),

    /// Open a live, refreshing GPU and inference-runtime control-room view.
    #[command(visible_alias = "watch")]
    Top(TopArgs),

    /// Run the Prometheus, health, and report service with continuous collection.
    #[command(visible_alias = "exporter")]
    Serve(ServeArgs),

    /// List GPU processes with VRAM, owner, container, pod, and command identity.
    Ps(PsArgs),

    /// Validate driver access, telemetry, process attribution, and inference endpoints.
    Doctor(DoctorArgs),

    /// Inspect checkpoint metadata, bytes, dtypes, and shard completeness without loading tensors.
    Artifact(ArtifactArgs),

    /// Inspect bounded, privacy-safe inference runtime evidence for explicit local PIDs.
    Runtime(RuntimeArgs),

    /// Estimate model weights, KV cache, headroom, and full-context concurrency.
    Capacity(CapacityArgs),

    /// Exercise an OpenAI-compatible chat endpoint and enforce latency and success SLOs.
    Canary(CanaryArgs),

    /// Run bounded, evidence-first inference benchmarks and deployment gates.
    Benchmark(BenchmarkArgs),

    /// Compare saved baseline/candidate canaries under explicit rollout gates.
    Rollout(RolloutArgs),

    /// Summarize peaks, availability, and recurring findings in NDJSON history.
    History {
        /// NDJSON file written by --history.
        path: PathBuf,

        #[arg(long, value_enum, default_value_t = CliFormat::Text)]
        format: CliFormat,
    },

    /// Compare JSON/NDJSON reports and optionally fail on operational regressions.
    Compare {
        /// Baseline JSON report or NDJSON history file.
        baseline: PathBuf,

        /// Current JSON report or NDJSON history file.
        current: PathBuf,

        /// Exit 2 for new warning/critical findings, removed GPUs, or endpoint outages.
        #[arg(long)]
        fail_on_regression: bool,

        #[arg(long, value_enum, default_value_t = CliFormat::Text)]
        format: CliFormat,
    },

    /// Collect a portable JSON support bundle for incident handoff.
    Bundle(BundleArgs),

    /// Create, validate, or safely inspect operational profiles.
    Config(ConfigArgs),

    /// Generate shell completion definitions.
    Completions {
        #[arg(value_enum)]
        shell: Shell,
    },

    /// Print only the version (also available as --version).
    Version,
}

#[allow(clippy::struct_excessive_bools)]
#[derive(clap::Args)]
struct DoctorArgs {
    /// Inference base or metrics URLs to validate.
    #[arg(long, value_delimiter = ',', value_parser = parse_probe_target)]
    probe: Vec<String>,

    /// Bearer token sent to inference endpoints; never included in output.
    #[arg(
        long,
        value_name = "TOKEN",
        conflicts_with_all = ["probe_token_file", "no_probe_auth"]
    )]
    probe_token: Option<String>,

    /// Read the inference bearer token from a file.
    #[arg(
        long,
        value_name = "PATH",
        conflicts_with_all = ["probe_token", "no_probe_auth"]
    )]
    probe_token_file: Option<PathBuf>,

    /// Disable probe credentials supplied by the environment.
    #[arg(long, conflicts_with_all = ["probe_token", "probe_token_file"])]
    no_probe_auth: bool,

    /// Permit cleartext HTTP to non-loopback inference metrics endpoints.
    #[arg(long)]
    allow_insecure_http: bool,

    /// Emit human-readable or JSON output.
    #[arg(long, value_enum, default_value_t = CliFormat::Text)]
    format: CliFormat,

    /// Deadline for each driver command and endpoint request.
    #[arg(long, default_value = "3s", value_parser = parse_nonzero_duration)]
    timeout: Duration,

    /// nvidia-smi binary or wrapper path.
    #[arg(
        long,
        env = "GPU_WATCHMAN_NVIDIA_SMI",
        hide_env_values = true,
        default_value = "nvidia-smi"
    )]
    nvidia_smi: PathBuf,
}

#[allow(clippy::struct_excessive_bools)]
#[derive(clap::Args)]
struct BundleArgs {
    /// Destination JSON file (created with mode 0600 on Unix).
    #[arg(long, default_value = "gpu-watchman-support.json")]
    output: PathBuf,

    /// Inference base or metrics URLs to include.
    #[arg(long, value_delimiter = ',', value_parser = parse_probe_target)]
    probe: Vec<String>,

    /// Bearer token sent to inference endpoints; never included in the bundle.
    #[arg(
        long,
        value_name = "TOKEN",
        conflicts_with_all = ["probe_token_file", "no_probe_auth"]
    )]
    probe_token: Option<String>,

    /// Read the inference bearer token from a file.
    #[arg(
        long,
        value_name = "PATH",
        conflicts_with_all = ["probe_token", "no_probe_auth"]
    )]
    probe_token_file: Option<PathBuf>,

    /// Disable probe credentials supplied by the environment.
    #[arg(long, conflicts_with_all = ["probe_token", "probe_token_file"])]
    no_probe_auth: bool,

    /// Permit cleartext HTTP to non-loopback inference metrics endpoints.
    #[arg(long)]
    allow_insecure_http: bool,

    /// Deadline for each driver command and endpoint request.
    #[arg(long, default_value = "3s", value_parser = parse_nonzero_duration)]
    timeout: Duration,

    /// nvidia-smi binary or wrapper path.
    #[arg(
        long,
        env = "GPU_WATCHMAN_NVIDIA_SMI",
        hide_env_values = true,
        default_value = "nvidia-smi"
    )]
    nvidia_smi: PathBuf,

    /// Skip kernel-log Xid collection in the bundle.
    #[arg(long)]
    no_xid: bool,
}

#[derive(Debug, Clone, Default, clap::Args)]
struct CollectionCliArgs {
    /// Restrict collection to physical GPU indexes or UUIDs.
    #[arg(
        long,
        value_name = "INDEX|UUID,...",
        value_delimiter = ',',
        help_heading = "Collection"
    )]
    gpu: Vec<String>,

    /// Clear a GPU filter supplied by the selected profile and inspect every GPU.
    #[arg(long, conflicts_with = "gpu", help_heading = "Collection")]
    all_gpus: bool,

    /// nvidia-smi binary or wrapper. Defaults to nvidia-smi on PATH.
    #[arg(
        long,
        value_name = "PATH",
        env = "GPU_WATCHMAN_NVIDIA_SMI",
        hide_env_values = true,
        help_heading = "Collection"
    )]
    nvidia_smi: Option<PathBuf>,

    /// Deadline for each driver or kernel-log command. Defaults to 3s.
    #[arg(
        long,
        visible_aliases = ["command-timeout", "timeout"],
        value_name = "DURATION",
        value_parser = parse_nonzero_duration,
        help_heading = "Collection"
    )]
    driver_timeout: Option<Duration>,

    /// Skip kernel-log Xid collection.
    #[arg(long, conflicts_with = "xid", help_heading = "Collection")]
    no_xid: bool,

    /// Force kernel-log Xid collection when a profile disables it.
    #[arg(long, conflicts_with = "no_xid", help_heading = "Collection")]
    xid: bool,
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Default, clap::Args)]
struct ProbeCliArgs {
    /// Inference base or metrics URLs for supported inference runtimes.
    #[arg(
        long,
        value_name = "URL,...",
        value_delimiter = ',',
        value_parser = parse_probe_target,
        conflicts_with = "no_probe",
        help_heading = "Inference"
    )]
    probe: Vec<String>,

    /// Clear inference endpoints supplied by the selected profile.
    #[arg(
        long,
        conflicts_with_all = [
            "probe",
            "probe_token",
            "probe_token_file",
            "no_probe_auth",
            "probe_timeout",
            "allow_insecure_probe_http",
            "deny_insecure_probe_http"
        ],
        help_heading = "Inference"
    )]
    no_probe: bool,

    /// Bearer token sent to inference metrics endpoints; prefer the file form.
    #[arg(
        long,
        value_name = "TOKEN",
        conflicts_with_all = ["probe_token_file", "no_probe_auth"],
        help_heading = "Inference"
    )]
    probe_token: Option<String>,

    /// Read the inference metrics bearer token from a file.
    #[arg(
        long,
        value_name = "PATH",
        conflicts_with_all = ["probe_token", "no_probe_auth"],
        help_heading = "Inference"
    )]
    probe_token_file: Option<PathBuf>,

    /// Disable profile or environment authentication for inference probes.
    #[arg(
        long,
        conflicts_with_all = ["probe_token", "probe_token_file"],
        help_heading = "Inference"
    )]
    no_probe_auth: bool,

    /// Inference endpoint request timeout. Defaults to 3s.
    #[arg(
        long,
        value_name = "DURATION",
        value_parser = parse_nonzero_duration,
        help_heading = "Inference"
    )]
    probe_timeout: Option<Duration>,

    /// Permit cleartext HTTP to non-loopback inference metrics endpoints.
    #[arg(
        long = "allow-insecure-http",
        conflicts_with_all = ["deny_insecure_probe_http", "no_probe"],
        help_heading = "Inference"
    )]
    allow_insecure_probe_http: bool,

    /// Override a profile and reject non-loopback cleartext inference probes.
    #[arg(
        long = "deny-insecure-http",
        conflicts_with_all = ["allow_insecure_probe_http", "no_probe"],
        help_heading = "Inference"
    )]
    deny_insecure_probe_http: bool,
}

#[derive(Debug, Clone, Default, clap::Args)]
struct HealthCliArgs {
    /// Exit 2 when this severity is present. Defaults to never.
    #[arg(long, value_enum, help_heading = "Health policy")]
    fail_on: Option<FailOn>,

    /// Require complete telemetry sources.
    #[arg(
        long,
        value_name = "SOURCE,...",
        value_delimiter = ',',
        value_parser = parse_required_source,
        conflicts_with = "no_require_source",
        help_heading = "Health policy"
    )]
    require_source: Vec<String>,

    /// Clear required telemetry sources supplied by the selected profile.
    #[arg(
        long,
        conflicts_with = "require_source",
        help_heading = "Health policy"
    )]
    no_require_source: bool,

    /// VRAM percentage that creates a warning. Defaults to 90.
    #[arg(
        long,
        value_name = "PERCENT",
        value_parser = parse_percentage_i64,
        help_heading = "Health policy"
    )]
    vram_warning: Option<i64>,

    /// VRAM percentage that creates a critical finding. Defaults to 99.
    #[arg(
        long,
        value_name = "PERCENT",
        value_parser = parse_percentage_i64,
        help_heading = "Health policy"
    )]
    vram_critical: Option<i64>,

    /// GPU temperature in Celsius that creates a warning. Defaults to 82.
    #[arg(long, value_name = "CELSIUS", help_heading = "Health policy")]
    temperature_warning: Option<i32>,

    /// GPU temperature in Celsius that creates a critical finding. Defaults to 90.
    #[arg(long, value_name = "CELSIUS", help_heading = "Health policy")]
    temperature_critical: Option<i32>,

    /// Inference KV-cache usage percentage that creates a warning. Defaults to 85.
    #[arg(
        long,
        value_name = "PERCENT",
        value_parser = parse_percentage_f64,
        help_heading = "Health policy"
    )]
    kv_cache_warning: Option<f64>,

    /// Inference KV-cache usage percentage that creates a critical finding. Defaults to 95.
    #[arg(
        long,
        value_name = "PERCENT",
        value_parser = parse_percentage_f64,
        help_heading = "Health policy"
    )]
    kv_cache_critical: Option<f64>,
}

#[derive(Debug, Clone, Default, clap::Args)]
struct ContinuousHealthCliArgs {
    #[command(flatten)]
    health: HealthCliArgs,

    /// Per-process VRAM growth in MiB that creates a warning. Defaults to 256.
    #[arg(
        long,
        value_name = "MIB",
        value_parser = parse_positive_i64,
        help_heading = "Health policy"
    )]
    process_growth_mib: Option<i64>,
}

#[derive(Default, clap::Args)]
struct SnapshotArgs {
    #[command(flatten)]
    collection: CollectionCliArgs,

    #[command(flatten)]
    probes: ProbeCliArgs,

    #[command(flatten)]
    health: HealthCliArgs,

    /// Output format for this snapshot.
    #[arg(long, value_enum, default_value_t = CliFormat::Text, help_heading = "Output")]
    format: CliFormat,

    /// Color policy for human-readable output.
    #[arg(long, value_enum, default_value_t = ColorChoice::Auto, help_heading = "Output")]
    color: ColorChoice,

    /// Include healthy idle GPUs in text output.
    #[arg(long, help_heading = "Output")]
    all: bool,

    /// Include clocks, PCI Express, topology, Xid, sources, and informational findings.
    #[arg(long, help_heading = "Output")]
    details: bool,

    /// Append the snapshot to this private NDJSON history file.
    #[arg(long, value_name = "PATH", help_heading = "Output")]
    history: Option<PathBuf>,

    /// Write only the history record; requires --history.
    #[arg(
        long,
        requires = "history",
        conflicts_with_all = ["all", "details"],
        help_heading = "Output"
    )]
    quiet: bool,
}

#[derive(Default, clap::Args)]
struct TopArgs {
    /// Refresh interval, at least 500ms. Defaults to 2s.
    #[arg(
        long,
        visible_alias = "watch",
        value_name = "DURATION",
        value_parser = parse_nonzero_duration,
        help_heading = "Runtime"
    )]
    interval: Option<Duration>,

    /// Append complete samples to this private NDJSON history file.
    #[arg(long, value_name = "PATH", help_heading = "Runtime")]
    history: Option<PathBuf>,

    /// Append samples instead of redrawing an interactive terminal.
    #[arg(long, help_heading = "Runtime")]
    no_clear: bool,

    #[command(flatten)]
    collection: CollectionCliArgs,

    #[command(flatten)]
    probes: ProbeCliArgs,

    #[command(flatten)]
    health: ContinuousHealthCliArgs,

    /// Output mode for the live stream.
    #[arg(long, value_enum, default_value_t = LiveFormat::Text, help_heading = "Output")]
    format: LiveFormat,

    /// Color policy for human-readable output.
    #[arg(long, value_enum, default_value_t = ColorChoice::Auto, help_heading = "Output")]
    color: ColorChoice,

    /// Include detailed hardware, topology, source, and finding evidence.
    #[arg(long, help_heading = "Output")]
    details: bool,
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Default, clap::Args)]
struct ServeArgs {
    /// HTTP listen address. Defaults to 127.0.0.1:9400.
    #[arg(long, value_name = "ADDR", help_heading = "Service")]
    listen: Option<String>,

    /// Permit a non-loopback listener; HTTP API authentication is also required.
    #[arg(long, conflicts_with = "deny_remote_listen", help_heading = "Service")]
    allow_remote_listen: bool,

    /// Override a profile and require a loopback listener.
    #[arg(long, conflicts_with = "allow_remote_listen", help_heading = "Service")]
    deny_remote_listen: bool,

    /// Collection interval, at least 500ms. Defaults to 5s.
    #[arg(
        long,
        visible_alias = "watch",
        value_name = "DURATION",
        value_parser = parse_nonzero_duration,
        help_heading = "Service"
    )]
    interval: Option<Duration>,

    /// Append complete samples to this private NDJSON history file.
    #[arg(
        long,
        value_name = "PATH",
        conflicts_with = "no_history",
        help_heading = "Service"
    )]
    history: Option<PathBuf>,

    /// Disable history configured by the selected profile.
    #[arg(long, conflicts_with = "history", help_heading = "Service")]
    no_history: bool,

    /// Report health becomes stale after this duration. Defaults to 2m.
    #[arg(
        long,
        value_name = "DURATION",
        value_parser = parse_nonzero_duration,
        help_heading = "Service"
    )]
    freshness: Option<Duration>,

    /// Require this bearer token on metrics, discovery, and report routes.
    #[arg(
        long,
        value_name = "TOKEN",
        conflicts_with_all = ["api_token_file", "no_api_auth"],
        help_heading = "Service"
    )]
    api_token: Option<String>,

    /// Read the HTTP API bearer token from a file.
    #[arg(
        long,
        value_name = "PATH",
        conflicts_with_all = ["api_token", "no_api_auth"],
        help_heading = "Service"
    )]
    api_token_file: Option<PathBuf>,

    /// Explicitly allow an unauthenticated loopback API and clear inherited credentials.
    #[arg(
        long,
        conflicts_with_all = ["api_token", "api_token_file"],
        help_heading = "Service"
    )]
    no_api_auth: bool,

    /// Also stream reports to stdout; serve is quiet by default.
    #[arg(long, value_enum, value_name = "FORMAT", help_heading = "Service")]
    emit: Option<LiveFormat>,

    /// Disable stdout streaming supplied by the selected profile.
    #[arg(
        long = "no-emit",
        visible_alias = "quiet",
        conflicts_with = "emit",
        help_heading = "Service"
    )]
    no_emit: bool,

    #[command(flatten)]
    collection: CollectionCliArgs,

    #[command(flatten)]
    probes: ProbeCliArgs,

    #[command(flatten)]
    health: ContinuousHealthCliArgs,
}

#[derive(Debug, Default, clap::Args)]
struct PsArgs {
    /// Restrict the process list to physical GPU indexes or UUIDs.
    #[arg(
        long,
        value_name = "INDEX|UUID,...",
        value_delimiter = ',',
        help_heading = "Collection"
    )]
    gpu: Vec<String>,

    /// Clear a GPU filter supplied by the selected profile and inspect every GPU.
    #[arg(long, conflicts_with = "gpu", help_heading = "Collection")]
    all_gpus: bool,

    /// nvidia-smi binary or wrapper. Defaults to nvidia-smi on PATH.
    #[arg(
        long,
        value_name = "PATH",
        env = "GPU_WATCHMAN_NVIDIA_SMI",
        hide_env_values = true,
        help_heading = "Collection"
    )]
    nvidia_smi: Option<PathBuf>,

    /// Deadline for each driver command. Defaults to 3s.
    #[arg(
        long,
        visible_aliases = ["command-timeout", "timeout"],
        value_name = "DURATION",
        value_parser = parse_nonzero_duration,
        help_heading = "Collection"
    )]
    driver_timeout: Option<Duration>,

    /// Emit a best-effort result and exit zero when GPU process telemetry is incomplete.
    #[arg(long, help_heading = "Policy")]
    allow_incomplete: bool,

    /// Process-list output format.
    #[arg(long, value_enum, default_value_t = CliFormat::Text, help_heading = "Output")]
    format: CliFormat,

    /// Color policy for human-readable output.
    #[arg(long, value_enum, default_value_t = ColorChoice::Auto, help_heading = "Output")]
    color: ColorChoice,
}

#[derive(Debug, clap::Args)]
struct ConfigArgs {
    #[command(subcommand)]
    command: ConfigCommands,
}

#[derive(Debug, Subcommand)]
enum ConfigCommands {
    /// Create a private starter profile without overwriting an existing file.
    Init {
        /// Destination path; defaults to --config or gpu-watchman.toml.
        #[arg(value_name = "PATH")]
        path: Option<PathBuf>,
    },

    /// Parse and semantically validate the complete configuration document.
    Validate,

    /// Print a redacted, normalized configuration without reading referenced secrets.
    Show {
        #[arg(long, value_enum, default_value_t = ConfigOutputFormat::Toml)]
        format: ConfigOutputFormat,
    },
}

#[derive(Debug, clap::Args)]
struct ArtifactArgs {
    #[command(subcommand)]
    command: ArtifactCommands,
}

#[derive(Debug, clap::Args)]
struct RuntimeArgs {
    #[command(subcommand)]
    command: RuntimeCommands,
}

#[derive(Debug, Subcommand)]
enum RuntimeCommands {
    /// Inspect engine, launch, driver, framework, and mapped-library evidence.
    Inspect {
        /// Explicit local process ID to inspect; repeat for up to 32 targets.
        #[arg(long, required = true, value_name = "PID", value_parser = parse_positive_u32)]
        pid: Vec<u32>,

        /// Exit successfully even when one or more bounded evidence sources are incomplete.
        #[arg(long)]
        allow_incomplete: bool,

        /// Emit human-readable, JSON, or compact NDJSON Runtime Fingerprint v1.
        #[arg(long, value_enum, default_value_t = CliFormat::Text)]
        format: CliFormat,
    },
}

#[derive(Debug, Subcommand)]
enum ArtifactCommands {
    /// Validate a safetensors file, sharded index, or unambiguous directory.
    Inspect {
        /// Local .safetensors file, .safetensors.index.json, or model directory.
        #[arg(value_name = "PATH")]
        path: PathBuf,

        /// Emit human-readable, JSON, or compact NDJSON Artifact Report v1.
        #[arg(long, value_enum, default_value_t = CliFormat::Text)]
        format: CliFormat,
    },
}

#[derive(Debug, clap::Args)]
struct CapacityArgs {
    /// Local Hugging Face config.json used to derive model geometry and parameter count.
    #[arg(long, value_name = "PATH")]
    model_config: Option<PathBuf>,

    /// Safetensors file, index, or directory whose verified serialized tensor
    /// bytes strengthen the analytical weight estimate without granting
    /// placement credit.
    #[arg(long, value_name = "PATH")]
    artifact: Option<PathBuf>,

    /// Expansion factor applied to serialized artifact tensor bytes before
    /// using them as a lower residency floor. Defaults to 1 when --artifact is
    /// supplied and must be calibrated for the deployed loader/runtime.
    #[arg(
        long,
        requires = "artifact",
        value_parser = parse_artifact_residency_multiplier
    )]
    artifact_residency_multiplier: Option<f64>,

    /// Total resident model parameter count in billions; for `MoE`, never use
    /// active-per-token parameters.
    #[arg(
        long,
        value_parser = parse_positive_f64,
        required_unless_present = "model_config"
    )]
    params: Option<f64>,

    /// Weight precision in bits (for example 16, 8, or 4).
    #[arg(long, default_value_t = 16.0, value_parser = parse_positive_f64)]
    weight_bits: f64,

    /// Legacy placement count; without topology flags this is tensor parallelism.
    /// With topology flags it is an optional expected TP x PP x DP world size.
    #[arg(long, value_parser = parse_positive_u32, help_heading = "Topology")]
    gpus: Option<u32>,

    /// Tensor-parallel degree. Defaults to 1 with explicit topology.
    #[arg(long, value_parser = parse_positive_u32, help_heading = "Topology")]
    tp: Option<u32>,

    /// Pipeline-parallel degree. Defaults to 1.
    #[arg(long, value_parser = parse_positive_u32, help_heading = "Topology")]
    pp: Option<u32>,

    /// Data-parallel replica count. Defaults to 1 and never reduces per-rank weights.
    #[arg(long, value_parser = parse_positive_u32, help_heading = "Topology")]
    dp: Option<u32>,

    /// Expert-parallel degree within TP x DP ranks. Defaults to 1.
    #[arg(long, value_parser = parse_positive_u32, help_heading = "Topology")]
    ep: Option<u32>,

    /// Routed expert count; required for detected `MoE` and loaded from config
    /// when a trusted expert-count alias is present.
    #[arg(long, value_parser = parse_positive_u32, help_heading = "Topology")]
    expert_count: Option<u32>,

    /// Percentage of total resident base-model weights belonging to routed
    /// experts; required for detected `MoE`.
    #[arg(long, value_parser = parse_percentage_f64, help_heading = "Topology")]
    expert_weight_percent: Option<f64>,

    /// Deployment/runtime-evidenced upper bound for routed-expert bytes assigned
    /// to one EP rank; omission charges 100%.
    #[arg(long, value_parser = parse_percentage_f64, help_heading = "Topology")]
    max_expert_rank_weight_percent: Option<f64>,

    /// Deployment/runtime-evidenced upper bound that independently holds for
    /// shared, expert, and overhead bytes on the heaviest pipeline stage;
    /// omission charges 100% of each.
    #[arg(long, value_parser = parse_percentage_f64, help_heading = "Topology")]
    max_stage_component_weight_percent: Option<f64>,

    /// Runtime-evidenced upper bound for transformer layers on the heaviest
    /// pipeline stage; omission charges all layers.
    #[arg(long, value_parser = parse_positive_u32, help_heading = "Topology")]
    max_stage_layers: Option<u32>,

    /// Physical VRAM in GiB on the smallest rank in the placement.
    #[arg(long, value_parser = parse_positive_f64)]
    gpu_vram: f64,

    /// Fraction of physical VRAM the runtime may use.
    #[arg(long, default_value_t = 0.9, value_parser = parse_utilization)]
    utilization: f64,

    /// Transformer layer count.
    #[arg(
        long,
        value_parser = parse_positive_u32,
        required_unless_present = "model_config"
    )]
    layers: Option<u32>,

    /// KV head count after grouped/multi-query attention.
    #[arg(
        long,
        value_parser = parse_positive_u32,
        required_unless_present = "model_config"
    )]
    kv_heads: Option<u32>,

    /// Runtime-evidenced upper bound for complete KV heads retained on one TP
    /// rank; omission charges all KV heads.
    #[arg(long, value_parser = parse_positive_u32, help_heading = "Topology")]
    max_kv_heads_per_rank: Option<u32>,

    /// Attention head dimension.
    #[arg(
        long,
        value_parser = parse_positive_u32,
        required_unless_present = "model_config"
    )]
    head_dim: Option<u32>,

    /// Maximum tokens retained per sequence.
    #[arg(long, default_value_t = 8192, value_parser = parse_positive_u64)]
    context: u64,

    /// Concurrent full-context sequences per data-parallel replica.
    #[arg(long, default_value_t = 1, value_parser = parse_positive_u32)]
    concurrency: u32,

    /// KV-cache element precision in bits.
    #[arg(long, default_value_t = 16.0, value_parser = parse_positive_f64)]
    kv_bits: f64,

    /// Per-GPU runtime/workspace overhead in GiB.
    #[arg(long, default_value_t = 4.0, value_parser = parse_nonnegative_f64)]
    runtime_overhead: f64,

    /// Weight metadata/dequantization overhead.
    #[arg(long, default_value_t = 5.0, value_parser = parse_nonnegative_f64)]
    weight_overhead_percent: f64,

    /// Deployment/runtime-evidenced upper bound for shared base-weight bytes on
    /// one TP rank; omission charges 100%.
    #[arg(long, value_parser = parse_percentage_f64, help_heading = "Topology")]
    max_shared_rank_weight_percent: Option<f64>,

    /// Emit human-readable, JSON, or compact NDJSON output.
    #[arg(long, value_enum, default_value_t = CliFormat::Text)]
    format: CliFormat,
}

#[allow(clippy::struct_excessive_bools)]
#[derive(clap::Args)]
struct CanaryArgs {
    /// OpenAI-compatible API base URL. Defaults to the local port 8000 v1 endpoint.
    #[arg(
        long,
        env = "GPU_WATCHMAN_INFERENCE_URL",
        hide_env_values = true,
        value_name = "URL"
    )]
    base_url: Option<String>,

    /// Permit cleartext HTTP to a non-loopback inference endpoint.
    #[arg(long, conflicts_with = "deny_insecure_http")]
    allow_insecure_http: bool,

    /// Override a profile and reject cleartext HTTP to non-loopback endpoints.
    #[arg(long, conflicts_with = "allow_insecure_http")]
    deny_insecure_http: bool,

    /// Model identifier sent to the chat completions endpoint.
    #[arg(
        long,
        env = "GPU_WATCHMAN_INFERENCE_MODEL",
        hide_env_values = true,
        value_name = "MODEL",
        value_parser = parse_nonempty_string
    )]
    model: Option<String>,

    /// API key sent as a bearer credential; prefer --api-key-file to avoid argv exposure.
    #[arg(
        long,
        value_name = "KEY",
        conflicts_with_all = ["api_key_file", "no_api_key"]
    )]
    api_key: Option<String>,

    /// Read the inference API key from a file.
    #[arg(
        long,
        value_name = "PATH",
        conflicts_with_all = ["api_key", "no_api_key"]
    )]
    api_key_file: Option<PathBuf>,

    /// Disable an API key supplied by the selected profile or environment.
    #[arg(long, conflicts_with_all = ["api_key", "api_key_file"])]
    no_api_key: bool,

    /// Prompt to send; visible in argv. Defaults to a deterministic health-check prompt.
    #[arg(
        long,
        value_name = "TEXT",
        value_parser = parse_nonempty_string,
        conflicts_with_all = ["prompt_file", "default_prompt"]
    )]
    prompt: Option<String>,

    /// Read the prompt from a UTF-8 file instead of the command line.
    #[arg(
        long,
        value_name = "PATH",
        conflicts_with_all = ["prompt", "default_prompt"]
    )]
    prompt_file: Option<PathBuf>,

    /// Ignore a profile prompt file and use the built-in deterministic prompt.
    #[arg(long, conflicts_with_all = ["prompt", "prompt_file"])]
    default_prompt: bool,

    /// Non-secret identity for a custom synthetic workload; required with a custom prompt.
    #[arg(
        long,
        value_name = "ID",
        value_parser = parse_workload_id,
        conflicts_with = "default_prompt"
    )]
    workload_id: Option<String>,

    /// Require generated text to contain this value; visible in argv.
    #[arg(
        long,
        value_name = "TEXT",
        value_parser = parse_nonempty_string,
        conflicts_with = "no_expect"
    )]
    expect: Option<String>,

    /// Disable an expected-text gate supplied by the selected profile.
    #[arg(long, conflicts_with = "expect")]
    no_expect: bool,

    /// Maximum completion tokens requested per attempt (maximum 65,536).
    #[arg(long, value_parser = parse_canary_max_tokens)]
    max_tokens: Option<u32>,

    /// Total inference requests to execute (maximum 10,000).
    #[arg(long, value_parser = parse_canary_count)]
    count: Option<u32>,

    /// Maximum simultaneous requests (maximum 64 and never above --count).
    #[arg(long, value_parser = parse_canary_concurrency)]
    concurrency: Option<u32>,

    /// Deadline for each inference request (maximum 5 minutes).
    #[arg(long, value_parser = parse_nonzero_duration)]
    timeout: Option<Duration>,

    /// Force server-sent event streaming when a profile disables it.
    #[arg(long, conflicts_with = "no_stream")]
    stream: bool,

    /// Disable server-sent event streaming.
    #[arg(
        long,
        conflicts_with_all = ["stream", "max_ttft", "min_output_tokens_per_second"]
    )]
    no_stream: bool,

    /// Fail when any successful request exceeds this time to first token.
    #[arg(
        long,
        value_name = "DURATION",
        value_parser = parse_nonzero_duration,
        conflicts_with = "no_max_ttft"
    )]
    max_ttft: Option<Duration>,

    /// Disable a maximum TTFT gate supplied by the selected profile.
    #[arg(long, conflicts_with = "max_ttft")]
    no_max_ttft: bool,

    /// Fail when any successful request exceeds this end-to-end latency.
    #[arg(
        long,
        value_name = "DURATION",
        value_parser = parse_nonzero_duration,
        conflicts_with = "no_max_e2e"
    )]
    max_e2e: Option<Duration>,

    /// Disable a maximum end-to-end latency gate supplied by the selected profile.
    #[arg(long, conflicts_with = "max_e2e")]
    no_max_e2e: bool,

    /// Fail when any authoritative per-request output rate is below this value.
    #[arg(
        long,
        visible_alias = "min-tps",
        value_name = "TOKENS/SECOND",
        value_parser = parse_nonnegative_f64,
        conflicts_with = "no_min_output_tokens_per_second"
    )]
    min_output_tokens_per_second: Option<f64>,

    /// Disable a minimum output-token rate gate supplied by the selected profile.
    #[arg(long = "no-min-tps", conflicts_with = "min_output_tokens_per_second")]
    no_min_output_tokens_per_second: bool,

    /// Minimum percentage of attempts that must succeed; at least one must pass.
    #[arg(
        long,
        value_parser = parse_percentage_f64
    )]
    min_success_percent: Option<f64>,

    /// Emit human-readable, JSON, or compact NDJSON output.
    #[arg(long, value_enum, default_value_t = CliFormat::Text)]
    format: CliFormat,
}

#[derive(clap::Args)]
struct BenchmarkArgs {
    #[command(subcommand)]
    command: BenchmarkCommands,
}

#[derive(Subcommand)]
enum BenchmarkCommands {
    /// Measure a closed-loop concurrency ladder and optionally verify one tested stage.
    Saturation(Box<SaturationArgs>),

    /// Compare two saved, exact-ladder saturation reports under fail-closed gates.
    Compare(SaturationCompareArgs),
}

#[allow(clippy::struct_excessive_bools)]
#[derive(clap::Args)]
struct SaturationArgs {
    /// OpenAI-compatible API base URL. Defaults to the local port 8000 v1 endpoint.
    #[arg(
        long,
        env = "GPU_WATCHMAN_INFERENCE_URL",
        hide_env_values = true,
        value_name = "URL"
    )]
    base_url: Option<String>,

    /// Permit cleartext HTTP to a non-loopback inference endpoint.
    #[arg(long, conflicts_with = "deny_insecure_http")]
    allow_insecure_http: bool,

    /// Override a profile and reject cleartext HTTP to non-loopback endpoints.
    #[arg(long, conflicts_with = "allow_insecure_http")]
    deny_insecure_http: bool,

    /// Model identifier sent to the chat-completions endpoint.
    #[arg(
        long,
        env = "GPU_WATCHMAN_INFERENCE_MODEL",
        hide_env_values = true,
        value_name = "MODEL",
        value_parser = parse_nonempty_string
    )]
    model: Option<String>,

    /// API key sent as a bearer credential; prefer --api-key-file to avoid argv exposure.
    #[arg(
        long,
        value_name = "KEY",
        conflicts_with_all = ["api_key_file", "no_api_key"]
    )]
    api_key: Option<String>,

    /// Read the inference API key from a private file.
    #[arg(
        long,
        value_name = "PATH",
        conflicts_with_all = ["api_key", "no_api_key"]
    )]
    api_key_file: Option<PathBuf>,

    /// Disable an API key supplied by the selected profile or environment.
    #[arg(long, conflicts_with_all = ["api_key", "api_key_file"])]
    no_api_key: bool,

    /// Synthetic prompt to send; visible in argv. The deterministic built-in prompt is safer.
    #[arg(
        long,
        value_name = "TEXT",
        value_parser = parse_nonempty_string,
        conflicts_with_all = ["prompt_file", "default_prompt"]
    )]
    prompt: Option<String>,

    /// Read the synthetic prompt from a bounded UTF-8 file.
    #[arg(
        long,
        value_name = "PATH",
        conflicts_with_all = ["prompt", "default_prompt"]
    )]
    prompt_file: Option<PathBuf>,

    /// Ignore a profile prompt file and use the built-in deterministic prompt.
    #[arg(long, conflicts_with_all = ["prompt", "prompt_file"])]
    default_prompt: bool,

    /// Non-secret identity for a custom synthetic workload; required with a custom prompt.
    #[arg(
        long,
        value_name = "ID",
        value_parser = parse_workload_id,
        conflicts_with = "default_prompt"
    )]
    workload_id: Option<String>,

    /// Require generated text to contain this value; visible in argv.
    #[arg(
        long,
        value_name = "TEXT",
        value_parser = parse_nonempty_string,
        conflicts_with = "no_expect"
    )]
    expect: Option<String>,

    /// Disable the built-in or profile-backed expected-text check.
    #[arg(long, conflicts_with = "expect")]
    no_expect: bool,

    /// Strictly increasing concurrency points to test, starting at one (maximum eight points).
    #[arg(
        long,
        required = true,
        value_name = "N[,N...]",
        value_delimiter = ',',
        num_args = 1..,
        value_parser = parse_canary_concurrency
    )]
    concurrency_stages: Vec<u32>,

    /// Unmeasured warmup requests per worker at every exact stage (1-10).
    #[arg(long, default_value_t = 2, value_parser = parse_benchmark_warmup_requests)]
    warmup_requests_per_worker: u32,

    /// Measured closed-loop requests per worker at every stage (1-100).
    #[arg(long, default_value_t = 20, value_parser = parse_benchmark_requests_per_worker)]
    requests_per_worker: u32,

    /// Maximum completion tokens requested per attempt (defaults to 128; maximum 65,536).
    #[arg(long, value_parser = parse_canary_max_tokens)]
    max_tokens: Option<u32>,

    /// Deadline for each inference request (defaults to 10 seconds; maximum 5 minutes).
    #[arg(long, value_parser = parse_nonzero_duration)]
    timeout: Option<Duration>,

    /// Maximum accepted response bytes per request (1 to 8 MiB).
    #[arg(
        long,
        default_value_t = 131_072,
        value_parser = parse_benchmark_response_limit
    )]
    response_limit_bytes: usize,

    /// Force server-sent event streaming when a profile disables it.
    #[arg(long, conflicts_with = "no_stream")]
    stream: bool,

    /// Disable server-sent event streaming; TTFT cannot then be measured.
    #[arg(long, conflicts_with_all = ["stream", "max_p95_ttft"])]
    no_stream: bool,

    /// Enforce all gates at this exact tested concurrency point.
    #[arg(long, value_parser = parse_canary_concurrency)]
    verify_concurrency: Option<u32>,

    /// Maximum failed-request percentage accepted at each stage.
    #[arg(long, default_value_t = 1.0, value_parser = parse_percentage_f64)]
    max_error_percent: f64,

    /// Maximum p95 time to first token at every measured stage (at least 20 samples required).
    #[arg(long, value_name = "DURATION", value_parser = parse_nonzero_duration)]
    max_p95_ttft: Option<Duration>,

    /// Maximum p95 end-to-end latency at every measured stage (at least 20 samples required).
    #[arg(long, value_name = "DURATION", value_parser = parse_nonzero_duration)]
    max_p95_e2e: Option<Duration>,

    /// Minimum successful request goodput at every measured stage.
    #[arg(long, value_name = "REQUESTS/SECOND", value_parser = parse_nonnegative_f64)]
    min_successful_requests_per_second: Option<f64>,

    /// Minimum authoritative completion-token goodput at every measured stage.
    #[arg(long, value_name = "TOKENS/SECOND", value_parser = parse_nonnegative_f64)]
    min_completion_token_goodput_per_second: Option<f64>,

    /// Stop after an exact-stage warmup or measurement reaches this error percentage.
    #[arg(long, default_value_t = 50.0, value_parser = parse_percentage_f64)]
    abort_error_percent: f64,

    /// Emit human-readable, JSON, or compact NDJSON output.
    #[arg(long, value_enum, default_value_t = CliFormat::Text)]
    format: CliFormat,
}

#[derive(clap::Args)]
struct SaturationCompareArgs {
    /// Saved baseline `SaturationBenchmarkReport` JSON or NDJSON file.
    baseline_benchmark: PathBuf,

    /// Saved candidate `SaturationBenchmarkReport` JSON or NDJSON file.
    candidate_benchmark: PathBuf,

    /// Maximum candidate p95 TTFT increase at every exact concurrency stage.
    #[arg(long, value_name = "PERCENT", value_parser = parse_nonnegative_f64)]
    max_p95_ttft_regression_percent: Option<f64>,

    /// Maximum candidate p95 end-to-end latency increase at every exact stage.
    #[arg(long, value_name = "PERCENT", value_parser = parse_nonnegative_f64)]
    max_p95_e2e_regression_percent: Option<f64>,

    /// Minimum candidate/baseline successful request-rate ratio at every stage.
    #[arg(
        long = "min-successful-rps-ratio",
        visible_alias = "min-successful-requests-per-second-ratio",
        value_name = "RATIO",
        value_parser = parse_nonnegative_f64
    )]
    min_successful_rps_ratio: Option<f64>,

    /// Minimum candidate/baseline authoritative completion-token goodput ratio.
    #[arg(
        long = "min-completion-token-goodput-ratio",
        value_name = "RATIO",
        value_parser = parse_nonnegative_f64
    )]
    min_completion_token_goodput_ratio: Option<f64>,

    /// Maximum candidate error-rate increase in percentage points at every stage.
    #[arg(
        long,
        value_name = "POINTS",
        value_parser = parse_percentage_f64
    )]
    max_error_percent_increase: Option<f64>,

    /// Exit 2 for regression, incompatible identity, or unavailable required evidence.
    #[arg(long)]
    fail_on_regression: bool,

    /// Emit human-readable, JSON, or compact NDJSON output.
    #[arg(long, value_enum, default_value_t = CliFormat::Text)]
    format: CliFormat,
}

#[derive(clap::Args)]
struct RolloutArgs {
    /// Saved baseline `CanaryReport` JSON or NDJSON file.
    baseline_canary: PathBuf,

    /// Saved candidate `CanaryReport` JSON or NDJSON file.
    candidate_canary: PathBuf,

    /// Maximum allowed candidate p95 TTFT increase relative to baseline.
    #[arg(long, value_name = "PERCENT", value_parser = parse_nonnegative_f64)]
    max_p95_ttft_regression_percent: Option<f64>,

    /// Maximum allowed candidate p95 end-to-end latency increase relative to baseline.
    #[arg(long, value_name = "PERCENT", value_parser = parse_nonnegative_f64)]
    max_p95_e2e_regression_percent: Option<f64>,

    /// Minimum candidate/baseline p50 authoritative output-token-rate ratio.
    #[arg(
        long = "min-output-tps-ratio",
        visible_alias = "min-p50-output-tps-ratio",
        value_name = "RATIO",
        value_parser = parse_nonnegative_f64
    )]
    min_output_tps_ratio: Option<f64>,

    /// Maximum allowed baseline-to-candidate success percentage-point drop.
    #[arg(
        long = "max-success-drop-percent",
        visible_alias = "max-success-percent-drop",
        value_name = "POINTS",
        value_parser = parse_percentage_f64
    )]
    max_success_drop_percent: Option<f64>,

    /// Exit 2 when identity is incompatible or any selected gate fails.
    #[arg(long)]
    fail_on_regression: bool,

    /// Emit human-readable, JSON, or compact NDJSON output.
    #[arg(long, value_enum, default_value_t = CliFormat::Text)]
    format: CliFormat,
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Clone)]
struct MonitorArgs {
    view: MonitorView,
    format: CliFormat,
    color: ColorChoice,
    watch: Option<Duration>,
    all: bool,
    details: bool,
    listen: Option<String>,
    probe: Vec<String>,
    probe_token: Option<String>,
    probe_token_file: Option<PathBuf>,
    probe_timeout: Duration,
    probe_allow_insecure_http: bool,
    history: Option<PathBuf>,
    command_timeout: Duration,
    fail_on: FailOn,
    require_source: Vec<String>,
    quiet: bool,
    no_clear: bool,
    gpu: Vec<String>,
    nvidia_smi: PathBuf,
    vram_warning: i64,
    vram_critical: i64,
    temperature_warning: i32,
    temperature_critical: i32,
    kv_cache_warning: f64,
    kv_cache_critical: f64,
    process_growth_mib: i64,
    freshness: Duration,
    api_token: Option<String>,
    api_token_file: Option<PathBuf>,
    allow_unauthenticated_api: bool,
    no_xid: bool,
    allow_remote_listen: bool,
}

impl Default for MonitorArgs {
    fn default() -> Self {
        Self {
            view: MonitorView::Full,
            format: CliFormat::Text,
            color: ColorChoice::Auto,
            watch: None,
            all: false,
            details: false,
            listen: None,
            probe: Vec::new(),
            probe_token: None,
            probe_token_file: None,
            probe_timeout: Duration::from_secs(3),
            probe_allow_insecure_http: false,
            history: None,
            command_timeout: Duration::from_secs(3),
            fail_on: FailOn::Never,
            require_source: Vec::new(),
            quiet: false,
            no_clear: false,
            gpu: Vec::new(),
            nvidia_smi: PathBuf::from("nvidia-smi"),
            vram_warning: 90,
            vram_critical: 99,
            temperature_warning: 82,
            temperature_critical: 90,
            kv_cache_warning: 85.0,
            kv_cache_critical: 95.0,
            process_growth_mib: 256,
            freshness: Duration::from_secs(120),
            api_token: None,
            api_token_file: None,
            allow_unauthenticated_api: false,
            no_xid: false,
            allow_remote_listen: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, ValueEnum, PartialEq, Eq)]
enum CliFormat {
    #[default]
    Text,
    Json,
    Ndjson,
}

#[derive(Debug, Clone, Copy, Default, ValueEnum, PartialEq, Eq)]
enum LiveFormat {
    #[default]
    Text,
    Ndjson,
}

#[derive(Debug, Clone, Copy, Default, ValueEnum, PartialEq, Eq)]
enum ConfigOutputFormat {
    #[default]
    Toml,
    Json,
}

impl From<LiveFormat> for CliFormat {
    fn from(value: LiveFormat) -> Self {
        match value {
            LiveFormat::Text => Self::Text,
            LiveFormat::Ndjson => Self::Ndjson,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, ValueEnum, PartialEq, Eq)]
enum ColorChoice {
    #[default]
    Auto,
    Always,
    Never,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum MonitorView {
    #[default]
    Full,
    Processes,
}

impl ColorChoice {
    fn enabled(self, terminal: bool) -> bool {
        match self {
            Self::Always => true,
            Self::Never => false,
            Self::Auto => {
                terminal
                    && std::env::var_os("NO_COLOR").is_none()
                    && std::env::var("TERM").map_or(true, |term| term != "dumb")
            }
        }
    }
}

impl From<CliFormat> for OutputFormat {
    fn from(value: CliFormat) -> Self {
        match value {
            CliFormat::Text => Self::Text,
            CliFormat::Json => Self::Json,
            CliFormat::Ndjson => Self::Ndjson,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum FailOn {
    Never,
    Warning,
    Critical,
}

impl From<ProfileFailOn> for FailOn {
    fn from(value: ProfileFailOn) -> Self {
        match value {
            ProfileFailOn::Never => Self::Never,
            ProfileFailOn::Warning => Self::Warning,
            ProfileFailOn::Critical => Self::Critical,
        }
    }
}

/// Parse the command line, run the selected workflow, and map failures to the CLI contract.
pub fn entrypoint() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            let exit_code = if error.exit_code() == 0 {
                0
            } else {
                EXIT_FAILURE
            };
            let _ = error.print();
            return ExitCode::from(exit_code);
        }
    };
    match run(cli) {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("gpu-watchman: {error:#}");
            ExitCode::from(EXIT_FAILURE)
        }
    }
}

fn run(cli: Cli) -> Result<u8> {
    let Cli {
        config,
        profile,
        command,
    } = cli;
    if let Some(Commands::Config(args)) = command {
        return run_config(args, config, profile.as_deref());
    }
    let profile_aware = matches!(
        &command,
        None | Some(
            Commands::Snapshot(_)
                | Commands::Top(_)
                | Commands::Serve(_)
                | Commands::Ps(_)
                | Commands::Canary(_)
                | Commands::Benchmark(BenchmarkArgs {
                    command: BenchmarkCommands::Saturation(_),
                })
        )
    );
    if !profile_aware && (config.is_some() || profile.is_some()) {
        bail!(
            "--config and --profile apply only to snapshot, top, serve, ps, canary, benchmark saturation, and config commands"
        );
    }
    if profile.is_some() && config.is_none() {
        bail!("--profile requires --config");
    }
    let loaded = if profile_aware {
        config.as_deref().map(load_config).transpose()?
    } else {
        None
    };
    let selected = loaded
        .as_ref()
        .map(|loaded| select_profile(&loaded.config, profile.as_deref()))
        .transpose()?
        .flatten();
    let operational_profile = selected.map(|selected| selected.profile);

    match command {
        Some(Commands::Snapshot(args)) => run_snapshot(args, operational_profile),
        Some(Commands::Top(args)) => run_top(args, operational_profile),
        Some(Commands::Serve(args)) => run_serve(args, operational_profile),
        Some(Commands::Ps(args)) => run_ps(args, operational_profile),
        Some(Commands::Version) => {
            let _ = write_stdout(&format!("{}\n", env!("CARGO_PKG_VERSION")))?;
            Ok(0)
        }
        Some(Commands::Artifact(args)) => run_artifact(args),
        Some(Commands::Runtime(args)) => run_runtime(args),
        Some(Commands::Capacity(args)) => run_capacity(&args),
        Some(Commands::Canary(args)) => run_canary(args, operational_profile),
        Some(Commands::Benchmark(args)) => run_benchmark(args, operational_profile),
        Some(Commands::Rollout(args)) => run_rollout(&args),
        Some(Commands::History { path, format }) => {
            let summary = history::analyze(&path)?;
            if format == CliFormat::Text {
                let _ = write_stdout(&history::render_text(&summary))?;
            } else {
                let _ = write_serialized(&summary, format)?;
            }
            Ok(0)
        }
        Some(Commands::Compare {
            baseline,
            current,
            fail_on_regression,
            format,
        }) => run_compare(&baseline, &current, fail_on_regression, format),
        Some(Commands::Bundle(args)) => run_bundle_command(args),
        Some(Commands::Completions { shell }) => {
            let mut output = Vec::new();
            clap_complete::generate(shell, &mut Cli::command(), "gpu-watchman", &mut output);
            let _ = write_stdout_bytes(&output)?;
            Ok(0)
        }
        Some(Commands::Doctor(args)) => run_doctor_command(args),
        Some(Commands::Config(_)) => unreachable!("config commands return before profile loading"),
        None => run_snapshot(SnapshotArgs::default(), operational_profile),
    }
}

fn run_artifact(args: ArtifactArgs) -> Result<u8> {
    match args.command {
        ArtifactCommands::Inspect { path, format } => {
            let report = inspect_artifact(&path)?;
            if format == CliFormat::Text {
                let _ = write_stdout(&render_artifact_text(&report))?;
            } else {
                let _ = write_serialized(&report, format)?;
            }
            Ok(0)
        }
    }
}

fn run_runtime(args: RuntimeArgs) -> Result<u8> {
    match args.command {
        RuntimeCommands::Inspect {
            pid,
            allow_incomplete,
            format,
        } => {
            let report = runtime::inspect(&pid)?;
            if format == CliFormat::Text {
                let _ = write_stdout(&render_runtime_fingerprint(&report))?;
            } else {
                let _ = write_serialized(&report, format)?;
            }
            Ok(if report.complete || allow_incomplete {
                0
            } else {
                EXIT_UNHEALTHY
            })
        }
    }
}

fn run_bundle_command(args: BundleArgs) -> Result<u8> {
    let BundleArgs {
        output,
        probe,
        probe_token,
        probe_token_file,
        no_probe_auth,
        allow_insecure_http,
        timeout,
        nvidia_smi,
        no_xid,
    } = args;
    validate_probe_target_count(&probe)?;
    if probe.is_empty()
        && (probe_token.is_some()
            || probe_token_file.is_some()
            || no_probe_auth
            || allow_insecure_http)
    {
        bail!("probe authentication and transport options require --probe");
    }
    let probe_token =
        resolve_probe_credential(&probe, probe_token, probe_token_file, no_probe_auth)?;
    bundle::write(
        &output,
        &nvidia_smi,
        timeout,
        &probe,
        probe_token,
        allow_insecure_http,
        no_xid,
    )?;
    eprintln!(
        "gpu-watchman: wrote support bundle {} (review process commands and cgroups before sharing)",
        output.display()
    );
    Ok(0)
}

fn run_doctor_command(args: DoctorArgs) -> Result<u8> {
    let DoctorArgs {
        probe,
        probe_token,
        probe_token_file,
        no_probe_auth,
        allow_insecure_http,
        format,
        timeout,
        nvidia_smi,
    } = args;
    validate_probe_target_count(&probe)?;
    if probe.is_empty()
        && (probe_token.is_some()
            || probe_token_file.is_some()
            || no_probe_auth
            || allow_insecure_http)
    {
        bail!("probe authentication and transport options require --probe");
    }
    let probe_token =
        resolve_probe_credential(&probe, probe_token, probe_token_file, no_probe_auth)?;
    run_doctor(
        &nvidia_smi,
        timeout,
        &probe,
        probe_token,
        allow_insecure_http,
        format,
    )
}

fn resolve_probe_credential(
    probes: &[String],
    probe_token: Option<String>,
    probe_token_file: Option<PathBuf>,
    no_probe_auth: bool,
) -> Result<Option<String>> {
    if probes.is_empty() {
        return Ok(None);
    }
    let (probe_token, probe_token_file) = credential_inputs(
        probe_token,
        probe_token_file,
        no_probe_auth,
        "GPU_WATCHMAN_PROBE_TOKEN",
        "GPU_WATCHMAN_PROBE_TOKEN_FILE",
        None,
    )?;
    resolve_secret(probe_token, probe_token_file.as_deref())
}

fn run_config(
    args: ConfigArgs,
    config_path: Option<PathBuf>,
    requested_profile: Option<&str>,
) -> Result<u8> {
    match args.command {
        ConfigCommands::Init { path } => {
            if requested_profile.is_some() {
                bail!("--profile is not valid with config init");
            }
            let path = path
                .or(config_path)
                .unwrap_or_else(|| PathBuf::from("gpu-watchman.toml"));
            init_config(&path)?;
            let _ = write_stdout(&format!(
                "created private configuration {}\n",
                path.display()
            ))?;
            Ok(0)
        }
        ConfigCommands::Validate => {
            let path = config_path.ok_or_else(|| {
                anyhow::anyhow!("config validate requires --config PATH or GPU_WATCHMAN_CONFIG")
            })?;
            let loaded = load_config(&path)?;
            let selected = select_profile(&loaded.config, requested_profile)?;
            let selection = selected.map_or_else(
                || "no default profile selected".to_owned(),
                |selected| format!("profile {:?} selected", selected.name),
            );
            let _ = write_stdout(&format!(
                "valid configuration {} ({} profile(s); {selection})\n",
                loaded.path.display(),
                loaded.config.profiles.len()
            ))?;
            Ok(0)
        }
        ConfigCommands::Show { format } => {
            let path = config_path.ok_or_else(|| {
                anyhow::anyhow!("config show requires --config PATH or GPU_WATCHMAN_CONFIG")
            })?;
            let loaded = load_config(&path)?;
            let selected_name = select_profile(&loaded.config, requested_profile)?
                .map(|selected| selected.name.to_owned());
            let mut visible = loaded.config;
            if let Some(name) = selected_name {
                visible.profiles.retain(|candidate, _| candidate == &name);
                visible.default_profile = Some(name);
            }
            let safe = safe_toml(&visible)?;
            match format {
                ConfigOutputFormat::Toml => {
                    let _ = write_stdout(&safe)?;
                }
                ConfigOutputFormat::Json => {
                    let safe_value: toml::Value = toml::from_str(&safe)?;
                    let mut output = serde_json::to_string_pretty(&safe_value)?;
                    output.push('\n');
                    let _ = write_stdout(&output)?;
                }
            }
            Ok(0)
        }
    }
}

fn run_snapshot(args: SnapshotArgs, profile: Option<&ProfileV1>) -> Result<u8> {
    if args.quiet && args.format != CliFormat::Text {
        bail!("snapshot --quiet writes history only; --format has no effect");
    }
    if args.format != CliFormat::Text
        && (args.all || args.details || args.color != ColorChoice::Auto)
    {
        bail!("snapshot --all, --details, and --color apply only to text output");
    }
    let mut monitor_args = monitor_args(args.collection, args.probes, args.health, None, profile)?;
    monitor_args.format = args.format;
    monitor_args.color = args.color;
    monitor_args.all = args.all;
    monitor_args.details = args.details;
    monitor_args.history = args.history;
    monitor_args.quiet = args.quiet;
    monitor::run(&monitor_args)
}

fn run_top(args: TopArgs, profile: Option<&ProfileV1>) -> Result<u8> {
    if args.format == LiveFormat::Ndjson
        && (args.no_clear || args.details || args.color != ColorChoice::Auto)
    {
        bail!("top --no-clear, --details, and --color apply only to text output");
    }
    let mut monitor_args = monitor_args(
        args.collection,
        args.probes,
        args.health.health,
        args.health.process_growth_mib,
        profile,
    )?;
    monitor_args.watch = Some(args.interval.unwrap_or(Duration::from_secs(2)));
    monitor_args.history = args.history;
    monitor_args.no_clear = args.no_clear;
    monitor_args.format = args.format.into();
    monitor_args.color = args.color;
    monitor_args.all = true;
    monitor_args.details = args.details;
    monitor::run(&monitor_args)
}

fn run_serve(args: ServeArgs, profile: Option<&ProfileV1>) -> Result<u8> {
    let service = profile.and_then(|profile| profile.service.as_ref());
    let allow_unauthenticated_api = args.no_api_auth;
    let mut monitor_args = monitor_args(
        args.collection,
        args.probes,
        args.health.health,
        args.health.process_growth_mib,
        profile,
    )?;
    let (api_token, api_token_file) = credential_inputs(
        args.api_token,
        args.api_token_file,
        args.no_api_auth,
        "GPU_WATCHMAN_API_TOKEN",
        "GPU_WATCHMAN_API_TOKEN_FILE",
        service.and_then(|service| service.api_token_file.clone()),
    )?;
    monitor_args.listen = Some(
        args.listen
            .or_else(|| service.and_then(|service| service.listen.clone()))
            .unwrap_or_else(|| "127.0.0.1:9400".to_owned()),
    );
    monitor_args.watch = Some(
        args.interval
            .or_else(|| service.and_then(|service| service.interval.map(Into::into)))
            .unwrap_or(Duration::from_secs(5)),
    );
    monitor_args.history = if args.no_history {
        None
    } else {
        args.history
            .or_else(|| service.and_then(|service| service.history_file.clone()))
    };
    monitor_args.freshness = args
        .freshness
        .or_else(|| service.and_then(|service| service.freshness.map(Into::into)))
        .unwrap_or(Duration::from_secs(120));
    monitor_args.api_token = api_token;
    monitor_args.api_token_file = api_token_file;
    monitor_args.allow_unauthenticated_api = allow_unauthenticated_api;
    monitor_args.allow_remote_listen = if args.deny_remote_listen {
        false
    } else if args.allow_remote_listen {
        true
    } else {
        service
            .and_then(|service| service.allow_remote_listen)
            .unwrap_or(false)
    };
    monitor_args.quiet = if args.no_emit {
        true
    } else if args.emit.is_some() {
        false
    } else {
        service.and_then(|service| service.quiet).unwrap_or(true)
    };
    monitor_args.format = args.emit.unwrap_or(LiveFormat::Text).into();
    monitor_args.color = ColorChoice::Never;
    monitor_args.no_clear = true;
    monitor_args.all = true;
    monitor::run(&monitor_args)
}

fn run_ps(args: PsArgs, profile: Option<&ProfileV1>) -> Result<u8> {
    if args.format != CliFormat::Text && args.color != ColorChoice::Auto {
        bail!("ps --color applies only to text output");
    }
    let collection = profile
        .and_then(|profile| profile.monitor.as_ref())
        .and_then(|monitor| monitor.collection.as_ref());
    let monitor_args = MonitorArgs {
        view: MonitorView::Processes,
        format: args.format,
        color: args.color,
        all: true,
        gpu: if args.all_gpus {
            Vec::new()
        } else if args.gpu.is_empty() {
            collection
                .and_then(|collection| collection.gpus.clone())
                .unwrap_or_default()
        } else {
            args.gpu
        },
        nvidia_smi: args
            .nvidia_smi
            .or_else(|| collection.and_then(|collection| collection.nvidia_smi.clone()))
            .unwrap_or_else(|| PathBuf::from("nvidia-smi")),
        command_timeout: args
            .driver_timeout
            .or_else(|| {
                collection.and_then(|collection| collection.command_timeout.map(Into::into))
            })
            .unwrap_or(Duration::from_secs(3)),
        require_source: if args.allow_incomplete {
            Vec::new()
        } else {
            vec!["nvidia.processes".to_owned()]
        },
        no_xid: true,
        ..MonitorArgs::default()
    };
    monitor::run(&monitor_args)
}

struct ResolvedMonitorHealth {
    fail_on: FailOn,
    required_sources: Vec<String>,
    vram_warning: i64,
    vram_critical: i64,
    temperature_warning: i32,
    temperature_critical: i32,
    kv_cache_warning: f64,
    kv_cache_critical: f64,
    process_growth_mib: i64,
}

fn resolve_monitor_health(
    cli: HealthCliArgs,
    process_growth_mib: Option<i64>,
    profile: Option<&MonitorHealthProfileV1>,
) -> ResolvedMonitorHealth {
    ResolvedMonitorHealth {
        fail_on: cli
            .fail_on
            .or_else(|| profile.and_then(|health| health.fail_on).map(Into::into))
            .unwrap_or(FailOn::Never),
        required_sources: if cli.no_require_source {
            Vec::new()
        } else if cli.require_source.is_empty() {
            profile
                .and_then(|health| health.required_sources.clone())
                .unwrap_or_default()
        } else {
            cli.require_source
        },
        vram_warning: cli
            .vram_warning
            .or_else(|| profile.and_then(|health| health.vram_warning_percent))
            .unwrap_or(90),
        vram_critical: cli
            .vram_critical
            .or_else(|| profile.and_then(|health| health.vram_critical_percent))
            .unwrap_or(99),
        temperature_warning: cli
            .temperature_warning
            .or_else(|| profile.and_then(|health| health.temperature_warning_c))
            .unwrap_or(82),
        temperature_critical: cli
            .temperature_critical
            .or_else(|| profile.and_then(|health| health.temperature_critical_c))
            .unwrap_or(90),
        kv_cache_warning: cli
            .kv_cache_warning
            .or_else(|| profile.and_then(|health| health.kv_cache_warning_percent))
            .unwrap_or(85.0),
        kv_cache_critical: cli
            .kv_cache_critical
            .or_else(|| profile.and_then(|health| health.kv_cache_critical_percent))
            .unwrap_or(95.0),
        process_growth_mib: process_growth_mib
            .or_else(|| profile.and_then(|health| health.process_growth_warning_mib))
            .unwrap_or(256),
    }
}

fn monitor_args(
    collection: CollectionCliArgs,
    probes: ProbeCliArgs,
    health: HealthCliArgs,
    process_growth_mib: Option<i64>,
    profile: Option<&ProfileV1>,
) -> Result<MonitorArgs> {
    let monitor = profile.and_then(|profile| profile.monitor.as_ref());
    let profile_collection = monitor.and_then(|monitor| monitor.collection.as_ref());
    let profile_inference = monitor.and_then(|monitor| monitor.inference.as_ref());
    let profile_health = monitor.and_then(|monitor| monitor.health.as_ref());
    let health = resolve_monitor_health(health, process_growth_mib, profile_health);
    let cli_probe_options = probes.probe_token.is_some()
        || probes.probe_token_file.is_some()
        || probes.no_probe_auth
        || probes.probe_timeout.is_some()
        || probes.allow_insecure_probe_http
        || probes.deny_insecure_probe_http;
    let probe = if probes.no_probe {
        Vec::new()
    } else if probes.probe.is_empty() {
        profile_inference
            .and_then(|inference| inference.urls.clone())
            .unwrap_or_default()
    } else {
        probes.probe
    };
    validate_probe_target_count(&probe)?;
    if probe.is_empty() && cli_probe_options {
        bail!("probe authentication, timeout, and transport options require an active endpoint");
    }
    let (probe_token, probe_token_file) = if probe.is_empty() {
        (None, None)
    } else {
        credential_inputs(
            probes.probe_token,
            probes.probe_token_file,
            probes.no_probe_auth,
            "GPU_WATCHMAN_PROBE_TOKEN",
            "GPU_WATCHMAN_PROBE_TOKEN_FILE",
            profile_inference.and_then(|inference| inference.token_file.clone()),
        )?
    };
    Ok(MonitorArgs {
        probe,
        probe_token,
        probe_token_file,
        probe_timeout: probes
            .probe_timeout
            .or_else(|| profile_inference.and_then(|inference| inference.timeout.map(Into::into)))
            .unwrap_or(Duration::from_secs(3)),
        command_timeout: collection
            .driver_timeout
            .or_else(|| {
                profile_collection.and_then(|collection| collection.command_timeout.map(Into::into))
            })
            .unwrap_or(Duration::from_secs(3)),
        probe_allow_insecure_http: if probes.deny_insecure_probe_http {
            false
        } else if probes.allow_insecure_probe_http {
            true
        } else {
            profile_inference
                .and_then(|inference| inference.allow_insecure_http)
                .unwrap_or(false)
        },
        fail_on: health.fail_on,
        require_source: health.required_sources,
        gpu: if collection.all_gpus {
            Vec::new()
        } else if collection.gpu.is_empty() {
            profile_collection
                .and_then(|collection| collection.gpus.clone())
                .unwrap_or_default()
        } else {
            collection.gpu
        },
        nvidia_smi: collection
            .nvidia_smi
            .or_else(|| profile_collection.and_then(|collection| collection.nvidia_smi.clone()))
            .unwrap_or_else(|| PathBuf::from("nvidia-smi")),
        vram_warning: health.vram_warning,
        vram_critical: health.vram_critical,
        temperature_warning: health.temperature_warning,
        temperature_critical: health.temperature_critical,
        kv_cache_warning: health.kv_cache_warning,
        kv_cache_critical: health.kv_cache_critical,
        process_growth_mib: health.process_growth_mib,
        no_xid: if collection.no_xid {
            true
        } else if collection.xid {
            false
        } else {
            !profile_collection
                .and_then(|collection| collection.collect_xid)
                .unwrap_or(true)
        },
        ..MonitorArgs::default()
    })
}

fn validate_probe_target_count(probes: &[String]) -> Result<()> {
    if probes.len() > MAX_PROBE_TARGETS {
        bail!("at most {MAX_PROBE_TARGETS} inference probe targets are allowed");
    }
    Ok(())
}

fn credential_inputs(
    cli_value: Option<String>,
    cli_file: Option<PathBuf>,
    clear: bool,
    value_environment: &str,
    file_environment: &str,
    profile_file: Option<PathBuf>,
) -> Result<(Option<String>, Option<PathBuf>)> {
    if clear {
        return Ok((None, None));
    }
    if cli_value.is_some() && cli_file.is_some() {
        bail!("inline and file-backed credential inputs conflict");
    }
    if cli_value.is_some() || cli_file.is_some() {
        return Ok((cli_value, cli_file));
    }
    let environment_value = std::env::var_os(value_environment)
        .map(|value| {
            value
                .into_string()
                .map_err(|_| anyhow::anyhow!("{value_environment} is not valid UTF-8"))
        })
        .transpose()?;
    let environment_file = std::env::var_os(file_environment).map(PathBuf::from);
    if environment_value.is_some() && environment_file.is_some() {
        bail!("{value_environment} conflicts with {file_environment}");
    }
    if environment_value.is_some() || environment_file.is_some() {
        return Ok((environment_value, environment_file));
    }
    Ok((None, profile_file))
}

struct ResolvedCanaryPolicy {
    stream: bool,
    allow_insecure_http: bool,
    thresholds: canary::CanaryThresholds,
}

fn resolve_canary_policy(
    args: &CanaryArgs,
    profile: Option<&CanaryProfileV1>,
) -> ResolvedCanaryPolicy {
    let slo = profile.and_then(|profile| profile.slo.as_ref());
    let stream = if args.no_stream {
        false
    } else if args.stream {
        true
    } else {
        profile.and_then(|profile| profile.stream).unwrap_or(true)
    };
    ResolvedCanaryPolicy {
        stream,
        allow_insecure_http: if args.deny_insecure_http {
            false
        } else if args.allow_insecure_http {
            true
        } else {
            profile
                .and_then(|profile| profile.allow_insecure_http)
                .unwrap_or(false)
        },
        thresholds: canary::CanaryThresholds {
            min_success_percent: args
                .min_success_percent
                .or_else(|| slo.and_then(|slo| slo.min_success_percent))
                .unwrap_or(100.0),
            max_ttft: if args.no_max_ttft || args.no_stream {
                None
            } else {
                args.max_ttft
                    .or_else(|| slo.and_then(|slo| slo.max_ttft.map(Into::into)))
            },
            max_e2e: if args.no_max_e2e {
                None
            } else {
                args.max_e2e
                    .or_else(|| slo.and_then(|slo| slo.max_e2e.map(Into::into)))
            },
            min_output_tokens_per_second: if args.no_min_output_tokens_per_second || args.no_stream
            {
                None
            } else {
                args.min_output_tokens_per_second
                    .or_else(|| slo.and_then(|slo| slo.min_output_tokens_per_second))
            },
        },
    }
}

fn run_canary(args: CanaryArgs, profile: Option<&ProfileV1>) -> Result<u8> {
    let profile = profile.and_then(|profile| profile.canary.as_ref());
    let slo = profile.and_then(|profile| profile.slo.as_ref());
    let policy = resolve_canary_policy(&args, profile);
    let cli_custom_prompt = args.prompt.is_some() || args.prompt_file.is_some();
    let use_profile_expectation = !args.default_prompt && !cli_custom_prompt;
    let prompt_file = if args.default_prompt {
        None
    } else {
        args.prompt_file
            .or_else(|| profile.and_then(|profile| profile.prompt_file.clone()))
    };
    let requested_workload_id = args.workload_id;
    let (prompt, built_in_prompt) = resolve_canary_prompt(args.prompt, prompt_file.as_deref())?;
    let workload_id = resolve_canary_workload_id(
        requested_workload_id,
        built_in_prompt,
        cli_custom_prompt,
        profile.and_then(|profile| profile.workload_id.clone()),
    )?;
    let expectation = if args.no_expect {
        None
    } else {
        args.expect
            .or_else(|| {
                use_profile_expectation
                    .then(|| slo.and_then(|slo| slo.expect.clone()))
                    .flatten()
            })
            .or_else(|| built_in_prompt.then(|| canary::DEFAULT_CANARY_EXPECTATION.to_owned()))
    };
    let (api_key, api_key_file) = credential_inputs(
        args.api_key,
        args.api_key_file,
        args.no_api_key,
        "GPU_WATCHMAN_INFERENCE_API_KEY",
        "GPU_WATCHMAN_INFERENCE_API_KEY_FILE",
        profile.and_then(|profile| profile.api_key_file.clone()),
    )?;
    let options = canary::CanaryOptions {
        base_url: args
            .base_url
            .or_else(|| profile.and_then(|profile| profile.base_url.clone()))
            .unwrap_or_else(|| "http://127.0.0.1:8000/v1".to_owned()),
        model: args
            .model
            .or_else(|| profile.and_then(|profile| profile.model.clone()))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "canary requires --model, GPU_WATCHMAN_INFERENCE_MODEL, or canary.model in the selected profile"
                )
            })?,
        workload_id,
        api_key: resolve_secret(api_key, api_key_file.as_deref())?,
        prompt,
        expectation,
        max_tokens: args
            .max_tokens
            .or_else(|| profile.and_then(|profile| profile.max_tokens))
            .unwrap_or(16),
        count: args
            .count
            .or_else(|| profile.and_then(|profile| profile.count))
            .unwrap_or(1),
        concurrency: args
            .concurrency
            .or_else(|| profile.and_then(|profile| profile.concurrency))
            .unwrap_or(1),
        timeout: args
            .timeout
            .or_else(|| profile.and_then(|profile| profile.timeout.map(Into::into)))
            .unwrap_or(Duration::from_secs(30)),
        stream: policy.stream,
        allow_insecure_http: policy.allow_insecure_http,
        thresholds: policy.thresholds,
        ..canary::CanaryOptions::default()
    };
    let report = canary::run(&options)?;
    if args.format == CliFormat::Text {
        let _ = write_stdout(&crate::presentation::render_canary(&report))?;
    } else {
        let _ = write_serialized(&report, args.format)?;
    }
    Ok(if report.status == crate::domain::CanaryStatus::Fail {
        EXIT_UNHEALTHY
    } else {
        0
    })
}

fn run_benchmark(args: BenchmarkArgs, profile: Option<&ProfileV1>) -> Result<u8> {
    match args.command {
        BenchmarkCommands::Saturation(args) => run_saturation_benchmark(*args, profile),
        BenchmarkCommands::Compare(args) => run_saturation_comparison(&args),
    }
}

fn run_saturation_benchmark(args: SaturationArgs, profile: Option<&ProfileV1>) -> Result<u8> {
    let profile = profile.and_then(|profile| profile.canary.as_ref());
    let profile_slo = profile.and_then(|profile| profile.slo.as_ref());
    let format = args.format;
    let stream = if args.no_stream {
        false
    } else if args.stream {
        true
    } else {
        profile.and_then(|profile| profile.stream).unwrap_or(true)
    };
    let allow_insecure_http = if args.deny_insecure_http {
        false
    } else if args.allow_insecure_http {
        true
    } else {
        profile
            .and_then(|profile| profile.allow_insecure_http)
            .unwrap_or(false)
    };
    let cli_custom_prompt = args.prompt.is_some() || args.prompt_file.is_some();
    let use_profile_expectation = !args.default_prompt && !cli_custom_prompt;
    let prompt_file = if args.default_prompt {
        None
    } else {
        args.prompt_file
            .or_else(|| profile.and_then(|profile| profile.prompt_file.clone()))
    };
    let (prompt, built_in_prompt) = resolve_canary_prompt(args.prompt, prompt_file.as_deref())?;
    let workload_id = resolve_canary_workload_id(
        args.workload_id,
        built_in_prompt,
        cli_custom_prompt,
        profile.and_then(|profile| profile.workload_id.clone()),
    )?;
    let expectation = if args.no_expect {
        None
    } else {
        args.expect
            .or_else(|| {
                use_profile_expectation
                    .then(|| profile_slo.and_then(|slo| slo.expect.clone()))
                    .flatten()
            })
            .or_else(|| built_in_prompt.then(|| canary::DEFAULT_CANARY_EXPECTATION.to_owned()))
    };
    let (api_key, api_key_file) = credential_inputs(
        args.api_key,
        args.api_key_file,
        args.no_api_key,
        "GPU_WATCHMAN_INFERENCE_API_KEY",
        "GPU_WATCHMAN_INFERENCE_API_KEY_FILE",
        profile.and_then(|profile| profile.api_key_file.clone()),
    )?;
    let options = saturation::SaturationOptions {
        base_url: args
            .base_url
            .or_else(|| profile.and_then(|profile| profile.base_url.clone()))
            .unwrap_or_else(|| "http://127.0.0.1:8000/v1".to_owned()),
        model: args
            .model
            .or_else(|| profile.and_then(|profile| profile.model.clone()))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "benchmark requires --model, GPU_WATCHMAN_INFERENCE_MODEL, or canary.model in the selected profile"
                )
            })?,
        workload_id,
        api_key: resolve_secret(api_key, api_key_file.as_deref())?,
        prompt,
        expectation,
        concurrency_stages: args.concurrency_stages,
        warmup_requests_per_worker: args.warmup_requests_per_worker,
        requests_per_worker: args.requests_per_worker,
        max_tokens: args
            .max_tokens
            .unwrap_or(saturation::DEFAULT_MAX_TOKENS),
        timeout: args.timeout.unwrap_or(Duration::from_secs(10)),
        max_body_bytes: args.response_limit_bytes,
        stream,
        allow_insecure_http,
        verify_concurrency: args.verify_concurrency,
        thresholds: saturation::SaturationThresholds {
            max_error_percent: args.max_error_percent,
            max_p95_ttft: args.max_p95_ttft,
            max_p95_e2e: args.max_p95_e2e,
            min_successful_requests_per_second: args.min_successful_requests_per_second,
            min_completion_token_goodput_per_second: args
                .min_completion_token_goodput_per_second,
            abort_error_percent: args.abort_error_percent,
        },
    };
    let report = saturation::run(&options)?;
    emit_saturation_benchmark(&report, format)
}

fn emit_saturation_benchmark(
    report: &crate::domain::SaturationBenchmarkReport,
    format: CliFormat,
) -> Result<u8> {
    if format == CliFormat::Text {
        let _ = write_stdout(&crate::presentation::render_saturation_benchmark(report))?;
    } else {
        let _ = write_serialized(report, format)?;
    }
    let verification_failed = matches!(
        report.verification.status,
        crate::domain::SaturationVerificationStatus::Fail
            | crate::domain::SaturationVerificationStatus::NotEvaluable
    );
    Ok(
        if report.status == crate::domain::SaturationRunStatus::Aborted || verification_failed {
            EXIT_UNHEALTHY
        } else {
            0
        },
    )
}

fn run_saturation_comparison(args: &SaturationCompareArgs) -> Result<u8> {
    let baseline = saturation_compare::load_saturation_report(&args.baseline_benchmark)?;
    let candidate = saturation_compare::load_saturation_report(&args.candidate_benchmark)?;
    let comparison = saturation_compare::compare(
        &baseline,
        &candidate,
        &crate::domain::SaturationComparisonPolicy {
            max_p95_ttft_regression_percent: args.max_p95_ttft_regression_percent,
            max_p95_e2e_regression_percent: args.max_p95_e2e_regression_percent,
            min_successful_requests_per_second_ratio: args.min_successful_rps_ratio,
            min_completion_token_goodput_per_second_ratio: args.min_completion_token_goodput_ratio,
            max_error_percent_increase_points: args.max_error_percent_increase,
            ..crate::domain::SaturationComparisonPolicy::default()
        },
    )?;
    match args.format {
        CliFormat::Text => {
            let _ = write_stdout(&crate::presentation::render_saturation_comparison(
                &comparison,
            ))?;
        }
        CliFormat::Json | CliFormat::Ndjson => {
            let _ = write_serialized(&comparison, args.format)?;
        }
    }
    Ok(
        if args.fail_on_regression
            && comparison.status != crate::domain::SaturationComparisonStatus::Pass
        {
            EXIT_UNHEALTHY
        } else {
            0
        },
    )
}

fn run_rollout(args: &RolloutArgs) -> Result<u8> {
    let baseline = rollout::load_canary_report(&args.baseline_canary)?;
    let candidate = rollout::load_canary_report(&args.candidate_canary)?;
    let comparison = rollout::compare(
        &baseline,
        &candidate,
        &rollout::RolloutThresholds {
            max_p95_ttft_regression_percent: args.max_p95_ttft_regression_percent,
            max_p95_e2e_regression_percent: args.max_p95_e2e_regression_percent,
            min_p50_output_tokens_per_second_ratio: args.min_output_tps_ratio,
            max_success_percent_drop: args.max_success_drop_percent,
        },
    )?;
    match args.format {
        CliFormat::Text => {
            let _ = write_stdout(&rollout::render_text(&comparison))?;
        }
        CliFormat::Json | CliFormat::Ndjson => {
            let _ = write_serialized(&comparison, args.format)?;
        }
    }
    Ok(if args.fail_on_regression && comparison.regression {
        EXIT_UNHEALTHY
    } else {
        0
    })
}

struct ResolvedCapacityGeometry {
    parameters_billion: f64,
    parameter_source: ModelParameterSource,
    model_type: Option<String>,
    layers: u32,
    kv_heads: u32,
    head_dim: u32,
    expert_count: Option<u32>,
    is_moe: bool,
    caveats: Vec<String>,
}

fn resolve_capacity_geometry(args: &CapacityArgs) -> Result<ResolvedCapacityGeometry> {
    if let Some(path) = args.model_config.as_deref() {
        let evidence = load_model_config_evidence(path)?;
        let geometry = model_geometry_from_evidence(
            &evidence,
            ModelGeometryOverrides {
                parameters_billion: args.params,
                layers: args.layers,
                kv_heads: args.kv_heads,
                head_dim: args.head_dim,
            },
        )?;
        Ok(ResolvedCapacityGeometry {
            parameters_billion: geometry.parameters_billion,
            parameter_source: geometry.parameter_source,
            model_type: Some(geometry.model_type),
            layers: geometry.layers,
            kv_heads: geometry.kv_heads,
            head_dim: geometry.head_dim,
            expert_count: geometry.expert_count,
            is_moe: geometry.is_moe,
            caveats: geometry.caveats,
        })
    } else {
        Ok(ResolvedCapacityGeometry {
            parameters_billion: args
                .params
                .ok_or_else(|| anyhow::anyhow!("--params or --model-config is required"))?,
            parameter_source: ModelParameterSource::ExplicitOverride,
            model_type: None,
            layers: args
                .layers
                .ok_or_else(|| anyhow::anyhow!("--layers or --model-config is required"))?,
            kv_heads: args
                .kv_heads
                .ok_or_else(|| anyhow::anyhow!("--kv-heads or --model-config is required"))?,
            head_dim: args
                .head_dim
                .ok_or_else(|| anyhow::anyhow!("--head-dim or --model-config is required"))?,
            expert_count: None,
            is_moe: false,
            caveats: Vec::new(),
        })
    }
}

struct ResolvedCapacityTopology {
    expected_gpu_count: Option<u32>,
    tensor_parallel_size: u32,
    pipeline_parallel_size: u32,
    data_parallel_size: u32,
    expert_parallel_size: u32,
    legacy_gpus_as_tp: bool,
}

fn resolve_capacity_topology(args: &CapacityArgs) -> ResolvedCapacityTopology {
    let explicit_topology =
        args.tp.is_some() || args.pp.is_some() || args.dp.is_some() || args.ep.is_some();
    if explicit_topology {
        ResolvedCapacityTopology {
            expected_gpu_count: args.gpus,
            tensor_parallel_size: args.tp.unwrap_or(1),
            pipeline_parallel_size: args.pp.unwrap_or(1),
            data_parallel_size: args.dp.unwrap_or(1),
            expert_parallel_size: args.ep.unwrap_or(1),
            legacy_gpus_as_tp: false,
        }
    } else {
        ResolvedCapacityTopology {
            expected_gpu_count: args.gpus,
            tensor_parallel_size: args.gpus.unwrap_or(1),
            pipeline_parallel_size: 1,
            data_parallel_size: 1,
            expert_parallel_size: 1,
            legacy_gpus_as_tp: args.gpus.is_some(),
        }
    }
}

fn run_capacity(args: &CapacityArgs) -> Result<u8> {
    let geometry = resolve_capacity_geometry(args)?;
    let topology = resolve_capacity_topology(args);
    let expert_count = match (args.expert_count, geometry.expert_count) {
        (Some(cli), Some(config)) if cli != config => {
            bail!("--expert-count ({cli}) does not match model config expert count ({config})")
        }
        (Some(cli), _) => Some(cli),
        (None, config) => config,
    };
    if geometry.is_moe && expert_count.is_none() {
        bail!(
            "detected mixture-of-experts model requires --expert-count and --expert-weight-percent placement metadata"
        );
    }
    let input = CapacityInput {
        parameters_billion: geometry.parameters_billion,
        parameter_source: geometry.parameter_source,
        model_type: geometry.model_type.clone(),
        weight_bits: args.weight_bits,
        expected_gpu_count: topology.expected_gpu_count,
        tensor_parallel_size: topology.tensor_parallel_size,
        pipeline_parallel_size: topology.pipeline_parallel_size,
        data_parallel_size: topology.data_parallel_size,
        expert_parallel_size: topology.expert_parallel_size,
        gpu_vram_gib: args.gpu_vram,
        memory_utilization: args.utilization,
        layers: geometry.layers,
        kv_heads: geometry.kv_heads,
        max_kv_heads_per_rank: args.max_kv_heads_per_rank,
        head_dim: geometry.head_dim,
        context_tokens: args.context,
        concurrent_sequences_per_data_parallel_replica: args.concurrency,
        kv_cache_bits: args.kv_bits,
        runtime_overhead_gib_per_rank: args.runtime_overhead,
        weight_overhead_percent: args.weight_overhead_percent,
        max_shared_rank_weight_percent: args.max_shared_rank_weight_percent,
        max_pipeline_stage_component_weight_percent: args.max_stage_component_weight_percent,
        max_pipeline_stage_layers: args.max_stage_layers,
        expert_count,
        expert_weight_percent: args.expert_weight_percent,
        max_expert_rank_weight_percent: args.max_expert_rank_weight_percent,
    };
    let artifact = args.artifact.as_deref().map(inspect_artifact).transpose()?;
    let mut report = if let Some(artifact) = artifact.as_ref() {
        estimate_with_artifact(
            &input,
            artifact,
            args.artifact_residency_multiplier.unwrap_or(1.0),
        )?
    } else {
        estimate(&input)?
    };
    if topology.legacy_gpus_as_tp {
        report.assumptions.push(CapacityAssumption {
            code: CapacityAssumptionCode::LegacyGpusInterpretedAsTp,
            detail: "Legacy --gpus was interpreted as tensor parallelism; use --tp/--pp/--dp for explicit placement."
                .to_owned(),
        });
    }
    if let Some(model_type) = geometry.model_type {
        let mut model_caveats = geometry.caveats;
        model_caveats.insert(
            0,
            format!("Model geometry loaded from config.json (model_type={model_type})."),
        );
        model_caveats.append(&mut report.caveats);
        report.caveats = model_caveats;
    }
    if args.format == CliFormat::Text {
        let _ = write_stdout(&render_capacity(&report))?;
    } else {
        let _ = write_serialized(&report, args.format)?;
    }
    Ok(u8::from(!report.fits) * EXIT_UNHEALTHY)
}

fn run_doctor(
    nvidia_smi: &std::path::Path,
    timeout: Duration,
    probes: &[String],
    probe_token: Option<String>,
    allow_insecure_http: bool,
    format: CliFormat,
) -> Result<u8> {
    let checks = doctor::run(
        nvidia_smi,
        timeout,
        probes,
        probe_token,
        allow_insecure_http,
    );
    if format == CliFormat::Text {
        let _ = write_stdout(&doctor::render_text(&checks))?;
    } else {
        let _ = write_serialized(&checks, format)?;
    }
    Ok(if doctor::failed(&checks) {
        EXIT_UNHEALTHY
    } else {
        0
    })
}

fn run_compare(
    baseline_path: &std::path::Path,
    current_path: &std::path::Path,
    fail_on_regression: bool,
    format: CliFormat,
) -> Result<u8> {
    let baseline = compare::load_report(baseline_path)?;
    let current = compare::load_report(current_path)?;
    let comparison = compare::compare(&baseline, &current);
    match format {
        CliFormat::Text => {
            let _ = write_stdout(&compare::render_text(&comparison))?;
        }
        CliFormat::Json | CliFormat::Ndjson => {
            let _ = write_serialized(&comparison, format)?;
        }
    }
    Ok(if fail_on_regression && comparison.regression {
        EXIT_UNHEALTHY
    } else {
        0
    })
}

fn parse_nonzero_duration(value: &str) -> Result<Duration, String> {
    let duration = humantime::parse_duration(value).map_err(|error| error.to_string())?;
    if duration.is_zero() {
        return Err("duration must be greater than zero".to_owned());
    }
    Ok(duration)
}

fn parse_probe_target(value: &str) -> Result<String, String> {
    let value = value.trim();
    if value.is_empty() {
        return Err("probe target must not be empty".to_owned());
    }
    Ok(value.to_owned())
}

fn parse_profile_name(value: &str) -> Result<String, String> {
    validate_profile_name(value).map_err(|error| error.to_string())?;
    Ok(value.to_owned())
}

fn parse_positive_f64(value: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|error| format!("invalid number: {error}"))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err("value must be a finite number greater than zero".to_owned());
    }
    Ok(parsed)
}

fn parse_artifact_residency_multiplier(value: &str) -> Result<f64, String> {
    let parsed = parse_positive_f64(value)?;
    if !(1.0..=MAX_ARTIFACT_RESIDENCY_MULTIPLIER).contains(&parsed) {
        return Err(format!(
            "value must be between 1 and {MAX_ARTIFACT_RESIDENCY_MULTIPLIER}"
        ));
    }
    Ok(parsed)
}

fn parse_nonnegative_f64(value: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|error| format!("invalid number: {error}"))?;
    if !parsed.is_finite() || parsed < 0.0 {
        return Err("value must be a finite non-negative number".to_owned());
    }
    Ok(parsed)
}

fn parse_utilization(value: &str) -> Result<f64, String> {
    let parsed = parse_positive_f64(value)?;
    if parsed > 1.0 {
        return Err("value must be greater than zero and at most one".to_owned());
    }
    Ok(parsed)
}

fn parse_positive_u32(value: &str) -> Result<u32, String> {
    let parsed = value
        .parse::<u32>()
        .map_err(|error| format!("invalid integer: {error}"))?;
    if parsed == 0 {
        return Err("value must be greater than zero".to_owned());
    }
    Ok(parsed)
}

fn parse_canary_count(value: &str) -> Result<u32, String> {
    let parsed = parse_positive_u32(value)?;
    if parsed > 10_000 {
        return Err("count must be at most 10,000".to_owned());
    }
    Ok(parsed)
}

fn parse_canary_max_tokens(value: &str) -> Result<u32, String> {
    let parsed = parse_positive_u32(value)?;
    if parsed > 65_536 {
        return Err("max tokens must be at most 65,536".to_owned());
    }
    Ok(parsed)
}

fn parse_canary_concurrency(value: &str) -> Result<u32, String> {
    let parsed = parse_positive_u32(value)?;
    if parsed > 64 {
        return Err("concurrency must be at most 64".to_owned());
    }
    Ok(parsed)
}

fn parse_benchmark_warmup_requests(value: &str) -> Result<u32, String> {
    let parsed = parse_positive_u32(value)?;
    if parsed > 10 {
        return Err("warmup requests per worker must be at most 10".to_owned());
    }
    Ok(parsed)
}

fn parse_benchmark_requests_per_worker(value: &str) -> Result<u32, String> {
    let parsed = parse_positive_u32(value)?;
    if parsed > 100 {
        return Err("requests per worker must be at most 100".to_owned());
    }
    Ok(parsed)
}

fn parse_benchmark_response_limit(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid byte count: {error}"))?;
    if parsed == 0 || parsed > 8 << 20 {
        return Err("response limit must be between 1 byte and 8 MiB".to_owned());
    }
    Ok(parsed)
}

fn parse_nonempty_string(value: &str) -> Result<String, String> {
    if value.trim().is_empty() {
        return Err("value cannot be empty".to_owned());
    }
    Ok(value.to_owned())
}

fn parse_workload_id(value: &str) -> Result<String, String> {
    if !crate::domain::valid_canary_workload_id(value) {
        return Err(format!(
            "workload ID must start with an ASCII letter or digit and use at most {} bytes of \
             letters, digits, dot, underscore, colon, slash, or hyphen",
            canary::MAX_WORKLOAD_ID_BYTES
        ));
    }
    Ok(value.to_owned())
}

fn parse_positive_u64(value: &str) -> Result<u64, String> {
    let parsed = value
        .parse::<u64>()
        .map_err(|error| format!("invalid integer: {error}"))?;
    if parsed == 0 {
        return Err("value must be greater than zero".to_owned());
    }
    Ok(parsed)
}

fn parse_positive_i64(value: &str) -> Result<i64, String> {
    let parsed = value
        .parse::<i64>()
        .map_err(|error| format!("invalid integer: {error}"))?;
    if parsed <= 0 {
        return Err("value must be greater than zero".to_owned());
    }
    Ok(parsed)
}

fn parse_percentage_i64(value: &str) -> Result<i64, String> {
    let parsed = value
        .parse::<i64>()
        .map_err(|error| format!("invalid integer: {error}"))?;
    if !(0..=100).contains(&parsed) {
        return Err("percentage must be between zero and 100".to_owned());
    }
    Ok(parsed)
}

fn parse_percentage_f64(value: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|error| format!("invalid number: {error}"))?;
    if !parsed.is_finite() || !(0.0..=100.0).contains(&parsed) {
        return Err("percentage must be a finite number between zero and 100".to_owned());
    }
    Ok(parsed)
}

fn parse_required_source(value: &str) -> Result<String, String> {
    let canonical = match value.trim().to_ascii_lowercase().as_str() {
        "inventory" | "nvidia.inventory" => "nvidia.inventory",
        "processes" | "nvidia.processes" => "nvidia.processes",
        "optional" | "nvidia.optional" => "nvidia.optional",
        "topology" | "nvidia.topology" => "nvidia.topology",
        "xid" | "kernel.xid" => "kernel.xid",
        _ => {
            return Err(
                "unknown source; use inventory, processes, optional, topology, or xid".to_owned(),
            );
        }
    };
    Ok(canonical.to_owned())
}

fn write_serialized(value: &impl Serialize, format: CliFormat) -> Result<bool> {
    let mut output = match format {
        CliFormat::Json => serde_json::to_string_pretty(value)?,
        CliFormat::Ndjson => serde_json::to_string(value)?,
        CliFormat::Text => bail!("text output must use its workflow renderer"),
    };
    output.push('\n');
    write_stdout(&output)
}

pub(super) fn write_stdout(output: &str) -> Result<bool> {
    write_stdout_bytes(output.as_bytes())
}

fn write_stdout_bytes(output: &[u8]) -> Result<bool> {
    let mut stdout = std::io::stdout().lock();
    match stdout.write_all(output).and_then(|()| stdout.flush()) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == ErrorKind::BrokenPipe => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn resolve_canary_prompt(
    value: Option<String>,
    path: Option<&std::path::Path>,
) -> Result<(String, bool)> {
    const MAX_PROMPT_BYTES: u64 = 1024 * 1024;

    if let Some(value) = value {
        if value.trim().is_empty() {
            bail!("prompt cannot be empty");
        }
        return Ok((value, false));
    }
    let Some(path) = path else {
        return Ok((canary::DEFAULT_CANARY_PROMPT.to_owned(), true));
    };

    let file = open_read_nonblocking(path, false)
        .map_err(|error| anyhow::anyhow!("open prompt file {}: {error}", path.display()))?;
    let metadata = file
        .metadata()
        .map_err(|error| anyhow::anyhow!("inspect prompt file {}: {error}", path.display()))?;
    if !metadata.is_file() {
        bail!("prompt path {} is not a regular file", path.display());
    }
    validate_trusted_input_file(path, &metadata, "prompt file", 0o022)?;
    if metadata.len() > MAX_PROMPT_BYTES {
        bail!("prompt file {} exceeds 1 MiB", path.display());
    }

    let mut prompt = String::new();
    file.take(MAX_PROMPT_BYTES + 1)
        .read_to_string(&mut prompt)
        .map_err(|error| anyhow::anyhow!("read prompt file {}: {error}", path.display()))?;
    if prompt.len() as u64 > MAX_PROMPT_BYTES {
        bail!("prompt file {} exceeds 1 MiB", path.display());
    }
    if prompt.trim().is_empty() {
        bail!("prompt file {} is empty", path.display());
    }
    Ok((prompt, false))
}

fn resolve_canary_workload_id(
    requested: Option<String>,
    built_in_prompt: bool,
    cli_custom_prompt: bool,
    configured: Option<String>,
) -> Result<String> {
    if built_in_prompt {
        if requested.is_some() {
            bail!("--workload-id requires a custom --prompt or --prompt-file");
        }
        return Ok(canary::DEFAULT_CANARY_WORKLOAD_ID.to_owned());
    }
    let configured = if cli_custom_prompt { None } else { configured };
    requested.or(configured).ok_or_else(|| {
        anyhow::anyhow!(
            "a custom inference prompt requires an explicit --workload-id or paired canary.workload_id"
        )
    })
}

pub(super) fn resolve_secret(
    value: Option<String>,
    path: Option<&std::path::Path>,
) -> Result<Option<String>> {
    const MAX_SECRET_BYTES: u64 = 64 * 1024;
    if let Some(value) = value {
        if value.len() > usize::try_from(MAX_SECRET_BYTES).unwrap_or(usize::MAX) {
            bail!("secret value exceeds 64 KiB");
        }
        validate_secret_value(&value)?;
        return Ok(Some(value));
    }
    let Some(path) = path else {
        return Ok(None);
    };
    let file = open_read_nonblocking(path, false)
        .map_err(|error| anyhow::anyhow!("open secret file {}: {error}", path.display()))?;
    let metadata = file
        .metadata()
        .map_err(|error| anyhow::anyhow!("inspect secret file {}: {error}", path.display()))?;
    if !metadata.is_file() {
        bail!("secret path {} is not a regular file", path.display());
    }
    validate_secret_file_trust(path, &metadata)?;
    if metadata.len() > MAX_SECRET_BYTES {
        bail!("secret file {} exceeds 64 KiB", path.display());
    }
    let mut value = String::new();
    file.take(MAX_SECRET_BYTES + 1)
        .read_to_string(&mut value)
        .map_err(|error| anyhow::anyhow!("read secret file {}: {error}", path.display()))?;
    if value.len() > usize::try_from(MAX_SECRET_BYTES).unwrap_or(usize::MAX) {
        bail!("secret file {} exceeds 64 KiB", path.display());
    }
    let value = value.trim().to_owned();
    validate_secret_value(&value)
        .with_context(|| format!("invalid secret file {}", path.display()))?;
    Ok(Some(value))
}

fn validate_secret_value(value: &str) -> Result<()> {
    if value.is_empty() {
        bail!("secret value cannot be empty");
    }
    if !value.bytes().all(|byte| matches!(byte, b'!'..=b'~')) {
        bail!("secret value must contain only visible ASCII without whitespace");
    }
    Ok(())
}

#[cfg(unix)]
fn validate_secret_file_trust(path: &std::path::Path, metadata: &std::fs::Metadata) -> Result<()> {
    validate_trusted_input_file(path, metadata, "secret file", 0o037)
}

#[cfg(unix)]
fn validate_trusted_input_file(
    path: &std::path::Path,
    metadata: &std::fs::Metadata,
    label: &str,
    forbidden_mode: u32,
) -> Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let current_uid = uzers::get_current_uid();
    if metadata.permissions().mode() & forbidden_mode != 0 {
        bail!(
            "{label} {} has permissions that allow an untrusted principal to modify or access it",
            path.display()
        );
    }
    if !matches!(metadata.uid(), 0) && metadata.uid() != current_uid {
        bail!(
            "{label} {} must be owned by the current user or root",
            path.display()
        );
    }
    let absolute_path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .with_context(|| format!("resolve current directory for {label}"))?
            .join(path)
    };
    let mut component_path = PathBuf::new();
    for component in absolute_path.components() {
        component_path.push(component.as_os_str());
        let component_metadata = std::fs::symlink_metadata(&component_path).with_context(|| {
            format!(
                "inspect {label} path component {}",
                component_path.display()
            )
        })?;
        if !matches!(component_metadata.uid(), 0) && component_metadata.uid() != current_uid {
            bail!(
                "{label} path component {} must be owned by the current user or root",
                component_path.display()
            );
        }
        if component_metadata.is_dir() {
            validate_trusted_input_directory(
                &component_path,
                &component_metadata,
                current_uid,
                label,
            )?;
        }
    }

    let canonical = std::fs::canonicalize(&absolute_path)
        .with_context(|| format!("resolve {label} {}", path.display()))?;
    let canonical_metadata = std::fs::metadata(&canonical)
        .with_context(|| format!("inspect resolved {label} {}", canonical.display()))?;
    if metadata.dev() != canonical_metadata.dev() || metadata.ino() != canonical_metadata.ino() {
        bail!("{label} path changed while it was being opened");
    }
    reject_permissive_acl(&canonical, label)?;
    for ancestor in canonical
        .parent()
        .into_iter()
        .flat_map(std::path::Path::ancestors)
    {
        let ancestor_metadata = std::fs::metadata(ancestor)
            .with_context(|| format!("inspect {label} ancestor {}", ancestor.display()))?;
        validate_trusted_input_directory(ancestor, &ancestor_metadata, current_uid, label)?;
    }
    Ok(())
}

#[cfg(unix)]
fn validate_trusted_input_directory(
    path: &std::path::Path,
    metadata: &std::fs::Metadata,
    current_uid: u32,
    label: &str,
) -> Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    if !metadata.is_dir() {
        bail!(
            "{label} path ancestor {} is not a directory",
            path.display()
        );
    }
    if !matches!(metadata.uid(), 0) && metadata.uid() != current_uid {
        bail!(
            "{label} path ancestor {} must be owned by the current user or root",
            path.display()
        );
    }
    let mode = metadata.permissions().mode();
    if mode & 0o022 != 0 && mode & 0o1000 == 0 {
        bail!(
            "{label} path ancestor {} must not be group/world-writable without the sticky bit",
            path.display()
        );
    }
    reject_permissive_acl(path, &format!("{label} ancestor"))?;
    Ok(())
}

#[cfg(not(unix))]
fn validate_secret_file_trust(_: &std::path::Path, _: &std::fs::Metadata) -> Result<()> {
    Ok(())
}

#[cfg(not(unix))]
fn validate_trusted_input_file(
    _: &std::path::Path,
    _: &std::fs::Metadata,
    _: &str,
    _: u32,
) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clap_schema_is_internally_consistent_and_globals_work_on_both_sides() {
        Cli::command().debug_assert();
        let mut command = Cli::command();
        command.build();
        let doctor_help = command
            .find_subcommand_mut("doctor")
            .unwrap()
            .render_long_help()
            .to_string();
        assert!(doctor_help.contains("Load a profile for"));
        assert!(doctor_help.contains("rejected by other workflows"));
        assert!(Cli::try_parse_from(["gpu-watchman", "doctor", "--probe", "   "]).is_err());
        assert!(Cli::try_parse_from(["gpu-watchman", "bundle", "--probe", ",,"]).is_err());

        for arguments in [
            vec![
                "gpu-watchman",
                "--config",
                "profiles.toml",
                "--profile",
                "prod",
                "snapshot",
            ],
            vec![
                "gpu-watchman",
                "snapshot",
                "--config",
                "profiles.toml",
                "--profile",
                "prod",
            ],
        ] {
            let cli = Cli::try_parse_from(arguments).unwrap();
            assert_eq!(
                cli.config.as_deref(),
                Some(std::path::Path::new("profiles.toml"))
            );
            assert_eq!(cli.profile.as_deref(), Some("prod"));
            assert!(matches!(cli.command, Some(Commands::Snapshot(_))));
        }
    }

    #[test]
    fn benchmark_compare_is_offline_and_parses_fail_closed_gates() {
        let cli = Cli::try_parse_from([
            "gpu-watchman",
            "benchmark",
            "compare",
            "baseline.json",
            "candidate.ndjson",
            "--max-p95-e2e-regression-percent",
            "12.5",
            "--min-successful-rps-ratio",
            "0.95",
            "--min-completion-token-goodput-ratio",
            "0.9",
            "--max-error-percent-increase",
            "2",
            "--fail-on-regression",
            "--format",
            "ndjson",
        ])
        .unwrap();
        let Some(Commands::Benchmark(BenchmarkArgs {
            command: BenchmarkCommands::Compare(args),
        })) = cli.command
        else {
            panic!("expected benchmark compare arguments");
        };
        assert_eq!(args.max_p95_e2e_regression_percent, Some(12.5));
        assert_eq!(args.min_successful_rps_ratio, Some(0.95));
        assert_eq!(args.min_completion_token_goodput_ratio, Some(0.9));
        assert_eq!(args.max_error_percent_increase, Some(2.0));
        assert!(args.fail_on_regression);
        assert_eq!(args.format, CliFormat::Ndjson);

        let cli = Cli::try_parse_from([
            "gpu-watchman",
            "--config",
            "must-not-be-loaded.toml",
            "benchmark",
            "compare",
            "baseline.json",
            "candidate.json",
        ])
        .unwrap();
        let error = run(cli).unwrap_err().to_string();
        assert!(error.contains("benchmark saturation"));
        assert!(!error.contains("must-not-be-loaded"));
    }

    #[test]
    fn secret_files_are_trimmed_and_source_aliases_are_canonical() {
        #[cfg(unix)]
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("token");
        std::fs::write(&path, "  secret-value\n").unwrap();
        #[cfg(unix)]
        {
            let mut permissions = std::fs::metadata(&path).unwrap().permissions();
            permissions.set_mode(0o600);
            std::fs::set_permissions(&path, permissions).unwrap();
        }

        assert_eq!(
            resolve_secret(None, Some(&path)).unwrap().as_deref(),
            Some("secret-value")
        );
        assert!(resolve_secret(Some("contains whitespace".to_owned()), None).is_err());
        assert!(resolve_secret(Some("line\nbreak".to_owned()), None).is_err());
        assert!(resolve_secret(None, Some(directory.path())).is_err());
        let oversized = directory.path().join("oversized-token");
        std::fs::File::create(&oversized)
            .unwrap()
            .set_len(64 * 1024 + 1)
            .unwrap();
        assert!(resolve_secret(None, Some(&oversized)).is_err());
        assert_eq!(
            parse_required_source("processes").unwrap(),
            "nvidia.processes"
        );
        assert!(parse_required_source("typo").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn secret_files_reject_unsafe_writers_but_allow_projected_symlinks() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let directory = tempfile::tempdir().unwrap();
        let unsafe_path = directory.path().join("group-writable-token");
        std::fs::write(&unsafe_path, "unsafe").unwrap();
        let mut permissions = std::fs::metadata(&unsafe_path).unwrap().permissions();
        permissions.set_mode(0o660);
        std::fs::set_permissions(&unsafe_path, permissions).unwrap();
        assert!(resolve_secret(None, Some(&unsafe_path)).is_err());

        let unsafe_directory = directory.path().join("group-writable-secrets");
        std::fs::create_dir(&unsafe_directory).unwrap();
        let mut permissions = std::fs::metadata(&unsafe_directory).unwrap().permissions();
        permissions.set_mode(0o770);
        std::fs::set_permissions(&unsafe_directory, permissions).unwrap();
        let nested_token = unsafe_directory.join("token");
        std::fs::write(&nested_token, "nested-token\n").unwrap();
        let mut permissions = std::fs::metadata(&nested_token).unwrap().permissions();
        permissions.set_mode(0o600);
        std::fs::set_permissions(&nested_token, permissions).unwrap();
        assert!(resolve_secret(None, Some(&nested_token)).is_err());

        let projected_target = directory.path().join("projected-token-v2");
        let projected_link = directory.path().join("projected-token");
        std::fs::write(&projected_target, "safe-token\n").unwrap();
        let mut permissions = std::fs::metadata(&projected_target).unwrap().permissions();
        permissions.set_mode(0o640);
        std::fs::set_permissions(&projected_target, permissions).unwrap();
        symlink(&projected_target, &projected_link).unwrap();
        assert_eq!(
            resolve_secret(None, Some(&projected_link))
                .unwrap()
                .as_deref(),
            Some("safe-token")
        );
    }

    #[test]
    fn automatic_color_honors_terminal_safety_environment() {
        assert!(ColorChoice::Always.enabled(false));
        assert!(!ColorChoice::Never.enabled(true));
    }

    #[test]
    fn only_the_implicit_canary_prompt_enables_the_default_expectation() {
        let (prompt, built_in) = resolve_canary_prompt(None, None).unwrap();
        assert_eq!(prompt, canary::DEFAULT_CANARY_PROMPT);
        assert!(built_in);

        let (prompt, built_in) =
            resolve_canary_prompt(Some(canary::DEFAULT_CANARY_PROMPT.to_owned()), None).unwrap();
        assert_eq!(prompt, canary::DEFAULT_CANARY_PROMPT);
        assert!(!built_in);
    }

    #[test]
    fn canary_workload_identity_tracks_the_effective_prompt_source() {
        assert_eq!(
            resolve_canary_workload_id(None, true, false, None).unwrap(),
            canary::DEFAULT_CANARY_WORKLOAD_ID
        );
        assert!(
            resolve_canary_workload_id(Some("wrong-v1".to_owned()), true, false, None).is_err()
        );
        assert_eq!(
            resolve_canary_workload_id(None, false, false, Some("profile-smoke-v1".to_owned()))
                .unwrap(),
            "profile-smoke-v1"
        );
        assert!(
            resolve_canary_workload_id(None, false, true, Some("profile-smoke-v1".to_owned()))
                .is_err()
        );
        assert_eq!(
            resolve_canary_workload_id(
                Some("cli-smoke-v2".to_owned()),
                false,
                true,
                Some("profile-smoke-v1".to_owned())
            )
            .unwrap(),
            "cli-smoke-v2"
        );
    }

    #[test]
    fn monitor_does_not_load_credentials_when_no_probe_is_active() {
        let config = crate::config::parse_config(
            r#"config_version = 1
[profiles.local.monitor.inference]
token_file = "missing-token"
"#,
        )
        .unwrap();
        let profile = &config.profiles["local"];
        let resolved = monitor_args(
            CollectionCliArgs::default(),
            ProbeCliArgs::default(),
            HealthCliArgs::default(),
            None,
            Some(profile),
        )
        .unwrap();

        assert!(resolved.probe.is_empty());
        assert!(resolved.probe_token.is_none());
        assert!(resolved.probe_token_file.is_none());
    }
}
