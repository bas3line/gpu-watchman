//! Semantic validation for strict operational profiles.

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use url::Url;

use super::schema::{CONFIG_VERSION, CanaryProfileV1, ConfigV1, MonitorHealthProfileV1, ProfileV1};
use crate::domain::{MAX_CANARY_WORKLOAD_ID_BYTES, valid_canary_workload_id};
use crate::inference::probe::MAX_PROBE_TARGETS;
use crate::security::{listen_is_loopback, url_is_loopback};

const MAX_PROFILES: usize = 128;
const MAX_PROFILE_NAME_BYTES: usize = 64;
const MAX_IDENTITY_BYTES: usize = 64 << 10;
const MAX_EXPECTATION_BYTES: usize = 64 << 10;
const MAX_LIST_ITEMS: usize = 256;
const MAX_REQUEST_COUNT: u32 = 10_000;
const MAX_CONCURRENCY: u32 = 64;
const MAX_COMPLETION_TOKENS: u32 = 65_536;
const MAX_REQUESTED_COMPLETION_TOKENS: u64 = 1_000_000;
const MAX_REQUEST_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const MAX_PLANNED_DURATION: Duration = Duration::from_secs(15 * 60);

/// A selected profile and its stable name.
#[derive(Debug, Clone, Copy)]
pub struct SelectedProfile<'a> {
    pub name: &'a str,
    pub profile: &'a ProfileV1,
}

/// Validate the complete document and every profile without contacting external systems.
pub fn validate_config(config: &ConfigV1) -> Result<()> {
    if config.config_version != CONFIG_VERSION {
        bail!(
            "unsupported config_version {}; this release requires {CONFIG_VERSION}",
            config.config_version
        );
    }
    if config.profiles.len() > MAX_PROFILES {
        bail!("configuration contains more than {MAX_PROFILES} profiles");
    }
    for (name, profile) in &config.profiles {
        validate_profile_name(name)?;
        validate_profile(profile).with_context(|| format!("invalid profile {name:?}"))?;
    }
    if let Some(default_profile) = config.default_profile.as_deref() {
        validate_profile_name(default_profile).context("invalid default_profile")?;
        if !config.profiles.contains_key(default_profile) {
            bail!("default_profile {default_profile:?} does not exist");
        }
    }
    Ok(())
}

/// Select an explicit profile, or the document default when no explicit name is supplied.
pub fn select_profile<'a>(
    config: &'a ConfigV1,
    requested: Option<&str>,
) -> Result<Option<SelectedProfile<'a>>> {
    let Some(name) = requested.or(config.default_profile.as_deref()) else {
        return Ok(None);
    };
    validate_profile_name(name).context("invalid selected profile")?;
    let (stored_name, profile) = config
        .profiles
        .get_key_value(name)
        .with_context(|| format!("profile {name:?} does not exist"))?;
    Ok(Some(SelectedProfile {
        name: stored_name.as_str(),
        profile,
    }))
}

/// Validate a profile name accepted by CLI and configuration selection.
pub fn validate_profile_name(name: &str) -> Result<()> {
    if name.is_empty() || name.len() > MAX_PROFILE_NAME_BYTES {
        bail!("profile names must contain 1 to {MAX_PROFILE_NAME_BYTES} bytes");
    }
    let mut characters = name.chars();
    let first = characters.next().expect("non-empty name");
    if !first.is_ascii_alphanumeric()
        || !characters.all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-')
        })
    {
        bail!(
            "profile names must start with an ASCII letter or digit and contain only letters, digits, '.', '_', or '-'"
        );
    }
    Ok(())
}

fn validate_profile(profile: &ProfileV1) -> Result<()> {
    if let Some(monitor) = profile.monitor.as_ref() {
        if let Some(collection) = monitor.collection.as_ref() {
            if collection
                .command_timeout
                .is_some_and(|duration| duration.0.is_zero())
            {
                bail!("monitor.collection.command_timeout must be positive");
            }
            validate_optional_list(collection.gpus.as_deref(), "monitor.collection.gpus")?;
            if let Some(path) = collection.nvidia_smi.as_deref() {
                validate_path(path, "monitor.collection.nvidia_smi")?;
            }
        }
        if let Some(inference) = monitor.inference.as_ref() {
            if inference
                .timeout
                .is_some_and(|duration| duration.0.is_zero())
            {
                bail!("monitor.inference.timeout must be positive");
            }
            if let Some(urls) = inference.urls.as_deref() {
                if urls.len() > MAX_PROBE_TARGETS {
                    bail!("monitor.inference.urls contains more than {MAX_PROBE_TARGETS} entries");
                }
                for value in urls {
                    let url = validate_http_url(value, "monitor.inference.urls", false)?;
                    if url.scheme() == "http"
                        && !url_is_loopback(&url)
                        && inference.allow_insecure_http != Some(true)
                    {
                        bail!(
                            "monitor.inference.urls uses remote cleartext HTTP; set allow_insecure_http = true explicitly"
                        );
                    }
                }
            }
            if let Some(path) = inference.token_file.as_deref() {
                validate_path(path, "monitor.inference.token_file")?;
            }
        }
        if let Some(health) = monitor.health.as_ref() {
            validate_monitor_health(health)?;
        }
    }

    if let Some(service) = profile.service.as_ref() {
        if let Some(listen) = service.listen.as_deref() {
            validate_nonempty_bounded(listen, 4096, "service.listen")?;
            let loopback = listen_is_loopback(listen).map_err(anyhow::Error::msg)?;
            if !loopback && service.allow_remote_listen != Some(true) {
                bail!("service.listen is non-loopback; set allow_remote_listen = true explicitly");
            }
        }
        if service
            .interval
            .is_some_and(|duration| duration.0 < Duration::from_millis(500))
        {
            bail!("service.interval must be at least 500ms");
        }
        if service
            .freshness
            .is_some_and(|duration| duration.0.is_zero())
        {
            bail!("service.freshness must be positive");
        }
        for (path, field) in [
            (service.history_file.as_deref(), "service.history_file"),
            (service.api_token_file.as_deref(), "service.api_token_file"),
        ] {
            if let Some(path) = path {
                validate_path(path, field)?;
            }
        }
    }

    if let Some(canary) = profile.canary.as_ref() {
        validate_canary(canary)?;
    }
    Ok(())
}

fn validate_monitor_health(health: &MonitorHealthProfileV1) -> Result<()> {
    let vram_warning = health.vram_warning_percent.unwrap_or(90);
    let vram_critical = health.vram_critical_percent.unwrap_or(99);
    if !(0..=100).contains(&vram_warning)
        || !(0..=100).contains(&vram_critical)
        || vram_warning >= vram_critical
    {
        bail!("monitor VRAM thresholds must satisfy 0 <= warning < critical <= 100");
    }

    let temperature_warning = health.temperature_warning_c.unwrap_or(82);
    let temperature_critical = health.temperature_critical_c.unwrap_or(90);
    if temperature_warning >= temperature_critical {
        bail!("monitor temperature warning must be below critical");
    }

    let kv_warning = health.kv_cache_warning_percent.unwrap_or(85.0);
    let kv_critical = health.kv_cache_critical_percent.unwrap_or(95.0);
    if !kv_warning.is_finite()
        || !kv_critical.is_finite()
        || !(0.0..=100.0).contains(&kv_warning)
        || !(0.0..=100.0).contains(&kv_critical)
        || kv_warning >= kv_critical
    {
        bail!("monitor KV-cache thresholds must satisfy 0 <= warning < critical <= 100");
    }
    if health
        .process_growth_warning_mib
        .is_some_and(|value| value <= 0)
    {
        bail!("monitor process_growth_warning_mib must be positive");
    }
    if let Some(sources) = health.required_sources.as_deref() {
        if sources.len() > MAX_LIST_ITEMS {
            bail!("monitor.health.required_sources contains more than {MAX_LIST_ITEMS} entries");
        }
        for source in sources {
            canonical_source_name(source)?;
        }
    }
    Ok(())
}

fn validate_canary(canary: &CanaryProfileV1) -> Result<()> {
    if let Some(base_url) = canary.base_url.as_deref() {
        let url = validate_http_url(base_url, "canary.base_url", false)?;
        if url.scheme() == "http"
            && !url_is_loopback(&url)
            && canary.allow_insecure_http != Some(true)
        {
            bail!(
                "canary.base_url uses remote cleartext HTTP; set allow_insecure_http = true explicitly"
            );
        }
    }
    if let Some(model) = canary.model.as_deref() {
        validate_nonempty_bounded(model, MAX_IDENTITY_BYTES, "canary.model")?;
    }
    if let Some(workload_id) = canary.workload_id.as_deref()
        && !valid_canary_workload_id(workload_id)
    {
        bail!(
            "canary.workload_id must start with an ASCII letter or digit and use at most \
             {MAX_CANARY_WORKLOAD_ID_BYTES} bytes of letters, digits, dot, underscore, colon, \
             slash, or hyphen"
        );
    }
    if canary.prompt_file.is_some() != canary.workload_id.is_some() {
        bail!("canary.prompt_file and canary.workload_id must be configured together");
    }
    for (path, field) in [
        (canary.api_key_file.as_deref(), "canary.api_key_file"),
        (canary.prompt_file.as_deref(), "canary.prompt_file"),
    ] {
        if let Some(path) = path {
            validate_path(path, field)?;
        }
    }

    let max_tokens = canary.max_tokens.unwrap_or(16);
    let count = canary.count.unwrap_or(1);
    let concurrency = canary.concurrency.unwrap_or(1);
    let timeout = canary
        .timeout
        .map_or(Duration::from_secs(30), |duration| duration.0);
    if max_tokens == 0 || max_tokens > MAX_COMPLETION_TOKENS {
        bail!("canary.max_tokens must be between 1 and {MAX_COMPLETION_TOKENS}");
    }
    if count == 0 || count > MAX_REQUEST_COUNT {
        bail!("canary.count must be between 1 and {MAX_REQUEST_COUNT}");
    }
    if concurrency == 0 || concurrency > count || concurrency > MAX_CONCURRENCY {
        bail!("canary.concurrency must be between 1 and min(count, {MAX_CONCURRENCY})");
    }
    if u64::from(count).saturating_mul(u64::from(max_tokens)) > MAX_REQUESTED_COMPLETION_TOKENS {
        bail!(
            "canary.count multiplied by max_tokens must not exceed {MAX_REQUESTED_COMPLETION_TOKENS}"
        );
    }
    if timeout.is_zero() || timeout > MAX_REQUEST_TIMEOUT {
        bail!("canary.timeout must be positive and at most 5 minutes");
    }
    let waves = count.div_ceil(concurrency);
    if timeout.checked_mul(waves).unwrap_or(Duration::MAX) > MAX_PLANNED_DURATION {
        bail!("canary request waves multiplied by timeout must not exceed 15 minutes");
    }

    if let Some(slo) = canary.slo.as_ref() {
        if let Some(expectation) = slo.expect.as_deref() {
            validate_nonempty_bounded(expectation, MAX_EXPECTATION_BYTES, "canary.slo.expect")?;
        }
        if slo.max_ttft.is_some_and(|duration| duration.0.is_zero())
            || slo.max_e2e.is_some_and(|duration| duration.0.is_zero())
        {
            bail!("canary SLO durations must be positive");
        }
        if slo
            .min_output_tokens_per_second
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            bail!("canary.slo.min_output_tokens_per_second must be finite and non-negative");
        }
        if slo
            .min_success_percent
            .is_some_and(|value| !value.is_finite() || !(0.0..=100.0).contains(&value))
        {
            bail!("canary.slo.min_success_percent must be finite and between zero and 100");
        }
        if canary.stream == Some(false)
            && (slo.max_ttft.is_some() || slo.min_output_tokens_per_second.is_some())
        {
            bail!("canary TTFT and output-token-rate SLOs require stream = true");
        }
    }
    Ok(())
}

fn validate_http_url(value: &str, field: &str, allow_userinfo: bool) -> Result<Url> {
    validate_nonempty_bounded(value, MAX_IDENTITY_BYTES, field)?;
    let url = Url::parse(value).with_context(|| format!("{field} must be an absolute URL"))?;
    if !matches!(url.scheme(), "http" | "https") || url.host().is_none() {
        bail!("{field} must use HTTP or HTTPS and include a host");
    }
    if !allow_userinfo && (!url.username().is_empty() || url.password().is_some()) {
        bail!("{field} must not include URL user information");
    }
    Ok(url)
}

fn validate_nonempty_bounded(value: &str, max: usize, field: &str) -> Result<()> {
    if value.trim().is_empty() || value.len() > max {
        bail!("{field} must contain 1 to {max} bytes");
    }
    Ok(())
}

fn validate_path(path: &Path, field: &str) -> Result<()> {
    if path.as_os_str().is_empty() {
        bail!("{field} must not be empty");
    }
    Ok(())
}

fn validate_optional_list(values: Option<&[String]>, field: &str) -> Result<()> {
    let Some(values) = values else {
        return Ok(());
    };
    if values.len() > MAX_LIST_ITEMS {
        bail!("{field} contains more than {MAX_LIST_ITEMS} entries");
    }
    for value in values {
        validate_nonempty_bounded(value, 4096, field)?;
    }
    Ok(())
}

pub(super) fn canonical_source_name(value: &str) -> Result<&'static str> {
    match value.trim() {
        "inventory" | "nvidia.inventory" => Ok("nvidia.inventory"),
        "processes" | "nvidia.processes" => Ok("nvidia.processes"),
        "optional" | "nvidia.optional" => Ok("nvidia.optional"),
        "topology" | "nvidia.topology" => Ok("nvidia.topology"),
        "xid" | "kernel.xid" => Ok("kernel.xid"),
        _ => bail!(
            "unknown telemetry source; expected inventory, processes, optional, topology, or xid"
        ),
    }
}
