//! Bounded configuration I/O, deterministic path handling, and safe rendering.

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use url::Url;

use super::schema::{CONFIG_VERSION, ConfigV1};
use super::validation::{canonical_source_name, validate_config};
use crate::security::{open_read_nonblocking, reject_permissive_acl};

/// Maximum accepted size for one operational configuration document.
pub const MAX_CONFIG_BYTES: u64 = 1 << 20;

/// Safe starter document written by [`init_config`].
pub const CONFIG_TEMPLATE: &str = r#"config_version = 1
default_profile = "local"

[profiles.local.monitor.collection]
command_timeout = "3s"
nvidia_smi = "nvidia-smi"
collect_xid = true

[profiles.local.monitor.inference]
urls = ["http://127.0.0.1:8000"]
timeout = "3s"
allow_insecure_http = false

[profiles.local.monitor.health]
fail_on = "never"
required_sources = ["inventory", "processes"]
vram_warning_percent = 90
vram_critical_percent = 99
temperature_warning_c = 82
temperature_critical_c = 90
kv_cache_warning_percent = 85
kv_cache_critical_percent = 95
process_growth_warning_mib = 256

[profiles.local.service]
listen = "127.0.0.1:9400"
interval = "5s"
freshness = "2m"
quiet = true
allow_remote_listen = false

[profiles.local.canary]
base_url = "http://127.0.0.1:8000/v1"
max_tokens = 16
count = 1
concurrency = 1
timeout = "30s"
stream = true
allow_insecure_http = false

[profiles.local.canary.slo]
expect = "gpu-watchman-ok"
min_success_percent = 100
"#;

/// A validated document whose file references have been made deterministic.
#[derive(Debug, Clone)]
pub struct LoadedConfig {
    pub path: PathBuf,
    pub base_dir: PathBuf,
    pub config: ConfigV1,
}

/// Read, parse, validate, and normalize a configuration document.
pub fn load_config(path: &Path) -> Result<LoadedConfig> {
    let absolute_path = absolute_path(path)?;
    if std::fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
        bail!(
            "configuration {} must not be a symbolic link",
            path.display()
        );
    }
    let mut file = open_read_nonblocking(path, true)
        .with_context(|| format!("open configuration {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("inspect configuration {}", path.display()))?;
    if !metadata.is_file() {
        bail!(
            "configuration path {} is not a regular file",
            path.display()
        );
    }
    if metadata.len() > MAX_CONFIG_BYTES {
        bail!("configuration {} exceeds the 1 MiB limit", path.display());
    }
    let trusted_path = reject_unsafe_path(&absolute_path, &metadata)?;

    let mut bytes = Vec::with_capacity(
        usize::try_from(metadata.len().min(MAX_CONFIG_BYTES)).unwrap_or_default(),
    );
    Read::by_ref(&mut file)
        .take(MAX_CONFIG_BYTES + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("read configuration {}", path.display()))?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_CONFIG_BYTES {
        bail!("configuration {} exceeds the 1 MiB limit", path.display());
    }
    let body = String::from_utf8(bytes)
        .with_context(|| format!("configuration {} is not UTF-8", path.display()))?;
    let mut config =
        parse_config(&body).with_context(|| format!("invalid configuration {}", path.display()))?;
    let base_dir = trusted_path
        .parent()
        .unwrap_or_else(|| Path::new("/"))
        .to_path_buf();
    normalize_profile_paths(&mut config, &base_dir);
    normalize_source_names(&mut config)?;
    Ok(LoadedConfig {
        path: trusted_path,
        base_dir,
        config,
    })
}

/// Parse and validate a UTF-8 TOML document without performing filesystem access.
pub fn parse_config(body: &str) -> Result<ConfigV1> {
    let value: toml::Value =
        toml::from_str(body).map_err(|error| sanitized_toml_error("parse TOML", &error))?;
    let version = value
        .get("config_version")
        .and_then(toml::Value::as_integer)
        .context("config_version must be present as an integer")?;
    if version != i64::from(CONFIG_VERSION) {
        bail!("unsupported config_version {version}; this release requires {CONFIG_VERSION}");
    }
    let config: ConfigV1 = value
        .try_into()
        .map_err(|error| sanitized_toml_error("decode version-1 schema", &error))?;
    validate_config(&config)?;
    Ok(config)
}

fn sanitized_toml_error(action: &str, error: &toml::de::Error) -> anyhow::Error {
    let message = error.message();
    let category = if message.contains("unknown field") {
        "configuration contains an unknown field"
    } else if message.contains("duplicate field") || message.contains("duplicate key") {
        "configuration contains a duplicate field"
    } else if message.contains("missing field") {
        "configuration is missing a required field"
    } else if action == "parse TOML" {
        "configuration is not valid TOML"
    } else {
        "configuration does not match the version-1 schema"
    };
    error.span().map_or_else(
        || anyhow::anyhow!("{action}: {category}; source excerpt omitted"),
        |span| {
            anyhow::anyhow!(
                "{action}: {category} near byte offset {}; source excerpt omitted",
                span.start
            )
        },
    )
}

/// Serialize a defensive copy, removing URL user information, fragments, and entire queries.
///
/// Secret and prompt files are represented only by their paths. Their contents are never read.
pub fn safe_toml(config: &ConfigV1) -> Result<String> {
    let mut safe = config.clone();
    for profile in safe.profiles.values_mut() {
        if let Some(urls) = profile
            .monitor
            .as_mut()
            .and_then(|monitor| monitor.inference.as_mut())
            .and_then(|inference| inference.urls.as_mut())
        {
            for url in urls {
                *url = redact_url(url);
            }
        }
        if let Some(base_url) = profile
            .canary
            .as_mut()
            .and_then(|canary| canary.base_url.as_mut())
        {
            *base_url = redact_url(base_url);
        }
    }
    toml::to_string_pretty(&safe).context("serialize safe configuration")
}

/// Create a starter configuration without overwriting an existing path.
///
/// On Unix the file is created with mode `0600`. Secret files remain separate references.
pub fn init_config(path: &Path) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create configuration directory {}", parent.display()))?;
    }
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    configure_private_mode(&mut options);
    let mut file = options
        .open(path)
        .with_context(|| format!("create configuration {}", path.display()))?;
    file.write_all(CONFIG_TEMPLATE.as_bytes())
        .with_context(|| format!("write configuration {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("sync configuration {}", path.display()))?;
    Ok(())
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()
            .context("resolve current directory for configuration")?
            .join(path))
    }
}

fn normalize_profile_paths(config: &mut ConfigV1, base_dir: &Path) {
    for profile in config.profiles.values_mut() {
        if let Some(collection) = profile
            .monitor
            .as_mut()
            .and_then(|monitor| monitor.collection.as_mut())
            && let Some(command) = collection.nvidia_smi.as_mut()
        {
            *command = resolve_command_path(base_dir, command);
        }
        if let Some(token_file) = profile
            .monitor
            .as_mut()
            .and_then(|monitor| monitor.inference.as_mut())
            .and_then(|inference| inference.token_file.as_mut())
        {
            *token_file = resolve_data_path(base_dir, token_file);
        }
        if let Some(service) = profile.service.as_mut() {
            if let Some(path) = service.history_file.as_mut() {
                *path = resolve_data_path(base_dir, path);
            }
            if let Some(path) = service.api_token_file.as_mut() {
                *path = resolve_data_path(base_dir, path);
            }
        }
        if let Some(canary) = profile.canary.as_mut() {
            if let Some(path) = canary.api_key_file.as_mut() {
                *path = resolve_data_path(base_dir, path);
            }
            if let Some(path) = canary.prompt_file.as_mut() {
                *path = resolve_data_path(base_dir, path);
            }
        }
    }
}

fn normalize_source_names(config: &mut ConfigV1) -> Result<()> {
    for profile in config.profiles.values_mut() {
        let Some(sources) = profile
            .monitor
            .as_mut()
            .and_then(|monitor| monitor.health.as_mut())
            .and_then(|health| health.required_sources.as_mut())
        else {
            continue;
        };
        for source in sources {
            *source = canonical_source_name(source)?.to_owned();
        }
    }
    Ok(())
}

fn resolve_data_path(base_dir: &Path, path: &Path) -> PathBuf {
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    };
    resolved.components().collect()
}

fn resolve_command_path(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() || is_bare_command(path) {
        path.to_path_buf()
    } else {
        resolve_data_path(base_dir, path)
    }
}

fn is_bare_command(path: &Path) -> bool {
    let value = path.to_string_lossy();
    !value.contains('/') && !value.contains('\\')
}

fn redact_url(value: &str) -> String {
    let Ok(mut url) = Url::parse(value) else {
        return "<invalid-url>".to_owned();
    };
    if url.cannot_be_a_base() || !matches!(url.scheme(), "http" | "https") {
        return "<invalid-url>".to_owned();
    }
    if url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
    {
        return value.to_owned();
    }
    let _ = url.set_password(None);
    let _ = url.set_username("");
    if url.query().is_some() {
        url.set_query(Some("REDACTED"));
    }
    url.set_fragment(None);
    url.to_string()
}

#[cfg(unix)]
fn configure_private_mode(options: &mut OpenOptions) {
    use std::os::unix::fs::OpenOptionsExt;
    options.mode(0o600);
}

#[cfg(not(unix))]
fn configure_private_mode(_: &mut OpenOptions) {}

#[cfg(unix)]
fn reject_unsafe_path(path: &Path, metadata: &std::fs::Metadata) -> Result<PathBuf> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let current_uid = uzers::get_current_uid();
    if metadata.permissions().mode() & 0o022 != 0 {
        bail!(
            "configuration {} must not be writable by group or other users",
            path.display()
        );
    }
    if !matches!(metadata.uid(), 0) && metadata.uid() != current_uid {
        bail!(
            "configuration {} must be owned by the current user or root",
            path.display()
        );
    }
    if std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect configuration path {}", path.display()))?
        .file_type()
        .is_symlink()
    {
        bail!(
            "configuration {} must not be a symbolic link",
            path.display()
        );
    }

    let canonical = std::fs::canonicalize(path)
        .with_context(|| format!("resolve configuration path {}", path.display()))?;
    let canonical_metadata = std::fs::metadata(&canonical)
        .with_context(|| format!("inspect resolved configuration {}", canonical.display()))?;
    if metadata.dev() != canonical_metadata.dev() || metadata.ino() != canonical_metadata.ino() {
        bail!("configuration path changed while it was being opened");
    }
    reject_permissive_acl(&canonical, "configuration")?;
    for ancestor in canonical.parent().into_iter().flat_map(Path::ancestors) {
        let ancestor_metadata = std::fs::metadata(ancestor)
            .with_context(|| format!("inspect configuration ancestor {}", ancestor.display()))?;
        let mode = ancestor_metadata.permissions().mode();
        let trusted_owner =
            matches!(ancestor_metadata.uid(), 0) || ancestor_metadata.uid() == current_uid;
        if !trusted_owner {
            bail!(
                "configuration ancestor {} must be owned by the current user or root",
                ancestor.display()
            );
        }
        if mode & 0o022 != 0 && mode & 0o1000 == 0 {
            bail!(
                "configuration ancestor {} must not be group/world-writable without the sticky bit",
                ancestor.display()
            );
        }
        reject_permissive_acl(ancestor, "configuration ancestor")?;
    }
    Ok(canonical)
}

#[cfg(not(unix))]
fn reject_unsafe_path(path: &Path, _: &std::fs::Metadata) -> Result<PathBuf> {
    std::fs::canonicalize(path)
        .with_context(|| format!("resolve configuration path {}", path.display()))
}
