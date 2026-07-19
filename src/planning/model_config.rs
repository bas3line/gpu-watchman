//! Hugging Face model configuration loading and geometry derivation.

use std::fmt;
use std::fs::File;
use std::io::{Read, Take};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::security::open_read_nonblocking;

/// Maximum accepted size for a local model configuration.
pub const MAX_CONFIG_BYTES: u64 = 8 * 1024 * 1024;
/// Maximum retained length of the public model-type identifier.
pub const MAX_MODEL_TYPE_BYTES: usize = 128;

const AUDITED_DENSE_GATED_MODEL_TYPES: &[&str] = &["llama", "mistral"];
const SPARSE_ROUTING_MARKERS: &[&str] = &[
    "num_experts_per_tok",
    "num_selected_experts",
    "experts_per_token",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "n_shared_experts",
    "num_shared_experts",
    "router_aux_loss_coef",
    "decoder_sparse_step",
    "first_k_dense_replace",
    "routed_scaling_factor",
    "topk_method",
];

/// Model dimensions needed by the capacity planner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelGeometry {
    /// Total model parameters, expressed in billions.
    pub parameters_billion: f64,
    /// Provenance of the parameter count used by the capacity estimate.
    #[serde(default)]
    pub parameter_source: ModelParameterSource,
    /// Number of transformer layers.
    pub layers: u32,
    /// Number of key/value attention heads retained in the KV cache.
    pub kv_heads: u32,
    /// Dimension of each attention head.
    pub head_dim: u32,
    /// Hugging Face model type identifier.
    pub model_type: String,
    /// Routed expert count detected in the config, excluding a dense count of one.
    #[serde(default)]
    pub expert_count: Option<u32>,
    /// Whether model type, expert count, or sparse-routing markers identify `MoE`.
    #[serde(default)]
    pub is_moe: bool,
    /// Assumptions and limitations that affect the derived values.
    pub caveats: Vec<String>,
}

/// Privacy-safe provenance for a capacity-planning parameter count.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelParameterSource {
    /// Legacy input did not retain enough information to identify provenance.
    #[default]
    Unknown,
    /// The caller explicitly supplied the parameter count.
    ExplicitOverride,
    /// `config.json` declared `num_parameters` or `parameter_count`.
    ConfigDeclared,
    /// The parameter count was estimated from effective dense-model geometry.
    DenseEstimate,
}

/// Bounded, parsed evidence from a Hugging Face `config.json`.
///
/// The configuration is deliberately opaque and is neither serializable nor
/// exposed through `Debug`: real-world configuration files can contain remote
/// repository URLs, access hints, or other deployment-specific values that
/// must not leak into reports.
pub struct ModelConfigEvidence {
    object: Map<String, Value>,
}

impl fmt::Debug for ModelConfigEvidence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelConfigEvidence")
            .field("config", &"<redacted>")
            .finish()
    }
}

/// Explicit capacity-planning geometry overrides.
///
/// Overrides take precedence over configuration aliases before dependent
/// geometry checks and parameter estimation run. This is important for `MoE`
/// models, whose parameter count cannot be estimated safely from dense-model
/// fields, and when operators intentionally model geometry different from the
/// checkpoint defaults.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ModelGeometryOverrides {
    /// Explicit model parameter count in billions.
    pub parameters_billion: Option<f64>,
    /// Explicit transformer layer count.
    pub layers: Option<u32>,
    /// Explicit KV head count.
    pub kv_heads: Option<u32>,
    /// Explicit attention head dimension.
    pub head_dim: Option<u32>,
}

/// Load bounded, opaque evidence from a local Hugging Face `config.json`.
///
/// This performs only file, size, JSON, and top-level object validation. Call
/// [`model_geometry_from_evidence`] with all explicit CLI overrides afterward
/// so those overrides are applied before dependent derivation and validation.
pub fn load_model_config_evidence(path: impl AsRef<Path>) -> Result<ModelConfigEvidence> {
    let path = path.as_ref();
    let file = open_read_nonblocking(path, false)
        .with_context(|| format!("failed to open model config {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to inspect model config {}", path.display()))?;
    ensure!(
        metadata.is_file(),
        "model config is not a regular file: {}",
        path.display()
    );
    ensure!(
        metadata.len() <= MAX_CONFIG_BYTES,
        "model config exceeds the {} MiB limit: {}",
        MAX_CONFIG_BYTES / (1024 * 1024),
        path.display()
    );

    let mut bytes = Vec::with_capacity(usize::try_from(metadata.len()).unwrap_or(0));
    let mut reader: Take<File> = file.take(MAX_CONFIG_BYTES + 1);
    reader
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read model config {}", path.display()))?;
    ensure!(
        u64::try_from(bytes.len()).unwrap_or(u64::MAX) <= MAX_CONFIG_BYTES,
        "model config exceeds the {} MiB limit: {}",
        MAX_CONFIG_BYTES / (1024 * 1024),
        path.display()
    );

    let object: Map<String, Value> = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid JSON in model config {}", path.display()))?;
    Ok(ModelConfigEvidence { object })
}

/// Load a local Hugging Face `config.json` and derive capacity-planning geometry.
///
/// Reads are capped even if the file grows after its metadata is inspected.
pub fn load_model_geometry(path: impl AsRef<Path>) -> Result<ModelGeometry> {
    let path = path.as_ref();
    let evidence = load_model_config_evidence(path)?;
    model_geometry_from_evidence(&evidence, ModelGeometryOverrides::default())
        .with_context(|| format!("invalid model geometry in {}", path.display()))
}

/// Derive capacity-planning geometry from a parsed Hugging Face configuration.
pub fn model_geometry_from_value(config: &Value) -> Result<ModelGeometry> {
    let object = config
        .as_object()
        .context("model config must contain a JSON object")?;
    model_geometry_from_object(object, ModelGeometryOverrides::default())
}

/// Resolve capacity-planning geometry from raw config evidence and overrides.
///
/// Explicit values are selected before validation and before any dense-model
/// parameter estimate is computed. Consequently, a parameter override can be
/// used with an `MoE` config, and layer/KV/head-dimension overrides participate in
/// any derived parameter count instead of leaving it based on stale geometry.
pub fn model_geometry_from_evidence(
    evidence: &ModelConfigEvidence,
    overrides: ModelGeometryOverrides,
) -> Result<ModelGeometry> {
    model_geometry_from_object(&evidence.object, overrides)
}

fn model_geometry_from_object(
    object: &Map<String, Value>,
    overrides: ModelGeometryOverrides,
) -> Result<ModelGeometry> {
    let model_type = required_non_empty_string(object, "model_type")?;
    validate_overrides(overrides)?;
    let expert_count = optional_u32(
        object,
        &["num_local_experts", "num_experts", "n_routed_experts"],
        "expert count",
    )?
    .filter(|count| *count > 1);
    let is_moe = expert_count.is_some()
        || model_type_indicates_moe(&model_type)
        || contains_sparse_routing_marker(object);
    if is_moe && overrides.parameters_billion.is_none() {
        bail!(
            "mixture-of-experts configs require an explicit total-resident parameter count override; config parameter fields may describe active rather than resident weights"
        );
    }

    let mut caveats = Vec::new();
    let layers = select_u32_override(
        overrides.layers,
        object,
        &["num_hidden_layers", "n_layer", "num_layers"],
        "layer count",
        "Layer count",
        &mut caveats,
    )?;
    let attention_heads = required_u32(
        object,
        &["num_attention_heads", "n_head"],
        "attention head count",
    )?;
    let hidden_size = required_u64(object, &["hidden_size", "n_embd", "d_model"], "hidden size")?;

    let kv_heads = if let Some(value) = overrides.kv_heads {
        caveats.push("KV head count was explicitly overridden.".to_owned());
        value
    } else if let Some(value) = optional_u32(object, &["num_key_value_heads"], "KV head count")? {
        value
    } else {
        caveats.push(
            "num_key_value_heads is absent; assuming one KV head per attention head.".to_owned(),
        );
        attention_heads
    };
    ensure!(
        kv_heads <= attention_heads,
        "KV head count ({kv_heads}) exceeds attention head count ({attention_heads})"
    );
    ensure!(
        attention_heads % kv_heads == 0,
        "attention head count ({attention_heads}) is not divisible by KV head count ({kv_heads})"
    );

    let head_dim = if let Some(value) = overrides.head_dim {
        caveats.push("Attention head dimension was explicitly overridden.".to_owned());
        value
    } else if let Some(value) = optional_u32(object, &["head_dim"], "head dimension")? {
        value
    } else {
        let attention_heads_u64 = u64::from(attention_heads);
        ensure!(
            hidden_size % attention_heads_u64 == 0,
            "hidden size ({hidden_size}) is not divisible by attention head count ({attention_heads}) and head_dim is absent"
        );
        let derived = hidden_size / attention_heads_u64;
        caveats.push(
            "head_dim is absent; derived it as hidden size divided by attention heads.".to_owned(),
        );
        u32::try_from(derived).context("derived head dimension exceeds the supported range")?
    };

    let (parameters_billion, parameter_source) = resolve_parameter_count(
        object,
        overrides,
        TransformerGeometry {
            layers,
            attention_heads,
            kv_heads,
            head_dim,
            hidden_size,
        },
        &model_type,
        &mut caveats,
    )?;
    ensure!(
        parameters_billion.is_finite() && parameters_billion > 0.0,
        "derived parameter count must be finite and greater than zero"
    );

    Ok(ModelGeometry {
        parameters_billion,
        parameter_source,
        layers,
        kv_heads,
        head_dim,
        model_type,
        expert_count,
        is_moe,
        caveats,
    })
}

fn resolve_parameter_count(
    object: &Map<String, Value>,
    overrides: ModelGeometryOverrides,
    geometry: TransformerGeometry,
    model_type: &str,
    caveats: &mut Vec<String>,
) -> Result<(f64, ModelParameterSource)> {
    if let Some(parameters_billion) = overrides.parameters_billion {
        caveats.push(
            "Parameter count was explicitly overridden; config parameter metadata and automatic estimation were not used."
                .to_owned(),
        );
        Ok((parameters_billion, ModelParameterSource::ExplicitOverride))
    } else {
        derive_parameters_billion(
            object,
            geometry,
            model_type,
            overrides.layers.is_some()
                || overrides.kv_heads.is_some()
                || overrides.head_dim.is_some(),
            caveats,
        )
    }
}

#[cfg(test)]
fn model_config_evidence_from_value(config: &Value) -> Result<ModelConfigEvidence> {
    let object = config
        .as_object()
        .context("model config must contain a JSON object")?;
    Ok(ModelConfigEvidence {
        object: object.clone(),
    })
}

fn validate_overrides(overrides: ModelGeometryOverrides) -> Result<()> {
    if let Some(parameters_billion) = overrides.parameters_billion {
        ensure!(
            parameters_billion.is_finite() && parameters_billion > 0.0,
            "parameter count override must be finite and greater than zero"
        );
    }
    for (label, value) in [
        ("layer count", overrides.layers),
        ("KV head count", overrides.kv_heads),
        ("head dimension", overrides.head_dim),
    ] {
        if let Some(value) = value {
            ensure!(value > 0, "{label} override must be greater than zero");
        }
    }
    Ok(())
}

fn select_u32_override(
    value: Option<u32>,
    object: &Map<String, Value>,
    aliases: &[&str],
    label: &str,
    caveat_label: &str,
    caveats: &mut Vec<String>,
) -> Result<u32> {
    if let Some(value) = value {
        caveats.push(format!("{caveat_label} was explicitly overridden."));
        Ok(value)
    } else {
        required_u32(object, aliases, label)
    }
}

fn derive_parameters_billion(
    object: &Map<String, Value>,
    geometry: TransformerGeometry,
    model_type: &str,
    geometry_overridden: bool,
    caveats: &mut Vec<String>,
) -> Result<(f64, ModelParameterSource)> {
    if let Some(parameters) = optional_u64(
        object,
        &["num_parameters", "parameter_count"],
        "parameter count",
    )? {
        caveats.push(
            "Parameter count comes from config.json and is not verified against checkpoint files."
                .to_owned(),
        );
        if geometry_overridden {
            caveats.push(
                "The config parameter count is independent of explicit geometry overrides; verify that it still matches the intended checkpoint."
                    .to_owned(),
            );
        }
        return Ok((
            parameters_to_billions(parameters),
            ModelParameterSource::ConfigDeclared,
        ));
    }

    let intermediate_size = required_u64(
        object,
        &["intermediate_size", "n_inner"],
        "intermediate size",
    )?;
    let vocab_size = required_u64(object, &["vocab_size"], "vocabulary size")?;
    let tied_embeddings = optional_bool(object, "tie_word_embeddings")?.unwrap_or_else(|| {
        caveats.push(
            "tie_word_embeddings is absent; assuming untied embeddings to avoid underestimating weights."
                .to_owned(),
        );
        false
    });
    let gated_mlp = gated_mlp_layout(object, model_type)?;
    let estimated = estimate_dense_parameters(DenseEstimateInput {
        layers: geometry.layers,
        attention_heads: geometry.attention_heads,
        kv_heads: geometry.kv_heads,
        head_dim: geometry.head_dim,
        hidden_size: geometry.hidden_size,
        intermediate_size,
        vocab_size,
        tied_embeddings,
        gated_mlp,
    })?;
    caveats.push(format!(
        "Parameter count is a dense architecture estimate ({}) from embeddings, attention, MLP, and normalization weights; checkpoint metadata is authoritative.",
        if gated_mlp { "gated MLP" } else { "two-projection MLP" }
    ));
    Ok((estimated, ModelParameterSource::DenseEstimate))
}

#[allow(clippy::cast_precision_loss)]
fn parameters_to_billions(parameters: u64) -> f64 {
    parameters as f64 / 1_000_000_000.0
}

#[derive(Debug, Clone, Copy)]
struct TransformerGeometry {
    layers: u32,
    attention_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    hidden_size: u64,
}

#[derive(Debug, Clone, Copy)]
struct DenseEstimateInput {
    layers: u32,
    attention_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    hidden_size: u64,
    intermediate_size: u64,
    vocab_size: u64,
    tied_embeddings: bool,
    gated_mlp: bool,
}

#[allow(clippy::cast_precision_loss)]
fn estimate_dense_parameters(input: DenseEstimateInput) -> Result<f64> {
    let hidden = u128::from(input.hidden_size);
    let intermediate = u128::from(input.intermediate_size);
    let vocab = u128::from(input.vocab_size);
    let q_width = u128::from(input.attention_heads)
        .checked_mul(u128::from(input.head_dim))
        .context("attention width overflowed")?;
    let kv_width = u128::from(input.kv_heads)
        .checked_mul(u128::from(input.head_dim))
        .context("KV attention width overflowed")?;

    let embeddings = checked_product(&[vocab, hidden], "embedding parameter count")?;
    // Q + output projections use q_width; K + V use kv_width.
    let attention = checked_product(
        &[
            2,
            hidden,
            q_width
                .checked_add(kv_width)
                .context("attention parameter count overflowed")?,
        ],
        "attention parameter count",
    )?;
    let mlp = checked_product(
        &[if input.gated_mlp { 3 } else { 2 }, hidden, intermediate],
        "MLP parameter count",
    )?;
    // Two per-layer normalization vectors plus a final normalization vector.
    let per_layer = attention
        .checked_add(mlp)
        .and_then(|value| value.checked_add(2 * hidden))
        .context("per-layer parameter count overflowed")?;
    let transformer = per_layer
        .checked_mul(u128::from(input.layers))
        .context("transformer parameter count overflowed")?;
    let output_head = if input.tied_embeddings { 0 } else { embeddings };
    let total = embeddings
        .checked_add(transformer)
        .and_then(|value| value.checked_add(hidden))
        .and_then(|value| value.checked_add(output_head))
        .context("model parameter count overflowed")?;
    let parameters_billion = total as f64 / 1_000_000_000.0;
    ensure!(
        parameters_billion.is_finite() && parameters_billion > 0.0,
        "estimated parameter count must be finite and greater than zero"
    );
    Ok(parameters_billion)
}

fn checked_product(values: &[u128], label: &str) -> Result<u128> {
    values.iter().try_fold(1_u128, |product, value| {
        product
            .checked_mul(*value)
            .with_context(|| format!("{label} overflowed"))
    })
}

fn model_type_indicates_moe(model_type: &str) -> bool {
    let model_type = model_type.to_ascii_lowercase();
    model_type == "mixtral"
        || model_type == "dbrx"
        || model_type == "jamba"
        || model_type.contains("_moe")
        || model_type.ends_with("moe")
        || model_type.starts_with("deepseek_v2")
        || model_type.starts_with("deepseek_v3")
}

fn contains_sparse_routing_marker(object: &Map<String, Value>) -> bool {
    let mut pending: Vec<&Value> = object.values().collect();
    if object
        .keys()
        .any(|key| SPARSE_ROUTING_MARKERS.contains(&key.as_str()))
    {
        return true;
    }
    while let Some(value) = pending.pop() {
        match value {
            Value::Object(nested) => {
                if nested
                    .keys()
                    .any(|key| SPARSE_ROUTING_MARKERS.contains(&key.as_str()))
                {
                    return true;
                }
                pending.extend(nested.values());
            }
            Value::Array(values) => pending.extend(values),
            Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
        }
    }
    false
}

fn gated_mlp_layout(object: &Map<String, Value>, model_type: &str) -> Result<bool> {
    let model_type = model_type.to_ascii_lowercase();
    ensure!(
        AUDITED_DENSE_GATED_MODEL_TYPES.contains(&model_type.as_str()),
        "automatic parameter estimation is not audited for model_type={model_type}; provide an explicit --params value or a config-declared total parameter count"
    );
    if optional_bool(object, "attention_bias")?.unwrap_or(false)
        || optional_bool(object, "mlp_bias")?.unwrap_or(false)
    {
        bail!(
            "automatic parameter estimation does not support attention or MLP biases; provide an explicit --params value"
        );
    }
    if let Some(projection) = optional_string(object, "feed_forward_proj")? {
        let projection = projection.to_ascii_lowercase();
        ensure!(
            projection.contains("gated") || projection.contains("glu"),
            "automatic parameter estimation expected a gated MLP for model_type={model_type}; provide an explicit --params value"
        );
    }
    Ok(true)
}

fn required_non_empty_string(object: &Map<String, Value>, key: &str) -> Result<String> {
    let value = object
        .get(key)
        .with_context(|| format!("missing required field {key}"))?;
    let text = value
        .as_str()
        .with_context(|| format!("{key} must be a string"))?
        .trim();
    ensure!(!text.is_empty(), "{key} must not be empty");
    if key == "model_type" {
        ensure!(
            text.len() <= MAX_MODEL_TYPE_BYTES,
            "model_type exceeds the {MAX_MODEL_TYPE_BYTES}-byte limit"
        );
        ensure!(
            text.bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.')),
            "model_type must be an ASCII identifier containing only letters, numbers, '.', '-', or '_'"
        );
    }
    Ok(text.to_owned())
}

fn optional_string<'a>(object: &'a Map<String, Value>, key: &str) -> Result<Option<&'a str>> {
    object
        .get(key)
        .map(|value| {
            value
                .as_str()
                .with_context(|| format!("{key} must be a string"))
        })
        .transpose()
}

fn optional_bool(object: &Map<String, Value>, key: &str) -> Result<Option<bool>> {
    object
        .get(key)
        .map(|value| {
            value
                .as_bool()
                .with_context(|| format!("{key} must be a boolean"))
        })
        .transpose()
}

fn required_u32(object: &Map<String, Value>, aliases: &[&str], label: &str) -> Result<u32> {
    let value = required_u64(object, aliases, label)?;
    u32::try_from(value).with_context(|| format!("{label} exceeds the supported range"))
}

fn optional_u32(object: &Map<String, Value>, aliases: &[&str], label: &str) -> Result<Option<u32>> {
    optional_u64(object, aliases, label)?
        .map(|value| {
            u32::try_from(value).with_context(|| format!("{label} exceeds the supported range"))
        })
        .transpose()
}

fn required_u64(object: &Map<String, Value>, aliases: &[&str], label: &str) -> Result<u64> {
    optional_u64(object, aliases, label)?
        .with_context(|| format!("missing required {label} ({})", aliases.join(" or ")))
}

fn optional_u64(object: &Map<String, Value>, aliases: &[&str], label: &str) -> Result<Option<u64>> {
    let mut selected: Option<(&str, u64)> = None;
    for alias in aliases {
        let Some(value) = object.get(*alias) else {
            continue;
        };
        let number = positive_integral_u64(value, alias)?;
        if let Some((selected_alias, selected_value)) = selected {
            ensure!(
                number == selected_value,
                "conflicting {label}: {selected_alias}={selected_value}, {alias}={number}"
            );
        } else {
            selected = Some((alias, number));
        }
    }
    Ok(selected.map(|(_, value)| value))
}

#[allow(clippy::cast_precision_loss)]
fn positive_integral_u64(value: &Value, key: &str) -> Result<u64> {
    if let Some(number) = value.as_u64() {
        ensure!(number > 0, "{key} must be greater than zero");
        return Ok(number);
    }
    let number = value
        .as_f64()
        .with_context(|| format!("{key} must be a positive integer"))?;
    ensure!(
        number.is_finite() && number > 0.0 && number.fract() == 0.0,
        "{key} must be a finite positive integer"
    );
    ensure!(
        number <= u64::MAX as f64,
        "{key} exceeds the supported range"
    );
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let number = number as u64;
    Ok(number)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use serde_json::json;
    use tempfile::NamedTempFile;

    use super::*;

    #[test]
    fn uses_configured_llama_parameter_count_and_geometry() {
        let geometry = model_geometry_from_value(&json!({
            "model_type": "llama",
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "head_dim": 128,
            "num_parameters": 7_000_000_000_u64
        }))
        .expect("valid geometry");

        assert!((geometry.parameters_billion - 7.0).abs() < f64::EPSILON);
        assert_eq!(
            geometry.parameter_source,
            ModelParameterSource::ConfigDeclared
        );
        assert_eq!(geometry.layers, 32);
        assert_eq!(geometry.kv_heads, 8);
        assert_eq!(geometry.head_dim, 128);
        assert_eq!(geometry.model_type, "llama");
        assert_eq!(geometry.expert_count, None);
        assert!(!geometry.is_moe);
        assert!(
            geometry
                .caveats
                .iter()
                .any(|item| item.contains("not verified"))
        );
    }

    #[test]
    fn estimates_gated_model_with_derived_head_dimension() {
        let geometry = model_geometry_from_value(&json!({
            "model_type": "mistral",
            "num_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "d_model": 1024,
            "intermediate_size": 4096,
            "vocab_size": 32_000,
            "tie_word_embeddings": true
        }))
        .expect("valid geometry");

        assert_eq!(geometry.layers, 2);
        assert_eq!(geometry.kv_heads, 2);
        assert_eq!(geometry.head_dim, 128);
        assert!(geometry.parameters_billion > 0.06);
        assert!(geometry.parameters_billion < 0.07);
        assert_eq!(
            geometry.parameter_source,
            ModelParameterSource::DenseEstimate
        );
        assert!(
            geometry
                .caveats
                .iter()
                .any(|item| item.contains("gated MLP"))
        );
    }

    #[test]
    fn accepts_gpt_aliases_and_falls_back_to_attention_heads_for_kv() {
        let geometry = model_geometry_from_value(&json!({
            "model_type": "gpt2",
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "n_inner": 3072,
            "vocab_size": 50_257,
            "tie_word_embeddings": true,
            "num_parameters": 124_000_000
        }))
        .expect("valid geometry");

        assert_eq!(geometry.layers, 12);
        assert_eq!(geometry.kv_heads, 12);
        assert_eq!(geometry.head_dim, 64);
        assert!(
            geometry
                .caveats
                .iter()
                .any(|item| item.contains("one KV head per attention head"))
        );
        assert_eq!(
            geometry.parameter_source,
            ModelParameterSource::ConfigDeclared
        );
    }

    #[test]
    fn rejects_conflicting_aliases_and_inconsistent_geometry() {
        let conflicting = model_geometry_from_value(&json!({
            "model_type": "llama",
            "num_hidden_layers": 32,
            "n_layer": 24,
            "num_attention_heads": 32,
            "hidden_size": 4096,
            "num_parameters": 1_000_000_000_u64
        }))
        .expect_err("conflicting layer aliases must fail");
        assert!(conflicting.to_string().contains("conflicting layer count"));

        let inconsistent = model_geometry_from_value(&json!({
            "model_type": "llama",
            "num_hidden_layers": 32,
            "num_attention_heads": 30,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "num_parameters": 1_000_000_000_u64
        }))
        .expect_err("inconsistent head counts must fail");
        assert!(inconsistent.to_string().contains("not divisible"));

        let evidence = model_config_evidence_from_value(&json!({
            "model_type": "llama",
            "num_hidden_layers": 32,
            "n_layer": 24,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "num_parameters": 1_000_000_000_u64
        }))
        .expect("raw evidence accepts unresolved aliases");
        let overridden = model_geometry_from_evidence(
            &evidence,
            ModelGeometryOverrides {
                layers: Some(16),
                ..ModelGeometryOverrides::default()
            },
        )
        .expect("explicit layer count replaces conflicting layer aliases");
        assert_eq!(overridden.layers, 16);
    }

    #[test]
    fn explicit_parameter_count_allows_moe_config() {
        let untrusted_config_count = model_geometry_from_value(&json!({
            "model_type": "mixtral",
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "num_local_experts": 8,
            "num_parameters": 13_000_000_000_u64
        }))
        .expect_err("MoE config counts can describe active rather than resident weights");
        assert!(
            untrusted_config_count
                .to_string()
                .contains("total-resident")
        );

        let evidence = model_config_evidence_from_value(&json!({
            "model_type": "mixtral",
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "num_local_experts": 8
        }))
        .expect("valid raw evidence");

        let geometry = model_geometry_from_evidence(
            &evidence,
            ModelGeometryOverrides {
                parameters_billion: Some(46.7),
                ..ModelGeometryOverrides::default()
            },
        )
        .expect("explicit parameter count avoids unsafe dense MoE estimation");

        assert!((geometry.parameters_billion - 46.7).abs() < f64::EPSILON);
        assert_eq!(
            geometry.parameter_source,
            ModelParameterSource::ExplicitOverride
        );
        assert_eq!(geometry.expert_count, Some(8));
        assert!(geometry.is_moe);
        assert!(
            geometry
                .caveats
                .iter()
                .any(|item| item.contains("explicitly overridden"))
        );
    }

    #[test]
    fn geometry_overrides_drive_dependent_dense_parameter_estimate() {
        let config = json!({
            "model_type": "mistral",
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "vocab_size": 32_000,
            "tie_word_embeddings": true
        });
        let evidence = model_config_evidence_from_value(&config).expect("valid raw evidence");
        let original = model_geometry_from_evidence(&evidence, ModelGeometryOverrides::default())
            .expect("derive original geometry");
        let overridden = model_geometry_from_evidence(
            &evidence,
            ModelGeometryOverrides {
                layers: Some(4),
                kv_heads: Some(4),
                head_dim: Some(64),
                ..ModelGeometryOverrides::default()
            },
        )
        .expect("derive overridden geometry");

        assert_eq!(overridden.layers, 4);
        assert_eq!(overridden.kv_heads, 4);
        assert_eq!(overridden.head_dim, 64);
        assert_eq!(
            overridden.parameter_source,
            ModelParameterSource::DenseEstimate
        );
        assert!(
            (overridden.parameters_billion - original.parameters_billion).abs() > f64::EPSILON,
            "parameter estimation must use effective override geometry"
        );

        let expected = estimate_dense_parameters(DenseEstimateInput {
            layers: 4,
            attention_heads: 8,
            kv_heads: 4,
            head_dim: 64,
            hidden_size: 1024,
            intermediate_size: 4096,
            vocab_size: 32_000,
            tied_embeddings: true,
            gated_mlp: true,
        })
        .expect("expected estimate");
        assert!((overridden.parameters_billion - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn model_config_evidence_debug_is_redacted_and_model_type_is_bounded() {
        let evidence = model_config_evidence_from_value(&json!({
            "model_type": "llama",
            "private_token": "do-not-log"
        }))
        .expect("object evidence");
        let debug = format!("{evidence:?}");
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("do-not-log"));

        let oversized = model_geometry_from_value(&json!({
            "model_type": "x".repeat(MAX_MODEL_TYPE_BYTES + 1),
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "hidden_size": 1024,
            "num_parameters": 1_000_000_000_u64
        }))
        .expect_err("oversized retained model type must fail");
        assert!(oversized.to_string().contains("128-byte limit"));

        let unsafe_identifier = model_geometry_from_value(&json!({
            "model_type": "https://secret.example/model",
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "hidden_size": 1024,
            "num_parameters": 1_000_000_000_u64
        }))
        .expect_err("model type must not retain URL-like values");
        assert!(unsafe_identifier.to_string().contains("ASCII identifier"));
    }

    #[test]
    fn legacy_model_geometry_deserializes_with_unknown_parameter_source() {
        let geometry: ModelGeometry = serde_json::from_value(json!({
            "parameters_billion": 7.0,
            "layers": 32,
            "kv_heads": 8,
            "head_dim": 128,
            "model_type": "llama",
            "caveats": []
        }))
        .expect("legacy geometry remains readable");

        assert_eq!(geometry.parameter_source, ModelParameterSource::Unknown);
        assert_eq!(geometry.expert_count, None);
        assert!(!geometry.is_moe);
    }

    #[test]
    fn rejects_zero_missing_and_moe_estimates() {
        let zero = model_geometry_from_value(&json!({
            "model_type": "llama",
            "num_hidden_layers": 0,
            "num_attention_heads": 8,
            "hidden_size": 1024,
            "num_parameters": 1_000_000_000_u64
        }))
        .expect_err("zero dimensions must fail");
        assert!(zero.to_string().contains("greater than zero"));

        let missing = model_geometry_from_value(&json!({
            "model_type": "llama",
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "hidden_size": 1024
        }))
        .expect_err("estimates need MLP dimensions");
        assert!(missing.to_string().contains("intermediate size"));

        let unknown = model_geometry_from_value(&json!({
            "model_type": "custom_transformer",
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "vocab_size": 32_000
        }))
        .expect_err("unknown architectures must not receive an optimistic dense estimate");
        assert!(unknown.to_string().contains("not audited"));

        let moe = model_geometry_from_value(&json!({
            "model_type": "mixtral",
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "vocab_size": 32_000,
            "num_local_experts": 8
        }))
        .expect_err("MoE estimates must require an explicit resident count");
        assert!(moe.to_string().contains("total-resident"));

        let sparse_marker = model_geometry_from_value(&json!({
            "model_type": "llama",
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "vocab_size": 32_000,
            "num_experts_per_tok": 2,
            "num_parameters": 1_000_000_000_u64
        }))
        .expect_err("sparse-routing markers must block config parameter counts");
        assert!(sparse_marker.to_string().contains("total-resident"));
    }

    #[test]
    fn loads_a_bounded_local_config() {
        let mut file = NamedTempFile::new().expect("temp file");
        write!(
            file,
            "{}",
            json!({
                "model_type": "llama",
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
                "num_parameters": 7_000_000_000_u64
            })
        )
        .expect("write config");

        let geometry = load_model_geometry(file.path()).expect("load config");
        assert!((geometry.parameters_billion - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn rejects_oversized_local_config_before_parsing() {
        let file = NamedTempFile::new().expect("temp file");
        file.as_file()
            .set_len(MAX_CONFIG_BYTES + 1)
            .expect("grow config");

        let error = load_model_geometry(file.path()).expect_err("oversized config must fail");
        assert!(error.to_string().contains("exceeds the 8 MiB limit"));
    }
}
