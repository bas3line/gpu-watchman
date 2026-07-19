//! Safety-first memory planning for model weights, KV cache, and runtime headroom.

use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::presentation::safe_inline;

use super::artifact::{
    ARTIFACT_REPORT_VERSION, ArtifactFormat, ArtifactLayout, ArtifactReport, dtype_bits,
};
use super::model_config::{MAX_MODEL_TYPE_BYTES, ModelParameterSource};

/// Machine-readable evidence version emitted by the capacity planner.
pub const CAPACITY_REPORT_VERSION: u32 = 3;
/// Maximum accepted expansion factor for serialized artifact bytes.
pub const MAX_ARTIFACT_RESIDENCY_MULTIPLIER: f64 = 1_000.0;
const MAX_ARTIFACT_ADJUSTED_BYTES: u64 = 8_000_000_000_000_000;

const MAX_PARAMETERS_BILLION: f64 = 1_000_000.0;
const MAX_PRECISION_BITS: f64 = 64.0;
const MAX_GPU_VRAM_GIB: f64 = 1_000_000.0;
const MAX_RUNTIME_OVERHEAD_GIB: f64 = 1_000_000.0;
const MAX_WEIGHT_OVERHEAD_PERCENT: f64 = 1_000.0;
const MAX_WORLD_SIZE: u32 = 1_000_000;
const MAX_MODEL_DIMENSION: u32 = 1_000_000;
const MAX_CONTEXT_TOKENS: u64 = 1_000_000_000_000;
const MAX_CONCURRENT_SEQUENCES_PER_REPLICA: u32 = 10_000_000;
const MAX_EXPERTS: u32 = 10_000_000;

/// User-supplied deployment and model geometry.
///
/// Parallelism dimensions describe rank placement. Expert parallelism overlays
/// the tensor/data-parallel ranks and therefore does not multiply world size.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacityInput {
    pub parameters_billion: f64,
    /// Provenance for the effective parameter count.
    pub parameter_source: ModelParameterSource,
    /// Bounded public model-family identifier when geometry came from a config.
    pub model_type: Option<String>,
    pub weight_bits: f64,
    /// Optional assertion about the number of GPUs available to the deployment.
    pub expected_gpu_count: Option<u32>,
    pub tensor_parallel_size: u32,
    pub pipeline_parallel_size: u32,
    pub data_parallel_size: u32,
    pub expert_parallel_size: u32,
    pub gpu_vram_gib: f64,
    pub memory_utilization: f64,
    pub layers: u32,
    pub kv_heads: u32,
    /// Upper bound for complete KV heads retained on one TP rank. Defaults to
    /// all KV heads because TP alone does not prove KV-cache sharding.
    pub max_kv_heads_per_rank: Option<u32>,
    pub head_dim: u32,
    pub context_tokens: u64,
    /// Full-context sequences resident on each data-parallel replica.
    pub concurrent_sequences_per_data_parallel_replica: u32,
    pub kv_cache_bits: f64,
    /// Runtime/workspace allowance reserved on every rank.
    pub runtime_overhead_gib_per_rank: f64,
    pub weight_overhead_percent: f64,
    /// Upper bound for shared base-weight bytes from the heaviest pipeline
    /// stage assigned to one tensor-parallel rank. Defaults to 100%.
    pub max_shared_rank_weight_percent: Option<f64>,
    /// Upper bound that independently applies to shared, expert, and overhead
    /// weight bytes assigned to the heaviest pipeline stage. A multi-stage plan
    /// defaults to 100%, never an even split.
    pub max_pipeline_stage_component_weight_percent: Option<f64>,
    /// Upper bound for transformer layers assigned to the heaviest pipeline
    /// stage. A multi-stage plan defaults to all layers.
    pub max_pipeline_stage_layers: Option<u32>,
    /// Total routed expert count. Must be paired with `expert_weight_percent`.
    pub expert_count: Option<u32>,
    /// Fraction of weight bytes belonging to routed experts.
    pub expert_weight_percent: Option<f64>,
    /// Upper bound for routed-expert bytes from the heaviest pipeline stage
    /// assigned to one expert-parallel rank. Defaults to 100%.
    pub max_expert_rank_weight_percent: Option<f64>,
}

/// Stable codes for assumptions made by the arithmetic model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapacityAssumptionCode {
    ParameterPrecisionWeightEstimate,
    ArtifactSerializedBytesAreResidencyFloor,
    ArtifactDoesNotProveRuntimePlacement,
    ConservativeSharedWeightPlacement,
    ExplicitSharedRankWeightUpperBound,
    WeightOverheadIsUnsharded,
    ConservativePipelineStageWeights,
    ExplicitPipelineStageComponentWeightUpperBound,
    ConservativePipelineStageLayers,
    ExplicitPipelineStageLayerUpperBound,
    KvCacheUsesLayerPlacement,
    ConservativeKvHeadPlacement,
    ExplicitKvHeadPlacementUpperBound,
    DataParallelConcurrencyIsPerReplica,
    ConservativeExpertWeightPlacement,
    ExplicitExpertRankWeightUpperBound,
    UniformGpuMemory,
    RuntimeOverheadIsPerRank,
    /// The application may add this when resolving the legacy `--gpus` option
    /// to tensor parallelism before calling the planner.
    LegacyGpusInterpretedAsTp,
}

/// Candidate selected as the logical base-weight memory floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapacityWeightBasis {
    ParameterPrecisionEstimate,
    ArtifactResidencyFloor,
}

/// Path-free Artifact Report v1 evidence used to strengthen a capacity plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacityArtifactEvidence {
    pub source_artifact_version: u32,
    pub artifact_format: ArtifactFormat,
    pub layout: ArtifactLayout,
    pub shard_files: u32,
    pub tensor_count: u64,
    pub tensor_elements: u64,
    pub serialized_tensor_bytes: u64,
    pub directory_descriptors_anchored: bool,
    pub residency_multiplier: f64,
    pub adjusted_residency_floor_gib_logical: f64,
    pub selected_as_base_weight_floor: bool,
}

/// Human context paired with a stable machine-readable assumption code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapacityAssumption {
    pub code: CapacityAssumptionCode,
    pub detail: String,
}

/// Validated topology and placement facts used by the estimate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacityTopologyEvidence {
    pub tensor_parallel_size: u32,
    pub pipeline_parallel_size: u32,
    pub data_parallel_size: u32,
    pub expert_parallel_size: u32,
    pub world_size: u32,
    pub ranks_per_pipeline_stage: u32,
    pub expected_gpu_count: Option<u32>,
    pub concurrent_sequences_per_data_parallel_replica: u32,
    pub worst_pipeline_stage_layers: u32,
    pub worst_pipeline_stage_component_weight_percent: f64,
    pub worst_shared_rank_weight_percent: f64,
    pub worst_expert_rank_weight_percent: Option<f64>,
    pub kv_heads_per_tensor_parallel_rank_upper_bound: u32,
    pub kv_head_replication_upper_bound: f64,
}

/// Weight-byte derivation for the rank with the largest modeled placement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacityWeightEvidence {
    pub base_weight_basis: CapacityWeightBasis,
    pub parameter_derived_weight_memory_gib_logical: f64,
    pub artifact_adjusted_weight_floor_gib_logical: Option<f64>,
    pub base_weight_memory_gib_logical: f64,
    pub weight_overhead_memory_gib_logical: f64,
    pub weight_memory_gib_logical: f64,
    pub shared_weight_memory_gib_logical: f64,
    pub expert_weight_memory_gib_logical: f64,
    pub shared_weight_memory_gib_worst_rank: f64,
    pub expert_weight_memory_gib_worst_rank: f64,
    pub weight_overhead_memory_gib_worst_rank: f64,
    pub weight_memory_gib_worst_rank: f64,
}

/// KV-cache derivation for a rank in the largest pipeline stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacityKvCacheEvidence {
    pub kv_cache_mib_per_sequence_logical: f64,
    pub kv_cache_mib_per_sequence_physical_per_data_parallel_replica_upper_bound: f64,
    pub kv_cache_mib_per_concurrency_slot_across_deployment_upper_bound: f64,
    pub kv_cache_mib_per_sequence_worst_rank: f64,
    pub kv_cache_gib_requested_worst_rank: f64,
}

/// Final per-rank memory budget and capacity result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacityMemoryEvidence {
    pub runtime_overhead_gib_per_rank: f64,
    pub estimated_memory_gib_worst_rank: f64,
    pub usable_memory_gib_per_rank: f64,
    pub headroom_gib_worst_rank: f64,
    pub estimated_max_concurrent_sequences_per_data_parallel_replica: u64,
}

/// Versioned, self-contained evidence emitted by the safety-first planner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacityReport {
    pub capacity_version: u32,
    pub input: CapacityInput,
    pub artifact: Option<CapacityArtifactEvidence>,
    pub topology: CapacityTopologyEvidence,
    pub weights: CapacityWeightEvidence,
    pub kv_cache: CapacityKvCacheEvidence,
    pub memory: CapacityMemoryEvidence,
    pub fits: bool,
    pub assumptions: Vec<CapacityAssumption>,
    pub caveats: Vec<String>,
}

struct ValidatedPlacement {
    world_size: u32,
    ranks_per_pipeline_stage: u32,
    worst_pipeline_stage_layers: u32,
    worst_pipeline_stage_component_weight_percent: f64,
    kv_heads_per_tensor_parallel_rank_upper_bound: u32,
    kv_head_replication_upper_bound: f64,
    expert_weight_fraction: f64,
    shared_rank_weight_fraction: f64,
    expert_rank_weight_fraction: f64,
    has_expert_weights: bool,
}

/// Estimate the worst modeled rank rather than averaging allocations across the
/// entire world. Invalid or ambiguous layouts are rejected instead of being
/// coerced into a result that could incorrectly claim a deployment fits.
pub fn estimate(input: &CapacityInput) -> Result<CapacityReport> {
    estimate_internal(input, None)
}

/// Strengthen a capacity estimate with a verified, path-free Artifact Report
/// v1. Serialized bytes are multiplied by an explicit expansion factor and are
/// used only as a floor: artifact evidence can never reduce the parameter-based
/// estimate or prove runtime placement.
pub fn estimate_with_artifact(
    input: &CapacityInput,
    artifact: &ArtifactReport,
    residency_multiplier: f64,
) -> Result<CapacityReport> {
    estimate_internal(
        input,
        Some(validate_artifact_evidence(artifact, residency_multiplier)?),
    )
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]
fn estimate_internal(
    input: &CapacityInput,
    mut artifact: Option<CapacityArtifactEvidence>,
) -> Result<CapacityReport> {
    let placement = validate(input)?;

    let parameter_derived_weight_bytes = finite_product(
        "weight byte estimate",
        input.parameters_billion * 1_000_000_000.0,
        input.weight_bits / 8.0,
    )?;
    let artifact_adjusted_weight_floor_gib_logical = artifact
        .as_ref()
        .map(|evidence| evidence.adjusted_residency_floor_gib_logical);
    let artifact_adjusted_weight_floor_bytes =
        artifact_adjusted_weight_floor_gib_logical.map(|gib| gib * 1024.0_f64.powi(3));
    let artifact_floor_selected = artifact_adjusted_weight_floor_bytes
        .is_some_and(|floor| floor > parameter_derived_weight_bytes);
    let base_weight_basis = if artifact_floor_selected {
        CapacityWeightBasis::ArtifactResidencyFloor
    } else {
        CapacityWeightBasis::ParameterPrecisionEstimate
    };
    let base_weight_bytes = artifact_adjusted_weight_floor_bytes
        .map_or(parameter_derived_weight_bytes, |floor| {
            floor.max(parameter_derived_weight_bytes)
        });
    if let Some(evidence) = artifact.as_mut() {
        evidence.selected_as_base_weight_floor = artifact_floor_selected;
    }
    let weight_overhead_bytes = finite_product(
        "weight overhead byte estimate",
        base_weight_bytes,
        input.weight_overhead_percent / 100.0,
    )?;
    let parameter_derived_weight_memory_gib_logical = bytes_to_gib(parameter_derived_weight_bytes);
    let base_weight_memory_gib_logical = bytes_to_gib(base_weight_bytes);
    let weight_overhead_memory_gib_logical = bytes_to_gib(weight_overhead_bytes);
    let weight_memory_gib_logical =
        base_weight_memory_gib_logical + weight_overhead_memory_gib_logical;
    ensure_finite("base weight memory", base_weight_memory_gib_logical)?;
    ensure_finite(
        "logical weight overhead memory",
        weight_overhead_memory_gib_logical,
    )?;
    ensure_finite("logical weight memory", weight_memory_gib_logical)?;

    let expert_weight_memory_gib_logical =
        base_weight_memory_gib_logical * placement.expert_weight_fraction;
    let shared_weight_memory_gib_logical =
        base_weight_memory_gib_logical - expert_weight_memory_gib_logical;
    let stage_fraction = placement.worst_pipeline_stage_component_weight_percent / 100.0;
    let shared_weight_memory_gib_worst_rank =
        shared_weight_memory_gib_logical * stage_fraction * placement.shared_rank_weight_fraction;
    let expert_weight_memory_gib_worst_rank =
        expert_weight_memory_gib_logical * stage_fraction * placement.expert_rank_weight_fraction;
    // Metadata, scales, and dequantization storage vary by artifact format. Do
    // not assume tensor/expert sharding for those bytes without artifact proof.
    let weight_overhead_memory_gib_worst_rank = weight_overhead_memory_gib_logical * stage_fraction;
    let weight_memory_gib_worst_rank = shared_weight_memory_gib_worst_rank
        + expert_weight_memory_gib_worst_rank
        + weight_overhead_memory_gib_worst_rank;
    ensure_finite(
        "worst-rank shared weight memory",
        shared_weight_memory_gib_worst_rank,
    )?;
    ensure_finite(
        "worst-rank expert weight memory",
        expert_weight_memory_gib_worst_rank,
    )?;
    ensure_finite(
        "worst-rank weight overhead memory",
        weight_overhead_memory_gib_worst_rank,
    )?;
    ensure_finite("worst-rank weight memory", weight_memory_gib_worst_rank)?;

    // K and V each retain one value per layer, KV head, head dimension, and token.
    let total_kv_elements = checked_u128_product(
        "total KV-cache geometry",
        &[
            2,
            u128::from(input.layers),
            u128::from(input.kv_heads),
            u128::from(input.head_dim),
            u128::from(input.context_tokens),
        ],
    )?;
    let worst_rank_kv_elements = checked_u128_product(
        "worst-rank KV-cache geometry",
        &[
            2,
            u128::from(placement.worst_pipeline_stage_layers),
            u128::from(placement.kv_heads_per_tensor_parallel_rank_upper_bound),
            u128::from(input.head_dim),
            u128::from(input.context_tokens),
        ],
    )?;
    let kv_cache_bytes_per_sequence_total = finite_product(
        "total KV-cache bytes",
        total_kv_elements as f64,
        input.kv_cache_bits / 8.0,
    )?;
    let kv_cache_bytes_per_sequence_worst_rank = finite_product(
        "worst-rank KV-cache bytes",
        worst_rank_kv_elements as f64,
        input.kv_cache_bits / 8.0,
    )?;
    ensure!(
        kv_cache_bytes_per_sequence_worst_rank > 0.0,
        "worst-rank KV-cache bytes must be greater than zero"
    );
    let kv_cache_mib_per_sequence_logical = bytes_to_mib(kv_cache_bytes_per_sequence_total);
    let kv_cache_mib_per_sequence_physical_per_data_parallel_replica_upper_bound = finite_product(
        "physical KV-cache memory upper bound per data-parallel replica",
        kv_cache_mib_per_sequence_logical,
        placement.kv_head_replication_upper_bound,
    )?;
    let kv_cache_mib_per_concurrency_slot_across_deployment_upper_bound = finite_product(
        "physical KV-cache memory upper bound for one sequence on every data-parallel replica",
        kv_cache_mib_per_sequence_physical_per_data_parallel_replica_upper_bound,
        f64::from(input.data_parallel_size),
    )?;
    let kv_cache_gib_requested_worst_rank = finite_product(
        "requested worst-rank KV-cache memory",
        bytes_to_gib(kv_cache_bytes_per_sequence_worst_rank),
        f64::from(input.concurrent_sequences_per_data_parallel_replica),
    )?;

    let usable_memory_gib_per_rank = input.gpu_vram_gib * input.memory_utilization;
    let fixed_memory_gib_worst_rank =
        weight_memory_gib_worst_rank + input.runtime_overhead_gib_per_rank;
    let estimated_memory_gib_worst_rank =
        fixed_memory_gib_worst_rank + kv_cache_gib_requested_worst_rank;
    let headroom_gib_worst_rank = usable_memory_gib_per_rank - estimated_memory_gib_worst_rank;
    ensure_finite("usable memory per rank", usable_memory_gib_per_rank)?;
    ensure_finite(
        "estimated memory on the worst rank",
        estimated_memory_gib_worst_rank,
    )?;
    ensure_finite("worst-rank headroom", headroom_gib_worst_rank)?;

    let kv_cache_gib_per_sequence_worst_rank = bytes_to_gib(kv_cache_bytes_per_sequence_worst_rank);
    let sequence_capacity = ((usable_memory_gib_per_rank - fixed_memory_gib_worst_rank).max(0.0)
        / kv_cache_gib_per_sequence_worst_rank)
        .floor();
    ensure!(
        sequence_capacity.is_finite() && sequence_capacity >= 0.0,
        "estimated sequence capacity is outside the supported range"
    );
    ensure!(
        sequence_capacity <= u64::MAX as f64,
        "estimated sequence capacity exceeds the supported range"
    );
    let estimated_max_concurrent_sequences_per_data_parallel_replica = sequence_capacity as u64;

    let mut assumptions = vec![
        assumption(
            CapacityAssumptionCode::ParameterPrecisionWeightEstimate,
            "Parameter count and precision form an analytical weight-byte estimate, not observed runtime residency; configured overhead is applied after selecting the strongest available base-weight floor.",
        ),
        assumption(
            CapacityAssumptionCode::WeightOverheadIsUnsharded,
            "Weight metadata and dequantization overhead is not divided by tensor or expert parallelism.",
        ),
        assumption(
            CapacityAssumptionCode::KvCacheUsesLayerPlacement,
            "KV cache on the worst rank follows the validated conservative pipeline-layer upper bound.",
        ),
        assumption(
            CapacityAssumptionCode::DataParallelConcurrencyIsPerReplica,
            "Requested concurrency is resident on every data-parallel replica; it is not divided by data parallelism.",
        ),
        assumption(
            CapacityAssumptionCode::UniformGpuMemory,
            "Every rank has the configured physical VRAM and utilization limit.",
        ),
        assumption(
            CapacityAssumptionCode::RuntimeOverheadIsPerRank,
            "Configured runtime overhead is reserved independently on every rank.",
        ),
    ];
    if artifact.is_some() {
        assumptions.push(assumption(
            CapacityAssumptionCode::ArtifactSerializedBytesAreResidencyFloor,
            "Verified serialized tensor bytes were multiplied by the operator-supplied residency factor and used only as a floor; the result is the maximum of that floor and the parameter-derived estimate.",
        ));
        assumptions.push(assumption(
            CapacityAssumptionCode::ArtifactDoesNotProveRuntimePlacement,
            "Checkpoint shard boundaries and tensor metadata do not provide TP, PP, DP, or EP placement credit.",
        ));
    }
    if input.max_shared_rank_weight_percent.is_none() {
        assumptions.push(assumption(
            CapacityAssumptionCode::ConservativeSharedWeightPlacement,
            "No shared-rank byte bound was supplied, so one TP rank is charged 100% of shared base weights on the heaviest pipeline stage.",
        ));
    } else {
        assumptions.push(assumption(
            CapacityAssumptionCode::ExplicitSharedRankWeightUpperBound,
            "The supplied shared-rank percentage bounds shared base-weight bytes from the heaviest pipeline stage on the worst TP rank; even sharding is not inferred.",
        ));
    }
    if input.pipeline_parallel_size > 1
        && input.max_pipeline_stage_component_weight_percent.is_none()
    {
        assumptions.push(assumption(
            CapacityAssumptionCode::ConservativePipelineStageWeights,
            "No pipeline component-weight bound was supplied, so the worst stage is charged 100% of shared, expert, and overhead bytes.",
        ));
    } else if input.pipeline_parallel_size > 1 {
        assumptions.push(assumption(
            CapacityAssumptionCode::ExplicitPipelineStageComponentWeightUpperBound,
            "The supplied percentage independently bounds shared, expert, and overhead bytes on the heaviest pipeline stage.",
        ));
    }
    if input.pipeline_parallel_size > 1 && input.max_pipeline_stage_layers.is_none() {
        assumptions.push(assumption(
            CapacityAssumptionCode::ConservativePipelineStageLayers,
            "No pipeline layer-placement bound was supplied, so the worst stage is charged all transformer layers for KV cache.",
        ));
    } else if input.pipeline_parallel_size > 1 {
        assumptions.push(assumption(
            CapacityAssumptionCode::ExplicitPipelineStageLayerUpperBound,
            "The supplied maximum pipeline-stage layer count bounds the KV-cache layer placement.",
        ));
    }
    if input.max_kv_heads_per_rank.is_none() {
        assumptions.push(assumption(
            CapacityAssumptionCode::ConservativeKvHeadPlacement,
            "No KV-head rank bound was supplied, so every TP rank is charged all complete KV heads; TP alone is not treated as proof of KV-cache sharding.",
        ));
    } else {
        assumptions.push(assumption(
            CapacityAssumptionCode::ExplicitKvHeadPlacementUpperBound,
            "The supplied complete KV-head count bounds the worst TP rank and may conservatively include uneven placement or replication.",
        ));
    }
    if placement.has_expert_weights {
        if input.max_expert_rank_weight_percent.is_none() {
            assumptions.push(assumption(
                CapacityAssumptionCode::ConservativeExpertWeightPlacement,
                "No expert-rank byte bound was supplied, so one rank is charged 100% of routed-expert bytes on the heaviest pipeline stage.",
            ));
        } else {
            assumptions.push(assumption(
                CapacityAssumptionCode::ExplicitExpertRankWeightUpperBound,
                "The supplied expert-rank percentage bounds routed-expert bytes on the worst rank; equal expert counts alone do not imply equal byte sizes.",
            ));
        }
    }

    let mut caveats = vec![
        "This is a worst-modeled-rank estimate, not a reservation guarantee.".to_owned(),
        "Activation peaks, CUDA graphs, kernels, allocator fragmentation, multimodal encoders, speculative decoding, and communication workspaces can require additional memory.".to_owned(),
        "Quantization metadata and dequantization workspaces vary by checkpoint format; calibrate overheads against the deployed artifact and runtime.".to_owned(),
    ];
    if artifact.is_some() {
        caveats.push(
            "Artifact evidence describes serialized checkpoint tensors, not loaded allocations; the residency multiplier and runtime overhead must cover format conversion, dequantization, duplicated tensors, and runtime-specific expansion."
                .to_owned(),
        );
        caveats.push(
            "Artifact shard boundaries do not prove TP, PP, DP, or EP placement; existing fail-closed placement bounds still apply independently."
                .to_owned(),
        );
        caveats.push(
            "The capacity planner does not prove that the inspected point-in-time artifact matches the model or runtime that will be deployed."
                .to_owned(),
        );
    }

    Ok(CapacityReport {
        capacity_version: CAPACITY_REPORT_VERSION,
        input: input.clone(),
        artifact,
        topology: CapacityTopologyEvidence {
            tensor_parallel_size: input.tensor_parallel_size,
            pipeline_parallel_size: input.pipeline_parallel_size,
            data_parallel_size: input.data_parallel_size,
            expert_parallel_size: input.expert_parallel_size,
            world_size: placement.world_size,
            ranks_per_pipeline_stage: placement.ranks_per_pipeline_stage,
            expected_gpu_count: input.expected_gpu_count,
            concurrent_sequences_per_data_parallel_replica: input
                .concurrent_sequences_per_data_parallel_replica,
            worst_pipeline_stage_layers: placement.worst_pipeline_stage_layers,
            worst_pipeline_stage_component_weight_percent: placement
                .worst_pipeline_stage_component_weight_percent,
            worst_shared_rank_weight_percent: placement.shared_rank_weight_fraction * 100.0,
            worst_expert_rank_weight_percent: placement
                .has_expert_weights
                .then_some(placement.expert_rank_weight_fraction * 100.0),
            kv_heads_per_tensor_parallel_rank_upper_bound: placement
                .kv_heads_per_tensor_parallel_rank_upper_bound,
            kv_head_replication_upper_bound: placement.kv_head_replication_upper_bound,
        },
        weights: CapacityWeightEvidence {
            base_weight_basis,
            parameter_derived_weight_memory_gib_logical,
            artifact_adjusted_weight_floor_gib_logical,
            base_weight_memory_gib_logical,
            weight_overhead_memory_gib_logical,
            weight_memory_gib_logical,
            shared_weight_memory_gib_logical,
            expert_weight_memory_gib_logical,
            shared_weight_memory_gib_worst_rank,
            expert_weight_memory_gib_worst_rank,
            weight_overhead_memory_gib_worst_rank,
            weight_memory_gib_worst_rank,
        },
        kv_cache: CapacityKvCacheEvidence {
            kv_cache_mib_per_sequence_logical,
            kv_cache_mib_per_sequence_physical_per_data_parallel_replica_upper_bound,
            kv_cache_mib_per_concurrency_slot_across_deployment_upper_bound,
            kv_cache_mib_per_sequence_worst_rank: bytes_to_mib(
                kv_cache_bytes_per_sequence_worst_rank,
            ),
            kv_cache_gib_requested_worst_rank,
        },
        memory: CapacityMemoryEvidence {
            runtime_overhead_gib_per_rank: input.runtime_overhead_gib_per_rank,
            estimated_memory_gib_worst_rank,
            usable_memory_gib_per_rank,
            headroom_gib_worst_rank,
            estimated_max_concurrent_sequences_per_data_parallel_replica,
        },
        fits: headroom_gib_worst_rank >= 0.0,
        assumptions,
        caveats,
    })
}

#[allow(clippy::cast_precision_loss)]
fn validate_artifact_evidence(
    report: &ArtifactReport,
    residency_multiplier: f64,
) -> Result<CapacityArtifactEvidence> {
    ensure_finite_range(
        "artifact_residency_multiplier",
        residency_multiplier,
        1.0,
        MAX_ARTIFACT_RESIDENCY_MULTIPLIER,
    )?;
    validate_artifact_contract(report)?;
    validate_artifact_totals(report)?;
    validate_artifact_layout(report)?;

    let adjusted_bytes =
        checked_ceil_scaled_bytes(report.summary.serialized_tensor_bytes, residency_multiplier)?;
    ensure!(
        adjusted_bytes <= MAX_ARTIFACT_ADJUSTED_BYTES,
        "artifact adjusted residency floor exceeds the supported capacity-planning range"
    );
    Ok(CapacityArtifactEvidence {
        source_artifact_version: report.artifact_version,
        artifact_format: report.artifact_format,
        layout: report.layout,
        shard_files: report.summary.shard_files,
        tensor_count: report.summary.tensor_count,
        tensor_elements: report.summary.tensor_elements,
        serialized_tensor_bytes: report.summary.serialized_tensor_bytes,
        directory_descriptors_anchored: report.verification.directory_descriptors_anchored,
        residency_multiplier,
        adjusted_residency_floor_gib_logical: bytes_to_gib(adjusted_bytes as f64),
        selected_as_base_weight_floor: false,
    })
}

fn checked_ceil_scaled_bytes(bytes: u64, multiplier: f64) -> Result<u64> {
    ensure!(
        multiplier.is_finite() && multiplier >= 1.0,
        "artifact residency multiplier must be finite and at least one"
    );
    let encoded = multiplier.to_bits();
    let exponent = i32::try_from((encoded >> 52) & 0x7ff)
        .map_err(|_| anyhow::anyhow!("artifact residency multiplier exponent is invalid"))?;
    ensure!(
        exponent > 0 && exponent < 0x7ff,
        "artifact residency multiplier is not a supported normal number"
    );
    let significand = u128::from((1_u64 << 52) | (encoded & ((1_u64 << 52) - 1)));
    let product = u128::from(bytes)
        .checked_mul(significand)
        .ok_or_else(|| anyhow::anyhow!("artifact adjusted residency floor overflows"))?;
    let binary_exponent = exponent - 1023 - 52;
    let rounded = if binary_exponent >= 0 {
        product
            .checked_shl(u32::try_from(binary_exponent).unwrap_or(u32::MAX))
            .ok_or_else(|| anyhow::anyhow!("artifact adjusted residency floor overflows"))?
    } else {
        let denominator = 1_u128
            .checked_shl(binary_exponent.unsigned_abs())
            .ok_or_else(|| anyhow::anyhow!("artifact residency multiplier is too precise"))?;
        let quotient = product / denominator;
        quotient
            .checked_add(u128::from(!product.is_multiple_of(denominator)))
            .ok_or_else(|| anyhow::anyhow!("artifact adjusted residency floor overflows"))?
    };
    u64::try_from(rounded)
        .map_err(|_| anyhow::anyhow!("artifact adjusted residency floor exceeds u64"))
}

fn validate_artifact_contract(report: &ArtifactReport) -> Result<()> {
    ensure!(
        report.artifact_version == ARTIFACT_REPORT_VERSION,
        "capacity planning requires Artifact Report v{ARTIFACT_REPORT_VERSION}"
    );
    ensure!(
        report.artifact_format == ArtifactFormat::Safetensors,
        "capacity planning does not support this artifact format"
    );
    ensure!(
        report.summary.shard_files > 0,
        "artifact evidence must contain at least one shard file"
    );
    ensure!(
        report.summary.safetensors_length_prefix_bytes == u64::from(report.summary.shard_files) * 8,
        "artifact length-prefix bytes do not match the shard count"
    );
    ensure!(
        report.summary.serialized_tensor_bytes <= MAX_ARTIFACT_ADJUSTED_BYTES,
        "artifact serialized tensor bytes exceed the supported capacity-planning range"
    );
    ensure!(
        report.verification.regular_shard_files
            && report.verification.final_symlinks_rejected
            && report.verification.headers_validated
            && report.verification.data_offsets_complete_without_holes,
        "artifact evidence is missing required file, header, or offset verification"
    );
    ensure!(
        !cfg!(unix) || report.verification.directory_descriptors_anchored,
        "artifact evidence is missing Unix descriptor-anchored resolution"
    );
    ensure!(
        !report.verification.tensor_payload_contents_read
            && !report.verification.payload_checksum_validated,
        "Artifact Report v1 payload-read verification flags are inconsistent"
    );
    ensure!(
        report.verification.shape_payload_bytes_unverified_tensors == 0
            && report.verification.shape_payload_bytes_verified_tensors
                == report.summary.tensor_count,
        "artifact evidence does not verify shape-to-payload bytes for every tensor"
    );
    ensure!(
        report
            .dtypes
            .iter()
            .all(|dtype| dtype.shape_payload_bytes_verified),
        "artifact evidence contains an unverified dtype group"
    );
    ensure!(
        report
            .dtypes
            .windows(2)
            .all(|pair| pair[0].dtype < pair[1].dtype),
        "artifact dtype groups must be unique and sorted"
    );
    Ok(())
}

fn validate_artifact_totals(report: &ArtifactReport) -> Result<()> {
    for dtype in &report.dtypes {
        ensure!(
            dtype.tensor_count > 0,
            "artifact dtype groups cannot be empty"
        );
        let bits = dtype_bits(&dtype.dtype)
            .ok_or_else(|| anyhow::anyhow!("artifact evidence contains an unsupported dtype"))?;
        let bit_length = dtype
            .tensor_elements
            .checked_mul(bits)
            .ok_or_else(|| anyhow::anyhow!("artifact dtype bit size overflows"))?;
        ensure!(
            bit_length.is_multiple_of(8) && bit_length / 8 == dtype.serialized_bytes,
            "artifact dtype elements, bit width, and serialized bytes disagree"
        );
    }

    let (dtype_tensors, dtype_elements, dtype_bytes) = report.dtypes.iter().try_fold(
        (0_u64, 0_u64, 0_u64),
        |(tensors, elements, bytes), dtype| {
            Ok::<_, anyhow::Error>((
                tensors
                    .checked_add(dtype.tensor_count)
                    .ok_or_else(|| anyhow::anyhow!("artifact dtype tensor count overflows"))?,
                elements
                    .checked_add(dtype.tensor_elements)
                    .ok_or_else(|| anyhow::anyhow!("artifact dtype element count overflows"))?,
                bytes
                    .checked_add(dtype.serialized_bytes)
                    .ok_or_else(|| anyhow::anyhow!("artifact dtype byte count overflows"))?,
            ))
        },
    )?;
    ensure!(
        dtype_tensors == report.summary.tensor_count
            && dtype_elements == report.summary.tensor_elements
            && dtype_bytes == report.summary.serialized_tensor_bytes,
        "artifact dtype totals do not match the artifact summary"
    );

    let minimum_container_bytes = report
        .summary
        .serialized_tensor_bytes
        .checked_add(report.summary.safetensors_header_json_bytes)
        .and_then(|bytes| bytes.checked_add(report.summary.safetensors_length_prefix_bytes))
        .ok_or_else(|| anyhow::anyhow!("artifact container byte evidence overflows"))?;
    ensure!(
        minimum_container_bytes == report.summary.serialized_shard_file_bytes,
        "artifact container bytes do not match payload, header, and prefix totals"
    );
    Ok(())
}

fn validate_artifact_layout(report: &ArtifactReport) -> Result<()> {
    match report.layout {
        ArtifactLayout::SingleFile => ensure!(
            report.summary.shard_files == 1
                && report.summary.index_file_bytes.is_none()
                && report.summary.declared_total_size_bytes.is_none()
                && report.verification.index_membership_validated.is_none()
                && report.verification.declared_total_size_validated.is_none(),
            "single-file artifact evidence contains inconsistent index fields"
        ),
        ArtifactLayout::ShardedIndex => ensure!(
            report
                .summary
                .index_file_bytes
                .is_some_and(|bytes| bytes > 0)
                && report.summary.declared_total_size_bytes
                    == Some(report.summary.serialized_tensor_bytes)
                && report.verification.index_membership_validated == Some(true)
                && report.verification.declared_total_size_validated == Some(true),
            "sharded artifact evidence is missing verified index fields"
        ),
    }
    Ok(())
}

/// Render a capacity report without accepting a second, potentially mismatched
/// copy of the planner input.
#[allow(clippy::too_many_lines)]
pub fn render_text(report: &CapacityReport) -> String {
    let state = if report.fits { "FITS" } else { "DOES NOT FIT" };
    let expected_gpus = report
        .topology
        .expected_gpu_count
        .map_or_else(|| "not asserted".to_owned(), |count| count.to_string());
    let model_type = report.input.model_type.as_deref().unwrap_or("manual");
    let expert_rank_bound = report
        .topology
        .worst_expert_rank_weight_percent
        .map_or_else(|| "n/a".to_owned(), |percent| format!("{percent:.2}%"));
    let weight_basis = match report.weights.base_weight_basis {
        CapacityWeightBasis::ParameterPrecisionEstimate => "parameter-precision estimate",
        CapacityWeightBasis::ArtifactResidencyFloor => "artifact residency floor",
    };
    let artifact_summary = report.artifact.as_ref().map_or_else(
        || "not supplied".to_owned(),
        |artifact| {
            let layout = match artifact.layout {
                ArtifactLayout::SingleFile => "single-file",
                ArtifactLayout::ShardedIndex => "sharded-index",
            };
            format!(
                "v{} {layout} | {} shard(s), {} tensor(s), {} serialized bytes x {:.3}",
                artifact.source_artifact_version,
                artifact.shard_files,
                artifact.tensor_count,
                artifact.serialized_tensor_bytes,
                artifact.residency_multiplier,
            )
        },
    );
    let artifact_floor = report
        .weights
        .artifact_adjusted_weight_floor_gib_logical
        .map_or_else(|| "not supplied".to_owned(), |gib| format!("{gib:.2} GiB"));
    let mut output = format!(
        "GPU Watchman capacity report v{}  {state}\n\
         Model       {:.3}B parameters @ {:.1}-bit weights\n\
         Provenance  {} | model_type={}\n\
         Topology    TP {} | PP {} | DP {} | EP {} | world {} | expected GPUs {}\n\
         Placement   worst stage {} layer(s), {:.2}% of each weight component\n\
         Weight map  shared rank <= {:.2}% | expert rank <= {}\n\
         Workload    {} sequence(s)/DP replica x {} tokens\n\n\
         Artifact    {artifact_summary}\n\
         Weights     {:.2} GiB logical | {:.2} GiB on worst rank\n  Base      {:.2} GiB logical | Overhead {:.2} GiB logical\n  Worst     Shared {:.2} | Expert {:.2} | unsharded overhead {:.2} GiB\n\
         Basis       {weight_basis} | parameter {:.2} GiB | artifact floor {artifact_floor}\n\
         KV cache    {:.1} MiB/sequence logical | {:.1} MiB physical/DP replica upper bound\n  Fleet     {:.1} MiB upper bound for one sequence on every DP replica\n  Rank      {:.2} MiB/sequence on worst rank\n  Layout    <= {} KV head(s)/TP rank | physical replication <= {:.2}x\n  Requested {:.4} GiB on worst rank\n\
         Runtime     {:.2} GiB/rank configured overhead\n\
         Estimated   {:.2}/{:.2} GiB on worst rank | headroom {:+.2} GiB\n\
         Max conc.   approximately {} full-context sequence(s)/DP replica\n\n\
         Assumptions\n",
        report.capacity_version,
        report.input.parameters_billion,
        report.input.weight_bits,
        parameter_source_code(report.input.parameter_source),
        safe_inline(model_type),
        report.topology.tensor_parallel_size,
        report.topology.pipeline_parallel_size,
        report.topology.data_parallel_size,
        report.topology.expert_parallel_size,
        report.topology.world_size,
        expected_gpus,
        report.topology.worst_pipeline_stage_layers,
        report
            .topology
            .worst_pipeline_stage_component_weight_percent,
        report.topology.worst_shared_rank_weight_percent,
        expert_rank_bound,
        report
            .topology
            .concurrent_sequences_per_data_parallel_replica,
        report.input.context_tokens,
        report.weights.weight_memory_gib_logical,
        report.weights.weight_memory_gib_worst_rank,
        report.weights.base_weight_memory_gib_logical,
        report.weights.weight_overhead_memory_gib_logical,
        report.weights.shared_weight_memory_gib_worst_rank,
        report.weights.expert_weight_memory_gib_worst_rank,
        report.weights.weight_overhead_memory_gib_worst_rank,
        report.weights.parameter_derived_weight_memory_gib_logical,
        report.kv_cache.kv_cache_mib_per_sequence_logical,
        report
            .kv_cache
            .kv_cache_mib_per_sequence_physical_per_data_parallel_replica_upper_bound,
        report
            .kv_cache
            .kv_cache_mib_per_concurrency_slot_across_deployment_upper_bound,
        report.kv_cache.kv_cache_mib_per_sequence_worst_rank,
        report
            .topology
            .kv_heads_per_tensor_parallel_rank_upper_bound,
        report.topology.kv_head_replication_upper_bound,
        report.kv_cache.kv_cache_gib_requested_worst_rank,
        report.memory.runtime_overhead_gib_per_rank,
        report.memory.estimated_memory_gib_worst_rank,
        report.memory.usable_memory_gib_per_rank,
        report.memory.headroom_gib_worst_rank,
        report
            .memory
            .estimated_max_concurrent_sequences_per_data_parallel_replica,
    );
    for assumption in &report.assumptions {
        output.push_str("  - [");
        output.push_str(assumption_code(assumption.code));
        output.push_str("] ");
        output.push_str(&safe_inline(&assumption.detail));
        output.push('\n');
    }
    output.push_str("\nCaveats\n");
    for caveat in &report.caveats {
        output.push_str("  - ");
        output.push_str(&safe_inline(caveat));
        output.push('\n');
    }
    output
}

#[allow(clippy::too_many_lines)]
fn validate(input: &CapacityInput) -> Result<ValidatedPlacement> {
    ensure!(
        input.parameter_source != ModelParameterSource::Unknown,
        "parameter_source must identify the effective parameter-count provenance"
    );
    if let Some(model_type) = input.model_type.as_deref() {
        ensure!(!model_type.is_empty(), "model_type must not be empty");
        ensure!(
            model_type.len() <= MAX_MODEL_TYPE_BYTES,
            "model_type exceeds the {MAX_MODEL_TYPE_BYTES}-byte limit"
        );
        ensure!(
            model_type
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.')),
            "model_type must be an ASCII identifier containing only letters, numbers, '.', '-', or '_'"
        );
    }
    if matches!(
        input.parameter_source,
        ModelParameterSource::ConfigDeclared | ModelParameterSource::DenseEstimate
    ) {
        ensure!(
            input.model_type.is_some(),
            "config-derived parameter provenance requires model_type"
        );
    }
    ensure_finite_range(
        "parameters_billion",
        input.parameters_billion,
        f64::MIN_POSITIVE,
        MAX_PARAMETERS_BILLION,
    )?;
    ensure_finite_range("weight_bits", input.weight_bits, 1.0, MAX_PRECISION_BITS)?;
    ensure_finite_range(
        "gpu_vram_gib",
        input.gpu_vram_gib,
        f64::MIN_POSITIVE,
        MAX_GPU_VRAM_GIB,
    )?;
    ensure_finite_range(
        "memory_utilization",
        input.memory_utilization,
        f64::MIN_POSITIVE,
        1.0,
    )?;
    ensure_finite_range(
        "kv_cache_bits",
        input.kv_cache_bits,
        1.0,
        MAX_PRECISION_BITS,
    )?;
    ensure_finite_range(
        "runtime_overhead_gib_per_rank",
        input.runtime_overhead_gib_per_rank,
        0.0,
        MAX_RUNTIME_OVERHEAD_GIB,
    )?;
    ensure_finite_range(
        "weight_overhead_percent",
        input.weight_overhead_percent,
        0.0,
        MAX_WEIGHT_OVERHEAD_PERCENT,
    )?;

    ensure_positive_bounded(
        "tensor_parallel_size",
        input.tensor_parallel_size,
        MAX_WORLD_SIZE,
    )?;
    ensure_positive_bounded(
        "pipeline_parallel_size",
        input.pipeline_parallel_size,
        MAX_WORLD_SIZE,
    )?;
    ensure_positive_bounded(
        "data_parallel_size",
        input.data_parallel_size,
        MAX_WORLD_SIZE,
    )?;
    ensure_positive_bounded(
        "expert_parallel_size",
        input.expert_parallel_size,
        MAX_WORLD_SIZE,
    )?;
    ensure_positive_bounded("layers", input.layers, MAX_MODEL_DIMENSION)?;
    ensure_positive_bounded("kv_heads", input.kv_heads, MAX_MODEL_DIMENSION)?;
    ensure_positive_bounded("head_dim", input.head_dim, MAX_MODEL_DIMENSION)?;
    ensure!(
        (1..=MAX_CONTEXT_TOKENS).contains(&input.context_tokens),
        "context_tokens must be between 1 and {MAX_CONTEXT_TOKENS}"
    );
    ensure!(
        (1..=MAX_CONCURRENT_SEQUENCES_PER_REPLICA)
            .contains(&input.concurrent_sequences_per_data_parallel_replica),
        "concurrent_sequences_per_data_parallel_replica must be between 1 and {MAX_CONCURRENT_SEQUENCES_PER_REPLICA}"
    );
    ensure!(
        input.pipeline_parallel_size <= input.layers,
        "pipeline_parallel_size ({}) exceeds layer count ({})",
        input.pipeline_parallel_size,
        input.layers
    );

    let ranks_per_pipeline_stage = input
        .tensor_parallel_size
        .checked_mul(input.data_parallel_size)
        .ok_or_else(|| anyhow::anyhow!("TP x DP overflows the supported world-size range"))?;
    let world_size = ranks_per_pipeline_stage
        .checked_mul(input.pipeline_parallel_size)
        .ok_or_else(|| anyhow::anyhow!("TP x PP x DP overflows the supported world-size range"))?;
    ensure!(
        world_size <= MAX_WORLD_SIZE,
        "TP x PP x DP world size ({world_size}) exceeds the supported maximum ({MAX_WORLD_SIZE})"
    );
    if let Some(expected_gpu_count) = input.expected_gpu_count {
        ensure!(
            (1..=MAX_WORLD_SIZE).contains(&expected_gpu_count),
            "expected_gpu_count must be between 1 and {MAX_WORLD_SIZE}"
        );
        ensure!(
            expected_gpu_count == world_size,
            "expected GPU count ({expected_gpu_count}) does not match TP x PP x DP world size ({world_size})"
        );
    }

    let minimum_kv_heads_per_rank = input.kv_heads / input.tensor_parallel_size
        + u32::from(!input.kv_heads.is_multiple_of(input.tensor_parallel_size));
    let kv_heads_per_tensor_parallel_rank_upper_bound =
        input.max_kv_heads_per_rank.unwrap_or(input.kv_heads);
    ensure!(
        (minimum_kv_heads_per_rank..=input.kv_heads)
            .contains(&kv_heads_per_tensor_parallel_rank_upper_bound),
        "max_kv_heads_per_rank must be between {minimum_kv_heads_per_rank} and {}",
        input.kv_heads
    );
    let kv_head_replication_upper_bound = f64::from(kv_heads_per_tensor_parallel_rank_upper_bound)
        * f64::from(input.tensor_parallel_size)
        / f64::from(input.kv_heads);
    let minimum_stage_weight_percent = 100.0 / f64::from(input.pipeline_parallel_size);
    let worst_pipeline_stage_component_weight_percent = input
        .max_pipeline_stage_component_weight_percent
        .unwrap_or(100.0);
    ensure_finite_range(
        "max_pipeline_stage_component_weight_percent",
        worst_pipeline_stage_component_weight_percent,
        minimum_stage_weight_percent,
        100.0,
    )?;
    let minimum_stage_layers = input.layers / input.pipeline_parallel_size
        + u32::from(!input.layers.is_multiple_of(input.pipeline_parallel_size));
    let worst_pipeline_stage_layers = input.max_pipeline_stage_layers.unwrap_or(input.layers);
    ensure!(
        (minimum_stage_layers..=input.layers).contains(&worst_pipeline_stage_layers),
        "max_pipeline_stage_layers must be between {minimum_stage_layers} and {}",
        input.layers
    );

    let minimum_shared_rank_weight_percent = 100.0 / f64::from(input.tensor_parallel_size);
    let shared_rank_weight_percent = input.max_shared_rank_weight_percent.unwrap_or(100.0);
    ensure_finite_range(
        "max_shared_rank_weight_percent",
        shared_rank_weight_percent,
        minimum_shared_rank_weight_percent,
        100.0,
    )?;
    let shared_rank_weight_fraction = shared_rank_weight_percent / 100.0;
    let (expert_weight_fraction, expert_rank_weight_fraction, has_expert_weights) = match (
        input.expert_count,
        input.expert_weight_percent,
    ) {
        (None, None) => {
            ensure!(
                input.expert_parallel_size == 1,
                "expert_parallel_size greater than 1 requires expert_count and expert_weight_percent"
            );
            ensure!(
                input.max_expert_rank_weight_percent.is_none(),
                "max_expert_rank_weight_percent requires expert_count and expert_weight_percent"
            );
            (0.0, 0.0, false)
        }
        (Some(expert_count), Some(expert_weight_percent)) => {
            ensure!(
                (1..=MAX_EXPERTS).contains(&expert_count),
                "expert_count must be between 1 and {MAX_EXPERTS}"
            );
            ensure_finite_range(
                "expert_weight_percent",
                expert_weight_percent,
                f64::MIN_POSITIVE,
                100.0,
            )?;
            ensure!(
                expert_count.is_multiple_of(input.expert_parallel_size),
                "expert_count ({expert_count}) must be divisible by expert_parallel_size ({})",
                input.expert_parallel_size
            );
            ensure!(
                ranks_per_pipeline_stage.is_multiple_of(input.expert_parallel_size),
                "expert_parallel_size ({}) must divide TP x DP ranks per pipeline stage ({ranks_per_pipeline_stage})",
                input.expert_parallel_size
            );
            let minimum_expert_rank_weight_percent = 100.0 / f64::from(input.expert_parallel_size);
            let expert_rank_weight_percent = input.max_expert_rank_weight_percent.unwrap_or(100.0);
            ensure_finite_range(
                "max_expert_rank_weight_percent",
                expert_rank_weight_percent,
                minimum_expert_rank_weight_percent,
                100.0,
            )?;
            (
                expert_weight_percent / 100.0,
                expert_rank_weight_percent / 100.0,
                true,
            )
        }
        _ => bail!("expert_count and expert_weight_percent must be supplied together"),
    };

    Ok(ValidatedPlacement {
        world_size,
        ranks_per_pipeline_stage,
        worst_pipeline_stage_layers,
        worst_pipeline_stage_component_weight_percent,
        kv_heads_per_tensor_parallel_rank_upper_bound,
        kv_head_replication_upper_bound,
        expert_weight_fraction,
        shared_rank_weight_fraction,
        expert_rank_weight_fraction,
        has_expert_weights,
    })
}

fn ensure_finite_range(name: &str, value: f64, minimum: f64, maximum: f64) -> Result<()> {
    ensure!(value.is_finite(), "{name} must be finite");
    ensure!(
        (minimum..=maximum).contains(&value),
        "{name} must be between {minimum} and {maximum}"
    );
    Ok(())
}

fn ensure_positive_bounded(name: &str, value: u32, maximum: u32) -> Result<()> {
    ensure!(
        (1..=maximum).contains(&value),
        "{name} must be between 1 and {maximum}"
    );
    Ok(())
}

fn ensure_finite(name: &str, value: f64) -> Result<()> {
    ensure!(value.is_finite(), "{name} is outside the supported range");
    Ok(())
}

fn finite_product(name: &str, left: f64, right: f64) -> Result<f64> {
    let product = left * right;
    ensure_finite(name, product)?;
    Ok(product)
}

fn checked_u128_product(name: &str, factors: &[u128]) -> Result<u128> {
    factors.iter().try_fold(1_u128, |product, factor| {
        product
            .checked_mul(*factor)
            .ok_or_else(|| anyhow::anyhow!("{name} overflows the supported range"))
    })
}

fn assumption(code: CapacityAssumptionCode, detail: &str) -> CapacityAssumption {
    CapacityAssumption {
        code,
        detail: detail.to_owned(),
    }
}

const fn assumption_code(code: CapacityAssumptionCode) -> &'static str {
    match code {
        CapacityAssumptionCode::ParameterPrecisionWeightEstimate => {
            "parameter_precision_weight_estimate"
        }
        CapacityAssumptionCode::ArtifactSerializedBytesAreResidencyFloor => {
            "artifact_serialized_bytes_are_residency_floor"
        }
        CapacityAssumptionCode::ArtifactDoesNotProveRuntimePlacement => {
            "artifact_does_not_prove_runtime_placement"
        }
        CapacityAssumptionCode::ConservativeSharedWeightPlacement => {
            "conservative_shared_weight_placement"
        }
        CapacityAssumptionCode::ExplicitSharedRankWeightUpperBound => {
            "explicit_shared_rank_weight_upper_bound"
        }
        CapacityAssumptionCode::WeightOverheadIsUnsharded => "weight_overhead_is_unsharded",
        CapacityAssumptionCode::ConservativePipelineStageWeights => {
            "conservative_pipeline_stage_weights"
        }
        CapacityAssumptionCode::ExplicitPipelineStageComponentWeightUpperBound => {
            "explicit_pipeline_stage_component_weight_upper_bound"
        }
        CapacityAssumptionCode::ConservativePipelineStageLayers => {
            "conservative_pipeline_stage_layers"
        }
        CapacityAssumptionCode::ExplicitPipelineStageLayerUpperBound => {
            "explicit_pipeline_stage_layer_upper_bound"
        }
        CapacityAssumptionCode::KvCacheUsesLayerPlacement => "kv_cache_uses_layer_placement",
        CapacityAssumptionCode::ConservativeKvHeadPlacement => "conservative_kv_head_placement",
        CapacityAssumptionCode::ExplicitKvHeadPlacementUpperBound => {
            "explicit_kv_head_placement_upper_bound"
        }
        CapacityAssumptionCode::DataParallelConcurrencyIsPerReplica => {
            "data_parallel_concurrency_is_per_replica"
        }
        CapacityAssumptionCode::ConservativeExpertWeightPlacement => {
            "conservative_expert_weight_placement"
        }
        CapacityAssumptionCode::ExplicitExpertRankWeightUpperBound => {
            "explicit_expert_rank_weight_upper_bound"
        }
        CapacityAssumptionCode::UniformGpuMemory => "uniform_gpu_memory",
        CapacityAssumptionCode::RuntimeOverheadIsPerRank => "runtime_overhead_is_per_rank",
        CapacityAssumptionCode::LegacyGpusInterpretedAsTp => "legacy_gpus_interpreted_as_tp",
    }
}

const fn parameter_source_code(source: ModelParameterSource) -> &'static str {
    match source {
        ModelParameterSource::Unknown => "unknown",
        ModelParameterSource::ExplicitOverride => "explicit_override",
        ModelParameterSource::ConfigDeclared => "config_declared",
        ModelParameterSource::DenseEstimate => "dense_estimate",
    }
}

fn bytes_to_gib(bytes: f64) -> f64 {
    bytes / 1024.0_f64.powi(3)
}

fn bytes_to_mib(bytes: f64) -> f64 {
    bytes / 1024.0_f64.powi(2)
}

#[cfg(test)]
mod tests {
    use super::super::artifact::{ArtifactDtypeSummary, ArtifactSummary, ArtifactVerification};
    use super::*;

    const EPSILON: f64 = 1.0e-9;

    fn parameters_for_weight_gib(weight_gib: f64, weight_bits: f64) -> f64 {
        weight_gib * 1024.0_f64.powi(3) * 8.0 / weight_bits / 1_000_000_000.0
    }

    fn common_input(weight_gib: f64) -> CapacityInput {
        CapacityInput {
            parameters_billion: parameters_for_weight_gib(weight_gib, 8.0),
            parameter_source: ModelParameterSource::ExplicitOverride,
            model_type: None,
            weight_bits: 8.0,
            expected_gpu_count: Some(4),
            tensor_parallel_size: 4,
            pipeline_parallel_size: 1,
            data_parallel_size: 1,
            expert_parallel_size: 1,
            gpu_vram_gib: 80.0,
            memory_utilization: 0.9,
            layers: 8,
            kv_heads: 8,
            max_kv_heads_per_rank: Some(2),
            head_dim: 64,
            context_tokens: 1024,
            concurrent_sequences_per_data_parallel_replica: 8,
            kv_cache_bits: 16.0,
            runtime_overhead_gib_per_rank: 4.0,
            weight_overhead_percent: 0.0,
            max_shared_rank_weight_percent: Some(25.0),
            max_pipeline_stage_component_weight_percent: None,
            max_pipeline_stage_layers: None,
            expert_count: None,
            expert_weight_percent: None,
            max_expert_rank_weight_percent: None,
        }
    }

    fn assert_near(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= EPSILON,
            "expected {expected}, got {actual}"
        );
    }

    fn artifact_report(serialized_tensor_bytes: u64) -> ArtifactReport {
        let tensor_count = u64::from(serialized_tensor_bytes > 0);
        let dtypes = (serialized_tensor_bytes > 0)
            .then(|| ArtifactDtypeSummary {
                dtype: "U8".to_owned(),
                tensor_count: 1,
                tensor_elements: serialized_tensor_bytes,
                serialized_bytes: serialized_tensor_bytes,
                shape_payload_bytes_verified: true,
            })
            .into_iter()
            .collect();
        ArtifactReport {
            artifact_version: ARTIFACT_REPORT_VERSION,
            artifact_format: ArtifactFormat::Safetensors,
            layout: ArtifactLayout::SingleFile,
            summary: ArtifactSummary {
                shard_files: 1,
                tensor_count,
                tensor_elements: serialized_tensor_bytes,
                serialized_tensor_bytes,
                serialized_shard_file_bytes: serialized_tensor_bytes + 10,
                safetensors_header_json_bytes: 2,
                safetensors_length_prefix_bytes: 8,
                index_file_bytes: None,
                declared_total_size_bytes: None,
            },
            dtypes,
            verification: ArtifactVerification {
                regular_shard_files: true,
                final_symlinks_rejected: true,
                directory_descriptors_anchored: cfg!(unix),
                headers_validated: true,
                data_offsets_complete_without_holes: true,
                index_membership_validated: None,
                declared_total_size_validated: None,
                shape_payload_bytes_verified_tensors: tensor_count,
                shape_payload_bytes_unverified_tensors: 0,
                tensor_payload_contents_read: false,
                payload_checksum_validated: false,
            },
            caveats: Vec::new(),
        }
    }

    fn gib_bytes(gib: u64) -> u64 {
        gib * 1024 * 1024 * 1024
    }

    #[test]
    fn vector_one_legacy_four_gpu_tensor_parallel_plan() {
        let report = estimate(&common_input(64.0)).unwrap();

        assert_eq!(report.capacity_version, 3);
        assert_eq!(
            report.weights.base_weight_basis,
            CapacityWeightBasis::ParameterPrecisionEstimate
        );
        assert!(report.artifact.is_none());
        assert_eq!(report.topology.world_size, 4);
        assert_near(report.weights.weight_memory_gib_worst_rank, 16.0);
        assert_near(report.kv_cache.kv_cache_mib_per_sequence_logical, 16.0);
        assert_near(
            report
                .kv_cache
                .kv_cache_mib_per_sequence_physical_per_data_parallel_replica_upper_bound,
            16.0,
        );
        assert_near(report.kv_cache.kv_cache_mib_per_sequence_worst_rank, 4.0);
        assert_near(report.memory.estimated_memory_gib_worst_rank, 20.03125);
        assert_near(report.memory.headroom_gib_worst_rank, 51.96875);
        assert_eq!(
            report
                .memory
                .estimated_max_concurrent_sequences_per_data_parallel_replica,
            13_312
        );
        assert!(report.fits);
    }

    #[test]
    fn vector_two_data_parallelism_does_not_divide_replica_concurrency() {
        let mut input = common_input(64.0);
        input.expected_gpu_count = Some(8);
        input.tensor_parallel_size = 1;
        input.max_shared_rank_weight_percent = Some(100.0);
        input.max_kv_heads_per_rank = Some(8);
        input.data_parallel_size = 8;
        input.gpu_vram_gib = 24.0;

        let report = estimate(&input).unwrap();

        assert_near(report.weights.weight_memory_gib_worst_rank, 64.0);
        assert_near(
            report
                .kv_cache
                .kv_cache_mib_per_concurrency_slot_across_deployment_upper_bound,
            128.0,
        );
        assert_near(report.kv_cache.kv_cache_gib_requested_worst_rank, 0.125);
        assert_near(report.memory.estimated_memory_gib_worst_rank, 68.125);
        assert_near(report.memory.headroom_gib_worst_rank, -46.525);
        assert!(!report.fits);
    }

    #[test]
    fn vector_three_kv_heads_are_replicated_when_tp_exceeds_heads() {
        let mut input = common_input(64.0);
        input.expected_gpu_count = Some(16);
        input.tensor_parallel_size = 16;
        input.max_shared_rank_weight_percent = Some(6.25);
        input.max_kv_heads_per_rank = Some(1);
        input.concurrent_sequences_per_data_parallel_replica = 32;

        let report = estimate(&input).unwrap();

        assert_eq!(
            report
                .topology
                .kv_heads_per_tensor_parallel_rank_upper_bound,
            1
        );
        assert_near(report.topology.kv_head_replication_upper_bound, 2.0);
        assert_near(report.kv_cache.kv_cache_mib_per_sequence_logical, 16.0);
        assert_near(
            report
                .kv_cache
                .kv_cache_mib_per_sequence_physical_per_data_parallel_replica_upper_bound,
            32.0,
        );
        assert_near(report.weights.weight_memory_gib_worst_rank, 4.0);
        assert_near(report.kv_cache.kv_cache_mib_per_sequence_worst_rank, 2.0);
        assert_near(report.kv_cache.kv_cache_gib_requested_worst_rank, 0.0625);
        assert_near(report.memory.estimated_memory_gib_worst_rank, 8.0625);
    }

    #[test]
    fn vector_four_pipeline_plan_uses_worst_stage_weights_and_layers() {
        let mut input = common_input(64.0);
        input.expected_gpu_count = Some(6);
        input.tensor_parallel_size = 2;
        input.max_shared_rank_weight_percent = Some(50.0);
        input.max_kv_heads_per_rank = Some(4);
        input.pipeline_parallel_size = 3;
        input.concurrent_sequences_per_data_parallel_replica = 10;
        input.max_pipeline_stage_component_weight_percent = Some(40.0);
        input.max_pipeline_stage_layers = Some(3);

        let report = estimate(&input).unwrap();

        assert_eq!(report.topology.world_size, 6);
        assert_eq!(report.topology.worst_pipeline_stage_layers, 3);
        assert_near(report.weights.weight_memory_gib_worst_rank, 12.8);
        assert_near(report.kv_cache.kv_cache_mib_per_sequence_worst_rank, 3.0);
        assert_near(
            report.kv_cache.kv_cache_gib_requested_worst_rank,
            0.029_296_875,
        );
        assert_near(
            report.memory.estimated_memory_gib_worst_rank,
            16.829_296_875,
        );
        assert_near(report.memory.headroom_gib_worst_rank, 55.170_703_125);
    }

    #[test]
    fn vector_five_expert_parallel_weights_are_not_divided_by_tp_twice() {
        let mut input = common_input(80.0);
        input.expected_gpu_count = Some(8);
        input.data_parallel_size = 2;
        input.expert_parallel_size = 8;
        input.gpu_vram_gib = 24.0;
        input.concurrent_sequences_per_data_parallel_replica = 16;
        input.expert_count = Some(64);
        input.expert_weight_percent = Some(75.0);
        input.max_expert_rank_weight_percent = Some(12.5);

        let report = estimate(&input).unwrap();

        assert_near(report.weights.shared_weight_memory_gib_worst_rank, 5.0);
        assert_near(report.weights.expert_weight_memory_gib_worst_rank, 7.5);
        assert_near(report.weights.weight_memory_gib_worst_rank, 12.5);
        assert_near(report.kv_cache.kv_cache_mib_per_sequence_worst_rank, 4.0);
        assert_near(report.memory.estimated_memory_gib_worst_rank, 16.5625);
        assert_near(report.memory.headroom_gib_worst_rank, 5.0375);
        assert_eq!(
            report
                .memory
                .estimated_max_concurrent_sequences_per_data_parallel_replica,
            1305
        );
        assert!(report.fits);
    }

    #[test]
    fn vector_six_rejects_expected_gpu_count_mismatch() {
        let mut input = common_input(64.0);
        input.expected_gpu_count = Some(8);
        input.tensor_parallel_size = 2;
        input.max_shared_rank_weight_percent = Some(50.0);
        input.pipeline_parallel_size = 2;
        input.data_parallel_size = 3;

        let error = estimate(&input).unwrap_err().to_string();

        assert!(error.contains("expected GPU count (8)"));
        assert!(error.contains("world size (12)"));
    }

    #[test]
    fn rejects_kv_head_bound_below_the_pigeonhole_minimum() {
        let mut input = common_input(64.0);
        input.expected_gpu_count = Some(3);
        input.tensor_parallel_size = 3;
        input.max_shared_rank_weight_percent = Some(100.0 / 3.0);

        assert!(
            estimate(&input)
                .unwrap_err()
                .to_string()
                .contains("max_kv_heads_per_rank must be between 3 and 8")
        );
    }

    #[test]
    fn rejects_missing_or_non_divisible_expert_metadata() {
        let mut missing = common_input(64.0);
        missing.expected_gpu_count = Some(8);
        missing.data_parallel_size = 2;
        missing.expert_parallel_size = 8;
        assert!(
            estimate(&missing)
                .unwrap_err()
                .to_string()
                .contains("requires expert_count")
        );

        let mut non_divisible = missing;
        non_divisible.expert_count = Some(63);
        non_divisible.expert_weight_percent = Some(75.0);
        assert!(
            estimate(&non_divisible)
                .unwrap_err()
                .to_string()
                .contains("expert_count (63) must be divisible")
        );
    }

    #[test]
    fn rejects_pipeline_stages_without_layers_or_a_valid_weight_bound() {
        let mut too_many_stages = common_input(64.0);
        too_many_stages.expected_gpu_count = Some(36);
        too_many_stages.pipeline_parallel_size = 9;
        assert!(
            estimate(&too_many_stages)
                .unwrap_err()
                .to_string()
                .contains("exceeds layer count")
        );

        let mut impossible_bound = common_input(64.0);
        impossible_bound.expected_gpu_count = Some(12);
        impossible_bound.pipeline_parallel_size = 3;
        impossible_bound.max_pipeline_stage_component_weight_percent = Some(30.0);
        assert!(
            estimate(&impossible_bound)
                .unwrap_err()
                .to_string()
                .contains("max_pipeline_stage_component_weight_percent")
        );
    }

    #[test]
    fn multi_stage_plan_defaults_to_conservative_full_weight_placement() {
        let mut input = common_input(64.0);
        input.expected_gpu_count = Some(8);
        input.pipeline_parallel_size = 2;

        let report = estimate(&input).unwrap();

        assert_near(report.weights.weight_memory_gib_worst_rank, 16.0);
        assert_eq!(report.topology.worst_pipeline_stage_layers, 8);
        assert_near(report.kv_cache.kv_cache_mib_per_sequence_worst_rank, 4.0);
        assert!(report.assumptions.iter().any(|assumption| {
            assumption.code == CapacityAssumptionCode::ConservativePipelineStageWeights
        }));
        assert!(report.assumptions.iter().any(|assumption| {
            assumption.code == CapacityAssumptionCode::ConservativePipelineStageLayers
        }));
    }

    #[test]
    fn rank_sharding_credit_requires_explicit_upper_bounds() {
        let mut shared = common_input(64.0);
        shared.max_shared_rank_weight_percent = None;
        let shared_report = estimate(&shared).unwrap();
        assert_near(
            shared_report.weights.shared_weight_memory_gib_worst_rank,
            64.0,
        );
        assert!(shared_report.assumptions.iter().any(|assumption| {
            assumption.code == CapacityAssumptionCode::ConservativeSharedWeightPlacement
        }));

        let mut expert = common_input(64.0);
        expert.expert_parallel_size = 4;
        expert.expert_count = Some(4);
        expert.expert_weight_percent = Some(100.0);
        let expert_report = estimate(&expert).unwrap();
        assert_near(
            expert_report.weights.expert_weight_memory_gib_worst_rank,
            64.0,
        );
        assert!(expert_report.assumptions.iter().any(|assumption| {
            assumption.code == CapacityAssumptionCode::ConservativeExpertWeightPlacement
        }));
    }

    #[test]
    fn kv_sharding_credit_requires_an_explicit_head_bound() {
        let mut input = common_input(64.0);
        input.max_kv_heads_per_rank = None;

        let report = estimate(&input).unwrap();

        assert_eq!(
            report
                .topology
                .kv_heads_per_tensor_parallel_rank_upper_bound,
            8
        );
        assert_near(report.topology.kv_head_replication_upper_bound, 4.0);
        assert_near(report.kv_cache.kv_cache_mib_per_sequence_worst_rank, 16.0);
        assert_near(
            report
                .kv_cache
                .kv_cache_mib_per_sequence_physical_per_data_parallel_replica_upper_bound,
            64.0,
        );
        assert!(report.assumptions.iter().any(|assumption| {
            assumption.code == CapacityAssumptionCode::ConservativeKvHeadPlacement
        }));
    }

    #[test]
    fn weight_overhead_is_not_divided_by_tensor_or_expert_parallelism() {
        let mut input = common_input(64.0);
        input.weight_overhead_percent = 10.0;

        let report = estimate(&input).unwrap();

        assert_near(report.weights.base_weight_memory_gib_logical, 64.0);
        assert_near(report.weights.weight_overhead_memory_gib_logical, 6.4);
        assert_near(report.weights.shared_weight_memory_gib_worst_rank, 16.0);
        assert_near(report.weights.weight_overhead_memory_gib_worst_rank, 6.4);
        assert_near(report.weights.weight_memory_gib_worst_rank, 22.4);
    }

    #[test]
    fn rejects_non_finite_values_and_checked_world_size_overflow() {
        let mut non_finite = common_input(64.0);
        non_finite.parameters_billion = f64::NAN;
        assert!(
            estimate(&non_finite)
                .unwrap_err()
                .to_string()
                .contains("must be finite")
        );

        let mut overflow = common_input(64.0);
        overflow.expected_gpu_count = None;
        overflow.tensor_parallel_size = MAX_WORLD_SIZE;
        overflow.pipeline_parallel_size = MAX_WORLD_SIZE;
        overflow.data_parallel_size = MAX_WORLD_SIZE;
        overflow.layers = MAX_MODEL_DIMENSION;
        overflow.kv_heads = MAX_WORLD_SIZE;
        assert!(
            estimate(&overflow)
                .unwrap_err()
                .to_string()
                .contains("overflows")
        );
    }

    #[test]
    fn rejects_ambiguous_or_unsafe_parameter_provenance() {
        let mut unknown = common_input(64.0);
        unknown.parameter_source = ModelParameterSource::Unknown;
        assert!(
            estimate(&unknown)
                .unwrap_err()
                .to_string()
                .contains("parameter_source")
        );

        let mut missing_type = common_input(64.0);
        missing_type.parameter_source = ModelParameterSource::ConfigDeclared;
        assert!(
            estimate(&missing_type)
                .unwrap_err()
                .to_string()
                .contains("requires model_type")
        );

        let mut unsafe_type = common_input(64.0);
        unsafe_type.model_type = Some("https://secret.example/model".to_owned());
        assert!(
            estimate(&unsafe_type)
                .unwrap_err()
                .to_string()
                .contains("ASCII identifier")
        );
    }

    #[test]
    fn artifact_floor_never_reduces_the_parameter_precision_estimate() {
        let input = common_input(64.0);

        let below = estimate_with_artifact(&input, &artifact_report(gib_bytes(32)), 1.0).unwrap();
        assert_near(below.weights.base_weight_memory_gib_logical, 64.0);
        assert_eq!(
            below.weights.base_weight_basis,
            CapacityWeightBasis::ParameterPrecisionEstimate
        );
        assert!(!below.artifact.unwrap().selected_as_base_weight_floor);

        let equal = estimate_with_artifact(&input, &artifact_report(gib_bytes(64)), 1.0).unwrap();
        assert_near(equal.weights.base_weight_memory_gib_logical, 64.0);
        assert_eq!(
            equal.weights.base_weight_basis,
            CapacityWeightBasis::ParameterPrecisionEstimate
        );

        let above = estimate_with_artifact(&input, &artifact_report(gib_bytes(80)), 1.0).unwrap();
        assert_near(above.weights.base_weight_memory_gib_logical, 80.0);
        assert_eq!(
            above.weights.base_weight_basis,
            CapacityWeightBasis::ArtifactResidencyFloor
        );
        assert!(above.artifact.unwrap().selected_as_base_weight_floor);
    }

    #[test]
    fn artifact_multiplier_is_monotonic_and_overhead_applies_once_after_max() {
        let mut input = common_input(64.0);
        input.weight_overhead_percent = 10.0;
        let artifact = artifact_report(gib_bytes(40));

        let doubled = estimate_with_artifact(&input, &artifact, 2.0).unwrap();
        assert_near(
            doubled.weights.parameter_derived_weight_memory_gib_logical,
            64.0,
        );
        assert_near(
            doubled
                .weights
                .artifact_adjusted_weight_floor_gib_logical
                .unwrap(),
            80.0,
        );
        assert_near(doubled.weights.base_weight_memory_gib_logical, 80.0);
        assert_near(doubled.weights.weight_overhead_memory_gib_logical, 8.0);
        assert_near(doubled.weights.weight_memory_gib_worst_rank, 28.0);

        let tripled = estimate_with_artifact(&input, &artifact, 3.0).unwrap();
        assert!(
            tripled.memory.estimated_memory_gib_worst_rank
                > doubled.memory.estimated_memory_gib_worst_rank
        );
    }

    #[test]
    fn artifact_evidence_can_only_preserve_or_reduce_fit_and_grants_no_placement_credit() {
        let mut input = common_input(64.0);
        input.max_shared_rank_weight_percent = None;
        let baseline = estimate(&input).unwrap();
        let strengthened =
            estimate_with_artifact(&input, &artifact_report(gib_bytes(80)), 1.0).unwrap();

        assert!(baseline.fits);
        assert!(!strengthened.fits);
        assert_near(
            strengthened.weights.shared_weight_memory_gib_worst_rank,
            80.0,
        );
        assert!(strengthened.assumptions.iter().any(|assumption| {
            assumption.code == CapacityAssumptionCode::ArtifactDoesNotProveRuntimePlacement
        }));
    }

    #[test]
    fn rejects_invalid_multiplier_or_forged_artifact_evidence() {
        let input = common_input(64.0);
        let artifact = artifact_report(gib_bytes(64));
        for multiplier in [f64::NAN, f64::INFINITY, 0.999, 1_001.0] {
            assert!(estimate_with_artifact(&input, &artifact, multiplier).is_err());
        }

        let mut unverified = artifact.clone();
        unverified.verification.headers_validated = false;
        assert!(estimate_with_artifact(&input, &unverified, 1.0).is_err());

        let mut inconsistent = artifact;
        inconsistent.dtypes[0].serialized_bytes -= 1;
        assert!(estimate_with_artifact(&input, &inconsistent, 1.0).is_err());

        let mut unknown_dtype = artifact_report(8);
        unknown_dtype.dtypes[0].dtype = "UNKNOWN".to_owned();
        assert!(estimate_with_artifact(&input, &unknown_dtype, 1.0).is_err());

        let mut impossible_dtype_geometry = artifact_report(8);
        impossible_dtype_geometry.dtypes[0].tensor_elements = 1;
        impossible_dtype_geometry.summary.tensor_elements = 1;
        assert!(estimate_with_artifact(&input, &impossible_dtype_geometry, 1.0).is_err());

        let excessive = artifact_report(MAX_ARTIFACT_ADJUSTED_BYTES + 1);
        assert!(estimate_with_artifact(&input, &excessive, 1.0).is_err());
    }

    #[test]
    fn artifact_floor_rounding_is_exact_for_the_f64_multiplier_value() {
        let bytes = 4_503_599_627_370_497;
        let multiplier = 1.000_000_000_000_000_2;

        assert_eq!(
            checked_ceil_scaled_bytes(bytes, multiplier).unwrap(),
            4_503_599_627_370_499
        );
        assert_eq!(checked_ceil_scaled_bytes(8, 1.125).unwrap(), 9);
        assert_eq!(checked_ceil_scaled_bytes(0, 1_000.0).unwrap(), 0);
    }

    #[test]
    fn empty_artifact_is_valid_but_cannot_weaken_capacity() {
        let input = common_input(64.0);
        let report = estimate_with_artifact(&input, &artifact_report(0), 100.0).unwrap();

        assert_near(report.weights.base_weight_memory_gib_logical, 64.0);
        assert_eq!(
            report.weights.base_weight_basis,
            CapacityWeightBasis::ParameterPrecisionEstimate
        );
        assert_eq!(report.artifact.unwrap().serialized_tensor_bytes, 0);
    }

    #[test]
    fn text_rendering_uses_versioned_report_evidence() {
        let report = estimate(&common_input(64.0)).unwrap();
        let text = render_text(&report);

        assert!(text.contains("capacity report v3  FITS"));
        assert!(text.contains("Artifact    not supplied"));
        assert!(text.contains("parameter-precision estimate"));
        assert!(text.contains("TP 4 | PP 1 | DP 1 | EP 1 | world 4"));
        assert!(text.contains("sequence(s)/DP replica"));
        assert!(text.contains("[data_parallel_concurrency_is_per_replica]"));
    }
}
