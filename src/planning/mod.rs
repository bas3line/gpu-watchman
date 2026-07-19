//! Offline inference deployment planning tools.

pub mod artifact;
pub mod capacity;
pub mod model_config;

pub use artifact::{
    ARTIFACT_REPORT_VERSION, ArtifactDtypeSummary, ArtifactFormat, ArtifactLayout, ArtifactReport,
    ArtifactSummary, ArtifactVerification, inspect as inspect_artifact,
    render_text as render_artifact_text,
};
pub use capacity::{
    CapacityArtifactEvidence, CapacityInput, CapacityReport, CapacityWeightBasis, estimate,
    estimate_with_artifact,
};
pub use model_config::{
    ModelConfigEvidence, ModelGeometry, ModelGeometryOverrides, ModelParameterSource,
    load_model_config_evidence, load_model_geometry, model_geometry_from_evidence,
    model_geometry_from_value,
};
