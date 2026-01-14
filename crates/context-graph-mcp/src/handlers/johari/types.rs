//! Johari handler types and constants.
//!
//! Request/response types for all Johari MCP handlers.
//! TASK-FIX-003: Fixed to match canonical Embedder::name() values.

use serde::{Deserialize, Serialize};

use context_graph_core::johari::NUM_EMBEDDERS;

/// Embedder names for response formatting.
///
/// CANONICAL SOURCE: `Embedder::name()` in `context-graph-core/src/teleological/embedder.rs`
/// These names MUST match exactly. See TASK-FIX-003 for rationale.
pub const EMBEDDER_NAMES: [&str; 13] = [
    "E1_Semantic",
    "E2_Temporal_Recent",
    "E3_Temporal_Periodic",
    "E4_Temporal_Positional",
    "E5_Causal",
    "E6_Sparse_Lexical",
    "E7_Code",
    "E8_Emotional",
    "E9_HDC",
    "E10_Multimodal",
    "E11_Entity",
    "E12_Late_Interaction",
    "E13_SPLADE",
];

/// Validate embedder index is within bounds.
#[inline]
#[allow(dead_code)]
pub fn validate_embedder_index(idx: usize) -> bool {
    idx < NUM_EMBEDDERS
}

// ============================================================================
// Request parameters
// ============================================================================

/// Request parameters for johari/get_distribution.
#[derive(Debug, Deserialize)]
pub struct GetDistributionParams {
    pub memory_id: String,
    #[serde(default)]
    pub include_confidence: bool,
    #[serde(default)]
    pub include_transition_predictions: bool,
}

/// Request parameters for johari/find_by_quadrant.
#[derive(Debug, Deserialize)]
pub struct FindByQuadrantParams {
    pub embedder_index: usize,
    pub quadrant: String,
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f32,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_min_confidence() -> f32 {
    0.0
}

fn default_top_k() -> usize {
    100
}

/// Request parameters for johari/transition.
#[derive(Debug, Deserialize)]
pub struct TransitionParams {
    pub memory_id: String,
    pub embedder_index: usize,
    pub to_quadrant: String,
    pub trigger: String,
}

/// Single transition in a batch.
#[derive(Debug, Deserialize)]
pub struct BatchTransitionItem {
    pub embedder_index: usize,
    pub to_quadrant: String,
    pub trigger: String,
}

/// Request parameters for johari/transition_batch.
#[derive(Debug, Deserialize)]
pub struct TransitionBatchParams {
    pub memory_id: String,
    pub transitions: Vec<BatchTransitionItem>,
}

/// Request parameters for johari/cross_space_analysis.
#[derive(Debug, Deserialize)]
pub struct CrossSpaceAnalysisParams {
    pub memory_ids: Vec<String>,
    #[serde(default = "default_analysis_type")]
    #[allow(dead_code)]
    pub analysis_type: String,
}

fn default_analysis_type() -> String {
    "blind_spots".to_string()
}

/// Request parameters for johari/transition_probabilities.
#[derive(Debug, Deserialize)]
pub struct TransitionProbabilitiesParams {
    pub embedder_index: usize,
    pub memory_id: String,
}

// ============================================================================
// Response types
// ============================================================================

/// Per-embedder quadrant info for response.
#[derive(Debug, Serialize)]
pub struct EmbedderQuadrantInfo {
    pub embedder_index: usize,
    pub embedder_name: &'static str,
    pub quadrant: String,
    pub soft_classification: SoftClassification,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicted_next_quadrant: Option<String>,
}

/// Soft classification weights.
#[derive(Debug, Serialize)]
pub struct SoftClassification {
    pub open: f32,
    pub hidden: f32,
    pub blind: f32,
    pub unknown: f32,
}

/// Summary statistics for distribution response.
#[derive(Debug, Serialize)]
pub struct DistributionSummary {
    pub open_count: usize,
    pub hidden_count: usize,
    pub blind_count: usize,
    pub unknown_count: usize,
    pub average_confidence: f32,
}
