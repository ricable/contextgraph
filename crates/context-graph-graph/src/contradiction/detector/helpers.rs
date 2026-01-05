//! Helper functions for contradiction detection.
//!
//! Contains utility functions for ID conversion, scoring, and type inference.

use std::time::{SystemTime, UNIX_EPOCH};

use uuid::Uuid;

use crate::storage::edges::GraphEdge;

use super::types::{CandidateInfo, ContradictionParams, ContradictionType, Domain};

/// Convert UUID to i64 for storage key operations.
#[inline]
pub fn uuid_to_i64(uuid: &Uuid) -> i64 {
    let bytes = uuid.as_bytes();
    i64::from_be_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

/// Generate unique edge ID from current time.
pub fn generate_edge_id() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

/// Compute confidence score from candidate info.
pub(crate) fn compute_confidence(info: &CandidateInfo, params: &ContradictionParams) -> f32 {
    let semantic_component = info.semantic_similarity * (1.0 - params.explicit_edge_weight);

    let edge_component = if info.has_explicit_edge {
        info.edge_weight.unwrap_or(0.5) * params.explicit_edge_weight
    } else {
        0.0
    };

    // Boost if both semantic and explicit evidence
    let combined = semantic_component + edge_component;
    let boost = if info.has_explicit_edge && info.semantic_similarity > 0.5 {
        1.2 // 20% boost for corroborating evidence
    } else {
        1.0
    };

    (combined * boost).clamp(0.0, 1.0)
}

/// Infer contradiction type from edge metadata.
pub fn infer_contradiction_type_from_edge(edge: &GraphEdge) -> ContradictionType {
    // Use domain hint for classification
    match edge.domain {
        Domain::Code => ContradictionType::LogicalInconsistency,
        _ => ContradictionType::DirectOpposition,
    }
}

/// Infer type from semantic similarity pattern.
pub fn infer_type_from_similarity(similarity: f32) -> ContradictionType {
    if similarity > 0.9 {
        ContradictionType::DirectOpposition
    } else if similarity > 0.7 {
        ContradictionType::LogicalInconsistency
    } else if similarity > 0.5 {
        ContradictionType::TemporalConflict
    } else {
        ContradictionType::CausalConflict
    }
}
