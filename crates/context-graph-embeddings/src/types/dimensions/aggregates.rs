//! Aggregate dimensions and compile-time validation.
//!
//! This module defines the total dimension for the 12-model Multi-Array Storage.

use super::constants::{
    CAUSAL, CODE, ENTITY, GRAPH, HDC, LATE_INTERACTION, MULTIMODAL, SEMANTIC, SPARSE,
    TEMPORAL_PERIODIC, TEMPORAL_POSITIONAL, TEMPORAL_RECENT,
};

// =============================================================================
// AGGREGATE DIMENSIONS
// =============================================================================

/// Total dimension across all 12 model embeddings (sum of projected dimensions).
///
/// Each embedding is stored SEPARATELY in Multi-Array Storage at its native dimension.
/// This constant represents the sum of all dimensions for memory allocation.
///
/// Calculated as:
/// 1024 + 512 + 512 + 512 + 768 + 1536 + 768 + 384 + 1024 + 768 + 384 + 128 = 8320
pub const TOTAL_DIMENSION: usize = SEMANTIC
    + TEMPORAL_RECENT
    + TEMPORAL_PERIODIC
    + TEMPORAL_POSITIONAL
    + CAUSAL
    + SPARSE
    + CODE
    + GRAPH
    + HDC
    + MULTIMODAL
    + ENTITY
    + LATE_INTERACTION;

/// Number of models in the ensemble.
pub const MODEL_COUNT: usize = 12;

// =============================================================================
// COMPILE-TIME VALIDATION
// =============================================================================

/// Compile-time assertion that TOTAL_DIMENSION equals expected value.
/// This will cause a compilation error if dimensions change incorrectly.
const _TOTAL_DIMENSION_CHECK: () = assert!(
    TOTAL_DIMENSION == 8320,
    "TOTAL_DIMENSION must equal 8320"
);

/// Compile-time assertion that MODEL_COUNT equals 12.
const _MODEL_COUNT_CHECK: () = assert!(MODEL_COUNT == 12, "MODEL_COUNT must equal 12");
