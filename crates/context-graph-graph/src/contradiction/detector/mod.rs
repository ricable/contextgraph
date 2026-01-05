//! Contradiction detection algorithm.
//!
//! # M04-T21: Implement Contradiction Detection
//!
//! Combines semantic similarity search with explicit CONTRADICTS edges
//! to identify conflicting knowledge in the graph.
//!
//! # FAIL FAST
//!
//! All errors are explicit - no graceful degradation.
//! Invalid inputs fail immediately with clear error messages.

mod detection;
mod helpers;
mod operations;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use types::{
    ContradictionParams, ContradictionResult, ContradictionType, Domain, EdgeType,
    NeurotransmitterWeights,
};

// Internal re-export for use within crate

// Re-export all public functions for backwards compatibility
pub use detection::contradiction_detect;
pub use operations::{check_contradiction, get_contradictions, mark_contradiction};

// Re-export helper functions that were previously public (used in tests)
pub use helpers::{
    generate_edge_id, infer_contradiction_type_from_edge, infer_type_from_similarity, uuid_to_i64,
};
