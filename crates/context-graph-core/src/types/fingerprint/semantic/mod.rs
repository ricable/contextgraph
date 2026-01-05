//! SemanticFingerprint: The core 13-embedding array data structure.
//!
//! This module provides the foundational data structure for the Teleological Vector Architecture.
//! It stores all 13 embedding types WITHOUT fusion to preserve full semantic information.
//!
//! # Design Philosophy
//!
//! **NO FUSION**: Each embedding space is preserved independently for:
//! - Per-space HNSW search (13x independent indexes)
//! - Per-space Johari quadrant classification
//! - Per-space teleological alignment computation
//! - 100% information preservation
//!
//! # Storage
//!
//! Typical storage is ~46KB per fingerprint (vs ~6KB fused = 67% info loss avoided).

mod constants;
mod fingerprint;
mod slice;
mod validation;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use constants::{
    E10_DIM, E11_DIM, E12_TOKEN_DIM, E13_SPLADE_VOCAB, E1_DIM, E2_DIM, E3_DIM, E4_DIM, E5_DIM,
    E6_SPARSE_VOCAB, E7_DIM, E8_DIM, E9_DIM, NUM_EMBEDDERS, TOTAL_DENSE_DIMS,
};
pub use fingerprint::SemanticFingerprint;
pub use slice::EmbeddingSlice;
