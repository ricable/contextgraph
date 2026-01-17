//! Embedding types for the 13-model teleological array.
//!
//! This module provides:
//! - `TokenPruningEmbedding` (E12): Token-level embedding with Quantizable support
//! - `DenseVector`: Generic dense vector for similarity computation
//! - `BinaryVector`: Bit-packed vector for Hamming distance
//!
//! Note: `SparseVector` for SPLADE is in `types::fingerprint::sparse`.

pub mod token_pruning;
pub mod vector;

pub use token_pruning::TokenPruningEmbedding;
pub use vector::{BinaryVector, DenseVector, VectorError};
