//! Single-embedder baseline for comparison.
//!
//! This module provides a traditional single-embedding RAG baseline using only E1 (1024D semantic).
//! This allows us to compare against the full 13-embedder multi-space approach.

pub mod single_embedder;
pub mod single_hnsw;

pub use single_embedder::SingleEmbedderBaseline;
pub use single_hnsw::SingleHnswIndex;
