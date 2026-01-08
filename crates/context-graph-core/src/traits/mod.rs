//! Core trait definitions for the Context Graph system.
//!
//! This module defines the key traits that form the contract between
//! components of the teleological memory architecture.
//!
//! # Traits
//!
//! - [`TeleologicalMemoryStore`]: Storage for teleological fingerprints (TASK-F008)
//! - [`MultiArrayEmbeddingProvider`]: 13-embedding generation (TASK-F007)
//! - [`GraphIndex`]: Graph traversal and indexing
//! - [`NervousLayer`]: Nervous system layer processing
//! - [`UtlProcessor`]: Unified Theory of Learning operations
//!
//! # REMOVED (Phase 1 Cleanup)
//!
//! - `EmbeddingProvider`: Deprecated single-embedding trait has been DELETED.
//!   Use `MultiArrayEmbeddingProvider` for 13-embedding SemanticFingerprint generation.

// NOTE: embedding_provider.rs has been DELETED - backwards compat cleanup
// Use multi_array_embedding::MultiArrayEmbeddingProvider instead
mod graph_index;
mod multi_array_embedding;
mod nervous_layer;
mod teleological_memory_store;
mod utl_processor;

#[cfg(test)]
mod teleological_memory_store_tests;

// NOTE: EmbeddingProvider has been DELETED - backwards compat cleanup
// Use MultiArrayEmbeddingProvider for 13-embedding SemanticFingerprint generation

// Graph index trait
pub use graph_index::GraphIndex;

// Multi-array embedding provider (13 embeddings) - TASK-F007
pub use multi_array_embedding::{
    MultiArrayEmbeddingOutput, MultiArrayEmbeddingProvider, SingleEmbedder, SparseEmbedder,
    TokenEmbedder,
};

// Nervous system layer trait
pub use nervous_layer::NervousLayer;

// Teleological memory store trait - TASK-F008
pub use teleological_memory_store::{
    TeleologicalMemoryStore, TeleologicalMemoryStoreExt, TeleologicalSearchOptions,
    TeleologicalSearchResult, TeleologicalStorageBackend,
};

// UTL processor trait
pub use utl_processor::UtlProcessor;
