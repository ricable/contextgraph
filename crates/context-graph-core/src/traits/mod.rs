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
mod teleological_memory_store;
mod utl_processor;

#[cfg(test)]
mod teleological_memory_store_tests;

// NOTE: EmbeddingProvider has been DELETED - backwards compat cleanup
// Use MultiArrayEmbeddingProvider for 13-embedding SemanticFingerprint generation

// Graph index trait
pub use graph_index::GraphIndex;

// Multi-array embedding provider (13 embeddings) - TASK-F007
// E4-FIX: Added EmbeddingMetadata for session sequence support
pub use multi_array_embedding::{
    EmbeddingMetadata, MultiArrayEmbeddingOutput, MultiArrayEmbeddingProvider, SingleEmbedder,
    SparseEmbedder, TokenEmbedder,
};

// Teleological memory store trait - TASK-F008
pub use teleological_memory_store::{
    NormalizationStrategyOption, SearchStrategy, TeleologicalMemoryStore,
    TeleologicalMemoryStoreExt, TeleologicalSearchOptions, TeleologicalSearchResult,
    TeleologicalStorageBackend, TemporalBreakdown,
};

// Temporal search options (ARCH-14)
pub use teleological_memory_store::{
    ChainRetrievalOptions, DecayFunction, MultiAnchorMode, PeriodicOptions, SequenceDirection,
    SequenceOptions, TemporalScale, TemporalSearchOptions, TimeWindow,
};

// E10 Intent Gate options (Phase 4)
pub use teleological_memory_store::IntentDirection;

// UTL processor trait
pub use utl_processor::UtlProcessor;
