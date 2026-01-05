//! Stub implementations for development and testing.
//!
//! These implementations provide deterministic mock behavior
//! for the Ghost System phase (Phase 0).
//!
//! Note: `InMemoryStore` has been moved to `context_graph_core::memory`
//! and is re-exported here for backward compatibility.

mod embedding_stub;
mod graph_index;
mod layers;
mod utl_stub;

pub use embedding_stub::StubEmbeddingProvider;
pub use graph_index::InMemoryGraphIndex;
pub use layers::{
    StubCoherenceLayer, StubLearningLayer, StubMemoryLayer, StubReflexLayer, StubSensingLayer,
};
pub use utl_stub::StubUtlProcessor;

// Re-export InMemoryStore from memory module for backward compatibility
pub use crate::memory::InMemoryStore;
