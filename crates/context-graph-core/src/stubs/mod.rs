//! Stub implementations for development and testing.
//!
//! These implementations provide deterministic mock behavior
//! for the Ghost System phase (Phase 0).

mod embedding_stub;
mod graph_index;
mod layers;
mod memory_stub;
mod utl_stub;

pub use embedding_stub::StubEmbeddingProvider;
pub use graph_index::InMemoryGraphIndex;
pub use layers::{
    StubCoherenceLayer, StubLearningLayer, StubMemoryLayer, StubReflexLayer, StubSensingLayer,
};
pub use memory_stub::InMemoryStore;
pub use utl_stub::StubUtlProcessor;
