//! Core trait definitions for the Context Graph system.

mod embedding_provider;
mod graph_index;
mod memory_store;
mod nervous_layer;
mod utl_processor;

pub use embedding_provider::{EmbeddingOutput, EmbeddingProvider};
pub use graph_index::GraphIndex;
pub use memory_store::{MemoryStore, SearchOptions};
pub use nervous_layer::NervousLayer;
pub use utl_processor::UtlProcessor;
