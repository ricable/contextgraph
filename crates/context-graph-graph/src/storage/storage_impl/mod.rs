//! GraphStorage backend wrapping RocksDB.
//!
//! Provides type-safe persistence for hyperbolic coordinates, entailment cones,
//! and edge adjacency lists.
//!
//! # Constitution Reference
//!
//! - AP-001: Never unwrap() in prod - all errors properly typed
//! - rules: Result<T,E> for fallible ops, thiserror for derivation
//!
//! # Binary Formats
//!
//! - PoincarePoint: 256 bytes (64 f32 little-endian)
//! - EntailmentCone: 268 bytes (256 apex + 4 aperture + 4 factor + 4 depth)
//! - NodeId: 8 bytes (i64 little-endian)
//! - Edges: bincode serialized Vec<GraphEdge>
//!
//! # Module Structure
//!
//! - [`types`]: Core types (PoincarePoint, EntailmentCone, NodeId, LegacyGraphEdge)
//! - [`serialization`]: Binary serialization for points and cones
//! - [`core`]: GraphStorage struct and initialization
//! - [`hyperbolic`]: Hyperbolic point operations
//! - [`cones`]: Entailment cone operations
//! - [`adjacency`]: Adjacency list operations
//! - [`batch`]: Batch write operations
//! - [`iteration`]: Iteration over stored data
//! - [`stats`]: Statistics and schema version operations
//! - [`edges`]: Full GraphEdge operations (M04-T15)

// ========== Submodules ==========

mod adjacency;
mod batch;
mod cones;
mod core;
mod graph_edges;
mod hyperbolic;
mod iteration;
mod serialization;
mod stats;
mod types;

#[cfg(test)]
mod graph_edges_tests;

// ========== Re-exports ==========

// Core types
pub use types::{EntailmentCone, LegacyGraphEdge, NodeId, PoincarePoint};

// GraphStorage
pub use core::GraphStorage;
