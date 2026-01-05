//! RocksDB storage backend for graph data.
//!
//! This module provides persistent storage for the knowledge graph using
//! RocksDB with column families for efficient data organization.
//!
//! # Column Families
//!
//! | Column Family | Key | Value | Optimization |
//! |---------------|-----|-------|--------------|
//! | adjacency | NodeId (8B i64) | Vec<GraphEdge> (bincode) | Prefix scans |
//! | hyperbolic | NodeId (8B i64) | [f32; 64] = 256 bytes | Point lookups |
//! | entailment_cones | NodeId (8B i64) | EntailmentCone = 268 bytes | Bloom filter |
//! | faiss_ids | NodeId (8B i64) | i64 = 8 bytes | Point lookups |
//! | nodes | NodeId (8B i64) | MemoryNode (bincode) | Point lookups |
//! | metadata | key string | JSON value | Small CF |
//!
//! # GPU Integration
//!
//! Data stored here is loaded into GPU memory for processing:
//! - Hyperbolic coordinates -> GPU for Poincare ball operations
//! - FAISS IDs -> GPU FAISS index for vector similarity
//! - Entailment cones -> GPU for hierarchy queries
//!
//! # Constitution Reference
//!
//! - db.vector: faiss_gpu
//! - storage: RocksDB 0.22
//! - SEC-06: Soft delete 30-day recovery
//! - perf.latency.faiss_1M_k100: <2ms (storage must not bottleneck)
//!
//! # Module Structure
//!
//! - [`storage_impl`]: GraphStorage implementation (M04-T13)
//! - [`migrations`]: Schema migration system (M04-T13a)
//! - [`edges`]: GraphEdge with Marblestone NT modulation (M04-T15)
//! - [`config`]: Storage configuration
//! - [`constants`]: Column family name constants
//! - [`descriptors`]: Column family descriptor generation

// ========== Submodules ==========

pub mod config;
pub mod constants;
pub mod descriptors;
pub mod edges;
pub mod migrations;
pub mod storage_impl;

#[cfg(test)]
mod tests;

// ========== Re-exports for backwards compatibility ==========

// GraphStorage and types (M04-T13)
pub use storage_impl::{EntailmentCone, GraphStorage, LegacyGraphEdge, NodeId, PoincarePoint};

// GraphEdge with Marblestone NT modulation (M04-T15)
pub use edges::{EdgeId, GraphEdge};
// Re-export core types from edges module for convenience
pub use edges::{Domain, EdgeType, NeurotransmitterWeights};

// Migrations (M04-T13a)
pub use migrations::{MigrationInfo, Migrations, SCHEMA_VERSION};

// Storage configuration
pub use config::StorageConfig;

// Column family constants
pub use constants::{
    ALL_COLUMN_FAMILIES, CF_ADJACENCY, CF_CONES, CF_EDGES, CF_FAISS_IDS, CF_HYPERBOLIC,
    CF_METADATA, CF_NODES,
};

// Column family descriptors and DB options
pub use descriptors::{get_column_family_descriptors, get_db_options};
