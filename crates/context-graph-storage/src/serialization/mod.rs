//! Binary serialization utilities for storage operations.
//!
//! This module provides efficient binary serialization for `MemoryNode`,
//! `GraphEdge`, `EmbeddingVector`, and `UUID` types. It is optimized for
//! both space efficiency and speed.
//!
//! # Serialization Strategy
//!
//! | Type | Format | Rationale |
//! |------|--------|-----------|
//! | `MemoryNode` | MessagePack | Handles `skip_serializing_if` correctly |
//! | `GraphEdge` | bincode | Optimal for fixed-layout structs |
//! | `EmbeddingVector` | Raw LE f32 | Maximum performance, no overhead |
//! | `UUID` | Raw bytes | Exactly 16 bytes, no overhead |
//!
//! # Performance Characteristics
//!
//! | Operation | Size | Latency |
//! |-----------|------|---------|
//! | MemoryNode (1536D) | ~6.5KB | < 100us |
//! | GraphEdge | ~200 bytes | < 50us |
//! | Embedding (1536D) | 6144 bytes | < 10us |
//! | UUID | 16 bytes | < 1us |
//!
//! # Constitution Compliance
//!
//! - AP-009: All functions preserve exact f32 values (no NaN/Infinity manipulation)
//! - Naming: `snake_case` functions, `PascalCase` types
//!
//! # Example: Round-trip Serialization
//!
//! ```rust
//! use context_graph_storage::serialization::{
//!     serialize_embedding, deserialize_embedding,
//!     serialize_uuid, deserialize_uuid,
//! };
//! use uuid::Uuid;
//!
//! // Embedding round-trip
//! let embedding = vec![0.5_f32; 100];
//! let bytes = serialize_embedding(&embedding);
//! let restored = deserialize_embedding(&bytes).unwrap();
//! assert_eq!(embedding, restored);
//!
//! // UUID round-trip
//! let id = Uuid::new_v4();
//! let bytes = serialize_uuid(&id);
//! let restored = deserialize_uuid(&bytes);
//! assert_eq!(id, restored);
//! ```
//!
//! # Why MessagePack for MemoryNode?
//!
//! `MemoryNode` contains `NodeMetadata` which uses serde attributes like
//! `#[serde(skip_serializing_if = "Option::is_none")]`. Bincode requires
//! fixed-layout serialization and doesn't support these attributes.
//! MessagePack properly handles serde attributes while still providing
//! compact binary output (smaller than JSON, similar to bincode).

mod edge;
mod embedding;
mod error;
mod node;
mod uuid_serde;

// Re-export all public types and functions for backwards compatibility
pub use self::edge::{deserialize_edge, serialize_edge};
pub use self::embedding::{deserialize_embedding, serialize_embedding};
pub use self::error::SerializationError;
pub use self::node::{deserialize_node, serialize_node};
pub use self::uuid_serde::{deserialize_uuid, serialize_uuid};

#[cfg(test)]
mod tests;
