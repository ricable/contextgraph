//! GraphEdge serialization using bincode.
//!
//! Uses bincode format which is optimal for fixed-layout structs like `GraphEdge`.

use context_graph_core::types::GraphEdge;

use super::error::SerializationError;

/// Serializes a `GraphEdge` to bincode bytes.
///
/// Uses bincode format which is optimal for fixed-layout structs like `GraphEdge`.
///
/// # Arguments
///
/// * `edge` - The `GraphEdge` to serialize
///
/// # Returns
///
/// * `Ok(Vec<u8>)` - Serialized bytes (~200 bytes)
/// * `Err(SerializationError::SerializeFailed)` - If serialization fails
///
/// # Errors
///
/// * `SerializationError::SerializeFailed` - Bincode serialization error
///
/// # Performance
///
/// `Constraint: latency < 50us for typical edge`
///
/// # Example
///
/// ```rust
/// use context_graph_storage::serialization::{serialize_edge, deserialize_edge};
/// use context_graph_storage::{GraphEdge, EdgeType, Domain};
/// use uuid::Uuid;
///
/// // Create edge
/// let edge = GraphEdge::new(
///     Uuid::new_v4(),
///     Uuid::new_v4(),
///     EdgeType::Semantic,
///     Domain::Code,
/// );
///
/// // Serialize
/// let bytes = serialize_edge(&edge).expect("serialize failed");
/// assert!(bytes.len() > 100 && bytes.len() < 500);
///
/// // Round-trip
/// let restored = deserialize_edge(&bytes).unwrap();
/// assert_eq!(edge, restored);
/// ```
pub fn serialize_edge(edge: &GraphEdge) -> Result<Vec<u8>, SerializationError> {
    bincode::serialize(edge).map_err(|e| SerializationError::SerializeFailed(e.to_string()))
}

/// Deserializes bincode bytes to a `GraphEdge`.
///
/// # Arguments
///
/// * `bytes` - The bincode bytes to deserialize
///
/// # Returns
///
/// * `Ok(GraphEdge)` - The deserialized edge
/// * `Err(SerializationError::DeserializeFailed)` - If deserialization fails
///
/// # Errors
///
/// * `SerializationError::DeserializeFailed` - Corrupt data, truncated bytes,
///   or schema mismatch
///
/// # Example
///
/// ```rust
/// use context_graph_storage::serialization::{serialize_edge, deserialize_edge};
/// use context_graph_storage::{GraphEdge, EdgeType, Domain};
/// use uuid::Uuid;
///
/// let edge = GraphEdge::new(
///     Uuid::new_v4(),
///     Uuid::new_v4(),
///     EdgeType::Causal,
///     Domain::Medical,
/// );
///
/// let bytes = serialize_edge(&edge).unwrap();
/// let restored = deserialize_edge(&bytes).unwrap();
/// assert_eq!(edge.edge_type, restored.edge_type);
/// assert_eq!(edge.domain, restored.domain);
/// ```
pub fn deserialize_edge(bytes: &[u8]) -> Result<GraphEdge, SerializationError> {
    bincode::deserialize(bytes).map_err(|e| SerializationError::DeserializeFailed(e.to_string()))
}
