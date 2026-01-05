//! MemoryNode serialization using MessagePack.
//!
//! Uses MessagePack (rmp-serde) format which properly handles serde's
//! `skip_serializing_if` attributes on `NodeMetadata`.

use context_graph_core::types::MemoryNode;

use super::error::SerializationError;

/// Serializes a `MemoryNode` to MessagePack bytes.
///
/// Uses MessagePack (rmp-serde) format which properly handles serde's
/// `skip_serializing_if` attributes on `NodeMetadata`.
///
/// # Arguments
///
/// * `node` - The `MemoryNode` to serialize
///
/// # Returns
///
/// * `Ok(Vec<u8>)` - Serialized bytes (~6.5KB for node with 1536D embedding)
/// * `Err(SerializationError::SerializeFailed)` - If serialization fails
///
/// # Errors
///
/// * `SerializationError::SerializeFailed` - MessagePack serialization error
///
/// # Performance
///
/// `Constraint: latency < 100us for typical node`
///
/// # Example
///
/// ```rust
/// use context_graph_storage::serialization::{serialize_node, deserialize_node};
/// use context_graph_storage::MemoryNode;
///
/// // Create a valid normalized embedding
/// let dim = 1536;
/// let val = 1.0_f32 / (dim as f32).sqrt();
/// let embedding = vec![val; dim];
///
/// // Create and serialize node
/// let node = MemoryNode::new("Test content".to_string(), embedding);
/// let bytes = serialize_node(&node).expect("serialize failed");
///
/// // Serialized size should be ~6.5KB
/// assert!(bytes.len() > 6000);
/// assert!(bytes.len() < 10000);
///
/// // Round-trip verification
/// let restored = deserialize_node(&bytes).unwrap();
/// assert_eq!(node.id, restored.id);
/// assert_eq!(node.content, restored.content);
/// ```
pub fn serialize_node(node: &MemoryNode) -> Result<Vec<u8>, SerializationError> {
    // Use named format for proper field handling with skip_serializing_if attrs
    rmp_serde::to_vec_named(node).map_err(|e| SerializationError::SerializeFailed(e.to_string()))
}

/// Deserializes MessagePack bytes to a `MemoryNode`.
///
/// # Arguments
///
/// * `bytes` - The MessagePack bytes to deserialize
///
/// # Returns
///
/// * `Ok(MemoryNode)` - The deserialized node
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
/// use context_graph_storage::serialization::{serialize_node, deserialize_node};
/// use context_graph_storage::MemoryNode;
///
/// // Create a node with normalized embedding
/// let dim = 1536;
/// let val = 1.0_f32 / (dim as f32).sqrt();
/// let node = MemoryNode::new("Test".to_string(), vec![val; dim]);
///
/// // Round-trip
/// let bytes = serialize_node(&node).unwrap();
/// let restored = deserialize_node(&bytes).unwrap();
/// assert_eq!(node, restored);
/// ```
pub fn deserialize_node(bytes: &[u8]) -> Result<MemoryNode, SerializationError> {
    rmp_serde::from_slice(bytes).map_err(|e| SerializationError::DeserializeFailed(e.to_string()))
}
