//! EmbeddingVector serialization using raw little-endian bytes.
//!
//! This uses raw bytes rather than bincode for maximum efficiency.
//! Each f32 becomes exactly 4 bytes in little-endian format.

use context_graph_core::types::EmbeddingVector;

use super::error::SerializationError;

/// Serialize embedding to raw f32 bytes (little-endian).
///
/// This uses raw bytes rather than bincode for maximum efficiency.
/// Each f32 becomes exactly 4 bytes in little-endian format.
///
/// # Arguments
/// * `embedding` - The embedding vector to serialize
///
/// # Returns
/// * `Vec<u8>` - Raw bytes (dim * 4 bytes, e.g., 6144 bytes for 1536D)
///
/// # Infallible
/// This function cannot fail - no Result wrapper needed.
///
/// # Example
/// ```rust
/// use context_graph_storage::serialization::serialize_embedding;
///
/// let embedding = vec![0.5_f32; 1536];
/// let bytes = serialize_embedding(&embedding);
/// assert_eq!(bytes.len(), 1536 * 4); // 6144 bytes
/// ```
pub fn serialize_embedding(embedding: &EmbeddingVector) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for &value in embedding {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Deserialize raw f32 bytes to embedding vector.
///
/// Converts little-endian bytes back to f32 values.
///
/// # Arguments
/// * `bytes` - Raw bytes (must be divisible by 4)
///
/// # Returns
/// * `Ok(EmbeddingVector)` - The deserialized embedding
/// * `Err(SerializationError::InvalidEmbeddingSize)` - If bytes.len() % 4 != 0
///
/// # Example
/// ```rust
/// use context_graph_storage::serialization::{serialize_embedding, deserialize_embedding};
///
/// let embedding = vec![0.25_f32; 100];
/// let bytes = serialize_embedding(&embedding);
/// let restored = deserialize_embedding(&bytes).unwrap();
/// assert_eq!(embedding, restored);
/// ```
pub fn deserialize_embedding(bytes: &[u8]) -> Result<EmbeddingVector, SerializationError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(SerializationError::InvalidEmbeddingSize {
            expected: ((bytes.len() / 4) + 1) * 4,
            actual: bytes.len(),
        });
    }

    let mut embedding = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        // SAFETY: chunks_exact(4) guarantees exactly 4 bytes
        let arr: [u8; 4] = chunk.try_into().expect("chunk is exactly 4 bytes");
        embedding.push(f32::from_le_bytes(arr));
    }
    Ok(embedding)
}
