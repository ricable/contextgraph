//! UUID serialization to raw bytes.
//!
//! Provides zero-overhead serialization of UUIDs to exactly 16 bytes.

use uuid::Uuid;

/// Serialize UUID to exactly 16 bytes.
///
/// # Arguments
/// * `id` - The UUID to serialize
///
/// # Returns
/// * `[u8; 16]` - The UUID bytes
///
/// # Infallible
/// This function cannot fail.
///
/// # Example
/// ```rust
/// use context_graph_storage::serialization::serialize_uuid;
/// use uuid::Uuid;
///
/// let id = Uuid::new_v4();
/// let bytes = serialize_uuid(&id);
/// assert_eq!(bytes.len(), 16);
/// ```
#[inline]
pub fn serialize_uuid(id: &Uuid) -> [u8; 16] {
    *id.as_bytes()
}

/// Deserialize 16 bytes to UUID.
///
/// # Arguments
/// * `bytes` - Exactly 16 bytes
///
/// # Returns
/// * `Uuid` - The deserialized UUID
///
/// # Infallible
/// This function cannot fail when given a valid [u8; 16] array.
///
/// # Example
/// ```rust
/// use context_graph_storage::serialization::{serialize_uuid, deserialize_uuid};
/// use uuid::Uuid;
///
/// let id = Uuid::new_v4();
/// let bytes = serialize_uuid(&id);
/// let restored = deserialize_uuid(&bytes);
/// assert_eq!(id, restored);
/// ```
#[inline]
pub fn deserialize_uuid(bytes: &[u8; 16]) -> Uuid {
    Uuid::from_bytes(*bytes)
}
