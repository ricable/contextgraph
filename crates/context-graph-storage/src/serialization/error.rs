//! Serialization error types.
//!
//! This module defines the error types used throughout the serialization module.

use thiserror::Error;

/// Errors that can occur during serialization/deserialization operations.
///
/// These errors indicate data format issues or corruption. They are converted
/// to [`StorageError::Serialization`](crate::StorageError::Serialization)
/// when propagated from storage operations.
///
/// # Design Notes
///
/// - `bincode::Error` does not implement `Clone`, so we store error messages as `String`
/// - All variants include enough context for debugging
/// - Implements `Clone` and `PartialEq` for testing
///
/// # Example: Error Construction
///
/// ```rust
/// use context_graph_storage::SerializationError;
///
/// // Invalid embedding size
/// let error = SerializationError::InvalidEmbeddingSize {
///     expected: 6144,
///     actual: 100,
/// };
/// assert!(error.to_string().contains("expected 6144"));
///
/// // Serialization failure
/// let error = SerializationError::SerializeFailed("corrupt data".to_string());
/// assert!(error.to_string().contains("Serialization failed"));
/// ```
///
/// # Example: Error Matching
///
/// ```rust
/// use context_graph_storage::serialization::{SerializationError, deserialize_embedding};
///
/// let bad_bytes = vec![0u8; 13]; // Not divisible by 4
/// match deserialize_embedding(&bad_bytes) {
///     Ok(_) => unreachable!(),
///     Err(SerializationError::InvalidEmbeddingSize { expected, actual }) => {
///         println!("Expected {} bytes, got {}", expected, actual);
///     }
///     Err(_) => unreachable!(),
/// }
/// ```
#[derive(Debug, Error, Clone, PartialEq)]
pub enum SerializationError {
    /// Serialization operation failed.
    ///
    /// Contains the underlying error message from bincode or MessagePack.
    /// Common causes:
    /// - Data structure incompatible with serialization format
    /// - Out of memory during serialization
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::SerializationError;
    ///
    /// let error = SerializationError::SerializeFailed("unexpected type".to_string());
    /// assert!(error.to_string().contains("Serialization failed"));
    /// ```
    #[error("Serialization failed: {0}")]
    SerializeFailed(String),

    /// Deserialization operation failed.
    ///
    /// Contains the underlying error message from bincode or MessagePack.
    /// Common causes:
    /// - Corrupted data in database
    /// - Schema version mismatch
    /// - Truncated data (incomplete write)
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::SerializationError;
    ///
    /// let error = SerializationError::DeserializeFailed("invalid format".to_string());
    /// assert!(error.to_string().contains("Deserialization failed"));
    /// ```
    #[error("Deserialization failed: {0}")]
    DeserializeFailed(String),

    /// Embedding byte array has invalid size.
    ///
    /// Embedding bytes must be divisible by 4 (size of f32).
    /// This error occurs when deserializing raw embedding bytes.
    ///
    /// # Fields
    ///
    /// - `expected`: The next valid byte count (rounded up to multiple of 4)
    /// - `actual`: The actual byte count received
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::SerializationError;
    ///
    /// let error = SerializationError::InvalidEmbeddingSize {
    ///     expected: 16, // Next multiple of 4
    ///     actual: 13,
    /// };
    /// assert!(error.to_string().contains("expected 16"));
    /// ```
    #[error("Invalid embedding size: expected {expected} bytes, got {actual}")]
    InvalidEmbeddingSize {
        /// Expected byte count (must be divisible by 4)
        expected: usize,
        /// Actual byte count received
        actual: usize,
    },

    /// UUID byte array has invalid size.
    ///
    /// UUID requires exactly 16 bytes. This error is included for
    /// completeness but is unlikely in practice since UUIDs use
    /// fixed-size arrays.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::SerializationError;
    ///
    /// let error = SerializationError::InvalidUuidSize { actual: 10 };
    /// assert!(error.to_string().contains("expected 16 bytes"));
    /// ```
    #[error("Invalid UUID bytes: expected 16 bytes, got {actual}")]
    InvalidUuidSize {
        /// Actual byte count received
        actual: usize,
    },
}
