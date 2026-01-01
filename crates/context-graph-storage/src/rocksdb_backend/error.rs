//! Storage error types for RocksDB backend.
//!
//! These errors cover database lifecycle operations and CRUD operations.
//! Designed for fail-fast debugging with descriptive error messages.

use crate::serialization::SerializationError;
use context_graph_core::types::ValidationError;
use thiserror::Error;

/// Storage operation errors.
///
/// These errors cover database lifecycle operations and CRUD operations.
/// Designed for fail-fast debugging with descriptive error messages.
///
/// # TASK-M02-017 Additions
/// - `NotFound`: Node/entity not found by ID
/// - `Serialization`: Serialization/deserialization errors
/// - `ValidationFailed`: Node validation failed before storage
#[derive(Debug, Error)]
pub enum StorageError {
    /// Database failed to open.
    #[error("Failed to open database at '{path}': {message}")]
    OpenFailed { path: String, message: String },

    /// Column family not found (should never happen if DB opened correctly).
    #[error("Column family '{name}' not found")]
    ColumnFamilyNotFound { name: String },

    /// Write operation failed.
    #[error("Write failed: {0}")]
    WriteFailed(String),

    /// Read operation failed.
    #[error("Read failed: {0}")]
    ReadFailed(String),

    /// Flush operation failed.
    #[error("Flush failed: {0}")]
    FlushFailed(String),

    /// Node not found by ID.
    ///
    /// Returned by `get_node()`, `update_node()`, and `delete_node()` when
    /// the requested node does not exist in the database.
    #[error("Node not found: {id}")]
    NotFound {
        /// The node ID that was not found (as string for display)
        id: String,
    },

    /// Serialization or deserialization error.
    ///
    /// Wraps errors from the serialization module during storage operations.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Node validation failed.
    ///
    /// Returned when `MemoryNode::validate()` fails before storage.
    /// Fail fast: invalid nodes are never stored.
    #[error("Validation error: {0}")]
    ValidationFailed(String),

    /// Index corruption detected during scan or validation.
    ///
    /// Indicates data integrity issues in secondary indexes.
    /// Should trigger investigation and potential index rebuild.
    #[error("Index corruption detected in {index_name}: {details}")]
    IndexCorrupted {
        /// Name of the corrupted index (e.g., "johari_open", "temporal")
        index_name: String,
        /// Details about the corruption (e.g., "UUID parse failed")
        details: String,
    },

    /// Generic internal error for unexpected failures.
    ///
    /// Used for internal errors that don't fit other categories.
    /// Should include diagnostic information for debugging.
    #[error("Internal storage error: {0}")]
    Internal(String),
}

impl From<SerializationError> for StorageError {
    fn from(e: SerializationError) -> Self {
        StorageError::Serialization(e.to_string())
    }
}

impl From<ValidationError> for StorageError {
    fn from(e: ValidationError) -> Self {
        StorageError::ValidationFailed(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_open_failed() {
        let error = StorageError::OpenFailed {
            path: "/tmp/test".to_string(),
            message: "permission denied".to_string(),
        };
        let msg = error.to_string();
        assert!(msg.contains("/tmp/test"));
        assert!(msg.contains("permission denied"));
    }

    #[test]
    fn test_error_column_family_not_found() {
        let error = StorageError::ColumnFamilyNotFound {
            name: "unknown_cf".to_string(),
        };
        let msg = error.to_string();
        assert!(msg.contains("unknown_cf"));
    }

    #[test]
    fn test_error_write_failed() {
        let error = StorageError::WriteFailed("disk full".to_string());
        assert!(error.to_string().contains("disk full"));
    }

    #[test]
    fn test_error_read_failed() {
        let error = StorageError::ReadFailed("io error".to_string());
        assert!(error.to_string().contains("io error"));
    }

    #[test]
    fn test_error_flush_failed() {
        let error = StorageError::FlushFailed("sync failed".to_string());
        assert!(error.to_string().contains("sync failed"));
    }

    #[test]
    fn test_error_debug() {
        let error = StorageError::WriteFailed("test".to_string());
        let debug = format!("{:?}", error);
        assert!(debug.contains("WriteFailed"));
    }

    #[test]
    fn test_error_not_found() {
        let error = StorageError::NotFound {
            id: "550e8400-e29b-41d4-a716-446655440000".to_string(),
        };
        let msg = error.to_string();
        assert!(msg.contains("Node not found"));
        assert!(msg.contains("550e8400"));
    }

    #[test]
    fn test_error_serialization() {
        let error = StorageError::Serialization("invalid msgpack".to_string());
        let msg = error.to_string();
        assert!(msg.contains("Serialization error"));
        assert!(msg.contains("invalid msgpack"));
    }

    #[test]
    fn test_error_validation_failed() {
        let error = StorageError::ValidationFailed("importance out of range".to_string());
        let msg = error.to_string();
        assert!(msg.contains("Validation error"));
        assert!(msg.contains("importance out of range"));
    }

    #[test]
    fn test_from_serialization_error() {
        let ser_error = SerializationError::SerializeFailed("test".to_string());
        let storage_error: StorageError = ser_error.into();
        assert!(matches!(storage_error, StorageError::Serialization(_)));
    }

    #[test]
    fn test_from_validation_error() {
        let val_error = ValidationError::InvalidEmbeddingDimension {
            expected: 1536,
            actual: 100,
        };
        let storage_error: StorageError = val_error.into();
        assert!(matches!(storage_error, StorageError::ValidationFailed(_)));
    }
}
