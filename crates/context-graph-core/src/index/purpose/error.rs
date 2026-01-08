//! Purpose index error types with fail-fast semantics.
//!
//! # CRITICAL: NO FALLBACKS
//!
//! All errors are fatal and must be handled explicitly.
//! There are no recovery mechanisms - errors propagate immediately.

use thiserror::Error;
use uuid::Uuid;

use super::super::error::IndexError;

/// Purpose index operation errors with detailed context for debugging.
///
/// # Error Handling Philosophy
///
/// - **NotFound**: Memory does not exist in the purpose index
/// - **InvalidConfidence**: Confidence values must be in [0.0, 1.0]
/// - **InvalidQuery**: Query parameters are malformed or invalid
/// - **DimensionMismatch**: Vector dimensions don't match PURPOSE_VECTOR_DIM (13)
/// - **ClusteringError**: Clustering algorithm failure
/// - **HnswError**: Wraps underlying HNSW index errors
/// - **PersistenceError**: Storage operations failed
///
/// # NO FALLBACKS
///
/// All errors must propagate. Do not catch and ignore.
#[derive(Error, Debug)]
pub enum PurposeIndexError {
    /// Memory not found in purpose index.
    ///
    /// The requested memory ID does not exist in the purpose index.
    /// This is NOT recoverable - the caller must handle missing data.
    #[error("PURPOSE INDEX ERROR: Memory {memory_id} not found in purpose index")]
    NotFound {
        /// The missing memory ID
        memory_id: Uuid,
    },

    /// Confidence value out of valid range [0.0, 1.0].
    ///
    /// Confidence values must be normalized floats.
    /// Values outside this range indicate upstream computation errors.
    #[error("PURPOSE INDEX ERROR: Invalid confidence value {value} - {context}")]
    InvalidConfidence {
        /// The invalid confidence value
        value: f32,
        /// Context describing where the invalid value was encountered
        context: String,
    },

    /// Query parameters are invalid or malformed.
    ///
    /// The query could not be executed due to invalid parameters.
    /// Fix the query construction before retrying.
    #[error("PURPOSE INDEX ERROR: Invalid query - {reason}")]
    InvalidQuery {
        /// Description of what makes the query invalid
        reason: String,
    },

    /// Vector dimension does not match PURPOSE_VECTOR_DIM (13).
    ///
    /// All purpose vectors must have exactly 13 dimensions.
    /// This error indicates a bug in vector construction.
    #[error("PURPOSE INDEX ERROR: Dimension mismatch - expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension (should be 13)
        expected: usize,
        /// Actual dimension received
        actual: usize,
    },

    /// Clustering algorithm failure.
    ///
    /// The clustering operation could not complete.
    /// This may indicate insufficient data or algorithm misconfiguration.
    #[error("PURPOSE INDEX ERROR: Clustering failed - {reason}")]
    ClusteringError {
        /// Description of the clustering failure
        reason: String,
    },

    /// Wraps underlying HNSW index errors.
    ///
    /// Propagates errors from the RealHnswIndex operations.
    /// The source error contains detailed context.
    #[error("PURPOSE INDEX ERROR: HNSW operation failed - {0}")]
    HnswError(#[from] IndexError),

    /// Storage/persistence operation failed.
    ///
    /// Could not read from or write to persistent storage.
    /// The context describes the operation, message contains details.
    #[error("PURPOSE INDEX ERROR: Persistence failed - {context}: {message}")]
    PersistenceError {
        /// What operation was attempted
        context: String,
        /// Underlying error message
        message: String,
    },
}

impl PurposeIndexError {
    /// Create a NotFound error for a memory ID.
    #[inline]
    pub fn not_found(memory_id: Uuid) -> Self {
        Self::NotFound { memory_id }
    }

    /// Create an InvalidConfidence error with context.
    #[inline]
    pub fn invalid_confidence(value: f32, context: impl Into<String>) -> Self {
        Self::InvalidConfidence {
            value,
            context: context.into(),
        }
    }

    /// Create an InvalidQuery error.
    #[inline]
    pub fn invalid_query(reason: impl Into<String>) -> Self {
        Self::InvalidQuery {
            reason: reason.into(),
        }
    }

    /// Create a DimensionMismatch error.
    #[inline]
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    /// Create a ClusteringError.
    #[inline]
    pub fn clustering(reason: impl Into<String>) -> Self {
        Self::ClusteringError {
            reason: reason.into(),
        }
    }

    /// Create a PersistenceError with context.
    #[inline]
    pub fn persistence(context: impl Into<String>, message: impl Into<String>) -> Self {
        Self::PersistenceError {
            context: context.into(),
            message: message.into(),
        }
    }
}

/// Result type for purpose index operations.
pub type PurposeIndexResult<T> = Result<T, PurposeIndexError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not_found_error_message() {
        let id = Uuid::new_v4();
        let err = PurposeIndexError::not_found(id);
        let msg = err.to_string();
        assert!(msg.contains(&id.to_string()));
        assert!(msg.contains("not found"));
        println!("[VERIFIED] NotFound error format: {}", msg);
    }

    #[test]
    fn test_invalid_confidence_error_message() {
        let err = PurposeIndexError::invalid_confidence(1.5, "goal alignment score");
        let msg = err.to_string();
        assert!(msg.contains("1.5"));
        assert!(msg.contains("goal alignment score"));
        println!("[VERIFIED] InvalidConfidence error format: {}", msg);
    }

    #[test]
    fn test_invalid_query_error_message() {
        let err = PurposeIndexError::invalid_query("k must be positive");
        let msg = err.to_string();
        assert!(msg.contains("k must be positive"));
        println!("[VERIFIED] InvalidQuery error format: {}", msg);
    }

    #[test]
    fn test_dimension_mismatch_error_message() {
        let err = PurposeIndexError::dimension_mismatch(13, 10);
        let msg = err.to_string();
        assert!(msg.contains("13"));
        assert!(msg.contains("10"));
        println!("[VERIFIED] DimensionMismatch error format: {}", msg);
    }

    #[test]
    fn test_clustering_error_message() {
        let err = PurposeIndexError::clustering("insufficient data points");
        let msg = err.to_string();
        assert!(msg.contains("insufficient data points"));
        println!("[VERIFIED] ClusteringError error format: {}", msg);
    }

    #[test]
    fn test_hnsw_error_wrapping() {
        let index_err = IndexError::NotFound {
            memory_id: Uuid::new_v4(),
        };
        let purpose_err: PurposeIndexError = index_err.into();
        let msg = purpose_err.to_string();
        assert!(msg.contains("HNSW operation failed"));
        println!("[VERIFIED] HnswError wrapping: {}", msg);
    }

    #[test]
    fn test_persistence_error_message() {
        let err = PurposeIndexError::persistence("saving index", "disk full");
        let msg = err.to_string();
        assert!(msg.contains("saving index"));
        assert!(msg.contains("disk full"));
        println!("[VERIFIED] PersistenceError error format: {}", msg);
    }

    #[test]
    fn test_purpose_index_result_type() {
        fn returns_result() -> PurposeIndexResult<u32> {
            Ok(42)
        }
        fn returns_error() -> PurposeIndexResult<u32> {
            Err(PurposeIndexError::invalid_query("test"))
        }
        assert!(returns_result().is_ok());
        assert!(returns_error().is_err());
        println!("[VERIFIED] PurposeIndexResult type alias works correctly");
    }
}
