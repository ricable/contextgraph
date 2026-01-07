//! Error types for context-graph-core.
//!
//! This module defines the central error type [`CoreError`] used throughout
//! the context-graph-core crate, along with the [`CoreResult<T>`] type alias.
//!
//! # Examples
//!
//! ```rust
//! use context_graph_core::CoreError;
//! use uuid::Uuid;
//!
//! fn lookup_node(id: Uuid) -> Result<(), CoreError> {
//!     Err(CoreError::NodeNotFound { id })
//! }
//!
//! let result = lookup_node(Uuid::nil());
//! assert!(result.is_err());
//! ```

use thiserror::Error;
use uuid::Uuid;

/// Top-level error type for context-graph-core operations.
///
/// Provides structured error variants for all failure modes in the core library,
/// enabling precise error handling and informative error messages.
///
/// # Examples
///
/// ```rust
/// use context_graph_core::CoreError;
/// use uuid::Uuid;
///
/// // Pattern matching on error variants
/// let error = CoreError::DimensionMismatch {
///     expected: 1536,
///     actual: 768,
/// };
///
/// match &error {
///     CoreError::DimensionMismatch { expected, actual } => {
///         assert_eq!(*expected, 1536);
///         assert_eq!(*actual, 768);
///     }
///     _ => panic!("unexpected variant"),
/// }
///
/// // Error display
/// assert!(error.to_string().contains("1536"));
/// ```
#[derive(Debug, Error)]
pub enum CoreError {
    /// A requested node was not found in the graph.
    ///
    /// # When This Occurs
    ///
    /// - Looking up a node by ID that does not exist
    /// - Referencing a deleted node
    /// - Using a stale node reference after graph modification
    #[error("Node not found: {id}")]
    NodeNotFound {
        /// The UUID of the node that was not found
        id: Uuid,
    },

    /// Embedding vector dimension does not match expected size.
    ///
    /// # When This Occurs
    ///
    /// - Providing embedding with wrong dimension (expected: 1536)
    /// - Mixing embeddings from different models
    /// - Corrupted embedding data during deserialization
    ///
    /// `Constraint: embedding.len() == 1536`
    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected embedding dimension (typically 1536)
        expected: usize,
        /// Actual embedding dimension provided
        actual: usize,
    },

    /// A field value failed validation constraints.
    ///
    /// # When This Occurs
    ///
    /// - Field value out of allowed range (e.g., importance > 1.0)
    /// - Invalid format for string fields
    /// - NaN or Infinity in numeric fields (AP-009 violation)
    #[error("Validation error: {field} - {message}")]
    ValidationError {
        /// Name of the field that failed validation
        field: String,
        /// Description of the validation failure
        message: String,
    },

    /// An error occurred during storage operations.
    ///
    /// # When This Occurs
    ///
    /// - Database connection failure
    /// - Write operation failure
    /// - Transaction rollback
    /// - Disk space exhaustion
    #[error("Storage error: {0}")]
    StorageError(String),

    /// An error occurred with index operations.
    ///
    /// # When This Occurs
    ///
    /// - HNSW index build failure
    /// - Secondary index corruption
    /// - Index query timeout
    /// - Out of memory during indexing
    #[error("Index error: {0}")]
    IndexError(String),

    /// Configuration is invalid or missing.
    ///
    /// # When This Occurs
    ///
    /// - Missing required configuration file
    /// - Invalid configuration value format
    /// - Conflicting configuration options
    /// - Environment variable parsing failure
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Error during UTL (Unified Theory of Learning) computation.
    ///
    /// # When This Occurs
    ///
    /// - Forgetting curve calculation failure
    /// - Emotional processing error
    /// - Importance scoring failure
    /// - Consolidation algorithm error
    #[error("UTL computation error: {0}")]
    UtlError(String),

    /// Error during nervous system layer processing.
    ///
    /// # When This Occurs
    ///
    /// - Routing layer failure
    /// - Processing timeout in specific layer
    /// - Inter-layer communication error
    /// - Layer initialization failure
    #[error("Layer processing error in {layer}: {message}")]
    LayerError {
        /// Name of the layer where error occurred
        layer: String,
        /// Description of the layer error
        message: String,
    },

    /// Requested feature is disabled in current configuration.
    ///
    /// # When This Occurs
    ///
    /// - Calling dream consolidation when disabled
    /// - Using optional features not enabled at compile time
    /// - Accessing premium features without license
    #[error("Feature disabled: {feature}")]
    FeatureDisabled {
        /// Name of the disabled feature
        feature: String,
    },

    /// Error during serialization or deserialization.
    ///
    /// # When This Occurs
    ///
    /// - JSON parsing failure
    /// - Invalid UTF-8 in content
    /// - Schema version mismatch
    /// - Corrupted stored data
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// An unexpected internal error occurred.
    ///
    /// # When This Occurs
    ///
    /// - Invariant violation detected
    /// - Unrecoverable state corruption
    /// - Bug in core logic
    /// - Resource exhaustion
    ///
    /// These errors typically indicate bugs and should be reported.
    #[error("Internal error: {0}")]
    Internal(String),

    /// Embedding generation failed.
    ///
    /// # When This Occurs
    ///
    /// - Empty content provided for embedding
    /// - Model not initialized or ready
    /// - GPU memory exhaustion during embedding
    /// - Embedding computation timeout
    ///
    /// `Constraint: single_embed <10ms, batch_embed_64 <50ms`
    #[error("Embedding error: {0}")]
    Embedding(String),

    /// A required field is missing from a data structure.
    ///
    /// # When This Occurs
    ///
    /// - Required field is None when computation requires it
    /// - Data structure constructed without mandatory fields
    /// - Deserialized data missing required properties
    ///
    /// # Constitution Compliance
    ///
    /// This error supports FAIL-FAST behavior. Rather than using
    /// fallback values (e.g., zero vectors) that mask bugs, missing
    /// required data immediately produces this error. This prevents
    /// silent corruption of computations like alignment scores.
    ///
    /// Per AP-007: No stubs/fallbacks in production code paths.
    #[error("Missing required field '{field}': {context}")]
    MissingField {
        /// Name of the missing field
        field: String,
        /// Explanation of why the field is required and context
        context: String,
    },
}

impl From<serde_json::Error> for CoreError {
    fn from(err: serde_json::Error) -> Self {
        CoreError::SerializationError(err.to_string())
    }
}

impl From<config::ConfigError> for CoreError {
    fn from(err: config::ConfigError) -> Self {
        CoreError::ConfigError(err.to_string())
    }
}

/// Result type alias for core operations.
pub type CoreResult<T> = Result<T, CoreError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CoreError::NodeNotFound { id: Uuid::nil() };
        assert!(err.to_string().contains("Node not found"));
    }

    #[test]
    fn test_dimension_mismatch() {
        let err = CoreError::DimensionMismatch {
            expected: 1536,
            actual: 768,
        };
        assert!(err.to_string().contains("1536"));
        assert!(err.to_string().contains("768"));
    }
}
