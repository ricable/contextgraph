//! Validation error types for MemoryNode.

use thiserror::Error;

use super::DEFAULT_EMBEDDING_DIM;

/// Errors that occur during MemoryNode validation.
///
/// Each variant provides specific context about what validation failed
/// and what values were involved, enabling actionable error messages.
///
/// # Constitution Compliance
/// - AP-009: Prevents NaN/Infinity by validating before storage
/// - Naming: PascalCase enum, snake_case fields
///
/// # Example
/// ```rust
/// use context_graph_core::types::ValidationError;
///
/// let error = ValidationError::InvalidEmbeddingDimension {
///     expected: 1536,
///     actual: 768,
/// };
/// assert!(error.to_string().contains("expected 1536"));
/// ```
#[derive(Debug, Clone, Error, PartialEq)]
pub enum ValidationError {
    /// Embedding vector has incorrect dimensions.
    /// Expected: 1536 (per constitution.yaml embedding spec)
    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    InvalidEmbeddingDimension {
        /// Required dimension (1536)
        expected: usize,
        /// Actual dimension provided
        actual: usize,
    },

    /// A numeric field value is outside its valid range.
    /// Used for importance [0.0, 1.0], valence [-1.0, 1.0], etc.
    #[error("Field '{field}' value {value} is out of bounds [{min}, {max}]")]
    OutOfBounds {
        /// Name of the field that failed validation
        field: String,
        /// The invalid value provided
        value: f64,
        /// Minimum allowed value (inclusive)
        min: f64,
        /// Maximum allowed value (inclusive)
        max: f64,
    },

    /// Content exceeds maximum allowed size.
    /// Limit: 1MB (1,048,576 bytes) per constitution.yaml
    #[error("Content size {size} bytes exceeds maximum allowed {max_size} bytes")]
    ContentTooLarge {
        /// Actual content size in bytes
        size: usize,
        /// Maximum allowed size (1,048,576 bytes)
        max_size: usize,
    },

    /// Embedding vector is not normalized (magnitude should be ~1.0).
    /// Tolerance: magnitude must be in [0.99, 1.01]
    #[error("Embedding not normalized: magnitude is {magnitude:.6}, expected ~1.0")]
    EmbeddingNotNormalized {
        /// Actual magnitude of the embedding vector
        magnitude: f64,
    },
}

impl ValidationError {
    /// Create an InvalidEmbeddingDimension error with the default expected dimension.
    pub fn invalid_dimension(actual: usize) -> Self {
        Self::InvalidEmbeddingDimension {
            expected: DEFAULT_EMBEDDING_DIM,
            actual,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_error_invalid_embedding_dimension() {
        let error = ValidationError::InvalidEmbeddingDimension {
            expected: 1536,
            actual: 768,
        };
        let msg = error.to_string();

        assert!(msg.contains("expected 1536"), "Should show expected dimension");
        assert!(msg.contains("got 768"), "Should show actual dimension");
        assert!(msg.contains("Invalid embedding dimension"), "Should have correct prefix");
    }

    #[test]
    fn test_validation_error_out_of_bounds() {
        let error = ValidationError::OutOfBounds {
            field: "importance".to_string(),
            value: 1.5,
            min: 0.0,
            max: 1.0,
        };
        let msg = error.to_string();

        assert!(msg.contains("importance"), "Should show field name");
        assert!(msg.contains("1.5"), "Should show invalid value");
        assert!(msg.contains("[0, 1]"), "Should show valid range");
    }

    #[test]
    fn test_validation_error_content_too_large() {
        let error = ValidationError::ContentTooLarge {
            size: 2_000_000,
            max_size: 1_048_576,
        };
        let msg = error.to_string();

        assert!(msg.contains("2000000"), "Should show actual size");
        assert!(msg.contains("1048576"), "Should show max size");
        assert!(msg.contains("exceeds maximum"), "Should indicate overflow");
    }

    #[test]
    fn test_validation_error_embedding_not_normalized() {
        let error = ValidationError::EmbeddingNotNormalized { magnitude: 0.85 };
        let msg = error.to_string();

        assert!(msg.contains("0.850000"), "Should show magnitude with precision");
        assert!(msg.contains("not normalized"), "Should indicate normalization issue");
        assert!(msg.contains("expected ~1.0"), "Should show expected value");
    }

    #[test]
    fn test_validation_error_implements_std_error() {
        let error: Box<dyn std::error::Error> =
            Box::new(ValidationError::InvalidEmbeddingDimension {
                expected: 1536,
                actual: 0,
            });
        let _ = error.to_string();
    }

    #[test]
    fn test_validation_error_clone() {
        let original = ValidationError::OutOfBounds {
            field: "test".to_string(),
            value: -0.5,
            min: 0.0,
            max: 1.0,
        };
        let cloned = original.clone();
        assert_eq!(original, cloned, "Clone must produce equal value");
    }

    #[test]
    fn test_validation_error_partial_eq() {
        let a = ValidationError::ContentTooLarge { size: 100, max_size: 50 };
        let b = ValidationError::ContentTooLarge { size: 100, max_size: 50 };
        let c = ValidationError::ContentTooLarge { size: 101, max_size: 50 };

        assert_eq!(a, b, "Same values should be equal");
        assert_ne!(a, c, "Different values should not be equal");
    }

    #[test]
    fn test_validation_error_debug_format() {
        let error = ValidationError::InvalidEmbeddingDimension {
            expected: 1536,
            actual: 512,
        };
        let debug_str = format!("{:?}", error);

        assert!(debug_str.contains("InvalidEmbeddingDimension"), "Debug should show variant");
        assert!(debug_str.contains("1536"), "Debug should show expected");
        assert!(debug_str.contains("512"), "Debug should show actual");
    }

    #[test]
    fn test_validation_error_out_of_bounds_negative_range() {
        let error = ValidationError::OutOfBounds {
            field: "emotional_valence".to_string(),
            value: -1.5,
            min: -1.0,
            max: 1.0,
        };
        let msg = error.to_string();

        assert!(msg.contains("-1.5"), "Should handle negative values");
        assert!(msg.contains("[-1, 1]"), "Should show negative range correctly");
    }

    #[test]
    fn test_validation_error_embedding_edge_magnitudes() {
        let too_small = ValidationError::EmbeddingNotNormalized { magnitude: 0.0 };
        let too_large = ValidationError::EmbeddingNotNormalized { magnitude: 100.0 };

        assert!(too_small.to_string().contains("0.000000"));
        assert!(too_large.to_string().contains("100.000000"));
    }

    #[test]
    fn test_invalid_dimension_helper() {
        let error = ValidationError::invalid_dimension(768);
        assert!(matches!(
            error,
            ValidationError::InvalidEmbeddingDimension { expected: 1536, actual: 768 }
        ));
    }
}
