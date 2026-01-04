//! UTL error types.
//!
//! This module defines comprehensive error types for all UTL (Unified Trace Likelihood)
//! computations, including surprise, coherence, emotional weighting, phase oscillation,
//! lifecycle transitions, and Johari quadrant classification.

use thiserror::Error;

/// Errors that can occur during UTL computation.
#[derive(Debug, Error)]
pub enum UtlError {
    /// Invalid computation result (NaN or Infinity)
    #[error("Invalid UTL computation: delta_s={delta_s}, delta_c={delta_c}, w_e={w_e}, phi={phi}. {reason}")]
    InvalidComputation {
        /// Surprise component value
        delta_s: f32,
        /// Coherence component value
        delta_c: f32,
        /// Emotional weight value
        w_e: f32,
        /// Phase angle value
        phi: f32,
        /// Reason for invalidity
        reason: String,
    },

    /// Invalid lambda weights (must sum to 1.0)
    #[error("Invalid lambda weights: novelty={novelty}, consolidation={consolidation}. {reason}")]
    InvalidLambdaWeights {
        /// Novelty lambda value
        novelty: f32,
        /// Consolidation lambda value
        consolidation: f32,
        /// Reason for invalidity
        reason: String,
    },

    /// Missing required context for computation
    #[error("Missing context for UTL computation: {0}")]
    MissingContext(String),

    /// Embedding dimension mismatch
    #[error("Embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },

    /// Graph access error
    #[error("Graph access error: {0}")]
    GraphError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Lifecycle transition error
    #[error("Invalid lifecycle transition from {from} to {to}: {reason}")]
    InvalidLifecycleTransition {
        /// Source stage
        from: String,
        /// Target stage
        to: String,
        /// Reason for invalidity
        reason: String,
    },

    /// Johari quadrant classification error
    #[error("Johari classification error: {0}")]
    JohariError(String),

    /// Phase oscillator error
    #[error("Phase oscillator error: {0}")]
    PhaseError(String),

    /// Surprise computation error
    #[error("Surprise computation error: {0}")]
    SurpriseError(String),

    /// Coherence computation error
    #[error("Coherence computation error: {0}")]
    CoherenceError(String),

    /// Empty input provided
    #[error("Empty input provided for UTL computation")]
    EmptyInput,

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Invalid parameter value
    #[error("Invalid parameter '{name}': {value}. {reason}")]
    InvalidParameter {
        /// Parameter name
        name: String,
        /// Parameter value as string
        value: String,
        /// Reason for invalidity
        reason: String,
    },

    /// Numeric overflow or underflow
    #[error("Numeric overflow/underflow in {operation}: {details}")]
    NumericOverflow {
        /// The operation that caused the overflow
        operation: String,
        /// Details about the overflow
        details: String,
    },

    /// Entropy computation error
    #[error("Entropy computation error: {0}")]
    EntropyError(String),

    /// Memory allocation or capacity error
    #[error("Memory/capacity error: {0}")]
    CapacityError(String),
}

/// Result type for UTL operations.
pub type UtlResult<T> = Result<T, UtlError>;

impl From<serde_json::Error> for UtlError {
    fn from(err: serde_json::Error) -> Self {
        UtlError::SerializationError(err.to_string())
    }
}

impl UtlError {
    /// Create an InvalidComputation error for NaN results.
    pub fn nan_result(delta_s: f32, delta_c: f32, w_e: f32, phi: f32) -> Self {
        UtlError::InvalidComputation {
            delta_s,
            delta_c,
            w_e,
            phi,
            reason: "Result is NaN".to_string(),
        }
    }

    /// Create an InvalidComputation error for infinite results.
    pub fn infinite_result(delta_s: f32, delta_c: f32, w_e: f32, phi: f32) -> Self {
        UtlError::InvalidComputation {
            delta_s,
            delta_c,
            w_e,
            phi,
            reason: "Result is infinite".to_string(),
        }
    }

    /// Create an InvalidLambdaWeights error for weights that don't sum to 1.0.
    pub fn lambda_sum_error(novelty: f32, consolidation: f32) -> Self {
        let sum = novelty + consolidation;
        UtlError::InvalidLambdaWeights {
            novelty,
            consolidation,
            reason: format!("Weights must sum to 1.0, got {}", sum),
        }
    }

    /// Create an InvalidLambdaWeights error for negative weights.
    pub fn negative_lambda(novelty: f32, consolidation: f32) -> Self {
        UtlError::InvalidLambdaWeights {
            novelty,
            consolidation,
            reason: "Lambda weights cannot be negative".to_string(),
        }
    }

    /// Create a lifecycle transition error.
    pub fn invalid_transition(
        from: impl Into<String>,
        to: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        UtlError::InvalidLifecycleTransition {
            from: from.into(),
            to: to.into(),
            reason: reason.into(),
        }
    }

    /// Create an invalid parameter error.
    pub fn invalid_param(
        name: impl Into<String>,
        value: impl ToString,
        reason: impl Into<String>,
    ) -> Self {
        UtlError::InvalidParameter {
            name: name.into(),
            value: value.to_string(),
            reason: reason.into(),
        }
    }

    /// Check if this error is recoverable (can be retried with different parameters).
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            UtlError::InvalidLambdaWeights { .. }
                | UtlError::InvalidParameter { .. }
                | UtlError::DimensionMismatch { .. }
                | UtlError::EmptyInput
        )
    }

    /// Check if this error indicates a computation issue.
    pub fn is_computation_error(&self) -> bool {
        matches!(
            self,
            UtlError::InvalidComputation { .. }
                | UtlError::SurpriseError(_)
                | UtlError::CoherenceError(_)
                | UtlError::PhaseError(_)
                | UtlError::EntropyError(_)
                | UtlError::NumericOverflow { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = UtlError::InvalidComputation {
            delta_s: 0.5,
            delta_c: 0.6,
            w_e: 1.0,
            phi: 0.0,
            reason: "test reason".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("delta_s=0.5"));
        assert!(msg.contains("test reason"));
    }

    #[test]
    fn test_dimension_mismatch_display() {
        let err = UtlError::DimensionMismatch {
            expected: 1536,
            actual: 768,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("1536"));
        assert!(msg.contains("768"));
    }

    #[test]
    fn test_empty_input_error() {
        let err = UtlError::EmptyInput;
        assert!(format!("{}", err).contains("Empty input"));
    }

    #[test]
    fn test_nan_result_helper() {
        let err = UtlError::nan_result(0.5, 0.3, 1.0, 0.0);
        let msg = format!("{}", err);
        assert!(msg.contains("NaN"));
        assert!(msg.contains("delta_s=0.5"));
    }

    #[test]
    fn test_infinite_result_helper() {
        let err = UtlError::infinite_result(0.5, 0.3, 1.0, 0.0);
        let msg = format!("{}", err);
        assert!(msg.contains("infinite"));
    }

    #[test]
    fn test_lambda_sum_error_helper() {
        let err = UtlError::lambda_sum_error(0.3, 0.5);
        let msg = format!("{}", err);
        assert!(msg.contains("0.3"));
        assert!(msg.contains("0.5"));
        assert!(msg.contains("sum to 1.0"));
    }

    #[test]
    fn test_negative_lambda_helper() {
        let err = UtlError::negative_lambda(-0.1, 1.1);
        let msg = format!("{}", err);
        assert!(msg.contains("negative"));
    }

    #[test]
    fn test_invalid_transition_helper() {
        let err = UtlError::invalid_transition("Active", "Dormant", "Missing intermediate state");
        let msg = format!("{}", err);
        assert!(msg.contains("Active"));
        assert!(msg.contains("Dormant"));
        assert!(msg.contains("Missing intermediate state"));
    }

    #[test]
    fn test_invalid_param_helper() {
        let err = UtlError::invalid_param("threshold", 1.5, "Must be in range [0, 1]");
        let msg = format!("{}", err);
        assert!(msg.contains("threshold"));
        assert!(msg.contains("1.5"));
        assert!(msg.contains("range [0, 1]"));
    }

    #[test]
    fn test_is_recoverable() {
        assert!(UtlError::EmptyInput.is_recoverable());
        assert!(UtlError::DimensionMismatch {
            expected: 10,
            actual: 5
        }
        .is_recoverable());
        assert!(UtlError::lambda_sum_error(0.3, 0.5).is_recoverable());
        assert!(!UtlError::GraphError("test".to_string()).is_recoverable());
    }

    #[test]
    fn test_is_computation_error() {
        assert!(UtlError::nan_result(0.0, 0.0, 0.0, 0.0).is_computation_error());
        assert!(UtlError::SurpriseError("test".to_string()).is_computation_error());
        assert!(UtlError::CoherenceError("test".to_string()).is_computation_error());
        assert!(UtlError::PhaseError("test".to_string()).is_computation_error());
        assert!(!UtlError::ConfigError("test".to_string()).is_computation_error());
    }

    #[test]
    fn test_lifecycle_transition_error() {
        let err = UtlError::InvalidLifecycleTransition {
            from: "Nascent".to_string(),
            to: "Archived".to_string(),
            reason: "Cannot skip intermediate stages".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Nascent"));
        assert!(msg.contains("Archived"));
        assert!(msg.contains("skip intermediate"));
    }

    #[test]
    fn test_johari_error() {
        let err = UtlError::JohariError("Invalid awareness scores".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Johari"));
        assert!(msg.contains("awareness scores"));
    }

    #[test]
    fn test_numeric_overflow_error() {
        let err = UtlError::NumericOverflow {
            operation: "exponential".to_string(),
            details: "Exponent too large".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("overflow"));
        assert!(msg.contains("exponential"));
    }

    #[test]
    fn test_from_serde_json_error() {
        let json_err = serde_json::from_str::<String>("invalid json").unwrap_err();
        let utl_err: UtlError = json_err.into();
        assert!(matches!(utl_err, UtlError::SerializationError(_)));
    }

    #[test]
    fn test_missing_context_error() {
        let err = UtlError::MissingContext("embedding vector required".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Missing context"));
        assert!(msg.contains("embedding vector"));
    }

    #[test]
    fn test_graph_error() {
        let err = UtlError::GraphError("Node not found".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Graph access error"));
        assert!(msg.contains("Node not found"));
    }

    #[test]
    fn test_config_error() {
        let err = UtlError::ConfigError("Invalid threshold value".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Configuration error"));
        assert!(msg.contains("threshold"));
    }

    #[test]
    fn test_entropy_error() {
        let err = UtlError::EntropyError("Distribution not normalized".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Entropy"));
        assert!(msg.contains("normalized"));
    }

    #[test]
    fn test_capacity_error() {
        let err = UtlError::CapacityError("Buffer overflow".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("capacity"));
        assert!(msg.contains("Buffer overflow"));
    }
}
