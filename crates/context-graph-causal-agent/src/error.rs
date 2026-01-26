//! Error types for the Causal Discovery Agent.
//!
//! NO BACKWARDS COMPATIBILITY - FAIL FAST WITH ROBUST LOGGING.

use thiserror::Error;
use uuid::Uuid;

/// Result type for causal agent operations.
pub type CausalAgentResult<T> = Result<T, CausalAgentError>;

/// Errors that can occur in the Causal Discovery Agent.
#[derive(Debug, Error)]
pub enum CausalAgentError {
    /// LLM model failed to load.
    #[error("LLM load failed: {message}")]
    LlmLoadError { message: String },

    /// LLM inference failed.
    #[error("LLM inference failed: {message}")]
    LlmInferenceError { message: String },

    /// LLM is not initialized.
    #[error("LLM not initialized - call load() first")]
    LlmNotInitialized,

    /// Failed to parse LLM response as JSON.
    #[error("Failed to parse LLM response: {message}")]
    LlmResponseParseError { message: String },

    /// Generic parse error.
    #[error("Parse error: {message}")]
    ParseError { message: String },

    /// Memory not found in store.
    #[error("Memory not found: {id}")]
    MemoryNotFound { id: Uuid },

    /// Storage operation failed.
    #[error("Storage error: {message}")]
    StorageError { message: String },

    /// Embedding operation failed.
    #[error("Embedding error: {message}")]
    EmbeddingError { message: String },

    /// Configuration error.
    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    /// Model file not found.
    #[error("Model file not found at: {path}")]
    ModelNotFound { path: String },

    /// Insufficient VRAM for model.
    #[error("Insufficient VRAM: need {required_mb}MB, have {available_mb}MB")]
    InsufficientVram {
        required_mb: usize,
        available_mb: usize,
    },

    /// Scanner found no candidate pairs.
    #[error("No candidate pairs found for causal analysis")]
    NoCandidatesFound,

    /// Service is already running.
    #[error("Causal discovery service is already running")]
    ServiceAlreadyRunning,

    /// Internal error.
    #[error("Internal error: {message}")]
    InternalError { message: String },

    /// IO error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

impl CausalAgentError {
    /// Create a storage error from any error type.
    pub fn storage<E: std::fmt::Display>(e: E) -> Self {
        Self::StorageError {
            message: e.to_string(),
        }
    }

    /// Create an embedding error from any error type.
    pub fn embedding<E: std::fmt::Display>(e: E) -> Self {
        Self::EmbeddingError {
            message: e.to_string(),
        }
    }

    /// Create an LLM inference error from any error type.
    pub fn llm_inference<E: std::fmt::Display>(e: E) -> Self {
        Self::LlmInferenceError {
            message: e.to_string(),
        }
    }
}
