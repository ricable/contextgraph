//! Error types for the graph relationship discovery agent.

use thiserror::Error;

/// Result type for graph agent operations.
pub type GraphAgentResult<T> = Result<T, GraphAgentError>;

/// Errors that can occur during graph relationship discovery.
#[derive(Debug, Error)]
pub enum GraphAgentError {
    /// LLM model failed to load.
    #[error("LLM load failed: {message}")]
    LlmLoadError { message: String },

    /// LLM inference failed.
    #[error("LLM inference failed: {message}")]
    LlmInferenceError { message: String },

    /// LLM not initialized - call load() first.
    #[error("LLM not initialized - call load() first")]
    LlmNotInitialized,

    /// Failed to parse LLM response.
    #[error("Failed to parse LLM response: {message}")]
    LlmResponseParseError { message: String },

    /// Storage operation failed.
    #[error("Storage error: {message}")]
    StorageError { message: String },

    /// Embedding operation failed.
    #[error("Embedding error: {message}")]
    EmbeddingError { message: String },

    /// Configuration error.
    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    /// Memory not found.
    #[error("Memory not found: {id}")]
    MemoryNotFound { id: uuid::Uuid },

    /// Service already running.
    #[error("Service already running")]
    ServiceAlreadyRunning,

    /// Service not running.
    #[error("Service not running")]
    ServiceNotRunning,

    /// Internal error.
    #[error("Internal error: {message}")]
    InternalError { message: String },
}

impl From<context_graph_causal_agent::CausalAgentError> for GraphAgentError {
    fn from(err: context_graph_causal_agent::CausalAgentError) -> Self {
        GraphAgentError::InternalError {
            message: format!("Causal agent error: {}", err),
        }
    }
}
