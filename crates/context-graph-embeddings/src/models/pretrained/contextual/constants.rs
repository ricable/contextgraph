//! Constants for ContextualModel (all-mpnet-base-v2).
//!
//! # Model Specification
//!
//! - Architecture: MPNet (microsoft/mpnet-base)
//! - Fine-tuned: sentence-transformers/all-mpnet-base-v2
//! - Training: 1.17B sentence pairs from diverse sources
//! - Output: 768D dense embedding optimized for contextual similarity

/// Output dimension for contextual embeddings.
pub const CONTEXTUAL_DIMENSION: usize = 768;

/// Maximum sequence length (from config.json max_position_embeddings).
pub const CONTEXTUAL_MAX_TOKENS: usize = 384;

/// Model name for logging and identification.
pub const CONTEXTUAL_MODEL_NAME: &str = "sentence-transformers/all-mpnet-base-v2";

/// Latency budget for single embedding (milliseconds).
pub const CONTEXTUAL_LATENCY_BUDGET_MS: u64 = 15;

/// Vocabulary size for MPNet tokenizer.
pub const CONTEXTUAL_VOCAB_SIZE: usize = 30527;

/// Number of hidden layers in MPNet encoder.
pub const CONTEXTUAL_NUM_LAYERS: usize = 12;

/// Number of attention heads per layer.
pub const CONTEXTUAL_NUM_HEADS: usize = 12;

/// Hidden size (matches CONTEXTUAL_DIMENSION for MPNet).
pub const CONTEXTUAL_HIDDEN_SIZE: usize = 768;

/// Intermediate FFN size.
pub const CONTEXTUAL_INTERMEDIATE_SIZE: usize = 3072;

/// Layer norm epsilon.
pub const CONTEXTUAL_LAYER_NORM_EPS: f64 = 1e-5;
