//! Contextual embedding model using sentence-transformers/all-mpnet-base-v2.
//!
//! This model (part of E10) produces 768D vectors optimized for contextual
//! similarity and relationship understanding. It supports asymmetric
//! intent/context embeddings via `embed_dual()`.
//!
//! # GPU Acceleration
//!
//! When the `candle` feature is enabled, this model uses GPU-accelerated
//! MPNet inference via Candle with the following pipeline:
//! 1. Tokenization with HuggingFace tokenizers
//! 2. GPU embedding lookup and position encoding
//! 3. GPU-accelerated transformer forward pass (12 layers)
//! 4. Mean pooling over sequence dimension
//! 5. L2 normalization on GPU
//! 6. Optional: Dual projection for asymmetric embeddings
//!
//! # Asymmetric Embeddings
//!
//! The `embed_dual()` method produces two distinct vectors:
//! - **Intent**: What the text is trying to accomplish
//! - **Context**: What contextual relationships the text establishes
//!
//! # Thread Safety
//! - `AtomicBool` for `loaded` state (lock-free reads)
//! - `RwLock` for model state (thread-safe state transitions)
//!
//! # Memory Layout
//! - Total estimated: ~440MB for FP32 weights (110M parameters)
//! - With FP16 quantization: ~220MB

mod constants;
mod marker_detection;
mod model;
mod projections;
mod state;

// Re-export public API
pub use constants::{
    CONTEXTUAL_DIMENSION, CONTEXTUAL_HIDDEN_SIZE, CONTEXTUAL_INTERMEDIATE_SIZE,
    CONTEXTUAL_LATENCY_BUDGET_MS, CONTEXTUAL_LAYER_NORM_EPS, CONTEXTUAL_MAX_TOKENS,
    CONTEXTUAL_MODEL_NAME, CONTEXTUAL_NUM_HEADS, CONTEXTUAL_NUM_LAYERS, CONTEXTUAL_VOCAB_SIZE,
};
pub use marker_detection::{
    context_pooling_weights, detect_context_markers, intent_pooling_weights, ContextMarkerResult,
    ContextType,
};
pub use model::ContextualModel;
pub use projections::{ContextProjectionWeights, CONTEXT_PROJECTION_SEED};
