//! Model state for ContextualModel.
//!
//! Manages the loaded/unloaded state of the contextual embedding model.

use tokenizers::Tokenizer;

use crate::gpu::BertWeights;

use super::projections::ContextProjectionWeights;

/// Internal state that varies based on whether the model is loaded.
#[allow(dead_code)]
pub enum ModelState {
    /// Unloaded - no weights in memory.
    Unloaded,

    /// Loaded with BERT-compatible weights and tokenizer.
    Loaded {
        /// Model weights on GPU.
        weights: Box<BertWeights>,
        /// HuggingFace tokenizer for text encoding (boxed to reduce enum size).
        tokenizer: Box<Tokenizer>,
        /// Context projection weights for asymmetric embeddings.
        projection: ContextProjectionWeights,
    },
}
