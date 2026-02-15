//! Shared model state type for pretrained embedders.
//!
//! Most pretrained models follow the same Unloaded/Loaded pattern with
//! weights + tokenizer. This generic eliminates the 6+ duplicate enum
//! definitions across the pretrained model modules.

use tokenizers::Tokenizer;

/// Internal state for pretrained model weight management.
///
/// Generic over the weight type `W` so different architectures
/// (BertWeights, QwenWeights, ClipWeights) can share the same enum.
///
/// Models with extra state in their Loaded variant (graph projections,
/// LoRA trained state, MLM heads) should define their own enum.
#[allow(dead_code)]
pub(crate) enum ModelState<W> {
    /// Unloaded - no weights in memory.
    Unloaded,

    /// Loaded with model weights and tokenizer (GPU-accelerated).
    Loaded {
        /// Model weights on GPU (type depends on architecture).
        weights: W,
        /// HuggingFace tokenizer for text encoding (boxed to reduce enum size).
        tokenizer: Box<Tokenizer>,
    },
}
