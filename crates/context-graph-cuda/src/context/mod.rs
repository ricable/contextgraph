//! Green Contexts GPU partitioning module.
//!
//! Provides 70% inference / 30% background SM partitioning for RTX 5090+.
//! Gracefully degrades on older GPUs without partitioning support.

pub mod green_contexts;

pub use green_contexts::{
    should_enable_green_contexts,
    should_enable_green_contexts_with_config,
    GreenContexts,
    GreenContextsConfig,
    GreenContext,
    GREEN_CONTEXTS_MIN_COMPUTE_MAJOR,
    GREEN_CONTEXTS_MIN_COMPUTE_MINOR,
    INFERENCE_PARTITION_PERCENT,
    BACKGROUND_PARTITION_PERCENT,
    MIN_SMS_FOR_PARTITIONING,
};
