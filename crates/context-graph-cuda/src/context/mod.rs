//! Green Contexts GPU partitioning module.
//!
//! Provides 70% inference / 30% background SM partitioning for RTX 5090+.
//! Gracefully degrades on older GPUs without partitioning support.

#[cfg(feature = "cuda")]
pub mod green_contexts;

#[cfg(feature = "cuda")]
pub use green_contexts::{
    should_enable_green_contexts, should_enable_green_contexts_with_config, GreenContext,
    GreenContexts, GreenContextsConfig, BACKGROUND_PARTITION_PERCENT,
    GREEN_CONTEXTS_MIN_COMPUTE_MAJOR, GREEN_CONTEXTS_MIN_COMPUTE_MINOR,
    INFERENCE_PARTITION_PERCENT, MIN_SMS_FOR_PARTITIONING,
};

// Stub types for non-cuda builds
#[cfg(not(feature = "cuda"))]
pub mod green_contexts_stub {
    pub const BACKGROUND_PARTITION_PERCENT: f32 = 0.3;
    pub const INFERENCE_PARTITION_PERCENT: f32 = 0.7;
    pub const GREEN_CONTEXTS_MIN_COMPUTE_MAJOR: i32 = 7;
    pub const GREEN_CONTEXTS_MIN_COMPUTE_MINOR: i32 = 0;
    pub const MIN_SMS_FOR_PARTITIONING: i32 = 80;

    pub fn should_enable_green_contexts(_compute_major: i32, _compute_minor: i32) -> bool {
        false
    }

    pub fn should_enable_green_contexts_with_config(
        _config: &GreenContextsConfig,
        _compute_major: i32,
        _compute_minor: i32,
    ) -> bool {
        false
    }

    #[derive(Debug, Clone)]
    pub struct GreenContextsConfig {
        pub enable_inference_partition: bool,
        pub enable_background_partition: bool,
    }

    #[derive(Debug, Clone)]
    pub struct GreenContext;

    #[derive(Debug, Clone)]
    pub struct GreenContexts;
}

// Re-export stubs when cuda is not enabled
#[cfg(not(feature = "cuda"))]
pub use green_contexts_stub::{
    should_enable_green_contexts, should_enable_green_contexts_with_config, GreenContext,
    GreenContexts, GreenContextsConfig, BACKGROUND_PARTITION_PERCENT,
    GREEN_CONTEXTS_MIN_COMPUTE_MAJOR, GREEN_CONTEXTS_MIN_COMPUTE_MINOR,
    INFERENCE_PARTITION_PERCENT, MIN_SMS_FOR_PARTITIONING,
};
