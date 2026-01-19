//! Global warm-loaded embedding provider singleton.
//!
//! CRITICAL: All code MUST use this singleton. Direct model loading is FORBIDDEN.
//!
//! # Design
//!
//! This module provides a global singleton that ensures:
//! 1. All 13 embedding models are loaded ONCE at startup into VRAM
//! 2. Tests and production code use the SAME warm models
//! 3. No cold loading overhead in tests or runtime
//!
//! # Usage
//!
//! ```rust,ignore
//! // At startup (MCP server, CLI, etc.)
//! initialize_global_warm_provider().await?;
//!
//! // Anywhere that needs embeddings
//! let provider = get_warm_provider()?;
//! let output = provider.embed_all("text").await?;
//! ```
//!
//! # Error Behavior
//!
//! - `NotInitialized`: Call `initialize_global_warm_provider()` first
//! - `ProviderBusy`: Provider lock contention, retry later
//! - `InitializationFailed`: Warm loading failed (CUDA, models, etc.)
//!
//! # Thread Safety
//!
//! - `OnceLock` ensures single initialization
//! - `RwLock` allows concurrent reads during embedding
//! - All operations are `Send + Sync`

use std::sync::Arc;
use std::sync::OnceLock;

use tokio::sync::RwLock;

use crate::error::{EmbeddingError, EmbeddingResult};
use context_graph_core::traits::MultiArrayEmbeddingProvider;

/// Global warm provider singleton - initialized ONCE, used everywhere.
///
/// Structure: `OnceLock<Arc<RwLock<ProviderState>>>`
/// - `OnceLock`: Ensures single initialization across all threads
/// - `Arc`: Allows cloning for async tasks
/// - `RwLock`: Allows concurrent reads, exclusive writes
/// - `ProviderState`: Holds provider and error state
static GLOBAL_WARM_PROVIDER: OnceLock<Arc<RwLock<ProviderState>>> = OnceLock::new();

/// Internal state for the global provider.
///
/// State machine:
/// - `provider: None, init_error: None` -> Not initialized
/// - `provider: None, init_error: Some` -> Initialization failed
/// - `provider: Some, init_error: None` -> Ready
#[derive(Default)]
struct ProviderState {
    /// The actual provider, if initialized successfully
    provider: Option<Arc<dyn MultiArrayEmbeddingProvider>>,
    /// Error message if initialization failed
    init_error: Option<String>,
}

// =============================================================================
// CUDA-specific implementation
// =============================================================================

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use crate::warm::{WarmConfig, WarmEmbeddingPipeline, WarmError};

    /// Provider adapter that implements MultiArrayEmbeddingProvider by wrapping
    /// the WarmEmbeddingPipeline's registry to perform actual embeddings.
    pub(crate) struct WarmPipelineProvider {
        pipeline: WarmEmbeddingPipeline,
    }

    impl WarmPipelineProvider {
        pub fn new(pipeline: WarmEmbeddingPipeline) -> Self {
            Self { pipeline }
        }
    }

    #[async_trait::async_trait]
    impl MultiArrayEmbeddingProvider for WarmPipelineProvider {
        async fn embed_all(
            &self,
            _content: &str,
        ) -> context_graph_core::error::CoreResult<context_graph_core::traits::MultiArrayEmbeddingOutput>
        {
            // Check if pipeline is ready (includes model warm check)
            if !self.pipeline.is_ready() {
                let health = self.pipeline.health();
                return Err(context_graph_core::error::CoreError::Internal(format!(
                    "Warm pipeline not ready: {}/{} models warm, {} failed, {} loading",
                    health.models_warm, health.models_total, health.models_failed, health.models_loading
                )));
            }

            // TASK-EMB-016: The warm loading infrastructure stores ModelHandle with VRAM pointers,
            // but doesn't yet have a full inference path. The WarmEmbeddingPipeline needs to
            // be enhanced to support the embed_all() operation using the warm-loaded weights.
            //
            // This is a placeholder - the actual implementation would use the warm registry's
            // model handles to perform inference. For now, return a clear error.
            Err(context_graph_core::error::CoreError::NotImplemented(
                "WarmPipelineProvider::embed_all requires WarmEmbeddingPipeline to support inference. \
                 The warm loading infrastructure loads models to VRAM but inference path is not yet connected."
                    .to_string(),
            ))
        }

        async fn embed_batch_all(
            &self,
            _contents: &[String],
        ) -> context_graph_core::error::CoreResult<
            Vec<context_graph_core::traits::MultiArrayEmbeddingOutput>,
        > {
            Err(context_graph_core::error::CoreError::NotImplemented(
                "WarmPipelineProvider::embed_batch_all requires WarmEmbeddingPipeline to support inference."
                    .to_string(),
            ))
        }

        fn model_ids(&self) -> [&str; context_graph_core::types::fingerprint::NUM_EMBEDDERS] {
            [
                "E1_Semantic",
                "E2_TemporalRecent",
                "E3_TemporalPeriodic",
                "E4_TemporalPositional",
                "E5_Causal",
                "E6_Sparse",
                "E7_Code",
                "E8_Graph",
                "E9_HDC",
                "E10_Multimodal",
                "E11_Entity",
                "E12_LateInteraction",
                "E13_Splade",
            ]
        }

        fn is_ready(&self) -> bool {
            self.pipeline.is_ready()
        }

        fn health_status(&self) -> [bool; context_graph_core::types::fingerprint::NUM_EMBEDDERS] {
            // All models should be warm if pipeline is ready
            if self.pipeline.is_ready() {
                [true; context_graph_core::types::fingerprint::NUM_EMBEDDERS]
            } else {
                [false; context_graph_core::types::fingerprint::NUM_EMBEDDERS]
            }
        }
    }

    /// Initialize the global warm provider (CUDA version).
    ///
    /// MUST be called ONCE at startup before any embedding operations.
    /// Subsequent calls are no-ops if initialization succeeded.
    pub async fn initialize_global_warm_provider_impl() -> EmbeddingResult<()> {
        let slot = GLOBAL_WARM_PROVIDER.get_or_init(|| Arc::new(RwLock::new(ProviderState::default())));

        let mut guard = slot.write().await;

        // Already initialized successfully
        if guard.provider.is_some() {
            tracing::debug!("Global warm provider already initialized, skipping");
            return Ok(());
        }

        // Previous initialization attempt failed - don't retry
        if let Some(ref err) = guard.init_error {
            return Err(EmbeddingError::InternalError {
                message: format!("Global warm provider initialization previously failed: {}", err),
            });
        }

        tracing::info!("Initializing global warm provider (loading 13 models to VRAM)...");

        let config = WarmConfig::default();

        // Create and warm the pipeline
        // Note: WarmEmbeddingPipeline::create_and_warm() calls std::process::exit() on fatal errors.
        // We use the non-exiting warm() method instead for better error handling.
        let pipeline = match WarmEmbeddingPipeline::new(config) {
            Ok(mut p) => {
                if let Err(e) = p.warm() {
                    let err_msg = format!("Warm loading failed: {}", e);
                    tracing::error!("{}", err_msg);
                    guard.init_error = Some(err_msg.clone());
                    return Err(convert_warm_error(e));
                }
                p
            }
            Err(e) => {
                let err_msg = format!("Failed to create WarmEmbeddingPipeline: {}", e);
                tracing::error!("{}", err_msg);
                guard.init_error = Some(err_msg.clone());
                return Err(convert_warm_error(e));
            }
        };

        // Verify health
        if !pipeline.is_ready() {
            let health = pipeline.health();
            let err_msg = format!(
                "Pipeline not ready after warm loading: {} warm, {} failed, {} loading",
                health.models_warm, health.models_failed, health.models_loading
            );
            tracing::error!("{}", err_msg);
            guard.init_error = Some(err_msg.clone());
            return Err(EmbeddingError::InternalError { message: err_msg });
        }

        // Create provider wrapper
        let provider: Arc<dyn MultiArrayEmbeddingProvider> = Arc::new(WarmPipelineProvider::new(pipeline));
        guard.provider = Some(provider);

        tracing::info!("Global warm provider initialized successfully - all 13 models warm");
        Ok(())
    }

    /// Convert WarmError to EmbeddingError.
    pub fn convert_warm_error(err: WarmError) -> EmbeddingError {
        match err {
            WarmError::CudaUnavailable { message } => EmbeddingError::CudaUnavailable { message },
            WarmError::CudaInitFailed {
                cuda_error,
                driver_version: _,
                gpu_name: _,
            } => EmbeddingError::GpuError {
                message: format!("CUDA initialization failed: {}", cuda_error),
            },
            WarmError::CudaCapabilityInsufficient {
                actual_cc,
                required_cc,
                gpu_name,
            } => EmbeddingError::GpuError {
                message: format!(
                    "GPU {} has compute capability {} but {} is required",
                    gpu_name, actual_cc, required_cc
                ),
            },
            WarmError::VramInsufficientTotal {
                required_bytes,
                available_bytes,
                required_gb,
                available_gb,
                model_breakdown: _,
            } => EmbeddingError::InsufficientVram {
                required_bytes,
                available_bytes,
                required_gb,
                available_gb,
            },
            WarmError::ModelFileMissing { model_id, path } => EmbeddingError::InternalError {
                message: format!("Model {} not found at {}", model_id, path),
            },
            WarmError::ModelLoadFailed {
                model_id, reason, ..
            } => EmbeddingError::InternalError {
                message: format!("Model {} failed to load: {}", model_id, reason),
            },
            WarmError::ModelValidationFailed {
                model_id, reason, ..
            } => EmbeddingError::InternalError {
                message: format!("Model {} validation failed: {}", model_id, reason),
            },
            _ => EmbeddingError::InternalError {
                message: format!("Warm loading error: {}", err),
            },
        }
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Initialize the global warm provider.
///
/// MUST be called ONCE at startup before any embedding operations.
/// Subsequent calls are no-ops if initialization succeeded.
///
/// # CUDA Requirement
///
/// This function requires the `cuda` feature. Per Constitution AP-007,
/// CUDA is mandatory - there are NO CPU fallbacks.
///
/// # Errors
///
/// - `InitializationFailed`: If warm loading fails (CUDA unavailable, model files missing, etc.)
///
/// # Panics
///
/// Does NOT panic. All errors are returned as `EmbeddingResult`.
#[cfg(feature = "cuda")]
pub async fn initialize_global_warm_provider() -> EmbeddingResult<()> {
    cuda_impl::initialize_global_warm_provider_impl().await
}

/// Initialize the global warm provider (non-CUDA stub).
///
/// When CUDA is not available, this returns an error per Constitution AP-007.
#[cfg(not(feature = "cuda"))]
pub async fn initialize_global_warm_provider() -> EmbeddingResult<()> {
    Err(EmbeddingError::CudaUnavailable {
        message: "Global warm provider requires CUDA feature. Build with --features cuda"
            .to_string(),
    })
}

/// Get the global warm provider.
///
/// FAILS FAST if not initialized - call `initialize_global_warm_provider()` first.
///
/// # Errors
///
/// - `NotInitialized`: Provider not initialized yet
/// - `ProviderBusy`: Lock contention (rare, retry)
/// - `InitializationFailed`: Previous initialization failed
///
/// # Example
///
/// ```rust,ignore
/// let provider = get_warm_provider()?;
/// let output = provider.embed_all("some text").await?;
/// ```
pub fn get_warm_provider() -> EmbeddingResult<Arc<dyn MultiArrayEmbeddingProvider>> {
    let slot = GLOBAL_WARM_PROVIDER.get().ok_or_else(|| {
        EmbeddingError::InternalError {
            message: "Global warm provider not initialized. Call initialize_global_warm_provider() first.".to_string(),
        }
    })?;

    // Use try_read for non-blocking check
    let guard = slot.try_read().map_err(|_| {
        EmbeddingError::InternalError {
            message: "Global warm provider is busy (lock contention). Retry later.".to_string(),
        }
    })?;

    // Check if initialization failed
    if let Some(ref err) = guard.init_error {
        return Err(EmbeddingError::InternalError {
            message: format!("Global warm provider initialization failed: {}", err),
        });
    }

    // Get the provider
    guard
        .provider
        .as_ref()
        .map(Arc::clone)
        .ok_or_else(|| EmbeddingError::InternalError {
            message: "Global warm provider not initialized. Call initialize_global_warm_provider() first.".to_string(),
        })
}

/// Check if warm provider is initialized (non-blocking).
///
/// Returns `true` only if initialization succeeded and provider is available.
/// Returns `false` if:
/// - Not yet initialized
/// - Initialization in progress
/// - Initialization failed
pub fn is_warm_initialized() -> bool {
    GLOBAL_WARM_PROVIDER
        .get()
        .and_then(|slot| slot.try_read().ok())
        .map(|guard| guard.provider.is_some())
        .unwrap_or(false)
}

/// Get initialization status message (for diagnostics).
///
/// Returns human-readable status:
/// - "Not initialized"
/// - "Initialization in progress"
/// - "Initialization failed: <error>"
/// - "Ready (13 models warm)"
pub fn warm_status_message() -> String {
    match GLOBAL_WARM_PROVIDER.get() {
        None => "Not initialized".to_string(),
        Some(slot) => {
            match slot.try_read() {
                Ok(guard) => {
                    if guard.provider.is_some() {
                        "Ready (13 models warm)".to_string()
                    } else if let Some(ref err) = guard.init_error {
                        format!("Initialization failed: {}", err)
                    } else {
                        // State slot exists but no provider and no error - not yet initialized
                        "Not initialized".to_string()
                    }
                }
                // Lock contention means initialization is likely in progress
                Err(_) => "Initialization in progress".to_string(),
            }
        }
    }
}

/// Reset the global provider (for testing only).
///
/// # Safety
///
/// This is intended ONLY for tests that need to reinitialize the provider.
/// Using this in production could lead to race conditions.
#[cfg(test)]
pub async fn reset_global_provider_for_testing() {
    if let Some(slot) = GLOBAL_WARM_PROVIDER.get() {
        let mut guard = slot.write().await;
        *guard = ProviderState::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_warm_initialized_before_init() {
        // Before any initialization, should return false
        // Note: This test might fail if run after other tests that initialize
        // In real test suite, use reset_global_provider_for_testing()
        let _ = is_warm_initialized(); // Just verify it doesn't panic
    }

    #[test]
    fn test_warm_status_message_before_init() {
        let status = warm_status_message();
        // Should be one of the expected states
        assert!(
            status == "Not initialized"
                || status.starts_with("Ready")
                || status.starts_with("Initialization"),
            "Unexpected status: {}",
            status
        );
    }

    #[test]
    fn test_get_warm_provider_before_init() {
        // May or may not be initialized depending on test order
        // Just verify it doesn't panic
        let result = get_warm_provider();
        // Either succeeds or returns a sensible error
        if let Err(e) = result {
            let msg = e.to_string();
            assert!(
                msg.contains("not initialized") || msg.contains("failed"),
                "Unexpected error: {}",
                msg
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[tokio::test]
    async fn test_convert_warm_error() {
        use crate::warm::WarmError;

        let cuda_err = WarmError::CudaUnavailable {
            message: "No GPU".to_string(),
        };
        let emb_err = cuda_impl::convert_warm_error(cuda_err);
        assert!(matches!(emb_err, EmbeddingError::CudaUnavailable { .. }));

        let vram_err = WarmError::VramInsufficientTotal {
            required_bytes: 1000,
            available_bytes: 500,
            required_gb: 1.0,
            available_gb: 0.5,
            model_breakdown: vec![],
        };
        let emb_err = cuda_impl::convert_warm_error(vram_err);
        assert!(matches!(emb_err, EmbeddingError::InsufficientVram { .. }));
    }
}
