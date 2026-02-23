//! FAISS FFI bindings bridge - delegates to context-graph-cuda.
//!
//! # Architecture
//!
//! This module serves as a bridge to the FAISS FFI bindings in `context-graph-cuda`.
//! ALL FAISS FFI declarations are centralized in `context-graph-cuda/src/ffi/faiss.rs`
//! per Constitution ARCH-06 (CUDA FFI only in context-graph-cuda).
//!
//! # FAISS GPU Availability
//!
//! FAISS GPU requires the `faiss-working` feature to be enabled.
//! Without this feature, FAISS operations will FAIL FAST with clear error messages.
//! There are NO fallbacks, NO stubs, NO workarounds - per Constitution ARCH-GPU-04.
//!
//! ## To Enable FAISS GPU
//!
//! 1. Rebuild FAISS from source with CUDA 13.1+ and sm_120 support:
//!    ```bash
//!    ./scripts/rebuild_faiss_gpu.sh
//!    ```
//!
//! 2. Build the workspace with the faiss-working feature:
//!    ```bash
//!    cargo build --features faiss-working
//!    ```
//!
//! # Constitution Compliance
//!
//! - ARCH-06: CUDA FFI only in context-graph-cuda (this module re-exports from there)
//! - ARCH-GPU-04: FAISS indexes use GPU (faiss-gpu) not CPU - NO FALLBACK
//! - AP-001: Fail fast, never unwrap() in prod
//! - AP-GPU-03: NEVER use CPU FAISS when GPU FAISS available

#![allow(non_snake_case)]

use crate::error::{GraphError, GraphResult};

/// Standard help message for FAISS GPU unavailability.
/// Used consistently across all error paths per fail-fast design.
const FAISS_UNAVAILABLE_HELP: &str =
    "Fix: 1) Run ./scripts/rebuild_faiss_gpu.sh \
     2) Build with: cargo build --features faiss-working \
     Per Constitution ARCH-GPU-04: NO CPU FALLBACK.";

// =============================================================================
// FEATURE-GATED RE-EXPORTS FROM context-graph-cuda
// =============================================================================

/// Check if FAISS GPU is available.
///
/// Returns `true` only when:
/// 1. `context-graph-cuda` was built with `faiss-working` feature
/// 2. FAISS reports at least one GPU
/// 3. GPU actually works (verified at runtime)
#[cfg(feature = "faiss-working")]
#[inline]
pub fn gpu_available() -> bool {
    context_graph_cuda::is_faiss_gpu_available()
}

#[cfg(not(feature = "faiss-working"))]
#[inline]
pub fn gpu_available() -> bool {
    false
}

/// Check if CUDA is available via driver API.
///
/// This works even when FAISS GPU is not available.
#[cfg(feature = "faiss-working")]
#[inline]
pub fn cuda_driver_available() -> bool {
    context_graph_cuda::ffi::cuda_available()
}

#[cfg(not(feature = "faiss-working"))]
#[inline]
pub fn cuda_driver_available() -> bool {
    false
}

/// Get a human-readable status of FAISS GPU availability.
#[cfg(feature = "faiss-working")]
#[inline]
pub fn faiss_status() -> &'static str {
    context_graph_cuda::faiss_status()
}

#[cfg(not(feature = "faiss-working"))]
#[inline]
pub fn faiss_status() -> &'static str {
    "FAISS GPU not available - faiss-working feature not enabled"
}

// =============================================================================
// WHEN faiss-working IS ENABLED: Re-export real FAISS FFI
// =============================================================================

#[cfg(feature = "faiss-working")]
mod real_faiss {
    //! Real FAISS FFI re-exports from context-graph-cuda.
    //!
    //! This module is ONLY compiled when faiss-working feature is enabled.

    // Re-export types
    pub use context_graph_cuda::FaissGpuResourcesProvider;
    pub use context_graph_cuda::FaissIndex;
    pub use context_graph_cuda::FaissStandardGpuResources;
    pub use context_graph_cuda::MetricType;
    pub use context_graph_cuda::FAISS_OK;

    // Re-export GpuResources RAII wrapper
    pub use context_graph_cuda::FaissGpuResources as GpuResources;

    // Re-export helper
    pub use context_graph_cuda::check_faiss_result as check_faiss_result_cuda;

    // Re-export FFI functions from the faiss module
    pub use context_graph_cuda::ffi::faiss::{
        faiss_Index_add_with_ids, faiss_Index_free, faiss_Index_is_trained, faiss_Index_ntotal,
        faiss_Index_search, faiss_Index_train, faiss_IndexIVF_set_nprobe,
        faiss_StandardGpuResources_free, faiss_StandardGpuResources_new, faiss_get_num_gpus,
        faiss_index_cpu_to_gpu, faiss_index_factory, faiss_read_index, faiss_write_index,
    };
}

#[cfg(feature = "faiss-working")]
pub use real_faiss::*;

/// Wrapper for check_faiss_result that converts to GraphError.
///
/// This provides a unified error interface for the graph crate.
#[cfg(feature = "faiss-working")]
#[inline]
pub fn check_faiss_result(code: i32, operation: &str) -> GraphResult<()> {
    if code == FAISS_OK {
        Ok(())
    } else {
        Err(GraphError::FaissIndexCreation(format!(
            "FAISS {} failed with code {}. \
             Check: 1) CUDA 13.1+ installed, 2) GPU available, 3) libfaiss_c.so linked. \
             Status: {}",
            operation,
            code,
            faiss_status()
        )))
    }
}

// =============================================================================
// WHEN faiss-working IS NOT ENABLED: Fail fast with clear errors
// =============================================================================

#[cfg(not(feature = "faiss-working"))]
mod fail_fast {
    //! Fail-fast stubs when FAISS is not available.
    //!
    //! These implementations ALWAYS return errors - no fallbacks, no workarounds.
    //! Per Constitution ARCH-GPU-04: FAISS indexes use GPU, NO CPU fallback.

    use super::{faiss_status, GraphError, GraphResult};
    use std::ffi::c_void;
    use std::os::raw::c_char;

    /// FAISS success code.
    pub const FAISS_OK: i32 = 0;

    /// Metric type for FAISS indexes.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum MetricType {
        InnerProduct = 0,
        #[default]
        L2 = 1,
    }

    /// Opaque FAISS index pointer (placeholder when FAISS not available).
    pub type FaissIndex = c_void;

    /// Opaque FAISS GPU resources provider pointer.
    pub type FaissGpuResourcesProvider = c_void;

    /// Opaque FAISS standard GPU resources pointer.
    pub type FaissStandardGpuResources = c_void;

    /// GPU resources wrapper - ALWAYS fails when FAISS is not available.
    ///
    /// This type cannot be instantiated because `new()` always returns an error.
    /// The methods besides `new()` exist only for API compatibility but will
    /// never be called in practice.
    pub struct GpuResources {
        // Private field prevents external construction
        _private: (),
    }

    impl GpuResources {
        /// Create new GPU resources.
        ///
        /// # Errors
        ///
        /// ALWAYS returns error when faiss-working feature is not enabled.
        /// There is NO fallback per Constitution ARCH-GPU-04.
        pub fn new() -> GraphResult<Self> {
            Err(GraphError::FaissGpuUnavailable {
                reason: format!(
                    "FAISS GPU is REQUIRED but not available. Status: {}",
                    faiss_status()
                ),
                help: super::FAISS_UNAVAILABLE_HELP.to_string(),
            })
        }

        /// Get the provider pointer for FFI calls.
        ///
        /// # Safety Note
        ///
        /// This method exists for API compatibility but will never be called
        /// in practice because `new()` always returns an error. Returns null
        /// to ensure fail-fast behavior if somehow reached.
        #[allow(unreachable_code)]
        pub fn as_provider(&self) -> *mut FaissGpuResourcesProvider {
            // This should NEVER be called - log error for debugging
            tracing::error!(
                target: "context_graph::faiss",
                "BUG: GpuResources::as_provider called but FAISS GPU unavailable. \
                 This indicates a bug - new() should have failed."
            );
            std::ptr::null_mut()
        }
    }

    impl std::fmt::Debug for GpuResources {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("GpuResources")
                .field("available", &false)
                .field("status", &faiss_status())
                .finish()
        }
    }

    /// Log fatal error and return -1 for any FAISS function call.
    fn fail_fast_faiss_call(function_name: &str) -> i32 {
        tracing::error!(
            target: "context_graph::faiss",
            "FATAL: {} called but FAISS GPU is REQUIRED and not available. \
             Status: {}. {}",
            function_name,
            faiss_status(),
            super::FAISS_UNAVAILABLE_HELP
        );
        -1
    }

    // All FAISS functions fail fast with clear errors

    /// # Safety
    /// FFI stub - FAISS GPU not available, always returns error.
    pub unsafe fn faiss_index_factory(
        _p_index: *mut *mut c_void,
        _d: i32,
        _description: *const c_char,
        _metric: MetricType,
    ) -> i32 {
        fail_fast_faiss_call("faiss_index_factory")
    }

    /// # Safety
    /// FFI stub - FAISS GPU not available, always returns error.
    pub unsafe fn faiss_index_cpu_to_gpu(
        _provider: *mut c_void,
        _device: i32,
        _index: *mut c_void,
        _p_out: *mut *mut c_void,
    ) -> i32 {
        fail_fast_faiss_call("faiss_index_cpu_to_gpu")
    }

    /// # Safety
    /// FFI stub - FAISS GPU not available, no-op.
    pub unsafe fn faiss_Index_free(_index: *mut c_void) {
        tracing::error!(
            target: "context_graph::faiss",
            "faiss_Index_free called but FAISS GPU not available - no-op"
        );
    }

    /// # Safety
    /// FFI stub - FAISS GPU not available, always returns error.
    pub unsafe fn faiss_Index_add_with_ids(
        _index: *mut c_void,
        _n: i64,
        _x: *const f32,
        _xids: *const i64,
    ) -> i32 {
        fail_fast_faiss_call("faiss_Index_add_with_ids")
    }

    /// # Safety
    /// FFI stub - FAISS GPU not available, always returns error.
    pub unsafe fn faiss_Index_search(
        _index: *const c_void,
        _n: i64,
        _x: *const f32,
        _k: i64,
        _distances: *mut f32,
        _labels: *mut i64,
    ) -> i32 {
        fail_fast_faiss_call("faiss_Index_search")
    }

    /// # Safety
    /// FFI stub - FAISS GPU not available, always returns error.
    pub unsafe fn faiss_Index_train(_index: *mut c_void, _n: i64, _x: *const f32) -> i32 {
        fail_fast_faiss_call("faiss_Index_train")
    }

    /// # Safety
    /// FFI stub - FAISS GPU not available, always returns error.
    pub unsafe fn faiss_Index_is_trained(_index: *const c_void) -> i32 {
        fail_fast_faiss_call("faiss_Index_is_trained")
    }

    /// # Safety
    /// FFI stub - FAISS GPU not available, returns -1 to signal error.
    pub unsafe fn faiss_Index_ntotal(_index: *const c_void) -> i64 {
        tracing::error!(
            target: "context_graph::faiss",
            "faiss_Index_ntotal called but FAISS GPU not available - returning -1 to signal error"
        );
        -1 // Return -1 to signal error (0 would be confused with empty index)
    }

    /// # Safety
    /// FFI stub - FAISS GPU not available, always returns error.
    pub unsafe fn faiss_IndexIVF_set_nprobe(_index: *mut c_void, _nprobe: usize) -> i32 {
        fail_fast_faiss_call("faiss_IndexIVF_set_nprobe")
    }

    /// # Safety
    /// FFI stub - FAISS GPU not available, always returns error.
    pub unsafe fn faiss_StandardGpuResources_new(_p_res: *mut *mut c_void) -> i32 {
        fail_fast_faiss_call("faiss_StandardGpuResources_new")
    }

    /// # Safety
    /// FFI stub - FAISS GPU not available, no-op.
    pub unsafe fn faiss_StandardGpuResources_free(_res: *mut c_void) {
        tracing::error!(
            target: "context_graph::faiss",
            "faiss_StandardGpuResources_free called but FAISS GPU not available - no-op"
        );
    }

    /// # Safety
    /// FFI stub - FAISS GPU not available, returns 0.
    pub unsafe fn faiss_get_num_gpus() -> i32 {
        tracing::warn!(
            target: "context_graph::faiss",
            "faiss_get_num_gpus called but FAISS GPU not available - returning 0"
        );
        0
    }

    /// # Safety
    /// FFI stub - FAISS GPU not available, always returns error.
    pub unsafe fn faiss_read_index(
        _fname: *const c_char,
        _io_flags: i32,
        _p_out: *mut *mut c_void,
    ) -> i32 {
        fail_fast_faiss_call("faiss_read_index")
    }

    /// # Safety
    /// FFI stub - FAISS GPU not available, always returns error.
    pub unsafe fn faiss_write_index(_index: *const c_void, _fname: *const c_char) -> i32 {
        fail_fast_faiss_call("faiss_write_index")
    }
}

#[cfg(not(feature = "faiss-working"))]
pub use fail_fast::*;

/// Wrapper for check_faiss_result when FAISS is not available.
#[cfg(not(feature = "faiss-working"))]
#[inline]
pub fn check_faiss_result(code: i32, operation: &str) -> GraphResult<()> {
    if code == FAISS_OK {
        Ok(())
    } else {
        Err(GraphError::FaissIndexCreation(format!(
            "FAISS {} failed with code {} - FAISS GPU is REQUIRED but not available. \
             Status: {}. \
             Fix: 1) Run ./scripts/rebuild_faiss_gpu.sh \
             2) Build with: cargo build --features faiss-working",
            operation,
            code,
            faiss_status()
        )))
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_available_returns_bool() {
        // This test verifies the function works without panicking
        let available = gpu_available();
        println!("FAISS GPU available: {}", available);
        println!("FAISS status: {}", faiss_status());

        // If faiss-working is enabled and GPU is available, this should be true
        // Otherwise it should be false - both are valid test outcomes
    }

    #[test]
    fn test_cuda_driver_available() {
        let available = cuda_driver_available();
        println!("CUDA driver available: {}", available);
    }

    #[test]
    fn test_metric_type_default() {
        assert_eq!(MetricType::default(), MetricType::L2);
    }

    #[test]
    fn test_metric_type_values() {
        assert_eq!(MetricType::InnerProduct as i32, 0);
        assert_eq!(MetricType::L2 as i32, 1);
    }

    #[test]
    fn test_check_faiss_result_ok() {
        assert!(check_faiss_result(FAISS_OK, "test").is_ok());
    }

    #[test]
    fn test_check_faiss_result_error() {
        let result = check_faiss_result(-1, "test");
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_str = err.to_string();
        assert!(
            err_str.contains("FAISS") || err_str.contains("faiss"),
            "Error should mention FAISS: {}",
            err_str
        );
    }

    #[test]
    fn test_faiss_status_not_empty() {
        let status = faiss_status();
        assert!(!status.is_empty(), "FAISS status should not be empty");
        println!("FAISS status: {}", status);
    }

    #[cfg(feature = "faiss-working")]
    #[test]
    fn test_gpu_resources_with_faiss_working() {
        if gpu_available() {
            // When FAISS GPU is available, this should succeed
            let result = GpuResources::new();
            assert!(
                result.is_ok(),
                "GpuResources::new should succeed when FAISS GPU is available"
            );
        } else {
            // When FAISS is built but GPU unavailable, it should fail with clear error
            let result = GpuResources::new();
            if result.is_err() {
                let err = result.unwrap_err();
                println!("Expected error (GPU unavailable): {}", err);
            }
        }
    }

    #[cfg(not(feature = "faiss-working"))]
    #[test]
    fn test_gpu_resources_without_faiss_working() {
        // Without faiss-working, GpuResources::new MUST fail
        let result = GpuResources::new();
        assert!(
            result.is_err(),
            "GpuResources::new MUST fail when faiss-working is not enabled"
        );

        let err = result.unwrap_err();
        let err_str = err.to_string();
        assert!(
            err_str.contains("FAISS") || err_str.contains("REQUIRED"),
            "Error should mention FAISS and requirement: {}",
            err_str
        );
        println!("Got expected error: {}", err_str);
    }

    #[cfg(not(feature = "faiss-working"))]
    #[test]
    fn test_faiss_ntotal_returns_error_signal() {
        // When FAISS is not available, faiss_Index_ntotal should return -1
        // to signal an error (not 0, which could be confused with empty index)
        let result = unsafe { faiss_Index_ntotal(std::ptr::null()) };
        assert_eq!(
            result, -1,
            "faiss_Index_ntotal should return -1 to signal error when FAISS unavailable"
        );
    }
}
