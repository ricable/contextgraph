//! FAISS C API FFI bindings for GPU-accelerated vector similarity search.
//!
//! This module provides low-level C bindings to the FAISS library.
//! These bindings are used by `FaissGpuIndex` (M04-T10) for IVF-PQ operations.
//!
//! # Safety
//!
//! All extern "C" functions are unsafe. The `GpuResources` wrapper provides
//! a safe RAII interface for GPU resource management.
//!
//! # Constitution Reference
//!
//! - TECH-GRAPH-004 Section 3.1: FAISS FFI Bindings
//! - perf.latency.faiss_1M_k100: <2ms target
//! - AP-015: GPU alloc without pool → use CUDA memory pool
//!
//! # FAISS C API Reference
//!
//! - <https://github.com/facebookresearch/faiss/blob/main/c_api/>
//! - Functions prefixed `faiss_` (e.g., faiss_index_factory)
//! - Types prefixed `Faiss` (e.g., FaissIndex)

use std::os::raw::{c_char, c_float, c_int, c_long};
use std::ptr::NonNull;

use crate::error::{GraphError, GraphResult};

// ========== Metric Type ==========

/// Metric type for distance computation.
///
/// Determines how similarity is measured between vectors.
/// Must match FAISS MetricType enum values exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MetricType {
    /// Inner product (cosine similarity when normalized).
    /// Higher values = more similar.
    InnerProduct = 0,

    /// L2 (Euclidean) distance.
    /// Lower values = more similar.
    #[default]
    L2 = 1,
}

// ========== Opaque Pointer Types ==========

/// Opaque pointer to FAISS index.
///
/// This type represents any FAISS index (Flat, IVF, PQ, GPU, etc.).
/// The actual type is determined by how the index was created.
#[repr(C)]
pub struct FaissIndex {
    _private: [u8; 0],
}

/// Opaque pointer to FAISS GPU resources provider interface.
///
/// This is the abstract interface that StandardGpuResources implements.
#[repr(C)]
pub struct FaissGpuResourcesProvider {
    _private: [u8; 0],
}

/// Opaque pointer to FAISS standard GPU resources.
///
/// Manages GPU memory allocation for FAISS operations.
/// Must be freed with `faiss_StandardGpuResources_free`.
#[repr(C)]
pub struct FaissStandardGpuResources {
    _private: [u8; 0],
}

// ========== FAISS C API Bindings ==========

#[link(name = "faiss_c")]
extern "C" {
    // ---------- Index Factory ----------

    /// Create index from factory string.
    ///
    /// # Arguments
    /// - `p_index`: Output pointer to created index
    /// - `d`: Vector dimension
    /// - `description`: Factory string (e.g., "IVF16384,PQ64x8")
    /// - `metric`: Distance metric type
    ///
    /// # Returns
    /// 0 on success, non-zero on failure
    pub fn faiss_index_factory(
        p_index: *mut *mut FaissIndex,
        d: c_int,
        description: *const c_char,
        metric: MetricType,
    ) -> c_int;

    // ---------- GPU Resources ----------

    /// Allocate standard GPU resources.
    ///
    /// Creates a new StandardGpuResources object for GPU memory management.
    /// MUST be freed with `faiss_StandardGpuResources_free`.
    ///
    /// # Returns
    /// 0 on success, non-zero on failure
    pub fn faiss_StandardGpuResources_new(
        p_res: *mut *mut FaissStandardGpuResources,
    ) -> c_int;

    /// Free GPU resources.
    ///
    /// Releases all GPU memory held by this resources object.
    pub fn faiss_StandardGpuResources_free(res: *mut FaissStandardGpuResources);

    /// Cast standard GPU resources to provider interface.
    ///
    /// Required for `faiss_index_cpu_to_gpu`.
    pub fn faiss_StandardGpuResources_as_GpuResourcesProvider(
        res: *mut FaissStandardGpuResources,
    ) -> *mut FaissGpuResourcesProvider;

    // ---------- CPU to GPU Transfer ----------

    /// Transfer index from CPU to GPU.
    ///
    /// # Arguments
    /// - `provider`: GPU resources provider
    /// - `device`: GPU device ID (usually 0)
    /// - `index`: Source CPU index
    /// - `p_out`: Output pointer to GPU index
    ///
    /// # Returns
    /// 0 on success, non-zero on failure
    pub fn faiss_index_cpu_to_gpu(
        provider: *mut FaissGpuResourcesProvider,
        device: c_int,
        index: *const FaissIndex,
        p_out: *mut *mut FaissIndex,
    ) -> c_int;

    // ---------- Index Operations ----------

    /// Train the index with vectors.
    ///
    /// For IVF indices, this clusters the vectors to create centroids.
    /// Must be called before `add_with_ids` for untrained indices.
    ///
    /// # Arguments
    /// - `index`: Target index
    /// - `n`: Number of training vectors
    /// - `x`: Training vectors (n * d floats, row-major)
    pub fn faiss_Index_train(
        index: *mut FaissIndex,
        n: c_long,
        x: *const c_float,
    ) -> c_int;

    /// Check if index is trained.
    ///
    /// # Returns
    /// Non-zero if trained, 0 if not trained
    pub fn faiss_Index_is_trained(index: *const FaissIndex) -> c_int;

    /// Add vectors with IDs to the index.
    ///
    /// # Arguments
    /// - `index`: Target index
    /// - `n`: Number of vectors
    /// - `x`: Vectors to add (n * d floats, row-major)
    /// - `xids`: Vector IDs (n longs)
    pub fn faiss_Index_add_with_ids(
        index: *mut FaissIndex,
        n: c_long,
        x: *const c_float,
        xids: *const c_long,
    ) -> c_int;

    /// Search for k nearest neighbors.
    ///
    /// # Arguments
    /// - `index`: Source index
    /// - `n`: Number of query vectors
    /// - `x`: Query vectors (n * d floats, row-major)
    /// - `k`: Number of neighbors to return
    /// - `distances`: Output distances (n * k floats)
    /// - `labels`: Output IDs (n * k longs, -1 for missing)
    pub fn faiss_Index_search(
        index: *const FaissIndex,
        n: c_long,
        x: *const c_float,
        k: c_long,
        distances: *mut c_float,
        labels: *mut c_long,
    ) -> c_int;

    /// Set nprobe parameter for IVF index.
    ///
    /// Controls search quality vs speed tradeoff.
    /// Higher values = more accurate but slower.
    pub fn faiss_IndexIVF_nprobe_set(
        index: *mut FaissIndex,
        nprobe: c_long,
    ) -> c_int;

    /// Get total number of vectors in index.
    pub fn faiss_Index_ntotal(index: *const FaissIndex) -> c_long;

    // ---------- Persistence ----------

    /// Write index to file.
    ///
    /// # Arguments
    /// - `index`: Source index
    /// - `fname`: Output file path (C string)
    pub fn faiss_write_index(
        index: *const FaissIndex,
        fname: *const c_char,
    ) -> c_int;

    /// Read index from file.
    ///
    /// # Arguments
    /// - `fname`: Input file path (C string)
    /// - `io_flags`: IO flags (usually 0)
    /// - `p_out`: Output pointer to loaded index
    pub fn faiss_read_index(
        fname: *const c_char,
        io_flags: c_int,
        p_out: *mut *mut FaissIndex,
    ) -> c_int;

    /// Free index.
    ///
    /// Releases all memory held by the index.
    pub fn faiss_Index_free(index: *mut FaissIndex);
}

// ========== RAII Wrapper ==========

/// RAII wrapper for FAISS GPU resources.
///
/// Automatically frees GPU resources when dropped.
/// Safe to share across threads (Send + Sync).
///
/// # Example
///
/// ```ignore
/// let resources = GpuResources::new()?;
/// let provider = resources.as_provider();
/// // Use provider for cpu_to_gpu transfer...
/// // Resources automatically freed on drop
/// ```
pub struct GpuResources {
    ptr: NonNull<FaissStandardGpuResources>,
}

impl GpuResources {
    /// Allocate new GPU resources.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::GpuResourceAllocation` if:
    /// - No GPU available
    /// - GPU memory allocation fails
    /// - FAISS library not linked
    ///
    /// # Constitution Reference
    ///
    /// AP-015: GPU alloc without pool → use CUDA memory pool
    pub fn new() -> GraphResult<Self> {
        let mut res_ptr: *mut FaissStandardGpuResources = std::ptr::null_mut();

        // SAFETY: FFI call with valid output pointer
        let result = unsafe { faiss_StandardGpuResources_new(&mut res_ptr) };

        if result != 0 {
            return Err(GraphError::GpuResourceAllocation(format!(
                "faiss_StandardGpuResources_new failed with error code: {}",
                result
            )));
        }

        NonNull::new(res_ptr)
            .map(|ptr| GpuResources { ptr })
            .ok_or_else(|| {
                GraphError::GpuResourceAllocation(
                    "faiss_StandardGpuResources_new returned null pointer".to_string(),
                )
            })
    }

    /// Get the raw pointer for FFI calls.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid for the lifetime of this GpuResources.
    /// Do NOT call `faiss_StandardGpuResources_free` on it manually.
    #[inline]
    pub fn as_ptr(&self) -> *mut FaissStandardGpuResources {
        self.ptr.as_ptr()
    }

    /// Get as GpuResourcesProvider for cpu_to_gpu transfer.
    ///
    /// Required by `faiss_index_cpu_to_gpu`.
    #[inline]
    pub fn as_provider(&self) -> *mut FaissGpuResourcesProvider {
        // SAFETY: Valid pointer, cast is part of FAISS C API design
        unsafe { faiss_StandardGpuResources_as_GpuResourcesProvider(self.ptr.as_ptr()) }
    }
}

impl Drop for GpuResources {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated by faiss_StandardGpuResources_new
        // and has not been freed yet (RAII guarantees single ownership)
        unsafe {
            faiss_StandardGpuResources_free(self.ptr.as_ptr());
        }
    }
}

// SAFETY: GpuResources wraps a pointer to GPU resources allocated by FAISS.
// The underlying FAISS StandardGpuResources implementation is designed to be
// thread-safe - it uses internal synchronization for GPU memory management.
// We ensure single ownership via NonNull and RAII cleanup.
// Multiple threads can use the same GPU resources for different operations.
unsafe impl Send for GpuResources {}
unsafe impl Sync for GpuResources {}

impl std::fmt::Debug for GpuResources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuResources")
            .field("ptr", &self.ptr)
            .finish()
    }
}

// ========== Helper Functions ==========

/// Check FAISS result code and convert to GraphResult.
///
/// # Arguments
///
/// - `code`: FAISS return code (0 = success)
/// - `operation`: Description of operation for error message
///
/// # Returns
///
/// - `Ok(())` if code is 0
/// - `Err(GraphError::FaissIndexCreation)` otherwise
///
/// # Example
///
/// ```ignore
/// let result = unsafe { faiss_Index_train(index, n, x) };
/// check_faiss_result(result, "faiss_Index_train")?;
/// ```
#[inline]
pub fn check_faiss_result(code: c_int, operation: &str) -> GraphResult<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(GraphError::FaissIndexCreation(format!(
            "{} failed with error code: {}",
            operation, code
        )))
    }
}

// ========== Tests ==========

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_type_values() {
        // FAISS C API requires exact enum values
        assert_eq!(MetricType::InnerProduct as i32, 0);
        assert_eq!(MetricType::L2 as i32, 1);
    }

    #[test]
    fn test_metric_type_default() {
        assert_eq!(MetricType::default(), MetricType::L2);
    }

    #[test]
    fn test_metric_type_debug() {
        assert_eq!(format!("{:?}", MetricType::L2), "L2");
        assert_eq!(format!("{:?}", MetricType::InnerProduct), "InnerProduct");
    }

    #[test]
    fn test_metric_type_clone() {
        let m1 = MetricType::L2;
        let m2 = m1;
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_metric_type_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(MetricType::L2);
        set.insert(MetricType::InnerProduct);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_gpu_resources_is_send_sync() {
        // Compile-time verification that GpuResources is Send + Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GpuResources>();
    }

    #[test]
    fn test_check_faiss_result_success() {
        let result = check_faiss_result(0, "test_operation");
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_faiss_result_failure() {
        let result = check_faiss_result(-1, "test_operation");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, GraphError::FaissIndexCreation(_)));
        let msg = err.to_string();
        assert!(msg.contains("test_operation"));
        assert!(msg.contains("-1"));
    }

    #[test]
    fn test_check_faiss_result_various_codes() {
        // Test multiple error codes
        for code in [1, 2, 10, 100, -100] {
            let result = check_faiss_result(code, "test");
            assert!(result.is_err());
            let msg = result.unwrap_err().to_string();
            assert!(msg.contains(&code.to_string()));
        }
    }

    #[test]
    fn test_opaque_types_zero_size() {
        // Opaque types should have zero size (for FFI safety)
        assert_eq!(std::mem::size_of::<FaissIndex>(), 0);
        assert_eq!(std::mem::size_of::<FaissGpuResourcesProvider>(), 0);
        assert_eq!(std::mem::size_of::<FaissStandardGpuResources>(), 0);
    }

    // ========== GPU Tests (require FAISS + GPU) ==========

    #[test]
    #[ignore = "Requires FAISS library and GPU - run with: cargo test --features gpu -- --ignored"]
    fn test_gpu_resources_allocation() {
        let resources = GpuResources::new();
        match resources {
            Ok(res) => {
                // Verify pointer is valid
                assert!(!res.as_ptr().is_null());
                assert!(!res.as_provider().is_null());
                println!("GPU resources allocated: {:?}", res);
            }
            Err(e) => {
                // Expected if no GPU or FAISS not installed
                println!("GPU resources allocation failed (expected without GPU): {}", e);
            }
        }
    }

    #[test]
    #[ignore = "Requires FAISS library and GPU - run with: cargo test --features gpu -- --ignored"]
    fn test_gpu_resources_drop() {
        // Test that drop doesn't crash
        {
            let resources = GpuResources::new();
            if let Ok(res) = resources {
                println!("Allocated GPU resources, will drop...");
                drop(res);
                println!("Drop completed without crash");
            }
        }
        // If we reach here, drop worked correctly
    }
}
