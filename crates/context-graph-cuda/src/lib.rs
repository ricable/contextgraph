#![deny(deprecated)]

//! CUDA acceleration for Context Graph.
//!
//! This crate provides GPU-accelerated operations for:
//! - Vector similarity search (cosine, dot product)
//! - Neural attention mechanisms
//! - Modern Hopfield network computations
//!
//! # Constitution AP-007 Compliance
//!
//! **CUDA is ALWAYS required - no stub implementations in production.**
//!
//! The `StubVectorOps` type is available ONLY in test builds (`#[cfg(test)]`)
//! and must NOT be used in production code paths. All production code must
//! use real CUDA implementations.
//!
//! # Target Hardware
//!
//! - RTX 5090 (32GB GDDR7, 1.8 TB/s bandwidth)
//! - CUDA 13.1 with Compute Capability 12.0
//! - Blackwell architecture optimizations
//!
//! # Example (Test Only)
//!
//! ```ignore
//! // StubVectorOps is only available in #[cfg(test)] builds
//! #[cfg(test)]
//! use context_graph_cuda::{StubVectorOps, VectorOps};
//!
//! #[cfg(test)]
//! fn test_example() {
//!     let ops = StubVectorOps::new();
//!     assert!(!ops.is_gpu_available());
//! }
//! ```

pub mod cone;
pub mod error;
pub mod ffi;
pub mod ops;
pub mod poincare;
pub mod safe;

// AP-007: StubVectorOps is TEST ONLY - not available in production builds
#[cfg(test)]
pub mod stub;

pub use cone::{
    cone_check_batch_cpu, cone_membership_score_cpu, get_cone_kernel_info, is_cone_gpu_available,
    ConeCudaConfig, ConeData, ConeKernelInfo, CONE_DATA_DIM, POINT_DIM,
};
#[cfg(feature = "cuda")]
pub use cone::{cone_check_batch_gpu, cone_check_single_gpu};
pub use error::{CudaError, CudaResult};
pub use ffi::{
    // CUDA Driver API exports
    cuda_result_to_string, decode_driver_version, is_cuda_success,
    cuDeviceGet, cuDeviceGetAttribute, cuDeviceGetCount, cuDeviceGetName,
    cuDeviceTotalMem_v2, cuDriverGetVersion, cuInit,
    // Context management (TASK-04)
    cuCtxCreate_v2, cuCtxDestroy_v2, cuCtxGetCurrent, cuCtxSetCurrent, cuMemGetInfo_v2,
    CUcontext, CUdevice, CUdevice_attribute, CUresult,
    CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_NO_DEVICE, CUDA_ERROR_NOT_INITIALIZED, CUDA_SUCCESS,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE,
    // FAISS FFI exports
    check_faiss_result, gpu_available, GpuResources, MetricType,
    FaissIndex, FaissGpuResourcesProvider, FaissStandardGpuResources,
    faiss_index_factory, faiss_Index_free, faiss_Index_train, faiss_Index_is_trained,
    faiss_Index_add_with_ids, faiss_Index_search, faiss_IndexIVF_set_nprobe,
    faiss_Index_ntotal, faiss_write_index, faiss_read_index,
    faiss_StandardGpuResources_new, faiss_StandardGpuResources_free,
    faiss_index_cpu_to_gpu, faiss_get_num_gpus,
    FAISS_OK,
};
// Safe RAII wrappers (TASK-04)
pub use safe::GpuDevice;
#[cfg(feature = "cuda")]
pub use ffi::gpu_count_direct;
pub use ops::VectorOps;
pub use poincare::{poincare_distance_batch_cpu, poincare_distance_cpu, PoincareCudaConfig};
#[cfg(feature = "cuda")]
pub use poincare::{poincare_distance_batch_gpu, poincare_distance_single_gpu};
// AP-007: StubVectorOps export is gated to test-only builds
// Allow deprecated usage in tests - the deprecation warning is intentional for production
#[cfg(test)]
#[allow(deprecated)]
pub use stub::StubVectorOps;
