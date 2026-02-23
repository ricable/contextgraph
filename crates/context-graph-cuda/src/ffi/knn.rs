//! FFI bindings for GPU k-NN kernel using CUDA Driver API.
//!
//! This provides a FAISS-free GPU k-NN implementation for HDBSCAN.
//! Uses PTX loaded via Driver API to avoid cudart static initialization
//! bugs on WSL2 with CUDA 13.1.
//!
//! # Constitution Compliance
//!
//! - ARCH-GPU-05: k-NN runs on GPU
//! - ARCH-GPU-06: Batch operations preferred
//!
//! # Feature Flags
//!
//! - `cuda`: Enable CUDA GPU support (requires NVIDIA GPU)
//! - Without `cuda`: Stub implementations that return NoDevice errors

use crate::error::{CudaError, CudaResult};

// =============================================================================
// STUB IMPLEMENTATIONS (always available)
// =============================================================================

/// Check if CUDA is available.
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        crate::ffi::knn::cuda_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get the number of CUDA devices.
pub fn cuda_device_count() -> CudaResult<i32> {
    #[cfg(feature = "cuda")]
    {
        crate::ffi::knn::cuda_device_count()
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(CudaError::NoDevice)
    }
}

/// RAII wrapper for GPU memory allocation.
pub struct GpuBuffer {
    #[cfg(feature = "cuda")]
    inner: crate::ffi::knn::GpuBuffer,
    #[cfg(not(feature = "cuda"))]
    _size: usize,
}

impl GpuBuffer {
    /// Allocate GPU memory.
    pub fn new(size: usize) -> CudaResult<Self> {
        #[cfg(feature = "cuda")]
        {
            Ok(Self {
                inner: crate::ffi::knn::GpuBuffer::new(size)?,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = size;
            Err(CudaError::NoDevice)
        }
    }

    /// Copy data from host to device.
    pub fn copy_from_host(&self, src: &[u8]) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            self.inner.copy_from_host(src)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = src;
            Err(CudaError::NoDevice)
        }
    }

    /// Copy data from device to host.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            self.inner.copy_to_host(dst)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = dst;
            Err(CudaError::NoDevice)
        }
    }

    /// Get the device pointer.
    #[cfg(feature = "cuda")]
    pub fn ptr(&self) -> crate::ffi::knn::CUdeviceptr {
        self.inner.ptr()
    }

    /// Get buffer size in bytes.
    pub fn size(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            self.inner.size()
        }
        #[cfg(not(feature = "cuda"))]
        {
            self._size
        }
    }
}

/// Compute core distances on GPU.
pub fn compute_core_distances_gpu(
    vectors: &[f32],
    vector_count: usize,
    dimension: usize,
    k: usize,
) -> CudaResult<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        // Call the internal implementation
        let mut core_distances = vec![0.0f32; vector_count];
        crate::ffi::knn::compute_core_distances_gpu(
            vectors,
            k,
            &mut core_distances,
        )?;
        Ok(core_distances)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (vectors, vector_count, dimension, k);
        Err(CudaError::NoDevice)
    }
}

/// Compute pairwise distances on GPU.
pub fn compute_pairwise_distances_gpu(
    queries: &[f32],
    database: &[f32],
    distances: &mut [f32],
) -> CudaResult<()> {
    #[cfg(feature = "cuda")]
    {
        crate::ffi::knn::compute_pairwise_distances_gpu(queries, database, distances)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (queries, database, distances);
        Err(CudaError::NoDevice)
    }
}
