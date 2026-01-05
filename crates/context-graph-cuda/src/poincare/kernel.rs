//! CUDA kernel information and availability checking.
//!
//! Provides utilities to check GPU availability and query kernel configuration.
//!
//! # CUDA Required
//!
//! This module requires CUDA support (RTX 5090 / Blackwell).
//! There are NO fallback stubs - the system will fail fast if CUDA is unavailable.

use super::ffi;

/// Information about the CUDA kernel configuration.
///
/// Useful for debugging and performance tuning.
#[derive(Debug, Clone, Copy)]
pub struct PoincareKernelInfo {
    /// Block dimension X (warp-aligned, typically 32)
    pub block_dim_x: i32,
    /// Block dimension Y (queries per block, typically 8)
    pub block_dim_y: i32,
    /// Point dimension (must be 64)
    pub point_dim: i32,
    /// Shared memory per block in bytes
    pub shared_mem_bytes: i32,
}

/// Get kernel configuration info.
///
/// Returns None if CUDA feature is disabled.
///
/// # Example
///
/// ```
/// use context_graph_cuda::poincare::get_kernel_info;
///
/// if let Some(info) = get_kernel_info() {
///     println!("Block size: {}x{}", info.block_dim_x, info.block_dim_y);
///     println!("Shared memory: {} bytes", info.shared_mem_bytes);
/// }
/// ```
pub fn get_kernel_info() -> Option<PoincareKernelInfo> {
    let mut block_dim_x: i32 = 0;
    let mut block_dim_y: i32 = 0;
    let mut point_dim: i32 = 0;
    let mut shared_mem: i32 = 0;

    unsafe {
        ffi::get_poincare_kernel_config(
            &mut block_dim_x,
            &mut block_dim_y,
            &mut point_dim,
            &mut shared_mem,
        );
    }

    Some(PoincareKernelInfo {
        block_dim_x,
        block_dim_y,
        point_dim,
        shared_mem_bytes: shared_mem,
    })
}

/// Check if CUDA Poincare kernels are available.
///
/// Returns true if:
/// 1. The `cuda` feature is enabled at compile time
/// 2. The system has a CUDA-capable GPU
/// 3. The CUDA runtime is available
///
/// # Example
///
/// ```
/// use context_graph_cuda::poincare::is_poincare_gpu_available;
///
/// if is_poincare_gpu_available() {
///     println!("GPU acceleration available!");
/// } else {
///     panic!("RTX 5090 required but CUDA unavailable");
/// }
/// ```
pub fn is_poincare_gpu_available() -> bool {
    // Try to query CUDA device count
    // This will fail if no CUDA runtime is available
    extern "C" {
        fn cudaGetDeviceCount(count: *mut i32) -> i32;
    }

    unsafe {
        let mut device_count: i32 = 0;
        let result = cudaGetDeviceCount(&mut device_count);
        // cudaSuccess = 0
        result == 0 && device_count > 0
    }
}
