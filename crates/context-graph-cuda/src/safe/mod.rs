//! Safe RAII wrappers for CUDA resources.
//!
//! This module provides memory-safe wrappers with automatic cleanup via Drop.
//!
//! # Constitution Compliance
//!
//! - ARCH-06: CUDA FFI only in context-graph-cuda
//! - AP-14: No .unwrap() - all errors propagated via Result

#[cfg(feature = "cuda")]
pub mod device;

// Stub GpuDevice for non-cuda builds
#[cfg(not(feature = "cuda"))]
pub mod device {
    use crate::error::{CudaError, CudaResult};

    /// Stub GpuDevice for non-CUDA builds
    #[derive(Debug)]
    pub struct GpuDevice;

    impl GpuDevice {
        pub fn new(_device_id: usize) -> CudaResult<Self> {
            Err(CudaError::NoDevice)
        }

        pub fn memory_usage_percent(&self) -> f32 {
            0.0
        }
    }
}

#[cfg(feature = "cuda")]
pub use device::GpuDevice;

#[cfg(not(feature = "cuda"))]
pub use device::GpuDevice;

/// Get current GPU memory usage as a percentage (0.0 - 1.0).
///
/// This is a convenience function that creates a temporary GpuDevice to query
/// memory info. For frequent polling, consider maintaining a GpuDevice instance.
///
/// # Returns
///
/// - `Ok(usage)` - Memory usage as a value between 0.0 and 1.0
/// - `Err(_)` - If GPU device cannot be accessed
///
/// # Example
///
/// ```ignore
/// let usage = gpu_memory_usage_percent()?;
/// println!("GPU memory usage: {:.1}%", usage * 100.0);
/// ```
#[cfg(feature = "cuda")]
pub fn gpu_memory_usage_percent() -> crate::error::CudaResult<f32> {
    let device = GpuDevice::new(0)?;
    Ok(device.memory_usage_percent())
}

/// Stub for non-cuda builds - always returns an error
#[cfg(not(feature = "cuda"))]
pub fn gpu_memory_usage_percent() -> crate::error::CudaResult<f32> {
    Err(crate::error::CudaError::NoDevice)
}
