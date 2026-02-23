//! Core GPU device initialization for RTX 5090 acceleration.
//!
//! # GPU-ONLY Architecture
//!
//! This module supports CUDA, Metal, and CPU fallbacks. Platform is detected
//! at runtime based on available hardware.
//!
//! # Requirements
//!
//! - **NVIDIA**: CUDA-capable GPU (target: RTX 5090 / Blackwell GB202)
//! - **Apple**: Metal MPS capable GPU (M1/M2/M3 series)
//! - **Fallback**: CPU if no GPU available
//!
//! # Singleton Pattern
//!
//! The GPU device is initialized once and shared globally. This ensures:
//! - Single GPU context for optimal memory management
//! - Consistent device placement across all operations
//! - Automatic cleanup on process exit

use candle_core::Device;
use std::sync::OnceLock;

use super::utils::query_gpu_info;
use crate::gpu::GpuInfo;

/// Global GPU device singleton.
pub(crate) static GPU_DEVICE: OnceLock<Device> = OnceLock::new();

/// GPU availability flag (cached for fast checks).
pub(crate) static GPU_AVAILABLE: OnceLock<bool> = OnceLock::new();

/// Cached GPU info for runtime queries.
pub(crate) static GPU_INFO: OnceLock<GpuInfo> = OnceLock::new();

/// Initialize result for thread-safe error handling.
pub(crate) static INIT_RESULT: OnceLock<Result<(), String>> = OnceLock::new();

/// Platform type for GPU device selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPlatform {
    /// NVIDIA CUDA GPU
    Cuda,
    /// Apple Metal MPS
    Metal,
    /// CPU fallback
    Cpu,
}

/// Create a new device based on platform availability.
///
/// This function attempts to create a GPU device in the following order:
/// 1. CUDA (NVIDIA GPUs)
/// 2. Metal (Apple Silicon)
/// 3. CPU (fallback)
///
/// # Arguments
///
/// * `device_id` - Device ordinal (typically 0)
///
/// # Returns
///
/// A new `Device` for the best available platform.
pub fn new_device(device_id: usize) -> Device {
    // Try CUDA first (default for NVIDIA)
    #[cfg(any(feature = "cuda", feature = "candle"))]
    {
        match Device::new_cuda(device_id) {
            Ok(device) => {
                tracing::info!("Using CUDA device {}", device_id);
                return device;
            }
            Err(e) => {
                tracing::debug!("CUDA not available: {}", e);
            }
        }
    }

    // Try Metal (Apple Silicon)
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(device_id) {
            Ok(device) => {
                tracing::info!("Using Metal device {}", device_id);
                return device;
            }
            Err(e) => {
                tracing::debug!("Metal not available: {}", e);
            }
        }
    }

    // Fallback to CPU
    tracing::info!("Using CPU device");
    Device::Cpu
}

/// Get the current platform type.
pub fn get_platform() -> GpuPlatform {
    #[cfg(all(feature = "cuda", feature = "candle"))]
    {
        if Device::new_cuda(0).is_ok() {
            return GpuPlatform::Cuda;
        }
    }
    #[cfg(feature = "metal")]
    {
        if Device::new_metal(0).is_ok() {
            return GpuPlatform::Metal;
        }
    }
    GpuPlatform::Cpu
}

/// Initialize the GPU device (call once at startup).
///
/// # GPU Architecture
///
/// This function supports multiple backends:
/// - CUDA: NVIDIA GPUs (RTX 5090 / Blackwell)
/// - Metal: Apple Silicon (M-series)
/// - CPU: Fallback when GPU unavailable
///
/// # Target Hardware
///
/// - Primary target: NVIDIA RTX 5090 (Blackwell GB202, 32GB VRAM)
/// - Apple Silicon: M1/M2/M3 with Metal MPS
/// - Minimum: CPU fallback always available
///
/// # Returns
///
/// Reference to the initialized device (GPU or CPU).
///
/// # Thread Safety
///
/// Safe to call from multiple threads; only the first call initializes.
///
/// # Example
///
/// ```
/// use context_graph_embeddings::gpu::init_gpu;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // GPU is available - RTX 5090 with CUDA 13.1 or Apple Silicon with Metal
///     let device = init_gpu()?;
///     println!("Device initialized: {:?}", device);
///     Ok(())
/// }
/// ```
pub fn init_gpu() -> Result<&'static Device, candle_core::Error> {
    // Check if already initialized
    if let Some(device) = GPU_DEVICE.get() {
        tracing::debug!("Device already initialized, returning cached device");
        return Ok(device);
    }

    // Check if previous initialization failed
    if let Some(Err(msg)) = INIT_RESULT.get() {
        tracing::error!("Device initialization previously failed: {}", msg);
        return Err(candle_core::Error::Msg(msg.clone()));
    }

    // Log initialization attempt with full context
    tracing::info!("=== Device Initialization Starting ===");
    tracing::info!("Backend priority: CUDA -> Metal -> CPU");
    #[cfg(feature = "cuda")]
    tracing::info!("CUDA feature: enabled");
    #[cfg(feature = "metal")]
    tracing::info!("Metal feature: enabled");
    tracing::info!("Attempting device initialization...");

    // Platform-aware device initialization (CUDA -> Metal -> CPU)
    let device = new_device(0);
    init_success(device)
}

/// Handle successful device initialization (GPU or CPU).
fn init_success(device: Device) -> Result<&'static Device, candle_core::Error> {
    // Determine if we got a GPU or CPU before storing
    let is_gpu = !matches!(device, Device::Cpu);

    // Store the device
    let _ = GPU_DEVICE.set(device);
    let _ = GPU_AVAILABLE.set(is_gpu);

    // Get device reference after storing
    let device_ref = GPU_DEVICE.get().unwrap();

    // Cache device info (works for CUDA; Metal/CPU get appropriate info)
    let info = query_gpu_info(device_ref);
    let _ = GPU_INFO.set(info.clone());
    let _ = INIT_RESULT.set(Ok(()));

    // Log success with comprehensive details
    if is_gpu {
        tracing::info!("=== GPU Initialization SUCCESS ===");
        tracing::info!("  Device: {}", info.name);
        #[cfg(feature = "cuda")]
        {
            tracing::info!("  VRAM: {}", super::utils::format_bytes(info.total_vram));
            tracing::info!("  Compute Capability: {}", info.compute_capability);
        }
    } else {
        tracing::info!("=== CPU Initialization SUCCESS (GPU unavailable) ===");
    }
    tracing::info!("  Status: Ready for tensor operations");

    Ok(device_ref)
}

/// Handle GPU initialization failure with CPU fallback.
fn init_failure(e: candle_core::Error) -> Result<&'static Device, candle_core::Error> {
    let msg = e.to_string();

    tracing::warn!("GPU initialization failed: {}", msg);
    tracing::info!("Falling back to CPU...");

    // Initialize CPU device as fallback
    let device = Device::Cpu;
    let _ = GPU_DEVICE.set(device);
    let _ = GPU_AVAILABLE.set(false);

    // Create basic CPU info
    let info = GpuInfo {
        name: "CPU".to_string(),
        total_vram: 0,
        compute_capability: "N/A".to_string(),
        available: false,
    };
    let _ = GPU_INFO.set(info);
    let _ = INIT_RESULT.set(Ok(()));

    tracing::info!("=== Using CPU Fallback ===");
    tracing::info!("  Status: CPU mode - GPU acceleration unavailable");

    Ok(GPU_DEVICE.get().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_gpu_available_before_init() {
        // Before init_gpu is called, GPU should not be available
        // (or it was already initialized by another test)
        let available = GPU_AVAILABLE.get().copied().unwrap_or(false);
        let initialized = GPU_DEVICE.get().is_some();
        // Either not available and not initialized, or both are true
        assert!(available == initialized || !available);
    }

    #[test]
    fn test_init_gpu_succeeds_on_cuda_hardware() {
        let result = init_gpu();
        assert!(result.is_ok(), "GPU init should succeed on CUDA hardware");
        assert!(*GPU_AVAILABLE.get().unwrap_or(&false));
    }
}
