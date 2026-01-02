//! GPU device management for RTX 5090 acceleration.
//!
//! # GPU-ONLY Architecture
//!
//! This module is **strictly GPU-only** with NO CPU fallback. If a CUDA-capable
//! GPU is not available, initialization will fail with a clear error.
//!
//! # Requirements
//!
//! - **Hardware**: NVIDIA CUDA-capable GPU (target: RTX 5090 / Blackwell GB202)
//! - **Driver**: CUDA 13.1+ with compatible NVIDIA drivers
//! - **Memory**: Minimum 16GB VRAM recommended (32GB for RTX 5090)
//!
//! # Singleton Pattern
//!
//! The GPU device is initialized once and shared globally. This ensures:
//! - Single CUDA context for optimal memory management
//! - Consistent device placement across all operations
//! - Automatic cleanup on process exit
//!
//! # Usage
//!
//! ```rust,ignore
//! use context_graph_embeddings::gpu::{init_gpu, device};
//!
//! // Initialize at startup - WILL FAIL if no GPU available
//! init_gpu()?;
//!
//! // Get device for tensor operations
//! let dev = device();
//! let tensor = Tensor::zeros((1024,), DType::F32, dev)?;
//! ```
//!
//! # Error Handling
//!
//! - [`init_gpu`] returns an error if CUDA is unavailable
//! - [`device`] panics if called before initialization
//! - NO silent fallback to CPU - errors are explicit and actionable

use candle_core::{Device, DType};
use std::sync::OnceLock;

use super::GpuInfo;

/// Global GPU device singleton.
static GPU_DEVICE: OnceLock<Device> = OnceLock::new();

/// GPU availability flag (cached for fast checks).
static GPU_AVAILABLE: OnceLock<bool> = OnceLock::new();

/// Cached GPU info for runtime queries.
static GPU_INFO: OnceLock<GpuInfo> = OnceLock::new();

/// Initialize result for thread-safe error handling.
static INIT_RESULT: OnceLock<Result<(), String>> = OnceLock::new();

/// Initialize the GPU device (call once at startup).
///
/// # GPU-Only Requirement
///
/// This function **requires** a CUDA-capable GPU. There is NO CPU fallback.
/// If no GPU is available, this function returns an error with detailed
/// diagnostic information.
///
/// # Target Hardware
///
/// - Primary target: NVIDIA RTX 5090 (Blackwell GB202, 32GB VRAM)
/// - Minimum requirement: Any CUDA-capable GPU with compute capability 6.0+
/// - Required driver: CUDA 13.1+ recommended
///
/// # Returns
///
/// Reference to the initialized GPU device, or error if CUDA unavailable.
///
/// # Errors
///
/// Returns [`candle_core::Error`] if:
/// - No CUDA-capable GPU is detected
/// - CUDA drivers are not installed or incompatible
/// - GPU is in use by another process with exclusive access
/// - Insufficient GPU memory for initialization
///
/// # Thread Safety
///
/// Safe to call from multiple threads; only the first call initializes.
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_embeddings::gpu::init_gpu;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // This WILL FAIL if no GPU is available - no fallback
///     let device = init_gpu()?;
///     println!("GPU initialized: {:?}", device);
///     Ok(())
/// }
/// ```
pub fn init_gpu() -> Result<&'static Device, candle_core::Error> {
    // Check if already initialized
    if let Some(device) = GPU_DEVICE.get() {
        tracing::debug!("GPU already initialized, returning cached device");
        return Ok(device);
    }

    // Check if previous initialization failed
    if let Some(Err(msg)) = INIT_RESULT.get() {
        tracing::error!("GPU initialization previously failed: {}", msg);
        return Err(candle_core::Error::Msg(msg.clone()));
    }

    // Log initialization attempt with full context
    tracing::info!("=== GPU Initialization Starting ===");
    tracing::info!("Target hardware: NVIDIA RTX 5090 / Blackwell GB202");
    tracing::info!("Target CUDA version: 13.1+");
    tracing::info!("Attempting CUDA device 0 initialization...");

    match Device::new_cuda(0) {
        Ok(device) => {
            // Store the device
            let _ = GPU_DEVICE.set(device);
            let _ = GPU_AVAILABLE.set(true);

            // Get device reference after storing
            let device_ref = GPU_DEVICE.get().unwrap();

            // Cache GPU info
            let info = query_gpu_info(device_ref);
            let _ = GPU_INFO.set(info.clone());
            let _ = INIT_RESULT.set(Ok(()));

            // Log success with comprehensive details
            tracing::info!("=== GPU Initialization SUCCESS ===");
            tracing::info!("  Device: {}", info.name);
            tracing::info!("  VRAM: {}", format_bytes(info.total_vram));
            tracing::info!("  Compute Capability: {}", info.compute_capability);
            tracing::info!("  Status: Ready for tensor operations");

            Ok(device_ref)
        }
        Err(e) => {
            let msg = e.to_string();
            let _ = GPU_AVAILABLE.set(false);
            let _ = INIT_RESULT.set(Err(msg.clone()));

            // ROBUST ERROR LOGGING - provide actionable information
            tracing::error!("=== GPU Initialization FAILED ===");
            tracing::error!("Error: {}", msg);
            tracing::error!("");
            tracing::error!("This crate REQUIRES a CUDA-capable GPU. NO CPU FALLBACK.");
            tracing::error!("");
            tracing::error!("Troubleshooting steps:");
            tracing::error!("  1. Verify NVIDIA GPU is present: nvidia-smi");
            tracing::error!("  2. Check CUDA installation: nvcc --version");
            tracing::error!("  3. Verify driver compatibility with CUDA 13.1+");
            tracing::error!("  4. Ensure GPU is not in exclusive compute mode");
            tracing::error!("  5. Check available GPU memory: nvidia-smi --query-gpu=memory.free --format=csv");
            tracing::error!("");
            tracing::error!("Target hardware: RTX 5090 (32GB VRAM, Compute 12.0)");
            tracing::error!("Minimum hardware: Any CUDA GPU with Compute 6.0+");

            Err(e)
        }
    }
}

/// Get the active GPU device.
///
/// # GPU-Only Requirement
///
/// This function returns the initialized CUDA GPU device. There is NO CPU
/// fallback. If the GPU was not initialized or initialization failed, this
/// function will panic.
///
/// # Panics
///
/// Panics if:
/// - [`init_gpu`] was not called first
/// - [`init_gpu`] was called but failed (no GPU available)
///
/// Always call [`init_gpu`] at application startup and handle errors there.
/// Do not catch the panic from this function - fix the initialization instead.
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_embeddings::gpu::{init_gpu, device};
///
/// // MUST initialize first - no fallback
/// init_gpu().expect("GPU required");
///
/// // Now device() is safe to call
/// let dev = device();
/// // Use dev for GPU tensor operations
/// ```
pub fn device() -> &'static Device {
    GPU_DEVICE.get().expect(
        "GPU not initialized - call init_gpu() at startup. \
         This crate requires a CUDA-capable GPU with no CPU fallback."
    )
}

/// Check if GPU is available and initialized.
///
/// # GPU-Only Architecture
///
/// This function checks if the GPU has been successfully initialized.
/// In the GPU-only architecture, this function returning `false` indicates
/// a critical failure state - the crate cannot function without GPU.
///
/// Returns `false` if:
/// - [`init_gpu`] was not called yet
/// - [`init_gpu`] was called but CUDA initialization failed
/// - No CUDA-capable GPU hardware found
/// - CUDA drivers not installed or incompatible
///
/// Returns `true` if:
/// - [`init_gpu`] was called and CUDA device 0 was successfully initialized
///
/// # Note
///
/// This function is primarily for diagnostic purposes. Application code
/// should call [`init_gpu`] and handle the error rather than checking
/// availability first.
pub fn is_gpu_available() -> bool {
    *GPU_AVAILABLE.get().unwrap_or(&false)
}

/// Default dtype for GPU embeddings.
///
/// Returns `F32` for maximum precision, which is optimal for:
/// - Accuracy-critical embedding comparisons
/// - RTX 5090's excellent F32 tensor core performance
///
/// # Alternative DTypes (use with caution)
///
/// - `F16`: Half precision for 2x memory savings (may reduce accuracy)
/// - `BF16`: Brain float for training stability (requires Ampere+)
///
/// # RTX 5090 Optimization
///
/// The RTX 5090 Blackwell architecture has excellent F32 performance,
/// so F32 is preferred over F16 unless memory-constrained.
pub fn default_dtype() -> DType {
    DType::F32
}

/// Get cached GPU information.
///
/// Returns information about the initialized GPU device, including:
/// - Device name (e.g., "NVIDIA GeForce RTX 5090")
/// - Total VRAM in bytes
/// - Compute capability (e.g., "12.0")
/// - Availability status
///
/// # Returns
///
/// Returns cached [`GpuInfo`] if GPU was initialized, or a default
/// "No GPU" info struct if [`init_gpu`] was not called or failed.
///
/// # Note
///
/// The returned info is cached at initialization time and does not
/// reflect real-time VRAM usage. Use `nvidia-smi` for live memory stats.
pub fn get_gpu_info() -> GpuInfo {
    GPU_INFO.get().cloned().unwrap_or_default()
}

/// Require GPU to be available, returning an error if not.
///
/// This is a convenience function that combines [`init_gpu`] with error
/// transformation to return a structured error type.
///
/// # Usage
///
/// ```rust,ignore
/// use context_graph_embeddings::gpu::require_gpu;
/// use context_graph_embeddings::error::EmbeddingError;
///
/// fn run_embeddings() -> Result<(), EmbeddingError> {
///     require_gpu()?;  // Returns EmbeddingError::GpuError if no GPU
///     // ... rest of embedding logic
///     Ok(())
/// }
/// ```
pub fn require_gpu() -> Result<&'static Device, crate::error::EmbeddingError> {
    init_gpu().map_err(|e| crate::error::EmbeddingError::GpuError {
        message: format!(
            "GPU initialization failed: {}. This crate requires a CUDA-capable GPU (target: RTX 5090). \
             No CPU fallback is available.",
            e
        ),
    })
}

/// Query GPU information from the device.
///
/// # Implementation Note
///
/// Candle doesn't expose detailed GPU info via cuDeviceGetAttribute or similar.
/// For RTX 5090 (Blackwell GB202), we use the known specifications.
/// Future versions may query actual hardware via cuda-sys bindings.
fn query_gpu_info(_device: &Device) -> GpuInfo {
    // TODO: When cuda-sys is available, query actual device properties:
    // - cuDeviceGetName
    // - cuDeviceTotalMem
    // - cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR)
    //
    // For now, use RTX 5090 target specifications
    GpuInfo {
        name: "NVIDIA GeForce RTX 5090".to_string(),
        total_vram: 32 * 1024 * 1024 * 1024, // 32GB GDDR7
        compute_capability: "12.0".to_string(), // Blackwell SM_120
        available: true,
    }
}

/// Format bytes as human-readable string.
fn format_bytes(bytes: usize) -> String {
    const GB: usize = 1024 * 1024 * 1024;
    const MB: usize = 1024 * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // UNIT TESTS (No GPU Required)
    // =========================================================================

    #[test]
    fn test_format_bytes_zero() {
        assert_eq!(format_bytes(0), "0 bytes");
    }

    #[test]
    fn test_format_bytes_megabytes() {
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(512 * 1024 * 1024), "512.0 MB");
    }

    #[test]
    fn test_format_bytes_gigabytes() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_bytes(32 * 1024 * 1024 * 1024), "32.0 GB");
    }

    #[test]
    fn test_format_bytes_small() {
        assert_eq!(format_bytes(100), "100 bytes");
        assert_eq!(format_bytes(1023), "1023 bytes");
    }

    #[test]
    fn test_default_dtype_is_f32() {
        // F32 is the optimal dtype for RTX 5090 embeddings
        assert_eq!(default_dtype(), DType::F32);
    }

    #[test]
    fn test_gpu_info_default() {
        // GpuInfo::default() should indicate no GPU
        let info = GpuInfo::default();
        assert_eq!(info.name, "No GPU");
        assert_eq!(info.total_vram, 0);
        assert!(!info.available);
    }

    #[test]
    fn test_is_gpu_available_before_init() {
        // Before init_gpu is called, GPU should not be available
        // (or it was already initialized by another test)
        let available = is_gpu_available();
        let initialized = GPU_DEVICE.get().is_some();
        // Either not available and not initialized, or both are true
        assert!(available == initialized || !available);
    }

    // =========================================================================
    // GPU INTEGRATION TESTS (Require CUDA Hardware)
    // These tests are ignored by default and run with: cargo test -- --ignored
    // =========================================================================

    #[test]
    #[ignore = "Requires CUDA GPU hardware - run with: cargo test -- --ignored"]
    fn test_init_gpu_succeeds_on_cuda_hardware() {
        let result = init_gpu();
        assert!(result.is_ok(), "GPU init should succeed on CUDA hardware");
        assert!(is_gpu_available());
    }

    #[test]
    #[ignore = "Requires CUDA GPU hardware - run with: cargo test -- --ignored"]
    fn test_device_returns_cuda_device_after_init() {
        let _ = init_gpu();
        let dev = device();
        // Device should be CUDA, not CPU
        assert!(dev.is_cuda(), "Device must be CUDA, not CPU");
    }

    #[test]
    #[ignore = "Requires CUDA GPU hardware - run with: cargo test -- --ignored"]
    fn test_gpu_info_after_init() {
        let _ = init_gpu();
        let info = get_gpu_info();
        assert!(info.available);
        assert!(!info.name.is_empty());
        assert!(info.total_vram > 0);
    }

    #[test]
    #[ignore = "Requires CUDA GPU hardware - run with: cargo test -- --ignored"]
    fn test_require_gpu_returns_device() {
        let result = require_gpu();
        assert!(result.is_ok());
        let dev = result.unwrap();
        assert!(dev.is_cuda());
    }
}
