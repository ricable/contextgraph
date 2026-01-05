//! Pre-flight checks and CUDA initialization for warm model loading.

use crate::warm::cuda_alloc::{
    GpuInfo, WarmCudaAllocator, MINIMUM_VRAM_BYTES, REQUIRED_COMPUTE_MAJOR, REQUIRED_COMPUTE_MINOR,
};
use crate::warm::error::WarmResult;

#[cfg(feature = "cuda")]
use crate::warm::config::WarmConfig;
#[cfg(feature = "cuda")]
use crate::warm::error::WarmError;
#[cfg(feature = "cuda")]
use super::constants::{GB, MODEL_SIZES};
#[cfg(feature = "cuda")]
use super::helpers::format_bytes;

/// Run pre-flight checks before loading.
///
/// Verifies:
/// - GPU meets compute capability requirements (12.0+)
/// - Sufficient VRAM available (32GB)
/// - CUDA context is valid
#[allow(unused_variables)]
pub fn run_preflight_checks(
    config: &crate::warm::config::WarmConfig,
    gpu_info: &mut Option<GpuInfo>,
) -> WarmResult<()> {
    tracing::info!("Running pre-flight checks...");

    // Check if CUDA is available
    #[cfg(not(feature = "cuda"))]
    {
        tracing::warn!("CUDA feature not enabled, running in stub mode");
        // In stub mode, we simulate successful checks for testing
        *gpu_info = Some(GpuInfo::new(
            0,
            "Simulated RTX 5090".to_string(),
            (REQUIRED_COMPUTE_MAJOR, REQUIRED_COMPUTE_MINOR),
            MINIMUM_VRAM_BYTES,
            "Simulated".to_string(),
        ));
        Ok(())
    }

    #[cfg(feature = "cuda")]
    {
        // Try to create a temporary allocator to query GPU info
        let allocator = WarmCudaAllocator::new(config.cuda_device_id)?;
        let info = allocator.get_gpu_info()?;

        tracing::info!(
            "GPU detected: {} (CC {}, {} VRAM)",
            info.name,
            info.compute_capability_string(),
            format_bytes(info.total_memory_bytes)
        );

        // Check compute capability
        if !info.meets_compute_requirement(REQUIRED_COMPUTE_MAJOR, REQUIRED_COMPUTE_MINOR) {
            return Err(WarmError::CudaCapabilityInsufficient {
                actual_cc: info.compute_capability_string(),
                required_cc: format!("{}.{}", REQUIRED_COMPUTE_MAJOR, REQUIRED_COMPUTE_MINOR),
                gpu_name: info.name.clone(),
            });
        }

        // Check VRAM
        if info.total_memory_bytes < MINIMUM_VRAM_BYTES {
            let required_gb = MINIMUM_VRAM_BYTES as f64 / GB as f64;
            let available_gb = info.total_memory_bytes as f64 / GB as f64;
            return Err(WarmError::VramInsufficientTotal {
                required_bytes: MINIMUM_VRAM_BYTES,
                available_bytes: info.total_memory_bytes,
                required_gb,
                available_gb,
                model_breakdown: MODEL_SIZES
                    .iter()
                    .map(|(id, size)| (id.to_string(), *size))
                    .collect(),
            });
        }

        *gpu_info = Some(info);
        tracing::info!("Pre-flight checks passed");
        Ok(())
    }
}

/// Initialize the CUDA allocator.
#[allow(unused_variables)]
pub fn initialize_cuda_allocator(
    config: &crate::warm::config::WarmConfig,
) -> WarmResult<Option<WarmCudaAllocator>> {
    tracing::info!(
        "Initializing CUDA allocator for device {}",
        config.cuda_device_id
    );

    #[cfg(not(feature = "cuda"))]
    {
        // In stub mode, we don't have a real allocator
        tracing::warn!("CUDA feature not enabled, skipping allocator initialization");
        Ok(None)
    }

    #[cfg(feature = "cuda")]
    {
        let allocator = WarmCudaAllocator::new(config.cuda_device_id)?;
        tracing::info!("CUDA allocator initialized successfully");
        Ok(Some(allocator))
    }
}
