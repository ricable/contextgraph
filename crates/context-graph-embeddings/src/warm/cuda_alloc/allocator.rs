//! CUDA memory allocator struct definition.
//!
//! # CUDA Required
//!
//! CUDA is REQUIRED for this module (RTX 5090 / Blackwell).
//! There are NO fallback stubs - the system will fail fast if CUDA is unavailable.

use super::gpu_info::GpuInfo;

/// CUDA memory allocator for warm model loading.
///
/// Provides non-evictable VRAM allocations via `cudaMalloc` for model weights.
/// This allocator ensures model weights remain resident in VRAM and are never
/// transparently migrated to system RAM.
///
/// # CUDA Required
///
/// CUDA is REQUIRED - no fallback to CPU. The system will fail fast
/// if CUDA is not available (RTX 5090 / Blackwell required).
///
/// # Usage
///
/// ```rust,ignore
/// // Initialize allocator for device 0
/// let allocator = WarmCudaAllocator::new(0)?;
///
/// // Check GPU capabilities
/// allocator.check_compute_capability(12, 0)?;
///
/// // Allocate protected (non-evictable) memory for model weights
/// let allocation = allocator.allocate_protected(800_000_000)?; // 800MB
///
/// // Use the allocation...
///
/// // Free when done (typically at shutdown)
/// allocator.free_protected(&allocation)?;
/// ```
///
/// # Thread Safety
///
/// The allocator is NOT internally synchronized. Wrap in `Arc<Mutex<_>>`
/// for multi-threaded access.
#[derive(Debug)]
#[allow(dead_code)] // Fields used conditionally based on `cuda` feature
pub struct WarmCudaAllocator {
    /// CUDA device ID this allocator is bound to.
    pub(crate) device_id: u32,

    /// Cached GPU information.
    pub(crate) gpu_info: Option<GpuInfo>,

    /// Track total allocated bytes for diagnostics.
    pub(crate) total_allocated_bytes: usize,

    /// Allocation history for debugging (last N allocations).
    pub(crate) allocation_history: Vec<String>,
}
