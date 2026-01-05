//! VRAM allocation tracking structures.
//!
//! Tracks allocations made via `cudaMalloc` for non-evictable model weights.

use super::constants::GB;

/// Represents a single VRAM allocation with metadata.
///
/// Tracks allocations made via `cudaMalloc` for non-evictable model weights.
/// Each allocation stores the device pointer, size, and protection status.
///
/// # Non-Evictable Guarantee
///
/// Allocations with `is_protected = true` are made via `cudaMalloc` and will
/// NOT be migrated to system RAM under memory pressure. This is critical
/// for inference latency guarantees.
///
/// # Thread Safety
///
/// `VramAllocation` is `Send + Sync` as it only contains primitive data.
/// The actual CUDA memory management is handled by the allocator.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VramAllocation {
    /// Raw CUDA device pointer (from cudaMalloc).
    ///
    /// This is an opaque handle representing the GPU memory address.
    /// Value of 0 indicates an invalid/freed allocation.
    pub ptr: u64,

    /// Size of the allocation in bytes.
    pub size_bytes: usize,

    /// CUDA device ID where this memory is allocated.
    pub device_id: u32,

    /// Whether this allocation is protected from eviction.
    ///
    /// - `true`: Allocated via `cudaMalloc` (non-evictable)
    /// - `false`: Allocated via `cudaMallocManaged` (can be evicted)
    ///
    /// For warm model loading, this should ALWAYS be `true`.
    pub is_protected: bool,
}

impl VramAllocation {
    /// Create a new protected (non-evictable) allocation record.
    #[must_use]
    pub fn new_protected(ptr: u64, size_bytes: usize, device_id: u32) -> Self {
        Self {
            ptr,
            size_bytes,
            device_id,
            is_protected: true,
        }
    }

    /// Create a new unprotected (evictable) allocation record.
    ///
    /// # Warning
    ///
    /// This should NOT be used for model weights in the warm loading system.
    /// Only use for temporary working memory that can tolerate eviction.
    #[must_use]
    pub fn new_evictable(ptr: u64, size_bytes: usize, device_id: u32) -> Self {
        Self {
            ptr,
            size_bytes,
            device_id,
            is_protected: false,
        }
    }

    /// Check if this allocation is valid (non-null pointer).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.ptr != 0
    }

    /// Get size in megabytes.
    #[must_use]
    pub fn size_mb(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get size in gigabytes.
    #[must_use]
    pub fn size_gb(&self) -> f64 {
        self.size_bytes as f64 / GB as f64
    }
}
