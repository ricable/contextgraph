//! CUDA allocation stub for non-CUDA builds.
//!
//! This module provides stub implementations when the `cuda` feature is not enabled.
//! Metal and CPU builds use standard Candle memory management instead.

use crate::warm::error::{WarmError, WarmResult};

/// VRAM allocation - stub for non-CUDA builds.
///
/// Mirrors the real VramAllocation struct for API compatibility.
#[derive(Debug, Clone)]
pub struct VramAllocation {
    /// Raw CUDA device pointer (from cudaMalloc).
    /// Value of 0 indicates an invalid/freed allocation (stub always returns 0).
    pub ptr: u64,

    /// Size of the allocation in bytes.
    pub size_bytes: usize,

    /// CUDA device ID where this memory is allocated (stub always returns 0).
    pub device_id: u32,
}

impl VramAllocation {
    /// Create a dummy allocation (no-op).
    pub fn new(size_bytes: usize) -> Self {
        Self {
            ptr: 0, // Stub returns null pointer
            size_bytes,
            device_id: 0,
        }
    }

    /// Dummy address (always 0).
    #[allow(dead_code)]
    pub fn as_ptr(&self) -> *const u8 {
        std::ptr::null()
    }

    /// Dummy mutable address (always 0).
    #[allow(dead_code)]
    pub fn as_mut_ptr(&self) -> *mut u8 {
        std::ptr::null_mut()
    }
}

/// Stub CUDA allocator - no-op for non-CUDA builds.
///
/// Mirrors the real WarmCudaAllocator for API compatibility.
#[derive(Debug, Clone)]
pub struct WarmCudaAllocator {
    /// CUDA device ID this allocator is bound to.
    pub(crate) device_id: u32,

    /// Cached GPU information.
    pub(crate) gpu_info: Option<GpuInfo>,

    /// Stub: always returns 0 VRAM available.
    pub(crate) total_allocated_bytes: usize,

    /// Allocation history for debugging (last N allocations).
    /// Stored as strings for compatibility with real implementation.
    pub(crate) allocation_history: Vec<String>,
}

impl WarmCudaAllocator {
    /// Create a stub allocator.
    ///
    /// Returns an error if the device cannot be initialized (stub always succeeds).
    pub fn new(device_id: u32) -> WarmResult<Self> {
        Ok(Self {
            device_id,
            gpu_info: None,
            total_allocated_bytes: 0,
            allocation_history: Vec::new(),
        })
    }

    /// Get GPU information (returns None for stub).
    pub fn get_gpu_info(&self) -> WarmResult<GpuInfo> {
        self.gpu_info.clone().ok_or_else(|| WarmError::CudaUnavailable {
            message: "GPU info not available in stub mode".to_string(),
        })
    }

    /// Get the device ID this allocator is bound to.
    #[must_use]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    /// Get total bytes currently allocated.
    #[must_use]
    pub fn total_allocated(&self) -> usize {
        self.total_allocated_bytes
    }

    /// Dummy allocation (always returns dummy allocation).
    pub fn allocate(&self, size: usize) -> WarmResult<VramAllocation> {
        Ok(VramAllocation::new(size))
    }

    /// Dummy protected allocation (always returns dummy allocation).
    /// In stub mode, this returns a null pointer (0) to ensure
    /// Constitution AP-007 compliance - we don't fake allocations.
    pub fn allocate_protected(&self, size: usize) -> WarmResult<VramAllocation> {
        // Stub returns null pointer - no fake data
        Ok(VramAllocation {
            ptr: 0,
            size_bytes: size,
            device_id: self.device_id,
        })
    }

    /// Dummy deallocation (no-op).
    #[allow(dead_code)]
    pub fn free_protected(&self, _alloc: &VramAllocation) -> WarmResult<()> {
        Ok(())
    }

    /// Stub: returns 0 for available VRAM.
    pub fn query_available_vram(&self) -> WarmResult<usize> {
        Ok(0)
    }

    /// Stub: returns 0 for total VRAM.
    pub fn query_total_vram(&self) -> WarmResult<usize> {
        Ok(0)
    }

    /// Returns empty history (as Vec<String> for compatibility).
    pub fn allocation_history(&self) -> &[String] {
        &self.allocation_history
    }

    /// Stub: returns zero statistics.
    pub fn statistics(&self) -> AllocatorStatistics {
        AllocatorStatistics {
            total_allocations: 0,
            total_deallocations: 0,
            current_allocations: 0,
            peak_allocations: 0,
        }
    }
}

/// Stub statistics.
#[derive(Debug, Clone, Default)]
pub struct AllocatorStatistics {
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub current_allocations: usize,
    pub peak_allocations: usize,
}

/// Constants - dummy values for API compatibility.
pub const FAKE_ALLOCATION_BASE_PATTERN: u8 = 0;
pub const FAKE_ALLOCATION_BASE_MASK: u8 = 0;
pub const GB: usize = 1024 * 1024 * 1024;
pub const GOLDEN_SIMILARITY_THRESHOLD: f32 = 0.0;
pub const MAX_ALLOCATION_HISTORY: usize = 100;
pub const MINIMUM_VRAM_BYTES: usize = 0;
pub const REQUIRED_COMPUTE_MAJOR: u32 = 0;
pub const REQUIRED_COMPUTE_MINOR: u32 = 0;
pub const SIN_WAVE_ENERGY_THRESHOLD: f32 = 0.0;

/// Stub GPU info - matches the CUDA GpuInfo API for compatibility.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// CUDA device ordinal (0-indexed).
    pub device_id: u32,
    /// GPU model name (e.g., "NVIDIA GeForce RTX 5090").
    pub name: String,
    /// Compute capability as (major, minor) version tuple.
    pub compute_capability: (u32, u32),
    /// Total VRAM in bytes.
    pub total_memory_bytes: usize,
    /// CUDA driver version string (e.g., "13.1.0").
    pub driver_version: String,
}

impl GpuInfo {
    /// Create a new stub GpuInfo.
    #[must_use]
    pub fn new(name: String, total_memory_bytes: usize) -> Self {
        Self {
            device_id: 0,
            name,
            compute_capability: (0, 0),
            total_memory_bytes,
            driver_version: "N/A".to_string(),
        }
    }

    /// Returns the compute capability as a string (e.g., "12.0").
    #[must_use]
    pub fn compute_capability_string(&self) -> String {
        format!("{}.{}", self.compute_capability.0, self.compute_capability.1)
    }

    /// Returns true if GPU is available.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.total_memory_bytes > 0
    }

    /// Check if the GPU meets the minimum compute requirement.
    #[must_use]
    pub fn meets_compute_requirement(&self, major: u32, minor: u32) -> bool {
        self.compute_capability.0 > major
            || (self.compute_capability.0 == major && self.compute_capability.1 >= minor)
    }

    /// Check if this GPU meets RTX 5090 requirements (stub always returns false for non-CUDA).
    #[must_use]
    pub fn meets_rtx_5090_requirements(&self) -> bool {
        false // Stub - not applicable for non-CUDA builds
    }
}

/// Format bytes to human-readable string (stub implementation).
#[allow(dead_code)]
pub fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}
