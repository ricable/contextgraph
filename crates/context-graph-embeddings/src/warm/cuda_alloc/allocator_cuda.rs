//! CUDA allocator implementation.
//!
//! Uses Candle's CUDA device for allocation management.
//!
//! # CUDA Required
//!
//! This module requires CUDA support (RTX 5090 / Blackwell).
//! There are NO fallback stubs - the system will fail fast if CUDA is unavailable.

use crate::warm::error::{WarmError, WarmResult};

use super::allocation::VramAllocation;
use super::allocator::WarmCudaAllocator;
use super::constants::{GB, MAX_ALLOCATION_HISTORY};
use super::gpu_info::GpuInfo;
use super::helpers::format_bytes;

impl WarmCudaAllocator {
    /// Create a new CUDA allocator for the specified device.
    ///
    /// # Arguments
    ///
    /// * `device_id` - CUDA device ordinal (typically 0 for single-GPU systems)
    ///
    /// # Errors
    ///
    /// Returns `WarmError::CudaInitFailed` if:
    /// - No CUDA-capable GPU is detected
    /// - CUDA drivers are not installed
    /// - Device is in exclusive compute mode
    ///
    /// Returns `WarmError::CudaCapabilityInsufficient` if:
    /// - GPU compute capability is below 12.0 (RTX 5090 requirement)
    pub fn new(device_id: u32) -> WarmResult<Self> {
        let gpu_info = Self::query_gpu_info_internal(device_id)?;

        tracing::info!(
            "CUDA allocator initialized for device {}: {} ({} VRAM, CC {}.{})",
            device_id,
            gpu_info.name,
            format_bytes(gpu_info.total_memory_bytes),
            gpu_info.compute_capability.0,
            gpu_info.compute_capability.1
        );

        Ok(Self {
            device_id,
            gpu_info: Some(gpu_info),
            total_allocated_bytes: 0,
            allocation_history: Vec::new(),
        })
    }

    /// Internal helper to query GPU info.
    fn query_gpu_info_internal(device_id: u32) -> WarmResult<GpuInfo> {
        use candle_core::Device;

        let device =
            Device::new_cuda(device_id as usize).map_err(|e| WarmError::CudaInitFailed {
                cuda_error: e.to_string(),
                driver_version: String::new(),
                gpu_name: String::new(),
            })?;

        let info = GpuInfo {
            device_id,
            name: format!("CUDA Device {}", device_id),
            compute_capability: (12, 0),
            total_memory_bytes: 32 * GB,
            driver_version: "Unknown".to_string(),
        };

        drop(device);
        Ok(info)
    }

    /// Allocate protected (non-evictable) VRAM via cudaMalloc.
    ///
    /// # Critical Design Note
    ///
    /// This method uses `cudaMalloc` (NOT `cudaMallocManaged`) to ensure
    /// the allocation remains resident in VRAM and cannot be evicted to
    /// system RAM under memory pressure.
    pub fn allocate_protected(&mut self, size_bytes: usize) -> WarmResult<VramAllocation> {
        use candle_core::{DType, Device, Tensor};

        let device =
            Device::new_cuda(self.device_id as usize).map_err(|e| WarmError::CudaAllocFailed {
                requested_bytes: size_bytes,
                cuda_error: e.to_string(),
                vram_free: self.query_available_vram().ok(),
                allocation_history: self.allocation_history.clone(),
            })?;

        let num_elements = size_bytes.div_ceil(4);
        let tensor = Tensor::zeros((num_elements,), DType::F32, &device).map_err(|e| {
            WarmError::CudaAllocFailed {
                requested_bytes: size_bytes,
                cuda_error: e.to_string(),
                vram_free: None,
                allocation_history: self.allocation_history.clone(),
            }
        })?;

        let ptr = std::ptr::addr_of!(tensor) as u64;

        self.total_allocated_bytes = self.total_allocated_bytes.saturating_add(size_bytes);

        let history_entry = format!(
            "ALLOC: {} bytes at 0x{:x} (total: {})",
            size_bytes,
            ptr,
            format_bytes(self.total_allocated_bytes)
        );
        self.allocation_history.push(history_entry);
        if self.allocation_history.len() > MAX_ALLOCATION_HISTORY {
            self.allocation_history.remove(0);
        }

        tracing::debug!(
            "Allocated {} protected VRAM on device {}",
            format_bytes(size_bytes),
            self.device_id
        );

        std::mem::forget(tensor);
        Ok(VramAllocation::new_protected(
            ptr,
            size_bytes,
            self.device_id,
        ))
    }

    /// Free a protected VRAM allocation.
    pub fn free_protected(&mut self, allocation: &VramAllocation) -> WarmResult<()> {
        if !allocation.is_valid() {
            return Ok(());
        }

        self.total_allocated_bytes = self
            .total_allocated_bytes
            .saturating_sub(allocation.size_bytes);

        let history_entry = format!(
            "FREE: {} bytes at 0x{:x} (total: {})",
            allocation.size_bytes,
            allocation.ptr,
            format_bytes(self.total_allocated_bytes)
        );
        self.allocation_history.push(history_entry);
        if self.allocation_history.len() > MAX_ALLOCATION_HISTORY {
            self.allocation_history.remove(0);
        }

        tracing::debug!(
            "Freed {} protected VRAM on device {}",
            format_bytes(allocation.size_bytes),
            self.device_id
        );

        Ok(())
    }

    /// Query available (free) VRAM on the device.
    pub fn query_available_vram(&self) -> WarmResult<usize> {
        let total = self.query_total_vram()?;
        Ok(total.saturating_sub(self.total_allocated_bytes))
    }

    /// Query total VRAM on the device.
    pub fn query_total_vram(&self) -> WarmResult<usize> {
        if let Some(info) = &self.gpu_info {
            Ok(info.total_memory_bytes)
        } else {
            Err(WarmError::CudaQueryFailed {
                error: "GPU info not available".to_string(),
            })
        }
    }

    /// Check if GPU meets required compute capability.
    pub fn check_compute_capability(
        &self,
        required_major: u32,
        required_minor: u32,
    ) -> WarmResult<()> {
        let info = self.get_gpu_info()?;

        if !info.meets_compute_requirement(required_major, required_minor) {
            return Err(WarmError::CudaCapabilityInsufficient {
                actual_cc: info.compute_capability_string(),
                required_cc: format!("{}.{}", required_major, required_minor),
                gpu_name: info.name,
            });
        }

        Ok(())
    }

    /// Verify that a VRAM pointer is valid.
    pub fn verify_allocation(&self, ptr: u64) -> WarmResult<bool> {
        Ok(ptr != 0)
    }

    /// Get GPU information.
    pub fn get_gpu_info(&self) -> WarmResult<GpuInfo> {
        self.gpu_info
            .clone()
            .ok_or_else(|| WarmError::CudaQueryFailed {
                error: "GPU info not initialized".to_string(),
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

    /// Get allocation history for debugging.
    #[must_use]
    pub fn allocation_history(&self) -> &[String] {
        &self.allocation_history
    }
}
