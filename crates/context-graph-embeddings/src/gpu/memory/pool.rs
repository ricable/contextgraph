//! Thread-safe GPU memory pool for concurrent VRAM access.

use std::sync::{Arc, RwLock};

use super::error::MemoryError;
use super::pressure::MemoryPressure;
use super::stats::MemoryStats;
use super::tracker::VramTracker;

/// Thread-safe GPU memory pool.
///
/// Wraps VramTracker with Arc<RwLock> for concurrent access.
#[derive(Debug, Clone)]
pub struct GpuMemoryPool {
    inner: Arc<RwLock<VramTracker>>,
}

impl GpuMemoryPool {
    /// Create a new GPU memory pool.
    pub fn new(total_bytes: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(VramTracker::new(total_bytes))),
        }
    }

    /// Create pool for RTX 5090 (32GB).
    pub fn rtx_5090() -> Self {
        Self::new(32 * 1024 * 1024 * 1024)
    }

    /// Allocate memory.
    pub fn allocate(&self, name: &str, bytes: usize) -> Result<(), MemoryError> {
        self.inner
            .write()
            .map_err(|_| MemoryError::LockPoisoned)?
            .allocate(name, bytes)
    }

    /// Deallocate memory.
    pub fn deallocate(&self, name: &str) -> usize {
        self.inner
            .write()
            .ok()
            .map(|mut t| t.deallocate(name))
            .unwrap_or(0)
    }

    /// Get current statistics.
    pub fn stats(&self) -> MemoryStats {
        self.inner
            .read()
            .ok()
            .map(|t| t.stats().clone())
            .unwrap_or_default()
    }

    /// Get available memory.
    pub fn available(&self) -> usize {
        self.inner.read().ok().map(|t| t.available()).unwrap_or(0)
    }

    /// Get current memory pressure level.
    ///
    /// Calculated from VRAM utilization:
    /// - Low: <50% utilization
    /// - Medium: 50-80% utilization
    /// - High: 80-95% utilization
    /// - Critical: >95% utilization (triggers eviction)
    pub fn pressure_level(&self) -> MemoryPressure {
        let stats = self.stats();
        let utilization = stats.utilization_percent();
        MemoryPressure::from_utilization(utilization)
    }

    /// Check if eviction should be triggered based on current pressure.
    ///
    /// Returns true if pressure is Critical (>95% utilization).
    pub fn should_evict(&self) -> bool {
        self.pressure_level().should_evict()
    }

    /// Get total capacity in bytes.
    pub fn total(&self) -> usize {
        self.stats().total_bytes
    }

    /// Get allocated bytes.
    pub fn allocated(&self) -> usize {
        self.stats().allocated_bytes
    }
}

impl Default for GpuMemoryPool {
    fn default() -> Self {
        Self::rtx_5090()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_thread_safe() {
        let pool = GpuMemoryPool::new(1000);

        // Clone for concurrent access
        let pool2 = pool.clone();

        assert!(pool.allocate("a", 300).is_ok());
        assert!(pool2.allocate("b", 300).is_ok());
        assert_eq!(pool.available(), 400);
        assert_eq!(pool2.available(), 400);
    }

    #[test]
    fn test_pressure_level_low() {
        let pool = GpuMemoryPool::new(1000);
        pool.allocate("test", 400).unwrap(); // 40%
        assert_eq!(pool.pressure_level(), MemoryPressure::Low);
        assert!(!pool.should_evict());
        println!("[PASS] pressure_level() returns Low at <50% utilization");
    }

    #[test]
    fn test_pressure_level_medium() {
        let pool = GpuMemoryPool::new(1000);
        pool.allocate("test", 600).unwrap(); // 60%
        assert_eq!(pool.pressure_level(), MemoryPressure::Medium);
        assert!(!pool.should_evict());
        println!("[PASS] pressure_level() returns Medium at 50-80% utilization");
    }

    #[test]
    fn test_pressure_level_high() {
        let pool = GpuMemoryPool::new(1000);
        pool.allocate("test", 900).unwrap(); // 90%
        assert_eq!(pool.pressure_level(), MemoryPressure::High);
        assert!(!pool.should_evict());
        println!("[PASS] pressure_level() returns High at 80-95% utilization");
    }

    #[test]
    fn test_pressure_level_critical() {
        let pool = GpuMemoryPool::new(1000);
        pool.allocate("test", 960).unwrap(); // 96%
        assert_eq!(pool.pressure_level(), MemoryPressure::Critical);
        assert!(pool.should_evict());
        println!("[PASS] pressure_level() returns Critical at >95% utilization");
    }

    #[test]
    fn test_total_and_allocated() {
        let pool = GpuMemoryPool::new(1000);
        assert_eq!(pool.total(), 1000);
        assert_eq!(pool.allocated(), 0);

        pool.allocate("test", 300).unwrap();
        assert_eq!(pool.allocated(), 300);
        assert_eq!(pool.available(), 700);
        println!("[PASS] total() and allocated() return correct values");
    }

    #[test]
    fn test_empty_pool_pressure() {
        let pool = GpuMemoryPool::new(1000);
        assert_eq!(pool.pressure_level(), MemoryPressure::Low);
        assert!(!pool.should_evict());
        println!("[PASS] Empty pool has Low pressure");
    }
}
