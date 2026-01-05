//! Storage configuration for graph storage.
//!
//! Provides configuration options for RocksDB storage backend with
//! validation and preset configurations for different workloads.

use crate::error::{GraphError, GraphResult};

/// Configuration for graph storage.
///
/// All parameters are validated before use via `validate()`.
/// Invalid configurations fail fast with `GraphError::InvalidConfig`.
///
/// # Constitution Reference
///
/// - perf.memory.gpu: <24GB (8GB headroom) - storage supports GPU batch loading
/// - perf.memory.graph_cap: >10M nodes
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Block cache size in bytes (default: 512MB).
    /// Shared across all column families for memory efficiency.
    pub block_cache_size: usize,

    /// Enable compression (default: true, uses LZ4).
    /// LZ4 provides fast decompression for GPU batch loading.
    pub enable_compression: bool,

    /// Bloom filter bits per key (default: 10).
    /// Higher values improve read performance at cost of memory.
    pub bloom_filter_bits: i32,

    /// Write buffer size in bytes (default: 64MB).
    /// Larger buffers improve write throughput.
    pub write_buffer_size: usize,

    /// Max write buffers (default: 3).
    /// More buffers allow concurrent writes during flush.
    pub max_write_buffers: i32,

    /// Target file size base in bytes (default: 64MB).
    /// Affects SST file sizes and compaction.
    pub target_file_size_base: u64,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            block_cache_size: 512 * 1024 * 1024, // 512MB
            enable_compression: true,
            bloom_filter_bits: 10,
            write_buffer_size: 64 * 1024 * 1024, // 64MB
            max_write_buffers: 3,
            target_file_size_base: 64 * 1024 * 1024, // 64MB
        }
    }
}

impl StorageConfig {
    /// Create config optimized for read-heavy workloads.
    ///
    /// Best for: inference, search, GPU batch loading
    /// - Larger block cache (1GB)
    /// - Higher bloom filter bits (14)
    #[must_use]
    pub fn read_optimized() -> Self {
        Self {
            block_cache_size: 1024 * 1024 * 1024, // 1GB
            bloom_filter_bits: 14,                // Higher for better read performance
            ..Default::default()
        }
    }

    /// Create config optimized for write-heavy workloads.
    ///
    /// Best for: bulk loading, training data ingestion
    /// - Larger write buffers (128MB)
    /// - More write buffers (5)
    #[must_use]
    pub fn write_optimized() -> Self {
        Self {
            write_buffer_size: 128 * 1024 * 1024, // 128MB
            max_write_buffers: 5,
            ..Default::default()
        }
    }

    /// Validate configuration, returning GraphError if invalid.
    ///
    /// Fails fast with clear error messages per constitution AP-001.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::InvalidConfig` if:
    /// - `block_cache_size` < 1MB
    /// - `bloom_filter_bits` not in 1..=20
    /// - `write_buffer_size` < 1MB
    /// - `max_write_buffers` < 1
    /// - `target_file_size_base` < 1MB
    pub fn validate(&self) -> GraphResult<()> {
        const MIN_SIZE: usize = 1024 * 1024; // 1MB

        if self.block_cache_size < MIN_SIZE {
            return Err(GraphError::InvalidConfig(format!(
                "block_cache_size must be >= 1MB, got {} bytes",
                self.block_cache_size
            )));
        }

        if self.bloom_filter_bits < 1 || self.bloom_filter_bits > 20 {
            return Err(GraphError::InvalidConfig(format!(
                "bloom_filter_bits must be 1..=20, got {}",
                self.bloom_filter_bits
            )));
        }

        if self.write_buffer_size < MIN_SIZE {
            return Err(GraphError::InvalidConfig(format!(
                "write_buffer_size must be >= 1MB, got {} bytes",
                self.write_buffer_size
            )));
        }

        if self.max_write_buffers < 1 {
            return Err(GraphError::InvalidConfig(format!(
                "max_write_buffers must be >= 1, got {}",
                self.max_write_buffers
            )));
        }

        if self.target_file_size_base < MIN_SIZE as u64 {
            return Err(GraphError::InvalidConfig(format!(
                "target_file_size_base must be >= 1MB, got {} bytes",
                self.target_file_size_base
            )));
        }

        Ok(())
    }
}
