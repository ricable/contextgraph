//! FAISS IVF-PQ GPU index configuration.

use serde::{Deserialize, Serialize};

/// Configuration for FAISS IVF-PQ GPU index.
///
/// Configures the FAISS GPU index for 10M+ vector search with <5ms latency.
///
/// # Performance Targets
/// - 10M vectors, k=10: <5ms latency
/// - 10M vectors, k=100: <10ms latency
/// - Memory: ~8GB VRAM for 10M 1536D vectors with PQ64x8
///
/// # Constitution Reference
/// - perf.latency.faiss_1M_k100: <2ms
/// - stack.deps: faiss@0.12+gpu
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexConfig {
    /// Vector dimension (must match embedding dimension).
    /// Default: 1536 per constitution embeddings.models.E7_Code
    pub dimension: usize,

    /// Number of inverted lists (clusters).
    /// Default: 16384 = 4 * sqrt(10M) for optimal recall/speed tradeoff
    pub nlist: usize,

    /// Number of clusters to probe during search.
    /// Default: 128 balances accuracy vs search time
    pub nprobe: usize,

    /// Number of product quantization segments.
    /// Must evenly divide dimension. Default: 64 (1536/64 = 24 bytes per segment)
    pub pq_segments: usize,

    /// Bits per quantization code.
    /// Valid values: 4, 8, 12, 16. Default: 8
    pub pq_bits: u8,

    /// GPU device ID.
    /// Default: 0 (primary GPU)
    pub gpu_id: i32,

    /// Use float16 for reduced memory.
    /// Default: true (halves VRAM usage)
    pub use_float16: bool,

    /// Minimum vectors required for training (256 * nlist).
    /// Default: 4,194,304 (256 * 16384)
    pub min_train_vectors: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            dimension: 1536,
            nlist: 16384,
            nprobe: 128,
            pq_segments: 64,
            pq_bits: 8,
            gpu_id: 0,
            use_float16: true,
            min_train_vectors: 4_194_304, // 256 * 16384
        }
    }
}

impl IndexConfig {
    /// Generate FAISS factory string for index creation.
    ///
    /// Returns format: "IVF{nlist},PQ{pq_segments}x{pq_bits}"
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::IndexConfig;
    /// let config = IndexConfig::default();
    /// assert_eq!(config.factory_string(), "IVF16384,PQ64x8");
    /// ```
    pub fn factory_string(&self) -> String {
        format!("IVF{},PQ{}x{}", self.nlist, self.pq_segments, self.pq_bits)
    }

    /// Calculate minimum training vectors based on nlist.
    ///
    /// FAISS requires at least 256 vectors per cluster for quality training.
    ///
    /// # Returns
    /// 256 * nlist
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::IndexConfig;
    /// let config = IndexConfig::default();
    /// assert_eq!(config.calculate_min_train_vectors(), 4_194_304);
    /// ```
    pub fn calculate_min_train_vectors(&self) -> usize {
        256 * self.nlist
    }
}
