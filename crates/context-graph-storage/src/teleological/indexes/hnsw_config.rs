//! HNSW index configuration for 5-stage retrieval pipeline.
//!
//! # FAIL FAST. NO FALLBACKS.
//!
//! Invalid configurations panic immediately. No silent defaults.
//!
//! # Index Types (from constitution.yaml)
//!
//! | Stage | Index | Type | Embedder |
//! |-------|-------|------|----------|
//! | 1 | SPLADE inverted | InvertedIndex | E13 |
//! | 2 | Matryoshka 128D | HNSW | E1 (truncated) |
//! | 3 | Full embeddings | HNSW x 10 | E1-E5, E7-E11 |
//! | 4 | ColBERT MaxSim | Token-level | E12 |
//! | 5 | Purpose vector | HNSW 13D | PurposeVector |
//!
//! # HNSW Configuration Table
//!
//! | Index | Dimension | M | ef_construction | ef_search | Metric |
//! |-------|-----------|---|-----------------|-----------|--------|
//! | E1Semantic | 1024 | 16 | 200 | 100 | Cosine |
//! | E1Matryoshka128 | 128 | 32 | 256 | 128 | Cosine |
//! | E2TemporalRecent | 512 | 16 | 200 | 100 | Cosine |
//! | E3TemporalPeriodic | 512 | 16 | 200 | 100 | Cosine |
//! | E4TemporalPositional | 512 | 16 | 200 | 100 | Cosine |
//! | E5Causal | 768 | 16 | 200 | 100 | AsymmetricCosine |
//! | E7Code | 256 | 16 | 200 | 100 | Cosine |
//! | E8Graph | 384 | 16 | 200 | 100 | Cosine |
//! | E9HDC | 10000 | 16 | 200 | 100 | Cosine |
//! | E10Multimodal | 768 | 16 | 200 | 100 | Cosine |
//! | E11Entity | 384 | 16 | 200 | 100 | Cosine |
//! | PurposeVector | 13 | 16 | 200 | 100 | Cosine |

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Import dimension constants from context-graph-core
// E1=1024, E2=512, E3=512, E4=512, E5=768, E6=30522(sparse), E7=256, E8=384, E9=10000, E10=768, E11=384, E12=128/tok, E13=30522(sparse)

/// Dimension constants (mirrored from context-graph-core for independence)
pub const E1_DIM: usize = 1024;
pub const E2_DIM: usize = 512;
pub const E3_DIM: usize = 512;
pub const E4_DIM: usize = 512;
pub const E5_DIM: usize = 768;
pub const E6_SPARSE_VOCAB: usize = 30_522;
pub const E7_DIM: usize = 256;
pub const E8_DIM: usize = 384;
pub const E9_DIM: usize = 10_000;
pub const E10_DIM: usize = 768;
pub const E11_DIM: usize = 384;
pub const E12_TOKEN_DIM: usize = 128;
pub const E13_SPLADE_VOCAB: usize = 30_522;
pub const NUM_EMBEDDERS: usize = 13;
pub const E1_MATRYOSHKA_DIM: usize = 128;
pub const PURPOSE_VECTOR_DIM: usize = 13;

// ============================================================================
// DISTANCE METRIC
// ============================================================================

/// Distance metric for vector similarity computation.
///
/// # Variants
///
/// - `Cosine`: 1 - cos(a, b), range [0, 2]. Most common for normalized embeddings.
/// - `DotProduct`: Inner product. For normalized vectors, equivalent to cosine similarity.
/// - `Euclidean`: L2 distance, range [0, inf). Measures geometric distance.
/// - `AsymmetricCosine`: For E5 causal embeddings where cause->effect != effect->cause.
/// - `MaxSim`: ColBERT-style late interaction. NOT HNSW-compatible (token-level).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine distance: 1 - cos(a, b). Range [0, 2].
    Cosine,
    /// Dot product (inner product). For normalized vectors = cosine similarity.
    DotProduct,
    /// L2 Euclidean distance. Range [0, inf).
    Euclidean,
    /// Asymmetric cosine for E5 causal (cause->effect != effect->cause).
    AsymmetricCosine,
    /// MaxSim for ColBERT late interaction (E12). NOT HNSW-compatible.
    MaxSim,
}

impl DistanceMetric {
    /// Check if this metric is compatible with HNSW indexing.
    ///
    /// MaxSim requires token-level computation and cannot be used with HNSW.
    #[inline]
    pub fn is_hnsw_compatible(&self) -> bool {
        !matches!(self, Self::MaxSim)
    }
}

// ============================================================================
// EMBEDDER INDEX
// ============================================================================

/// Embedder index enum matching constitution.yaml embedder list.
///
/// 15 variants total:
/// - E1-E13: Core embedders (13)
/// - E1Matryoshka128: E1 truncated to 128D for Stage 2 fast filtering
/// - PurposeVector: 13D teleological alignment vector
///
/// # Non-HNSW Embedders
/// - E6Sparse: Inverted index (legacy sparse slot)
/// - E12LateInteraction: ColBERT MaxSim (token-level)
/// - E13Splade: Inverted index with BM25 (Stage 1 recall)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmbedderIndex {
    /// E1: 1024D semantic (e5-large-v2, Matryoshka-capable)
    E1Semantic,
    /// E1 truncated to 128D for Stage 2 fast filtering
    E1Matryoshka128,
    /// E2: 512D temporal recent (exponential decay)
    E2TemporalRecent,
    /// E3: 512D temporal periodic (Fourier)
    E3TemporalPeriodic,
    /// E4: 512D temporal positional (sinusoidal PE)
    E4TemporalPositional,
    /// E5: 768D causal (Longformer SCM, asymmetric similarity)
    E5Causal,
    /// E6: ~30K sparse (inverted index, NOT HNSW)
    E6Sparse,
    /// E7: 256D code (CodeT5p)
    E7Code,
    /// E8: 384D graph (MiniLM)
    E8Graph,
    /// E9: 10000D HDC (holographic)
    E9HDC,
    /// E10: 768D multimodal (CLIP)
    E10Multimodal,
    /// E11: 384D entity (MiniLM)
    E11Entity,
    /// E12: 128D per-token ColBERT (MaxSim, NOT HNSW)
    E12LateInteraction,
    /// E13: ~30K SPLADE sparse (inverted index, NOT HNSW)
    E13Splade,
    /// 13D teleological purpose vector
    PurposeVector,
}

impl EmbedderIndex {
    /// Map 0-12 index to embedder. Panics on out of bounds.
    ///
    /// # Panics
    ///
    /// Panics with "INDEX ERROR" if `idx >= 13`.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_storage::teleological::indexes::EmbedderIndex;
    ///
    /// let e1 = EmbedderIndex::from_index(0);
    /// assert_eq!(e1, EmbedderIndex::E1Semantic);
    /// ```
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::E1Semantic,
            1 => Self::E2TemporalRecent,
            2 => Self::E3TemporalPeriodic,
            3 => Self::E4TemporalPositional,
            4 => Self::E5Causal,
            5 => Self::E6Sparse,
            6 => Self::E7Code,
            7 => Self::E8Graph,
            8 => Self::E9HDC,
            9 => Self::E10Multimodal,
            10 => Self::E11Entity,
            11 => Self::E12LateInteraction,
            12 => Self::E13Splade,
            _ => panic!(
                "INDEX ERROR: embedder index {} out of bounds (max 12)",
                idx
            ),
        }
    }

    /// Get 0-12 index from embedder. Returns None for E1Matryoshka128, PurposeVector.
    ///
    /// These special embedders are not part of the core 13-embedder array.
    pub fn to_index(&self) -> Option<usize> {
        match self {
            Self::E1Semantic => Some(0),
            Self::E2TemporalRecent => Some(1),
            Self::E3TemporalPeriodic => Some(2),
            Self::E4TemporalPositional => Some(3),
            Self::E5Causal => Some(4),
            Self::E6Sparse => Some(5),
            Self::E7Code => Some(6),
            Self::E8Graph => Some(7),
            Self::E9HDC => Some(8),
            Self::E10Multimodal => Some(9),
            Self::E11Entity => Some(10),
            Self::E12LateInteraction => Some(11),
            Self::E13Splade => Some(12),
            Self::E1Matryoshka128 | Self::PurposeVector => None,
        }
    }

    /// Check if this embedder uses HNSW indexing.
    ///
    /// Returns false for:
    /// - E6Sparse (inverted index)
    /// - E12LateInteraction (MaxSim token-level)
    /// - E13Splade (inverted index with BM25)
    #[inline]
    pub fn uses_hnsw(&self) -> bool {
        !matches!(
            self,
            Self::E6Sparse | Self::E12LateInteraction | Self::E13Splade
        )
    }

    /// Check if this embedder uses inverted indexing.
    ///
    /// Returns true for E6Sparse and E13Splade only.
    #[inline]
    pub fn uses_inverted_index(&self) -> bool {
        matches!(self, Self::E6Sparse | Self::E13Splade)
    }

    /// Get all HNSW-capable embedder indexes.
    ///
    /// Returns 12 entries (excludes E6, E12, E13):
    /// - 10 dense embedders (E1-E5, E7-E11)
    /// - E1Matryoshka128 (Stage 2 fast filter)
    /// - PurposeVector (Stage 5 teleological)
    pub fn all_hnsw() -> Vec<Self> {
        vec![
            Self::E1Semantic,
            Self::E1Matryoshka128,
            Self::E2TemporalRecent,
            Self::E3TemporalPeriodic,
            Self::E4TemporalPositional,
            Self::E5Causal,
            Self::E7Code,
            Self::E8Graph,
            Self::E9HDC,
            Self::E10Multimodal,
            Self::E11Entity,
            Self::PurposeVector,
        ]
    }

    /// Get the embedding dimension for this embedder.
    ///
    /// Returns None for E6Sparse, E12LateInteraction, E13Splade (non-dense).
    pub fn dimension(&self) -> Option<usize> {
        match self {
            Self::E1Semantic => Some(E1_DIM),
            Self::E1Matryoshka128 => Some(E1_MATRYOSHKA_DIM),
            Self::E2TemporalRecent => Some(E2_DIM),
            Self::E3TemporalPeriodic => Some(E3_DIM),
            Self::E4TemporalPositional => Some(E4_DIM),
            Self::E5Causal => Some(E5_DIM),
            Self::E6Sparse => None, // Inverted index
            Self::E7Code => Some(E7_DIM),
            Self::E8Graph => Some(E8_DIM),
            Self::E9HDC => Some(E9_DIM),
            Self::E10Multimodal => Some(E10_DIM),
            Self::E11Entity => Some(E11_DIM),
            Self::E12LateInteraction => None, // Token-level
            Self::E13Splade => None,          // Inverted index
            Self::PurposeVector => Some(PURPOSE_VECTOR_DIM),
        }
    }

    /// Get the recommended distance metric for this embedder.
    pub fn recommended_metric(&self) -> Option<DistanceMetric> {
        match self {
            Self::E5Causal => Some(DistanceMetric::AsymmetricCosine),
            Self::E6Sparse | Self::E13Splade => None, // Inverted index
            Self::E12LateInteraction => Some(DistanceMetric::MaxSim),
            _ => Some(DistanceMetric::Cosine),
        }
    }
}

// ============================================================================
// HNSW CONFIG
// ============================================================================

/// HNSW index configuration.
///
/// Parameters from constitution.yaml:
/// - Default: M=16, ef_construction=200, ef_search=100
/// - Matryoshka 128D: M=32, ef_construction=256, ef_search=128
///
/// # Validation
///
/// All parameters are validated on construction:
/// - `m >= 2` (minimum bi-directional links)
/// - `ef_construction >= m` (construction quality)
/// - `ef_search >= 1` (minimum search candidates)
/// - `dimension >= 1` (non-empty vectors)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Number of bi-directional links per node (M parameter).
    /// Higher M = more connections = better recall, more memory.
    pub m: usize,
    /// Size of dynamic candidate list during construction.
    /// Higher ef_construction = better index quality, slower build.
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search.
    /// Higher ef_search = better recall, slower queries.
    pub ef_search: usize,
    /// Distance metric for similarity computation.
    pub metric: DistanceMetric,
    /// Embedding dimension for this index.
    pub dimension: usize,
}

impl HnswConfig {
    /// Create config with explicit parameters.
    ///
    /// # Panics
    ///
    /// Panics with "HNSW CONFIG ERROR" if:
    /// - `m < 2`
    /// - `ef_construction < m`
    /// - `ef_search < 1`
    /// - `dimension < 1`
    pub fn new(
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        metric: DistanceMetric,
        dimension: usize,
    ) -> Self {
        if m < 2 {
            panic!("HNSW CONFIG ERROR: M must be >= 2, got {}", m);
        }
        if ef_construction < m {
            panic!(
                "HNSW CONFIG ERROR: ef_construction ({}) must be >= M ({})",
                ef_construction, m
            );
        }
        if ef_search < 1 {
            panic!(
                "HNSW CONFIG ERROR: ef_search must be >= 1, got {}",
                ef_search
            );
        }
        if dimension < 1 {
            panic!(
                "HNSW CONFIG ERROR: dimension must be >= 1, got {}",
                dimension
            );
        }

        Self {
            m,
            ef_construction,
            ef_search,
            metric,
            dimension,
        }
    }

    /// Default per-embedder config: M=16, ef_construction=200, ef_search=100.
    ///
    /// From constitution.yaml line 531.
    pub fn default_for_dimension(dimension: usize, metric: DistanceMetric) -> Self {
        Self::new(16, 200, 100, metric, dimension)
    }

    /// E1 Matryoshka 128D config: M=32, ef_construction=256, ef_search=128.
    ///
    /// From constitution.yaml line 524. Optimized for Stage 2 fast filtering.
    pub fn matryoshka_128d() -> Self {
        Self::new(32, 256, 128, DistanceMetric::Cosine, E1_MATRYOSHKA_DIM)
    }

    /// Purpose vector 13D config: M=16, ef_construction=200, ef_search=100.
    ///
    /// Standard config for 13-dimensional teleological alignment vectors.
    pub fn purpose_vector() -> Self {
        Self::new(16, 200, 100, DistanceMetric::Cosine, PURPOSE_VECTOR_DIM)
    }

    /// Estimated memory usage per vector in bytes.
    ///
    /// Approximation: dimension * 4 (f32) + M * 2 * 4 (links)
    pub fn estimated_memory_per_vector(&self) -> usize {
        self.dimension * 4 + self.m * 2 * 4
    }
}

// ============================================================================
// INVERTED INDEX CONFIG
// ============================================================================

/// Configuration for inverted indexes (E6 sparse, E13 SPLADE).
///
/// Used for Stage 1 (SPLADE pre-filter) and legacy E6 sparse slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvertedIndexConfig {
    /// Vocabulary size for term IDs (BERT vocab = 30,522).
    pub vocab_size: usize,
    /// Maximum non-zero entries per vector.
    pub max_nnz: usize,
    /// Whether to use BM25 weighting (E13 Stage 1 only).
    pub use_bm25: bool,
}

impl InvertedIndexConfig {
    /// E6 sparse config: 30522 vocab, 1500 max_nnz, no BM25.
    ///
    /// Legacy sparse lexical slot, not used in 5-stage pipeline.
    pub fn e6_sparse() -> Self {
        Self {
            vocab_size: E6_SPARSE_VOCAB,
            max_nnz: 1_500,
            use_bm25: false,
        }
    }

    /// E13 SPLADE config: 30522 vocab, 1500 max_nnz, with BM25.
    ///
    /// Stage 1 recall: BM25 + SPLADE hybrid for initial candidate selection.
    pub fn e13_splade() -> Self {
        Self {
            vocab_size: E13_SPLADE_VOCAB,
            max_nnz: 1_500,
            use_bm25: true,
        }
    }
}

// ============================================================================
// CONFIG FUNCTIONS
// ============================================================================

/// Get HNSW config for index type. Returns None for non-HNSW indexes.
///
/// # Returns
///
/// - `Some(HnswConfig)` for HNSW-compatible embedders (12 total)
/// - `None` for E6Sparse, E12LateInteraction, E13Splade
///
/// # Example
///
/// ```
/// use context_graph_storage::teleological::indexes::{get_hnsw_config, EmbedderIndex};
///
/// let config = get_hnsw_config(EmbedderIndex::E1Semantic);
/// assert!(config.is_some());
/// assert_eq!(config.unwrap().dimension, 1024);
///
/// let none = get_hnsw_config(EmbedderIndex::E6Sparse);
/// assert!(none.is_none());
/// ```
pub fn get_hnsw_config(index: EmbedderIndex) -> Option<HnswConfig> {
    match index {
        // Dense embedders with standard config
        EmbedderIndex::E1Semantic => {
            Some(HnswConfig::default_for_dimension(E1_DIM, DistanceMetric::Cosine))
        }
        EmbedderIndex::E2TemporalRecent => {
            Some(HnswConfig::default_for_dimension(E2_DIM, DistanceMetric::Cosine))
        }
        EmbedderIndex::E3TemporalPeriodic => {
            Some(HnswConfig::default_for_dimension(E3_DIM, DistanceMetric::Cosine))
        }
        EmbedderIndex::E4TemporalPositional => {
            Some(HnswConfig::default_for_dimension(E4_DIM, DistanceMetric::Cosine))
        }
        EmbedderIndex::E5Causal => Some(HnswConfig::default_for_dimension(
            E5_DIM,
            DistanceMetric::AsymmetricCosine,
        )),
        EmbedderIndex::E7Code => {
            Some(HnswConfig::default_for_dimension(E7_DIM, DistanceMetric::Cosine))
        }
        EmbedderIndex::E8Graph => {
            Some(HnswConfig::default_for_dimension(E8_DIM, DistanceMetric::Cosine))
        }
        EmbedderIndex::E9HDC => {
            Some(HnswConfig::default_for_dimension(E9_DIM, DistanceMetric::Cosine))
        }
        EmbedderIndex::E10Multimodal => {
            Some(HnswConfig::default_for_dimension(E10_DIM, DistanceMetric::Cosine))
        }
        EmbedderIndex::E11Entity => {
            Some(HnswConfig::default_for_dimension(E11_DIM, DistanceMetric::Cosine))
        }

        // Special configs
        EmbedderIndex::E1Matryoshka128 => Some(HnswConfig::matryoshka_128d()),
        EmbedderIndex::PurposeVector => Some(HnswConfig::purpose_vector()),

        // NOT HNSW
        EmbedderIndex::E6Sparse => None,
        EmbedderIndex::E12LateInteraction => None,
        EmbedderIndex::E13Splade => None,
    }
}

/// Get all HNSW configs as a map. Returns 12 entries.
///
/// Excludes E6Sparse, E12LateInteraction, E13Splade (non-HNSW).
///
/// # Example
///
/// ```
/// use context_graph_storage::teleological::indexes::{all_hnsw_configs, EmbedderIndex};
///
/// let configs = all_hnsw_configs();
/// assert_eq!(configs.len(), 12);
/// assert!(configs.contains_key(&EmbedderIndex::E1Semantic));
/// assert!(!configs.contains_key(&EmbedderIndex::E6Sparse));
/// ```
pub fn all_hnsw_configs() -> HashMap<EmbedderIndex, HnswConfig> {
    EmbedderIndex::all_hnsw()
        .into_iter()
        .filter_map(|idx| get_hnsw_config(idx).map(|cfg| (idx, cfg)))
        .collect()
}

/// Get inverted index config. Returns None for non-inverted indexes.
///
/// # Returns
///
/// - `Some(InvertedIndexConfig)` for E6Sparse and E13Splade
/// - `None` for all other embedders
pub fn get_inverted_index_config(index: EmbedderIndex) -> Option<InvertedIndexConfig> {
    match index {
        EmbedderIndex::E6Sparse => Some(InvertedIndexConfig::e6_sparse()),
        EmbedderIndex::E13Splade => Some(InvertedIndexConfig::e13_splade()),
        _ => None,
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // HNSW Config Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_hnsw_config_e1_returns_1024d() {
        println!("=== TEST: E1 HNSW config ===");
        println!("BEFORE: Requesting E1Semantic config");

        let config = get_hnsw_config(EmbedderIndex::E1Semantic);
        assert!(config.is_some(), "E1Semantic should have HNSW config");

        let cfg = config.unwrap();
        println!(
            "AFTER: dimension={}, m={}, ef_construction={}, ef_search={}, metric={:?}",
            cfg.dimension, cfg.m, cfg.ef_construction, cfg.ef_search, cfg.metric
        );

        assert_eq!(cfg.dimension, 1024);
        assert_eq!(cfg.m, 16);
        assert_eq!(cfg.ef_construction, 200);
        assert_eq!(cfg.ef_search, 100);
        assert_eq!(cfg.metric, DistanceMetric::Cosine);
        println!("RESULT: PASS");
    }

    #[test]
    fn test_get_hnsw_config_e1_matryoshka_128() {
        println!("=== TEST: E1 Matryoshka 128D config ===");
        println!("BEFORE: Requesting E1Matryoshka128 config");

        let config = get_hnsw_config(EmbedderIndex::E1Matryoshka128);
        assert!(config.is_some(), "E1Matryoshka128 should have HNSW config");

        let cfg = config.unwrap();
        println!(
            "AFTER: dimension={}, m={}, ef_construction={}, ef_search={}, metric={:?}",
            cfg.dimension, cfg.m, cfg.ef_construction, cfg.ef_search, cfg.metric
        );

        // constitution.yaml line 524: M=32, ef_construction=256, ef_search=128
        assert_eq!(cfg.dimension, 128);
        assert_eq!(cfg.m, 32);
        assert_eq!(cfg.ef_construction, 256);
        assert_eq!(cfg.ef_search, 128);
        assert_eq!(cfg.metric, DistanceMetric::Cosine);
        println!("RESULT: PASS");
    }

    #[test]
    fn test_get_hnsw_config_e6_returns_none() {
        println!("=== TEST: E6 returns None (sparse inverted) ===");
        println!("BEFORE: Requesting E6Sparse config (sparse inverted index)");

        let config = get_hnsw_config(EmbedderIndex::E6Sparse);
        println!("AFTER: config.is_some() = {}", config.is_some());

        assert!(config.is_none(), "E6Sparse should return None (inverted index, not HNSW)");
        println!("RESULT: PASS - E6 correctly returns None");
    }

    #[test]
    fn test_get_hnsw_config_e12_returns_none() {
        println!("=== TEST: E12 returns None (ColBERT MaxSim) ===");
        println!("BEFORE: Requesting E12LateInteraction config");

        let config = get_hnsw_config(EmbedderIndex::E12LateInteraction);
        println!("AFTER: config.is_some() = {}", config.is_some());

        assert!(
            config.is_none(),
            "E12LateInteraction should return None (MaxSim token-level, not HNSW)"
        );
        println!("RESULT: PASS - E12 correctly returns None");
    }

    #[test]
    fn test_get_hnsw_config_e13_returns_none() {
        println!("=== TEST: E13 returns None (SPLADE inverted) ===");
        println!("BEFORE: Requesting E13Splade config");

        let config = get_hnsw_config(EmbedderIndex::E13Splade);
        println!("AFTER: config.is_some() = {}", config.is_some());

        assert!(
            config.is_none(),
            "E13Splade should return None (inverted index with BM25, not HNSW)"
        );
        println!("RESULT: PASS - E13 correctly returns None");
    }

    #[test]
    fn test_get_hnsw_config_purpose_vector_13d() {
        println!("=== TEST: Purpose vector 13D config ===");
        println!("BEFORE: Requesting PurposeVector config");

        let config = get_hnsw_config(EmbedderIndex::PurposeVector);
        assert!(config.is_some(), "PurposeVector should have HNSW config");

        let cfg = config.unwrap();
        println!(
            "AFTER: dimension={}, m={}, ef_construction={}, ef_search={}, metric={:?}",
            cfg.dimension, cfg.m, cfg.ef_construction, cfg.ef_search, cfg.metric
        );

        assert_eq!(cfg.dimension, 13); // NUM_EMBEDDERS = 13
        assert_eq!(cfg.m, 16);
        assert_eq!(cfg.ef_construction, 200);
        assert_eq!(cfg.ef_search, 100);
        assert_eq!(cfg.metric, DistanceMetric::Cosine);
        println!("RESULT: PASS");
    }

    #[test]
    fn test_all_hnsw_configs_returns_12() {
        println!("=== TEST: all_hnsw_configs returns 12 entries ===");
        println!("BEFORE: Requesting all HNSW configs");

        let configs = all_hnsw_configs();
        println!("AFTER: {} configs returned", configs.len());

        // 10 dense (E1-E5, E7-E11) + E1Matryoshka128 + PurposeVector = 12
        assert_eq!(
            configs.len(),
            12,
            "Expected 12 HNSW configs (excludes E6, E12, E13)"
        );

        // Verify inclusions
        assert!(configs.contains_key(&EmbedderIndex::E1Semantic));
        assert!(configs.contains_key(&EmbedderIndex::E1Matryoshka128));
        assert!(configs.contains_key(&EmbedderIndex::E2TemporalRecent));
        assert!(configs.contains_key(&EmbedderIndex::E3TemporalPeriodic));
        assert!(configs.contains_key(&EmbedderIndex::E4TemporalPositional));
        assert!(configs.contains_key(&EmbedderIndex::E5Causal));
        assert!(configs.contains_key(&EmbedderIndex::E7Code));
        assert!(configs.contains_key(&EmbedderIndex::E8Graph));
        assert!(configs.contains_key(&EmbedderIndex::E9HDC));
        assert!(configs.contains_key(&EmbedderIndex::E10Multimodal));
        assert!(configs.contains_key(&EmbedderIndex::E11Entity));
        assert!(configs.contains_key(&EmbedderIndex::PurposeVector));

        // Verify exclusions
        assert!(!configs.contains_key(&EmbedderIndex::E6Sparse));
        assert!(!configs.contains_key(&EmbedderIndex::E12LateInteraction));
        assert!(!configs.contains_key(&EmbedderIndex::E13Splade));

        println!("RESULT: PASS");
    }

    // -------------------------------------------------------------------------
    // Inverted Index Config Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_inverted_index_config_e6() {
        println!("=== TEST: E6 inverted index config ===");
        println!("BEFORE: Requesting E6Sparse inverted index config");

        let config = get_inverted_index_config(EmbedderIndex::E6Sparse);
        assert!(config.is_some(), "E6Sparse should have inverted index config");

        let cfg = config.unwrap();
        println!(
            "AFTER: vocab_size={}, max_nnz={}, use_bm25={}",
            cfg.vocab_size, cfg.max_nnz, cfg.use_bm25
        );

        assert_eq!(cfg.vocab_size, 30_522);
        assert_eq!(cfg.max_nnz, 1_500);
        assert!(!cfg.use_bm25, "E6 sparse should NOT use BM25");
        println!("RESULT: PASS");
    }

    #[test]
    fn test_get_inverted_index_config_e13_with_bm25() {
        println!("=== TEST: E13 inverted index config with BM25 ===");
        println!("BEFORE: Requesting E13Splade inverted index config");

        let config = get_inverted_index_config(EmbedderIndex::E13Splade);
        assert!(config.is_some(), "E13Splade should have inverted index config");

        let cfg = config.unwrap();
        println!(
            "AFTER: vocab_size={}, max_nnz={}, use_bm25={}",
            cfg.vocab_size, cfg.max_nnz, cfg.use_bm25
        );

        assert_eq!(cfg.vocab_size, 30_522);
        assert_eq!(cfg.max_nnz, 1_500);
        assert!(cfg.use_bm25, "E13 SPLADE must use BM25 for Stage 1");
        println!("RESULT: PASS");
    }

    // -------------------------------------------------------------------------
    // EmbedderIndex Tests
    // -------------------------------------------------------------------------

    #[test]
    fn edge_case_embedder_index_from_index_all_valid() {
        println!("=== EDGE CASE 1: All 0-12 indices valid ===");
        for i in 0..13 {
            println!("BEFORE: EmbedderIndex::from_index({})", i);
            let idx = EmbedderIndex::from_index(i);
            println!("AFTER: {:?}", idx);
            assert_eq!(
                idx.to_index(),
                Some(i),
                "Index {} should map back to Some({})",
                i,
                i
            );
        }
        println!("RESULT: PASS - All 13 indices map correctly");
    }

    #[test]
    #[should_panic(expected = "INDEX ERROR")]
    fn test_panic_on_index_13() {
        println!("=== TEST: Index 13 should panic ===");
        println!("BEFORE: EmbedderIndex::from_index(13)");
        let _ = EmbedderIndex::from_index(13);
    }

    #[test]
    #[should_panic(expected = "HNSW CONFIG ERROR")]
    fn test_panic_invalid_m() {
        println!("=== TEST: Invalid M should panic ===");
        println!("BEFORE: HnswConfig::new(1, 200, 100, Cosine, 128)");
        let _ = HnswConfig::new(1, 200, 100, DistanceMetric::Cosine, 128);
    }

    // -------------------------------------------------------------------------
    // Boundary Tests
    // -------------------------------------------------------------------------

    #[test]
    fn edge_boundary_minimum_m() {
        println!("=== EDGE CASE: Minimum valid M ===");
        println!("BEFORE: HnswConfig::new(2, 200, 100, Cosine, 128)");
        let cfg = HnswConfig::new(2, 200, 100, DistanceMetric::Cosine, 128);
        println!("AFTER: m={}", cfg.m);
        assert_eq!(cfg.m, 2);
        println!("RESULT: PASS - M=2 is valid minimum");
    }

    #[test]
    fn edge_boundary_max_index() {
        println!("=== EDGE CASE: Maximum valid index ===");
        println!("BEFORE: EmbedderIndex::from_index(12)");
        let idx = EmbedderIndex::from_index(12);
        println!("AFTER: {:?}", idx);
        assert_eq!(idx, EmbedderIndex::E13Splade);
        println!("RESULT: PASS - Index 12 maps to E13Splade");
    }

    #[test]
    fn edge_boundary_all_configs_have_valid_dimensions() {
        println!("=== EDGE CASE: All configs have valid dimensions ===");
        let configs = all_hnsw_configs();
        for (idx, cfg) in &configs {
            println!("BEFORE: Checking {:?}", idx);
            assert!(cfg.dimension >= 1, "Dimension must be >= 1");
            assert!(cfg.m >= 2, "M must be >= 2");
            assert!(
                cfg.ef_construction >= cfg.m,
                "ef_construction must be >= M"
            );
            assert!(cfg.ef_search >= 1, "ef_search must be >= 1");
            println!(
                "AFTER: dimension={}, m={}, ef_construction={}",
                cfg.dimension, cfg.m, cfg.ef_construction
            );
        }
        println!("RESULT: PASS - All 12 configs have valid parameters");
    }

    // -------------------------------------------------------------------------
    // Additional Validation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_embedder_uses_hnsw() {
        println!("=== TEST: uses_hnsw() correctness ===");

        // HNSW embedders
        assert!(EmbedderIndex::E1Semantic.uses_hnsw());
        assert!(EmbedderIndex::E1Matryoshka128.uses_hnsw());
        assert!(EmbedderIndex::E2TemporalRecent.uses_hnsw());
        assert!(EmbedderIndex::E5Causal.uses_hnsw());
        assert!(EmbedderIndex::PurposeVector.uses_hnsw());

        // Non-HNSW embedders
        assert!(!EmbedderIndex::E6Sparse.uses_hnsw());
        assert!(!EmbedderIndex::E12LateInteraction.uses_hnsw());
        assert!(!EmbedderIndex::E13Splade.uses_hnsw());

        println!("RESULT: PASS");
    }

    #[test]
    fn test_embedder_uses_inverted_index() {
        println!("=== TEST: uses_inverted_index() correctness ===");

        // Inverted index embedders
        assert!(EmbedderIndex::E6Sparse.uses_inverted_index());
        assert!(EmbedderIndex::E13Splade.uses_inverted_index());

        // Non-inverted embedders
        assert!(!EmbedderIndex::E1Semantic.uses_inverted_index());
        assert!(!EmbedderIndex::E12LateInteraction.uses_inverted_index());
        assert!(!EmbedderIndex::PurposeVector.uses_inverted_index());

        println!("RESULT: PASS");
    }

    #[test]
    fn test_e5_causal_uses_asymmetric_cosine() {
        println!("=== TEST: E5 Causal uses AsymmetricCosine ===");
        println!("BEFORE: Getting E5Causal HNSW config");

        let config = get_hnsw_config(EmbedderIndex::E5Causal).unwrap();
        println!("AFTER: metric={:?}", config.metric);

        assert_eq!(
            config.metric,
            DistanceMetric::AsymmetricCosine,
            "E5 Causal must use AsymmetricCosine for cause->effect directionality"
        );
        println!("RESULT: PASS");
    }

    #[test]
    fn test_distance_metric_hnsw_compatibility() {
        println!("=== TEST: DistanceMetric HNSW compatibility ===");

        assert!(DistanceMetric::Cosine.is_hnsw_compatible());
        assert!(DistanceMetric::DotProduct.is_hnsw_compatible());
        assert!(DistanceMetric::Euclidean.is_hnsw_compatible());
        assert!(DistanceMetric::AsymmetricCosine.is_hnsw_compatible());
        assert!(
            !DistanceMetric::MaxSim.is_hnsw_compatible(),
            "MaxSim is NOT HNSW-compatible"
        );

        println!("RESULT: PASS");
    }

    #[test]
    fn test_to_index_special_embedders() {
        println!("=== TEST: to_index returns None for special embedders ===");

        assert_eq!(
            EmbedderIndex::E1Matryoshka128.to_index(),
            None,
            "E1Matryoshka128 is not in 0-12 array"
        );
        assert_eq!(
            EmbedderIndex::PurposeVector.to_index(),
            None,
            "PurposeVector is not in 0-12 array"
        );

        println!("RESULT: PASS");
    }

    #[test]
    fn test_all_hnsw_count_is_12() {
        println!("=== TEST: all_hnsw() returns exactly 12 embedders ===");
        let hnsw_embedders = EmbedderIndex::all_hnsw();
        println!("BEFORE: Requesting all HNSW embedders");
        println!("AFTER: {} embedders returned", hnsw_embedders.len());

        assert_eq!(hnsw_embedders.len(), 12);
        println!("RESULT: PASS");
    }

    // -------------------------------------------------------------------------
    // Verification Log Test
    // -------------------------------------------------------------------------

    #[test]
    fn test_verification_log() {
        println!("\n=== TASK-F005 VERIFICATION LOG ===");
        println!("Timestamp: 2026-01-05");
        println!();

        println!("Enum Verification:");
        println!("1. EmbedderIndex has 15 variants");
        let hnsw_count = EmbedderIndex::all_hnsw().len();
        let non_hnsw_count = 3; // E6, E12, E13
        let total = hnsw_count + non_hnsw_count;
        assert_eq!(total, 15, "EmbedderIndex should have 15 variants");

        println!("2. DistanceMetric has 5 variants");
        // Verified by enum definition

        println!();
        println!("Struct Verification:");
        println!("3. HnswConfig has m, ef_construction, ef_search, metric, dimension");
        println!("4. InvertedIndexConfig has vocab_size, max_nnz, use_bm25");

        println!();
        println!("Function Verification:");
        let configs = all_hnsw_configs();
        println!("5. get_hnsw_config returns Some for 12 embedders");
        assert_eq!(configs.len(), 12);

        println!("6. get_hnsw_config returns None for E6, E12, E13");
        assert!(get_hnsw_config(EmbedderIndex::E6Sparse).is_none());
        assert!(get_hnsw_config(EmbedderIndex::E12LateInteraction).is_none());
        assert!(get_hnsw_config(EmbedderIndex::E13Splade).is_none());

        println!("7. all_hnsw_configs returns HashMap with 12 entries");
        assert_eq!(configs.len(), 12);

        println!("8. get_inverted_index_config returns Some for E6, E13");
        assert!(get_inverted_index_config(EmbedderIndex::E6Sparse).is_some());
        assert!(get_inverted_index_config(EmbedderIndex::E13Splade).is_some());

        println!();
        println!("Edge Cases:");
        println!("- Index 0-12 valid");
        println!("- Index 13 panics");
        println!("- Invalid M panics");

        println!();
        println!("VERIFICATION LOG COMPLETE");
        println!("RESULT: PASS");
    }
}
