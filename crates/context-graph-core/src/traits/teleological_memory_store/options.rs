//! Search options for teleological memory queries.
//!
//! # Search Strategies
//!
//! Three search strategies are available:
//!
//! - **E1Only** (default): Uses only E1 Semantic HNSW index. Fastest, backward compatible.
//! - **MultiSpace**: Weighted fusion of semantic embedders (E1, E5, E7, E10).
//!   Temporal embedders (E2-E4) are excluded from scoring per research findings.
//! - **Pipeline**: Full 3-stage retrieval: Recall → Score → Re-rank.
//!
//! # Research References
//!
//! - [Cascading Retrieval](https://www.pinecone.io/blog/cascading-retrieval/) - 48% improvement
//! - [Fusion Analysis](https://dl.acm.org/doi/10.1145/3596512) - Convex combination beats RRF
//! - [ColBERT Late Interaction](https://weaviate.io/blog/late-interaction-overview)

use serde::{Deserialize, Serialize};

use crate::types::fingerprint::SemanticFingerprint;

/// Search strategy for semantic queries.
///
/// Controls how the 13-embedder multi-space index is used for ranking.
///
/// # Key Insight
///
/// Temporal embedders (E2-E4) measure TIME proximity, not TOPIC similarity.
/// They are excluded from similarity scoring and applied as post-retrieval boosts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SearchStrategy {
    /// E1 HNSW only. Backward compatible, fastest.
    #[default]
    E1Only,

    /// Weighted fusion of semantic embedders (E1, E5, E7, E10).
    /// Temporal embedders (E2-E4) have weight 0.0 per AP-71.
    MultiSpace,

    /// Full 3-stage pipeline: Recall → Score → Re-rank.
    /// Stage 1: E13 SPLADE + E1 for broad recall.
    /// Stage 2: Multi-space scoring with semantic embedders.
    /// Stage 3: Optional E12 ColBERT re-ranking.
    Pipeline,
}

/// Score normalization strategy for multi-space fusion.
///
/// Applied before combining scores from multiple embedders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationStrategyOption {
    /// No normalization - use raw similarity scores.
    None,

    /// Min-max normalization to [0, 1] range.
    #[default]
    MinMax,

    /// Z-score normalization (mean=0, std=1), scaled to [0, 1].
    ZScore,

    /// Convex combination (research-backed best practice).
    /// See: https://dl.acm.org/doi/10.1145/3596512
    Convex,
}

/// Search options for teleological memory queries.
///
/// Controls filtering, pagination, and result formatting for
/// semantic and purpose-based searches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeleologicalSearchOptions {
    /// Maximum number of results to return.
    /// Default: 10, Max: 1000
    pub top_k: usize,

    /// Minimum similarity threshold [0.0, 1.0].
    /// Results below this threshold are filtered out.
    /// Default: 0.0 (no filtering)
    pub min_similarity: f32,

    /// Include soft-deleted items in results.
    /// Default: false
    pub include_deleted: bool,

    /// Embedder indices to use for search (0-12).
    /// Empty = use all embedders.
    pub embedder_indices: Vec<usize>,

    /// Optional semantic fingerprint for computing per-embedder scores.
    /// When provided, enables computation of actual cosine similarity scores
    /// for each embedder instead of returning zeros.
    #[serde(skip)]
    pub semantic_query: Option<SemanticFingerprint>,

    /// Whether to include original content text in search results.
    ///
    /// When `true`, the `content` field of `TeleologicalSearchResult` will be
    /// populated with the original text (if available). When `false` (default),
    /// the `content` field will be `None` for better performance.
    ///
    /// Default: `false` (opt-in for performance reasons)
    ///
    /// TASK-CONTENT-005: Added for content hydration in search results.
    #[serde(default)]
    pub include_content: bool,

    // =========================================================================
    // Multi-Space Search Options (TASK-MULTISPACE)
    // =========================================================================

    /// Search strategy: E1Only (default), MultiSpace, or Pipeline.
    ///
    /// - `E1Only`: Backward compatible, uses only E1 Semantic HNSW.
    /// - `MultiSpace`: Weighted fusion of semantic embedders.
    /// - `Pipeline`: Full 3-stage retrieval with optional re-ranking.
    ///
    /// Default: `E1Only` for backward compatibility.
    #[serde(default)]
    pub strategy: SearchStrategy,

    /// Weight profile name for multi-space scoring.
    ///
    /// Available profiles:
    /// - `"semantic_search"`: General queries (E1: 35%, E7: 20%, E5/E10: 15%)
    /// - `"code_search"`: Programming queries (E7: 40%, E1: 20%)
    /// - `"causal_reasoning"`: "Why" questions (E5: 45%, E1: 20%)
    /// - `"fact_checking"`: Entity/fact queries (E11: 40%, E6: 15%)
    /// - `"category_weighted"`: Constitution-compliant category weights
    ///
    /// All profiles have E2-E4 (temporal) = 0.0 per research findings.
    /// Default: `"semantic_search"`.
    #[serde(default)]
    pub weight_profile: Option<String>,

    /// Recency boost factor [0.0, 1.0].
    ///
    /// Applied POST-retrieval as: `final = semantic * (1.0 - boost) + temporal * boost`.
    /// Uses E2 temporal embedding similarity for recency scoring.
    ///
    /// - `0.0`: No recency boost (default)
    /// - `0.5`: Balance semantic and recency
    /// - `1.0`: Strong recency preference
    ///
    /// Per ARCH-14: Temporal is a POST-retrieval boost, not similarity.
    #[serde(default)]
    pub recency_boost: f32,

    /// Enable E12 ColBERT re-ranking (Stage 3 in Pipeline strategy).
    ///
    /// More accurate but slower. Per AP-73: ColBERT is for re-ranking only.
    /// Default: `false`.
    #[serde(default)]
    pub enable_rerank: bool,

    /// Normalization strategy for score fusion.
    ///
    /// Applied before combining scores from multiple embedders.
    /// Default: `MinMax`.
    #[serde(default)]
    pub normalization: NormalizationStrategyOption,
}

impl Default for TeleologicalSearchOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_similarity: 0.0,
            include_deleted: false,
            embedder_indices: Vec::new(),
            semantic_query: None,
            include_content: false, // TASK-CONTENT-005: Opt-in for performance
            // Multi-space options (TASK-MULTISPACE)
            strategy: SearchStrategy::default(),
            weight_profile: None,
            recency_boost: 0.0,
            enable_rerank: false,
            normalization: NormalizationStrategyOption::default(),
        }
    }
}

impl TeleologicalSearchOptions {
    /// Create options for a quick top-k search.
    #[inline]
    pub fn quick(top_k: usize) -> Self {
        Self {
            top_k,
            ..Default::default()
        }
    }

    /// Create options with minimum similarity threshold.
    #[inline]
    pub fn with_min_similarity(mut self, threshold: f32) -> Self {
        self.min_similarity = threshold;
        self
    }

    /// Create options filtering by specific embedders.
    #[inline]
    pub fn with_embedders(mut self, indices: Vec<usize>) -> Self {
        self.embedder_indices = indices;
        self
    }

    /// Attach semantic fingerprint for computing per-embedder similarity scores.
    /// When provided, computes actual cosine similarities between query and
    /// stored semantic fingerprints instead of returning zeros.
    #[inline]
    pub fn with_semantic_query(mut self, semantic: SemanticFingerprint) -> Self {
        self.semantic_query = Some(semantic);
        self
    }

    /// Set whether to include original content text in search results.
    ///
    /// When `true`, content will be fetched and included in results.
    /// Default is `false` for better performance.
    ///
    /// TASK-CONTENT-005: Builder method for content inclusion.
    #[inline]
    pub fn with_include_content(mut self, include: bool) -> Self {
        self.include_content = include;
        self
    }

    // =========================================================================
    // Multi-Space Search Builder Methods (TASK-MULTISPACE)
    // =========================================================================

    /// Set the search strategy.
    ///
    /// # Arguments
    ///
    /// * `strategy` - One of `E1Only`, `MultiSpace`, or `Pipeline`
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::traits::{TeleologicalSearchOptions, SearchStrategy};
    ///
    /// let opts = TeleologicalSearchOptions::quick(10)
    ///     .with_strategy(SearchStrategy::MultiSpace);
    /// ```
    #[inline]
    pub fn with_strategy(mut self, strategy: SearchStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the weight profile for multi-space scoring.
    ///
    /// # Arguments
    ///
    /// * `profile` - Profile name (e.g., "semantic_search", "code_search")
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::traits::TeleologicalSearchOptions;
    ///
    /// let opts = TeleologicalSearchOptions::quick(10)
    ///     .with_weight_profile("code_search");
    /// ```
    #[inline]
    pub fn with_weight_profile(mut self, profile: &str) -> Self {
        self.weight_profile = Some(profile.to_string());
        self
    }

    /// Set the recency boost factor.
    ///
    /// Applied POST-retrieval as: `final = semantic * (1.0 - boost) + temporal * boost`.
    ///
    /// # Arguments
    ///
    /// * `factor` - Boost factor [0.0, 1.0]. Clamped to valid range.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::traits::TeleologicalSearchOptions;
    ///
    /// let opts = TeleologicalSearchOptions::quick(10)
    ///     .with_recency_boost(0.3);
    /// ```
    #[inline]
    pub fn with_recency_boost(mut self, factor: f32) -> Self {
        self.recency_boost = factor.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable E12 ColBERT re-ranking.
    ///
    /// Only effective with `SearchStrategy::Pipeline`.
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable re-ranking
    #[inline]
    pub fn with_rerank(mut self, enable: bool) -> Self {
        self.enable_rerank = enable;
        self
    }

    /// Set the normalization strategy for score fusion.
    ///
    /// # Arguments
    ///
    /// * `strategy` - Normalization strategy
    #[inline]
    pub fn with_normalization(mut self, strategy: NormalizationStrategyOption) -> Self {
        self.normalization = strategy;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_options_default() {
        let opts = TeleologicalSearchOptions::default();
        assert_eq!(opts.top_k, 10);
        assert_eq!(opts.min_similarity, 0.0);
        assert!(!opts.include_deleted);
        assert!(opts.embedder_indices.is_empty());
    }

    #[test]
    fn test_search_options_quick() {
        let opts = TeleologicalSearchOptions::quick(50);
        assert_eq!(opts.top_k, 50);
    }

    #[test]
    fn test_search_options_builder() {
        let opts = TeleologicalSearchOptions::quick(20)
            .with_min_similarity(0.5)
            .with_embedders(vec![0, 1, 2]);

        assert_eq!(opts.top_k, 20);
        assert_eq!(opts.min_similarity, 0.5);
        assert_eq!(opts.embedder_indices, vec![0, 1, 2]);
    }
}
