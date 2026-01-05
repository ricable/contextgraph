//! HNSW and inverted index configuration for 5-stage retrieval pipeline.
//!
//! # Index Types
//!
//! | Stage | Index | Type | Embedder |
//! |-------|-------|------|----------|
//! | 1 | SPLADE inverted | InvertedIndex | E13 |
//! | 2 | Matryoshka 128D | HNSW | E1 (truncated) |
//! | 3 | Full embeddings | HNSW x 10 | E1-E5, E7-E11 |
//! | 4 | ColBERT MaxSim | Token-level | E12 |
//! | 5 | Purpose vector | HNSW 13D | PurposeVector |
//!
//! # FAIL FAST. NO FALLBACKS.
//!
//! Invalid configurations panic immediately. No silent defaults.
//!
//! # Module Structure
//!
//! - [`hnsw_config`]: HNSW configuration types (`HnswConfig`, `EmbedderIndex`, `DistanceMetric`)
//! - [`metrics`]: Distance computation functions (`compute_distance`, `cosine_similarity`)
//!
//! # Example
//!
//! ```
//! use context_graph_storage::teleological::indexes::{
//!     get_hnsw_config, all_hnsw_configs, EmbedderIndex, DistanceMetric,
//!     compute_distance, recommended_metric,
//! };
//!
//! // Get HNSW config for E1 semantic embedder
//! let config = get_hnsw_config(EmbedderIndex::E1Semantic).unwrap();
//! assert_eq!(config.dimension, 1024);
//!
//! // Get all HNSW configs (12 total)
//! let configs = all_hnsw_configs();
//! assert_eq!(configs.len(), 12);
//!
//! // Compute cosine distance
//! let a = vec![1.0, 0.0, 0.0];
//! let b = vec![0.0, 1.0, 0.0];
//! let dist = compute_distance(&a, &b, DistanceMetric::Cosine);
//! ```

// hnsw_config is now a directory module (hnsw_config/mod.rs)
pub mod hnsw_config;
pub mod metrics;

// Re-export from hnsw_config
pub use hnsw_config::{
    // Enums
    DistanceMetric,
    EmbedderIndex,
    // Structs
    HnswConfig,
    InvertedIndexConfig,
    // Functions
    all_hnsw_configs,
    get_hnsw_config,
    get_inverted_index_config,
};

// Re-export dimension constants
pub use hnsw_config::{
    E10_DIM, E11_DIM, E12_TOKEN_DIM, E13_SPLADE_VOCAB, E1_DIM, E1_MATRYOSHKA_DIM, E2_DIM, E3_DIM,
    E4_DIM, E5_DIM, E6_SPARSE_VOCAB, E7_DIM, E8_DIM, E9_DIM, NUM_EMBEDDERS, PURPOSE_VECTOR_DIM,
};

// Re-export from metrics
pub use metrics::{compute_distance, cosine_similarity, distance_to_similarity, recommended_metric};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports_hnsw_config() {
        println!("=== TEST: Module exports HNSW config types ===");

        // Verify enum exports
        let _ = EmbedderIndex::E1Semantic;
        let _ = DistanceMetric::Cosine;

        // Verify struct constructors
        let config = HnswConfig::matryoshka_128d();
        assert_eq!(config.dimension, 128);

        // Verify function exports
        let configs = all_hnsw_configs();
        assert_eq!(configs.len(), 12);

        println!("RESULT: PASS");
    }

    #[test]
    fn test_module_exports_metrics() {
        println!("=== TEST: Module exports metrics functions ===");

        // Verify function exports
        let metric = recommended_metric(0);
        assert_eq!(metric, DistanceMetric::Cosine);

        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];

        let dist = compute_distance(&a, &b, DistanceMetric::Cosine);
        assert!(dist < 1e-6);

        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);

        let converted = distance_to_similarity(0.0, DistanceMetric::Cosine);
        assert!((converted - 1.0).abs() < 1e-6);

        println!("RESULT: PASS");
    }

    #[test]
    fn test_module_exports_dimension_constants() {
        println!("=== TEST: Module exports dimension constants ===");

        assert_eq!(E1_DIM, 1024);
        assert_eq!(E2_DIM, 512);
        assert_eq!(E3_DIM, 512);
        assert_eq!(E4_DIM, 512);
        assert_eq!(E5_DIM, 768);
        assert_eq!(E6_SPARSE_VOCAB, 30_522);
        assert_eq!(E7_DIM, 256);
        assert_eq!(E8_DIM, 384);
        assert_eq!(E9_DIM, 10_000);
        assert_eq!(E10_DIM, 768);
        assert_eq!(E11_DIM, 384);
        assert_eq!(E12_TOKEN_DIM, 128);
        assert_eq!(E13_SPLADE_VOCAB, 30_522);
        assert_eq!(NUM_EMBEDDERS, 13);
        assert_eq!(E1_MATRYOSHKA_DIM, 128);
        assert_eq!(PURPOSE_VECTOR_DIM, 13);

        println!("RESULT: PASS");
    }

    #[test]
    fn test_all_acceptance_criteria() {
        println!("\n=== TASK-F005 ACCEPTANCE CRITERIA VERIFICATION ===");
        println!();

        // AC1: EmbedderIndex enum: 15 variants
        println!("AC1: EmbedderIndex has 15 variants");
        let hnsw = EmbedderIndex::all_hnsw();
        assert_eq!(hnsw.len(), 12); // 15 - 3 (E6, E12, E13)
        println!("    - 12 HNSW variants verified");
        println!("    - 3 non-HNSW variants (E6, E12, E13)");
        println!("    PASS");

        // AC2: HnswConfig struct
        println!();
        println!("AC2: HnswConfig struct has m, ef_construction, ef_search, metric, dimension");
        let cfg = get_hnsw_config(EmbedderIndex::E1Semantic).unwrap();
        let _ = cfg.m;
        let _ = cfg.ef_construction;
        let _ = cfg.ef_search;
        let _ = cfg.metric;
        let _ = cfg.dimension;
        println!("    PASS");

        // AC3: DistanceMetric enum: 5 variants
        println!();
        println!("AC3: DistanceMetric has 5 variants");
        let _ = DistanceMetric::Cosine;
        let _ = DistanceMetric::DotProduct;
        let _ = DistanceMetric::Euclidean;
        let _ = DistanceMetric::AsymmetricCosine;
        let _ = DistanceMetric::MaxSim;
        println!("    PASS");

        // AC4: InvertedIndexConfig struct
        println!();
        println!("AC4: InvertedIndexConfig has vocab_size, max_nnz, use_bm25");
        let inv_cfg = get_inverted_index_config(EmbedderIndex::E6Sparse).unwrap();
        let _ = inv_cfg.vocab_size;
        let _ = inv_cfg.max_nnz;
        let _ = inv_cfg.use_bm25;
        println!("    PASS");

        // AC5: get_hnsw_config returns None for E6, E12, E13
        println!();
        println!("AC5: get_hnsw_config returns None for E6, E12, E13");
        assert!(get_hnsw_config(EmbedderIndex::E6Sparse).is_none());
        assert!(get_hnsw_config(EmbedderIndex::E12LateInteraction).is_none());
        assert!(get_hnsw_config(EmbedderIndex::E13Splade).is_none());
        println!("    PASS");

        // AC6: all_hnsw_configs returns 12 entries
        println!();
        println!("AC6: all_hnsw_configs returns HashMap with 12 entries");
        let configs = all_hnsw_configs();
        assert_eq!(configs.len(), 12);
        println!("    PASS");

        // AC7: get_inverted_index_config returns Some for E6, E13 only
        println!();
        println!("AC7: get_inverted_index_config returns Some for E6, E13 only");
        assert!(get_inverted_index_config(EmbedderIndex::E6Sparse).is_some());
        assert!(get_inverted_index_config(EmbedderIndex::E13Splade).is_some());
        assert!(get_inverted_index_config(EmbedderIndex::E1Semantic).is_none());
        println!("    PASS");

        // AC8: recommended_metric function
        println!();
        println!("AC8: recommended_metric(embedder_idx) maps 0-12");
        assert_eq!(recommended_metric(0), DistanceMetric::Cosine);
        assert_eq!(recommended_metric(4), DistanceMetric::AsymmetricCosine);
        assert_eq!(recommended_metric(11), DistanceMetric::MaxSim);
        println!("    PASS");

        // AC9: compute_distance function
        println!();
        println!("AC9: compute_distance(a, b, metric)");
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let dist = compute_distance(&a, &b, DistanceMetric::Cosine);
        assert!((dist - 1.0).abs() < 1e-6);
        println!("    PASS");

        // AC10: distance_to_similarity function
        println!();
        println!("AC10: distance_to_similarity(distance, metric)");
        let sim = distance_to_similarity(0.0, DistanceMetric::Cosine);
        assert!((sim - 1.0).abs() < 1e-6);
        println!("    PASS");

        println!();
        println!("=== ALL ACCEPTANCE CRITERIA PASSED ===");
    }
}
