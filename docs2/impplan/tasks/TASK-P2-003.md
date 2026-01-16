# TASK-P2-003: EmbedderConfig Registry

```xml
<task_spec id="TASK-P2-003" version="1.0">
<metadata>
  <title>EmbedderConfig Registry Implementation</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>16</sequence>
  <phase>2</phase>
  <implements>
    <requirement_ref>REQ-P2-02</requirement_ref>
    <requirement_ref>REQ-P2-05</requirement_ref>
  </implements>
  <depends_on>
    <!-- Foundation type - no dependencies -->
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
</metadata>

<context>
Implements the EmbedderConfig registry that provides static configuration for
all 13 embedders. This includes dimension, distance metric, quantization method,
category classification, topic weight, and other metadata.

The registry uses compile-time constants to ensure configuration is always available
and consistent throughout the system. Each embedder is assigned to one of four
categories (Semantic, Temporal, Relational, Structural) which determines its
topic_weight for weighted similarity calculations.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE2-EMBEDDING-13SPACE.md#static_configuration</file>
</input_context_files>

<prerequisites>
  <check>Embedder enum exists (TASK-P2-001)</check>
  <check>EmbedderCategory enum exists (TASK-P2-003b)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create EmbedderConfig struct with category field
    - Create DistanceMetric enum
    - Create QuantizationConfig enum
    - Create static EMBEDDER_CONFIGS array with category assignments
    - Implement get_config(embedder) function
    - Implement convenience getters (get_dimension, get_distance_metric, get_category, get_topic_weight, etc.)
    - Assign correct category to each of the 13 embedders:
      - Semantic: E1, E5, E6, E7, E10, E12, E13
      - Temporal: E2, E3, E4
      - Relational: E8, E11
      - Structural: E9
  </in_scope>
  <out_of_scope>
    - Runtime configuration changes
    - Model loading (embedder implementation detail)
    - Actual embedding computation
    - EmbedderCategory enum definition (handled in TASK-P2-003b)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/embedding/config.rs">
      #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
      pub enum DistanceMetric {
          Cosine,
          Euclidean,
          Jaccard,
          Hamming,
          MaxSim,
          TransE,
      }

      #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
      pub enum QuantizationConfig {
          PQ8 { num_subvectors: usize, bits_per_code: usize },
          Float8,
          Binary,
          Inverted,
          None,
      }

      #[derive(Debug, Clone, Copy)]
      pub struct EmbedderConfig {
          pub embedder: Embedder,
          pub dimension: usize,
          pub distance_metric: DistanceMetric,
          pub quantization: QuantizationConfig,
          pub category: EmbedderCategory,
          pub is_asymmetric: bool,
          pub is_sparse: bool,
      }

      pub fn get_config(embedder: Embedder) -> &amp;'static EmbedderConfig;
      pub fn get_dimension(embedder: Embedder) -> usize;
      pub fn get_distance_metric(embedder: Embedder) -> DistanceMetric;
      pub fn get_category(embedder: Embedder) -> EmbedderCategory;
      pub fn get_topic_weight(embedder: Embedder) -> f32;
      pub fn is_asymmetric(embedder: Embedder) -> bool;
      pub fn is_sparse(embedder: Embedder) -> bool;
      pub fn is_semantic(embedder: Embedder) -> bool;
      pub fn is_temporal(embedder: Embedder) -> bool;
    </signature>
  </signatures>

  <constraints>
    - All configurations are compile-time constants
    - get_config returns &amp;'static reference (no allocation)
    - Dimensions match technical spec exactly
    - Distance metrics match technical spec exactly
    - Category assignments must be correct for all 13 embedders
    - topic_weight derived from category via EmbedderCategory::topic_weight()
  </constraints>

  <verification>
    - E1 has dimension 1024 and Cosine metric
    - E5 has is_asymmetric = true
    - E6 and E13 have is_sparse = true
    - E9 has Hamming metric
    - All 13 embedders have valid configs
    - Category verification:
      - E1 (Semantic) has category Semantic, topic_weight 1.0
      - E2 (TempRecent) has category Temporal, topic_weight 0.0
      - E3 (TempPeriodic) has category Temporal, topic_weight 0.0
      - E4 (TempPosition) has category Temporal, topic_weight 0.0
      - E5 (Causal) has category Semantic, topic_weight 1.0
      - E6 (Sparse) has category Semantic, topic_weight 1.0
      - E7 (Code) has category Semantic, topic_weight 1.0
      - E8 (Emotional) has category Relational, topic_weight 0.5
      - E9 (HDC) has category Structural, topic_weight 0.5
      - E10 (Multimodal) has category Semantic, topic_weight 1.0
      - E11 (Entity) has category Relational, topic_weight 0.5
      - E12 (LateInteract) has category Semantic, topic_weight 1.0
      - E13 (SPLADE) has category Semantic, topic_weight 1.0
    - is_semantic() returns true for E1, E5, E6, E7, E10, E12, E13
    - is_temporal() returns true for E2, E3, E4
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/embedding/config.rs

use serde::{Serialize, Deserialize};
use super::Embedder;
use super::category::EmbedderCategory;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity (1 - cosine_distance)
    Cosine,
    /// Euclidean (L2) distance
    Euclidean,
    /// Jaccard index for sparse vectors
    Jaccard,
    /// Hamming distance for binary vectors
    Hamming,
    /// MaxSim for late interaction (ColBERT-style)
    MaxSim,
    /// TransE scoring for knowledge graph embeddings
    TransE,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationConfig {
    /// Product Quantization with 8-bit codes
    PQ8 { num_subvectors: usize, bits_per_code: usize },
    /// 8-bit floating point
    Float8,
    /// Binary (1-bit per dimension, already packed)
    Binary,
    /// Inverted index for sparse vectors
    Inverted,
    /// No quantization
    None,
}

#[derive(Debug, Clone, Copy)]
pub struct EmbedderConfig {
    pub embedder: Embedder,
    pub dimension: usize,
    pub distance_metric: DistanceMetric,
    pub quantization: QuantizationConfig,
    pub category: EmbedderCategory,
    pub is_asymmetric: bool,
    pub is_sparse: bool,
}

/// Static configuration for all 13 embedders
pub static EMBEDDER_CONFIGS: [EmbedderConfig; 13] = [
    // E1: Semantic (1024D, Cosine, PQ8) - Category: Semantic
    EmbedderConfig {
        embedder: Embedder::E1Semantic,
        dimension: 1024,
        distance_metric: DistanceMetric::Cosine,
        quantization: QuantizationConfig::PQ8 { num_subvectors: 32, bits_per_code: 8 },
        category: EmbedderCategory::Semantic,
        is_asymmetric: false,
        is_sparse: false,
    },
    // E2: Temporal Recent (512D, Cosine, Float8) - Category: Temporal
    EmbedderConfig {
        embedder: Embedder::E2TempRecent,
        dimension: 512,
        distance_metric: DistanceMetric::Cosine,
        quantization: QuantizationConfig::Float8,
        category: EmbedderCategory::Temporal,
        is_asymmetric: false,
        is_sparse: false,
    },
    // E3: Temporal Periodic (512D, Cosine, Float8) - Category: Temporal
    EmbedderConfig {
        embedder: Embedder::E3TempPeriodic,
        dimension: 512,
        distance_metric: DistanceMetric::Cosine,
        quantization: QuantizationConfig::Float8,
        category: EmbedderCategory::Temporal,
        is_asymmetric: false,
        is_sparse: false,
    },
    // E4: Temporal Position (512D, Cosine, Float8) - Category: Temporal
    EmbedderConfig {
        embedder: Embedder::E4TempPosition,
        dimension: 512,
        distance_metric: DistanceMetric::Cosine,
        quantization: QuantizationConfig::Float8,
        category: EmbedderCategory::Temporal,
        is_asymmetric: false,
        is_sparse: false,
    },
    // E5: Causal (768D, Cosine, PQ8, asymmetric) - Category: Semantic
    EmbedderConfig {
        embedder: Embedder::E5Causal,
        dimension: 768,
        distance_metric: DistanceMetric::Cosine,
        quantization: QuantizationConfig::PQ8 { num_subvectors: 24, bits_per_code: 8 },
        category: EmbedderCategory::Semantic,
        is_asymmetric: true,
        is_sparse: false,
    },
    // E6: Sparse BoW/TF-IDF (~30K, Jaccard, Inverted) - Category: Semantic
    EmbedderConfig {
        embedder: Embedder::E6Sparse,
        dimension: 30000,
        distance_metric: DistanceMetric::Jaccard,
        quantization: QuantizationConfig::Inverted,
        category: EmbedderCategory::Semantic,
        is_asymmetric: false,
        is_sparse: true,
    },
    // E7: Code (1536D, Cosine, PQ8) - Category: Semantic
    EmbedderConfig {
        embedder: Embedder::E7Code,
        dimension: 1536,
        distance_metric: DistanceMetric::Cosine,
        quantization: QuantizationConfig::PQ8 { num_subvectors: 48, bits_per_code: 8 },
        category: EmbedderCategory::Semantic,
        is_asymmetric: false,
        is_sparse: false,
    },
    // E8: Emotional (384D, Cosine, Float8) - Category: Relational
    EmbedderConfig {
        embedder: Embedder::E8Emotional,
        dimension: 384,
        distance_metric: DistanceMetric::Cosine,
        quantization: QuantizationConfig::Float8,
        category: EmbedderCategory::Relational,
        is_asymmetric: false,
        is_sparse: false,
    },
    // E9: HDC (1024 bits, Hamming, Binary) - Category: Structural
    EmbedderConfig {
        embedder: Embedder::E9HDC,
        dimension: 1024,
        distance_metric: DistanceMetric::Hamming,
        quantization: QuantizationConfig::Binary,
        category: EmbedderCategory::Structural,
        is_asymmetric: false,
        is_sparse: false,
    },
    // E10: Multimodal (768D, Cosine, PQ8) - Category: Semantic
    EmbedderConfig {
        embedder: Embedder::E10Multimodal,
        dimension: 768,
        distance_metric: DistanceMetric::Cosine,
        quantization: QuantizationConfig::PQ8 { num_subvectors: 24, bits_per_code: 8 },
        category: EmbedderCategory::Semantic,
        is_asymmetric: false,
        is_sparse: false,
    },
    // E11: Entity (384D, TransE, Float8) - Category: Relational
    EmbedderConfig {
        embedder: Embedder::E11Entity,
        dimension: 384,
        distance_metric: DistanceMetric::TransE,
        quantization: QuantizationConfig::Float8,
        category: EmbedderCategory::Relational,
        is_asymmetric: false,
        is_sparse: false,
    },
    // E12: Late Interaction (128D per token, MaxSim, Float8) - Category: Semantic
    EmbedderConfig {
        embedder: Embedder::E12LateInteract,
        dimension: 128, // Per token
        distance_metric: DistanceMetric::MaxSim,
        quantization: QuantizationConfig::Float8,
        category: EmbedderCategory::Semantic,
        is_asymmetric: false,
        is_sparse: false,
    },
    // E13: SPLADE (~30K, Jaccard, Inverted) - Category: Semantic
    EmbedderConfig {
        embedder: Embedder::E13SPLADE,
        dimension: 30000,
        distance_metric: DistanceMetric::Jaccard,
        quantization: QuantizationConfig::Inverted,
        category: EmbedderCategory::Semantic,
        is_asymmetric: false,
        is_sparse: true,
    },
];

/// Get configuration for a specific embedder
pub fn get_config(embedder: Embedder) -> &amp;'static EmbedderConfig {
    &amp;EMBEDDER_CONFIGS[embedder.index()]
}

/// Get the expected dimension for an embedder
pub fn get_dimension(embedder: Embedder) -> usize {
    get_config(embedder).dimension
}

/// Get the distance metric for an embedder
pub fn get_distance_metric(embedder: Embedder) -> DistanceMetric {
    get_config(embedder).distance_metric
}

/// Check if embedder uses asymmetric similarity
pub fn is_asymmetric(embedder: Embedder) -> bool {
    get_config(embedder).is_asymmetric
}

/// Check if embedder produces sparse vectors
pub fn is_sparse(embedder: Embedder) -> bool {
    get_config(embedder).is_sparse
}

/// Get quantization config for an embedder
pub fn get_quantization(embedder: Embedder) -> QuantizationConfig {
    get_config(embedder).quantization
}

/// Get the category for an embedder
pub fn get_category(embedder: Embedder) -> EmbedderCategory {
    get_config(embedder).category
}

/// Get the topic weight for an embedder (derived from category)
pub fn get_topic_weight(embedder: Embedder) -> f32 {
    get_config(embedder).category.topic_weight()
}

/// Check if embedder is in the Semantic category
pub fn is_semantic(embedder: Embedder) -> bool {
    get_config(embedder).category.is_semantic()
}

/// Check if embedder is in the Temporal category
pub fn is_temporal(embedder: Embedder) -> bool {
    get_config(embedder).category.is_temporal()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_embedders_configured() {
        for embedder in Embedder::all() {
            let config = get_config(embedder);
            assert_eq!(config.embedder, embedder);
            assert!(config.dimension > 0);
        }
    }

    #[test]
    fn test_e1_semantic_config() {
        let config = get_config(Embedder::E1Semantic);
        assert_eq!(config.dimension, 1024);
        assert_eq!(config.distance_metric, DistanceMetric::Cosine);
        assert!(!config.is_asymmetric);
        assert!(!config.is_sparse);
    }

    #[test]
    fn test_e5_causal_asymmetric() {
        assert!(is_asymmetric(Embedder::E5Causal));
        assert!(!is_asymmetric(Embedder::E1Semantic));
    }

    #[test]
    fn test_sparse_embedders() {
        assert!(is_sparse(Embedder::E6Sparse));
        assert!(is_sparse(Embedder::E13SPLADE));
        assert!(!is_sparse(Embedder::E1Semantic));
    }

    #[test]
    fn test_e9_binary_hamming() {
        let config = get_config(Embedder::E9HDC);
        assert_eq!(config.distance_metric, DistanceMetric::Hamming);
        assert_eq!(config.quantization, QuantizationConfig::Binary);
    }

    #[test]
    fn test_e12_maxsim() {
        let config = get_config(Embedder::E12LateInteract);
        assert_eq!(config.distance_metric, DistanceMetric::MaxSim);
        assert_eq!(config.dimension, 128); // Per token
    }

    #[test]
    fn test_category_assignments() {
        // Semantic embedders
        assert_eq!(get_category(Embedder::E1Semantic), EmbedderCategory::Semantic);
        assert_eq!(get_category(Embedder::E5Causal), EmbedderCategory::Semantic);
        assert_eq!(get_category(Embedder::E6Sparse), EmbedderCategory::Semantic);
        assert_eq!(get_category(Embedder::E7Code), EmbedderCategory::Semantic);
        assert_eq!(get_category(Embedder::E10Multimodal), EmbedderCategory::Semantic);
        assert_eq!(get_category(Embedder::E12LateInteract), EmbedderCategory::Semantic);
        assert_eq!(get_category(Embedder::E13SPLADE), EmbedderCategory::Semantic);

        // Temporal embedders
        assert_eq!(get_category(Embedder::E2TempRecent), EmbedderCategory::Temporal);
        assert_eq!(get_category(Embedder::E3TempPeriodic), EmbedderCategory::Temporal);
        assert_eq!(get_category(Embedder::E4TempPosition), EmbedderCategory::Temporal);

        // Relational embedders
        assert_eq!(get_category(Embedder::E8Emotional), EmbedderCategory::Relational);
        assert_eq!(get_category(Embedder::E11Entity), EmbedderCategory::Relational);

        // Structural embedders
        assert_eq!(get_category(Embedder::E9HDC), EmbedderCategory::Structural);
    }

    #[test]
    fn test_topic_weights() {
        // Semantic = 1.0
        assert_eq!(get_topic_weight(Embedder::E1Semantic), 1.0);
        assert_eq!(get_topic_weight(Embedder::E7Code), 1.0);

        // Temporal = 0.0
        assert_eq!(get_topic_weight(Embedder::E2TempRecent), 0.0);
        assert_eq!(get_topic_weight(Embedder::E3TempPeriodic), 0.0);

        // Relational = 0.5
        assert_eq!(get_topic_weight(Embedder::E8Emotional), 0.5);
        assert_eq!(get_topic_weight(Embedder::E11Entity), 0.5);

        // Structural = 0.5
        assert_eq!(get_topic_weight(Embedder::E9HDC), 0.5);
    }

    #[test]
    fn test_is_semantic() {
        assert!(is_semantic(Embedder::E1Semantic));
        assert!(is_semantic(Embedder::E5Causal));
        assert!(is_semantic(Embedder::E7Code));
        assert!(!is_semantic(Embedder::E2TempRecent));
        assert!(!is_semantic(Embedder::E8Emotional));
        assert!(!is_semantic(Embedder::E9HDC));
    }

    #[test]
    fn test_is_temporal() {
        assert!(is_temporal(Embedder::E2TempRecent));
        assert!(is_temporal(Embedder::E3TempPeriodic));
        assert!(is_temporal(Embedder::E4TempPosition));
        assert!(!is_temporal(Embedder::E1Semantic));
        assert!(!is_temporal(Embedder::E8Emotional));
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/embedding/config.rs">EmbedderConfig and registry</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/embedding/mod.rs">Add pub mod config and re-exports</file>
</files_to_modify>

<validation_criteria>
  <criterion>All 13 embedders have configurations</criterion>
  <criterion>Dimensions match technical specification exactly</criterion>
  <criterion>Distance metrics match technical specification exactly</criterion>
  <criterion>E5 is the only asymmetric embedder</criterion>
  <criterion>E6 and E13 are the only sparse embedders</criterion>
  <criterion>get_config returns &amp;'static reference</criterion>
  <criterion>All 13 embedders have correct category assignment</criterion>
  <criterion>Semantic embedders (E1, E5-E7, E10, E12-E13) have topic_weight 1.0</criterion>
  <criterion>Temporal embedders (E2-E4) have topic_weight 0.0</criterion>
  <criterion>Relational embedders (E8, E11) have topic_weight 0.5</criterion>
  <criterion>Structural embedder (E9) has topic_weight 0.5</criterion>
  <criterion>is_semantic() and is_temporal() helper methods work correctly</criterion>
</validation_criteria>

<test_commands>
  <command description="Run config tests">cargo test --package context-graph-core config</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Ensure TASK-P2-003b is complete (EmbedderCategory enum)
- [ ] Create config.rs in embedding directory
- [ ] Implement DistanceMetric enum
- [ ] Implement QuantizationConfig enum
- [ ] Implement EmbedderConfig struct with category field
- [ ] Create static EMBEDDER_CONFIGS array with category assignments
- [ ] Implement get_config and convenience functions
- [ ] Implement get_category() and get_topic_weight() functions
- [ ] Implement is_semantic() and is_temporal() helper functions
- [ ] Write unit tests for all configurations
- [ ] Write unit tests for category assignments (all 13 embedders)
- [ ] Write unit tests for topic_weight values
- [ ] Verify dimensions and metrics match spec
- [ ] Run tests to verify
- [ ] Proceed to TASK-P2-004
