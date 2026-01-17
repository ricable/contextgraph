# TASK-P4-003: HDBSCANParams and ClusterSelectionMethod

```xml
<task_spec id="TASK-P4-003" version="2.0">
<metadata>
  <title>HDBSCAN Parameter Types Implementation</title>
  <status>completed</status>
  <completed_date>2026-01-17</completed_date>
  <layer>foundation</layer>
  <sequence>29</sequence>
  <phase>4</phase>
  <implements>
    <requirement_ref>REQ-P4-01</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P3-004</task_ref>
    <task_ref>TASK-P4-001</task_ref>
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
  <last_updated>2025-01-17</last_updated>
</metadata>

<context>
Implements the parameter configuration types for HDBSCAN clustering algorithm.
HDBSCANParams contains min_cluster_size, min_samples, cluster_selection_method,
and distance metric. ClusterSelectionMethod enum defines EOM (Excess of Mass)
and Leaf selection strategies.

Default values match the constitution (clustering.parameters.min_cluster_size: 3).

CRITICAL ARCHITECTURE RULES (from CLAUDE.md constitution):
- ARCH-04: Temporal embedders (E2-E4) NEVER count toward topic detection
- ARCH-09: Topic threshold is weighted_agreement >= 2.5 (not raw space count)
- clustering.parameters.min_cluster_size: 3
- clustering.parameters.silhouette_threshold: 0.3
</context>

<codebase_state>
VERIFIED CURRENT STATE (as of 2025-01-17):

1. EXISTING CLUSTERING MODULE STRUCTURE:
   File: crates/context-graph-core/src/clustering/mod.rs
   Contains:
   - pub mod cluster;
   - pub mod error;
   - pub mod membership;
   - pub mod topic;
   Exports: Cluster, ClusterError, ClusterMembership, Topic, TopicPhase, TopicProfile, TopicStability

2. EMBEDDER ENUM LOCATION AND VARIANTS:
   File: crates/context-graph-core/src/teleological/embedder.rs
   Import: use crate::teleological::Embedder;

   ACTUAL VARIANTS (DO NOT USE WRONG NAMES):
   - Embedder::Semantic (index 0) - NOT "E1Semantic"
   - Embedder::TemporalRecent (index 1)
   - Embedder::TemporalPeriodic (index 2)
   - Embedder::TemporalPositional (index 3)
   - Embedder::Causal (index 4)
   - Embedder::Sparse (index 5) - NOT "E6Sparse"
   - Embedder::Code (index 6) - NOT "E7Code"
   - Embedder::Emotional (index 7)
   - Embedder::Hdc (index 8)
   - Embedder::Multimodal (index 9)
   - Embedder::Entity (index 10)
   - Embedder::LateInteraction (index 11)
   - Embedder::KeywordSplade (index 12) - NOT "E13SPLADE"

3. DISTANCE METRIC LOCATION:
   File: crates/context-graph-core/src/index/config.rs
   Import: use crate::index::config::DistanceMetric;

   VARIANTS:
   - DistanceMetric::Cosine
   - DistanceMetric::DotProduct
   - DistanceMetric::Euclidean
   - DistanceMetric::AsymmetricCosine
   - DistanceMetric::MaxSim
   - DistanceMetric::Jaccard

4. GET_DISTANCE_METRIC HELPER:
   File: crates/context-graph-core/src/embeddings/config.rs
   Import: use crate::embeddings::config::get_distance_metric;
   Signature: pub fn get_distance_metric(embedder: Embedder) -> DistanceMetric

5. CLUSTER ERROR (ALREADY EXISTS):
   File: crates/context-graph-core/src/clustering/error.rs
   Import: use super::error::ClusterError; (from within clustering module)

   InvalidParameter variant exists:
   ClusterError::InvalidParameter { message: String }

   Helper: ClusterError::invalid_parameter(message: impl Into&lt;String&gt;) -> Self
</codebase_state>

<input_context_files>
  <file purpose="data_models" must_read="true">docs2/impplan/technical/TECH-PHASE4-CLUSTERING.md#data_models</file>
  <file purpose="static_config" must_read="true">docs2/impplan/technical/TECH-PHASE4-CLUSTERING.md#static_configuration</file>
  <file purpose="constitution" must_read="true">CLAUDE.md (clustering.parameters section)</file>
  <file purpose="distance_metric" must_read="true">crates/context-graph-core/src/index/config.rs</file>
  <file purpose="embedder_enum" must_read="true">crates/context-graph-core/src/teleological/embedder.rs</file>
  <file purpose="embeddings_config" must_read="true">crates/context-graph-core/src/embeddings/config.rs</file>
  <file purpose="existing_error" must_read="true">crates/context-graph-core/src/clustering/error.rs</file>
  <file purpose="existing_module" must_read="true">crates/context-graph-core/src/clustering/mod.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P3-004 complete (DistanceMetric exists in index/config.rs)</check>
  <check>TASK-P4-001 complete (ClusterMembership, Cluster types exist)</check>
  <check>ClusterError with InvalidParameter variant exists in clustering/error.rs</check>
  <check>Embedder enum exists in teleological/embedder.rs with 13 variants</check>
  <check>get_distance_metric helper exists in embeddings/config.rs</check>
</prerequisites>

<scope>
  <in_scope>
    - Create ClusterSelectionMethod enum (EOM, Leaf) with Default derive
    - Create HDBSCANParams struct with min_cluster_size, min_samples, cluster_selection_method, metric
    - Define default parameters from constitution (min_cluster_size=3)
    - Implement validation logic using existing ClusterError::InvalidParameter
    - Parameter adjustment by embedder type using get_distance_metric
    - Builder pattern with with_* methods
    - Serialize/Deserialize support
  </in_scope>
  <out_of_scope>
    - HDBSCAN algorithm implementation (TASK-P4-005)
    - Actual clustering execution
    - Topic detection logic
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/clustering/hdbscan.rs">
      #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
      pub enum ClusterSelectionMethod {
          #[default]
          EOM,
          Leaf,
      }

      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct HDBSCANParams {
          pub min_cluster_size: usize,
          pub min_samples: usize,
          pub cluster_selection_method: ClusterSelectionMethod,
          pub metric: DistanceMetric,
      }

      impl HDBSCANParams {
          pub fn default_for_space(embedder: Embedder) -> Self;
          pub fn with_min_cluster_size(mut self, size: usize) -> Self;
          pub fn with_min_samples(mut self, samples: usize) -> Self;
          pub fn with_selection_method(mut self, method: ClusterSelectionMethod) -> Self;
          pub fn with_metric(mut self, metric: DistanceMetric) -> Self;
          pub fn validate(&amp;self) -> Result&lt;(), ClusterError&gt;;
          pub fn is_viable_for_size(&amp;self, n_points: usize) -> bool;
      }

      pub fn hdbscan_defaults() -> HDBSCANParams;
    </signature>
  </signatures>

  <constraints>
    - min_cluster_size >= 2 (fail fast if violated)
    - min_samples >= 1 (fail fast if violated)
    - min_samples <= min_cluster_size (fail fast if violated)
    - EOM is default method
    - Distance metric retrieved via get_distance_metric(embedder)
    - All validation errors use ClusterError::invalid_parameter() helper
  </constraints>

  <verification>
    - Default values match constitution (min_cluster_size=3, min_samples=2, EOM)
    - Validation rejects invalid combinations with descriptive error messages
    - Builder pattern works for customization
    - Space-specific defaults use correct distance metrics from embeddings/config.rs
    - All 13 embedder variants handled in default_for_space
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/clustering/hdbscan.rs

//! HDBSCAN clustering parameters.
//!
//! Provides configuration types for HDBSCAN algorithm.
//! Per constitution clustering.parameters.min_cluster_size: 3

use serde::{Deserialize, Serialize};

use crate::embeddings::config::get_distance_metric;
use crate::index::config::DistanceMetric;
use crate::teleological::Embedder;

use super::error::ClusterError;

/// Cluster selection method for HDBSCAN.
///
/// Determines how clusters are extracted from the HDBSCAN hierarchy.
///
/// # Example
///
/// ```
/// use context_graph_core::clustering::hdbscan::ClusterSelectionMethod;
///
/// let method = ClusterSelectionMethod::default();
/// assert_eq!(method, ClusterSelectionMethod::EOM);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ClusterSelectionMethod {
    /// Excess of Mass - default, good general purpose.
    /// Selects clusters based on persistence in the hierarchy.
    #[default]
    EOM,
    /// Leaf clusters only - more granular clustering.
    /// Selects only the leaf nodes of the hierarchy tree.
    Leaf,
}

impl ClusterSelectionMethod {
    /// Get description of this method.
    pub fn description(&amp;self) -> &amp;'static str {
        match self {
            ClusterSelectionMethod::EOM => "Excess of Mass - good general purpose clustering",
            ClusterSelectionMethod::Leaf => "Leaf clusters only - more granular clustering",
        }
    }
}

/// Parameters for HDBSCAN clustering algorithm.
///
/// Per constitution: clustering.parameters.min_cluster_size: 3
///
/// # Example
///
/// ```
/// use context_graph_core::clustering::hdbscan::{HDBSCANParams, hdbscan_defaults};
/// use context_graph_core::teleological::Embedder;
///
/// // Use defaults
/// let params = hdbscan_defaults();
/// assert_eq!(params.min_cluster_size, 3);
///
/// // Or space-specific
/// let semantic_params = HDBSCANParams::default_for_space(Embedder::Semantic);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HDBSCANParams {
    /// Minimum number of points to form a cluster.
    /// Per constitution: 3
    pub min_cluster_size: usize,

    /// Minimum samples for a point to be considered a core point.
    /// Must be <= min_cluster_size.
    pub min_samples: usize,

    /// Method for selecting clusters from hierarchy.
    pub cluster_selection_method: ClusterSelectionMethod,

    /// Distance metric to use.
    /// Retrieved via get_distance_metric(embedder) for space-specific params.
    pub metric: DistanceMetric,
}

impl Default for HDBSCANParams {
    fn default() -> Self {
        Self {
            min_cluster_size: 3, // Per constitution
            min_samples: 2,
            cluster_selection_method: ClusterSelectionMethod::EOM,
            metric: DistanceMetric::Cosine,
        }
    }
}

impl HDBSCANParams {
    /// Create default params for a specific embedding space.
    ///
    /// Distance metric is retrieved from embeddings config.
    /// Sparse spaces (Sparse, KeywordSplade) use larger cluster sizes.
    ///
    /// # Arguments
    ///
    /// * `embedder` - The embedding space to configure for
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::clustering::hdbscan::HDBSCANParams;
    /// use context_graph_core::teleological::Embedder;
    /// use context_graph_core::index::config::DistanceMetric;
    ///
    /// let params = HDBSCANParams::default_for_space(Embedder::Sparse);
    /// assert_eq!(params.metric, DistanceMetric::Jaccard);
    /// assert_eq!(params.min_cluster_size, 5); // Larger for sparse
    /// ```
    pub fn default_for_space(embedder: Embedder) -> Self {
        let metric = get_distance_metric(embedder);

        let (min_cluster, min_samples) = match embedder {
            // Sparse spaces need larger clusters due to high dimensionality
            Embedder::Sparse | Embedder::KeywordSplade => (5, 3),
            // Code clustering - use constitution default
            Embedder::Code => (3, 2),
            // All other spaces use constitution default
            _ => (3, 2),
        };

        Self {
            min_cluster_size: min_cluster,
            min_samples,
            cluster_selection_method: ClusterSelectionMethod::EOM,
            metric,
        }
    }

    /// Set minimum cluster size.
    ///
    /// Value is NOT automatically clamped - use validate() to check.
    #[must_use]
    pub fn with_min_cluster_size(mut self, size: usize) -> Self {
        self.min_cluster_size = size;
        self
    }

    /// Set minimum samples.
    ///
    /// Value is NOT automatically clamped - use validate() to check.
    #[must_use]
    pub fn with_min_samples(mut self, samples: usize) -> Self {
        self.min_samples = samples;
        self
    }

    /// Set cluster selection method.
    #[must_use]
    pub fn with_selection_method(mut self, method: ClusterSelectionMethod) -> Self {
        self.cluster_selection_method = method;
        self
    }

    /// Set distance metric.
    #[must_use]
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Validate parameters.
    ///
    /// Fails fast with descriptive error messages.
    ///
    /// # Errors
    ///
    /// Returns `ClusterError::InvalidParameter` if:
    /// - min_cluster_size < 2
    /// - min_samples < 1
    /// - min_samples > min_cluster_size
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::clustering::hdbscan::HDBSCANParams;
    /// use context_graph_core::index::config::DistanceMetric;
    /// use context_graph_core::clustering::hdbscan::ClusterSelectionMethod;
    ///
    /// let invalid = HDBSCANParams {
    ///     min_cluster_size: 1, // Invalid!
    ///     min_samples: 1,
    ///     cluster_selection_method: ClusterSelectionMethod::EOM,
    ///     metric: DistanceMetric::Cosine,
    /// };
    /// assert!(invalid.validate().is_err());
    /// ```
    pub fn validate(&amp;self) -> Result&lt;(), ClusterError&gt; {
        if self.min_cluster_size < 2 {
            return Err(ClusterError::invalid_parameter(format!(
                "min_cluster_size must be >= 2, got {}. HDBSCAN requires at least 2 points to form a cluster.",
                self.min_cluster_size
            )));
        }

        if self.min_samples < 1 {
            return Err(ClusterError::invalid_parameter(format!(
                "min_samples must be >= 1, got {}. At least 1 sample is required for core point determination.",
                self.min_samples
            )));
        }

        if self.min_samples > self.min_cluster_size {
            return Err(ClusterError::invalid_parameter(format!(
                "min_samples ({}) must be <= min_cluster_size ({}). A core point cannot require more samples than the minimum cluster size.",
                self.min_samples, self.min_cluster_size
            )));
        }

        Ok(())
    }

    /// Check if these params will work for a given data size.
    ///
    /// Returns false if there are fewer points than min_cluster_size.
    #[inline]
    pub fn is_viable_for_size(&amp;self, n_points: usize) -> bool {
        n_points >= self.min_cluster_size
    }
}

/// Get default HDBSCAN parameters.
///
/// Returns params matching constitution defaults:
/// - min_cluster_size: 3
/// - min_samples: 2
/// - cluster_selection_method: EOM
/// - metric: Cosine
pub fn hdbscan_defaults() -> HDBSCANParams {
    HDBSCANParams::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // DEFAULT VALUES TESTS
    // =========================================================================

    #[test]
    fn test_default_params_match_constitution() {
        let params = hdbscan_defaults();

        // Per constitution: clustering.parameters.min_cluster_size: 3
        assert_eq!(params.min_cluster_size, 3, "min_cluster_size must be 3 per constitution");
        assert_eq!(params.min_samples, 2, "min_samples should be 2");
        assert_eq!(params.cluster_selection_method, ClusterSelectionMethod::EOM, "EOM is default");
        assert_eq!(params.metric, DistanceMetric::Cosine, "Cosine is default metric");

        // Validate should pass for defaults
        assert!(params.validate().is_ok(), "Default params must be valid");

        println!("[PASS] test_default_params_match_constitution - defaults verified against constitution");
    }

    #[test]
    fn test_cluster_selection_method_default() {
        let method = ClusterSelectionMethod::default();
        assert_eq!(method, ClusterSelectionMethod::EOM);

        println!("[PASS] test_cluster_selection_method_default - EOM is default");
    }

    // =========================================================================
    // SPACE-SPECIFIC PARAMS TESTS
    // =========================================================================

    #[test]
    fn test_default_for_space_semantic() {
        let params = HDBSCANParams::default_for_space(Embedder::Semantic);

        assert_eq!(params.metric, DistanceMetric::Cosine, "Semantic uses Cosine");
        assert_eq!(params.min_cluster_size, 3, "Semantic uses constitution default");
        assert!(params.validate().is_ok());

        println!("[PASS] test_default_for_space_semantic - metric={:?}, min_cluster_size={}",
                 params.metric, params.min_cluster_size);
    }

    #[test]
    fn test_default_for_space_sparse() {
        let params = HDBSCANParams::default_for_space(Embedder::Sparse);

        assert_eq!(params.metric, DistanceMetric::Jaccard, "Sparse uses Jaccard");
        assert_eq!(params.min_cluster_size, 5, "Sparse uses larger clusters");
        assert_eq!(params.min_samples, 3);
        assert!(params.validate().is_ok());

        println!("[PASS] test_default_for_space_sparse - metric={:?}, min_cluster_size={}",
                 params.metric, params.min_cluster_size);
    }

    #[test]
    fn test_default_for_space_keyword_splade() {
        let params = HDBSCANParams::default_for_space(Embedder::KeywordSplade);

        assert_eq!(params.metric, DistanceMetric::Jaccard, "KeywordSplade uses Jaccard");
        assert_eq!(params.min_cluster_size, 5, "KeywordSplade uses larger clusters");
        assert!(params.validate().is_ok());

        println!("[PASS] test_default_for_space_keyword_splade - metric={:?}", params.metric);
    }

    #[test]
    fn test_default_for_space_code() {
        let params = HDBSCANParams::default_for_space(Embedder::Code);

        // Code embedder should use constitution defaults
        assert_eq!(params.min_cluster_size, 3);
        assert_eq!(params.min_samples, 2);
        assert!(params.validate().is_ok());

        println!("[PASS] test_default_for_space_code - uses constitution defaults");
    }

    #[test]
    fn test_default_for_all_embedders() {
        // Verify all 13 embedder variants work
        for embedder in Embedder::all() {
            let params = HDBSCANParams::default_for_space(embedder);

            assert!(
                params.validate().is_ok(),
                "default_for_space({:?}) must produce valid params",
                embedder
            );
        }

        println!("[PASS] test_default_for_all_embedders - all 13 variants produce valid params");
    }

    // =========================================================================
    // VALIDATION TESTS - FAIL FAST
    // =========================================================================

    #[test]
    fn test_validation_rejects_min_cluster_size_below_2() {
        let params = HDBSCANParams {
            min_cluster_size: 1,
            min_samples: 1,
            cluster_selection_method: ClusterSelectionMethod::EOM,
            metric: DistanceMetric::Cosine,
        };

        let result = params.validate();
        assert!(result.is_err(), "min_cluster_size=1 must be rejected");

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("min_cluster_size"), "Error must mention field name");
        assert!(err_msg.contains("2"), "Error must mention minimum value");

        println!("[PASS] test_validation_rejects_min_cluster_size_below_2 - error: {}", err_msg);
    }

    #[test]
    fn test_validation_rejects_min_samples_zero() {
        let params = HDBSCANParams {
            min_cluster_size: 3,
            min_samples: 0,
            cluster_selection_method: ClusterSelectionMethod::EOM,
            metric: DistanceMetric::Cosine,
        };

        let result = params.validate();
        assert!(result.is_err(), "min_samples=0 must be rejected");

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("min_samples"), "Error must mention field name");

        println!("[PASS] test_validation_rejects_min_samples_zero - error: {}", err_msg);
    }

    #[test]
    fn test_validation_rejects_samples_greater_than_cluster_size() {
        let params = HDBSCANParams {
            min_cluster_size: 3,
            min_samples: 5, // > min_cluster_size
            cluster_selection_method: ClusterSelectionMethod::EOM,
            metric: DistanceMetric::Cosine,
        };

        let result = params.validate();
        assert!(result.is_err(), "min_samples > min_cluster_size must be rejected");

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("min_samples"), "Error must mention min_samples");
        assert!(err_msg.contains("min_cluster_size"), "Error must mention min_cluster_size");

        println!("[PASS] test_validation_rejects_samples_greater_than_cluster_size - error: {}", err_msg);
    }

    #[test]
    fn test_validation_accepts_boundary_values() {
        // Minimum valid: min_cluster_size=2, min_samples=1
        let minimal = HDBSCANParams {
            min_cluster_size: 2,
            min_samples: 1,
            cluster_selection_method: ClusterSelectionMethod::EOM,
            metric: DistanceMetric::Cosine,
        };
        assert!(minimal.validate().is_ok(), "Minimal valid params must pass");

        // Equal values: min_samples == min_cluster_size
        let equal = HDBSCANParams {
            min_cluster_size: 5,
            min_samples: 5,
            cluster_selection_method: ClusterSelectionMethod::Leaf,
            metric: DistanceMetric::Euclidean,
        };
        assert!(equal.validate().is_ok(), "Equal min_samples and min_cluster_size must pass");

        println!("[PASS] test_validation_accepts_boundary_values - boundary cases accepted");
    }

    // =========================================================================
    // BUILDER PATTERN TESTS
    // =========================================================================

    #[test]
    fn test_builder_pattern() {
        let params = HDBSCANParams::default()
            .with_min_cluster_size(10)
            .with_min_samples(5)
            .with_selection_method(ClusterSelectionMethod::Leaf)
            .with_metric(DistanceMetric::Euclidean);

        assert_eq!(params.min_cluster_size, 10);
        assert_eq!(params.min_samples, 5);
        assert_eq!(params.cluster_selection_method, ClusterSelectionMethod::Leaf);
        assert_eq!(params.metric, DistanceMetric::Euclidean);
        assert!(params.validate().is_ok());

        println!("[PASS] test_builder_pattern - all builder methods work");
    }

    #[test]
    fn test_builder_does_not_auto_clamp() {
        // Builder should NOT auto-clamp - validation is explicit
        let params = HDBSCANParams::default()
            .with_min_cluster_size(1) // Invalid value
            .with_min_samples(0);      // Invalid value

        assert_eq!(params.min_cluster_size, 1, "Builder must not modify value");
        assert_eq!(params.min_samples, 0, "Builder must not modify value");
        assert!(params.validate().is_err(), "Validation must catch invalid values");

        println!("[PASS] test_builder_does_not_auto_clamp - explicit validation required");
    }

    // =========================================================================
    // VIABILITY TESTS
    // =========================================================================

    #[test]
    fn test_is_viable_for_size() {
        let params = hdbscan_defaults(); // min_cluster_size = 3

        assert!(!params.is_viable_for_size(0), "0 points not viable");
        assert!(!params.is_viable_for_size(1), "1 point not viable");
        assert!(!params.is_viable_for_size(2), "2 points not viable");
        assert!(params.is_viable_for_size(3), "3 points exactly viable");
        assert!(params.is_viable_for_size(100), "100 points viable");
        assert!(params.is_viable_for_size(1_000_000), "1M points viable");

        println!("[PASS] test_is_viable_for_size - boundary checks correct");
    }

    // =========================================================================
    // SERIALIZATION TESTS
    // =========================================================================

    #[test]
    fn test_serialization_roundtrip() {
        let params = HDBSCANParams::default_for_space(Embedder::Causal)
            .with_min_cluster_size(7)
            .with_selection_method(ClusterSelectionMethod::Leaf);

        let json = serde_json::to_string(&amp;params).expect("serialize must succeed");
        let restored: HDBSCANParams = serde_json::from_str(&amp;json).expect("deserialize must succeed");

        assert_eq!(params.min_cluster_size, restored.min_cluster_size);
        assert_eq!(params.min_samples, restored.min_samples);
        assert_eq!(params.cluster_selection_method, restored.cluster_selection_method);
        assert_eq!(params.metric, restored.metric);

        println!("[PASS] test_serialization_roundtrip - JSON: {}", json);
    }

    #[test]
    fn test_cluster_selection_method_serialization() {
        for method in [ClusterSelectionMethod::EOM, ClusterSelectionMethod::Leaf] {
            let json = serde_json::to_string(&amp;method).expect("serialize");
            let restored: ClusterSelectionMethod = serde_json::from_str(&amp;json).expect("deserialize");
            assert_eq!(method, restored);
        }

        println!("[PASS] test_cluster_selection_method_serialization - both variants work");
    }

    // =========================================================================
    // DESCRIPTION TESTS
    // =========================================================================

    #[test]
    fn test_cluster_selection_method_description() {
        assert!(!ClusterSelectionMethod::EOM.description().is_empty());
        assert!(!ClusterSelectionMethod::Leaf.description().is_empty());

        // Descriptions should be different
        assert_ne!(
            ClusterSelectionMethod::EOM.description(),
            ClusterSelectionMethod::Leaf.description()
        );

        println!("[PASS] test_cluster_selection_method_description - descriptions exist");
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/clustering/hdbscan.rs">HDBSCANParams and ClusterSelectionMethod (params section, algorithm added in P4-005)</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/clustering/mod.rs">
    Add: pub mod hdbscan;
    Add to exports: pub use hdbscan::{ClusterSelectionMethod, HDBSCANParams, hdbscan_defaults};
  </file>
</files_to_modify>

<validation_criteria>
  <criterion id="VC-01">Default values: min_cluster_size=3 (per constitution), min_samples=2, EOM</criterion>
  <criterion id="VC-02">Validation rejects min_cluster_size &lt; 2 with descriptive error</criterion>
  <criterion id="VC-03">Validation rejects min_samples &lt; 1 with descriptive error</criterion>
  <criterion id="VC-04">Validation rejects min_samples > min_cluster_size with descriptive error</criterion>
  <criterion id="VC-05">Builder pattern allows customization without auto-clamping</criterion>
  <criterion id="VC-06">Space-specific defaults use correct distance metrics from get_distance_metric()</criterion>
  <criterion id="VC-07">All 13 embedder variants produce valid params in default_for_space</criterion>
  <criterion id="VC-08">Sparse and KeywordSplade use min_cluster_size=5, min_samples=3</criterion>
  <criterion id="VC-09">Serialization roundtrip preserves all fields</criterion>
  <criterion id="VC-10">Error messages are descriptive and include actual values</criterion>
</validation_criteria>

<full_state_verification>
  <source_of_truth>
    <item>Constitution clustering.parameters.min_cluster_size: 3 - verify default matches</item>
    <item>Constitution clustering.parameters.silhouette_threshold: 0.3 - informational for future tasks</item>
    <item>Embedder::all() returns exactly 13 variants</item>
    <item>get_distance_metric() in embeddings/config.rs returns correct metric per embedder</item>
    <item>ClusterError::invalid_parameter() helper exists in clustering/error.rs</item>
  </source_of_truth>

  <execute_and_inspect>
    <step>Run: cargo test --package context-graph-core hdbscan -- --nocapture</step>
    <step>Verify all 13 default_for_space tests pass with correct metrics</step>
    <step>Verify validation error messages contain field names and actual values</step>
    <step>Verify serialization roundtrip preserves all fields exactly</step>
  </execute_and_inspect>

  <boundary_edge_case_audit>
    <case>min_cluster_size=2, min_samples=1 - minimum valid, must pass</case>
    <case>min_cluster_size=2, min_samples=2 - equal values, must pass</case>
    <case>min_cluster_size=1, min_samples=1 - below minimum, must fail with clear error</case>
    <case>min_cluster_size=3, min_samples=0 - zero samples, must fail with clear error</case>
    <case>min_cluster_size=3, min_samples=4 - samples > cluster_size, must fail with clear error</case>
    <case>is_viable_for_size(0) - must return false</case>
    <case>is_viable_for_size(exact min_cluster_size) - must return true</case>
  </boundary_edge_case_audit>

  <evidence_of_success>
    <log>All tests pass with [PASS] prefix output</log>
    <log>cargo check --package context-graph-core succeeds with no warnings</log>
    <log>cargo clippy --package context-graph-core -- -D warnings succeeds</log>
    <log>Module exports visible: verify use context_graph_core::clustering::HDBSCANParams; compiles</log>
  </evidence_of_success>
</full_state_verification>

<test_commands>
  <command description="Run hdbscan param tests with output">cargo test --package context-graph-core hdbscan -- --nocapture</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
  <command description="Run clippy">cargo clippy --package context-graph-core -- -D warnings</command>
  <command description="Verify module export">cargo build --package context-graph-core 2>&amp;1 | head -20</command>
  <command description="Run all clustering tests">cargo test --package context-graph-core clustering</command>
</test_commands>

<implementation_notes>
CRITICAL IMPORT PATHS (DO NOT USE WRONG PATHS):
- use crate::teleological::Embedder;  (NOT crate::embedding::Embedder)
- use crate::index::config::DistanceMetric;  (NOT crate::embedding::config::DistanceMetric)
- use crate::embeddings::config::get_distance_metric;  (NOT crate::embedding::config)
- use super::error::ClusterError;  (within clustering module)

EMBEDDER VARIANT NAMES (DO NOT USE WRONG NAMES):
- Embedder::Sparse (NOT E6Sparse)
- Embedder::KeywordSplade (NOT E13SPLADE)
- Embedder::Code (NOT E7Code)
- Embedder::Semantic (NOT E1Semantic)

FAIL FAST PRINCIPLE:
- Do NOT auto-clamp values in builder methods
- Validation is explicit via validate()
- Error messages must be descriptive with actual values
- Use ClusterError::invalid_parameter() helper

NO MOCK DATA:
- Tests use real Embedder variants
- Tests verify actual get_distance_metric() returns
- Tests check actual module structure
</implementation_notes>
</task_spec>
```

## Execution Checklist

- [x] Read input context files (embeddings/config.rs, teleological/embedder.rs, index/config.rs, clustering/error.rs)
- [x] Verify ClusterError::invalid_parameter() helper exists
- [x] Verify get_distance_metric() function exists with correct signature
- [x] Create hdbscan.rs with ClusterSelectionMethod enum
- [x] Implement HDBSCANParams struct with correct imports
- [x] Implement Default trait (min_cluster_size=3 per constitution)
- [x] Implement default_for_space for all 13 embedders
- [x] Implement builder methods (with_*)
- [x] Implement validate() with descriptive error messages
- [x] Implement is_viable_for_size()
- [x] Implement hdbscan_defaults()
- [x] Add ClusterSelectionMethod::description()
- [x] Write comprehensive unit tests (18 unit tests + 18 FSV integration tests)
- [x] Update clustering/mod.rs with pub mod hdbscan and exports
- [x] Run tests: cargo test --package context-graph-core hdbscan -- --nocapture (18 passed)
- [x] Verify all tests show [PASS] output
- [x] Run: cargo clippy --package context-graph-core -- -D warnings (hdbscan module clean)
- [x] Code-simplifier review completed (minor improvement: removed redundant match arm, added PartialEq)
- [ ] Proceed to TASK-P4-004

## Discrepancies Fixed in v2.0

1. **Wrong import paths corrected:**
   - `crate::embedding::Embedder` → `crate::teleological::Embedder`
   - `crate::embedding::config::get_distance_metric` → `crate::embeddings::config::get_distance_metric`
   - `crate::embedding::config::DistanceMetric` → `crate::index::config::DistanceMetric`

2. **Wrong Embedder variant names corrected:**
   - `Embedder::E6Sparse` → `Embedder::Sparse`
   - `Embedder::E13SPLADE` → `Embedder::KeywordSplade`
   - `Embedder::E7Code` → `Embedder::Code`
   - `Embedder::E1Semantic` → `Embedder::Semantic`

3. **Added Full State Verification section**

4. **Added comprehensive boundary/edge case tests**

5. **Clarified fail-fast validation (no auto-clamping)**

6. **Added descriptive error message requirements**

7. **Added TASK-P4-001 as dependency (ClusterMembership, Cluster now exist)**

8. **Added implementation notes with explicit import paths and variant names**
