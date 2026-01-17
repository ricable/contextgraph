//! HDBSCAN clustering parameters.
//!
//! Provides configuration types for HDBSCAN algorithm.
//! Per constitution clustering.parameters.min_cluster_size: 3
//!
//! # Architecture
//!
//! Per constitution:
//! - ARCH-09: Topic threshold is weighted_agreement >= 2.5
//! - clustering.parameters.min_cluster_size: 3
//! - clustering.parameters.silhouette_threshold: 0.3

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
    pub fn description(&self) -> &'static str {
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
            // All other spaces use constitution default (min_cluster_size: 3)
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
    pub fn validate(&self) -> Result<(), ClusterError> {
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
    pub fn is_viable_for_size(&self, n_points: usize) -> bool {
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
        assert_eq!(
            params.min_cluster_size, 3,
            "min_cluster_size must be 3 per constitution"
        );
        assert_eq!(params.min_samples, 2, "min_samples should be 2");
        assert_eq!(
            params.cluster_selection_method,
            ClusterSelectionMethod::EOM,
            "EOM is default"
        );
        assert_eq!(
            params.metric,
            DistanceMetric::Cosine,
            "Cosine is default metric"
        );

        // Validate should pass for defaults
        assert!(params.validate().is_ok(), "Default params must be valid");

        println!(
            "[PASS] test_default_params_match_constitution - defaults verified against constitution"
        );
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
        assert_eq!(
            params.min_cluster_size, 3,
            "Semantic uses constitution default"
        );
        assert!(params.validate().is_ok());

        println!(
            "[PASS] test_default_for_space_semantic - metric={:?}, min_cluster_size={}",
            params.metric, params.min_cluster_size
        );
    }

    #[test]
    fn test_default_for_space_sparse() {
        let params = HDBSCANParams::default_for_space(Embedder::Sparse);

        assert_eq!(params.metric, DistanceMetric::Jaccard, "Sparse uses Jaccard");
        assert_eq!(params.min_cluster_size, 5, "Sparse uses larger clusters");
        assert_eq!(params.min_samples, 3);
        assert!(params.validate().is_ok());

        println!(
            "[PASS] test_default_for_space_sparse - metric={:?}, min_cluster_size={}",
            params.metric, params.min_cluster_size
        );
    }

    #[test]
    fn test_default_for_space_keyword_splade() {
        let params = HDBSCANParams::default_for_space(Embedder::KeywordSplade);

        assert_eq!(
            params.metric,
            DistanceMetric::Jaccard,
            "KeywordSplade uses Jaccard"
        );
        assert_eq!(params.min_cluster_size, 5, "KeywordSplade uses larger clusters");
        assert!(params.validate().is_ok());

        println!(
            "[PASS] test_default_for_space_keyword_splade - metric={:?}",
            params.metric
        );
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
        assert!(
            err_msg.contains("min_cluster_size"),
            "Error must mention field name"
        );
        assert!(err_msg.contains("2"), "Error must mention minimum value");

        println!(
            "[PASS] test_validation_rejects_min_cluster_size_below_2 - error: {}",
            err_msg
        );
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
        assert!(
            err_msg.contains("min_samples"),
            "Error must mention field name"
        );

        println!(
            "[PASS] test_validation_rejects_min_samples_zero - error: {}",
            err_msg
        );
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
        assert!(
            result.is_err(),
            "min_samples > min_cluster_size must be rejected"
        );

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("min_samples"),
            "Error must mention min_samples"
        );
        assert!(
            err_msg.contains("min_cluster_size"),
            "Error must mention min_cluster_size"
        );

        println!(
            "[PASS] test_validation_rejects_samples_greater_than_cluster_size - error: {}",
            err_msg
        );
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
        assert!(
            equal.validate().is_ok(),
            "Equal min_samples and min_cluster_size must pass"
        );

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
            .with_min_samples(0); // Invalid value

        assert_eq!(params.min_cluster_size, 1, "Builder must not modify value");
        assert_eq!(params.min_samples, 0, "Builder must not modify value");
        assert!(
            params.validate().is_err(),
            "Validation must catch invalid values"
        );

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

        let json = serde_json::to_string(&params).expect("serialize must succeed");
        let restored: HDBSCANParams =
            serde_json::from_str(&json).expect("deserialize must succeed");

        assert_eq!(params.min_cluster_size, restored.min_cluster_size);
        assert_eq!(params.min_samples, restored.min_samples);
        assert_eq!(
            params.cluster_selection_method,
            restored.cluster_selection_method
        );
        assert_eq!(params.metric, restored.metric);

        println!("[PASS] test_serialization_roundtrip - JSON: {}", json);
    }

    #[test]
    fn test_cluster_selection_method_serialization() {
        for method in [ClusterSelectionMethod::EOM, ClusterSelectionMethod::Leaf] {
            let json = serde_json::to_string(&method).expect("serialize");
            let restored: ClusterSelectionMethod = serde_json::from_str(&json).expect("deserialize");
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

    // =========================================================================
    // SPECIFIC DISTANCE METRIC TESTS FOR EACH EMBEDDER
    // =========================================================================

    #[test]
    fn test_metric_per_embedder() {
        // Semantic embedders: Check specific metrics
        assert_eq!(
            HDBSCANParams::default_for_space(Embedder::Semantic).metric,
            DistanceMetric::Cosine
        );
        assert_eq!(
            HDBSCANParams::default_for_space(Embedder::Causal).metric,
            DistanceMetric::AsymmetricCosine
        );
        assert_eq!(
            HDBSCANParams::default_for_space(Embedder::Sparse).metric,
            DistanceMetric::Jaccard
        );
        assert_eq!(
            HDBSCANParams::default_for_space(Embedder::Code).metric,
            DistanceMetric::Cosine
        );
        assert_eq!(
            HDBSCANParams::default_for_space(Embedder::Multimodal).metric,
            DistanceMetric::Cosine
        );
        assert_eq!(
            HDBSCANParams::default_for_space(Embedder::LateInteraction).metric,
            DistanceMetric::MaxSim
        );
        assert_eq!(
            HDBSCANParams::default_for_space(Embedder::KeywordSplade).metric,
            DistanceMetric::Jaccard
        );

        // Temporal embedders
        assert_eq!(
            HDBSCANParams::default_for_space(Embedder::TemporalRecent).metric,
            DistanceMetric::Cosine
        );
        assert_eq!(
            HDBSCANParams::default_for_space(Embedder::TemporalPeriodic).metric,
            DistanceMetric::Cosine
        );
        assert_eq!(
            HDBSCANParams::default_for_space(Embedder::TemporalPositional).metric,
            DistanceMetric::Cosine
        );

        // Relational embedders
        assert_eq!(
            HDBSCANParams::default_for_space(Embedder::Emotional).metric,
            DistanceMetric::Cosine
        );
        assert_eq!(
            HDBSCANParams::default_for_space(Embedder::Entity).metric,
            DistanceMetric::Cosine
        );

        // Structural embedder
        assert_eq!(
            HDBSCANParams::default_for_space(Embedder::Hdc).metric,
            DistanceMetric::Cosine
        );

        println!("[PASS] test_metric_per_embedder - all 13 embedders have correct metrics");
    }
}
