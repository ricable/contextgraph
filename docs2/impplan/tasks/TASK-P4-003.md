# TASK-P4-003: HDBSCANParams and ClusterSelectionMethod

```xml
<task_spec id="TASK-P4-003" version="1.0">
<metadata>
  <title>HDBSCAN Parameter Types Implementation</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>29</sequence>
  <phase>4</phase>
  <implements>
    <requirement_ref>REQ-P4-01</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P3-004</task_ref>
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
</metadata>

<context>
Implements the parameter configuration types for HDBSCAN clustering algorithm.
HDBSCANParams contains min_cluster_size, min_samples, cluster_selection_method,
and distance metric. ClusterSelectionMethod enum defines EOM (Excess of Mass)
and Leaf selection strategies.

Default values match the technical specification: min_cluster_size=3, min_samples=2, EOM.
</context>

<input_context_files>
  <file purpose="data_models">docs2/impplan/technical/TECH-PHASE4-CLUSTERING.md#data_models</file>
  <file purpose="static_config">docs2/impplan/technical/TECH-PHASE4-CLUSTERING.md#static_configuration</file>
  <file purpose="distance">crates/context-graph-core/src/retrieval/distance.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P3-004 complete (DistanceMetric exists)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create ClusterSelectionMethod enum (EOM, Leaf)
    - Create HDBSCANParams struct
    - Define default parameters from spec
    - Implement validation logic
    - Parameter adjustment by embedder type
  </in_scope>
  <out_of_scope>
    - HDBSCAN algorithm implementation (TASK-P4-005)
    - Actual clustering execution
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/clustering/hdbscan.rs">
      #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
      pub enum ClusterSelectionMethod {
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
          pub fn validate(&amp;self) -> Result&lt;(), ClusterError&gt;;
      }

      pub fn hdbscan_defaults() -> HDBSCANParams;
    </signature>
  </signatures>

  <constraints>
    - min_cluster_size >= 2
    - min_samples >= 1
    - min_samples <= min_cluster_size
    - EOM is default method
    - Distance metric matches embedder config
  </constraints>

  <verification>
    - Default values match spec (3, 2, EOM)
    - Validation rejects invalid combinations
    - Builder pattern works for customization
    - Space-specific defaults are correct
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/clustering/hdbscan.rs (params section)

use serde::{Serialize, Deserialize};
use crate::embedding::Embedder;
use crate::embedding::config::{get_distance_metric, DistanceMetric};
use super::error::ClusterError;

/// Cluster selection method for HDBSCAN
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ClusterSelectionMethod {
    /// Excess of Mass - default, good general purpose
    #[default]
    EOM,
    /// Leaf clusters only - more granular
    Leaf,
}

impl ClusterSelectionMethod {
    /// Get description of this method
    pub fn description(&amp;self) -> &amp;'static str {
        match self {
            ClusterSelectionMethod::EOM => "Excess of Mass - good general purpose clustering",
            ClusterSelectionMethod::Leaf => "Leaf clusters only - more granular clustering",
        }
    }
}

/// Parameters for HDBSCAN clustering algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HDBSCANParams {
    /// Minimum number of points to form a cluster
    pub min_cluster_size: usize,
    /// Minimum samples for a point to be considered a core point
    pub min_samples: usize,
    /// Method for selecting clusters from hierarchy
    pub cluster_selection_method: ClusterSelectionMethod,
    /// Distance metric to use
    pub metric: DistanceMetric,
}

impl Default for HDBSCANParams {
    fn default() -> Self {
        Self {
            min_cluster_size: 3,
            min_samples: 2,
            cluster_selection_method: ClusterSelectionMethod::EOM,
            metric: DistanceMetric::Cosine,
        }
    }
}

impl HDBSCANParams {
    /// Create default params for a specific embedding space
    pub fn default_for_space(embedder: Embedder) -> Self {
        let metric = get_distance_metric(embedder);

        let (min_cluster, min_samples) = match embedder {
            // Sparse spaces may need different parameters
            Embedder::E6Sparse | Embedder::E13SPLADE => (5, 3),
            // Code clustering tends to be more specific
            Embedder::E7Code => (3, 2),
            // Default for most spaces
            _ => (3, 2),
        };

        Self {
            min_cluster_size: min_cluster,
            min_samples,
            cluster_selection_method: ClusterSelectionMethod::EOM,
            metric,
        }
    }

    /// Set minimum cluster size
    pub fn with_min_cluster_size(mut self, size: usize) -> Self {
        self.min_cluster_size = size.max(2);
        self
    }

    /// Set minimum samples
    pub fn with_min_samples(mut self, samples: usize) -> Self {
        self.min_samples = samples.max(1);
        self
    }

    /// Set cluster selection method
    pub fn with_selection_method(mut self, method: ClusterSelectionMethod) -> Self {
        self.cluster_selection_method = method;
        self
    }

    /// Set distance metric
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Validate parameters
    pub fn validate(&amp;self) -> Result&lt;(), ClusterError&gt; {
        if self.min_cluster_size < 2 {
            return Err(ClusterError::InvalidParameter {
                message: format!(
                    "min_cluster_size must be >= 2, got {}",
                    self.min_cluster_size
                ),
            });
        }

        if self.min_samples < 1 {
            return Err(ClusterError::InvalidParameter {
                message: format!("min_samples must be >= 1, got {}", self.min_samples),
            });
        }

        if self.min_samples > self.min_cluster_size {
            return Err(ClusterError::InvalidParameter {
                message: format!(
                    "min_samples ({}) must be <= min_cluster_size ({})",
                    self.min_samples, self.min_cluster_size
                ),
            });
        }

        Ok(())
    }

    /// Check if these params will work for a given data size
    pub fn is_viable_for_size(&amp;self, n_points: usize) -> bool {
        n_points >= self.min_cluster_size
    }
}

/// Get default HDBSCAN parameters
pub fn hdbscan_defaults() -> HDBSCANParams {
    HDBSCANParams::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let params = hdbscan_defaults();
        assert_eq!(params.min_cluster_size, 3);
        assert_eq!(params.min_samples, 2);
        assert_eq!(params.cluster_selection_method, ClusterSelectionMethod::EOM);
    }

    #[test]
    fn test_space_specific_params() {
        let semantic_params = HDBSCANParams::default_for_space(Embedder::E1Semantic);
        assert_eq!(semantic_params.metric, DistanceMetric::Cosine);
        assert_eq!(semantic_params.min_cluster_size, 3);

        let sparse_params = HDBSCANParams::default_for_space(Embedder::E6Sparse);
        assert_eq!(sparse_params.metric, DistanceMetric::Jaccard);
        assert_eq!(sparse_params.min_cluster_size, 5);
    }

    #[test]
    fn test_validation_valid() {
        let params = hdbscan_defaults();
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_validation_invalid_cluster_size() {
        let params = HDBSCANParams {
            min_cluster_size: 1,
            min_samples: 1,
            cluster_selection_method: ClusterSelectionMethod::EOM,
            metric: DistanceMetric::Cosine,
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_validation_invalid_samples() {
        let params = HDBSCANParams {
            min_cluster_size: 3,
            min_samples: 5, // > min_cluster_size
            cluster_selection_method: ClusterSelectionMethod::EOM,
            metric: DistanceMetric::Cosine,
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_builder_pattern() {
        let params = HDBSCANParams::default()
            .with_min_cluster_size(5)
            .with_min_samples(3)
            .with_selection_method(ClusterSelectionMethod::Leaf);

        assert_eq!(params.min_cluster_size, 5);
        assert_eq!(params.min_samples, 3);
        assert_eq!(params.cluster_selection_method, ClusterSelectionMethod::Leaf);
    }

    #[test]
    fn test_viability_check() {
        let params = hdbscan_defaults();
        assert!(!params.is_viable_for_size(2)); // Too few points
        assert!(params.is_viable_for_size(3));  // Exactly enough
        assert!(params.is_viable_for_size(100)); // Plenty
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/clustering/hdbscan.rs">HDBSCANParams and ClusterSelectionMethod (params section, algorithm added in P4-005)</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/clustering/mod.rs">Add pub mod hdbscan</file>
</files_to_modify>

<validation_criteria>
  <criterion>Default values: min_cluster_size=3, min_samples=2, EOM</criterion>
  <criterion>Validation rejects min_cluster_size &lt; 2</criterion>
  <criterion>Validation rejects min_samples > min_cluster_size</criterion>
  <criterion>Builder pattern allows customization</criterion>
  <criterion>Space-specific defaults use correct distance metrics</criterion>
</validation_criteria>

<test_commands>
  <command description="Run hdbscan param tests">cargo test --package context-graph-core hdbscan</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create hdbscan.rs (params section only)
- [ ] Implement ClusterSelectionMethod enum
- [ ] Implement HDBSCANParams struct
- [ ] Implement default_for_space method
- [ ] Implement builder pattern methods
- [ ] Implement validation logic
- [ ] Add hdbscan_defaults function
- [ ] Write unit tests
- [ ] Run tests to verify
- [ ] Proceed to TASK-P4-004
