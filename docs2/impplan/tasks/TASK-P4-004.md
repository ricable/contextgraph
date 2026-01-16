# TASK-P4-004: BIRCHParams and ClusteringFeature

```xml
<task_spec id="TASK-P4-004" version="1.0">
<metadata>
  <title>BIRCH Parameter and ClusteringFeature Types Implementation</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>30</sequence>
  <phase>4</phase>
  <implements>
    <requirement_ref>REQ-P4-02</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P4-001</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
</metadata>

<context>
Implements BIRCHParams configuration and ClusteringFeature (CF) struct.
BIRCHParams contains branching_factor, threshold, max_node_entries.
ClusteringFeature is the statistical summary used in BIRCH (n, linear_sum, squared_sum)
with methods for centroid, radius, merging, and distance calculation.

BIRCH enables O(log n) incremental clustering for real-time memory insertion.
</context>

<input_context_files>
  <file purpose="data_models">docs2/impplan/technical/TECH-PHASE4-CLUSTERING.md#data_models</file>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE4-CLUSTERING.md#component_contracts</file>
</input_context_files>

<prerequisites>
  <check>TASK-P4-001 complete (ClusterError exists)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create BIRCHParams struct
    - Create ClusteringFeature struct
    - Implement CF arithmetic (centroid, radius, diameter)
    - Implement CF merge operation
    - Implement CF distance calculation
    - Default values from spec
  </in_scope>
  <out_of_scope>
    - BIRCH tree structure (TASK-P4-006)
    - Tree insertion algorithm
    - Node splitting
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/clustering/birch.rs">
      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct BIRCHParams {
          pub branching_factor: usize,
          pub threshold: f32,
          pub max_node_entries: usize,
      }

      impl BIRCHParams {
          pub fn new(branching_factor: usize, threshold: f32, max_entries: usize) -> Self;
          pub fn default_for_space(embedder: Embedder) -> Self;
          pub fn validate(&amp;self) -> Result&lt;(), ClusterError&gt;;
      }

      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct ClusteringFeature {
          pub n: u32,
          pub ls: Vec&lt;f32&gt;,
          pub ss: f32,
      }

      impl ClusteringFeature {
          pub fn new(dimension: usize) -> Self;
          pub fn from_point(point: &amp;[f32]) -> Self;
          pub fn centroid(&amp;self) -> Vec&lt;f32&gt;;
          pub fn radius(&amp;self) -> f32;
          pub fn diameter(&amp;self) -> f32;
          pub fn merge(&amp;mut self, other: &amp;ClusteringFeature);
          pub fn distance(&amp;self, other: &amp;ClusteringFeature) -> f32;
          pub fn add_point(&amp;mut self, point: &amp;[f32]);
      }

      pub fn birch_defaults() -> BIRCHParams;
    </signature>
  </signatures>

  <constraints>
    - branching_factor >= 2
    - threshold > 0.0
    - max_node_entries >= branching_factor
    - CF dimension must be consistent
    - Default: branching_factor=50, threshold=0.3, max_entries=50
  </constraints>

  <verification>
    - centroid = ls / n
    - radius = sqrt(ss/n - centroid²)
    - merge correctly combines two CFs
    - CF distance is centroid distance
    - Default values match spec
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/clustering/birch.rs (params and CF section)

use serde::{Serialize, Deserialize};
use crate::embedding::Embedder;
use super::error::ClusterError;

/// Parameters for BIRCH clustering algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BIRCHParams {
    /// Maximum number of children per non-leaf node
    pub branching_factor: usize,
    /// Threshold for cluster radius (adaptive)
    pub threshold: f32,
    /// Maximum entries per leaf node
    pub max_node_entries: usize,
}

impl Default for BIRCHParams {
    fn default() -> Self {
        Self {
            branching_factor: 50,
            threshold: 0.3,
            max_node_entries: 50,
        }
    }
}

impl BIRCHParams {
    /// Create new BIRCH params
    pub fn new(branching_factor: usize, threshold: f32, max_entries: usize) -> Self {
        Self {
            branching_factor,
            threshold,
            max_node_entries: max_entries,
        }
    }

    /// Create params for a specific embedding space
    pub fn default_for_space(embedder: Embedder) -> Self {
        // Adjust threshold based on space characteristics
        let threshold = match embedder {
            // Sparse spaces need tighter threshold
            Embedder::E6Sparse | Embedder::E13SPLADE => 0.4,
            // Code embeddings are more specific
            Embedder::E7Code => 0.25,
            // Default for most spaces
            _ => 0.3,
        };

        Self {
            branching_factor: 50,
            threshold,
            max_node_entries: 50,
        }
    }

    /// Set branching factor
    pub fn with_branching_factor(mut self, bf: usize) -> Self {
        self.branching_factor = bf.max(2);
        self
    }

    /// Set threshold
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.max(0.001);
        self
    }

    /// Validate parameters
    pub fn validate(&amp;self) -> Result&lt;(), ClusterError&gt; {
        if self.branching_factor < 2 {
            return Err(ClusterError::InvalidParameter {
                message: format!(
                    "branching_factor must be >= 2, got {}",
                    self.branching_factor
                ),
            });
        }

        if self.threshold <= 0.0 {
            return Err(ClusterError::InvalidParameter {
                message: format!("threshold must be > 0, got {}", self.threshold),
            });
        }

        if self.max_node_entries < self.branching_factor {
            return Err(ClusterError::InvalidParameter {
                message: format!(
                    "max_node_entries ({}) must be >= branching_factor ({})",
                    self.max_node_entries, self.branching_factor
                ),
            });
        }

        Ok(())
    }
}

/// Get default BIRCH parameters
pub fn birch_defaults() -> BIRCHParams {
    BIRCHParams::default()
}

/// Clustering Feature - statistical summary for BIRCH
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringFeature {
    /// Number of points
    pub n: u32,
    /// Linear sum (sum of all points)
    pub ls: Vec&lt;f32&gt;,
    /// Squared sum (sum of squared norms)
    pub ss: f32,
}

impl ClusteringFeature {
    /// Create empty CF with given dimension
    pub fn new(dimension: usize) -> Self {
        Self {
            n: 0,
            ls: vec![0.0; dimension],
            ss: 0.0,
        }
    }

    /// Create CF from a single point
    pub fn from_point(point: &amp;[f32]) -> Self {
        let ss = point.iter().map(|x| x * x).sum();
        Self {
            n: 1,
            ls: point.to_vec(),
            ss,
        }
    }

    /// Get dimension of this CF
    pub fn dimension(&amp;self) -> usize {
        self.ls.len()
    }

    /// Compute centroid (mean point)
    pub fn centroid(&amp;self) -> Vec&lt;f32&gt; {
        if self.n == 0 {
            return self.ls.clone();
        }
        self.ls.iter().map(|x| x / self.n as f32).collect()
    }

    /// Compute radius (RMS distance from centroid to points)
    /// radius = sqrt(SS/N - ||centroid||²)
    pub fn radius(&amp;self) -> f32 {
        if self.n == 0 {
            return 0.0;
        }

        let centroid = self.centroid();
        let centroid_norm_sq: f32 = centroid.iter().map(|x| x * x).sum();
        let variance = (self.ss / self.n as f32) - centroid_norm_sq;

        if variance < 0.0 {
            0.0 // Numerical precision issue
        } else {
            variance.sqrt()
        }
    }

    /// Compute diameter (average pairwise distance)
    /// For large clusters: diameter ≈ 2 * radius
    pub fn diameter(&amp;self) -> f32 {
        if self.n <= 1 {
            return 0.0;
        }

        // Approximation: diameter ≈ 2 * radius for spherical clusters
        2.0 * self.radius()
    }

    /// Merge another CF into this one
    pub fn merge(&amp;mut self, other: &amp;ClusteringFeature) {
        if other.n == 0 {
            return;
        }

        if self.n == 0 {
            self.n = other.n;
            self.ls = other.ls.clone();
            self.ss = other.ss;
            return;
        }

        // Ensure dimensions match
        debug_assert_eq!(
            self.ls.len(),
            other.ls.len(),
            "CF dimension mismatch"
        );

        self.n += other.n;
        for (a, b) in self.ls.iter_mut().zip(other.ls.iter()) {
            *a += b;
        }
        self.ss += other.ss;
    }

    /// Add a single point to this CF
    pub fn add_point(&amp;mut self, point: &amp;[f32]) {
        if self.ls.is_empty() {
            self.ls = vec![0.0; point.len()];
        }

        debug_assert_eq!(
            self.ls.len(),
            point.len(),
            "Point dimension mismatch"
        );

        self.n += 1;
        for (a, b) in self.ls.iter_mut().zip(point.iter()) {
            *a += b;
        }
        self.ss += point.iter().map(|x| x * x).sum::<f32>();
    }

    /// Compute distance between this CF and another (centroid distance)
    pub fn distance(&amp;self, other: &amp;ClusteringFeature) -> f32 {
        let c1 = self.centroid();
        let c2 = other.centroid();

        c1.iter()
            .zip(c2.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt()
    }

    /// Check if a point would fit within threshold
    pub fn would_fit(&amp;self, point: &amp;[f32], threshold: f32) -> bool {
        if self.n == 0 {
            return true;
        }

        // Check if adding point would keep radius within threshold
        let mut test_cf = self.clone();
        test_cf.add_point(point);
        test_cf.radius() <= threshold
    }

    /// Check if empty
    pub fn is_empty(&amp;self) -> bool {
        self.n == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_birch_defaults() {
        let params = birch_defaults();
        assert_eq!(params.branching_factor, 50);
        assert_eq!(params.threshold, 0.3);
        assert_eq!(params.max_node_entries, 50);
    }

    #[test]
    fn test_birch_validation() {
        let params = birch_defaults();
        assert!(params.validate().is_ok());

        let bad_bf = BIRCHParams::new(1, 0.3, 50);
        assert!(bad_bf.validate().is_err());

        let bad_threshold = BIRCHParams::new(50, 0.0, 50);
        assert!(bad_threshold.validate().is_err());
    }

    #[test]
    fn test_cf_from_point() {
        let point = vec![1.0, 2.0, 3.0];
        let cf = ClusteringFeature::from_point(&amp;point);

        assert_eq!(cf.n, 1);
        assert_eq!(cf.ls, point);
        assert_eq!(cf.ss, 14.0); // 1 + 4 + 9
        assert_eq!(cf.centroid(), point);
    }

    #[test]
    fn test_cf_centroid() {
        let mut cf = ClusteringFeature::new(3);
        cf.add_point(&amp;[1.0, 0.0, 0.0]);
        cf.add_point(&amp;[3.0, 0.0, 0.0]);

        let centroid = cf.centroid();
        assert_eq!(centroid, vec![2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cf_merge() {
        let mut cf1 = ClusteringFeature::from_point(&amp;[1.0, 2.0]);
        let cf2 = ClusteringFeature::from_point(&amp;[3.0, 4.0]);

        cf1.merge(&amp;cf2);

        assert_eq!(cf1.n, 2);
        assert_eq!(cf1.ls, vec![4.0, 6.0]);
        assert_eq!(cf1.centroid(), vec![2.0, 3.0]);
    }

    #[test]
    fn test_cf_distance() {
        let cf1 = ClusteringFeature::from_point(&amp;[0.0, 0.0]);
        let cf2 = ClusteringFeature::from_point(&amp;[3.0, 4.0]);

        let dist = cf1.distance(&amp;cf2);
        assert!((dist - 5.0).abs() < 1e-5); // 3-4-5 triangle
    }

    #[test]
    fn test_cf_radius() {
        let mut cf = ClusteringFeature::new(2);
        // Add points at (-1, 0) and (1, 0)
        cf.add_point(&amp;[-1.0, 0.0]);
        cf.add_point(&amp;[1.0, 0.0]);

        // Centroid is (0, 0), each point is distance 1 away
        let radius = cf.radius();
        assert!((radius - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cf_would_fit() {
        let cf = ClusteringFeature::from_point(&amp;[0.0, 0.0]);

        // Point close by should fit
        assert!(cf.would_fit(&amp;[0.1, 0.1], 1.0));

        // Point far away should not fit
        assert!(!cf.would_fit(&amp;[10.0, 10.0], 1.0));
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/clustering/birch.rs">BIRCHParams and ClusteringFeature (tree added in P4-006)</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/clustering/mod.rs">Add pub mod birch</file>
</files_to_modify>

<validation_criteria>
  <criterion>Default values: branching_factor=50, threshold=0.3, max_entries=50</criterion>
  <criterion>CF centroid = ls / n</criterion>
  <criterion>CF radius computed correctly</criterion>
  <criterion>CF merge combines n, ls, ss correctly</criterion>
  <criterion>CF distance is centroid Euclidean distance</criterion>
  <criterion>would_fit checks if point fits within threshold</criterion>
</validation_criteria>

<test_commands>
  <command description="Run birch tests">cargo test --package context-graph-core birch</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create birch.rs (params and CF section)
- [ ] Implement BIRCHParams struct
- [ ] Implement ClusteringFeature struct
- [ ] Implement centroid calculation
- [ ] Implement radius and diameter
- [ ] Implement merge operation
- [ ] Implement distance calculation
- [ ] Add would_fit method
- [ ] Write unit tests
- [ ] Run tests to verify
- [ ] Proceed to TASK-P4-005
