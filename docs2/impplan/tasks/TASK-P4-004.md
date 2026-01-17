# TASK-P4-004: BIRCHParams and ClusteringFeature

```xml
<task_spec id="TASK-P4-004" version="3.0">
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
    <task_ref>TASK-P4-003</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <last_updated>2026-01-17</last_updated>
</metadata>

<context>
Implements BIRCHParams configuration and ClusteringFeature (CF) struct for
incremental clustering. BIRCH (Balanced Iterative Reducing and Clustering
using Hierarchies) enables O(log n) insertion for real-time memory clustering.

ClusteringFeature is the core statistical summary used in BIRCH CF-trees:
- n: number of points
- ls: linear sum (sum of all points)
- ss: squared sum (sum of squared norms)

Key CF operations:
- centroid() = ls / n
- radius() = sqrt(ss/n - ||centroid||^2)
- merge() combines two CFs additively
- distance() = Euclidean between centroids

Per constitution BIRCH_DEFAULTS:
- branching_factor: 50
- threshold: 0.3 (adaptive)
- max_node_entries: 50
</context>

<codebase_state>
VERIFIED CURRENT STATE (as of 2026-01-17):

1. EXISTING CLUSTERING MODULE STRUCTURE:
   File: crates/context-graph-core/src/clustering/mod.rs
   Current modules:
   - pub mod cluster;
   - pub mod error;
   - pub mod hdbscan;
   - pub mod membership;
   - pub mod topic;

   Current exports:
   - pub use cluster::Cluster;
   - pub use error::ClusterError;
   - pub use hdbscan::{hdbscan_defaults, ClusterSelectionMethod, HDBSCANParams};
   - pub use membership::ClusterMembership;
   - pub use topic::{Topic, TopicPhase, TopicProfile, TopicStability};

2. EMBEDDER ENUM LOCATION AND VARIANTS:
   File: crates/context-graph-core/src/teleological/embedder.rs
   Import: use crate::teleological::Embedder;

   ACTUAL VARIANTS (13 total):
   - Embedder::Semantic (index 0) - E1
   - Embedder::TemporalRecent (index 1) - E2
   - Embedder::TemporalPeriodic (index 2) - E3
   - Embedder::TemporalPositional (index 3) - E4
   - Embedder::Causal (index 4) - E5
   - Embedder::Sparse (index 5) - E6
   - Embedder::Code (index 6) - E7
   - Embedder::Emotional (index 7) - E8 (Graph/Relational)
   - Embedder::Hdc (index 8) - E9 (Structural)
   - Embedder::Multimodal (index 9) - E10
   - Embedder::Entity (index 10) - E11
   - Embedder::LateInteraction (index 11) - E12
   - Embedder::KeywordSplade (index 12) - E13

   Helper method: Embedder::all() -> impl Iterator&lt;Item = Embedder&gt;

3. CLUSTER ERROR (ALREADY EXISTS):
   File: crates/context-graph-core/src/clustering/error.rs
   Import: use super::error::ClusterError; (from within clustering module)

   Relevant variants:
   - ClusterError::InvalidParameter { message: String }
   - ClusterError::DimensionMismatch { expected: usize, actual: usize }

   Helper constructors:
   - ClusterError::invalid_parameter(message: impl Into&lt;String&gt;) -> Self
   - ClusterError::dimension_mismatch(expected: usize, actual: usize) -> Self

4. HDBSCAN PARAMS PATTERN (REFERENCE):
   File: crates/context-graph-core/src/clustering/hdbscan.rs
   Follow the same patterns:
   - Builder methods with #[must_use]
   - validate() returns Result&lt;(), ClusterError&gt;
   - default_for_space(embedder: Embedder) method
   - Public function: hdbscan_defaults() -> HDBSCANParams
   - Tests with [PASS] println! for CI visibility

5. DISTANCE METRIC (for reference):
   File: crates/context-graph-core/src/index/config.rs
   - DistanceMetric::Euclidean (use for CF distance)
</codebase_state>

<input_context_files>
  <file purpose="data_models" must_read="true">docs2/impplan/technical/TECH-PHASE4-CLUSTERING.md#data_models</file>
  <file purpose="component_spec" must_read="true">docs2/impplan/technical/TECH-PHASE4-CLUSTERING.md#component_contracts</file>
  <file purpose="constitution" must_read="true">CLAUDE.md (BIRCH_DEFAULTS section)</file>
  <file purpose="existing_error" must_read="true">crates/context-graph-core/src/clustering/error.rs</file>
  <file purpose="existing_module" must_read="true">crates/context-graph-core/src/clustering/mod.rs</file>
  <file purpose="reference_pattern" must_read="true">crates/context-graph-core/src/clustering/hdbscan.rs</file>
  <file purpose="embedder_enum" must_read="true">crates/context-graph-core/src/teleological/embedder.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P4-001 complete (ClusterError, ClusterMembership, Cluster exist)</check>
  <check>TASK-P4-003 complete (HDBSCANParams pattern established)</check>
  <check>ClusterError::InvalidParameter and DimensionMismatch variants exist</check>
  <check>Embedder enum exists with 13 variants and all() method</check>
</prerequisites>

<scope>
  <in_scope>
    - Create BIRCHParams struct with branching_factor, threshold, max_node_entries
    - Create ClusteringFeature struct with n, ls, ss
    - Implement CF arithmetic: centroid(), radius(), diameter()
    - Implement CF merge operation (additive)
    - Implement CF distance calculation (Euclidean centroid distance)
    - Implement add_point(), from_point(), would_fit()
    - Default values from constitution (branching_factor=50, threshold=0.3, max_entries=50)
    - Space-specific thresholds via default_for_space()
    - Builder pattern with with_* methods
    - Fail-fast validation
    - Serialize/Deserialize support
    - Comprehensive tests with edge cases
  </in_scope>
  <out_of_scope>
    - BIRCH tree structure (TASK-P4-006)
    - Tree insertion algorithm (TASK-P4-006)
    - Node splitting (TASK-P4-006)
    - Leaf/NonLeaf node types (TASK-P4-006)
    - Threshold adaptation algorithm (TASK-P4-006)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/clustering/birch.rs">
      //! BIRCH clustering parameters and ClusteringFeature.
      //!
      //! Per constitution: BIRCH_DEFAULTS
      //! - branching_factor: 50
      //! - threshold: 0.3
      //! - max_node_entries: 50

      use serde::{Deserialize, Serialize};
      use crate::teleological::Embedder;
      use super::error::ClusterError;

      /// Parameters for BIRCH clustering algorithm.
      #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
      pub struct BIRCHParams {
          pub branching_factor: usize,
          pub threshold: f32,
          pub max_node_entries: usize,
      }

      impl Default for BIRCHParams {
          fn default() -> Self; // branching_factor=50, threshold=0.3, max_entries=50
      }

      impl BIRCHParams {
          pub fn new(branching_factor: usize, threshold: f32, max_entries: usize) -> Self;
          pub fn default_for_space(embedder: Embedder) -> Self;
          pub fn with_branching_factor(self, bf: usize) -> Self;
          pub fn with_threshold(self, threshold: f32) -> Self;
          pub fn with_max_node_entries(self, entries: usize) -> Self;
          pub fn validate(&amp;self) -> Result&lt;(), ClusterError&gt;;
      }

      pub fn birch_defaults() -> BIRCHParams;

      /// Clustering Feature - statistical summary for BIRCH.
      ///
      /// CF = (n, LS, SS) where:
      /// - n = number of points
      /// - LS = linear sum (vector sum of all points)
      /// - SS = squared sum (scalar sum of squared norms)
      #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
      pub struct ClusteringFeature {
          pub n: u32,
          pub ls: Vec&lt;f32&gt;,
          pub ss: f32,
      }

      impl ClusteringFeature {
          pub fn new(dimension: usize) -> Self;
          pub fn from_point(point: &amp;[f32]) -> Self;
          pub fn dimension(&amp;self) -> usize;
          pub fn is_empty(&amp;self) -> bool;
          pub fn centroid(&amp;self) -> Vec&lt;f32&gt;;
          pub fn radius(&amp;self) -> f32;
          pub fn diameter(&amp;self) -> f32;
          pub fn merge(&amp;mut self, other: &amp;ClusteringFeature) -> Result&lt;(), ClusterError&gt;;
          pub fn add_point(&amp;mut self, point: &amp;[f32]) -> Result&lt;(), ClusterError&gt;;
          pub fn distance(&amp;self, other: &amp;ClusteringFeature) -> Result&lt;f32, ClusterError&gt;;
          pub fn would_fit(&amp;self, point: &amp;[f32], threshold: f32) -> bool;
      }
    </signature>
  </signatures>

  <constraints>
    - branching_factor >= 2 (fail fast if violated)
    - threshold > 0.0 (fail fast if violated)
    - max_node_entries >= branching_factor (fail fast if violated)
    - CF dimension must be consistent in merge/add_point (fail fast with DimensionMismatch)
    - Default: branching_factor=50, threshold=0.3, max_entries=50
    - NO auto-clamping in builder methods - validation is explicit
    - Handle n=0 gracefully in centroid/radius/diameter (return sensible defaults)
    - Handle negative variance in radius (numerical precision issue, return 0.0)
  </constraints>

  <verification>
    - Default values match constitution (50, 0.3, 50)
    - centroid() = ls / n (return zero vector if n=0)
    - radius() = sqrt(ss/n - ||centroid||^2), handle negative variance
    - diameter() = 2 * radius (approximation for spherical clusters)
    - merge() correctly combines n, ls, ss with dimension check
    - distance() = Euclidean distance between centroids with dimension check
    - would_fit() checks if adding point keeps radius within threshold
    - All 13 embedder variants work with default_for_space()
    - Validation errors are descriptive with actual values
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/clustering/birch.rs

//! BIRCH clustering parameters and ClusteringFeature.
//!
//! Provides configuration types for BIRCH (Balanced Iterative Reducing and
//! Clustering using Hierarchies) algorithm. BIRCH enables O(log n) incremental
//! clustering for real-time memory insertion.
//!
//! # Constitution Defaults
//!
//! Per constitution BIRCH_DEFAULTS:
//! - branching_factor: 50
//! - threshold: 0.3 (adaptive)
//! - max_node_entries: 50
//!
//! # Clustering Feature (CF)
//!
//! The CF is a triple (n, LS, SS) that summarizes a set of points:
//! - n: number of points
//! - LS: linear sum (vector sum of all points)
//! - SS: squared sum (scalar sum of squared norms)
//!
//! Key property: CFs are additive. CF(A u B) = CF(A) + CF(B)

use serde::{Deserialize, Serialize};

use crate::teleological::Embedder;

use super::error::ClusterError;

// =============================================================================
// BIRCHParams
// =============================================================================

/// Parameters for BIRCH clustering algorithm.
///
/// Per constitution: branching_factor=50, threshold=0.3, max_node_entries=50
///
/// # Example
///
/// ```
/// use context_graph_core::clustering::birch::{BIRCHParams, birch_defaults};
/// use context_graph_core::teleological::Embedder;
///
/// // Use defaults
/// let params = birch_defaults();
/// assert_eq!(params.branching_factor, 50);
///
/// // Or space-specific
/// let code_params = BIRCHParams::default_for_space(Embedder::Code);
/// assert!(code_params.threshold &lt; 0.3); // Code embeddings more specific
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BIRCHParams {
    /// Maximum number of children per non-leaf node.
    /// Controls tree width. Higher = flatter tree but more work per node.
    pub branching_factor: usize,

    /// Threshold for cluster radius.
    /// Points within this radius merge into same CF. Adaptive in practice.
    pub threshold: f32,

    /// Maximum entries per leaf node.
    /// When exceeded, node splits.
    pub max_node_entries: usize,
}

impl Default for BIRCHParams {
    fn default() -> Self {
        Self {
            branching_factor: 50, // Per constitution
            threshold: 0.3,        // Per constitution
            max_node_entries: 50,  // Per constitution
        }
    }
}

impl BIRCHParams {
    /// Create new BIRCH params.
    ///
    /// Values are NOT automatically validated - call validate() to check.
    pub fn new(branching_factor: usize, threshold: f32, max_entries: usize) -> Self {
        Self {
            branching_factor,
            threshold,
            max_node_entries: max_entries,
        }
    }

    /// Create params for a specific embedding space.
    ///
    /// Adjusts threshold based on space characteristics:
    /// - Sparse spaces (Sparse, KeywordSplade): 0.4 (looser for high dimensionality)
    /// - Code embeddings: 0.25 (tighter for specificity)
    /// - All other spaces: 0.3 (constitution default)
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::clustering::birch::BIRCHParams;
    /// use context_graph_core::teleological::Embedder;
    ///
    /// let sparse_params = BIRCHParams::default_for_space(Embedder::Sparse);
    /// assert_eq!(sparse_params.threshold, 0.4);
    ///
    /// let code_params = BIRCHParams::default_for_space(Embedder::Code);
    /// assert_eq!(code_params.threshold, 0.25);
    /// ```
    pub fn default_for_space(embedder: Embedder) -> Self {
        let threshold = match embedder {
            // Sparse spaces need looser threshold due to high dimensionality
            Embedder::Sparse | Embedder::KeywordSplade => 0.4,
            // Code embeddings are more specific, need tighter threshold
            Embedder::Code => 0.25,
            // All other spaces use constitution default
            _ => 0.3,
        };

        Self {
            branching_factor: 50,
            threshold,
            max_node_entries: 50,
        }
    }

    /// Set branching factor.
    ///
    /// Value is NOT automatically clamped - use validate() to check.
    #[must_use]
    pub fn with_branching_factor(mut self, bf: usize) -> Self {
        self.branching_factor = bf;
        self
    }

    /// Set threshold.
    ///
    /// Value is NOT automatically clamped - use validate() to check.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set max node entries.
    ///
    /// Value is NOT automatically clamped - use validate() to check.
    #[must_use]
    pub fn with_max_node_entries(mut self, entries: usize) -> Self {
        self.max_node_entries = entries;
        self
    }

    /// Validate parameters.
    ///
    /// Fails fast with descriptive error messages.
    ///
    /// # Errors
    ///
    /// Returns `ClusterError::InvalidParameter` if:
    /// - branching_factor &lt; 2
    /// - threshold &lt;= 0.0 or threshold is NaN/Infinity
    /// - max_node_entries &lt; branching_factor
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::clustering::birch::BIRCHParams;
    ///
    /// let invalid = BIRCHParams::new(1, 0.3, 50);
    /// assert!(invalid.validate().is_err());
    /// ```
    pub fn validate(&amp;self) -> Result&lt;(), ClusterError&gt; {
        if self.branching_factor &lt; 2 {
            return Err(ClusterError::invalid_parameter(format!(
                "branching_factor must be >= 2, got {}. BIRCH tree nodes need at least 2 children.",
                self.branching_factor
            )));
        }

        if self.threshold &lt;= 0.0 || self.threshold.is_nan() || self.threshold.is_infinite() {
            return Err(ClusterError::invalid_parameter(format!(
                "threshold must be > 0.0 and finite, got {}. Threshold controls cluster compactness.",
                self.threshold
            )));
        }

        if self.max_node_entries &lt; self.branching_factor {
            return Err(ClusterError::invalid_parameter(format!(
                "max_node_entries ({}) must be >= branching_factor ({}). Leaf nodes must hold at least branching_factor entries.",
                self.max_node_entries, self.branching_factor
            )));
        }

        Ok(())
    }
}

/// Get default BIRCH parameters.
///
/// Returns params matching constitution defaults:
/// - branching_factor: 50
/// - threshold: 0.3
/// - max_node_entries: 50
pub fn birch_defaults() -> BIRCHParams {
    BIRCHParams::default()
}

// =============================================================================
// ClusteringFeature
// =============================================================================

/// Clustering Feature - statistical summary for BIRCH.
///
/// A CF is a triple (n, LS, SS) that summarizes a set of d-dimensional points:
/// - n: number of data points
/// - LS: linear sum, d-dimensional vector = Sum Xi
/// - SS: squared sum, scalar = Sum ||Xi||^2
///
/// # Key Properties
///
/// 1. **Additivity**: CF(A u B) = CF(A) + CF(B)
/// 2. **Sufficient Statistics**: Can compute centroid, radius, diameter
/// 3. **Compact**: O(d) space regardless of n
///
/// # Example
///
/// ```
/// use context_graph_core::clustering::birch::ClusteringFeature;
///
/// let mut cf = ClusteringFeature::from_point(&amp;[1.0, 2.0, 3.0]);
/// cf.add_point(&amp;[2.0, 3.0, 4.0]).unwrap();
///
/// assert_eq!(cf.n, 2);
/// assert_eq!(cf.centroid(), vec![1.5, 2.5, 3.5]);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusteringFeature {
    /// Number of data points summarized.
    pub n: u32,
    /// Linear sum: Sum Xi (d-dimensional vector).
    pub ls: Vec&lt;f32&gt;,
    /// Squared sum: Sum ||Xi||^2 (scalar).
    pub ss: f32,
}

impl ClusteringFeature {
    /// Create empty CF with given dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            n: 0,
            ls: vec![0.0; dimension],
            ss: 0.0,
        }
    }

    /// Create CF from a single point.
    pub fn from_point(point: &amp;[f32]) -> Self {
        let ss: f32 = point.iter().map(|x| x * x).sum();
        Self {
            n: 1,
            ls: point.to_vec(),
            ss,
        }
    }

    /// Get dimension of this CF.
    #[inline]
    pub fn dimension(&amp;self) -> usize {
        self.ls.len()
    }

    /// Check if CF is empty (no points).
    #[inline]
    pub fn is_empty(&amp;self) -> bool {
        self.n == 0
    }

    /// Compute centroid (mean point).
    ///
    /// centroid = LS / n
    ///
    /// Returns zero vector if n=0.
    pub fn centroid(&amp;self) -> Vec&lt;f32&gt; {
        if self.n == 0 {
            return self.ls.clone(); // Zero vector
        }
        let n_f32 = self.n as f32;
        self.ls.iter().map(|x| x / n_f32).collect()
    }

    /// Compute radius (RMS distance from centroid to points).
    ///
    /// radius = sqrt(SS/n - ||centroid||^2)
    ///
    /// Returns 0.0 if n=0 or if variance is negative (numerical precision).
    pub fn radius(&amp;self) -> f32 {
        if self.n == 0 {
            return 0.0;
        }

        let centroid = self.centroid();
        let centroid_norm_sq: f32 = centroid.iter().map(|x| x * x).sum();
        let variance = (self.ss / self.n as f32) - centroid_norm_sq;

        // Handle numerical precision issues
        if variance &lt; 0.0 || variance.is_nan() {
            0.0
        } else {
            variance.sqrt()
        }
    }

    /// Compute diameter (average pairwise distance approximation).
    ///
    /// diameter approx 2 * radius (approximation for spherical clusters)
    ///
    /// Returns 0.0 if n &lt;= 1.
    pub fn diameter(&amp;self) -> f32 {
        if self.n &lt;= 1 {
            return 0.0;
        }
        2.0 * self.radius()
    }

    /// Merge another CF into this one.
    ///
    /// CF(A u B) = (n_A + n_B, LS_A + LS_B, SS_A + SS_B)
    ///
    /// # Errors
    ///
    /// Returns `ClusterError::DimensionMismatch` if dimensions differ.
    pub fn merge(&amp;mut self, other: &amp;ClusteringFeature) -> Result&lt;(), ClusterError&gt; {
        if other.n == 0 {
            return Ok(()); // Merging empty CF is no-op
        }

        if self.n == 0 {
            // Self is empty, just copy other
            self.n = other.n;
            self.ls = other.ls.clone();
            self.ss = other.ss;
            return Ok(());
        }

        // Check dimension match
        if self.ls.len() != other.ls.len() {
            return Err(ClusterError::dimension_mismatch(self.ls.len(), other.ls.len()));
        }

        // Additive merge
        self.n += other.n;
        for (a, b) in self.ls.iter_mut().zip(other.ls.iter()) {
            *a += b;
        }
        self.ss += other.ss;

        Ok(())
    }

    /// Add a single point to this CF.
    ///
    /// # Errors
    ///
    /// Returns `ClusterError::DimensionMismatch` if point dimension differs.
    pub fn add_point(&amp;mut self, point: &amp;[f32]) -> Result&lt;(), ClusterError&gt; {
        // Initialize dimension if empty
        if self.ls.is_empty() {
            self.ls = vec![0.0; point.len()];
        }

        // Check dimension match
        if self.ls.len() != point.len() {
            return Err(ClusterError::dimension_mismatch(self.ls.len(), point.len()));
        }

        self.n += 1;
        for (a, b) in self.ls.iter_mut().zip(point.iter()) {
            *a += b;
        }
        self.ss += point.iter().map(|x| x * x).sum::&lt;f32&gt;();

        Ok(())
    }

    /// Compute Euclidean distance between centroids of two CFs.
    ///
    /// # Errors
    ///
    /// Returns `ClusterError::DimensionMismatch` if dimensions differ.
    pub fn distance(&amp;self, other: &amp;ClusteringFeature) -> Result&lt;f32, ClusterError&gt; {
        if self.ls.len() != other.ls.len() {
            return Err(ClusterError::dimension_mismatch(self.ls.len(), other.ls.len()));
        }

        let c1 = self.centroid();
        let c2 = other.centroid();

        let dist_sq: f32 = c1
            .iter()
            .zip(c2.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();

        Ok(dist_sq.sqrt())
    }

    /// Check if a point would fit within threshold after merging.
    ///
    /// Creates a temporary merge and checks if radius &lt;= threshold.
    pub fn would_fit(&amp;self, point: &amp;[f32], threshold: f32) -> bool {
        if self.n == 0 {
            return true; // Empty CF accepts anything
        }

        // Dimension mismatch means doesn't fit
        if self.ls.len() != point.len() {
            return false;
        }

        // Check if adding point would keep radius within threshold
        let mut test_cf = self.clone();
        if test_cf.add_point(point).is_err() {
            return false;
        }
        test_cf.radius() &lt;= threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // BIRCHParams DEFAULT VALUES TESTS
    // =========================================================================

    #[test]
    fn test_birch_defaults_match_constitution() {
        let params = birch_defaults();

        // Per constitution: BIRCH_DEFAULTS
        assert_eq!(params.branching_factor, 50, "branching_factor must be 50 per constitution");
        assert!((params.threshold - 0.3).abs() &lt; f32::EPSILON, "threshold must be 0.3 per constitution");
        assert_eq!(params.max_node_entries, 50, "max_node_entries must be 50 per constitution");

        // Validate should pass for defaults
        assert!(params.validate().is_ok(), "Default params must be valid");

        println!("[PASS] test_birch_defaults_match_constitution - defaults verified");
    }

    // =========================================================================
    // BIRCHParams SPACE-SPECIFIC TESTS
    // =========================================================================

    #[test]
    fn test_default_for_space_sparse() {
        let params = BIRCHParams::default_for_space(Embedder::Sparse);
        assert!((params.threshold - 0.4).abs() &lt; f32::EPSILON, "Sparse should use 0.4 threshold");
        assert!(params.validate().is_ok());
        println!("[PASS] test_default_for_space_sparse - threshold=0.4");
    }

    #[test]
    fn test_default_for_space_keyword_splade() {
        let params = BIRCHParams::default_for_space(Embedder::KeywordSplade);
        assert!((params.threshold - 0.4).abs() &lt; f32::EPSILON, "KeywordSplade should use 0.4 threshold");
        assert!(params.validate().is_ok());
        println!("[PASS] test_default_for_space_keyword_splade - threshold=0.4");
    }

    #[test]
    fn test_default_for_space_code() {
        let params = BIRCHParams::default_for_space(Embedder::Code);
        assert!((params.threshold - 0.25).abs() &lt; f32::EPSILON, "Code should use 0.25 threshold");
        assert!(params.validate().is_ok());
        println!("[PASS] test_default_for_space_code - threshold=0.25");
    }

    #[test]
    fn test_default_for_space_semantic() {
        let params = BIRCHParams::default_for_space(Embedder::Semantic);
        assert!((params.threshold - 0.3).abs() &lt; f32::EPSILON, "Semantic should use 0.3 threshold");
        assert!(params.validate().is_ok());
        println!("[PASS] test_default_for_space_semantic - threshold=0.3");
    }

    #[test]
    fn test_default_for_all_embedders() {
        // Verify all 13 embedder variants produce valid params
        for embedder in Embedder::all() {
            let params = BIRCHParams::default_for_space(embedder);
            assert!(
                params.validate().is_ok(),
                "default_for_space({:?}) must produce valid params",
                embedder
            );
        }
        println!("[PASS] test_default_for_all_embedders - all 13 variants produce valid params");
    }

    // =========================================================================
    // BIRCHParams VALIDATION TESTS - FAIL FAST
    // =========================================================================

    #[test]
    fn test_validation_rejects_branching_factor_below_2() {
        let params = BIRCHParams::new(1, 0.3, 50);
        let result = params.validate();
        assert!(result.is_err(), "branching_factor=1 must be rejected");

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("branching_factor"), "Error must mention field name");
        assert!(err_msg.contains("2"), "Error must mention minimum value");

        println!("[PASS] test_validation_rejects_branching_factor_below_2 - error: {}", err_msg);
    }

    #[test]
    fn test_validation_rejects_zero_threshold() {
        let params = BIRCHParams::new(50, 0.0, 50);
        let result = params.validate();
        assert!(result.is_err(), "threshold=0.0 must be rejected");

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("threshold"), "Error must mention field name");

        println!("[PASS] test_validation_rejects_zero_threshold - error: {}", err_msg);
    }

    #[test]
    fn test_validation_rejects_negative_threshold() {
        let params = BIRCHParams::new(50, -0.1, 50);
        let result = params.validate();
        assert!(result.is_err(), "threshold=-0.1 must be rejected");
        println!("[PASS] test_validation_rejects_negative_threshold");
    }

    #[test]
    fn test_validation_rejects_nan_threshold() {
        let params = BIRCHParams::new(50, f32::NAN, 50);
        let result = params.validate();
        assert!(result.is_err(), "threshold=NaN must be rejected");
        println!("[PASS] test_validation_rejects_nan_threshold");
    }

    #[test]
    fn test_validation_rejects_infinite_threshold() {
        let params = BIRCHParams::new(50, f32::INFINITY, 50);
        let result = params.validate();
        assert!(result.is_err(), "threshold=INFINITY must be rejected");
        println!("[PASS] test_validation_rejects_infinite_threshold");
    }

    #[test]
    fn test_validation_rejects_max_entries_below_branching() {
        let params = BIRCHParams::new(50, 0.3, 40);
        let result = params.validate();
        assert!(result.is_err(), "max_node_entries &lt; branching_factor must be rejected");

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("max_node_entries"), "Error must mention max_node_entries");
        assert!(err_msg.contains("branching_factor"), "Error must mention branching_factor");

        println!("[PASS] test_validation_rejects_max_entries_below_branching - error: {}", err_msg);
    }

    #[test]
    fn test_validation_accepts_boundary_values() {
        // Minimum valid: branching_factor=2, threshold=0.001, max_entries=2
        let minimal = BIRCHParams::new(2, 0.001, 2);
        assert!(minimal.validate().is_ok(), "Minimal valid params must pass");

        // Equal values: max_entries == branching_factor
        let equal = BIRCHParams::new(100, 1.0, 100);
        assert!(equal.validate().is_ok(), "Equal max_entries and branching_factor must pass");

        println!("[PASS] test_validation_accepts_boundary_values");
    }

    // =========================================================================
    // BIRCHParams BUILDER TESTS
    // =========================================================================

    #[test]
    fn test_builder_pattern() {
        let params = BIRCHParams::default()
            .with_branching_factor(100)
            .with_threshold(0.5)
            .with_max_node_entries(200);

        assert_eq!(params.branching_factor, 100);
        assert!((params.threshold - 0.5).abs() &lt; f32::EPSILON);
        assert_eq!(params.max_node_entries, 200);
        assert!(params.validate().is_ok());

        println!("[PASS] test_builder_pattern - all builder methods work");
    }

    #[test]
    fn test_builder_does_not_auto_clamp() {
        // Builder should NOT auto-clamp - validation is explicit
        let params = BIRCHParams::default()
            .with_branching_factor(1)  // Invalid
            .with_threshold(0.0);       // Invalid

        assert_eq!(params.branching_factor, 1, "Builder must not modify value");
        assert!((params.threshold - 0.0).abs() &lt; f32::EPSILON, "Builder must not modify value");
        assert!(params.validate().is_err(), "Validation must catch invalid values");

        println!("[PASS] test_builder_does_not_auto_clamp - explicit validation required");
    }

    // =========================================================================
    // BIRCHParams SERIALIZATION TESTS
    // =========================================================================

    #[test]
    fn test_birch_params_serialization_roundtrip() {
        let params = BIRCHParams::default_for_space(Embedder::Code)
            .with_branching_factor(75)
            .with_max_node_entries(100);

        let json = serde_json::to_string(&amp;params).expect("serialize must succeed");
        let restored: BIRCHParams = serde_json::from_str(&amp;json).expect("deserialize must succeed");

        assert_eq!(params.branching_factor, restored.branching_factor);
        assert!((params.threshold - restored.threshold).abs() &lt; f32::EPSILON);
        assert_eq!(params.max_node_entries, restored.max_node_entries);

        println!("[PASS] test_birch_params_serialization_roundtrip - JSON: {}", json);
    }

    // =========================================================================
    // ClusteringFeature CREATION TESTS
    // =========================================================================

    #[test]
    fn test_cf_new_empty() {
        let cf = ClusteringFeature::new(128);
        assert_eq!(cf.n, 0);
        assert_eq!(cf.dimension(), 128);
        assert_eq!(cf.ls.len(), 128);
        assert!(cf.ls.iter().all(|&amp;x| x == 0.0));
        assert_eq!(cf.ss, 0.0);
        assert!(cf.is_empty());

        println!("[PASS] test_cf_new_empty - dimension={}", cf.dimension());
    }

    #[test]
    fn test_cf_from_point() {
        let point = vec![1.0, 2.0, 3.0];
        let cf = ClusteringFeature::from_point(&amp;point);

        assert_eq!(cf.n, 1);
        assert_eq!(cf.ls, point);
        assert_eq!(cf.ss, 14.0); // 1 + 4 + 9 = 14
        assert_eq!(cf.dimension(), 3);
        assert!(!cf.is_empty());
        assert_eq!(cf.centroid(), point);

        println!("[PASS] test_cf_from_point - n={}, ss={}", cf.n, cf.ss);
    }

    // =========================================================================
    // ClusteringFeature CENTROID TESTS
    // =========================================================================

    #[test]
    fn test_cf_centroid_single_point() {
        let point = vec![1.0, 2.0, 3.0];
        let cf = ClusteringFeature::from_point(&amp;point);
        assert_eq!(cf.centroid(), point);
        println!("[PASS] test_cf_centroid_single_point");
    }

    #[test]
    fn test_cf_centroid_two_points() {
        let mut cf = ClusteringFeature::new(3);
        cf.add_point(&amp;[1.0, 0.0, 0.0]).unwrap();
        cf.add_point(&amp;[3.0, 0.0, 0.0]).unwrap();

        let centroid = cf.centroid();
        assert_eq!(centroid, vec![2.0, 0.0, 0.0]);

        println!("[PASS] test_cf_centroid_two_points - centroid={:?}", centroid);
    }

    #[test]
    fn test_cf_centroid_empty() {
        let cf = ClusteringFeature::new(3);
        let centroid = cf.centroid();
        assert_eq!(centroid, vec![0.0, 0.0, 0.0], "Empty CF should return zero vector");

        println!("[PASS] test_cf_centroid_empty - returns zero vector");
    }

    // =========================================================================
    // ClusteringFeature RADIUS TESTS
    // =========================================================================

    #[test]
    fn test_cf_radius_single_point() {
        let cf = ClusteringFeature::from_point(&amp;[1.0, 2.0, 3.0]);
        assert_eq!(cf.radius(), 0.0, "Single point has zero radius");

        println!("[PASS] test_cf_radius_single_point - radius=0");
    }

    #[test]
    fn test_cf_radius_two_symmetric_points() {
        let mut cf = ClusteringFeature::new(2);
        cf.add_point(&amp;[-1.0, 0.0]).unwrap();
        cf.add_point(&amp;[1.0, 0.0]).unwrap();

        // Centroid is (0, 0), each point is distance 1 away
        let radius = cf.radius();
        assert!((radius - 1.0).abs() &lt; 1e-5);

        println!("[PASS] test_cf_radius_two_symmetric_points - radius={}", radius);
    }

    #[test]
    fn test_cf_radius_empty() {
        let cf = ClusteringFeature::new(3);
        assert_eq!(cf.radius(), 0.0, "Empty CF should have zero radius");

        println!("[PASS] test_cf_radius_empty - radius=0");
    }

    // =========================================================================
    // ClusteringFeature DIAMETER TESTS
    // =========================================================================

    #[test]
    fn test_cf_diameter_two_points() {
        let mut cf = ClusteringFeature::new(2);
        cf.add_point(&amp;[-1.0, 0.0]).unwrap();
        cf.add_point(&amp;[1.0, 0.0]).unwrap();

        let diameter = cf.diameter();
        // diameter approx 2 * radius = 2 * 1.0 = 2.0
        assert!((diameter - 2.0).abs() &lt; 1e-5);

        println!("[PASS] test_cf_diameter_two_points - diameter={}", diameter);
    }

    #[test]
    fn test_cf_diameter_single_point() {
        let cf = ClusteringFeature::from_point(&amp;[1.0, 2.0]);
        assert_eq!(cf.diameter(), 0.0, "Single point has zero diameter");

        println!("[PASS] test_cf_diameter_single_point - diameter=0");
    }

    // =========================================================================
    // ClusteringFeature MERGE TESTS
    // =========================================================================

    #[test]
    fn test_cf_merge_two_cfs() {
        let mut cf1 = ClusteringFeature::from_point(&amp;[1.0, 2.0]);
        let cf2 = ClusteringFeature::from_point(&amp;[3.0, 4.0]);

        cf1.merge(&amp;cf2).unwrap();

        assert_eq!(cf1.n, 2);
        assert_eq!(cf1.ls, vec![4.0, 6.0]);
        assert_eq!(cf1.ss, 5.0 + 25.0); // (1+4) + (9+16) = 30
        assert_eq!(cf1.centroid(), vec![2.0, 3.0]);

        println!("[PASS] test_cf_merge_two_cfs - n={}, ls={:?}", cf1.n, cf1.ls);
    }

    #[test]
    fn test_cf_merge_with_empty() {
        let mut cf1 = ClusteringFeature::from_point(&amp;[1.0, 2.0]);
        let cf2 = ClusteringFeature::new(2);

        cf1.merge(&amp;cf2).unwrap();

        assert_eq!(cf1.n, 1, "Merging empty CF should not change count");
        assert_eq!(cf1.ls, vec![1.0, 2.0], "Merging empty CF should not change LS");

        println!("[PASS] test_cf_merge_with_empty - no change");
    }

    #[test]
    fn test_cf_merge_into_empty() {
        let mut cf1 = ClusteringFeature::new(2);
        let cf2 = ClusteringFeature::from_point(&amp;[1.0, 2.0]);

        cf1.merge(&amp;cf2).unwrap();

        assert_eq!(cf1.n, 1);
        assert_eq!(cf1.ls, vec![1.0, 2.0]);

        println!("[PASS] test_cf_merge_into_empty - copied from other");
    }

    #[test]
    fn test_cf_merge_dimension_mismatch() {
        let mut cf1 = ClusteringFeature::from_point(&amp;[1.0, 2.0]);
        let cf2 = ClusteringFeature::from_point(&amp;[1.0, 2.0, 3.0]);

        let result = cf1.merge(&amp;cf2);
        assert!(result.is_err(), "Dimension mismatch must fail");

        let err = result.unwrap_err();
        assert!(matches!(err, ClusterError::DimensionMismatch { .. }));

        println!("[PASS] test_cf_merge_dimension_mismatch - correctly rejected");
    }

    // =========================================================================
    // ClusteringFeature ADD_POINT TESTS
    // =========================================================================

    #[test]
    fn test_cf_add_point() {
        let mut cf = ClusteringFeature::new(3);
        cf.add_point(&amp;[1.0, 2.0, 3.0]).unwrap();
        cf.add_point(&amp;[4.0, 5.0, 6.0]).unwrap();

        assert_eq!(cf.n, 2);
        assert_eq!(cf.ls, vec![5.0, 7.0, 9.0]);

        println!("[PASS] test_cf_add_point - n={}", cf.n);
    }

    #[test]
    fn test_cf_add_point_dimension_mismatch() {
        let mut cf = ClusteringFeature::from_point(&amp;[1.0, 2.0]);

        let result = cf.add_point(&amp;[1.0, 2.0, 3.0]);
        assert!(result.is_err(), "Dimension mismatch must fail");

        println!("[PASS] test_cf_add_point_dimension_mismatch - correctly rejected");
    }

    // =========================================================================
    // ClusteringFeature DISTANCE TESTS
    // =========================================================================

    #[test]
    fn test_cf_distance_3_4_5_triangle() {
        let cf1 = ClusteringFeature::from_point(&amp;[0.0, 0.0]);
        let cf2 = ClusteringFeature::from_point(&amp;[3.0, 4.0]);

        let dist = cf1.distance(&amp;cf2).unwrap();
        assert!((dist - 5.0).abs() &lt; 1e-5, "3-4-5 triangle: distance should be 5");

        println!("[PASS] test_cf_distance_3_4_5_triangle - dist={}", dist);
    }

    #[test]
    fn test_cf_distance_same_point() {
        let cf1 = ClusteringFeature::from_point(&amp;[1.0, 2.0, 3.0]);
        let cf2 = ClusteringFeature::from_point(&amp;[1.0, 2.0, 3.0]);

        let dist = cf1.distance(&amp;cf2).unwrap();
        assert!(dist.abs() &lt; 1e-5, "Same point distance should be 0");

        println!("[PASS] test_cf_distance_same_point - dist={}", dist);
    }

    #[test]
    fn test_cf_distance_dimension_mismatch() {
        let cf1 = ClusteringFeature::from_point(&amp;[1.0, 2.0]);
        let cf2 = ClusteringFeature::from_point(&amp;[1.0, 2.0, 3.0]);

        let result = cf1.distance(&amp;cf2);
        assert!(result.is_err(), "Dimension mismatch must fail");

        println!("[PASS] test_cf_distance_dimension_mismatch - correctly rejected");
    }

    // =========================================================================
    // ClusteringFeature WOULD_FIT TESTS
    // =========================================================================

    #[test]
    fn test_cf_would_fit_close_point() {
        let cf = ClusteringFeature::from_point(&amp;[0.0, 0.0]);
        assert!(cf.would_fit(&amp;[0.1, 0.1], 1.0), "Close point should fit");

        println!("[PASS] test_cf_would_fit_close_point");
    }

    #[test]
    fn test_cf_would_fit_far_point() {
        let cf = ClusteringFeature::from_point(&amp;[0.0, 0.0]);
        assert!(!cf.would_fit(&amp;[10.0, 10.0], 1.0), "Far point should not fit");

        println!("[PASS] test_cf_would_fit_far_point");
    }

    #[test]
    fn test_cf_would_fit_empty_cf() {
        let cf = ClusteringFeature::new(2);
        assert!(cf.would_fit(&amp;[100.0, 100.0], 0.1), "Empty CF should accept any point");

        println!("[PASS] test_cf_would_fit_empty_cf");
    }

    #[test]
    fn test_cf_would_fit_dimension_mismatch() {
        let cf = ClusteringFeature::from_point(&amp;[0.0, 0.0]);
        assert!(!cf.would_fit(&amp;[1.0, 2.0, 3.0], 10.0), "Dimension mismatch should not fit");

        println!("[PASS] test_cf_would_fit_dimension_mismatch");
    }

    // =========================================================================
    // ClusteringFeature SERIALIZATION TESTS
    // =========================================================================

    #[test]
    fn test_cf_serialization_roundtrip() {
        let mut cf = ClusteringFeature::from_point(&amp;[1.0, 2.0, 3.0]);
        cf.add_point(&amp;[4.0, 5.0, 6.0]).unwrap();

        let json = serde_json::to_string(&amp;cf).expect("serialize must succeed");
        let restored: ClusteringFeature = serde_json::from_str(&amp;json).expect("deserialize must succeed");

        assert_eq!(cf.n, restored.n);
        assert_eq!(cf.ls, restored.ls);
        assert!((cf.ss - restored.ss).abs() &lt; f32::EPSILON);

        println!("[PASS] test_cf_serialization_roundtrip - JSON: {}", json);
    }

    // =========================================================================
    // EDGE CASE TESTS
    // =========================================================================

    #[test]
    fn test_cf_high_dimensional() {
        // Test with 1024 dimensions (like E1 semantic embeddings)
        let dim = 1024;
        let mut cf = ClusteringFeature::new(dim);

        let point: Vec&lt;f32&gt; = (0..dim).map(|i| i as f32 * 0.001).collect();
        cf.add_point(&amp;point).unwrap();

        assert_eq!(cf.dimension(), dim);
        assert_eq!(cf.n, 1);

        println!("[PASS] test_cf_high_dimensional - dim={}", dim);
    }

    #[test]
    fn test_cf_numerical_precision_radius() {
        // Create scenario that could cause negative variance due to precision
        let mut cf = ClusteringFeature::new(2);
        // Add same point multiple times
        for _ in 0..1000 {
            cf.add_point(&amp;[1e-10, 1e-10]).unwrap();
        }

        let radius = cf.radius();
        assert!(!radius.is_nan(), "Radius should not be NaN");
        assert!(radius >= 0.0, "Radius should not be negative");

        println!("[PASS] test_cf_numerical_precision_radius - radius={}", radius);
    }

    #[test]
    fn test_cf_large_values() {
        let cf = ClusteringFeature::from_point(&amp;[1e6, 1e6, 1e6]);

        let centroid = cf.centroid();
        assert!((centroid[0] - 1e6).abs() &lt; 1.0, "Large values should work");

        println!("[PASS] test_cf_large_values");
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/clustering/birch.rs">BIRCHParams and ClusteringFeature (tree structure added in TASK-P4-006)</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/clustering/mod.rs">
    Add: pub mod birch;
    Add to exports: pub use birch::{birch_defaults, BIRCHParams, ClusteringFeature};
  </file>
</files_to_modify>

<validation_criteria>
  <criterion id="VC-01">Default values: branching_factor=50, threshold=0.3, max_entries=50 (per constitution)</criterion>
  <criterion id="VC-02">Validation rejects branching_factor &lt; 2 with descriptive error</criterion>
  <criterion id="VC-03">Validation rejects threshold &lt;= 0.0 with descriptive error</criterion>
  <criterion id="VC-04">Validation rejects threshold that is NaN or Infinity</criterion>
  <criterion id="VC-05">Validation rejects max_node_entries &lt; branching_factor with descriptive error</criterion>
  <criterion id="VC-06">Builder pattern allows customization without auto-clamping</criterion>
  <criterion id="VC-07">Space-specific defaults: Sparse/KeywordSplade=0.4, Code=0.25, others=0.3</criterion>
  <criterion id="VC-08">All 13 embedder variants produce valid params in default_for_space</criterion>
  <criterion id="VC-09">centroid() = ls / n, returns zero vector if n=0</criterion>
  <criterion id="VC-10">radius() = sqrt(ss/n - ||centroid||^2), handles negative variance</criterion>
  <criterion id="VC-11">diameter() = 2 * radius approximation</criterion>
  <criterion id="VC-12">merge() correctly combines n, ls, ss with dimension check</criterion>
  <criterion id="VC-13">add_point() increments n, updates ls and ss with dimension check</criterion>
  <criterion id="VC-14">distance() computes Euclidean between centroids with dimension check</criterion>
  <criterion id="VC-15">would_fit() checks if merged radius &lt;= threshold</criterion>
  <criterion id="VC-16">Serialization roundtrip preserves all fields</criterion>
  <criterion id="VC-17">Error messages are descriptive and include actual values</criterion>
</validation_criteria>

<full_state_verification>
  <source_of_truth>
    <item>Constitution BIRCH_DEFAULTS: branching_factor=50, threshold=0.3, max_node_entries=50</item>
    <item>ClusterError::invalid_parameter() helper exists in clustering/error.rs</item>
    <item>ClusterError::dimension_mismatch() helper exists in clustering/error.rs</item>
    <item>Embedder::all() returns exactly 13 variants</item>
    <item>clustering/mod.rs exports are consistent</item>
  </source_of_truth>

  <execute_and_inspect>
    <step>Run: cargo test --package context-graph-core birch -- --nocapture</step>
    <step>Verify all tests pass with [PASS] prefix output</step>
    <step>Verify BIRCHParams serialization roundtrip in test output</step>
    <step>Verify ClusteringFeature operations in test output</step>
    <step>Run: cargo clippy --package context-graph-core -- -D warnings</step>
    <step>Verify module exports: cargo build --package context-graph-core</step>
  </execute_and_inspect>

  <boundary_edge_case_audit>
    <case id="BIRCH-EDGE-01">branching_factor=2, threshold=0.001, max_entries=2 - minimum valid, must pass</case>
    <case id="BIRCH-EDGE-02">branching_factor=1, threshold=0.3, max_entries=50 - below minimum, must fail with clear error</case>
    <case id="BIRCH-EDGE-03">threshold=0.0 - zero threshold, must fail with clear error</case>
    <case id="BIRCH-EDGE-04">threshold=NaN - NaN threshold, must fail with clear error</case>
    <case id="BIRCH-EDGE-05">threshold=INFINITY - infinite threshold, must fail with clear error</case>
    <case id="BIRCH-EDGE-06">max_node_entries &lt; branching_factor - must fail with clear error</case>
    <case id="CF-EDGE-01">Empty CF: centroid returns zero vector, radius=0, diameter=0</case>
    <case id="CF-EDGE-02">Single point: centroid=point, radius=0, diameter=0</case>
    <case id="CF-EDGE-03">Merge empty into non-empty: no change</case>
    <case id="CF-EDGE-04">Merge non-empty into empty: copy values</case>
    <case id="CF-EDGE-05">Dimension mismatch in merge: DimensionMismatch error</case>
    <case id="CF-EDGE-06">Dimension mismatch in add_point: DimensionMismatch error</case>
    <case id="CF-EDGE-07">Dimension mismatch in distance: DimensionMismatch error</case>
    <case id="CF-EDGE-08">Dimension mismatch in would_fit: returns false</case>
    <case id="CF-EDGE-09">High-dimensional (1024): all operations work</case>
    <case id="CF-EDGE-10">Numerical precision causing negative variance: radius returns 0.0</case>
  </boundary_edge_case_audit>

  <evidence_of_success>
    <log>All tests pass with [PASS] prefix output</log>
    <log>cargo check --package context-graph-core succeeds with no warnings</log>
    <log>cargo clippy --package context-graph-core -- -D warnings succeeds</log>
    <log>Module exports visible: use context_graph_core::clustering::BIRCHParams; compiles</log>
    <log>Module exports visible: use context_graph_core::clustering::ClusteringFeature; compiles</log>
  </evidence_of_success>
</full_state_verification>

<test_commands>
  <command description="Run birch tests with output">cargo test --package context-graph-core birch -- --nocapture</command>
  <command description="Run all clustering tests">cargo test --package context-graph-core clustering -- --nocapture</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
  <command description="Run clippy">cargo clippy --package context-graph-core -- -D warnings</command>
  <command description="Verify module export">cargo build --package context-graph-core 2>&amp;1 | head -20</command>
</test_commands>

<manual_verification_protocol>
After completing implementation, you MUST perform Full State Verification:

1. DEFINE SOURCE OF TRUTH:
   - BIRCHParams defaults are stored in code as const values
   - ClusteringFeature operations are mathematical transformations
   - Verify by running cargo test and inspecting [PASS] output

2. EXECUTE AND INSPECT:
   Run the following commands and verify output:
   ```bash
   # Run all birch tests with verbose output
   cargo test --package context-graph-core birch -- --nocapture 2>&amp;1 | tee /tmp/birch_test.log

   # Verify all tests pass
   grep -c "\[PASS\]" /tmp/birch_test.log
   # Expected: 30+ (one per test)

   # Verify no failures
   grep -c "FAILED" /tmp/birch_test.log
   # Expected: 0

   # Verify module exports work
   echo 'use context_graph_core::clustering::{BIRCHParams, ClusteringFeature};' > /tmp/test_import.rs
   rustc --edition 2021 --emit=metadata -L target/debug/deps /tmp/test_import.rs 2>&amp;1 || echo "Expected to fail (just checking syntax)"
   ```

3. BOUNDARY AND EDGE CASE AUDIT:
   For each edge case listed above, verify the test:
   - Prints state BEFORE operation
   - Prints state AFTER operation
   - Prints expected vs actual outcome

   Example from test output:
   ```
   [PASS] test_validation_rejects_branching_factor_below_2 - error: branching_factor must be >= 2, got 1...
   ```

4. EVIDENCE OF SUCCESS:
   Capture and include in response:
   - Screenshot or copy of test output showing all [PASS]
   - Clippy output showing no warnings
   - Build output showing successful compilation
</manual_verification_protocol>

<implementation_notes>
CRITICAL IMPORT PATHS (DO NOT USE WRONG PATHS):
- use crate::teleological::Embedder;  (NOT crate::embedding::Embedder)
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
- Use ClusterError::invalid_parameter() for validation errors
- Use ClusterError::dimension_mismatch() for dimension errors

NUMERICAL STABILITY:
- Handle n=0 cases gracefully (return sensible defaults)
- Handle negative variance in radius (return 0.0)
- Handle NaN/Infinity in threshold validation

NO MOCK DATA:
- Tests use real Embedder variants
- Tests verify actual mathematical properties
- Tests check actual module structure

FOLLOW HDBSCAN PATTERN:
- Same structure as hdbscan.rs
- Same test style with [PASS] println!
- Same documentation style
- Same error handling pattern
</implementation_notes>
</task_spec>
```

## Execution Checklist

- [ ] Read input context files (clustering/error.rs, clustering/mod.rs, hdbscan.rs for pattern)
- [ ] Verify ClusterError::invalid_parameter() helper exists
- [ ] Verify ClusterError::dimension_mismatch() helper exists
- [ ] Create birch.rs with BIRCHParams struct
- [ ] Implement Default for BIRCHParams (50, 0.3, 50 per constitution)
- [ ] Implement new(), default_for_space() for BIRCHParams
- [ ] Implement builder methods with_* for BIRCHParams
- [ ] Implement validate() with descriptive errors
- [ ] Create ClusteringFeature struct with n, ls, ss
- [ ] Implement new(), from_point(), dimension(), is_empty()
- [ ] Implement centroid() with n=0 handling
- [ ] Implement radius() with negative variance handling
- [ ] Implement diameter()
- [ ] Implement merge() with dimension check (returns Result)
- [ ] Implement add_point() with dimension check (returns Result)
- [ ] Implement distance() with dimension check (returns Result)
- [ ] Implement would_fit()
- [ ] Add birch_defaults() function
- [ ] Write comprehensive unit tests (30+ tests)
- [ ] All tests include [PASS] println! statements
- [ ] Update clustering/mod.rs with pub mod birch and exports
- [ ] Run: cargo test --package context-graph-core birch -- --nocapture
- [ ] Verify all tests pass with [PASS] output
- [ ] Run: cargo clippy --package context-graph-core -- -D warnings
- [ ] Verify no clippy warnings
- [ ] Run: cargo build --package context-graph-core
- [ ] Verify build succeeds
- [ ] Proceed to TASK-P4-005

## Discrepancies Fixed in v3.0

1. **Corrected import paths:**
   - `crate::embedding::Embedder` -> `crate::teleological::Embedder`

2. **Corrected Embedder variant names:**
   - `Embedder::E6Sparse` -> `Embedder::Sparse`
   - `Embedder::E13SPLADE` -> `Embedder::KeywordSplade`
   - `Embedder::E7Code` -> `Embedder::Code`

3. **Added dimension checking to CF operations:**
   - merge() now returns Result<(), ClusterError>
   - add_point() now returns Result<(), ClusterError>
   - distance() now returns Result<f32, ClusterError>
   - Use ClusterError::dimension_mismatch() for errors

4. **Added numerical stability handling:**
   - Handle n=0 in centroid/radius/diameter
   - Handle negative variance in radius (numerical precision)
   - Handle NaN/Infinity in threshold validation

5. **Added comprehensive edge case tests:**
   - Empty CF operations
   - Dimension mismatch scenarios
   - High-dimensional vectors (1024D)
   - Numerical precision edge cases

6. **Added Full State Verification section**

7. **Added Manual Verification Protocol**

8. **Added dependency on TASK-P4-003** (for established patterns)

9. **Clarified fail-fast validation (no auto-clamping in builders)**

10. **Added PartialEq derive for test assertions**

11. **Fixed pseudo_code to be complete working implementation**
