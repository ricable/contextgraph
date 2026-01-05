//! Purpose Vector computation trait.
//!
//! Defines the interface for computing purpose vectors that align memories
//! to goal hierarchies.

use async_trait::async_trait;

use crate::types::fingerprint::{PurposeVector, SemanticFingerprint};

use super::goals::GoalHierarchy;

/// Configuration for purpose vector computation.
///
/// Controls how alignments are computed across embedding spaces and
/// whether hierarchical propagation is applied.
#[derive(Clone, Debug)]
pub struct PurposeComputeConfig {
    /// Goal hierarchy to align against.
    ///
    /// Must contain a North Star goal for computation to succeed.
    pub hierarchy: GoalHierarchy,

    /// Whether to propagate alignment up the hierarchy.
    ///
    /// When true, child goal alignments contribute to the overall
    /// alignment score using propagation weights.
    pub hierarchical_propagation: bool,

    /// Base/Strategic weighting for hierarchical propagation.
    ///
    /// First value is weight for North Star alignment,
    /// second is weight for child goal contributions.
    /// Default: (0.7, 0.3)
    pub propagation_weights: (f32, f32),

    /// Minimum alignment threshold for relevance.
    ///
    /// Alignments below this value may be treated as zero.
    /// Default: 0.0 (no minimum)
    pub min_alignment: f32,
}

impl Default for PurposeComputeConfig {
    fn default() -> Self {
        Self {
            hierarchy: GoalHierarchy::new(),
            hierarchical_propagation: true,
            propagation_weights: (0.7, 0.3),
            min_alignment: 0.0,
        }
    }
}

impl PurposeComputeConfig {
    /// Create a new config with the given hierarchy.
    pub fn with_hierarchy(hierarchy: GoalHierarchy) -> Self {
        Self {
            hierarchy,
            ..Default::default()
        }
    }

    /// Set whether to use hierarchical propagation.
    pub fn with_propagation(mut self, enabled: bool) -> Self {
        self.hierarchical_propagation = enabled;
        self
    }

    /// Set the propagation weights.
    pub fn with_weights(mut self, base_weight: f32, child_weight: f32) -> Self {
        self.propagation_weights = (base_weight.clamp(0.0, 1.0), child_weight.clamp(0.0, 1.0));
        self
    }

    /// Set minimum alignment threshold.
    pub fn with_min_alignment(mut self, min: f32) -> Self {
        self.min_alignment = min.clamp(0.0, 1.0);
        self
    }
}

/// Errors during purpose computation.
#[derive(Debug, thiserror::Error)]
pub enum PurposeComputeError {
    /// No North Star goal defined in the hierarchy.
    #[error("No North Star goal defined in hierarchy")]
    NoNorthStar,

    /// The fingerprint has no embeddings to compute alignment.
    #[error("Empty fingerprint - no embeddings to compute alignment")]
    EmptyFingerprint,

    /// Goal embedding dimension doesn't match expected size.
    #[error("Goal embedding dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        got: usize,
    },

    /// General computation failure.
    #[error("Computation failed: {0}")]
    ComputationFailed(String),
}

/// Trait for computing purpose vectors.
///
/// Implementations of this trait compute alignments between semantic fingerprints
/// and goal hierarchies to produce purpose vectors.
///
/// # Example
///
/// ```ignore
/// use context_graph_core::purpose::{
///     PurposeVectorComputer, PurposeComputeConfig, DefaultPurposeComputer
/// };
/// use context_graph_core::types::fingerprint::SemanticFingerprint;
///
/// let computer = DefaultPurposeComputer::new();
/// let fingerprint = SemanticFingerprint::zeroed();
/// let config = PurposeComputeConfig::default();
///
/// let purpose = computer.compute_purpose(&fingerprint, &config).await?;
/// ```
#[async_trait]
pub trait PurposeVectorComputer: Send + Sync {
    /// Compute purpose vector for a semantic fingerprint.
    ///
    /// Calculates alignment to the North Star goal for each of the 13
    /// embedding spaces.
    ///
    /// # Arguments
    ///
    /// * `fingerprint` - The semantic fingerprint to compute alignment for
    /// * `config` - Configuration including goal hierarchy
    ///
    /// # Returns
    ///
    /// A `PurposeVector` with alignments for all 13 spaces, or an error
    /// if computation fails.
    ///
    /// # Errors
    ///
    /// Returns `NoNorthStar` if the config's hierarchy has no North Star goal.
    async fn compute_purpose(
        &self,
        fingerprint: &SemanticFingerprint,
        config: &PurposeComputeConfig,
    ) -> Result<PurposeVector, PurposeComputeError>;

    /// Batch compute purpose vectors.
    ///
    /// More efficient than individual calls when computing for multiple
    /// fingerprints against the same goal hierarchy.
    ///
    /// # Arguments
    ///
    /// * `fingerprints` - Slice of fingerprints to compute
    /// * `config` - Shared configuration for all computations
    ///
    /// # Returns
    ///
    /// Vector of purpose vectors in the same order as input fingerprints.
    async fn compute_purpose_batch(
        &self,
        fingerprints: &[SemanticFingerprint],
        config: &PurposeComputeConfig,
    ) -> Result<Vec<PurposeVector>, PurposeComputeError>;

    /// Recompute purpose vector when goals change.
    ///
    /// Used when the goal hierarchy is updated and existing memories
    /// need their purpose vectors recalculated.
    ///
    /// # Arguments
    ///
    /// * `fingerprint` - The fingerprint to recompute
    /// * `old_hierarchy` - Previous goal hierarchy (for comparison/logging)
    /// * `new_hierarchy` - New goal hierarchy to align against
    ///
    /// # Returns
    ///
    /// Updated purpose vector aligned to the new hierarchy.
    async fn recompute_for_goal_change(
        &self,
        fingerprint: &SemanticFingerprint,
        old_hierarchy: &GoalHierarchy,
        new_hierarchy: &GoalHierarchy,
    ) -> Result<PurposeVector, PurposeComputeError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = PurposeComputeConfig::default();
        assert!(config.hierarchy.is_empty());
        assert!(config.hierarchical_propagation);
        assert_eq!(config.propagation_weights, (0.7, 0.3));
        assert_eq!(config.min_alignment, 0.0);
        println!("[VERIFIED] PurposeComputeConfig default values are correct");
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = PurposeComputeConfig::default()
            .with_propagation(false)
            .with_weights(0.8, 0.2)
            .with_min_alignment(0.5);

        assert!(!config.hierarchical_propagation);
        assert_eq!(config.propagation_weights, (0.8, 0.2));
        assert_eq!(config.min_alignment, 0.5);
        println!("[VERIFIED] PurposeComputeConfig builder pattern works");
    }

    #[test]
    fn test_config_weight_clamping() {
        let config = PurposeComputeConfig::default()
            .with_weights(1.5, -0.5)
            .with_min_alignment(2.0);

        assert_eq!(config.propagation_weights, (1.0, 0.0));
        assert_eq!(config.min_alignment, 1.0);
        println!("[VERIFIED] Config values are clamped to valid ranges");
    }

    #[test]
    fn test_error_display() {
        let e1 = PurposeComputeError::NoNorthStar;
        assert!(e1.to_string().contains("North Star"));

        let e2 = PurposeComputeError::EmptyFingerprint;
        assert!(e2.to_string().contains("Empty"));

        let e3 = PurposeComputeError::DimensionMismatch {
            expected: 1024,
            got: 512,
        };
        assert!(e3.to_string().contains("1024"));
        assert!(e3.to_string().contains("512"));

        let e4 = PurposeComputeError::ComputationFailed("test error".into());
        assert!(e4.to_string().contains("test error"));

        println!("[VERIFIED] PurposeComputeError display messages are correct");
    }
}
