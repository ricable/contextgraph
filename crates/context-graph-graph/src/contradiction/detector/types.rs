//! Type definitions for contradiction detection.
//!
//! Contains structs and enums used throughout the contradiction detection module.

use uuid::Uuid;

use crate::error::{GraphError, GraphResult};

// Re-export core types
pub use context_graph_core::marblestone::{
    ContradictionType, Domain, EdgeType, NeurotransmitterWeights,
};

/// Result from contradiction detection.
///
/// Contains information about a detected contradiction between two nodes.
#[derive(Debug, Clone)]
pub struct ContradictionResult {
    /// The node that contradicts the query node (UUID).
    pub contradicting_node_id: Uuid,

    /// Type of contradiction detected.
    pub contradiction_type: ContradictionType,

    /// Overall confidence score [0, 1].
    /// Higher values indicate stronger contradiction evidence.
    pub confidence: f32,

    /// Semantic similarity to query node [0, 1].
    /// High similarity with contradiction indicates direct opposition.
    pub semantic_similarity: f32,

    /// Weight of explicit CONTRADICTS edge (if exists).
    pub edge_weight: Option<f32>,

    /// Whether there's an explicit contradiction edge.
    pub has_explicit_edge: bool,

    /// Evidence supporting the contradiction.
    pub evidence: Vec<String>,
}

impl ContradictionResult {
    /// Check if this is a high-confidence contradiction.
    ///
    /// # Arguments
    /// * `threshold` - Minimum confidence to consider high
    #[inline]
    pub fn is_high_confidence(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }

    /// Get severity based on type and confidence.
    ///
    /// Severity = confidence * type_severity
    /// DirectOpposition has highest type severity (1.0)
    #[inline]
    pub fn severity(&self) -> f32 {
        self.confidence * self.contradiction_type.severity()
    }
}

/// Parameters for contradiction detection.
///
/// Controls sensitivity, search depth, and evidence weighting.
#[derive(Debug, Clone)]
pub struct ContradictionParams {
    /// Minimum confidence threshold [0, 1].
    /// Contradictions below this are not returned.
    pub threshold: f32,

    /// Number of semantic similarity candidates to consider.
    pub semantic_k: usize,

    /// Minimum semantic similarity to consider [0, 1].
    pub min_similarity: f32,

    /// BFS depth for graph exploration.
    pub graph_depth: usize,

    /// Weight given to explicit edges vs semantic similarity.
    /// Higher = more weight to explicit edges.
    /// Range [0, 1].
    pub explicit_edge_weight: f32,
}

impl Default for ContradictionParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            semantic_k: 50,
            min_similarity: 0.3,
            graph_depth: 2,
            explicit_edge_weight: 0.6,
        }
    }
}

impl ContradictionParams {
    /// Builder: set confidence threshold.
    #[must_use]
    pub fn threshold(mut self, t: f32) -> Self {
        self.threshold = t.clamp(0.0, 1.0);
        self
    }

    /// Builder: set semantic k.
    #[must_use]
    pub fn semantic_k(mut self, k: usize) -> Self {
        self.semantic_k = k;
        self
    }

    /// Builder: set minimum similarity.
    #[must_use]
    pub fn min_similarity(mut self, s: f32) -> Self {
        self.min_similarity = s.clamp(0.0, 1.0);
        self
    }

    /// Builder: set graph depth.
    #[must_use]
    pub fn graph_depth(mut self, d: usize) -> Self {
        self.graph_depth = d;
        self
    }

    /// Builder: high sensitivity (lower threshold, more candidates).
    #[must_use]
    pub fn high_sensitivity(self) -> Self {
        self.threshold(0.3).semantic_k(100).min_similarity(0.2)
    }

    /// Builder: low sensitivity (higher threshold, fewer candidates).
    #[must_use]
    pub fn low_sensitivity(self) -> Self {
        self.threshold(0.7).semantic_k(20).min_similarity(0.5)
    }

    /// Validate parameters - FAIL FAST.
    pub fn validate(&self) -> GraphResult<()> {
        if self.semantic_k == 0 {
            return Err(GraphError::InvalidInput(
                "semantic_k must be > 0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.threshold) {
            return Err(GraphError::InvalidInput(format!(
                "threshold must be in [0, 1], got {}",
                self.threshold
            )));
        }
        if !(0.0..=1.0).contains(&self.min_similarity) {
            return Err(GraphError::InvalidInput(format!(
                "min_similarity must be in [0, 1], got {}",
                self.min_similarity
            )));
        }
        if !(0.0..=1.0).contains(&self.explicit_edge_weight) {
            return Err(GraphError::InvalidInput(format!(
                "explicit_edge_weight must be in [0, 1], got {}",
                self.explicit_edge_weight
            )));
        }
        Ok(())
    }
}

/// Internal candidate info for scoring.
pub(crate) struct CandidateInfo {
    pub semantic_similarity: f32,
    pub has_explicit_edge: bool,
    pub edge_weight: Option<f32>,
    pub edge_type: Option<ContradictionType>,
}
