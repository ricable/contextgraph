//! Core types for entailment query operations.
//!
//! This module defines the fundamental data structures used throughout
//! the entailment query system.

use crate::config::HyperbolicConfig;
use crate::entailment::cones::EntailmentCone;
use crate::hyperbolic::poincare::PoincarePoint;

/// Direction of entailment query traversal.
///
/// - `Ancestors`: Find concepts that entail (are more general than) the query node
/// - `Descendants`: Find concepts that are entailed by (are more specific than) the query node
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntailmentDirection {
    /// Find ancestors (more general concepts that contain this node in their cones)
    Ancestors,
    /// Find descendants (more specific concepts contained in this node's cone)
    Descendants,
}

/// Result of an entailment query for a single node.
///
/// Contains the node's hyperbolic embedding, entailment cone, and
/// membership score indicating strength of entailment relationship.
#[derive(Debug, Clone)]
pub struct EntailmentResult {
    /// Node ID (i64 for RocksDB compatibility)
    pub node_id: i64,
    /// Hyperbolic position in Poincare ball
    pub point: PoincarePoint,
    /// Entailment cone for this node
    pub cone: EntailmentCone,
    /// Membership score in [0, 1] indicating entailment strength
    pub membership_score: f32,
    /// Depth in hierarchy (from root)
    pub depth: u32,
    /// Whether this is a direct relationship (depth 1 in BFS)
    pub is_direct: bool,
}

/// Parameters for entailment queries.
///
/// Controls traversal depth, result limits, and filtering thresholds.
#[derive(Debug, Clone)]
pub struct EntailmentQueryParams {
    /// Maximum BFS depth for candidate generation (default: 3)
    pub max_depth: u32,
    /// Maximum number of results to return (default: 100)
    pub max_results: usize,
    /// Minimum membership score threshold (default: 0.7)
    pub min_membership_score: f32,
    /// Hyperbolic configuration for distance calculations
    pub hyperbolic_config: HyperbolicConfig,
}

impl Default for EntailmentQueryParams {
    fn default() -> Self {
        Self {
            max_depth: 3,
            max_results: 100,
            min_membership_score: 0.7,
            hyperbolic_config: HyperbolicConfig::default(),
        }
    }
}

impl EntailmentQueryParams {
    /// Create params with custom max depth.
    pub fn with_max_depth(mut self, depth: u32) -> Self {
        self.max_depth = depth;
        self
    }

    /// Create params with custom max results.
    pub fn with_max_results(mut self, max: usize) -> Self {
        self.max_results = max;
        self
    }

    /// Create params with custom minimum membership score.
    pub fn with_min_score(mut self, score: f32) -> Self {
        self.min_membership_score = score;
        self
    }

    /// Create params with custom hyperbolic config.
    pub fn with_hyperbolic_config(mut self, config: HyperbolicConfig) -> Self {
        self.hyperbolic_config = config;
        self
    }
}

/// Result of a batch entailment check.
#[derive(Debug, Clone)]
pub struct BatchEntailmentResult {
    /// Ancestor node ID
    pub ancestor_id: i64,
    /// Descendant node ID
    pub descendant_id: i64,
    /// Whether descendant is entailed by ancestor
    pub is_entailed: bool,
    /// Membership score in [0, 1]
    pub score: f32,
}

/// Result of a lowest common ancestor query.
#[derive(Debug, Clone)]
pub struct LcaResult {
    /// Lowest common ancestor node ID (None if no common ancestor found)
    pub lca_id: Option<i64>,
    /// Hyperbolic point of LCA (if found)
    pub lca_point: Option<PoincarePoint>,
    /// Entailment cone of LCA (if found)
    pub lca_cone: Option<EntailmentCone>,
    /// Distance from node_a to LCA (in BFS depth)
    pub depth_from_a: u32,
    /// Distance from node_b to LCA (in BFS depth)
    pub depth_from_b: u32,
}
