//! Type definitions for DFS traversal.
//!
//! Contains the core types used by DFS algorithms including
//! parameters, node IDs, and re-exports.

// Re-export edge types for convenience
pub use crate::storage::edges::{Domain, EdgeType};

/// Node ID type for DFS (i64 for storage compatibility).
pub type NodeId = i64;

/// Parameters for DFS traversal.
///
/// Controls depth limits, node limits, and filtering behavior.
#[derive(Debug, Clone)]
pub struct DfsParams {
    /// Maximum depth to traverse (None = unlimited).
    /// Depth 0 is the start node.
    pub max_depth: Option<usize>,

    /// Maximum number of nodes to visit (default: 10000).
    /// Prevents runaway traversal on dense graphs.
    pub max_nodes: Option<usize>,

    /// Filter to specific edge types (None = all types).
    pub edge_types: Option<Vec<EdgeType>>,

    /// Domain for NT weight modulation.
    pub domain: Domain,

    /// Minimum edge weight threshold (after modulation).
    /// Edges below this weight are not traversed.
    pub min_weight: f32,
}

impl Default for DfsParams {
    fn default() -> Self {
        Self {
            max_depth: Some(10),
            max_nodes: Some(10_000),
            edge_types: None,
            domain: Domain::General,
            min_weight: 0.0,
        }
    }
}

impl DfsParams {
    /// Builder: set max depth.
    #[must_use]
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Builder: set unlimited depth.
    #[must_use]
    pub fn unlimited_depth(mut self) -> Self {
        self.max_depth = None;
        self
    }

    /// Builder: set max nodes.
    #[must_use]
    pub fn max_nodes(mut self, nodes: usize) -> Self {
        self.max_nodes = Some(nodes);
        self
    }

    /// Builder: set edge types filter.
    #[must_use]
    pub fn edge_types(mut self, types: Vec<EdgeType>) -> Self {
        self.edge_types = Some(types);
        self
    }

    /// Builder: set domain for NT weight modulation.
    #[must_use]
    pub fn domain(mut self, domain: Domain) -> Self {
        self.domain = domain;
        self
    }

    /// Builder: set minimum weight threshold.
    #[must_use]
    pub fn min_weight(mut self, weight: f32) -> Self {
        self.min_weight = weight;
        self
    }
}
