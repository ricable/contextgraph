//! DFS traversal result type.
//!
//! Contains the result structure returned by DFS traversal operations.

use std::collections::HashMap;

use super::types::NodeId;

/// Result of DFS traversal.
#[derive(Debug, Clone)]
pub struct DfsResult {
    /// Visited node IDs in DFS pre-order.
    pub visited_order: Vec<NodeId>,

    /// Depth at which each node was discovered.
    pub depths: HashMap<NodeId, usize>,

    /// Parent node for each discovered node (start node has None).
    pub parents: HashMap<NodeId, Option<NodeId>>,

    /// Edges traversed: (source, target, effective_weight).
    pub edges_traversed: Vec<(NodeId, NodeId, f32)>,
}

impl DfsResult {
    /// Create a new empty result.
    #[must_use]
    pub fn new() -> Self {
        Self {
            visited_order: Vec::new(),
            depths: HashMap::new(),
            parents: HashMap::new(),
            edges_traversed: Vec::new(),
        }
    }

    /// Reconstruct path from start to target.
    ///
    /// Returns None if target was not visited.
    #[must_use]
    pub fn path_to(&self, target: NodeId) -> Option<Vec<NodeId>> {
        if !self.parents.contains_key(&target) {
            return None;
        }

        let mut path = vec![target];
        let mut current = target;

        while let Some(Some(parent)) = self.parents.get(&current) {
            path.push(*parent);
            current = *parent;
        }

        path.reverse();
        Some(path)
    }

    /// Get total node count.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.visited_order.len()
    }

    /// Get maximum depth reached.
    #[must_use]
    pub fn max_depth_reached(&self) -> usize {
        self.depths.values().copied().max().unwrap_or(0)
    }
}

impl Default for DfsResult {
    fn default() -> Self {
        Self::new()
    }
}
