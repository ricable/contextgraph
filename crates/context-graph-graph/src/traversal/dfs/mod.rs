//! DFS (Depth-First Search) graph traversal with Marblestone domain modulation.
//!
//! Explores the graph depth-first using an ITERATIVE approach (explicit stack).
//! NO RECURSION is used to avoid stack overflow on large graphs.
//!
//! # Performance
//!
//! Target: No stack overflow on 100,000+ node graphs.
//! Uses Vec<(NodeId, usize)> as explicit stack.
//! Uses HashSet for O(1) visited lookup.
//!
//! # Constitution Reference
//!
//! - edge_model.nt_weights.formula: Canonical modulation formula
//! - AP-009: NaN/Infinity clamped to valid range

mod iterator;
mod result;
mod traversal;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public API for backwards compatibility
pub use self::iterator::DfsIterator;
pub use self::result::DfsResult;
pub use self::traversal::{dfs_domain_neighborhood, dfs_neighborhood, dfs_traverse};
pub use self::types::{DfsParams, Domain, EdgeType, NodeId};
