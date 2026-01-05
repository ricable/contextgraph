//! Lowest Common Ancestor (LCA) operations.
//!
//! This module provides functions for finding the lowest common ancestor
//! of nodes in the entailment hierarchy.

use crate::error::GraphResult;
use crate::hyperbolic::PoincareBall;
use crate::storage::GraphStorage;

use super::conversion::{storage_to_entailment_cone, storage_to_hyperbolic_point};
use super::traversal::collect_ancestors;
use super::types::{EntailmentQueryParams, LcaResult};

/// Find the lowest common ancestor of two nodes in the entailment hierarchy.
///
/// The LCA is the most specific (deepest) concept that entails both input nodes.
///
/// # Algorithm
///
/// 1. Collect all ancestors of node_a using BFS up the hierarchy
/// 2. Collect all ancestors of node_b using BFS up the hierarchy
/// 3. Find intersection of ancestor sets
/// 4. Return the ancestor with highest depth (most specific)
///
/// # Arguments
///
/// * `storage` - GraphStorage for retrieving data
/// * `node_a` - First node
/// * `node_b` - Second node
/// * `params` - Query parameters (max_depth limits search)
///
/// # Returns
///
/// * `Ok(LcaResult)` - LCA result (lca_id may be None if no common ancestor)
/// * `Err(GraphError::MissingHyperbolicData)` - If required data is missing
///
/// # Performance
///
/// O(n + m) where n, m = nodes visited in each BFS, target <10ms
///
/// # Example
///
/// ```ignore
/// // Find LCA of "Dog" and "Cat" (should be "Animal")
/// let result = lowest_common_ancestor(&storage, dog_id, cat_id, &params)?;
/// if let Some(lca) = result.lca_id {
///     println!("LCA: {} at depth {}", lca, result.depth_from_a + result.depth_from_b);
/// }
/// ```
pub fn lowest_common_ancestor(
    storage: &GraphStorage,
    node_a: i64,
    node_b: i64,
    params: &EntailmentQueryParams,
) -> GraphResult<LcaResult> {
    let ball = PoincareBall::new(params.hyperbolic_config.clone());

    // Collect ancestors of node_a with their depths
    let ancestors_a = collect_ancestors(storage, node_a, params.max_depth, &ball)?;

    // Collect ancestors of node_b with their depths
    let ancestors_b = collect_ancestors(storage, node_b, params.max_depth, &ball)?;

    // Find intersection and pick the one with maximum hierarchy depth (most specific)
    let mut best_lca: Option<(i64, u32, u32, u32)> = None; // (id, depth_from_a, depth_from_b, hierarchy_depth)

    for (&ancestor_id, &(depth_a, hierarchy_depth_a)) in &ancestors_a {
        if let Some(&(depth_b, _)) = ancestors_b.get(&ancestor_id) {
            match &best_lca {
                None => {
                    best_lca = Some((ancestor_id, depth_a, depth_b, hierarchy_depth_a));
                }
                Some((_, _, _, best_depth)) => {
                    // Prefer higher hierarchy depth (more specific ancestor)
                    if hierarchy_depth_a > *best_depth {
                        best_lca = Some((ancestor_id, depth_a, depth_b, hierarchy_depth_a));
                    }
                }
            }
        }
    }

    match best_lca {
        Some((lca_id, depth_from_a, depth_from_b, _)) => {
            // Convert storage types to hyperbolic/entailment types
            let lca_point = storage
                .get_hyperbolic(lca_id)?
                .map(|p| storage_to_hyperbolic_point(&p));
            let lca_cone = match storage.get_cone(lca_id)? {
                Some(c) => Some(storage_to_entailment_cone(&c)?),
                None => None,
            };

            Ok(LcaResult {
                lca_id: Some(lca_id),
                lca_point,
                lca_cone,
                depth_from_a,
                depth_from_b,
            })
        }
        None => Ok(LcaResult {
            lca_id: None,
            lca_point: None,
            lca_cone: None,
            depth_from_a: 0,
            depth_from_b: 0,
        }),
    }
}
