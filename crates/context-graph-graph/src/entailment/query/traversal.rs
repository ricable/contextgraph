//! BFS traversal for entailment queries.
//!
//! This module provides graph traversal operations for finding ancestors
//! and descendants in the entailment hierarchy.
//!
//! # Performance Targets
//!
//! - BFS + filter (depth 3): <10ms

use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::GraphResult;
use crate::hyperbolic::PoincareBall;
use crate::storage::GraphStorage;

use super::conversion::{storage_to_entailment_cone, storage_to_hyperbolic_point};
use super::types::{EntailmentDirection, EntailmentQueryParams, EntailmentResult};

/// Query for entailment relationships starting from a node.
///
/// Uses BFS to generate candidates, then filters by cone containment.
///
/// # Algorithm
///
/// 1. Initialize BFS queue with query node
/// 2. For each node at current depth:
///    - Get neighbors from adjacency list
///    - For Ancestors: check if neighbor's cone contains query point
///    - For Descendants: check if query's cone contains neighbor point
/// 3. Filter results by membership score >= min_membership_score
/// 4. Sort by membership score descending, limit to max_results
///
/// # Arguments
///
/// * `storage` - GraphStorage for retrieving hyperbolic embeddings and cones
/// * `query_node` - Starting node ID for the query
/// * `direction` - Whether to find ancestors or descendants
/// * `params` - Query parameters (depth, limits, thresholds)
///
/// # Returns
///
/// * `Ok(Vec<EntailmentResult>)` - Sorted results by membership score
/// * `Err(GraphError::NodeNotFound)` - If query node doesn't exist
/// * `Err(GraphError::MissingHyperbolicData)` - If required data is missing
///
/// # Performance
///
/// O(n * d) where n = nodes visited by BFS, d = constant (angle computation)
/// Target: <10ms for depth 3 on typical graph
///
/// # Example
///
/// ```ignore
/// use context_graph_graph::entailment::query::{
///     entailment_query, EntailmentDirection, EntailmentQueryParams
/// };
///
/// let results = entailment_query(
///     &storage,
///     node_id,
///     EntailmentDirection::Ancestors,
///     &EntailmentQueryParams::default(),
/// )?;
///
/// for result in results {
///     println!("Node {} entails query with score {}", result.node_id, result.membership_score);
/// }
/// ```
pub fn entailment_query(
    storage: &GraphStorage,
    query_node: i64,
    direction: EntailmentDirection,
    params: &EntailmentQueryParams,
) -> GraphResult<Vec<EntailmentResult>> {
    use crate::error::GraphError;

    // FAIL FAST: Get query node's hyperbolic data and convert to hyperbolic types
    let storage_query_point = storage.get_hyperbolic(query_node)?.ok_or_else(|| {
        tracing::error!(
            node_id = query_node,
            "Missing hyperbolic data for query node"
        );
        GraphError::MissingHyperbolicData(query_node)
    })?;
    let query_point = storage_to_hyperbolic_point(&storage_query_point);

    let storage_query_cone = storage.get_cone(query_node)?.ok_or_else(|| {
        tracing::error!(node_id = query_node, "Missing cone data for query node");
        GraphError::NodeNotFound(query_node.to_string())
    })?;
    let query_cone = storage_to_entailment_cone(&storage_query_cone)?;

    let ball = PoincareBall::new(params.hyperbolic_config.clone());

    // BFS for candidate generation
    let mut visited: HashSet<i64> = HashSet::new();
    let mut queue: VecDeque<(i64, u32)> = VecDeque::new(); // (node_id, depth)
    let mut results: Vec<EntailmentResult> = Vec::new();

    visited.insert(query_node);
    queue.push_back((query_node, 0));

    while let Some((current_id, current_depth)) = queue.pop_front() {
        // Stop if we've exceeded max depth
        if current_depth >= params.max_depth {
            continue;
        }

        // Get neighbors
        let neighbors = storage.get_adjacency(current_id)?;

        for edge in neighbors {
            let neighbor_id = edge.target;

            // Skip already visited
            if visited.contains(&neighbor_id) {
                continue;
            }
            visited.insert(neighbor_id);

            // Get neighbor's hyperbolic data and convert
            let storage_neighbor_point = match storage.get_hyperbolic(neighbor_id)? {
                Some(p) => p,
                None => {
                    tracing::debug!(
                        node_id = neighbor_id,
                        "Skipping node without hyperbolic data"
                    );
                    continue;
                }
            };
            let neighbor_point = storage_to_hyperbolic_point(&storage_neighbor_point);

            let storage_neighbor_cone = match storage.get_cone(neighbor_id)? {
                Some(c) => c,
                None => {
                    tracing::debug!(node_id = neighbor_id, "Skipping node without cone data");
                    continue;
                }
            };
            let neighbor_cone = storage_to_entailment_cone(&storage_neighbor_cone)?;

            // Check containment based on direction
            let (is_entailed, score) = match direction {
                EntailmentDirection::Ancestors => {
                    // Ancestor's cone should contain query point
                    let contains = neighbor_cone.contains(&query_point, &ball);
                    let score = neighbor_cone.membership_score(&query_point, &ball);
                    (contains, score)
                }
                EntailmentDirection::Descendants => {
                    // Query's cone should contain neighbor point
                    let contains = query_cone.contains(&neighbor_point, &ball);
                    let score = query_cone.membership_score(&neighbor_point, &ball);
                    (contains, score)
                }
            };

            // Only include if membership score meets threshold
            if score >= params.min_membership_score {
                results.push(EntailmentResult {
                    node_id: neighbor_id,
                    point: neighbor_point,
                    cone: neighbor_cone.clone(),
                    membership_score: score,
                    depth: neighbor_cone.depth,
                    is_direct: current_depth == 0,
                });
            }

            // Continue BFS even if not entailed (might have descendants that are)
            if is_entailed || score > 0.0 {
                queue.push_back((neighbor_id, current_depth + 1));
            }
        }
    }

    // Sort by membership score (descending)
    results.sort_by(|a, b| {
        b.membership_score
            .partial_cmp(&a.membership_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Limit to max_results
    results.truncate(params.max_results);

    Ok(results)
}

/// Helper: Collect ancestors of a node using BFS with cone containment.
///
/// Returns HashMap of ancestor_id -> (bfs_depth, hierarchy_depth)
pub(crate) fn collect_ancestors(
    storage: &GraphStorage,
    start_node: i64,
    max_depth: u32,
    ball: &PoincareBall,
) -> GraphResult<HashMap<i64, (u32, u32)>> {
    let mut ancestors: HashMap<i64, (u32, u32)> = HashMap::new();

    // Get start node's point and convert
    let start_point = match storage.get_hyperbolic(start_node)? {
        Some(p) => storage_to_hyperbolic_point(&p),
        None => return Ok(ancestors), // No ancestors if no hyperbolic data
    };

    let mut visited: HashSet<i64> = HashSet::new();
    let mut queue: VecDeque<(i64, u32)> = VecDeque::new();

    visited.insert(start_node);
    queue.push_back((start_node, 0));

    while let Some((current_id, current_depth)) = queue.pop_front() {
        if current_depth >= max_depth {
            continue;
        }

        // Get neighbors (potential ancestors)
        let neighbors = storage.get_adjacency(current_id)?;

        for edge in neighbors {
            let neighbor_id = edge.target;

            if visited.contains(&neighbor_id) {
                continue;
            }
            visited.insert(neighbor_id);

            // Get neighbor's cone and convert to check if it's an ancestor
            let storage_neighbor_cone = match storage.get_cone(neighbor_id)? {
                Some(c) => c,
                None => continue,
            };
            let neighbor_cone = match storage_to_entailment_cone(&storage_neighbor_cone) {
                Ok(c) => c,
                Err(_) => continue,
            };

            // Ancestor's cone should contain the start point
            if neighbor_cone.contains(&start_point, ball) {
                ancestors.insert(neighbor_id, (current_depth + 1, neighbor_cone.depth));
            }

            // Continue BFS regardless
            queue.push_back((neighbor_id, current_depth + 1));
        }
    }

    Ok(ancestors)
}
