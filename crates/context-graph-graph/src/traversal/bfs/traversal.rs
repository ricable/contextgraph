//! BFS traversal algorithms.
//!
//! Core BFS implementation with domain modulation support.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::GraphResult;
use crate::storage::GraphStorage;

use super::types::{uuid_to_i64, BfsParams, BfsResult, Domain, NodeId};

/// Perform BFS traversal from a starting node.
///
/// # Arguments
/// * `storage` - Graph storage backend
/// * `start` - Starting node ID (i64)
/// * `params` - Traversal parameters
///
/// # Returns
/// * `Ok(BfsResult)` - Traversal results
/// * `Err(GraphError::Storage*)` - Storage access failed
///
/// # Performance
/// Target: <100ms for depth=6 on 10M node graph
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_graph::traversal::bfs::{bfs_traverse, BfsParams, Domain};
///
/// let params = BfsParams::default()
///     .max_depth(3)
///     .domain(Domain::Code)
///     .min_weight(0.3);
///
/// let result = bfs_traverse(&storage, start_node, params)?;
/// println!("Found {} nodes at {} depth levels",
///     result.node_count(), result.max_depth_reached);
///
/// // Access modulated weights
/// for edge in &result.edges {
///     let w = edge.get_modulated_weight(Domain::Code);
///     println!("Edge {} -> {} weight: {}", edge.source, edge.target, w);
/// }
/// ```
pub fn bfs_traverse(
    storage: &GraphStorage,
    start: NodeId,
    params: BfsParams,
) -> GraphResult<BfsResult> {
    // Pre-allocate with reasonable capacity
    let mut visited: HashSet<NodeId> = HashSet::with_capacity(params.max_nodes.min(10000));
    let mut frontier: VecDeque<(NodeId, usize)> = VecDeque::with_capacity(1000);
    let mut result_nodes: Vec<NodeId> = Vec::with_capacity(params.max_nodes.min(10000));
    let mut result_edges = if params.include_edges {
        Vec::with_capacity(params.max_nodes.min(10000))
    } else {
        Vec::new()
    };
    let mut depth_counts: HashMap<usize, usize> = HashMap::new();
    let mut max_depth_reached: usize = 0;
    let mut truncated = false;

    // Initialize with start node
    frontier.push_back((start, 0));
    visited.insert(start);

    while let Some((current_node, depth)) = frontier.pop_front() {
        // Check node limit BEFORE processing
        if result_nodes.len() >= params.max_nodes {
            truncated = true;
            log::debug!(
                "BFS truncated at {} nodes (limit: {})",
                result_nodes.len(),
                params.max_nodes
            );
            break;
        }

        // Add current node to results
        result_nodes.push(current_node);
        *depth_counts.entry(depth).or_insert(0) += 1;
        max_depth_reached = max_depth_reached.max(depth);

        // Don't expand if at max depth
        if depth >= params.max_depth {
            continue;
        }

        // Get full edges with Marblestone fields using get_outgoing_edges
        let edges = storage.get_outgoing_edges(current_node)?;

        for edge in edges {
            // Filter by edge type if specified
            if let Some(ref allowed_types) = params.edge_types {
                if !allowed_types.contains(&edge.edge_type) {
                    continue;
                }
            }

            // Get effective weight (with domain modulation if specified)
            let effective_weight = if let Some(domain) = params.domain_filter {
                edge.get_modulated_weight(domain)
            } else {
                edge.weight
            };

            // Filter by minimum weight
            if effective_weight < params.min_weight {
                continue;
            }

            // Convert UUID target to i64 for visited tracking
            let target_i64 = uuid_to_i64(&edge.target);

            // Skip if already visited
            if visited.contains(&target_i64) {
                continue;
            }

            // Add to visited and frontier
            visited.insert(target_i64);
            frontier.push_back((target_i64, depth + 1));

            // Collect edge if requested
            if params.include_edges {
                result_edges.push(edge);
            }
        }
    }

    log::debug!(
        "BFS complete: {} nodes, {} edges, max_depth={}",
        result_nodes.len(),
        result_edges.len(),
        max_depth_reached
    );

    Ok(BfsResult {
        nodes: result_nodes,
        edges: result_edges,
        depth_counts,
        start_node: start,
        max_depth_reached,
        truncated,
    })
}

/// Find shortest path between two nodes using BFS.
///
/// # Arguments
/// * `storage` - Graph storage backend
/// * `start` - Starting node ID
/// * `target` - Target node ID
/// * `max_depth` - Maximum search depth
///
/// # Returns
/// * `Ok(Some(path))` - Path from start to target (inclusive)
/// * `Ok(None)` - No path found within max_depth
pub fn bfs_shortest_path(
    storage: &GraphStorage,
    start: NodeId,
    target: NodeId,
    max_depth: usize,
) -> GraphResult<Option<Vec<NodeId>>> {
    if start == target {
        return Ok(Some(vec![start]));
    }

    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut frontier: VecDeque<(NodeId, usize)> = VecDeque::new();
    let mut parent: HashMap<NodeId, NodeId> = HashMap::new();

    frontier.push_back((start, 0));
    visited.insert(start);

    while let Some((current_node, depth)) = frontier.pop_front() {
        if depth >= max_depth {
            continue;
        }

        let edges = storage.get_outgoing_edges(current_node)?;

        for edge in edges {
            let target_i64 = uuid_to_i64(&edge.target);

            if visited.contains(&target_i64) {
                continue;
            }

            parent.insert(target_i64, current_node);
            visited.insert(target_i64);

            if target_i64 == target {
                // Reconstruct path
                let mut path = vec![target];
                let mut current = target;

                while let Some(&prev) = parent.get(&current) {
                    path.push(prev);
                    current = prev;
                }

                path.reverse();
                return Ok(Some(path));
            }

            frontier.push_back((target_i64, depth + 1));
        }
    }

    Ok(None)
}

/// Get all nodes within a given distance from start.
///
/// Convenience wrapper around bfs_traverse.
pub fn bfs_neighborhood(
    storage: &GraphStorage,
    start: NodeId,
    max_distance: usize,
) -> GraphResult<Vec<NodeId>> {
    let params = BfsParams::with_depth(max_distance).include_edges(false);
    let result = bfs_traverse(storage, start, params)?;
    Ok(result.nodes)
}

/// Get nodes within distance, filtered by domain.
///
/// Returns only nodes reachable via edges with weight >= min_weight
/// after domain modulation.
pub fn bfs_domain_neighborhood(
    storage: &GraphStorage,
    start: NodeId,
    max_distance: usize,
    domain: Domain,
    min_weight: f32,
) -> GraphResult<Vec<NodeId>> {
    let params = BfsParams::with_depth(max_distance)
        .domain(domain)
        .min_weight(min_weight)
        .include_edges(false);
    let result = bfs_traverse(storage, start, params)?;
    Ok(result.nodes)
}
