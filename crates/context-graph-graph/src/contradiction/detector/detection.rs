//! Core contradiction detection algorithm.
//!
//! Combines semantic similarity search with explicit CONTRADICTS edges
//! to identify conflicting knowledge in the graph.

use std::collections::HashMap;

use tracing::info;
use uuid::Uuid;

use crate::error::{GraphError, GraphResult};
use crate::index::FaissGpuIndex;
use crate::search::{semantic_search, NodeMetadataProvider, SearchFilters};
use crate::storage::GraphStorage;
use crate::traversal::bfs::{bfs_traverse, BfsParams};

use super::helpers::{
    compute_confidence, infer_contradiction_type_from_edge, infer_type_from_similarity, uuid_to_i64,
};
use super::types::{CandidateInfo, ContradictionParams, ContradictionResult, EdgeType};

/// Detect contradictions for a given node.
///
/// Combines semantic similarity search with explicit CONTRADICTS edges
/// to find potentially conflicting knowledge.
///
/// # Algorithm
///
/// 1. Validate inputs (FAIL FAST)
/// 2. Semantic search for similar nodes (k candidates)
/// 3. BFS to find nodes with CONTRADICTS edges
/// 4. Combine and score contradictions
/// 5. Classify contradiction types
/// 6. Filter by threshold
/// 7. Sort by confidence descending
///
/// # Arguments
///
/// * `index` - FAISS GPU index for semantic search
/// * `storage` - Graph storage backend
/// * `node_id` - Node to check for contradictions (UUID)
/// * `node_embedding` - Embedding as raw f32 slice
/// * `params` - Detection parameters
/// * `metadata` - Optional metadata provider for filtering
///
/// # Returns
///
/// Vector of contradictions above threshold, sorted by confidence descending.
///
/// # Errors
///
/// * `GraphError::InvalidInput` - Invalid parameters (FAIL FAST)
/// * `GraphError::FaissSearchFailed` - FAISS search failed
/// * `GraphError::Storage` - Storage access failed
///
/// # Example
///
/// ```ignore
/// use context_graph_graph::contradiction::{contradiction_detect, ContradictionParams};
///
/// let params = ContradictionParams::default().high_sensitivity();
/// let results = contradiction_detect(
///     &index,
///     &storage,
///     node_id,
///     &embedding,
///     params,
///     None,
/// )?;
///
/// for result in results {
///     println!("Contradiction: {} (confidence: {})",
///         result.contradicting_node_id,
///         result.confidence);
/// }
/// ```
pub fn contradiction_detect<M: NodeMetadataProvider>(
    index: &FaissGpuIndex,
    storage: &GraphStorage,
    node_id: Uuid,
    node_embedding: &[f32],
    params: ContradictionParams,
    metadata: Option<&M>,
) -> GraphResult<Vec<ContradictionResult>> {
    // FAIL FAST validation
    if node_embedding.is_empty() {
        return Err(GraphError::InvalidInput(
            "node_embedding cannot be empty".to_string(),
        ));
    }

    params.validate()?;

    let mut candidates: HashMap<Uuid, CandidateInfo> = HashMap::new();

    // Step 1: Semantic search for similar nodes
    let filters = SearchFilters::new().with_min_similarity(params.min_similarity);

    let semantic_results = semantic_search(
        index,
        node_embedding,
        params.semantic_k,
        Some(filters),
        metadata,
    )?;

    for item in semantic_results.items.iter() {
        // Skip if no node_id resolved or if it's the query node itself
        if let Some(item_node_id) = item.node_id {
            if item_node_id != node_id {
                candidates.insert(
                    item_node_id,
                    CandidateInfo {
                        semantic_similarity: item.similarity,
                        has_explicit_edge: false,
                        edge_weight: None,
                        edge_type: None,
                    },
                );
            }
        }
    }

    // Step 2: BFS to find CONTRADICTS edges
    // Convert UUID to i64 for storage operations
    let node_id_i64 = uuid_to_i64(&node_id);

    let bfs_params = BfsParams::default()
        .max_depth(params.graph_depth)
        .edge_types(vec![EdgeType::Contradicts]);

    let bfs_result = bfs_traverse(storage, node_id_i64, bfs_params)?;

    for edge in bfs_result.edges.iter() {
        if edge.edge_type == EdgeType::Contradicts {
            let target = if edge.source == node_id {
                edge.target
            } else {
                edge.source
            };

            candidates
                .entry(target)
                .and_modify(|info| {
                    info.has_explicit_edge = true;
                    info.edge_weight = Some(edge.weight);
                    info.edge_type = Some(infer_contradiction_type_from_edge(edge));
                })
                .or_insert(CandidateInfo {
                    semantic_similarity: 0.0,
                    has_explicit_edge: true,
                    edge_weight: Some(edge.weight),
                    edge_type: Some(infer_contradiction_type_from_edge(edge)),
                });
        }
    }

    // Step 3: Score and classify contradictions
    let mut results: Vec<ContradictionResult> = Vec::with_capacity(candidates.len());

    for (candidate_id, info) in candidates {
        let confidence = compute_confidence(&info, &params);

        if confidence >= params.threshold {
            let contradiction_type = info
                .edge_type
                .unwrap_or_else(|| infer_type_from_similarity(info.semantic_similarity));

            results.push(ContradictionResult {
                contradicting_node_id: candidate_id,
                contradiction_type,
                confidence,
                semantic_similarity: info.semantic_similarity,
                edge_weight: info.edge_weight,
                has_explicit_edge: info.has_explicit_edge,
                evidence: Vec::new(),
            });
        }
    }

    // Sort by confidence descending
    results.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    info!(
        node_id = %node_id,
        candidates_found = results.len(),
        threshold = params.threshold,
        "Contradiction detection complete"
    );

    Ok(results)
}
