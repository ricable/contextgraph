//! CRUD operations for contradiction management.
//!
//! Contains functions for checking, marking, and retrieving contradictions.

use std::time::{SystemTime, UNIX_EPOCH};

use tracing::info;
use uuid::Uuid;

use crate::error::{GraphError, GraphResult};
use crate::storage::edges::GraphEdge;
use crate::storage::GraphStorage;

use super::helpers::{generate_edge_id, infer_contradiction_type_from_edge, uuid_to_i64};
use super::types::{
    ContradictionResult, ContradictionType, Domain, EdgeType, NeurotransmitterWeights,
};

/// Check for contradiction between two specific nodes.
///
/// Looks for explicit CONTRADICTS edge between the nodes.
///
/// # Arguments
///
/// * `storage` - Graph storage backend
/// * `node_a` - First node UUID
/// * `node_b` - Second node UUID
///
/// # Returns
///
/// * `Some(ContradictionResult)` if explicit contradiction exists
/// * `None` if no explicit contradiction edge found
pub fn check_contradiction(
    storage: &GraphStorage,
    node_a: Uuid,
    node_b: Uuid,
) -> GraphResult<Option<ContradictionResult>> {
    let node_a_i64 = uuid_to_i64(&node_a);
    let edges = storage.get_outgoing_edges(node_a_i64)?;

    for edge in edges {
        if edge.edge_type == EdgeType::Contradicts && edge.target == node_b {
            return Ok(Some(ContradictionResult {
                contradicting_node_id: node_b,
                contradiction_type: infer_contradiction_type_from_edge(&edge),
                confidence: edge.confidence,
                semantic_similarity: 0.0,
                edge_weight: Some(edge.weight),
                has_explicit_edge: true,
                evidence: Vec::new(),
            }));
        }
    }

    Ok(None)
}

/// Mark two nodes as contradicting.
///
/// Creates bidirectional CONTRADICTS edges with inhibitory-heavy NT modulation.
///
/// # Arguments
///
/// * `storage` - Mutable graph storage backend
/// * `node_a` - First node UUID
/// * `node_b` - Second node UUID
/// * `contradiction_type` - Type of contradiction
/// * `confidence` - Confidence score [0, 1]
///
/// # Errors
///
/// * `GraphError::InvalidInput` - Self-contradiction or invalid confidence (FAIL FAST)
/// * `GraphError::Storage` - Storage write failed
pub fn mark_contradiction(
    storage: &GraphStorage,
    node_a: Uuid,
    node_b: Uuid,
    _contradiction_type: ContradictionType,
    confidence: f32,
) -> GraphResult<()> {
    // FAIL FAST validation
    if node_a == node_b {
        return Err(GraphError::InvalidInput(
            "Cannot create self-contradiction".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&confidence) {
        return Err(GraphError::InvalidInput(format!(
            "Confidence must be in [0, 1], got {}",
            confidence
        )));
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Create bidirectional contradiction edges using the factory method
    let edge_a_to_b = GraphEdge {
        id: generate_edge_id(),
        source: node_a,
        target: node_b,
        edge_type: EdgeType::Contradicts,
        weight: EdgeType::Contradicts.default_weight(),
        confidence,
        domain: Domain::General,
        // Inhibitory-heavy NT profile per M04-T26
        neurotransmitter_weights: NeurotransmitterWeights {
            excitatory: 0.2,
            inhibitory: 0.7,
            modulatory: 0.1,
        },
        is_amortized_shortcut: false,
        steering_reward: 0.5,
        traversal_count: 0,
        created_at: now,
        last_traversed_at: 0,
    };

    let edge_b_to_a = GraphEdge {
        id: generate_edge_id(),
        source: node_b,
        target: node_a,
        edge_type: EdgeType::Contradicts,
        weight: EdgeType::Contradicts.default_weight(),
        confidence,
        domain: Domain::General,
        neurotransmitter_weights: NeurotransmitterWeights {
            excitatory: 0.2,
            inhibitory: 0.7,
            modulatory: 0.1,
        },
        is_amortized_shortcut: false,
        steering_reward: 0.5,
        traversal_count: 0,
        created_at: now,
        last_traversed_at: 0,
    };

    storage.put_edge(&edge_a_to_b)?;
    storage.put_edge(&edge_b_to_a)?;

    info!(
        node_a = %node_a,
        node_b = %node_b,
        confidence = confidence,
        "Marked contradiction between nodes"
    );

    Ok(())
}

/// Get all contradictions for a node from storage.
///
/// Returns all explicit CONTRADICTS edges from the given node.
///
/// # Arguments
///
/// * `storage` - Graph storage backend
/// * `node_id` - Node UUID to get contradictions for
pub fn get_contradictions(
    storage: &GraphStorage,
    node_id: Uuid,
) -> GraphResult<Vec<ContradictionResult>> {
    let node_id_i64 = uuid_to_i64(&node_id);
    let edges = storage.get_outgoing_edges(node_id_i64)?;

    let results: Vec<ContradictionResult> = edges
        .iter()
        .filter(|e| e.edge_type == EdgeType::Contradicts)
        .map(|e| ContradictionResult {
            contradicting_node_id: e.target,
            contradiction_type: infer_contradiction_type_from_edge(e),
            confidence: e.confidence,
            semantic_similarity: 0.0,
            edge_weight: Some(e.weight),
            has_explicit_edge: true,
            evidence: Vec::new(),
        })
        .collect();

    Ok(results)
}
