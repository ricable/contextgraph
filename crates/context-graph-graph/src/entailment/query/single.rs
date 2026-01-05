//! Single-pair entailment operations.
//!
//! This module provides O(1) entailment checks between individual node pairs.
//!
//! # Performance Targets
//!
//! - Single containment check: <1ms (O(1) angle computation)
//!
//! # Constitution Reference
//!
//! - perf.latency.entailment_check: <1ms

use crate::config::HyperbolicConfig;
use crate::error::{GraphError, GraphResult};
use crate::hyperbolic::PoincareBall;
use crate::storage::GraphStorage;

use super::conversion::{storage_to_entailment_cone, storage_to_hyperbolic_point};

/// Check if node A is entailed by node B (A is an ancestor of B).
///
/// This is an O(1) operation using cone containment check.
///
/// # Definition
///
/// A is entailed by B iff B's hyperbolic point is contained in A's cone.
/// In other words, A is a more general concept that subsumes B.
///
/// # Arguments
///
/// * `storage` - GraphStorage for retrieving data
/// * `ancestor_id` - Potential ancestor node (more general)
/// * `descendant_id` - Potential descendant node (more specific)
/// * `config` - Hyperbolic configuration
///
/// # Returns
///
/// * `Ok(true)` - descendant is contained in ancestor's cone
/// * `Ok(false)` - descendant is NOT contained in ancestor's cone
/// * `Err(GraphError::MissingHyperbolicData)` - If required data is missing
///
/// # Performance
///
/// O(1) - single angle computation, target <1ms
///
/// # Example
///
/// ```ignore
/// // Check if "Animal" entails "Dog" (Dog is a kind of Animal)
/// let animal_id = 1;
/// let dog_id = 42;
/// let is_kind_of = is_entailed_by(&storage, animal_id, dog_id, &config)?;
/// assert!(is_kind_of); // Dog IS-A Animal
/// ```
pub fn is_entailed_by(
    storage: &GraphStorage,
    ancestor_id: i64,
    descendant_id: i64,
    config: &HyperbolicConfig,
) -> GraphResult<bool> {
    // FAIL FAST: Get ancestor's cone and convert
    let storage_ancestor_cone = storage.get_cone(ancestor_id)?.ok_or_else(|| {
        tracing::error!(node_id = ancestor_id, "Missing cone data for ancestor node");
        GraphError::MissingHyperbolicData(ancestor_id)
    })?;
    let ancestor_cone = storage_to_entailment_cone(&storage_ancestor_cone)?;

    // FAIL FAST: Get descendant's hyperbolic point and convert
    let storage_descendant_point = storage.get_hyperbolic(descendant_id)?.ok_or_else(|| {
        tracing::error!(
            node_id = descendant_id,
            "Missing hyperbolic data for descendant node"
        );
        GraphError::MissingHyperbolicData(descendant_id)
    })?;
    let descendant_point = storage_to_hyperbolic_point(&storage_descendant_point);

    let ball = PoincareBall::new(config.clone());
    let is_contained = ancestor_cone.contains(&descendant_point, &ball);

    tracing::trace!(
        ancestor_id = ancestor_id,
        descendant_id = descendant_id,
        is_entailed = is_contained,
        "Entailment check completed"
    );

    Ok(is_contained)
}

/// Get the membership score for a descendant relative to an ancestor.
///
/// This quantifies "how much" the descendant belongs to the ancestor's concept.
///
/// # Returns
///
/// Score in [0, 1]:
/// - 1.0 = fully contained in cone (strong IS-A relationship)
/// - <1.0 = partially outside cone (weak relationship)
/// - approaching 0 = far outside cone (no relationship)
///
/// # Formula (CANONICAL - DO NOT MODIFY)
///
/// - If angle <= aperture: score = 1.0
/// - If angle > aperture: score = exp(-2.0 * (angle - aperture))
///
/// # Arguments
///
/// * `storage` - GraphStorage for retrieving data
/// * `ancestor_id` - Ancestor node whose cone defines the concept
/// * `descendant_id` - Descendant node to score
/// * `config` - Hyperbolic configuration
///
/// # Returns
///
/// * `Ok(f32)` - Membership score in [0, 1]
/// * `Err(GraphError::MissingHyperbolicData)` - If required data is missing
///
/// # Performance
///
/// O(1) - single angle computation, target <1ms
pub fn entailment_score(
    storage: &GraphStorage,
    ancestor_id: i64,
    descendant_id: i64,
    config: &HyperbolicConfig,
) -> GraphResult<f32> {
    // FAIL FAST: Get ancestor's cone and convert
    let storage_ancestor_cone = storage.get_cone(ancestor_id)?.ok_or_else(|| {
        tracing::error!(node_id = ancestor_id, "Missing cone data for ancestor node");
        GraphError::MissingHyperbolicData(ancestor_id)
    })?;
    let ancestor_cone = storage_to_entailment_cone(&storage_ancestor_cone)?;

    // FAIL FAST: Get descendant's hyperbolic point and convert
    let storage_descendant_point = storage.get_hyperbolic(descendant_id)?.ok_or_else(|| {
        tracing::error!(
            node_id = descendant_id,
            "Missing hyperbolic data for descendant node"
        );
        GraphError::MissingHyperbolicData(descendant_id)
    })?;
    let descendant_point = storage_to_hyperbolic_point(&storage_descendant_point);

    let ball = PoincareBall::new(config.clone());
    let score = ancestor_cone.membership_score(&descendant_point, &ball);

    tracing::trace!(
        ancestor_id = ancestor_id,
        descendant_id = descendant_id,
        score = score,
        "Entailment score computed"
    );

    Ok(score)
}
