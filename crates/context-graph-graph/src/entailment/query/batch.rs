//! Batch entailment operations.
//!
//! This module provides efficient batch checking for multiple node pairs.
//!
//! # Performance Targets
//!
//! - Batch check (1000 pairs): <100ms

use std::collections::HashMap;

use crate::config::HyperbolicConfig;
use crate::entailment::cones::EntailmentCone;
use crate::error::{GraphError, GraphResult};
use crate::hyperbolic::poincare::PoincarePoint;
use crate::hyperbolic::PoincareBall;
use crate::storage::GraphStorage;

use super::conversion::{storage_to_entailment_cone, storage_to_hyperbolic_point};
use super::types::BatchEntailmentResult;

/// Check entailment for multiple (ancestor, descendant) pairs.
///
/// More efficient than individual calls due to reduced storage round-trips
/// when nodes appear in multiple pairs.
///
/// # Arguments
///
/// * `storage` - GraphStorage for retrieving data
/// * `pairs` - Vec of (ancestor_id, descendant_id) pairs to check
/// * `config` - Hyperbolic configuration
///
/// # Returns
///
/// * `Ok(Vec<BatchEntailmentResult>)` - Results for all pairs (same order as input)
/// * `Err(GraphError::MissingHyperbolicData)` - If any required data is missing
///
/// # Performance
///
/// O(n) where n = number of pairs, target <100ms for 1000 pairs
///
/// # Example
///
/// ```ignore
/// let pairs = vec![
///     (1, 42),  // Animal -> Dog
///     (1, 43),  // Animal -> Cat
///     (42, 100), // Dog -> Poodle
/// ];
/// let results = entailment_check_batch(&storage, &pairs, &config)?;
/// ```
pub fn entailment_check_batch(
    storage: &GraphStorage,
    pairs: &[(i64, i64)],
    config: &HyperbolicConfig,
) -> GraphResult<Vec<BatchEntailmentResult>> {
    let ball = PoincareBall::new(config.clone());
    let mut results = Vec::with_capacity(pairs.len());

    // Cache for repeated lookups (store converted types)
    let mut cone_cache: HashMap<i64, EntailmentCone> = HashMap::new();
    let mut point_cache: HashMap<i64, PoincarePoint> = HashMap::new();

    for &(ancestor_id, descendant_id) in pairs {
        // Get or cache ancestor cone (with conversion)
        let ancestor_cone = if let Some(cone) = cone_cache.get(&ancestor_id) {
            cone.clone()
        } else {
            let storage_cone = storage.get_cone(ancestor_id)?.ok_or_else(|| {
                tracing::error!(node_id = ancestor_id, "Missing cone data in batch check");
                GraphError::MissingHyperbolicData(ancestor_id)
            })?;
            let cone = storage_to_entailment_cone(&storage_cone)?;
            cone_cache.insert(ancestor_id, cone.clone());
            cone
        };

        // Get or cache descendant point (with conversion)
        let descendant_point = if let Some(point) = point_cache.get(&descendant_id) {
            point.clone()
        } else {
            let storage_point = storage.get_hyperbolic(descendant_id)?.ok_or_else(|| {
                tracing::error!(
                    node_id = descendant_id,
                    "Missing hyperbolic data in batch check"
                );
                GraphError::MissingHyperbolicData(descendant_id)
            })?;
            let point = storage_to_hyperbolic_point(&storage_point);
            point_cache.insert(descendant_id, point.clone());
            point
        };

        // Compute containment and score
        let is_entailed = ancestor_cone.contains(&descendant_point, &ball);
        let score = ancestor_cone.membership_score(&descendant_point, &ball);

        results.push(BatchEntailmentResult {
            ancestor_id,
            descendant_id,
            is_entailed,
            score,
        });
    }

    tracing::debug!(
        pair_count = pairs.len(),
        cache_hits = cone_cache.len() + point_cache.len(),
        "Batch entailment check completed"
    );

    Ok(results)
}
