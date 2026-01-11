//! Hebbian Learning Implementation
//!
//! Implements the Hebbian learning rule for NREM phase memory consolidation:
//! "Neurons that fire together wire together"
//!
//! Formula: dw_ij = eta * phi_i * phi_j
//!
//! Constitution Reference: docs2/constitution.yaml Section dream.phases.nrem

use std::collections::{HashMap, HashSet};
use tracing::{debug, trace};
use uuid::Uuid;

use super::types::{HebbianConfig, NodeActivation};

/// Result of applying Hebbian updates to a set of edges.
#[derive(Debug, Clone, Default)]
pub struct HebbianUpdateResult {
    /// Number of edges strengthened (weight increased)
    pub edges_strengthened: usize,
    /// Number of edges weakened (weight decreased due to decay)
    pub edges_weakened: usize,
    /// Number of edges marked for pruning (below floor)
    pub edges_to_prune: usize,
    /// Sum of absolute weight deltas
    pub total_delta: f32,
    /// Average weight delta
    pub average_delta: f32,
    /// Maximum weight delta observed
    pub max_delta: f32,
    /// Edge IDs marked for pruning (source, target)
    pub prune_candidates: Vec<(Uuid, Uuid)>,
}

/// A single edge update computed by Hebbian learning.
#[derive(Debug, Clone, Copy)]
pub struct EdgeUpdate {
    /// Source node ID
    pub source_id: Uuid,
    /// Target node ID
    pub target_id: Uuid,
    /// Current weight before update
    pub old_weight: f32,
    /// New weight after update
    pub new_weight: f32,
    /// Raw delta from Hebbian formula
    pub hebbian_delta: f32,
    /// Whether this edge should be pruned
    pub should_prune: bool,
}

impl EdgeUpdate {
    /// Compute the absolute change in weight.
    pub fn weight_change(&self) -> f32 {
        (self.new_weight - self.old_weight).abs()
    }

    /// Check if weight was strengthened.
    pub fn was_strengthened(&self) -> bool {
        self.new_weight > self.old_weight
    }

    /// Check if weight was weakened.
    pub fn was_weakened(&self) -> bool {
        self.new_weight < self.old_weight
    }
}

/// Hebbian learning engine for NREM phase.
///
/// Manages the weight update process for co-activated node pairs.
#[derive(Debug, Clone)]
pub struct HebbianEngine {
    config: HebbianConfig,
    activations: HashMap<Uuid, f32>,
    edge_weights: HashMap<(Uuid, Uuid), f32>,
    updated_edges: Vec<EdgeUpdate>,
    stats: HebbianUpdateResult,
}

impl HebbianEngine {
    /// Create a new Hebbian engine with the given configuration.
    pub fn new(config: HebbianConfig) -> Self {
        config.validate(); // Fail fast on invalid config
        Self {
            config,
            activations: HashMap::new(),
            edge_weights: HashMap::new(),
            updated_edges: Vec::new(),
            stats: HebbianUpdateResult::default(),
        }
    }

    /// Create with constitution defaults.
    pub fn with_defaults() -> Self {
        Self::new(HebbianConfig::default())
    }

    /// Set activations for a batch of nodes.
    pub fn set_activations(&mut self, activations: &[NodeActivation]) {
        self.activations.clear();
        for activation in activations {
            if activation.is_significant() {
                self.activations.insert(activation.node_id, activation.phi);
            }
        }
        debug!(
            "Set {} significant activations for Hebbian update",
            self.activations.len()
        );
    }

    /// Load current edge weights for processing.
    pub fn load_edges(&mut self, edges: &[(Uuid, Uuid, f32)]) {
        self.edge_weights.clear();
        for (source, target, weight) in edges {
            self.edge_weights.insert((*source, *target), *weight);
        }
        debug!("Loaded {} edges for Hebbian update", self.edge_weights.len());
    }

    /// Compute Hebbian updates for all loaded edges.
    ///
    /// Applies: dw_ij = eta * phi_i * phi_j
    ///
    /// Returns vector of edge updates to be applied.
    pub fn compute_updates(&mut self) -> Vec<EdgeUpdate> {
        self.updated_edges.clear();
        self.stats = HebbianUpdateResult::default();

        let mut total_delta = 0.0f32;
        let mut max_delta = 0.0f32;

        for ((source, target), current_weight) in &self.edge_weights {
            let phi_i = self.activations.get(source).copied().unwrap_or(0.0);
            let phi_j = self.activations.get(target).copied().unwrap_or(0.0);

            // Hebbian delta: dw = eta * phi_i * phi_j
            let hebbian_delta = self.compute_delta(phi_i, phi_j);

            // Apply decay: w_decayed = w * (1 - decay)
            let decayed = current_weight * (1.0 - self.config.weight_decay);

            // New weight with bounds
            let new_weight =
                (decayed + hebbian_delta).clamp(self.config.weight_floor, self.config.weight_cap);

            // Check for pruning
            let should_prune = new_weight <= self.config.weight_floor;

            let update = EdgeUpdate {
                source_id: *source,
                target_id: *target,
                old_weight: *current_weight,
                new_weight,
                hebbian_delta,
                should_prune,
            };

            // Update statistics
            if update.was_strengthened() {
                self.stats.edges_strengthened += 1;
            } else if update.was_weakened() {
                self.stats.edges_weakened += 1;
            }

            if should_prune {
                self.stats.edges_to_prune += 1;
                self.stats.prune_candidates.push((*source, *target));
            }

            let delta = update.weight_change();
            total_delta += delta;
            max_delta = max_delta.max(delta);

            trace!(
                "Edge ({}, {}): {} -> {} (delta={:.6})",
                source,
                target,
                current_weight,
                new_weight,
                hebbian_delta
            );

            self.updated_edges.push(update);
        }

        // Finalize statistics
        if !self.updated_edges.is_empty() {
            self.stats.total_delta = total_delta;
            self.stats.average_delta = total_delta / self.updated_edges.len() as f32;
            self.stats.max_delta = max_delta;
        }

        debug!(
            "Computed {} updates: {} strengthened, {} weakened, {} to prune",
            self.updated_edges.len(),
            self.stats.edges_strengthened,
            self.stats.edges_weakened,
            self.stats.edges_to_prune
        );

        self.updated_edges.clone()
    }

    /// Compute the Hebbian delta for a pair of activations.
    ///
    /// Formula: dw = eta * phi_i * phi_j
    #[inline]
    pub fn compute_delta(&self, phi_i: f32, phi_j: f32) -> f32 {
        self.config.learning_rate * phi_i * phi_j
    }

    /// Get the current statistics.
    pub fn stats(&self) -> &HebbianUpdateResult {
        &self.stats
    }

    /// Get the configuration.
    pub fn config(&self) -> &HebbianConfig {
        &self.config
    }

    /// Reset engine state for a new cycle.
    pub fn reset(&mut self) {
        self.activations.clear();
        self.edge_weights.clear();
        self.updated_edges.clear();
        self.stats = HebbianUpdateResult::default();
    }
}

/// Select memories for replay with recency bias.
///
/// Prioritizes recent memories while maintaining diversity.
///
/// # Arguments
/// * `memories` - List of (memory_id, timestamp_ms, phi_value) tuples
/// * `recency_bias` - How much to favor recent memories [0.0, 1.0]
/// * `limit` - Maximum number of memories to select
///
/// # Returns
/// Selected memory IDs for replay
pub fn select_replay_memories(
    memories: &[(Uuid, u64, f32)],
    recency_bias: f32,
    limit: usize,
) -> Vec<Uuid> {
    if memories.is_empty() || limit == 0 {
        return Vec::new();
    }

    let recency_bias = recency_bias.clamp(0.0, 1.0);

    let min_time = memories.iter().map(|(_, t, _)| *t).min().unwrap_or(0);
    let max_time = memories.iter().map(|(_, t, _)| *t).max().unwrap_or(1);
    let time_range = (max_time - min_time).max(1) as f32;

    let mut scored: Vec<(Uuid, f32)> = memories
        .iter()
        .map(|(id, timestamp, phi)| {
            let recency = (*timestamp - min_time) as f32 / time_range;
            let score = recency_bias * recency + (1.0 - recency_bias) * phi;
            (*id, score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(limit).map(|(id, _)| id).collect()
}

/// Find co-activated node pairs from a set of activations.
///
/// Two nodes are co-activated if both have phi > threshold.
///
/// # Arguments
/// * `activations` - Map of node_id -> phi value
/// * `threshold` - Minimum phi for a node to be considered active
///
/// # Returns
/// Set of (source, target) pairs where both nodes are active
pub fn find_coactivated_pairs(
    activations: &HashMap<Uuid, f32>,
    threshold: f32,
) -> HashSet<(Uuid, Uuid)> {
    let active_nodes: Vec<Uuid> = activations
        .iter()
        .filter(|(_, &phi)| phi > threshold)
        .map(|(id, _)| *id)
        .collect();

    let mut pairs = HashSet::new();

    // Create all pairs (ordered to avoid duplicates)
    for (i, &node_a) in active_nodes.iter().enumerate() {
        for &node_b in active_nodes.iter().skip(i + 1) {
            if node_a < node_b {
                pairs.insert((node_a, node_b));
            } else {
                pairs.insert((node_b, node_a));
            }
        }
    }

    pairs
}

/// Apply Kuramoto coupling for neural synchronization.
///
/// Formula: d(theta_i)/dt = omega_i + (K/N) * sum_j(sin(theta_j - theta_i))
///
/// Constitution: coupling_strength K = 0.9 during NREM (NOT 10.0)
///
/// # Arguments
/// * `phases` - Current phase values (theta) for each node
/// * `coupling_strength` - Kuramoto K parameter (0.9 per constitution)
/// * `dt` - Time step
///
/// # Returns
/// Updated phase values
pub fn kuramoto_coupling(
    phases: &HashMap<Uuid, f32>,
    coupling_strength: f32,
    dt: f32,
) -> HashMap<Uuid, f32> {
    let n = phases.len() as f32;
    if n < 2.0 {
        return phases.clone();
    }

    let mut new_phases = HashMap::new();

    for (node_i, &theta_i) in phases {
        let coupling_sum: f32 = phases.values().map(|&theta_j| (theta_j - theta_i).sin()).sum();

        let d_theta = (coupling_strength / n) * coupling_sum;
        let new_theta = theta_i + d_theta * dt;

        // Wrap to [0, 2*PI]
        let two_pi = 2.0 * std::f32::consts::PI;
        let wrapped = new_theta % two_pi;
        new_phases.insert(
            *node_i,
            if wrapped < 0.0 {
                wrapped + two_pi
            } else {
                wrapped
            },
        );
    }

    new_phases
}

/// Compute Kuramoto order parameter (synchronization measure).
///
/// r = |1/N * sum_j(e^(i*theta_j))|
///
/// Returns value in [0, 1]:
/// - 0 = completely desynchronized
/// - 1 = perfectly synchronized
pub fn kuramoto_order_parameter(phases: &HashMap<Uuid, f32>) -> f32 {
    if phases.is_empty() {
        return 0.0;
    }

    let n = phases.len() as f32;
    let (sum_cos, sum_sin): (f32, f32) = phases
        .values()
        .map(|&theta| (theta.cos(), theta.sin()))
        .fold((0.0, 0.0), |(sc, ss), (c, s)| (sc + c, ss + s));

    let mean_cos = sum_cos / n;
    let mean_sin = sum_sin / n;

    (mean_cos * mean_cos + mean_sin * mean_sin).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // HebbianEngine Tests
    // ============================================================

    #[test]
    fn test_hebbian_engine_creation() {
        let engine = HebbianEngine::with_defaults();
        assert_eq!(engine.config().learning_rate, 0.01);
        assert_eq!(engine.config().weight_decay, 0.001);
        assert_eq!(engine.config().weight_floor, 0.05);
        assert_eq!(engine.config().weight_cap, 1.0);
        assert_eq!(engine.config().coupling_strength, 0.9);
    }

    #[test]
    fn test_hebbian_delta_calculation() {
        let engine = HebbianEngine::with_defaults();
        // eta=0.01, phi_i=0.8, phi_j=0.9
        // delta = 0.01 * 0.8 * 0.9 = 0.0072
        let delta = engine.compute_delta(0.8, 0.9);
        assert!((delta - 0.0072).abs() < 1e-6);
    }

    #[test]
    fn test_hebbian_delta_zero_activation() {
        let engine = HebbianEngine::with_defaults();
        assert_eq!(engine.compute_delta(0.0, 0.9), 0.0);
        assert_eq!(engine.compute_delta(0.8, 0.0), 0.0);
    }

    #[test]
    fn test_hebbian_update_strengthening() {
        let mut engine = HebbianEngine::with_defaults();
        let node_a = Uuid::new_v4();
        let node_b = Uuid::new_v4();

        engine.set_activations(&[
            NodeActivation::new(node_a, 0.8),
            NodeActivation::new(node_b, 0.9),
        ]);
        engine.load_edges(&[(node_a, node_b, 0.5)]);

        let updates = engine.compute_updates();

        assert_eq!(updates.len(), 1);
        let update = &updates[0];
        assert!(update.was_strengthened(), "Weight should be strengthened");
        assert!(!update.should_prune);
    }

    #[test]
    fn test_hebbian_update_decay_only() {
        let mut engine = HebbianEngine::with_defaults();
        let node_a = Uuid::new_v4();
        let node_b = Uuid::new_v4();

        // Zero activations = no Hebbian update, only decay
        engine.set_activations(&[]);
        engine.load_edges(&[(node_a, node_b, 0.5)]);

        let updates = engine.compute_updates();

        assert_eq!(updates.len(), 1);
        let update = &updates[0];
        assert!(update.was_weakened(), "Weight should decay");
    }

    #[test]
    fn test_hebbian_update_pruning() {
        let mut engine = HebbianEngine::with_defaults();
        let node_a = Uuid::new_v4();
        let node_b = Uuid::new_v4();

        // Start at floor, decay triggers prune
        engine.set_activations(&[]);
        engine.load_edges(&[(node_a, node_b, 0.05)]);

        let updates = engine.compute_updates();

        assert_eq!(updates.len(), 1);
        let update = &updates[0];
        assert!(update.should_prune, "Edge at floor should be pruned");
    }

    #[test]
    fn test_hebbian_weight_cap() {
        let mut engine = HebbianEngine::with_defaults();
        let node_a = Uuid::new_v4();
        let node_b = Uuid::new_v4();

        engine.set_activations(&[
            NodeActivation::new(node_a, 1.0),
            NodeActivation::new(node_b, 1.0),
        ]);
        engine.load_edges(&[(node_a, node_b, 0.99)]);

        let updates = engine.compute_updates();

        assert_eq!(updates.len(), 1);
        assert!(
            updates[0].new_weight <= 1.0,
            "Weight should be capped at 1.0"
        );
    }

    // ============================================================
    // Memory Selection Tests
    // ============================================================

    #[test]
    fn test_select_replay_memories_recency() {
        let mem1 = Uuid::new_v4();
        let mem2 = Uuid::new_v4();
        let mem3 = Uuid::new_v4();

        let memories = vec![(mem1, 1000, 0.5), (mem2, 2000, 0.5), (mem3, 3000, 0.5)];

        // High recency bias should prefer recent
        let selected = select_replay_memories(&memories, 0.9, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0], mem3); // Most recent first
    }

    #[test]
    fn test_select_replay_memories_phi() {
        let mem1 = Uuid::new_v4();
        let mem2 = Uuid::new_v4();
        let mem3 = Uuid::new_v4();

        let memories = vec![
            (mem1, 1000, 0.9), // Old but high phi
            (mem2, 2000, 0.1),
            (mem3, 3000, 0.5),
        ];

        // Low recency bias should prefer high phi
        let selected = select_replay_memories(&memories, 0.1, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0], mem1); // Highest phi first
    }

    #[test]
    fn test_select_replay_memories_empty() {
        let selected = select_replay_memories(&[], 0.8, 10);
        assert!(selected.is_empty());
    }

    // ============================================================
    // Co-activation Tests
    // ============================================================

    #[test]
    fn test_find_coactivated_pairs() {
        let node_a = Uuid::new_v4();
        let node_b = Uuid::new_v4();
        let node_c = Uuid::new_v4();

        let mut activations = HashMap::new();
        activations.insert(node_a, 0.8);
        activations.insert(node_b, 0.9);
        activations.insert(node_c, 0.05); // Below threshold

        let pairs = find_coactivated_pairs(&activations, 0.1);

        assert_eq!(pairs.len(), 1); // Only (a, b)
    }

    #[test]
    fn test_find_coactivated_pairs_all_active() {
        let node_a = Uuid::new_v4();
        let node_b = Uuid::new_v4();
        let node_c = Uuid::new_v4();

        let mut activations = HashMap::new();
        activations.insert(node_a, 0.8);
        activations.insert(node_b, 0.9);
        activations.insert(node_c, 0.7);

        let pairs = find_coactivated_pairs(&activations, 0.1);

        assert_eq!(pairs.len(), 3); // (a,b), (a,c), (b,c)
    }

    // ============================================================
    // Kuramoto Tests
    // ============================================================

    #[test]
    fn test_kuramoto_order_parameter_synchronized() {
        let mut phases = HashMap::new();
        phases.insert(Uuid::new_v4(), 0.0);
        phases.insert(Uuid::new_v4(), 0.0);
        phases.insert(Uuid::new_v4(), 0.0);

        let r = kuramoto_order_parameter(&phases);
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_kuramoto_order_parameter_desynchronized() {
        let mut phases = HashMap::new();
        let pi = std::f32::consts::PI;

        phases.insert(Uuid::new_v4(), 0.0);
        phases.insert(Uuid::new_v4(), 2.0 * pi / 3.0);
        phases.insert(Uuid::new_v4(), 4.0 * pi / 3.0);

        let r = kuramoto_order_parameter(&phases);
        assert!(r < 0.1, "Evenly spaced phases should be desynchronized");
    }

    #[test]
    fn test_kuramoto_coupling_convergence() {
        let mut phases = HashMap::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        phases.insert(id1, 0.0);
        phases.insert(id2, 1.0);

        let initial_r = kuramoto_order_parameter(&phases);

        // Apply coupling iterations
        let mut current = phases;
        for _ in 0..100 {
            current = kuramoto_coupling(&current, 0.9, 0.01); // Use constitution K=0.9
        }

        let final_r = kuramoto_order_parameter(&current);

        assert!(
            final_r > initial_r,
            "Coupling should increase synchronization"
        );
    }

    // ============================================================
    // Edge Case Tests
    // ============================================================

    #[test]
    fn test_kuramoto_single_node() {
        let mut phases = HashMap::new();
        let id = Uuid::new_v4();
        phases.insert(id, 1.5);

        let result = kuramoto_coupling(&phases, 0.9, 0.01);
        assert_eq!(result.len(), 1);
        // Single node should be unchanged (coupling sum is 0)
        assert!((result[&id] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_kuramoto_empty() {
        let phases = HashMap::new();
        let r = kuramoto_order_parameter(&phases);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_select_replay_memories_limit_zero() {
        let mem1 = Uuid::new_v4();
        let memories = vec![(mem1, 1000, 0.5)];
        let selected = select_replay_memories(&memories, 0.8, 0);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_find_coactivated_pairs_single_node() {
        let node_a = Uuid::new_v4();
        let mut activations = HashMap::new();
        activations.insert(node_a, 0.8);

        let pairs = find_coactivated_pairs(&activations, 0.1);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_find_coactivated_pairs_empty() {
        let activations = HashMap::new();
        let pairs = find_coactivated_pairs(&activations, 0.1);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_hebbian_engine_reset() {
        let mut engine = HebbianEngine::with_defaults();
        let node_a = Uuid::new_v4();
        let node_b = Uuid::new_v4();

        engine.set_activations(&[NodeActivation::new(node_a, 0.8)]);
        engine.load_edges(&[(node_a, node_b, 0.5)]);
        engine.compute_updates();

        engine.reset();

        assert!(engine.stats().edges_strengthened == 0);
        assert!(engine.stats().edges_weakened == 0);
    }
}
