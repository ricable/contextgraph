# TASK-DREAM-P0-003: Hebbian Learning Implementation

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-DREAM-P0-003 |
| **Spec Ref** | SPEC-DREAM-001 |
| **Layer** | 2 (Logic) |
| **Priority** | P0 - Critical |
| **Effort** | 4 hours |
| **Dependencies** | TASK-DREAM-P0-001 |
| **Blocks** | TASK-DREAM-P0-006 |

---

## 1. Objective

Implement the Hebbian learning algorithm for NREM phase memory consolidation. This replaces the stub implementation with actual weight updates using the formula `dw_ij = eta * phi_i * phi_j`, including weight decay, pruning, and shortcut candidate detection.

---

## 2. Input Context Files

```yaml
must_read:
  - path: crates/context-graph-core/src/dream/nrem.rs
    purpose: Existing NremPhase stub to be extended
  - path: crates/context-graph-core/src/dream/types.rs
    purpose: HebbianConfig and NodeActivation types (from TASK-001)
  - path: crates/context-graph-core/src/dream/amortized.rs
    purpose: AmortizedLearner for shortcut detection
  - path: crates/context-graph-core/src/types/graph_edge/edge.rs
    purpose: GraphEdge for weight updates

should_read:
  - path: crates/context-graph-core/src/dream/mod.rs
    purpose: Dream module constants
  - path: crates/context-graph-core/src/dream/constants
    purpose: Constitution-mandated values
```

---

## 3. Files to Create/Modify

### 3.1 Create: `crates/context-graph-core/src/dream/hebbian.rs`

```rust
//! Hebbian Learning Implementation
//!
//! Implements the Hebbian learning rule for NREM phase memory consolidation:
//! "Neurons that fire together wire together"
//!
//! Formula: dw_ij = eta * phi_i * phi_j
//!
//! Constitution Reference: Section dream.phases.nrem

use std::collections::{HashMap, HashSet};

use tracing::{debug, trace, warn};
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

    /// Edge IDs marked for pruning
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
    /// Configuration parameters
    config: HebbianConfig,

    /// Active node activations (phi values)
    activations: HashMap<Uuid, f32>,

    /// Edges being processed in current cycle
    edge_weights: HashMap<(Uuid, Uuid), f32>,

    /// Edges that have been updated
    updated_edges: Vec<EdgeUpdate>,

    /// Statistics for current cycle
    stats: HebbianUpdateResult,
}

impl HebbianEngine {
    /// Create a new Hebbian engine with the given configuration.
    pub fn new(config: HebbianConfig) -> Self {
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
    ///
    /// Activations represent the "firing" level of each node during replay.
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
    ///
    /// Call this with edges between activated nodes.
    pub fn load_edges(&mut self, edges: &[(Uuid, Uuid, f32)]) {
        self.edge_weights.clear();
        for (source, target, weight) in edges {
            self.edge_weights.insert((*source, *target), *weight);
        }

        debug!("Loaded {} edges for Hebbian update", self.edge_weights.len());
    }

    /// Compute Hebbian updates for all loaded edges.
    ///
    /// Applies the formula: dw_ij = eta * phi_i * phi_j
    ///
    /// # Returns
    ///
    /// Vector of edge updates to be applied
    pub fn compute_updates(&mut self) -> Vec<EdgeUpdate> {
        self.updated_edges.clear();
        self.stats = HebbianUpdateResult::default();

        let mut total_delta = 0.0f32;
        let mut max_delta = 0.0f32;

        for ((source, target), current_weight) in &self.edge_weights {
            // Get activations for both nodes
            let phi_i = self.activations.get(source).copied().unwrap_or(0.0);
            let phi_j = self.activations.get(target).copied().unwrap_or(0.0);

            // Compute Hebbian delta
            let hebbian_delta = self.compute_delta(phi_i, phi_j);

            // Apply decay
            let decayed = current_weight * (1.0 - self.config.weight_decay);

            // Compute new weight with bounds
            let new_weight = (decayed + hebbian_delta)
                .clamp(self.config.weight_floor, self.config.weight_cap);

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
/// Prioritizes recent memories while maintaining some diversity.
///
/// # Arguments
///
/// * `memories` - List of (memory_id, timestamp_ms, phi_value) tuples
/// * `recency_bias` - How much to favor recent memories (0.0 = uniform, 1.0 = most recent only)
/// * `limit` - Maximum number of memories to select
///
/// # Returns
///
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

    // Find time range
    let min_time = memories.iter().map(|(_, t, _)| *t).min().unwrap_or(0);
    let max_time = memories.iter().map(|(_, t, _)| *t).max().unwrap_or(1);
    let time_range = (max_time - min_time).max(1) as f32;

    // Compute scores combining recency and phi
    let mut scored: Vec<(Uuid, f32)> = memories
        .iter()
        .map(|(id, timestamp, phi)| {
            // Normalize timestamp to [0, 1] where 1 is most recent
            let recency = (*timestamp - min_time) as f32 / time_range;

            // Combine recency and phi with bias
            let score = recency_bias * recency + (1.0 - recency_bias) * phi;

            (*id, score)
        })
        .collect();

    // Sort by score descending
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top memories
    scored.into_iter().take(limit).map(|(id, _)| id).collect()
}

/// Find co-activated node pairs from a set of activations.
///
/// Two nodes are co-activated if both have phi > threshold.
///
/// # Arguments
///
/// * `activations` - Map of node_id -> phi value
/// * `threshold` - Minimum phi for a node to be considered active
///
/// # Returns
///
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

    // Create all pairs (avoid duplicates by ordering)
    for (i, &node_a) in active_nodes.iter().enumerate() {
        for &node_b in active_nodes.iter().skip(i + 1) {
            // Order by UUID to ensure consistent pair representation
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
/// Updates phase values to synchronize co-activated nodes.
///
/// Formula: d(theta_i)/dt = omega_i + (K/N) * sum_j(sin(theta_j - theta_i))
///
/// # Arguments
///
/// * `phases` - Current phase values (theta) for each node
/// * `coupling_strength` - Kuramoto K parameter (Constitution: 10.0)
/// * `dt` - Time step
///
/// # Returns
///
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
        // Compute coupling term
        let coupling_sum: f32 = phases
            .values()
            .map(|&theta_j| (theta_j - theta_i).sin())
            .sum();

        // Update phase
        let d_theta = (coupling_strength / n) * coupling_sum;
        let new_theta = theta_i + d_theta * dt;

        // Wrap to [0, 2*PI]
        let wrapped = new_theta % (2.0 * std::f32::consts::PI);
        new_phases.insert(*node_i, if wrapped < 0.0 { wrapped + 2.0 * std::f32::consts::PI } else { wrapped });
    }

    new_phases
}

/// Compute Kuramoto order parameter (synchronization measure).
///
/// r = |1/N * sum_j(e^(i*theta_j))|
///
/// Returns value in [0, 1] where:
/// - 0 = completely desynchronized
/// - 1 = perfectly synchronized
pub fn kuramoto_order_parameter(phases: &HashMap<Uuid, f32>) -> f32 {
    if phases.is_empty() {
        return 0.0;
    }

    let n = phases.len() as f32;

    // Compute mean of e^(i*theta) = cos(theta) + i*sin(theta)
    let (sum_cos, sum_sin): (f32, f32) = phases
        .values()
        .map(|&theta| (theta.cos(), theta.sin()))
        .fold((0.0, 0.0), |(sc, ss), (c, s)| (sc + c, ss + s));

    let mean_cos = sum_cos / n;
    let mean_sin = sum_sin / n;

    // Order parameter is magnitude
    (mean_cos * mean_cos + mean_sin * mean_sin).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn make_activation(phi: f32) -> NodeActivation {
        NodeActivation {
            node_id: Uuid::new_v4(),
            phi,
            timestamp: Some(Instant::now()),
        }
    }

    #[test]
    fn test_hebbian_engine_creation() {
        let engine = HebbianEngine::with_defaults();
        assert_eq!(engine.config().learning_rate, 0.01);
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

        // If either phi is 0, delta should be 0
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

        // Weight should increase (high activations)
        assert!(update.was_strengthened(), "Weight should be strengthened");
        assert!(!update.should_prune, "Should not prune");
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

        // Weight should decrease (decay only)
        assert!(update.was_weakened(), "Weight should decay");
    }

    #[test]
    fn test_hebbian_update_pruning() {
        let mut engine = HebbianEngine::with_defaults();

        let node_a = Uuid::new_v4();
        let node_b = Uuid::new_v4();

        // Start at floor, decay should trigger prune
        engine.set_activations(&[]);
        engine.load_edges(&[(node_a, node_b, 0.05)]); // At floor

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

        // Max activations, high weight
        engine.set_activations(&[
            NodeActivation::new(node_a, 1.0),
            NodeActivation::new(node_b, 1.0),
        ]);

        engine.load_edges(&[(node_a, node_b, 0.99)]);

        let updates = engine.compute_updates();

        assert_eq!(updates.len(), 1);
        let update = &updates[0];

        // Weight should be capped at 1.0
        assert!(update.new_weight <= 1.0, "Weight should be capped at 1.0");
    }

    #[test]
    fn test_select_replay_memories_recency() {
        let mem1 = Uuid::new_v4();
        let mem2 = Uuid::new_v4();
        let mem3 = Uuid::new_v4();

        let memories = vec![
            (mem1, 1000, 0.5), // Old, medium phi
            (mem2, 2000, 0.5), // Middle, medium phi
            (mem3, 3000, 0.5), // Recent, medium phi
        ];

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
            (mem2, 2000, 0.1), // Middle, low phi
            (mem3, 3000, 0.5), // Recent, medium phi
        ];

        // Low recency bias should prefer high phi
        let selected = select_replay_memories(&memories, 0.1, 2);

        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0], mem1); // Highest phi first
    }

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

        // Should have 1 pair (a, b) since c is below threshold
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn test_kuramoto_order_parameter_synchronized() {
        let mut phases = HashMap::new();
        phases.insert(Uuid::new_v4(), 0.0);
        phases.insert(Uuid::new_v4(), 0.0);
        phases.insert(Uuid::new_v4(), 0.0);

        // All at same phase = perfectly synchronized
        let r = kuramoto_order_parameter(&phases);
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_kuramoto_order_parameter_desynchronized() {
        let mut phases = HashMap::new();
        let pi = std::f32::consts::PI;

        // Evenly spaced phases = desynchronized
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
        phases.insert(id2, 1.0); // Different phase

        let initial_r = kuramoto_order_parameter(&phases);

        // Apply coupling iterations
        let mut current = phases;
        for _ in 0..100 {
            current = kuramoto_coupling(&current, 10.0, 0.01);
        }

        let final_r = kuramoto_order_parameter(&current);

        // Should become more synchronized
        assert!(
            final_r > initial_r,
            "Coupling should increase synchronization"
        );
    }
}
```

### 3.2 Modify: `crates/context-graph-core/src/dream/nrem.rs`

Update NremPhase to use HebbianEngine:

```rust
// Add imports at top:
use super::hebbian::{HebbianEngine, HebbianUpdateResult, select_replay_memories, find_coactivated_pairs};
use super::types::{HebbianConfig, NodeActivation};

// Add field to NremPhase struct (after line 61):
    /// Hebbian learning engine
    hebbian_engine: HebbianEngine,

// Update new() method:
pub fn new() -> Self {
    Self {
        duration: constants::NREM_DURATION,
        coupling: constants::NREM_COUPLING,
        recency_bias: constants::NREM_RECENCY_BIAS,
        learning_rate: 0.01,
        batch_size: 64,
        weight_decay: 0.001,
        weight_floor: 0.05,
        weight_cap: 1.0,
        hebbian_engine: HebbianEngine::with_defaults(),
    }
}

// Update NremReport struct (after line 95):
    /// Detailed Hebbian statistics
    pub hebbian_stats: HebbianUpdateStats,

    /// Activation pairs that were processed
    pub activation_pairs_count: usize,

// Update the process method to use hebbian_engine instead of stub
```

### 3.3 Modify: `crates/context-graph-core/src/dream/mod.rs`

Add hebbian module export:

```rust
// Add after poincare_walk module:
pub mod hebbian;

// Add to re-exports:
pub use hebbian::{
    HebbianEngine,
    HebbianUpdateResult,
    EdgeUpdate,
    select_replay_memories,
    find_coactivated_pairs,
    kuramoto_coupling,
    kuramoto_order_parameter,
};
```

---

## 4. Definition of Done

### 4.1 Type Signatures (Exact)

```rust
pub struct HebbianUpdateResult {
    pub edges_strengthened: usize,
    pub edges_weakened: usize,
    pub edges_to_prune: usize,
    pub total_delta: f32,
    pub average_delta: f32,
    pub max_delta: f32,
    pub prune_candidates: Vec<(Uuid, Uuid)>,
}

pub struct EdgeUpdate {
    pub source_id: Uuid,
    pub target_id: Uuid,
    pub old_weight: f32,
    pub new_weight: f32,
    pub hebbian_delta: f32,
    pub should_prune: bool,
}

pub struct HebbianEngine { /* internal */ }
impl HebbianEngine {
    pub fn new(config: HebbianConfig) -> Self;
    pub fn with_defaults() -> Self;
    pub fn set_activations(&mut self, activations: &[NodeActivation]);
    pub fn load_edges(&mut self, edges: &[(Uuid, Uuid, f32)]);
    pub fn compute_updates(&mut self) -> Vec<EdgeUpdate>;
    pub fn compute_delta(&self, phi_i: f32, phi_j: f32) -> f32;
    pub fn stats(&self) -> &HebbianUpdateResult;
    pub fn config(&self) -> &HebbianConfig;
    pub fn reset(&mut self);
}

pub fn select_replay_memories(
    memories: &[(Uuid, u64, f32)],
    recency_bias: f32,
    limit: usize,
) -> Vec<Uuid>;

pub fn find_coactivated_pairs(
    activations: &HashMap<Uuid, f32>,
    threshold: f32,
) -> HashSet<(Uuid, Uuid)>;

pub fn kuramoto_coupling(
    phases: &HashMap<Uuid, f32>,
    coupling_strength: f32,
    dt: f32,
) -> HashMap<Uuid, f32>;

pub fn kuramoto_order_parameter(phases: &HashMap<Uuid, f32>) -> f32;
```

### 4.2 Validation Criteria

| Criterion | Check |
|-----------|-------|
| Compiles | `cargo build -p context-graph-core` |
| Tests pass | `cargo test -p context-graph-core dream::hebbian` |
| No clippy warnings | `cargo clippy -p context-graph-core` |
| Hebbian formula correct | `dw = eta * phi_i * phi_j` verified |
| Weight decay applied | `w_new = w_old * (1 - decay)` |
| Weight floor enforced | Pruning at 0.05 |
| Weight cap enforced | Capped at 1.0 |
| Recency bias works | Higher bias = more recent memories |
| Kuramoto converges | Order parameter increases |

### 4.3 Test Coverage Requirements

- [ ] HebbianEngine with default config
- [ ] Hebbian delta formula correctness
- [ ] Hebbian delta with zero activation
- [ ] Edge strengthening with high phi
- [ ] Edge weakening with decay only
- [ ] Edge pruning at weight floor
- [ ] Weight capping at 1.0
- [ ] Memory selection with high recency bias
- [ ] Memory selection with low recency bias
- [ ] Co-activated pair detection
- [ ] Kuramoto synchronized order parameter ~1.0
- [ ] Kuramoto desynchronized order parameter ~0.0
- [ ] Kuramoto coupling increases synchronization

---

## 5. Implementation Notes

### 5.1 Hebbian Learning Formula

```
dw_ij = eta * phi_i * phi_j

where:
  eta = 0.01 (learning rate)
  phi_i = activation of node i [0.0, 1.0]
  phi_j = activation of node j [0.0, 1.0]

Full update:
  w_new = (w_old * (1 - decay)) + dw_ij
  w_final = clamp(w_new, floor, cap)
```

### 5.2 Constitution Compliance

- `learning_rate = 0.01`
- `weight_decay = 0.001`
- `weight_floor = 0.05`
- `weight_cap = 1.0`
- `recency_bias = 0.8`
- `coupling_strength = 10.0` (Kuramoto K)

### 5.3 Performance Considerations

- Use HashMap for O(1) activation lookup
- Batch edge updates to minimize allocations
- Consider SIMD for Kuramoto sum computation

---

## 6. Estimated Effort Breakdown

| Phase | Duration |
|-------|----------|
| HebbianEngine core | 60 min |
| Edge update logic | 45 min |
| Memory selection | 30 min |
| Kuramoto coupling | 30 min |
| Unit tests | 60 min |
| Integration with NremPhase | 15 min |
| **Total** | **4 hours** |
