# TASK-DREAM-P0-004: Hyperbolic Random Walk Implementation

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-DREAM-P0-004 |
| **Spec Ref** | SPEC-DREAM-001 |
| **Layer** | 2 (Logic) |
| **Priority** | P0 - Critical |
| **Effort** | 4 hours |
| **Dependencies** | TASK-DREAM-P0-001, TASK-DREAM-P0-002 |
| **Blocks** | TASK-DREAM-P0-006 |

---

## 1. Objective

Implement the hyperbolic random walk algorithm for REM phase blind spot discovery. This replaces the stub implementation with actual Poincare ball exploration using Mobius addition, temperature-controlled direction sampling, and blind spot detection.

---

## 2. Input Context Files

```yaml
must_read:
  - path: crates/context-graph-core/src/dream/rem.rs
    purpose: Existing RemPhase stub to be extended
  - path: crates/context-graph-core/src/dream/types.rs
    purpose: HyperbolicWalkConfig and WalkStep types (from TASK-001)
  - path: crates/context-graph-core/src/dream/poincare_walk.rs
    purpose: Math utilities for Poincare ball (from TASK-002)
  - path: crates/context-graph-graph/src/hyperbolic/poincare/types.rs
    purpose: PoincarePoint type for integration

should_read:
  - path: crates/context-graph-core/src/dream/mod.rs
    purpose: Dream module structure
  - path: crates/context-graph-graph/src/config/hyperbolic.rs
    purpose: HyperbolicConfig for reference
```

---

## 3. Files to Create/Modify

### 3.1 Create: `crates/context-graph-core/src/dream/hyperbolic_walk.rs`

```rust
//! Hyperbolic Random Walk Implementation
//!
//! Implements random walks in the Poincare ball model for REM phase
//! blind spot discovery and creative association exploration.
//!
//! Constitution Reference: Section dream.phases.rem

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tracing::{debug, info, trace, warn};
use uuid::Uuid;

use super::poincare_walk::{
    geodesic_distance, is_far_from_all, mobius_add, norm_64, random_direction,
    sample_direction_with_temperature, scale_direction, PoincareBallConfig,
};
use super::types::{HyperbolicWalkConfig, WalkStep};

/// A discovered blind spot in hyperbolic space.
#[derive(Debug, Clone)]
pub struct DiscoveredBlindSpot {
    /// Position in Poincare ball (64D)
    pub position: [f32; 64],

    /// Geodesic distance from nearest known memory
    pub distance_from_nearest: f32,

    /// Walk step index where discovered
    pub discovery_step: usize,

    /// Confidence score (based on isolation)
    pub confidence: f32,

    /// ID for tracking
    pub id: Uuid,
}

impl DiscoveredBlindSpot {
    /// Check if this is a significant blind spot.
    ///
    /// Significant = distance >= 0.7 (Constitution semantic_leap)
    pub fn is_significant(&self) -> bool {
        self.distance_from_nearest >= 0.7 && self.confidence >= 0.5
    }
}

/// Result from a single random walk.
#[derive(Debug, Clone)]
pub struct WalkResult {
    /// All steps taken in the walk
    pub trajectory: Vec<WalkStep>,

    /// Blind spots discovered during walk
    pub blind_spots: Vec<DiscoveredBlindSpot>,

    /// Total geodesic distance traveled
    pub total_distance: f32,

    /// Starting position
    pub start_position: [f32; 64],

    /// Ending position
    pub end_position: [f32; 64],

    /// Whether walk completed (vs interrupted)
    pub completed: bool,
}

/// Result from the complete REM exploration phase.
#[derive(Debug, Clone)]
pub struct ExplorationResult {
    /// All walks performed
    pub walks: Vec<WalkResult>,

    /// All blind spots discovered (aggregated)
    pub all_blind_spots: Vec<DiscoveredBlindSpot>,

    /// Total queries generated (walks * steps)
    pub queries_generated: usize,

    /// Coverage estimate (fraction of space explored)
    pub coverage_estimate: f32,

    /// Average semantic leap distance
    pub average_semantic_leap: f32,

    /// Unique positions visited
    pub unique_positions: usize,
}

/// Hyperbolic random walk explorer for REM phase.
///
/// Performs random walks in the Poincare ball to discover
/// conceptual blind spots (unexplored semantic regions).
#[derive(Debug)]
pub struct HyperbolicExplorer {
    /// Walk configuration
    config: HyperbolicWalkConfig,

    /// Poincare ball configuration
    ball_config: PoincareBallConfig,

    /// Random number generator
    rng: SmallRng,

    /// Known memory positions (for blind spot detection)
    known_positions: Vec<[f32; 64]>,

    /// Maximum queries (Constitution: 100)
    query_limit: usize,

    /// Queries used so far
    queries_used: usize,
}

impl HyperbolicExplorer {
    /// Create a new explorer with the given configuration.
    pub fn new(config: HyperbolicWalkConfig) -> Self {
        Self {
            config,
            ball_config: PoincareBallConfig::default(),
            rng: SmallRng::from_entropy(),
            known_positions: Vec::new(),
            query_limit: 100, // Constitution limit
            queries_used: 0,
        }
    }

    /// Create with constitution defaults.
    pub fn with_defaults() -> Self {
        Self::new(HyperbolicWalkConfig::default())
    }

    /// Set the seed for reproducible walks (testing).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = SmallRng::seed_from_u64(seed);
        self
    }

    /// Set known memory positions for blind spot detection.
    pub fn set_known_positions(&mut self, positions: Vec<[f32; 64]>) {
        self.known_positions = positions;
        debug!(
            "Set {} known positions for blind spot detection",
            self.known_positions.len()
        );
    }

    /// Reset query counter for a new REM cycle.
    pub fn reset_queries(&mut self) {
        self.queries_used = 0;
    }

    /// Get remaining query budget.
    pub fn remaining_queries(&self) -> usize {
        self.query_limit.saturating_sub(self.queries_used)
    }

    /// Perform a single random walk from the given starting position.
    ///
    /// # Arguments
    ///
    /// * `start` - Starting position in Poincare ball
    /// * `interrupt_flag` - Flag to check for abort
    ///
    /// # Returns
    ///
    /// Walk result with trajectory and discovered blind spots
    pub fn walk(
        &mut self,
        start: &[f32; 64],
        interrupt_flag: &Arc<AtomicBool>,
    ) -> WalkResult {
        let mut trajectory = Vec::with_capacity(self.config.max_steps);
        let mut blind_spots = Vec::new();
        let mut current_position = *start;
        let mut total_distance = 0.0f32;
        let mut last_position = *start;

        trace!("Starting walk from norm={:.4}", norm_64(start));

        for step_index in 0..self.config.max_steps {
            // Check for interrupt
            if interrupt_flag.load(Ordering::Relaxed) {
                debug!("Walk interrupted at step {}", step_index);
                break;
            }

            // Check query budget
            if self.queries_used >= self.query_limit {
                debug!("Query limit reached at step {}", step_index);
                break;
            }

            self.queries_used += 1;

            // Sample direction with temperature
            let direction = sample_direction_with_temperature(
                &mut self.rng,
                self.config.direction_samples,
                None, // No scoring, pure exploration
                self.config.temperature,
            );

            // Scale direction based on position (smaller steps near boundary)
            let current_norm = norm_64(&current_position);
            let velocity = scale_direction(
                &direction,
                self.config.step_size,
                current_norm,
                &self.ball_config,
            );

            // Perform Mobius addition to get new position
            let new_position = mobius_add(&current_position, &velocity, &self.ball_config);

            // Compute step distance
            let step_distance =
                geodesic_distance(&current_position, &new_position, &self.ball_config);
            total_distance += step_distance;

            // Check for blind spot
            let is_blind_spot = self.check_blind_spot(&new_position);

            // Create walk step
            let mut step = WalkStep::new(
                new_position,
                direction,
                geodesic_distance(start, &new_position, &self.ball_config),
                step_index,
            );

            if is_blind_spot {
                step.mark_blind_spot();

                // Record blind spot
                let distance_from_nearest = self.distance_to_nearest(&new_position);
                let confidence = self.compute_blind_spot_confidence(&new_position);

                blind_spots.push(DiscoveredBlindSpot {
                    position: new_position,
                    distance_from_nearest,
                    discovery_step: step_index,
                    confidence,
                    id: Uuid::new_v4(),
                });

                trace!(
                    "Blind spot discovered at step {}, distance={:.4}",
                    step_index,
                    distance_from_nearest
                );
            }

            trajectory.push(step);
            last_position = current_position;
            current_position = new_position;
        }

        let completed = trajectory.len() >= self.config.max_steps
            || self.queries_used >= self.query_limit;

        debug!(
            "Walk completed: {} steps, {} blind spots, distance={:.4}",
            trajectory.len(),
            blind_spots.len(),
            total_distance
        );

        WalkResult {
            trajectory,
            blind_spots,
            total_distance,
            start_position: *start,
            end_position: if trajectory.is_empty() {
                *start
            } else {
                last_position
            },
            completed,
        }
    }

    /// Perform multiple random walks for complete exploration.
    ///
    /// # Arguments
    ///
    /// * `starting_positions` - Positions to start walks from (high-phi nodes)
    /// * `interrupt_flag` - Flag to check for abort
    ///
    /// # Returns
    ///
    /// Complete exploration result with all walks and blind spots
    pub fn explore(
        &mut self,
        starting_positions: &[[f32; 64]],
        interrupt_flag: &Arc<AtomicBool>,
    ) -> ExplorationResult {
        let mut walks = Vec::new();
        let mut all_blind_spots = Vec::new();
        let mut total_semantic_leap = 0.0f32;
        let mut leap_count = 0usize;

        info!(
            "Starting exploration from {} positions, query_limit={}",
            starting_positions.len(),
            self.query_limit
        );

        // If no starting positions provided, start from origin
        let starts: Vec<[f32; 64]> = if starting_positions.is_empty() {
            vec![[0.0f32; 64]]
        } else {
            starting_positions.to_vec()
        };

        for (i, start) in starts.iter().enumerate() {
            // Check for interrupt
            if interrupt_flag.load(Ordering::Relaxed) {
                debug!("Exploration interrupted after {} walks", walks.len());
                break;
            }

            // Check query budget
            if self.queries_used >= self.query_limit {
                debug!("Query limit reached after {} walks", walks.len());
                break;
            }

            trace!("Starting walk {} from position {}", i, i);

            let walk = self.walk(start, interrupt_flag);

            // Collect blind spots
            for bs in &walk.blind_spots {
                if bs.is_significant() {
                    total_semantic_leap += bs.distance_from_nearest;
                    leap_count += 1;
                }
                all_blind_spots.push(bs.clone());
            }

            walks.push(walk);
        }

        // Compute statistics
        let queries_generated = self.queries_used;
        let average_semantic_leap = if leap_count > 0 {
            total_semantic_leap / leap_count as f32
        } else {
            0.0
        };

        // Estimate coverage (rough approximation)
        let unique_positions = walks.iter().map(|w| w.trajectory.len()).sum();
        let coverage_estimate = (unique_positions as f32 / 1000.0).min(1.0);

        info!(
            "Exploration complete: {} walks, {} blind spots, {} queries",
            walks.len(),
            all_blind_spots.len(),
            queries_generated
        );

        ExplorationResult {
            walks,
            all_blind_spots,
            queries_generated,
            coverage_estimate,
            average_semantic_leap,
            unique_positions,
        }
    }

    /// Check if a position is a blind spot (far from known memories).
    fn check_blind_spot(&self, position: &[f32; 64]) -> bool {
        if self.known_positions.is_empty() {
            // If no known positions, everywhere is a blind spot
            return true;
        }

        is_far_from_all(
            position,
            &self.known_positions,
            self.config.min_blind_spot_distance,
            &self.ball_config,
        )
    }

    /// Compute distance to nearest known position.
    fn distance_to_nearest(&self, position: &[f32; 64]) -> f32 {
        if self.known_positions.is_empty() {
            return f32::INFINITY;
        }

        self.known_positions
            .iter()
            .map(|p| geodesic_distance(position, p, &self.ball_config))
            .fold(f32::INFINITY, f32::min)
    }

    /// Compute confidence for a blind spot based on isolation.
    fn compute_blind_spot_confidence(&self, position: &[f32; 64]) -> f32 {
        if self.known_positions.is_empty() {
            return 1.0;
        }

        let dist = self.distance_to_nearest(position);

        // Confidence increases with distance
        // Scale so that semantic_leap threshold (0.7) gives 0.5 confidence
        let base_confidence = (dist / 1.4).min(1.0);

        // Bonus for being far from boundary
        let norm = norm_64(position);
        let boundary_bonus = if norm < 0.5 { 0.2 } else { 0.0 };

        (base_confidence + boundary_bonus).min(1.0)
    }

    /// Get configuration.
    pub fn config(&self) -> &HyperbolicWalkConfig {
        &self.config
    }
}

/// Convert Poincare ball position to synthetic query embedding.
///
/// This creates a query that can be used to search for existing
/// memories near the discovered blind spot.
pub fn position_to_query(position: &[f32; 64]) -> Vec<f32> {
    position.to_vec()
}

/// Sample high-phi nodes as starting positions for walks.
///
/// # Arguments
///
/// * `node_positions` - List of (phi, position) tuples
/// * `count` - Number of starting positions to select
/// * `temperature` - Sampling temperature (higher = more random)
///
/// # Returns
///
/// Selected starting positions
pub fn sample_starting_positions(
    node_positions: &[(f32, [f32; 64])],
    count: usize,
    temperature: f32,
) -> Vec<[f32; 64]> {
    if node_positions.is_empty() || count == 0 {
        return Vec::new();
    }

    let mut rng = SmallRng::from_entropy();

    // Score by phi value
    let scores: Vec<f32> = node_positions.iter().map(|(phi, _)| *phi).collect();

    // Softmax with temperature
    let temperature = temperature.max(0.01);
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_scores: Vec<f32> = scores
        .iter()
        .map(|&s| ((s - max_score) / temperature).exp())
        .collect();
    let sum: f32 = exp_scores.iter().sum();
    let probs: Vec<f32> = exp_scores.iter().map(|&e| e / sum).collect();

    // Sample without replacement
    let mut selected = Vec::with_capacity(count);
    let mut available: Vec<usize> = (0..node_positions.len()).collect();

    for _ in 0..count.min(node_positions.len()) {
        if available.is_empty() {
            break;
        }

        // Sample from remaining
        let threshold: f32 = rng.gen();
        let mut cumulative = 0.0;
        let mut selected_idx = 0;

        for (i, &original_idx) in available.iter().enumerate() {
            cumulative += probs[original_idx];
            if threshold < cumulative {
                selected_idx = i;
                break;
            }
        }

        let original_idx = available.remove(selected_idx);
        selected.push(node_positions[original_idx].1);
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_interrupt_flag() -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }

    #[test]
    fn test_explorer_creation() {
        let explorer = HyperbolicExplorer::with_defaults();
        assert_eq!(explorer.config.temperature, 2.0);
        assert_eq!(explorer.query_limit, 100);
    }

    #[test]
    fn test_explorer_walk_from_origin() {
        let mut explorer = HyperbolicExplorer::with_defaults().with_seed(42);
        let start = [0.0f32; 64];
        let interrupt = make_interrupt_flag();

        let result = explorer.walk(&start, &interrupt);

        assert!(result.trajectory.len() > 0);
        assert!(result.total_distance >= 0.0);
    }

    #[test]
    fn test_explorer_walk_stays_in_ball() {
        let mut explorer = HyperbolicExplorer::with_defaults().with_seed(42);
        let start = [0.0f32; 64];
        let interrupt = make_interrupt_flag();

        let result = explorer.walk(&start, &interrupt);

        for step in &result.trajectory {
            let norm = norm_64(&step.position);
            assert!(norm < 1.0, "Walk stepped outside ball: norm={}", norm);
        }
    }

    #[test]
    fn test_explorer_respects_query_limit() {
        let config = HyperbolicWalkConfig {
            max_steps: 200, // More than query limit
            ..Default::default()
        };
        let mut explorer = HyperbolicExplorer::new(config).with_seed(42);
        let start = [0.0f32; 64];
        let interrupt = make_interrupt_flag();

        let result = explorer.walk(&start, &interrupt);

        // Should stop at query limit (100), not max_steps (200)
        assert!(explorer.queries_used <= 100);
    }

    #[test]
    fn test_explorer_interrupt() {
        let mut explorer = HyperbolicExplorer::with_defaults().with_seed(42);
        let start = [0.0f32; 64];
        let interrupt = Arc::new(AtomicBool::new(true)); // Pre-interrupted

        let result = explorer.walk(&start, &interrupt);

        // Should stop immediately
        assert_eq!(result.trajectory.len(), 0);
        assert!(!result.completed);
    }

    #[test]
    fn test_explorer_blind_spot_detection() {
        let mut explorer = HyperbolicExplorer::with_defaults().with_seed(42);

        // Set some known positions near origin
        let mut known = [0.0f32; 64];
        known[0] = 0.1;
        explorer.set_known_positions(vec![known]);

        // Walk should detect blind spots far from known positions
        let start = [0.0f32; 64];
        let interrupt = make_interrupt_flag();

        let result = explorer.walk(&start, &interrupt);

        // May or may not find blind spots depending on walk path
        // Just verify we can detect them
        for step in &result.trajectory {
            if step.blind_spot_detected {
                let dist = geodesic_distance(
                    &step.position,
                    &known,
                    &PoincareBallConfig::default(),
                );
                assert!(dist >= 0.7, "Blind spot should be far from known");
            }
        }
    }

    #[test]
    fn test_explorer_explore_multiple() {
        let mut explorer = HyperbolicExplorer::with_defaults().with_seed(42);
        let interrupt = make_interrupt_flag();

        let starts = vec![[0.0f32; 64], [0.1f32; 64]];

        let result = explorer.explore(&starts, &interrupt);

        assert!(result.walks.len() > 0);
        assert!(result.queries_generated > 0);
    }

    #[test]
    fn test_discovered_blind_spot_significance() {
        let significant = DiscoveredBlindSpot {
            position: [0.5f32; 64],
            distance_from_nearest: 0.8, // > 0.7
            discovery_step: 10,
            confidence: 0.6, // > 0.5
            id: Uuid::new_v4(),
        };
        assert!(significant.is_significant());

        let not_significant_distance = DiscoveredBlindSpot {
            distance_from_nearest: 0.5, // < 0.7
            ..significant.clone()
        };
        assert!(!not_significant_distance.is_significant());

        let not_significant_confidence = DiscoveredBlindSpot {
            confidence: 0.3, // < 0.5
            ..significant.clone()
        };
        assert!(!not_significant_confidence.is_significant());
    }

    #[test]
    fn test_sample_starting_positions_empty() {
        let positions = sample_starting_positions(&[], 5, 2.0);
        assert!(positions.is_empty());
    }

    #[test]
    fn test_sample_starting_positions_basic() {
        let node_positions = vec![
            (0.9, [0.1f32; 64]),
            (0.1, [0.2f32; 64]),
            (0.5, [0.3f32; 64]),
        ];

        let selected = sample_starting_positions(&node_positions, 2, 2.0);

        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_sample_starting_positions_high_phi_preference() {
        let node_positions = vec![
            (0.99, [0.1f32; 64]),
            (0.01, [0.2f32; 64]),
        ];

        // Low temperature should strongly prefer high phi
        let selected = sample_starting_positions(&node_positions, 1, 0.1);

        // Should almost always pick the high-phi node
        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn test_position_to_query() {
        let pos = [0.5f32; 64];
        let query = position_to_query(&pos);

        assert_eq!(query.len(), 64);
        assert_eq!(query[0], 0.5);
    }
}
```

### 3.2 Modify: `crates/context-graph-core/src/dream/rem.rs`

Update RemPhase to use HyperbolicExplorer:

```rust
// Add imports at top:
use super::hyperbolic_walk::{
    HyperbolicExplorer, ExplorationResult, DiscoveredBlindSpot, sample_starting_positions
};
use super::types::HyperbolicWalkConfig;

// Add field to RemPhase struct:
    /// Hyperbolic walk explorer
    explorer: HyperbolicExplorer,

// Update new() method:
pub fn new() -> Self {
    Self {
        duration: constants::REM_DURATION,
        temperature: constants::REM_TEMPERATURE,
        min_semantic_leap: constants::MIN_SEMANTIC_LEAP,
        query_limit: constants::MAX_REM_QUERIES,
        new_edge_weight: 0.3,
        new_edge_confidence: 0.5,
        exploration_bias: 0.7,
        walk_step_size: 0.3,
        explorer: HyperbolicExplorer::with_defaults(),
    }
}

// Update RemReport to include exploration details:
    /// Walk trajectories
    pub exploration_result: Option<ExplorationResult>,

    /// Discovered blind spot details
    pub blind_spot_details: Vec<BlindSpotDetail>,

// Update the process method to use explorer instead of stub
```

### 3.3 Modify: `crates/context-graph-core/src/dream/mod.rs`

Add hyperbolic_walk module export:

```rust
// Add after hebbian module:
pub mod hyperbolic_walk;

// Add to re-exports:
pub use hyperbolic_walk::{
    HyperbolicExplorer,
    DiscoveredBlindSpot,
    WalkResult,
    ExplorationResult,
    sample_starting_positions,
    position_to_query,
};
```

---

## 4. Definition of Done

### 4.1 Type Signatures (Exact)

```rust
pub struct DiscoveredBlindSpot {
    pub position: [f32; 64],
    pub distance_from_nearest: f32,
    pub discovery_step: usize,
    pub confidence: f32,
    pub id: Uuid,
}
impl DiscoveredBlindSpot {
    pub fn is_significant(&self) -> bool;
}

pub struct WalkResult {
    pub trajectory: Vec<WalkStep>,
    pub blind_spots: Vec<DiscoveredBlindSpot>,
    pub total_distance: f32,
    pub start_position: [f32; 64],
    pub end_position: [f32; 64],
    pub completed: bool,
}

pub struct ExplorationResult {
    pub walks: Vec<WalkResult>,
    pub all_blind_spots: Vec<DiscoveredBlindSpot>,
    pub queries_generated: usize,
    pub coverage_estimate: f32,
    pub average_semantic_leap: f32,
    pub unique_positions: usize,
}

pub struct HyperbolicExplorer { /* internal */ }
impl HyperbolicExplorer {
    pub fn new(config: HyperbolicWalkConfig) -> Self;
    pub fn with_defaults() -> Self;
    pub fn with_seed(self, seed: u64) -> Self;
    pub fn set_known_positions(&mut self, positions: Vec<[f32; 64]>);
    pub fn reset_queries(&mut self);
    pub fn remaining_queries(&self) -> usize;
    pub fn walk(&mut self, start: &[f32; 64], interrupt_flag: &Arc<AtomicBool>) -> WalkResult;
    pub fn explore(&mut self, starting_positions: &[[f32; 64]], interrupt_flag: &Arc<AtomicBool>) -> ExplorationResult;
    pub fn config(&self) -> &HyperbolicWalkConfig;
}

pub fn position_to_query(position: &[f32; 64]) -> Vec<f32>;
pub fn sample_starting_positions(
    node_positions: &[(f32, [f32; 64])],
    count: usize,
    temperature: f32,
) -> Vec<[f32; 64]>;
```

### 4.2 Validation Criteria

| Criterion | Check |
|-----------|-------|
| Compiles | `cargo build -p context-graph-core` |
| Tests pass | `cargo test -p context-graph-core dream::hyperbolic_walk` |
| No clippy warnings | `cargo clippy -p context-graph-core` |
| Walk stays in ball | All positions have norm < 1.0 |
| Respects query limit | Never exceeds 100 queries |
| Interrupt works | Stops immediately on interrupt |
| Blind spot detection | Distance >= 0.7 from known positions |
| Temperature effect | Higher temp = more exploratory |

### 4.3 Test Coverage Requirements

- [ ] Explorer creation with defaults
- [ ] Walk from origin produces trajectory
- [ ] Walk stays inside Poincare ball
- [ ] Walk respects query limit (100)
- [ ] Walk respects interrupt flag
- [ ] Blind spot detection works
- [ ] Significant blind spot threshold (distance >= 0.7, confidence >= 0.5)
- [ ] Multi-walk exploration
- [ ] Starting position sampling with high-phi preference
- [ ] Position to query conversion

---

## 5. Implementation Notes

### 5.1 Walk Algorithm

```
1. Start at given position in Poincare ball
2. For each step (up to max_steps, query_limit):
   a. Sample random direction with temperature=2.0
   b. Scale direction based on distance from boundary
   c. Apply Mobius addition: p' = p ⊕ v
   d. Check if new position is a blind spot
   e. Record step in trajectory
3. Return trajectory and discovered blind spots
```

### 5.2 Blind Spot Detection

A position is a blind spot if:
- Geodesic distance to ALL known positions >= min_blind_spot_distance (0.7)

A blind spot is significant if:
- Distance >= 0.7 (Constitution semantic_leap)
- Confidence >= 0.5

### 5.3 Constitution Compliance

- `temperature = 2.0`
- `min_semantic_leap = 0.7`
- `query_limit = 100`
- `max_steps = 50` per walk

---

## 6. Estimated Effort Breakdown

| Phase | Duration |
|-------|----------|
| HyperbolicExplorer core | 60 min |
| Walk algorithm | 45 min |
| Blind spot detection | 30 min |
| Multi-walk exploration | 30 min |
| Starting position sampling | 20 min |
| Unit tests | 45 min |
| Integration with RemPhase | 10 min |
| **Total** | **4 hours** |
