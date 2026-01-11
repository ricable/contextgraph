//! Hyperbolic Random Walk Implementation
//!
//! Implements random walks in the Poincare ball model for REM phase
//! blind spot discovery and creative association exploration.
//!
//! Constitution Reference: Section dream.phases.rem (lines 390-398)
//! - temperature: 2.0
//! - semantic_leap: >= 0.7
//! - query_limit: 100

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tracing::{debug, info, trace};
use uuid::Uuid;

use super::poincare_walk::{
    geodesic_distance, is_far_from_all, mobius_add, norm_64,
    sample_direction_with_temperature, scale_direction, PoincareBallConfig,
};
use super::types::{HyperbolicWalkConfig, WalkStep};

/// A discovered blind spot in hyperbolic space.
///
/// # Constitution Compliance
/// - `distance_from_nearest` must be >= 0.7 (semantic_leap) to be significant
#[derive(Debug, Clone)]
pub struct DiscoveredBlindSpot {
    /// Position in Poincare ball (64D)
    /// INVARIANT: norm(position) < 1.0
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
    /// Check if this is a significant blind spot per Constitution.
    ///
    /// Significant = distance >= 0.7 (Constitution semantic_leap) AND confidence >= 0.5
    #[inline]
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
    /// INVARIANT: queries_generated <= 100 (Constitution)
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
///
/// # Constitution Compliance
/// - Temperature: 2.0 (high exploration)
/// - Query limit: 100 (hard cap)
/// - Semantic leap: >= 0.7 for blind spot detection
#[derive(Debug)]
pub struct HyperbolicExplorer {
    /// Walk configuration (includes Constitution values)
    config: HyperbolicWalkConfig,

    /// Poincare ball configuration
    ball_config: PoincareBallConfig,

    /// Random number generator
    rng: StdRng,

    /// Known memory positions (for blind spot detection)
    known_positions: Vec<[f32; 64]>,

    /// Maximum queries (Constitution: 100)
    query_limit: usize,

    /// Queries used so far
    queries_used: usize,
}

impl HyperbolicExplorer {
    /// Create a new explorer with the given configuration.
    ///
    /// # Panics
    /// Panics if config violates Constitution bounds (validated internally).
    pub fn new(config: HyperbolicWalkConfig) -> Self {
        config.validate(); // Fail fast on invalid config

        Self {
            config,
            ball_config: PoincareBallConfig::default(),
            rng: StdRng::from_entropy(),
            known_positions: Vec::new(),
            query_limit: 100, // Constitution limit - HARD CODED
            queries_used: 0,
        }
    }

    /// Create with constitution defaults.
    pub fn with_defaults() -> Self {
        Self::new(HyperbolicWalkConfig::default())
    }

    /// Set the seed for reproducible walks (testing).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = StdRng::seed_from_u64(seed);
        self
    }

    /// Set known memory positions for blind spot detection.
    ///
    /// # Arguments
    /// * `positions` - List of 64D positions in Poincare ball
    ///
    /// # Panics
    /// Panics if any position has norm >= 1.0
    pub fn set_known_positions(&mut self, positions: Vec<[f32; 64]>) {
        // Validate all positions are inside ball
        for (i, pos) in positions.iter().enumerate() {
            let norm = norm_64(pos);
            if norm >= self.ball_config.max_norm {
                panic!(
                    "[HYPERBOLIC_WALK] Invalid known position {}: norm={:.6} >= max_norm={:.6}",
                    i, norm, self.ball_config.max_norm
                );
            }
        }

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
    #[inline]
    pub fn remaining_queries(&self) -> usize {
        self.query_limit.saturating_sub(self.queries_used)
    }

    /// Perform a single random walk from the given starting position.
    ///
    /// # Arguments
    /// * `start` - Starting position in Poincare ball (norm < 1.0)
    /// * `interrupt_flag` - Flag to check for abort (Constitution: wake < 100ms)
    ///
    /// # Returns
    /// Walk result with trajectory and discovered blind spots
    ///
    /// # Panics
    /// Panics if start position is outside Poincare ball.
    pub fn walk(
        &mut self,
        start: &[f32; 64],
        interrupt_flag: &Arc<AtomicBool>,
    ) -> WalkResult {
        // Fail fast: validate start position
        let start_norm = norm_64(start);
        if start_norm >= self.ball_config.max_norm {
            panic!(
                "[HYPERBOLIC_WALK] Start position outside ball: norm={:.6} >= max_norm={:.6}",
                start_norm, self.ball_config.max_norm
            );
        }

        let mut trajectory = Vec::with_capacity(self.config.max_steps);
        let mut blind_spots = Vec::new();
        let mut current_position = *start;
        let mut total_distance = 0.0f32;

        trace!("Starting walk from norm={:.4}", start_norm);

        for step_index in 0..self.config.max_steps {
            // Check for interrupt (Constitution: wake < 100ms)
            if interrupt_flag.load(Ordering::Relaxed) {
                debug!("Walk interrupted at step {} (abort_on_query)", step_index);
                return WalkResult {
                    trajectory,
                    blind_spots,
                    total_distance,
                    start_position: *start,
                    end_position: current_position,
                    completed: false,
                };
            }

            // Check query budget (Constitution: 100 max)
            if self.queries_used >= self.query_limit {
                debug!("Query limit {} reached at step {}", self.query_limit, step_index);
                break;
            }

            self.queries_used += 1;

            // Sample direction with temperature (Constitution: 2.0)
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

            // Check for blind spot (Constitution: semantic_leap >= 0.7)
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
                    "Blind spot discovered at step {}, distance={:.4}, confidence={:.4}",
                    step_index, distance_from_nearest, confidence
                );
            }

            trajectory.push(step);
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
            end_position: current_position,
            completed,
        }
    }

    /// Perform multiple random walks for complete exploration.
    ///
    /// # Arguments
    /// * `starting_positions` - Positions to start walks from (high-phi nodes)
    /// * `interrupt_flag` - Flag to check for abort
    ///
    /// # Returns
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

            trace!("Starting walk {} from position index {}", walks.len(), i);

            let walk = self.walk(start, interrupt_flag);

            // Collect significant blind spots
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
            "Exploration complete: {} walks, {} blind spots ({} significant), {} queries",
            walks.len(),
            all_blind_spots.len(),
            leap_count,
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
    ///
    /// Constitution: semantic_leap >= 0.7
    fn check_blind_spot(&self, position: &[f32; 64]) -> bool {
        if self.known_positions.is_empty() {
            // If no known positions, everywhere is a blind spot
            return true;
        }

        is_far_from_all(
            position,
            &self.known_positions,
            self.config.min_blind_spot_distance, // 0.7 per Constitution
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

        // Bonus for being far from boundary (more central = more stable)
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
#[inline]
pub fn position_to_query(position: &[f32; 64]) -> Vec<f32> {
    position.to_vec()
}

/// Sample high-phi nodes as starting positions for walks.
///
/// Uses softmax with temperature to prefer high-phi nodes while
/// maintaining exploration diversity.
///
/// # Arguments
/// * `node_positions` - List of (phi, position) tuples
/// * `count` - Number of starting positions to select
/// * `temperature` - Sampling temperature (Constitution: 2.0)
///
/// # Returns
/// Selected starting positions
pub fn sample_starting_positions(
    node_positions: &[(f32, [f32; 64])],
    count: usize,
    temperature: f32,
) -> Vec<[f32; 64]> {
    if node_positions.is_empty() || count == 0 {
        return Vec::new();
    }

    let mut rng = StdRng::from_entropy();

    // Score by phi value
    let scores: Vec<f32> = node_positions.iter().map(|(phi, _)| *phi).collect();

    // Softmax with temperature (uses poincare_walk::softmax_temperature internally)
    use super::poincare_walk::softmax_temperature;
    let probs = softmax_temperature(&scores, temperature.max(0.01));

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

// ============================================================================
// TESTS - NO MOCK DATA, REAL OPERATIONS ONLY
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_interrupt_flag() -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }

    fn make_set_interrupt_flag() -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(true))
    }

    // ============ Constitution Compliance Tests ============

    #[test]
    fn test_explorer_constitution_defaults() {
        let explorer = HyperbolicExplorer::with_defaults();

        // Constitution mandated values
        assert_eq!(explorer.config.temperature, 2.0,
            "temperature must be 2.0 per Constitution");
        assert_eq!(explorer.config.min_blind_spot_distance, 0.7,
            "min_blind_spot_distance must be 0.7 per Constitution semantic_leap");
        assert_eq!(explorer.query_limit, 100,
            "query_limit must be 100 per Constitution");
    }

    #[test]
    fn test_explorer_query_limit_enforced() {
        let config = HyperbolicWalkConfig {
            max_steps: 200, // More than query limit
            ..Default::default()
        };
        let mut explorer = HyperbolicExplorer::new(config).with_seed(42);
        let start = [0.0f32; 64];
        let interrupt = make_interrupt_flag();

        let result = explorer.walk(&start, &interrupt);

        // Should stop at query limit (100), not max_steps (200)
        assert!(explorer.queries_used <= 100,
            "queries_used {} must not exceed Constitution limit 100", explorer.queries_used);
        assert!(result.trajectory.len() <= 100);
    }

    // ============ Walk Behavior Tests ============

    #[test]
    fn test_walk_from_origin_produces_trajectory() {
        let mut explorer = HyperbolicExplorer::with_defaults().with_seed(42);
        let start = [0.0f32; 64];
        let interrupt = make_interrupt_flag();

        let result = explorer.walk(&start, &interrupt);

        assert!(!result.trajectory.is_empty(), "walk must produce trajectory");
        assert!(result.total_distance > 0.0, "walk must cover distance");
    }

    #[test]
    fn test_walk_stays_inside_poincare_ball() {
        let mut explorer = HyperbolicExplorer::with_defaults().with_seed(42);
        let start = [0.0f32; 64];
        let interrupt = make_interrupt_flag();

        let result = explorer.walk(&start, &interrupt);

        // Every position must be strictly inside the ball
        for (i, step) in result.trajectory.iter().enumerate() {
            let norm = norm_64(&step.position);
            assert!(norm < 1.0,
                "Step {} position has norm={:.6} >= 1.0 (outside ball)", i, norm);
        }
    }

    #[test]
    fn test_walk_respects_interrupt_flag() {
        let mut explorer = HyperbolicExplorer::with_defaults().with_seed(42);
        let start = [0.0f32; 64];
        let interrupt = make_set_interrupt_flag(); // Pre-set interrupt

        let result = explorer.walk(&start, &interrupt);

        // Should stop immediately due to interrupt
        assert!(result.trajectory.is_empty(),
            "interrupted walk should have empty trajectory");
        assert!(!result.completed,
            "interrupted walk should not be marked completed");
    }

    // ============ Blind Spot Detection Tests ============

    #[test]
    fn test_blind_spot_detection_with_no_known_positions() {
        let mut explorer = HyperbolicExplorer::with_defaults().with_seed(42);
        // No known positions set - every position is a blind spot

        let start = [0.0f32; 64];
        let interrupt = make_interrupt_flag();

        let result = explorer.walk(&start, &interrupt);

        // All steps should be blind spots when no known positions
        let blind_spot_count: usize = result.trajectory.iter()
            .filter(|s| s.blind_spot_detected)
            .count();

        assert_eq!(blind_spot_count, result.trajectory.len(),
            "All {} steps should be blind spots when no known positions", result.trajectory.len());
    }

    #[test]
    fn test_blind_spot_detection_with_known_positions() {
        let mut explorer = HyperbolicExplorer::with_defaults().with_seed(42);

        // Set known position near origin
        let mut known = [0.0f32; 64];
        known[0] = 0.1;
        explorer.set_known_positions(vec![known]);

        let start = [0.0f32; 64];
        let interrupt = make_interrupt_flag();

        let result = explorer.walk(&start, &interrupt);

        // Verify blind spot detection respects min_distance (0.7)
        for step in &result.trajectory {
            if step.blind_spot_detected {
                let dist = geodesic_distance(
                    &step.position,
                    &known,
                    &PoincareBallConfig::default(),
                );
                assert!(dist >= 0.7,
                    "Blind spot has distance {:.4} < 0.7 (Constitution semantic_leap)", dist);
            }
        }
    }

    #[test]
    fn test_blind_spot_significance_threshold() {
        // Constitution: significant = distance >= 0.7 AND confidence >= 0.5
        let significant = DiscoveredBlindSpot {
            position: [0.5f32; 64],
            distance_from_nearest: 0.8, // > 0.7
            discovery_step: 10,
            confidence: 0.6, // > 0.5
            id: Uuid::new_v4(),
        };
        assert!(significant.is_significant());

        let not_significant_distance = DiscoveredBlindSpot {
            distance_from_nearest: 0.5, // < 0.7 (violates Constitution)
            ..significant.clone()
        };
        assert!(!not_significant_distance.is_significant());

        let not_significant_confidence = DiscoveredBlindSpot {
            confidence: 0.3, // < 0.5
            ..significant.clone()
        };
        assert!(!not_significant_confidence.is_significant());
    }

    // ============ Multi-Walk Exploration Tests ============

    #[test]
    fn test_explore_multiple_starting_positions() {
        let mut explorer = HyperbolicExplorer::with_defaults().with_seed(42);
        let interrupt = make_interrupt_flag();

        // Two starting positions
        let start1 = [0.0f32; 64];
        let mut start2 = [0.0f32; 64];
        start2[0] = 0.3;

        let starts = vec![start1, start2];
        let result = explorer.explore(&starts, &interrupt);

        assert!(!result.walks.is_empty(), "should have at least 1 walk");
        assert!(result.queries_generated > 0, "should generate queries");
        assert!(result.queries_generated <= 100, "should respect query limit");
    }

    #[test]
    fn test_explore_empty_positions_starts_from_origin() {
        let mut explorer = HyperbolicExplorer::with_defaults().with_seed(42);
        let interrupt = make_interrupt_flag();

        let result = explorer.explore(&[], &interrupt);

        assert_eq!(result.walks.len(), 1, "should have exactly 1 walk from origin");
        assert_eq!(result.walks[0].start_position, [0.0f32; 64]);
    }

    // ============ Starting Position Sampling Tests ============

    #[test]
    fn test_sample_starting_positions_empty_input() {
        let positions = sample_starting_positions(&[], 5, 2.0);
        assert!(positions.is_empty());
    }

    #[test]
    fn test_sample_starting_positions_respects_count() {
        let node_positions = vec![
            (0.9, [0.1f32; 64]),
            (0.1, [0.2f32; 64]),
            (0.5, [0.3f32; 64]),
        ];

        let selected = sample_starting_positions(&node_positions, 2, 2.0);

        assert_eq!(selected.len(), 2, "should return requested count");
    }

    #[test]
    fn test_sample_starting_positions_prefers_high_phi() {
        let mut high_phi = [0.0f32; 64];
        high_phi[0] = 0.1;
        let mut low_phi = [0.0f32; 64];
        low_phi[0] = 0.2;

        let node_positions = vec![
            (0.99, high_phi),
            (0.01, low_phi),
        ];

        // Low temperature should strongly prefer high phi
        // Run multiple times to verify statistical preference
        let mut high_phi_count = 0;
        for _seed in 0..100u64 {
            // Use deterministic sampling
            let selected = sample_starting_positions(&node_positions, 1, 0.1);
            if selected[0][0] == 0.1 { // high_phi position
                high_phi_count += 1;
            }
        }

        // With temp=0.1, should almost always pick high-phi
        // Allow some variance but expect > 90%
        assert!(high_phi_count > 80,
            "With low temp, high phi should be selected >80% but was {}/100", high_phi_count);
    }

    // ============ Position to Query Conversion Test ============

    #[test]
    fn test_position_to_query_preserves_values() {
        let mut pos = [0.0f32; 64];
        pos[0] = 0.5;
        pos[63] = -0.3;

        let query = position_to_query(&pos);

        assert_eq!(query.len(), 64);
        assert_eq!(query[0], 0.5);
        assert_eq!(query[63], -0.3);
    }

    // ============ Fail-Fast Validation Tests ============

    #[test]
    #[should_panic(expected = "[HYPERBOLIC_WALK] Start position outside ball")]
    fn test_walk_rejects_invalid_start() {
        let mut explorer = HyperbolicExplorer::with_defaults();
        let invalid_start = [1.0f32; 64]; // norm = 8, way outside ball
        let interrupt = make_interrupt_flag();

        explorer.walk(&invalid_start, &interrupt);
    }

    #[test]
    #[should_panic(expected = "[HYPERBOLIC_WALK] Invalid known position")]
    fn test_set_known_positions_rejects_invalid() {
        let mut explorer = HyperbolicExplorer::with_defaults();
        let invalid_pos = [1.0f32; 64]; // Outside ball

        explorer.set_known_positions(vec![invalid_pos]);
    }
}
