//! Poincare Ball Math Utilities for Dream Walks
//!
//! Implements hyperbolic geometry operations for REM phase exploration:
//! - Mobius addition for random walk steps
//! - Geodesic distance for blind spot detection
//! - Random direction sampling with temperature
//! - Projection to keep points inside the ball
//!
//! Constitution Reference: docs2/constitution.yaml lines 391-394
//!   - temperature: 2.0
//!   - semantic_leap: 0.7

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// Configuration for Poincare ball operations.
///
/// Uses constitution-mandated values for numerical stability
/// during long random walks in REM phase.
#[derive(Debug, Clone, Copy)]
pub struct PoincareBallConfig {
    /// Maximum norm for valid points (< 1.0)
    /// Constitution: derived from boundary stability requirements
    pub max_norm: f32,

    /// Epsilon for numerical stability
    pub epsilon: f32,

    /// Curvature (negative for hyperbolic, standard = -1.0)
    pub curvature: f32,
}

impl Default for PoincareBallConfig {
    fn default() -> Self {
        Self {
            max_norm: 0.99999,
            epsilon: 1e-7,
            curvature: -1.0,
        }
    }
}

impl PoincareBallConfig {
    /// Validate configuration values.
    ///
    /// # Panics
    /// Panics with detailed error if configuration is invalid.
    pub fn validate(&self) {
        if self.max_norm >= 1.0 || self.max_norm <= 0.0 {
            panic!(
                "[POINCARE_WALK] Invalid max_norm at {}:{}: expected 0 < max_norm < 1.0, got {:.6}",
                file!(), line!(), self.max_norm
            );
        }
        if self.epsilon <= 0.0 || self.epsilon >= 1e-3 {
            panic!(
                "[POINCARE_WALK] Invalid epsilon at {}:{}: expected 0 < epsilon < 1e-3, got {:e}",
                file!(), line!(), self.epsilon
            );
        }
        if self.curvature >= 0.0 {
            panic!(
                "[POINCARE_WALK] Invalid curvature at {}:{}: expected < 0 for hyperbolic, got {:.6}",
                file!(), line!(), self.curvature
            );
        }
    }
}

/// Compute squared Euclidean norm of a 64D vector.
#[inline]
pub fn norm_squared_64(v: &[f32; 64]) -> f32 {
    v.iter().map(|&x| x * x).sum()
}

/// Compute Euclidean norm of a 64D vector.
#[inline]
pub fn norm_64(v: &[f32; 64]) -> f32 {
    norm_squared_64(v).sqrt()
}

/// Compute inner product of two 64D vectors.
#[inline]
pub fn inner_product_64(a: &[f32; 64], b: &[f32; 64]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Validate that a point is strictly inside the Poincare ball.
///
/// # Panics
/// Panics with detailed error if point is outside the ball.
#[inline]
pub fn validate_in_ball(point: &[f32; 64], config: &PoincareBallConfig, context: &str) {
    let norm = norm_64(point);
    if norm >= config.max_norm {
        panic!(
            "[POINCARE_WALK] Point outside ball at {}:{} ({}): norm = {:.6}, max = {:.6}",
            file!(), line!(), context, norm, config.max_norm
        );
    }
}

/// Project a point to stay strictly inside the Poincare ball.
///
/// If norm >= max_norm, rescales the point to have norm = max_norm - epsilon.
///
/// # Arguments
/// * `point` - Point to project (modified in place)
/// * `config` - Ball configuration with max_norm
///
/// # Returns
/// Whether projection was needed
pub fn project_to_ball(point: &mut [f32; 64], config: &PoincareBallConfig) -> bool {
    let norm = norm_64(point);

    if norm >= config.max_norm {
        let target_norm = config.max_norm - config.epsilon;
        let scale = target_norm / norm.max(config.epsilon);

        for x in point.iter_mut() {
            *x *= scale;
        }
        true
    } else {
        false
    }
}

/// Mobius addition in the Poincare ball model.
///
/// Computes p ⊕ v in hyperbolic space using the Mobius addition formula:
///
/// ```text
/// p ⊕ v = ((1 + 2c<p,v> + c||v||²)p + (1 - c||p||²)v) / (1 + 2c<p,v> + c²||p||²||v||²)
/// ```
///
/// where c = |curvature| (typically 1.0).
///
/// # Arguments
/// * `p` - Current position in Poincare ball
/// * `v` - Velocity/displacement vector
/// * `config` - Ball configuration
///
/// # Returns
/// New position after Mobius addition, projected to stay in ball
///
/// # Panics
/// Panics if input point p is outside the ball.
pub fn mobius_add(
    p: &[f32; 64],
    v: &[f32; 64],
    config: &PoincareBallConfig,
) -> [f32; 64] {
    // Fail fast: validate input
    validate_in_ball(p, config, "mobius_add input p");

    let c = config.curvature.abs(); // typically 1.0
    let p_sq = norm_squared_64(p);
    let v_sq = norm_squared_64(v);
    let pv = inner_product_64(p, v);

    // Denominator: 1 + 2c<p,v> + c²||p||²||v||²
    let denom = 1.0 + 2.0 * c * pv + c * c * p_sq * v_sq;

    // Fail fast on degenerate case
    if denom.abs() < config.epsilon {
        panic!(
            "[POINCARE_WALK] Degenerate Mobius addition at {}:{}: denom = {:e}",
            file!(), line!(), denom
        );
    }

    // Numerator coefficients
    let coeff_p = 1.0 + 2.0 * c * pv + c * v_sq;
    let coeff_v = 1.0 - c * p_sq;

    // Compute result
    let mut result = [0.0f32; 64];
    for i in 0..64 {
        result[i] = (coeff_p * p[i] + coeff_v * v[i]) / denom;
    }

    // Project to ensure we stay in the ball
    project_to_ball(&mut result, config);

    result
}

/// Compute geodesic distance in the Poincare ball model.
///
/// Uses the formula:
/// ```text
/// d(p, q) = (1/√c) * acosh(1 + 2c||p - q||² / ((1 - c||p||²)(1 - c||q||²)))
/// ```
///
/// # Arguments
/// * `p` - First point
/// * `q` - Second point
/// * `config` - Ball configuration
///
/// # Returns
/// Geodesic distance (always >= 0)
///
/// # Panics
/// Panics if either input point is outside the ball.
pub fn geodesic_distance(
    p: &[f32; 64],
    q: &[f32; 64],
    config: &PoincareBallConfig,
) -> f32 {
    // Fail fast: validate inputs
    validate_in_ball(p, config, "geodesic_distance input p");
    validate_in_ball(q, config, "geodesic_distance input q");

    let c = config.curvature.abs();
    let p_sq = norm_squared_64(p);
    let q_sq = norm_squared_64(q);

    // ||p - q||²
    let diff_sq: f32 = p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi - qi).powi(2))
        .sum();

    // Denominators with epsilon guard
    let denom_p = (1.0 - c * p_sq).max(config.epsilon);
    let denom_q = (1.0 - c * q_sq).max(config.epsilon);

    // Argument to acosh
    let arg = 1.0 + 2.0 * c * diff_sq / (denom_p * denom_q);

    // acosh(x) = ln(x + sqrt(x² - 1)) for x >= 1
    if arg <= 1.0 {
        return 0.0;
    }

    let sqrt_c = c.sqrt();
    (arg + (arg * arg - 1.0).sqrt()).ln() / sqrt_c
}

/// Generate a random direction vector on the 64D unit sphere.
///
/// Uses the Gaussian method: sample from N(0,1) for each component,
/// then normalize to unit length.
///
/// # Arguments
/// * `rng` - Random number generator
///
/// # Returns
/// Unit vector in R^64
///
/// # Panics
/// Panics if RNG produces degenerate all-zero samples.
pub fn random_direction<R: Rng>(rng: &mut R) -> [f32; 64] {
    let normal = StandardNormal;
    let mut direction = [0.0f32; 64];

    for x in direction.iter_mut() {
        *x = normal.sample(rng);
    }

    // Normalize
    let norm = norm_64(&direction);
    if norm < 1e-10 {
        panic!(
            "[POINCARE_WALK] Degenerate random direction at {}:{}: norm = {:e}",
            file!(), line!(), norm
        );
    }

    for x in direction.iter_mut() {
        *x /= norm;
    }

    direction
}

/// Compute softmax with temperature.
///
/// P(i) = exp(score_i / T) / sum_j(exp(score_j / T))
///
/// Constitution: temperature = 2.0 (line 393)
///
/// # Arguments
/// * `scores` - Raw scores
/// * `temperature` - Temperature parameter (higher = more uniform)
///
/// # Returns
/// Probability distribution over scores
///
/// # Panics
/// Panics if scores is empty or temperature is invalid.
pub fn softmax_temperature(scores: &[f32], temperature: f32) -> Vec<f32> {
    if scores.is_empty() {
        panic!(
            "[POINCARE_WALK] Empty scores array at {}:{}",
            file!(), line!()
        );
    }

    if temperature <= 0.0 {
        panic!(
            "[POINCARE_WALK] Invalid temperature at {}:{}: expected > 0, got {:.6}",
            file!(), line!(), temperature
        );
    }

    // Scale by temperature
    let scaled: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();

    // Find max for numerical stability (log-sum-exp trick)
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max)
    let exp_scores: Vec<f32> = scaled.iter().map(|&s| (s - max).exp()).collect();

    // Normalize
    let sum: f32 = exp_scores.iter().sum();
    if sum < 1e-10 {
        panic!(
            "[POINCARE_WALK] Softmax sum underflow at {}:{}: sum = {:e}",
            file!(), line!(), sum
        );
    }

    exp_scores.iter().map(|&e| e / sum).collect()
}

/// Sample multiple random directions and select one via softmax with temperature.
///
/// Higher temperature (> 1.0) makes selection more uniform (exploratory).
/// Lower temperature (< 1.0) makes selection more greedy toward high scores.
///
/// Constitution: temperature = 2.0 (line 393)
///
/// # Arguments
/// * `rng` - Random number generator
/// * `n_samples` - Number of directions to sample (must be > 0)
/// * `scores` - Optional scores for each direction (if None, uniform selection)
/// * `temperature` - Softmax temperature
///
/// # Returns
/// Selected direction vector
///
/// # Panics
/// Panics if n_samples is 0 or scores length doesn't match n_samples.
pub fn sample_direction_with_temperature<R: Rng>(
    rng: &mut R,
    n_samples: usize,
    scores: Option<&[f32]>,
    temperature: f32,
) -> [f32; 64] {
    if n_samples == 0 {
        panic!(
            "[POINCARE_WALK] n_samples must be > 0 at {}:{}",
            file!(), line!()
        );
    }

    // Validate scores length if provided
    if let Some(s) = scores {
        if s.len() != n_samples {
            panic!(
                "[POINCARE_WALK] Scores length mismatch at {}:{}: expected {}, got {}",
                file!(), line!(), n_samples, s.len()
            );
        }
    }

    // Generate candidate directions
    let candidates: Vec<[f32; 64]> = (0..n_samples)
        .map(|_| random_direction(rng))
        .collect();

    // Use provided scores or uniform
    let scores_vec: Vec<f32> = match scores {
        Some(s) => s.to_vec(),
        None => vec![1.0; n_samples],
    };

    // Apply softmax with temperature
    let probs = softmax_temperature(&scores_vec, temperature);

    // Sample from distribution
    let mut cumulative = 0.0;
    let threshold: f32 = rng.gen();

    for (i, &prob) in probs.iter().enumerate() {
        cumulative += prob;
        if threshold < cumulative {
            return candidates[i];
        }
    }

    // Return last candidate (floating point edge case)
    candidates[n_samples - 1]
}

/// Scale a direction vector by step size, respecting Poincare geometry.
///
/// In hyperbolic space, movement near the boundary requires smaller
/// Euclidean steps to achieve the same geodesic distance.
///
/// # Arguments
/// * `direction` - Unit direction vector
/// * `step_size` - Desired step size in Euclidean terms
/// * `current_norm` - Current position's norm
/// * `config` - Ball configuration
///
/// # Returns
/// Scaled velocity vector safe for Mobius addition
///
/// # Panics
/// Panics if direction is not unit length or current_norm >= max_norm.
pub fn scale_direction(
    direction: &[f32; 64],
    step_size: f32,
    current_norm: f32,
    config: &PoincareBallConfig,
) -> [f32; 64] {
    // Validate direction is unit length
    let dir_norm = norm_64(direction);
    if (dir_norm - 1.0).abs() > 1e-4 {
        panic!(
            "[POINCARE_WALK] Direction not unit length at {}:{}: norm = {:.6}",
            file!(), line!(), dir_norm
        );
    }

    // Validate current position is inside ball
    if current_norm >= config.max_norm {
        panic!(
            "[POINCARE_WALK] Current position outside ball at {}:{}: norm = {:.6}",
            file!(), line!(), current_norm
        );
    }

    // Near the boundary, we need smaller steps
    // Factor: (1 - ||p||²) / 2 scales appropriately
    let boundary_factor = ((1.0 - current_norm * current_norm) / 2.0).max(config.epsilon);
    let effective_step = step_size * boundary_factor;

    let mut result = *direction;
    for x in result.iter_mut() {
        *x *= effective_step;
    }

    result
}

/// Check if a point is far from all reference points (blind spot detection).
///
/// Constitution: semantic_leap >= 0.7 (line 394)
///
/// # Arguments
/// * `point` - Point to check
/// * `reference_points` - Set of reference points (visited nodes)
/// * `min_distance` - Minimum geodesic distance to be "far" (0.7 per constitution)
/// * `config` - Ball configuration
///
/// # Returns
/// True if point is far from all reference points (potential blind spot)
pub fn is_far_from_all(
    point: &[f32; 64],
    reference_points: &[[f32; 64]],
    min_distance: f32,
    config: &PoincareBallConfig,
) -> bool {
    for ref_point in reference_points {
        let dist = geodesic_distance(point, ref_point, config);
        if dist < min_distance {
            return false;
        }
    }
    true
}

/// Compute the Riemannian gradient direction from p toward q.
///
/// This gives the direction to move in Poincare ball to approach q.
///
/// # Arguments
/// * `p` - Current position
/// * `q` - Target position
/// * `config` - Ball configuration
///
/// # Returns
/// Direction vector (not normalized)
pub fn direction_toward(
    p: &[f32; 64],
    q: &[f32; 64],
    config: &PoincareBallConfig,
) -> [f32; 64] {
    // Use -p ⊕ q to get the direction
    let neg_p = {
        let mut neg = *p;
        for x in neg.iter_mut() {
            *x = -*x;
        }
        neg
    };

    mobius_add(&neg_p, q, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Deterministic RNG for reproducible tests
    fn make_rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(42)
    }

    /// Create a point at given norm along first axis
    fn point_at_norm(norm: f32) -> [f32; 64] {
        let mut p = [0.0f32; 64];
        p[0] = norm;
        p
    }

    // ============ Configuration Tests ============

    #[test]
    fn test_config_default_values() {
        let config = PoincareBallConfig::default();
        assert!((config.max_norm - 0.99999).abs() < 1e-8);
        assert!((config.epsilon - 1e-7).abs() < 1e-10);
        assert!((config.curvature - (-1.0)).abs() < 1e-8);
    }

    #[test]
    fn test_config_validate_passes() {
        let config = PoincareBallConfig::default();
        config.validate(); // Should not panic
    }

    #[test]
    #[should_panic(expected = "[POINCARE_WALK] Invalid max_norm")]
    fn test_config_rejects_max_norm_too_high() {
        let config = PoincareBallConfig {
            max_norm: 1.0, // Invalid: must be < 1.0
            ..Default::default()
        };
        config.validate();
    }

    #[test]
    #[should_panic(expected = "[POINCARE_WALK] Invalid curvature")]
    fn test_config_rejects_positive_curvature() {
        let config = PoincareBallConfig {
            curvature: 1.0, // Invalid: must be < 0
            ..Default::default()
        };
        config.validate();
    }

    // ============ Basic Math Tests ============

    #[test]
    fn test_norm_squared_64_known_value() {
        // 64 elements of 0.125, squared = 0.015625, sum = 1.0
        let v = [0.125f32; 64];
        let expected = 64.0 * 0.015625; // 1.0
        let actual = norm_squared_64(&v);
        assert!((actual - expected).abs() < 1e-6,
            "expected {}, got {}", expected, actual);
    }

    #[test]
    fn test_norm_64_known_value() {
        let v = [0.125f32; 64];
        let expected = 1.0f32; // sqrt(1.0)
        let actual = norm_64(&v);
        assert!((actual - expected).abs() < 1e-6,
            "expected {}, got {}", expected, actual);
    }

    #[test]
    fn test_inner_product_64_orthogonal() {
        let mut a = [0.0f32; 64];
        let mut b = [0.0f32; 64];
        a[0] = 1.0;
        b[1] = 1.0;

        let result = inner_product_64(&a, &b);
        assert!(result.abs() < 1e-10, "orthogonal vectors should have 0 inner product");
    }

    #[test]
    fn test_inner_product_64_parallel() {
        let v = [0.125f32; 64];
        let result = inner_product_64(&v, &v);
        let expected = norm_squared_64(&v);
        assert!((result - expected).abs() < 1e-6);
    }

    // ============ Projection Tests ============

    #[test]
    fn test_project_to_ball_inside_unchanged() {
        let config = PoincareBallConfig::default();
        let mut point = point_at_norm(0.5);
        let original = point;

        let projected = project_to_ball(&mut point, &config);

        assert!(!projected, "should not need projection");
        assert_eq!(point, original, "point should be unchanged");
    }

    #[test]
    fn test_project_to_ball_outside_projected() {
        let config = PoincareBallConfig::default();
        let mut point = point_at_norm(1.5); // Outside ball

        let projected = project_to_ball(&mut point, &config);
        let new_norm = norm_64(&point);

        assert!(projected, "should need projection");
        assert!(new_norm < config.max_norm,
            "projected norm {} should be < max_norm {}", new_norm, config.max_norm);
    }

    #[test]
    fn test_project_to_ball_boundary_projected() {
        let config = PoincareBallConfig::default();
        let mut point = point_at_norm(0.99999); // At boundary

        let projected = project_to_ball(&mut point, &config);
        let new_norm = norm_64(&point);

        assert!(projected, "should need projection at boundary");
        assert!(new_norm < config.max_norm - config.epsilon / 2.0);
    }

    // ============ Mobius Addition Tests ============

    #[test]
    fn test_mobius_add_origin_returns_v() {
        let config = PoincareBallConfig::default();
        let origin = [0.0f32; 64];
        let v = point_at_norm(0.5);

        let result = mobius_add(&origin, &v, &config);

        // Adding v to origin should give approximately v
        assert!((result[0] - 0.5).abs() < 1e-5);
        for i in 1..64 {
            assert!(result[i].abs() < 1e-6);
        }
    }

    #[test]
    fn test_mobius_add_stays_in_ball() {
        let config = PoincareBallConfig::default();
        let mut rng = make_rng();

        for _ in 0..20 {
            let mut p = random_direction(&mut rng);
            for x in p.iter_mut() {
                *x *= 0.5;
            }

            let mut v = random_direction(&mut rng);
            for x in v.iter_mut() {
                *x *= 0.3;
            }

            let result = mobius_add(&p, &v, &config);
            let norm = norm_64(&result);

            assert!(norm < config.max_norm,
                "result norm {} should be < max_norm {}", norm, config.max_norm);
        }
    }

    #[test]
    #[should_panic(expected = "[POINCARE_WALK] Point outside ball")]
    fn test_mobius_add_rejects_outside_ball() {
        let config = PoincareBallConfig::default();
        let p = point_at_norm(1.5); // Outside ball
        let v = point_at_norm(0.1);

        mobius_add(&p, &v, &config);
    }

    // ============ Geodesic Distance Tests ============

    #[test]
    fn test_geodesic_distance_same_point_zero() {
        let config = PoincareBallConfig::default();
        let p = point_at_norm(0.5);

        let dist = geodesic_distance(&p, &p, &config);

        assert!(dist.abs() < 1e-6, "distance to self should be 0, got {}", dist);
    }

    #[test]
    fn test_geodesic_distance_symmetric() {
        let config = PoincareBallConfig::default();
        let mut rng = make_rng();

        for _ in 0..10 {
            let mut p = random_direction(&mut rng);
            for x in p.iter_mut() { *x *= 0.3; }

            let mut q = random_direction(&mut rng);
            for x in q.iter_mut() { *x *= 0.4; }

            let d1 = geodesic_distance(&p, &q, &config);
            let d2 = geodesic_distance(&q, &p, &config);

            assert!((d1 - d2).abs() < 1e-5,
                "distance should be symmetric: {} vs {}", d1, d2);
        }
    }

    #[test]
    fn test_geodesic_distance_triangle_inequality() {
        let config = PoincareBallConfig::default();
        let p = point_at_norm(0.3);
        let mut q = [0.0f32; 64];
        q[1] = 0.4;
        let mut r = [0.0f32; 64];
        r[2] = 0.5;

        let d_pq = geodesic_distance(&p, &q, &config);
        let d_qr = geodesic_distance(&q, &r, &config);
        let d_pr = geodesic_distance(&p, &r, &config);

        assert!(d_pr <= d_pq + d_qr + 1e-5,
            "triangle inequality violated: {} > {} + {}", d_pr, d_pq, d_qr);
    }

    #[test]
    #[should_panic(expected = "[POINCARE_WALK] Point outside ball")]
    fn test_geodesic_distance_rejects_outside_ball() {
        let config = PoincareBallConfig::default();
        let p = point_at_norm(1.5);
        let q = point_at_norm(0.5);

        geodesic_distance(&p, &q, &config);
    }

    // ============ Random Direction Tests ============

    #[test]
    fn test_random_direction_unit_length() {
        let mut rng = make_rng();

        for _ in 0..20 {
            let dir = random_direction(&mut rng);
            let norm = norm_64(&dir);

            assert!((norm - 1.0).abs() < 1e-5,
                "direction should be unit length, got {}", norm);
        }
    }

    #[test]
    fn test_random_direction_reproducible() {
        let mut rng1 = make_rng();
        let mut rng2 = make_rng();

        let dir1 = random_direction(&mut rng1);
        let dir2 = random_direction(&mut rng2);

        assert_eq!(dir1, dir2, "same seed should produce same direction");
    }

    // ============ Softmax Tests ============

    #[test]
    fn test_softmax_temperature_sums_to_one() {
        let scores = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax_temperature(&scores, 2.0); // Constitution temperature

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax should sum to 1, got {}", sum);
    }

    #[test]
    fn test_softmax_temperature_uniform_input() {
        let scores = vec![1.0, 1.0, 1.0, 1.0];
        let probs = softmax_temperature(&scores, 2.0);

        for p in &probs {
            assert!((*p - 0.25).abs() < 0.01, "uniform scores should give uniform probs");
        }
    }

    #[test]
    fn test_softmax_high_temp_more_uniform() {
        let scores = vec![1.0, 2.0, 3.0];

        let probs_high = softmax_temperature(&scores, 10.0);
        let probs_low = softmax_temperature(&scores, 0.1);

        let range_high = probs_high.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - probs_high.iter().cloned().fold(f32::INFINITY, f32::min);
        let range_low = probs_low.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - probs_low.iter().cloned().fold(f32::INFINITY, f32::min);

        assert!(range_high < range_low,
            "high temp range {} should be < low temp range {}", range_high, range_low);
    }

    #[test]
    #[should_panic(expected = "[POINCARE_WALK] Empty scores array")]
    fn test_softmax_rejects_empty() {
        let scores: Vec<f32> = vec![];
        softmax_temperature(&scores, 2.0);
    }

    #[test]
    #[should_panic(expected = "[POINCARE_WALK] Invalid temperature")]
    fn test_softmax_rejects_zero_temp() {
        let scores = vec![1.0, 2.0];
        softmax_temperature(&scores, 0.0);
    }

    // ============ Direction Sampling Tests ============

    #[test]
    fn test_sample_direction_returns_unit() {
        let mut rng = make_rng();
        let dir = sample_direction_with_temperature(&mut rng, 5, None, 2.0);
        let norm = norm_64(&dir);

        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    #[should_panic(expected = "[POINCARE_WALK] n_samples must be > 0")]
    fn test_sample_direction_rejects_zero_samples() {
        let mut rng = make_rng();
        sample_direction_with_temperature(&mut rng, 0, None, 2.0);
    }

    #[test]
    #[should_panic(expected = "[POINCARE_WALK] Scores length mismatch")]
    fn test_sample_direction_rejects_wrong_scores_length() {
        let mut rng = make_rng();
        let scores = vec![1.0, 2.0]; // 2 scores
        sample_direction_with_temperature(&mut rng, 5, Some(&scores), 2.0); // 5 samples
    }

    // ============ Scale Direction Tests ============

    #[test]
    fn test_scale_direction_smaller_near_boundary() {
        let config = PoincareBallConfig::default();
        let dir = {
            let mut d = [0.0f32; 64];
            d[0] = 1.0;
            d
        };

        let scaled_origin = scale_direction(&dir, 0.1, 0.0, &config);
        let scaled_boundary = scale_direction(&dir, 0.1, 0.9, &config);

        let norm_origin = norm_64(&scaled_origin);
        let norm_boundary = norm_64(&scaled_boundary);

        assert!(norm_origin > norm_boundary,
            "step near origin ({}) should be larger than near boundary ({})",
            norm_origin, norm_boundary);
    }

    #[test]
    #[should_panic(expected = "[POINCARE_WALK] Direction not unit length")]
    fn test_scale_direction_rejects_non_unit() {
        let config = PoincareBallConfig::default();
        let dir = point_at_norm(0.5); // Not unit length
        scale_direction(&dir, 0.1, 0.0, &config);
    }

    // ============ Blind Spot Detection Tests ============

    #[test]
    fn test_is_far_from_all_empty_refs() {
        let config = PoincareBallConfig::default();
        let point = point_at_norm(0.5);
        let refs: Vec<[f32; 64]> = vec![];

        assert!(is_far_from_all(&point, &refs, 0.7, &config),
            "should be far from empty reference set");
    }

    #[test]
    fn test_is_far_from_all_close_ref() {
        let config = PoincareBallConfig::default();
        let point = point_at_norm(0.5);
        let ref_point = point_at_norm(0.51); // Very close
        let refs = vec![ref_point];

        assert!(!is_far_from_all(&point, &refs, 0.7, &config),
            "should NOT be far from close reference");
    }

    #[test]
    fn test_is_far_from_all_semantic_leap_threshold() {
        // Constitution: semantic_leap >= 0.7
        let config = PoincareBallConfig::default();
        let origin = [0.0f32; 64];

        // Create a reference point that's exactly at semantic_leap distance
        let mut ref_point = [0.0f32; 64];
        ref_point[0] = 0.6; // Creates meaningful distance from origin
        let refs = vec![ref_point];

        let dist = geodesic_distance(&origin, &ref_point, &config);

        // Point should be classified based on 0.7 threshold
        let result = is_far_from_all(&origin, &refs, 0.7, &config);
        assert_eq!(result, dist >= 0.7,
            "is_far_from_all result {} inconsistent with distance {} vs threshold 0.7",
            result, dist);
    }

    // ============ Direction Toward Tests ============

    #[test]
    fn test_direction_toward_reduces_distance() {
        let config = PoincareBallConfig::default();
        let p = point_at_norm(0.3);
        let mut q = [0.0f32; 64];
        q[1] = 0.4;

        let dir = direction_toward(&p, &q, &config);

        // Taking a step in this direction should reduce distance
        let mut scaled_dir = dir;
        for x in scaled_dir.iter_mut() { *x *= 0.01; }
        let p_new = mobius_add(&p, &scaled_dir, &config);

        let dist_before = geodesic_distance(&p, &q, &config);
        let dist_after = geodesic_distance(&p_new, &q, &config);

        assert!(dist_after < dist_before,
            "moving toward q should reduce distance: {} -> {}", dist_before, dist_after);
    }
}
