# TASK-DREAM-P0-002: Poincare Ball Math Utilities

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-DREAM-P0-002 |
| **Spec Ref** | SPEC-DREAM-001 |
| **Layer** | 1 (Foundation) |
| **Priority** | P0 - Critical |
| **Effort** | 3 hours |
| **Dependencies** | TASK-DREAM-P0-001 |
| **Blocks** | TASK-DREAM-P0-004 |

---

## 1. Objective

Implement Poincare ball mathematical utilities for hyperbolic random walks in REM phase. This includes Mobius addition, geodesic distance, random direction generation, and projection utilities specific to dream exploration.

---

## 2. Input Context Files

```yaml
must_read:
  - path: crates/context-graph-graph/src/hyperbolic/poincare/types.rs
    purpose: Existing PoincarePoint type definition
  - path: crates/context-graph-graph/src/hyperbolic/poincare/ops.rs
    purpose: Existing Poincare operations (project, norm)
  - path: crates/context-graph-graph/src/hyperbolic/mobius/operations.rs
    purpose: Existing Mobius operations for reference
  - path: crates/context-graph-graph/src/hyperbolic/mobius/maps.rs
    purpose: Mobius maps for transformations
  - path: crates/context-graph-core/src/dream/types.rs
    purpose: WalkStep and HyperbolicWalkConfig types (from TASK-001)

should_read:
  - path: crates/context-graph-graph/src/config/hyperbolic.rs
    purpose: HyperbolicConfig for max_norm and epsilon values
  - path: crates/context-graph-cuda/src/poincare/cpu.rs
    purpose: CPU fallback implementations for reference
```

---

## 3. Files to Create/Modify

### 3.1 Create: `crates/context-graph-core/src/dream/poincare_walk.rs`

```rust
//! Poincare Ball Math Utilities for Dream Walks
//!
//! Implements hyperbolic geometry operations for REM phase exploration:
//! - Mobius addition for random walk steps
//! - Geodesic distance for blind spot detection
//! - Random direction sampling with temperature
//! - Projection to keep points inside the ball

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// Configuration for Poincare ball operations.
///
/// Uses stricter bounds than general hyperbolic config to ensure
/// numerical stability during long random walks.
#[derive(Debug, Clone, Copy)]
pub struct PoincareBallConfig {
    /// Maximum norm for valid points (< 1.0)
    pub max_norm: f32,

    /// Epsilon for numerical stability
    pub epsilon: f32,

    /// Curvature (negative for hyperbolic, usually -1.0)
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

/// Project a point to stay strictly inside the Poincare ball.
///
/// If norm >= max_norm, rescales the point to have norm = max_norm - epsilon.
///
/// # Arguments
///
/// * `point` - Point to project (modified in place)
/// * `config` - Ball configuration with max_norm
///
/// # Returns
///
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
/// Computes p + v in hyperbolic space using the Mobius addition formula:
///
/// ```text
/// p ⊕ v = ((1 + 2<p,v> + ||v||²)p + (1 - ||p||²)v) / (1 + 2<p,v> + ||p||²||v||²)
/// ```
///
/// This is the fundamental operation for moving in hyperbolic space.
///
/// # Arguments
///
/// * `p` - Current position in Poincare ball
/// * `v` - Velocity/displacement vector
/// * `config` - Ball configuration
///
/// # Returns
///
/// New position after Mobius addition, projected to stay in ball
pub fn mobius_add(
    p: &[f32; 64],
    v: &[f32; 64],
    config: &PoincareBallConfig,
) -> [f32; 64] {
    let p_sq = norm_squared_64(p);
    let v_sq = norm_squared_64(v);
    let pv = inner_product_64(p, v);

    // Denominator: 1 + 2<p,v> + ||p||²||v||²
    let denom = 1.0 + 2.0 * pv + p_sq * v_sq;

    // Avoid division by zero
    if denom.abs() < config.epsilon {
        return *p; // Return unchanged if degenerate
    }

    // Numerator coefficients
    let coeff_p = 1.0 + 2.0 * pv + v_sq;
    let coeff_v = 1.0 - p_sq;

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
/// d(p, q) = acosh(1 + 2||p - q||² / ((1 - ||p||²)(1 - ||q||²)))
/// ```
///
/// # Arguments
///
/// * `p` - First point
/// * `q` - Second point
/// * `config` - Ball configuration
///
/// # Returns
///
/// Geodesic distance (always >= 0)
pub fn geodesic_distance(
    p: &[f32; 64],
    q: &[f32; 64],
    config: &PoincareBallConfig,
) -> f32 {
    let p_sq = norm_squared_64(p);
    let q_sq = norm_squared_64(q);

    // ||p - q||²
    let diff_sq: f32 = p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi - qi).powi(2))
        .sum();

    // Denominators
    let denom_p = (1.0 - p_sq).max(config.epsilon);
    let denom_q = (1.0 - q_sq).max(config.epsilon);

    // Argument to acosh
    let arg = 1.0 + 2.0 * diff_sq / (denom_p * denom_q);

    // acosh(x) = ln(x + sqrt(x² - 1)) for x >= 1
    // Handle numerical edge case where arg might be slightly < 1
    if arg <= 1.0 {
        return 0.0;
    }

    (arg + (arg * arg - 1.0).sqrt()).ln()
}

/// Generate a random direction vector on the 64D unit sphere.
///
/// Uses the Gaussian method: sample from N(0,1) for each component,
/// then normalize to unit length.
///
/// # Arguments
///
/// * `rng` - Random number generator
///
/// # Returns
///
/// Unit vector in R^64
pub fn random_direction<R: Rng>(rng: &mut R) -> [f32; 64] {
    let normal = StandardNormal;
    let mut direction = [0.0f32; 64];

    for x in direction.iter_mut() {
        *x = normal.sample(rng);
    }

    // Normalize
    let norm = norm_64(&direction);
    if norm > 1e-10 {
        for x in direction.iter_mut() {
            *x /= norm;
        }
    } else {
        // Degenerate case: return arbitrary direction
        direction[0] = 1.0;
    }

    direction
}

/// Sample multiple random directions and select one via softmax with temperature.
///
/// Higher temperature (> 1.0) makes selection more uniform (exploratory).
/// Lower temperature (< 1.0) makes selection more greedy toward high scores.
///
/// # Arguments
///
/// * `rng` - Random number generator
/// * `n_samples` - Number of directions to sample
/// * `scores` - Optional scores for each direction (if None, uniform selection)
/// * `temperature` - Softmax temperature (Constitution: 2.0)
///
/// # Returns
///
/// Selected direction vector
pub fn sample_direction_with_temperature<R: Rng>(
    rng: &mut R,
    n_samples: usize,
    scores: Option<&[f32]>,
    temperature: f32,
) -> [f32; 64] {
    if n_samples == 0 {
        return random_direction(rng);
    }

    // Generate candidate directions
    let candidates: Vec<[f32; 64]> = (0..n_samples)
        .map(|_| random_direction(rng))
        .collect();

    // If no scores provided, use uniform random selection
    let scores = match scores {
        Some(s) if s.len() == n_samples => s.to_vec(),
        _ => vec![1.0; n_samples],
    };

    // Apply softmax with temperature
    let probs = softmax_temperature(&scores, temperature);

    // Sample from distribution
    let mut cumulative = 0.0;
    let threshold: f32 = rng.gen();

    for (i, &prob) in probs.iter().enumerate() {
        cumulative += prob;
        if threshold < cumulative {
            return candidates[i];
        }
    }

    // Fallback to last candidate
    candidates.last().cloned().unwrap_or_else(|| random_direction(rng))
}

/// Compute softmax with temperature.
///
/// P(i) = exp(score_i / T) / sum_j(exp(score_j / T))
///
/// # Arguments
///
/// * `scores` - Raw scores
/// * `temperature` - Temperature parameter (higher = more uniform)
///
/// # Returns
///
/// Probability distribution over scores
pub fn softmax_temperature(scores: &[f32], temperature: f32) -> Vec<f32> {
    if scores.is_empty() {
        return vec![];
    }

    let temperature = temperature.max(0.01); // Prevent division by zero

    // Scale by temperature
    let scaled: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();

    // Find max for numerical stability
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max)
    let exp_scores: Vec<f32> = scaled.iter().map(|&s| (s - max).exp()).collect();

    // Normalize
    let sum: f32 = exp_scores.iter().sum();
    if sum < 1e-10 {
        // All zeros, return uniform
        return vec![1.0 / scores.len() as f32; scores.len()];
    }

    exp_scores.iter().map(|&e| e / sum).collect()
}

/// Scale a direction vector by step size, respecting Poincare geometry.
///
/// In hyperbolic space, movement near the boundary requires smaller
/// Euclidean steps to achieve the same geodesic distance.
///
/// # Arguments
///
/// * `direction` - Unit direction vector
/// * `step_size` - Desired step size in Euclidean terms
/// * `current_norm` - Current position's norm
/// * `config` - Ball configuration
///
/// # Returns
///
/// Scaled velocity vector safe for Mobius addition
pub fn scale_direction(
    direction: &[f32; 64],
    step_size: f32,
    current_norm: f32,
    config: &PoincareBallConfig,
) -> [f32; 64] {
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

/// Check if a point is near another set of points (for blind spot detection).
///
/// # Arguments
///
/// * `point` - Point to check
/// * `reference_points` - Set of reference points
/// * `min_distance` - Minimum geodesic distance to be considered "far"
/// * `config` - Ball configuration
///
/// # Returns
///
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
///
/// * `p` - Current position
/// * `q` - Target position
/// * `config` - Ball configuration
///
/// # Returns
///
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

    fn make_rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(42)
    }

    #[test]
    fn test_norm_squared_64() {
        let v = [0.1f32; 64];
        let expected = 64.0 * 0.01; // 0.64
        let actual = norm_squared_64(&v);
        assert!((actual - expected).abs() < 1e-6);
    }

    #[test]
    fn test_norm_64() {
        let v = [0.1f32; 64];
        let expected = (64.0 * 0.01_f32).sqrt(); // 0.8
        let actual = norm_64(&v);
        assert!((actual - expected).abs() < 1e-6);
    }

    #[test]
    fn test_inner_product_64() {
        let a = [0.5f32; 64];
        let b = [0.5f32; 64];
        let expected = 64.0 * 0.25; // 16.0
        let actual = inner_product_64(&a, &b);
        assert!((actual - expected).abs() < 1e-5);
    }

    #[test]
    fn test_project_to_ball_inside() {
        let config = PoincareBallConfig::default();
        let mut point = [0.1f32; 64]; // norm = 0.8, inside ball

        let projected = project_to_ball(&mut point, &config);

        assert!(!projected); // Should not need projection
    }

    #[test]
    fn test_project_to_ball_outside() {
        let config = PoincareBallConfig::default();
        let mut point = [0.2f32; 64]; // norm = 1.6, outside ball

        let projected = project_to_ball(&mut point, &config);

        assert!(projected); // Should need projection
        assert!(norm_64(&point) < 1.0); // Should be inside ball now
    }

    #[test]
    fn test_mobius_add_origin() {
        let config = PoincareBallConfig::default();
        let origin = [0.0f32; 64];
        let mut v = [0.0f32; 64];
        v[0] = 0.5;

        let result = mobius_add(&origin, &v, &config);

        // Adding v to origin should give v
        assert!((result[0] - 0.5).abs() < 1e-6);
        for i in 1..64 {
            assert!(result[i].abs() < 1e-6);
        }
    }

    #[test]
    fn test_mobius_add_stays_in_ball() {
        let config = PoincareBallConfig::default();
        let mut rng = make_rng();

        // Start at various points and add random directions
        for _ in 0..10 {
            let mut p = random_direction(&mut rng);
            for x in p.iter_mut() {
                *x *= 0.5; // Start at norm 0.5
            }

            let v = {
                let d = random_direction(&mut rng);
                let mut scaled = d;
                for x in scaled.iter_mut() {
                    *x *= 0.3;
                }
                scaled
            };

            let result = mobius_add(&p, &v, &config);
            let norm = norm_64(&result);

            assert!(norm < 1.0, "Result norm {} should be < 1.0", norm);
        }
    }

    #[test]
    fn test_geodesic_distance_same_point() {
        let config = PoincareBallConfig::default();
        let p = [0.1f32; 64];

        let dist = geodesic_distance(&p, &p, &config);

        assert!(dist.abs() < 1e-6, "Distance to self should be 0");
    }

    #[test]
    fn test_geodesic_distance_origin_to_point() {
        let config = PoincareBallConfig::default();
        let origin = [0.0f32; 64];
        let mut p = [0.0f32; 64];
        p[0] = 0.5;

        let dist = geodesic_distance(&origin, &p, &config);

        // Should be positive
        assert!(dist > 0.0, "Distance should be positive");

        // Should be finite
        assert!(dist.is_finite(), "Distance should be finite");
    }

    #[test]
    fn test_geodesic_distance_symmetric() {
        let config = PoincareBallConfig::default();
        let mut rng = make_rng();

        let mut p = random_direction(&mut rng);
        for x in p.iter_mut() {
            *x *= 0.3;
        }

        let mut q = random_direction(&mut rng);
        for x in q.iter_mut() {
            *x *= 0.4;
        }

        let d1 = geodesic_distance(&p, &q, &config);
        let d2 = geodesic_distance(&q, &p, &config);

        assert!((d1 - d2).abs() < 1e-6, "Distance should be symmetric");
    }

    #[test]
    fn test_random_direction_unit_length() {
        let mut rng = make_rng();

        for _ in 0..10 {
            let dir = random_direction(&mut rng);
            let norm = norm_64(&dir);

            assert!((norm - 1.0).abs() < 1e-6, "Direction should be unit length");
        }
    }

    #[test]
    fn test_softmax_temperature_uniform() {
        let scores = vec![1.0, 1.0, 1.0];
        let probs = softmax_temperature(&scores, 2.0);

        // Uniform scores should give uniform probs
        for p in &probs {
            assert!((*p - 0.333).abs() < 0.01);
        }

        // Should sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_temperature_high_temp() {
        let scores = vec![1.0, 2.0, 3.0];

        // High temperature should make distribution more uniform
        let probs_high = softmax_temperature(&scores, 10.0);
        let probs_low = softmax_temperature(&scores, 0.1);

        // With high temp, max prob should be closer to min prob
        let max_high = probs_high.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_high = probs_high.iter().cloned().fold(f32::INFINITY, f32::min);
        let range_high = max_high - min_high;

        let max_low = probs_low.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_low = probs_low.iter().cloned().fold(f32::INFINITY, f32::min);
        let range_low = max_low - min_low;

        assert!(range_high < range_low, "High temp should have smaller range");
    }

    #[test]
    fn test_scale_direction_boundary_factor() {
        let config = PoincareBallConfig::default();
        let dir = {
            let mut d = [0.0f32; 64];
            d[0] = 1.0;
            d
        };

        // Near origin, should have larger effective step
        let scaled_origin = scale_direction(&dir, 0.1, 0.0, &config);

        // Near boundary, should have smaller effective step
        let scaled_boundary = scale_direction(&dir, 0.1, 0.9, &config);

        let norm_origin = norm_64(&scaled_origin);
        let norm_boundary = norm_64(&scaled_boundary);

        assert!(
            norm_origin > norm_boundary,
            "Step near origin ({}) should be larger than near boundary ({})",
            norm_origin,
            norm_boundary
        );
    }

    #[test]
    fn test_is_far_from_all() {
        let config = PoincareBallConfig::default();
        let origin = [0.0f32; 64];

        // Point at origin should be close to references near origin
        let mut ref_near = [0.0f32; 64];
        ref_near[0] = 0.1;
        let references = vec![ref_near];

        // High threshold: origin is NOT far from ref_near
        assert!(!is_far_from_all(&origin, &references, 1.0, &config));

        // Low threshold: origin IS "far enough" from ref_near
        assert!(is_far_from_all(&origin, &references, 0.01, &config));
    }

    #[test]
    fn test_sample_direction_with_temperature() {
        let mut rng = make_rng();

        let dir = sample_direction_with_temperature(&mut rng, 5, None, 2.0);
        let norm = norm_64(&dir);

        // Should be unit vector
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
```

### 3.2 Modify: `crates/context-graph-core/src/dream/mod.rs`

Add export for poincare_walk module:

```rust
// Add after types module:
pub mod poincare_walk;

// Add to re-exports:
pub use poincare_walk::{
    PoincareBallConfig,
    mobius_add,
    geodesic_distance,
    random_direction,
    sample_direction_with_temperature,
    scale_direction,
    is_far_from_all,
    project_to_ball,
    softmax_temperature,
};
```

### 3.3 Update: `crates/context-graph-core/Cargo.toml`

Add dependencies if not present:

```toml
[dependencies]
rand = "0.8"
rand_distr = "0.4"
rand_chacha = "0.3"  # For tests
```

---

## 4. Definition of Done

### 4.1 Function Signatures (Exact)

```rust
pub fn norm_squared_64(v: &[f32; 64]) -> f32;
pub fn norm_64(v: &[f32; 64]) -> f32;
pub fn inner_product_64(a: &[f32; 64], b: &[f32; 64]) -> f32;

pub fn project_to_ball(point: &mut [f32; 64], config: &PoincareBallConfig) -> bool;

pub fn mobius_add(
    p: &[f32; 64],
    v: &[f32; 64],
    config: &PoincareBallConfig,
) -> [f32; 64];

pub fn geodesic_distance(
    p: &[f32; 64],
    q: &[f32; 64],
    config: &PoincareBallConfig,
) -> f32;

pub fn random_direction<R: Rng>(rng: &mut R) -> [f32; 64];

pub fn sample_direction_with_temperature<R: Rng>(
    rng: &mut R,
    n_samples: usize,
    scores: Option<&[f32]>,
    temperature: f32,
) -> [f32; 64];

pub fn softmax_temperature(scores: &[f32], temperature: f32) -> Vec<f32>;

pub fn scale_direction(
    direction: &[f32; 64],
    step_size: f32,
    current_norm: f32,
    config: &PoincareBallConfig,
) -> [f32; 64];

pub fn is_far_from_all(
    point: &[f32; 64],
    reference_points: &[[f32; 64]],
    min_distance: f32,
    config: &PoincareBallConfig,
) -> bool;
```

### 4.2 Validation Criteria

| Criterion | Check |
|-----------|-------|
| Compiles without errors | `cargo build -p context-graph-core` |
| All tests pass | `cargo test -p context-graph-core dream::poincare_walk` |
| No clippy warnings | `cargo clippy -p context-graph-core -- -D warnings` |
| Mobius add stays in ball | All results have norm < 1.0 |
| Geodesic distance symmetric | d(p,q) == d(q,p) |
| Geodesic distance to self | d(p,p) == 0 |
| Random directions unit length | norm == 1.0 for all |
| Softmax sums to 1 | sum(probs) == 1.0 |
| Temperature effect | Higher T = more uniform distribution |

### 4.3 Test Coverage Requirements

- [ ] `norm_squared_64` correct calculation
- [ ] `norm_64` correct calculation
- [ ] `inner_product_64` correct calculation
- [ ] `project_to_ball` no-op for points inside ball
- [ ] `project_to_ball` projects points outside ball
- [ ] `mobius_add` at origin returns velocity
- [ ] `mobius_add` result always inside ball
- [ ] `geodesic_distance` to self is 0
- [ ] `geodesic_distance` is symmetric
- [ ] `geodesic_distance` positive for different points
- [ ] `random_direction` produces unit vectors
- [ ] `softmax_temperature` sums to 1
- [ ] `softmax_temperature` higher temp = more uniform
- [ ] `scale_direction` smaller near boundary
- [ ] `is_far_from_all` correctly identifies blind spots

---

## 5. Implementation Notes

### 5.1 Mathematical Formulas

**Mobius Addition:**
```
p ⊕ v = ((1 + 2<p,v> + ||v||²)p + (1 - ||p||²)v) / (1 + 2<p,v> + ||p||²||v||²)
```

**Geodesic Distance:**
```
d(p, q) = acosh(1 + 2||p - q||² / ((1 - ||p||²)(1 - ||q||²)))
```

**Softmax with Temperature:**
```
P(i) = exp(score_i / T) / Σ_j exp(score_j / T)
```

### 5.2 Numerical Stability

1. **Max norm check**: Always project after Mobius addition
2. **Epsilon guards**: Use 1e-7 for division-by-zero prevention
3. **Acosh edge case**: Return 0 if argument <= 1.0
4. **Log-sum-exp trick**: Subtract max before exponentiating in softmax

### 5.3 Performance Considerations

- All 64D operations are O(64) - compiler will auto-vectorize
- No heap allocations in hot path (except `softmax_temperature` return)
- Consider SIMD intrinsics for future optimization

---

## 6. Estimated Effort Breakdown

| Phase | Duration |
|-------|----------|
| Core math functions | 45 min |
| Mobius addition | 30 min |
| Geodesic distance | 20 min |
| Random direction sampling | 30 min |
| Softmax with temperature | 15 min |
| Unit tests | 45 min |
| Documentation | 15 min |
| **Total** | **3 hours** |
