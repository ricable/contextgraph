//! PoincareBall implementation with Mobius algebra operations.
//!
//! # Poincare Ball Model
//!
//! The Poincare ball model represents hyperbolic space as the open unit ball.
//! Mobius operations provide the algebra for vector addition, distances, and
//! exponential/logarithmic maps between tangent spaces and the manifold.
//!
//! # Mathematics
//!
//! - Mobius addition: x ⊕ y = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y) /
//!   (1 + 2c<x,y> + c²||x||²||y||²)
//! - Distance: d(x,y) = (2/√c) * arctanh(√c * ||(-x) ⊕ y||)
//! - Exp map: Maps tangent vector at x to point on manifold
//! - Log map: Maps point y to tangent vector at x (inverse of exp_map)
//!
//! # Performance Targets
//!
//! - distance(): <10μs per pair
//! - mobius_add(): <5μs per operation
//!
//! # Constitution Reference
//!
//! - perf.latency.entailment_check: <1ms (this contributes ~1% budget)
//! - contextprd.md Section 4.4: Poincare Ball d(x,y) formula

use crate::config::HyperbolicConfig;
use crate::hyperbolic::poincare::PoincarePoint;

/// Poincare ball model with Mobius algebra operations.
///
/// Provides hyperbolic geometry operations for the knowledge graph's
/// hierarchical embeddings. Points near origin represent general concepts;
/// points near boundary represent specific concepts.
///
/// # Example
///
/// ```
/// use context_graph_graph::hyperbolic::{PoincareBall, PoincarePoint};
/// use context_graph_graph::config::HyperbolicConfig;
///
/// let config = HyperbolicConfig::default();
/// let ball = PoincareBall::new(config);
///
/// let origin = PoincarePoint::origin();
/// let mut coords = [0.0f32; 64];
/// coords[0] = 0.5;
/// let point = PoincarePoint::from_coords(coords);
///
/// // Distance from origin
/// let d = ball.distance(&origin, &point);
/// assert!(d > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct PoincareBall {
    config: HyperbolicConfig,
}

impl PoincareBall {
    /// Create a new Poincare ball with given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - HyperbolicConfig with curvature, eps, max_norm settings
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::PoincareBall;
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// let config = HyperbolicConfig::default();
    /// let ball = PoincareBall::new(config);
    /// assert_eq!(ball.config().curvature, -1.0);
    /// ```
    #[inline]
    pub fn new(config: HyperbolicConfig) -> Self {
        Self { config }
    }

    /// Get reference to configuration.
    #[inline]
    pub fn config(&self) -> &HyperbolicConfig {
        &self.config
    }

    /// Compute the conformal factor λ_x = 2 / (1 - c||x||²).
    ///
    /// The conformal factor scales tangent vectors between Euclidean
    /// and hyperbolic metrics at point x.
    #[inline]
    fn conformal_factor(&self, x: &PoincarePoint) -> f32 {
        let c = self.config.abs_curvature();
        let x_norm_sq = x.norm_squared();
        let denom = (1.0 - c * x_norm_sq).max(self.config.eps);
        2.0 / denom
    }

    /// Mobius addition in Poincare ball.
    ///
    /// Formula: x ⊕ y = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y) /
    ///                  (1 + 2c<x,y> + c²||x||²||y||²)
    ///
    /// where c = |curvature|
    ///
    /// # Arguments
    ///
    /// * `x` - First point in Poincare ball
    /// * `y` - Second point in Poincare ball
    ///
    /// # Returns
    ///
    /// Result point, projected to stay inside ball if needed.
    ///
    /// # Performance
    ///
    /// Target: <5μs. O(64) operations with potential SIMD optimization.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::{PoincareBall, PoincarePoint};
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// let ball = PoincareBall::new(HyperbolicConfig::default());
    /// let origin = PoincarePoint::origin();
    /// let mut coords = [0.0f32; 64];
    /// coords[0] = 0.3;
    /// let point = PoincarePoint::from_coords(coords);
    ///
    /// // Adding origin returns the other point
    /// let result = ball.mobius_add(&origin, &point);
    /// assert!((result.coords[0] - 0.3).abs() < 1e-6);
    /// ```
    pub fn mobius_add(&self, x: &PoincarePoint, y: &PoincarePoint) -> PoincarePoint {
        let c = self.config.abs_curvature();
        let x_norm_sq = x.norm_squared();
        let y_norm_sq = y.norm_squared();

        // Inner product <x, y>
        let xy_dot: f32 = x.coords.iter()
            .zip(y.coords.iter())
            .map(|(a, b)| a * b)
            .sum();

        let num_coeff_x = 1.0 + 2.0 * c * xy_dot + c * y_norm_sq;
        let num_coeff_y = 1.0 - c * x_norm_sq;
        let denom = 1.0 + 2.0 * c * xy_dot + c * c * x_norm_sq * y_norm_sq;

        // Avoid division by zero
        let safe_denom = if denom.abs() < self.config.eps {
            self.config.eps
        } else {
            denom
        };

        let mut result = PoincarePoint::origin();
        for i in 0..64 {
            result.coords[i] = (num_coeff_x * x.coords[i] + num_coeff_y * y.coords[i]) / safe_denom;
        }

        // Project to ensure result stays inside ball
        result.project(&self.config);
        result
    }

    /// Compute Poincare ball distance between two points.
    ///
    /// Formula: d(x,y) = (2/√c) * arctanh(√c * ||(-x) ⊕ y||)
    ///
    /// Uses Mobius subtraction for accurate hyperbolic distance.
    ///
    /// # Arguments
    ///
    /// * `x` - First point
    /// * `y` - Second point
    ///
    /// # Returns
    ///
    /// Hyperbolic distance (always >= 0), or NaN if input contains NaN.
    ///
    /// # Performance
    ///
    /// Target: <10μs. O(64) with Mobius add and one atanh.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::{PoincareBall, PoincarePoint};
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// let ball = PoincareBall::new(HyperbolicConfig::default());
    ///
    /// // Distance from point to itself is 0
    /// let point = PoincarePoint::origin();
    /// assert_eq!(ball.distance(&point, &point), 0.0);
    ///
    /// // Distance is symmetric
    /// let mut coords = [0.0f32; 64];
    /// coords[0] = 0.5;
    /// let p1 = PoincarePoint::origin();
    /// let p2 = PoincarePoint::from_coords(coords);
    /// let d1 = ball.distance(&p1, &p2);
    /// let d2 = ball.distance(&p2, &p1);
    /// assert!((d1 - d2).abs() < 1e-6);
    /// ```
    pub fn distance(&self, x: &PoincarePoint, y: &PoincarePoint) -> f32 {
        // Propagate NaN inputs
        if x.coords.iter().any(|c| c.is_nan()) || y.coords.iter().any(|c| c.is_nan()) {
            return f32::NAN;
        }

        let c = self.config.abs_curvature();
        let sqrt_c = c.sqrt();

        // Compute (-x) ⊕ y using Mobius subtraction
        let mut neg_x = x.clone();
        for coord in &mut neg_x.coords {
            *coord = -*coord;
        }
        let diff = self.mobius_add(&neg_x, y);
        let diff_norm = diff.norm();

        // Handle identical points
        if diff_norm < self.config.eps {
            return 0.0;
        }

        // d(x,y) = (2/√c) * arctanh(√c * ||(-x) ⊕ y||)
        let arg = (sqrt_c * diff_norm).min(1.0 - self.config.eps);
        (2.0 / sqrt_c) * arg.atanh()
    }

    /// Exponential map: tangent vector at x -> point on manifold.
    ///
    /// Maps a vector v in the tangent space at x to a point on the Poincare ball
    /// along the geodesic starting at x with initial direction v.
    ///
    /// # Arguments
    ///
    /// * `x` - Base point on manifold
    /// * `v` - Tangent vector at x (64D array)
    ///
    /// # Returns
    ///
    /// Point on Poincare ball.
    ///
    /// # Mathematical Formula
    ///
    /// exp_x(v) = x ⊕ (tanh(√c * λ_x * ||v|| / 2) * v / (√c * ||v||))
    ///
    /// where λ_x = 2 / (1 - c||x||²) is the conformal factor.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::{PoincareBall, PoincarePoint};
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// let ball = PoincareBall::new(HyperbolicConfig::default());
    /// let origin = PoincarePoint::origin();
    ///
    /// // Zero tangent vector returns base point
    /// let zero_v = [0.0f32; 64];
    /// let result = ball.exp_map(&origin, &zero_v);
    /// assert_eq!(result.coords[0], 0.0);
    /// ```
    pub fn exp_map(&self, x: &PoincarePoint, v: &[f32; 64]) -> PoincarePoint {
        let c = self.config.abs_curvature();
        let sqrt_c = c.sqrt();

        // ||v|| in tangent space
        let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Handle zero tangent vector
        if v_norm < self.config.eps {
            return x.clone();
        }

        // Conformal factor at x: λ_x = 2 / (1 - c||x||²)
        let lambda_x = self.conformal_factor(x);

        // tanh(√c * λ_x * ||v|| / 2)
        // Clamp tanh_factor to stay strictly inside ball (tanh saturates to 1.0 in f32)
        let tanh_arg = sqrt_c * lambda_x * v_norm / 2.0;
        let tanh_factor = tanh_arg.tanh().min(self.config.max_norm);

        // Create direction point: (tanh_factor / (sqrt_c * v_norm)) * v
        let mut direction = PoincarePoint::origin();
        let scale = tanh_factor / (sqrt_c * v_norm);
        for (coord, &v_i) in direction.coords.iter_mut().zip(v.iter()) {
            *coord = scale * v_i;
        }

        // Ensure direction is strictly inside the ball before Mobius add
        let dir_norm = direction.norm();
        if dir_norm >= self.config.max_norm {
            // Scale to be strictly inside: max_norm - eps
            let target_norm = self.config.max_norm - self.config.eps;
            let scale_factor = target_norm / dir_norm;
            for i in 0..64 {
                direction.coords[i] *= scale_factor;
            }
        }

        // Mobius add with x
        let mut result = self.mobius_add(x, &direction);

        // Final check: ensure result is strictly inside ball
        let result_norm = result.norm();
        if result_norm >= self.config.max_norm {
            let target_norm = self.config.max_norm - self.config.eps;
            let scale_factor = target_norm / result_norm;
            for i in 0..64 {
                result.coords[i] *= scale_factor;
            }
        }

        result
    }

    /// Logarithmic map: point on manifold -> tangent vector at x.
    ///
    /// Returns the tangent vector at x that points toward y.
    /// This is the inverse of exp_map.
    ///
    /// # Arguments
    ///
    /// * `x` - Base point
    /// * `y` - Target point
    ///
    /// # Returns
    ///
    /// Tangent vector at x pointing toward y.
    ///
    /// # Mathematical Formula
    ///
    /// log_x(y) = (2 / (λ_x * √c)) * arctanh(√c * ||(-x) ⊕ y||) * ((-x) ⊕ y) / ||(-x) ⊕ y||
    ///
    /// where λ_x = 2 / (1 - c||x||²) is the conformal factor.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::{PoincareBall, PoincarePoint};
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// let ball = PoincareBall::new(HyperbolicConfig::default());
    /// let origin = PoincarePoint::origin();
    ///
    /// // Log map from point to itself returns zero vector
    /// let v = ball.log_map(&origin, &origin);
    /// assert!(v.iter().all(|&x| x.abs() < 1e-6));
    /// ```
    pub fn log_map(&self, x: &PoincarePoint, y: &PoincarePoint) -> [f32; 64] {
        let c = self.config.abs_curvature();
        let sqrt_c = c.sqrt();

        // Compute (-x) ⊕ y
        let mut neg_x = x.clone();
        for coord in &mut neg_x.coords {
            *coord = -*coord;
        }
        let diff = self.mobius_add(&neg_x, y);

        let diff_norm = diff.norm();

        // Handle identical points
        if diff_norm < self.config.eps {
            return [0.0; 64];
        }

        // Conformal factor at x: λ_x = 2 / (1 - c||x||²)
        let lambda_x = self.conformal_factor(x);

        // arctanh(√c * ||(-x) ⊕ y||), clamped to avoid NaN
        let arg = (sqrt_c * diff_norm).min(1.0 - self.config.eps);
        let arctanh_val = arg.atanh();

        // Scale: (2 / (λ_x * √c)) * arctanh(...) / ||diff||
        let scale = (2.0 / (lambda_x * sqrt_c)) * arctanh_val / diff_norm;

        let mut result = [0.0; 64];
        for (r, &coord) in result.iter_mut().zip(diff.coords.iter()) {
            *r = scale * coord;
        }

        result
    }
}

// ============================================================================
// TESTS - REAL DATA ONLY, NO MOCKS (per constitution REQ-KG-TEST)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_ball() -> PoincareBall {
        PoincareBall::new(HyperbolicConfig::default())
    }

    fn make_point(first_coord: f32) -> PoincarePoint {
        let mut coords = [0.0f32; 64];
        coords[0] = first_coord;
        PoincarePoint::from_coords(coords)
    }

    fn make_point_2d(x: f32, y: f32) -> PoincarePoint {
        let mut coords = [0.0f32; 64];
        coords[0] = x;
        coords[1] = y;
        PoincarePoint::from_coords(coords)
    }

    // ========== CONSTRUCTION TESTS ==========

    #[test]
    fn test_new_creates_ball_with_config() {
        let config = HyperbolicConfig::with_curvature(-0.5);
        let ball = PoincareBall::new(config.clone());
        assert_eq!(ball.config().curvature, -0.5);
    }

    #[test]
    fn test_config_accessor() {
        let ball = default_ball();
        assert_eq!(ball.config().dim, 64);
        assert_eq!(ball.config().curvature, -1.0);
    }

    // ========== MOBIUS ADDITION TESTS ==========

    #[test]
    fn test_mobius_add_with_origin_returns_other() {
        let ball = default_ball();
        let origin = PoincarePoint::origin();
        let point = make_point(0.3);

        // x ⊕ 0 = x
        let result = ball.mobius_add(&point, &origin);
        assert!((result.coords[0] - 0.3).abs() < 1e-5);

        // 0 ⊕ y = y
        let result2 = ball.mobius_add(&origin, &point);
        assert!((result2.coords[0] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_mobius_add_result_inside_ball() {
        let ball = default_ball();
        let p1 = make_point(0.5);
        let p2 = make_point(0.3);

        let result = ball.mobius_add(&p1, &p2);
        assert!(result.is_valid(), "Mobius add result must be inside ball");
    }

    #[test]
    fn test_mobius_add_near_boundary() {
        let ball = default_ball();
        // Points close to boundary
        let p1 = make_point(0.9);
        let p2 = make_point(0.8);

        let result = ball.mobius_add(&p1, &p2);
        assert!(result.is_valid_for_config(ball.config()));
    }

    #[test]
    fn test_mobius_add_opposite_directions() {
        let ball = default_ball();
        let p1 = make_point(0.3);
        let p2 = make_point(-0.3);

        // Opposite points should partially cancel
        let result = ball.mobius_add(&p1, &p2);
        assert!(result.norm() < 0.3);
    }

    // ========== DISTANCE TESTS ==========

    #[test]
    fn test_distance_same_point_is_zero() {
        let ball = default_ball();
        let point = make_point(0.5);
        assert_eq!(ball.distance(&point, &point), 0.0);
    }

    #[test]
    fn test_distance_origin_to_origin_is_zero() {
        let ball = default_ball();
        let origin = PoincarePoint::origin();
        assert_eq!(ball.distance(&origin, &origin), 0.0);
    }

    #[test]
    fn test_distance_is_symmetric() {
        let ball = default_ball();
        let p1 = make_point(0.3);
        let p2 = make_point(0.6);

        let d1 = ball.distance(&p1, &p2);
        let d2 = ball.distance(&p2, &p1);
        assert!((d1 - d2).abs() < 1e-6, "Distance must be symmetric");
    }

    #[test]
    fn test_distance_is_nonnegative() {
        let ball = default_ball();
        let p1 = make_point(0.3);
        let p2 = make_point(-0.5);
        assert!(ball.distance(&p1, &p2) >= 0.0);
    }

    #[test]
    fn test_distance_triangle_inequality() {
        let ball = default_ball();
        let p1 = make_point(0.1);
        let p2 = make_point(0.3);
        let p3 = make_point(0.5);

        let d12 = ball.distance(&p1, &p2);
        let d23 = ball.distance(&p2, &p3);
        let d13 = ball.distance(&p1, &p3);

        // d(p1, p3) <= d(p1, p2) + d(p2, p3)
        assert!(d13 <= d12 + d23 + 1e-6, "Triangle inequality violated");
    }

    #[test]
    fn test_distance_from_origin_monotonic() {
        let ball = default_ball();
        let origin = PoincarePoint::origin();

        // Distance increases as we move further from origin
        let p1 = make_point(0.1);
        let p2 = make_point(0.5);
        let p3 = make_point(0.9);

        let d1 = ball.distance(&origin, &p1);
        let d2 = ball.distance(&origin, &p2);
        let d3 = ball.distance(&origin, &p3);

        assert!(d1 < d2, "d(0, 0.1) < d(0, 0.5)");
        assert!(d2 < d3, "d(0, 0.5) < d(0, 0.9)");
    }

    #[test]
    fn test_distance_near_boundary_larger() {
        let ball = default_ball();
        // In hyperbolic space, distances near boundary are larger
        let origin = PoincarePoint::origin();
        let near_boundary = make_point(0.99);

        let d = ball.distance(&origin, &near_boundary);
        // For c=-1, d(0, r) = 2 * arctanh(r), so d(0, 0.99) ≈ 5.3
        assert!(d > 4.0, "Distance near boundary should be large: {}", d);
    }

    // ========== EXP MAP TESTS ==========

    #[test]
    fn test_exp_map_zero_tangent_returns_base() {
        let ball = default_ball();
        let base = make_point(0.3);
        let zero_v = [0.0f32; 64];

        let result = ball.exp_map(&base, &zero_v);
        assert!((result.coords[0] - base.coords[0]).abs() < 1e-6);
    }

    #[test]
    fn test_exp_map_from_origin() {
        let ball = default_ball();
        let origin = PoincarePoint::origin();
        let mut v = [0.0f32; 64];
        v[0] = 0.5;

        let result = ball.exp_map(&origin, &v);
        assert!(result.is_valid());
        assert!(result.coords[0] > 0.0, "Should move in direction of v");
    }

    #[test]
    fn test_exp_map_result_inside_ball() {
        let ball = default_ball();
        let base = make_point(0.5);
        let mut v = [0.0f32; 64];
        v[0] = 10.0; // Large tangent vector

        let result = ball.exp_map(&base, &v);
        assert!(result.is_valid_for_config(ball.config()), "exp_map result must be inside ball");
    }

    // ========== LOG MAP TESTS ==========

    #[test]
    fn test_log_map_same_point_is_zero() {
        let ball = default_ball();
        let point = make_point(0.3);

        let v = ball.log_map(&point, &point);
        let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(v_norm < 1e-6, "log_map(x, x) should be zero vector");
    }

    #[test]
    fn test_log_map_from_origin() {
        let ball = default_ball();
        let origin = PoincarePoint::origin();
        let target = make_point(0.3);

        let v = ball.log_map(&origin, &target);
        // Should point in positive x direction
        assert!(v[0] > 0.0);
    }

    // ========== ROUND-TRIP TESTS (CRITICAL) ==========

    #[test]
    fn test_exp_log_roundtrip_from_origin() {
        let ball = default_ball();
        let origin = PoincarePoint::origin();
        let mut v_orig = [0.0f32; 64];
        v_orig[0] = 0.5;
        v_orig[1] = 0.3;

        // exp_map -> log_map should recover original tangent vector
        let point = ball.exp_map(&origin, &v_orig);
        let v_recovered = ball.log_map(&origin, &point);

        for i in 0..64 {
            assert!(
                (v_orig[i] - v_recovered[i]).abs() < 1e-4,
                "Roundtrip failed at index {}: {} vs {}",
                i, v_orig[i], v_recovered[i]
            );
        }
    }

    #[test]
    fn test_log_exp_roundtrip() {
        let ball = default_ball();
        let base = make_point(0.2);
        let target = make_point(0.5);

        // log_map -> exp_map should approximately recover target
        let v = ball.log_map(&base, &target);
        let recovered = ball.exp_map(&base, &v);

        for i in 0..64 {
            assert!(
                (target.coords[i] - recovered.coords[i]).abs() < 1e-4,
                "Roundtrip failed at index {}: {} vs {}",
                i, target.coords[i], recovered.coords[i]
            );
        }
    }

    // ========== EDGE CASES ==========

    #[test]
    fn test_distance_with_nan_coords_returns_nan() {
        let ball = default_ball();
        let mut coords = [0.0f32; 64];
        coords[0] = f32::NAN;
        let p1 = PoincarePoint::from_coords(coords);
        let p2 = PoincarePoint::origin();

        let d = ball.distance(&p1, &p2);
        assert!(d.is_nan(), "Distance with NaN input should be NaN");
    }

    #[test]
    fn test_mobius_add_handles_small_denominator() {
        let ball = default_ball();
        // Create points that might cause small denominator
        let mut coords1 = [0.0f32; 64];
        let mut coords2 = [0.0f32; 64];
        coords1[0] = 0.99;
        coords2[0] = -0.99;

        let p1 = PoincarePoint::from_coords(coords1);
        let p2 = PoincarePoint::from_coords(coords2);

        let result = ball.mobius_add(&p1, &p2);
        // Should not panic or produce NaN
        assert!(!result.coords[0].is_nan(), "Should handle small denominator");
        assert!(result.is_valid());
    }

    #[test]
    fn test_custom_curvature() {
        let config = HyperbolicConfig::with_curvature(-0.5);
        let ball = PoincareBall::new(config);

        let p1 = PoincarePoint::origin();
        let p2 = make_point(0.5);

        let d = ball.distance(&p1, &p2);
        // With lower curvature magnitude, distances should be different
        assert!(d > 0.0);
        assert!(!d.is_nan());
    }

    // ========== MATHEMATICAL PROPERTY TESTS ==========

    #[test]
    fn test_distance_formula_verification() {
        // Verify against known formula: d(0, r) = 2 * arctanh(r) for c=-1
        let ball = default_ball();
        let origin = PoincarePoint::origin();
        let r = 0.5;
        let point = make_point(r);

        let computed = ball.distance(&origin, &point);
        let expected = 2.0 * r.atanh();

        assert!(
            (computed - expected).abs() < 1e-5,
            "Distance formula mismatch: computed={}, expected={}",
            computed, expected
        );
    }

    // ========== PERFORMANCE SANITY TESTS ==========

    #[test]
    fn test_distance_many_calls() {
        let ball = default_ball();
        let p1 = make_point(0.3);
        let p2 = make_point(0.6);

        // Run many iterations to check for consistency
        let first_distance = ball.distance(&p1, &p2);
        for _ in 0..1000 {
            let d = ball.distance(&p1, &p2);
            assert!((d - first_distance).abs() < 1e-10, "Distance should be deterministic");
        }
    }

    // ========== 2D POINT TESTS ==========

    #[test]
    fn test_mobius_add_2d_points() {
        let ball = default_ball();
        let p1 = make_point_2d(0.2, 0.1);
        let p2 = make_point_2d(0.1, 0.2);

        let result = ball.mobius_add(&p1, &p2);
        assert!(result.is_valid());
        // Both coordinates should be non-zero
        assert!(result.coords[0].abs() > 0.01);
        assert!(result.coords[1].abs() > 0.01);
    }

    #[test]
    fn test_distance_2d_points() {
        let ball = default_ball();
        let p1 = make_point_2d(0.3, 0.4);
        let p2 = make_point_2d(-0.3, 0.4);

        let d = ball.distance(&p1, &p2);
        assert!(d > 0.0);
        assert!(!d.is_nan());
    }

    // ========== ADDITIONAL EDGE CASE TESTS ==========

    #[test]
    fn test_exp_map_large_tangent_vector() {
        let ball = default_ball();
        let base = PoincarePoint::origin();
        let mut v = [0.0f32; 64];
        v[0] = 100.0; // Very large

        let result = ball.exp_map(&base, &v);
        assert!(result.is_valid_for_config(ball.config()));
        // Should approach boundary but stay inside
        assert!(result.norm() > 0.9);
    }

    #[test]
    fn test_log_map_points_near_boundary() {
        let ball = default_ball();
        let p1 = make_point(0.1);
        let p2 = make_point(0.95);

        let v = ball.log_map(&p1, &p2);
        let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(v_norm > 0.0, "Log map should produce non-zero vector");
        assert!(!v_norm.is_nan());
    }
}
