//! PoincarePoint implementation for 64D hyperbolic space.
//!
//! # Poincare Ball Model
//!
//! The Poincare ball model represents hyperbolic space as the interior of a
//! unit ball. Points must satisfy ||x|| < 1 (strictly inside). Points near
//! the boundary represent specific/leaf concepts; points near origin represent
//! general/root concepts.
//!
//! # Performance
//!
//! - Memory: 256 bytes per point (64 * 4 bytes, 64-byte aligned)
//! - norm_squared(): O(64) with SIMD optimization potential
//! - project(): O(64) when rescaling needed
//!
//! # Constitution Reference
//!
//! - hyperbolic.dim: 64
//! - hyperbolic.max_norm: 0.99999 (1.0 - 1e-5)
//! - perf.latency.entailment_check: <1ms

use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use crate::config::HyperbolicConfig;

/// Point in 64-dimensional Poincare ball model of hyperbolic space.
///
/// # Constraint
///
/// `||coords|| < 1.0` (strictly inside unit ball)
///
/// # Memory Layout
///
/// - Size: 256 bytes (64 × f32)
/// - Alignment: 64 bytes (cache line aligned for SIMD)
/// - repr(C): FFI-compatible for CUDA kernels (M04-T23)
///
/// # Example
///
/// ```
/// use context_graph_graph::hyperbolic::PoincarePoint;
/// use context_graph_graph::config::HyperbolicConfig;
///
/// let origin = PoincarePoint::origin();
/// assert_eq!(origin.norm(), 0.0);
/// assert!(origin.is_valid());
///
/// let config = HyperbolicConfig::default();
/// let mut point = PoincarePoint::from_coords([0.9; 64]);
/// point.project(&config);
/// // Floating point tolerance for norm after projection
/// assert!(point.norm() <= config.max_norm + 1e-6);
/// ```
#[serde_as]
#[repr(C, align(64))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoincarePoint {
    /// Coordinates in 64-dimensional Euclidean embedding space.
    /// Invariant: sum(coords[i]^2) < 1.0 for valid points.
    #[serde_as(as = "[_; 64]")]
    pub coords: [f32; 64],
}

impl Default for PoincarePoint {
    /// Creates origin point (center of Poincare ball).
    fn default() -> Self {
        Self::origin()
    }
}

impl PoincarePoint {
    /// Creates the origin point (all zeros).
    ///
    /// The origin is the center of the Poincare ball, representing the most
    /// general/root concept in hierarchical embeddings.
    ///
    /// # Returns
    ///
    /// Point with all 64 coordinates set to 0.0
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::PoincarePoint;
    ///
    /// let origin = PoincarePoint::origin();
    /// assert_eq!(origin.norm(), 0.0);
    /// assert!(origin.coords.iter().all(|&x| x == 0.0));
    /// ```
    #[inline]
    pub fn origin() -> Self {
        Self { coords: [0.0; 64] }
    }

    /// Creates a point from coordinates.
    ///
    /// # Warning
    ///
    /// Does NOT validate norm. Call `project()` after if norm may exceed 1.0.
    /// For validated construction, use `from_coords_projected()`.
    ///
    /// # Arguments
    ///
    /// * `coords` - 64-element array of f32 coordinates
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::PoincarePoint;
    ///
    /// let coords = [0.1f32; 64];
    /// let point = PoincarePoint::from_coords(coords);
    /// assert!(point.is_valid()); // norm = sqrt(64 * 0.01) = 0.8
    /// ```
    #[inline]
    pub fn from_coords(coords: [f32; 64]) -> Self {
        Self { coords }
    }

    /// Creates a validated point, projecting if necessary.
    ///
    /// Unlike `from_coords()`, this ensures the resulting point is valid
    /// by projecting to `max_norm` if the input norm is too large.
    ///
    /// # Arguments
    ///
    /// * `coords` - 64-element array of f32 coordinates
    /// * `config` - HyperbolicConfig containing max_norm
    ///
    /// # Returns
    ///
    /// Point guaranteed to have norm < max_norm
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::PoincarePoint;
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// let config = HyperbolicConfig::default();
    /// let coords = [1.0f32; 64]; // norm = 8.0, way outside ball
    /// let point = PoincarePoint::from_coords_projected(coords, &config);
    /// assert!(point.is_valid());
    /// // Floating point tolerance for norm after projection
    /// assert!(point.norm() <= config.max_norm + 1e-6);
    /// ```
    pub fn from_coords_projected(coords: [f32; 64], config: &HyperbolicConfig) -> Self {
        let mut point = Self { coords };
        point.project(config);
        point
    }

    /// Computes squared Euclidean norm of coordinates.
    ///
    /// More efficient than `norm()` when comparing magnitudes since it avoids
    /// the sqrt operation. Use this for validation checks.
    ///
    /// # Returns
    ///
    /// Sum of squared coordinates: Σ(coords[i]²)
    ///
    /// # Performance
    ///
    /// O(64) additions and multiplications. Compiler will auto-vectorize with
    /// SIMD when available.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::PoincarePoint;
    ///
    /// let point = PoincarePoint::origin();
    /// assert_eq!(point.norm_squared(), 0.0);
    ///
    /// let mut coords = [0.0f32; 64];
    /// coords[0] = 0.5;
    /// let point = PoincarePoint::from_coords(coords);
    /// assert_eq!(point.norm_squared(), 0.25);
    /// ```
    #[inline]
    pub fn norm_squared(&self) -> f32 {
        self.coords.iter().map(|&x| x * x).sum()
    }

    /// Computes Euclidean norm of coordinates.
    ///
    /// # Returns
    ///
    /// sqrt(Σ(coords[i]²))
    ///
    /// # Performance
    ///
    /// O(64) with one sqrt at the end. Use `norm_squared()` if you only need
    /// to compare magnitudes.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::PoincarePoint;
    ///
    /// let mut coords = [0.0f32; 64];
    /// coords[0] = 0.6;
    /// coords[1] = 0.8;
    /// let point = PoincarePoint::from_coords(coords);
    /// assert!((point.norm() - 1.0).abs() < 1e-6); // 0.36 + 0.64 = 1.0
    /// ```
    #[inline]
    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }

    /// Projects point to stay strictly inside the Poincare ball.
    ///
    /// If `||coords|| >= max_norm`, rescales all coordinates so that
    /// `||coords|| = max_norm`. This prevents numerical instability that
    /// occurs when points approach or exceed the unit ball boundary.
    ///
    /// # Arguments
    ///
    /// * `config` - HyperbolicConfig containing max_norm (default: 0.99999)
    ///
    /// # Algorithm
    ///
    /// 1. Compute current norm
    /// 2. If norm >= max_norm: scale = max_norm / norm
    /// 3. Multiply all coordinates by scale
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::PoincarePoint;
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// let config = HyperbolicConfig::default();
    /// let mut point = PoincarePoint::from_coords([0.2f32; 64]);
    /// // norm = sqrt(64 * 0.04) = 1.6, exceeds max_norm
    /// point.project(&config);
    /// assert!(point.norm() < 1.0);
    /// ```
    pub fn project(&mut self, config: &HyperbolicConfig) {
        let norm = self.norm();
        if norm >= config.max_norm {
            let scale = config.max_norm / norm;
            for c in &mut self.coords {
                *c *= scale;
            }
        }
    }

    /// Creates a projected copy without modifying self.
    ///
    /// # Arguments
    ///
    /// * `config` - HyperbolicConfig containing max_norm
    ///
    /// # Returns
    ///
    /// New PoincarePoint with norm <= max_norm
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::PoincarePoint;
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// let config = HyperbolicConfig::default();
    /// let original = PoincarePoint::from_coords([0.2f32; 64]);
    /// let projected = original.projected(&config);
    /// // Floating point tolerance for norm after projection
    /// assert!(projected.norm() <= config.max_norm + 1e-6);
    /// // original unchanged
    /// assert!((original.norm() - projected.norm()).abs() > 0.5);
    /// ```
    pub fn projected(&self, config: &HyperbolicConfig) -> Self {
        let mut result = self.clone();
        result.project(config);
        result
    }

    /// Checks if point is valid (strictly inside unit ball).
    ///
    /// # Returns
    ///
    /// `true` if `||coords|| < 1.0`, `false` otherwise
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::PoincarePoint;
    ///
    /// let origin = PoincarePoint::origin();
    /// assert!(origin.is_valid());
    ///
    /// let boundary = PoincarePoint::from_coords([0.125f32; 64]);
    /// // norm = sqrt(64 * 0.015625) = 1.0, ON boundary = invalid
    /// assert!(!boundary.is_valid());
    /// ```
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.norm_squared() < 1.0
    }

    /// Checks if point is valid with given config's max_norm.
    ///
    /// Stricter than `is_valid()` - checks against config's max_norm
    /// rather than 1.0.
    ///
    /// # Arguments
    ///
    /// * `config` - HyperbolicConfig containing max_norm
    ///
    /// # Returns
    ///
    /// `true` if `||coords|| < config.max_norm`
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::PoincarePoint;
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// let config = HyperbolicConfig::default(); // max_norm = 0.99999
    /// let point = PoincarePoint::from_coords([0.12f32; 64]);
    /// // norm ≈ 0.96, valid for both
    /// assert!(point.is_valid());
    /// assert!(point.is_valid_for_config(&config));
    /// ```
    #[inline]
    pub fn is_valid_for_config(&self, config: &HyperbolicConfig) -> bool {
        self.norm() < config.max_norm
    }
}

impl PartialEq for PoincarePoint {
    /// Compares two points for exact coordinate equality.
    ///
    /// # Warning
    ///
    /// Due to floating-point precision, use with caution. For approximate
    /// equality, compare `(a.coords[i] - b.coords[i]).abs() < epsilon`.
    fn eq(&self, other: &Self) -> bool {
        self.coords == other.coords
    }
}

// ============================================================================
// TESTS - MUST USE REAL DATA, NO MOCKS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========== CONSTRUCTION TESTS ==========

    #[test]
    fn test_origin_is_zero_vector() {
        let origin = PoincarePoint::origin();
        for &c in &origin.coords {
            assert_eq!(c, 0.0, "Origin must have all zero coordinates");
        }
        assert_eq!(origin.coords.len(), 64, "Must have exactly 64 dimensions");
    }

    #[test]
    fn test_origin_has_zero_norm() {
        let origin = PoincarePoint::origin();
        assert_eq!(origin.norm(), 0.0);
        assert_eq!(origin.norm_squared(), 0.0);
    }

    #[test]
    fn test_default_is_origin() {
        let default = PoincarePoint::default();
        let origin = PoincarePoint::origin();
        assert_eq!(default, origin);
    }

    #[test]
    fn test_from_coords_preserves_values() {
        let mut coords = [0.0f32; 64];
        coords[0] = 0.1;
        coords[63] = 0.2;
        let point = PoincarePoint::from_coords(coords);
        assert_eq!(point.coords[0], 0.1);
        assert_eq!(point.coords[63], 0.2);
        assert_eq!(point.coords[32], 0.0);
    }

    #[test]
    fn test_from_coords_projected_ensures_validity() {
        let config = HyperbolicConfig::default();
        let coords = [1.0f32; 64]; // norm = 8.0
        let point = PoincarePoint::from_coords_projected(coords, &config);
        assert!(point.is_valid(), "Projected point must be valid");
        // After projection, norm is at most max_norm (may be exactly equal due to floating point)
        assert!(point.norm() <= config.max_norm + 1e-6);
    }

    // ========== NORM TESTS ==========

    #[test]
    fn test_norm_squared_single_nonzero() {
        let mut coords = [0.0f32; 64];
        coords[0] = 0.5;
        let point = PoincarePoint::from_coords(coords);
        assert_eq!(point.norm_squared(), 0.25);
    }

    #[test]
    fn test_norm_squared_multiple_nonzero() {
        let mut coords = [0.0f32; 64];
        coords[0] = 0.3;
        coords[1] = 0.4;
        let point = PoincarePoint::from_coords(coords);
        // 0.09 + 0.16 = 0.25
        assert!((point.norm_squared() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_norm_pythagorean() {
        let mut coords = [0.0f32; 64];
        coords[0] = 0.6;
        coords[1] = 0.8;
        let point = PoincarePoint::from_coords(coords);
        // sqrt(0.36 + 0.64) = sqrt(1.0) = 1.0
        assert!((point.norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_norm_uniform_coords() {
        let coords = [0.1f32; 64];
        let point = PoincarePoint::from_coords(coords);
        // sqrt(64 * 0.01) = sqrt(0.64) = 0.8
        assert!((point.norm() - 0.8).abs() < 1e-6);
    }

    // ========== PROJECTION TESTS ==========

    #[test]
    fn test_project_inside_ball_unchanged() {
        let config = HyperbolicConfig::default();
        let coords = [0.05f32; 64]; // norm = sqrt(64*0.0025) = 0.4
        let mut point = PoincarePoint::from_coords(coords);
        let original_norm = point.norm();
        point.project(&config);
        assert!((point.norm() - original_norm).abs() < 1e-6);
    }

    #[test]
    fn test_project_outside_ball_rescaled() {
        let config = HyperbolicConfig::default();
        let coords = [0.2f32; 64]; // norm = sqrt(64*0.04) = 1.6
        let mut point = PoincarePoint::from_coords(coords);
        assert!(!point.is_valid()); // outside ball
        point.project(&config);
        assert!(point.is_valid());
        assert!((point.norm() - config.max_norm).abs() < 1e-6);
    }

    #[test]
    fn test_project_at_boundary() {
        let config = HyperbolicConfig::default();
        // Create point exactly at boundary (norm = 1.0)
        let mut coords = [0.0f32; 64];
        coords[0] = 1.0;
        let mut point = PoincarePoint::from_coords(coords);
        assert!(!point.is_valid()); // AT boundary = invalid
        point.project(&config);
        assert!(point.is_valid());
    }

    #[test]
    fn test_projected_returns_new_point() {
        let config = HyperbolicConfig::default();
        let coords = [0.2f32; 64];
        let original = PoincarePoint::from_coords(coords);
        let projected = original.projected(&config);
        // Original unchanged
        assert!((original.norm() - 1.6).abs() < 0.1);
        // Projected is valid
        assert!(projected.is_valid());
    }

    #[test]
    fn test_project_preserves_direction() {
        let config = HyperbolicConfig::default();
        let mut coords = [0.0f32; 64];
        coords[0] = 2.0;
        coords[1] = 1.0;
        let mut point = PoincarePoint::from_coords(coords);
        point.project(&config);
        // Ratio should be preserved
        let ratio = point.coords[0] / point.coords[1];
        assert!((ratio - 2.0).abs() < 1e-5);
    }

    // ========== VALIDITY TESTS ==========

    #[test]
    fn test_is_valid_origin() {
        let origin = PoincarePoint::origin();
        assert!(origin.is_valid());
    }

    #[test]
    fn test_is_valid_inside_ball() {
        let coords = [0.05f32; 64]; // norm ≈ 0.4
        let point = PoincarePoint::from_coords(coords);
        assert!(point.is_valid());
    }

    #[test]
    fn test_is_valid_at_boundary_false() {
        let mut coords = [0.0f32; 64];
        coords[0] = 1.0; // norm = 1.0 exactly
        let point = PoincarePoint::from_coords(coords);
        assert!(!point.is_valid(), "Point AT boundary is invalid");
    }

    #[test]
    fn test_is_valid_outside_ball_false() {
        let coords = [0.2f32; 64]; // norm ≈ 1.6
        let point = PoincarePoint::from_coords(coords);
        assert!(!point.is_valid());
    }

    #[test]
    fn test_is_valid_for_config() {
        let config = HyperbolicConfig::default(); // max_norm = 0.99999
        let coords = [0.12f32; 64]; // norm ≈ 0.96
        let point = PoincarePoint::from_coords(coords);
        assert!(point.is_valid_for_config(&config));
    }

    // ========== MEMORY LAYOUT TESTS ==========

    #[test]
    fn test_size_is_256_bytes() {
        assert_eq!(
            std::mem::size_of::<PoincarePoint>(),
            256,
            "PoincarePoint must be 256 bytes (64 × f32)"
        );
    }

    #[test]
    fn test_alignment_is_64_bytes() {
        assert_eq!(
            std::mem::align_of::<PoincarePoint>(),
            64,
            "PoincarePoint must be 64-byte aligned for SIMD"
        );
    }

    // ========== EQUALITY TESTS ==========

    #[test]
    fn test_equality_same_coords() {
        let coords = [0.1f32; 64];
        let a = PoincarePoint::from_coords(coords);
        let b = PoincarePoint::from_coords(coords);
        assert_eq!(a, b);
    }

    #[test]
    fn test_inequality_different_coords() {
        let a = PoincarePoint::from_coords([0.1f32; 64]);
        let mut coords = [0.1f32; 64];
        coords[0] = 0.2;
        let b = PoincarePoint::from_coords(coords);
        assert_ne!(a, b);
    }

    // ========== CLONE TESTS ==========

    #[test]
    fn test_clone_independent() {
        let coords = [0.1f32; 64];
        let original = PoincarePoint::from_coords(coords);
        let mut cloned = original.clone();
        cloned.coords[0] = 0.9;
        assert_eq!(original.coords[0], 0.1, "Clone must be independent");
    }

    // ========== EDGE CASES ==========

    #[test]
    fn test_edge_case_very_small_norm() {
        let coords = [1e-10f32; 64];
        let point = PoincarePoint::from_coords(coords);
        assert!(point.is_valid());
        assert!(point.norm() > 0.0);
    }

    #[test]
    fn test_edge_case_near_max_norm() {
        let config = HyperbolicConfig::default();
        // norm ≈ 0.999 (just under max_norm)
        let scale = 0.999 / (64.0_f32).sqrt();
        let coords = [scale; 64];
        let point = PoincarePoint::from_coords(coords);
        assert!(point.is_valid_for_config(&config));
    }

    #[test]
    fn test_edge_case_negative_coords() {
        let mut coords = [0.0f32; 64];
        coords[0] = -0.5;
        coords[1] = 0.5;
        let point = PoincarePoint::from_coords(coords);
        // norm = sqrt(0.25 + 0.25) = sqrt(0.5) ≈ 0.707
        assert!(point.is_valid());
        assert!((point.norm() - (0.5_f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_edge_case_project_zero_vector() {
        let config = HyperbolicConfig::default();
        let mut origin = PoincarePoint::origin();
        origin.project(&config); // Should not panic
        assert_eq!(origin.norm(), 0.0);
    }

    #[test]
    fn test_edge_case_nan_detection() {
        let mut coords = [0.0f32; 64];
        coords[0] = f32::NAN;
        let point = PoincarePoint::from_coords(coords);
        // norm_squared and norm will be NaN
        assert!(point.norm().is_nan());
        // is_valid should return false for NaN
        assert!(!point.is_valid()); // NaN < 1.0 is false
    }

    #[test]
    fn test_edge_case_infinity() {
        let mut coords = [0.0f32; 64];
        coords[0] = f32::INFINITY;
        let point = PoincarePoint::from_coords(coords);
        assert!(!point.is_valid());
    }
}
