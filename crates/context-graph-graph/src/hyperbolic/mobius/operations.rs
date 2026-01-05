//! Core Mobius algebra operations for Poincare ball.
//!
//! Implements Mobius addition and distance calculations.
//!
//! # Mathematics
//!
//! - Mobius addition: x + y = ((1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y) /
//!   (1 + 2c<x,y> + c^2||x||^2||y||^2)
//! - Distance: d(x,y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||(-x) + y||)
//!
//! # Performance Targets
//!
//! - distance(): <10us per pair
//! - mobius_add(): <5us per operation

use super::PoincareBall;
use crate::hyperbolic::poincare::PoincarePoint;

impl PoincareBall {
    /// Compute the conformal factor lambda_x = 2 / (1 - c||x||^2).
    ///
    /// The conformal factor scales tangent vectors between Euclidean
    /// and hyperbolic metrics at point x.
    #[inline]
    pub(crate) fn conformal_factor(&self, x: &PoincarePoint) -> f32 {
        let c = self.config.abs_curvature();
        let x_norm_sq = x.norm_squared();
        let denom = (1.0 - c * x_norm_sq).max(self.config.eps);
        2.0 / denom
    }

    /// Mobius addition in Poincare ball.
    ///
    /// Formula: x + y = ((1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y) /
    ///                  (1 + 2c<x,y> + c^2||x||^2||y||^2)
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
    /// Target: <5us. O(64) operations with potential SIMD optimization.
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
        let xy_dot: f32 = x
            .coords
            .iter()
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
    /// Formula: d(x,y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||(-x) + y||)
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
    /// Target: <10us. O(64) with Mobius add and one atanh.
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

        // Compute (-x) + y using Mobius subtraction
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

        // d(x,y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||(-x) + y||)
        let arg = (sqrt_c * diff_norm).min(1.0 - self.config.eps);
        (2.0 / sqrt_c) * arg.atanh()
    }
}
