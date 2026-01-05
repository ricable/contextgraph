//! Exponential and logarithmic maps for Poincare ball.
//!
//! These maps transfer between tangent spaces and the manifold:
//! - Exp map: tangent vector at x -> point on manifold
//! - Log map: point on manifold -> tangent vector at x
//!
//! # Mathematical Formulas
//!
//! - exp_x(v) = x + (tanh(sqrt(c) * lambda_x * ||v|| / 2) * v / (sqrt(c) * ||v||))
//! - log_x(y) = (2 / (lambda_x * sqrt(c))) * arctanh(sqrt(c) * ||(-x) + y||) * ((-x) + y) / ||(-x) + y||
//!
//! where lambda_x = 2 / (1 - c||x||^2) is the conformal factor.

use super::PoincareBall;
use crate::hyperbolic::poincare::PoincarePoint;

impl PoincareBall {
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
    /// exp_x(v) = x + (tanh(sqrt(c) * lambda_x * ||v|| / 2) * v / (sqrt(c) * ||v||))
    ///
    /// where lambda_x = 2 / (1 - c||x||^2) is the conformal factor.
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

        // Conformal factor at x: lambda_x = 2 / (1 - c||x||^2)
        let lambda_x = self.conformal_factor(x);

        // tanh(sqrt(c) * lambda_x * ||v|| / 2)
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
    /// log_x(y) = (2 / (lambda_x * sqrt(c))) * arctanh(sqrt(c) * ||(-x) + y||) * ((-x) + y) / ||(-x) + y||
    ///
    /// where lambda_x = 2 / (1 - c||x||^2) is the conformal factor.
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

        // Compute (-x) + y
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

        // Conformal factor at x: lambda_x = 2 / (1 - c||x||^2)
        let lambda_x = self.conformal_factor(x);

        // arctanh(sqrt(c) * ||(-x) + y||), clamped to avoid NaN
        let arg = (sqrt_c * diff_norm).min(1.0 - self.config.eps);
        let arctanh_val = arg.atanh();

        // Scale: (2 / (lambda_x * sqrt(c))) * arctanh(...) / ||diff||
        let scale = (2.0 / (lambda_x * sqrt_c)) * arctanh_val / diff_norm;

        let mut result = [0.0; 64];
        for (r, &coord) in result.iter_mut().zip(diff.coords.iter()) {
            *r = scale * coord;
        }

        result
    }
}
