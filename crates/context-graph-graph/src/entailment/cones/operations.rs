//! Core operations for entailment cones.
//!
//! This module contains containment checks, membership scoring, and aperture operations.

use crate::hyperbolic::mobius::PoincareBall;
use crate::hyperbolic::poincare::PoincarePoint;

use super::EntailmentCone;

impl EntailmentCone {
    /// Get the effective aperture after applying adjustment factor.
    ///
    /// Result is clamped to valid range (0, π/2].
    ///
    /// # Formula
    ///
    /// `effective = (aperture * aperture_factor).clamp(ε, π/2)`
    /// where ε is a small positive value to prevent zero aperture.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::entailment::cones::EntailmentCone;
    ///
    /// let mut cone = EntailmentCone::default();
    /// cone.aperture = 0.5;
    /// cone.aperture_factor = 1.5;
    /// assert!((cone.effective_aperture() - 0.75).abs() < 1e-6);
    /// ```
    #[inline]
    pub fn effective_aperture(&self) -> f32 {
        const MIN_APERTURE: f32 = 1e-6;
        let effective = self.aperture * self.aperture_factor;
        effective.clamp(MIN_APERTURE, std::f32::consts::FRAC_PI_2)
    }

    /// Check if a point is contained within this cone.
    ///
    /// # Algorithm
    ///
    /// 1. Compute angle between point direction and cone axis (toward origin)
    /// 2. Return angle <= effective_aperture()
    ///
    /// # Performance Target
    ///
    /// <50μs on CPU
    ///
    /// # Arguments
    ///
    /// * `point` - Point to check for containment
    /// * `ball` - PoincareBall for hyperbolic operations
    ///
    /// # Edge Cases
    ///
    /// - Point at apex: return true (angle = 0)
    /// - Apex at origin: return true (degenerate cone contains all)
    /// - Zero-length tangent vectors: return true (numerical edge case)
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::{PoincareBall, PoincarePoint};
    /// use context_graph_graph::config::{HyperbolicConfig, ConeConfig};
    /// use context_graph_graph::entailment::cones::EntailmentCone;
    ///
    /// let config = ConeConfig::default();
    /// let apex = PoincarePoint::origin();
    /// let cone = EntailmentCone::new(apex.clone(), 0, &config).unwrap();
    /// let ball = PoincareBall::new(HyperbolicConfig::default());
    ///
    /// // Point at apex is always contained
    /// assert!(cone.contains(&apex, &ball));
    /// ```
    pub fn contains(&self, point: &PoincarePoint, ball: &PoincareBall) -> bool {
        let angle = self.compute_angle(point, ball);
        angle <= self.effective_aperture()
    }

    /// Compute angle between point direction and cone axis.
    ///
    /// # Algorithm
    ///
    /// 1. tangent = log_map(apex, point) - direction to point in tangent space
    /// 2. to_origin = log_map(apex, origin) - cone axis direction
    /// 3. cos_angle = dot(tangent, to_origin) / (||tangent|| * ||to_origin||)
    /// 4. angle = acos(cos_angle.clamp(-1.0, 1.0))
    ///
    /// # Edge Cases Return 0.0:
    ///
    /// - Point at apex (distance < eps)
    /// - Apex at origin (norm < eps)
    /// - Zero-length tangent or to_origin vectors
    ///
    /// # Performance
    ///
    /// Contributes ~40μs to total <50μs budget.
    pub(crate) fn compute_angle(&self, point: &PoincarePoint, ball: &PoincareBall) -> f32 {
        let config = ball.config();

        // Edge case: point at apex - angle is 0
        let apex_to_point_dist = ball.distance(&self.apex, point);
        if apex_to_point_dist < config.eps {
            return 0.0;
        }

        // Edge case: apex at origin (degenerate cone contains all)
        if self.apex.norm() < config.eps {
            return 0.0;
        }

        // Compute tangent vectors
        let tangent = ball.log_map(&self.apex, point);
        let origin = PoincarePoint::origin();
        let to_origin = ball.log_map(&self.apex, &origin);

        // Compute norms
        let tangent_norm: f32 = tangent.iter().map(|x| x * x).sum::<f32>().sqrt();
        let to_origin_norm: f32 = to_origin.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Edge case: degenerate tangent vectors
        if tangent_norm < config.eps || to_origin_norm < config.eps {
            return 0.0;
        }

        // Compute angle via dot product
        let dot: f32 = tangent
            .iter()
            .zip(to_origin.iter())
            .map(|(a, b)| a * b)
            .sum();

        let cos_angle = (dot / (tangent_norm * to_origin_norm)).clamp(-1.0, 1.0);
        cos_angle.acos()
    }

    /// Compute soft membership score for a point.
    ///
    /// # Returns
    ///
    /// Value in [0, 1] where:
    /// - 1.0 = fully contained within cone
    /// - approaching 0 = far outside cone
    ///
    /// # CANONICAL FORMULA (DO NOT MODIFY)
    ///
    /// - If angle <= effective_aperture: score = 1.0
    /// - If angle > effective_aperture: score = exp(-2.0 * (angle - aperture))
    ///
    /// # Arguments
    ///
    /// * `point` - Point to compute membership score for
    /// * `ball` - PoincareBall for hyperbolic operations
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::{PoincareBall, PoincarePoint};
    /// use context_graph_graph::config::{HyperbolicConfig, ConeConfig};
    /// use context_graph_graph::entailment::cones::EntailmentCone;
    ///
    /// let config = ConeConfig::default();
    /// let apex = PoincarePoint::origin();
    /// let cone = EntailmentCone::new(apex.clone(), 0, &config).unwrap();
    /// let ball = PoincareBall::new(HyperbolicConfig::default());
    ///
    /// // Point at apex has score 1.0
    /// assert_eq!(cone.membership_score(&apex, &ball), 1.0);
    /// ```
    pub fn membership_score(&self, point: &PoincarePoint, ball: &PoincareBall) -> f32 {
        let angle = self.compute_angle(point, ball);
        let aperture = self.effective_aperture();

        if angle <= aperture {
            1.0
        } else {
            (-2.0 * (angle - aperture)).exp()
        }
    }

    /// Update aperture factor based on training signal.
    ///
    /// # Arguments
    ///
    /// * `delta` - Adjustment to aperture_factor
    ///   - Positive delta widens the cone (more inclusive)
    ///   - Negative delta narrows the cone (more exclusive)
    ///
    /// # Invariant
    ///
    /// Result is clamped to [0.5, 2.0] range per constitution constraints.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::entailment::cones::EntailmentCone;
    ///
    /// let mut cone = EntailmentCone::default();
    /// assert_eq!(cone.aperture_factor, 1.0);
    ///
    /// // Positive delta widens
    /// cone.update_aperture(0.3);
    /// assert!((cone.aperture_factor - 1.3).abs() < 1e-6);
    ///
    /// // Negative delta narrows
    /// cone.update_aperture(-0.5);
    /// assert!((cone.aperture_factor - 0.8).abs() < 1e-6);
    /// ```
    pub fn update_aperture(&mut self, delta: f32) {
        self.aperture_factor = (self.aperture_factor + delta).clamp(0.5, 2.0);
    }
}
