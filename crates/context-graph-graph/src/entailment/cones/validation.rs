//! Validation methods for EntailmentCone.
//!
//! This module contains validation logic for cone invariants.

use crate::error::GraphError;

use super::EntailmentCone;

impl EntailmentCone {
    /// Validate cone parameters.
    ///
    /// # Returns
    ///
    /// `true` if all invariants hold:
    /// - apex.is_valid() (norm < 1.0)
    /// - aperture in (0, π/2]
    /// - aperture_factor in [0.5, 2.0]
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::entailment::cones::EntailmentCone;
    ///
    /// let cone = EntailmentCone::default();
    /// assert!(cone.is_valid());
    /// ```
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.apex.is_valid()
            && self.aperture > 0.0
            && self.aperture <= std::f32::consts::FRAC_PI_2
            && self.aperture_factor >= 0.5
            && self.aperture_factor <= 2.0
    }

    /// Validate cone and return detailed error.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Cone is valid
    /// * `Err(GraphError)` - Specific validation failure
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::entailment::cones::EntailmentCone;
    ///
    /// let cone = EntailmentCone::default();
    /// assert!(cone.validate().is_ok());
    /// ```
    pub fn validate(&self) -> Result<(), GraphError> {
        if !self.apex.is_valid() {
            return Err(GraphError::InvalidHyperbolicPoint {
                norm: self.apex.norm(),
            });
        }
        if self.aperture <= 0.0 || self.aperture > std::f32::consts::FRAC_PI_2 {
            return Err(GraphError::InvalidAperture(self.aperture));
        }
        if self.aperture_factor < 0.5 || self.aperture_factor > 2.0 {
            return Err(GraphError::InvalidConfig(format!(
                "aperture_factor {} outside valid range [0.5, 2.0]",
                self.aperture_factor
            )));
        }
        Ok(())
    }
}
