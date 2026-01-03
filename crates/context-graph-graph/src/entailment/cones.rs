//! EntailmentCone implementation for O(1) IS-A hierarchy queries.
//!
//! An entailment cone in hyperbolic space enables efficient hierarchical
//! reasoning. A concept's cone contains all concepts it subsumes (entails).
//! Checking if concept A is a subconcept of B is O(1): check if A's position
//! lies within B's cone.
//!
//! # Aperture Decay
//!
//! Aperture decreases with hierarchy depth:
//! - Root concepts have wide cones (capture many descendants)
//! - Leaf concepts have narrow cones (very specific)
//! - Formula: `aperture = base * decay^depth`, clamped to [min, max]
//!
//! # Performance Targets
//!
//! - Cone containment check: <50μs CPU
//! - Entailment check: <1ms total
//! - Target hardware: RTX 5090, CUDA 13.1, Compute 12.0
//!
//! # Constitution Reference
//!
//! - perf.latency.entailment_check: <1ms
//! - Section 9 "HYPERBOLIC ENTAILMENT CONES" in contextprd.md

use serde::{Deserialize, Serialize};

use crate::config::ConeConfig;
use crate::error::GraphError;
use crate::hyperbolic::mobius::PoincareBall;
use crate::hyperbolic::poincare::PoincarePoint;

/// Entailment cone for O(1) IS-A hierarchy queries.
///
/// A cone rooted at `apex` with angular width `aperture * aperture_factor`
/// contains all points (concepts) that are entailed by the apex concept.
///
/// # Memory Layout
///
/// - apex: 256 bytes (64 f32 coords, 64-byte aligned)
/// - aperture: 4 bytes
/// - aperture_factor: 4 bytes
/// - depth: 4 bytes
/// - Total: 268 bytes (with padding for alignment)
///
/// # Invariants
///
/// - `apex.is_valid()` must be true (norm < 1.0)
/// - `aperture` in (0, π/2]
/// - `aperture_factor` in [0.5, 2.0]
///
/// # Example
///
/// ```
/// use context_graph_graph::hyperbolic::poincare::PoincarePoint;
/// use context_graph_graph::config::ConeConfig;
/// use context_graph_graph::entailment::cones::EntailmentCone;
///
/// let apex = PoincarePoint::origin();
/// let config = ConeConfig::default();
/// let cone = EntailmentCone::new(apex, 0, &config).expect("valid cone");
/// assert!(cone.is_valid());
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntailmentCone {
    /// Apex point of the cone in Poincare ball.
    pub apex: PoincarePoint,
    /// Base aperture in radians (computed from depth via ConeConfig).
    pub aperture: f32,
    /// Adjustment factor for aperture (learned during training).
    pub aperture_factor: f32,
    /// Depth in hierarchy (0 = root concept).
    pub depth: u32,
}

impl EntailmentCone {
    /// Create a new entailment cone at given apex position.
    ///
    /// # Arguments
    ///
    /// * `apex` - Position in Poincare ball (must satisfy ||coords|| < 1)
    /// * `depth` - Hierarchy depth (affects aperture via decay)
    /// * `config` - ConeConfig for aperture computation
    ///
    /// # Returns
    ///
    /// * `Ok(EntailmentCone)` - Valid cone
    /// * `Err(GraphError::InvalidHyperbolicPoint)` - If apex is invalid
    /// * `Err(GraphError::InvalidAperture)` - If computed aperture is invalid
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::poincare::PoincarePoint;
    /// use context_graph_graph::config::ConeConfig;
    /// use context_graph_graph::entailment::cones::EntailmentCone;
    ///
    /// let apex = PoincarePoint::origin();
    /// let config = ConeConfig::default();
    /// let cone = EntailmentCone::new(apex, 0, &config).expect("valid cone");
    /// assert!(cone.is_valid());
    /// assert_eq!(cone.depth, 0);
    /// assert_eq!(cone.aperture, config.base_aperture);
    /// ```
    pub fn new(apex: PoincarePoint, depth: u32, config: &ConeConfig) -> Result<Self, GraphError> {
        // FAIL FAST: Validate apex immediately
        if !apex.is_valid() {
            tracing::error!(
                norm = apex.norm(),
                "Invalid apex point: norm must be < 1.0"
            );
            return Err(GraphError::InvalidHyperbolicPoint { norm: apex.norm() });
        }

        let aperture = config.compute_aperture(depth);

        // FAIL FAST: Validate computed aperture
        if aperture <= 0.0 || aperture > std::f32::consts::FRAC_PI_2 {
            tracing::error!(
                aperture = aperture,
                depth = depth,
                "Invalid aperture computed from config"
            );
            return Err(GraphError::InvalidAperture(aperture));
        }

        Ok(Self {
            apex,
            aperture,
            aperture_factor: 1.0,
            depth,
        })
    }

    /// Create cone with explicit aperture (for deserialization/testing).
    ///
    /// # Arguments
    ///
    /// * `apex` - Position in Poincare ball
    /// * `aperture` - Explicit aperture in radians
    /// * `depth` - Hierarchy depth
    ///
    /// # Returns
    ///
    /// * `Ok(EntailmentCone)` - Valid cone
    /// * `Err(GraphError)` - If validation fails
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_graph::hyperbolic::poincare::PoincarePoint;
    /// use context_graph_graph::entailment::cones::EntailmentCone;
    ///
    /// let apex = PoincarePoint::origin();
    /// let cone = EntailmentCone::with_aperture(apex, 0.5, 0).expect("valid cone");
    /// assert_eq!(cone.aperture, 0.5);
    /// ```
    pub fn with_aperture(
        apex: PoincarePoint,
        aperture: f32,
        depth: u32,
    ) -> Result<Self, GraphError> {
        // FAIL FAST: Validate apex
        if !apex.is_valid() {
            tracing::error!(norm = apex.norm(), "Invalid apex point");
            return Err(GraphError::InvalidHyperbolicPoint { norm: apex.norm() });
        }

        // FAIL FAST: Validate aperture range
        if aperture <= 0.0 || aperture > std::f32::consts::FRAC_PI_2 {
            tracing::error!(
                aperture = aperture,
                "Aperture out of valid range (0, π/2]"
            );
            return Err(GraphError::InvalidAperture(aperture));
        }

        Ok(Self {
            apex,
            aperture,
            aperture_factor: 1.0,
            depth,
        })
    }

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
    /// # Performance Target
    ///
    /// <50μs on CPU
    ///
    /// # Arguments
    ///
    /// * `point` - Point to check for containment
    /// * `ball` - PoincareBall for hyperbolic operations
    ///
    /// # Note
    ///
    /// Full implementation in M04-T07. This is a stub.
    pub fn contains(&self, _point: &PoincarePoint, _ball: &PoincareBall) -> bool {
        // M04-T07 will implement:
        // 1. Compute log_map from apex to point (tangent vector)
        // 2. Compute angle between tangent and cone axis
        // 3. Return angle <= effective_aperture()
        todo!("Containment logic implemented in M04-T07")
    }

    /// Compute soft membership score for a point.
    ///
    /// # Returns
    ///
    /// - 1.0 if contained within cone
    /// - Exponentially decaying value if outside
    ///
    /// # Formula (Canonical)
    ///
    /// - If angle <= aperture: score = 1.0
    /// - If angle > aperture: score = exp(-2.0 * (angle - aperture))
    ///
    /// # Arguments
    ///
    /// * `point` - Point to compute membership score for
    /// * `ball` - PoincareBall for hyperbolic operations
    ///
    /// # Note
    ///
    /// Full implementation in M04-T07. This is a stub.
    pub fn membership_score(&self, _point: &PoincarePoint, _ball: &PoincareBall) -> f32 {
        todo!("Membership score implemented in M04-T07")
    }

    /// Update aperture factor based on training signal.
    ///
    /// # Arguments
    ///
    /// * `delta` - Adjustment to apply to aperture_factor
    ///
    /// # Note
    ///
    /// Full implementation in M04-T07. This is a stub.
    pub fn update_aperture(&mut self, _delta: f32) {
        todo!("Update aperture implemented in M04-T07")
    }

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

impl Default for EntailmentCone {
    /// Create a default cone at origin with base aperture.
    ///
    /// # Returns
    ///
    /// Cone with:
    /// - apex: origin point
    /// - aperture: 1.0 (ConeConfig default base_aperture)
    /// - aperture_factor: 1.0
    /// - depth: 0
    fn default() -> Self {
        Self {
            apex: PoincarePoint::origin(),
            aperture: 1.0, // ConeConfig default base_aperture
            aperture_factor: 1.0,
            depth: 0,
        }
    }
}

// ============================================================================
// TESTS - MUST USE REAL DATA, NO MOCKS (per constitution REQ-KG-TEST)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ConeConfig;
    use crate::hyperbolic::poincare::PoincarePoint;

    // ========== CONSTRUCTION TESTS ==========

    /// Test creation with default ConeConfig
    #[test]
    fn test_new_with_default_config() {
        let config = ConeConfig::default();
        let apex = PoincarePoint::origin();
        let cone = EntailmentCone::new(apex, 0, &config).expect("Should create valid cone");

        assert!(cone.is_valid());
        assert_eq!(cone.depth, 0);
        assert_eq!(cone.aperture, config.base_aperture);
        assert_eq!(cone.aperture_factor, 1.0);
    }

    /// Test aperture decay with depth
    #[test]
    fn test_aperture_decay_with_depth() {
        let config = ConeConfig::default();
        let apex = PoincarePoint::origin();

        let cone_d0 = EntailmentCone::new(apex.clone(), 0, &config).expect("valid");
        let cone_d1 = EntailmentCone::new(apex.clone(), 1, &config).expect("valid");
        let cone_d5 = EntailmentCone::new(apex.clone(), 5, &config).expect("valid");

        // Deeper = narrower aperture
        assert!(
            cone_d0.aperture > cone_d1.aperture,
            "depth 0 should have wider aperture than depth 1"
        );
        assert!(
            cone_d1.aperture > cone_d5.aperture,
            "depth 1 should have wider aperture than depth 5"
        );

        // Verify formula: base * decay^depth
        let expected_d1 = config.base_aperture * config.aperture_decay;
        assert!(
            (cone_d1.aperture - expected_d1).abs() < 1e-6,
            "Expected {}, got {}",
            expected_d1,
            cone_d1.aperture
        );
    }

    /// Test very deep node clamps to min_aperture
    #[test]
    fn test_deep_node_min_aperture() {
        let config = ConeConfig::default();
        let apex = PoincarePoint::origin();
        let cone = EntailmentCone::new(apex, 100, &config).expect("valid");

        assert_eq!(
            cone.aperture, config.min_aperture,
            "Very deep node should clamp to min_aperture"
        );
    }

    /// Test with_aperture constructor
    #[test]
    fn test_with_aperture_constructor() {
        let apex = PoincarePoint::origin();
        let cone = EntailmentCone::with_aperture(apex, 0.5, 3).expect("valid");

        assert_eq!(cone.aperture, 0.5);
        assert_eq!(cone.depth, 3);
        assert_eq!(cone.aperture_factor, 1.0);
        assert!(cone.is_valid());
    }

    // ========== EFFECTIVE APERTURE TESTS ==========

    /// Test effective_aperture with factor
    #[test]
    fn test_effective_aperture() {
        let mut cone = EntailmentCone::default();
        cone.aperture = 0.5;
        cone.aperture_factor = 1.5;

        let effective = cone.effective_aperture();
        assert!(
            (effective - 0.75).abs() < 1e-6,
            "Expected 0.75, got {}",
            effective
        );
    }

    /// Test effective_aperture clamping to pi/2
    #[test]
    fn test_effective_aperture_clamp_max() {
        let mut cone = EntailmentCone::default();
        cone.aperture = 1.5;
        cone.aperture_factor = 2.0;

        let effective = cone.effective_aperture();
        assert!(
            (effective - std::f32::consts::FRAC_PI_2).abs() < 1e-6,
            "Should clamp to π/2, got {}",
            effective
        );
    }

    /// Test effective_aperture clamping to minimum
    #[test]
    fn test_effective_aperture_clamp_min() {
        let mut cone = EntailmentCone::default();
        cone.aperture = 1e-8;
        cone.aperture_factor = 0.5;

        let effective = cone.effective_aperture();
        assert!(
            effective >= 1e-6,
            "Should clamp to minimum epsilon, got {}",
            effective
        );
    }

    // ========== VALIDATION TESTS ==========

    /// Test invalid apex rejection (FAIL FAST)
    #[test]
    fn test_invalid_apex_fails_fast() {
        let config = ConeConfig::default();
        // Create point with norm >= 1
        let mut coords = [0.0f32; 64];
        coords[0] = 1.0; // norm = 1.0, invalid for Poincare ball
        let invalid_apex = PoincarePoint::from_coords(coords);

        let result = EntailmentCone::new(invalid_apex, 0, &config);
        assert!(result.is_err(), "Should reject invalid apex");

        match result {
            Err(GraphError::InvalidHyperbolicPoint { norm }) => {
                assert!(
                    (norm - 1.0).abs() < 1e-6,
                    "Should report correct norm in error"
                );
            }
            _ => panic!("Expected InvalidHyperbolicPoint error"),
        }
    }

    /// Test with_aperture validation - zero aperture
    #[test]
    fn test_with_aperture_zero_fails() {
        let apex = PoincarePoint::origin();
        let result = EntailmentCone::with_aperture(apex, 0.0, 0);
        assert!(matches!(result, Err(GraphError::InvalidAperture(_))));
    }

    /// Test with_aperture validation - negative aperture
    #[test]
    fn test_with_aperture_negative_fails() {
        let apex = PoincarePoint::origin();
        let result = EntailmentCone::with_aperture(apex, -0.5, 0);
        assert!(matches!(result, Err(GraphError::InvalidAperture(_))));
    }

    /// Test with_aperture validation - aperture > π/2
    #[test]
    fn test_with_aperture_exceeds_max_fails() {
        let apex = PoincarePoint::origin();
        let result = EntailmentCone::with_aperture(apex, 2.0, 0);
        assert!(matches!(result, Err(GraphError::InvalidAperture(_))));
    }

    /// Test validate() returns specific errors
    #[test]
    fn test_validate_specific_errors() {
        let mut cone = EntailmentCone::default();
        assert!(cone.validate().is_ok());

        // Invalid aperture_factor below 0.5
        cone.aperture_factor = 0.3;
        let result = cone.validate();
        assert!(matches!(result, Err(GraphError::InvalidConfig(_))));

        // Invalid aperture_factor above 2.0
        cone.aperture_factor = 2.5;
        let result = cone.validate();
        assert!(matches!(result, Err(GraphError::InvalidConfig(_))));
    }

    /// Test is_valid returns false for invalid configurations
    #[test]
    fn test_is_valid_invalid_configurations() {
        let mut cone = EntailmentCone::default();
        assert!(cone.is_valid());

        // Invalid aperture
        cone.aperture = 0.0;
        assert!(!cone.is_valid(), "Zero aperture should be invalid");

        cone.aperture = 1.0;
        cone.aperture_factor = 0.4;
        assert!(
            !cone.is_valid(),
            "aperture_factor below 0.5 should be invalid"
        );

        cone.aperture_factor = 2.1;
        assert!(
            !cone.is_valid(),
            "aperture_factor above 2.0 should be invalid"
        );
    }

    // ========== SERIALIZATION TESTS ==========

    /// Test serialization roundtrip preserves all fields
    #[test]
    fn test_serde_roundtrip() {
        let config = ConeConfig::default();
        let mut coords = [0.0f32; 64];
        coords[0] = 0.5;
        coords[1] = 0.3;
        let apex = PoincarePoint::from_coords(coords);

        let original = EntailmentCone::new(apex, 5, &config).expect("valid");

        let serialized = bincode::serialize(&original).expect("Serialization failed");
        let deserialized: EntailmentCone =
            bincode::deserialize(&serialized).expect("Deserialization failed");

        assert_eq!(original.depth, deserialized.depth);
        assert!((original.aperture - deserialized.aperture).abs() < 1e-6);
        assert!((original.aperture_factor - deserialized.aperture_factor).abs() < 1e-6);
        assert!(deserialized.is_valid());
    }

    /// Test serialized size is approximately 268 bytes
    #[test]
    fn test_serialized_size() {
        let cone = EntailmentCone::default();
        let serialized = bincode::serialize(&cone).expect("Serialization failed");

        // 256 (coords) + 4 (aperture) + 4 (factor) + 4 (depth) = 268
        // bincode may add small overhead
        assert!(
            serialized.len() >= 268,
            "Serialized size should be at least 268 bytes, got {}",
            serialized.len()
        );
        assert!(
            serialized.len() <= 280,
            "Serialized size should not exceed 280 bytes, got {}",
            serialized.len()
        );
    }

    // ========== DEFAULT TESTS ==========

    /// Test Default creates valid cone
    #[test]
    fn test_default_is_valid() {
        let cone = EntailmentCone::default();
        assert!(cone.is_valid());
        assert!(cone.validate().is_ok());
        assert_eq!(cone.depth, 0);
        assert_eq!(cone.aperture, 1.0);
        assert_eq!(cone.aperture_factor, 1.0);
    }

    /// Test default apex is at origin
    #[test]
    fn test_default_apex_is_origin() {
        let cone = EntailmentCone::default();
        assert_eq!(cone.apex.norm(), 0.0);
    }

    // ========== EDGE CASE TESTS ==========

    /// Test apex at origin edge case
    #[test]
    fn test_apex_at_origin() {
        let config = ConeConfig::default();
        let origin = PoincarePoint::origin();

        // Should create valid cone (degenerate but allowed)
        let cone = EntailmentCone::new(origin, 0, &config).expect("valid");
        assert!(cone.is_valid());
        assert_eq!(cone.apex.norm(), 0.0);
    }

    /// Test apex near boundary
    #[test]
    fn test_apex_near_boundary() {
        let config = ConeConfig::default();
        // Create point with norm ≈ 0.99 (near but inside boundary)
        let scale = 0.99 / (64.0_f32).sqrt();
        let coords = [scale; 64];
        let apex = PoincarePoint::from_coords(coords);

        assert!(apex.is_valid(), "Apex should be valid");
        let cone = EntailmentCone::new(apex, 0, &config).expect("valid");
        assert!(cone.is_valid());
    }

    /// Test aperture at maximum (π/2)
    #[test]
    fn test_aperture_at_max() {
        let apex = PoincarePoint::origin();
        let cone =
            EntailmentCone::with_aperture(apex, std::f32::consts::FRAC_PI_2, 0).expect("valid");
        assert!(cone.is_valid());
        assert_eq!(cone.aperture, std::f32::consts::FRAC_PI_2);
    }

    /// Test aperture_factor at bounds
    #[test]
    fn test_aperture_factor_at_bounds() {
        let mut cone = EntailmentCone::default();

        // At lower bound
        cone.aperture_factor = 0.5;
        assert!(cone.is_valid());
        assert!(cone.validate().is_ok());

        // At upper bound
        cone.aperture_factor = 2.0;
        assert!(cone.is_valid());
        assert!(cone.validate().is_ok());
    }

    /// Integration test with real ConeConfig values
    #[test]
    fn test_real_config_integration() {
        // Use actual default values from config.rs
        let config = ConeConfig {
            min_aperture: 0.1,
            max_aperture: 1.5,
            base_aperture: 1.0,
            aperture_decay: 0.85,
            membership_threshold: 0.7,
        };

        let apex = PoincarePoint::origin();

        // Test multiple depths
        for depth in [0, 1, 5, 10, 20] {
            let cone = EntailmentCone::new(apex.clone(), depth, &config).expect("valid");
            assert!(cone.is_valid(), "Cone at depth {} should be valid", depth);
            assert!(
                cone.aperture >= config.min_aperture,
                "Aperture at depth {} should be >= min_aperture",
                depth
            );
            assert!(
                cone.aperture <= config.max_aperture,
                "Aperture at depth {} should be <= max_aperture",
                depth
            );
        }
    }

    /// Test Clone trait
    #[test]
    fn test_clone_is_independent() {
        let config = ConeConfig::default();
        let mut coords = [0.0f32; 64];
        coords[0] = 0.3;
        let apex = PoincarePoint::from_coords(coords);

        let original = EntailmentCone::new(apex, 5, &config).expect("valid");
        let cloned = original.clone();

        // Clone should have identical values
        assert_eq!(original.aperture, cloned.aperture);
        assert_eq!(original.depth, cloned.depth);
        assert_eq!(original.aperture_factor, cloned.aperture_factor);

        // Verify expected aperture: 1.0 * 0.85^5 = 0.4437
        let expected_aperture = 1.0 * 0.85_f32.powi(5);
        assert!(
            (original.aperture - expected_aperture).abs() < 0.01,
            "Expected {}, got {}",
            expected_aperture,
            original.aperture
        );
        assert_eq!(original.depth, 5);
    }

    /// Test Debug trait
    #[test]
    fn test_debug_output() {
        let cone = EntailmentCone::default();
        let debug_str = format!("{:?}", cone);
        assert!(debug_str.contains("EntailmentCone"));
        assert!(debug_str.contains("apex"));
        assert!(debug_str.contains("aperture"));
        assert!(debug_str.contains("aperture_factor"));
        assert!(debug_str.contains("depth"));
    }

    /// Test with various valid apex positions
    #[test]
    fn test_various_apex_positions() {
        let config = ConeConfig::default();

        // Near origin
        let mut coords1 = [0.0f32; 64];
        coords1[0] = 0.01;
        let apex1 = PoincarePoint::from_coords(coords1);
        assert!(EntailmentCone::new(apex1, 0, &config).is_ok());

        // Medium distance
        let mut coords2 = [0.0f32; 64];
        coords2[0] = 0.5;
        coords2[1] = 0.3;
        let apex2 = PoincarePoint::from_coords(coords2);
        assert!(EntailmentCone::new(apex2, 0, &config).is_ok());

        // Near boundary
        let mut coords3 = [0.0f32; 64];
        coords3[0] = 0.9;
        let apex3 = PoincarePoint::from_coords(coords3);
        assert!(EntailmentCone::new(apex3, 0, &config).is_ok());
    }

    /// Test depth affects aperture monotonically
    #[test]
    fn test_depth_aperture_monotonicity() {
        let config = ConeConfig::default();
        let apex = PoincarePoint::origin();

        let mut prev_aperture = f32::INFINITY;
        for depth in 0..20 {
            let cone = EntailmentCone::new(apex.clone(), depth, &config).expect("valid");
            assert!(
                cone.aperture <= prev_aperture,
                "Aperture should decrease or stay same with depth"
            );
            prev_aperture = cone.aperture;
        }
    }

    /// Test PartialEq for PoincarePoint in apex
    #[test]
    fn test_apex_equality() {
        let config = ConeConfig::default();
        let apex = PoincarePoint::origin();

        let cone1 = EntailmentCone::new(apex.clone(), 0, &config).expect("valid");
        let cone2 = EntailmentCone::new(apex, 0, &config).expect("valid");

        assert_eq!(cone1.apex, cone2.apex);
        assert_eq!(cone1.aperture, cone2.aperture);
        assert_eq!(cone1.depth, cone2.depth);
    }

    /// Test NaN handling in apex
    #[test]
    fn test_nan_apex_rejected() {
        let config = ConeConfig::default();
        let mut coords = [0.0f32; 64];
        coords[0] = f32::NAN;
        let apex = PoincarePoint::from_coords(coords);

        // NaN norm < 1.0 is false, so should be rejected
        let result = EntailmentCone::new(apex, 0, &config);
        assert!(result.is_err(), "NaN apex should be rejected");
    }

    /// Test infinity handling in apex
    #[test]
    fn test_infinity_apex_rejected() {
        let config = ConeConfig::default();
        let mut coords = [0.0f32; 64];
        coords[0] = f32::INFINITY;
        let apex = PoincarePoint::from_coords(coords);

        let result = EntailmentCone::new(apex, 0, &config);
        assert!(result.is_err(), "Infinity apex should be rejected");
    }
}
