//! Core type definitions for entailment cones.
//!
//! This module contains the [`EntailmentCone`] struct and its Default implementation.

use serde::{Deserialize, Serialize};

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
