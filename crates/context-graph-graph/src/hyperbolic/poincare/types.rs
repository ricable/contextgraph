//! PoincarePoint type definition.
//!
//! Defines the core type for 64D hyperbolic space representation using the
//! Poincare ball model.

use serde::{Deserialize, Serialize};
use serde_with::serde_as;

/// Point in 64-dimensional Poincare ball model of hyperbolic space.
///
/// # Constraint
///
/// `||coords|| < 1.0` (strictly inside unit ball)
///
/// # Memory Layout
///
/// - Size: 256 bytes (64 * f32)
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
