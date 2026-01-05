//! Type definitions for Poincare ball model.
//!
//! Contains the core `PoincareBall` struct for Mobius algebra operations.

use crate::config::HyperbolicConfig;

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
    pub(crate) config: HyperbolicConfig,
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
}
