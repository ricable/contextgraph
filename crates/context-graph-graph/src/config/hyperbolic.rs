//! Hyperbolic (Poincare ball) configuration.

use serde::{Deserialize, Serialize};

use crate::error::GraphError;

/// Hyperbolic (Poincare ball) configuration.
///
/// Configures the Poincare ball model for representing hierarchical
/// relationships in hyperbolic space.
///
/// # Mathematics
/// - d(x,y) = arcosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
/// - Curvature must be negative (typically -1.0)
/// - All points must have norm < 1.0
///
/// # Constitution Reference
/// - edge_model.nt_weights: Neurotransmitter weighting in hyperbolic space
/// - perf.latency.entailment_check: <1ms
///
/// # Example
/// ```
/// use context_graph_graph::config::HyperbolicConfig;
///
/// let config = HyperbolicConfig::default();
/// assert_eq!(config.dim, 64);
/// assert_eq!(config.curvature, -1.0);
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HyperbolicConfig {
    /// Dimension of hyperbolic space (typically 64 for knowledge graphs).
    /// Must be positive.
    pub dim: usize,

    /// Curvature of hyperbolic space. MUST be negative.
    /// Default: -1.0 (unit hyperbolic space)
    /// Validated in validate().
    pub curvature: f32,

    /// Epsilon for numerical stability in hyperbolic operations.
    /// Prevents division by zero and NaN in distance calculations.
    /// Default: 1e-7
    pub eps: f32,

    /// Maximum norm for points (keeps points strictly inside ball boundary).
    /// Points with norm >= max_norm will be projected back inside.
    /// Must be in open interval (0, 1). Default: 1.0 - 1e-5 = 0.99999
    pub max_norm: f32,
}

impl Default for HyperbolicConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            curvature: -1.0,
            eps: 1e-7,
            max_norm: 1.0 - 1e-5, // 0.99999
        }
    }
}

impl HyperbolicConfig {
    /// Create config with custom curvature.
    ///
    /// # Arguments
    /// * `curvature` - Must be negative. Use validate() to check.
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::HyperbolicConfig;
    /// let config = HyperbolicConfig::with_curvature(-0.5);
    /// assert_eq!(config.curvature, -0.5);
    /// assert_eq!(config.dim, 64); // other fields use defaults
    /// ```
    pub fn with_curvature(curvature: f32) -> Self {
        Self {
            curvature,
            ..Default::default()
        }
    }

    /// Get absolute value of curvature.
    ///
    /// Useful for formulas that need |c| rather than c.
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::HyperbolicConfig;
    /// let config = HyperbolicConfig::default();
    /// assert_eq!(config.abs_curvature(), 1.0);
    /// ```
    #[inline]
    pub fn abs_curvature(&self) -> f32 {
        self.curvature.abs()
    }

    /// Scale factor derived from curvature: sqrt(|c|)
    ///
    /// Used in Mobius operations and distance calculations.
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::HyperbolicConfig;
    /// let config = HyperbolicConfig::default();
    /// assert_eq!(config.scale(), 1.0); // sqrt(|-1.0|) = 1.0
    /// ```
    #[inline]
    pub fn scale(&self) -> f32 {
        self.abs_curvature().sqrt()
    }

    /// Validate that all configuration parameters are mathematically valid
    /// for the Poincare ball model.
    ///
    /// # Validation Rules
    /// - `dim` > 0: Dimension must be positive
    /// - `curvature` < 0: Must be negative for hyperbolic space
    /// - `eps` > 0: Must be positive for numerical stability
    /// - `max_norm` in (0, 1): Must be strictly between 0 and 1
    ///
    /// # Errors
    /// Returns `GraphError::InvalidConfig` with descriptive message if any
    /// parameter is invalid. Returns the FIRST error encountered (fail-fast).
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// // Valid config passes
    /// let valid = HyperbolicConfig::default();
    /// assert!(valid.validate().is_ok());
    ///
    /// // Invalid curvature fails
    /// let mut invalid = HyperbolicConfig::default();
    /// invalid.curvature = 1.0; // positive is invalid
    /// assert!(invalid.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), GraphError> {
        // Check dimension
        if self.dim == 0 {
            return Err(GraphError::InvalidConfig(
                "dim must be positive (got 0)".to_string(),
            ));
        }

        // Check curvature - MUST be negative for hyperbolic space
        if self.curvature >= 0.0 {
            return Err(GraphError::InvalidConfig(format!(
                "curvature must be negative for hyperbolic space (got {})",
                self.curvature
            )));
        }

        // Check for NaN curvature
        if self.curvature.is_nan() {
            return Err(GraphError::InvalidConfig(
                "curvature cannot be NaN".to_string(),
            ));
        }

        // Check epsilon
        if self.eps <= 0.0 {
            return Err(GraphError::InvalidConfig(format!(
                "eps must be positive for numerical stability (got {})",
                self.eps
            )));
        }

        // Check for NaN eps
        if self.eps.is_nan() {
            return Err(GraphError::InvalidConfig("eps cannot be NaN".to_string()));
        }

        // Check max_norm - must be in open interval (0, 1)
        if self.max_norm <= 0.0 || self.max_norm >= 1.0 {
            return Err(GraphError::InvalidConfig(format!(
                "max_norm must be in open interval (0, 1), got {}",
                self.max_norm
            )));
        }

        // Check for NaN max_norm
        if self.max_norm.is_nan() {
            return Err(GraphError::InvalidConfig(
                "max_norm cannot be NaN".to_string(),
            ));
        }

        Ok(())
    }

    /// Create a validated config with custom curvature.
    ///
    /// Returns error if curvature is invalid (>= 0 or NaN).
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// let config = HyperbolicConfig::try_with_curvature(-0.5).unwrap();
    /// assert_eq!(config.curvature, -0.5);
    ///
    /// // Invalid curvature returns error
    /// assert!(HyperbolicConfig::try_with_curvature(1.0).is_err());
    /// ```
    pub fn try_with_curvature(curvature: f32) -> Result<Self, GraphError> {
        let config = Self {
            curvature,
            ..Default::default()
        };
        config.validate()?;
        Ok(config)
    }
}
