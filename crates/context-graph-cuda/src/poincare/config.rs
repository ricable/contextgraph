//! Configuration for Poincare CUDA operations.
//!
//! Provides type-safe configuration with validation.

use crate::error::{CudaError, CudaResult};

use super::constants::{DEFAULT_CURVATURE, POINCARE_DIM};

/// Configuration for Poincare CUDA operations.
///
/// # Example
///
/// ```
/// use context_graph_cuda::poincare::PoincareCudaConfig;
///
/// let config = PoincareCudaConfig::default();
/// assert_eq!(config.dim, 64);
/// assert!((config.curvature - (-1.0)).abs() < 1e-6);
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct PoincareCudaConfig {
    /// Dimension of Poincare ball (must be 64 for CUDA kernel).
    pub dim: usize,
    /// Curvature (must be negative, default -1.0).
    pub curvature: f32,
}

impl Default for PoincareCudaConfig {
    fn default() -> Self {
        Self {
            dim: POINCARE_DIM,
            curvature: DEFAULT_CURVATURE,
        }
    }
}

impl PoincareCudaConfig {
    /// Create config with custom curvature.
    ///
    /// # Errors
    ///
    /// Returns error if curvature is not negative.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_cuda::poincare::PoincareCudaConfig;
    ///
    /// let config = PoincareCudaConfig::with_curvature(-0.5).unwrap();
    /// assert!((config.curvature - (-0.5)).abs() < 1e-6);
    ///
    /// // Invalid curvature returns error
    /// assert!(PoincareCudaConfig::with_curvature(0.5).is_err());
    /// ```
    pub fn with_curvature(curvature: f32) -> CudaResult<Self> {
        if curvature >= 0.0 {
            return Err(CudaError::InvalidConfig(
                "Poincare curvature must be negative".to_string(),
            ));
        }
        if curvature.is_nan() {
            return Err(CudaError::InvalidConfig(
                "Poincare curvature cannot be NaN".to_string(),
            ));
        }
        Ok(Self {
            dim: POINCARE_DIM,
            curvature,
        })
    }

    /// Create config with custom dimension and curvature.
    ///
    /// # Errors
    ///
    /// Returns error if dimension is not 64 or curvature is not negative.
    pub fn with_dim_and_curvature(dim: usize, curvature: f32) -> CudaResult<Self> {
        if dim != POINCARE_DIM {
            return Err(CudaError::InvalidConfig(format!(
                "Poincare dimension must be {} for CUDA kernel, got {}",
                POINCARE_DIM, dim
            )));
        }
        Self::with_curvature(curvature)
    }

    /// Validate configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Dimension is not 64
    /// - Curvature is not negative
    /// - Curvature is NaN
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_cuda::poincare::PoincareCudaConfig;
    ///
    /// let config = PoincareCudaConfig::default();
    /// assert!(config.validate().is_ok());
    ///
    /// let bad_config = PoincareCudaConfig { dim: 128, curvature: -1.0 };
    /// assert!(bad_config.validate().is_err());
    /// ```
    pub fn validate(&self) -> CudaResult<()> {
        if self.dim != POINCARE_DIM {
            return Err(CudaError::InvalidConfig(format!(
                "Poincare dimension must be {} for CUDA kernel, got {}",
                POINCARE_DIM, self.dim
            )));
        }
        if self.curvature >= 0.0 {
            return Err(CudaError::InvalidConfig(
                "Poincare curvature must be negative".to_string(),
            ));
        }
        if self.curvature.is_nan() {
            return Err(CudaError::InvalidConfig(
                "Poincare curvature cannot be NaN".to_string(),
            ));
        }
        Ok(())
    }

    /// Get absolute value of curvature (always positive).
    #[inline]
    pub fn abs_curvature(&self) -> f32 {
        self.curvature.abs()
    }
}
