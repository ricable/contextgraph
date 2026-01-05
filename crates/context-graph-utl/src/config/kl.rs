//! KL divergence computation settings.
//!
//! Controls how KL divergence is computed for measuring information gain
//! and distribution differences in the UTL framework.

use serde::{Deserialize, Serialize};

/// KL divergence computation settings.
///
/// Controls how KL divergence is computed for measuring information gain
/// and distribution differences in the UTL framework.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KlConfig {
    /// Epsilon for numerical stability (prevent log(0)).
    /// Range: `[1e-10, 1e-6]`
    pub epsilon: f64,

    /// Enable symmetric KL divergence (Jensen-Shannon).
    pub symmetric: bool,

    /// Maximum KL divergence value (for clamping).
    pub max_value: f64,

    /// Smoothing factor for probability distributions.
    /// Range: `[0.0, 0.1]`
    pub smoothing: f64,

    /// Number of bins for histogram-based estimation.
    pub histogram_bins: usize,

    /// Enable adaptive binning.
    pub adaptive_binning: bool,
}

impl Default for KlConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-8,
            symmetric: false,
            max_value: 100.0,
            smoothing: 0.01,
            histogram_bins: 256,
            adaptive_binning: true,
        }
    }
}

impl KlConfig {
    /// Validate the KL configuration.
    pub fn validate(&self) -> Result<(), String> {
        if !(1e-10..=1e-6).contains(&self.epsilon) {
            return Err(format!(
                "epsilon must be in [1e-10, 1e-6], got {}",
                self.epsilon
            ));
        }
        if self.max_value <= 0.0 {
            return Err(format!("max_value must be > 0, got {}", self.max_value));
        }
        if !(0.0..=0.1).contains(&self.smoothing) {
            return Err(format!(
                "smoothing must be in [0, 0.1], got {}",
                self.smoothing
            ));
        }
        if self.histogram_bins == 0 {
            return Err("histogram_bins must be > 0".to_string());
        }
        Ok(())
    }
}
