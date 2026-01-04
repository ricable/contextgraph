//! KL divergence computation for surprise measurement.
//!
//! Implements Kullback-Leibler divergence for measuring the difference between
//! probability distributions. KL divergence is a fundamental measure for
//! quantifying information gain and surprise in probabilistic contexts.
//!
//! # Mathematical Background
//!
//! KL divergence from distribution P to Q is defined as:
//! ```text
//! D_KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))
//! ```
//!
//! This module also supports symmetric KL divergence (Jensen-Shannon style):
//! ```text
//! D_sym(P, Q) = 0.5 * D_KL(P || Q) + 0.5 * D_KL(Q || P)
//! ```
//!
//! # Numerical Stability
//!
//! Per AP-009, all outputs are clamped to valid ranges with no NaN or Infinity values.
//! Epsilon smoothing is applied to prevent log(0) errors.

use crate::config::{KlConfig, SurpriseConfig};
use crate::error::{UtlError, UtlResult};

/// Compute KL divergence between two probability distributions.
///
/// # Arguments
///
/// * `p` - The "true" or reference distribution
/// * `q` - The "approximate" or model distribution
/// * `epsilon` - Small value for numerical stability (prevents log(0))
///
/// # Returns
///
/// The KL divergence value D_KL(P || Q), always non-negative.
/// Returns 0.0 for empty or mismatched distributions.
///
/// # Example
///
/// ```
/// use context_graph_utl::surprise::compute_kl_divergence;
///
/// let p = vec![0.25, 0.25, 0.25, 0.25]; // Uniform
/// let q = vec![0.1, 0.2, 0.3, 0.4];     // Non-uniform
///
/// let kl = compute_kl_divergence(&p, &q, 1e-10);
/// assert!(kl > 0.0); // Different distributions have positive KL
/// ```
pub fn compute_kl_divergence(p: &[f32], q: &[f32], epsilon: f32) -> f32 {
    // Handle edge cases
    if p.is_empty() || q.is_empty() || p.len() != q.len() {
        return 0.0;
    }

    let eps = epsilon.max(1e-15);
    let mut kl = 0.0f64;

    for (p_i, q_i) in p.iter().zip(q.iter()) {
        let p_val = (*p_i as f64).max(eps as f64);
        let q_val = (*q_i as f64).max(eps as f64);

        // D_KL = sum(P * log(P/Q))
        kl += p_val * (p_val / q_val).ln();
    }

    // Clamp to valid range (KL is always non-negative, but floating point errors can occur)
    let result = kl.max(0.0) as f32;

    // Handle potential NaN/Infinity per AP-009
    if result.is_nan() || result.is_infinite() {
        0.0
    } else {
        result
    }
}

/// KL divergence calculator with configurable settings.
///
/// Provides a stateful calculator for KL divergence with options for
/// symmetric computation, smoothing, and maximum value clamping.
///
/// # Example
///
/// ```
/// use context_graph_utl::surprise::KlDivergenceCalculator;
///
/// let calc = KlDivergenceCalculator::default();
/// let p = vec![0.5, 0.5];
/// let q = vec![0.25, 0.75];
///
/// let kl = calc.compute(&p, &q).unwrap();
/// assert!(kl >= 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct KlDivergenceCalculator {
    /// Epsilon for numerical stability.
    epsilon: f64,
    /// Whether to compute symmetric KL divergence.
    symmetric: bool,
    /// Maximum value for clamping output.
    max_value: f64,
    /// Smoothing factor for distributions.
    smoothing: f64,
}

impl Default for KlDivergenceCalculator {
    fn default() -> Self {
        Self {
            epsilon: 1e-8,
            symmetric: false,
            max_value: 100.0,
            smoothing: 0.01,
        }
    }
}

impl KlDivergenceCalculator {
    /// Create a new KL divergence calculator from SurpriseConfig.
    ///
    /// # Arguments
    ///
    /// * `config` - The surprise configuration containing KL settings
    pub fn from_config(_config: &SurpriseConfig, kl_config: &KlConfig) -> Self {
        Self {
            epsilon: kl_config.epsilon,
            symmetric: kl_config.symmetric,
            max_value: kl_config.max_value,
            smoothing: kl_config.smoothing,
        }
    }

    /// Create a new KL divergence calculator with custom settings.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Small value for numerical stability
    /// * `symmetric` - Whether to compute symmetric KL divergence
    /// * `max_value` - Maximum output value for clamping
    pub fn new(epsilon: f64, symmetric: bool, max_value: f64) -> Self {
        Self {
            epsilon: epsilon.max(1e-15),
            symmetric,
            max_value: max_value.max(0.0),
            smoothing: 0.01,
        }
    }

    /// Set the smoothing factor for distributions.
    pub fn with_smoothing(mut self, smoothing: f64) -> Self {
        self.smoothing = smoothing.clamp(0.0, 0.5);
        self
    }

    /// Enable or disable symmetric KL divergence.
    pub fn with_symmetric(mut self, symmetric: bool) -> Self {
        self.symmetric = symmetric;
        self
    }

    /// Compute KL divergence between two distributions.
    ///
    /// # Arguments
    ///
    /// * `p` - The reference distribution
    /// * `q` - The comparison distribution
    ///
    /// # Returns
    ///
    /// The KL divergence value, or an error if computation fails.
    ///
    /// # Errors
    ///
    /// Returns `UtlError::EmptyInput` if either distribution is empty.
    /// Returns `UtlError::DimensionMismatch` if distributions have different lengths.
    pub fn compute(&self, p: &[f32], q: &[f32]) -> UtlResult<f32> {
        // Validate inputs
        if p.is_empty() || q.is_empty() {
            return Err(UtlError::EmptyInput);
        }

        if p.len() != q.len() {
            return Err(UtlError::DimensionMismatch {
                expected: p.len(),
                actual: q.len(),
            });
        }

        // Apply smoothing if configured
        let (p_smooth, q_smooth) = if self.smoothing > 0.0 {
            (self.smooth_distribution(p), self.smooth_distribution(q))
        } else {
            (p.to_vec(), q.to_vec())
        };

        // Compute KL divergence
        let kl = if self.symmetric {
            self.compute_symmetric_kl(&p_smooth, &q_smooth)
        } else {
            self.compute_asymmetric_kl(&p_smooth, &q_smooth)
        };

        // Clamp to max value and handle NaN/Infinity per AP-009
        let result = if kl.is_nan() || kl.is_infinite() {
            0.0
        } else {
            kl.clamp(0.0, self.max_value as f32)
        };

        Ok(result)
    }

    /// Compute normalized KL divergence in range [0, 1].
    ///
    /// Uses a sigmoid-like transformation to map unbounded KL to [0, 1].
    ///
    /// # Arguments
    ///
    /// * `p` - The reference distribution
    /// * `q` - The comparison distribution
    ///
    /// # Returns
    ///
    /// Normalized surprise value in [0, 1].
    pub fn compute_normalized(&self, p: &[f32], q: &[f32]) -> UtlResult<f32> {
        let kl = self.compute(p, q)?;

        // Apply sigmoid-like normalization: 1 - exp(-kl)
        // This maps [0, inf) to [0, 1)
        let normalized = 1.0 - (-kl).exp();

        // Clamp to ensure valid range per AP-009
        Ok(normalized.clamp(0.0, 1.0))
    }

    /// Smooth a distribution by mixing with uniform.
    fn smooth_distribution(&self, dist: &[f32]) -> Vec<f32> {
        let n = dist.len() as f64;
        let uniform = 1.0 / n;
        let alpha = self.smoothing;

        dist.iter()
            .map(|&x| {
                let smoothed = (1.0 - alpha) * (x as f64) + alpha * uniform;
                smoothed as f32
            })
            .collect()
    }

    /// Compute asymmetric KL divergence D_KL(P || Q).
    fn compute_asymmetric_kl(&self, p: &[f32], q: &[f32]) -> f32 {
        let mut kl = 0.0f64;

        for (p_i, q_i) in p.iter().zip(q.iter()) {
            let p_val = (*p_i as f64).max(self.epsilon);
            let q_val = (*q_i as f64).max(self.epsilon);
            kl += p_val * (p_val / q_val).ln();
        }

        kl.max(0.0) as f32
    }

    /// Compute symmetric KL divergence.
    fn compute_symmetric_kl(&self, p: &[f32], q: &[f32]) -> f32 {
        let kl_pq = self.compute_asymmetric_kl(p, q);
        let kl_qp = self.compute_asymmetric_kl(q, p);
        0.5 * (kl_pq + kl_qp)
    }

    /// Get the current epsilon value.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Check if symmetric mode is enabled.
    pub fn is_symmetric(&self) -> bool {
        self.symmetric
    }

    /// Get the maximum value for clamping.
    pub fn max_value(&self) -> f64 {
        self.max_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kl_divergence_identical() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let kl = compute_kl_divergence(&p, &p, 1e-10);
        assert!(
            kl.abs() < 1e-6,
            "KL of identical distributions should be ~0"
        );
    }

    #[test]
    fn test_kl_divergence_different() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.1, 0.2, 0.3, 0.4];
        let kl = compute_kl_divergence(&p, &q, 1e-10);
        assert!(kl > 0.0, "KL of different distributions should be positive");
    }

    #[test]
    fn test_kl_divergence_empty() {
        let empty: Vec<f32> = vec![];
        let p = vec![0.5, 0.5];
        assert_eq!(compute_kl_divergence(&empty, &p, 1e-10), 0.0);
        assert_eq!(compute_kl_divergence(&p, &empty, 1e-10), 0.0);
    }

    #[test]
    fn test_kl_divergence_mismatched_lengths() {
        let p = vec![0.5, 0.5];
        let q = vec![0.33, 0.33, 0.34];
        assert_eq!(compute_kl_divergence(&p, &q, 1e-10), 0.0);
    }

    #[test]
    fn test_kl_divergence_uniform() {
        // Uniform distribution has maximum entropy
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let peaked = vec![0.9, 0.033, 0.033, 0.034];

        let kl_uniform_to_peaked = compute_kl_divergence(&uniform, &peaked, 1e-10);
        let kl_peaked_to_uniform = compute_kl_divergence(&peaked, &uniform, 1e-10);

        // Both should be positive but different (asymmetry)
        assert!(kl_uniform_to_peaked > 0.0);
        assert!(kl_peaked_to_uniform > 0.0);
    }

    #[test]
    fn test_kl_calculator_default() {
        let calc = KlDivergenceCalculator::default();
        assert!(!calc.is_symmetric());
        assert!(calc.epsilon() > 0.0);
    }

    #[test]
    fn test_kl_calculator_compute() {
        let calc = KlDivergenceCalculator::default();
        let p = vec![0.5, 0.5];
        let q = vec![0.25, 0.75];

        let result = calc.compute(&p, &q);
        assert!(result.is_ok());
        assert!(result.unwrap() >= 0.0);
    }

    #[test]
    fn test_kl_calculator_empty_input() {
        let calc = KlDivergenceCalculator::default();
        let empty: Vec<f32> = vec![];
        let p = vec![0.5, 0.5];

        assert!(matches!(
            calc.compute(&empty, &p),
            Err(UtlError::EmptyInput)
        ));
        assert!(matches!(
            calc.compute(&p, &empty),
            Err(UtlError::EmptyInput)
        ));
    }

    #[test]
    fn test_kl_calculator_dimension_mismatch() {
        let calc = KlDivergenceCalculator::default();
        let p = vec![0.5, 0.5];
        let q = vec![0.33, 0.33, 0.34];

        let result = calc.compute(&p, &q);
        assert!(matches!(result, Err(UtlError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_kl_calculator_symmetric() {
        let calc = KlDivergenceCalculator::new(1e-8, true, 100.0);
        assert!(calc.is_symmetric());

        let p = vec![0.5, 0.5];
        let q = vec![0.25, 0.75];

        let kl_pq = calc.compute(&p, &q).unwrap();
        let kl_qp = calc.compute(&q, &p).unwrap();

        // Symmetric KL should be the same in both directions
        assert!((kl_pq - kl_qp).abs() < 1e-6);
    }

    #[test]
    fn test_kl_calculator_normalized() {
        let calc = KlDivergenceCalculator::default();
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.1, 0.2, 0.3, 0.4];

        let normalized = calc.compute_normalized(&p, &q).unwrap();
        assert!(normalized >= 0.0 && normalized <= 1.0);
    }

    #[test]
    fn test_kl_calculator_with_smoothing() {
        let calc = KlDivergenceCalculator::default().with_smoothing(0.1);
        let p = vec![1.0, 0.0]; // Would cause log(0) without smoothing
        let q = vec![0.0, 1.0];

        let result = calc.compute(&p, &q);
        assert!(result.is_ok());
        assert!(result.unwrap() >= 0.0);
    }

    #[test]
    fn test_kl_calculator_clamping() {
        let calc = KlDivergenceCalculator::new(1e-15, false, 10.0);

        // Create distributions that would produce large KL
        let p = vec![0.99, 0.01];
        let q = vec![0.01, 0.99];

        let result = calc.compute(&p, &q).unwrap();
        assert!(result <= 10.0, "Result should be clamped to max_value");
    }

    #[test]
    fn test_no_nan_infinity() {
        // Test edge cases that might produce NaN or Infinity
        let calc = KlDivergenceCalculator::default();

        // Zero probabilities (should be handled by epsilon)
        let p = vec![0.0, 1.0];
        let q = vec![1.0, 0.0];
        let result = calc.compute(&p, &q).unwrap();
        assert!(!result.is_nan());
        assert!(!result.is_infinite());

        // Very small probabilities
        let p2 = vec![1e-10, 1.0 - 1e-10];
        let q2 = vec![1.0 - 1e-10, 1e-10];
        let result2 = calc.compute(&p2, &q2).unwrap();
        assert!(!result2.is_nan());
        assert!(!result2.is_infinite());
    }

    #[test]
    fn test_kl_non_negativity() {
        let calc = KlDivergenceCalculator::default();

        // KL divergence should always be non-negative
        for _ in 0..10 {
            let p = vec![0.1, 0.2, 0.3, 0.4];
            let q = vec![0.4, 0.3, 0.2, 0.1];
            let kl = calc.compute(&p, &q).unwrap();
            assert!(kl >= 0.0, "KL divergence must be non-negative");
        }
    }
}
