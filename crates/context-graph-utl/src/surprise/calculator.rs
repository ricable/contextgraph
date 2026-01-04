//! Main surprise calculator combining KL divergence and embedding distance.
//!
//! This module provides the primary `SurpriseCalculator` struct that computes
//! overall surprise (Delta-S) by combining multiple signals:
//! - KL divergence for distribution comparison
//! - Embedding distance for semantic novelty
//!
//! # Constitution Reference
//!
//! The surprise component (Delta-S) is part of the UTL formula:
//! `L = f((Delta-S x Delta-C) * w_e * cos(phi))`
//!
//! Where Delta-S represents entropy/novelty in range [0, 1].
//!
//! # Numerical Stability
//!
//! Per AP-009, all outputs are clamped to valid ranges with no NaN or Infinity values.

use crate::config::{KlConfig, SurpriseConfig};
use crate::error::{UtlError, UtlResult};

use super::embedding_distance::EmbeddingDistanceCalculator;
use super::kl_divergence::KlDivergenceCalculator;

/// Main calculator for computing surprise (Delta-S) values.
///
/// Combines multiple signals to compute a unified surprise score:
/// - Embedding-based surprise: measures semantic novelty
/// - KL divergence: measures distribution change
///
/// The final surprise score is a weighted combination of these signals,
/// clamped to the valid [0, 1] range per AP-009.
///
/// # Example
///
/// ```
/// use context_graph_utl::config::SurpriseConfig;
/// use context_graph_utl::surprise::SurpriseCalculator;
///
/// let config = SurpriseConfig::default();
/// let calculator = SurpriseCalculator::new(&config);
///
/// let current = vec![0.1, 0.2, 0.3, 0.4];
/// let history = vec![
///     vec![0.15, 0.25, 0.35, 0.25],
///     vec![0.12, 0.22, 0.32, 0.34],
/// ];
///
/// let surprise = calculator.compute_surprise(&current, &history);
/// assert!(surprise >= 0.0 && surprise <= 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct SurpriseCalculator {
    /// Weight for entropy-based surprise component.
    entropy_weight: f32,
    /// Boost factor for novel items.
    novelty_boost: f32,
    /// Decay rate for repeated exposure.
    repetition_decay: f32,
    /// Minimum surprise threshold.
    min_threshold: f32,
    /// Maximum surprise value.
    max_value: f32,
    /// Whether to use EMA smoothing.
    use_ema: bool,
    /// EMA alpha (smoothing factor).
    ema_alpha: f32,
    /// Embedding distance calculator.
    embedding_calc: EmbeddingDistanceCalculator,
    /// KL divergence calculator.
    kl_calc: KlDivergenceCalculator,
    /// Current EMA state for smoothing.
    ema_state: Option<f32>,
}

impl SurpriseCalculator {
    /// Create a new SurpriseCalculator from configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The surprise configuration settings
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::config::SurpriseConfig;
    /// use context_graph_utl::surprise::SurpriseCalculator;
    ///
    /// let config = SurpriseConfig::default();
    /// let calculator = SurpriseCalculator::new(&config);
    /// ```
    pub fn new(config: &SurpriseConfig) -> Self {
        Self {
            entropy_weight: config.entropy_weight,
            novelty_boost: config.novelty_boost,
            repetition_decay: config.repetition_decay,
            min_threshold: config.min_threshold,
            max_value: config.max_value,
            use_ema: config.use_ema,
            ema_alpha: config.ema_alpha,
            embedding_calc: EmbeddingDistanceCalculator::from_config(config),
            kl_calc: KlDivergenceCalculator::default(),
            ema_state: None,
        }
    }

    /// Create a calculator with custom KL divergence settings.
    ///
    /// # Arguments
    ///
    /// * `config` - The surprise configuration settings
    /// * `kl_config` - The KL divergence configuration
    pub fn with_kl_config(config: &SurpriseConfig, kl_config: &KlConfig) -> Self {
        Self {
            entropy_weight: config.entropy_weight,
            novelty_boost: config.novelty_boost,
            repetition_decay: config.repetition_decay,
            min_threshold: config.min_threshold,
            max_value: config.max_value,
            use_ema: config.use_ema,
            ema_alpha: config.ema_alpha,
            embedding_calc: EmbeddingDistanceCalculator::from_config(config),
            kl_calc: KlDivergenceCalculator::from_config(config, kl_config),
            ema_state: None,
        }
    }

    /// Compute overall surprise from current embedding and history.
    ///
    /// This is the main entry point for surprise computation. It combines
    /// embedding distance with optional KL divergence to produce a unified
    /// surprise score.
    ///
    /// # Arguments
    ///
    /// * `current` - The current embedding vector
    /// * `history` - Historical embedding vectors (most recent first)
    ///
    /// # Returns
    ///
    /// Surprise value in [0, 1]. Higher values indicate more novelty.
    /// Returns 1.0 for empty history (maximum surprise for first item).
    /// Returns 0.0 for empty current embedding or errors.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::config::SurpriseConfig;
    /// use context_graph_utl::surprise::SurpriseCalculator;
    ///
    /// let config = SurpriseConfig::default();
    /// let calculator = SurpriseCalculator::new(&config);
    ///
    /// let current = vec![0.1, 0.2, 0.3, 0.4];
    /// let history = vec![vec![0.15, 0.25, 0.35, 0.25]];
    ///
    /// let surprise = calculator.compute_surprise(&current, &history);
    /// assert!(surprise >= 0.0 && surprise <= 1.0);
    /// ```
    pub fn compute_surprise(&self, current: &[f32], history: &[Vec<f32>]) -> f32 {
        // Handle empty current embedding
        if current.is_empty() {
            return 0.0;
        }

        // Handle empty history (maximum surprise)
        if history.is_empty() {
            return self.clamp_result(1.0);
        }

        // Compute embedding-based surprise
        let embedding_surprise = match self.embedding_calc.compute_surprise(current, history) {
            Ok(s) => s,
            Err(_) => return 0.0,
        };

        // Combine with entropy weight
        let raw_surprise = embedding_surprise * self.entropy_weight;

        // Apply novelty boost for high surprise
        let boosted = if raw_surprise > 0.5 {
            raw_surprise * self.novelty_boost
        } else {
            raw_surprise
        };

        // Apply minimum threshold
        let thresholded = if boosted < self.min_threshold {
            0.0
        } else {
            boosted
        };

        self.clamp_result(thresholded)
    }

    /// Compute surprise with optional result returning.
    ///
    /// Unlike `compute_surprise`, this method returns a Result to allow
    /// proper error handling when validation fails.
    ///
    /// # Arguments
    ///
    /// * `current` - The current embedding vector
    /// * `history` - Historical embedding vectors
    ///
    /// # Returns
    ///
    /// Result containing surprise value or error.
    ///
    /// # Errors
    ///
    /// Returns `UtlError::EmptyInput` if current embedding is empty.
    pub fn compute_surprise_checked(
        &self,
        current: &[f32],
        history: &[Vec<f32>],
    ) -> UtlResult<f32> {
        if current.is_empty() {
            return Err(UtlError::EmptyInput);
        }

        if history.is_empty() {
            return Ok(self.clamp_result(1.0));
        }

        let embedding_surprise = self.embedding_calc.compute_surprise(current, history)?;
        let raw_surprise = embedding_surprise * self.entropy_weight;

        let boosted = if raw_surprise > 0.5 {
            raw_surprise * self.novelty_boost
        } else {
            raw_surprise
        };

        let thresholded = if boosted < self.min_threshold {
            0.0
        } else {
            boosted
        };

        Ok(self.clamp_result(thresholded))
    }

    /// Compute surprise with EMA smoothing.
    ///
    /// This method maintains state across calls to smooth the surprise signal
    /// using exponential moving average.
    ///
    /// # Arguments
    ///
    /// * `current` - The current embedding vector
    /// * `history` - Historical embedding vectors
    ///
    /// # Returns
    ///
    /// Smoothed surprise value in [0, 1].
    pub fn compute_surprise_smoothed(&mut self, current: &[f32], history: &[Vec<f32>]) -> f32 {
        let raw = self.compute_surprise(current, history);

        if !self.use_ema {
            return raw;
        }

        let smoothed = match self.ema_state {
            Some(prev) => self.ema_alpha * raw + (1.0 - self.ema_alpha) * prev,
            None => raw,
        };

        self.ema_state = Some(smoothed);
        self.clamp_result(smoothed)
    }

    /// Compute surprise using KL divergence between distributions.
    ///
    /// This method is useful when comparing probability distributions
    /// rather than embedding vectors.
    ///
    /// # Arguments
    ///
    /// * `current_dist` - Current probability distribution
    /// * `reference_dist` - Reference probability distribution
    ///
    /// # Returns
    ///
    /// Normalized surprise value in [0, 1].
    pub fn compute_kl_surprise(
        &self,
        current_dist: &[f32],
        reference_dist: &[f32],
    ) -> UtlResult<f32> {
        let kl = self
            .kl_calc
            .compute_normalized(current_dist, reference_dist)?;
        Ok(self.clamp_result(kl * self.entropy_weight))
    }

    /// Compute combined surprise from both embedding and distribution signals.
    ///
    /// This method provides the most comprehensive surprise measurement by
    /// combining semantic novelty (embeddings) with distributional change (KL).
    ///
    /// # Arguments
    ///
    /// * `current_embedding` - Current embedding vector
    /// * `embedding_history` - Historical embeddings
    /// * `current_dist` - Current probability distribution (optional)
    /// * `reference_dist` - Reference probability distribution (optional)
    ///
    /// # Returns
    ///
    /// Combined surprise value in [0, 1].
    pub fn compute_combined_surprise(
        &self,
        current_embedding: &[f32],
        embedding_history: &[Vec<f32>],
        current_dist: Option<&[f32]>,
        reference_dist: Option<&[f32]>,
    ) -> f32 {
        let embedding_surprise = self.compute_surprise(current_embedding, embedding_history);

        // If distributions provided, combine with KL surprise
        let combined = match (current_dist, reference_dist) {
            (Some(curr), Some(ref_)) => {
                let kl_surprise = self.compute_kl_surprise(curr, ref_).unwrap_or(0.0);
                // Weighted combination (entropy_weight for KL, 1-entropy_weight for embedding)
                self.entropy_weight * kl_surprise + (1.0 - self.entropy_weight) * embedding_surprise
            }
            _ => embedding_surprise,
        };

        self.clamp_result(combined)
    }

    /// Apply repetition decay based on occurrence count.
    ///
    /// Reduces surprise for items that have been seen multiple times.
    ///
    /// # Arguments
    ///
    /// * `base_surprise` - The base surprise value
    /// * `repetition_count` - Number of times this item has been seen
    ///
    /// # Returns
    ///
    /// Decayed surprise value.
    pub fn apply_repetition_decay(&self, base_surprise: f32, repetition_count: u32) -> f32 {
        if repetition_count == 0 {
            return self.clamp_result(base_surprise);
        }

        // Exponential decay: surprise * (1 - decay)^count
        let decay_factor = (1.0 - self.repetition_decay).powi(repetition_count as i32);
        self.clamp_result(base_surprise * decay_factor)
    }

    /// Reset the EMA state for smoothing.
    pub fn reset_ema(&mut self) {
        self.ema_state = None;
    }

    /// Get the current EMA state.
    pub fn ema_state(&self) -> Option<f32> {
        self.ema_state
    }

    /// Clamp result to valid range per AP-009.
    fn clamp_result(&self, value: f32) -> f32 {
        if value.is_nan() {
            0.0
        } else if value.is_infinite() {
            if value.is_sign_positive() {
                self.max_value
            } else {
                0.0
            }
        } else {
            value.clamp(0.0, self.max_value)
        }
    }

    /// Get the entropy weight.
    pub fn entropy_weight(&self) -> f32 {
        self.entropy_weight
    }

    /// Get the novelty boost factor.
    pub fn novelty_boost(&self) -> f32 {
        self.novelty_boost
    }

    /// Get the repetition decay rate.
    pub fn repetition_decay(&self) -> f32 {
        self.repetition_decay
    }

    /// Get the minimum threshold.
    pub fn min_threshold(&self) -> f32 {
        self.min_threshold
    }

    /// Get the maximum value.
    pub fn max_value(&self) -> f32 {
        self.max_value
    }
}

impl Default for SurpriseCalculator {
    fn default() -> Self {
        Self::new(&SurpriseConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_calculator() -> SurpriseCalculator {
        SurpriseCalculator::new(&SurpriseConfig::default())
    }

    #[test]
    fn test_calculator_creation() {
        let config = SurpriseConfig::default();
        let calc = SurpriseCalculator::new(&config);

        assert_eq!(calc.entropy_weight(), config.entropy_weight);
        assert_eq!(calc.novelty_boost(), config.novelty_boost);
        assert_eq!(calc.min_threshold(), config.min_threshold);
    }

    #[test]
    fn test_calculator_default() {
        let calc = SurpriseCalculator::default();
        assert!(calc.entropy_weight() > 0.0);
        assert!(calc.max_value() <= 1.0);
    }

    #[test]
    fn test_compute_surprise_empty_current() {
        let calc = create_calculator();
        let empty: Vec<f32> = vec![];
        let history = vec![vec![0.1, 0.2]];

        let surprise = calc.compute_surprise(&empty, &history);
        assert_eq!(surprise, 0.0);
    }

    #[test]
    fn test_compute_surprise_empty_history() {
        let calc = create_calculator();
        let current = vec![0.1, 0.2, 0.3];
        let history: Vec<Vec<f32>> = vec![];

        let surprise = calc.compute_surprise(&current, &history);
        assert_eq!(surprise, 1.0, "Empty history should give maximum surprise");
    }

    #[test]
    fn test_compute_surprise_identical() {
        let calc = create_calculator();
        let current = vec![0.1, 0.2, 0.3, 0.4];
        let history = vec![vec![0.1, 0.2, 0.3, 0.4]];

        let surprise = calc.compute_surprise(&current, &history);
        assert!(
            surprise < 0.1,
            "Identical embedding should have low surprise"
        );
    }

    #[test]
    fn test_compute_surprise_different() {
        let calc = create_calculator();
        let current = vec![0.9, 0.05, 0.03, 0.02];
        let history = vec![vec![0.1, 0.2, 0.3, 0.4]];

        let surprise = calc.compute_surprise(&current, &history);
        assert!(
            surprise > 0.0,
            "Different embeddings should have positive surprise"
        );
        assert!(surprise <= 1.0, "Surprise should be at most 1.0");
    }

    #[test]
    fn test_compute_surprise_range() {
        let calc = create_calculator();
        let current = vec![0.5, 0.3, 0.15, 0.05];
        let history = vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.25, 0.25, 0.25, 0.25]];

        let surprise = calc.compute_surprise(&current, &history);
        assert!(
            surprise >= 0.0 && surprise <= 1.0,
            "Surprise should be in [0, 1]"
        );
    }

    #[test]
    fn test_compute_surprise_checked_error() {
        let calc = create_calculator();
        let empty: Vec<f32> = vec![];
        let history = vec![vec![0.1, 0.2]];

        let result = calc.compute_surprise_checked(&empty, &history);
        assert!(matches!(result, Err(UtlError::EmptyInput)));
    }

    #[test]
    fn test_compute_surprise_checked_success() {
        let calc = create_calculator();
        let current = vec![0.1, 0.2, 0.3];
        let history = vec![vec![0.15, 0.25, 0.35]];

        let result = calc.compute_surprise_checked(&current, &history);
        assert!(result.is_ok());
        let surprise = result.unwrap();
        assert!(surprise >= 0.0 && surprise <= 1.0);
    }

    #[test]
    fn test_compute_surprise_smoothed() {
        let mut calc = create_calculator();
        let current = vec![0.1, 0.2, 0.3, 0.4];
        let history = vec![vec![0.5, 0.3, 0.15, 0.05]];

        // First call sets EMA state
        let first = calc.compute_surprise_smoothed(&current, &history);
        assert!(calc.ema_state().is_some());

        // Second call should be smoothed
        let second = calc.compute_surprise_smoothed(&current, &history);
        assert!(second >= 0.0 && second <= 1.0);

        // Reset EMA
        calc.reset_ema();
        assert!(calc.ema_state().is_none());
    }

    #[test]
    fn test_compute_kl_surprise() {
        let calc = create_calculator();
        let current = vec![0.25, 0.25, 0.25, 0.25];
        let reference = vec![0.1, 0.2, 0.3, 0.4];

        let result = calc.compute_kl_surprise(&current, &reference);
        assert!(result.is_ok());
        let surprise = result.unwrap();
        assert!(surprise >= 0.0 && surprise <= 1.0);
    }

    #[test]
    fn test_compute_combined_surprise() {
        let calc = create_calculator();
        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        let history = vec![vec![0.15, 0.25, 0.35, 0.25]];
        let dist = vec![0.25, 0.25, 0.25, 0.25];
        let ref_dist = vec![0.1, 0.2, 0.3, 0.4];

        // Without distributions
        let surprise1 = calc.compute_combined_surprise(&embedding, &history, None, None);
        assert!(surprise1 >= 0.0 && surprise1 <= 1.0);

        // With distributions
        let surprise2 =
            calc.compute_combined_surprise(&embedding, &history, Some(&dist), Some(&ref_dist));
        assert!(surprise2 >= 0.0 && surprise2 <= 1.0);
    }

    #[test]
    fn test_repetition_decay() {
        let calc = create_calculator();
        let base = 0.8;

        // No repetitions
        let no_decay = calc.apply_repetition_decay(base, 0);
        assert!((no_decay - base).abs() < 1e-6);

        // With repetitions
        let decayed = calc.apply_repetition_decay(base, 5);
        assert!(decayed < base, "Repeated items should have lower surprise");
        assert!(decayed >= 0.0);
    }

    #[test]
    fn test_no_nan_infinity() {
        let calc = create_calculator();

        // Test with edge case inputs
        let zero = vec![0.0, 0.0, 0.0];
        let normal = vec![0.1, 0.2, 0.7];
        let history = vec![normal.clone()];

        let surprise = calc.compute_surprise(&zero, &history);
        assert!(!surprise.is_nan(), "Should not produce NaN");
        assert!(!surprise.is_infinite(), "Should not produce Infinity");

        // Test clamping
        let result = calc.clamp_result(f32::NAN);
        assert_eq!(result, 0.0);

        let result = calc.clamp_result(f32::INFINITY);
        assert_eq!(result, calc.max_value());

        let result = calc.clamp_result(f32::NEG_INFINITY);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_min_threshold() {
        // Create config with higher min threshold
        let config = SurpriseConfig {
            min_threshold: 0.3,
            ..Default::default()
        };
        let calc = SurpriseCalculator::new(&config);

        // Very similar embeddings should produce below-threshold surprise
        let current = vec![0.25, 0.25, 0.25, 0.25];
        let history = vec![vec![0.24, 0.26, 0.25, 0.25]];

        let surprise = calc.compute_surprise(&current, &history);
        // Either 0 (below threshold) or >= min_threshold
        assert!(surprise == 0.0 || surprise >= 0.3);
    }

    #[test]
    fn test_max_value_clamping() {
        let config = SurpriseConfig {
            max_value: 0.8,
            novelty_boost: 2.0,
            ..Default::default()
        };
        let calc = SurpriseCalculator::new(&config);

        let current = vec![1.0, 0.0, 0.0];
        let history = vec![vec![0.0, 1.0, 0.0]];

        let surprise = calc.compute_surprise(&current, &history);
        assert!(surprise <= 0.8, "Surprise should be clamped to max_value");
    }

    #[test]
    fn test_with_kl_config() {
        let config = SurpriseConfig::default();
        let kl_config = KlConfig {
            symmetric: true,
            ..Default::default()
        };

        let calc = SurpriseCalculator::with_kl_config(&config, &kl_config);

        let dist1 = vec![0.5, 0.5];
        let dist2 = vec![0.25, 0.75];

        let kl = calc.compute_kl_surprise(&dist1, &dist2);
        assert!(kl.is_ok());
    }
}
