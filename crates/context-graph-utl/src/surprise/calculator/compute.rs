//! Surprise computation methods for SurpriseCalculator.
//!
//! Contains all computation methods, accessors, and utility functions.

use crate::error::{UtlError, UtlResult};

use super::types::SurpriseCalculator;

impl SurpriseCalculator {
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
    pub(crate) fn clamp_result(&self, value: f32) -> f32 {
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
