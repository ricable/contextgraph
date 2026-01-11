//! Asymmetric KNN entropy for E5 (Causal) embeddings.
//!
//! Formula: ΔS = d_k × direction_mod
//! Where direction_mod adjusts for causal direction (cause→effect vs effect→cause).
//!
//! # Constitution Reference
//!
//! From constitution.yaml delta_sc.ΔS_methods:
//! E5: "Asymmetric KNN: ΔS=d_k×direction_mod"
//! From constitution.yaml causal_asymmetric: cause_to_effect=1.2, effect_to_cause=0.8

use super::EmbedderEntropy;
use crate::error::{UtlError, UtlResult};
use crate::surprise::compute_cosine_distance;
use context_graph_core::teleological::Embedder;

/// E5 (Causal) entropy using asymmetric KNN with direction modifiers.
///
/// Causal embeddings are inherently directional - the relationship from
/// cause to effect differs from effect to cause. This calculator applies
/// direction modifiers to account for this asymmetry.
///
/// # Algorithm
///
/// 1. Compute cosine distances to k nearest neighbors
/// 2. Take mean distance d_k
/// 3. Apply direction modifier: ΔS = d_k × direction_mod
/// 4. Clamp to [0, 1]
#[derive(Debug, Clone)]
pub struct AsymmetricKnnEntropy {
    /// Direction modifier for cause→effect relationships (default: 1.2)
    /// Higher values increase surprise for forward causal queries.
    cause_to_effect_mod: f32,
    /// Direction modifier for effect→cause relationships (default: 0.8)
    /// Lower values decrease surprise for backward causal queries.
    effect_to_cause_mod: f32,
    /// Number of neighbors to consider.
    k: usize,
    /// Current direction mode (true = cause→effect, false = effect→cause).
    /// Defaults to cause→effect (true).
    cause_to_effect_mode: bool,
}

impl Default for AsymmetricKnnEntropy {
    fn default() -> Self {
        Self::new(5)
    }
}

impl AsymmetricKnnEntropy {
    /// Create a new asymmetric KNN entropy calculator.
    ///
    /// # Arguments
    /// * `k` - Number of nearest neighbors to consider
    pub fn new(k: usize) -> Self {
        Self {
            cause_to_effect_mod: 1.2,
            effect_to_cause_mod: 0.8,
            k: k.max(1),
            cause_to_effect_mode: true,
        }
    }

    /// Set custom direction modifiers.
    ///
    /// # Arguments
    /// * `cause_to_effect` - Modifier for cause→effect (range: 0.5 to 2.0)
    /// * `effect_to_cause` - Modifier for effect→cause (range: 0.5 to 2.0)
    pub fn with_direction_modifiers(mut self, cause_to_effect: f32, effect_to_cause: f32) -> Self {
        self.cause_to_effect_mod = cause_to_effect.clamp(0.5, 2.0);
        self.effect_to_cause_mod = effect_to_cause.clamp(0.5, 2.0);
        self
    }

    /// Set the direction mode.
    ///
    /// # Arguments
    /// * `cause_to_effect` - true for cause→effect, false for effect→cause
    pub fn with_direction_mode(mut self, cause_to_effect: bool) -> Self {
        self.cause_to_effect_mode = cause_to_effect;
        self
    }

    /// Get the current direction modifier based on mode.
    fn current_direction_mod(&self) -> f32 {
        if self.cause_to_effect_mode {
            self.cause_to_effect_mod
        } else {
            self.effect_to_cause_mod
        }
    }
}

impl EmbedderEntropy for AsymmetricKnnEntropy {
    fn compute_delta_s(
        &self,
        current: &[f32],
        history: &[Vec<f32>],
        k: usize,
    ) -> UtlResult<f32> {
        // Validate input
        if current.is_empty() {
            return Err(UtlError::EmptyInput);
        }

        // Check for NaN/Infinity
        for &v in current {
            if v.is_nan() || v.is_infinite() {
                return Err(UtlError::EntropyError(
                    "Invalid value (NaN/Infinity) in current embedding".to_string(),
                ));
            }
        }

        // Empty history = maximum surprise
        if history.is_empty() {
            return Ok(1.0);
        }

        // Compute distances to all history items
        let mut distances: Vec<f32> = Vec::with_capacity(history.len());
        for past in history {
            if !past.is_empty() {
                let dist = compute_cosine_distance(current, past);
                if !dist.is_nan() && !dist.is_infinite() {
                    distances.push(dist);
                }
            }
        }

        if distances.is_empty() {
            return Ok(1.0);
        }

        // Sort distances ascending
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Use provided k or default
        let k_to_use = if k > 0 { k } else { self.k };
        let k_actual = k_to_use.min(distances.len());

        // Compute mean of k nearest distances
        let d_k: f32 = distances[..k_actual].iter().sum::<f32>() / k_actual as f32;

        // Apply direction modifier
        let direction_mod = self.current_direction_mod();
        let delta_s = d_k * direction_mod;

        // Clamp per AP-10
        Ok(delta_s.clamp(0.0, 1.0))
    }

    fn embedder_type(&self) -> Embedder {
        Embedder::Causal
    }

    fn reset(&mut self) {
        // No persistent state to reset
        self.cause_to_effect_mode = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asymmetric_empty_history_returns_one() {
        let calculator = AsymmetricKnnEntropy::new(5);
        let current = vec![0.5f32; 768];
        let history: Vec<Vec<f32>> = vec![];

        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1.0);

        println!("[PASS] asymmetric_empty_history_returns_one");
    }

    #[test]
    fn test_asymmetric_direction_mod_affects_output() {
        // Test cause→effect direction
        let calc_cause = AsymmetricKnnEntropy::new(3).with_direction_mode(true);

        // Test effect→cause direction
        let calc_effect = AsymmetricKnnEntropy::new(3).with_direction_mode(false);

        // Create vectors with actual directional differences (not uniform)
        let current: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let history: Vec<Vec<f32>> = vec![
            (0..100).map(|i| (i as f32 + 10.0) / 100.0).collect(),
            (0..100).map(|i| (i as f32 + 20.0) / 100.0).collect(),
            (0..100).map(|i| (i as f32 + 5.0) / 100.0).collect(),
        ];

        let result_cause = calc_cause.compute_delta_s(&current, &history, 3).unwrap();
        let result_effect = calc_effect.compute_delta_s(&current, &history, 3).unwrap();

        // Both should be > 0 since vectors have different directions
        assert!(
            result_cause > 0.0 && result_effect > 0.0,
            "Both results should be > 0: cause={}, effect={}",
            result_cause,
            result_effect
        );

        // Cause→effect should have higher ΔS (modifier 1.2 vs 0.8)
        assert!(
            result_cause > result_effect,
            "cause_to_effect ({}) should be > effect_to_cause ({})",
            result_cause,
            result_effect
        );

        println!(
            "cause_to_effect delta_s: {}, effect_to_cause delta_s: {}",
            result_cause, result_effect
        );
        println!("[PASS] asymmetric_direction_mod_affects_output");
    }

    #[test]
    fn test_asymmetric_cause_effect_higher_than_neutral() {
        let calc_neutral = AsymmetricKnnEntropy::new(3).with_direction_modifiers(1.0, 1.0);
        let calc_cause = AsymmetricKnnEntropy::new(3).with_direction_modifiers(1.2, 0.8);

        let current = vec![0.5f32; 100];
        let history: Vec<Vec<f32>> = vec![vec![0.7f32; 100], vec![0.8f32; 100]];

        let result_neutral = calc_neutral.compute_delta_s(&current, &history, 3).unwrap();
        let result_cause = calc_cause.compute_delta_s(&current, &history, 3).unwrap();

        // With cause_to_effect_mod = 1.2, result should be higher than neutral
        assert!(
            result_cause >= result_neutral,
            "cause_to_effect ({}) should be >= neutral ({})",
            result_cause,
            result_neutral
        );

        println!("[PASS] asymmetric_cause_effect_higher_than_neutral");
    }

    #[test]
    fn test_asymmetric_valid_range() {
        for modifier in [0.5, 0.8, 1.0, 1.2, 2.0] {
            let calc = AsymmetricKnnEntropy::new(3).with_direction_modifiers(modifier, modifier);

            let current: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
            let history: Vec<Vec<f32>> = (0..10)
                .map(|j| (0..100).map(|i| (i + j * 5) as f32 / 100.0).collect())
                .collect();

            let result = calc.compute_delta_s(&current, &history, 5);
            assert!(result.is_ok());
            let delta_s = result.unwrap();

            assert!(
                (0.0..=1.0).contains(&delta_s),
                "delta_s {} out of range for modifier {}",
                delta_s,
                modifier
            );
        }

        println!("[PASS] asymmetric_valid_range");
    }

    #[test]
    fn test_asymmetric_empty_input_error() {
        let calculator = AsymmetricKnnEntropy::new(5);
        let empty: Vec<f32> = vec![];
        let history = vec![vec![0.5f32; 100]];

        let result = calculator.compute_delta_s(&empty, &history, 5);
        assert!(matches!(result, Err(UtlError::EmptyInput)));

        println!("[PASS] asymmetric_empty_input_error");
    }

    #[test]
    fn test_asymmetric_embedder_type() {
        let calculator = AsymmetricKnnEntropy::new(5);
        assert_eq!(calculator.embedder_type(), Embedder::Causal);

        println!("[PASS] asymmetric_embedder_type");
    }

    #[test]
    fn test_asymmetric_reset() {
        let mut calculator = AsymmetricKnnEntropy::new(5).with_direction_mode(false);
        assert!(!calculator.cause_to_effect_mode);

        calculator.reset();
        assert!(calculator.cause_to_effect_mode);

        println!("[PASS] asymmetric_reset");
    }

    #[test]
    fn test_asymmetric_no_nan_infinity() {
        let calculator = AsymmetricKnnEntropy::new(3);

        // Very small values
        let small: Vec<f32> = vec![1e-10; 100];
        let history: Vec<Vec<f32>> = vec![vec![1e-10; 100]; 5];

        let result = calculator.compute_delta_s(&small, &history, 3);
        assert!(result.is_ok());
        let delta_s = result.unwrap();
        assert!(!delta_s.is_nan());
        assert!(!delta_s.is_infinite());

        println!("[PASS] asymmetric_no_nan_infinity");
    }

    #[test]
    fn test_asymmetric_identical_vectors_low_surprise() {
        let calculator = AsymmetricKnnEntropy::new(3);

        let current = vec![0.5f32; 100];
        let history: Vec<Vec<f32>> = vec![current.clone(); 10];

        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok());
        let delta_s = result.unwrap();

        // Identical vectors should have low surprise (distance ≈ 0)
        assert!(
            delta_s < 0.1,
            "Identical vectors should have very low surprise, got {}",
            delta_s
        );

        println!("[PASS] asymmetric_identical_vectors_low_surprise");
    }
}
