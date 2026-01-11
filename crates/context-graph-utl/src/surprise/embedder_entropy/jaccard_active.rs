//! Jaccard similarity of active dimensions for E13 (SPLADE) embeddings.
//!
//! Formula: ΔS = 1 - jaccard(active_dims(current), active_dims(history))
//! For sparse vectors where most dimensions are zero.
//!
//! # Constitution Reference
//!
//! From constitution.yaml delta_sc.ΔS_methods:
//! E6,E13: "IDF/Jaccard: ΔS=IDF(dims) or 1-jaccard"

use super::EmbedderEntropy;
use crate::error::{UtlError, UtlResult};
use context_graph_core::teleological::Embedder;
use std::collections::HashSet;

/// E13 (SPLADE) entropy using Jaccard similarity of active dimensions.
///
/// Sparse vectors (like SPLADE) have most dimensions at zero. This calculator
/// computes surprise based on the overlap of active (non-zero) dimensions
/// between the current embedding and history.
///
/// # Algorithm
///
/// 1. Extract active dimensions (values > threshold) from current embedding
/// 2. For each history item, compute Jaccard similarity of active dimensions
/// 3. Average the top-k Jaccard similarities
/// 4. ΔS = 1 - avg_jaccard, clamped to [0, 1]
#[derive(Debug, Clone)]
pub struct JaccardActiveEntropy {
    /// Activation threshold for considering a dimension "active".
    /// Default: 0.0 (any non-zero value is active)
    activation_threshold: f32,
    /// Smoothing factor for empty union case.
    smoothing: f32,
}

impl Default for JaccardActiveEntropy {
    fn default() -> Self {
        Self::new()
    }
}

impl JaccardActiveEntropy {
    /// Create a new Jaccard active entropy calculator.
    pub fn new() -> Self {
        Self {
            activation_threshold: 0.0,
            smoothing: 0.01,
        }
    }

    /// Set the activation threshold.
    ///
    /// # Arguments
    /// * `threshold` - Minimum value for a dimension to be considered active (range: 0.0 to 0.1)
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.activation_threshold = threshold.clamp(0.0, 0.1);
        self
    }

    /// Set the smoothing factor for empty union handling.
    ///
    /// # Arguments
    /// * `smoothing` - Smoothing value (range: 0.001 to 0.1)
    pub fn with_smoothing(mut self, smoothing: f32) -> Self {
        self.smoothing = smoothing.clamp(0.001, 0.1);
        self
    }

    /// Extract active dimension indices from embedding.
    fn get_active_dims(&self, embedding: &[f32]) -> HashSet<usize> {
        embedding
            .iter()
            .enumerate()
            .filter(|(_, &v)| v.abs() > self.activation_threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Compute Jaccard similarity between two sets of active dimensions.
    fn jaccard_similarity(&self, a: &HashSet<usize>, b: &HashSet<usize>) -> f32 {
        if a.is_empty() && b.is_empty() {
            // Both empty = consider them identical (low surprise)
            return 1.0 - self.smoothing;
        }

        let intersection = a.intersection(b).count();
        let union = a.union(b).count();

        if union == 0 {
            return self.smoothing;
        }

        intersection as f32 / union as f32
    }
}

impl EmbedderEntropy for JaccardActiveEntropy {
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

        // Get active dimensions for current embedding
        let active_current = self.get_active_dims(current);

        // Compute Jaccard similarity to each history item
        let mut jaccard_scores: Vec<f32> = Vec::with_capacity(history.len());

        for h in history {
            if h.is_empty() {
                continue;
            }

            // Validate history item
            let mut has_invalid = false;
            for &v in h {
                if v.is_nan() || v.is_infinite() {
                    has_invalid = true;
                    break;
                }
            }
            if has_invalid {
                continue;
            }

            let active_h = self.get_active_dims(h);
            let jaccard = self.jaccard_similarity(&active_current, &active_h);
            jaccard_scores.push(jaccard);
        }

        if jaccard_scores.is_empty() {
            return Ok(1.0);
        }

        // Sort descending (best matches first)
        jaccard_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Average top-k scores
        let k_actual = k.min(jaccard_scores.len()).max(1);
        let avg_jaccard: f32 = jaccard_scores[..k_actual].iter().sum::<f32>() / k_actual as f32;

        // ΔS = 1 - jaccard
        let delta_s = 1.0 - avg_jaccard;

        // Clamp per AP-10
        Ok(delta_s.clamp(0.0, 1.0))
    }

    fn embedder_type(&self) -> Embedder {
        Embedder::KeywordSplade
    }

    fn reset(&mut self) {
        // No persistent state to reset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaccard_identical_sparse_returns_zero() {
        let calculator = JaccardActiveEntropy::new();

        // Create sparse vector with specific active dimensions
        let mut sparse = vec![0.0f32; 1000];
        sparse[10] = 0.5;
        sparse[50] = 0.8;
        sparse[100] = 0.3;

        let history = vec![sparse.clone()];

        let result = calculator.compute_delta_s(&sparse, &history, 5);
        assert!(result.is_ok());
        let delta_s = result.unwrap();

        assert!(
            delta_s < 0.05,
            "Identical sparse vectors should have near-zero surprise, got {}",
            delta_s
        );

        println!("[PASS] jaccard_identical_sparse_returns_zero");
    }

    #[test]
    fn test_jaccard_disjoint_returns_one() {
        let calculator = JaccardActiveEntropy::new();

        // Create two sparse vectors with completely different active dims
        let mut sparse1 = vec![0.0f32; 100];
        sparse1[0] = 0.5;
        sparse1[1] = 0.5;
        sparse1[2] = 0.5;

        let mut sparse2 = vec![0.0f32; 100];
        sparse2[50] = 0.5;
        sparse2[51] = 0.5;
        sparse2[52] = 0.5;

        let history = vec![sparse2];

        let result = calculator.compute_delta_s(&sparse1, &history, 5);
        assert!(result.is_ok());
        let delta_s = result.unwrap();

        assert!(
            delta_s > 0.95,
            "Disjoint sparse vectors should have high surprise, got {}",
            delta_s
        );

        println!("[PASS] jaccard_disjoint_returns_one");
    }

    #[test]
    fn test_jaccard_handles_empty_union() {
        let calculator = JaccardActiveEntropy::new();

        // All zeros = no active dimensions
        let all_zeros = vec![0.0f32; 100];
        let history = vec![vec![0.0f32; 100]];

        let result = calculator.compute_delta_s(&all_zeros, &history, 5);
        assert!(result.is_ok());
        let delta_s = result.unwrap();

        // With smoothing, should return a small value (not 1.0)
        assert!(
            (0.0..=1.0).contains(&delta_s),
            "delta_s out of range: {}",
            delta_s
        );
        assert!(
            delta_s < 0.1,
            "Empty vectors treated as similar should have low surprise"
        );

        println!("[PASS] jaccard_handles_empty_union");
    }

    #[test]
    fn test_jaccard_partial_overlap() {
        let calculator = JaccardActiveEntropy::new();

        // 50% overlap
        let mut current = vec![0.0f32; 100];
        current[0] = 0.5;
        current[1] = 0.5;
        current[2] = 0.5;
        current[3] = 0.5;

        let mut history_item = vec![0.0f32; 100];
        history_item[0] = 0.5;
        history_item[1] = 0.5;
        history_item[10] = 0.5;
        history_item[11] = 0.5;

        let history = vec![history_item];

        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok());
        let delta_s = result.unwrap();

        // Jaccard = 2/6 = 0.333, so ΔS ≈ 0.667
        assert!(
            delta_s > 0.5 && delta_s < 0.75,
            "Partial overlap should give moderate surprise, got {}",
            delta_s
        );

        println!("[PASS] jaccard_partial_overlap");
    }

    #[test]
    fn test_jaccard_empty_history_returns_one() {
        let calculator = JaccardActiveEntropy::new();
        let current = vec![0.5f32; 100];
        let history: Vec<Vec<f32>> = vec![];

        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1.0);

        println!("[PASS] jaccard_empty_history_returns_one");
    }

    #[test]
    fn test_jaccard_empty_input_error() {
        let calculator = JaccardActiveEntropy::new();
        let empty: Vec<f32> = vec![];
        let history = vec![vec![0.5f32; 100]];

        let result = calculator.compute_delta_s(&empty, &history, 5);
        assert!(matches!(result, Err(UtlError::EmptyInput)));

        println!("[PASS] jaccard_empty_input_error");
    }

    #[test]
    fn test_jaccard_embedder_type() {
        let calculator = JaccardActiveEntropy::new();
        assert_eq!(calculator.embedder_type(), Embedder::KeywordSplade);

        println!("[PASS] jaccard_embedder_type");
    }

    #[test]
    fn test_jaccard_threshold_affects_active_dims() {
        let calc_zero = JaccardActiveEntropy::new().with_threshold(0.0);
        let calc_high = JaccardActiveEntropy::new().with_threshold(0.05);

        // Vector with small values
        let current: Vec<f32> = vec![0.01; 100];

        // With threshold 0.0, all 100 dims are active
        let active_zero = calc_zero.get_active_dims(&current);
        assert_eq!(active_zero.len(), 100);

        // With threshold 0.05, no dims are active
        let active_high = calc_high.get_active_dims(&current);
        assert_eq!(active_high.len(), 0);

        println!("[PASS] jaccard_threshold_affects_active_dims");
    }

    #[test]
    fn test_jaccard_valid_range() {
        let calculator = JaccardActiveEntropy::new();

        // Test various sparse patterns
        for active_count in [0, 1, 10, 50, 100] {
            let mut current = vec![0.0f32; 200];
            for i in 0..active_count {
                current[i] = 0.5;
            }

            let mut history_item = vec![0.0f32; 200];
            for i in (active_count / 2)..(active_count + active_count / 2) {
                if i < 200 {
                    history_item[i] = 0.5;
                }
            }

            let history = vec![history_item];
            let result = calculator.compute_delta_s(&current, &history, 5);

            assert!(result.is_ok());
            let delta_s = result.unwrap();
            assert!(
                (0.0..=1.0).contains(&delta_s),
                "delta_s {} out of range for active_count {}",
                delta_s,
                active_count
            );
        }

        println!("[PASS] jaccard_valid_range");
    }

    #[test]
    fn test_jaccard_no_nan_infinity() {
        let calculator = JaccardActiveEntropy::new();

        // Very small values
        let small: Vec<f32> = vec![1e-10; 100];
        let history: Vec<Vec<f32>> = vec![vec![1e-10; 100]; 5];

        let result = calculator.compute_delta_s(&small, &history, 5);
        assert!(result.is_ok());
        let delta_s = result.unwrap();
        assert!(!delta_s.is_nan());
        assert!(!delta_s.is_infinite());

        println!("[PASS] jaccard_no_nan_infinity");
    }
}
