//! Hamming distance based entropy for sparse embeddings (E6).
//!
//! For sparse vectors like SPLADE, computes delta_s based on
//! symmetric difference of active dimensions (Hamming distance).
//!
//! # Constitution Reference
//!
//! From constitution.yaml delta_sc.ΔS_methods:
//! E6,E13: "IDF/Jaccard: ΔS=IDF(dims) or 1-jaccard"
//!
//! Note: E6 uses Hamming distance (symmetric difference), while E13 uses
//! Jaccard similarity (intersection/union). Both operate on sparse vectors'
//! active dimensions but with different semantics.

use std::collections::HashSet;

use context_graph_core::teleological::Embedder;

use super::EmbedderEntropy;
use crate::error::{UtlError, UtlResult};

/// Hamming distance based entropy calculator for sparse embeddings.
///
/// Computes delta_s as the normalized symmetric difference between
/// active dimensions (non-zero values) in current and historical vectors.
///
/// This is appropriate for sparse embeddings (E6) where most
/// dimensions are zero and only a few are "active".
///
/// # Algorithm
///
/// 1. Extract active dimensions (values > threshold) from embeddings
/// 2. For each history item, compute Hamming distance (symmetric difference / union)
/// 3. Find k-nearest (smallest distances)
/// 4. Average the k-nearest distances
/// 5. Clamp to [0, 1]
#[derive(Debug, Clone)]
pub struct HammingSparseEntropy {
    /// Threshold above which a dimension is considered "active"
    activation_threshold: f32,
}

impl Default for HammingSparseEntropy {
    fn default() -> Self {
        Self::new()
    }
}

impl HammingSparseEntropy {
    /// Create new HammingSparseEntropy with default threshold.
    pub fn new() -> Self {
        Self {
            activation_threshold: 0.0, // Any positive value is active
        }
    }

    /// Create with custom activation threshold.
    ///
    /// # Arguments
    /// * `threshold` - Minimum value for a dimension to be considered active
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.activation_threshold = threshold.clamp(0.0, 0.1);
        self
    }

    /// Extract active dimension indices from a sparse vector.
    fn get_active_dims(&self, vector: &[f32]) -> HashSet<usize> {
        vector
            .iter()
            .enumerate()
            .filter(|(_, &v)| v.abs() > self.activation_threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Compute Hamming distance between two sets of active dimensions.
    ///
    /// Hamming distance = |symmetric_difference| / |union|
    ///
    /// Returns normalized value in [0.0, 1.0].
    fn hamming_distance(&self, current: &HashSet<usize>, past: &HashSet<usize>) -> f32 {
        // Handle empty sets case
        if current.is_empty() && past.is_empty() {
            // Both empty = identical = 0 distance
            return 0.0;
        }

        let sym_diff = current.symmetric_difference(past).count();
        let union_size = current.union(past).count().max(1);

        sym_diff as f32 / union_size as f32
    }
}

impl EmbedderEntropy for HammingSparseEntropy {
    /// Compute delta_s using Hamming distance to k-nearest history items.
    ///
    /// FAIL FAST: Returns error for empty input.
    /// Returns 1.0 for empty history (maximum surprise).
    fn compute_delta_s(
        &self,
        current: &[f32],
        history: &[Vec<f32>],
        k: usize,
    ) -> UtlResult<f32> {
        // FAIL FAST: Empty current vector is an error
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

        // Extract active dimensions from current vector
        let current_active = self.get_active_dims(current);

        // Compute Hamming distance to each history item
        let mut distances: Vec<f32> = Vec::with_capacity(history.len());

        for past in history {
            if past.is_empty() {
                continue;
            }

            // Validate history item
            let mut has_invalid = false;
            for &v in past {
                if v.is_nan() || v.is_infinite() {
                    has_invalid = true;
                    break;
                }
            }
            if has_invalid {
                continue;
            }

            let past_active = self.get_active_dims(past);
            let dist = self.hamming_distance(&current_active, &past_active);
            distances.push(dist);
        }

        if distances.is_empty() {
            return Ok(1.0);
        }

        // Sort to find k-nearest (smallest distances)
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Average of k-nearest distances
        let k_actual = k.max(1).min(distances.len());
        let mean_dist: f32 = distances[..k_actual].iter().sum::<f32>() / k_actual as f32;

        // Clamp to valid range per AP-10
        Ok(mean_dist.clamp(0.0, 1.0))
    }

    fn embedder_type(&self) -> Embedder {
        Embedder::Sparse
    }

    fn reset(&mut self) {
        // No state to reset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_sparse_identical_vectors() {
        let calculator = HammingSparseEntropy::new();

        let current = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let history = vec![vec![0.0, 1.0, 0.0, 1.0, 0.0]]; // Identical

        let delta_s = calculator.compute_delta_s(&current, &history, 1).unwrap();
        assert_eq!(delta_s, 0.0, "Identical vectors should have delta_s = 0");

        println!("[PASS] hamming_sparse_identical_vectors");
    }

    #[test]
    fn test_hamming_sparse_completely_different() {
        let calculator = HammingSparseEntropy::new();

        let current = vec![1.0, 0.0, 1.0, 0.0];
        let history = vec![vec![0.0, 1.0, 0.0, 1.0]]; // No overlap

        let delta_s = calculator.compute_delta_s(&current, &history, 1).unwrap();
        assert_eq!(
            delta_s, 1.0,
            "Completely different vectors should have delta_s = 1"
        );

        println!("[PASS] hamming_sparse_completely_different");
    }

    #[test]
    fn test_hamming_sparse_partial_overlap() {
        let calculator = HammingSparseEntropy::new();

        let current = vec![0.0, 1.0, 0.0, 1.0, 0.0]; // Active: {1, 3}
        let history = vec![vec![0.0, 1.0, 0.0, 0.0, 1.0]]; // Active: {1, 4}

        // sym_diff = {3, 4}, union = {1, 3, 4}
        // Hamming = 2/3 = 0.666...
        let delta_s = calculator.compute_delta_s(&current, &history, 1).unwrap();
        assert!(
            (delta_s - 0.666).abs() < 0.01,
            "Partial overlap delta_s should be ~0.67, got {}",
            delta_s
        );

        println!("[PASS] hamming_sparse_partial_overlap");
    }

    #[test]
    fn test_hamming_sparse_empty_history() {
        let calculator = HammingSparseEntropy::new();

        let current = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let history: Vec<Vec<f32>> = vec![];

        let delta_s = calculator.compute_delta_s(&current, &history, 1).unwrap();
        assert_eq!(
            delta_s, 1.0,
            "Empty history should return maximum surprise"
        );

        println!("[PASS] hamming_sparse_empty_history");
    }

    #[test]
    fn test_hamming_sparse_empty_current_fails() {
        let calculator = HammingSparseEntropy::new();

        let current: Vec<f32> = vec![];
        let history = vec![vec![0.0, 1.0]];

        let result = calculator.compute_delta_s(&current, &history, 1);
        assert!(
            result.is_err(),
            "Empty current vector should FAIL FAST"
        );

        println!("[PASS] hamming_sparse_empty_current_fails");
    }

    #[test]
    fn test_hamming_sparse_both_empty_sets() {
        let calculator = HammingSparseEntropy::new();

        // All zeros = no active dimensions
        let current = vec![0.0, 0.0, 0.0, 0.0];
        let history = vec![vec![0.0, 0.0, 0.0, 0.0]];

        let delta_s = calculator.compute_delta_s(&current, &history, 1).unwrap();
        assert_eq!(
            delta_s, 0.0,
            "Both empty active sets should have delta_s = 0 (identical)"
        );

        println!("[PASS] hamming_sparse_both_empty_sets");
    }

    #[test]
    fn test_hamming_sparse_k_nearest() {
        let calculator = HammingSparseEntropy::new();

        let current = vec![0.0, 1.0, 1.0, 0.0]; // Active: {1, 2}

        let history = vec![
            vec![0.0, 1.0, 1.0, 0.0], // Identical: dist = 0
            vec![0.0, 1.0, 0.0, 0.0], // Active: {1}, dist = 1/2 = 0.5
            vec![0.0, 0.0, 0.0, 1.0], // Active: {3}, dist = 3/3 = 1.0
        ];

        // k=1: should use only the nearest (0)
        let delta_k1 = calculator.compute_delta_s(&current, &history, 1).unwrap();
        assert_eq!(delta_k1, 0.0, "k=1 should return nearest distance 0");

        // k=2: average of (0, 0.5) = 0.25
        let delta_k2 = calculator.compute_delta_s(&current, &history, 2).unwrap();
        assert!(
            (delta_k2 - 0.25).abs() < 0.01,
            "k=2 should return ~0.25, got {}",
            delta_k2
        );

        println!("[PASS] hamming_sparse_k_nearest");
    }

    #[test]
    fn test_hamming_sparse_threshold() {
        let calc_default = HammingSparseEntropy::new();
        let calc_high = HammingSparseEntropy::new().with_threshold(0.05);

        // Small values below high threshold
        let current = vec![0.01, 0.5, 0.01, 0.5];
        let history = vec![vec![0.01, 0.5, 0.01, 0.5]];

        // Default threshold (0.0): all 4 dims active
        let delta_default = calc_default
            .compute_delta_s(&current, &history, 1)
            .unwrap();
        assert_eq!(delta_default, 0.0);

        // High threshold (0.05): only dims with 0.5 active (2 dims)
        let delta_high = calc_high.compute_delta_s(&current, &history, 1).unwrap();
        assert_eq!(delta_high, 0.0);

        println!("[PASS] hamming_sparse_threshold");
    }

    #[test]
    fn test_hamming_sparse_realistic_splade() {
        let calculator = HammingSparseEntropy::new();

        // Simulate SPLADE-like sparse vectors (mostly zeros, few active)
        let mut current = vec![0.0f32; 30522]; // BERT vocab size
        current[100] = 0.8; // "word1"
        current[500] = 0.6; // "word2"
        current[1000] = 0.4; // "word3"

        let mut history_item = vec![0.0f32; 30522];
        history_item[100] = 0.7; // Same "word1"
        history_item[500] = 0.5; // Same "word2"
        history_item[2000] = 0.3; // Different "word4"

        // Active current: {100, 500, 1000}
        // Active history: {100, 500, 2000}
        // sym_diff = {1000, 2000}, union = {100, 500, 1000, 2000}
        // Hamming = 2/4 = 0.5
        let history = vec![history_item];
        let delta_s = calculator.compute_delta_s(&current, &history, 1).unwrap();

        assert!(
            (delta_s - 0.5).abs() < 0.01,
            "Realistic SPLADE vectors should have delta_s ~0.5, got {}",
            delta_s
        );

        println!("[PASS] hamming_sparse_realistic_splade");
    }

    #[test]
    fn test_hamming_sparse_embedder_type() {
        let calculator = HammingSparseEntropy::new();
        assert_eq!(calculator.embedder_type(), Embedder::Sparse);

        println!("[PASS] hamming_sparse_embedder_type");
    }

    #[test]
    fn test_hamming_sparse_valid_range() {
        let calculator = HammingSparseEntropy::new();

        // Test various sparse patterns
        for active_count in [0, 1, 5, 10, 50] {
            let mut current = vec![0.0f32; 100];
            for i in 0..active_count {
                current[i] = 0.5;
            }

            let mut history_item = vec![0.0f32; 100];
            for i in (active_count / 2)..(active_count + active_count / 2) {
                if i < 100 {
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
            assert!(!delta_s.is_nan());
            assert!(!delta_s.is_infinite());
        }

        println!("[PASS] hamming_sparse_valid_range");
    }

    #[test]
    fn test_hamming_sparse_nan_infinity_input() {
        let calculator = HammingSparseEntropy::new();

        // NaN in current
        let nan_current = vec![0.0, f32::NAN, 0.5];
        let history = vec![vec![0.0, 0.5, 0.5]];
        let result = calculator.compute_delta_s(&nan_current, &history, 1);
        assert!(result.is_err(), "NaN input should error");

        // Infinity in current
        let inf_current = vec![0.0, f32::INFINITY, 0.5];
        let result = calculator.compute_delta_s(&inf_current, &history, 1);
        assert!(result.is_err(), "Infinity input should error");

        println!("[PASS] hamming_sparse_nan_infinity_input");
    }
}
