//! Hamming distance to prototypes entropy for E9 (HDC) embeddings.
//!
//! Formula: ΔS = min_hamming(e, prototypes) / dim
//! For HDC binary patterns projected to dense representation.
//!
//! # Constitution Reference
//!
//! From constitution.yaml delta_sc.ΔS_methods:
//! E9: "Hamming: ΔS=min_hamming/dim"

use super::EmbedderEntropy;
use crate::error::{UtlError, UtlResult};
use context_graph_core::teleological::Embedder;

/// E9 (HDC) entropy using Hamming distance to learned prototypes.
///
/// Hyperdimensional Computing (HDC) uses binary-like patterns. This calculator
/// binarizes embeddings using a threshold and computes Hamming distance to
/// the nearest learned prototype.
///
/// # Algorithm
///
/// 1. Learn prototypes from history (cluster centroids)
/// 2. Binarize current embedding using threshold
/// 3. Find minimum Hamming distance to any prototype
/// 4. ΔS = min_hamming / dim, clamped to [0, 1]
#[derive(Debug, Clone)]
pub struct HammingPrototypeEntropy {
    /// Learned prototypes (binarized form stored as bool vectors).
    prototypes: Vec<Vec<bool>>,
    /// Threshold for binarizing float embeddings.
    binarization_threshold: f32,
    /// Maximum number of prototypes to maintain.
    max_prototypes: usize,
}

impl Default for HammingPrototypeEntropy {
    fn default() -> Self {
        Self::new(100)
    }
}

impl HammingPrototypeEntropy {
    /// Create a new Hamming prototype entropy calculator.
    ///
    /// # Arguments
    /// * `max_prototypes` - Maximum prototypes to maintain (range: 10 to 1000)
    pub fn new(max_prototypes: usize) -> Self {
        Self {
            prototypes: Vec::new(),
            binarization_threshold: 0.5,
            max_prototypes: max_prototypes.clamp(10, 1000),
        }
    }

    /// Set the binarization threshold.
    ///
    /// # Arguments
    /// * `threshold` - Threshold for converting float to binary (range: 0.0 to 1.0)
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.binarization_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Add a prototype from a float embedding.
    ///
    /// Binarizes the embedding and adds it to the prototype set.
    /// If max_prototypes is reached, removes the oldest prototype (FIFO eviction).
    pub fn add_prototype(&mut self, embedding: &[f32]) {
        if embedding.is_empty() {
            return;
        }

        let binary = self.binarize(embedding);

        // Evict oldest if at capacity
        if self.prototypes.len() >= self.max_prototypes {
            self.prototypes.remove(0);
        }

        self.prototypes.push(binary);
    }

    /// Learn prototypes from a set of embeddings using simple clustering.
    ///
    /// # Arguments
    /// * `embeddings` - Training embeddings
    /// * `n_prototypes` - Number of prototypes to learn
    pub fn learn_prototypes(&mut self, embeddings: &[Vec<f32>], n_prototypes: usize) {
        if embeddings.is_empty() || n_prototypes == 0 {
            return;
        }

        self.prototypes.clear();

        let n = n_prototypes.min(embeddings.len()).min(self.max_prototypes);

        // Simple approach: take evenly spaced samples
        let step = embeddings.len() / n;
        for i in 0..n {
            let idx = i * step;
            if idx < embeddings.len() {
                self.add_prototype(&embeddings[idx]);
            }
        }
    }

    /// Binarize a float embedding.
    fn binarize(&self, embedding: &[f32]) -> Vec<bool> {
        embedding
            .iter()
            .map(|&v| v > self.binarization_threshold)
            .collect()
    }

    /// Compute Hamming distance between two binary vectors.
    fn hamming_distance(a: &[bool], b: &[bool]) -> usize {
        let len = a.len().min(b.len());
        let mut distance = 0usize;

        for i in 0..len {
            if a[i] != b[i] {
                distance += 1;
            }
        }

        // Add difference in length as additional distance
        distance += a.len().abs_diff(b.len());

        distance
    }

    /// Get the number of current prototypes.
    pub fn prototype_count(&self) -> usize {
        self.prototypes.len()
    }
}

impl EmbedderEntropy for HammingPrototypeEntropy {
    fn compute_delta_s(
        &self,
        current: &[f32],
        history: &[Vec<f32>],
        _k: usize,
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

        // If no prototypes, try to learn from history
        let prototypes_to_use: Vec<Vec<bool>> = if self.prototypes.is_empty() {
            if history.is_empty() {
                return Ok(1.0); // Maximum surprise with no reference
            }

            // Learn temporary prototypes from history
            if history.len() < 5 {
                // Not enough for meaningful prototypes
                return Ok(1.0);
            }

            // Create temporary prototypes
            let n_temp = (history.len() / 5).min(self.max_prototypes);
            let step = history.len() / n_temp.max(1);

            (0..n_temp)
                .filter_map(|i| {
                    let idx = i * step;
                    if idx < history.len() && !history[idx].is_empty() {
                        Some(self.binarize(&history[idx]))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            self.prototypes.clone()
        };

        if prototypes_to_use.is_empty() {
            return Ok(1.0);
        }

        // Binarize current embedding
        let current_binary = self.binarize(current);
        let dim = current_binary.len();

        if dim == 0 {
            return Err(UtlError::EmptyInput);
        }

        // Find minimum Hamming distance to any prototype
        let mut min_hamming = usize::MAX;

        for prototype in &prototypes_to_use {
            let distance = Self::hamming_distance(&current_binary, prototype);
            if distance < min_hamming {
                min_hamming = distance;
            }
        }

        if min_hamming == usize::MAX {
            return Ok(1.0);
        }

        // Normalize by dimension: ΔS = min_hamming / dim
        let delta_s = min_hamming as f32 / dim as f32;

        // Clamp per AP-10
        Ok(delta_s.clamp(0.0, 1.0))
    }

    fn embedder_type(&self) -> Embedder {
        Embedder::Hdc
    }

    fn reset(&mut self) {
        self.prototypes.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_exact_prototype_returns_zero() {
        let mut calculator = HammingPrototypeEntropy::new(10);

        // Create a specific pattern
        let pattern: Vec<f32> = (0..100)
            .map(|i| if i % 2 == 0 { 0.8 } else { 0.2 })
            .collect();

        calculator.add_prototype(&pattern);

        // Query with the exact same pattern
        let result = calculator.compute_delta_s(&pattern, &[], 5);
        assert!(result.is_ok());
        let delta_s = result.unwrap();

        assert!(
            delta_s < 0.01,
            "Exact prototype match should have near-zero surprise, got {}",
            delta_s
        );

        println!("BEFORE: Added pattern as prototype");
        println!("AFTER: delta_s = {}", delta_s);
        println!("[PASS] hamming_exact_prototype_returns_zero");
    }

    #[test]
    fn test_hamming_orthogonal_returns_high() {
        let mut calculator = HammingPrototypeEntropy::new(10).with_threshold(0.5);

        // Create prototype with all values above threshold
        let prototype: Vec<f32> = vec![0.9; 100];
        calculator.add_prototype(&prototype);

        // Query with all values below threshold (opposite binary pattern)
        let opposite: Vec<f32> = vec![0.1; 100];

        let result = calculator.compute_delta_s(&opposite, &[], 5);
        assert!(result.is_ok());
        let delta_s = result.unwrap();

        // All bits different = maximum distance
        assert!(
            delta_s > 0.9,
            "Orthogonal pattern should have high surprise, got {}",
            delta_s
        );

        println!("[PASS] hamming_orthogonal_returns_high");
    }

    #[test]
    fn test_hamming_max_prototypes_enforced() {
        // Note: max_prototypes is clamped to [10, 1000], so use 10 as minimum
        let mut calculator = HammingPrototypeEntropy::new(10);

        // Add more than max prototypes (15 > 10)
        for i in 0..15 {
            calculator.add_prototype(&vec![i as f32 * 0.1; 100]);
        }

        assert_eq!(
            calculator.prototype_count(),
            10,
            "Should not exceed max_prototypes (10)"
        );

        println!("BEFORE: added 15 prototypes to calculator with max=10");
        println!(
            "AFTER: prototypes.len() = {}",
            calculator.prototype_count()
        );
        println!("[PASS] hamming_max_prototypes_enforced");
    }

    #[test]
    fn test_hamming_empty_history_with_no_prototypes() {
        let calculator = HammingPrototypeEntropy::new(10);
        let current = vec![0.5f32; 100];
        let history: Vec<Vec<f32>> = vec![];

        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1.0);

        println!("[PASS] hamming_empty_history_with_no_prototypes");
    }

    #[test]
    fn test_hamming_learns_from_history() {
        let calculator = HammingPrototypeEntropy::new(10);

        let current = vec![0.6f32; 100];
        let history: Vec<Vec<f32>> = (0..20).map(|_| vec![0.6f32; 100]).collect();

        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok());
        let delta_s = result.unwrap();

        // Should have low surprise since current is similar to history
        assert!(
            delta_s < 0.5,
            "Should have low surprise with matching history, got {}",
            delta_s
        );

        println!("[PASS] hamming_learns_from_history");
    }

    #[test]
    fn test_hamming_empty_input_error() {
        let calculator = HammingPrototypeEntropy::new(10);
        let empty: Vec<f32> = vec![];
        let history = vec![vec![0.5f32; 100]];

        let result = calculator.compute_delta_s(&empty, &history, 5);
        assert!(matches!(result, Err(UtlError::EmptyInput)));

        println!("[PASS] hamming_empty_input_error");
    }

    #[test]
    fn test_hamming_embedder_type() {
        let calculator = HammingPrototypeEntropy::new(10);
        assert_eq!(calculator.embedder_type(), Embedder::Hdc);

        println!("[PASS] hamming_embedder_type");
    }

    #[test]
    fn test_hamming_reset() {
        let mut calculator = HammingPrototypeEntropy::new(10);

        calculator.add_prototype(&vec![0.5f32; 100]);
        calculator.add_prototype(&vec![0.6f32; 100]);
        assert_eq!(calculator.prototype_count(), 2);

        calculator.reset();
        assert_eq!(calculator.prototype_count(), 0);

        println!("[PASS] hamming_reset");
    }

    #[test]
    fn test_hamming_valid_range() {
        let mut calculator = HammingPrototypeEntropy::new(20);

        // Add various prototypes
        for i in 0..10 {
            calculator.add_prototype(&vec![i as f32 * 0.1; 100]);
        }

        // Test various inputs
        for test_val in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let current: Vec<f32> = vec![test_val; 100];
            let result = calculator.compute_delta_s(&current, &[], 5);

            assert!(result.is_ok());
            let delta_s = result.unwrap();
            assert!((0.0..=1.0).contains(&delta_s));
            assert!(!delta_s.is_nan());
            assert!(!delta_s.is_infinite());
        }

        println!("[PASS] hamming_valid_range");
    }

    #[test]
    fn test_hamming_distance_function() {
        let a = vec![true, true, false, false];
        let b = vec![true, false, true, false];

        let dist = HammingPrototypeEntropy::hamming_distance(&a, &b);
        assert_eq!(dist, 2); // Two positions differ

        let c = vec![true, true, true, true];
        let dist2 = HammingPrototypeEntropy::hamming_distance(&a, &c);
        assert_eq!(dist2, 2); // Two positions differ

        println!("[PASS] hamming_distance_function");
    }

    #[test]
    fn test_hamming_learn_prototypes() {
        let mut calculator = HammingPrototypeEntropy::new(10);

        let embeddings: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32 * 0.05; 50]).collect();

        calculator.learn_prototypes(&embeddings, 5);

        assert!(
            calculator.prototype_count() > 0 && calculator.prototype_count() <= 5,
            "Should have learned prototypes"
        );

        println!("[PASS] hamming_learn_prototypes");
    }
}
