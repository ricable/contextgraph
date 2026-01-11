//! Default KNN-based entropy for embedders without specialized methods.
//!
//! Formula: ΔS = σ((d_k - μ) / σ) normalized
//! Used for: E2-E4, E6-E8, E10-E12
//!
//! # Constitution Reference
//!
//! From constitution.yaml delta_sc.ΔS_methods:
//! E2-4,E8: "KNN: ΔS=σ((d_k-μ)/σ)"

use super::EmbedderEntropy;
use crate::config::SurpriseConfig;
use crate::error::{UtlError, UtlResult};
use crate::surprise::{compute_cosine_distance, EmbeddingDistanceCalculator};
use context_graph_core::teleological::Embedder;

/// Default KNN-based entropy for embedders without specialized methods.
///
/// Uses normalized k-nearest-neighbor distance as the surprise measure.
/// This is a robust fallback that works well for most embedding types.
///
/// # Algorithm
///
/// 1. Compute cosine distances to all history embeddings
/// 2. Sort and take k nearest
/// 3. Compute mean of k-nearest distances
/// 4. Normalize using sigmoid: ΔS = sigmoid((d_k - μ) / σ)
/// 5. Clamp to [0, 1]
#[derive(Debug, Clone)]
pub struct DefaultKnnEntropy {
    /// The embedder type this instance handles.
    embedder: Embedder,
    /// Internal distance calculator (reserved for future use).
    #[allow(dead_code)]
    distance_calc: EmbeddingDistanceCalculator,
    /// Running mean of distances for normalization.
    running_mean: f32,
    /// Running variance of distances for normalization.
    running_variance: f32,
    /// Number of samples seen for running statistics.
    sample_count: usize,
    /// EMA alpha for updating running statistics.
    #[allow(dead_code)]
    ema_alpha: f32,
}

impl DefaultKnnEntropy {
    /// Create a new default KNN entropy calculator.
    ///
    /// # Arguments
    /// * `embedder` - The embedder type this calculator handles
    pub fn new(embedder: Embedder) -> Self {
        Self {
            embedder,
            distance_calc: EmbeddingDistanceCalculator::default(),
            running_mean: 0.5,
            running_variance: 0.1,
            sample_count: 0,
            ema_alpha: 0.1,
        }
    }

    /// Create from SurpriseConfig.
    ///
    /// # Arguments
    /// * `embedder` - The embedder type this calculator handles
    /// * `config` - Surprise configuration
    pub fn from_config(embedder: Embedder, config: &SurpriseConfig) -> Self {
        Self {
            embedder,
            distance_calc: EmbeddingDistanceCalculator::from_config(config),
            running_mean: 0.5,
            running_variance: 0.1,
            sample_count: 0,
            ema_alpha: config.ema_alpha,
        }
    }

    /// Sigmoid function for normalization.
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Update running statistics with a new distance value.
    #[allow(dead_code)]
    fn update_statistics(&mut self, distance: f32) {
        self.sample_count += 1;

        if self.sample_count == 1 {
            self.running_mean = distance;
            self.running_variance = 0.1;
        } else {
            // EMA update
            let diff = distance - self.running_mean;
            self.running_mean += self.ema_alpha * diff;
            self.running_variance =
                (1.0 - self.ema_alpha) * (self.running_variance + self.ema_alpha * diff * diff);

            // Prevent variance from becoming too small
            self.running_variance = self.running_variance.max(0.01);
        }
    }
}

impl EmbedderEntropy for DefaultKnnEntropy {
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
            if past.is_empty() {
                continue;
            }

            let dist = compute_cosine_distance(current, past);
            if !dist.is_nan() && !dist.is_infinite() {
                distances.push(dist);
            }
        }

        if distances.is_empty() {
            return Ok(1.0);
        }

        // Sort distances ascending
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Take k nearest
        let k_actual = k.max(1).min(distances.len());
        let k_nearest = &distances[..k_actual];

        // Compute mean of k-nearest distances
        let d_k: f32 = k_nearest.iter().sum::<f32>() / k_actual as f32;

        // Normalize: z = (d_k - μ) / σ
        let std_dev = self.running_variance.sqrt().max(0.1);
        let z = (d_k - self.running_mean) / std_dev;

        // Sigmoid normalization
        let delta_s = Self::sigmoid(z);

        // Clamp per AP-10
        Ok(delta_s.clamp(0.0, 1.0))
    }

    fn embedder_type(&self) -> Embedder {
        self.embedder
    }

    fn reset(&mut self) {
        self.running_mean = 0.5;
        self.running_variance = 0.1;
        self.sample_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_knn_empty_history_returns_one() {
        let calculator = DefaultKnnEntropy::new(Embedder::TemporalRecent);
        let current = vec![0.5f32; 512];
        let history: Vec<Vec<f32>> = vec![];

        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1.0);

        println!("[PASS] default_knn_empty_history_returns_one");
    }

    #[test]
    fn test_default_knn_identical_vectors_low_surprise() {
        let calculator = DefaultKnnEntropy::new(Embedder::Graph);

        let current = vec![0.5f32; 384];
        let history: Vec<Vec<f32>> = vec![current.clone(); 10];

        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok());
        let delta_s = result.unwrap();

        // Distance is ~0, so normalized value should be below center
        assert!(
            delta_s < 0.6,
            "Identical vectors should have low surprise, got {}",
            delta_s
        );

        println!("[PASS] default_knn_identical_vectors_low_surprise");
    }

    #[test]
    fn test_default_knn_different_vectors_high_surprise() {
        let calculator = DefaultKnnEntropy::new(Embedder::TemporalPeriodic);

        // Current vector is orthogonal to history
        let mut current = vec![0.0f32; 512];
        current[0] = 1.0;

        let mut history_item = vec![0.0f32; 512];
        history_item[256] = 1.0;

        let history = vec![history_item];

        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok());
        let delta_s = result.unwrap();

        // Orthogonal vectors should have higher surprise
        assert!(
            delta_s > 0.5,
            "Orthogonal vectors should have high surprise, got {}",
            delta_s
        );

        println!("[PASS] default_knn_different_vectors_high_surprise");
    }

    #[test]
    fn test_default_knn_handles_all_fallback_embedders() {
        // Test that it works for all embedders it's meant to handle
        let fallback_embedders = [
            Embedder::TemporalRecent,
            Embedder::TemporalPeriodic,
            Embedder::TemporalPositional,
            Embedder::Sparse,
            Embedder::Code,
            Embedder::Graph,
            Embedder::Multimodal,
            Embedder::Entity,
            Embedder::LateInteraction,
        ];

        for embedder in fallback_embedders {
            let calculator = DefaultKnnEntropy::new(embedder);
            assert_eq!(calculator.embedder_type(), embedder);

            let current = vec![0.5f32; 100];
            let history = vec![vec![0.6f32; 100]; 5];

            let result = calculator.compute_delta_s(&current, &history, 3);
            assert!(
                result.is_ok(),
                "Failed for {:?}: {:?}",
                embedder,
                result.err()
            );

            let delta_s = result.unwrap();
            assert!((0.0..=1.0).contains(&delta_s));
        }

        println!("[PASS] default_knn_handles_all_fallback_embedders");
    }

    #[test]
    fn test_default_knn_empty_input_error() {
        let calculator = DefaultKnnEntropy::new(Embedder::TemporalRecent);
        let empty: Vec<f32> = vec![];
        let history = vec![vec![0.5f32; 100]];

        let result = calculator.compute_delta_s(&empty, &history, 5);
        assert!(matches!(result, Err(UtlError::EmptyInput)));

        println!("[PASS] default_knn_empty_input_error");
    }

    #[test]
    fn test_default_knn_reset() {
        let mut calculator = DefaultKnnEntropy::new(Embedder::Graph);

        // Simulate some updates
        calculator.update_statistics(0.3);
        calculator.update_statistics(0.7);
        assert!(calculator.sample_count > 0);

        calculator.reset();

        assert_eq!(calculator.sample_count, 0);
        assert_eq!(calculator.running_mean, 0.5);
        assert_eq!(calculator.running_variance, 0.1);

        println!("[PASS] default_knn_reset");
    }

    #[test]
    fn test_default_knn_from_config() {
        let mut config = SurpriseConfig::default();
        config.ema_alpha = 0.2;
        config.sample_count = 50;

        let calculator = DefaultKnnEntropy::from_config(Embedder::Code, &config);
        assert_eq!(calculator.embedder_type(), Embedder::Code);
        assert_eq!(calculator.ema_alpha, 0.2);

        println!("[PASS] default_knn_from_config");
    }

    #[test]
    fn test_default_knn_valid_range() {
        let calculator = DefaultKnnEntropy::new(Embedder::Entity);

        // Test various input patterns
        for test_pattern in 0..5 {
            let current: Vec<f32> = (0..384)
                .map(|i| (i + test_pattern * 100) as f32 / 384.0)
                .collect();

            let history: Vec<Vec<f32>> = (0..10)
                .map(|j| {
                    (0..384)
                        .map(|i| (i + j * 50) as f32 / 384.0)
                        .collect()
                })
                .collect();

            let result = calculator.compute_delta_s(&current, &history, 5);
            assert!(result.is_ok());
            let delta_s = result.unwrap();

            assert!((0.0..=1.0).contains(&delta_s));
            assert!(!delta_s.is_nan());
            assert!(!delta_s.is_infinite());
        }

        println!("[PASS] default_knn_valid_range");
    }

    #[test]
    fn test_default_knn_sigmoid() {
        // Test sigmoid function
        assert!((DefaultKnnEntropy::sigmoid(0.0) - 0.5).abs() < 0.001);
        assert!(DefaultKnnEntropy::sigmoid(10.0) > 0.99);
        assert!(DefaultKnnEntropy::sigmoid(-10.0) < 0.01);

        println!("[PASS] default_knn_sigmoid");
    }

    #[test]
    fn test_default_knn_k_parameter() {
        let calculator = DefaultKnnEntropy::new(Embedder::Multimodal);

        let current = vec![0.5f32; 768];
        let history: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![0.5 + (i as f32) * 0.02; 768])
            .collect();

        // Compare k=1 vs k=10
        let result_k1 = calculator.compute_delta_s(&current, &history, 1).unwrap();
        let result_k10 = calculator.compute_delta_s(&current, &history, 10).unwrap();

        // Both should be valid
        assert!((0.0..=1.0).contains(&result_k1));
        assert!((0.0..=1.0).contains(&result_k10));

        // k=1 uses only nearest, k=10 averages more neighbors
        // Results should differ (unless by coincidence)
        println!("k=1: {}, k=10: {}", result_k1, result_k10);

        println!("[PASS] default_knn_k_parameter");
    }

    #[test]
    fn test_default_knn_update_statistics() {
        let mut calculator = DefaultKnnEntropy::new(Embedder::TemporalRecent);

        assert_eq!(calculator.sample_count, 0);

        calculator.update_statistics(0.3);
        assert_eq!(calculator.sample_count, 1);
        assert_eq!(calculator.running_mean, 0.3);

        calculator.update_statistics(0.7);
        assert_eq!(calculator.sample_count, 2);
        // Mean should move toward 0.7
        assert!(calculator.running_mean > 0.3);

        println!("[PASS] default_knn_update_statistics");
    }
}
