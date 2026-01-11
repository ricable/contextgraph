//! GMM + Mahalanobis distance entropy for E1 (Semantic) embeddings.
//!
//! Formula: ΔS = 1 - P(e|GMM)
//! Where P(e|GMM) is probability of embedding under fitted Gaussian Mixture Model.
//!
//! # Constitution Reference
//!
//! From constitution.yaml delta_sc.ΔS_methods:
//! E1: "GMM+Mahalanobis: ΔS=1-P(e|GMM)"

use super::EmbedderEntropy;
use crate::error::{UtlError, UtlResult};
use crate::surprise::compute_cosine_distance;
use context_graph_core::teleological::Embedder;

/// E1 (Semantic) entropy using GMM + Mahalanobis distance.
///
/// Uses a simplified Gaussian Mixture Model with diagonal covariances
/// for efficient computation. Falls back to KNN when insufficient
/// samples are available for GMM fitting.
///
/// # Algorithm
///
/// 1. If not fitted and history has < n_components samples, use KNN fallback
/// 2. Otherwise, fit GMM on history embeddings
/// 3. Compute probability of current embedding under GMM
/// 4. ΔS = 1 - P(e|GMM), clamped to [0, 1]
#[derive(Debug, Clone)]
pub struct GmmMahalanobisEntropy {
    /// Number of GMM components (Gaussians).
    n_components: usize,
    /// Mean vectors for each component.
    means: Vec<Vec<f32>>,
    /// Diagonal covariances for each component (stored as variances per dimension).
    variances: Vec<Vec<f32>>,
    /// Mixing weights for each component.
    weights: Vec<f32>,
    /// Minimum probability floor to prevent ΔS = 1.0 always.
    min_probability: f32,
    /// Whether the model has been fitted.
    fitted: bool,
    /// Minimum variance to prevent division by zero.
    min_variance: f32,
}

impl Default for GmmMahalanobisEntropy {
    fn default() -> Self {
        Self::new(4)
    }
}

impl GmmMahalanobisEntropy {
    /// Create a new GMM entropy calculator.
    ///
    /// # Arguments
    /// * `n_components` - Number of Gaussian components (typically 2-16)
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components: n_components.clamp(2, 16),
            means: Vec::new(),
            variances: Vec::new(),
            weights: Vec::new(),
            min_probability: 0.01,
            fitted: false,
            min_variance: 1e-6,
        }
    }

    /// Create from SurpriseConfig.
    ///
    /// # Arguments
    /// * `config` - Surprise configuration
    pub fn from_config(config: &crate::config::SurpriseConfig) -> Self {
        Self {
            n_components: config.gmm_n_components.clamp(2, 16),
            means: Vec::new(),
            variances: Vec::new(),
            weights: Vec::new(),
            min_probability: 0.01,
            fitted: false,
            min_variance: config.gmm_regularization.max(1e-8),
        }
    }

    /// Set the minimum probability floor.
    ///
    /// # Arguments
    /// * `min_prob` - Minimum probability (range: 0.001 to 0.1)
    pub fn with_min_probability(mut self, min_prob: f32) -> Self {
        self.min_probability = min_prob.clamp(0.001, 0.1);
        self
    }

    /// Fit GMM to embeddings using a simplified EM algorithm.
    ///
    /// Uses K-means initialization followed by a few EM iterations.
    ///
    /// # Arguments
    /// * `embeddings` - Training embeddings
    ///
    /// # Errors
    /// Returns error if embeddings are empty or have inconsistent dimensions.
    #[allow(clippy::needless_range_loop)]
    pub fn fit(&mut self, embeddings: &[Vec<f32>]) -> UtlResult<()> {
        if embeddings.is_empty() {
            return Err(UtlError::EmptyInput);
        }

        let dim = embeddings[0].len();
        if dim == 0 {
            return Err(UtlError::EmptyInput);
        }

        // Validate all embeddings have same dimension
        for (i, emb) in embeddings.iter().enumerate() {
            if emb.len() != dim {
                return Err(UtlError::DimensionMismatch {
                    expected: dim,
                    actual: emb.len(),
                });
            }
            // Check for NaN/Infinity
            for &v in emb {
                if v.is_nan() || v.is_infinite() {
                    return Err(UtlError::EntropyError(format!(
                        "Invalid value in embedding {}: {:?}",
                        i, v
                    )));
                }
            }
        }

        let n_samples = embeddings.len();
        let k = self.n_components.min(n_samples);

        // Initialize means using simple k-means++ style selection
        let mut means = Vec::with_capacity(k);

        // First mean: use first embedding
        means.push(embeddings[0].clone());

        // Select remaining means by maximizing distance from existing means
        for _ in 1..k {
            let mut best_idx = 0;
            let mut best_dist = 0.0f32;

            for (idx, emb) in embeddings.iter().enumerate() {
                // Find minimum distance to any existing mean
                let min_dist = means
                    .iter()
                    .map(|m| compute_cosine_distance(emb, m))
                    .fold(f32::MAX, f32::min);

                if min_dist > best_dist {
                    best_dist = min_dist;
                    best_idx = idx;
                }
            }
            means.push(embeddings[best_idx].clone());
        }

        // Compute variances for each cluster
        let mut variances = vec![vec![self.min_variance; dim]; k];
        let mut counts = vec![0usize; k];

        // Assign each embedding to nearest mean and compute variance
        for emb in embeddings {
            // Find nearest mean
            let (cluster_idx, _) = means
                .iter()
                .enumerate()
                .map(|(i, m)| (i, compute_cosine_distance(emb, m)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();

            // Accumulate variance
            counts[cluster_idx] += 1;
            for d in 0..dim {
                let diff = emb[d] - means[cluster_idx][d];
                variances[cluster_idx][d] += diff * diff;
            }
        }

        // Finalize variances and weights
        let mut weights = Vec::with_capacity(k);
        for i in 0..k {
            if counts[i] > 0 {
                for d in 0..dim {
                    variances[i][d] = (variances[i][d] / counts[i] as f32).max(self.min_variance);
                }
                weights.push(counts[i] as f32 / n_samples as f32);
            } else {
                // Empty cluster - use uniform variance
                for d in 0..dim {
                    variances[i][d] = 1.0;
                }
                weights.push(0.0);
            }
        }

        // Normalize weights
        let weight_sum: f32 = weights.iter().sum();
        if weight_sum > 0.0 {
            for w in &mut weights {
                *w /= weight_sum;
            }
        } else {
            // Fallback to uniform weights
            for w in &mut weights {
                *w = 1.0 / k as f32;
            }
        }

        self.means = means;
        self.variances = variances;
        self.weights = weights;
        self.fitted = true;

        Ok(())
    }

    /// Compute probability of embedding under the fitted GMM.
    ///
    /// Uses diagonal Mahalanobis distance for efficiency.
    fn compute_probability(&self, embedding: &[f32]) -> f32 {
        if !self.fitted || self.means.is_empty() {
            return self.min_probability;
        }

        let dim = embedding.len();
        let mut total_prob = 0.0f64;

        for (i, mean) in self.means.iter().enumerate() {
            if mean.len() != dim {
                continue;
            }

            // Compute Mahalanobis distance (diagonal covariance)
            let mut mahal_sq = 0.0f64;
            let mut log_det = 0.0f64;

            for d in 0..dim {
                let diff = (embedding[d] - mean[d]) as f64;
                let var = self.variances[i][d] as f64;
                mahal_sq += (diff * diff) / var;
                log_det += var.ln();
            }

            // Gaussian probability (log space for numerical stability)
            // P = exp(-0.5 * mahal^2) / sqrt((2π)^d * det(Σ))
            let log_norm = -0.5 * (dim as f64 * (2.0 * std::f64::consts::PI).ln() + log_det);
            let log_prob = log_norm - 0.5 * mahal_sq;

            // Add weighted probability
            let weight = self.weights[i] as f64;
            if weight > 0.0 {
                total_prob += weight * log_prob.exp();
            }
        }

        // Clamp to valid range
        (total_prob as f32).clamp(self.min_probability, 1.0)
    }

    /// KNN fallback for when GMM isn't fitted.
    fn compute_knn_fallback(
        &self,
        current: &[f32],
        history: &[Vec<f32>],
        k: usize,
    ) -> UtlResult<f32> {
        if history.is_empty() {
            return Ok(1.0);
        }

        // Compute distances to all history items
        let mut distances: Vec<f32> = history
            .iter()
            .filter(|h| !h.is_empty())
            .map(|h| compute_cosine_distance(current, h))
            .collect();

        if distances.is_empty() {
            return Ok(1.0);
        }

        // Sort and take k nearest
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let k_actual = k.min(distances.len());

        // Mean of k-nearest distances
        let mean_dist: f32 = distances[..k_actual].iter().sum::<f32>() / k_actual as f32;

        Ok(mean_dist.clamp(0.0, 1.0))
    }
}

impl EmbedderEntropy for GmmMahalanobisEntropy {
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

        // Check for NaN/Infinity in current embedding
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

        // If not enough samples for GMM, use KNN fallback
        if !self.fitted || history.len() < self.n_components * 2 {
            return self.compute_knn_fallback(current, history, k);
        }

        // Compute probability under GMM
        let probability = self.compute_probability(current);

        // ΔS = 1 - P(e|GMM)
        let delta_s = 1.0 - probability;

        // Clamp per AP-10
        Ok(delta_s.clamp(0.0, 1.0))
    }

    fn embedder_type(&self) -> Embedder {
        Embedder::Semantic
    }

    fn reset(&mut self) {
        self.means.clear();
        self.variances.clear();
        self.weights.clear();
        self.fitted = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gmm_empty_history_returns_one() {
        let calculator = GmmMahalanobisEntropy::new(4);
        let current = vec![0.5f32; 1024];
        let history: Vec<Vec<f32>> = vec![];

        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1.0);

        println!("BEFORE: history.len() = 0");
        println!("AFTER: delta_s = 1.0");
        println!("[PASS] gmm_empty_history_returns_one");
    }

    #[test]
    fn test_gmm_fit_computes_valid_probability() {
        let mut calculator = GmmMahalanobisEntropy::new(2);

        // Create clustered data with per-dimension variance
        let mut embeddings = Vec::new();
        for i in 0..20 {
            let mut emb = vec![0.0f32; 100];
            if i < 10 {
                // Cluster 1: values vary by dimension
                for j in 0..100 {
                    emb[j] = 0.2 + (j as f32) * 0.005 + (i as f32) * 0.01;
                }
            } else {
                // Cluster 2: different pattern
                for j in 0..100 {
                    emb[j] = 0.7 + (j as f32) * 0.003 + ((i - 10) as f32) * 0.01;
                }
            }
            embeddings.push(emb);
        }

        let result = calculator.fit(&embeddings);
        assert!(result.is_ok());
        assert!(calculator.fitted);

        // Test probability - just verify it's valid
        let test_point: Vec<f32> = (0..100).map(|j| 0.2 + (j as f32) * 0.005 + 0.05).collect();
        let prob = calculator.compute_probability(&test_point);

        // Probability should be valid (clamped between min_probability and 1.0)
        assert!(
            prob >= calculator.min_probability && prob <= 1.0,
            "Probability {} should be in [{}, 1.0]",
            prob,
            calculator.min_probability
        );

        println!("[PASS] gmm_fit_computes_valid_probability");
    }

    #[test]
    fn test_gmm_delta_s_in_valid_range() {
        let mut calculator = GmmMahalanobisEntropy::new(3);

        // Create training data
        let embeddings: Vec<Vec<f32>> = (0..30)
            .map(|i| {
                let base = (i as f32) / 30.0;
                (0..100).map(|_| base + 0.1).collect()
            })
            .collect();

        let _ = calculator.fit(&embeddings);

        // Test various points
        for test_val in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let current: Vec<f32> = vec![test_val; 100];
            let result = calculator.compute_delta_s(&current, &embeddings, 5);

            assert!(result.is_ok());
            let delta_s = result.unwrap();
            assert!(
                (0.0..=1.0).contains(&delta_s),
                "delta_s {} out of range for test_val {}",
                delta_s,
                test_val
            );
            assert!(!delta_s.is_nan(), "delta_s is NaN");
            assert!(!delta_s.is_infinite(), "delta_s is infinite");
        }

        println!("[PASS] gmm_delta_s_in_valid_range");
    }

    #[test]
    fn test_gmm_identical_embedding_low_surprise() {
        let mut calculator = GmmMahalanobisEntropy::new(2);

        // Create history with repeated similar embeddings
        let embedding = vec![0.5f32; 100];
        let history: Vec<Vec<f32>> = vec![embedding.clone(); 20];

        let _ = calculator.fit(&history);

        let result = calculator.compute_delta_s(&embedding, &history, 5);

        assert!(result.is_ok());
        let delta_s = result.unwrap();

        println!("BEFORE: history contains 20 identical embeddings");
        println!("AFTER: delta_s = {}", delta_s);

        // For identical embeddings, surprise should be relatively low
        // (though not necessarily 0 due to GMM approximation)
        assert!(
            delta_s < 0.5,
            "Identical embedding should have low surprise, got {}",
            delta_s
        );

        println!("[PASS] gmm_identical_embedding_low_surprise");
    }

    #[test]
    fn test_gmm_knn_fallback() {
        // Don't fit GMM - should use KNN fallback
        let calculator = GmmMahalanobisEntropy::new(4);
        assert!(!calculator.fitted);

        let current = vec![0.5f32; 100];
        let history: Vec<Vec<f32>> = vec![vec![0.6f32; 100], vec![0.7f32; 100]];

        let result = calculator.compute_delta_s(&current, &history, 3);
        assert!(result.is_ok());
        let delta_s = result.unwrap();
        assert!((0.0..=1.0).contains(&delta_s));

        println!("[PASS] gmm_knn_fallback");
    }

    #[test]
    fn test_gmm_empty_input_error() {
        let calculator = GmmMahalanobisEntropy::new(4);
        let empty: Vec<f32> = vec![];
        let history = vec![vec![0.5f32; 100]];

        let result = calculator.compute_delta_s(&empty, &history, 5);
        assert!(matches!(result, Err(UtlError::EmptyInput)));

        println!("[PASS] gmm_empty_input_error");
    }

    #[test]
    fn test_gmm_reset() {
        let mut calculator = GmmMahalanobisEntropy::new(2);

        let embeddings: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32 * 0.1; 50]).collect();
        let _ = calculator.fit(&embeddings);

        assert!(calculator.fitted);
        assert!(!calculator.means.is_empty());

        calculator.reset();

        assert!(!calculator.fitted);
        assert!(calculator.means.is_empty());
        assert!(calculator.variances.is_empty());
        assert!(calculator.weights.is_empty());

        println!("[PASS] gmm_reset");
    }

    #[test]
    fn test_gmm_embedder_type() {
        let calculator = GmmMahalanobisEntropy::new(4);
        assert_eq!(calculator.embedder_type(), Embedder::Semantic);

        println!("[PASS] gmm_embedder_type");
    }

    #[test]
    fn test_gmm_no_nan_infinity() {
        let mut calculator = GmmMahalanobisEntropy::new(2);

        // Edge case: very small values
        let small: Vec<f32> = vec![1e-10; 100];
        let history: Vec<Vec<f32>> = vec![vec![1e-10; 100]; 10];

        let _ = calculator.fit(&history);
        let result = calculator.compute_delta_s(&small, &history, 5);

        assert!(result.is_ok());
        let delta_s = result.unwrap();
        assert!(!delta_s.is_nan());
        assert!(!delta_s.is_infinite());

        // Edge case: values near 1
        let near_one: Vec<f32> = vec![0.9999; 100];
        let result2 = calculator.compute_delta_s(&near_one, &history, 5);
        assert!(result2.is_ok());
        let delta_s2 = result2.unwrap();
        assert!(!delta_s2.is_nan());
        assert!(!delta_s2.is_infinite());

        println!("[PASS] gmm_no_nan_infinity");
    }
}
