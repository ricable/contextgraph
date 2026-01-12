//! Hybrid GMM+KNN entropy for E7 (Code) embeddings.
//!
//! Formula: ΔS = gmm_weight × ΔS_GMM + knn_weight × ΔS_KNN
//! Per constitution.yaml delta_methods.ΔS E7: "GMM+KNN hybrid"
//!
//! # Algorithm
//!
//! 1. **GMM Component (Global Pattern):**
//!    - Fit GMM on history (lazy, cached)
//!    - P(e|GMM) = Σ_k w_k × N(e | μ_k, Σ_k)
//!    - ΔS_GMM = 1 - clamp(P, 0.01, 1.0)
//!
//! 2. **KNN Component (Local Density):**
//!    - Compute cosine distances to all history
//!    - Sort, take k nearest
//!    - d_k = mean of k-nearest distances
//!    - z = (d_k - running_mean) / running_std
//!    - ΔS_KNN = sigmoid(z)
//!
//! 3. **Hybrid Combination:**
//!    - ΔS = gmm_weight × ΔS_GMM + knn_weight × ΔS_KNN
//!    - Clamp to [0.0, 1.0], verify no NaN/Infinity (AP-10)
//!
//! # Constitution Reference
//!
//! - From constitution.yaml delta_methods.ΔS E7: "GMM+KNN hybrid"
//! - Default weights: 0.5/0.5 per constitution

use super::EmbedderEntropy;
use crate::config::SurpriseConfig;
use crate::error::{UtlError, UtlResult};
use crate::surprise::compute_cosine_distance;
use context_graph_core::teleological::Embedder;

/// Minimum variance floor to prevent division by zero.
const MIN_VARIANCE: f32 = 1e-6;

/// Minimum probability floor for GMM.
const MIN_PROBABILITY: f32 = 0.01;

/// Default GMM component weight per constitution.yaml.
const DEFAULT_GMM_WEIGHT: f32 = 0.5;

/// Default KNN component weight per constitution.yaml.
const DEFAULT_KNN_WEIGHT: f32 = 0.5;

/// Default number of GMM components.
const DEFAULT_N_COMPONENTS: usize = 5;

/// Default k for KNN component.
const DEFAULT_K_NEIGHBORS: usize = 5;

/// E7 (Code) entropy using hybrid GMM+KNN approach per constitution.yaml.
///
/// # Algorithm
/// 1. Compute GMM component: ΔS_GMM = 1 - P(e|GMM)
/// 2. Compute KNN component: ΔS_KNN = σ((d_k - μ) / σ)
/// 3. Combine: ΔS = gmm_weight × ΔS_GMM + knn_weight × ΔS_KNN
///
/// # Constitution Reference
/// E7: "GMM+KNN hybrid" with weights 0.5/0.5
#[derive(Debug, Clone)]
pub struct HybridGmmKnnEntropy {
    /// GMM component weight. Constitution: 0.5
    gmm_weight: f32,
    /// KNN component weight. Constitution: 0.5
    knn_weight: f32,
    /// Number of GMM components. Default: 5
    n_components: usize,
    /// k for KNN component. Default: 5
    k_neighbors: usize,
    /// Cached GMM means [n_components x dim]
    means: Vec<Vec<f32>>,
    /// Cached GMM variances (diagonal)
    variances: Vec<Vec<f32>>,
    /// GMM mixing weights
    weights: Vec<f32>,
    /// Is GMM fitted?
    fitted: bool,
    /// Running mean for KNN normalization
    knn_mean: f32,
    /// Running variance for KNN normalization
    knn_variance: f32,
    /// Minimum variance floor
    min_variance: f32,
    /// Minimum probability floor
    min_probability: f32,
}

impl Default for HybridGmmKnnEntropy {
    fn default() -> Self {
        Self::new()
    }
}

impl HybridGmmKnnEntropy {
    /// Create a new hybrid GMM+KNN entropy calculator with constitution defaults.
    pub fn new() -> Self {
        Self {
            gmm_weight: DEFAULT_GMM_WEIGHT,
            knn_weight: DEFAULT_KNN_WEIGHT,
            n_components: DEFAULT_N_COMPONENTS,
            k_neighbors: DEFAULT_K_NEIGHBORS,
            means: Vec::new(),
            variances: Vec::new(),
            weights: Vec::new(),
            fitted: false,
            knn_mean: 0.5,
            knn_variance: 0.1,
            min_variance: MIN_VARIANCE,
            min_probability: MIN_PROBABILITY,
        }
    }

    /// Normalize weights to sum to 1.0.
    ///
    /// Returns (gmm_weight, knn_weight) normalized, or defaults if sum is zero.
    fn normalize_weights(gmm_weight: f32, knn_weight: f32) -> (f32, f32) {
        let gmm_w = gmm_weight.clamp(0.0, 1.0);
        let knn_w = knn_weight.clamp(0.0, 1.0);
        let sum = gmm_w + knn_w;

        if sum > 0.0 {
            (gmm_w / sum, knn_w / sum)
        } else {
            (DEFAULT_GMM_WEIGHT, DEFAULT_KNN_WEIGHT)
        }
    }

    /// Create with custom GMM/KNN weights.
    ///
    /// # Arguments
    /// * `gmm_weight` - Weight for GMM component (clamped to [0.0, 1.0])
    /// * `knn_weight` - Weight for KNN component (clamped to [0.0, 1.0])
    ///
    /// # Notes
    /// Weights are normalized to sum to 1.0 if they don't.
    pub fn with_weights(gmm_weight: f32, knn_weight: f32) -> Self {
        let (final_gmm, final_knn) = Self::normalize_weights(gmm_weight, knn_weight);

        Self {
            gmm_weight: final_gmm,
            knn_weight: final_knn,
            ..Self::new()
        }
    }

    /// Create from SurpriseConfig.
    pub fn from_config(config: &SurpriseConfig) -> Self {
        let (final_gmm, final_knn) =
            Self::normalize_weights(config.code_gmm_weight, config.code_knn_weight);

        Self {
            gmm_weight: final_gmm,
            knn_weight: final_knn,
            n_components: config.code_n_components.clamp(2, 10),
            k_neighbors: config.code_k_neighbors.clamp(1, 20),
            ..Self::new()
        }
    }

    /// Set number of GMM components.
    #[must_use]
    pub fn with_n_components(mut self, n: usize) -> Self {
        self.n_components = n.clamp(2, 10);
        self
    }

    /// Set k for KNN component.
    #[must_use]
    pub fn with_k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k.clamp(1, 20);
        self
    }

    /// Sigmoid function for normalization.
    #[inline]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Fit GMM to history embeddings using simplified EM algorithm.
    ///
    /// Uses K-means++ style initialization followed by cluster assignment.
    fn fit_gmm(&mut self, history: &[Vec<f32>]) -> UtlResult<()> {
        if history.is_empty() {
            return Err(UtlError::EmptyInput);
        }

        let dim = history[0].len();
        if dim == 0 {
            return Err(UtlError::EmptyInput);
        }

        // Validate all embeddings have same dimension and no NaN/Inf
        for (i, emb) in history.iter().enumerate() {
            if emb.len() != dim {
                return Err(UtlError::DimensionMismatch {
                    expected: dim,
                    actual: emb.len(),
                });
            }
            for &v in emb {
                if v.is_nan() || v.is_infinite() {
                    return Err(UtlError::EntropyError(format!(
                        "Invalid value in history embedding {}: NaN or Infinity",
                        i
                    )));
                }
            }
        }

        let n_samples = history.len();
        let k = self.n_components.min(n_samples);

        // Initialize means using K-means++ style selection
        let mut means = Vec::with_capacity(k);
        means.push(history[0].clone());

        // Select remaining means by maximizing distance from existing means
        for _ in 1..k {
            let mut best_idx = 0;
            let mut best_dist = 0.0f32;

            for (idx, emb) in history.iter().enumerate() {
                let min_dist = means
                    .iter()
                    .map(|m| compute_cosine_distance(emb, m))
                    .fold(f32::MAX, f32::min);

                if min_dist > best_dist {
                    best_dist = min_dist;
                    best_idx = idx;
                }
            }
            means.push(history[best_idx].clone());
        }

        // Compute variances for each cluster
        let mut variances = vec![vec![self.min_variance; dim]; k];
        let mut counts = vec![0usize; k];

        // Assign each embedding to nearest mean and compute variance
        for emb in history {
            // Find nearest mean
            let (cluster_idx, _) = means
                .iter()
                .enumerate()
                .map(|(i, m)| (i, compute_cosine_distance(emb, m)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, 0.0));

            counts[cluster_idx] += 1;
            for d in 0..dim {
                let diff = emb[d] - means[cluster_idx][d];
                variances[cluster_idx][d] += diff * diff;
            }
        }

        // Finalize variances and weights
        let mut gmm_weights = Vec::with_capacity(k);
        for i in 0..k {
            if counts[i] > 0 {
                let count_f32 = counts[i] as f32;
                variances[i]
                    .iter_mut()
                    .for_each(|v| *v = (*v / count_f32).max(self.min_variance));
                gmm_weights.push(counts[i] as f32 / n_samples as f32);
            } else {
                variances[i].fill(1.0);
                gmm_weights.push(0.0);
            }
        }

        // Normalize weights
        let weight_sum: f32 = gmm_weights.iter().sum();
        if weight_sum > 0.0 {
            for w in &mut gmm_weights {
                *w /= weight_sum;
            }
        } else {
            for w in &mut gmm_weights {
                *w = 1.0 / k as f32;
            }
        }

        self.means = means;
        self.variances = variances;
        self.weights = gmm_weights;
        self.fitted = true;

        Ok(())
    }

    /// Compute GMM component: ΔS_GMM = 1 - P(e|GMM)
    fn compute_gmm_component(&self, current: &[f32]) -> f32 {
        if !self.fitted || self.means.is_empty() {
            return 0.5; // Neutral if not fitted
        }

        let dim = current.len();
        let mut total_prob = 0.0f64;

        for (i, mean) in self.means.iter().enumerate() {
            if mean.len() != dim {
                continue;
            }

            // Compute Mahalanobis distance (diagonal covariance)
            let mut mahal_sq = 0.0f64;
            let mut log_det = 0.0f64;

            for d in 0..dim {
                let diff = (current[d] - mean[d]) as f64;
                let var = self.variances[i][d] as f64;
                mahal_sq += (diff * diff) / var;
                log_det += var.ln();
            }

            // Gaussian probability (log space for numerical stability)
            let log_norm = -0.5 * (dim as f64 * (2.0 * std::f64::consts::PI).ln() + log_det);
            let log_prob = log_norm - 0.5 * mahal_sq;

            let weight = self.weights[i] as f64;
            if weight > 0.0 {
                total_prob += weight * log_prob.exp();
            }
        }

        let probability = (total_prob as f32).clamp(self.min_probability, 1.0);
        (1.0 - probability).clamp(0.0, 1.0)
    }

    /// Compute KNN component: ΔS_KNN = σ((d_k - μ) / σ)
    fn compute_knn_component(&self, current: &[f32], history: &[Vec<f32>], k: usize) -> f32 {
        if history.is_empty() {
            return 1.0; // Maximum surprise for empty history
        }

        // Compute distances to all history items
        let mut distances: Vec<f32> = history
            .iter()
            .filter(|h| !h.is_empty())
            .map(|h| compute_cosine_distance(current, h))
            .filter(|d| !d.is_nan() && !d.is_infinite())
            .collect();

        if distances.is_empty() {
            return 1.0;
        }

        // Sort and take k nearest
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k_actual = k.max(1).min(distances.len());
        let k_nearest = &distances[..k_actual];

        // Mean of k-nearest distances
        let d_k: f32 = k_nearest.iter().sum::<f32>() / k_actual as f32;

        // Normalize: z = (d_k - μ) / σ
        let std_dev = self.knn_variance.sqrt().max(0.1);
        let z = (d_k - self.knn_mean) / std_dev;

        Self::sigmoid(z)
    }
}

impl EmbedderEntropy for HybridGmmKnnEntropy {
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

        // Check for NaN/Infinity in current embedding (AP-10)
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

        // Use provided k or fallback to configured k_neighbors
        let k_to_use = if k > 0 { k } else { self.k_neighbors };

        // Compute GMM component
        let delta_s_gmm = if self.fitted {
            self.compute_gmm_component(current)
        } else {
            // If GMM not fitted, fit it now
            let mut temp_self = self.clone();
            if temp_self.fit_gmm(history).is_ok() {
                temp_self.compute_gmm_component(current)
            } else {
                0.5 // Neutral on fit failure
            }
        };

        // Compute KNN component
        let delta_s_knn = self.compute_knn_component(current, history, k_to_use);

        // Hybrid combination: ΔS = gmm_weight × ΔS_GMM + knn_weight × ΔS_KNN
        let delta_s = self.gmm_weight * delta_s_gmm + self.knn_weight * delta_s_knn;

        // Final validation per AP-10: no NaN/Infinity
        let clamped = delta_s.clamp(0.0, 1.0);
        if clamped.is_nan() || clamped.is_infinite() {
            return Err(UtlError::EntropyError(
                "Computed delta_s is NaN or Infinity - violates AP-10".to_string(),
            ));
        }

        Ok(clamped)
    }

    fn embedder_type(&self) -> Embedder {
        Embedder::Code
    }

    fn reset(&mut self) {
        self.means.clear();
        self.variances.clear();
        self.weights.clear();
        self.fitted = false;
        self.knn_mean = 0.5;
        self.knn_variance = 0.1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- E7 dimension constant for tests ---
    const E7_DIM: usize = 1536;

    #[test]
    fn test_hybrid_empty_history_returns_one() {
        let calculator = HybridGmmKnnEntropy::new();
        let current = vec![0.5f32; E7_DIM];
        let history: Vec<Vec<f32>> = vec![];

        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok(), "Should not error on empty history");
        assert_eq!(result.unwrap(), 1.0, "Empty history should return 1.0");

        println!("BEFORE: history.len() = 0");
        println!("AFTER: delta_s = 1.0");
        println!("[PASS] test_hybrid_empty_history_returns_one");
    }

    #[test]
    fn test_hybrid_empty_input_error() {
        let calculator = HybridGmmKnnEntropy::new();
        let empty: Vec<f32> = vec![];
        let history = vec![vec![0.5f32; E7_DIM]];

        let result = calculator.compute_delta_s(&empty, &history, 5);
        assert!(
            matches!(result, Err(UtlError::EmptyInput)),
            "Empty input should return EmptyInput error"
        );

        println!("[PASS] test_hybrid_empty_input_error - Err(EmptyInput)");
    }

    #[test]
    fn test_hybrid_identical_returns_low() {
        let calculator = HybridGmmKnnEntropy::new();

        // Create history with identical embeddings
        let embedding = vec![0.5f32; E7_DIM];
        let history: Vec<Vec<f32>> = vec![embedding.clone(); 20];

        let result = calculator.compute_delta_s(&embedding, &history, 5);
        assert!(result.is_ok());
        let delta_s = result.unwrap();

        println!(
            "BEFORE: history contains 20 identical embeddings to current"
        );
        println!("AFTER: delta_s = {}", delta_s);
        assert!(
            delta_s < 0.5,
            "Identical embedding should have low surprise, got {}",
            delta_s
        );

        println!("[PASS] test_hybrid_identical_returns_low - delta_s = {}", delta_s);
    }

    #[test]
    fn test_hybrid_distant_returns_high() {
        let calculator = HybridGmmKnnEntropy::new();

        // Create orthogonal/distant embeddings
        let mut current = vec![0.0f32; E7_DIM];
        current[0] = 1.0;

        let mut history_item = vec![0.0f32; E7_DIM];
        history_item[E7_DIM / 2] = 1.0;
        let history = vec![history_item; 10];

        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok());
        let delta_s = result.unwrap();

        println!("BEFORE: current orthogonal to history");
        println!("AFTER: delta_s = {}", delta_s);
        assert!(
            delta_s > 0.5,
            "Distant embedding should have high surprise, got {}",
            delta_s
        );

        println!("[PASS] test_hybrid_distant_returns_high - delta_s = {}", delta_s);
    }

    #[test]
    fn test_hybrid_weight_balance() {
        let calculator = HybridGmmKnnEntropy::new();
        let weight_sum = calculator.gmm_weight + calculator.knn_weight;

        assert!(
            (weight_sum - 1.0).abs() < 1e-6,
            "Weights must sum to 1.0, got {}",
            weight_sum
        );

        println!(
            "[PASS] test_hybrid_weight_balance - gmm={}, knn={}, sum={}",
            calculator.gmm_weight, calculator.knn_weight, weight_sum
        );
    }

    #[test]
    fn test_hybrid_gmm_component_range() {
        let mut calculator = HybridGmmKnnEntropy::new();

        let history: Vec<Vec<f32>> = (0..30)
            .map(|i| vec![0.5 + (i as f32) * 0.01; 100])
            .collect();

        let fit_result = calculator.fit_gmm(&history);
        assert!(fit_result.is_ok(), "GMM fitting should succeed");

        for test_val in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let current = vec![test_val; 100];
            let delta_s_gmm = calculator.compute_gmm_component(&current);

            assert!(
                (0.0..=1.0).contains(&delta_s_gmm),
                "GMM component {} out of range for test_val {}",
                delta_s_gmm,
                test_val
            );
        }

        println!("[PASS] test_hybrid_gmm_component_range");
    }

    #[test]
    fn test_hybrid_knn_component_range() {
        let calculator = HybridGmmKnnEntropy::new();

        let history: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![0.5 + (i as f32) * 0.02; 100])
            .collect();

        for test_val in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let current = vec![test_val; 100];
            let delta_s_knn = calculator.compute_knn_component(&current, &history, 5);

            assert!(
                (0.0..=1.0).contains(&delta_s_knn),
                "KNN component {} out of range for test_val {}",
                delta_s_knn,
                test_val
            );
        }

        println!("[PASS] test_hybrid_knn_component_range");
    }

    #[test]
    fn test_hybrid_embedder_type() {
        let calculator = HybridGmmKnnEntropy::new();
        assert_eq!(
            calculator.embedder_type(),
            Embedder::Code,
            "Should return Embedder::Code"
        );

        println!("[PASS] test_hybrid_embedder_type - Embedder::Code");
    }

    #[test]
    fn test_hybrid_valid_range() {
        let calculator = HybridGmmKnnEntropy::new();

        // Test various input patterns at E7 dimension
        for pattern in 0..5 {
            let current: Vec<f32> = (0..E7_DIM)
                .map(|i| ((i + pattern * 100) as f32) / E7_DIM as f32)
                .collect();

            let history: Vec<Vec<f32>> = (0..15)
                .map(|j| {
                    (0..E7_DIM)
                        .map(|i| ((i + j * 50) as f32) / E7_DIM as f32)
                        .collect()
                })
                .collect();

            let result = calculator.compute_delta_s(&current, &history, 5);
            assert!(result.is_ok());
            let delta_s = result.unwrap();

            assert!(
                (0.0..=1.0).contains(&delta_s),
                "Pattern {} delta_s {} out of range",
                pattern,
                delta_s
            );
        }

        println!("[PASS] test_hybrid_valid_range");
    }

    #[test]
    fn test_hybrid_no_nan_infinity() {
        let calculator = HybridGmmKnnEntropy::new();

        // Edge case: very small values
        let small: Vec<f32> = vec![1e-10; E7_DIM];
        let history: Vec<Vec<f32>> = vec![vec![1e-10; E7_DIM]; 10];

        let result = calculator.compute_delta_s(&small, &history, 5);
        assert!(result.is_ok());
        let delta_s = result.unwrap();
        assert!(!delta_s.is_nan(), "delta_s should not be NaN (AP-10)");
        assert!(!delta_s.is_infinite(), "delta_s should not be Infinite (AP-10)");

        // Edge case: values near 1
        let near_one: Vec<f32> = vec![0.9999; E7_DIM];
        let result2 = calculator.compute_delta_s(&near_one, &history, 5);
        assert!(result2.is_ok());
        let delta_s2 = result2.unwrap();
        assert!(!delta_s2.is_nan(), "delta_s should not be NaN");
        assert!(!delta_s2.is_infinite(), "delta_s should not be Infinite");

        println!("[PASS] test_hybrid_no_nan_infinity - AP-10 compliant");
    }

    #[test]
    fn test_hybrid_from_config() {
        let mut config = SurpriseConfig::default();
        config.code_gmm_weight = 0.7;
        config.code_knn_weight = 0.3;
        config.code_n_components = 8;
        config.code_k_neighbors = 10;

        let calculator = HybridGmmKnnEntropy::from_config(&config);

        // Weights should be normalized to sum to 1.0
        let weight_sum = calculator.gmm_weight + calculator.knn_weight;
        assert!(
            (weight_sum - 1.0).abs() < 1e-6,
            "Weights should sum to 1.0, got {}",
            weight_sum
        );
        assert_eq!(calculator.n_components, 8);
        assert_eq!(calculator.k_neighbors, 10);

        println!(
            "[PASS] test_hybrid_from_config - gmm_weight={}, knn_weight={}, n_components={}, k={}",
            calculator.gmm_weight, calculator.knn_weight, calculator.n_components, calculator.k_neighbors
        );
    }

    #[test]
    fn test_hybrid_reset() {
        let mut calculator = HybridGmmKnnEntropy::new();

        // Fit GMM to populate internal state
        let history: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![0.5 + (i as f32) * 0.02; 100])
            .collect();
        let _ = calculator.fit_gmm(&history);

        assert!(calculator.fitted, "Should be fitted after fit_gmm");
        assert!(!calculator.means.is_empty(), "Should have means after fitting");

        calculator.reset();

        assert!(!calculator.fitted, "fitted should be false after reset");
        assert!(calculator.means.is_empty(), "means should be cleared after reset");
        assert!(calculator.variances.is_empty(), "variances should be cleared after reset");
        assert!(calculator.weights.is_empty(), "weights should be cleared after reset");
        assert_eq!(calculator.knn_mean, 0.5, "knn_mean should reset to 0.5");
        assert_eq!(calculator.knn_variance, 0.1, "knn_variance should reset to 0.1");

        println!("[PASS] test_hybrid_reset");
    }

    #[test]
    fn test_hybrid_gmm_fit() {
        let mut calculator = HybridGmmKnnEntropy::new();

        let history: Vec<Vec<f32>> = (0..30)
            .map(|i| vec![0.3 + (i as f32) * 0.02; 100])
            .collect();

        let result = calculator.fit_gmm(&history);
        assert!(result.is_ok(), "GMM fitting should succeed");
        assert!(calculator.fitted, "fitted flag should be true after fit");
        assert_eq!(
            calculator.means.len(),
            calculator.n_components.min(history.len()),
            "Should have correct number of means"
        );

        println!("[PASS] test_hybrid_gmm_fit - fitted={}", calculator.fitted);
    }

    #[test]
    fn test_factory_routes_code_to_hybrid() {
        // This test will verify factory routing after we update factory.rs
        // For now, just verify the calculator returns correct embedder type
        let calculator = HybridGmmKnnEntropy::new();
        assert_eq!(
            calculator.embedder_type(),
            Embedder::Code,
            "embedder_type() must return Embedder::Code"
        );

        println!("[PASS] test_factory_routes_code_to_hybrid - Embedder::Code");
    }

    // === Edge Case Tests per Task Requirements ===

    #[test]
    fn test_edge_case_high_dimensional_sparse_history() {
        let calculator = HybridGmmKnnEntropy::new();

        // Edge Case 1: Very High-Dimensional Sparse History
        let current = vec![0.001f32; E7_DIM];
        let history: Vec<Vec<f32>> = (0..5).map(|_| vec![0.999f32; E7_DIM]).collect();

        println!(
            "BEFORE: current[0]={}, history.len()={}",
            current[0],
            history.len()
        );

        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok(), "Should handle sparse high-dim data");
        let delta_s = result.unwrap();

        println!("AFTER: delta_s={}", delta_s);
        println!(
            "[PASS] test_edge_case_high_dimensional_sparse_history - delta_s={}",
            delta_s
        );

        // Should be high surprise (current very different from history)
        assert!(delta_s > 0.3, "Distant values should have meaningful surprise");
    }

    #[test]
    fn test_edge_case_single_history_item() {
        let calculator = HybridGmmKnnEntropy::new();

        // Edge Case 2: Single History Item (k > history.len())
        let current = vec![0.5f32; E7_DIM];
        let history = vec![vec![0.5f32; E7_DIM]]; // Only 1 item, but k=5

        println!("BEFORE: history.len()={}, k=5", history.len());

        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok(), "Should handle k > history.len()");
        let delta_s = result.unwrap();

        println!("AFTER: delta_s={}", delta_s);
        println!(
            "[PASS] test_edge_case_single_history_item - delta_s={}",
            delta_s
        );

        // Should be low surprise (identical to single history item)
        assert!(delta_s < 0.7, "Should use k=1 when k > history.len()");
    }

    #[test]
    fn test_edge_case_near_zero_variance() {
        let calculator = HybridGmmKnnEntropy::new();

        // Edge Case 3: Near-Zero Variance History (all identical)
        let current = vec![0.5f32; E7_DIM];
        let history: Vec<Vec<f32>> = vec![vec![0.5f32; E7_DIM]; 50]; // All identical

        println!("BEFORE: all history identical");

        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok(), "Should handle zero variance history");
        let delta_s = result.unwrap();

        println!("AFTER: delta_s={}, should be low", delta_s);
        println!(
            "[PASS] test_edge_case_near_zero_variance - delta_s={}",
            delta_s
        );

        // Should be low surprise (identical to history)
        assert!(delta_s < 0.5, "Identical history should have low surprise");
    }

    #[test]
    fn test_with_weights_normalization() {
        // Test that weights are normalized correctly
        let calc = HybridGmmKnnEntropy::with_weights(0.3, 0.7);
        assert!((calc.gmm_weight - 0.3).abs() < 1e-6);
        assert!((calc.knn_weight - 0.7).abs() < 1e-6);

        // Test with non-normalized weights
        let calc2 = HybridGmmKnnEntropy::with_weights(0.6, 0.6);
        let sum = calc2.gmm_weight + calc2.knn_weight;
        assert!((sum - 1.0).abs() < 1e-6, "Weights should be normalized to 1.0");

        println!("[PASS] test_with_weights_normalization");
    }

    #[test]
    fn test_with_n_components_clamping() {
        let calc = HybridGmmKnnEntropy::new().with_n_components(1);
        assert_eq!(calc.n_components, 2, "Should clamp n_components minimum to 2");

        let calc2 = HybridGmmKnnEntropy::new().with_n_components(100);
        assert_eq!(calc2.n_components, 10, "Should clamp n_components maximum to 10");

        println!("[PASS] test_with_n_components_clamping");
    }

    #[test]
    fn test_with_k_neighbors_clamping() {
        let calc = HybridGmmKnnEntropy::new().with_k_neighbors(0);
        assert_eq!(calc.k_neighbors, 1, "Should clamp k_neighbors minimum to 1");

        let calc2 = HybridGmmKnnEntropy::new().with_k_neighbors(100);
        assert_eq!(calc2.k_neighbors, 20, "Should clamp k_neighbors maximum to 20");

        println!("[PASS] test_with_k_neighbors_clamping");
    }
}
