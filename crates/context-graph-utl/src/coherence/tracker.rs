//! Coherence tracker implementation.
//!
//! The [`CoherenceTracker`] maintains a rolling window of recent embeddings
//! and computes coherence scores based on the three-component formula.
//!
//! # Constitution Reference
//!
//! Per constitution.yaml line 166:
//! ```text
//! ΔC = 0.4×Connectivity + 0.4×ClusterFit + 0.2×Consistency
//! ```
//!
//! # Components
//!
//! 1. **Connectivity (α=0.4)**: How well connected the vertex is in the graph
//! 2. **ClusterFit (β=0.4)**: Silhouette coefficient measuring cluster membership
//! 3. **Consistency (γ=0.2)**: Temporal stability from rolling window variance
//!
//! EMA smoothing is applied for stability across updates.

use crate::config::CoherenceConfig;
use crate::error::{UtlError, UtlResult};

use super::cluster_fit::{compute_cluster_fit, ClusterContext, ClusterFitConfig, ClusterFitResult};
use super::structural::StructuralCoherenceCalculator;
use super::window::RollingWindow;

/// Result of coherence computation with diagnostics.
///
/// Provides transparency into each component for debugging and monitoring.
///
/// # Constitution Reference
///
/// Per constitution.yaml line 166:
/// ΔC = 0.4×Connectivity + 0.4×ClusterFit + 0.2×Consistency
#[derive(Debug, Clone)]
pub struct CoherenceResult {
    /// Final coherence score [0, 1].
    /// Computed as: α×Connectivity + β×ClusterFit + γ×Consistency
    pub score: f32,

    /// Connectivity component value [0, 1].
    /// Measures graph-based coherence from neighbor relationships.
    pub connectivity: f32,

    /// ClusterFit component value [0, 1].
    /// Normalized silhouette coefficient from cluster membership.
    pub cluster_fit: f32,

    /// Consistency component value [0, 1].
    /// Temporal stability from rolling window variance.
    pub consistency: f32,

    /// ClusterFit detailed result (if available).
    /// Contains raw silhouette, intra/inter distances.
    pub cluster_fit_result: Option<ClusterFitResult>,

    /// Whether fallback was used for any component.
    /// True if ClusterFit or connectivity computation failed.
    pub used_fallback: bool,
}

impl CoherenceResult {
    /// Create a new coherence result.
    pub fn new(
        score: f32,
        connectivity: f32,
        cluster_fit: f32,
        consistency: f32,
        cluster_fit_result: Option<ClusterFitResult>,
        used_fallback: bool,
    ) -> Self {
        Self {
            score: score.clamp(0.0, 1.0),
            connectivity: connectivity.clamp(0.0, 1.0),
            cluster_fit: cluster_fit.clamp(0.0, 1.0),
            consistency: consistency.clamp(0.0, 1.0),
            cluster_fit_result,
            used_fallback,
        }
    }

    /// Create a fallback result with neutral values.
    pub fn fallback() -> Self {
        Self {
            score: 0.5,
            connectivity: 0.5,
            cluster_fit: 0.5,
            consistency: 0.5,
            cluster_fit_result: None,
            used_fallback: true,
        }
    }
}

/// Graph context for connectivity computation.
///
/// Contains neighbor embeddings for structural coherence calculation.
#[derive(Debug, Clone)]
pub struct GraphContext {
    /// Node embedding to evaluate.
    pub node_embedding: Vec<f32>,

    /// Embeddings of neighbor vertices in the graph.
    pub neighbor_embeddings: Vec<Vec<f32>>,
}

impl GraphContext {
    /// Create a new graph context.
    pub fn new(node_embedding: Vec<f32>, neighbor_embeddings: Vec<Vec<f32>>) -> Self {
        Self {
            node_embedding,
            neighbor_embeddings,
        }
    }
}

/// Coherence tracker that maintains history and computes coherence scores.
///
/// Uses a rolling window to track recent embeddings and computes coherence
/// using the three-component formula from constitution.yaml:
///
/// ```text
/// ΔC = α×Connectivity + β×ClusterFit + γ×Consistency
/// ```
///
/// # Constitution Reference
///
/// Per constitution.yaml line 166:
/// - α (connectivity_weight) = 0.4
/// - β (cluster_fit_weight) = 0.4
/// - γ (consistency_weight) = 0.2
///
/// # Example
///
/// ```ignore
/// use context_graph_utl::coherence::{CoherenceTracker, ClusterContext, GraphContext};
/// use context_graph_utl::config::CoherenceConfig;
///
/// let config = CoherenceConfig::default();
/// let mut tracker = CoherenceTracker::new(&config);
///
/// // Create cluster context for ClusterFit
/// let cluster_context = ClusterContext::new(
///     vec![vec![0.5, 0.5], vec![0.6, 0.6]],  // same cluster
///     vec![vec![0.9, 0.1], vec![0.8, 0.2]],  // nearest cluster
/// );
///
/// // Compute coherence
/// let vertex = vec![0.55, 0.55];
/// let connectivity = 0.8;
/// let coherence = tracker.compute_coherence(&vertex, connectivity, &cluster_context);
/// println!("Coherence: {}", coherence);
/// ```
#[derive(Debug, Clone)]
pub struct CoherenceTracker {
    /// Rolling window of recent embeddings.
    window: RollingWindow<Vec<f32>>,

    /// Weight for connectivity component (α). Default: 0.4
    connectivity_weight: f32,

    /// Weight for cluster fit component (β). Default: 0.4
    cluster_fit_weight: f32,

    /// Weight for consistency component (γ). Default: 0.2
    consistency_weight: f32,

    /// Configuration for ClusterFit calculation.
    cluster_fit_config: ClusterFitConfig,

    /// Structural coherence calculator for connectivity.
    structural_calculator: StructuralCoherenceCalculator,

    /// Minimum coherence threshold.
    min_threshold: f32,

    /// EMA smoothed coherence value.
    ema_coherence: Option<f32>,

    /// EMA smoothing factor (alpha).
    ema_alpha: f32,

    /// Whether to use EMA smoothing.
    use_ema: bool,

    /// Number of neighbors to consider.
    neighbor_count: usize,

    /// Weight for semantic similarity contribution (legacy).
    similarity_weight: f32,
}

impl CoherenceTracker {
    /// Create a new coherence tracker with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Coherence configuration settings.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use context_graph_utl::coherence::CoherenceTracker;
    /// use context_graph_utl::config::CoherenceConfig;
    ///
    /// let config = CoherenceConfig::default();
    /// let tracker = CoherenceTracker::new(&config);
    ///
    /// // Weights are per constitution.yaml line 166
    /// assert_eq!(tracker.connectivity_weight(), 0.4);
    /// assert_eq!(tracker.cluster_fit_weight(), 0.4);
    /// assert_eq!(tracker.consistency_weight(), 0.2);
    /// ```
    pub fn new(config: &CoherenceConfig) -> Self {
        Self {
            window: RollingWindow::new(config.neighbor_count.max(10)),
            connectivity_weight: config.connectivity_weight,
            cluster_fit_weight: config.cluster_fit_weight,
            consistency_weight: config.consistency_weight,
            cluster_fit_config: config.cluster_fit.clone(),
            structural_calculator: StructuralCoherenceCalculator::default(),
            min_threshold: config.min_threshold,
            ema_coherence: None,
            ema_alpha: 0.3, // Default EMA alpha
            use_ema: true,
            neighbor_count: config.neighbor_count,
            similarity_weight: config.similarity_weight,
        }
    }

    /// Create a tracker with custom EMA settings.
    ///
    /// # Arguments
    ///
    /// * `config` - Coherence configuration settings.
    /// * `ema_alpha` - EMA smoothing factor (0 = no smoothing, 1 = no memory).
    pub fn with_ema(config: &CoherenceConfig, ema_alpha: f32) -> Self {
        let mut tracker = Self::new(config);
        tracker.ema_alpha = ema_alpha.clamp(0.0, 1.0);
        tracker
    }

    /// Disable EMA smoothing.
    pub fn without_ema(config: &CoherenceConfig) -> Self {
        let mut tracker = Self::new(config);
        tracker.use_ema = false;
        tracker
    }

    /// Compute coherence (Delta-C) for a vertex using three-component formula.
    ///
    /// Uses the constitution formula:
    /// ```text
    /// ΔC = α×Connectivity + β×ClusterFit + γ×Consistency
    /// ```
    ///
    /// # Arguments
    ///
    /// * `vertex` - The embedding vector to evaluate
    /// * `connectivity` - Pre-computed connectivity score from StructuralCoherenceCalculator
    /// * `cluster_context` - Cluster context for ClusterFit computation
    ///
    /// # Returns
    ///
    /// Coherence score in [0, 1]
    ///
    /// # Edge Cases
    ///
    /// - If ClusterFit fails, uses fallback (0.5) and logs warning
    /// - If connectivity is NaN, uses fallback (0.5)
    /// - Per AP-10: No NaN/Infinity in output
    ///
    /// # Example
    ///
    /// ```ignore
    /// use context_graph_utl::coherence::{CoherenceTracker, ClusterContext};
    /// use context_graph_utl::config::CoherenceConfig;
    ///
    /// let config = CoherenceConfig::default();
    /// let mut tracker = CoherenceTracker::new(&config);
    ///
    /// let vertex = vec![0.55, 0.55, 0.55];
    /// let connectivity = 0.8;
    /// let cluster_context = ClusterContext::new(
    ///     vec![vec![0.5, 0.5, 0.5], vec![0.6, 0.6, 0.6]],
    ///     vec![vec![0.9, 0.9, 0.9]],
    /// );
    ///
    /// let coherence = tracker.compute_coherence(&vertex, connectivity, &cluster_context);
    /// assert!(coherence >= 0.0 && coherence <= 1.0);
    /// ```
    pub fn compute_coherence(
        &mut self,
        vertex: &[f32],
        connectivity: f32,
        cluster_context: &ClusterContext,
    ) -> f32 {
        // 1. Compute ClusterFit using the silhouette coefficient
        let cluster_fit_result = compute_cluster_fit(vertex, cluster_context, &self.cluster_fit_config);
        let cluster_fit = cluster_fit_result.score;

        // 2. Get consistency from rolling window
        let consistency = self.compute_consistency();

        // 3. Validate connectivity (AP-10: no NaN/Inf)
        let connectivity = if connectivity.is_nan() || connectivity.is_infinite() {
            tracing::warn!("Connectivity is NaN/Inf, using fallback 0.5");
            0.5
        } else {
            connectivity.clamp(0.0, 1.0)
        };

        // 4. Apply three-component formula per constitution.yaml line 166:
        // ΔC = α×Connectivity + β×ClusterFit + γ×Consistency
        let coherence = self.connectivity_weight * connectivity
            + self.cluster_fit_weight * cluster_fit
            + self.consistency_weight * consistency;

        // 5. Clamp and return (AP-10: no NaN/Inf)
        let result = coherence.clamp(0.0, 1.0);

        // Ensure no NaN/Inf in output per AP-10
        if result.is_nan() || result.is_infinite() {
            0.5
        } else {
            result
        }
    }

    /// Compute coherence with all components computed internally.
    ///
    /// Convenience method that computes all three components and returns
    /// detailed diagnostics.
    ///
    /// # Arguments
    ///
    /// * `vertex` - The vertex embedding
    /// * `graph_context` - Graph context for connectivity computation
    /// * `cluster_context` - Cluster context for ClusterFit
    ///
    /// # Returns
    ///
    /// `CoherenceResult` with score and all component values
    ///
    /// # Example
    ///
    /// ```ignore
    /// use context_graph_utl::coherence::{CoherenceTracker, ClusterContext, GraphContext};
    /// use context_graph_utl::config::CoherenceConfig;
    ///
    /// let config = CoherenceConfig::default();
    /// let mut tracker = CoherenceTracker::new(&config);
    ///
    /// let vertex = vec![0.55, 0.55, 0.55];
    /// let graph_context = GraphContext::new(
    ///     vertex.clone(),
    ///     vec![vec![0.5, 0.5, 0.5], vec![0.6, 0.4, 0.5]],
    /// );
    /// let cluster_context = ClusterContext::new(
    ///     vec![vec![0.5, 0.5, 0.5], vec![0.6, 0.6, 0.6]],
    ///     vec![vec![0.9, 0.9, 0.9]],
    /// );
    ///
    /// let result = tracker.compute_coherence_full(&vertex, &graph_context, &cluster_context);
    /// println!("Score: {}, Connectivity: {}, ClusterFit: {}, Consistency: {}",
    ///          result.score, result.connectivity, result.cluster_fit, result.consistency);
    /// ```
    pub fn compute_coherence_full(
        &mut self,
        vertex: &[f32],
        graph_context: &GraphContext,
        cluster_context: &ClusterContext,
    ) -> CoherenceResult {
        let mut used_fallback = false;

        // 1. Compute connectivity from structural calculator
        let connectivity = self.structural_calculator.compute(
            &graph_context.node_embedding,
            &graph_context.neighbor_embeddings,
        );

        // Validate connectivity (AP-10)
        let connectivity = if connectivity.is_nan() || connectivity.is_infinite() {
            tracing::warn!("Connectivity computation returned NaN/Inf, using fallback");
            used_fallback = true;
            0.5
        } else {
            connectivity.clamp(0.0, 1.0)
        };

        // 2. Compute cluster fit
        let cf_result = compute_cluster_fit(vertex, cluster_context, &self.cluster_fit_config);
        let cluster_fit = cf_result.score;

        // Check if cluster fit used fallback (score = fallback_value and silhouette = 0)
        if (cf_result.score - self.cluster_fit_config.fallback_value).abs() < 0.001
            && cf_result.silhouette.abs() < 0.001
            && cf_result.intra_distance.abs() < 0.001
        {
            used_fallback = true;
        }

        // 3. Compute consistency from rolling window
        let consistency = self.compute_consistency();

        // 4. Apply formula: ΔC = α×Connectivity + β×ClusterFit + γ×Consistency
        let score = (self.connectivity_weight * connectivity
            + self.cluster_fit_weight * cluster_fit
            + self.consistency_weight * consistency)
            .clamp(0.0, 1.0);

        // Ensure no NaN/Inf per AP-10
        let score = if score.is_nan() || score.is_infinite() {
            used_fallback = true;
            0.5
        } else {
            score
        };

        // 5. Update rolling window for future consistency calculations
        self.update(vertex);

        CoherenceResult::new(
            score,
            connectivity,
            cluster_fit,
            consistency,
            Some(cf_result),
            used_fallback,
        )
    }

    /// Get the cluster fit configuration.
    pub fn cluster_fit_config(&self) -> &ClusterFitConfig {
        &self.cluster_fit_config
    }

    /// Get the connectivity weight (α).
    pub fn connectivity_weight(&self) -> f32 {
        self.connectivity_weight
    }

    /// Get the cluster fit weight (β).
    pub fn cluster_fit_weight(&self) -> f32 {
        self.cluster_fit_weight
    }

    /// Get the consistency weight (γ).
    pub fn consistency_weight(&self) -> f32 {
        self.consistency_weight
    }

    /// Update the component weights.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Connectivity weight
    /// * `beta` - ClusterFit weight
    /// * `gamma` - Consistency weight
    ///
    /// # Panics
    ///
    /// Panics if weights do not sum to approximately 1.0 (tolerance: 0.01)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use context_graph_utl::coherence::CoherenceTracker;
    /// use context_graph_utl::config::CoherenceConfig;
    ///
    /// let config = CoherenceConfig::default();
    /// let mut tracker = CoherenceTracker::new(&config);
    ///
    /// // Custom weights that sum to 1.0
    /// tracker.set_weights(0.5, 0.3, 0.2);
    /// assert_eq!(tracker.connectivity_weight(), 0.5);
    /// assert_eq!(tracker.cluster_fit_weight(), 0.3);
    /// assert_eq!(tracker.consistency_weight(), 0.2);
    /// ```
    pub fn set_weights(&mut self, alpha: f32, beta: f32, gamma: f32) {
        let sum = alpha + beta + gamma;
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Weights must sum to 1.0, got {} (alpha={}, beta={}, gamma={})",
            sum,
            alpha,
            beta,
            gamma
        );

        self.connectivity_weight = alpha;
        self.cluster_fit_weight = beta;
        self.consistency_weight = gamma;
    }

    // ========================================================================
    // Legacy methods for backward compatibility
    // ========================================================================

    /// Compute coherence score for the current embedding against history.
    ///
    /// This is the legacy method that uses similarity-based coherence.
    /// For the three-component formula, use [`compute_coherence`] instead.
    ///
    /// # Arguments
    ///
    /// * `current` - The current embedding vector.
    /// * `history` - Historical embedding vectors for comparison.
    ///
    /// # Returns
    ///
    /// A coherence score in the range `[0, 1]`.
    pub fn compute_coherence_legacy(&self, current: &[f32], history: &[Vec<f32>]) -> f32 {
        if history.is_empty() {
            // No history - return minimum threshold (not completely incoherent)
            return self.min_threshold;
        }

        // Compute average similarity with history
        let avg_similarity = self.compute_average_similarity(current, history);

        // Compute consistency from the window (if available)
        let consistency = self.compute_consistency();

        // Combine with weights (legacy formula)
        let raw_coherence =
            (self.similarity_weight * avg_similarity) + (self.consistency_weight * consistency);

        // Normalize and clamp
        let normalized = raw_coherence / (self.similarity_weight + self.consistency_weight);
        normalized.clamp(0.0, 1.0)
    }

    /// Update the rolling window with a new embedding.
    ///
    /// This should be called after processing each new embedding to maintain
    /// an accurate history for coherence computation.
    ///
    /// # Arguments
    ///
    /// * `embedding` - The embedding vector to add to history.
    pub fn update(&mut self, embedding: &[f32]) {
        self.window.push(embedding.to_vec());
    }

    /// Update with a new embedding and compute coherence in one step.
    ///
    /// This is a convenience method that updates the window and computes
    /// coherence against the current window contents.
    ///
    /// # Arguments
    ///
    /// * `embedding` - The new embedding vector.
    ///
    /// # Returns
    ///
    /// The coherence score for the new embedding (using legacy formula).
    pub fn update_and_compute(&mut self, embedding: &[f32]) -> f32 {
        let history: Vec<Vec<f32>> = self.window.to_vec();
        let coherence = self.compute_coherence_legacy(embedding, &history);

        // Apply EMA smoothing
        let smoothed = if self.use_ema {
            self.apply_ema(coherence)
        } else {
            coherence
        };

        // Update window after computing coherence
        self.update(embedding);

        smoothed
    }

    /// Get the EMA-smoothed coherence value.
    ///
    /// # Returns
    ///
    /// `Some(coherence)` if at least one update has been processed,
    /// `None` otherwise.
    pub fn smoothed_coherence(&self) -> Option<f32> {
        self.ema_coherence
    }

    /// Get the number of embeddings in the history window.
    pub fn history_len(&self) -> usize {
        self.window.len()
    }

    /// Check if the tracker has sufficient history for reliable coherence.
    ///
    /// Returns `true` if at least 2 embeddings are in history.
    pub fn has_sufficient_history(&self) -> bool {
        self.window.len() >= 2
    }

    /// Clear all history from the tracker.
    pub fn clear(&mut self) {
        self.window.clear();
        self.ema_coherence = None;
    }

    /// Compute average cosine similarity with history.
    fn compute_average_similarity(&self, current: &[f32], history: &[Vec<f32>]) -> f32 {
        if history.is_empty() {
            return 0.0;
        }

        // Take only the most recent neighbors
        let neighbors: Vec<_> = history.iter().rev().take(self.neighbor_count).collect();

        if neighbors.is_empty() {
            return 0.0;
        }

        let total_similarity: f32 = neighbors
            .iter()
            .map(|h| cosine_similarity(current, h))
            .sum();

        let avg = total_similarity / neighbors.len() as f32;

        // Convert similarity [-1, 1] to coherence [0, 1]
        (avg + 1.0) / 2.0
    }

    /// Compute consistency score from the rolling window.
    fn compute_consistency(&self) -> f32 {
        if self.window.len() < 2 {
            // Not enough data for consistency - return neutral value
            return 0.5;
        }

        // Compute variance of embeddings
        let variance_vec = self.window.variance_vec();

        match variance_vec {
            Some(variances) => {
                // Average variance across dimensions
                let avg_variance: f32 = variances.iter().sum::<f32>() / variances.len() as f32;

                // Convert variance to consistency: high variance = low consistency
                // Use exponential decay: consistency = exp(-k * variance)
                let k = 5.0; // Decay constant
                let consistency = (-k * avg_variance).exp();

                consistency.clamp(0.0, 1.0)
            }
            None => 0.5, // Return neutral if variance computation fails
        }
    }

    /// Apply EMA smoothing to a coherence value.
    fn apply_ema(&mut self, coherence: f32) -> f32 {
        let smoothed = match self.ema_coherence {
            Some(prev) => self.ema_alpha * coherence + (1.0 - self.ema_alpha) * prev,
            None => coherence,
        };

        self.ema_coherence = Some(smoothed);
        smoothed
    }
}

/// Compute cosine similarity between two vectors.
///
/// # Arguments
///
/// * `a` - First vector.
/// * `b` - Second vector.
///
/// # Returns
///
/// Cosine similarity in the range `[-1, 1]`.
/// Returns 0.0 if either vector has zero magnitude.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
}

/// Validate coherence computation inputs.
///
/// # Arguments
///
/// * `current` - Current embedding to validate.
/// * `history` - Historical embeddings to validate.
///
/// # Returns
///
/// `Ok(())` if inputs are valid, `Err` with description otherwise.
#[allow(dead_code)]
pub fn validate_inputs(current: &[f32], history: &[Vec<f32>]) -> UtlResult<()> {
    if current.is_empty() {
        return Err(UtlError::EmptyInput);
    }

    let dim = current.len();

    for h in history.iter() {
        if h.len() != dim {
            return Err(UtlError::DimensionMismatch {
                expected: dim,
                actual: h.len(),
            });
        }
    }

    // Check for NaN/Inf values
    for &val in current {
        if val.is_nan() || val.is_infinite() {
            return Err(UtlError::CoherenceError(
                "Current embedding contains NaN or Inf".to_string(),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_tracker() -> CoherenceTracker {
        let config = CoherenceConfig::default();
        CoherenceTracker::new(&config)
    }

    // ========================================================================
    // Three-Component Formula Tests (TASK-UTL-P1-008)
    // ========================================================================

    #[test]
    fn test_coherence_three_component_formula() {
        let config = CoherenceConfig::default();
        let mut tracker = CoherenceTracker::new(&config);

        // Create test cluster context with well-clustered point
        let cluster_context = ClusterContext::new(
            vec![
                vec![0.5, 0.5, 0.5, 0.5],
                vec![0.6, 0.6, 0.4, 0.4],
            ],
            vec![
                vec![0.9, 0.1, 0.0, 0.0],
                vec![0.85, 0.15, 0.0, 0.0],
            ],
        );

        let vertex = vec![0.55, 0.55, 0.45, 0.45];
        let connectivity = 0.8;

        let coherence = tracker.compute_coherence(&vertex, connectivity, &cluster_context);

        // Verify result is in valid range
        assert!((0.0..=1.0).contains(&coherence));

        // Verify formula: ΔC = 0.4×Connectivity + 0.4×ClusterFit + 0.2×Consistency
        // With connectivity = 0.8, expect contribution of 0.4 * 0.8 = 0.32
        // ClusterFit should be high (well-clustered), ~0.7-0.9
        // Consistency starts at 0.5 (neutral), contribution = 0.2 * 0.5 = 0.1
        // Expected: 0.32 + 0.4*[0.7-0.9] + 0.1 = 0.70-0.78
        assert!(
            coherence > 0.5,
            "Expected coherence > 0.5 for well-clustered point, got {}",
            coherence
        );
    }

    #[test]
    fn test_coherence_default_weights() {
        let config = CoherenceConfig::default();
        let tracker = CoherenceTracker::new(&config);

        // Verify default weights per constitution.yaml line 166
        assert!(
            (tracker.connectivity_weight() - 0.4).abs() < 0.001,
            "Expected connectivity_weight = 0.4, got {}",
            tracker.connectivity_weight()
        );
        assert!(
            (tracker.cluster_fit_weight() - 0.4).abs() < 0.001,
            "Expected cluster_fit_weight = 0.4, got {}",
            tracker.cluster_fit_weight()
        );
        assert!(
            (tracker.consistency_weight() - 0.2).abs() < 0.001,
            "Expected consistency_weight = 0.2, got {}",
            tracker.consistency_weight()
        );

        // Verify weights sum to 1.0
        let sum = tracker.connectivity_weight()
            + tracker.cluster_fit_weight()
            + tracker.consistency_weight();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Weights should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_coherence_custom_weights() {
        let config = CoherenceConfig::default();
        let mut tracker = CoherenceTracker::new(&config);

        // Set custom weights that sum to 1.0
        tracker.set_weights(0.5, 0.3, 0.2);

        assert!(
            (tracker.connectivity_weight() - 0.5).abs() < 0.001,
            "Expected connectivity_weight = 0.5"
        );
        assert!(
            (tracker.cluster_fit_weight() - 0.3).abs() < 0.001,
            "Expected cluster_fit_weight = 0.3"
        );
        assert!(
            (tracker.consistency_weight() - 0.2).abs() < 0.001,
            "Expected consistency_weight = 0.2"
        );
    }

    #[test]
    #[should_panic(expected = "Weights must sum to 1.0")]
    fn test_coherence_weights_sum_assertion() {
        let config = CoherenceConfig::default();
        let mut tracker = CoherenceTracker::new(&config);

        // This should panic - weights don't sum to 1.0
        tracker.set_weights(0.5, 0.5, 0.5);
    }

    #[test]
    fn test_coherence_cluster_fit_fallback() {
        let config = CoherenceConfig::default();
        let mut tracker = CoherenceTracker::new(&config);

        // Empty cluster context should trigger fallback
        let cluster_context = ClusterContext::new(vec![], vec![]);

        let vertex = vec![0.5, 0.5, 0.5, 0.5];
        let connectivity = 0.8;

        let coherence = tracker.compute_coherence(&vertex, connectivity, &cluster_context);

        // Should still return valid result with fallback for ClusterFit
        assert!((0.0..=1.0).contains(&coherence));

        // With fallback ClusterFit = 0.5, connectivity = 0.8, consistency = 0.5
        // Expected: 0.4*0.8 + 0.4*0.5 + 0.2*0.5 = 0.32 + 0.2 + 0.1 = 0.62
        assert!(
            (coherence - 0.62).abs() < 0.1,
            "Expected coherence near 0.62 with fallback, got {}",
            coherence
        );
    }

    #[test]
    fn test_coherence_connectivity_nan_fallback() {
        let config = CoherenceConfig::default();
        let mut tracker = CoherenceTracker::new(&config);

        let cluster_context = ClusterContext::new(
            vec![vec![0.5, 0.5], vec![0.6, 0.6]],
            vec![vec![0.9, 0.1]],
        );

        let vertex = vec![0.55, 0.55];

        // Pass NaN connectivity
        let coherence = tracker.compute_coherence(&vertex, f32::NAN, &cluster_context);

        // Should still return valid result with fallback for connectivity
        assert!((0.0..=1.0).contains(&coherence));
        assert!(
            !coherence.is_nan(),
            "Output should not be NaN per AP-10"
        );

        // Test with infinity
        let coherence_inf = tracker.compute_coherence(&vertex, f32::INFINITY, &cluster_context);
        assert!((0.0..=1.0).contains(&coherence_inf));
        assert!(
            !coherence_inf.is_infinite(),
            "Output should not be Inf per AP-10"
        );
    }

    #[test]
    fn test_coherence_all_zeros() {
        let config = CoherenceConfig::default();
        let mut tracker = CoherenceTracker::new(&config);

        // All zero embeddings
        let cluster_context = ClusterContext::new(
            vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]],
            vec![vec![0.0, 0.0, 0.0]],
        );

        let vertex = vec![0.0, 0.0, 0.0];
        let connectivity = 0.0;

        let coherence = tracker.compute_coherence(&vertex, connectivity, &cluster_context);

        // Should return valid result (fallback will be used)
        assert!((0.0..=1.0).contains(&coherence));
    }

    #[test]
    fn test_coherence_all_ones() {
        let config = CoherenceConfig::default();
        let mut tracker = CoherenceTracker::new(&config);

        // Perfectly clustered with all ones
        let cluster_context = ClusterContext::new(
            vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]],
            vec![vec![-1.0, -1.0, -1.0]], // Opposite cluster
        );

        let vertex = vec![1.0, 1.0, 1.0];
        let connectivity = 1.0;

        let coherence = tracker.compute_coherence(&vertex, connectivity, &cluster_context);

        // All components at max should give high coherence
        assert!((0.0..=1.0).contains(&coherence));
        assert!(
            coherence > 0.8,
            "Expected high coherence with all max values, got {}",
            coherence
        );
    }

    #[test]
    fn test_coherence_result_components() {
        let config = CoherenceConfig::default();
        let mut tracker = CoherenceTracker::new(&config);

        let vertex = vec![0.55, 0.55, 0.55];
        let graph_context = GraphContext::new(
            vertex.clone(),
            vec![vec![0.5, 0.5, 0.5], vec![0.6, 0.4, 0.5]],
        );
        let cluster_context = ClusterContext::new(
            vec![vec![0.5, 0.5, 0.5], vec![0.6, 0.6, 0.6]],
            vec![vec![0.9, 0.1, 0.0]],
        );

        let result = tracker.compute_coherence_full(&vertex, &graph_context, &cluster_context);

        // Verify all components are present and in valid range
        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert!(result.connectivity >= 0.0 && result.connectivity <= 1.0);
        assert!(result.cluster_fit >= 0.0 && result.cluster_fit <= 1.0);
        assert!(result.consistency >= 0.0 && result.consistency <= 1.0);

        // Verify ClusterFit result is present
        assert!(result.cluster_fit_result.is_some());

        // Verify formula: score = α×connectivity + β×cluster_fit + γ×consistency
        let expected_score = 0.4 * result.connectivity
            + 0.4 * result.cluster_fit
            + 0.2 * result.consistency;
        assert!(
            (result.score - expected_score).abs() < 0.001,
            "Score {} doesn't match formula result {}",
            result.score,
            expected_score
        );
    }

    #[test]
    fn test_coherence_integration() {
        let config = CoherenceConfig::default();
        let mut tracker = CoherenceTracker::new(&config);

        // Create test cluster context with well-separated clusters
        let cluster_context = ClusterContext::new(
            vec![
                vec![0.5, 0.5, 0.5],
                vec![0.6, 0.6, 0.6],
            ],
            vec![
                vec![0.9, 0.9, 0.9],
                vec![0.95, 0.95, 0.95],
            ],
        );

        let vertex = vec![0.55, 0.55, 0.55];
        let connectivity = 0.8;

        let coherence = tracker.compute_coherence(&vertex, connectivity, &cluster_context);

        // Verify result is in valid range
        assert!((0.0..=1.0).contains(&coherence));

        // With vertex well-clustered:
        // - connectivity = 0.8 (provided)
        // - cluster_fit should be high (~0.7-0.9) since vertex is close to same_cluster
        // - consistency = 0.5 (neutral, no history yet)
        // Expected: 0.4*0.8 + 0.4*[0.7-0.9] + 0.2*0.5 = ~0.70-0.78
        assert!(
            (coherence - 0.70).abs() < 0.15,
            "Coherence {} not near expected ~0.70",
            coherence
        );
    }

    #[test]
    fn test_coherence_misclassified_point() {
        let config = CoherenceConfig::default();
        let mut tracker = CoherenceTracker::new(&config);

        // Vertex closer to nearest_cluster than same_cluster (misclassified)
        let cluster_context = ClusterContext::new(
            vec![
                vec![0.0, 0.0, 0.9, 0.1],  // Far from query
                vec![0.0, 0.0, 0.8, 0.2],  // Far from query
            ],
            vec![
                vec![0.85, 0.15, 0.0, 0.0],  // Close to query
                vec![0.88, 0.12, 0.0, 0.0],  // Close to query
            ],
        );

        let vertex = vec![0.9, 0.1, 0.0, 0.0];  // Closer to nearest_cluster
        let connectivity = 0.5;

        let coherence = tracker.compute_coherence(&vertex, connectivity, &cluster_context);

        // Verify result is in valid range
        assert!((0.0..=1.0).contains(&coherence));

        // Misclassified point should have lower ClusterFit (silhouette < 0)
        // This means cluster_fit score < 0.5 (since score = (silhouette + 1) / 2)
        // Total coherence should be lower
        assert!(
            coherence < 0.6,
            "Expected lower coherence for misclassified point, got {}",
            coherence
        );
    }

    // ========================================================================
    // Legacy Tests (preserved for backward compatibility)
    // ========================================================================

    #[test]
    fn test_tracker_new() {
        let config = CoherenceConfig::default();
        let tracker = CoherenceTracker::new(&config);

        assert_eq!(tracker.similarity_weight, config.similarity_weight);
        assert_eq!(tracker.consistency_weight, config.consistency_weight);
        assert!(tracker.window.is_empty());
    }

    #[test]
    fn test_tracker_with_ema() {
        let config = CoherenceConfig::default();
        let tracker = CoherenceTracker::with_ema(&config, 0.5);
        assert_eq!(tracker.ema_alpha, 0.5);
        assert!(tracker.use_ema);
    }

    #[test]
    fn test_tracker_without_ema() {
        let config = CoherenceConfig::default();
        let tracker = CoherenceTracker::without_ema(&config);
        assert!(!tracker.use_ema);
    }

    #[test]
    fn test_compute_coherence_empty_history() {
        let tracker = default_tracker();
        let current = vec![0.1, 0.2, 0.3, 0.4];

        let coherence = tracker.compute_coherence_legacy(&current, &[]);
        assert_eq!(coherence, tracker.min_threshold);
    }

    #[test]
    fn test_compute_coherence_single_history() {
        let tracker = default_tracker();
        let current = vec![0.1, 0.2, 0.3, 0.4];
        let history = vec![vec![0.1, 0.2, 0.3, 0.4]]; // Identical

        let coherence = tracker.compute_coherence_legacy(&current, &history);
        assert!((0.0..=1.0).contains(&coherence));
        // Identical vectors should have high coherence
        assert!(
            coherence > 0.7,
            "Expected high coherence, got {}",
            coherence
        );
    }

    #[test]
    fn test_compute_coherence_similar_history() {
        let tracker = default_tracker();
        let current = vec![0.1, 0.2, 0.3, 0.4];
        let history = vec![
            vec![0.12, 0.22, 0.28, 0.38],
            vec![0.11, 0.21, 0.29, 0.39],
            vec![0.09, 0.19, 0.31, 0.41],
        ];

        let coherence = tracker.compute_coherence_legacy(&current, &history);
        assert!((0.0..=1.0).contains(&coherence));
        // Similar vectors should have high coherence
        assert!(
            coherence > 0.6,
            "Expected high coherence for similar embeddings"
        );
    }

    #[test]
    fn test_compute_coherence_dissimilar_history() {
        let tracker = default_tracker();
        let current = vec![0.1, 0.2, 0.3, 0.4];
        let history = vec![vec![-0.4, -0.3, -0.2, -0.1], vec![0.9, 0.1, 0.0, 0.0]];

        let coherence = tracker.compute_coherence_legacy(&current, &history);
        assert!((0.0..=1.0).contains(&coherence));
        // Dissimilar vectors should have lower coherence
        assert!(
            coherence < 0.7,
            "Expected lower coherence for dissimilar embeddings"
        );
    }

    #[test]
    fn test_update() {
        let mut tracker = default_tracker();
        assert_eq!(tracker.history_len(), 0);

        tracker.update(&[0.1, 0.2, 0.3]);
        assert_eq!(tracker.history_len(), 1);

        tracker.update(&[0.2, 0.3, 0.4]);
        assert_eq!(tracker.history_len(), 2);
    }

    #[test]
    fn test_update_and_compute() {
        let mut tracker = default_tracker();

        // First update - no history yet
        let c1 = tracker.update_and_compute(&[0.1, 0.2, 0.3, 0.4]);
        assert!((0.0..=1.0).contains(&c1));
        assert_eq!(tracker.history_len(), 1);

        // Second update - has history
        let c2 = tracker.update_and_compute(&[0.12, 0.22, 0.28, 0.38]);
        assert!((0.0..=1.0).contains(&c2));
        assert_eq!(tracker.history_len(), 2);

        // EMA should be applied
        assert!(tracker.smoothed_coherence().is_some());
    }

    #[test]
    fn test_has_sufficient_history() {
        let mut tracker = default_tracker();
        assert!(!tracker.has_sufficient_history());

        tracker.update(&[0.1, 0.2, 0.3]);
        assert!(!tracker.has_sufficient_history());

        tracker.update(&[0.2, 0.3, 0.4]);
        assert!(tracker.has_sufficient_history());
    }

    #[test]
    fn test_clear() {
        let mut tracker = default_tracker();
        tracker.update(&[0.1, 0.2, 0.3]);
        tracker.update(&[0.2, 0.3, 0.4]);
        tracker.update_and_compute(&[0.3, 0.4, 0.5]);

        assert!(tracker.history_len() > 0);
        assert!(tracker.smoothed_coherence().is_some());

        tracker.clear();
        assert_eq!(tracker.history_len(), 0);
        assert!(tracker.smoothed_coherence().is_none());
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_mismatched() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_validate_inputs_valid() {
        let current = vec![0.1, 0.2, 0.3];
        let history = vec![vec![0.1, 0.2, 0.3], vec![0.2, 0.3, 0.4]];

        assert!(validate_inputs(&current, &history).is_ok());
    }

    #[test]
    fn test_validate_inputs_empty_current() {
        let current: Vec<f32> = vec![];
        let history: Vec<Vec<f32>> = vec![];

        assert!(matches!(
            validate_inputs(&current, &history),
            Err(UtlError::EmptyInput)
        ));
    }

    #[test]
    fn test_validate_inputs_dimension_mismatch() {
        let current = vec![0.1, 0.2, 0.3];
        let history = vec![vec![0.1, 0.2]]; // Different dimension

        assert!(matches!(
            validate_inputs(&current, &history),
            Err(UtlError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_validate_inputs_nan() {
        let current = vec![0.1, f32::NAN, 0.3];
        let history: Vec<Vec<f32>> = vec![];

        assert!(matches!(
            validate_inputs(&current, &history),
            Err(UtlError::CoherenceError(_))
        ));
    }

    #[test]
    fn test_ema_smoothing() {
        let config = CoherenceConfig::default();
        let mut tracker = CoherenceTracker::with_ema(&config, 0.5);

        // Add initial data
        tracker.update(&[0.1, 0.2, 0.3, 0.4]);

        // First computation
        let _c1 = tracker.update_and_compute(&[0.1, 0.2, 0.3, 0.4]);

        // Second computation with different data
        let _c2 = tracker.update_and_compute(&[0.5, 0.6, 0.7, 0.8]);

        // EMA should smooth the transition
        let smoothed = tracker.smoothed_coherence().unwrap();
        assert!((0.0..=1.0).contains(&smoothed));

        // The smoothed value should be between the raw values
        // (approximately, due to EMA)
    }

    #[test]
    fn test_graph_context_new() {
        let node_emb = vec![0.5, 0.5, 0.5];
        let neighbor_embs = vec![vec![0.4, 0.4, 0.4], vec![0.6, 0.6, 0.6]];

        let ctx = GraphContext::new(node_emb.clone(), neighbor_embs.clone());

        assert_eq!(ctx.node_embedding, node_emb);
        assert_eq!(ctx.neighbor_embeddings, neighbor_embs);
    }

    #[test]
    fn test_coherence_result_new() {
        let result = CoherenceResult::new(0.75, 0.8, 0.7, 0.6, None, false);

        assert_eq!(result.score, 0.75);
        assert_eq!(result.connectivity, 0.8);
        assert_eq!(result.cluster_fit, 0.7);
        assert_eq!(result.consistency, 0.6);
        assert!(result.cluster_fit_result.is_none());
        assert!(!result.used_fallback);
    }

    #[test]
    fn test_coherence_result_clamping() {
        // Test that values are clamped to [0, 1]
        let result = CoherenceResult::new(1.5, -0.1, 2.0, -0.5, None, false);

        assert_eq!(result.score, 1.0);
        assert_eq!(result.connectivity, 0.0);
        assert_eq!(result.cluster_fit, 1.0);
        assert_eq!(result.consistency, 0.0);
    }

    #[test]
    fn test_coherence_result_fallback() {
        let result = CoherenceResult::fallback();

        assert_eq!(result.score, 0.5);
        assert_eq!(result.connectivity, 0.5);
        assert_eq!(result.cluster_fit, 0.5);
        assert_eq!(result.consistency, 0.5);
        assert!(result.cluster_fit_result.is_none());
        assert!(result.used_fallback);
    }
}
