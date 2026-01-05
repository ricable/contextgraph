//! Coherence tracker implementation.
//!
//! The [`CoherenceTracker`] maintains a rolling window of recent embeddings
//! and computes coherence scores based on semantic similarity and consistency
//! over time.
//!
//! # Algorithm
//!
//! Coherence is computed as a weighted combination of:
//! 1. **Semantic similarity**: How similar the current embedding is to recent history
//! 2. **Consistency**: How stable the embeddings have been over time (low variance = high consistency)
//!
//! The formula is:
//! ```text
//! coherence = (similarity_weight * avg_similarity) + (consistency_weight * consistency_score)
//! ```
//!
//! EMA smoothing is applied for stability across updates.

use crate::config::CoherenceConfig;
use crate::error::{UtlError, UtlResult};

use super::window::RollingWindow;

/// Coherence tracker that maintains history and computes coherence scores.
///
/// Uses a rolling window to track recent embeddings and computes coherence
/// based on semantic similarity and temporal consistency.
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
/// // Add some historical embeddings
/// tracker.update(&[0.1, 0.2, 0.3, 0.4]);
/// tracker.update(&[0.15, 0.25, 0.35, 0.25]);
///
/// // Compute coherence for a new embedding
/// let current = vec![0.12, 0.22, 0.32, 0.34];
/// let history = vec![vec![0.1, 0.2, 0.3, 0.4]];
/// let coherence = tracker.compute_coherence(&current, &history);
/// println!("Coherence: {}", coherence);
/// ```
#[derive(Debug, Clone)]
pub struct CoherenceTracker {
    /// Rolling window of recent embeddings.
    window: RollingWindow<Vec<f32>>,

    /// Weight for semantic similarity contribution.
    similarity_weight: f32,

    /// Weight for consistency contribution.
    consistency_weight: f32,

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
    /// ```
    pub fn new(config: &CoherenceConfig) -> Self {
        Self {
            window: RollingWindow::new(config.neighbor_count.max(10)),
            similarity_weight: config.similarity_weight,
            consistency_weight: config.consistency_weight,
            min_threshold: config.min_threshold,
            ema_coherence: None,
            ema_alpha: 0.3, // Default EMA alpha
            use_ema: true,
            neighbor_count: config.neighbor_count,
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

    /// Compute coherence score for the current embedding against history.
    ///
    /// # Arguments
    ///
    /// * `current` - The current embedding vector.
    /// * `history` - Historical embedding vectors for comparison.
    ///
    /// # Returns
    ///
    /// A coherence score in the range `[0, 1]`.
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
    /// let current = vec![0.1, 0.2, 0.3, 0.4];
    /// let history = vec![
    ///     vec![0.15, 0.25, 0.35, 0.25],
    ///     vec![0.12, 0.22, 0.32, 0.34],
    /// ];
    ///
    /// let coherence = tracker.compute_coherence(&current, &history);
    /// assert!(coherence >= 0.0 && coherence <= 1.0);
    /// ```
    pub fn compute_coherence(&self, current: &[f32], history: &[Vec<f32>]) -> f32 {
        if history.is_empty() {
            // No history - return minimum threshold (not completely incoherent)
            return self.min_threshold;
        }

        // Compute average similarity with history
        let avg_similarity = self.compute_average_similarity(current, history);

        // Compute consistency from the window (if available)
        let consistency = self.compute_consistency();

        // Combine with weights
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
    /// The coherence score for the new embedding.
    pub fn update_and_compute(&mut self, embedding: &[f32]) -> f32 {
        let history: Vec<Vec<f32>> = self.window.to_vec();
        let coherence = self.compute_coherence(embedding, &history);

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

        let coherence = tracker.compute_coherence(&current, &[]);
        assert_eq!(coherence, tracker.min_threshold);
    }

    #[test]
    fn test_compute_coherence_single_history() {
        let tracker = default_tracker();
        let current = vec![0.1, 0.2, 0.3, 0.4];
        let history = vec![vec![0.1, 0.2, 0.3, 0.4]]; // Identical

        let coherence = tracker.compute_coherence(&current, &history);
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

        let coherence = tracker.compute_coherence(&current, &history);
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

        let coherence = tracker.compute_coherence(&current, &history);
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
}
