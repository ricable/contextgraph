//! Embedding distance computation for semantic novelty.
//!
//! This module provides functions and calculators for measuring semantic distance
//! between embedding vectors, which is used to compute surprise/novelty in the
//! context graph system.
//!
//! # Distance Metrics
//!
//! - **Cosine distance**: Measures angular difference between vectors (1 - cosine similarity)
//! - **Euclidean distance**: Standard L2 distance (normalized to [0, 1])
//!
//! # Numerical Stability
//!
//! Per AP-009, all outputs are clamped to valid ranges [0, 1] with no NaN or Infinity values.

use crate::config::SurpriseConfig;
use crate::error::{UtlError, UtlResult};

/// Compute cosine distance between two embedding vectors.
///
/// Cosine distance is defined as `1 - cosine_similarity`, where cosine similarity
/// is the dot product of normalized vectors. Returns a value in [0, 1] where:
/// - 0 means identical direction
/// - 1 means orthogonal vectors
/// - Values close to 2 would indicate opposite directions (clamped to 1)
///
/// # Arguments
///
/// * `a` - First embedding vector
/// * `b` - Second embedding vector
///
/// # Returns
///
/// Cosine distance in range [0, 1]. Returns 0.0 for empty or zero-magnitude vectors.
///
/// # Example
///
/// ```
/// use context_graph_utl::surprise::compute_cosine_distance;
///
/// let a = vec![1.0, 0.0, 0.0];
/// let b = vec![0.0, 1.0, 0.0];
///
/// let dist = compute_cosine_distance(&a, &b);
/// assert!((dist - 1.0).abs() < 1e-6); // Orthogonal vectors have distance ~1
/// ```
pub fn compute_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    // Handle edge cases
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    // Use the minimum length if dimensions mismatch
    let len = a.len().min(b.len());

    // Compute dot product and magnitudes
    let mut dot = 0.0f64;
    let mut mag_a = 0.0f64;
    let mut mag_b = 0.0f64;

    for i in 0..len {
        let a_val = a[i] as f64;
        let b_val = b[i] as f64;
        dot += a_val * b_val;
        mag_a += a_val * a_val;
        mag_b += b_val * b_val;
    }

    let mag_a = mag_a.sqrt();
    let mag_b = mag_b.sqrt();

    // Handle zero magnitude vectors
    if mag_a < 1e-15 || mag_b < 1e-15 {
        return 0.0;
    }

    // Cosine similarity
    let cosine_sim = dot / (mag_a * mag_b);

    // Cosine distance = 1 - cosine_similarity
    // Clamp to [0, 1] to handle floating point errors and opposite vectors
    let distance = (1.0 - cosine_sim) as f32;

    // Handle NaN/Infinity per AP-009
    if distance.is_nan() || distance.is_infinite() {
        0.0
    } else {
        distance.clamp(0.0, 1.0)
    }
}

/// Compute embedding-based surprise from current embedding and recent history.
///
/// The surprise is computed as the minimum distance from the current embedding
/// to any recent embedding, representing how different the current item is from
/// recently seen items.
///
/// # Arguments
///
/// * `current` - The current embedding vector
/// * `recent` - List of recent embedding vectors
///
/// # Returns
///
/// Surprise value in range [0, 1]. Higher values indicate more novelty.
/// Returns 1.0 if history is empty (maximum surprise for first item).
///
/// # Example
///
/// ```
/// use context_graph_utl::surprise::compute_embedding_surprise;
///
/// let current = vec![0.1, 0.2, 0.3];
/// let recent = vec![
///     vec![0.15, 0.25, 0.35],  // Similar
///     vec![0.9, 0.05, 0.05],   // Different
/// ];
///
/// let surprise = compute_embedding_surprise(&current, &recent);
/// assert!(surprise >= 0.0 && surprise <= 1.0);
/// ```
pub fn compute_embedding_surprise(current: &[f32], recent: &[Vec<f32>]) -> f32 {
    // Empty current embedding has no surprise
    if current.is_empty() {
        return 0.0;
    }

    // Empty history means maximum surprise (completely novel)
    if recent.is_empty() {
        return 1.0;
    }

    // Find minimum distance to any recent embedding
    let mut min_distance = f32::MAX;

    for past in recent {
        if !past.is_empty() {
            let dist = compute_cosine_distance(current, past);
            if dist < min_distance {
                min_distance = dist;
            }
        }
    }

    // If no valid comparisons, return maximum surprise
    if min_distance == f32::MAX {
        return 1.0;
    }

    // The surprise is the minimum distance (clamped per AP-009)
    min_distance.clamp(0.0, 1.0)
}

/// Calculator for embedding distance-based surprise.
///
/// Provides configurable embedding distance computation with support for
/// different distance metrics, weighting, and history handling.
///
/// # Example
///
/// ```
/// use context_graph_utl::surprise::EmbeddingDistanceCalculator;
///
/// let calc = EmbeddingDistanceCalculator::default();
/// let current = vec![0.1, 0.2, 0.3];
/// let history = vec![vec![0.15, 0.25, 0.35]];
///
/// let surprise = calc.compute_surprise(&current, &history).unwrap();
/// assert!(surprise >= 0.0 && surprise <= 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct EmbeddingDistanceCalculator {
    /// Expected embedding dimension (0 means any dimension is accepted).
    expected_dimension: usize,
    /// Weight for distance aggregation (exponential weighting of history).
    recency_weight: f32,
    /// Maximum history items to consider.
    max_history: usize,
    /// Whether to use average distance instead of minimum.
    use_average: bool,
    /// Novelty boost factor for very different embeddings.
    novelty_boost: f32,
}

impl Default for EmbeddingDistanceCalculator {
    fn default() -> Self {
        Self {
            expected_dimension: 0,
            recency_weight: 0.9,
            max_history: 100,
            use_average: false,
            novelty_boost: 1.0,
        }
    }
}

impl EmbeddingDistanceCalculator {
    /// Create a new calculator from SurpriseConfig.
    pub fn from_config(config: &SurpriseConfig) -> Self {
        Self {
            expected_dimension: 0,
            recency_weight: if config.use_ema {
                config.ema_alpha
            } else {
                0.9
            },
            max_history: config.sample_count,
            use_average: false,
            novelty_boost: config.novelty_boost,
        }
    }

    /// Create a new calculator with custom settings.
    pub fn new(expected_dimension: usize, max_history: usize) -> Self {
        Self {
            expected_dimension,
            max_history,
            ..Default::default()
        }
    }

    /// Set the recency weight for exponential decay.
    pub fn with_recency_weight(mut self, weight: f32) -> Self {
        self.recency_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Enable average distance mode instead of minimum.
    pub fn with_average_mode(mut self, use_average: bool) -> Self {
        self.use_average = use_average;
        self
    }

    /// Set the novelty boost factor.
    pub fn with_novelty_boost(mut self, boost: f32) -> Self {
        self.novelty_boost = boost.clamp(0.5, 2.0);
        self
    }

    /// Compute surprise from current embedding and history.
    ///
    /// # Arguments
    ///
    /// * `current` - The current embedding vector
    /// * `history` - Historical embedding vectors (most recent first)
    ///
    /// # Returns
    ///
    /// Surprise value in [0, 1], or error if validation fails.
    ///
    /// # Errors
    ///
    /// Returns `UtlError::EmptyInput` if current embedding is empty.
    /// Returns `UtlError::DimensionMismatch` if embeddings have unexpected dimensions.
    pub fn compute_surprise(&self, current: &[f32], history: &[Vec<f32>]) -> UtlResult<f32> {
        // Validate current embedding
        if current.is_empty() {
            return Err(UtlError::EmptyInput);
        }

        // Check dimension if specified
        if self.expected_dimension > 0 && current.len() != self.expected_dimension {
            return Err(UtlError::DimensionMismatch {
                expected: self.expected_dimension,
                actual: current.len(),
            });
        }

        // Empty history means maximum surprise
        if history.is_empty() {
            return Ok(1.0);
        }

        // Limit history to max_history items
        let history_slice = if history.len() > self.max_history {
            &history[..self.max_history]
        } else {
            history
        };

        let surprise = if self.use_average {
            self.compute_average_distance(current, history_slice)?
        } else {
            self.compute_weighted_min_distance(current, history_slice)?
        };

        // Apply novelty boost and clamp per AP-009
        let boosted = surprise * self.novelty_boost;
        Ok(boosted.clamp(0.0, 1.0))
    }

    /// Compute cosine distance between two embeddings.
    ///
    /// # Errors
    ///
    /// Returns `UtlError::DimensionMismatch` if vectors have different lengths
    /// and expected_dimension is set.
    pub fn compute_distance(&self, a: &[f32], b: &[f32]) -> UtlResult<f32> {
        if a.is_empty() || b.is_empty() {
            return Err(UtlError::EmptyInput);
        }

        if self.expected_dimension > 0 {
            if a.len() != self.expected_dimension {
                return Err(UtlError::DimensionMismatch {
                    expected: self.expected_dimension,
                    actual: a.len(),
                });
            }
            if b.len() != self.expected_dimension {
                return Err(UtlError::DimensionMismatch {
                    expected: self.expected_dimension,
                    actual: b.len(),
                });
            }
        }

        Ok(compute_cosine_distance(a, b))
    }

    /// Compute weighted minimum distance with recency decay.
    fn compute_weighted_min_distance(
        &self,
        current: &[f32],
        history: &[Vec<f32>],
    ) -> UtlResult<f32> {
        let mut min_distance = f32::MAX;
        let mut weight = 1.0f32;

        for past in history {
            if past.is_empty() {
                continue;
            }

            // Validate dimension if required
            if self.expected_dimension > 0 && past.len() != self.expected_dimension {
                // Skip mismatched dimensions in history
                continue;
            }

            let dist = compute_cosine_distance(current, past);
            // Weight older items less
            let weighted_dist = dist / weight;

            if weighted_dist < min_distance {
                min_distance = weighted_dist;
            }

            weight *= self.recency_weight;
        }

        if min_distance == f32::MAX {
            Ok(1.0) // No valid comparisons
        } else {
            Ok(min_distance.clamp(0.0, 1.0))
        }
    }

    /// Compute average distance to history.
    fn compute_average_distance(&self, current: &[f32], history: &[Vec<f32>]) -> UtlResult<f32> {
        let mut sum = 0.0f64;
        let mut count = 0usize;
        let mut weight = 1.0f64;

        for past in history {
            if past.is_empty() {
                continue;
            }

            if self.expected_dimension > 0 && past.len() != self.expected_dimension {
                continue;
            }

            let dist = compute_cosine_distance(current, past) as f64;
            sum += dist * weight;
            count += 1;
            weight *= self.recency_weight as f64;
        }

        if count == 0 {
            Ok(1.0)
        } else {
            let avg = (sum / count as f64) as f32;
            Ok(avg.clamp(0.0, 1.0))
        }
    }

    /// Get the expected embedding dimension.
    pub fn expected_dimension(&self) -> usize {
        self.expected_dimension
    }

    /// Get the maximum history size.
    pub fn max_history(&self) -> usize {
        self.max_history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let dist = compute_cosine_distance(&a, &a);
        assert!(
            dist.abs() < 1e-6,
            "Identical vectors should have distance ~0"
        );
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let dist = compute_cosine_distance(&a, &b);
        assert!(
            (dist - 1.0).abs() < 1e-6,
            "Orthogonal vectors should have distance ~1"
        );
    }

    #[test]
    fn test_cosine_distance_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let dist = compute_cosine_distance(&a, &b);
        // Opposite vectors have cosine similarity of -1, so distance would be 2
        // But we clamp to [0, 1]
        assert!(dist <= 1.0);
    }

    #[test]
    fn test_cosine_distance_empty() {
        let empty: Vec<f32> = vec![];
        let a = vec![1.0, 2.0];
        assert_eq!(compute_cosine_distance(&empty, &a), 0.0);
        assert_eq!(compute_cosine_distance(&a, &empty), 0.0);
        assert_eq!(compute_cosine_distance(&empty, &empty), 0.0);
    }

    #[test]
    fn test_cosine_distance_zero_vector() {
        let zero = vec![0.0, 0.0, 0.0];
        let a = vec![1.0, 2.0, 3.0];
        assert_eq!(compute_cosine_distance(&zero, &a), 0.0);
        assert_eq!(compute_cosine_distance(&a, &zero), 0.0);
    }

    #[test]
    fn test_cosine_distance_range() {
        // Test various vector pairs
        let pairs = vec![
            (vec![1.0, 0.0], vec![0.707, 0.707]),
            (vec![0.5, 0.5], vec![0.3, 0.7]),
            (vec![0.1, 0.9], vec![0.9, 0.1]),
        ];

        for (a, b) in pairs {
            let dist = compute_cosine_distance(&a, &b);
            assert!((0.0..=1.0).contains(&dist), "Distance should be in [0, 1]");
        }
    }

    #[test]
    fn test_embedding_surprise_empty_history() {
        let current = vec![0.1, 0.2, 0.3];
        let history: Vec<Vec<f32>> = vec![];
        let surprise = compute_embedding_surprise(&current, &history);
        assert_eq!(surprise, 1.0, "Empty history should give maximum surprise");
    }

    #[test]
    fn test_embedding_surprise_identical() {
        let current = vec![0.1, 0.2, 0.3];
        let history = vec![vec![0.1, 0.2, 0.3]];
        let surprise = compute_embedding_surprise(&current, &history);
        assert!(
            surprise < 0.01,
            "Identical embedding should have low surprise"
        );
    }

    #[test]
    fn test_embedding_surprise_similar() {
        let current = vec![0.1, 0.2, 0.3];
        let history = vec![vec![0.15, 0.25, 0.35], vec![0.9, 0.05, 0.05]];
        let surprise = compute_embedding_surprise(&current, &history);
        // Should be closer to the similar one
        assert!(surprise < 0.5);
    }

    #[test]
    fn test_calculator_default() {
        let calc = EmbeddingDistanceCalculator::default();
        assert_eq!(calc.expected_dimension(), 0);
        assert_eq!(calc.max_history(), 100);
    }

    #[test]
    fn test_calculator_compute_surprise() {
        let calc = EmbeddingDistanceCalculator::default();
        let current = vec![0.1, 0.2, 0.3, 0.4];
        let history = vec![vec![0.15, 0.25, 0.35, 0.25]];

        let result = calc.compute_surprise(&current, &history);
        assert!(result.is_ok());
        let surprise = result.unwrap();
        assert!((0.0..=1.0).contains(&surprise));
    }

    #[test]
    fn test_calculator_empty_input() {
        let calc = EmbeddingDistanceCalculator::default();
        let empty: Vec<f32> = vec![];
        let history = vec![vec![0.1, 0.2]];

        assert!(matches!(
            calc.compute_surprise(&empty, &history),
            Err(UtlError::EmptyInput)
        ));
    }

    #[test]
    fn test_calculator_dimension_mismatch() {
        let calc = EmbeddingDistanceCalculator::new(3, 100);
        let current = vec![0.1, 0.2]; // Wrong dimension
        let history = vec![vec![0.1, 0.2, 0.3]];

        let result = calc.compute_surprise(&current, &history);
        assert!(matches!(result, Err(UtlError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_calculator_average_mode() {
        let calc = EmbeddingDistanceCalculator::default().with_average_mode(true);
        let current = vec![0.1, 0.2, 0.3];
        let history = vec![
            vec![0.15, 0.25, 0.35],
            vec![0.2, 0.3, 0.4],
            vec![0.9, 0.05, 0.05],
        ];

        let result = calc.compute_surprise(&current, &history);
        assert!(result.is_ok());
        let surprise = result.unwrap();
        assert!((0.0..=1.0).contains(&surprise));
    }

    #[test]
    fn test_calculator_novelty_boost() {
        let calc_normal = EmbeddingDistanceCalculator::default();
        let calc_boosted = EmbeddingDistanceCalculator::default().with_novelty_boost(1.5);

        let current = vec![0.1, 0.2, 0.3];
        let history = vec![vec![0.5, 0.3, 0.2]];

        let normal = calc_normal.compute_surprise(&current, &history).unwrap();
        let boosted = calc_boosted.compute_surprise(&current, &history).unwrap();

        // Boosted should be higher (but still clamped to 1.0)
        assert!(boosted >= normal || boosted == 1.0);
    }

    #[test]
    fn test_calculator_compute_distance() {
        let calc = EmbeddingDistanceCalculator::default();
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let result = calc.compute_distance(&a, &b);
        assert!(result.is_ok());
        let dist = result.unwrap();
        assert!((dist - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_no_nan_infinity() {
        let _calc = EmbeddingDistanceCalculator::default();

        // Edge cases that might produce NaN
        let zero = vec![0.0, 0.0, 0.0];
        let normal = vec![1.0, 2.0, 3.0];

        let dist = compute_cosine_distance(&zero, &normal);
        assert!(!dist.is_nan());
        assert!(!dist.is_infinite());

        // Very small values
        let tiny = vec![1e-15, 1e-15, 1e-15];
        let dist2 = compute_cosine_distance(&tiny, &normal);
        assert!(!dist2.is_nan());
        assert!(!dist2.is_infinite());
    }

    #[test]
    fn test_recency_weighting() {
        let calc = EmbeddingDistanceCalculator::default().with_recency_weight(0.5);
        let current = vec![0.5, 0.5];

        // More recent item is more different, but older item is closer
        let history = vec![
            vec![0.9, 0.1], // Recent but different
            vec![0.5, 0.5], // Old but identical
        ];

        // With recency weighting, recent items matter more
        let surprise = calc.compute_surprise(&current, &history).unwrap();
        assert!((0.0..=1.0).contains(&surprise));
    }
}
