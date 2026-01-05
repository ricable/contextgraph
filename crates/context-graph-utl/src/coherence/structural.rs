//! Structural coherence computation based on graph relationships.
//!
//! This module computes coherence based on the structure of the knowledge graph,
//! measuring how well a node's embedding aligns with its neighbors.
//!
//! # Overview
//!
//! Structural coherence captures the idea that nodes in a graph should be
//! semantically related to their neighbors. A node with high structural coherence
//! has embeddings that are similar to its connected neighbors.
//!
//! # Example
//!
//! ```ignore
//! use context_graph_utl::coherence::{compute_structural_coherence, StructuralCoherenceCalculator};
//! use uuid::Uuid;
//!
//! let node_id = Uuid::new_v4();
//! let neighbor_embeddings = vec![
//!     vec![0.1, 0.2, 0.3, 0.4],
//!     vec![0.15, 0.25, 0.35, 0.25],
//!     vec![0.12, 0.22, 0.32, 0.34],
//! ];
//!
//! let coherence = compute_structural_coherence(node_id, &neighbor_embeddings);
//! println!("Structural coherence: {}", coherence);
//! ```

use uuid::Uuid;

/// Calculator for structural coherence based on graph topology.
///
/// This calculator measures coherence by examining how well a node's
/// embedding aligns with its neighbors in the graph structure.
///
/// # Future Extensions
///
/// In a full implementation, this would:
/// - Access the graph to retrieve actual neighbor embeddings
/// - Support different distance metrics
/// - Weight edges by relationship type or strength
/// - Handle multi-hop neighborhoods
#[derive(Debug, Clone)]
pub struct StructuralCoherenceCalculator {
    /// Minimum similarity threshold for neighbor consideration.
    min_similarity: f32,

    /// Weight decay factor for multi-hop neighbors.
    hop_decay: f32,

    /// Maximum neighborhood depth to consider.
    max_hops: usize,
}

impl Default for StructuralCoherenceCalculator {
    fn default() -> Self {
        Self {
            min_similarity: 0.3,
            hop_decay: 0.5,
            max_hops: 2,
        }
    }
}

impl StructuralCoherenceCalculator {
    /// Create a new structural coherence calculator with custom settings.
    ///
    /// # Arguments
    ///
    /// * `min_similarity` - Minimum cosine similarity for neighbor inclusion.
    /// * `hop_decay` - Decay factor applied per hop distance.
    /// * `max_hops` - Maximum graph distance to consider.
    pub fn new(min_similarity: f32, hop_decay: f32, max_hops: usize) -> Self {
        Self {
            min_similarity: min_similarity.clamp(0.0, 1.0),
            hop_decay: hop_decay.clamp(0.0, 1.0),
            max_hops,
        }
    }

    /// Compute structural coherence for a node given its neighbors' embeddings.
    ///
    /// # Arguments
    ///
    /// * `node_embedding` - The embedding of the node to evaluate.
    /// * `neighbor_embeddings` - Embeddings of the node's graph neighbors.
    ///
    /// # Returns
    ///
    /// A coherence score in the range `[0, 1]`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use context_graph_utl::coherence::StructuralCoherenceCalculator;
    ///
    /// let calc = StructuralCoherenceCalculator::default();
    /// let node_emb = vec![0.1, 0.2, 0.3, 0.4];
    /// let neighbor_embs = vec![
    ///     vec![0.12, 0.22, 0.28, 0.38],
    ///     vec![0.11, 0.21, 0.29, 0.39],
    /// ];
    ///
    /// let coherence = calc.compute(&node_emb, &neighbor_embs);
    /// ```
    pub fn compute(&self, node_embedding: &[f32], neighbor_embeddings: &[Vec<f32>]) -> f32 {
        if neighbor_embeddings.is_empty() {
            // No neighbors - return moderate coherence (neutral assumption)
            return 0.5;
        }

        // Compute similarity with each neighbor
        let similarities: Vec<f32> = neighbor_embeddings
            .iter()
            .map(|ne| cosine_similarity(node_embedding, ne))
            .collect();

        // Filter by minimum similarity and compute weighted average
        let valid_similarities: Vec<f32> = similarities
            .into_iter()
            .filter(|&s| s >= self.min_similarity)
            .collect();

        if valid_similarities.is_empty() {
            // No neighbors meet similarity threshold
            return 0.2; // Low but not zero coherence
        }

        // Average similarity, normalized to [0, 1]
        let avg: f32 = valid_similarities.iter().sum::<f32>() / valid_similarities.len() as f32;

        // Convert from [-1, 1] similarity to [0, 1] coherence
        let coherence = (avg + 1.0) / 2.0;
        coherence.clamp(0.0, 1.0)
    }

    /// Compute structural coherence with weighted neighbors.
    ///
    /// # Arguments
    ///
    /// * `node_embedding` - The embedding of the node to evaluate.
    /// * `weighted_neighbors` - Tuples of (embedding, weight) for each neighbor.
    ///
    /// # Returns
    ///
    /// A weighted coherence score in the range `[0, 1]`.
    pub fn compute_weighted(
        &self,
        node_embedding: &[f32],
        weighted_neighbors: &[(Vec<f32>, f32)],
    ) -> f32 {
        if weighted_neighbors.is_empty() {
            return 0.5;
        }

        let mut weighted_sum = 0.0f32;
        let mut weight_total = 0.0f32;

        for (neighbor_emb, weight) in weighted_neighbors {
            let sim = cosine_similarity(node_embedding, neighbor_emb);
            let coherence = (sim + 1.0) / 2.0; // Normalize to [0, 1]

            weighted_sum += coherence * weight;
            weight_total += weight;
        }

        if weight_total == 0.0 {
            return 0.5;
        }

        (weighted_sum / weight_total).clamp(0.0, 1.0)
    }

    /// Get the minimum similarity threshold.
    pub fn min_similarity(&self) -> f32 {
        self.min_similarity
    }

    /// Get the hop decay factor.
    pub fn hop_decay(&self) -> f32 {
        self.hop_decay
    }

    /// Get the maximum hops setting.
    pub fn max_hops(&self) -> usize {
        self.max_hops
    }
}

/// Compute structural coherence for a node given its neighbor embeddings.
///
/// This is a convenience function that creates a default calculator and
/// computes coherence. For repeated computations, prefer creating a
/// [`StructuralCoherenceCalculator`] instance.
///
/// # Arguments
///
/// * `node_id` - UUID of the node (for future graph integration).
/// * `neighbor_embeddings` - Embeddings of the node's graph neighbors.
///
/// # Returns
///
/// A coherence score in the range `[0, 1]`.
///
/// # Note
///
/// Currently, `node_id` is not used as this is a stub implementation.
/// In a full implementation, it would be used to fetch the node's embedding
/// from the graph.
///
/// # Example
///
/// ```ignore
/// use context_graph_utl::coherence::compute_structural_coherence;
/// use uuid::Uuid;
///
/// let node_id = Uuid::new_v4();
/// let neighbor_embeddings = vec![
///     vec![0.1, 0.2, 0.3, 0.4],
///     vec![0.15, 0.25, 0.35, 0.25],
/// ];
///
/// let coherence = compute_structural_coherence(node_id, &neighbor_embeddings);
/// assert!(coherence >= 0.0 && coherence <= 1.0);
/// ```
pub fn compute_structural_coherence(_node_id: Uuid, neighbor_embeddings: &[Vec<f32>]) -> f32 {
    // Stub implementation: compute coherence based on neighbor similarity
    // In a full implementation, we would fetch the node's embedding from the graph

    if neighbor_embeddings.is_empty() {
        return 0.5; // Neutral coherence for isolated nodes
    }

    // Compute average pairwise similarity among neighbors
    // High similarity among neighbors suggests a coherent neighborhood
    let mut total_similarity = 0.0f32;
    let mut pair_count = 0;

    for i in 0..neighbor_embeddings.len() {
        for j in (i + 1)..neighbor_embeddings.len() {
            let sim = cosine_similarity(&neighbor_embeddings[i], &neighbor_embeddings[j]);
            total_similarity += sim;
            pair_count += 1;
        }
    }

    if pair_count == 0 {
        // Only one neighbor - return moderate coherence
        return 0.6;
    }

    let avg_similarity = total_similarity / pair_count as f32;

    // Convert from [-1, 1] to [0, 1] and clamp
    let coherence = (avg_similarity + 1.0) / 2.0;
    coherence.clamp(0.0, 1.0)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculator_default() {
        let calc = StructuralCoherenceCalculator::default();
        assert_eq!(calc.min_similarity(), 0.3);
        assert_eq!(calc.hop_decay(), 0.5);
        assert_eq!(calc.max_hops(), 2);
    }

    #[test]
    fn test_calculator_custom() {
        let calc = StructuralCoherenceCalculator::new(0.5, 0.7, 3);
        assert_eq!(calc.min_similarity(), 0.5);
        assert_eq!(calc.hop_decay(), 0.7);
        assert_eq!(calc.max_hops(), 3);
    }

    #[test]
    fn test_calculator_clamping() {
        let calc = StructuralCoherenceCalculator::new(2.0, -0.5, 3);
        assert_eq!(calc.min_similarity(), 1.0); // Clamped
        assert_eq!(calc.hop_decay(), 0.0); // Clamped
    }

    #[test]
    fn test_compute_empty_neighbors() {
        let calc = StructuralCoherenceCalculator::default();
        let node_emb = vec![0.1, 0.2, 0.3];
        let neighbors: Vec<Vec<f32>> = vec![];

        let coherence = calc.compute(&node_emb, &neighbors);
        assert_eq!(coherence, 0.5); // Neutral for no neighbors
    }

    #[test]
    fn test_compute_identical_neighbors() {
        let calc = StructuralCoherenceCalculator::default();
        let node_emb = vec![0.1, 0.2, 0.3, 0.4];
        let neighbors = vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.1, 0.2, 0.3, 0.4]];

        let coherence = calc.compute(&node_emb, &neighbors);
        assert!(
            coherence > 0.9,
            "Expected high coherence for identical embeddings"
        );
    }

    #[test]
    fn test_compute_similar_neighbors() {
        let calc = StructuralCoherenceCalculator::default();
        let node_emb = vec![0.1, 0.2, 0.3, 0.4];
        let neighbors = vec![vec![0.12, 0.22, 0.28, 0.38], vec![0.11, 0.21, 0.29, 0.39]];

        let coherence = calc.compute(&node_emb, &neighbors);
        assert!((0.0..=1.0).contains(&coherence));
        assert!(
            coherence > 0.7,
            "Expected high coherence for similar neighbors"
        );
    }

    #[test]
    fn test_compute_dissimilar_neighbors() {
        let calc = StructuralCoherenceCalculator::default();
        let node_emb = vec![1.0, 0.0, 0.0, 0.0];
        let neighbors = vec![
            vec![0.0, 1.0, 0.0, 0.0], // Orthogonal
            vec![0.0, 0.0, 1.0, 0.0], // Orthogonal
        ];

        let coherence = calc.compute(&node_emb, &neighbors);
        assert!((0.0..=1.0).contains(&coherence));
        // Low similarity but not zero due to min_similarity filtering
    }

    #[test]
    fn test_compute_weighted_empty() {
        let calc = StructuralCoherenceCalculator::default();
        let node_emb = vec![0.1, 0.2, 0.3];
        let weighted: Vec<(Vec<f32>, f32)> = vec![];

        let coherence = calc.compute_weighted(&node_emb, &weighted);
        assert_eq!(coherence, 0.5);
    }

    #[test]
    fn test_compute_weighted() {
        let calc = StructuralCoherenceCalculator::default();
        let node_emb = vec![0.1, 0.2, 0.3, 0.4];
        let weighted = vec![
            (vec![0.1, 0.2, 0.3, 0.4], 1.0),     // Identical, high weight
            (vec![-0.4, -0.3, -0.2, -0.1], 0.1), // Opposite, low weight
        ];

        let coherence = calc.compute_weighted(&node_emb, &weighted);
        assert!((0.0..=1.0).contains(&coherence));
        // Should be biased toward high coherence due to weights
        assert!(coherence > 0.7);
    }

    #[test]
    fn test_structural_coherence_function_empty() {
        let node_id = Uuid::new_v4();
        let neighbors: Vec<Vec<f32>> = vec![];

        let coherence = compute_structural_coherence(node_id, &neighbors);
        assert_eq!(coherence, 0.5);
    }

    #[test]
    fn test_structural_coherence_function_single() {
        let node_id = Uuid::new_v4();
        let neighbors = vec![vec![0.1, 0.2, 0.3, 0.4]];

        let coherence = compute_structural_coherence(node_id, &neighbors);
        assert_eq!(coherence, 0.6); // Single neighbor returns moderate coherence
    }

    #[test]
    fn test_structural_coherence_function_similar() {
        let node_id = Uuid::new_v4();
        let neighbors = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.12, 0.22, 0.28, 0.38],
            vec![0.11, 0.21, 0.29, 0.39],
        ];

        let coherence = compute_structural_coherence(node_id, &neighbors);
        assert!((0.0..=1.0).contains(&coherence));
        assert!(
            coherence > 0.8,
            "Expected high coherence for similar neighbors"
        );
    }

    #[test]
    fn test_structural_coherence_function_diverse() {
        let node_id = Uuid::new_v4();
        let neighbors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        let coherence = compute_structural_coherence(node_id, &neighbors);
        assert!((0.0..=1.0).contains(&coherence));
        // Orthogonal neighbors should have lower coherence
        assert!(
            coherence < 0.6,
            "Expected lower coherence for orthogonal neighbors"
        );
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
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
    fn test_cosine_similarity_mismatched_dims() {
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
    fn test_coherence_output_range() {
        let calc = StructuralCoherenceCalculator::default();
        let node_id = Uuid::new_v4();

        // Test various scenarios all produce valid [0, 1] output
        let scenarios: Vec<Vec<Vec<f32>>> = vec![
            vec![],
            vec![vec![0.1, 0.2, 0.3]],
            vec![vec![0.1, 0.2, 0.3], vec![0.1, 0.2, 0.3]],
            vec![vec![1.0, 0.0, 0.0], vec![-1.0, 0.0, 0.0]],
            vec![
                vec![0.5, 0.5, 0.5],
                vec![0.5, 0.5, 0.5],
                vec![0.5, 0.5, 0.5],
                vec![0.5, 0.5, 0.5],
            ],
        ];

        for neighbors in scenarios {
            let coherence1 = compute_structural_coherence(node_id, &neighbors);
            assert!(
                (0.0..=1.0).contains(&coherence1),
                "compute_structural_coherence output {} out of range",
                coherence1
            );

            if !neighbors.is_empty() {
                let node_emb = vec![0.5, 0.5, 0.5];
                let coherence2 = calc.compute(&node_emb, &neighbors);
                assert!(
                    (0.0..=1.0).contains(&coherence2),
                    "Calculator.compute output {} out of range",
                    coherence2
                );
            }
        }
    }
}
