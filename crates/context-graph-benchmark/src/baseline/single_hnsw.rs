//! Single HNSW index for single-embedder baseline.
//!
//! This provides a simpler HNSW wrapper specifically for benchmarking
//! the single-embedder baseline without the complexity of the full
//! multi-space index system.

use std::collections::HashMap;
use std::sync::RwLock;
use uuid::Uuid;

use crate::util::{cosine_similarity, similarity_sort_desc};

/// Single HNSW index using in-memory brute force for simplicity.
///
/// In production, this would use usearch like the main system, but for
/// benchmarking purposes we use a simple brute force implementation to
/// avoid external dependencies in the benchmark crate.
///
/// The performance characteristics should still be representative since
/// we measure both systems the same way.
pub struct SingleHnswIndex {
    /// Dimension of vectors.
    dimension: usize,
    /// HNSW M parameter (stored for API compatibility).
    #[allow(dead_code)]
    m: usize,
    /// ef_construction parameter.
    #[allow(dead_code)]
    ef_construction: usize,
    /// ef_search parameter.
    #[allow(dead_code)]
    ef_search: usize,
    /// Stored vectors.
    vectors: RwLock<HashMap<Uuid, Vec<f32>>>,
}

impl SingleHnswIndex {
    /// Create a new index.
    pub fn new(dimension: usize, m: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            dimension,
            m,
            ef_construction,
            ef_search,
            vectors: RwLock::new(HashMap::new()),
        }
    }

    /// Insert a vector.
    pub fn insert(&self, id: Uuid, vector: &[f32]) {
        if vector.len() != self.dimension {
            panic!(
                "FAIL FAST: Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            );
        }

        let mut vectors = self.vectors.write().unwrap();
        vectors.insert(id, vector.to_vec());
    }

    /// Remove a vector.
    pub fn remove(&self, id: &Uuid) -> bool {
        let mut vectors = self.vectors.write().unwrap();
        vectors.remove(id).is_some()
    }

    /// Search for k nearest neighbors.
    ///
    /// Uses brute force cosine similarity for simplicity.
    /// Returns (id, similarity) pairs sorted by similarity descending.
    pub fn search(&self, query: &[f32], k: usize, min_similarity: Option<f32>) -> Vec<(Uuid, f32)> {
        if query.len() != self.dimension {
            return Vec::new();
        }

        let vectors = self.vectors.read().unwrap();

        let mut similarities: Vec<(Uuid, f32)> = vectors
            .iter()
            .map(|(id, vec)| (*id, cosine_similarity(query, vec)))
            .filter(|(_, sim)| min_similarity.map(|min| *sim >= min).unwrap_or(true))
            .collect();

        // Sort by similarity descending, with UUID as tiebreaker for determinism
        similarities.sort_by(similarity_sort_desc);

        similarities.truncate(k);
        similarities
    }

    /// Get number of vectors.
    pub fn len(&self) -> usize {
        self.vectors.read().unwrap().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.read().unwrap().is_empty()
    }

    /// Get all IDs.
    pub fn ids(&self) -> Vec<Uuid> {
        self.vectors.read().unwrap().keys().copied().collect()
    }

    /// Check if contains an ID.
    pub fn contains(&self, id: &Uuid) -> bool {
        self.vectors.read().unwrap().contains_key(id)
    }

    /// Get dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Compute Euclidean distance between two vectors.
#[allow(dead_code)]
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let index = SingleHnswIndex::new(3, 16, 200, 100);

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        index.insert(id1, &[1.0, 0.0, 0.0]);
        index.insert(id2, &[0.0, 1.0, 0.0]);

        let results = index.search(&[1.0, 0.0, 0.0], 1, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id1);
        assert!((results[0].1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cosine_similarity_fn() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.01);
        assert!(cosine_similarity(&a, &c).abs() < 0.01);
    }

    #[test]
    fn test_min_similarity_threshold() {
        let index = SingleHnswIndex::new(3, 16, 200, 100);

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        index.insert(id1, &[1.0, 0.0, 0.0]);
        index.insert(id2, &[0.5, 0.5, 0.707]);

        // Search with high threshold should only return very similar
        let results = index.search(&[1.0, 0.0, 0.0], 10, Some(0.9));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id1);
    }
}
