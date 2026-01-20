//! Shared utility functions for benchmarking.

use std::cmp::Ordering;
use uuid::Uuid;

/// Compute cosine similarity between two vectors.
///
/// Returns 0.0 if vectors have different lengths or either has zero norm.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Sort comparator for (Uuid, f32) pairs by similarity descending.
///
/// Uses UUID as tiebreaker for deterministic ordering when similarities are equal.
pub fn similarity_sort_desc(a: &(Uuid, f32), b: &(Uuid, f32)) -> Ordering {
    match b.1.partial_cmp(&a.1) {
        Some(Ordering::Equal) | None => a.0.cmp(&b.0),
        Some(ord) => ord,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.01);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_similarity_sort_desc() {
        let id1 = Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap();
        let id2 = Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap();

        let mut items = vec![(id2, 0.5), (id1, 0.8), (id1, 0.5)];
        items.sort_by(similarity_sort_desc);

        // Should be: (id1, 0.8), (id1, 0.5), (id2, 0.5) - highest sim first, then UUID tiebreaker
        assert_eq!(items[0].0, id1);
        assert_eq!(items[0].1, 0.8);
    }
}
