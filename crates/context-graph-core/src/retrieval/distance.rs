//! Distance and similarity metrics for the 13 embedding spaces.
//!
//! This module provides unified distance/similarity computation across all
//! embedding types: dense, sparse, binary, and token-level.
//!
//! # Design Philosophy
//!
//! Most similarity functions delegate to existing vector type methods:
//! - DenseVector::cosine_similarity()
//! - SparseVector::jaccard_similarity()
//! - BinaryVector::hamming_distance()
//!
//! This module adds:
//! - max_sim() for ColBERT late interaction (E12)
//! - transe_similarity() for knowledge graph embeddings (E11)
//! - compute_similarity_for_space() dispatcher
//!
//! # All outputs are normalized to [0.0, 1.0]

use crate::embeddings::{BinaryVector, DenseVector};
use crate::teleological::Embedder;
use crate::types::fingerprint::{EmbeddingRef, SemanticFingerprint, SparseVector};

/// Compute cosine similarity between two dense vectors.
///
/// Thin wrapper that creates DenseVectors and delegates to existing method.
/// Returns 0.0 for zero-magnitude vectors (AP-10: no NaN).
///
/// # Arguments
/// * `a` - First dense embedding as f32 slice
/// * `b` - Second dense embedding as f32 slice
///
/// # Returns
/// Similarity in [0.0, 1.0] where 1.0 = identical direction
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    // Check for zero vectors before computation (AP-10 compliance)
    let mag_a_sq: f32 = a.iter().map(|x| x * x).sum();
    let mag_b_sq: f32 = b.iter().map(|x| x * x).sum();
    if mag_a_sq == 0.0 || mag_b_sq == 0.0 {
        return 0.0;
    }

    let vec_a = DenseVector::new(a.to_vec());
    let vec_b = DenseVector::new(b.to_vec());

    let raw_sim = vec_a.cosine_similarity(&vec_b);

    // Normalize from [-1, 1] to [0, 1]
    // DenseVector.cosine_similarity already clamps to [-1, 1]
    (raw_sim + 1.0) / 2.0
}

/// Compute Jaccard similarity between two sparse vectors.
///
/// Thin wrapper that delegates to SparseVector::jaccard_similarity().
/// Returns |A ∩ B| / |A ∪ B| based on non-zero indices.
///
/// # Returns
/// Similarity in [0.0, 1.0] where 1.0 = identical index sets
pub fn jaccard_similarity(a: &SparseVector, b: &SparseVector) -> f32 {
    a.jaccard_similarity(b)
}

/// Compute Hamming similarity between two binary vectors.
///
/// Converts Hamming distance to similarity: 1.0 - (distance / max_bits).
///
/// # Returns
/// Similarity in [0.0, 1.0] where 1.0 = identical bit patterns
pub fn hamming_similarity(a: &BinaryVector, b: &BinaryVector) -> f32 {
    let distance = a.hamming_distance(b);
    let max_bits = a.bit_len().max(b.bit_len());

    if max_bits == 0 {
        return 1.0; // Empty vectors are identical
    }

    1.0 - (distance as f32 / max_bits as f32)
}

/// Compute MaxSim for late interaction (ColBERT-style).
///
/// For each query token, find max cosine similarity to any memory token.
/// Return mean of all max similarities.
///
/// # Algorithm
/// ```text
/// MaxSim = (1/|Q|) * Σ_q∈Q max_m∈M cos(q, m)
/// ```
///
/// # Arguments
/// * `query_tokens` - Query token embeddings (each 128D for E12)
/// * `memory_tokens` - Memory token embeddings
///
/// # Returns
/// Similarity in [0.0, 1.0], returns 0.0 if either list is empty
pub fn max_sim(query_tokens: &[Vec<f32>], memory_tokens: &[Vec<f32>]) -> f32 {
    if query_tokens.is_empty() || memory_tokens.is_empty() {
        return 0.0;
    }

    let mut total_max = 0.0_f32;

    for q_tok in query_tokens {
        let mut max_sim_for_token = 0.0_f32;

        for m_tok in memory_tokens {
            let sim = cosine_similarity(q_tok, m_tok);
            max_sim_for_token = max_sim_for_token.max(sim);
        }

        total_max += max_sim_for_token;
    }

    total_max / query_tokens.len() as f32
}

/// Compute TransE-style similarity for knowledge graph embeddings.
///
/// Uses inverse of Euclidean distance: 1 / (1 + distance).
/// This maps distance [0, ∞) to similarity (0, 1].
///
/// # Returns
/// Similarity in (0.0, 1.0] where 1.0 = identical vectors (distance = 0)
pub fn transe_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let vec_a = DenseVector::new(a.to_vec());
    let vec_b = DenseVector::new(b.to_vec());

    let distance = vec_a.euclidean_distance(&vec_b);
    1.0 / (1.0 + distance)
}

/// Compute similarity for a specific embedding space.
///
/// This is the main dispatcher that routes to the appropriate similarity
/// function based on the embedder type.
///
/// # Metrics by Embedder
/// - E1 (Semantic): Cosine
/// - E2-E4 (Temporal): Cosine
/// - E5 (Causal): Cosine (asymmetric handled at embedding time)
/// - E6 (Sparse): Jaccard
/// - E7 (Code): Cosine
/// - E8 (Emotional): Cosine
/// - E9 (HDC): Cosine (stored as projected dense, not binary)
/// - E10 (Multimodal): Cosine
/// - E11 (Entity): TransE
/// - E12 (LateInteraction): MaxSim
/// - E13 (KeywordSplade): Jaccard
///
/// # Arguments
/// * `embedder` - Which embedding space to compare
/// * `query` - Query fingerprint
/// * `memory` - Memory fingerprint
///
/// # Returns
/// Similarity in [0.0, 1.0]
pub fn compute_similarity_for_space(
    embedder: Embedder,
    query: &SemanticFingerprint,
    memory: &SemanticFingerprint,
) -> f32 {
    let query_ref = query.get(embedder);
    let memory_ref = memory.get(embedder);

    match (query_ref, memory_ref) {
        (EmbeddingRef::Dense(q), EmbeddingRef::Dense(m)) => {
            match embedder {
                Embedder::Entity => transe_similarity(q, m),
                _ => cosine_similarity(q, m),
            }
        }
        (EmbeddingRef::Sparse(q), EmbeddingRef::Sparse(m)) => jaccard_similarity(q, m),
        (EmbeddingRef::TokenLevel(q), EmbeddingRef::TokenLevel(m)) => max_sim(q, m),
        _ => {
            // Type mismatch - should never happen with valid fingerprints
            tracing::error!(
                embedder = %embedder.name(),
                "Type mismatch in compute_similarity_for_space"
            );
            0.0
        }
    }
}

/// Compute all 13 similarities between query and memory fingerprints.
///
/// Returns an array indexed by Embedder::index().
///
/// # Returns
/// Array of 13 similarity scores in [0.0, 1.0]
pub fn compute_all_similarities(
    query: &SemanticFingerprint,
    memory: &SemanticFingerprint,
) -> [f32; 13] {
    let mut scores = [0.0_f32; 13];

    for embedder in Embedder::all() {
        scores[embedder.index()] = compute_similarity_for_space(embedder, query, memory);
    }

    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Cosine Similarity Tests
    // =========================================================================

    #[test]
    fn test_cosine_identical_normalized() {
        let v: Vec<f32> = vec![0.6, 0.8, 0.0]; // Already normalized (magnitude = 1.0)
        let sim = cosine_similarity(&v, &v);
        // Raw cosine = 1.0, normalized = (1.0 + 1.0) / 2.0 = 1.0
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Identical vectors should have similarity 1.0, got {}",
            sim
        );
        println!(
            "[PASS] cosine_similarity of identical normalized vectors = {:.6}",
            sim
        );
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        // Raw cosine = 0.0, normalized = (0.0 + 1.0) / 2.0 = 0.5
        assert!(
            (sim - 0.5).abs() < 1e-5,
            "Orthogonal vectors should have similarity 0.5, got {}",
            sim
        );
        println!(
            "[PASS] cosine_similarity of orthogonal vectors = {:.6}",
            sim
        );
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        // Raw cosine = -1.0, normalized = (-1.0 + 1.0) / 2.0 = 0.0
        assert!(
            sim.abs() < 1e-5,
            "Opposite vectors should have similarity ~0.0, got {}",
            sim
        );
        assert!(sim >= 0.0 && sim <= 1.0, "Result should be in [0, 1]");
        println!(
            "[PASS] cosine_similarity of opposite vectors = {:.6}",
            sim
        );
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        // AP-10: Zero vectors return 0.0 similarity (not NaN)
        assert_eq!(sim, 0.0, "Zero vector should return 0.0 (AP-10)");
        assert!(!sim.is_nan(), "AP-10 violation: Result must not be NaN");
        println!(
            "[PASS] cosine_similarity with zero vector = {:.6} (AP-10 compliant)",
            sim
        );
    }

    #[test]
    fn test_cosine_empty_vector() {
        let a: Vec<f32> = vec![];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Empty vector should return 0.0");
        println!("[PASS] cosine_similarity with empty vector = 0.0");
    }

    #[test]
    fn test_cosine_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Dimension mismatch should return 0.0");
        println!("[PASS] cosine_similarity with dimension mismatch = 0.0");
    }

    // =========================================================================
    // Jaccard Similarity Tests
    // =========================================================================

    #[test]
    fn test_jaccard_identical() {
        let v = SparseVector::new(vec![0, 5, 10], vec![1.0, 1.0, 1.0]).unwrap();
        let sim = jaccard_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Identical sparse vectors should have similarity 1.0, got {}",
            sim
        );
        println!(
            "[PASS] jaccard_similarity of identical vectors = {:.6}",
            sim
        );
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a = SparseVector::new(vec![0, 1, 2], vec![1.0, 1.0, 1.0]).unwrap();
        let b = SparseVector::new(vec![5, 6, 7], vec![1.0, 1.0, 1.0]).unwrap();
        let sim = jaccard_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Disjoint sets should have similarity 0.0");
        println!("[PASS] jaccard_similarity of disjoint sets = {:.6}", sim);
    }

    #[test]
    fn test_jaccard_partial_overlap() {
        let a = SparseVector::new(vec![0, 1, 2], vec![1.0, 1.0, 1.0]).unwrap();
        let b = SparseVector::new(vec![1, 2, 3], vec![1.0, 1.0, 1.0]).unwrap();
        let sim = jaccard_similarity(&a, &b);
        // Intersection: {1, 2} = 2 elements
        // Union: {0, 1, 2, 3} = 4 elements
        // Jaccard = 2/4 = 0.5
        assert!((sim - 0.5).abs() < 1e-5, "Expected 0.5, got {}", sim);
        println!(
            "[PASS] jaccard_similarity with 50% overlap = {:.6}",
            sim
        );
    }

    #[test]
    fn test_jaccard_empty() {
        let a = SparseVector::empty();
        let b = SparseVector::empty();
        let sim = jaccard_similarity(&a, &b);
        // Empty sets: Defined as 0.0 by SparseVector::jaccard_similarity
        assert_eq!(sim, 0.0, "Empty vectors should have similarity 0.0");
        println!("[PASS] jaccard_similarity of empty vectors = {:.6}", sim);
    }

    // =========================================================================
    // Hamming Similarity Tests
    // =========================================================================

    #[test]
    fn test_hamming_identical() {
        let v = BinaryVector::zeros(64);
        let sim = hamming_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Identical binary vectors should have similarity 1.0, got {}",
            sim
        );
        println!(
            "[PASS] hamming_similarity of identical vectors = {:.6}",
            sim
        );
    }

    #[test]
    fn test_hamming_all_different() {
        let mut a = BinaryVector::zeros(64);
        let b = BinaryVector::zeros(64);

        // Set all bits in 'a' to 1, leave 'b' as 0
        for i in 0..64 {
            a.set_bit(i, true);
        }

        let sim = hamming_similarity(&a, &b);
        assert_eq!(sim, 0.0, "All different bits should have similarity 0.0");
        println!(
            "[PASS] hamming_similarity of opposite vectors = {:.6}",
            sim
        );
    }

    #[test]
    fn test_hamming_half_different() {
        let mut a = BinaryVector::zeros(64);
        let b = BinaryVector::zeros(64);

        // Set first 32 bits to 1 in 'a'
        for i in 0..32 {
            a.set_bit(i, true);
        }

        let sim = hamming_similarity(&a, &b);
        // 32 bits different out of 64 = 0.5 distance = 0.5 similarity
        assert!((sim - 0.5).abs() < 1e-5, "Expected 0.5, got {}", sim);
        println!(
            "[PASS] hamming_similarity with 50% difference = {:.6}",
            sim
        );
    }

    #[test]
    fn test_hamming_empty() {
        let a = BinaryVector::zeros(0);
        let b = BinaryVector::zeros(0);
        let sim = hamming_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Empty binary vectors should be identical (similarity 1.0)"
        );
        println!("[PASS] hamming_similarity of empty vectors = {:.6}", sim);
    }

    // =========================================================================
    // MaxSim (Late Interaction) Tests
    // =========================================================================

    #[test]
    fn test_max_sim_identical() {
        let tokens = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let sim = max_sim(&tokens, &tokens);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Identical token sets should have MaxSim 1.0, got {}",
            sim
        );
        println!("[PASS] max_sim of identical token sets = {:.6}", sim);
    }

    #[test]
    fn test_max_sim_empty_query() {
        let empty: Vec<Vec<f32>> = vec![];
        let tokens = vec![vec![1.0, 0.0, 0.0]];
        let sim = max_sim(&empty, &tokens);
        assert_eq!(sim, 0.0, "Empty query should return 0.0");
        println!("[PASS] max_sim with empty query = 0.0");
    }

    #[test]
    fn test_max_sim_empty_memory() {
        let tokens = vec![vec![1.0, 0.0, 0.0]];
        let empty: Vec<Vec<f32>> = vec![];
        let sim = max_sim(&tokens, &empty);
        assert_eq!(sim, 0.0, "Empty memory should return 0.0");
        println!("[PASS] max_sim with empty memory = 0.0");
    }

    #[test]
    fn test_max_sim_partial_match() {
        let query = vec![
            vec![1.0, 0.0, 0.0], // Will match first memory token perfectly
            vec![0.0, 0.0, 1.0], // Orthogonal to all memory tokens
        ];
        let memory = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let sim = max_sim(&query, &memory);
        // First query token: max sim = 1.0 (perfect match)
        // Second query token: max sim = 0.5 (orthogonal = 0.5 in normalized space)
        // Average = (1.0 + 0.5) / 2 = 0.75
        assert!((sim - 0.75).abs() < 1e-5, "Expected 0.75, got {}", sim);
        println!("[PASS] max_sim with partial match = {:.6}", sim);
    }

    // =========================================================================
    // TransE Similarity Tests
    // =========================================================================

    #[test]
    fn test_transe_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = transe_similarity(&v, &v);
        // Distance = 0, so similarity = 1 / (1 + 0) = 1.0
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Identical vectors should have TransE similarity 1.0, got {}",
            sim
        );
        println!(
            "[PASS] transe_similarity of identical vectors = {:.6}",
            sim
        );
    }

    #[test]
    fn test_transe_unit_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = transe_similarity(&a, &b);
        // Distance = 1.0, so similarity = 1 / (1 + 1) = 0.5
        assert!(
            (sim - 0.5).abs() < 1e-5,
            "Unit distance should have TransE similarity 0.5, got {}",
            sim
        );
        println!("[PASS] transe_similarity at unit distance = {:.6}", sim);
    }

    #[test]
    fn test_transe_empty_vector() {
        let a: Vec<f32> = vec![];
        let b = vec![1.0, 2.0, 3.0];
        let sim = transe_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Empty vector should return 0.0");
        println!("[PASS] transe_similarity with empty vector = 0.0");
    }

    // =========================================================================
    // compute_similarity_for_space Tests
    // =========================================================================

    #[test]
    fn test_compute_similarity_semantic() {
        let mut query = SemanticFingerprint::zeroed();
        let mut memory = SemanticFingerprint::zeroed();

        // Set identical semantic embeddings
        query.e1_semantic = vec![1.0; 1024];
        memory.e1_semantic = vec![1.0; 1024];

        let sim = compute_similarity_for_space(Embedder::Semantic, &query, &memory);
        assert!((sim - 1.0).abs() < 1e-5, "Expected 1.0, got {}", sim);
        println!(
            "[PASS] compute_similarity_for_space(Semantic) = {:.6}",
            sim
        );
    }

    #[test]
    fn test_compute_similarity_sparse() {
        let mut query = SemanticFingerprint::zeroed();
        let mut memory = SemanticFingerprint::zeroed();

        // Set identical sparse embeddings
        query.e6_sparse = SparseVector::new(vec![0, 5, 10], vec![1.0, 1.0, 1.0]).unwrap();
        memory.e6_sparse = SparseVector::new(vec![0, 5, 10], vec![1.0, 1.0, 1.0]).unwrap();

        let sim = compute_similarity_for_space(Embedder::Sparse, &query, &memory);
        assert!((sim - 1.0).abs() < 1e-5, "Expected 1.0, got {}", sim);
        println!("[PASS] compute_similarity_for_space(Sparse) = {:.6}", sim);
    }

    #[test]
    fn test_compute_similarity_late_interaction() {
        let mut query = SemanticFingerprint::zeroed();
        let mut memory = SemanticFingerprint::zeroed();

        // Set identical late interaction embeddings
        query.e12_late_interaction = vec![vec![1.0; 128], vec![0.5; 128]];
        memory.e12_late_interaction = vec![vec![1.0; 128], vec![0.5; 128]];

        let sim = compute_similarity_for_space(Embedder::LateInteraction, &query, &memory);
        assert!((sim - 1.0).abs() < 1e-5, "Expected 1.0, got {}", sim);
        println!(
            "[PASS] compute_similarity_for_space(LateInteraction) = {:.6}",
            sim
        );
    }

    #[test]
    fn test_compute_similarity_entity_uses_transe() {
        let mut query = SemanticFingerprint::zeroed();
        let mut memory = SemanticFingerprint::zeroed();

        // Set entity embeddings with unit distance
        query.e11_entity = vec![0.0; 384];
        memory.e11_entity = vec![0.0; 384];
        memory.e11_entity[0] = 1.0; // Distance = 1.0

        let sim = compute_similarity_for_space(Embedder::Entity, &query, &memory);
        // TransE: 1 / (1 + 1) = 0.5
        assert!((sim - 0.5).abs() < 1e-5, "Expected 0.5 (TransE), got {}", sim);
        println!(
            "[PASS] compute_similarity_for_space(Entity) uses TransE = {:.6}",
            sim
        );
    }

    // =========================================================================
    // compute_all_similarities Tests
    // =========================================================================

    #[test]
    fn test_compute_all_similarities() {
        let query = SemanticFingerprint::zeroed();
        let memory = SemanticFingerprint::zeroed();

        let scores = compute_all_similarities(&query, &memory);

        assert_eq!(scores.len(), 13);
        for (i, score) in scores.iter().enumerate() {
            assert!(
                *score >= 0.0 && *score <= 1.0,
                "Score {} for embedder {} out of range: {}",
                i,
                Embedder::from_index(i).unwrap().name(),
                score
            );
        }
        println!("[PASS] compute_all_similarities returns 13 valid scores");
    }

    // =========================================================================
    // Edge Case / Boundary Tests
    // =========================================================================

    #[test]
    fn test_nan_not_produced() {
        // Test various edge cases that might produce NaN
        let zero = vec![0.0, 0.0, 0.0];
        let normal = vec![1.0, 2.0, 3.0];

        let results = [
            cosine_similarity(&zero, &zero),
            cosine_similarity(&zero, &normal),
            transe_similarity(&zero, &zero),
        ];

        for (i, r) in results.iter().enumerate() {
            assert!(!r.is_nan(), "Result {} is NaN - AP-10 violation", i);
            assert!(!r.is_infinite(), "Result {} is infinite", i);
        }
        println!("[PASS] No NaN produced in edge cases (AP-10 compliance)");
    }

    #[test]
    fn test_all_results_in_range() {
        // Generate various test cases
        let test_vecs = [
            (vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]),
            (vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]),
            (vec![0.0, 0.0, 0.0], vec![1.0, 2.0, 3.0]),
            (vec![-1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]),
        ];

        for (a, b) in &test_vecs {
            let cos = cosine_similarity(a, b);
            let transe = transe_similarity(a, b);

            assert!(cos >= 0.0 && cos <= 1.0, "Cosine {} out of range", cos);
            assert!(
                transe >= 0.0 && transe <= 1.0,
                "TransE {} out of range",
                transe
            );
        }
        println!("[PASS] All results in [0.0, 1.0] range");
    }

    // =========================================================================
    // Manual Edge Case Verification (TASK-P3-004 requirement: 3+ edge cases)
    // =========================================================================

    #[test]
    fn edge_case_1_very_small_magnitudes() {
        // Edge Case 1: Very small values that might underflow
        let small_a = vec![1e-20_f32, 1e-20_f32, 1e-20_f32];
        let small_b = vec![1e-20_f32, 1e-20_f32, 1e-20_f32];
        let sim = cosine_similarity(&small_a, &small_b);

        assert!(!sim.is_nan(), "FAIL: NaN produced for very small values");
        assert!(!sim.is_infinite(), "FAIL: Infinity produced for very small values");
        assert!(sim >= 0.0 && sim <= 1.0, "FAIL: Result out of [0,1] range: {}", sim);
        println!("[PASS] Edge Case 1: Very small magnitudes handled correctly, sim = {}", sim);
    }

    #[test]
    fn edge_case_2_large_values_overflow_prevention() {
        // Edge Case 2: Large values that might overflow during magnitude calculation
        let large_a = vec![1e19_f32, 1e19_f32, 1e19_f32];
        let large_b = vec![1e19_f32, 1e19_f32, 1e19_f32];
        let sim = cosine_similarity(&large_a, &large_b);

        assert!(!sim.is_nan(), "FAIL: NaN produced for large values");
        assert!(!sim.is_infinite(), "FAIL: Infinity produced for large values");
        assert!(sim >= 0.0 && sim <= 1.0, "FAIL: Result out of [0,1] range: {}", sim);
        println!("[PASS] Edge Case 2: Large values handled correctly, sim = {}", sim);
    }

    #[test]
    fn edge_case_3_single_token_opposite_directions() {
        // Edge Case 3: Single-element late interaction with opposite directions
        let query = vec![vec![1.0_f32]];
        let memory = vec![vec![-1.0_f32]];
        let sim = max_sim(&query, &memory);

        // Raw cosine = -1.0, normalized = (-1+1)/2 = 0.0
        assert!(!sim.is_nan(), "FAIL: NaN in single-token MaxSim");
        assert!(sim >= 0.0 && sim <= 1.0, "FAIL: Result out of [0,1] range: {}", sim);
        assert!(sim.abs() < 1e-5, "Expected ~0.0 for opposite directions, got {}", sim);
        println!("[PASS] Edge Case 3: Single-token opposite MaxSim = {}", sim);
    }

    #[test]
    fn edge_case_all_13_spaces_zeroed_fingerprints() {
        // Verify all 13 embedding spaces handle zeroed fingerprints gracefully
        let zeroed = SemanticFingerprint::zeroed();
        let scores = compute_all_similarities(&zeroed, &zeroed);

        for (i, score) in scores.iter().enumerate() {
            let embedder = Embedder::from_index(i).unwrap();
            assert!(
                !score.is_nan(),
                "FAIL: NaN for {} space with zeroed fingerprints",
                embedder.name()
            );
            assert!(
                !score.is_infinite(),
                "FAIL: Infinity for {} space with zeroed fingerprints",
                embedder.name()
            );
            assert!(
                *score >= 0.0 && *score <= 1.0,
                "FAIL: {} out of [0,1] range: {}",
                embedder.name(),
                score
            );
        }
        println!("[PASS] All 13 spaces handle zeroed fingerprints without NaN/Infinity");
    }
}
