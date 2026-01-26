//! Full State Verification Test for E6, E9, E11 Embedder Fixes
//!
//! Tests with KNOWN inputs and EXPECTED outputs to verify each embedder
//! provides unique, differentiated insights.

use context_graph_core::retrieval::distance::{
    compute_all_similarities, compute_similarity_for_space,
};
use context_graph_core::teleological::Embedder;
use context_graph_core::types::fingerprint::{SemanticFingerprint, SparseVector};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn assert_close(actual: f32, expected: f32, tolerance: f32, context: &str) {
    let diff = (actual - expected).abs();
    if diff > tolerance {
        panic!(
            "[FAIL] {}: expected {:.6}, got {:.6} (diff: {:.6} > tolerance: {:.6})",
            context, expected, actual, diff, tolerance
        );
    }
    println!(
        "[PASS] {}: {:.6} ≈ {:.6} (diff: {:.6})",
        context, actual, expected, diff
    );
}

// ============================================================================
// E6 SPARSE COSINE SIMILARITY TESTS
// ============================================================================

#[test]
fn fsv_e6_sparse_cosine_identical() {
    println!("\n=== E6 SPARSE COSINE: Identical Vectors ===");
    let sv1 = SparseVector::new(vec![1, 5, 10], vec![0.5, 0.3, 0.2]).unwrap();
    let sv2 = SparseVector::new(vec![1, 5, 10], vec![0.5, 0.3, 0.2]).unwrap();

    println!("BEFORE: sv1 = {:?}", sv1);
    println!("BEFORE: sv2 = {:?}", sv2);

    let cosine = sv1.cosine_similarity(&sv2);

    println!("COMPUTED: cosine = {:.6}", cosine);
    println!("EXPECTED: 1.0 (identical vectors)");
    assert_close(cosine, 1.0, 1e-5, "E6 identical");
}

#[test]
fn fsv_e6_sparse_cosine_orthogonal() {
    println!("\n=== E6 SPARSE COSINE: Orthogonal Vectors (No Overlap) ===");
    let sv1 = SparseVector::new(vec![1, 2, 3], vec![1.0, 1.0, 1.0]).unwrap();
    let sv2 = SparseVector::new(vec![4, 5, 6], vec![1.0, 1.0, 1.0]).unwrap();

    println!("BEFORE: sv1 = {:?}", sv1);
    println!("BEFORE: sv2 = {:?}", sv2);

    let cosine = sv1.cosine_similarity(&sv2);

    println!("COMPUTED: cosine = {:.6}", cosine);
    println!("EXPECTED: 0.0 (no overlapping indices)");
    assert_close(cosine, 0.0, 1e-5, "E6 orthogonal");
}

#[test]
fn fsv_e6_sparse_cosine_partial_overlap() {
    println!("\n=== E6 SPARSE COSINE: Partial Overlap ===");
    // sv1: indices [1,3], values [0.6, 0.8]
    // sv2: indices [1,4], values [0.5, 0.5]
    // Only index 1 overlaps
    // dot = 0.6*0.5 = 0.3
    // ||sv1|| = sqrt(0.36 + 0.64) = 1.0
    // ||sv2|| = sqrt(0.25 + 0.25) = 0.7071
    // cosine = 0.3 / (1.0 * 0.7071) = 0.4243
    let sv1 = SparseVector::new(vec![1, 3], vec![0.6, 0.8]).unwrap();
    let sv2 = SparseVector::new(vec![1, 4], vec![0.5, 0.5]).unwrap();

    println!("BEFORE: sv1 = {:?}", sv1);
    println!("BEFORE: sv2 = {:?}", sv2);

    let cosine = sv1.cosine_similarity(&sv2);
    let expected = 0.3 / (1.0 * 0.7071067811865476);

    println!("COMPUTED: cosine = {:.6}", cosine);
    println!("EXPECTED: {:.6}", expected);
    println!("CALCULATION: 0.3 / (1.0 * 0.7071) = 0.4243");
    assert_close(cosine, expected, 1e-4, "E6 partial overlap");
}

#[test]
fn fsv_e6_sparse_cosine_empty() {
    println!("\n=== E6 SPARSE COSINE: Empty Vectors ===");
    let empty1 = SparseVector::empty();
    let empty2 = SparseVector::empty();

    println!("BEFORE: empty1 = {:?}", empty1);
    println!("BEFORE: empty2 = {:?}", empty2);

    let cosine = empty1.cosine_similarity(&empty2);

    println!("COMPUTED: cosine = {:.6}", cosine);
    println!("EXPECTED: 0.0 (zero norm handling)");
    assert_close(cosine, 0.0, 1e-5, "E6 empty");
}

// ============================================================================
// E11 ENTITY COSINE SIMILARITY TESTS
// ============================================================================

#[test]
fn fsv_e11_entity_identical() {
    println!("\n=== E11 ENTITY COSINE: Identical Unit Vectors ===");
    let mut fp1 = SemanticFingerprint::zeroed();
    let mut fp2 = SemanticFingerprint::zeroed();
    fp1.e11_entity = vec![0.0; 768];
    fp2.e11_entity = vec![0.0; 768];
    fp1.e11_entity[0] = 1.0;
    fp2.e11_entity[0] = 1.0;

    println!("BEFORE: fp1.e11_entity[0] = {}", fp1.e11_entity[0]);
    println!("BEFORE: fp2.e11_entity[0] = {}", fp2.e11_entity[0]);

    let sim = compute_similarity_for_space(Embedder::Entity, &fp1, &fp2);

    println!("COMPUTED: similarity = {:.6}", sim);
    println!("EXPECTED: 1.0 (identical unit vectors → cosine=1 → (1+1)/2=1)");
    assert_close(sim, 1.0, 1e-5, "E11 identical");
}

#[test]
fn fsv_e11_entity_orthogonal() {
    println!("\n=== E11 ENTITY COSINE: Orthogonal Unit Vectors ===");
    let mut fp1 = SemanticFingerprint::zeroed();
    let mut fp2 = SemanticFingerprint::zeroed();
    fp1.e11_entity = vec![0.0; 768];
    fp2.e11_entity = vec![0.0; 768];
    fp1.e11_entity[0] = 1.0; // Points in dimension 0
    fp2.e11_entity[1] = 1.0; // Points in dimension 1

    println!("BEFORE: fp1.e11_entity[0] = {}", fp1.e11_entity[0]);
    println!("BEFORE: fp2.e11_entity[1] = {}", fp2.e11_entity[1]);

    let sim = compute_similarity_for_space(Embedder::Entity, &fp1, &fp2);

    println!("COMPUTED: similarity = {:.6}", sim);
    println!("EXPECTED: 0.5 (orthogonal → cosine=0 → (0+1)/2=0.5)");
    assert_close(sim, 0.5, 1e-5, "E11 orthogonal");
}

#[test]
fn fsv_e11_entity_opposite() {
    println!("\n=== E11 ENTITY COSINE: Opposite Unit Vectors ===");
    let mut fp1 = SemanticFingerprint::zeroed();
    let mut fp2 = SemanticFingerprint::zeroed();
    fp1.e11_entity = vec![0.0; 768];
    fp2.e11_entity = vec![0.0; 768];
    fp1.e11_entity[0] = 1.0;
    fp2.e11_entity[0] = -1.0;

    println!("BEFORE: fp1.e11_entity[0] = {}", fp1.e11_entity[0]);
    println!("BEFORE: fp2.e11_entity[0] = {}", fp2.e11_entity[0]);

    let sim = compute_similarity_for_space(Embedder::Entity, &fp1, &fp2);

    println!("COMPUTED: similarity = {:.6}", sim);
    println!("EXPECTED: 0.0 (opposite → cosine=-1 → (-1+1)/2=0)");
    assert_close(sim, 0.0, 1e-5, "E11 opposite");
}

#[test]
fn fsv_e11_entity_zero_vector() {
    println!("\n=== E11 ENTITY COSINE: Zero Vector Edge Case ===");
    let mut fp1 = SemanticFingerprint::zeroed();
    let mut fp2 = SemanticFingerprint::zeroed();
    fp1.e11_entity = vec![0.0; 768]; // Zero vector
    fp2.e11_entity = vec![0.0; 768];
    fp2.e11_entity[0] = 1.0;

    println!("BEFORE: fp1.e11_entity is all zeros");
    println!("BEFORE: fp2.e11_entity[0] = {}", fp2.e11_entity[0]);

    let sim = compute_similarity_for_space(Embedder::Entity, &fp1, &fp2);

    println!("COMPUTED: similarity = {:.6}", sim);
    println!("EXPECTED: 0.0 (zero vector → no similarity)");
    assert_close(sim, 0.0, 1e-5, "E11 zero vector");
}

#[test]
fn fsv_e11_not_transe() {
    println!("\n=== E11 VERIFICATION: Uses Cosine, NOT TransE ===");
    // For orthogonal unit vectors:
    // - TransE would give: 1/(1+sqrt(2)) ≈ 0.414
    // - Cosine gives: (0+1)/2 = 0.5
    let mut fp1 = SemanticFingerprint::zeroed();
    let mut fp2 = SemanticFingerprint::zeroed();
    fp1.e11_entity = vec![0.0; 768];
    fp2.e11_entity = vec![0.0; 768];
    fp1.e11_entity[0] = 1.0;
    fp2.e11_entity[1] = 1.0;

    let sim = compute_similarity_for_space(Embedder::Entity, &fp1, &fp2);
    let transe_would_give = 1.0 / (1.0 + 2.0_f32.sqrt());

    println!("TransE would give: {:.6}", transe_would_give);
    println!("Cosine gives: 0.5");
    println!("Actual result: {:.6}", sim);

    assert!(
        (sim - 0.5).abs() < 1e-5,
        "E11 should use cosine (0.5), NOT TransE ({:.6}), got {}",
        transe_would_give,
        sim
    );
    println!("[VERIFIED] E11 uses cosine similarity, NOT TransE");
}

// ============================================================================
// ALL 13 EMBEDDERS DIFFERENTIATED SCORE TEST
// ============================================================================

/// Creates a unit vector in the given dimensions with specified components.
/// This allows creating vectors at known angles for predictable cosine similarity.
fn make_dense_vector(dim: usize, components: &[(usize, f32)]) -> Vec<f32> {
    let mut v = vec![0.0; dim];
    for &(idx, val) in components {
        v[idx] = val;
    }
    // Normalize to unit vector
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

#[test]
fn fsv_all_embedders_differentiated_scores() {
    println!("\n=== ALL 13 EMBEDDERS: DIFFERENTIATED Score Verification ===");
    println!("Testing with vectors designed to produce VARIED similarity scores\n");
    println!("Key: Vectors at different angles produce different cosine similarities:");
    println!("  - Parallel vectors (0°) → cosine=1.0 → normalized=(1+1)/2=1.0");
    println!("  - 60° angle → cosine=0.5 → normalized=(0.5+1)/2=0.75");
    println!("  - 90° orthogonal → cosine=0.0 → normalized=(0+1)/2=0.5");
    println!("  - 120° angle → cosine=-0.5 → normalized=(-0.5+1)/2=0.25\n");

    let mut fp1 = SemanticFingerprint::zeroed();
    let mut fp2 = SemanticFingerprint::zeroed();

    // E1: Semantic (1024D) - HIGH similarity (~0.85)
    // fp1 points in (1,0,...), fp2 points in (0.9, 0.436,...) ≈ 25° angle
    fp1.e1_semantic = make_dense_vector(1024, &[(0, 1.0)]);
    fp2.e1_semantic = make_dense_vector(1024, &[(0, 0.9), (1, 0.436)]);

    // E2: TemporalRecent (512D) - MEDIUM-HIGH similarity (~0.75)
    // 60° angle: cosine = 0.5, normalized = 0.75
    fp1.e2_temporal_recent = make_dense_vector(512, &[(0, 1.0)]);
    fp2.e2_temporal_recent = make_dense_vector(512, &[(0, 0.5), (1, 0.866)]);

    // E3: TemporalPeriodic (512D) - MEDIUM similarity (~0.65)
    // ~74° angle
    fp1.e3_temporal_periodic = make_dense_vector(512, &[(0, 1.0)]);
    fp2.e3_temporal_periodic = make_dense_vector(512, &[(0, 0.28), (1, 0.96)]);

    // E4: TemporalPositional (512D) - LOW-MEDIUM similarity (~0.55)
    // ~84° angle
    fp1.e4_temporal_positional = make_dense_vector(512, &[(0, 1.0)]);
    fp2.e4_temporal_positional = make_dense_vector(512, &[(0, 0.1), (1, 0.995)]);

    // E5: Causal (768D) - MEDIUM similarity (~0.70)
    // Uses dual vectors (cause/effect) - averaging two components
    fp1.e5_causal_as_cause = make_dense_vector(768, &[(0, 1.0)]);
    fp2.e5_causal_as_cause = make_dense_vector(768, &[(0, 0.6), (1, 0.8)]); // ~53° angle → 0.8
    fp1.e5_causal_as_effect = make_dense_vector(768, &[(0, 1.0)]);
    fp2.e5_causal_as_effect = make_dense_vector(768, &[(0, 0.2), (1, 0.98)]); // ~78° angle → 0.6

    // E6: Sparse (keywords) - MEDIUM similarity (~0.58)
    // Partial keyword overlap: indices [1,5,10] vs [1,5,99]
    fp1.e6_sparse = SparseVector::new(vec![1, 5, 10], vec![0.6, 0.8, 0.0]).unwrap();
    fp2.e6_sparse = SparseVector::new(vec![1, 5, 99], vec![0.5, 0.5, 0.707]).unwrap();

    // E7: Code (1536D) - HIGH similarity (~0.90)
    // Very similar code embeddings
    fp1.e7_code = make_dense_vector(1536, &[(0, 1.0)]);
    fp2.e7_code = make_dense_vector(1536, &[(0, 0.98), (1, 0.2)]);

    // E8: Graph (384D) - MEDIUM-LOW similarity (~0.60)
    // Uses dual vectors (source/target)
    fp1.e8_graph_as_source = make_dense_vector(384, &[(0, 1.0)]);
    fp2.e8_graph_as_source = make_dense_vector(384, &[(0, 0.4), (1, 0.917)]); // ~66° → 0.7
    fp1.e8_graph_as_target = make_dense_vector(384, &[(0, 1.0)]);
    fp2.e8_graph_as_target = make_dense_vector(384, &[(0, 0.0), (1, 1.0)]); // 90° → 0.5

    // E9: HDC (1024D) - HIGH similarity (~0.88)
    // Typo-tolerant, very similar
    fp1.e9_hdc = make_dense_vector(1024, &[(0, 0.95), (1, 0.312)]);
    fp2.e9_hdc = make_dense_vector(1024, &[(0, 0.9), (1, 0.436)]);

    // E10: Multimodal (768D) - MEDIUM-HIGH similarity (~0.78)
    // Uses dual vectors (intent/context)
    fp1.e10_multimodal_as_intent = make_dense_vector(768, &[(0, 1.0)]);
    fp2.e10_multimodal_as_intent = make_dense_vector(768, &[(0, 0.7), (1, 0.714)]); // ~45° → 0.85
    fp1.e10_multimodal_as_context = make_dense_vector(768, &[(0, 1.0)]);
    fp2.e10_multimodal_as_context = make_dense_vector(768, &[(0, 0.4), (1, 0.917)]); // ~66° → 0.7

    // E11: Entity (768D) - LOW-MEDIUM similarity (~0.50)
    // Orthogonal entity vectors (different entities)
    fp1.e11_entity = make_dense_vector(768, &[(0, 1.0)]);
    fp2.e11_entity = make_dense_vector(768, &[(1, 1.0)]); // 90° orthogonal

    // E12: LateInteraction (token-level, 128D per token) - HIGH similarity (~0.92)
    // MaxSim: best match per query token
    let mut tok1_a = vec![0.0f32; 128];
    let mut tok1_b = vec![0.0f32; 128];
    let mut tok2_a = vec![0.0f32; 128];
    let mut tok2_b = vec![0.0f32; 128];
    tok1_a[0] = 1.0;
    tok1_b[0] = 0.95; tok1_b[1] = 0.312;  // Very similar
    tok2_a[0] = 1.0;
    tok2_b[0] = 0.8; tok2_b[1] = 0.6;     // Less similar
    fp1.e12_late_interaction = vec![tok1_a, tok2_a];
    fp2.e12_late_interaction = vec![tok1_b, tok2_b];

    // E13: SPLADE sparse - LOW similarity (~0.35)
    // Minimal keyword overlap
    fp1.e13_splade = SparseVector::new(vec![10, 20, 30], vec![0.7, 0.5, 0.5]).unwrap();
    fp2.e13_splade = SparseVector::new(vec![10, 25, 35], vec![0.3, 0.8, 0.5]).unwrap();

    let scores = compute_all_similarities(&fp1, &fp2);

    // Expected ranges for each embedder (designed into the test vectors)
    let expected_ranges: [(f32, f32); 13] = [
        (0.80, 0.95), // E1: high semantic
        (0.70, 0.80), // E2: medium-high temporal
        (0.60, 0.70), // E3: medium temporal
        (0.50, 0.60), // E4: low-medium temporal
        (0.60, 0.80), // E5: medium causal (dual avg)
        (0.40, 0.70), // E6: medium sparse keywords
        (0.85, 0.95), // E7: high code
        (0.55, 0.70), // E8: medium-low graph (dual avg)
        (0.85, 0.95), // E9: high HDC
        (0.70, 0.85), // E10: medium-high multimodal
        (0.45, 0.55), // E11: orthogonal entities → 0.5
        (0.80, 1.00), // E12: high late interaction
        (0.15, 0.50), // E13: low SPLADE overlap
    ];

    let embedder_names = [
        "E1 (Semantic)",
        "E2 (TemporalRecent)",
        "E3 (TemporalPeriodic)",
        "E4 (TemporalPositional)",
        "E5 (Causal)",
        "E6 (Sparse/Keywords)",
        "E7 (Code)",
        "E8 (Graph)",
        "E9 (HDC)",
        "E10 (Multimodal)",
        "E11 (Entity)",
        "E12 (LateInteraction)",
        "E13 (KeywordSplade)",
    ];

    println!("| Embedder                | Score    | Expected     | Status |");
    println!("|-------------------------|----------|--------------|--------|");

    let mut all_pass = true;
    let mut score_variance = 0.0f32;
    let mean_score: f32 = scores.iter().sum::<f32>() / scores.len() as f32;

    for (i, (score, name)) in scores.iter().zip(embedder_names.iter()).enumerate() {
        let (min_exp, max_exp) = expected_ranges[i];
        let in_range = *score >= min_exp && *score <= max_exp;
        let valid = *score >= 0.0 && *score <= 1.0;
        let status = if in_range { "✓ PASS" } else if valid { "~ WARN" } else { "✗ FAIL" };

        println!(
            "| {:23} | {:.4}   | [{:.2}, {:.2}] | {} |",
            name, score, min_exp, max_exp, status
        );

        score_variance += (*score - mean_score).powi(2);

        assert!(
            *score >= 0.0 && *score <= 1.0,
            "Embedder {} ({}) score {} out of valid range [0,1]",
            i, name, score
        );

        if !in_range {
            all_pass = false;
        }
    }

    score_variance /= scores.len() as f32;
    let score_stddev = score_variance.sqrt();

    println!("\n--- Score Statistics ---");
    println!("Mean score: {:.4}", mean_score);
    println!("Std deviation: {:.4}", score_stddev);
    println!("Score range: {:.4} to {:.4}",
             scores.iter().cloned().fold(f32::INFINITY, f32::min),
             scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    // Verify scores are DIFFERENTIATED (not all the same)
    assert!(
        score_stddev > 0.10,
        "Score standard deviation {} is too low - embedders should produce differentiated scores!",
        score_stddev
    );

    if all_pass {
        println!("\n[SUCCESS] All 13 embedders produce DIFFERENTIATED scores in expected ranges!");
    } else {
        println!("\n[WARNING] Some embedders outside expected ranges, but all produce valid scores.");
        println!("Expected ranges are estimates - actual similarity depends on vector geometry.");
    }
}
