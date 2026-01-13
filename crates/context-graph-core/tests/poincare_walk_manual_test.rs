//! Manual Testing for Poincare Walk Math Utilities
//!
//! This file contains comprehensive manual tests with synthetic data
//! to verify all mathematical operations produce correct outputs.
//!
//! Tests use deterministic RNG (seed=42) for reproducibility.
//! All expected outputs are computed from known mathematical formulas.

#![allow(clippy::needless_range_loop)] // Intentional index-based iteration for clarity in tests

use context_graph_core::dream::{
    PoincareBallConfig,
    direction_toward,
    geodesic_distance,
    inner_product_64,
    is_far_from_all,
    mobius_add,
    norm_64,
    norm_squared_64,
    project_to_ball,
    random_direction,
    scale_direction,
    softmax_temperature,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn make_rng() -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(42)
}

fn point_at_norm(norm: f32) -> [f32; 64] {
    let mut p = [0.0f32; 64];
    p[0] = norm;
    p
}

// ============================================================================
// MANUAL TEST 1: norm_squared_64 with known synthetic data
// ============================================================================
#[test]
fn manual_test_norm_squared_64_synthetic() {
    // Test Case 1: 64 elements of 0.125
    // Expected: 64 * 0.125^2 = 64 * 0.015625 = 1.0
    let v = [0.125f32; 64];
    let result = norm_squared_64(&v);
    let expected = 1.0f32;
    assert!(
        (result - expected).abs() < 1e-6,
        "FAIL: norm_squared_64([0.125; 64]) = {}, expected {}",
        result, expected
    );
    println!("PASS: norm_squared_64([0.125; 64]) = {} (expected: {})", result, expected);

    // Test Case 2: All zeros
    let zeros = [0.0f32; 64];
    let result = norm_squared_64(&zeros);
    assert!(
        result.abs() < 1e-10,
        "FAIL: norm_squared_64([0; 64]) = {}, expected 0.0",
        result
    );
    println!("PASS: norm_squared_64([0; 64]) = {} (expected: 0.0)", result);

    // Test Case 3: First element = 0.5, rest = 0
    let single = point_at_norm(0.5);
    let result = norm_squared_64(&single);
    let expected = 0.25f32;
    assert!(
        (result - expected).abs() < 1e-6,
        "FAIL: norm_squared_64([0.5, 0..]) = {}, expected {}",
        result, expected
    );
    println!("PASS: norm_squared_64([0.5, 0..]) = {} (expected: {})", result, expected);
}

// ============================================================================
// MANUAL TEST 2: norm_64 with known synthetic data
// ============================================================================
#[test]
fn manual_test_norm_64_synthetic() {
    // Test Case: sqrt(64 * 0.125^2) = sqrt(1.0) = 1.0
    let v = [0.125f32; 64];
    let result = norm_64(&v);
    let expected = 1.0f32;
    assert!(
        (result - expected).abs() < 1e-6,
        "FAIL: norm_64([0.125; 64]) = {}, expected {}",
        result, expected
    );
    println!("PASS: norm_64([0.125; 64]) = {} (expected: {})", result, expected);
}

// ============================================================================
// MANUAL TEST 3: inner_product_64 with known synthetic data
// ============================================================================
#[test]
fn manual_test_inner_product_64_synthetic() {
    // Test Case 1: Orthogonal vectors (0 product)
    let mut a = [0.0f32; 64];
    let mut b = [0.0f32; 64];
    a[0] = 1.0;
    b[1] = 1.0;
    let result = inner_product_64(&a, &b);
    assert!(
        result.abs() < 1e-10,
        "FAIL: orthogonal inner_product = {}, expected 0.0",
        result
    );
    println!("PASS: orthogonal inner_product = {} (expected: 0.0)", result);

    // Test Case 2: Self-inner product = norm squared
    let v = [0.125f32; 64];
    let result = inner_product_64(&v, &v);
    let expected = 1.0f32; // = norm_squared
    assert!(
        (result - expected).abs() < 1e-6,
        "FAIL: self inner_product = {}, expected {}",
        result, expected
    );
    println!("PASS: self inner_product = {} (expected: {})", result, expected);
}

// ============================================================================
// MANUAL TEST 4: mobius_add with origin (identity property)
// ============================================================================
#[test]
fn manual_test_mobius_add_origin_identity() {
    let config = PoincareBallConfig::default();
    let origin = [0.0f32; 64];
    let v = point_at_norm(0.5);

    // Adding v to origin should give v (identity property)
    let result = mobius_add(&origin, &v, &config);

    assert!(
        (result[0] - 0.5).abs() < 1e-5,
        "FAIL: mobius_add(origin, [0.5, 0..]) first element = {}, expected 0.5",
        result[0]
    );
    println!("PASS: mobius_add(origin, [0.5, 0..]) first element = {} (expected: ~0.5)", result[0]);

    // All other elements should be 0
    for i in 1..64 {
        assert!(
            result[i].abs() < 1e-6,
            "FAIL: mobius_add(origin, [0.5, 0..]) element {} = {}, expected 0.0",
            i, result[i]
        );
    }
    println!("PASS: All other 63 elements are ~0.0");
}

// ============================================================================
// MANUAL TEST 5: geodesic_distance (symmetry and self-distance)
// ============================================================================
#[test]
fn manual_test_geodesic_distance_properties() {
    let config = PoincareBallConfig::default();

    // Test Case 1: Self-distance = 0
    let p = point_at_norm(0.5);
    let result = geodesic_distance(&p, &p, &config);
    assert!(
        result.abs() < 1e-6,
        "FAIL: geodesic_distance(p, p) = {}, expected 0.0",
        result
    );
    println!("PASS: geodesic_distance(p, p) = {} (expected: 0.0)", result);

    // Test Case 2: Symmetry
    let mut q = [0.0f32; 64];
    q[1] = 0.4;
    let d1 = geodesic_distance(&p, &q, &config);
    let d2 = geodesic_distance(&q, &p, &config);
    assert!(
        (d1 - d2).abs() < 1e-5,
        "FAIL: geodesic_distance asymmetric: d(p,q)={}, d(q,p)={}",
        d1, d2
    );
    println!("PASS: geodesic_distance symmetry: d(p,q)={}, d(q,p)={}", d1, d2);

    // Test Case 3: Known distance from origin
    let origin = [0.0f32; 64];
    let point = point_at_norm(0.5);
    let dist = geodesic_distance(&origin, &point, &config);
    // For c=1, d(0, p) = 2 * atanh(||p||) = 2 * atanh(0.5) ≈ 1.0986
    let expected = 2.0 * 0.5f32.atanh();
    assert!(
        (dist - expected).abs() < 0.01,
        "FAIL: geodesic from origin = {}, expected ~{}",
        dist, expected
    );
    println!("PASS: geodesic_distance(origin, [0.5, 0..]) = {} (expected: ~{})", dist, expected);
}

// ============================================================================
// MANUAL TEST 6: softmax_temperature with known synthetic data
// ============================================================================
#[test]
fn manual_test_softmax_temperature_synthetic() {
    // Test Case 1: Uniform scores -> uniform probabilities
    let scores = vec![1.0, 1.0, 1.0, 1.0];
    let probs = softmax_temperature(&scores, 2.0);

    let sum: f32 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "FAIL: softmax sum = {}, expected 1.0",
        sum
    );
    println!("PASS: softmax sum = {} (expected: 1.0)", sum);

    for (i, &p) in probs.iter().enumerate() {
        assert!(
            (p - 0.25).abs() < 0.01,
            "FAIL: uniform softmax prob[{}] = {}, expected ~0.25",
            i, p
        );
    }
    println!("PASS: uniform scores -> uniform probs: {:?}", probs);

    // Test Case 2: High temperature = more uniform
    let scores = vec![1.0, 10.0]; // Very different scores
    let probs_high = softmax_temperature(&scores, 10.0);
    let probs_low = softmax_temperature(&scores, 0.1);

    let range_high = probs_high[1] - probs_high[0];
    let range_low = probs_low[1] - probs_low[0];

    assert!(
        range_high < range_low,
        "FAIL: high temp should be more uniform. high_range={}, low_range={}",
        range_high, range_low
    );
    println!("PASS: high temp more uniform. T=10.0 range={}, T=0.1 range={}", range_high, range_low);
}

// ============================================================================
// MANUAL TEST 7: random_direction is unit length
// ============================================================================
#[test]
fn manual_test_random_direction_unit_length() {
    let mut rng = make_rng();

    for i in 0..10 {
        let dir = random_direction(&mut rng);
        let norm = norm_64(&dir);
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "FAIL: random_direction {} norm = {}, expected 1.0",
            i, norm
        );
    }
    println!("PASS: 10 random_direction calls all produced unit vectors");
}

// ============================================================================
// MANUAL TEST 8: project_to_ball behavior
// ============================================================================
#[test]
fn manual_test_project_to_ball_behavior() {
    let config = PoincareBallConfig::default();

    // Test Case 1: Inside point unchanged
    let mut inside = point_at_norm(0.5);
    let original = inside;
    let projected = project_to_ball(&mut inside, &config);

    assert!(!projected, "FAIL: inside point should not need projection");
    assert_eq!(inside, original, "FAIL: inside point should be unchanged");
    println!("PASS: inside point (norm=0.5) unchanged by projection");

    // Test Case 2: Outside point projected inside
    let mut outside = point_at_norm(1.5);
    let projected = project_to_ball(&mut outside, &config);
    let new_norm = norm_64(&outside);

    assert!(projected, "FAIL: outside point should need projection");
    assert!(
        new_norm < config.max_norm,
        "FAIL: projected norm {} >= max_norm {}",
        new_norm, config.max_norm
    );
    println!("PASS: outside point (norm=1.5) projected to norm={}", new_norm);
}

// ============================================================================
// MANUAL TEST 9: is_far_from_all with constitution semantic_leap threshold
// ============================================================================
#[test]
fn manual_test_is_far_from_all_semantic_leap() {
    let config = PoincareBallConfig::default();

    // Test Case 1: Empty references = always far
    let point = point_at_norm(0.5);
    let refs: Vec<[f32; 64]> = vec![];
    let result = is_far_from_all(&point, &refs, 0.7, &config);
    assert!(result, "FAIL: should be far from empty set");
    println!("PASS: is_far_from_all with empty refs = true");

    // Test Case 2: Close reference = not far
    let ref_point = point_at_norm(0.51); // Very close to point
    let refs = vec![ref_point];
    let result = is_far_from_all(&point, &refs, 0.7, &config);
    assert!(!result, "FAIL: should NOT be far from close point");
    println!("PASS: is_far_from_all with close ref = false");

    // Test Case 3: Far reference = far
    let mut far_point = [0.0f32; 64];
    far_point[1] = 0.9; // On a different axis, far away
    let refs = vec![far_point];
    let dist = geodesic_distance(&point, &far_point, &config);
    let result = is_far_from_all(&point, &refs, 0.7, &config);

    println!("Distance to far_point: {}", dist);
    // Only assert "is far" if distance actually >= 0.7
    assert_eq!(
        result, dist >= 0.7,
        "FAIL: is_far_from_all result {} inconsistent with distance {} vs threshold 0.7",
        result, dist
    );
    println!("PASS: is_far_from_all correctly evaluates distance {} vs threshold 0.7", dist);
}

// ============================================================================
// MANUAL TEST 10: direction_toward reduces distance
// ============================================================================
#[test]
fn manual_test_direction_toward_convergence() {
    let config = PoincareBallConfig::default();
    let p = point_at_norm(0.3);
    let mut q = [0.0f32; 64];
    q[1] = 0.4;

    let dir = direction_toward(&p, &q, &config);

    // Take a small step in the direction
    let mut scaled_dir = dir;
    for x in scaled_dir.iter_mut() {
        *x *= 0.01;
    }
    let p_new = mobius_add(&p, &scaled_dir, &config);

    let dist_before = geodesic_distance(&p, &q, &config);
    let dist_after = geodesic_distance(&p_new, &q, &config);

    assert!(
        dist_after < dist_before,
        "FAIL: direction_toward should reduce distance: {} -> {}",
        dist_before, dist_after
    );
    println!(
        "PASS: direction_toward reduced distance from {} to {} (Δ={})",
        dist_before, dist_after, dist_before - dist_after
    );
}

// ============================================================================
// MANUAL TEST 11: scale_direction boundary behavior
// ============================================================================
#[test]
fn manual_test_scale_direction_boundary() {
    let config = PoincareBallConfig::default();
    let mut dir = [0.0f32; 64];
    dir[0] = 1.0; // Unit direction

    // Step at origin (norm=0) should be larger than at boundary (norm=0.9)
    let scaled_origin = scale_direction(&dir, 0.1, 0.0, &config);
    let scaled_boundary = scale_direction(&dir, 0.1, 0.9, &config);

    let norm_origin = norm_64(&scaled_origin);
    let norm_boundary = norm_64(&scaled_boundary);

    assert!(
        norm_origin > norm_boundary,
        "FAIL: step at origin ({}) should be > step at boundary ({})",
        norm_origin, norm_boundary
    );
    println!(
        "PASS: scale_direction at origin (norm={}) > at boundary (norm={})",
        norm_origin, norm_boundary
    );
    println!("  Origin step magnitude: {}", norm_origin);
    println!("  Boundary step magnitude: {}", norm_boundary);
    println!("  Ratio: {:.2}x", norm_origin / norm_boundary);
}

// ============================================================================
// MANUAL TEST 12: Triangle inequality for geodesic distance
// ============================================================================
#[test]
fn manual_test_geodesic_triangle_inequality() {
    let config = PoincareBallConfig::default();

    // Three well-separated points
    let p = point_at_norm(0.3);
    let mut q = [0.0f32; 64];
    q[1] = 0.4;
    let mut r = [0.0f32; 64];
    r[2] = 0.5;

    let d_pq = geodesic_distance(&p, &q, &config);
    let d_qr = geodesic_distance(&q, &r, &config);
    let d_pr = geodesic_distance(&p, &r, &config);

    assert!(
        d_pr <= d_pq + d_qr + 1e-5,
        "FAIL: triangle inequality: d(p,r)={} > d(p,q)+d(q,r)={}",
        d_pr, d_pq + d_qr
    );
    println!("PASS: Triangle inequality holds:");
    println!("  d(p,q) = {}", d_pq);
    println!("  d(q,r) = {}", d_qr);
    println!("  d(p,r) = {} <= d(p,q)+d(q,r) = {}", d_pr, d_pq + d_qr);
}

// ============================================================================
// MANUAL TEST 13: Mobius addition stays in ball
// ============================================================================
#[test]
fn manual_test_mobius_stays_in_ball() {
    let config = PoincareBallConfig::default();
    let mut rng = make_rng();

    // Generate random points and velocities, verify result stays in ball
    for i in 0..20 {
        let mut p = random_direction(&mut rng);
        for x in p.iter_mut() { *x *= 0.5; } // Scale to be inside ball

        let mut v = random_direction(&mut rng);
        for x in v.iter_mut() { *x *= 0.3; }

        let result = mobius_add(&p, &v, &config);
        let result_norm = norm_64(&result);

        assert!(
            result_norm < config.max_norm,
            "FAIL: mobius_add iteration {} produced norm={} >= max_norm={}",
            i, result_norm, config.max_norm
        );
    }
    println!("PASS: 20 random mobius_add operations all stayed in ball");
}

// ============================================================================
// MANUAL TEST 14: Reproducibility with deterministic RNG
// ============================================================================
#[test]
fn manual_test_reproducibility() {
    let mut rng1 = make_rng();
    let mut rng2 = make_rng();

    let dir1 = random_direction(&mut rng1);
    let dir2 = random_direction(&mut rng2);

    assert_eq!(dir1, dir2, "FAIL: same seed should produce same direction");
    println!("PASS: Deterministic RNG produces reproducible results");
    println!("  First 3 elements: [{:.6}, {:.6}, {:.6}]", dir1[0], dir1[1], dir1[2]);
}

// ============================================================================
// MANUAL TEST 15: Constitution value validation
// ============================================================================
#[test]
fn manual_test_constitution_values() {
    // Verify default config matches constitution
    let config = PoincareBallConfig::default();

    assert!(
        (config.max_norm - 0.99999).abs() < 1e-8,
        "FAIL: max_norm should be 0.99999"
    );
    assert!(
        (config.epsilon - 1e-7).abs() < 1e-10,
        "FAIL: epsilon should be 1e-7"
    );
    assert!(
        (config.curvature - (-1.0)).abs() < 1e-8,
        "FAIL: curvature should be -1.0"
    );

    config.validate(); // Should not panic

    println!("PASS: PoincareBallConfig default values match constitution");
    println!("  max_norm = {}", config.max_norm);
    println!("  epsilon = {:e}", config.epsilon);
    println!("  curvature = {}", config.curvature);
}
