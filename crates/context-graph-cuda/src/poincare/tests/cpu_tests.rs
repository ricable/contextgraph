//! CPU reference implementation tests for Poincare distance.
//!
//! REAL DATA ONLY, NO MOCKS (per constitution REQ-KG-TEST)

use crate::poincare::{poincare_distance_batch_cpu, poincare_distance_cpu};

// ========== CPU Reference Implementation Tests ==========

#[test]
fn test_cpu_distance_same_point_is_zero() {
    let point = [0.1f32; 64];
    let dist = poincare_distance_cpu(&point, &point, -1.0);
    assert!(
        dist.abs() < 1e-5,
        "Distance to self should be ~0, got {}",
        dist
    );
}

#[test]
fn test_cpu_distance_origin_to_origin() {
    let origin = [0.0f32; 64];
    let dist = poincare_distance_cpu(&origin, &origin, -1.0);
    assert!(dist.abs() < 1e-6);
}

#[test]
fn test_cpu_distance_is_symmetric() {
    let mut x = [0.0f32; 64];
    let mut y = [0.0f32; 64];
    x[0] = 0.3;
    y[0] = 0.6;

    let d1 = poincare_distance_cpu(&x, &y, -1.0);
    let d2 = poincare_distance_cpu(&y, &x, -1.0);

    assert!(
        (d1 - d2).abs() < 1e-5,
        "Distance should be symmetric: {} vs {}",
        d1,
        d2
    );
}

#[test]
fn test_cpu_distance_is_nonnegative() {
    let mut x = [0.0f32; 64];
    let mut y = [0.0f32; 64];
    x[0] = 0.3;
    y[0] = -0.5;

    let dist = poincare_distance_cpu(&x, &y, -1.0);
    assert!(dist >= 0.0, "Distance should be non-negative");
}

#[test]
fn test_cpu_distance_from_origin() {
    let origin = [0.0f32; 64];
    let mut point = [0.0f32; 64];
    point[0] = 0.5;

    let dist = poincare_distance_cpu(&origin, &point, -1.0);

    // For c=-1, using direct formula from origin:
    // d(0, p) = 2 * arctanh(sqrt(||p||^2 / (1 - ||p||^2)))
    // With ||p||^2 = 0.25: arg = sqrt(0.25 / 0.75) = sqrt(1/3)
    // d(0, p) = 2 * arctanh(sqrt(1/3)) ~ 1.317
    let norm_sq = 0.25_f32;
    let arg = (norm_sq / (1.0 - norm_sq)).sqrt();
    let expected = 2.0 * arg.atanh();
    assert!(
        (dist - expected).abs() < 1e-4,
        "Expected {}, got {}",
        expected,
        dist
    );
}

#[test]
fn test_cpu_distance_monotonic_from_origin() {
    let origin = [0.0f32; 64];

    let mut p1 = [0.0f32; 64];
    p1[0] = 0.1;
    let mut p2 = [0.0f32; 64];
    p2[0] = 0.5;
    let mut p3 = [0.0f32; 64];
    p3[0] = 0.9;

    let d1 = poincare_distance_cpu(&origin, &p1, -1.0);
    let d2 = poincare_distance_cpu(&origin, &p2, -1.0);
    let d3 = poincare_distance_cpu(&origin, &p3, -1.0);

    assert!(d1 < d2, "d(0, 0.1) < d(0, 0.5)");
    assert!(d2 < d3, "d(0, 0.5) < d(0, 0.9)");
}

#[test]
fn test_cpu_distance_near_boundary_large() {
    let origin = [0.0f32; 64];
    let mut near_boundary = [0.0f32; 64];
    near_boundary[0] = 0.99;

    let dist = poincare_distance_cpu(&origin, &near_boundary, -1.0);

    // Near boundary, hyperbolic distance grows rapidly
    assert!(
        dist > 4.0,
        "Distance near boundary should be large, got {}",
        dist
    );
}

#[test]
fn test_cpu_distance_custom_curvature() {
    let mut x = [0.0f32; 64];
    let mut y = [0.0f32; 64];
    x[0] = 0.3;
    y[0] = 0.6;

    let d1 = poincare_distance_cpu(&x, &y, -1.0);
    let d2 = poincare_distance_cpu(&x, &y, -0.5);

    // Different curvatures should give different distances
    assert!((d1 - d2).abs() > 0.01, "Curvature should affect distance");
}

#[test]
fn test_cpu_batch_dimensions() {
    let n_queries = 10;
    let n_database = 20;

    let queries: Vec<f32> = (0..(n_queries * 64)).map(|i| (i as f32) * 0.001).collect();
    let database: Vec<f32> = (0..(n_database * 64))
        .map(|i| (i as f32) * 0.0005)
        .collect();

    let distances = poincare_distance_batch_cpu(&queries, &database, n_queries, n_database, -1.0);

    assert_eq!(distances.len(), n_queries * n_database);
}

#[test]
fn test_cpu_batch_matches_single() {
    let n_queries = 5;
    let n_database = 5;

    // Create random-ish points (deterministic)
    let queries: Vec<f32> = (0..(n_queries * 64))
        .map(|i| ((i * 17 + 3) % 100) as f32 * 0.005 - 0.25)
        .collect();
    let database: Vec<f32> = (0..(n_database * 64))
        .map(|i| ((i * 23 + 7) % 100) as f32 * 0.005 - 0.25)
        .collect();

    let batch_distances =
        poincare_distance_batch_cpu(&queries, &database, n_queries, n_database, -1.0);

    // Compare with individual computations
    for i in 0..n_queries {
        for j in 0..n_database {
            let q: &[f32; 64] = queries[i * 64..(i + 1) * 64].try_into().unwrap();
            let db: &[f32; 64] = database[j * 64..(j + 1) * 64].try_into().unwrap();
            let single_dist = poincare_distance_cpu(q, db, -1.0);
            let batch_dist = batch_distances[i * n_database + j];

            assert!(
                (single_dist - batch_dist).abs() < 1e-5,
                "Mismatch at [{}, {}]: {} vs {}",
                i,
                j,
                single_dist,
                batch_dist
            );
        }
    }
}

// ========== Edge Case Tests ==========

#[test]
fn test_cpu_edge_case_zero_vectors() {
    let zero = [0.0f32; 64];
    let dist = poincare_distance_cpu(&zero, &zero, -1.0);
    assert!(dist.abs() < 1e-6, "Zero to zero should be 0");
}

#[test]
fn test_cpu_edge_case_boundary_points() {
    // Points very close to boundary (norm ~ 0.99)
    let scale = 0.99 / (64.0_f32).sqrt();
    let boundary_point: [f32; 64] = [scale; 64];

    let dist = poincare_distance_cpu(&boundary_point, &boundary_point, -1.0);
    assert!(dist.abs() < 1e-4, "Same point distance should be ~0");
    assert!(dist.is_finite(), "Distance must be finite");
}

#[test]
fn test_cpu_edge_case_opposite_points() {
    // Points on opposite sides of the ball
    let mut x = [0.0f32; 64];
    let mut y = [0.0f32; 64];
    x[0] = 0.9;
    y[0] = -0.9;

    let dist = poincare_distance_cpu(&x, &y, -1.0);
    assert!(
        dist > 3.0,
        "Opposite points should have large distance: {}",
        dist
    );
    assert!(dist.is_finite(), "Distance must be finite");
}
