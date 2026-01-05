//! CPU reference implementation for Poincare distance computation.
//!
//! Provides reference implementations for testing GPU kernel correctness
//! and as a fallback when GPU is unavailable.
//!
//! # Formula
//!
//! ```text
//! d(x,y) = (2/sqrt(c)) * arctanh(sqrt(c * ||x-y||^2 / ((1-c*||x||^2)(1-c*||y||^2))))
//! ```
//!
//! where c = |curvature| (always positive for hyperbolic space).

use super::constants::POINCARE_EPS;

/// Compute Poincare distance on CPU (reference implementation).
///
/// This is the direct formula implementation, mathematically equivalent to
/// the GPU kernel. Used for testing and as a fallback when GPU is unavailable.
///
/// # Formula
///
/// ```text
/// d(x,y) = (2/sqrt(c)) * arctanh(sqrt(c * ||x-y||^2 / ((1-c*||x||^2)(1-c*||y||^2))))
/// ```
///
/// # Arguments
///
/// * `x` - First point (64-element array)
/// * `y` - Second point (64-element array)
/// * `curvature` - Poincare ball curvature (must be negative)
///
/// # Returns
///
/// Hyperbolic distance (always >= 0).
///
/// # Example
///
/// ```
/// use context_graph_cuda::poincare::poincare_distance_cpu;
///
/// let x = [0.0f32; 64];
/// let mut y = [0.0f32; 64];
/// y[0] = 0.5;
///
/// let dist = poincare_distance_cpu(&x, &y, -1.0);
/// assert!(dist > 0.0);
///
/// // Distance to self is zero
/// let dist_self = poincare_distance_cpu(&x, &x, -1.0);
/// assert!(dist_self < 1e-6);
/// ```
pub fn poincare_distance_cpu(x: &[f32; 64], y: &[f32; 64], curvature: f32) -> f32 {
    let c = curvature.abs();
    let sqrt_c = c.sqrt();

    // Compute norms
    let x_norm_sq: f32 = x.iter().map(|v| v * v).sum();
    let y_norm_sq: f32 = y.iter().map(|v| v * v).sum();

    // Compute ||x - y||^2
    let diff_norm_sq: f32 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - yi) * (xi - yi))
        .sum();

    // Handle identical points
    if diff_norm_sq < POINCARE_EPS {
        return 0.0;
    }

    // Denominators with numerical stability
    let denom_x = (1.0 - c * x_norm_sq).max(POINCARE_EPS);
    let denom_y = (1.0 - c * y_norm_sq).max(POINCARE_EPS);

    // arctanh argument
    let arg_sq = c * diff_norm_sq / (denom_x * denom_y);
    let arg = arg_sq.max(0.0).sqrt().min(1.0 - POINCARE_EPS);

    // Poincare distance
    (2.0 / sqrt_c) * arg.atanh()
}

/// Compute batch Poincare distances on CPU.
///
/// Reference implementation for testing GPU kernel correctness.
///
/// # Arguments
///
/// * `queries` - Query vectors, flattened \[n_queries * 64\]
/// * `database` - Database vectors, flattened \[n_database * 64\]
/// * `n_queries` - Number of query points
/// * `n_database` - Number of database points
/// * `curvature` - Poincare ball curvature (must be negative)
///
/// # Returns
///
/// Distance matrix, flattened \[n_queries * n_database\], row-major.
pub fn poincare_distance_batch_cpu(
    queries: &[f32],
    database: &[f32],
    n_queries: usize,
    n_database: usize,
    curvature: f32,
) -> Vec<f32> {
    assert_eq!(queries.len(), n_queries * 64, "Invalid query array size");
    assert_eq!(
        database.len(),
        n_database * 64,
        "Invalid database array size"
    );

    let mut distances = vec![0.0f32; n_queries * n_database];

    for i in 0..n_queries {
        let q_start = i * 64;
        let q: &[f32; 64] = queries[q_start..q_start + 64].try_into().unwrap();

        for j in 0..n_database {
            let db_start = j * 64;
            let db: &[f32; 64] = database[db_start..db_start + 64].try_into().unwrap();

            distances[i * n_database + j] = poincare_distance_cpu(q, db, curvature);
        }
    }

    distances
}
