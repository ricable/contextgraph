//! ClusterFit types for silhouette-based coherence component.
//!
//! # Constitution Reference
//!
//! Per constitution.yaml line 166:
//! ΔC = 0.4×Connectivity + 0.4×ClusterFit + 0.2×Consistency
//!
//! ClusterFit measures how well a vertex fits within its semantic cluster
//! using the silhouette coefficient: s = (b - a) / max(a, b)
//!
//! # Output Range
//!
//! All outputs are clamped per AP-10 (no NaN/Infinity):
//! - `ClusterFitResult.score`: [0, 1]
//! - `ClusterFitResult.silhouette`: [-1, 1]

use serde::{Deserialize, Serialize};

/// Configuration for ClusterFit calculation.
///
/// # Constitution Reference
/// - Line 166: ΔC = 0.4×Connectivity + 0.4×ClusterFit + 0.2×Consistency
/// - ClusterFit uses silhouette score: (b - a) / max(a, b)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterFitConfig {
    /// Minimum cluster size for valid calculation.
    /// Default: 2 (silhouette requires at least 2 members)
    pub min_cluster_size: usize,

    /// Distance metric to use.
    /// Default: Cosine (matches semantic embedding space)
    pub distance_metric: DistanceMetric,

    /// Fallback value when cluster fit cannot be computed.
    /// Default: 0.5 (neutral - per AP-10 no NaN allowed)
    pub fallback_value: f32,

    /// Maximum cluster members to sample for performance.
    /// Default: 1000 (prevents O(n²) explosion)
    pub max_sample_size: usize,
}

impl Default for ClusterFitConfig {
    fn default() -> Self {
        Self {
            min_cluster_size: 2,
            distance_metric: DistanceMetric::default(),
            fallback_value: 0.5,
            max_sample_size: 1000,
        }
    }
}

/// Distance metric options for cluster distance calculation.
///
/// Used to compute intra-cluster distance (a) and inter-cluster distance (b)
/// for silhouette coefficient: s = (b - a) / max(a, b)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine distance: 1 - cosine_similarity.
    /// Best for normalized embeddings (most common).
    #[default]
    Cosine,
    /// Euclidean (L2) distance.
    /// Use for non-normalized embeddings.
    Euclidean,
    /// Manhattan (L1) distance.
    /// More robust to outliers.
    Manhattan,
}

/// Cluster context providing embeddings for cluster fit calculation.
///
/// Contains the data needed to compute silhouette coefficient for a vertex.
#[derive(Debug, Clone)]
pub struct ClusterContext {
    /// Embeddings of vertices in the same cluster (excluding the query vertex).
    /// Must have at least `min_cluster_size - 1` members for valid computation.
    pub same_cluster: Vec<Vec<f32>>,

    /// Embeddings of vertices in the nearest other cluster.
    /// Used to compute inter-cluster distance (b).
    pub nearest_cluster: Vec<Vec<f32>>,

    /// Optional precomputed cluster centroids for efficiency.
    /// Index corresponds to cluster ID.
    pub centroids: Option<Vec<Vec<f32>>>,
}

impl ClusterContext {
    /// Create new cluster context.
    pub fn new(same_cluster: Vec<Vec<f32>>, nearest_cluster: Vec<Vec<f32>>) -> Self {
        Self {
            same_cluster,
            nearest_cluster,
            centroids: None,
        }
    }

    /// Create with precomputed centroids.
    pub fn with_centroids(
        same_cluster: Vec<Vec<f32>>,
        nearest_cluster: Vec<Vec<f32>>,
        centroids: Vec<Vec<f32>>,
    ) -> Self {
        Self {
            same_cluster,
            nearest_cluster,
            centroids: Some(centroids),
        }
    }
}

/// Result of ClusterFit calculation with diagnostics.
///
/// # Output Range
/// - `score`: [0, 1] normalized for UTL formula
/// - `silhouette`: [-1, 1] raw coefficient
#[derive(Debug, Clone)]
pub struct ClusterFitResult {
    /// Normalized cluster fit score [0, 1].
    /// Derived from silhouette: (silhouette + 1) / 2
    pub score: f32,

    /// Raw silhouette coefficient [-1, 1].
    /// -1 = wrong cluster, 0 = boundary, +1 = well-clustered
    pub silhouette: f32,

    /// Mean intra-cluster distance (a).
    /// Average distance to same-cluster members.
    pub intra_distance: f32,

    /// Mean nearest-cluster distance (b).
    /// Average distance to nearest other cluster.
    pub inter_distance: f32,
}

impl ClusterFitResult {
    /// Create result from raw silhouette and distances.
    ///
    /// Automatically computes normalized score.
    pub fn new(silhouette: f32, intra_distance: f32, inter_distance: f32) -> Self {
        // Normalize silhouette from [-1, 1] to [0, 1]
        let score = (silhouette + 1.0) / 2.0;
        Self {
            score: score.clamp(0.0, 1.0),
            silhouette: silhouette.clamp(-1.0, 1.0),
            intra_distance,
            inter_distance,
        }
    }

    /// Create a fallback result when computation is not possible.
    pub fn fallback(value: f32) -> Self {
        Self {
            score: value.clamp(0.0, 1.0),
            silhouette: 0.0, // Neutral
            intra_distance: 0.0,
            inter_distance: 0.0,
        }
    }
}

// ============================================================================
// Distance Computation Functions
// ============================================================================

/// Compute vector magnitude (L2 norm).
fn magnitude(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Compute cosine distance between two vectors.
///
/// Cosine distance = 1 - cosine_similarity
/// Range: [0, 2] but typically [0, 1] for normalized vectors.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Cosine distance clamped to [0, 2]. Returns 0.0 for edge cases.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a = magnitude(a);
    let mag_b = magnitude(b);

    // Handle zero magnitude vectors gracefully
    if mag_a < 1e-10 || mag_b < 1e-10 {
        return 0.0;
    }

    let cosine_sim = dot / (mag_a * mag_b);

    // Cosine distance = 1 - similarity
    // Clamp similarity first to handle floating point errors
    let clamped_sim = cosine_sim.clamp(-1.0, 1.0);
    let distance = 1.0 - clamped_sim;

    // Handle potential NaN/Infinity per AP-10
    if distance.is_nan() || distance.is_infinite() {
        0.0
    } else {
        distance.clamp(0.0, 2.0)
    }
}

/// Compute Euclidean (L2) distance between two vectors.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Euclidean distance (non-negative). Returns 0.0 for edge cases.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let sum_sq: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();

    let distance = sum_sq.sqrt();

    // Handle potential NaN/Infinity per AP-10
    if distance.is_nan() || distance.is_infinite() {
        0.0
    } else {
        distance
    }
}

/// Compute Manhattan (L1) distance between two vectors.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Manhattan distance (non-negative). Returns 0.0 for edge cases.
fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let distance: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();

    // Handle potential NaN/Infinity per AP-10
    if distance.is_nan() || distance.is_infinite() {
        0.0
    } else {
        distance
    }
}

/// Compute distance between two vectors using the specified metric.
fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::Euclidean => euclidean_distance(a, b),
        DistanceMetric::Manhattan => manhattan_distance(a, b),
    }
}

/// Compute mean distance from a query vector to a set of cluster members.
///
/// # Arguments
/// * `query` - The query embedding vector
/// * `cluster` - The cluster member embeddings
/// * `metric` - Distance metric to use
/// * `max_sample` - Maximum number of members to sample
///
/// # Returns
/// Mean distance, or None if cluster is empty or has no valid members.
fn mean_distance_to_cluster(
    query: &[f32],
    cluster: &[Vec<f32>],
    metric: DistanceMetric,
    max_sample: usize,
) -> Option<f32> {
    if cluster.is_empty() || query.is_empty() {
        return None;
    }

    // Sample if cluster is too large
    let members: Vec<&Vec<f32>> = if cluster.len() > max_sample {
        // Simple deterministic sampling: take evenly spaced members
        let step = cluster.len() / max_sample;
        cluster.iter().step_by(step.max(1)).take(max_sample).collect()
    } else {
        cluster.iter().collect()
    };

    let mut sum = 0.0f64;
    let mut count = 0usize;

    for member in members {
        // Skip members with mismatched dimensions
        if member.len() != query.len() {
            continue;
        }
        // Skip zero-magnitude members for cosine distance
        if metric == DistanceMetric::Cosine && magnitude(member) < 1e-10 {
            continue;
        }

        let dist = compute_distance(query, member, metric);
        sum += dist as f64;
        count += 1;
    }

    if count == 0 {
        None
    } else {
        let mean = (sum / count as f64) as f32;
        // Ensure no NaN/Infinity per AP-10
        if mean.is_nan() || mean.is_infinite() {
            None
        } else {
            Some(mean)
        }
    }
}

// ============================================================================
// Main Computation Function
// ============================================================================

/// Compute silhouette coefficient for a query embedding.
///
/// The silhouette coefficient measures how well a point fits within its assigned
/// cluster compared to the nearest other cluster:
///
/// ```text
/// silhouette = (b - a) / max(a, b)
/// ```
///
/// Where:
/// - `a` = mean intra-cluster distance (distance to same-cluster members)
/// - `b` = mean nearest-cluster distance (distance to nearest other cluster)
///
/// # Constitution Reference
///
/// Per constitution.yaml line 166:
/// - ΔC = 0.4×Connectivity + 0.4×ClusterFit + 0.2×Consistency
/// - Output range [0, 1] (normalized from silhouette [-1, 1])
/// - No NaN/Infinity (AP-10)
///
/// # Arguments
///
/// * `query` - The embedding vector to evaluate
/// * `context` - Cluster context containing same-cluster and nearest-cluster members
/// * `config` - Configuration for the calculation
///
/// # Returns
///
/// `ClusterFitResult` containing:
/// - `score`: Normalized score [0, 1] for use in UTL formula
/// - `silhouette`: Raw silhouette coefficient [-1, 1]
/// - `intra_distance`: Mean distance to same-cluster members (a)
/// - `inter_distance`: Mean distance to nearest-cluster members (b)
///
/// # Edge Cases
///
/// Returns fallback result when:
/// - Query is empty or zero-magnitude
/// - Same-cluster has fewer than `min_cluster_size - 1` valid members
/// - Nearest-cluster is empty
/// - All distances are zero or invalid
///
/// # Example
///
/// ```ignore
/// use context_graph_utl::coherence::{
///     compute_cluster_fit, ClusterContext, ClusterFitConfig, DistanceMetric,
/// };
///
/// let query = vec![0.1, 0.2, 0.3, 0.4];
/// let same_cluster = vec![
///     vec![0.12, 0.22, 0.28, 0.38],
///     vec![0.11, 0.21, 0.29, 0.39],
/// ];
/// let nearest_cluster = vec![
///     vec![0.8, 0.1, 0.05, 0.05],
///     vec![0.7, 0.2, 0.05, 0.05],
/// ];
///
/// let context = ClusterContext::new(same_cluster, nearest_cluster);
/// let config = ClusterFitConfig::default();
///
/// let result = compute_cluster_fit(&query, &context, &config);
/// assert!(result.score >= 0.0 && result.score <= 1.0);
/// assert!(result.silhouette >= -1.0 && result.silhouette <= 1.0);
/// ```
pub fn compute_cluster_fit(
    query: &[f32],
    context: &ClusterContext,
    config: &ClusterFitConfig,
) -> ClusterFitResult {
    // Edge case: empty query
    if query.is_empty() {
        return ClusterFitResult::fallback(config.fallback_value);
    }

    // Edge case: zero-magnitude query (for cosine distance)
    if config.distance_metric == DistanceMetric::Cosine && magnitude(query) < 1e-10 {
        return ClusterFitResult::fallback(config.fallback_value);
    }

    // Check minimum cluster size requirement
    // We need at least min_cluster_size - 1 other members (query is one member)
    let min_required = config.min_cluster_size.saturating_sub(1);
    if context.same_cluster.len() < min_required {
        return ClusterFitResult::fallback(config.fallback_value);
    }

    // Edge case: empty nearest cluster
    if context.nearest_cluster.is_empty() {
        return ClusterFitResult::fallback(config.fallback_value);
    }

    // Compute intra-cluster distance (a)
    let intra_distance = match mean_distance_to_cluster(
        query,
        &context.same_cluster,
        config.distance_metric,
        config.max_sample_size,
    ) {
        Some(d) => d,
        None => return ClusterFitResult::fallback(config.fallback_value),
    };

    // Compute inter-cluster distance (b) - distance to nearest cluster
    let inter_distance = match mean_distance_to_cluster(
        query,
        &context.nearest_cluster,
        config.distance_metric,
        config.max_sample_size,
    ) {
        Some(d) => d,
        None => return ClusterFitResult::fallback(config.fallback_value),
    };

    // Compute silhouette coefficient: s = (b - a) / max(a, b)
    let max_dist = intra_distance.max(inter_distance);

    let silhouette = if max_dist < 1e-10 {
        // Both distances are effectively zero - neutral result
        0.0
    } else {
        let s = (inter_distance - intra_distance) / max_dist;
        // Ensure no NaN/Infinity per AP-10
        if s.is_nan() || s.is_infinite() {
            0.0
        } else {
            s.clamp(-1.0, 1.0)
        }
    };

    ClusterFitResult::new(silhouette, intra_distance, inter_distance)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Configuration and Type Tests
    // =========================================================================

    #[test]
    fn test_cluster_fit_config_default() {
        let config = ClusterFitConfig::default();
        assert_eq!(config.min_cluster_size, 2);
        assert_eq!(config.distance_metric, DistanceMetric::Cosine);
        assert_eq!(config.fallback_value, 0.5);
        assert_eq!(config.max_sample_size, 1000);
    }

    #[test]
    fn test_distance_metric_default() {
        assert_eq!(DistanceMetric::default(), DistanceMetric::Cosine);
    }

    #[test]
    fn test_cluster_context_new() {
        let same = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let nearest = vec![vec![0.5, 0.6]];
        let ctx = ClusterContext::new(same.clone(), nearest.clone());

        assert_eq!(ctx.same_cluster.len(), 2);
        assert_eq!(ctx.nearest_cluster.len(), 1);
        assert!(ctx.centroids.is_none());
    }

    #[test]
    fn test_cluster_context_with_centroids() {
        let same = vec![vec![0.1, 0.2]];
        let nearest = vec![vec![0.5, 0.6]];
        let centroids = vec![vec![0.2, 0.3], vec![0.6, 0.7]];

        let ctx = ClusterContext::with_centroids(same, nearest, centroids);
        assert!(ctx.centroids.is_some());
        assert_eq!(ctx.centroids.unwrap().len(), 2);
    }

    #[test]
    fn test_cluster_fit_result_new() {
        // Perfect clustering: silhouette = 1.0
        let result = ClusterFitResult::new(1.0, 0.1, 0.9);
        assert_eq!(result.score, 1.0);
        assert_eq!(result.silhouette, 1.0);

        // Worst clustering: silhouette = -1.0
        let result = ClusterFitResult::new(-1.0, 0.9, 0.1);
        assert_eq!(result.score, 0.0);
        assert_eq!(result.silhouette, -1.0);

        // Boundary: silhouette = 0.0
        let result = ClusterFitResult::new(0.0, 0.5, 0.5);
        assert_eq!(result.score, 0.5);
        assert_eq!(result.silhouette, 0.0);
    }

    #[test]
    fn test_cluster_fit_result_clamps_output() {
        // Test clamping for out-of-range values
        let result = ClusterFitResult::new(1.5, 0.1, 0.9);
        assert_eq!(result.score, 1.0);
        assert_eq!(result.silhouette, 1.0);

        let result = ClusterFitResult::new(-1.5, 0.9, 0.1);
        assert_eq!(result.score, 0.0);
        assert_eq!(result.silhouette, -1.0);
    }

    #[test]
    fn test_cluster_fit_result_fallback() {
        let result = ClusterFitResult::fallback(0.5);
        assert_eq!(result.score, 0.5);
        assert_eq!(result.silhouette, 0.0);
        assert_eq!(result.intra_distance, 0.0);
        assert_eq!(result.inter_distance, 0.0);
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = ClusterFitConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let restored: ClusterFitConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.min_cluster_size, restored.min_cluster_size);
        assert_eq!(config.distance_metric, restored.distance_metric);
        assert_eq!(config.fallback_value, restored.fallback_value);
    }

    #[test]
    fn test_distance_metric_serialization() {
        let metric = DistanceMetric::Euclidean;
        let json = serde_json::to_string(&metric).unwrap();
        assert!(json.contains("Euclidean"));

        let restored: DistanceMetric = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, DistanceMetric::Euclidean);
    }

    // =========================================================================
    // Distance Function Tests
    // =========================================================================

    #[test]
    fn test_magnitude() {
        // Unit vector
        let v = vec![1.0, 0.0, 0.0];
        assert!((magnitude(&v) - 1.0).abs() < 1e-6);

        // Zero vector
        let zero = vec![0.0, 0.0, 0.0];
        assert!(magnitude(&zero).abs() < 1e-10);

        // 3-4-5 triangle
        let v345 = vec![3.0, 4.0];
        assert!((magnitude(&v345) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let dist = cosine_distance(&a, &a);
        assert!(dist.abs() < 1e-6, "Identical vectors should have distance ~0");
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let dist = cosine_distance(&a, &b);
        assert!(
            (dist - 1.0).abs() < 1e-6,
            "Orthogonal vectors should have distance ~1"
        );
    }

    #[test]
    fn test_cosine_distance_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let dist = cosine_distance(&a, &b);
        assert!(
            (dist - 2.0).abs() < 1e-6,
            "Opposite vectors should have distance ~2"
        );
    }

    #[test]
    fn test_cosine_distance_empty() {
        let empty: Vec<f32> = vec![];
        let a = vec![1.0, 2.0];
        assert_eq!(cosine_distance(&empty, &a), 0.0);
        assert_eq!(cosine_distance(&a, &empty), 0.0);
    }

    #[test]
    fn test_cosine_distance_zero_vector() {
        let zero = vec![0.0, 0.0, 0.0];
        let a = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_distance(&zero, &a), 0.0);
        assert_eq!(cosine_distance(&a, &zero), 0.0);
    }

    #[test]
    fn test_cosine_distance_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_distance(&a, &b), 0.0);
    }

    #[test]
    fn test_euclidean_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let dist = euclidean_distance(&a, &a);
        assert!(dist.abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance_basic() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance_empty() {
        let empty: Vec<f32> = vec![];
        let a = vec![1.0, 2.0];
        assert_eq!(euclidean_distance(&empty, &a), 0.0);
    }

    #[test]
    fn test_manhattan_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let dist = manhattan_distance(&a, &a);
        assert!(dist.abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance_basic() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = manhattan_distance(&a, &b);
        assert!((dist - 7.0).abs() < 1e-6); // |3-0| + |4-0| = 7
    }

    #[test]
    fn test_manhattan_distance_empty() {
        let empty: Vec<f32> = vec![];
        let a = vec![1.0, 2.0];
        assert_eq!(manhattan_distance(&empty, &a), 0.0);
    }

    #[test]
    fn test_compute_distance_metric_dispatch() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        let cosine = compute_distance(&a, &b, DistanceMetric::Cosine);
        let euclidean = compute_distance(&a, &b, DistanceMetric::Euclidean);
        let manhattan = compute_distance(&a, &b, DistanceMetric::Manhattan);

        // Different metrics should give different results
        assert!((euclidean - 5.0).abs() < 1e-6);
        assert!((manhattan - 7.0).abs() < 1e-6);
        // Cosine returns 0 for zero vector
        assert_eq!(cosine, 0.0);
    }

    // =========================================================================
    // Mean Distance Tests
    // =========================================================================

    #[test]
    fn test_mean_distance_empty_cluster() {
        let query = vec![1.0, 0.0, 0.0];
        let cluster: Vec<Vec<f32>> = vec![];

        let result = mean_distance_to_cluster(&query, &cluster, DistanceMetric::Cosine, 1000);
        assert!(result.is_none());
    }

    #[test]
    fn test_mean_distance_empty_query() {
        let query: Vec<f32> = vec![];
        let cluster = vec![vec![1.0, 0.0, 0.0]];

        let result = mean_distance_to_cluster(&query, &cluster, DistanceMetric::Cosine, 1000);
        assert!(result.is_none());
    }

    #[test]
    fn test_mean_distance_single_member() {
        let query = vec![1.0, 0.0, 0.0];
        let cluster = vec![vec![0.0, 1.0, 0.0]]; // Orthogonal

        let result = mean_distance_to_cluster(&query, &cluster, DistanceMetric::Cosine, 1000);
        assert!(result.is_some());
        let dist = result.unwrap();
        assert!((dist - 1.0).abs() < 1e-6); // Orthogonal = distance 1
    }

    #[test]
    fn test_mean_distance_multiple_members() {
        let query = vec![1.0, 0.0, 0.0];
        let cluster = vec![
            vec![1.0, 0.0, 0.0], // Identical = 0
            vec![0.0, 1.0, 0.0], // Orthogonal = 1
        ];

        let result = mean_distance_to_cluster(&query, &cluster, DistanceMetric::Cosine, 1000);
        assert!(result.is_some());
        let dist = result.unwrap();
        assert!((dist - 0.5).abs() < 1e-6); // Mean of 0 and 1 = 0.5
    }

    #[test]
    fn test_mean_distance_skips_mismatched_dimensions() {
        let query = vec![1.0, 0.0, 0.0];
        let cluster = vec![
            vec![1.0, 0.0], // Wrong dimension - skipped
            vec![0.0, 1.0, 0.0], // Orthogonal = 1
        ];

        let result = mean_distance_to_cluster(&query, &cluster, DistanceMetric::Cosine, 1000);
        assert!(result.is_some());
        let dist = result.unwrap();
        // Only the orthogonal vector counts
        assert!((dist - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_distance_sampling() {
        let query = vec![1.0, 0.0];

        // Create a large cluster
        let mut cluster = Vec::new();
        for i in 0..100 {
            let angle = (i as f32) * std::f32::consts::PI / 100.0;
            cluster.push(vec![angle.cos(), angle.sin()]);
        }

        // With max_sample=10, should still work
        let result = mean_distance_to_cluster(&query, &cluster, DistanceMetric::Cosine, 10);
        assert!(result.is_some());
        let dist = result.unwrap();
        assert!(dist >= 0.0);
    }

    // =========================================================================
    // compute_cluster_fit Tests
    // =========================================================================

    #[test]
    fn test_compute_cluster_fit_basic() {
        let query = vec![0.1, 0.2, 0.3, 0.4];

        // Same cluster: similar vectors
        let same_cluster = vec![
            vec![0.12, 0.22, 0.28, 0.38],
            vec![0.11, 0.21, 0.29, 0.39],
        ];

        // Nearest cluster: quite different vectors
        let nearest_cluster = vec![
            vec![0.8, 0.1, 0.05, 0.05],
            vec![0.7, 0.2, 0.05, 0.05],
        ];

        let context = ClusterContext::new(same_cluster, nearest_cluster);
        let config = ClusterFitConfig::default();

        let result = compute_cluster_fit(&query, &context, &config);

        // Should have positive silhouette (well-clustered)
        assert!(result.silhouette > 0.0, "Expected positive silhouette");
        assert!(result.score > 0.5, "Expected score > 0.5 for well-clustered");

        // Verify output ranges per AP-10
        assert!(
            (0.0..=1.0).contains(&result.score),
            "Score should be in [0, 1]"
        );
        assert!(
            (-1.0..=1.0).contains(&result.silhouette),
            "Silhouette should be in [-1, 1]"
        );
        assert!(result.intra_distance >= 0.0, "Intra distance >= 0");
        assert!(result.inter_distance >= 0.0, "Inter distance >= 0");
    }

    #[test]
    fn test_compute_cluster_fit_wrong_cluster() {
        let query = vec![0.9, 0.1, 0.0, 0.0];

        // Same cluster: vectors very different from query
        let same_cluster = vec![
            vec![0.0, 0.0, 0.9, 0.1],
            vec![0.0, 0.0, 0.8, 0.2],
        ];

        // Nearest cluster: vectors similar to query
        let nearest_cluster = vec![
            vec![0.85, 0.15, 0.0, 0.0],
            vec![0.88, 0.12, 0.0, 0.0],
        ];

        let context = ClusterContext::new(same_cluster, nearest_cluster);
        let config = ClusterFitConfig::default();

        let result = compute_cluster_fit(&query, &context, &config);

        // Should have negative silhouette (wrong cluster)
        assert!(
            result.silhouette < 0.0,
            "Expected negative silhouette for wrong cluster"
        );
        assert!(result.score < 0.5, "Expected score < 0.5 for wrong cluster");
    }

    #[test]
    fn test_compute_cluster_fit_boundary() {
        let query = vec![0.5, 0.5, 0.0, 0.0];

        // Same cluster and nearest cluster equally distant
        let same_cluster = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];

        let nearest_cluster = vec![vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]];

        let context = ClusterContext::new(same_cluster, nearest_cluster);
        let config = ClusterFitConfig::default();

        let result = compute_cluster_fit(&query, &context, &config);

        // Should have silhouette close to 0 (boundary case)
        // The exact value depends on the specific vectors
        assert!(
            (-1.0..=1.0).contains(&result.silhouette),
            "Silhouette should be valid"
        );
        assert!(
            (0.0..=1.0).contains(&result.score),
            "Score should be in [0, 1]"
        );
    }

    #[test]
    fn test_compute_cluster_fit_empty_query() {
        let query: Vec<f32> = vec![];
        let context = ClusterContext::new(vec![vec![1.0, 0.0]], vec![vec![0.0, 1.0]]);
        let config = ClusterFitConfig::default();

        let result = compute_cluster_fit(&query, &context, &config);

        // Should return fallback
        assert_eq!(result.score, config.fallback_value);
        assert_eq!(result.silhouette, 0.0);
    }

    #[test]
    fn test_compute_cluster_fit_zero_magnitude_query() {
        let query = vec![0.0, 0.0, 0.0];
        let context = ClusterContext::new(vec![vec![1.0, 0.0, 0.0]], vec![vec![0.0, 1.0, 0.0]]);
        let config = ClusterFitConfig::default(); // Uses cosine distance

        let result = compute_cluster_fit(&query, &context, &config);

        // Should return fallback for zero-magnitude with cosine
        assert_eq!(result.score, config.fallback_value);
    }

    #[test]
    fn test_compute_cluster_fit_empty_same_cluster() {
        let query = vec![1.0, 0.0, 0.0];
        let context = ClusterContext::new(vec![], vec![vec![0.0, 1.0, 0.0]]);
        let config = ClusterFitConfig::default();

        let result = compute_cluster_fit(&query, &context, &config);

        // Should return fallback (need min_cluster_size - 1 members)
        assert_eq!(result.score, config.fallback_value);
    }

    #[test]
    fn test_compute_cluster_fit_empty_nearest_cluster() {
        let query = vec![1.0, 0.0, 0.0];
        let context = ClusterContext::new(vec![vec![0.9, 0.1, 0.0]], vec![]);
        let config = ClusterFitConfig::default();

        let result = compute_cluster_fit(&query, &context, &config);

        // Should return fallback
        assert_eq!(result.score, config.fallback_value);
    }

    #[test]
    fn test_compute_cluster_fit_insufficient_same_cluster() {
        let query = vec![1.0, 0.0, 0.0];
        let context = ClusterContext::new(vec![], vec![vec![0.0, 1.0, 0.0]]);

        // Require 3 members (so need 2 in same_cluster)
        let mut config = ClusterFitConfig::default();
        config.min_cluster_size = 3;

        let result = compute_cluster_fit(&query, &context, &config);

        // Should return fallback
        assert_eq!(result.score, config.fallback_value);
    }

    #[test]
    fn test_compute_cluster_fit_euclidean_metric() {
        let query = vec![0.0, 0.0];

        let same_cluster = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let nearest_cluster = vec![vec![3.0, 0.0], vec![0.0, 3.0]];

        let context = ClusterContext::new(same_cluster, nearest_cluster);
        let mut config = ClusterFitConfig::default();
        config.distance_metric = DistanceMetric::Euclidean;

        let result = compute_cluster_fit(&query, &context, &config);

        // Intra distance: mean of dist([0,0], [1,0]) and dist([0,0], [0,1]) = 1.0
        // Inter distance: mean of dist([0,0], [3,0]) and dist([0,0], [0,3]) = 3.0
        // Silhouette = (3.0 - 1.0) / 3.0 = 2/3 ~ 0.667
        assert!(result.silhouette > 0.6, "Expected high positive silhouette");
        assert!((result.intra_distance - 1.0).abs() < 1e-6);
        assert!((result.inter_distance - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_cluster_fit_manhattan_metric() {
        let query = vec![0.0, 0.0];

        let same_cluster = vec![vec![1.0, 1.0]]; // Manhattan dist = 2

        let nearest_cluster = vec![vec![3.0, 3.0]]; // Manhattan dist = 6

        let context = ClusterContext::new(same_cluster, nearest_cluster);
        let mut config = ClusterFitConfig::default();
        config.distance_metric = DistanceMetric::Manhattan;

        let result = compute_cluster_fit(&query, &context, &config);

        // Silhouette = (6 - 2) / 6 = 4/6 ~ 0.667
        assert!(result.silhouette > 0.6);
        assert!((result.intra_distance - 2.0).abs() < 1e-6);
        assert!((result.inter_distance - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_cluster_fit_custom_fallback() {
        let query: Vec<f32> = vec![];
        let context = ClusterContext::new(vec![], vec![]);

        let mut config = ClusterFitConfig::default();
        config.fallback_value = 0.75;

        let result = compute_cluster_fit(&query, &context, &config);

        assert_eq!(result.score, 0.75);
    }

    #[test]
    fn test_compute_cluster_fit_no_nan_infinity() {
        // Test various edge cases that might produce NaN/Infinity

        let test_cases: Vec<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> = vec![
            // Zero vectors everywhere
            (
                vec![0.0, 0.0, 0.0],
                vec![vec![0.0, 0.0, 0.0]],
                vec![vec![0.0, 0.0, 0.0]],
            ),
            // Very small values
            (
                vec![1e-15, 1e-15, 1e-15],
                vec![vec![1e-15, 1e-15, 1e-15]],
                vec![vec![1e-15, 1e-15, 1e-15]],
            ),
            // Very large values
            (
                vec![1e30, 1e30, 1e30],
                vec![vec![1e30, 1e30, 1e30]],
                vec![vec![1e30, 1e30, 1e30]],
            ),
            // Mixed extreme values
            (
                vec![1e-30, 1e30],
                vec![vec![1e-30, 1e30]],
                vec![vec![1e30, 1e-30]],
            ),
        ];

        let config = ClusterFitConfig::default();

        for (query, same, nearest) in test_cases {
            let context = ClusterContext::new(same, nearest);
            let result = compute_cluster_fit(&query, &context, &config);

            assert!(!result.score.is_nan(), "Score should not be NaN");
            assert!(!result.score.is_infinite(), "Score should not be infinite");
            assert!(!result.silhouette.is_nan(), "Silhouette should not be NaN");
            assert!(
                !result.silhouette.is_infinite(),
                "Silhouette should not be infinite"
            );
            assert!(
                !result.intra_distance.is_nan(),
                "Intra distance should not be NaN"
            );
            assert!(
                !result.inter_distance.is_nan(),
                "Inter distance should not be NaN"
            );
        }
    }

    #[test]
    fn test_compute_cluster_fit_output_ranges() {
        // Comprehensive test of output ranges per AP-10

        let query = vec![0.5, 0.5, 0.0, 0.0];
        let same_cluster = vec![
            vec![0.6, 0.4, 0.0, 0.0],
            vec![0.4, 0.6, 0.0, 0.0],
            vec![0.55, 0.45, 0.0, 0.0],
        ];
        let nearest_cluster = vec![
            vec![0.0, 0.0, 0.5, 0.5],
            vec![0.0, 0.0, 0.6, 0.4],
        ];

        let context = ClusterContext::new(same_cluster, nearest_cluster);

        // Test all metrics
        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::Manhattan,
        ] {
            let mut config = ClusterFitConfig::default();
            config.distance_metric = metric;

            let result = compute_cluster_fit(&query, &context, &config);

            assert!(
                (0.0..=1.0).contains(&result.score),
                "{:?}: Score {} out of [0, 1]",
                metric,
                result.score
            );
            assert!(
                (-1.0..=1.0).contains(&result.silhouette),
                "{:?}: Silhouette {} out of [-1, 1]",
                metric,
                result.silhouette
            );
            assert!(
                result.intra_distance >= 0.0,
                "{:?}: Intra distance {} < 0",
                metric,
                result.intra_distance
            );
            assert!(
                result.inter_distance >= 0.0,
                "{:?}: Inter distance {} < 0",
                metric,
                result.inter_distance
            );
        }
    }

    #[test]
    fn test_compute_cluster_fit_perfect_clustering() {
        // Query is identical to same-cluster members, very far from nearest-cluster
        let query = vec![1.0, 0.0, 0.0];
        let same_cluster = vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
        let nearest_cluster = vec![vec![-1.0, 0.0, 0.0]];

        let context = ClusterContext::new(same_cluster, nearest_cluster);
        let config = ClusterFitConfig::default();

        let result = compute_cluster_fit(&query, &context, &config);

        // Should have maximum silhouette (close to 1)
        // Intra distance = 0, Inter distance = 2 (opposite vectors)
        // But silhouette formula gives 1 when a=0: (b - 0) / b = 1
        assert!(
            result.silhouette > 0.9,
            "Expected silhouette close to 1, got {}",
            result.silhouette
        );
        assert!(result.score > 0.9);
    }

    #[test]
    fn test_compute_cluster_fit_identical_clusters() {
        // Both clusters have identical vectors to query
        let query = vec![0.5, 0.5, 0.0, 0.0];
        let same_cluster = vec![vec![0.5, 0.5, 0.0, 0.0]];
        let nearest_cluster = vec![vec![0.5, 0.5, 0.0, 0.0]];

        let context = ClusterContext::new(same_cluster, nearest_cluster);
        let config = ClusterFitConfig::default();

        let result = compute_cluster_fit(&query, &context, &config);

        // Both distances are 0, so silhouette = 0 (neutral)
        assert!(
            result.silhouette.abs() < 1e-6,
            "Expected silhouette ~0, got {}",
            result.silhouette
        );
        assert!(
            (result.score - 0.5).abs() < 1e-6,
            "Expected score ~0.5, got {}",
            result.score
        );
    }

    #[test]
    fn test_compute_cluster_fit_sampling() {
        // Test that sampling doesn't break the calculation
        let query = vec![1.0, 0.0];

        // Create large clusters
        let mut same_cluster = Vec::new();
        let mut nearest_cluster = Vec::new();

        for i in 0..2000 {
            // Same cluster: close to [1, 0]
            let noise = (i as f32) * 0.0001;
            same_cluster.push(vec![1.0 - noise, noise]);

            // Nearest cluster: close to [0, 1]
            nearest_cluster.push(vec![noise, 1.0 - noise]);
        }

        let context = ClusterContext::new(same_cluster, nearest_cluster);
        let mut config = ClusterFitConfig::default();
        config.max_sample_size = 100; // Force sampling

        let result = compute_cluster_fit(&query, &context, &config);

        // Should still compute valid result
        assert!(
            (0.0..=1.0).contains(&result.score),
            "Score {} out of range",
            result.score
        );
        assert!(result.silhouette > 0.0, "Should be well-clustered");
    }
}
