# TASK-UTL-P1-007: Implement Silhouette Calculation and Distance Methods

## Metadata

| Field | Value |
|-------|-------|
| **ID** | TASK-UTL-P1-007 |
| **Title** | Implement Silhouette Calculation and Distance Methods |
| **Status** | blocked |
| **Layer** | logic (Layer 2) |
| **Sequence** | 2 of 3 |
| **Implements** | REQ-UTL-002-01, REQ-UTL-002-02, REQ-UTL-002-03, REQ-UTL-002-04, REQ-UTL-002-05 |
| **Depends On** | TASK-UTL-P1-002 |
| **Estimated Complexity** | medium |
| **Estimated Duration** | 4-6 hours |
| **Spec Reference** | SPEC-UTL-002 |
| **Gap Reference** | GAP 2 from MASTER-CONSCIOUSNESS-GAP-ANALYSIS.md |

---

## Context

This task implements the **core algorithm** (Layer 2) for ClusterFit: the silhouette coefficient calculation and its supporting distance methods. This is the computational heart of the ClusterFit component.

**Gap Being Addressed:**
> Resolution from gap analysis: "Add to `computeCoherence()`:
> ```
> private computeClusterFit(vertex: Vertex): number {
>   const silhouette = this.computeSilhouetteScore(vertex);
>   return (silhouette + 1) / 2; // Normalize to [0,1]
> }
> ```"

The silhouette coefficient measures how similar a point is to its own cluster compared to other clusters:
- `s(i) = (b(i) - a(i)) / max(a(i), b(i))`
- Where `a(i)` is mean intra-cluster distance and `b(i)` is mean nearest-cluster distance

---

## Input Context Files

| File | Purpose | Read Before |
|------|---------|-------------|
| `crates/context-graph-utl/src/coherence/cluster_fit.rs` | Types from TASK-UTL-P1-002 | Required |
| `crates/context-graph-utl/src/coherence/tracker.rs` | Reference for cosine_similarity function | Required |
| `crates/context-graph-utl/src/coherence/structural.rs` | Reference for structural coherence patterns | Recommended |
| `crates/context-graph-utl/src/config/coherence.rs` | Configuration structures | Required |
| `crates/context-graph-utl/src/error.rs` | Error types | Required |
| `specs/functional/SPEC-UTL-002.md` | Full specification | Required |

---

## Prerequisites

- [ ] **TASK-UTL-P1-002 completed** (types exist and compile)
- [ ] ClusterFitConfig, DistanceMetric, ClusterContext, ClusterFitResult types exist
- [ ] UtlError::ClusterFitError variant exists

---

## Scope

### In Scope

- Implement `ClusterFitCalculator::new(config)` constructor
- Implement `ClusterFitCalculator::compute(vertex, context)` main entry point
- Implement `compute_distance(a, b, metric) -> f32` for all three metrics:
  - Cosine distance (1 - cosine_similarity)
  - Euclidean (L2) distance
  - Manhattan (L1) distance
- Implement `compute_intra_cluster_distance(vertex, same_cluster) -> f32`
- Implement `compute_inter_cluster_distance(vertex, nearest_cluster) -> f32`
- Implement `compute_silhouette(a, b) -> f32` raw silhouette [-1, 1]
- Implement `normalize_silhouette(s) -> f32` to [0, 1]
- Handle edge cases: empty clusters, single member, NaN values
- Unit tests with known sklearn reference values

### Out of Scope

- Integration with CoherenceTracker (TASK-UTL-P1-008)
- Cluster assignment/discovery logic (external dependency)
- GPU acceleration (future optimization)

---

## Definition of Done

### Exact Signatures Required

```rust
// File: crates/context-graph-utl/src/coherence/cluster_fit.rs

/// Calculator for cluster fit using silhouette coefficient.
///
/// The silhouette coefficient measures how similar a point is to its own
/// cluster compared to other clusters. A high value indicates the point
/// is well-matched to its cluster.
///
/// # Algorithm
///
/// 1. Compute mean intra-cluster distance (a): average distance to same-cluster members
/// 2. Compute mean inter-cluster distance (b): average distance to nearest other cluster
/// 3. Compute silhouette: s = (b - a) / max(a, b)
/// 4. Normalize to [0, 1]: score = (s + 1) / 2
///
/// # Performance
///
/// `Constraint: < 2ms p95 per vertex`
#[derive(Debug, Clone)]
pub struct ClusterFitCalculator {
    config: ClusterFitConfig,
}

impl ClusterFitCalculator {
    /// Create a new ClusterFitCalculator with the given configuration.
    ///
    /// # Arguments
    /// * `config` - Configuration for cluster fit calculation
    ///
    /// # Example
    /// ```
    /// let config = ClusterFitConfig::default();
    /// let calculator = ClusterFitCalculator::new(config);
    /// ```
    pub fn new(config: ClusterFitConfig) -> Self {
        Self { config }
    }

    /// Compute cluster fit score for a vertex given cluster context.
    ///
    /// # Arguments
    /// * `vertex` - The embedding of the vertex to evaluate
    /// * `context` - Cluster context containing same-cluster and nearest-cluster embeddings
    ///
    /// # Returns
    /// `ClusterFitResult` with normalized score [0, 1] and diagnostics
    ///
    /// # Errors
    /// Returns `UtlError::ClusterFitError` if computation fails due to invalid input
    ///
    /// # Edge Cases
    /// - Empty same_cluster: returns fallback (0.5)
    /// - Empty nearest_cluster: returns fallback (0.5)
    /// - Single member cluster: returns fallback (0.5)
    /// - NaN/Inf values: returns fallback with warning log
    ///
    /// `Constraint: < 2ms p95`
    pub fn compute(
        &self,
        vertex: &[f32],
        context: &ClusterContext,
    ) -> UtlResult<ClusterFitResult>;

    /// Compute distance between two vectors using the configured metric.
    ///
    /// # Arguments
    /// * `a` - First vector
    /// * `b` - Second vector
    ///
    /// # Returns
    /// Distance value (always non-negative)
    ///
    /// # Panics
    /// Panics if vectors have different dimensions (debug builds only)
    pub fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32;
}

impl Default for ClusterFitCalculator {
    fn default() -> Self {
        Self::new(ClusterFitConfig::default())
    }
}

// ============================================================================
// Private helper functions (internal implementation)
// ============================================================================

/// Compute mean intra-cluster distance (a) for a vertex.
///
/// This is the average distance from the vertex to all other members
/// of the same cluster.
///
/// # Arguments
/// * `vertex` - The embedding of the vertex
/// * `same_cluster` - Embeddings of other vertices in the same cluster
/// * `metric` - Distance metric to use
///
/// # Returns
/// Mean intra-cluster distance, or 0.0 if cluster is empty
fn compute_intra_cluster_distance(
    vertex: &[f32],
    same_cluster: &[Vec<f32>],
    metric: DistanceMetric,
) -> f32;

/// Compute mean nearest-cluster distance (b) for a vertex.
///
/// This is the average distance from the vertex to all members
/// of the nearest other cluster.
///
/// # Arguments
/// * `vertex` - The embedding of the vertex
/// * `nearest_cluster` - Embeddings of vertices in the nearest other cluster
/// * `metric` - Distance metric to use
///
/// # Returns
/// Mean inter-cluster distance, or f32::MAX if cluster is empty
fn compute_inter_cluster_distance(
    vertex: &[f32],
    nearest_cluster: &[Vec<f32>],
    metric: DistanceMetric,
) -> f32;

/// Compute raw silhouette coefficient.
///
/// Formula: s = (b - a) / max(a, b)
///
/// # Arguments
/// * `intra_distance` - Mean intra-cluster distance (a)
/// * `inter_distance` - Mean nearest-cluster distance (b)
///
/// # Returns
/// Silhouette coefficient in range [-1, 1]
///
/// # Edge Cases
/// - If max(a, b) == 0: returns 0.0 (all points identical)
fn compute_silhouette(intra_distance: f32, inter_distance: f32) -> f32;

/// Normalize silhouette from [-1, 1] to [0, 1].
///
/// Formula: normalized = (silhouette + 1.0) / 2.0
///
/// This maps:
/// - silhouette = -1.0 (misclassified) -> 0.0
/// - silhouette = 0.0 (borderline) -> 0.5
/// - silhouette = 1.0 (perfect fit) -> 1.0
fn normalize_silhouette(silhouette: f32) -> f32;

/// Compute cosine distance between two vectors.
///
/// Cosine distance = 1 - cosine_similarity
///
/// # Note
/// Returns 1.0 (max distance) if either vector has zero magnitude.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32;

/// Compute Euclidean (L2) distance between two vectors.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32;

/// Compute Manhattan (L1) distance between two vectors.
fn manhattan_distance(a: &[f32], b: &[f32]) -> f32;

/// Sample cluster members if exceeding max_sample_size for performance.
fn sample_if_needed(cluster: &[Vec<f32>], max_size: usize) -> Vec<Vec<f32>>;
```

### Constraints (from constitution.yaml)

- **AP-10**: No NaN/Infinity in UTL calculations - MUST check and fallback
- **AP-09**: No unbounded operations - sample large clusters
- **ARCH-02**: Apples-to-apples comparison - same embedder type only
- Silhouette coefficient MUST be in range [-1, 1]
- Normalized score MUST be in range [0, 1]
- Empty same_cluster: return fallback (0.5)
- Empty nearest_cluster: return fallback (0.5)
- Single member cluster: return fallback (0.5)
- Match sklearn silhouette_score within 0.001 tolerance

### Verification Commands

```bash
# Compile check
cargo check -p context-graph-utl

# Run tests
cargo test -p context-graph-utl --lib -- cluster_fit --nocapture

# Lint check
cargo clippy -p context-graph-utl -- -D warnings

# Doc check
cargo doc -p context-graph-utl --no-deps
```

---

## Pseudo-code

```rust
impl ClusterFitCalculator {
    pub fn compute(&self, vertex: &[f32], context: &ClusterContext) -> UtlResult<ClusterFitResult> {
        // 1. Edge case: insufficient same-cluster data
        if context.same_cluster.is_empty() ||
           context.same_cluster.len() < self.config.min_cluster_size - 1 {
            return Ok(ClusterFitResult::fallback(self.config.fallback_value));
        }

        // 2. Edge case: no other cluster to compare
        if context.nearest_cluster.is_empty() {
            return Ok(ClusterFitResult::fallback(self.config.fallback_value));
        }

        // 3. Sample if cluster too large (performance constraint)
        let same_cluster = sample_if_needed(
            &context.same_cluster,
            self.config.max_sample_size
        );
        let nearest_cluster = sample_if_needed(
            &context.nearest_cluster,
            self.config.max_sample_size
        );

        // 4. Compute distances
        let a = compute_intra_cluster_distance(
            vertex,
            &same_cluster,
            self.config.distance_metric
        );
        let b = compute_inter_cluster_distance(
            vertex,
            &nearest_cluster,
            self.config.distance_metric
        );

        // 5. Compute silhouette: s = (b - a) / max(a, b)
        let silhouette = compute_silhouette(a, b);

        // 6. Check for NaN/Inf (AP-10 compliance)
        if silhouette.is_nan() || silhouette.is_infinite() {
            log::warn!("ClusterFit: NaN/Inf detected, using fallback");
            return Ok(ClusterFitResult {
                score: self.config.fallback_value,
                silhouette: 0.0,
                intra_distance: a,
                inter_distance: b,
            });
        }

        // 7. Normalize to [0, 1]
        let score = normalize_silhouette(silhouette);

        Ok(ClusterFitResult {
            score,
            silhouette,
            intra_distance: a,
            inter_distance: b,
        })
    }
}

fn compute_silhouette(a: f32, b: f32) -> f32 {
    let max_dist = a.max(b);
    if max_dist == 0.0 {
        return 0.0; // All points identical
    }
    (b - a) / max_dist
}

fn normalize_silhouette(s: f32) -> f32 {
    ((s + 1.0) / 2.0).clamp(0.0, 1.0)
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 1.0; // Max distance for zero vectors
    }

    let similarity = dot / (mag_a * mag_b);
    1.0 - similarity.clamp(-1.0, 1.0)
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum()
}
```

---

## Files to Create

None - all code goes in existing file from TASK-UTL-P1-002.

---

## Files to Modify

| Path | Modification |
|------|--------------|
| `crates/context-graph-utl/src/coherence/cluster_fit.rs` | Add `ClusterFitCalculator` impl and all helper functions |

---

## Validation Criteria

| Criterion | Verification Method |
|-----------|---------------------|
| Silhouette matches sklearn within 0.001 | Unit test with known reference values |
| Perfect fit (a=0, b>0) returns score=1.0 | Unit test assertion |
| Misclassified (a>b) returns score<0.5 | Unit test assertion |
| Empty cluster returns fallback=0.5 | Unit test assertion |
| NaN/Inf triggers fallback | Unit test with edge case inputs |
| All three distance metrics work correctly | Unit tests per metric |
| Latency < 2ms p95 | Benchmark test |
| No clippy warnings | `cargo clippy` clean |

---

## Required Tests

Implement in `cluster_fit.rs` tests module:

| Test Name | Description |
|-----------|-------------|
| `test_silhouette_matches_sklearn_well_separated` | Compare to sklearn for well-separated clusters |
| `test_silhouette_matches_sklearn_overlapping` | Compare to sklearn for overlapping clusters |
| `test_perfect_cluster_fit` | Vertex at cluster centroid returns ~1.0 |
| `test_misclassified_point` | Vertex closer to other cluster returns <0.5 |
| `test_empty_same_cluster_fallback` | Empty same_cluster returns 0.5 |
| `test_empty_nearest_cluster_fallback` | Empty nearest_cluster returns 0.5 |
| `test_single_member_fallback` | Single member cluster returns 0.5 |
| `test_nan_handling` | NaN in input triggers fallback |
| `test_inf_handling` | Inf in input triggers fallback |
| `test_cosine_distance` | Cosine distance computation |
| `test_euclidean_distance` | Euclidean distance computation |
| `test_manhattan_distance` | Manhattan distance computation |
| `test_sampling_large_cluster` | Large clusters are sampled |
| `test_dimension_consistency` | Dimension mismatch handled |

---

## Reference Values (sklearn validation)

```python
# Test case 1: Well-separated clusters
from sklearn.metrics import silhouette_samples
import numpy as np

# Cluster A points
cluster_a = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0.5]])
# Cluster B points
cluster_b = np.array([[10, 10], [11, 10], [10, 11]])

X = np.vstack([cluster_a, cluster_b])
labels = [0, 0, 0, 0, 1, 1, 1]

# Point [0.5, 0.5] (index 3) silhouette
s = silhouette_samples(X, labels)[3]
# Expected: ~0.85 (well-clustered)

# Test case 2: Borderline point
cluster_a = np.array([[0, 0], [1, 1]])
cluster_b = np.array([[2, 2], [3, 3]])
point = np.array([[1.5, 1.5]])  # Borderline

X = np.vstack([cluster_a, point, cluster_b])
labels = [0, 0, 0, 1, 1]

s = silhouette_samples(X, labels)[2]
# Expected: ~0.0 (borderline)

# Test case 3: Misclassified point
cluster_a = np.array([[0, 0], [1, 1]])
cluster_b = np.array([[5, 5], [6, 6], [7, 7]])
wrong_point = np.array([[5.5, 5.5]])  # Should be in B but labeled A

X = np.vstack([cluster_a, wrong_point, cluster_b])
labels = [0, 0, 0, 1, 1, 1]

s = silhouette_samples(X, labels)[2]
# Expected: negative (misclassified)
```

---

## Notes

- The silhouette coefficient is a well-established clustering metric from sklearn
- Our implementation must match standard definitions for correctness
- Performance optimization (SIMD, GPU) is a future task if needed
- The distance metric is configurable for flexibility across embedding types
- Sampling large clusters is essential to maintain latency budget

---

## Related Tasks

- **TASK-UTL-P1-002**: Types and configuration (prerequisite)
- **TASK-UTL-P1-008**: Integration into CoherenceTracker (next task)
