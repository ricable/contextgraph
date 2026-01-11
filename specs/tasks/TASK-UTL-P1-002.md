# TASK-UTL-P1-002: Create ClusterFit Types and Cluster Interface

## Metadata

| Field | Value |
|-------|-------|
| **ID** | TASK-UTL-P1-002 |
| **Title** | Create ClusterFit Types and Cluster Interface |
| **Status** | ready |
| **Layer** | foundation (Layer 1) |
| **Sequence** | 1 of 3 |
| **Implements** | REQ-UTL-002-01, REQ-UTL-002-08 |
| **Depends On** | None |
| **Estimated Complexity** | low |
| **Estimated Duration** | 2-3 hours |
| **Spec Reference** | SPEC-UTL-002 |
| **Gap Reference** | GAP 2 from MASTER-CONSCIOUSNESS-GAP-ANALYSIS.md |

---

## Context

This is the **foundation task** (Layer 1) for ClusterFit implementation. It establishes the core types, configuration structures, and distance metric interfaces that all subsequent ClusterFit tasks depend upon.

**Gap Being Addressed:**
> "ΔC = Σwₖ × (EdgeAlign + SubGraphDensity + ClusterFit)" - PRD
> Currently only EdgeAlign and SubGraphDensity implemented. **Missing**: ClusterFit component with silhouette score.

This task creates NO business logic - only type definitions, configuration structures, and trait definitions. All algorithmic implementations come in subsequent tasks (TASK-UTL-P1-006, TASK-UTL-P1-007).

---

## Input Context Files

| File | Purpose | Read Before |
|------|---------|-------------|
| `crates/context-graph-utl/src/config/coherence.rs` | Existing coherence config to extend | Required |
| `crates/context-graph-utl/src/error.rs` | Existing error types to extend | Required |
| `crates/context-graph-utl/src/coherence/mod.rs` | Module structure to update | Required |
| `docs2/constitution.yaml` | Authoritative weights: α=0.4, β=0.4, γ=0.2 | Required |
| `specs/functional/SPEC-UTL-002.md` | Full specification | Reference |

---

## Prerequisites

- [x] context-graph-utl crate exists and compiles
- [x] Coherence module structure exists at `crates/context-graph-utl/src/coherence/`
- [ ] Spec SPEC-UTL-002 approved

---

## Scope

### In Scope

- Create `ClusterFitConfig` struct with configuration options
- Create `DistanceMetric` enum (Cosine, Euclidean, Manhattan)
- Create `ClusterContext` struct for cluster data
- Create `ClusterFitResult` struct for computation results
- Create `ClusterFitError` variants in error module
- Update `CoherenceConfig` to include ClusterFit weights
- Create empty `cluster_fit.rs` module with type exports

### Out of Scope

- Distance computation implementation (TASK-UTL-P1-003)
- Silhouette calculation logic (TASK-UTL-P1-003)
- Integration with CoherenceTracker (TASK-UTL-P1-004)
- Any business logic

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-utl/src/coherence/cluster_fit.rs

/// Configuration for ClusterFit calculation
#[derive(Debug, Clone)]
pub struct ClusterFitConfig {
    /// Minimum cluster size for valid calculation (default: 2)
    pub min_cluster_size: usize,

    /// Distance metric to use (default: Cosine)
    pub distance_metric: DistanceMetric,

    /// Fallback value when cluster fit cannot be computed (default: 0.5)
    pub fallback_value: f32,

    /// Maximum cluster members to sample for performance (default: 1000)
    pub max_sample_size: usize,
}

/// Distance metric options for cluster distance calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    /// Cosine distance: 1 - cosine_similarity
    #[default]
    Cosine,
    /// Euclidean (L2) distance
    Euclidean,
    /// Manhattan (L1) distance
    Manhattan,
}

/// Cluster context providing embeddings for cluster fit calculation
#[derive(Debug, Clone)]
pub struct ClusterContext {
    /// Embeddings of vertices in the same cluster (excluding the query vertex)
    pub same_cluster: Vec<Vec<f32>>,

    /// Embeddings of vertices in the nearest other cluster
    pub nearest_cluster: Vec<Vec<f32>>,

    /// Optional precomputed cluster centroids for efficiency
    pub centroids: Option<Vec<Vec<f32>>>,
}

/// Result of ClusterFit calculation with diagnostics
#[derive(Debug, Clone)]
pub struct ClusterFitResult {
    /// Normalized cluster fit score [0, 1]
    pub score: f32,

    /// Raw silhouette coefficient [-1, 1]
    pub silhouette: f32,

    /// Mean intra-cluster distance (a)
    pub intra_distance: f32,

    /// Mean nearest-cluster distance (b)
    pub inter_distance: f32,
}
```

```rust
// File: crates/context-graph-utl/src/config/coherence.rs (modifications)

/// Updated CoherenceConfig with ClusterFit weights
#[derive(Debug, Clone)]
pub struct CoherenceConfig {
    // ... existing fields ...

    /// Weight for connectivity component (α, default: 0.4)
    pub connectivity_weight: f32,

    /// Weight for cluster fit component (β, default: 0.4)
    pub cluster_fit_weight: f32,

    /// Weight for consistency component (γ, default: 0.2)
    pub consistency_weight: f32,

    /// ClusterFit specific configuration
    pub cluster_fit: ClusterFitConfig,
}
```

```rust
// File: crates/context-graph-utl/src/error.rs (additions)

/// Error variants for ClusterFit computation
#[derive(Debug, Error)]
pub enum UtlError {
    // ... existing variants ...

    /// ClusterFit computation error
    #[error("ClusterFit error: {0}")]
    ClusterFitError(String),

    /// Insufficient cluster data
    #[error("Insufficient cluster data: need at least {required} members, got {actual}")]
    InsufficientClusterData { required: usize, actual: usize },
}
```

### Constraints

- All structs MUST derive `Debug, Clone`
- All configuration fields MUST have sensible defaults
- Weights MUST match constitution: α=0.4, β=0.4, γ=0.2
- NO `any` type anywhere
- NO business logic in this task
- Follow naming conventions from constitution.yaml

### Verification

```bash
cargo check -p context-graph-utl
cargo test -p context-graph-utl --lib -- cluster_fit::tests
cargo clippy -p context-graph-utl -- -D warnings
```

---

## Pseudo-code

```rust
// cluster_fit.rs - Types only, no implementation

mod cluster_fit {
    // 1. Define ClusterFitConfig with Default impl
    struct ClusterFitConfig { min_cluster_size, distance_metric, fallback_value, max_sample_size }
    impl Default: min_cluster_size=2, distance_metric=Cosine, fallback_value=0.5, max_sample_size=1000

    // 2. Define DistanceMetric enum
    enum DistanceMetric { Cosine, Euclidean, Manhattan }
    impl Default: Cosine

    // 3. Define ClusterContext - holds cluster embeddings
    struct ClusterContext { same_cluster: Vec<Vec<f32>>, nearest_cluster, centroids }

    // 4. Define ClusterFitResult - computation output
    struct ClusterFitResult { score, silhouette, intra_distance, inter_distance }

    // 5. Tests for type construction and defaults
    #[cfg(test)]
    mod tests {
        test default_config_matches_constitution()
        test cluster_context_creation()
        test cluster_fit_result_creation()
    }
}

// coherence.rs config update
// Add connectivity_weight=0.4, cluster_fit_weight=0.4, consistency_weight=0.2

// error.rs update
// Add ClusterFitError and InsufficientClusterData variants
```

---

## Files to Create

| Path | Description |
|------|-------------|
| `crates/context-graph-utl/src/coherence/cluster_fit.rs` | ClusterFit types and configuration |

---

## Files to Modify

| Path | Modification |
|------|--------------|
| `crates/context-graph-utl/src/coherence/mod.rs` | Add `pub mod cluster_fit;` export |
| `crates/context-graph-utl/src/config/coherence.rs` | Add weight fields and ClusterFitConfig |
| `crates/context-graph-utl/src/error.rs` | Add ClusterFitError variants |

---

## Validation Criteria

| Criterion | Verification |
|-----------|--------------|
| All types compile without errors | `cargo check` passes |
| Default weights match constitution (0.4, 0.4, 0.2) | Unit test assertion |
| No clippy warnings | `cargo clippy` clean |
| Types are exported in module | Import test compiles |
| ClusterFitConfig has sensible defaults | Default::default() test |

---

## Test Commands

```bash
# Type check
cargo check -p context-graph-utl

# Run unit tests
cargo test -p context-graph-utl --lib -- cluster_fit --nocapture

# Clippy lint
cargo clippy -p context-graph-utl -- -D warnings

# Verify exports
cargo doc -p context-graph-utl --no-deps
```

---

## Notes

- This is a **pure types** task - no algorithms or logic
- Subsequent tasks (TASK-UTL-P1-003, TASK-UTL-P1-004) depend on these types
- Follow Rust naming conventions from constitution.yaml
