# TASK-UTL-P1-008: Integrate ClusterFit into CoherenceTracker

## Metadata

| Field | Value |
|-------|-------|
| **ID** | TASK-UTL-P1-008 |
| **Title** | Integrate ClusterFit into CoherenceTracker |
| **Status** | blocked |
| **Layer** | surface (Layer 3) |
| **Sequence** | 3 of 3 |
| **Implements** | REQ-UTL-002-05, REQ-UTL-002-06, REQ-UTL-002-07, REQ-UTL-002-08 |
| **Depends On** | TASK-UTL-P1-002, TASK-UTL-P1-007 |
| **Estimated Complexity** | medium |
| **Estimated Duration** | 3-4 hours |
| **Spec Reference** | SPEC-UTL-002 |
| **Gap Reference** | GAP 2 from MASTER-CONSCIOUSNESS-GAP-ANALYSIS.md |

---

## Context

This is the **surface integration task** (Layer 3) for ClusterFit. It wires the ClusterFitCalculator into the existing CoherenceTracker to complete the three-component coherence formula.

**Gap Being Addressed:**
> PRD requires: `ΔC = Σwₖ × (EdgeAlign + SubGraphDensity + ClusterFit)`
> Currently only EdgeAlign and SubGraphDensity are implemented.
> This task adds ClusterFit to complete the formula.

After this task, the coherence calculation will use:
```
ΔC = 0.4 × Connectivity + 0.4 × ClusterFit + 0.2 × Consistency
```

---

## Input Context Files

| File | Purpose | Read Before |
|------|---------|-------------|
| `crates/context-graph-utl/src/coherence/cluster_fit.rs` | ClusterFitCalculator from TASK-UTL-P1-007 | Required |
| `crates/context-graph-utl/src/coherence/tracker.rs` | Existing CoherenceTracker to modify | Required |
| `crates/context-graph-utl/src/coherence/structural.rs` | StructuralCoherenceCalculator reference | Recommended |
| `crates/context-graph-utl/src/config/coherence.rs` | CoherenceConfig with weights | Required |
| `docs2/constitution.yaml` | Formula: α=0.4, β=0.4, γ=0.2 | Required |
| `specs/functional/SPEC-UTL-002.md` | Full specification | Required |

---

## Prerequisites

- [ ] **TASK-UTL-P1-002 completed** (types exist)
- [ ] **TASK-UTL-P1-007 completed** (ClusterFitCalculator implemented)
- [ ] ClusterFitCalculator passes all unit tests
- [ ] CoherenceTracker exists and compiles

---

## Scope

### In Scope

- Add `ClusterFitCalculator` as a field in `CoherenceTracker`
- Modify `compute_coherence()` to use three-component formula
- Add method to retrieve cluster context for a vertex
- Wire ClusterFitCalculator into coherence computation pipeline
- Update tests to verify three-component formula
- Handle ClusterFit failures with fallback and logging

### Out of Scope

- Cluster discovery/assignment (external service)
- Performance optimization (GPU acceleration)
- MCP tool integration (separate task)

---

## Definition of Done

### Exact Signatures Required

```rust
// File: crates/context-graph-utl/src/coherence/tracker.rs (modifications)

use super::cluster_fit::{ClusterFitCalculator, ClusterContext, ClusterFitResult};

/// Tracks coherence (Delta-C) for vertices using three-component formula.
///
/// # Constitution Reference
/// > delta_sc.ΔC: "α×Connectivity + β×ClusterFit + γ×Consistency (0.4, 0.4, 0.2)"
///
/// # Components
/// - Connectivity (α=0.4): From StructuralCoherenceCalculator (edge alignment)
/// - ClusterFit (β=0.4): From ClusterFitCalculator (silhouette coefficient)
/// - Consistency (γ=0.2): From rolling window variance
pub struct CoherenceTracker {
    // ... existing fields ...

    /// Calculator for cluster fit (silhouette-based).
    cluster_fit_calculator: ClusterFitCalculator,

    /// Weight for connectivity component (α). Default: 0.4
    connectivity_weight: f32,

    /// Weight for cluster fit component (β). Default: 0.4
    cluster_fit_weight: f32,

    /// Weight for consistency component (γ). Default: 0.2
    consistency_weight: f32,
}

impl CoherenceTracker {
    /// Create a new CoherenceTracker with configuration.
    ///
    /// # Arguments
    /// * `config` - Coherence configuration including weights
    ///
    /// # Example
    /// ```
    /// let config = CoherenceConfig::default();
    /// let tracker = CoherenceTracker::new(config);
    /// ```
    pub fn new(config: CoherenceConfig) -> Self;

    /// Compute coherence (Delta-C) for a vertex.
    ///
    /// Uses the three-component formula:
    /// `ΔC = α×Connectivity + β×ClusterFit + γ×Consistency`
    ///
    /// # Arguments
    /// * `vertex` - The vertex embedding to evaluate
    /// * `connectivity` - Pre-computed connectivity score from StructuralCoherenceCalculator
    /// * `cluster_context` - Cluster context for ClusterFit computation
    ///
    /// # Returns
    /// Coherence score in [0, 1]
    ///
    /// # Edge Cases
    /// - If ClusterFit fails, uses fallback (0.5) and logs warning
    /// - If connectivity is NaN, uses fallback (0.5)
    ///
    /// `Constraint: < 5ms p95 total`
    pub fn compute_coherence(
        &mut self,
        vertex: &[f32],
        connectivity: f32,
        cluster_context: &ClusterContext,
    ) -> f32;

    /// Compute coherence with all components computed internally.
    ///
    /// Convenience method that computes all three components.
    ///
    /// # Arguments
    /// * `vertex` - The vertex embedding
    /// * `graph_context` - Graph context for connectivity computation
    /// * `cluster_context` - Cluster context for ClusterFit
    ///
    /// # Returns
    /// Coherence score in [0, 1] with diagnostics
    pub fn compute_coherence_full(
        &mut self,
        vertex: &[f32],
        graph_context: &GraphContext,
        cluster_context: &ClusterContext,
    ) -> CoherenceResult;

    /// Get the cluster fit calculator for direct access.
    pub fn cluster_fit_calculator(&self) -> &ClusterFitCalculator;

    /// Update the component weights.
    ///
    /// # Arguments
    /// * `alpha` - Connectivity weight
    /// * `beta` - ClusterFit weight
    /// * `gamma` - Consistency weight
    ///
    /// # Panics
    /// Panics if weights do not sum to approximately 1.0 (tolerance: 0.01)
    pub fn set_weights(&mut self, alpha: f32, beta: f32, gamma: f32);
}

/// Result of coherence computation with diagnostics.
#[derive(Debug, Clone)]
pub struct CoherenceResult {
    /// Final coherence score [0, 1]
    pub score: f32,

    /// Connectivity component value [0, 1]
    pub connectivity: f32,

    /// ClusterFit component value [0, 1]
    pub cluster_fit: f32,

    /// Consistency component value [0, 1]
    pub consistency: f32,

    /// ClusterFit detailed result (if available)
    pub cluster_fit_result: Option<ClusterFitResult>,

    /// Whether fallback was used for any component
    pub used_fallback: bool,
}
```

### Updated compute_coherence Implementation

```rust
impl CoherenceTracker {
    pub fn compute_coherence(
        &mut self,
        vertex: &[f32],
        connectivity: f32,
        cluster_context: &ClusterContext,
    ) -> f32 {
        // 1. Compute ClusterFit
        let cluster_fit = match self.cluster_fit_calculator.compute(vertex, cluster_context) {
            Ok(result) => result.score,
            Err(e) => {
                log::warn!("ClusterFit computation failed: {}, using fallback", e);
                self.cluster_fit_calculator.config().fallback_value
            }
        };

        // 2. Get consistency from rolling window
        let consistency = self.compute_consistency();

        // 3. Validate inputs (AP-10: no NaN/Inf)
        let connectivity = if connectivity.is_nan() || connectivity.is_infinite() {
            log::warn!("Connectivity is NaN/Inf, using fallback");
            0.5
        } else {
            connectivity.clamp(0.0, 1.0)
        };

        // 4. Apply three-component formula
        // ΔC = α×Connectivity + β×ClusterFit + γ×Consistency
        let coherence = self.connectivity_weight * connectivity
            + self.cluster_fit_weight * cluster_fit
            + self.consistency_weight * consistency;

        // 5. Clamp and return
        coherence.clamp(0.0, 1.0)
    }
}
```

### Constraints (from constitution.yaml)

- Default weights MUST be α=0.4, β=0.4, γ=0.2
- Weights MUST sum to 1.0 (tolerance: 0.01)
- **AP-10**: No NaN/Infinity in UTL calculations
- ClusterFit failure MUST use fallback, not propagate error
- Total latency MUST be < 5ms p95
- All component values MUST be in [0, 1]

### Verification Commands

```bash
# Compile check
cargo check -p context-graph-utl

# Run all coherence tests
cargo test -p context-graph-utl --lib -- coherence --nocapture

# Run integration tests
cargo test -p context-graph-utl --lib -- coherence::tests::integration --nocapture

# Lint check
cargo clippy -p context-graph-utl -- -D warnings

# Benchmark
cargo bench -p context-graph-utl -- coherence
```

---

## Pseudo-code

```rust
impl CoherenceTracker {
    pub fn new(config: CoherenceConfig) -> Self {
        Self {
            // ... existing fields ...
            cluster_fit_calculator: ClusterFitCalculator::new(config.cluster_fit.clone()),
            connectivity_weight: config.connectivity_weight, // 0.4
            cluster_fit_weight: config.cluster_fit_weight,   // 0.4
            consistency_weight: config.consistency_weight,   // 0.2
        }
    }

    pub fn compute_coherence_full(
        &mut self,
        vertex: &[f32],
        graph_context: &GraphContext,
        cluster_context: &ClusterContext,
    ) -> CoherenceResult {
        let mut used_fallback = false;

        // 1. Compute connectivity from structural calculator
        let connectivity = self.structural_calculator
            .compute_connectivity(vertex, graph_context)
            .unwrap_or_else(|e| {
                log::warn!("Connectivity failed: {}", e);
                used_fallback = true;
                0.5
            });

        // 2. Compute cluster fit
        let cluster_fit_result = self.cluster_fit_calculator
            .compute(vertex, cluster_context);

        let (cluster_fit, cf_result) = match cluster_fit_result {
            Ok(result) => (result.score, Some(result)),
            Err(e) => {
                log::warn!("ClusterFit failed: {}", e);
                used_fallback = true;
                (0.5, None)
            }
        };

        // 3. Compute consistency
        let consistency = self.compute_consistency();

        // 4. Apply formula: ΔC = α×Connectivity + β×ClusterFit + γ×Consistency
        let score = (self.connectivity_weight * connectivity
            + self.cluster_fit_weight * cluster_fit
            + self.consistency_weight * consistency)
            .clamp(0.0, 1.0);

        // 5. Update rolling window for future consistency calculations
        self.update_history(score);

        CoherenceResult {
            score,
            connectivity,
            cluster_fit,
            consistency,
            cluster_fit_result: cf_result,
            used_fallback,
        }
    }

    pub fn set_weights(&mut self, alpha: f32, beta: f32, gamma: f32) {
        let sum = alpha + beta + gamma;
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Weights must sum to 1.0, got {}",
            sum
        );

        self.connectivity_weight = alpha;
        self.cluster_fit_weight = beta;
        self.consistency_weight = gamma;
    }
}
```

---

## Files to Create

None - all modifications to existing files.

---

## Files to Modify

| Path | Modification |
|------|--------------|
| `crates/context-graph-utl/src/coherence/tracker.rs` | Add ClusterFitCalculator field, modify compute_coherence |
| `crates/context-graph-utl/src/coherence/mod.rs` | Ensure cluster_fit module is exported |

---

## Validation Criteria

| Criterion | Verification Method |
|-----------|---------------------|
| Three-component formula used | Unit test with known values |
| Default weights are 0.4, 0.4, 0.2 | Unit test assertion |
| ClusterFit failure uses fallback | Unit test with error injection |
| Weights sum to 1.0 | set_weights assertion |
| Total coherence in [0, 1] | Property test |
| Latency < 5ms p95 | Benchmark test |
| No clippy warnings | `cargo clippy` clean |
| Backward compatible | Existing tests still pass |

---

## Required Tests

Implement in `tracker.rs` tests module:

| Test Name | Description |
|-----------|-------------|
| `test_coherence_three_component_formula` | Verify α×C + β×CF + γ×Cons formula |
| `test_coherence_default_weights` | Default weights are 0.4, 0.4, 0.2 |
| `test_coherence_custom_weights` | set_weights applies correctly |
| `test_coherence_weights_sum_assertion` | Weights must sum to 1.0 |
| `test_coherence_cluster_fit_fallback` | ClusterFit error uses fallback |
| `test_coherence_connectivity_nan_fallback` | NaN connectivity uses fallback |
| `test_coherence_all_zeros` | All components 0 returns 0 |
| `test_coherence_all_ones` | All components 1 returns 1 |
| `test_coherence_result_components` | CoherenceResult has all components |
| `test_coherence_integration` | End-to-end with real data |

### Integration Test Example

```rust
#[test]
fn test_coherence_integration() {
    let config = CoherenceConfig::default();
    let mut tracker = CoherenceTracker::new(config);

    // Create test cluster context
    let cluster_context = ClusterContext {
        same_cluster: vec![
            vec![0.5, 0.5, 0.5],
            vec![0.6, 0.6, 0.6],
        ],
        nearest_cluster: vec![
            vec![0.9, 0.9, 0.9],
            vec![0.95, 0.95, 0.95],
        ],
        centroids: None,
    };

    let vertex = vec![0.55, 0.55, 0.55];
    let connectivity = 0.8;

    let coherence = tracker.compute_coherence(&vertex, connectivity, &cluster_context);

    // Verify result is in valid range
    assert!(coherence >= 0.0 && coherence <= 1.0);

    // Verify approximate expected value
    // With vertex well-clustered, expect high cluster_fit (~0.9)
    // connectivity = 0.8
    // consistency starts at 0.5 (neutral)
    // Expected: 0.4*0.8 + 0.4*0.9 + 0.2*0.5 = 0.32 + 0.36 + 0.1 = 0.78
    assert!((coherence - 0.78).abs() < 0.15, "Coherence {} not near expected 0.78", coherence);
}
```

---

## Notes

- This completes the ClusterFit integration into the coherence pipeline
- The three-component formula matches constitution.yaml exactly
- Fallback behavior ensures robustness without propagating errors
- CoherenceResult provides transparency for debugging and monitoring
- After this task, the UTL learning score `L = f((ΔS × ΔC) · wₑ · cos φ)` will use complete ΔC

---

## Related Tasks

- **TASK-UTL-P1-002**: Types and configuration (prerequisite)
- **TASK-UTL-P1-007**: Silhouette calculation (prerequisite)
- **TASK-MCP-xxx**: MCP tool integration (future, uses CoherenceTracker)
