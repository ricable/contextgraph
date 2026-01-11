# SPEC-UTL-002: ClusterFit Coherence Component

## Metadata

| Field | Value |
|-------|-------|
| **ID** | SPEC-UTL-002 |
| **Version** | 1.0 |
| **Status** | approved |
| **Owner** | ContextGraph Team |
| **Created** | 2026-01-11 |
| **Last Updated** | 2026-01-11 |
| **Related Specs** | SPEC-UTL-001, constitution.yaml |
| **Priority** | P1 (Substantial Gap) |

---

## 1. Overview

### 1.1 Purpose

This specification defines the ClusterFit component for coherence (Delta-C) calculation in the UTL (Unified Theory of Learning) system. ClusterFit measures how well a vertex fits within its semantic cluster, completing the three-component coherence formula required by the PRD.

### 1.2 Problem Statement

The current coherence calculation is **incomplete**. Per the PRD and constitution.yaml:

```
ΔC = α × Connectivity + β × ClusterFit + γ × Consistency (0.4, 0.4, 0.2)
```

**Current State**:
- Connectivity (EdgeAlign): IMPLEMENTED via `StructuralCoherenceCalculator`
- Consistency: IMPLEMENTED via `CoherenceTracker`
- **ClusterFit: MISSING**

Without ClusterFit, the Delta-C calculation is biased and incomplete, affecting:
- UTL learning scores: `L = f((ΔS × ΔC) · wₑ · cos φ)`
- Johari Window classification accuracy
- Memory consolidation decisions
- Teleological alignment measurements

### 1.3 Success Criteria

1. ClusterFit component returns values in [0, 1]
2. Silhouette score calculation follows standard algorithm
3. Integrates seamlessly with existing coherence pipeline
4. Latency budget: < 2ms per vertex calculation
5. All tests pass with > 90% coverage

---

## 2. User Stories

### US-UTL-002-01: Compute ClusterFit for Vertex
**Priority**: must-have

**Narrative**:
> As the UTL system,
> I want to compute ClusterFit for a vertex,
> So that I can accurately measure how well the vertex belongs to its semantic cluster.

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-01 | A vertex with embedding and cluster assignments | ClusterFit is computed | Returns silhouette coefficient in [-1, 1] normalized to [0, 1] |
| AC-02 | A vertex with no neighbors | ClusterFit is computed | Returns 0.5 (neutral) |
| AC-03 | A vertex identical to cluster centroid | ClusterFit is computed | Returns 1.0 (perfect fit) |
| AC-04 | A vertex at cluster boundary | ClusterFit is computed | Returns value near 0.5 |

### US-UTL-002-02: Integrate ClusterFit into Coherence
**Priority**: must-have

**Narrative**:
> As the coherence computation module,
> I want ClusterFit integrated into `compute_coherence()`,
> So that Delta-C uses the complete three-component formula.

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-05 | Coherence computation with all components | `compute_coherence()` called | Returns weighted sum: 0.4×Connectivity + 0.4×ClusterFit + 0.2×Consistency |
| AC-06 | ClusterFit component fails | `compute_coherence()` called | Uses fallback value 0.5, logs warning |
| AC-07 | Custom weights provided | `compute_coherence()` called | Uses provided weights (α, β, γ) |

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Story Ref | Priority | Rationale |
|----|-------------|-----------|----------|-----------|
| REQ-UTL-002-01 | System SHALL compute silhouette coefficient for a vertex given its embedding and cluster context | US-UTL-002-01 | must | Core algorithm for ClusterFit |
| REQ-UTL-002-02 | System SHALL compute mean intra-cluster distance (a) as average distance to same-cluster members | US-UTL-002-01 | must | Silhouette numerator component |
| REQ-UTL-002-03 | System SHALL compute nearest-cluster distance (b) as mean distance to nearest other cluster | US-UTL-002-01 | must | Silhouette denominator component |
| REQ-UTL-002-04 | System SHALL normalize silhouette from [-1, 1] to [0, 1] range | US-UTL-002-01 | must | Consistent with coherence range |
| REQ-UTL-002-05 | System SHALL expose `compute_cluster_fit(vertex, cluster_context) -> f32` | US-UTL-002-02 | must | API contract |
| REQ-UTL-002-06 | System SHALL update `compute_coherence()` to use formula: α×Connectivity + β×ClusterFit + γ×Consistency | US-UTL-002-02 | must | PRD compliance |
| REQ-UTL-002-07 | System SHALL use default weights α=0.4, β=0.4, γ=0.2 from constitution | US-UTL-002-02 | must | Constitution alignment |
| REQ-UTL-002-08 | System SHALL support configurable weights via CoherenceConfig | US-UTL-002-02 | should | Flexibility |

### 3.2 Non-Functional Requirements

| ID | Category | Requirement | Metric | Rationale |
|----|----------|-------------|--------|-----------|
| NFR-UTL-002-01 | Performance | ClusterFit computation SHALL complete in < 2ms p95 | Latency | Coherence budget is < 5ms total |
| NFR-UTL-002-02 | Accuracy | Silhouette implementation SHALL match sklearn reference within 0.001 | Precision | Correctness |
| NFR-UTL-002-03 | Robustness | System SHALL handle empty clusters gracefully | Error handling | Stability |
| NFR-UTL-002-04 | Testability | All public functions SHALL have unit tests | Coverage > 90% | Quality |

---

## 4. Technical Design

### 4.1 Data Structures

```rust
/// Configuration for ClusterFit calculation
#[derive(Debug, Clone)]
pub struct ClusterFitConfig {
    /// Minimum cluster size for valid calculation
    pub min_cluster_size: usize,

    /// Distance metric to use
    pub distance_metric: DistanceMetric,

    /// Fallback value when cluster fit cannot be computed
    pub fallback_value: f32,
}

/// Distance metric options
#[derive(Debug, Clone, Copy, Default)]
pub enum DistanceMetric {
    #[default]
    Cosine,
    Euclidean,
    Manhattan,
}

/// Cluster context for a vertex
#[derive(Debug, Clone)]
pub struct ClusterContext {
    /// Embeddings of vertices in the same cluster
    pub same_cluster: Vec<Vec<f32>>,

    /// Embeddings of vertices in nearest other cluster
    pub nearest_cluster: Vec<Vec<f32>>,

    /// Optional: precomputed cluster centroids for efficiency
    pub centroids: Option<Vec<Vec<f32>>>,
}

/// Result of ClusterFit calculation with diagnostics
#[derive(Debug, Clone)]
pub struct ClusterFitResult {
    /// Normalized cluster fit score [0, 1]
    pub score: f32,

    /// Raw silhouette coefficient [-1, 1]
    pub silhouette: f32,

    /// Mean intra-cluster distance
    pub intra_distance: f32,

    /// Mean nearest-cluster distance
    pub inter_distance: f32,
}
```

### 4.2 Algorithm: Silhouette Coefficient

The silhouette coefficient for a single sample is defined as:

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

Where:
- a(i) = mean distance from i to all other points in same cluster
- b(i) = mean distance from i to all points in nearest other cluster
```

**Normalization to [0, 1]**:
```
cluster_fit = (silhouette + 1.0) / 2.0
```

**Edge Cases**:
- Single-member cluster: silhouette = 0 → cluster_fit = 0.5
- No other clusters: silhouette = 0 → cluster_fit = 0.5
- Perfect fit (a=0, b>0): silhouette = 1 → cluster_fit = 1.0
- Misclassified (a>b): silhouette < 0 → cluster_fit < 0.5

### 4.3 Updated Coherence Formula

```rust
pub fn compute_coherence(
    &self,
    connectivity: f32,    // From StructuralCoherenceCalculator
    cluster_fit: f32,     // NEW: From ClusterFitCalculator
    consistency: f32,     // From CoherenceTracker
) -> f32 {
    let alpha = self.config.connectivity_weight;  // 0.4
    let beta = self.config.cluster_fit_weight;    // 0.4
    let gamma = self.config.consistency_weight;   // 0.2

    let coherence = alpha * connectivity + beta * cluster_fit + gamma * consistency;
    coherence.clamp(0.0, 1.0)
}
```

### 4.4 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     CoherenceTracker                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐  ┌─────────────────────┐              │
│  │ StructuralCoherence │  │  ClusterFitCalc     │ ◄── NEW      │
│  │    Calculator       │  │  (Silhouette)       │              │
│  └──────────┬──────────┘  └──────────┬──────────┘              │
│             │ connectivity           │ cluster_fit              │
│             │                        │                          │
│             ▼                        ▼                          │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                 compute_coherence()                        │ │
│  │  ΔC = α×Connectivity + β×ClusterFit + γ×Consistency       │ │
│  └────────────────────────────┬──────────────────────────────┘ │
│                               │                                 │
│                     ┌─────────┴─────────┐                       │
│                     │ RollingWindow     │ consistency           │
│                     │ (variance-based)  │                       │
│                     └───────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Edge Cases

| ID | Scenario | Expected Behavior |
|----|----------|-------------------|
| EC-01 | Vertex is only member of its cluster | Return 0.5 (neutral), cannot compute meaningful silhouette |
| EC-02 | Only one cluster exists in graph | Return 0.5 (neutral), no inter-cluster comparison possible |
| EC-03 | Cluster context has NaN/Inf values | Return fallback 0.5, log warning |
| EC-04 | Empty same_cluster vector | Return 0.5 (neutral) |
| EC-05 | Empty nearest_cluster vector | Return 0.5 (neutral) |
| EC-06 | Very large cluster (>10K members) | Sample cluster members to maintain latency budget |

---

## 6. Error States

| ID | Condition | Message | Recovery |
|----|-----------|---------|----------|
| ERR-01 | Dimension mismatch between embeddings | "ClusterFit: embedding dimension mismatch" | Return fallback, log error |
| ERR-02 | Invalid distance metric | "ClusterFit: unsupported distance metric" | Use default (cosine) |
| ERR-03 | Computation overflow | "ClusterFit: numerical overflow in distance" | Return fallback, log error |

---

## 7. Test Plan

### 7.1 Unit Tests

| ID | Type | Description | Req Ref |
|----|------|-------------|---------|
| TC-01 | unit | Silhouette coefficient matches sklearn for known data | REQ-UTL-002-01 |
| TC-02 | unit | Intra-cluster distance computation correct | REQ-UTL-002-02 |
| TC-03 | unit | Inter-cluster distance computation correct | REQ-UTL-002-03 |
| TC-04 | unit | Normalization maps [-1,1] to [0,1] | REQ-UTL-002-04 |
| TC-05 | unit | Single-member cluster returns 0.5 | EC-01 |
| TC-06 | unit | Empty clusters handled gracefully | EC-04, EC-05 |
| TC-07 | unit | Coherence formula weights applied correctly | REQ-UTL-002-06 |
| TC-08 | unit | Default weights match constitution | REQ-UTL-002-07 |

### 7.2 Integration Tests

| ID | Type | Description | Req Ref |
|----|------|-------------|---------|
| TC-09 | integration | ClusterFit integrates with CoherenceTracker | US-UTL-002-02 |
| TC-10 | integration | End-to-end coherence computation uses all three components | REQ-UTL-002-06 |
| TC-11 | integration | ClusterFit failure triggers fallback | AC-06 |

### 7.3 Performance Tests

| ID | Type | Description | Metric |
|----|------|-------------|--------|
| TC-12 | benchmark | ClusterFit latency < 2ms p95 | NFR-UTL-002-01 |
| TC-13 | benchmark | Coherence computation < 5ms p95 total | Perf budget |

---

## 8. Dependencies

### 8.1 Input Dependencies

| Dependency | Source | Description |
|------------|--------|-------------|
| Vertex embedding | Graph storage | The 13-dimensional teleological fingerprint |
| Cluster assignments | Clustering service | Which cluster each vertex belongs to |
| Neighbor embeddings | Graph storage | Embeddings of same-cluster and nearest-cluster vertices |

### 8.2 Output Consumers

| Consumer | Usage |
|----------|-------|
| UTL learning system | Delta-C feeds into learning score L |
| Johari Window classifier | Uses ΔC for quadrant classification |
| Memory consolidation | Coherence threshold for consolidation decisions |
| SELF_EGO_NODE | Identity continuity calculation |

---

## 9. Implementation Constraints

From constitution.yaml:

1. **AP-10**: No NaN/Infinity in UTL calculations
2. **AP-09**: No unbounded caches - cluster caches must have limits
3. **ARCH-02**: Apples-to-apples comparison - same embedder type only
4. **Latency**: Coherence layer budget is < 5ms total

---

## 10. Rollout Plan

1. **Phase 1**: Implement ClusterFitCalculator with unit tests
2. **Phase 2**: Integrate into CoherenceTracker
3. **Phase 3**: Update compute_coherence() to use three-component formula
4. **Phase 4**: Performance validation and tuning
5. **Phase 5**: Documentation update

---

## 11. References

- constitution.yaml: `delta_sc.ΔC` section
- PRD: UTL Learning Core requirements
- MASTER-CONSCIOUSNESS-GAP-ANALYSIS.md: GAP 2 - ClusterFit Missing
- sklearn.metrics.silhouette_score: Reference implementation

---

## Appendix A: Constitution Alignment

From `constitution.yaml`:

```yaml
delta_sc:
  ΔC: "α×Connectivity + β×ClusterFit + γ×Consistency (0.4, 0.4, 0.2)"
```

This specification fully implements the ClusterFit component (β×ClusterFit) to complete the coherence formula.
