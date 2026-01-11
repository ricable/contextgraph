# SPEC-UTL-001: compute_delta_sc MCP Tool

## Metadata

| Field | Value |
|-------|-------|
| **ID** | SPEC-UTL-001 |
| **Version** | 1.0 |
| **Status** | approved |
| **Owner** | ContextGraph Team |
| **Created** | 2026-01-11 |
| **Last Updated** | 2026-01-11 |
| **Related Specs** | SPEC-UTL-002 (ClusterFit), SPEC-UTL-003, constitution.yaml |
| **Priority** | P1 (Substantial Gap - GAP 1) |

---

## 1. Overview

### 1.1 Purpose

This specification defines the `compute_delta_sc` MCP tool for computing entropy (Delta-S) and coherence (Delta-C) changes when updating a vertex in the knowledge graph. This tool is essential for the UTL (Unified Theory of Learning) system to calculate learning scores and classify memories in the Johari Window.

### 1.2 Problem Statement

Per the MASTER-CONSCIOUSNESS-GAP-ANALYSIS.md (GAP 1), the `compute_delta_sc` MCP tool is **missing**. The PRD and constitution.yaml mandate:

```yaml
gwt_tools: [get_consciousness_state, get_workspace_status, get_kuramoto_sync,
            get_ego_state, trigger_workspace_broadcast, adjust_coupling,
            get_johari_classification, compute_delta_sc]  # <-- MISSING
```

**Impact of Missing Tool**:
- External systems cannot compute entropy/coherence deltas
- Claude Code hooks cannot trigger UTL learning calculations
- Johari Window classification lacks real-time update capability
- Memory consolidation decisions cannot be made autonomously

### 1.3 Success Criteria

1. MCP tool registered and discoverable via `tools/list`
2. Returns per-embedder Delta-S values using constitution-specified methods
3. Returns aggregate Delta-C using three-component formula
4. Computes Johari quadrant classification
5. Latency budget: < 25ms p95
6. All 13 embedders supported with specialized methods

---

## 2. User Stories

### US-UTL-001-01: Compute Delta-S/Delta-C for Vertex Update
**Priority**: must-have

**Narrative**:
> As the UTL learning system,
> I want to compute Delta-S and Delta-C when a vertex embedding changes,
> So that I can calculate the learning score and update memory importance.

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-01 | A vertex ID and old/new embeddings | `compute_delta_sc` is called | Returns Delta-S per embedder and aggregate Delta-C |
| AC-02 | Valid teleological fingerprint | Delta-S computed | Uses constitution-specified method for each embedder (E1: GMM, E2-4: KNN, etc.) |
| AC-03 | Valid embeddings | Delta-C computed | Uses formula: 0.4×Connectivity + 0.4×ClusterFit + 0.2×Consistency |
| AC-04 | Delta-S and Delta-C computed | Johari classification | Returns quadrant (Open, Blind, Hidden, Unknown) per thresholds |

### US-UTL-001-02: Support Per-Embedder Entropy Methods
**Priority**: must-have

**Narrative**:
> As the entropy computation module,
> I want specialized Delta-S methods for each embedder,
> So that entropy is measured appropriately for each semantic space.

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-05 | E1 (Semantic) embedding | Delta-S computed | Uses GMM+Mahalanobis: ΔS = 1 - P(e|GMM) |
| AC-06 | E2-4, E8 embeddings | Delta-S computed | Uses KNN: ΔS = σ((d_k - μ) / σ) |
| AC-07 | E5 (Causal) embedding | Delta-S computed | Uses Asymmetric KNN: ΔS = d_k × direction_mod |
| AC-08 | E6, E13 (Sparse) embeddings | Delta-S computed | Uses IDF/Jaccard: ΔS = IDF(dims) or 1-jaccard |
| AC-09 | E7 (Code) embedding | Delta-S computed | Uses GMM+KNN hybrid: ΔS = 0.5×GMM + 0.5×KNN |
| AC-10 | E9 (HDC/Binary) embedding | Delta-S computed | Uses Hamming: ΔS = min_hamming / dim |
| AC-11 | E10 (Multimodal) embedding | Delta-S computed | Uses Cross-modal KNN: ΔS = avg(d_text, d_image) |
| AC-12 | E11 (Entity) embedding | Delta-S computed | Uses TransE: ΔS = ||h + r - t|| |
| AC-13 | E12 (LateInteraction) embedding | Delta-S computed | Uses Token KNN: ΔS = max_token(d_k) |

### US-UTL-001-03: Johari Window Classification
**Priority**: must-have

**Narrative**:
> As the consciousness monitoring system,
> I want Johari Window classification for each embedder space,
> So that I can understand the knowledge state in each semantic dimension.

**Acceptance Criteria**:

| ID | Given | When | Then |
|----|-------|------|------|
| AC-14 | ΔS ≤ 0.5 AND ΔC > 0.5 | Johari computed | Returns "Open" (aware knowledge) |
| AC-15 | ΔS > 0.5 AND ΔC ≤ 0.5 | Johari computed | Returns "Blind" (discovery opportunity) |
| AC-16 | ΔS ≤ 0.5 AND ΔC ≤ 0.5 | Johari computed | Returns "Hidden" (dormant knowledge) |
| AC-17 | ΔS > 0.5 AND ΔC > 0.5 | Johari computed | Returns "Unknown" (frontier knowledge) |

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Story Ref | Priority | Rationale |
|----|-------------|-----------|----------|-----------|
| REQ-UTL-001 | System SHALL expose MCP tool `gwt/compute_delta_sc` | US-UTL-001-01 | must | MCP protocol compliance |
| REQ-UTL-002 | Tool SHALL accept `vertex_id`, `old_fingerprint`, `new_fingerprint` parameters | US-UTL-001-01 | must | Input contract |
| REQ-UTL-003 | Tool SHALL return `delta_s_per_embedder: [f32; 13]` array | US-UTL-001-02 | must | Per-embedder entropy |
| REQ-UTL-004 | Tool SHALL return `delta_s_aggregate: f32` (weighted average) | US-UTL-001-01 | must | Combined entropy |
| REQ-UTL-005 | Tool SHALL return `delta_c: f32` in [0, 1] range | US-UTL-001-01 | must | Coherence value |
| REQ-UTL-006 | Tool SHALL return `johari_quadrants: [JohariQuadrant; 13]` | US-UTL-001-03 | must | Per-embedder classification |
| REQ-UTL-007 | Tool SHALL return `johari_aggregate: JohariQuadrant` | US-UTL-001-03 | must | Overall classification |
| REQ-UTL-008 | Tool SHALL return `utl_learning_potential: f32` computed as ΔS × ΔC | US-UTL-001-01 | must | Combined learning metric |
| REQ-UTL-009 | System SHALL use E1 GMM+Mahalanobis method for semantic entropy | US-UTL-001-02 | must | Constitution alignment |
| REQ-UTL-010 | System SHALL use KNN method for E2, E3, E4, E8 entropy | US-UTL-001-02 | must | Constitution alignment |
| REQ-UTL-011 | System SHALL use asymmetric KNN for E5 (causal) entropy | US-UTL-001-02 | must | Causality direction |
| REQ-UTL-012 | System SHALL use IDF/Jaccard for E6, E13 sparse entropy | US-UTL-001-02 | must | Sparse embedding handling |
| REQ-UTL-013 | System SHALL use GMM+KNN hybrid for E7 (code) entropy | US-UTL-001-02 | must | Code semantics |
| REQ-UTL-014 | System SHALL use Hamming distance for E9 (binary) entropy | US-UTL-001-02 | must | Binary embedding handling |
| REQ-UTL-015 | System SHALL use cross-modal KNN for E10 entropy | US-UTL-001-02 | must | Multimodal handling |
| REQ-UTL-016 | System SHALL use TransE distance for E11 entropy | US-UTL-001-02 | must | Entity embedding handling |
| REQ-UTL-017 | System SHALL use token-level MaxSim for E12 entropy | US-UTL-001-02 | must | Late interaction handling |
| REQ-UTL-018 | System SHALL compute Delta-C using formula: 0.4×Connectivity + 0.4×ClusterFit + 0.2×Consistency | US-UTL-001-01 | must | Constitution formula |
| REQ-UTL-019 | System SHALL use Johari thresholds: θ = 0.5 from constitution | US-UTL-001-03 | must | Constitution alignment |
| REQ-UTL-020 | Tool SHALL optionally accept `include_diagnostics: bool` for detailed output | US-UTL-001-01 | should | Debugging support |

### 3.2 Non-Functional Requirements

| ID | Category | Requirement | Metric | Rationale |
|----|----------|-------------|--------|-----------|
| NFR-UTL-001 | Performance | Tool SHALL complete in < 25ms p95 | Latency | inject_context budget |
| NFR-UTL-002 | Performance | Tool SHALL complete in < 50ms p99 | Latency | constitution perf budget |
| NFR-UTL-003 | Accuracy | Delta-S SHALL be in [0, 1] range | Bounds | Normalized output |
| NFR-UTL-004 | Accuracy | Delta-C SHALL be in [0, 1] range | Bounds | Normalized output |
| NFR-UTL-005 | Robustness | Tool SHALL handle missing embedders gracefully | Error handling | Partial fingerprints |
| NFR-UTL-006 | Testability | All entropy methods SHALL have unit tests | Coverage > 90% | Quality |
| NFR-UTL-007 | Observability | Tool SHALL emit tracing spans for each embedder | Tracing | Debugging |

---

## 4. Technical Design

### 4.1 MCP Tool Interface

```rust
/// Request parameters for compute_delta_sc
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeDeltaScRequest {
    /// Vertex identifier (UUID)
    pub vertex_id: Uuid,

    /// Old teleological fingerprint (13 embeddings)
    pub old_fingerprint: TeleologicalFingerprint,

    /// New teleological fingerprint (13 embeddings)
    pub new_fingerprint: TeleologicalFingerprint,

    /// Optional: include detailed diagnostics
    #[serde(default)]
    pub include_diagnostics: bool,

    /// Optional: override Johari threshold (default 0.5)
    pub johari_threshold: Option<f32>,
}

/// Response from compute_delta_sc
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeDeltaScResponse {
    /// Per-embedder entropy change [0, 1]
    pub delta_s_per_embedder: [f32; 13],

    /// Aggregate entropy change (weighted average)
    pub delta_s_aggregate: f32,

    /// Coherence change [0, 1]
    pub delta_c: f32,

    /// Per-embedder Johari classification
    pub johari_quadrants: [JohariQuadrant; 13],

    /// Aggregate Johari classification
    pub johari_aggregate: JohariQuadrant,

    /// Combined UTL learning potential: ΔS × ΔC
    pub utl_learning_potential: f32,

    /// Optional diagnostics (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<DeltaScDiagnostics>,
}

/// Johari Window quadrant
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JohariQuadrant {
    /// ΔS ≤ 0.5, ΔC > 0.5 - Known and integrated
    Open,
    /// ΔS > 0.5, ΔC ≤ 0.5 - High novelty, low integration
    Blind,
    /// ΔS ≤ 0.5, ΔC ≤ 0.5 - Low novelty, low integration
    Hidden,
    /// ΔS > 0.5, ΔC > 0.5 - High novelty, high integration (frontier)
    Unknown,
}

/// Detailed diagnostics for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaScDiagnostics {
    /// Per-embedder computation method used
    pub methods_used: [String; 13],

    /// Connectivity component of Delta-C
    pub connectivity: f32,

    /// ClusterFit component of Delta-C
    pub cluster_fit: f32,

    /// Consistency component of Delta-C
    pub consistency: f32,

    /// Computation time in microseconds
    pub computation_time_us: u64,

    /// Embedder weights used
    pub embedder_weights: [f32; 13],
}
```

### 4.2 Entropy Methods by Embedder

From constitution.yaml `delta_sc.ΔS_methods`:

| Embedder | Method | Formula | Implementation |
|----------|--------|---------|----------------|
| E1 | GMM+Mahalanobis | ΔS = 1 - P(e\|GMM) | `GmmEntropyCalculator` |
| E2-4, E8 | KNN | ΔS = σ((d_k - μ) / σ) | `KnnEntropyCalculator` |
| E5 | Asymmetric KNN | ΔS = d_k × direction_mod | `AsymmetricKnnEntropyCalculator` |
| E6, E13 | IDF/Jaccard | ΔS = IDF(dims) or 1-jaccard | `SparseEntropyCalculator` |
| E7 | GMM+KNN Hybrid | ΔS = 0.5×GMM + 0.5×KNN | `HybridEntropyCalculator` |
| E9 | Hamming | ΔS = min_hamming / dim | `HammingEntropyCalculator` |
| E10 | Cross-modal KNN | ΔS = avg(d_text, d_image) | `CrossModalEntropyCalculator` |
| E11 | TransE | ΔS = \|\|h + r - t\|\| | `TransEEntropyCalculator` |
| E12 | Token KNN (MaxSim) | ΔS = max_token(d_k) | `MaxSimEntropyCalculator` |

### 4.3 Coherence Computation

```rust
/// Compute Delta-C using three-component formula
pub fn compute_delta_c(
    vertex_id: Uuid,
    new_fingerprint: &TeleologicalFingerprint,
    graph_context: &GraphContext,
) -> f32 {
    const ALPHA: f32 = 0.4;  // Connectivity weight
    const BETA: f32 = 0.4;   // ClusterFit weight
    const GAMMA: f32 = 0.2;  // Consistency weight

    let connectivity = compute_connectivity(vertex_id, new_fingerprint, graph_context);
    let cluster_fit = compute_cluster_fit(vertex_id, new_fingerprint, graph_context);
    let consistency = compute_consistency(vertex_id, new_fingerprint, graph_context);

    (ALPHA * connectivity + BETA * cluster_fit + GAMMA * consistency).clamp(0.0, 1.0)
}
```

### 4.4 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         compute_delta_sc MCP Handler                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Request: { vertex_id, old_fingerprint, new_fingerprint }                   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     DeltaScComputer                                   │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │                                                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────┐     │   │
│  │  │              EntropyComputerRegistry                        │     │   │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │     │   │
│  │  │  │E1: GMM  │ │E2-4:KNN │ │E5:Asym  │ │E6:IDF   │ ...       │     │   │
│  │  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘           │     │   │
│  │  │       │           │           │           │                 │     │   │
│  │  │       └───────────┴───────────┴───────────┘                 │     │   │
│  │  │                       │                                     │     │   │
│  │  │              delta_s_per_embedder: [f32; 13]                │     │   │
│  │  └─────────────────────────────────────────────────────────────┘     │   │
│  │                              │                                        │   │
│  │  ┌───────────────────────────┴───────────────────────────────────┐   │   │
│  │  │                   CoherenceComputer                            │   │   │
│  │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐                 │   │   │
│  │  │  │Connectivity│ │ ClusterFit │ │ Consistency│                 │   │   │
│  │  │  │   0.4      │ │    0.4     │ │    0.2     │                 │   │   │
│  │  │  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘                 │   │   │
│  │  │        │              │              │                         │   │   │
│  │  │        └──────────────┼──────────────┘                         │   │   │
│  │  │                       ▼                                        │   │   │
│  │  │              delta_c: f32                                      │   │   │
│  │  └────────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                        │   │
│  │  ┌───────────────────────────┴───────────────────────────────────┐   │   │
│  │  │                  JohariClassifier                              │   │   │
│  │  │  For each (ΔS, ΔC) pair:                                      │   │   │
│  │  │    ΔS ≤ 0.5 ∧ ΔC > 0.5  → Open                                │   │   │
│  │  │    ΔS > 0.5 ∧ ΔC ≤ 0.5  → Blind                               │   │   │
│  │  │    ΔS ≤ 0.5 ∧ ΔC ≤ 0.5  → Hidden                              │   │   │
│  │  │    ΔS > 0.5 ∧ ΔC > 0.5  → Unknown                             │   │   │
│  │  └────────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                        │   │
│  └──────────────────────────────┴────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  Response: { delta_s_per_embedder, delta_s_aggregate, delta_c,              │
│              johari_quadrants, johari_aggregate, utl_learning_potential }   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Edge Cases

| ID | Scenario | Expected Behavior |
|----|----------|-------------------|
| EC-01 | Old fingerprint has missing embedder (partial) | Use fallback ΔS = 0.5 for missing embedder, log warning |
| EC-02 | New fingerprint has missing embedder | Return error: incomplete fingerprint violates ARCH-05 |
| EC-03 | Fingerprints are identical | Return ΔS = 0.0 for all embedders, ΔC = current coherence |
| EC-04 | Vertex not found in graph | Compute ΔS from embeddings only, skip connectivity component |
| EC-05 | No cluster context available | ClusterFit = 0.5 (neutral), log warning |
| EC-06 | Binary embedding (E9) has wrong dimensions | Return error: dimension mismatch |
| EC-07 | Sparse embedding (E6, E13) is empty | ΔS = 1.0 (maximum entropy) |
| EC-08 | Token embedding (E12) has no tokens | ΔS = 0.5 (neutral) |

---

## 6. Error States

| ID | HTTP | Condition | Message | Recovery |
|----|------|-----------|---------|----------|
| ERR-UTL-001 | -32602 | Missing required parameter | "Missing required parameter: {param}" | Return immediately |
| ERR-UTL-002 | -32602 | Invalid UUID format | "Invalid vertex_id format" | Return immediately |
| ERR-UTL-003 | -32602 | Incomplete fingerprint | "Fingerprint must contain all 13 embedders (ARCH-05)" | Return immediately |
| ERR-UTL-004 | -32603 | Entropy computation failed | "Failed to compute entropy for embedder {idx}: {error}" | Return partial result with fallback |
| ERR-UTL-005 | -32603 | Coherence computation failed | "Failed to compute coherence: {error}" | Return with ΔC = 0.5 fallback |
| ERR-UTL-006 | -32603 | Dimension mismatch | "Embedding dimension mismatch for E{idx}: expected {exp}, got {got}" | Return error |

---

## 7. Test Plan

### 7.1 Unit Tests

| ID | Type | Description | Req Ref |
|----|------|-------------|---------|
| TC-001 | unit | GMM entropy returns value in [0, 1] | REQ-UTL-009 |
| TC-002 | unit | KNN entropy returns value in [0, 1] | REQ-UTL-010 |
| TC-003 | unit | Asymmetric KNN applies direction modifier | REQ-UTL-011 |
| TC-004 | unit | Jaccard entropy handles sparse vectors | REQ-UTL-012 |
| TC-005 | unit | Hybrid GMM+KNN averages correctly | REQ-UTL-013 |
| TC-006 | unit | Hamming distance normalized to [0, 1] | REQ-UTL-014 |
| TC-007 | unit | Cross-modal averages text and image distances | REQ-UTL-015 |
| TC-008 | unit | TransE distance computed correctly | REQ-UTL-016 |
| TC-009 | unit | MaxSim uses max token distance | REQ-UTL-017 |
| TC-010 | unit | Delta-C uses correct weights (0.4, 0.4, 0.2) | REQ-UTL-018 |
| TC-011 | unit | Johari classification correct for all quadrants | REQ-UTL-019 |
| TC-012 | unit | Missing embedder returns fallback | EC-01 |
| TC-013 | unit | Identical fingerprints return zero entropy | EC-03 |

### 7.2 Integration Tests

| ID | Type | Description | Req Ref |
|----|------|-------------|---------|
| TC-014 | integration | MCP tool registered and discoverable | REQ-UTL-001 |
| TC-015 | integration | Full pipeline with real fingerprints | US-UTL-001-01 |
| TC-016 | integration | Coherence components integrate correctly | REQ-UTL-018 |
| TC-017 | integration | Response matches schema | REQ-UTL-003-008 |
| TC-018 | integration | Diagnostics included when requested | REQ-UTL-020 |

### 7.3 Performance Tests

| ID | Type | Description | Metric |
|----|------|-------------|--------|
| TC-019 | benchmark | Compute latency < 25ms p95 | NFR-UTL-001 |
| TC-020 | benchmark | Compute latency < 50ms p99 | NFR-UTL-002 |
| TC-021 | benchmark | All 13 embedders computed in parallel | Throughput |

### 7.4 Property-Based Tests

| ID | Type | Description | Property |
|----|------|-------------|----------|
| TC-022 | property | Delta-S always in [0, 1] | Bounded output |
| TC-023 | property | Delta-C always in [0, 1] | Bounded output |
| TC-024 | property | Johari quadrant always valid enum | Type safety |
| TC-025 | property | UTL potential = ΔS × ΔC | Invariant |

---

## 8. Dependencies

### 8.1 Input Dependencies

| Dependency | Source | Description |
|------------|--------|-------------|
| TeleologicalFingerprint | context-graph-core | 13-embedder fingerprint type |
| GraphContext | context-graph-storage | Graph connectivity information |
| ClusterContext | context-graph-core | Cluster assignments for ClusterFit |
| CoherenceTracker | context-graph-core | Historical consistency data |

### 8.2 Output Consumers

| Consumer | Usage |
|----------|-------|
| UTL Learning System | Computes L = f((ΔS × ΔC) · wₑ · cos φ) |
| Johari Window Display | Shows knowledge state visualization |
| Memory Consolidation | Decides whether to consolidate vertex |
| SELF_EGO_NODE | Updates identity trajectory |
| Claude Code Hooks | PostToolUse triggers learning |

---

## 9. Implementation Constraints

From constitution.yaml:

1. **ARCH-02**: Apples-to-apples comparison only - same embedder to same embedder
2. **ARCH-05**: All 13 embedders must be present in fingerprint
3. **AP-10**: No NaN/Infinity in UTL calculations
4. **AP-09**: No unbounded caches for entropy computation
5. **Latency**: inject_context budget is < 25ms p95

---

## 10. Rollout Plan

1. **Phase 1**: Implement DeltaScRequest/DeltaScResponse types (TASK-UTL-P1-001)
2. **Phase 2**: Implement entropy calculators per embedder (TASK-UTL-P1-002)
3. **Phase 3**: Wire coherence computation with ClusterFit (TASK-UTL-P1-002)
4. **Phase 4**: Register MCP handler (TASK-UTL-P1-003)
5. **Phase 5**: Integration tests (TASK-UTL-P1-004)

---

## 11. References

- constitution.yaml: `delta_sc` section
- constitution.yaml: `embeddings` section (per-embedder specs)
- constitution.yaml: `gwt_tools` list
- MASTER-CONSCIOUSNESS-GAP-ANALYSIS.md: GAP 1 - compute_delta_sc Missing
- SPEC-UTL-002: ClusterFit component (dependency)
- PRD: UTL Learning Core requirements

---

## Appendix A: Constitution Alignment

From `constitution.yaml`:

```yaml
delta_sc:
  ΔS_methods:
    E1: "GMM+Mahalanobis: ΔS=1-P(e|GMM)"
    E2-4,E8: "KNN: ΔS=σ((d_k-μ)/σ)"
    E5: "Asymmetric KNN: ΔS=d_k×direction_mod"
    E6,E13: "IDF/Jaccard: ΔS=IDF(dims) or 1-jaccard"
    E7: "GMM+KNN: ΔS=0.5×GMM+0.5×KNN"
    E9: "Hamming: ΔS=min_hamming/dim"
    E10: "Cross-modal KNN: ΔS=avg(d_text,d_image)"
    E11: "TransE: ΔS=||h+r-t||"
    E12: "Token KNN: ΔS=max_token(d_k)"

  ΔC: "α×Connectivity + β×ClusterFit + γ×Consistency (0.4, 0.4, 0.2)"

  johari:
    Open: "ΔS≤0.5∧ΔC>0.5"
    Blind: "ΔS>0.5∧ΔC≤0.5"
    Hidden: "ΔS≤0.5∧ΔC≤0.5"
    Unknown: "ΔS>0.5∧ΔC>0.5"
```

This specification fully implements the `compute_delta_sc` MCP tool as required by the gwt_tools list in constitution.yaml.

---

## Appendix B: Cross-Space Insights

Per constitution.yaml `delta_sc.cross_space_insights`:

| Pattern | Meaning | Recommended Action |
|---------|---------|-------------------|
| Open(semantic) + Blind(causal) | Knows WHAT but not WHY | Explore causal relationships |
| Blind(semantic) + Open(code) | Code without context | Add documentation/comments |
| Unknown(all) | Frontier knowledge | High-priority learning opportunity |
| Hidden(all) | Obsolete knowledge | Candidate for pruning |
