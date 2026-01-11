# TASK-DELTA-P1-002: DeltaScComputer Implementation

## Metadata

| Field | Value |
|-------|-------|
| **ID** | TASK-DELTA-P1-002 |
| **Version** | 1.0 |
| **Status** | ready |
| **Layer** | logic |
| **Sequence** | 2 of 4 |
| **Priority** | P1 |
| **Estimated Complexity** | high |
| **Estimated Duration** | 4-6 hours |
| **Implements** | REQ-UTL-009 through REQ-UTL-019 |
| **Depends On** | TASK-DELTA-P1-001 |
| **Spec Ref** | SPEC-UTL-001 |
| **Gap Ref** | MASTER-CONSCIOUSNESS-GAP-ANALYSIS.md GAP 1 |

---

## Context

This task implements the core business logic for computing Delta-S (entropy) and Delta-C (coherence) across all 13 embedders. It integrates existing per-embedder entropy calculators with coherence computation and Johari classification.

**Why This Second**: Following Inside-Out, Bottom-Up:
1. Types (TASK-DELTA-P1-001) must exist first
2. Business logic depends on types but not on MCP registration
3. Logic can be unit tested in isolation before surface wiring

**Gap Being Addressed**:
> GAP 1: UTL compute_delta_sc MCP Tool Missing
> Core computation logic that powers the MCP tool

---

## Input Context Files

| Purpose | File |
|---------|------|
| Request/Response types | `crates/context-graph-mcp/src/types/delta_sc.rs` (from TASK-DELTA-P1-001) |
| Embedder entropy trait | `crates/context-graph-utl/src/surprise/embedder_entropy/mod.rs` |
| Entropy factory | `crates/context-graph-utl/src/surprise/embedder_entropy/factory.rs` |
| All entropy calculators | `crates/context-graph-utl/src/surprise/embedder_entropy/*.rs` |
| Coherence tracker | `crates/context-graph-utl/src/coherence/` (if exists) |
| Johari manager | `crates/context-graph-core/src/johari/default_manager.rs` |
| Constitution | `docs2/constitution.yaml#delta_sc` |

---

## Prerequisites

| Check | Verification |
|-------|--------------|
| TASK-DELTA-P1-001 complete | Types exist in `crates/context-graph-mcp/src/types/delta_sc.rs` |
| EmbedderEntropy trait exists | `grep "pub trait EmbedderEntropy" crates/context-graph-utl/` |
| EmbedderEntropyFactory exists | `grep "EmbedderEntropyFactory" crates/context-graph-utl/` |
| JohariQuadrant::classify exists | `grep "JohariQuadrant" crates/context-graph-core/` |
| NUM_EMBEDDERS = 13 | `grep "NUM_EMBEDDERS.*13" crates/context-graph-core/` |

---

## Scope

### In Scope

- Create `DeltaScComputer` struct that orchestrates computation
- Implement `compute()` method returning `ComputeDeltaScResponse`
- Wire existing `EmbedderEntropyFactory::create_all()` for per-embedder Delta-S
- Implement coherence computation with three components:
  - Connectivity (use existing or stub returning 0.5)
  - ClusterFit (from SPEC-UTL-002 or stub if not ready)
  - Consistency (use existing or stub returning 0.5)
- Implement Johari classification using thresholds from constitution
- Add comprehensive unit tests for all 13 embedder methods
- Add property-based tests for output bounds

### Out of Scope

- MCP handler registration (TASK-DELTA-P1-003)
- Integration tests with real graph data (TASK-DELTA-P1-004)
- Full ClusterFit implementation (covered by SPEC-UTL-002)

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-mcp/src/services/delta_sc_computer.rs

use std::time::Instant;
use uuid::Uuid;

use context_graph_core::johari::NUM_EMBEDDERS;
use context_graph_core::teleological::Embedder;
use context_graph_core::types::fingerprint::TeleologicalFingerprint;
use context_graph_core::types::JohariQuadrant;
use context_graph_utl::surprise::embedder_entropy::{EmbedderEntropy, EmbedderEntropyFactory};
use context_graph_utl::config::SurpriseConfig;

use crate::error::McpResult;
use crate::types::delta_sc::{ComputeDeltaScRequest, ComputeDeltaScResponse, DeltaScDiagnostics};

/// Constitution weights for coherence computation.
const COHERENCE_CONNECTIVITY_WEIGHT: f32 = 0.4;
const COHERENCE_CLUSTER_FIT_WEIGHT: f32 = 0.4;
const COHERENCE_CONSISTENCY_WEIGHT: f32 = 0.2;

/// Default Johari threshold from constitution.
const DEFAULT_JOHARI_THRESHOLD: f32 = 0.5;

/// Method names for diagnostics, per constitution.yaml delta_sc.Delta_S_methods.
const EMBEDDER_METHODS: [&str; 13] = [
    "GMM+Mahalanobis",   // E1 Semantic
    "KNN",               // E2 TemporalRecent
    "KNN",               // E3 TemporalPeriodic
    "KNN",               // E4 TemporalPositional
    "Asymmetric KNN",    // E5 Causal
    "IDF/Jaccard",       // E6 Sparse
    "GMM+KNN Hybrid",    // E7 Code
    "KNN",               // E8 Graph
    "Hamming",           // E9 Hdc
    "Cross-modal KNN",   // E10 Multimodal
    "TransE",            // E11 Entity
    "Token MaxSim",      // E12 LateInteraction
    "Jaccard",           // E13 KeywordSplade
];

/// Computes Delta-S (entropy) and Delta-C (coherence) for fingerprint updates.
///
/// Orchestrates per-embedder entropy calculators and coherence components
/// per constitution.yaml delta_sc section.
///
/// # Thread Safety
/// This struct is Send + Sync for use in async MCP handlers.
pub struct DeltaScComputer {
    /// Per-embedder entropy calculators (created once, reused).
    entropy_calculators: Vec<Box<dyn EmbedderEntropy>>,

    /// Surprise configuration.
    config: SurpriseConfig,
}

impl DeltaScComputer {
    /// Create a new DeltaScComputer with default configuration.
    pub fn new() -> Self;

    /// Create with custom configuration.
    pub fn with_config(config: SurpriseConfig) -> Self;

    /// Compute Delta-S and Delta-C for a fingerprint update.
    ///
    /// # Arguments
    /// * `request` - The computation request with old/new fingerprints
    ///
    /// # Returns
    /// Complete response with per-embedder and aggregate values.
    ///
    /// # Errors
    /// - `McpError::InvalidFingerprint` if fingerprints incomplete
    /// - `McpError::ComputationError` if entropy calculation fails
    ///
    /// # Performance
    /// Target: < 25ms p95 per constitution.yaml perf.latency.inject_context
    pub async fn compute(&self, request: &ComputeDeltaScRequest) -> McpResult<ComputeDeltaScResponse>;

    /// Validate that both fingerprints contain all 13 embedders.
    fn validate_fingerprints(
        &self,
        old: &TeleologicalFingerprint,
        new: &TeleologicalFingerprint,
    ) -> McpResult<()>;

    /// Compute Delta-S for all 13 embedders.
    async fn compute_delta_s_all(
        &self,
        old: &TeleologicalFingerprint,
        new: &TeleologicalFingerprint,
    ) -> McpResult<[f32; 13]>;

    /// Compute weighted aggregate Delta-S.
    fn compute_aggregate_delta_s(&self, per_embedder: &[f32; 13], weights: &[f32; 13]) -> f32;

    /// Compute Delta-C with three components.
    /// Returns (delta_c, connectivity, cluster_fit, consistency).
    async fn compute_delta_c(
        &self,
        vertex_id: Uuid,
        fingerprint: &TeleologicalFingerprint,
    ) -> McpResult<(f32, f32, f32, f32)>;

    /// Classify Johari quadrant for a single (Delta-S, Delta-C) pair.
    fn classify_johari(delta_s: f32, delta_c: f32, threshold: f32) -> JohariQuadrant;

    /// Classify Johari for all 13 embedders.
    fn classify_johari_per_embedder(
        &self,
        delta_s_per_embedder: &[f32; 13],
        delta_c: f32,
        threshold: f32,
    ) -> [JohariQuadrant; 13];
}

impl Default for DeltaScComputer {
    fn default() -> Self {
        Self::new()
    }
}
```

### Constraints

- All f32 outputs MUST be clamped to [0.0, 1.0] per AP-10
- NO NaN or Infinity values (validate and clamp)
- Use existing `EmbedderEntropyFactory` - do NOT reimplement calculators
- Johari classification MUST match constitution thresholds exactly
- Coherence weights MUST be 0.4, 0.4, 0.2 per constitution
- Method names in diagnostics MUST match constitution.yaml delta_sc.Delta_S_methods
- If ClusterFit not available from SPEC-UTL-002, use stub returning 0.5

### Verification

```bash
# Unit tests pass
cargo test -p context-graph-mcp delta_sc_computer::tests -- --nocapture

# All outputs in valid range
cargo test -p context-graph-mcp test_all_outputs_bounded

# Johari classification correct
cargo test -p context-graph-mcp test_johari_classification

# No clippy warnings
cargo clippy -p context-graph-mcp -- -D warnings
```

---

## Pseudo Code

```rust
// compute() implementation:
fn compute(&self, request: &ComputeDeltaScRequest) -> McpResult<ComputeDeltaScResponse> {
    let start = Instant::now();

    // 1. Validate fingerprints (ARCH-05: all 13 embedders required)
    self.validate_fingerprints(&request.old_fingerprint, &request.new_fingerprint)?;

    // 2. Compute per-embedder Delta-S
    let delta_s_per_embedder = self.compute_delta_s_all(
        &request.old_fingerprint,
        &request.new_fingerprint,
    ).await?;

    // 3. Compute aggregate Delta-S (uniform weights for now)
    let embedder_weights = [1.0 / 13.0; 13];
    let delta_s_aggregate = self.compute_aggregate_delta_s(&delta_s_per_embedder, &embedder_weights);

    // 4. Compute Delta-C (coherence)
    let (delta_c, connectivity, cluster_fit, consistency) = self.compute_delta_c(
        request.vertex_id,
        &request.new_fingerprint,
    ).await?;

    // 5. Classify Johari quadrants
    let johari_threshold = request.johari_threshold.unwrap_or(DEFAULT_JOHARI_THRESHOLD);
    let johari_quadrants = self.classify_johari_per_embedder(&delta_s_per_embedder, delta_c, johari_threshold);
    let johari_aggregate = Self::classify_johari(delta_s_aggregate, delta_c, johari_threshold);

    // 6. Compute UTL learning potential
    let utl_learning_potential = delta_s_aggregate * delta_c;

    // 7. Build diagnostics if requested
    let diagnostics = if request.include_diagnostics {
        Some(DeltaScDiagnostics {
            methods_used: EMBEDDER_METHODS.map(String::from),
            connectivity,
            cluster_fit,
            consistency,
            computation_time_us: start.elapsed().as_micros() as u64,
            embedder_weights,
        })
    } else {
        None
    };

    Ok(ComputeDeltaScResponse {
        delta_s_per_embedder,
        delta_s_aggregate,
        delta_c,
        johari_quadrants,
        johari_aggregate,
        utl_learning_potential,
        diagnostics,
    })
}

// compute_delta_s_all():
async fn compute_delta_s_all(&self, old: &TeleologicalFingerprint, new: &TeleologicalFingerprint) -> McpResult<[f32; 13]> {
    let mut delta_s = [0.0f32; 13];

    for idx in 0..13 {
        let old_embedding = old.semantic.get_embedding(Embedder::from_index(idx)?);
        let new_embedding = new.semantic.get_embedding(Embedder::from_index(idx)?);

        // Build history from old embedding
        let history = vec![old_embedding.to_vec()];

        // Use appropriate calculator
        let calculator = &self.entropy_calculators[idx];
        let raw_delta_s = calculator.compute_delta_s(new_embedding, &history, 5)?;

        // Clamp to [0, 1] per AP-10
        delta_s[idx] = raw_delta_s.clamp(0.0, 1.0);
    }

    Ok(delta_s)
}

// classify_johari():
fn classify_johari(delta_s: f32, delta_c: f32, threshold: f32) -> JohariQuadrant {
    // Per constitution.yaml delta_sc.johari
    match (delta_s <= threshold, delta_c > threshold) {
        (true, true) => JohariQuadrant::Open,    // Low surprise, high coherence
        (false, false) => JohariQuadrant::Blind, // High surprise, low coherence
        (true, false) => JohariQuadrant::Hidden, // Low surprise, low coherence
        (false, true) => JohariQuadrant::Unknown, // High surprise, high coherence
    }
}
```

---

## Files to Create

| Path | Description |
|------|-------------|
| `crates/context-graph-mcp/src/services/delta_sc_computer.rs` | Core computation logic |

---

## Files to Modify

| Path | Change |
|------|--------|
| `crates/context-graph-mcp/src/services/mod.rs` | Add `pub mod delta_sc_computer;` |
| `crates/context-graph-mcp/Cargo.toml` | Ensure `context-graph-utl` dependency |

---

## Validation Criteria

| Criterion | Verification Method |
|-----------|---------------------|
| All 13 embedder methods wired | Test each embedder produces output |
| Delta-S in [0, 1] for all embedders | Property test with random fingerprints |
| Delta-C in [0, 1] | Property test |
| Johari thresholds match constitution | Unit tests for boundary conditions |
| Coherence weights are 0.4/0.4/0.2 | Code review + unit test |
| No NaN/Infinity (AP-10) | Test with edge cases (zero vectors, etc.) |
| Performance < 25ms | Benchmark test |

---

## Test Commands

```bash
# Unit tests
cargo test -p context-graph-mcp delta_sc_computer -- --nocapture

# Property-based tests
cargo test -p context-graph-mcp test_delta_bounds

# Johari classification tests
cargo test -p context-graph-mcp test_johari

# Benchmark (if configured)
cargo bench -p context-graph-mcp delta_sc

# Full test suite
cargo test -p context-graph-mcp --lib
```

---

## Notes

- This task implements the bulk of SPEC-UTL-001 requirements
- Existing EmbedderEntropyFactory already handles per-embedder method selection
- If SPEC-UTL-002 (ClusterFit) is not complete, use stub returning 0.5
- The coherence components may need stubs initially; wire real implementations when available
- async is used even though computation is CPU-bound to match handler pattern
