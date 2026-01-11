# TASK-UTL-P1-001: DeltaSc Request/Response Types

## Metadata

| Field | Value |
|-------|-------|
| **ID** | TASK-UTL-P1-001 |
| **Version** | 1.0 |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 1 |
| **Priority** | P1 |
| **Estimated Complexity** | low |
| **Implements** | REQ-UTL-001, REQ-UTL-002, REQ-UTL-003, REQ-UTL-004, REQ-UTL-005, REQ-UTL-006, REQ-UTL-007, REQ-UTL-008 |
| **Depends On** | (none - first task) |
| **Spec Ref** | SPEC-UTL-001 |

---

## Context

This is the foundational task for the `compute_delta_sc` MCP tool. It creates the request/response types that define the tool's interface contract. All subsequent tasks depend on these types.

**Why This First**: Following the Inside-Out, Bottom-Up pattern:
1. Types must exist before business logic can use them
2. Types define the exact contract that handlers and tests will verify
3. No dependencies on other new code

---

## Input Context Files

| Purpose | File |
|---------|------|
| Schema reference | `specs/functional/SPEC-UTL-001.md#technical-design` |
| Existing fingerprint types | `crates/context-graph-core/src/types/fingerprint/teleological/types.rs` |
| Johari quadrant enum | `crates/context-graph-core/src/types/johari.rs` |
| Naming conventions | `docs2/constitution.yaml#naming` |
| NUM_EMBEDDERS constant | `crates/context-graph-core/src/johari/manager.rs` (pub const NUM_EMBEDDERS: usize = 13) |

---

## Prerequisites

| Check | Verification |
|-------|--------------|
| TeleologicalFingerprint exists | `grep -r "pub struct TeleologicalFingerprint" crates/context-graph-core/` |
| JohariQuadrant exists | `grep -r "pub enum JohariQuadrant" crates/context-graph-core/` |
| Serde available | `Cargo.toml` includes `serde = { features = ["derive"] }` |
| uuid available | `Cargo.toml` includes `uuid = { features = ["serde"] }` |

---

## Scope

### In Scope

- Create `ComputeDeltaScRequest` struct with all required fields
- Create `ComputeDeltaScResponse` struct with all required fields
- Create `DeltaScDiagnostics` struct for optional detailed output
- Implement `serde::Serialize` and `serde::Deserialize` for all types
- Add comprehensive doc comments referencing constitution.yaml
- Add unit tests for serialization round-trips

### Out of Scope

- MCP handler implementation (TASK-UTL-P1-003)
- Entropy computation logic (TASK-UTL-P1-002)
- Coherence computation logic (TASK-UTL-P1-002)
- Integration tests (TASK-UTL-P1-004)

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-mcp/src/types/delta_sc.rs

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use context_graph_core::types::fingerprint::TeleologicalFingerprint;
use context_graph_core::types::JohariQuadrant;

/// Request parameters for gwt/compute_delta_sc MCP tool.
///
/// Computes entropy (ΔS) and coherence (ΔC) changes when updating a vertex.
/// Per constitution.yaml gwt_tools, this is required for UTL learning.
///
/// # Constitution Reference
/// - delta_sc.ΔS_methods: Per-embedder entropy computation
/// - delta_sc.ΔC: α×Connectivity + β×ClusterFit + γ×Consistency (0.4, 0.4, 0.2)
/// - delta_sc.johari: Quadrant classification thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeDeltaScRequest {
    /// Vertex identifier (UUID).
    pub vertex_id: Uuid,

    /// Old teleological fingerprint (13 embeddings).
    /// Required: Must contain all 13 embedders per ARCH-05.
    pub old_fingerprint: TeleologicalFingerprint,

    /// New teleological fingerprint (13 embeddings).
    /// Required: Must contain all 13 embedders per ARCH-05.
    pub new_fingerprint: TeleologicalFingerprint,

    /// Include detailed diagnostics in response.
    /// Default: false
    #[serde(default)]
    pub include_diagnostics: bool,

    /// Override Johari threshold (default 0.5 from constitution).
    /// Range: [0.35, 0.65] per adaptive_thresholds.priors.θ_joh
    #[serde(skip_serializing_if = "Option::is_none")]
    pub johari_threshold: Option<f32>,
}

/// Response from gwt/compute_delta_sc MCP tool.
///
/// Contains per-embedder and aggregate ΔS/ΔC values plus Johari classification.
///
/// # Constitution Reference
/// - utl.canonical: L = f((ΔS × ΔC) · wₑ · cos φ)
/// - utl.johari: Per-space quadrant classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeDeltaScResponse {
    /// Per-embedder entropy change [0, 1].
    /// Index corresponds to embedder: 0=E1(Semantic), ..., 12=E13(KeywordSplade).
    pub delta_s_per_embedder: [f32; 13],

    /// Aggregate entropy change (weighted average across embedders).
    /// Range: [0.0, 1.0]
    pub delta_s_aggregate: f32,

    /// Coherence change [0, 1].
    /// Formula: 0.4×Connectivity + 0.4×ClusterFit + 0.2×Consistency
    pub delta_c: f32,

    /// Per-embedder Johari classification.
    /// Based on (ΔS, ΔC) thresholds from constitution.
    pub johari_quadrants: [JohariQuadrant; 13],

    /// Aggregate Johari classification (based on aggregate ΔS and ΔC).
    pub johari_aggregate: JohariQuadrant,

    /// Combined UTL learning potential: ΔS × ΔC.
    /// High values (>0.5) indicate high learning opportunity.
    pub utl_learning_potential: f32,

    /// Optional diagnostics (if requested).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<DeltaScDiagnostics>,
}

/// Detailed diagnostics for debugging and monitoring.
///
/// Only included when `include_diagnostics: true` in request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaScDiagnostics {
    /// Per-embedder computation method used.
    /// E.g., ["GMM+Mahalanobis", "KNN", "KNN", ...]
    pub methods_used: [String; 13],

    /// Connectivity component of Delta-C (weight: 0.4).
    pub connectivity: f32,

    /// ClusterFit component of Delta-C (weight: 0.4).
    pub cluster_fit: f32,

    /// Consistency component of Delta-C (weight: 0.2).
    pub consistency: f32,

    /// Computation time in microseconds.
    pub computation_time_us: u64,

    /// Embedder weights used for aggregate calculation.
    pub embedder_weights: [f32; 13],
}
```

### Constraints

- All fields MUST match SPEC-UTL-001 exactly
- NO `any` type or dynamic typing
- NO default implementations that bypass validation
- All f32 fields document their valid range
- Follow constitution.yaml naming conventions (snake_case for fields)
- Array size MUST be const 13 (not Vec) per ARCH-05
- JohariQuadrant MUST be reused from context-graph-core (no duplication)

### Verification

```bash
# Type check compiles
cargo check -p context-graph-mcp

# Unit tests pass
cargo test -p context-graph-mcp delta_sc::tests

# Clippy passes
cargo clippy -p context-graph-mcp -- -D warnings
```

---

## Pseudo Code

```
// File structure:
// crates/context-graph-mcp/src/types/delta_sc.rs  <- NEW
// crates/context-graph-mcp/src/types/mod.rs       <- MODIFY (add pub mod delta_sc)

// delta_sc.rs structure:
1. Imports (serde, uuid, context_graph_core types)
2. ComputeDeltaScRequest struct with derive macros
3. ComputeDeltaScResponse struct with derive macros
4. DeltaScDiagnostics struct with derive macros
5. #[cfg(test)] module with:
   - test_request_serialization_roundtrip
   - test_response_serialization_roundtrip
   - test_diagnostics_omitted_when_none
   - test_johari_threshold_validation
```

---

## Files to Create

| Path | Description |
|------|-------------|
| `crates/context-graph-mcp/src/types/delta_sc.rs` | Request/Response types with tests |

---

## Files to Modify

| Path | Change |
|------|--------|
| `crates/context-graph-mcp/src/types/mod.rs` | Add `pub mod delta_sc;` and re-export types |

---

## Validation Criteria

| Criterion | Verification Method |
|-----------|---------------------|
| Types compile without errors | `cargo check -p context-graph-mcp` |
| Serialization roundtrip works | Unit test `test_request_serialization_roundtrip` |
| Response fields match SPEC-UTL-001 | Code review against spec |
| Array size is exactly 13 | Compile-time (fixed array type) |
| No clippy warnings | `cargo clippy -p context-graph-mcp -- -D warnings` |
| Doc comments reference constitution | Code review |

---

## Test Commands

```bash
# Compile check
cargo check -p context-graph-mcp

# Run unit tests
cargo test -p context-graph-mcp delta_sc::tests -- --nocapture

# Clippy
cargo clippy -p context-graph-mcp -- -D warnings

# Doc tests
cargo test -p context-graph-mcp --doc
```

---

## Notes

- This task has no dependencies and can be executed immediately
- The types define the contract that TASK-UTL-P1-002 and TASK-UTL-P1-003 will implement
- JohariQuadrant is reused from context-graph-core to maintain single source of truth
- TeleologicalFingerprint is reused (not wrapped) per ARCH-01 atomicity
