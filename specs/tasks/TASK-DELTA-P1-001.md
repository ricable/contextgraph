# TASK-DELTA-P1-001: ComputeDeltaSc Request/Response Types

## Metadata

| Field | Value |
|-------|-------|
| **ID** | TASK-DELTA-P1-001 |
| **Version** | 2.0 |
| **Status** | COMPLETE |
| **Layer** | foundation |
| **Sequence** | 1 of 4 |
| **Priority** | P1 |
| **Estimated Complexity** | low |
| **Implements** | REQ-UTL-001, REQ-UTL-002, REQ-UTL-003, REQ-UTL-004, REQ-UTL-005, REQ-UTL-006, REQ-UTL-007, REQ-UTL-008 |
| **Depends On** | (none - first task) |
| **Spec Ref** | SPEC-UTL-001 |
| **Gap Ref** | MASTER-CONSCIOUSNESS-GAP-ANALYSIS.md GAP 1 |
| **Last Audited** | 2026-01-11 |

---

## ⚠️ CRITICAL: CURRENT STATE AUDIT (2026-01-11)

### What ACTUALLY EXISTS (vs. Original Task Description)

**IMPORTANT**: This task originally described creating separate type files (`types/delta_sc.rs`). That approach was **NOT implemented**. Instead, the implementation is:

| Original Plan | Actual Implementation |
|---------------|----------------------|
| `types/delta_sc.rs` with `ComputeDeltaScRequest` struct | **Does NOT exist** - types parsed inline from JSON |
| `types/delta_sc.rs` with `ComputeDeltaScResponse` struct | **Does NOT exist** - response built inline as `json!({...})` |
| `types/delta_sc.rs` with `DeltaScDiagnostics` struct | **Does NOT exist** - diagnostics built inline |
| `types/mod.rs` with module export | **No `types/` directory exists in context-graph-mcp** |

### Actual File Locations

| Component | Actual Location | Line Numbers |
|-----------|-----------------|--------------|
| Handler implementation | `crates/context-graph-mcp/src/handlers/utl.rs` | 1117-1387 |
| Tool registration | `crates/context-graph-mcp/src/tools.rs` | 318-380, 1112 |
| Tests | `crates/context-graph-mcp/src/handlers/tests/utl.rs` | 1-500+ |
| JohariQuadrant type | `crates/context-graph-core/src/types/johari/quadrant.rs` | - |
| TeleologicalFingerprint type | `crates/context-graph-core/src/types/fingerprint/teleological/core.rs` | - |
| classify_johari helper | `crates/context-graph-mcp/src/handlers/utl.rs` | 1390-1403 |
| sparse_to_dense_truncated helper | `crates/context-graph-mcp/src/handlers/utl.rs` | 1406-1416 |
| mean_pool_tokens helper | `crates/context-graph-mcp/src/handlers/utl.rs` | 1418-1437 |

### Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| `gwt/compute_delta_sc` handler | ✅ IMPLEMENTED | `handle_gwt_compute_delta_sc()` at utl.rs:1141 |
| Tool registration in tools/list | ✅ IMPLEMENTED | tools.rs:318-380 |
| Parameter parsing (vertex_id, fingerprints) | ✅ IMPLEMENTED | FAIL FAST error handling |
| Per-embedder ΔS computation | ✅ IMPLEMENTED | Uses EmbedderEntropyFactory |
| Aggregate ΔS computation | ✅ IMPLEMENTED | Equal weight average |
| Johari quadrant classification | ✅ IMPLEMENTED | classify_johari() helper |
| `include_diagnostics` parameter | ✅ IMPLEMENTED | Optional detailed output |
| `johari_threshold` parameter | ✅ IMPLEMENTED | Clamped to [0.35, 0.65] |
| AP-10 compliance (values [0,1], no NaN/Inf) | ✅ IMPLEMENTED | With clamping and warnings |
| ΔC Connectivity component | ✅ IMPLEMENTED | Uses CoherenceTracker similarity computation |
| ΔC ClusterFit component | ✅ IMPLEMENTED | Uses silhouette-based cluster_fit with divergent cluster |
| ΔC Consistency component | ✅ IMPLEMENTED | Uses CoherenceTracker window variance |
| Specialized per-embedder ΔS calculators | ⚠️ PARTIAL | Currently uses EmbedderEntropyFactory (per-embedder specialization is future work) |

### REMAINING GAPS (FUTURE WORK)

1. **~~ClusterFit Component~~**: ✅ COMPLETED - Implemented in TASK-DELTA-P1-001 using silhouette-based cluster fit
2. **Per-Embedder Specialized Calculators**: SPEC-UTL-001 §3.2 specifies different methods per embedder (future enhancement):
   - E1 Semantic: GMM + Mahalanobis distance (Σ updated via EMA)
   - E2 Syntactic: KNN with k=5
   - E3-E6, E8, E10, E12: KNN with k=5
   - E7 Entity: TransE distance + Hamming for IDs
   - E9 Sparse: MaxSim (ColBERT-style) with IDF weighting
   - E11 Relational: Hamming ratio for encoded edges
   - E13 Keyword: MaxSim + term overlap
3. **Separate Type Definitions**: Consider extracting inline JSON schemas to proper Rust types (optional refactor)

---

## Context

This is the foundational task for the `compute_delta_sc` MCP tool. It defines the request/response interface contract for computing Delta-S (entropy) and Delta-C (coherence) changes.

**Why This First**: Following the Inside-Out, Bottom-Up pattern from prdtospec.md:
1. Types must exist before business logic can use them
2. Types define the exact contract that handlers and tests will verify
3. No dependencies on other new code

**Gap Being Addressed**:
> GAP 1: UTL compute_delta_sc MCP Tool Missing
> External systems cannot compute entropy/coherence deltas

---

## Input Context Files (VERIFIED LOCATIONS)

| Purpose | File | Status |
|---------|------|--------|
| Schema reference | `specs/functional/SPEC-UTL-001.md#technical-design` | ✅ Exists |
| Constitution requirements | `docs2/constitution.yaml#delta_sc` | ✅ Exists |
| Existing handler | `crates/context-graph-mcp/src/handlers/utl.rs` | ✅ Exists (lines 1117-1387) |
| Tool registration | `crates/context-graph-mcp/src/tools.rs` | ✅ Exists (lines 318-380) |
| Tests | `crates/context-graph-mcp/src/handlers/tests/utl.rs` | ✅ Exists |
| TeleologicalFingerprint | `crates/context-graph-core/src/types/fingerprint/teleological/core.rs` | ✅ Exists |
| JohariQuadrant enum | `crates/context-graph-core/src/types/johari/quadrant.rs` | ✅ Exists |
| NUM_EMBEDDERS constant | `crates/context-graph-core/src/johari/manager.rs` | ✅ Exists (value: 13) |
| CoherenceTracker | `crates/context-graph-core/src/coherence/tracker.rs` | ✅ Exists |
| EmbedderEntropyFactory | `crates/context-graph-core/src/surprise/factory.rs` | ✅ Exists |

---

## Prerequisites (VERIFIED)

| Check | Verification | Status |
|-------|--------------|--------|
| TeleologicalFingerprint exists | `crates/context-graph-core/src/types/fingerprint/teleological/core.rs` | ✅ PASS |
| JohariQuadrant exists | `crates/context-graph-core/src/types/johari/quadrant.rs` | ✅ PASS |
| Serde available | `Cargo.toml` includes serde with derive | ✅ PASS |
| uuid available | `Cargo.toml` includes uuid with serde | ✅ PASS |
| Handler compiles | `cargo check -p context-graph-mcp` | ✅ PASS |
| Tests pass | `cargo test -p context-graph-mcp gwt_compute_delta_sc` | ✅ PASS |

---

## Scope

### In Scope (WHAT REMAINS)

1. **Extract inline types to proper Rust structs** (optional refactor):
   - `ComputeDeltaScRequest` struct
   - `ComputeDeltaScResponse` struct
   - `DeltaScDiagnostics` struct

2. **Implement ClusterFit component** for ΔC computation (constitution.yaml requires 0.4*ClusterFit)

3. **Implement specialized per-embedder ΔS calculators** per SPEC-UTL-001 §3.2

4. **Add Full State Verification tests** (see FSV section below)

### Out of Scope

- Handler skeleton (ALREADY EXISTS)
- Tool registration (ALREADY EXISTS)
- Basic tests (ALREADY EXIST)
- Parameter parsing (ALREADY EXISTS)

---

## Current Implementation Reference

### Handler Signature (ACTUAL)

```rust
// File: crates/context-graph-mcp/src/handlers/utl.rs (lines 1141-1145)

pub(super) async fn handle_gwt_compute_delta_sc(
    &self,
    id: Option<JsonRpcId>,
    params: Option<serde_json::Value>,
) -> JsonRpcResponse
```

### Response Structure (ACTUAL - line 1360-1367)

```json
{
    "delta_s_per_embedder": [f32; 13],
    "delta_s_aggregate": f32,
    "delta_c": f32,
    "johari_quadrants": ["Open"|"Blind"|"Hidden"|"Unknown"; 13],
    "johari_aggregate": "Open"|"Blind"|"Hidden"|"Unknown",
    "utl_learning_potential": f32,
    "diagnostics": { ... }  // if include_diagnostics=true
}
```

### Johari Classification (ACTUAL - lines 1390-1403)

```rust
fn classify_johari(delta_s: f32, delta_c: f32, threshold: f32) -> JohariQuadrant {
    match (delta_s < threshold, delta_c > threshold) {
        (true, true) => JohariQuadrant::Open,    // Low surprise, high coherence
        (false, false) => JohariQuadrant::Blind, // High surprise, low coherence
        (true, false) => JohariQuadrant::Hidden, // Low surprise, low coherence
        (false, true) => JohariQuadrant::Unknown, // High surprise, high coherence
    }
}
```

---

## Full State Verification (FSV) Requirements

### Source of Truth

| Data | Source | Verification Method |
|------|--------|---------------------|
| Input TeleologicalFingerprint | MCP request JSON | Deserialize with `serde_json::from_value` |
| Per-embedder ΔS values | Computed via EmbedderEntropyFactory | Verify against expected per SPEC-UTL-001 §3.2 |
| ΔC value | Computed via CoherenceTracker | Verify formula: 0.4×Connectivity + 0.4×ClusterFit + 0.2×Consistency |
| Johari quadrants | classify_johari() output | Verify against threshold logic |
| Response JSON | Handler return value | Parse and verify all fields present with valid ranges |

### Execute & Inspect Pattern

For each test:
1. **Setup**: Create known input fingerprints with predictable embedding values
2. **Execute**: Call `handle_gwt_compute_delta_sc()`
3. **Inspect**: Parse response JSON and verify:
   - All 13 per-embedder ΔS values in [0, 1]
   - delta_s_aggregate = mean(delta_s_per_embedder)
   - delta_c in [0, 1]
   - Each johari_quadrant matches classify_johari(delta_s[i], delta_c, threshold)
   - utl_learning_potential = delta_s_aggregate × delta_c
4. **Evidence**: Log actual values with expected values for comparison

### Boundary & Edge Case Audit

| Edge Case | Input | Expected Output | Verification |
|-----------|-------|-----------------|--------------|
| Identical fingerprints | old_fp == new_fp | delta_s_per_embedder ≈ [0; 13], johari = "Hidden" or "Open" | Test exists |
| Maximum difference | old_fp zeros, new_fp ones | delta_s_per_embedder ≈ [1; 13] | Add test |
| Empty embeddings | All embeddings empty/zeroed | Graceful handling, valid response | Add test |
| NaN in input | Fingerprint with NaN values | Clamped to valid range, warning logged | Test exists (implicit) |
| Inf in input | Fingerprint with Inf values | Clamped to valid range, warning logged | Add test |
| Johari threshold boundary | johari_threshold = 0.5 exactly | Consistent quadrant assignment | Test exists |
| Johari threshold min | johari_threshold = 0.35 | Clamped, valid quadrants | Add test |
| Johari threshold max | johari_threshold = 0.65 | Clamped, valid quadrants | Add test |
| Mismatched embedding types | Dense vs Sparse in same position | Handle gracefully with max surprise (1.0) | Implemented (line 1281-1286) |

### Evidence of Success

After each test execution, log:
```
[TEST] gwt/compute_delta_sc FSV
  Input: vertex_id={uuid}, old_fp.e1_semantic[0..3]={values}, new_fp.e1_semantic[0..3]={values}
  Output: delta_s_agg={value}, delta_c={value}, johari_agg={quadrant}
  Expected: delta_s_agg∈[0,1], delta_c∈[0,1], johari∈{Open,Blind,Hidden,Unknown}
  Status: PASS/FAIL
```

---

## Manual Testing Requirements

### Synthetic Test Data

#### Test Case 1: Basic Happy Path
```json
{
  "vertex_id": "550e8400-e29b-41d4-a716-446655440001",
  "old_fingerprint": {
    "semantic": {
      "e1_semantic": [0.5, 0.5, ..., 0.5],  // 1024 values
      "e2_syntactic": [0.5, 0.5, ..., 0.5], // 512 values
      // ... all 13 embedders with 0.5 values
    },
    "purpose": { /* default */ },
    "johari": { /* default */ },
    "content_hash": "0000...0000"
  },
  "new_fingerprint": {
    "semantic": {
      "e1_semantic": [0.7, 0.7, ..., 0.7],  // 1024 values
      "e2_syntactic": [0.7, 0.7, ..., 0.7], // 512 values
      // ... all 13 embedders with 0.7 values
    },
    "purpose": { /* default */ },
    "johari": { /* default */ },
    "content_hash": "1111...1111"
  },
  "include_diagnostics": true,
  "johari_threshold": 0.5
}
```

**Expected Output**:
- `delta_s_per_embedder`: Array of 13 floats, each in [0, 1]
- `delta_s_aggregate`: Float in [0, 1], should show moderate change
- `delta_c`: Float in [0, 1]
- `johari_quadrants`: Array of 13 strings from {Open, Blind, Hidden, Unknown}
- `johari_aggregate`: String from {Open, Blind, Hidden, Unknown}
- `utl_learning_potential`: Float in [0, 1]
- `diagnostics`: Object with per_embedder array, johari_threshold, coherence_config

#### Test Case 2: No Change (Identical Fingerprints)
```json
{
  "vertex_id": "550e8400-e29b-41d4-a716-446655440002",
  "old_fingerprint": { /* identical to new_fingerprint */ },
  "new_fingerprint": { /* identical to old_fingerprint */ },
  "include_diagnostics": false
}
```

**Expected Output**:
- `delta_s_per_embedder`: Close to [0, 0, ..., 0]
- `delta_s_aggregate`: Close to 0
- `johari_aggregate`: Likely "Hidden" (low ΔS, low ΔC) or "Open" (low ΔS, high ΔC)

#### Test Case 3: Maximum Change
```json
{
  "vertex_id": "550e8400-e29b-41d4-a716-446655440003",
  "old_fingerprint": { /* all zeros */ },
  "new_fingerprint": { /* all ones */ },
  "include_diagnostics": true
}
```

**Expected Output**:
- `delta_s_per_embedder`: Close to [1, 1, ..., 1]
- `delta_s_aggregate`: Close to 1
- `johari_aggregate`: Likely "Blind" or "Unknown" (high ΔS)

#### Test Case 4: Error - Missing Parameters
```json
{
  "vertex_id": "550e8400-e29b-41d4-a716-446655440004"
  // Missing old_fingerprint and new_fingerprint
}
```

**Expected Output**:
- Error response with code -32602 (INVALID_PARAMS)
- Error message: "Missing 'old_fingerprint' parameter"

#### Test Case 5: Error - Invalid UUID
```json
{
  "vertex_id": "not-a-valid-uuid",
  "old_fingerprint": { /* valid */ },
  "new_fingerprint": { /* valid */ }
}
```

**Expected Output**:
- Error response with code -32602 (INVALID_PARAMS)
- Error message containing "Invalid UUID format"

### Manual Testing Procedure

1. Start MCP server: `cargo run -p context-graph-mcp`
2. Send JSON-RPC request via stdin or TCP
3. Capture response
4. Verify response against expected output
5. Log results with timestamps

---

## Validation Criteria

| Criterion | Verification Method | Status |
|-----------|---------------------|--------|
| Tool discoverable via tools/list | `cargo test -p context-graph-mcp test_tools_list_includes_compute_delta_sc` | ✅ PASS |
| Valid request returns success | `cargo test -p context-graph-mcp test_gwt_compute_delta_sc_valid` | ✅ PASS |
| Per-embedder count = 13 | `cargo test -p context-graph-mcp test_gwt_compute_delta_sc_per_embedder_count` | ✅ PASS |
| AP-10 range compliance | `cargo test -p context-graph-mcp test_gwt_compute_delta_sc_ap10_range_compliance` | ✅ PASS |
| Johari quadrant values valid | `cargo test -p context-graph-mcp test_gwt_compute_delta_sc_johari_quadrant_values` | ✅ PASS |
| Diagnostics included when requested | `cargo test -p context-graph-mcp test_gwt_compute_delta_sc_diagnostics` | ✅ PASS |
| Missing params returns error | `cargo test -p context-graph-mcp test_gwt_compute_delta_sc_missing_params` | ✅ PASS |
| Invalid UUID returns error | `cargo test -p context-graph-mcp test_gwt_compute_delta_sc_invalid_uuid` | ✅ PASS |
| ClusterFit component implemented | Manual code review | ✅ PASS |
| Per-embedder specialized calculators | Code review against SPEC-UTL-001 §3.2 | ⚠️ PARTIAL (future work) |
| FSV tests implemented | `cargo test handlers::tests::utl` | ✅ PASS (32 tests) |
| Edge case tests implemented | `cargo test ec01 ec02 ec03 ec04 ec05 ec06` | ✅ PASS |
| Manual verification tests | `cargo test manual_delta_sc` | ✅ PASS (7 tests) |

---

## Test Commands

```bash
# Full test suite
cargo test -p context-graph-mcp -- --nocapture

# Specific gwt/compute_delta_sc tests
cargo test -p context-graph-mcp gwt_compute_delta_sc -- --nocapture

# UTL handler tests only
cargo test -p context-graph-mcp handlers::tests::utl -- --nocapture

# Clippy check
cargo clippy -p context-graph-mcp -- -D warnings

# Doc tests
cargo test -p context-graph-mcp --doc
```

---

## Recommendations

### Pros/Cons Analysis

#### Option A: Keep Current Inline Implementation
**Pros**:
- Already working and tested
- No migration risk
- Simpler codebase (fewer files)

**Cons**:
- Less type safety (JSON parsed at runtime)
- Harder to refactor response structure
- Documentation spread across handler code

#### Option B: Extract to Proper Type Structs
**Pros**:
- Better type safety with compile-time checks
- Easier to evolve API (change struct, compiler finds issues)
- Cleaner separation of concerns
- Better documentation via struct-level docs

**Cons**:
- Migration effort required
- Risk of breaking existing tests
- More files to maintain

#### Recommendation: Option A (Keep Current) for Now
The current implementation is functional and well-tested. Focus remaining effort on:
1. Implementing ClusterFit component (TASK-UTL-P1-002)
2. Implementing specialized per-embedder calculators (new sub-task)
3. Adding FSV edge case tests

---

## Root Cause Analysis Guidelines

If tests fail or errors occur:

1. **Check Logs First**: Look for `gwt/compute_delta_sc:` prefix in logs
2. **Parameter Parsing**: Verify JSON structure matches expected format
3. **Fingerprint Deserialization**: Check TeleologicalFingerprint::deserialize errors
4. **Entropy Computation**: Look for NaN/Inf warnings in logs
5. **Coherence Computation**: Check CoherenceTracker errors
6. **Range Violations**: Any value outside [0, 1] indicates clamping issue

### Common Failure Modes

| Symptom | Likely Cause | Resolution |
|---------|--------------|------------|
| "Missing parameters" error | JSON structure incorrect | Check request format matches tool schema |
| "Invalid UUID format" error | Malformed vertex_id | Ensure valid UUID v4 format |
| "Failed to parse fingerprint" | Fingerprint JSON malformed | Verify all 13 embedders present with correct structure |
| ΔS = 1.0 for all embedders | Embedding type mismatch | Check old/new fingerprints have matching types |
| ΔC always 0.5 | NaN/Inf in computation | Check input embeddings for invalid values |
| Johari all "Unknown" | High ΔS, high ΔC | Expected for large changes |

---

## Notes

- Handler at `utl.rs:1119-1437` is the source of truth for current behavior
- **ClusterFit IMPLEMENTED** (2026-01-11): Uses silhouette-based cluster fit with `create_divergent_cluster()`
- **ΔC Formula**: `ΔC = 0.4×Connectivity + 0.4×ClusterFit + 0.2×Consistency` (per constitution.yaml)
- Tests in `handlers/tests/utl.rs` provide comprehensive coverage (32 tests)
- Manual verification tests in `handlers/tests/manual_delta_sc_verification.rs` (7 tests)
- No separate types/ directory exists in context-graph-mcp (inline JSON used)
- AP-10 compliance enforced with `.clamp(0.0, 1.0)` and NaN/Inf checks

## Completion Summary (2026-01-11)

| Item | Status |
|------|--------|
| ClusterFit component | ✅ Implemented using silhouette coefficient |
| Three-component ΔC formula | ✅ ALPHA=0.4, BETA=0.4, GAMMA=0.2 |
| FSV tests | ✅ 5 formula verification tests |
| Edge case tests | ✅ 6 edge case tests (EC01-EC06) |
| Manual verification | ✅ 7 synthetic data tests |
| Code review | ✅ Simplified (removed redundant variables, renamed function) |
| All tests passing | ✅ 32 UTL tests + 7 manual tests |
