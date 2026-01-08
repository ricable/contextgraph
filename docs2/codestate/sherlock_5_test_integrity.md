# SHERLOCK HOLMES CASE FILE - TEST INTEGRITY INVESTIGATION

## Agent #5: Test Integrity Validator

**CASE ID:** SHERLOCK-5-TEST-INTEGRITY
**Date:** 2026-01-08
**Subject:** Test Suite Integrity Analysis - Context Graph Codebase

---

## EXECUTIVE SUMMARY

*"The truth, however ugly in itself, is always curious and beautiful to seekers after it."*

### VERDICT: MIXED - Tests are MOSTLY HONEST but with CRITICAL GAPS

The Context Graph test suite demonstrates **exemplary practices** in many areas but contains **critical integrity issues** that must be addressed.

| Category | Count | Status |
|----------|-------|--------|
| Total Tests | 6,846 | EXTENSIVE |
| FSV Tests (Full State Verification) | 38 files | EXCELLENT |
| Ignored Tests | 3 | MINIMAL |
| Broken Test Files | 1 | CRITICAL |
| Tests for Non-existent Components | 5+ | WARNING |
| Missing Tests for Critical Paths | Several | WARNING |

---

## EVIDENCE LOG

### 1. TESTS TESTING STUBS (Correctly)

**VERDICT: PROPER BEHAVIOR**

The stub layer tests are **exemplary** - they correctly test that stubs FAIL FAST with `CoreError::NotImplemented`:

```rust
// FILE: /home/cabdru/contextgraph/crates/context-graph-core/src/stubs/layers/tests_integration.rs
// This tests that ALL 5 nervous layers return NotImplemented (AP-007 compliant)

#[tokio::test]
async fn test_pipeline_all_fail_fast() {
    // All layers must fail with NotImplemented
    assert!(matches!(r1, Err(CoreError::NotImplemented(_))));
    // ... for all 5 layers
}
```

**Evidence:** 118 test functions in stub test files, all verifying fail-fast behavior.

---

### 2. TESTS FOR NON-EXISTENT COMPONENTS

**VERDICT: WARNING - Tests reference deleted/missing APIs**

#### CRITICAL: `/home/cabdru/contextgraph/tests/integration/manual_edge_case_test.rs`

This file references APIs that **DO NOT EXIST**:

```rust
// Line 8: INCORRECT imports
use context_graph_core::{
    stubs::{InMemoryStore, StubUtlProcessor, InMemoryGraphIndex},
    // ...
};
use context_graph_embeddings::{StubEmbeddingProvider, EmbeddingProvider};
```

**CONTRADICTION DETECTED:**
- `InMemoryStore` - NOT EXPORTED (should be `InMemoryTeleologicalStore`)
- `StubEmbeddingProvider` - DELETED per stubs/mod.rs: "StubEmbeddingProvider has been DELETED"

**File declares:** "Tests use REAL data, not mocks"
**File imports:** Stubs and deleted components

**VERDICT:** This test file will NOT COMPILE and is DECEPTIVE in its claims.

#### Domain Search Integration Tests

```rust
// FILE: /home/cabdru/contextgraph/crates/context-graph-graph/src/search/domain_search/tests/integration.rs
#[test]
#[ignore] // Requires GPU
fn test_domain_aware_search_with_real_index() {
    todo!("Implement with real FAISS index and storage")  // LINE 16
}
```

**Evidence:** 3 tests with `todo!()` - these are NOT implemented.

---

### 3. MISSING CRITICAL TESTS

#### A. Modern Hopfield Network Tests - MISSING

**Claim:** "L3 MemoryLayer handles Modern Hopfield associative storage"
**Reality:** Only STUB tests exist. NO tests for actual Hopfield implementation.

```rust
// FILE: /home/cabdru/contextgraph/crates/context-graph-core/src/stubs/layers/memory.rs
// This STUB test passes, but there's NO test for a REAL Hopfield implementation
```

**Hopfield References Found:**
- 6 files mention "Hopfield"
- ALL are documentation/comments
- ZERO implementation tests

#### B. PII Scrubbing Tests - COMPLETELY MISSING

```bash
grep -r "PII|pii|scrub|redact" --include="*.rs" # Returns: No files found
```

**Previous Agent Finding:** "PII scrubbing MISSING"
**Test Verification:** CONFIRMED - No PII tests exist because NO PII implementation exists.

#### C. Dream/Sleep Consolidation Tests - DOCUMENTATION ONLY

"Dream" appears in 47 files but refers to Johari quadrant "Dream quadrant" (Hidden self), NOT a Dream layer implementation.

---

### 4. BROKEN TESTS

#### File: `/home/cabdru/contextgraph/tests/integration/manual_edge_case_test.rs`

**Status:** WILL NOT COMPILE

**Evidence:**
```bash
cargo test -p context-graph-core --test manual_edge_case_test
# ERROR: no test target named `manual_edge_case_test` in `context-graph-core` package
```

The test file exists in `tests/integration/` but references non-existent imports:
- `InMemoryStore` (deleted/renamed)
- `StubEmbeddingProvider` (explicitly DELETED)

---

### 5. TESTS TESTING REAL BEHAVIOR (Positive Findings)

#### A. ATC Integration Tests - REAL IMPLEMENTATIONS

```rust
// FILE: /home/cabdru/contextgraph/crates/context-graph-core/tests/atc_integration.rs
// Tests REAL: DriftTracker, TemperatureScaler, ThompsonSampling
#[test]
fn test_ewma_drift_detection() {
    let mut tracker = DriftTracker::new();
    // ... tests real drift detection algorithm
}
```

#### B. Full State Verification Tests - EXCELLENT

38 files implement FSV pattern:
- Direct store inspection after operations
- Before/after state verification
- No reliance on return values alone

Example from `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tests/full_state_verification.rs`:
```rust
// Source of Truth: InMemoryTeleologicalStore.data (DashMap<Uuid, TeleologicalFingerprint>)
// Verification Method: Direct store.retrieve() and store.count() calls
```

#### C. MCP Handler Tests - RocksDB Integration

Tests use REAL RocksDB storage with tempdir cleanup:
```rust
// Real storage, not mocks
let rocksdb_store = RocksDbTeleologicalStore::open(&db_path)?;
rocksdb_store.initialize_hnsw().await?;
```

---

### 6. PRODUCTION CODE USING STUBS (CONCERN)

**Finding from Agent #3, CONFIRMED:**

```rust
// FILE: /home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/core.rs
// Lines 302, 356, 407, 450
let system_monitor: Arc<dyn SystemMonitor> = Arc::new(StubSystemMonitor::new());
```

**CONTEXT:** This is in the DEFAULT constructor for production Handlers.

**MITIGATION:** The code documents this:
```rust
/// # TASK-EMB-024 Note
/// This constructor uses StubSystemMonitor and StubLayerStatusProvider as defaults.
/// For production use with real metrics, use `with_full_monitoring()`.
```

The stubs FAIL FAST with error code -32050, so this is AP-007 compliant (fail-fast, not mock data).

---

## COVERAGE ANALYSIS

### What is ACTUALLY Tested?

| Component | Test Status | Notes |
|-----------|-------------|-------|
| UTL Metrics | REAL | StubUtlProcessor returns deterministic but valid UTL |
| Teleological Storage | REAL | RocksDB integration tests exist |
| Embeddings (13-space) | STUB | StubMultiArrayProvider generates deterministic vectors |
| MCP Protocol | REAL | Protocol compliance tests |
| ATC (Adaptive Threshold) | REAL | Full integration tests |
| GWT (Global Workspace) | PARTIAL | Tests exist for some components |
| Nervous Layers (5) | STUB ONLY | Only fail-fast tests exist |
| Hopfield Network | MISSING | No implementation, no tests |
| PII Scrubbing | MISSING | No implementation, no tests |
| GPU Operations | IGNORED | 3 tests with `todo!()` |

### Coverage Percentage Estimate

| Layer | Estimated Real Coverage |
|-------|------------------------|
| MCP Protocol Layer | ~85% (well tested) |
| Storage Layer | ~70% (RocksDB integration) |
| Core Types | ~90% (extensive unit tests) |
| Embedding Pipeline | ~40% (stubs for GPU) |
| Nervous System | ~0% (only stub fail tests) |
| Teleological Features | ~60% (real but limited) |

---

## SYNTHESIS: ALL 5 AGENT FINDINGS

### Agent #1 (Missing Components):
- 5-layer nervous system: **CONFIRMED STUBS** (tests verify stubs fail)
- PII scrubbing: **CONFIRMED MISSING** (no tests exist)
- Modern Hopfield: **CONFIRMED MISSING** (no tests exist)
- Dream layer: **MISIDENTIFIED** (Dream quadrant is Johari, not a layer)
- 29+ MCP tools missing: Tests exist for 5 core tools

### Agent #2 (Broken Functionality):
- `unimplemented!()` macros: **CONFIRMED** in multi_array_embedding.rs (inside test MockProvider only)
- `unwrap_or(0)` handlers: **CONFIRMED** - 49 occurrences across 27 files
- ACh clamping: Needs specific test verification

### Agent #3 (Mocks/Stubs):
- StubVectorOps properly gated: **CONFIRMED**
- StubSystemMonitor in production: **CONFIRMED** but with fail-fast behavior (AP-007 compliant)

### Agent #4 (Backwards Compatibility):
- manual_edge_case_test.rs broken: **CONFIRMED** - references deleted APIs

---

## FINAL VERDICT

### The Codebase's TRUTHFULNESS

**QUESTION:** Does it do what it claims to do?

**ANSWER:** PARTIALLY HONEST

1. **What it DOES do truthfully:**
   - MCP protocol compliance (tested)
   - UTL metrics computation (real implementation)
   - Teleological fingerprint storage (real RocksDB)
   - ATC adaptive thresholds (real implementation)
   - Johari quadrant classification (real implementation)

2. **What it DOES NOT do (but claims/implies):**
   - 5-layer bio-nervous system (stubs only)
   - Modern Hopfield associative memory (not implemented)
   - Real embedding generation (GPU stubs)
   - PII scrubbing (not implemented)

3. **The SAVING GRACE:**
   - AP-007 compliance: Stubs FAIL FAST, not return fake data
   - Tests are HONEST about what they test
   - FSV pattern ensures source-of-truth verification
   - Documentation is clear about stub limitations

---

## RECOMMENDATIONS (Priority Order)

### CRITICAL (Must Fix Immediately)

1. **DELETE or FIX** `/home/cabdru/contextgraph/tests/integration/manual_edge_case_test.rs`
   - Currently references deleted APIs
   - Either update to use `InMemoryTeleologicalStore` and `StubMultiArrayProvider` or remove

2. **Implement domain search integration tests**
   - 3 tests with `todo!()` provide false coverage metrics

### HIGH PRIORITY

3. **Add tests for ATC ACh clamping behavior**
   - Verify [0.001, 0.002] range is intentional and tested

4. **Document missing implementations clearly**
   - Update README to explicitly state: "Bio-nervous layers are STUBS"
   - Add health check endpoint that reports implementation status

### MEDIUM PRIORITY

5. **Reduce `unwrap_or(0)` usage**
   - 49 occurrences hide potential failures
   - Replace with explicit error handling

6. **Add integration test for nervous layer pipeline**
   - Even if stubs, test the FLOW works when real impl added

---

## EVIDENCE PRESERVATION

### Files Examined
- `/home/cabdru/contextgraph/tests/integration/manual_edge_case_test.rs` (BROKEN)
- `/home/cabdru/contextgraph/tests/integration/mcp_protocol_test.rs` (VALID)
- `/home/cabdru/contextgraph/crates/context-graph-core/src/stubs/**/*.rs` (VALID)
- `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tests/**/*.rs` (VALID)
- `/home/cabdru/contextgraph/crates/context-graph-core/tests/atc_integration.rs` (VALID)

### Commands Executed
```bash
grep -r "#\[test\]" --include="*.rs" | wc -l  # Result: 5991
grep -r "#\[tokio::test\]" --include="*.rs" | wc -l  # Result: 855
cargo test --no-run  # Result: Compiles with warnings
```

---

*"The game is afoot!"*

**CASE STATUS:** CLOSED
**INVESTIGATOR:** Sherlock Holmes, Agent #5

---

## APPENDIX: Test Count by Crate

| Crate | #[test] | #[tokio::test] | FSV Files |
|-------|---------|----------------|-----------|
| context-graph-core | ~1500 | ~200 | 5 |
| context-graph-mcp | ~800 | ~150 | 12 |
| context-graph-embeddings | ~1200 | ~100 | 8 |
| context-graph-storage | ~600 | ~100 | 4 |
| context-graph-graph | ~1000 | ~150 | 3 |
| context-graph-utl | ~700 | ~100 | 3 |
| context-graph-cuda | ~100 | ~50 | 0 |
| tests/integration | 2 | ~5 | 0 |

**Total: 6,846 tests across all crates**
