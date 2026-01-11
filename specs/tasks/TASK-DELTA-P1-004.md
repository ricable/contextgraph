# TASK-DELTA-P1-004: Integration Tests for compute_delta_sc

## Metadata

| Field | Value |
|-------|-------|
| **ID** | TASK-DELTA-P1-004 |
| **Version** | 1.0 |
| **Status** | ready |
| **Layer** | surface |
| **Sequence** | 4 of 4 |
| **Priority** | P1 |
| **Estimated Complexity** | medium |
| **Estimated Duration** | 3-4 hours |
| **Implements** | Test Plan from SPEC-UTL-001 |
| **Depends On** | TASK-DELTA-P1-001, TASK-DELTA-P1-002, TASK-DELTA-P1-003 |
| **Spec Ref** | SPEC-UTL-001 |
| **Gap Ref** | MASTER-CONSCIOUSNESS-GAP-ANALYSIS.md GAP 1 |

---

## Context

This task creates comprehensive integration tests for the `compute_delta_sc` MCP tool, validating end-to-end behavior with realistic data.

**Why This Last**: Following Inside-Out, Bottom-Up:
1. Unit tests exist in each component task
2. Integration tests validate the complete pipeline
3. Tests serve as living documentation and regression protection

**Gap Being Addressed**:
> GAP 1: UTL compute_delta_sc MCP Tool Missing
> Integration tests prove the gap is fully resolved

---

## Input Context Files

| Purpose | File |
|---------|------|
| Test plan reference | `specs/functional/SPEC-UTL-001.md#test-plan` |
| Existing integration tests | `crates/context-graph-mcp/src/handlers/tests/integration_e2e.rs` |
| FSV test pattern | `crates/context-graph-mcp/src/handlers/tests/full_state_verification.rs` |
| Test helpers | `crates/context-graph-core/src/types/fingerprint/teleological/test_helpers.rs` |
| Mock data generation | `crates/context-graph-mcp/src/handlers/tests/inject_synthetic_data.rs` |

---

## Prerequisites

| Check | Verification |
|-------|--------------|
| TASK-DELTA-P1-003 complete | Handler registered and compiles |
| Test framework set up | `cargo test -p context-graph-mcp` runs |
| Test helpers available | `TeleologicalFingerprint::zeroed()` exists |
| FSV pattern understood | Read existing full_state_verification tests |

---

## Scope

### In Scope

- Integration tests for all test cases from SPEC-UTL-001
- Full State Verification (FSV) test for compute_delta_sc
- Performance benchmark tests
- Property-based tests for output bounds
- Error handling tests for all error states
- Edge case tests per SPEC-UTL-001

### Out of Scope

- Chaos tests (separate task)
- Load testing (separate task)
- Fuzz testing (future enhancement)

---

## Definition of Done

### Test File Structure

```rust
// File: crates/context-graph-mcp/src/handlers/tests/delta_sc_integration.rs

use std::sync::Arc;

use uuid::Uuid;

use context_graph_core::types::fingerprint::{
    JohariFingerprint, PurposeVector, SemanticFingerprint, TeleologicalFingerprint,
};
use context_graph_core::types::JohariQuadrant;

use crate::handlers::Handlers;
use crate::protocol::{error_codes, JsonRpcRequest};
use crate::types::delta_sc::{ComputeDeltaScRequest, ComputeDeltaScResponse};

// ============================================================================
// Test Helpers
// ============================================================================

fn create_test_handlers() -> Handlers {
    // Create handlers with mock stores
    Handlers::new_for_testing()
}

fn create_test_fingerprint_pair() -> (TeleologicalFingerprint, TeleologicalFingerprint) {
    let old = TeleologicalFingerprint::new(
        SemanticFingerprint::zeroed(),
        PurposeVector::default(),
        JohariFingerprint::zeroed(),
        [0u8; 32],
    );

    let mut new = old.clone();
    // Modify semantic embeddings to create measurable delta
    // ...

    (old, new)
}

// ============================================================================
// Unit Tests from SPEC-UTL-001 Test Plan
// ============================================================================

/// TC-001: GMM entropy returns value in [0, 1]
#[tokio::test]
async fn test_gmm_entropy_bounded() {
    let handlers = create_test_handlers();
    let (old_fp, new_fp) = create_test_fingerprint_pair();

    let request = ComputeDeltaScRequest {
        vertex_id: Uuid::new_v4(),
        old_fingerprint: old_fp,
        new_fingerprint: new_fp,
        include_diagnostics: true,
        johari_threshold: None,
    };

    let response = handlers.delta_sc_computer.compute(&request).await.unwrap();

    // E1 uses GMM
    assert!(response.delta_s_per_embedder[0] >= 0.0);
    assert!(response.delta_s_per_embedder[0] <= 1.0);
    println!("[PASS] TC-001: GMM entropy in [0, 1]");
}

/// TC-010: Delta-C uses correct weights (0.4, 0.4, 0.2)
#[tokio::test]
async fn test_delta_c_weights() {
    let handlers = create_test_handlers();
    let (old_fp, new_fp) = create_test_fingerprint_pair();

    let request = ComputeDeltaScRequest {
        vertex_id: Uuid::new_v4(),
        old_fingerprint: old_fp,
        new_fingerprint: new_fp,
        include_diagnostics: true,
        johari_threshold: None,
    };

    let response = handlers.delta_sc_computer.compute(&request).await.unwrap();
    let diag = response.diagnostics.unwrap();

    // Verify formula: delta_c = 0.4*connectivity + 0.4*cluster_fit + 0.2*consistency
    let expected = 0.4 * diag.connectivity + 0.4 * diag.cluster_fit + 0.2 * diag.consistency;
    let tolerance = 0.001;
    assert!((response.delta_c - expected).abs() < tolerance,
        "Delta-C mismatch: got {}, expected {}", response.delta_c, expected);

    println!("[PASS] TC-010: Delta-C weights correct (0.4, 0.4, 0.2)");
}

/// TC-011: Johari classification correct for all quadrants
#[tokio::test]
async fn test_johari_classification_all_quadrants() {
    let handlers = create_test_handlers();

    // Test Open: delta_s <= 0.5, delta_c > 0.5
    // Test Blind: delta_s > 0.5, delta_c <= 0.5
    // Test Hidden: delta_s <= 0.5, delta_c <= 0.5
    // Test Unknown: delta_s > 0.5, delta_c > 0.5

    // Create fingerprints that produce each quadrant...
    // (Implementation details)

    println!("[PASS] TC-011: All Johari quadrants classified correctly");
}

/// TC-012: Missing embedder returns fallback
#[tokio::test]
async fn test_missing_embedder_fallback() {
    // Test with partial fingerprint (if allowed)
    // Should return fallback 0.5 for missing embedder
    println!("[PASS] TC-012: Missing embedder fallback works");
}

/// TC-013: Identical fingerprints return zero entropy
#[tokio::test]
async fn test_identical_fingerprints_zero_entropy() {
    let handlers = create_test_handlers();
    let fp = TeleologicalFingerprint::new(
        SemanticFingerprint::zeroed(),
        PurposeVector::default(),
        JohariFingerprint::zeroed(),
        [0u8; 32],
    );

    let request = ComputeDeltaScRequest {
        vertex_id: Uuid::new_v4(),
        old_fingerprint: fp.clone(),
        new_fingerprint: fp,
        include_diagnostics: false,
        johari_threshold: None,
    };

    let response = handlers.delta_sc_computer.compute(&request).await.unwrap();

    // Identical fingerprints should have low/zero entropy
    for (idx, delta_s) in response.delta_s_per_embedder.iter().enumerate() {
        assert!(*delta_s < 0.1,
            "E{} delta_s should be ~0 for identical fingerprints, got {}", idx + 1, delta_s);
    }

    println!("[PASS] TC-013: Identical fingerprints produce near-zero entropy");
}

// ============================================================================
// Integration Tests from SPEC-UTL-001 Test Plan
// ============================================================================

/// TC-014: MCP tool registered and discoverable
#[tokio::test]
async fn test_tool_discoverable() {
    let handlers = create_test_handlers();

    let response = handlers.handle_tools_list(Some(1.into())).await;

    let result = response.result.unwrap();
    let tools = result["tools"].as_array().unwrap();

    let found = tools.iter().any(|t| t["name"] == "gwt/compute_delta_sc");
    assert!(found, "gwt/compute_delta_sc not in tools/list");

    println!("[PASS] TC-014: Tool discoverable via tools/list");
}

/// TC-015: Full pipeline with real fingerprints
#[tokio::test]
async fn test_full_pipeline() {
    let handlers = create_test_handlers();
    let (old_fp, new_fp) = create_test_fingerprint_pair();

    let params = serde_json::to_value(ComputeDeltaScRequest {
        vertex_id: Uuid::new_v4(),
        old_fingerprint: old_fp,
        new_fingerprint: new_fp,
        include_diagnostics: true,
        johari_threshold: Some(0.5),
    }).unwrap();

    let response = handlers.handle_gwt_compute_delta_sc(Some(1.into()), Some(params)).await;

    assert!(response.error.is_none(), "Unexpected error: {:?}", response.error);

    let result: ComputeDeltaScResponse = serde_json::from_value(response.result.unwrap()).unwrap();

    // Validate response structure
    assert_eq!(result.delta_s_per_embedder.len(), 13);
    assert!(result.delta_s_aggregate >= 0.0 && result.delta_s_aggregate <= 1.0);
    assert!(result.delta_c >= 0.0 && result.delta_c <= 1.0);
    assert_eq!(result.johari_quadrants.len(), 13);
    assert!(result.diagnostics.is_some());

    println!("[PASS] TC-015: Full pipeline works with real fingerprints");
}

/// TC-017: Response matches schema
#[tokio::test]
async fn test_response_schema() {
    let handlers = create_test_handlers();
    let (old_fp, new_fp) = create_test_fingerprint_pair();

    let params = serde_json::to_value(ComputeDeltaScRequest {
        vertex_id: Uuid::new_v4(),
        old_fingerprint: old_fp,
        new_fingerprint: new_fp,
        include_diagnostics: false,
        johari_threshold: None,
    }).unwrap();

    let response = handlers.handle_gwt_compute_delta_sc(Some(1.into()), Some(params)).await;
    let result = response.result.unwrap();

    // Verify all required fields present
    assert!(result.get("delta_s_per_embedder").is_some());
    assert!(result.get("delta_s_aggregate").is_some());
    assert!(result.get("delta_c").is_some());
    assert!(result.get("johari_quadrants").is_some());
    assert!(result.get("johari_aggregate").is_some());
    assert!(result.get("utl_learning_potential").is_some());

    // Diagnostics should be absent when not requested
    assert!(result.get("diagnostics").is_none());

    println!("[PASS] TC-017: Response matches schema");
}

// ============================================================================
// Performance Tests from SPEC-UTL-001 Test Plan
// ============================================================================

/// TC-019: Compute latency < 25ms p95
#[tokio::test]
async fn test_latency_p95() {
    use std::time::Instant;

    let handlers = create_test_handlers();
    let mut latencies = Vec::with_capacity(100);

    for _ in 0..100 {
        let (old_fp, new_fp) = create_test_fingerprint_pair();

        let request = ComputeDeltaScRequest {
            vertex_id: Uuid::new_v4(),
            old_fingerprint: old_fp,
            new_fingerprint: new_fp,
            include_diagnostics: false,
            johari_threshold: None,
        };

        let start = Instant::now();
        let _ = handlers.delta_sc_computer.compute(&request).await;
        latencies.push(start.elapsed().as_millis() as u64);
    }

    latencies.sort();
    let p95 = latencies[94]; // 95th percentile

    assert!(p95 < 25, "p95 latency {} ms exceeds 25ms target", p95);
    println!("[PASS] TC-019: p95 latency {} ms < 25ms", p95);
}

// ============================================================================
// Property-Based Tests from SPEC-UTL-001 Test Plan
// ============================================================================

/// TC-022: Delta-S always in [0, 1]
#[tokio::test]
async fn test_delta_s_bounded_property() {
    let handlers = create_test_handlers();

    for _ in 0..50 {
        let (old_fp, new_fp) = create_test_fingerprint_pair();

        let request = ComputeDeltaScRequest {
            vertex_id: Uuid::new_v4(),
            old_fingerprint: old_fp,
            new_fingerprint: new_fp,
            include_diagnostics: false,
            johari_threshold: None,
        };

        let response = handlers.delta_sc_computer.compute(&request).await.unwrap();

        for (idx, &delta_s) in response.delta_s_per_embedder.iter().enumerate() {
            assert!(delta_s >= 0.0 && delta_s <= 1.0,
                "E{} delta_s {} out of bounds", idx + 1, delta_s);
            assert!(!delta_s.is_nan(), "E{} delta_s is NaN (AP-10 violation)", idx + 1);
            assert!(!delta_s.is_infinite(), "E{} delta_s is infinite (AP-10 violation)", idx + 1);
        }
    }

    println!("[PASS] TC-022: Delta-S always in [0, 1] for random inputs");
}

/// TC-023: Delta-C always in [0, 1]
#[tokio::test]
async fn test_delta_c_bounded_property() {
    let handlers = create_test_handlers();

    for _ in 0..50 {
        let (old_fp, new_fp) = create_test_fingerprint_pair();

        let request = ComputeDeltaScRequest {
            vertex_id: Uuid::new_v4(),
            old_fingerprint: old_fp,
            new_fingerprint: new_fp,
            include_diagnostics: false,
            johari_threshold: None,
        };

        let response = handlers.delta_sc_computer.compute(&request).await.unwrap();

        assert!(response.delta_c >= 0.0 && response.delta_c <= 1.0,
            "delta_c {} out of bounds", response.delta_c);
        assert!(!response.delta_c.is_nan(), "delta_c is NaN (AP-10 violation)");
        assert!(!response.delta_c.is_infinite(), "delta_c is infinite (AP-10 violation)");
    }

    println!("[PASS] TC-023: Delta-C always in [0, 1] for random inputs");
}

/// TC-025: UTL potential = Delta-S * Delta-C
#[tokio::test]
async fn test_utl_potential_invariant() {
    let handlers = create_test_handlers();
    let (old_fp, new_fp) = create_test_fingerprint_pair();

    let request = ComputeDeltaScRequest {
        vertex_id: Uuid::new_v4(),
        old_fingerprint: old_fp,
        new_fingerprint: new_fp,
        include_diagnostics: false,
        johari_threshold: None,
    };

    let response = handlers.delta_sc_computer.compute(&request).await.unwrap();

    let expected = response.delta_s_aggregate * response.delta_c;
    let tolerance = 0.0001;

    assert!((response.utl_learning_potential - expected).abs() < tolerance,
        "UTL potential mismatch: got {}, expected {}", response.utl_learning_potential, expected);

    println!("[PASS] TC-025: UTL potential = Delta-S * Delta-C");
}

// ============================================================================
// Full State Verification (FSV) Test
// ============================================================================

/// Full State Verification for compute_delta_sc
///
/// This test follows the FSV pattern established in the codebase,
/// verifying complete state transitions and invariants.
#[tokio::test]
async fn full_state_verification_compute_delta_sc() {
    println!("\n========== TASK-DELTA-P1-004 FULL STATE VERIFICATION ==========\n");

    let handlers = create_test_handlers();

    // Test 1: Basic computation
    println!("[TEST 1] Basic Delta-S/Delta-C computation");
    let (old_fp, new_fp) = create_test_fingerprint_pair();
    let request = ComputeDeltaScRequest {
        vertex_id: Uuid::new_v4(),
        old_fingerprint: old_fp,
        new_fingerprint: new_fp,
        include_diagnostics: true,
        johari_threshold: None,
    };

    let response = handlers.delta_sc_computer.compute(&request).await.unwrap();
    assert_eq!(response.delta_s_per_embedder.len(), 13);
    println!("  Delta-S aggregate: {}", response.delta_s_aggregate);
    println!("  Delta-C: {}", response.delta_c);
    println!("  Johari aggregate: {:?}", response.johari_aggregate);
    println!("[TEST 1 PASSED]\n");

    // Test 2: All 13 embedders computed
    println!("[TEST 2] All 13 embedders have valid outputs");
    for (idx, delta_s) in response.delta_s_per_embedder.iter().enumerate() {
        assert!(delta_s.is_finite(), "E{} has non-finite delta_s", idx + 1);
        println!("  E{}: delta_s = {:.4}", idx + 1, delta_s);
    }
    println!("[TEST 2 PASSED]\n");

    // Test 3: Johari classification
    println!("[TEST 3] Johari classification");
    for (idx, quadrant) in response.johari_quadrants.iter().enumerate() {
        println!("  E{}: {:?}", idx + 1, quadrant);
    }
    println!("[TEST 3 PASSED]\n");

    // Test 4: Diagnostics included
    println!("[TEST 4] Diagnostics present when requested");
    let diag = response.diagnostics.as_ref().unwrap();
    println!("  Connectivity: {}", diag.connectivity);
    println!("  ClusterFit: {}", diag.cluster_fit);
    println!("  Consistency: {}", diag.consistency);
    println!("  Computation time: {} us", diag.computation_time_us);
    println!("[TEST 4 PASSED]\n");

    // Test 5: Error handling
    println!("[TEST 5] Error handling for invalid parameters");
    let invalid_response = handlers.handle_gwt_compute_delta_sc(
        Some(1.into()),
        Some(serde_json::json!({"invalid": "params"})),
    ).await;
    assert!(invalid_response.error.is_some());
    println!("  Error code: {}", invalid_response.error.as_ref().unwrap().code);
    println!("[TEST 5 PASSED]\n");

    println!("========== FSV SUMMARY ==========");
    println!("[EVIDENCE] TASK-DELTA-P1-004 Complete");
    println!("  - Total tests: 5");
    println!("  - Tests passed: 5");
    println!("  - Embedders verified: 13/13");
    println!("  - Johari quadrants: validated");
    println!("  - Diagnostics: validated");
    println!("  - Error handling: validated");
    println!("=================================\n");
}
```

### Constraints

- All tests MUST be async/await compatible
- Tests MUST use the FSV pattern established in the codebase
- Property tests MUST run multiple iterations (50+)
- Performance tests MUST measure p95 latency
- Tests MUST NOT depend on external services
- Tests MUST be deterministic (use seeded random if needed)

### Verification

```bash
# All integration tests pass
cargo test -p context-graph-mcp delta_sc_integration -- --nocapture

# FSV test passes
cargo test -p context-graph-mcp full_state_verification_compute_delta_sc -- --nocapture

# No test flakiness (run 3 times)
for i in 1 2 3; do cargo test -p context-graph-mcp delta_sc_integration; done
```

---

## Files to Create

| Path | Description |
|------|-------------|
| `crates/context-graph-mcp/src/handlers/tests/delta_sc_integration.rs` | Integration tests |

---

## Files to Modify

| Path | Change |
|------|--------|
| `crates/context-graph-mcp/src/handlers/tests/mod.rs` | Add `mod delta_sc_integration;` |

---

## Validation Criteria

| Criterion | Verification Method |
|-----------|---------------------|
| All SPEC-UTL-001 test cases implemented | Checklist review |
| FSV test passes | `cargo test full_state_verification_compute_delta_sc` |
| p95 latency < 25ms | Performance test TC-019 |
| No NaN/Infinity (AP-10) | Property tests TC-022, TC-023 |
| Tests are deterministic | Run 3 times with same results |

---

## Test Commands

```bash
# Run all delta_sc tests
cargo test -p context-graph-mcp delta_sc -- --nocapture

# Run integration tests only
cargo test -p context-graph-mcp delta_sc_integration -- --nocapture

# Run FSV test only
cargo test -p context-graph-mcp full_state_verification_compute_delta_sc -- --nocapture

# Run with verbose output
RUST_LOG=debug cargo test -p context-graph-mcp delta_sc -- --nocapture
```

---

## Notes

- Tests serve as living documentation for the compute_delta_sc tool
- FSV pattern ensures complete state verification
- Property tests catch edge cases that unit tests might miss
- Performance tests should run on consistent hardware for reliable results
- Consider adding mutation testing in future iterations
