---
id: "M04-T14a"
title: "Add Result-Returning NT Weight Validation Wrapper"
description: |
  context-graph-core has validate() -> bool.
  This task adds a Graph-layer wrapper: validate_or_error() -> GraphResult<()>
  Uses existing GraphError::InvalidNtWeights for detailed error messages.
  NO BACKWARDS COMPATIBILITY - fail fast with robust error logging.
layer: "logic"
status: "complete"
priority: "high"
estimated_hours: 1
sequence: 20
depends_on:
  - "M04-T14"
spec_refs:
  - "TECH-GRAPH-004 Section 4.1"
  - "REQ-KG-065"
  - "constitution.yaml AP-001"
files_to_create:
  - path: "crates/context-graph-graph/src/marblestone/validation.rs"
    description: "NT weight validation wrapper with GraphError integration"
files_to_modify:
  - path: "crates/context-graph-graph/src/marblestone/mod.rs"
    description: "Add validation module"
test_file: "crates/context-graph-graph/tests/nt_validation_tests.rs"
completed: "2026-01-03"
commit: "a4dbcd9"
---

## ✅ TASK STATUS: COMPLETE

**Completed**: 2026-01-03
**Commit**: a4dbcd9
**Verified by**: sherlock-holmes subagent

### Implementation Summary

| Component | Location | Status |
|-----------|----------|--------|
| `validate_or_error()` | `context-graph-graph/src/marblestone/validation.rs` | ✅ IMPLEMENTED |
| `compute_effective_validated()` | `context-graph-graph/src/marblestone/validation.rs` | ✅ IMPLEMENTED |
| Module registration | `context-graph-graph/src/marblestone/mod.rs` | ✅ UPDATED |
| Validation tests | `crates/context-graph-graph/tests/nt_validation_tests.rs` | ✅ 24 TESTS PASS |

---

## Context

The core crate provides `validate() -> bool` for simple validation checks. The graph layer needs `GraphResult<()>` for proper error propagation through the `?` operator. This task created a validation wrapper that:
1. Uses existing `validate()` logic
2. Returns `GraphError::InvalidNtWeights` with field-specific details
3. Follows fail-fast principle (AP-001)

### Constitution Reference
- `AP-001`: Never unwrap() in prod - all errors properly typed
- `edge_model.nt_weights`: All weights in [0.0, 1.0]

---

## Implementation Reference

### validate_or_error() - Location: `crates/context-graph-graph/src/marblestone/validation.rs`

```rust
pub fn validate_or_error(weights: &NeurotransmitterWeights) -> GraphResult<()> {
    // Check excitatory field
    if weights.excitatory < 0.0
        || weights.excitatory > 1.0
        || weights.excitatory.is_nan()
        || weights.excitatory.is_infinite()
    {
        log::error!(
            "VALIDATION FAILED: excitatory={} (must be in [0.0, 1.0])",
            weights.excitatory
        );
        return Err(GraphError::InvalidNtWeights {
            field: "excitatory".to_string(),
            value: weights.excitatory,
        });
    }
    // Similar checks for inhibitory, modulatory...
    Ok(())
}
```

### compute_effective_validated() - Combined validation + computation

```rust
pub fn compute_effective_validated(
    weights: &NeurotransmitterWeights,
    base_weight: f32,
) -> GraphResult<f32> {
    validate_or_error(weights)?;
    Ok(weights.compute_effective_weight(base_weight))
}
```

### Module Registration - `crates/context-graph-graph/src/marblestone/mod.rs`

```rust
mod validation;
pub use validation::{compute_effective_validated, validate_or_error};
```

---

## Verification Evidence

### File Exists
```bash
$ ls -la crates/context-graph-graph/src/marblestone/
validation.rs  # 227 lines, ✅ EXISTS
mod.rs         # 60 lines, ✅ UPDATED
```

### Test Results
```bash
$ cargo test -p context-graph-graph nt_validation -- --nocapture
running 24 tests
test test_validate_real_domain_code ... ok
test test_validate_all_real_domains ... ok
test test_invalid_excitatory_returns_field_name ... ok
test test_compute_effective_validated_success ... ok
test test_nan_is_invalid ... ok
test test_infinity_is_invalid ... ok
test test_boundary_values_accepted ... ok
[...more tests...]
test result: ok. 24 passed; 0 failed
```

### Error Field Names Verified
```
InvalidNtWeights { field: "excitatory", value: 1.5 }  ✅
InvalidNtWeights { field: "inhibitory", value: -0.1 } ✅
InvalidNtWeights { field: "modulatory", value: NaN }  ✅
```

---

## Edge Cases Verified (With BEFORE/AFTER Logging)

### Edge Case 1: NaN Value
```
BEFORE: Testing NaN excitatory weight
AFTER: result = Err(InvalidNtWeights { field: "excitatory", value: NaN })
VERIFIED ✅
```

### Edge Case 2: Infinity Value
```
BEFORE: Testing +Infinity inhibitory weight
AFTER: result = Err(InvalidNtWeights { field: "inhibitory", value: inf })
VERIFIED ✅
```

### Edge Case 3: Boundary Values (0.0 and 1.0)
```
BEFORE: Testing boundary min (0.0, 0.0, 0.0)
AFTER: min boundary result = Ok(())
BEFORE: Testing boundary max (1.0, 1.0, 1.0)
AFTER: max boundary result = Ok(())
VERIFIED ✅
```

---

## Definition of Done (ALL MET)

### Signatures Verified
- [x] `validate_or_error(&valid_weights).is_ok()` returns true
- [x] `validate_or_error(&invalid_weights)` returns `Err(InvalidNtWeights)`
- [x] Error contains correct field name and value
- [x] NaN/Infinity treated as invalid
- [x] Boundary values (0.0, 1.0) accepted
- [x] Compiles with `cargo build`
- [x] Tests pass with `cargo test`
- [x] No clippy warnings

### Constraints Met
- [x] NO auto-clamping - fail fast on invalid weights
- [x] Check order: excitatory → inhibitory → modulatory
- [x] Handle NaN and Infinity as invalid
- [x] Use `log::error!` for failures, `log::debug!` for success

### Verification Commands (All Pass)
```bash
cargo build -p context-graph-graph  # ✅ PASS
cargo test -p context-graph-graph nt_validation -- --nocapture  # ✅ 24/24 PASS
cargo clippy -p context-graph-graph -- -D warnings  # ✅ NO WARNINGS
```

---

## Source of Truth Verification

### Files Created/Modified
```bash
$ cat crates/context-graph-graph/src/marblestone/validation.rs | wc -l
227  # ✅ EXISTS
```

### GraphError Variant Used
```bash
$ grep -n "InvalidNtWeights" crates/context-graph-graph/src/error.rs
148:#[error("Invalid NT weights: {field} = {value} (must be in [0.0, 1.0])")]
149:InvalidNtWeights { field: String, value: f32 },
```

---

## Related Tasks
- M04-T14: NT weight integration (COMPLETE - prerequisite)
- M04-T15: Integrate NT weights into GraphEdge (NEXT)
