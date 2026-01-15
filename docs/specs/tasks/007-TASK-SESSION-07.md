# TASK-SESSION-07: Implement classify_ic() Function

```xml
<task_spec id="TASK-SESSION-07" version="2.0">
<metadata>
  <title>Implement classify_ic() Function</title>
  <status>completed</status>
  <layer>logic</layer>
  <sequence>7</sequence>
  <implements>
    <requirement_ref>REQ-SESSION-07</requirement_ref>
    <constitution_ref>IDENTITY-002</constitution_ref>
  </implements>
  <depends_on><!-- None --></depends_on>
  <estimated_hours>0.5</estimated_hours>
  <last_audit>2026-01-15</last_audit>
</metadata>
```

## Objective

Implement IC classification function with IDENTITY-002 constitution thresholds:
- **Healthy**: IC >= 0.9
- **Good**: 0.7 <= IC < 0.9
- **Warning**: 0.5 <= IC < 0.7
- **Degraded**: IC < 0.5

## Current Codebase State (Verified 2026-01-15)

### Existing Infrastructure

1. **Threshold Constants Already Exist** in `crates/context-graph-core/src/gwt/ego_node/types.rs`:
   ```rust
   pub const IC_CRITICAL_THRESHOLD: f32 = 0.5;  // line 23
   pub const IC_WARNING_THRESHOLD: f32 = 0.7;   // line 28
   pub const IC_HEALTHY_THRESHOLD: f32 = 0.9;   // line 33
   ```

2. **IdentityStatus Enum Exists** in same file (types.rs:77-86):
   ```rust
   pub enum IdentityStatus {
       Healthy,   // IC >= 0.9
       Warning,   // 0.7 <= IC < 0.9
       Degraded,  // 0.5 <= IC < 0.7
       Critical,  // IC < 0.5
   }
   ```

3. **Target File**: `crates/context-graph-core/src/gwt/session_identity/manager.rs`
   - Currently exports: `compute_ic`, `compute_kuramoto_r`, `SessionIdentityManager` trait
   - Uses `cosine_similarity_13d` from `crate::gwt` (already public per AP-39)

4. **Module Export Path**: `crates/context-graph-core/src/gwt/session_identity/mod.rs`
   - Must add exports for new functions

### Task Scope Clarification

**The task requires adding NEW classification functions**, not duplicating existing constants:
- `classify_ic(ic: f32) -> &'static str` - Returns string label
- `is_ic_crisis(ic: f32) -> bool` - Quick check for IC < 0.5
- `is_ic_warning(ic: f32) -> bool` - Quick check for 0.5 <= IC < 0.7
- `classify_sync(r: f64) -> &'static str` - Kuramoto r classification

**These functions USE the existing constants, they don't replace them.**

## Implementation Steps

### Step 1: Add classify_ic() to manager.rs

Location: `crates/context-graph-core/src/gwt/session_identity/manager.rs`

Add after line 141 (after the `compute_ic` function):

```rust
// =============================================================================
// IDENTITY-002 Classification Functions
// =============================================================================

/// Classify IC value per IDENTITY-002 constitution thresholds.
///
/// # Thresholds (IDENTITY-002)
/// - Healthy: IC >= 0.9
/// - Good: 0.7 <= IC < 0.9
/// - Warning: 0.5 <= IC < 0.7
/// - Degraded: IC < 0.5
///
/// # Returns
/// Static string label (no allocation)
#[inline]
pub fn classify_ic(ic: f32) -> &'static str {
    match ic {
        ic if ic >= 0.9 => "Healthy",
        ic if ic >= 0.7 => "Good",
        ic if ic >= 0.5 => "Warning",
        _ => "Degraded",
    }
}

/// Returns true if IC indicates identity crisis (< 0.5).
///
/// Per AP-26 and AP-38: IC < 0.5 MUST trigger automatic dream consolidation.
///
/// # Constitution Reference
/// - AP-26: IC<0.5 MUST trigger dream - no silent failures
/// - AP-38: IC<0.5 MUST auto-trigger dream
#[inline]
pub fn is_ic_crisis(ic: f32) -> bool {
    ic < 0.5
}

/// Returns true if IC is in warning range (0.5 <= IC < 0.7).
///
/// Per IDENTITY-002: This range indicates degraded identity but not crisis.
#[inline]
pub fn is_ic_warning(ic: f32) -> bool {
    ic >= 0.5 && ic < 0.7
}

/// Classify Kuramoto order parameter r.
///
/// # Thresholds (gwt.kuramoto.thresholds in constitution)
/// - Good synchronization: r >= 0.8
/// - Partial synchronization: 0.5 <= r < 0.8
/// - Fragmented: r < 0.5
///
/// # Returns
/// Static string description
#[inline]
pub fn classify_sync(r: f64) -> &'static str {
    match r {
        r if r >= 0.8 => "Good synchronization",
        r if r >= 0.5 => "Partial synchronization",
        _ => "Fragmented",
    }
}
```

### Step 2: Add Unit Tests

Add to `manager.rs` after existing tests (inside `#[cfg(test)] mod tests`):

```rust
// =========================================================================
// TC-SESSION-09: classify_ic() Threshold Boundaries
// =========================================================================
#[test]
fn tc_session_09_classify_ic_threshold_boundaries() {
    println!("\n=== TC-SESSION-09: classify_ic() Threshold Boundaries ===");

    // SOURCE OF TRUTH: IDENTITY-002 from constitution.yaml
    // Healthy: IC >= 0.9
    // Good: 0.7 <= IC < 0.9
    // Warning: 0.5 <= IC < 0.7
    // Degraded: IC < 0.5

    let test_cases: Vec<(f32, &str)> = vec![
        // Healthy boundary
        (1.0, "Healthy"),
        (0.95, "Healthy"),
        (0.90, "Healthy"),
        // Good boundary
        (0.899, "Good"),
        (0.75, "Good"),
        (0.70, "Good"),
        // Warning boundary
        (0.699, "Warning"),
        (0.55, "Warning"),
        (0.50, "Warning"),
        // Degraded boundary
        (0.499, "Degraded"),
        (0.25, "Degraded"),
        (0.0, "Degraded"),
        // Edge case: negative (clamped)
        (-0.1, "Degraded"),
    ];

    for (ic, expected) in test_cases {
        let actual = classify_ic(ic);
        println!("classify_ic({:.3}) = '{}' (expected: '{}')", ic, actual, expected);
        assert_eq!(
            actual, expected,
            "classify_ic({}) should return '{}', got '{}'",
            ic, expected, actual
        );
    }

    println!("RESULT: PASS - All threshold boundaries correct");
}

#[test]
fn tc_session_09_is_ic_crisis() {
    println!("\n=== TC-SESSION-09: is_ic_crisis() ===");

    // SOURCE OF TRUTH: AP-26, AP-38 - crisis is IC < 0.5

    // Crisis cases (IC < 0.5)
    assert!(is_ic_crisis(0.0), "0.0 should be crisis");
    assert!(is_ic_crisis(0.25), "0.25 should be crisis");
    assert!(is_ic_crisis(0.49), "0.49 should be crisis");
    assert!(is_ic_crisis(0.499), "0.499 should be crisis");

    // Non-crisis cases (IC >= 0.5)
    assert!(!is_ic_crisis(0.5), "0.5 should NOT be crisis");
    assert!(!is_ic_crisis(0.50), "0.50 should NOT be crisis");
    assert!(!is_ic_crisis(0.51), "0.51 should NOT be crisis");
    assert!(!is_ic_crisis(0.7), "0.7 should NOT be crisis");
    assert!(!is_ic_crisis(1.0), "1.0 should NOT be crisis");

    println!("RESULT: PASS - is_ic_crisis() boundary correct at 0.5");
}

#[test]
fn tc_session_09_is_ic_warning() {
    println!("\n=== TC-SESSION-09: is_ic_warning() ===");

    // SOURCE OF TRUTH: IDENTITY-002 - warning is 0.5 <= IC < 0.7

    // Below warning (crisis range)
    assert!(!is_ic_warning(0.49), "0.49 should NOT be warning (is crisis)");
    assert!(!is_ic_warning(0.0), "0.0 should NOT be warning (is crisis)");

    // Warning range
    assert!(is_ic_warning(0.5), "0.5 should be warning");
    assert!(is_ic_warning(0.55), "0.55 should be warning");
    assert!(is_ic_warning(0.69), "0.69 should be warning");

    // Above warning (good or healthy range)
    assert!(!is_ic_warning(0.7), "0.7 should NOT be warning (is good)");
    assert!(!is_ic_warning(0.71), "0.71 should NOT be warning");
    assert!(!is_ic_warning(0.9), "0.9 should NOT be warning (is healthy)");
    assert!(!is_ic_warning(1.0), "1.0 should NOT be warning");

    println!("RESULT: PASS - is_ic_warning() boundaries correct at 0.5 and 0.7");
}

#[test]
fn tc_session_09_classify_sync() {
    println!("\n=== TC-SESSION-09: classify_sync() ===");

    // SOURCE OF TRUTH: constitution.yaml gwt.kuramoto.thresholds
    // coherent: r >= 0.8
    // fragmented: r < 0.5

    // Good synchronization
    assert_eq!(classify_sync(1.0), "Good synchronization");
    assert_eq!(classify_sync(0.9), "Good synchronization");
    assert_eq!(classify_sync(0.8), "Good synchronization");

    // Partial synchronization
    assert_eq!(classify_sync(0.79), "Partial synchronization");
    assert_eq!(classify_sync(0.6), "Partial synchronization");
    assert_eq!(classify_sync(0.5), "Partial synchronization");

    // Fragmented
    assert_eq!(classify_sync(0.49), "Fragmented");
    assert_eq!(classify_sync(0.3), "Fragmented");
    assert_eq!(classify_sync(0.0), "Fragmented");

    println!("RESULT: PASS - classify_sync() thresholds correct");
}
```

### Step 3: Update mod.rs Exports

File: `crates/context-graph-core/src/gwt/session_identity/mod.rs`

Change line 22 from:
```rust
pub use manager::{compute_ic, compute_kuramoto_r, SessionIdentityManager};
```

To:
```rust
pub use manager::{
    classify_ic, classify_sync, compute_ic, compute_kuramoto_r,
    is_ic_crisis, is_ic_warning, SessionIdentityManager,
};
```

### Step 4: Update GWT mod.rs Exports

File: `crates/context-graph-core/src/gwt/mod.rs`

Change lines 101-104 from:
```rust
pub use session_identity::{
    clear_cache, compute_ic, compute_kuramoto_r, update_cache, IdentityCache,
    SessionIdentityManager, SessionIdentitySnapshot, MAX_TRAJECTORY_LEN,
};
```

To:
```rust
pub use session_identity::{
    classify_ic, classify_sync, clear_cache, compute_ic, compute_kuramoto_r,
    is_ic_crisis, is_ic_warning, update_cache, IdentityCache,
    SessionIdentityManager, SessionIdentitySnapshot, MAX_TRAJECTORY_LEN,
};
```

## IC Threshold Table (IDENTITY-002)

| IC Range | Classification | is_ic_crisis | is_ic_warning | Action |
|----------|---------------|--------------|---------------|--------|
| >= 0.9 | Healthy | false | false | Normal operation |
| [0.7, 0.9) | Good | false | false | Normal operation |
| [0.5, 0.7) | Warning | false | true | Log warning |
| < 0.5 | Degraded | true | false | Auto-dream trigger (AP-26, AP-38) |

## Definition of Done

### Acceptance Criteria

- [ ] classify_ic(0.91) returns "Healthy"
- [ ] classify_ic(0.90) returns "Healthy"
- [ ] classify_ic(0.89) returns "Good"
- [ ] classify_ic(0.70) returns "Good"
- [ ] classify_ic(0.69) returns "Warning"
- [ ] classify_ic(0.50) returns "Warning"
- [ ] classify_ic(0.49) returns "Degraded"
- [ ] is_ic_crisis returns true for IC < 0.5
- [ ] is_ic_crisis returns false for IC >= 0.5
- [ ] is_ic_warning returns true for 0.5 <= IC < 0.7
- [ ] is_ic_warning returns false for IC < 0.5 or IC >= 0.7
- [ ] classify_sync(0.8) returns "Good synchronization"
- [ ] classify_sync(0.5) returns "Partial synchronization"
- [ ] classify_sync(0.49) returns "Fragmented"

### Constraints

- All functions MUST be `#[inline]`
- Return `&'static str` (no allocation)
- Boundary conditions MUST match IDENTITY-002 exactly
- NO backwards compatibility shims - fail fast if something is wrong

## Verification Commands

```bash
# Build
cargo build -p context-graph-core

# Run specific tests
cargo test -p context-graph-core tc_session_09 -- --nocapture

# Run all session_identity tests
cargo test -p context-graph-core session_identity -- --nocapture
```

## Full State Verification Protocol

After implementing, you MUST:

### 1. Source of Truth Definition

The source of truth is the **test output from `cargo test`**. The functions are pure (no side effects, no database writes), so verification is done via:
- Return value assertions in tests
- Boundary value coverage

### 2. Execute & Inspect

```bash
# Run tests with full output
cargo test -p context-graph-core tc_session_09 -- --nocapture 2>&1 | tee /tmp/test_output.log

# Verify test count (should be 4 tests for tc_session_09_*)
grep -c "test tc_session_09" /tmp/test_output.log

# Verify all passed
grep "RESULT: PASS" /tmp/test_output.log
```

### 3. Boundary & Edge Case Audit

The tests include these edge cases:
1. **Exact boundary values**: 0.5, 0.7, 0.9 (verify correct classification)
2. **Just below boundaries**: 0.499, 0.699, 0.899 (must classify lower)
3. **Negative input**: -0.1 (must return "Degraded", not panic)

For each edge case, the test prints:
```
classify_ic(0.500) = 'Warning' (expected: 'Warning')
```

### 4. Evidence of Success

After tests pass, verify exports work:
```bash
# Verify functions are exported correctly
grep -n "classify_ic\|is_ic_crisis\|is_ic_warning\|classify_sync" \
  crates/context-graph-core/src/gwt/session_identity/mod.rs \
  crates/context-graph-core/src/gwt/mod.rs
```

Expected output should show exports in both files.

## Manual Testing Procedure

### Test 1: Boundary Values

Create a test binary or use `cargo test` with `--nocapture`:

| Input IC | Expected Output | Verify |
|----------|-----------------|--------|
| 0.90 | "Healthy" | Exact boundary |
| 0.899999 | "Good" | Float precision below |
| 0.70 | "Good" | Exact boundary |
| 0.699999 | "Warning" | Float precision below |
| 0.50 | "Warning" | Exact boundary |
| 0.499999 | "Degraded" | Float precision below |

### Test 2: Crisis Detection

| Input IC | is_ic_crisis() | is_ic_warning() |
|----------|----------------|-----------------|
| 0.49 | true | false |
| 0.50 | false | true |
| 0.69 | false | true |
| 0.70 | false | false |

### Test 3: Integration Check

After implementation, verify TASK-SESSION-08 can use these functions:
```rust
// In dream_trigger.rs (TASK-SESSION-08)
use crate::gwt::session_identity::is_ic_crisis;

if is_ic_crisis(ic) {
    // trigger dream
}
```

## Failure Modes (Fail Fast)

| Failure | Error | Resolution |
|---------|-------|------------|
| Wrong threshold | Test assertion failure | Fix threshold in match arm |
| Float precision issue | Boundary test fails | Use epsilon comparison if needed |
| Export missing | Compile error | Add to mod.rs exports |
| Function not inline | Performance test fails | Add #[inline] attribute |

## Next Task

After completion, proceed to **008-TASK-SESSION-08** (dream_trigger Module).

TASK-SESSION-08 depends on:
- `is_ic_crisis(ic: f32) -> bool` from this task
- `classify_ic(ic: f32) -> &'static str` for logging

```xml
</task_spec>
```
