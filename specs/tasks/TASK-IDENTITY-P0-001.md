# Task Specification: Identity Continuity Types

**Task ID:** TASK-IDENTITY-P0-001
**Version:** 2.1.0
**Status:** Complete
**Layer:** Foundation
**Sequence:** 1
**Estimated Complexity:** Low
**Completed:** 2026-01-11

---

## Metadata

| Field | Value |
|-------|-------|
| Implements | REQ-IDENTITY-001, REQ-IDENTITY-002, REQ-IDENTITY-005 |
| Depends On | None (foundation task) |
| Blocks | TASK-IDENTITY-P0-003, TASK-IDENTITY-P0-004 |
| Priority | P0 - Critical |

---

## CRITICAL: Current State Audit (2026-01-11)

**This audit was performed against the live codebase. The task was originally written assuming certain structures did NOT exist, but they DO.**

### What ALREADY EXISTS in `ego_node.rs`:

| Item | Status | Location | Notes |
|------|--------|----------|-------|
| `IdentityStatus` enum | **EXISTS** | Lines 193-203 | Healthy/Warning/Degraded/Critical - CORRECT thresholds |
| `IdentityContinuity` struct | **EXISTS** | Lines 176-186 | Has `identity_coherence`, `recent_continuity`, `kuramoto_order_parameter`, `status` |
| `compute_status_from_coherence()` | **EXISTS** | Lines 223-230 | Correct: >0.9 Healthy, >=0.7 Warning, >=0.5 Degraded, <0.5 Critical |
| `IdentityContinuity::update()` | **EXISTS** | Lines 232-244 | Computes IC = pv_cosine * kuramoto_r correctly |
| `Serialize, Deserialize` derives | **EXISTS** | Lines 176, 193 | Both structs are serializable |
| Unit tests for thresholds | **EXISTS** | Lines 392-444 | All boundary tests pass |

### What is MISSING (actual work for this task):

| Item | Why Needed | Constitution Reference |
|------|------------|------------------------|
| `computed_at: DateTime<Utc>` field | Timestamp when IC was computed | SPEC-IDENTITY-001 Section 5.1 |
| `IdentityContinuityResult::new(purpose_continuity, kuramoto_r)` | Factory with exact signature | PRD requirement |
| `IdentityContinuityResult::first_vector()` | Special case for first PV | EC-IDENTITY-01 |
| `is_in_crisis(&self) -> bool` | Convenience method: IC < 0.7 | REQ-IDENTITY-004 |
| `is_critical(&self) -> bool` | Convenience method: IC < 0.5 | REQ-IDENTITY-007 |

---

## Context

The existing `IdentityContinuity` struct provides the core IC calculation logic. This task adds:
1. A `computed_at` timestamp for auditing
2. Convenience methods for crisis detection (`is_in_crisis`, `is_critical`)
3. Factory methods (`new`, `first_vector`) matching the SPEC signature

**DECISION: Extend `IdentityContinuity` rather than create a separate `IdentityContinuityResult` struct.** This avoids duplication and maintains backward compatibility with existing code that uses `IdentityContinuity`.

---

## Input Context Files

| File | Purpose | Line References |
|------|---------|-----------------|
| `crates/context-graph-core/src/gwt/ego_node.rs` | Existing IdentityContinuity, IdentityStatus types | Lines 176-251 |
| `docs2/constitution.yaml` | Threshold definitions | Lines 365-392 (gwt.self_ego_node.identity_continuity) |
| `specs/functional/SPEC-IDENTITY-001.md` | Functional requirements | Section 5.1 Data Model |

---

## Prerequisites

- [x] Rust workspace compiles successfully (`cargo build -p context-graph-core` passes)
- [x] context-graph-core crate exists
- [x] gwt module exists with ego_node.rs
- [x] `IdentityContinuity` and `IdentityStatus` already exist and work correctly
- [x] Existing tests pass: `cargo test -p context-graph-core ego_node` (20 tests pass)

---

## Scope

### In Scope

1. Add `computed_at: DateTime<Utc>` field to `IdentityContinuity`
2. Add `IdentityContinuity::new(purpose_continuity: f32, kuramoto_r: f32) -> Self` factory method
3. Add `IdentityContinuity::first_vector() -> Self` factory method for initial case
4. Add `is_in_crisis(&self) -> bool` method (returns true if `identity_coherence < 0.7`)
5. Add `is_critical(&self) -> bool` method (returns true if `identity_coherence < 0.5`)
6. Update existing tests to verify new methods
7. Add serialization round-trip test for `computed_at` field

### Out of Scope

- IC computation logic changes (TASK-IDENTITY-P0-003)
- Crisis detection events (TASK-IDENTITY-P0-004)
- Crisis protocol execution (TASK-IDENTITY-P0-005)
- Workspace integration (TASK-IDENTITY-P0-006)
- MCP tool exposure (TASK-IDENTITY-P0-007)

---

## Definition of Done

### Exact Changes Required

#### 1. Update `IdentityContinuity` struct (Lines 176-186)

**BEFORE:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityContinuity {
    pub recent_continuity: f32,
    pub kuramoto_order_parameter: f32,
    pub identity_coherence: f32,
    pub status: IdentityStatus,
}
```

**AFTER:**
```rust
/// Tracks identity continuity over time
///
/// # Constitution Reference
/// From constitution.yaml lines 365-392:
/// - identity_continuity: "IC = cos(PV_t, PV_{t-1}) x r(t)"
/// - Thresholds: healthy>0.9, warning<0.7, dream<0.5
///
/// # Persistence (TASK-GWT-P1-001)
/// Serializable for diagnostic/recovery purposes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IdentityContinuity {
    /// IC = cos(PV_t, PV_{t-1}) x r(t), clamped to [0, 1]
    pub identity_coherence: f32,
    /// Cosine similarity between consecutive purpose vectors
    pub recent_continuity: f32,
    /// Order parameter r from Kuramoto sync
    pub kuramoto_order_parameter: f32,
    /// Status classification based on IC thresholds
    pub status: IdentityStatus,
    /// Timestamp of computation
    pub computed_at: DateTime<Utc>,
}
```

#### 2. Add factory methods to `IdentityContinuity` impl

```rust
impl IdentityContinuity {
    /// Create a new result with computed values
    ///
    /// # Arguments
    /// * `purpose_continuity` - cos(PV_t, PV_{t-1})
    /// * `kuramoto_r` - Kuramoto order parameter r(t)
    ///
    /// # Returns
    /// Result with IC = purpose_continuity * kuramoto_r, clamped to [0, 1]
    ///
    /// # Example
    /// ```rust
    /// let result = IdentityContinuity::new(0.9, 0.85);
    /// assert!((result.identity_coherence - 0.765).abs() < 1e-6);
    /// ```
    pub fn new(purpose_continuity: f32, kuramoto_r: f32) -> Self {
        // Clamp inputs to valid ranges
        let cos_clamped = purpose_continuity.clamp(-1.0, 1.0);
        let r_clamped = kuramoto_r.clamp(0.0, 1.0);

        // Compute IC = cos * r, clamp negative to 0
        let ic = (cos_clamped * r_clamped).max(0.0).min(1.0);

        // Determine status from IC
        let status = Self::compute_status_from_coherence(ic);

        Self {
            identity_coherence: ic,
            recent_continuity: cos_clamped,
            kuramoto_order_parameter: r_clamped,
            status,
            computed_at: Utc::now(),
        }
    }

    /// Create result for first purpose vector (no previous)
    ///
    /// Returns IC = 1.0, Status = Healthy
    /// Per EC-IDENTITY-01: First purpose vector defaults to healthy
    pub fn first_vector() -> Self {
        Self {
            identity_coherence: 1.0,
            recent_continuity: 1.0,
            kuramoto_order_parameter: 1.0,
            status: IdentityStatus::Healthy,
            computed_at: Utc::now(),
        }
    }

    /// Check if identity is in crisis (IC < 0.7)
    ///
    /// # Constitution Reference
    /// From constitution.yaml line 369:
    /// - warning<0.7 threshold indicates identity drift
    #[inline]
    pub fn is_in_crisis(&self) -> bool {
        self.identity_coherence < 0.7
    }

    /// Check if identity is critical (IC < 0.5)
    ///
    /// # Constitution Reference
    /// From constitution.yaml line 369:
    /// - dream<0.5 threshold triggers introspective dream
    #[inline]
    pub fn is_critical(&self) -> bool {
        self.identity_coherence < 0.5
    }

    // ... existing methods remain unchanged ...
}
```

#### 3. Update existing `IdentityContinuity::new()` (Lines 208-215)

The existing `new()` method initializes with `identity_coherence: 0.0`. This needs to be updated to also set `computed_at`:

**BEFORE:**
```rust
pub fn new() -> Self {
    let identity_coherence = 0.0;
    Self {
        recent_continuity: 1.0,
        kuramoto_order_parameter: 0.0,
        identity_coherence,
        status: Self::compute_status_from_coherence(identity_coherence),
    }
}
```

**AFTER:**
```rust
/// Create new IdentityContinuity with default initial state
///
/// Starting with identity_coherence=0.0 means status=Critical (IC < 0.5)
/// per constitution.yaml lines 387-392
pub fn default_initial() -> Self {
    let identity_coherence = 0.0;
    Self {
        identity_coherence,
        recent_continuity: 1.0,
        kuramoto_order_parameter: 0.0,
        status: Self::compute_status_from_coherence(identity_coherence),
        computed_at: Utc::now(),
    }
}
```

**NOTE**: Rename the existing `new()` to `default_initial()` to avoid confusion with the new factory method `new(purpose_continuity, kuramoto_r)`. Update callers accordingly.

#### 4. Update `IdentityContinuity::update()` (Lines 232-244)

Add `computed_at` update:

```rust
pub fn update(&mut self, pv_cosine: f32, kuramoto_r: f32) -> CoreResult<IdentityStatus> {
    self.recent_continuity = pv_cosine.clamp(-1.0, 1.0);
    self.kuramoto_order_parameter = kuramoto_r.clamp(0.0, 1.0);

    // Identity coherence = cosine × r
    self.identity_coherence = (pv_cosine * kuramoto_r).clamp(0.0, 1.0);

    // Determine status using canonical computation
    self.status = Self::compute_status_from_coherence(self.identity_coherence);

    // Update timestamp
    self.computed_at = Utc::now();

    Ok(self.status)
}
```

---

## Constraints

1. `identity_coherence` MUST be clamped to [0.0, 1.0]
2. Negative cosine values (opposite PVs) MUST be clamped to 0.0 in IC calculation
3. All types MUST derive `Serialize, Deserialize` for persistence
4. All types MUST derive `Debug, Clone`
5. `IdentityContinuity` MUST derive `PartialEq` for testing
6. Status thresholds MUST match constitution.yaml exactly:
   - Healthy: IC > 0.9
   - Warning: 0.7 <= IC <= 0.9
   - Degraded: 0.5 <= IC < 0.7
   - Critical: IC < 0.5
7. NO `unwrap()` or `expect()` in library code
8. NO backward-compatibility shims - if something breaks, fix the callers

---

## Verification Commands

```bash
# Build check
cargo build -p context-graph-core

# Type check
cargo check -p context-graph-core

# Run ALL tests for ego_node module
cargo test -p context-graph-core ego_node -- --nocapture

# Run specific new tests
cargo test -p context-graph-core identity_continuity_new_factory -- --nocapture
cargo test -p context-graph-core identity_continuity_first_vector -- --nocapture
cargo test -p context-graph-core identity_continuity_is_in_crisis -- --nocapture
cargo test -p context-graph-core identity_continuity_is_critical -- --nocapture
cargo test -p context-graph-core identity_continuity_serialization_with_timestamp -- --nocapture

# Clippy check (MUST pass with no warnings)
cargo clippy -p context-graph-core -- -D warnings

# Verify serialization round-trip with bincode
cargo test -p context-graph-core identity_continuity_bincode_roundtrip -- --nocapture
```

---

## Test Cases (with Full State Verification)

```rust
#[cfg(test)]
mod identity_continuity_tests {
    use super::*;

    // =========================================================================
    // Factory Method Tests
    // =========================================================================

    #[test]
    fn test_identity_continuity_new_factory_computes_ic_correctly() {
        println!("=== TEST: new() factory computes IC = cos * r ===");

        // BEFORE: No IdentityContinuity exists
        // EXECUTE: Create with known values
        let result = IdentityContinuity::new(0.9, 0.85);

        // AFTER: Verify IC computation
        let expected_ic = 0.9 * 0.85; // 0.765
        assert!((result.identity_coherence - expected_ic).abs() < 1e-6,
            "IC should be {} but was {}", expected_ic, result.identity_coherence);
        assert_eq!(result.status, IdentityStatus::Warning,
            "IC=0.765 should be Warning (0.7 <= IC <= 0.9)");

        // EVIDENCE: Print computed state
        println!("INPUT: purpose_continuity=0.9, kuramoto_r=0.85");
        println!("OUTPUT: identity_coherence={:.4}, status={:?}",
            result.identity_coherence, result.status);
        println!("EVIDENCE: IC = 0.9 * 0.85 = 0.765 (Warning)");
    }

    #[test]
    fn test_identity_continuity_new_factory_clamps_negative_cosine() {
        println!("=== TEST: new() clamps negative cosine to IC >= 0 ===");

        // BEFORE: Negative cosine (opposite vectors)
        let result = IdentityContinuity::new(-0.8, 0.9);

        // AFTER: IC should be 0.0 (clamped), not -0.72
        assert!(result.identity_coherence >= 0.0,
            "IC must be >= 0, but was {}", result.identity_coherence);
        assert_eq!(result.status, IdentityStatus::Critical,
            "IC=0.0 should be Critical");

        println!("INPUT: purpose_continuity=-0.8, kuramoto_r=0.9");
        println!("OUTPUT: identity_coherence={:.4}, status={:?}",
            result.identity_coherence, result.status);
        println!("EVIDENCE: Negative cosine clamped to IC=0.0");
    }

    #[test]
    fn test_identity_continuity_new_factory_clamps_inputs() {
        println!("=== TEST: new() clamps inputs to valid ranges ===");

        // Out-of-range inputs
        let result = IdentityContinuity::new(1.5, 2.0); // Should clamp to 1.0, 1.0

        assert!((result.recent_continuity - 1.0).abs() < 1e-6,
            "purpose_continuity should clamp to 1.0");
        assert!((result.kuramoto_order_parameter - 1.0).abs() < 1e-6,
            "kuramoto_r should clamp to 1.0");
        assert!((result.identity_coherence - 1.0).abs() < 1e-6,
            "IC should be 1.0 * 1.0 = 1.0");

        println!("INPUT: purpose_continuity=1.5, kuramoto_r=2.0");
        println!("OUTPUT: recent_continuity={:.4}, kuramoto_order_parameter={:.4}",
            result.recent_continuity, result.kuramoto_order_parameter);
        println!("EVIDENCE: Inputs clamped to [−1,1] and [0,1]");
    }

    #[test]
    fn test_identity_continuity_first_vector_returns_healthy() {
        println!("=== TEST: first_vector() returns IC=1.0, Healthy ===");

        let result = IdentityContinuity::first_vector();

        assert_eq!(result.identity_coherence, 1.0);
        assert_eq!(result.status, IdentityStatus::Healthy);
        assert_eq!(result.recent_continuity, 1.0);
        assert_eq!(result.kuramoto_order_parameter, 1.0);

        println!("OUTPUT: identity_coherence=1.0, status=Healthy");
        println!("EVIDENCE: First vector defaults to perfect continuity");
    }

    #[test]
    fn test_identity_continuity_first_vector_has_timestamp() {
        println!("=== TEST: first_vector() has computed_at timestamp ===");

        let before = Utc::now();
        let result = IdentityContinuity::first_vector();
        let after = Utc::now();

        assert!(result.computed_at >= before && result.computed_at <= after,
            "computed_at should be between test start and end");

        println!("computed_at: {:?}", result.computed_at);
        println!("EVIDENCE: Timestamp is set correctly");
    }

    // =========================================================================
    // is_in_crisis() and is_critical() Tests
    // =========================================================================

    #[test]
    fn test_identity_continuity_is_in_crisis_boundary() {
        println!("=== TEST: is_in_crisis() boundary at IC=0.7 ===");

        // Exactly 0.7 should NOT be crisis
        let at_boundary = IdentityContinuity::new(0.7, 1.0);
        assert!(!at_boundary.is_in_crisis(),
            "IC=0.7 should NOT be in crisis (boundary is < 0.7)");

        // Just below 0.7 IS crisis
        let below_boundary = IdentityContinuity::new(0.699, 1.0);
        assert!(below_boundary.is_in_crisis(),
            "IC=0.699 should be in crisis");

        println!("IC=0.7: is_in_crisis={}", at_boundary.is_in_crisis());
        println!("IC=0.699: is_in_crisis={}", below_boundary.is_in_crisis());
        println!("EVIDENCE: Boundary at 0.7 is exclusive (< not <=)");
    }

    #[test]
    fn test_identity_continuity_is_critical_boundary() {
        println!("=== TEST: is_critical() boundary at IC=0.5 ===");

        // Exactly 0.5 should NOT be critical
        let at_boundary = IdentityContinuity::new(0.5, 1.0);
        assert!(!at_boundary.is_critical(),
            "IC=0.5 should NOT be critical (boundary is < 0.5)");

        // Just below 0.5 IS critical
        let below_boundary = IdentityContinuity::new(0.499, 1.0);
        assert!(below_boundary.is_critical(),
            "IC=0.499 should be critical");

        println!("IC=0.5: is_critical={}", at_boundary.is_critical());
        println!("IC=0.499: is_critical={}", below_boundary.is_critical());
        println!("EVIDENCE: Boundary at 0.5 is exclusive (< not <=)");
    }

    #[test]
    fn test_identity_continuity_crisis_methods_consistent_with_status() {
        println!("=== TEST: is_in_crisis/is_critical consistent with status ===");

        // Healthy: IC > 0.9
        let healthy = IdentityContinuity::new(1.0, 0.95);
        assert!(!healthy.is_in_crisis());
        assert!(!healthy.is_critical());
        assert_eq!(healthy.status, IdentityStatus::Healthy);

        // Warning: 0.7 <= IC <= 0.9
        let warning = IdentityContinuity::new(0.8, 1.0);
        assert!(!warning.is_in_crisis());
        assert!(!warning.is_critical());
        assert_eq!(warning.status, IdentityStatus::Warning);

        // Degraded: 0.5 <= IC < 0.7
        let degraded = IdentityContinuity::new(0.6, 1.0);
        assert!(degraded.is_in_crisis()); // < 0.7
        assert!(!degraded.is_critical()); // >= 0.5
        assert_eq!(degraded.status, IdentityStatus::Degraded);

        // Critical: IC < 0.5
        let critical = IdentityContinuity::new(0.3, 1.0);
        assert!(critical.is_in_crisis());
        assert!(critical.is_critical());
        assert_eq!(critical.status, IdentityStatus::Critical);

        println!("Healthy: is_in_crisis={}, is_critical={}", healthy.is_in_crisis(), healthy.is_critical());
        println!("Warning: is_in_crisis={}, is_critical={}", warning.is_in_crisis(), warning.is_critical());
        println!("Degraded: is_in_crisis={}, is_critical={}", degraded.is_in_crisis(), degraded.is_critical());
        println!("Critical: is_in_crisis={}, is_critical={}", critical.is_in_crisis(), critical.is_critical());
    }

    // =========================================================================
    // Serialization Tests
    // =========================================================================

    #[test]
    fn test_identity_continuity_bincode_roundtrip() {
        println!("=== TEST: bincode serialization roundtrip ===");

        let original = IdentityContinuity::new(0.85, 0.9);

        // Serialize
        let serialized = bincode::serialize(&original)
            .expect("Serialization must not fail");

        // Deserialize
        let deserialized: IdentityContinuity = bincode::deserialize(&serialized)
            .expect("Deserialization must not fail");

        // Verify all fields
        assert_eq!(original.identity_coherence, deserialized.identity_coherence);
        assert_eq!(original.recent_continuity, deserialized.recent_continuity);
        assert_eq!(original.kuramoto_order_parameter, deserialized.kuramoto_order_parameter);
        assert_eq!(original.status, deserialized.status);
        assert_eq!(original.computed_at, deserialized.computed_at);

        println!("Original: {:?}", original);
        println!("Deserialized: {:?}", deserialized);
        println!("EVIDENCE: All fields preserved through serialization");
    }

    #[test]
    fn test_identity_continuity_json_roundtrip() {
        println!("=== TEST: JSON serialization roundtrip ===");

        let original = IdentityContinuity::new(0.75, 0.8);

        // Serialize to JSON
        let json = serde_json::to_string(&original)
            .expect("JSON serialization must not fail");

        // Deserialize from JSON
        let deserialized: IdentityContinuity = serde_json::from_str(&json)
            .expect("JSON deserialization must not fail");

        assert_eq!(original.identity_coherence, deserialized.identity_coherence);
        assert_eq!(original.status, deserialized.status);

        println!("JSON: {}", json);
        println!("EVIDENCE: JSON serialization works correctly");
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_identity_continuity_zero_r_gives_critical() {
        println!("=== EDGE CASE: Zero Kuramoto r ===");

        // r = 0 means no synchronization, IC = 0
        let result = IdentityContinuity::new(1.0, 0.0);

        assert_eq!(result.identity_coherence, 0.0);
        assert_eq!(result.status, IdentityStatus::Critical);
        assert!(result.is_critical());

        println!("INPUT: purpose_continuity=1.0, kuramoto_r=0.0");
        println!("OUTPUT: IC=0.0, status=Critical");
        println!("EVIDENCE: Zero sync means zero identity coherence");
    }

    #[test]
    fn test_identity_continuity_perfect_values() {
        println!("=== EDGE CASE: Perfect values ===");

        let result = IdentityContinuity::new(1.0, 1.0);

        assert_eq!(result.identity_coherence, 1.0);
        assert_eq!(result.status, IdentityStatus::Healthy);
        assert!(!result.is_in_crisis());
        assert!(!result.is_critical());

        println!("EVIDENCE: Perfect continuity and sync gives IC=1.0, Healthy");
    }

    // =========================================================================
    // Full State Verification
    // =========================================================================

    #[test]
    fn fsv_identity_continuity_full_lifecycle() {
        println!("=== FULL STATE VERIFICATION: IdentityContinuity lifecycle ===");

        // SOURCE OF TRUTH: IdentityContinuity struct fields after operations

        // 1. Create with factory
        let result = IdentityContinuity::new(0.8, 0.9);

        println!("\nSTATE AFTER new(0.8, 0.9):");
        println!("  identity_coherence: {:.4}", result.identity_coherence);
        println!("  recent_continuity: {:.4}", result.recent_continuity);
        println!("  kuramoto_order_parameter: {:.4}", result.kuramoto_order_parameter);
        println!("  status: {:?}", result.status);
        println!("  computed_at: {:?}", result.computed_at);
        println!("  is_in_crisis: {}", result.is_in_crisis());
        println!("  is_critical: {}", result.is_critical());

        // Verify expected values
        assert!((result.identity_coherence - 0.72).abs() < 1e-6,
            "IC should be 0.8 * 0.9 = 0.72");
        assert_eq!(result.status, IdentityStatus::Warning,
            "0.72 is in Warning range [0.7, 0.9]");
        assert!(!result.is_in_crisis(),
            "0.72 >= 0.7, not in crisis");

        // 2. Create first_vector
        let first = IdentityContinuity::first_vector();

        println!("\nSTATE AFTER first_vector():");
        println!("  identity_coherence: {:.4}", first.identity_coherence);
        println!("  status: {:?}", first.status);

        assert_eq!(first.identity_coherence, 1.0);
        assert_eq!(first.status, IdentityStatus::Healthy);

        // 3. Verify serialization persistence
        let serialized = bincode::serialize(&result).unwrap();
        let restored: IdentityContinuity = bincode::deserialize(&serialized).unwrap();

        println!("\nSTATE AFTER serialization roundtrip:");
        println!("  identity_coherence preserved: {}",
            result.identity_coherence == restored.identity_coherence);
        println!("  computed_at preserved: {}",
            result.computed_at == restored.computed_at);

        assert_eq!(result, restored, "Serialization must preserve all fields");

        println!("\nEVIDENCE OF SUCCESS:");
        println!("  - Factory method computes IC correctly");
        println!("  - first_vector() returns healthy initial state");
        println!("  - Crisis methods work correctly");
        println!("  - Serialization preserves all fields including timestamp");
    }
}
```

---

## Files to Create

None - all additions go to existing file.

---

## Files to Modify

| File | Changes |
|------|---------|
| `crates/context-graph-core/src/gwt/ego_node.rs` | Add `computed_at` field, factory methods, convenience methods |

---

## Callers to Update

When renaming `IdentityContinuity::new()` to `default_initial()`, search for and update:

```bash
# Find all usages
grep -rn "IdentityContinuity::new()" crates/

# Expected locations:
# - ego_node.rs: SelfAwarenessLoop::new() (line 257)
# - ego_node.rs: tests (lines 392, 408, etc.)
```

---

## Validation Criteria

| Criterion | Verification Method | Source of Truth |
|-----------|---------------------|-----------------|
| `IdentityContinuity::new(cos, r)` computes IC correctly | Unit test `test_identity_continuity_new_factory_computes_ic_correctly` | `identity_coherence` field value |
| `first_vector()` returns IC=1.0, Healthy | Unit test `test_identity_continuity_first_vector_returns_healthy` | `identity_coherence`, `status` fields |
| `is_in_crisis()` returns true for IC < 0.7 | Unit test `test_identity_continuity_is_in_crisis_boundary` | Method return value |
| `is_critical()` returns true for IC < 0.5 | Unit test `test_identity_continuity_is_critical_boundary` | Method return value |
| Serialization round-trip preserves `computed_at` | Unit test `test_identity_continuity_bincode_roundtrip` | Deserialized struct equality |
| No clippy warnings | `cargo clippy -D warnings` | Exit code 0 |
| All existing tests pass | `cargo test -p context-graph-core ego_node` | 20+ tests pass |

---

## Full State Verification Requirements

**YOU MUST PERFORM THESE STEPS AFTER COMPLETING THE LOGIC:**

### 1. Define the Source of Truth
The source of truth is the `IdentityContinuity` struct fields:
- `identity_coherence: f32`
- `recent_continuity: f32`
- `kuramoto_order_parameter: f32`
- `status: IdentityStatus`
- `computed_at: DateTime<Utc>`

### 2. Execute & Inspect
After creating an `IdentityContinuity`, you MUST:
1. Read the struct fields directly
2. Print the values to stdout
3. Assert they match expected values

### 3. Boundary & Edge Case Audit
Manually simulate these 3 edge cases and print state BEFORE and AFTER:

| Edge Case | Input | Expected Output | Verification |
|-----------|-------|-----------------|--------------|
| Zero Kuramoto r | `new(1.0, 0.0)` | IC=0.0, Critical | Print and assert |
| Negative cosine | `new(-0.8, 0.9)` | IC=0.0, Critical | Print and assert |
| Perfect values | `new(1.0, 1.0)` | IC=1.0, Healthy | Print and assert |

### 4. Evidence of Success
Run tests with `--nocapture` and capture the output log showing:
- Actual values computed
- Status transitions
- Serialization roundtrip results

```bash
cargo test -p context-graph-core identity_continuity -- --nocapture 2>&1 | tee test_output.log
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | Claude Opus 4.5 | Initial task specification |
| 2.0.0 | 2026-01-11 | Claude Opus 4.5 | **MAJOR UPDATE**: Audit against codebase revealed existing implementations. Updated task to reflect reality: extend `IdentityContinuity` instead of creating new struct. Added FSV requirements. |
| 2.1.0 | 2026-01-11 | Claude Opus 4.5 | **COMPLETED**: All features implemented, 16 unit tests pass, FSV verified, clippy clean. |

---

## Completion Notes

### Implementation Summary

All requirements from this task have been implemented:

1. ✅ Added `computed_at: DateTime<Utc>` field to `IdentityContinuity`
2. ✅ Added `PartialEq` derive to struct
3. ✅ Renamed `new()` to `default_initial()`
4. ✅ Created `new(purpose_continuity, kuramoto_r)` factory method
5. ✅ Added `first_vector()` factory method (IC=1.0, Healthy)
6. ✅ Added `is_in_crisis(&self)` method (IC < 0.7)
7. ✅ Added `is_critical(&self)` method (IC < 0.5)
8. ✅ Updated `update()` to set `computed_at` timestamp
9. ✅ Updated all callers in ego_node.rs, gwt_integration.rs, gwt_providers.rs

### Verification Results

```
cargo build -p context-graph-core     # ✅ PASS
cargo build -p context-graph-mcp      # ✅ PASS
cargo clippy -p context-graph-core -- -D warnings  # ✅ PASS (0 warnings)
cargo test -p context-graph-core identity_continuity -- --nocapture  # ✅ 16 tests pass
cargo test -p context-graph-core --test gwt_integration  # ✅ 19 tests pass
```

### Files Modified

| File | Changes |
|------|---------|
| `crates/context-graph-core/src/gwt/ego_node.rs` | Added computed_at field, PartialEq derive, factory methods, convenience methods, updated callers, added 16 tests |
| `crates/context-graph-core/tests/gwt_integration.rs` | Updated `IdentityContinuity::new()` → `default_initial()` |
| `crates/context-graph-mcp/src/handlers/gwt_providers.rs` | Updated `IdentityContinuity::new()` → `default_initial()` |

### Code Review

Code-simplifier agent rated the implementation as **Production-Ready** with:
- Excellent code clarity (clear naming, well-documented with constitution references)
- Excellent consistency (consistent patterns across all methods)
- Excellent Rust best practices (proper use of clamp, #[inline], derives)
- No redundancy found
- Comprehensive test coverage including FSV tests
