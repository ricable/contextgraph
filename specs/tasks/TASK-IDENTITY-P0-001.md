# Task Specification: Identity Continuity Types

**Task ID:** TASK-IDENTITY-P0-001
**Version:** 1.0.0
**Status:** Ready
**Layer:** Foundation
**Sequence:** 1
**Estimated Complexity:** Low

---

## Metadata

| Field | Value |
|-------|-------|
| Implements | REQ-IDENTITY-001, REQ-IDENTITY-002, REQ-IDENTITY-005 |
| Depends On | None (foundation task) |
| Blocks | TASK-IDENTITY-P0-003, TASK-IDENTITY-P0-004 |
| Priority | P0 - Critical |

---

## Context

This is the foundation task for the Identity Continuity Loop. It creates the core type definitions that all subsequent tasks depend on. The types must be serializable for future persistence integration.

The existing `IdentityContinuity` struct in `ego_node.rs` already has the basic structure, but we need to add:
1. `IdentityContinuityResult` for computation results
2. Ensure all types are properly serializable
3. Add documentation with PRD references

---

## Input Context Files

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/gwt/ego_node.rs` | Existing IdentityContinuity, IdentityStatus types |
| `docs2/constitution.yaml` | Lines 365-392 for threshold definitions |
| `specs/functional/SPEC-IDENTITY-001.md` | Functional requirements |

---

## Prerequisites

- [x] Rust workspace compiles successfully
- [x] context-graph-core crate exists
- [x] gwt module exists with ego_node.rs

---

## Scope

### In Scope

1. Create `IdentityContinuityResult` struct with all fields from SPEC-IDENTITY-001
2. Ensure `IdentityStatus` enum matches constitution thresholds exactly
3. Add `Serialize, Deserialize` derives to new types
4. Add comprehensive documentation with constitution references
5. Add unit tests for type construction and serialization

### Out of Scope

- IC computation logic (TASK-IDENTITY-P0-003)
- Crisis detection logic (TASK-IDENTITY-P0-004)
- Workspace integration (TASK-IDENTITY-P0-006)

---

## Definition of Done

### Exact Signatures Required

```rust
// File: crates/context-graph-core/src/gwt/ego_node.rs

/// Result of identity continuity computation
///
/// Captures the full IC calculation: IC = cos(PV_t, PV_{t-1}) x r(t)
///
/// # Constitution Reference
/// From constitution.yaml lines 365-392:
/// - identity_continuity: "IC = cos(PV_t, PV_{t-1}) x r(t)"
/// - Thresholds: healthy>0.9, warning<0.7, dream<0.5
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IdentityContinuityResult {
    /// IC = cos(PV_t, PV_{t-1}) x r(t), clamped to [0, 1]
    pub identity_coherence: f32,
    /// Cosine similarity between consecutive purpose vectors
    pub purpose_continuity: f32,
    /// Kuramoto order parameter at computation time
    pub kuramoto_r: f32,
    /// Status classification based on IC thresholds
    pub status: IdentityStatus,
    /// Timestamp of computation
    pub computed_at: DateTime<Utc>,
}

impl IdentityContinuityResult {
    /// Create a new result with computed values
    ///
    /// # Arguments
    /// * `purpose_continuity` - cos(PV_t, PV_{t-1})
    /// * `kuramoto_r` - Kuramoto order parameter r(t)
    ///
    /// # Returns
    /// Result with IC = purpose_continuity * kuramoto_r, clamped to [0, 1]
    pub fn new(purpose_continuity: f32, kuramoto_r: f32) -> Self;

    /// Create result for first purpose vector (no previous)
    /// Returns IC = 1.0, Status = Healthy
    pub fn first_vector() -> Self;

    /// Check if identity is in crisis (IC < 0.7)
    pub fn is_in_crisis(&self) -> bool;

    /// Check if identity is critical (IC < 0.5)
    pub fn is_critical(&self) -> bool;
}
```

### Constraints

1. `identity_coherence` MUST be clamped to [0.0, 1.0]
2. Negative cosine values (opposite PVs) MUST be clamped to 0.0
3. All types MUST derive `Serialize, Deserialize` for persistence
4. All types MUST derive `Debug, Clone`
5. Status thresholds MUST match constitution.yaml exactly:
   - Healthy: IC > 0.9
   - Warning: 0.7 <= IC <= 0.9
   - Degraded: 0.5 <= IC < 0.7
   - Critical: IC < 0.5
6. NO `unwrap()` or `expect()` in library code

### Verification Commands

```bash
# Build check
cargo build -p context-graph-core

# Type check
cargo check -p context-graph-core

# Run tests for ego_node module
cargo test -p context-graph-core ego_node

# Verify serialization works
cargo test -p context-graph-core identity_continuity_result_serialization

# Clippy check
cargo clippy -p context-graph-core -- -D warnings
```

---

## Pseudo Code

```rust
// In ego_node.rs, add after existing IdentityContinuity struct:

/// Result of identity continuity computation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IdentityContinuityResult {
    pub identity_coherence: f32,      // IC value
    pub purpose_continuity: f32,      // cos(PV_t, PV_{t-1})
    pub kuramoto_r: f32,              // r(t)
    pub status: IdentityStatus,       // Threshold classification
    pub computed_at: DateTime<Utc>,   // When computed
}

impl IdentityContinuityResult {
    pub fn new(purpose_continuity: f32, kuramoto_r: f32) -> Self {
        // Clamp inputs to valid ranges
        let cos_clamped = purpose_continuity.clamp(-1.0, 1.0);
        let r_clamped = kuramoto_r.clamp(0.0, 1.0);

        // Compute IC = cos * r, clamp negative to 0
        let ic = (cos_clamped * r_clamped).max(0.0).min(1.0);

        // Determine status from IC
        let status = Self::status_from_ic(ic);

        Self {
            identity_coherence: ic,
            purpose_continuity: cos_clamped,
            kuramoto_r: r_clamped,
            status,
            computed_at: Utc::now(),
        }
    }

    pub fn first_vector() -> Self {
        Self {
            identity_coherence: 1.0,
            purpose_continuity: 1.0,
            kuramoto_r: 1.0,
            status: IdentityStatus::Healthy,
            computed_at: Utc::now(),
        }
    }

    fn status_from_ic(ic: f32) -> IdentityStatus {
        match ic {
            x if x > 0.9 => IdentityStatus::Healthy,
            x if x >= 0.7 => IdentityStatus::Warning,
            x if x >= 0.5 => IdentityStatus::Degraded,
            _ => IdentityStatus::Critical,
        }
    }

    pub fn is_in_crisis(&self) -> bool {
        self.identity_coherence < 0.7
    }

    pub fn is_critical(&self) -> bool {
        self.identity_coherence < 0.5
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
| `crates/context-graph-core/src/gwt/ego_node.rs` | Add `IdentityContinuityResult` struct and impl |

---

## Validation Criteria

| Criterion | Verification Method |
|-----------|---------------------|
| IdentityContinuityResult compiles | `cargo build` |
| Serialization round-trip works | Unit test |
| Status thresholds match constitution | Unit tests for each boundary |
| is_in_crisis returns true for IC < 0.7 | Unit test |
| is_critical returns true for IC < 0.5 | Unit test |
| No clippy warnings | `cargo clippy -D warnings` |

---

## Test Cases

```rust
#[cfg(test)]
mod identity_result_tests {
    use super::*;

    #[test]
    fn test_new_with_perfect_continuity() {
        let result = IdentityContinuityResult::new(1.0, 0.9);
        assert!((result.identity_coherence - 0.9).abs() < 1e-6);
        assert_eq!(result.status, IdentityStatus::Warning); // 0.9 is Warning boundary
    }

    #[test]
    fn test_new_healthy_status() {
        let result = IdentityContinuityResult::new(1.0, 0.95);
        assert!(result.identity_coherence > 0.9);
        assert_eq!(result.status, IdentityStatus::Healthy);
    }

    #[test]
    fn test_new_with_zero_r() {
        let result = IdentityContinuityResult::new(1.0, 0.0);
        assert_eq!(result.identity_coherence, 0.0);
        assert_eq!(result.status, IdentityStatus::Critical);
    }

    #[test]
    fn test_new_with_negative_cosine() {
        let result = IdentityContinuityResult::new(-0.5, 0.9);
        assert!(result.identity_coherence >= 0.0);
        assert_eq!(result.status, IdentityStatus::Critical);
    }

    #[test]
    fn test_first_vector() {
        let result = IdentityContinuityResult::first_vector();
        assert_eq!(result.identity_coherence, 1.0);
        assert_eq!(result.status, IdentityStatus::Healthy);
    }

    #[test]
    fn test_is_in_crisis_boundary() {
        let crisis = IdentityContinuityResult::new(0.69, 1.0);
        let not_crisis = IdentityContinuityResult::new(0.70, 1.0);

        assert!(crisis.is_in_crisis());
        assert!(!not_crisis.is_in_crisis());
    }

    #[test]
    fn test_is_critical_boundary() {
        let critical = IdentityContinuityResult::new(0.49, 1.0);
        let not_critical = IdentityContinuityResult::new(0.50, 1.0);

        assert!(critical.is_critical());
        assert!(!not_critical.is_critical());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let result = IdentityContinuityResult::new(0.85, 0.9);
        let serialized = bincode::serialize(&result).unwrap();
        let deserialized: IdentityContinuityResult = bincode::deserialize(&serialized).unwrap();

        assert_eq!(result.identity_coherence, deserialized.identity_coherence);
        assert_eq!(result.status, deserialized.status);
    }
}
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | Claude Opus 4.5 | Initial task specification |
