# Task Specification: Identity Continuity Computation

**Task ID:** TASK-IDENTITY-P0-003
**Version:** 1.0.0
**Status:** Ready
**Layer:** Logic
**Sequence:** 3
**Estimated Complexity:** Medium

---

## Metadata

| Field | Value |
|-------|-------|
| Implements | REQ-IDENTITY-001, REQ-IDENTITY-002, REQ-IDENTITY-003 |
| Depends On | TASK-IDENTITY-P0-001, TASK-IDENTITY-P0-002 |
| Blocks | TASK-IDENTITY-P0-004, TASK-IDENTITY-P0-006 |
| Priority | P0 - Critical |

---

## Context

This is the core computation task for the Identity Continuity Loop. It implements the IC formula:

```
IC = cos(PV_t, PV_{t-1}) x r(t)
```

Where:
- `PV_t` = Current purpose vector (13D)
- `PV_{t-1}` = Previous purpose vector (13D)
- `r(t)` = Kuramoto order parameter at time t
- `IC` = Identity coherence, clamped to [0, 1]

The existing `SelfAwarenessLoop` has a `cycle()` method that computes alignment, but it needs to be refactored to use the new types and provide a cleaner interface for continuous monitoring.

---

## Input Context Files

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/gwt/ego_node.rs` | Existing cosine_similarity, SelfAwarenessLoop |
| `specs/tasks/TASK-IDENTITY-P0-001.md` | IdentityContinuityResult type |
| `specs/tasks/TASK-IDENTITY-P0-002.md` | PurposeVectorHistory interface |
| `docs2/constitution.yaml` | Lines 365-392 for IC formula |

---

## Prerequisites

- [x] TASK-IDENTITY-P0-001 completed (IdentityContinuityResult exists)
- [x] TASK-IDENTITY-P0-002 completed (PurposeVectorHistory exists)
- [x] Rust workspace compiles

---

## Scope

### In Scope

1. Create `IdentityContinuityMonitor` struct
2. Implement `compute_continuity()` method with IC formula
3. Implement cosine similarity for 13D vectors
4. Handle edge cases (first vector, zero vectors, no sync)
5. Add comprehensive unit tests including boundary conditions

### Out of Scope

- Crisis detection and protocol (TASK-IDENTITY-P0-004, TASK-IDENTITY-P0-005)
- Workspace event subscription (TASK-IDENTITY-P0-006)
- MCP tool integration (TASK-IDENTITY-P0-007)

---

## Definition of Done

### Exact Signatures Required

```rust
// File: crates/context-graph-core/src/gwt/ego_node.rs

/// Monitor for continuous identity coherence tracking
///
/// Computes IC = cos(PV_t, PV_{t-1}) x r(t) per constitution.yaml line 369.
/// Maintains purpose vector history and provides real-time IC status.
///
/// # Example
/// ```
/// let mut monitor = IdentityContinuityMonitor::new();
/// let result = monitor.compute_continuity(&new_pv, kuramoto_r)?;
/// if result.is_in_crisis() {
///     // Handle identity crisis
/// }
/// ```
#[derive(Debug)]
pub struct IdentityContinuityMonitor {
    /// Purpose vector history
    history: PurposeVectorHistory,
    /// Last computed result
    last_result: Option<IdentityContinuityResult>,
    /// Crisis threshold (0.7 per constitution)
    crisis_threshold: f32,
}

impl IdentityContinuityMonitor {
    /// Create new monitor with default crisis threshold (0.7)
    pub fn new() -> Self;

    /// Create with custom crisis threshold
    pub fn with_threshold(threshold: f32) -> Self;

    /// Compute identity continuity for a new purpose vector
    ///
    /// # Algorithm
    /// 1. Get previous PV from history (if exists)
    /// 2. Compute cos(PV_t, PV_{t-1}) using cosine similarity
    /// 3. Multiply by Kuramoto r(t)
    /// 4. Clamp result to [0, 1]
    /// 5. Determine status from thresholds
    /// 6. Store PV in history
    /// 7. Return IdentityContinuityResult
    ///
    /// # Arguments
    /// * `new_pv` - The new purpose vector PV_t
    /// * `kuramoto_r` - Current Kuramoto order parameter
    ///
    /// # Returns
    /// * `CoreResult<IdentityContinuityResult>` with IC and status
    ///
    /// # Edge Cases
    /// - First vector: Returns IC = 1.0, Healthy
    /// - Zero vector: Returns IC = 0.0, Critical
    /// - r = 0: Returns IC = 0.0, Critical
    pub fn compute_continuity(
        &mut self,
        new_pv: &[f32; 13],
        kuramoto_r: f32,
    ) -> CoreResult<IdentityContinuityResult>;

    /// Get the last computed result, if any
    pub fn last_result(&self) -> Option<&IdentityContinuityResult>;

    /// Get current identity coherence value
    pub fn identity_coherence(&self) -> f32;

    /// Get current identity status
    pub fn current_status(&self) -> IdentityStatus;

    /// Check if currently in crisis (IC < threshold)
    pub fn is_in_crisis(&self) -> bool;

    /// Get the history length
    pub fn history_len(&self) -> usize;
}

impl Default for IdentityContinuityMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute cosine similarity between two 13D purpose vectors
///
/// # Arguments
/// * `v1` - First purpose vector
/// * `v2` - Second purpose vector
///
/// # Returns
/// * Cosine similarity in range [-1, 1]
/// * Returns 0.0 if either vector has zero magnitude
///
/// # Performance
/// Constraint: < 1us for 13D vectors
pub fn cosine_similarity_13d(v1: &[f32; 13], v2: &[f32; 13]) -> f32;
```

### Constants

```rust
/// Default crisis threshold per constitution.yaml line 369
pub const IC_CRISIS_THRESHOLD: f32 = 0.7;

/// Critical threshold triggering dream
pub const IC_CRITICAL_THRESHOLD: f32 = 0.5;
```

### Constraints

1. IC MUST be clamped to [0.0, 1.0]
2. Negative cosine values MUST result in IC = 0.0 (clamped)
3. Zero magnitude vectors MUST return cosine = 0.0 (no division by zero)
4. cosine_similarity_13d MUST be numerically stable (handle near-zero magnitudes)
5. compute_continuity MUST update history after computation
6. compute_continuity MUST return first_vector() result for empty history
7. NO panics from any input combination
8. Performance: cosine_similarity_13d < 1us

### Verification Commands

```bash
# Build
cargo build -p context-graph-core

# Run all identity tests
cargo test -p context-graph-core identity_continuity_monitor

# Run edge case tests
cargo test -p context-graph-core identity_edge_cases

# Benchmark cosine similarity
cargo bench -p context-graph-core cosine_13d

# Clippy
cargo clippy -p context-graph-core -- -D warnings
```

---

## Pseudo Code

```rust
pub const IC_CRISIS_THRESHOLD: f32 = 0.7;
pub const IC_CRITICAL_THRESHOLD: f32 = 0.5;

pub fn cosine_similarity_13d(v1: &[f32; 13], v2: &[f32; 13]) -> f32 {
    let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let mag1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Handle zero magnitude vectors
    const EPSILON: f32 = 1e-8;
    if mag1 < EPSILON || mag2 < EPSILON {
        return 0.0;
    }

    dot / (mag1 * mag2)
}

#[derive(Debug)]
pub struct IdentityContinuityMonitor {
    history: PurposeVectorHistory,
    last_result: Option<IdentityContinuityResult>,
    crisis_threshold: f32,
}

impl IdentityContinuityMonitor {
    pub fn new() -> Self {
        Self::with_threshold(IC_CRISIS_THRESHOLD)
    }

    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            history: PurposeVectorHistory::new(),
            last_result: None,
            crisis_threshold: threshold.clamp(0.0, 1.0),
        }
    }

    pub fn compute_continuity(
        &mut self,
        new_pv: &[f32; 13],
        kuramoto_r: f32,
    ) -> CoreResult<IdentityContinuityResult> {
        // Validate inputs
        let r_clamped = kuramoto_r.clamp(0.0, 1.0);

        // Check for first vector case
        if self.history.is_empty() {
            self.history.push(*new_pv, "Initial purpose vector");
            let result = IdentityContinuityResult::first_vector();
            self.last_result = Some(result.clone());
            return Ok(result);
        }

        // Get previous PV
        let prev_pv = self.history.current()
            .ok_or_else(|| CoreError::internal("History inconsistency"))?;

        // Compute cosine similarity
        let cos = cosine_similarity_13d(new_pv, prev_pv);

        // Compute IC = cos * r, handling negative cosine
        let result = IdentityContinuityResult::new(cos, r_clamped);

        // Update history
        self.history.push(*new_pv, format!(
            "IC={:.4}, status={:?}",
            result.identity_coherence,
            result.status
        ));

        // Store result
        self.last_result = Some(result.clone());

        Ok(result)
    }

    pub fn last_result(&self) -> Option<&IdentityContinuityResult> {
        self.last_result.as_ref()
    }

    pub fn identity_coherence(&self) -> f32 {
        self.last_result
            .as_ref()
            .map(|r| r.identity_coherence)
            .unwrap_or(0.0)
    }

    pub fn current_status(&self) -> IdentityStatus {
        self.last_result
            .as_ref()
            .map(|r| r.status)
            .unwrap_or(IdentityStatus::Critical)
    }

    pub fn is_in_crisis(&self) -> bool {
        self.identity_coherence() < self.crisis_threshold
    }

    pub fn history_len(&self) -> usize {
        self.history.len()
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
| `crates/context-graph-core/src/gwt/ego_node.rs` | Add `IdentityContinuityMonitor`, `cosine_similarity_13d`, constants |

---

## Validation Criteria

| Criterion | Verification Method |
|-----------|---------------------|
| IC formula correct: cos(PV_t, PV_{t-1}) x r(t) | Unit tests with known values |
| First vector returns IC = 1.0 | Unit test |
| Zero r returns IC = 0.0 | Unit test |
| Zero vector returns IC = 0.0 | Unit test |
| Negative cosine clamped to 0 | Unit test |
| History updated after computation | Unit test |
| is_in_crisis correct at boundary | Unit test for 0.699 vs 0.7 |
| No panics from edge cases | Chaos tests |
| cosine_similarity_13d < 1us | Benchmark |

---

## Test Cases

```rust
#[cfg(test)]
mod identity_continuity_monitor_tests {
    use super::*;

    fn uniform_pv(val: f32) -> [f32; 13] {
        [val; 13]
    }

    fn orthogonal_pvs() -> ([f32; 13], [f32; 13]) {
        let mut v1 = [0.0f32; 13];
        let mut v2 = [0.0f32; 13];
        v1[0] = 1.0;
        v2[1] = 1.0;
        (v1, v2)
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = uniform_pv(0.5);
        let cos = cosine_similarity_13d(&v, &v);
        assert!((cos - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let (v1, v2) = orthogonal_pvs();
        let cos = cosine_similarity_13d(&v1, &v2);
        assert!(cos.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let v1 = uniform_pv(1.0);
        let v2 = uniform_pv(-1.0);
        let cos = cosine_similarity_13d(&v1, &v2);
        assert!((cos - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let v1 = uniform_pv(0.5);
        let v2 = [0.0f32; 13];
        let cos = cosine_similarity_13d(&v1, &v2);
        assert_eq!(cos, 0.0);
    }

    #[test]
    fn test_first_vector_is_healthy() {
        let mut monitor = IdentityContinuityMonitor::new();
        let result = monitor.compute_continuity(&uniform_pv(0.5), 0.9).unwrap();

        assert_eq!(result.identity_coherence, 1.0);
        assert_eq!(result.status, IdentityStatus::Healthy);
        assert!(!monitor.is_in_crisis());
    }

    #[test]
    fn test_identical_vectors_ic_equals_r() {
        let mut monitor = IdentityContinuityMonitor::new();
        let pv = uniform_pv(0.8);

        // First vector
        monitor.compute_continuity(&pv, 0.9).unwrap();

        // Same vector again - cos = 1.0, so IC = r
        let result = monitor.compute_continuity(&pv, 0.85).unwrap();
        assert!((result.identity_coherence - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_orthogonal_vectors_ic_zero() {
        let mut monitor = IdentityContinuityMonitor::new();
        let (v1, v2) = orthogonal_pvs();

        monitor.compute_continuity(&v1, 0.9).unwrap();
        let result = monitor.compute_continuity(&v2, 0.9).unwrap();

        assert!(result.identity_coherence.abs() < 1e-6);
        assert_eq!(result.status, IdentityStatus::Critical);
        assert!(monitor.is_in_crisis());
    }

    #[test]
    fn test_zero_r_is_critical() {
        let mut monitor = IdentityContinuityMonitor::new();

        monitor.compute_continuity(&uniform_pv(0.5), 0.9).unwrap();
        let result = monitor.compute_continuity(&uniform_pv(0.5), 0.0).unwrap();

        assert_eq!(result.identity_coherence, 0.0);
        assert_eq!(result.status, IdentityStatus::Critical);
    }

    #[test]
    fn test_negative_cosine_clamped() {
        let mut monitor = IdentityContinuityMonitor::new();

        // Opposite vectors
        monitor.compute_continuity(&uniform_pv(1.0), 0.9).unwrap();
        let result = monitor.compute_continuity(&uniform_pv(-1.0), 0.9).unwrap();

        assert!(result.identity_coherence >= 0.0);
    }

    #[test]
    fn test_history_updated() {
        let mut monitor = IdentityContinuityMonitor::new();

        assert_eq!(monitor.history_len(), 0);

        monitor.compute_continuity(&uniform_pv(0.5), 0.9).unwrap();
        assert_eq!(monitor.history_len(), 1);

        monitor.compute_continuity(&uniform_pv(0.6), 0.9).unwrap();
        assert_eq!(monitor.history_len(), 2);
    }

    #[test]
    fn test_crisis_threshold_boundary() {
        let mut monitor = IdentityContinuityMonitor::new();

        // Setup: first vector
        monitor.compute_continuity(&uniform_pv(1.0), 1.0).unwrap();

        // IC = 0.69 (just below 0.7)
        let result = monitor.compute_continuity(&uniform_pv(1.0), 0.69).unwrap();
        assert!(result.is_in_crisis());
        assert!(monitor.is_in_crisis());

        // Reset with new first vector
        let mut monitor2 = IdentityContinuityMonitor::new();
        monitor2.compute_continuity(&uniform_pv(1.0), 1.0).unwrap();

        // IC = 0.70 (at threshold)
        let result2 = monitor2.compute_continuity(&uniform_pv(1.0), 0.70).unwrap();
        assert!(!result2.is_in_crisis());
        assert!(!monitor2.is_in_crisis());
    }

    #[test]
    fn test_status_getters() {
        let mut monitor = IdentityContinuityMonitor::new();

        // Before any computation
        assert_eq!(monitor.identity_coherence(), 0.0);
        assert_eq!(monitor.current_status(), IdentityStatus::Critical);

        // After computation
        monitor.compute_continuity(&uniform_pv(0.5), 0.95).unwrap();
        assert_eq!(monitor.identity_coherence(), 1.0); // First vector
        assert_eq!(monitor.current_status(), IdentityStatus::Healthy);
    }

    #[test]
    fn test_custom_threshold() {
        let mut monitor = IdentityContinuityMonitor::with_threshold(0.8);

        monitor.compute_continuity(&uniform_pv(1.0), 1.0).unwrap();
        monitor.compute_continuity(&uniform_pv(1.0), 0.75).unwrap();

        // 0.75 < 0.8 threshold
        assert!(monitor.is_in_crisis());
    }
}
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | Claude Opus 4.5 | Initial task specification |
