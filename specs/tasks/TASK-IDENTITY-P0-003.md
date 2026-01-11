# Task Specification: Identity Continuity Monitor

**Task ID:** TASK-IDENTITY-P0-003
**Version:** 2.0.0
**Status:** Ready
**Layer:** Logic
**Sequence:** 3
**Estimated Complexity:** Medium

---

## Metadata

| Field | Value |
|-------|-------|
| Implements | REQ-IDENTITY-001, REQ-IDENTITY-002, REQ-IDENTITY-003 |
| Depends On | TASK-IDENTITY-P0-001 (COMPLETED), TASK-IDENTITY-P0-002 (COMPLETED) |
| Blocks | TASK-IDENTITY-P0-004, TASK-IDENTITY-P0-006 |
| Priority | P0 - Critical |

---

## CRITICAL: Implementation Policies

### NO Backwards Compatibility
- **FAIL FAST**: Any invalid input MUST return `CoreError` immediately
- **NO silent fallbacks**: Do not default to safe values without explicit documentation
- **Robust error logging**: All errors MUST include context (file, line, function, values)

### NO Mock Data in Tests
- ALL tests MUST use real data structures from the codebase
- Tests MUST exercise actual `PurposeVectorHistory` and `IdentityContinuity` types
- Tests MUST verify integration with existing GWT system
- NO `#[cfg(test)]` mock implementations

---

## Current Codebase State (Audited 2026-01-11)

### Verified Dependencies

| Dependency | Status | Location | Key Exports |
|------------|--------|----------|-------------|
| TASK-IDENTITY-P0-001 | **COMPLETED** | `ego_node.rs:370-460` | `IdentityContinuity::new()`, `::first_vector()`, `::is_in_crisis()`, `::is_critical()` |
| TASK-IDENTITY-P0-002 | **COMPLETED** | `ego_node.rs:81-160` | `PurposeVectorHistory`, `PurposeVectorHistoryProvider` trait, VecDeque-based ring buffer |

### Existing Types Available (ego_node.rs)

```rust
// Line 340-351: IdentityContinuity struct
pub struct IdentityContinuity {
    pub identity_coherence: f32,      // IC = cos(PV_t, PV_{t-1}) x r(t), clamped [0, 1]
    pub recent_continuity: f32,       // Cosine similarity between consecutive PVs
    pub kuramoto_order_parameter: f32, // Order parameter r from Kuramoto sync
    pub status: IdentityStatus,       // Status classification
    pub computed_at: DateTime<Utc>,   // Timestamp
}

// Line 358-368: IdentityStatus enum
pub enum IdentityStatus {
    Healthy,   // IC > 0.9
    Warning,   // 0.7 ≤ IC ≤ 0.9
    Degraded,  // IC < 0.7
    Critical,  // IC < 0.5
}

// Line 81-86: PurposeVectorHistory struct
pub struct PurposeVectorHistory {
    history: VecDeque<PurposeSnapshot>,
    pub max_size: usize,  // Default: 1000
}

// Line 91-130: PurposeVectorHistoryProvider trait
pub trait PurposeVectorHistoryProvider {
    fn push(&mut self, pv: [f32; 13], context: impl Into<String>) -> Option<[f32; 13]>;
    fn current(&self) -> Option<&[f32; 13]>;
    fn previous(&self) -> Option<&[f32; 13]>;
    fn is_empty(&self) -> bool;
    fn len(&self) -> usize;
}
```

### Existing Factory Methods (TASK-IDENTITY-P0-001)

```rust
// Line 385-403: IdentityContinuity::new()
pub fn new(purpose_continuity: f32, kuramoto_r: f32) -> Self;

// Line 424-432: IdentityContinuity::first_vector()
pub fn first_vector() -> Self;  // Returns IC=1.0, Healthy

// Line 440-442: IdentityContinuity::is_in_crisis()
pub fn is_in_crisis(&self) -> bool;  // IC < 0.7

// Line 450+: IdentityContinuity::is_critical()
pub fn is_critical(&self) -> bool;  // IC < 0.5
```

---

## Context

This task creates `IdentityContinuityMonitor`, a wrapper around `PurposeVectorHistory` that provides continuous IC monitoring using the completed P0-001 and P0-002 types.

### IC Formula (constitution.yaml line 369)
```
IC = cos(PV_t, PV_{t-1}) x r(t)
```

Where:
- `PV_t` = Current purpose vector (13D)
- `PV_{t-1}` = Previous purpose vector (13D)
- `r(t)` = Kuramoto order parameter at time t
- `IC` = Identity coherence, clamped to [0, 1]

---

## Input Context Files

| File | Purpose | Lines |
|------|---------|-------|
| `crates/context-graph-core/src/gwt/ego_node.rs` | Target implementation file | ~1300 lines |
| `crates/context-graph-core/src/gwt/mod.rs` | Module exports | Check current exports |
| `docs2/constitution.yaml` | IC formula reference | Lines 365-392 |
| `specs/tasks/TASK-IDENTITY-P0-001.md` | IdentityContinuity factory methods | COMPLETED |
| `specs/tasks/TASK-IDENTITY-P0-002.md` | PurposeVectorHistory interface | COMPLETED |

---

## Scope

### In Scope

1. Create `IdentityContinuityMonitor` struct in `ego_node.rs`
2. Implement `compute_continuity()` method using existing `IdentityContinuity::new()`
3. Create public `cosine_similarity_13d()` function (extract from private impl)
4. Handle edge cases (first vector, zero vectors, no sync)
5. Add comprehensive unit tests with REAL data only
6. Export from `mod.rs`

### Out of Scope

- Crisis detection protocol (TASK-IDENTITY-P0-004)
- Dream triggering (TASK-IDENTITY-P0-005)
- Workspace event subscription (TASK-IDENTITY-P0-006)
- MCP tool integration (TASK-IDENTITY-P0-007)

---

## Definition of Done

### Exact Signatures Required

```rust
// File: crates/context-graph-core/src/gwt/ego_node.rs

/// Default crisis threshold per constitution.yaml line 369
pub const IC_CRISIS_THRESHOLD: f32 = 0.7;

/// Critical threshold triggering dream
pub const IC_CRITICAL_THRESHOLD: f32 = 0.5;

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
/// # Errors
/// This function does NOT return errors - zero magnitude is handled gracefully.
///
/// # Performance
/// Constraint: < 1us for 13D vectors
pub fn cosine_similarity_13d(v1: &[f32; 13], v2: &[f32; 13]) -> f32;

/// Monitor for continuous identity coherence tracking
///
/// Computes IC = cos(PV_t, PV_{t-1}) x r(t) per constitution.yaml line 369.
/// Wraps PurposeVectorHistory and uses IdentityContinuity for results.
///
/// # Example
/// ```ignore
/// let mut monitor = IdentityContinuityMonitor::new();
/// let result = monitor.compute_continuity(&new_pv, kuramoto_r)?;
/// if result.is_in_crisis() {
///     // Handle identity crisis
/// }
/// ```
#[derive(Debug)]
pub struct IdentityContinuityMonitor {
    /// Purpose vector history (from TASK-IDENTITY-P0-002)
    history: PurposeVectorHistory,
    /// Last computed result
    last_result: Option<IdentityContinuity>,
    /// Crisis threshold (default: 0.7 per constitution)
    crisis_threshold: f32,
}

impl IdentityContinuityMonitor {
    /// Create new monitor with default crisis threshold (0.7)
    pub fn new() -> Self;

    /// Create with custom crisis threshold
    ///
    /// # Arguments
    /// * `threshold` - Crisis threshold, clamped to [0.0, 1.0]
    pub fn with_threshold(threshold: f32) -> Self;

    /// Create with custom history capacity
    ///
    /// # Arguments
    /// * `max_history_size` - Maximum entries in purpose vector history
    pub fn with_capacity(max_history_size: usize) -> Self;

    /// Compute identity continuity for a new purpose vector
    ///
    /// # Algorithm
    /// 1. Check for empty history -> return IdentityContinuity::first_vector()
    /// 2. Get previous PV from history via PurposeVectorHistoryProvider::current()
    /// 3. Compute cos(PV_t, PV_{t-1}) using cosine_similarity_13d()
    /// 4. Create IdentityContinuity::new(cos, kuramoto_r) (handles clamping)
    /// 5. Push new PV to history via PurposeVectorHistoryProvider::push()
    /// 6. Store and return result
    ///
    /// # Arguments
    /// * `new_pv` - The new purpose vector PV_t (13D)
    /// * `kuramoto_r` - Current Kuramoto order parameter [0, 1]
    ///
    /// # Returns
    /// * `Ok(IdentityContinuity)` on success
    /// * `Err(CoreError)` on internal errors (fail-fast)
    ///
    /// # Edge Cases
    /// - First vector: Returns IdentityContinuity::first_vector() (IC=1.0, Healthy)
    /// - Zero vector: Returns IC=0.0, Critical (cosine_similarity_13d returns 0.0)
    /// - r=0: Returns IC=0.0, Critical (IdentityContinuity::new handles clamping)
    pub fn compute_continuity(
        &mut self,
        new_pv: &[f32; 13],
        kuramoto_r: f32,
    ) -> CoreResult<IdentityContinuity>;

    /// Get the last computed result, if any
    pub fn last_result(&self) -> Option<&IdentityContinuity>;

    /// Get current identity coherence value (0.0 if no computation yet)
    pub fn identity_coherence(&self) -> f32;

    /// Get current identity status (Critical if no computation yet)
    pub fn current_status(&self) -> IdentityStatus;

    /// Check if currently in crisis (IC < threshold)
    pub fn is_in_crisis(&self) -> bool;

    /// Get the history length
    pub fn history_len(&self) -> usize;

    /// Get the crisis threshold
    pub fn crisis_threshold(&self) -> f32;
}

impl Default for IdentityContinuityMonitor {
    fn default() -> Self {
        Self::new()
    }
}
```

### Module Export (mod.rs)

Add to existing exports in `crates/context-graph-core/src/gwt/mod.rs`:

```rust
pub use ego_node::{
    // Existing exports...
    IdentityContinuity, IdentityStatus, SelfAwarenessLoop, SelfEgoNode, SelfReflectionResult,
    // NEW exports for P0-003:
    IdentityContinuityMonitor, cosine_similarity_13d, IC_CRISIS_THRESHOLD, IC_CRITICAL_THRESHOLD,
};
```

---

## Constraints (FAIL FAST)

1. IC MUST be clamped to [0.0, 1.0] - handled by `IdentityContinuity::new()`
2. Negative cosine values result in IC = 0.0 - handled by `IdentityContinuity::new()`
3. Zero magnitude vectors MUST return cosine = 0.0 (no division by zero panic)
4. `cosine_similarity_13d` MUST be numerically stable (use EPSILON = 1e-8)
5. `compute_continuity` MUST call `history.push()` after computation
6. `compute_continuity` MUST return `IdentityContinuity::first_vector()` for empty history
7. **NO panics from any input combination**
8. Performance: `cosine_similarity_13d` < 1us

---

## Full State Verification (FSV) Requirements

### 1. Source of Truth

| Artifact | Location | Verification Method |
|----------|----------|---------------------|
| IdentityContinuityMonitor struct | `ego_node.rs` | `grep "pub struct IdentityContinuityMonitor"` |
| cosine_similarity_13d function | `ego_node.rs` | `grep "pub fn cosine_similarity_13d"` |
| IC_CRISIS_THRESHOLD constant | `ego_node.rs` | `grep "pub const IC_CRISIS_THRESHOLD"` |
| IC_CRITICAL_THRESHOLD constant | `ego_node.rs` | `grep "pub const IC_CRITICAL_THRESHOLD"` |
| Module exports | `mod.rs` | `grep "IdentityContinuityMonitor"` in mod.rs |

### 2. Execute & Inspect (Separate Read Operations)

After implementation, verify with SEPARATE commands (not piped):

```bash
# Step 1: Build succeeds
cargo build -p context-graph-core 2>&1 | tee /tmp/p003_build.log
echo "BUILD_EXIT_CODE: $?"

# Step 2: Tests pass
cargo test -p context-graph-core identity_continuity_monitor 2>&1 | tee /tmp/p003_test.log
echo "TEST_EXIT_CODE: $?"

# Step 3: Clippy clean
cargo clippy -p context-graph-core -- -D warnings 2>&1 | tee /tmp/p003_clippy.log
echo "CLIPPY_EXIT_CODE: $?"

# Step 4: Verify struct exists
grep -n "pub struct IdentityContinuityMonitor" crates/context-graph-core/src/gwt/ego_node.rs

# Step 5: Verify function exists
grep -n "pub fn cosine_similarity_13d" crates/context-graph-core/src/gwt/ego_node.rs

# Step 6: Verify exports
grep -n "IdentityContinuityMonitor" crates/context-graph-core/src/gwt/mod.rs
```

### 3. Boundary & Edge Case Audit (Minimum 3)

| Edge Case | Input | Expected Output | Test Name |
|-----------|-------|-----------------|-----------|
| Empty history (first vector) | Any valid PV, any r | IC=1.0, Healthy | `test_first_vector_returns_healthy` |
| Zero magnitude vector | `[0.0; 13]`, r=0.9 | IC=0.0, Critical | `test_zero_vector_returns_critical` |
| Zero Kuramoto r | Valid PV, r=0.0 | IC=0.0, Critical | `test_zero_kuramoto_returns_critical` |
| Opposite vectors (cos=-1) | v1, -v1, r=1.0 | IC=0.0, Critical | `test_opposite_vectors_clamped_to_zero` |
| Crisis boundary (IC=0.699) | cos=1.0, r=0.699 | is_in_crisis()=true | `test_crisis_boundary_below` |
| Crisis boundary (IC=0.700) | cos=1.0, r=0.700 | is_in_crisis()=false | `test_crisis_boundary_at` |

### 4. Evidence of Success Logs

The implementation MUST log the following on successful completion:

```
[INFO] IdentityContinuityMonitor initialized with threshold=0.7
[DEBUG] compute_continuity: first_vector case, returning IC=1.0
[DEBUG] compute_continuity: cos=0.95, r=0.88, IC=0.836, status=Warning
[DEBUG] compute_continuity: history_len=5, is_in_crisis=false
```

Use `tracing` crate with appropriate log levels.

---

## Manual Verification Checklist

After implementation, manually verify:

- [ ] `cargo build -p context-graph-core` succeeds with 0 errors
- [ ] `cargo test -p context-graph-core identity_continuity_monitor` shows all tests passing
- [ ] `cargo clippy -p context-graph-core -- -D warnings` shows 0 warnings
- [ ] `grep "pub struct IdentityContinuityMonitor" ego_node.rs` returns line number
- [ ] `grep "pub fn cosine_similarity_13d" ego_node.rs` returns line number
- [ ] `grep "IdentityContinuityMonitor" mod.rs` confirms export
- [ ] Run integration test that creates monitor, computes 3 continuities, verifies history_len=3

---

## Pseudo Code

```rust
pub const IC_CRISIS_THRESHOLD: f32 = 0.7;
pub const IC_CRITICAL_THRESHOLD: f32 = 0.5;

/// Epsilon for numerical stability in magnitude comparisons
const COSINE_EPSILON: f32 = 1e-8;

pub fn cosine_similarity_13d(v1: &[f32; 13], v2: &[f32; 13]) -> f32 {
    let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let mag1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Handle zero magnitude vectors - no division by zero
    if mag1 < COSINE_EPSILON || mag2 < COSINE_EPSILON {
        return 0.0;
    }

    dot / (mag1 * mag2)
}

#[derive(Debug)]
pub struct IdentityContinuityMonitor {
    history: PurposeVectorHistory,
    last_result: Option<IdentityContinuity>,
    crisis_threshold: f32,
}

impl IdentityContinuityMonitor {
    pub fn new() -> Self {
        Self::with_threshold(IC_CRISIS_THRESHOLD)
    }

    pub fn with_threshold(threshold: f32) -> Self {
        tracing::info!("IdentityContinuityMonitor initialized with threshold={}", threshold);
        Self {
            history: PurposeVectorHistory::new(),
            last_result: None,
            crisis_threshold: threshold.clamp(0.0, 1.0),
        }
    }

    pub fn with_capacity(max_history_size: usize) -> Self {
        Self {
            history: PurposeVectorHistory::with_capacity(max_history_size),
            last_result: None,
            crisis_threshold: IC_CRISIS_THRESHOLD,
        }
    }

    pub fn compute_continuity(
        &mut self,
        new_pv: &[f32; 13],
        kuramoto_r: f32,
    ) -> CoreResult<IdentityContinuity> {
        // Edge case: first vector
        if self.history.is_empty() {
            let result = IdentityContinuity::first_vector();
            self.history.push(*new_pv, "Initial purpose vector");
            self.last_result = Some(result.clone());
            tracing::debug!("compute_continuity: first_vector case, returning IC=1.0");
            return Ok(result);
        }

        // Get previous PV (guaranteed to exist since history not empty)
        let prev_pv = self.history.current()
            .ok_or_else(|| CoreError::internal("History inconsistency: current() returned None for non-empty history"))?;

        // Compute cosine similarity
        let cos = cosine_similarity_13d(new_pv, prev_pv);

        // Create result using existing factory (handles clamping)
        let result = IdentityContinuity::new(cos, kuramoto_r);

        // Update history with context
        self.history.push(*new_pv, format!(
            "IC={:.4}, status={:?}",
            result.identity_coherence,
            result.status
        ));

        tracing::debug!(
            "compute_continuity: cos={:.4}, r={:.4}, IC={:.4}, status={:?}",
            cos, kuramoto_r, result.identity_coherence, result.status
        );

        // Store result
        self.last_result = Some(result.clone());

        tracing::debug!(
            "compute_continuity: history_len={}, is_in_crisis={}",
            self.history.len(),
            self.is_in_crisis()
        );

        Ok(result)
    }

    pub fn last_result(&self) -> Option<&IdentityContinuity> {
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

    pub fn crisis_threshold(&self) -> f32 {
        self.crisis_threshold
    }
}

impl Default for IdentityContinuityMonitor {
    fn default() -> Self {
        Self::new()
    }
}
```

---

## Test Cases (NO MOCK DATA)

```rust
#[cfg(test)]
mod identity_continuity_monitor_tests {
    use super::*;

    // Helper: Create a uniform purpose vector (real data)
    fn uniform_pv(val: f32) -> [f32; 13] {
        [val; 13]
    }

    // Helper: Create orthogonal purpose vectors (real data)
    fn orthogonal_pvs() -> ([f32; 13], [f32; 13]) {
        let mut v1 = [0.0f32; 13];
        let mut v2 = [0.0f32; 13];
        v1[0] = 1.0;  // Unit vector along first axis
        v2[1] = 1.0;  // Unit vector along second axis
        (v1, v2)
    }

    // ===== cosine_similarity_13d tests =====

    #[test]
    fn test_cosine_identical_vectors() {
        let v = uniform_pv(0.5);
        let cos = cosine_similarity_13d(&v, &v);
        assert!((cos - 1.0).abs() < 1e-6, "Identical vectors should have cos=1.0, got {}", cos);
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let (v1, v2) = orthogonal_pvs();
        let cos = cosine_similarity_13d(&v1, &v2);
        assert!(cos.abs() < 1e-6, "Orthogonal vectors should have cos=0.0, got {}", cos);
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let v1 = uniform_pv(1.0);
        let v2 = uniform_pv(-1.0);
        let cos = cosine_similarity_13d(&v1, &v2);
        assert!((cos - (-1.0)).abs() < 1e-6, "Opposite vectors should have cos=-1.0, got {}", cos);
    }

    #[test]
    fn test_cosine_zero_vector_first() {
        let v1 = [0.0f32; 13];
        let v2 = uniform_pv(0.5);
        let cos = cosine_similarity_13d(&v1, &v2);
        assert_eq!(cos, 0.0, "Zero first vector should return 0.0");
    }

    #[test]
    fn test_cosine_zero_vector_second() {
        let v1 = uniform_pv(0.5);
        let v2 = [0.0f32; 13];
        let cos = cosine_similarity_13d(&v1, &v2);
        assert_eq!(cos, 0.0, "Zero second vector should return 0.0");
    }

    #[test]
    fn test_cosine_both_zero_vectors() {
        let v1 = [0.0f32; 13];
        let v2 = [0.0f32; 13];
        let cos = cosine_similarity_13d(&v1, &v2);
        assert_eq!(cos, 0.0, "Both zero vectors should return 0.0");
    }

    // ===== IdentityContinuityMonitor tests =====

    #[test]
    fn test_first_vector_returns_healthy() {
        let mut monitor = IdentityContinuityMonitor::new();
        let result = monitor.compute_continuity(&uniform_pv(0.5), 0.9).unwrap();

        assert_eq!(result.identity_coherence, 1.0, "First vector should have IC=1.0");
        assert_eq!(result.status, IdentityStatus::Healthy, "First vector should be Healthy");
        assert!(!monitor.is_in_crisis(), "First vector should not be in crisis");
        assert_eq!(monitor.history_len(), 1, "History should have 1 entry");
    }

    #[test]
    fn test_identical_vectors_ic_equals_r() {
        let mut monitor = IdentityContinuityMonitor::new();
        let pv = uniform_pv(0.8);

        // First vector
        monitor.compute_continuity(&pv, 0.9).unwrap();

        // Same vector again - cos=1.0, so IC=r
        let result = monitor.compute_continuity(&pv, 0.85).unwrap();
        assert!(
            (result.identity_coherence - 0.85).abs() < 1e-6,
            "IC should equal r when cos=1.0, got {}",
            result.identity_coherence
        );
    }

    #[test]
    fn test_orthogonal_vectors_critical() {
        let mut monitor = IdentityContinuityMonitor::new();
        let (v1, v2) = orthogonal_pvs();

        monitor.compute_continuity(&v1, 0.9).unwrap();
        let result = monitor.compute_continuity(&v2, 0.9).unwrap();

        assert!(result.identity_coherence.abs() < 1e-6, "Orthogonal vectors should give IC=0");
        assert_eq!(result.status, IdentityStatus::Critical, "IC=0 should be Critical");
        assert!(monitor.is_in_crisis(), "IC=0 should be in crisis");
    }

    #[test]
    fn test_zero_kuramoto_returns_critical() {
        let mut monitor = IdentityContinuityMonitor::new();

        monitor.compute_continuity(&uniform_pv(0.5), 0.9).unwrap();
        let result = monitor.compute_continuity(&uniform_pv(0.5), 0.0).unwrap();

        assert_eq!(result.identity_coherence, 0.0, "r=0 should give IC=0");
        assert_eq!(result.status, IdentityStatus::Critical, "IC=0 should be Critical");
    }

    #[test]
    fn test_opposite_vectors_clamped_to_zero() {
        let mut monitor = IdentityContinuityMonitor::new();

        monitor.compute_continuity(&uniform_pv(1.0), 0.9).unwrap();
        let result = monitor.compute_continuity(&uniform_pv(-1.0), 0.9).unwrap();

        assert!(result.identity_coherence >= 0.0, "IC should be clamped to >=0");
        assert_eq!(result.identity_coherence, 0.0, "Opposite vectors with negative cos should give IC=0");
    }

    #[test]
    fn test_zero_vector_returns_critical() {
        let mut monitor = IdentityContinuityMonitor::new();

        monitor.compute_continuity(&uniform_pv(0.5), 0.9).unwrap();
        let result = monitor.compute_continuity(&[0.0f32; 13], 0.9).unwrap();

        assert_eq!(result.identity_coherence, 0.0, "Zero vector should give IC=0");
        assert_eq!(result.status, IdentityStatus::Critical);
    }

    #[test]
    fn test_history_updated_correctly() {
        let mut monitor = IdentityContinuityMonitor::new();

        assert_eq!(monitor.history_len(), 0, "Empty monitor should have history_len=0");

        monitor.compute_continuity(&uniform_pv(0.5), 0.9).unwrap();
        assert_eq!(monitor.history_len(), 1);

        monitor.compute_continuity(&uniform_pv(0.6), 0.9).unwrap();
        assert_eq!(monitor.history_len(), 2);

        monitor.compute_continuity(&uniform_pv(0.7), 0.9).unwrap();
        assert_eq!(monitor.history_len(), 3);
    }

    #[test]
    fn test_crisis_boundary_below() {
        let mut monitor = IdentityContinuityMonitor::new();

        monitor.compute_continuity(&uniform_pv(1.0), 1.0).unwrap();
        // IC = cos(1.0) * 0.699 = 1.0 * 0.699 = 0.699 < 0.7
        let result = monitor.compute_continuity(&uniform_pv(1.0), 0.699).unwrap();

        assert!(result.is_in_crisis(), "IC=0.699 should be in crisis (< 0.7)");
        assert!(monitor.is_in_crisis());
    }

    #[test]
    fn test_crisis_boundary_at() {
        let mut monitor = IdentityContinuityMonitor::new();

        monitor.compute_continuity(&uniform_pv(1.0), 1.0).unwrap();
        // IC = cos(1.0) * 0.70 = 1.0 * 0.70 = 0.70 >= 0.7
        let result = monitor.compute_continuity(&uniform_pv(1.0), 0.70).unwrap();

        assert!(!result.is_in_crisis(), "IC=0.70 should NOT be in crisis (>= 0.7)");
        assert!(!monitor.is_in_crisis());
    }

    #[test]
    fn test_custom_threshold() {
        let mut monitor = IdentityContinuityMonitor::with_threshold(0.8);
        assert_eq!(monitor.crisis_threshold(), 0.8);

        monitor.compute_continuity(&uniform_pv(1.0), 1.0).unwrap();
        monitor.compute_continuity(&uniform_pv(1.0), 0.75).unwrap();

        // 0.75 < 0.8 threshold
        assert!(monitor.is_in_crisis(), "IC=0.75 should be in crisis with threshold=0.8");
    }

    #[test]
    fn test_threshold_clamped() {
        let monitor = IdentityContinuityMonitor::with_threshold(1.5);
        assert_eq!(monitor.crisis_threshold(), 1.0, "Threshold should be clamped to 1.0");

        let monitor2 = IdentityContinuityMonitor::with_threshold(-0.5);
        assert_eq!(monitor2.crisis_threshold(), 0.0, "Threshold should be clamped to 0.0");
    }

    #[test]
    fn test_getters_before_computation() {
        let monitor = IdentityContinuityMonitor::new();

        assert_eq!(monitor.identity_coherence(), 0.0, "No computation yet should return 0.0");
        assert_eq!(monitor.current_status(), IdentityStatus::Critical, "No computation should be Critical");
        assert!(monitor.last_result().is_none(), "No computation should have no result");
        assert!(monitor.is_in_crisis(), "No computation should be in crisis");
    }

    #[test]
    fn test_getters_after_computation() {
        let mut monitor = IdentityContinuityMonitor::new();

        monitor.compute_continuity(&uniform_pv(0.5), 0.95).unwrap();

        assert_eq!(monitor.identity_coherence(), 1.0, "First vector gives IC=1.0");
        assert_eq!(monitor.current_status(), IdentityStatus::Healthy);
        assert!(monitor.last_result().is_some());
        assert!(!monitor.is_in_crisis());
    }

    #[test]
    fn test_default_impl() {
        let monitor = IdentityContinuityMonitor::default();
        assert_eq!(monitor.crisis_threshold(), IC_CRISIS_THRESHOLD);
    }

    // ===== Integration test with real sequence =====

    #[test]
    fn test_realistic_sequence() {
        let mut monitor = IdentityContinuityMonitor::new();

        // Simulate a sequence of purpose vectors with gradual drift
        let pv1 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let pv2 = [0.51, 0.49, 0.52, 0.48, 0.51, 0.49, 0.50, 0.51, 0.49, 0.50, 0.51, 0.49, 0.50];
        let pv3 = [0.52, 0.48, 0.53, 0.47, 0.52, 0.48, 0.50, 0.52, 0.48, 0.50, 0.52, 0.48, 0.50];

        // First vector
        let r1 = monitor.compute_continuity(&pv1, 0.95).unwrap();
        assert_eq!(r1.status, IdentityStatus::Healthy);

        // Small drift - should remain healthy
        let r2 = monitor.compute_continuity(&pv2, 0.94).unwrap();
        assert!(r2.identity_coherence > 0.9, "Small drift should maintain high IC");

        // Continued small drift
        let r3 = monitor.compute_continuity(&pv3, 0.93).unwrap();
        assert!(r3.identity_coherence > 0.85, "Continued small drift should stay above warning");

        assert_eq!(monitor.history_len(), 3);
    }
}
```

---

## Files to Create

None - all additions go to existing files.

---

## Files to Modify

| File | Changes |
|------|---------|
| `crates/context-graph-core/src/gwt/ego_node.rs` | Add `IdentityContinuityMonitor`, `cosine_similarity_13d`, constants, tests |
| `crates/context-graph-core/src/gwt/mod.rs` | Add exports for new types |

---

## Implementation Checklist

### Constants & Functions
- [ ] Define `IC_CRISIS_THRESHOLD` constant (0.7)
- [ ] Define `IC_CRITICAL_THRESHOLD` constant (0.5)
- [ ] Define `COSINE_EPSILON` constant (1e-8)
- [ ] Implement `cosine_similarity_13d()` function
- [ ] Handle zero magnitude vectors (return 0.0, no panic)

### IdentityContinuityMonitor Struct
- [ ] Define struct with `history`, `last_result`, `crisis_threshold` fields
- [ ] Implement `new()` with default threshold
- [ ] Implement `with_threshold()` with clamping
- [ ] Implement `with_capacity()` for custom history size
- [ ] Implement `Default` trait

### compute_continuity Method
- [ ] Handle first vector case using `IdentityContinuity::first_vector()`
- [ ] Get previous PV via `PurposeVectorHistoryProvider::current()`
- [ ] Compute cosine similarity using `cosine_similarity_13d()`
- [ ] Create result using `IdentityContinuity::new(cos, kuramoto_r)`
- [ ] Update history via `history.push()`
- [ ] Store `last_result`
- [ ] Add tracing debug logs

### Getter Methods
- [ ] Implement `last_result()`
- [ ] Implement `identity_coherence()`
- [ ] Implement `current_status()`
- [ ] Implement `is_in_crisis()`
- [ ] Implement `history_len()`
- [ ] Implement `crisis_threshold()`

### Exports
- [ ] Add `IdentityContinuityMonitor` to `mod.rs` exports
- [ ] Add `cosine_similarity_13d` to `mod.rs` exports
- [ ] Add `IC_CRISIS_THRESHOLD` to `mod.rs` exports
- [ ] Add `IC_CRITICAL_THRESHOLD` to `mod.rs` exports

### Tests (NO MOCK DATA)
- [ ] Test `cosine_similarity_13d` with identical vectors
- [ ] Test `cosine_similarity_13d` with orthogonal vectors
- [ ] Test `cosine_similarity_13d` with opposite vectors
- [ ] Test `cosine_similarity_13d` with zero vectors
- [ ] Test first vector returns Healthy
- [ ] Test identical vectors give IC=r
- [ ] Test orthogonal vectors give Critical
- [ ] Test zero Kuramoto r gives Critical
- [ ] Test opposite vectors clamped to 0
- [ ] Test history updated correctly
- [ ] Test crisis boundary (0.699 vs 0.700)
- [ ] Test custom threshold
- [ ] Test getters before/after computation
- [ ] Test realistic sequence integration

### Verification
- [ ] `cargo build` succeeds
- [ ] `cargo test identity_continuity_monitor` passes
- [ ] `cargo clippy -- -D warnings` clean
- [ ] Verify exports in `mod.rs`

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | Claude Opus 4.5 | Initial task specification |
| 1.1.0 | 2026-01-11 | Claude Opus 4.5 | Added implementation checklist and relationship to existing code |
| 2.0.0 | 2026-01-11 | Claude Opus 4.5 | Complete rewrite: FSV requirements, no mock data policy, fail-fast, accurate codebase audit, manual verification checklist |
