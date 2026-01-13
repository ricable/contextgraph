# Task Specification: Crisis Detection

**Task ID:** TASK-IDENTITY-P0-004
**Version:** 2.0.0
**Status:** **COMPLETED** (2026-01-12)
**Layer:** Logic
**Sequence:** 4
**Estimated Complexity:** Low

---

## CRITICAL POLICIES - READ FIRST

### NO Backwards Compatibility
- **FAIL FAST**: Invalid input MUST return `CoreError` immediately
- **NO silent fallbacks**: Do not default to safe values when errors occur
- **Robust error logging**: All errors MUST include context (file, line, function, values)
- **NO workarounds**: If something fails, it errors out with clear diagnostics

### NO Mock Data in Tests
- ALL tests MUST use REAL data structures (`IdentityContinuityMonitor`, `IdentityStatus`)
- Tests MUST verify actual state changes in the monitor
- NO `#[cfg(test)]` mock implementations that hide broken functionality
- Tests must prove the system works end-to-end

---

## Codebase Audit (2026-01-12)

### WHAT EXISTS (Verified with exact file paths and line numbers):

| Component | Location | Line(s) | Status |
|-----------|----------|---------|--------|
| `IdentityContinuityMonitor` | `gwt/ego_node/monitor.rs` | 23-178 | EXISTS |
| `compute_continuity()` method | `gwt/ego_node/monitor.rs` | 87-115 | EXISTS |
| `identity_coherence()` getter | `gwt/ego_node/monitor.rs` | 125-127 | EXISTS |
| `current_status()` getter | `gwt/ego_node/monitor.rs` | 130-132 | EXISTS |
| `is_in_crisis()` getter | `gwt/ego_node/monitor.rs` | 135-142 | EXISTS |
| `IdentityStatus` enum | `gwt/ego_node/types.rs` | 44-54 | EXISTS (Healthy/Warning/Degraded/Critical) |
| `IC_CRISIS_THRESHOLD` | `gwt/ego_node/types.rs` | 14 | EXISTS (0.7) |
| `IC_CRITICAL_THRESHOLD` | `gwt/ego_node/types.rs` | 18 | EXISTS (0.5) |
| `IdentityContinuity` struct | `gwt/ego_node/identity_continuity.rs` | 21-32 | EXISTS |

### WHAT DOES NOT EXIST (Must be created by this task):

| Component | Expected Location | Status |
|-----------|-------------------|--------|
| `CRISIS_EVENT_COOLDOWN` constant | `gwt/ego_node/types.rs` | **DOES NOT EXIST** |
| `CrisisDetectionResult` struct | `gwt/ego_node/monitor.rs` | **DOES NOT EXIST** |
| `detect_crisis()` method | `IdentityContinuityMonitor` | **DOES NOT EXIST** |
| `previous_status` field | `IdentityContinuityMonitor` | **DOES NOT EXIST** |
| `last_event_time` field | `IdentityContinuityMonitor` | **DOES NOT EXIST** |
| `mark_event_emitted()` method | `IdentityContinuityMonitor` | **DOES NOT EXIST** |
| `previous_status()` getter | `IdentityContinuityMonitor` | **DOES NOT EXIST** |
| `status_changed()` method | `IdentityContinuityMonitor` | **DOES NOT EXIST** |
| `entering_critical()` method | `IdentityContinuityMonitor` | **DOES NOT EXIST** |

---

## Metadata

| Field | Value |
|-------|-------|
| Implements | IDENTITY-004 from constitution.yaml |
| Depends On | TASK-IDENTITY-P0-003 (COMPLETED) |
| Blocks | TASK-IDENTITY-P0-005, TASK-IDENTITY-P0-006 |
| Priority | P0 - Critical |

---

## Context

### What This Task Does

Once identity continuity is computed via `compute_continuity()`, we need to detect when the system **transitions** between identity states. This task adds:

1. **Crisis detection result struct** - Captures all transition information
2. **Status transition tracking** - Tracks previous vs current status
3. **Cooldown mechanism** - Prevents event spam during IC fluctuations
4. **Helper methods** - For downstream consumers (P0-005, P0-006)

### Constitution Reference

Per constitution.yaml lines 387-392 (identity thresholds):
- `IC > 0.9`: Healthy
- `0.7 <= IC <= 0.9`: Warning
- `0.5 <= IC < 0.7`: Degraded
- `IC < 0.5`: Critical (triggers dream consolidation)

### Why This Matters

Without crisis detection:
- P0-005 (Crisis Protocol) cannot know when to execute
- P0-006 (GWT Attention Wiring) cannot emit `WorkspaceEvent::IdentityCritical`
- Dream system cannot receive identity crisis triggers
- No cooldown = event spam crashes downstream listeners

---

## Prerequisites

- [x] TASK-IDENTITY-P0-003 completed (`IdentityContinuityMonitor` exists)
- [x] `IdentityContinuityMonitor` has `compute_continuity()` method
- [x] `IdentityStatus` enum exists with all 4 variants
- [x] `IC_CRISIS_THRESHOLD` and `IC_CRITICAL_THRESHOLD` constants exist

---

## Scope

### In Scope

1. Add `CRISIS_EVENT_COOLDOWN` constant (30 seconds)
2. Add `CrisisDetectionResult` struct with all transition fields
3. Add `previous_status: IdentityStatus` field to monitor
4. Add `last_event_time: Option<Instant>` field to monitor
5. Implement `detect_crisis()` method
6. Implement `mark_event_emitted()` method
7. Implement helper getters: `previous_status()`, `status_changed()`, `entering_critical()`
8. Add `status_ordinal()` helper function
9. Unit tests for ALL state transitions

### Out of Scope

- Crisis protocol execution (TASK-IDENTITY-P0-005)
- Workspace event emission (TASK-IDENTITY-P0-006)
- Dream integration (Dream subsystem)

---

## Definition of Done

### 1. Add Constant to `types.rs`

```rust
// File: crates/context-graph-core/src/gwt/ego_node/types.rs
// ADD AFTER LINE 22 (after COSINE_EPSILON):

use std::time::Duration;

/// Minimum time between crisis event emissions (30 seconds)
/// Prevents event spam during IC fluctuations
/// Per constitution.yaml: throttle workspace events
pub const CRISIS_EVENT_COOLDOWN: Duration = Duration::from_secs(30);
```

### 2. Add `CrisisDetectionResult` Struct

```rust
// File: crates/context-graph-core/src/gwt/ego_node/monitor.rs
// ADD AFTER imports:

use std::time::{Duration, Instant};
use super::types::CRISIS_EVENT_COOLDOWN;

/// Result of crisis detection analysis
///
/// Contains all information needed by CrisisProtocol (P0-005)
/// to decide what actions to take.
///
/// # Fields
/// - `identity_coherence`: Current IC value (0.0-1.0)
/// - `previous_status`: Status before this detection
/// - `current_status`: Status after this detection
/// - `status_changed`: True if status transitioned
/// - `entering_crisis`: True if transitioned FROM Healthy to any lower state
/// - `entering_critical`: True if transitioned TO Critical from any other state
/// - `recovering`: True if status improved (lower ordinal -> higher ordinal)
/// - `time_since_last_event`: Time since last crisis event was emitted
/// - `can_emit_event`: True if cooldown allows new event emission
#[derive(Debug, Clone, PartialEq)]
pub struct CrisisDetectionResult {
    /// Current IC value
    pub identity_coherence: f32,
    /// Previous status (before this computation)
    pub previous_status: IdentityStatus,
    /// Current status (after this computation)
    pub current_status: IdentityStatus,
    /// Whether status changed
    pub status_changed: bool,
    /// Whether entering crisis (transition from Healthy to Warning/Degraded/Critical)
    pub entering_crisis: bool,
    /// Whether entering critical (transition to Critical specifically)
    pub entering_critical: bool,
    /// Whether recovering (transition from lower to higher status)
    pub recovering: bool,
    /// Time since last crisis event emission
    pub time_since_last_event: Option<Duration>,
    /// Whether cooldown allows event emission
    pub can_emit_event: bool,
}
```

### 3. Update `IdentityContinuityMonitor` Struct

```rust
// File: crates/context-graph-core/src/gwt/ego_node/monitor.rs
// MODIFY the struct definition (lines 23-31):

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityContinuityMonitor {
    /// Purpose vector history buffer
    history: PurposeVectorHistory,
    /// Cached last computation result
    last_result: Option<IdentityContinuity>,
    /// Configurable crisis threshold (default: IC_CRISIS_THRESHOLD = 0.7)
    crisis_threshold: f32,

    // === TASK-IDENTITY-P0-004: NEW FIELDS ===
    /// Previous status for transition detection (default: Healthy)
    #[serde(default = "default_healthy_status")]
    previous_status: IdentityStatus,
    /// Last time a crisis event was emitted (not serialized)
    #[serde(skip)]
    last_event_time: Option<Instant>,
}

/// Default status for deserialization
fn default_healthy_status() -> IdentityStatus {
    IdentityStatus::Healthy
}
```

### 4. Update Constructor Methods

```rust
// File: crates/context-graph-core/src/gwt/ego_node/monitor.rs
// MODIFY new() method:

pub fn new() -> Self {
    Self {
        history: PurposeVectorHistory::new(),
        last_result: None,
        crisis_threshold: IC_CRISIS_THRESHOLD,
        previous_status: IdentityStatus::Healthy, // TASK-P0-004
        last_event_time: None,                    // TASK-P0-004
    }
}

// Also update with_threshold() and with_capacity() similarly
```

### 5. Implement `detect_crisis()` Method

```rust
// File: crates/context-graph-core/src/gwt/ego_node/monitor.rs
// ADD after existing methods (around line 165):

impl IdentityContinuityMonitor {
    /// Detect crisis state and track transitions
    ///
    /// # Algorithm
    /// 1. Get current status from last_result
    /// 2. Compare with previous_status to detect transitions
    /// 3. Compute entering_crisis (from Healthy to lower)
    /// 4. Compute entering_critical (to Critical from any other)
    /// 5. Compute recovering (status improvement)
    /// 6. Check cooldown for event emission
    /// 7. Update previous_status for next call
    ///
    /// # Returns
    /// `CrisisDetectionResult` with all transition information
    ///
    /// # Panics
    /// Never panics. Returns default result if no computation has occurred.
    pub fn detect_crisis(&mut self) -> CrisisDetectionResult {
        // Get current values
        let current_status = self.current_status().unwrap_or(IdentityStatus::Healthy);
        let ic = self.identity_coherence().unwrap_or(1.0);
        let prev_status = self.previous_status;

        // Detect transitions
        let status_changed = current_status != prev_status;

        // Entering crisis = transitioning FROM Healthy to any lower state
        let entering_crisis = status_changed
            && prev_status == IdentityStatus::Healthy
            && current_status != IdentityStatus::Healthy;

        // Entering critical = transitioning TO Critical from any other state
        let entering_critical = status_changed
            && current_status == IdentityStatus::Critical
            && prev_status != IdentityStatus::Critical;

        // Recovering = improving status (lower ordinal to higher ordinal)
        let recovering = status_changed
            && status_ordinal(current_status) > status_ordinal(prev_status);

        // Cooldown check
        let time_since_last_event = self.last_event_time.map(|t| t.elapsed());
        let can_emit_event = match time_since_last_event {
            None => true, // No previous event, can emit
            Some(elapsed) => elapsed >= CRISIS_EVENT_COOLDOWN,
        };

        // Update previous_status for next detection
        self.previous_status = current_status;

        CrisisDetectionResult {
            identity_coherence: ic,
            previous_status: prev_status,
            current_status,
            status_changed,
            entering_crisis,
            entering_critical,
            recovering,
            time_since_last_event,
            can_emit_event,
        }
    }

    /// Get the previous status (before last detection)
    #[inline]
    pub fn previous_status(&self) -> IdentityStatus {
        self.previous_status
    }

    /// Check if status changed in last detection
    #[inline]
    pub fn status_changed(&self) -> bool {
        self.current_status()
            .map(|curr| curr != self.previous_status)
            .unwrap_or(false)
    }

    /// Check if currently entering critical state
    #[inline]
    pub fn entering_critical(&self) -> bool {
        self.current_status()
            .map(|curr| {
                curr == IdentityStatus::Critical
                    && self.previous_status != IdentityStatus::Critical
            })
            .unwrap_or(false)
    }

    /// Mark that a crisis event was emitted (resets cooldown timer)
    #[inline]
    pub fn mark_event_emitted(&mut self) {
        self.last_event_time = Some(Instant::now());
    }

    /// Get time since last event emission
    #[inline]
    pub fn time_since_last_event(&self) -> Option<Duration> {
        self.last_event_time.map(|t| t.elapsed())
    }
}

/// Convert status to ordinal for comparison
/// Higher ordinal = healthier state
/// Critical=0, Degraded=1, Warning=2, Healthy=3
#[inline]
fn status_ordinal(status: IdentityStatus) -> u8 {
    match status {
        IdentityStatus::Critical => 0,
        IdentityStatus::Degraded => 1,
        IdentityStatus::Warning => 2,
        IdentityStatus::Healthy => 3,
    }
}
```

### 6. Update Module Exports

```rust
// File: crates/context-graph-core/src/gwt/ego_node/mod.rs
// ADD to re-exports after line 42:

pub use monitor::{CrisisDetectionResult, IdentityContinuityMonitor};
// Also update types export:
pub use types::{
    IdentityStatus, PurposeSnapshot, SelfReflectionResult,
    IC_CRISIS_THRESHOLD, IC_CRITICAL_THRESHOLD, MAX_PV_HISTORY_SIZE,
    CRISIS_EVENT_COOLDOWN,
};
```

---

## Full State Verification (FSV) Requirements

### 1. Source of Truth

| Artifact | Location | Verification Command |
|----------|----------|---------------------|
| `CRISIS_EVENT_COOLDOWN` constant | `gwt/ego_node/types.rs` | `grep -n "CRISIS_EVENT_COOLDOWN" crates/context-graph-core/src/gwt/ego_node/types.rs` |
| `CrisisDetectionResult` struct | `gwt/ego_node/monitor.rs` | `grep -n "pub struct CrisisDetectionResult" crates/context-graph-core/src/gwt/ego_node/monitor.rs` |
| `detect_crisis()` method | `gwt/ego_node/monitor.rs` | `grep -n "pub fn detect_crisis" crates/context-graph-core/src/gwt/ego_node/monitor.rs` |
| `previous_status` field | `gwt/ego_node/monitor.rs` | `grep -n "previous_status:" crates/context-graph-core/src/gwt/ego_node/monitor.rs` |
| Module export | `gwt/ego_node/mod.rs` | `grep -n "CrisisDetectionResult" crates/context-graph-core/src/gwt/ego_node/mod.rs` |

### 2. Execute & Inspect

After implementation, run these commands IN ORDER:

```bash
# Step 1: Build the crate
cargo build -p context-graph-core 2>&1 | tee /tmp/p004_build.log
echo "BUILD_EXIT_CODE: $?"

# Step 2: Run all crisis detection tests
cargo test -p context-graph-core crisis_detection -- --nocapture 2>&1 | tee /tmp/p004_test.log
echo "TEST_EXIT_CODE: $?"

# Step 3: Run clippy with strict warnings
cargo clippy -p context-graph-core -- -D warnings 2>&1 | tee /tmp/p004_clippy.log
echo "CLIPPY_EXIT_CODE: $?"

# Step 4: Verify exports compile
cargo check -p context-graph-core 2>&1 | grep -E "(error|CrisisDetectionResult)"
```

### 3. Boundary & Edge Case Audit (5 Cases)

| # | Edge Case | Input | Expected Output | Test Name |
|---|-----------|-------|-----------------|-----------|
| 1 | First detection (no history) | New monitor, no `compute_continuity()` called | `IC=1.0`, `Healthy`, `status_changed=false` | `test_crisis_detection_first_call_no_history` |
| 2 | Same status (no transition) | Healthy -> Healthy | `status_changed=false`, `entering_crisis=false` | `test_crisis_detection_same_status_no_change` |
| 3 | Enter crisis from Healthy | Healthy -> Warning | `entering_crisis=true`, `entering_critical=false` | `test_crisis_detection_entering_crisis_from_healthy` |
| 4 | Enter critical directly | Healthy -> Critical | `entering_crisis=true`, `entering_critical=true` | `test_crisis_detection_direct_to_critical` |
| 5 | Recovery | Critical -> Degraded | `recovering=true`, `entering_crisis=false` | `test_crisis_detection_recovery_from_critical` |

### 4. Evidence of Success

After running tests, verify with:

```bash
# Check test output for FSV evidence
grep -E "(EVIDENCE|BEFORE|AFTER|status_changed|entering_crisis|entering_critical)" /tmp/p004_test.log

# Expected output patterns:
# BEFORE: status = Healthy, previous_status = Healthy
# AFTER: status = Warning, previous_status = Warning
# EVIDENCE: entering_crisis = true
```

---

## Test Cases (NO MOCK DATA)

```rust
// File: crates/context-graph-core/src/gwt/ego_node/tests/tests_crisis_detection.rs

#[cfg(test)]
mod crisis_detection_tests {
    use super::*;
    use crate::gwt::ego_node::{
        CrisisDetectionResult, IdentityContinuityMonitor, IdentityStatus,
        CRISIS_EVENT_COOLDOWN,
    };
    use std::time::Duration;

    /// Create a uniform purpose vector for testing
    fn uniform_pv(val: f32) -> [f32; 13] {
        [val; 13]
    }

    /// Setup monitor at a specific status by computing continuity
    fn setup_monitor_at_status(status: IdentityStatus) -> IdentityContinuityMonitor {
        let mut monitor = IdentityContinuityMonitor::new();

        // First vector (establishes baseline, always Healthy IC=1.0)
        monitor.compute_continuity(&uniform_pv(1.0), 1.0, "baseline");

        // Second vector to set target status based on IC thresholds
        let kuramoto_r = match status {
            IdentityStatus::Healthy => 0.95,   // IC > 0.9
            IdentityStatus::Warning => 0.80,   // 0.7 <= IC <= 0.9
            IdentityStatus::Degraded => 0.60,  // 0.5 <= IC < 0.7
            IdentityStatus::Critical => 0.30,  // IC < 0.5
        };

        // Same PV (cosine=1.0), so IC = 1.0 * r = r
        monitor.compute_continuity(&uniform_pv(1.0), kuramoto_r, "target_status");

        // Call detect_crisis to update previous_status
        let _ = monitor.detect_crisis();

        monitor
    }

    #[test]
    fn test_crisis_detection_first_call_no_history() {
        println!("=== FSV: First detection with no compute_continuity ===");

        let mut monitor = IdentityContinuityMonitor::new();

        // BEFORE
        println!("BEFORE: history_len = {}", monitor.history_len());
        assert_eq!(monitor.history_len(), 0);

        // EXECUTE
        let result = monitor.detect_crisis();

        // AFTER
        println!("AFTER: IC = {}, status = {:?}", result.identity_coherence, result.current_status);
        println!("AFTER: status_changed = {}", result.status_changed);

        // First call with no history should default to Healthy
        assert!((result.identity_coherence - 1.0).abs() < 1e-6);
        assert_eq!(result.current_status, IdentityStatus::Healthy);
        assert!(!result.status_changed); // Previous was also Healthy (default)
        assert!(result.can_emit_event); // No cooldown initially

        println!("EVIDENCE: First detection correctly returns Healthy default");
    }

    #[test]
    fn test_crisis_detection_same_status_no_change() {
        println!("=== FSV: Same status = no transition ===");

        let mut monitor = setup_monitor_at_status(IdentityStatus::Healthy);

        // BEFORE
        let prev = monitor.previous_status();
        println!("BEFORE: previous_status = {:?}", prev);

        // Stay healthy (same vector, high r)
        monitor.compute_continuity(&uniform_pv(1.0), 0.95, "stay_healthy");

        // EXECUTE
        let result = monitor.detect_crisis();

        // AFTER
        println!("AFTER: current_status = {:?}", result.current_status);
        println!("AFTER: status_changed = {}", result.status_changed);

        assert!(!result.status_changed);
        assert!(!result.entering_crisis);
        assert!(!result.entering_critical);
        assert!(!result.recovering);

        println!("EVIDENCE: No status change when staying Healthy");
    }

    #[test]
    fn test_crisis_detection_entering_crisis_from_healthy() {
        println!("=== FSV: Healthy -> Warning = entering_crisis ===");

        let mut monitor = setup_monitor_at_status(IdentityStatus::Healthy);

        // BEFORE
        println!("BEFORE: previous_status = {:?}", monitor.previous_status());
        assert_eq!(monitor.previous_status(), IdentityStatus::Healthy);

        // Drop to Warning (IC ~ 0.75)
        monitor.compute_continuity(&uniform_pv(1.0), 0.75, "drop_to_warning");

        // EXECUTE
        let result = monitor.detect_crisis();

        // AFTER
        println!("AFTER: IC = {:.4}", result.identity_coherence);
        println!("AFTER: previous_status = {:?}, current_status = {:?}",
            result.previous_status, result.current_status);
        println!("AFTER: status_changed = {}, entering_crisis = {}",
            result.status_changed, result.entering_crisis);

        assert!(result.status_changed);
        assert!(result.entering_crisis);
        assert!(!result.entering_critical); // Warning, not Critical
        assert!(!result.recovering);
        assert_eq!(result.previous_status, IdentityStatus::Healthy);
        assert_eq!(result.current_status, IdentityStatus::Warning);

        println!("EVIDENCE: entering_crisis correctly detected");
    }

    #[test]
    fn test_crisis_detection_entering_critical() {
        println!("=== FSV: Warning -> Critical = entering_critical ===");

        let mut monitor = setup_monitor_at_status(IdentityStatus::Warning);

        // BEFORE
        println!("BEFORE: previous_status = {:?}", monitor.previous_status());
        assert_eq!(monitor.previous_status(), IdentityStatus::Warning);

        // Drop to Critical (IC ~ 0.30)
        monitor.compute_continuity(&uniform_pv(1.0), 0.30, "drop_to_critical");

        // EXECUTE
        let result = monitor.detect_crisis();

        // AFTER
        println!("AFTER: IC = {:.4}", result.identity_coherence);
        println!("AFTER: entering_critical = {}", result.entering_critical);

        assert!(result.status_changed);
        assert!(!result.entering_crisis); // Not from Healthy
        assert!(result.entering_critical);
        assert!(!result.recovering);
        assert_eq!(result.current_status, IdentityStatus::Critical);

        println!("EVIDENCE: entering_critical correctly detected");
    }

    #[test]
    fn test_crisis_detection_direct_to_critical() {
        println!("=== FSV: Healthy -> Critical = both flags ===");

        let mut monitor = setup_monitor_at_status(IdentityStatus::Healthy);

        // Crash directly to Critical
        monitor.compute_continuity(&uniform_pv(1.0), 0.20, "crash_to_critical");

        // EXECUTE
        let result = monitor.detect_crisis();

        // AFTER
        println!("AFTER: entering_crisis = {}, entering_critical = {}",
            result.entering_crisis, result.entering_critical);

        assert!(result.status_changed);
        assert!(result.entering_crisis); // From Healthy
        assert!(result.entering_critical); // To Critical

        println!("EVIDENCE: Direct Healthy->Critical sets both flags");
    }

    #[test]
    fn test_crisis_detection_recovery_from_critical() {
        println!("=== FSV: Critical -> Degraded = recovering ===");

        let mut monitor = setup_monitor_at_status(IdentityStatus::Critical);

        // BEFORE
        println!("BEFORE: previous_status = {:?}", monitor.previous_status());

        // Recover to Degraded (IC ~ 0.55)
        monitor.compute_continuity(&uniform_pv(1.0), 0.55, "recovery");

        // EXECUTE
        let result = monitor.detect_crisis();

        // AFTER
        println!("AFTER: recovering = {}", result.recovering);

        assert!(result.status_changed);
        assert!(!result.entering_crisis);
        assert!(!result.entering_critical);
        assert!(result.recovering);
        assert_eq!(result.previous_status, IdentityStatus::Critical);
        assert_eq!(result.current_status, IdentityStatus::Degraded);

        println!("EVIDENCE: Recovery correctly detected");
    }

    #[test]
    fn test_crisis_detection_cooldown_initially_can_emit() {
        println!("=== FSV: No cooldown initially ===");

        let mut monitor = setup_monitor_at_status(IdentityStatus::Healthy);
        monitor.compute_continuity(&uniform_pv(1.0), 0.30, "crisis");

        let result = monitor.detect_crisis();

        assert!(result.can_emit_event);
        assert!(result.time_since_last_event.is_none());

        println!("EVIDENCE: First event can be emitted (no cooldown)");
    }

    #[test]
    fn test_crisis_detection_cooldown_blocks_rapid_events() {
        println!("=== FSV: Cooldown blocks rapid events ===");

        let mut monitor = setup_monitor_at_status(IdentityStatus::Healthy);

        // First crisis
        monitor.compute_continuity(&uniform_pv(1.0), 0.30, "crisis1");
        let result1 = monitor.detect_crisis();
        assert!(result1.can_emit_event);

        // Mark event emitted
        monitor.mark_event_emitted();

        // Immediately detect again
        monitor.compute_continuity(&uniform_pv(1.0), 0.25, "crisis2");
        let result2 = monitor.detect_crisis();

        // VERIFY
        println!("AFTER: can_emit_event = {}", result2.can_emit_event);
        println!("AFTER: time_since_last_event = {:?}", result2.time_since_last_event);

        assert!(!result2.can_emit_event);
        assert!(result2.time_since_last_event.unwrap() < CRISIS_EVENT_COOLDOWN);

        println!("EVIDENCE: Cooldown correctly blocks rapid event emission");
    }

    #[test]
    fn test_status_ordinal_ordering() {
        // Verify ordinal function works correctly
        use super::status_ordinal;

        assert!(status_ordinal(IdentityStatus::Healthy) > status_ordinal(IdentityStatus::Warning));
        assert!(status_ordinal(IdentityStatus::Warning) > status_ordinal(IdentityStatus::Degraded));
        assert!(status_ordinal(IdentityStatus::Degraded) > status_ordinal(IdentityStatus::Critical));

        println!("EVIDENCE: Status ordinals correctly ordered");
    }
}
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/gwt/ego_node/tests/tests_crisis_detection.rs` | Test module for crisis detection |

---

## Files to Modify

| File | Changes |
|------|---------|
| `crates/context-graph-core/src/gwt/ego_node/types.rs` | Add `CRISIS_EVENT_COOLDOWN` constant |
| `crates/context-graph-core/src/gwt/ego_node/monitor.rs` | Add `CrisisDetectionResult`, update struct, add methods |
| `crates/context-graph-core/src/gwt/ego_node/mod.rs` | Update exports |
| `crates/context-graph-core/src/gwt/ego_node/tests/mod.rs` | Add `mod tests_crisis_detection;` |

---

## Implementation Checklist

### Phase 1: Constants and Types
- [ ] Add `use std::time::Duration;` to types.rs
- [ ] Add `CRISIS_EVENT_COOLDOWN` constant to types.rs
- [ ] Add `CrisisDetectionResult` struct to monitor.rs
- [ ] Add `status_ordinal()` function to monitor.rs

### Phase 2: Monitor Struct Updates
- [ ] Add `previous_status: IdentityStatus` field
- [ ] Add `last_event_time: Option<Instant>` field
- [ ] Add `#[serde(default)]` and `#[serde(skip)]` attributes
- [ ] Add `default_healthy_status()` helper function
- [ ] Update `new()` constructor
- [ ] Update `with_threshold()` constructor
- [ ] Update `with_capacity()` constructor

### Phase 3: Methods
- [ ] Implement `detect_crisis()` method
- [ ] Implement `previous_status()` getter
- [ ] Implement `status_changed()` method
- [ ] Implement `entering_critical()` method
- [ ] Implement `mark_event_emitted()` method
- [ ] Implement `time_since_last_event()` getter

### Phase 4: Exports and Tests
- [ ] Update `mod.rs` exports for `CrisisDetectionResult`
- [ ] Update `mod.rs` exports for `CRISIS_EVENT_COOLDOWN`
- [ ] Create `tests_crisis_detection.rs` test file
- [ ] Add `mod tests_crisis_detection;` to tests/mod.rs
- [ ] Write all 8 test cases

### Phase 5: Verification
- [ ] `cargo build -p context-graph-core` succeeds
- [ ] `cargo test -p context-graph-core crisis_detection` passes all tests
- [ ] `cargo clippy -p context-graph-core -- -D warnings` clean
- [ ] All FSV verification commands succeed
- [ ] All grep checks find expected patterns

---

## State Machine Diagram

```
                 +---------+
                 | Healthy | ordinal=3
                 +---------+
                      |
                      | IC drops below 0.9
                      | entering_crisis=true
                      v
                 +---------+
          +----->| Warning | ordinal=2
          |      +---------+
          |           |
          |           | IC drops below 0.7
  recovery|           v
          |      +---------+
          +------| Degraded| ordinal=1
          |      +---------+
          |           |
          |           | IC drops below 0.5
          |           | entering_critical=true
          |           v
          |      +----------+
          +------|  Critical| ordinal=0
                 +----------+
                      |
                      | emits IdentityCritical event
                      v
                 [Dream System]
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | Claude Opus 4.5 | Initial task specification |
| 1.1.0 | 2026-01-11 | Claude Opus 4.5 | Added implementation checklist and state machine |
| 2.0.0 | 2026-01-12 | Claude Opus 4.5 | **COMPLETE REWRITE**: Added codebase audit, FSV requirements, NO mock data policy, fail-fast policy, exact file paths, comprehensive test cases, verification commands |
