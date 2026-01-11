# Task Specification: Crisis Detection

**Task ID:** TASK-IDENTITY-P0-004
**Version:** 1.0.0
**Status:** Ready
**Layer:** Logic
**Sequence:** 4
**Estimated Complexity:** Low

---

## Metadata

| Field | Value |
|-------|-------|
| Implements | REQ-IDENTITY-004, REQ-IDENTITY-005 |
| Depends On | TASK-IDENTITY-P0-003 |
| Blocks | TASK-IDENTITY-P0-005 |
| Priority | P0 - Critical |

---

## Context

Once identity continuity is computed, we need to detect when the system enters a crisis state. This task adds the detection logic and status transition tracking to `IdentityContinuityMonitor`.

Per constitution.yaml lines 387-392:
- IC > 0.9: Healthy
- 0.7 <= IC <= 0.9: Warning
- 0.5 <= IC < 0.7: Degraded
- IC < 0.5: Critical

The system should:
1. Track status transitions
2. Detect when entering crisis (IC < 0.7)
3. Detect critical state (IC < 0.5) requiring dream intervention

---

## Input Context Files

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/gwt/ego_node.rs` | IdentityContinuityMonitor from TASK-003 |
| `specs/tasks/TASK-IDENTITY-P0-003.md` | Monitor implementation |
| `docs2/constitution.yaml` | Lines 387-392 for thresholds |

---

## Prerequisites

- [x] TASK-IDENTITY-P0-003 completed
- [x] IdentityContinuityMonitor exists with compute_continuity()

---

## Scope

### In Scope

1. Add `CrisisDetectionResult` struct for detection outcomes
2. Add status transition tracking to monitor
3. Implement `detect_crisis()` method
4. Add cooldown logic to prevent event spam
5. Unit tests for all state transitions

### Out of Scope

- Crisis protocol execution (TASK-IDENTITY-P0-005)
- Workspace event emission (TASK-IDENTITY-P0-005, TASK-IDENTITY-P0-006)

---

## Definition of Done

### Exact Signatures Required

```rust
// File: crates/context-graph-core/src/gwt/ego_node.rs

/// Result of crisis detection analysis
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
    /// Whether entering crisis (transition to Warning, Degraded, or Critical)
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

/// Minimum time between crisis event emissions
pub const CRISIS_EVENT_COOLDOWN: Duration = Duration::from_secs(30);

impl IdentityContinuityMonitor {
    /// Detect crisis state and track transitions
    ///
    /// # Returns
    /// CrisisDetectionResult with transition information
    ///
    /// # Cooldown Logic
    /// Events can only be emitted if CRISIS_EVENT_COOLDOWN (30s) has passed
    /// since the last emission, preventing event spam during fluctuations.
    pub fn detect_crisis(&mut self) -> CrisisDetectionResult;

    /// Get the previous status (before last computation)
    pub fn previous_status(&self) -> IdentityStatus;

    /// Check if status changed in last computation
    pub fn status_changed(&self) -> bool;

    /// Check if entering critical state
    pub fn entering_critical(&self) -> bool;

    /// Reset last event time (call after emitting event)
    pub fn mark_event_emitted(&mut self);
}
```

### Additional Fields for Monitor

```rust
// Add to IdentityContinuityMonitor struct:
/// Previous status for transition detection
previous_status: IdentityStatus,
/// Last time a crisis event was emitted
last_event_time: Option<Instant>,
```

### Constraints

1. Status transitions MUST be tracked accurately
2. `entering_crisis` MUST be true when transitioning from Healthy to any lower state
3. `entering_critical` MUST be true ONLY when transitioning to Critical
4. `recovering` MUST be true when status improves (Critical -> Degraded, etc.)
5. Cooldown MUST prevent rapid event emission
6. `can_emit_event` MUST be false during cooldown period
7. NO status change detection for first computation (no previous)

### Verification Commands

```bash
# Build
cargo build -p context-graph-core

# Run crisis detection tests
cargo test -p context-graph-core crisis_detection

# Clippy
cargo clippy -p context-graph-core -- -D warnings
```

---

## Pseudo Code

```rust
use std::time::{Duration, Instant};

pub const CRISIS_EVENT_COOLDOWN: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, PartialEq)]
pub struct CrisisDetectionResult {
    pub identity_coherence: f32,
    pub previous_status: IdentityStatus,
    pub current_status: IdentityStatus,
    pub status_changed: bool,
    pub entering_crisis: bool,
    pub entering_critical: bool,
    pub recovering: bool,
    pub time_since_last_event: Option<Duration>,
    pub can_emit_event: bool,
}

// Add to IdentityContinuityMonitor:
impl IdentityContinuityMonitor {
    // Update fields in struct definition:
    // previous_status: IdentityStatus,
    // last_event_time: Option<Instant>,

    pub fn detect_crisis(&mut self) -> CrisisDetectionResult {
        let current_status = self.current_status();
        let previous_status = self.previous_status;
        let ic = self.identity_coherence();

        let status_changed = current_status != previous_status;

        // Entering crisis = transitioning from Healthy to Warning/Degraded/Critical
        let entering_crisis = status_changed
            && previous_status == IdentityStatus::Healthy
            && current_status != IdentityStatus::Healthy;

        // Entering critical = transitioning TO Critical from any other state
        let entering_critical = status_changed
            && current_status == IdentityStatus::Critical
            && previous_status != IdentityStatus::Critical;

        // Recovering = improving status (lower ordinal to higher)
        let recovering = status_changed && status_ordinal(current_status) > status_ordinal(previous_status);

        // Cooldown check
        let time_since_last_event = self.last_event_time.map(|t| t.elapsed());
        let can_emit_event = match time_since_last_event {
            None => true, // No previous event
            Some(elapsed) => elapsed >= CRISIS_EVENT_COOLDOWN,
        };

        // Update previous status for next detection
        self.previous_status = current_status;

        CrisisDetectionResult {
            identity_coherence: ic,
            previous_status,
            current_status,
            status_changed,
            entering_crisis,
            entering_critical,
            recovering,
            time_since_last_event,
            can_emit_event,
        }
    }

    pub fn previous_status(&self) -> IdentityStatus {
        self.previous_status
    }

    pub fn status_changed(&self) -> bool {
        self.previous_status != self.current_status()
    }

    pub fn entering_critical(&self) -> bool {
        self.current_status() == IdentityStatus::Critical
            && self.previous_status != IdentityStatus::Critical
    }

    pub fn mark_event_emitted(&mut self) {
        self.last_event_time = Some(Instant::now());
    }
}

/// Convert status to ordinal for comparison
/// Higher ordinal = healthier
fn status_ordinal(status: IdentityStatus) -> u8 {
    match status {
        IdentityStatus::Critical => 0,
        IdentityStatus::Degraded => 1,
        IdentityStatus::Warning => 2,
        IdentityStatus::Healthy => 3,
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
| `crates/context-graph-core/src/gwt/ego_node.rs` | Add CrisisDetectionResult, extend IdentityContinuityMonitor |

---

## Validation Criteria

| Criterion | Verification Method |
|-----------|---------------------|
| status_changed correct for transitions | Unit tests for all transitions |
| entering_crisis triggers on Healthy -> lower | Unit test |
| entering_critical only for -> Critical | Unit tests |
| recovering detects improvement | Unit tests |
| Cooldown respected | Unit test with time mock |
| First computation has no transition | Unit test |

---

## Test Cases

```rust
#[cfg(test)]
mod crisis_detection_tests {
    use super::*;

    fn uniform_pv(val: f32) -> [f32; 13] {
        [val; 13]
    }

    fn setup_monitor_at_status(status: IdentityStatus) -> IdentityContinuityMonitor {
        let mut monitor = IdentityContinuityMonitor::new();

        // First vector (Healthy)
        monitor.compute_continuity(&uniform_pv(1.0), 1.0).unwrap();

        // Set to desired status based on IC thresholds
        let r = match status {
            IdentityStatus::Healthy => 0.95,  // IC > 0.9
            IdentityStatus::Warning => 0.80,  // 0.7 <= IC <= 0.9
            IdentityStatus::Degraded => 0.60, // 0.5 <= IC < 0.7
            IdentityStatus::Critical => 0.30, // IC < 0.5
        };

        monitor.compute_continuity(&uniform_pv(1.0), r).unwrap();
        monitor.detect_crisis(); // Update previous_status
        monitor
    }

    #[test]
    fn test_no_change_same_status() {
        let mut monitor = setup_monitor_at_status(IdentityStatus::Healthy);

        // Stay healthy
        monitor.compute_continuity(&uniform_pv(1.0), 0.95).unwrap();
        let result = monitor.detect_crisis();

        assert!(!result.status_changed);
        assert!(!result.entering_crisis);
        assert!(!result.entering_critical);
        assert!(!result.recovering);
    }

    #[test]
    fn test_entering_crisis_from_healthy() {
        let mut monitor = setup_monitor_at_status(IdentityStatus::Healthy);

        // Drop to Warning (IC = 0.75)
        monitor.compute_continuity(&uniform_pv(1.0), 0.75).unwrap();
        let result = monitor.detect_crisis();

        assert!(result.status_changed);
        assert!(result.entering_crisis);
        assert!(!result.entering_critical);
        assert!(!result.recovering);
        assert_eq!(result.previous_status, IdentityStatus::Healthy);
        assert_eq!(result.current_status, IdentityStatus::Warning);
    }

    #[test]
    fn test_entering_critical() {
        let mut monitor = setup_monitor_at_status(IdentityStatus::Warning);

        // Drop to Critical (IC = 0.3)
        monitor.compute_continuity(&uniform_pv(1.0), 0.30).unwrap();
        let result = monitor.detect_crisis();

        assert!(result.status_changed);
        assert!(!result.entering_crisis); // Not from Healthy
        assert!(result.entering_critical);
        assert!(!result.recovering);
        assert_eq!(result.current_status, IdentityStatus::Critical);
    }

    #[test]
    fn test_recovering_from_critical() {
        let mut monitor = setup_monitor_at_status(IdentityStatus::Critical);

        // Recover to Degraded (IC = 0.55)
        monitor.compute_continuity(&uniform_pv(1.0), 0.55).unwrap();
        let result = monitor.detect_crisis();

        assert!(result.status_changed);
        assert!(!result.entering_crisis);
        assert!(!result.entering_critical);
        assert!(result.recovering);
        assert_eq!(result.previous_status, IdentityStatus::Critical);
        assert_eq!(result.current_status, IdentityStatus::Degraded);
    }

    #[test]
    fn test_cooldown_initially_can_emit() {
        let mut monitor = setup_monitor_at_status(IdentityStatus::Healthy);

        monitor.compute_continuity(&uniform_pv(1.0), 0.30).unwrap();
        let result = monitor.detect_crisis();

        assert!(result.can_emit_event);
        assert!(result.time_since_last_event.is_none());
    }

    #[test]
    fn test_cooldown_blocks_rapid_events() {
        let mut monitor = setup_monitor_at_status(IdentityStatus::Healthy);

        // First crisis
        monitor.compute_continuity(&uniform_pv(1.0), 0.30).unwrap();
        let result1 = monitor.detect_crisis();
        assert!(result1.can_emit_event);

        // Mark event emitted
        monitor.mark_event_emitted();

        // Immediately try again
        monitor.compute_continuity(&uniform_pv(1.0), 0.25).unwrap();
        let result2 = monitor.detect_crisis();

        assert!(!result2.can_emit_event);
        assert!(result2.time_since_last_event.unwrap() < CRISIS_EVENT_COOLDOWN);
    }

    #[test]
    fn test_first_computation_no_transition() {
        let mut monitor = IdentityContinuityMonitor::new();

        // First vector
        monitor.compute_continuity(&uniform_pv(1.0), 0.30).unwrap();
        let result = monitor.detect_crisis();

        // First vector is always Healthy (IC = 1.0)
        // But since there's no "previous" in meaningful sense,
        // status_changed should reflect the initial state transition
        // Note: Implementation detail - first detect_crisis sets baseline
        assert_eq!(result.current_status, IdentityStatus::Healthy);
    }

    #[test]
    fn test_status_ordinal_ordering() {
        assert!(status_ordinal(IdentityStatus::Healthy) > status_ordinal(IdentityStatus::Warning));
        assert!(status_ordinal(IdentityStatus::Warning) > status_ordinal(IdentityStatus::Degraded));
        assert!(status_ordinal(IdentityStatus::Degraded) > status_ordinal(IdentityStatus::Critical));
    }

    #[test]
    fn test_direct_to_critical_from_healthy() {
        let mut monitor = setup_monitor_at_status(IdentityStatus::Healthy);

        // Crash directly to Critical
        monitor.compute_continuity(&uniform_pv(1.0), 0.20).unwrap();
        let result = monitor.detect_crisis();

        assert!(result.status_changed);
        assert!(result.entering_crisis);
        assert!(result.entering_critical);
        assert!(!result.recovering);
    }
}
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | Claude Opus 4.5 | Initial task specification |
