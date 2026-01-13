# Task Specification: Crisis Protocol

**Task ID:** TASK-IDENTITY-P0-005
**Version:** 2.0.0
**Status:** Ready
**Layer:** Logic
**Sequence:** 5
**Estimated Complexity:** Medium

---

## CRITICAL POLICIES - READ FIRST

### NO Backwards Compatibility
- **FAIL FAST**: Invalid input MUST return `CoreError` immediately
- **NO silent fallbacks**: Do not default to safe values when errors occur
- **Robust error logging**: All errors MUST include context (file, line, function, values)
- **NO workarounds**: If something fails, it errors out with clear diagnostics

### NO Mock Data in Tests
- ALL tests MUST use REAL data structures (`SelfEgoNode`, `CrisisDetectionResult`, `WorkspaceEvent`)
- Tests MUST verify actual state changes in ego node and broadcaster
- NO `#[cfg(test)]` mock implementations that hide broken functionality
- Tests must prove the system works end-to-end

---

## Codebase Audit (2026-01-12)

### WHAT EXISTS (Verified with exact file paths and line numbers):

| Component | Location | Line(s) | Status |
|-----------|----------|---------|--------|
| `SelfEgoNode` | `gwt/ego_node/self_ego_node.rs` | 19-135 | EXISTS |
| `record_purpose_snapshot()` | `gwt/ego_node/self_ego_node.rs` | 66-80 | EXISTS |
| `identity_trajectory: Vec<PurposeSnapshot>` | `gwt/ego_node/self_ego_node.rs` | 30 | EXISTS |
| `WorkspaceEvent` | `gwt/workspace/events.rs` | 10-41 | EXISTS |
| `WorkspaceEvent::IdentityCritical` | `gwt/workspace/events.rs` | 36-40 | EXISTS |
| `WorkspaceEventListener` trait | `gwt/workspace/events.rs` | 44-46 | EXISTS |
| `WorkspaceEventBroadcaster` | `gwt/workspace/events.rs` | 49-100 | EXISTS |
| `IdentityStatus` | `gwt/ego_node/types.rs` | 44-54 | EXISTS |
| `IC_CRITICAL_THRESHOLD` | `gwt/ego_node/types.rs` | 18 | EXISTS (0.5) |

### WHAT DOES NOT EXIST (Must be created by this task):

| Component | Expected Location | Status |
|-----------|-------------------|--------|
| `CrisisProtocolResult` struct | `gwt/ego_node/crisis_protocol.rs` | **DOES NOT EXIST** |
| `CrisisAction` enum | `gwt/ego_node/crisis_protocol.rs` | **DOES NOT EXIST** |
| `IdentityCrisisEvent` struct | `gwt/ego_node/crisis_protocol.rs` | **DOES NOT EXIST** |
| `CrisisProtocol` struct | `gwt/ego_node/crisis_protocol.rs` | **DOES NOT EXIST** |
| `CrisisProtocol::execute()` | `gwt/ego_node/crisis_protocol.rs` | **DOES NOT EXIST** |
| `crisis_protocol.rs` file | `gwt/ego_node/` | **DOES NOT EXIST** |

### Dependencies from P0-004 (MUST BE COMPLETED FIRST):

| Component | Location | Status |
|-----------|----------|--------|
| `CrisisDetectionResult` | `gwt/ego_node/monitor.rs` | **CREATED BY P0-004** |
| `detect_crisis()` | `IdentityContinuityMonitor` | **CREATED BY P0-004** |
| `mark_event_emitted()` | `IdentityContinuityMonitor` | **CREATED BY P0-004** |
| `CRISIS_EVENT_COOLDOWN` | `gwt/ego_node/types.rs` | **CREATED BY P0-004** |

### Current WorkspaceEvent::IdentityCritical State:

```rust
// File: crates/context-graph-core/src/gwt/workspace/events.rs (lines 36-40)
IdentityCritical {
    identity_coherence: f32,
    reason: String,
    timestamp: DateTime<Utc>,
}
// NOTE: This task will add previous_status and current_status fields
```

---

## Metadata

| Field | Value |
|-------|-------|
| Implements | IDENTITY-006, IDENTITY-007 from constitution.yaml |
| Depends On | TASK-IDENTITY-P0-004 (Crisis Detection) |
| Blocks | TASK-IDENTITY-P0-006 (GWT Attention Wiring) |
| Priority | P0 - Critical |

---

## Context

### What This Task Does

When `CrisisDetectionResult` indicates identity crisis, the system must execute a protocol:

1. **IC < 0.7 (Warning/Degraded/Critical):** Record purpose snapshot for trajectory analysis
2. **IC < 0.5 (Critical):** Generate `IdentityCrisisEvent` for workspace broadcast
3. **Cooldown check:** Only emit event if cooldown allows (from P0-004)
4. **Track actions:** Return comprehensive result with all actions taken

### Constitution Reference

Per constitution.yaml lines 387-392 (identity thresholds):
- `IC < 0.7`: Warning/Degraded state - record snapshot
- `IC < 0.5`: Critical state - trigger dream consolidation via workspace event

Per constitution.yaml line 391:
- "dream<0.5" - critical threshold triggers introspective dream

### Why This Matters

Without crisis protocol:
- No snapshots recorded during identity drift (loses debugging context)
- No workspace events generated (Dream system never receives crisis signal)
- No action tracking (MCP tools can't report what happened)
- P0-006 (GWT Wiring) has nothing to emit

---

## Prerequisites

- [ ] TASK-IDENTITY-P0-004 completed (`CrisisDetectionResult`, `detect_crisis()`, `mark_event_emitted()`)
- [x] `SelfEgoNode` exists with `record_purpose_snapshot()` method
- [x] `WorkspaceEvent::IdentityCritical` variant exists
- [x] `WorkspaceEventBroadcaster` exists with `broadcast()` method

---

## Scope

### In Scope

1. Create `crisis_protocol.rs` file
2. Define `CrisisAction` enum with all action types
3. Define `IdentityCrisisEvent` struct
4. Define `CrisisProtocolResult` struct
5. Define `CrisisProtocol` struct
6. Implement `CrisisProtocol::new()` constructor
7. Implement `CrisisProtocol::execute()` async method
8. Implement `IdentityCrisisEvent::from_detection()` factory
9. Implement `IdentityCrisisEvent::to_workspace_event()` conversion
10. Update `WorkspaceEvent::IdentityCritical` to include status fields
11. Update module exports
12. Comprehensive unit tests

### Out of Scope

- Actual workspace broadcasting (P0-006 wires this to GWT)
- Dream system integration (Dream subsystem listens to events)
- MCP tool exposure (P0-007)

---

## Definition of Done

### 1. Create Crisis Protocol File

```rust
// File: crates/context-graph-core/src/gwt/ego_node/crisis_protocol.rs

//! Crisis Protocol - Identity crisis response execution
//!
//! Implements the crisis response protocol per constitution.yaml lines 387-392:
//! - IC < 0.7: Record purpose snapshot
//! - IC < 0.5: Generate IdentityCrisisEvent for broadcast

use std::sync::Arc;
use std::time::Duration;
use chrono::{DateTime, Utc};
use tokio::sync::RwLock;

use crate::error::CoreResult;
use super::identity_continuity::IdentityContinuity;
use super::monitor::{CrisisDetectionResult, IdentityContinuityMonitor};
use super::self_ego_node::SelfEgoNode;
use super::types::{IdentityStatus, CRISIS_EVENT_COOLDOWN};
use crate::gwt::workspace::WorkspaceEvent;

/// Actions taken during crisis protocol execution
///
/// Each action is recorded in `CrisisProtocolResult.actions` for
/// auditability and MCP tool reporting.
#[derive(Debug, Clone, PartialEq)]
pub enum CrisisAction {
    /// Purpose snapshot was recorded in SelfEgoNode
    SnapshotRecorded {
        /// Context message saved with snapshot
        context: String,
    },
    /// IdentityCrisisEvent was generated
    EventGenerated {
        /// Event type identifier
        event_type: String,
    },
    /// Event was not emitted due to cooldown
    EventSkippedCooldown {
        /// Time remaining until cooldown expires
        remaining: Duration,
    },
    /// Introspection mode was triggered (event emitted successfully)
    IntrospectionTriggered,
}

/// Event payload for identity crisis
///
/// Contains all information needed for workspace broadcast and
/// Dream system consumption.
#[derive(Debug, Clone)]
pub struct IdentityCrisisEvent {
    /// Current identity coherence (IC) value
    pub identity_coherence: f32,
    /// Status before crisis detection
    pub previous_status: IdentityStatus,
    /// Current status (should be Critical)
    pub current_status: IdentityStatus,
    /// Human-readable reason for crisis
    pub reason: String,
    /// Timestamp of event generation
    pub timestamp: DateTime<Utc>,
}

impl IdentityCrisisEvent {
    /// Create event from detection result
    ///
    /// # Arguments
    /// * `detection` - Result from `detect_crisis()`
    /// * `reason` - Human-readable reason for crisis
    ///
    /// # Returns
    /// `IdentityCrisisEvent` with all fields populated
    pub fn from_detection(
        detection: &CrisisDetectionResult,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            identity_coherence: detection.identity_coherence,
            previous_status: detection.previous_status,
            current_status: detection.current_status,
            reason: reason.into(),
            timestamp: Utc::now(),
        }
    }

    /// Convert to WorkspaceEvent for broadcasting
    ///
    /// # Returns
    /// `WorkspaceEvent::IdentityCritical` variant
    pub fn to_workspace_event(&self) -> WorkspaceEvent {
        WorkspaceEvent::IdentityCritical {
            identity_coherence: self.identity_coherence,
            previous_status: format!("{:?}", self.previous_status),
            current_status: format!("{:?}", self.current_status),
            reason: self.reason.clone(),
            timestamp: self.timestamp,
        }
    }
}

/// Result of crisis protocol execution
///
/// Contains all information about what actions were taken during
/// crisis protocol execution, for MCP reporting and debugging.
#[derive(Debug, Clone)]
pub struct CrisisProtocolResult {
    /// The detection result that triggered this protocol
    pub detection: CrisisDetectionResult,
    /// Whether a purpose snapshot was recorded
    pub snapshot_recorded: bool,
    /// Context message for recorded snapshot (if any)
    pub snapshot_context: Option<String>,
    /// Generated event (if any)
    pub event: Option<IdentityCrisisEvent>,
    /// Whether event was actually emittable (cooldown check passed)
    pub event_emitted: bool,
    /// All actions taken during execution
    pub actions: Vec<CrisisAction>,
}

/// Executes crisis protocol based on detection results
///
/// Holds reference to `SelfEgoNode` for snapshot recording.
/// Does NOT hold broadcaster reference - P0-006 handles broadcasting.
pub struct CrisisProtocol {
    /// Reference to ego node for snapshot recording
    ego_node: Arc<RwLock<SelfEgoNode>>,
}

impl CrisisProtocol {
    /// Create new protocol with ego node reference
    ///
    /// # Arguments
    /// * `ego_node` - Arc-wrapped SelfEgoNode for snapshot recording
    pub fn new(ego_node: Arc<RwLock<SelfEgoNode>>) -> Self {
        Self { ego_node }
    }

    /// Execute crisis protocol based on detection result
    ///
    /// # Algorithm
    /// 1. If status is NOT Healthy (IC < 0.7):
    ///    a. Record purpose snapshot with context in SelfEgoNode
    /// 2. If entering_critical OR current_status == Critical:
    ///    a. Generate IdentityCrisisEvent
    ///    b. Check cooldown via detection.can_emit_event
    ///    c. If can emit: mark event_emitted = true, add IntrospectionTriggered
    ///    d. If cannot emit: add EventSkippedCooldown with remaining time
    /// 3. Return CrisisProtocolResult with all actions
    ///
    /// # Arguments
    /// * `detection` - Result from `detect_crisis()`
    /// * `monitor` - Mutable reference to monitor for `mark_event_emitted()`
    ///
    /// # Returns
    /// `CoreResult<CrisisProtocolResult>` with all actions taken
    ///
    /// # Errors
    /// Returns `CoreError` if `record_purpose_snapshot` fails
    pub async fn execute(
        &self,
        detection: CrisisDetectionResult,
        monitor: &mut IdentityContinuityMonitor,
    ) -> CoreResult<CrisisProtocolResult> {
        let mut actions = Vec::new();
        let mut snapshot_recorded = false;
        let mut snapshot_context: Option<String> = None;
        let mut event: Option<IdentityCrisisEvent> = None;
        let mut event_emitted = false;

        // Step 1: Handle any crisis state (IC < 0.7 = not Healthy)
        if detection.current_status != IdentityStatus::Healthy {
            let context = format!(
                "Identity crisis: IC={:.4}, status={:?} (was {:?})",
                detection.identity_coherence,
                detection.current_status,
                detection.previous_status
            );

            // Record snapshot in ego node
            {
                let mut ego = self.ego_node.write().await;
                ego.record_purpose_snapshot(&context)?;
            }

            snapshot_recorded = true;
            snapshot_context = Some(context.clone());
            actions.push(CrisisAction::SnapshotRecorded { context });

            tracing::debug!(
                ic = %detection.identity_coherence,
                status = ?detection.current_status,
                "Crisis protocol: snapshot recorded"
            );
        }

        // Step 2: Handle critical state (IC < 0.5)
        if detection.entering_critical || detection.current_status == IdentityStatus::Critical {
            let reason = if detection.entering_critical {
                format!(
                    "Identity entered critical state: IC dropped from {:?} to Critical (IC={:.4})",
                    detection.previous_status,
                    detection.identity_coherence
                )
            } else {
                format!(
                    "Identity remains critical: IC={:.4}",
                    detection.identity_coherence
                )
            };

            let crisis_event = IdentityCrisisEvent::from_detection(&detection, &reason);
            event = Some(crisis_event.clone());

            actions.push(CrisisAction::EventGenerated {
                event_type: "IdentityCritical".to_string(),
            });

            // Check cooldown
            if detection.can_emit_event {
                event_emitted = true;
                monitor.mark_event_emitted();
                actions.push(CrisisAction::IntrospectionTriggered);

                tracing::warn!(
                    ic = %detection.identity_coherence,
                    "Crisis protocol: IdentityCritical event ready for emission"
                );
            } else {
                // Calculate remaining cooldown time
                let remaining = detection
                    .time_since_last_event
                    .map(|elapsed| CRISIS_EVENT_COOLDOWN.saturating_sub(elapsed))
                    .unwrap_or(Duration::ZERO);

                actions.push(CrisisAction::EventSkippedCooldown { remaining });

                tracing::debug!(
                    remaining_secs = %remaining.as_secs(),
                    "Crisis protocol: event skipped due to cooldown"
                );
            }
        }

        Ok(CrisisProtocolResult {
            detection,
            snapshot_recorded,
            snapshot_context,
            event,
            event_emitted,
            actions,
        })
    }
}
```

### 2. Update WorkspaceEvent::IdentityCritical

```rust
// File: crates/context-graph-core/src/gwt/workspace/events.rs
// MODIFY IdentityCritical variant (lines 36-40):

/// Identity coherence critical (IC < 0.5) - triggers dream consolidation
/// From constitution.yaml lines 387-392: "dream<0.5"
///
/// # TASK-IDENTITY-P0-005: Added previous_status and current_status
IdentityCritical {
    identity_coherence: f32,
    /// Status before crisis (e.g., "Healthy", "Warning", "Degraded")
    previous_status: String,
    /// Current status (should be "Critical")
    current_status: String,
    reason: String,
    timestamp: DateTime<Utc>,
},
```

### 3. Update Module Exports

```rust
// File: crates/context-graph-core/src/gwt/ego_node/mod.rs
// ADD after line 32:

mod crisis_protocol;

// ADD to re-exports after line 46:
pub use crisis_protocol::{
    CrisisAction, CrisisProtocol, CrisisProtocolResult, IdentityCrisisEvent,
};
```

---

## Full State Verification (FSV) Requirements

### 1. Source of Truth

| Artifact | Location | Verification Command |
|----------|----------|---------------------|
| `crisis_protocol.rs` file | `gwt/ego_node/crisis_protocol.rs` | `ls -la crates/context-graph-core/src/gwt/ego_node/crisis_protocol.rs` |
| `CrisisProtocol` struct | `gwt/ego_node/crisis_protocol.rs` | `grep -n "pub struct CrisisProtocol" crates/context-graph-core/src/gwt/ego_node/crisis_protocol.rs` |
| `execute()` method | `gwt/ego_node/crisis_protocol.rs` | `grep -n "pub async fn execute" crates/context-graph-core/src/gwt/ego_node/crisis_protocol.rs` |
| `IdentityCrisisEvent` struct | `gwt/ego_node/crisis_protocol.rs` | `grep -n "pub struct IdentityCrisisEvent" crates/context-graph-core/src/gwt/ego_node/crisis_protocol.rs` |
| `CrisisAction` enum | `gwt/ego_node/crisis_protocol.rs` | `grep -n "pub enum CrisisAction" crates/context-graph-core/src/gwt/ego_node/crisis_protocol.rs` |
| Updated `IdentityCritical` | `gwt/workspace/events.rs` | `grep -n "previous_status:" crates/context-graph-core/src/gwt/workspace/events.rs` |
| Module export | `gwt/ego_node/mod.rs` | `grep -n "CrisisProtocol" crates/context-graph-core/src/gwt/ego_node/mod.rs` |

### 2. Execute & Inspect

After implementation, run these commands IN ORDER:

```bash
# Step 1: Build the crate
cargo build -p context-graph-core 2>&1 | tee /tmp/p005_build.log
echo "BUILD_EXIT_CODE: $?"

# Step 2: Run all crisis protocol tests
cargo test -p context-graph-core crisis_protocol -- --nocapture 2>&1 | tee /tmp/p005_test.log
echo "TEST_EXIT_CODE: $?"

# Step 3: Run clippy with strict warnings
cargo clippy -p context-graph-core -- -D warnings 2>&1 | tee /tmp/p005_clippy.log
echo "CLIPPY_EXIT_CODE: $?"

# Step 4: Verify structs exist
grep -n "pub struct CrisisProtocol" crates/context-graph-core/src/gwt/ego_node/crisis_protocol.rs
grep -n "pub struct IdentityCrisisEvent" crates/context-graph-core/src/gwt/ego_node/crisis_protocol.rs
grep -n "pub enum CrisisAction" crates/context-graph-core/src/gwt/ego_node/crisis_protocol.rs

# Step 5: Verify WorkspaceEvent update
grep -A5 "IdentityCritical {" crates/context-graph-core/src/gwt/workspace/events.rs
```

### 3. Boundary & Edge Case Audit (5 Cases)

| # | Edge Case | Input | Expected Output | Test Name |
|---|-----------|-------|-----------------|-----------|
| 1 | Healthy status | `detection.current_status == Healthy` | `snapshot_recorded=false`, `event=None` | `test_protocol_healthy_no_action` |
| 2 | Warning status | `detection.current_status == Warning` | `snapshot_recorded=true`, `event=None` | `test_protocol_warning_snapshot_only` |
| 3 | Critical status (can emit) | `detection.current_status == Critical, can_emit=true` | `event_emitted=true`, `IntrospectionTriggered` | `test_protocol_critical_event_emitted` |
| 4 | Critical status (cooldown) | `detection.current_status == Critical, can_emit=false` | `event_emitted=false`, `EventSkippedCooldown` | `test_protocol_critical_cooldown_blocks` |
| 5 | Event conversion | `IdentityCrisisEvent::to_workspace_event()` | `WorkspaceEvent::IdentityCritical` with all fields | `test_event_to_workspace_event` |

### 4. Evidence of Success

After running tests, verify with:

```bash
# Check test output for FSV evidence
grep -E "(EVIDENCE|BEFORE|AFTER|snapshot_recorded|event_emitted|IntrospectionTriggered)" /tmp/p005_test.log

# Verify ego node trajectory was modified
grep -E "(identity_trajectory|record_purpose_snapshot)" /tmp/p005_test.log

# Expected output patterns:
# BEFORE: ego.identity_trajectory.len() = 0
# AFTER: ego.identity_trajectory.len() = 1
# EVIDENCE: Snapshot recorded in ego node
# EVIDENCE: IdentityCritical event generated
```

---

## Test Cases (NO MOCK DATA)

```rust
// File: crates/context-graph-core/src/gwt/ego_node/tests/tests_crisis_protocol.rs

#[cfg(test)]
mod crisis_protocol_tests {
    use super::*;
    use crate::gwt::ego_node::{
        CrisisAction, CrisisProtocol, CrisisProtocolResult, IdentityCrisisEvent,
        CrisisDetectionResult, IdentityContinuityMonitor, IdentityStatus,
        SelfEgoNode, CRISIS_EVENT_COOLDOWN,
    };
    use crate::gwt::workspace::WorkspaceEvent;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::RwLock;

    /// Create test ego node wrapped in Arc<RwLock>
    fn create_test_ego() -> Arc<RwLock<SelfEgoNode>> {
        Arc::new(RwLock::new(SelfEgoNode::new()))
    }

    /// Create a detection result for testing
    fn create_detection(
        ic: f32,
        prev: IdentityStatus,
        curr: IdentityStatus,
        can_emit: bool,
    ) -> CrisisDetectionResult {
        CrisisDetectionResult {
            identity_coherence: ic,
            previous_status: prev,
            current_status: curr,
            status_changed: prev != curr,
            entering_crisis: prev == IdentityStatus::Healthy && curr != IdentityStatus::Healthy,
            entering_critical: curr == IdentityStatus::Critical && prev != IdentityStatus::Critical,
            recovering: false,
            time_since_last_event: if can_emit { None } else { Some(Duration::from_secs(5)) },
            can_emit_event: can_emit,
        }
    }

    #[tokio::test]
    async fn test_protocol_healthy_no_action() {
        println!("=== FSV: Healthy status = no actions ===");

        let ego = create_test_ego();
        let protocol = CrisisProtocol::new(ego.clone());
        let mut monitor = IdentityContinuityMonitor::new();

        let detection = create_detection(
            0.95,
            IdentityStatus::Healthy,
            IdentityStatus::Healthy,
            true,
        );

        // BEFORE
        let before_len = ego.read().await.identity_trajectory.len();
        println!("BEFORE: identity_trajectory.len() = {}", before_len);

        // EXECUTE
        let result = protocol.execute(detection, &mut monitor).await.unwrap();

        // AFTER
        let after_len = ego.read().await.identity_trajectory.len();
        println!("AFTER: identity_trajectory.len() = {}", after_len);
        println!("AFTER: snapshot_recorded = {}", result.snapshot_recorded);
        println!("AFTER: event = {:?}", result.event);

        assert!(!result.snapshot_recorded);
        assert!(result.snapshot_context.is_none());
        assert!(result.event.is_none());
        assert!(!result.event_emitted);
        assert!(result.actions.is_empty());
        assert_eq!(before_len, after_len);

        println!("EVIDENCE: Healthy status takes no action");
    }

    #[tokio::test]
    async fn test_protocol_warning_snapshot_only() {
        println!("=== FSV: Warning status = snapshot only ===");

        let ego = create_test_ego();
        let protocol = CrisisProtocol::new(ego.clone());
        let mut monitor = IdentityContinuityMonitor::new();

        let detection = create_detection(
            0.75,
            IdentityStatus::Healthy,
            IdentityStatus::Warning,
            true,
        );

        // BEFORE
        let before_len = ego.read().await.identity_trajectory.len();
        println!("BEFORE: identity_trajectory.len() = {}", before_len);

        // EXECUTE
        let result = protocol.execute(detection, &mut monitor).await.unwrap();

        // AFTER
        let after_len = ego.read().await.identity_trajectory.len();
        println!("AFTER: identity_trajectory.len() = {}", after_len);
        println!("AFTER: snapshot_recorded = {}", result.snapshot_recorded);

        assert!(result.snapshot_recorded);
        assert!(result.snapshot_context.is_some());
        assert!(result.event.is_none()); // Warning doesn't generate event
        assert!(!result.event_emitted);
        assert_eq!(after_len, before_len + 1);

        // Verify action
        assert!(result.actions.iter().any(|a| matches!(a, CrisisAction::SnapshotRecorded { .. })));
        assert!(!result.actions.iter().any(|a| matches!(a, CrisisAction::EventGenerated { .. })));

        // Verify snapshot content
        let ego_read = ego.read().await;
        let snapshot = ego_read.identity_trajectory.last().unwrap();
        assert!(snapshot.context.contains("Warning"));
        println!("EVIDENCE: Snapshot context = {}", snapshot.context);
    }

    #[tokio::test]
    async fn test_protocol_critical_event_emitted() {
        println!("=== FSV: Critical status (can emit) = event emitted ===");

        let ego = create_test_ego();
        let protocol = CrisisProtocol::new(ego.clone());
        let mut monitor = IdentityContinuityMonitor::new();

        let detection = create_detection(
            0.30,
            IdentityStatus::Warning,
            IdentityStatus::Critical,
            true, // Can emit
        );

        // EXECUTE
        let result = protocol.execute(detection, &mut monitor).await.unwrap();

        // VERIFY
        println!("AFTER: snapshot_recorded = {}", result.snapshot_recorded);
        println!("AFTER: event = {:?}", result.event);
        println!("AFTER: event_emitted = {}", result.event_emitted);

        assert!(result.snapshot_recorded);
        assert!(result.event.is_some());
        assert!(result.event_emitted);

        // Verify actions
        assert!(result.actions.iter().any(|a| matches!(a, CrisisAction::SnapshotRecorded { .. })));
        assert!(result.actions.iter().any(|a| matches!(a, CrisisAction::EventGenerated { .. })));
        assert!(result.actions.iter().any(|a| matches!(a, CrisisAction::IntrospectionTriggered)));

        // Verify event content
        let event = result.event.unwrap();
        assert!((event.identity_coherence - 0.30).abs() < 1e-6);
        assert_eq!(event.current_status, IdentityStatus::Critical);
        assert!(event.reason.contains("critical"));

        println!("EVIDENCE: IdentityCritical event generated and marked emitted");
    }

    #[tokio::test]
    async fn test_protocol_critical_cooldown_blocks() {
        println!("=== FSV: Critical status (cooldown) = event blocked ===");

        let ego = create_test_ego();
        let protocol = CrisisProtocol::new(ego.clone());
        let mut monitor = IdentityContinuityMonitor::new();

        let detection = create_detection(
            0.30,
            IdentityStatus::Warning,
            IdentityStatus::Critical,
            false, // Cannot emit (cooldown)
        );

        // EXECUTE
        let result = protocol.execute(detection, &mut monitor).await.unwrap();

        // VERIFY
        println!("AFTER: event = {:?}", result.event);
        println!("AFTER: event_emitted = {}", result.event_emitted);

        assert!(result.event.is_some()); // Event generated
        assert!(!result.event_emitted); // But not emitted

        // Verify cooldown action
        let has_cooldown_action = result.actions.iter().any(|a| {
            matches!(a, CrisisAction::EventSkippedCooldown { .. })
        });
        assert!(has_cooldown_action);

        // No introspection triggered
        let has_introspection = result.actions.iter().any(|a| {
            matches!(a, CrisisAction::IntrospectionTriggered)
        });
        assert!(!has_introspection);

        println!("EVIDENCE: Event blocked by cooldown");
    }

    #[tokio::test]
    async fn test_event_to_workspace_event() {
        println!("=== FSV: IdentityCrisisEvent converts to WorkspaceEvent ===");

        let detection = create_detection(
            0.30,
            IdentityStatus::Warning,
            IdentityStatus::Critical,
            true,
        );

        // Create event
        let event = IdentityCrisisEvent::from_detection(&detection, "Test reason");
        println!("BEFORE: event.identity_coherence = {}", event.identity_coherence);

        // Convert to workspace event
        let ws_event = event.to_workspace_event();

        // VERIFY
        if let WorkspaceEvent::IdentityCritical {
            identity_coherence,
            previous_status,
            current_status,
            reason,
            ..
        } = ws_event
        {
            println!("AFTER: identity_coherence = {}", identity_coherence);
            println!("AFTER: previous_status = {}", previous_status);
            println!("AFTER: current_status = {}", current_status);
            println!("AFTER: reason = {}", reason);

            assert!((identity_coherence - 0.30).abs() < 1e-6);
            assert_eq!(previous_status, "Warning");
            assert_eq!(current_status, "Critical");
            assert_eq!(reason, "Test reason");

            println!("EVIDENCE: WorkspaceEvent::IdentityCritical created correctly");
        } else {
            panic!("Expected IdentityCritical event variant");
        }
    }

    #[tokio::test]
    async fn test_protocol_direct_to_critical() {
        println!("=== FSV: Healthy -> Critical = snapshot + event ===");

        let ego = create_test_ego();
        let protocol = CrisisProtocol::new(ego.clone());
        let mut monitor = IdentityContinuityMonitor::new();

        let detection = create_detection(
            0.20,
            IdentityStatus::Healthy,
            IdentityStatus::Critical,
            true,
        );

        // EXECUTE
        let result = protocol.execute(detection, &mut monitor).await.unwrap();

        // VERIFY
        assert!(result.snapshot_recorded);
        assert!(result.event.is_some());
        assert!(result.event_emitted);
        assert_eq!(result.actions.len(), 3); // Snapshot, EventGenerated, IntrospectionTriggered

        println!("EVIDENCE: Direct Healthy->Critical takes all actions");
    }

    #[tokio::test]
    async fn test_protocol_degraded_snapshot_no_event() {
        println!("=== FSV: Degraded status = snapshot, no event ===");

        let ego = create_test_ego();
        let protocol = CrisisProtocol::new(ego.clone());
        let mut monitor = IdentityContinuityMonitor::new();

        let detection = create_detection(
            0.55, // IC in Degraded range (0.5-0.7)
            IdentityStatus::Warning,
            IdentityStatus::Degraded,
            true,
        );

        // EXECUTE
        let result = protocol.execute(detection, &mut monitor).await.unwrap();

        // VERIFY
        assert!(result.snapshot_recorded);
        assert!(result.event.is_none()); // Degraded doesn't trigger event
        assert!(!result.event_emitted);
        assert_eq!(result.actions.len(), 1); // Only SnapshotRecorded

        println!("EVIDENCE: Degraded status only records snapshot");
    }

    #[tokio::test]
    async fn test_event_reason_entering_critical() {
        println!("=== FSV: Entering critical reason message ===");

        let mut detection = create_detection(
            0.30,
            IdentityStatus::Warning,
            IdentityStatus::Critical,
            true,
        );
        detection.entering_critical = true;

        let ego = create_test_ego();
        let protocol = CrisisProtocol::new(ego);
        let mut monitor = IdentityContinuityMonitor::new();

        let result = protocol.execute(detection, &mut monitor).await.unwrap();

        let event = result.event.unwrap();
        assert!(event.reason.contains("entered critical"));
        println!("EVIDENCE: Reason contains 'entered critical': {}", event.reason);
    }

    #[tokio::test]
    async fn test_event_reason_remains_critical() {
        println!("=== FSV: Remains critical reason message ===");

        let mut detection = create_detection(
            0.25,
            IdentityStatus::Critical,
            IdentityStatus::Critical,
            true,
        );
        detection.entering_critical = false; // Already was critical
        detection.status_changed = false;

        let ego = create_test_ego();
        let protocol = CrisisProtocol::new(ego);
        let mut monitor = IdentityContinuityMonitor::new();

        let result = protocol.execute(detection, &mut monitor).await.unwrap();

        let event = result.event.unwrap();
        assert!(event.reason.contains("remains critical"));
        println!("EVIDENCE: Reason contains 'remains critical': {}", event.reason);
    }
}
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/gwt/ego_node/crisis_protocol.rs` | Crisis protocol implementation |
| `crates/context-graph-core/src/gwt/ego_node/tests/tests_crisis_protocol.rs` | Test module |

---

## Files to Modify

| File | Changes |
|------|---------|
| `crates/context-graph-core/src/gwt/workspace/events.rs` | Add `previous_status`, `current_status` to `IdentityCritical` |
| `crates/context-graph-core/src/gwt/ego_node/mod.rs` | Add `mod crisis_protocol;` and exports |
| `crates/context-graph-core/src/gwt/ego_node/tests/mod.rs` | Add `mod tests_crisis_protocol;` |

---

## Implementation Checklist

### Phase 1: Create Crisis Protocol File
- [ ] Create `crisis_protocol.rs` file
- [ ] Add imports and module doc
- [ ] Define `CrisisAction` enum with all variants
- [ ] Define `IdentityCrisisEvent` struct
- [ ] Implement `IdentityCrisisEvent::from_detection()`
- [ ] Implement `IdentityCrisisEvent::to_workspace_event()`
- [ ] Define `CrisisProtocolResult` struct
- [ ] Define `CrisisProtocol` struct
- [ ] Implement `CrisisProtocol::new()`
- [ ] Implement `CrisisProtocol::execute()` async method

### Phase 2: Update WorkspaceEvent
- [ ] Add `previous_status: String` field to `IdentityCritical`
- [ ] Add `current_status: String` field to `IdentityCritical`
- [ ] Update all existing usages (search for `IdentityCritical {`)

### Phase 3: Module Exports
- [ ] Add `mod crisis_protocol;` to `ego_node/mod.rs`
- [ ] Add exports for `CrisisAction`, `CrisisProtocol`, `CrisisProtocolResult`, `IdentityCrisisEvent`

### Phase 4: Tests
- [ ] Create `tests_crisis_protocol.rs` file
- [ ] Add `mod tests_crisis_protocol;` to tests/mod.rs
- [ ] Write all 8 test cases
- [ ] Ensure all tests use REAL data structures

### Phase 5: Verification
- [ ] `cargo build -p context-graph-core` succeeds
- [ ] `cargo test -p context-graph-core crisis_protocol` passes all tests
- [ ] `cargo clippy -p context-graph-core -- -D warnings` clean
- [ ] All FSV verification commands succeed
- [ ] All grep checks find expected patterns

---

## Integration with Downstream Tasks

### How P0-006 (GWT Attention Wiring) Uses This

```rust
// In IdentityContinuityListener (P0-006):
async fn process_event(&self, event: &WorkspaceEvent) {
    // ... compute IC ...
    let detection = monitor.detect_crisis();

    if detection.current_status != IdentityStatus::Healthy {
        // Use CrisisProtocol from P0-005
        let protocol_result = self.protocol.execute(detection, &mut monitor).await?;

        // Emit event if allowed
        if protocol_result.event_emitted {
            if let Some(crisis_event) = protocol_result.event {
                let ws_event = crisis_event.to_workspace_event();
                broadcaster.broadcast(ws_event).await;
            }
        }
    }
}
```

### Event Flow Diagram

```
MemoryEnters Event
       |
       v
IdentityContinuityListener (P0-006)
       |
       v
IdentityContinuityMonitor.compute_continuity()
       |
       v
IdentityContinuityMonitor.detect_crisis() (P0-004)
       |
       v
CrisisProtocol.execute() (P0-005)
       |
       +---> Record snapshot (if IC < 0.7)
       |
       +---> Generate IdentityCrisisEvent (if IC < 0.5)
       |
       v
IdentityCrisisEvent.to_workspace_event()
       |
       v
WorkspaceEventBroadcaster.broadcast()
       |
       v
DreamEventListener / NeuromodulationEventListener
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | Claude Opus 4.5 | Initial task specification |
| 1.1.0 | 2026-01-11 | Claude Opus 4.5 | Added dream integration documentation |
| 2.0.0 | 2026-01-12 | Claude Opus 4.5 | **COMPLETE REWRITE**: Added codebase audit, FSV requirements, NO mock data policy, fail-fast policy, exact file paths, comprehensive test cases, verification commands, integration documentation |
