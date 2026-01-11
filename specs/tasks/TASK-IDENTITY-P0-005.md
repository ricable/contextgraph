# Task Specification: Crisis Protocol

**Task ID:** TASK-IDENTITY-P0-005
**Version:** 1.0.0
**Status:** Ready
**Layer:** Logic
**Sequence:** 5
**Estimated Complexity:** Medium

---

## Metadata

| Field | Value |
|-------|-------|
| Implements | REQ-IDENTITY-006, REQ-IDENTITY-007 |
| Depends On | TASK-IDENTITY-P0-004 |
| Blocks | TASK-IDENTITY-P0-006 |
| Priority | P0 - Critical |

---

## Context

When identity coherence drops below thresholds, the system must execute a crisis protocol:

1. **IC < 0.7 (Warning/Degraded):** Record purpose snapshot for trajectory analysis
2. **IC < 0.5 (Critical):** Emit `WorkspaceEvent::IdentityCritical` to trigger dream intervention

Per constitution.yaml lines 387-392:
- "IC < 0.7 trigger identity crisis protocol"
- "IC < 0.5 dream consolidation"

The crisis protocol must:
- Record context for later analysis
- Emit appropriate workspace events
- Coordinate with the GWT broadcast system
- Respect cooldown to prevent event spam

---

## Input Context Files

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/gwt/ego_node.rs` | IdentityContinuityMonitor, CrisisDetectionResult |
| `crates/context-graph-core/src/gwt/workspace.rs` | WorkspaceEvent, WorkspaceBroadcaster |
| `specs/tasks/TASK-IDENTITY-P0-004.md` | Crisis detection |
| `docs2/constitution.yaml` | Lines 387-392 |

---

## Prerequisites

- [x] TASK-IDENTITY-P0-004 completed
- [x] CrisisDetectionResult exists
- [x] WorkspaceEvent::IdentityCritical exists in workspace.rs

---

## Scope

### In Scope

1. Create `CrisisProtocol` struct with execution logic
2. Implement `execute()` method for crisis handling
3. Generate `WorkspaceEvent::IdentityCritical` for critical state
4. Record purpose snapshot with detailed context
5. Return `CrisisProtocolResult` with actions taken

### Out of Scope

- Actual workspace broadcasting (TASK-IDENTITY-P0-006)
- Dream system integration (handled by Dream subsystem listening to events)

---

## Definition of Done

### Exact Signatures Required

```rust
// File: crates/context-graph-core/src/gwt/ego_node.rs

/// Result of crisis protocol execution
#[derive(Debug, Clone)]
pub struct CrisisProtocolResult {
    /// The crisis detection that triggered this
    pub detection: CrisisDetectionResult,
    /// Whether a purpose snapshot was recorded
    pub snapshot_recorded: bool,
    /// Context message for the snapshot
    pub snapshot_context: Option<String>,
    /// Event generated (if any)
    pub event: Option<IdentityCrisisEvent>,
    /// Whether event was actually emittable (cooldown check)
    pub event_emitted: bool,
    /// Actions taken
    pub actions: Vec<CrisisAction>,
}

/// Actions that can be taken during crisis protocol
#[derive(Debug, Clone, PartialEq)]
pub enum CrisisAction {
    /// Recorded purpose snapshot
    SnapshotRecorded { context: String },
    /// Generated identity critical event
    EventGenerated { event_type: String },
    /// Skipped event due to cooldown
    EventSkippedCooldown { remaining: Duration },
    /// Triggered introspection mode
    IntrospectionTriggered,
}

/// Event payload for identity crisis
#[derive(Debug, Clone)]
pub struct IdentityCrisisEvent {
    /// Current identity coherence
    pub identity_coherence: f32,
    /// Previous status
    pub previous_status: IdentityStatus,
    /// Current status
    pub current_status: IdentityStatus,
    /// Reason for crisis
    pub reason: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Executes crisis protocol based on detection results
pub struct CrisisProtocol {
    /// Reference to ego node for snapshot recording
    ego_node: Arc<RwLock<SelfEgoNode>>,
}

impl CrisisProtocol {
    /// Create new protocol with ego node reference
    pub fn new(ego_node: Arc<RwLock<SelfEgoNode>>) -> Self;

    /// Execute crisis protocol based on detection result
    ///
    /// # Algorithm
    /// 1. If in crisis (IC < 0.7):
    ///    a. Record purpose snapshot with context
    /// 2. If critical (IC < 0.5):
    ///    a. Generate IdentityCrisisEvent
    ///    b. Check cooldown for emission
    /// 3. Return result with all actions taken
    ///
    /// # Arguments
    /// * `detection` - Result from detect_crisis()
    /// * `monitor` - Mutable reference to monitor for cooldown update
    ///
    /// # Returns
    /// CrisisProtocolResult with actions taken and event (if generated)
    pub async fn execute(
        &self,
        detection: CrisisDetectionResult,
        monitor: &mut IdentityContinuityMonitor,
    ) -> CoreResult<CrisisProtocolResult>;
}

impl IdentityCrisisEvent {
    /// Create event from detection result
    pub fn from_detection(detection: &CrisisDetectionResult, reason: impl Into<String>) -> Self;

    /// Convert to WorkspaceEvent for broadcasting
    pub fn to_workspace_event(&self) -> WorkspaceEvent;
}
```

### Constraints

1. Snapshot MUST be recorded for any crisis (IC < 0.7)
2. Event MUST only be generated for critical state (IC < 0.5)
3. Event MUST only be emitted if cooldown allows
4. Cooldown MUST be marked after event emission
5. Snapshot context MUST include IC value and status
6. All operations MUST be async-safe (use Arc<RwLock>)
7. NO blocking operations in execute()

### Verification Commands

```bash
# Build
cargo build -p context-graph-core

# Run crisis protocol tests
cargo test -p context-graph-core crisis_protocol

# Clippy
cargo clippy -p context-graph-core -- -D warnings
```

---

## Pseudo Code

```rust
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct CrisisProtocolResult {
    pub detection: CrisisDetectionResult,
    pub snapshot_recorded: bool,
    pub snapshot_context: Option<String>,
    pub event: Option<IdentityCrisisEvent>,
    pub event_emitted: bool,
    pub actions: Vec<CrisisAction>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CrisisAction {
    SnapshotRecorded { context: String },
    EventGenerated { event_type: String },
    EventSkippedCooldown { remaining: Duration },
    IntrospectionTriggered,
}

#[derive(Debug, Clone)]
pub struct IdentityCrisisEvent {
    pub identity_coherence: f32,
    pub previous_status: IdentityStatus,
    pub current_status: IdentityStatus,
    pub reason: String,
    pub timestamp: DateTime<Utc>,
}

impl IdentityCrisisEvent {
    pub fn from_detection(detection: &CrisisDetectionResult, reason: impl Into<String>) -> Self {
        Self {
            identity_coherence: detection.identity_coherence,
            previous_status: detection.previous_status,
            current_status: detection.current_status,
            reason: reason.into(),
            timestamp: Utc::now(),
        }
    }

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

pub struct CrisisProtocol {
    ego_node: Arc<RwLock<SelfEgoNode>>,
}

impl CrisisProtocol {
    pub fn new(ego_node: Arc<RwLock<SelfEgoNode>>) -> Self {
        Self { ego_node }
    }

    pub async fn execute(
        &self,
        detection: CrisisDetectionResult,
        monitor: &mut IdentityContinuityMonitor,
    ) -> CoreResult<CrisisProtocolResult> {
        let mut actions = Vec::new();
        let mut snapshot_recorded = false;
        let mut snapshot_context = None;
        let mut event = None;
        let mut event_emitted = false;

        // Step 1: Handle any crisis (IC < 0.7)
        if detection.current_status != IdentityStatus::Healthy {
            let context = format!(
                "Identity crisis: IC={:.4}, status={:?} (was {:?})",
                detection.identity_coherence,
                detection.current_status,
                detection.previous_status
            );

            // Record snapshot
            {
                let mut ego = self.ego_node.write().await;
                ego.record_purpose_snapshot(&context)?;
            }

            snapshot_recorded = true;
            snapshot_context = Some(context.clone());
            actions.push(CrisisAction::SnapshotRecorded { context });
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
            } else {
                let remaining = CRISIS_EVENT_COOLDOWN
                    .checked_sub(detection.time_since_last_event.unwrap_or_default())
                    .unwrap_or_default();
                actions.push(CrisisAction::EventSkippedCooldown { remaining });
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

---

## Files to Create

None - all additions go to existing file.

---

## Files to Modify

| File | Changes |
|------|---------|
| `crates/context-graph-core/src/gwt/ego_node.rs` | Add CrisisProtocol, CrisisProtocolResult, IdentityCrisisEvent, CrisisAction |

---

## Validation Criteria

| Criterion | Verification Method |
|-----------|---------------------|
| Snapshot recorded for IC < 0.7 | Unit test |
| No snapshot for Healthy | Unit test |
| Event generated for Critical | Unit test |
| Event NOT generated for Warning/Degraded | Unit test |
| Cooldown respected | Unit test |
| Cooldown marked after emission | Unit test |
| to_workspace_event converts correctly | Unit test |
| Async safety | Integration test |

---

## Test Cases

```rust
#[cfg(test)]
mod crisis_protocol_tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    fn create_test_ego() -> Arc<RwLock<SelfEgoNode>> {
        Arc::new(RwLock::new(SelfEgoNode::new()))
    }

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
            time_since_last_event: None,
            can_emit_event: can_emit,
        }
    }

    #[tokio::test]
    async fn test_snapshot_recorded_for_warning() {
        let ego = create_test_ego();
        let protocol = CrisisProtocol::new(ego.clone());
        let mut monitor = IdentityContinuityMonitor::new();

        let detection = create_detection(
            0.75,
            IdentityStatus::Healthy,
            IdentityStatus::Warning,
            true,
        );

        let result = protocol.execute(detection, &mut monitor).await.unwrap();

        assert!(result.snapshot_recorded);
        assert!(result.snapshot_context.is_some());
        assert!(result.event.is_none()); // No event for Warning
        assert!(!result.event_emitted);

        // Verify snapshot in ego node
        let ego_read = ego.read().await;
        assert!(!ego_read.identity_trajectory.is_empty());
    }

    #[tokio::test]
    async fn test_no_snapshot_for_healthy() {
        let ego = create_test_ego();
        let protocol = CrisisProtocol::new(ego);
        let mut monitor = IdentityContinuityMonitor::new();

        let detection = create_detection(
            0.95,
            IdentityStatus::Healthy,
            IdentityStatus::Healthy,
            true,
        );

        let result = protocol.execute(detection, &mut monitor).await.unwrap();

        assert!(!result.snapshot_recorded);
        assert!(result.snapshot_context.is_none());
        assert!(result.event.is_none());
    }

    #[tokio::test]
    async fn test_event_generated_for_critical() {
        let ego = create_test_ego();
        let protocol = CrisisProtocol::new(ego);
        let mut monitor = IdentityContinuityMonitor::new();

        let detection = create_detection(
            0.30,
            IdentityStatus::Warning,
            IdentityStatus::Critical,
            true,
        );

        let result = protocol.execute(detection, &mut monitor).await.unwrap();

        assert!(result.snapshot_recorded);
        assert!(result.event.is_some());
        assert!(result.event_emitted);

        let event = result.event.unwrap();
        assert_eq!(event.current_status, IdentityStatus::Critical);
        assert!(event.reason.contains("critical"));
    }

    #[tokio::test]
    async fn test_event_blocked_by_cooldown() {
        let ego = create_test_ego();
        let protocol = CrisisProtocol::new(ego);
        let mut monitor = IdentityContinuityMonitor::new();

        let detection = create_detection(
            0.30,
            IdentityStatus::Warning,
            IdentityStatus::Critical,
            false, // Cooldown active
        );

        let result = protocol.execute(detection, &mut monitor).await.unwrap();

        assert!(result.event.is_some()); // Event generated
        assert!(!result.event_emitted); // But not emitted

        // Should have cooldown skip action
        assert!(result.actions.iter().any(|a| matches!(a, CrisisAction::EventSkippedCooldown { .. })));
    }

    #[tokio::test]
    async fn test_to_workspace_event() {
        let detection = create_detection(
            0.30,
            IdentityStatus::Warning,
            IdentityStatus::Critical,
            true,
        );

        let event = IdentityCrisisEvent::from_detection(&detection, "Test reason");
        let ws_event = event.to_workspace_event();

        // Verify it converts to correct variant
        if let WorkspaceEvent::IdentityCritical { identity_coherence, reason, .. } = ws_event {
            assert!((identity_coherence - 0.30).abs() < 1e-6);
            assert_eq!(reason, "Test reason");
        } else {
            panic!("Expected IdentityCritical event");
        }
    }

    #[tokio::test]
    async fn test_actions_list_comprehensive() {
        let ego = create_test_ego();
        let protocol = CrisisProtocol::new(ego);
        let mut monitor = IdentityContinuityMonitor::new();

        let detection = create_detection(
            0.30,
            IdentityStatus::Healthy,
            IdentityStatus::Critical,
            true,
        );

        let result = protocol.execute(detection, &mut monitor).await.unwrap();

        // Should have: snapshot, event generated, introspection triggered
        assert!(result.actions.iter().any(|a| matches!(a, CrisisAction::SnapshotRecorded { .. })));
        assert!(result.actions.iter().any(|a| matches!(a, CrisisAction::EventGenerated { .. })));
        assert!(result.actions.iter().any(|a| matches!(a, CrisisAction::IntrospectionTriggered)));
    }
}
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | Claude Opus 4.5 | Initial task specification |
