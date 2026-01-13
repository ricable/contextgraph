//! Tests for Crisis Protocol (TASK-IDENTITY-P0-005)
//!
//! Full State Verification tests for CrisisProtocol.execute()

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use crate::gwt::ego_node::{
    CrisisAction, CrisisDetectionResult, CrisisProtocol, IdentityContinuityMonitor,
    IdentityCrisisEvent, IdentityStatus, SelfEgoNode, CRISIS_EVENT_COOLDOWN,
};
use crate::gwt::workspace::WorkspaceEvent;

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
        time_since_last_event: if can_emit {
            None
        } else {
            Some(Duration::from_secs(5))
        },
        can_emit_event: can_emit,
    }
}

#[tokio::test]
async fn test_protocol_healthy_no_action() {
    println!("=== FSV: Healthy status = no actions ===");

    let ego = create_test_ego();
    let protocol = CrisisProtocol::new(ego.clone());
    let mut monitor = IdentityContinuityMonitor::new();

    let detection = create_detection(0.95, IdentityStatus::Healthy, IdentityStatus::Healthy, true);

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

    let detection = create_detection(0.75, IdentityStatus::Healthy, IdentityStatus::Warning, true);

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
    assert!(result
        .actions
        .iter()
        .any(|a| matches!(a, CrisisAction::SnapshotRecorded { .. })));
    assert!(!result
        .actions
        .iter()
        .any(|a| matches!(a, CrisisAction::EventGenerated { .. })));

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
    assert!(result
        .actions
        .iter()
        .any(|a| matches!(a, CrisisAction::SnapshotRecorded { .. })));
    assert!(result
        .actions
        .iter()
        .any(|a| matches!(a, CrisisAction::EventGenerated { .. })));
    assert!(result
        .actions
        .iter()
        .any(|a| matches!(a, CrisisAction::IntrospectionTriggered)));

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
    let has_cooldown_action = result
        .actions
        .iter()
        .any(|a| matches!(a, CrisisAction::EventSkippedCooldown { .. }));
    assert!(has_cooldown_action);

    // No introspection triggered
    let has_introspection = result
        .actions
        .iter()
        .any(|a| matches!(a, CrisisAction::IntrospectionTriggered));
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
    println!(
        "BEFORE: event.identity_coherence = {}",
        event.identity_coherence
    );

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

    let detection = create_detection(0.20, IdentityStatus::Healthy, IdentityStatus::Critical, true);

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
    println!(
        "EVIDENCE: Reason contains 'entered critical': {}",
        event.reason
    );
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
    println!(
        "EVIDENCE: Reason contains 'remains critical': {}",
        event.reason
    );
}

#[tokio::test]
async fn test_cooldown_remaining_calculation() {
    println!("=== FSV: Cooldown remaining time calculation ===");

    let ego = create_test_ego();
    let protocol = CrisisProtocol::new(ego.clone());
    let mut monitor = IdentityContinuityMonitor::new();

    // Simulate 5 seconds since last event, cooldown is 30 seconds
    let elapsed = Duration::from_secs(5);
    let expected_remaining = CRISIS_EVENT_COOLDOWN.saturating_sub(elapsed);

    let detection = CrisisDetectionResult {
        identity_coherence: 0.30,
        previous_status: IdentityStatus::Warning,
        current_status: IdentityStatus::Critical,
        status_changed: true,
        entering_crisis: false,
        entering_critical: true,
        recovering: false,
        time_since_last_event: Some(elapsed),
        can_emit_event: false,
    };

    let result = protocol.execute(detection, &mut monitor).await.unwrap();

    // Find the cooldown action
    let cooldown_action = result.actions.iter().find(|a| {
        matches!(a, CrisisAction::EventSkippedCooldown { .. })
    });

    assert!(cooldown_action.is_some());
    if let Some(CrisisAction::EventSkippedCooldown { remaining }) = cooldown_action {
        println!("AFTER: remaining cooldown = {:?}", remaining);
        // Should be approximately 25 seconds (30 - 5)
        assert_eq!(*remaining, expected_remaining);
        println!("EVIDENCE: Cooldown remaining correctly calculated as {:?}", remaining);
    }
}
