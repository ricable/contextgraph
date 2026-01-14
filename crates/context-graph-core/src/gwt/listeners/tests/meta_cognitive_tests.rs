//! Tests for MetaCognitiveEventListener

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::gwt::listeners::MetaCognitiveEventListener;
use crate::gwt::meta_cognitive::MetaCognitiveLoop;
use crate::gwt::workspace::{WorkspaceEvent, WorkspaceEventListener};
use chrono::Utc;

// ============================================================
// FSV Tests for MetaCognitiveEventListener
// ============================================================

#[tokio::test]
async fn test_fsv_meta_listener_workspace_empty() {
    println!("=== FSV: MetaCognitiveEventListener - WorkspaceEmpty at threshold ===");

    // SETUP
    let meta_cognitive = Arc::new(RwLock::new(MetaCognitiveLoop::new()));
    let epistemic_flag = Arc::new(AtomicBool::new(false));
    let listener = MetaCognitiveEventListener::new(meta_cognitive.clone(), epistemic_flag.clone());

    // BEFORE
    let before_flag = listener.is_epistemic_action_triggered();
    println!("BEFORE: epistemic_action_triggered = {}", before_flag);
    assert!(!before_flag, "Flag must start as false");

    // EXECUTE with duration at threshold (5000ms)
    // PRD Section 2.5.3: "workspace_empty: No memory r > 0.8 for 5s"
    let event = WorkspaceEvent::WorkspaceEmpty {
        duration_ms: 5000,
        timestamp: Utc::now(),
    };
    listener.on_event(&event);

    // AFTER
    let after_flag = listener.is_epistemic_action_triggered();
    println!("AFTER: epistemic_action_triggered = {}", after_flag);

    // VERIFY
    assert!(after_flag, "Flag must be set to true when duration >= 5000ms");

    // EVIDENCE
    println!("EVIDENCE: Epistemic action flag correctly set on WorkspaceEmpty at threshold");
}

#[tokio::test]
async fn test_meta_listener_ignores_other_events() {
    println!("=== TEST: MetaCognitiveEventListener ignores non-WorkspaceEmpty ===");

    let meta_cognitive = Arc::new(RwLock::new(MetaCognitiveLoop::new()));
    let epistemic_flag = Arc::new(AtomicBool::new(false));
    let listener = MetaCognitiveEventListener::new(meta_cognitive.clone(), epistemic_flag.clone());

    // Send MemoryEnters - should be ignored
    let event = WorkspaceEvent::MemoryEnters {
        id: Uuid::new_v4(),
        order_parameter: 0.85,
        timestamp: Utc::now(),
        fingerprint: None, // TASK-IDENTITY-P0-006
    };
    listener.on_event(&event);

    // Send MemoryExits - should be ignored
    let event = WorkspaceEvent::MemoryExits {
        id: Uuid::new_v4(),
        order_parameter: 0.65,
        timestamp: Utc::now(),
    };
    listener.on_event(&event);

    assert!(
        !listener.is_epistemic_action_triggered(),
        "Flag should remain false for non-WorkspaceEmpty events"
    );
    println!("EVIDENCE: MetaCognitiveEventListener correctly ignores non-WorkspaceEmpty events");
}

#[tokio::test]
async fn test_meta_listener_zero_duration() {
    println!("=== EDGE CASE: MetaCognitiveEventListener with duration=0 ===");

    let meta_cognitive = Arc::new(RwLock::new(MetaCognitiveLoop::new()));
    let epistemic_flag = Arc::new(AtomicBool::new(false));
    let listener = MetaCognitiveEventListener::new(meta_cognitive.clone(), epistemic_flag.clone());

    // Execute with zero duration (below 5000ms threshold)
    let event = WorkspaceEvent::WorkspaceEmpty {
        duration_ms: 0,
        timestamp: Utc::now(),
    };
    listener.on_event(&event);

    // PRD Section 2.5.3: requires 5s (5000ms) before triggering
    assert!(
        !listener.is_epistemic_action_triggered(),
        "Flag must remain false for duration_ms=0 (below 5000ms threshold)"
    );
    println!("EVIDENCE: Zero duration correctly handled (no trigger)");
}

#[tokio::test]
async fn test_meta_listener_reset() {
    println!("=== TEST: MetaCognitiveEventListener reset ===");

    let meta_cognitive = Arc::new(RwLock::new(MetaCognitiveLoop::new()));
    let epistemic_flag = Arc::new(AtomicBool::new(false));
    let listener = MetaCognitiveEventListener::new(meta_cognitive.clone(), epistemic_flag.clone());

    // Trigger the flag (must use duration >= 5000ms threshold)
    let event = WorkspaceEvent::WorkspaceEmpty {
        duration_ms: 6000,
        timestamp: Utc::now(),
    };
    listener.on_event(&event);
    assert!(listener.is_epistemic_action_triggered());

    // Reset
    listener.reset_epistemic_action();
    assert!(!listener.is_epistemic_action_triggered());

    println!("EVIDENCE: Epistemic action flag correctly reset");
}

#[tokio::test]
async fn test_workspace_empty_below_threshold_no_trigger() {
    println!("=== FSV: WorkspaceEmpty below 5000ms threshold ===");

    // SETUP
    let meta_cognitive = Arc::new(RwLock::new(MetaCognitiveLoop::new()));
    let epistemic_flag = Arc::new(AtomicBool::new(false));
    let listener = MetaCognitiveEventListener::new(meta_cognitive.clone(), epistemic_flag.clone());

    // BEFORE
    assert!(!listener.is_epistemic_action_triggered(), "Flag must start false");

    // EXECUTE with 4999ms (below threshold)
    let event = WorkspaceEvent::WorkspaceEmpty {
        duration_ms: 4999,
        timestamp: Utc::now(),
    };
    listener.on_event(&event);

    // AFTER - flag must remain false
    assert!(
        !listener.is_epistemic_action_triggered(),
        "Flag must remain false for duration_ms < 5000"
    );

    println!("EVIDENCE: duration_ms=4999 correctly did NOT trigger epistemic action");
}

#[tokio::test]
async fn test_workspace_empty_at_threshold_triggers() {
    println!("=== FSV: WorkspaceEmpty at exactly 5000ms threshold ===");

    // SETUP
    let meta_cognitive = Arc::new(RwLock::new(MetaCognitiveLoop::new()));
    let epistemic_flag = Arc::new(AtomicBool::new(false));
    let listener = MetaCognitiveEventListener::new(meta_cognitive.clone(), epistemic_flag.clone());

    // BEFORE
    assert!(!listener.is_epistemic_action_triggered(), "Flag must start false");

    // EXECUTE with exactly 5000ms (at threshold)
    let event = WorkspaceEvent::WorkspaceEmpty {
        duration_ms: 5000,
        timestamp: Utc::now(),
    };
    listener.on_event(&event);

    // AFTER - flag must be true
    assert!(
        listener.is_epistemic_action_triggered(),
        "Flag must be true for duration_ms >= 5000"
    );

    println!("EVIDENCE: duration_ms=5000 correctly triggered epistemic action");
}

#[tokio::test]
async fn test_workspace_empty_above_threshold_triggers() {
    println!("=== FSV: WorkspaceEmpty above 5000ms threshold ===");

    // SETUP
    let meta_cognitive = Arc::new(RwLock::new(MetaCognitiveLoop::new()));
    let epistemic_flag = Arc::new(AtomicBool::new(false));
    let listener = MetaCognitiveEventListener::new(meta_cognitive.clone(), epistemic_flag.clone());

    // BEFORE
    assert!(!listener.is_epistemic_action_triggered(), "Flag must start false");

    // EXECUTE with 10000ms (above threshold)
    let event = WorkspaceEvent::WorkspaceEmpty {
        duration_ms: 10000,
        timestamp: Utc::now(),
    };
    listener.on_event(&event);

    // AFTER - flag must be true
    assert!(
        listener.is_epistemic_action_triggered(),
        "Flag must be true for duration_ms >= 5000"
    );

    println!("EVIDENCE: duration_ms=10000 correctly triggered epistemic action");
}
