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
    println!("=== FSV: MetaCognitiveEventListener - WorkspaceEmpty ===");

    // SETUP
    let meta_cognitive = Arc::new(RwLock::new(MetaCognitiveLoop::new()));
    let epistemic_flag = Arc::new(AtomicBool::new(false));
    let listener = MetaCognitiveEventListener::new(meta_cognitive.clone(), epistemic_flag.clone());

    // BEFORE
    let before_flag = listener.is_epistemic_action_triggered();
    println!("BEFORE: epistemic_action_triggered = {}", before_flag);
    assert!(!before_flag, "Flag must start as false");

    // EXECUTE
    let event = WorkspaceEvent::WorkspaceEmpty {
        duration_ms: 500,
        timestamp: Utc::now(),
    };
    listener.on_event(&event);

    // AFTER
    let after_flag = listener.is_epistemic_action_triggered();
    println!("AFTER: epistemic_action_triggered = {}", after_flag);

    // VERIFY
    assert!(after_flag, "Flag must be set to true");

    // EVIDENCE
    println!("EVIDENCE: Epistemic action flag correctly set on WorkspaceEmpty");
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

    // Execute with zero duration
    let event = WorkspaceEvent::WorkspaceEmpty {
        duration_ms: 0,
        timestamp: Utc::now(),
    };
    listener.on_event(&event);

    assert!(
        listener.is_epistemic_action_triggered(),
        "Flag should be set even with zero duration"
    );
    println!("EVIDENCE: Zero duration handled correctly");
}

#[tokio::test]
async fn test_meta_listener_reset() {
    println!("=== TEST: MetaCognitiveEventListener reset ===");

    let meta_cognitive = Arc::new(RwLock::new(MetaCognitiveLoop::new()));
    let epistemic_flag = Arc::new(AtomicBool::new(false));
    let listener = MetaCognitiveEventListener::new(meta_cognitive.clone(), epistemic_flag.clone());

    // Trigger the flag
    let event = WorkspaceEvent::WorkspaceEmpty {
        duration_ms: 100,
        timestamp: Utc::now(),
    };
    listener.on_event(&event);
    assert!(listener.is_epistemic_action_triggered());

    // Reset
    listener.reset_epistemic_action();
    assert!(!listener.is_epistemic_action_triggered());

    println!("EVIDENCE: Epistemic action flag correctly reset");
}
