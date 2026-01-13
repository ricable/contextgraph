//! Tests for DreamEventListener

use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::gwt::listeners::DreamEventListener;
use crate::gwt::workspace::{WorkspaceEvent, WorkspaceEventListener};
use chrono::Utc;

// ============================================================
// FSV Tests for DreamEventListener
// ============================================================

#[tokio::test]
async fn test_fsv_dream_listener_memory_exits() {
    println!("=== FSV: DreamEventListener - MemoryExits ===");

    // SETUP
    let dream_queue = Arc::new(RwLock::new(Vec::new()));
    let listener = DreamEventListener::new(dream_queue.clone());
    let memory_id = Uuid::new_v4();

    // BEFORE
    let before_len = {
        let queue = dream_queue.read().await;
        queue.len()
    };
    println!("BEFORE: queue.len() = {}", before_len);
    assert_eq!(before_len, 0, "Queue must start empty");

    // EXECUTE
    let event = WorkspaceEvent::MemoryExits {
        id: memory_id,
        order_parameter: 0.65,
        timestamp: Utc::now(),
    };
    listener.on_event(&event);

    // AFTER - SEPARATE READ
    let after_len = {
        let queue = dream_queue.read().await;
        queue.len()
    };
    let queued_id = {
        let queue = dream_queue.read().await;
        queue.first().cloned()
    };
    println!("AFTER: queue.len() = {}", after_len);

    // VERIFY
    assert_eq!(after_len, 1, "Queue must have exactly 1 item");
    assert_eq!(queued_id, Some(memory_id), "Queued ID must match");

    // EVIDENCE
    println!("EVIDENCE: Memory {:?} correctly queued for dream replay", memory_id);
}

#[tokio::test]
async fn test_dream_listener_ignores_other_events() {
    println!("=== TEST: DreamEventListener ignores non-MemoryExits ===");

    let dream_queue = Arc::new(RwLock::new(Vec::new()));
    let listener = DreamEventListener::new(dream_queue.clone());

    // Send MemoryEnters - should be ignored
    let event = WorkspaceEvent::MemoryEnters {
        id: Uuid::new_v4(),
        order_parameter: 0.85,
        timestamp: Utc::now(),
        fingerprint: None, // TASK-IDENTITY-P0-006
    };
    listener.on_event(&event);

    // Send WorkspaceEmpty - should be ignored
    let event = WorkspaceEvent::WorkspaceEmpty {
        duration_ms: 1000,
        timestamp: Utc::now(),
    };
    listener.on_event(&event);

    let queue_len = {
        let queue = dream_queue.read().await;
        queue.len()
    };

    assert_eq!(queue_len, 0, "Queue should remain empty for non-MemoryExits events");
    println!("EVIDENCE: DreamEventListener correctly ignores non-MemoryExits events");
}

#[tokio::test]
async fn test_dream_listener_identity_critical() {
    println!("=== TEST: DreamEventListener handles IdentityCritical ===");

    let dream_queue = Arc::new(RwLock::new(Vec::new()));
    let listener = DreamEventListener::new(dream_queue.clone());

    // Send IdentityCritical - should log but not queue
    let event = WorkspaceEvent::IdentityCritical {
        identity_coherence: 0.35,
        previous_status: "Warning".to_string(),
        current_status: "Critical".to_string(),
        reason: "Test critical".to_string(),
        timestamp: Utc::now(),
    };
    listener.on_event(&event);

    let queue_len = {
        let queue = dream_queue.read().await;
        queue.len()
    };

    assert_eq!(queue_len, 0, "Queue should remain empty for IdentityCritical");
    println!("EVIDENCE: IdentityCritical event handled without queuing");
}
