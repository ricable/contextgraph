//! Event wiring integration tests for GwtSystem
//!
//! TASK-GWT-P1-002: Tests for:
//! - Listener wiring on GwtSystem creation
//! - MemoryEnters event dopamine boost
//! - MemoryExits event dream queue
//! - WorkspaceEmpty epistemic action trigger
//! - Concurrent broadcast handling

use crate::gwt::{GwtSystem, WorkspaceEvent};
use std::sync::Arc;
use uuid::Uuid;

// ============================================================
// Test: GwtSystem has listeners wired
// ============================================================
#[tokio::test]
async fn test_gwt_system_listeners_wired() {
    println!("=== FSV: GwtSystem has listeners wired ===");

    let gwt = GwtSystem::new().await.expect("GwtSystem must create");

    // VERIFY: 4 listeners should be registered (TASK-IDENTITY-P0-006 adds identity listener)
    let listener_count = gwt.event_broadcaster.listener_count().await;
    println!("Listener count: {}", listener_count);

    assert_eq!(listener_count, 4, "Should have 4 listeners registered");
    println!("EVIDENCE: GwtSystem correctly wired 4 event listeners (including identity)");
}

// ============================================================
// Test: MemoryEnters event boosts dopamine
// ============================================================
#[tokio::test]
async fn test_memory_enters_boosts_dopamine() {
    println!("=== FSV: MemoryEnters event boosts dopamine ===");

    let gwt = GwtSystem::new().await.unwrap();

    // BEFORE
    let before_da = {
        let mgr = gwt.neuromod_manager.read().await;
        mgr.get_hopfield_beta()
    };
    println!("BEFORE: dopamine = {:.3}", before_da);

    // EXECUTE - Broadcast MemoryEnters event
    let event = WorkspaceEvent::MemoryEnters {
        id: Uuid::new_v4(),
        order_parameter: 0.85,
        timestamp: chrono::Utc::now(),
        fingerprint: None, // TASK-IDENTITY-P0-006
    };
    gwt.event_broadcaster.broadcast(event).await;

    // AFTER - Read via separate lock
    let after_da = {
        let mgr = gwt.neuromod_manager.read().await;
        mgr.get_hopfield_beta()
    };
    println!("AFTER: dopamine = {:.3}", after_da);

    // VERIFY
    assert!(
        after_da > before_da,
        "Dopamine should increase on MemoryEnters"
    );
    let expected = before_da + 0.2; // DA_WORKSPACE_INCREMENT
    assert!(
        (after_da - expected).abs() < f32::EPSILON,
        "Expected dopamine {:.3}, got {:.3}",
        expected,
        after_da
    );

    println!("EVIDENCE: MemoryEnters correctly boosted dopamine by 0.2");
}

// ============================================================
// Test: MemoryExits event queues for dream
// ============================================================
#[tokio::test]
async fn test_memory_exits_queues_for_dream() {
    println!("=== FSV: MemoryExits event queues for dream ===");

    let gwt = GwtSystem::new().await.unwrap();

    // BEFORE
    let before_len = gwt.dream_queue_len().await;
    println!("BEFORE: dream_queue.len() = {}", before_len);
    assert_eq!(before_len, 0, "Dream queue should start empty");

    // EXECUTE - Broadcast MemoryExits event
    let memory_id = Uuid::new_v4();
    let event = WorkspaceEvent::MemoryExits {
        id: memory_id,
        order_parameter: 0.65,
        timestamp: chrono::Utc::now(),
    };
    gwt.event_broadcaster.broadcast(event).await;

    // AFTER
    let after_len = gwt.dream_queue_len().await;
    println!("AFTER: dream_queue.len() = {}", after_len);

    // VERIFY
    assert_eq!(after_len, 1, "Dream queue should have 1 item");

    // Drain and verify the ID
    let drained = gwt.drain_dream_queue().await;
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0], memory_id, "Queued ID should match");

    println!(
        "EVIDENCE: MemoryExits correctly queued memory {:?} for dream",
        memory_id
    );
}

// ============================================================
// Test: WorkspaceEmpty triggers epistemic action
// ============================================================
#[tokio::test]
async fn test_workspace_empty_triggers_epistemic_action() {
    println!("=== FSV: WorkspaceEmpty triggers epistemic action ===");

    let gwt = GwtSystem::new().await.unwrap();

    // BEFORE
    let before_flag = gwt.is_epistemic_action_triggered();
    println!("BEFORE: epistemic_action_triggered = {}", before_flag);
    assert!(!before_flag, "Flag should start as false");

    // EXECUTE - Broadcast WorkspaceEmpty event (>= 5000ms per TASK-FIX-001)
    let event = WorkspaceEvent::WorkspaceEmpty {
        duration_ms: 5000, // Must be >= 5000ms threshold per PRD 2.5.3
        timestamp: chrono::Utc::now(),
    };
    gwt.event_broadcaster.broadcast(event).await;

    // AFTER
    let after_flag = gwt.is_epistemic_action_triggered();
    println!("AFTER: epistemic_action_triggered = {}", after_flag);

    // VERIFY
    assert!(after_flag, "Flag should be set after WorkspaceEmpty");

    // Reset and verify
    gwt.reset_epistemic_action();
    assert!(!gwt.is_epistemic_action_triggered(), "Flag should be reset");

    println!("EVIDENCE: WorkspaceEmpty correctly triggers epistemic action");
}

// ============================================================
// Full Integration Test: Event Flow
// ============================================================
#[tokio::test]
async fn test_full_event_flow_integration() {
    println!("=== FSV: Full Event Flow Integration ===");

    let gwt = GwtSystem::new().await.unwrap();

    // Verify initial state
    println!("\nINITIAL STATE:");
    println!(
        "  - listener_count: {}",
        gwt.event_broadcaster.listener_count().await
    );
    println!("  - dream_queue.len: {}", gwt.dream_queue_len().await);
    println!(
        "  - epistemic_action: {}",
        gwt.is_epistemic_action_triggered()
    );

    // Get initial dopamine
    let initial_da = {
        let mgr = gwt.neuromod_manager.read().await;
        mgr.get_hopfield_beta()
    };
    println!("  - dopamine: {:.3}", initial_da);

    // SCENARIO 1: Memory enters workspace
    println!("\nSCENARIO 1: Memory enters workspace");
    let winner_id = Uuid::new_v4();
    gwt.event_broadcaster
        .broadcast(WorkspaceEvent::MemoryEnters {
            id: winner_id,
            order_parameter: 0.88,
            timestamp: chrono::Utc::now(),
            fingerprint: None, // TASK-IDENTITY-P0-006
        })
        .await;

    let after_enter_da = {
        let mgr = gwt.neuromod_manager.read().await;
        mgr.get_hopfield_beta()
    };
    println!("  - dopamine after entry: {:.3}", after_enter_da);
    assert!(after_enter_da > initial_da, "DA should increase on entry");

    // SCENARIO 2: Losing memory exits
    println!("\nSCENARIO 2: Loser exits workspace");
    let loser_id = Uuid::new_v4();
    gwt.event_broadcaster
        .broadcast(WorkspaceEvent::MemoryExits {
            id: loser_id,
            order_parameter: 0.65,
            timestamp: chrono::Utc::now(),
        })
        .await;

    let dream_queue_len = gwt.dream_queue_len().await;
    println!("  - dream_queue.len: {}", dream_queue_len);
    assert_eq!(dream_queue_len, 1, "Loser should be queued");

    // SCENARIO 3: Workspace becomes empty
    println!("\nSCENARIO 3: Workspace becomes empty");
    gwt.event_broadcaster
        .broadcast(WorkspaceEvent::WorkspaceEmpty {
            duration_ms: 5000, // Must be >= 5000ms threshold per PRD 2.5.3
            timestamp: chrono::Utc::now(),
        })
        .await;

    println!(
        "  - epistemic_action: {}",
        gwt.is_epistemic_action_triggered()
    );
    assert!(
        gwt.is_epistemic_action_triggered(),
        "Epistemic action should trigger"
    );

    // FINAL STATE
    println!("\nFINAL STATE:");
    let final_da = {
        let mgr = gwt.neuromod_manager.read().await;
        mgr.get_hopfield_beta()
    };
    println!(
        "  - dopamine: {:.3} (started at {:.3})",
        final_da, initial_da
    );
    println!("  - dream_queue.len: {}", gwt.dream_queue_len().await);
    println!(
        "  - epistemic_action: {}",
        gwt.is_epistemic_action_triggered()
    );

    println!("\nEVIDENCE: Full event flow correctly processed");
}

// ============================================================
// Edge Case: Concurrent broadcasts
// ============================================================
#[tokio::test]
async fn test_concurrent_event_broadcast() {
    println!("=== EDGE CASE: Concurrent event broadcasts ===");

    let gwt = Arc::new(GwtSystem::new().await.unwrap());

    // Spawn multiple concurrent broadcast tasks
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let gwt_clone = Arc::clone(&gwt);
            tokio::spawn(async move {
                let event = WorkspaceEvent::MemoryEnters {
                    id: Uuid::new_v4(),
                    order_parameter: 0.8 + (i as f32 * 0.01),
                    timestamp: chrono::Utc::now(),
                    fingerprint: None, // TASK-IDENTITY-P0-006
                };
                gwt_clone.event_broadcaster.broadcast(event).await;
                i
            })
        })
        .collect();

    // Wait for all tasks
    for handle in handles {
        handle.await.expect("Task should complete");
    }

    // Verify dopamine increased 10 times
    let final_da = {
        let mgr = gwt.neuromod_manager.read().await;
        mgr.get_hopfield_beta()
    };

    // 10 increments of 0.2 = 2.0 increase from baseline 3.0 = 5.0 (clamped)
    println!("Final dopamine: {:.3}", final_da);
    assert!(final_da >= 3.0 + 0.2, "Dopamine should have increased");

    println!("EVIDENCE: Concurrent broadcasts handled without deadlock");
}
