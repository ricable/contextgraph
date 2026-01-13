//! Tests for NeuromodulationEventListener

use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::gwt::listeners::NeuromodulationEventListener;
use crate::gwt::workspace::{WorkspaceEvent, WorkspaceEventListener};
use crate::neuromod::{NeuromodulationManager, DA_BASELINE};
use chrono::Utc;

// ============================================================
// FSV Tests for NeuromodulationEventListener
// ============================================================

#[tokio::test]
async fn test_fsv_neuromod_listener_dopamine_boost() {
    println!("=== FSV: NeuromodulationEventListener - Dopamine Boost ===");

    // SETUP
    let neuromod = Arc::new(RwLock::new(NeuromodulationManager::new()));
    let listener = NeuromodulationEventListener::new(neuromod.clone());
    let memory_id = Uuid::new_v4();

    // BEFORE - Read via separate lock
    let before_da = {
        let mgr = neuromod.read().await;
        mgr.get_hopfield_beta() // Returns dopamine value
    };
    println!("BEFORE: dopamine = {:.3}", before_da);
    assert!(
        (before_da - DA_BASELINE).abs() < f32::EPSILON,
        "Dopamine must start at baseline"
    );

    // EXECUTE
    let event = WorkspaceEvent::MemoryEnters {
        id: memory_id,
        order_parameter: 0.85,
        timestamp: Utc::now(),
        fingerprint: None, // TASK-IDENTITY-P0-006
    };
    listener.on_event(&event);

    // AFTER - Read via SEPARATE lock
    let after_da = {
        let mgr = neuromod.read().await;
        mgr.get_hopfield_beta()
    };
    println!("AFTER: dopamine = {:.3}", after_da);

    // VERIFY
    let expected_da = before_da + 0.2; // DA_WORKSPACE_INCREMENT
    assert!(
        (after_da - expected_da).abs() < f32::EPSILON,
        "Expected dopamine {:.3}, got {:.3}",
        expected_da,
        after_da
    );

    // EVIDENCE
    println!(
        "EVIDENCE: Dopamine correctly increased by 0.2 (from {:.3} to {:.3})",
        before_da, after_da
    );
}

#[tokio::test]
async fn test_neuromod_listener_ignores_other_events() {
    println!("=== TEST: NeuromodulationEventListener ignores non-MemoryEnters ===");

    let neuromod = Arc::new(RwLock::new(NeuromodulationManager::new()));
    let listener = NeuromodulationEventListener::new(neuromod.clone());

    let initial_da = {
        let mgr = neuromod.read().await;
        mgr.get_hopfield_beta()
    };

    // Send MemoryExits - should be ignored
    let event = WorkspaceEvent::MemoryExits {
        id: Uuid::new_v4(),
        order_parameter: 0.65,
        timestamp: Utc::now(),
    };
    listener.on_event(&event);

    // Send WorkspaceEmpty - should be ignored
    let event = WorkspaceEvent::WorkspaceEmpty {
        duration_ms: 1000,
        timestamp: Utc::now(),
    };
    listener.on_event(&event);

    let final_da = {
        let mgr = neuromod.read().await;
        mgr.get_hopfield_beta()
    };

    assert!(
        (final_da - initial_da).abs() < f32::EPSILON,
        "Dopamine should remain unchanged for non-MemoryEnters events"
    );
    println!("EVIDENCE: NeuromodulationEventListener correctly ignores non-MemoryEnters events");
}

#[tokio::test]
async fn test_neuromod_listener_at_max() {
    println!("=== EDGE CASE: NeuromodulationEventListener at max dopamine ===");

    let neuromod = Arc::new(RwLock::new(NeuromodulationManager::new()));
    let listener = NeuromodulationEventListener::new(neuromod.clone());

    // Set dopamine to max
    {
        let mut mgr = neuromod.write().await;
        use crate::neuromod::ModulatorType;
        mgr.set(ModulatorType::Dopamine, 5.0).unwrap();
    }

    let before_da = {
        let mgr = neuromod.read().await;
        mgr.get_hopfield_beta()
    };
    println!("BEFORE: dopamine = {:.3} (at max)", before_da);

    // Trigger event
    let event = WorkspaceEvent::MemoryEnters {
        id: Uuid::new_v4(),
        order_parameter: 0.85,
        timestamp: Utc::now(),
        fingerprint: None, // TASK-IDENTITY-P0-006
    };
    listener.on_event(&event);

    let after_da = {
        let mgr = neuromod.read().await;
        mgr.get_hopfield_beta()
    };
    println!("AFTER: dopamine = {:.3}", after_da);

    // Verify clamped to max
    assert!(
        after_da <= 5.0,
        "Dopamine must be clamped to max (5.0), got {}",
        after_da
    );
    println!("EVIDENCE: Dopamine correctly clamped to max");
}
