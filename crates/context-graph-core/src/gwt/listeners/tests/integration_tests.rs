//! Integration tests for all workspace event listeners
//!
//! # Constitution Compliance
//!
//! Per AP-26: DreamEventListener requires TriggerManager and callback.
//! Tests that include IdentityCritical events below threshold MUST use
//! the full `new()` constructor with proper callback handling.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use parking_lot::Mutex;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::dream::TriggerManager;
use crate::gwt::listeners::{DreamConsolidationCallback, DreamEventListener, MetaCognitiveEventListener, NeuromodulationEventListener};
use crate::gwt::meta_cognitive::MetaCognitiveLoop;
use crate::gwt::workspace::{WorkspaceEvent, WorkspaceEventListener};
use crate::neuromod::{NeuromodulationManager, DA_BASELINE};
use chrono::Utc;

// ============================================================
// Integration test: All listeners receive events
// ============================================================

#[tokio::test]
async fn test_all_listeners_receive_all_events() {
    println!("=== INTEGRATION: All listeners receive all event types ===");

    // Setup all listeners
    // AP-26: Use full constructor with TriggerManager and callback for IC event handling
    let dream_queue = Arc::new(RwLock::new(Vec::new()));
    let trigger_manager = Arc::new(Mutex::new(TriggerManager::new()));
    let dream_trigger_count = Arc::new(AtomicUsize::new(0));
    let dtc = Arc::clone(&dream_trigger_count);
    let dream_callback: DreamConsolidationCallback = Arc::new(move |_| {
        dtc.fetch_add(1, Ordering::SeqCst);
    });
    let dream_listener = DreamEventListener::new(
        dream_queue.clone(),
        trigger_manager,
        dream_callback,
    );

    let neuromod = Arc::new(RwLock::new(NeuromodulationManager::new()));
    let neuromod_listener = NeuromodulationEventListener::new(neuromod.clone());

    let meta_cognitive = Arc::new(RwLock::new(MetaCognitiveLoop::new()));
    let epistemic_flag = Arc::new(AtomicBool::new(false));
    let meta_listener = MetaCognitiveEventListener::new(meta_cognitive.clone(), epistemic_flag.clone());

    // Create events - IC=0.4 < 0.5 threshold will trigger dream consolidation
    let events = vec![
        WorkspaceEvent::MemoryEnters {
            id: Uuid::new_v4(),
            order_parameter: 0.85,
            timestamp: Utc::now(),
            fingerprint: None, // TASK-IDENTITY-P0-006
        },
        WorkspaceEvent::MemoryExits {
            id: Uuid::new_v4(),
            order_parameter: 0.65,
            timestamp: Utc::now(),
        },
        WorkspaceEvent::WorkspaceEmpty {
            duration_ms: 5000, // Must be >= 5000ms threshold per TASK-FIX-001
            timestamp: Utc::now(),
        },
        WorkspaceEvent::WorkspaceConflict {
            memories: vec![Uuid::new_v4(), Uuid::new_v4()],
            timestamp: Utc::now(),
        },
        WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.4, // Below threshold 0.5 - will trigger dream
            previous_status: "Warning".to_string(),
            current_status: "Critical".to_string(),
            reason: "Test".to_string(),
            timestamp: Utc::now(),
        },
    ];

    // Broadcast to all listeners (no panics expected)
    for event in &events {
        dream_listener.on_event(event);
        neuromod_listener.on_event(event);
        meta_listener.on_event(event);
    }

    // Verify expected state changes
    let dream_queue_len = {
        let queue = dream_queue.read().await;
        queue.len()
    };
    assert_eq!(dream_queue_len, 1, "Dream queue should have 1 MemoryExits");

    let final_da = {
        let mgr = neuromod.read().await;
        mgr.get_hopfield_beta()
    };
    assert!(final_da > DA_BASELINE, "Dopamine should be above baseline");

    assert!(
        meta_listener.is_epistemic_action_triggered(),
        "Epistemic flag should be set"
    );

    // AP-26: Verify dream consolidation was triggered by IC < 0.5
    let triggers = dream_trigger_count.load(Ordering::SeqCst);
    assert_eq!(triggers, 1, "Dream should trigger once for IC=0.4 < 0.5 threshold");

    println!("EVIDENCE: All listeners correctly processed all event types without panic");
    println!("EVIDENCE: Dream consolidation triggered {} time(s) per AP-26", triggers);
}
