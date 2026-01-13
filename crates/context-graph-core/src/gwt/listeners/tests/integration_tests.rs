//! Integration tests for all workspace event listeners

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::gwt::listeners::{DreamEventListener, MetaCognitiveEventListener, NeuromodulationEventListener};
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
    let dream_queue = Arc::new(RwLock::new(Vec::new()));
    let dream_listener = DreamEventListener::new(dream_queue.clone());

    let neuromod = Arc::new(RwLock::new(NeuromodulationManager::new()));
    let neuromod_listener = NeuromodulationEventListener::new(neuromod.clone());

    let meta_cognitive = Arc::new(RwLock::new(MetaCognitiveLoop::new()));
    let epistemic_flag = Arc::new(AtomicBool::new(false));
    let meta_listener = MetaCognitiveEventListener::new(meta_cognitive.clone(), epistemic_flag.clone());

    // Create events
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
            duration_ms: 500,
            timestamp: Utc::now(),
        },
        WorkspaceEvent::WorkspaceConflict {
            memories: vec![Uuid::new_v4(), Uuid::new_v4()],
            timestamp: Utc::now(),
        },
        WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.4,
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

    println!("EVIDENCE: All listeners correctly processed all event types without panic");
}
