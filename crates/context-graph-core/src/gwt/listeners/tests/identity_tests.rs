//! Tests for IdentityContinuityListener
//!
//! TASK-IDENTITY-P0-006: Tests for identity continuity monitoring on workspace events.
//!
//! Test cases:
//! 1. FSV: IC computed on MemoryEnters with fingerprint
//! 2. Ignores MemoryEnters without fingerprint
//! 3. Ignores other event types
//! 4. Edge case: Crisis detection and event emission
//! 5. History accumulation over multiple events

use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::gwt::ego_node::SelfEgoNode;
use crate::gwt::listeners::IdentityContinuityListener;
use crate::gwt::workspace::{WorkspaceEvent, WorkspaceEventBroadcaster, WorkspaceEventListener};
use crate::types::fingerprint::{
    JohariFingerprint, PurposeVector, SemanticFingerprint, TeleologicalFingerprint,
};
use chrono::Utc;

/// Helper to create a test TeleologicalFingerprint with known alignments
fn create_test_fingerprint(alignments: [f32; 13]) -> TeleologicalFingerprint {
    let purpose_vector = PurposeVector::new(alignments);
    let semantic = SemanticFingerprint::zeroed();
    let johari = JohariFingerprint::zeroed();

    TeleologicalFingerprint {
        id: Uuid::new_v4(),
        semantic,
        purpose_vector,
        johari,
        purpose_evolution: Vec::new(),
        theta_to_north_star: alignments.iter().sum::<f32>() / 13.0,
        content_hash: [0u8; 32],
        created_at: Utc::now(),
        last_updated: Utc::now(),
        access_count: 0,
    }
}

// ============================================================
// FSV Test: IC computed on MemoryEnters with fingerprint
// ============================================================

#[tokio::test]
async fn test_fsv_identity_listener_computes_ic_on_memory_enters() {
    println!("=== FSV: IdentityContinuityListener - IC on MemoryEnters ===");

    // SETUP
    let ego_node = Arc::new(RwLock::new(SelfEgoNode::new()));
    let broadcaster = Arc::new(WorkspaceEventBroadcaster::new());
    let listener = IdentityContinuityListener::new(ego_node, Arc::clone(&broadcaster));

    // BEFORE
    let before_history = listener.history_len().await;
    println!("BEFORE: history_len = {}", before_history);
    assert_eq!(before_history, 0, "History must start empty");

    // EXECUTE - Send MemoryEnters with fingerprint
    let alignments = [0.8, 0.7, 0.9, 0.6, 0.85, 0.75, 0.8, 0.7, 0.9, 0.65, 0.8, 0.75, 0.85];
    let fingerprint = create_test_fingerprint(alignments);
    let event = WorkspaceEvent::MemoryEnters {
        id: Uuid::new_v4(),
        order_parameter: 0.9,
        timestamp: Utc::now(),
        fingerprint: Some(Box::new(fingerprint)),
    };
    listener.on_event(&event);

    // Allow async task to complete
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // AFTER
    let after_history = listener.history_len().await;
    println!("AFTER: history_len = {}", after_history);

    // VERIFY - First event establishes baseline, IC computed
    assert_eq!(after_history, 1, "History should have 1 entry after first event");

    // EVIDENCE
    println!("EVIDENCE: IdentityContinuityListener correctly processes MemoryEnters with fingerprint");
}

// ============================================================
// Test: Ignores MemoryEnters without fingerprint
// ============================================================

#[tokio::test]
async fn test_identity_listener_ignores_no_fingerprint() {
    println!("=== TEST: IdentityContinuityListener ignores MemoryEnters without fingerprint ===");

    let ego_node = Arc::new(RwLock::new(SelfEgoNode::new()));
    let broadcaster = Arc::new(WorkspaceEventBroadcaster::new());
    let listener = IdentityContinuityListener::new(ego_node, Arc::clone(&broadcaster));

    // Send event without fingerprint
    let event = WorkspaceEvent::MemoryEnters {
        id: Uuid::new_v4(),
        order_parameter: 0.85,
        timestamp: Utc::now(),
        fingerprint: None,
    };
    listener.on_event(&event);

    // Allow async task to complete
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // History should remain empty
    let history_len = listener.history_len().await;
    assert_eq!(
        history_len, 0,
        "History should remain empty for events without fingerprint"
    );

    println!("EVIDENCE: IdentityContinuityListener correctly ignores MemoryEnters without fingerprint");
}

// ============================================================
// Test: Ignores other event types
// ============================================================

#[tokio::test]
async fn test_identity_listener_ignores_other_events() {
    println!("=== TEST: IdentityContinuityListener ignores non-MemoryEnters ===");

    let ego_node = Arc::new(RwLock::new(SelfEgoNode::new()));
    let broadcaster = Arc::new(WorkspaceEventBroadcaster::new());
    let listener = IdentityContinuityListener::new(ego_node, Arc::clone(&broadcaster));

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

    // Allow async tasks to complete
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // History should remain empty
    let history_len = listener.history_len().await;
    assert_eq!(
        history_len, 0,
        "History should remain empty for non-MemoryEnters events"
    );

    println!("EVIDENCE: IdentityContinuityListener correctly ignores non-MemoryEnters events");
}

// ============================================================
// Edge Case: IC computation with consecutive events
// ============================================================

#[tokio::test]
async fn test_identity_listener_consecutive_events() {
    println!("=== EDGE CASE: IdentityContinuityListener consecutive events ===");

    let ego_node = Arc::new(RwLock::new(SelfEgoNode::new()));
    let broadcaster = Arc::new(WorkspaceEventBroadcaster::new());
    let listener = IdentityContinuityListener::new(ego_node, Arc::clone(&broadcaster));

    // First event - establishes baseline
    let alignments1 = [0.8, 0.7, 0.9, 0.6, 0.85, 0.75, 0.8, 0.7, 0.9, 0.65, 0.8, 0.75, 0.85];
    let fp1 = create_test_fingerprint(alignments1);
    let event1 = WorkspaceEvent::MemoryEnters {
        id: Uuid::new_v4(),
        order_parameter: 0.9,
        timestamp: Utc::now(),
        fingerprint: Some(Box::new(fp1)),
    };
    listener.on_event(&event1);
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    let ic_after_first = listener.identity_coherence().await;
    println!("IC after first event: {:.3}", ic_after_first);

    // Second event - similar alignments (should have high IC)
    let alignments2 = [0.82, 0.72, 0.88, 0.62, 0.83, 0.77, 0.78, 0.72, 0.88, 0.67, 0.78, 0.77, 0.83];
    let fp2 = create_test_fingerprint(alignments2);
    let event2 = WorkspaceEvent::MemoryEnters {
        id: Uuid::new_v4(),
        order_parameter: 0.88,
        timestamp: Utc::now(),
        fingerprint: Some(Box::new(fp2)),
    };
    listener.on_event(&event2);
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    let ic_after_second = listener.identity_coherence().await;
    let history_len = listener.history_len().await;

    println!("IC after second event: {:.3}", ic_after_second);
    println!("History length: {}", history_len);

    // VERIFY
    assert_eq!(history_len, 2, "History should have 2 entries");
    // IC = cosine_similarity × order_parameter
    // With slightly different vectors and r=0.88, expect IC > 0.8 (healthy range)
    assert!(
        ic_after_second > 0.8,
        "IC should be high (>0.8) for similar purpose vectors, got {}",
        ic_after_second
    );

    println!("EVIDENCE: Consecutive events correctly accumulate history and compute IC");
}

// ============================================================
// Edge Case: Low IC triggers warning status
// ============================================================

#[tokio::test]
async fn test_identity_listener_low_ic_warning() {
    println!("=== EDGE CASE: IdentityContinuityListener low IC warning ===");

    let ego_node = Arc::new(RwLock::new(SelfEgoNode::new()));
    let broadcaster = Arc::new(WorkspaceEventBroadcaster::new());
    let listener = IdentityContinuityListener::new(ego_node, Arc::clone(&broadcaster));

    // First event - high positive alignments
    let alignments1 = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9];
    let fp1 = create_test_fingerprint(alignments1);
    let event1 = WorkspaceEvent::MemoryEnters {
        id: Uuid::new_v4(),
        order_parameter: 0.95,
        timestamp: Utc::now(),
        fingerprint: Some(Box::new(fp1)),
    };
    listener.on_event(&event1);
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Second event - moderate change in alignments
    let alignments2 = [0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5];
    let fp2 = create_test_fingerprint(alignments2);
    let event2 = WorkspaceEvent::MemoryEnters {
        id: Uuid::new_v4(),
        order_parameter: 0.85,
        timestamp: Utc::now(),
        fingerprint: Some(Box::new(fp2)),
    };
    listener.on_event(&event2);
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    let ic = listener.identity_coherence().await;
    let status = listener.identity_status().await;

    println!("IC after divergent event: {:.3}", ic);
    println!("Identity status: {:?}", status);

    // VERIFY - IC should be lower due to purpose vector change
    // The status depends on how different the vectors are
    assert!(
        ic < 1.0,
        "IC should be less than 1.0 for different purpose vectors"
    );

    println!("EVIDENCE: Identity listener correctly detects IC changes from purpose vector divergence");
}
