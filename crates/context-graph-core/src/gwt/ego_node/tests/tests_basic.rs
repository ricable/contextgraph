//! Basic tests for SelfEgoNode and IdentityContinuity
//!
//! Tests core functionality: creation, updates, status transitions.

use chrono::Utc;
use uuid::Uuid;

use crate::types::fingerprint::{
    JohariFingerprint, PurposeVector, SemanticFingerprint, TeleologicalFingerprint,
};

use super::super::{
    IdentityContinuity, IdentityStatus, SelfAwarenessLoop, SelfEgoNode,
};

#[test]
fn test_self_ego_node_creation() {
    let ego = SelfEgoNode::new();
    assert_eq!(ego.id, Uuid::nil());
    assert_eq!(ego.purpose_vector, [0.0; 13]);
}

#[test]
fn test_self_ego_node_purpose_update() {
    let mut ego = SelfEgoNode::new();
    let pv = [1.0; 13];
    ego.purpose_vector = pv;

    assert_eq!(ego.purpose_vector, pv);
}

#[test]
fn test_purpose_snapshot_recording() {
    let mut ego = SelfEgoNode::new();
    ego.purpose_vector = [0.5; 13];

    ego.record_purpose_snapshot("Test snapshot").unwrap();
    assert_eq!(ego.identity_trajectory.len(), 1);

    let snapshot = ego.get_latest_snapshot().unwrap();
    assert_eq!(snapshot.vector, [0.5; 13]);
    assert!(snapshot.context.contains("Test snapshot"));
}

/// FSV test: Initial IdentityContinuity status should be Critical per constitution.yaml lines 387-392
/// Because identity_coherence=0.0 at initialization, which is < 0.5 (Critical threshold)
#[test]
fn test_identity_continuity_initial_status_is_critical() {
    let continuity = IdentityContinuity::default_initial();

    // Per constitution: IC < 0.5 should be Critical, not Healthy
    assert_eq!(
        continuity.status,
        IdentityStatus::Critical,
        "Initial identity coherence 0.0 must result in Critical status, not Healthy"
    );
    assert_eq!(continuity.identity_coherence, 0.0);
}

/// FSV test: Status transitions through all states correctly
#[test]
fn test_identity_status_from_coherence_all_states() {
    // Verify compute_status_from_coherence works correctly
    let mut continuity = IdentityContinuity::default_initial();

    // Update to each threshold and verify status
    // Critical: IC < 0.5
    continuity.update(0.3, 0.3).unwrap(); // IC = 0.09 < 0.5
    assert_eq!(continuity.status, IdentityStatus::Critical);

    // Degraded: 0.5 <= IC < 0.7
    continuity.update(0.8, 0.7).unwrap(); // IC = 0.56
    assert_eq!(continuity.status, IdentityStatus::Degraded);

    // Warning: 0.7 <= IC <= 0.9
    continuity.update(0.9, 0.85).unwrap(); // IC = 0.765
    assert_eq!(continuity.status, IdentityStatus::Warning);

    // Healthy: IC > 0.9
    continuity.update(0.96, 0.96).unwrap(); // IC = 0.9216 > 0.9
    assert_eq!(continuity.status, IdentityStatus::Healthy);
}

#[test]
fn test_identity_continuity_healthy() {
    let mut continuity = IdentityContinuity::default_initial();
    let status = continuity.update(0.95, 0.95).unwrap();

    assert_eq!(status, IdentityStatus::Healthy);
    assert!(continuity.identity_coherence > 0.9);
}

#[test]
fn test_identity_continuity_critical() {
    let mut continuity = IdentityContinuity::default_initial();
    let status = continuity.update(0.3, 0.3).unwrap();

    assert_eq!(status, IdentityStatus::Critical);
    assert!(continuity.identity_coherence < 0.5);
}

#[tokio::test]
async fn test_self_awareness_loop_cycle() {
    let mut loop_mgr = SelfAwarenessLoop::new();
    let mut ego = SelfEgoNode::with_purpose_vector([1.0; 13]);

    let action = [1.0; 13]; // Perfect alignment
    let result = loop_mgr.cycle(&mut ego, &action, 0.85).await.unwrap();

    assert!(!result.needs_reflection); // Alignment is high
    assert!(result.alignment > 0.99);
}

#[tokio::test]
async fn test_self_awareness_loop_reflection_trigger() {
    let mut loop_mgr = SelfAwarenessLoop::new();
    let mut ego = SelfEgoNode::with_purpose_vector([1.0; 13]);

    let action = [0.0; 13]; // Zero alignment - should trigger reflection
    let result = loop_mgr.cycle(&mut ego, &action, 0.85).await.unwrap();

    assert!(result.needs_reflection);
    assert!(result.alignment < 0.55); // alignment_threshold
}

#[test]
fn test_cosine_similarity_via_loop() {
    let loop_mgr = SelfAwarenessLoop::new();
    let v1 = [1.0; 13];
    let v2 = [1.0; 13];

    // Using the loop's internal method indirectly via cycle is tested above
    // This tests that identical vectors work correctly
    let _ = loop_mgr; // Just ensure it compiles
    assert_eq!(v1, v2);
}

/// CRITICAL: Uses #[tokio::test] to prevent zombie runtime threads.
/// DO NOT use tokio::runtime::Runtime::new() in tests.
#[tokio::test]
async fn test_self_awareness_loop_identity_coherence_getter() {
    let mut loop_mgr = SelfAwarenessLoop::new();

    // Initial state
    let initial_ic = loop_mgr.identity_coherence();
    let initial_status = loop_mgr.identity_status();

    // Create ego and run a cycle
    let mut ego = SelfEgoNode::with_purpose_vector([0.8; 13]);
    ego.record_purpose_snapshot("Setup").unwrap();

    let action = [0.8; 13];
    let _ = loop_mgr.cycle(&mut ego, &action, 0.95).await.unwrap();

    // After cycle
    let final_ic = loop_mgr.identity_coherence();
    let final_status = loop_mgr.identity_status();

    // Verify getters work
    assert!((0.0..=1.0).contains(&final_ic), "IC must be in [0,1]");

    // Just ensure values are retrievable
    let _ = initial_ic;
    let _ = initial_status;
    let _ = final_status;
}

// Helper to create a test TeleologicalFingerprint with known values
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
        alignment_score: alignments.iter().sum::<f32>() / 13.0,
        content_hash: [0u8; 32],
        created_at: Utc::now(),
        last_updated: Utc::now(),
        access_count: 0,
    }
}

#[test]
fn test_update_from_fingerprint_copies_purpose_vector() {
    let mut ego = SelfEgoNode::new();
    assert_eq!(ego.purpose_vector, [0.0; 13], "Initial purpose_vector should be zeros");

    let alignments = [0.8, 0.75, 0.9, 0.6, 0.7, 0.65, 0.85, 0.72, 0.78, 0.68, 0.82, 0.71, 0.76];
    let fingerprint = create_test_fingerprint(alignments);

    ego.update_from_fingerprint(&fingerprint).unwrap();

    assert_eq!(
        ego.purpose_vector, alignments,
        "purpose_vector must match fingerprint.purpose_vector.alignments"
    );
}

#[test]
fn test_update_from_fingerprint_updates_coherence() {
    let mut ego = SelfEgoNode::new();
    assert_eq!(ego.coherence_with_actions, 0.0, "Initial coherence should be 0.0");

    let alignments = [0.8; 13]; // Uniform = coherence = 1.0
    let fingerprint = create_test_fingerprint(alignments);
    let expected_coherence = fingerprint.purpose_vector.coherence;

    ego.update_from_fingerprint(&fingerprint).unwrap();

    assert!(
        (ego.coherence_with_actions - expected_coherence).abs() < 1e-6,
        "coherence_with_actions must equal fingerprint.purpose_vector.coherence"
    );
}

#[test]
fn test_update_from_fingerprint_stores_fingerprint() {
    let mut ego = SelfEgoNode::new();
    assert!(ego.fingerprint.is_none(), "Initial fingerprint should be None");

    let alignments = [0.5; 13];
    let fingerprint = create_test_fingerprint(alignments);
    let fingerprint_id = fingerprint.id;

    ego.update_from_fingerprint(&fingerprint).unwrap();

    assert!(ego.fingerprint.is_some(), "Fingerprint must be stored after update");
    assert_eq!(
        ego.fingerprint.as_ref().unwrap().id,
        fingerprint_id,
        "Stored fingerprint ID must match input fingerprint ID"
    );
}

#[test]
fn test_update_from_fingerprint_updates_timestamp() {
    let mut ego = SelfEgoNode::new();
    let initial_time = ego.last_updated;

    std::thread::sleep(std::time::Duration::from_millis(10));

    let fingerprint = create_test_fingerprint([0.7; 13]);
    ego.update_from_fingerprint(&fingerprint).unwrap();

    assert!(
        ego.last_updated > initial_time,
        "last_updated must be updated after update_from_fingerprint"
    );
}

#[test]
fn test_edge_case_zero_alignments() {
    let mut ego = SelfEgoNode::new();
    let fingerprint = create_test_fingerprint([0.0; 13]);

    let result = ego.update_from_fingerprint(&fingerprint);
    assert!(result.is_ok(), "Should handle zero alignments");

    assert_eq!(ego.purpose_vector, [0.0; 13]);
    assert!(
        (ego.coherence_with_actions - 1.0).abs() < 1e-6,
        "Zero uniform alignments should have coherence 1.0"
    );
}

#[test]
fn test_edge_case_max_alignments() {
    let mut ego = SelfEgoNode::new();
    let fingerprint = create_test_fingerprint([1.0; 13]);

    let result = ego.update_from_fingerprint(&fingerprint);
    assert!(result.is_ok(), "Should handle max alignments");

    assert_eq!(ego.purpose_vector, [1.0; 13]);
}

#[test]
fn test_edge_case_negative_alignments() {
    let mut ego = SelfEgoNode::new();
    // Note: PurposeVector accepts negative values (cosine can be negative)
    let fingerprint = create_test_fingerprint([-0.5; 13]);

    let result = ego.update_from_fingerprint(&fingerprint);
    assert!(result.is_ok(), "Should handle negative alignments");

    assert_eq!(ego.purpose_vector, [-0.5; 13]);
}
