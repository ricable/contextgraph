//! Full State Verification (FSV) lifecycle tests
//!
//! Comprehensive lifecycle and integration tests for ego_node module.

use super::super::{
    cosine_similarity_13d, IdentityContinuity, IdentityContinuityMonitor, IdentityStatus,
    PurposeVectorHistory, PurposeVectorHistoryProvider,
};

/// Helper: Create a purpose vector with uniform values
fn uniform_pv(val: f32) -> [f32; 13] {
    [val; 13]
}

// =========================================================================
// Full State Verification Lifecycle Tests
// =========================================================================

#[test]
fn fsv_identity_continuity_full_lifecycle() {
    let result = IdentityContinuity::new(0.8, 0.9);

    assert!(
        (result.identity_coherence - 0.72).abs() < 1e-6,
        "IC should be 0.8 * 0.9 = 0.72"
    );
    assert_eq!(result.status, IdentityStatus::Warning);
    assert!(!result.is_in_crisis());

    let first = IdentityContinuity::first_vector();
    assert_eq!(first.identity_coherence, 1.0);
    assert_eq!(first.status, IdentityStatus::Healthy);

    let serialized = bincode::serialize(&result).unwrap();
    let restored: IdentityContinuity = bincode::deserialize(&serialized).unwrap();
    assert_eq!(result, restored);
}

#[test]
fn fsv_purpose_vector_history_full_lifecycle() {
    let mut history = PurposeVectorHistory::with_max_size(5);
    assert!(history.is_empty());
    assert!(!history.is_first_vector());

    let prev1 = history.push(uniform_pv(0.5), "First");
    assert!(prev1.is_none());
    assert!(history.is_first_vector());
    assert_eq!(*history.current().unwrap(), uniform_pv(0.5));

    let prev2 = history.push(uniform_pv(0.6), "Second");
    assert!(prev2.is_some());
    assert!(!history.is_first_vector());
    assert_eq!(*history.current().unwrap(), uniform_pv(0.6));
    assert_eq!(*history.previous().unwrap(), uniform_pv(0.5));

    history.push(uniform_pv(0.7), "Third");
    history.push(uniform_pv(0.8), "Fourth");
    history.push(uniform_pv(0.9), "Fifth");
    assert_eq!(history.len(), 5);

    history.push(uniform_pv(1.0), "Sixth");
    assert_eq!(history.len(), 5);

    let oldest = history.history().front().unwrap().vector[0];
    assert!((oldest - 0.6).abs() < 1e-6);

    let serialized = bincode::serialize(&history).unwrap();
    let restored: PurposeVectorHistory = bincode::deserialize(&serialized).unwrap();
    assert_eq!(restored.len(), history.len());
    assert_eq!(restored.max_size, history.max_size);
    assert_eq!(restored.current(), history.current());
}

#[test]
fn fsv_identity_continuity_monitor_complete_lifecycle() {
    let mut monitor = IdentityContinuityMonitor::with_capacity(10);
    assert!(monitor.is_empty());
    assert_eq!(monitor.history_len(), 0);
    assert!(monitor.last_result().is_none());

    let pv1 = [0.85, 0.78, 0.92, 0.67, 0.73, 0.61, 0.88, 0.75, 0.81, 0.69, 0.84, 0.72, 0.79];
    let result1 = monitor.compute_continuity(&pv1, 0.95, "Initial purpose alignment");

    assert!(monitor.is_first_vector());
    assert_eq!(result1.identity_coherence, 1.0);
    assert_eq!(result1.status, IdentityStatus::Healthy);
    assert!(!monitor.is_in_crisis());

    let pv2 = [0.84, 0.77, 0.91, 0.68, 0.74, 0.62, 0.87, 0.74, 0.80, 0.70, 0.83, 0.71, 0.78];
    let result2 = monitor.compute_continuity(&pv2, 0.93, "Minor adjustment");

    assert!(!monitor.is_first_vector());
    assert!(result2.recent_continuity > 0.99);
    assert!(result2.identity_coherence > 0.9);

    let pv4 = [0.2, 0.95, 0.3, 0.85, 0.25, 0.9, 0.35, 0.88, 0.28, 0.92, 0.32, 0.87, 0.29];
    monitor.compute_continuity(&pv4, 0.9, "Shifted");
    let result4 = monitor.compute_continuity(&pv4, 0.2, "Low consciousness");

    assert!((result4.recent_continuity - 1.0).abs() < 1e-6);
    assert!((result4.identity_coherence - 0.2).abs() < 1e-6);
    assert_eq!(result4.status, IdentityStatus::Critical);
    assert!(monitor.is_in_crisis());

    let serialized = bincode::serialize(&monitor).expect("serialize");
    let restored: IdentityContinuityMonitor = bincode::deserialize(&serialized).expect("deserialize");

    assert_eq!(restored.history_len(), monitor.history_len());
    assert_eq!(restored.identity_coherence(), monitor.identity_coherence());
    assert_eq!(restored.crisis_threshold(), monitor.crisis_threshold());
}

#[test]
fn fsv_identity_continuity_monitor_edge_cases() {
    let mut monitor1 = IdentityContinuityMonitor::new();
    let pv_pos = uniform_pv(1.0);
    let pv_neg = uniform_pv(-1.0);

    monitor1.compute_continuity(&pv_pos, 0.9, "Positive");
    let result1 = monitor1.compute_continuity(&pv_neg, 0.9, "Negative");

    assert!((result1.recent_continuity - (-1.0)).abs() < 1e-6);
    assert_eq!(result1.identity_coherence, 0.0);
    assert_eq!(result1.status, IdentityStatus::Critical);

    let mut monitor2 = IdentityContinuityMonitor::new();
    let pv = uniform_pv(0.8);

    monitor2.compute_continuity(&pv, 0.9, "First");
    let result2 = monitor2.compute_continuity(&pv, 0.0, "Zero sync");

    assert_eq!(result2.identity_coherence, 0.0);
    assert_eq!(result2.status, IdentityStatus::Critical);

    let mut monitor3 = IdentityContinuityMonitor::new();
    monitor3.compute_continuity(&uniform_pv(1.0), 1.0, "First");
    let result3 = monitor3.compute_continuity(&uniform_pv(1.0), 1.0, "Perfect");

    assert_eq!(result3.identity_coherence, 1.0);
    assert_eq!(result3.status, IdentityStatus::Healthy);

    let mut monitor4 = IdentityContinuityMonitor::new();
    monitor4.compute_continuity(&uniform_pv(0.8), 1.0, "First");
    let result4 = monitor4.compute_continuity(&uniform_pv(0.8), 0.7, "Boundary");

    assert!((result4.identity_coherence - 0.7).abs() < 1e-6);
    assert!(!monitor4.is_in_crisis());
}

#[test]
fn fsv_cosine_similarity_13d_mathematical_properties() {
    let a = [0.8, 0.7, 0.9, 0.6, 0.75, 0.65, 0.85, 0.72, 0.78, 0.68, 0.82, 0.71, 0.76];
    let b = [0.5, 0.9, 0.3, 0.85, 0.4, 0.88, 0.35, 0.82, 0.45, 0.87, 0.38, 0.84, 0.42];

    let cos_ab = cosine_similarity_13d(&a, &b);
    let cos_ba = cosine_similarity_13d(&b, &a);

    assert!(
        (cos_ab - cos_ba).abs() < 1e-10,
        "Symmetry violation: {} != {}",
        cos_ab,
        cos_ba
    );

    let cos_aa = cosine_similarity_13d(&a, &a);
    assert!((cos_aa - 1.0).abs() < 1e-10);

    let scaled_a: [f32; 13] = std::array::from_fn(|i| a[i] * 3.7);
    let cos_scaled = cosine_similarity_13d(&scaled_a, &a);
    assert!((cos_scaled - 1.0).abs() < 1e-6);

    let test_vectors: Vec<[f32; 13]> = vec![
        std::array::from_fn(|i| (i as f32 * 0.1).sin()),
        std::array::from_fn(|i| (i as f32 * 0.2).cos()),
        std::array::from_fn(|i| ((i + 3) as f32).sqrt()),
        std::array::from_fn(|_| -0.5),
        std::array::from_fn(|i| if i % 2 == 0 { 1.0 } else { -1.0 }),
    ];

    for (i, v1) in test_vectors.iter().enumerate() {
        for (j, v2) in test_vectors.iter().enumerate() {
            let cos = cosine_similarity_13d(v1, v2);
            assert!(
                (-1.0..=1.0).contains(&cos),
                "Bounded violation at ({}, {}): {}",
                i,
                j,
                cos
            );
        }
    }
}
