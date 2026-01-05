//! Tests for GraphEdge traversal recording and tracking.

use std::time::{SystemTime, UNIX_EPOCH};

use uuid::Uuid;

use crate::storage::edges::{Domain, EdgeType, GraphEdge};

#[test]
fn test_record_traversal_success() {
    let mut edge = GraphEdge::new(
        1,
        Uuid::nil(),
        Uuid::nil(),
        EdgeType::Semantic,
        0.5,
        Domain::General,
    );
    edge.steering_reward = 0.5;

    // Record successful traversal with alpha=0.1
    edge.record_traversal(true, 0.1);
    // steering_reward = 0.9 * 0.5 + 0.1 * 1.0 = 0.45 + 0.1 = 0.55
    assert!((edge.steering_reward - 0.55).abs() < 1e-6);
    assert_eq!(edge.traversal_count, 1);
    assert!(edge.last_traversed_at > 0);
}

#[test]
fn test_record_traversal_failure() {
    let mut edge = GraphEdge::new(
        1,
        Uuid::nil(),
        Uuid::nil(),
        EdgeType::Semantic,
        0.5,
        Domain::General,
    );
    edge.steering_reward = 0.5;

    // Record failed traversal
    edge.record_traversal(false, 0.1);
    // steering_reward = 0.9 * 0.5 + 0.1 * 0.0 = 0.45
    assert!((edge.steering_reward - 0.45).abs() < 1e-6);
    assert_eq!(edge.traversal_count, 1);
}

#[test]
fn test_record_traversal_sequence() {
    let mut edge = GraphEdge::new(
        1,
        Uuid::nil(),
        Uuid::nil(),
        EdgeType::Semantic,
        0.5,
        Domain::General,
    );
    edge.steering_reward = 0.5;

    // Success then failure
    edge.record_traversal(true, 0.1);
    assert!((edge.steering_reward - 0.55).abs() < 1e-6);

    edge.record_traversal(false, 0.1);
    // 0.9 * 0.55 + 0.1 * 0.0 = 0.495
    assert!((edge.steering_reward - 0.495).abs() < 1e-6);
    assert_eq!(edge.traversal_count, 2);
}

#[test]
fn test_steering_reward_clamped() {
    let mut edge = GraphEdge::new(
        1,
        Uuid::nil(),
        Uuid::nil(),
        EdgeType::Semantic,
        0.5,
        Domain::General,
    );

    // Force to max via repeated success
    for _ in 0..100 {
        edge.record_traversal(true, 0.5);
    }
    assert!(edge.steering_reward <= 1.0);

    // Force to min via repeated failure
    for _ in 0..100 {
        edge.record_traversal(false, 0.5);
    }
    assert!(edge.steering_reward >= 0.0);
}

#[test]
fn test_freshness_never_traversed() {
    let edge = GraphEdge::default();
    assert_eq!(edge.freshness(), u64::MAX);
}

#[test]
fn test_traversed_since() {
    let mut edge = GraphEdge::default();
    let before = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
        - 10;

    assert!(!edge.traversed_since(before));

    edge.record_traversal_default(true);
    assert!(edge.traversed_since(before));
}
