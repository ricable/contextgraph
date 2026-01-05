//! Tests for GraphEdge modulated weight calculations.

use uuid::Uuid;

use crate::storage::edges::{Domain, EdgeType, GraphEdge};

#[test]
fn test_modulated_weight_canonical_formula() {
    let mut edge = GraphEdge::new(
        1,
        Uuid::nil(),
        Uuid::nil(),
        EdgeType::Semantic,
        0.5,
        Domain::Code,
    );
    edge.steering_reward = 0.5; // steering_factor = 1.0

    // Code domain weights: e=0.6, i=0.3, m=0.4
    // net_activation = 0.6 - 0.3 + (0.4 * 0.5) = 0.3 + 0.2 = 0.5

    // Query same domain: domain_bonus = 0.1
    // w_eff = 0.5 * (1.0 + 0.5 + 0.1) * 1.0 = 0.5 * 1.6 = 0.8
    let w = edge.get_modulated_weight(Domain::Code);
    assert!((w - 0.8).abs() < 0.01, "Expected 0.8, got {}", w);

    // Query different domain: domain_bonus = 0.0
    // w_eff = 0.5 * (1.0 + 0.5 + 0.0) * 1.0 = 0.5 * 1.5 = 0.75
    let w = edge.get_modulated_weight(Domain::Legal);
    assert!((w - 0.75).abs() < 0.01, "Expected 0.75, got {}", w);
}

#[test]
fn test_modulated_weight_clamping() {
    let mut edge = GraphEdge::new(
        1,
        Uuid::nil(),
        Uuid::nil(),
        EdgeType::Semantic,
        1.0,
        Domain::Creative,
    );
    edge.steering_reward = 1.0; // steering_factor = 1.5

    // Creative: e=0.8, i=0.1, m=0.6
    // net_activation = 0.8 - 0.1 + (0.6 * 0.5) = 0.7 + 0.3 = 1.0
    // w_eff = 1.0 * (1.0 + 1.0 + 0.1) * 1.5 = 1.0 * 2.1 * 1.5 = 3.15
    // Clamped to 1.0
    let w = edge.get_modulated_weight(Domain::Creative);
    assert!((w - 1.0).abs() < 1e-6, "Expected 1.0 (clamped), got {}", w);
}

#[test]
fn test_modulated_weight_zero_base() {
    let edge = GraphEdge::new(
        1,
        Uuid::nil(),
        Uuid::nil(),
        EdgeType::Semantic,
        0.0,
        Domain::General,
    );
    let w = edge.get_modulated_weight(Domain::General);
    assert!((w - 0.0).abs() < 1e-6, "Zero base weight should give zero");
}

#[test]
fn test_steering_factor_range() {
    let mut edge = GraphEdge::new(
        1,
        Uuid::nil(),
        Uuid::nil(),
        EdgeType::Semantic,
        0.5,
        Domain::General,
    );

    // Min steering: factor = 0.5
    edge.steering_reward = 0.0;
    let w_min = edge.get_modulated_weight(Domain::General);

    // Max steering: factor = 1.5
    edge.steering_reward = 1.0;
    let w_max = edge.get_modulated_weight(Domain::General);

    // Max should be 3x min (1.5 / 0.5)
    assert!(w_max > w_min, "Max steering should give higher weight");
}

#[test]
fn test_composite_score() {
    let mut edge = GraphEdge::new(
        1,
        Uuid::nil(),
        Uuid::nil(),
        EdgeType::Semantic,
        0.8,
        Domain::Code,
    );
    edge.confidence = 0.5;
    edge.steering_reward = 0.5;

    let modulated = edge.get_modulated_weight(Domain::Code);
    let composite = edge.composite_score(Domain::Code);

    assert!((composite - modulated * 0.5).abs() < 1e-6);
}
