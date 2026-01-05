//! Tests for GraphEdge helper methods.

use uuid::Uuid;

use crate::storage::edges::{Domain, EdgeType, GraphEdge};

#[test]
fn test_mark_as_shortcut() {
    let mut edge = GraphEdge::default();
    assert!(!edge.is_amortized_shortcut);
    edge.mark_as_shortcut();
    assert!(edge.is_amortized_shortcut);
}

#[test]
fn test_update_confidence() {
    let mut edge = GraphEdge::default();
    edge.update_confidence(0.8);
    assert!((edge.confidence - 0.8).abs() < 1e-6);

    // Test clamping
    edge.update_confidence(1.5);
    assert!((edge.confidence - 1.0).abs() < 1e-6);

    edge.update_confidence(-0.5);
    assert!((edge.confidence - 0.0).abs() < 1e-6);
}

#[test]
fn test_edge_type_variants() {
    let types = EdgeType::all();
    assert_eq!(types.len(), 5);
    assert!(types.contains(&EdgeType::Semantic));
    assert!(types.contains(&EdgeType::Temporal));
    assert!(types.contains(&EdgeType::Causal));
    assert!(types.contains(&EdgeType::Hierarchical));
    assert!(types.contains(&EdgeType::Contradicts));
}

#[test]
fn test_equality_by_id() {
    let edge1 = GraphEdge::new(
        1,
        Uuid::new_v4(),
        Uuid::new_v4(),
        EdgeType::Semantic,
        0.5,
        Domain::General,
    );
    let edge2 = GraphEdge::new(
        1,
        Uuid::new_v4(),
        Uuid::new_v4(),
        EdgeType::Causal,
        0.9,
        Domain::Code,
    );
    let edge3 = GraphEdge::new(
        2,
        Uuid::nil(),
        Uuid::nil(),
        EdgeType::Semantic,
        0.5,
        Domain::General,
    );

    // Same ID = equal (regardless of other fields)
    assert_eq!(edge1, edge2);
    // Different ID = not equal
    assert_ne!(edge1, edge3);
}

#[test]
fn test_base_weight() {
    let edge = GraphEdge::new(
        1,
        Uuid::nil(),
        Uuid::nil(),
        EdgeType::Semantic,
        0.75,
        Domain::General,
    );
    assert!((edge.base_weight() - 0.75).abs() < 1e-6);
}
